import os
from typing import Tuple

import yaml
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau

from src.nets.cqtnet import CQTNet, CQTNetMTL
from src.nets.coverhunter import CoverHunter
from src.nets.lyracnet import LyraCNet, LyraCNetMTL
from src.losses import init_loss
from src.lr_schedulers import (
    CosineAnnealingWarmRestartsWithWarmup,
    WarmupPiecewiseConstantScheduler,
    ExponentialWithMinLR
)

def count_model_parameters(model, verbose: bool = True) -> Tuple[int, int]:
    """Counts the number of parameters in a model."""

    grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_grad_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    if verbose:
        print(f"\nTotal number of\n    trainable parameters: {grad_params:,}")
        print(
            f"non-trainable parameters: {non_grad_params:>{len(str(grad_params))+2},}"
        )

    return grad_params, non_grad_params

    
def build_model(config: dict, device: str, mode: str) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """builds model and loss.
    Args:
        config (dict): config with parameters
        device (str): device, eg. cuda, cpu
        mode (str): train or infer
    Raises:
        ValueError: 
    Returns:
        Tuple[torch.nn.Module, torch.nn.Module]: model, loss
    """
        # Init Loss
    loss_config = config["TRAIN"]["LOSS"]
    loss_config_inductive = config["TRAIN"].get("LOSS_INDUCTIVE")
    
    if loss_config_inductive:
        assert config["MODEL"]["ARCHITECTURE"].endswith("-mtl"), "Inductive loss only works with inductive models"
    
    if mode == "train":
        loss_func = init_loss(loss_config, loss_config_inductive)
    
    else:
        loss_func = None
        
    if config["MODEL"]["ARCHITECTURE"].upper() == "CQTNET":
        model = CQTNet(
            ch_in=config["MODEL"]["CONV_CHANNEL"],
            embed_dim=config["MODEL"]["EMBEDDING_SIZE"],
            norm=config["MODEL"]["NORMALIZATION"],
            pool=config["MODEL"]["POOLING"],
            l2_normalize=config["MODEL"]["L2_NORMALIZE"],
            neck=config["MODEL"]["NECK"],
            loss_config=loss_config
        ).to(device)
    elif config["MODEL"]["ARCHITECTURE"].upper() == "CQTNET-MTL":
        model = CQTNetMTL(
            ch_in=config["MODEL"]["CONV_CHANNEL"],
            embed_dim=config["MODEL"]["EMBEDDING_SIZE"],
            norm=config["MODEL"]["NORMALIZATION"],
            pool=config["MODEL"]["POOLING"],
            l2_normalize=config["MODEL"]["L2_NORMALIZE"],
            neck=config["MODEL"]["NECK"],
            loss_config=loss_config,
            loss_config_inductive=loss_config_inductive
        ).to(device)
    elif config["MODEL"]["ARCHITECTURE"].upper() == "COVERHUNTER":
        model = CoverHunter(
            input_dim=config["MODEL"]["FREQUENCY_BINS"],
            embed_dim=config["MODEL"]["EMBEDDING_SIZE"],
            output_dim=config["MODEL"]["OUTPUT_DIM"],
            attention_dim=config["MODEL"]["ATTENTION_DIM"],
            num_blocks=config["MODEL"]["NUM_BLOCKS"],
            output_cls=config["MODEL"]["OUTPUT_CLS"],
            l2_normalize=config["MODEL"]["L2_NORMALIZE"]
        ).to(device)
    elif config["MODEL"]["ARCHITECTURE"].upper() == "LYRACNET":
        model = LyraCNet(
            depth=config["MODEL"]["DEPTH"], 
            embed_dim=config["MODEL"]["EMBEDDING_SIZE"], 
            num_blocks=config["MODEL"]["NUM_BLOCKS"],
            widen_factor=config["MODEL"]["WIDEN_FACTOR"],
            neck="bnneck",
            loss_config=loss_config
            ).to(device)
    elif config["MODEL"]["ARCHITECTURE"].upper() == "LYRACNET-MTL":
        model = LyraCNetMTL(
            depth=config["MODEL"]["DEPTH"],
            embed_dim=config["MODEL"]["EMBEDDING_SIZE"],
            num_blocks=config["MODEL"]["NUM_BLOCKS"],
            widen_factor=config["MODEL"]["WIDEN_FACTOR"],
            neck="bnneck",
            loss_config=loss_config,
            loss_config_inductive=loss_config_inductive
            ).to(device) 
    else:
        raise ValueError("Model architecture not recognized.")
    _, _ = count_model_parameters(model)
    return model, loss_func


def save_model(
    save_dir,
    config,
    model,
    optimizer,
    scheduler,
    scaler,
    train_loss,
    mAP,
    date_time,
    epoch,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    # Add the model path
    config["MODEL"]["CHECKPOINT_PATH"] = os.path.abspath(
        os.path.join(save_dir, "model_checkpoint.pth")
    )
    # Save the config
    with open(os.path.join(save_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    # Save the model and everything else
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else {}
            ),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else {},
            "train_loss": train_loss,
            "mAP": mAP,
            "date_time": date_time,
        },
        config["MODEL"]["CHECKPOINT_PATH"],
    )
    print(f"Model saved in {save_dir}")


def load_model(config: dict, device: str, mode="train"):

    assert mode in ["train", "infer"], "Mode must be either 'train' or 'infer'"
    
    model, loss_func = build_model(config, device, mode)

    if mode == "train":

        if config["TRAIN"]["OPTIMIZER"].upper() == "ADAM":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=config["TRAIN"]["LR"]["LR"]
            )
        elif config["TRAIN"]["OPTIMIZER"].upper() == "ADAMW":
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config["TRAIN"]["LR"]["LR"], 
                betas=(config["TRAIN"]["ADAM_B1"], config["TRAIN"]["ADAM_B2"])
            )
        else:
            raise ValueError("Optimizer not recognized.")
        if config["TRAIN"].get("WEIGHT_DECAY"):
            for param_group in optimizer.param_groups:
                param_group['weight_decay'] = config["TRAIN"]["WEIGHT_DECAY"]
            
        if "PARAMS" in config["TRAIN"]["LR"]:
            lr_params = {
                (k.lower() if not k.startswith("T_") else k): v
                for k, v in config["TRAIN"]["LR"]["PARAMS"].items()
            }
        if config["TRAIN"]["LR"]["SCHEDULE"].upper() == "STEP":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **lr_params)
        elif config["TRAIN"]["LR"]["SCHEDULE"].upper() == "MULTISTEP":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                **lr_params,
            )
        elif config["TRAIN"]["LR"]["SCHEDULE"].upper() == "EXPONENTIAL":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                **lr_params,
            )
        elif config["TRAIN"]["LR"]["SCHEDULE"].upper() == "EXPONENTIAL-MIN_LR":
            scheduler = ExponentialWithMinLR(
                optimizer,
                **lr_params,
            )
        elif config["TRAIN"]["LR"]["SCHEDULE"].upper() == "NONE":
            scheduler = None
        elif config["TRAIN"]["LR"]["SCHEDULE"].upper() == "COSINE-WARMUP":
            scheduler = CosineAnnealingWarmRestartsWithWarmup(
                optimizer,
                **lr_params,
            )
        elif config["TRAIN"]["LR"]["SCHEDULE"].upper() == "COSINE":

            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                **lr_params,
            )
        elif config["TRAIN"]["LR"]["SCHEDULE"].upper() == "LIN-WARMUP-PCWS":
            scheduler = WarmupPiecewiseConstantScheduler(
                optimizer,
                eta_min=config["TRAIN"]["LR"]["LR"],
                **lr_params,
            )
        elif config["TRAIN"]["LR"]["SCHEDULE"].upper() == "REDUCE-ON-PLATEAU":
            scheduler = ReduceLROnPlateau(
                optimizer,
                **lr_params,
            )
        else:
            raise ValueError("Learning rate scheduler not recognized.")

        if config["TRAIN"]["AUTOMATIC_MIXED_PRECISION"]:
            print("\033[32mUsing Automatic Mixed Precision...\033[0m")
            scaler = torch.amp.GradScaler("cuda")
        else:
            print("Using full precision...")
            scaler = None

        start_epoch = 1
        train_loss = 0.0
        mAP = 0.0

        if "CHECKPOINT_PATH" in config["MODEL"]:
            checkpoint_path = config["MODEL"]["CHECKPOINT_PATH"]
            if os.path.isfile(checkpoint_path):
                print(
                    f"\033[32mLoading the model checkpoint from {checkpoint_path}\033[0m"
                )
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                print(f"Model loaded from epoch {checkpoint['epoch']}")
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                # TODO Give the user the ability to discard the scheduler or create it from scratch
                if scheduler is not None:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                if scaler is not None:
                    scaler.load_state_dict(checkpoint["scaler_state_dict"])
                start_epoch = checkpoint["epoch"] + 1  # start_epoch = last_epoch +1
                train_loss = checkpoint["train_loss"]
                mAP = checkpoint["mAP"]
            else:
                print(f"\033[31mNo checkpoint found at {checkpoint_path}\033[0m")
                print("Training from scratch.")
        else:
            print("\033[31mNo checkpoint path provided\033[0m")
            print("Training from scratch.")
        return model, loss_func, optimizer, scheduler, scaler, start_epoch, train_loss, mAP

    else:
        if "CHECKPOINT_PATH" in config["MODEL"]:
            checkpoint_path = config["MODEL"]["CHECKPOINT_PATH"]
            if os.path.isfile(checkpoint_path):
                print(
                    f"\033[32mLoading the model checkpoint from {checkpoint_path}\033[0m"
                )
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                print(f"Model loaded from epoch {checkpoint['epoch']}")
            else:
                raise ValueError(f"No checkpoint found at {checkpoint_path}")
        else:
            raise ValueError("No checkpoint path provided in the config file.")
        return model
