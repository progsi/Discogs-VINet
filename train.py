"""Training script for the model."""

import os
from datetime import datetime
import time
import yaml
import argparse
from typing import Tuple, Union, Type
from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import DataLoader
from pytorch_metric_learning import miners, samplers

from evaluate import evaluate
from src.dataset import TrainDataset, TestDataset
from src.utils import load_model, save_model
from src.utilities.utils import format_time
from src.losses import init_loss, WeightedMultiloss, FocalLoss, requires_cls_labels

SEED = 27  # License plate code of Gaziantep, gastronomical capital of Türkiye


def train_epoch(
    model: Type[torch.nn.Module],
    loader: DataLoader,
    loss_func: Type[torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    scheduler: Union[torch.optim.lr_scheduler.LRScheduler, None],
    scaler: Union[torch.cuda.amp.GradScaler, None],  # type: ignore
    device: torch.device,
    all_labels: torch.Tensor = None,
) -> Tuple[float, float, Union[float, None]]:
    """Train the model for one epoch. Return the average loss of the epoch."""

    amp = scaler is not None
    cls = all_labels is not None # whether we have classification loss needed
    
    model.train()
    losses, triplet_stats = [], []
    for i, (features, labels) in enumerate(tqdm(loader)):
        features = features.unsqueeze(1).to(device)  # (B,F,T) -> (B,1,F,T)
        labels = labels.to(device)  # (B,)
        optimizer.zero_grad()  # TODO set_to_none=True?
        
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            
            embeddings, y = model(features)
            
            if cls:
                loss = loss_func(embeddings, labels, y, all_labels)
                
            else:
                loss = loss_func(embeddings, labels)
                
        if amp:
            scaler.scale(loss).backward()  # type: ignore
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        losses.append(loss.detach().item())
        if (i + 1) % (len(loader) // 25) == 0 or i == len(loader) - 1:
            print(
                f"[{(i+1):>{len(str(len(loader)))}}/{len(loader)}], Batch Loss: {loss.item():.4f}"
            )
            # if MultiLoss, also record individual losses
            if isinstance(loss_func, WeightedMultiloss):
                wandb.log(loss_func.get_stats())

    if scheduler is not None:
        scheduler.step()
        lr = scheduler.optimizer.param_groups[0]["lr"]
    else:
        lr = optimizer.param_groups[0]["lr"]

    # Return the average loss of the epoch
    epoch_loss = np.array(losses).mean().item()

    return epoch_loss, lr


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("config_path", type=str, help="Path to the configuration file.")
    parser.add_argument(
        "--save-frequency", type=int, default=1, help="Save the model every N epochs."
    )
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=1,
        help="Evaluate the model every N epochs.",
    )
    parser.add_argument(
        "--chunk-size",
        "-b",
        type=int,
        default=1024,
        help="Chunk size to use during metrics calculation.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=10,
        help="Number of workers to use in the DataLoader.",
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="Do not use wandb to log experiments."
    )
    parser.add_argument(
        "--wandb-id", type=str, default=None, help="Wandb id to resume an experiment."
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="VI-after_ismir",
        help="Wandb project name.",
    )
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    print("\n\033[32mArguments:\033[0m")
    for arg in vars(args):
        print(f"\033[32m{arg}: {getattr(args, arg)}\033[0m")

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    print("\n\033[36mExperiment Configuration:\033[0m")
    print(
        "\033[36m" + yaml.dump(config, indent=4, width=120, sort_keys=False) + "\033[0m"
    )

    if not args.no_wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            config=config,
            id=wandb.util.generate_id() if args.wandb_id is None else args.wandb_id,  # type: ignore
            resume="allow",
        )
    else:
        print("\033[31mNot logging the training process.\033[0m")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n\033[34mDevice: {device}\033[0m")

    torch.backends.cudnn.deterministic = config["TRAIN"]["CUDA_DETERMINISTIC"]  # type: ignore
    torch.backends.cudnn.benchmark = config["TRAIN"]["CUDA_BENCHMARK"]  # type: ignore

    # Load or create the model
    model, optimizer, scheduler, scaler, start_epoch, train_loss, best_mAP = load_model(
        config, device
    )

    save_dir = os.path.join(config["MODEL"]["CHECKPOINT_DIR"], config["MODEL"]["NAME"])
    last_save_dir = os.path.join(save_dir, "last_epoch")
    best_save_dir = os.path.join(save_dir, "best_epoch")
    print("Checkpoints will be saved to: ", save_dir)

    date_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    print("Creating the dataset...")
    train_dataset = TrainDataset(
        config["TRAIN"]["TRAIN_CLIQUES"],
        config["TRAIN"]["FEATURES_DIR"],
        context_length=config["TRAIN"]["CONTEXT_LENGTH"],
        mean_downsample_factor=config["MODEL"]["DOWNSAMPLE_FACTOR"],
        clique_usage_ratio=config["TRAIN"]["CLIQUE_USAGE_RATIO"],
    )
    
    sampler = samplers.MPerClassSampler(train_dataset.labels,
                                      batch_size=config["TRAIN"]["BATCH_SIZE"], 
                                      m=config["TRAIN"]["M_PER_CLASS"],
                                      length_before_new_iter=len(train_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["TRAIN"]["BATCH_SIZE"],
        collate_fn=train_dataset.collate_fn,
        drop_last=True,
        num_workers=args.num_workers,
        sampler=sampler
    )

    # To evaluate the model in an Information Retrieval setting
    eval_dataset = TestDataset(
        config["TRAIN"]["VALIDATION_CLIQUES"],
        config["TRAIN"]["FEATURES_DIR"],
        mean_downsample_factor=config["MODEL"]["DOWNSAMPLE_FACTOR"],
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1, # TODO: large batch size + padding?
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    loss_func = init_loss(config["TRAIN"]["LOSS"])
    if requires_cls_labels(loss_func):
        all_labels = torch.tensor(train_dataset.clique_nums).to(device)
    else:
        all_labels = None
    # init the triplet loss and mining
    
    # Log the initial lr
    if not args.no_wandb:
        if scheduler is not None:
            lr_current = scheduler.optimizer.param_groups[0]["lr"]
        else:
            lr_current = optimizer.param_groups[0]["lr"]
        wandb.log(
            {
                "epoch": start_epoch - 1,
                "lr": lr_current,
            }
        )

    print("Training the model...")
    for epoch in range(start_epoch, config["TRAIN"]["EPOCHS"] + 1):

        t0 = time.monotonic()
        print(f" Epoch: [{epoch}/{config['TRAIN']['EPOCHS']}] ".center(25, "="))
        train_loss, lr_current = train_epoch(
            model,
            train_loader,
            loss_func,
            optimizer,
            scheduler,
            scaler=scaler,
            device=device,
            all_labels=all_labels,
        )
        t_train = time.monotonic() - t0
        print(f"Average epoch Loss: {train_loss:.6f}, in {format_time(t_train)}")
        if not args.no_wandb:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "train_time": t_train,
                    "epoch": epoch,
                    "lr": lr_current,
                    "difficult_triplets": triplet_stats,
                }
            )

        if epoch % args.save_frequency == 0 or epoch == config["TRAIN"]["EPOCHS"]:
            save_model(
                last_save_dir,
                config=config,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                train_loss=train_loss,
                mAP=best_mAP,
                date_time=date_time,
                epoch=epoch,
            )

        if epoch % args.eval_frequency == 0 or epoch == config["TRAIN"]["EPOCHS"]:
            print("Evaluating the model...")
            t0 = time.monotonic()
            metrics = evaluate(
                model,
                eval_loader,
                similarity_search=config["MODEL"]["SIMILARITY_SEARCH"],
                chunk_size=args.chunk_size,
                noise_works=False,
                amp=config["TRAIN"]["AUTOMATIC_MIXED_PRECISION"],
                device=device,
            )
            t_eval = time.monotonic() - t0
            print(
                f"MAP: {metrics['MAP']:.3f}, MR1: {metrics['MR1']:.2f} - {format_time(t_eval)}"
            )

            if metrics["MAP"] >= best_mAP:
                best_mAP = metrics["MAP"]
                save_model(
                    best_save_dir,
                    config=config,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    train_loss=train_loss,
                    mAP=best_mAP,
                    date_time=date_time,
                    epoch=epoch,
                )
            if not args.no_wandb:
                wandb.log(
                    {
                        **metrics,
                        "eval_time": t_eval,
                        "epoch": epoch,
                        "best_MAP": best_mAP,
                    }
                )

    print("===Training finished===")
    if not args.no_wandb:
        wandb.finish()

    #############
    print("Done!")
