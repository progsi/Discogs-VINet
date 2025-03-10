import os
import csv
import time
import yaml
import argparse
from typing import Type
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from src.dataset import TestDataset
from src.utils import load_model
from src.utilities.utils import format_time
from src.utilities.metrics import calculate_metrics


@torch.no_grad()
def evaluate(
    model: Type[torch.nn.Module],
    loader: DataLoader,
    similarity_search: str,
    chunk_size: int,
    noise_works: bool,
    amp: bool,
    device: torch.device,
    genres_multihot: torch.Tensor = None,
    genre_idx_to_label: dict = None,
) -> dict:
    """Evaluate the model by simulating the retrieval task. Compute the embeddings
    of all versions and calculate the pairwise distances. Calculate the mean average
    precision of the retrieval task. Metric calculations are done on the cpu but you
    can choose the device for the model. Since we normalize the embeddings, MCSS is
    equivalent to NNS. Please refer to the argparse arguments for more information.

    Parameters:
    -----------
    model : torch.nn.Module
        Model to evaluate
    loader : torch.utils.data.DataLoader
        DataLoader containing the test set cliques
    similarity_search: str
        Similarity search function. MIPS, NNS, or MCSS.
    chunk_size : int
        Chunk size to use during metrics calculation.
    noise_works : bool
        Flag to indicate if the dataset contains noise works.
    amp : bool
        Flag to indicate if Automatic Mixed Precision should be used.
    device : torch.device
        Device to use for inference and metric calculation.
    genres_multihot : torch.Tensor
        Tensor with multi-hot genres.
    genre_idx_to_label :
        Dictionary mapping genre indices to genre labels.

    Returns:
    --------
    metrics : dict
        Dictionary containing the evaluation metrics. See utilities.metrics.calculate_metrics
    """

    t0 = time.monotonic()

    model.eval()
    
    # Preallocate tensors to avoid https://github.com/pytorch/pytorch/issues/13246
    N = len(loader)
    d = model.embed_dim
    
    embeddings = torch.zeros((N, d))
    labels = torch.zeros(N)
    
    print("Extracting embeddings...")
    for i, (feature, label) in tqdm(enumerate(loader), total=N):

        feature = feature.unsqueeze(1).to(device)  # (1,F,T) -> (1,1,F,T)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            embedding, _ = model(feature)
            
        embeddings[i] = embedding
        labels[i] = label
        
    print(f"Extraction time: {format_time(time.monotonic() - t0)}")

    embeddings = embeddings.to(device)
    labels = labels.to(device)
    
    # If there are no noise works, remove the cliques with single versions
    # this may happen due to the feature extraction process.
    if not noise_works:
        # Count each label's occurrence
        unique_labels, counts = torch.unique(labels, return_counts=True)
        # Filter labels that occur more than once
        valid_labels = unique_labels[counts > 1]
        # Create a mask for indices where labels appear more than once
        keep_mask = torch.isin(labels, valid_labels)
        if keep_mask.sum() < len(labels):
            print("Removing single version cliques...")
            embeddings = embeddings[keep_mask]
            labels = labels[keep_mask]

    print("Calculating metrics...")
    t0 = time.monotonic()
    metrics = calculate_metrics(
        embeddings,
        labels,
        similarity_search=similarity_search,
        chunk_size=chunk_size,
        device=device,
        genres_multihot=genres_multihot,
        genre_idx_to_label=genre_idx_to_label
    )
    print(f"Calculation time: {format_time(time.monotonic() - t0)}")

    return metrics


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="""Path to the configuration file of the trained model. 
        The config will be used to find model weigths.""",
    )
    parser.add_argument(
        "test_cliques",
        type=str,
        help="""Path to the test cliques.json file. 
        Can be SHS100K, Da-TACOS or DiscogsVI.""",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--similarity-search",
        "-s",
        type=str,
        default=None,
        choices=["MIPS", "MCSS", "NNS"],
        help="""Similarity search function to use for the evaluation. 
        MIPS: Maximum Inner Product Search, 
        MCSS: Maximum Cosine Similarity Search, 
        NNS: Nearest Neighbour Search.""",
    )
    parser.add_argument(
        "--features-dir",
        "-f",
        type=str,
        default=None,
        help="""Path to the features directory. 
        Optional, by default uses the path in the config file.""",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Test batch size.",
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
        "--cross-genre",
        action="store_true",
        help="""Flag to enable cross-genre evaluation.""",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="""Flag to disable the GPU. If not provided, 
        the GPU will be used if available.""",
    )
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="""Flag to disable Automatic Mixed Precision for inference. 
        If not provided, AMP usage will depend on the model config file.""",
    )
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = yaml.safe_load(f)
    print("\033[36m\nExperiment Configuration:\033[0m")
    print(
        "\033[36m" + yaml.dump(config, indent=4, width=120, sort_keys=False) + "\033[0m"
    )

    if args.features_dir is None:
        print("\033[31mFeatures directory NOT provided.\033[0m")
        args.features_dir = config["TRAIN"]["FEATURES_DIR"]
    print(f"\033[31mFeatures directory: {args.features_dir}\033[0m\n")

    # To evaluate the model in an Information Retrieval setting
    eval_dataset = TestDataset(
        args.test_cliques,
        args.features_dir,
        mean_downsample_factor=config["MODEL"]["DOWNSAMPLE_FACTOR"],
        cross_genre=args.cross_genre,
        scale=config["TRAIN"]["SCALE"],
        min_length=config["TRAIN"]["MIN_LENGTH"])
    
    eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
            collate_fn=eval_dataset.collate_fn if args.batch_size > 1 else None)
    
    if args.no_gpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\033[31mDevice: {device}\033[0m\n")

    if args.disable_amp:
        config["TRAIN"]["AUTOMATIC_MIXED_PRECISION"] = False

    model = load_model(config, device, mode="infer")

    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.output_dir = os.path.join(
            script_dir, "logs", "evaluation", config["MODEL"]["NAME"]
        )
        if "best_epoch" in config["MODEL"]["CHECKPOINT_PATH"]:
            args.output_dir = os.path.join(args.output_dir, "best_epoch")
        elif "last_epoch" in config["MODEL"]["CHECKPOINT_PATH"]:
            args.output_dir = os.path.join(args.output_dir, "last_epoch")

    if eval_dataset.discogs_vi:
        args.output_dir = os.path.join(args.output_dir, "DiscogsVI")
    elif eval_dataset.datacos:
        args.output_dir = os.path.join(args.output_dir, "Da-TACOS")
    elif eval_dataset.shs100k:
        args.output_dir = os.path.join(args.output_dir, "SHS100K")
    else:
        raise ValueError("Dataset not recognized.")
    print(f"\033[31mOutput directory: {args.output_dir}\033[0m\n")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.similarity_search is None:
        args.similarity_search = config["MODEL"]["SIMILARITY_SEARCH"]
        
    if args.cross_genre:
        genres_multihot = eval_dataset.get_all_genres_multihot()
        genre_idx_to_label = eval_dataset.idx_to_genre
    else:
        genres_multihot = None
        genre_idx_to_label = None

    print("Evaluating...")
    t0 = time.monotonic()
    metrics = evaluate(
        model,
        eval_loader,
        similarity_search=args.similarity_search,
        chunk_size=args.chunk_size,
        noise_works=eval_dataset.datacos,
        amp=config["TRAIN"]["AUTOMATIC_MIXED_PRECISION"],
        device=device,
        genres_multihot=genres_multihot,
        genre_idx_to_label=genre_idx_to_label
    )
    print(f"Total time: {format_time(time.monotonic() - t0)}")

    eval_path = os.path.join(args.output_dir, "evaluation_metrics.csv")
    print(f"Saving the evaluation results in: {eval_path}")
    with open(eval_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for scheme, scheme_metrics in metrics.items():
            for submetric, value in scheme_metrics.items():
                writer.writerow([f"{scheme}_{submetric}", value])

    #############
    print("Done!")
