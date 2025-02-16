from typing import Dict

import torch
from torchmetrics.retrieval import RetrievalMAP, RetrievalMRR, RetrievalHitRate

from src.utilities.tensor_op import (
    pairwise_cosine_similarity,
    pairwise_distance_matrix,
    pairwise_dot_product,
    create_class_matrix,
)


def compute_chunkwise(S: torch.Tensor, 
                      C: torch.Tensor, 
                      k: int, 
                      chunk_size: int, 
                      device: str = "cpu") -> Dict[str, float]:
    """Compute the evaluation metrics in chunks to avoid memory issues.
    Args:
        S (torch.Tensor): similarity matrix
        C (torch.Tensor): class membership matrix
        k (int): first ranks to consider for P@k
        chunk_size (int): chunks size
    device (str, optional): GPU or CPU, like for torch. Defaults to "cpu".
    Returns:
        Dict[str, float]: dict with metrics
    """
    ranking = torch.arange(1, k + 1, dtype=torch.float32, device=device).unsqueeze(
    0
    )  # (1, N-1)
    
    # Initialize the tensors for storing the evaluation metrics
    TOP1, TOP10 = torch.tensor([]), torch.tensor([])
    TR, MAP = torch.tensor([]), torch.tensor([])

    # Iterate over the chunks
    for i, (s, c) in enumerate(
        zip(S.split(chunk_size), C.split(chunk_size))
    ):

        # Move the tensors to the device
        s = s.to(device)
        c = c.to(device)  # Full class matrix would require 53GB of memory

        # Number of relevant items for each query in the chunk
        n_relevant = torch.sum(c, 1)  # (B,)

        # Check if there are relevant items for each query
        assert torch.all(
            n_relevant > 0
        ), "There must be at least one relevant item for each query"

        # For each embedding, find the indices of the k most similar embeddings
        _, spred = torch.topk(s, k, dim=1)  # (B', N-1)
        # Get the relevance values of the k most similar embeddings
        relevance = torch.gather(c, 1, spred)  # (B', N-1)

        # Number of relevant items in the top 1 and 10
        top1 = relevance[:, 0].int().cpu()
        top10 = relevance[:, :10].int().sum(1).cpu()

        # Get the rank of the first correct prediction by tie breaking
        temp = (
            torch.arange(k, dtype=torch.float32, device=device).unsqueeze(0) * 1e-6
        )  # (1, N-1)
        _, sel = torch.topk(relevance - temp, 1, dim=1)  # (B', 1)
        
        # NOTE: implemented code to address the issue mentioned in https://github.com/furkanyesiler/re-move/issues/4
        # Identify queries with positive results
        has_positives = n_relevant > 0
        # Knock out queries with no positives
        sel = sel.float()
        sel[~has_positives] = torch.nan

        top_rank = sel.squeeze(1).float().cpu() + 1  # (B',)

        # Calculate the average precision for each embedding
        prec_at_k = torch.cumsum(relevance, 1).div_(ranking)  # (B', N-1)
        ap = torch.sum(prec_at_k * relevance, 1).div_(n_relevant).cpu()  # (B',)

        # Concatenate the results from all chunks
        TR = torch.cat((TR, top_rank))
        MAP = torch.cat((MAP, ap))
        TOP1 = torch.cat((TOP1, top1))
        TOP10 = torch.cat((TOP10, top10))

    # computing the final evaluation metrics
    TOP1 = TOP1.int().sum().item()
    TOP10 = TOP10.int().sum().item()
    MR1 = torch.nanmean((sel+1).float()).item() # before: TR.mean().item()
    MRR = torch.nanmean((1/TR).float()).item() # before: (1 / TR).mean().item()
    MAP = MAP.mean().item()
    return {
        "MAP": round(MAP, 3),
        "MRR": round(MRR, 3),
        "MR1": round(MR1, 2),
        "Top1": TOP1,
        "Top10": TOP10 if k > 10 else None,
    }

def compute_all(S: torch.Tensor, 
                C: torch.Tensor, 
                device: str = "cpu") -> Dict[str, float]:
    S, C = S.to(device), C.to(device)
    mAP = RetrievalMAP(empty_target_action="skip").compute(S, C)
    mRR = RetrievalMRR(empty_target_action="skip").compute(S, C)
    top1 = RetrievalHitRate(k=10, empty_target_action="skip").compute(S, C)
    return {
        "MAP": mAP.item(),
        "MRR": mRR.item(),
        "HR@10": top1.item(),
    }
    
        
def calculate_metrics(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    similarity_search: str = "MIPS",
    noise_works: bool = False,
    chunk_size: int = 256,
    device: str = "cpu",
    genres: torch.Tensor = None,
) -> Dict[str, float]:
    """Perform similarity search for a set of embeddings and calculate the following
    metrics using the ground truth labels.
        mAP => Mean Average Precision
        MRR => Mean Reciprocal Rank of the first correct prediction
        MR => Mean Rank of the first correct prediction
        Top1 => Number of correct predictions in the first closest point
        Top10 => Number of correct predictions in the first 10 closest points

    Adapted from: https://github.com/furkanyesiler/re-move/blob/master/utils/metrics.py

    Parameters:
    -----------
    embeddings: torch.Tensor
        2D tensor of shape (m, n) where m is the number of samples, n is the dimension
        of the embeddings.
    labels: torch.Tensor
        1D tensor of shape (m,) where m is the number of samples and labels[i] is the
        integer label of the i-th sample.
    similarity_search: str = "MIPS"
        The similarity search function to use. "NNS", "MCSS", or "MIPS".
    noise_works: bool = False
        If True, the dataset contains noise works, which are not included in metrics;
        otherwise, they are included.
    chunk_size: int = 256
        The size of the chunks to use during the evaluation.
    device: str = "cpu"
        The device to use for the calculations.
    Returns:
    --------
    metrics: dict
        Dictionary containing the performance metrics.

    """
    # TODO: implement cross-genre eval.
    in_genre = torch.eq(genres.unsqueeze(1), genres.unsqueeze(0)).all(dim=-1)
    cross_genre =  torch.matmul(genres.float(), genres.t().float()) > 0
    
    assert labels.dim() == 1, "Labels must be a 1D tensor"
    assert (
        embeddings.dim() == 2
    ), f"Embeddings must be a 2D tensor got {embeddings.shape}"
    assert embeddings.size(0) == labels.size(
        0
    ), "Embeddings and labels must have the same size"
    if similarity_search not in ["NNS", "MCSS", "MIPS"]:
        raise ValueError(
            "Similarity must be either euclidean, inner product, or cosine."
        )
    assert chunk_size > 0, "Chunk size must be positive"
    assert chunk_size <= len(
        labels
    ), "Chunk size must be smaller than the number of queries"

    # For unity
    similarity_search = similarity_search.upper()

    # Number of total items in the dataset
    N = len(labels)

    # Number of similar embeddings to consider for each query
    k = N - 1
    # Create the ranking tensor for AP calculation

    # Each row indicates if the columns are from the same clique,
    # diagonal is set to 0
    C = create_class_matrix(labels, zero_diagonal=True).float()

    # Compute the pairwise similarity matrix
    if similarity_search == "MIPS":
        S = pairwise_dot_product(embeddings)  # (B, N)
    elif similarity_search == "MCSS":
        S = pairwise_cosine_similarity(embeddings)  # (B, N)
    else:
        # Use low precision for faster calculations
        S = -1 * pairwise_distance_matrix(embeddings, precision="low")  # (B, N)

    # Set the similarity of each query with itself to -inf
    S = S.fill_diagonal_(float("-inf"))

    # If Da-TACOS, remove queries with no relevant items (noise works)
    if noise_works:
        non_noise_indices = n_relevant.bool()
        S = S[non_noise_indices]  # (B', N)
        C = C[non_noise_indices]  # (B', N)
        n_relevant = n_relevant[non_noise_indices]  # (B',)

    # storing the evaluation metrics
    if chunk_size:
        metrics = compute_chunkwise(S, C, k, chunk_size, device) 
    
    # printing the evaluation metrics
    for k, v in metrics.items():
        if k in ["Top1", "Top10"]:
            print(f"{k:>5}: {v}")
        else:
            print(f"{k:>5}: {v:.3f}")

    return metrics

# TODO: remove after testing
def calculate_metrics_old(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    similarity_search: str = "MIPS",
    noise_works: bool = False,
    chunk_size: int = 256,
    device: str = "cpu",
    genres: torch.Tensor = None,
) -> dict:
    """Perform similarity search for a set of embeddings and calculate the following
    metrics using the ground truth labels.
        mAP => Mean Average Precision
        MRR => Mean Reciprocal Rank of the first correct prediction
        MR => Mean Rank of the first correct prediction
        Top1 => Number of correct predictions in the first closest point
        Top10 => Number of correct predictions in the first 10 closest points

    Adapted from: https://github.com/furkanyesiler/re-move/blob/master/utils/metrics.py

    Parameters:
    -----------
    embeddings: torch.Tensor
        2D tensor of shape (m, n) where m is the number of samples, n is the dimension
        of the embeddings.
    labels: torch.Tensor
        1D tensor of shape (m,) where m is the number of samples and labels[i] is the
        integer label of the i-th sample.
    similarity_search: str = "MIPS"
        The similarity search function to use. "NNS", "MCSS", or "MIPS".
    noise_works: bool = False
        If True, the dataset contains noise works, which are not included in metrics;
        otherwise, they are included.
    chunk_size: int = 256
        The size of the chunks to use during the evaluation.
    device: str = "cpu"
        The device to use for the calculations.
    Returns:
    --------
    metrics: dict
        Dictionary containing the performance metrics.

    """
    # TODO: implement cross-genre eval.
    in_genre = torch.eq(genres.unsqueeze(1), genres.unsqueeze(0)).all(dim=-1)
    cross_genre =  torch.matmul(genres.float(), genres.t().float()) > 0
    
    assert labels.dim() == 1, "Labels must be a 1D tensor"
    assert (
        embeddings.dim() == 2
    ), f"Embeddings must be a 2D tensor got {embeddings.shape}"
    assert embeddings.size(0) == labels.size(
        0
    ), "Embeddings and labels must have the same size"
    if similarity_search not in ["NNS", "MCSS", "MIPS"]:
        raise ValueError(
            "Similarity must be either euclidean, inner product, or cosine."
        )
    assert chunk_size > 0, "Chunk size must be positive"
    assert chunk_size <= len(
        labels
    ), "Chunk size must be smaller than the number of queries"

    # For unity
    similarity_search = similarity_search.upper()

    # Number of total items in the dataset
    N = len(labels)

    # Number of similar embeddings to consider for each query
    k = N - 1
    # Create the ranking tensor for AP calculation
    ranking = torch.arange(1, k + 1, dtype=torch.float32, device=device).unsqueeze(
        0
    )  # (1, N-1)

    # Each row indicates if the columns are from the same clique,
    # diagonal is set to 0
    class_matrix = create_class_matrix(labels, zero_diagonal=True).float()

    # Initialize the tensors for storing the evaluation metrics
    TOP1, TOP10 = torch.tensor([]), torch.tensor([])
    TR, MAP = torch.tensor([]), torch.tensor([])

    # Iterate over the chunks
    for i, (Q, C) in enumerate(
        zip(embeddings.split(chunk_size), class_matrix.split(chunk_size))
    ):

        # Move the tensors to the device
        Q = Q.to(device)
        C = C.to(device)  # Full class matrix would require 53GB of memory

        # Number of relevant items for each query in the chunk
        n_relevant = torch.sum(C, 1)  # (B,)

        # Compute the pairwise similarity matrix
        if similarity_search == "MIPS":
            S = pairwise_dot_product(Q, embeddings)  # (B, N)
        elif similarity_search == "MCSS":
            S = pairwise_cosine_similarity(Q, embeddings)  # (B, N)
        else:
            # Use low precision for faster calculations
            S = -1 * pairwise_distance_matrix(Q, embeddings, precision="low")  # (B, N)

        # Set the similarity of each query with itself to -inf
        torch.diagonal(S, offset=i * chunk_size).fill_(float("-inf"))

        # If Da-TACOS, remove queries with no relevant items (noise works)
        if noise_works:
            non_noise_indices = n_relevant.bool()
            S = S[non_noise_indices]  # (B', N)
            C = C[non_noise_indices]  # (B', N)
            n_relevant = n_relevant[non_noise_indices]  # (B',)

        # Check if there are relevant items for each query
        assert torch.all(
            n_relevant > 0
        ), "There must be at least one relevant item for each query"

        # For each embedding, find the indices of the k most similar embeddings
        _, spred = torch.topk(S, k, dim=1)  # (B', N-1)
        # Get the relevance values of the k most similar embeddings
        relevance = torch.gather(C, 1, spred)  # (B', N-1)

        # Number of relevant items in the top 1 and 10
        top1 = relevance[:, 0].int().cpu()
        top10 = relevance[:, :10].int().sum(1).cpu()

        # Get the rank of the first correct prediction by tie breaking
        temp = (
            torch.arange(k, dtype=torch.float32, device=device).unsqueeze(0) * 1e-6
        )  # (1, N-1)
        _, sel = torch.topk(relevance - temp, 1, dim=1)  # (B', 1)
        
        # NOTE: implemented code to address the issue mentioned in https://github.com/furkanyesiler/re-move/issues/4
        # Identify queries with positive results
        has_positives = n_relevant > 0
        # Knock out queries with no positives
        sel = sel.float()
        sel[~has_positives] = torch.nan

        top_rank = sel.squeeze(1).float().cpu() + 1  # (B',)

        # Calculate the average precision for each embedding
        prec_at_k = torch.cumsum(relevance, 1).div_(ranking)  # (B', N-1)
        ap = torch.sum(prec_at_k * relevance, 1).div_(n_relevant).cpu()  # (B',)

        # Concatenate the results from all chunks
        TR = torch.cat((TR, top_rank))
        MAP = torch.cat((MAP, ap))
        TOP1 = torch.cat((TOP1, top1))
        TOP10 = torch.cat((TOP10, top10))

    # computing the final evaluation metrics
    TOP1 = TOP1.int().sum().item()
    TOP10 = TOP10.int().sum().item()
    MR1 = torch.nanmean((sel+1).float()).item() # before: TR.mean().item()
    MRR = torch.nanmean((1/TR).float()).item() # before: (1 / TR).mean().item()
    MAP = MAP.mean().item()

    # storing the evaluation metrics
    metrics = {
        "MAP": round(MAP, 3),
        "MRR": round(MRR, 3),
        "MR1": round(MR1, 2),
        "Top1": TOP1,
        "Top10": TOP10 if k > 10 else None,
    }

    # printing the evaluation metrics
    for k, v in metrics.items():
        if k in ["Top1", "Top10"]:
            print(f"{k:>5}: {v}")
        else:
            print(f"{k:>5}: {v:.3f}")

    return metrics
