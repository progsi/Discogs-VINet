from typing import Dict, Tuple

import torch
from torchmetrics.retrieval import RetrievalMAP, RetrievalMRR, RetrievalHitRate

from src.utilities.tensor_op import (
    pairwise_cosine_similarity,
    pairwise_distance_matrix,
    pairwise_dot_product,
    create_class_matrix,
)

MAP = RetrievalMAP(empty_target_action="skip", top_k=None)
MRR = RetrievalMRR(empty_target_action="skip", top_k=None)
HR10 = RetrievalHitRate(empty_target_action="skip", top_k=10)

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
    TR, MAP = torch.tensor([]), torch.tensor([])

    has_positives = torch.sum(C, 1) > 0
    S, C = S[has_positives], C[has_positives]
    # Iterate over the chunks
    for i, (s, c) in enumerate(zip(S.split(chunk_size), C.split(chunk_size))):

        # Move the tensors to the device
        s = s.to(device)
        c = c.to(device)  # Full class matrix would require 53GB of memory

        # Number of relevant items for each query in the chunk
        n_relevant = torch.sum(c, 1)  # (B,)
        
        # # Check if there are relevant items for each query
        # # TODO: recheck
        assert not torch.all(torch.sum(c, dim=1) == 0), "There must be at least one relevant item for each query"

        # For each embedding, find the indices of the k most similar embeddings
        _, spred = torch.topk(s, k, dim=1)  # (B', N-1)
        # Get the relevance values of the k most similar embeddings
        relevance = torch.gather(c, 1, spred)  # (B', N-1)

        # Get the rank of the first correct prediction by tie breaking
        temp = (
            torch.arange(k, dtype=torch.float32, device=device).unsqueeze(0) * 1e-6
        )  # (1, N-1)
        _, sel = torch.topk(relevance - temp, 1, dim=1)  # (B', 1)

        top_rank = sel.squeeze(1).float().cpu() + 1  # (B',)

        # Calculate the average precision for each embedding
        prec_at_k = torch.cumsum(relevance, 1).div_(ranking)  # (B', N-1)
        ap = torch.sum(prec_at_k * relevance, 1).div_(n_relevant).cpu()  # (B',)

        # Concatenate the results from all chunks
        TR = torch.cat((TR, top_rank))
        MAP = torch.cat((MAP, ap))

    # computing the final evaluation metrics
    MR1 = torch.nanmean(TR).float().item() # before: TR.mean().item()
    MRR = torch.nanmean((1/TR).float()).item() # before: (1 / TR).mean().item()
    MAP = MAP.mean().item()
    return {
        "MAP": round(MAP, 3),
        "MRR": round(MRR, 3),
        "MR1": round(MR1, 2),
        "nQueries": round(C.shape[0]),
        "nRelevant": round(torch.sum(C).item()),
        "avgRelPerQ": round(torch.mean(torch.sum(C, 1)).item(), 2)
    }
    
    
def MR1(preds: torch.Tensor, target: torch.Tensor, device: str) -> torch.Tensor:
    """
    Compute the mean rank for relevant items in the predictions.
    Args:
        preds (torch.Tensor): A tensor of predicted scores (higher scores indicate more relevant items).
        target (torch.Tensor): A tensor of true relationships (0 for irrelevant, 1 for relevant).
    Returns:
        torch.Tensor: The mean rank of relevant items for each query.
    """
    has_positives = torch.sum(target, 1) > 0
    
    _, spred = torch.topk(preds, preds.size(1), dim=1)
    found = torch.gather(target, 1, spred)
    temp = torch.arange(preds.size(1)).to(device).float() * 1e-6
    _, sel = torch.topk(found - temp, 1, dim=1)
    
    sel = sel.float()
    sel[~has_positives] = torch.nan
    
    return torch.nanmean((sel+1).float())

def compute_all(S: torch.Tensor, 
                C: torch.Tensor, 
                device: str = "cpu") -> Dict[str, float]:
    
    S, C = S.to(device), C.to(device)
    indexes = torch.arange(S.size(0), device=device).unsqueeze(1).expand_as(S)

    mAP = MAP(S, C, indexes)
    mRR = MRR(S, C, indexes)
    hr10 = HR10(S, C, indexes)
    mr1 = MR1(S, C, device)
    return {
        "MAP": round(mAP.item(), 3),
        "MRR": round(mRR.item(), 3),
        "MR1": round(mr1.item(), 3),
        "HR@10": round(hr10.item(), 3),
        "nQueries": round(C.shape[0]),
        "nRelevant": round(torch.sum(C).item()),
        "avgRelPerQ": round(torch.mean(torch.sum(C, 1)).item(), 2)
    }


def mask_tensors(S: torch.Tensor, 
                          C: torch.Tensor, 
                          mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get masked tensors. For eg. genre-metrics.
    Args:
        S (torch.Tensor): similarities, will be masked with nan
        C (torch.Tensor): clique relationship, will be masked with 0
        mask (torch.Tensor): 
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: masked tensors
    """
    s, c = S.clone(), C.clone()
    s[~mask], c[~mask] = float("-inf"), 0
    return s, c
    
def cross_genre_metrics(S: torch.Tensor, 
                        C: torch.Tensor,
                        chunk_size: int,
                        genres: torch.Tensor,
                        device: str = "cpu") -> Dict[str, float]:
    """Do the cross-genre evaluation.
    Args:
        S (torch.Tensor): similarities
        C (torch.Tensor): clique memberships
        chunk_size (int): 
        genres (torch.Tensor): genres multi-hot encoded
        device (str, optional): Defaults to "cpu".
    Returns:
        Dict[str, float]: metrics
    """
    # Versions with the exact same genre(s)
    mask_same = torch.eq(genres.unsqueeze(1), genres.unsqueeze(0)).all(dim=-1)
    mask_same = mask_same | ~C
    S_same , C_same = mask_tensors(S, C, mask_same)
    
    # Versions with at least one genre in common
    mask_similar =  torch.matmul(genres.float(), genres.t().float()) > 0
    mask_similar = mask_similar | ~C
    S_similar , C_similar = mask_tensors(S, C, mask_similar)
    
    # Versions with no genre in common
    mask_cross = (~mask_same & ~mask_similar) | ~C
    S_cross , C_cross = mask_tensors(S, C, mask_cross)
    
    if chunk_size:
        merics_same = compute_chunkwise(S_same, C_same, S.size(1)-1, chunk_size, device)
        metrics_partly = compute_chunkwise(S_similar, C_similar, S.size(1)-1, chunk_size, device)
        metrics_none = compute_chunkwise(S_cross, C_cross, S.size(1)-1, chunk_size, device)
    else:
        merics_same = compute_all(S_same, C_same, device)
        metrics_partly = compute_all(S_similar , C_similar, device)
        metrics_none = compute_all(S_cross , C_cross, device)
    return {
        "Same-Genre": merics_same,
        "Similar-Genre": metrics_partly,
        "Cross-Genre": metrics_none
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
    C = create_class_matrix(labels.to("cpu"), zero_diagonal=True).float()

    # Compute the pairwise similarity matrix
    # NOTE: changed the device for product calc to cpu to save GPU memory
    if similarity_search == "MIPS":
        S = pairwise_dot_product(embeddings.to("cpu"))  # (B, N) 
    elif similarity_search == "MCSS":
        S = pairwise_cosine_similarity(embeddings.to("cpu"))  # (B, N)
    else:
        # Use low precision for faster calculations
        S = -1 * pairwise_distance_matrix(embeddings, precision="low".to("cpu"))  # (B, N)

    # Set the similarity of each query with itself to -inf
    S = S.fill_diagonal_(float("-inf"))

    # If Da-TACOS, remove queries with no relevant items (noise works)
    if noise_works:
        non_noise_indices = n_relevant.bool()
        S = S[non_noise_indices]  # (B', N)
        C = C[non_noise_indices]  # (B', N)
        n_relevant = n_relevant[non_noise_indices]  # (B',)

    # storing the evaluation metrics
    metrics = {}
    if chunk_size:
        metrics["Overall"] = compute_chunkwise(S, C, k, chunk_size, device) 
    else:
        metrics["Overall"] = compute_all(S, C, device)
    
    if genres is not None:
        metrics = metrics | cross_genre_metrics(S, C, chunk_size, genres, device)

    # printing the evaluation metrics
    for scheme, metrics in metrics.items():
        print(f"Scheme: {scheme}")
        for metric, value in metrics.items():
            print(f"{metric:>5}: {value}")
    return metrics
