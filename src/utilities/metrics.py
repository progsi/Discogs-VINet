from typing import Dict, Tuple

import torch
from torchmetrics.retrieval import RetrievalMAP, RetrievalMRR, RetrievalHitRate

from src.utilities.tensor_op import (
    create_class_matrix,
    pairwise_similarity_search,
)

MAP = RetrievalMAP(empty_target_action="skip", top_k=None)
MRR = RetrievalMRR(empty_target_action="skip", top_k=None)
HR10 = RetrievalHitRate(empty_target_action="skip", top_k=10)

def compute_partial(s: torch.Tensor,
                    c: torch.Tensor,
                    r: torch.Tensor,
                    k: int,
                    device: str = "cpu") -> Tuple[float, float]:
    """_summary_
    Args:
        s (torch.Tensor): subset of similarities
        c (torch.Tensor): subset of true relationships
        r (torch.Tensor): ranking tensor
        k (int): k first ranks to consider
        device (str, optional): Torch device string. Defaults to "cpu".
    Returns:
        Tuple[float, float]: average precision, top rank
    """
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
    prec_at_k = torch.cumsum(relevance, 1).div_(r)  # (B', N-1)
    ap = torch.sum(prec_at_k * relevance, 1).div_(n_relevant).cpu()  # (B',)
    return ap, top_rank
    
def compute_chunkwise(X: torch.Tensor, 
                      C: torch.Tensor, 
                      k: int, 
                      chunk_size: int, 
                      device: str = "cpu", 
                      similarity_search: str = None,
                      genres_multihot: torch.Tensor = None,
                      genre_idx_to_label: Dict[int,str] = None) -> Dict[str, float]:
    """Compute the evaluation metrics in chunks to avoid memory issues.
    Args:
        X (torch.Tensor): either similarity matrix or embeddings
        C (torch.Tensor): class membership matrix
        k (int): first ranks to consider for P@k
        chunk_size (int): chunks size
        precomputed (bool): if True, X is similarity matrix, else embeddings
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
    X, C = X[has_positives], C[has_positives]
    
    # init cross genre metrics
    if genres_multihot is not None:
        genres_multihot = genres_multihot[has_positives]
        masks = cross_genre_masks(genres_multihot)
        
        if genre_idx_to_label is not None:
            masks = masks | genre_wise_masks(genre_idx_to_label, genres_multihot)
        
        metrics_genre = {}
        for name, _ in masks.items():
            metrics_genre[name] = {
                "TR": torch.tensor([]), "MAP": torch.tensor([]),
                "nQueries": 0, "nRelevant": 0}

    # Iterate over the chunks
    for i, (x, c) in enumerate(zip(X.split(chunk_size), C.split(chunk_size))):
        i = i * chunk_size

        # Move the tensors to the device
        x = x.to(device)
        
        if similarity_search: # X is embedding, so we need to compute
            s = pairwise_similarity_search(x, X)
        else: # X is similarity matrix, so we can directly use it
            s = x
        
        c = c.to(device)  

        ap, top_rank = compute_partial(s, c, ranking, k, device)
        # Concatenate the results from all chunks
        TR = torch.cat((TR, top_rank))
        MAP = torch.cat((MAP, ap))
        
        # NOTE: sry, this function is blowing up, I will refactor it later
        if genres_multihot is not None:
            # get chunks of same genres
            for name, cross_mask in masks.items():
                _cross_mask = cross_mask[i:i+chunk_size]
                _s, _c = mask_tensors(s, c, _cross_mask)
                _has_positives = torch.sum(_c, axis=1) > 0
                _s, _c = _s[_has_positives], _c[_has_positives]
                if len(_s) > 0:
                    ap, top_rank = compute_partial(_s, _c, ranking, k, device)
                    metrics_genre[name]["TR"] = torch.cat((metrics_genre[name]["TR"], top_rank))
                    metrics_genre[name]["MAP"] = torch.cat((metrics_genre[name]["MAP"], ap))
                    nqs, nrel = _c.shape[0], torch.sum(_c).item()
                    metrics_genre[name]["nQueries"] += nqs
                    metrics_genre[name]["nRelevant"] += nrel

    # computing the final evaluation metrics
    MR1 = torch.nanmean(TR).float().item() # before: TR.mean().item()
    MRR = torch.nanmean((1/TR).float()).item() # before: (1 / TR).mean().item()
    MAP = MAP.mean().item()
    
    results = {}
    results["Overall"] = {
        "MAP": round(MAP, 3),
        "MRR": round(MRR, 3),
        "MR1": round(MR1, 2),
        "nQueries": round(C.shape[0]),
        "nRelevant": round(torch.sum(C).item()),
        "avgRelPerQ": round(torch.mean(torch.sum(C, 1)).item(), 2)
    }
    
    if genres_multihot is not None:
        results["Genre"] = {}
        for group, metrics in metrics_genre.items():
            group = group.replace(" ", "_")
            nQs, nRel = metrics["nQueries"], metrics["nRelevant"]
            results["Genre"][group + "-" + "MAP"] = round(metrics["MAP"].mean().item(), 3)
            results["Genre"][group + "-" + "MRR"] = round(torch.nanmean((1/metrics["TR"]).float()).item(), 3)
            results["Genre"][group + "-" + "MR1"] = round(torch.nanmean(metrics["TR"]).float().item(), 2)
            results["Genre"][group + "-" + "nQueries"] = round(nQs)
            results["Genre"][group + "-" + "nRelevant"] = round(nRel)
            results["Genre"][group + "-" + "avgRelPerQ"] = round(nRel / nQs, 2)
    return results
    
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
                device: str = "cpu",
                genre: torch.Tensor = None) -> Dict[str, float]:
    
    results = compute_metrics_all(S, C, device)

    if genre is not None:
        genre_results = cross_genre_metrics_all(S, C, genre, device)
        results = results | genre_results
    return results

def compute_metrics_all(S: torch.Tensor, C: torch.Tensor, device: str = "cpu") -> Dict[str, float]:
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
    
def cross_genre_masks(genres_multihot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generates genre masks.
    Args:
        genres_multihot (torch.Tensor): one-hot-encoded genre tensor
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: same, similar, different genre masks
    """
    mask_match = torch.eq(genres_multihot.unsqueeze(1), genres_multihot.unsqueeze(0)).all(dim=-1)
    mask_match &= genres_multihot.sum(dim=-1).unsqueeze(1) > 0
    mask_min1 =  torch.matmul(genres_multihot.float(), genres_multihot.t().float()) > 0
    mask_overlap = mask_min1 & ~mask_match
    mask_mismatch = ~mask_min1
    return {
        "Genre-Match": mask_match, 
        "Genre-Min1": mask_min1, 
        "Genre-Overlap": mask_overlap, 
        "Genre-Mismatch": mask_mismatch
        }

def genre_wise_masks(genre_idx_to_label: Dict[int,str], genres_multihot: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Generates genre masks.
    Args:
        genres_multihot (torch.Tensor): one-hot-encoded genre tensor
    Returns:
        Dict[str, torch.Tensor]: genre masks
    """
    genre_masks = {}
    for genre, label in genre_idx_to_label.items():
        mask = genres_multihot[:, genre] == 1
        genre_masks[label] = mask.unsqueeze(1).repeat(1, genres_multihot.size(0))
    return genre_masks

def cross_genre_metrics_all(S: torch.Tensor, 
                        C: torch.Tensor,
                        genres_multihot: torch.Tensor,
                        genre_idx_to_label: Dict[int,str] = None,
                        device: str = "cpu") -> Dict[str, float]:
    """Do the cross-genre evaluation.
    Args:
        S (torch.Tensor): similarities
        C (torch.Tensor): clique memberships
        chunk_size (int): 
        genres_multihot (torch.Tensor): genres multi-hot encoded
        genre_idx_to_label: (Dict[int,str]): mapping from genre index to label
        device (str, optional): Defaults to "cpu".
    Returns:
        Dict[str, float]: metrics
    """
    results = {}
    masks_cross = cross_genre_masks(genres_multihot)
    
    for name, mask in masks_cross.items():
        S, C = mask_tensors(S, C, mask)
        metrics = compute_metrics_all(S, C, device)
        results[name] = metrics
    
    return results

def calculate_metrics(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    similarity_search: str = "MIPS",
    chunk_size: int = 256,
    device: str = "cpu",
    genres_multihot: torch.Tensor = None,
    genre_idx_to_label: Dict[int, str] = None,
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
    genres_multihot: 
        Multi-hot encoded genre labels per version
    genre_idx_to_label
        Mapping from genre index to names
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
    precompute = N < 40_000
    if precompute:
        S = pairwise_similarity_search(embeddings.to("cpu"))  # (B, N)

        # Set the similarity of each query with itself to -inf
        S = S.fill_diagonal_(float("-inf"))
    else:
        S = None

    # storing the evaluation metrics
    metrics = {}
    if chunk_size:
        if S is not None:
            X = S
        else: 
            X = embeddings
        metrics = compute_chunkwise(X, C, k, 
                                               chunk_size, 
                                               device, 
                                               similarity_search if not precompute else None,
                                               genres_multihot,
                                               genre_idx_to_label) 
    else:
        metrics = compute_all(S, C, device, genres_multihot, genre_idx_to_label)
    
    # printing the evaluation metrics
    for scheme, submetrics in metrics.items():
        print(f"Scheme: {scheme}")
        for submetric, value in submetrics.items():
            print(f"{submetric:>5}: {value}")
    return metrics
