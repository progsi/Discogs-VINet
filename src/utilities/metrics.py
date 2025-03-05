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
                      genres: torch.Tensor = None) -> Dict[str, float]:
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
    
    if genres is not None:
        eq_genre, sim_genre, other_genre = cross_genre_masks(genres)
        eq_genre = eq_genre[has_positives] | (C == 0)
        sim_genre = sim_genre[has_positives] | (C == 0)
        other_genre = other_genre[has_positives] | (C == 0)
        
        TR_eq, MAP_eq = torch.tensor([]), torch.tensor([])
        nQs_eq, nRel_eq = 0, 0
        TR_sim, MAP_sim = torch.tensor([]), torch.tensor([])
        nQs_sim, nRel_sim = 0, 0
        TR_oth, MAP_oth = torch.tensor([]), torch.tensor([])
        nQs_oth, nRel_oth = 0, 0

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
        if genres is not None:
            # get chunks of same genres
            eq = eq_genre[i:i+chunk_size]
            eq_s, eq_c = mask_tensors(s, c, eq)
            eq_has_positives = torch.sum(eq_c, axis=1) > 0
            eq_s, eq_c = eq_s[eq_has_positives], eq_c[eq_has_positives]
            if len(eq_s) > 0:
                ap_eq, tr_eq = compute_partial(eq_s, eq_c, ranking, k, device)
                TR_eq = torch.cat((TR_eq, tr_eq))
                MAP_eq = torch.cat((MAP_eq, ap_eq))
                nqs_eq, nrel_eq = eq_c.shape[0], torch.sum(eq_c).item()
                nQs_eq += nqs_eq
                nRel_eq += nrel_eq

            # get chunk of similar genres
            similar = sim_genre[i:i+chunk_size]
            sim_s, sim_c = mask_tensors(s, c, similar)
            sim_has_positives = torch.sum(sim_c, axis=1) > 0
            sim_s, sim_c = sim_s[sim_has_positives], sim_c[sim_has_positives]
            if len(sim_s) > 0:
                ap_sim, tr_sim = compute_partial(sim_s, sim_c, ranking, k, device)
                TR_sim = torch.cat((TR_sim, tr_sim))
                MAP_sim = torch.cat((MAP_sim, ap_sim))
                nqs_sim, nrel_sim = sim_c.shape[0], torch.sum(sim_c).item()
                nQs_sim += nqs_sim 
                nRel_sim += nrel_sim
            
            # get chunk of other genres
            other = other_genre[i:i+chunk_size]
            oth_s, oth_c = mask_tensors(s, c, other)
            oth_has_positives = torch.sum(oth_c, axis=1) > 0
            oth_s, oth_c = oth_s[oth_has_positives], oth_c[oth_has_positives]
            if len(oth_s) > 0:
                ap_oth, tr_oth = compute_partial(oth_s, oth_c, ranking, k, device)
                TR_oth = torch.cat((TR_oth, tr_oth))
                MAP_oth = torch.cat((MAP_oth, ap_oth))
                nqs_oth, nrel_oth = oth_c.shape[0], torch.sum(oth_c).item()
                nQs_oth += nqs_oth 
                nRel_oth += nrel_oth

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
    
    if genres is not None:
        results["Cross-Genre"] = {
            # same genre
            "MAP-Same": round(MAP_eq.mean().item(), 3),
            "MRR-Same": round(torch.nanmean((1/TR_eq).float()).item(), 3),
            "MR1-Same": round(torch.nanmean(TR_eq).float().item(), 2),
            "nQueries-Same": round(nQs_eq),
            "nRelevant-Same": round(nRel_eq),
            "avgRelPerQ-Same": round(nRel_eq / nQs_eq, 2),
            # similar genre
            "MAP-Similar": round(MAP_sim.mean().item(), 3),
            "MRR-Similar": round(torch.nanmean((1/TR_sim).float()).item(), 3),
            "MR1-Similar": round(torch.nanmean(TR_sim).float().item(), 2),
            "nQueries-Similar": round(nQs_sim),
            "nRelevant-Similar": round(nRel_sim),
            "avgRelPerQ-Similar": round(nRel_sim / nQs_sim, 2),
            # cross genre
            "MAP-Cross": round(MAP_oth.mean().item(), 3),
            "MRR-Cross": round(torch.nanmean((1/TR_oth).float()).item(), 3),
            "MR1-Cross": round(torch.nanmean(TR_oth).float().item(), 2),
            "nQueries-Cross": round(nQs_oth),
            "nRelevant-Cross": round(nRel_oth),
            "avgRelPerQ-Cross": round(nRel_oth / nQs_oth, 2)
        }
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
    
    S, C = S.to(device), C.to(device)
    indexes = torch.arange(S.size(0), device=device).unsqueeze(1).expand_as(S)

    mAP = MAP(S, C, indexes)
    mRR = MRR(S, C, indexes)
    hr10 = HR10(S, C, indexes)
    mr1 = MR1(S, C, device)
    results = {
        "MAP": round(mAP.item(), 3),
        "MRR": round(mRR.item(), 3),
        "MR1": round(mr1.item(), 3),
        "HR@10": round(hr10.item(), 3),
        "nQueries": round(C.shape[0]),
        "nRelevant": round(torch.sum(C).item()),
        "avgRelPerQ": round(torch.mean(torch.sum(C, 1)).item(), 2)
    }

    if genre is not None:
        genre_results = cross_genre_metrics_all(S, C, genre, device)
        results = results | genre_results
    return results
    
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
    
def cross_genre_masks(genres: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generates genre masks.
    Args:
        genres (torch.Tensor): one-hot-encoded genre tensor
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: same, similar, different genre masks
    """
    mask_same = torch.eq(genres.unsqueeze(1), genres.unsqueeze(0)).all(dim=-1)
    mask_similar =  torch.matmul(genres.float(), genres.t().float()) > 0
    mask_cross = (~mask_same & ~mask_similar)
    return mask_same, mask_similar, mask_cross


def cross_genre_metrics_all(S: torch.Tensor, 
                        C: torch.Tensor,
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
    mask_same, mask_similar, mask_cross = cross_genre_masks(genres)
    negatives = (C != 1)
    
    mask_same = mask_same | negatives
    S_same , C_same = mask_tensors(S, C, mask_same)
    
    # Versions with at least one genre in common
    mask_similar = mask_similar | negatives
    S_similar , C_similar = mask_tensors(S, C, mask_similar)
    
    # Versions with no genre in common
    mask_cross = (~mask_same & ~mask_similar) | negatives
    S_cross , C_cross = mask_tensors(S, C, mask_cross)
    
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
                                               genres) 
    else:
        metrics = compute_all(S, C, device, genres)
    
    # printing the evaluation metrics
    for scheme, submetrics in metrics.items():
        print(f"Scheme: {scheme}")
        for submetric, value in submetrics.items():
            print(f"{submetric:>5}: {value}")
    return metrics
