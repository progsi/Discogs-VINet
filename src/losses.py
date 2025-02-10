from typing import List, Dict, Type, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utilities.tensor_op import pairwise_distance_matrix, create_pos_neg_masks

INDUCTIVE_KEY = "LOSS_INDUCTIVE"
TRIPLET_LOSS = 'triplet'
CENTER_LOSS = 'center'
SOFTMAX_LOSS = 'softmax'
PROTOTYPICAL_LOSS = 'prototypical'
FOCAL_LOSS = 'focal'

def init_single_loss(loss_name: str, loss_params: dict) -> Type[nn.Module]:
    """Initialize a single loss function.
    Args:
        loss_name (str): name from config file
        loss_params (dict): parameters from config file
    Returns:
        Type[nn.Module]: initialized loss class
    """
    print(f"Initializing {loss_name} loss with parameters:")
    for key, value in loss_params.items():
        print(f"    {key.lower()}: {value}")
    if loss_name == TRIPLET_LOSS.upper():
        return TripletMarginLoss(margin=loss_params['MARGIN'], 
                                 positive_mining=loss_params['POSITIVE_MINING'],
                                 negative_mining=loss_params['NEGATIVE_MINING'])
    elif loss_name == CENTER_LOSS.upper():
        return CenterLoss(num_cls=loss_params['NUM_CLS'], feat_dim=loss_params['FEAT_DIM'])
    elif loss_name == FOCAL_LOSS.upper():
        return FocalLoss(gamma=loss_params['GAMMA'])
    elif loss_name == SOFTMAX_LOSS.upper():
        return nn.CrossEntropyLoss(label_smoothing=loss_params['LABEL_SMOOTHING'])
    elif loss_name == PROTOTYPICAL_LOSS.upper():
        return PrototypicalLoss(n_support=loss_params['N_SUPPORT'])

def init_loss(loss_config: dict, loss_config_inductive: dict) -> Type[nn.Module]:
    """Initialize a loss function, either single loss or weihted Multiloss.
    Args:
        loss_config (dict): dictionary based on config file
        loss_config_inductive (dict): dictionary based on config file
    Returns:
        Type[nn.Module]: initialized loss class
    """
    assert len(loss_config.keys()) >= 1, f"Minimum one loss is required!"
    if len(loss_config.keys()) == 1:
        loss_name, loss_params = list(loss_config.items())[0]
        return init_single_loss(loss_name, loss_params)
    else:
        if loss_config_inductive:
            return WeightedMultilossInductive(loss_config, loss_config_inductive)
        else:
            return WeightedMultiloss(loss_config)

def is_cls_loss(loss_name: str) -> bool:
    """Check if the loss is a classification loss.
    Args:
        loss_name (str): name of the loss
    Returns:
        bool: True if classification loss
    """
    return loss_name in ['SOFTMAX', 'FOCAL']

def requires_cls_labels(loss: nn.Module) -> bool:
    """Check if the loss class requires all cls labels (eg. Center Loss or Classification loss). 
    Args:
        loss (nn.Module): loss module
    Returns:
        bool: 
    """
    return isinstance(loss, nn.CrossEntropyLoss) or isinstance(loss, CenterLoss) or isinstance(loss, WeightedMultiloss)

class WeightedMultiloss(nn.Module):
    """A class to make the training with multiple losses easier.
    Args:
        loss_config (dict): sub-dict with loss configuration
    """
    def __init__(self, loss_config: dict):
        super(WeightedMultiloss, self).__init__()

        self.losses, self.loss_stats = self._init_losses(loss_config)
    
    def _init_losses(self, loss_config: dict):
        losses = {}
        loss_stats = {}
        for key, loss_params in loss_config.items():
            name_key = loss_params.get('LOSS_NAME', None)
            init_name = name_key if name_key else key
            loss_fn = init_single_loss(init_name, loss_params)
            weight = loss_params.get('WEIGHT', 1.0)
            losses[key] = (loss_fn, weight)
            loss_stats[key] = {}
        return losses, loss_stats
            
    def get_stats(self):
        return self.loss_stats
    
    def forward(self, preds: Dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        """Calculates Multiloss.
        Args:
            preds (Union[torch.Tensor, Dict[str, torch.Tensor]]): Mapping of loss names to predictions (embeddings or cls) per loss
            labels (torch.Tensor): Target labels.
        Returns:
            torch.Tensor: total loss
        """
        
        total_loss = 0
        
        for loss_name, (loss_fn, weight) in self.losses.items():
            
            x = preds[loss_name] 

            # compute and weigh
            loss = loss_fn(x, labels)
            loss_weighted = loss * weight
            
            # collect stats
            self.loss_stats[loss_name]["unweighted"] = loss.detach().item()
            self.loss_stats[loss_name]["weighted"] = loss_weighted.detach().item()
            total_loss += loss_weighted   
        return total_loss
    
class WeightedMultilossInductive(WeightedMultiloss):
    def __init__(self, loss_config: dict, loss_config_inductive: dict):
        super(WeightedMultilossInductive, self).__init__(loss_config)
        self.inductive_losses, self.inductive_stats = self._init_losses(loss_config_inductive)
        
    def forward(self, preds: Dict[str, torch.Tensor], labels: Dict[str,torch.Tensor]) -> torch.Tensor:
        """Calculates Multiloss.
        Args:
            preds (Union[torch.Tensor, Dict[str, torch.Tensor]]): Mapping of loss names to predictions (embeddings or cls) per loss
            labels (torch.Tensor): Target labels.
        Returns:
            torch.Tensor: total loss
        """
        
        total_loss = super().forward(preds, labels["cls"])
        # TODO: must implement properly here!
        for inductive_cls, (loss_fn, weight) in self.inductive_losses.items(): 
            
            x = preds[inductive_cls] 

            # compute and weigh
            loss = loss_fn(x, labels[inductive_cls.lower()])
            loss_weighted = loss * weight
            
            # collect stats
            self.inductive_stats[inductive_cls]["unweighted"] = loss.detach().item()
            self.inductive_stats[inductive_cls]["weighted"] = loss_weighted.detach().item()
            total_loss += loss_weighted
        return total_loss
            

class TripletMarginLoss(nn.Module):
    """A class to compute the triplet loss for given embeddings. 
    Adopted from https://github.com/raraz15/Discogs-VINet/blob/main/model/loss.py
        positive_mining: str
            Either "random", "easy, or "hard".
        negative_mining: str
            Either "random" or "hard".
        margin: float
            The margin value in the triplet loss. In the case of semi-hard mining, this
            value is also used to sample the negatives.
        squared_distance: bool
            If True, the pairwise distance matrix is squared before computing the loss.
        non_zero_mean: bool
            If True, the loss is averaged only over the non-zero losses.
        stats: bool
            If True, return the number of positive triplets in the batch. Else return None.
    """
    def __init__(self, margin: float, positive_mining: str, negative_mining: str, squared_distance: bool = False, 
                 non_zero_mean: bool = False, stats: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.margin = margin
        self.positive_mining_mode = positive_mining
        self.negative_mining_mode = negative_mining
        self.squared_distance = squared_distance
        self.non_zero_mean = non_zero_mean
        self.stats = stats
        
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Union[int, None]]:
        """Compute the triplet loss for given embeddings. We use online sampling to
        select the positive and negative samples. You can choose between random and hard
        mining for both the negatives and positives. The margin value is used to compute the
        triplet loss. If squared_distance is True, the pairwise distance matrix is squared.

        Parameters:
        -----------
        embeddings: torch.Tensor
            2D tensor of shape (n, d) where n is the number of samples and d is the
            dimensionality of the samples.
        labels: torch.Tensor
            1D tensor of shape (n,) where labels[i] is the int label of the i-th sample.
        Returns:
        --------
        loss: torch.Tensor
            Tensor of shape (1,) representing the average triplet loss of the batch.
        num_unsatisfied_triplets: int
            Number of triplets that do not satisfy the margin condition in the batch.
            Only returned if stats is True.
        """

        # Compute the pairwise distance matrix of the anchors
        distance_matrix = pairwise_distance_matrix(embeddings, squared=self.squared_distance)

        # Create masks for the positive and negative samples
        mask_pos, mask_neg = create_pos_neg_masks(labels)

        # Sample the positives first
        if self.positive_mining_mode.lower() == "random":
            dist_AP, _ = self.random_positive_sampling(distance_matrix, mask_pos)
        elif self.positive_mining_mode.lower() == "hard":
            dist_AP, _ = self.hard_positive_mining(distance_matrix, mask_pos)
        elif self.positive_mining_mode.lower() == "easy":
            dist_AP, _ = self.easy_positive_mining(distance_matrix, mask_pos)
        else:
            raise ValueError("Other positive mining types are not supported.")

        if self.negative_mining_mode.lower() == "random":
            dist_AN, _ = self.random_negative_sampling(distance_matrix, mask_neg)
        elif self.negative_mining_mode.lower() == "hard":
            dist_AN, _ = self.hard_negative_mining(distance_matrix, mask_neg)
        elif self.negative_mining_mode.lower() == "semi-hard":
            dist_AN, _ = self.semi_hard_negative_mining(
                distance_matrix, dist_AP, mask_neg, self.margin
            )
        else:
            raise ValueError("Other negative mining types are not supported.")

        # Compute the triplet loss
        loss = F.relu(dist_AP - dist_AN + self.margin)

        # See how many triplets per batch are positive
        if self.stats:
            num_unsatisfied_triplets = int((loss > 0).sum().item())
        else:
            num_unsatisfied_triplets = None

        # Average the loss over the batch (can filter out zero losses if needed)
        if self.non_zero_mean:
            mask = loss > 0
            if any(mask):
                loss = loss[mask]
        loss = loss.mean()

        # TODO: implement num_unsatisfied_triplets
        if num_unsatisfied_triplets is not None:
            return loss, num_unsatisfied_triplets
        else:
            return loss
    
    @staticmethod
    def random_positive_sampling(
        distance_matrix: torch.Tensor, mask_pos: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given a pairwise distance matrix of all the samples and a mask that indicates the
        possible indices for sampling positives, randomly sample a positive sample for each
        anchor. All samples are treated as anchor points.

        Parameters:
        -----------
        distance_matrix: torch.Tensor
            2D tensor of shape (n, n) where distance_matrix[i, j] is the distance between
            x[i] and y[j], i.e. the pairwise distance matrix.
        mask_pos: torch.Tensor
            See create_pos_neg_masks() for details.

        Returns:
        --------
        anchor_pos_distances: torch.Tensor
            1D tensor of shape (n,), distances between the anchors and their chosen
            positive samples.
        positive_indices: torch.Tensor
            1D tensor of shape (n,) where positive_indices[i] is the index of the positive
            sample for the i-th anchor point.
        """

        # Get the indices of the positive samples for each anchor point
        positive_indices = torch.multinomial(mask_pos, 1)

        # Get the distances between the anchors and their positive samples
        anchor_pos_distances = torch.gather(distance_matrix, 1, positive_indices)

        return anchor_pos_distances.squeeze(1), positive_indices.squeeze(1)
    
    @staticmethod
    def random_negative_sampling(
        distance_matrix: torch.Tensor, mask_neg: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given a pairwise distance matrix of all the samples and a mask that indicates the
        possible indices for sampling negatives, randomly sample a negative sample for each
        anchor. All samples are treated as anchor points.

        Parameters:
        -----------
        distance_matrix: torch.Tensor
            2D tensor of shape (n, n) where distance_matrix[i, j] is the distance between
            x[i] and y[j], i.e. the pairwise distance matrix.
        mask_neg: torch.Tensor
            See create_pos_neg_masks() for details.

        Returns:
        --------
        anchor_neg_distances: torch.Tensor
            1D tensor of shape (n,), distances between the anchors and their chosen
            negative samples.
        negative_indices: torch.Tensor
            1D tensor of shape (n,) where negative_indices[i] is the index of the negative
            sample for the i-th anchor point.
        """

        # Get the indices of the negative samples for each anchor point
        negative_indices = torch.multinomial(mask_neg, 1)

        # Get the distances between the anchors and their negative samples
        anchor_neg_distances = torch.gather(distance_matrix, 1, negative_indices)

        return anchor_neg_distances.squeeze(1), negative_indices.squeeze(1)
    
    @staticmethod
    def hard_positive_mining(
        distance_matrix: torch.Tensor, mask_pos: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given a pairwise distance matrix of all the anchors and a mask that indicates the
        possible indices for sampling positives, mine the hardest positive sample for each
        anchor. All samples are treated as anchor points.

        Parameters:
        -----------
        distance_matrix: torch.Tensor
            2D tensor of shape (n, n) where distance_matrix[i, j] is the distance between
            x[i] and y[j], i.e. the pairwise distance matrix.
        mask_pos: torch.Tensor
            See create_pos_neg_masks() for details.

        Returns:
        --------
        anchor_pos_distances: torch.Tensor
            1D tensor of shape (n,) containing the distances between the anchors and their
            chosen positive samples.
        positive_indices: torch.Tensor
            1D tensor of shape (n,) where positive_indices[i] is the index of the hardest
            positive sample for the i-th anchor point.
        """

        # Select the hardest positive for each anchor
        anchor_pos_distances, positive_indices = torch.max(distance_matrix * mask_pos, 1)

        return anchor_pos_distances, positive_indices

    @staticmethod
    def hard_negative_mining(
        distance_matrix: torch.Tensor, mask_neg: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given a pairwise distance matrix of all the anchors and a mask that indicates the
        possible indices for sampling negatives, mine the hardest negative sample for each
        anchor. All samples are treated as anchor points.

        Parameters:
        -----------
        distance_matrix: torch.Tensor
            2D tensor of shape (n, n) where distance_matrix[i, j] is the distance between
            x[i] and y[j], i.e. the pairwise distance matrix.
        mask_neg: torch.Tensor
            See create_pos_neg_masks() for details.

        Returns:
        --------
        anchor_neg_distances: torch.Tensor
            1D tensor of shape (n,) containing the distances between the anchors and their
            chosen negative samples.
        negative_indices: torch.Tensor
            1D tensor of shape (n,) where negative_indices[i] is the index of the hardest
            negative sample for the i-th anchor point.
        """

        # make sure same data type as distance_matrix
        inf = torch.tensor(
            float("inf"), device=distance_matrix.device, dtype=distance_matrix.dtype
        )
        zero = torch.tensor(0.0, device=distance_matrix.device, dtype=distance_matrix.dtype)

        # Modify the distance matrix to only consider the negative samples
        mask_neg = torch.where(mask_neg == 0, inf, zero)

        # Get the indices of the hardest negative samples for each anchor point
        anchor_neg_distances, negative_indices = torch.min(distance_matrix + mask_neg, 1)

        return anchor_neg_distances, negative_indices

    # TODO: check if this is correct
    @staticmethod
    def semi_hard_negative_mining(
        distance_matrix: torch.Tensor,
        dist_AP: torch.Tensor,
        mask_neg: torch.Tensor,
        margin: float,
        mode: str = "hard",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given a pairwise distance matrix of all the anchors and a mask that indicates the
        possible indices for sampling negatives, mine the semi-hard negative sample for each
        anchor. All samples are treated as anchor points. If there are no possible semi-hard
        negatives, sample randomly.

        Parameters:
        -----------
        distance_matrix: torch.Tensor
            2D tensor of shape (n, n) where distance_matrix[i, j] is the distance between
            x[i] and y[j], i.e. the pairwise distance matrix.
        dist_AP: torch.Tensor
            1D tensor of shape (n,) where dist_AP[i] is the distance between the i-th anchor
            and its positive sample.
        mask_neg: torch.Tensor
            See create_pos_neg_masks() for details.
        margin: float
            Margin for the triplet loss.
        mode: str
            Either "hard" or "random". If "hard", the hardest negative sample from the region
            is selected for each anchor. If "random", a random negative sample is selected
            from the region.

        Returns:
        --------
        anchor_neg_distances: torch.Tensor
            1D tensor of shape (n,) containing the distances between the anchors and their
            chosen negative samples.
        negative_indices: torch.Tensor
            1D tensor of shape (n,) where negative_indices[i] is the index of the semi-hard
            negative sample for the i-th anchor point.
        """

        raise NotImplementedError("Having difficulty in implementing it.")

        assert mode in {"hard", "random"}, "mode must be either 'hard' or 'random'"
        assert margin > 0, "margin must be greater than 0"
        assert (
            distance_matrix.shape[0] == dist_AP.shape[0]
        ), "distance_matrix and dist_AP must have the same length"
        assert dist_AP.ndim == 1, "dist_AP must be a 1D tensor"

        # Get the region for semi-hard negatives
        mask_semi_hard_neg = (
            (dist_AP.unsqueeze(1) < distance_matrix)
            & (distance_matrix < (dist_AP.unsqueeze(1) + margin))
            & mask_neg.bool()
        ).float()

        # Initialize the tensors to store the distances and indices of the semi-hard negatives
        device = distance_matrix.device
        n = distance_matrix.shape[0]
        anchor_neg_distances = torch.zeros(n, dtype=torch.float32, device=device)
        negative_indices = torch.zeros(n, dtype=torch.int32, device=device)

        # Search for a semi-hard negative for each anchor, positive pair
        for i in range(n):
            dist = distance_matrix[i].unsqueeze(0)
            mask = mask_semi_hard_neg[i].unsqueeze(0)
            # check if the hollow-sphere is empty
            if mask.any():  # there is at least one semi-hard negative
                if mode == "hard":  # choose the hardest example in the hollow-sphere
                    negative_indices[i], anchor_neg_distances[i] = hard_negative_mining(
                        dist, mask
                    )
                else:  # choose a random example in the hollow-sphere
                    negative_indices[i], anchor_neg_distances[i] = random_negative_sampling(
                        dist, mask
                    )
            else:  # there are no semi-hard negatives
                if mode == "hard":  # resort to hard negatives
                    negative_indices[i], anchor_neg_distances[i] = hard_negative_mining(
                        dist, mask_neg[i].unsqueeze(0)
                    )
                else:  # resort to random negatives
                    negative_indices[i], anchor_neg_distances[i] = random_negative_sampling(
                        dist, mask_neg[i].unsqueeze(0)
                    )
        return anchor_neg_distances, negative_indices

    @staticmethod
    def easy_positive_mining(
        distance_matrix: torch.Tensor, mask_pos: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given a pairwise distance matrix of all the anchors and a mask that indicates the
        possible indices for sampling positives, mine the easiest positive sample for each
        anchor. All samples are treated as anchor points.

        Parameters:
        -----------
        distance_matrix: torch.Tensor
            2D tensor of shape (n, n) where distance_matrix[i, j] is the distance between
            x[i] and y[j], i.e. the pairwise distance matrix.
        mask_pos: torch.Tensor
            See create_pos_neg_masks() for details.

        Returns:
        --------
        anchor_pos_distances: torch.Tensor
            1D tensor of shape (n,) containing the distances between the anchors and their
            chosen positive samples.
        positive_indices: torch.Tensor
            1D tensor of shape (n,) where positive_indices[i] is the index of the hardest
            positive sample for the i-th anchor point.
        """

        # make sure same data type as distance_matrix
        inf = torch.tensor(
            float("inf"), device=distance_matrix.device, dtype=distance_matrix.dtype
        )
        zero = torch.tensor(0.0, device=distance_matrix.device, dtype=distance_matrix.dtype)

        # Modify the distance matrix to only consider the positive samples
        mask_pos = torch.where(mask_pos == 0, inf, zero)

        # Select the easiest positive for each anchor
        anchor_pos_distances, positive_indices = torch.min(distance_matrix + mask_pos, 1)

        return anchor_pos_distances, positive_indices
    
class CenterLoss(nn.Module):
    """Adopted from https://github.com/KaiyangZhou/pytorch-center-loss/blob/master/center_loss.py
    Reference: https://link.springer.com/chapter/10.1007/978-3-319-46478-7_31
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_cls: int = 10, feat_dim: int = 2, device: str = 'cuda'):
        super(CenterLoss, self).__init__()
        self.num_cls = num_cls 
        self.feat_dim = feat_dim
        self.device = device

        self.centers = nn.Parameter(torch.randn(self.num_cls, self.feat_dim).to(self.device))

    def forward(self, embeddings: torch.tensor, labels: torch.tensor) -> torch.tensor:
        """Compute loss.
        Args:
            embeddings (torch.tensor): embeddings
            labels (torch.tensor): class labels
        Returns:
            torch.tensor: loss
        """
        N = embeddings.size(0)
        distmat = torch.pow(embeddings, 2).sum(dim=1, keepdim=True).expand(N, self.num_cls) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_cls, N).t()
        distmat.to(embeddings.dtype).addmm_(embeddings, self.centers.to(embeddings.dtype).t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_cls).long().to(self.device)
        labels = labels.unsqueeze(1).expand(N, self.num_cls)
        mask = labels.eq(classes.expand(N, self.num_cls))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / N

        return loss

class PrototypicalLoss(torch.nn.Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    Adopted from: https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch/blob/master/src/prototypical_loss.py
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py
    '''
    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, embeddings, labels):
        '''
        Compute the barycentres by averaging the features of n_support
        samples for each class in target, computes then the distances from each
        samples' features to each one of the barycentres, computes the
        log_probability for each n_query samples for each one of the current
        classes, of appartaining to a class c, loss and accuracy are then computed
        and returned
        Args:
        - embeddings: the model output for a batch of samples
        - labels: ground truth for the above batch of samples
        '''
        def supp_idxs(c):
            # FIXME when torch will support where as np
            return labels.eq(c).nonzero()[:self.n_support].squeeze(1)

        # FIXME when torch.unique will be available on cuda too
        classes = torch.unique(labels)
        n_classes = len(classes)
        # FIXME when torch will support where as np
        # assuming n_query, n_target constants
        n_query = labels.eq(classes[0].item()).sum().item() - self.n_support

        support_idxs = list(map(supp_idxs, classes))

        prototypes = torch.stack([embeddings[idx_list].mean(0) for idx_list in support_idxs])
        # FIXME when torch will support where as np
        query_idxs = torch.stack(list(map(lambda c: labels.eq(c).nonzero()[self.n_support:], classes))).view(-1)

        query_samples = embeddings[query_idxs]
        dists = pairwise_distance_matrix(query_samples, prototypes)

        log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

        target_inds = torch.arange(0, n_classes).to(labels.device)
        target_inds = target_inds.view(n_classes, 1, 1)
        target_inds = target_inds.expand(n_classes, n_query, 1).long()

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()

        return loss_val # ,  acc_val TODO: is this acc_val needed?
  
class FocalLoss(nn.Module):
    """Adopted from: https://github.com/Liu-Feng-deeplearning/CoverHunter/blob/main/src/loss.py 
    Reference https://arxiv.org/abs/1708.02002
    """
    def __init__(self, gamma: float = 2., alpha: List = None, num_cls: int = -1,
                reduction: str = 'mean'):

        super(FocalLoss, self).__init__()
        if reduction not in ['mean', 'sum']:
            raise NotImplementedError(
            'Reduction {} not implemented.'.format(reduction))
        self._reduction = reduction
        self._alpha = alpha
        self._gamma = gamma
        if alpha is not None:
            assert len(alpha) <= num_cls, "{} != {}".format(len(alpha), num_cls)
            self._alpha = torch.tensor(self._alpha)
        self._eps = torch.finfo(torch.float32).eps
        return

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """ compute focal loss for pred and label
        Args:
            y_pred: [batch_size, num_cls]
            y_true: [batch_size]
        Returns:
            loss
        """
        b = y_pred.size(0)
        y_pred_softmax = torch.nn.Softmax(dim=1)(y_pred) + self._eps
        ce = -torch.log(y_pred_softmax)
        ce = ce.gather(1, y_true.view(-1, 1))

        y_pred_softmax = y_pred_softmax.gather(1, y_true.view(-1, 1))
        weight = torch.pow(torch.sub(1., y_pred_softmax), self._gamma)

        if self._alpha is not None:
            self._alpha = self._alpha.to(y_pred.device)
            alpha = self._alpha.gather(0, y_true.view(-1))
            alpha = alpha.unsqueeze(1)
            alpha = alpha / torch.sum(alpha) * b
            weight = torch.mul(alpha, weight)
        fl_loss = torch.mul(weight, ce).squeeze(1)
        return self._reduce(fl_loss)

    def _reduce(self, x):
        if self._reduction == 'mean':
            return torch.mean(x)
        else:
            return torch.sum(x)

class FocalLoss2(nn.Module):
    """Adopted from https://pytorch.org/vision/main/_modules/torchvision/ops/focal_loss.html
    """
    def __init__(self, gamma=0, alpha=None, size_average=True, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        p = torch.sigmoid(y_pred)
        ce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction="none")
        p_t = p * y_true + (1 - p) * (1 - y_true)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha and self.alpha >= 0:
            alpha_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
            loss = alpha_t * loss

        # Check reduction option and return loss accordingly
        if self.reduction == "none" or self.reduction is None:
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{self.reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        return loss
            

