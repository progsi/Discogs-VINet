import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from torch import Tensor
from typing import Union, Type
from pytorch_metric_learning import losses, miners

def init_single_loss(loss_name: str, loss_params: dict) -> Type[nn.Module]:
    """Initialize a single loss function.
    Args:
        loss_name (str): name from config file
        loss_params (dict): parameters from config file
    Returns:
        Type[nn.Module]: initialized loss class
    """
    print(f"Initializing {loss_name.lower()} loss with parameters:\n")
    for key, value in loss_params.items():
        print(f"{key.lower()}: {value}")
    if loss_name == 'TRIPLET':
        return TripletMarginLoss(margin=loss_params['MARGIN'], mining=loss_params['MINING'])
    elif loss_name == 'CENTER':
        return CenterLoss(num_classes=loss_params['NUM_CLASSES'], feat_dim=loss_params['FEAT_DIM'])
    elif loss_name == 'FOCAL':
        return FocalLoss(num_classes=loss_params['GAMMA'])
    elif loss_name == 'SOFTMAX':
        return nn.CrossEntropyLoss()

def init_loss(loss_config: dict) -> Type[nn.Module]:
    """Initialize a loss function, either single loss or weihted Multiloss.
    Args:
        loss_config (dict): dictionary based on config file
    Returns:
        Type[nn.Module]: initialized loss class
    """
    assert len(loss_config.keys()) >= 1, f"Minimum one loss is required!"
    if len(loss_config.keys()) == 1:
        loss_name, loss_params = list(loss_config.items())[0]
        return init_single_loss(loss_name, loss_params)
    else:
        return WeightedMultiloss(loss_config)

class WeightedMultiloss(nn.Module):
    def __init__(self, loss_config: dict, **kwargs):
        super().__init__(WeightedMultiloss, **kwargs)
        
        self.losses = nn.ModuleDict()
        
        for loss_name, loss_params in loss_config.items():
            self.losses[loss_name] = (init_single_loss(loss_name, loss_params), loss_params.get('WEIGHT', 1.0))
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        loss = 0
        for loss_fn, weight in self.losses.values():
            loss += weight * loss_fn(x, y)
        return loss

class TripletMarginLoss(losses.TripletMarginLoss):
    def __init__(self, mining: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mining = mining
        self.miner = miners.TripletMarginMiner(margin=self.margin, type_of_triplets=self.mining)
        
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculate loss.
        Args:
            x (torch.Tensor): embeddings
            y (torch.Tensor): class labels
        Returns:
            torch.Tensor: loss
        """
        hard_pairs = self.miner(x, y)
        loss = super(x, y, hard_pairs)
        return loss
        
class CenterLoss(nn.Module):
    """Adopted from https://github.com/KaiyangZhou/pytorch-center-loss/blob/master/center_loss.py
    Reference: https://link.springer.com/chapter/10.1007/978-3-319-46478-7_31
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes: int = 10, feat_dim: int = 2, device: str = 'cuda'):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes 
        self.feat_dim = feat_dim
        self.device = device

        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))


    def forward(self, x: torch.tensor, y: torch.tensor) -> torch.tensor:
        """Compute loss.
        Args:
            x (torch.tensor): embeddings
            y (torch.tensor): class labels
        Returns:
            torch.tensor: loss
        """
        N = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(N, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, N).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long().to(self.device)
        y = y.unsqueeze(1).expand(N, self.num_classes)
        mask = y.eq(classes.expand(N, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / B

        return loss
    
class FocalLoss(nn.Module):
    """Adopted from: https://github.com/mathiaszinnen/focal_loss_torch/blob/main/focal_loss/focal_loss.py
    Reference: https://arxiv.org/abs/1708.02002v2
    Args:
        gamma (float):  The focal loss focusing parameter.
        weights (Union[None, Tensor]): Rescaling weight given to each class.
        If given, has to be a Tensor of size C. optional.
        reduction (str): Specifies the reduction to apply to the output.
        it should be one of the following 'none', 'mean', or 'sum'.
        default 'mean'.
        ignore_index (int): Specifies a target value that is ignored and
        does not contribute to the input gradient. optional.
        eps (float): smoothing to prevent log from returning inf.
    """
    def __init__(
            self,
            gamma,
            weights: Union[None, Tensor] = None,
            reduction: str = 'mean',
            ignore_index=-100,
            eps=1e-16
            ) -> None:
        super().__init__()
        if reduction not in ['mean', 'none', 'sum']:
            raise NotImplementedError(
                'Reduction {} not implemented.'.format(reduction)
                )
        assert weights is None or isinstance(weights, Tensor), \
            'weights should be of type Tensor or None, but {} given'.format(
                type(weights))
        self.reduction = reduction
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.eps = eps
        self.weights = weights

    def _get_weights(self, y: Tensor) -> Tensor:
        if self.weights is None:
            return torch.ones(y.shape[0])
        weights = y * self.weights
        return weights.sum(dim=-1)

    def _process_target(
            self, y: Tensor, num_classes: int
            ) -> Tensor:
        
        #convert all ignore_index elements to zero to avoid error in one_hot
        #note - the choice of value 0 is arbitrary, but it should not matter as these elements will be ignored in the loss calculation
        y = y * (y != self.ignore_index) 
        y = y.view(-1)
        return one_hot(y, num_classes=num_classes)

    def _process_preds(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            x = torch.vstack([1 - x, x])
            x = x.permute(1, 0)
            return x
        return x.view(-1, x.shape[-1])

    def _calc_pt(
            self, y: Tensor, x: Tensor, mask: Tensor
            ) -> Tensor:
        p = y * x
        p = p.sum(dim=-1)
        p = p * ~mask
        return p

    def forward(self, x: torch.tensor, y: torch.tensor) -> torch.tensor:
        """Compute loss.
        Args:
            x (torch.tensor): embeddings
            y (torch.tensor): class labels
        Returns:
            torch.tensor: loss
        """
        assert torch.all((x >= 0.0) & (x <= 1.0)), ValueError(
            'The predictions values should be between 0 and 1, \
                make sure to pass the values to sigmoid for binary \
                classification or softmax for multi-class classification'
        )
        mask = y == self.ignore_index
        mask = mask.view(-1)
        x = self._process_preds(x)
        num_classes = x.shape[-1]
        y = self._process_target(y, num_classes, mask)
        weights = self._get_weights(y).to(x.device)
        pt = self._calc_pt(y, x, mask)
        focal = 1 - pt
        nll = -torch.log(self.eps + pt)
        nll = nll.masked_fill(mask, 0)
        loss = weights * (focal ** self.gamma) * nll
        return self._reduce(loss, mask, weights)

    def _reduce(self, x: Tensor, mask: Tensor, weights: Tensor) -> Tensor:
        if self.reduction == 'mean':
            return x.sum() / (~mask * weights).sum()
        elif self.reduction == 'sum':
            return x.sum()
        else:
            return x