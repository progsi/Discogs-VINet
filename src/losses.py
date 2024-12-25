from typing import List, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses, miners, distances

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
    if loss_name == 'TRIPLET':
        return TripletMarginLoss(margin=loss_params['MARGIN'], mining=loss_params['MINING'])
    elif loss_name == 'CENTER':
        return CenterLoss(num_cls=loss_params['NUM_CLS'], feat_dim=loss_params['FEAT_DIM'])
    elif loss_name == 'FOCAL':
        return FocalLoss(gamma=loss_params['GAMMA'])
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
    def __init__(self, loss_config: dict, **kwargs):
        super(WeightedMultiloss, self).__init__()
        
        self.losses_with_weights = []  # Store (loss_name, loss_fn, weight, is_cls_loss) tuples
        self.loss_stats = {} # to store current loss stats

        for loss_name, loss_params in loss_config.items():
            loss_fn = init_single_loss(loss_name, loss_params)
            weight = loss_params.get('WEIGHT', 1.0)
            is_cls = is_cls_loss(loss_name)
            self.losses_with_weights.append((loss_name, loss_fn, weight, is_cls))
            self.loss_stats[loss_name] = {}
        
    def forward(self, x_emb: torch.Tensor, y_emb: torch.Tensor, 
                x_cls: torch.Tensor, y_cls) -> torch.Tensor:
        """Calculates the Multiloss and returns individual losses.
        Args:
            x_emb (torch.Tensor): embeddings
            y_emb (torch.Tensor): class labels per embedding
            x_cls (torch.Tensor): output after BNN
            y_cls (torch.Tensor): softmax of all classes specified
        Returns:
            tuple: Total loss and a dictionary of individual losses.
        """
        total_loss = 0
        
        for loss_name, loss_fn, weight, is_cls in self.losses_with_weights:
            x = x_cls if is_cls else x_emb
            y = y_cls if is_cls else y_emb
            loss = loss_fn(x, y)
            loss_weighted = loss * weight
            self.loss_stats[loss_name]["unweighted"] = loss.detach().item()
            self.loss_stats[loss_name]["weighted"] = loss_weighted.detach().item()
            total_loss += loss_weighted   
        return total_loss
    
    def get_stats(self):
        return self.loss_stats

class TripletMarginLoss(nn.Module):
    def __init__(self, margin: float, mining: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mining = mining
        self.margin = margin
        self.loss = losses.TripletMarginLoss(margin=self.margin)
        self.miner = miners.TripletMarginMiner(margin=self.margin, type_of_triplets=self.mining)
        
    def forward(self, x_emb: torch.Tensor, y_cls: torch.Tensor) -> torch.Tensor:
        """Calculate loss.
        Args:
            x_emb (torch.Tensor): embeddings
            y_cls (torch.Tensor): class labels
        Returns:
            torch.Tensor: loss
        """
        hard_pairs = self.miner(x_emb, y_cls)
        loss = self.loss(embeddings=x_emb, labels=y_cls, indices_tuple=hard_pairs)
        return loss
    
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

    def forward(self, x_emb: torch.tensor, y_cls: torch.tensor) -> torch.tensor:
        """Compute loss.
        Args:
            x_emb (torch.tensor): embeddings
            y_cls (torch.tensor): class labels
        Returns:
            torch.tensor: loss
        """
        N = x_emb.size(0)
        distmat = torch.pow(x_emb, 2).sum(dim=1, keepdim=True).expand(N, self.num_cls) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_cls, N).t()
        distmat.to(x_emb.dtype).addmm_(x_emb, self.centers.to(x_emb.dtype).t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_cls).long().to(self.device)
        y_cls = y_cls.unsqueeze(1).expand(N, self.num_cls)
        mask = y_cls.eq(classes.expand(N, self.num_cls))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / N

        return loss

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
    ce = ce.gather(1, y_true.view(1, -1))

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
            

