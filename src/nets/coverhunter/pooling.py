import torch
import torch.nn.functional as F
from typing import Optional
import logging
from src.nets.coverhunter.layers import Conv1d, Linear


class AttentiveStatisticsPooling(torch.nn.Module):
  """This class implements an attentive statistic pooling layer for each channel.
  It returns the concatenated mean and std of the input tensor.

  Arguments
  ---------
  channels: int
      The number of input channels.
  output_channels: int
      The number of output channels.
  """

  def __init__(self, channels, output_channels):
    super().__init__()

    self.eps = 1e-12
    self.linear = Linear(channels * 3, channels)
    self.tanh = torch.nn.Tanh()
    self.conv = Conv1d(
      in_channels=channels, out_channels=channels, kernel_size=1
    )
    self.final_layer = torch.nn.Linear(channels * 2, output_channels,
                                        bias=False)
    logging.info("Init AttentiveStatisticsPooling with {}->{}".format(
      channels, output_channels))
    return

  @staticmethod
  def _compute_statistics(x: torch.Tensor,
                          m: torch.Tensor,
                          eps: float,
                          dim: int = 2):
    mean = (m * x).sum(dim)
    std = torch.sqrt((m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(eps))
    return mean, std

  def forward(self, x: torch.Tensor):
    """Calculates mean and std for a batch (input tensor).

    Args:
      x : torch.Tensor
          Tensor of shape [N, L, C].
    """

    x = x.transpose(1, 2)
    L = x.shape[-1]
    lengths = torch.ones(x.shape[0], device=x.device)
    mask = self.length_to_mask(lengths * L, max_len=L, device=x.device)
    mask = mask.unsqueeze(1)
    total = mask.sum(dim=2, keepdim=True).float()

    mean, std = self._compute_statistics(x, mask / total, self.eps)
    mean = mean.unsqueeze(2).repeat(1, 1, L)
    std = std.unsqueeze(2).repeat(1, 1, L)
    attn = torch.cat([x, mean, std], dim=1)
    attn = self.conv(self.tanh(self.linear(
      attn.transpose(1, 2)).transpose(1, 2)))

    attn = attn.masked_fill(mask == 0, float("-inf"))  # Filter out zero-padding
    attn = F.softmax(attn, dim=2)
    mean, std = self._compute_statistics(x, attn, self.eps)
    pooled_stats = self.final_layer(torch.cat((mean, std), dim=1))
    return pooled_stats

  def forward_with_mask(self, x: torch.Tensor,
                        lengths: Optional[torch.Tensor] = None):
    """Calculates mean and std for a batch (input tensor).

    Args:
      x : torch.Tensor
          Tensor of shape [N, C, L].
      lengths:
    """
    L = x.shape[-1]

    if lengths is None:
      lengths = torch.ones(x.shape[0], device=x.device)

    # Make binary mask of shape [N, 1, L]
    mask = self.length_to_mask(lengths * L, max_len=L, device=x.device)
    mask = mask.unsqueeze(1)

    # Expand the temporal context of the pooling layer by allowing the
    # self-attention to look at global properties of the utterance.

    # torch.std is unstable for backward computation
    # https://github.com/pytorch/pytorch/issues/4320
    total = mask.sum(dim=2, keepdim=True).float()
    mean, std = self._compute_statistics(x, mask / total, self.eps)

    mean = mean.unsqueeze(2).repeat(1, 1, L)
    std = std.unsqueeze(2).repeat(1, 1, L)
    attn = torch.cat([x, mean, std], dim=1)

    # Apply layers
    attn = self.conv(self.tanh(self.linear(attn, lengths)))

    # Filter out zero-paddings
    attn = attn.masked_fill(mask == 0, float("-inf"))

    attn = F.softmax(attn, dim=2)
    mean, std = self._compute_statistics(x, attn, self.eps)
    # Append mean and std of the batch
    pooled_stats = torch.cat((mean, std), dim=1)
    pooled_stats = pooled_stats.unsqueeze(2)
    return pooled_stats

  @staticmethod
  def length_to_mask(length: torch.Tensor,
                     max_len: Optional[int] = None,
                     dtype: Optional[torch.dtype] = None,
                     device: Optional[torch.device] = None):
    """Creates a binary mask for each sequence.

    Arguments
    ---------
    length : torch.LongTensor
        Containing the length of each sequence in the batch. Must be 1D.
    max_len : int
        Max length for the mask, also the size of the second dimension.
    dtype : torch.dtype, default: None
        The dtype of the generated mask.
    device: torch.device, default: None
        The device to put the mask variable.

    Returns
    -------
    mask : tensor
        The binary mask.

    Example
    -------
    """
    assert len(length.shape) == 1

    if max_len is None:
      max_len = length.max().long().item()  # using arange to generate mask
    mask = torch.arange(
      max_len, device=length.device, dtype=length.dtype
    ).expand(len(length), max_len) < length.unsqueeze(1)

    if dtype is None:
      dtype = length.dtype

    if device is None:
      device = length.device

    mask = torch.as_tensor(mask, dtype=dtype, device=device)
    return mask
