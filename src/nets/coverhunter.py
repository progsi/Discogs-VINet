from typing import Tuple
import torch

from src.utilities.tensor_op import l2_normalize
from src.nets.conformer import ConformerEncoder
from src.nets.pooling import AttentiveStatisticsPooling

# CoverHunter parameters
INPUT_DIM = 84 # instead of 96 in CoverHunter paper
EMBED_DIM = 128

# Conformer parameters
ATTENTION_DIM = 256
NUM_BLOCKS = 6
OUTPUT_DIM = 128


class Encoder(torch.nn.Module):
  """Encoding part of CoverHunter, returning a variable length processed CQT feature.
  """
  def __init__(
    self, 
    input_dim: int, 
    output_dim: int, 
    attention_dim: int, 
    num_blocks: int
    ):
    super(Encoder, self).__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.attention_dim = attention_dim
    self.num_blocks = num_blocks
    
    self.global_cmvn = torch.nn.BatchNorm1d(self.input_dim)
    
    self.blocks = ConformerEncoder(
              input_dim=self.input_dim,
              encoder_dim=self.output_dim,
              num_layers=self.num_blocks,
              num_attention_heads=4, # TODO: to config
              feed_forward_expansion_factor=2, # TODO: to config
              conv_kernel_size=15 # TODO: to config
              )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # B x 1 x F x T --> B x T x F 
    x = x.squeeze(1)

    x = self.global_cmvn(x).transpose(1, 2) # TODO: need to remove at inference?

    # xs_lens: B
    x_lens = torch.full(
      [x.size(0)], 
      fill_value=x.size(1), 
      dtype=torch.long).to(x.device).long()
    x, _ = self.blocks(x, x_lens)
    # Out: B x T/4 x F
    return x

   
class Model(torch.nn.Module):
  """CoverHunter model. Includes an encoder, pooling layer, bottleneck and classifier. 
  """
  def __init__(
      self, 
      input_dim: int = INPUT_DIM, 
      embed_dim: int = EMBED_DIM, 
      output_dim: int = OUTPUT_DIM, 
      attention_dim: int = ATTENTION_DIM,
      num_blocks: int = NUM_BLOCKS,  
      output_cls: int = None,
      l2_normalize: bool = True):
    super(Model, self).__init__()
    self.input_dim = input_dim
    self.embed_dim = embed_dim
    self.output_cls = output_cls
    self.ouput_dim = output_dim
    self.attention_dim = attention_dim
    self.num_blocks = num_blocks
    
    self.l2_normalize = l2_normalize
    
    self.encoder = Encoder(input_dim=self.input_dim, output_dim=self.ouput_dim,
                           attention_dim=self.attention_dim, num_blocks=self.num_blocks)

    self.pool_layer = AttentiveStatisticsPooling(
      self.embed_dim, output_channels=self.embed_dim)
    
    if self.output_cls is not None:
      # Define bottleneck layer
      self.bottleneck = torch.nn.BatchNorm1d(self.embed_dim)
      self.bottleneck.bias.requires_grad_(False)

      # Define cls_layer, applied after the bottleneck
      self.cls_layer = torch.nn.Sequential(
        self.bottleneck,  # Include bottleneck as part of the pipeline
        torch.nn.Linear(self.embed_dim, self.output_cls, bias=False)
      )
    else:
      self.cls_layer = None

  def embed(self, x: torch.Tensor) -> torch.Tensor:
    x = self.encoder(x)
    x = self.pool_layer(x)
    return x
  
  def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    x = self.embed(x) # compute embedding
    if self.cls_layer is not None:
      y = self.cls_layer(x) # compute cls output
    else:
      y = None
    
    if self.l2_normalize:
      x = l2_normalize(x)
    return x, y