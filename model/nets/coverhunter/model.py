from typing import Tuple
import torch

from model.nets.coverhunter.conformer import ConformerEncoder
from model.nets.coverhunter.pooling import AttentiveStatisticsPooling

# CoverHunter parameters
INPUT_DIM = 84 # instead of 96 in CoverHunter paper
EMBED_DIM = 128

# Conformer parameters
ATTENTION_DIM = 256
NUM_BLOCKS = 6
OUTPUT_DIM = 128

# Bottleneck
OUTPUT_CES = 30_000

class Encoder(torch.nn.Module):
  """Encoding part of CoverHunter, returning a variable length processed CQT feature.
  """

  def __init__(self, input_dim: int = INPUT_DIM, output_dim: int = OUTPUT_DIM, 
               attention_dim: int = ATTENTION_DIM, num_blocks: int = NUM_BLOCKS):
    super(Encoder, self).__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.attention_dim = attention_dim
    self.num_blocks = num_blocks

    self.global_cmvn = torch.nn.BatchNorm1d(self.input_dim)
    self.encoder = ConformerEncoder(
      input_size=self.input_dim,
      output_size=self.output_dim,
      linear_units=self.attention_dim,
      num_blocks=self.num_blocks)

    return

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x.squeeze()
    x = self.global_cmvn(x).transpose(1, 2)
    xs_lens = torch.full(
      [x.size(0)], 
      fill_value=x.size(1), 
      dtype=torch.long).to(x.device).long()
    x, _ = self.encoder(x, xs_lens=xs_lens, decoding_chunk_size=-1)
    return x

   
class Model(torch.nn.Module):
  def __init__(self, input_dim: int = INPUT_DIM, embed_dim: int = EMBED_DIM, 
               output_dim: int = OUTPUT_DIM, attention_dim: int = ATTENTION_DIM,
               num_blocks: int = NUM_BLOCKS,  output_cls: int = OUTPUT_CES):
    super(Model, self).__init__()
    self.input_dim = input_dim
    self.embed_dim = embed_dim
    self.output_cls = output_cls
    self.ouput_dim = output_dim
    self.attention_dim = attention_dim
    self.num_blocks = num_blocks
    
    self.encoder = Encoder(input_dim=self.input_dim, output_dim=self.ouput_dim,
                           attention_dim=self.attention_dim, num_blocks=self.num_blocks)

    self.pool_layer = AttentiveStatisticsPooling(
      self.embed_dim, output_channels=self.embed_dim)
    self.ce_layer = torch.nn.Linear(
      self.embed_dim, self.output_cls, bias=False)

    return

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.encoder(x)
    x = self.pool_layer(x)
    return x

  @torch.jit.ignore
  def inference(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
      embed = self.forward(feat)
      embed_ce = self.ce_layer(embed)
    return embed, embed_ce


