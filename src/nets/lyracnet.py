from typing import Dict, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nets.pooling import GeM, IBN
from src.nets.neck import SimpleNeck, BNNeck

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.ibn = IBN(out_planes) # replacing bn2 by IBN
        
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.dropout = dropout
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes,
                                                                kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual
        
    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual is True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.conv1(out if self.equalInOut else x)
        out = self.ibn(out)
        out = self.relu2(out)
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)
        

class NetworkBlock(nn.Module):
    def __init__(self, num_layers, in_planes, out_planes, block, stride, dropout=0.0,
                 activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, num_layers, stride, dropout, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, num_layers, stride, dropout,
                    activate_before_residual):
        layers = []
        for i in range(int(num_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, dropout, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)
    
    
class LyraCNet(nn.Module):
    def __init__(self, 
                 depth: int, 
                 embed_dim: int, 
                 num_blocks: int, 
                 widen_factor: int, 
                 neck: str = "linear",
                 loss_config: Dict[str,Union[int,str]] = None,
                 dropout: float = 0.0, 
                 dense_dropout: float = 0.0
                 ):
        super(LyraCNet, self).__init__()
        
        assert not (neck == "bnneck" and loss_config is None), "BNNeck requires loss_config!"
        self.num_blocks = num_blocks
        self.embed_dim = embed_dim
        
        self.nChannels = [16]
        for i in range(num_blocks):
            self.nChannels.append(16 * widen_factor * (2 ** i))
        
        assert((depth - 4) % 6 == 0) # TODO: check why the constants?
        n = (depth - 4) // 6
        
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(1, self.nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        
        # Create network blocks dynamically
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            activate_before = i == 0 # only for first block
            # stride = 1 if i == 0 else 2
            self.blocks.append(NetworkBlock(n, self.nChannels[i], self.nChannels[i + 1], block, 2, dropout, activate_before))
        
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(self.nChannels[-1], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.drop = nn.Dropout(dense_dropout)        
        self.pooling = GeM() # flattening necessary?
        
        if neck != "bnneck":
            self.neck = SimpleNeck(self.embed_dim, embed_dim, neck)
        else:
            self.neck = BNNeck(self.embed_dim, embed_dim, loss_config)
        
        # NOTE: removed  
        # self.nChannels = self.nChannels[-1]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None: # necessary, due to non-bias FC layers
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        # Input shape: [B, 1, F, T]
        x = self.conv1(x)
        
        for block in self.blocks:
            x = block(x)
        x = self.bn1(x)
        x = self.relu(x) 
        x = self.pooling(x)
        x = x.view(-1, self.nChannels[-1])
        
        out_tensor, out_dict = self.neck(x)
        
        return out_tensor, out_dict

    
class InductiveNeck(nn.Module):
    def __init__(self, 
                 channels: List[int], 
                 output_dim: int, 
                 num_blocks: int, 
                 pool: torch.nn.Module,
                 projection: str,
                 dropout: float = 0.0):
        super().__init__()
        
        self.nChannels = channels
        self.output_dim = output_dim
        self.projection = projection
        self.pool = pool
        self.dropout = dropout
        self.num_blocks = num_blocks
        self.blocks = self.init_blocks()
        
        self.neck = nn.Sequential(
            self.blocks,
            pool,
            nn.Flatten(),
            SimpleNeck(self.blocks[-1].layer[-1].conv2.out_channels, output_dim, projection)
        )
    
    def init_blocks(self):
        """Initialize blocks following the logic of CQTNet.
        Returns:
            torch.nn.Sequential: blocks
        """
        
        blocks = nn.ModuleList()

        for i in range(1, self.num_blocks + 1):
            
            # TODO: put these blocks appropriately
            blocks.append(NetworkBlock(4, self.nChannels[i], self.nChannels[i] * 2, BasicBlock, 2, self.dropout, False))
            
        return nn.Sequential(*blocks)
        
    def forward(self, x):
        return self.neck(x)
    
    
class LyraCNetMTL(LyraCNet):
    def __init__(self, 
                depth: int, 
                embed_dim: int, 
                num_blocks: int, 
                widen_factor: int, 
                neck: str = "linear",
                loss_config: Dict[str,Union[int,str]] = None,
                dropout: float = 0.0, 
                dense_dropout: float = 0.0,
                loss_config_inductive: Dict[str,Union[int,str]] = None,
                ):
        super().__init__(depth, embed_dim, num_blocks, widen_factor, neck, loss_config, 
                        dropout, dense_dropout)
            
        self.loss_config = loss_config
        self.loss_config_inductive = loss_config_inductive
        
        if len(self.loss_config) > 1:
            assert neck == "bnneck", "BNNeck required for multi-loss!"
            self.out_key = None # key for the output loss
        else:
            self.out_key = list(self.loss_config.keys())[0]

        self.mapping = {}
        self.necks = nn.ModuleDict()
        for loss_name, config in self.loss_config_inductive.items():
            assert loss_name not in loss_config, "Inductive loss cannot be in the same config!"
            
            after_block = config["AFTER_BLOCK"]
            if not after_block in self.mapping:
                self.mapping[after_block] = [loss_name]
            else:
                self.mapping[after_block].append(loss_name)

            self.necks[loss_name] = InductiveNeck(
                channels=self.nChannels[after_block:],
                output_dim=config["OUTPUT_DIM"],
                num_blocks=config["NUM_BLOCKS"],
                pool=GeM(),
                projection=config["PROJECTION"])
                
    def forward(self, x):
        
        mtl_dict = {}
        
        x = self.conv1(x)
        
        for i, block in enumerate(self.blocks):
            x = block(x)
            if self.training and i in self.mapping:
                for loss_name in self.mapping[i]:
                    out, _ = self.necks[loss_name](x)
                    mtl_dict[loss_name] = out

        x = self.bn1(x)
        x = self.relu(x) 
        x = self.pooling(x)
        x = x.view(-1, self.nChannels[-1])
        
        out_tensor, out_dict = self.neck(x)
        out_dict = out_dict | mtl_dict 
        
        return out_tensor, out_dict