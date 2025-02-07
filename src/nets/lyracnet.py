from typing import Dict, Union

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
                 dense_dropout: float = 0.0,
                 indictive_heads: Dict[str,int] = None):
        super(LyraCNet, self).__init__()
        
        assert not (neck == "bnneck" and loss_config is None), "BNNeck requires loss_config!"
        self.num_blocks = num_blocks
        self.embed_dim = embed_dim
        self.indictive_heads = indictive_heads is not None
        
        nChannels = [16]
        for i in range(num_blocks):
            nChannels.append(16 * widen_factor * (2 ** i))
        
        assert((depth - 4) % 6 == 0) # TODO: check why the constants?
        n = (depth - 4) // 6
        
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(1, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        
        # Create network blocks dynamically
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            activate_before = i == 0 # only for first block
            # stride = 1 if i == 0 else 2
            self.blocks.append(NetworkBlock(n, nChannels[i], nChannels[i + 1], block, 2, dropout, activate_before))
        
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[-1], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.drop = nn.Dropout(dense_dropout)        
        self.pooling = GeM() # flattening necessary?
        
        if self.indictive_heads:
            self.inductive_heads = nn.ModuleDict()
            for label, num_classes in indictive_heads.items():
                self.inductive_heads[label] = nn.Linear(nChannels[-1], num_classes)
        
        if neck != "bnneck":
            self.neck = SimpleNeck(self.embed_dim, embed_dim, neck)
        else:
            self.neck = BNNeck(self.embed_dim, embed_dim, loss_config)
                    
        self.nChannels = nChannels[-1]

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
        x = x.view(-1, self.nChannels)
        
        out = self.neck(x)
        
        if self.indictive_heads:
            for label, head in self.inductive_heads.items():
                out[label] = head(x)
        
        return out


