import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nets.pooling import GeM, IBN

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(out_planes) # replaced by IBN
        self.ibn = IBN(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        
        out = self.conv1(out if self.equalInOut else x)
        # out = self.bn2(out) # replaced by IBN
        out = self.ibn(out)
        out = self.relu2(out)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class LyraCNet(nn.Module):
    def __init__(self, depth, embed_dim, num_blocks, widen_factor, num_classes, dropRate=0.0):
        super(LyraCNet, self).__init__()
        
        self.num_blocks = num_blocks
        self.embed_dim = embed_dim
        
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
            stride = 1 if i == 0 else 2
            self.blocks.append(NetworkBlock(n, nChannels[i], nChannels[i + 1], block, stride, dropRate))
        
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[-1])
        self.relu = nn.ReLU(inplace=True)
        self.pooling = GeM()
        self.fc1 = nn.Linear(nChannels[-1], embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.fc3 = nn.Linear(embed_dim, num_classes)
        self.nChannels = nChannels[-1]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    
    def forward(self, x):
        # Input shape: [B, 1, F, T]
        x = self.conv1(x)
        for block in self.blocks:
            x = block(x)
        x = self.relu(self.bn1(x)) # this needed?
        # x = F.avg_pool2d(x, 8) replaced by GeM
        x = self.pooling(x)
        x = x.view(-1, self.nChannels)
        y1 = self.fc1(x) # for prototypical loss
        y2 = self.fc1(y1) # for triplet + center loss
        y3 = self.fc3(y2) # for classification loss
        return y1, y2, y3
        
        