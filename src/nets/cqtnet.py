from typing import Union, Dict, Tuple
import torch
import torch.nn as nn

from src.utilities.tensor_op import l2_normalize
from src.nets.pooling import IBN, GeM, SoftPool
from src.nets.neck import SimpleNeck, BNNeck

                
class NetworkBlock(nn.Module):
    def __init__(self, 
                 ch_in: int,
                 ch_out: int, 
                 kernel_size: Tuple[int,int], 
                 dilation: Tuple[int,int], 
                 norm: Union[nn.Module,IBN],
                 padding: Tuple[int,int] = 0, 
                 pool: nn.Module = None,
                 bias: bool = False):
        super(NetworkBlock, self).__init__()
        
        layers = [
            nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            bias=bias,
            ),
            norm(ch_out),
            nn.ReLU(inplace=True)
        ]
        
        if pool is not None:
            layers.append(pool)
        
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)
    
class CQTNet(nn.Module):
    def __init__(
        self,
        ch_in: int,
        embed_dim: int,
        norm: str = "bn",
        pool: str = "adaptive_max",
        l2_normalize: bool = True,
        neck: str = "linear",
        loss_config: Dict[str,Union[int,str]] = None,
    ):
        super().__init__()

        assert ch_in > 0
        assert embed_dim > 0
        assert not (neck == "bnneck" and loss_config is None), "BNNeck requires loss_config!"

        self.embed_dim = embed_dim
        self.l2_normalize = l2_normalize

        if norm.lower() == "bn":
            norm = nn.BatchNorm2d
        elif norm.lower() == "ibn":
            norm = IBN
        elif norm.lower() == "in":
            norm = nn.InstanceNorm2d
        else:
            raise NotImplementedError
        
        self.blocks = nn.ModuleList([
            # 1
            NetworkBlock(ch_in=1, 
                         ch_out=ch_in, 
                         kernel_size=(12, 3), 
                         dilation=(1, 1), 
                         padding=(6, 0), 
                         norm=norm),
            # 2
            NetworkBlock(ch_in=ch_in, 
                         ch_out=2 * ch_in, 
                         kernel_size=(13, 3), 
                         dilation=(1, 2), 
                         bias=False, 
                         norm=norm,
                         pool=nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1))),
            # 3
            NetworkBlock(ch_in=2 * ch_in,
                         ch_out=2 * ch_in,
                         kernel_size=(13, 3),
                         dilation=(1, 1),
                         bias=False,
                         norm=norm),
            # 4
            NetworkBlock(ch_in=2 * ch_in,
                         ch_out=2 * ch_in,
                         kernel_size=(3, 3),
                         dilation=(1, 2),
                         bias=False,
                         norm=norm,
                         pool=nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1))),
            # 5
            NetworkBlock(ch_in=2 * ch_in,
                         ch_out=4 * ch_in,
                         kernel_size=(3, 3),
                         dilation=(1, 1),
                         bias=False,
                         norm=norm),
            # 6
            NetworkBlock(ch_in=4 * ch_in,
                         ch_out=4 * ch_in,
                         kernel_size=(3, 3),
                         dilation=(1, 2),
                         norm=norm,
                         pool=nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1))),
            # 7
            NetworkBlock(ch_in=4 * ch_in,
                         ch_out=8 * ch_in,
                         kernel_size=(3, 3),
                         dilation=(1, 1),
                         bias=False,
                         norm=norm),
            # 8
            NetworkBlock(ch_in=8 * ch_in,
                         ch_out=8 * ch_in,
                         kernel_size=(3, 3),
                         dilation=(1, 2),
                         bias=False,
                         norm=norm,
                         pool=nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1))),
            # 9
            NetworkBlock(ch_in=8 * ch_in, 
                         ch_out=16 * ch_in,
                         kernel_size=(3, 3),
                         dilation=(1, 1),
                         bias=False,
                         norm=norm),
            # 10
            NetworkBlock(ch_in=16 * ch_in,
                         ch_out=16 * ch_in,
                         kernel_size=(3, 3),
                         dilation=(1, 2),
                         bias=False,
                         norm=norm)
            ])

        if pool.lower() == "gem":
            self.pool = GeM()
        elif pool.lower() == "adaptive_max":
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        elif pool.lower() == "softpool":
            self.pool = SoftPool(16 * ch_in)
        else:
            raise NotImplementedError

        if neck != "bnneck":
            self.neck = SimpleNeck(16 * ch_in, embed_dim, neck)
        else:
            self.neck = BNNeck(16 * ch_in, embed_dim, loss_config)
        # TODO add batch norm here (If BNNeck)

    def forward(self, x):
        
        for i, block in enumerate(self.blocks):
            x = block(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x, loss_dict = self.neck(x)

        # L2 normalization with 0 norm handling
        if self.l2_normalize:
            x = l2_normalize(x, precision="high")

        return x, loss_dict
    
    
class CQTNetMTL(CQTNet):
    def __init__(
        self,
        ch_in: int,
        embed_dim: int,
        norm: str = "bn",
        pool: str = "adaptive_max",
        l2_normalize: bool = True,
        neck: str = "linear",
        loss_config: Dict[str,Union[int,str]] = None,
        loss_config_inductive: Dict[str,Union[int,str]] = None,
    ):
        super().__init__(ch_in, embed_dim, norm, pool, l2_normalize, neck, loss_config)
        self.loss_config_inductive = loss_config_inductive
        
        self.mapping_heads = {}
        self.inductive_heads = nn.ModuleDict()
        for loss_name, config in self.loss_config_inductive.items():
            assert loss_name not in self.loss_config, "Inductive loss cannot be in the same config!"
            
            after_block = config["after_block"]
            if not after_block in self.mapping_heads:
                self.mapping_heads[after_block] = [loss_name]
            else:
                self.mapping_heads[after_block].append(loss_name)
            # TODO: does this work though?
            block_output_dim = 16 * ch_in if config["after_block"] == len(self.blocks) else self.blocks[after_block].layer[-1].num_features
            self.inductive_heads[loss_name] = nn.Sequential(
                nn.AdaptiveMaxPool2d((1, 1)),
                SimpleNeck(input_dim=block_output_dim, 
                           embed_dim=config["output_dim"], 
                           projection=config["projection"])
            )

    def forward(self, x):
        
        loss_dict = {}
        for i, block in enumerate(self.blocks):
            if i in self.mapping_heads:
                for loss_name in self.mapping_heads[i]:
                    out, _ = self.inductive_heads[loss_name](x)
                    loss_dict[loss_name] = out
            x = block(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x, neck_dict = self.neck(x)

        loss_dict.update(neck_dict)
        # L2 normalization with 0 norm handling
        if self.l2_normalize:
            x = l2_normalize(x, precision="high")

        return x, loss_dict