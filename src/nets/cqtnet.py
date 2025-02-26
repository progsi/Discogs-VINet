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
        
        self.ch_in = ch_in
        self.ch_out = ch_out
        
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

        self.loss_config = loss_config
        self.embed_dim = embed_dim
        self.l2_normalize = l2_normalize

        if norm.lower() == "bn":
            self.norm = nn.BatchNorm2d
        elif norm.lower() == "ibn":
            self.norm = IBN
        elif norm.lower() == "in":
            self.normm = nn.InstanceNorm2d
        else:
            raise NotImplementedError
        
        self.blocks = nn.ModuleList([
            # 1
            NetworkBlock(ch_in=1, 
                         ch_out=ch_in, 
                         kernel_size=(12, 3), 
                         dilation=(1, 1), 
                         padding=(6, 0), 
                         norm=self.norm),
            # 2
            NetworkBlock(ch_in=ch_in, 
                         ch_out=2 * ch_in, 
                         kernel_size=(13, 3), 
                         dilation=(1, 2), 
                         bias=False, 
                         norm=self.norm,
                         pool=nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1))),
            # 3
            NetworkBlock(ch_in=2 * ch_in,
                         ch_out=2 * ch_in,
                         kernel_size=(13, 3),
                         dilation=(1, 1),
                         bias=False,
                         norm=self.norm),
            # 4
            NetworkBlock(ch_in=2 * ch_in,
                         ch_out=2 * ch_in,
                         kernel_size=(3, 3),
                         dilation=(1, 2),
                         bias=False,
                         norm=self.norm,
                         pool=nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1))),
            # 5
            NetworkBlock(ch_in=2 * ch_in,
                         ch_out=4 * ch_in,
                         kernel_size=(3, 3),
                         dilation=(1, 1),
                         bias=False,
                         norm=self.norm),
            # 6
            NetworkBlock(ch_in=4 * ch_in,
                         ch_out=4 * ch_in,
                         kernel_size=(3, 3),
                         dilation=(1, 2),
                         norm=self.norm,
                         pool=nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1))),
            # 7
            NetworkBlock(ch_in=4 * ch_in,
                         ch_out=8 * ch_in,
                         kernel_size=(3, 3),
                         dilation=(1, 1),
                         bias=False,
                         norm=self.norm),
            # 8
            NetworkBlock(ch_in=8 * ch_in,
                         ch_out=8 * ch_in,
                         kernel_size=(3, 3),
                         dilation=(1, 2),
                         bias=False,
                         norm=self.norm,
                         pool=nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1))),
            # 9
            NetworkBlock(ch_in=8 * ch_in, 
                         ch_out=16 * ch_in,
                         kernel_size=(3, 3),
                         dilation=(1, 1),
                         bias=False,
                         norm=self.norm),
            # 10
            NetworkBlock(ch_in=16 * ch_in,
                         ch_out=16 * ch_in,
                         kernel_size=(3, 3),
                         dilation=(1, 2),
                         bias=False,
                         norm=self.norm)
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
    
    
class InductiveNeck(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 num_blocks: int, 
                 norm: torch.nn.Module,
                 pool: torch.nn.Module,
                 projection: str):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.projection = projection
        self.norm = norm
        self.pool = pool
        
        self.num_blocks = num_blocks
        self.blocks = self.init_blocks()
        
        self.neck = nn.Sequential(
            self.blocks,
            pool,
            nn.Flatten(),
            SimpleNeck(self.blocks[-1].ch_out, output_dim, projection)
        )
    
    def init_blocks(self):
        """Initialize blocks following the logic of CQTNet.
        Returns:
            torch.nn.Sequential: blocks
        """
        
        blocks = nn.ModuleList()
        
        ch_in = self.input_dim
        ch_out = self.input_dim
        pool = nn.Identity()

        for i in range(1, self.num_blocks + 1):
            
            if i % 2 == 0:
                ch_out = ch_out * 2
                dilation=(1, 2)  
                
                if i != self.num_blocks: # to avoid double pooling               
                    pool=nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1))
                    
            else:
                dilation=(1, 1)
                pool=nn.Identity()
                
            blocks.append(
                NetworkBlock(
                ch_in=ch_in,
                ch_out=ch_out,
                kernel_size=(3, 3),
                dilation=dilation,
                norm=self.norm,
                pool=pool
                )
            )
            
            ch_in = ch_out
            
        return nn.Sequential(*blocks)
        
    def forward(self, x):
        return self.neck(x)
    
       
class CQTNetMTL(CQTNet):
    """CQTNet with multi-task learning.
    Args:
        ch_in (int): 
        embed_dim (int): 
        norm (str, optional): . Defaults to "bn".
        pool (str, optional): . Defaults to "adaptive_max".
        l2_normalize (bool, optional): . Defaults to True.
        neck (str, optional): . Defaults to "linear".
        loss_config (Dict[str,Union[int,str]], optional): . Defaults to None.
        loss_config_inductive (Dict[str,Union[int,str]], optional): . Defaults to None.
    """
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
            # TODO: does this work though?
            block_output_dim = 16 * ch_in if config["AFTER_BLOCK"] == len(self.blocks) else self.blocks[after_block].ch_out
            self.necks[loss_name] = InductiveNeck(
                input_dim=block_output_dim,
                output_dim=config["OUTPUT_DIM"],
                num_blocks=config["NUM_BLOCKS"],
                norm=self.norm,
                pool=self.pool,
                projection=config["PROJECTION"])

    def forward(self, x):
        
        loss_dict = {}
        
        for i, block in enumerate(self.blocks):
            x = block(x)
            if self.training and i in self.mapping:
                for loss_name in self.mapping[i]:
                    out, _ = self.necks[loss_name](x)
                    loss_dict[loss_name] = out
        
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x, neck_dict = self.neck(x)

        # merge dicts
        if neck_dict is not None:
            loss_dict = loss_dict | neck_dict   
        else:
            loss_dict[self.out_key] = x

        # L2 normalization with 0 norm handling
        if self.l2_normalize:
            x = l2_normalize(x, precision="high")

        return x, loss_dict