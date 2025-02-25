import torch
import torch.nn as nn

from src.nets.pooling import Linear 

class SimpleNeck(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, projection: str = "linear"):
        super(SimpleNeck, self).__init__()
        
        if projection.lower() == "linear":
            self.layer =  Linear(input_dim, embed_dim, bias=False)
        elif projection.lower() == "affine":
            self.layer = Linear(input_dim, embed_dim)
        elif projection.lower() == "mlp":
            self.layer = nn.Sequential(
                Linear(input_dim, 2 * input_dim),
                nn.ReLU(inplace=True),
                Linear(2 * input_dim, embed_dim, bias=False),  # TODO bias=True?
            )
        elif projection.lower() == "none":
            self.layer = nn.Identity()
        else:
            raise NotImplementedError
        
    def forward(self, x):
        x = self.layer(x)
        return x, None

class BNNeck(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, loss_config: dict, bias: bool = False):
        """BNNeck module for LyraCNet.
        Args:
            input_dim (int): input dimension
            embed_dim (int): embedding dimensions
            output_dim (int): output dimension (typically, number of cliques)
            loss_config (dict): loss configuration
            bias (bool, optional): whether to use bias in the FC layers. Defaults to False.
        """
        super(BNNeck, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.loss_config = loss_config
        self.bias = bias
        
        self.INFERENCE = "INFERENCE"

        self.layers = self._init_layers()     
    
    def _init_layers(self) -> nn.ModuleDict:
        """Initialize layers based on loss config input and embedding dimension.
        Returns:
            nn.ModuleDict: dict. of layers
        """
        layers = nn.ModuleDict()
        
        input_dim = self.input_dim
        
        for i, (loss_name, config) in enumerate(self.loss_config.items()):
            
            assert not (config.get("SHARE_BN") and i == 0), "First loss cannot share output!"
            
            # if not shared
            if not config.get("SHARE_BN"): 
                layer = nn.Linear(input_dim, self.embed_dim, bias=self.bias)
                # if not last item
                if not i == len(self.loss_config) - 1:
                    layers[loss_name] = layer
                else:
                    layers[self.INFERENCE] = layer
                    layers[loss_name] = nn.Linear(self.embed_dim, config["OUTPUT_DIM"], bias=self.bias)   
            else:
                layers[loss_name] = nn.Identity()
                    
            input_dim = self.embed_dim
        return layers          
        
    def forward(self, x):
        outputs = {}
        emb = None
        for loss_name, layer in self.layers.items():
            
            x = layer(x)
            
            if loss_name != self.INFERENCE:
                outputs[loss_name] = x
            else:
                emb = x
        
        return emb, outputs