
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import math
from typing import Optional, List, Union
from abc import ABC
import math
from torchvision.models.vision_transformer import ViT_B_16_Weights, ViT_B_32_Weights
import os


from .blocks import SelfAttention, MLP
from .vit import ViTBlock
from einops import reduce
 
"""
Vision Transformer that ranks tokens in each layer of the encoder.
"""


# Rank ViT Block
class RankViTBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        
        
        # Attention block
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.self_attention = SelfAttention(hidden_dim, num_heads, attention_dropout)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim=hidden_dim, mlp_dim=mlp_dim)

        # Sort tokens
        self.sort = False
        self.current_budget = 1.0
    
  
    def sort_and_drop(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        
        class_token = input[:, 0:1, :]
        input = input[:, 1:, :]

        # get token magnitudes
        # (batch_size, seq_length, hidden_dim) -> (batch_size, seq_length)
        token_magnitudes = torch.norm(input, dim=-1)
        
        # get sorted indices 
        # (batch_size, seq_length) -> (batch_size, seq_length, 1)
        sorted_indices = torch.argsort(token_magnitudes, dim=-1, descending=True)
        sorted_indices = sorted_indices.unsqueeze(-1)

        # gather sorted input
        sorted_input = torch.gather(input, dim=1, index=sorted_indices.expand(-1, -1, input.shape[-1]))

        # discard tokens
        num_tokens_to_keep = math.ceil(input.shape[1] * self.current_budget)
        remaining_tokens = sorted_input[:, :num_tokens_to_keep, :]

        return torch.cat([class_token, remaining_tokens], dim=1)
    

    
    
    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")

        if self.current_budget != 1:
            # this is not optimal bcs norms probably do not change much
            # sorting them at every block might be a waste of resources
            input = self.sort_and_drop(input)

        x = self.ln_1(input)  
        x = self.self_attention(x)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y

    
    def set_budget(self, budget: float):
        self.current_budget = budget


# ViT Encoder
class RankViTEncoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        rankvit_layers: Optional[List[Union[int, float]]] = None
        ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: List = []
        for i in range(num_layers):
            if i in rankvit_layers:
                layers.append(RankViTBlock(
                            num_heads,
                            hidden_dim,
                            mlp_dim,
                            dropout,
                            attention_dropout,
                            ))
            else: 
                layers.append(ViTBlock(
                            num_heads,
                            hidden_dim,
                            mlp_dim,
                            dropout,
                            attention_dropout,
                            ))
            

        self.layers = nn.Sequential(*layers)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        input = self.dropout(input)
        input = self.layers(input)
        return self.ln(input)



class RankVisionTransformer(nn.Module):

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        num_registers: int = 0,
        num_class_tokens: int = 1,
        torch_pretrained_weights: Optional[str] = None,
        timm_pretrained_weights: Optional[str] = None,
        rankvit_layers: Optional[List[Union[int, float]]] = None
    ):
        super().__init__()
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.num_heads = num_heads
        self.num_registers = num_registers
        self.num_class_tokens = num_class_tokens
        self.rankvit_layers = rankvit_layers



        self.conv_proj = nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size)

        seq_length = (image_size // patch_size) ** 2

        # Add class tokens
        self.class_tokens = nn.Parameter(torch.zeros(1, num_class_tokens, hidden_dim))
        seq_length += num_class_tokens

        # Add registers
        if num_registers > 0:
            raise ValueError("Registers are not supported yet for this model.")
            self.register_tokens = nn.Parameter(torch.zeros(1, num_registers, hidden_dim))
            seq_length += num_registers

        self.encoder = RankViTEncoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout, 
            rankvit_layers
            )


        self.seq_length = seq_length

        self.head = nn.Linear(hidden_dim, num_classes)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)


        fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
        nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
        if self.conv_proj.bias is not None:
            nn.init.zeros_(self.conv_proj.bias)
        

        self.load_weights(torch_pretrained_weights, timm_pretrained_weights)


    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Add registers
        if self.num_registers > 0:
            batch_register_tokens = self.register_tokens.expand(n, -1, -1)
            x = torch.cat([batch_register_tokens, x], dim=1)
        
        # Expand the class token to the full batch
        batch_class_tokens = self.class_tokens.expand(n, -1, -1)
        x = torch.cat([batch_class_tokens, x], dim=1)

        # Pass through the encoder
        x = self.encoder(x)

        # Get all class tokens and average them
        x = x[:, 0:self.num_class_tokens]
        x = reduce(x, 'n c e -> n e', reduction='sum')

        # Classification head
        x = self.head(x)

        return x
    

    def set_budget(self, budget: float):
        self.current_budget = budget
        #for i, vitblock in enumerate(self.encoder.layers):
            #vitblock.set_budget(budget[i] if isinstance (budget, list) else budget)
        for i in self.rankvit_layers:
            self.encoder.layers[i].set_budget(budget[i] if isinstance (budget, list) else budget)


    def load_weights(
        self, 
        torch_pretrained_weights: Optional[str] = None, 
        timm_pretrained_weights: Optional[List] = None):
        """
        Loads pretrained weights into the model.

        Args:
            torch_pretrained_weights (str, optional): Path to the torch pretrained weights file or a URL to download from. Defaults to None.
            timm_pretrained_weights (List, optional): List containing the name of the timm model and the variant to load pretrained weights from. Defaults to None.
        
        Example:
            torch_pretrained_weights = 'ViT_B_16_Weights[IMAGENET1K_V1]'
            timm_pretrained_weights = ['facebookresearch/deit_base_patch16_224', 'deit_base_patch16_224']
            torch_pretrained_weights = 'path/to/torchweights.pth'
            timm_pretrained_weights = 'path/to/timmweights.pth'
        """
        
        # they cannot be both not None
        assert not (torch_pretrained_weights and timm_pretrained_weights), "You cannot load weights from both torch and timm at the same time."
        
        
        if torch_pretrained_weights is not None:
            print('Loading torch pretrained weights: ', torch_pretrained_weights)
            from .adapters import adapt_torch_state_dict
            if not os.path.exists(str(torch_pretrained_weights)):
                print('Downloading torch pretrained weights: ', torch_pretrained_weights)
                torch_pretrained_weights = eval(torch_pretrained_weights).get_state_dict(progress=False)
                adapted_state_dict = adapt_torch_state_dict(torch_pretrained_weights, num_classes=self.num_classes)
            else:
                torch_pretrained_weights = torch.load(torch_pretrained_weights)
                print(f'Loaded torch pretrained weights with these keys {list(torch_pretrained_weights.keys())}. I assume the model weights are in the the "model" key.')
                torch_pretrained_weights = torch_pretrained_weights['model']
                adapted_state_dict = adapt_torch_state_dict(torch_pretrained_weights, num_classes=self.num_classes)
            self.load_state_dict(adapted_state_dict, strict=False)
        elif timm_pretrained_weights is not None:
            print('Loading timm pretrained weights: ', timm_pretrained_weights)
            from .adapters import adapt_timm_state_dict
            if not os.path.exists(str(timm_pretrained_weights)):
                print('Downloading timm pretrained weights: ', timm_pretrained_weights)
                model = torch.hub.load(timm_pretrained_weights[0], timm_pretrained_weights[1], pretrained=True)
                timm_pretrained_weights = model.state_dict()
                del model
            else:
                timm_pretrained_weights = torch.load(timm_pretrained_weights)
                print(f'Loaded timm pretrained weights with these keys {list(timm_pretrained_weights.keys())}. I assume the model weights are in the the "model" key.')
                timm_pretrained_weights = timm_pretrained_weights['model']
            adapted_state_dict = adapt_timm_state_dict(timm_pretrained_weights, num_classes=self.num_classes)
            self.load_state_dict(adapted_state_dict, strict=False)
