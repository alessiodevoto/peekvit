import os
import torch
from torch import nn
import torch.nn.functional as F
import math
from typing import Optional, List
from abc import ABC
from torchvision.models.vision_transformer import ViT_B_16_Weights, ViT_B_32_Weights
import random

from .blocks import SelfAttention, MLP
 
"""
Token Dropping Vision Transformer
"""


# ViT Block
class DropViTBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        max_token_drop: float = 1.0,
        enable_token_drop: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.max_token_drop = max_token_drop
        self.enable_token_drop = enable_token_drop
        
        # Attention block
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.self_attention = SelfAttention(hidden_dim, num_heads, attention_dropout)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim=hidden_dim, mlp_dim=mlp_dim)

    @staticmethod
    def minibatch_token_drop(input, max_token_drop):
        
        class_token = input[:, 0:1, :]
        input = input[:, 1:, :]
        
        drop_perc = torch.randint(0, int(100 * (0.01 + max_token_drop)), size = (1,)) / 100. # Generates a random number in [0, max_token_drop) - max_token_drop should be at most 1.0

        num_kept_tokens = math.floor(input.shape[1] * (1 - drop_perc))
        input = input[:, :num_kept_tokens, :] # Removes the trailing (1 - drop_perc)% tokens

        input_after_dropping = torch.cat([class_token, input], dim=1)

        return input_after_dropping

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        
        if self.enable_token_drop and self.training:
            input = DropViTBlock.minibatch_token_drop(input, self.max_token_drop)
        else:
            pass

        x = self.ln_1(input)
        x = self.self_attention(x)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


# ViT Encoder
class DropViTEncoder(nn.Module):
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
        max_token_drop: float = 1.0,
        drop_blocks: List[int] = [],
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: List = []
        for i in range(num_layers):
            
            drop_enabled = (i in drop_blocks)

            if drop_enabled:
                print(f"\n\nToken drop enabled on layer at index {i}\n\n")

            layers.append(DropViTBlock(
                            num_heads,
                            hidden_dim,
                            mlp_dim,
                            dropout,
                            attention_dropout,
                            max_token_drop = max_token_drop,
                            enable_token_drop = drop_enabled,
                            ))

        self.layers = nn.Sequential(*layers)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        input = self.dropout(input)
        input = self.layers(input)
        return self.ln(input)


class DropVisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""
    """ 
        During training a % of tokens has been dropped.
    """


    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        max_token_drop: float = 1.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        drop_blocks: List[int] = [],
        representation_size: Optional[int] = None,
        num_registers: int = 0,
        num_class_tokens: int = 1,
        torch_pretrained_weights: Optional[str] = None,
        timm_pretrained_weights: Optional[List] = None,
        remove_layers: List[int] = [],
    ):
        """
        Args:
            image_size (int): The size of the input image.
            patch_size (int): The size of each patch in the image.
            num_layers (int): The number of layers in the transformer encoder.
            num_heads (int): The number of attention heads in the transformer encoder.
            hidden_dim (int): The hidden dimension size in the transformer encoder.
            mlp_dim (int): The dimension size of the feed-forward network in the transformer encoder.
            max_token_drop (float): The maximum proportion of token that can be dropped at training time.
            dropout (float, optional): The dropout rate. Defaults to 0.0.
            attention_dropout (float, optional): The dropout rate for attention weights. Defaults to 0.0.
            num_classes (int, optional): The number of output classes. Defaults to 1000.
            representation_size (int, optional): The size of the output representation. Defaults to None.
            num_registers (int, optional): The number of register tokens to be added. Defaults to 0.
            num_class_tokens (int, optional): The number of class tokens to be added. Defaults to 1.
            remove_layers (List[int], optional): The list of layers to be removed from the model after loading from a checkpoint. Defaults to [].
            torch_pretrained_weights (str, optional): The path to the pretrained weights in the Torch format. Defaults to None
                Example: 'ViT_B_16_Weights[IMAGENET1K_V1]'.
                See options at https://github.com/pytorch/vision/blob/a52607ece94aedbe41107617ace22a8da91efc25/torchvision/models/vision_transformer.py#L351
            timm_pretrained_weights (List, optional): The path to the pretrained weights in the Timm format. Defaults to None. 
                Example: ['facebookresearch/deit_base_patch16_224', 'deit_base_patch16_224']
        """
        
        super().__init__()
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.max_token_drop = max_token_drop
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.num_heads = num_heads
        self.num_registers = num_registers
        self.num_class_tokens = num_class_tokens
        self.drop_blocks = drop_blocks
        
        self.conv_proj = nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size)

        seq_length = (image_size // patch_size) ** 2

        # Add class tokens
        self.class_tokens = nn.Parameter(torch.zeros(1, num_class_tokens, hidden_dim))
        seq_length += num_class_tokens

        # Add registers
        if num_registers > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, num_registers, hidden_dim))
            seq_length += num_registers

        self.encoder = DropViTEncoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            max_token_drop,
            drop_blocks,
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

        if remove_layers:
            self.remove_layers(remove_layers)
            


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
        x = torch.sum(x, dim=1)

        # Classification head
        x = self.head(x)

        return x
    
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
   
    def remove_layers(self, remove_layers: List[int]):
        """
        Removes layers from the model.

        Args:
            remove_layers (List[int]): List of layer indices to remove.
        """
        print('Removing layers: ', remove_layers)
        print('Initial number of layers:', len(self.encoder.layers))

        for i in sorted(remove_layers, reverse=True):
            del self.encoder.layers[i] 

        print('Final number of layers:', len(self.encoder.layers))
    
    def set_drop_blocks_enabled(self, drop_blocks: List[int]):
        """
        Enables token drop on the layers specified in the drop_blocks argument.

        Args:
            drop_blocks (List[int]): List of layers on which you want to enable token dropping.
        """
        self.drop_blocks = drop_blocks
        i = 0
        for drop_idx in drop_blocks:
            self.encoder.layers[drop_idx].enable_token_drop = True
        
    def set_max_drop(self, max_token_drop: float):
        """
        Set a model maximum percentage of dropped tokens. Tokens are dropped up to that value, if
        the encoder layer also has token dropping enabled.

        Args:
            max_token_drop (float): Maximum percentage of dropped tokens.
        """
        self.max_token_drop = max_token_drop

        for encoder_block in self.encoder.layers:
            encoder_block.max_token_drop = max_token_drop