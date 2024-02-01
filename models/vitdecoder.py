import math
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from typing import List, Optional
import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

from .vit import ViTBlock


"""
Tranformer that can be used as a decoder, as it returns an image. 
This model takes as input a sequence of tokens and returns an image.
"""


# ViT Encoder
class ViTDecoder(nn.Module):
    """
    Transformer Model Encoder for sequence to sequence translation.
    Only difference with ViTEncoder is that we do not add positional embeddings.
    """

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.dropout = nn.Dropout(dropout)
        layers: List = []
        for i in range(num_layers):
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
        input = self.dropout(input)
        input = self.layers(input)
        return self.ln(input)


class VisionTransformerDecoder(torch.nn.Module):
    def __init__(self,
                image_size: int,
                patch_size: int,
                hidden_dim: int,
                mlp_dim: int, 
                seq_length: int,
                num_layers: int,
                num_heads: int,
                dropout: float,
                attention_dropout: float,
                ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embedding = torch.nn.Parameter(torch.empty(1, seq_length-1, hidden_dim).normal_(std=0.02))
        self.image_size = image_size
        self.patch_size = patch_size

        self.encoder = ViTDecoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout
            )

        self.head = torch.nn.Linear(hidden_dim, 3 * patch_size ** 2)
        self.patch2img = Rearrange('b (h w) (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size)


    def forward(self, tokens, mask):
        
        batch, seq_length, hidden_dim = tokens.shape
        
        # Replace masked tokens with mask token
        # mask is shape (batch_size, seq_length, 1) and contains 0s where tokens are masked, floats > 0 otherwise
        # we should first make the mask contain 0s and 1s, in a differentiable way
        mask = torch.sigmoid(mask*100)
        mask = mask.expand(-1, -1, hidden_dim) 
        tokens = tokens * mask + self.mask_token * (1-mask)

        # Add positional embeddings
        tokens += self.pos_embedding
            
        # Pass through the encoder
        tokens = self.encoder(tokens)

        # We should recover the original image from the tokens
        # (batch_size, seq_length, hidden_dim) -> (batch_size, 3, image_size, image_size)
        tokens = self.head(tokens)
        img = self.patch2img(tokens)

        return img


