import torch
from torch import nn
import torch.nn.functional as F
import math
from typing import Optional, List
from abc import ABC
from torchvision.models.vision_transformer import ViT_B_16_Weights, ViT_B_32_Weights
from pytorch3d.ops import knn_points
from einops.layers.torch import Reduce



from .blocks import SelfAttention, MLP
 
"""
Plain Point Cloud Transformer.
"""


# PCT Block
class PCTBlock(nn.Module):
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
        
    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        
        x = self.ln_1(input)
        x = self.self_attention(x)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y
    
#ARPE: Absolute Relative Position Encoding
class ARPE(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, npoints=1024):
        super(ARPE, self).__init__()

        N0 = 512
        k0 = 32
        self.k = int(k0 * npoints / N0)



        self.lin1 = nn.Linear(2*in_channels, 2*in_channels)
        self.lin2 = nn.Linear(2*in_channels, out_channels)

        self.bn1 = nn.BatchNorm1d(2*in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.max_pooling_layer = Reduce('bn k f -> bn 1 f', 'max')
     
    def forward(self, x):
    
        B, N, C = x.shape  # B: batch size, N: number of points, C: channels

        knn = knn_points(x, x, K=self.k, return_nn=True)[2] # B, N, K, C

        diffs = x.unsqueeze(2) - knn  # B, N, K, C
        x = torch.cat([x.unsqueeze(2).repeat(1, 1, self.k, 1), diffs], dim=-1) # B, N, K, 2*C
        x = F.elu(self.bn1(self.lin1(x.view(B*N, self.k, 2*C)).transpose(1,2)).transpose(1,2)) # B*N, K, 2*C
        x = self.max_pooling_layer(x).squeeze(2) # B*N, 1, 2*C -> B*N, 2*C
        x = F.elu(self.bn2(self.lin2(x.view(B, N, 2*C)).transpose(1,2)).transpose(1,2)) # B, N, out_channels

        return x # B, N, 2*C

# PCT Encoder
class PCTEncoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
            self,
            num_layers: int,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float,
            attention_dropout: float,
    ):
        super().__init__()          
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                PCTBlock(
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = self.dropout(input)
        for layer in self.layers:
            input = layer(input)
        return input
    
class PointCloudTransformer(nn.module):

    def __init__(
            self,
            num_layers: int,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
            num_classes: int = 40,
            representation_size: Optional[int] = None,
            num_registers: int = 0,
            num_class_tokens: int = 0,
            torch_pretrained_weights: Optional[str] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.num_heads = num_heads
        self.num_registers = num_registers
        self.num_class_tokens = num_class_tokens

