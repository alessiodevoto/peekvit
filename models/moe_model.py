import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import math
from typing import Optional, List
from abc import ABC

from .blocks import GumbelSoftmax, SelfAttention, MLP
 



# Abstract class for MoE
class MoE(ABC, nn.Module):
  pass



# A differentiable TopK Gate
class TopKGate(nn.Module):

  def __init__(self, input_dim, num_experts):
    super().__init__()
    self.gate = nn.Linear(input_dim, num_experts)
    self.activation = GumbelSoftmax(dim=-1, hard=True)

  def forward(self, x):
    scores = self.gate(x)
    return self.activation(scores)



# ViT MoE of MLPs
class MLPMoE(MoE):

    def __init__(self, hidden_dim, mlp_dim, num_experts):
      super().__init__()
      self.gating_network = TopKGate(hidden_dim, num_experts)
      self.num_experts = num_experts
      self.experts = nn.ModuleList([MLP(hidden_dim, mlp_dim) for _ in range(num_experts)])
      
    def forward_one(self, x):
      # if there is only one expert, just return its output without gating
      return self.experts[0](x)
      
    def forward_moe(self, x):

      torch._assert(x.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {x.shape}")

      # compute the gating probabilities
      self.gating_probs = self.gating_network(x)   # batch, seq, exps

      # iterate over the experts and compute their outputs, then stack them
      expert_outputs = [expert(x) for expert in self.experts]
      output = torch.stack(expert_outputs, dim=0)
      output = torch.einsum('ebsd, bse -> bsd', output, self.gating_probs)

      return output
  
    def forward(self, x):
      if self.num_experts == 1:
        return self.forward_one(x)
      else:
        return self.forward_moe(x)


# ViT MoE of Self Attention
class AttentionMoE(MoE):

    def __init__(self, input_dim, num_heads, num_experts, dropout=0.0):
      super().__init__()
      self.gating_network = TopKGate(input_dim, num_experts)
      self.num_experts = num_experts
      self.experts = nn.ModuleList([SelfAttention(input_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_experts)])


    def forward_one(self, x):
      # if there is only one expert, just return its output without gating
      return self.experts[0](x)

    def forward_moe(self, x):

      torch._assert(x.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {x.shape}")

      # compute the gating probabilities
      self.gating_probs = self.gating_network(x)   # batch, seq, exp

      # iterate over the experts and compute their outputs, then stack them
      expert_outputs = [expert(x) for expert in self.experts]
      output = torch.stack(expert_outputs, dim=0)
      output = torch.einsum('ebsd, bse -> bsd', output, self.gating_probs)

      return output
    
    def forward(self, x):
      if self.num_experts == 1:
        return self.forward_one(x)
      else:
        return self.forward_moe(x)


# ViT MoE Block
class ViTBlockMoE(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        mlp_num_experts: int = 1,
        attn_num_experts: int = 1
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.self_attention = AttentionMoE(hidden_dim, num_heads, attn_num_experts, attention_dropout)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLPMoE(hidden_dim=hidden_dim, mlp_dim=mlp_dim, num_experts=mlp_num_experts)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        
        x = self.ln_1(input)
        x = self.self_attention(x)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


# ViT Encoder
class ViTEncoderMoE(nn.Module):
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
        mlp_moes: List = None,
        attn_moes: List = None
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.mlp_moes = mlp_moes or [1] * num_layers
        self.attn_moes = attn_moes or [1] * num_layers
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = ViTBlockMoE(
                                                num_heads,
                                                hidden_dim,
                                                mlp_dim,
                                                dropout,
                                                attention_dropout,
                                                mlp_num_experts = self.mlp_moes[i],
                                                attn_num_experts = self.attn_moes[i]
                                                )

        self.layers = nn.Sequential(layers)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        input = self.dropout(input)
        input = self.layers(input)
        return self.ln(input)


# ViT MoE
class VisionTransformerMoE(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

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
        mlp_moes: List = None,
        attn_moes: List = None
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
        self.mlp_moes = mlp_moes or [1] * num_layers
        self.attn_moes = attn_moes or [1] * num_layers


        self.conv_proj = nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size)

        seq_length = (image_size // patch_size) ** 2

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.encoder = ViTEncoderMoE(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            mlp_moes,
            attn_moes
            )


        self.seq_length = seq_length

        self.head = nn.Linear(hidden_dim, num_classes)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)


        fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
        nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
        if self.conv_proj.bias is not None:
            nn.init.zeros_(self.conv_proj.bias)


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

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.head(x)

        return x




# Test
"""
vitmoe = VisionTransformerMoE(image_size = 64, num_layers=2, patch_size=32, num_heads=4, hidden_dim=4, mlp_dim=32, dropout=0.0, attention_dropout=0.0, mlp_moes=[5, 10])
data = torch.randn(2, 3, 64, 64)
out = vitmoe(data)
out.mean().backward()
"""