import torch
from torch import nn
import torch.nn.functional as F
import math
from typing import Optional, List, Union
from abc import ABC
from torchvision.models.vision_transformer import ViT_B_16_Weights, ViT_B_32_Weights


from .blocks import SelfAttention, MLP
 
"""
Vision Transformer that drops tokens across the batch in training.
"""

# Drop Module
class DropModule(nn.Module):
    def __init__(self, drop_rate: float):
        super().__init__()
        


# Batch ViT Block
class BatchRankViTBlock(nn.Module):
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

        # Drop tokens
        self.drop = False
        self.current_budget = 1.0
    
    def drop_tokens(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")

        if self.training and not self.drop:
            return input

        class_token = input[:, 0:1, :]
        input = input[:, 1:, :]

        # get token magnitudes
        # (batch_size, seq_length, hidden_dim) -> (batch_size, seq_length)
        token_magnitudes = torch.norm(input, dim=-1)

        # average over batch
        # (batch_size, seq_length) -> (seq_length)
        avg_magnitudes = torch.mean(token_magnitudes, dim=0)

        # get sorted indices
        idx = torch.argsort(avg_magnitudes, descending=True)

        # get the number of tokens to mask
        num_tokens_to_keep = math.ceil(input.shape[1] * self.current_budget)

        # drop tokens
        input = input[:, idx[:num_tokens_to_keep], :]

        # add class token back
        input = torch.cat([class_token, input], dim=1)

        return input

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        
        # notice that drop_tokens is implemented as no-op if sort_tokens is False
        input = self.drop_tokens(input) # only has effect during training

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
class BatchRankViTEncoder(nn.Module):
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
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: List = []
        for i in range(num_layers):
            layers.append(BatchRankViTBlock(
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



class BatchRankVisionTransformer(nn.Module):
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
        num_registers: int = 0,
        num_class_tokens: int = 1,
        torch_pretrained_weights: Optional[str] = None,
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



        self.conv_proj = nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size)

        seq_length = (image_size // patch_size) ** 2

        # Add class tokens
        self.class_tokens = nn.Parameter(torch.zeros(1, num_class_tokens, hidden_dim))
        seq_length += num_class_tokens

        # Add registers
        if num_registers > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, num_registers, hidden_dim))
            seq_length += num_registers

        self.encoder = BatchRankViTEncoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout
            )


        self.seq_length = seq_length

        self.head = nn.Linear(hidden_dim, num_classes)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)


        fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
        nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
        if self.conv_proj.bias is not None:
            nn.init.zeros_(self.conv_proj.bias)
        

        if torch_pretrained_weights is not None:
            from .adapters import adapt_torch_state_dict
            torch_pretrained_weights = eval(torch_pretrained_weights).get_state_dict()
            adapted_state_dict = adapt_torch_state_dict(torch_pretrained_weights, num_classes=num_classes)
            self.load_state_dict(adapted_state_dict, strict=False)


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
    

    def enable_ranking(self, sort_tokens: Union[bool, List[bool]] = False):
        """
        Enable dropping for the BatchRankVit model.

        Args:
            sort_tokens (Union[bool, List[bool]], optional): 
                A boolean value or a list of boolean values indicating whether to sort tokens for each BatchRankVit block. 
                If a single boolean value is provided, it will be applied to all BatchRankVit blocks. 
                If a list of boolean values is provided, each value will be applied to the corresponding RankVit block. 
                Defaults to False.
        """
        if isinstance(sort_tokens, bool):
            sort_tokens = [sort_tokens] * len(self.encoder.layers)

        for rankvitblock, sort in zip(self.encoder.layers, sort_tokens):
            rankvitblock.sort = sort
    

    def set_budget(self, budget: float):
        self.current_budget = budget
        for rankvitblock in self.encoder.layers:
            if hasattr(rankvitblock, 'set_budget'):
                rankvitblock.set_budget(budget)

