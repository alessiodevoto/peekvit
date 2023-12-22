import torch
from torch import nn
import torch.nn.functional as F
import math
from typing import Optional, List, Union, Literal
from abc import ABC
from einops import reduce

from .blocks import SelfAttention, MLP, GumbelSigmoid

"""
A Vision Transformer with Residual Gating which can be placed at any layer.
The residual gating is implemented as a learnable vector for each token.
"""

class ResidualModule(ABC, nn.Module):
  pass


class ResidualGate(nn.Module):
    def __init__(self, hidden_dim, temp=1.0, gate_type='gumbel', sigmoid_bias:float = 10):
        super().__init__()
        self.projection = nn.Linear(hidden_dim, 1)
        self.temp = temp
        self.gate_type = gate_type
        self.sigmoid_bias = sigmoid_bias
        if gate_type == 'gumbel':
            self.gate = GumbelSigmoid(hard=True)
        elif gate_type == 'sigmoid':
            self.gate = nn.Sigmoid()
        else:
            raise ValueError(f'Unknown gate type {gate_type}')

    def forward(self, x):
        mask_log = self.projection(x) / self.temp
        # print(mask_log)
        if self.gate_type == 'gumbel':
            mask = self.gate(mask_log)
        elif self.gate_type == 'sigmoid':
            # print(torch.sigmoid(mask_log + self.sigmoid_bias))
            # mask = F.relu(torch.sigmoid(mask_log * 0.5 + self.sigmoid_bias) - 0.5) 
            mask = torch.sigmoid(mask_log * 0.5 + self.sigmoid_bias) # squeeze sigmoid
            # mask = (torch.sigmoid(mask_log + self.sigmoid_bias)  - 0.5)
        
        return mask





# ViT Block
class ResidualViTBlock(ResidualModule):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        temp: float = 1.0,
        add_input: bool = False,
        num_class_tokens: int = 1,
        num_registers: int = 0,
        skip : Literal['attention', 'mlp', 'attention+mlp', 'none'] = None,
        gate_type: Literal['gumbel', 'sigmoid'] = 'gumbel',
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.num_special_tokens = num_class_tokens + num_registers

        # residual settings
        self.gate_type = gate_type
        self.skip = skip
        if skip in {'attention', 'mlp', 'attention+mlp'}:
            self.temp = temp
            self.add_input = add_input
            self.residual_gate = ResidualGate(hidden_dim, temp=temp, gate_type=gate_type)

        # Attention block
        self.ln_1 = nn.LayerNorm(hidden_dim, eps=1e-06)
        self.self_attention = SelfAttention(hidden_dim, num_heads, dropout=attention_dropout)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = nn.LayerNorm(hidden_dim, eps=1e-06)
        self.mlp = MLP(hidden_dim=hidden_dim, mlp_dim=mlp_dim)


    def forward_skip_attention(self, input: torch.Tensor):
        # we should mask only non special tokens
        special_tokens = input[:, :self.num_special_tokens, :]
        img_tokens = input[:, self.num_special_tokens:, :]

        # residual gating, here we learn a scalar for each token
        self.mask = self.residual_gate(img_tokens)
        masked_tokens = self.mask * img_tokens

        # concatenate special tokens and masked input
        masked_input = torch.cat([special_tokens, masked_tokens], dim=1)

        x = self.ln_1(masked_input)
        x = self.self_attention(x)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)

        return y 

    def forward_skip_mlp(self, input: torch.Tensor):

        x = self.ln_1(input)
        x = self.self_attention(x)
        x = self.dropout(x)
        x = x + input
        
        # we should mask only non special tokens
        special_tokens = x[:, :self.num_special_tokens, :]
        img_tokens = x[:, self.num_special_tokens:, :]

        # residual gating, here we learn a scalar for each token
        self.mask = self.residual_gate(img_tokens)
        masked_tokens = self.mask * img_tokens

        # concatenate special tokens and masked input
        masked_input = torch.cat([special_tokens, masked_tokens], dim=1)
        y = self.ln_2(masked_input)
        y = self.mlp(y)

        if self.add_input:
            y = y + x
        
        return y


    def forward_skip_attention_mlp(self, input: torch.Tensor):
        
        # we should mask only non special tokens
        special_tokens = input[:, :self.num_special_tokens, :]
        img_tokens = input[:, self.num_special_tokens:, :]


        # residual gating, here we learn a scalar for each token
        self.mask = self.residual_gate(img_tokens) 
        masked_tokens = self.mask  * img_tokens 
        unmasked_tokens = img_tokens * (1-self.mask)

        """print(self.mask.shape, img_tokens.shape)
        print('img_tokens')
        print(img_tokens)
        print('mask')
        print(self.mask)
        print('masked_tokens')
        print(masked_tokens)"""

        # concatenate special tokens and masked input
        masked_input = torch.cat([special_tokens, masked_tokens], dim=1)

        # plain forward
        y = self.plain_forward(masked_input)

        if self.add_input:
            # only img_tokens should be added to the output
            y = y + torch.cat([torch.zeros_like(special_tokens), unmasked_tokens], dim=1)

        return y

    def plain_forward(self, input: torch.Tensor):
        x = self.ln_1(input)
        x = self.self_attention(x)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")

        if self.skip == 'attention':
            return self.forward_skip_attention(input)
        elif self.skip == 'mlp':
            return self.forward_skip_mlp(input)
        elif self.skip == 'attention+mlp':
            return self.forward_skip_attention_mlp(input)
        else:
            return self.plain_forward(input)



# ViT Encoder
class ResidualViTEncoder(nn.Module):
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
        residual_layers: Optional[List] = None,
        add_input: bool = False,
        num_class_tokens: int = 1,
        num_registers: int = 0,
        gate_type: Literal['gumbel', 'sigmoid'] = 'gumbel',
        gate_temp: float = 1.0,

    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.num_class_tokens  = num_class_tokens
        self.num_registers = num_registers
        self.num_special_tokens = num_class_tokens + num_registers

        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers:List = []
        for i in range(num_layers):
            layers.append(ResidualViTBlock(
                            num_heads,
                            hidden_dim,
                            mlp_dim,
                            dropout,
                            attention_dropout,
                            skip = residual_layers[i],
                            add_input=add_input,
                            num_class_tokens=num_class_tokens,
                            num_registers=num_registers,
                            gate_type=gate_type,
                            temp=gate_temp
                            ))

        self.layers = nn.Sequential(*layers)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        input = self.dropout(input)
        input = self.layers(input)
        return self.ln(input)


class ResidualVisionTransformer(nn.Module):
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
        residual_layers: Optional[List] = None,
        add_input: bool = True,
        num_class_tokens: int = 1,
        gate_type: Literal['gumbel', 'sigmoid'] = 'gumbel',
        gate_temp: float = 1.0,
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
        self.num_registers = num_registers
        self.num_class_tokens = num_class_tokens
        
        # assume all layers are residual by default
        self.residual_layers = residual_layers or [None] * num_layers

        self.conv_proj = nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size)

        seq_length = (image_size // patch_size) ** 2

        # Add class tokens
        self.class_tokens = nn.Parameter(torch.zeros(1, num_class_tokens, hidden_dim))
        seq_length += num_class_tokens

        # Add registers
        if num_registers > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, num_registers, hidden_dim))
            seq_length += num_registers
        
        self.num_special_tokens = num_class_tokens + num_registers

        self.encoder = ResidualViTEncoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            residual_layers=self.residual_layers,
            add_input=add_input,
            gate_type=gate_type, 
            gate_temp=gate_temp,
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
        #print(x.shape)
        x = reduce(x, 'n c e -> n e', reduction='sum')

        # Classification head
        x = self.head(x)

        return x


