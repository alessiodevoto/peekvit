from random import random
import torch
from torch import nn
import torch.nn.functional as F
import math
from typing import Optional, List, Union, Literal
from abc import ABC
from einops import reduce

from .blocks import SelfAttention, MLP, GumbelSigmoid, SigmoidWithTemp

"""
A Vision Transformer with Residual Gating which can be placed at any layer.
"""

class ResidualModule(ABC, nn.Module):
  pass


class ResidualGate(nn.Module):
    def __init__(self, hidden_dim, threshold:Union[float, str] = 0.5, temp=1.0, gate_type='gumbel', sigmoid_bias:float = 10.0):
        super().__init__()
        self.projection = nn.Linear(hidden_dim, 1)
        self.temp = temp
        self.gate_type = gate_type
        self.sigmoid_bias = sigmoid_bias

        if gate_type == 'gumbel':
            self.gate = GumbelSigmoid(hard=True, temp=temp, bias=sigmoid_bias)
        elif gate_type == 'sigmoid':
            self.gate = SigmoidWithTemp(temp=temp, bias=sigmoid_bias)
        else:
            raise ValueError(f'Unknown gate type {gate_type}')
        
        if gate_type == 'gumbel' and threshold != 0.5:
            raise ValueError(f'Gumbel gate cannot have a threshold different from 0.5')
        

        # threshold might be a fixed float or a learnable parameter, initialize it accordingly
        if isinstance(threshold, float):
            self.threshold = threshold
        elif threshold == 'learnable':
            self.threshold = nn.Parameter(torch.tensor(0.5))


    def forward(self, x, budget : float = None, threshold: float = None):
        
        # x is (batch_size, seq_length, hidden_dim)
        assert x.dim() == 3, f'Expected (batch_size, seq_length, hidden_dim) got {x.shape}'
        assert budget is None or threshold is None, 'Cannot specify both budget and threshold'

        # compute the mask by first projecting each token into a scalar and then applying the gate
        mask_log = self.projection(x) 
        mask = self.gate(mask_log)
        
        
        if self.gate_type == 'sigmoid':
            # in this case the mask is soft, so we have to push it to 0 or 1
            if budget is not None:
                # if the budget is specified as a fixed float, we push the mask to 0 or 1 based on the budget
                mask = F.relu(mask - (1-budget)) 
            elif threshold is not None:
                # if the threshold is specified as a fixed float, we push the mask to 0 or 1 based on the threshold
                mask = F.relu(mask - threshold)
                self.threshold = threshold
            else:
                # if the threshold is learnable, we push the mask to 0 or 1 based on the learnable threshold
                mask = F.relu(mask - self.threshold)
        else:
            assert budget is None, 'Gumbel gate does not support budget'
            

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
        gate_bias: float = 10.0,
        gate_threshold: float = 0.5,
        budget_token: Union[bool, List, Literal['learnable']] = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.budget_token = budget_token 
        self.num_special_tokens = num_class_tokens + num_registers

        # residual settings
        self.gate_type = gate_type
        self.skip = skip
        if skip in {'attention', 'mlp', 'attention+mlp'}:
            self.temp = temp
            self.add_input = add_input
            self.residual_gate = ResidualGate(hidden_dim, threshold=gate_threshold, temp=temp, gate_type=gate_type, sigmoid_bias=gate_bias)

        # Attention block
        self.ln_1 = nn.LayerNorm(hidden_dim, eps=1e-06)
        self.self_attention = SelfAttention(hidden_dim, num_heads, dropout=attention_dropout)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = nn.LayerNorm(hidden_dim, eps=1e-06)
        self.mlp = MLP(hidden_dim=hidden_dim, mlp_dim=mlp_dim)

        # budget learner
        if self.budget_token == 'learnable':
            self.budget_token_gate = nn.Linear(hidden_dim, 1)


    def forward_skip_attention(self, input: torch.Tensor):
        # we should mask only non special tokens
        special_tokens = input[:, :self.num_special_tokens, :]
        img_tokens = input[:, self.num_special_tokens:, :]

        if self.budget_token:
            # budget token is the last token
            budget_token = img_tokens[:, -1:, :] 
            img_tokens = img_tokens[:, :-1, :]


        # residual gating, here we learn a scalar for each token
        self.mask = self.residual_gate(img_tokens, budget=budget_token.mean() if self.budget_token else None) 
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

        if self.budget_token:
            # budget token is the last token
            budget_token = img_tokens[:, -1:, :] 
            img_tokens = img_tokens[:, :-1, :]

        # residual gating, here we learn a scalar for each token
        # residual gating, here we learn a scalar for each token
        self.mask = self.residual_gate(img_tokens, budget=budget_token.mean() if self.budget_token else None) 
        masked_tokens = self.mask * img_tokens 

        # concatenate special tokens and masked input
        masked_input = torch.cat([special_tokens, masked_tokens], dim=1)

        if self.budget_token:
            masked_input = torch.cat([masked_input, budget_token], dim=1)

        y = self.ln_2(masked_input)
        y = self.mlp(y)

        if self.add_input:
            # only img_tokens should be added to the output
            unmasked_tokens = img_tokens * (1-self.mask)
            y = y + torch.cat([torch.zeros_like(special_tokens), unmasked_tokens], dim=1)
        
        return y


    def forward_skip_attention_mlp(self, input: torch.Tensor):
        
        # we should mask only non special tokens
        special_tokens = input[:, :self.num_special_tokens, :]
        img_tokens = input[:, self.num_special_tokens:, :]

        current_budget, threshold = None, None
        if self.budget_token:
            # if we are using a budget token in input, it is the last token
            # if self.budget_token == 'learnable':
            budget_token = img_tokens[:, -1:, :] 
            img_tokens = img_tokens[:, :-1, :]
            """elif self.budget_token == 'learnable_interpolate':
                budget_token = img_tokens[:, -2:, :]
                img_tokens = img_tokens[:, :-2, :]"""
            current_budget = budget_token.mean()
        
        if self.budget_token == 'learnable':
            # if the budget token is a learnable parameter, we use it to compute the threshold
            threshold = torch.sigmoid(self.budget_token_gate(budget_token))
            current_budget = None


        # residual gating, here we learn a scalar [0 or 1s] to mask each token
        self.mask = self.residual_gate(img_tokens, budget=current_budget, threshold=threshold)
        
        # mask the input
        masked_tokens = self.mask * img_tokens 
        
        # concatenate back special tokens, masked input
        masked_input = torch.cat([special_tokens, masked_tokens], dim=1)

        # add budget token back
        if self.budget_token:
            masked_input = torch.cat([masked_input, budget_token], dim=1)

        # plain forward
        y = self.plain_forward(masked_input)

        if self.add_input:
            # only img_tokens should be added to the output
            unmasked_tokens = img_tokens * (1-self.mask)
            y = y + torch.cat([torch.zeros_like(special_tokens), unmasked_tokens], dim=1)

        return y

    # TODO this is messy, it was a last second change when I found out the layenorm was changing the representation
    # of zero tokens in the input, so I had to add a mask to the layernorm. I should clean this up. 
    # notice that if we don't make sure all masked tokens are actually zero, the flop count will be wrong
    def plain_forward(self, input: torch.Tensor):
        if self.skip == 'attention+mlp':
            expanded_mask = torch.cat([torch.ones((self.mask.size(0), 1, self.mask.size(2)), device=self.mask.device), self.mask, torch.ones((self.mask.size(0), 1, self.mask.size(2)), device=self.mask.device)], dim=1)
        else:
            expanded_mask = torch.tensor(1.0, device=input.device)
        x = expanded_mask * self.ln_1(input) 
        
        x = expanded_mask * self.self_attention(x)
        x = self.dropout(x)
        x = x + input

        y = expanded_mask * self.ln_2(x)
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
        gate_bias: float = 10.0,
        gate_threshold: float = 0.5,
        budget_token: Union[bool, List, Literal['learnable']] = False,

    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.num_class_tokens  = num_class_tokens
        self.num_registers = num_registers
        self.num_special_tokens = num_class_tokens + num_registers
        self.budget_token = budget_token
        
        self.num_budget_tokens = 0 if not budget_token else 1

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
                            temp=gate_temp,
                            gate_bias=gate_bias,
                            gate_threshold=gate_threshold,
                            budget_token=budget_token
                            ))

        self.layers = nn.Sequential(*layers)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        
        if self.budget_token:
            # budget token is the last token
            budget_tokens = input[:, -self.num_budget_tokens:, :]
            input = input[:, :-self.num_budget_tokens, :]

        input = input + self.pos_embedding
        if self.budget_token:
            input = torch.cat([input, budget_tokens], dim=1)
        input = self.dropout(input)
        input = self.layers(input)
        return self.ln(input)



class ResidualVisionTransformer(nn.Module):
    """
    Residual Vision Transformer model for image classification.

    Args:
        image_size (int): The size of the input image.
        patch_size (int): The size of each patch in the image.
        num_layers (int): The number of layers in the model.
        num_heads (int): The number of attention heads in each layer.
        hidden_dim (int): The dimensionality of the hidden layers.
        mlp_dim (int): The dimensionality of the MLP layers.
        dropout (float, optional): The dropout rate. Defaults to 0.0.
        attention_dropout (float, optional): The dropout rate for attention layers. Defaults to 0.0.
        num_classes (int, optional): The number of output classes. Defaults to 1000.
        representation_size (int, optional): The size of the output representation. Defaults to None.
        
        num_registers (int, optional): The number of register tokens. Defaults to 0.
        residual_layers (List, optional): The list of residual layers. It must be a list of length `num_layers`, where each 
            element of list can be {`attention+mlp`, `attention`, `mlp`}. Defaults to None.
        add_input (bool, optional): Whether to add the input to the output, thus making it possible to skip tokens and reuse them
            later on. Defaults to False. DO NOT USE IT AS IT IS BUGGY. (TODO change this piece of doc)
        num_class_tokens (int, optional): The number of class tokens. Defaults to 1. Notice that the final head will average all
            the class tokens to produce the final output.
        gate_type (Literal['gumbel', 'sigmoid'], optional): The type of gate for residual layers. Defaults to 'gumbel'.
        gate_temp (float, optional): The temperature for the gate. Defaults to 1.0.
        gate_threshold (float, optional): The threshold for the gate. Defaults to 0.5.
        add_budget_token (bool, str, optional): Whether to add a budget token at the end of each sequence. It can be:
            - False to not add a budget token
            - True to sample a budget token in `budget_interval`
            - a tuple-like to specify a set of budgets to sample from, 
            - a float to have a fixed budget across training 
            - 'learnable' to add a learnable budget token and sample a value between 0 and 1 to multiply to it
            - learnable interpolate to have 2 trainable budget tokens and sample a value (budget) to interpolate between them.  
            For now, the same budget is sampled for each batch. Defaults to False.
        budget_interval (Tuple): the min and max value for the budget. Defaults to (0, 1)

    """

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
        add_input: bool = False,
        num_class_tokens: int = 1,
        gate_type: Literal['gumbel', 'sigmoid'] = 'gumbel',
        gate_temp: float = 1.0,
        gate_bias: float = 10.0,
        gate_threshold: float = 0.5,
        add_budget_token: Union[bool, List, Literal['learnable', 'learnable_interpolate']] = False,
        budget_interval: Optional[List] = (0,1)
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
        self.add_budget_token = add_budget_token
        self.current_budget = None
        self.gate_temp = gate_temp 
        self.gate_bias = gate_bias
        self.budget_interval = budget_interval
        
        # assume all layers are residual by default
        self.residual_layers = residual_layers or ['attention+mlp'] * num_layers

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
            gate_bias=gate_bias,
            gate_threshold=gate_threshold,
            budget_token=add_budget_token,
            )
    
        self.seq_length = seq_length

        if self.add_budget_token:
            # base case where we add a homogeneous token of floats
            seq_length += 1
            self.num_budget_tokens = 1

            # we add a sampled budget token
            if self.add_budget_token == 'learnable':
                self.learnable_budget_token_1 = nn.Parameter(torch.randn(1, 1, hidden_dim))
            
            # we add 2 sampled budget tokens
            if self.add_budget_token == 'learnable_interpolate':
                self.learnable_budget_token_1 = nn.Parameter(torch.randn(1, 1, hidden_dim))
                self.learnable_budget_token_2 = nn.Parameter(torch.randn(1, 1, hidden_dim))
                seq_length += 1
                self.num_budget_tokens = 2
            
            


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
    
    def _sample_budget(self):
        """
        Sample a budget, i.e. a float between 0 and 1, based on the self.add_budget_token parameter.
        """
        if isinstance(self.add_budget_token, (list, tuple)):
            return torch.tensor(random.choice(self.add_budget_token))
        elif isinstance(self.add_budget_token, float):
            return torch.tensor(self.add_budget_token)
        else:
            return torch.rand(1) * (self.budget_interval[1] - self.budget_interval[0]) + self.budget_interval[0]

    def _add_budget_token(self, x):
        """
        Adds a budget token to the input tensor based on the self.budget. 
        After calling this method, self.current_budget will contain the value of the current budget.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).

        Returns:
            torch.Tensor: Tensor with the budget token added, of shape (batch_size, seq_len+1, hidden_dim).
        """
        n = x.shape[0]
        if self.training:
            sampled_budget = self._sample_budget()
            self.current_budget = sampled_budget.to(x.device)
        else:
            assert self.current_budget is not None, 'Budget token not set. Call set_budget() before forward() to evaluate the model on a chosen budget.'

        

        if self.add_budget_token == 'learnable':
            batch_budget_token_1 = self.learnable_budget_token_1.expand(n, -1, -1) * self.current_budget
            x = torch.cat([x, batch_budget_token_1], dim=1)
        elif self.add_budget_token == 'learnable_interpolate':
            batch_budget_token_1 = self.learnable_budget_token_1.expand(n, -1, -1) * self.current_budget
            batch_budget_token_2 = self.learnable_budget_token_2.expand(n, -1, -1) * (1-self.current_budget)
            x = torch.cat([ x, batch_budget_token_1 + batch_budget_token_2], dim=1)
        else:
            budget_token = torch.empty((n, 1, self.hidden_dim), device=x.device).fill_(self.current_budget)
            x = torch.cat([x, budget_token], dim=1)
        
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

        # Add budget token
        if self.add_budget_token:
            x = self._add_budget_token(x)
        
        # Pass through the encoder
        x = self.encoder(x)

        # Get all class tokens and average them
        x = x[:, 0:self.num_class_tokens]
        #print(x.shape)
        x = reduce(x, 'n c e -> n e', reduction='sum')

        # Classification head
        x = self.head(x)

        return x


    def set_budget(self, budget: float):
        self.current_budget = budget

    