import torch
from torch import nn
import torch.nn.functional as F
import math
from typing import Optional, List, Union, Literal
from abc import ABC
from einops import reduce

from .blocks import SelfAttention, MLP, GumbelSigmoid, SigmoidWithTemp
from .residualvit import ResidualViTEncoder
from .vitdecoder import VisionTransformerDecoder

"""
A Vision Transformer with Residual Gating which can be placed at any layer
and a Decoder which can be used to reconstruct the input image.
"""




class ResidualVisionTransformerWithDecoder(nn.Module):
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
            - True to sample a budget token in [0,1]
            - a tuple-like to specify a set of budgets to sample from, a float to have a fixed budget 
            across training, 
            - 'learnable' to add a learnable budget token and sample a value between 0 and 1 to multiply to it
            - learnable interpolate to have 2 trainable budget tokens and sample a value (budget) to interpolate between them.  
            For now, the same budget is sampled for each batch. Defaults to False.
        
        decoder_hidden_dim (int, optional): The dimensionality of the hidden layers in the decoder. Defaults to same as encoder.
        decoder_num_layers (int, optional): The number of layers in the decoder. Defaults to same as encoder.
        decoder_num_heads (int, optional): The number of attention heads in each layer of the decoder. Defaults to same as encoder..
        decoder_mlp_dim (int, optional): The dimensionality of the MLP layers in the decoder.Defaults to same as encoder.
        decoder_dropout (float, optional): The dropout rate for the decoder. Defaults to same as encoder.
        decoder_attention_dropout (float, optional): The dropout rate for attention layers in the decoder. Defaults to same as encoder.
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
        sample_budget: Union[bool, List] = False,
        add_budget_token: Union[bool, List, Literal['learnable', 'learnable_interpolate']] = False,

        decoder_hidden_dim: int = None,
        decoder_num_layers: int = None,
        decoder_num_heads: int = None,
        decoder_mlp_dim: int = None,
        decoder_dropout: float = 0.0,
        decoder_attention_dropout: float = 0.0,
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
        self.budget = add_budget_token
        self.sample_budget = sample_budget
        self.current_budget = None
        self.gate_temp = gate_temp 
        self.gate_bias = gate_bias
        
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

        if self.budget:
            seq_length += 1
            self.num_budget_tokens = 1
        
        if self.budget == 'learnable' or self.budget == 'learnable_interpolate':
            self.learnable_budget_token_1 = nn.Parameter(torch.randn(1, 1, hidden_dim))
            # actually we need two learnable parameters for the non interpolate case
            self.learnable_budget_token_2 = nn.Parameter(torch.randn(1, 1, hidden_dim))
            
            


        self.head = nn.Linear(hidden_dim, num_classes)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)


        fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
        nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
        if self.conv_proj.bias is not None:
            nn.init.zeros_(self.conv_proj.bias)
        

        # Decoder
        self.decoder_hidden_dim = decoder_hidden_dim or hidden_dim
        self.decoder_num_layers = decoder_num_layers or num_layers
        self.decoder_num_heads = decoder_num_heads or num_heads
        self.decoder_mlp_dim = decoder_mlp_dim or mlp_dim
        self.decoder_dropout = decoder_dropout or dropout
        self.decoder_attention_dropout = decoder_attention_dropout or attention_dropout

        self.decoder = VisionTransformerDecoder(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=self.decoder_num_layers,
            num_heads=self.decoder_num_heads,
            hidden_dim=self.decoder_hidden_dim,
            mlp_dim=self.decoder_mlp_dim,
            seq_length=self.seq_length,
            dropout=self.decoder_dropout,
            attention_dropout=self.decoder_attention_dropout,
            )



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
        if isinstance(self.budget, float):
            # fixed budget
            sampled_budget = self.budget
        elif isinstance(self.budget, list) or isinstance(self.budget, tuple):
            # select a random value from the possible budgets
            idx = torch.randint(0, len(self.budget), (1,)).item()
            sampled_budget = self.budget[idx]
        elif self.budget == True:
            # sample a random budget in [0,1)
            sampled_budget = torch.rand(1).item()
        return sampled_budget


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
            budget_token = torch.empty((n, 1, self.hidden_dim), device=x.device)
            
            if self.training:
                # for now the token is the same for all the batch
                if isinstance(self.budget, float):
                    # fixed budget
                    self.current_budget = self.budget
                elif isinstance(self.budget, list) or isinstance(self.budget, tuple):
                    # select a random value from the possible budgets
                    idx = torch.randint(0, len(self.budget), (1,)).item()
                    self.current_budget = self.budget[idx]
                elif self.budget == True:
                    # sample a random budget in [0,1)
                    self.current_budget = torch.rand(1, device=x.device).item()
                elif self.budget == 'learnable':
                    # in this case we use a single learnable parameter and multiply it by a random budget
                    self.current_budget = torch.rand(1, device=x.device).item()
                    batch_budget_token_1 = self.learnable_budget_token_1.expand(n, -1, -1) 
                    x = torch.cat([x, self.current_budget * batch_budget_token_1], dim=1)
                    return x
                elif self.budget == 'learnable_interpolate':
                    # in this case we use two learnable parameters and interpolate between them with a random budget
                    self.current_budget = torch.rand(1, device=x.device).item()
                    batch_budget_token_1 = self.learnable_budget_token_1.expand(n, -1, -1) 
                    batch_budget_token_2 = self.learnable_budget_token_2.expand(n, -1, -1)
                    x = torch.cat([ x,
                                    self.current_budget * batch_budget_token_1 +
                                    (1-self.current_budget) * batch_budget_token_2], dim=1)
                    return x
                    
                
                budget_token = budget_token.fill_(self.current_budget)
                self.current_budget = budget_token.mean().item()
                x = torch.cat([x, budget_token], dim=1)
                return x
            
            else:
                assert self.current_budget is not None, 'Budget token not set. Call set_budget() before forward() to evaluate the model on a chosen budget.'

                
                if self.budget == 'learnable':
                    # in this case we use two learnable parameters and interpolate between them with a random budget
                    batch_budget_token_1 = self.learnable_budget_token_1.expand(n, -1, -1) 
                    x = torch.cat([x, self.current_budget * batch_budget_token_1], dim=1)
                    
                elif self.budget == 'learnable_interpolate':
                    # in this case we use two learnable parameters and interpolate between them with a random budget
                    batch_budget_token_1 = self.learnable_budget_token_1.expand(n, -1, -1) 
                    batch_budget_token_2 = self.learnable_budget_token_2.expand(n, -1, -1)
                    x = torch.cat([ x,
                                    self.current_budget * batch_budget_token_1 +
                                    (1-self.current_budget) * batch_budget_token_2], dim=1)
                    
                else:
                    budget_token = budget_token.fill_(self.current_budget)
                    self.current_budget = budget_token.mean().item()
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
        if self.budget:
            x = self._add_budget_token(x)
        
        # Pass through the encoder
        x = self.encoder(x)

        # Get all class tokens and average them
        pre_logits = x[:, 0:self.num_class_tokens]
        pre_logits = reduce(pre_logits, 'n c e -> n e', reduction='sum')
        logits = self.head(pre_logits)

        # Get other tokens, we should exclude clas, register and budget tokens
        tokens = x[:, self.num_class_tokens + self.num_registers:-self.num_budget_tokens,: ]

        # get mask of last residual layer
        mask = self.encoder.layers[-1].mask # (batch_size, sequence_len, 1)

        # Pass through the decoder
        reconstructed_images, reconstructed_images_mask = self.decoder(tokens, mask)


        return logits, reconstructed_images, reconstructed_images_mask


    def set_budget(self, budget: float):
        self.current_budget = budget

    