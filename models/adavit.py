import torch
from torch import nn
import torch.nn.functional as F
import math
from typing import Optional, List
from abc import ABC
from torchvision.models.vision_transformer import ViT_B_16_Weights, ViT_B_32_Weights

import numpy as np
from .blocks import SelfAttention, MLP
from torch.autograd import Variable
 
"""
Adaptive Vision Transformer (AViT) as per https://arxiv.org/pdf/2112.07658.pdf.
"""

"""def get_distribution_target(mode='gaussian', length=12, max=1, standardized=True, target_depth=8, buffer=0.02):
    # this gets the distributional target to regularize the ACT halting scores towards
    if mode == 'gaussian':
        from scipy.stats import norm
        # now get a serios of length
        data = np.arange(length)
        data = norm.pdf(data, loc=target_depth, scale=1)

        if standardized:
            print('\nReshaping distribution to be top-1 sum 1 - error at {}'.format(buffer))
            scaling_factor = (1.-buffer) / sum(data[:target_depth])
            data *= scaling_factor

        return data"""


# ViT Block
class AViTBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        gate_scale: float = 10,
        gate_center: float = 30,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.gate_scale = gate_scale
        self.gate_center = gate_center
        
        
        # Attention block
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.self_attention = SelfAttention(hidden_dim, num_heads, attention_dropout)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim=hidden_dim, mlp_dim=mlp_dim)


    
    def forward_act(self, x, mask=None):

        debug=False
        analyze_delta = True
        bs, token, dim = x.shape

        # x is bs, seq_len, token_dim
        # mask is bs, seq_len
        # print('Input shape', x.shape)
        # print('Mask shape', mask.shape)
        # print('Mask', 1-mask)

        if mask is None:
            x = x + self.self_attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        else:
            x = x + self.self_attention(self.ln_1(x*(1-mask).view(bs, token, 1))*(1-mask).view(bs, token, 1)) #, mask=mask)
            x = x + self.mlp(self.ln_2(x*(1-mask).view(bs, token, 1))*(1-mask).view(bs, token, 1))

        
        gate_scale, gate_center = self.gate_scale, self.gate_center
        halting_score_token = torch.sigmoid(x[:,:,0] * gate_scale - gate_center)
        # initially first position used for layer halting, second for token
        # now discarding position 1
        halting_score = [-1, halting_score_token]
        

        return x, halting_score


# ViT Encoder
class AViTEncoder(nn.Module):

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        eps: float = 0.01,
        gate_scale: float = 10,
        gate_center: float = 30,
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.eps = eps
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: List = []
        for i in range(num_layers):
            layers.append(AViTBlock(
                            num_heads,
                            hidden_dim,
                            mlp_dim,
                            dropout,
                            attention_dropout,
                            gate_scale,
                            gate_center
                            ))

        self.layers = nn.ModuleList(layers)
        self.ln = nn.LayerNorm(hidden_dim)

        # for token act part
        self.c_token = None
        self.R_token = None
        self.mask_token = None
        self.rho_token = None
        self.counter_token = None
        self.seq_length = seq_length

        


    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        input = self.dropout(input)

        
        return self.forward_features_act_token(input)
    

    def forward_features_act_token(self, x):
        

        # now start the act part
        bs = x.size()[0]  # The batch size


        if self.c_token is None or bs != self.c_token.size()[0]:
            self.c_token = Variable(torch.zeros(bs, self.seq_length).cuda())
            self.R_token = Variable(torch.ones(bs, self.seq_length).cuda())
            self.mask_token = Variable(torch.ones(bs, self.seq_length).cuda())
            self.rho_token = Variable(torch.zeros(bs, self.seq_length).cuda())
            self.counter_token = Variable(torch.ones(bs, self.seq_length).cuda())

        c_token = self.c_token.clone()
        R_token = self.R_token.clone()
        mask_token = self.mask_token.clone()
        self.rho_token = self.rho_token.detach() * 0.
        self.counter_token = self.counter_token.detach() * 0 + 1.
        # Will contain the output of this residual layer (weighted sum of outputs of the residual blocks)
        output = None
        # Use out to backbone
        out = x

        
        self.halting_score_layer = []

        for i, adaptive_layer in enumerate(self.layers):

            # block out all the parts that are not used
            out.data = out.data * mask_token.float().view(bs, self.seq_length, 1)

            # evaluate layer and get halting probability for each sample
            # block_output, h_lst = l.forward_act(out)    # h is a vector of length bs, block_output a 3D tensor
            block_output, h_lst = adaptive_layer.forward_act(out, 1.-mask_token.float())    # h is a vector of length bs, block_output a 3D tensor

            self.halting_score_layer.append(torch.mean(h_lst[1][1:]))

            out = block_output.clone()              # Deep copy needed for the next layer

            _, h_token = h_lst # h is layer_halting score, h_token is token halting score, first position discarded

            # here, 1 is remaining, 0 is blocked
            block_output = block_output * mask_token.float().view(bs, self.seq_length, 1)

            # Is this the last layer in the block?
            if i==len(self.layers)-1:
                h_token = nn.Parameter(torch.ones(bs, self.seq_length).cuda())

            # for token part
            c_token = c_token + h_token
            self.rho_token = self.rho_token + mask_token.float()

            # Case 1: threshold reached in this iteration
            # token part
            reached_token = c_token > 1 - self.eps
            reached_token = reached_token.float() * mask_token.float()
            delta1 = block_output * R_token.view(bs, self.seq_length, 1) * reached_token.view(bs, self.seq_length, 1)
            self.rho_token = self.rho_token + R_token * reached_token

            # Case 2: threshold not reached
            # token part
            not_reached_token = c_token < 1 - self.eps
            not_reached_token = not_reached_token.float()
            R_token = R_token - (not_reached_token.float() * h_token)
            delta2 = block_output * h_token.view(bs, self.seq_length, 1) * not_reached_token.view(bs, self.seq_length, 1)

            self.counter_token = self.counter_token + not_reached_token # These data points will need at least one more layer

            # Update the mask
            mask_token = c_token < 1 - self.eps

            if output is None:
                output = delta1 + delta2
            else:
                output = output + (delta1 + delta2)

        x = self.ln(output)

        return x
        




class AdaptiveVisionTransformer(nn.Module):
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
        eps: float = 0.01,
        gate_scale: float = 10,
        gate_center: float = 30,
        torch_pretrained_weights: Optional[str] = None,
        timm_pretrained_weights: Optional[List] = None,
    ):
        """
        Args:
            image_size (int): The size of the input image.
            patch_size (int): The size of each patch in the image.
            num_layers (int): The number of layers in the transformer encoder.
            num_heads (int): The number of attention heads in the transformer encoder.
            hidden_dim (int): The hidden dimension size in the transformer encoder.
            mlp_dim (int): The dimension size of the feed-forward network in the transformer encoder.
            dropout (float, optional): The dropout rate. Defaults to 0.0.
            attention_dropout (float, optional): The dropout rate for attention weights. Defaults to 0.0.
            num_classes (int, optional): The number of output classes. Defaults to 1000.
            representation_size (int, optional): The size of the output representation. Defaults to None.
            num_registers (int, optional): The number of register tokens to be added. Defaults to 0.
            num_class_tokens (int, optional): The number of class tokens to be added. Defaults to 1.
            eps (float, optional): The epsilon value for the ACT. Defaults to 0.01.
            gate_scale (float, optional): The scale value for the ACT. Defaults to 10.
            gate_center (float, optional): The center value for the ACT. Defaults to 30.
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
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.num_heads = num_heads
        self.num_registers = num_registers
        self.num_class_tokens = num_class_tokens
        self.num_layers = num_layers
        self.eps = eps
        self.gate_scale = gate_scale
        self.gate_center = gate_center
        


        self.conv_proj = nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size)

        seq_length = (image_size // patch_size) ** 2

        # Add class tokens
        self.class_tokens = nn.Parameter(torch.zeros(1, num_class_tokens, hidden_dim))
        seq_length += num_class_tokens

        # Add registers
        if num_registers > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, num_registers, hidden_dim))
            seq_length += num_registers

        self.encoder = AViTEncoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            eps,
            gate_scale,
            gate_center
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
            print('Loading torch pretrained weights: ', torch_pretrained_weights)
            from .adapters import adapt_torch_state_dict
            torch_pretrained_weights = eval(torch_pretrained_weights).get_state_dict(progress=False)
            adapted_state_dict = adapt_torch_state_dict(torch_pretrained_weights, num_classes=num_classes)
            self.load_state_dict(adapted_state_dict, strict=False)
        elif timm_pretrained_weights is not None:
            print('Loading timm pretrained weights: ', timm_pretrained_weights)
            from .adapters import adapt_timm_state_dict
            model = torch.hub.load(timm_pretrained_weights[0], timm_pretrained_weights[1], pretrained=True)
            timm_pretrained_weights = model.state_dict()
            adapted_state_dict = adapt_timm_state_dict(timm_pretrained_weights, num_classes=num_classes)
            self.load_state_dict(adapted_state_dict, strict=False)
            del model




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

    
    