import torch
from torch import nn
import torch.nn.functional as F
import math
from typing import Optional, List
from abc import ABC
from pytorch3d.ops import knn_points
from einops.layers.torch import Reduce

from .residualvit import ResidualModule

from .blocks import SelfAttention, MLP, GumbelSoftmax
 
"""
Adaptive Point Cloud Transformer.
"""

#DropPredictor from dynamicViT
class DropPredictor(nn.Module):
    """ Computes the log-probabilities of dropping a token, adapted from PredictorLG here:
    https://github.com/raoyongming/DynamicViT/blob/48ac52643a637ed5a4cf7c7d429dcf17243794cd/models/dyvit.py#L287 """
    def __init__(self, embed_dim):
        super().__init__()

        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x, prev_decision):
        x = self.in_conv(x)
        B, N, C = x.size()
        local_x = x[:,:, :C//2]
        global_x = (x[:,:, C//2:] * prev_decision).sum(dim=1, keepdim=True) / (torch.sum(prev_decision, dim=1, keepdim=True)+1e-20)
        x = torch.cat([local_x, global_x.expand(B, N, C//2)], dim=-1)
        return self.out_conv(x)

# PCT Block
class AdaPTBlock(ResidualModule):
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
        self.drop_target = 0.0
        self.mask = None
        self.skip = 'temp'

        # Attention block
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.self_attention = SelfAttention(hidden_dim, num_heads, attention_dropout)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim=hidden_dim, mlp_dim=mlp_dim)

    def set_target(self, target):
        self.drop_target = target
        
    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor] = None, decision: Optional[torch.Tensor] = None) -> torch.Tensor:
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")

        x = self.ln_1(input)
        if mask is not None:
            # sorry for the name confusion. mask is of shape (B, N, N), and decision is of shape (B, N, 1), self.mask name is used in the losses
            self.mask = decision

            # repeat mask for all heads
            mask = mask.repeat(self.self_attention.num_heads,1,1)
            mask = torch.logical_not(mask)
            x = self.self_attention(x, mask)
        else:
            x = self.self_attention(x)
        x = x + input
        x = self.mlp(self.ln_2(x)) + x
        #x = self.dropout(x)
        #x = x + input

        #y = self.ln_2(x)
        #y = self.mlp(y)
        return x
    
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

        return x # B, N, out_channels

# AdaPT Encoder
class AdaPTEncoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
            self,
            num_layers: int,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float,
            attention_dropout: float,
            drop_layers: List[int] = None,
            drop_target: List[float] = None,
            num_class_tokens: int = 1,
    ):
        super().__init__()          
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.num_class_tokens = num_class_tokens
        self.gs = GumbelSoftmax(dim=-1, hard=True)
        self.masks = None
        self.layers = nn.ModuleList(
            [
                AdaPTBlock(
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.current_budget = drop_target
        if drop_layers is not None:
            self.drop_layers = drop_layers
            for i, l in enumerate(drop_layers):
                self.layers[l].set_target(drop_target**i)

            self.drop_predictors = nn.ModuleList([DropPredictor(hidden_dim) for _ in range(len(drop_layers))])

        self.warmup = 1

    def get_decisions(self, input:torch.Tensor, prev_decision:torch.Tensor, p:int):

        # Get decisions from drop predictors
        decisions = self.drop_predictors[p](input, prev_decision)

        return decisions

    def _build_mask(self, decisions: torch.Tensor):
        
        # always use class tokens
        decisions = torch.cat((torch.ones_like(decisions[:,0:1,0:1]), decisions),1)

        # build mask from decisions
        mask = (decisions*decisions.transpose(1,2) + torch.eye(decisions.shape[2],device=decisions.device)).bool()
        return mask

    def batch_index_select(self, x, idx):
        if len(x.size()) == 3:
            B, N, C = x.size()
            N_new = idx.size(1)
            offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
            idx = idx + offset
            out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
            return out
        elif len(x.size()) == 2:
            B, N = x.size()
            N_new = idx.size(1)
            offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
            idx = idx + offset
            out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
            return out
        else:
            raise NotImplementedError

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.dropout(input)
        B, N, C = x.shape

        mask = None
        prev_decision = torch.ones(B, N-1, 1, dtype=x.dtype, device=x.device)
        self.masks = []
        p = 0

        for i, layer in enumerate(self.layers):
            if i in self.drop_layers:

                # Detach class tokens
                if self.num_class_tokens > 0:
                    class_tokens = x[:, 0:self.num_class_tokens]
                    x = x[:, self.num_class_tokens:]

                # Get predictions from drop predictor
                soft_decision = self.get_decisions(x, prev_decision, p)
                p += 1
                
                # Slow warmup TODO
                keepall = torch.cat((torch.zeros_like(soft_decision[:,:,0:1]), torch.ones_like(soft_decision[:,:,1:2])),2) 
                soft_decision = soft_decision*self.warmup + keepall*(1-self.warmup)
                
                # Apply Gumbel-Softmax
                decision = self.gs(soft_decision)[:,:,1:2]
                
                # Update previous decisions
                decision = decision*prev_decision
                prev_decision = decision
                
                if self.training:
                    #update mask for tokens in training
                    mask = self._build_mask(decisions=decision)
                else:
                    #sample tokens at inference time
                    num_keep_tokens = int((1-layer.drop_target) * (N))
                    keep_policy = torch.argsort(soft_decision[:,:,1], dim=1, descending=True)[:, :num_keep_tokens]
                    x = self.batch_index_select(x, keep_policy)
                    #print(drop_target[p],num_keep_tokens, x.shape)
                    prev_decision = self.batch_index_select(prev_decision, keep_policy)

                # reattach class tokens
                if self.num_class_tokens > 0:
                    x = torch.cat([class_tokens, x], dim=1)

                self.masks.append(decision)

            x = layer(x, mask, decision)
        return x    
            

# Classification Head    
class Classf_head(nn.Module):
    def __init__(self, in_channels, n_classes) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.n_classes = n_classes
        self.lin1 = nn.Linear(in_channels, in_channels//2)
        self.lin2 = nn.Linear(in_channels//2, n_classes)
        self.bn1 = nn.BatchNorm1d(in_channels//2)
        self.dp = nn.Dropout(0.5)

    def forward(self, x):
        
        x = F.gelu(self.bn1(self.lin1(x)))
        x = self.lin2(self.dp(x))
        return x


class AdaptivePointCloudTransformer(nn.Module):

    def __init__(
            self,
            num_points: int,
            num_layers: int,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
            num_classes: int = 40,
            representation_size: Optional[int] = None,
            num_registers: int = 0,
            num_class_tokens: int = 1,
            torch_pretrained_weights: Optional[str] = None,
            drop_layers: Optional[List[int]] = None,
            drop_target: Optional[float] = None,
            warmup_start: int = 0,
            warmup_steps: int = 50,
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
        self.num_layers = num_layers
        self.num_points = num_points
        self.drop_layers = drop_layers
        self.current_budget = drop_target
        self.warmup_start = warmup_start
        self.warmup_steps = warmup_steps
        self.masks = None

        # Embedder
        self.embedder = ARPE(in_channels=3, out_channels=hidden_dim, npoints=num_points)
        
        # Add class tokens
        if num_class_tokens > 0:    
            self.class_tokens = nn.Parameter(torch.zeros(1, num_class_tokens, hidden_dim))

        # Add registers
        if num_registers > 0:
            self.registers = nn.Parameter(torch.zeros(1, num_registers, hidden_dim))
        
        # PCT Encoder
        self.encoder = AdaPTEncoder(
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            drop_layers=drop_layers,
            drop_target=drop_target,
        )

        # Classification Head
        self.head = Classf_head(hidden_dim, num_classes)

        if torch_pretrained_weights is not None:
            from .adapters import adapt_torch_state_dict
            torch_pretrained_weights = eval(torch_pretrained_weights).get_state_dict()
            adapted_state_dict = adapt_torch_state_dict(torch_pretrained_weights, num_classes=num_classes)
            self.load_state_dict(adapted_state_dict, strict=False)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        torch._assert(x.dim() == 3, f"Expected (batch_size, num_points, channels) got {x.shape}")
        
        x = self.embedder(x)
        
        return x # B, N, hidden_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Embedding
        x = self._process_input(x)
        b = x.shape[0]

        # Add registers
        if self.num_registers > 0:
            x = torch.cat([self.registers.expand(b, -1, -1), x], dim=1)

        # Add class tokens
        if self.num_class_tokens > 0:
            x = torch.cat([self.class_tokens.expand(b, -1, -1), x], dim=1)

        # Pass through PCT Encoder
        x = self.encoder(x)

        if self.num_class_tokens > 0:
            # Sum class tokens (?)
            x = x[:, 0:self.num_class_tokens]
            x = torch.sum(x, dim=1)
        else:
            # Average pooling
            x = torch.mean(x, dim=1)

        # Classification Head
        x = self.head(x)

        return x
    
    def set_budget(self, budget: float):
        self.current_budget = budget
        