import torch
from torch import nn
import torch.nn.functional as F
import math
from typing import Optional, List, Union
# from pytorch3d.ops import knn_points
from einops.layers.torch import Reduce



from .blocks import SelfAttention, MLP
 
"""
Plain Point Cloud Transformer.
"""


# PCT Block
class RankingPCTBlock(nn.Module):
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
    
        # Sort tokens
        self.sort = False
        self.current_budget = 1.0
    
    @staticmethod
    def sort_tokens(input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        
        class_token = input[:, 0:1, :]
        input = input[:, 1:, :]

        # get token magnitudes
        # (batch_size, seq_length, hidden_dim) -> (batch_size, seq_length)
        token_magnitudes = torch.norm(input, dim=-1)
        
        # get sorted indices 
        # (batch_size, seq_length) -> (batch_size, seq_length, 1)
        sorted_indices = torch.argsort(token_magnitudes, dim=-1, descending=True)
        sorted_indices = sorted_indices.unsqueeze(-1)

        # gather sorted input
        sorted_input = torch.gather(input, dim=1, index=sorted_indices.expand(-1, -1, input.shape[-1]))

        sorted_input = torch.cat([class_token, sorted_input], dim=1)
        return sorted_input
    

    def mask_tokens(self, input: torch.Tensor):
        if not self.training or not self.sort:
            return input

        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        # we assume input is sorted in descending order
        # the last n% tokens are masked by setting them to 0
        # where n is the current budget

        class_token = input[:, 0:1, :]
        input = input[:, 1:, :]

        # get the number of tokens to mask
        # (batch_size, seq_length) -> (batch_size, seq_length)
        num_tokens_to_keep = math.ceil(input.shape[1] * self.current_budget)

        
        # mask the tokens
        # (batch_size, seq_length) -> (batch_size, seq_length, 1)
        mask = torch.zeros_like(input)
        mask[:, :num_tokens_to_keep, :] = 1
        
        # apply the mask
        # (batch_size, seq_length, hidden_dim) -> (batch_size, seq_length, hidden_dim)
        masked_input = input * mask

        masked_input = torch.cat([class_token, masked_input], dim=1)

        return masked_input
    

    def drop_tokens(self, input: torch.Tensor):
        if self.training or not self.sort:
            return input
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        # we assume input is sorted in descending order
        # the last n% tokens are masked by setting them to 0
        # where n is the current budget

        num_tokens_to_keep = math.ceil(input.shape[1] * self.current_budget)

        return input[:, :num_tokens_to_keep, :]
    

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")

        if self.sort:
            input = self.sort_tokens(input)
        
        # notice that both mask_tokens and drop_tokens are implemented as no-ops if sort_tokens is False
        input = self.mask_tokens(input) # only has effect during training
        input = self.drop_tokens(input) # only has effect during evaluation

        #x = self.ln_1(input)  
        #x = self.mask_tokens(x)      
        #x = self.self_attention(x)
        #x = self.dropout(x)
        #x = x + input

        #y = self.ln_2(x)
        #y = self.mask_tokens(y)
        #y = self.mlp(y)

        x = self.ln_1(input)
        x = self.mask_tokens(x)
        x = self.self_attention(x) + x
        x = self.mlp(self.mask_tokens(self.ln_2(x))) + x


        return x# + y
    

    def set_budget(self, budget: float):
        self.current_budget = budget
        
    
    
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
                RankingPCTBlock(
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


class RankPointCloudTransformer(nn.Module):

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

        # Embedder
        self.embedder = ARPE(in_channels=3, out_channels=hidden_dim, npoints=num_points)
        
        # Add class tokens

        self.class_tokens = nn.Parameter(torch.zeros(1, num_class_tokens, hidden_dim))

        # Add registers
        if num_registers > 0:
            self.registers = nn.Parameter(torch.zeros(1, num_registers, hidden_dim))
        
        # PCT Encoder
        self.encoder = PCTEncoder(
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
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
        #x = torch.cat([self.class_tokens.expand(b, -1, -1), x], dim=1)

        # Pass through PCT Encoder
        x = self.encoder(x)

        # Sum class tokens (?)
        #x = x[:, 0:self.num_class_tokens]
        #x = torch.sum(x, dim=1)

        #TEST, Average pooling
        x = torch.mean(x, dim=1)

        # Classification Head
        x = self.head(x)

        return x

    def enable_ranking(self, sort_tokens: Union[bool, List[bool]] = False):
        """
        Enable ranking for the RankVit model.

        Args:
            sort_tokens (Union[bool, List[bool]], optional): 
                A boolean value or a list of boolean values indicating whether to sort tokens for each RankVit block. 
                If a single boolean value is provided, it will be applied to all RankVit blocks. 
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