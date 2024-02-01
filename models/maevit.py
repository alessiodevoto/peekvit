import math
from typing import List, Optional
import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

from .vit import ViTBlock



class TokenShuffle(torch.nn.Module):
    """
    Token Shuffle Layer.
    Given a sequence of shape (batch_size, seq_length, hidden_dim), this layer shuffles each sequence in the batch randomly.
    """

    def __init__(self, mask_ratio: float):
        super().__init__()
        self.mask_ratio = mask_ratio

         # from BERT

    

    def forward(self, input: torch.Tensor):
            """
            Forward pass of the Maevit model.

            Args:
                input (torch.Tensor): Input tensor of shape (batch_size, seq_length, hidden_dim).

            Returns:
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing the shuffled input tensor,
                forward permutation tensor, and backward permutation tensor.
            """
            
            torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
            batch_size, seq_length, hidden_dim = input.shape

            # number of tokens to mask
            num_mask_tokens = int(self.mask_ratio * seq_length)
            
            # we should shuffle tokens inside each sequence in the batch
            # for now we shuffle all the tokens according to the same perm
            forward_perm = torch.randperm(seq_length, device=input.device)
            backward_perm = torch.argsort(forward_perm)

            # shuffle tokens
            input = input[:, forward_perm, :]

            # remove tokens
            input = input[:, :-num_mask_tokens, :]

            return input, forward_perm, backward_perm


# ViT Encoder
class MAEViTEncoder(nn.Module):
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



class MAEVisionTransformerEncoder(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        mask_ratio: float,
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

        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT

        # Add class tokens
        self.class_tokens = nn.Parameter(torch.zeros(1, num_class_tokens, hidden_dim))
        seq_length += num_class_tokens

        # Add registers
        if num_registers > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, num_registers, hidden_dim))
            seq_length += num_registers

        self.encoder = MAEViTEncoder(
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
        
        if mask_ratio > 0.0:
            self.token_shuffle = TokenShuffle(mask_ratio)

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

        

    def forward(self, input: torch.Tensor):
        
        # Reshape and permute the input tensor
        x = self._process_input(input)

        x = x + self.pos_embedding
        n = x.shape[0]

        # Shuffle tokens
        forward_perm = None
        backward_perm = None
        if self.mask_ratio > 0.0 and self.training:
            x, forward_perm, backward_perm = self.token_shuffle(x)


        # Add registers
        if self.num_registers > 0:
            batch_register_tokens = self.register_tokens.expand(n, -1, -1)
            x = torch.cat([batch_register_tokens, x], dim=1)
        
        # Expand the class token to the full batch
        batch_class_tokens = self.class_tokens.expand(n, -1, -1)
        x = torch.cat([batch_class_tokens, x], dim=1)

        # Pass through the encoder
        x = self.encoder(x)

        # class tokens and head
        class_tokens = x[:, 0:self.num_class_tokens].sum(dim=1)
        logits = self.head(class_tokens)

        register_tokens = x[:, self.num_class_tokens:self.num_class_tokens+self.num_registers] if self.num_registers > 0 else None
        x = x[:, self.num_class_tokens+self.num_registers:]

        return logits, x, forward_perm, backward_perm

class MAEVisionTransformerDecoder(torch.nn.Module):
    def __init__(self,
                image_size: int,
                patch_size: int,
                decoder_hidden_dim: int,
                decoder_mlp_dim: int, 
                seq_length: int,
                num_decoder_layers: int,
                num_decoder_heads: int,
                decoder_dropout: float,
                decoder_attention_dropout: float,
                ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, decoder_hidden_dim))
        self.pos_embedding = torch.nn.Parameter(torch.empty(1, seq_length-1, decoder_hidden_dim).normal_(std=0.02))
        self.image_size = image_size
        self.patch_size = patch_size

        self.encoder = MAEViTEncoder(
            seq_length,
            num_decoder_layers,
            num_decoder_heads,
            decoder_hidden_dim,
            decoder_mlp_dim,
            decoder_dropout,
            decoder_attention_dropout
            )

        self.head = torch.nn.Linear(decoder_hidden_dim, 3 * patch_size ** 2)
        self.patch2img = Rearrange('b (h w) (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size)


    def forward(self, tokens, backward_indices=None, mask=None):
        
        batch, seq_length, hidden_dim = tokens.shape
        
        assert backward_indices or mask, "Either backward_indices or mask must be provided"

        if backward_indices:
            # this means we have received a subset of tokens, as the others have been dropped.
            # we need to add a class token and positional embeddings
            # the way we do it is by adding a mask token and then undo the shuffling of the tokens

            # Add mask tokens
            num_missing_tokens = backward_indices.shape[0] - seq_length
            batch_mask_tokens = self.mask_token.expand(batch, num_missing_tokens, -1)
            tokens = torch.cat([tokens, batch_mask_tokens], dim=1)


            # Undo the shuffling
            tokens = tokens[:, backward_indices, :]

            # Add positional embeddings
            tokens += self.pos_embedding
        else:
            # this means we have received a mask
            # the mask represents the tokens that have been dropped
            # we should replace the dropped tokens with the mask token
            # and add positional embeddings
            # mask has shape (batch_size, seq_length)
            # tokens has shape (batch_size, seq_length, hidden_dim)
            # mask_token has shape (1, 1, hidden_dim)
            # pos_embedding has shape (1, seq_length-1, hidden_dim)

            # Replace masked tokens with mask token
            tokens = tokens.masked_fill(mask.unsqueeze(-1), self.mask_token)
            
            

        # Pass through the encoder
        tokens = self.encoder(tokens)

        # We should recover the original image from the tokens
        # (batch_size, seq_length, hidden_dim) -> (batch_size, 3, image_size, image_size)
        tokens = self.head(tokens)
        img = self.patch2img(tokens)

        return img


class MAEVisionTransformer(torch.nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        mask_ratio: float,
        decoder_hidden_dim: int,
        decoder_mlp_dim: int, 
        num_decoder_layers: int,
        num_decoder_heads: int,
        decoder_dropout: float,
        decoder_attention_dropout: float,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        num_registers: int = 0,
        num_class_tokens: int = 1,
        torch_pretrained_weights: Optional[str] = None,
    ):

        super().__init__()

        self.mae_encoder = MAEVisionTransformerEncoder(
            image_size,
            patch_size,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            mask_ratio,
            dropout,
            attention_dropout,
            num_classes,
            representation_size,
            num_registers,
            num_class_tokens,
            torch_pretrained_weights,
        )

        

        self.mae_decoder = MAEVisionTransformerDecoder(
            image_size,
            patch_size,
            decoder_hidden_dim,
            decoder_mlp_dim, 
            self.mae_encoder.seq_length,
            num_decoder_layers,
            num_decoder_heads,
            decoder_dropout,
            decoder_attention_dropout,
        )

    def forward(self, input: torch.Tensor):
        logits, tokens, forward_perm, backward_perm = self.mae_encoder(input)

        img = self.mae_decoder(tokens, backward_perm)
        return logits, img



"""if __name__ == '__main__':
    model = MAEVisionTransformer(
    image_size=32,
    patch_size=4,
    num_layers=4,
    num_heads=4,
    hidden_dim=96,
    mlp_dim=128,
    mask_ratio=0.75,
    decoder_hidden_dim=96,
    decoder_mlp_dim=128,
    num_decoder_layers=4,
    num_decoder_heads=4,
    decoder_dropout=0.0,
    decoder_attention_dropout=0.0,
    dropout=0.1,
    attention_dropout=0.0,
    num_classes=10,
    representation_size=None,
    num_registers=0,
    num_class_tokens=1,
    torch_pretrained_weights=None,
    )



    input = torch.randn(1, 3, 32, 32)
    logits, img = model(input)"""