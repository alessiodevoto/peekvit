import math
from typing import List, Optional
import torch
from torch import nn
import timm

from einops import rearrange
from einops.layers.torch import Rearrange

#from peekvit.models.vit import ViTBlock
from hydra.utils import instantiate
import timm
class MAEVisionTransformer(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):

        super().__init__()

        self.mae_encoder = instantiate(cfg.encoder) #VisionTransformer(**kwargs)
        self.mae_decoder = instantiate(cfg.decoder) #VisionTransformer(**kwargs)

        

        self.patch_size = self.mae_encoder.patch_size

        in_chans, decoder_embed_dim = 3, self.mae_decoder.hidden_dim
        self.decoder_norm = torch.nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.patch_size**2 * in_chans, bias=True) # decoder to patch
        

        self.norm_pix_loss = False
        print()

    def forward(self, imgs: torch.Tensor, return_pred_images: bool = False):
        
        
        tokens = self.mae_encoder(imgs)

        tokens = self.mae_decoder(tokens)

        # Final steps of decoder
        tokens = self.decoder_norm(tokens)
        pred = self.decoder_pred(tokens)
        loss = self.forward_loss(imgs, pred)
        if return_pred_images==False:
            return loss
        else:
            return loss, self.unpatchify(pred)
        

    def forward_loss(self, imgs, pred):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        # mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        
        #loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        loss = loss.mean()
        
        return loss
    
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size #self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x
    
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

from timm.models.vision_transformer import Attention, Block, VisionTransformer

from tome.utils import parse_r


def make_tome_class(transformer_class):
    class PoolVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            # self._tome_info["r"] = parse_r(len(self.blocks), self.r)
            self._tome_info["size"] = None
            self._tome_info["source"] = None

            return super().forward(*args, **kwdargs)

    return PoolVisionTransformer



def encoder_patch(
    model, trace_source: bool = False, prop_attn: bool = True
):
    """
    Applies ToMe to the first half of the transformer blocks.
    Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For training and for evaluating MAE models off the shelf set this to be False.
    """
    ToMeVisionTransformer = make_tome_class(model.__class__)

    model.__class__ = ToMeVisionTransformer
    model.r = 0
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        # "class_token": model.cls_token is not None,
        "distill_token": False,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    # Get the total number of blocks
    #total_blocks = len([m for m in model.modules() if isinstance(m, Block)])
    #half_blocks = total_blocks // 2

    # Counter for the number of blocks processed
    #block_counter = 0

    # for module in model.modules():
    #     if isinstance(module, Block):
    #         if block_counter < half_blocks:
    #             # Apply ToMeBlock only to the first half of blocks
    #             module.__class__ = ToMeBlock
    #             module._tome_info = model._tome_info
    #         if block_counter == half_blocks-1:
    #             module.__class__ = ToMeBlockMLP
    #         else:
    #             module.__class__ = ToMeBlocknoMerge
    #             module._tome_info = model._tome_info
    #         block_counter += 1
    #     elif isinstance(module, Attention):
    #         module.__class__ = ToMeAttention