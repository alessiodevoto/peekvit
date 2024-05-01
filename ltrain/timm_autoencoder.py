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

from patch.timm_custom_patch import apply_patch_tome_encoder, apply_patch_tome_decoder

class MAEVisionTransformer(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):

        super().__init__()

        self.mae_encoder = timm.create_model('deit_tiny_patch16_224', pretrained=True) #VisionTransformer(**kwargs)
        self.mae_decoder = timm.create_model('deit_tiny_patch16_224', pretrained=True) #VisionTransformer(**kwargs)

        # Create the patch autoencoder
        apply_patch_autoencoder(self.mae_encoder, self.mae_decoder)

        # Create encoder tome patch 
        apply_patch_tome_encoder(self.mae_encoder, trace_source=True, prop_attn=True)
        apply_patch_tome_decoder(self.mae_decoder, prop_attn=False)

        if type(cfg.encoder.r)==int:
            self.mae_encoder.r = cfg.encoder.r
        else: 
            self.mae_encoder.r = list(cfg.encoder.r)

        self.patch_size = 16

        in_chans, decoder_embed_dim = 3, 192
        self.decoder_norm = torch.nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.patch_size**2 * in_chans, bias=True) # decoder to patch
        

        self.norm_pix_loss = False
        print()

    def forward(self, imgs: torch.Tensor, return_pred_images: bool = False):
        # imgs: [N, 3, H, W]
        
        tokens = self.mae_encoder(imgs)
        self.mae_decoder._tome_info["layer_source"] = self.mae_encoder._tome_info["layer_source"]

        tokens = self.mae_decoder(tokens)

        # Final steps of decoder
        tokens = self.decoder_norm(tokens)
        pred = self.decoder_pred(tokens)
        
        #loss = self.forward_loss_targeted(imgs, pred, source=self.mae_encoder._tome_info["source"])
        
        loss = self.forward_loss(imgs, pred)

        if return_pred_images==False:
            return loss
        else:
            return loss, self.unpatchify(pred)
    
    def forward_loss_targeted(self, imgs, pred, source):
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

        source = source[:, :, 1:]
        b, m, n = source.shape

        loss = 0
        for img in range(b):
            img_source = source[img]
            for row in range(m):
                idx = torch.where(img_source[row] != 0)[0]
                if len(idx)>0:
                    img_loss = ((pred[img, idx] - target[img, idx])**2) * idx.shape[0]
                    img_loss = img_loss.mean(dim=-1).mean()
                    loss = loss + img_loss
        loss = loss / b
        # loss = (pred - target) ** 2
        # loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        
        # loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        # loss = loss.mean()
        
        return loss


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
def make_vision_encoder(transformer_class):
    class VisionTransformerEncoder(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.forward_features(x)
            
            # 1. Avoid pooling, so that we can get the tokens as output of encoder
            # x = self.forward_head(x)
            return x

    return VisionTransformerEncoder

def make_vision_decoder(transformer_class):
    class VisionTransformerDecoder(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """
        # def forward(self, *args, **kwdargs) -> torch.Tensor:
        #     # self._tome_info["r"] = parse_r(len(self.blocks), self.r)
        #     # self._tome_info["size"] = None
        #     # self._tome_info["source"] = None

        #     return 
        
        def forward_features(self, x: torch.Tensor) -> torch.Tensor:
            #x = self.patch_embed(x)
            # Get rid of CLS token
            # x = self._pos_embed(x)
            x = self.patch_drop(x)
            x = self.norm_pre(x)
            x = self.blocks(x)
            x = self.norm(x)
            return x

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.forward_features(x)
            
            # 1. Avoid pooling, so that we can get the tokens as output of encoder
            # x = self.forward_head(x)
            
            # Get rid of CLS token
            x = x[:, 1:]
            return x

    return VisionTransformerDecoder

def apply_patch_autoencoder(
    model_encoder: VisionTransformer,
    model_decoder: VisionTransformer,
):
    
    
    VisionTransformerEncoder = make_vision_encoder(model_encoder.__class__)
    model_encoder.__class__ = VisionTransformerEncoder

    VisionTransformerDecoder = make_vision_decoder(model_decoder.__class__)
    model_decoder.__class__ = VisionTransformerDecoder
    
    # We need to set no_embed_class to True to avoid CLS token
    model_decoder.cls_token = None
    # model_decoder.pos_embed = model_decoder.pos_embed[:,1:] 
