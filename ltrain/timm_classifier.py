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

from patch.timm_custom_patch import apply_patch_tome_encoder, apply_patch_tome_decoder, apply_patch_tome_classifier

class MAEVisionTransformer(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):

        super().__init__()

        self.mae_encoder = timm.create_model('deit_tiny_patch16_224', pretrained=True) #VisionTransformer(**kwargs)
        self.mae_classifier = timm.create_model('deit_tiny_patch16_224', pretrained=True) #VisionTransformer(**kwargs)

        # Get the number of blocks to skip or keep
        n_blocks = len(list(cfg.encoder.r))
        total_blocks = len(self.mae_encoder.blocks)
        # Finalize the first part
        self.mae_encoder.blocks = self.mae_classifier.blocks[:n_blocks]
        
        # Make classifier
        self.mae_classifier.blocks = self.mae_classifier.blocks[n_blocks:]

        # Create the patch autoencoder
        apply_patch_ecn_classifier(self.mae_encoder, self.mae_classifier)

        # Create encoder tome patch 
        apply_patch_tome_encoder(self.mae_encoder, trace_source=True, prop_attn=True)
        apply_patch_tome_classifier(self.mae_classifier, prop_attn=False)

        if type(cfg.encoder.r)==int:
            self.mae_encoder.r = cfg.encoder.r
        else: 
            self.mae_encoder.r = list(cfg.encoder.r)

        self.patch_size = 16

        in_chans, decoder_embed_dim = 3, 192
        # Last part of the classifier network
        self.head = nn.Linear(decoder_embed_dim, cfg.dataset.num_classes)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)


        self.norm_pix_loss = False
        # Classifier loss
        self.classifier_loss = instantiate(cfg.loss.classification_loss)
        print()

    def forward(self, imgs: torch.Tensor, labels: torch.Tensor, return_pred_labels: bool = False):
        # imgs: [N, 3, H, W]
        
        tokens = self.mae_encoder(imgs)
        
        # Image classification
        cls_tokens = self.mae_classifier(tokens)
        class_preds = self.head(cls_tokens)
        
        # Classification Loss
        classification_loss = self.classifier_loss(class_preds, labels)
        if return_pred_labels==True:
            return classification_loss, class_preds
        
        return classification_loss
    
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
        def forward_features(self, x):
            x = self.patch_embed(x)
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            if self.dist_token is None:
                x = torch.cat((cls_token, x), dim=1)
            else:
                x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = self.pos_drop(x + self.pos_embed)
            x = self.blocks(x)
            x = self.norm(x)
            # if self.dist_token is None:
            #     return self.pre_logits(x[:, 0])
            # else:
            #     return x[:, 0], x[:, 1]
            return x

    return VisionTransformerEncoder


def make_vision_classifier(transformer_class):
    class VisionTransformerClassifier(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """
        
        def forward_features(self, x: torch.Tensor) -> torch.Tensor:
            x = self.blocks(x)
            x = self.norm(x)
            return x
        
        def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        
            # Global pooling
            return x[:, 0]  # class token

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.forward_features(x)
            x = self.forward_head(x)
            return x

    return VisionTransformerClassifier

    
def apply_patch_ecn_classifier(
    model_encoder: VisionTransformer,
    model_classifier: VisionTransformer,
):
    
    
    VisionTransformerEncoder = make_vision_encoder(model_encoder.__class__)
    model_encoder.__class__ = VisionTransformerEncoder

    VisionTransformerClassifier = make_vision_classifier(model_classifier.__class__)
    model_classifier.__class__ = VisionTransformerClassifier

   
    
   
