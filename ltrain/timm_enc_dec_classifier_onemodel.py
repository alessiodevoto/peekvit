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
        
        
        # First part of the network
        self.mae_encoder = timm.create_model('deit_tiny_patch16_224', pretrained=True) #VisionTransformer(**kwargs)
        
        # Second part of the network
        self.mae_decoder_classifier = timm.create_model('deit_tiny_patch16_224', pretrained=True) #VisionTransformer(**kwargs)

        # Get the number of blocks to skip or keep
        total_blocks = len(self.mae_encoder.blocks)
        n_blocks = len(list(cfg.encoder.r))
        # Preserve additional blocks for refinement of reconstruction 
        preserve_n_blocks = 2

        # Ensure n_blocks + preserve_n_blocks is less than the total number of blocks
        assert n_blocks + preserve_n_blocks < len(self.mae_encoder.blocks), "The number of blocks to keep and preserve is greater than the total number of blocks"

        # Finalize the first part
        self.mae_encoder.blocks = self.mae_encoder.blocks[:n_blocks]
        
        # Finalize the second part
        
        self.mae_decoder_classifier.blocks = self.mae_decoder_classifier.blocks[total_blocks - n_blocks:]


        # Create the patch autoencoder
        apply_patch_ecn_dec_classifier(self.mae_encoder,self.mae_decoder_classifier)

        # Create encoder tome patch 
        apply_patch_tome_encoder(self.mae_encoder, trace_source=True, prop_attn=True)
        apply_patch_tome_decoder(self.mae_decoder_classifier, prop_attn=False)

        if type(cfg.encoder.r)==int:
            self.mae_encoder.r = cfg.encoder.r
        else: 
            self.mae_encoder.r = list(cfg.encoder.r)
        
        self.use_trace_loss = cfg.decoder.use_trace_loss

        self.patch_size = 16

        # Last part of the decoder network
        in_chans, decoder_embed_dim = 3, 192
        self.decoder_norm = torch.nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.patch_size**2 * in_chans, bias=True) # decoder to patch
        
        # Last part of the classifier network
        self.head = nn.Linear(decoder_embed_dim, cfg.dataset.num_classes)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)


        self.norm_pix_loss = False
        # Classifier loss
        self.classifier_loss = instantiate(cfg.loss.classification_loss)

        print()


    def forward(self, imgs: torch.Tensor, labels: torch.Tensor, return_pred_images: bool = False, return_pred_labels=False):
        # imgs: [N, 3, H, W]
        
        tokens = self.mae_encoder(imgs)
        self.mae_decoder_classifier._tome_info["layer_source"] = self.mae_encoder._tome_info["layer_source"]

        trace = self.mae_encoder._tome_info["layer_source"].copy()
        if self.use_trace_loss == True:
            H = torch.matmul(trace[0].permute(0,2,1), trace[1].permute(0,2,1))
            for trace_idx in range(2, len(trace)):
                H = torch.matmul(H, trace[trace_idx].permute(0,2,1))
            
            patch_trace = H.sum(dim=1)
            self.trace_not_merged_patches = []
            for img_idx in range(patch_trace.shape[0]):
                not_merged_patches = torch.where(patch_trace[img_idx]==1)[0]
                
                not_merged_patches = (H[img_idx][:, not_merged_patches].sum(1) ==1)
                
                # Eliminate CLS
                not_merged_patches = not_merged_patches[1:]

                self.trace_not_merged_patches.append(not_merged_patches)
            
        
        # Image reconstruction and classification
        cls_tokens, tokens_decoded = self.mae_decoder_classifier(tokens)
        
        # Image classification
        class_preds = self.head(cls_tokens)

        # Final steps of decoder
        tokens_decoded = self.decoder_norm(tokens_decoded)
        pred_pathces = self.decoder_pred(tokens_decoded)
        
        # Classification Loss
        classification_loss = self.classifier_loss(class_preds, labels)

        # Decoder loss
        reconstruction_loss = self.forward_loss(imgs, pred_pathces)
    
        if return_pred_images==True:
            return reconstruction_loss, classification_loss, self.unpatchify(pred_pathces)
        
        elif return_pred_labels==True:
            return reconstruction_loss, classification_loss, class_preds
        
        else: 
            return reconstruction_loss, classification_loss

    


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

        if self.use_trace_loss == False: 
            loss = (pred - target) ** 2
            loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
            
            loss = loss.mean() 
        else:
            loss = 0 
            for img_idx, trace in enumerate(self.trace_not_merged_patches):
                loss += ((pred[img_idx][trace] - target[img_idx][trace])**2).mean() 
            
            loss = loss / len(self.trace_not_merged_patches)


        
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

def make_vision_decoder_classifier(transformer_class):
    class VisionTransformerDecoderClassifier(transformer_class):
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
            # x = self.patch_drop(x)
            # x = self.norm_pre(x)
            x = self.blocks(x)
            x = self.norm(x)
            return x

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.forward_features(x)
            
            # 1. Avoid pooling, so that we can get the tokens as output of encoder
            # x = self.forward_head(x)
            
            # Get rid of CLS token
            cls = x[:, 0]
            tokens = x[:, 1:]
            return cls, tokens

    return VisionTransformerDecoderClassifier


def apply_patch_ecn_dec_classifier(
    model_encoder: VisionTransformer,
    model_decoder: VisionTransformer,
):
    
    
    VisionTransformerEncoder = make_vision_encoder(model_encoder.__class__)
    model_encoder.__class__ = VisionTransformerEncoder

    VisionTransformerDecoderClassifier = make_vision_decoder_classifier(model_decoder.__class__)
    model_decoder.__class__ = VisionTransformerDecoderClassifier
    
    # We need to set no_embed_class to True to avoid CLS token
    model_decoder.cls_token = None
    # model_decoder.pos_embed = model_decoder.pos_embed[:,1:] 
