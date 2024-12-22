import torch
from timm.models.vision_transformer import Attention, Block, VisionTransformer

def make_vision_encoder(transformer_class):
    # By default, we do not transmit CLS token
    class VisionTransformerEncoder(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.forward_features(x)
            return x
        def forward_features(self, x):
            x = self.patch_embed(x)
            
            # Here no CLS token is added hence self.pos_embed has to be modified a bit
            x = self.pos_drop(x + self.pos_embed[:,1:,:])
            x = self.blocks(x)
            x = self.norm(x)
            return x

    return VisionTransformerEncoder

def make_vision_decoder(transformer_class):
    class VisionTransformerDecoder(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """
        
        def forward_features(self, x: torch.Tensor) -> torch.Tensor:
            x = self.blocks(x)
            x = self.norm(x)
            return x

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.forward_features(x)
            return x
    return VisionTransformerDecoder

def make_vision_classifier(transformer_class):
    # I believe classifier adds a CLS token in the input
    class VisionTransformerClassifier(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            
            # Get the CLS token
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            
            # Add the CLS token to the input
            x = torch.cat((cls_token, x), dim=1)
            # Check if self has the attribute _tome_info
            if not hasattr(self, '_tome_info'):
                self._tome_info = {}
            
            if self._tome_info.get('not_merged_patches', None) !=None:
                x[:,0,:] = x[:,0,:] + self.pos_embed[:,0,:]
                #x = self.pos_drop(x + self.pos_embed[:,self._tome_info['not_merged_patches']])
            else:
                x = self.pos_drop(x + self.pos_embed)
            
            # Process
            x = self.forward_features(x)
            cls = self.forward_head(x)
            return cls, x[:, 1:]
        
        def forward_features(self, x: torch.Tensor) -> torch.Tensor:
            x = self.blocks(x)
            x = self.norm(x)
            return x
        
        def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
            # Global pooling
            return x[:, 0]  # class token

        

    return VisionTransformerClassifier


def apply_patch_ecn_dec_classifier(
    model_encoder: VisionTransformer,
    model_decoder: VisionTransformer,
    model_classifier=None,
):
    # It applies the necessary patches to the encoder, decoder and classifier
    VisionTransformerEncoder = make_vision_encoder(model_encoder.__class__)
    model_encoder.__class__ = VisionTransformerEncoder

    VisionTransformerClassifier = make_vision_classifier(model_classifier.__class__)
    model_classifier.__class__ = VisionTransformerClassifier

    VisionTransformerDecoder = make_vision_decoder(model_decoder.__class__)
    model_decoder.__class__ = VisionTransformerDecoder
    
    # We need to set no_embed_class to True to avoid CLS token
    model_decoder.cls_token = None
    model_encoder.cls_token = None