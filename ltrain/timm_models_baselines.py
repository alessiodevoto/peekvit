import math
from typing import List, Optional
import torch
from torch import nn
import timm
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange

#from peekvit.models.vit import ViTBlock
from hydra.utils import instantiate
import timm

from patch.timm_custom_patch import apply_patch_tome_encoder, apply_patch_tome_decoder, apply_patch_tome_classifier
from noise_block import NoiseBlock
from cnn import calculate_flattened_size, CustomCNN, CustomTransformerTopology
from baselines import PCAReconstructor, Autoencoder
from patch_utils import apply_patch_ecn_dec_classifier

# # Initial dimensions
# height = 196
# width = 132

# # Calculate the flattened size
# flattened_size = calculate_flattened_size(height, width, layers)
# print(f'Flattened size: {flattened_size}')
class MAEVisionTransformer(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):

        super().__init__()
        self.logger = None
        # This is a general model that has to configure the overall behaviour of the model
        # It should allow for training: 
        # 1. ecoder-classifier (classification)
        # 2. encoder-decoder (reconstruction)
        # 3. encoder-decoder-classifier (reconstruction and classification)
        # 3.1 decoder and classifier can be the same or different models.

        self.initialize_network_base(cfg)

        # ------------------------ Define type of compression --------------------------
        self.compressor_name = cfg.compressor.name
        if self.compressor_name == "PCA":
            # Means that the compression is done through PCA
            self.compressor = PCAReconstructor(q=cfg.compressor.q, niter=cfg.compressor.niter)
        elif self.compressor_name == "Tome":
            apply_patch_tome_encoder(self.mae_encoder, trace_source=True, prop_attn=True)
            apply_patch_tome_decoder(self.decoder, prop_attn=False)
            
            self.mae_encoder.r = cfg.compressor.r if isinstance(cfg.compressor.r, int) else list(cfg.compressor.r)
    
        elif self.compressor_name == "No Compression":
            pass
        elif self.compressor_name == "AE":
            self.compressor = Autoencoder(
                input_dim=cfg.compressor.input_dim,
                encoding_dim=cfg.compressor.encoding_dim,
                activation = cfg.compressor.activation
            )
        elif self.compressor_name == "Tome+AE":
            apply_patch_tome_encoder(self.mae_encoder, trace_source=True, prop_attn=True)
            apply_patch_tome_decoder(self.decoder, prop_attn=False)
            
            self.mae_encoder.r = cfg.compressor.r if isinstance(cfg.compressor.r, int) else list(cfg.compressor.r)

            self.compressor = Autoencoder(
                input_dim=cfg.compressor.input_dim,
                encoding_dim=cfg.compressor.encoding_dim,
                activation = cfg.compressor.activation
            )
        elif self.compressor_name == "Tome+onlyclassifier":
            apply_patch_tome_encoder(self.mae_encoder, trace_source=True, prop_attn=True)
            #apply_patch_tome_decoder(self.decoder, prop_attn=False)
            apply_patch_tome_classifier(self.classifier, prop_attn=False)
            self.mae_encoder.r = cfg.compressor.r if isinstance(cfg.compressor.r, int) else list(cfg.compressor.r)

        else: 
            raise ValueError("The model type is not supported")
           
        if self.model_type =='paralel_models':
            apply_patch_tome_classifier(self.classifier, prop_attn=False)



    def initialize_network_base(self, cfg):
        # ------------------------ First part of the network --------------------------
        self.mae_encoder = timm.create_model('deit_tiny_patch16_224', pretrained=True) #VisionTransformer(**kwargs)
        # Get the number of blocks to skip or keep
        total_blocks = len(self.mae_encoder.blocks)
        n_blocks = 6
            
        # Take the topology classifier block:
        trnasformer_layer = self.mae_encoder.blocks[-1]
        
        # Finalize the first part
        self.mae_encoder.blocks = self.mae_encoder.blocks[:n_blocks]
        
        n_blocks += cfg.refinment_blocks
        assert n_blocks < total_blocks, "The number of blocks to keep and preserve is greater than the total number of blocks"

        # ------------------------ Define the network --------------------------
        self.reconstruct_images = cfg.reconstruct_images
        self.classfy_images = cfg.classfy_images
        self.model_type = cfg.model_type
        
        # This is the general setting of Encoder_Decoder_two_models_sequential
        # Define classifier and decoder
        self.classifier = timm.create_model('deit_tiny_patch16_224', pretrained=True) 
        self.decoder = timm.create_model('deit_tiny_patch16_224', pretrained=True) 
        # Assign the blocks
        self.classifier.blocks = self.classifier.blocks[n_blocks:]
        self.decoder.blocks = self.decoder.blocks[total_blocks - n_blocks:]
        self.train_classifier_separetely = False

        # ------------------------ Transmission parameters --------------------------
        # Noise block
        self.noise_block = NoiseBlock()

        # # It is necessary to make sure merging is correct
        # if cfg.transmit_cls_token == False:
        #     self.mae_encoder.cls_token = None

        # Transmit CLS token from encoder theough the channel
        self.transmit_cls_token = cfg.transmit_cls_token
        apply_patch_ecn_dec_classifier(
            model_encoder = self.mae_encoder,
            model_decoder = self.decoder,
            model_classifier = self.classifier,
        )
        self.patch_size = 16
        # ------------------------ Heads --------------------------
        # Last part of the decoder network
        in_chans, decoder_embed_dim = 3, 192
        self.decoder_norm = torch.nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.patch_size**2 * in_chans, bias=True) # decoder to patch
        
        # Last part of the classifier network
        self.head = nn.Linear(decoder_embed_dim, cfg.dataset.num_classes)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        # ------------------------ Loss/Metrics --------------------------
        self.norm_pix_loss = False
        # Classifier loss
        self.classifier_loss = instantiate(cfg.loss.classification_loss)
        self.use_trace_loss = cfg.compressor.get('use_trace_loss', False)
        
        # To maintain the best model accuracy
        self.best_validation_acc = -1
        self.best_val_loss_mse = -1
        self.best_val_loss_cl = -1
        
        

    def forward(
            self, 
            imgs: torch.Tensor, 
            labels: torch.Tensor,
            return_pred_images: bool = False, 
            return_pred_labels=False, 
            snr_db=None,
        ):
        # imgs: [N, 3, H, W]
        
        tokens = self.mae_encoder(imgs)

        if self.compressor_name in ["Tome"]:
            
            # Noise: Add noise
            tokens = self.noise_block(x=tokens, snr_db=snr_db)

            # Compression is performed in the encoder 
            num_tokens_compressed = tokens.shape[1]
            num_elements_copressed = tokens.shape[1] * tokens.shape[2]

            self.decoder._tome_info["layer_source"] = self.mae_encoder._tome_info["layer_source"]

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
                    if self.transmit_cls_token == True:
                        not_merged_patches = not_merged_patches[1:]

                    self.trace_not_merged_patches.append(not_merged_patches)
        elif self.compressor_name in ["Tome+onlyclassifier"]:
            # Final steps of decoder
            num_tokens_decompressed = 196
            num_elements_decopressed = 196 * 192
            # Noise: Add noise
            tokens = self.noise_block(x=tokens, snr_db=snr_db)

            # Compression is performed in the encoder 
            num_tokens_compressed = tokens.shape[1]
            num_elements_copressed = tokens.shape[1] * tokens.shape[2]

            trace = self.mae_encoder._tome_info["layer_source"].copy()
            if len(trace) > 1:
                H = torch.matmul(trace[0].permute(0,2,1), trace[1].permute(0,2,1))
                for trace_idx in range(2, len(trace)):
                    H = torch.matmul(H, trace[trace_idx].permute(0,2,1))
                
                patch_trace = H.sum(dim=1)
                self.trace_not_merged_patches = []
                for img_idx in range(patch_trace.shape[0]):
                    not_merged_patches = torch.where(patch_trace[img_idx]==1)[0]
                    
                    not_merged_patches = (H[img_idx][:, not_merged_patches].sum(1) ==1)
                    
                    # Eliminate CLS
                    if self.transmit_cls_token == True:
                        not_merged_patches = not_merged_patches[1:]

                    self.trace_not_merged_patches.append(not_merged_patches)
                self.classifier._tome_info["not_merged_patches"] = [torch.concat([torch.tensor([True]).to(i.device),i]) for i in self.trace_not_merged_patches]
            else:
                self.classifier._tome_info["not_merged_patches"] = None

        elif self.compressor_name in ["PCA"]:
            tokens_projected, V, A_mean = self.compressor.project(tokens)
            num_tokens_compressed = tokens_projected.shape[1] + V.shape[1] + A_mean.shape[1]
            
            num_elements_copressed = tokens_projected.shape[1] * tokens_projected.shape[2] 
            + V.shape[1] * V.shape[2]
            + A_mean.shape[1] * A_mean.shape[2]

            # Channel noise: Add noise
            tokens_projected = self.noise_block(x=tokens_projected, snr_db=snr_db)
            # V = self.noise_block(x=V, snr_db=snr_db)
            # A_mean = self.noise_block(x=A_mean, snr_db=snr_db)

            tokens = self.compressor.reconstruct_from_projection(tokens_projected, V, A_mean)
            
        elif self.compressor_name in ["No Compression"]:
            tokens = self.noise_block(x=tokens, snr_db=snr_db)
            num_tokens_compressed = tokens.shape[1]
            num_elements_copressed = tokens.shape[1] * tokens.shape[2]
        
        elif self.compressor_name in ["AE"]:
            tokens = self.compressor.encode(tokens)

            tokens = self.noise_block(x=tokens, snr_db=snr_db)

            num_tokens_compressed = tokens.shape[1]
            num_elements_copressed = tokens.shape[1] * tokens.shape[2]

            tokens = self.compressor.decode(tokens)
        elif self.compressor_name in ["Tome+AE"]:
            # Tome has already been applied

            # Encode 
            tokens = self.compressor.encode(tokens)

            # Noise: Add noise
            tokens = self.noise_block(x=tokens, snr_db=snr_db)


            # Compression is performed in the encoder 
            num_tokens_compressed = tokens.shape[1]
            num_elements_copressed = tokens.shape[1] * tokens.shape[2]

            self.decoder._tome_info["layer_source"] = self.mae_encoder._tome_info["layer_source"]

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
                    if self.transmit_cls_token == True:
                        not_merged_patches = not_merged_patches[1:]

                    self.trace_not_merged_patches.append(not_merged_patches)
            
            tokens = self.compressor.decode(tokens)
            
        else:
            raise ValueError("The model type is not supported")

        if self.compressor_name in ["Tome+onlyclassifier"]:
            tokens_decoded = tokens
        else:
            tokens_decoded = self.decoder(tokens)
            # Final steps of decoder
            num_tokens_decompressed = tokens_decoded.shape[1]
            num_elements_decopressed = tokens_decoded.shape[1] * tokens_decoded.shape[2]
            
        # Image classification
        if self.model_type == "Encoder_Decoder_two_models_sequential":
            cls_tokens = self.classifier(tokens_decoded)
        
        elif self.model_type == "paralel_models":
            if tokens.shape[1] != 196 and "Tome" in self.compressor_name:
                self.classifier._tome_info["not_merged_patches"] = True
            cls_tokens = self.classifier(tokens)
        
        else:
            pass        

        # Final step of classification
        class_preds = self.head(cls_tokens)
        
        self.logger.log({"Token compression": np.round(num_tokens_compressed / num_tokens_decompressed, 3)})
        self.logger.log({"Elements compression": np.round(num_elements_copressed / num_elements_decopressed, 3)})

        tokens_decoded = self.decoder_norm(tokens_decoded)
        pred_pathces = self.decoder_pred(tokens_decoded)
        
        # Classification Loss
        if self.classfy_images == True:
            classification_loss = self.classifier_loss(class_preds, labels)
        else:
            classification_loss = torch.tensor([0]).to(pred_pathces.device)

        # Decoder loss
        if self.reconstruct_images == True:
            reconstruction_loss = self.forward_loss(imgs, pred_pathces)
        else:
            reconstruction_loss = torch.tensor([0]).to(classification_loss.device)
        
        output_dict = {
            "reconstruction_loss": reconstruction_loss,
            "classification_loss": classification_loss,
            "class_preds": class_preds,
            "reconstructed_image": None, #self.unpatchify(pred_pathces) if self.reconstruct_images == True else None,
        }
        return output_dict
    
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

