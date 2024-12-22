import math
from typing import List, Optional
import torch
from torch import nn
import timm
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange

from torch_geometric.utils import scatter 
from hydra.utils import instantiate
import timm
import torch_geometric
from torch_geometric.data import Data, Batch
import sys

import torch_geometric.utils
sys.path.append('/home/lev/projects_6g/peekvit/ltrain/TopoBenchmark')
import TopoBenchmark.topobenchmark
#from TopoBenchmark.topobenchmark import DomainData
import hydra
from patch.timm_custom_patch import apply_patch_tome_encoder, apply_patch_tome_decoder, apply_patch_tome_classifier
from noise_block import NoiseBlock, CommunicationPipeline
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
            self.decoder = None

        # elif self.compressor_name == "Tome":
        #     apply_patch_tome_encoder(self.mae_encoder, trace_source=True, prop_attn=True)
        #     apply_patch_tome_decoder(self.decoder, prop_attn=False)
            
        #     self.mae_encoder.r = cfg.compressor.r if isinstance(cfg.compressor.r, int) else list(cfg.compressor.r)
    
        elif self.compressor_name == "No Compression":
            pass

        elif self.compressor_name == "AE":
            self.decoder = None
        
        elif self.compressor_name == "AE+gnn":
            self.decoder = None

            
        elif self.compressor_name in ["tome", "tome_gnn", "tome_topology"]:
            apply_patch_tome_encoder(self.mae_encoder, trace_source=True, prop_attn=True)
            #apply_patch_tome_decoder(self.decoder, prop_attn=False)
            #apply_patch_tome_classifier(self.classifier, prop_attn=False)
            self.mae_encoder.r = cfg.compressor.r if isinstance(cfg.compressor.r, int) else list(cfg.compressor.r)

        else: 
            raise ValueError("The model type is not supported")
           
        
        apply_patch_tome_classifier(self.classifier, prop_attn=False)
        self.cls_token = self.classifier.cls_token
        n_blocks = cfg['classifier_num_blocks']
        self.classifier.blocks = self.classifier.blocks[:n_blocks]

        if "gnn" in self.compressor_name:
            # Redefine the classifier
            cfg = {'backbone':
                {
                    '_target_':'torch_geometric.nn.models.GIN',
                    'in_channels': 192,
                    'hidden_channels': 192,
                    'num_layers': 2,
                    'dropout': 0,
                    'act': 'relu',
                },
            }         
            self.classifier_gnn = hydra.utils.instantiate(cfg['backbone'])
        elif self.compressor_name in ["tome_topology"]:
            cfg = {'backbone':
                {
                    '_target_':'topobenchmark.nn.backbones.combinatorial.gccn.TopoTune',
                    
                    'GNN': {
                        '_target_': 'topobenchmark.nn.backbones.graph.IdentityGCN',
                        'in_channels': 192,
                        'out_channels': 192,
                        'hidden_channels': 192,
                        'num_layers': 2,
                        'dropout': 0.0,
                        'norm': 'BatchNorm',

                    },
                    'neighborhoods': ['2-up_incidence-0', '2-down_incidence-2'],
                    'layers': 2,
                    'use_edge_attr': False,
                    'activation': 'relu',
                },
            }   

            self.classifier_topo = hydra.utils.instantiate(cfg['backbone'])


            



    def initialize_network_base(self, cfg):
        # ------------------------ First part of the network --------------------------
        self.mae_encoder = timm.create_model('deit_tiny_patch16_224', pretrained=True) #VisionTransformer(**kwargs)
        # Get the number of blocks to skip or keep
        total_blocks = len(self.mae_encoder.blocks)
        n_blocks = 6
            
        # Take the topology classifier block:
        # trnasformer_layer = self.mae_encoder.blocks[-1]
        
        # Finalize the first part
        self.mae_encoder.blocks = self.mae_encoder.blocks[:n_blocks]
        
        n_blocks += cfg.refinment_blocks
        assert n_blocks < total_blocks, "The number of blocks to keep and preserve is greater than the total number of blocks"

        # ------------------------ Define the network --------------------------
        
        # This is the general setting of Encoder_Decoder_two_models_sequential
        # Define classifier and decoder
        self.classifier = timm.create_model('deit_tiny_patch16_224', pretrained=True) 
        self.decoder = timm.create_model('deit_tiny_patch16_224', pretrained=True) 
        # Assign the blocks
        self.classifier.blocks = self.classifier.blocks[n_blocks:]
        self.decoder.blocks = self.decoder.blocks[total_blocks - n_blocks:]
        

        # ------------------------ Transmission pipeline --------------------------
        communication_channel_encoder = instantiate(cfg.ch_encoder.encoder) # BaseRealToComplexNN()
        communication_channel_decoder = instantiate(cfg.ch_decoder.decoder)
        noise_block = NoiseBlock()
        self.initial_num_elements = 224 * 224 * 3
        self.initial_num_tokens = 196

        self.communication_channel = CommunicationPipeline(
            encoder=communication_channel_encoder,
            channel=noise_block,
            decoder=communication_channel_decoder,
        )

        #self.noise_block = NoiseBlock()

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

        tokens = self.communication_channel(tokens, snr_db=snr_db)
        
        # Track the compression 
        num_tokens_compressed = tokens.shape[1]
        num_elements_copressed = self.communication_channel.num_elements_to_transmit

        
        if self.compressor_name in ["tome", "tome_gnn"]:
            pass
            #H = self.calculate_incidence(self.mae_encoder._tome_info["layer_source"].copy())
        else:
            pass
  
        # -----IMAGE CLASSIFICATION-----
        if "gnn" in self.compressor_name and "tome" not in self.compressor_name: 
            cls_tokens, tokens = self.classifier(tokens)
            tokens = torch.cat((cls_tokens.unsqueeze(1), tokens), dim=1)

            edge_index = generate_edge_index_with_cls(tokens.shape[1]) # generate_edge_index(tokens.shape[1])
            batch = create_torch_geometric_batch(tokens, edge_index)
            updated_tokes = self.classifier_gnn(batch.x, batch.edge_index.to(batch.x.device))
            cls_tokens = scatter(updated_tokes, batch.batch, dim=0, reduce='mean')
        
        elif "gnn" in self.compressor_name and "tome" in self.compressor_name:
            
            trace = self.mae_encoder._tome_info["layer_source"].copy()
            while len(trace) > 0:
                layer_source = trace.pop(-1).permute(0,2,1)
                layer_source = layer_source / layer_source.sum(1, keepdim=True)
                tokens = torch.matmul(layer_source, tokens)
            
            # Switch not_merged_patches to None to add PE to all tokens 
            self.classifier._tome_info["not_merged_patches"] = None
            cls_tokens, tokens = self.classifier(tokens)
            
            tokens = torch.cat((cls_tokens.unsqueeze(1), tokens), dim=1)
            edge_index = generate_edge_index_with_cls(tokens.shape[1]) # generate_edge_index(tokens.shape[1])
            batch = create_torch_geometric_batch(tokens, edge_index)
            updated_tokes = self.classifier_gnn(batch.x, batch.edge_index.to(batch.x.device))
            cls_tokens = scatter(updated_tokes, batch.batch, dim=0, reduce='mean')
        
        elif self.compressor_name in ["tome_topology"]:
            
            trace = self.mae_encoder._tome_info["layer_source"].copy()
            H, incedinces, cells = self.calculate_incidence(trace, tokens, add_cls_token=True)

            trace = self.mae_encoder._tome_info["layer_source"].copy()
            while len(trace) > 0:
                layer_source = trace.pop(-1).permute(0,2,1)
                layer_source = layer_source / layer_source.sum(1, keepdim=True)
                tokens = torch.matmul(layer_source, tokens)
            
            cls_tokens, tokens = self.classifier(tokens)

            tokens = torch.cat((cls_tokens.unsqueeze(1), tokens), dim=1)
            edge_index = generate_edge_index_with_cls(tokens.shape[1])
            A = torch_geometric.utils.to_dense_adj(edge_index).squeeze(0).to_sparse_coo()
            lst_data = []
            for idx, a in enumerate(zip(incedinces, cells)):
                incidence, cell = a
                data = Data()
                data['x']= tokens[idx]
                #fake x_1
                data['x_1'] = tokens[idx]
                data['x_2']= cell     
                data['2-up_incidence-0'] = incidence
                data['2-down_incidence-2'] = incidence.T

                data['shape'] = [tokens[idx].shape[0], 0, cell.shape[0]]
                
                lst_data.append(data)
            
            batch = list_data_objects_to_batch(lst_data)
            batch['x_0'] = batch.pop("x") 
            a = self.classifier_topo(batch)
            batch['x_0'] = batch['x_0'] + a[0]
            cls_tokens = scatter(batch['x_0'], batch.batch_0, dim=0, reduce='mean')

            
        else:
            if tokens.shape[1] != 196 and "tome" in self.compressor_name:
                self.classifier._tome_info["not_merged_patches"] = True
            cls_tokens, tokens = self.classifier(tokens)

        # Final step of classification
        class_preds = self.head(cls_tokens)

        # Classification Loss
        classification_loss = self.classifier_loss(class_preds, labels)
       
                
        self.logger.log(
            {
                "Token compression rate": np.round(num_tokens_compressed / self.initial_num_tokens, 4),
                "Elements compression rate (n/k)": np.round(num_elements_copressed / self.initial_num_elements, 4),
                "Elements compression rate (mine)": np.round(num_elements_copressed / (num_tokens_compressed*192), 4),
                
                "Tokens to send": num_tokens_compressed,
                "Elements to send": num_elements_copressed,

            }
        )
        
        

        output_dict = {
            "reconstruction_loss": torch.tensor([0]).to(classification_loss.device),
            "classification_loss": classification_loss,
            "class_preds": class_preds,
            "reconstructed_image": None, #self.unpatchify(pred_pathces) if self.reconstruct_images == True else None,
        }
        return output_dict
    
    def calculate_incidence(self, trace, tokens, add_cls_token=False):
        
        # trace = self.mae_encoder._tome_info["layer_source"].copy()
        
        if len(trace) > 1:
            H = torch.matmul(trace[0].permute(0,2,1), trace[1].permute(0,2,1))
            for trace_idx in range(2, len(trace)):
                H = torch.matmul(H, trace[trace_idx].permute(0,2,1))
            
            if add_cls_token == True:
                H = torch.cat((torch.zeros(1,1, H.shape[-1]).expand((H.shape[0],-1,-1)).to(H.device), H), dim=1)

            patch_trace = H.sum(dim=1)
            
            incedinces, cells = [], []
            for img_idx in range(patch_trace.shape[0]):
                merged_pathces = torch.where(patch_trace[img_idx] > 1)[0]
                incidence_0_2 = (H[img_idx][:, merged_pathces].to_sparse_coo())
                incedinces.append(incidence_0_2)
                cells.append(tokens[img_idx][merged_pathces])

        else:
            self.classifier._tome_info["not_merged_patches"] = None
            H = None
            incedinces = None
            cells = None

        return H, incedinces, cells

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

def generate_edge_index_with_cls(n_patches):
    
    """
    Generate edge_index for a grid with a CLS token connected to all nodes.

    Args:
        n_patches (int): Total number of patches, including the CLS token.

    Returns:
        torch.Tensor: Edge index of shape [2, num_edges].
    """
    # Exclude the CLS token for grid layout
    n_grid_patches = n_patches - 1  # Remaining patches (excluding CLS)
    grid_size = int(n_grid_patches ** 0.5)
    assert grid_size ** 2 == n_grid_patches, "Number of grid patches must form a square grid."

    edges = []

    # Connect CLS token (node 0) to all other nodes
    for node in range(1, n_patches):  # CLS token is 0, others start at 1
        edges.append([0, node])  # CLS -> Node
        edges.append([node, 0])  # Node -> CLS (bidirectional)

    # Generate edges for the grid layout (excluding CLS token)
    for i in range(grid_size):
        for j in range(grid_size):
            current_node = i * grid_size + j + 1  # Offset by 1 for CLS
            # Add edges to neighbors (right, down, left, up)
            if j + 1 < grid_size:  # Right neighbor
                edges.append([current_node, current_node + 1])
            if i + 1 < grid_size:  # Down neighbor
                edges.append([current_node, current_node + grid_size])
            if j - 1 >= 0:  # Left neighbor
                edges.append([current_node, current_node - 1])
            if i - 1 >= 0:  # Up neighbor
                edges.append([current_node, current_node - grid_size])

    # Convert edges to a tensor
    edge_index = torch.tensor(edges, dtype=torch.long).T  # Shape [2, num_edges]

    edge_index = torch_geometric.utils.to_undirected(edge_index)

    return edge_index




def generate_edge_index(n_patches):

    # Determine grid dimensions (assuming square grid)
    grid_size = int(n_patches ** 0.5)
    assert grid_size ** 2 == n_patches, "Number of patches must form a square grid."

    # Generate edge_index for a single grid
    edges = []
    for i in range(grid_size):
        for j in range(grid_size):
            current_node = i * grid_size + j
            # Add edges to neighbors (right, down, left, up)
            if j + 1 < grid_size:  # Right neighbor
                edges.append([current_node, current_node + 1])
            if i + 1 < grid_size:  # Down neighbor
                edges.append([current_node, current_node + grid_size])
            if j - 1 >= 0:  # Left neighbor
                edges.append([current_node, current_node - 1])
            if i - 1 >= 0:  # Up neighbor
                edges.append([current_node, current_node - grid_size])

    # Convert edges to a tensor
    edge_index = torch.tensor(edges, dtype=torch.long).T  # Shape [2, num_edges]

    
    return edge_index


def generate_fully_connected_edge_index(num_nodes):
    """
    Generate edge_index for a fully connected graph.

    Args:
        num_nodes (int): Total number of nodes in the graph.

    Returns:
        torch.Tensor: Edge index of shape [2, num_edges], representing a fully connected graph.
    """
    # Generate all pairs of nodes
    row = torch.arange(num_nodes).repeat_interleave(num_nodes)
    col = torch.arange(num_nodes).repeat(num_nodes)
    
    # Combine rows and columns into edge_index
    edge_index = torch.stack([row, col], dim=0)  # Shape [2, num_nodes^2]

    return edge_index



def create_torch_geometric_batch(images, edge_index):
    """
    Create a torch_geometric Batch object for a batch of images.

    Args:
        images (torch.Tensor): Batch of images of shape [n_images, n_patches, patch_dim].
        edge_index (torch.Tensor): Edge index for a single image of shape [2, num_edges].

    Returns:
        Batch: A torch_geometric Batch object.
    """
    n_images, n_patches, patch_dim = images.size()
    data_list = []

    for i in range(n_images):
        # Select patches for the current image
        x = images[i]  # Shape [n_patches, patch_dim]

        # Create a Data object for the current image
        data = Data(x=x, edge_index=edge_index)
        data_list.append(data)

    # Combine all Data objects into a Batch
    batch = Batch.from_data_list(data_list)
    return batch


### OLD CODE

# if self.compressor_name in ["Tome"]:        
#             num_tokens_compressed = tokens.shape[1]
#             num_elements_copressed = tokens.shape[1] * tokens.shape[2]

#             self.decoder._tome_info["layer_source"] = self.mae_encoder._tome_info["layer_source"]

#             trace = self.mae_encoder._tome_info["layer_source"].copy()

#             if self.use_trace_loss == True:
#                 H = torch.matmul(trace[0].permute(0,2,1), trace[1].permute(0,2,1))
#                 for trace_idx in range(2, len(trace)):
#                     H = torch.matmul(H, trace[trace_idx].permute(0,2,1))
                
#                 patch_trace = H.sum(dim=1)
#                 self.trace_not_merged_patches = []
#                 for img_idx in range(patch_trace.shape[0]):
#                     not_merged_patches = torch.where(patch_trace[img_idx]==1)[0]
                    
#                     not_merged_patches = (H[img_idx][:, not_merged_patches].sum(1) ==1)
                    
#                     # Eliminate CLS
#                     if self.transmit_cls_token == True:
#                         not_merged_patches = not_merged_patches[1:]

#                     self.trace_not_merged_patches.append(not_merged_patches)

def list_data_objects_to_batch(list_data_objects):
    r"""Overwrite `torch_geometric.data.DataLoader` collate function to use the `DomainData` class.

    This ensures that the `torch_geometric` dataloaders work with sparse matrices that are not necessarily named `adj`. The function also generates the batch slices for the different cell dimensions.

    Parameters
    ----------
    batch : list
        List of data objects (e.g., `torch_geometric.data.Data`).

    Returns
    -------
    torch_geometric.data.Batch
        A `torch_geometric.data.Batch` object.
    """
    from collections import defaultdict
    data_list = []
    batch_idx_dict = defaultdict(list)

    # Keep track of the running index for each cell dimension
    running_idx = {}

    for batch_idx, data in enumerate(list_data_objects):
        # values, keys = b[0], b[1]
        # data = DomainData()
        # for key, value in zip(keys, values, strict=False):
        #     if torch_geometric.utils.is_sparse(value):
        #         value = value.coalesce()
        #     data[key] = value

        keys = data.keys()
        # Generate batch_slice values for x_1, x_2, x_3, ...
        x_keys = [el for el in keys if ("x_" in el)]
        for x_key in x_keys:
            if x_key != "x_0":
                if x_key != "x_hyperedges":
                    cell_dim = int(x_key.split("_")[1])
                else:
                    cell_dim = x_key.split("_")[1]

                current_number_of_cells = data[x_key].shape[0]

                batch_idx_dict[f"batch_{cell_dim}"].append(
                    torch.tensor([[batch_idx] * current_number_of_cells])
                )

                if (
                    running_idx.get(f"cell_running_idx_number_{cell_dim}")
                    is None
                ):
                    running_idx[f"cell_running_idx_number_{cell_dim}"] = (
                        current_number_of_cells
                    )

                else:
                    running_idx[f"cell_running_idx_number_{cell_dim}"] += (
                        current_number_of_cells
                    )

        data_list.append(data)

    batch = torch_geometric.data.Batch.from_data_list(data_list)

    # Rename batch.batch to batch.batch_0 for consistency
    batch["batch_0"] = batch.pop("batch")

    # Add batch slices to batch
    for key, value in batch_idx_dict.items():
        batch[key] = torch.cat(value, dim=1).squeeze(0).long()

    # Ensure shape is torch.Tensor
    # "shape" describes the number of n_cells in each graph
    if batch.get("shape") is not None:
        cell_statistics = batch.pop("shape")
        batch["cell_statistics"] = torch.Tensor(cell_statistics).long()

    return batch