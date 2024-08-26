import torch

class PCAReconstructor:
    def __init__(self, q=5, niter=2):
        self.q = q
        self.niter = niter
        
        self.mean_ = None
    
    def decompose(self, A):
        mean = A.mean(dim=1, keepdim=True)            
        U, S, V = torch.pca_lowrank(A, q=self.q, center=True, niter=self.niter)
        return U, S, V, mean

    def reconstruct(self, U, S, V, mean):    
        SV = torch.einsum('bq,bdq->bqd', S, V)
        A_reconstructed = U @ SV

        if self.center and mean is not None:
            A_reconstructed += mean
        return A_reconstructed

    def compute_reconstruction_loss(self, A):
        U, S, V, mean = self.decompose(A)    
        A_reconstructed = self.reconstruct(U, S, V, mean)
        loss = ((A - A_reconstructed) ** 2).sum()
        return loss
    
    def project(self, A):
        """Decompose A and project it to the principal components"""
        torch.random.manual_seed(42)
        U, S, V, A_mean = self.decompose(A)
        A_projected = torch.einsum('bnd,bdi->bni', (A - A_mean), V)
        return A_projected, V, A_mean
    
    def reconstruct_from_projection(self, A_projected, V, A_mean):
        """Reconstruct A from the projected data"""
        A_reconstructed = torch.einsum('bni,bdi->bnd', A_projected, V) + A_mean
        return A_reconstructed
    


# VAE

import torch
from torch import nn
from torch.nn import functional as F
from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple
# from torch import tensor as Tensor
Tensor = TypeVar('torch.tensor')
# import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, activation):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            #nn.LayerNorm(input_dim),
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(),
            #nn.ReLU() if activation else nn.Identity(),
            #nn.Linear(encoding_dim, encoding_dim),
            
        )
        self.layer_norm_encoding = nn.LayerNorm(encoding_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            #nn.LayerNorm(encoding_dim),
            nn.Linear(encoding_dim, input_dim),
            nn.ReLU()
            #nn.ReLU() if activation else nn.Identity(),
            #nn.Linear(input_dim, input_dim),
        )
        
        self.layer_norm_decoder = nn.LayerNorm(input_dim)
    # def forward(self, x):
    #     encoded = self.encoder(x)
    #     decoded = self.decoder(encoded)
    #     return decoded
    
    def encode(self, x):
        encoded = self.encoder(x)
        encoded = self.layer_norm_encoding(encoded)
        return encoded
    
    def decode(self, x):
        decoded = self.decoder(x)
        decoded = self.layer_norm_decoder(decoded)
        return decoded

