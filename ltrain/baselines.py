import torch

class PCAReconstructor:
    def __init__(self, q=5, niter=2):
        self.q = q
        self.niter = niter
        self.center = True
        

    def decompose(self, A):
        if self.center:
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