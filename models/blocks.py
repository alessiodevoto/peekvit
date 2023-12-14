import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import gumbel_softmax, sigmoid


# A differentiable Gumbel Softmax module
class GumbelSoftmax(nn.Module):
    def __init__(self, dim, hard):
      super().__init__()
      self.dim = dim
      self.hard = hard
    def forward(self, x):
      # check if model is in training mode
      if self.training:
        return gumbel_softmax(x, dim= self.dim, hard = self.hard)
      else:
         # at inference time, we use the argmax of the gumbel softmax
        return F.one_hot(x.argmax(dim=self.dim), num_classes=x.shape[-1]).float()


# A differentiable Sigmoid with temperature module
class SigmoidWithTemp(nn.Module):
    def __init__(self, temp):
      super().__init__()
      self.temp = temp
    def forward(self, x):
      return sigmoid(x / self.temp)


# ViT MLP
class MLP(nn.Module):
    def __init__(self, hidden_dim, mlp_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


# ViT Self Attention
class SelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads, dropout=0.0):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(input_dim, num_heads, batch_first=True, dropout=dropout)

    def forward(self, x):
        out, weights = self.self_attention(query=x, key=x, value=x, need_weights=True)
        return out


# A class to add gaussian noise to the input
class Noise(nn.Module):
    def __init__(self, std):
      super().__init__()
      self.std = std

    def forward(self, x):
      return x + torch.randn_like(x, requires_grad=False) * self.std


# A class to add random noise to the input according to a specific signal to noise ratio
class SNRNoise(nn.Module):
    def __init__(self, snr):
      super().__init__()
      self.snr = snr

    def forward(self, x):
      # compute the signal power
      signal_power = torch.mean(x ** 2, dim=-1, keepdim=True)
      # compute the noise power
      noise_power = signal_power / self.snr
      # compute the standard deviation of the noise
      std = torch.sqrt(noise_power)
      # add the noise
      return x + torch.randn_like(x, requires_grad=False) * std