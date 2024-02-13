from typing import Any, Literal
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import gumbel_softmax, sigmoid


"""
A set of nn Modules for various use cases.
"""


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


# A differentiable Gumbel Sigmoid module
class GumbelSigmoid(nn.Module):
    def __init__(self, hard: bool = True, temp: float = 1.0, bias: float = 0.0):
      super().__init__()
      self.hard = hard
      self.temp = temp  
      self.bias = bias 
    
    @staticmethod
    def gumbel_sigmoid(logits: torch.Tensor, tau: float = 1, bias: float = 0.0, hard: bool = False, eps: float = 1e-10):
      gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
      gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
      y_soft = torch.sigmoid(gumbels + bias)

      if hard:
          # Straight through.
          y_hard = torch.round(y_soft)
          ret = y_hard - y_soft.detach() + y_soft
      else:
          # Reparametrization trick.
          ret = y_soft
      return ret

    def forward(self, x):
      # check if model is in training mode
      if self.training:
        return self.gumbel_sigmoid(x, hard = self.hard, tau = self.temp, bias = self.bias)
      else:
        # at inference, we use a hard threshold
        return torch.round(torch.sigmoid(x))



# A differentiable Sigmoid with temperature module
class SigmoidWithTemp(nn.Module):
    def __init__(self, bias: float = 0, temp: float = 1.0):
      super().__init__()
      self.temp = temp
      self.bias = bias

    def forward(self, x):
      return sigmoid((x / self.temp) + self.bias)
      
      

# ViT MLP
class MLP(nn.Module):
    def __init__(self, hidden_dim, mlp_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x) # TODO change to gelu
        x = self.fc2(x)
        return x


# ViT Self Attention
class SelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads, dropout=0.0):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(input_dim, num_heads, batch_first=True, dropout=dropout)
        self.num_heads = num_heads

    def forward(self, x, mask=None):
        out, weights = self.self_attention(x, x, x, attn_mask=mask, need_weights=True)
        return out



# A class to add random noise to the input according to a specific signal to noise ratio or std deviation
class NoiseBlock(nn.Module):
    def __init__(self, noise_type: Literal['gaussian', 'token_drop'] = 'gaussian', snr = None, std = None, prob = None):
      super().__init__()
      self.noise_type = noise_type
      self.snr = snr
      self.std = std
      self.prob = prob
 
      if not any([snr, std, prob]):
        print('Lazy initialization of noise block. Please set the noise parameters using set_snr, set_std or set_prob before using the block.')
      
      if std:
          raise ValueError('std is not supported anymore. Please use snr instead.')


    def forward_snr(self, x: torch.Tensor):
      """
      Adds noise to the input according to the given signal to noise ratio.
      The input is assumed to be of shape (batch_size, sequence_length, hidden_dim).
      """
      # compute the signal power
      signal_power = torch.mean(x ** 2, dim=-1, keepdim=True)
      # compute the noise power
      noise_power = signal_power / self.snr
      # compute the standard deviation of the noise
      std = torch.sqrt(noise_power)
      # add the noise
      return x + torch.randn_like(x, requires_grad=False) * std
  
    def forward_std(self, x: torch.Tensor):
      """
      Adds noise to the input according to the given standard deviation.
      The input is assumed to be of shape (batch_size, sequence_length, hidden_dim).
      """
      return x + torch.randn_like(x, requires_grad=False) * self.std
    
    
    def forward_token_drop(self, x: torch.Tensor):
      """
      Masks tokens by zeroing them according to the given probability.
      """
      if self.prob == 0:
        return x

      # input has shape (batch_size, sequence_length, hidden_dim)
      noise = torch.ones_like(x, requires_grad=False)
      # compute the number of tokens to mask
      num_mask = int(self.prob * x.shape[1])
      # sample the indices of the tokens to mask
      mask_indices = torch.randperm(x.shape[1])[:num_mask]
      # mask the tokens
      noise[:, mask_indices, :] = 0
      # add the noise
      return x * noise
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor):
      """
      Adds noise to the input according to the given signal to noise ratio.
      The input is assumed to be of shape (batch_size, sequence_length, hidden_dim).
      """
      if self.snr is not None:
        return self.forward_snr(x)
      elif self.std is not None:
        return self.forward_std(x)
      else:
         return self.forward_token_drop(x)
    
    def set_snr(self, snr: float):
      assert self.noise_type == 'gaussian'
      self.snr = snr
      self.std = None
      self.prob = None
    
    def set_prob(self, prob: float):
      assert self.noise_type == 'token_drop'
      self.snr = None
      self.std = None
      self.prob = prob

    def set_value(self, value: float):
      if self.noise_type == 'gaussian':
        self.set_snr(value)
      else:
        self.set_prob(value)


        