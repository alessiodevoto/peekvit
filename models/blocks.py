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
    def __init__(self, hard):
      super().__init__()
      self.hard = hard
    
    @staticmethod
    def gumbel_sigmoid(logits: torch.Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10):
      gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
      gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
      y_soft = torch.sigmoid(gumbels)

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
        return self.gumbel_sigmoid(x, hard = self.hard)
      else:
        # at inference, we use a hard threshold
        return torch.round(torch.sigmoid(x))



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
        x = F.relu(x) # TODO change to gelu
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



# A class to add random noise to the input according to a specific signal to noise ratio or std deviation
class SNRNoise(nn.Module):
    def __init__(self, noise_type: str = 'gaussian', snr = None, std = None):
      super().__init__()
      self.snr = snr
      self.std = std
      self.noise_type = noise_type

      if snr is None and std is None:
        raise ValueError("Either snr or std must be specified")
      elif snr is not None and std is not None:
        raise ValueError("Only one of snr or std must be specified")
      
      if noise_type not in {'gaussian'}:
        raise ValueError("noise_type must be one of {'gaussian'}")
    
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
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor):
      """
      Adds noise to the input according to the given signal to noise ratio.
      The input is assumed to be of shape (batch_size, sequence_length, hidden_dim).
      """
      if self.snr is not None:
        return self.forward_snr(x)
      else:
        return self.forward_std(x)




'''class TokenNoise(nn.Module):
    def __init__(self, p=0.5):
        """
        Custom dropout module.

        Args:
            p (float): Probability of an element to be zeroed. Default: 0.5
        """
        super(TokenNoise, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, but got {}".format(p))
        self.p = p

    def forward(self, x):
        """
        Forward pass of the custom dropout module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying dropout.
        """
        if self.training:
            # Generate a binary mask with zeros and ones based on the dropout probability
            mask = torch.rand_like(x) > self.p
            # Scale the output to account for dropped elements during training
            return x * mask / (1 - self.p)
        else:
            # During evaluation, just return the input as is
            return x

# Example usage:
# Create an instance of CustomDropout with a dropout probability of 0.5
dropout_layer = CustomDropout(p=0.5)

# Apply the dropout layer to an input tensor
input_tensor = torch.randn(1, 10)
output_tensor = dropout_layer(input_tensor)

# Print the input and output tensors
print("Input Tensor:", input_tensor)
print("Output Tensor:", output_tensor)'''

        