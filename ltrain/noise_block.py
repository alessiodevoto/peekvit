from typing import Any, Literal
import torch
from torch import nn



class NoiseBlock(nn.Module):
    def __init__(self,):
      super().__init__()
   
    @torch.no_grad()
    def generate_noise(self, x: torch.Tensor, snr_db=None): 
      """
      Adds noise to the input according to the given signal to noise ratio.
      The input is assumed to be of shape (batch_size, sequence_length, hidden_dim).
      """
      #assert snr_linear > 0, "SNR must be greater than 0"
      # Compute the signal power
      signal_power = torch.mean(x ** 2, dim=-1, keepdim=True)
      
      # Compute the noise power
      # Convert SNR from dB to linear scale
      snr_linear = 10**(snr_db / 10)

      noise_power = signal_power / snr_linear.to(signal_power.device)
      
      # Compute the standard deviation of the noise
      std = torch.sqrt(noise_power)
      noise = torch.randn_like(x, requires_grad=False) * std
      
      return noise
    
    #@torch.no_grad()
    def forward(self, x: torch.Tensor, snr_db = 0):
      """
      Adds noise to the input according to the given signal to noise ratio.
      The input is assumed to be of shape (batch_size, sequence_length, hidden_dim).
      """
      if snr_db is None:
        # Sample snr from uniform distribution between 1, 10
        # This a lirear snr, to map it to db use the following SNR_linear=10^(SNR_db/10)
        snr_db = torch.randint(-10, 10, (1,))
      else:
        snr_db = torch.tensor(snr_db).unsqueeze(0)
      
      noise = self.generate_noise(x, snr_db=snr_db)

      # Add the noise to the input
      x = x + noise
      return x
     
    
   