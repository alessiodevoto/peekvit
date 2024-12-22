from typing import Any, Literal
import torch
from torch import nn

import numpy as np
from copy import deepcopy




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
     
    
class CommunicationPipeline(nn.Module):
    def __init__(self, encoder, channel, decoder, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.encoder = encoder if encoder is not None else lambda x: x
        self.decoder = decoder if decoder is not None else lambda x: x
        self.channel = channel if channel is not None else lambda x: x
        
        self.num_elements_to_transmit = None

    def forward(self, x, snr_db=None):
        # Maps the input into complex space
        x = self.encoder(x)

        # Get number of elements to transmit
        self.calculate_num_values(x)

        # Adds noise to the input
        x = self.channel(x, snr_db=snr_db)

        # Decodes the transmitter/moised signal back to real space
        decoded = self.decoder(x)

        return decoded   
    
    def calculate_num_values(self, x):
        B, C, H = x.shape
        self.num_elements_to_transmit = C * H
        
        
     
def get_layers(input_size, output_size=1.0, n_layers=2, n_copy=1, invert=False, drop_last_activation=False):
    if isinstance(output_size, float):
        output_size = int(input_size * output_size)

    shapes = np.linspace(input_size, output_size, num=n_layers + 1, endpoint=True, dtype=int)

    model = []

    for s in range(len(shapes) - 1):
        model.append(nn.Linear(shapes[s], shapes[s + 1]))
        model.append(nn.ReLU())

    if drop_last_activation:
        model = model[:-1]
    # if invert:
    #     model = model[::-1]

    model = nn.Sequential(*model)

    models = []
    for _ in range(n_copy):
        _model = deepcopy(model)

        for m in _model.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

        models.append(_model)

    return shapes[-1], models

class BaseRealToComplexNN(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 n_layers=2,
                 normalize=True,
                 transpose=False,
                 drop_last_activation=False,
                 sincos=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        output_size, (cc, rr) = get_layers(input_size=input_size, output_size=output_size,
                                           n_layers=n_layers, drop_last_activation=drop_last_activation,
                                           n_copy=2, invert=False)

        self.r_fl, self.c_fl = rr, cc

        self.normalize = normalize
        self.transpose = transpose
        self.sincos = sincos
        self.output_size = output_size

    def forward(self, x, *args, **kwargs):
        if self.transpose:
            x = x.permute(0, 2, 1)

        a, b = self.r_fl(x), self.c_fl(x)

        if self.sincos:
            a, b = torch.cos(a), torch.sin(b)

        x = torch.complex(a, b)

        if self.normalize:
            x = x / torch.norm(x, 2, -1, keepdim=True)

        return x


class ConcatComplexToRealNN(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 n_layers=2,
                 normalize=False,
                 transpose=False,
                 drop_last_activation=True,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        if isinstance(input_size, float):
          input_size = int(input_size * output_size)

        out_shape, (cc,) = get_layers(input_size=input_size * 2, output_size=output_size,
                                      n_layers=n_layers, drop_last_activation=drop_last_activation,
                                      n_copy=1, invert=False)

        self.d_f = cc
        self.transpose = transpose
        self.normalize = normalize

        self.layer_norm_decoder = nn.LayerNorm(output_size)
    def forward(self, x=None, *args, **kwargs):
        if self.normalize:
            x = x / torch.norm(x, 2, -1, keepdim=True)

        x = torch.cat((x.real, x.imag), -1)
        x = self.d_f(x)
        x = self.layer_norm_decoder(x)

        if self.transpose:
            x = x.permute(0, 2, 1)

        return x