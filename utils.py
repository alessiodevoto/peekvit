import torch
from einops import rearrange
import numpy as np
from typing import Tuple 
import os  
from datetime import datetime
from os.path import join


def make_experiment_directory(base_path):
    """
    Creates a new directory for an experiment based on the current date and time.
    Args:
        base_path (str): The base path where the experiment directory will be created.
    Returns:
        str: The path of the newly created experiment directory.
    """
    now = datetime.now()
    formatted_date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    return join(base_path, formatted_date_time)


def make_batch(x: torch.Tensor):
  """
  Converts the given input to a batch of size 1 if it is not already a batch.
  """
  if len(x.shape) == 3:
    return x.unsqueeze(0)


def get_model_device(model: torch.nn.Module):
    """
    Retrieves the device of the given model.
    """
    return next(model.parameters()).device


def get_moes(model):
    """
    Retrieves all MoE (Mixture of Experts) modules from the given model.

    Args:
        model (nn.Module): The model to search for MoE modules.

    Returns:
        dict: A dictionary containing the names of MoE modules as keys and the modules themselves as values.
    """
    from models.moevit import MoE
    moes = {}
    for module_name, module in model.named_modules():
        if isinstance(module, MoE) and module.num_experts > 1:
            moes[module_name] = module

    return moes


def get_last_forward_gates(model):
    """
    Retrieves the last forward gating probabilities for each MoE module in the model.

    Args:
        model: The model containing MoE modules.

    Returns:
        dict: A dictionary mapping the module names to their corresponding gating probabilities.
        gatin probs shape: (batch_size, sequence_len, num_experts)
    """

    from models.moevit import MoE
    gates = {}
    for module_name, module in model.named_modules():
        if isinstance(module, MoE) and module.num_experts > 1:
            gates[module_name] = module.gating_probs.detach()

    return gates



def get_forward_masks(model):
    """
    Retrieves the forward masks from a given residual model.

    Args:
        model: The model from which to retrieve the forward masks.

    Returns:
        masks: A dictionary containing the forward masks for each ResidualModule in the model.
               The masks have shape (batch_size, sequence_len, 1).
    """
    from models.residualvit import ResidualModule
    masks = {}
    for module_name, module in model.named_modules():
        if isinstance(module, ResidualModule):
            masks[module_name] = module.mask.detach() # (batch_size, sequence_len, 1)

    return masks



def prepare_for_matplotlib(t):
    """
    Prepares the given tensor for matplotlib by converting it to a numpy array and reshaping it to have channels last.
    If it is a pytorch tensor, converts it to a numpy array and reshapes it to have channels last.
    If it is a numpy a numpy array, reshapes it to have channels last.
    """
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    if isinstance(t, np.ndarray) and len(t.shape) == 3 and t.shape[0] in {3, 1}:
        t = rearrange(t, 'c h w -> h w c')
        
    return t


def denormalize(t: torch.Tensor, mean: Tuple, std: Tuple):
    """
    Denormalizes the given tensor with the given mean and standard deviation.
    """
    mean = torch.tensor(mean).view(1, -1, 1, 1)
    std = torch.tensor(std).view(1, -1, 1, 1)
    return t * std + mean



def add_noise(model, layer, noise_std, noise_snr):
    """
    Adds a noise module to the specified layer of the model's encoder. The model must be a transformer, 
    and it must have an encoder with a layers attribute.
    
    Args:
        model (nn.Module): The model to which the noise module will be added.
        layer (int): The index of the layer where the noise module will be inserted.
        noise_std (float): The standard deviation of the noise.
        noise_snr (float): The signal-to-noise ratio of the noise.
    """
    from models.blocks import SNRNoise
    noise_module = SNRNoise(std=noise_std, snr=noise_snr)
    model.encoder.layers.insert(layer, noise_module)
    


class SimpleLogger:
    """
    Simple logger for logging to stdout and to a file.
    """
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        if not os.path.exists(os.path.dirname(log_file_path)):
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        self.log_file = open(log_file_path, 'w')
    
    def log(self, *args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=self.log_file)
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()