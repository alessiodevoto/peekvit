import torch
from einops import rearrange
import numpy as np
from typing import Tuple 
import os   


def get_moes(model):
    """
    Retrieves all MoE (Mixture of Experts) modules from the given model.

    Args:
        model (nn.Module): The model to search for MoE modules.

    Returns:
        dict: A dictionary containing the names of MoE modules as keys and the modules themselves as values.
    """
    from models.moe_model import MoE
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

    from models.moe_model import MoE
    gates = {}
    for module_name, module in model.named_modules():
        if isinstance(module, MoE) and module.num_experts > 1:
            gates[module_name] = module.gating_probs.detach()

    return gates



def get_forward_masks(model):
    """
    """

    from models.residual_model import ResidualModule
    masks = {}
    for module_name, module in model.named_modules():
        if isinstance(module, ResidualModule):
            masks[module_name] = module.mask.detach() # (batch_size, sequence_len, 1)

    return masks



def prepare_for_matplotlib(t):
    """
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