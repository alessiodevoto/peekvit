import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from collections import OrderedDict
from typing import Any, List
import torch
from datetime import datetime
from os.path import join
from peekvit.models.models import build_model


def make_experiment_directory(dir_path):
    """
    Create an experiment directory with subdirectories for checkpoints.

    Args:
        dir_path (str): The path of the experiment directory.

    Returns:
        str: The path of the created experiment directory.
    """
    os.makedirs(dir_path, exist_ok=True)

    checkpoints_dir = join(dir_path, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    images_dir = join(dir_path, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    return dir_path, checkpoints_dir


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


######################################################## MoEs ##################################################################



def get_moes(model):
    """
    Retrieves all MoE (Mixture of Experts) modules from the given model.

    Args:
        model (nn.Module): The model to search for MoE modules.

    Returns:
        dict: A dictionary containing the names of MoE modules as keys and the modules themselves as values.
    """
    from peekvit.models.moevit import MoE
    moes = {}
    for module_name, module in model.named_modules():
        if isinstance(module, MoE) and module.num_experts > 1: # only add MoE modules with more than 1 expert
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

    from peekvit.models.moevit import MoE
    gates = {}
    for module_name, module in model.named_modules():
        if isinstance(module, MoE) and module.num_experts > 1:
            gates[module_name] = module.gating_probs

    return gates


######################################################## Residual ##################################################################


def get_forward_masks(model, incremental=False):
    """
    Retrieves the forward masks from a given residual model.

    Args:
        model: The model from which to retrieve the forward masks.

    Returns:
        masks: A dictionary containing the forward masks for each ResidualModule in the model.
               The masks have shape (batch_size, sequence_len, 1).
    """
    from peekvit.models.residualvit import ResidualModule
    masks = {}
    previous_mask = torch.tensor(1.0)
    for module_name, module in model.named_modules():
        if isinstance(module, ResidualModule) and module.skip not in {None, 'none'}:
            if not incremental:
                masks[module_name] = module.mask  # (batch_size, sequence_len, 1)
            else:
                masks[module_name] = (module.mask * previous_mask.ceil()) 
                previous_mask = masks[module_name]
    
    return masks


def get_learned_thresholds(model):

    from peekvit.models.residualvit import ResidualModule
    thresholds = {}
    for module_name, module in model.named_modules():
        if isinstance(module, ResidualModule) and module.skip not in {None, 'none'}:
            t = module.residual_gate.threshold # scalar
            t = t if isinstance(t, float) else t.item()
            thresholds[module_name] = t

    return thresholds


######################################################## Ranking ##################################################################

def get_rankingvit_blocks(model):
    """
    Retrieves the RankingViT blocks from a given model.

    Args:
        model: The model from which to retrieve the RankingViT blocks.

    Returns:
        blocks: A dictionary containing the RankingViT blocks for each RankingViTBlock in the model.
    """
    from peekvit.models.rankvit import RankViTBlock
    blocks = {}
    for module_name, module in model.named_modules():
        if isinstance(module, RankViTBlock):
            blocks[module_name] = module
    
    return blocks


######################################################## Noise ##################################################################


def add_noise(model, layer: int, noise_type:str,  std: float = None, snr: float = None, prob: float = None, **kwargs):
    """
    Adds a noise module to the specified layer of the model's encoder. The model must be a transformer, 
    and it must have an encoder with a `layers` attribute containing Transformer blocks.
    
    Args:
        model (nn.Module): The model to which the noise module will be added.
        layer (int): The index of the layer where the noise module will be inserted.
        noise_type (str): The type of noise to add. Must be one of {'gaussian', 'token_drop'}.
        noise_std (float): The standard deviation of the noise.
        noise_snr (float): The signal-to-noise ratio of the noise.
        prob (float): The probability of applying the noise for token dropping.
    """
    from peekvit.models.blocks import NoiseBlock
    noise_module = NoiseBlock(noise_type=noise_type, std=std, snr=snr, prob=prob)
    new_layers = list(model.encoder.layers)

    # this could be an ordered dict or a list, deal with both cases

    if isinstance(new_layers[0], tuple):
        # if the first layer is a tuple, this is a dictionary of layers
        new_layers.insert(layer, ('noise', noise_module))
        new_layers = OrderedDict(new_layers)
        model.encoder.layers = torch.nn.Sequential(new_layers)
    else:
        # this is just a list
        new_layers.insert(layer, noise_module)
        model.encoder.layers = torch.nn.Sequential(*new_layers)
    
    return noise_module



######################################################## Training ##################################################################


def save_state(path, model, model_args, noise_args, optimizer, epoch, skip_optimizer=True):
    """
    Saves the state of the given model and optimizer to the specified path.
    """
    os.makedirs(path, exist_ok=True)
    state = {
        'model_class': model.__class__.__name__,
        'noise_args': dict(noise_args) if noise_args else None,
        'model_args': dict(model_args) if model_args else None,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict() if not skip_optimizer else None,
        'epoch': epoch
    }
    torch.save(state, f'{path}/epoch_{epoch:03}.pth')


def load_state(path, model : Any =None, optimizer : Any = None, strict: bool=False):
    """
    Load the model state from a given path.

    Args:
        path (str): The path to the saved model state.
        model (torch.nn.Module, optional): The model to load the state into. If None, a new model will be created based on the saved state.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load the state into. If None, the optimizer state will not be loaded.

    Returns:
        tuple: A tuple containing the loaded model, optimizer, the epoch number, the model args and the noise args.
    """
    state = torch.load(path)
    if model is None:
        # create model based on saved state
        # TODO edit this to load with hydra
        print('Creating model based on saved state')
        print(state['model_class'])
        print(state['model_args'])
        state['model_args'].pop('torch_pretrained_weights', None)
        state['model_args'].pop('_target_', None)
        model = build_model(state['model_class'], state['model_args'], state['noise_args'])
    
    try:
        res = model.load_state_dict(state['state_dict'], strict=strict)
        if len(res[0]) > 0:
            print('Some parameters are not present in the checkpoint and will be randomly initialized: ', res[0])
    except RuntimeError as e:
        print(e)
        print('The model state dict could not be loaded. This is probably because the checkpoint has a different architecture.')
        print('Checkpoint class: ', state['model_class'])
        print('Model class: ', model.__class__.__name__)
        print('Checkpoint args: ', state['model_args'])
        

    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
        
    return model, optimizer, state['epoch'], state['model_args'], state['noise_args']



def get_checkpoint_path(experiment_dir, epoch='last'):
    """
    Get the path of the checkpoint corresponding to the given epoch in the experiment directory.

    Args:
        experiment_dir (str): The directory path where the experiment is stored.
        epoch (int): The epoch number.

    Returns:
        str: The path of the checkpoint corresponding to the given epoch in the experiment directory.
    """
    checkpoints_dir = join(experiment_dir, 'checkpoints')
    
    if not os.path.exists(checkpoints_dir) or os.listdir(checkpoints_dir) == []:
        return None

    if epoch is None or epoch == 'last':
        checkpoint = sorted(os.listdir(checkpoints_dir))[-1]
    else:
        checkpoint = f'epoch_{epoch:03}.pth'
    return join(checkpoints_dir, checkpoint)
    

