from collections import OrderedDict
import torch
import os  
from datetime import datetime
from os.path import join
from models.models import build_model
from pprint import pprint


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


######################################################## MoEs ##################################################################



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

    from models.moevit import MoE
    gates = {}
    for module_name, module in model.named_modules():
        if isinstance(module, MoE) and module.num_experts > 1:
            gates[module_name] = module.gating_probs

    return gates


######################################################## Residual ##################################################################


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
        if isinstance(module, ResidualModule) and module.skip not in {None, 'none'}:
            masks[module_name] = module.mask # (batch_size, sequence_len, 1)

    return masks


######################################################## Noise ##################################################################


def add_noise(model, layer: int, noise_type:str,  std: float = None, snr: float = None, **kwargs):
    """
    Adds a noise module to the specified layer of the model's encoder. The model must be a transformer, 
    and it must have an encoder with a layers attribute.
    
    Args:
        model (nn.Module): The model to which the noise module will be added.
        layer (int): The index of the layer where the noise module will be inserted.
        noise_type (str): The type of noise to add. Must be one of {'gaussian'}.
        noise_std (float): The standard deviation of the noise.
        noise_snr (float): The signal-to-noise ratio of the noise.
    """
    from models.blocks import SNRNoise
    noise_module = SNRNoise(noise_type=noise_type, std=std, snr=snr)
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
    return model


######################################################## Training ##################################################################


def save_state(path, model, model_args, noise_args, optimizer, epoch, skip_optimizer=True):
    """
    Saves the state of the given model and optimizer to the specified path.
    """
    os.makedirs(path, exist_ok=True)
    state = {
        'model_class': model.__class__.__name__,
        'noise_args': noise_args,
        'model_args': model_args,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict() if not skip_optimizer else None,
        'epoch': epoch
    }
    torch.save(state, f'{path}/epoch_{epoch:03}.pth')


def load_state(path, model=None, optimizer=None):
    """
    Load the model state from a given path.

    Args:
        path (str): The path to the saved model state.
        model (torch.nn.Module, optional): The model to load the state into. If None, a new model will be created based on the saved state.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load the state into. If None, the optimizer state will not be loaded.

    Returns:
        tuple: A tuple containing the loaded model, optimizer, and the epoch number.
    """
    state = torch.load(path)
    if model is None:
        # create model based on saved state
        model = build_model(state['model_class'], state['model_args'], state['noise_args'])
        #model = state['model_class'](**state['model_args'])
    model.load_state_dict(state['state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
        
    return model, optimizer, state['epoch']



######################################################## Model editing ##################################################################



'''def from_vitblock_to_residualvitblock(vitblock, residualvitblock_args):
    """
    Converts a ViT block to a ResidualViT block.

    Args:
        vitblock: The ViT block to be converted.
        residualvitblock_args: Additional arguments for initializing the ResidualViT block.

    Returns:
        The converted ResidualViT block.
    """
    # transform a vit model to a residualvit model
    from models.residualvit import ResidualViTBlock

    # initialize residual block with same parameters as vit block
    residual_block = ResidualViTBlock(
        num_heads=vitblock.num_heads,
        hidden_dim=vitblock.hidden_dim,
        mlp_dim=vitblock.mlp_dim,
        dropout=vitblock.dropout.p,
        attention_dropout=vitblock.self_attention.self_attention.dropout,
        **residualvitblock_args
    )

    # copy weights from vit block to residualvit block
    residual_block.ln_1 = vitblock.ln_1
    residual_block.self_attention = vitblock.self_attention
    residual_block.dropout = vitblock.dropout
    residual_block.ln_2 = vitblock.ln_2
    residual_block.mlp = vitblock.mlp

    return residual_block'''


def add_residual_gates(residualvit_model, residual_gates_args):
    from models.residualvit import ResidualGate, ResidualViTBlock
    skip = residual_gates_args['skip']
    gate_type = residual_gates_args['gate_type']
    add_input = residual_gates_args['add_input']
    temp = residual_gates_args['temp']
    for module_name, module in residualvit_model.named_modules():
        if isinstance(module, ResidualViTBlock):
            print(f'Adding residual gate to {module_name}')
            module.skip = skip  
            module.add_input = add_input
            module.residual_gate = ResidualGate(module.hidden_dim, temp=temp, gate_type=gate_type)
    return residualvit_model


def freeze_module(module):
    # freeze all parameters of the module
    for param in module.parameters():
        param.requires_grad = False


"""def train_only_gates(residualvit_model):
    # freeze all parameters except the gates
    from models.residualvit import ResidualGate
    for module_name, module in residualvit_model.named_modules():
        if isinstance(module, ResidualGate):
            for param in module.parameters():
                param.requires_grad = True
        else:
            print(f'Freezing {module_name}')
            freeze_module(module)
    return residualvit_model"""

def train_only_gates_and_cls_token(residualvit_model):
    # freeze all parameters except the gates
    for param_name, param in residualvit_model.named_parameters():
        if 'gate' or 'class' in param_name:
            param.requires_grad = True
        else:
            # print(f'Freezing {param_name}')
            param.requires_grad = False
    return residualvit_model

def reinit_class_tokens(model):
    # reinitialize the class token
    for param_name, param in model.named_parameters():
        if 'class' in param_name:
            print(f'Reinitializing {param_name}')
            torch.nn.init.normal_(param, mean=0.0, std=0.02)
            print('Reinitialized')
    return model


