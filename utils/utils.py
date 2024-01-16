from collections import OrderedDict
from typing import List
import torch
import os  
from datetime import datetime
from os.path import join
from models.models import build_model


def make_experiment_directory(base_path, is_eval:bool=False):
    """
    Creates a new directory for an experiment based on the current date and time.
    Args:
        base_path (str): The base path where the experiment directory will be created.
        is_eval (bool): Whether this is an evaluation experiment or not. 
    Returns:
        str: The path of the newly created experiment directory.
        str: The formatted date and time of the experiment, i.e. the experiment name.
    """
    now = datetime.now()
    formatted_date_time = now.strftime("%Y_%m_%d_%H_%M_%S") + '_eval' if is_eval else now.strftime("%Y_%m_%d_%H_%M_%S")
    exp_dir = join(base_path, formatted_date_time)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir, formatted_date_time


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


def get_forward_masks(model, incremental=False):
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

    from models.residualvit import ResidualModule
    thresholds = {}
    for module_name, module in model.named_modules():
        if isinstance(module, ResidualModule) and module.skip not in {None, 'none'}:
            t = module.residual_gate.threshold # scalar
            t = t if isinstance(t, float) else t.item()
            thresholds[module_name] = t

    return thresholds


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
    from models.blocks import NoiseBlock
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
        tuple: A tuple containing the loaded model, optimizer, the epoch number, the model args and the noise args.
    """
    state = torch.load(path)
    if model is None:
        # create model based on saved state
        model = build_model(state['model_class'], state['model_args'], state['noise_args'])
    model.load_state_dict(state['state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
        
    return model, optimizer, state['epoch'], state['model_args'], state['noise_args']



######################################################## Model editing ##################################################################


def add_residual_gates(residualvit_model, residual_gates_args):
    from models.residualvit import ResidualGate, ResidualViTBlock
    skip = residual_gates_args['residual_layers']
    gate_type = residual_gates_args['gate_type']
    add_input = residual_gates_args['add_input']
    temp = residual_gates_args['gate_temp']
    i = 0
    for module_name, module in residualvit_model.named_modules():
        if isinstance(module, ResidualViTBlock) and skip[i] in {'attention+mlp', 'attention', 'mlp'}:
            print(f'Adding residual gate to {module_name}')
            module.skip = skip[i]  
            module.add_input = add_input
            module.residual_gate = ResidualGate(module.hidden_dim, temp=temp, gate_type=gate_type)
            i += 1
    return residualvit_model


def freeze_module(module):
    # freeze all parameters of the module
    for param in module.parameters():
        param.requires_grad = False


def train_only_gates_and_cls_token(residualvit_model, verbose:bool=False):
    # freeze all parameters except the gates
    print('Freezing all parameters except the gates and the class token')
    
    frozen_params, trainable_params = [], []
    for param_name, param in residualvit_model.named_parameters():
        if any([x in param_name for x in ['gate', 'class', 'head', 'threshold']]):
            param.requires_grad = True
            trainable_params.append(param_name)
        else:
            param.requires_grad = False
            frozen_params.append(param_name)
    
    if verbose:
        print('Trainable parameters:', trainable_params)
        print('Frozen parameters:', frozen_params)

    return residualvit_model


def train_only_these_params(residualvit_model, params_list: List, verbose:bool=False):
    
    print('Freezing all parameters except for those containing any of these words in their names: ', params_list)
    
    frozen_params, trainable_params = [], []
    for param_name, param in residualvit_model.named_parameters():
        if any([x in param_name for x in params_list]):
            param.requires_grad = True
            trainable_params.append(param_name)
        else:
            param.requires_grad = False
            frozen_params.append(param_name)
    
    if verbose:
        print('Trainable parameters:', trainable_params)
        print('Frozen parameters:', frozen_params)

    return residualvit_model

def reinit_class_tokens(model):
    # reinitialize the class token
    for param_name, param in model.named_parameters():
        if 'class' in param_name:
            print(f'Reinitializing {param_name}...', end=' ')
            torch.nn.init.normal_(param, mean=0.0, std=0.02)
            print('Reinitialized!')
    return model


######################################################## Mix ##################################################################

# map imagenette to imagenet classes via a transform
class ImagenetToImagenetteLabel(object):
    def __init__(self):
        super().__init__()
        self.mapping = {
            0: 0,       # tench
            1: 217,     # english springer
            2: 482,     # cassette player
            3: 491,     # chainsaw
            4: 497,     # church
            5: 566,     # french horn
            6: 569,     # garbage
            7: 571,     # gas
            8: 574,     # golf ball
            9: 701,     # parachute
        }

    def __call__(self, label):
        return self.mapping[label]

