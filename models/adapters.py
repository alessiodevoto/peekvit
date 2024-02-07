from .residualvit import ResidualVisionTransformer
from .eeresidualvit import EEResidualVisionTransformer
import torch
from torch import nn
import re


@torch.no_grad()
def from_vit_to_residual_vit(vit_checkpoint, model_args:dict = None):
    """
    Converts a Vision Transformer (ViT) checkpoint to a Residual Vision Transformer (ResidualViT) model.

    Args:
        vit_checkpoint (str): Path to the ViT checkpoint file.
        model_args (dict, optional): Arguments to initialize the ResidualViT model. Defaults to None. If None, the arguments from the checkpoint are used.

    Returns:
        ResidualViT: The converted ResidualViT model.
        Args: The Resvit model arguments.
    """
    
    # load weights from vit checkpoint
    state = torch.load(vit_checkpoint)
    print('Loading weights from class: ', state['model_class'])
    vit_weights = state['state_dict']
    checkpoint_model_args = state['model_args']

    model_args = model_args if model_args is not None else checkpoint_model_args

    # build residual vit, with randomly init weights
    residual_vit = ResidualVisionTransformer(**model_args)

    # copy weights from vit to residual vit
    res = residual_vit.load_state_dict(vit_weights, strict=False)

    print('Some parameters are not present in the checkpoint and will be randomly initialized: ', res[0])

    return residual_vit



@torch.no_grad()
def from_vit_to_eeresidual_vit(vit_checkpoint, residual_vit_args:dict = None):
    """
    Converts a Vision Transformer (ViT) checkpoint to a Residual Vision Transformer (ResidualViT) model.

    Args:
        vit_checkpoint (str): Path to the ViT checkpoint file.
        residual_vit_args (dict, optional): Additional arguments to initialize the ResidualViT model. Defaults to None.

    Returns:
        ResidualViT: The converted ResidualViT model.
        Args: The Resvit model arguments.
    """
    
    # load weights from vit checkpoint
    state = torch.load(vit_checkpoint)
    print('Loading weights from class: ', state['model_class'])
    vit_weights = state['state_dict']
    model_args = state['model_args']

    # build residual vit, with randomly init weights
    residual_vit = EEResidualVisionTransformer(**model_args, **residual_vit_args)

    # copy weights from vit to residual vit
    res = residual_vit.load_state_dict(vit_weights, strict=False)

    print('Some parameters are not present in the checkpoint and will be randomly initialized: ', res[0])

    model_args.update(residual_vit_args)
        
    return residual_vit, model_args


def adapt_torch_state_dict(state_dict, num_classes:int):
    """
    Adapt the weights of a Pytorch Vision Transformer state dictionary to a VisionTransformer as defined in this repository. 
    Possibly edit the head to match the number of classes.

    Possible state dicts to be passes are found at https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py
    
    Args:
        state_dict (nn.Module): The state dictionary containing the weights of a Pytorch Vision Transformer.
        num_classes (int): The number of classes for which to adapt the weights.
        
    Returns:
        dict: The adapted state dictionary with updated keys.
    """
    
    new_state_dict = {}
    def adapt_param_name(param):
        p = param.replace('mlp.0', 'mlp.fc1').replace('mlp.3', 'mlp.fc2').replace('heads.head', 'head')
        p = p.replace('mlp.linear_1', 'mlp.fc1').replace('mlp.linear_2', 'mlp.fc2')
        if p.count('self_attention') == 1:
            p = p.replace('self_attention', 'self_attention.self_attention')
        
        if p == 'class_token':
            return 'class_tokens'
        
        p = re.sub(r'encoder_layer_(\d)', r'\1', p)
        return p

    for param_name, param in state_dict.items():
        new_param_name = adapt_param_name(param_name)
        new_state_dict[new_param_name] = param
    
    # if num classes is different from the original, replace the head with a randomly initialized one
    old_head_shape = new_state_dict['head.weight'].shape
    if old_head_shape[0] != num_classes:
        print('Loading weights for a different number of classes. Replacing head with random weights. You should fine-tune the model.')
        new_head_shape = (num_classes, old_head_shape[1])
        new_state_dict['head.weight'] = torch.zeros(new_head_shape)
        new_state_dict['head.bias'] = torch.zeros(num_classes)
    
    return new_state_dict


#################################################################################################################### 
# A dictionary that maps the names of the parameters in the ViT checkpoint to the names of the parameters in the ResidualViT model.
# Obsolete, to be removed.

torch_params_map = {
    # same
    'ln_1.bias' : 'ln_1.bias',
    'ln_1.weight' : 'ln_1.weight',
    'ln_2.bias' : 'ln_2.bias',
    'ln_2.weight' : 'ln_2.weight',
    # vit -> residual vit
    'mlp.0.bias' : 'mlp.fc1.bias',
    'mlp.0.weight' : 'mlp.fc1.weight',
    'mlp.3.bias' : 'mlp.fc2.bias',
    'mlp.3.weight' : 'mlp.fc2.weight',
    'self_attention.in_proj_bias' : 'self_attention.self_attention.in_proj_bias',
    'self_attention.in_proj_weight' : 'self_attention.self_attention.in_proj_weight',
    'self_attention.out_proj.bias' : 'self_attention.self_attention.out_proj.bias',
    'self_attention.out_proj.weight' : 'self_attention.self_attention.out_proj.weight',
    # residual vit -> vit
    'mlp.fc1.bias' : 'mlp.0.bias',
    'mlp.fc1.weight' : 'mlp.0.weight',
    'mlp.fc2.bias' : 'mlp.3.bias',
    'mlp.fc2.weight' : 'mlp.3.weight',
    'self_attention.self_attention.in_proj_bias' : 'self_attention.in_proj_bias',
    'self_attention.self_attention.in_proj_weight' : 'self_attention.in_proj_weight',
    'self_attention.self_attention.out_proj.bias' : 'self_attention.out_proj.bias',
    'self_attention.self_attention.out_proj.weight' : 'self_attention.out_proj.weight'
}

