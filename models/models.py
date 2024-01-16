from .moevit import VisionTransformerMoE
from .residualvit import ResidualVisionTransformer  
from .vit import VisionTransformer
from .EEresidualvit import EEResidualVisionTransformer

import torch
from torch import nn
import re

MODELS_MAP = {
    'visiontransformer': VisionTransformer,
    'VisionTransformer': VisionTransformer,
    'vit': VisionTransformer,
    
    'residualvisiontransformer': ResidualVisionTransformer,
    'ResidualVisionTransformer': ResidualVisionTransformer,
    'residualvit': ResidualVisionTransformer,
    
    'visiontransformermoe': VisionTransformerMoE,
    'VisionTransformerMoE': VisionTransformerMoE,
    'vitmoe': VisionTransformerMoE, 

    'EEResidualVisionTransformer': EEResidualVisionTransformer,
    'eeResidualVisionTransformer': EEResidualVisionTransformer,
    'eeResidualvit': EEResidualVisionTransformer,

}





def adapt_weights(state_dict: nn.Module, num_classes:int):
    # TODO comment and maybe move somewhere else
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
        new_head_shape = (num_classes, old_head_shape[1])
        new_state_dict['head.weight'] = torch.zeros(new_head_shape)
        new_state_dict['head.bias'] = torch.zeros(num_classes)
    
    
    return new_state_dict


def build_model(model_class, model_args, noise_args=None):
    """
    Build a model based on the given model class and arguments. Possibly add noise.
    
    Args:
        model_class (str): The class name of the model.
        model_args (dict): The arguments to be passed to the model constructor.
        noise_settings (dict, optional): The settings for adding noise to the model.
        
    Returns:
        model: The built model.
    """

    # handle the case where we have a pretrained model not from peekvit
    pretrained_weights = model_args.pop('torch_pretrained_weights', False)
    

    model = MODELS_MAP[model_class](**model_args)
    if pretrained_weights:
        state_dict = adapt_weights(pretrained_weights, model_args['num_classes'])
        model.load_state_dict(state_dict, strict=True)

    

    # add noise if requested
    if noise_args is not None and noise_args != {}:
        from utils.utils import add_noise
        noise_module = add_noise(model, **noise_args)
        noise_module.set_value(0.0)
        print('WARNING: Loaded model with noise. Noise will be set to 0.0, you can change this by calling model.noise_module.set_value(new_noise_value)')


    return model