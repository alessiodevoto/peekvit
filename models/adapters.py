from models.residualvit import ResidualVisionTransformer
from models.EEresidualvit import EEResidualVisionTransformer
import torch



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



@torch.no_grad()
def from_vit_to_residual_vit(vit_checkpoint, residual_vit_args:dict = None, model_args:dict = None):
    """
    Converts a Vision Transformer (ViT) checkpoint to a Residual Vision Transformer (ResidualViT) model.

    Args:
        vit_checkpoint (str): Path to the ViT checkpoint file.
        residual_vit_args (dict, optional): Additional arguments to initialize the ResidualViT model. Defaults to None.
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
    model_args.pop('pretrained', None)

    # build residual vit, with randomly init weights
    residual_vit = ResidualVisionTransformer(**model_args, **residual_vit_args)

    # copy weights from vit to residual vit
    res = residual_vit.load_state_dict(vit_weights, strict=False)

    print('Some parameters are not present in the checkpoint and will be randomly initialized: ', res[0])

    model_args.update(residual_vit_args)
        
    return residual_vit, model_args


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





