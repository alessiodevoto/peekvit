from torchvision.models import VisionTransformer as TorchVisionTransformer
from torchvision.models.vision_transformer import EncoderBlock as TorchEncoderBlock
from models.residualvit import ResidualVisionTransformer, ResidualViTBlock
from models.vit import VisionTransformer
import torch




params_map = {
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
def from_torch_vit_to_residual_vit(torch_vit_args, torch_weights, residual_vit_args):

    # build torch vit and possibly load weights
    torch_vit = TorchVisionTransformer(**torch_vit_args)
    if torch_weights is not None:
        torch_vit.load_state_dict(torch_weights.get_state_dict(progress=False))
    
    # build residual vit, with randomly init weights
    residual_vit = ResidualVisionTransformer(**torch_vit_args, **residual_vit_args)

    # copy weights from torch vit to residual vit
    torch_vit_blocks = torch_vit.encoder.layers
    residual_vit_blocks = residual_vit.encoder.layers

    # iterate over blocks and copy weights
    for torch_block, residual_block in zip(torch_vit_blocks, residual_vit_blocks):
        # copy weights from torch block to residual block
        torch_block_weight_dict = torch_block.state_dict()
        residual_block_weight_dict = residual_block.state_dict()
        for torch_key in torch_block_weight_dict.keys():
            # parameters have different names in torch and residual vit
            residual_key = params_map[torch_key]
            residual_block_weight_dict[residual_key] = torch_block_weight_dict[torch_key].data.clone()
        residual_block.load_state_dict(residual_block_weight_dict)
    
    # if torch vit has class token, copy it to residual vit
        
    
    return residual_vit, torch_vit

    


def from_vit_to_residual_vit(vit_args, vit_weights, residual_vit_args):
    # build vit and possibly load weights
    vit = VisionTransformer(**vit_args)
    if vit_weights is not None:
        vit.load_state_dict(vit_weights)
    
    # build residual vit, with randomly init weights
    residual_vit = ResidualVisionTransformer(**vit_args, **residual_vit_args)

    # copy weights from vit to residual vit
    vit_blocks = vit.encoder.layers
    residual_vit_blocks = residual_vit.encoder.layers

    # iterate over blocks and copy weights
    for vit_block, residual_block in zip(vit_blocks, residual_vit_blocks):
        # copy weights from vit block to residual block
        vit_block_weight_dict = vit_block.state_dict()
        residual_block_weight_dict = residual_block.state_dict()
        for vit_key in vit_block_weight_dict.keys():
            # parameters have different names in vit and residual vit
            residual_key = params_map[vit_key]
            residual_block_weight_dict[residual_key] = vit_block_weight_dict[vit_key].data.clone()
        residual_block.load_state_dict(residual_block_weight_dict)
    
    # if vit has class token, copy it to residual vit
        
    
    return residual_vit, vit





