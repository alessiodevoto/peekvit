from torchvision.models import VisionTransformer as TorchVisionTransformer
from torchvision.models.vision_transformer import EncoderBlock as TorchEncoderBlock
from models.residualvit import ResidualVisionTransformer, ResidualViTBlock
from models.vit import VisionTransformer
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
def from_torch_vit_to_residual_vit(torch_vit_args, torch_weights, residual_vit_args, num_classes:int = 1000):
    """
    Converts a TorchVisionTransformer model to a ResidualVisionTransformer model.

    Args:
        torch_vit_args (dict): Arguments for building the TorchVisionTransformer model.
        torch_weights (torch.nn.Module): Pretrained weights for the TorchVisionTransformer model.
        residual_vit_args (dict): Arguments for building the ResidualVisionTransformer model.
        num_classes (int, optional): Number of output classes. Defaults to 1000.

    Returns:
        residual_vit (ResidualVisionTransformer): Converted ResidualVisionTransformer model.
        torch_vit (TorchVisionTransformer): Original TorchVisionTransformer model.
    """
    
    # build torch vit and possibly load weights
    torch_vit = TorchVisionTransformer(**torch_vit_args)
    if torch_weights is not None:
        torch_vit.load_state_dict(torch_weights.get_state_dict(progress=False))
    
    # build residual vit, with randomly init weights
    if num_classes is not None:
        torch_vit_args['num_classes'] = num_classes
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
            residual_key = torch_params_map[torch_key]
            residual_block_weight_dict[residual_key] = torch_block_weight_dict[torch_key].data.clone()
        residual_block.load_state_dict(residual_block_weight_dict)
    
    # if torch vit has class token, copy it to residual vit
    torch_params = torch_vit.state_dict()

    # copy class token, encoder ln and pos embedding
    residual_vit.class_tokens.data = torch_params['class_token'].data.clone()
    residual_vit.encoder.ln.weight.data = torch_params['encoder.ln.weight'].data.clone()
    residual_vit.encoder.ln.bias.data = torch_params['encoder.ln.bias'].data.clone()
    residual_vit.conv_proj.weight.data = torch_params['conv_proj.weight'].data.clone()
    residual_vit.conv_proj.bias.data = torch_params['conv_proj.bias'].data.clone()
    residual_vit.encoder.pos_embedding.data = torch_params['encoder.pos_embedding'].data.clone()

    if num_classes != 1000:
        print('num_classes != 1000. Head is randomly initialized')
    else:
        # copy head
        residual_vit.head.weight.data = torch_params['heads.head.weight'].data.clone()
        residual_vit.head.bias.data = torch_params['heads.head.bias'].data.clone()
    
    return residual_vit, torch_vit

    

@torch.no_grad()
def from_vit_to_residual_vit(vit_checkpoint, residual_vit_args:dict = None):
    """
    Converts a Vision Transformer (ViT) checkpoint to a Residual Vision Transformer (ResidualViT) model.

    Args:
        vit_checkpoint (str): Path to the ViT checkpoint file.
        residual_vit_args (dict, optional): Additional arguments to initialize the ResidualViT model. Defaults to None.

    Returns:
        ResidualViT: The converted ResidualViT model.
    """
    
    # load weights from vit checkpoint
    state = torch.load(vit_checkpoint)
    print('Loading weights from class: ', state['model_class'])
    vit_weights = state['state_dict']
    model_args = state['model_args']

    # build residual vit, with randomly init weights
    residual_vit = ResidualVisionTransformer(**model_args, **residual_vit_args)

    # copy weights from vit to residual vit
    res = residual_vit.load_state_dict(vit_weights, strict=False)

    print('Some parameters are not present in the checkpoint and will be randomly initialized: ', res[0])
        
    return residual_vit





