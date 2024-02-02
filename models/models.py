from .moevit import VisionTransformerMoE
from .residualvit import ResidualVisionTransformer  
from .vit import VisionTransformer
from .eeresidualvit import EEResidualVisionTransformer
from .adapters import adapt_torch_state_dict
from .encdecresidualvit import ResidualVisionTransformerWithDecoder


###########################################################################################################################

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

    'ResidualVisionTransformerWithDecoder': ResidualVisionTransformerWithDecoder,

}


###########################################################################################################################


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

    if model_class not in MODELS_MAP:
        raise ValueError(f'Unknown model class {model_class}. Available models are {list(MODELS_MAP.keys())}')

    # handle the case where we have a pretrained model not from peekvit
    torch_pretrained_weights = model_args.pop('torch_pretrained_weights', None)
    

    model = MODELS_MAP[model_class](**model_args)
    if torch_pretrained_weights:
        state_dict = adapt_torch_state_dict(torch_pretrained_weights, model_args['num_classes'])
        model.load_state_dict(state_dict, strict=True)

    
    # add noise if requested
    if noise_args is not None and noise_args != {}:
        from utils.utils import add_noise
        noise_module = add_noise(model, **noise_args)
        noise_module.set_value(0.0)
        print('Loaded model with noise. Noise will be set to 0.0, you can change this by calling model.noise_module.set_value(new_noise_value)')


    return model