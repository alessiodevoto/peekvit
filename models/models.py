from .moevit import VisionTransformerMoE
from .residualvit import ResidualVisionTransformer  
from .vit import VisionTransformer
from torchvision.models import VisionTransformer as TorchVisionTransformer

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

    'torchvisiontransformer': TorchVisionTransformer,
    'TorchVisionTransformer': TorchVisionTransformer,
    'torchvit': TorchVisionTransformer
}


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
    
    model = MODELS_MAP[model_class](**model_args)
    
    if noise_args is not None and noise_args != {}:
        from utils.utils import add_noise
        noise_module = add_noise(model, **noise_args)
        noise_module.set_value(0.0)
        print('WARNING: Loaded model with noise. Noise will be set to 0.0, you can change this by calling model.noise_module.set_value(new_noise_value)')


    return model