from .moevit import VisionTransformerMoE
from .residualvit import ResidualVisionTransformer  
from .vit import VisionTransformer

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
}


def build_model(model_class, model_args, noise_settings=None):
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

    if noise_settings is not None:
        print(noise_settings)
        from peekvit.utils import add_noise
        noise_layers = noise_settings['layers']
        for layer in noise_layers:
            model = add_noise(model, layer, **noise_settings)

    return model