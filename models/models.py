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


def build_model(model_class, model_args, noise_args):
    model = MODELS_MAP[model_class](**model_args)
    return model