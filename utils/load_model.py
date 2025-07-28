import yaml 
from torchvision import transforms

def load_model(config_path: str):
    """
    Load a model from a given configuration file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        model: The loaded model.
        model_name (str): Name of the loaded model.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    if config['model-source'] == 'torchvision':
        model, model_name = load_torchvision_model(config)
    elif config['model-source'] == 'timm':
        model, model_name = load_timm_model(config)
    elif config['model-source'] == 'hmax':
        model, model_name = load_hmax_model(config)
    else:
        raise NotImplementedError(f"Model {config['model-name']} is not supported.")
    
    hook_interval = config.get('hook-interval', 5)
    
    return model, model_name, hook_interval

def load_torchvision_model(config: dict):
    if config['model-name'] == 'resnet50':
        from torchvision.models import resnet50
        model = resnet50(weights="IMAGENET1K_V1")
        model_name = 'resnet50'
    elif config['model-name'] == 'alexnet':
        from torchvision.models import alexnet
        model = alexnet(weights="IMAGENET1K_V1")
        model_name = 'alexnet'
    else:
        raise NotImplementedError(f"Model {config['model-name']} is not supported in torchvision.")

    return model, model_name

def load_timm_model(config: dict):
    pass

def load_hmax_model(config: dict):
    pass

def resolve_transform(model_config: str):
    """
    Resolve the transformation function based on the model configuration.

    Args:
        model_config (str): Path to the model configuration file.

    Returns:
        transform: The transformation function.
    """
    with open(model_config, 'r') as file:
        config = yaml.safe_load(file)

    transform_specs = config.get('transform', {})
    transform_list = []

    for spec in transform_specs:
        if spec['name'] == 'Resize':
            transform_list.append(transforms.Resize((spec['size'][0], spec['size'][1])))
        elif spec['name'] == 'ToTensor':
            transform_list.append(transforms.ToTensor())
        elif spec['name'] == 'Normalize':
            transform_list.append(transforms.Normalize(
                mean=spec['mean'],
                std=spec['std']
            ))
        else:
            raise NotImplementedError(f"Transform {spec['name']} is not supported.")
    
    transform = transforms.Compose(transform_list)
    return transform
