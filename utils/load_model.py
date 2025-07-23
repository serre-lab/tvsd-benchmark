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
    
    if config['model-name'] == 'resnet50':
        from torchvision.models import resnet50
        model = resnet50(weights="IMAGENET1K_V1")
        model_name = 'resnet50'
    else:
        raise NotImplementedError(f"Model {config['model-name']} is not supported.")
    
    return model, model_name

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
