import yaml

import timm
import torch
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
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    if config["model-source"] == "torchvision":
        model, model_name = load_torchvision_model(config)
    elif config["model-source"] == "timm":
        model, model_name = load_timm_model(config)
    elif config["model-source"] == "hmax":
        model, model_name = load_hmax_model(config)
    else:
        raise NotImplementedError(f"Model {config['model-name']} is not supported.")

    hook_interval = config.get("hook-interval", 5)

    return model, model_name, hook_interval


def load_torchvision_model(config: dict):
    if config["model-name"] == "resnet50":
        from torchvision.models import resnet50

        model = resnet50(weights="IMAGENET1K_V1")
        model_name = "resnet50"
    elif config["model-name"] == "alexnet":
        from torchvision.models import alexnet

        model = alexnet(weights="IMAGENET1K_V1")
        model_name = "alexnet"
    elif config["model-name"] == "inception_v3":
        from torchvision.models import inception_v3

        model = inception_v3(weights="IMAGENET1K_V1")
        model_name = "inception_v3"
    else:
        raise NotImplementedError(
            f"Model {config['model-name']} is not supported in torchvision."
        )

    return model, model_name


def load_timm_model(config: dict):
    model = timm.create_model(config["model-name"], pretrained=True)
    model_name = config["model-name"]
    return model, model_name


def load_hmax_model(config: dict):
    if config["model-type"] == "chresmax_v3":
        # HMAX models live in a custom timm fork that is not pip-installable.
        # Import lazily so the rest of the pipeline works without it.
        try:
            from timm.models.RESMAX import chresmax_v3
        except ImportError as e:
            raise ImportError(
                "chresmax_v3 requires the custom pytorch-image-models fork "
                "(https://github.com/npant14/pytorch-image-models) to be installed."
            ) from e

        checkpoint = torch.load(
            config["hmax-info"]["ckpt_path"],
            map_location="cuda" if torch.cuda.is_available() else "cpu",
            weights_only=False,
        )
        model = (
            chresmax_v3(
                num_classes=config["hmax-info"]["num_classes"],
                big_size=config["hmax-info"]["big_size"],
                small_size=config["hmax-info"]["small_size"],
                in_chans=3,
                ip_scale_bands=config["hmax-info"]["ip"],
                classifier_input_size=config["hmax-info"]["classifier_input_size"],
                contrastive_loss=True,
                pyramid=False,
                bypass=True,
                main_route=False,
                c_scoring="v2",
            )
            .to("cuda" if torch.cuda.is_available() else "cpu")
            .eval()
        )
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        model_name = "chresmax_v3"
    else:
        raise NotImplementedError(
            f"Model {config['model-type']} is not supported in HMAX."
        )

    return model, model_name


def resolve_transform(model_config: str):
    """
    Resolve the transformation function based on the model configuration.

    Args:
        model_config (str): Path to the model configuration file.

    Returns:
        transform: The transformation function.
    """
    with open(model_config, "r") as file:
        config = yaml.safe_load(file)

    transform_specs = config.get("transform", {})

    # timm models carry their own input size / crop / normalization in their
    # pretrained config. Setting `transform: timm` builds the model's native
    # eval transform, so every timm model gets the correct preprocessing for
    # free (no need to hand-write a transform per model).
    if transform_specs == "timm":
        return resolve_timm_transform(config["model-name"])

    transform_list = []

    for spec in transform_specs:
        if spec["name"] == "Resize":
            transform_list.append(transforms.Resize((spec["size"][0], spec["size"][1])))
        elif spec["name"] == "ToTensor":
            transform_list.append(transforms.ToTensor())
        elif spec["name"] == "Normalize":
            transform_list.append(
                transforms.Normalize(mean=spec["mean"], std=spec["std"])
            )
        else:
            raise NotImplementedError(f"Transform {spec['name']} is not supported.")

    transform = transforms.Compose(transform_list)
    return transform


def resolve_timm_transform(model_name: str):
    """
    Build a timm model's native evaluation transform from its pretrained
    config (input size, crop, interpolation, normalization).

    This reads the model's `pretrained_cfg` and does not download weights, so
    it works for any pretrained timm model without instantiating it.

    Args:
        model_name (str): Name of the timm model.

    Returns:
        transform: The model's native eval transform.
    """
    from timm.data import create_transform, resolve_data_config

    pretrained_cfg = timm.get_pretrained_cfg(model_name).to_dict()
    data_config = resolve_data_config(args={}, pretrained_cfg=pretrained_cfg)
    return create_transform(**data_config, is_training=False)
