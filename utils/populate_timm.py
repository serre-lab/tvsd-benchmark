import os
import timm
import yaml


def populate_timm():
    """
    Generate a config file for every pretrained timm model.

    Each config uses `transform: timm`, which tells `resolve_transform` to
    build the model's native eval transform from its pretrained config at load
    time (correct input size, crop, interpolation, and normalization). This
    keeps every model correct without hand-writing a transform per model.
    """
    os.makedirs("configs/timm", exist_ok=True)
    for model in timm.list_models(pretrained=True):
        model_config = {
            "model-name": model,
            "model-type": "timm",
            "model-source": "timm",
            "hook-interval": 5,
            "transform": "timm",
        }
        with open(f"configs/timm/{model}.yaml", "w") as f:
            yaml.dump(model_config, f)


if __name__ == "__main__":
    populate_timm()
