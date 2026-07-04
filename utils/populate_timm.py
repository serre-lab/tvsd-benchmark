import os
import timm
import yaml


def populate_timm():
    """Write a config for every pretrained timm model (uses `transform: timm`)."""
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
