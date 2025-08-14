import os
import timm
import yaml


def populate_timm():
    standard_imagenet_transform = [
        {"name": "Resize", "size": [224, 224]},
        {"name": "ToTensor"},
        {
            "name": "Normalize",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    ]
    for model in timm.list_models(pretrained=True):
        model_config = {
            "model-name": model,
            "model-type": "timm",
            "model-source": "timm",
            "hook-interval": 5,
            "transform": standard_imagenet_transform,
        }
        os.makedirs(f"configs/timm", exist_ok=True)
        with open(f"configs/timm/{model}.yaml", "w") as f:
            yaml.dump(model_config, f)


if __name__ == "__main__":
    populate_timm()
