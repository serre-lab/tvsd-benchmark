"""Tests for utils.load_model, focused on timm compatibility."""
import os
import tempfile

import pytest
import yaml


def test_module_imports_without_hmax_fork():
    """load_model must import even when the custom HMAX fork is absent.

    Regression test: the module previously did a top-level import of a model
    from a hardcoded cluster path, which made it unimportable everywhere else.
    """
    import utils.load_model  # noqa: F401


def _write_config(tmp_path, **overrides):
    config = {
        "model-name": "resnet50",
        "model-type": "timm",
        "model-source": "timm",
        "hook-interval": 5,
        "transform": "timm",
    }
    config.update(overrides)
    path = os.path.join(tmp_path, "model.yaml")
    with open(path, "w") as f:
        yaml.dump(config, f)
    return path


def test_resolve_timm_transform_uses_native_input_size():
    """`transform: timm` should build the model's native eval transform."""
    from torchvision import transforms

    from utils.load_model import resolve_transform

    with tempfile.TemporaryDirectory() as tmp_path:
        # vit_base_patch16_384 has a non-default 384px input size, so it
        # verifies we honor per-model config rather than a hardcoded 224.
        config_path = _write_config(tmp_path, **{"model-name": "vit_base_patch16_384"})
        transform = resolve_transform(config_path)

    assert isinstance(transform, transforms.Compose)
    crops = [t for t in transform.transforms if isinstance(t, transforms.CenterCrop)]
    assert crops, "expected a CenterCrop in the timm transform"
    assert tuple(crops[0].size) == (384, 384)


def test_resolve_transform_still_supports_explicit_specs():
    """The existing spec-based transform path must keep working."""
    from torchvision import transforms

    from utils.load_model import resolve_transform

    with tempfile.TemporaryDirectory() as tmp_path:
        config_path = _write_config(
            tmp_path,
            **{
                "model-source": "torchvision",
                "transform": [
                    {"name": "Resize", "size": [224, 224]},
                    {"name": "ToTensor"},
                    {
                        "name": "Normalize",
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225],
                    },
                ],
            },
        )
        transform = resolve_transform(config_path)

    assert isinstance(transform, transforms.Compose)
    assert any(isinstance(t, transforms.Normalize) for t in transform.transforms)
