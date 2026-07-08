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


def test_resolve_timm_transform_unknown_model_raises_clearly():
    """An unknown timm model name must raise a clear ValueError.

    Regression: get_pretrained_cfg returns None for unknown models, which
    previously blew up with an opaque AttributeError on `.to_dict()`.
    """
    from utils.load_model import resolve_transform

    with tempfile.TemporaryDirectory() as tmp_path:
        config_path = _write_config(tmp_path, **{"model-name": "definitely_not_a_timm_model"})
        with pytest.raises(ValueError, match="Unknown timm model"):
            resolve_transform(config_path)


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


def test_load_model_unknown_source_raises():
    """An unrecognized model-source must fail loudly, not silently."""
    from utils.load_model import load_model

    with tempfile.TemporaryDirectory() as tmp_path:
        config_path = _write_config(tmp_path, **{"model-source": "bogus"})
        with pytest.raises(NotImplementedError):
            load_model(config_path)


def test_resolve_transform_unsupported_spec_raises():
    """An unknown transform spec name must raise rather than be dropped."""
    from utils.load_model import resolve_transform

    with tempfile.TemporaryDirectory() as tmp_path:
        config_path = _write_config(
            tmp_path,
            **{
                "model-source": "torchvision",
                "transform": [{"name": "NotARealTransform"}],
            },
        )
        with pytest.raises(NotImplementedError, match="NotARealTransform"):
            resolve_transform(config_path)


def test_load_hmax_unsupported_type_raises():
    """load_hmax_model only knows chresmax_v3; others must raise."""
    from utils.load_model import load_hmax_model

    with pytest.raises(NotImplementedError):
        load_hmax_model({"model-type": "unknown_hmax", "model-name": "x"})


def test_load_hmax_chresmax_reports_missing_fork():
    """When the custom timm fork is absent, chresmax_v3 must raise a clear
    ImportError pointing at the fork -- not an opaque error deeper in the load."""
    from utils.load_model import load_hmax_model

    try:
        import timm.models.RESMAX  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="fork"):
            load_hmax_model({"model-type": "chresmax_v3"})
    else:  # pragma: no cover - only when the fork is actually installed
        pytest.skip("custom timm fork is installed; missing-fork path not exercised")
