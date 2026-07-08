"""Tests for utils.hooks.Activations (layer resolution + capture logic)."""

import torch
import torch.nn as nn

from utils.hooks import Activations


def _toy_model():
    # index 0: Linear(4->3), index 1: ReLU, index 2: Linear(3->2)
    return nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))


class TestResolveLayer:
    def test_digit_path_indexes_sequential(self):
        model = _toy_model()
        act = Activations(output_dir="x", model_name="m", dataset_name="d")
        layer = act._resolve_layer(model, "0")
        assert isinstance(layer, nn.Linear)
        assert layer.in_features == 4

    def test_attribute_path(self):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Sequential(nn.Linear(4, 3))

        act = Activations("x", "m", "d")
        layer = act._resolve_layer(Net(), "backbone.0")
        assert isinstance(layer, nn.Linear)


class TestRegister:
    def test_register_attaches_hooks(self):
        model = _toy_model()
        act = Activations("x", "m", "d")
        act.register(model, ["0", "2"])
        assert len(act.hooks) == 2

    def test_unresolvable_layer_is_skipped(self):
        model = _toy_model()
        act = Activations("x", "m", "d")
        # Should not raise; the bad name is skipped and no hook is attached.
        act.register(model, ["nonexistent"])
        assert len(act.hooks) == 0


class TestInferenceCapture:
    def test_forward_stores_float16_activations_per_batch(self):
        model = _toy_model()
        act = Activations("x", "m", "d")
        act.register(model, ["0", "2"])

        act.set_batch(0)
        _ = model(torch.randn(5, 4))
        act.finalize_batch_inference()

        captured = act.get()
        assert set(captured.keys()) == {"0", "2"}
        tensor = captured["0"][0][0]
        # layer 0 is Linear(4->3), flattened over batch of 5
        assert tuple(tensor.shape) == (5, 3)
        assert tensor.dtype == torch.float16

    def test_clear_resets_state(self):
        model = _toy_model()
        act = Activations("x", "m", "d")
        act.register(model, ["0"])
        act.set_batch(0)
        _ = model(torch.randn(2, 4))
        act.finalize_batch_inference()
        assert act.get()

        act.clear()
        assert act.get() == {}
        assert act.hooks == []


class TestTrainingPCA:
    def test_ipca_components_clamped_to_feature_count(self):
        model = _toy_model()
        # Ask for more PCA components than the layer has features.
        act = Activations("x", "m", "d", pca_components=10)
        act.register(model, ["0"])
        act.set_training_mode(True)

        act.set_batch(0)
        _ = model(torch.randn(5, 4))
        act.finalize_batch_training()

        ipca = act.get_ipca_models()["0"]
        # layer 0 emits 3 features, so n_components is clamped from 10 to 3.
        assert ipca.n_components == 3
