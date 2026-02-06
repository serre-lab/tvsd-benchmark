"""Tests for timm model loading and activation extraction."""

import pytest
import torch
import numpy as np
import os
import tempfile
import yaml
from pathlib import Path

from utils.load_model import load_model
from utils.hooks import Activations
from utils.timm_helpers import detect_model_family, process_activations


class TestTimmModelLoading:
    """Tests for loading timm models via config."""

    def test_load_timm_resnet(self, tmp_path):
        """Test loading a timm ResNet model."""
        config_path = tmp_path / "resnet_config.yaml"
        config = {
            "model-name": "resnet50",
            "model-source": "timm",
            "pretrained": False,
            "hook-interval": 5,
            "transform": [
                {"name": "Resize", "size": [224, 224]},
                {"name": "ToTensor"},
                {"name": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            ],
        }
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        model, model_name, hook_interval = load_model(str(config_path))
        
        assert model is not None
        assert model_name == "resnet50"
        assert hook_interval == 5
        assert hasattr(model, "eval")

    def test_load_timm_vit(self, tmp_path):
        """Test loading a timm Vision Transformer model."""
        config_path = tmp_path / "vit_config.yaml"
        config = {
            "model-name": "vit_tiny_patch16_224",
            "model-source": "timm",
            "pretrained": False,
            "hook-interval": 5,
            "transform": [
                {"name": "Resize", "size": [224, 224]},
                {"name": "ToTensor"},
                {"name": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            ],
        }
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        model, model_name, hook_interval = load_model(str(config_path))
        
        assert model is not None
        assert model_name == "vit_tiny_patch16_224"
        assert hook_interval == 5

    def test_load_timm_convnext(self, tmp_path):
        """Test loading a timm ConvNeXt model."""
        config_path = tmp_path / "convnext_config.yaml"
        config = {
            "model-name": "convnext_tiny",
            "model-source": "timm",
            "pretrained": False,
            "hook-interval": 5,
            "transform": [
                {"name": "Resize", "size": [224, 224]},
                {"name": "ToTensor"},
                {"name": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            ],
        }
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        model, model_name, hook_interval = load_model(str(config_path))
        
        assert model is not None
        assert model_name == "convnext_tiny"
        assert hook_interval == 5

    def test_load_timm_swin(self, tmp_path):
        """Test loading a timm Swin Transformer model."""
        config_path = tmp_path / "swin_config.yaml"
        config = {
            "model-name": "swin_tiny_patch4_window7_224",
            "model-source": "timm",
            "pretrained": False,
            "hook-interval": 5,
            "transform": [
                {"name": "Resize", "size": [224, 224]},
                {"name": "ToTensor"},
                {"name": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            ],
        }
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        model, model_name, hook_interval = load_model(str(config_path))
        
        assert model is not None
        assert model_name == "swin_tiny_patch4_window7_224"
        assert hook_interval == 5


class TestTimmModelFamilyDetection:
    """Tests for model family detection."""

    def test_detect_vit(self):
        """Test detecting Vision Transformer models."""
        assert detect_model_family("vit_base_patch16_224") == "vit"
        assert detect_model_family("vit_tiny_patch16_224") == "vit"
        assert detect_model_family("vision_transformer_small") == "vit"

    def test_detect_swin(self):
        """Test detecting Swin Transformer models."""
        assert detect_model_family("swin_base_patch4_window7_224") == "swin"
        assert detect_model_family("swin_tiny_patch4_window7_224") == "swin"

    def test_detect_convnext(self):
        """Test detecting ConvNeXt models."""
        assert detect_model_family("convnext_base") == "convnext"
        assert detect_model_family("convnext_tiny") == "convnext"

    def test_detect_resnet(self):
        """Test detecting ResNet models."""
        assert detect_model_family("resnet50") == "resnet"
        assert detect_model_family("resnet101") == "resnet"

    def test_detect_default(self):
        """Test default detection for unknown models."""
        assert detect_model_family("some_other_model") == "default"


class TestTimmActivationExtraction:
    """Tests for activation extraction from timm models."""

    @pytest.fixture
    def dummy_batch(self):
        """Create a dummy batch of images."""
        return torch.randn(2, 3, 224, 224)

    @pytest.fixture
    def output_dir(self, tmp_path):
        """Create a temporary output directory."""
        return str(tmp_path)

    def test_resnet_activations(self, dummy_batch, output_dir):
        """Test activation extraction from ResNet (easy baseline)."""
        import timm
        model = timm.create_model("resnet50", pretrained=False)
        model.eval()

        activations = Activations(
            output_dir=output_dir,
            model_name="resnet50",
            dataset_name="test",
            pca_components=None,
        )

        # Register hooks on a few layers
        layer_names = ["layer1.0", "layer2.0", "layer3.0"]
        activations.register(model, layer_names)
        
        # Forward pass
        activations.set_batch(0)
        activations.set_training_mode(False)
        with torch.no_grad():
            _ = model(dummy_batch)
        activations.finalize_batch_inference()

        # Check activations were captured
        assert len(activations.activations) == len(layer_names)
        for layer_name in layer_names:
            assert layer_name in activations.activations
            assert 0 in activations.activations[layer_name]
            # Check shape: should be [batch_size, features]
            act_tensor = activations.activations[layer_name][0][0]
            assert act_tensor.shape[0] == dummy_batch.shape[0]
            assert len(act_tensor.shape) == 2

    def test_vit_activations(self, dummy_batch, output_dir):
        """Test activation extraction from Vision Transformer (token/cls shapes)."""
        import timm
        model = timm.create_model("vit_tiny_patch16_224", pretrained=False)
        model.eval()

        activations = Activations(
            output_dir=output_dir,
            model_name="vit_tiny_patch16_224",
            dataset_name="test",
            pca_components=None,
        )

        # Register hooks on transformer blocks
        layer_names = ["blocks.0", "blocks.5", "blocks.11"]
        activations.register(model, layer_names)
        
        # Forward pass
        activations.set_batch(0)
        activations.set_training_mode(False)
        with torch.no_grad():
            _ = model(dummy_batch)
        activations.finalize_batch_inference()

        # Check activations were captured
        assert len(activations.activations) > 0
        for layer_name in activations.activations.keys():
            assert 0 in activations.activations[layer_name]
            # Check shape: should be [batch_size, features] after processing
            act_tensor = activations.activations[layer_name][0][0]
            assert act_tensor.shape[0] == dummy_batch.shape[0]
            assert len(act_tensor.shape) == 2

    def test_convnext_activations(self, dummy_batch, output_dir):
        """Test activation extraction from ConvNeXt (stages)."""
        import timm
        model = timm.create_model("convnext_tiny", pretrained=False)
        model.eval()

        activations = Activations(
            output_dir=output_dir,
            model_name="convnext_tiny",
            dataset_name="test",
            pca_components=None,
        )

        # Register hooks on stages
        layer_names = ["stages.0", "stages.1", "stages.2"]
        activations.register(model, layer_names)
        
        # Forward pass
        activations.set_batch(0)
        activations.set_training_mode(False)
        with torch.no_grad():
            _ = model(dummy_batch)
        activations.finalize_batch_inference()

        # Check activations were captured
        assert len(activations.activations) > 0
        for layer_name in activations.activations.keys():
            assert 0 in activations.activations[layer_name]
            # Check shape: should be [batch_size, features]
            act_tensor = activations.activations[layer_name][0][0]
            assert act_tensor.shape[0] == dummy_batch.shape[0]
            assert len(act_tensor.shape) == 2

    def test_swin_activations(self, dummy_batch, output_dir):
        """Test activation extraction from Swin Transformer (hierarchical tokens)."""
        import timm
        model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False)
        model.eval()

        activations = Activations(
            output_dir=output_dir,
            model_name="swin_tiny_patch4_window7_224",
            dataset_name="test",
            pca_components=None,
        )

        # Register hooks on layers
        layer_names = ["layers.0", "layers.1", "layers.2"]
        activations.register(model, layer_names)
        
        # Forward pass
        activations.set_batch(0)
        activations.set_training_mode(False)
        with torch.no_grad():
            _ = model(dummy_batch)
        activations.finalize_batch_inference()

        # Check activations were captured
        assert len(activations.activations) > 0
        for layer_name in activations.activations.keys():
            assert 0 in activations.activations[layer_name]
            # Check shape: should be [batch_size, features]
            act_tensor = activations.activations[layer_name][0][0]
            assert act_tensor.shape[0] == dummy_batch.shape[0]
            assert len(act_tensor.shape) == 2

    def test_activation_shape_processing(self):
        """Test that activation processing handles different output shapes correctly."""
        batch_size = 4
        
        # Test ViT-style output (batch, num_tokens, embed_dim)
        vit_output = torch.randn(batch_size, 197, 768)  # 196 patches + 1 cls token
        processed = process_activations(vit_output, "vit")
        assert processed.shape == (batch_size, 197 * 768)
        
        # Test ConvNeXt-style output (batch, channels, H, W)
        convnext_output = torch.randn(batch_size, 96, 56, 56)
        processed = process_activations(convnext_output, "convnext")
        assert processed.shape == (batch_size, 96 * 56 * 56)
        
        # Test Swin-style output (batch, H*W, embed_dim)
        swin_output = torch.randn(batch_size, 3136, 96)  # 56*56 patches
        processed = process_activations(swin_output, "swin")
        assert processed.shape == (batch_size, 3136 * 96)

    def test_no_crashes_on_forward_pass(self, dummy_batch, output_dir):
        """Test that models don't crash during forward pass with hooks."""
        import timm
        
        models_to_test = [
            "resnet50",
            "vit_tiny_patch16_224",
            "convnext_tiny",
            "swin_tiny_patch4_window7_224",
        ]
        
        for model_name in models_to_test:
            model = timm.create_model(model_name, pretrained=False)
            model.eval()
            
            activations = Activations(
                output_dir=output_dir,
                model_name=model_name,
                dataset_name="test",
                pca_components=None,
            )
            
            # Get some layer names
            layer_names = [name for name, _ in model.named_modules()]
            # Take every 50th layer to avoid too many hooks
            sample_layers = layer_names[::50][:3]
            
            if sample_layers:
                activations.register(model, sample_layers)
                activations.set_batch(0)
                activations.set_training_mode(False)
                
                # This should not crash
                with torch.no_grad():
                    _ = model(dummy_batch)
                activations.finalize_batch_inference()


class TestTimmActivationSaving:
    """Tests for saving timm model activations."""

    @pytest.fixture
    def dummy_batch(self):
        """Create a dummy batch of images."""
        return torch.randn(2, 3, 224, 224)

    @pytest.fixture
    def output_dir(self, tmp_path):
        """Create a temporary output directory."""
        return str(tmp_path)

    def test_save_resnet_activations(self, dummy_batch, output_dir):
        """Test saving ResNet activations to disk."""
        import timm
        model = timm.create_model("resnet50", pretrained=False)
        model.eval()

        activations = Activations(
            output_dir=output_dir,
            model_name="resnet50",
            dataset_name="test",
            pca_components=None,
        )

        layer_names = ["layer1.0"]
        activations.register(model, layer_names)
        
        activations.set_batch(0)
        activations.set_training_mode(False)
        with torch.no_grad():
            _ = model(dummy_batch)
        activations.finalize_batch_inference()
        
        # Save and check file exists
        activations.save()
        
        expected_path = Path(output_dir) / "activations" / "test" / "resnet50" / "layer1.0" / "activations.pt"
        assert expected_path.exists()
        
        # Load and verify
        loaded = torch.load(expected_path)
        assert loaded.shape[0] == dummy_batch.shape[0]
        assert len(loaded.shape) == 2
