"""Tests for utils.dataset module."""
import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
import h5py
import tempfile
import os

from utils.dataset import TVSD_BaseDataset, TVSD_Dataset, TVSD_TestDataset, THINGS_Dataset


class TestTHINGSDataset:
    """Tests for THINGS_Dataset class."""

    def test_initialization(self):
        """Test dataset initialization."""
        root_dir = "/test/path"
        paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
        dataset = THINGS_Dataset(root_dir, paths, transform=None)

        assert dataset.root_dir == root_dir
        assert dataset.paths == paths
        assert dataset.transform is None
        assert len(dataset) == 3

    def test_len(self):
        """Test __len__ method."""
        paths = ["img1.jpg", "img2.jpg"]
        dataset = THINGS_Dataset("/test", paths)
        assert len(dataset) == 2


class TestTVSDBaseDataset:
    """Tests for TVSD_BaseDataset class."""

    def test_get_arrays_from_region_monkeyF_V1(self):
        """Test array indices for monkeyF V1 region."""
        with patch.object(TVSD_BaseDataset, '_get_paths', return_value=[]):
            with patch.object(TVSD_BaseDataset, '_get_responses', return_value=(None, None)):
                dataset = TVSD_BaseDataset(monkey="monkeyF", region="V1")
                arrays = dataset._get_arrays_from_region()
                assert arrays == list(range(0, 8))

    def test_get_arrays_from_region_monkeyF_IT(self):
        """Test array indices for monkeyF IT region."""
        with patch.object(TVSD_BaseDataset, '_get_paths', return_value=[]):
            with patch.object(TVSD_BaseDataset, '_get_responses', return_value=(None, None)):
                dataset = TVSD_BaseDataset(monkey="monkeyF", region="IT")
                arrays = dataset._get_arrays_from_region()
                assert arrays == list(range(8, 13))

    def test_get_arrays_from_region_monkeyF_V4(self):
        """Test array indices for monkeyF V4 region."""
        with patch.object(TVSD_BaseDataset, '_get_paths', return_value=[]):
            with patch.object(TVSD_BaseDataset, '_get_responses', return_value=(None, None)):
                dataset = TVSD_BaseDataset(monkey="monkeyF", region="V4")
                arrays = dataset._get_arrays_from_region()
                assert arrays == list(range(13, 16))

    def test_get_arrays_from_region_monkeyN_V1(self):
        """Test array indices for monkeyN V1 region."""
        with patch.object(TVSD_BaseDataset, '_get_paths', return_value=[]):
            with patch.object(TVSD_BaseDataset, '_get_responses', return_value=(None, None)):
                dataset = TVSD_BaseDataset(monkey="monkeyN", region="V1")
                arrays = dataset._get_arrays_from_region()
                assert arrays == list(range(0, 8))

    def test_get_arrays_from_region_monkeyN_V4(self):
        """Test array indices for monkeyN V4 region."""
        with patch.object(TVSD_BaseDataset, '_get_paths', return_value=[]):
            with patch.object(TVSD_BaseDataset, '_get_responses', return_value=(None, None)):
                dataset = TVSD_BaseDataset(monkey="monkeyN", region="V4")
                arrays = dataset._get_arrays_from_region()
                assert arrays == list(range(8, 12))

    def test_get_arrays_from_region_monkeyN_IT(self):
        """Test array indices for monkeyN IT region."""
        with patch.object(TVSD_BaseDataset, '_get_paths', return_value=[]):
            with patch.object(TVSD_BaseDataset, '_get_responses', return_value=(None, None)):
                dataset = TVSD_BaseDataset(monkey="monkeyN", region="IT")
                arrays = dataset._get_arrays_from_region()
                assert arrays == list(range(12, 16))

    def test_get_arrays_from_region_invalid(self):
        """Test invalid region raises ValueError."""
        with patch.object(TVSD_BaseDataset, '_get_paths', return_value=[]):
            with pytest.raises(ValueError, match="Invalid region"):
                TVSD_BaseDataset(monkey="monkeyF", region="INVALID")

    def test_get_array_idxs(self):
        """Test that array_idxs are correctly computed from arrays."""
        with patch.object(TVSD_BaseDataset, '_get_paths', return_value=[]):
            with patch.object(TVSD_BaseDataset, '_get_responses', return_value=(None, None)):
                dataset = TVSD_BaseDataset(monkey="monkeyF", region="V1")
                # V1 has arrays 0-7, each array has 64 channels
                # So indices should be 0-511
                expected = list(range(0, 8 * 64))
                assert dataset.array_idxs == expected

    def test_getitem_int_index(self):
        """Test __getitem__ with integer index."""
        responses = torch.randn(100, 512)
        reliability = torch.randn(512)

        with patch.object(TVSD_BaseDataset, '_get_paths', return_value=[]):
            with patch.object(TVSD_BaseDataset, '_get_responses', return_value=(responses, reliability)):
                dataset = TVSD_BaseDataset(monkey="monkeyF", region="V1")
                response, rel = dataset[0]
                assert torch.allclose(response, responses[0])
                assert torch.allclose(rel, reliability)

    def test_getitem_slice_index(self):
        """Test __getitem__ with slice index."""
        responses = torch.randn(100, 512)
        reliability = torch.randn(512)

        with patch.object(TVSD_BaseDataset, '_get_paths', return_value=[]):
            with patch.object(TVSD_BaseDataset, '_get_responses', return_value=(responses, reliability)):
                dataset = TVSD_BaseDataset(monkey="monkeyF", region="V1")
                response, rel = dataset[0:10]
                assert torch.allclose(response, responses[0:10])
                assert torch.allclose(rel, reliability)

    def test_getitem_list_index(self):
        """Test __getitem__ with list index."""
        responses = torch.randn(100, 512)
        reliability = torch.randn(512)

        with patch.object(TVSD_BaseDataset, '_get_paths', return_value=[]):
            with patch.object(TVSD_BaseDataset, '_get_responses', return_value=(responses, reliability)):
                dataset = TVSD_BaseDataset(monkey="monkeyF", region="V1")
                indices = [0, 5, 10]
                response, rel = dataset[indices]
                assert torch.allclose(response, responses[indices])
                assert torch.allclose(rel, reliability)

    def test_getitem_invalid_index(self):
        """Test __getitem__ with invalid index type."""
        responses = torch.randn(100, 512)
        reliability = torch.randn(512)

        with patch.object(TVSD_BaseDataset, '_get_paths', return_value=[]):
            with patch.object(TVSD_BaseDataset, '_get_responses', return_value=(responses, reliability)):
                dataset = TVSD_BaseDataset(monkey="monkeyF", region="V1")
                with pytest.raises(TypeError, match="Unsupported index type"):
                    _ = dataset["invalid"]


class TestTVSDTestDataset:
    """Tests for TVSD_TestDataset class."""

    def test_compute_reliability_basic(self):
        """Test basic reliability computation."""
        # Create synthetic data: 30 reps, 100 stimuli, 10 neuroids
        np.random.seed(42)
        data = np.random.randn(30, 100, 10)

        with patch.object(TVSD_TestDataset, '_get_paths', return_value=[]):
            with patch.object(TVSD_TestDataset, '_get_responses'):
                dataset = TVSD_TestDataset(monkey="monkeyF", region="V1")
                reliability = dataset._compute_reliability(data, n_boot=10, random_state=42)

                assert reliability.shape == (10,)
                assert np.all(reliability >= -1) and np.all(reliability <= 1)

    def test_compute_reliability_reproducibility(self):
        """Test that same seed gives identical reliability scores."""
        np.random.seed(42)
        data = np.random.randn(30, 100, 10)

        with patch.object(TVSD_TestDataset, '_get_paths', return_value=[]):
            with patch.object(TVSD_TestDataset, '_get_responses'):
                dataset = TVSD_TestDataset(monkey="monkeyF", region="V1")
                rel1 = dataset._compute_reliability(data, n_boot=10, random_state=42)
                rel2 = dataset._compute_reliability(data, n_boot=10, random_state=42)

                assert np.allclose(rel1, rel2)

    def test_compute_reliability_different_seeds(self):
        """Test that different seeds give different reliability scores."""
        np.random.seed(42)
        data = np.random.randn(30, 100, 10)

        with patch.object(TVSD_TestDataset, '_get_paths', return_value=[]):
            with patch.object(TVSD_TestDataset, '_get_responses'):
                dataset = TVSD_TestDataset(monkey="monkeyF", region="V1")
                rel1 = dataset._compute_reliability(data, n_boot=10, random_state=42)
                rel2 = dataset._compute_reliability(data, n_boot=10, random_state=99)

                # Should be different but correlated
                assert not np.allclose(rel1, rel2)
                from scipy.stats import pearsonr
                corr, _ = pearsonr(rel1, rel2)
                assert corr > 0.8  # Should be highly correlated

    def test_compute_reliability_subset_reps(self):
        """Test reliability computation with subset of repetitions."""
        np.random.seed(42)
        data = np.random.randn(30, 100, 10)

        with patch.object(TVSD_TestDataset, '_get_paths', return_value=[]):
            with patch.object(TVSD_TestDataset, '_get_responses'):
                dataset = TVSD_TestDataset(monkey="monkeyF", region="V1")
                reliability = dataset._compute_reliability(
                    data, n_boot=10, n_reps_subset=15, random_state=42
                )

                assert reliability.shape == (10,)
                assert np.all(reliability >= -1) and np.all(reliability <= 1)

    def test_init_recompute_reliability_params(self):
        """Test initialization parameters for recompute_reliability."""
        with patch.object(TVSD_TestDataset, '_get_paths', return_value=[]):
            with patch.object(TVSD_TestDataset, '_get_responses', return_value=(np.zeros((30, 100, 10)), torch.zeros(10))):
                dataset = TVSD_TestDataset(
                    monkey="monkeyF",
                    region="V1",
                    recompute_reliability=True,
                    n_boot=50,
                    n_reps_subset=20,
                    random_state=123
                )

                assert dataset.recompute_reliability == True
                assert dataset.n_boot == 50
                assert dataset.n_reps_subset == 20
                assert dataset.random_state == 123

    def test_init_default_params(self):
        """Test default initialization parameters."""
        with patch.object(TVSD_TestDataset, '_get_paths', return_value=[]):
            with patch.object(TVSD_TestDataset, '_get_responses', return_value=(np.zeros((30, 100, 10)), torch.zeros(10))):
                dataset = TVSD_TestDataset(monkey="monkeyF", region="V1")

                assert dataset.recompute_reliability == False
                assert dataset.n_boot == 30
                assert dataset.n_reps_subset is None
                assert dataset.random_state is None


class TestDatasetIntegration:
    """Integration tests requiring less mocking."""

    def test_array_channel_mapping_consistency(self):
        """Test that array-to-channel mapping is consistent across regions."""
        with patch.object(TVSD_BaseDataset, '_get_paths', return_value=[]):
            with patch.object(TVSD_BaseDataset, '_get_responses', return_value=(None, None)):
                # Test that total channels add up to 1024 (16 arrays * 64 channels)
                v1_dataset = TVSD_BaseDataset(monkey="monkeyF", region="V1")
                it_dataset = TVSD_BaseDataset(monkey="monkeyF", region="IT")
                v4_dataset = TVSD_BaseDataset(monkey="monkeyF", region="V4")

                v1_channels = len(v1_dataset.array_idxs)
                it_channels = len(it_dataset.array_idxs)
                v4_channels = len(v4_dataset.array_idxs)

                # Check total channels
                assert v1_channels + it_channels + v4_channels == 16 * 64

                # Check no overlap
                v1_set = set(v1_dataset.array_idxs)
                it_set = set(it_dataset.array_idxs)
                v4_set = set(v4_dataset.array_idxs)

                assert len(v1_set & it_set) == 0
                assert len(v1_set & v4_set) == 0
                assert len(it_set & v4_set) == 0
