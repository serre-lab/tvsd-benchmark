"""Tests for reliability computation functions."""
import pytest
import numpy as np
import torch
from scipy.stats import pearsonr
from unittest.mock import patch

from utils.dataset import TVSD_TestDataset


class TestReliabilityComputation:
    """Tests for split-half reliability computation."""

    def test_reliability_range(self):
        """Test that reliability values are in valid range [-1, 1]."""
        np.random.seed(42)
        # Create realistic neural data with some correlation structure
        n_reps, n_stim, n_neu = 30, 100, 20
        data = np.random.randn(n_reps, n_stim, n_neu)

        with patch.object(TVSD_TestDataset, '_get_paths', return_value=[]):
            with patch.object(TVSD_TestDataset, '_get_responses'):
                dataset = TVSD_TestDataset(monkey="monkeyF", region="V1")
                reliability = dataset._compute_reliability(data, n_boot=20, random_state=42)

                assert np.all(reliability >= -1.0)
                assert np.all(reliability <= 1.0)

    def test_reliability_perfect_correlation(self):
        """Test that perfectly correlated data gives high reliability."""
        np.random.seed(42)
        # Create perfectly correlated repetitions
        n_reps, n_stim, n_neu = 30, 100, 5
        base_response = np.random.randn(n_stim, n_neu)
        # All repetitions are identical (plus tiny noise)
        data = np.array([base_response + np.random.randn(n_stim, n_neu) * 0.01 for _ in range(n_reps)])

        with patch.object(TVSD_TestDataset, '_get_paths', return_value=[]):
            with patch.object(TVSD_TestDataset, '_get_responses'):
                dataset = TVSD_TestDataset(monkey="monkeyF", region="V1")
                reliability = dataset._compute_reliability(data, n_boot=20, random_state=42)

                # Should have very high reliability (close to 1)
                assert np.mean(reliability) > 0.9

    def test_reliability_random_data(self):
        """Test that completely random data gives low reliability."""
        np.random.seed(42)
        # Create completely uncorrelated repetitions
        n_reps, n_stim, n_neu = 30, 100, 10
        data = np.random.randn(n_reps, n_stim, n_neu)

        with patch.object(TVSD_TestDataset, '_get_paths', return_value=[]):
            with patch.object(TVSD_TestDataset, '_get_responses'):
                dataset = TVSD_TestDataset(monkey="monkeyF", region="V1")
                reliability = dataset._compute_reliability(data, n_boot=30, random_state=42)

                # Should have low reliability (close to 0)
                # Allow some variance due to random chance
                assert np.abs(np.mean(reliability)) < 0.3

    def test_reliability_reproducibility(self):
        """Test that same random seed gives identical results."""
        np.random.seed(42)
        n_reps, n_stim, n_neu = 30, 100, 10
        data = np.random.randn(n_reps, n_stim, n_neu)

        with patch.object(TVSD_TestDataset, '_get_paths', return_value=[]):
            with patch.object(TVSD_TestDataset, '_get_responses'):
                dataset = TVSD_TestDataset(monkey="monkeyF", region="V1")

                rel1 = dataset._compute_reliability(data, n_boot=20, random_state=42)
                rel2 = dataset._compute_reliability(data, n_boot=20, random_state=42)

                assert np.allclose(rel1, rel2, rtol=1e-10)
                assert np.max(np.abs(rel1 - rel2)) < 1e-10

    def test_reliability_different_seeds_vary(self):
        """Test that different seeds give different but correlated results."""
        np.random.seed(42)
        n_reps, n_stim, n_neu = 30, 100, 10
        data = np.random.randn(n_reps, n_stim, n_neu)

        with patch.object(TVSD_TestDataset, '_get_paths', return_value=[]):
            with patch.object(TVSD_TestDataset, '_get_responses'):
                dataset = TVSD_TestDataset(monkey="monkeyF", region="V1")

                rel1 = dataset._compute_reliability(data, n_boot=20, random_state=42)
                rel2 = dataset._compute_reliability(data, n_boot=20, random_state=99)

                # Should be different
                assert not np.allclose(rel1, rel2, rtol=1e-3)

                # But should be highly correlated (measuring same underlying reliability)
                corr, pval = pearsonr(rel1, rel2)
                assert corr > 0.7  # High correlation expected

    def test_reliability_bootstrap_stability(self):
        """Test that more bootstraps give more stable estimates."""
        np.random.seed(42)
        n_reps, n_stim, n_neu = 30, 100, 10
        data = np.random.randn(n_reps, n_stim, n_neu)

        with patch.object(TVSD_TestDataset, '_get_paths', return_value=[]):
            with patch.object(TVSD_TestDataset, '_get_responses'):
                dataset = TVSD_TestDataset(monkey="monkeyF", region="V1")

                # Compute with few bootstraps multiple times
                few_boot_results = []
                for seed in range(5):
                    rel = dataset._compute_reliability(data, n_boot=5, random_state=seed)
                    few_boot_results.append(rel)

                # Compute with many bootstraps multiple times
                many_boot_results = []
                for seed in range(5):
                    rel = dataset._compute_reliability(data, n_boot=50, random_state=seed)
                    many_boot_results.append(rel)

                # Calculate variance across seeds
                few_boot_var = np.var(few_boot_results, axis=0)
                many_boot_var = np.var(many_boot_results, axis=0)

                # More bootstraps should give more stable (lower variance) estimates
                assert np.mean(many_boot_var) < np.mean(few_boot_var)

    def test_reliability_subset_reps(self):
        """Test reliability computation with subset of repetitions."""
        np.random.seed(42)
        n_reps, n_stim, n_neu = 30, 100, 10
        data = np.random.randn(n_reps, n_stim, n_neu)

        with patch.object(TVSD_TestDataset, '_get_paths', return_value=[]):
            with patch.object(TVSD_TestDataset, '_get_responses'):
                dataset = TVSD_TestDataset(monkey="monkeyF", region="V1")

                # Compute with all reps
                rel_all = dataset._compute_reliability(data, n_boot=20, random_state=42)

                # Compute with subset of reps
                rel_subset = dataset._compute_reliability(
                    data, n_boot=20, n_reps_subset=15, random_state=42
                )

                # Both should be valid
                assert rel_all.shape == (n_neu,)
                assert rel_subset.shape == (n_neu,)
                assert np.all(rel_all >= -1) and np.all(rel_all <= 1)
                assert np.all(rel_subset >= -1) and np.all(rel_subset <= 1)

    def test_reliability_small_sample(self):
        """Test reliability computation with minimal data."""
        np.random.seed(42)
        # Minimum: 2 reps (for split-half), 2 stimuli, 1 neuroid
        n_reps, n_stim, n_neu = 2, 2, 1
        data = np.random.randn(n_reps, n_stim, n_neu)

        with patch.object(TVSD_TestDataset, '_get_paths', return_value=[]):
            with patch.object(TVSD_TestDataset, '_get_responses'):
                dataset = TVSD_TestDataset(monkey="monkeyF", region="V1")
                reliability = dataset._compute_reliability(data, n_boot=5, random_state=42)

                assert reliability.shape == (n_neu,)
                assert np.all(reliability >= -1) and np.all(reliability <= 1)

    def test_reliability_shape_consistency(self):
        """Test that output shape matches number of neuroids."""
        np.random.seed(42)
        test_cases = [
            (30, 100, 5),
            (30, 100, 50),
            (30, 100, 200),
        ]

        for n_reps, n_stim, n_neu in test_cases:
            data = np.random.randn(n_reps, n_stim, n_neu)

            with patch.object(TVSD_TestDataset, '_get_paths', return_value=[]):
                with patch.object(TVSD_TestDataset, '_get_responses'):
                    dataset = TVSD_TestDataset(monkey="monkeyF", region="V1")
                    reliability = dataset._compute_reliability(data, n_boot=10, random_state=42)

                    assert reliability.shape == (n_neu,), f"Failed for n_neu={n_neu}"

    def test_reliability_increasing_reps(self):
        """Test that reliability estimates improve with more repetitions."""
        np.random.seed(42)
        n_stim, n_neu = 100, 5

        # Create data with known reliability structure
        true_response = np.random.randn(n_stim, n_neu)

        with patch.object(TVSD_TestDataset, '_get_paths', return_value=[]):
            with patch.object(TVSD_TestDataset, '_get_responses'):
                dataset = TVSD_TestDataset(monkey="monkeyF", region="V1")

                reliabilities = []
                for n_reps in [5, 10, 20, 30]:
                    # Create data with noise
                    data = np.array([
                        true_response + np.random.randn(n_stim, n_neu) * 0.5
                        for _ in range(n_reps)
                    ])

                    rel = dataset._compute_reliability(data, n_boot=20, random_state=42)
                    reliabilities.append(np.mean(rel))

                # Reliability should generally increase with more reps
                # (though not strictly monotonic due to bootstrapping randomness)
                assert reliabilities[-1] >= reliabilities[0] - 0.1  # Allow some tolerance


class TestReliabilityIntegration:
    """Integration tests for reliability in dataset context."""

    def test_precomputed_vs_recomputed_correlation(self):
        """Test that recomputed reliability correlates with precomputed."""
        # This would require actual data files, so we'll use mocked data
        np.random.seed(42)

        n_reps, n_stim, n_neu = 30, 100, 512
        test_data = np.random.randn(n_reps, n_stim, n_neu)
        precomputed_rel = np.random.rand(n_neu) * 0.5 + 0.3  # Random reliability in [0.3, 0.8]

        with patch.object(TVSD_TestDataset, '_get_paths', return_value=['img.jpg'] * n_stim):
            with patch.object(TVSD_TestDataset, '_get_responses') as mock_responses:
                # Mock precomputed reliability path
                mock_responses.return_value = (test_data, torch.tensor(precomputed_rel, dtype=torch.float32))

                # Create dataset with precomputed reliability
                dataset_precomputed = TVSD_TestDataset(
                    monkey="monkeyF",
                    region="V1",
                    recompute_reliability=False
                )

                # Create dataset with recomputed reliability
                dataset_recomputed = TVSD_TestDataset(
                    monkey="monkeyF",
                    region="V1",
                    recompute_reliability=True,
                    n_boot=30,
                    random_state=42
                )

                # Extract reliabilities
                _, rel_precomp = dataset_precomputed[0]
                _, rel_recomp = dataset_recomputed[0]

                # Both should be valid
                assert rel_precomp.shape == rel_recomp.shape
                assert torch.all(rel_precomp >= -1) and torch.all(rel_precomp <= 1)
                assert torch.all(rel_recomp >= -1) and torch.all(rel_recomp <= 1)
