from utils.dataset import TVSD_TestDataset
import numpy as np
from scipy.stats import pearsonr

# Test 1: Default behavior (precomputed reliability)
print("=== Test 1: Precomputed Reliability ===")
dataset_precomputed = TVSD_TestDataset(region="IT")
print(f"Responses shape: {dataset_precomputed.responses.shape}")
print(f"Reliability shape: {dataset_precomputed.reliability.shape}")
print(f"Reliability (first 5): {dataset_precomputed.reliability[:5]}")

# Test 2: Recomputed reliability
print("\n=== Test 2: Recomputed Reliability ===")
dataset_recomputed = TVSD_TestDataset(
    region="IT",
    recompute_reliability=True,
    n_boot=30,
    random_state=42
)
print(f"Responses shape: {dataset_recomputed.responses.shape}")
print(f"Reliability shape: {dataset_recomputed.reliability.shape}")
print(f"Reliability (first 5): {dataset_recomputed.reliability[:5]}")

# Test 3: Compare precomputed vs recomputed
print("\n=== Test 3: Correlation Between Precomputed and Recomputed ===")
precomp_rels = dataset_precomputed.reliability.numpy()
recomp_rels = dataset_recomputed.reliability.numpy()
corr, pval = pearsonr(precomp_rels, recomp_rels)
print(f"Correlation: {corr:.4f}, p-value: {pval:.4e}")

# Test 4: Test with different random seed (should give different results)
print("\n=== Test 4: Different Random Seed ===")
dataset_recomputed2 = TVSD_TestDataset(
    region="IT",
    recompute_reliability=True,
    n_boot=30,
    random_state=99
)
recomp_rels2 = dataset_recomputed2.reliability.numpy()
print(f"Same seed correlation: {pearsonr(recomp_rels, recomp_rels)[0]:.4f}")
print(f"Different seed correlation: {pearsonr(recomp_rels, recomp_rels2)[0]:.4f}")

# Test 5: Same seed should give identical results
print("\n=== Test 5: Reproducibility with Same Seed ===")
dataset_recomputed3 = TVSD_TestDataset(
    region="IT",
    recompute_reliability=True,
    n_boot=30,
    random_state=42
)
recomp_rels3 = dataset_recomputed3.reliability.numpy()
print(f"Identical (seed=42 vs seed=42): {np.allclose(recomp_rels, recomp_rels3)}")
print(f"Max difference: {np.max(np.abs(recomp_rels - recomp_rels3))}")
