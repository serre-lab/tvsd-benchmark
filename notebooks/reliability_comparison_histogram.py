"""
TVSD Neuroid Reliability Comparison

This script compares precomputed and computed neuroid reliabilities for the TVSD dataset.
"""

import sys

sys.path.append("..")

from utils.dataset import TVSD_TestDataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

plt.style.use("seaborn-v0_8-darkgrid")

# Load dataset with precomputed reliability
print("Loading precomputed reliabilities...")
dataset_precomputed = TVSD_TestDataset(region="IT")
precomp_rels = dataset_precomputed.reliability.numpy()

# Load dataset with recomputed reliability
print("Computing reliabilities...")
dataset_recomputed = TVSD_TestDataset(
    region="IT", recompute_reliability=True, n_boot=30, random_state=42
)
recomp_rels = dataset_recomputed.reliability.numpy()

print(f"\nNumber of neuroids: {len(precomp_rels)}")
print(f"Precomputed reliability range: [{precomp_rels.min():.3f}, {precomp_rels.max():.3f}]")
print(f"Recomputed reliability range: [{recomp_rels.min():.3f}, {recomp_rels.max():.3f}]")

# Correlation Analysis
corr, pval = pearsonr(precomp_rels, recomp_rels)
print(f"\nPearson correlation: {corr:.4f}")
print(f"P-value: {pval:.4e}")
print(f"Mean absolute difference: {np.mean(np.abs(precomp_rels - recomp_rels)):.4f}")

# Create histogram comparison figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram 1: Precomputed reliabilities
axes[0, 0].hist(precomp_rels, bins=50, alpha=0.7, color="blue", edgecolor="black")
axes[0, 0].set_xlabel("Reliability", fontsize=12)
axes[0, 0].set_ylabel("Frequency", fontsize=12)
axes[0, 0].set_title("Precomputed Neuroid Reliabilities", fontsize=14, fontweight="bold")
axes[0, 0].axvline(
    precomp_rels.mean(),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Mean: {precomp_rels.mean():.3f}",
)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Histogram 2: Recomputed reliabilities
axes[0, 1].hist(recomp_rels, bins=50, alpha=0.7, color="green", edgecolor="black")
axes[0, 1].set_xlabel("Reliability", fontsize=12)
axes[0, 1].set_ylabel("Frequency", fontsize=12)
axes[0, 1].set_title("Recomputed Neuroid Reliabilities", fontsize=14, fontweight="bold")
axes[0, 1].axvline(
    recomp_rels.mean(),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Mean: {recomp_rels.mean():.3f}",
)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Histogram 3: Overlayed comparison
axes[1, 0].hist(
    precomp_rels, bins=50, alpha=0.5, color="blue", edgecolor="black", label="Precomputed"
)
axes[1, 0].hist(
    recomp_rels, bins=50, alpha=0.5, color="green", edgecolor="black", label="Recomputed"
)
axes[1, 0].set_xlabel("Reliability", fontsize=12)
axes[1, 0].set_ylabel("Frequency", fontsize=12)
axes[1, 0].set_title("Overlayed Comparison", fontsize=14, fontweight="bold")
axes[1, 0].legend(fontsize=11)
axes[1, 0].grid(True, alpha=0.3)

# Scatter plot: Precomputed vs Recomputed
axes[1, 1].scatter(precomp_rels, recomp_rels, alpha=0.5, s=20)
axes[1, 1].plot([0, 1], [0, 1], "r--", linewidth=2, label="y=x")
axes[1, 1].set_xlabel("Precomputed Reliability", fontsize=12)
axes[1, 1].set_ylabel("Recomputed Reliability", fontsize=12)
axes[1, 1].set_title(f"Correlation: r={corr:.4f}", fontsize=14, fontweight="bold")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_aspect("equal", adjustable="box")

plt.tight_layout()
plt.savefig("notebooks/reliability_comparison.png", dpi=300, bbox_inches="tight")
print("\nFigure saved as 'notebooks/reliability_comparison.png'")

# Statistical Summary
print("\n" + "=" * 60)
print("PRECOMPUTED RELIABILITIES")
print("=" * 60)
print(f"Mean:     {precomp_rels.mean():.4f}")
print(f"Median:   {np.median(precomp_rels):.4f}")
print(f"Std Dev:  {precomp_rels.std():.4f}")
print(f"Min:      {precomp_rels.min():.4f}")
print(f"Max:      {precomp_rels.max():.4f}")
print(f"Q1:       {np.percentile(precomp_rels, 25):.4f}")
print(f"Q3:       {np.percentile(precomp_rels, 75):.4f}")

print("\n" + "=" * 60)
print("RECOMPUTED RELIABILITIES")
print("=" * 60)
print(f"Mean:     {recomp_rels.mean():.4f}")
print(f"Median:   {np.median(recomp_rels):.4f}")
print(f"Std Dev:  {recomp_rels.std():.4f}")
print(f"Min:      {recomp_rels.min():.4f}")
print(f"Max:      {recomp_rels.max():.4f}")
print(f"Q1:       {np.percentile(recomp_rels, 25):.4f}")
print(f"Q3:       {np.percentile(recomp_rels, 75):.4f}")

print("\n" + "=" * 60)
print("DIFFERENCE STATISTICS")
print("=" * 60)
diff = precomp_rels - recomp_rels
print(f"Mean difference:     {diff.mean():.4f}")
print(f"Mean abs difference: {np.abs(diff).mean():.4f}")
print(f"Std of difference:   {diff.std():.4f}")
print(f"Max abs difference:  {np.abs(diff).max():.4f}")

# Difference Distribution
fig, ax = plt.subplots(figsize=(10, 6))

diff = precomp_rels - recomp_rels
ax.hist(diff, bins=50, alpha=0.7, color="purple", edgecolor="black")
ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero difference")
ax.axvline(
    diff.mean(), color="orange", linestyle="--", linewidth=2, label=f"Mean: {diff.mean():.4f}"
)
ax.set_xlabel("Difference (Precomputed - Recomputed)", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
ax.set_title("Distribution of Reliability Differences", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("notebooks/reliability_difference_distribution.png", dpi=300, bbox_inches="tight")
print("Figure saved as 'notebooks/reliability_difference_distribution.png'")

plt.show()
