import os
import argparse
import numpy as np
import torch

from utils.dataset import TVSD_Dataset, TVSD_TestDataset
from utils.load_model import load_model
from utils.brainscore import score_train_test


def _load_activations(path, device):
    """Load saved activations and flatten to (n_stimuli, features)."""
    a = torch.load(path, map_location=device)
    return a.reshape(a.shape[0], -1).detach().cpu().float().numpy()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, model_name, _ = load_model(args.model_config)

    # Train responses (single-trial train_MUA) fit the mapping; the held-out test
    # responses (repetition-averaged test_MUA) are what we score against. Neuron
    # selection uses the single-trial reliability (`reliab`); the noise ceiling is
    # the split-half + Spearman-Brown internal consistency of the 30-rep average.
    train_ds = TVSD_Dataset(root_dir=args.root_dir, monkey=args.monkey, region=args.region)
    test_ds = TVSD_TestDataset(
        root_dir=args.root_dir,
        monkey=args.monkey,
        region=args.region,
        recompute_reliability=True,
        spearman_brown=True,
        n_boot=args.ceiling_n_boot,
        random_state=args.random_state,
    )

    single_trial_reliab = train_ds.reliability.numpy()  # (C,)
    Y_train_all = np.asarray(train_ds.responses)  # (n_train, C)
    Y_test_all = np.asarray(test_ds.responses).mean(axis=0)  # (n_test, C) = test_MUA
    ceiling_all = test_ds.reliability.numpy()  # (C,) SB ceiling

    mask = single_trial_reliab > args.reliability_threshold
    Y_train_all = Y_train_all[:, mask]
    Y_test_all = Y_test_all[:, mask]
    ceiling = ceiling_all[mask]
    print(
        f"{int(mask.sum())} neuroids retained with single-trial reliability "
        f"> {args.reliability_threshold} (median ceiling {np.nanmedian(ceiling):.3f})"
    )

    train_dir = f"{args.output_dir}/activations/TVSD_train/{model_name}"
    test_dir = f"{args.output_dir}/activations/TVSD_test/{model_name}"

    layer_scores = {}
    for layer in sorted(os.listdir(train_dir)):
        print(f"===== EVALUATING LAYER: {layer} =========")
        train_path = f"{train_dir}/{layer}/activations.pt"
        test_path = f"{test_dir}/{layer}/activations.pt"
        if not (os.path.exists(train_path) and os.path.exists(test_path)):
            print(f"Missing train/test activations for layer {layer}. Skipping.")
            continue

        X_train = _load_activations(train_path, device)
        X_test = _load_activations(test_path, device)

        # Align rows with responses in case generation was truncated (e.g. --max_batches).
        n_tr = min(X_train.shape[0], Y_train_all.shape[0])
        n_te = min(X_test.shape[0], Y_test_all.shape[0])
        X_train, Y_train = X_train[:n_tr], Y_train_all[:n_tr]
        X_test, Y_test = X_test[:n_te], Y_test_all[:n_te]
        print(f"train X{X_train.shape} Y{Y_train.shape} | test X{X_test.shape} Y{Y_test.shape}")

        if args.noise_test:
            X_train = np.random.normal(size=X_train.shape)
            X_test = np.random.normal(size=X_test.shape)
        if args.permutation_test:
            np.random.shuffle(Y_test)

        layer_score, layer_std = score_train_test(
            X_train,
            Y_train,
            X_test,
            Y_test,
            reducer=args.reducer,
            correlation_fn=args.correlation_fn,
            pca_components=args.pca_components,
            skip_pca=args.skip_pca,
            standardize=args.standardize,
            ceiling=ceiling,
            ceiling_normalize=args.ceiling_normalize,
            n_boot_ci=args.n_boot_ci,
            random_state=args.random_state,
        )
        layer_scores[layer] = {"score": layer_score, "std": layer_std}
        print(f"Score: {layer_score}, Std: {layer_std}")

    print("Final Layer Scores:")
    for layer, scores in layer_scores.items():
        print(f"{layer}: Score = {scores['score']}, Std = {scores['std']}")
    results_file = f"{args.output_dir}/results/{model_name}/{args.monkey}_arr_{args.region}.csv"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "w") as f:
        f.write("Layer,Score,Std\n")
        for layer, scores in layer_scores.items():
            f.write(f"{layer},{scores['score']},{scores['std']}\n")
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TVSD alignment pipeline.")
    parser.add_argument(
        "--model_config",
        type=str,
        required=True,
        help="Path to the model configuration file.",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=f"{os.getcwd()}/data/TVSD",
        help="Root directory of the TVSD dataset.",
    )
    parser.add_argument(
        "--monkey",
        type=str,
        default="monkeyF",
        help="Monkey name to use in the dataset.",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="IT",
        choices=["V1", "V4", "IT"],
        help="Which brain region to benchmark.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"{os.getcwd()}/outputs",
        help="Directory holding generated activations and results.",
    )
    parser.add_argument(
        "--reliability_threshold",
        type=float,
        default=0.3,
        help="Keep neuroids whose single-trial reliability exceeds this (TVSD-paper selection).",
    )
    parser.add_argument(
        "--reducer",
        type=str,
        default="median",
        choices=["mean", "median"],
        help="Reduction across neuroids.",
    )
    parser.add_argument(
        "--correlation_fn",
        type=str,
        default="pearson",
        choices=["pearson", "spearman"],
        help="Correlation function for predictivity.",
    )
    parser.add_argument(
        "--pca_components",
        type=int,
        default=100,
        help="PCA components to reduce features to (fit on train, applied to test).",
    )
    parser.add_argument(
        "--skip_pca",
        action="store_true",
        help="Skip in-benchmark PCA (e.g. features were already reduced at generation).",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Z-score features/targets (fit on train). Off by default to match Brain-Score.",
    )
    parser.add_argument(
        "--ceiling_normalize",
        action="store_true",
        help="Normalize scores by the median noise ceiling (Brain-Score-style).",
    )
    parser.add_argument(
        "--ceiling_n_boot",
        type=int,
        default=30,
        help="Bootstrap splits for the split-half+SB noise ceiling from test reps.",
    )
    parser.add_argument(
        "--n_boot_ci",
        type=int,
        default=100,
        help="Bootstrap resamples over test stimuli for the score's std.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for the ceiling and bootstrap.",
    )
    parser.add_argument(
        "--noise_test",
        action="store_true",
        help="Replace activations with pure noise (null control).",
    )
    parser.add_argument(
        "--permutation_test",
        action="store_true",
        help="Randomly permute test responses (null control).",
    )

    args = parser.parse_args()
    main(args)
