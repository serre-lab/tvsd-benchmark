import os
import argparse
import json
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.dataset import TVSD_Dataset
from utils.hooks import Activations
from utils.load_model import load_model, resolve_transform
from utils.brainscore import compute_brain_score

def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model, model_name = load_model(args.model_config)
    layers = [name for name, module in model.named_modules() if 'relu' not in name]
    hook_layers = [layers[i] for i in range(0, len(layers), args.hook_interval)]

    tvsd_dataset = TVSD_Dataset(
        root_dir=args.root_dir,
        monkey=args.monkey,
        split=args.split,
        array=args.array
    )

    layer_scores = {}
    for layer in hook_layers:
        print(f"===== EVALUATING LAYER: {layer} =========")
        layer_dir = f"{args.output_dir}/activations/TVSD/{model_name}/{layer}"
        activations = []
        for chunk_idx in len(os.listdir(layer_dir)):
            file_path = f"{layer_dir}/chunk_{chunk_idx}.pt"
            try:
                chunk_activations = torch.load(file_path, weights_only=True) # [Chunk, Batch, ...]
                chunk_activations = chunk_activations.reshape(-1, *chunk_activations.shape[2:])  # [Chunk * Batch, ...]
                activations.append(chunk_activations)
            except FileNotFoundError:
                print(f"File {file_path} not found. Stopping evaluation for this layer.")
                break
        activations = torch.cat(activations, dim=0) if activations else None
        if activations is not None:
            print(f"Layer: {layer}, Activations shape: {activations.shape}")
        else:
            print(f"No activations found for layer: {layer}")

        activations = activations.reshape(activations.shape[0], -1).detach().cpu().numpy()  # [B, H * W * C]
        neural_responses, reliability = tvsd_dataset[:activations.shape[0]]
        reliability_mask = reliability > args.reliability_threshold
        neural_responses = neural_responses[:, reliability_mask] # [B, C']

        if args.noise_test:
            activations = np.random.normal(size=activations.shape)  # Use random noise for null results
        if args.permutation_test:
            np.random.shuffle(neural_responses)

        print(f"{neural_responses.shape[1]} neural responses retained with reliability > {args.reliability_threshold}")

        layer_score, layer_std = compute_brain_score(X=activations, Y=neural_responses, n_splits=args.n_splits, reducer=args.reducer, correlation_fn=args.correlation_fn, pca_components=args.pca_components)
        layer_scores[layer] = {
            'score': layer_score,
            'std': layer_std
        }
        print(f"Score: {layer_score}, Std: {layer_std}")

    print("Final Layer Scores:")
    for layer, scores in layer_scores.items():
        print(f"{layer}: Score = {scores['score']}, Std = {scores['std']}")
    results_file = f"{args.output_dir}/results/{model_name}/arr_{args.array}.csv"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        f.write("Layer,Score,Std\n")
        for layer, scores in layer_scores.items():
            f.write(f"{layer},{scores['score']},{scores['std']}\n")
    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TVSD alignment pipeline.")
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model configuration file.')
    parser.add_argument('--root_dir', type=str, default="/users/jamullik/scratch/TVSD-real/data/TVSD", help='Root directory of the TVSD dataset.')
    parser.add_argument('--monkey', type=str, default="monkeyF", help='Monkey name to use in the dataset.')
    parser.add_argument('--split', type=str, default="train", help='Dataset split to use (train//test).')
    parser.add_argument('--array', type=int, default=0, help='Which neural array to use.')
    parser.add_argument('--hook_interval', type=int, default=5, help='Interval between hooked layers.')
    parser.add_argument('--output_dir', type=str, default="/users/jamullik/scratch/TVSD-real/outputs", help='Directory to save activations.')
    parser.add_argument('--max_chunks', type=int, help='Maximum number of chunks to process.')
    parser.add_argument('--reliability_threshold', type=float, default=0.3, help='Reliability threshold for neural responses.')
    parser.add_argument('--reducer', type=str, default='median', choices=['mean', 'median'], help='Reduction method for brain score.')
    parser.add_argument('--correlation_fn', type=str, default='pearson', choices=['pearson', 'spearman'], help='Correlation function to use for brain score computation.')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of splits for KFold cross-validation.')
    parser.add_argument('--pca_components', type=int, default=100, help='Number of PCA components to use.')
    parser.add_argument('--noise_test', action='store_true', help='Run with pure noise to test. Useful for debugging.')
    parser.add_argument('--permutation_test', action='store_true', help='Randomly permute neural responses. Useful for debugging.')

    args = parser.parse_args()
    main(args)
