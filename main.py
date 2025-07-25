# deprecated
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
    model = model.to(device).eval()
    transform = resolve_transform(args.model_config)
    layers = [name for name, module in model.named_modules() if 'relu' not in name]
    hook_layers = [layers[i] for i in range(0, len(layers), args.hook_interval)]

    activations = Activations(
        output_dir=args.output_dir,
        model_name=model_name,
        dataset_name='TVSD'
    )
    activations.register(model, hook_layers)

    tvsd_dataset = TVSD_Dataset(
        root_dir=args.root_dir,
        monkey=args.monkey,
        split=args.split,
        array=args.array
    )
    things_dataset = tvsd_dataset.get_things(
        things_path=args.things_path,
        split=args.split,
        transform=transform
    )
    things_loader = DataLoader(things_dataset, batch_size=args.batch_size, shuffle=False)

    if not args.skip_generation:
        chunk = 0
        activations.set_chunk(chunk)
        for i, batch in tqdm(enumerate(things_loader)):
            with torch.no_grad():
                _ = model(batch.to(device))
            
            if (i + 1) % args.save_every == 0:
                activations.flush()
                chunk += 1
                activations.set_chunk(chunk)
                if args.max_chunks and chunk >= args.max_chunks:
                    print(f"Reached maximum chunks: {args.max_chunks}. Stopping.")
                    break

            torch.cuda.empty_cache()
        activations.clear()
        print("Activations saved successfully.")

    max_chunks = args.max_chunks if args.max_chunks else chunk + 1

    layer_scores = {}
    for layer in hook_layers:
        print(f"===== EVALUATING LAYER: {layer} =========")
        layer_dir = f"{args.output_dir}/activations/TVSD/{model_name}/{layer}"
        activations = []
        for chunk_idx in range(max_chunks):
            file_path = f"{layer_dir}/chunk_{chunk_idx}.pt"
            try:
                chunk_activations = torch.load(file_path, weights_only=True) # [C, B, ...]
                chunk_activations = chunk_activations.reshape(-1, *chunk_activations.shape[2:])  # [C * B, ...]
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
    metadata_file = f"{args.output_dir}/results/{model_name}/arr_{args.array}_metadata.json"
    os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
    metadata = {
        "model": model_name,
        "region": tvsd_dataset.region,
        "array": args.array,
        "monkey": args.monkey,
        "hook_interval": args.hook_interval,
        "batch_size": args.batch_size,
        "save_every": args.save_every,
        "max_chunks": max_chunks,
        "n_imgs": activations.shape[0] if activations is not None else 0,
        "reliability_threshold": args.reliability_threshold,
        "reducer": args.reducer,
        "correlation_function": args.correlation_fn
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Results saved to {results_file}, metadata saved to {metadata_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TVSD alignment pipeline.")
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model configuration file.')
    parser.add_argument('--root_dir', type=str, default="/users/jamullik/scratch/TVSD-real/data/TVSD", help='Root directory of the TVSD dataset.')
    parser.add_argument('--monkey', type=str, default="monkeyF", help='Monkey name to use in the dataset.')
    parser.add_argument('--split', type=str, default="train", help='Dataset split to use (train//test).')
    parser.add_argument('--array', type=int, default=0, help='Which neural array to use.')
    parser.add_argument('--things_path', type=str, default="/users/jamullik/scratch/TVSD-real/data/object_images", help='Path to the THINGS dataset.')
    parser.add_argument('--hook_interval', type=int, default=5, help='Interval between hooked layers.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for the DataLoader.')
    parser.add_argument('--output_dir', type=str, default="/users/jamullik/scratch/TVSD-real/outputs", help='Directory to save activations.')
    parser.add_argument('--save_every', type=int, default=8, help='Number of batches after which to save activations.')
    parser.add_argument('--max_chunks', type=int, help='Maximum number of chunks to process.')
    parser.add_argument('--reliability_threshold', type=float, default=0.3, help='Reliability threshold for neural responses.')
    parser.add_argument('--reducer', type=str, default='median', choices=['mean', 'median'], help='Reduction method for brain score.')
    parser.add_argument('--correlation_fn', type=str, default='pearson', choices=['pearson', 'spearman'], help='Correlation function to use for brain score computation.')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of splits for KFold cross-validation.')
    parser.add_argument('--pca_components', type=int, default=100, help='Number of PCA components to use.')
    parser.add_argument('--skip_generation', action='store_true', help='Skip generation of activations if they already exist.')
    parser.add_argument('--noise_test', action='store_true', help='Run with pure noise to test. Useful for debugging.')

    args = parser.parse_args()
    main(args)
