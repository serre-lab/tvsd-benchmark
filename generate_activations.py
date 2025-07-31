import os
import argparse
import json
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Optional

from utils.dataset import TVSD_Dataset
from utils.hooks import Activations
from utils.load_model import load_model, resolve_transform

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model, model_name, hook_interval = load_model(args.model_config)
    model = model.to(device).eval()
    transform = resolve_transform(args.model_config)
    layers = [name for name, module in model.named_modules() if 'relu' not in name]
    hook_layers = [layers[i] for i in range(0, len(layers), hook_interval)]

    activations = Activations(
        output_dir=args.output_dir,
        model_name=model_name,
        dataset_name='TVSD',
        pca_components=args.pca_components
    )
    activations.register(model, hook_layers)

    tvsd_dataset = TVSD_Dataset(
        root_dir=args.root_dir,
        monkey=args.monkey,
        split="train",
        array=args.array
    )
    things_dataset = tvsd_dataset.get_things(
        things_path=args.things_path,
        split="train",
        transform=transform
    )
    things_loader = DataLoader(things_dataset, batch_size=args.batch_size, shuffle=False)
    shuffled_things_loader = DataLoader(things_dataset, batch_size=args.batch_size, shuffle=True)

    if args.pca_components is not None:
        print("Training IPCA models...")
        max_pca_train_batches = args.max_pca_train_batches
        activations.set_training_mode(True)
        
        for i, batch in tqdm(enumerate(shuffled_things_loader), desc="Training IPCA"):
            if args.max_pca_train_batches and i >= int(args.max_pca_train_batches):
                break
            activations.set_batch(i)
            with torch.no_grad():
                _ = model(batch.to(device))
            activations.finalize_batch_training()
        
        torch.cuda.empty_cache()
        activations.save_ipca_models()
        print("IPCA training completed.")

    print("Generating activations...")
    activations.set_training_mode(False)
    
    for i, batch in tqdm(enumerate(things_loader), desc="Generating activations"):
        if args.max_batches and i >= args.max_batches:
            break
        activations.set_batch(i)
        with torch.no_grad():
            _ = model(batch.to(device))
        activations.finalize_batch_inference()
    
    torch.cuda.empty_cache()
    activations.flush()
    
    print("Activations saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TVSD alignment pipeline.")
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model configuration file.')
    parser.add_argument('--root_dir', type=str, default=f"{os.getcwd()}/data/TVSD", help='Root directory of the TVSD dataset.')
    parser.add_argument('--monkey', type=str, default="monkeyF", help='Monkey name to use in the dataset.')
    parser.add_argument('--array', type=int, default=0, help='Which neural array to use.')
    parser.add_argument('--things_path', type=str, default=f"{os.getcwd()}/data/THINGS/object_images", help='Path to the THINGS dataset.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for the DataLoader.')
    parser.add_argument('--output_dir', type=str, default=f"{os.getcwd()}/outputs", help='Directory to save activations.')
    parser.add_argument('--max_batches', type=int, help='Maximum number of batches to process.')
    parser.add_argument('--max_pca_train_batches', type=str, help='Maximum number of batches to train IPCAs on.')
    parser.add_argument('--pca_components', type=int, help='Number of PCA components to use. Will not apply PCA if None.', default=None)
    parser.add_argument('--skip_interval', type=int, default=1, help='Skip every n-th image in the dataset.')

    args = parser.parse_args()
    main(args)
