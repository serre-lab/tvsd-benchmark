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
        split="train",
        array=args.array
    )
    things_dataset = tvsd_dataset.get_things(
        things_path=args.things_path,
        split="train",
        transform=transform
    )
    things_loader = DataLoader(things_dataset, batch_size=args.batch_size, shuffle=False)

    for batch in tqdm(things_loader):
        with torch.no_grad():
            _ = model(batch.to(device))
    
    torch.cuda.empty_cache()
    activations.flush(pca_components=args.pca_components)
    
    print("Activations saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TVSD alignment pipeline.")
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model configuration file.')
    parser.add_argument('--root_dir', type=str, default=f"{os.getcwd()}/data/TVSD", help='Root directory of the TVSD dataset.')
    parser.add_argument('--monkey', type=str, default="monkeyF", help='Monkey name to use in the dataset.')
    parser.add_argument('--array', type=int, default=0, help='Which neural array to use.')
    parser.add_argument('--things_path', type=str, default=f"{os.getcwd()}/data/THINGS/object_images", help='Path to the THINGS dataset.')
    parser.add_argument('--hook_interval', type=int, default=5, help='Interval between hooked layers.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for the DataLoader.')
    parser.add_argument('--output_dir', type=str, default=f"{os.getcwd()}/outputs", help='Directory to save activations.')
    parser.add_argument('--max_chunks', type=int, help='Maximum number of chunks to process.')
    parser.add_argument('--pca_components', type=int,help='Number of PCA components to use. Will not apply PCA if None.', default=None)

    args = parser.parse_args()
    main(args)
