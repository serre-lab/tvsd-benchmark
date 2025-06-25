"""
GIVEN: model & layer
1. add hooks to model
2. run model on THINGS
3. collect activations from hooks
4. load TVSD activations
5. run correlation analysis
6. save results

WILL ADD utils.py later, for now put everything here
"""

import os
import yaml
import argparse
from tqdm import tqdm
from typing import List, Tuple

import torch
import torchvision.transforms as transforms

from utils import load_model, load_path_order, PathDataset, getattrd, load_responses

def run_benchmark(model: str,
                  model_path: str,
                  model_type: str,
                  input_size: Tuple[int, int],
                  layer: str,
                  tvsd_path: str,
                  things_path: str,
                  monkey: str,
                  output_dir: str):
    """Run the TVSD benchmark for a given model and layer."""
    model = load_model(model_path, model_type)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Loaded model: {model}")

    train_paths = load_path_order(os.path.join(tvsd_path, monkey, '_logs', 'things_imgs.mat'), 'train')
    test_paths = load_path_order(os.path.join(tvsd_path, monkey, '_logs', 'things_imgs.mat'), 'test')
    train_dataset = PathDataset(
        root_dir=os.path.join(things_path, 'object_images'),
        paths=train_paths,
        transform=transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    test_dataset = PathDataset(
        root_dir=os.path.join(things_path, 'object_images'),
        paths=test_paths,
        transform=transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )

    full_activations = {} 

    for mode in ['train', 'test']:
        dataset = train_dataset if mode == 'train' else test_dataset
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

        # Prepare to collect activations
        activations = {layer: []}

        def make_hook(name):
            def hook(module, _input, output):
                # [B, C, H, W]
                activations[name].append(output.detach().cpu())
            return hook

        hook = make_hook(layer)
        module = getattrd(model, layer)
        handle = module.register_forward_hook(hook)

        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader)):
                data = data.to(device)
                _ = model(data)

        handle.remove()
        full_activations[mode] = torch.cat(activations[layer], dim=0)
        full_activations[mode] = full_activations[mode].reshape(full_activations[mode].shape[0], -1)  # Flatten activations
        print(f"Collected activations for {mode} set, shape: {activations[layer].shape}")

    # load TVSD responses
    train_responses, train_reliability = load_responses(tvsd_path, monkey, 'train')
    test_responses, test_reliability = load_responses(tvsd_path, monkey, 'test')
    print(f"Loaded TVSD responses for {monkey}.")

    for arr in range(16):
        print(f"Processing array {arr}...")

        # Select the current array's responses
        train_arr_set = train_responses[:, arr * 64 : (arr + 1) * 64]   # [N, 64]
        test_arr_set = test_responses[:, arr * 64 : (arr + 1) * 64]     # [N', 64]

        train_arr_set = torch.tensor(train_arr_set, dtype=torch.float32).to(device)
        test_arr_set = torch.tensor(test_arr_set, dtype=torch.float32).to(device)

        for idx in range(64):
            # Get the activations for the current neuron
            train_neuron_activations = full_activations['train'][:, idx]
            test_neuron_activations = full_activations['test'][:, idx]
            # Calculate correlations
            X = torch.cat([activations['train'],
                           torch.ones((activations['train'].shape[0], 1), device=device)], 
                           dim=1)
            y = train_neuron_activations
            solution = torch.linalg.lstsq(X, y.unsqueeze(1)).solution.squeeze(1)
            
        
