import os
import h5py
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np

train_things_paths_path = '/users/jamullik/scratch/TVSD-real/monkeyF/_logs/things_imgs.mat'

with h5py.File(train_things_paths_path, "r") as f:
    test_set = f["test_imgs"]
    num_samples = test_set["things_path"].shape[0]
    test_paths = [
        (
            f[test_set["things_path"][i][0]][()]
            .tobytes()
            .decode("utf-16")
            .replace("\\", "/")
        )
        for i in range(num_samples)
    ]

model = models.resnet18(pretrained=True)
model.eval()

layers = [
    'layer1',
    'layer2',
    'layer3',
]

class PathDataset(Dataset):
    def __init__(self, root_dir, paths, transform=None):
        self.root_dir = root_dir
        self.paths = paths
        self.transform = transform
        self.loader = torchvision.datasets.folder.default_loader

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, self.paths[idx])
        img = self.loader(path)
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)
    
train_dataset = PathDataset(
    root_dir='/users/jamullik/scratch/TVSD-real/THINGS/object_images',
    paths=test_paths,  # Replace with actual paths
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
)
test_dataset = PathDataset(
    root_dir='/users/jamullik/scratch/TVSD-real/THINGS/object_images',
    paths=test_paths,
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
)

activations = {layer: [] for layer in layers}

def make_hook(name):
    def hook(module, _input, output):
        # [B, C, H, W]
        activations[name].append(output.detach().cpu())

    return hook

hooks = []
for layer_name in layers:
    module = getattr(model, layer_name)
    hooks.append(module.register_forward_hook(make_hook(layer_name)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
output_dir = '/users/jamullik/scratch/TVSD-real/monkeyF/activations'

model.to(device)

with torch.no_grad():
    for i, data in enumerate(train_loader):
        data = data.to(device)
        _ = model(data)
        print(f"Processed batch {i + 1}/{len(train_loader)}")

# Save activations
for layer in layers:
    layer_activations = torch.cat(activations[layer], dim=0)
    save_path = os.path.join(output_dir, f"{layer}_test_activations.pt")
    torch.save(layer_activations, save_path)
    print(f"Saved activations for {layer} to {save_path}")
