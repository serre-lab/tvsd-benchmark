import os
from functools import reduce
import h5py
import torchvision
import torch
from torch.utils.data import Dataset, DataLoader

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

def load_path_order(things_imgs_path, mode):
    with h5py.File(things_imgs_path, "r") as f:
        pathset = f[f"{mode}_imgs"]
        num_samples = pathset["things_path"].shape[0]
        train_paths = [
            (
                f[pathset["things_path"][i][0]][()]
                .tobytes()
                .decode("utf-16")
                .replace("\\", "/")
            )
            for i in range(num_samples)
        ]

class NoDefaultProvided(object):
    pass

def getattrd(obj, name, default=NoDefaultProvided):
    """
    Same as getattr(), but allows dot notation lookup
    Discussed in:
    http://stackoverflow.com/questions/11975781
    """

    try:
        return reduce(getattr, name.split("."), obj)
    except AttributeError as e:
        if default != NoDefaultProvided:
            return default
        raise

def load_responses(tvsd_path, monkey, mode):
    with h5py.File(f'{tvsd_path}/{monkey}/THINGS_normMUA.mat', 'r') as f:
        dataset = f[f"{mode}"][()]
        reliability = torch.mean(
            torch.tensor(f["reliab"][()], dtype=torch.float32), dim=0
        )
    return dataset, reliability
