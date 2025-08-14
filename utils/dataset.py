import os
import h5py
import numpy as np
import torchvision
import torch
from torch.utils.data import Dataset

from typing import Callable


class THINGS_Dataset(Dataset):
    def __init__(self, root_dir: str, paths: list, transform: Callable = None):
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


class TVSD_Dataset(Dataset):
    def __init__(
        self,
        root_dir: str = "/users/jamullik/scratch/TVSD-real/data/TVSD",
        monkey: str = "monkeyF",
        split: str = "train",
        array: int = 0,
    ):
        self.root_dir = os.path.join(root_dir, monkey)
        self.monkey = monkey
        self.array = array
        self.split = split
        self.array_idxs = self._get_array_idxs()
        self.region = self._get_region()
        self.paths = self._get_paths(split)
        self.responses, self.reliability = self._get_responses(split)

    def _get_array_idxs(self):
        return list(range(self.array * 64, (self.array + 1) * 64))

    def _get_region(self):
        region_dict = {
            "monkeyN": {
                range(0, 8): "V1",
                range(8, 12): "V4",
                range(12, 16): "IT",
            },
            "monkeyF": {
                range(0, 8): "V1",
                range(8, 13): "IT",
                range(13, 16): "V4",
            },
        }
        monkey_dict = region_dict.get(self.monkey, {})
        for idx_range, region in monkey_dict.items():
            if self.array in idx_range:
                return region
        raise ValueError(f"Invalid array index {self.array} for monkey {self.monkey}")

    def _get_paths(self, split: str):
        if split not in ["train", "test"]:
            raise ValueError("Split must be 'train' or 'test'")
        filepath = os.path.join(self.root_dir, "_logs", "things_imgs.mat")
        key = "train_imgs" if split == "train" else "test_imgs"
        with h5py.File(filepath, "r") as f:
            dataset = f[key]
            num_samples = dataset["things_path"].shape[0]
            paths = [
                (
                    f[dataset["things_path"][i][0]][()]
                    .tobytes()
                    .decode("utf-16")
                    .replace("\\", "/")
                )
                for i in range(num_samples)
            ]
        return paths

    def _get_responses(self, split: str):
        with h5py.File(f"{self.root_dir}/THINGS_normMUA.mat", "r") as f:
            dataset = f[f"{split}_MUA"][()]
            reliability = torch.mean(
                torch.tensor(f["reliab"][()], dtype=torch.float32), dim=0
            )
        return dataset[:, self.array_idxs], reliability[self.array_idxs]

    def get_things(
        self,
        things_path: str = "/users/jamullik/scratch/TVSD-real/data/object_images",
        split: str = "train",
        transform: Callable = None,
    ) -> THINGS_Dataset:
        if split not in ["train", "val"]:
            raise ValueError("Split must be 'train' or 'val'")
        return THINGS_Dataset(things_path, self.paths, transform=transform)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.responses[idx], self.reliability
        elif (
            isinstance(idx, slice)
            or isinstance(idx, list)
            or isinstance(idx, torch.Tensor)
            or isinstance(idx, np.ndarray)
        ):
            return self.responses[idx], self.reliability
        else:
            raise TypeError(f"Unsupported index type: {type(idx)}")

    def __len__(self):
        return len(self.responses)


if __name__ == "__main__":
    tvsd = TVSD_Dataset()
    things = tvsd.get_things(split="train")
    print(f"Number of training images: {len(things)}")
    print(f"Sample image: {things[0]}")
