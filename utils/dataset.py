import os
import h5py
import numpy as np
import torchvision
import torch
from torch.utils.data import Dataset
from scipy.stats import pearsonr

from typing import Callable, Optional


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


class TVSD_BaseDataset(Dataset):
    """Base class for TVSD datasets with shared functionality."""

    def __init__(
        self,
        root_dir: str = "/users/jamullik/scratch/tvsd/data/TVSD",
        monkey: str = "monkeyF",
        region: str = "V1",  # options: "V1", "V4", "IT"
        split: str = "train",
    ):
        self.root_dir = os.path.join(root_dir, monkey)
        self.monkey = monkey
        self.region = region
        self.split = split
        self.array_idxs = self._get_array_idxs()
        self.paths = self._get_paths()
        self.responses, self.reliability = self._get_responses()

    def _get_array_idxs(self):
        array_nums = self._get_arrays_from_region()
        array_idxs = set()
        for array_num in array_nums:
            idxs = list(range(array_num * 64, (array_num + 1) * 64))
            array_idxs.update(idxs)
        return sorted(list(array_idxs))

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

    def _get_arrays_from_region(self):
        region_to_array = {
            "monkeyN": {
                "V1": list(range(0, 8)),
                "V4": list(range(8, 12)),
                "IT": list(range(12, 16)),
            },
            "monkeyF": {
                "V1": list(range(0, 8)),
                "IT": list(range(8, 13)),
                "V4": list(range(13, 16)),
            },
        }
        monkey_dict = region_to_array.get(self.monkey, {})
        arrays = monkey_dict.get(self.region, None)
        if arrays is None:
            raise ValueError(f"Invalid region {self.region} for monkey {self.monkey}")
        return arrays

    def _get_paths(self):
        """Get image paths for the split."""
        filepath = os.path.join(self.root_dir, "_logs", "things_imgs.mat")
        key = f"{self.split}_imgs"
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

    def _get_responses(self):
        """Get neural responses. Must be implemented by subclasses due to different data structures."""
        raise NotImplementedError("Subclasses must implement _get_responses()")

    def get_things(
        self,
        things_path: str = "/users/jamullik/scratch/TVSD-real/data/object_images",
        transform: Callable = None,
    ) -> THINGS_Dataset:
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


class TVSD_Dataset(TVSD_BaseDataset):
    """TVSD dataset for training split. Responses shape: [samples, channels]"""

    def __init__(
        self,
        root_dir: str = "/users/jamullik/scratch/tvsd/data/TVSD",
        monkey: str = "monkeyF",
        region: str = "V1",
    ):
        super().__init__(root_dir=root_dir, monkey=monkey, region=region, split="train")

    def _get_responses(self):
        with h5py.File(f"{self.root_dir}/THINGS_normMUA.mat", "r") as f:
            dataset = f["train_MUA"][()]
            print(f.keys())
            print("dataset shape", dataset.shape)
            reliability = torch.mean(
                torch.tensor(f["reliab"][()], dtype=torch.float32), dim=0
            )
        # Train data: 2D indexing [samples, channels]
        return dataset[:, self.array_idxs], reliability[self.array_idxs]


class TVSD_TestDataset(TVSD_BaseDataset):
    """TVSD dataset for test split. Responses shape: [repetitions, samples, channels]"""

    def __init__(
        self,
        root_dir: str = "/users/jamullik/scratch/tvsd/data/TVSD",
        monkey: str = "monkeyF",
        region: str = "V1",
        recompute_reliability: bool = False,
        n_boot: int = 30,
        n_reps_subset: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        """
        Args:
            root_dir: Root directory of the TVSD dataset
            monkey: Monkey name ("monkeyF" or "monkeyN")
            region: Brain region ("V1", "V4", or "IT")
            recompute_reliability: If True, compute reliability from test repetitions instead of using precomputed scores
            n_boot: Number of bootstrap splits for reliability computation (only used if recompute_reliability=True)
            n_reps_subset: If not None, randomly sample this many reps for reliability computation (only used if recompute_reliability=True)
            random_state: Random seed for reliability computation (only used if recompute_reliability=True)
        """
        self.recompute_reliability = recompute_reliability
        self.n_boot = n_boot
        self.n_reps_subset = n_reps_subset
        self.random_state = random_state
        super().__init__(root_dir=root_dir, monkey=monkey, region=region, split="test")

    def _get_responses(self):
        with h5py.File(f"{self.root_dir}/THINGS_normMUA.mat", "r") as f:
            dataset = f["test_MUA_reps"][
                ()
            ]  # dataset is 30 x 100 x 1024 (repetitions x images x neuroids)
            precomputed_reliability = torch.mean(
                torch.tensor(f["reliab"][()], dtype=torch.float32), dim=0
            )

        # Test data: 3D indexing [repetitions, samples, channels]
        responses = dataset[:, :, self.array_idxs]

        if self.recompute_reliability:
            # Compute reliability from test repetitions
            reliability = self._compute_reliability(
                responses,
                n_boot=self.n_boot,
                n_reps_subset=self.n_reps_subset,
                random_state=self.random_state,
            )
            reliability = torch.tensor(reliability, dtype=torch.float32)
        else:
            # Use precomputed reliability scores
            reliability = precomputed_reliability[self.array_idxs]

        return responses, reliability

    def _compute_reliability(
        self,
        data: np.ndarray,
        n_boot: int = 30,
        n_reps_subset: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute split-half reliability via bootstrapped correlations.

        Args:
            data: array of shape (reps, stimuli, neuroids)
            n_boot: number of bootstrap splits
            n_reps_subset: if not None, randomly sample this many reps instead of using all
            random_state: random seed for reproducibility

        Returns:
            reliabilities: array of shape (neuroids,) with reliability scores
        """
        rng = np.random.default_rng(random_state)
        reps, n_stim, n_neu = data.shape
        reliabilities = np.zeros(n_neu)

        for neu in range(n_neu):
            neuroid_data = data[:, :, neu]  # shape (reps, stimuli)
            corrs = []
            for _ in range(n_boot):
                # choose subset of repetitions
                if n_reps_subset is not None:
                    chosen = rng.choice(reps, n_reps_subset, replace=False)
                    reps_data = neuroid_data[chosen]
                else:
                    reps_data = neuroid_data

                # shuffle + split
                reps_idx = rng.permutation(reps_data.shape[0])
                half = reps_data.shape[0] // 2
                group1 = reps_data[reps_idx[:half]].mean(axis=0)
                group2 = reps_data[reps_idx[half:]].mean(axis=0)

                # correlation across stimuli
                r, _ = pearsonr(group1, group2)
                corrs.append(r)
            reliabilities[neu] = np.mean(corrs)
        return reliabilities


if __name__ == "__main__":
    tvsd = TVSD_Dataset()
    things = tvsd.get_things()
    print(f"Number of training images: {len(things)}")
    print(f"Sample image: {things[0]}")
