from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class RMDataset(Dataset):
    """Torch Dataset for rMD coordinates and CVs.

    Each item is a tuple: (coords, cvs)
    - coords: float32 tensor of shape (D_coords,)
    - cvs: float32 tensor of shape (D_cvs,)
    """

    def __init__(self, coords: torch.Tensor, cvs: torch.Tensor):
        assert coords.ndim == 2 and cvs.ndim == 2, "coords and cvs must be 2D tensors"
        assert coords.size(0) == cvs.size(0), "coords and cvs must have the same number of samples"
        self.coords = coords
        self.cvs = cvs

    def __len__(self):
        return self.coords.size(0)

    def __getitem__(self, idx):
        return self.coords[idx], self.cvs[idx]


@dataclass
class RMDDataConfig:
    coords_path: str
    cvs_path: str
    batch_size: int = 64
    train_frac: float = 0.8
    seed: int = 42
    max_samples: Optional[int] = None
    shuffle: bool = True


class RMDDataModule:
    """Data module to load .npy arrays, split, and provide DataLoaders.

    Expected file shapes (full-scale):
      - coords: (N, 9696)
      - cvs:    (N, 3)

    For testing and development, smaller shapes are supported.
    """

    def __init__(self, config: RMDDataConfig):
        self.config = config
        self._train_ds: Optional[RMDataset] = None
        self._val_ds: Optional[RMDataset] = None

    @staticmethod
    def _load_npy(path: str) -> np.ndarray:
        arr = np.load(path)
        if not isinstance(arr, np.ndarray):
            raise ValueError(f"File {path} does not contain a numpy array")
        return arr

    def setup(self) -> Tuple[RMDataset, RMDataset]:
        cfg = self.config
        coords_np = self._load_npy(cfg.coords_path)
        cvs_np = self._load_npy(cfg.cvs_path)

        if coords_np.ndim != 2 or cvs_np.ndim != 2:
            raise ValueError("coords and cvs arrays must be 2D")
        if coords_np.shape[0] != cvs_np.shape[0]:
            raise ValueError("coords and cvs must have the same number of rows (samples)")

        n = coords_np.shape[0]
        if cfg.max_samples is not None:
            n = min(n, int(cfg.max_samples))
            coords_np = coords_np[:n]
            cvs_np = cvs_np[:n]

        # Convert to tensors (float32)
        coords = torch.from_numpy(coords_np).float()
        cvs = torch.from_numpy(cvs_np).float()

        # Deterministic shuffle before split
        g = torch.Generator()
        g.manual_seed(cfg.seed)
        perm = torch.randperm(coords.size(0), generator=g)
        coords = coords[perm]
        cvs = cvs[perm]

        # Split
        train_count = int(coords.size(0) * cfg.train_frac)
        self._train_ds = RMDataset(coords[:train_count], cvs[:train_count])
        self._val_ds = RMDataset(coords[train_count:], cvs[train_count:])
        return self._train_ds, self._val_ds

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        if self._train_ds is None or self._val_ds is None:
            self.setup()
        cfg = self.config
        train_loader = DataLoader(self._train_ds, batch_size=cfg.batch_size, shuffle=cfg.shuffle)
        val_loader = DataLoader(self._val_ds, batch_size=cfg.batch_size, shuffle=False)
        return train_loader, val_loader
