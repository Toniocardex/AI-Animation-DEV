"""
Dataset per VQ-VAE: RGB + indici palette per Cross-Entropy.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.palette_utils import load_palette_tensor, rgb_to_indices


class VQVAEPaletteDataset(Dataset):
    """Carica sprite RGB 256×256 e mappa ogni pixel all'indice della palette globale."""

    def __init__(
        self,
        split_dir: str,
        palette_path: str | Path,
        image_size: int = 256,
    ):
        self.split_dir = Path(split_dir)
        self.palette = load_palette_tensor(palette_path)
        self.image_size = image_size

        ann_path = self.split_dir / "annotations.json"
        with open(ann_path, encoding="utf-8") as f:
            self.annotations = json.load(f)

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ann = self.annotations[idx]
        img_path = self.split_dir / ann["sprite"]
        img = Image.open(img_path).convert("RGBA")
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.NEAREST)

        a = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(a).permute(2, 0, 1)
        rgb = tensor[:3]
        alpha = tensor[3:4]

        pal = self.palette
        target_idx = rgb_to_indices(rgb.cpu(), pal)

        return {
            "rgb": rgb,
            "alpha": alpha,
            "palette_idx": target_idx,
        }
