"""
Palette globale per VQ-VAE categorico (indici colore per pixel).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    from sklearn.cluster import MiniBatchKMeans
except ImportError as e:
    MiniBatchKMeans = None  # type: ignore


def build_global_palette_from_split(
    split_dir: str | Path,
    num_colors: int = 64,
    sample_pixels: int = 500_000,
    annotations_name: str = "annotations.json",
    seed: int = 42,
) -> torch.Tensor:
    """
    Campiona pixel RGB dai PNG dello split e calcola centri K-means (palette fissa).

    Returns:
        palette: FloatTensor [K, 3] in [0, 1]
    """
    if MiniBatchKMeans is None:
        raise ImportError("Serve scikit-learn: pip install scikit-learn")

    split_dir = Path(split_dir)
    ann_path = split_dir / annotations_name
    if not ann_path.is_file():
        raise FileNotFoundError(ann_path)

    with open(ann_path, encoding="utf-8") as f:
        annotations = json.load(f)

    rng = random.Random(seed)
    samples: list[np.ndarray] = []
    target = sample_pixels

    for ann in tqdm(annotations, desc="Campionamento pixel palette"):
        rel = ann.get("sprite")
        if not rel:
            continue
        p = split_dir / rel
        if not p.is_file():
            continue
        img = np.array(Image.open(p).convert("RGBA"), dtype=np.uint8)
        rgb = img[:, :, :3].reshape(-1, 3)
        a = img[:, :, 3].reshape(-1)
        vis = rgb[a > 8]
        if vis.size == 0:
            continue
        n = min(2048, len(vis))
        idx = rng.sample(range(len(vis)), n) if len(vis) > n else range(len(vis))
        samples.append(vis[list(idx)])

        if sum(len(s) for s in samples) >= target:
            break

    if not samples:
        raise RuntimeError("Nessun pixel campionato per la palette.")

    X = np.concatenate(samples, axis=0).astype(np.float32) / 255.0
    if len(X) > sample_pixels:
        X = X[rng.sample(range(len(X)), sample_pixels)]

    k = min(num_colors, len(X))
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        batch_size=4096,
        n_init="auto",
    )
    kmeans.fit(X)
    centers = np.clip(kmeans.cluster_centers_, 0.0, 1.0)
    palette = torch.from_numpy(centers.astype(np.float32))
    return palette


def save_palette_json(palette: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (palette.cpu().numpy().clip(0, 1) * 255.0).round().astype(int).tolist()
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"num_colors": len(arr), "rgb": arr}, f, indent=2)


def load_palette_tensor(path: str | Path) -> torch.Tensor:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rgb = torch.tensor(data["rgb"], dtype=torch.float32) / 255.0
    return rgb


def rgb_to_indices(rgb_chw: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    """
    rgb_chw: [3, H, W] in [0,1]
    palette: [K, 3]
    Returns: LongTensor [H, W]
    """
    c, h, w = rgb_chw.shape
    pal = palette.to(rgb_chw.device, dtype=rgb_chw.dtype)
    flat = rgb_chw.permute(1, 2, 0).reshape(-1, 3)
    d = torch.cdist(flat, pal, p=2.0)
    return d.argmin(dim=1).view(h, w).long()


def indices_to_rgb(idx_hw: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    """idx_hw: [H,W] long -> [3,H,W] float"""
    h, w = idx_hw.shape
    pal = palette.to(idx_hw.device)
    rgb = pal[idx_hw.view(-1)].view(h, w, 3).permute(2, 0, 1)
    return rgb
