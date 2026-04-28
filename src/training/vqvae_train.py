"""
Training VQ-VAE (CE palette + loss VQ).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.palette_utils import (
    build_global_palette_from_split,
    load_palette_tensor,
    save_palette_json,
)
from src.dataset.vqvae_dataset import VQVAEPaletteDataset
from src.model.vqvae_pixelart import VQVAE2DPixelArt


def train_vqvae(
    split_dir: str = "data/final/train",
    palette_path: str = "configs/palette.json",
    rebuild_palette: bool = False,
    num_colors: int = 64,
    num_embeddings: int = 512,
    latent_dim: int = 256,
    batch_size: int = 8,
    num_epochs: int = 80,
    lr: float = 3e-4,
    image_size: int = 256,
    device: str | None = None,
):
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    Path("checkpoints").mkdir(exist_ok=True)
    Path("configs").mkdir(exist_ok=True)

    pal_path = Path(palette_path)
    if rebuild_palette or not pal_path.is_file():
        print("Costruzione palette globale (K-means)...")
        palette = build_global_palette_from_split(
            split_dir, num_colors=num_colors, sample_pixels=400_000
        )
        save_palette_json(palette, pal_path)
        print(f"  Salvata: {pal_path} ({len(palette)} colori)\n")
    else:
        palette = load_palette_tensor(pal_path)
        print(f"Uso palette esistente: {pal_path} ({palette.shape[0]} colori)\n")
    num_colors = palette.shape[0]

    ds = VQVAEPaletteDataset(split_dir, pal_path, image_size=image_size)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = None
    val_path = Path("data/final/val") / "annotations.json"
    if val_path.is_file():
        val_ds = VQVAEPaletteDataset("data/final/val", pal_path, image_size=image_size)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = VQVAE2DPixelArt(
        image_size=image_size,
        num_colors=num_colors,
        num_embeddings=num_embeddings,
        latent_dim=latent_dim,
    ).to(dev)

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)

    best_val = float("inf")
    for epoch in range(num_epochs):
        model.train()
        tot = 0.0
        n = 0
        pbar = tqdm(loader, desc=f"VQ-VAE ep {epoch+1}/{num_epochs}", leave=False)
        for batch in pbar:
            rgb = batch["rgb"].to(dev)
            tgt = batch["palette_idx"].to(dev)
            logits, vq_loss, _ = model(rgb)
            ce = F.cross_entropy(logits, tgt)
            loss = ce + vq_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * rgb.size(0)
            n += rgb.size(0)
            pbar.set_postfix(ce=f"{ce.item():.3f}", vq=f"{vq_loss.item():.3f}")

        sched.step()
        train_loss = tot / max(n, 1)

        msg = f"Epoch {epoch+1} | train {train_loss:.4f}"
        if val_loader is not None:
            model.eval()
            vt = 0.0
            vn = 0
            with torch.no_grad():
                for batch in val_loader:
                    rgb = batch["rgb"].to(dev)
                    tgt = batch["palette_idx"].to(dev)
                    logits, vq_loss, _ = model(rgb)
                    ce = F.cross_entropy(logits, tgt)
                    vt += (ce + vq_loss).item() * rgb.size(0)
                    vn += rgb.size(0)
            vloss = vt / max(vn, 1)
            msg += f" | val {vloss:.4f}"
            if vloss < best_val:
                best_val = vloss
                torch.save(model.state_dict(), "checkpoints/vqvae_best.pth")
                print(msg + " [best]")
            else:
                print(msg)
        else:
            print(msg)

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"checkpoints/vqvae_ep{epoch+1}.pth")

    torch.save(model.state_dict(), "checkpoints/vqvae_final.pth")
    torch.save(
        {
            "num_embeddings": num_embeddings,
            "latent_dim": latent_dim,
            "num_colors": num_colors,
            "image_size": image_size,
            "n_down": model.n_down,
        },
        "checkpoints/vqvae_meta.pt",
    )
    print("\n[OK] VQ-VAE: checkpoints/vqvae_final.pth + vqvae_meta.pt")


if __name__ == "__main__":
    train_vqvae()
