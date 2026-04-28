"""
Generazione da Prior + VQ-VAE decoder (pixel categorici).
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from src.data.palette_utils import indices_to_rgb, load_palette_tensor
from src.model.ar_prior import AutoregressivePrior
from src.model.vqvae_pixelart import VQVAE2DPixelArt


def generate_vqvae_sample(
    out_path: str = "outputs/vqvae_prior_sample.png",
    vqvae_ckpt: str = "checkpoints/vqvae_final.pth",
    prior_ckpt: str = "checkpoints/prior_final.pth",
    prior_meta: str = "checkpoints/prior_meta.pt",
    palette_path: str = "configs/palette.json",
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 0.92,
    device: str | None = None,
) -> None:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    palette = load_palette_tensor(palette_path)

    meta = torch.load(prior_meta, map_location="cpu")
    h, w = int(meta["latent_h"]), int(meta["latent_w"])
    vocab = int(meta["vocab_size"])

    vm = torch.load("checkpoints/vqvae_meta.pt", map_location="cpu")
    vqvae = VQVAE2DPixelArt(
        image_size=int(vm["image_size"]),
        num_colors=int(vm["num_colors"]),
        num_embeddings=int(vm["num_embeddings"]),
        latent_dim=int(vm["latent_dim"]),
        n_down=int(vm.get("n_down", 5)),
    ).to(dev)
    vqvae.load_state_dict(torch.load(vqvae_ckpt, map_location=dev), strict=True)

    prior = AutoregressivePrior(
        vocab_size=vocab,
        max_seq_len=h * w,
        n_layer=int(meta["n_layer"]),
        n_head=int(meta["n_head"]),
        n_embd=int(meta["n_embd"]),
    ).to(dev)
    prior.load_state_dict(torch.load(prior_ckpt, map_location=dev), strict=True)

    logits, _ = prior.generate(
        vqvae, h, w, dev, temperature=temperature, top_k=top_k, top_p=top_p
    )
    pred = logits.argmax(dim=1)
    rgb = indices_to_rgb(pred[0].cpu(), palette.cpu())
    rgb_u8 = (rgb.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb_u8, mode="RGB").save(out_path)
    print(f"[OK] Salvato: {out_path}")


if __name__ == "__main__":
    generate_vqvae_sample()
