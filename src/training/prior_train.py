"""
Estrae codici latenti VQ e addestra il Prior Transformer.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.dataset.vqvae_dataset import VQVAEPaletteDataset
from src.model.ar_prior import AutoregressivePrior, build_prior_sequences
from src.model.vqvae_pixelart import VQVAE2DPixelArt


@torch.no_grad()
def extract_latent_codes(
    loader: DataLoader,
    vqvae: VQVAE2DPixelArt,
    device: torch.device,
) -> torch.Tensor:
    """Ritorna [N, L] LongTensor di indici codebook, L = H_lat*W_lat."""
    vqvae.eval()
    chunks: list[torch.Tensor] = []
    for batch in tqdm(loader, desc="Estrazione codici VQ"):
        rgb = batch["rgb"].to(device)
        idx = vqvae.encode_to_indices(rgb)
        chunks.append(idx.cpu().view(idx.size(0), -1))
    return torch.cat(chunks, dim=0)


def train_prior(
    vqvae_ckpt: str = "checkpoints/vqvae_final.pth",
    palette_path: str = "configs/palette.json",
    split_dir: str = "data/final/train",
    num_layers: int = 6,
    n_head: int = 8,
    n_embd: int = 256,
    batch_size: int = 64,
    num_epochs: int = 100,
    lr: float = 3e-4,
    image_size: int = 256,
    device: str | None = None,
):
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    Path("checkpoints").mkdir(exist_ok=True)

    ckpt = Path(vqvae_ckpt)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Addestra prima il VQ-VAE: {vqvae_ckpt}")

    from src.data.palette_utils import load_palette_tensor

    num_colors = load_palette_tensor(palette_path).shape[0]
    ds = VQVAEPaletteDataset(split_dir, palette_path, image_size=image_size)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)

    vm_path = Path("checkpoints/vqvae_meta.pt")
    if vm_path.is_file():
        vm = torch.load(vm_path, map_location="cpu")
        vqvae = VQVAE2DPixelArt(
            image_size=int(vm["image_size"]),
            num_colors=int(vm["num_colors"]),
            num_embeddings=int(vm["num_embeddings"]),
            latent_dim=int(vm["latent_dim"]),
            n_down=int(vm.get("n_down", 5)),
        ).to(dev)
    else:
        vqvae = VQVAE2DPixelArt(
            image_size=image_size,
            num_colors=num_colors,
        ).to(dev)
    vqvae.load_state_dict(torch.load(ckpt, map_location=dev), strict=True)

    h, w = vqvae.latent_hw
    seq_len = h * w
    vocab = vqvae.num_embeddings

    print(f"Latente {h}x{w} = {seq_len} token | vocab codebook = {vocab}")
    codes = extract_latent_codes(loader, vqvae, dev)
    torch.save(codes, "data/processed/vq_codes_train.pt")
    print(f"Codici salvati: data/processed/vq_codes_train.pt shape {tuple(codes.shape)}")

    bos_id = vocab
    inp, tgt = build_prior_sequences(codes, bos_id)
    td = TensorDataset(inp, tgt)
    train_loader = DataLoader(td, batch_size=batch_size, shuffle=True, drop_last=True)

    prior = AutoregressivePrior(
        vocab_size=vocab,
        max_seq_len=seq_len,
        n_layer=num_layers,
        n_head=n_head,
        n_embd=n_embd,
    ).to(dev)

    opt = optim.AdamW(prior.parameters(), lr=lr, weight_decay=0.01)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)

    for epoch in range(num_epochs):
        prior.train()
        tot = 0.0
        n = 0
        for xin, xtgt in tqdm(train_loader, desc=f"Prior {epoch+1}/{num_epochs}", leave=False):
            xin, xtgt = xin.to(dev), xtgt.to(dev)
            opt.zero_grad(set_to_none=True)
            logits = prior(xin)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab), xtgt.reshape(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
            opt.step()
            tot += loss.item() * xin.size(0)
            n += xin.size(0)
        sched.step()
        print(f"Epoch {epoch+1} | CE {tot/max(n,1):.4f}")

        if (epoch + 1) % 20 == 0:
            torch.save(prior.state_dict(), f"checkpoints/prior_ep{epoch+1}.pth")

    torch.save(prior.state_dict(), "checkpoints/prior_final.pth")
    meta = {
        "vocab_size": vocab,
        "seq_len": seq_len,
        "latent_h": h,
        "latent_w": w,
        "n_layer": num_layers,
        "n_head": n_head,
        "n_embd": n_embd,
    }
    torch.save(meta, "checkpoints/prior_meta.pt")
    print("\n[OK] Prior: checkpoints/prior_final.pth + prior_meta.pt")


if __name__ == "__main__":
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("configs").mkdir(parents=True, exist_ok=True)
    train_prior()
