"""
VQ-VAE per pixel art: latenti quantizzati + decoder categorico (logits palette).
Upsampling solo nearest; GroupNorm, no BatchNorm.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """VQ con STE, codebook learnable, commitment loss (beta)."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_beta: float = 0.25,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_beta = commitment_beta
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z_e: [B, D, H, W]
        Returns:
            z_st: [B, D, H, W] straight-through
            vq_loss: scalar
            indices: [B, H, W] long
        """
        b, d, h, w = z_e.shape
        z = z_e.permute(0, 2, 3, 1).contiguous().view(-1, d)
        # L2 distance to codebook
        d2 = (
            z.pow(2).sum(dim=1, keepdim=True)
            - 2 * z @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1, keepdim=True).t()
        )
        idx = d2.argmin(dim=1)
        z_q = self.embedding(idx).view(b, h, w, d).permute(0, 3, 1, 2).contiguous()

        loss_codebook = F.mse_loss(z_q, z_e.detach())
        loss_commit = F.mse_loss(z_e, z_q.detach())
        vq_loss = loss_codebook + self.commitment_beta * loss_commit

        z_st = z_e + (z_q - z_e).detach()
        indices = idx.view(b, h, w)
        return z_st, vq_loss, indices

    def embed_codebook(self, indices: torch.Tensor) -> torch.Tensor:
        """indices [B,H,W] -> z_q [B,D,H,W]"""
        return self.embedding(indices).permute(0, 3, 1, 2).contiguous()


class Encoder(nn.Module):
    """Downsample x5: 256 -> 8 (stride 2 conv)."""

    def __init__(self, in_ch: int, base_ch: int, latent_dim: int, n_down: int = 5):
        super().__init__()
        ch = base_ch
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, ch, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(min(8, ch), ch),
            nn.GELU(),
        ]
        for i in range(1, n_down):
            ch_next = min(base_ch * (2 ** min(i, 3)), 512)
            layers.extend(
                [
                    nn.Conv2d(ch, ch_next, kernel_size=4, stride=2, padding=1),
                    nn.GroupNorm(min(8, ch_next), ch_next),
                    nn.GELU(),
                ]
            )
            ch = ch_next
        layers.extend(
            [
                nn.Conv2d(ch, latent_dim, kernel_size=3, padding=1),
                nn.GroupNorm(min(8, latent_dim), latent_dim),
                nn.GELU(),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """Nearest upsample + conv per ogni scala; logits [B, num_colors, H, W]."""

    def __init__(
        self,
        latent_dim: int,
        base_ch: int,
        num_colors: int,
        out_hw: tuple[int, int],
        n_up: int = 5,
    ):
        super().__init__()
        self.out_hw = out_hw
        self.n_up = n_up
        ch = min(base_ch * 8, 512)
        self.stem = nn.Sequential(
            nn.Conv2d(latent_dim, ch, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, ch), ch),
            nn.GELU(),
        )
        ups: list[nn.Module] = []
        for i in range(n_up):
            ch_next = max(ch // 2, 64)
            ups.append(
                nn.Sequential(
                    nn.Conv2d(ch, ch_next, kernel_size=3, padding=1),
                    nn.GroupNorm(min(8, ch_next), ch_next),
                    nn.GELU(),
                )
            )
            ch = ch_next
        self.up_blocks = nn.ModuleList(ups)
        self.head = nn.Conv2d(ch, num_colors, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.stem(z)
        for blk in self.up_blocks:
            h = F.interpolate(h, scale_factor=2.0, mode="nearest")
            h = blk(h)
        if h.shape[-2:] != self.out_hw:
            h = F.interpolate(h, size=self.out_hw, mode="nearest")
        return self.head(h)


class VQVAE2DPixelArt(nn.Module):
    """
    rgb [B,3,H,W] -> encoder -> VQ -> decoder -> logits [B,K,H,W]
    """

    def __init__(
        self,
        image_size: int = 256,
        num_colors: int = 64,
        num_embeddings: int = 512,
        latent_dim: int = 256,
        base_ch: int = 64,
        n_down: int = 5,
        commitment_beta: float = 0.25,
    ):
        super().__init__()
        self.image_size = image_size
        self.num_colors = num_colors
        self.num_embeddings = num_embeddings
        self.latent_dim = latent_dim
        self.n_down = n_down

        self.encoder = Encoder(3, base_ch, latent_dim, n_down=n_down)
        self.vq = VectorQuantizer(num_embeddings, latent_dim, commitment_beta)
        self.decoder = Decoder(
            latent_dim, base_ch, num_colors, (image_size, image_size), n_up=n_down
        )

    @property
    def latent_hw(self) -> tuple[int, int]:
        h = self.image_size // (2**self.n_down)
        return h, h

    def forward(self, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_e = self.encoder(rgb)
        z_st, vq_loss, indices = self.vq(z_e)
        logits = self.decoder(z_st)
        return logits, vq_loss, indices

    @torch.no_grad()
    def encode_to_indices(self, rgb: torch.Tensor) -> torch.Tensor:
        self.eval()
        z_e = self.encoder(rgb)
        _, _, idx = self.vq(z_e)
        return idx

    def decode_from_latent_indices(self, idx_hw: torch.Tensor) -> torch.Tensor:
        """idx [B,H,W] -> logits [B,num_colors,H_img,W_img]"""
        z = self.vq.embed_codebook(idx_hw)
        return self.decoder(z)

