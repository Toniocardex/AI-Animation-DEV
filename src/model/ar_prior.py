"""
Prior autoregressivo (Transformer decoder-only) su token VQ-VAE.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from src.model.vqvae_pixelart import VQVAE2DPixelArt


def _apply_top_k_top_p(
    logits: torch.Tensor, top_k: int | None, top_p: float | None
) -> torch.Tensor:
    """logits [1, V]"""
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, k, dim=-1)
        logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        mask = cum > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        logits = logits.scatter(-1, sorted_idx, sorted_logits)
    return logits


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, max_seq_len: int, dropout: float = 0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.d_head = n_embd // n_head
        self.dropout_p = dropout

        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)
        causal = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("causal", causal.view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        nh, dh = self.n_head, self.d_head
        q = q.view(b, t, nh, dh).transpose(1, 2)
        k = k.view(b, t, nh, dh).transpose(1, 2)
        v = v.view(b, t, nh, dh).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(dh))
        mask = self.causal[:, :, :t, :t].eq(0)
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.dropout(self.proj(y))


class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, max_seq_len: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        hidden = 4 * n_embd
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class AutoregressivePrior(nn.Module):
    """
    Next-token prediction su token codebook VQ (0 .. vocab_size-1).
    BOS = vocab_size (riga extra in embedding).
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        n_layer: int = 6,
        n_head: int = 8,
        n_embd: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.bos_token_id = vocab_size
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size + 1, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, n_embd))
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(n_embd, n_head, max_seq_len, dropout)
            for _ in range(n_layer)
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx: [B, T] con T <= max_seq_len
        return logits: [B, T, vocab_size]
        """
        b, t = idx.shape
        assert t <= self.max_seq_len
        x = self.token_emb(idx) + self.pos_emb[:, :t, :]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x)

    @torch.no_grad()
    def generate(
        self,
        vqvae: VQVAE2DPixelArt,
        latent_h: int,
        latent_w: int,
        device: torch.device,
        temperature: float = 1.0,
        top_k: int | None = 50,
        top_p: float | None = 0.9,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Campiona L = H*W token, decodifica con VQ-VAE.
        Returns:
            logits_rgb_palette: [1, num_colors, H, W]
            idx_hw: [1, H, W] token VQ
        """
        self.eval()
        vqvae.eval()
        L = latent_h * latent_w
        out: list[int] = []
        ctx = torch.full((1, 1), self.bos_token_id, dtype=torch.long, device=device)

        for _ in range(L):
            if ctx.size(1) > self.max_seq_len:
                raise RuntimeError("Sequenza supera max_seq_len")
            logits = self.forward(ctx)[:, -1, :] / max(temperature, 1e-6)
            logits = _apply_top_k_top_p(logits, top_k, top_p)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            out.append(next_id)
            ctx = torch.cat(
                [ctx, torch.tensor([[next_id]], device=device, dtype=torch.long)],
                dim=1,
            )

        idx_hw = torch.tensor(out, dtype=torch.long, device=device).view(
            1, latent_h, latent_w
        )
        dec_logits = vqvae.decode_from_latent_indices(idx_hw)
        return dec_logits, idx_hw


def build_prior_sequences(codes_flat: torch.Tensor, bos_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    codes_flat: [N, L] indici VQ
    inp: [N, L] = BOS + t0..t_{L-2}  (se L=64, inp has 64 tokens)
    tgt: [N, L] = t0..t_{L-1}
    """
    n, L = codes_flat.shape
    bos = torch.full((n, 1), bos_id, dtype=torch.long)
    full = torch.cat([bos, codes_flat], dim=1)
    return full[:, :-1], full[:, 1:]
