"""
Canali di condizionamento spaziali (broadcast) per animazione + progresso frame.
Usati in training e inferenza: stessa geometria [B, C, H, W] concatenata all'RGB.
"""
from __future__ import annotations

import torch

# Ordine stabile (include "unknown" per etichette non previste dal catalogo)
ANIMATION_CLASSES = ("attack", "death", "idle", "jump", "run", "walk", "unknown")
NUM_ANIMATION_CLASSES = len(ANIMATION_CLASSES)
# one-hot animazione + 1 canale progresso temporale [0, 1]
COND_CHANNELS = NUM_ANIMATION_CLASSES + 1


def animation_name_to_index(name: str) -> int:
    key = (name or "").lower().strip()
    try:
        return ANIMATION_CLASSES.index(key)
    except ValueError:
        return ANIMATION_CLASSES.index("unknown")


def _frame_progress(
    frame_idx: torch.Tensor, total_frames: torch.Tensor
) -> torch.Tensor:
    """
    frame_idx, total_frames: [B] long
    Ritorna [B] float in [0, 1]. Un solo frame -> 0.
    """
    tf = total_frames.long().clamp(min=1)
    single = tf <= 1
    denom = (tf - 1).clamp(min=1).float()
    prog = frame_idx.float().clamp(min=0) / denom
    prog = torch.where(single, torch.zeros_like(prog), prog)
    return prog.clamp(0.0, 1.0)


def build_cond_planes(
    anim_idx: torch.Tensor,
    frame_idx: torch.Tensor,
    total_frames: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Costruisce tensor [B, COND_CHANNELS, H, W] sul device di anim_idx.
    """
    B = anim_idx.shape[0]
    device = anim_idx.device
    idx = anim_idx.long().clamp(0, NUM_ANIMATION_CLASSES - 1)
    oh = torch.nn.functional.one_hot(idx, num_classes=NUM_ANIMATION_CLASSES).to(
        dtype=torch.float32, device=device
    )
    oh = oh.view(B, NUM_ANIMATION_CLASSES, 1, 1).expand(
        B, NUM_ANIMATION_CLASSES, height, width
    )
    fp = _frame_progress(frame_idx.long(), total_frames.long())
    fp = fp.view(B, 1, 1, 1).expand(B, 1, height, width)
    return torch.cat([oh, fp], dim=1)
