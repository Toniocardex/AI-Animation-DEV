"""
src/training/quick_train.py (MODIFICATO)
Training loop con UniversalPixelArtLoss per dataset multi-autore.
Include early stopping, checkpoint periodici, preview val, SSIM, CSV e TensorBoard.
"""
from __future__ import annotations

import csv
from typing import Any
import random
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter

    _TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None  # type: ignore[misc, assignment]
    _TENSORBOARD_AVAILABLE = False

from src.model.simple_model import (
    SimpleUNet,
    INPUT_RGB_POSE_COND,
    compose_residual_rgba,
)
from src.model.conditioning import (
    COND_CHANNELS,
    animation_name_to_index,
    build_cond_planes,
)
from src.training.losses_universal import UniversalPixelArtLoss, LossPresets


class QuickDataset(Dataset):
    """
    Dataset: target = sprite al frame corrente; ref = RGB (tipicamente frame 0 della clip);
    pose = silhouette del frame corrente.

    Con ``cross_pose_prob`` / ``cross_ref_prob`` (solo training) si mescolano esempi per
    avvicinare l'inferenza "motion-from" / retarget: stesso target, ma pose o ref presi
    da un'altra clip con la stessa animazione. La validation usa di default prob=0.
    """

    def __init__(
        self,
        split_dir: str,
        cross_ref_prob: float = 0.0,
        cross_pose_prob: float = 0.0,
    ):
        self.split_dir = Path(split_dir)
        self.cross_ref_prob = float(cross_ref_prob)
        self.cross_pose_prob = float(cross_pose_prob)

        ann_path = self.split_dir / "annotations.json"
        with open(ann_path) as f:
            self.annotations = json.load(f)

        self._ref_index = self._build_ref_indices()
        self._clip_sorted_indices: dict[str, list[int]] = {}
        self._clip_to_ref_ann: dict[str, int] = {}
        self._animation_to_clips: dict[str, list[str]] = {}
        self._build_clip_retrieval()

    def _build_ref_indices(self) -> list[int]:
        """Per ogni clip, indice del frame di riferimento (preferenza: frame_idx == 0)."""
        from collections import defaultdict

        by_clip: dict[str, list[int]] = defaultdict(list)
        for i, ann in enumerate(self.annotations):
            cid = ann.get("clip_id")
            if cid is not None:
                by_clip[cid].append(i)
        first_for_clip: dict[str, int] = {}
        for cid, indices in by_clip.items():
            zeros = [
                i
                for i in indices
                if int(self.annotations[i].get("frame_idx", -1)) == 0
            ]
            first_for_clip[cid] = zeros[0] if zeros else indices[0]
        out: list[int] = []
        for i, ann in enumerate(self.annotations):
            cid = ann.get("clip_id")
            if cid is not None and cid in first_for_clip:
                out.append(first_for_clip[cid])
            else:
                out.append(i)
        return out

    def _build_clip_retrieval(self) -> None:
        from collections import defaultdict

        by_clip: dict[str, list[int]] = defaultdict(list)
        for i, ann in enumerate(self.annotations):
            cid = ann.get("clip_id")
            if cid is not None:
                by_clip[cid].append(i)
        for cid, inds in by_clip.items():
            inds_sorted = sorted(
                inds, key=lambda ii: int(self.annotations[ii].get("frame_idx", 0))
            )
            self._clip_sorted_indices[cid] = inds_sorted
            zeros = [
                ii
                for ii in inds_sorted
                if int(self.annotations[ii].get("frame_idx", -1)) == 0
            ]
            self._clip_to_ref_ann[cid] = zeros[0] if zeros else inds_sorted[0]

        clips_per_anim: dict[str, set[str]] = defaultdict(set)
        for ann in self.annotations:
            cid = ann.get("clip_id")
            if cid:
                clips_per_anim[ann.get("animation", "idle")].add(cid)
        self._animation_to_clips = {a: list(s) for a, s in clips_per_anim.items()}

    def _other_clips_same_anim(self, clip_id: str | None, animation: str) -> list[str]:
        if not clip_id:
            return []
        alts = [c for c in self._animation_to_clips.get(animation, []) if c != clip_id]
        return alts

    def _best_pose_ann_in_clip(self, clip_id: str, want_frame: int) -> int | None:
        sorted_idx = self._clip_sorted_indices.get(clip_id)
        if not sorted_idx:
            return None
        best_j: int | None = None
        best_d = 10**9
        for j in sorted_idx:
            fj = int(self.annotations[j].get("frame_idx", 0))
            d = abs(fj - want_frame)
            if d < best_d:
                best_d = d
                best_j = j
        return best_j

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]

        img_path = self.split_dir / ann["sprite"]
        img = Image.open(img_path).convert("RGBA")

        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)

        rgb = tensor[:3]
        alpha = tensor[3:4]

        animation = ann.get("animation", "idle")
        clip_id = ann.get("clip_id")
        frame_idx_i = int(ann.get("frame_idx", 0))

        pose_path = ann.get("pose")
        if pose_path and (self.split_dir / pose_path).exists():
            po = Image.open(self.split_dir / pose_path).convert("L")
            pose_arr = np.array(po, dtype=np.float32) / 255.0
            pose = torch.from_numpy(pose_arr).unsqueeze(0)
        else:
            pose = alpha.clone()

        ref_i = self._ref_index[idx]
        ref_ann = self.annotations[ref_i]
        ref_path = self.split_dir / ref_ann["sprite"]

        alts = self._other_clips_same_anim(clip_id, animation)
        r_draw = random.random()
        p_pose = self.cross_pose_prob
        p_ref = self.cross_ref_prob
        if alts and p_pose + p_ref > 0:
            if r_draw < p_pose:
                other_c = random.choice(alts)
                pose_j = self._best_pose_ann_in_clip(other_c, frame_idx_i)
                if pose_j is not None:
                    pj = self.annotations[pose_j].get("pose")
                    if pj and (self.split_dir / pj).exists():
                        po2 = Image.open(self.split_dir / pj).convert("L")
                        pose_arr2 = np.array(po2, dtype=np.float32) / 255.0
                        pose = torch.from_numpy(pose_arr2).unsqueeze(0)
            elif r_draw < p_pose + p_ref:
                other_c = random.choice(alts)
                ref_ann_idx = self._clip_to_ref_ann[other_c]
                ref_ann = self.annotations[ref_ann_idx]
                ref_path = self.split_dir / ref_ann["sprite"]

        ref_img = Image.open(ref_path).convert("RGBA")
        ref_arr = np.array(ref_img, dtype=np.float32) / 255.0
        ref_rgb = torch.from_numpy(ref_arr[:, :, :3]).permute(2, 0, 1)

        anim_idx = animation_name_to_index(animation)
        frame_idx = frame_idx_i
        total_frames = int(ann.get("total_frames", 1))

        return {
            "sprite": rgb,
            "alpha": alpha,
            "ref_rgb": ref_rgb,
            "pose": pose,
            "anim_idx": torch.tensor(anim_idx, dtype=torch.long),
            "frame_idx": torch.tensor(frame_idx, dtype=torch.long),
            "total_frames": torch.tensor(total_frames, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Early stopping (monitora val_loss)
# ---------------------------------------------------------------------------


class EarlyStopping:
    """
    Interrompe il training se val_loss non migliora per ``patience`` epoche consecutive.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = max(0, int(patience))
        self.min_delta = float(min_delta)
        self._best = float("inf")
        self.bad_epochs = 0

    @property
    def enabled(self) -> bool:
        return self.patience > 0

    def step(self, val_loss: float) -> bool:
        """
        Aggiorna lo stato. Ritorna True se bisogna fermare il training.
        """
        if not self.enabled:
            return False
        if val_loss < self._best - self.min_delta:
            self._best = val_loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                return True
        return False


# ---------------------------------------------------------------------------
# SSIM (fedeltà strutturale RGB, finestra box come proxy leggero)
# ---------------------------------------------------------------------------


def batch_ssim_rgb(
    pred: torch.Tensor,
    target: torch.Tensor,
    window: int = 11,
) -> float:
    """
    SSIM medio su batch [B, 3, H, W] in [0, 1]. Un canale alla volta, poi media.
    """
    if pred.shape[-1] < window or pred.shape[-2] < window:
        window = min(pred.shape[-1], pred.shape[-2], window)
        if window % 2 == 0:
            window = max(3, window - 1)
    pad = max(window // 2, 0)
    C1 = 0.01**2
    C2 = 0.03**2
    pred = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    ch_scores: list[torch.Tensor] = []
    for c in range(pred.shape[1]):
        x = pred[:, c : c + 1, :, :]
        y = target[:, c : c + 1, :, :]
        mu_x = F.avg_pool2d(x, window, stride=1, padding=pad)
        mu_y = F.avg_pool2d(y, window, stride=1, padding=pad)
        sigma_x = F.avg_pool2d(x * x, window, stride=1, padding=pad) - mu_x * mu_x
        sigma_y = F.avg_pool2d(y * y, window, stride=1, padding=pad) - mu_y * mu_y
        sigma_xy = F.avg_pool2d(x * y, window, stride=1, padding=pad) - mu_x * mu_y
        sigma_x = sigma_x.clamp(min=0.0)
        sigma_y = sigma_y.clamp(min=0.0)
        num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        den = (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2) + 1e-8
        ch_scores.append(num / den)
    ssim_map = torch.stack(ch_scores, dim=1).mean(dim=1)
    return float(ssim_map.mean().item())


# ---------------------------------------------------------------------------
# Preview validation: target | pred affiancati
# ---------------------------------------------------------------------------


def _rgba_tensors_to_pil(rgb: torch.Tensor, alpha: torch.Tensor) -> Image.Image:
    """rgb [3,H,W], alpha [1,H,W] in [0,1]."""
    r = rgb.detach().float().cpu().clamp(0, 1).numpy()
    a = alpha.detach().float().cpu().clamp(0, 1).numpy()
    h, w = r.shape[1], r.shape[2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = (r.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    rgba[:, :, 3] = (a[0] * 255.0).round().astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def save_validation_previews(
    model: nn.Module,
    val_ds: QuickDataset,
    indices: list[int],
    device: torch.device,
    epoch: int,
    out_dir: Path,
) -> None:
    """Salva PNG affiancati (GT | pred) per alcuni indici del validation set."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for k, idx in enumerate(indices):
            sample = val_ds[idx]
            sprite = sample["sprite"].unsqueeze(0).to(device)
            alpha = sample["alpha"].unsqueeze(0).to(device)
            ref_rgb = sample["ref_rgb"].unsqueeze(0).to(device)
            pose = sample["pose"].unsqueeze(0).to(device)
            anim_idx = sample["anim_idx"].unsqueeze(0).to(device)
            frame_idx = sample["frame_idx"].unsqueeze(0).to(device)
            total_frames = sample["total_frames"].unsqueeze(0).to(device)
            _, _, H, W = sprite.shape
            cond = build_cond_planes(anim_idx, frame_idx, total_frames, H, W)
            x_in = torch.cat([ref_rgb, pose, cond], dim=1)
            raw = model(x_in)
            pred = compose_residual_rgba(ref_rgb, raw)

            gt_pil = _rgba_tensors_to_pil(sprite[0], alpha[0])
            pr_pil = _rgba_tensors_to_pil(pred[0, :3], pred[0, 3:4])
            w, h = gt_pil.size
            sheet = Image.new("RGBA", (w * 2, h), (40, 40, 40, 255))
            sheet.paste(gt_pil, (0, 0))
            sheet.paste(pr_pil, (w, 0))
            path = out_dir / f"epoch_{epoch + 1:04d}_sample_{k}_gt_pred.png"
            sheet.save(path, "PNG")


def train_loop(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_ds: QuickDataset,
    criterion: UniversalPixelArtLoss,
    optimizer: optim.Optimizer,
    scheduler_cosine: Any,
    scheduler_plateau: optim.lr_scheduler.ReduceLROnPlateau | None,
    device: torch.device,
    num_epochs: int,
    early_stopping: EarlyStopping,
    preview_every: int,
    preview_indices: list[int],
    checkpoint_backup_every: int,
    csv_path: Path,
    tb_writer: object | None,
    loss_preset_label: str,
) -> int:
    """
    Loop principale: train/val, SSIM, logging, checkpoint, early stopping, preview.

    Ritorna il numero di epoche completate (1-based dell'ultima epoca eseguita).
    """
    best_val_loss = float("inf")
    preview_dir = Path("outputs/previews")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_new = not csv_path.is_file()

    epoch_bar = tqdm(
        range(num_epochs),
        desc="Epochs",
        position=0,
        leave=True,
        dynamic_ncols=True,
    )

    last_epoch = 0
    for epoch in epoch_bar:
        last_epoch = epoch + 1
        # --- Train ---
        model.train()
        train_loss = 0.0
        train_losses_breakdown = {
            "reconstruction": 0.0,
            "color_economy": 0.0,
            "palette_snap": 0.0,
            "cluster": 0.0,
            "antialiasing": 0.0,
            "outline": 0.0,
            "perceptual": 0.0,
        }

        for batch in tqdm(
            train_loader,
            desc=f"Ep {epoch + 1}/{num_epochs} train",
            leave=False,
        ):
            sprite = batch["sprite"].to(device)
            alpha = batch["alpha"].to(device)
            ref_rgb = batch["ref_rgb"].to(device)
            pose = batch["pose"].to(device)
            anim_idx = batch["anim_idx"].to(device)
            frame_idx = batch["frame_idx"].to(device)
            total_frames = batch["total_frames"].to(device)
            _, _, H, W = sprite.shape
            cond = build_cond_planes(anim_idx, frame_idx, total_frames, H, W)
            x_in = torch.cat([ref_rgb, pose, cond], dim=1)

            raw = model(x_in)
            pred = compose_residual_rgba(ref_rgb, raw)
            loss, breakdown = criterion(pred[:, :3], sprite, alpha)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            for k, v in breakdown.items():
                if k in train_losses_breakdown:
                    train_losses_breakdown[k] += v.item()

        train_loss /= len(train_loader)
        for k in train_losses_breakdown:
            train_losses_breakdown[k] /= len(train_loader)

        # --- Validation + SSIM ---
        model.eval()
        val_loss = 0.0
        val_losses_breakdown = {k: 0.0 for k in train_losses_breakdown}
        ssim_batches: list[float] = []

        with torch.no_grad():
            for batch in tqdm(
                val_loader,
                desc=f"Ep {epoch + 1}/{num_epochs} val",
                leave=False,
            ):
                sprite = batch["sprite"].to(device)
                alpha = batch["alpha"].to(device)
                ref_rgb = batch["ref_rgb"].to(device)
                pose = batch["pose"].to(device)
                anim_idx = batch["anim_idx"].to(device)
                frame_idx = batch["frame_idx"].to(device)
                total_frames = batch["total_frames"].to(device)
                _, _, H, W = sprite.shape
                cond = build_cond_planes(anim_idx, frame_idx, total_frames, H, W)
                x_in = torch.cat([ref_rgb, pose, cond], dim=1)

                raw = model(x_in)
                pred = compose_residual_rgba(ref_rgb, raw)
                loss, breakdown = criterion(pred[:, :3], sprite, alpha)

                val_loss += loss.item()
                for k, v in breakdown.items():
                    if k in val_losses_breakdown:
                        val_losses_breakdown[k] += v.item()
                ssim_batches.append(batch_ssim_rgb(pred[:, :3], sprite))

        val_loss /= len(val_loader)
        for k in val_losses_breakdown:
            val_losses_breakdown[k] /= len(val_loader)
        val_ssim = float(np.mean(ssim_batches)) if ssim_batches else 0.0

        lr = optimizer.param_groups[0]["lr"]

        # --- Barra epoche: metriche compatte ---
        postfix = {
            "trn": f"{train_loss:.4f}",
            "val": f"{val_loss:.4f}",
            "ssim": f"{val_ssim:.4f}",
            "lr": f"{lr:.1e}",
        }
        if early_stopping.enabled:
            postfix["es"] = f"{early_stopping.bad_epochs}/{early_stopping.patience}"
        epoch_bar.set_postfix(**postfix)

        # Riga di log leggibile (oltre alla tqdm)
        print(
            f"Epoch {epoch + 1:3d}/{num_epochs} | "
            f"Train {train_loss:.4f} | Val {val_loss:.4f} | SSIM {val_ssim:.4f} | "
            f"lr {lr:.2e} | "
            f"train Pal {train_losses_breakdown['palette_snap']:.4f}"
        )

        if (epoch + 1) % 10 == 0:
            print("  [train breakdown]")
            for k, v in train_losses_breakdown.items():
                print(f"    {k:20s}: {v:.4f}")

        # --- CSV ---
        with open(csv_path, "a", newline="", encoding="utf-8") as cf:
            cw = csv.writer(cf)
            if csv_new:
                cw.writerow(
                    [
                        "epoch",
                        "train_loss",
                        "val_loss",
                        "val_ssim",
                        "lr",
                        "loss_preset",
                    ]
                )
                csv_new = False
            cw.writerow(
                [
                    epoch + 1,
                    f"{train_loss:.6f}",
                    f"{val_loss:.6f}",
                    f"{val_ssim:.6f}",
                    f"{lr:.8f}",
                    loss_preset_label,
                ]
            )

        # --- TensorBoard ---
        if tb_writer is not None:
            tb_writer.add_scalar("loss/train", train_loss, epoch)
            tb_writer.add_scalar("loss/val", val_loss, epoch)
            tb_writer.add_scalar("ssim/val", val_ssim, epoch)
            tb_writer.add_scalar("optim/lr", lr, epoch)

        # --- Solo nuovo minimo val -> model_best ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/model_best.pth")
            print("             [OK] Nuovo best (val_loss) -> checkpoints/model_best.pth")

        # --- Backup periodico ---
        if checkpoint_backup_every > 0 and (epoch + 1) % checkpoint_backup_every == 0:
            bak = Path(f"checkpoints/checkpoint_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), bak)
            print(f"             [OK] Backup: {bak}")

        # --- Preview visivo validation ---
        if preview_every > 0 and (epoch + 1) % preview_every == 0:
            save_validation_previews(
                model, val_ds, preview_indices, device, epoch, preview_dir
            )
            print(f"             [OK] Preview val -> {preview_dir}/epoch_{epoch + 1:04d}_*.png")

        # --- Scheduler: coseno ogni epoca; plateau su val_loss ---
        scheduler_cosine.step()
        if scheduler_plateau is not None:
            scheduler_plateau.step(val_loss)

        # --- Early stopping ---
        if early_stopping.step(val_loss):
            print(
                f"\n[EarlyStopping] Stop: val_loss senza miglioramento "
                f"per {early_stopping.patience} epoche (min_delta={early_stopping.min_delta})."
            )
            break

    return last_epoch


def train(
    loss_preset: str = "multi_author_sharp",
    num_epochs: int = 50,
    cross_ref_prob: float = 0.22,
    cross_pose_prob: float = 0.22,
    resume: str | None = None,
    early_stopping_patience: int = 12,
    early_stopping_min_delta: float = 0.0,
    preview_every: int = 5,
    checkpoint_backup_every: int = 10,
    plateau_patience: int = 5,
    csv_log_path: str = "checkpoints/training_metrics.csv",
    tensorboard_dir: str | None = "runs/pixelart_train",
):
    """
    Setup dataset / loss / optimizer e delega a ``train_loop``.

    Args:
        early_stopping_patience: epoche senza miglioramento val_loss prima dello stop (0 = disattivato).
        preview_every: ogni N epoche salva 4 confronti GT|pred in ``outputs/previews/`` (0 = off).
        checkpoint_backup_every: salva ``checkpoint_epoch_N.pth`` ogni N epoche (0 = off).
        plateau_patience: ReduceLROnPlateau su val_loss (0 = solo CosineAnnealingLR).
        csv_log_path: append metriche epoch per epoch.
        tensorboard_dir: cartella SummaryWriter (None = disattiva TensorBoard).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    Path("checkpoints").mkdir(exist_ok=True)
    Path("outputs/previews").mkdir(parents=True, exist_ok=True)

    model = SimpleUNet().to(device)
    if resume:
        ck = Path(resume)
        if ck.is_file():
            try:
                blob = torch.load(ck, map_location=device, weights_only=True)
            except TypeError:
                blob = torch.load(ck, map_location=device)
            model.load_state_dict(blob, strict=True)
            print(f"[OK] Resume pesi da: {ck.resolve()}\n")
        else:
            print(f"[WARN] Resume non trovato ({resume}), init da zero.\n")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parametri modello: {total_params:,}")
    print(
        f"Input rete: RGB ref (frame 0 clip) + pose (1ch) + {COND_CHANNELS} cond "
        f"(animazione + progresso). Canali totali: {INPUT_RGB_POSE_COND}."
    )
    print(
        "Uscita: RGB = clamp(ref + tanh(delta)), alpha = sigmoid (meno compressione sul colore).\n"
        "Checkpoint precedenti (4 canali tutti sigmoid) non compatibili.\n"
    )
    print(
        f"Mix retarget (solo train): cross_pose_prob={cross_pose_prob:.2f}, "
        f"cross_ref_prob={cross_ref_prob:.2f} | val: task standard (0).\n"
    )

    print("Caricamento dataset...")
    train_ds = QuickDataset(
        "data/final/train",
        cross_ref_prob=cross_ref_prob,
        cross_pose_prob=cross_pose_prob,
    )
    val_ds = QuickDataset("data/final/val")

    if len(train_ds) == 0 or len(val_ds) == 0:
        print("[ERROR] Dataset vuoto: esegui prima catalog/build con asset validi.")
        print("Suggerimento: python run.py catalog && python run.py build")
        return

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=4, num_workers=2)

    n_val = len(val_ds)
    preview_indices = sorted(
        {0, n_val // 4, n_val // 2, max(0, (3 * n_val) // 4), n_val - 1}
    )[:4]

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(
        f"Early stopping: patience={early_stopping_patience} (0=off) | "
        f"Preview ogni {preview_every} ep. | Backup ogni {checkpoint_backup_every} ep.\n"
    )

    preset_map = {
        "multi_author_balanced": LossPresets.multi_author_balanced(),
        "multi_author_strict": LossPresets.multi_author_strict(),
        "multi_author_permissive": LossPresets.multi_author_permissive(),
        "multi_author_sharp": LossPresets.multi_author_sharp(),
        "single_author": LossPresets.single_author(),
    }

    if loss_preset not in preset_map:
        print(f"[WARN] Preset '{loss_preset}' non trovato")
        print(f"   Opzioni: {list(preset_map.keys())}")
        loss_preset = "multi_author_sharp"
        print(f"   Uso default: {loss_preset}\n")

    weights = preset_map[loss_preset]
    criterion = UniversalPixelArtLoss(max_colors=32, weights=weights)

    print(f"Loss Preset: {loss_preset}")
    print(criterion.get_weights_info())

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scheduler_plateau: optim.lr_scheduler.ReduceLROnPlateau | None = None
    if plateau_patience > 0:
        scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=plateau_patience,
            min_lr=1e-7,
        )
        print(
            f"LR: CosineAnnealingLR(T_max={num_epochs}) + "
            f"ReduceLROnPlateau(patience={plateau_patience})\n"
        )
    else:
        print(f"LR: solo CosineAnnealingLR(T_max={num_epochs})\n")

    early = EarlyStopping(patience=early_stopping_patience, min_delta=early_stopping_min_delta)

    tb_writer = None
    if tensorboard_dir and _TENSORBOARD_AVAILABLE:
        tb_path = Path(tensorboard_dir)
        tb_path.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=str(tb_path))
        print(f"TensorBoard: {tb_path.resolve()} (tensorboard --logdir {tb_path})\n")
    elif tensorboard_dir and not _TENSORBOARD_AVAILABLE:
        print("[WARN] TensorBoard non disponibile (installa torch con tensorboard). CSV ok.\n")

    print("=== Training Started ===\n")

    train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        val_ds=val_ds,
        criterion=criterion,
        optimizer=optimizer,
        scheduler_cosine=scheduler_cosine,
        scheduler_plateau=scheduler_plateau,
        device=device,
        num_epochs=num_epochs,
        early_stopping=early,
        preview_every=preview_every,
        preview_indices=preview_indices,
        checkpoint_backup_every=checkpoint_backup_every,
        csv_path=Path(csv_log_path),
        tb_writer=tb_writer,
        loss_preset_label=loss_preset,
    )

    if tb_writer is not None:
        tb_writer.close()

    torch.save(model.state_dict(), "checkpoints/model_final.pth")

    print("\n[OK] Training completato!")
    print("Best model: checkpoints/model_best.pth")
    print("Final weights: checkpoints/model_final.pth")
    print(f"Metriche CSV: {csv_log_path}")
    print(f"Loss preset: {loss_preset}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset",
        default="multi_author_sharp",
        help="Preset loss: balanced, strict, permissive, sharp (meno fangosità), single_author",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Numero di epoch",
    )

    parser.add_argument(
        "--cross-ref-prob",
        type=float,
        default=0.22,
        help="Train: prob. ref da altro clip (stessa animazione). Val sempre 0.",
    )
    parser.add_argument(
        "--cross-pose-prob",
        type=float,
        default=0.22,
        help="Train: prob. pose da altro clip (stessa animazione). Val sempre 0.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="PTH",
        help="Carica pesi da checkpoint (es. checkpoints/model_final.pth) per finetune.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=12,
        help="Stop se val_loss non migliora per N epoche (0 = disattivato).",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Miglioramento minimo val_loss per considerare 'progresso'.",
    )
    parser.add_argument(
        "--preview-every",
        type=int,
        default=5,
        help="Ogni N epoche salva preview GT|pred in outputs/previews/ (0 = off).",
    )
    parser.add_argument(
        "--checkpoint-backup-every",
        type=int,
        default=10,
        help="Salva checkpoints/checkpoint_epoch_N.pth ogni N epoche (0 = off).",
    )
    parser.add_argument(
        "--plateau-patience",
        type=int,
        default=5,
        help="ReduceLROnPlateau su val_loss; 0 = solo cosine schedule.",
    )
    parser.add_argument(
        "--csv-log",
        default="checkpoints/training_metrics.csv",
        help="Append metriche train/val/ssim per epoca.",
    )
    parser.add_argument(
        "--tensorboard-dir",
        default="runs/pixelart_train",
        help="Log TensorBoard (None se vuoto: usa --no-tensorboard da run.py).",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disattiva TensorBoard anche se installato.",
    )

    args = parser.parse_args()

    train(
        loss_preset=args.preset,
        num_epochs=args.epochs,
        cross_ref_prob=args.cross_ref_prob,
        cross_pose_prob=args.cross_pose_prob,
        resume=args.resume,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        preview_every=args.preview_every,
        checkpoint_backup_every=args.checkpoint_backup_every,
        plateau_patience=args.plateau_patience,
        csv_log_path=args.csv_log,
        tensorboard_dir=None if args.no_tensorboard else args.tensorboard_dir,
    )
