#!/usr/bin/env python3
"""
compare_checkpoints.py
======================

Script di inferenza / benchmarking per confrontare i checkpoint `.pth` del progetto
**AI Animation Dev** (UNet condizionato su animazione + ref RGB + pose).

NOTA IMPORTANTE
---------------
Questo repository **non** usa Hugging Face Diffusers: il modello e' PyTorch puro
(`SimpleUNet`). Non esiste "prompt testuale" ne' rumore di diffusione; per avere
un confronto stabile tra epoche si usano:

- **Casi di test** = etichetta leggibile + tipo di animazione (`idle`, `run`, …)
  + stesso **sprite di riferimento** (`--ref-image`) e stessi iperparametri.
- **Seed** = fissato su `torch` / `numpy` / `random` per eventuale aleatorieta'
  futura o per script che estendono la pipeline.

La griglia X/Y e' analoga al richiesto "checkpoint x condizione":
- colonne = checkpoint ordinati per epoca (numerico);
- righe = casi di test (ex-"prompt");
- upscale solo **NEAREST** per pixel-perfect.

Uso (dalla root del progetto)::

    python tools/compare_checkpoints.py \\
        --checkpoints-dir checkpoints \\
        --ref-image data/raw/licensed/AllCharacters/Amphibian/Amphibian_attack1.png \\
        --out outputs/checkpoint_grid.png \\
        --cell-size 128 \\
        --upscale 8 \\
        --seed 12345
"""

from __future__ import annotations

import argparse
import contextlib
import io
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Root progetto = parent di tools/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

try:
    from src.inference.quick_generate import AnimationDevGenerator
except ModuleNotFoundError:
    print("[ERROR] Esegui lo script dalla root del progetto (dove c'e' src/).")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Casi di test (equivalente concettuale ai "prompt": animazione + etichetta)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TestCase:
    """Una riga della griglia: nome mostrato e tipo di animazione condizionato."""

    label: str
    animation: str


# Lista predefinita; sovrascrivibile da CLI con --animations
DEFAULT_TEST_CASES: list[TestCase] = [
    TestCase("idle", "idle"),
    TestCase("walk", "walk"),
    TestCase("run", "run"),
    TestCase("attack", "attack"),
]


def set_global_seed(seed: int) -> None:
    """Allinea generatori pseudo-casuali (utile per estensioni future)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discover_checkpoints(checkpoints_dir: Path) -> list[Path]:
    """Tutti i `.pth` nella cartella, ordinati per epoca numerica crescente."""

    def sort_key(p: Path) -> tuple:
        name = p.name
        m = re.match(r"model_ep(\d+)\.pth$", name, re.I)
        if m:
            return (0, int(m.group(1)), name)
        if name.lower() == "model_best.pth":
            return (1, 0, name)
        if name.lower() == "model_final.pth":
            return (1, 1, name)
        return (2, 0, name)

    files = sorted(checkpoints_dir.glob("*.pth"), key=sort_key)
    return files


def nn_upscale(img: Image.Image, factor: int) -> Image.Image:
    """Upscale pixel-perfect (vietato bilinear/bicubic/lanczos)."""
    if factor < 1:
        raise ValueError("upscale factor >= 1")
    w, h = img.size
    return img.resize((w * factor, h * factor), Image.NEAREST)


def try_load_font(size: int) -> ImageFont.ImageFont:
    """Font leggibile; fallback a default PIL."""
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def unload_generator(gen: AnimationDevGenerator | None) -> None:
    """Libera VRAM prima del checkpoint successivo."""
    if gen is None:
        return
    del gen
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def render_cell(
    gen: AnimationDevGenerator,
    case: TestCase,
    ref_image: str | None,
    input_frame: int,
    cell_size: int,
    chain_frames: bool,
    enhance: bool,
    palette_cap: bool,
) -> Image.Image:
    """Una sola anteprima (primo frame) per cella."""
    with contextlib.redirect_stdout(io.StringIO()):
        frames = gen.generate_sheet(
            animation=case.animation,
            frame_count=1,
            output_size=cell_size,
            source_image=ref_image,
            input_frame=input_frame,
            chain_frames=chain_frames,
            motion_from=None,
            enhance_output=enhance,
            palette_cap=palette_cap,
            palette_levels=3,
            palette_json=None,
            alpha_hard=False,
            alpha_threshold=0.5,
            alpha_hard_mode="relative",
        )
    if not frames:
        return Image.new("RGBA", (cell_size, cell_size), (40, 40, 40, 255))
    return frames[0].convert("RGBA")


def build_comparison_grid(
    checkpoints_dir: Path,
    test_cases: list[TestCase],
    ref_image: str | None,
    input_frame: int,
    cell_size: int,
    upscale: int,
    seed: int,
    out_path: Path,
    chain_frames: bool,
    enhance: bool,
    palette_cap: bool,
) -> None:
    ckpts = discover_checkpoints(checkpoints_dir)
    if not ckpts:
        raise FileNotFoundError(
            f"Nessun .pth in {checkpoints_dir}. Addestra il modello o controlla il percorso."
        )

    set_global_seed(seed)

    margin_top = 56
    margin_left = 8
    label_col_w = 160
    label_row_h = 28

    font_title = try_load_font(16)
    font_small = try_load_font(13)

    # Dimensioni celle dopo upscale
    up_w = cell_size * upscale
    up_h = cell_size * upscale

    n_rows = len(test_cases)
    n_cols = len(ckpts)

    grid_w = margin_left + label_col_w + n_cols * (up_w + 4)
    grid_h = margin_top + n_rows * (up_h + label_row_h + 4)

    canvas = Image.new("RGB", (grid_w, grid_h), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)

    draw.text((margin_left, 12), "Checkpoint comparison (NEAREST upscale)", fill=(220, 220, 220), font=font_title)
    draw.text(
        (margin_left, 32),
        f"seed={seed} | cell={cell_size}px | x{upscale} | ref={ref_image or 'none'}",
        fill=(160, 160, 160),
        font=font_small,
    )

    gen: AnimationDevGenerator | None = None

    try:
        for c_idx, ckpt in enumerate(ckpts):
            col_x = margin_left + label_col_w + c_idx * (up_w + 4)
            header = ckpt.stem
            draw.text(
                (col_x, margin_top - label_row_h),
                header[:32],
                fill=(200, 200, 120),
                font=font_small,
            )

            print(f"[{c_idx + 1}/{n_cols}] Carico {ckpt.name} ...")
            unload_generator(gen)
            gen = AnimationDevGenerator(str(ckpt))

            for r_idx, case in enumerate(test_cases):
                row_y = margin_top + r_idx * (up_h + label_row_h + 4)
                if c_idx == 0:
                    draw.text(
                        (margin_left, row_y + up_h // 2 - 8),
                        case.label[:24],
                        fill=(180, 200, 255),
                        font=font_small,
                    )

                cell = render_cell(
                    gen,
                    case,
                    ref_image=ref_image,
                    input_frame=input_frame,
                    cell_size=cell_size,
                    chain_frames=chain_frames,
                    enhance=enhance,
                    palette_cap=palette_cap,
                )
                big = nn_upscale(cell, upscale)
                canvas.paste(big.convert("RGB"), (col_x, row_y))

        unload_generator(gen)
        gen = None

    finally:
        unload_generator(gen)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, "PNG")
    print(f"\n[OK] Griglia salvata: {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Griglia X/Y checkpoint vs condizioni (pipeline AI Animation Dev, non Diffusers)."
    )
    p.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=ROOT / "checkpoints",
        help="Cartella con model_ep*.pth, model_best.pth, ...",
    )
    p.add_argument(
        "--ref-image",
        type=str,
        default=None,
        help="PNG ref fisso (consigliato per confronto sensato).",
    )
    p.add_argument("--input-frame", type=int, default=0, help="Colonna spritesheet ref (se strip).")
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "checkpoint_comparison_grid.png",
        help="PNG output della griglia.",
    )
    p.add_argument("--cell-size", type=int, default=128, choices=[64, 128, 256])
    p.add_argument("--upscale", type=int, default=8, help="Fattore NEAREST (es. 4 o 8).")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument(
        "--animations",
        type=str,
        default=None,
        help='Lista animazioni separata da virgola, es. "idle,walk,run,attack". Le etichette = nomi.',
    )
    p.add_argument("--chain-frames", action="store_true", help="Attiva catena frame (default: off per confronto stabile).")
    p.add_argument("--no-enhance", action="store_true")
    p.add_argument("--no-palette-cap", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.animations:
        parts = [x.strip() for x in args.animations.split(",") if x.strip()]
        test_cases = [TestCase(label=p, animation=p) for p in parts]
    else:
        test_cases = list(DEFAULT_TEST_CASES)

    build_comparison_grid(
        checkpoints_dir=args.checkpoints_dir.resolve(),
        test_cases=test_cases,
        ref_image=args.ref_image,
        input_frame=args.input_frame,
        cell_size=args.cell_size,
        upscale=args.upscale,
        seed=args.seed,
        out_path=args.out.resolve(),
        chain_frames=args.chain_frames,
        enhance=not args.no_enhance,
        palette_cap=not args.no_palette_cap,
    )


if __name__ == "__main__":
    main()
