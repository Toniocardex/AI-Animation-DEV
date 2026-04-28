"""
src/dataset/quick_build.py
Costruisce il dataset finale partendo dal manifest.json
"""
import hashlib
import json
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm

try:
    from src.preprocessing.quick_processor import QuickSpriteProcessor, load_hint_file
except ModuleNotFoundError:
    from src_preprocessing_quick_processor import QuickSpriteProcessor, load_hint_file


def _clip_id(source_path: str, animation: str, flip_aug: bool) -> str:
    """ID stabile per clip (stesso file sorgente + animazione + eventuale mirror)."""
    key = f"{source_path}|{animation}|{int(flip_aug)}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def build_dataset():
    """Costruisce il dataset da zero."""

    # Carica manifest
    manifest_path = "data/raw/licensed/manifest.json"
    if not Path(manifest_path).exists():
        print(f"[ERROR] Manifest non trovato: {manifest_path}")
        print("Esegui prima: python tools/catalog_assets.py")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    processor = QuickSpriteProcessor()
    all_frames = []

    print("=== Processing Asset ===\n")

    # Processa ogni entry del manifest
    for entry in tqdm(manifest, desc="Processing"):
        path = entry["path"]
        animation = entry["animation"]

        # Carica hint file se esiste
        hint = load_hint_file(path)

        try:
            if entry.get("asset_type") == "spritesheet_horizontal":
                # Spritesheet orizzontale
                estimated_frames = 8
                if isinstance(hint, dict):
                    estimated_frames = int(hint.get("frames", estimated_frames))
                frames = processor.process_and_split(
                    img_path=path, animation=animation, estimated_frames=estimated_frames
                )
            else:
                # Frame singolo
                frames = processor.process_single_frame(
                    img_path=path, animation=animation
                )

            src_posix = str(Path(path).as_posix())
            for f in frames:
                f["source_path"] = src_posix
                f["flip_aug"] = False
            all_frames.extend(frames)

        except Exception as e:
            print(f"  [WARN] Errore su {path}: {e}")
            continue

    print(f"\n[OK] Totale frame estratti: {len(all_frames)}\n")

    # Augmentation: flip orizzontale
    print("Augmentation (flip orizzontale)...")
    augmented = list(all_frames)
    for frame in all_frames:
        flipped = {
            **frame,
            "image": frame["image"].transpose(Image.FLIP_LEFT_RIGHT),
            "augmented": True,
            "source_path": frame.get("source_path", ""),
            "flip_aug": True,
        }
        augmented.append(flipped)

    print(f"[OK] Dopo augmentation: {len(augmented)} frame\n")

    # Split
    random.shuffle(augmented)
    n = len(augmented)
    splits = {
        "train": augmented[: int(n * 0.8)],
        "val": augmented[int(n * 0.8) : int(n * 0.9)],
        "test": augmented[int(n * 0.9) :],
    }

    # Salva
    print("=== Saving Dataset ===\n")
    for split_name, frames in splits.items():
        split_dir = Path(f"data/final/{split_name}")
        sprites_dir = split_dir / "sprites"
        poses_dir = split_dir / "poses"
        sprites_dir.mkdir(parents=True, exist_ok=True)
        poses_dir.mkdir(parents=True, exist_ok=True)

        # Pulisci sprite vecchi per mantenere coerenza 1:1 con annotations.json
        for old_png in sprites_dir.glob("*.png"):
            old_png.unlink()
        for old_png in poses_dir.glob("*.png"):
            old_png.unlink()

        annotations = []

        for i, frame in enumerate(frames):
            stem = f"{i:06d}"
            sprite_path = sprites_dir / f"{stem}.png"
            pose_path = poses_dir / f"{stem}.png"
            frame["image"].save(str(sprite_path), "PNG")

            pose_img = QuickSpriteProcessor.pose_from_rgba(frame["image"])
            pose_img.save(str(pose_path), "PNG")

            src = frame.get("source_path", "")
            flip = bool(frame.get("flip_aug", False))
            cid = _clip_id(src, frame["animation"], flip)

            annotations.append(
                {
                    "id": stem,
                    "sprite": f"sprites/{stem}.png",
                    "pose": f"poses/{stem}.png",
                    "clip_id": cid,
                    "animation": frame["animation"],
                    "frame_idx": frame["frame_idx"],
                    "total_frames": int(frame.get("total_frames", 1)),
                    "augmented": frame.get("augmented", False),
                }
            )

        ann_path = split_dir / "annotations.json"
        with open(ann_path, "w") as f:
            json.dump(annotations, f, indent=2)

        print(f"[OK] {split_name:10s}: {len(frames):5d} frame -> {split_dir}")

    print("\n[OK] Dataset costruito con successo!\n")


if __name__ == "__main__":
    build_dataset()
