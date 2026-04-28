"""
src/inference/quick_generate.py
Generatore sprite con supporto multi-risoluzione
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

from src.model.simple_model import (
    SimpleUNet,
    INPUT_RGB_POSE_COND,
    compose_residual_rgba,
)
from src.model.conditioning import animation_name_to_index, build_cond_planes

try:
    from src.preprocessing.quick_processor import QuickSpriteProcessor
except ModuleNotFoundError:
    from src_preprocessing_quick_processor import QuickSpriteProcessor


def infer_animation_from_stem(stem: str) -> str:
    """Stessa logica di tools/catalog_assets.py (coerenza train/infer)."""
    name_lower = stem.lower()
    if "walk" in name_lower:
        return "walk"
    if "run" in name_lower:
        return "run"
    if "attack" in name_lower or "slash" in name_lower:
        return "attack"
    if "jump" in name_lower:
        return "jump"
    if "death" in name_lower or "die" in name_lower:
        return "death"
    return "idle"


def pick_random_source_entry(
    licensed_root: str = "data/raw/licensed",
) -> tuple[str | None, str | None]:
    """
    Sceglie un asset casuale e l'animazione coerente (manifest o nome file).

    Ritorna (path, animation). Se usi ``idle`` su un ref ``death``, il modello
    tende a collassare (uscita grigia / alpha bassa): con --random-input
    sovrascriviamo --animation con questo valore.
    """
    root = Path(licensed_root)
    pairs: list[tuple[str, str]] = []
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, OSError):
            manifest = []
        for entry in manifest:
            p = entry.get("path")
            if not p:
                continue
            pp = Path(p)
            if not pp.is_file() or pp.suffix.lower() not in (".png", ".gif"):
                continue
            anim = entry.get("animation") or infer_animation_from_stem(pp.stem)
            pairs.append((str(pp), str(anim)))
    if not pairs and root.is_dir():
        for pp in root.rglob("*.png"):
            if pp.is_file():
                pairs.append((str(pp), infer_animation_from_stem(pp.stem)))
    if not pairs:
        return None, None
    path, anim = random.choice(pairs)
    return path, anim


class AnimationDevGenerator:
    """AI Animation Dev - Generatore sprite a multiple risoluzioni."""

    SUPPORTED_SIZES = [64, 128, 256]

    def __init__(self, checkpoint_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = SimpleUNet().to(self.device)

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint non trovato: {checkpoint_path}")

        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )
        self.model.eval()

        print(f"[OK] Modello caricato: {checkpoint_path}")
        print(f"  Device: {self.device}\n")

    def generate_sheet(
        self,
        animation: str,
        frame_count: int = 8,
        output_size: int = 256,
        source_image: str | None = None,
        input_frame: int = 0,
        chain_frames: bool = True,
        motion_from: str | None = None,
        enhance_output: bool = True,
        palette_cap: bool = True,
        palette_levels: int = 3,
    ) -> list[Image.Image]:
        """
        Genera un'animazione completa.

        Il modello è addestrato con RGB ref (aspetto) + pose (silhouette tempo t).
        Con ``motion_from`` (spritesheet orizzontale pro) la sequenza di pose segue
        i professionisti; ``--input`` è l'aspetto del mostro (ref).

        chain_frames: se True e senza motion_from, il RGB ref del frame successivo
        è l'RGB dell'output precedente (workaround legacy).
        """
        if output_size not in self.SUPPORTED_SIZES:
            raise ValueError(
                f"output_size deve essere uno di {self.SUPPORTED_SIZES}"
            )

        print(f"Generazione: {animation} ({frame_count} frame @ {output_size}x{output_size}px)")
        if enhance_output:
            print("  Post-enhance: ON (nitidezza + contrasto leggeri sul RGB)")
        pl = int(max(2, min(palette_levels, 32)))
        if palette_cap:
            grid_n = pl**3
            print(
                f"  Palette cap: ON (RGB L={pl} livelli/canale -> griglia fino a {grid_n} colori)"
            )
        print()
        motion_poses = None
        if motion_from:
            motion_poses = self._poses_from_motion_strip(motion_from, frame_count)
            print(
                f"  Movimento da: {motion_from} ({len(motion_poses)} pose / ciclo colonne)\n"
            )
            chain_frames = False
        elif chain_frames:
            print(
                "  Catena frame: ON (ref frame N+1 = RGB composito su grigio 128 dall'output "
                "precedente, cosi' alpha=0 non diventa nero)\n"
            )
        else:
            print(
                "  Catena frame: OFF (stesso ref; pose = silhouette ref → spesso statico)\n"
            )

        frames = []
        ref_rgb, pose_static = self._load_ref_and_static_pose(
            source_image, input_frame=input_frame
        )
        prev_ref_rgb: np.ndarray | None = None

        for i in range(frame_count):
            if motion_poses is not None:
                pose_hw = motion_poses[i]
            elif pose_static is not None:
                pose_hw = pose_static
            else:
                pose_hw = np.zeros((256, 256), dtype=np.uint8)

            if chain_frames and i > 0 and prev_ref_rgb is not None:
                ref_in = prev_ref_rgb
            else:
                ref_in = ref_rgb

            frame_256 = self._generate_single_frame(
                animation,
                i,
                frame_count,
                ref_rgb=ref_in,
                pose_hw=pose_hw,
            )

            if enhance_output:
                frame_256 = self._post_enhance_rgba(frame_256)
            if palette_cap:
                frame_256 = self._apply_palette_cap_rgba(frame_256, levels=pl)

            prev_ref_rgb = self._chain_ref_rgb_from_rgba(frame_256)

            if output_size < 256:
                frame = self._downscale(frame_256, output_size)
            else:
                frame = frame_256

            frames.append(frame)

            if (i + 1) % max(1, frame_count // 4) == 0 or i == frame_count - 1:
                print(f"  {i+1}/{frame_count} frame [OK]")

        print()
        return frames

    @staticmethod
    def _chain_ref_rgb_from_rgba(img: Image.Image) -> np.ndarray:
        """
        RGB da usare come ref al frame successivo (catena).

        Se si copiano solo i canali RGB del PNG, dove alpha=0 spesso c'e' nero:
        il ref diventa quasi tutto nero e il modello collassa in strisce/artefatti.
        Si compone su grigio 128 come il fallback senza --input (coerente col training).
        """
        arr = np.asarray(img.convert("RGBA"), dtype=np.float32)
        rgb = arr[:, :, :3] / 255.0
        a = arr[:, :, 3:4] / 255.0
        bg = 128.0 / 255.0
        out = rgb * a + bg * (1.0 - a)
        return np.clip(out * 255.0, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def _apply_palette_cap_rgba(img: Image.Image, levels: int = 3) -> Image.Image:
        """
        Quantizza il RGB su L livelli per canale (L=3 → 27 colori, L=4 → 64).
        L'alpha resta invariato; nessun filtro bilineare (solo arrotondamento su scala discreta).
        """
        arr = np.asarray(img.convert("RGBA"), dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 4:
            return img
        L = int(max(2, min(int(levels), 32)))
        rgb = arr[:, :, :3].astype(np.float32) / 255.0
        denom = float(max(L - 1, 1))
        q = np.round(np.clip(rgb, 0.0, 1.0) * (L - 1)) / denom
        out_rgb = (q * 255.0).astype(np.uint8)
        out = np.dstack([out_rgb, arr[:, :, 3]])
        return Image.fromarray(out, mode="RGBA")

    def _post_enhance_rgba(self, img: Image.Image) -> Image.Image:
        """Migliora percezione nitidezza senza cambiare architettura (solo inferenza)."""
        arr = np.asarray(img.convert("RGBA"), dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 4:
            return img
        rgb = arr[:, :, :3]
        alpha = arr[:, :, 3]
        pil = Image.fromarray(rgb, mode="RGB")
        pil = pil.filter(
            ImageFilter.UnsharpMask(radius=0.8, percent=125, threshold=2)
        )
        pil = ImageEnhance.Contrast(pil).enhance(1.1)
        rgb2 = np.clip(np.asarray(pil, dtype=np.float32), 0.0, 255.0).astype(np.uint8)
        out = np.dstack([rgb2, alpha])
        return Image.fromarray(out, mode="RGBA")

    def _poses_from_motion_strip(self, motion_path: str, frame_count: int) -> list[np.ndarray]:
        """Estrae una silhouette 256×256 per ogni istante (cicla sulle colonne dello strip)."""
        path = Path(motion_path)
        if not path.exists():
            raise FileNotFoundError(f"motion-from non trovato: {motion_path}")
        processor = QuickSpriteProcessor()
        _, n_strip = processor.extract_frame_for_inference(str(path), 0)
        n_strip = max(1, n_strip)
        out: list[np.ndarray] = []
        for i in range(frame_count):
            col = i % n_strip
            rgba, _ = processor.extract_frame_for_inference(str(path), col)
            out.append(
                np.array(QuickSpriteProcessor.pose_from_rgba(rgba), dtype=np.uint8)
            )
        return out

    def _generate_single_frame(
        self,
        animation: str,
        frame_idx: int,
        total_frames: int,
        ref_rgb: np.ndarray | None = None,
        pose_hw: np.ndarray | None = None,
    ) -> Image.Image:
        """Un frame: concat ref RGB + pose L + cond → RGBA."""
        if ref_rgb is None:
            ref_img = np.ones((256, 256, 3), dtype=np.uint8) * 128
        else:
            ref_img = np.clip(ref_rgb, 0, 255).astype(np.uint8).copy()

        if pose_hw is None:
            pose_img = np.zeros((256, 256), dtype=np.uint8)
        else:
            pose_img = np.clip(pose_hw, 0, 255).astype(np.uint8).copy()

        ref_t = (
            torch.from_numpy(ref_img.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )
        pose_t = (
            torch.from_numpy(pose_img.astype(np.float32) / 255.0)
            .view(1, 1, 256, 256)
            .to(self.device)
        )
        anim_t = torch.tensor(
            [animation_name_to_index(animation)], dtype=torch.long, device=self.device
        )
        frame_t = torch.tensor([frame_idx], dtype=torch.long, device=self.device)
        tot_t = torch.tensor(
            [max(1, int(total_frames))], dtype=torch.long, device=self.device
        )
        _, _, h, w = ref_t.shape
        cond = build_cond_planes(anim_t, frame_t, tot_t, h, w)
        input_tensor = torch.cat([ref_t, pose_t, cond], dim=1)
        assert input_tensor.shape[1] == INPUT_RGB_POSE_COND, (
            f"Attesi {INPUT_RGB_POSE_COND} canali in input, got {input_tensor.shape[1]}"
        )

        with torch.no_grad():
            raw = self.model(input_tensor)
            composed = compose_residual_rgba(ref_t, raw)

        output_np = composed[0].cpu().numpy().transpose(1, 2, 0)
        output_np = np.clip(output_np * 255.0, 0.0, 255.0).astype(np.uint8)

        return Image.fromarray(output_np, "RGBA")

    def _load_ref_and_static_pose(
        self, source_image: str | None, input_frame: int = 0
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Ref RGB (aspetto mostro) + silhouette statica dalla stessa crop.
        """
        if not source_image:
            return None, None

        src_path = Path(source_image)
        if not src_path.exists():
            print(f"[WARN] Input sorgente non trovato: {source_image}")
            print("       Uso fallback su canvas grigio.\n")
            return None, None

        processor = QuickSpriteProcessor()
        frame_rgba, n_strip = processor.extract_frame_for_inference(
            str(src_path), frame_index=input_frame
        )
        arr = np.array(frame_rgba, dtype=np.uint8)
        rgb = arr[:, :, :3]
        pose = np.array(
            QuickSpriteProcessor.pose_from_rgba(frame_rgba), dtype=np.uint8
        )
        if n_strip > 1:
            print(
                f"[OK] Ref mostro: {source_image} (spritesheet {n_strip} colonne, indice {input_frame})\n"
            )
        else:
            print(f"[OK] Ref mostro caricato: {source_image}\n")
        return rgb, pose

    def _downscale(self, img: Image.Image, target_size: int) -> Image.Image:
        """
        Downscala preservando la struttura pixel art.
        Usa NEAREST per mantenere bordi netti.
        """
        if img.width <= target_size:
            return img

        # Downscala con NEAREST
        downscaled = img.resize((target_size, target_size), Image.NEAREST)

        return downscaled

    def export_sheet(
        self,
        frames: list[Image.Image],
        output_path: str,
        output_size: int,
        animation_label: str | None = None,
        chain_frames: bool = True,
        motion_from: str | None = None,
        enhance_output: bool = True,
        palette_cap: bool = True,
        palette_levels: int = 3,
    ):
        """
        Esporta i frame come spritesheet.

        Args:
            frames: lista di frame PIL Image
            output_path: dove salvare il PNG
            output_size: dimensione cella (64, 128, o 256)
        """
        n = len(frames)
        sheet_w = n * output_size
        sheet_h = output_size

        sheet = Image.new("RGBA", (sheet_w, sheet_h), (0, 0, 0, 0))

        for i, frame in enumerate(frames):
            sheet.paste(frame, (i * output_size, 0))

        sheet.save(output_path, "PNG")

        # Salva metadata
        pl = int(max(2, min(int(palette_levels), 32)))
        metadata = {
            "animation": animation_label or "generated",
            "frame_count": n,
            "cell_size": output_size,
            "fps": 12,
            "loop": True,
            "chain_frames": chain_frames,
            "motion_from": motion_from,
            "enhance_output": enhance_output,
            "palette_cap": palette_cap,
            "palette_levels": pl,
            "palette_grid_max_colors": pl**3,
        }

        meta_path = output_path.replace(".png", ".json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[OK] Spritesheet salvata: {output_path}")
        print(f"  Metadata: {meta_path}\n")


def main(
    checkpoint: str = "checkpoints/model_final.pth",
    source_image: str | None = None,
    frame_count: int = 8,
    animation: str = "idle",
    input_frame: int = 0,
    chain_frames: bool = True,
    motion_from: str | None = None,
    enhance_output: bool = True,
    palette_cap: bool = True,
    palette_levels: int = 3,
    random_source: bool = False,
):
    """Entry point inferenza."""

    # Verifica checkpoint
    if not Path(checkpoint).exists():
        print(f"[ERROR] Checkpoint non trovato: {checkpoint}")
        print("Esegui prima il training: python src/training/quick_train.py")
        return

    if random_source and source_image:
        print("[WARN] --random-input ignorato perche' e' gia' impostato --input.\n")

    if random_source and not source_image:
        picked, picked_anim = pick_random_source_entry()
        if picked:
            source_image = picked
            print(f"[OK] Input casuale dal dataset:\n  {source_image}\n")
            if picked_anim:
                animation = picked_anim
                print(
                    f"[OK] Animazione allineata al ref (evita mismatch idle+death, ecc.): "
                    f"{animation}\n"
                )
        else:
            print(
                "[WARN] --random-input: nessun PNG in data/raw/licensed "
                "(manifest vuoto o assente: esegui python run.py catalog).\n"
            )

    if not source_image and motion_from is None:
        print(
            "[TIP] Senza --input il ref e' un canvas grigio: l'uscita puo' restare "
            "grigia/trasparente. Usa --input <sprite.png> oppure --random-input.\n"
        )

    # Crea generator
    generator = AnimationDevGenerator(checkpoint)

    # Crea cartella output
    Path("outputs").mkdir(parents=True, exist_ok=True)

    # Genera a diverse risoluzioni
    print("=== Generazione Sprite ===\n")

    eff_chain = chain_frames and not motion_from

    for size in [256, 128, 64]:
        frames = generator.generate_sheet(
            animation=animation,
            frame_count=frame_count,
            output_size=size,
            source_image=source_image,
            input_frame=input_frame,
            chain_frames=eff_chain,
            motion_from=motion_from,
            enhance_output=enhance_output,
            palette_cap=palette_cap,
            palette_levels=palette_levels,
        )
        generator.export_sheet(
            frames,
            f"outputs/animation_{size}x{size}.png",
            size,
            animation_label=animation,
            chain_frames=eff_chain,
            motion_from=motion_from,
            enhance_output=enhance_output,
            palette_cap=palette_cap,
            palette_levels=palette_levels,
        )


if __name__ == "__main__":
    main()
