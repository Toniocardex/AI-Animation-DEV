"""
src/preprocessing/quick_processor.py
Preprocessa spritesheet a qualsiasi risoluzione e le normalizza a 256×256
"""
import numpy as np
from PIL import Image
from pathlib import Path
import json


class QuickSpriteProcessor:
    """
    Processa spritesheet:
    1. Estrae frame singoli
    2. Rimuove background
    3. Normalizza a 256×256
    4. Ritorna frame processati
    """

    TARGET_SIZE = (256, 256)
    PIVOT_Y = 230  # 90% dell'altezza

    @staticmethod
    def pose_from_rgba(rgba: Image.Image) -> Image.Image:
        """Silhouette 256×256 (L) dall'alpha, allineata al canvas dello sprite."""
        arr = np.array(rgba.convert("RGBA"), dtype=np.uint8)
        alpha = arr[:, :, 3]
        return Image.fromarray(alpha, mode="L")

    def process_and_split(
        self, img_path: str, animation: str, estimated_frames: int = 8
    ):
        """
        Processa una spritesheet orizzontale.

        Args:
            img_path: percorso al file immagine
            animation: nome dell'animazione (idle, walk, run, ecc)
            estimated_frames: numero frame da estrarre se detection fallisce

        Returns:
            lista di frame processati con metadata
        """
        img = Image.open(img_path).convert("RGBA")

        # Rileva numero di frame dalla geometria
        if img.width > img.height * 2:
            # Probabilmente spritesheet orizzontale
            n_frames = round(img.width / img.height)
        else:
            # Usa stima
            n_frames = estimated_frames

        # Se numero frame è strano, usa stima
        if n_frames < 1:
            n_frames = estimated_frames

        frame_w = img.width // n_frames
        frame_h = img.height

        frames = []

        for i in range(n_frames):
            x = i * frame_w
            frame = img.crop((x, 0, x + frame_w, frame_h))

            # Rimuovi background
            frame = self._remove_bg(frame)

            # Normalizza a canvas
            frame = self._normalize_canvas(frame)

            frames.append(
                {
                    "image": frame,
                    "animation": animation,
                    "frame_idx": i,
                    "total_frames": n_frames,
                }
            )

        return frames

    def extract_frame_for_inference(
        self, img_path: str, frame_index: int = 0
    ) -> tuple[Image.Image, int]:
        """
        Allinea l'inferenza al training: spritesheet orizzontale -> un solo frame
        su canvas 256×256 (stesso flusso di process_and_split + normalize).

        Se l'immagine non è una striscia orizzontale (width > height*2), usa
        l'intera immagine come un unico frame (come process_single_frame).

        Returns:
            (frame_rgba_256, n_strip_frames) con n_strip_frames=1 se non è striscia.
        """
        img = Image.open(img_path).convert("RGBA")
        w, h = img.size

        if w > h * 2:
            n_frames = max(1, round(w / max(h, 1)))
            frame_w = w // n_frames
            frame_index = max(0, min(int(frame_index), n_frames - 1))
            x = frame_index * frame_w
            frame = img.crop((x, 0, x + frame_w, h))
        else:
            n_frames = 1
            frame = img

        frame = self._remove_bg(frame)
        return self._normalize_canvas(frame), n_frames

    def process_single_frame(self, img_path: str, animation: str = "idle"):
        """
        Processa un singolo frame.

        Args:
            img_path: percorso al file
            animation: tipo di animazione

        Returns:
            lista con un singolo frame processato
        """
        img = Image.open(img_path).convert("RGBA")
        img = self._remove_bg(img)
        img = self._normalize_canvas(img)

        return [
            {
                "image": img,
                "animation": animation,
                "frame_idx": 0,
                "total_frames": 1,
            }
        ]

    def _remove_bg(self, img: Image.Image) -> Image.Image:
        """Rimuove il background dell'immagine."""
        data = np.array(img)

        # Se già ha trasparenza, usa quella
        if data[:, :, 3].min() == 0:
            return img

        # Altrimenti rileva colore angolo (background)
        bg = data[0, 0, :3]
        diff = np.abs(data[:, :, :3].astype(int) - bg.astype(int))
        mask = diff.max(axis=2) < 20
        data[mask, 3] = 0

        return Image.fromarray(data)

    def _normalize_canvas(self, img: Image.Image) -> Image.Image:
        """
        Normalizza il frame a canvas 256×256 con pivot allineato.
        
        Processo:
        1. Crop al contenuto non-trasparente
        2. Scale a altezza 128px (metà canvas)
        3. Centra orizzontalmente
        4. Allinea i piedi al pivot Y
        """
        # Crop a contenuto
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
        else:
            # Immagine vuota
            return Image.new("RGBA", self.TARGET_SIZE, (0, 0, 0, 0))

        # Scale a altezza 128px (mantenendo proporzioni)
        if img.height > 0:
            ratio = 128 / img.height
            new_w = max(1, int(img.width * ratio))
            img = img.resize((new_w, 128), Image.NEAREST)  # NEAREST per pixel art

        # Centra su canvas 256×256
        canvas = Image.new("RGBA", self.TARGET_SIZE, (0, 0, 0, 0))
        x = (self.TARGET_SIZE[0] - img.width) // 2
        y = self.PIVOT_Y - img.height

        canvas.paste(img, (x, max(0, y)))

        return canvas


def load_hint_file(img_path: str) -> dict | None:
    """
    Carica file hint se esiste accanto all'immagine.
    
    Formato del hint file (img.hint.json):
    {
        "frame_width": 64,
        "frame_height": 80,
        "frames": 8,
        "start_x": 0,
        "start_y": 0
    }
    """
    hint_path = img_path.replace(".png", ".hint.json").replace(".gif", ".hint.json")
    if Path(hint_path).exists():
        try:
            with open(hint_path) as f:
                return json.load(f)
        except Exception as e:
            print(f"Errore lettura hint file {hint_path}: {e}")
    return None


if __name__ == "__main__":
    # Test veloce
    processor = QuickSpriteProcessor()

    # Simula processing
    print("QuickSpriteProcessor caricato e pronto")
