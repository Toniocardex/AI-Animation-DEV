"""
tools/catalog_assets.py
Cataloga gli asset da addestrare, con supporto per hint file
"""
import json
from pathlib import Path
from PIL import Image
import numpy as np


class SimpleAssetCataloguer:
    """
    Cataloga i file immagine da un pack.
    Crea un manifest.json con metadata di ogni asset.
    """

    def __init__(self):
        self.entries = []

    def catalog_pack(self, pack_dir: str, license_type: str, author: str):
        """
        Cataloga tutti i file immagine in una cartella.

        Args:
            pack_dir: percorso alla cartella del pack
            license_type: "commercial_training_ok", "cc0", ecc
            author: nome dell'autore
        """
        pack_path = Path(pack_dir)

        image_files = list(pack_path.glob("*.png")) + \
                      list(pack_path.glob("*.gif")) + \
                      list(pack_path.glob("**/*.png")) + \
                      list(pack_path.glob("**/*.gif"))

        # Rimuovi duplicati
        image_files = list(set(image_files))
        image_files = self._dedupe_nobkg_variants(image_files)

        print(f"Trovati {len(image_files)} file in {pack_dir}")

        for img_path in sorted(image_files):
            try:
                img = Image.open(img_path)

                # Salta se troppo piccolo
                if img.width < 32 or img.height < 32:
                    continue

                # Salta immagini di preview (bannert, poster, ecc)
                if self._is_preview_image(img_path.name):
                    continue

                # Inferisci animazione dal nome file
                name_lower = img_path.stem.lower()

                animation = "idle"
                if "walk" in name_lower:
                    animation = "walk"
                elif "run" in name_lower:
                    animation = "run"
                elif "attack" in name_lower or "slash" in name_lower:
                    animation = "attack"
                elif "jump" in name_lower:
                    animation = "jump"
                elif "death" in name_lower or "die" in name_lower:
                    animation = "death"

                # Rileva tipo asset
                asset_type = self._detect_asset_type(img)

                # Carica hint file se esiste
                hint = self._load_hint(img_path)

                entry = {
                    "path": str(img_path),
                    "filename": img_path.name,
                    "license": license_type,
                    "author": author,
                    "animation": animation,
                    "width": img.width,
                    "height": img.height,
                    "asset_type": asset_type,
                    "hint": hint,
                    "usable": True,
                }

                self.entries.append(entry)
                print(f"  [OK] {img_path.name:40s} -> {animation}")

            except Exception as e:
                print(f"  [ERR] {img_path.name}: {e}")

    def _is_preview_image(self, filename: str) -> bool:
        """Rileva se è un'immagine di preview."""
        keywords = [
            "preview", "banner", "poster", "cover", "thumbnail",
            "icon", "ui", "button", "logo", "splash",
        ]
        name_lower = filename.lower()
        return any(kw in name_lower for kw in keywords)

    def _dedupe_nobkg_variants(self, image_files: list[Path]) -> list[Path]:
        """
        Deduplica varianti con prefisso noBKG_.
        Se esistono sia X.png che noBKG_X.png, mantiene solo noBKG_X.png.
        """
        grouped: dict[str, list[Path]] = {}
        for img_path in image_files:
            name = img_path.name.lower()
            if name.startswith("nobkg_"):
                canonical = name[len("nobkg_"):]
            else:
                canonical = name
            grouped.setdefault(canonical, []).append(img_path)

        selected: list[Path] = []
        for variants in grouped.values():
            # Preferisci variante noBKG_ se disponibile
            preferred = None
            for candidate in variants:
                if candidate.name.lower().startswith("nobkg_"):
                    preferred = candidate
                    break
            selected.append(preferred or sorted(variants)[0])

        return selected

    def _detect_asset_type(self, img: Image.Image) -> str:
        """Distingue spritesheet da frame singolo."""
        ratio = img.width / max(img.height, 1)
        if ratio > 2.5:
            return "spritesheet_horizontal"
        if abs(ratio - 1.0) < 0.2:  # Quasi quadrata
            return "spritesheet_grid"
        return "single_frame"

    def _load_hint(self, img_path: Path) -> dict | None:
        """Carica hint file se esiste."""
        hint_path = img_path.with_suffix(".hint.json")
        if hint_path.exists():
            try:
                with open(hint_path) as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def save_manifest(self, output_path: str):
        """Salva il manifest."""
        with open(output_path, "w") as f:
            json.dump(self.entries, f, indent=2)

        print(f"\n[OK] Manifest salvato: {output_path}")
        print(f"  Total entries: {len(self.entries)}\n")

        # Riepilogo per animazione
        from collections import Counter
        animations = Counter(e["animation"] for e in self.entries)
        print("Distribuzione animazioni:")
        for anim, count in animations.most_common():
            print(f"  {anim:15s}: {count:4d}")
        print()


def main():
    """
    Cataloga i pack con licenza training-ok.
    Modifica qui per aggiungere i tuoi pack.
    """
    cataloguer = SimpleAssetCataloguer()

    # MODIFICA QUESTA SEZIONE CON I TUOI PACK
    packs = [
        {
            "dir": "data/raw/licensed/AllCharacters",
            "license": "commercial_training_ok",
            "author": "AllCharacters",
        },
        # Aggiungi altri pack qui
        # {
        #     "dir": "data/raw/licensed/fantasy_pack",
        #     "license": "cc0",
        #     "author": "OpenGameArt",
        # },
    ]

    print("=== Asset Cataloguer ===\n")

    for pack in packs:
        if Path(pack["dir"]).exists():
            cataloguer.catalog_pack(
                pack_dir=pack["dir"],
                license_type=pack["license"],
                author=pack["author"],
            )
        else:
            print(f"[WARN] Pack non trovato: {pack['dir']}\n")

    # Salva manifest unificato
    cataloguer.save_manifest("data/raw/licensed/manifest.json")


if __name__ == "__main__":
    main()
