"""
src/training/quick_train.py
Training loop con validation
"""
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

try:
    from src.model.simple_model import SimpleUNet
    from src.training.losses_universal import UniversalPixelArtLoss, LossPresets
except ModuleNotFoundError:
    from src_model_simple_model import SimpleUNet
    from src_training_losses_universal import UniversalPixelArtLoss, LossPresets


class QuickDataset(Dataset):
    """Dataset che carica sprite e le relative annotazioni."""

    def __init__(self, split_dir: str):
        self.split_dir = Path(split_dir)

        ann_path = self.split_dir / "annotations.json"
        with open(ann_path) as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]

        img_path = self.split_dir / ann["sprite"]
        img = Image.open(img_path).convert("RGBA")

        # Converti in tensor float [0, 1]
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [C, H, W]

        # Separa RGB e alpha
        rgb = tensor[:3]  # [3, 256, 256]
        alpha = tensor[3:4]  # [1, 256, 256]

        return {
            "sprite": rgb,
            "alpha": alpha,
        }


def train(loss_preset: str = "multi_author_balanced", num_epochs: int = 50):
    """Training loop principale con preset loss multi-autore."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Crea cartelle
    Path("checkpoints").mkdir(exist_ok=True)

    # Modello
    model = SimpleUNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parametri modello: {total_params:,}\n")

    # Dati
    print("Caricamento dataset...")
    train_ds = QuickDataset("data/final/train")
    val_ds = QuickDataset("data/final/val")

    train_loader = DataLoader(
        train_ds, batch_size=4, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(val_ds, batch_size=4, num_workers=2)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}\n")

    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    preset_map = {
        "multi_author_balanced": LossPresets.multi_author_balanced(),
        "multi_author_strict": LossPresets.multi_author_strict(),
        "multi_author_permissive": LossPresets.multi_author_permissive(),
        "single_author": LossPresets.single_author(),
    }
    if loss_preset not in preset_map:
        print(f"⚠️ Preset '{loss_preset}' non valido, uso multi_author_balanced")
        loss_preset = "multi_author_balanced"
    criterion = UniversalPixelArtLoss(max_colors=32, weights=preset_map[loss_preset])
    print(f"Loss preset: {loss_preset}")
    print(criterion.get_weights_info())
    best_val_loss = float("inf")

    print("=== Training Started ===\n")

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [train]", leave=False):
            sprite = batch["sprite"].to(device)
            alpha = batch["alpha"].to(device)

            # Il modello impara a riprodurre lo sprite in input
            pred = model(sprite)

            loss, _ = criterion(pred[:, :3], sprite, alpha)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [val]", leave=False):
                sprite = batch["sprite"].to(device)
                alpha = batch["alpha"].to(device)

                pred = model(sprite)

                loss, _ = criterion(pred[:, :3], sprite, alpha)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Print
        print(f"Epoch {epoch+1:3d} | "
              f"Train: {train_loss:.4f} | "
              f"Val: {val_loss:.4f}")

        # Salva checkpoint migliore
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/model_best.pth")
            print("             ✓ Nuovo best model salvato")

        # Salva ogni 10 epoch
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"checkpoints/model_ep{epoch+1}.pth")

        scheduler.step()

    # Salva modello finale
    torch.save(model.state_dict(), "checkpoints/model_final.pth")
    print("\n✅ Training completato!")
    print(f"Best model salvato in: checkpoints/model_best.pth")
    print(f"Final model salvato in: checkpoints/model_final.pth")
    print(f"Loss preset usato: {loss_preset}")


if __name__ == "__main__":
    train()
