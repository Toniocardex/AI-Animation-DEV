"""
src/model/simple_model.py
U-Net leggera from scratch per pixel art
"""
import torch
import torch.nn as nn

from src.model.conditioning import COND_CHANNELS


class ResBlock(nn.Module):
    """Residual block semplice."""

    def __init__(self, channels: int):
        super().__init__()

        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)

        return x + h  # Skip connection


class DownBlock(nn.Module):
    """Encoder block con downsampling."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.res = ResBlock(out_ch)
        self.down = nn.Conv2d(out_ch, out_ch, 2, stride=2)
        self.norm = nn.GroupNorm(8, out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.norm(self.conv(x)))
        x = self.res(x)
        skip = x
        x = self.down(x)
        return x, skip  # (downsampled, skip connection)


class UpBlock(nn.Module):
    """Decoder block con upsampling."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.conv = nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1)
        self.res = ResBlock(out_ch)
        self.norm = nn.GroupNorm(8, out_ch)
        self.act = nn.ReLU()

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.norm(self.conv(x)))
        x = self.res(x)
        return x


# RGB (aspetto ref) + 1 canale pose (silhouette temporale) + condizioni broadcast
RGB_REF_CHANNELS = 3
POSE_CHANNELS = 1
INPUT_RGB_POSE_COND = RGB_REF_CHANNELS + POSE_CHANNELS + COND_CHANNELS


def compose_residual_rgba(ref_rgb: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
    """
    Combina ref RGB con delta predetto (primi 3 canali raw) e alpha (4° canale).

    raw[:, :3] = tanh(δ) in (-1, 1) — niente Sigmoid sul colore (meno compressione verso il grigio).
    """
    rgb = (ref_rgb + raw[:, :3]).clamp(0.0, 1.0)
    return torch.cat([rgb, raw[:, 3:4]], dim=1)


class SimpleUNet(nn.Module):
    """
    U-Net semplice per pixel art.
    Input:  [B, 3 + 1 + COND_CHANNELS, 256, 256] (RGB ref + pose L + cond)
    Output: [B, 4, 256, 256] — primi 3 canali: delta RGB (tanh); 4°: alpha (sigmoid).
    In training/inferenza: RGB finale = clamp(ref + delta).
    """

    CHANNELS = [32, 64, 128, 256]

    def __init__(self):
        super().__init__()
        C = self.CHANNELS
        in_rgb_cond = INPUT_RGB_POSE_COND

        # Encoder
        self.down1 = DownBlock(in_rgb_cond, C[0])  # 256 → 128
        self.down2 = DownBlock(C[0], C[1])   # 128 → 64
        self.down3 = DownBlock(C[1], C[2])   # 64 → 32
        self.down4 = DownBlock(C[2], C[3])   # 32 → 16

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(C[3], C[3] * 2, 3, padding=1),
            nn.GroupNorm(8, C[3] * 2),
            nn.ReLU(),
            nn.Conv2d(C[3] * 2, C[3], 3, padding=1),
            nn.GroupNorm(8, C[3]),
            nn.ReLU(),
        )
        self.btn_res = ResBlock(C[3])

        # Decoder
        self.up4 = UpBlock(C[3], C[3], C[2])  # 16 → 32
        self.up3 = UpBlock(C[2], C[2], C[1])  # 32 → 64
        self.up2 = UpBlock(C[1], C[1], C[0])  # 64 → 128
        self.up1 = UpBlock(C[0], C[0], C[0])  # 128 → 256

        # Testa: delta RGB (tanh, no sigmoid) + alpha (sigmoid)
        self.out_mid = nn.Sequential(
            nn.Conv2d(C[0], 16, 3, padding=1),
            nn.ReLU(),
        )
        self.out_delta = nn.Conv2d(16, 3, 1)
        self.out_alpha = nn.Sequential(
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x: [B, 3 + 1 + COND_CHANNELS, 256, 256]
        return: [B, 4, 256, 256]
        """
        # Encoder con skip connections
        x, s1 = self.down1(x)  # [B, C[0], 128, 128]
        x, s2 = self.down2(x)  # [B, C[1], 64, 64]
        x, s3 = self.down3(x)  # [B, C[2], 32, 32]
        x, s4 = self.down4(x)  # [B, C[3], 16, 16]

        # Bottleneck
        x = self.bottleneck(x)  # [B, C[3], 16, 16]
        x = self.btn_res(x)

        # Decoder
        x = self.up4(x, s4)  # [B, C[2], 32, 32]
        x = self.up3(x, s3)  # [B, C[1], 64, 64]
        x = self.up2(x, s2)  # [B, C[0], 128, 128]
        x = self.up1(x, s1)  # [B, C[0], 256, 256]

        h = self.out_mid(x)
        delta = torch.tanh(self.out_delta(h))
        alpha = self.out_alpha(h)
        return torch.cat([delta, alpha], dim=1)


def count_parameters(model: nn.Module) -> int:
    """Conta i parametri del modello."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from src.model.conditioning import build_cond_planes

    model = SimpleUNet()
    print(f"SimpleUNet creato")
    print(f"Parametri: {count_parameters(model):,}")

    # Test forward pass (RGB + cond)
    rgb = torch.randn(2, 3, 256, 256)
    pose = torch.randn(2, 1, 256, 256)
    cond = build_cond_planes(
        torch.tensor([0, 2]),
        torch.tensor([0, 3]),
        torch.tensor([8, 8]),
        256,
        256,
    )
    x = torch.cat([rgb, pose, cond], dim=1)
    raw = model(x)
    y = compose_residual_rgba(rgb, raw)
    print(f"Input shape:  {x.shape}")
    print(f"Raw shape:    {raw.shape}")
    print(f"Composed RGBA:{y.shape}")
    assert y.shape == (2, 4, 256, 256), "Output shape mismatch!"
    print("✓ Forward pass OK")
