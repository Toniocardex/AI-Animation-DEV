"""
src/training/losses_universal.py
Loss function con tutorial rules per pixel art da dataset multi-autore
Forza coerenza strutturale ignorando le differenze stilistiche
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class UniversalPixelArtLoss(nn.Module):
    """
    Loss function specializzata per dataset multi-autore.
    
    Problema: dataset da 5-10 autori diversi contiene conflitti stilistici
    ├── Autore A: pixel isolati ok, palette satura
    ├── Autore B: no pixel isolati, palette desaturata
    ├── Autore C: bordi morbidi, colori vivaci
    └── ...
    
    Soluzione: Loss con tutorial rules che forza coerenza strutturale
    sopra le differenze stilistiche.
    
    Regole applicate (estratte dai tutorial):
    1. No pixel isolati (cluster minimo 2px)
    2. No anti-aliasing (bordi netti)
    3. Palette limitata (max N colori)
    4. Contorni scuri (outline enforcement)
    5. Ricostruzione fedele (base loss)
    """

    def __init__(self, max_colors: int = 32, weights: dict = None):
        """
        Args:
            max_colors: numero massimo di colori per sprite
            weights: dizionario con pesi per ogni loss component
                    default: bilanciato per dataset multi-autore
        """
        super().__init__()
        self.max_colors = max_colors

        # Pesi ottimizzati per dataset multi-autore
        self.weights = weights or {
            "reconstruction": 1.0,
            "color_economy": 1.5,
            "palette_snap": 1.0,  # max ~32 colori (3 livelli/canale -> 27 triple)
            "cluster": 2.0,
            "antialiasing": 2.2,
            "outline": 1.0,
            "perceptual": 0.5,
        }

        # Aggiorna con custom weights se forniti
        if weights:
            self.weights.update(weights)

    # ────────────────────────────────────────────────────────────────────
    # LOSS 1: RICOSTRUZIONE BASE
    # ────────────────────────────────────────────────────────────────────

    def reconstruction_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        alpha: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Fedeltà base tra output e target (L1).
        Con alpha: pesa di più i pixel del personaggio (0.2 sfondo + 0.8*alpha)
        così texture e palette interne non restano secondarie rispetto al fondo.
        """
        diff = (pred - target).abs()
        if alpha is None:
            return diff.mean()
        w = 0.2 + 0.8 * alpha.clamp(0.0, 1.0)
        w = w.expand_as(pred)
        return diff.mul(w).sum() / w.sum().clamp(min=1e-6)

    # ────────────────────────────────────────────────────────────────────
    # LOSS 2: COLOR ECONOMY
    # ────────────────────────────────────────────────────────────────────

    def color_economy_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Penalizza output con troppi colori unici.
        Non vincola QUALI colori, solo QUANTI.
        
        Regola tutorial: "Pixel art usa palette limitata"
        
        Risolve il conflitto:
        - Autore A usa 50 colori
        - Autore B usa 16 colori
        → Forza il modello a stare in range ragionevole (24-32)
        """
        B, C, H, W = pred.shape

        # Quantizza a 8 livelli per valutare unicità
        # (riduce noise dovuto a floating point)
        quantized = (pred * 8).floor() / 8
        pixels = quantized.permute(0, 2, 3, 1).reshape(B, -1, C)

        # Stima diversità cromatica via varianza
        # Alta varianza = troppi colori diversi
        color_variance = pixels.var(dim=1).mean()

        return color_variance

    def palette_snap_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Spinge i colori verso una griglia grossolana così che le triple RGB
        possibili siano al massimo L^3 con L = round(max_colors^(1/3)).
        Per max_colors=32 -> L=3 -> 27 combinazioni (sotto il tetto 32).
        """
        L = max(2, int(round(self.max_colors ** (1.0 / 3.0))))
        q = torch.round(pred.clamp(0.0, 1.0) * (L - 1)) / (L - 1)
        return F.l1_loss(pred, q)

    # ────────────────────────────────────────────────────────────────────
    # LOSS 3: CLUSTER ENFORCEMENT (NO PIXEL ISOLATI)
    # ────────────────────────────────────────────────────────────────────

    def cluster_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Penalizza rumore a singolo pixel su patch quasi uniformi.

        La versione precedente (media |Δ| tra vicini) minimizzava *ogni* contrasto
        locale e appiattiva anche la texture interna dello sprite (silhouette piatta).

        Qui: luminosità y, media locale 3×3 μ; varianza locale σ². Su patch uniforme
        σ²≈0 e un pixel fuori posto ha |y−μ| alto → penalizzato. Sui bordi veri σ²
        è alta → peso basso (exp(−σ²/τ)).
        """
        y = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        mu = F.avg_pool2d(y, 3, stride=1, padding=1)
        ey = F.avg_pool2d(y, 3, stride=1, padding=1)
        ey2 = F.avg_pool2d(y * y, 3, stride=1, padding=1)
        local_var = (ey2 - ey * ey).clamp(min=0.0)
        dev = (y - mu).abs()
        tau = 0.03
        gate = torch.exp(-local_var / (tau + 1e-8))
        return (dev * gate).mean()

    # ────────────────────────────────────────────────────────────────────
    # LOSS 4: ANTI-ALIASING PREVENTION
    # ────────────────────────────────────────────────────────────────────

    def antialiasing_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Penalizza transizioni graduali di colore (anti-aliasing).
        Forza transizioni nette: colore A → colore B (no intermedi).
        
        Regola tutorial: "Il passaggio tra colori deve essere netto"
        
        Risolve il conflitto:
        - Autore A: usa anti-alias (stile più morbido)
        - Autore B: no anti-alias (stile più crudo)
        → Forza output senza anti-alias = standard pixel art
        """
        # Calcola gradienti
        grad_h = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        grad_v = pred[:, :, 1:, :] - pred[:, :, :-1, :]

        # In pixel art reale i gradienti sono bimodali:
        # - Vicini a 0 (stesso colore) ✓
        # - Grandi > 0.4 (cambio colore netto) ✓
        # - Intermedi 0.1-0.4 (anti-alias) ✗ penalizza questi
        
        def intermediate_penalty(grad):
            abs_grad = grad.abs()
            # Penalizza range 0.05-0.45 (valori "sospetti" intermedi)
            penalty = torch.where(
                (abs_grad > 0.03) & (abs_grad < 0.5),
                abs_grad,
                torch.zeros_like(abs_grad),
            )
            return penalty.mean()

        return intermediate_penalty(grad_h) + intermediate_penalty(grad_v)

    # ────────────────────────────────────────────────────────────────────
    # LOSS 5: OUTLINE ENFORCEMENT (CONTORNI SCURI)
    # ────────────────────────────────────────────────────────────────────

    def outline_loss(self, pred: torch.Tensor, 
                     alpha: torch.Tensor) -> torch.Tensor:
        """
        Garantisce che i bordi del personaggio siano 1px netti e scuri.
        
        Regola tutorial: "Il contorno esterno deve essere sempre scuro, 1px"
        
        Risolve il conflitto:
        - Autore A: contorno in nero puro
        - Autore B: contorno in grigio scuro
        - Autore C: contorno in marrone scuro
        → Forza i contorni a essere sempre nella fascia scura
        """
        # Rileva bordi via Sobel sull'alpha channel
        sobel_h = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_v = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        sobel_h = sobel_h.to(pred.device)
        sobel_v = sobel_v.to(pred.device)

        edge_h = F.conv2d(alpha, sobel_h, padding=1)
        edge_v = F.conv2d(alpha, sobel_v, padding=1)
        edge_magnitude = torch.sqrt(edge_h**2 + edge_v**2 + 1e-8)

        # Ai bordi (edge_magnitude alta), il colore deve essere scuro
        edge_mask = (edge_magnitude > 0.5).float()

        # Luminosità media ai bordi (vogliamo che sia bassa = scura)
        luminosity = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        border_luminosity = (luminosity * edge_mask).sum() / (edge_mask.sum() + 1e-8)

        return border_luminosity  # Minimizza = bordi scuri

    # ────────────────────────────────────────────────────────────────────
    # LOSS 6: PERCEPTUAL LOSS (OPTIONAL)
    # ────────────────────────────────────────────────────────────────────

    def perceptual_loss(self, pred: torch.Tensor, 
                       target: torch.Tensor) -> torch.Tensor:
        """
        Misura somiglianza percettiva generale (non pixel-per-pixel).
        Utile per catturare coerenza di alto livello tra autori diversi.
        """
        # Versione semplificata: differenza su blur morbido
        pred_blur = F.avg_pool2d(pred, kernel_size=3, stride=1, padding=1)
        target_blur = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)
        return F.l1_loss(pred_blur, target_blur)

    # ────────────────────────────────────────────────────────────────────
    # FORWARD PASS
    # ────────────────────────────────────────────────────────────────────

    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        alpha: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict]:
        """
        Calcola loss totale come combinazione pesata di sub-losses.
        
        Args:
            pred: output del modello [B, 3, 256, 256] RGB
            target: target desiderato [B, 3, 256, 256] RGB
            alpha: canale alpha [B, 1, 256, 256] per outline loss
        
        Returns:
            (loss_totale, dizionario_breakdown)
        """
        losses = {}

        # Loss base (alpha enfatizza area personaggio)
        losses["reconstruction"] = self.reconstruction_loss(pred, target, alpha)
        losses["color_economy"] = self.color_economy_loss(pred)
        losses["palette_snap"] = self.palette_snap_loss(pred)
        losses["cluster"] = self.cluster_loss(pred)
        losses["antialiasing"] = self.antialiasing_loss(pred)

        # Optional losses
        if alpha is not None:
            losses["outline"] = self.outline_loss(pred, alpha)

        losses["perceptual"] = self.perceptual_loss(pred, target)

        # Loss totale pesata
        total = sum(
            self.weights.get(k, 1.0) * v 
            for k, v in losses.items()
        )

        return total, losses

    def get_weights_info(self) -> str:
        """Ritorna stringa con info sui pesi."""
        info = "Loss Weights:\n"
        for name, weight in sorted(self.weights.items()):
            info += f"  {name:20s}: {weight:.2f}\n"
        return info


# ════════════════════════════════════════════════════════════════════════════
# PRESET CONFIGURATIONS
# ════════════════════════════════════════════════════════════════════════════

class LossPresets:
    """Configurazioni preset per diversi tipi di dataset."""

    @staticmethod
    def multi_author_balanced() -> dict:
        """
        Per dataset da 5-10 autori diversi.
        Bilanciamento ottimale tra coerenza e flessibilità stilistica.
        
        ✓ Cluster su isolati (non texture densa)
        ✓ Anti-aliasing FORTE (bordi netti)
        ✓ Color economy MODERATO (palette limitata ma varia)
        """
        return {
            "reconstruction": 1.25,
            "color_economy": 1.5,
            "palette_snap": 1.1,
            "cluster": 0.65,
            "antialiasing": 2.5,
            "outline": 1.0,
            "perceptual": 0.5,
        }

    @staticmethod
    def multi_author_strict() -> dict:
        """
        Per dataset molto eterogeneo con stili molto diversi.
        Massima coerenza strutturale.
        
        ✓ Tutti i vincoli FORTI
        ✓ Meno flessibilità stilistica
        """
        return {
            "reconstruction": 1.2,
            "color_economy": 2.0,
            "palette_snap": 1.5,
            "cluster": 0.9,
            "antialiasing": 3.0,
            "outline": 1.5,
            "perceptual": 1.0,
        }

    @staticmethod
    def multi_author_permissive() -> dict:
        """
        Per dataset già coerente da 2-3 autori.
        Massima flessibilità stilistica mantenendo regole base.
        
        ✓ Vincoli MODERATI
        ✓ Massima varietà stilistica
        """
        return {
            "reconstruction": 1.0,
            "color_economy": 1.0,
            "palette_snap": 0.6,
            "cluster": 1.5,
            "antialiasing": 1.8,
            "outline": 0.5,
            "perceptual": 0.3,
        }

    @staticmethod
    def multi_author_sharp() -> dict:
        """
        Meno “fangosità”: più peso su ricostruzione e perceptual,
        cluster isolato (non più media gradiente) + econ palette moderate.
        """
        return {
            "reconstruction": 1.55,
            "color_economy": 1.15,
            "palette_snap": 1.0,
            "cluster": 0.35,
            "antialiasing": 2.35,
            "outline": 0.85,
            "perceptual": 0.55,
        }

    @staticmethod
    def single_author() -> dict:
        """
        Per dataset da UN SOLO autore.
        Meno vincoli, il dataset insegna lo stile.
        """
        return {
            "reconstruction": 1.0,
            "color_economy": 0.5,
            "palette_snap": 1.0,
            "cluster": 0.5,
            "antialiasing": 2.0,
            "outline": 0.3,
            "perceptual": 0.1,
        }


if __name__ == "__main__":
    # Test
    print("UniversalPixelArtLoss loaded")
    
    # Test preset
    preset = LossPresets.multi_author_balanced()
    print("\nPreset multi_author_balanced:")
    for k, v in preset.items():
        print(f"  {k}: {v}")
