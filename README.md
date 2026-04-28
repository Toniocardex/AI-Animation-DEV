# AI Animation Dev

Generatore di sprite sheet pixel art per giochi 2D platform usando AI.

Genera animazioni consistenti a partire dai tuoi asset, con supporto multi-risoluzione (64×64, 128×128, 256×256).

> Nota: questo export contiene sia file "flat" in root (es. `src_model_simple_model.py`)
> sia la struttura package (`src/...`, `tools/...`) da usare nel progetto finale.
> Lo script `run.py` ora supporta entrambi i formati.

---

## Installazione

### 1. Setup ambiente

```bash
# Crea la cartella del progetto
mkdir gothic-sprite-ai
cd gothic-sprite-ai

# Crea virtualenv
python3 -m venv venv

# Attiva virtualenv
# Su macOS/Linux:
source venv/bin/activate
# Su Windows:
venv\Scripts\activate

# Installa dipendenze
pip install -r requirements.txt
```

### 2. Organizza i tuoi asset

Copia i tuoi pack nella cartella corretta:

```
data/raw/licensed/
├── enemy_pack_kbpixel/
│   ├── idle.png
│   ├── run.png
│   ├── attack.png
│   └── ...
├── altro_pack/
│   ├── sprite1.png
│   ├── sprite2.png
│   └── ...
```

---

## Workflow

### Step 1: Cataloga gli asset

```bash
python run.py catalog
```

Questo:
- Scansiona i pack nella cartella `data/raw/licensed/`
- Rileva il tipo di animazione dal nome file
- Crea `manifest.json` con metadata

Output:
```
✓ idle.png                         → idle
✓ run.png                          → run
✓ attack.png                       → attack

Manifest salvato: data/raw/licensed/manifest.json
Totale entry: 7
```

**Se il sistema sbaglia a riconoscere il tipo di animazione:**

Crea un file hint accanto al PNG:

```json
// idle.hint.json
{
  "frame_width": 64,
  "frame_height": 80,
  "frames": 8,
  "start_x": 0,
  "start_y": 0
}
```

Il sistema lo leggerà automaticamente.

### Step 2: Costruisci il dataset

```bash
python run.py build
```

Questo:
- Processa ogni sprite dal manifest
- Estrae i frame dalle spritesheet
- Normalizza tutto a 256×256px
- Crea train/val/test split (80/10/10)
- Applica augmentation (flip orizzontale)

Output:
```
Processing Asset
████████████ 100% | 7/7

Totale frame estratti: 56
Dopo augmentation: 112 frame

Saving Dataset
✓ train     :    90 frame → data/final/train
✓ val       :    11 frame → data/final/val
✓ test      :    11 frame → data/final/test

✅ Dataset costruito con successo!
```

### Step 3: Addestra il modello

```bash
python run.py train --preset multi_author_balanced --epochs 50
```

Questo:
- Carica il dataset
- Addestra una U-Net leggera
- Applica preset di loss multi-autore (`--preset`)
- Salva checkpoint ogni 10 epoch
- Salva il best model

Output:
```
Device: cuda
Parametri modello: 3,450,892

Train: 90 | Val: 11

=== Training Started ===

Epoch  1 | Train: 0.1523 | Val: 0.1402
Epoch  2 | Train: 0.1204 | Val: 0.1089
             ✓ Nuovo best model salvato
...
Epoch 50 | Train: 0.0234 | Val: 0.0198

✅ Training completato!
Best model salvato in: checkpoints/model_best.pth
Final model salvato in: checkpoints/model_final.pth
```

Durata stimata: 5-15 minuti con GPU, 30-60 minuti con CPU.

### Step 4: Genera sprite

```bash
python run.py generate
```

Questo:
- Carica il modello addestrato
- Genera 8 frame di test animation
- Esporta a 256×256, 128×128, 64×64

Output:
```
✓ Modello caricato: checkpoints/model_final.pth
  Device: cuda

=== Generazione Sprite ===

Generazione: test_animation (8 frame @ 256×256px)
  8/8 frame ✓

✓ Spritesheet salvata: outputs/animation_256x256.png
  Metadata: outputs/animation_256x256.json

...

Generazione: test_animation (8 frame @ 64×64px)
  8/8 frame ✓

✓ Spritesheet salvato: outputs/animation_64x64.png
  Metadata: outputs/animation_64x64.json
```

Output generati:
- `outputs/animation_256x256.png` + `.json`
- `outputs/animation_128x128.png` + `.json`
- `outputs/animation_64x64.png` + `.json`

---

## Configurazione

### Personalizzare il training

Modifica `src/training/quick_train.py`:

```python
num_epochs = 50              # Aumenta per più qualità
batch_size = 4              # Aumenta se hai VRAM
learning_rate = 0.0001      # Più alto = training più veloce
```

### Personalizzare la generazione

Modifica `src/inference/quick_generate.py`:

```python
generator.generate_sheet(
    animation="idle",
    frame_count=8,
    output_size=256  # 64, 128, o 256
)
```

---

## Struttura Cartelle

```
gothic-sprite-ai/
├── data/
│   ├── raw/licensed/           ← I TUOI PACK VANNO QUI
│   │   └── manifest.json       ← Generato da catalog
│   └── final/
│       ├── train/              ← Dataset training
│       ├── val/                ← Dataset validation
│       └── test/               ← Dataset test
├── src/
│   ├── preprocessing/          ← Normalizzazione sprite
│   ├── dataset/                ← Dataset builder
│   ├── model/                  ← Modello U-Net
│   ├── training/               ← Training loop
│   └── inference/              ← Generazione sprite
├── tools/
│   └── catalog_assets.py       ← Catalogazione
├── checkpoints/                ← Modelli salvati
├── outputs/                    ← Sprite generati
└── run.py                      ← Entrypoint
```

---

## Troubleshooting

### "Manifest non trovato"

Esegui prima `python run.py catalog`

### "Checkpoint non trovato"

Esegui prima `python run.py train`

### CUDA out of memory

Riduci la batch size in `src/training/quick_train.py`:
```python
batch_size = 2  # Era 4
```

### Sprites di qualità bassa

Aumenta il numero di epoch:
```python
num_epochs = 100  # Era 50
```

---

## Risoluzioni Supportate

La generazione supporta:
- **256×256** - Massima qualità
- **128×128** - Buona qualità
- **64×64** - Qualità accettabile

Il modello è addestrato a 256×256 e downscala automaticamente.

Se vuoi massima qualità a 128×128 o 64×64, devi readdestare:

Modifica `src/preprocessing/quick_processor.py`:
```python
TARGET_SIZE = (128, 128)  # Invece di (256, 256)
```

Poi esegui il training da capo.

---

## Support

Problemi?
- Verifica che il dataset sia stato costruito: `dir data\final\train\sprites`
- Verifica il manifest: `python -c "import json;print(len(json.load(open('data/raw/licensed/manifest.json'))))"`
- Controlla i log durante il training
- Se CUDA non è disponibile, il training usa CPU (ma è più lento)

---

## License

Usa questo codice come vuoi. Rispetta la licenza dei tuoi asset.

