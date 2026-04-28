# Quick Start con Cursor

## Come Ricreare la Struttura in Cursor

> Nota: per il training multi-autore usa `src_training_quick_train_MODIFIED.py`
> come sorgente di `src/training/quick_train.py`.

### 1. Crea la cartella principale

```bash
mkdir gothic-sprite-ai
cd gothic-sprite-ai
```

### 2. Copia i file scaricati

Tutti i file sono nella cartella outputs di Claude:

**File di configurazione:**
- `configs_palette.yaml` → `configs/palette.yaml`
- `requirements.txt` → `requirements.txt`
- `run.py` → `run.py`
- `README.md` → `README.md`

**Modulo src/**
- `src___init__.py` → `src/__init__.py`

**src/preprocessing/**
- `src_preprocessing___init__.py` → `src/preprocessing/__init__.py`
- `src_preprocessing_quick_processor.py` → `src/preprocessing/quick_processor.py`

**src/dataset/**
- `src_dataset___init__.py` → `src/dataset/__init__.py`
- `src_dataset_quick_build.py` → `src/dataset/quick_build.py`

**src/model/**
- `src_model___init__.py` → `src/model/__init__.py`
- `src_model_simple_model.py` → `src/model/simple_model.py`

**src/training/**
- `src_training___init__.py` → `src/training/__init__.py`
- `src_training_quick_train.py` → `src/training/quick_train.py`

**src/inference/**
- `src_inference___init__.py` → `src/inference/__init__.py`
- `src_inference_quick_generate.py` → `src/inference/quick_generate.py`

**tools/**
- `tools___init__.py` → `tools/__init__.py`
- `tools_catalog_assets.py` → `tools/catalog_assets.py`

### 3. In Cursor: Workflow Consigliato

1. **Apri la cartella in Cursor:** File → Open Folder → seleziona `gothic-sprite-ai`

2. **Crea la struttura rapidamente:**
   - Usa Ctrl+Shift+P → "New File" per creare i file uno per uno
   - Oppure usa il terminal integrato:
   ```bash
   mkdir -p data/raw/licensed data/processed data/final/{train,val,test} src/{preprocessing,dataset,model,training,inference} tools configs checkpoints outputs
   ```

3. **Copia-incolla il contenuto:**
   - Apri ogni file dai risultati di Claude
   - Copia il contenuto (Ctrl+A, Ctrl+C)
   - In Cursor: Crea nuovo file (Ctrl+N)
   - Salva con il nome corretto (Ctrl+S)
   - Incolla il contenuto

### 4. Setup ambiente

Nel terminal di Cursor:

```bash
# Create virtualenv
python3 -m venv venv

# Activate
source venv/bin/activate  # macOS/Linux
# oppure
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 5. Copia i tuoi asset

```bash
cp -r ~/path/to/your/pack data/raw/licensed/
```

### 6. Esegui il workflow

```bash
# 1. Cataloga
python run.py catalog

# 2. Build dataset
python run.py build

# 3. Train
python run.py train --preset multi_author_balanced --epochs 50

# 4. Generate
python run.py generate
```

---

## File Structure Finale

```
gothic-sprite-ai/
├── venv/                          ← creato da pip
├── data/
│   ├── raw/
│   │   └── licensed/
│   │       ├── enemy_pack_kbpixel/
│   │       │   ├── idle.png
│   │       │   ├── run.png
│   │       │   └── ...
│   │       └── manifest.json      ← creato da catalog
│   ├── processed/                 ← vuoto (per future versioni)
│   └── final/
│       ├── train/
│       │   ├── sprites/
│       │   └── annotations.json
│       ├── val/
│       │   ├── sprites/
│       │   └── annotations.json
│       └── test/
│           ├── sprites/
│           └── annotations.json
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── quick_processor.py
│   ├── dataset/
│   │   ├── __init__.py
│   │   └── quick_build.py
│   ├── model/
│   │   ├── __init__.py
│   │   └── simple_model.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── quick_train.py
│   └── inference/
│       ├── __init__.py
│       └── quick_generate.py
├── tools/
│   ├── __init__.py
│   └── catalog_assets.py
├── configs/
│   └── palette.yaml
├── checkpoints/                   ← creati da training
│   ├── model_best.pth
│   ├── model_final.pth
│   └── ...
├── outputs/                       ← creati da generate
│   ├── animation_256x256.png
│   ├── animation_256x256.json
│   ├── animation_128x128.png
│   ├── animation_128x128.json
│   ├── animation_64x64.png
│   └── animation_64x64.json
├── requirements.txt
├── run.py
└── README.md
```

---

## Shortcuts Cursor

**Creazione rapida file:**
- Ctrl+N: Nuovo file
- Ctrl+S: Salva

**Terminal integrato:**
- Ctrl+` : Apri/chiudi

**Esecuzione commands:**
- Ctrl+Shift+P: Command palette

**Modifica veloce:**
- Ctrl+H: Find and replace
- Ctrl+F: Find

---

## Primo Test

Dopo il setup, fai un test veloce:

```bash
# Test import
python -c "from src.model.simple_model import SimpleUNet; print('✓ Modello importato correttamente')"

python -c "from src.preprocessing.quick_processor import QuickSpriteProcessor; print('✓ Processor importato correttamente')"

# Se entrambi funzionano, sei pronto!
```

---

## Note Importanti

1. **Virtualenv:** Assicurati di attivare sempre il venv prima di eseguire comandi
2. **Path:** Esegui sempre `python run.py` dalla cartella principale (dove c'è `run.py`)
3. **GPU:** Se non hai CUDA, il training userà CPU (più lento ma funziona)
4. **Asset:** I tuoi pack DEVONO avere licenza training-compatible

---

## Supporto

Se qualcosa non funziona:

1. Verifica che il file esista: `ls -la` (macOS/Linux) oppure `dir` (Windows)
2. Verifica che sia nel percorso corretto
3. Controlla l'indentazione del Python (gli spazi importano!)
4. Leggi i messaggi d'errore con attenzione

Buon divertimento con AI Animation Dev! 🎨

