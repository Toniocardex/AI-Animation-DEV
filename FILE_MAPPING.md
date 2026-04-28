# File Mapping Guide

## Come disporre i file scaricati

Questo file ti dice esattamente dove mettere ogni file scaricato dalla cartella outputs.

> Nota: tieni `src_training_quick_train.py` come fallback, ma usa
> `src_training_quick_train_MODIFIED.py` come `src/training/quick_train.py`.

### Istruzioni Generali

1. Crea una nuova cartella vuota: `gothic-sprite-ai`
2. Per ogni file qui sotto:
   - Scarica il file
   - Crea la cartella se non esiste
   - Sposta il file con il nome indicato

---

## Mapping Completo

### Root (cartella principale)

| File Scaricato | Destinazione | Note |
|---|---|---|
| `requirements.txt` | `./requirements.txt` | Dipendenze Python |
| `run.py` | `./run.py` | Entrypoint principale |
| `README.md` | `./README.md` | Documentazione |
| `QUICK_START_CURSOR.md` | `./QUICK_START_CURSOR.md` | Guida setup Cursor |

### configs/

| File Scaricato | Destinazione |
|---|---|
| `configs_palette.yaml` | `./configs/palette.yaml` |

### src/

| File Scaricato | Destinazione |
|---|---|
| `src___init__.py` | `./src/__init__.py` |

### src/preprocessing/

| File Scaricato | Destinazione |
|---|---|
| `src_preprocessing___init__.py` | `./src/preprocessing/__init__.py` |
| `src_preprocessing_quick_processor.py` | `./src/preprocessing/quick_processor.py` |

### src/dataset/

| File Scaricato | Destinazione |
|---|---|
| `src_dataset___init__.py` | `./src/dataset/__init__.py` |
| `src_dataset_quick_build.py` | `./src/dataset/quick_build.py` |

### src/model/

| File Scaricato | Destinazione |
|---|---|
| `src_model___init__.py` | `./src/model/__init__.py` |
| `src_model_simple_model.py` | `./src/model/simple_model.py` |

### src/training/

| File Scaricato | Destinazione |
|---|---|
| `src_training___init__.py` | `./src/training/__init__.py` |
| `src_training_losses_universal.py` | `./src/training/losses_universal.py` |
| `src_training_quick_train_MODIFIED.py` | `./src/training/quick_train.py` |

### src/inference/

| File Scaricato | Destinazione |
|---|---|
| `src_inference___init__.py` | `./src/inference/__init__.py` |
| `src_inference_quick_generate.py` | `./src/inference/quick_generate.py` |

### tools/

| File Scaricato | Destinazione |
|---|---|
| `tools___init__.py` | `./tools/__init__.py` |
| `tools_catalog_assets.py` | `./tools/catalog_assets.py` |

---

## Cartelle da Creare (Vuote)

Crea queste cartelle (saranno popolate durante l'esecuzione):

```
data/
├── raw/
│   └── licensed/        ← Metti i tuoi asset qui
├── processed/           ← Creato da build
└── final/
    ├── train/           ← Creato da build
    ├── val/             ← Creato da build
    └── test/            ← Creato da build

checkpoints/            ← Creato da train
outputs/                ← Creato da generate
```

---

## Comando Bash per Setup Automatico

Se usi macOS/Linux, puoi automatizzare:

```bash
# Crea tutte le cartelle
mkdir -p gothic-sprite-ai/{src/{preprocessing,dataset,model,training,inference},tools,configs,data/{raw/licensed,processed,final/{train,val,test}},checkpoints,outputs}

# Vai nella cartella
cd gothic-sprite-ai

# Scarica/copia i file dalla cartella outputs di Claude
# Poi organizzali secondo il mapping sopra

# Crea virtualenv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Pronto!
```

---

## Verifiche

Dopo aver copiato tutti i file, verifica che la struttura sia corretta:

```powershell
# Verifica che tutti i file Python siano al posto giusto
Get-ChildItem -Recurse -Filter *.py | Select-Object -ExpandProperty FullName
```

Se un file manca, scaricalo di nuovo e copialo al posto giusto.

---

## Troubleshooting

**Errore: "No module named 'src'"**
- Verifica che `src/__init__.py` esista
- Verifica che stai eseguendo da dentro la cartella `gothic-sprite-ai`

**Errore: "No such file or directory"**
- Verifica il percorso del file
- Assicurati che la cartella esista

**Import errors**
- Assicurati che i nomi dei file siano esattamente come indicato
- Python distingue maiuscole e minuscole nei nomi file

---

## Prompt per Cursor

Se usi Cursor, puoi fare così:

1. Apri Cursor in una cartella vuota
2. Apri il terminal integrato
3. Esegui:
```bash
mkdir -p src/{preprocessing,dataset,model,training,inference} tools configs data/{raw/licensed,processed,final/{train,val,test}} checkpoints outputs
```

4. Poi copia-incolla i file uno per uno usando l'interfaccia di Cursor

---

Fatto! Ora puoi eseguire `python run.py catalog` e iniziare il workflow.

