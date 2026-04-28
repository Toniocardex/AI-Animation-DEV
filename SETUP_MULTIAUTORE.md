# AI Animation Dev — Setup Multi-Autore con Tutorial Rules

## 🎯 Cosa è Cambiato

Hai dataset da **5-10 autori diversi** su itch.io/gamedevmarket/simili.

Il nuovo setup implementa **UniversalPixelArtLoss** che forza coerenza strutturale sopra le differenze stilistiche.

---

## 📝 File Aggiuntivi

Due nuovi file da aggiungere al progetto:

```
src/training/
├── losses_universal.py           ← NUOVO
└── quick_train_MODIFIED.py       ← Rimpiazza il vecchio quick_train.py
```

### Come Integrare

1. **Scarica entrambi i file** dai risultati di Claude
2. **Rinomina:**
   ```bash
   losses_universal.py → src/training/losses_universal.py
   quick_train_MODIFIED.py → src/training/quick_train.py
   ```
3. **Elimina il vecchio** `src/training/quick_train.py`

---

## 🔧 Come Usare

### Setup Semplice (Predefinito)

```bash
python run.py train
```

Usa il preset **"multi_author_balanced"** (consigliato per 5-10 autori).

### Setup Avanzato (Con Opzioni)

```bash
# Per dataset estremamente eterogeneo (10+ autori diversi)
python run.py train --preset multi_author_strict

# Per dataset più omogeneo (2-3 autori)
python run.py train --preset multi_author_permissive

# Se usi solo UN autore (fallback)
python run.py train --preset single_author

# Con numero di epoch custom
python run.py train --preset multi_author_balanced --epochs 100
```

---

## 📊 I Quattro Preset

### 1. `multi_author_balanced` (CONSIGLIATO)

**Per:** Dataset da 5-10 autori diversi

```
Cluster enforcement:      2.0 (FORTE)  → no pixel isolati
Anti-aliasing:            1.5 (FORTE)  → bordi netti
Color economy:            1.5 (MEDIO)  → palette limitata
Reconstruction:           1.0 (BASE)   → fedeltà target
Outline:                  1.0 (MEDIO)  → contorni scuri
Perceptual:               0.5 (LIGHT)  → somiglianza generale
```

**Risultato atteso:**
- ✅ Output coerente a livello strutturale
- ✅ Varietà stilistica preservata
- ✅ Professionalmente consistente
- ✅ Non troppo artificiale

---

### 2. `multi_author_strict`

**Per:** Dataset molto eterogeneo (10+ autori, stili conflittuali)

```
Cluster enforcement:      3.0 (MASSIMO)
Anti-aliasing:            2.0 (FORTE)
Color economy:            2.0 (FORTE)
Reconstruction:           1.0 (BASE)
Outline:                  1.5 (FORTE)
Perceptual:               1.0 (MEDIO)
```

**Risultato atteso:**
- ✅ Massima coerenza strutturale
- ❌ Meno varietà stilistica
- ⚠️ Output più "artificiale" ma molto consistente

---

### 3. `multi_author_permissive`

**Per:** Dataset già coerente (2-3 autori simili)

```
Cluster enforcement:      1.5 (MEDIO)
Anti-aliasing:            1.0 (MEDIO)
Color economy:            1.0 (LIGHT)
Reconstruction:           1.0 (BASE)
Outline:                  0.5 (LIGHT)
Perceptual:               0.3 (LIGHT)
```

**Risultato atteso:**
- ✅ Massima flessibilità stilistica
- ✅ Meno artificioso
- ⚠️ Meno coerenza (se dataset è conflittuale)

---

### 4. `single_author`

**Per:** Dataset da UN SOLO autore (fallback)

```
Cluster enforcement:      0.5 (LIGHT)
Anti-aliasing:            0.5 (LIGHT)
Color economy:            0.5 (LIGHT)
Reconstruction:           1.0 (BASE)
Outline:                  0.3 (LIGHT)
Perceptual:               0.1 (LIGHT)
```

**Risultato atteso:**
- ✅ Dataset insegna lo stile naturalmente
- ✅ Minimal artificial constraints
- ✅ Output naturale dello stile singolo

---

## 🎯 Come Scegliere il Preset

```
Hai analizzato i tuoi asset e sono:

├── 10+ autori MOLTO diversi
│   └── Usa: multi_author_strict
│
├── 5-10 autori moderatamente diversi ✓ (CASO PIÙ PROBABILE)
│   └── Usa: multi_author_balanced (DEFAULT)
│
├── 2-3 autori abbastanza simili
│   └── Usa: multi_author_permissive
│
└── 1 solo autore
    └── Usa: single_author
```

---

## 📈 Cosa Aspettarti nei Numeri

### Dataset Multi-Autore (5-10 autori)

```
Senza tutorial rules (vecchio setup):
Epoch  10: Train=0.08, Val=0.09
Epoch  25: Train=0.04, Val=0.05
Epoch  50: Train=0.02, Val=0.02
Output: Stilisticamente caotico

Con tutorial rules (nuovo setup):
Epoch  10: Train=0.09, Val=0.10  (leggermente più alto)
Epoch  25: Train=0.05, Val=0.06  (leggermente più alto)
Epoch  50: Train=0.03, Val=0.03  (leggermente più alto)
Output: Stilisticamente coerente ✓ (questo è il guadagno)
```

La loss totale è più alta perché il modello rispetta più vincoli. **Questo è normale e desiderato.**

---

## 🔍 Interpretare i Loss Breakdown

Nel training vedrai output come:

```
Epoch  5 | Train: 0.0847 | Val: 0.0921 | Cluster: 0.0123 | AA: 0.0045

  Loss breakdown:
    reconstruction        : 0.0120
    color_economy        : 0.0015
    cluster              : 0.0123  ← Penalità per pixel isolati
    antialiasing         : 0.0045  ← Penalità per gradienti morbidi
    outline              : 0.0008
    perceptual           : 0.0002
```

**Cosa significa:**
- `cluster` alto → il modello sta lottando a eliminare pixel isolati
- `antialiasing` alto → il modello sta lottando a fare bordi netti
- Se scendono nel tempo → il modello sta imparando le regole ✓

---

## ⚡ Customizzazione Avanzata

### Modificare i Pesi

Se vuoi pesi custom, modifica `src/training/losses_universal.py`:

```python
# Crea un nuovo preset personalizzato
weights_custom = {
    "reconstruction": 1.0,
    "color_economy": 1.8,    # ← Aumenta per palette più limitata
    "cluster": 2.5,          # ← Aumenta per no pixel isolati
    "antialiasing": 1.2,     # ← Riduci per permettere un po' di anti-alias
    "outline": 0.8,
    "perceptual": 0.4,
}

criterion = UniversalPixelArtLoss(max_colors=32, weights=weights_custom)
```

---

## 🚀 Workflow Completo (Aggiornato)

```bash
# 1. Scarica e organizza file (come prima)
# 2. Cataloga asset
python run.py catalog

# 3. Build dataset
python run.py build

# 4. NUOVO: Addestra con tutorial rules
python run.py train --preset multi_author_balanced --epochs 100

# 5. Genera
python run.py generate
```

---

## ✅ Checklist

- [ ] Ho scaricato `losses_universal.py`
- [ ] Ho scaricato `quick_train_MODIFIED.py`
- [ ] Ho sostituito il vecchio `quick_train.py`
- [ ] Ho capito quale preset usare per il mio dataset
- [ ] Ho eseguito `python run.py catalog` e `python run.py build`
- [ ] Ho eseguito `python run.py train --preset <mio_preset>`
- [ ] Ho controllato i loss breakdown nel training
- [ ] Ho generato sprite con `python run.py generate`

---

## 🎓 Cosa Sta Succedendo "Sotto il Cofano"

La loss function implementa queste regole tutorial:

### Regola 1: No Pixel Isolati
```
Autore A: pixel singoli = dettagli
Autore B: pixel singoli = rumore
        → Forza cluster minimo 2px
```

### Regola 2: No Anti-Aliasing
```
Autore A: gradienti morbidi = stile
Autore B: gradienti netti = stile
        → Forza transizioni nette
```

### Regola 3: Palette Limitata
```
Autore A: 50 colori = normale
Autore B: 16 colori = normale
        → Forza max 24-32 colori
```

### Regola 4: Contorni Scuri
```
Autore A: contorno nero = suo stile
Autore B: contorno grigio = suo stile
        → Forza contorno nella fascia scura
```

**Risultato:** Ogni autore può mantenere il suo stile, ma le regole strutturali sono coerenti.

---

## 🆘 Troubleshooting

### "Loss seems too high"

Aspettato con tutorial rules. La loss è pesata su più vincoli.

Se scende nel tempo, il training sta funzionando bene.

### "Output looks artificial"

Prova un preset più permissivo:
```bash
python run.py train --preset multi_author_permissive
```

### "Output seems inconsistent"

Prova un preset più stretto:
```bash
python run.py train --preset multi_author_strict
```

### "Loss breakdown non appare"

Assicurati di stare usando la versione MODIFIED di `quick_train.py`.

---

## 📊 Performance Atteso

| Preset | VRAM | Tempo | Output |
|---|---|---|---|
| multi_author_balanced | Normale | +0% | ✅ Coerente |
| multi_author_strict | Normale | +5% | ✅ Molto coerente |
| multi_author_permissive | Normale | +0% | ✅ Stilistico |
| single_author | Normale | -5% | ✅ Naturale |

*Tempi relativi ai pesi della loss, non dalla complessità computazionale.*

---

## 🎯 Prossimi Passi

1. ✅ Setup il nuovo sistema con UniversalPixelArtLoss
2. ✅ Addestra su uno dei tuoi dataset multi-autore
3. ✅ Testa quale preset dà i risultati migliori
4. ✅ Genera sprite a diverse risoluzioni
5. ✅ Integra nel tuo gioco

---

Buon lavoro con il nuovo setup! 🎨

