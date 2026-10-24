# Face Recognition & Adversarial Attacks Platform

Tento repozitár obsahuje kompletnú platformu pre trénovanie, testovanie a vyhodnocovanie modelov tvárovej biometrie (Face Recognition) voči digitálnym a fyzickým adversariálnym útokom. Kód vznikol ako praktická súčasť diplomovej práce.

## Obsah
- [Architektúra projektu](#architektúra-projektu)
- [Fyzické útoky a tlač vzorov](#fyzické-útoky-a-tlač-vzorov)
- [Inštalácia](#inštalácia)
- [Návod na použitie](#návod-na-použitie)
  - [1. Interaktívna GUI Platforma (Gradio)](#1-interaktívna-gui-platforma)
  - [2. Hromadné testovanie útokov (Batched Evaluation)](#2-hromadné-testovanie-útokov)
  - [3. Predspracovanie dát (MTCNN)](#3-predspracovanie-dát)
  - [4. Trénovanie vlastného modelu](#4-trénovanie-vlastného-modelu)
- [Štruktúra repozitára](#štruktúra-repozitára)

---

## Architektúra projektu

Projekt je navrhnutý modulárne, aby umožňoval jednoduché pridávanie nových modelov a útokov:

### Podporované modely
Všetky modely sú integrované cez jednotné rozhranie `FaceModelWrapper` v `models/wrappers.py`, ktoré zabezpečuje, že výstupom je vždy $L_2$-normalizovaný embedding.
- **BenchmarkCNN:** Vlastný referenčný model postavený od nuly (4x ConvBlock, PReLU, AdaptiveAvgPool2d).
- **FaceNet:** SOTA model (InceptionResnetV1) trénovaný pomocou Triplet Loss.
- **ArcFace:** SOTA model (IResNet50) využívajúci Additive Angular Margin Loss.
- **AdaFace:** SOTA model (IResNet50) prispôsobujúci margin na základe kvality obrazu.

### Implementované adversariálne útoky
Súbory sa nachádzajú v zložke `attacks/`. Útoky sú optimalizované pre tvárovú biometriu a snažia sa maximalizovať kosínusovú vzdialenosť embeddingov.
- **FGSM:** Jednokrokový útok (Fast Gradient Sign Method) - `fgsm.py`.
- **PGD:** Iteračný útok s projekciou (Projected Gradient Descent) - `pgd.py`.
- **BIM:** Základná iteračná metóda (Basic Iterative Method) - `bim.py`.
- **MI-FGSM:** Iteračný útok s využitím momentovej stabilizácie gradientov - `mifgsm.py`.
- **C&W (L2):** Špičkový optimalizačný útok (Carlini & Wagner) kalkulovaný v `arctanh` priestore pre minimalizáciu vizuálneho šumu - `cw.py`.

**Fyzické (tlačiteľné) útoky** – simulujú reálny nosič (náplasť/nálepka) vytlačený na papieri a nalepený na tvár:
- **Športová páska na nos (Breathe Right):** `nasal_strip_patch.py`, parametre `style` (čierna/karbón/telová), `width_scale`.
- **Constellation Patch (Pimple Patches):** `constellation_patch.py`, vzory `trio_cheeks_nose`, `quad_full`, `cluster_5`, `asymmetric_random`, `star_patches`, `adversarial_freckles`; polohy viazané na 3D landmarky.
- **Adversariálny leukoplast:** `adversarial_patch.py`, pozícia (`nose_bridge`, `cheek`, `forehead`) a veľkosť.
- **Hybridný útok (páska + nálepky):** `hybrid_patch.py` – kombinuje nosovú pásku a hydrokoloidné nálepky. Obsahuje fyzikálnu EoT pipeline (perspektívna homografia, kamerová/JPEG simulácia, osvetlenie pokožky), vysokorozlíšené textúry (96×256 / 96×96), režim obmedzenej palety/logo (`palette=`, `logo_mask=`), kompenzáciu sklonu povrchu a generovanie tlačového A4 hárku (`generate_printable_sheet`, mierka 1:1 z IPD).
- **Ensemble hybrid (Téza D):** `ensemble_hybrid_patch.py` – hybridný útok súčasne na viac modelov s voľbou agregácie gradientov (`grad_avg`), MI-FGSM momentum, DI-FGSM input diversity a feature-space stratou (`feature_weight`), plus voliteľný PAD hook (`pad_model`).

### Detekcia živosti (PAD)
- **MiniFASNetV2:** `models/pad_minifasnet.py` (+ `models/minifasnet_arch.py`) – prezentuje `live` pravdepodobnosť (`load_pad_model()`), slúži na overenie, či tlačený nosič obíde liveness detektor. `forward` je diferencovateľný voči vstupu (váhy zmrazené), takže sa dá použiť ako penalizácia v útoku.

---

## Fyzické útoky a tlač vzorov

Tlačiteľné útoky sa dajú spustiť dvoma spôsobmi:

1. **Interaktívne v Gradio appke** (`python app.py`): v Záložke 1 vyber typ útoku (Hybrid, Constellation, Športová páska, Leukoplast), nastav polohu myšou (drag & drop) aj jemným posuvníkom, zvoľ rádius/iterácie a stlač **Spustiť útok**. Vpravo sa zobrazí tvár po útoku a **tlačový hárok A4 (100 % mierka)**.
2. **Dávkovo:** `python run_patch_experiments.py --series 3,4,T` spustí SÉRIU 3 (hybrid na jednotlivých modeloch), SÉRIU 4 (ensemble) a transfer maticu; výstupy sú v `results/patch_eval/` a `results/transfer_matrix.csv`.

**Tlač vzorov (A4 hárok):**
- Hárok sa generuje v presnej mierke **1:1 (100 %)**; over ju pravitkom podľa kalibračného meradla 50 mm na hárku.
- Mierka mm/px sa odvodí z interpupilárnej vzdialenosti (IPD ≈ 63 mm) tváre z fotky.
- **Nálepky sú na hárku nakreslené ako rovné kruhy** (perspektívne skrátenie sa kompenzuje), aby sa dali vystrihnúť a nalepiť; obrys (alpha) zodpovedá skutočnému tvaru nálepky.
- Tlač na matný samolepiaci papier, profil farieb sRGB, **bez** škálovania „Fit to page“.
- Exportovaný hárok sa vždy uloží aj do `printable_patches/printable_patch_sheet_last.png`.

**Kalibrácia prahu a hromadná evaluácia:**
```bash
# per-model cosine threshold @ FAR=1e-3 (používá LFW páry, fallback na CASIA)
python evaluate_batched.py --calibrate
# hromadná evaluácia digitálnych útokov (2000 snímok) -> results/batched_evaluation.csv
python evaluate_batched.py
# hybrid patch evaluácia (per-image) -> pripíše do results/batched_evaluation.csv
python evaluate_batched.py --hybrid --hybrid-n 100 --hybrid-iters 40
```

---

## Inštalácia

### 1. Požiadavky
- Python 3.11
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) / Anaconda
- NVIDIA GPU s CUDA 11.8+ (silne odporúčané pre C&W útok a hromadné testovanie)

### 2. Vytvorenie prostredia
```bash
conda env create -f environment.yml
conda activate cnn-benchmark
```

### 3. Inštalácia doplnkových knižníc
```bash
# Inštalácia PyTorch balíkov s ignorovaním conda dependencies pre zamedzenie konfliktov
pip install --no-deps facenet-pytorch tqdm
# Inštalácia GUI frameworku a manipulácie s dátami
pip install gradio pandas opencv-python
```

---

## Návod na použitie

### 1. Interaktívna GUI Platforma
Aplikácia umožňuje vizuálne testovanie útokov a rýchly zber vlastného datasetu cez webkameru.
```bash
python app.py
```
Po spustení sa vypíše lokálna URL adresa (napr. `http://127.0.0.1:7860`), ktorú stačí otvoriť v prehliadači.
**Funkcie aplikácie:**
- **Záložka 1 (Testovanie útokov):** Zhotovenie/nahratie fotky, výber modelu, výber útoku (PGD, C&W...), nastavenie $Epsilon$ a vizualizácia pridávaného šumu a výslednej zraniteľnosti.
- **Záložka 2 (Zber dát a Finetuning):** Vytváranie custom datasetu. Využíva integrovaný `MTCNN` pre okamžitú detekciu a orezanie tváre z webkamery. Umožňuje jedným klikom priamo spustiť Transfer Learning (`train_finetune.py`) na BenchmarkCNN modeli pre okamžité pridanie nových identít.

### 2. Hromadné testovanie útokov
Slúži na masívnu akademickú evaluáciu odolnosti modelov na veľkej vzorke dát (napr. 2000 obrázkov).
```bash
python evaluate_batched.py
```
- **Priebeh:** Skript automaticky načíta testovaciu sadu, na každom modeli (aby sa šetrila VRAM) postupne spustí všetkých 5 útokov v dávkach (batch size = 32).
- **Výstup:** Vygeneruje sa podrobný report `results/batched_evaluation.csv`, ktorý obsahuje metriky:
  - Attack Success Rate (%)
  - Priemerná podobnosť
  - Priemerný $L_2$ šum (energetická náročnosť útoku)
  - Priemerný $L_\infty$ šum
  - Výpočtový čas potrebný na jeden obrázok (ms)

### 3. Predspracovanie dát (MTCNN crop)
Pokiaľ pracujete s nespracovaným datasetom (napr. LFW, CASIA v raw formáte), tento skript automaticky pomocou MTCNN deteguje, zarovná a oreže tváre na rozmer $112 \times 112$ pixelov pomocou multiprocessingu.
```bash
# Predvolené nastavenie (CPU, automatický počet workerov)
python utils/preprocess.py

# Vlastný počet workerov
python utils/preprocess.py --workers 8
```

### 4. Trénovanie vlastného modelu
Pre spustenie tréningu referenčného modelu `BenchmarkCNN` od nuly:
```bash
python train.py
```
Tréning podporuje *Mixed Precision* (zvyšuje rýchlosť na GPU), *Cosine Annealing* a ukladá logy do zložky `results/`. Najlepšie váhy sa uložia do `models/checkpoints/benchmark_cnn_best.pth`.

---

## Štruktúra repozitára

```text
face-recognition-adversarial/
├── app.py                     # Gradio GUI platforma
├── evaluate_batched.py        # Skript pre hromadné testovanie útokov a metrík
├── evaluate_attacks.py        # Pôvodný hodnotiaci a vizualizačný skript
├── train.py                   # Trénovací skript pre BenchmarkCNN
├── train_finetune.py          # Skript pre Transfer Learning (dotrénovanie)
│
├── attacks/                   # Implementácia útokov
│   ├── fgsm.py
│   ├── pgd.py
│   ├── bim.py
│   ├── mifgsm.py
│   ├── cw.py                  # Carlini & Wagner optimalizovaný pre embeddingy
│   ├── nasal_strip_patch.py   # Športová páska na nos (Breathe Right)
│   ├── constellation_patch.py # Konštelácia mikro-nálepiek + 3D geometria povrchu
│   ├── adversarial_patch.py   # Adversariálny leukoplast
│   ├── hybrid_patch.py        # Hybridný útok (páska + nálepky), EoT, tlačový hárok
│   └── ensemble_hybrid_patch.py  # Ensemble + gradient averaging + PAD hook
│
├── models/                    # Architektúry sietí a wrappery
│   ├── benchmark_cnn.py       # Baseline Vlastný model
│   ├── iresnet.py             # SOTA architektúra (pre ArcFace)
│   ├── adaface_net.py         # SOTA architektúra (pre AdaFace)
│   ├── minifasnet_arch.py     # Architektúra PAD detektora
│   ├── pad_minifasnet.py      # MiniFASNetV2 liveness (PAD)
│   ├── wrappers.py            # Zjednotené rozhranie pre modely (FaceModelWrapper)
│   └── checkpoints/           # Predtrénované váhy modelov (.pth)
│
├── run_patch_experiments.py   # Fyzické útoky: SÉRIA 1-4 + transfer matica (--series)
├── run_heldout_experiment.py  # Held-out transfer protokol (train 3 modely, test 4.)
├── run_universal_patch.py     # Univerzálny patch cez viac identít
├── run_physical_sweep.py      # Široký sweep fyzických útokov a parametrov
│
├── utils/
│   ├── preprocess.py          # Skript na hromadné MTCNN spracovanie
│   └── visualize.py
│
├── data/                      # Priečinky pre datasety (ignorované gitom)
│   ├── raw/
│   ├── processed/
│   └── custom_dataset/        # Dáta zozbierané cez Gradio
│
├── results/                   # CSV reporty a vygenerované obrázky z testov
└── vintage-fiit-thesis/       # Dokumentácia a text diplomovej práce v jazyku Typst
```