# Face Recognition & Adversarial Attacks Platform

Tento repozitár obsahuje kompletnú platformu pre trénovanie, testovanie a vyhodnocovanie modelov tvárovej biometrie (Face Recognition) voči digitálnym a fyzickým adversariálnym útokom. Kód vznikol ako praktická súčasť diplomovej práce.

## Obsah
- [Architektúra projektu](#architektúra-projektu)
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
│   └── cw.py                  # Carlini & Wagner optimalizovaný pre embeddingy
│
├── models/                    # Architektúry sietí a wrappery
│   ├── benchmark_cnn.py       # Baseline Vlastný model
│   ├── iresnet.py             # SOTA architektúra (pre ArcFace)
│   ├── adaface_net.py         # SOTA architektúra (pre AdaFace)
│   ├── wrappers.py            # Zjednotené rozhranie pre modely (FaceModelWrapper)
│   └── checkpoints/           # Predtrénované váhy modelov (.pth)
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