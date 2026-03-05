# Face Recognition & Adversarial Attacks

Diplomová práca — rozpoznávanie tvárí s CNN a adversariálne útoky.

## Požiadavky

- Python 3.11
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) / Anaconda
- NVIDIA GPU s CUDA 11.8 (voliteľné, ale odporúčané)

## Inštalácia

### 1. Vytvorenie conda prostredia

```bash
conda env create -f environment.yml
conda activate cnn-benchmark
```

### 2. Inštalácia ďalších závislostí (pip)

```bash
pip install facenet-pytorch tqdm
```
## Spustenie

### Predspracovanie datasetu (MTCNN crop)

Pred trénovaním je potrebné detekovať a orezať tváre na 112×112 px:

```bash
python utils/preprocess.py
```

Skript používa multiprocessing — automaticky využije dostupné CPU jadrá.

Voliteľné parametre:

| Parameter    | Popis                                  | Default                |
|-------------|----------------------------------------|------------------------|
| `--device`  | Zariadenie pre MTCNN (`cpu` / `cuda`)  | `cpu`                  |
| `--workers` | Počet paralelných workerov             | počet jadier − 2       |

Príklady:

```bash
# Predvolené nastavenie (CPU, automatický počet workerov)
python utils/preprocess.py

# Vlastný počet workerov
python utils/preprocess.py --workers 8
```

### Overenie modelu

Rýchly test, či SimpleFaceCNN funguje:

```bash
python train.py
```