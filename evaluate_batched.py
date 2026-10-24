#!/home/kozel/miniconda/envs/cnn-benchmark/bin/python
import os
import sys
_CONDA_PY = "/home/kozel/miniconda/envs/cnn-benchmark/bin/python"
if "cnn-benchmark" not in sys.executable and os.path.exists(_CONDA_PY):
    os.execv(_CONDA_PY, [_CONDA_PY] + sys.argv)
import ctypes
try:
    ctypes.CDLL("/home/kozel/miniconda/envs/cnn-benchmark/lib/libpng16.so.16", mode=ctypes.RTLD_GLOBAL)
except Exception:
    pass
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import pandas as pd
import random
import time
import argparse
import json

import numpy as np

from models.wrappers import FaceNetWrapper, BenchmarkCNNWrapper, ArcFaceWrapper, AdaFaceWrapper
from attacks.fgsm import fgsm_attack_untargeted
from attacks.pgd import pgd_attack_untargeted
from attacks.bim import bim_attack_untargeted
from attacks.mifgsm import mifgsm_attack_untargeted
from attacks.cw import cw_l2_attack_untargeted

# --- Nastavenia ---
DATA_DIR = "data/processed/casia"
BATCH_SIZE = 32  # Znížené z 64 na 32 kvôli kapacite VRAM (6GB na RTX 3060) pri väčších modeloch ako ArcFace
NUM_TEST_IMAGES = 2000  # Finálny test pre diplomovku
THRESHOLD = 0.5         # Fallback hranica cosine similarity (použitá ak chýba kalibrácia)
THRESHOLDS_FILE = "results/thresholds.json"  # Per-model kalibrované hranice (FAR=1e-3)
CALIB_FAR = 1e-3        # Cieľová False Accept Rate pre kalibráciu
CALIB_PAIRS = 4000      # Počet párov pre kalibračnú distribúciu
RESULTS_FILE = "results/batched_evaluation.csv"

# Parametre útokov
EPSILON = 8 / 255.0     # Štandard pre L_inf útoky na tváre/ImageNet
ALPHA = 2 / 255.0       # Štandardný krok
NUM_ITER = 20           # Štandard pre PGD-20 / BIM / MI-FGSM

def load_benchmark_cnn(device):
    checkpoint_path = "models/checkpoints/benchmark_cnn_best_run1.pth"
    if not os.path.exists(checkpoint_path):
        return None
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Snažíme sa nájsť počet tried z datasetu, ak zlyhá, dáme 10575 (casia default)
        try:
            num_classes = len(datasets.ImageFolder(DATA_DIR).classes)
        except:
            num_classes = 10575
        model = BenchmarkCNNWrapper(num_classes=num_classes, checkpoint_path=checkpoint_path, device=device)
        return model
    except Exception as e:
        print(f"Chyba pri načítaní BenchmarkCNN: {e}")
        return None


def build_model_creators(device):
    """Modely (embedding) - rovnaké poradie/menný priestor ako v main()."""
    return {
        "FaceNet": lambda: FaceNetWrapper(device=device),
        "ArcFace": lambda: ArcFaceWrapper(device=device),
        "AdaFace": lambda: AdaFaceWrapper(device=device),
        "BenchmarkCNN": lambda: load_benchmark_cnn(device),
    }


def _load_pairs_from_lfw(data_dir, num_pairs, seed=0):
    """
    Skúsi nájsť LFW-style páry (pairs.txt / pairs.csv) pod data/. Vráti zoznam
    (path1, path2, is_genuine) alebo None ak nenájde.
    pairs.csv formát (repo export): header `name,imagenum1,imagenum2` = genuine
    páry; impostor páry sa dovygenerujú medzi rôznymi identitami.
    """
    candidates = [
        os.path.join(data_dir, "pairs.txt"),
        os.path.join(data_dir, "pairs.csv"),
        "data/lfw/pairs.txt",
        "data/lfw_pairs/pairs.txt",
        "data/raw/lfw/pairs.txt",
        "data/raw/lfw/pairs.csv",
    ]
    pairs_path = next((p for p in candidates if os.path.exists(p)), None)
    if pairs_path is None:
        return None
    root = os.path.dirname(pairs_path)
    deep = os.path.join(root, "lfw-deepfunneled")
    if os.path.isdir(deep):
        root = deep

    def _img(name, idx):
        return os.path.join(root, name, f"{name}_{int(idx):04d}.jpg")

    rng = random.Random(seed)
    pairs = []
    try:
        with open(pairs_path) as f:
            lines = f.read().strip().splitlines()
        if pairs_path.endswith(".csv"):
            rows = [ln.strip().split(",") for ln in lines if ln.strip()]
            if rows and rows[0][0].strip().lower() == "name":
                rows = rows[1:]
            names = sorted({r[0].strip() for r in rows if len(r) >= 3})
            for r in rows:
                if len(r) < 3:
                    continue
                pairs.append((_img(r[0].strip(), r[1]), _img(r[0].strip(), r[2]), True))
            # Impostor páry: rovnaký počet, náhodné dvojice rôznych identít
            file_cache = {}

            def _sample_img(name):
                if name not in file_cache:
                    d = os.path.join(root, name)
                    file_cache[name] = sorted(os.listdir(d)) if os.path.isdir(d) else []
                files = file_cache[name]
                return os.path.join(root, name, rng.choice(files)) if files else ""

            n_genuine = len(pairs)
            while sum(1 for p in pairs if not p[2]) < n_genuine and len(names) >= 2:
                n1, n2 = rng.sample(names, 2)
                p1, p2 = _sample_img(n1), _sample_img(n2)
                if p1 and p2:
                    pairs.append((p1, p2, False))
        else:
            for line in lines[1:] if lines and not lines[0].split()[0].isdigit() else lines:
                parts = line.split()
                if len(parts) == 3:  # genuine: name idx1 idx2
                    p1, p2 = _img(parts[0], parts[1]), _img(parts[0], parts[2])
                    pairs.append((p1, p2, True))
                elif len(parts) == 4:  # impostor: name1 idx1 name2 idx2
                    p1, p2 = _img(parts[0], parts[1]), _img(parts[2], parts[3])
                    pairs.append((p1, p2, False))
    except Exception:
        return None

    pairs = [p for p in pairs if p[0] and p[1] and os.path.exists(p[0]) and os.path.exists(p[1])]
    if not pairs:
        return None
    rng.shuffle(pairs)
    return pairs[:num_pairs]


def _dataset_pairs(dataset, num_pairs, seed=0):
    """Vytvorí (idx1, idx2, is_genuine) páry z datasetu (dataset pairs)."""
    rng = random.Random(seed)
    by_label = {}
    for i, (_, label) in enumerate(dataset.samples):
        by_label.setdefault(label, []).append(i)
    labels = list(by_label.keys())
    pairs = []
    guard = 0
    while len(pairs) < num_pairs and guard < num_pairs * 20:
        guard += 1
        if rng.random() < 0.5 and len(labels) >= 1:
            lab = rng.choice(labels)
            if len(by_label[lab]) >= 2:
                a, b = rng.sample(by_label[lab], 2)
                pairs.append((a, b, True))
        else:
            if len(labels) >= 2:
                l1, l2 = rng.sample(labels, 2)
                a = rng.choice(by_label[l1])
                b = rng.choice(by_label[l2])
                pairs.append((a, b, False))
    return pairs


def calibrate_thresholds(device, model_creators, data_dir=DATA_DIR,
                         far=CALIB_FAR, num_pairs=CALIB_PAIRS):
    """
    Per-model kalibrácia: cosine threshold pri FAR=1e-3 z distribúcie
    genuine/impostor podobností (LFW pairs ak sú k dispozícii, inak dataset pairs).
    """
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    lfw_transform = transforms.Compose([
        transforms.CenterCrop(140),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    lfw = _load_pairs_from_lfw(data_dir, num_pairs)
    thresholds = {}

    for model_name, creator in model_creators.items():
        model = creator()
        if model is None:
            print(f"[calib] Model {model_name} sa nepodarilo načítať, preskakujem.")
            continue
        model.eval()
        est = getattr(model, "embedding_size", 512)

        genuine, impostor = [], []
        with torch.no_grad():
            if lfw is not None:
                from PIL import Image as _Image
                for p1, p2, is_gen in lfw:
                    im1 = _Image.open(p1).convert("RGB")
                    im2 = _Image.open(p2).convert("RGB")
                    tf_use = lfw_transform if im1.size[0] >= 200 else transform
                    t1 = tf_use(im1).unsqueeze(0).to(device)
                    t2 = tf_use(im2).unsqueeze(0).to(device)
                    s = F.cosine_similarity(model(t1), model(t2)).item()
                    (genuine if is_gen else impostor).append(s)
            else:
                ds = datasets.ImageFolder(data_dir, transform=transform)
                pairs = _dataset_pairs(ds, num_pairs)
                for a, b, is_gen in pairs:
                    t1 = ds[a][0].unsqueeze(0).to(device)
                    t2 = ds[b][0].unsqueeze(0).to(device)
                    s = F.cosine_similarity(model(t1), model(t2)).item()
                    (genuine if is_gen else impostor).append(s)

        imp = np.sort(np.array(impostor))[::-1]  # zostupne
        if len(imp) == 0:
            print(f"[calib] {model_name}: žiadne impostor páry, fallback {THRESHOLD}.")
            thresholds[model_name] = THRESHOLD
            continue
        idx = min(int(far * len(imp)), len(imp) - 1)
        thr = float(imp[idx])
        fnr = float(np.mean(np.array(genuine) < thr)) if genuine else float("nan")
        thresholds[model_name] = round(thr, 4)
        print(f"[calib] {model_name}: thr@FAR={far:g} = {thr:.4f} "
              f"(genuine mean={np.mean(genuine) if genuine else float('nan'):.4f}, "
              f"impostor mean={np.mean(imp):.4f}, FNR={fnr:.4f})")
        del model
        torch.cuda.empty_cache()

    return thresholds


def save_thresholds(thresholds, path=THRESHOLDS_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"[calib] Thresholds uložené do {path}")


def load_thresholds(path=THRESHOLDS_FILE):
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def main_hybrid(device, num_images=100, num_iters=40):
    """
    Hybrid patch eval na CASIA/LFW obrázkoch (per-image optimalizácia patcha
    s EoT). ASR = podiel obrázkov s cos(orig, adv) < kalibrovaný per-model
    threshold (results/thresholds.json, fallback THRESHOLD).
    Výsledky sa pripoja do RESULTS_FILE (rovnaký stĺpcový formát).
    """
    from attacks.hybrid_patch import hybrid_patch_attack

    print(f"\n=== HYBRID PATCH EVAL ({num_images} obrázkov, {num_iters} iter/obr) ===")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    total = len(full_dataset)
    indices = random.sample(range(total), min(num_images, total))
    thresholds = load_thresholds(THRESHOLDS_FILE)
    model_creators = build_model_creators(device)
    results = []
    if os.path.exists(RESULTS_FILE):
        try:
            results = pd.read_csv(RESULTS_FILE).to_dict("records")
        except Exception:
            results = []

    for model_name, creator in model_creators.items():
        model = creator()
        if model is None:
            print(f"Model {model_name} sa nepodarilo načítať. Preskakujem.")
            continue
        model.eval()
        threshold = float(thresholds.get(model_name, THRESHOLD))
        sims, l2s, linfs = [], [], []
        successes = 0
        t_start = time.time()
        for i, idx in enumerate(indices):
            x, _ = full_dataset[idx]
            x = x.unsqueeze(0).to(device)
            t0 = time.time()
            adv, _, info = hybrid_patch_attack(model=model, image=x,
                                               num_iter=num_iters, eot=True)
            with torch.no_grad():
                sim = float(F.cosine_similarity(model(x), model(adv)).mean())
                pert = (adv - x).view(-1)
                l2s.append(float(torch.norm(pert, p=2)))
                linfs.append(float(torch.norm(pert, p=float("inf"))))
            sims.append(sim)
            successes += int(sim < threshold)
            if (i + 1) % 10 == 0:
                print(f"  [{model_name}] {i + 1}/{len(indices)} obrázkov "
                      f"(posl. sim={sim:.4f}, {time.time() - t0:.1f}s/obr)", flush=True)
        elapsed = time.time() - t_start
        n = len(indices)
        success_rate = 100.0 * successes / max(1, n)
        row = {
            "Model": model_name,
            "Attack": "HybridPatch",
            "Num_Images": n,
            "Batch_Size": 1,
            "Epsilon": "n/a (fyzický patch)",
            "Num_Iter": num_iters,
            "Threshold": round(threshold, 4),
            "Success_Rate_pct": round(success_rate, 2),
            "Mean_Similarity": round(float(np.mean(sims)), 4),
            "Mean_L2_Noise": round(float(np.mean(l2s)), 4),
            "Mean_Linf_Noise": round(float(np.mean(linfs)), 4),
            "Time_Seconds": round(elapsed, 2),
            "Time_Per_Image_ms": round(1000.0 * elapsed / max(1, n), 2),
        }
        results.append(row)
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        pd.DataFrame(results).to_csv(RESULTS_FILE, index=False)
        print(f"-> [{model_name}] HybridPatch ASR: {success_rate:.2f}% "
              f"(sim < {threshold:.4f}) | mean sim={row['Mean_Similarity']:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()

    print(f"\n✅ Hybrid eval dokončený. Výsledky: {RESULTS_FILE}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Používam zariadenie: {device}")

    # 1. Príprava datasetu a Dataloaderu
    print("Načítavam dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    
    # Náhodný výber podmnožiny
    total_images = len(full_dataset)
    indices = random.sample(range(total_images), min(NUM_TEST_IMAGES, total_images))
    test_subset = Subset(full_dataset, indices)
    
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Veľkosť testovacej sady: {len(test_subset)} obrázkov (Batch size: {BATCH_SIZE})")

    # 2. Definícia modelov (Inicializujú sa postupne kvôli VRAM)
    model_creators = build_model_creators(device)

    # 2b. Per-model kalibrované hranice (FAR=1e-3) - inak fallback THRESHOLD
    thresholds = load_thresholds(THRESHOLDS_FILE)
    if thresholds:
        print(f"Načítané kalibrované thresholds z {THRESHOLDS_FILE}: {thresholds}")
    else:
        print(f"Žiadny {THRESHOLDS_FILE} - používam fallback THRESHOLD={THRESHOLD} "
              f"(spusti `--calibrate` pre kalibráciu).")

    # 3. Definícia útokov
    attacks = {
        "FGSM": lambda m, img: fgsm_attack_untargeted(m, img, epsilon=EPSILON),
        "PGD": lambda m, img: pgd_attack_untargeted(m, img, epsilon=EPSILON, alpha=ALPHA, num_iter=NUM_ITER),
        "BIM": lambda m, img: bim_attack_untargeted(m, img, epsilon=EPSILON, alpha=ALPHA, num_iter=NUM_ITER),
        "MI-FGSM": lambda m, img: mifgsm_attack_untargeted(m, img, epsilon=EPSILON, alpha=ALPHA, num_iter=NUM_ITER),
        "C&W": lambda m, img: cw_l2_attack_untargeted(m, img, max_iter=100) # 100 iterácií by malo s novými parametrami stačiť
    }

    # Príprava na ukladanie výsledkov
    results = []
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

    # 4. Samotná evaluácia po modeloch
    for model_name, creator in model_creators.items():
        print(f"\n{'='*50}")
        print(f"=== Inicializujem model: {model_name} ===")
        model = creator()
        
        if model is None:
            print(f"Model {model_name} sa nepodarilo načítať. Preskakujem.")
            continue
            
        # Prepni model do eval režimu
        model.eval()

        # Kalibrovaná hranica pre tento model
        threshold = float(thresholds.get(model_name, THRESHOLD))

        for attack_name, attack_fn in attacks.items():
            print(f"\n--- Spúšťam útok: {attack_name} na modeli {model_name} ---")
            start_time = time.time()
            
            total_images_processed = 0
            successful_attacks = 0  # Počet obrázkov, kde similarity klesla pod THRESHOLD
            avg_similarity_drop = 0.0
            avg_l2_norm = 0.0
            avg_linf_norm = 0.0

            for batch_idx, (images, labels) in enumerate(test_loader):
                images = images.to(device)
                
                # 1. Získaj pôvodný embedding
                with torch.no_grad():
                    orig_embs = model(images)
                
                # 2. Vytvor adversariálne obrázky
                adv_images = attack_fn(model, images)
                
                # 3. Získaj adversariálne embeddingy
                with torch.no_grad():
                    adv_embs = model(adv_images)
                
                # 4. Vypočítaj Cosine Similarity pre celý batch
                similarities = F.cosine_similarity(orig_embs, adv_embs)
                
                # Výpočet veľkosti šumu (perturbácie)
                perturbation = adv_images - images
                # L2 norma (celková energia šumu) - prepočítaná na batch
                l2 = torch.norm(perturbation.view(perturbation.size(0), -1), p=2, dim=1).sum().item()
                # L_inf norma (maximálna zmena pixelu)
                linf = torch.norm(perturbation.view(perturbation.size(0), -1), p=float('inf'), dim=1).sum().item()
                
                # 5. Vyhodnoť úspešnosť v batchi (kalibrovaný per-model threshold)
                batch_success = (similarities < threshold).sum().item()
                successful_attacks += batch_success
                avg_similarity_drop += similarities.sum().item()
                avg_l2_norm += l2
                avg_linf_norm += linf
                total_images_processed += len(images)
                
                # Progres v konzole
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Spracovaných dávok: {batch_idx + 1}/{len(test_loader)}")

            end_time = time.time()
            elapsed = end_time - start_time
            
            # Výpočet finálnych metrík
            success_rate = (successful_attacks / total_images_processed) * 100
            mean_sim = avg_similarity_drop / total_images_processed
            mean_l2 = avg_l2_norm / total_images_processed
            mean_linf = avg_linf_norm / total_images_processed
            time_per_image = (elapsed / total_images_processed) * 1000 # v milisekundách
            
            print(f"-> Úspešnosť útoku (Success Rate): {success_rate:.2f}% (Similarity < {threshold:.4f})")
            print(f"-> Priemerná Cosine Sim. po útoku: {mean_sim:.4f}")
            print(f"-> Priemerný L2 šum: {mean_l2:.4f}, L_inf šum: {mean_linf:.4f}")
            print(f"-> Trvanie: {elapsed:.2f} s ({time_per_image:.2f} ms / obrázok)")

            # Ulož výsledok
            results.append({
                "Model": model_name,
                "Attack": attack_name,
                "Num_Images": total_images_processed,
                "Batch_Size": BATCH_SIZE,
                "Epsilon": EPSILON,
                "Num_Iter": NUM_ITER if attack_name != "FGSM" else 1,
                "Threshold": round(threshold, 4),
                "Success_Rate_pct": round(success_rate, 2),
                "Mean_Similarity": round(mean_sim, 4),
                "Mean_L2_Noise": round(mean_l2, 4),
                "Mean_Linf_Noise": round(mean_linf, 4),
                "Time_Seconds": round(elapsed, 2),
                "Time_Per_Image_ms": round(time_per_image, 2)
            })
            
            # Priebežne ukladaj do CSV
            df = pd.DataFrame(results)
            df.to_csv(RESULTS_FILE, index=False)

        # Uvoľni pamäť GPU po každom modeli
        del model
        torch.cuda.empty_cache()

    print(f"\n✅ Evaluácia dokončená. Výsledky uložené do {RESULTS_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch evaluácia útokov (s per-model kalibráciou thresholdov).")
    parser.add_argument("--calibrate", action="store_true",
                        help="Vypočítaj per-model cosine threshold @ FAR=1e-3 a ulož do results/thresholds.json")
    parser.add_argument("--far", type=float, default=CALIB_FAR, help="Cieľová False Accept Rate (default 1e-3)")
    parser.add_argument("--calib-pairs", type=int, default=CALIB_PAIRS, help="Počet párov pre kalibráciu")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="Koreň datasetu (alebo LFW s pairs.txt)")
    parser.add_argument("--thresholds-file", type=str, default=THRESHOLDS_FILE, help="Cesta k thresholds.json")
    parser.add_argument("--hybrid", action="store_true",
                        help="Namiesto digitálnych útokov spusti hybrid patch eval (per-image)")
    parser.add_argument("--hybrid-n", type=int, default=100, help="Počet obrázkov pre hybrid eval")
    parser.add_argument("--hybrid-iters", type=int, default=40, help="Iterácií optimalizácie patcha")
    args = parser.parse_args()

    if args.calibrate:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[calib] Zariadenie: {_device}")
        _th = calibrate_thresholds(_device, build_model_creators(_device),
                                   data_dir=args.data_dir, far=args.far, num_pairs=args.calib_pairs)
        save_thresholds(_th, args.thresholds_file)
    elif args.hybrid:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        main_hybrid(_device, num_images=args.hybrid_n, num_iters=args.hybrid_iters)
    else:
        main()
