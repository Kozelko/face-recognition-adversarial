#!/home/kozel/miniconda/envs/cnn-benchmark/bin/python
"""
Široký sweep fyzických adversariálnych útokov: hľadáme konfiguráciu, ktorá
REÁLNE oklame model (sim < 0.5), nielen páska na nos + 2 nálepky.

Skúša viac rodín a parametrov:
  - hybrid: rôzna šírka pásu (strip_w_scale), rádius nálepiek, vzory (2/3 nálepky),
    plnofarebný RGB vzor aj obmedzená paleta (kinesio 3 farby),
  - constellation: trio / cluster_5 / quad_full s väčším rádiusom,
  - nosová páska: úzka vs široká (width_scale),
  - leukoplast: rôzne veľkosti,
  - ensemble (4 modely, gradient averaging + momentum + feature loss).

Výstup: results/physical_sweep.csv (po riadkoch) + súhrn vytlačený a zoradený
podľa priemernej podobnosti (nižšia = silnejší útok).
"""
import argparse
import csv
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
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

from run_patch_experiments import build_all_models
from attacks.constellation_patch import extract_landmarks
from attacks.hybrid_patch import hybrid_patch_attack
from attacks.constellation_patch import constellation_patch_attack
from attacks.nasal_strip_patch import nasal_strip_attack
from attacks.adversarial_patch import adversarial_patch_attack
from attacks.ensemble_hybrid_patch import ensemble_hybrid_patch_attack

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "custom_dataset")
KINESIO_3 = [[0.05, 0.05, 0.05], [0.85, 0.55, 0.25], [0.85, 0.15, 0.15]]


def load_photos(n):
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize([0.5] * 3, [0.5] * 3)])
    out = []
    for ident in sorted(os.listdir(DATA_ROOT)):
        d = os.path.join(DATA_ROOT, ident)
        if not os.path.isdir(d):
            continue
        fs = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg", ".png", ".jpeg")))
        if not fs:
            continue
        p = os.path.join(d, fs[0])
        out.append((f"{ident}/{fs[0]}", tf(Image.open(p).convert("RGB").resize((112, 112))).unsqueeze(0).to(DEVICE)))
        if len(out) >= n:
            break
    return out


def configs():
    """(label, druh, kwargs) — druh: hybrid|const|nasal|leuko|ens"""
    C = []
    # --- hybrid ---
    C.append(("H_base r4.5 s1.0", "hybrid", dict(dot_radius=4.5, dot_pattern="2_cheeks", strip_w_scale=1.0)))
    C.append(("H_big r8 s1.8", "hybrid", dict(dot_radius=8.0, dot_pattern="2_cheeks", strip_w_scale=1.8)))
    C.append(("H_big3 r8 s1.8 +brada", "hybrid", dict(dot_radius=8.0, dot_pattern="cheeks_chin", strip_w_scale=1.8)))
    C.append(("H_rgb r8 s1.8", "hybrid", dict(dot_radius=8.0, dot_pattern="cheeks_chin", strip_w_scale=1.8, strip_style="rgb")))
    C.append(("H_paleta3 r8 s1.8", "hybrid", dict(dot_radius=8.0, dot_pattern="cheeks_chin", strip_w_scale=1.8, palette=KINESIO_3)))
    C.append(("H_huge r8 s3.0", "hybrid", dict(dot_radius=8.0, dot_pattern="cheeks_chin", strip_w_scale=3.0)))
    C.append(("H_rgb huge s3.0", "hybrid", dict(dot_radius=8.0, dot_pattern="cheeks_chin", strip_w_scale=3.0, strip_style="rgb")))
    # --- constellation ---
    C.append(("C_trio r4", "const", dict(pattern="trio_cheeks_nose", radius=4.0)))
    C.append(("C_cluster5 r6", "const", dict(pattern="cluster_5", radius=6.0)))
    C.append(("C_quad r7", "const", dict(pattern="quad_full", radius=7.0)))
    # --- nosová páska ---
    C.append(("N_uzka", "nasal", dict(style="athlete_black", width_scale=1.0)))
    C.append(("N_siroka x2", "nasal", dict(style="athlete_black", width_scale=2.0)))
    # --- leukoplast ---
    C.append(("L_nos 7x18", "leuko", dict(position="nose_bridge", patch_size=(7, 18))))
    C.append(("L_nos 14x36", "leuko", dict(position="nose_bridge", patch_size=(14, 36))))
    C.append(("L_huge 20x50", "leuko", dict(position="nose_bridge", patch_size=(20, 50))))
    # --- ensemble (nová metóda) ---
    C.append(("E_ga_big", "ens", dict(dot_radius=8.0, dot_pattern="cheeks_chin", strip_w_scale=1.8,
                                      grad_avg=True, momentum=1.0, feature_weight=0.5)))
    return C


def run_one(kind, model, image, landmarks, iters, kwargs):
    cu = {"mtcnn": None, "landmarks": landmarks}
    if kind == "hybrid":
        return hybrid_patch_attack(model=model, image=image, num_iter=iters, eot=True, **kwargs, **cu)
    if kind == "const":
        return constellation_patch_attack(model=model, image=image, num_iter=iters, eot=True, **kwargs, **cu)
    if kind == "nasal":
        return nasal_strip_attack(model=model, image=image, num_iter=iters, eot=True, **kwargs, **cu)
    if kind == "leuko":
        return adversarial_patch_attack(model=model, image=image, num_iter=iters, eot=True, **kwargs, **cu)
    if kind == "ens":
        return ensemble_hybrid_patch_attack(models=model, image=image, num_iter=iters, eot=True, seed=0, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, default="ArcFace,BenchmarkCNN")
    ap.add_argument("--photos", type=int, default=2)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--out", type=str, default="results/physical_sweep.csv")
    args = ap.parse_args()

    all_models = build_all_models(DEVICE)
    wanted_models = [m.strip() for m in args.models.split(",") if m.strip() in all_models]
    photos = load_photos(args.photos)
    lms = {name: extract_landmarks(t, return_mesh=False) for name, t in photos}

    print(f"DEVICE={DEVICE} | modely={wanted_models} | fotky={len(photos)} | iters={args.iters}", flush=True)
    rows = []
    for label, kind, kwargs in configs():
        for mname in wanted_models:
            model = all_models[mname]
            for pname, x in photos:
                t0 = time.time()
                try:
                    out = run_one(kind, (all_models if kind == "ens" else model),
                                  x, lms[pname], args.iters, kwargs)
                    adv = out[0]
                    info = out[2] if len(out) > 2 else {}
                    with torch.no_grad():
                        if kind == "ens":
                            sim = float(np.mean(list(info.get("per_model_similarity", {}).values())))
                        else:
                            sim = float(F.cosine_similarity(model(x), model(adv)).mean())
                    occ = float(info.get("face_occlusion_pct", 0.0))
                    ok = 1 if sim < 0.5 else 0
                except Exception as e:
                    sim, occ, ok = float("nan"), 0.0, 0
                    print(f"  [CHYBA] {label}/{mname}: {e}", flush=True)
                rows.append({"config": label, "kind": kind, "model": mname, "photo": pname,
                             "sim": round(sim, 4), "success": ok, "occ_pct": round(occ, 2),
                             "sec": round(time.time() - t0, 1)})
                flag = "USPECH" if ok else "     "
                print(f"  {flag} [{label:24s}] {mname:12s} {pname[:20]:20s} sim={sim:.3f} occ={occ:.1f}% "
                      f"({time.time() - t0:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "kind", "model", "photo", "sim", "success", "occ_pct", "sec"])
        w.writeheader(); w.writerows(rows)

    print("\n=== SÚHRN (zoradené podľa priemernej sim, nižšia = silnejší útok) ===")
    agg = {}
    for r in rows:
        if np.isnan(r["sim"]):
            continue
        for key in (r["config"], f"{r['config']}|{r['model']}"):
            a = agg.setdefault(key, {"sims": [], "succ": 0, "occ": [], "model": r["model"]})
            a["sims"].append(r["sim"]); a["succ"] += r["success"]; a["occ"].append(r["occ_pct"])
    for key in sorted(agg, key=lambda k: np.mean(agg[k]["sims"])):
        a = agg[key]
        print(f"  {key:40s} mean_sim={np.mean(a['sims']):.3f} ASR={100*a['succ']/len(a['sims']):4.0f}% "
              f"occ={np.mean(a['occ']):.1f}%")
    print(f"CSV: {args.out}", flush=True)


if __name__ == "__main__":
    main()
