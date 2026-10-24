#!/home/kozel/miniconda/envs/cnn-benchmark/bin/python
"""
Held-out cross-model transfer protokol (Téza D).

Tréning patchov na 3 modeloch, test na 4. (held-out) modeli — AdaFace.
Porovnáva single-source baseline proti ensemble variantom:

  A: single-source (ArcFace)                         -> held-out AdaFace
  B: ensemble 3 modelov, súčet relu-loss (baseline)  -> held-out AdaFace
  C: ensemble 3 modelov + gradient averaging
     + MI-FGSM momentum + feature-space loss (nové)  -> held-out AdaFace
  D: C + DI-FGSM input diversity                     -> held-out AdaFace
  E: C s MI-FGSM sign-updatom (namiesto Adam)        -> held-out AdaFace

Používa rovnaký protokol ako run_transfer_matrix (num_iter, eot=True, seed),
aby boli čísla porovnateľné s results/transfer_matrix.csv.

Príklad:
    python run_heldout_experiment.py --configs A,B,C,D --n-images 10 --num-iter 60
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
from PIL import Image
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN

from run_patch_experiments import build_all_models
from attacks.constellation_patch import extract_landmarks
from attacks.ensemble_hybrid_patch import ensemble_hybrid_patch_attack

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "custom_dataset")
HELD_OUT = "AdaFace"
SUCCESS_THR = 0.5  # rovnaká hranica ako run_transfer_matrix (cos < 0.5)


def load_photos(n_images: int):
    """Vyvážený výber fotiek cez všetky identity v data/custom_dataset."""
    identities = sorted(d for d in os.listdir(DATA_ROOT)
                        if os.path.isdir(os.path.join(DATA_ROOT, d)))
    per_id = max(1, n_images // max(1, len(identities)))
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize([0.5] * 3, [0.5] * 3)])
    out = []
    for ident in identities:
        id_dir = os.path.join(DATA_ROOT, ident)
        files = sorted(f for f in os.listdir(id_dir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
        for f in files[:per_id]:
            if len(out) >= n_images:
                break
            pil = Image.open(os.path.join(id_dir, f)).convert("RGB").resize((112, 112))
            out.append((f"{ident}/{f}", tf(pil).unsqueeze(0).to(DEVICE)))
        if len(out) >= n_images:
            break
    return out


def build_configs(ensemble3: dict, arcface, pad_model=None):
    """Konfigurácie útokov (kwargs pre ensemble_hybrid_patch_attack)."""
    base_geo = dict(dot_radius=9.5, dot_pattern="cheeks_chin", strip_w_scale=2.0)
    cfg = {
        "A_single_arcface": ({"ArcFace": arcface}, dict(**base_geo, grad_avg=False, momentum=0.0, feature_weight=0.0)),
        "B_ens3_sum":       (ensemble3, dict(**base_geo, grad_avg=False, momentum=0.0, feature_weight=0.0)),
        "C_ens3_ga_mom_feat": (ensemble3, dict(**base_geo, grad_avg=True, momentum=0.9,
                                               feature_weight=0.6)),
        "D_ens3_ga_di":     (ensemble3, dict(**base_geo, grad_avg=True, momentum=0.9,
                                             feature_weight=0.6, input_diversity=True,
                                             pad_model=pad_model, pad_weight=0.25 if pad_model is not None else 0.0)),
        "E_ens3_ga_sign":   (ensemble3, dict(**base_geo, grad_avg=True, momentum=0.9,
                                             feature_weight=0.6, input_diversity=True, update_rule="sign",
                                             pad_model=pad_model, pad_weight=0.25 if pad_model is not None else 0.0)),
    }
    return cfg


def main():
    ap = argparse.ArgumentParser(description="Held-out transfer: train 3 modely, test AdaFace")
    ap.add_argument("--configs", type=str, default="A,B,C,D,E",
                    help="Čiarkou oddelené kľúče A,B,C,D,E")
    ap.add_argument("--n-images", type=int, default=10)
    ap.add_argument("--num-iter", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="results/heldout_transfer.csv")
    args = ap.parse_args()

    wanted = {c.strip().upper() for c in args.configs.split(",") if c.strip()}
    print(f"DEVICE={DEVICE} | configs={sorted(wanted)} | n_images={args.n_images} "
          f"| num_iter={args.num_iter}", flush=True)

    all_models = build_all_models(DEVICE)
    missing = [n for n in ("ArcFace", "AdaFace", "FaceNet", "BenchmarkCNN") if n not in all_models]
    if missing:
        raise RuntimeError(f"Chýbajú modely: {missing}")
    held_out_model = all_models[HELD_OUT]
    ensemble3 = {k: v for k, v in all_models.items() if k != HELD_OUT}
    from models.pad_minifasnet import load_pad_model
    pad_model = load_pad_model(device=DEVICE)
    print(f"Train modely: {list(ensemble3.keys())} | held-out: {HELD_OUT} | PAD: {pad_model is not None}", flush=True)

    mtcnn = MTCNN(image_size=112, margin=0, device="cpu", post_process=False)
    photos = load_photos(args.n_images)
    print(f"Načítané fotky ({len(photos)}): {[p for p, _ in photos]}", flush=True)

    landmarks_map = {}
    for name, t in photos:
        landmarks_map[name] = extract_landmarks(t, mtcnn=mtcnn, return_mesh=True)
    print("Landmarky a 3D FaceMesh pripravené.", flush=True)

    with torch.no_grad():
        orig_heldout = {name: held_out_model(t).detach() for name, t in photos}

    configs = build_configs(ensemble3, all_models["ArcFace"], pad_model=pad_model)
    rows = []
    for cfg_key in ("A_single_arcface", "B_ens3_sum", "C_ens3_ga_mom_feat",
                    "D_ens3_ga_di", "E_ens3_ga_sign"):
        letter = cfg_key[0]
        if letter not in wanted:
            continue
        models, kwargs = configs[cfg_key]
        for name, x in photos:
            torch.manual_seed(args.seed)
            t0 = time.time()
            adv, _, info = ensemble_hybrid_patch_attack(
                models=models, image=x, num_iter=args.num_iter,
                eot=True, landmarks=landmarks_map[name], seed=args.seed, **kwargs,
            )
            dt = time.time() - t0
            with torch.no_grad():
                held_sim = float(torch.nn.functional.cosine_similarity(
                    orig_heldout[name], held_out_model(adv)).mean())
                pad_live = float(pad_model(adv).mean().item()) if pad_model is not None else float("nan")
            train_sims = info["per_model_similarity"]
            train_mean = float(np.mean(list(train_sims.values()))) if train_sims else float("nan")
            train_asr = float(np.mean([bool(v) for v in info["per_model_success"].values()])) * 100.0
            row = {
                "config": cfg_key,
                "photo": name,
                "heldout_sim": round(held_sim, 4),
                "heldout_success": int(held_sim < SUCCESS_THR),
                "train_mean_sim": round(train_mean, 4),
                "train_asr_pct": round(train_asr, 1),
                "pad_live_prob": round(pad_live, 4),
                "pad_live_pass": int(pad_live >= 0.5) if not np.isnan(pad_live) else 1,
                "occ_pct": info["face_occlusion_pct"],
                "num_iter": args.num_iter,
                "sec": round(dt, 1),
            }
            rows.append(row)
            print(f"  [{cfg_key}] {name}: held-out {HELD_OUT} sim={held_sim:.4f} "
                  f"({'ÚSPECH' if held_sim < SUCCESS_THR else 'neuspech'}) "
                  f"| train={train_mean:.4f} | PAD={pad_live:.3f} | occ={info['face_occlusion_pct']:.1f}% "
                  f"| {dt:.1f}s", flush=True)

            # Inkrementálny zápis po každej fotke
            out_path = args.out
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            fieldnames = ["config", "photo", "heldout_sim", "heldout_success", "train_mean_sim",
                          "train_asr_pct", "pad_live_prob", "pad_live_pass", "occ_pct", "num_iter", "sec"]
            with open(out_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(rows)

    print("\n=== SÚHRN (held-out AdaFace, prah cos<0.5) ===", flush=True)
    for cfg_key in sorted({r["config"] for r in rows}):
        sub = [r for r in rows if r["config"] == cfg_key]
        asr = 100.0 * sum(r["heldout_success"] for r in sub) / max(1, len(sub))
        ms = float(np.mean([r["heldout_sim"] for r in sub]))
        wb = float(np.mean([r["train_mean_sim"] for r in sub]))
        pl = float(np.mean([r["pad_live_prob"] for r in sub]))
        pp = 100.0 * sum(r["pad_live_pass"] for r in sub) / max(1, len(sub))
        print(f"  {cfg_key:24s} ASR={asr:5.1f}%  held-out sim={ms:.4f}  "
              f"white-box sim={wb:.4f}  PAD live={pl:.3f} ({pp:.0f}% LIVE) (n={len(sub)})", flush=True)
    print(f"CSV: {out_path}", flush=True)


if __name__ == "__main__":
    main()
