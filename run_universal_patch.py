#!/home/kozel/miniconda/envs/cnn-benchmark/bin/python
"""
Univerzálny patch (Téza D): JEDNA sada nálepiek/páska natrénovaná cez VIAC
identít naraz, aby fungovala na kohokoľvek — nie len na jednu tvár.

Tréning: ensemble 3 modelov (ArcFace, FaceNet, BenchmarkCNN) + gradient
averaging + MI-FGSM momentum + feature-space loss, multi-image/multi-identity
tréning (náhodná fotka per iteráciu, dodge loss per fotka).
Eval: natrénované patche sa aplikujú na NEVIDENÉ fotky všetkých identít
(apply_patch_set — rovnaká kompozícia ako pri tlači), meria sa per-identita
sim/ASR pre každý trénovací model aj held-out AdaFace.

Príklad:
    python run_universal_patch.py --train-per-id 3 --test-per-id 2 --num-iter 120
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
from attacks.ensemble_hybrid_patch import ensemble_hybrid_patch_attack, apply_patch_set

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "custom_dataset")
SUCCESS_THR = 0.5  # konzistentné s run_transfer_matrix


def load_dataset(train_per_id: int, test_per_id: int):
    """Po identitách: (train_fotky, test_fotky) v tvare (meno, tensor)."""
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize([0.5] * 3, [0.5] * 3)])
    data = {}
    for ident in sorted(os.listdir(DATA_ROOT)):
        id_dir = os.path.join(DATA_ROOT, ident)
        if not os.path.isdir(id_dir):
            continue
        files = sorted(f for f in os.listdir(id_dir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
        train, test = [], []
        for i, f in enumerate(files):
            if len(train) < train_per_id:
                bucket = train
            elif len(test) < test_per_id:
                bucket = test
            else:
                break
            pil = Image.open(os.path.join(id_dir, f)).convert("RGB").resize((112, 112))
            bucket.append((f"{ident}/{f}", tf(pil).unsqueeze(0).to(DEVICE)))
        data[ident] = (train, test)
    return data


def main():
    ap = argparse.ArgumentParser(description="Univerzálny patch cez viac identít")
    ap.add_argument("--train-per-id", type=int, default=3)
    ap.add_argument("--test-per-id", type=int, default=2)
    ap.add_argument("--num-iter", type=int, default=120)
    ap.add_argument("--dot-radius", type=float, default=9.5)
    ap.add_argument("--dot-pattern", type=str, default="cheeks_chin")
    ap.add_argument("--strip-w-scale", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="results/universal_patch_eval.csv")
    ap.add_argument("--sheet", type=str, default="printable_patches/universal_patch_sheet.png")
    args = ap.parse_args()

    print(f"DEVICE={DEVICE} | train_per_id={args.train_per_id} | test_per_id={args.test_per_id} "
          f"| num_iter={args.num_iter}", flush=True)

    all_models = build_all_models(DEVICE)
    missing = [n for n in ("ArcFace", "AdaFace", "FaceNet", "BenchmarkCNN") if n not in all_models]
    if missing:
        raise RuntimeError(f"Chýbajú modely: {missing}")
    ensemble3 = {k: v for k, v in all_models.items() if k != "AdaFace"}
    eval_models = dict(all_models)  # 3 trénovacie + held-out AdaFace

    from models.pad_minifasnet import load_pad_model
    pad_model = load_pad_model(device=DEVICE)

    mtcnn = MTCNN(image_size=112, margin=0, device="cpu", post_process=False)
    data = load_dataset(args.train_per_id, args.test_per_id)
    train_all = [(name, t) for ident in sorted(data) for name, t in data[ident][0]]
    print(f"Tréningové fotky ({len(train_all)}): {[n for n, _ in train_all]}", flush=True)
    for ident, (_, test) in data.items():
        print(f"  test {ident}: {[n for n, _ in test]}", flush=True)

    landmarks = {}
    for name, t in train_all + [(n, t) for _, ts in data.values() for n, t in ts]:
        landmarks[name] = extract_landmarks(t, mtcnn=mtcnn, return_mesh=True)
    print("Landmarky a 3D FaceMesh pripravené.", flush=True)

    os.makedirs(os.path.dirname(args.sheet) or ".", exist_ok=True)

    # --- Tréning univerzálneho patchu (multi-identity) ---
    t0 = time.time()
    torch.manual_seed(args.seed)
    adv, sheet, info = ensemble_hybrid_patch_attack(
        models=ensemble3,
        image=train_all[0][1],
        images=[t for _, t in train_all[1:]],
        num_iter=args.num_iter,
        eot=True,
        grad_avg=True, momentum=0.9, feature_weight=0.6,
        input_diversity=True, update_rule="sign",
        pad_model=pad_model, pad_weight=0.25 if pad_model is not None else 0.0,
        dot_radius=args.dot_radius, dot_pattern=args.dot_pattern,
        strip_w_scale=args.strip_w_scale,
        mtcnn=mtcnn,
        landmarks=landmarks[train_all[0][0]],
        output_path=args.sheet,
        seed=args.seed,
    )
    patches = info["patches"]
    print(f"\nTréning hotový za {time.time() - t0:.0f}s | train mean sim="
          f"{info['final_similarity']:.4f} | occ={info['face_occlusion_pct']:.1f}% "
          f"| per-model={info['per_model_similarity']}", flush=True)

    # --- Eval na NEVIDENÝCH fotkách všetkých identít ---
    rows = []
    print("\n=== EVAL univerzálneho patchu (nevidené fotky) ===", flush=True)
    for ident in sorted(data):
        for name, x in data[ident][1]:
            adv_x, _, ainfo = apply_patch_set(patches, x, landmarks=landmarks[name],
                                             dot_radius=args.dot_radius,
                                             dot_pattern=args.dot_pattern,
                                             strip_w_scale=args.strip_w_scale)
            with torch.no_grad():
                row = {"photo": name, "identity": ident,
                       "occ_pct": round(ainfo["face_occlusion_pct"], 2)}
                sims = {}
                for mname, m in eval_models.items():
                    emb0 = m(x)
                    emb1 = m(adv_x)
                    s = float(torch.nn.functional.cosine_similarity(emb0, emb1).mean())
                    sims[mname] = s
                    row[f"sim_{mname}"] = round(s, 4)
                    row[f"success_{mname}"] = int(s < SUCCESS_THR)
                row["sim_mean_train3"] = round(float(np.mean(
                    [sims[k] for k in ensemble3])), 4)
                row["success_train3_all"] = int(all(sims[k] < SUCCESS_THR for k in ensemble3))
                if pad_model is not None:
                    pad_p = float(pad_model(adv_x).mean().item())
                else:
                    pad_p = 1.0
                row["pad_live_prob"] = round(pad_p, 4)
                row["pad_live_pass"] = int(pad_p >= 0.5)
            rows.append(row)
            print(f"  {name}: sims=" +
                  " ".join(f"{k}={row['sim_' + k]:.3f}" for k in eval_models) +
                  f" | mean_train3={row['sim_mean_train3']:.3f} | PAD={row['pad_live_prob']:.2f} | occ={row['occ_pct']:.1f}%",
                  flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fieldnames = ["photo", "identity", "occ_pct"] + \
        [f"sim_{k}" for k in eval_models] + \
        [f"success_{k}" for k in eval_models] + \
        ["sim_mean_train3", "success_train3_all", "pad_live_prob", "pad_live_pass"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print("\n=== SÚHRN PER IDENTITA (nevidené fotky, thr cos<0.5) ===", flush=True)
    for ident in sorted(data):
        sub = [r for r in rows if r["identity"] == ident]
        line = f"  {ident:22s}"
        for mname in eval_models:
            ms = float(np.mean([r[f"sim_{mname}"] for r in sub]))
            asr = 100.0 * np.mean([r[f"success_{mname}"] for r in sub])
            line += f" | {mname}: sim={ms:.3f} ASR={asr:.0f}%"
        print(line, flush=True)
    all_rows = rows
    print(f"\nCELKOM: mean train3 sim="
          f"{float(np.mean([r['sim_mean_train3'] for r in all_rows])):.4f} | "
          f"ASR(train3 všetky modely)="
          f"{100.0 * np.mean([r['success_train3_all'] for r in all_rows]):.1f}% | "
          f"ASR(AdaFace held-out)="
          f"{100.0 * np.mean([r['success_AdaFace'] for r in all_rows]):.1f}% | "
          f"PAD Live="
          f"{100.0 * np.mean([r['pad_live_pass'] for r in all_rows]):.1f}%", flush=True)
    print(f"CSV: {args.out} | hárok: {args.sheet}", flush=True)


if __name__ == "__main__":
    main()

