#!/home/kozel/miniconda/envs/cnn-benchmark/bin/python
"""
Vyhodnotenie odolnosti útokov voči vstupným obranným mechanizmom (DP III):
Porovnanie:
  1. Digitálny PGD (eps=8/255, 20 iterácií na ArcFace)
  2. Single-Model Physical Patch (ArcFace)
  3. BioMesh-SpecPatch (Ensemble 3 + GA + DI + Sign + PAD)

Testované obrany (Input Transformation Defenses):
  - None (Bez obrany)
  - JPEG_Q50 (JPEG kompresia kvalita 50)
  - GaussianBlur (Gaussovo rozmazanie 3x3, sigma=1.0)
  - BitDepth_4bit (Feature Squeezing / kvantizácia na 4 bity na kanál)
  - MedianFilter_3x3 (Mediánový filter 3x3 proti lokálnym perturbáciám)
"""

import argparse
import csv
import io
import json
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

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN

from run_patch_experiments import build_all_models
from attacks.constellation_patch import extract_landmarks
from attacks.ensemble_hybrid_patch import ensemble_hybrid_patch_attack
from attacks.pgd import pgd_attack_untargeted

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "custom_dataset")
THR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "thresholds.json")


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    arr = ((x.squeeze(0).detach().cpu().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def pil_to_tensor(pil: Image.Image, device) -> torch.Tensor:
    arr = np.asarray(pil, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def apply_defense(x: torch.Tensor, defense_name: str) -> torch.Tensor:
    if defense_name == "None":
        return x
    elif defense_name == "JPEG_Q50":
        pil = tensor_to_pil(x)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=50)
        buf.seek(0)
        pil_jpeg = Image.open(buf).convert("RGB")
        return pil_to_tensor(pil_jpeg, x.device)
    elif defense_name == "GaussianBlur":
        pil = tensor_to_pil(x).filter(ImageFilter.GaussianBlur(radius=1.0))
        return pil_to_tensor(pil, x.device)
    elif defense_name == "BitDepth_4bit":
        # Kvantizácia z [-1, 1] -> [0, 1] na 16 úrovní (4-bit) a späť
        u = (x.clamp(-1.0, 1.0) + 1.0) * 0.5
        u_q = torch.round(u * 15.0) / 15.0
        return (u_q * 2.0 - 1.0)
    elif defense_name == "MedianFilter_3x3":
        pil = tensor_to_pil(x).filter(ImageFilter.MedianFilter(size=3))
        return pil_to_tensor(pil, x.device)
    else:
        raise ValueError(f"Neznáma obrana: {defense_name}")


def load_test_photos(n_images: int = 10):
    identities = sorted(d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d)))
    per_id = max(1, n_images // len(identities))
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
    out = []
    for ident in identities:
        id_dir = os.path.join(DATA_ROOT, ident)
        files = sorted(f for f in os.listdir(id_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        for f in files[:per_id]:
            if len(out) >= n_images:
                break
            pil = Image.open(os.path.join(id_dir, f)).convert("RGB").resize((112, 112))
            out.append((f"{ident}/{f}", tf(pil).unsqueeze(0).to(DEVICE)))
    return out


def main():
    ap = argparse.ArgumentParser(description="Vyhodnotenie útokov voči vstupným obranám")
    ap.add_argument("--n-images", type=int, default=10)
    ap.add_argument("--num-iter", type=int, default=80)
    ap.add_argument("--out", type=str, default="/tmp/defense_eval.csv")
    args = ap.parse_args()

    all_models = build_all_models(DEVICE)
    ensemble3 = {k: v for k, v in all_models.items() if k != "AdaFace"}

    from models.pad_minifasnet import load_pad_model
    pad_model = load_pad_model(device=DEVICE)

    mtcnn = MTCNN(image_size=112, margin=0, device="cpu", post_process=False)
    photos = load_test_photos(args.n_images)
    print(f"Načítaných {len(photos)} fotiek na vyhodnotenie obrán.", flush=True)

    defenses = ["None", "JPEG_Q50", "GaussianBlur", "BitDepth_4bit", "MedianFilter_3x3"]
    rows = []

    for name, x in photos:
        lm = extract_landmarks(x, mtcnn=mtcnn, return_mesh=True)
        with torch.no_grad():
            orig_embs = {mname: m(x).detach() for mname, m in all_models.items()}

        # 1. Digitálny PGD (eps=8/255 v [-1, 1] -> eps=16/255=0.0627)
        adv_pgd = pgd_attack_untargeted(all_models["ArcFace"], x, epsilon=16.0/255.0, alpha=2.0/255.0, num_iter=20)

        # 2. Single-Model Physical Patch (ArcFace)
        adv_single, _, _ = ensemble_hybrid_patch_attack(
            models={"ArcFace": all_models["ArcFace"]},
            image=x, num_iter=args.num_iter, eot=True,
            grad_avg=False, momentum=0.0, feature_weight=0.0,
            dot_radius=9.5, dot_pattern="cheeks_chin", strip_w_scale=2.0,
            mtcnn=mtcnn, landmarks=lm, seed=0,
        )

        # 3. BioMesh-SpecPatch (Náš plný útok)
        adv_biomesh, _, _ = ensemble_hybrid_patch_attack(
            models=ensemble3,
            image=x, num_iter=args.num_iter, eot=True,
            grad_avg=True, momentum=0.9, feature_weight=0.6,
            input_diversity=True, update_rule="sign",
            pad_model=pad_model, pad_weight=0.25 if pad_model is not None else 0.0,
            dot_radius=9.5, dot_pattern="cheeks_chin", strip_w_scale=2.0,
            mtcnn=mtcnn, landmarks=lm, seed=0,
        )

        attack_variants = {
            "Digital_PGD_ArcFace": adv_pgd,
            "SinglePatch_ArcFace": adv_single,
            "BioMesh_SpecPatch": adv_biomesh,
        }

        for atk_name, adv_img in attack_variants.items():
            for d_name in defenses:
                def_img = apply_defense(adv_img, d_name)
                with torch.no_grad():
                    sims = {}
                    for mname, m in all_models.items():
                        emb_def = m(def_img)
                        sims[mname] = float(F.cosine_similarity(orig_embs[mname], emb_def).mean().item())
                    train3_sim = float(np.mean([sims[k] for k in ensemble3]))
                    train3_asr = 100.0 * float(np.mean([1.0 if sims[k] < 0.5 else 0.0 for k in ensemble3]))
                    ada_sim = sims["AdaFace"]
                    ada_success = int(ada_sim < 0.5)
                    pad_p = float(pad_model(def_img).mean().item()) if pad_model is not None else 1.0
                    pad_pass = int(pad_p >= 0.5)

                rows.append({
                    "attack": atk_name,
                    "defense": d_name,
                    "photo": name,
                    "sim_ArcFace": round(sims["ArcFace"], 4),
                    "sim_FaceNet": round(sims["FaceNet"], 4),
                    "sim_BenchmarkCNN": round(sims["BenchmarkCNN"], 4),
                    "sim_AdaFace": round(ada_sim, 4),
                    "train3_mean_sim": round(train3_sim, 4),
                    "train3_asr_pct": round(train3_asr, 1),
                    "heldout_adaface_success": ada_success,
                    "pad_live_prob": round(pad_p, 4),
                    "pad_live_pass": pad_pass,
                })
        print(f"Hotová fotka {name}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n=== SÚHRN ODOLNOSTI VOČI OBRANÁM ===", flush=True)
    for atk_name in ("Digital_PGD_ArcFace", "SinglePatch_ArcFace", "BioMesh_SpecPatch"):
        print(f"\n--- {atk_name} ---", flush=True)
        for d_name in defenses:
            sub = [r for r in rows if r["attack"] == atk_name and r["defense"] == d_name]
            arc_asr = 100.0 * np.mean([1.0 if r["sim_ArcFace"] < 0.5 else 0.0 for r in sub])
            t3_asr = float(np.mean([r["train3_asr_pct"] for r in sub]))
            ada_asr = 100.0 * float(np.mean([r["heldout_adaface_success"] for r in sub]))
            ada_sim = float(np.mean([r["sim_AdaFace"] for r in sub]))
            pad_rate = 100.0 * float(np.mean([r["pad_live_pass"] for r in sub]))
            print(f"  {d_name:18s} | ArcFace ASR={arc_asr:5.1f}% | Train3 ASR={t3_asr:5.1f}% "
                  f"| Held-out AdaFace ASR={ada_asr:5.1f}% (sim={ada_sim:.3f}) | PAD Live={pad_rate:5.1f}%",
                  flush=True)


if __name__ == "__main__":
    main()
