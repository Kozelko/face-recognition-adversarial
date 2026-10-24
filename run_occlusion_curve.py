#!/home/kozel/miniconda/envs/cnn-benchmark/bin/python
"""
Potvrdenie + jemná occlusion-vs-ASR krivka pre fyzické útoky, ktoré vyšli
v run_physical_sweep.py ako funkčné.

Pre každú konfiguráciu: 3 modely × viac identít (1 fotka/identita).
Výstup: results/occlusion_curve.csv + results/occlusion_curve_summary.json.
Pre najlepšie konfigurácie uloží aj tlačový hárok do printable_patches/.
"""
import csv
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
import time

import numpy as np
import torch

from run_physical_sweep import load_photos, run_one, KINESIO_3, DEVICE, DATA_ROOT

ITERS = 80
MODELS = ["ArcFace", "FaceNet", "BenchmarkCNN"]


def configs():
    return [
        ("pal3 s1.0 r8", "hybrid", dict(dot_radius=8.0, dot_pattern="cheeks_chin", strip_w_scale=1.0, palette=KINESIO_3)),
        ("pal3 s1.4 r8", "hybrid", dict(dot_radius=8.0, dot_pattern="cheeks_chin", strip_w_scale=1.4, palette=KINESIO_3)),
        ("pal3 s1.8 r8", "hybrid", dict(dot_radius=8.0, dot_pattern="cheeks_chin", strip_w_scale=1.8, palette=KINESIO_3)),
        ("pal3 s2.2 r8", "hybrid", dict(dot_radius=8.0, dot_pattern="cheeks_chin", strip_w_scale=2.2, palette=KINESIO_3)),
        ("pal3 s2.6 r8", "hybrid", dict(dot_radius=8.0, dot_pattern="cheeks_chin", strip_w_scale=2.6, palette=KINESIO_3)),
        ("rgb s1.8 r8", "hybrid", dict(dot_radius=8.0, dot_pattern="cheeks_chin", strip_w_scale=1.8, strip_style="rgb")),
        ("rgb s3.0 r8", "hybrid", dict(dot_radius=8.0, dot_pattern="cheeks_chin", strip_w_scale=3.0, strip_style="rgb")),
        ("leuko 20x50", "leuko", dict(position="nose_bridge", patch_size=(20, 50))),
        ("quad r7", "const", dict(pattern="quad_full", radius=7.0)),
        ("quad r8", "const", dict(pattern="quad_full", radius=8.0)),
    ]


def main():
    from run_patch_experiments import build_all_models
    from attacks.constellation_patch import extract_landmarks
    all_models = build_all_models(DEVICE)
    photos = load_photos(5)            # 5 identít
    lms = {n: extract_landmarks(t, return_mesh=False) for n, t in photos}
    print(f"modely={MODELS} | identity={len(photos)} | iters={ITERS}", flush=True)

    rows = []
    for label, kind, kw in configs():
        saved = False
        for mname in MODELS:
            model = all_models[mname]
            for pname, x in photos:
                t0 = time.time()
                try:
                    out = run_one(kind, model, x, lms[pname], ITERS, kw)
                    adv = out[0]
                    info = out[2] if len(out) > 2 else {}
                    if not saved and len(out) > 1 and out[1] is not None:
                        try:
                            import cv2
                            os.makedirs("printable_patches", exist_ok=True)
                            cv2.imwrite(os.path.join("printable_patches", f"sweep_{label.replace(' ', '_')}.png"), out[1])
                        except Exception:
                            pass
                        saved = True
                    with torch.no_grad():
                        sim = float(torch.nn.functional.cosine_similarity(model(x), model(adv)).mean())
                    occ = float(info.get("face_occlusion_pct", 0.0))
                    ok = int(sim < 0.5)
                except Exception as e:
                    sim, occ, ok = float("nan"), 0.0, 0
                    print(f"  [CHYBA] {label}/{mname}/{pname}: {e}", flush=True)
                rows.append({"config": label, "kind": kind, "model": mname, "photo": pname,
                             "sim": round(sim, 4), "success": ok, "occ_pct": round(occ, 2)})
                print(f"  [{'OK ' if ok else '   '}] {label:14s} {mname:12s} {pname[:18]:18s} "
                      f"sim={sim:.3f} occ={occ:.1f}% ({time.time()-t0:.0f}s)", flush=True)

    with open("results/occlusion_curve.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "kind", "model", "photo", "sim", "success", "occ_pct"])
        w.writeheader(); w.writerows(rows)

    agg = {}
    for r in rows:
        if np.isnan(r["sim"]):
            continue
        a = agg.setdefault(r["config"], {"sim": [], "succ": 0, "occ": [], "pm": {}})
        a["sim"].append(r["sim"]); a["succ"] += r["success"]; a["occ"].append(r["occ_pct"])
        a["pm"].setdefault(r["model"], []).append(r["sim"])
    out = []
    for k, a in agg.items():
        out.append({"config": k, "occ_pct": round(np.mean(a["occ"]), 1), "mean_sim": round(np.mean(a["sim"]), 3),
                    "ASR": round(100 * a["succ"] / len(a["sim"])), "n": len(a["sim"]),
                    "per_model": {m: round(np.mean(v), 3) for m, v in a["pm"].items()}})
    out.sort(key=lambda x: x["occ_pct"])
    json.dump(out, open("results/occlusion_curve_summary.json", "w"), indent=2, ensure_ascii=False)

    print("\n=== OCCLUSION vs ASR (3 modely, 5 identít) ===")
    print(f"{'config':14s} {'occ%':>5} {'meanSim':>8} {'ASR':>5} n  per-model sim")
    for o in out:
        print(f"{o['config']:14s} {o['occ_pct']:5.1f} {o['mean_sim']:8.3f} {o['ASR']:4d}% {o['n']:2d}  " +
              " ".join(f"{m[:6]}={v:.2f}" for m, v in o["per_model"].items()))


if __name__ == "__main__":
    main()
