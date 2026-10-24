#!/home/kozel/miniconda/envs/cnn-benchmark/bin/python
"""
Experimentálny a hodnotiaci skript pre adversariálne fyzické útoky (Leukoplast & Pimple Patches).
Rýchle vykonávanie:
- Landmarky extrahované len 1x pre každú fotku
- Vizualizácie generované a ukladané cez OpenCV (okamžité uloženie)
- Priebežný výpis s flush=True
"""

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
import json
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

from models.wrappers import ArcFaceWrapper, AdaFaceWrapper, FaceNetWrapper
from evaluate_attacks import load_benchmark_cnn, denormalize
from attacks.constellation_patch import constellation_patch_attack, extract_landmarks
from attacks.adversarial_patch import adversarial_patch_attack
from attacks.hybrid_patch import hybrid_patch_attack
from attacks.ensemble_hybrid_patch import ensemble_hybrid_patch_attack

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Nahradenie hardcoded cesty (/home/kozel/.gemini/...): default do results/, s env override.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.environ.get("PATCH_EVAL_OUT_DIR", os.path.join(_PROJECT_ROOT, "results", "patch_eval"))
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(_PROJECT_ROOT, "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)
TRANSFER_CSV = os.path.join(RESULTS_DIR, "transfer_matrix.csv")


def build_all_models(device):
    """Pokusí sa načítať všetky 4 rozpoznávacie modely (embedding modely)."""
    builders = {
        "ArcFace": lambda: ArcFaceWrapper(device=device),
        "AdaFace": lambda: AdaFaceWrapper(device=device),
        "FaceNet": lambda: FaceNetWrapper(device=device),
        "BenchmarkCNN": lambda: load_benchmark_cnn(device),
    }
    out = {}
    for name, fn in builders.items():
        try:
            m = fn()
            if m is not None:
                m.eval()
                out[name] = m
        except Exception as e:
            print(f"[transfer] Model {name} sa nepodarilo načítať: {e}", flush=True)
    return out


def run_transfer_matrix(tensors, landmarks_map, mtcnn, all_models, num_iter=60, out_csv=None):
    """
    Transfer matrix: útok generovaný na zdrojovom modeli (source),
    vyhodnotený BEZ gradientov na každom cieľovom modeli (target).
    Hodnota = attack success rate (podiel úspešných útokov, sim < 0.5).
    Obsahuje aj riadok Ensemble_LOO (tréning na 3 modeloch, test na 4. held-out modeli).
    """
    if not all_models:
        print("[transfer] Žiadne modely na transfer matrix.", flush=True)
        return []
    csv_target = out_csv or TRANSFER_CSV
    names = list(all_models.keys())
    success_counts = {s: {t: 0 for t in names} for s in names}
    mean_sims = {s: {t: [] for t in names} for s in names}
    totals = {s: 0 for s in names}

    attack_kwargs = dict(
        dot_radius=9.5,
        dot_pattern="cheeks_chin",
        strip_w_scale=2.0,
        grad_avg=True,
        momentum=0.9,
        feature_weight=0.6,
        input_diversity=True,
    )

    for s_name, s_model in all_models.items():
        print(f"\n[transfer] Generujem útok na zdrojovom modeli: {s_name}", flush=True)
        for p_name, x in tensors:
            lms = landmarks_map.get(p_name)
            try:
                adv_x, _sheet, _info = ensemble_hybrid_patch_attack(
                    models={s_name: s_model}, image=x, num_iter=num_iter,
                    eot=True, landmarks=lms, seed=0, **attack_kwargs
                )
            except Exception as e:
                print(f"  [transfer] {s_name} zlyhal generovanie: {e}", flush=True)
                continue
            totals[s_name] += 1
            for t_name, t_model in all_models.items():
                with torch.no_grad():
                    orig = t_model(x)
                    adv = t_model(adv_x.to(next(t_model.parameters()).device))
                    sim = torch.nn.functional.cosine_similarity(orig, adv).mean().item()
                mean_sims[s_name][t_name].append(sim)
                if sim < 0.5:
                    success_counts[s_name][t_name] += 1
        row = {t: round(100.0 * success_counts[s_name][t] / max(1, totals[s_name]), 1) for t in names}
        sim_row = {t: round(float(np.mean(mean_sims[s_name][t])), 4) for t in names if mean_sims[s_name][t]}
        print(f"  ASR [{s_name}] -> {row} | mean_sim -> {sim_row}", flush=True)

    # Leave-One-Out Ensemble (tréning na 3 modeloch okrem t_name, test na held-out t_name)
    loo_key = "Ensemble_LOO_3to1"
    success_counts[loo_key] = {t: 0 for t in names}
    mean_sims[loo_key] = {t: [] for t in names}
    totals[loo_key] = len(tensors)
    print(f"\n[transfer] Generujem Leave-One-Out Ensemble (3 -> 1 held-out):", flush=True)
    for t_name, t_model in all_models.items():
        src_models = {k: v for k, v in all_models.items() if k != t_name}
        for p_name, x in tensors:
            lms = landmarks_map.get(p_name)
            adv_x, _sheet, _info = ensemble_hybrid_patch_attack(
                models=src_models, image=x, num_iter=num_iter,
                eot=True, landmarks=lms, seed=0, **attack_kwargs
            )
            with torch.no_grad():
                orig = t_model(x)
                adv = t_model(adv_x.to(next(t_model.parameters()).device))
                sim = torch.nn.functional.cosine_similarity(orig, adv).mean().item()
            mean_sims[loo_key][t_name].append(sim)
            if sim < 0.5:
                success_counts[loo_key][t_name] += 1
    loo_row = {t: round(100.0 * success_counts[loo_key][t] / max(1, totals[loo_key]), 1) for t in names}
    loo_sim = {t: round(float(np.mean(mean_sims[loo_key][t])), 4) for t in names if mean_sims[loo_key][t]}
    print(f"  ASR [{loo_key}] -> {loo_row} | mean_sim -> {loo_sim}", flush=True)

    rows = []
    for s_name in names + [loo_key]:
        entry = {"source": s_name}
        for t_name in names:
            entry[t_name] = round(100.0 * success_counts[s_name][t_name] / max(1, totals[s_name]), 1)
        rows.append(entry)

    if pd is not None:
        df = pd.DataFrame(rows).set_index("source")
        df.to_csv(csv_target)
        print(f"[transfer] Transfer matrix uložená do: {csv_target}", flush=True)
    else:
        with open(csv_target, "w") as f:
            cols = names
            f.write("source," + ",".join(cols) + "\n")
            for entry in rows:
                f.write(entry["source"] + "," + ",".join(str(entry[c]) for c in cols) + "\n")
        print(f"[transfer] Transfer matrix (CSV ručne) uložená do: {csv_target}", flush=True)
    return rows

def save_cv2_comparison(orig_tensor, adv_tensor, patch_crop, info, model_name, attack_type, filename):
    # Denormalizácia [-1, 1] -> [0, 255] BGR
    orig_np = denormalize(orig_tensor).squeeze().permute(1, 2, 0).cpu().numpy()
    adv_np = denormalize(adv_tensor).squeeze().permute(1, 2, 0).cpu().numpy()
    
    orig_bgr = (orig_np * 255.0).clip(0, 255).astype(np.uint8)
    orig_bgr = cv2.cvtColor(orig_bgr, cv2.COLOR_RGB2BGR)
    
    adv_bgr = (adv_np * 255.0).clip(0, 255).astype(np.uint8)
    adv_bgr = cv2.cvtColor(adv_bgr, cv2.COLOR_RGB2BGR)

    # Rozdiel / Maska
    diff_np = np.abs(adv_np - orig_np)
    diff_bgr = (diff_np * 6.0 * 255.0).clip(0, 255).astype(np.uint8)
    diff_bgr = cv2.cvtColor(diff_bgr, cv2.COLOR_RGB2BGR)

    # Patch textúra
    p_np = ((patch_crop.permute(1, 2, 0).numpy() + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    p_bgr = cv2.cvtColor(p_np, cv2.COLOR_RGB2BGR)
    p_bgr = cv2.resize(p_bgr, (112, 112), interpolation=cv2.INTER_NEAREST)

    # Zväčšenie každého panelu na 224x224 pre detailný náhľad
    h, w = 224, 224
    p1 = cv2.resize(orig_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)
    p2 = cv2.resize(adv_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)
    p3 = cv2.resize(diff_bgr, (w, h), interpolation=cv2.INTER_NEAREST)
    p4 = cv2.resize(p_bgr, (w, h), interpolation=cv2.INTER_NEAREST)

    # Spojenie do jedného riadku
    combined = cv2.hconcat([p1, p2, p3, p4])
    canvas_h, canvas_w = h + 50, combined.shape[1]
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[50:50+h, :combined.shape[1]] = combined

    # Textový popis
    sim_drop = f"Sim: {info['orig_similarity']:.3f} -> {info['final_similarity']:.3f}"
    occ = f"Occ: {info['face_occlusion_pct']}%"
    status = "OKLANÝ (PASS)" if info['success'] else "NEOKLANÝ (FAIL)"
    header = f"{attack_type} | {model_name} | {info.get('pattern', info.get('position'))} | {sim_drop} | {occ} | {status}"
    
    cv2.putText(canvas, header, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    
    # Popisky panelov
    cv2.putText(canvas, "1. Povodna tvar", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "2. Adversarialna", (w + 15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if info['success'] else (0, 165, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "3. Umiestnenie (6x)", (2*w + 15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "4. Vzor tlace", (3*w + 15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    save_path = os.path.join(OUT_DIR, filename)
    cv2.imwrite(save_path, canvas)
    return save_path

def main(series: str = "1,2,3,4,T"):
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    os.chdir(PROJECT_ROOT)
    SERIES = {s.strip().upper() for s in series.split(",")}
    print(f"Aktívne série: {sorted(SERIES)}", flush=True)

    print(f"Spúšťam optimalizovaný benchmark na zariadení: {DEVICE}", flush=True)
    mtcnn = MTCNN(image_size=112, margin=0, device="cpu", post_process=False)
    
    # 1. Modely
    models = {
        "BenchmarkCNN": load_benchmark_cnn(DEVICE),
        "ArcFace": ArcFaceWrapper(device=DEVICE),
    }

    # 2. Vstupné fotky Petra
    photo_names = ["20260503_144825.jpg", "20260503_152428.jpg", "20260517_234908.jpg"]
    photo_paths = [os.path.join("data/custom_dataset/brandajsky_peter", f) for f in photo_names]
    
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
    
    tensors = []
    landmarks_map = {}
    for p in photo_paths:
        if os.path.exists(p):
            fname = os.path.basename(p)
            pil_img = Image.open(p).convert("RGB").resize((112, 112))
            t = tf(pil_img).unsqueeze(0).to(DEVICE)
            tensors.append((fname, t))
            # Extrahuje landmarky vopred 1x
            lms = extract_landmarks(t, mtcnn=mtcnn)
            landmarks_map[fname] = lms
            print(f"Fotka {fname}: landmarky pripravené.", flush=True)

    results = []

    # 3. Testy pre Constellation Patch (Pimple Patches)
    while "1" in SERIES:
        print("\n=== SÉRIA 1: CONSTELLATION PATCH (PIMPLE PATCHES) ===", flush=True)
        patterns = [("3 body (Nos+Lica)", "trio_cheeks_nose"), ("4 body (Full)", "quad_full")]
        radii = [4.0, 5.0, 6.0]
        iter_list = [40, 80, 120]

        for model_name, model in models.items():
            if model is None:
                continue
            print(f"\n>> Model: {model_name}", flush=True)
            for pat_label, pat_key in patterns:
                for r in radii:
                    for n_iter in iter_list:
                        sim_drops = []
                        occlusions = []
                        successes = []

                        saved_img = False
                        for p_name, x in tensors:
                            lms = landmarks_map[p_name]
                            adv_x, crop, info = constellation_patch_attack(
                                model=model,
                                image=x,
                                pattern=pat_key,
                                radius=r,
                                num_iter=n_iter,
                                eot=True,
                                landmarks=lms,
                            )
                            sim_drops.append(info["final_similarity"])
                            occlusions.append(info["face_occlusion_pct"])
                            successes.append(info["success"])

                            if not saved_img:
                                fname = f"constellation_{model_name}_{pat_key}_r{int(r)}_iter{n_iter}.png"
                                save_cv2_comparison(x, adv_x, crop, info, model_name, "Constellation", fname)
                                saved_img = True

                        avg_sim = float(np.mean(sim_drops))
                        avg_occ = float(np.mean(occlusions))
                        asr = float(np.mean(successes)) * 100.0

                        entry = {
                            "attack": "Constellation",
                            "model": model_name,
                            "pattern": pat_label,
                            "radius_px": r,
                            "iterations": n_iter,
                            "avg_final_similarity": round(avg_sim, 4),
                            "avg_occlusion_pct": round(avg_occ, 2),
                            "attack_success_rate": round(asr, 1),
                        }
                        results.append(entry)
                        status_icon = "✅" if asr >= 50.0 else "⚠️"
                        print(f"  {status_icon} [{model_name}] {pat_label} | r={r}px | iters={n_iter:3d} -> Sim: {avg_sim:.4f} | Occ: {avg_occ:.1f}% | ASR: {asr:.0f}%", flush=True)

        break
    # 4. Testy pre Leukoplast 2.0
    while "2" in SERIES:
        print("\n=== SÉRIA 2: ADVERSARIAL LEUKOPLAST 2.0 ===", flush=True)
        positions = [("Koren nosa", "nose_bridge"), ("Lice", "cheek")]
        for model_name, model in models.items():
            if model is None:
                continue
            print(f"\n>> Model: {model_name}", flush=True)
            for pos_label, pos_key in positions:
                for n_iter in [40, 80, 120]:
                    sim_drops = []
                    occlusions = []
                    successes = []
                    saved_img = False

                    for p_name, x in tensors:
                        lms = landmarks_map[p_name]
                        adv_x, crop, info = adversarial_patch_attack(
                            model=model,
                            image=x,
                            position=pos_key,
                            num_iter=n_iter,
                            eot=True,
                            landmarks=lms,
                        )
                        sim_drops.append(info["final_similarity"])
                        occlusions.append(info["face_occlusion_pct"])
                        successes.append(info["success"])

                        if not saved_img:
                            fname = f"leukoplast_{model_name}_{pos_key}_iter{n_iter}.png"
                            save_cv2_comparison(x, adv_x, crop, info, model_name, "Leukoplast", fname)
                            saved_img = True

                    avg_sim = float(np.mean(sim_drops))
                    avg_occ = float(np.mean(occlusions))
                    asr = float(np.mean(successes)) * 100.0

                    entry = {
                        "attack": "Leukoplast",
                        "model": model_name,
                        "position": pos_label,
                        "iterations": n_iter,
                        "avg_final_similarity": round(avg_sim, 4),
                        "avg_occlusion_pct": round(avg_occ, 2),
                        "attack_success_rate": round(asr, 1),
                    }
                    results.append(entry)
                    status_icon = "✅" if asr >= 50.0 else "⚠️"
                    print(f"  {status_icon} [{model_name}] Leukoplast {pos_label} | iters={n_iter:3d} -> Sim: {avg_sim:.4f} | Occ: {avg_occ:.1f}% | ASR: {asr:.0f}%", flush=True)

        break
    # === SÉRIA 3: HYBRIDNÝ F( Breathe Right + nálepky) útok ===
    while "3" in SERIES:
        print("\n=== SÉRIA 3: HYBRID PATCH (Breathe Right + Pimple Patches) ===", flush=True)
        for model_name, model in models.items():
            if model is None:
                continue
            print(f"\n>> Model: {model_name}", flush=True)
            for dot_label, dot_key in [("2 líca", "2_cheeks"), ("1 líce", "1_cheek")]:
                for n_iter in [40, 80, 120]:
                    sim_drops, occlusions, successes = [], [], []
                    saved_img = False
                    for p_name, x in tensors:
                        lms = landmarks_map[p_name]
                        adv_x, crop, info = hybrid_patch_attack(
                            model=model, image=x, strip_style="athlete_black",
                            dot_pattern=dot_key, num_iter=n_iter, eot=True, landmarks=lms,
                        )
                        sim_drops.append(info["final_similarity"])
                        occlusions.append(info["face_occlusion_pct"])
                        successes.append(info["success"])
                        if not saved_img:
                            fname = f"hybrid_{model_name}_{dot_key}_iter{n_iter}.png"
                            save_cv2_comparison(x, adv_x, info["patches"]["strip"], info, model_name, "Hybrid", fname)
                            saved_img = True
                    avg_sim = float(np.mean(sim_drops))
                    avg_occ = float(np.mean(occlusions))
                    asr = float(np.mean(successes)) * 100.0
                    results.append({
                        "attack": "Hybrid", "model": model_name, "pattern": dot_label,
                        "iterations": n_iter, "avg_final_similarity": round(avg_sim, 4),
                        "avg_occlusion_pct": round(avg_occ, 2), "attack_success_rate": round(asr, 1),
                    })
                    icon = "✅" if asr >= 50.0 else "⚠️"
                    print(f"  {icon} [{model_name}] Hybrid {dot_label} | iters={n_iter:3d} -> Sim: {avg_sim:.4f} | Occ: {avg_occ:.1f}% | ASR: {asr:.0f}%", flush=True)

        break
    # === SÉRIA 4: ENSEMBLE + NATURALNESS HYBRID (Téza D) ===
    while "4" in SERIES:
        print("\n=== SÉRIA 4: ENSEMBLE + NATURALNESS HYBRID ===", flush=True)
        all_models = build_all_models(DEVICE)
        if len(all_models) >= 2:
            for n_iter in [60, 100]:
                sims, successes, occs = [], [], []
                saved_img = False
                for p_name, x in tensors:
                    lms = landmarks_map[p_name]
                    adv_x, sheet, info = ensemble_hybrid_patch_attack(
                        models=all_models, image=x, num_iter=n_iter,
                        eot=True, landmarks=lms, seed=0,
                    )
                    sims.append(info["final_similarity"])
                    occs.append(info["face_occlusion_pct"])
                    successes.append(info["success"])
                    if not saved_img:
                        fname = f"ensemble_hybrid_iter{n_iter}.png"
                        save_cv2_comparison(x, adv_x, info["patches"].get("strip"), info,
                                            "Ensemble", "EnsembleHybrid", fname)
                        cv2.imwrite(os.path.join(OUT_DIR, f"ensemble_sheet_iter{n_iter}.png"), sheet)
                        saved_img = True
                avg_sim = float(np.mean(sims))
                asr = float(np.mean(successes)) * 100.0
                results.append({
                    "attack": "EnsembleHybrid", "model": "+".join(all_models.keys()),
                    "iterations": n_iter, "avg_final_similarity": round(avg_sim, 4),
                    "avg_occlusion_pct": round(float(np.mean(occs)), 2),
                    "attack_success_rate": round(asr, 1),
                })
                print(f"  [ensemble] iters={n_iter:3d} -> mean Sim: {avg_sim:.4f} | ensemble ASR: {asr:.0f}%", flush=True)
        else:
            print("  [ensemble] Preskočené: nedostatok modelov.", flush=True)

        break
    # === TRANSFER MATRIX (source = model generujúci útok, target = model hodnotený) ===
    while "T" in SERIES:
        if "all_models" not in locals():
            all_models = build_all_models(DEVICE)
        print("\n=== TRANSFER MATRIX ===", flush=True)
        try:
            run_transfer_matrix(tensors, landmarks_map, mtcnn, all_models, num_iter=60)
        except Exception as e:
            print(f"  [transfer] Zlyhalo: {e}", flush=True)

        break
    # Uloženie dát do JSON
    json_path = os.path.join(OUT_DIR, "benchmark_stats.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBenchmark úspešne dokončený! Štatistiky: {json_path}", flush=True)

if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser(description="Benchmark fyzických patch útokov")
    _ap.add_argument("--series", type=str, default="1,2,3,4,T",
                     help="Ktoré bloky spustiť: 1,2,3,4 a T (T = transfer matrix)."
                          "Napr. --series 3,4 spustí len hybrid + ensemble.")
    _args = _ap.parse_args()
    main(series=_args.series)
