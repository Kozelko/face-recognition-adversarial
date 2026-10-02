"""
Experimentálny a hodnotiaci skript pre adversariálne fyzické útoky (Leukoplast & Pimple Patches).
Rýchle vykonávanie:
- Landmarky extrahované len 1x pre každú fotku
- Vizualizácie generované a ukladané cez OpenCV (okamžité uloženie)
- Priebežný výpis s flush=True
"""

import os
import time
import json
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN

from models.wrappers import ArcFaceWrapper
from evaluate_attacks import load_benchmark_cnn, denormalize
from attacks.constellation_patch import constellation_patch_attack, extract_landmarks
from attacks.adversarial_patch import adversarial_patch_attack

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "/home/kozel/.gemini/antigravity-cli/brain/46572df5-3a3e-4bc1-9906-bc4b386a8089/patch_eval"
os.makedirs(OUT_DIR, exist_ok=True)

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

def main():
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    os.chdir(PROJECT_ROOT)

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

    # 4. Testy pre Leukoplast 2.0
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

    # Uloženie dát do JSON
    json_path = os.path.join(OUT_DIR, "benchmark_stats.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBenchmark úspešne dokončený! Štatistiky: {json_path}", flush=True)

if __name__ == "__main__":
    main()
