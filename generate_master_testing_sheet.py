#!/usr/bin/env python3
"""
Generátor kompletného master testovacieho hárku A4 (300 DPI) pre diplomovú prácu.
Vytlačí 4 odlišné sady nálepiek na 1 hárok pre rôzne scenáre experimentov:
  1. SADA 1: [HLAVNÁ SOTA] Plný Ensemble 4 modelov (Čierna športová páska)
  2. SADA 2: [AKADEMICKÝ DÔKAZ] Black-Box Leave-One-Out (AdaFace vynechaný)
  3. SADA 3: [MAXIMÁLNA SILA] Plné RGB (Adversarial Art)
  4. SADA 4: [DISKRÉTNY ŠTÝL] Telová lekárenská páska (Tan Fabric)
Súčasťou každého riadku je aj referenčný náhľad simulovanej tváre pre presné nalepenie.
"""

import os
import sys
import math

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from evaluate_attacks import load_benchmark_cnn
from models.wrappers import BenchmarkCNNWrapper, FaceNetWrapper, ArcFaceWrapper, AdaFaceWrapper
from models.pad_minifasnet import load_pad_model
from attacks.ensemble_hybrid_patch import ensemble_hybrid_patch_attack


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MASTER SHEET] Používam zariadenie: {device}")

    # 1. Načítanie modelov (vrátane natrénovaného BenchmarkCNN checkpointu)
    print("[MASTER SHEET] Načítavam modely...")
    eval_models = {
        "AdaFace": AdaFaceWrapper(device=device),
        "ArcFace": ArcFaceWrapper(device=device),
        "FaceNet": FaceNetWrapper(pretrained="vggface2", device=device),
        "BenchmarkCNN": load_benchmark_cnn(device),
    }

    # Rozšírený surrogate pool pre Sadu 2 (Black-Box Leave-One-Out bez AdaFace)
    surrogates_no_ada = {
        "ArcFace": eval_models["ArcFace"],
        "FaceNet": eval_models["FaceNet"],
        "FaceNet (CASIA)": FaceNetWrapper(pretrained="casia-webface", device=device),
        "BenchmarkCNN": eval_models["BenchmarkCNN"],
    }
    ft_ckpt = "models/checkpoints/benchmark_cnn_finetuned.pth"
    if os.path.exists(ft_ckpt):
        ck = torch.load(ft_ckpt, map_location=device)
        surrogates_no_ada["BenchmarkCNN (Finetuned)"] = BenchmarkCNNWrapper(
            num_classes=ck.get("num_classes", 10575), checkpoint_path=ft_ckpt, device=device
        )

    pad_model = load_pad_model(device=device)

    # 2. Načítanie najnovšej fotky používateľa a výpočet 1:N centroidov galérie
    p_dir = "data/custom_dataset/brandajsky_peter"
    side_cal_dir = os.path.join(p_dir, "side_calibration")
    os.makedirs(side_cal_dir, exist_ok=True)
    for sf in [
        "20261025_155955_517637.jpg",
        "20261025_160012_324967.jpg",
        "20261025_160056_182344.jpg",
        "20261025_160120_562309.jpg",
        "20261025_160234_112415.jpg",
        "20261025_160255_151409.jpg",
    ]:
        src_p = os.path.join(p_dir, sf)
        if os.path.exists(src_p):
            os.replace(src_p, os.path.join(side_cal_dir, sf))
    imgs = sorted([os.path.join(p_dir, f) for f in os.listdir(p_dir) if f.endswith((".jpg", ".png"))])
    photo_path = "data/custom_dataset/brandajsky_peter/20261024_221137_133138.jpg"
    if not os.path.exists(photo_path):
        photo_path = imgs[-1]

    print(f"[MASTER SHEET] Používam fotku tváre: {photo_path} (galéria: {len(imgs)} fotiek)")
    img_bgr = cv2.imread(photo_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_t = (torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 127.5 - 1.0).unsqueeze(0).to(device)

    all_loaded = {**eval_models, **surrogates_no_ada}
    centroids = {}
    orig_embs = {}
    for m_name, m in all_loaded.items():
        embs = []
        for p in imgs:
            b = cv2.imread(p)
            r = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
            t = (torch.from_numpy(r).permute(2, 0, 1).float() / 127.5 - 1.0).unsqueeze(0).to(device)
            with torch.no_grad():
                embs.append(m(t))
        centroids[m_name] = F.normalize(torch.cat(embs, dim=0).mean(dim=0, keepdim=True), p=2, dim=1)
        with torch.no_grad():
            orig_embs[m_name] = m(img_t)

    print("\n[CLEAN BASELINE] Podobnosť čistej fotky voči 1:N databáze:")
    for k in ["AdaFace", "ArcFace", "FaceNet", "BenchmarkCNN"]:
        s_clean = F.cosine_similarity(centroids[k], orig_embs[k]).item()
        print(f"  {k:14s}: 1:N={s_clean:.4f}")

    # 3. Zoznam konfigurácií pre 4 riadky hárku
    configs = [
        {
            "id": "sada_1",
            "title": "SADA 1: [HLAVNA SOTA] Plny Ensemble 4 modelov (Cierna sportova paska)",
            "desc": "Ucel: Znefunkcnenie vsetkych 4 modelov naraz (AdaFace, ArcFace, FaceNet, BCNN).",
            "models": eval_models,
            "strip_style": "athlete_black",
            "strip_w_scale": 1.15,
            "dot_radius": 5.5,
            "iters": 160,
        },
        {
            "id": "sada_2",
            "title": "SADA 2: [AKADEMICKY DOKAZ] Black-Box Transfer (AdaFace vynechany z treningu)",
            "desc": "Ucel: Dokaz cistej black-box prenosnosti na nevidenu SOTA siet AdaFace (Teza D).",
            "models": surrogates_no_ada,
            "strip_style": "athlete_black",
            "strip_w_scale": 1.15,
            "dot_radius": 5.5,
            "iters": 160,
        },
        {
            "id": "sada_3",
            "title": "SADA 3: [MAXIMALNA SILA] Plne RGB (Adversarial Art)",
            "desc": "Ucel: Maximalny kontrast v plnom RGB gamute pri slabsom osvetleni / vacsom odstupe.",
            "models": eval_models,
            "strip_style": "adversarial_rgb",
            "strip_w_scale": 1.18,
            "dot_radius": 5.8,
            "iters": 160,
        },
        {
            "id": "sada_4",
            "title": "SADA 4: [DISKRETNY STYL] Telova lekarenska paska (Tan Fabric)",
            "desc": "Ucel: Stealth / nenapadny nosic v teplom telovo-bezovom odtieni lekarenskej naplaste.",
            "models": eval_models,
            "strip_style": "tan_fabric",
            "strip_w_scale": 1.15,
            "dot_radius": 5.5,
            "iters": 160,
        },
    ]

    # Rozmery A4 pri 300 DPI
    DPI = 300
    MM_TO_PX = DPI / 25.4
    sheet_w = int(round(210.0 * MM_TO_PX))
    sheet_h = int(round(297.0 * MM_TO_PX))
    master_sheet = np.ones((sheet_h, sheet_w, 3), dtype=np.uint8) * 255

    # Hlavička hárku
    cv2.putText(master_sheet, "Adversarial Physical Hybrid Patch - MASTER TESTOVACI HAROK (300 DPI, 100% mierka)",
                (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(master_sheet, "Vytlac na matny papier bez zmeny mierky (Actual Size / 100%). Mierka odvodena z IPD tvare.",
                (40, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (80, 80, 80), 1, cv2.LINE_AA)

    # Kontrolné meradlo 50 mm
    bar_x, bar_y = 40, 135
    bar_len = int(round(50.0 * MM_TO_PX))
    cv2.line(master_sheet, (bar_x, bar_y), (bar_x + bar_len, bar_y), (0, 0, 0), 3)
    cv2.line(master_sheet, (bar_x, bar_y - 12), (bar_x, bar_y + 12), (0, 0, 0), 3)
    cv2.line(master_sheet, (bar_x + bar_len, bar_y - 12), (bar_x + bar_len, bar_y + 12), (0, 0, 0), 3)
    cv2.putText(master_sheet, "Kontrolne meritko: presne 50 mm (over pravitkom po tlaci)", (bar_x, bar_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 1, cv2.LINE_AA)

    # Návod na nalepenie (vždy rovnaké anatomické miesta)
    guide_x = sheet_w - 780
    guide_y = 45
    cv2.rectangle(master_sheet, (guide_x - 15, guide_y - 15), (sheet_w - 30, guide_y + 115), (235, 235, 235), -1)
    cv2.rectangle(master_sheet, (guide_x - 15, guide_y - 15), (sheet_w - 30, guide_y + 115), (180, 180, 180), 1)
    cv2.putText(master_sheet, "KAM LEPIT (VSETKY SADY MAJU ROVNAKU POLOHU):", (guide_x, guide_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(master_sheet, "[N] PASKA: Vodorovne cez stred chrbta nosa (nad dierkami)", (guide_x, guide_y + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(master_sheet, "[L] LAVE LICE: Horna licna kost pod okom (vlavo na fotke)", (guide_x, guide_y + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(master_sheet, "[P] PRAVE LICE: Horna licna kost pod okom (vpravo na fotke)", (guide_x, guide_y + 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, cv2.LINE_AA)

    # Riadky pre jednotlivé sady
    start_y = 240
    row_gap = 780  # px medzi sadami pri 300 DPI
    sim_panels = []

    for idx, cfg in enumerate(configs):
        print(f"\n---> Optimalizujem {cfg['title']} ({cfg['iters']} iter)...")
        g_cent = {k: centroids[k] for k in cfg["models"].keys() if k in centroids}
        adv_img, temp_sheet, info = ensemble_hybrid_patch_attack(
            models=cfg["models"],
            image=img_t,
            strip_style=cfg["strip_style"],
            dot_radius=cfg["dot_radius"],
            dot_pattern="2_cheeks",
            strip_w_scale=cfg["strip_w_scale"],
            num_iter=cfg["iters"],
            lr=0.09,
            grad_avg=False,
            feature_weight=cfg.get("feature_weight", 0.0),
            arch_priors=cfg.get("arch_priors"),
            margin_low=cfg.get("margin_low"),
            pad_model=pad_model,
            gallery_centroids=g_cent,
            seed=42,
        )

        # Výpočet presných 1:1 a 1:N skóre na všetkých 4 hodnotených modeloch
        res_parts = []
        for k in ["AdaFace", "ArcFace", "FaceNet", "BenchmarkCNN"]:
            with torch.no_grad():
                ae = eval_models[k](adv_img)
                s11 = F.cosine_similarity(orig_embs[k], ae).item()
                s1n = F.cosine_similarity(centroids[k], ae).item()
            print(f"  {k:14s}: 1:1={s11:.4f} | 1:N={s1n:.4f}")
            res_parts.append(f"{k}: 1:1={s11:.2f} / 1:N={s1n:.2f}")
        pad_live = (info.get("pad_final") or 0.999) * 100.0
        print(f"  PAD Liveness  : {pad_live:.1f}% | Occlusion: {info.get('face_occlusion_pct', 0):.2f}%")

        # Vyrežeme prvú sadu z vyrenderovaného hárku bez duplicitného nadpisu "SADA 1 (hlavna)"
        # Šípka HORE začína na y ~ 544 px, nálepky + popisky končia na y ~ 900 px
        row_crop = temp_sheet[542:930, 30:1820]
        curr_y = start_y + idx * row_gap

        # Rámček a nadpis sady
        cv2.rectangle(master_sheet, (30, curr_y - 35), (sheet_w - 30, curr_y + 695), (248, 248, 248), -1)
        cv2.rectangle(master_sheet, (30, curr_y - 35), (sheet_w - 30, curr_y + 695), (205, 205, 205), 1)

        cv2.putText(master_sheet, cfg["title"], (50, curr_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 40, 160), 2, cv2.LINE_AA)
        cv2.putText(master_sheet, cfg["desc"], (50, curr_y + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (80, 80, 80), 1, cv2.LINE_AA)
        metrics_line = "Meranie (prah 0.50):  " + "   |   ".join(res_parts) + f"   |   PAD: {pad_live:.1f}%"
        cv2.putText(master_sheet, metrics_line, (50, curr_y + 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.39, (20, 110, 20), 1, cv2.LINE_AA)

        # Vloženie vyrezaných nálepiek v mierke 1:1
        ch, cw = row_crop.shape[:2]
        paste_y = curr_y + 95
        # Zvýraznené biele plátno pod nálepkami na vystrihnutie
        cv2.rectangle(master_sheet, (45, paste_y - 10), (55 + cw, paste_y + ch + 10), (255, 255, 255), -1)
        cv2.rectangle(master_sheet, (45, paste_y - 10), (55 + cw, paste_y + ch + 10), (225, 225, 225), 1)
        master_sheet[paste_y:paste_y + ch, 50:50 + cw] = row_crop

        # Vpravo v riadku: Čistý textový návod a schéma bez fotografií tváre
        box_w = 460
        box_h = 240
        fx0 = sheet_w - box_w - 65
        fy0 = curr_y + 95
        cv2.rectangle(master_sheet, (fx0, fy0), (fx0 + box_w, fy0 + box_h), (255, 255, 255), -1)
        cv2.rectangle(master_sheet, (fx0, fy0), (fx0 + box_w, fy0 + box_h), (210, 210, 210), 1)
        cv2.putText(master_sheet, "POKYNY K NALEPENIU:", (fx0 + 16, fy0 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(master_sheet, "1. [N] PASKA: Cez stred nosa (mostik)", (fx0 + 16, fy0 + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.putText(master_sheet, "2. [L] LAVE LICE: Pod lavym okom", (fx0 + 16, fy0 + 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.putText(master_sheet, "3. [P] PRAVE LICE: Pod pravym okom", (fx0 + 16, fy0 + 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.putText(master_sheet, "Sipka ^ HORE smeruje nahor k ociam", (fx0 + 16, fy0 + 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 120, 0), 1, cv2.LINE_AA)

        # Panel pre kontrolný detailný výstup (iba do samostatného sim_faces súboru, nie na hárok)
        adv_np = ((adv_img[0].detach().permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        adv_bgr = cv2.cvtColor(adv_np, cv2.COLOR_RGB2BGR)
        panel = cv2.resize(adv_bgr, (360, 360), interpolation=cv2.INTER_NEAREST)
        cv2.putText(panel, f"SADA {idx + 1} ({cfg['strip_style']})", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2, cv2.LINE_AA)
        sim_panels.append(panel)
        print(f"  Vložená {cfg['title']} na pozíciu y={curr_y}")

    # Uloženie kontrolného náhľadu 4 simulovaných tvárí vedľa seba
    sim_grid = np.hstack(sim_panels)
    cv2.imwrite("/tmp/sim_faces_4suites.png", sim_grid)

    # Uloženie finálneho master hárku
    cv2.imwrite("/tmp/master_test_sheet_A4.png", master_sheet)
    preview = cv2.resize(master_sheet, (620, 877), interpolation=cv2.INTER_AREA)
    cv2.imwrite("/tmp/master_preview.png", preview)

    artifact_dir = "/home/kozel/.gemini/antigravity-cli/brain/aafe2c41-1e52-4b3a-9394-211a1485f675"
    if os.path.exists(artifact_dir):
        cv2.imwrite(os.path.join(artifact_dir, "master_test_sheet_A4.png"), master_sheet)
        cv2.imwrite(os.path.join(artifact_dir, "master_preview.png"), preview)
        cv2.imwrite(os.path.join(artifact_dir, "sim_faces_4suites.png"), sim_grid)
    for p_dst in [
        "/home/kozel/fiit/ING/diplomka/face-recognition-adversarial/printable_patches/master_test_sheet_A4.png",
        "/mnt/c/Users/Kozel/Desktop/master_test_sheet_A4.png",
        "/mnt/c/Users/Kozel/Downloads/master_test_sheet_A4.png",
    ]:
        try:
            cv2.imwrite(p_dst, master_sheet)
            print("  Uložené na:", p_dst)
        except Exception:
            pass
    print(f"\n=======================================================")
    print(f"✅ VŠETKO HOTOVÉ! Master hárok bol úspešne vygenerovaný:")
    print(f"   --> /tmp/master_test_sheet_A4.png ({sheet_w}x{sheet_h} px, 300 DPI)")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
