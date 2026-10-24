#!/usr/bin/env python3
"""
Generátor kompletného master testovacieho hárku A4 (300 DPI) pre diplomovú prácu.
Vytlačí 4 odlišné sady nálepiek na 1 hárok pre rôzne scenáre experimentov:
  1. SADA 1: [HLAVNÁ SOTA] Plný Ensemble 4 modelov (Čierna športová páska)
  2. SADA 2: [AKADEMICKÝ DÔKAZ] Black-Box Leave-One-Out (AdaFace vynechaný)
  3. SADA 3: [MAXIMÁLNA SILA] Plné RGB (Adversarial Art)
  4. SADA 4: [DISKRÉTNY ŠTÝL] Telová lekárenská páska (Tan Fabric)
  + Bonus: Rezervné lícne body v rôznych priemeroch (4.5, 5.5, 6.5 mm).
"""

import os
import sys
import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from models.wrappers import BenchmarkCNNWrapper, FaceNetWrapper, ArcFaceWrapper, AdaFaceWrapper
from models.pad_minifasnet import load_pad_model
from attacks.ensemble_hybrid_patch import ensemble_hybrid_patch_attack

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MASTER SHEET] Používam zariadenie: {device}")

    # 1. Načítanie modelov
    print("[MASTER SHEET] Načítavam modely...")
    all_models = {
        "AdaFace": AdaFaceWrapper(device=device),
        "ArcFace": ArcFaceWrapper(device=device),
        "FaceNet": FaceNetWrapper(pretrained="vggface2", device=device),
        "BenchmarkCNN": BenchmarkCNNWrapper(device=device),
    }
    pad_model = load_pad_model(device=device)

    # 2. Načítanie najnovšej fotky používateľa
    photo_path = "data/custom_dataset/brandajsky_peter/20261024_221137_133138.jpg"
    if not os.path.exists(photo_path):
        # Fallback na ľubovoľnú inú fotku
        p_dir = "data/custom_dataset/brandajsky_peter"
        imgs = sorted([os.path.join(p_dir, f) for f in os.listdir(p_dir) if f.endswith((".jpg", ".png"))])
        photo_path = imgs[-1]

    print(f"[MASTER SHEET] Používam fotku tváre: {photo_path}")
    img_bgr = cv2.imread(photo_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_t = (torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 127.5 - 1.0).unsqueeze(0).to(device)

    # 3. Zoznam konfigurácií pre 4 riadky hárku
    configs = [
        {
            "id": "sada_1",
            "title": "SADA 1: [HLAVNA SOTA] Plny Ensemble 4 modelov (Cierna sportova)",
            "desc": "Ucel: Znefunkcnenie vsetkych 4 modelov naraz (AdaFace, ArcFace, FaceNet, BCNN).",
            "models": all_models,
            "strip_style": "athlete_black",
            "strip_w_scale": 1.15,
            "dot_radius": 5.5,
            "iters": 120,
        },
        {
            "id": "sada_2",
            "title": "SADA 2: [AKADEMICKY DOKAZ] Black-Box Transfer (AdaFace vynechany z treningu)",
            "desc": "Ucel: Dokaz cistej black-box prenosnosti na nevidenu SOTA siet AdaFace (Teza D).",
            "models": {k: v for k, v in all_models.items() if k != "AdaFace"},
            "strip_style": "athlete_black",
            "strip_w_scale": 1.15,
            "dot_radius": 5.5,
            "iters": 120,
        },
        {
            "id": "sada_3",
            "title": "SADA 3: [MAXIMALNA SILA] Plne RGB (Adversarial Art)",
            "desc": "Ucel: 100% garancia prelomenia aj pri slabsom osvetleni / vacsom odstupe od kamery.",
            "models": all_models,
            "strip_style": "adversarial_rgb",
            "strip_w_scale": 1.20,
            "dot_radius": 6.0,
            "iters": 120,
        },
        {
            "id": "sada_4",
            "title": "SADA 4: [DISKRETNY STYL] Telova lekarenska paska (Tan Fabric)",
            "desc": "Ucel: Stealth / nenapadny nosic podobny beznym lekarenskym naplastiam.",
            "models": all_models,
            "strip_style": "tan_fabric",
            "strip_w_scale": 1.10,
            "dot_radius": 5.5,
            "iters": 120,
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
    guide_x = sheet_w - 750
    guide_y = 45
    cv2.rectangle(master_sheet, (guide_x - 15, guide_y - 15), (sheet_w - 30, guide_y + 115), (235, 235, 235), -1)
    cv2.rectangle(master_sheet, (guide_x - 15, guide_y - 15), (sheet_w - 30, guide_y + 115), (180, 180, 180), 1)
    cv2.putText(master_sheet, "KAM LEPIT (VSETKY SADY MAJU ROVNAKU POLOHU):", (guide_x, guide_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(master_sheet, "[N] PASKA: Cez stred chrbta nosa (Breathe Right pozicia)", (guide_x, guide_y + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(master_sheet, "[L] LAVE LICE: Horna licna kost (vlavo na monitore)", (guide_x, guide_y + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(master_sheet, "[P] PRAVE LICE: Horna licna kost (vpravo na monitore)", (guide_x, guide_y + 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, cv2.LINE_AA)

    # Riadky pre jednotlivé sady
    start_y = 260
    row_gap = 720  # px medzi sadami pri 300 DPI

    for idx, cfg in enumerate(configs):
        print(f"\n---> Optimalizujem {cfg['title']} ({cfg['iters']} iter)...")
        adv_img, temp_sheet, info = ensemble_hybrid_patch_attack(
            models=cfg["models"],
            image=img_t,
            strip_style=cfg["strip_style"],
            dot_radius=cfg["dot_radius"],
            dot_pattern="2_cheeks",
            strip_w_scale=cfg["strip_w_scale"],
            num_iter=cfg["iters"],
            lr=0.08,
            pad_model=pad_model,
        )

        # Vyrežeme prvú sadu (SADA 1) z vyrenderovaného hárku:
        # V generátore je prvá sada od y ~ 500 do ~ 1100 px
        row_crop = temp_sheet[460:1120, 20:2400]
        curr_y = start_y + idx * row_gap

        # Rámček a nadpis sady
        cv2.rectangle(master_sheet, (30, curr_y - 30), (sheet_w - 30, curr_y + 640), (248, 248, 248), -1)
        cv2.rectangle(master_sheet, (30, curr_y - 30), (sheet_w - 30, curr_y + 640), (210, 210, 210), 1)

        cv2.putText(master_sheet, cfg["title"], (50, curr_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 40, 160), 2, cv2.LINE_AA)
        cv2.putText(master_sheet, cfg["desc"], (50, curr_y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (90, 90, 90), 1, cv2.LINE_AA)

        # Vloženie vyrezaných nálepiek
        ch, cw = row_crop.shape[:2]
        paste_y = curr_y + 55
        master_sheet[paste_y:paste_y + ch, 50:50 + cw] = row_crop
        print(f"  Vložená {cfg['title']} na pozíciu y={curr_y}")

    # Uloženie finálneho master hárku
    cv2.imwrite("/tmp/master_test_sheet_A4.png", master_sheet)
    artifact_dir = "/home/kozel/.gemini/antigravity-cli/brain/aafe2c41-1e52-4b3a-9394-211a1485f675"
    if os.path.exists(artifact_dir):
        cv2.imwrite(os.path.join(artifact_dir, "master_test_sheet_A4.png"), master_sheet)
    try:
        ws_out = "/home/kozel/fiit/ING/diplomka/face-recognition-adversarial/printable_patches/master_test_sheet_A4.png"
        cv2.imwrite(ws_out, master_sheet)
    except Exception:
        pass
    print(f"\n=======================================================")
    print(f"✅ VŠETKO HOTOVÉ! Master hárok bol úspešne vygenerovaný:")
    print(f"   --> /tmp/master_test_sheet_A4.png ({sheet_w}x{sheet_h} px, 300 DPI)")
    print(f"=======================================================")

if __name__ == "__main__":
    main()
