import os
import sys
import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_DIR = "/home/kozel/fiit/ING/diplomka/face-recognition-adversarial"
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

from evaluate_attacks import load_benchmark_cnn
from models.wrappers import BenchmarkCNNWrapper, FaceNetWrapper, ArcFaceWrapper, AdaFaceWrapper
from models.pad_minifasnet import load_pad_model
from attacks.ensemble_hybrid_patch import ensemble_hybrid_patch_attack

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[EXACT SHEET] Device: {device}")

eval_models = {
    "AdaFace": AdaFaceWrapper(device=device),
    "ArcFace": ArcFaceWrapper(device=device),
    "FaceNet": FaceNetWrapper(pretrained="vggface2", device=device),
    "BenchmarkCNN": load_benchmark_cnn(device),
}
pad_model = load_pad_model(device=device)

p_dir = "data/custom_dataset/brandajsky_peter"
imgs = sorted([os.path.join(p_dir, f) for f in os.listdir(p_dir) if f.endswith((".jpg", ".png"))])
photo_path = "data/custom_dataset/brandajsky_peter/20261024_221137_133138.jpg"
if not os.path.exists(photo_path):
    photo_path = imgs[-1]

img_bgr = cv2.imread(photo_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_t = (torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 127.5 - 1.0).unsqueeze(0).to(device)

centroids = {}
orig_embs = {}
for m_name, m in eval_models.items():
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

print("\n---> Optimalizujem SADA 3: [MAXIMÁLNA SILA] Plné RGB...")
adv_rgb, _, info_rgb = ensemble_hybrid_patch_attack(
    models=eval_models,
    image=img_t,
    strip_style="adversarial_rgb",
    strip_w_scale=1.52,
    dot_radius=6.2,
    dot_pattern="2_cheeks",
    num_iter=160,
    lr=0.09,
    grad_avg=False,
    pad_model=pad_model,
    gallery_centroids=centroids,
    seed=42,
)

print("\n---> Optimalizujem SADA 1: [ČIERNA ŠPORTOVÁ PÁSKA AKO NA FOTKE]...")
adv_blk, _, info_blk = ensemble_hybrid_patch_attack(
    models=eval_models,
    image=img_t,
    strip_style="athlete_black",
    strip_w_scale=1.52,
    dot_radius=6.2,
    dot_pattern="2_cheeks",
    num_iter=160,
    lr=0.09,
    grad_avg=False,
    pad_model=pad_model,
    gallery_centroids=centroids,
    seed=42,
)

DPI = 300
MM_TO_PX = DPI / 25.4
sheet_w = int(round(210.0 * MM_TO_PX))
sheet_h = int(round(297.0 * MM_TO_PX))
sheet = np.ones((sheet_h, sheet_w, 3), dtype=np.uint8) * 255

# 50 mm kalibračné pravítko
bar_x, bar_y = 100, 120
bar_len = int(round(50.0 * MM_TO_PX))
cv2.line(sheet, (bar_x, bar_y), (bar_x + bar_len, bar_y), (0, 0, 0), 3)
cv2.line(sheet, (bar_x, bar_y - 14), (bar_x, bar_y + 14), (0, 0, 0), 3)
cv2.line(sheet, (bar_x + bar_len, bar_y - 14), (bar_x + bar_len, bar_y + 14), (0, 0, 0), 3)
cv2.putText(sheet, "50.0 mm (100% mierka)", (bar_x + 15, bar_y - 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 1, cv2.LINE_AA)

def tensor_to_rgb(t):
    p = t.detach().permute(1, 2, 0).cpu().numpy()
    return np.clip((p + 1.0) * 0.5 * 255.0, 0, 255).astype(np.uint8)

# -------------------------------------------------------------
# FUNKCIA PRE PRESNÝ ANATOMICKÝ TVAR Z FOTKY POUŽÍVATEĽA
# -------------------------------------------------------------
def draw_exact_user_strip(tex_rgb, w_mm=58.0, hc_mm=13.0, hw_mm=20.0, x_pos=100, y_pos=260):
    W = int(round(w_mm * MM_TO_PX))
    Hc = int(round(hc_mm * MM_TO_PX))
    Hw = int(round(hw_mm * MM_TO_PX))
    R_x = int(round(8.5 * MM_TO_PX))
    R_y = Hw // 2

    canvas = np.zeros((Hw, W), dtype=np.uint8)
    cx = W // 2
    cy = Hw // 2
    half_w = W // 2

    x_limit = half_w - R_x
    xs = np.linspace(-x_limit, x_limit, 300)

    pts_top = []
    pts_bot = []

    for x in xs:
        u = x / x_limit
        s = 0.5 * (1.0 - math.cos(abs(u) * math.pi))
        h_local = Hc + (Hw - Hc) * (s ** 1.1)

        top_y = -h_local * 0.5
        bot_y = h_local * 0.5
        pts_top.append((int(round(cx + x)), int(round(cy + top_y))))
        pts_bot.append((int(round(cx + x)), int(round(cy + bot_y))))

    pts_all = []
    pts_all.extend(pts_top)

    for a in np.linspace(-math.pi/2, math.pi/2, 50):
        px = cx + x_limit + R_x * math.cos(a)
        py = cy + R_y * math.sin(a)
        pts_all.append((int(round(px)), int(round(py))))

    pts_all.extend(pts_bot[::-1])

    for a in np.linspace(math.pi/2, 3*math.pi/2, 50):
        px = cx - x_limit + R_x * math.cos(a)
        py = cy + R_y * math.sin(a)
        pts_all.append((int(round(px)), int(round(py))))

    pts_arr = np.array(pts_all, dtype=np.int32)
    cv2.fillPoly(canvas, [pts_arr], 255)

    # Textúra s bilaterálnym odšumením
    tex_res = cv2.resize(tex_rgb, (W, Hw), interpolation=cv2.INTER_CUBIC)
    tex_res = cv2.bilateralFilter(tex_res, d=7, sigmaColor=35, sigmaSpace=7)

    # Vloženie na hárok
    a_norm = (canvas.astype(np.float32) / 255.0)[:, :, np.newaxis]
    target_roi = sheet[y_pos:y_pos + Hw, x_pos:x_pos + W].astype(np.float32)
    blended = tex_res.astype(np.float32) * a_norm + target_roi * (1.0 - a_norm)
    sheet[y_pos:y_pos + Hw, x_pos:x_pos + W] = np.clip(blended, 0, 255).astype(np.uint8)

    # Hladký rezný obrys
    poly_sheet = pts_arr + np.array([x_pos, y_pos])
    cv2.polylines(sheet, [poly_sheet], True, (60, 60, 60), 2, cv2.LINE_AA)

    # Šípka HORE
    arr_cx = x_pos + W // 2
    arr_top = y_pos - 18
    cv2.fillPoly(sheet, [np.array([[arr_cx, arr_top - 18], [arr_cx - 12, arr_top], [arr_cx + 12, arr_top]], dtype=np.int32)], (0, 140, 0))
    cv2.putText(sheet, "HORE (k ociam)", (arr_cx + 18, arr_top - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 130, 0), 1, cv2.LINE_AA)

    return W, Hw

def draw_circle_patch(tex_rgb, d_mm=15.0, x_pos=100, y_pos=260, label=""):
    D = int(round(d_mm * MM_TO_PX))
    R = D // 2
    mask = np.zeros((D, D), dtype=np.uint8)
    cv2.circle(mask, (R, R), R, 255, -1, cv2.LINE_AA)

    tex_res = cv2.resize(tex_rgb, (D, D), interpolation=cv2.INTER_CUBIC)
    tex_res = cv2.bilateralFilter(tex_res, d=7, sigmaColor=35, sigmaSpace=7)

    a_norm = (mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
    target_roi = sheet[y_pos:y_pos + D, x_pos:x_pos + D].astype(np.float32)
    blended = tex_res.astype(np.float32) * a_norm + target_roi * (1.0 - a_norm)
    sheet[y_pos:y_pos + D, x_pos:x_pos + D] = np.clip(blended, 0, 255).astype(np.uint8)

    cv2.circle(sheet, (x_pos + R, y_pos + R), R, (60, 60, 60), 2, cv2.LINE_AA)
    if label:
        cv2.putText(sheet, label, (x_pos + R - 18, y_pos - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1, cv2.LINE_AA)
    return D, D

strip_rgb_tex = tensor_to_rgb(info_rgb["patches"]["strip"])
left_rgb_tex = tensor_to_rgb(info_rgb["patches"]["left_dot"])
right_rgb_tex = tensor_to_rgb(info_rgb["patches"]["right_dot"])

strip_blk_tex = tensor_to_rgb(info_blk["patches"]["strip"])
left_blk_tex = tensor_to_rgb(info_blk["patches"]["left_dot"])
right_blk_tex = tensor_to_rgb(info_blk["patches"]["right_dot"])

# 4 čisté riadky: Sada RGB, Sada Čierna, a 2 náhradné kópie (keby sa zle vystrihlo)
rows_config = [
    ("RGB", strip_rgb_tex, left_rgb_tex, right_rgb_tex),
    ("CIERNA", strip_blk_tex, left_blk_tex, right_blk_tex),
    ("RGB (nahradna)", strip_rgb_tex, left_rgb_tex, right_rgb_tex),
    ("CIERNA (nahradna)", strip_blk_tex, left_blk_tex, right_blk_tex),
]

start_y = 260
row_gap = 460

for i, (name, s_tex, l_tex, r_tex) in enumerate(rows_config):
    curr_y = start_y + i * row_gap
    cv2.putText(sheet, f"SADA: {name}", (100, curr_y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 40, 40), 1, cv2.LINE_AA)

    cx = 100
    w_strip, h_strip = draw_exact_user_strip(s_tex, 58.0, 13.0, 20.0, cx, curr_y)
    cx += w_strip + 100

    d_dot = int(round(15.0 * MM_TO_PX))
    dot_y = curr_y + (h_strip - d_dot) // 2

    d1, _ = draw_circle_patch(l_tex, 15.0, cx, dot_y, label="Lice L")
    cx += d1 + 60

    d2, _ = draw_circle_patch(r_tex, 15.0, cx, dot_y, label="Lice P")

# Export
png_out = "/mnt/c/Users/Kozel/Desktop/master_test_sheet_A4.png"
pdf_out = "/mnt/c/Users/Kozel/Desktop/master_test_sheet_A4.pdf"

cv2.imwrite(png_out, sheet)
img_pil = Image.fromarray(cv2.cvtColor(sheet, cv2.COLOR_BGR2RGB))
img_pil.save(pdf_out, "PDF", resolution=300.0)

for p_dir in [
    "/home/kozel/fiit/ING/diplomka/face-recognition-adversarial/printable_patches",
    "/home/kozel/.gemini/antigravity-cli/brain/aafe2c41-1e52-4b3a-9394-211a1485f675",
]:
    try:
        os.makedirs(p_dir, exist_ok=True)
        cv2.imwrite(os.path.join(p_dir, "master_test_sheet_A4.png"), sheet)
        img_pil.save(os.path.join(p_dir, "master_test_sheet_A4.pdf"), "PDF", resolution=300.0)
    except Exception:
        pass

preview = cv2.resize(sheet, (620, 877), interpolation=cv2.INTER_AREA)
cv2.imwrite("/home/kozel/.gemini/antigravity-cli/brain/aafe2c41-1e52-4b3a-9394-211a1485f675/master_preview.png", preview)

print("\n=======================================================")
print(f"✅ HOTOVO! Vytvorený hárok s presným tvarom z fotky:")
print(f"   --> {pdf_out}")
print(f"   --> {png_out}")
print("=======================================================")
