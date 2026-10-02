"""
Implementácia hybridného fyzického útoku (Breathe Right páska na nos + nálepky na tvár).
Umožňuje optimalizáciu textúry a export tlačového hárku v mierke 1:1.
"""

import math
import os
import ctypes
try:
    ctypes.CDLL('/home/kozel/miniconda/envs/cnn-benchmark/lib/python3.11/site-packages/opencv_contrib_python.libs/libpng16-ef62451c.so.16.44.0', mode=ctypes.RTLD_GLOBAL)
except Exception:
    pass
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from attacks.constellation_patch import extract_landmarks, create_elliptical_patch_mask, get_mesh_point_geometry, get_surface_geometry_at_coords
from attacks.nasal_strip_patch import create_nasal_strip_mask


def get_hybrid_geometry(
    landmarks: list[tuple[float, float]],
    mesh_info: dict | None = None,
    img_size: int = 112,
    dot_radius: float = 4.5,
    dot_pattern: str = "2_cheeks",
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    strip_ox: float = 0.0,
    strip_oy: float = 0.0,
    left_ox: float = 0.0,
    left_oy: float = 0.0,
    right_ox: float = 0.0,
    right_oy: float = 0.0,
    chin_ox: float = 0.0,
    chin_oy: float = 0.0,
):
    """
    Vypočíta 3D geometriu, zaoblené masky a povrchový reliéf pre hybridný útok:
    1. Športová páska Breathe Right na dolnej tretine nosa (bod 195/197).
    2. Akné nálepky na lícach (a prípadne brade).
    Dynamicky prepočítava 3D zakrivenie, perspektívu (slant) a rotáciu (tilt) na každom posunutom bode!
    """
    if isinstance(landmarks, tuple):
        landmarks, mesh_info = landmarks

    lx, ly = landmarks[0]  # ľavé oko
    rx, ry = landmarks[1]  # pravé oko
    nx, ny = landmarks[2]  # špička nosa
    mlx, mly = landmarks[3] if len(landmarks) >= 5 else (36.0, 88.0)
    mrx, mry = landmarks[4] if len(landmarks) >= 5 else (76.0, 88.0)

    dx = rx - lx
    dy = ry - ly
    roll = math.atan2(dy, dx)
    eye_dist = math.sqrt(dx**2 + dy**2)
    yaw = mesh_info.get("yaw", 0.0) if mesh_info else 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    has_mesh = mesh_info is not None and "points_3d" in mesh_info and len(mesh_info["points_3d"]) > 400
    pts = mesh_info["points_3d"] if has_mesh else None

    # Páska na nos
    if has_mesh:
        p195 = pts[195] if len(pts) > 195 else pts[197]
        p197 = pts[197]
        strip_cx = float(p195[0])
        strip_cy = float(p195[1] * 0.70 + p197[1] * 0.30)
    else:
        strip_cx = (lx + rx) * 0.5
        strip_cy = (ly + ry) * 0.5 + (ny - (ly + ry) * 0.5) * 0.42

    strip_cx = float(np.clip(strip_cx + offset_x + strip_ox, 10.0, 102.0))
    strip_cy = float(np.clip(strip_cy + offset_y + strip_oy, 10.0, 102.0))

    base_w = max(16.0, min(24.0, eye_dist * 0.44))
    if has_mesh and len(pts) > 168:
        p_bridge = pts[168] if len(pts) > 168 else pts[6]
        p_tip = pts[1] if len(pts) > 1 else pts[4]
        nv_x = p_tip[0] - p_bridge[0]
        nv_y = p_tip[1] - p_bridge[1]
        if abs(nv_x) + abs(nv_y) > 1e-4:
            strip_angle = math.atan2(nv_y, nv_x) - math.pi * 0.5
        else:
            strip_angle = roll
        y_rel = (strip_cy - 50.0) / 20.0
        base_w = base_w * (1.0 + float(np.clip(y_rel * 0.15, -0.15, 0.20)))
    else:
        strip_angle = roll

    base_hc = base_w * 0.23
    base_hw = base_w * 0.29

    strip_mask, strip_relief, strip_shadow, strip_sheen = create_nasal_strip_mask(
        img_size=img_size,
        center=(strip_cy, strip_cx),
        width=base_w,
        height_center=base_hc,
        height_wings=base_hw,
        angle_rad=strip_angle,
        yaw=yaw,
        feather=0.70,
    )
    strip_mask = strip_mask.to(device)
    strip_relief = strip_relief.to(device)

    scy_i, scx_i = int(round(strip_cy)), int(round(strip_cx))
    s_pad_h = int(math.ceil(base_hw * 0.5)) + 3
    s_pad_w = int(math.ceil(base_w * 0.5)) + 3
    sy1, sy2 = max(0, scy_i - s_pad_h), min(img_size, scy_i + s_pad_h)
    sx1, sx2 = max(0, scx_i - s_pad_w), min(img_size, scx_i + s_pad_w)

    boxes = {
        "strip": (sy1, sy2, sx1, sx2),
    }

    # Nálepky na lícach
    r_dot = max(2.5, min(6.5, dot_radius))

    # Ľavé líce
    if has_mesh and len(pts) > 205:
        p_l = pts[205]
        lcx, lcy = float(p_l[0]), float(p_l[1])
    else:
        lcx = lx + (nx - lx) * 0.14
        lcy = ly + (mly - ly) * 0.42

    lcx = float(np.clip(lcx + offset_x + left_ox, 6.0, 106.0))
    lcy = float(np.clip(lcy + offset_y + left_oy, 6.0, 106.0))

    if has_mesh:
        l_cos, l_tilt, _ = get_surface_geometry_at_coords(pts, lcx, lcy, roll)
    else:
        l_cos, l_tilt = 0.85, roll + 0.35

    l_mask, l_relief, l_sh, _ = create_elliptical_patch_mask(
        img_size=img_size,
        center=(lcy, lcx),
        radius_x=r_dot * l_cos,
        radius_y=r_dot,
        angle_rad=l_tilt,
        feather=0.75,
    )
    l_mask, l_relief = l_mask.to(device), l_relief.to(device)

    d_pad = int(math.ceil(r_dot)) + 3
    lcy_i, lcx_i = int(round(lcy)), int(round(lcx))
    ly1, ly2 = max(0, lcy_i - d_pad), min(img_size, lcy_i + d_pad)
    lx1, lx2 = max(0, lcx_i - d_pad), min(img_size, lcx_i + d_pad)
    boxes["left_dot"] = (ly1, ly2, lx1, lx2)

    full_mask = torch.maximum(strip_mask, l_mask)
    full_relief = torch.ones_like(full_mask)
    full_relief = torch.where(strip_mask > 0.1, strip_relief, full_relief)
    full_relief = torch.where(l_mask > 0.1, l_relief, full_relief)

    # Pravé líce
    if "1_cheek" not in dot_pattern and "1 nálepka" not in dot_pattern:
        if has_mesh and len(pts) > 425:
            p_r = pts[425]
            rcx, rcy = float(p_r[0]), float(p_r[1])
        else:
            rcx = rx - (rx - nx) * 0.14
            rcy = ry + (mry - ry) * 0.42

        rcx = float(np.clip(rcx + offset_x + right_ox, 6.0, 106.0))
        rcy = float(np.clip(rcy + offset_y + right_oy, 6.0, 106.0))

        if has_mesh:
            r_cos, r_tilt, _ = get_surface_geometry_at_coords(pts, rcx, rcy, roll)
        else:
            r_cos, r_tilt = 0.85, roll - 0.35

        r_mask, r_relief, _, _ = create_elliptical_patch_mask(
            img_size=img_size,
            center=(rcy, rcx),
            radius_x=r_dot * r_cos,
            radius_y=r_dot,
            angle_rad=r_tilt,
            feather=0.75,
        )
        r_mask, r_relief = r_mask.to(device), r_relief.to(device)

        rcy_i, rcx_i = int(round(rcy)), int(round(rcx))
        ry1, ry2 = max(0, rcy_i - d_pad), min(img_size, rcy_i + d_pad)
        rx1, rx2 = max(0, rcx_i - d_pad), min(img_size, rcx_i + d_pad)
        boxes["right_dot"] = (ry1, ry2, rx1, rx2)

        full_mask = torch.maximum(full_mask, r_mask)
        full_relief = torch.where(r_mask > 0.1, r_relief, full_relief)

    # Brada (ak je zvolený vzor s bradou)
    if "brada" in dot_pattern.lower() or "chin" in dot_pattern.lower() or "3 nálepk" in dot_pattern:
        if has_mesh and len(pts) > 175:
            p_c = pts[175]
            ccx, ccy = float(p_c[0]), float(p_c[1])
        else:
            ccx = (mlx + mrx) * 0.5
            ccy = max(mly, mry) + (ny - (ly + ry) * 0.5) * 0.42

        ccx = float(np.clip(ccx + offset_x + chin_ox, 6.0, 106.0))
        ccy = float(np.clip(ccy + offset_y + chin_oy, 6.0, 106.0))

        if has_mesh:
            c_cos, c_tilt, _ = get_surface_geometry_at_coords(pts, ccx, ccy, roll)
        else:
            c_cos, c_tilt = 0.88, roll

        c_mask, c_relief, _, _ = create_elliptical_patch_mask(
            img_size=img_size,
            center=(ccy, ccx),
            radius_x=r_dot * c_cos,
            radius_y=r_dot,
            angle_rad=c_tilt,
            feather=0.75,
        )
        c_mask, c_relief = c_mask.to(device), c_relief.to(device)

        ccy_i, ccx_i = int(round(ccy)), int(round(ccx))
        cy1, cy2 = max(0, ccy_i - d_pad), min(img_size, ccy_i + d_pad)
        cx1, cx2 = max(0, ccx_i - d_pad), min(img_size, ccx_i + d_pad)
        boxes["chin_dot"] = (cy1, cy2, cx1, cx2)

        full_mask = torch.maximum(full_mask, c_mask)
        full_relief = torch.where(c_mask > 0.1, c_relief, full_relief)
        boxes["chin_mask"] = c_mask
        boxes["chin_relief"] = c_relief

    boxes["strip_mask"] = strip_mask
    boxes["strip_relief"] = strip_relief
    boxes["left_mask"] = l_mask
    boxes["left_relief"] = l_relief
    if "right_mask" not in boxes and "1_cheek" not in dot_pattern and "1 nálepka" not in dot_pattern:
        boxes["right_mask"] = r_mask
        boxes["right_relief"] = r_relief

    full_mask = torch.clamp(full_mask, 0.0, 1.0)
    return full_mask, full_relief, boxes


def generate_printable_sheet(patches: dict, output_path: str | None = None) -> np.ndarray:
    """
    Generuje tlačový hárok (300 DPI) v presnej mierke (100% Actual Size):
    - Športová nosná páska Breathe Right: 32 mm x 12 mm
    - Akné nálepky na líca: priemer 10 mm
    - Kontrolné meradlo (50 mm)
    """
    DPI = 300
    MM_TO_PX = DPI / 25.4

    sheet_w = int(105 * MM_TO_PX)
    sheet_h = int(80 * MM_TO_PX)
    sheet = np.ones((sheet_h, sheet_w, 3), dtype=np.uint8) * 255

    # Text a inštrukcie
    cv2.putText(sheet, "Adversarial Physical Hybrid Patch - Tlacovy harok (100% mierka)", (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(sheet, "Vytlac na matny samolepiaci papier bez zmeny mierky (Actual Size / 100%).",
                (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (80, 80, 80), 1, cv2.LINE_AA)

    # Kalibračné meradlo (50 mm)
    bar_x, bar_y = 30, 120
    bar_len = int(50 * MM_TO_PX)
    cv2.line(sheet, (bar_x, bar_y), (bar_x + bar_len, bar_y), (0, 0, 0), 3)
    cv2.line(sheet, (bar_x, bar_y - 10), (bar_x, bar_y + 10), (0, 0, 0), 3)
    cv2.line(sheet, (bar_x + bar_len, bar_y - 10), (bar_x + bar_len, bar_y + 10), (0, 0, 0), 3)
    cv2.putText(sheet, "Kontrolne meritko: presne 50 mm (over pravitkom po tlaci)", (bar_x, bar_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)

    # Páska na nos
    bw_px = int(32 * MM_TO_PX)
    bh_px = int(12 * MM_TO_PX)
    strip_tensor = patches.get("strip", patches.get("bandage"))
    s_np = ((strip_tensor.permute(1, 2, 0).numpy() + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    s_bgr = cv2.cvtColor(s_np, cv2.COLOR_RGB2BGR)
    s_resized = cv2.resize(s_bgr, (bw_px, bh_px), interpolation=cv2.INTER_LANCZOS4)

    bx, by = 50, 200
    sheet[by:by+bh_px, bx:bx+bw_px] = s_resized
    cv2.rectangle(sheet, (bx-1, by-1), (bx+bw_px+1, by+bh_px+1), (150, 150, 150), 1)
    cv2.putText(sheet, "1. Breathe Right paska na nos (32 x 12 mm)", (bx, by + bh_px + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 1, cv2.LINE_AA)

    # Nálepky na tvár
    dot_d_px = int(10 * MM_TO_PX)
    dot_keys = [k for k in ["left_dot", "right_dot", "chin_dot"] if k in patches]
    for i, key in enumerate(dot_keys):
        label = "2. Lave lice (10 mm)" if key == "left_dot" else ("3. Prave lice (10 mm)" if key == "right_dot" else "4. Brada (10 mm)")
        d_np = ((patches[key].permute(1, 2, 0).numpy() + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
        d_bgr = cv2.cvtColor(d_np, cv2.COLOR_RGB2BGR)
        d_resized = cv2.resize(d_bgr, (dot_d_px, dot_d_px), interpolation=cv2.INTER_LANCZOS4)

        dx = bx + bw_px + 50 + i * (dot_d_px + 40)
        dy = by
        sheet[dy:dy+dot_d_px, dx:dx+dot_d_px] = d_resized
        radius = dot_d_px // 2
        cv2.circle(sheet, (dx + radius, dy + radius), radius + 1, (150, 150, 150), 1)
        cv2.putText(sheet, label, (dx - 10, dy + dot_d_px + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(sheet, "Poznamka: Tlac s profilom farieb Standard/sRGB na matny samolepiaci papier.",
                (30, sheet_h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1, cv2.LINE_AA)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, sheet)

    # Vždy ulož aj do workspace zložky printable_patches/
    ws_copy = "printable_patches/printable_patch_sheet_last.png"
    try:
        os.makedirs("printable_patches", exist_ok=True)
        cv2.imwrite(ws_copy, sheet)
    except Exception:
        pass

    return sheet


def hybrid_patch_attack(
    model: nn.Module,
    image: torch.Tensor,
    strip_style: str = "athlete_black",
    patch_size: tuple[int, int] | None = None,
    dot_radius: float = 4.5,
    dot_pattern: str = "2_cheeks",
    num_iter: int = 100,
    lr: float = 0.07,
    eot: bool = True,
    tv_weight: float = 0.02,
    target_emb: torch.Tensor | None = None,
    mtcnn = None,
    landmarks = None,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    strip_ox: float = 0.0,
    strip_oy: float = 0.0,
    left_ox: float = 0.0,
    left_oy: float = 0.0,
    right_ox: float = 0.0,
    right_oy: float = 0.0,
    chin_ox: float = 0.0,
    chin_oy: float = 0.0,
) -> tuple[torch.Tensor, np.ndarray, dict]:
    """
    Vygeneruje optimalizovaný hybridný útok:
    Anatomická Breathe Right páska na nos + Akné nálepky na lícach (a brade).
    """
    device = image.device
    mesh_info = None
    if landmarks is None:
        landmarks, mesh_info = extract_landmarks(image, mtcnn=mtcnn, return_mesh=True)
    elif isinstance(landmarks, tuple):
        landmarks, mesh_info = landmarks

    full_mask, full_relief, boxes = get_hybrid_geometry(
        landmarks=landmarks,
        mesh_info=mesh_info,
        dot_radius=dot_radius,
        dot_pattern=dot_pattern,
        offset_x=offset_x,
        offset_y=offset_y,
        strip_ox=strip_ox,
        strip_oy=strip_oy,
        left_ox=left_ox,
        left_oy=left_oy,
        right_ox=right_ox,
        right_oy=right_oy,
        chin_ox=chin_ox,
        chin_oy=chin_oy,
    )
    full_mask = full_mask.to(device)
    full_relief = full_relief.to(device)

    sy1, sy2, sx1, sx2 = boxes["strip"]
    ly1, ly2, lx1, lx2 = boxes["left_dot"]
    has_right = "right_dot" in boxes
    has_chin = "chin_dot" in boxes

    sh, sw = sy2 - sy1, sx2 - sx1
    lh, lw = ly2 - ly1, lx2 - lx1

    # Základné farby podľa štýlu (športová karbónová/čierna nosná páska + telové hydrokoloidné nálepky alebo plnofarebný RGB vzor)
    is_rgb = any(k in strip_style.lower() for k in ["rgb", "farebn", "art", "adversarial"])
    n_ch = 3 if is_rgb else 1

    if is_rgb:
        base_strip = torch.zeros((1, 3, 1, 1), device=device)
        base_dots = torch.zeros((1, 3, 1, 1), device=device)
        delta_scale_strip = 1.0
        delta_scale_dots = 1.0
    elif strip_style in ["pro_carbon_logo", "carbon_sport"]:
        y_grid = torch.arange(sy1, sy2, device=device).view(-1, 1) - (sy1 + sy2) * 0.5
        x_grid = torch.arange(sx1, sx2, device=device).view(1, -1) - (sx1 + sx2) * 0.5
        carbon_pattern = (0.09 * torch.sin((x_grid + y_grid) * 2.5) * torch.cos((x_grid - y_grid) * 2.5)).view(1, 1, sh, sw)
        base_strip = torch.tensor([-0.65, -0.65, -0.65], device=device).view(1, 3, 1, 1) + carbon_pattern
        base_dots = torch.tensor([0.68, 0.45, 0.22], device=device).view(1, 3, 1, 1)
        delta_scale_strip = 0.70
        delta_scale_dots = 0.50
    elif strip_style == "tan_fabric":
        base_strip = torch.tensor([0.65, 0.44, 0.25], device=device).view(1, 3, 1, 1)
        base_dots = torch.tensor([0.68, 0.45, 0.22], device=device).view(1, 3, 1, 1)
        delta_scale_strip = 0.70
        delta_scale_dots = 0.50
    else:
        # Čierna športová
        base_strip = torch.tensor([-0.72, -0.72, -0.72], device=device).view(1, 3, 1, 1)
        base_dots = torch.tensor([0.68, 0.45, 0.22], device=device).view(1, 3, 1, 1)
        delta_scale_strip = 0.70
        delta_scale_dots = 0.50

    delta_strip = nn.Parameter(torch.zeros((1, n_ch, sh, sw), device=device))
    delta_left = nn.Parameter(torch.zeros((1, n_ch, lh, lw), device=device))
    params = [delta_strip, delta_left]

    if has_right:
        ry1, ry2, rx1, rx2 = boxes["right_dot"]
        rh, rw = ry2 - ry1, rx2 - rx1
        delta_right = nn.Parameter(torch.zeros((1, n_ch, rh, rw), device=device))
        params.append(delta_right)

    if has_chin:
        cy1, cy2, cx1, cx2 = boxes["chin_dot"]
        ch, cw = cy2 - cy1, cx2 - cx1
        delta_chin = nn.Parameter(torch.zeros((1, n_ch, ch, cw), device=device))
        params.append(delta_chin)

    opt = torch.optim.Adam(params, lr=lr)

    with torch.no_grad():
        orig_emb = model(image).detach()

    B, C, H, W = image.shape

    for _ in range(num_iter):
        opt.zero_grad()

        patch_s = torch.clamp(base_strip + delta_strip * delta_scale_strip, -1.0, 1.0)
        patch_l = torch.clamp(base_dots + delta_left * delta_scale_dots, -1.0, 1.0)

        canvas = image.clone()
        canvas[:, :, sy1:sy2, sx1:sx2] = patch_s
        canvas[:, :, ly1:ly2, lx1:lx2] = patch_l
        if has_right:
            patch_r = torch.clamp(base_dots + delta_right * delta_scale_dots, -1.0, 1.0)
            canvas[:, :, ry1:ry2, rx1:rx2] = patch_r
        if has_chin:
            patch_c = torch.clamp(base_dots + delta_chin * delta_scale_dots, -1.0, 1.0)
            canvas[:, :, cy1:cy2, cx1:cx2] = patch_c

        if eot:
            max_angle_deg = 4.0
            max_trans = 0.02
            angle_rad = (torch.rand(B, device=device) * 2 - 1) * (max_angle_deg * math.pi / 180.0)
            tx = (torch.rand(B, device=device) * 2 - 1) * max_trans
            ty = (torch.rand(B, device=device) * 2 - 1) * max_trans
            cos_a, sin_a = torch.cos(angle_rad), torch.sin(angle_rad)

            theta = torch.zeros((B, 2, 3), device=device)
            theta[:, 0, 0] = cos_a
            theta[:, 0, 1] = -sin_a
            theta[:, 0, 2] = tx
            theta[:, 1, 0] = sin_a
            theta[:, 1, 1] = cos_a
            theta[:, 1, 2] = ty

            grid = F.affine_grid(theta, (B, C, H, W), align_corners=False)
            trans_canvas = F.grid_sample(canvas, grid, mode="bilinear", padding_mode="border", align_corners=False)
            trans_mask = F.grid_sample(full_mask, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            trans_relief = F.grid_sample(full_relief, grid, mode="bilinear", padding_mode="border", align_corners=False)

            brightness = (torch.rand(B, 1, 1, 1, device=device) * 2 - 1) * 0.05 + 1.0
            trans_canvas = torch.clamp(trans_canvas * brightness, -1.0, 1.0)

            skin_lum = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
            light_mult = torch.clamp(skin_lum * 0.35 + 1.0, 0.65, 1.35)
            lit_patch = torch.clamp(trans_canvas * light_mult * trans_relief, -1.0, 1.0)

            patch_blend_factor = 1.0 if is_rgb else 0.92
            blended = patch_blend_factor * lit_patch + (1.0 - patch_blend_factor) * image
            adv_img = (1.0 - trans_mask) * image + trans_mask * blended
        else:
            skin_lum = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
            light_mult = torch.clamp(skin_lum * 0.35 + 1.0, 0.65, 1.35)
            lit_patch = torch.clamp(canvas * light_mult * full_relief, -1.0, 1.0)
            patch_blend_factor = 1.0 if is_rgb else 0.92
            blended = patch_blend_factor * lit_patch + (1.0 - patch_blend_factor) * image
            adv_img = (1.0 - full_mask) * image + full_mask * blended

        adv_img = adv_img.contiguous()
        adv_emb = model(adv_img)

        if target_emb is not None:
            sim = F.cosine_similarity(target_emb, adv_emb).mean()
            loss = -sim
        else:
            sim = F.cosine_similarity(orig_emb, adv_emb).mean()
            loss = sim

        tv_s = (torch.sum((delta_strip[:, :, 1:, :] - delta_strip[:, :, :-1, :])**2) +
                torch.sum((delta_strip[:, :, :, 1:] - delta_strip[:, :, :, :-1])**2)) / delta_strip.numel()
        tv_l = (torch.sum((delta_left[:, :, 1:, :] - delta_left[:, :, :-1, :])**2) +
                torch.sum((delta_left[:, :, :, 1:] - delta_left[:, :, :, :-1])**2)) / delta_left.numel()
        tv_loss = tv_s + tv_l
        reg_loss = torch.mean(delta_strip**2) + torch.mean(delta_left**2)

        if has_right:
            tv_r = (torch.sum((delta_right[:, :, 1:, :] - delta_right[:, :, :-1, :])**2) +
                    torch.sum((delta_right[:, :, :, 1:] - delta_right[:, :, :, :-1])**2)) / delta_right.numel()
            tv_loss = tv_loss + tv_r
            reg_loss = reg_loss + torch.mean(delta_right**2)

        if has_chin:
            tv_c = (torch.sum((delta_chin[:, :, 1:, :] - delta_chin[:, :, :-1, :])**2) +
                    torch.sum((delta_chin[:, :, :, 1:] - delta_chin[:, :, :, :-1])**2)) / delta_chin.numel()
            tv_loss = tv_loss + tv_c
            reg_loss = reg_loss + torch.mean(delta_chin**2)

        total_loss = loss + tv_weight * tv_loss + 0.01 * reg_loss

        total_loss.backward()
        opt.step()

        with torch.no_grad():
            if is_rgb:
                delta_strip.clamp_(-1.0, 1.0)
                delta_left.clamp_(-1.0, 1.0)
                if has_right:
                    delta_right.clamp_(-1.0, 1.0)
                if has_chin:
                    delta_chin.clamp_(-1.0, 1.0)
            else:
                delta_strip.clamp_(-0.80, 0.80)
                delta_left.clamp_(-0.60, 0.60)
                if has_right:
                    delta_right.clamp_(-0.60, 0.60)
                if has_chin:
                    delta_chin.clamp_(-0.60, 0.60)

    with torch.no_grad():
        final_s = torch.clamp(base_strip + delta_strip * delta_scale_strip, -1.0, 1.0)
        final_l = torch.clamp(base_dots + delta_left * delta_scale_dots, -1.0, 1.0)

        final_canvas = image.clone()
        final_canvas[:, :, sy1:sy2, sx1:sx2] = final_s
        final_canvas[:, :, ly1:ly2, lx1:lx2] = final_l
        if has_right:
            final_r = torch.clamp(base_dots + delta_right * delta_scale_dots, -1.0, 1.0)
            final_canvas[:, :, ry1:ry2, rx1:rx2] = final_r
        if has_chin:
            final_c = torch.clamp(base_dots + delta_chin * delta_scale_dots, -1.0, 1.0)
            final_canvas[:, :, cy1:cy2, cx1:cx2] = final_c

        skin_lum = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        light_mult = torch.clamp(skin_lum * 0.35 + 1.0, 0.65, 1.35)
        lit_patch = torch.clamp(final_canvas * light_mult * full_relief, -1.0, 1.0)
        final_adv_img = (1.0 - full_mask) * image + full_mask * (patch_blend_factor * lit_patch + (1.0 - patch_blend_factor) * image)
        final_adv_img = final_adv_img.clamp(-1.0, 1.0)

        final_emb = model(final_adv_img)
        if target_emb is not None:
            final_sim = F.cosine_similarity(target_emb, final_emb).item()
            orig_sim = F.cosine_similarity(target_emb, orig_emb).item()
            success = final_sim > 0.5
        else:
            final_sim = F.cosine_similarity(orig_emb, final_emb).item()
            orig_sim = 1.0
            success = final_sim < 0.5

    patches_dict = {
        "strip": final_s.squeeze(0).cpu(),
        "left_dot": final_l.squeeze(0).cpu(),
    }
    if has_right:
        patches_dict["right_dot"] = final_r.squeeze(0).cpu()
    if has_chin:
        patches_dict["chin_dot"] = final_c.squeeze(0).cpu()

    printable_sheet = generate_printable_sheet(patches_dict)

    occ_pct = float((full_mask.sum() / (112 * 112) * 100.0).item())

    info = {
        "attack": "Hybrid Patch (Breathe Right páska + Pimple Patches)",
        "strip_style": strip_style,
        "dot_pattern": dot_pattern,
        "face_occlusion_pct": round(occ_pct, 2),
        "orig_similarity": orig_sim,
        "final_similarity": final_sim,
        "success": success,
        "iterations": num_iter,
        "patches": patches_dict,
    }

    return final_adv_img.detach(), printable_sheet, info
