"""
Implementácia nosnej pásky (Nasal Strip Patch) pre tvárovú biometriu.
Maska vychádza z geometrie nosového mostíka s aplikáciou 3D zakrivenia a tieňovania.
"""

import math
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

from attacks.constellation_patch import extract_landmarks


def create_nasal_strip_mask(
    img_size: int,
    center: tuple[float, float],
    width: float,
    height_center: float,
    height_wings: float,
    angle_rad: float,
    yaw: float = 0.0,
    feather: float = 0.70,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vytvorí presnú 3D masku, reliéf vzpruh, valcové tieňovanie a kontaktný tieň pre Breathe Right pásku.
    Vráti: (mask, surface_relief, drop_shadow, specular_sheen)
    """
    cy, cx = center
    y = torch.arange(img_size, dtype=torch.float32).view(-1, 1) - cy
    x = torch.arange(img_size, dtype=torch.float32).view(1, -1) - cx

    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # 2D rotácia podľa náklonu hlavy (roll)
    xr = x * cos_a + y * sin_a
    yr_raw = -x * sin_a + y * cos_a

    half_w = max(5.0, width * 0.5)

    u = xr / half_w
    u_c = torch.clamp(torch.abs(u), 0.0, 1.15)

    # Jemné motýlie rozšírenie krídeliek na bokoch nosa:
    local_half_h = 0.5 * (height_center + (height_wings - height_center) * torch.pow(torch.clamp(u_c, 0.0, 1.0), 1.6))

    # Signed Distance Field (SDF) pre motýlí tvar so zaoblenými koncami
    corner_r = 1.6
    dx = torch.clamp(torch.abs(xr) - (half_w - corner_r), min=0.0)
    dy = torch.clamp(torch.abs(yr_raw) - (local_half_h - corner_r), min=0.0)
    corner_dist = torch.sqrt(dx**2 + dy**2)

    in_strip_dist = torch.where(torch.abs(xr) <= half_w - corner_r, torch.abs(yr_raw) - local_half_h, corner_dist - corner_r)
    mask = torch.clamp(0.5 - in_strip_dist / max(0.4, feather), 0.0, 1.0).view(1, 1, img_size, img_size)

    shading_lateral = torch.clamp(1.0 - 0.22 * (u_c ** 2), 0.78, 1.05)
    if yaw != 0.0:
        shading_lateral = shading_lateral * (1.0 + 0.15 * math.sin(yaw) * torch.clamp(u, -1.0, 1.0))
    shading = shading_lateral.view(1, 1, img_size, img_size)

    rail_offset = max(1.0, height_center * 0.26)
    rail1 = torch.exp(-((yr_raw - rail_offset) ** 2) / 0.75)
    rail2 = torch.exp(-((yr_raw + rail_offset) ** 2) / 0.75)
    rails = ((rail1 + rail2) * 0.09).view(1, 1, img_size, img_size)

    surface_relief = shading + rails

    # Kontaktný tieň a odlesk
    shadow_yr = yr_raw - 0.75
    dy_s = torch.clamp(torch.abs(shadow_yr) - (local_half_h - corner_r), min=0.0)
    shadow_dist = torch.where(torch.abs(xr) <= half_w - corner_r, torch.abs(shadow_yr) - local_half_h, torch.sqrt(dx**2 + dy_s**2) - corner_r)
    shadow_raw = torch.clamp(0.5 - shadow_dist / 0.9, 0.0, 1.0).view(1, 1, img_size, img_size)
    drop_shadow = (torch.clamp(shadow_raw - mask, 0.0, 1.0) * 0.32).view(1, 1, img_size, img_size)

    spec_yr = yr_raw + local_half_h * 0.60
    specular_sheen = (torch.clamp(1.0 - torch.abs(spec_yr) / 0.9, 0.0, 1.0) ** 2 * mask * 0.10).view(1, 1, img_size, img_size)

    return mask, surface_relief, drop_shadow, specular_sheen


def apply_nasal_strip_blending(
    image: torch.Tensor,
    canvas_strip: torch.Tensor,
    mask: torch.Tensor,
    surface_relief: torch.Tensor,
    drop_shadow: torch.Tensor | None = None,
    specular_sheen: torch.Tensor | None = None,
    alpha_opacity: float = 0.94,
) -> torch.Tensor:
    """
    Fotorealistické zlúčenie Breathe Right pásky s tvárou:
    - Kontaktný mikro-tieň na pokožke pod páskou
    - Lokálna adaptácia na osvetlenie pokožky nosa
    - 3D klenutie a dvojité plastové vzpruhy
    - Hladké okraje a jemný matný odlesk
    """
    if drop_shadow is not None:
        base_image = image * (1.0 - drop_shadow)
    else:
        base_image = image

    skin_lum = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
    light_mult = torch.clamp(skin_lum * 0.30 + 1.0, 0.70, 1.30)

    lit_strip = torch.clamp(canvas_strip * light_mult * surface_relief, -1.0, 1.0)

    if specular_sheen is not None:
        lit_strip = torch.clamp(lit_strip + specular_sheen, -1.0, 1.0)

    blended_strip = alpha_opacity * lit_strip + (1.0 - alpha_opacity) * base_image
    output = (1.0 - mask) * base_image + mask * blended_strip
    return output.contiguous()


def apply_nasal_strip_eot(
    image: torch.Tensor,
    canvas_strip: torch.Tensor,
    mask: torch.Tensor,
    surface_relief: torch.Tensor,
    drop_shadow: torch.Tensor | None = None,
    specular_sheen: torch.Tensor | None = None,
    alpha_opacity: float = 0.94,
    max_angle_deg: float = 3.5,
    max_trans: float = 0.015,
    color_jitter: float = 0.04,
    noise_std: float = 0.005,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Expectation over Transformation (EoT) pre fyzickú prenosnosť nosnej pásky:
    simuluje drobný náklon hlavy pred kamerou, posun a zmeny osvetlenia.
    """
    device = image.device
    B, C, H, W = image.shape

    angle_rad = (torch.rand(B, device=device) * 2 - 1) * (max_angle_deg * math.pi / 180.0)
    tx = (torch.rand(B, device=device) * 2 - 1) * max_trans
    ty = (torch.rand(B, device=device) * 2 - 1) * max_trans

    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)

    theta = torch.zeros((B, 2, 3), device=device)
    theta[:, 0, 0] = cos_a
    theta[:, 0, 1] = -sin_a
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin_a
    theta[:, 1, 1] = cos_a
    theta[:, 1, 2] = ty

    grid = F.affine_grid(theta, (B, C, H, W), align_corners=False)
    trans_strip = F.grid_sample(canvas_strip, grid, mode="bilinear", padding_mode="border", align_corners=False)
    trans_mask = F.grid_sample(mask, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    trans_mask = torch.clamp(trans_mask, 0.0, 1.0)
    trans_relief = F.grid_sample(surface_relief, grid, mode="bilinear", padding_mode="border", align_corners=False)

    trans_shadow = None
    if drop_shadow is not None:
        trans_shadow = F.grid_sample(drop_shadow, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    trans_spec = None
    if specular_sheen is not None:
        trans_spec = F.grid_sample(specular_sheen, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    # Svetelný jitter
    brightness = (torch.rand(B, 1, 1, 1, device=device) * 2 - 1) * color_jitter + 1.0
    trans_strip = torch.clamp(trans_strip * brightness, -1.0, 1.0)

    patched_img = apply_nasal_strip_blending(
        image=image,
        canvas_strip=trans_strip,
        mask=trans_mask,
        surface_relief=trans_relief,
        drop_shadow=trans_shadow,
        specular_sheen=trans_spec,
        alpha_opacity=alpha_opacity,
    )

    if noise_std > 0:
        noise = torch.randn_like(patched_img) * noise_std
        patched_img = torch.clamp(patched_img + noise, -1.0, 1.0).contiguous()

    return patched_img, trans_mask


def nasal_strip_attack(
    model: nn.Module,
    image: torch.Tensor,
    style: str = "athlete_black",
    width_scale: float = 1.0,
    num_iter: int = 80,
    lr: float = 0.10,
    eot: bool = True,
    tv_weight: float = 0.02,
    target_emb: torch.Tensor | None = None,
    mtcnn = None,
    landmarks = None,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Vygeneruje a optimalizuje anatomicky presnú športovú nosnú pásku (Breathe Right) na dolnej tretine nosa.
    Podporuje čiernu športovú, karbónovú s mikro-vzorom (Pro Carbon Logo) a telovú látkovú náplasť.
    Umožňuje interaktívne manuálne ladenie polohy (offset_x, offset_y).
    """
    device = image.device

    # 1. Získanie 3D landmarkov nosa a sklonu hlavy
    mesh_info = None
    if landmarks is None:
        landmarks, mesh_info = extract_landmarks(image, mtcnn=mtcnn, return_mesh=True)
    elif isinstance(landmarks, tuple):
        landmarks, mesh_info = landmarks

    lx, ly = landmarks[0]  # ľavé oko
    rx, ry = landmarks[1]  # pravé oko
    nx, ny = landmarks[2]  # špička nosa

    dx = rx - lx
    dy = ry - ly
    roll = math.atan2(dy, dx)
    eye_dist = math.sqrt(dx**2 + dy**2)

    # Anatomické umiestnenie Breathe Right pásky:
    # Páska sa aplikuje cez nosový mostík a nosové chlopne (dolná časť nosa, tesne nad špičkou nosa).
    # Nesmie zasahovať do očí ani koreňa nosa!
    if mesh_info and "points_3d" in mesh_info and len(mesh_info["points_3d"]) > 197:
        pts = mesh_info["points_3d"]
        p195 = pts[195] if len(pts) > 195 else pts[197]
        p197 = pts[197]
        cx = float(p195[0])
        cy = float(p195[1] * 0.70 + p197[1] * 0.30)
    else:
        # 42 % vzdialenosti medzi očami a špičkou nosa = presne nosový mostík nad chlopňami
        cx = (lx + rx) * 0.5
        cy = (ly + ry) * 0.5 + (ny - (ly + ry) * 0.5) * 0.42

    # Aplikácia manuálneho posunu od používateľa
    cx = float(np.clip(cx + offset_x, 10.0, 102.0))
    cy = float(np.clip(cy + offset_y, 10.0, 102.0))

    yaw = mesh_info.get("yaw", 0.0) if mesh_info else 0.0

    # Anatomické proporcie Breathe Right pásky:
    # Šírka je cca 42-45 % vzdialenosti očí (cca 18-22 px, bezpečne pod kútikmi očí)
    base_w = max(16.0, min(25.0, eye_dist * 0.45 * width_scale))
    base_hc = base_w * 0.23   # ~5.2 px v strede chrbta nosa
    base_hw = base_w * 0.29   # ~6.8 px na bočných krídlach

    if mesh_info and "points_3d" in mesh_info and len(mesh_info["points_3d"]) > 168:
        pts = mesh_info["points_3d"]
        p_bridge = pts[168] if len(pts) > 168 else pts[6]
        p_tip = pts[1] if len(pts) > 1 else pts[4]
        nv_x = p_tip[0] - p_bridge[0]
        nv_y = p_tip[1] - p_bridge[1]
        if abs(nv_x) + abs(nv_y) > 1e-4:
            strip_angle = math.atan2(nv_y, nv_x) - math.pi * 0.5
        else:
            strip_angle = roll
    else:
        strip_angle = roll

    # 2. Vytvorenie 3D motýlej masky a reliéfu
    mask, surface_relief, drop_shadow, specular_sheen = create_nasal_strip_mask(
        img_size=112,
        center=(cy, cx),
        width=base_w,
        height_center=base_hc,
        height_wings=base_hw,
        angle_rad=strip_angle,
        yaw=yaw,
        feather=0.70,
    )
    mask = mask.to(device)
    surface_relief = surface_relief.to(device)
    drop_shadow = drop_shadow.to(device)
    specular_sheen = specular_sheen.to(device)

    # 3. Bounding box pre výrez a optimalizačný tenzor
    cy_int = int(round(cy))
    cx_int = int(round(cx))
    pad_h = int(math.ceil(base_hw * 0.5)) + 3
    pad_w = int(math.ceil(base_w * 0.5)) + 3

    y1 = max(0, cy_int - pad_h)
    y2 = min(112, cy_int + pad_h)
    x1 = max(0, cx_int - pad_w)
    x2 = min(112, cx_int + pad_w)
    h_patch = y2 - y1
    w_patch = x2 - x1

    is_rgb = any(k in style.lower() for k in ["rgb", "farebn", "art", "adversarial"])
    n_ch = 3 if is_rgb else 1
    delta_lum = nn.Parameter(torch.zeros((1, n_ch, h_patch, w_patch), device=device))

    if is_rgb:
        base_color = torch.zeros((1, 3, 1, 1), device=device)
        color_scale = torch.ones((1, 3, 1, 1), device=device)
        opacity = 1.0
    elif style in ["pro_carbon_logo", "carbon_sport"]:
        # Tmavý športový karbón s mikro-mriežkou a strieborno-grafitovým akcentom (bez dúhového šumu)
        y_grid = torch.arange(y1, y2, device=device).view(-1, 1) - cy
        x_grid = torch.arange(x1, x2, device=device).view(1, -1) - cx
        carbon_pattern = (0.09 * torch.sin((x_grid + y_grid) * 2.5) * torch.cos((x_grid - y_grid) * 2.5)).view(1, 1, h_patch, w_patch)
        base_color = torch.tensor([-0.65, -0.65, -0.65], device=device).view(1, 3, 1, 1) + carbon_pattern
        color_scale = torch.tensor([0.70, 0.70, 0.70], device=device).view(1, 3, 1, 1)
        opacity = 0.96
    elif style == "clear_silicone":
        # Polopriehľadný silikón
        base_color = torch.tensor([0.22, 0.20, 0.18], device=device).view(1, 3, 1, 1)
        color_scale = torch.tensor([0.28, 0.25, 0.22], device=device).view(1, 3, 1, 1)
        opacity = 0.74
    elif style == "tan_fabric":
        # Telová klasická Breathe Right náplasť (teplý béžový látkový odtieň)
        base_color = torch.tensor([0.65, 0.44, 0.25], device=device).view(1, 3, 1, 1)
        color_scale = torch.tensor([0.30, 0.22, 0.14], device=device).view(1, 3, 1, 1)
        opacity = 0.92
    else:  # "athlete_black"
        # Matná športová karbónovo-čierna páska
        base_color = torch.tensor([-0.72, -0.72, -0.72], device=device).view(1, 3, 1, 1)
        color_scale = torch.tensor([0.50, 0.50, 0.50], device=device).view(1, 3, 1, 1)
        opacity = 0.95

    optimizer = torch.optim.Adam([delta_lum], lr=lr)

    with torch.no_grad():
        orig_emb = model(image).detach()

    for _ in range(num_iter):
        optimizer.zero_grad()

        strip_param = torch.clamp(base_color + delta_lum * color_scale, -1.0, 1.0)
        canvas_strip = image.clone()
        canvas_strip[:, :, y1:y2, x1:x2] = strip_param

        if eot:
            adv_img, _ = apply_nasal_strip_eot(
                image=image,
                canvas_strip=canvas_strip,
                mask=mask,
                surface_relief=surface_relief,
                drop_shadow=drop_shadow,
                specular_sheen=specular_sheen,
                alpha_opacity=opacity,
                max_angle_deg=3.5,
                max_trans=0.015,
                color_jitter=0.03,
                noise_std=0.005,
            )
        else:
            adv_img = apply_nasal_strip_blending(
                image=image,
                canvas_strip=canvas_strip,
                mask=mask,
                surface_relief=surface_relief,
                drop_shadow=drop_shadow,
                specular_sheen=specular_sheen,
                alpha_opacity=opacity,
            )

        adv_img = adv_img.contiguous()
        adv_emb = model(adv_img)

        if target_emb is not None:
            sim = F.cosine_similarity(target_emb, adv_emb).mean()
            loss = -sim
        else:
            sim = F.cosine_similarity(orig_emb, adv_emb).mean()
            loss = sim

        diff_h = delta_lum[:, :, 1:, :] - delta_lum[:, :, :-1, :]
        diff_w = delta_lum[:, :, :, 1:] - delta_lum[:, :, :, :-1]
        tv = (torch.sum(diff_h ** 2) + torch.sum(diff_w ** 2)) / delta_lum.numel()

        reg_loss = torch.mean(delta_lum ** 2)
        total_loss = loss + tv_weight * tv + 0.005 * reg_loss

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            if is_rgb:
                delta_lum.clamp_(-1.0, 1.0)
            else:
                delta_lum.clamp_(-0.85, 0.85)

    with torch.no_grad():
        final_strip = torch.clamp(base_color + delta_lum * color_scale, -1.0, 1.0)
        final_canvas = image.clone()
        final_canvas[:, :, y1:y2, x1:x2] = final_strip

        final_adv_img = apply_nasal_strip_blending(
            image=image,
            canvas_strip=final_canvas,
            mask=mask,
            surface_relief=surface_relief,
            drop_shadow=drop_shadow,
            specular_sheen=specular_sheen,
            alpha_opacity=opacity,
        ).clamp(-1.0, 1.0)

        final_emb = model(final_adv_img)
        if target_emb is not None:
            final_sim = F.cosine_similarity(target_emb, final_emb).item()
            orig_sim = F.cosine_similarity(target_emb, orig_emb).item()
            success = final_sim > 0.5
        else:
            final_sim = F.cosine_similarity(orig_emb, final_emb).item()
            orig_sim = 1.0
            success = final_sim < 0.5

        strip_crop = final_strip.squeeze(0).detach().cpu()

    area_pct = float((mask.sum() / (112 * 112) * 100.0).item())

    info = {
        "style": style,
        "width_scale": width_scale,
        "face_occlusion_pct": round(area_pct, 2),
        "orig_similarity": orig_sim,
        "final_similarity": final_sim,
        "success": success,
        "iterations": num_iter,
        "dimensions_px": (h_patch, w_patch),
    }

    return final_adv_img.detach(), strip_crop, info


def generate_printable_nasal_strip_sheet(
    strip_crop: torch.Tensor,
    filename: str = "printable_nasal_strip_sheet.png",
    real_width_mm: float = 32.0,
    real_height_wings_mm: float = 12.0,
) -> np.ndarray:
    """
    Vygeneruje 300 DPI tlačový hárok v mierke 1:1 s reálnymi rozmermi a vodiacimi líniami na vystrihnutie.
    """
    dpmm = 300.0 / 25.4
    page_w = int(round(100.0 * dpmm))
    page_h = int(round(80.0 * dpmm))

    sheet = np.ones((page_h, page_w, 3), dtype=np.uint8) * 255

    crop_np = ((strip_crop.permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

    target_w_px = int(round(real_width_mm * dpmm))
    target_h_px = int(round(real_height_wings_mm * dpmm))

    resized_strip = cv2.resize(crop_np, (target_w_px, target_h_px), interpolation=cv2.INTER_LANCZOS4)

    sx = (page_w - target_w_px) // 2
    sy = (page_h - target_h_px) // 2 - int(10 * dpmm)

    sheet[sy:sy + target_h_px, sx:sx + target_w_px] = cv2.cvtColor(resized_strip, cv2.COLOR_RGB2BGR)

    cv2.rectangle(sheet, (sx - 2, sy - 2), (sx + target_w_px + 2, sy + target_h_px + 2), (180, 180, 180), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_title = "Adversarial Nasal Strip (Breathe Right) - 300 DPI 1:1"
    text_size = f"Dimensions: {real_width_mm:.1f} mm x {real_height_wings_mm:.1f} mm"
    cv2.putText(sheet, text_title, (int(10 * dpmm), page_h - int(22 * dpmm)), font, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(sheet, text_size, (int(10 * dpmm), page_h - int(12 * dpmm)), font, 0.45, (80, 80, 80), 1, cv2.LINE_AA)

    r_x = page_w - int(35 * dpmm)
    r_y = page_h - int(16 * dpmm)
    r_len = int(10.0 * dpmm)
    cv2.line(sheet, (r_x, r_y), (r_x + r_len, r_y), (0, 0, 0), 2)
    cv2.putText(sheet, "10 mm", (r_x, r_y - 6), font, 0.35, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(filename, sheet)
    return sheet
