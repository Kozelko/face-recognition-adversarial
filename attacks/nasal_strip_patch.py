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

from attacks.constellation_patch import extract_landmarks, get_surface_geometry_at_coords


def create_nasal_strip_mask(
    img_size: int,
    center: tuple[float, float],
    width: float,
    height_center: float,
    height_wings: float,
    angle_rad: float,
    yaw: float = 0.0,
    feather: float = 0.70,
    arch_factor: float = 1.0,
    arch_drop_px: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vytvorí presnú 3D masku, reliéf vzpruh, valcové tieňovanie a kontaktný tieň pre Breathe Right pásku.
    Vráti: (mask, surface_relief, drop_shadow, specular_sheen)

    arch_drop_px = o koľko pixelov klesne stredová os pásky na jej koncoch (páska obopína
    nos, konce zídu po stenách nosa ku krídlom). arch_factor je 0..1 váha (1 = plne na nose).
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
    if abs(yaw) > 0.04:
        scale_diff = float(np.clip(0.18 * math.sin(yaw), -0.22, 0.22))
        half_w_left = half_w * (1.0 - scale_diff)
        half_w_right = half_w * (1.0 + scale_diff)
        half_w_local = torch.where(xr >= 0, half_w_right, half_w_left)
    else:
        half_w_local = torch.full_like(xr, half_w)

    hc_half = max(1.5, 0.5 * height_center)
    hw_half = max(2.2, 0.5 * height_wings)

    # 3D anatomické klenutie: páska je prehnutá cez chrbát nosa (∩) a jej konce
    # klesajú po stenách nosa ku krídlom – presne ako reálna Breathe Right páska.
    u_w = torch.clamp(torch.abs(xr) / half_w_local, 0.0, 1.0)
    arch_drop = max(0.0, float(arch_factor)) * max(0.0, float(arch_drop_px))
    yr_arch = yr_raw - arch_drop * (u_w ** 1.8)

    # Plynulý atletický tvar Breathe Right:
    # Jemný pás s hladkým rozšírením do kupolovitých zaoblených koncov
    r_cap = half_w_local * 0.28
    x_dome = torch.clamp(half_w_local - r_cap, min=1.0)

    u_waist = torch.clamp(torch.abs(xr) / x_dome, 0.0, 1.0)
    s_waist = 0.5 * (1.0 - torch.cos(u_waist * math.pi))
    h_local = hc_half + (hw_half - hc_half) * s_waist

    dist_waist = torch.abs(yr_arch) - h_local
    dx = torch.clamp(torch.abs(xr) - x_dome, min=0.0)
    dist_dome = torch.sqrt((dx * (hw_half / r_cap)) ** 2 + yr_arch ** 2) - hw_half

    in_strip_dist = torch.where(torch.abs(xr) <= x_dome, dist_waist, dist_dome)
    mask = torch.clamp(0.5 - in_strip_dist / max(0.35, feather), 0.0, 1.0).view(1, 1, img_size, img_size)

    # Výraznejšie valcové stmavenie na bočných stenách nosa
    shading_lateral = torch.clamp(1.02 - 0.20 * (u_w ** 2), 0.80, 1.04)
    if yaw != 0.0:
        shading_lateral = shading_lateral * (1.0 + 0.12 * math.sin(yaw) * torch.clamp(xr / half_w_local, -1.0, 1.0))
    shading = shading_lateral.view(1, 1, img_size, img_size)

    rail_offset = max(1.0, hc_half * 0.50)
    rail1 = torch.exp(-((yr_arch - rail_offset) ** 2) / 0.85)
    rail2 = torch.exp(-((yr_arch + rail_offset) ** 2) / 0.85)
    rails = ((rail1 + rail2) * 0.08).view(1, 1, img_size, img_size)

    surface_relief = shading + rails

    # Kontaktný tieň a odlesk kopírujúci tvar nosa
    shadow_yr = yr_arch - 0.70
    dist_w_s = torch.abs(shadow_yr) - h_local
    dist_d_s = torch.sqrt((dx * (hw_half / r_cap)) ** 2 + shadow_yr ** 2) - hw_half
    shadow_dist = torch.where(torch.abs(xr) <= x_dome, dist_w_s, dist_d_s)
    shadow_raw = torch.clamp(0.5 - shadow_dist / 0.85, 0.0, 1.0).view(1, 1, img_size, img_size)
    drop_shadow = (torch.clamp(shadow_raw - mask, 0.0, 1.0) * 0.28).view(1, 1, img_size, img_size)

    spec_yr = yr_arch + h_local * 0.50
    specular_sheen = (torch.clamp(1.0 - torch.abs(spec_yr) / 0.85, 0.0, 1.0) ** 2 * mask * 0.10).view(1, 1, img_size, img_size)

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


def compute_strip_3d_adaptation(
    pts_3d: list[tuple[float, float, float]] | np.ndarray | None,
    cx: float,
    cy: float,
    base_w: float,
    base_hc: float,
    base_hw: float,
    roll: float,
    global_yaw: float = 0.0,
    nose_angle: float | None = None,
) -> tuple[float, float, float, float, float, float]:
    """
    Vypočíta plnú 3D povrchovú adaptáciu pásky na nos/tvár podľa reálnej 3D geometrie pokožky:
    1. 3D klenutie (arch_factor): Na chrbte nosa prirodzený oblúk nosového mostíka (~0.08).
       Pri posune pásky na líce, bradu alebo čelo sa klenutie plynule vyrovná (arch -> 0).
    2. 3D rotácia (effective_angle): Na nose sa páska orientuje podľa samotného nosa
       (nose_angle = spojnica nosových krídel), takže pri natočení hlavy sa stáča spolu
       s nosom a jej konce sedia na krídlach. Pri posune mimo nos sa plynule vráti k očnému rollu.
    3. Perspektívne skrátenie (eff_w, eff_hc, eff_hw): Prispôsobenie plochej projekcie sklonu povrchu (cos_slant).
    4. Lokálna asymetria krídel (effective_yaw): Meria rozdiel hĺbky Z medzi ľavým a pravým krídlom pásky,
       čím zabezpečuje prirodzené perspektívne skrátenie vzdialenejšieho krídelka.
    """
    fallback_angle = roll if nose_angle is None else float(nose_angle)
    if pts_3d is None or len(pts_3d) < 100:
        return base_w, base_hc, base_hw, fallback_angle, global_yaw, 0.08

    # 1. Os chrbta nosa (od glabelly medzi obočím po špičku nosa)
    p_top = pts_3d[168] if len(pts_3d) > 168 else pts_3d[6]
    p_tip = pts_3d[4] if len(pts_3d) > 4 else (pts_3d[1] if len(pts_3d) > 1 else pts_3d[195])

    dx_ridge = float(p_tip[0] - p_top[0])
    dy_ridge = float(p_tip[1] - p_top[1])
    ridge_len = max(5.0, math.hypot(dx_ridge, dy_ridge))

    # Normála k osi nosa (laterálny/vodorovný smer v rovine tváre)
    nx_ridge = -dy_ridge / ridge_len
    ny_ridge = dx_ridge / ridge_len

    # Laterálna odchýlka od chrbta nosa (koľko px je páska posunutá doľava/doprava od nosa)
    d_lateral = abs((cx - float(p_top[0])) * nx_ridge + (cy - float(p_top[1])) * ny_ridge)

    # on_nose_weight = 1.0 kdekoľvek na zvislej osi nosa (koreň, stred, dolný mostík, špička)
    # Klesá na 0 iba pri reálnom bočnom posune na líce (d_lateral > 8 px)
    on_nose_weight = float(np.exp(-(d_lateral ** 2) / (2.0 * (6.5 ** 2))))

    # 2. Lokálna 3D geometria povrchu
    cos_s, tilt_angle, z_c = get_surface_geometry_at_coords(pts_3d, cx, cy, roll)

    # 3. Klenutie (arch_factor): váha 0..1 – na nose plná (páska sa ohne cez chrbát nosa),
    # pri posune mimo nos na líce sa plynule vyrovná
    arch_factor = float(on_nose_weight)

    # 4. Uhol natočenia (effective_angle):
    # Na nose sa páska orientuje podľa SAMOTNÉHO NOSA (nose_angle = spojnica nosových krídel),
    # nie podľa očnej línie. Pri natočení hlavy (yaw) sa tak páska stáča spolu s nosom
    # a jej konce ostávajú na nosových krídlach namiesto toho, aby utekali na líce/oko.
    # Iba pri posune mimo nos na líce sa plynule vráti k očnému rollu s jemným sklonom líca.
    if nose_angle is None:
        nose_angle = roll
    angle_to_nose = (float(nose_angle) - roll + math.pi) % (2.0 * math.pi) - math.pi
    angle_diff = (tilt_angle - roll + math.pi) % (2.0 * math.pi) - math.pi
    cheek_tilt = (1.0 - on_nose_weight) * 0.20 * math.sin(angle_diff)
    effective_angle = roll + on_nose_weight * angle_to_nose + cheek_tilt

    # 5. Mierny a prirodzený perspektívny yaw:
    effective_yaw = float(global_yaw * (0.85 + 0.15 * on_nose_weight))

    # 6. Perspektívne rozmery (skrátenie na šikmých lícach):
    w_scale = float(np.clip(0.90 + 0.10 * cos_s, 0.80, 1.0)) if on_nose_weight < 0.5 else 1.0
    h_scale = float(np.clip(0.75 + 0.25 * cos_s, 0.70, 1.0)) if on_nose_weight < 0.5 else 1.0

    eff_w = base_w * w_scale
    eff_hc = base_hc * h_scale
    eff_hw = base_hw * h_scale

    return eff_w, eff_hc, eff_hw, effective_angle, effective_yaw, arch_factor


def compute_nasal_strip_placement(
    landmarks,
    mesh_info: dict | None = None,
    width_scale: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> dict:
    """
    Jediný zdroj pravdy pre polohu, orientáciu a rozmery nosnej pásky (Breathe Right).
    Používa ho útok, live náhľad v app.py aj hybridný útok, aby páska vždy sedela
    presne tam, kde ju používateľ vidí.

    Ukotvenie je odmerané z oficiálnej fotky Breathe Right (páska na tvári):
    1. Stred pása = nosový mostík medzi landmarkami 195 a 5. Presne tam páska
       v reklamnej fotke sedí (zmerané: stred pása y=53, stred 195/5 y=53.6)
       a zodpovedá to aj oficiálnemu návodu „across the bridge of your nose,
       just above where your nose flares out“.
    2. Priečna os = spojnica nosových krídel (landmarky 48/278), takže pri natočení
       hlavy (yaw) sa páska stáča spolu s nosom a jej konce sedia na krídlach.
    3. Šírka pása = šírka nosa (krídla 48/278).
    4. Klenutie: konce pása klesajú ku spojnici krídel, takže páska sa viditeľne
       ohne cez chrbát nosa (∩) namiesto rovnej tyče.

    Vracia dict s hotovými parametrami pre create_nasal_strip_mask().
    """
    if isinstance(landmarks, tuple):
        landmarks, mesh_info = landmarks

    lx, ly = landmarks[0]  # ľavé oko
    rx, ry = landmarks[1]  # pravé oko
    nx, ny = landmarks[2]  # špička nosa

    dx = rx - lx
    dy = ry - ly
    roll = math.atan2(dy, dx)
    eye_dist = math.sqrt(dx**2 + dy**2)

    pts = None
    if mesh_info and "points_3d" in mesh_info and len(mesh_info["points_3d"]) > 100:
        pts = mesh_info["points_3d"]
    yaw = float(mesh_info.get("yaw", 0.0)) if mesh_info else 0.0

    has_alar = pts is not None and len(pts) > 278
    has_bridge = pts is not None and len(pts) > 197

    alar_mid_x = alar_mid_y = None
    if has_alar:
        # Spojnica nosových krídel = skutočná priečna os nosa (vpravo od ľavého krídla)
        alar_l = pts[48]
        alar_r = pts[278]
        ax = float(alar_r[0] - alar_l[0])
        ay = float(alar_r[1] - alar_l[1])
        if ax < 0.0:
            ax, ay = -ax, -ay
        nose_w = math.hypot(ax, ay)
        nose_angle = math.atan2(ay, ax)
        alar_mid_x = float(alar_l[0] + alar_r[0]) * 0.5
        alar_mid_y = float(alar_l[1] + alar_r[1]) * 0.5
        alar_w = nose_w * 0.89
        base_w = max(17.8, min(23.6, alar_w)) * float(width_scale)
    else:
        nose_angle = None
        base_w = max(17.8, min(23.6, eye_dist * 0.48)) * float(width_scale)

    base_hc = base_w * (13.0 / 58.0)  # stred chrbta nosa
    base_hw = base_w * (20.0 / 58.0)  # bočné krídla klesajúce po stenách nosa

    # Stred pásky = nosový mostík medzi landmarkami 195 a 5 (presne ako na oficiálnej
    # fotke Breathe Right). Nesedí na špičke nosa ani na jeho koreni.
    if has_bridge:
        cx = float(pts[195][0] + pts[5][0]) * 0.5
        cy = float(pts[195][1] + pts[5][1]) * 0.5
    else:
        # Bez 3D siete: stred medzi očnou líniou a špičkou nosa (cca 62 %)
        cx = (lx + rx) * 0.5
        cy = (ly + ry) * 0.5 + (ny - (ly + ry) * 0.5) * 0.62

    cx = float(np.clip(cx + float(offset_x), 10.0, 102.0))
    cy = float(np.clip(cy + float(offset_y), 10.0, 102.0))

    # Klenutie: konce pásky klesajú ku spojnici krídel, teda sa páska ohne cez nos.
    # Zmerané na oficiálnej fotke: koniec pásky klesne o cca 1/4 až 1/3 vzdialenosti
    # stred -> krídla (na 112 px frame to vychádza ~3 px).
    half_w = base_w * 0.5
    if has_alar:
        drop = -(alar_mid_x - cx) * math.sin(nose_angle) + (alar_mid_y - cy) * math.cos(nose_angle)
    else:
        drop = 0.30 * (ny - (ly + ry) * 0.5)
    arch_px = float(np.clip(0.35 * drop, 0.0, 0.50 * half_w))

    eff_w, eff_hc, eff_hw, strip_angle, eff_yaw, arch_f = compute_strip_3d_adaptation(
        pts, cx, cy, base_w, base_hc, base_hw, roll, yaw, nose_angle=nose_angle
    )

    return {
        "cx": cx,
        "cy": cy,
        "arch_px": arch_px,
        "base_w": base_w,
        "base_hc": base_hc,
        "base_hw": base_hw,
        "eff_w": eff_w,
        "eff_hc": eff_hc,
        "eff_hw": eff_hw,
        "angle": strip_angle,
        "yaw": eff_yaw,
        "arch": arch_f,
        "roll": roll,
        "eye_dist": eye_dist,
        "on_nose": bool(has_alar),
    }


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

    # Anatomické umiestnenie Breathe Right pásky – jediný zdroj pravdy
    # (rovnaký ako v live náhľade v app.py a v hybridnom útoku).
    place = compute_nasal_strip_placement(
        landmarks, mesh_info,
        width_scale=float(width_scale),
        offset_x=float(offset_x),
        offset_y=float(offset_y),
    )
    cx = place["cx"]
    cy = place["cy"]
    eff_w = place["eff_w"]
    eff_hc = place["eff_hc"]
    eff_hw = place["eff_hw"]
    strip_angle = place["angle"]
    eff_yaw = place["yaw"]
    arch_f = place["arch"]
    arch_px = place["arch_px"]

    # 2. Vytvorenie 3D motýlej masky a reliéfu s adaptovanou 3D geometriou
    mask, surface_relief, drop_shadow, specular_sheen = create_nasal_strip_mask(
        img_size=112,
        center=(cy, cx),
        width=eff_w,
        height_center=eff_hc,
        height_wings=eff_hw,
        angle_rad=strip_angle,
        yaw=eff_yaw,
        feather=0.70,
        arch_factor=arch_f,
        arch_drop_px=arch_px,
    )
    mask = mask.to(device)
    surface_relief = surface_relief.to(device)
    drop_shadow = drop_shadow.to(device)
    specular_sheen = specular_sheen.to(device)

    # 3. Bounding box pre výrez a optimalizačný tenzor
    cy_int = int(round(cy))
    cx_int = int(round(cx))
    pad_h = int(math.ceil(eff_hw * 0.5)) + 3
    pad_w = int(math.ceil(eff_w * 0.5)) + 3

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
    real_width_mm: float = 58.0,
    real_height_wings_mm: float = 20.0,
) -> np.ndarray:
    """
    Jednoduchý 300 DPI tlačový hárok v mierke 1:1 s vodiacimi líniami na vystrihnutie.
    Rozmery sú reálne rozmery Breathe Right pásky (58 mm x 20 mm na krídelkách,
    13 mm v strede). Pozn.: aplikácia používa presnejší tlačový hárok
    `attacks.hybrid_patch.generate_printable_sheet` (kreslí aj presný obrys tvaru);
    táto funkcia ostáva ako jednoduchá alternativa pre samostatnú pásku.
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
