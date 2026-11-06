"""
Implementácia hybridného fyzického útoku (Breathe Right páska na nos + nálepky na tvár).
Umožňuje optimalizáciu textúry a export tlačového hárku v mierke 1:1.
"""

import math
import os
import ctypes
try:
    ctypes.CDLL('/home/kozel/miniconda/envs/cnn-benchmark/lib/libpng16.so.16', mode=ctypes.RTLD_GLOBAL)
except Exception:
    pass
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from attacks.constellation_patch import extract_landmarks, create_elliptical_patch_mask, get_mesh_point_geometry, get_surface_geometry_at_coords
from attacks.nasal_strip_patch import create_nasal_strip_mask, compute_nasal_strip_placement, rasterize_barycentric_nasal_strip_mesh


# --- Reálne rozmery Breathe Right pásky (mm) ---
# Plochá páska sa na nos ovine (konce obídu krídla dozadu), preto je jej PREMIETNUTÁ
# šírka na tvári len ~60 % dĺžky – presne to kreslí create_nasal_strip_mask()
# (base_hc = 13/58, base_hw = 20/58 šírky). Tlačový hárok kreslí pásku 1:1 v týchto
# reálnych rozmeroch, takže vytlačený papierik sedí na reálnu Breathe Right.
STRIP_LEN_MM = 58.0      # celková dĺžka pásky
STRIP_WAIST_MM = 14.0    # výška v strede (štíhly pružinový pás)
STRIP_WINGS_MM = 20.0    # výška na koncoch (oblé krídelká)
DOT_DIAM_MM = 15.0       # priemer okrúhlych nálepiek

# --- Vysoko rozlíšené textúrové buffery (aby 300 DPI vzor nebol "kockatý") ---
TEX_STRIP_SIZE = (96, 256)   # (H, W) textúra nosnej pásky
TEX_DOT_SIZE = (96, 96)      # (H, W) textúra nálepky
EOT_BATCH = 4                # počet EoT transformácií na krok (rýchla odozva + silný fyzický transfer)

# Je kornia k dispozícii? JPEG simulácia je implementovaná vlastnou blokovou DCT
# aproximáciou (viď _jpeg_approx), takže na kornii nezávisíme.
try:
    import kornia as _kornia  # noqa: F401
    _HAS_KORNIA = True
except Exception:
    _HAS_KORNIA = False

# Štandardná JPEG kvantizačná tabuľka (luminancia, JPEG Annex K)
_JPEG_LUMA_Q = torch.tensor([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=torch.float32)


def _gblur_fixed(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Rýchle diferencovateľné separabilné Gaussovo rozmazanie s pevnou sigmou."""
    if sigma <= 0.05:
        return x
    c = x.shape[1]
    r = max(1, int(math.ceil(sigma * 2.8)))
    t = torch.arange(-r, r + 1, dtype=torch.float32, device=x.device)
    wts = torch.exp(-(t ** 2) / (2.0 * sigma * sigma))
    wts = wts / wts.sum()
    kx = wts.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    ky = wts.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    out = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), kx, groups=c)
    out = F.conv2d(F.pad(out, (0, 0, r, r), mode="replicate"), ky, groups=c)
    return out


def _illumination_gain(image: torch.Tensor, boxes: dict | None = None) -> torch.Tensor:
    """
    Fotometricky kalibrované 3-kanálové (RGB) osvetlenie scény:
    1. Na nose (sigma ~1.25 px) prenáša presný lokálny smerový tieň stien nosa z fotky 1:1.
    2. Na lícach/brade (sigma ~2.80 px) meria makro-osvetlenie miestnosti s rozptýleným izbovým
       svetlom, aby chĺpky/póry pod papierovou nálepkou nespôsobili neprirodzené stmavnutie.
    3. Prenáša reálnu expozíciu a farebnú teplotu svetla v miestnosti (WB).
    Vracia tenzor (B, 3, H, W).
    """
    img01 = torch.clamp(image * 0.5 + 0.5, 0.0, 1.0)
    b, c, h, w = img01.shape
    s = max(1.0, float(h) / 112.0)

    skin_nose = _gblur_fixed(img01, sigma=1.25 * s)
    skin_cheek = _gblur_fixed(img01, sigma=2.80 * s)

    lum_nose = 0.299 * skin_nose[:, 0:1] + 0.587 * skin_nose[:, 1:2] + 0.114 * skin_nose[:, 2:3]
    lum_cheek = 0.299 * skin_cheek[:, 0:1] + 0.587 * skin_cheek[:, 1:2] + 0.114 * skin_cheek[:, 2:3]

    y0, y1 = int(h * 0.25), int(h * 0.68)
    x0, x1 = int(w * 0.24), int(w * 0.76)
    roi_lum = lum_nose[:, :, y0:y1, x0:x1]
    roi_rgb = skin_nose[:, :, y0:y1, x0:x1]

    ref_lum = (
        roi_lum.mean(dim=(-2, -1), keepdim=True)
        + 0.75 * roi_lum.std(dim=(-2, -1), keepdim=True)
    ).clamp_min(0.14)

    # Odhad intenzity a farby zdroja svetla nezávislý od odtieňa pleti (cez top 5 % odleskov T-zóny a očných bielok)
    flat_lum = roi_lum.flatten(start_dim=-2)
    k_top = max(4, int(flat_lum.shape[-1] * 0.05))
    top_vals, top_idx = torch.topk(flat_lum, k=k_top, dim=-1)
    spec_lum = top_vals.mean(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(0.16)
    exposure = torch.clamp(torch.maximum(ref_lum / 0.58, spec_lum / 0.76), 0.52, 1.08)

    # V pomere lum_nose / ref_lum sa albedo (melanín) pokožky vykráti -> ostane čistý 3D smerový tieň scény!
    shade_nose = torch.clamp(lum_nose / ref_lum, 0.36, 1.06)
    shade_cheek = torch.clamp(0.34 + 0.66 * (lum_cheek / ref_lum), 0.54, 1.06)

    if boxes is not None and "strip_mask" in boxes:
        sm = boxes["strip_mask"].to(image.device)
        if sm.shape[-2:] != (h, w):
            sm = F.interpolate(sm, size=(h, w), mode="bilinear", align_corners=False)
        spatial_lum = torch.where(sm > 0.02, shade_nose, shade_cheek)
    else:
        # Spodná časť tváre (líca/brada mimo nosového pruhu x=0.36..0.64) používa makro-osvetlenie
        xs = torch.linspace(0.0, 1.0, w, device=image.device).view(1, 1, 1, w)
        in_nose_col = ((xs >= 0.36) & (xs <= 0.64)).float()
        spatial_lum = in_nose_col * shade_nose + (1.0 - in_nose_col) * shade_cheek

    skin_neutral = torch.tensor([1.10, 0.96, 0.94], dtype=torch.float32, device=image.device).view(1, 3, 1, 1)
    mean_rgb = roi_rgb.mean(dim=(-2, -1), keepdim=True)
    mean_l = roi_lum.mean(dim=(-2, -1), keepdim=True).clamp_min(0.10)
    wb_skin = (mean_rgb / mean_l) / skin_neutral

    flat_rgb = roi_rgb.flatten(start_dim=-2)
    spec_rgb = torch.gather(flat_rgb, -1, top_idx.expand(-1, 3, -1)).mean(dim=-1, keepdim=True).unsqueeze(-1)
    wb_spec = spec_rgb / spec_lum

    raw_wb = 0.55 * wb_spec + 0.45 * wb_skin
    wb_luma = (0.299 * raw_wb[:, 0:1] + 0.587 * raw_wb[:, 1:2] + 0.114 * raw_wb[:, 2:3]).clamp_min(0.10)
    raw_wb = raw_wb / wb_luma
    wb_rgb = torch.clamp(1.0 + 0.72 * (raw_wb - 1.0), 0.76, 1.24)

    return torch.clamp(spatial_lum * exposure * wb_rgb, 0.28, 1.20)


def _region_field(field: torch.Tensor, box: tuple[int, int, int, int], tex_h: int, tex_w: int, pad: int = 2) -> torch.Tensor:
    """Vyreže oblasť okolo boxu (s malým presahom) a diferencovateľne ju prevzorkuje
    do rozlíšenia textúrového buffera."""
    y1, y2, x1, x2 = box
    h, w = field.shape[-2], field.shape[-1]
    py1, py2 = max(0, y1 - pad), min(h, y2 + pad)
    px1, px2 = max(0, x1 - pad), min(w, x2 + pad)
    crop = field[:, :, py1:py2, px1:px2]
    return F.interpolate(crop, size=(tex_h, tex_w), mode="bilinear", align_corners=False)


def _paper_albedo(texture: torch.Tensor) -> torch.Tensor:
    """
    Fyzikálna odrazivosť (albedo v [0, 1]) matného 80g kancelárskeho papiera vytlačeného
    na Canon PIXMA TS3450 (PG-545 pigment black + CL-546 dye color):
      - Stage 1 rozpíjanie kvapiek atramentu do vlákien papiera (dot gain blur),
      - farebná sýtosť CMY atramentu na matnom papieri (~0.72),
      - najtmavšia matná čierna odráža rozptýlené izbové svetlo (~0.17),
      - nepotlačený biely papier má difúznu odrazivosť ~0.94.
    """
    tex01 = torch.clamp(texture * 0.5 + 0.5, 0.0, 1.0)
    sig_ink = 1.10 if tex01.shape[-1] >= 128 else 0.65
    tex_ink = _gblur_fixed(tex01, sigma=sig_ink)
    luma = 0.299 * tex_ink[:, 0:1] + 0.587 * tex_ink[:, 1:2] + 0.114 * tex_ink[:, 2:3]
    tex_matte = torch.clamp(luma + (tex_ink - luma) * 0.72, 0.0, 1.0)
    return 0.17 + tex_matte * (0.94 - 0.17)


def _warp_flat_strip_to_box(
    albedo_hr: torch.Tensor,
    box: tuple[int, int, int, int],
    strip_geom: dict,
    device: torch.device,
) -> torch.Tensor:
    """
    3D valcové + elastické sedlové ovinutie plochej Breathe Right pásky (96x256) na chrbát
    a šikmé bočné steny nosa v 2x supersamplingu (100 % zhodné s create_nasal_strip_mask).
    Ak je k dispozícii predpočítaná barycentrická 3D UV mriežka (mesh_uv_grid), použije
    priamo reálnu 3D trianguláciu nosa (MediaPipe).
    """
    mesh_uv = strip_geom.get("mesh_uv_grid")
    if mesh_uv is not None:
        grid = mesh_uv.to(device)
        if albedo_hr.shape[0] > 1 and grid.shape[0] == 1:
            grid = grid.expand(albedo_hr.shape[0], -1, -1, -1)
        warped_2x = F.grid_sample(albedo_hr, grid, mode="bilinear", padding_mode="border", align_corners=False)
        return F.avg_pool2d(warped_2x, kernel_size=2, stride=2)

    y1, y2, x1, x2 = box
    bh, bw = max(1, y2 - y1), max(1, x2 - x1)
    sh2, sw2 = bh * 2, bw * 2
    ys = torch.linspace(y1 + 0.25, y2 - 0.25, sh2, device=device)
    xs = torch.linspace(x1 + 0.25, x2 - 0.25, sw2, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    cx = float(strip_geom["cx"])
    cy = float(strip_geom["cy"])
    half_w = max(4.0, float(strip_geom["eff_w"]) * 0.5)
    hc_half = max(1.5, float(strip_geom["eff_hc"]) * 0.5)
    hw_half = max(1.8, float(strip_geom["eff_hw"]) * 0.5)
    ang = float(strip_geom.get("angle", 0.0))
    arch_drop = max(0.0, float(strip_geom.get("arch_drop", 0.0)))
    yaw = float(strip_geom.get("yaw", 0.0))

    dx = xx - cx
    dy = yy - cy
    ca, sa = math.cos(ang), math.sin(ang)
    xr = dx * ca + dy * sa
    yr_raw = -dx * sa + dy * ca

    if abs(yaw) > 0.04:
        scale_diff = float(np.clip(0.18 * math.sin(yaw), -0.22, 0.22))
        half_w_local = torch.where(
            xr >= 0.0,
            torch.full_like(xr, half_w * (1.0 + scale_diff)),
            torch.full_like(xr, half_w * (1.0 - scale_diff)),
        )
    else:
        half_w_local = torch.full_like(xr, half_w)

    u_proj = torch.clamp(xr / torch.clamp(half_w_local, min=2.0), -1.0, 1.0)
    u_abs = torch.abs(u_proj)

    # Valcové rozvinutie pozdĺž stien nosa (stred čelne ku kamere, bočné krídla sklonené ~64°)
    theta_max = 1.12
    sin_tmax = math.sin(theta_max)
    u_flat = torch.asin(torch.clamp(u_proj * sin_tmax, -0.999, 0.999)) / theta_max

    # Elastický profil pružinového pásu zhodný s create_nasal_strip_mask
    u_t = torch.clamp(u_abs / 0.94, 0.0, 1.0)
    arch_profile = torch.clamp(3.0 * (u_t ** 2) - 2.0 * (u_t ** 3), 0.0, 1.0) ** 1.85
    yr_arch = yr_raw - arch_drop * arch_profile

    r_cap = half_w_local * 0.18
    x_dome = torch.clamp(half_w_local - r_cap, min=1.0)
    u_waist = torch.clamp(torch.abs(xr) / x_dome, 0.0, 1.0)
    s_waist = u_waist ** 1.75
    h_local = hc_half + (hw_half - hc_half) * s_waist

    waist_ratio_paper = STRIP_WAIST_MM / STRIP_WINGS_MM
    u_paper_w = torch.clamp(torch.abs(u_flat) / 0.78, 0.0, 1.0)
    paper_h_frac = waist_ratio_paper + (1.0 - waist_ratio_paper) * (u_paper_w ** 1.75)
    v_flat = (yr_arch / torch.clamp(h_local, min=1.0)) * paper_h_frac

    grid = torch.stack([u_flat, v_flat], dim=-1).unsqueeze(0)
    if albedo_hr.shape[0] > 1:
        grid = grid.expand(albedo_hr.shape[0], -1, -1, -1)
    warped_2x = F.grid_sample(albedo_hr, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return F.avg_pool2d(warped_2x, kernel_size=2, stride=2)


def _warp_flat_dot_to_box(
    albedo_hr: torch.Tensor,
    box: tuple[int, int, int, int],
    dot_geom_tuple: tuple,
    r_dot: float,
    device: torch.device,
) -> torch.Tensor:
    """
    3D perspektívne skrátenie (foreshortening) plochej kruhovej nálepky (96x96) na naklonené
    líce/bradu pomocou symetrického tenzora M^{-1} (orientácia „HORE“ ostáva nahor k očiam).
    """
    y1, y2, x1, x2 = box
    bh, bw = max(1, y2 - y1), max(1, x2 - x1)
    sh2, sw2 = bh * 2, bw * 2
    ys = torch.linspace(y1 + 0.25, y2 - 0.25, sh2, device=device)
    xs = torch.linspace(x1 + 0.25, x2 - 0.25, sw2, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    cx_d, cy_d, cos_d, tilt_d = (
        float(dot_geom_tuple[0]),
        float(dot_geom_tuple[1]),
        max(0.38, float(dot_geom_tuple[2])),
        float(dot_geom_tuple[3]),
    )
    dx = xx - cx_d
    dy = yy - cy_d
    ux, uy = math.cos(tilt_d), math.sin(tilt_d)
    vx, vy = -math.sin(tilt_d), math.cos(tilt_d)
    du = (dx * ux + dy * uy) / cos_d
    dv = dx * vx + dy * vy
    r_eff = max(1.0, float(r_dot))
    u_dot = (du * ux + dv * vx) / r_eff
    v_dot = (du * uy + dv * vy) / r_eff

    grid = torch.stack([u_dot, v_dot], dim=-1).unsqueeze(0)
    if albedo_hr.shape[0] > 1:
        grid = grid.expand(albedo_hr.shape[0], -1, -1, -1)
    warped_2x = F.grid_sample(albedo_hr, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return F.avg_pool2d(warped_2x, kernel_size=2, stride=2)


def _compose_region(
    texture: torch.Tensor,
    box: tuple[int, int, int, int],
    tex_size: tuple[int, int],
    image: torch.Tensor,
    gain_full: torch.Tensor,
    relief_full: torch.Tensor,
    blend: float,
) -> torch.Tensor:
    """
    Spätná kompatibilita pre ploché regióny bez 3D geometrie.
    """
    tex_h, tex_w = tex_size
    gain_hr = _region_field(gain_full, box, tex_h, tex_w)
    relief_hr = _region_field(relief_full, box, tex_h, tex_w)
    img_hr = _region_field(image, box, tex_h, tex_w)
    albedo_hr = _paper_albedo(texture)
    lit01 = torch.clamp(albedo_hr * gain_hr * (0.10 + 0.90 * relief_hr) + 0.03 * gain_hr, 0.0, 1.0)
    lit = lit01 * 2.0 - 1.0
    comp = blend * lit + (1.0 - blend) * img_hr
    return comp.clamp(-1.0, 1.0)


def _build_canvas(image, region_specs, gain_full, relief_full, blend, boxes: dict | None = None):
    """
    Vloží všetky textúry do scény 112x112 s plným 3D ovinutím na nos a líca,
    aplikáciou lokálneho 3-kanálového osvetlenia miestnosti a Lambertovho 3D tieňovania.
    """
    canvas = image.clone()
    comps = {}
    device = image.device
    strip_geom = (boxes or {}).get("strip_geom")
    dot_geom_map = (boxes or {}).get("dot_geom") or {}
    r_dot_eff = float((boxes or {}).get("dot_radius_eff", 4.6))

    for key, (texture, box, tex_size) in region_specs.items():
        y1, y2, x1, x2 = box
        albedo_hr = _paper_albedo(texture)

        if key == "strip" and strip_geom is not None:
            albedo_box = _warp_flat_strip_to_box(albedo_hr, box, strip_geom, device)
        elif key in dot_geom_map:
            albedo_box = _warp_flat_dot_to_box(albedo_hr, box, dot_geom_map[key], r_dot_eff, device)
        else:
            albedo_box = F.interpolate(albedo_hr, size=(y2 - y1, x2 - x1), mode="bilinear", align_corners=False, antialias=True)

        gain_box = gain_full[:, :, y1:y2, x1:x2]
        relief_box = relief_full[:, :, y1:y2, x1:x2]
        img_box = image[:, :, y1:y2, x1:x2]

        # Multiplikatívna fotometria matného papiera bez dvojitého stmavovania + jemný odlesk rezanej hrany
        lit_box01 = torch.clamp(albedo_box * gain_box * relief_box, 0.0, 1.0)
        lit_box = lit_box01 * 2.0 - 1.0
        box_patch = (blend * lit_box + (1.0 - blend) * img_box).clamp(-1.0, 1.0)
        canvas[:, :, y1:y2, x1:x2] = box_patch
        comps[key] = (albedo_hr * 2.0 - 1.0).clamp(-1.0, 1.0)
    return canvas, comps


def _laplacian_pyramid_blend(
    bg: torch.Tensor,
    fg: torch.Tensor,
    mask: torch.Tensor,
    sigmas: tuple[float, float] = (1.4, 3.8),
    mask_sigmas: tuple[float, float, float] = (0.4, 1.3, 3.0),
) -> torch.Tensor:
    """
    Viacpásmové frekvenčné spájanie okrajov nálepky s pokožkou (Laplacian Pyramid Blending):
      - L0 (vysoké frekvencie: zrno, mikroskopické detaily tlače, ostrý rez papiera) -> maska M0 (úzky prechod).
      - L1 (stredné frekvencie: tiene hrán a gradienty) -> maska M1 (stredný prechod).
      - G2 (nízke frekvencie: difúzne okolité osvetlenie a odrazivosť) -> maska M2 (široký prechod).
    Zabezpečuje matematicky identickú rekonštrukciu pokožky mimo nálepky a prirodzený prechod bez viditeľného švu.
    """
    if bg.dim() == 3:
        bg = bg.unsqueeze(0)
    if fg.dim() == 3:
        fg = fg.unsqueeze(0)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    elif mask.dim() == 2:
        mask = mask.view(1, 1, mask.shape[0], mask.shape[1])

    s1, s2 = sigmas
    ms0, ms1, ms2 = mask_sigmas

    bg_g1 = _gblur_fixed(bg, s1)
    bg_g2 = _gblur_fixed(bg_g1, s2)
    bg_l0 = bg - bg_g1
    bg_l1 = bg_g1 - bg_g2
    bg_res = bg_g2

    fg_g1 = _gblur_fixed(fg, s1)
    fg_g2 = _gblur_fixed(fg_g1, s2)
    fg_l0 = fg - fg_g1
    fg_l1 = fg_g1 - fg_g2
    fg_res = fg_g2

    m0 = _gblur_fixed(mask, ms0) if ms0 > 0.08 else mask
    m1 = _gblur_fixed(mask, ms1)
    m2 = _gblur_fixed(mask, ms2)

    b0 = m0 * fg_l0 + (1.0 - m0) * bg_l0
    b1 = m1 * fg_l1 + (1.0 - m1) * bg_l1
    b_res = m2 * fg_res + (1.0 - m2) * bg_res

    return b0 + b1 + b_res


def composite_realistic_stickers(
    image: torch.Tensor,
    canvas: torch.Tensor,
    full_mask: torch.Tensor,
    boxes: dict | None = None,
) -> torch.Tensor:
    """
    Fotorealistická finálna kompozícia nálepiek do fotky („digitálny útok“):
      1. Jemný kontaktný tieň hrany papiera na pokožke (drop_shadow).
      2. Bod 5: Viacpásmové frekvenčné Laplacian Pyramid Blending pre prirodzený prechod hrany bez švu.
      3. Jemný svetlý lem nožnicami vystrihnutého 80g papiera.
      4. Kalibrované optické rozmazanie objektívu webkamery (Stage 2 PSF blur prispôsobený
         ostrosti fotky) cez nálepku a jej tesný okraj.
      5. Prenos senzorového mikro-zrna z pôvodnej fotky na povrch nálepky.
    Plne diferencovateľné pre optimalizáciu aj evaluáciu.
    """
    if full_mask.dim() == 3:
        full_mask = full_mask.unsqueeze(1)
    elif full_mask.dim() == 2:
        full_mask = full_mask.view(1, 1, full_mask.shape[0], full_mask.shape[1])

    b, _, h, _ = image.shape
    s = max(1.0, float(h) / 112.0)
    device = image.device

    drop_shadow = (boxes or {}).get("drop_shadow")
    if drop_shadow is not None:
        ds = drop_shadow.to(device)
        if ds.dim() == 3:
            ds = ds.unsqueeze(1)
        elif ds.dim() == 2:
            ds = ds.view(1, 1, ds.shape[0], ds.shape[1])
        img01 = torch.clamp(image * 0.5 + 0.5, 0.0, 1.0)
        img_shaded = torch.clamp(img01 * (1.0 - ds), 0.0, 1.0) * 2.0 - 1.0
    else:
        img_shaded = image

    # Jemný 0.5 px odlesk hrany vystrihnutého bieleho papiera
    edge_ring = torch.clamp(4.0 * full_mask * (1.0 - full_mask), 0.0, 1.0)
    canvas_rimmed = torch.clamp(canvas + 0.08 * edge_ring, -1.0, 1.0)

    # Bod 5: Laplacian Pyramid Blending spája nálepku s pokožkou po frekvenčných pásmach
    raw = _laplacian_pyramid_blend(
        img_shaded,
        canvas_rimmed,
        full_mask,
        sigmas=(1.4 * s, 3.8 * s),
        mask_sigmas=(0.4 * s, 1.3 * s, 3.0 * s),
    )

    img01_ref = torch.clamp(image * 0.5 + 0.5, 0.0, 1.0)
    ref_sig = torch.full((img01_ref.shape[0],), 1.15 * s, dtype=torch.float32, device=device)
    img_low = _random_gaussian_blur(img01_ref, ref_sig)
    hf_detail = (img01_ref - img_low).abs().mean(dim=(1, 2, 3)).detach()
    psf_sigma = torch.clamp((0.68 - 7.5 * hf_detail) * s, 0.46 * s, 0.78 * s)
    if psf_sigma.shape[0] != b:
        psf_sigma = psf_sigma.expand(b)

    blurred_raw = _random_gaussian_blur(raw, psf_sigma)

    # Rozmazanie aplikujeme len na nálepku a jej tesný prechod do pokožky (zvyšok tváre ostane 100 % nedotknutý)
    m_dil = F.max_pool2d(full_mask, kernel_size=3, stride=1, padding=1)
    blur_halo = F.avg_pool2d(m_dil, kernel_size=3, stride=1, padding=1)
    blur_weight = torch.clamp(blur_halo * 1.15, 0.0, 1.0)

    # Prirodzené senzorové zrno webkamery z pôvodnej fotky
    sensor_grain = (image - (img_low * 2.0 - 1.0)).detach()
    out = (1.0 - blur_weight) * raw + blur_weight * blurred_raw + full_mask * (0.36 * sensor_grain)
    return out.clamp(-1.0, 1.0)


def _dct_matrix(n: int, device) -> torch.Tensor:
    """Ortonormálna DCT-II matica n x n."""
    k = torch.arange(n, dtype=torch.float32, device=device).view(-1, 1)
    x = torch.arange(n, dtype=torch.float32, device=device).view(1, -1)
    d = torch.cos(math.pi * (2.0 * x + 1.0) * k / (2.0 * n)) * math.sqrt(2.0 / n)
    d[0] = d[0] * math.sqrt(0.5)
    return d


def _jpeg_approx(x: torch.Tensor, quality: float) -> torch.Tensor:
    """
    Diferencovateľná JPEG aproximácia: 8x8 bloková DCT-II + kvantizácia
    (round so straight-through gradientom) + inverzná DCT. Použitá namiesto
    kornia.jpeg_codec, aby bola simulácia print -> kamera -> JPEG reprodukovateľná
    bez ďalšej závislosti.
    """
    b = x.shape[0]
    c = x.shape[1]
    h, w = x.shape[-2], x.shape[-1]
    pad_h, pad_w = (-h) % 8, (-w) % 8
    xp = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate") if (pad_h or pad_w) else x
    x255 = (xp * 0.5 + 0.5) * 255.0
    hp, wp = x255.shape[-2], x255.shape[-1]
    blocks = x255.reshape(b, c, hp // 8, 8, wp // 8, 8).permute(0, 1, 2, 4, 3, 5).reshape(b, c, (hp // 8) * (wp // 8), 8, 8)
    d = _dct_matrix(8, x.device)
    coef = d @ blocks @ d.t()
    q = float(quality)
    scale = 5000.0 / max(q, 1.0) if q < 50.0 else 200.0 - 2.0 * q
    qstep = torch.clamp(torch.floor((_JPEG_LUMA_Q.to(x.device) * scale + 50.0) / 100.0), min=1.0)
    coef_q = coef + (torch.round(coef / qstep) * qstep - coef).detach()
    blocks_q = d.t() @ coef_q @ d
    rec = blocks_q.reshape(b, c, hp // 8, wp // 8, 8, 8).permute(0, 1, 2, 4, 3, 5).reshape(b, c, hp, wp)
    rec = torch.clamp(rec, 0.0, 255.0) / 255.0 * 2.0 - 1.0
    return rec[:, :, :h, :w]


def _random_gaussian_blur(x: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    """Per-sample gaussovské rozmazanie (sigma 0..2 px), diferencovateľné."""
    b, c, h, w = x.shape
    half = 6
    t = torch.arange(-half, half + 1, dtype=torch.float32, device=x.device)
    wts = torch.exp(-(t.view(1, -1) ** 2) / (2.0 * sigmas.view(-1, 1) ** 2 + 1e-6))
    wts = wts / wts.sum(dim=1, keepdim=True)
    xr = x.reshape(1, b * c, h, w)
    ky = wts.view(b, 1, -1, 1).expand(b, c, -1, 1).reshape(b * c, 1, 2 * half + 1, 1)
    kx = wts.view(b, 1, 1, -1).expand(b, c, 1, -1).reshape(b * c, 1, 1, 2 * half + 1)
    xr = F.conv2d(F.pad(xr, (0, 0, half, half), mode="replicate"), ky, groups=b * c)
    xr = F.conv2d(F.pad(xr, (half, half, 0, 0), mode="replicate"), kx, groups=b * c)
    return xr.reshape(b, c, h, w)


def _homography_from_corners(corners, device) -> torch.Tensor:
    """
    Homografia mapujúca jednotkový štvorec (0,0),(1,0),(1,1),(0,1) na zadané rohy
    (Heckbertova uzavretá forma). Slúži na simuláciu perspektívy z malých náhodných
    posunov rohov (nie iba afinná transformácia).
    """
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = corners
    dx1, dx2, dx3 = x1 - x2, x3 - x2, x0 - x1 + x2 - x3
    dy1, dy2, dy3 = y1 - y2, y3 - y2, y0 - y1 + y2 - y3
    if abs(dx3) < 1e-9 and abs(dy3) < 1e-9:
        g = h_p = 0.0
        a, b_, c_ = x1 - x0, x2 - x1, x0
        d_, e_, f_ = y1 - y0, y2 - y1, y0
    else:
        den = dx1 * dy2 - dx2 * dy1
        if abs(den) < 1e-9:
            den = 1e-9 if den >= 0.0 else -1e-9
        g = (dx3 * dy2 - dx2 * dy3) / den
        h_p = (dx1 * dy3 - dx3 * dy1) / den
        a, b_, c_ = x1 - x0 + g * x1, x3 - x0 + h_p * x3, x0
        d_, e_, f_ = y1 - y0 + g * y1, y3 - y0 + h_p * y3, y0
    return torch.tensor([[a, b_, c_], [d_, e_, f_], [g, h_p, 1.0]], dtype=torch.float32, device=device)


def _eot_grid(b: int, h: int, w: int, device) -> torch.Tensor:
    """
    EoT geometria print -> kamera -> align (výstup v [-1, 1] pre F.grid_sample):
      * perspektívna homografia z náhodných rohových offsetov (+-3.5 %),
      * centrovaná rotácia +-8 deg, translácia +-3 %, škála 0.92-1.08 okolo stredu tváre (0, 0),
      * jemný crop/alignment jitter celej tváre (+-2 px, +-2 deg, škála 0.97-1.03).
    """
    ys = torch.linspace(-1.0, 1.0, h, device=device)
    xs = torch.linspace(-1.0, 1.0, w, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    uv01 = torch.stack([(xx + 1.0) * 0.5, (yy + 1.0) * 0.5, torch.ones_like(xx)], dim=-1)
    grids = []
    for _ in range(b):
        base_corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        corners = [(cx + np.random.uniform(-0.035, 0.035), cy + np.random.uniform(-0.035, 0.035))
                   for (cx, cy) in base_corners]
        hmat = _homography_from_corners(corners, device)
        vh = uv01 @ hmat.t()
        den = vh[..., 2]
        den = torch.where(den.abs() < 1e-4, torch.full_like(den, 1e-4), den)
        x_m11 = (vh[..., 0] / den) * 2.0 - 1.0
        y_m11 = (vh[..., 1] / den) * 2.0 - 1.0
        xy1 = torch.stack([x_m11, y_m11, torch.ones_like(x_m11)], dim=-1)

        ang = np.random.uniform(-8.0, 8.0) * math.pi / 180.0
        sc = np.random.uniform(0.92, 1.08)
        tx, ty = np.random.uniform(-0.03, 0.03), np.random.uniform(-0.03, 0.03)
        aff = torch.tensor([
            [sc * math.cos(ang), -sc * math.sin(ang), tx],
            [sc * math.sin(ang), sc * math.cos(ang), ty],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float32, device=device)

        jpx = 2.0 / 56.0
        jang = np.random.uniform(-2.0, 2.0) * math.pi / 180.0
        jsc = np.random.uniform(0.97, 1.03)
        jx, jy = np.random.uniform(-jpx, jpx), np.random.uniform(-jpx, jpx)
        jit = torch.tensor([
            [jsc * math.cos(jang), -jsc * math.sin(jang), jx],
            [jsc * math.sin(jang), jsc * math.cos(jang), jy],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float32, device=device)

        m_center = jit @ aff
        vc = xy1 @ m_center.t()
        grids.append(vc[..., :2])
    return torch.stack(grids, dim=0)


def _patch_placement_jitter(canvas_r: torch.Tensor, mask_r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Simuluje reálnu nepresnosť ručného nalepenia nálepky pred zrkadlom/kamerou (+-1.2 px, +-2 deg, +-3% škála)."""
    b, _, h, w = canvas_r.shape
    device = canvas_r.device
    thetas = []
    for _ in range(b):
        ang = np.random.uniform(-2.0, 2.0) * math.pi / 180.0
        sc = np.random.uniform(0.97, 1.03)
        tx = np.random.uniform(-1.2, 1.2) / (w * 0.5)
        ty = np.random.uniform(-1.2, 1.2) / (h * 0.5)
        thetas.append([
            [sc * math.cos(ang), -sc * math.sin(ang), tx],
            [sc * math.sin(ang), sc * math.cos(ang), ty],
        ])
    theta = torch.tensor(thetas, dtype=torch.float32, device=device)
    p_grid = F.affine_grid(theta, canvas_r.size(), align_corners=False)
    c_j = F.grid_sample(canvas_r, p_grid, mode="bilinear", padding_mode="border", align_corners=False)
    m_j = F.grid_sample(mask_r, p_grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return c_j, torch.clamp(m_j, 0.0, 1.0)


def _camera_pipeline(x: torch.Tensor) -> torch.Tensor:
    """
    Fyzikálna degradácia snímania (print -> kamera), ale PROBABILISTICKÁ a mierna:
    každá degradácia sa aplikuje len na časť vzoriek, aby sa útok optimalizoval
    súčasne na čistý digitálny prípad AJ na zašumený print. Príliš agresívny
    vždy-zapnutý blur/JPEG inak vyruší adversariálnu perturbáciu (overené:
    EoT off -> sim 0.58, EoT on s tvrdou pipeline -> 0.78).
    """
    b = x.shape[0]
    device = x.device

    def gate(prob: float) -> torch.Tensor:
        return (torch.rand(b, 1, 1, 1, device=device) < prob).to(x.dtype)

    # Jemná medzisnímková variácia gamutu kamery (základný matný papier Canon TS3450 sa už
    # modeluje priamo na nálepke v _paper_albedo, takže tu neduplikujeme silnú desaturáciu tváre)
    luma = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    chroma_scale = 0.90 + torch.rand(b, 1, 1, 1, device=device) * 0.12
    x_desat = luma + (x - luma) * chroma_scale
    black_floor = -0.92 + (torch.rand(b, 1, 1, 1, device=device) * 2.0 - 1.0) * 0.05
    white_ceil = 0.92 + (torch.rand(b, 1, 1, 1, device=device) * 2.0 - 1.0) * 0.05
    x_inkjet = black_floor + (torch.clamp(x_desat, -1.0, 1.0) + 1.0) * 0.5 * (white_ceil - black_floor)
    x = x * (1.0 - gate(0.50)) + x_inkjet * gate(0.50)

    # Rozmazanie (sigma 0..1, len 60 % vzoriek)
    sigma = torch.rand(b, device=device) * 1.0
    x_blur = _random_gaussian_blur(x, sigma)
    x = x * (1.0 - gate(0.6)) + x_blur * gate(0.6)

    # Jas/kontrast (mierne)
    brightness = (torch.rand(b, 1, 1, 1, device=device) * 2.0 - 1.0) * 0.08
    contrast = 1.0 + (torch.rand(b, 1, 1, 1, device=device) * 2.0 - 1.0) * 0.10
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x_bc = torch.clamp((x - mean) * contrast + mean + brightness, -1.0, 1.0)
    x = x * (1.0 - gate(0.5)) + x_bc * gate(0.5)

    # Teplota farieb / white balance
    wb = 1.0 + (torch.rand(b, 3, 1, 1, device=device) * 2.0 - 1.0) * 0.05
    x = torch.clamp(x * wb, -1.0, 1.0)

    # Senzorový šum (len 50 %)
    noise = torch.randn_like(x) * (torch.rand(b, 1, 1, 1, device=device) * 0.015)
    x = x + noise * gate(0.5)

    # JPEG (kvalita 75-95, len 60 % vzoriek)
    quality = 75.0 + float(torch.rand(1).item()) * 20.0
    x_j = _jpeg_approx(x, quality)
    x = x * (1.0 - gate(0.6)) + x_j * gate(0.6)
    return x.clamp(-1.0, 1.0)


def _quantize_to_palette(tex: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    """Straight-through kvantizácia textúry na najbližšiu farbu z palety.

    tex: (1,C,H,W); palette: (K,C). Vzor tak používa len K farieb (logo/produkt).
    """
    c = tex.shape[1]
    h, w = tex.shape[-2], tex.shape[-1]
    flat = tex.squeeze(0).permute(1, 2, 0).reshape(-1, c)
    d = ((flat.unsqueeze(1) - palette.unsqueeze(0)) ** 2).sum(-1)
    idx = d.argmin(1)
    q = palette[idx].reshape(h, w, c).permute(2, 0, 1).unsqueeze(0)
    return tex + (q - tex).detach()


def get_hybrid_geometry(
    landmarks: list[tuple[float, float]],
    mesh_info: dict | None = None,
    img_size: int = 112,
    dot_radius: float = 5.0,
    dot_pattern: str = "cheeks_chin",
    strip_w_scale: float = 1.0,
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
    1. Športová páska Breathe Right na dolnej tretine nosa (bod 195/5, bezpečne pod očami).
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    has_mesh = mesh_info is not None and "points_3d" in mesh_info and len(mesh_info["points_3d"]) > 400
    pts = mesh_info["points_3d"] if has_mesh else None

    # Anatomické umiestnenie Breathe Right pásky – spoločný zdroj pravdy s live náhľadom v app.py
    # (páska sa ukotvuje na nosové krídla, nie na očný stred, a stáča sa spolu s nosom)
    strip_place = compute_nasal_strip_placement(
        landmarks, mesh_info,
        width_scale=float(strip_w_scale),
        offset_x=float(offset_x) + float(strip_ox),
        offset_y=float(offset_y) + float(strip_oy),
    )
    strip_cx = strip_place["cx"]
    strip_cy = strip_place["cy"]
    base_w = strip_place["base_w"]
    base_hc = strip_place["base_hc"]
    base_hw = strip_place["base_hw"]
    eff_w = strip_place["eff_w"]
    eff_hc = strip_place["eff_hc"]
    eff_hw = strip_place["eff_hw"]
    strip_angle = strip_place["angle"]
    eff_yaw = strip_place["yaw"]
    arch_f = strip_place["arch"]
    arch_px = strip_place["arch_px"]

    s_img = max(1.0, float(img_size) / 112.0)
    eff_feather = 0.50 if s_img <= 1.2 else 0.72

    scy_i, scx_i = int(round(strip_cy)), int(round(strip_cx))
    arch_drop_eff = max(0.0, float(arch_f) * float(arch_px))
    s_pad_top = int(math.ceil(eff_hw * 0.60)) + 3
    s_pad_bot = int(math.ceil(eff_hw * 0.65 + arch_drop_eff)) + 3
    s_pad_w = int(math.ceil(eff_w * 0.5)) + 3
    sy1, sy2 = max(0, scy_i - s_pad_top), min(img_size, scy_i + s_pad_bot)
    sx1, sx2 = max(0, scx_i - s_pad_w), min(img_size, scx_i + s_pad_w)
    strip_box = (sy1, sy2, sx1, sx2)

    mesh_uv_grid = None
    if has_mesh and pts is not None and len(pts) > 278:
        try:
            strip_mask, strip_relief, strip_shadow, strip_sheen, mesh_uv_grid = rasterize_barycentric_nasal_strip_mesh(
                pts_3d=pts,
                cx=strip_cx,
                cy=strip_cy,
                eff_w=eff_w,
                eff_hc=eff_hc,
                eff_hw=eff_hw,
                strip_angle=strip_angle,
                arch_px=arch_px,
                box=strip_box,
                img_size=img_size,
                device=device,
                feather=eff_feather,
            )
        except Exception:
            mesh_uv_grid = None

    if mesh_uv_grid is None:
        strip_mask, strip_relief, strip_shadow, strip_sheen = create_nasal_strip_mask(
            img_size=img_size,
            center=(strip_cy, strip_cx),
            width=eff_w,
            height_center=eff_hc,
            height_wings=eff_hw,
            angle_rad=strip_angle,
            yaw=eff_yaw,
            feather=eff_feather,
            arch_factor=arch_f,
            arch_drop_px=arch_px,
        )

    strip_flat_mask, _, _, _ = create_nasal_strip_mask(
        img_size=img_size,
        center=(strip_cy, strip_cx),
        width=eff_w,
        height_center=eff_hc,
        height_wings=eff_hw,
        angle_rad=strip_angle,
        yaw=eff_yaw,
        feather=eff_feather,
        arch_factor=0.0,
    )
    strip_mask = strip_mask.to(device)
    strip_flat_mask = strip_flat_mask.to(device)
    strip_relief = strip_relief.to(device)
    full_shadow = strip_shadow.to(device)

    boxes = {
        "strip": strip_box,
        "strip_angle": float(strip_angle),
        "strip_arch": float(arch_f),
        "strip_arch_px": float(arch_px),
        "strip_flat_mask": strip_flat_mask,
        "strip_geom": {
            "cx": float(strip_cx),
            "cy": float(strip_cy),
            "eff_w": float(eff_w),
            "eff_hc": float(eff_hc),
            "eff_hw": float(eff_hw),
            "angle": float(strip_angle),
            "yaw": float(eff_yaw),
            "arch_drop": float(arch_drop_eff),
            "box": strip_box,
            "mesh_uv_grid": mesh_uv_grid,
        },
    }

    # Nálepky na lícach
    r_dot = max(2.0, min(14.0, float(dot_radius)))

    # Ľavé líce (odmerané z reálnych fotiek s 15 mm nálepkami: medzi pts[101] a pts[205])
    if has_mesh and len(pts) > 205:
        p_l = pts[205]
        p_l_in = pts[101] if len(pts) > 101 else p_l
        lcx = float(0.65 * p_l[0] + 0.35 * p_l_in[0])
        lcy = float(0.70 * p_l[1] + 0.30 * p_l_in[1]) + 0.8
    else:
        lcx = lx - (nx - lx) * 0.04
        lcy = ly + (mly - ly) * 0.46

    lcx = float(np.clip(lcx + offset_x + left_ox, 6.0, 106.0))
    lcy = float(np.clip(lcy + offset_y + left_oy, 6.0, 106.0))

    if has_mesh:
        l_cos_raw, l_tilt, _ = get_surface_geometry_at_coords(pts, lcx, lcy, roll)
        l_cos = float(np.clip(l_cos_raw, 0.87, 0.96))
    else:
        l_cos, l_tilt = 0.90, roll

    l_mask, l_relief, l_sh, _ = create_elliptical_patch_mask(
        img_size=img_size,
        center=(lcy, lcx),
        radius_x=r_dot * l_cos,
        radius_y=r_dot,
        angle_rad=l_tilt,
        feather=eff_feather,
    )
    l_mask, l_relief = l_mask.to(device), l_relief.to(device)
    full_shadow = torch.maximum(full_shadow, l_sh.to(device) * 0.48)

    d_pad = int(math.ceil(r_dot)) + 3
    lcy_i, lcx_i = int(round(lcy)), int(round(lcx))
    ly1, ly2 = max(0, lcy_i - d_pad), min(img_size, lcy_i + d_pad)
    lx1, lx2 = max(0, lcx_i - d_pad), min(img_size, lcx_i + d_pad)
    boxes["left_dot"] = (ly1, ly2, lx1, lx2)
    boxes["foreshortening_left"] = float(l_cos)
    boxes.setdefault("dot_geom", {})["left_dot"] = (lcx, lcy, float(l_cos), float(l_tilt))

    full_mask = torch.maximum(strip_mask, l_mask)
    full_relief = torch.ones_like(full_mask)
    full_relief = torch.where(strip_mask > 0.1, strip_relief, full_relief)
    full_relief = torch.where(l_mask > 0.1, torch.clamp(l_relief, 0.90, 1.02), full_relief)

    # Pravé líce (odmerané z reálnych fotiek s 15 mm nálepkami: medzi pts[330] a pts[425])
    if "1_cheek" not in dot_pattern and "1 nálepka" not in dot_pattern:
        if has_mesh and len(pts) > 425:
            p_r = pts[425]
            p_r_in = pts[330] if len(pts) > 330 else p_r
            rcx = float(0.65 * p_r[0] + 0.35 * p_r_in[0])
            rcy = float(0.70 * p_r[1] + 0.30 * p_r_in[1]) + 0.8
        else:
            rcx = rx + (rx - nx) * 0.04
            rcy = ry + (mry - ry) * 0.46

        rcx = float(np.clip(rcx + offset_x + right_ox, 6.0, 106.0))
        rcy = float(np.clip(rcy + offset_y + right_oy, 6.0, 106.0))

        if has_mesh:
            r_cos_raw, r_tilt, _ = get_surface_geometry_at_coords(pts, rcx, rcy, roll)
            r_cos = float(np.clip(r_cos_raw, 0.87, 0.96))
        else:
            r_cos, r_tilt = 0.90, roll

        r_mask, r_relief, r_sh, _ = create_elliptical_patch_mask(
            img_size=img_size,
            center=(rcy, rcx),
            radius_x=r_dot * r_cos,
            radius_y=r_dot,
            angle_rad=r_tilt,
            feather=eff_feather,
        )
        r_mask, r_relief = r_mask.to(device), r_relief.to(device)
        full_shadow = torch.maximum(full_shadow, r_sh.to(device) * 0.48)

        rcy_i, rcx_i = int(round(rcy)), int(round(rcx))
        ry1, ry2 = max(0, rcy_i - d_pad), min(img_size, rcy_i + d_pad)
        rx1, rx2 = max(0, rcx_i - d_pad), min(img_size, rcx_i + d_pad)
        boxes["right_dot"] = (ry1, ry2, rx1, rx2)
        boxes["foreshortening_right"] = float(r_cos)
        boxes.setdefault("dot_geom", {})["right_dot"] = (rcx, rcy, float(r_cos), float(r_tilt))

        full_mask = torch.maximum(full_mask, r_mask)
        full_relief = torch.where(r_mask > 0.1, torch.clamp(r_relief, 0.90, 1.02), full_relief)

    # Brada (posunutá o -3 px vyššie do labiomentálnej ryhy, aby sa neorezávala o spodný okraj)
    if "brada" in dot_pattern.lower() or "chin" in dot_pattern.lower() or "3 nálepk" in dot_pattern:
        if has_mesh and len(pts) > 175:
            p_c = pts[175]
            ccx, ccy = float(p_c[0]), float(p_c[1]) - 3.0
        else:
            ccx = (mlx + mrx) * 0.5
            ccy = max(mly, mry) + (ny - (ly + ry) * 0.5) * 0.42 - 3.0

        ccx = float(np.clip(ccx + offset_x + chin_ox, 6.0, 106.0))
        ccy = float(np.clip(ccy + offset_y + chin_oy, 6.0, 103.0))

        if has_mesh:
            c_cos, c_tilt, _ = get_surface_geometry_at_coords(pts, ccx, ccy, roll)
        else:
            c_cos, c_tilt = 0.88, roll

        c_mask, c_relief, c_sh, _ = create_elliptical_patch_mask(
            img_size=img_size,
            center=(ccy, ccx),
            radius_x=r_dot * c_cos,
            radius_y=r_dot,
            angle_rad=c_tilt,
            feather=0.85,
        )
        c_mask, c_relief = c_mask.to(device), c_relief.to(device)
        full_shadow = torch.maximum(full_shadow, c_sh.to(device) * 0.75)

        ccy_i, ccx_i = int(round(ccy)), int(round(ccx))
        cy1, cy2 = max(0, ccy_i - d_pad), min(img_size, ccy_i + d_pad)
        cx1, cx2 = max(0, ccx_i - d_pad), min(img_size, ccx_i + d_pad)
        boxes["chin_dot"] = (cy1, cy2, cx1, cx2)
        boxes["foreshortening_chin"] = float(c_cos)
        boxes.setdefault("dot_geom", {})["chin_dot"] = (ccx, ccy, float(c_cos), float(c_tilt))

        full_mask = torch.maximum(full_mask, c_mask)
        full_relief = torch.where(c_mask > 0.1, c_relief, full_relief)
        boxes["chin_mask"] = c_mask
        boxes["chin_relief"] = c_relief

    full_mask = torch.clamp(full_mask, 0.0, 1.0)
    boxes["drop_shadow"] = torch.clamp(full_shadow * (1.0 - full_mask), 0.0, 0.25)
    boxes["strip_mask"] = strip_mask
    boxes["strip_relief"] = strip_relief
    boxes["strip_w_scale"] = float(strip_w_scale)
    boxes["dot_radius_eff"] = float(r_dot)
    boxes["left_mask"] = l_mask
    boxes["left_relief"] = l_relief
    if "right_mask" not in boxes and "1_cheek" not in dot_pattern and "1 nálepka" not in dot_pattern:
        boxes["right_mask"] = r_mask
        boxes["right_relief"] = r_relief

    return full_mask, full_relief, boxes


def generate_printable_sheet(
    patches: dict,
    output_path: str | None = None,
    mm_per_px: float | None = None,
    landmarks=None,
    masks: dict | None = None,
    geometry: dict | None = None,
    max_sets: int = 3,
) -> np.ndarray:
    """
    Generuje tlačový hárok (300 DPI) v presnej mierke (100% Actual Size).

    Fyzická velkosť každého prvku = px rozmer textúry/boxu * mm_per_px
    (žiadne hardcoded 32x12 mm / 10 mm). Predvolené mm_per_px sa odvodí z IPD
    63 mm a nameranej vzdialenosti zreníc útočenej tváre; fallback 0.56 mm/px.

    masks:    {key: (H, W) pole v [0,1]} - presný obrys (alpha) vzoru, aby
              používateľ strihal po skutočnom tvare a nie po obdĺžniku.
    geometry: {key: {"mm_w", "mm_h", "unsquash", "rotate_deg", "label"}},
              kde "unsquash" kompenzuje perspektívne skrátenie (1/l_cos, resp.
              1/r_cos) - elipsa zošliapnutá v čelnom crop-e sa vytlačí ako
              kruh zamýšľaného reálneho priemeru.
    """
    DPI = 300
    MM_TO_PX = DPI / 25.4
    masks = masks or {}
    geometry = geometry or {}

    # --- Mierka: mm/px z IPD (63 mm) a vzdialenosti zreníc v 112x112 crop-e ---
    if mm_per_px is None:
        mm_per_px = 0.56
        try:
            if landmarks is not None and len(landmarks) >= 2:
                ipd_px = math.hypot(float(landmarks[1][0]) - float(landmarks[0][0]),
                                    float(landmarks[1][1]) - float(landmarks[0][1]))
                if ipd_px > 2.0:
                    mm_per_px = 63.0 / ipd_px
        except Exception:
            mm_per_px = 0.56
    mm_per_px = float(mm_per_px)

    def _render_flat_dot(key, rgb_tex, geo, flat):
        """Textúra kruhovej nálepky je už natívne parametrizovaná v plochých papierových súradniciach."""
        r_img = float(flat["r"])
        d_mm = 2.0 * r_img * mm_per_px
        side = max(8, int(round(d_mm * MM_TO_PX)))
        flat_rgb = cv2.resize(rgb_tex, (side, side), interpolation=cv2.INTER_CUBIC)
        c = (side - 1) / 2.0
        jj, ii = np.meshgrid(np.arange(side, dtype=np.float32), np.arange(side, dtype=np.float32))
        dist = np.sqrt((jj - c) ** 2 + (ii - c) ** 2) / max(1.0, side * 0.5)
        alpha = (np.clip((1.0 - dist) * 16.0 + 0.5, 0.0, 1.0) * 255.0).astype(np.uint8)
        return {
            "key": key,
            "rgb": np.ascontiguousarray(flat_rgb),
            "alpha": np.ascontiguousarray(alpha),
            "mm_w": d_mm,
            "mm_h": d_mm,
            "label": geo.get("label", key),
        }

    def _render_region(key: str):
        tensor = patches.get(key)
        if tensor is None and key == "strip":
            tensor = patches.get("bandage")
        if tensor is None:
            return None
        geo = geometry.get(key, {}) or {}
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        p = tensor.detach().permute(1, 2, 0).cpu().numpy()
        rgb = np.clip((p + 1.0) * 0.5 * 255.0, 0, 255).astype(np.uint8)
        m = masks.get(key) if masks else None
        if m is not None:
            m = np.asarray(m, dtype=np.float32)
            if m.ndim == 3:
                m = m[0]
            if m.shape[:2] != rgb.shape[:2]:
                m = cv2.resize(m, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
            alpha = (np.clip(m, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            alpha = np.full(rgb.shape[:2], 255, dtype=np.uint8)

        flat = geo.get("flat")
        if flat is not None:
            return _render_flat_dot(key, rgb, geo, flat)

        # Kompenzácia foreshorteningu (1/l_cos, resp. 1/r_cos) pred fyzickou mierkou
        unsquash = float(geo.get("unsquash", 1.0))
        if abs(unsquash - 1.0) > 1e-3:
            nw = max(1, int(round(rgb.shape[1] * unsquash)))
            rgb = cv2.resize(rgb, (nw, rgb.shape[0]), interpolation=cv2.INTER_LANCZOS4)
            alpha = cv2.resize(alpha, (nw, alpha.shape[0]), interpolation=cv2.INTER_LANCZOS4)

        # Fyzická veľkosť = px rozmer * mm_per_px (tlač 1:1)
        mm_w = float(geo.get("mm_w", rgb.shape[1] * mm_per_px))
        mm_h = float(geo.get("mm_h", rgb.shape[0] * mm_per_px))
        tw = max(2, int(round(mm_w * MM_TO_PX)))
        th = max(2, int(round(mm_h * MM_TO_PX)))
        rgb = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_CUBIC)
        alpha = cv2.resize(alpha, (tw, th), interpolation=cv2.INTER_CUBIC)

        # Aplikácia strip_angle na finálny renderovaný vzor na hárku
        rot_deg = float(geo.get("rotate_deg", 0.0))
        if abs(rot_deg) > 1e-3:
            diag = int(math.ceil(math.hypot(th, tw))) + 2
            pad = np.zeros((diag, diag, 4), dtype=np.uint8)
            y0, x0 = (diag - th) // 2, (diag - tw) // 2
            pad[y0:y0 + th, x0:x0 + tw, :3] = rgb
            pad[y0:y0 + th, x0:x0 + tw, 3] = alpha
            rm = cv2.getRotationMatrix2D((diag / 2.0, diag / 2.0), rot_deg, 1.0)
            rot = cv2.warpAffine(pad, rm, (diag, diag), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
            rgb, alpha = rot[:, :, :3], rot[:, :, 3]

        # Orezanie prázdneho priehľadného okraja boxu a výpočet skutočného rozmeru masky v mm
        ys_nz, xs_nz = np.nonzero(alpha > 20)
        if len(ys_nz) > 0 and len(xs_nz) > 0:
            ay1 = max(0, int(ys_nz.min()))
            ay2 = min(alpha.shape[0], int(ys_nz.max()) + 1)
            ax1 = max(0, int(xs_nz.min()))
            ax2 = min(alpha.shape[1], int(xs_nz.max()) + 1)
            mm_w = float((int(xs_nz.max()) - int(xs_nz.min()) + 1) / MM_TO_PX)
            mm_h = float((int(ys_nz.max()) - int(ys_nz.min()) + 1) / MM_TO_PX)
            rgb = rgb[ay1:ay2, ax1:ax2]
            alpha = alpha[ay1:ay2, ax1:ax2]

        return {
            "key": key,
            "rgb": np.ascontiguousarray(rgb),
            "alpha": np.ascontiguousarray(alpha),
            "mm_w": mm_w,
            "mm_h": mm_h,
            "label": geo.get("label", key),
        }

    items = []
    for key in ["strip", "left_dot", "right_dot", "chin_dot"]:
        rendered = _render_region(key)
        if rendered is not None:
            items.append(rendered)

    # --- Layout: pevny format A4 (210 x 297 mm), tlac 1:1 ---
    sheet_w_mm, sheet_h_mm = 210.0, 297.0
    sheet_w = int(round(sheet_w_mm * MM_TO_PX))
    sheet_h = int(round(sheet_h_mm * MM_TO_PX))
    sheet = np.ones((sheet_h, sheet_w, 3), dtype=np.uint8) * 255

    strip_geo = geometry.get("strip", {}) or {}
    strip_w_sc = float(strip_geo.get("strip_w_scale", 1.0))
    strip_scale_ratio = float(np.clip(strip_w_sc, 0.75, 1.35))
    eff_strip_len_mm = STRIP_LEN_MM * strip_scale_ratio
    eff_strip_waist_mm = STRIP_WAIST_MM * strip_scale_ratio
    eff_strip_wings_mm = STRIP_WINGS_MM * strip_scale_ratio

    dot_geo_any = geometry.get("left_dot") or geometry.get("right_dot") or {}
    r_dot_eff_val = float(dot_geo_any.get("dot_radius_eff", 5.0))
    eff_dot_diam_mm = DOT_DIAM_MM * float(np.clip(r_dot_eff_val / 5.0, 0.65, 1.50))

    # Kalibračné meradlo (50 mm)
    bar_x, bar_y = 100, 120
    bar_len = int(round(50.0 * MM_TO_PX))
    cv2.line(sheet, (bar_x, bar_y), (bar_x + bar_len, bar_y), (0, 0, 0), 3)
    cv2.line(sheet, (bar_x, bar_y - 14), (bar_x, bar_y + 14), (0, 0, 0), 3)
    cv2.line(sheet, (bar_x + bar_len, bar_y - 14), (bar_x + bar_len, bar_y + 14), (0, 0, 0), 3)
    cv2.putText(sheet, "50.0 mm (100% mierka)", (bar_x + 15, bar_y - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(
        sheet,
        f"Paska: {eff_strip_len_mm:.1f}x{eff_strip_wings_mm:.1f} mm | Nalepky: {eff_dot_diam_mm:.1f} mm | Lave lice = pod tvoje vlastne lave oko",
        (bar_x + bar_len + 45, bar_y - 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA,
    )
    cv2.putText(
        sheet,
        "TIP: Neprelepuj vrch lesklou paskou (odlesk z monitora). Lep zospodu obojstrannou paskou alebo tycinkovym lepidlom.",
        (bar_x, bar_y + 38),
        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (90, 90, 90), 1, cv2.LINE_AA,
    )

    def _canon_ts3450_precompensate(rgb_u8: np.ndarray) -> np.ndarray:
        """Pred-kompenzácia pre atramentovú tlačiareň Canon PIXMA TS3450 (PG-545 / CL-546)
        na bežný A4 papier: S-krivka kontrastu + 1.18x sýtosť + jemné doostrenie hrán."""
        x = rgb_u8.astype(np.float32) / 255.0
        # 1. S-krivka kontrastu proti vsakovaniu atramentu do papiera
        x_centered = (x - 0.5) * 1.16
        x_scurve = np.clip(0.5 + x_centered / np.sqrt(1.0 + 0.32 * (x_centered ** 2)), 0.0, 1.0)
        # 2. Zvýšenie farebnej sýtosti (kompenzácia CMY dye desaturácie na matnom papieri)
        luma = 0.299 * x_scurve[..., 0:1] + 0.587 * x_scurve[..., 1:2] + 0.114 * x_scurve[..., 2:3]
        x_sat = np.clip(luma + (x_scurve - luma) * 1.18, 0.0, 1.0)
        out_u8 = np.clip(x_sat * 255.0, 0, 255).astype(np.uint8)
        # 3. Jemný Unsharp Mask proti rozpíjaniu kvapiek atramentu (dot gain)
        blur = cv2.GaussianBlur(out_u8, (0, 0), sigmaX=1.0)
        return cv2.addWeighted(out_u8, 1.22, blur, -0.22, 0)

    def _to_rgb(t):
        if isinstance(t, np.ndarray):
            raw = t if t.dtype == np.uint8 else np.clip(t * 255.0, 0, 255).astype(np.uint8)
            return _canon_ts3450_precompensate(raw)
        if isinstance(t, torch.Tensor):
            p = t.detach().squeeze()
            if p.dim() == 3:
                p = p.permute(1, 2, 0)
            p_np = p.cpu().numpy()
            if p_np.min() < -0.05:
                raw = np.clip((p_np + 1.0) * 0.5 * 255.0, 0, 255).astype(np.uint8)
            else:
                raw = np.clip(p_np * 255.0, 0, 255).astype(np.uint8)
            return _canon_ts3450_precompensate(raw)
        return np.zeros((96, 96, 3), dtype=np.uint8)

    def _extract_active_strip_tex(tex_rgb, out_w=None, out_h=None):
        flat_s = (geometry.get("strip") or {}).get("flat_strip")
        if flat_s is not None and out_w is not None and out_h is not None:
            # Textúra pásky (96x256) je optimalizovaná priamo v plochých papierových súradniciach
            # a počas forward passu sa 3D ovíja na nos cez _warp_flat_strip_to_box -> 1:1 bez orezu!
            return cv2.resize(tex_rgb, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

        m = masks.get("strip") if masks else None
        if m is None:
            return tex_rgb
        m_arr = np.asarray(m, dtype=np.float32)
        if m_arr.ndim == 3:
            m_arr = m_arr[0]
        if m_arr.shape[:2] != tex_rgb.shape[:2]:
            m_arr = cv2.resize(m_arr, (tex_rgb.shape[1], tex_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        rot_deg = float((geometry.get("strip") or {}).get("rotate_deg", 0.0))
        work_rgb = tex_rgb
        work_m = m_arr
        if abs(rot_deg) > 0.25:
            h0, w0 = work_rgb.shape[:2]
            rm = cv2.getRotationMatrix2D((w0 / 2.0, h0 / 2.0), rot_deg, 1.0)
            work_rgb = cv2.warpAffine(work_rgb, rm, (w0, h0), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            work_m = cv2.warpAffine(work_m, rm, (w0, h0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        ys_nz, xs_nz = np.nonzero(work_m > 0.14)
        if len(ys_nz) > 4 and len(xs_nz) > 4:
            ay1, ay2 = int(ys_nz.min()), int(ys_nz.max()) + 1
            ax1, ax2 = int(xs_nz.min()), int(xs_nz.max()) + 1
            return np.ascontiguousarray(work_rgb[ay1:ay2, ax1:ax2])
        return work_rgb

    def _extract_active_dot_tex(tex_rgb, key, D):
        flat = (geometry.get(key) or {}).get("flat")
        if flat is not None:
            return cv2.resize(tex_rgb, (D, D), interpolation=cv2.INTER_CUBIC)
        m = masks.get(key) if masks else None
        if m is not None:
            m_arr = np.asarray(m, dtype=np.float32)
            if m_arr.ndim == 3:
                m_arr = m_arr[0]
            if m_arr.shape[:2] != tex_rgb.shape[:2]:
                m_arr = cv2.resize(m_arr, (tex_rgb.shape[1], tex_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
            ys_nz, xs_nz = np.nonzero(m_arr > 0.14)
            if len(ys_nz) > 4 and len(xs_nz) > 4:
                tex_rgb = tex_rgb[int(ys_nz.min()):int(ys_nz.max()) + 1, int(xs_nz.min()):int(xs_nz.max()) + 1]
        return cv2.resize(tex_rgb, (D, D), interpolation=cv2.INTER_CUBIC)

    def _draw_exact_strip(tex_rgb, w_mm=STRIP_LEN_MM, hc_mm=STRIP_WAIST_MM,
                          hw_mm=STRIP_WINGS_MM, x_pos=100, y_pos=260):
        W = int(round(w_mm * MM_TO_PX))
        Hc = int(round(hc_mm * MM_TO_PX))
        Hw = int(round(hw_mm * MM_TO_PX))
        R_x = int(round(8.5 * (w_mm / STRIP_LEN_MM) * MM_TO_PX))
        R_y = Hw // 2

        canvas = np.zeros((Hw, W), dtype=np.uint8)
        cx = W // 2
        cy = Hw // 2
        half_w = W // 2

        x_limit = max(4, half_w - R_x)
        xs = np.linspace(-x_limit, x_limit, 300)

        pts_top = []
        pts_bot = []
        for x in xs:
            u = abs(x / x_limit)
            h_local = Hc + (Hw - Hc) * (u ** 1.75)
            pts_top.append((int(round(cx + x)), int(round(cy - h_local * 0.5))))
            pts_bot.append((int(round(cx + x)), int(round(cy + h_local * 0.5))))

        pts_all = []
        pts_all.extend(pts_top)
        for a in np.linspace(-math.pi / 2, math.pi / 2, 50):
            pts_all.append((int(round(cx + x_limit + R_x * math.cos(a))), int(round(cy + R_y * math.sin(a)))))
        pts_all.extend(pts_bot[::-1])
        for a in np.linspace(math.pi / 2, 3 * math.pi / 2, 50):
            pts_all.append((int(round(cx - x_limit + R_x * math.cos(a))), int(round(cy + R_y * math.sin(a)))))

        pts_arr = np.array(pts_all, dtype=np.int32)
        cv2.fillPoly(canvas, [pts_arr], 255)

        active_strip_rgb = _extract_active_strip_tex(tex_rgb, out_w=W, out_h=Hw)
        tex_res = active_strip_rgb if active_strip_rgb.shape[:2] == (Hw, W) else cv2.resize(active_strip_rgb, (W, Hw), interpolation=cv2.INTER_CUBIC)

        a_norm = (canvas.astype(np.float32) / 255.0)[:, :, np.newaxis]
        target_roi = sheet[y_pos:y_pos + Hw, x_pos:x_pos + W].astype(np.float32)
        blended = tex_res.astype(np.float32) * a_norm + target_roi * (1.0 - a_norm)
        sheet[y_pos:y_pos + Hw, x_pos:x_pos + W] = np.clip(blended, 0, 255).astype(np.uint8)

        poly_sheet = pts_arr + np.array([x_pos, y_pos])
        cv2.polylines(sheet, [poly_sheet], True, (60, 60, 60), 2, cv2.LINE_AA)

        arr_cx = x_pos + W // 2
        arr_top = y_pos - 16
        cv2.fillPoly(sheet, [np.array([[arr_cx, arr_top - 16], [arr_cx - 11, arr_top], [arr_cx + 11, arr_top]], dtype=np.int32)], (0, 140, 0))
        cv2.putText(sheet, "HORE (k ociam)", (arr_cx + 16, arr_top - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 130, 0), 1, cv2.LINE_AA)
        return W, Hw

    def _draw_exact_circle(tex_rgb, key="", d_mm=15.0, x_pos=100, y_pos=260, label=""):
        D = int(round(d_mm * MM_TO_PX))
        R = D // 2
        mask_c = np.zeros((D, D), dtype=np.uint8)
        cv2.circle(mask_c, (R, R), R, 255, -1, cv2.LINE_AA)

        tex_res = _extract_active_dot_tex(tex_rgb, key, D) if key else cv2.resize(tex_rgb, (D, D), interpolation=cv2.INTER_CUBIC)

        a_norm = (mask_c.astype(np.float32) / 255.0)[:, :, np.newaxis]
        target_roi = sheet[y_pos:y_pos + D, x_pos:x_pos + D].astype(np.float32)
        blended = tex_res.astype(np.float32) * a_norm + target_roi * (1.0 - a_norm)
        sheet[y_pos:y_pos + D, x_pos:x_pos + D] = np.clip(blended, 0, 255).astype(np.uint8)

        cv2.circle(sheet, (x_pos + R, y_pos + R), R, (60, 60, 60), 2, cv2.LINE_AA)
        # Jemná zelená šípka HORE na hornom okraji kruhu pre presnú rotáciu na líci
        top_cx = x_pos + R
        top_cy = y_pos - 5
        cv2.fillPoly(sheet, [np.array([[top_cx, top_cy - 10], [top_cx - 7, top_cy], [top_cx + 7, top_cy]], dtype=np.int32)], (0, 140, 0))
        if label:
            cv2.putText(sheet, label, (max(20, x_pos + R - 62), y_pos - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (50, 50, 50), 1, cv2.LINE_AA)
        return D, D

    # Vykreslenie presných tvarov podľa reálnych rozmerov (2 sady vedľa seba / pod sebou na A4)
    has_strip = ("strip" in patches) or ("bandage" in patches)
    if has_strip:
        s_tex = _to_rgb(patches.get("strip", patches.get("bandage")))
        has_l = "left_dot" in patches
        has_r = "right_dot" in patches
        has_c = "chin_dot" in patches
        l_tex = _to_rgb(patches["left_dot"]) if has_l else None
        r_tex = _to_rgb(patches["right_dot"]) if has_r else None
        c_tex = _to_rgb(patches["chin_dot"]) if has_c else None

        rows = [
            ("SADA 1 (hlavna)", 280),
            ("SADA 2 (rezerva)", 760),
            ("SADA 3 (rezerva)", 1240),
        ][:max(1, int(max_sets))]
        for name, y_row in rows:
            cv2.putText(sheet, name, (100, y_row - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (40, 40, 40), 1, cv2.LINE_AA)
            cx = 100
            w_strip, h_strip = _draw_exact_strip(
                s_tex, eff_strip_len_mm, eff_strip_waist_mm, eff_strip_wings_mm, cx, y_row
            )
            cx += w_strip + 90
            d_dot = int(round(eff_dot_diam_mm * MM_TO_PX))
            dot_y = y_row + (h_strip - d_dot) // 2
            # Z tvojho anatomického pohľadu: right_dot = pod tvoje ľavé oko, left_dot = pod tvoje pravé oko
            if r_tex is not None:
                d1, _ = _draw_exact_circle(r_tex, "right_dot", eff_dot_diam_mm, cx, dot_y, "Tvoje LAVE lice")
                cx += d1 + 75
            if l_tex is not None:
                d2, _ = _draw_exact_circle(l_tex, "left_dot", eff_dot_diam_mm, cx, dot_y, "Tvoje PRAVE lice")
                cx += d2 + 75
            if c_tex is not None:
                _draw_exact_circle(c_tex, "chin_dot", eff_dot_diam_mm, cx, dot_y, "Tvoja BRADA")
    else:
        cx, cy = 100, 280
        for it in items:
            _draw_exact_circle(it["rgb"], "", eff_dot_diam_mm, cx, cy, it["key"])
            cx += int(round((eff_dot_diam_mm + 10.0) * MM_TO_PX))

    sheet_bgr = cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)
    if output_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            cv2.imwrite(output_path, sheet_bgr)
        except Exception:
            pass

    ws_copy = "printable_patches/printable_patch_sheet_last.png"
    try:
        os.makedirs("printable_patches", exist_ok=True)
        cv2.imwrite(ws_copy, sheet_bgr)
    except Exception:
        pass

    print("[PRINTABLE] mm_per_px=%.4f sheet=%.1fx%.1fmm | vykreslené: páska=%.1fx%.1fmm "
          "(stred %.1fmm), nálepky=%.1fmm kruhy | medzivýpočet: %s"
          % (mm_per_px, sheet_w_mm, sheet_h_mm, eff_strip_len_mm, eff_strip_wings_mm, eff_strip_waist_mm,
             eff_dot_diam_mm,
             " ".join("%s=%.2fx%.2fmm" % (it["key"], it["mm_w"], it["mm_h"]) for it in items)))

    return sheet


def derive_mm_per_px(landmarks, default: float = 0.56) -> float:
    """Odvodí mm/px z reálnej interpupilárnej vzdialenosti (IPD ~63 mm).

    landmarks: zoznam (x, y), alebo (landmarks, mesh_info), alebo None.
    Fallback na `default`, ak sa IPD nedá zmerať.
    """
    lm = landmarks
    if isinstance(lm, tuple):
        lm = lm[0]
    try:
        ipd_px = math.hypot(float(lm[1][0]) - float(lm[0][0]),
                            float(lm[1][1]) - float(lm[0][1]))
        if ipd_px > 2.0:
            return 63.0 / ipd_px
    except Exception:
        pass
    return float(default)


def build_printable_payload(boxes: dict, spec_items: dict, mm_per_px: float):
    """Zostaví (masks_np, geo) pre `generate_printable_sheet` z `boxes` a `final_specs`.

    spec_items: {key: (box, tex_size)} kde box=(y1,y2,x1,x2), tex_size=(H,W).
    Zabezpečuje rovnakú fyzickú mierku (mm) a alpha obrys pre každý nosič.
    """
    r_dot_eff = float(boxes.get("dot_radius_eff", 5.5))
    strip_w_scale = float(boxes.get("strip_w_scale", 1.15))
    strip_angle = float(boxes.get("strip_angle", 0.0))
    cos_by_key = {
        "left_dot": float(boxes.get("foreshortening_left", 0.85)),
        "right_dot": float(boxes.get("foreshortening_right", 0.85)),
        "chin_dot": float(boxes.get("foreshortening_chin", 0.88)),
    }
    mask_keys = {"strip": "strip_mask", "left_dot": "left_mask",
                 "right_dot": "right_mask", "chin_dot": "chin_mask"}
    labels = {"strip": "1. Breathe Right páska na nos", "right_dot": "2. Tvoje ĽAVÉ líce (pod ľavé oko)",
              "left_dot": "3. Tvoje PRAVÉ líce (pod pravé oko)", "chin_dot": "4. Brada"}
    masks_np, geo = {}, {}
    for key, (box, tex_size) in spec_items.items():
        y1, y2, x1, x2 = box
        if key == "strip" and "strip_mask" in boxes and "strip_flat_mask" in boxes:
            m = torch.maximum(boxes["strip_mask"], boxes["strip_flat_mask"])
        else:
            m = boxes.get(mask_keys.get(key))
        if m is not None:
            masks_np[key] = _region_field(m, box, tex_size[0], tex_size[1], pad=0).squeeze().cpu().numpy()
        if key == "strip":
            geo[key] = {"mm_w": (x2 - x1) * mm_per_px * 1.14, "mm_h": (y2 - y1) * mm_per_px,
                        "unsquash": 1.14, "rotate_deg": -math.degrees(strip_angle),
                        "strip_w_scale": strip_w_scale,
                        "flat_strip": boxes.get("strip_geom"),
                        "label": labels[key]}
        else:
            geo[key] = {"mm_w": 2.0 * r_dot_eff * mm_per_px, "mm_h": 2.0 * r_dot_eff * mm_per_px,
                        "unsquash": 1.0 / max(0.45, cos_by_key.get(key, 0.85)),
                        "rotate_deg": 0.0,
                        "dot_radius_eff": r_dot_eff,
                        "label": labels[key]}
            dg = (boxes.get("dot_geom") or {}).get(key)
            if dg is not None:
                geo[key]["flat"] = {"cx": dg[0], "cy": dg[1], "cos": dg[2], "tilt": dg[3],
                                    "r": r_dot_eff, "box": tuple(box), "tex": tuple(tex_size)}
    return masks_np, geo


def generate_emblem_masks(
    th_s: int = TEX_STRIP_SIZE[0],
    tw_s: int = TEX_STRIP_SIZE[1],
    th_d: int = TEX_DOT_SIZE[0],
    tw_d: int = TEX_DOT_SIZE[1],
    style: str = "chevron_stars",
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Vygeneruje ostro vyhladené (4x supersampled) vektorové masky loga pre nosnú pásku
    (th_s, tw_s) a nálepky na líca/bradu (th_d, tw_d).
    Rozmery sú striktne vnútri aktívnej oblasti masky (zúžený stred pásky +-14 px,
    aktívny kruh nálepky r~25 px), takže logo sa nikdy neořeže a zaberá < 10 % plochy,
    čím zachováva > 90 % plnej adversariálnej kapacity pozadia.
    """
    s_lower = str(style).lower()
    if any(k in s_lower for k in ["vypnut", "bez_loga", "čistá", "clean", "none", "bez loga"]):
        return None, None

    ss = 4
    strip_emblem = None
    dot_emblem = None

    # 1. Nosná páska: kompaktný dvojitý športový chevron (ukazuje HORE) alebo lekárenský krížik
    hs_hi, ws_hi = th_s * ss, tw_s * ss
    cx_s, cy_s = ws_hi // 2, hs_hi // 2
    canvas_s = np.zeros((hs_hi, ws_hi), dtype=np.uint8)

    if any(k in s_lower for k in ["chevron", "karbón", "carbon", "sport", "šport", "breathe", "plné", "dynamick", "rgb", "star", "hviez"]):
        # Horný hlavný športový chevron ^ (výška v 1x: [-8.5, +1.5] px -> bezpečne v zúženom páse)
        p_top = np.array([
            [cx_s, cy_s - int(8.5 * ss)],
            [cx_s + int(17.5 * ss), cy_s + int(1.0 * ss)],
            [cx_s + int(14.5 * ss), cy_s + int(3.5 * ss)],
            [cx_s, cy_s - int(4.3 * ss)],
            [cx_s - int(14.5 * ss), cy_s + int(3.5 * ss)],
            [cx_s - int(17.5 * ss), cy_s + int(1.0 * ss)],
        ], dtype=np.int32)
        # Dolný akcentový chevron ^ (výška v 1x: [-0.5, +7.5] px)
        p_bot = np.array([
            [cx_s, cy_s - int(0.8 * ss)],
            [cx_s + int(11.5 * ss), cy_s + int(5.5 * ss)],
            [cx_s + int(8.8 * ss), cy_s + int(7.8 * ss)],
            [cx_s, cy_s + int(3.0 * ss)],
            [cx_s - int(8.8 * ss), cy_s + int(7.8 * ss)],
            [cx_s - int(11.5 * ss), cy_s + int(5.5 * ss)],
        ], dtype=np.int32)
        cv2.fillPoly(canvas_s, [p_top, p_bot], 255, lineType=cv2.LINE_AA)
        strip_emblem = cv2.resize(canvas_s, (tw_s, th_s), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    elif any(k in s_lower for k in ["kríž", "cross", "telov", "tan", "medic"]):
        cv2.rectangle(canvas_s, (cx_s - int(2.8 * ss), cy_s - int(8.2 * ss)), (cx_s + int(2.8 * ss), cy_s + int(8.2 * ss)), 255, -1)
        cv2.rectangle(canvas_s, (cx_s - int(8.5 * ss), cy_s - int(2.8 * ss)), (cx_s + int(8.5 * ss), cy_s + int(2.8 * ss)), 255, -1)
        strip_emblem = cv2.resize(canvas_s, (tw_s, th_s), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    # 2. Nálepky na líca/bradu: geometrická 5-cípa Starface hviezda (r_out=11.0 px v 96x96) alebo krížik
    hd_hi, wd_hi = th_d * ss, tw_d * ss
    cx_d, cy_d = wd_hi // 2, hd_hi // 2
    canvas_d = np.zeros((hd_hi, wd_hi), dtype=np.uint8)

    if any(k in s_lower for k in ["star", "hviez", "chevron", "karbón", "carbon", "sport", "šport", "breathe", "plné", "dynamick", "rgb"]):
        r_out = 11.0 * ss
        r_in = 4.6 * ss
        star_pts = []
        for k in range(10):
            ang = -math.pi / 2.0 + k * (math.pi / 5.0)
            rad = r_out if (k % 2 == 0) else r_in
            star_pts.append([int(round(cx_d + rad * math.cos(ang))), int(round(cy_d + rad * math.sin(ang)))])
        cv2.fillPoly(canvas_d, [np.array(star_pts, dtype=np.int32)], 255, lineType=cv2.LINE_AA)
        dot_emblem = cv2.resize(canvas_d, (tw_d, th_d), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    elif any(k in s_lower for k in ["kríž", "cross", "telov", "tan", "medic"]):
        cv2.rectangle(canvas_d, (cx_d - int(2.4 * ss), cy_d - int(7.2 * ss)), (cx_d + int(2.4 * ss), cy_d + int(7.2 * ss)), 255, -1)
        cv2.rectangle(canvas_d, (cx_d - int(7.2 * ss), cy_d - int(2.4 * ss)), (cx_d + int(7.2 * ss), cy_d + int(2.4 * ss)), 255, -1)
        dot_emblem = cv2.resize(canvas_d, (tw_d, th_d), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    return strip_emblem, dot_emblem


def apply_logo_emblem(
    tex: torch.Tensor,
    key: str,
    logo_mask,
    strip_style: str = "athlete_black",
    is_rgb: bool = False,
) -> torch.Tensor:
    """
    Vykreslí ostrý emblém (chevron / hviezdičku / krížik) priamo do centrálnych ~5-10 %
    textúry BEZ akéhokoľvek tlmenia zvyšných ~90-95 % adversariálneho pozadia (m == 0 -> 1.0 * tex).
    """
    if logo_mask is None:
        return tex
    m_raw = None
    if isinstance(logo_mask, dict):
        m_raw = logo_mask.get(key)
        if m_raw is None:
            m_raw = logo_mask.get("strip" if key == "strip" else "dot")
    elif key == "strip":
        m_raw = logo_mask
    if m_raw is None:
        return tex

    device = tex.device
    m = m_raw.to(device) if isinstance(m_raw, torch.Tensor) else torch.tensor(m_raw, dtype=torch.float32, device=device)
    if m.dim() == 2:
        m = m.view(1, 1, *m.shape[-2:])
    elif m.dim() == 3:
        m = m.view(1, *m.shape)
    m = F.interpolate(m.float(), size=tex.shape[-2:], mode="bilinear", align_corners=False).clamp(0.0, 1.0)

    style_l = str(strip_style).lower()
    if is_rgb:
        # Kontrastný tmavý obrys (halo) + čisto biele jadro, aby logo svietilo na akejkoľvek RGB farbe
        m_dil = F.max_pool2d(m, kernel_size=5, stride=1, padding=2)
        ring = (m_dil - m).clamp(0.0, 1.0)
        halo_col = torch.tensor([-0.86, -0.86, -0.88], device=device).view(1, 3, 1, 1)
        fill_col = torch.clamp(torch.tensor([0.94, 0.95, 0.97], device=device).view(1, 3, 1, 1) + 0.06 * tex, -1.0, 1.0)
        tex = tex * (1.0 - 0.88 * ring) + halo_col * (0.88 * ring)
        tex = tex * (1.0 - 0.92 * m) + fill_col * (0.92 * m)
    elif "tan" in style_l:
        m_dil = F.max_pool2d(m, kernel_size=3, stride=1, padding=1)
        ring = (m_dil - m).clamp(0.0, 1.0)
        halo_col = torch.tensor([0.94, 0.94, 0.94], device=device).view(1, 3, 1, 1)
        fill_col = torch.clamp(torch.tensor([0.74, -0.58, -0.58], device=device).view(1, 3, 1, 1) + 0.06 * tex, -1.0, 1.0)
        tex = tex * (1.0 - 0.80 * ring) + halo_col * (0.80 * ring)
        tex = tex * (1.0 - 0.90 * m) + fill_col * (0.90 * m)
    else:
        # Tmavý karbón / čierna športová páska -> ostrý strieborno-biely emblém
        fill_col = torch.clamp(torch.tensor([0.86, 0.88, 0.92], device=device).view(1, 3, 1, 1) + 0.08 * tex, -1.0, 1.0)
        tex = tex * (1.0 - 0.90 * m) + fill_col * (0.90 * m)
    return tex


def hybrid_patch_attack(
    model: nn.Module,
    image: torch.Tensor,
    strip_style: str = "athlete_black",
    patch_size: tuple[int, int] | None = None,
    dot_radius: float = 4.5,
    dot_pattern: str = "cheeks_chin",
    strip_w_scale: float = 1.0,
    num_iter: int = 100,
    lr: float = 0.09,
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
    palette: list | None = None,
    logo_mask: torch.Tensor | None = None,
    mm_per_px: float | None = None,
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
    else:
        _, mesh_info = extract_landmarks(image, mtcnn=mtcnn, return_mesh=True)

    full_mask, full_relief, boxes = get_hybrid_geometry(
        landmarks=landmarks,
        mesh_info=mesh_info,
        dot_radius=dot_radius,
        dot_pattern=dot_pattern,
        strip_w_scale=strip_w_scale,
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

    strip_box = (sy1, sy2, sx1, sx2)
    left_box = (ly1, ly2, lx1, lx2)
    sh, sw = sy2 - sy1, sx2 - sx1
    lh, lw = ly2 - ly1, lx2 - lx1

    r_dot_eff = float(boxes.get("dot_radius_eff", dot_radius))
    strip_angle = float(boxes.get("strip_angle", 0.0))
    cos_left = float(boxes.get("foreshortening_left", 0.85))
    cos_right = float(boxes.get("foreshortening_right", 0.85))
    cos_chin = float(boxes.get("foreshortening_chin", 0.88))

    # --- Mierka tlace: mm na pixel z IPD (63 mm) nad nameranou vzdialenostou zrenic ---
    if mm_per_px is None:
        mm_per_px = 0.56
        try:
            ipd_px = math.hypot(float(landmarks[1][0]) - float(landmarks[0][0]),
                                float(landmarks[1][1]) - float(landmarks[0][1]))
            if ipd_px > 2.0:
                mm_per_px = 63.0 / ipd_px
        except Exception:
            mm_per_px = 0.56
    mm_per_px = float(mm_per_px)

    # Základné farby podľa štýlu (športová karbónová/čierna nosná páska + telové hydrokoloidné nálepky alebo plnofarebný RGB vzor)
    is_rgb = any(k in strip_style.lower() for k in ["rgb", "farebn", "art", "adversarial"]) or (palette is not None)
    n_ch = 3 if is_rgb else 1

    # Obmedzená paleta (logo): vzor smie používať len K farieb -> vyzerá ako
    # reálny produkt/logo, nie ako šum. Hodnoty v [-1,1] alebo [0,1].
    palette_t = None
    if palette is not None:
        palette_t = torch.tensor(palette, dtype=torch.float32, device=device)
        if float(palette_t.min()) >= -1e-6 and float(palette_t.max()) <= 1.0 + 1e-6:
            palette_t = palette_t * 2.0 - 1.0

    if is_rgb:
        base_strip = torch.zeros((1, 3, 1, 1), device=device)
        base_dots = torch.zeros((1, 3, 1, 1), device=device)
        delta_scale_strip = 1.0
        delta_scale_dots = 1.0
    elif strip_style in ["pro_carbon_logo", "carbon_sport"]:
        th_t, tw_t = TEX_STRIP_SIZE
        y_grid = torch.arange(th_t, device=device).view(-1, 1)
        x_grid = torch.arange(tw_t, device=device).view(1, -1)
        u = (x_grid / float(tw_t)) * 20.0
        v = (y_grid / float(th_t)) * 6.0
        carbon_pattern = (0.08 * torch.sin((u + v) * 2.5) * torch.cos((u - v) * 2.5)).view(1, 1, th_t, tw_t)
        base_strip = torch.tensor([-0.52, -0.52, -0.50], device=device).view(1, 3, 1, 1) + carbon_pattern
        base_dots = torch.tensor([-0.50, -0.50, -0.48], device=device).view(1, 3, 1, 1)
        delta_scale_strip = 0.65
        delta_scale_dots = 0.65
    elif strip_style == "tan_fabric":
        base_strip = torch.tensor([0.55, 0.36, 0.18], device=device).view(1, 3, 1, 1)
        base_dots = torch.tensor([0.58, 0.38, 0.20], device=device).view(1, 3, 1, 1)
        delta_scale_strip = 0.65
        delta_scale_dots = 0.55
    elif strip_style in ["clear_silicone", "clear"]:
        base_strip = torch.tensor([0.45, 0.45, 0.48], device=device).view(1, 3, 1, 1)
        base_dots = torch.tensor([0.58, 0.38, 0.20], device=device).view(1, 3, 1, 1)
        delta_scale_strip = 0.50
        delta_scale_dots = 0.50
    else:
        # Čierna športová páska + čierne športové nálepky
        base_strip = torch.tensor([-0.54, -0.54, -0.52], device=device).view(1, 3, 1, 1)
        base_dots = torch.tensor([-0.52, -0.52, -0.50], device=device).view(1, 3, 1, 1)
        delta_scale_strip = 0.65
        delta_scale_dots = 0.65

    # --- Vrstva: paleta (logo) + voliteľná šablóna tvaru ---
    def _prep_tex(tex, key):
        if not is_rgb and strip_style not in ["tan_fabric", "clear_silicone", "clear"]:
            tex = torch.clamp(tex, -0.98, 0.14)
        if palette_t is not None:
            tex = _quantize_to_palette(tex, palette_t)
        if logo_mask is not None:
            tex = apply_logo_emblem(tex, key, logo_mask, strip_style=strip_style, is_rgb=is_rgb)
        return tex

    # --- Optimalizované textúry vo VLASTNOM vysokom rozlíšení (žiadne ~20x6 px) ---
    delta_strip = nn.Parameter(torch.zeros((1, n_ch) + TEX_STRIP_SIZE, device=device))
    delta_left = nn.Parameter(torch.zeros((1, n_ch) + TEX_DOT_SIZE, device=device))
    params = [delta_strip, delta_left]

    if has_right:
        ry1, ry2, rx1, rx2 = boxes["right_dot"]
        delta_right = nn.Parameter(torch.zeros((1, n_ch) + TEX_DOT_SIZE, device=device))
        params.append(delta_right)

    if has_chin:
        cy1, cy2, cx1, cx2 = boxes["chin_dot"]
        delta_chin = nn.Parameter(torch.zeros((1, n_ch) + TEX_DOT_SIZE, device=device))
        params.append(delta_chin)

    opt = torch.optim.Adam(params, lr=lr)

    with torch.no_grad():
        orig_emb = model(image).detach()

    B, C, H, W = image.shape

    patch_blend_factor = 0.93 if is_rgb else 0.88
    gain_full = _illumination_gain(image, boxes=boxes).detach()
    mask_b = full_mask.expand(B, -1, -1, -1) if full_mask.shape[0] != B else full_mask

    for _ in range(num_iter):
        opt.zero_grad()

        # Vysokorozlíšené textúry (Adam optimalizuje priamo v 96x256 / 96x96 bufferoch)
        tex_strip = torch.clamp(base_strip + delta_strip * delta_scale_strip, -1.0, 1.0)
        tex_left = torch.clamp(base_dots + delta_left * delta_scale_dots, -1.0, 1.0)
        region_specs = {
            "strip": (_prep_tex(tex_strip, "strip"), strip_box, TEX_STRIP_SIZE),
            "left_dot": (_prep_tex(tex_left, "left_dot"), left_box, TEX_DOT_SIZE),
        }
        if has_right:
            region_specs["right_dot"] = (
                _prep_tex(torch.clamp(base_dots + delta_right * delta_scale_dots, -1.0, 1.0), "right_dot"),
                (ry1, ry2, rx1, rx2), TEX_DOT_SIZE)
        if has_chin:
            region_specs["chin_dot"] = (
                _prep_tex(torch.clamp(base_dots + delta_chin * delta_scale_dots, -1.0, 1.0), "chin_dot"),
                (cy1, cy2, cx1, cx2), TEX_DOT_SIZE)

        # Zložený patch (3D ovinutie + osvetlenie scény + reliéf + priehľadnosť papiera)
        canvas, _ = _build_canvas(image, region_specs, gain_full, full_relief, patch_blend_factor, boxes=boxes)

        if eot:
            # Geometria: perspektívna homografia z rohových offsetov + rotácia/translácia/škála
            # + jemný crop/alignment jitter celej tváre (bod 8)
            K = EOT_BATCH
            canvas_rep = canvas.repeat(K, 1, 1, 1) if canvas.shape[0] != K else canvas
            image_rep = image.repeat(K, 1, 1, 1) if image.shape[0] != K else image
            mask_rep = mask_b.repeat(K, 1, 1, 1) if mask_b.shape[0] != K else mask_b
            canvas_rep, mask_rep = _patch_placement_jitter(canvas_rep, mask_rep)
            grid = _eot_grid(K, H, W, device)
            trans_canvas = F.grid_sample(canvas_rep, grid, mode="bilinear", padding_mode="border", align_corners=False)
            trans_image = F.grid_sample(image_rep, grid, mode="bilinear", padding_mode="border", align_corners=False)
            trans_mask = F.grid_sample(mask_rep, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            trans_mask = torch.clamp(trans_mask, 0.0, 1.0)
            adv_eot = (1.0 - trans_mask) * trans_image + trans_mask * trans_canvas
            # Kamera: rozmazanie, jas/kontrast, teplota farieb, senzorový šum, JPEG
            adv_eot = _camera_pipeline(adv_eot)
            # Baseline: TÁ ISTÁ warpovaná tvár BEZ patcha -> loss meria iba vplyv patcha,
            # nie geometrickú zmenu identity (to bola príčina slabého útoku, sim 0.77).
            with torch.no_grad():
                base_emb = model(_camera_pipeline(trans_image.contiguous())).detach()
            emb_eot = model(adv_eot.contiguous())
            # Čistý digitálny člen (vrátane tieňa hrany papiera a optického rozmazania fotky)
            clean_comp = composite_realistic_stickers(image, canvas, full_mask, boxes=boxes)
            emb_clean = model(clean_comp.contiguous())
            if target_emb is not None:
                tgt = target_emb.expand(emb_eot.shape[0], -1) if target_emb.shape[0] != emb_eot.shape[0] else target_emb
                sim = 0.5 * F.cosine_similarity(tgt, emb_eot).mean() + 0.5 * F.cosine_similarity(target_emb, emb_clean).mean()
                loss = -sim
            else:
                sim = 0.5 * F.cosine_similarity(base_emb, emb_eot).mean() + 0.5 * F.cosine_similarity(orig_emb, emb_clean).mean()
                loss = sim
        else:
            adv_img = composite_realistic_stickers(image, canvas, full_mask, boxes=boxes).contiguous()
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
        final_tex_strip = torch.clamp(base_strip + delta_strip * delta_scale_strip, -1.0, 1.0)
        final_tex_left = torch.clamp(base_dots + delta_left * delta_scale_dots, -1.0, 1.0)
        final_specs = {
            "strip": (_prep_tex(final_tex_strip, "strip"), strip_box, TEX_STRIP_SIZE),
            "left_dot": (_prep_tex(final_tex_left, "left_dot"), left_box, TEX_DOT_SIZE),
        }
        if has_right:
            final_specs["right_dot"] = (
                _prep_tex(torch.clamp(base_dots + delta_right * delta_scale_dots, -1.0, 1.0), "right_dot"),
                (ry1, ry2, rx1, rx2), TEX_DOT_SIZE)
        if has_chin:
            final_specs["chin_dot"] = (
                _prep_tex(torch.clamp(base_dots + delta_chin * delta_scale_dots, -1.0, 1.0), "chin_dot"),
                (cy1, cy2, cx1, cx2), TEX_DOT_SIZE)

        final_canvas, final_comps = _build_canvas(image, final_specs, gain_full, full_relief, patch_blend_factor, boxes=boxes)
        final_adv_img = composite_realistic_stickers(image, final_canvas, full_mask, boxes=boxes)

        final_emb = model(final_adv_img)
        if target_emb is not None:
            final_sim = F.cosine_similarity(target_emb, final_emb).item()
            orig_sim = F.cosine_similarity(target_emb, orig_emb).item()
            success = final_sim > 0.5
        else:
            final_sim = F.cosine_similarity(orig_emb, final_emb).item()
            orig_sim = 1.0
            success = final_sim < 0.5

    patches_dict = {key: spec[0].squeeze(0).detach().cpu() for key, spec in final_specs.items()}
    spec_items = {key: (spec[1], spec[2]) for key, spec in final_specs.items()}
    masks_np, geo = build_printable_payload(boxes, spec_items, mm_per_px)

    printable_sheet = generate_printable_sheet(
        patches_dict,
        mm_per_px=mm_per_px,
        landmarks=landmarks,
        masks=masks_np,
        geometry=geo,
    )

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
