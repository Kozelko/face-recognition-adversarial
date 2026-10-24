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
from attacks.nasal_strip_patch import create_nasal_strip_mask


# --- Vysoko rozlíšené textúrové buffery (aby 300 DPI vzor nebol "kockatý") ---
TEX_STRIP_SIZE = (96, 256)   # (H, W) textúra nosnej pásky
TEX_DOT_SIZE = (96, 96)      # (H, W) textúra nálepky
EOT_BATCH = 8                # počet EoT transformácií na optimalizačný krok

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


def _illumination_gain(image: torch.Tensor) -> torch.Tensor:
    """
    Fotometricky adaptívne osvetlenie pokožky:
    Nálepka neodráža viac svetla, než do miestnosti reálne dopadá.
    1. Lokálny tieň/svetlo tváre (3D gradient nosa/líc).
    2. Adaptácia na celkový jas tváre (ak je webkamera tmavá / v tieni,
       nálepka automaticky stmavne a nesvieti ako umelý cut-and-paste objekt).
    """
    skin_lum = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
    lum01 = torch.clamp(skin_lum * 0.5 + 0.5, 0.0, 1.0)
    k = 5
    t = torch.arange(k, dtype=torch.float32, device=image.device) - (k - 1) * 0.5
    wts = torch.exp(-(t ** 2) / (2.0 * 1.2 ** 2))
    wts = wts / wts.sum()
    blurred = F.conv2d(F.pad(lum01, (k // 2, k // 2, 0, 0), mode="replicate"), wts.view(1, 1, 1, k))
    blurred = F.conv2d(F.pad(blurred, (0, 0, k // 2, k // 2), mode="replicate"), wts.view(1, 1, k, 1))
    mean_lum = blurred.mean(dim=(-2, -1), keepdim=True).clamp_min(0.15)
    rel_illum = blurred / mean_lum
    ambient_factor = torch.clamp(mean_lum / 0.52, 0.45, 1.05)
    return torch.clamp(rel_illum * ambient_factor * 0.94, 0.25, 1.15)


def _region_field(field: torch.Tensor, box: tuple[int, int, int, int], tex_h: int, tex_w: int, pad: int = 2) -> torch.Tensor:
    """Vyreže oblasť okolo boxu (s malým presahom) a diferencovateľne ju prevzorkuje
    do rozlíšenia textúrového buffera."""
    y1, y2, x1, x2 = box
    h, w = field.shape[-2], field.shape[-1]
    py1, py2 = max(0, y1 - pad), min(h, y2 + pad)
    px1, px2 = max(0, x1 - pad), min(w, x2 + pad)
    crop = field[:, :, py1:py2, px1:px2]
    return F.interpolate(crop, size=(tex_h, tex_w), mode="bilinear", align_corners=False)


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
    Zloží jednu vysokorozlíšenú textúru na finálnu "fyzickú" podobu patche:
    (textúra -> [0,1]) * osvetlenie * reliéf  ->  späť do [-1,1]  ->  priehľadnosť.
    Presne táto zložená podoba sa neskôr downsample-uje do scény pre model
    a zároveň sa tlačí (konzistencia trénovania a tlače).
    """
    tex_h, tex_w = tex_size
    gain_hr = _region_field(gain_full, box, tex_h, tex_w)
    relief_hr = _region_field(relief_full, box, tex_h, tex_w)
    img_hr = _region_field(image, box, tex_h, tex_w)
    tex01 = torch.clamp(texture * 0.5 + 0.5, 0.0, 1.0)
    lit01 = torch.clamp(tex01 * gain_hr * relief_hr, 0.0, 1.0)
    lit = lit01 * 2.0 - 1.0
    comp = blend * lit + (1.0 - blend) * img_hr
    return comp.clamp(-1.0, 1.0)


def _build_canvas(image, region_specs, gain_full, relief_full, blend):
    """
    Vloží všetky textúry do scény 112x112: vysokorozlíšená textúra sa zloží,
    diferencovateľne downsample-uje do cieľového boxu a tam sa zamaskuje.
    Vracia (canvas, comps) - comps sú vysokorozlíšené zložené patche.
    """
    canvas = image.clone()
    comps = {}
    for key, (texture, box, tex_size) in region_specs.items():
        comp_hr = _compose_region(texture, box, tex_size, image, gain_full, relief_full, blend)
        y1, y2, x1, x2 = box
        box_patch = F.interpolate(comp_hr, size=(y2 - y1, x2 - x1), mode="bilinear", align_corners=False)
        canvas[:, :, y1:y2, x1:x2] = box_patch
        comps[key] = comp_hr
    return canvas, comps


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
    quality = 75.0 + float(torch.rand(1, device=device).item()) * 20.0
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
    dot_radius: float = 5.5,
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
    eye_dist = math.sqrt(dx**2 + dy**2)
    yaw = mesh_info.get("yaw", 0.0) if mesh_info else 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    has_mesh = mesh_info is not None and "points_3d" in mesh_info and len(mesh_info["points_3d"]) > 400
    pts = mesh_info["points_3d"] if has_mesh else None

    # Anatomické umiestnenie Breathe Right pásky (stredná až dolná časť chrbta nosa nad chlopňami, mimo očí!)
    if has_mesh:
        p195 = pts[195] if len(pts) > 195 else pts[197]
        p5 = pts[5] if len(pts) > 5 else pts[195]
        strip_cx = float(p195[0] * 0.50 + p5[0] * 0.50)
        strip_cy = float(p195[1] * 0.50 + p5[1] * 0.50)
    else:
        strip_cx = (lx + rx) * 0.5
        strip_cy = (ly + ry) * 0.5 + (ny - (ly + ry) * 0.5) * 0.54

    strip_cx = float(np.clip(strip_cx + offset_x + strip_ox, 10.0, 102.0))
    strip_cy = float(np.clip(strip_cy + offset_y + strip_oy, 10.0, 102.0))

    if has_mesh and len(pts) > 278:
        alar_w = math.hypot(pts[278][0] - pts[48][0], pts[278][1] - pts[48][1]) * 0.94
        base_w = max(18.0, min(25.5, alar_w))
    else:
        base_w = max(18.0, min(25.0, eye_dist * 0.52))
    base_w = base_w * float(strip_w_scale)

    if has_mesh and len(pts) > 168:
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
        "strip_angle": float(strip_angle),
    }

    # Nálepky na lícach
    r_dot = max(2.0, min(14.0, float(dot_radius)))

    # Ľavé líce (stred malárnej oblasti, bezpečne mimo nosovej pásky)
    if has_mesh and len(pts) > 205:
        p_l = pts[205]
        lcx, lcy = float(p_l[0]), float(p_l[1]) - 1.5
    else:
        lcx = lx - (nx - lx) * 0.12
        lcy = ly + (mly - ly) * 0.45

    lcx = float(np.clip(lcx + offset_x + left_ox, 6.0, 106.0))
    lcy = float(np.clip(lcy + offset_y + left_oy, 6.0, 106.0))

    if has_mesh:
        l_cos, l_tilt, _ = get_surface_geometry_at_coords(pts, lcx, lcy, roll)
    else:
        l_cos, l_tilt = 0.85, roll

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
    boxes["foreshortening_left"] = float(l_cos)
    boxes.setdefault("dot_geom", {})["left_dot"] = (lcx, lcy, float(l_cos), float(l_tilt))

    full_mask = torch.maximum(strip_mask, l_mask)
    full_relief = torch.ones_like(full_mask)
    full_relief = torch.where(strip_mask > 0.1, strip_relief, full_relief)
    full_relief = torch.where(l_mask > 0.1, l_relief, full_relief)

    # Pravé líce (stred malárnej oblasti, bezpečne mimo nosovej pásky)
    if "1_cheek" not in dot_pattern and "1 nálepka" not in dot_pattern:
        if has_mesh and len(pts) > 425:
            p_r = pts[425]
            rcx, rcy = float(p_r[0]), float(p_r[1]) - 1.5
        else:
            rcx = rx + (rx - nx) * 0.12
            rcy = ry + (mry - ry) * 0.45

        rcx = float(np.clip(rcx + offset_x + right_ox, 6.0, 106.0))
        rcy = float(np.clip(rcy + offset_y + right_oy, 6.0, 106.0))

        if has_mesh:
            r_cos, r_tilt, _ = get_surface_geometry_at_coords(pts, rcx, rcy, roll)
        else:
            r_cos, r_tilt = 0.85, roll

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
        boxes["foreshortening_right"] = float(r_cos)
        boxes.setdefault("dot_geom", {})["right_dot"] = (rcx, rcy, float(r_cos), float(r_tilt))

        full_mask = torch.maximum(full_mask, r_mask)
        full_relief = torch.where(r_mask > 0.1, r_relief, full_relief)

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
        boxes["foreshortening_chin"] = float(c_cos)
        boxes.setdefault("dot_geom", {})["chin_dot"] = (ccx, ccy, float(c_cos), float(c_tilt))

        full_mask = torch.maximum(full_mask, c_mask)
        full_relief = torch.where(c_mask > 0.1, c_relief, full_relief)
        boxes["chin_mask"] = c_mask
        boxes["chin_relief"] = c_relief

    boxes["strip_mask"] = strip_mask
    boxes["strip_relief"] = strip_relief
    boxes["dot_radius_eff"] = float(r_dot)
    boxes["left_mask"] = l_mask
    boxes["left_relief"] = l_relief
    if "right_mask" not in boxes and "1_cheek" not in dot_pattern and "1 nálepka" not in dot_pattern:
        boxes["right_mask"] = r_mask
        boxes["right_relief"] = r_relief

    full_mask = torch.clamp(full_mask, 0.0, 1.0)
    return full_mask, full_relief, boxes


def generate_printable_sheet(
    patches: dict,
    output_path: str | None = None,
    mm_per_px: float | None = None,
    landmarks=None,
    masks: dict | None = None,
    geometry: dict | None = None,
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
        """Nálepka ako ROVNÝCH KRUH. Textura, ktorú model videl, leží na naklonenom
        povrchu (elipsa). Tu sa inverzne zobrazí: pre každý bod rovného kruhu sa
        vypočíta, kam dopadol v obraze (M = skrátenie o cos pozdĺž osi sklonu) a odtiaľ
        sa vzáme farba. Orientácia („hore“) ostane ako na tvári, odstráni sa len
        perspektívne skrátenie."""
        img_size = 112
        y1, y2, x1, x2 = flat["box"]
        py1, py2 = max(0, y1 - 2), min(img_size, y2 + 2)
        px1, px2 = max(0, x1 - 2), min(img_size, x2 + 2)
        ch, cw = py2 - py1, px2 - px1
        th_, tw_ = rgb_tex.shape[0], rgb_tex.shape[1]
        r_img = float(flat["r"])
        d_mm = 2.0 * r_img * mm_per_px
        margin_mm = 1.0
        side = max(8, int(round((d_mm + 2.0 * margin_mm) * MM_TO_PX)))
        c = (side - 1) / 2.0
        s = (d_mm * MM_TO_PX / 2.0) / r_img            # rovné px na 1 px obrazu
        jj, ii = np.meshgrid(np.arange(side, dtype=np.float32), np.arange(side, dtype=np.float32))
        fx, fy = (jj - c) / s, (ii - c) / s             # súrdnice v px obrazu (rovný kruh)
        a = float(flat["tilt"]); cs = float(flat["cos"])
        ux, uy, vx, vy = math.cos(a), math.sin(a), -math.sin(a), math.cos(a)
        fu, fv = fx * ux + fy * uy, fx * vx + fy * vy
        sx = cs * fu * ux + fv * vx                     # posun v obraze
        sy = cs * fu * uy + fv * vy
        map_x = ((flat["cx"] + sx - px1 + 0.5) * tw_ / cw - 0.5).astype(np.float32)
        map_y = ((flat["cy"] + sy - py1 + 0.5) * th_ / ch - 0.5).astype(np.float32)
        flat_rgb = cv2.remap(rgb_tex, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        dist = np.sqrt(fx * fx + fy * fy) / r_img
        alpha = np.clip((1.0 - dist) * (r_img / 0.75) + 0.5, 0.0, 1.0)
        alpha = (alpha * 255.0).astype(np.uint8)
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
        rgb = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_LANCZOS4)
        alpha = cv2.resize(alpha, (tw, th), interpolation=cv2.INTER_LANCZOS4)

        # Aplikácia strip_angle na finálny renderovaný vzor na hárku
        rot_deg = float(geo.get("rotate_deg", 0.0))
        if abs(rot_deg) > 1e-3:
            diag = int(math.ceil(math.hypot(th, tw))) + 2
            pad = np.zeros((diag, diag, 4), dtype=np.uint8)
            y0, x0 = (diag - th) // 2, (diag - tw) // 2
            pad[y0:y0 + th, x0:x0 + tw, :3] = rgb
            pad[y0:y0 + th, x0:x0 + tw, 3] = alpha
            rm = cv2.getRotationMatrix2D((diag / 2.0, diag / 2.0), rot_deg, 1.0)
            rot = cv2.warpAffine(pad, rm, (diag, diag), flags=cv2.INTER_LANCZOS4,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
            rgb, alpha = rot[:, :, :3], rot[:, :, 3]
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
    margin_mm, top_mm, gap_mm = 10.0, 42.0, 8.0
    for it in items:
        it["w_mm"] = it["rgb"].shape[1] / MM_TO_PX
        it["h_mm"] = it["rgb"].shape[0] / MM_TO_PX
    sheet_w_mm, sheet_h_mm = 210.0, 297.0
    sheet_w = int(round(sheet_w_mm * MM_TO_PX))
    sheet_h = int(round(sheet_h_mm * MM_TO_PX))
    sheet = np.ones((sheet_h, sheet_w, 3), dtype=np.uint8) * 255

    # Text a inštrukcie
    cv2.putText(sheet, "Adversarial Physical Hybrid Patch - Tlacovy harok (100% mierka)", (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(sheet, "Vytlac na matny papier bez zmeny mierky (Actual Size / 100%).",
                (30, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(sheet, "DOLEZITE PRED STRIHANIM: Na zadnu stranu papiera si ceruzkou napis N^, L^, P^ (sipka ^ = strana HORE k ociam)!",
                (30, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 20, 20), 1, cv2.LINE_AA)

    # Kalibračné meradlo (50 mm)
    bar_x, bar_y = 30, 138
    bar_len = int(50 * MM_TO_PX)
    cv2.line(sheet, (bar_x, bar_y), (bar_x + bar_len, bar_y), (0, 0, 0), 3)
    cv2.line(sheet, (bar_x, bar_y - 10), (bar_x, bar_y + 10), (0, 0, 0), 3)
    cv2.line(sheet, (bar_x + bar_len, bar_y - 10), (bar_x + bar_len, bar_y + 10), (0, 0, 0), 3)
    cv2.putText(sheet, "Kontrolne meritko: presne 50 mm (over pravitkom po tlaci)", (bar_x, bar_y + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 1, cv2.LINE_AA)

    short_tags = {
        "strip": "[N] NOS",
        "left_dot": "[L] LAVE LICE (vlavo na monitore)",
        "right_dot": "[P] PRAVE LICE (vpravo na monitore)",
        "chin_dot": "[B] BRADA (volitelne)",
    }

    def _paste(rgb, alpha, x_px, y_px):
        h, w = rgb.shape[:2]
        y_px, x_px = max(0, int(y_px)), max(0, int(x_px))
        if y_px + h > sheet.shape[0] or x_px + w > sheet.shape[1]:
            return
        a = (alpha.astype(np.float32) / 255.0)[..., None]
        region = sheet[y_px:y_px + h, x_px:x_px + w].astype(np.float32)
        sheet[y_px:y_px + h, x_px:x_px + w] = np.clip(
            region * (1.0 - a) + rgb.astype(np.float32) * a, 0.0, 255.0).astype(np.uint8)

    def _paste_row(y0_px, note=""):
        x_cursor = int(round(margin_mm * MM_TO_PX))
        if note:
            cv2.putText(sheet, note, (x_cursor, y0_px - 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2, cv2.LINE_AA)
        for it in items:
            h_i, w_i = it["rgb"].shape[:2]
            cx = x_cursor + w_i // 2
            # Šípka HORE (^ HORE) tesne nad vrchným okrajom nálepky
            ay = max(12, y0_px - 8)
            pts_arrow = np.array([[cx, ay - 12], [cx - 9, ay], [cx + 9, ay]], dtype=np.int32)
            cv2.fillPoly(sheet, [pts_arrow], (0, 140, 0))
            cv2.putText(sheet, "HORE", (cx + 13, ay - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 130, 0), 1, cv2.LINE_AA)

            _paste(it["rgb"], it["alpha"], x_cursor, y0_px)
            tag_str = short_tags.get(it["key"], it["label"])
            dim_str = f"({it['mm_w']:.1f} x {it['mm_h']:.1f} mm)"
            cv2.putText(sheet, tag_str, (x_cursor, y0_px + h_i + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(sheet, dim_str, (x_cursor, y0_px + h_i + 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 90), 1, cv2.LINE_AA)
            x_cursor += int(round((it["w_mm"] + gap_mm + 8.0) * MM_TO_PX))

    row_h_px = max([it["rgb"].shape[0] for it in items], default=0)
    _paste_row(int(round((top_mm + 6.0) * MM_TO_PX)), "SADA 1 (hlavna)")
    _paste_row(int(round((top_mm + 6.0) * MM_TO_PX)) + row_h_px + int(round(52 * MM_TO_PX)), "SADA 2 (rezerva)")

    cv2.putText(sheet, "Poznamka: Tlac s profilom farieb Standard/sRGB na matny papier (mierka 100%).",
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

    print("[PRINTABLE] mm_per_px=%.4f sheet=%.1fx%.1fmm " % (mm_per_px, sheet_w_mm, sheet_h_mm) +
          " ".join("%s=%.2fx%.2fmm(%dx%dpx)" % (it["key"], it["mm_w"], it["mm_h"],
                                                 it["rgb"].shape[1], it["rgb"].shape[0]) for it in items))

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
    r_dot_eff = float(boxes.get("dot_radius_eff", 4.5))
    strip_angle = float(boxes.get("strip_angle", 0.0))
    cos_by_key = {
        "left_dot": float(boxes.get("foreshortening_left", 0.85)),
        "right_dot": float(boxes.get("foreshortening_right", 0.85)),
        "chin_dot": float(boxes.get("foreshortening_chin", 0.88)),
    }
    mask_keys = {"strip": "strip_mask", "left_dot": "left_mask",
                 "right_dot": "right_mask", "chin_dot": "chin_mask"}
    labels = {"strip": "1. Breathe Right páska na nos", "left_dot": "2. Ľavé líce",
              "right_dot": "3. Pravé líce", "chin_dot": "4. Brada"}
    masks_np, geo = {}, {}
    for key, (box, tex_size) in spec_items.items():
        y1, y2, x1, x2 = box
        m = boxes.get(mask_keys.get(key))
        if m is not None:
            masks_np[key] = _region_field(m, box, tex_size[0], tex_size[1]).squeeze().cpu().numpy()
        if key == "strip":
            geo[key] = {"mm_w": (x2 - x1) * mm_per_px, "mm_h": (y2 - y1) * mm_per_px,
                        "unsquash": 1.0, "rotate_deg": -math.degrees(strip_angle),
                        "label": labels[key]}
        else:
            geo[key] = {"mm_w": 2.0 * r_dot_eff * mm_per_px, "mm_h": 2.0 * r_dot_eff * mm_per_px,
                        "unsquash": 1.0 / max(0.45, cos_by_key.get(key, 0.85)),
                        "rotate_deg": 0.0, "label": labels[key]}
            dg = (boxes.get("dot_geom") or {}).get(key)
            if dg is not None:
                # Presné údaje na "rozbalenie" nálepky: stred, cos sklonu, os sklonu,
                # box textury. generate_printable_sheet z toho vyrenderuje čistý kruh.
                geo[key]["flat"] = {"cx": dg[0], "cy": dg[1], "cos": dg[2], "tilt": dg[3],
                                    "r": r_dot_eff, "box": tuple(box), "tex": tuple(tex_size)}
    return masks_np, geo


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
        carbon_pattern = (0.09 * torch.sin((u + v) * 2.5) * torch.cos((u - v) * 2.5)).view(1, 1, th_t, tw_t)
        base_strip = torch.tensor([-0.65, -0.65, -0.65], device=device).view(1, 3, 1, 1) + carbon_pattern
        base_dots = torch.tensor([0.68, 0.45, 0.22], device=device).view(1, 3, 1, 1)
        delta_scale_strip = 0.70
        delta_scale_dots = 0.50
    elif strip_style == "tan_fabric":
        base_strip = torch.tensor([0.65, 0.44, 0.25], device=device).view(1, 3, 1, 1)
        base_dots = torch.tensor([0.68, 0.45, 0.22], device=device).view(1, 3, 1, 1)
        delta_scale_strip = 0.70
        delta_scale_dots = 0.50
    elif strip_style in ["clear_silicone", "clear"]:
        base_strip = torch.tensor([0.45, 0.45, 0.48], device=device).view(1, 3, 1, 1)
        base_dots = torch.tensor([0.68, 0.45, 0.22], device=device).view(1, 3, 1, 1)
        delta_scale_strip = 0.50
        delta_scale_dots = 0.50
    else:
        # Čierna športová
        base_strip = torch.tensor([-0.72, -0.72, -0.72], device=device).view(1, 3, 1, 1)
        base_dots = torch.tensor([0.68, 0.45, 0.22], device=device).view(1, 3, 1, 1)
        delta_scale_strip = 0.70
        delta_scale_dots = 0.50

    # --- Vrstva: paleta (logo) + voliteľná šablóna tvaru ---
    def _prep_tex(tex, key):
        if palette_t is not None:
            tex = _quantize_to_palette(tex, palette_t)
        if logo_mask is not None:
            m = logo_mask
            if m.dim() == 2:
                m = m.view(1, 1, *m.shape[-2:])
            elif m.dim() == 3:
                m = m.view(1, *m.shape)
            m = F.interpolate(m.float(), size=tex.shape[-2:], mode="nearest")
            base_c = base_strip if key == "strip" else base_dots
            tex = tex * m + base_c.expand_as(tex) * (1.0 - m)
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

    patch_blend_factor = 0.88 if is_rgb else 0.85
    gain_full = _illumination_gain(image).detach()
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

        # Zložený patch (osvetlenie * reliéf * priehľadnosť materiálu) sa diferencovateľne
        # downsample-uje do boxu, takže model vidí presne to, čo sa neskôr tlačí.
        canvas, _ = _build_canvas(image, region_specs, gain_full, full_relief, patch_blend_factor)

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
            # Čistý digitálny člen: patch na neposunutej tvári, aby útok ostal silný
            # aj bez fyzickej transformácie a final_sim na čistom obrázku klesol.
            emb_clean = model(((1.0 - full_mask) * image + full_mask * canvas).contiguous())
            if target_emb is not None:
                tgt = target_emb.expand(emb_eot.shape[0], -1) if target_emb.shape[0] != emb_eot.shape[0] else target_emb
                sim = 0.5 * F.cosine_similarity(tgt, emb_eot).mean() + 0.5 * F.cosine_similarity(target_emb, emb_clean).mean()
                loss = -sim
            else:
                sim = 0.5 * F.cosine_similarity(base_emb, emb_eot).mean() + 0.5 * F.cosine_similarity(orig_emb, emb_clean).mean()
                loss = sim
        else:
            adv_img = ((1.0 - full_mask) * image + full_mask * canvas).contiguous()
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

        final_canvas, final_comps = _build_canvas(image, final_specs, gain_full, full_relief, patch_blend_factor)
        final_adv_img = (1.0 - full_mask) * image + full_mask * final_canvas
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

    # patches_dict obsahuje PRESNE to, čo model dostal: vysokorozlíšenú zloženú podobu
    # patcha (rovnaké light_mult/osvetlenie, rovnaký reliéf, rovnaká priehľadnosť),
    # ktorej diferencovateľný downsample je vstupom modelu. Čo sa vytlačí = čo sa optimalizovalo.
    patches_dict = {key: comp.squeeze(0).cpu() for key, comp in final_comps.items()}

    masks_np = {}
    mask_keys = {"strip": "strip_mask", "left_dot": "left_mask",
                 "right_dot": "right_mask", "chin_dot": "chin_mask"}
    for key, (_, box, tex_size) in final_specs.items():
        m = boxes.get(mask_keys[key])
        if m is not None:
            masks_np[key] = _region_field(m.to(device), box, tex_size[0], tex_size[1]).squeeze().cpu().numpy()

    geo = {
        "strip": {
            "mm_w": (sx2 - sx1) * mm_per_px,
            "mm_h": (sy2 - sy1) * mm_per_px,
            "unsquash": 1.0,
            "rotate_deg": -math.degrees(strip_angle),
            "label": "1. Breathe Right paska na nos",
        },
        "left_dot": {
            "mm_w": 2.0 * r_dot_eff * mm_per_px,
            "mm_h": 2.0 * r_dot_eff * mm_per_px,
            "unsquash": 1.0 / max(0.45, cos_left),
            "rotate_deg": 0.0,
            "label": "2. Lave lice",
        },
    }
    if has_right:
        geo["right_dot"] = {
            "mm_w": 2.0 * r_dot_eff * mm_per_px,
            "mm_h": 2.0 * r_dot_eff * mm_per_px,
            "unsquash": 1.0 / max(0.45, cos_right),
            "rotate_deg": 0.0,
            "label": "3. Prave lice",
        }
    if has_chin:
        geo["chin_dot"] = {
            "mm_w": 2.0 * r_dot_eff * mm_per_px,
            "mm_h": 2.0 * r_dot_eff * mm_per_px,
            "unsquash": 1.0 / max(0.45, cos_chin),
            "rotate_deg": 0.0,
            "label": "4. Brada",
        }

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
