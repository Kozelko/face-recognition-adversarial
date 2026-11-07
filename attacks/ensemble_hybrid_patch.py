"""
Ensemble + Naturalness Hybrid Physical Attack (Téza D - novinka).

Zovšeobecnenie fyzického hybridného útoku (Breathe Right páska na nose +
hydrokoloidné nálepky na lícach) na ENSEMBLE rozpoznávacích modelov súčasne,
s kalibrovanými marginami, regularizáciou tlačiteľnosti (NPS), TV hladkosťou
a naturalness stratou (produktová / kinesio paleta, nízkofrekvenčná textúra).

Kľúčové vlastnosti:
- Loss = suma cez modely, každý má vlastný DODGE člen s KALIBROVANOU marginou:
      loss_model = max(0, sim(adv_emb, orig_emb) - margin_low[model])
  takže každý model je tlačený pod svoju vlastnú hranicu.
- Voliteľný per-model `target_emb` (impersonation): namiesto dodge sa maximalizuje
  podobnosť voči cieľovému embeddingu.
- Regularizátory: NPS (tlačiteľný gamut), TV, NATURALNESS (nízkofrekvenčná,
  produktová paleta - kinesio páska / kozmetická nálepka, nie šum).
- PAD-AWARE hook `pad_model=None`: ak je poskytnutý, pridá penalizáciu držiacu
  liveness skóre blízko clean baseline. Ak je None, preskočí sa graceful.
  POZOR: V repozitári NEEXISTUJÚ žiadne PAD váhy (MiniFASNet). Hook je
  plug-and-play - stačí neskôr dodať natrénované MiniFASNet váhy a odovzdať
  callable vrátiaci liveness skóre.
- Multi-image tréning: akceptuje zoznam fotiek tej istej identity (`images`),
  v každej iterácii vyberie náhodnú fotku -> menej overfittingu na jeden frame.

Musí sa dať importovať samostatne:
    python -c "import attacks.ensemble_hybrid_patch"
"""

import math
import ctypes
try:  # pragma: no cover - platform specific OpenCV/PNG shim used elsewhere in repo
    ctypes.CDLL(
        '/home/kozel/miniconda/envs/cnn-benchmark/lib/libpng16.so.16',
        mode=ctypes.RTLD_GLOBAL,
    )
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Znovupoužité stabilné helpery (čítané signatúry) ---
from attacks.hybrid_patch import (
    get_hybrid_geometry,
    generate_printable_sheet,
    apply_logo_emblem,
    TEX_STRIP_SIZE,
    TEX_DOT_SIZE,
    EOT_BATCH,
    _gblur_fixed,
    _illumination_gain,
    _build_canvas,
    composite_realistic_stickers,
    _eot_grid,
    _patch_placement_jitter,
    _camera_pipeline,
    _quantize_to_palette,
    derive_mm_per_px,
    build_printable_payload,
)
from attacks.constellation_patch import extract_landmarks, close_mediapipe_landmarker
from attacks.nasal_strip_patch import create_nasal_strip_mask  # noqa: F401  (re-export pre parity)


# Tlačiteľný gamut - malý súbor anchor farieb (kinesio páska / kozmetická náplasť).
# Hodnoty v rozsahu [-1, 1] (rovnaká normalizácia ako vstupné tenzory).
PRINTABLE_ANCHORS = [
    [-0.72, -0.72, -0.72],   # čierna športová páska
    [-0.95, -0.95, -0.95],   # hlboká čierna
    [-0.35, -0.35, -0.33],   # tmavý grafit / karbón
    [0.65, 0.44, 0.25],      # telová / beige
    [0.68, 0.45, 0.22],      # hydrokoloidná nálepka
    [0.90, 0.85, 0.70],      # krémová
    [0.95, 0.95, 0.95],      # matná biela
    [0.05, 0.35, -0.55],     # modrá kinesio
    [0.40, 0.20, 0.10],      # hnedá
]

# Vyvážené váhy architektúr podľa svetového nasadenia (iResNet ArcFace/AdaFace + InceptionResnetV1 FaceNet):
# ArcFace (iResNet50 Angular Margin) a AdaFace (Quality-Adaptive Margin) tvoria základ >90 % produkčných
# systémov (InsightFace, KYC, hraničné kontroly, CCTV) a majú najhustejšie uhlové zhluky na hypersfére.
DEFAULT_ARCH_PRIORS = {
    "ArcFace": 4.60,
    "AdaFace": 2.45,
    "FaceNet": 1.15,
    "FaceNet (CASIA)": 2.15,
    "BenchmarkCNN": 1.35,
    "BenchmarkCNN (Finetuned)": 1.35,
}

DEFAULT_MARGIN_LOW = 0.22   # prísna hranica pre bezpečnú rezervu pod prahom 0.50 pri fyzickej tlači

# Priehľadnosť materiálu (0.94 zabezpečuje plný kontrast matného papiera pri zachovaní fotometrie).
PATCH_BLEND_FACTOR = 0.94


def _anchors_tensor(device):
    return torch.tensor(PRINTABLE_ANCHORS, dtype=torch.float32, device=device)


def _nps_loss(patch: torch.Tensor) -> torch.Tensor:
    """Non-Printability Score: každý pixel tlačíme k najbližšiemu anchorovi (tlačiteľný gamut)."""
    p = patch.squeeze(0).permute(1, 2, 0).reshape(-1, 3)          # (N, 3)
    anchors = _anchors_tensor(patch.device)                       # (K, 3)
    d = ((p.unsqueeze(1) - anchors.unsqueeze(0)) ** 2).sum(-1)    # (N, K)
    return d.min(dim=1).values.mean()


def _tv_loss(patch: torch.Tensor) -> torch.Tensor:
    dh = patch[:, :, 1:, :] - patch[:, :, :-1, :]
    dw = patch[:, :, :, 1:] - patch[:, :, :, :-1]
    return (dh.pow(2).sum() + dw.pow(2).sum()) / patch.numel()


def _naturalness_loss(patch: torch.Tensor) -> torch.Tensor:
    """
    Naturalness: potláča vysokofrekvenčný šum (podporuje nízkofrekvenčnú,
    produktovú textúru) + ťahá priemernú farbu k produktovej palete.
    """
    lp = F.avg_pool2d(patch, 3, stride=1, padding=1)
    hf = (patch - lp).pow(2).mean()
    mean_col = patch.squeeze(0).mean(dim=(1, 2))                  # (3,)
    anchors = _anchors_tensor(patch.device)
    pal = ((mean_col.unsqueeze(0) - anchors) ** 2).sum(-1).min()
    return hf + 0.25 * pal


def _ti_smooth_grad(g: torch.Tensor, k: int = 5) -> torch.Tensor:
    """TI-FGSM (Dong et al., CVPR 2019): separabilné Gaussovo vyhladenie priestorového
    gradientu pre invarianciu voči posunu receptívnych polí medzi architektúrami."""
    if g is None or g.dim() != 4 or k <= 1:
        return g
    c = g.shape[1]
    t = torch.arange(k, dtype=g.dtype, device=g.device) - (k - 1) * 0.5
    sigma = 0.3 * ((k - 1) * 0.5 - 1.0) + 0.8
    w1d = torch.exp(-(t ** 2) / (2.0 * sigma * sigma))
    w1d = w1d / w1d.sum()
    kh = w1d.view(1, 1, k, 1).expand(c, 1, k, 1)
    kw = w1d.view(1, 1, 1, k).expand(c, 1, 1, k)
    pad = k // 2
    out = F.conv2d(F.pad(g, (0, 0, pad, pad), mode="replicate"), kh, groups=c)
    out = F.conv2d(F.pad(out, (pad, pad, 0, 0), mode="replicate"), kw, groups=c)
    return out


def _tex_size_for(key: str) -> tuple[int, int]:
    """Vysokorozlíšený textúrový buffer: páska 96x256, každá nálepka 96x96."""
    return TEX_STRIP_SIZE if key == "strip" else TEX_DOT_SIZE


def _build_region_specs(patches, boxes):
    """region_specs = {key: (tex_CHW, box=(y1,y2,x1,x2), tex_size=(H,W))} pre _build_canvas."""
    specs = {}
    for key, tex in patches.items():
        box = boxes.get(key)
        if box is None:
            continue
        y1, y2, x1, x2 = box
        specs[key] = (tex, (y1, y2, x1, x2), _tex_size_for(key))
    return specs


def _eot_forward(canvas: torch.Tensor, image: torch.Tensor, full_mask: torch.Tensor,
                 eot: bool, K: int | None = None, boxes: dict | None = None):
    """Fyzikálne EoT: kontaktný tieň + jitter nalepenia + perspektívna homografia + kamera."""
    if K is None:
        K = EOT_BATCH
    if not eot:
        adv = composite_realistic_stickers(image, canvas, full_mask, boxes=boxes)
        return adv, None
    b, _, h, w = canvas.shape
    k = K if b == 1 else b
    drop_shadow = (boxes or {}).get("drop_shadow")
    if drop_shadow is not None:
        ds = drop_shadow.to(image.device)
        img_sh = torch.clamp((image * 0.5 + 0.5) * (1.0 - ds), 0.0, 1.0) * 2.0 - 1.0
    else:
        img_sh = image
    canvas_r = canvas.repeat(k, 1, 1, 1) if b == 1 else canvas
    image_r = image.repeat(k, 1, 1, 1) if b == 1 else image
    img_sh_r = img_sh.repeat(k, 1, 1, 1) if b == 1 else img_sh
    mask_r = full_mask.repeat(k, 1, 1, 1) if full_mask.shape[0] == 1 else full_mask
    canvas_r, mask_r = _patch_placement_jitter(canvas_r, mask_r)
    grid = _eot_grid(k, h, w, canvas.device)
    trans_canvas = F.grid_sample(canvas_r, grid, mode="bilinear", padding_mode="border", align_corners=False)
    trans_image = F.grid_sample(image_r, grid, mode="bilinear", padding_mode="border", align_corners=False)
    trans_img_sh = F.grid_sample(img_sh_r, grid, mode="bilinear", padding_mode="border", align_corners=False)
    trans_mask = F.grid_sample(mask_r, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    trans_mask = torch.clamp(trans_mask, 0.0, 1.0)
    adv = (1.0 - trans_mask) * trans_img_sh + trans_mask * trans_canvas
    base = _camera_pipeline(trans_image.contiguous())
    return _camera_pipeline(adv), base


def _div_params(h: int, w: int, prob: float = 0.65):
    """DI-FGSM (Xie et al. 2019) input diversity: s pravdepodobnosťou `prob`
    vráti parametre (nh, nw, oh, ow) jemného resize+pad/crop pre medzimodelový transfer."""
    if float(torch.rand(1).item()) >= prob:
        return None
    s = float(torch.empty(1).uniform_(0.92, 1.08).item())
    nh, nw = max(8, int(round(h * s))), max(8, int(round(w * s)))
    oh = int(torch.randint(0, abs(nh - h) + 1, (1,)).item())
    ow = int(torch.randint(0, abs(nw - w) + 1, (1,)).item())
    return (nh, nw, oh, ow)


def _apply_div(x: torch.Tensor, params) -> torch.Tensor:
    """Aplikuje DI transformáciu z _div_params (resize + replicate pad/crop späť na (h, w))."""
    if params is None:
        return x
    nh, nw, oh, ow = params
    _, _, h, w = x.shape
    x_r = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
    if nh >= h:
        x_r = x_r[:, :, oh:oh + h, :]
    else:
        x_r = F.pad(x_r, (0, 0, oh, h - nh - oh), mode="replicate")
    if nw >= w:
        x_r = x_r[:, :, :, ow:ow + w]
    else:
        x_r = F.pad(x_r, (ow, w - nw - ow, 0, 0), mode="replicate")
    return x_r


def _forward_raw(model, x):
    """(norm_emb, multi_stage_raw_feat) — raw obsahuje L2-normalizované konvolučné
    feature mapy stredných vrstiev + výstupný embedding."""
    try:
        out = model(x, return_raw=True)
    except TypeError:
        out = model(x)
    if isinstance(out, tuple):
        return out[0], out[1]
    return out, out


def apply_patch_set(patches: dict, image: torch.Tensor, landmarks=None, mtcnn=None,
                    dot_radius: float = 5.0, dot_pattern: str = "cheeks_chin",
                    strip_w_scale: float = 1.0,
                    strip_ox: float = 0.0, strip_oy: float = 0.0,
                    left_ox: float = 0.0, left_oy: float = 0.0,
                    right_ox: float = 0.0, right_oy: float = 0.0,
                    chin_ox: float = 0.0, chin_oy: float = 0.0) -> tuple:
    """
    Aplikuje NATRÉNOVANÚ sadu patchov (info["patches"] z tréningu) na NOVÚ
    fotku — iná identita/frame než v tréningu (univerzálny patch, held-out eval).
    """
    device = image.device
    mesh_info = None
    if landmarks is None:
        lm, mesh_info = extract_landmarks(image, mtcnn=mtcnn, return_mesh=True)
    elif isinstance(landmarks, tuple):
        lm, mesh_info = landmarks
    else:
        lm = landmarks
        _, mesh_info = extract_landmarks(image, mtcnn=mtcnn, return_mesh=True)
    full_mask, full_relief, boxes = get_hybrid_geometry(
        landmarks=lm, mesh_info=mesh_info,
        dot_radius=dot_radius, dot_pattern=dot_pattern, strip_w_scale=strip_w_scale,
        strip_ox=strip_ox, strip_oy=strip_oy,
        left_ox=left_ox, left_oy=left_oy,
        right_ox=right_ox, right_oy=right_oy,
        chin_ox=chin_ox, chin_oy=chin_oy,
    )
    full_mask = full_mask.to(device)
    full_relief = full_relief.to(device)
    tex = {k: (v.unsqueeze(0) if v.dim() == 3 else v).to(device) for k, v in patches.items()}
    region_specs = _build_region_specs(tex, boxes)
    gain_full = _illumination_gain(image, boxes=boxes)
    canvas, _ = _build_canvas(image, region_specs, gain_full, full_relief, PATCH_BLEND_FACTOR, boxes=boxes)
    adv = composite_realistic_stickers(image, canvas, full_mask, boxes=boxes)
    info = {
        "face_occlusion_pct": float((full_mask.sum() / (112 * 112) * 100.0).item()),
        "boxes": boxes,
        "landmarks": lm,
    }
    return adv, full_mask, info


def ensemble_hybrid_patch_attack(
    models: dict,
    image: torch.Tensor,
    images: list | None = None,
    num_iter: int = 100,
    lr: float = 0.11,
    eot: bool = True,
    margin_low: dict | None = None,
    targets: dict | None = None,
    strip_style: str = "rgb",
    dot_radius: float = 5.0,
    dot_pattern: str = "cheeks_chin",
    strip_w_scale: float = 1.0,
    nps_weight: float = 0.004,
    tv_weight: float = 0.008,
    naturalness_weight: float = 0.004,
    reg_weight: float = 0.002,
    pad_model=None,
    pad_pairs: list | None = None,
    pad_weight: float = 0.5,
    mtcnn=None,
    landmarks=None,
    output_path: str | None = None,
    mm_per_px: float | None = None,
    seed: int | None = None,
    grad_avg: bool = False,
    momentum: float = 0.0,
    input_diversity: bool = False,
    feature_weight: float = 0.0,
    feature_margin: float = 0.5,
    update_rule: str = "adam",
    multiscale: bool = True,
    ti_kernel: int = 5,
    strip_ox: float = 0.0,
    strip_oy: float = 0.0,
    left_ox: float = 0.0,
    left_oy: float = 0.0,
    right_ox: float = 0.0,
    right_oy: float = 0.0,
    chin_ox: float = 0.0,
    chin_oy: float = 0.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    palette: list | None = None,
    logo_mask: torch.Tensor | None = None,
    gallery_centroids: dict[str, torch.Tensor] | None = None,
    arch_priors: dict[str, float] | None = None,
    impostor_centroids: dict[str, torch.Tensor] | None = None,
    impostor_weight: float = 0.45,
) -> tuple:
    """
    BioMesh-SpecPatch: Ensemblový 3D-geometrický frekvenčne-ukotvený hybridný útok (Téza D).
    Kombinuje:
      1. Multi-Scale Spectral Pyramid parametrizáciu textúr (nízko- a strednofrekvenčné
         štruktúrne kontúry pre black-box transfer a fyzickú tlačiteľnosť),
      2. Mid-Level Convolutional Feature Dispersion cez priestorové vrstvy (Stage 2–4),
      3. TI-FGSM Gaussovo vyhladenie priestorového gradientu + DI-FGSM input diversity,
      4. RMS-normalizované spriemerovanie gradientov modelov (Cosine/RMS Gradient Alignment),
      5. PAD-aware ko-optimalizáciu pre zachovanie liveness skóre (MiniFASNetV2).
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    device = image.device

    # --- Multi-image príprava ---
    image_list = [image]
    if images:
        for im in images:
            if im is not None:
                image_list.append(im.to(device))

    # Geometria a predpočítané orig embeddingy pre každú (fotka, model) kombináciu.
    geo = []
    is_lm_seq = (
        isinstance(landmarks, list)
        and len(landmarks) > 0
        and isinstance(landmarks[0], tuple)
        and len(landmarks[0]) == 2
        and isinstance(landmarks[0][0], list)
    )
    for i_idx, base in enumerate(image_list):
        mesh_info = None
        if is_lm_seq and i_idx < len(landmarks):
            lm, mesh_info = landmarks[i_idx]
        elif i_idx == 0 and landmarks is not None and not is_lm_seq:
            if isinstance(landmarks, tuple):
                lm, mesh_info = landmarks
            else:
                lm = landmarks
                _, mesh_info = extract_landmarks(base, mtcnn=mtcnn, return_mesh=True)
        else:
            lm, mesh_info = extract_landmarks(base, mtcnn=mtcnn, return_mesh=True)
        full_mask, full_relief, boxes = get_hybrid_geometry(
            landmarks=lm, mesh_info=mesh_info,
            dot_radius=dot_radius, dot_pattern=dot_pattern, strip_w_scale=strip_w_scale,
            offset_x=offset_x, offset_y=offset_y,
            strip_ox=strip_ox, strip_oy=strip_oy,
            left_ox=left_ox, left_oy=left_oy,
            right_ox=right_ox, right_oy=right_oy,
            chin_ox=chin_ox, chin_oy=chin_oy,
        )
        full_mask_d = full_mask.to(device)
        full_relief_d = full_relief.to(device)
        gain_full_d = _illumination_gain(base, boxes=boxes).detach()
        geo.append((base, full_mask_d, full_relief_d, boxes, lm, gain_full_d))
    close_mediapipe_landmarker()

    margin = {name: float((margin_low or {}).get(name, DEFAULT_MARGIN_LOW)) for name in models}
    targets = targets or {}

    # Automatické načítanie impostor centroidov (David Beckham / cudzí klaster) pre Bod 2
    if impostor_centroids is None and impostor_weight > 0.0:
        try:
            import os
            gal_path = os.path.join(os.path.dirname(__file__), "..", "data", "gallery_centroids.pt")
            if os.path.exists(gal_path):
                c_data = torch.load(gal_path, map_location=device, weights_only=False)
                g_dict = c_data.get("galleries", c_data)
                imp_dict = {}
                for m_name in models:
                    if m_name in g_dict:
                        for id_name, id_vec in g_dict[m_name].items():
                            if "peter" not in id_name.lower() and "brandajsky" not in id_name.lower():
                                imp_dict[m_name] = id_vec.to(device)
                                break
                if imp_dict:
                    impostor_centroids = imp_dict
        except Exception:
            impostor_centroids = None

    is_rgb = any(k in strip_style.lower() for k in ("rgb", "farebn", "art", "adversarial")) or (palette is not None)
    if is_rgb and palette is None:
        base_strip = torch.full((1, 3, 1, 1), 0.16, device=device)
        base_dot = torch.full((1, 3, 1, 1), 0.16, device=device)
    elif strip_style in ["pro_carbon_logo", "carbon_sport"]:
        base_strip = torch.tensor([-0.50, -0.50, -0.48], device=device).view(1, 3, 1, 1)
        base_dot = torch.tensor([-0.48, -0.48, -0.46], device=device).view(1, 3, 1, 1)
    elif strip_style == "tan_fabric":
        base_strip = torch.tensor([0.35, 0.20, 0.08], device=device).view(1, 3, 1, 1)
        base_dot = torch.tensor([0.35, 0.20, 0.08], device=device).view(1, 3, 1, 1)
    elif strip_style in ["clear_silicone", "clear"]:
        base_strip = torch.tensor([0.45, 0.45, 0.48], device=device).view(1, 3, 1, 1)
        base_dot = torch.tensor([0.35, 0.20, 0.08], device=device).view(1, 3, 1, 1)
    else:
        # Športová čierna páska + čierne športové nálepky
        base_strip = torch.tensor([-0.52, -0.52, -0.50], device=device).view(1, 3, 1, 1)
        base_dot = torch.tensor([-0.50, -0.50, -0.48], device=device).view(1, 3, 1, 1)

    # Jemná diagonálna karbónová textúra pre športové čierne varianty
    def _make_carbon_weave(th_c: int, tw_c: int, amp: float = 0.06) -> torch.Tensor:
        yg = torch.arange(th_c, device=device, dtype=torch.float32).view(-1, 1)
        xg = torch.arange(tw_c, device=device, dtype=torch.float32).view(1, -1)
        u = (xg / float(tw_c)) * (22.0 if tw_c > 128 else 9.0)
        v = (yg / float(th_c)) * 7.0
        return (amp * torch.sin((u + v) * 2.5) * torch.cos((u - v) * 2.5)).view(1, 1, th_c, tw_c)

    carbon_weave_strip = _make_carbon_weave(*TEX_STRIP_SIZE, amp=0.065)
    carbon_weave_dot = _make_carbon_weave(*TEX_DOT_SIZE, amp=0.055)

    palette_t = None
    if palette is not None:
        palette_t = torch.tensor(palette, dtype=torch.float32, device=device)
        if float(palette_t.min()) >= -1e-6 and float(palette_t.max()) <= 1.0 + 1e-6:
            palette_t = palette_t * 2.0 - 1.0

    def _prep_tex(tex, key):
        if palette_t is not None:
            tex = _quantize_to_palette(tex, palette_t)
        if logo_mask is not None:
            tex = apply_logo_emblem(tex, key, logo_mask, strip_style=strip_style, is_rgb=is_rgb)
        return tex

    active_keys = [k for k in ("strip", "left_dot", "right_dot", "chin_dot") if geo[0][3].get(k) is not None]
    params = nn.ParameterDict()
    for key in active_keys:
        th, tw = _tex_size_for(key)
        if multiscale:
            c_h, c_w = (12, 30) if key == "strip" else (12, 12)
            m_h, m_w = (24, 60) if key == "strip" else (24, 24)
            params[f"{key}_coarse"] = nn.Parameter(torch.zeros((1, 3, c_h, c_w), device=device))
            params[f"{key}_mid"] = nn.Parameter(torch.zeros((1, 3, m_h, m_w), device=device))
        else:
            params[key] = nn.Parameter(torch.zeros((1, 3, th, tw), device=device))
    param_list = [params[k] for k in params]

    opt = torch.optim.Adam(param_list, lr=lr, betas=(0.85, 0.999))

    def _current_patches():
        out = {}
        for key in active_keys:
            init = base_strip if key == "strip" else base_dot
            th, tw = _tex_size_for(key)
            if multiscale:
                c_up = F.interpolate(params[f"{key}_coarse"], size=(th, tw), mode="bicubic", align_corners=False)
                m_up = F.interpolate(params[f"{key}_mid"], size=(th, tw), mode="bicubic", align_corners=False)
                delta = _gblur_fixed(0.95 * c_up + 0.78 * m_up, sigma=0.75)
            else:
                delta = params[key]

            if not is_rgb:
                delta_lum = delta.mean(dim=1, keepdim=True)
                delta_chr = (delta - delta_lum).clamp(-0.35, 0.35)
                delta_wb = (delta[:, 0:1] - delta[:, 2:3]).clamp(-0.50, 0.50)
                v_wb = torch.tensor([1.0, 0.35, -0.85], device=device).view(1, 3, 1, 1)
                if strip_style == "tan_fabric":
                    c_tan = torch.tensor([0.28, 0.12, -0.04], device=device).view(1, 3, 1, 1)
                    s_tan = torch.tensor([0.75, 0.68, 0.60], device=device).view(1, 3, 1, 1)
                    raw = torch.clamp(c_tan + delta_lum * s_tan + delta_wb * v_wb * 0.25 + delta_chr * 0.20, -0.35, 0.88)
                elif strip_style in ["clear_silicone", "clear"]:
                    raw = torch.clamp(init + delta_lum * 0.80 + delta_wb * v_wb * 0.22 + delta_chr * 0.22, -0.60, 0.92)
                else:
                    # Skutočná tmavá športová / karbónová paleta (žiadne biele/žlté fľaky, zachovaný silný kontrast [-0.98, +0.14])
                    weave = carbon_weave_strip if key == "strip" else carbon_weave_dot
                    c_dark = init + weave
                    raw = c_dark + delta_lum * 0.78 + delta_wb * v_wb * 0.16 + delta_chr * 0.16
                    raw = torch.clamp(raw, -0.98, 0.14)
            else:
                # Plný tlačiteľný RGB rozsah bez klipovania na extrémne +-1.0
                raw = torch.clamp(init + delta, -0.96, 0.96)

            raw = _prep_tex(raw, key)
            out[key] = raw
        return out

    # --- PAD baseline (ak je hook poskytnutý) ---
    pad_baseline = None
    if pad_model is not None:
        ref_list = pad_pairs if pad_pairs else [g[0] for g in geo[:4]]
        try:
            with torch.no_grad():
                scores = [float(pad_model(p[0:1].to(device)).mean()) for p in ref_list if p is not None]
            if scores:
                pad_baseline = float(np.mean(scores))
        except Exception:
            pad_baseline = None

    # --- Cache orig embeddingov (norm + multi-stage raw feature) ---
    orig_cache = {}

    def _orig_emb_and_raw(base_idx, model_name):
        key = (base_idx, model_name)
        if key not in orig_cache:
            with torch.no_grad():
                emb, raw = _forward_raw(models[model_name], geo[base_idx][0])
                orig_cache[key] = (emb.detach(), raw.detach())
        return orig_cache[key]

    def _orig_emb(base_idx, model_name):
        return _orig_emb_and_raw(base_idx, model_name)[0]

    def _orig_raw(base_idx, model_name):
        return _orig_emb_and_raw(base_idx, model_name)[1]

    def _flat_grads(grads, smooth: bool = False, norm_per_tensor: bool = True):
        flats = []
        for p, g in zip(param_list, grads):
            if g is None:
                flats.append(torch.zeros(p.numel(), device=device))
            else:
                g_proc = _ti_smooth_grad(g, ti_kernel) if (smooth and g.shape[-1] >= 32) else g
                if norm_per_tensor:
                    t_rms = g_proc.pow(2).mean().sqrt().clamp_min(1e-8)
                    g_proc = g_proc / t_rms
                flats.append(g_proc.reshape(-1))
        return torch.cat(flats)

    use_ga = bool(grad_avg) and len(models) > 0
    n_params = sum(p.numel() for p in param_list)
    mom_flat = torch.zeros(n_params, device=device) if (use_ga and momentum > 0.0) else None
    batch_photos = min(3, len(geo)) if len(geo) > 1 else 1
    eot_k = max(3, EOT_BATCH // batch_photos)

    # --- Optimalizácia (zmrazenie váh CNN modelov šetrí ~650 MB .grad bufferov a dL/dW autograd) ---
    saved_grad_flags: list[tuple[nn.Parameter, bool]] = []
    for m_obj in list(models.values()) + ([pad_model] if pad_model is not None else []):
        if isinstance(m_obj, nn.Module):
            for p_m in m_obj.parameters():
                if p_m.requires_grad:
                    saved_grad_flags.append((p_m, True))
                    p_m.requires_grad_(False)

    best_score = float("inf")
    best_params = None

    try:
      for it in range(num_iter):
        opt.zero_grad(set_to_none=True)

        if len(geo) == 1:
            selected_indices = [0]
        else:
            aux_perm = [i + 1 for i in torch.randperm(len(geo) - 1)[:max(1, batch_photos - 1)].tolist()]
            selected_indices = [0] + aux_perm

        patches = _current_patches()

        # Pre každú fotku v mini-batchi pripravíme EoT a clean vstupy
        step_batches = []
        for idx in selected_indices:
            base, full_mask, full_relief, boxes, _, gain_full = geo[idx]
            region_specs = _build_region_specs(patches, boxes)
            canvas, _ = _build_canvas(base, region_specs, gain_full, full_relief, PATCH_BLEND_FACTOR, boxes=boxes)

            model_input, base_warps = _eot_forward(canvas, base, full_mask, eot, K=eot_k, boxes=boxes)
            model_input = model_input.contiguous()
            clean_input = composite_realistic_stickers(base, canvas, full_mask, boxes=boxes).contiguous()

            if input_diversity:
                div = _div_params(*model_input.shape[-2:])
                model_input = _apply_div(model_input, div)
                if base_warps is not None:
                    base_warps = _apply_div(base_warps, div)

            step_batches.append((idx, model_input, base_warps, clean_input))

        # Regularizátory (pre plné RGB bez palety netlačíme farby k béžovým anchorom)
        tv = sum(_tv_loss(p) for p in patches.values()) / max(1, len(patches))
        reg = sum(p.pow(2).mean() for p in params.values()) / max(1, len(params))
        if is_rgb and palette is None:
            ink_save = sum(F.relu(0.05 - p.mean()) for p in patches.values()) / max(1, len(patches))
            reg_total = tv_weight * tv + reg_weight * reg + 0.045 * ink_save
        else:
            nps = sum(_nps_loss(p) for p in patches.values()) / max(1, len(patches))
            nat = sum(_naturalness_loss(p) for p in patches.values()) / max(1, len(patches))
            reg_total = nps_weight * nps + tv_weight * tv + naturalness_weight * nat + reg_weight * reg

        # PAD-AWARE penalizácia (chráni liveness >= 0.64 bez VRAM preťaženia GPU)
        pad_term = None
        pad_score = None
        if pad_model is not None:
            try:
                pad_eot_s = pad_model(step_batches[0][1]).mean()
                pad_clean_s = pad_model(step_batches[0][3]).mean()
                pad_score = 0.60 * pad_clean_s + 0.40 * pad_eot_s
                target_pad = min(0.65, max(0.58, pad_baseline - 0.05)) if pad_baseline is not None else 0.62
                pad_term = pad_weight * (F.relu(target_pad - pad_clean_s) + 0.6 * F.relu(0.55 - pad_eot_s))
            except Exception:
                pad_term = None

        def _model_loss(name, m):
            total_m = torch.zeros((), device=device)
            sim_primary = None
            sim_aux_list = []
            n_aux = max(1, len(step_batches) - 1)
            is_targeted = (name in targets and targets[name] is not None)
            use_raw = (not is_targeted) and (feature_weight > 0.0)

            # Zlúčenie všetkých fotiek v mini-batchi do 1 spoločného GPU forwardu na model
            # (znižuje počet CUDA kernel launchov 4.5× pri identickom výstupe v .eval() režime).
            grad_chunks = []
            grad_slices = []
            warp_chunks = []
            warp_map = {}
            g_off = 0
            w_off = 0
            for sb_i, (_, m_in, b_w, c_in) in enumerate(step_batches):
                km, kc = m_in.shape[0], c_in.shape[0]
                grad_chunks.append(m_in)
                grad_chunks.append(c_in)
                grad_slices.append((g_off, g_off + km, g_off + km + kc))
                g_off += km + kc
                if (not is_targeted) and (b_w is not None):
                    kw = b_w.shape[0]
                    warp_chunks.append(b_w)
                    warp_map[sb_i] = (w_off, w_off + kw)
                    w_off += kw

            grad_cat = torch.cat(grad_chunks, dim=0)
            if use_raw:
                all_emb, all_raw = _forward_raw(m, grad_cat)
            else:
                all_emb = m(grad_cat)
                all_raw = None

            if warp_chunks:
                warp_cat = torch.cat(warp_chunks, dim=0)
                with torch.no_grad():
                    if use_raw:
                        all_w_emb, all_w_raw = _forward_raw(m, warp_cat)
                        all_w_emb, all_w_raw = all_w_emb.detach(), all_w_raw.detach()
                    else:
                        all_w_emb = m(warp_cat).detach()
                        all_w_raw = None
            else:
                all_w_emb = all_w_raw = None

            for sb_i, (idx, model_input, base_warps, clean_input) in enumerate(step_batches):
                s0, s1, s2 = grad_slices[sb_i]
                adv_emb = all_emb[s0:s1]
                clean_emb = all_emb[s1:s2]
                adv_raw = all_raw[s0:s1] if all_raw is not None else None
                clean_raw = all_raw[s1:s2] if all_raw is not None else None

                pose_w = 1.0 / float(len(step_batches))
                if is_targeted:
                    s_i = 0.60 * F.cosine_similarity(targets[name], adv_emb).mean() \
                          + 0.40 * F.cosine_similarity(targets[name], clean_emb).mean()
                    total_m = total_m - pose_w * s_i
                    val_s = float(-s_i.detach().item())
                    if idx == 0:
                        sim_primary = val_s
                    else:
                        sim_aux_list.append(val_s)
                    continue

                orig_emb, orig_raw = _orig_emb_and_raw(idx, name)
                if sb_i in warp_map:
                    w0, w1 = warp_map[sb_i]
                    base_emb = all_w_emb[w0:w1]
                    base_raw = all_w_raw[w0:w1] if all_w_raw is not None else None
                    sim_eot = F.cosine_similarity(base_emb, adv_emb).mean()
                else:
                    base_raw = orig_raw
                    sim_eot = F.cosine_similarity(orig_emb, adv_emb).mean()

                sim_clean = F.cosine_similarity(orig_emb, clean_emb).mean()
                sim = 0.60 * sim_eot + 0.40 * sim_clean
                if gallery_centroids and name in gallery_centroids and gallery_centroids[name] is not None:
                    g_emb = gallery_centroids[name]
                    sim_g_eot = F.cosine_similarity(g_emb, adv_emb).mean()
                    sim_g_clean = F.cosine_similarity(g_emb, clean_emb).mean()
                    sim_g = 0.60 * sim_g_eot + 0.40 * sim_g_clean
                    hinge_g = 1.8 * F.relu(sim_g_eot - 0.33) + 1.6 * F.relu(sim_g_clean - 0.33)
                    sim = 0.35 * sim + 0.65 * sim_g + hinge_g
                    val_s = float((0.55 * sim_g_eot + 0.45 * sim_g_clean).detach().item())
                else:
                    val_s = float((0.55 * sim_eot + 0.45 * sim_clean).detach().item())
                if idx == 0:
                    sim_primary = val_s
                else:
                    sim_aux_list.append(val_s)
                l_i = sim + 1.7 * F.relu(sim_eot - 0.34) + 1.4 * F.relu(sim_clean - 0.34) + 1.0 * F.relu(sim - margin[name])

                # Impostor-Anchor Repulsion (Bod 2): aktívne ťahá adv embedding do cudzieho klastra (512-D hypersféra),
                # čím odstraňuje stagnáciu gradientu v okolí rovníka (ArcFace / AdaFace hlboko pod prah).
                if impostor_centroids and name in impostor_centroids and impostor_centroids[name] is not None and impostor_weight > 0.0:
                    imp_emb = impostor_centroids[name]
                    if imp_emb.dim() == 1:
                        imp_emb = imp_emb.unsqueeze(0)
                    sim_imp_eot = F.cosine_similarity(imp_emb, adv_emb).mean()
                    sim_imp_clean = F.cosine_similarity(imp_emb, clean_emb).mean()
                    sim_imp = 0.60 * sim_imp_eot + 0.40 * sim_imp_clean
                    l_i = l_i - impostor_weight * sim_imp + 0.25 * F.relu(0.30 - sim_imp)

                if use_raw and adv_raw is not None:
                    feat_eot = F.cosine_similarity(base_raw, adv_raw, dim=1).mean()
                    feat_clean = F.cosine_similarity(orig_raw, clean_raw, dim=1).mean()
                    feat_sim = 0.60 * feat_eot + 0.40 * feat_clean
                    l_i = l_i + feature_weight * feat_sim

                total_m = total_m + pose_w * l_i
            all_s = ([sim_primary] if sim_primary is not None else []) + sim_aux_list
            if not all_s:
                sim_out = 0.0
            else:
                sim_out = 0.45 * (sum(all_s) / len(all_s)) + 0.55 * max(all_s)
            return total_m, sim_out

        active_priors = arch_priors if arch_priors is not None else DEFAULT_ARCH_PRIORS

        if use_ga:
            model_grads = []
            cur_sims = []
            for name, m in models.items():
                m_loss, m_sim = _model_loss(name, m)
                cur_sims.append(m_sim)
                g = torch.autograd.grad(m_loss, param_list,
                                        retain_graph=True, allow_unused=True)
                del m_loss
                flat = _flat_grads(g, smooth=True, norm_per_tensor=False)
                rms = flat.pow(2).mean().sqrt().clamp_min(1e-8)
                hardness_w = (max(0.10, m_sim) ** 2.2) * (2.5 if m_sim > 0.38 else 0.5)
                arch_w = active_priors.get(name, 1.0)
                model_grads.append((flat / rms) * (hardness_w * arch_w))
            if cur_sims:
                over_50 = sum(max(0.0, s - 0.46) * 10.0 for s in cur_sims)
                pad_pen = max(0.0, 0.54 - float(pad_score.detach().item())) * 6.0 if pad_score is not None else 0.0
                sc = over_50 + pad_pen + (sum(cur_sims) / len(cur_sims))
                if sc < best_score:
                    best_score = sc
                    best_params = {k: v.detach().clone() for k, v in params.items()}
                stop_thr = 0.12 if input_diversity else 0.30
                if it >= 55 and all(s < stop_thr for s in cur_sims) and (pad_score is None or pad_score >= 0.50):
                    opt.zero_grad(set_to_none=True)
                    break
            direction = torch.stack(model_grads).sum(0)
            dir_rms = direction.pow(2).mean().sqrt().clamp_min(1e-8)
            direction = direction / dir_rms

            aux = reg_total if pad_term is None else reg_total + pad_term
            g_aux = _flat_grads(torch.autograd.grad(aux, param_list, allow_unused=True), smooth=False, norm_per_tensor=False)
            del aux, reg_total, pad_term, step_batches, patches
            g_aux_rms = g_aux.pow(2).mean().sqrt().clamp_min(1e-8)
            aux_weight = 0.16 if (pad_score is not None and pad_score < 0.52) else 0.05
            direction = direction + aux_weight * (g_aux / g_aux_rms)

            if mom_flat is not None:
                d_norm = direction / direction.pow(2).mean().sqrt().clamp_min(1e-8)
                if update_rule == "sign":
                    mom_flat = momentum * mom_flat + d_norm
                else:
                    mom_flat = 0.55 * mom_flat + 0.45 * d_norm
                direction = mom_flat

            offset = 0
            for p in param_list:
                p.grad = direction[offset:offset + p.numel()].view_as(p).clone()
                offset += p.numel()

            if update_rule == "sign":
                cos_decay = 0.5 * (1.0 + math.cos(math.pi * it / max(1, num_iter)))
                step_sz = lr * (0.35 + 0.65 * cos_decay)
                with torch.no_grad():
                    for p in param_list:
                        p.add_(-step_sz * p.grad.sign())
                opt.zero_grad(set_to_none=True)
            else:
                opt.step()
        else:
            # Sekvenčná akumulácia gradientov po jednom modeli: vo VRAM žije vždy len 1 CNN graf
            # (~1.1 GB namiesto 5.84 GB -> nulový WDDM PCIe paging vo WSL2 na 6 GB GPU).
            aux_total = reg_total if pad_term is None else (reg_total + pad_term)
            model_items = list(models.items())
            if model_items:
                aux_total.backward(retain_graph=True)
            else:
                aux_total.backward()
            del aux_total, reg_total, pad_term

            cur_sims = []
            n_models = len(model_items)
            for m_idx, (name, m) in enumerate(model_items):
                m_loss, m_sim = _model_loss(name, m)
                cur_sims.append(m_sim)
                w_m = active_priors.get(name, 1.0) * (
                    1.0 + 5.5 * max(0.0, m_sim - 0.34) + 8.5 * max(0.0, m_sim - 0.44)
                )
                (w_m * m_loss).backward(retain_graph=(m_idx < n_models - 1))
                del m_loss
            del step_batches, patches

            if cur_sims:
                over_50 = sum(max(0.0, s - 0.45) * 16.0 for s in cur_sims)
                pad_pen = max(0.0, 0.60 - float(pad_score.detach().item())) * 8.0 if pad_score is not None else 0.0
                sc = over_50 + pad_pen + (sum(cur_sims) / len(cur_sims))
                if sc < best_score:
                    best_score = sc
                    best_params = {k: v.detach().clone() for k, v in params.items()}
                stop_thr = 0.12 if input_diversity else 0.30
                if it >= 55 and all(s < stop_thr for s in cur_sims) and (pad_score is None or pad_score >= 0.50):
                    opt.zero_grad(set_to_none=True)
                    break

            with torch.no_grad():
                for p in param_list:
                    if p.grad is not None and p.shape[-1] >= 32:
                        p.grad.copy_(_ti_smooth_grad(p.grad, ti_kernel))
            opt.step()

        with torch.no_grad():
            for p in params.values():
                p.clamp_(-1.8, 1.8)
    finally:
        for p_m, flag in saved_grad_flags:
            p_m.requires_grad_(flag)

    # --- Finálna kompozícia (bez EOT) ---
    with torch.no_grad():
        if best_params is not None:
            for k, v in best_params.items():
                params[k].copy_(v)
        base, full_mask, full_relief, boxes, lm, gain_full = geo[0]
        patches = _current_patches()
        region_specs = _build_region_specs(patches, boxes)
        canvas, comps = _build_canvas(base, region_specs, gain_full, full_relief, PATCH_BLEND_FACTOR, boxes=boxes)
        final_adv = composite_realistic_stickers(base, canvas, full_mask, boxes=boxes)

        per_model_sim = {}
        per_model_success = {}
        for name, m in models.items():
            emb = m(final_adv)
            if name in targets and targets[name] is not None:
                sim = float(F.cosine_similarity(targets[name], emb).mean())
                per_model_sim[name] = sim
                per_model_success[name] = sim > margin[name]
            else:
                orig_emb = _orig_emb(0, name)
                sim = float(F.cosine_similarity(orig_emb, emb).mean())
                per_model_sim[name] = sim
                per_model_success[name] = sim < margin[name]

        occ_pct = float((full_mask.sum() / (112 * 112) * 100.0).item())
        mean_sim = float(np.mean(list(per_model_sim.values()))) if per_model_sim else 0.0
        ensemble_success = bool(all(per_model_success.values())) if per_model_success else False

        pad_final = None
        if pad_model is not None:
            try:
                pad_final = float(pad_model(final_adv).mean().item())
            except Exception:
                pad_final = None

    # Intrinsic patches pre apply_patch_set (bez dvojitej kompozície!)
    intrinsic_patches_dict = {k: patches[k].squeeze(0).detach().cpu() for k in patches}
    # Printable comps pre 300 DPI tlačový hárok
    printable_patches_dict = {k: comps[k].squeeze(0).detach().cpu() for k in comps}

    if mm_per_px is None:
        mm_per_px = derive_mm_per_px(lm)
    mm_per_px = float(mm_per_px)

    final_specs = {}
    for key in ("strip", "left_dot", "right_dot", "chin_dot"):
        box = boxes.get(key)
        if box is not None:
            final_specs[key] = (box, _tex_size_for(key))
    masks_np, geo_print = build_printable_payload(boxes, final_specs, mm_per_px)

    # Tlačový hárok (300 DPI) musí obsahovať čistú intrizickú textúru s plným tlačovým
    # kontrastom a gamutom, bez stmavnutia z webkamery (aby papier v tvojej izbe
    # prirodzene odrážal reálne izbové svetlo a netlačil sa tmavý šum).
    printable_sheet = generate_printable_sheet(
        intrinsic_patches_dict,
        output_path=output_path,
        mm_per_px=mm_per_px,
        landmarks=lm,
        masks=masks_np,
        geometry=geo_print,
    )

    info = {
        "attack": "BioMesh-SpecPatch (Ensemble Hybrid Physical Attack)",
        "strip_style": strip_style,
        "models": list(models.keys()),
        "margin_low": margin,
        "orig_similarity": 1.0,
        "final_similarity": mean_sim,
        "per_model_similarity": per_model_sim,
        "per_model_success": per_model_success,
        "success": ensemble_success,
        "face_occlusion_pct": round(occ_pct, 2),
        "iterations": num_iter,
        "eot": eot,
        "pad_model_used": pad_model is not None and pad_baseline is not None,
        "pad_baseline": pad_baseline,
        "pad_final": pad_final,
        "n_images": len(image_list),
        "grad_avg": use_ga,
        "momentum": momentum if use_ga else 0.0,
        "input_diversity": bool(input_diversity),
        "feature_weight": feature_weight,
        "multiscale": bool(multiscale),
        "update_rule": update_rule if use_ga else "adam",
        "patches": intrinsic_patches_dict,
    }

    return final_adv.detach(), printable_sheet, info
