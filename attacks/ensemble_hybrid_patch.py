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
    TEX_STRIP_SIZE,
    TEX_DOT_SIZE,
    EOT_BATCH,
    _illumination_gain,
    _build_canvas,
    _eot_grid,
    _patch_placement_jitter,
    _camera_pipeline,
    _quantize_to_palette,
    derive_mm_per_px,
    build_printable_payload,
)
from attacks.constellation_patch import extract_landmarks
from attacks.nasal_strip_patch import create_nasal_strip_mask  # noqa: F401  (re-export pre parity)


# Tlačiteľný gamut - malý súbor anchor farieb (kinesio páska / kozmetická nálepka).
# Hodnoty v rozsahu [-1, 1] (rovnaká normalizácia ako vstupné tenzory).
PRINTABLE_ANCHORS = [
    [-0.72, -0.72, -0.72],   # čierna športová páska
    [-1.00, -1.00, -1.00],   # hlboká čierna
    [0.65, 0.44, 0.25],      # telová / beige
    [0.68, 0.45, 0.22],      # hydrokoloidná nálepka
    [0.90, 0.85, 0.70],      # krémová
    [1.00, 1.00, 1.00],      # biela
    [0.05, 0.35, -0.55],     # modrá kinesio
    [0.40, 0.20, 0.10],      # hnedá
]

# Apriórne váhy modelov podľa ich globálneho trhového a priemyselného nasadenia v praxi:
# - Margin loss (ArcFace, AdaFace, CosFace): ~85 % moderných komerčných a bezpečnostných systémov
# - Triplet loss (FaceNet / Inception): ~10 % (legacy / edge systémy)
# - Softmax CNN: < 5 % (akademické baselines)
DEFAULT_ARCH_PRIORS = {
    "ArcFace": 1.5,
    "AdaFace": 1.4,
    "FaceNet": 0.8,
    "FaceNet (CASIA)": 0.7,
    "BenchmarkCNN": 0.4,
    "BenchmarkCNN (Finetuned)": 0.4,
}

DEFAULT_MARGIN_LOW = 0.30   # konzervatívna hranica; kalibruj cez margin_low dict

# Priehľadnosť materiálu (0.93 zabezpečuje plný kontrast matného papiera pri zachovaní fotometrie).
PATCH_BLEND_FACTOR = 0.93


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
                 eot: bool, K: int | None = None):
    """Fyzikálne EoT: jitter nalepenia + perspektívna homografia + crop jitter + kamera."""
    if K is None:
        K = EOT_BATCH
    if not eot:
        adv = ((1.0 - full_mask) * image + full_mask * canvas).clamp(-1.0, 1.0)
        return adv, None
    b, _, h, w = canvas.shape
    k = K if b == 1 else b
    canvas_r = canvas.repeat(k, 1, 1, 1) if b == 1 else canvas
    image_r = image.repeat(k, 1, 1, 1) if b == 1 else image
    mask_r = full_mask.repeat(k, 1, 1, 1) if full_mask.shape[0] == 1 else full_mask
    canvas_r, mask_r = _patch_placement_jitter(canvas_r, mask_r)
    grid = _eot_grid(k, h, w, canvas.device)
    trans_canvas = F.grid_sample(canvas_r, grid, mode="bilinear", padding_mode="border", align_corners=False)
    trans_image = F.grid_sample(image_r, grid, mode="bilinear", padding_mode="border", align_corners=False)
    trans_mask = F.grid_sample(mask_r, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    trans_mask = torch.clamp(trans_mask, 0.0, 1.0)
    adv = (1.0 - trans_mask) * trans_image + trans_mask * trans_canvas
    base = _camera_pipeline(trans_image.contiguous())
    return _camera_pipeline(adv), base


def _div_params(h: int, w: int, prob: float = 0.7):
    """DI-FGSM (Xie et al. 2019) input diversity: s pravdepodobnosťou `prob`
    vráti parametre (nh, nw, oh, ow) náhodného resize+pad/crop; inak None."""
    if float(torch.rand(1).item()) >= prob:
        return None
    s = float(torch.empty(1).uniform_(0.88, 1.12).item())
    nh, nw = max(8, int(round(h * s))), max(8, int(round(w * s)))
    oh = int(torch.randint(0, abs(nh - h) + 1, (1,)).item())
    ow = int(torch.randint(0, abs(nw - w) + 1, (1,)).item())
    return (nh, nw, oh, ow)


def _apply_div(x: torch.Tensor, params) -> torch.Tensor:
    """Aplikuje DI transformáciu z _div_params (resize + pad/crop späť na (h, w))."""
    if params is None:
        return x
    nh, nw, oh, ow = params
    _, _, h, w = x.shape
    x_r = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
    if nh >= h:
        x_r = x_r[:, :, oh:oh + h, :]
    else:
        x_r = F.pad(x_r, (0, 0, oh, h - nh - oh))
    if nw >= w:
        x_r = x_r[:, :, :, ow:ow + w]
    else:
        x_r = F.pad(x_r, (ow, w - nw - ow, 0, 0))
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
                    dot_radius: float = 5.5, dot_pattern: str = "cheeks_chin",
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
    gain_full = _illumination_gain(image)
    canvas, _ = _build_canvas(image, region_specs, gain_full, full_relief, PATCH_BLEND_FACTOR)
    adv = ((1.0 - full_mask) * image + full_mask * canvas).clamp(-1.0, 1.0)
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
    dot_radius: float = 4.5,
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
    for i_idx, base in enumerate(image_list):
        mesh_info = None
        if i_idx == 0 and landmarks is not None:
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
        geo.append((base, full_mask.to(device), full_relief.to(device), boxes, lm))

    margin = {name: float((margin_low or {}).get(name, DEFAULT_MARGIN_LOW)) for name in models}
    targets = targets or {}

    is_rgb = any(k in strip_style.lower() for k in ("rgb", "farebn", "art", "adversarial")) or (palette is not None)
    if is_rgb and palette is None:
        base_strip = torch.zeros((1, 3, 1, 1), device=device)
        base_dot = torch.zeros((1, 3, 1, 1), device=device)
    elif strip_style in ["pro_carbon_logo", "carbon_sport"]:
        base_strip = torch.tensor([-0.65, -0.65, -0.65], device=device).view(1, 3, 1, 1)
        base_dot = torch.tensor([0.35, 0.20, 0.08], device=device).view(1, 3, 1, 1)
    elif strip_style == "tan_fabric":
        base_strip = torch.tensor([0.35, 0.20, 0.08], device=device).view(1, 3, 1, 1)
        base_dot = torch.tensor([0.35, 0.20, 0.08], device=device).view(1, 3, 1, 1)
    elif strip_style in ["clear_silicone", "clear"]:
        base_strip = torch.tensor([0.45, 0.45, 0.48], device=device).view(1, 3, 1, 1)
        base_dot = torch.tensor([0.35, 0.20, 0.08], device=device).view(1, 3, 1, 1)
    else:
        # Čierna športová páska (matná) + prirodzenejšie telové/hydrokoloidné bodky
        base_strip = torch.tensor([-0.72, -0.72, -0.72], device=device).view(1, 3, 1, 1)
        base_dot = torch.tensor([0.30, 0.18, 0.06], device=device).view(1, 3, 1, 1)

    palette_t = None
    if palette is not None:
        palette_t = torch.tensor(palette, dtype=torch.float32, device=device)
        if float(palette_t.min()) >= -1e-6 and float(palette_t.max()) <= 1.0 + 1e-6:
            palette_t = palette_t * 2.0 - 1.0

    def _prep_tex(tex, key):
        if palette_t is not None:
            tex = _quantize_to_palette(tex, palette_t)
        if logo_mask is not None:
            m = logo_mask.to(device)
            if m.dim() == 2:
                m = m.view(1, 1, *m.shape[-2:])
            elif m.dim() == 3:
                m = m.view(1, *m.shape)
            m = F.interpolate(m.float(), size=tex.shape[-2:], mode="nearest")
            base_c = base_strip if key == "strip" else base_dot
            tex = tex * m + base_c.expand_as(tex) * (1.0 - m)
        return tex

    active_keys = [k for k in ("strip", "left_dot", "right_dot", "chin_dot") if geo[0][3].get(k) is not None]
    params = nn.ParameterDict()
    for key in active_keys:
        th, tw = _tex_size_for(key)
        if multiscale:
            params[f"{key}_coarse"] = nn.Parameter(torch.zeros((1, 3, max(6, th // 8), max(6, tw // 8)), device=device))
            params[f"{key}_mid"] = nn.Parameter(torch.zeros((1, 3, max(12, th // 4), max(12, tw // 4)), device=device))
            params[f"{key}_fine"] = nn.Parameter(torch.zeros((1, 3, th, tw), device=device))
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
                f_up = params[f"{key}_fine"]
                delta = 0.72 * c_up + 0.58 * m_up + 0.42 * f_up
            else:
                delta = params[key]

            if not is_rgb:
                delta_lum = delta.mean(dim=1, keepdim=True)
                delta_chr = (delta - delta_lum).clamp(-0.45, 0.45)
                delta_wb = (delta[:, 0:1] - delta[:, 2:3]).clamp(-0.65, 0.65)
                v_wb = torch.tensor([1.0, 0.35, -0.85], device=device).view(1, 3, 1, 1)
                if strip_style == "tan_fabric":
                    c_tan = torch.tensor([0.24, 0.08, -0.06], device=device).view(1, 3, 1, 1)
                    s_tan = torch.tensor([0.92, 0.84, 0.76], device=device).view(1, 3, 1, 1)
                    raw = c_tan + delta_lum * s_tan + delta_wb * v_wb * 0.38 + delta_chr * 0.30
                elif strip_style in ["clear_silicone", "clear"]:
                    raw = init + delta_lum * 0.90 + delta_wb * v_wb * 0.25 + delta_chr * 0.25
                else:
                    if key == "strip":
                        c_blk = torch.tensor([-0.24, -0.24, -0.22], device=device).view(1, 3, 1, 1)
                        raw = c_blk + delta_lum * 0.96 + delta_wb * v_wb * 0.30 + delta_chr * 0.25
                    else:
                        c_dot = torch.tensor([0.16, 0.04, -0.06], device=device).view(1, 3, 1, 1)
                        s_dot = torch.tensor([0.92, 0.86, 0.78], device=device).view(1, 3, 1, 1)
                        raw = c_dot + delta_lum * s_dot + delta_wb * v_wb * 0.36 + delta_chr * 0.30
            else:
                raw = init + delta

            raw = torch.clamp(raw, -1.0, 1.0)
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

    # --- Optimalizácia ---
    best_score = float("inf")
    best_params = None

    for it in range(num_iter):
        opt.zero_grad()

        if len(geo) == 1:
            selected_indices = [0]
        else:
            aux_perm = [i + 1 for i in torch.randperm(len(geo) - 1)[:max(1, batch_photos - 1)].tolist()]
            selected_indices = [0] + aux_perm

        patches = _current_patches()

        # Pre každú fotku v mini-batchi pripravíme EoT a clean vstupy
        step_batches = []
        for idx in selected_indices:
            base, full_mask, full_relief, boxes, _ = geo[idx]
            gain_full = _illumination_gain(base)
            region_specs = _build_region_specs(patches, boxes)
            canvas, _ = _build_canvas(base, region_specs, gain_full, full_relief, PATCH_BLEND_FACTOR)

            model_input, base_warps = _eot_forward(canvas, base, full_mask, eot, K=eot_k)
            model_input = model_input.contiguous()
            clean_input = ((1.0 - full_mask) * base + full_mask * canvas).clamp(-1.0, 1.0).contiguous()

            if input_diversity:
                div = _div_params(*model_input.shape[-2:])
                model_input = _apply_div(model_input, div)
                if base_warps is not None:
                    base_warps = _apply_div(base_warps, div)
                clean_input = _apply_div(clean_input, div)

            step_batches.append((idx, model_input, base_warps, clean_input))

        # Regularizátory
        nps = sum(_nps_loss(p) for p in patches.values()) / max(1, len(patches))
        tv = sum(_tv_loss(p) for p in patches.values()) / max(1, len(patches))
        nat = sum(_naturalness_loss(p) for p in patches.values()) / max(1, len(patches))
        reg = sum(p.pow(2).mean() for p in params.values()) / max(1, len(params))
        reg_total = nps_weight * nps + tv_weight * tv + naturalness_weight * nat + reg_weight * reg

        # PAD-AWARE penalizácia
        pad_term = None
        if pad_model is not None and pad_baseline is not None:
            try:
                pad_score = pad_model(step_batches[0][1]).mean()
                pad_term = pad_weight * F.relu(pad_baseline - 0.03 - pad_score)
            except Exception:
                pad_term = None

        def _model_loss(name, m):
            total_m = torch.zeros((), device=device)
            sim_primary = None
            sim_aux_list = []
            n_aux = max(1, len(step_batches) - 1)
            for idx, model_input, base_warps, clean_input in step_batches:
                pose_w = 1.0 if len(step_batches) == 1 else (0.68 if idx == 0 else (0.32 / n_aux))
                if name in targets and targets[name] is not None:
                    s_i = 0.5 * F.cosine_similarity(targets[name], m(model_input)).mean() \
                          + 0.5 * F.cosine_similarity(targets[name], m(clean_input)).mean()
                    total_m = total_m - pose_w * s_i
                    val_s = float(-s_i.detach().item())
                    if idx == 0:
                        sim_primary = val_s
                    else:
                        sim_aux_list.append(val_s)
                    continue

                if feature_weight > 0.0:
                    adv_emb, adv_raw = _forward_raw(m, model_input)
                    clean_emb, clean_raw = _forward_raw(m, clean_input)
                else:
                    adv_emb = m(model_input)
                    clean_emb = m(clean_input)
                    adv_raw = clean_raw = None

                orig_emb, orig_raw = _orig_emb_and_raw(idx, name)
                if base_warps is not None:
                    with torch.no_grad():
                        if feature_weight > 0.0:
                            base_emb, base_raw = _forward_raw(m, base_warps)
                            base_emb, base_raw = base_emb.detach(), base_raw.detach()
                        else:
                            base_emb = m(base_warps).detach()
                            base_raw = None
                    sim_eot = F.cosine_similarity(base_emb, adv_emb).mean()
                else:
                    base_raw = orig_raw
                    sim_eot = F.cosine_similarity(orig_emb, adv_emb).mean()

                sim_clean = F.cosine_similarity(orig_emb, clean_emb).mean()
                sim = 0.35 * sim_eot + 0.65 * sim_clean
                if gallery_centroids and name in gallery_centroids and gallery_centroids[name] is not None:
                    g_emb = gallery_centroids[name]
                    sim_g_eot = F.cosine_similarity(g_emb, adv_emb).mean()
                    sim_g_clean = F.cosine_similarity(g_emb, clean_emb).mean()
                    sim_g = 0.35 * sim_g_eot + 0.65 * sim_g_clean
                    sim = 0.45 * sim + 0.55 * sim_g + 1.8 * F.relu(sim_g_clean - 0.42)
                    val_s = float(sim_g_clean.detach().item())
                else:
                    val_s = float(sim_clean.detach().item())
                if idx == 0:
                    sim_primary = val_s
                else:
                    sim_aux_list.append(val_s)
                l_i = sim + 1.6 * F.relu(sim_clean - 0.44) + 0.8 * F.relu(sim - margin[name])

                if feature_weight > 0.0 and adv_raw is not None:
                    feat_eot = F.cosine_similarity(base_raw, adv_raw, dim=1).mean()
                    feat_clean = F.cosine_similarity(orig_raw, clean_raw, dim=1).mean()
                    feat_sim = 0.55 * feat_eot + 0.45 * feat_clean
                    l_i = l_i + feature_weight * feat_sim

                total_m = total_m + pose_w * l_i
            if sim_primary is None:
                sim_out = sum(sim_aux_list) / max(1, len(sim_aux_list))
            elif sim_aux_list:
                sim_out = 0.82 * sim_primary + 0.18 * (sum(sim_aux_list) / len(sim_aux_list))
            else:
                sim_out = sim_primary
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
                flat = _flat_grads(g, smooth=True)
                rms = flat.pow(2).mean().sqrt().clamp_min(1e-8)
                hardness_w = (max(0.10, m_sim) ** 2.2) * (2.2 if m_sim > 0.40 else 0.4)
                arch_w = active_priors.get(name, 1.0)
                model_grads.append((flat / rms) * (hardness_w * arch_w))
            if cur_sims:
                over_50 = sum(max(0.0, s - 0.492) * 8.0 for s in cur_sims)
                sc = over_50 + (sum(cur_sims) / len(cur_sims))
                if sc < best_score:
                    best_score = sc
                    best_params = {k: v.detach().clone() for k, v in params.items()}
            direction = torch.stack(model_grads).sum(0)
            dir_rms = direction.pow(2).mean().sqrt().clamp_min(1e-8)
            direction = direction / dir_rms

            aux = reg_total if pad_term is None else reg_total + pad_term
            g_aux = _flat_grads(torch.autograd.grad(aux, param_list, allow_unused=True), smooth=False, norm_per_tensor=True)
            direction = direction + 0.05 * g_aux

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
            loss = torch.zeros((), device=device)
            cur_sims = []
            for name, m in models.items():
                m_loss, m_sim = _model_loss(name, m)
                cur_sims.append(m_sim)
                w_m = active_priors.get(name, 1.0) * (1.0 + 3.5 * max(0.0, m_sim - 0.42))
                loss = loss + w_m * m_loss
            if cur_sims:
                over_50 = sum(max(0.0, s - 0.492) * 8.0 for s in cur_sims)
                sc = over_50 + (sum(cur_sims) / len(cur_sims))
                if sc < best_score:
                    best_score = sc
                    best_params = {k: v.detach().clone() for k, v in params.items()}

            total = loss + reg_total
            if pad_term is not None:
                total = total + pad_term

            total.backward()
            with torch.no_grad():
                for p in param_list:
                    if p.grad is not None and p.shape[-1] >= 32:
                        p.grad.copy_(_ti_smooth_grad(p.grad, ti_kernel))
            opt.step()

        with torch.no_grad():
            for p in params.values():
                p.clamp_(-1.5, 1.5)

    # --- Finálna kompozícia (bez EOT) ---
    with torch.no_grad():
        if best_params is not None:
            for k, v in best_params.items():
                params[k].copy_(v)
        base, full_mask, full_relief, boxes, lm = geo[0]
        patches = _current_patches()
        gain_full = _illumination_gain(base)
        region_specs = _build_region_specs(patches, boxes)
        canvas, comps = _build_canvas(base, region_specs, gain_full, full_relief, PATCH_BLEND_FACTOR)
        final_adv = ((1.0 - full_mask) * base + full_mask * canvas).clamp(-1.0, 1.0)

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
