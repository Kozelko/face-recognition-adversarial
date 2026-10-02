"""
Implementácia adversariálneho útoku pomocou konštelácie nálepiek (Constellation Attack).
Pozície nálepiek sa dynamicky viažu na 3D landmarky tváre (MediaPipe / MTCNN).
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- MediaPipe 3D FaceLandmarker Singleton ---
_mp_landmarker = None

def get_mediapipe_landmarker():
    global _mp_landmarker
    if _mp_landmarker is None:
        try:
            import os
            import mediapipe as mp
            from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
            from mediapipe.tasks.python import BaseOptions
            
            model_path = os.path.join(os.path.dirname(__file__), "..", "models", "checkpoints", "face_landmarker.task")
            model_path = os.path.abspath(model_path)
            if os.path.exists(model_path):
                options = FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=model_path),
                    num_faces=1
                )
                _mp_landmarker = FaceLandmarker.create_from_options(options)
        except Exception:
            _mp_landmarker = None
    return _mp_landmarker


def extract_landmarks(image_tensor: torch.Tensor, mtcnn=None, return_mesh: bool = False):
    """
    Extrahuje landmarky z tváre (112x112).
    1. Primárne skúša Google MediaPipe 3D Face Mesh (478 bodov so skutočnou hĺbkou Z).
    2. Ako zálohu používa MTCNN (5 bodov) a kanonické súradnice.
    """
    canonical_5 = [(36.0, 48.0), (76.0, 48.0), (56.0, 68.0), (36.0, 88.0), (76.0, 88.0)]
    mesh_info = None

    # Denormalizácia z [-1, 1] do [0, 255] uint8 pre detektory
    img_np = ((image_tensor[0].detach().permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

    # Detekcia cez MediaPipe FaceMesh
    landmarker = get_mediapipe_landmarker()
    if landmarker is not None:
        try:
            import mediapipe as mp
            import cv2
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
            res = landmarker.detect(mp_img)
            if res.face_landmarks and len(res.face_landmarks) > 0:
                raw = res.face_landmarks[0]
                pts_3d = [(float(p.x * 112.0), float(p.y * 112.0), float(p.z * 112.0)) for p in raw]

                lm5 = [
                    (pts_3d[33][0], pts_3d[33][1]),
                    (pts_3d[263][0], pts_3d[263][1]),
                    (pts_3d[1][0], pts_3d[1][1]),
                    (pts_3d[61][0], pts_3d[61][1]),
                    (pts_3d[291][0], pts_3d[291][1]),
                ]

                dx = pts_3d[263][0] - pts_3d[33][0]
                dy = pts_3d[263][1] - pts_3d[33][1]
                dz = pts_3d[263][2] - pts_3d[33][2]
                roll = math.atan2(dy, dx)
                yaw = math.atan2(dz, dx)

                mesh_info = {
                    "points_3d": pts_3d,
                    "roll": roll,
                    "yaw": yaw,
                    "left_cheek_3d": pts_3d[205],
                    "right_cheek_3d": pts_3d[425],
                    "nose_bridge_3d": pts_3d[6],
                }
                return (lm5, mesh_info) if return_mesh else lm5
        except Exception:
            pass

    # Fallback na MTCNN
    if mtcnn is not None:
        try:
            from PIL import Image
            pil_img = Image.fromarray(img_np)
            boxes, probs, landmarks = mtcnn.detect(pil_img, landmarks=True)
            if landmarks is not None and len(landmarks) > 0 and landmarks[0] is not None:
                lm = landmarks[0]
                if len(lm) == 5 and all(0 <= pt[0] <= 112 and 0 <= pt[1] <= 112 for pt in lm):
                    lm5 = [(float(pt[0]), float(pt[1])) for pt in lm]
                    return (lm5, None) if return_mesh else lm5
        except Exception:
            pass

    return (canonical_5, None) if return_mesh else canonical_5


def get_mesh_point_geometry(pts: list[tuple[float, float, float]], idx: int, roll: float) -> tuple[float, float]:
    """
    Vypočíta 3D normálu povrchu, perspektívne skrátenie (foreshortening) a uhol natočenia elipsy
    pre daný landmark z MediaPipe 3D siete (478 bodov).
    """
    neighbors = {
        205: (50, 101, 118, 187),   # Ľavé predné líce (anterior malar)
        50: (123, 205, 118, 147),   # Ľavá lícna kosť (upper malar)
        101: (205, 6, 118, 207),    # Ľavé vnútorné líce (medial cheek)
        118: (50, 6, 108, 205),     # Ľavé podočie (infraorbital)
        187: (147, 205, 118, 214),  # Ľavé dolné líce (lower cheek)
        425: (330, 280, 347, 411),  # Pravé predné líce (anterior malar)
        280: (425, 352, 347, 376),  # Pravá lícna kosť (upper malar)
        330: (6, 425, 347, 427),    # Pravé vnútorné líce (medial cheek)
        347: (6, 280, 337, 425),    # Pravé podočie (infraorbital)
        411: (280, 376, 347, 434),  # Pravé dolné líce (lower cheek)
        195: (45, 275, 197, 5),     # Stredný mostík nosa (mid bridge)
        197: (45, 275, 6, 195),     # Horný mostík nosa (upper bridge)
        5: (45, 275, 195, 4),       # Dolný chrbát nosa / supratip
        98: (48, 195, 98, 97),      # Ľavé krídlo nosa (alar flare)
        327: (195, 278, 327, 326),  # Pravé krídlo nosa (alar flare)
        175: (148, 377, 199, 152),  # Brada (mental protuberance)
        199: (148, 377, 175, 18),   # Brada horná
        10: (67, 297, 10, 151),     # Čelo stred
        9: (67, 297, 10, 8),        # Čelo dolné
    }
    if idx not in neighbors or idx >= len(pts):
        return 0.90, roll
    l, r, u, d = neighbors[idx]
    vl = np.array(pts[l])
    vr = np.array(pts[r])
    vu = np.array(pts[u])
    vd = np.array(pts[d])
    vh = vr - vl
    vv = vd - vu
    n = np.cross(vh, vv)
    norm = np.linalg.norm(n)
    if norm < 1e-6:
        return 0.90, roll
    n = n / norm
    cos_slant = float(np.clip(abs(n[2]), 0.58, 1.0))
    slant_angle = math.atan2(float(n[1]), float(n[0]))
    tilt = roll + (slant_angle if abs(n[0]) > 0.15 else 0.0)
    return cos_slant, tilt


def get_surface_geometry_at_coords(
    pts_3d: list[tuple[float, float, float]] | np.ndarray,
    qx: float,
    qy: float,
    base_roll: float = 0.0,
) -> tuple[float, float, float]:
    """
    Vypočíta 3D zakrivenie, perspektívne skrátenie (foreshortening), smer sklonu a relatívnu hĺbku
    pre ľubovoľnú zadanú súradnicu (qx, qy) na tvári na základe 3D MediaPipe siete (478 bodov).

    Vráti:
    (cos_slant, tilt_angle, depth_z)
    - cos_slant: pomer sploštenia elipsy (1.0 = kolmý pohľad, 0.45 = bočný sklon)
    - tilt_angle: uhol natočenia elipsy v radiánoch (zarovnanie s kontúrou tváre)
    - depth_z: lokálna hĺbka pre tieňovanie
    """
    if pts_3d is None or len(pts_3d) < 10:
        return 0.88, base_roll, 0.0

    pts = np.asarray(pts_3d, dtype=np.float32)
    dists_sq = (pts[:, 0] - qx) ** 2 + (pts[:, 1] - qy) ** 2
    nearest_indices = np.argsort(dists_sq)[:6]

    center_pt = pts[nearest_indices[0]]
    diffs = pts[nearest_indices] - center_pt
    A = diffs[:, :2]
    b = diffs[:, 2]
    try:
        slope, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        n = np.array([-slope[0], -slope[1], 1.0], dtype=np.float32)
        norm = np.linalg.norm(n)
        if norm > 1e-6:
            n = n / norm
        else:
            n = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    except Exception:
        n = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    if n[2] < 0:
        n = -n

    cos_slant = float(np.clip(n[2], 0.45, 1.0))
    gradient_dir = math.atan2(float(n[1]), float(n[0]))
    tilt_angle = gradient_dir + math.pi * 0.5

    return cos_slant, tilt_angle, float(center_pt[2])


def get_anatomical_patch_specs(
    landmarks: list[tuple[float, float]],
    pattern: str = "trio_cheeks_nose",
    base_radius: float = 4.2,
    mesh_info: dict | None = None,
    seed: int | None = None,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    spot_offsets: dict | list | None = None,
) -> list[dict]:
    """
    Vypočíta anatomicky presné umiestnenie, 3D sklon, eliptické polomery a rotáciu nálepiek/pieh.
    - Presné anatomické zóny (skutočné líca, žiadne umiestňovanie k ušiam!).
    - 3D perspektívne skrátenie (foreshortening) a uhol natočenia elipsy podľa sklonu pokožky.
    - Dynamická variabilita a rôzne polohy pri každom spustení (pomocou seedu/rng).
    - Podpora manuálneho posunu (offset_x, offset_y) pre interaktívne testovanie.
    """
    import random
    rng = random.Random(seed if seed is not None else random.randint(1, 1000000))

    lx, ly = landmarks[0]  # ľavé oko
    rx, ry = landmarks[1]  # pravé oko
    nx, ny = landmarks[2]  # nos
    mlx, mly = landmarks[3]  # ľavé ústa
    mrx, mry = landmarks[4]  # pravé ústa

    dx = rx - lx
    dy = ry - ly
    roll = math.atan2(dy, dx)
    eye_dist = math.sqrt(dx**2 + dy**2)
    scale = max(0.75, min(1.25, eye_dist / 42.0))
    r = base_radius * scale

    has_mesh = mesh_info is not None and "points_3d" in mesh_info and len(mesh_info["points_3d"]) > 400
    pts = mesh_info["points_3d"] if has_mesh else None

    specs = []

    # Pomocná funkcia na vytvorenie špecifikácie jedného bodu z indexu siete alebo záložných súradníc
    def make_spot(name: str, mesh_idx: int | None, fallback_xy: tuple[float, float], fallback_cos: float, fallback_tilt: float, shape: str = "circle", r_scale: float = 1.0):
        if has_mesh and mesh_idx is not None and mesh_idx < len(pts):
            p = pts[mesh_idx]
            cos_slant, tilt = get_mesh_point_geometry(pts, mesh_idx, roll)
            jx = rng.uniform(-1.0, 1.0)
            jy = rng.uniform(-1.0, 1.0)
            cx = float(p[0]) + jx
            cy = float(p[1]) + jy
        else:
            fb_x, fb_y = fallback_xy
            cos_slant = fallback_cos
            tilt = fallback_tilt
            jx = rng.uniform(-1.2, 1.2)
            jy = rng.uniform(-1.2, 1.2)
            cx = fb_x + jx
            cy = fb_y + jy

        curr_r = r * r_scale
        specs.append({
            "name": name,
            "center": (min(106.0, max(6.0, cy)), min(106.0, max(6.0, cx))),
            "rx": curr_r * cos_slant,
            "ry": curr_r,
            "angle": tilt,
            "shape": shape,
        })

    # Záložné anatomické súradnice (skutočné líca pod očami, NIE na okraji tváre pri uchu!)
    fb_left_cheek = (lx + (nx - lx) * 0.14, ly + (mly - ly) * 0.42)
    fb_right_cheek = (rx - (rx - nx) * 0.14, ry + (mry - ry) * 0.42)
    fb_nose_bridge = ((lx + rx) * 0.5, (ly + ry) * 0.5 + (ny - (ly + ry) * 0.5) * 0.45)
    fb_chin = ((mlx + mrx) * 0.5, max(mly, mry) + (ny - (ly + ry) * 0.5) * 0.42)
    fb_forehead = ((lx + rx) * 0.5, (ly + ry) * 0.5 - (ny - (ly + ry) * 0.5) * 0.65)

    # REŽIM 1: Adversariálne pehy (Adversarial Freckles - 8 mikro-bodov v motýlej zóne)
    if pattern in ["adversarial_freckles", "freckles"]:
        r_freckle = max(1.4, min(r * 0.42, 2.2))
        freckle_offsets = [
            (-0.12, -0.06), (0.12, -0.05), (-0.35, 0.12), (0.35, 0.14),
            (-0.55, 0.28), (0.55, 0.26), (-0.22, 0.35), (0.22, 0.36)
        ]
        for idx, (ox, oy) in enumerate(freckle_offsets):
            jx = rng.uniform(-1.8, 1.8)
            jy = rng.uniform(-1.5, 1.5)
            fcx = nx + ox * (rx - lx) * 0.48 + jx
            fcy = (ly + ry) * 0.5 + 8.0 + oy * (ny - ly) * 0.9 + jy
            fr = r_freckle * rng.uniform(0.85, 1.15)
            f_angle = roll + rng.uniform(-0.2, 0.2)
            specs.append({
                "name": f"freckle_{idx}",
                "center": (min(108.0, max(4.0, fcy)), min(108.0, max(4.0, fcx))),
                "rx": fr * 0.90,
                "ry": fr,
                "angle": f_angle,
                "shape": "circle",
            })

    # REŽIM 2: Asymetrické náhodné akné nálepky (3 body na rôznych častiach tváre)
    elif pattern in ["asymmetric_random", "asymmetric"]:
        zone_candidates = [
            ("asym_left_cheek", rng.choice([205, 50, 101, 118]), fb_left_cheek, 0.82, roll + 0.35),
            ("asym_right_cheek", rng.choice([425, 280, 330, 347]), fb_right_cheek, 0.82, roll - 0.35),
            ("asym_nose", rng.choice([195, 197, 5, 98, 327]), fb_nose_bridge, 0.92, roll),
            ("asym_chin", rng.choice([175, 199]), fb_chin, 0.90, roll),
            ("asym_forehead", rng.choice([10, 9]), fb_forehead, 0.92, roll),
        ]
        chosen_zones = rng.sample(zone_candidates, 3)
        for name, m_idx, fb_xy, fb_cos, fb_tilt in chosen_zones:
            make_spot(name, m_idx, fb_xy, fb_cos, fb_tilt)

    # REŽIM 3: Kozmetické akné hviezdičky (Starface - 3 žlté hviezdičky)
    elif pattern in ["star_patches", "starface", "stars"]:
        left_idx = rng.choice([205, 101, 118])
        right_idx = rng.choice([425, 330, 347])
        nose_idx = rng.choice([195, 197, 5])
        make_spot("star_left_cheek", left_idx, fb_left_cheek, 0.82, roll + 0.30, shape="star", r_scale=1.18)
        make_spot("star_right_cheek", right_idx, fb_right_cheek, 0.82, roll - 0.30, shape="star", r_scale=1.18)
        make_spot("star_nose", nose_idx, fb_nose_bridge, 0.92, roll, shape="star", r_scale=1.18)

    # REŽIM 4: 5 bodov: Cluster (Líca + Nos + Brada) [Prah účinnosti]
    elif pattern in ["cluster_5", "cluster", "5_cluster"]:
        left_idx1 = rng.choice([205, 50])
        left_idx2 = rng.choice([101, 118])
        right_idx = rng.choice([425, 280, 330])
        nose_idx = rng.choice([195, 197, 5])
        chin_idx = rng.choice([175, 199])
        make_spot("cluster_left_1", left_idx1, fb_left_cheek, 0.82, roll + 0.35)
        make_spot("cluster_left_2", left_idx2, (lx + (nx - lx) * 0.28, ly + (mly - ly) * 0.52), 0.85, roll + 0.25)
        make_spot("cluster_right", right_idx, fb_right_cheek, 0.82, roll - 0.35)
        make_spot("cluster_nose", nose_idx, fb_nose_bridge, 0.92, roll)
        make_spot("cluster_chin", chin_idx, fb_chin, 0.90, roll)

    # REŽIM 5: Štandardné anatomické vzory (3 body: Nos + Obe líca ako predvolené)
    else:
        if pattern in ["trio_cheeks_nose", "quad_full", "cheeks_only"]:
            left_idx = rng.choice([205, 50, 101, 118])
            right_idx = rng.choice([425, 280, 330, 347])
            make_spot("left_cheek", left_idx, fb_left_cheek, 0.82, roll + 0.35)
            make_spot("right_cheek", right_idx, fb_right_cheek, 0.82, roll - 0.35)

        if pattern in ["trio_cheeks_nose", "quad_full", "t_zone"]:
            nose_idx = rng.choice([195, 197, 5])
            make_spot("nose_bridge", nose_idx, fb_nose_bridge, 0.92, roll)

        if pattern in ["quad_full", "t_zone"]:
            chin_idx = rng.choice([175, 199])
            make_spot("chin", chin_idx, fb_chin, 0.90, roll)

        if pattern == "t_zone":
            forehead_idx = rng.choice([10, 9])
            make_spot("forehead", forehead_idx, fb_forehead, 0.92, roll)

    # Aplikácia manuálneho posunu (individuálny pre každý bod alebo globálny offset_x, offset_y)
    for idx, s in enumerate(specs):
        cy, cx = s["center"]
        ox, oy = offset_x, offset_y
        if spot_offsets is not None:
            if isinstance(spot_offsets, dict):
                for k in [s.get("name"), idx]:
                    if k in spot_offsets:
                        sox, soy = spot_offsets[k]
                        ox += sox
                        oy += soy
                        break
            elif isinstance(spot_offsets, (list, tuple)) and idx < len(spot_offsets):
                sox, soy = spot_offsets[idx]
                ox += sox
                oy += soy

        new_cx = float(np.clip(cx + ox, 4.0, 108.0))
        new_cy = float(np.clip(cy + oy, 4.0, 108.0))
        s["center"] = (new_cy, new_cx)

        # Prepočítaj 3D zakrivenie, perspektívu a natočenie na novej posunutej pozícii
        if has_mesh:
            c_slant, c_tilt, c_z = get_surface_geometry_at_coords(pts, new_cx, new_cy, roll)
            base_r = s["ry"]
            s["rx"] = base_r * c_slant
            s["angle"] = c_tilt
            s["depth_z"] = c_z

    return specs


def create_elliptical_patch_mask(
    img_size: int,
    center: tuple[float, float],
    radius_x: float,
    radius_y: float,
    angle_rad: float,
    feather: float = 0.8,
    shape: str = "circle",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vytvorí 2D antialiased masku, 3D hydrokoloidný dome reliéf,
    kontaktný mikro-tieň na pokožku (drop shadow) a specular odlesk na zaoblenej hrane.
    Podporuje eliptické nálepky aj hviezdičky (Starface).
    """
    cy, cx = center
    y = torch.arange(img_size, dtype=torch.float32).view(-1, 1) - cy
    x = torch.arange(img_size, dtype=torch.float32).view(1, -1) - cx

    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # 2D rotácia podľa náklonu tváre/líca
    xr = x * cos_a + y * sin_a
    yr = -x * sin_a + y * cos_a

    raw_dist = torch.sqrt((xr / max(1.0, radius_x)) ** 2 + (yr / max(1.0, radius_y)) ** 2)

    if shape == "star":
        # 5-cípa hviezdička (Starface kozmetická nálepka)
        theta = torch.atan2(yr, xr)
        star_mod = 1.0 + 0.32 * torch.cos(5.0 * (theta - angle_rad))
        dist = raw_dist / torch.clamp(star_mod, 0.65, 1.35)
    else:
        dist = raw_dist

    # Mäkký beveled okraj (anti-aliasing)
    min_r = min(radius_x, radius_y)
    slope = min_r / max(0.5, feather)
    mask = torch.clamp((1.0 - dist) * slope + 0.5, 0.0, 1.0).view(1, 1, img_size, img_size)

    # 3D dome reliéf nálepky (stred hrubší ~0.35 mm, okraje hladko splývajú)
    dome = (1.0 - 0.20 * torch.clamp(dist, 0.0, 1.2)**2).view(1, 1, img_size, img_size)

    # 3D Kontaktný mikro-tieň na pokožku (Drop shadow posunutý o ~0.9 px nadol)
    shadow_yr = yr - 0.90
    shadow_xr = xr - 0.35
    shadow_raw = torch.sqrt((shadow_xr / max(1.0, radius_x)) ** 2 + (shadow_yr / max(1.0, radius_y)) ** 2)
    if shape == "star":
        st_theta = torch.atan2(shadow_yr, shadow_xr)
        st_mod = 1.0 + 0.32 * torch.cos(5.0 * (st_theta - angle_rad))
        shadow_dist = shadow_raw / torch.clamp(st_mod, 0.65, 1.35)
    else:
        shadow_dist = shadow_raw

    shadow_mask = torch.clamp((1.0 - shadow_dist) * (slope * 0.85) + 0.5, 0.0, 1.0).view(1, 1, img_size, img_size)
    drop_shadow = (torch.clamp(shadow_mask - mask, 0.0, 1.0) * 0.35).view(1, 1, img_size, img_size)

    # Specular rim (matný odlesk na hornej klenutej hrane nálepky)
    spec_yr = yr + radius_y * 0.45
    spec_dist = torch.sqrt((xr / max(1.0, radius_x * 0.85)) ** 2 + (spec_yr / max(1.0, radius_y * 0.45)) ** 2)
    specular_rim = (torch.clamp(1.0 - spec_dist, 0.0, 1.0) ** 2 * mask * 0.14).view(1, 1, img_size, img_size)

    return mask, dome, drop_shadow, specular_rim


def total_variation_loss(patch: torch.Tensor) -> torch.Tensor:
    """
    TV strata pre hladkú tlač textúry bez vysokofrekvenčného šumu.
    """
    diff_h = patch[:, :, :, 1:, :] - patch[:, :, :, :-1, :]
    diff_w = patch[:, :, :, :, 1:] - patch[:, :, :, :, :-1]
    tv = torch.sum(diff_h ** 2) + torch.sum(diff_w ** 2)
    return tv / patch.numel()


def apply_realistic_blending(
    image: torch.Tensor,
    canvas_patches: torch.Tensor,
    mask: torch.Tensor,
    dome: torch.Tensor | None = None,
    drop_shadow: torch.Tensor | None = None,
    specular_rim: torch.Tensor | None = None,
    alpha_opacity: float = 0.88,
) -> torch.Tensor:
    """
    Realistické prelínanie s pokožkou:
    - 3D kontaktný tieň pod nálepkou vrhnutý na pokožku
    - Hydrokoloidná nálepka je z 88 % nepriehľadná, z 12 % preberá prirodzený odtieň a štruktúru pokožky.
    - Lokálna adaptácia svetla, 3D dome reliéf a matný odlesk pre hĺbkový fyzický vzhľad.
    """
    # 1. Podklad s kontaktným mikro-tieňom
    if drop_shadow is not None:
        base_skin = image * (1.0 - drop_shadow)
    else:
        base_skin = image

    skin_lum = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
    skin_light_mult = torch.clamp(skin_lum * 0.35 + 1.0, 0.65, 1.35)

    if dome is not None:
        lit_patches = torch.clamp(canvas_patches * skin_light_mult * dome, -1.0, 1.0)
    else:
        lit_patches = torch.clamp(canvas_patches * skin_light_mult, -1.0, 1.0)

    if specular_rim is not None:
        lit_patches = torch.clamp(lit_patches + specular_rim, -1.0, 1.0)

    blended_patch = alpha_opacity * lit_patches + (1.0 - alpha_opacity) * base_skin
    output = (1.0 - mask) * base_skin + mask * blended_patch
    return output.contiguous()


def apply_constellation_eot(
    image: torch.Tensor,
    canvas_patches: torch.Tensor,
    mask: torch.Tensor,
    dome: torch.Tensor | None = None,
    drop_shadow: torch.Tensor | None = None,
    specular_rim: torch.Tensor | None = None,
    alpha_opacity: float = 0.88,
    max_angle_deg: float = 5.0,
    max_trans: float = 0.025,
    color_jitter: float = 0.05,
    noise_std: float = 0.008,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    EoT transformácia (Expectation over Transformation):
    Simuluje fyzické nepresnosti (pohyb hlavy pred kamerou, posun, svetelné výkyvy).
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
    trans_patch = F.grid_sample(canvas_patches, grid, mode="bilinear", padding_mode="border", align_corners=False)
    trans_mask = F.grid_sample(mask, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    trans_mask = torch.clamp(trans_mask, 0.0, 1.0)

    trans_dome = None
    if dome is not None:
        trans_dome = F.grid_sample(dome, grid, mode="bilinear", padding_mode="border", align_corners=False)

    trans_shadow = None
    if drop_shadow is not None:
        trans_shadow = F.grid_sample(drop_shadow, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    trans_spec = None
    if specular_rim is not None:
        trans_spec = F.grid_sample(specular_rim, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    # Svetelný jitter
    brightness = (torch.rand(B, 1, 1, 1, device=device) * 2 - 1) * color_jitter + 1.0
    trans_patch = torch.clamp(trans_patch * brightness, -1.0, 1.0)

    # Realistické zlúčenie
    patched_image = apply_realistic_blending(
        image=image,
        canvas_patches=trans_patch,
        mask=trans_mask,
        dome=trans_dome,
        drop_shadow=trans_shadow,
        specular_rim=trans_spec,
        alpha_opacity=alpha_opacity,
    )

    if noise_std > 0:
        noise = torch.randn_like(patched_image) * noise_std
        patched_image = torch.clamp(patched_image + noise, -1.0, 1.0).contiguous()

    return patched_image, trans_mask


def constellation_patch_attack(
    model: nn.Module,
    image: torch.Tensor,
    pattern: str = "trio_cheeks_nose",
    radius: float = 4.0,
    base_radius: float | None = None,
    num_iter: int = 70,
    lr: float = 0.07,
    eot: bool = True,
    tv_weight: float = 0.02,
    target_emb: torch.Tensor | None = None,
    mtcnn = None,
    landmarks = None,
    seed: int | None = None,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    spot_offsets: dict | list | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Generuje realistický útok pomocou konštelácie mikro-nálepiek (Pimple Patches).
    - Polohy a tvary sa dynamicky prispôsobujú landmarkom tváre a sklonu pokožky.
    - Oklúzia tváre je minimálna (~2 až 5 %).
    """
    if base_radius is not None:
        radius = float(base_radius)
    device = image.device

    # 1. Detekcia landmarkov z tváre (ak neboli dodané vopred)
    mesh_info = None
    if landmarks is None:
        landmarks, mesh_info = extract_landmarks(image, mtcnn=mtcnn, return_mesh=True)
    elif isinstance(landmarks, tuple):
        landmarks, mesh_info = landmarks
    specs = get_anatomical_patch_specs(
        landmarks,
        pattern=pattern,
        base_radius=radius,
        mesh_info=mesh_info,
        seed=seed,
        offset_x=offset_x,
        offset_y=offset_y,
        spot_offsets=spot_offsets,
    )
    num_spots = len(specs)

    full_mask = torch.zeros((1, 1, 112, 112), device=device)
    full_dome = torch.ones((1, 1, 112, 112), device=device)
    full_shadow = torch.zeros((1, 1, 112, 112), device=device)
    full_specular = torch.zeros((1, 1, 112, 112), device=device)
    for s in specs:
        m, d, sh, sp = create_elliptical_patch_mask(
            img_size=112,
            center=s["center"],
            radius_x=s["rx"],
            radius_y=s["ry"],
            angle_rad=s["angle"],
            feather=0.75,
            shape=s.get("shape", "circle"),
        )
        m, d, sh, sp = m.to(device), d.to(device), sh.to(device), sp.to(device)
        full_mask = torch.maximum(full_mask, m)
        full_dome = torch.where(m > 0.1, d, full_dome)
        full_shadow = torch.maximum(full_shadow, sh)
        full_specular = torch.maximum(full_specular, sp)

    with torch.no_grad():
        orig_emb = model(image).detach()

    patch_dim = int(math.ceil(max(s["ry"], s["rx"]) * 2.5)) + 4
    patch_dim = max(12, patch_dim)

    if "freckle" in pattern:
        base_color = torch.tensor([0.42, 0.26, 0.15], device=device).view(1, 1, 3, 1, 1)
        opacity = 0.82
    elif "star" in pattern:
        base_color = torch.tensor([0.72, 0.65, 0.16], device=device).view(1, 1, 3, 1, 1)
        opacity = 0.92
    else:
        base_color = torch.tensor([0.68, 0.45, 0.22], device=device).view(1, 1, 3, 1, 1)
        opacity = 0.88

    delta = nn.Parameter(torch.zeros((1, num_spots, 1, patch_dim, patch_dim), device=device))
    optimizer = torch.optim.Adam([delta], lr=lr)

    for _ in range(num_iter):
        optimizer.zero_grad()

        patch_param = torch.clamp(base_color + delta, -1.0, 1.0)
        canvas = image.clone()
        half = patch_dim // 2

        for idx, s in enumerate(specs):
            cy_int = int(round(s["center"][0]))
            cx_int = int(round(s["center"][1]))

            y1 = max(0, cy_int - half)
            y2 = min(112, cy_int + half)
            x1 = max(0, cx_int - half)
            x2 = min(112, cx_int + half)

            py1 = half - (cy_int - y1)
            py2 = py1 + (y2 - y1)
            px1 = half - (cx_int - x1)
            px2 = px1 + (x2 - x1)

            canvas[:, :, y1:y2, x1:x2] = patch_param[0, idx, :, py1:py2, px1:px2]

        if eot:
            adv_img, _ = apply_constellation_eot(
                image=image,
                canvas_patches=canvas,
                mask=full_mask,
                dome=full_dome,
                drop_shadow=full_shadow,
                specular_rim=full_specular,
                alpha_opacity=opacity,
                max_angle_deg=4.0,
                max_trans=0.02,
                color_jitter=0.05,
                noise_std=0.006,
            )
        else:
            adv_img = apply_realistic_blending(
                image=image,
                canvas_patches=canvas,
                mask=full_mask,
                dome=full_dome,
                drop_shadow=full_shadow,
                specular_rim=full_specular,
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

        # TV strata pre hladký organický povrch
        diff_h = delta[:, :, :, 1:, :] - delta[:, :, :, :-1, :]
        diff_w = delta[:, :, :, :, 1:] - delta[:, :, :, :, :-1]
        tv = (torch.sum(diff_h ** 2) + torch.sum(diff_w ** 2)) / delta.numel()

        reg_loss = torch.mean(delta ** 2)
        total_loss = loss + tv_weight * tv + 0.04 * reg_loss

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            delta.clamp_(-0.45, 0.45)

    with torch.no_grad():
        final_patch = torch.clamp(base_color + delta, -1.0, 1.0)
        final_canvas = image.clone()
        half = patch_dim // 2

        for idx, s in enumerate(specs):
            cy_int = int(round(s["center"][0]))
            cx_int = int(round(s["center"][1]))

            y1 = max(0, cy_int - half)
            y2 = min(112, cy_int + half)
            x1 = max(0, cx_int - half)
            x2 = min(112, cx_int + half)

            py1 = half - (cy_int - y1)
            py2 = py1 + (y2 - y1)
            px1 = half - (cx_int - x1)
            px2 = px1 + (x2 - x1)

            final_canvas[:, :, y1:y2, x1:x2] = final_patch[0, idx, :, py1:py2, px1:px2]

        final_adv_img = apply_realistic_blending(
            image=image,
            canvas_patches=final_canvas,
            mask=full_mask,
            dome=full_dome,
            drop_shadow=full_shadow,
            specular_rim=full_specular,
            alpha_opacity=opacity,
        )
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

        # Spojenie vzorov nálepiek vedľa seba pre náhľad tlače
        patches_cpu = final_patch[0].detach().cpu()
        micro_crop = torch.cat([patches_cpu[i] for i in range(num_spots)], dim=2)

    # Plocha tváre v percentách
    area_pct = float((full_mask.sum() / (112 * 112) * 100.0).item())

    info = {
        "pattern": pattern,
        "spots_count": num_spots,
        "radius_px": radius,
        "face_occlusion_pct": round(area_pct, 2),
        "orig_similarity": orig_sim,
        "final_similarity": final_sim,
        "success": success,
        "iterations": num_iter,
        "landmarks_used": landmarks,
    }

    return final_adv_img.detach(), micro_crop, info
