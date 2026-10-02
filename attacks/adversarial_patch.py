"""
Implementácia adversariálneho leukoplastu (Adversarial Patch) pre tvárovú biometriu.
Maska rešpektuje odhadnuté 3D klenutie nosa a sklon hlavy.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from attacks.constellation_patch import extract_landmarks


def get_patch_geometry(
    position: str = "nose_bridge",
    img_size: int = 112,
    patch_size: tuple[int, int] | None = None,
    landmarks: list[tuple[float, float]] | None = None,
    mesh_info: dict | None = None,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
):
    """
    Vráti masku zaobleného, 3D pokrčeného leukoplastu a jeho ohraničujúci obdĺžnik.
    Anatomicky prispôsobený nosu bez zasahovania do očí.
    """
    if isinstance(landmarks, tuple):
        landmarks, mesh_info = landmarks

    if landmarks is not None and len(landmarks) >= 5:
        lx, ly = landmarks[0]
        rx, ry = landmarks[1]
        nx, ny = landmarks[2]
        dx = rx - lx
        dy = ry - ly
        roll = math.atan2(dy, dx)
        eye_dist = math.sqrt(dx**2 + dy**2)
        scale = max(0.7, min(1.25, eye_dist / 42.0))
    else:
        roll = 0.0
        scale = 1.0
        lx, ly = 36.0, 48.0
        rx, ry = 76.0, 48.0
        nx, ny = 56.0, 68.0

    if position == "nose_bridge":
        # Anatomická veľkosť: cca 17x7 px (nie 26 px!), presne sedí na nosovom mostíku
        base_h = patch_size[0] if patch_size else 7
        base_w = patch_size[1] if patch_size else 17
        h = int(round(base_h * scale))
        w = int(round(base_w * scale))

        if mesh_info and "points_3d" in mesh_info and len(mesh_info["points_3d"]) > 197:
            pts = mesh_info["points_3d"]
            p195 = pts[195] if len(pts) > 195 else pts[197]
            cx = float(p195[0])
            cy = float(p195[1])
        else:
            cx = (lx + rx) * 0.5
            cy = (ly + ry) * 0.5 + (ny - (ly + ry) * 0.5) * 0.44
        angle = roll
    elif position == "cheek":
        # Náplasť na líci: 10 x 20 px, bezpečne pod okom, mimo nosa
        base_h = patch_size[0] if patch_size else 9
        base_w = patch_size[1] if patch_size else 18
        h = int(round(base_h * scale))
        w = int(round(base_w * scale))
        dx_l = max(5.0, nx - lx)
        cx = lx + (nx - lx) * 0.14
        cy = ly + (ny - ly) * 0.55
        angle = roll - math.radians(12)
    elif position == "forehead":
        # Náplasť na čele: 8 x 22 px
        base_h = patch_size[0] if patch_size else 8
        base_w = patch_size[1] if patch_size else 22
        h = int(round(base_h * scale))
        w = int(round(base_w * scale))
        cx = (lx + rx) * 0.5
        cy = (ly + ry) * 0.5 - (ny - (ly + ry) * 0.5) * 0.70
        angle = roll
    else:
        h, w = 7, 17
        cx, cy = 56.0, 56.0
        angle = 0.0

    # Aplikácia manuálneho posunu
    cx = float(np.clip(cx + offset_x, 8.0, 104.0))
    cy = float(np.clip(cy + offset_y, 8.0, 104.0))

    # SDF pre zaoblený obdĺžnik
    y = torch.arange(img_size, dtype=torch.float32).view(-1, 1) - cy
    x = torch.arange(img_size, dtype=torch.float32).view(1, -1) - cx

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    xr = x * cos_a + y * sin_a
    yr = -x * sin_a + y * cos_a

    corner_radius = 2.5
    half_w = max(4.0, w / 2.0 - corner_radius)
    half_h = max(2.5, h / 2.0 - corner_radius)

    dx_sdf = torch.abs(xr) - half_w
    dy_sdf = torch.abs(yr) - half_h
    outside = torch.sqrt(torch.clamp(dx_sdf, min=0.0)**2 + torch.clamp(dy_sdf, min=0.0)**2)
    inside = torch.clamp(torch.max(dx_sdf, dy_sdf), max=0.0)
    sdf = outside + inside - corner_radius

    mask = torch.clamp(0.5 - sdf / 0.85, 0.0, 1.0).view(1, 1, img_size, img_size)

    # 3D reliéf a tieňovanie
    nx_coord = torch.clamp(torch.abs(xr) / max(1.0, w / 2.0), 0.0, 1.0)
    shading = (1.0 - 0.22 * (nx_coord ** 2)).view(1, 1, img_size, img_size)
    weave = (0.025 * torch.sin(xr * 2.2) * torch.cos(yr * 2.2)).view(1, 1, img_size, img_size)
    crease = (0.035 * torch.sin(xr * 0.85)).view(1, 1, img_size, img_size)
    surface_relief = shading + weave + crease

    cy_int = int(round(cy))
    cx_int = int(round(cx))
    pad_h = h // 2 + 3
    pad_w = w // 2 + 3
    y1 = max(0, cy_int - pad_h)
    y2 = min(img_size, cy_int + pad_h)
    x1 = max(0, cx_int - pad_w)
    x2 = min(img_size, cx_int + pad_w)

    return mask, surface_relief, (y1, y2, x1, x2)


def total_variation_loss(patch: torch.Tensor) -> torch.Tensor:
    """
    TV loss pre hladkosť textúry.
    """
    diff_h = patch[:, :, 1:, :] - patch[:, :, :-1, :]
    diff_w = patch[:, :, :, 1:] - patch[:, :, :, :-1]
    tv = torch.sum(diff_h ** 2) + torch.sum(diff_w ** 2)
    return tv / patch.numel()


def apply_bandage_blending(
    image: torch.Tensor,
    canvas_patch: torch.Tensor,
    mask: torch.Tensor,
    surface_relief: torch.Tensor,
    alpha_opacity: float = 0.90,
) -> torch.Tensor:
    """
    Zlúči textúru leukoplastu s tvárou:
    - 3D cylindrické tieňovanie nosa, mikro-pokrčenie a štruktúra tkaniny
    - Adaptácia na lokálny jas okolitej pokožky
    - Hladké zaoblené okraje
    """
    skin_lum = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
    light_mult = torch.clamp(skin_lum * 0.35 + 1.0, 0.6, 1.4)

    # Aplikácia 3D reliéfu a svetla
    shaped_patch = torch.clamp(canvas_patch * light_mult * surface_relief, -1.0, 1.0)

    # Prelínanie s pokožkou
    blended = alpha_opacity * shaped_patch + (1.0 - alpha_opacity) * image
    output = (1.0 - mask) * image + mask * blended
    return output.contiguous()


def apply_eot_transform(
    image: torch.Tensor,
    canvas_patch: torch.Tensor,
    mask: torch.Tensor,
    surface_relief: torch.Tensor,
    max_angle_deg: float = 5.0,
    max_trans: float = 0.025,
    color_jitter: float = 0.05,
    noise_std: float = 0.008,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    EoT náhodné transformácie simulujúce nepresnosť a pohyb pred kamerou.
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
    trans_patch = F.grid_sample(canvas_patch, grid, mode="bilinear", padding_mode="border", align_corners=False)
    trans_mask = F.grid_sample(mask, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    trans_relief = F.grid_sample(surface_relief, grid, mode="bilinear", padding_mode="border", align_corners=False)
    trans_mask = torch.clamp(trans_mask, 0.0, 1.0)

    # Svetelná variácia
    brightness = (torch.rand(B, 1, 1, 1, device=device) * 2 - 1) * color_jitter + 1.0
    trans_patch = torch.clamp(trans_patch * brightness, -1.0, 1.0)

    patched_image = apply_bandage_blending(image, trans_patch, trans_mask, trans_relief)

    if noise_std > 0:
        noise = torch.randn_like(patched_image) * noise_std
        patched_image = torch.clamp(patched_image + noise, -1.0, 1.0).contiguous()

    return patched_image, trans_mask


def adversarial_patch_attack(
    model: nn.Module,
    image: torch.Tensor,
    position: str = "nose_bridge",
    patch_size: tuple[int, int] | None = None,
    num_iter: int = 70,
    lr: float = 0.07,
    eot: bool = True,
    tv_weight: float = 0.02,
    target_emb: torch.Tensor | None = None,
    mtcnn = None,
    landmarks = None,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Vygeneruje a optimalizuje anatomicky presný, pokrčený leukoplast.
    """
    device = image.device
    mesh_info = None
    if landmarks is None:
        landmarks, mesh_info = extract_landmarks(image, mtcnn=mtcnn, return_mesh=True)
    elif isinstance(landmarks, tuple):
        landmarks, mesh_info = landmarks

    mask, surface_relief, (y1, y2, x1, x2) = get_patch_geometry(
        position=position,
        img_size=112,
        patch_size=patch_size,
        landmarks=landmarks,
        mesh_info=mesh_info,
        offset_x=offset_x,
        offset_y=offset_y,
    )
    mask = mask.to(device)
    surface_relief = surface_relief.to(device)

    with torch.no_grad():
        orig_emb = model(image).detach()

    h_patch = y2 - y1
    w_patch = x2 - x1
    base_color = torch.tensor([0.62, 0.38, 0.18], device=device).view(1, 3, 1, 1)
    delta_lum = nn.Parameter(torch.zeros((1, 1, h_patch, w_patch), device=device))

    optimizer = torch.optim.Adam([delta_lum], lr=lr)

    for _ in range(num_iter):
        optimizer.zero_grad()

        patch_param = torch.clamp(base_color + delta_lum, -1.0, 1.0)
        canvas_patch = image.clone()
        canvas_patch[:, :, y1:y2, x1:x2] = patch_param

        if eot:
            adv_img, _ = apply_eot_transform(
                image=image,
                canvas_patch=canvas_patch,
                mask=mask,
                surface_relief=surface_relief,
                max_angle_deg=4.0,
                max_trans=0.02,
                color_jitter=0.05,
                noise_std=0.006,
            )
        else:
            adv_img = apply_bandage_blending(image, canvas_patch, mask, surface_relief)

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
        tv_loss = (torch.sum(diff_h ** 2) + torch.sum(diff_w ** 2)) / delta_lum.numel()

        reg_loss = torch.mean(delta_lum ** 2)
        total_loss = loss + tv_weight * tv_loss + 0.04 * reg_loss

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            delta_lum.clamp_(-0.45, 0.45)

    with torch.no_grad():
        final_patch = torch.clamp(base_color + delta_lum, -1.0, 1.0)
        final_canvas = image.clone()
        final_canvas[:, :, y1:y2, x1:x2] = final_patch
        final_adv_img = apply_bandage_blending(image, final_canvas, mask, surface_relief)
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

        patch_crop = final_patch.squeeze(0).detach().cpu()

    area_pct = float((mask.sum() / (112 * 112) * 100.0).item())

    info = {
        "position": position,
        "mode": "impersonation" if target_emb is not None else "dodging",
        "face_occlusion_pct": round(area_pct, 2),
        "orig_similarity": orig_sim,
        "final_similarity": final_sim,
        "success": success,
        "iterations": num_iter,
        "patch_shape": (h_patch, w_patch),
    }

    return final_adv_img.detach(), patch_crop, info
