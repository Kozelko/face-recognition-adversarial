#!/home/kozel/miniconda/envs/cnn-benchmark/bin/python
import os
os.environ["ORT_LOG_SEVERITY_LEVEL"] = "3"
import sys
_CONDA_PY = "/home/kozel/miniconda/envs/cnn-benchmark/bin/python"
if "cnn-benchmark" not in sys.executable and os.path.exists(_CONDA_PY):
    os.execv(_CONDA_PY, [_CONDA_PY] + sys.argv)
import math
import json
import base64

# Automatické nastavenie pracovného adresára na koreň projektu
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import ctypes
try:
    ctypes.CDLL('/home/kozel/miniconda/envs/cnn-benchmark/lib/libpng16.so.16', mode=ctypes.RTLD_GLOBAL)
except Exception:
    pass

import gradio as gr
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image, ImageOps
from datetime import datetime
import shutil

from evaluate_attacks import load_benchmark_cnn, denormalize
from train_finetune import run_finetuning
from models.wrappers import FaceNetWrapper, ArcFaceWrapper, AdaFaceWrapper
from attacks.fgsm import fgsm_attack_untargeted
from attacks.pgd import pgd_attack_untargeted
from attacks.bim import bim_attack_untargeted
from attacks.mifgsm import mifgsm_attack_untargeted
from attacks.cw import cw_l2_attack_untargeted
from attacks.adversarial_patch import adversarial_patch_attack, get_patch_geometry
from attacks.constellation_patch import constellation_patch_attack, create_elliptical_patch_mask, get_anatomical_patch_specs, extract_landmarks, get_surface_geometry_at_coords
from attacks.hybrid_patch import hybrid_patch_attack, get_hybrid_geometry
from attacks.ensemble_hybrid_patch import ensemble_hybrid_patch_attack
from attacks.nasal_strip_patch import nasal_strip_attack, generate_printable_nasal_strip_sheet, create_nasal_strip_mask, compute_nasal_strip_placement
from models.pad_minifasnet import load_pad_model

# Globálne nastavenia
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Jediný zdroj pravdy pre rozsah jemného posunu prvkov (px v priestore 112x112).
# Používa ho JS drag overlay (clamp úchytky) aj Gradio slidery, aby sa úchytka
# nikdy nedostala mimo rozsah, ktorý vie slider reprezentovať.
OFFSET_LIMIT_PX = 25.0
DATASET_DIR = "data/custom_dataset"

# Detekcia tvárí (MTCNN + Haar Cascade fallback)
from facenet_pytorch import MTCNN
from PIL import ImageOps
mtcnn_primary = MTCNN(image_size=112, margin=20, thresholds=[0.5, 0.6, 0.6], device=DEVICE, post_process=False, select_largest=True)
mtcnn_relaxed = MTCNN(image_size=112, margin=20, thresholds=[0.3, 0.4, 0.4], device=DEVICE, post_process=False, select_largest=True)
mtcnn = mtcnn_primary  # kompatibilita s attack modulmi

# Haar Cascade záloha pre webkamery
cascade_paths = [
    "/home/kozel/miniconda/envs/cnn-benchmark/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    "/home/kozel/miniconda/envs/cnn-benchmark/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml",
]
haar_cascade = None
for cp in cascade_paths:
    if os.path.exists(cp):
        c = cv2.CascadeClassifier(cp)
        if not c.empty():
            haar_cascade = c
            break

# Modely
models_dict = {}
pad_detector = None

def load_finetuned_cnn(device):
    from models.wrappers import BenchmarkCNNWrapper
    checkpoint_path = "models/checkpoints/benchmark_cnn_finetuned.pth"
    if not os.path.exists(checkpoint_path):
        return None
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        num_classes = checkpoint.get("num_classes", 10575)
        model = BenchmarkCNNWrapper(num_classes=num_classes, checkpoint_path=checkpoint_path, device=device)
        return model
    except Exception as e:
        print(f"Chyba pri načítaní Finetuned BenchmarkCNN: {e}")
        return None

def init_models():
    global models_dict, pad_detector
    if not models_dict:
        print("Načítavam modely do pamäte...")
        models_dict["FaceNet"] = FaceNetWrapper(device=DEVICE)
        try:
            models_dict["FaceNet (CASIA)"] = FaceNetWrapper(pretrained="casia-webface", device=DEVICE)
        except Exception as e:
            print(f"FaceNet (CASIA) load failed: {e}")
        models_dict["ArcFace"] = ArcFaceWrapper(device=DEVICE)
        models_dict["AdaFace"] = AdaFaceWrapper(device=DEVICE)
        bcnn = load_benchmark_cnn(DEVICE)
        if bcnn is not None:
            models_dict["BenchmarkCNN"] = bcnn
        
        bcnn_finetuned = load_finetuned_cnn(DEVICE)
        if bcnn_finetuned is not None:
            models_dict["BenchmarkCNN (Finetuned)"] = bcnn_finetuned
            print("Finetuned BenchmarkCNN úspešne načítaný.")

        pad_detector = load_pad_model(device=DEVICE)
        if pad_detector is not None:
            print("PAD detektor (MiniFASNetV2) úspešne načítaný.")
            
        print("Modely úspešne načítané.")

# Inicializuj modely pri štarte aplikácie
init_models()

# Identifikácia osôb a registrácia galérie
gallery_cache = {}

def get_model_gallery(model, model_name):
    if model_name in gallery_cache:
        return gallery_cache[model_name]
    
    gallery = {}
    import glob
    from PIL import Image
    from torchvision import transforms
    
    trans = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    if os.path.exists(DATASET_DIR):
        for person in sorted(os.listdir(DATASET_DIR)):
            p_dir = os.path.join(DATASET_DIR, person)
            if not os.path.isdir(p_dir):
                continue
            valid_exts = (".jpg", ".jpeg", ".png")
            img_paths = [os.path.join(p_dir, f) for f in sorted(os.listdir(p_dir)) if f.lower().endswith(valid_exts)]
            if not img_paths:
                continue
            with torch.no_grad():
                embs = []
                for p in img_paths:
                    im = Image.open(p).convert("RGB")
                    t = trans(im).unsqueeze(0).to(DEVICE)
                    embs.append(model(t))
                if embs:
                    cent = F.normalize(torch.cat(embs, dim=0).mean(dim=0, keepdim=True), p=2, dim=1)
                    gallery[person] = cent
    gallery_cache[model_name] = gallery
    return gallery

def get_dataset_classes():
    if not os.path.exists(DATASET_DIR):
        return ["brandajsky_peter"]
    dirs = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    classes = sorted(dirs)
    if "brandajsky_peter" in classes:
        classes.remove("brandajsky_peter")
        classes.insert(0, "brandajsky_peter")
    return classes if classes else ["brandajsky_peter"]

def format_display_name(person_id):
    if "brandajsky" in person_id or "peter" in person_id:
        return "Peter Brandajský"
    return person_id.replace("_", " ").title()

def format_short_name(person_id):
    if "brandajsky" in person_id or "peter" in person_id:
        return "Peter"
    parts = person_id.split("_")
    return parts[-1].capitalize() if parts else person_id.capitalize()

def predict_person_identity(model, model_name, img_tensor, is_attack=False, sim_1to1=None):
    """Textový výpis rozpoznania: najbližší profil v databáze + (ak existuje) klasifikátor."""
    clf = None
    if hasattr(model, "predict_identity"):
        clf = model.predict_identity(img_tensor)

    gallery = get_model_gallery(model, model_name)
    best_person = None
    best_sim = -1.0
    with torch.no_grad():
        emb = model(img_tensor)
        for person, cent in gallery.items():
            sim = F.cosine_similarity(emb, cent).item()
            if sim > best_sim:
                best_sim = sim
                best_person = format_display_name(person)

    lines = ["Po útoku" if is_attack else "Pôvodná fotka"]
    if is_attack and sim_1to1 is not None:
        lines.append(f"Podobnosť s pôvodnou fotkou: {sim_1to1:.4f} (prah 0.50)")

    if best_person is not None and best_sim >= 0.50:
        lines.append(f"Najbližší profil v databáze: {best_person}, podobnosť {best_sim:.4f} -> rozpoznaný")
    elif best_person is not None:
        lines.append(f"Najbližší profil v databáze: {best_person}, podobnosť {best_sim:.4f} -> nerozpoznaný (pod 0.50)")
    else:
        lines.append("Databáza profilov je prázdna.")

    if clf:
        label = format_display_name(clf["raw_label"])
        lines.append(f"Klasifikátor: {label}, istota {clf['confidence'] * 100.0:.1f} %")
        sorted_probs = sorted(clf["probabilities"].items(), key=lambda x: x[1], reverse=True)
        lines.append("Pravdepodobnosti: " + " | ".join(
            f"{format_short_name(k)} {v * 100.0:.1f} %" for k, v in sorted_probs))
    return "\n".join(lines)

def to_pil_image(image):
    if image is None:
        return None
    from PIL import Image
    import numpy as np
    
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, str):
        if os.path.exists(image):
            return Image.open(image).convert("RGB")
        return None
    if isinstance(image, dict):
        val = image.get("path") or image.get("composite") or image.get("image") or image.get("background")
        if val is not None and val is not image:
            return to_pil_image(val)
        return None
    if isinstance(image, np.ndarray):
        arr = image.copy()
        if np.issubdtype(arr.dtype, np.floating):
            if arr.max() <= 1.01:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            
        if len(arr.shape) == 2:
            return Image.fromarray(arr).convert("RGB")
        elif len(arr.shape) == 3:
            if arr.shape[2] == 4:
                return Image.fromarray(arr).convert("RGB")
            elif arr.shape[2] == 3:
                return Image.fromarray(arr, mode="RGB")
    return None

def crop_face_from_image(pil_img):
    """
    Robustná viacúrovňová detekcia tváre (112×112 px):
    1. Už orezaná (112×112)
    2. MTCNN primárna
    3. MTCNN uvoľnená (zvýšená citlivosť)
    4. MTCNN s vyrovnaným kontrastom (pre tmavé webkamery)
    5. OpenCV Haar Cascade (klasický spoľahlivý detektor tvárí)
    6. Inteligentný stredový výrez (Center Crop pre webkamery)
    """
    if pil_img is None:
        return None, "Žiadny obrázok"

    w, h = pil_img.size
    if w == 112 and h == 112:
        arr = np.array(pil_img).astype(np.float32)
        t = torch.from_numpy(arr).permute(2, 0, 1).to(DEVICE)
        return t, "Pôvodný 112×112 px výrez"

    # Pre webkamera zábery s vysokým rozlíšením (napr. 1080p) normalizujeme mierku
    proc_img = pil_img
    max_dim = max(w, h)
    if max_dim > 720:
        scale = 720.0 / max_dim
        new_w, new_h = max(112, int(w * scale)), max(112, int(h * scale))
        proc_img = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # MTCNN detekcia
    try:
        t = mtcnn_primary(proc_img)
        if t is not None:
            return t, "MTCNN (vysoká presnosť)"
    except Exception:
        pass

    try:
        t = mtcnn_relaxed(proc_img)
        if t is not None:
            return t, "MTCNN (zvýšená citlivosť)"
    except Exception:
        pass

    try:
        enhanced = ImageOps.autocontrast(proc_img)
        t = mtcnn_relaxed(enhanced)
        if t is not None:
            return t, "MTCNN (vyrovnaný kontrast)"
    except Exception:
        pass

    # Haar Cascade fallback
    if haar_cascade is not None:
        try:
            cv_img = np.array(proc_img)
            if len(cv_img.shape) == 3 and cv_img.shape[2] == 3:
                gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
            else:
                gray = cv_img
            gray = cv2.equalizeHist(gray)
            faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
            if len(faces) > 0:
                faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
                x, y, fw, fh = faces[0]
                mx, my = int(fw * 0.20), int(fh * 0.20)
                x1 = max(0, x - mx)
                y1 = max(0, y - my)
                x2 = min(proc_img.width, x + fw + mx)
                y2 = min(proc_img.height, y + fh + my)
                cropped = proc_img.crop((x1, y1, x2, y2)).resize((112, 112), Image.Resampling.BILINEAR)
                arr = np.array(cropped).astype(np.float32)
                t = torch.from_numpy(arr).permute(2, 0, 1).to(DEVICE)
                return t, "OpenCV Haar Cascade (webkamera)"
        except Exception:
            pass

    # Stredový výrez fallback
    try:
        min_dim = min(w, h)
        cx, cy = w // 2, h // 2
        box_size = int(min_dim * 0.75)
        x1 = max(0, cx - box_size // 2)
        y1 = max(0, cy - box_size // 2)
        x2 = min(w, x1 + box_size)
        y2 = min(h, y1 + box_size)
        cropped = pil_img.crop((x1, y1, x2, y2)).resize((112, 112), Image.Resampling.BILINEAR)
        arr = np.array(cropped).astype(np.float32)
        t = torch.from_numpy(arr).permute(2, 0, 1).to(DEVICE)
        return t, "Stredový výrez (fallback)"
    except Exception:
        pass

    return None, "Detekcia zlyhala"

def get_face_tensor_from_image(image):
    pil_img = to_pil_image(image)
    if pil_img is None:
        return None, None, "Žiadny obrázok. Nahraj fotku alebo sa odfoť cez webkameru."
        
    # Kontrola čierneho/prázdneho záberu
    arr_check = np.array(pil_img)
    if arr_check.mean() < 3.0:
        return None, None, "Webkamera poslala čierny záber. Skontroluj krytku kamery a skús znova."
        
    face_t_255, method = crop_face_from_image(pil_img)
    if face_t_255 is None:
        return None, None, "Nepodarilo sa nájsť tvár na obrázku."
        
    t = (face_t_255.float() / 255.0 - 0.5) / 0.5
    if len(t.shape) == 3:
        t = t.unsqueeze(0)
    return t.to(DEVICE), method, None

def run_identify(image, model_name, ensemble_mode=""):
    if image is None:
        return None, None, None, "Nahrajte alebo odfoťte fotku.", "", ""
    
    img_tensor, method, err_msg = get_face_tensor_from_image(image)
    if img_tensor is None:
        return None, None, None, err_msg, "", ""

    model = models_dict.get(model_name)
    if model is None:
        return None, None, None, f"Model {model_name} nie je dostupný.", "", ""
        
    clean_id = predict_person_identity(model, model_name, img_tensor, is_attack=False)
    face_np = (denormalize(img_tensor).squeeze().permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    
    status = "Test rozpoznania identity (živá kamera / fotka)\n"
    status += f"Cieľový model: {model_name} | Detekcia: {method} (112x112 px)"
    holdout_model_name = None
    if "Samostatný" in str(ensemble_mode):
        status += " | Režim: Samostatný model (White-Box)"
    elif "Plný" in str(ensemble_mode):
        status += " | Režim: Plný Ensemble 4 modelov"
    elif "Ensemble" in str(ensemble_mode) or "Black-Box" in str(ensemble_mode):
        status += f" | Režim: Black-Box cieľ ({model_name})"
        holdout_model_name = model_name
    if pad_detector is not None:
        with torch.no_grad():
            p_live = float(pad_detector(img_tensor).mean().item())
        pad_verdict = "ŽIVÁ TVÁR (LIVE)" if p_live >= 0.5 else "PODVRH (SPOOF)"
        status += f"\nPAD Liveness (MiniFASNetV2): P(live) = {p_live * 100.0:.1f} % -> {pad_verdict}"

    # Prehľad všetkých modelov pre rýchle overenie vytlačeného útoku pred kamerou
    cross_lines = ["Prehľad všetkých modelov (pre test vytlačených nálepiek pred kamerou):"]
    for m_name, m_obj in models_dict.items():
        if m_obj is None:
            continue
        id_lines = predict_person_identity(m_obj, m_name, img_tensor, is_attack=False).splitlines()
        summary_line = id_lines[1] if len(id_lines) > 1 else id_lines[0]
        tag = " 🎯 [BLACK-BOX]" if m_name == holdout_model_name else ""
        cross_lines.append(f"• {m_name}{tag}: {summary_line}")
    adv_id = "\n".join(cross_lines)
    
    return face_np, face_np, None, clean_id, adv_id, status

# Interaktívny náhľad umiestnenia nálepiek & Drag and Drop plátno
_live_preview_cache = {
    "cached_img_id": None,
    "face_np": None,
    "landmarks": None,
    "mesh_info": None,
}

def extract_base_patch_coords(landmarks, mesh, nose_is_dot=False, strip_scale=1.0):
    if landmarks is None:
        return None
    lx, ly = landmarks[0]
    rx, ry = landmarks[1]
    nx, ny = landmarks[2]

    if nose_is_dot:
        # Nosová bodka konštelácie (Pimple Patches) – drží sa stredu nosového mostíka
        if mesh and "points_3d" in mesh and len(mesh["points_3d"]) > 197:
            pts = mesh["points_3d"]
            p195 = pts[195]
            p5 = pts[5] if len(pts) > 5 else pts[195]
            strip_x = float(p195[0] * 0.50 + p5[0] * 0.50) * 2.0
            strip_y = float(p195[1] * 0.50 + p5[1] * 0.50) * 2.0
        else:
            strip_x = ((lx + rx) * 0.5) * 2.0
            strip_y = ((ly + ry) * 0.5 + (ny - (ly + ry) * 0.5) * 0.54) * 2.0
    else:
        # Športová páska na nos – presne rovnaké ukotvenie, aké použije samotný útok
        place = compute_nasal_strip_placement(landmarks, mesh, width_scale=float(strip_scale) if strip_scale else 1.0)
        strip_x = float(place["cx"]) * 2.0
        strip_y = float(place["cy"]) * 2.0

    if mesh and "points_3d" in mesh and len(mesh["points_3d"]) > 205:
        left_x = float(mesh["points_3d"][205][0]) * 2.0
        left_y = (float(mesh["points_3d"][205][1]) - 1.5) * 2.0
    else:
        left_x = (lx - (nx - lx) * 0.12) * 2.0
        left_y = (ly + (ny - ly) * 0.45) * 2.0

    if mesh and "points_3d" in mesh and len(mesh["points_3d"]) > 425:
        right_x = float(mesh["points_3d"][425][0]) * 2.0
        right_y = (float(mesh["points_3d"][425][1]) - 1.5) * 2.0
    else:
        right_x = (rx + (rx - nx) * 0.12) * 2.0
        right_y = (ry + (ny - ry) * 0.45) * 2.0

    if mesh and "points_3d" in mesh and len(mesh["points_3d"]) > 175:
        chin_x = float(mesh["points_3d"][175][0]) * 2.0
        chin_y = (float(mesh["points_3d"][175][1]) - 3.0) * 2.0
    else:
        chin_x = ((lx + rx) * 0.5) * 2.0
        chin_y = (ny + (ny - ly) * 1.32) * 2.0

    strip_label = "[N] Nos" if nose_is_dot else "[N] Páska na nos"
    strip_type = "dot" if nose_is_dot else "strip"
    return {
        "strip": {"x": round(strip_x, 1), "y": round(strip_y, 1), "label": strip_label, "short": "N▲", "color": "#00e5ff", "type": strip_type},
        "left": {"x": round(left_x, 1), "y": round(left_y, 1), "label": "[L] Ľavé líce (vľavo)", "short": "L▲", "color": "#d500f9", "type": "dot"},
        "right": {"x": round(right_x, 1), "y": round(right_y, 1), "label": "[P] Pravé líce (vpravo)", "short": "P▲", "color": "#00e676", "type": "dot"},
        "chin": {"x": round(chin_x, 1), "y": round(chin_y, 1), "label": "[B] Brada", "short": "B▲", "color": "#ff9100", "type": "dot"},
    }

def get_active_handles_for_attack(attack_name, constellation_pattern):
    is_nasal = attack_name == "Športová páska na nos (Breathe Right)"
    is_leukoplast = attack_name == "Adversarial Patch (Leukoplast)"
    is_const = attack_name == "Constellation Patch (Pimple Patches)"
    is_hybrid = ("Hybrid Patch" in str(attack_name)) or ("BioMesh-SpecPatch" in str(attack_name))

    pat_str = str(constellation_pattern).lower()

    if is_nasal or is_leukoplast:
        return {"strip": True, "left": False, "right": False, "chin": False}

    if is_hybrid:
        has_chin = "brada" in pat_str or "3 nálepky" in pat_str
        has_right = "1 nálepka" not in pat_str
        return {
            "strip": True,
            "left": True,
            "right": has_right,
            "chin": has_chin,
        }

    if is_const:
        has_nose = any(k in pat_str for k in ["nos", "cluster", "star", "t-zóna", "asymetrické", "pehy", "freckles"]) and "iba líca" not in pat_str
        has_chin = any(k in pat_str for k in ["brada", "cluster", "quad", "full", "t-zóna"])
        has_left = "t-zóna" not in pat_str
        has_right = "t-zóna" not in pat_str and "1 nálepka" not in pat_str
        return {
            "strip": has_nose,
            "left": has_left,
            "right": has_right,
            "chin": has_chin,
        }

    return {"strip": False, "left": False, "right": False, "chin": False}

HEAD_SCRIPTS = """
<style>
#drag_sync_hidden_group {
    position: absolute !important;
    width: 0 !important;
    height: 0 !important;
    opacity: 0 !important;
    pointer-events: none !important;
    overflow: hidden !important;
}
body.webcam-flip-video video {
    transform: scaleX(-1) !important;
}
</style>
<script>
window.__toggleWebcamFlip = function() {
    document.body.classList.toggle('webcam-flip-video');
    return document.body.classList.contains('webcam-flip-video');
};
window.__faceDragState = {
    draggingKey: null,
    hoverKey: null,
    dragOffset: { x: 0, y: 0 },
    handles: {},
    base: {},
    active: {},
    offsets: {}
};

window.__getFaceDragOffsets = function() {
    const state = window.__faceDragState;
    if (!state || !state.base) return null;
    // Rozsah sliderov (jeden zdroj pravdy s backendom cez data-offsets).
    const lim = parseFloat(state.offsets && state.offsets.offset_limit) || 25.0;

    function getOff(k, axis) {
        if (!state.handles || !state.handles[k] || !state.handles[k].active || !state.base[k]) return 0.0;
        const d = (state.handles[k][axis] - state.base[k][axis]) / 2.0;
        const dc = Math.max(-lim, Math.min(lim, d));
        return Math.round(dc * 2.0) / 2.0;
    }

    return {
        strip_ox: getOff('strip', 'x'),
        strip_oy: getOff('strip', 'y'),
        left_ox: getOff('left', 'x'),
        left_oy: getOff('left', 'y'),
        right_ox: getOff('right', 'x'),
        right_oy: getOff('right', 'y'),
        chin_ox: getOff('chin', 'x'),
        chin_oy: getOff('chin', 'y'),
    };
};

window.__resetFaceDragOffsets = function() {
    const state = window.__faceDragState;
    if (!state || !state.base) return;
    for (const k of Object.keys(state.base)) {
        if (state.handles && state.handles[k]) {
            state.handles[k].x = state.base[k].x;
            state.handles[k].y = state.base[k].y;
        }
    }
    if (window.__redrawFaceDragCanvas) {
        window.__redrawFaceDragCanvas();
    }
};

window.initFaceDragOverlay = function(imgEl) {
    if (!imgEl) return;
    const wrapper = imgEl.parentElement;
    if (!wrapper) return;
    const canvas = wrapper.querySelector('#face_drag_canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const tooltip = wrapper.parentElement.querySelector('#drag_tooltip_text');

    let base = {}, active = {}, curOffsets = {};
    try {
        base = JSON.parse(imgEl.getAttribute('data-base') || '{}');
        active = JSON.parse(imgEl.getAttribute('data-active') || '{}');
        curOffsets = JSON.parse(imgEl.getAttribute('data-offsets') || '{}');
    } catch(e) {
        console.error('Error parsing face drag data:', e);
    }

    const state = window.__faceDragState;
    state.base = base;
    state.active = active;
    state.offsets = curOffsets;
    state.handles = {};

    // Úchytka sa nikdy nesmie dostať mimo rozsah Gradio sliderov (±offset_limit px),
    // inak by sa posun ticho stratil pri synchronizácii so serverom.
    const limitCanvasPx = (parseFloat(curOffsets.offset_limit) || 25.0) * 2.0;
    function clampHandle(k, x, y) {
        const b = state.base[k] || { x: x, y: y };
        return {
            x: Math.max(12, Math.min(212, Math.max(b.x - limitCanvasPx, Math.min(b.x + limitCanvasPx, x)))),
            y: Math.max(12, Math.min(212, Math.max(b.y - limitCanvasPx, Math.min(b.y + limitCanvasPx, y))))
        };
    }

    for (const k of Object.keys(base)) {
        const p0 = clampHandle(k, base[k].x + (curOffsets[k + '_ox'] || 0) * 2, base[k].y + (curOffsets[k + '_oy'] || 0) * 2);
        state.handles[k] = {
            x: p0.x,
            y: p0.y,
            label: base[k].label,
            short: base[k].short || k.toUpperCase(),
            color: base[k].color,
            type: base[k].type,
            active: !!active[k]
        };
    }

    function draw() {
        ctx.clearRect(0, 0, 224, 224);
        for (const k of Object.keys(state.handles)) {
            if (!state.handles[k].active) continue;
            const h = state.handles[k];
            const isHover = (state.hoverKey === k) || (state.draggingKey === k);

            ctx.save();
            if (h.type === 'strip') {
                ctx.fillStyle = isHover ? "rgba(0, 229, 255, 0.22)" : "rgba(0, 229, 255, 0.04)";
                ctx.strokeStyle = h.color;
                ctx.lineWidth = isHover ? 2.0 : 1.2;
                if (!isHover) ctx.setLineDash([3, 2]);
                ctx.shadowColor = h.color;
                ctx.shadowBlur = isHover ? 8 : 2;

                const sScale = parseFloat(state.offsets.strip_scale) || 1.15;
                // Presná veľkosť a náklon pásky z backendu (aby obrys sedel 1:1 s vykreslenou páskou)
                const wPx = parseFloat(state.offsets.strip_w_px) || 0;
                const hPx = parseFloat(state.offsets.strip_h_px) || 0;
                const sw = wPx > 0 ? wPx : Math.round(37 * sScale);
                const sh = hPx > 0 ? hPx : Math.round(11 * sScale);
                const arch = Math.round(4.2 * sScale);
                const sAng = (parseFloat(state.offsets.strip_angle_deg) || 0) * Math.PI / 180.0;
                const xL = -sw/2, xR = sw/2;
                const yTop = -sh/2 - arch * 0.35, yBot = sh/2 - arch * 0.35;

                ctx.save();
                ctx.translate(h.x, h.y);
                ctx.rotate(sAng);
                ctx.beginPath();
                ctx.moveTo(xL, yTop + arch);
                ctx.quadraticCurveTo(0, yTop - arch * 0.85, xR, yTop + arch);
                ctx.lineTo(xR, yBot + arch);
                ctx.quadraticCurveTo(0, yBot - arch * 0.85, xL, yBot + arch);
                ctx.closePath();
                ctx.fill();
                ctx.stroke();
                ctx.restore();
                ctx.setLineDash([]);

                ctx.font = "bold 9px monospace";
                ctx.fillStyle = h.color;
                ctx.textAlign = "center";
                ctx.fillText(h.short, h.x, h.y + yTop - arch * 0.5 - 3);
            } else {
                ctx.fillStyle = isHover ? "rgba(213, 0, 249, 0.25)" : "rgba(0, 0, 0, 0.05)";
                ctx.strokeStyle = h.color;
                ctx.lineWidth = isHover ? 2.0 : 1.2;
                if (!isHover) ctx.setLineDash([3, 2]);
                ctx.shadowColor = h.color;
                ctx.shadowBlur = isHover ? 8 : 2;

                const dRad = parseFloat(state.offsets.patch_radius) || 5.5;
                const pr = (dRad * 2.0) + (isHover ? 1.5 : 0.0);
                ctx.beginPath();
                ctx.arc(h.x, h.y, pr, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();
                ctx.setLineDash([]);

                ctx.font = "bold 9px monospace";
                ctx.fillStyle = h.color;
                ctx.textAlign = "center";
                ctx.fillText(h.short, h.x, h.y - pr - 3);
            }
            ctx.restore();
        }
    }
    window.__redrawFaceDragCanvas = draw;

    function getCoords(e) {
        const r = canvas.getBoundingClientRect();
        return {
            x: (e.clientX - r.left) * (224 / r.width),
            y: (e.clientY - r.top) * (224 / r.height)
        };
    }

    function findHandle(p) {
        let best = null, minDist = 22;
        for (const k of Object.keys(state.handles)) {
            if (!state.handles[k].active) continue;
            const d = Math.hypot(p.x - state.handles[k].x, p.y - state.handles[k].y);
            if (d < minDist) {
                minDist = d;
                best = k;
            }
        }
        return best;
    }

    canvas.onpointerdown = (e) => {
        const p = getCoords(e);
        const k = findHandle(p);
        if (k) {
            state.draggingKey = k;
            state.dragOffset.x = state.handles[k].x - p.x;
            state.dragOffset.y = state.handles[k].y - p.y;
            try { canvas.setPointerCapture(e.pointerId); } catch(err) {}
            canvas.style.cursor = "grabbing";
            draw();
        }
    };

    canvas.onpointermove = (e) => {
        const p = getCoords(e);
        if (state.draggingKey) {
            const h = state.handles[state.draggingKey];
            const cp = clampHandle(state.draggingKey, p.x + state.dragOffset.x, p.y + state.dragOffset.y);
            h.x = cp.x;
            h.y = cp.y;
            draw();
            const ox = (Math.round(((h.x - state.base[state.draggingKey].x) / 2) * 2) / 2).toFixed(1);
            const oy = (Math.round(((h.y - state.base[state.draggingKey].y) / 2) * 2) / 2).toFixed(1);
            if (tooltip) {
                tooltip.innerText = `${h.label}: ΔX = ${ox} px, ΔY = ${oy} px`;
                tooltip.style.color = "#38bdf8";
            }
        } else {
            const k = findHandle(p);
            state.hoverKey = k;
            canvas.style.cursor = k ? "grab" : "default";
            draw();
            if (tooltip) {
                if (k) {
                    const ox = (Math.round(((state.handles[k].x - state.base[k].x) / 2) * 2) / 2).toFixed(1);
                    const oy = (Math.round(((state.handles[k].y - state.base[k].y) / 2) * 2) / 2).toFixed(1);
                    tooltip.innerText = `Chyť ${state.handles[k].label} (ΔX=${ox}, ΔY=${oy})`;
                    tooltip.style.color = "#a78bfa";
                } else {
                    tooltip.innerText = "Chyť prvok myšou a posuň po tvári";
                    tooltip.style.color = "#9ca3af";
                }
            }
        }
    };

    function onDragEnd(e) {
        if (state.draggingKey) {
            try { canvas.releasePointerCapture(e.pointerId); } catch(err) {}
            state.draggingKey = null;
            state.hoverKey = null;
            canvas.style.cursor = "default";
            draw();

            if (tooltip) {
                tooltip.innerText = "⏳ Prepočítavam 3D zakrivenie tváre...";
                tooltip.style.color = "#facc15";
            }

            const syncBtn = document.querySelector("#drag_sync_btn button, #drag_sync_btn, button#drag_sync_btn");
            if (syncBtn) {
                syncBtn.click();
            }
        }
    }

    canvas.onpointerup = onDragEnd;
    canvas.onpointercancel = onDragEnd;
    draw();
};

if (typeof window !== 'undefined' && !window.__faceDragObserverSetup) {
    window.__faceDragObserverSetup = true;
    const observer = new MutationObserver(() => {
        const img = document.querySelector('#face_drag_img');
        if (img && window.initFaceDragOverlay) {
            const sig = (img.getAttribute('src') || '').slice(-40) + '|' + (img.getAttribute('data-offsets') || '') + '|' + (img.getAttribute('data-active') || '');
            if (img !== window.__lastFaceImg || sig !== window.__lastFaceSig) {
                window.__lastFaceImg = img;
                window.__lastFaceSig = sig;
                window.initFaceDragOverlay(img);
            }
        }
    });
    observer.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['data-offsets', 'data-active', 'src'] });
}
</script>
"""

def build_drag_canvas_html(preview_img, base_coords, active_handles, offsets_dict, attack_name):
    if preview_img is None or base_coords is None:
        return """
        <div style="width:224px; height:224px; display:flex; flex-direction:column; align-items:center; justify-content:center; background:#111827; border-radius:8px; border:2px dashed #374151; color:#9ca3af; text-align:center; padding:12px; font-size:12px; user-select:none; margin:0 auto;">
          <span>Nahraj fotku alebo sa odfoť pre aktiváciu plátna</span>
        </div>
        """

    _, enc = cv2.imencode(".jpg", cv2.cvtColor(preview_img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    img_b64 = "data:image/jpeg;base64," + base64.b64encode(enc).decode("ascii")

    base_json = json.dumps(base_coords).replace("'", "&#39;")
    active_json = json.dumps(active_handles if active_handles else {}).replace("'", "&#39;")
    offsets_json = json.dumps(offsets_dict if offsets_dict else {}).replace("'", "&#39;")

    return f"""
    <div style="display:flex; flex-direction:column; align-items:center; user-select:none; margin:0 auto;">
      <div style="position:relative; width:224px; height:224px; border-radius:8px; overflow:hidden; border:2px solid #38bdf8; box-shadow:0 4px 10px rgba(0,0,0,0.5); background:#111827;">
        <img id="face_drag_img" src="{img_b64}" 
             style="width:224px; height:224px; display:block; object-fit:cover; pointer-events:none;" 
             onload="(window.initFaceDragOverlay ? window.initFaceDragOverlay(this) : setTimeout(() => window.initFaceDragOverlay && window.initFaceDragOverlay(this), 50))"
             data-base='{base_json}'
             data-active='{active_json}'
             data-offsets='{offsets_json}' />
        <canvas id="face_drag_canvas" width="224" height="224" style="position:absolute; top:0; left:0; width:224px; height:224px; touch-action:none; cursor:default; z-index:10;"></canvas>
      </div>
      <div id="drag_tooltip_text" style="margin-top:6px; font-size:12px; color:#38bdf8; min-height:18px; text-align:center; font-family:monospace; font-weight:500;">
        Chyť prvok myšou a posuň po tvári
      </div>
    </div>
    """

def format_occlusion_badge(tot_pct: float, strip_pct: float | None = None, dots_pct: float | None = None, is_digital: bool = False) -> str:
    if is_digital:
        return """
        <div style="background:#0f172a; border:1px solid #334155; border-radius:8px; padding:8px 12px; text-align:center; min-width:145px; display:flex; flex-direction:column; justify-content:center; box-sizing:border-box; height:100%;">
            <div style="font-size:10px; text-transform:uppercase; color:#64748b; font-weight:600; letter-spacing:0.5px;">Zakrytie tváre</div>
            <div style="font-size:20px; font-weight:700; color:#94a3b8; margin:2px 0;">0.0 %</div>
            <div style="font-size:10px; color:#64748b;">Digitálny šum</div>
        </div>
        """
    if tot_pct <= 3.2:
        badge_color = "#34d399"
        tag = "Nenápadná náplasť"
    elif tot_pct <= 5.0:
        badge_color = "#38bdf8"
        tag = "Športový kompromis"
    else:
        badge_color = "#fbbf24"
        tag = "Robustný Black-Box"

    details = []
    if strip_pct is not None and strip_pct > 0.05:
        details.append(f"Páska: {strip_pct:.1f}%")
    if dots_pct is not None and dots_pct > 0.05:
        details.append(f"Nálepky: {dots_pct:.1f}%")
    sub_text = " | ".join(details) if details else tag

    # Odhad voči čistej pokožke tváre (face hull):
    # Samotná tvár bez pozadia, vlasov a slúchadiel tvorí cca 42-45 % z 112x112 tenzora (~5200 px).
    skin_est_pct = min(100.0, tot_pct * 2.22)

    return f"""
    <div style="background:#0f172a; border:1.5px solid {badge_color}; border-radius:8px; padding:6px 12px; text-align:center; box-shadow:0 2px 8px rgba(0,0,0,0.3); min-width:165px; display:flex; flex-direction:column; justify-content:center; box-sizing:border-box; height:100%;">
        <div style="font-size:10px; text-transform:uppercase; color:#94a3b8; font-weight:600; letter-spacing:0.5px;">Aktuálne zakrytie tváre</div>
        <div style="font-size:20px; font-weight:800; color:{badge_color}; line-height:1.2; margin:2px 0;">{tot_pct:.1f} % <span style="font-size:11px; font-weight:500; color:#94a3b8;">vstupu</span></div>
        <div style="font-size:11px; color:#38bdf8; font-weight:600; margin-bottom:2px;">≈ {skin_est_pct:.1f} % pleti tváre</div>
        <div style="font-size:10px; color:#cbd5e1; font-family:monospace;">{sub_text}</div>
    </div>
    """

def _clamp_offset_px(v) -> float:
    """Drží jemný posun prvku v rozsahu, ktorý reprezentujú Gradio slidery."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(f):
        return 0.0
    return max(-OFFSET_LIMIT_PX, min(OFFSET_LIMIT_PX, f))


def _mask_bbox_px(mask_np, scale=2.0):
    """Rozmer (šírka, výška) obrysu masky v pixeloch náhľadového plátna (112 px * scale)."""
    ys, xs = np.where(mask_np > 0.25)
    if len(ys) == 0:
        return None
    return (
        float(xs.max() - xs.min() + 1) * scale,
        float(ys.max() - ys.min() + 1) * scale,
    )


def render_live_placement_preview(
    image,
    attack_name,
    patch_pos,
    patch_size_choice,
    constellation_pattern,
    patch_radius,
    strip_scale=1.0,
    strip_ox=0.0,
    strip_oy=0.0,
    left_ox=0.0,
    left_oy=0.0,
    right_ox=0.0,
    right_oy=0.0,
    chin_ox=0.0,
    chin_oy=0.0,
):
    if image is None:
        return None, build_drag_canvas_html(None, None, None, None, attack_name), format_occlusion_badge(0.0)

    try:
        img_id = (id(image), getattr(image, "size", None))
        if img_id != _live_preview_cache["cached_img_id"] or _live_preview_cache["face_np"] is None:
            face_tensor, method, err = get_face_tensor_from_image(image)
            if face_tensor is None:
                return None, build_drag_canvas_html(None, None, None, None, attack_name), format_occlusion_badge(0.0)
            lms, mesh = extract_landmarks(face_tensor, mtcnn=mtcnn, return_mesh=True)
            face_np = ((face_tensor[0].detach().permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            _live_preview_cache["cached_img_id"] = img_id
            _live_preview_cache["face_np"] = face_np
            _live_preview_cache["landmarks"] = lms
            _live_preview_cache["mesh_info"] = mesh
        else:
            face_np = _live_preview_cache["face_np"]
            lms = _live_preview_cache["landmarks"]
            mesh = _live_preview_cache["mesh_info"]

        if face_np is None or lms is None:
            return None, build_drag_canvas_html(None, None, None, None, attack_name), format_occlusion_badge(0.0)

        canvas = face_np.copy()
        sox = _clamp_offset_px(strip_ox)
        soy = _clamp_offset_px(strip_oy)
        lox = _clamp_offset_px(left_ox)
        loy = _clamp_offset_px(left_oy)
        rox = _clamp_offset_px(right_ox)
        roy = _clamp_offset_px(right_oy)
        cox = _clamp_offset_px(chin_ox)
        coy = _clamp_offset_px(chin_oy)

        cur_tot_pct = 0.0
        cur_strip_pct = 0.0
        cur_dots_pct = 0.0
        strip_box_px = None
        strip_angle_deg = 0.0

        # Športová páska na nos (Breathe Right)
        if attack_name == "Športová páska na nos (Breathe Right)":
            s_w_scale = float(strip_scale) if strip_scale is not None else 1.0
            place = compute_nasal_strip_placement(lms, mesh, width_scale=s_w_scale, offset_x=sox, offset_y=soy)
            mask, relief, _, _ = create_nasal_strip_mask(
                112, (place["cy"], place["cx"]),
                place["eff_w"], place["eff_hc"], place["eff_hw"],
                place["angle"], place["yaw"],
                arch_factor=place["arch"], arch_drop_px=place["arch_px"],
            )
            m_np = mask.squeeze().cpu().numpy()
            r_np = relief.squeeze().cpu().numpy()
            strip_box_px = _mask_bbox_px(m_np)
            strip_angle_deg = float(place["angle"]) * 180.0 / math.pi

            cur_tot_pct = float(mask.sum().item()) / (112.0 * 112.0) * 100.0
            cur_strip_pct = cur_tot_pct
            cur_dots_pct = 0.0

            if "karbón" in str(patch_pos).lower() or "carbon" in str(patch_pos).lower():
                color = np.array([38, 40, 44], dtype=np.float32)
            elif "telová" in str(patch_pos).lower() or "tan" in str(patch_pos).lower():
                color = np.array([210, 168, 125], dtype=np.float32)
            elif "priesvitná" in str(patch_pos).lower() or "clear" in str(patch_pos).lower():
                color = np.array([225, 225, 230], dtype=np.float32)
            else:
                color = np.array([22, 22, 22], dtype=np.float32)

            for c in range(3):
                canvas[:, :, c] = np.clip((1.0 - m_np) * canvas[:, :, c] + m_np * color[c] * r_np, 0, 255).astype(np.uint8)

        # Constellation Patch (Pimple Patches)
        elif attack_name == "Constellation Patch (Pimple Patches)":
            spot_offsets = {
                "nose_bridge": (sox, soy), "asym_nose": (sox, soy), "star_nose": (sox, soy), "cluster_nose": (sox, soy), 0: (sox, soy),
                "left_cheek": (lox, loy), "asym_left_cheek": (lox, loy), "star_left_cheek": (lox, loy), "cluster_left_1": (lox, loy), 1: (lox, loy),
                "right_cheek": (rox, roy), "asym_right_cheek": (rox, roy), "star_right_cheek": (rox, roy), "cluster_right": (rox, roy), 2: (rox, roy),
                "chin": (cox, coy), "asym_chin": (cox, coy), "cluster_chin": (cox, coy), 3: (cox, coy),
            }
            pat_map = {
                "3 body: Nos + Obe líca": "trio_cheeks_nose",
                "5 bodov: Cluster (Líca + Nos + Brada) [Prah účinnosti]": "cluster_5",
                "Asymetrické nálepky (Náhodné 3 body)": "asymmetric_random",
                "Starface hviezdičky (3 žlté hviezdičky)": "star_patches",
                "Adversariálne pehy / Freckles (8 mikro-bodov)": "adversarial_freckles",
                "4 body: Nos + Líca + Brada (Full)": "quad_full",
                "2 body: Iba líca": "cheeks_only",
                "T-Zóna: Čelo + Nos + Brada": "t_zone"
            }
            pat_key = pat_map.get(constellation_pattern, "trio_cheeks_nose")
            r_val = float(patch_radius) if patch_radius else 4.5
            specs = get_anatomical_patch_specs(lms, pattern=pat_key, base_radius=r_val, mesh_info=mesh, spot_offsets=spot_offsets, seed=42)
            const_full_m = torch.zeros((1, 1, 112, 112), dtype=torch.float32)
            for s in specs:
                m, d, _, _ = create_elliptical_patch_mask(112, s["center"], s["rx"], s["ry"], s["angle"], shape=s.get("shape", "circle"))
                const_full_m = torch.maximum(const_full_m, m)
                m_np = m.squeeze().cpu().numpy()
                d_np = d.squeeze().cpu().numpy()
                if "star" in pat_key:
                    col = np.array([250, 225, 45], dtype=np.float32)
                elif "freckle" in pat_key:
                    col = np.array([125, 78, 48], dtype=np.float32)
                else:
                    col = np.array([220, 185, 145], dtype=np.float32)
                for c in range(3):
                    canvas[:, :, c] = np.clip((1.0 - m_np * 0.90) * canvas[:, :, c] + m_np * 0.90 * col[c] * d_np, 0, 255).astype(np.uint8)

            cur_tot_pct = float(const_full_m.sum().item()) / (112.0 * 112.0) * 100.0
            cur_strip_pct = 0.0
            cur_dots_pct = cur_tot_pct

        # Adversarial Patch (Leukoplast)
        elif attack_name == "Adversarial Patch (Leukoplast)":
            pos_map = {"Koreň nosa (nose_bridge)": "nose_bridge", "Líce (cheek)": "cheek", "Čelo (forehead)": "forehead"}
            size_map = {"Stredný (7x18 px / 22x8 mm)": (7, 18), "Malý (5x14 px / 18x6 mm)": (5, 14), "Veľký (9x22 px / 26x10 mm)": (9, 22)}
            c_pos = pos_map.get(patch_pos, "nose_bridge")
            c_sz = size_map.get(patch_size_choice, (7, 18))
            mask, relief, _ = get_patch_geometry(c_pos, 112, c_sz, lms, mesh, sox, soy)
            m_np = mask.squeeze().cpu().numpy()
            r_np = relief.squeeze().cpu().numpy()
            cur_tot_pct = float(mask.sum().item()) / (112.0 * 112.0) * 100.0
            cur_strip_pct = cur_tot_pct
            cur_dots_pct = 0.0
            col = np.array([215, 172, 128], dtype=np.float32)
            for c in range(3):
                canvas[:, :, c] = np.clip((1.0 - m_np * 0.92) * canvas[:, :, c] + m_np * 0.92 * col[c] * r_np, 0, 255).astype(np.uint8)

        # Hybrid Patch (Páska na nos + Pimple Patches) / BioMesh-SpecPatch
        elif ("Hybrid Patch" in str(attack_name)) or ("BioMesh-SpecPatch" in str(attack_name)):
            dot_pat = "cheeks_chin" if "3 nálepky" in str(constellation_pattern) else ("1_cheek" if "1 nálepka" in str(constellation_pattern) else "2_cheeks")
            s_w_scale = float(strip_scale) if strip_scale is not None else 1.0
            f_mask, f_relief, boxes = get_hybrid_geometry(
                landmarks=lms,
                mesh_info=mesh,
                img_size=112,
                dot_radius=float(patch_radius) if patch_radius else 5.5,
                dot_pattern=dot_pat,
                strip_w_scale=s_w_scale,
                offset_x=0.0,
                offset_y=0.0,
                strip_ox=sox,
                strip_oy=soy,
                left_ox=lox,
                left_oy=loy,
                right_ox=rox,
                right_oy=roy,
                chin_ox=cox,
                chin_oy=coy,
            )
            strip_m = boxes["strip_mask"]
            strip_r = boxes["strip_relief"]
            sm_np = strip_m.squeeze().cpu().numpy()
            sr_np = strip_r.squeeze().cpu().numpy()
            strip_box_px = _mask_bbox_px(sm_np)
            strip_angle_deg = float(boxes.get("strip_angle", 0.0)) * 180.0 / math.pi
            if "rgb" in str(patch_pos).lower() or "farebn" in str(patch_pos).lower() or "art" in str(patch_pos).lower():
                s_col = np.array([235, 90, 50], dtype=np.float32)
                d_col = np.array([245, 190, 40], dtype=np.float32)
            elif "karbón" in str(patch_pos).lower() or "carbon" in str(patch_pos).lower():
                s_col = np.array([38, 40, 44], dtype=np.float32)
                d_col = np.array([220, 185, 145], dtype=np.float32)
            elif "telová" in str(patch_pos).lower() or "tan" in str(patch_pos).lower():
                s_col = np.array([210, 168, 125], dtype=np.float32)
                d_col = np.array([220, 185, 145], dtype=np.float32)
            elif "priesvitná" in str(patch_pos).lower() or "clear" in str(patch_pos).lower():
                s_col = np.array([225, 225, 230], dtype=np.float32)
                d_col = np.array([220, 185, 145], dtype=np.float32)
            else:
                s_col = np.array([22, 22, 22], dtype=np.float32)
                d_col = np.array([220, 185, 145], dtype=np.float32)

            for c in range(3):
                canvas[:, :, c] = np.clip((1.0 - sm_np) * canvas[:, :, c] + sm_np * s_col[c] * sr_np, 0, 255).astype(np.uint8)

            dot_mask = torch.clamp(f_mask - strip_m.to(f_mask.device), 0.0, 1.0)
            dm_np = dot_mask.squeeze().cpu().numpy()
            for c in range(3):
                canvas[:, :, c] = np.clip((1.0 - dm_np * 0.90) * canvas[:, :, c] + dm_np * 0.90 * d_col[c], 0, 255).astype(np.uint8)

            cur_tot_pct = float(f_mask.sum().item()) / (112.0 * 112.0) * 100.0
            cur_strip_pct = float(strip_m.sum().item()) / (112.0 * 112.0) * 100.0
            cur_dots_pct = float(dot_mask.sum().item()) / (112.0 * 112.0) * 100.0

        res_img = cv2.resize(canvas, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        nose_is_dot = ("Constellation" in attack_name) and get_active_handles_for_attack(attack_name, constellation_pattern)["strip"]
        base_coords = extract_base_patch_coords(
            lms, mesh, nose_is_dot=nose_is_dot,
            strip_scale=float(strip_scale) if strip_scale is not None else 1.0,
        )
        active_handles = get_active_handles_for_attack(attack_name, constellation_pattern)
        offsets_dict = {
            "strip_ox": sox, "strip_oy": soy,
            "left_ox": lox, "left_oy": loy,
            "right_ox": rox, "right_oy": roy,
            "chin_ox": cox, "chin_oy": coy,
            "strip_scale": float(strip_scale) if strip_scale is not None else 1.0,
            "patch_radius": float(patch_radius) if patch_radius is not None else 5.5,
            "strip_w_px": strip_box_px[0] if strip_box_px else 0.0,
            "strip_h_px": strip_box_px[1] if strip_box_px else 0.0,
            "strip_angle_deg": strip_angle_deg,
            # Rozsah posunu, do ktorého JS drží úchytky (aby sa úchytka nikdy
            # nedostala mimo rozsah sliderov) – jeden zdroj pravdy s backendom.
            "offset_limit": OFFSET_LIMIT_PX,
        }
        drag_html = build_drag_canvas_html(res_img, base_coords, active_handles, offsets_dict, attack_name)
        is_digital = str(attack_name) in ["FGSM", "PGD", "BIM", "MI-FGSM", "C&W"]
        badge_html = format_occlusion_badge(cur_tot_pct, cur_strip_pct, cur_dots_pct, is_digital=is_digital)
        return res_img, drag_html, badge_html
    except Exception:
        return None, build_drag_canvas_html(None, None, None, None, attack_name), format_occlusion_badge(0.0)


def render_ar_stream_frame(
    frame,
    constellation_pattern="2 nálepky: Obe líca",
    patch_radius=4.5,
    strip_scale=1.0,
    strip_ox=0.0,
    strip_oy=0.0,
    left_ox=0.0,
    left_oy=0.0,
    right_ox=0.0,
    right_oy=0.0,
    chin_ox=0.0,
    chin_oy=0.0,
):
    """Živý AR Zameriavač: na každom snímku z webkamery deteguje 3D FaceMesh a vykreslí
    presné obrysy [N] NOS ^, [L] LAVE LICE ^, [P] PRAVE LICE ^ (a prípadne [B] BRADA ^)
    v reálnom čase, aby si používateľ vedel nalepiť nálepky presne na správne miesto."""
    if frame is None:
        return None, None
    try:
        pil_img = to_pil_image(frame)
        if pil_img is None:
            return None, None
        face_tensor, _, err = get_face_tensor_from_image(pil_img)
        if face_tensor is None:
            arr = np.array(pil_img.resize((336, 336)))
            cv2.putText(arr, "Namier tvar do stredu kamery...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 80, 80), 2, cv2.LINE_AA)
            return arr, pil_img

        lms, mesh = extract_landmarks(face_tensor, mtcnn=mtcnn, return_mesh=True)
        face_112 = ((face_tensor[0].detach().permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        scale = 3
        out = cv2.resize(face_112, (112 * scale, 112 * scale), interpolation=cv2.INTER_LINEAR)

        dot_pat = "cheeks_chin" if "3 nálepky" in str(constellation_pattern) else ("1_cheek" if "1 nálepka" in str(constellation_pattern) else "2_cheeks")
        r_px = float(patch_radius) if patch_radius else 5.5
        s_w_scale = float(strip_scale) if strip_scale else 1.0
        _, _, boxes = get_hybrid_geometry(
            landmarks=lms,
            mesh_info=mesh,
            img_size=112,
            dot_radius=r_px,
            dot_pattern=dot_pat,
            strip_w_scale=s_w_scale,
            strip_ox=float(strip_ox or 0.0),
            strip_oy=float(strip_oy or 0.0),
            left_ox=float(left_ox or 0.0),
            left_oy=float(left_oy or 0.0),
            right_ox=float(right_ox or 0.0),
            right_oy=float(right_oy or 0.0),
            chin_ox=float(chin_ox or 0.0),
            chin_oy=float(chin_oy or 0.0),
        )

        # 1. Obrys nosového pásika [N]
        sm = boxes.get("strip_mask")
        if sm is not None:
            sm_u8 = (cv2.resize(sm.squeeze().detach().cpu().numpy(), (112 * scale, 112 * scale)) > 0.25).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(sm_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, (0, 229, 255), 2, cv2.LINE_AA)
            sy1, sy2, sx1, sx2 = boxes["strip"]
            scx, scy = int((sx1 + sx2) * 0.5 * scale), int((sy1 + sy2) * 0.5 * scale)
            cv2.drawMarker(out, (scx, scy), (0, 229, 255), cv2.MARKER_CROSS, 12, 2)
            cv2.putText(out, "[N] NOS ^", (max(4, scx - 36), max(16, int(sy1 * scale) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 229, 255), 2, cv2.LINE_AA)

        # 2. Obrysy bodiek [L], [P], [B]
        dot_geom = boxes.get("dot_geom", {})
        dot_meta = [
            ("left_dot", "[L] LAVE ^", (213, 0, 249)),
            ("right_dot", "[P] PRAVE ^", (0, 230, 118)),
            ("chin_dot", "[B] BRADA ^", (255, 145, 0)),
        ]
        for dkey, dlabel, dcol in dot_meta:
            if dkey not in dot_geom:
                continue
            cx, cy, d_cos, d_tilt = dot_geom[dkey]
            center_pt = (int(round(cx * scale)), int(round(cy * scale)))
            axes = (max(4, int(round(r_px * d_cos * scale))), max(4, int(round(r_px * scale))))
            ang_deg = float(d_tilt * 180.0 / np.pi)
            cv2.ellipse(out, center_pt, axes, ang_deg, 0, 360, dcol, 2, cv2.LINE_AA)
            cv2.drawMarker(out, center_pt, dcol, cv2.MARKER_CROSS, 10, 2)
            # Šípka HORE na vrchu elipsy
            top_y = max(14, center_pt[1] - axes[1] - 6)
            cv2.putText(out, dlabel, (max(4, center_pt[0] - 40), top_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, dcol, 2, cv2.LINE_AA)

        cv2.putText(out, "Zivy AR Zameriavac (^ = strana HORE k ociam)", (8, 328),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
        return out, pil_img
    except Exception:
        return None, None


def _format_attack_status(method, similarity, occ_pct=None, offsets=None, hint="", sim_1toN=None):
    lines = [f"Metóda: {method}"]
    if sim_1toN is not None:
        lines.append(f"Podobnosť 1:N (voči databáze): {sim_1toN:.4f} (prah 0.50) | Podobnosť 1:1 (rovnaký výrez): {similarity:.4f}")
    else:
        lines.append(f"Podobnosť s pôvodnou fotkou: {similarity:.4f} (prah 0.50)")
    if occ_pct is not None:
        lines.append(f"Zakrytá plocha tváre: {occ_pct:.1f} %")
    if offsets:
        lines.append("Posun (px): " + ", ".join(offsets))
    eff_sim = sim_1toN if sim_1toN is not None else similarity
    if eff_sim < 0.5:
        lines.append("Výsledok: ÚTOK ÚSPEŠNÝ – model osobu v databáze NEROZPOZNAL (podobnosť pod 0.50).")
    else:
        lines.append("Výsledok: model tvár stále rozpoznal (podobnosť nad 0.50)." + (f" {hint}" if hint else ""))
    return "\n".join(lines)


# Funkcie pre záložku 1: Útoky
def run_attack(
    image,
    model_name,
    attack_name,
    epsilon,
    alpha,
    num_iter,
    patch_position="Koreň nosa (nose_bridge)",
    patch_size_choice="Stredný (7x18 px / 22x8 mm)",
    constellation_pattern="2 nálepky: Obe líca",
    patch_radius=5.5,
    strip_scale=1.15,
    hybrid_style_mode="Vypnutá (plné RGB / podľa štýlu)",
    ensemble_mode="3 modely | Black-Box: AdaFace (vynechaný z tréningu)",
    strip_ox=0.0,
    strip_oy=0.0,
    left_ox=0.0,
    left_oy=0.0,
    right_ox=0.0,
    right_oy=0.0,
    chin_ox=0.0,
    chin_oy=0.0,
):
    if image is None:
        return None, None, None, "", "", "Najprv nahraj alebo odfoť obrázok."
    
    img_tensor, method, err_msg = get_face_tensor_from_image(image)
    if img_tensor is None:
        return None, None, None, "", "", err_msg

    model = models_dict.get(model_name)
    if model is None:
        return None, None, None, "", "", f"Model {model_name} nie je dostupný."
    
    # Rozpoznanie identity na čistej tvári
    clean_id = predict_person_identity(model, model_name, img_tensor, is_attack=False)

    # Pôvodný embedding
    with torch.no_grad():
        orig_emb = model(img_tensor)
        
    patch_info = None
    constellation_info = None
    hybrid_info = None
    nasal_info = None
    printable_sheet = None

    size_map = {
        "Stredný (7x18 px / 22x8 mm)": (7, 18),
        "Malý (5x14 px / 18x6 mm)": (5, 14),
        "Veľký (9x22 px / 26x10 mm)": (9, 22),
    }
    chosen_patch_size = size_map.get(patch_size_choice, (7, 18))

    sox = float(strip_ox) if strip_ox is not None else 0.0
    soy = float(strip_oy) if strip_oy is not None else 0.0
    lox = float(left_ox) if left_ox is not None else 0.0
    loy = float(left_oy) if left_oy is not None else 0.0
    rox = float(right_ox) if right_ox is not None else 0.0
    roy = float(right_oy) if right_oy is not None else 0.0
    cox = float(chin_ox) if chin_ox is not None else 0.0
    coy = float(chin_oy) if chin_oy is not None else 0.0

    print(f"[útok] {attack_name} | cieľ={model_name} | ensemble_mode={ensemble_mode} | posuny páska=({sox}, {soy}) ľavé=({lox}, {loy}) pravé=({rox}, {roy}) brada=({cox}, {coy})")

    # Výber a spustenie útoku
    if attack_name == "FGSM":
        adv_tensor = fgsm_attack_untargeted(model, img_tensor, epsilon=epsilon)
    elif attack_name == "PGD":
        adv_tensor = pgd_attack_untargeted(model, img_tensor, epsilon=epsilon, alpha=alpha, num_iter=int(num_iter))
    elif attack_name == "BIM":
        adv_tensor = bim_attack_untargeted(model, img_tensor, epsilon=epsilon, alpha=alpha, num_iter=int(num_iter))
    elif attack_name == "MI-FGSM":
        adv_tensor = mifgsm_attack_untargeted(model, img_tensor, epsilon=epsilon, alpha=alpha, num_iter=int(num_iter))
    elif attack_name == "C&W":
        adv_tensor = cw_l2_attack_untargeted(model, img_tensor, c=10.0, kappa=0.0, max_iter=int(num_iter))
    elif attack_name == "Adversarial Patch (Leukoplast)":
        pos_map = {
            "Koreň nosa (nose_bridge)": "nose_bridge",
            "Líce (cheek)": "cheek",
            "Čelo (forehead)": "forehead"
        }
        chosen_pos = pos_map.get(patch_position, "nose_bridge")
        adv_tensor, patch_crop, patch_info = adversarial_patch_attack(
            model=model,
            image=img_tensor,
            position=chosen_pos,
            patch_size=chosen_patch_size,
            num_iter=int(num_iter),
            eot=True,
            mtcnn=mtcnn,
            offset_x=sox,
            offset_y=soy,
        )
    elif attack_name == "Športová páska na nos (Breathe Right)":
        p_str = str(patch_position).lower()
        if "telov" in p_str or "tan" in p_str:
            chosen_style = "tan_fabric"
        elif "priesvitn" in p_str or "clear" in p_str:
            chosen_style = "clear_silicone"
        elif "karbón" in p_str or "carbon" in p_str:
            chosen_style = "pro_carbon_logo"
        elif "rgb" in p_str or "farebn" in p_str or "digitáln" in p_str:
            chosen_style = "adversarial_rgb"
        else:
            chosen_style = "athlete_black"
        adv_tensor, patch_crop, nasal_info = nasal_strip_attack(
            model=model,
            image=img_tensor,
            style=chosen_style,
            width_scale=float(strip_scale) if strip_scale is not None else 1.0,
            num_iter=int(num_iter),
            eot=True,
            mtcnn=mtcnn,
            offset_x=sox,
            offset_y=soy,
        )
    elif attack_name == "Constellation Patch (Pimple Patches)":
        pat_map = {
            "3 body: Nos + Obe líca": "trio_cheeks_nose",
            "5 bodov: Cluster (Líca + Nos + Brada) [Prah účinnosti]": "cluster_5",
            "Asymetrické nálepky (Náhodné 3 body)": "asymmetric_random",
            "Starface hviezdičky (3 žlté hviezdičky)": "star_patches",
            "Adversariálne pehy / Freckles (8 mikro-bodov)": "adversarial_freckles",
            "4 body: Nos + Líca + Brada (Full)": "quad_full",
            "2 body: Iba líca": "cheeks_only",
            "T-Zóna: Čelo + Nos + Brada": "t_zone"
        }
        chosen_pat = pat_map.get(constellation_pattern, "trio_cheeks_nose")
        spot_offsets = {
            "nose_bridge": (sox, soy), "asym_nose": (sox, soy), "star_nose": (sox, soy), "cluster_nose": (sox, soy), 0: (sox, soy),
            "left_cheek": (lox, loy), "asym_left_cheek": (lox, loy), "star_left_cheek": (lox, loy), "cluster_left_1": (lox, loy), 1: (lox, loy),
            "right_cheek": (rox, roy), "asym_right_cheek": (rox, roy), "star_right_cheek": (rox, roy), "cluster_right": (rox, roy), 2: (rox, roy),
            "chin": (cox, coy), "asym_chin": (cox, coy), "cluster_chin": (cox, coy), 3: (cox, coy),
        }
        adv_tensor, patch_crop, constellation_info = constellation_patch_attack(
            model=model,
            image=img_tensor,
            pattern=chosen_pat,
            radius=float(patch_radius),
            num_iter=int(num_iter),
            eot=True,
            mtcnn=mtcnn,
            offset_x=0.0,
            offset_y=0.0,
            spot_offsets=spot_offsets,
        )
    elif ("Hybrid Patch" in str(attack_name)) or ("BioMesh-SpecPatch" in str(attack_name)):
        h_iters = int(num_iter)
        if "3 nálepky" in str(constellation_pattern):
            dot_pat = "cheeks_chin"
        elif "1 nálepka" in str(constellation_pattern):
            dot_pat = "1_cheek"
        else:
            dot_pat = "2_cheeks"

        style_str = str(patch_position)
        chosen_strip_style = "athlete_black"
        chosen_palette = None
        chosen_logo_mask = None

        if "Čierna" in style_str or "Athlete" in style_str:
            chosen_strip_style = "athlete_black"
        elif "Telová" in style_str or "Tan" in style_str:
            chosen_strip_style = "tan_fabric"
        elif "Priesvitná" in style_str or "Clear" in style_str:
            chosen_strip_style = "clear_silicone"
        elif "Karbón" in style_str or "Carbon" in style_str:
            chosen_strip_style = "pro_carbon_logo"
            chosen_palette = [[-1, -1, -1], [0.9, 0.1, 0.1]]
        elif "Kinesio" in style_str:
            chosen_strip_style = "athlete_black"
            chosen_palette = [[-1, -1, -1], [0.7, 0.45, 0.25], [-0.6, -0.2, -0.9]]
        elif "hviezda" in style_str.lower() or "starface" in style_str.lower():
            chosen_strip_style = "athlete_black"
            chosen_palette = [[-1, -1, -1], [0.9, 0.1, 0.1]]
            _yy, _xx = np.mgrid[0:96, 0:96]
            _ang = np.arctan2(_yy - 47.5, _xx - 47.5)
            _rad = np.hypot(_xx - 47.5, _yy - 47.5)
            chosen_logo_mask = torch.from_numpy(
                ((_rad < 40) & (np.cos(5 * _ang) > 0.15)).astype(np.float32)
            ).to(img_tensor.device)
        elif "RGB" in style_str or "digitálny" in style_str or "farebn" in style_str.lower():
            chosen_strip_style = "adversarial_rgb"
        else:
            chosen_strip_style = "athlete_black"

        s_w_scale = float(strip_scale) if strip_scale is not None else 1.0
        holdout_model_name = None
        if "BioMesh-SpecPatch" in str(attack_name):
            core_order = ("BenchmarkCNN", "BenchmarkCNN (Finetuned)", "FaceNet", "FaceNet (CASIA)", "ArcFace", "AdaFace")
            if "Samostatný" in str(ensemble_mode):
                ens_models = {model_name: model}
                holdout_model_name = None
            elif "Plný" in str(ensemble_mode):
                ens_models = {
                    k: models_dict[k]
                    for k in core_order
                    if k in models_dict and models_dict[k] is not None
                }
                holdout_model_name = None
            else:
                # Black-Box Leave-One-Out: trénuj na zvyšných, vynechaj model_name
                ens_models = {
                    k: models_dict[k]
                    for k in core_order
                    if k in models_dict and models_dict[k] is not None and k != model_name
                }
                holdout_model_name = model_name

            if not ens_models:
                ens_models = {model_name: model}

            # Zistíme priradenú osobu v 1:N galérii a jej centroidy pre jednotlivé modely
            target_gallery = get_model_gallery(model, model_name)
            matched_person = None
            best_clean_g = -1.0
            for p_key, c_emb in target_gallery.items():
                sc = F.cosine_similarity(orig_emb, c_emb).item()
                if sc > best_clean_g:
                    best_clean_g = sc
                    matched_person = p_key

            g_cent = {}
            if matched_person is not None:
                for k_m, m_obj in ens_models.items():
                    gal_m = get_model_gallery(m_obj, k_m)
                    if matched_person in gal_m:
                        g_cent[k_m] = gal_m[matched_person]

            adv_tensor, printable_sheet, hybrid_info = ensemble_hybrid_patch_attack(
                models=ens_models,
                image=img_tensor,
                strip_style=chosen_strip_style,
                dot_radius=float(patch_radius),
                dot_pattern=dot_pat,
                strip_w_scale=s_w_scale,
                num_iter=h_iters,
                lr=0.09,
                eot=True,
                grad_avg=False,
                feature_weight=0.0,
                input_diversity=False,
                multiscale=True,
                ti_kernel=3,
                pad_model=pad_detector,
                pad_weight=0.20 if pad_detector is not None else 0.0,
                gallery_centroids=g_cent if g_cent else None,
                palette=chosen_palette,
                logo_mask=chosen_logo_mask,
                mtcnn=mtcnn,
                offset_x=0.0,
                offset_y=0.0,
                strip_ox=sox,
                strip_oy=soy,
                left_ox=lox,
                left_oy=loy,
                right_ox=rox,
                right_oy=roy,
                chin_ox=cox,
                chin_oy=coy,
            )
        else:
            adv_tensor, printable_sheet, hybrid_info = hybrid_patch_attack(
                model=model,
                image=img_tensor,
                strip_style=chosen_strip_style,
                dot_radius=float(patch_radius),
                dot_pattern=dot_pat,
                strip_w_scale=s_w_scale,
                num_iter=h_iters,
                eot=True,
                palette=chosen_palette,
                logo_mask=chosen_logo_mask,
                mtcnn=mtcnn,
                offset_x=0.0,
                offset_y=0.0,
                strip_ox=sox,
                strip_oy=soy,
                left_ox=lox,
                left_oy=loy,
                right_ox=rox,
                right_oy=roy,
                chin_ox=cox,
                chin_oy=coy,
            )
    else:
        return None, None, None, "", "", "Neznámy útok."
        
    # Adversariálny embedding a 1:N podobnosť voči databáze
    with torch.no_grad():
        adv_emb = model(adv_tensor)
        
    similarity = F.cosine_similarity(orig_emb, adv_emb).item()
    sim_1toN = None
    gal_target = get_model_gallery(model, model_name)
    if gal_target:
        with torch.no_grad():
            sim_1toN = max(F.cosine_similarity(adv_emb, c_emb).item() for c_emb in gal_target.values())
    
    # Konverzia späť na obrázky pre zobrazenie
    adv_img_np = denormalize(adv_tensor).squeeze().permute(1, 2, 0).cpu().numpy()
    
    def _offset_parts(active):
        parts = []
        if active["strip"]:
            parts.append(f"páska ({sox:+.1f}, {soy:+.1f})")
        if active["left"]:
            parts.append(f"ľavé líce ({lox:+.1f}, {loy:+.1f})")
        if active["right"]:
            parts.append(f"pravé líce ({rox:+.1f}, {roy:+.1f})")
        if active["chin"]:
            parts.append(f"brada ({cox:+.1f}, {coy:+.1f})")
        return parts

    if patch_info is not None:
        patch_np = ((patch_crop.detach().cpu().permute(1, 2, 0).numpy() + 1.0) * 0.5 * 255.0).clip(0, 255).astype("uint8")
        diff_np = cv2.resize(patch_np, (112, 112), interpolation=cv2.INTER_NEAREST)
        status_text = _format_attack_status(
            f"leukoplast ({patch_position})", similarity,
            patch_info.get("face_occlusion_pct", 2.5), [f"leukoplast ({sox:+.1f}, {soy:+.1f})"],
            "Skús 80-120 iterácií.")
    elif nasal_info is not None:
        patch_np = ((patch_crop.detach().cpu().permute(1, 2, 0).numpy() + 1.0) * 0.5 * 255.0).clip(0, 255).astype("uint8")
        diff_np = cv2.resize(patch_np, (112, 112), interpolation=cv2.INTER_NEAREST)
        status_text = _format_attack_status(
            f"športová nosová páska ({nasal_info.get('style', 'athlete_black')})", similarity,
            nasal_info.get("face_occlusion_pct", 2.4), [f"páska ({sox:+.1f}, {soy:+.1f})"],
            "Skús 80-120 iterácií.")
    elif constellation_info is not None:
        patch_np = ((patch_crop.detach().cpu().permute(1, 2, 0).numpy() + 1.0) * 0.5 * 255.0).clip(0, 255).astype("uint8")
        diff_np = cv2.resize(patch_np, (112, 112), interpolation=cv2.INTER_NEAREST)
        rad_px = constellation_info.get("radius_px", patch_radius)
        if "freckle" in constellation_pattern.lower():
            pat_label = "pehy"
        elif "star" in constellation_pattern.lower():
            pat_label = "hviezdičkové nálepky"
        else:
            pat_label = f"nálepky ({constellation_pattern})"
        status_text = _format_attack_status(
            f"{pat_label}, polomer {rad_px} px", similarity,
            constellation_info.get("face_occlusion_pct", 3.0),
            _offset_parts(get_active_handles_for_attack(attack_name, constellation_pattern)),
            "Skús viac iterácií alebo 4 body.")
    elif hybrid_info is not None:
        diff_np = cv2.cvtColor(printable_sheet, cv2.COLOR_BGR2RGB)
        if "BioMesh-SpecPatch" in str(attack_name):
            if "Samostatný" in str(ensemble_mode):
                label_name = f"BioMesh-SpecPatch (Samostatný model: {model_name} | White-Box)"
            elif "Plný" in str(ensemble_mode):
                label_name = f"BioMesh-SpecPatch (Plný Ensemble 4 modelov | Vyhodnotenie: {model_name})"
            else:
                label_name = f"BioMesh-SpecPatch (Tréning: 3 modely | Black-Box cieľ: {model_name})"
        else:
            label_name = "páska na nose + nálepky na lícach"
        status_text = _format_attack_status(
            label_name, similarity,
            hybrid_info.get("face_occlusion_pct", 5.7),
            _offset_parts(get_active_handles_for_attack(attack_name, constellation_pattern)),
            "Skús 160 iterácií alebo mierku pásky 1.15 / nálepky 5.5 px.",
            sim_1toN=sim_1toN,
        )
        if "per_model_similarity" in hybrid_info:
            sim_parts = []
            with torch.no_grad():
                for k, v11 in hybrid_info["per_model_similarity"].items():
                    m_k = models_dict.get(k)
                    g_k = get_model_gallery(m_k, k) if m_k is not None else {}
                    if m_k is not None and g_k:
                        emb_k = m_k(adv_tensor)
                        v1n = max(F.cosine_similarity(emb_k, c_e).item() for c_e in g_k.values())
                        sim_parts.append(f"{k}: 1:N={v1n:.3f} (1:1={v11:.3f})")
                    else:
                        sim_parts.append(f"{k}: {v11:.3f}")
            sims_str = ", ".join(sim_parts)
            if "Samostatný" in str(ensemble_mode):
                status_text += f"\nOptimalizovaný model ({model_name}): {sims_str}"
            else:
                status_text += f"\nTréningové modely (White-Box Ensemble): {sims_str}"
        if holdout_model_name and holdout_model_name in models_dict:
            eff_bb = sim_1toN if sim_1toN is not None else similarity
            bb_verdict = "PRELOMENÝ (< 0.50)" if eff_bb < 0.5 else "STÁLE ROZPOZNANÝ (>= 0.50)"
            if sim_1toN is not None:
                status_text += f"\n🎯 Vynechaný Black-Box model ({holdout_model_name}): 1:N = {sim_1toN:.4f} (1:1 = {similarity:.4f}) -> {bb_verdict}"
            else:
                status_text += f"\n🎯 Vynechaný Black-Box model ({holdout_model_name}): podobnosť = {similarity:.4f} -> {bb_verdict}"
        if pad_detector is not None:
            with torch.no_grad():
                p_live_adv = float(pad_detector(adv_tensor).mean().item())
            status_text += f"\nPAD Liveness (MiniFASNetV2): P(live) = {p_live_adv * 100.0:.1f} % ({'LIVE' if p_live_adv >= 0.5 else 'SPOOF'})"
        status_text += "\nTlačový hárok (A4, 100 % mierka, 300 DPI) je vpravo — vytlač, vystrihni a nalep."
    else:
        # Zvýraznený šum pre klasické digitálne útoky (uint8 0..255)
        diff_np = (adv_tensor - img_tensor).squeeze().permute(1, 2, 0).cpu().numpy()
        diff_np = ((diff_np * 10 + 0.5).clip(0, 1) * 255.0).astype(np.uint8)
        status_text = _format_attack_status(f"digitálny útok {attack_name}", similarity)

    # Vyhodnotenie identity po útoku
    adv_id = predict_person_identity(model, model_name, adv_tensor, is_attack=True, sim_1to1=similarity)
    
    # Pôvodná tvár a adversariálna tvár v uint8 [0, 255]
    orig_face_np = (denormalize(img_tensor).squeeze().permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    adv_img_np = (denormalize(adv_tensor).squeeze().permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        
    return orig_face_np, adv_img_np, diff_np, clean_id, adv_id, status_text

# Funkcie pre záložku 2: Zber dát
def save_image_to_dataset(image, person_name):
    if image is None:
        return "Žiadny obrázok na uloženie. Najprv odfoť alebo nahraj fotku.", None
    if not person_name or not str(person_name).strip():
        return "Zadaj meno osoby alebo vyber existujúcu.", None
    
    pil_img = to_pil_image(image)
    if pil_img is None:
        return "Obrázok sa nepodarilo spracovať.", None
        
    arr_check = np.array(pil_img)
    if arr_check.mean() < 3.0:
        return "Webkamera poslala čierny záber. Skontroluj krytku kamery a odfoť sa znova.", None
        
    face_t_255, method = crop_face_from_image(pil_img)
    if face_t_255 is None:
        return "Na fotke sa nepodarilo nájsť tvár.", None
        
    face_np = face_t_255.squeeze().permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()

    person_name = str(person_name).strip().replace(" ", "_").lower()
    person_dir = os.path.join(DATASET_DIR, person_name)
    os.makedirs(person_dir, exist_ok=True)
    
    # Unikátny timestamp s mikrosekundami
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{timestamp}.jpg"
    filepath = os.path.join(person_dir, filename)
    
    img_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath, img_bgr)
    
    # Premažeme cache galérie, aby nová fotka okamžite platila
    gallery_cache.clear()
    
    count = len([f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    readable = format_display_name(person_name)
    status_msg = f"Fotka uložená.\nProfil: {readable} ({person_name})\nDetekcia tváre: {method}\nSúbor: {filename}\nFotiek v profile: {count}"
    
    return status_msg, face_np

def handle_finetune(progress=gr.Progress()):
    gallery_cache.clear()
    success, msg = run_finetuning(dataset_dir=DATASET_DIR, epochs=15, lr=0.001, progress=progress)
    if success:
        # Skús znovu načítať nový model do pamäte
        bcnn_finetuned = load_finetuned_cnn(DEVICE)
        if bcnn_finetuned is not None:
            models_dict["BenchmarkCNN (Finetuned)"] = bcnn_finetuned
            return msg, gr.update(choices=list(models_dict.keys()), value="BenchmarkCNN (Finetuned)")
    return msg, gr.update()

# Gradio UI Custom CSS pre striktný readonly vzhľad dropdownov (zamedzenie prepisovania textu a blikania kurzora)
custom_css = """
.strict-select input,
.strict-select .secondary-wrap input,
.strict-select div[data-testid="dropdown"] input,
.strict-select input[role="combobox"],
.strict-select input[role="listbox"],
.gr-dropdown input {
    caret-color: transparent !important;
    cursor: pointer !important;
    user-select: none !important;
    -webkit-user-select: none !important;
}
.strict-select,
.strict-select div,
.strict-select .wrap,
.strict-select .wrap-inner,
.strict-select .secondary-wrap,
.gr-dropdown,
.gr-dropdown div {
    cursor: pointer !important;
}
"""

with gr.Blocks(title="Face Recognition & Adversarial Attacks") as app:
    gr.Markdown("# Rozpoznávanie tváre a adversariálne útoky")
    gr.Markdown("Testovanie útokov na modely rozpoznávania tváre a zber vlastných fotiek.")
    
    with gr.Tabs():
        # TAB 1: Testovanie útokov
        with gr.TabItem("Testovanie útokov"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Fotka a nastavenie útoku")
                    tab1_current_img = gr.State(None)

                    with gr.Row():
                        tab1_source_radio = gr.Radio(
                            ["Webkamera", "Nahrať súbor"],
                            value="Webkamera",
                            label="Zdroj fotografie",
                            scale=2
                        )
                        tab1_flip_cam_btn = gr.Button("↔️ Otočiť kameru L/P", variant="secondary", size="sm", scale=1)
                        tab1_retake_btn = gr.Button("Odfoť znova", visible=False, variant="secondary", scale=1)

                    tab1_webcam = gr.Image(
                        sources=["webcam"],
                        streaming=False,
                        type="pil",
                        format="jpeg",
                        webcam_options=gr.WebcamOptions(mirror=False),
                        label="Webkamera (namier tvár a stlač spúšť)",
                        visible=True,
                        height=210
                    )

                    tab1_upload = gr.Image(
                        sources=["upload"],
                        type="pil",
                        format="jpeg",
                        label="Nahraj fotku z disku",
                        visible=False,
                        height=210
                    )

                    tab1_photo_preview = gr.Image(
                        type="pil",
                        format="jpeg",
                        label="Vybraná fotka",
                        interactive=False,
                        visible=False,
                        height=210
                    )

                    tab1_snap_msg = gr.Markdown(visible=False)

                    def format_strategy_info(attack_nm, model_nm, ens_mode):
                        if "BioMesh-SpecPatch" not in str(attack_nm):
                            return ""
                        core_order = ["BenchmarkCNN", "BenchmarkCNN (Finetuned)", "FaceNet", "FaceNet (CASIA)", "ArcFace", "AdaFace"]
                        available_others = [m for m in core_order if m in models_dict and m != model_nm]
                        if "Samostatný" in str(ens_mode):
                            return f"🎯 **Režim: Samostatný model (White-Box)** | BioMesh-SpecPatch (pyramída + 3D mesh + PAD) optimalizuje priamo na modeli **{model_nm}**."
                        elif "Plný" in str(ens_mode):
                            all_avail = [m for m in core_order if m in models_dict]
                            return f"🌐 **Režim: Plný Ensemble (White-Box)** | Optimalizuje spoločný gradient zo všetkých modelov (**{' + '.join(all_avail)}**) a vyhodnocuje sa na **{model_nm}**."
                        else:
                            return f"🛡️ **Režim: Black-Box Transfer** | Optimalizuje na ensembli {len(available_others)} modelov (**{' + '.join(available_others)}**) s trhovými váhami → Testuje oklamanie neznámeho modelu **{model_nm}**."

                    with gr.Row():
                        attack_dropdown = gr.Dropdown(
                            choices=[
                                "BioMesh-SpecPatch (Ensemble + PAD)",
                                "Hybrid Patch (Páska na nos + Pimple Patches)",
                                "Constellation Patch (Pimple Patches)",
                                "Športová páska na nos (Breathe Right)",
                                "Adversarial Patch (Leukoplast)",
                                "FGSM", "PGD", "BIM", "MI-FGSM", "C&W",
                            ], 
                            value="BioMesh-SpecPatch (Ensemble + PAD)", 
                            label="Typ Útoku",
                            filterable=False,
                            elem_classes=["strict-select"],
                            scale=1,
                        )
                        model_dropdown = gr.Dropdown(
                            choices=list(models_dict.keys()), 
                            value="AdaFace" if "AdaFace" in models_dict else "BenchmarkCNN", 
                            label="Cieľový Model (vyhodnotenie)",
                            filterable=False,
                            elem_classes=["strict-select"],
                            scale=1,
                        )

                    ensemble_mode_dropdown = gr.Dropdown(
                        choices=[
                            "Plný Ensemble 4 modelov (White-Box na všetky)",
                            "Ensemble 3 modelov (Black-Box Leave-One-Out – cieľový model vynechaný)",
                            "Samostatný model (White-Box priamo na cieľový model)",
                        ],
                        value="Plný Ensemble 4 modelov (White-Box na všetky)",
                        label="Režim optimalizácie BioMesh-SpecPatch",
                        filterable=False,
                        elem_classes=["strict-select"],
                        visible=True,
                    )

                    model_strategy_info = gr.Markdown(
                        value=format_strategy_info(
                            "BioMesh-SpecPatch (Ensemble + PAD)",
                            "AdaFace" if "AdaFace" in models_dict else "BenchmarkCNN",
                            "Plný Ensemble 4 modelov (White-Box na všetky)"
                        ),
                        visible=True,
                    )

                    with gr.Row():
                        patch_pos_dropdown = gr.Dropdown(
                            choices=[
                                "1. Čierna športová páska (Athlete Black - Breathe Right)",
                                "2. Telová klasická (Tan Fabric - lekárenská)",
                                "3. Priesvitná silikónová (Clear Silicone)",
                                "4. Karbón s logom (Pro Carbon Logo)",
                                "5. Kinesio páska (3 športové farby)",
                                "6. Kozmetické hviezdičky (Starface Style)",
                                "7. Neobmedzený digitálny vzor (Plné RGB)",
                            ],
                            value="1. Čierna športová páska (Athlete Black - Breathe Right)",
                            label="Vzhľad a materiál náplastí (Materiál & Dizajn)",
                            filterable=False,
                            elem_classes=["strict-select"],
                            visible=True
                        )
                        constellation_pattern_dropdown = gr.Dropdown(
                            choices=[
                                "2 nálepky: Obe líca",
                                "3 nálepky: Obe líca + Brada",
                                "1 nálepka: Iba ľavé líce"
                            ],
                            value="2 nálepky: Obe líca",
                            label="Rozmiestnenie nálepiek (pre Hybrid útok)",
                            filterable=False,
                            elem_classes=["strict-select"],
                            visible=True
                        )

                    with gr.Row():
                        strip_preset_dropdown = gr.Dropdown(
                            choices=[
                                "1. Reálna Breathe Right (1:1 / mierka 1.0)",
                                "2. Športová stredná (mierka 1.15)",
                                "3. Zosilnená páska (mierka 1.25)",
                                "4. Robustná široká (mierka 1.35)",
                                "Vlastná páska (posuvník)",
                            ],
                            value="2. Športová stredná (mierka 1.15)",
                            label="Predvoľba pásky na nos",
                            filterable=False,
                            elem_classes=["strict-select"],
                            visible=True
                        )
                        dots_preset_dropdown = gr.Dropdown(
                            choices=[
                                "1. Diskrétne mikro-nálepky (r = 4.5 px)",
                                "2. Stredné hydrokoloidné (r = 5.5 px)",
                                "3. Zosilnené krytie (r = 6.5 px)",
                                "4. Maximálne krytie (r = 7.5 px)",
                                "Vlastné nálepky (posuvník)",
                            ],
                            value="2. Stredné hydrokoloidné (r = 5.5 px)",
                            label="Predvoľba nálepiek na líca",
                            filterable=False,
                            elem_classes=["strict-select"],
                            visible=True
                        )
                        patch_size_dropdown = gr.Dropdown(
                            choices=["Stredný (7x18 px / 22x8 mm)", "Malý (5x14 px / 18x6 mm)", "Veľký (9x22 px / 26x10 mm)"],
                            value="Stredný (7x18 px / 22x8 mm)",
                            label="Veľkosť leukoplastu",
                            filterable=False,
                            elem_classes=["strict-select"],
                            visible=False
                        )
                        hybrid_style_dropdown = gr.Dropdown(
                            choices=[
                                "Vypnutá (plné RGB / podľa štýlu)",
                                "Kinesio 3 farby",
                                "Logo 2 farby (čierna/červená)",
                                "Logo 4 farby",
                                "Logo: hviezda (2 farby)",
                            ],
                            value="Vypnutá (plné RGB / podľa štýlu)",
                            label="Režim vzoru (paleta / logo)",
                            filterable=False,
                            elem_classes=["strict-select"],
                            visible=False
                        )

                    with gr.Row():
                        strip_scale_slider = gr.Slider(
                            minimum=0.7,
                            maximum=1.50,
                            value=1.15,
                            step=0.05,
                            label="Veľkosť pásky na nos (1.0 = reálna Breathe Right)",
                            scale=2,
                            visible=True
                        )
                        patch_radius_slider = gr.Slider(
                            minimum=2.5,
                            maximum=7.5,
                            value=5.5,
                            step=0.5,
                            label="Veľkosť nálepiek na lícach (Polomer px)",
                            scale=2,
                            visible=True
                        )
                        live_occlusion_badge = gr.HTML(
                            value=format_occlusion_badge(2.6, strip_pct=1.3, dots_pct=1.3),
                            scale=1,
                            visible=True
                        )

                    with gr.Group(visible=True) as live_preview_group:
                        gr.Markdown("#### Umiestnenie na tvári (ťahaj prvky myšou | `N▲`=Nos, `L▲`=Ľavé líce vľavo, `P▲`=Pravé líce vpravo)")
                        drag_canvas_html = gr.HTML(elem_id="drag_canvas_container")
                        reset_offsets_btn = gr.Button("Resetovať pozície do stredu", size="sm", variant="secondary")
                        live_preview_image = gr.Image(visible=False)
                        with gr.Group(elem_id="drag_sync_hidden_group"):
                            drag_sync_btn = gr.Button("sync_offsets", elem_id="drag_sync_btn")

                    with gr.Accordion("🎯 Živý AR Zameriavač na webkamere (sleduje tvár pri lepení)", open=False):
                        gr.Markdown(
                            "Zapni kameru nižšie: počas pohybu hlavy ti v reálnom čase vykresľuje presné obrysy, kam nalepiť "
                            "**`[N] NOS ▲`**, **`[L] ĽAVÉ LÍCE ▲`** (vľavo na monitore) a **`[P] PRAVÉ LÍCE ▲`** (vpravo na monitore). "
                            "Šípka **`▲`** ukazuje, ktorá strana nálepky z hárku ide **HORE k očiam**."
                        )
                        ar_latest_pil = gr.State(None)
                        with gr.Row():
                            ar_webcam_stream = gr.Image(
                                sources=["webcam"],
                                streaming=True,
                                type="pil",
                                webcam_options=gr.WebcamOptions(mirror=False),
                                label="Živý vstup z webkamery",
                                height=240,
                            )
                            ar_guide_output = gr.Image(
                                type="numpy",
                                label="AR Šablóna na tvojej tvári (336×336)",
                                interactive=False,
                                height=240,
                            )
                        ar_use_frame_btn = gr.Button("📸 Použiť aktuálny záber zo zameriavača a rozpoznať tvár", variant="primary", size="sm")

                    with gr.Row():
                        attack_btn = gr.Button("Spustiť útok", variant="primary", size="lg")
                        identify_btn = gr.Button("Rozpoznať tvár", variant="secondary", size="lg")

                    with gr.Accordion("Pokročilé parametre", open=False):
                        with gr.Row():
                            epsilon_slider = gr.Slider(minimum=1/255.0, maximum=32/255.0, value=8/255.0, step=1/255.0, label="Epsilon (sila šumu)", visible=False)
                            alpha_slider = gr.Slider(minimum=1/255.0, maximum=10/255.0, value=2/255.0, step=1/255.0, label="Alpha (krok)", visible=False)
                            iter_slider = gr.Slider(minimum=10, maximum=250, value=65, step=5, label="Počet iterácií", visible=True)

                        with gr.Group(visible=True) as offsets_accordion:
                            gr.Markdown("##### Jemný posun prvkov (px)")
                            with gr.Row(visible=True) as strip_offset_row:
                                strip_ox_slider = gr.Slider(-OFFSET_LIMIT_PX, OFFSET_LIMIT_PX, value=0.0, step=0.5, label="Nos / páska X (px)")
                                strip_oy_slider = gr.Slider(-OFFSET_LIMIT_PX, OFFSET_LIMIT_PX, value=0.0, step=0.5, label="Nos / páska Y (px)")
                            with gr.Row(visible=True) as left_offset_row:
                                left_ox_slider = gr.Slider(-OFFSET_LIMIT_PX, OFFSET_LIMIT_PX, value=0.0, step=0.5, label="Ľavé líce X (px)")
                                left_oy_slider = gr.Slider(-OFFSET_LIMIT_PX, OFFSET_LIMIT_PX, value=0.0, step=0.5, label="Ľavé líce Y (px)")
                            with gr.Row(visible=True) as right_offset_row:
                                right_ox_slider = gr.Slider(-OFFSET_LIMIT_PX, OFFSET_LIMIT_PX, value=0.0, step=0.5, label="Pravé líce X (px)")
                                right_oy_slider = gr.Slider(-OFFSET_LIMIT_PX, OFFSET_LIMIT_PX, value=0.0, step=0.5, label="Pravé líce Y (px)")
                            with gr.Row(visible=False) as chin_offset_row:
                                chin_ox_slider = gr.Slider(-OFFSET_LIMIT_PX, OFFSET_LIMIT_PX, value=0.0, step=0.5, label="Brada X (px)")
                                chin_oy_slider = gr.Slider(-OFFSET_LIMIT_PX, OFFSET_LIMIT_PX, value=0.0, step=0.5, label="Brada Y (px)")

                    with gr.Accordion("Uložiť fotku do datasetu", open=False):
                        with gr.Row():
                            tab1_person_name = gr.Dropdown(
                                choices=get_dataset_classes(),
                                value="brandajsky_peter",
                                label="Profil pre uloženie",
                                allow_custom_value=True
                            )
                            tab1_save_btn = gr.Button("Uložiť fotku", variant="secondary")
                        tab1_save_status = gr.Textbox(label="Status uloženia", lines=2)

                    def update_attack_controls(selected_attack, current_model, current_ens_mode):
                        is_bounded = selected_attack in ["FGSM", "PGD", "BIM", "MI-FGSM"]
                        is_step = selected_attack in ["PGD", "BIM", "MI-FGSM"]
                        is_nasal = selected_attack == "Športová páska na nos (Breathe Right)"
                        is_leukoplast = selected_attack == "Adversarial Patch (Leukoplast)"
                        is_const = selected_attack == "Constellation Patch (Pimple Patches)"
                        is_specpatch = "BioMesh-SpecPatch" in str(selected_attack)
                        is_hybrid = ("Hybrid Patch" in str(selected_attack)) or is_specpatch
                        is_physical = is_nasal or is_leukoplast or is_const or is_hybrid
                        is_iter = selected_attack in ["PGD", "BIM", "MI-FGSM", "C&W", "Adversarial Patch (Leukoplast)", "Športová páska na nos (Breathe Right)", "Constellation Patch (Pimple Patches)"] or is_hybrid

                        default_iters = 20
                        if selected_attack == "FGSM":
                            default_iters = 1
                        elif selected_attack in ["PGD", "BIM", "MI-FGSM"]:
                            default_iters = 20
                        elif selected_attack == "C&W":
                            default_iters = 100
                        elif selected_attack in ["Adversarial Patch (Leukoplast)", "Constellation Patch (Pimple Patches)", "Športová páska na nos (Breathe Right)"]:
                            default_iters = 80
                        elif is_hybrid:
                            default_iters = 65

                        show_leuko_size = is_leukoplast
                        show_dots_radius = is_const or is_hybrid
                        show_strip_scale = is_nasal or is_hybrid
                        show_strip_preset = is_nasal or is_hybrid
                        show_dots_preset = is_const or is_hybrid

                        if is_nasal or is_hybrid:
                            pos_update = gr.update(
                                visible=True,
                                choices=[
                                    "1. Čierna športová páska (Athlete Black - Breathe Right)",
                                    "2. Telová klasická (Tan Fabric - lekárenská)",
                                    "3. Priesvitná silikónová (Clear Silicone)",
                                    "4. Karbón s logom (Pro Carbon Logo)",
                                    "5. Kinesio páska (3 športové farby)",
                                    "6. Kozmetické hviezdičky (Starface Style)",
                                    "7. Neobmedzený digitálny vzor (Plné RGB)",
                                ],
                                value="1. Čierna športová páska (Athlete Black - Breathe Right)",
                                label="Vzhľad a materiál náplastí (Materiál & Dizajn)",
                                filterable=False,
                            )
                        elif is_leukoplast:
                            pos_update = gr.update(
                                visible=True,
                                choices=["Koreň nosa (nose_bridge)", "Líce (cheek)", "Čelo (forehead)"],
                                value="Koreň nosa (nose_bridge)",
                                label="Poloha leukoplastu (pre Patch útok)",
                                filterable=False,
                            )
                        else:
                            pos_update = gr.update(visible=False)

                        default_pattern = "2 nálepky: Obe líca" if is_hybrid else "3 body: Nos + Obe líca"
                        if is_hybrid:
                            pattern_update = gr.update(
                                visible=True,
                                choices=["2 nálepky: Obe líca", "3 nálepky: Obe líca + Brada", "1 nálepka: Iba ľavé líce"],
                                value="2 nálepky: Obe líca",
                                label="Rozmiestnenie nálepiek (pre Hybrid útok)",
                                filterable=False,
                            )
                        elif is_const:
                            pattern_update = gr.update(
                                visible=True,
                                choices=[
                                    "3 body: Nos + Obe líca",
                                    "5 bodov: Cluster (Líca + Nos + Brada) [Prah účinnosti]",
                                    "Asymetrické nálepky (Náhodné 3 body)",
                                    "Starface hviezdičky (3 žlté hviezdičky)",
                                    "Adversariálne pehy / Freckles (8 mikro-bodov)",
                                    "4 body: Nos + Líca + Brada (Full)",
                                    "2 body: Iba líca",
                                    "T-Zóna: Čelo + Nos + Brada"
                                ],
                                value="3 body: Nos + Obe líca",
                                label="Vzor mikro-nálepiek / pieh (pre Constellation útok)",
                                filterable=False,
                            )
                        else:
                            pattern_update = gr.update(visible=False)

                        active = get_active_handles_for_attack(selected_attack, default_pattern)
                        info_text = format_strategy_info(selected_attack, current_model, current_ens_mode)

                        return [
                            gr.update(visible=is_bounded),
                            gr.update(visible=is_step),
                            gr.update(visible=is_iter, value=default_iters),
                            pos_update,
                            gr.update(visible=show_leuko_size),
                            pattern_update,
                            gr.update(visible=show_strip_preset),
                            gr.update(visible=show_dots_preset),
                            gr.update(visible=show_dots_radius),
                            gr.update(visible=show_strip_scale),
                            gr.update(visible=False),
                            gr.update(visible=is_specpatch),
                            gr.update(visible=is_specpatch, value=info_text),
                            gr.update(visible=is_physical),
                            gr.update(visible=active["strip"]),
                            gr.update(visible=active["left"]),
                            gr.update(visible=active["right"]),
                            gr.update(visible=active["chin"]),
                            gr.update(visible=is_physical),
                            gr.update(visible=is_physical),
                        ]

                    attack_dropdown.change(
                        fn=update_attack_controls,
                        inputs=[attack_dropdown, model_dropdown, ensemble_mode_dropdown],
                        outputs=[
                            epsilon_slider,
                            alpha_slider,
                            iter_slider,
                            patch_pos_dropdown,
                            patch_size_dropdown,
                            constellation_pattern_dropdown,
                            strip_preset_dropdown,
                            dots_preset_dropdown,
                            patch_radius_slider,
                            strip_scale_slider,
                            hybrid_style_dropdown,
                            ensemble_mode_dropdown,
                            model_strategy_info,
                            offsets_accordion,
                            strip_offset_row,
                            left_offset_row,
                            right_offset_row,
                            chin_offset_row,
                            live_preview_group,
                            live_occlusion_badge,
                        ],
                        show_progress="hidden"
                    )

                    def on_strategy_change(att, mdl, ens):
                        return format_strategy_info(att, mdl, ens)

                    for trigger in [model_dropdown, ensemble_mode_dropdown]:
                        trigger.change(
                            fn=on_strategy_change,
                            inputs=[attack_dropdown, model_dropdown, ensemble_mode_dropdown],
                            outputs=[model_strategy_info],
                            show_progress="hidden",
                        )

                    def on_strip_preset_change(choice, cur_r, img, attack_nm, p_pos, p_sz, c_pat, sox, soy, lox, loy, rox, roy, cox, coy):
                        c_str = str(choice).strip()
                        if c_str.startswith("2.") or "1.15" in c_str:
                            s_val = 1.15
                        elif c_str.startswith("3.") or "1.25" in c_str:
                            s_val = 1.25
                        elif c_str.startswith("4.") or "1.35" in c_str:
                            s_val = 1.35
                        elif c_str.startswith("1.") or "Reálna" in c_str or "1.0" in c_str:
                            s_val = 1.0
                        else:
                            return gr.update(), gr.update(), gr.update(), gr.update()

                        prev_img, drag_html, badge_html = render_live_placement_preview(
                            image=img,
                            attack_name=attack_nm,
                            patch_pos=p_pos,
                            patch_size_choice=p_sz,
                            constellation_pattern=c_pat,
                            patch_radius=float(cur_r or 4.5),
                            strip_scale=s_val,
                            strip_ox=float(sox or 0.0),
                            strip_oy=float(soy or 0.0),
                            left_ox=float(lox or 0.0),
                            left_oy=float(loy or 0.0),
                            right_ox=float(rox or 0.0),
                            right_oy=float(roy or 0.0),
                            chin_ox=float(cox or 0.0),
                            chin_oy=float(coy or 0.0),
                        )
                        return gr.update(value=s_val), prev_img, drag_html, badge_html

                    def on_dots_preset_change(choice, cur_s, img, attack_nm, p_pos, p_sz, c_pat, sox, soy, lox, loy, rox, roy, cox, coy):
                        c_str = str(choice).strip()
                        if c_str.startswith("2.") or "5.5" in c_str:
                            r_val = 5.5
                        elif c_str.startswith("3.") or "6.5" in c_str:
                            r_val = 6.5
                        elif c_str.startswith("4.") or "7.5" in c_str:
                            r_val = 7.5
                        elif c_str.startswith("1.") or "4.5" in c_str:
                            r_val = 4.5
                        else:
                            return gr.update(), gr.update(), gr.update(), gr.update()

                        prev_img, drag_html, badge_html = render_live_placement_preview(
                            image=img,
                            attack_name=attack_nm,
                            patch_pos=p_pos,
                            patch_size_choice=p_sz,
                            constellation_pattern=c_pat,
                            patch_radius=r_val,
                            strip_scale=float(cur_s or 1.0),
                            strip_ox=float(sox or 0.0),
                            strip_oy=float(soy or 0.0),
                            left_ox=float(lox or 0.0),
                            left_oy=float(loy or 0.0),
                            right_ox=float(rox or 0.0),
                            right_oy=float(roy or 0.0),
                            chin_ox=float(cox or 0.0),
                            chin_oy=float(coy or 0.0),
                        )
                        return gr.update(value=r_val), prev_img, drag_html, badge_html

                    strip_preset_dropdown.change(
                        fn=on_strip_preset_change,
                        inputs=[
                            strip_preset_dropdown,
                            patch_radius_slider,
                            tab1_current_img,
                            attack_dropdown,
                            patch_pos_dropdown,
                            patch_size_dropdown,
                            constellation_pattern_dropdown,
                            strip_ox_slider,
                            strip_oy_slider,
                            left_ox_slider,
                            left_oy_slider,
                            right_ox_slider,
                            right_oy_slider,
                            chin_ox_slider,
                            chin_oy_slider,
                        ],
                        outputs=[
                            strip_scale_slider,
                            live_preview_image,
                            drag_canvas_html,
                            live_occlusion_badge,
                        ],
                        show_progress="hidden",
                    )

                    dots_preset_dropdown.change(
                        fn=on_dots_preset_change,
                        inputs=[
                            dots_preset_dropdown,
                            strip_scale_slider,
                            tab1_current_img,
                            attack_dropdown,
                            patch_pos_dropdown,
                            patch_size_dropdown,
                            constellation_pattern_dropdown,
                            strip_ox_slider,
                            strip_oy_slider,
                            left_ox_slider,
                            left_oy_slider,
                            right_ox_slider,
                            right_oy_slider,
                            chin_ox_slider,
                            chin_oy_slider,
                        ],
                        outputs=[
                            patch_radius_slider,
                            live_preview_image,
                            drag_canvas_html,
                            live_occlusion_badge,
                        ],
                        show_progress="hidden",
                    )

                    def update_pattern_controls(attack_nm, pattern):
                        active = get_active_handles_for_attack(attack_nm, pattern)
                        return [
                            gr.update(visible=active["strip"]),
                            gr.update(visible=active["left"]),
                            gr.update(visible=active["right"]),
                            gr.update(visible=active["chin"]),
                        ]

                    constellation_pattern_dropdown.change(
                        fn=update_pattern_controls,
                        inputs=[attack_dropdown, constellation_pattern_dropdown],
                        outputs=[
                            strip_offset_row,
                            left_offset_row,
                            right_offset_row,
                            chin_offset_row,
                        ],
                        show_progress="hidden"
                    )

                    live_preview_inputs = [
                        tab1_current_img,
                        attack_dropdown,
                        patch_pos_dropdown,
                        patch_size_dropdown,
                        constellation_pattern_dropdown,
                        patch_radius_slider,
                        strip_scale_slider,
                        strip_ox_slider,
                        strip_oy_slider,
                        left_ox_slider,
                        left_oy_slider,
                        right_ox_slider,
                        right_oy_slider,
                        chin_ox_slider,
                        chin_oy_slider,
                    ]
                    for comp in [
                        strip_ox_slider,
                        strip_oy_slider,
                        left_ox_slider,
                        left_oy_slider,
                        right_ox_slider,
                        right_oy_slider,
                        chin_ox_slider,
                        chin_oy_slider,
                        patch_radius_slider,
                        strip_scale_slider,
                        patch_pos_dropdown,
                        patch_size_dropdown,
                        constellation_pattern_dropdown,
                        attack_dropdown,
                        tab1_current_img,
                    ]:
                        comp.change(
                            fn=render_live_placement_preview,
                            inputs=live_preview_inputs,
                            outputs=[live_preview_image, drag_canvas_html, live_occlusion_badge],
                            show_progress="hidden",
                        )

                    def handle_drag_sync(img, attack_nm, p_pos, p_sz, c_pat, p_rad, s_scale, strip_ox, strip_oy, left_ox, left_oy, right_ox, right_oy, chin_ox, chin_oy):
                        active = get_active_handles_for_attack(attack_nm, c_pat)
                        sox = _clamp_offset_px(strip_ox) if active["strip"] else 0.0
                        soy = _clamp_offset_px(strip_oy) if active["strip"] else 0.0
                        lox = _clamp_offset_px(left_ox) if active["left"] else 0.0
                        loy = _clamp_offset_px(left_oy) if active["left"] else 0.0
                        rox = _clamp_offset_px(right_ox) if active["right"] else 0.0
                        roy = _clamp_offset_px(right_oy) if active["right"] else 0.0
                        cox = _clamp_offset_px(chin_ox) if active["chin"] else 0.0
                        coy = _clamp_offset_px(chin_oy) if active["chin"] else 0.0

                        prev_img, drag_html, badge_html = render_live_placement_preview(
                            image=img,
                            attack_name=attack_nm,
                            patch_pos=p_pos,
                            patch_size_choice=p_sz,
                            constellation_pattern=c_pat,
                            patch_radius=p_rad,
                            strip_scale=s_scale,
                            strip_ox=sox,
                            strip_oy=soy,
                            left_ox=lox,
                            left_oy=loy,
                            right_ox=rox,
                            right_oy=roy,
                            chin_ox=cox,
                            chin_oy=coy,
                        )
                        return [
                            gr.update(value=sox),
                            gr.update(value=soy),
                            gr.update(value=lox),
                            gr.update(value=loy),
                            gr.update(value=rox),
                            gr.update(value=roy),
                            gr.update(value=cox),
                            gr.update(value=coy),
                            prev_img,
                            drag_html,
                            badge_html,
                        ]

                    def handle_reset_offsets(img, attack_nm, p_pos, p_sz, c_pat, p_rad, s_scale, *rest):
                        prev_img, drag_html, badge_html = render_live_placement_preview(
                            image=img,
                            attack_name=attack_nm,
                            patch_pos=p_pos,
                            patch_size_choice=p_sz,
                            constellation_pattern=c_pat,
                            patch_radius=p_rad,
                            strip_scale=s_scale,
                            strip_ox=0.0,
                            strip_oy=0.0,
                            left_ox=0.0,
                            left_oy=0.0,
                            right_ox=0.0,
                            right_oy=0.0,
                            chin_ox=0.0,
                            chin_oy=0.0,
                        )
                        return [
                            gr.update(value=0.0),
                            gr.update(value=0.0),
                            gr.update(value=0.0),
                            gr.update(value=0.0),
                            gr.update(value=0.0),
                            gr.update(value=0.0),
                            gr.update(value=0.0),
                            gr.update(value=0.0),
                            prev_img,
                            drag_html,
                            badge_html,
                        ]

                    drag_sync_btn.click(
                        fn=handle_drag_sync,
                        inputs=live_preview_inputs,
                        outputs=[
                            strip_ox_slider,
                            strip_oy_slider,
                            left_ox_slider,
                            left_oy_slider,
                            right_ox_slider,
                            right_oy_slider,
                            chin_ox_slider,
                            chin_oy_slider,
                            live_preview_image,
                            drag_canvas_html,
                            live_occlusion_badge,
                        ],
                        js="""(img, attack, p_pos, p_sz, c_pat, p_rad, s_scale, ...rest) => {
                            try {
                                const offs = window.__getFaceDragOffsets ? window.__getFaceDragOffsets() : null;
                                if (offs) {
                                    return [img, attack, p_pos, p_sz, c_pat, p_rad, s_scale, offs.strip_ox, offs.strip_oy, offs.left_ox, offs.left_oy, offs.right_ox, offs.right_oy, offs.chin_ox, offs.chin_oy];
                                }
                            } catch(e) {
                                console.error("drag_sync_btn js error:", e);
                            }
                            return [img, attack, p_pos, p_sz, c_pat, p_rad, s_scale, ...rest];
                        }""",
                        show_progress="hidden",
                    )

                    reset_offsets_btn.click(
                        fn=handle_reset_offsets,
                        inputs=live_preview_inputs,
                        outputs=[
                            strip_ox_slider,
                            strip_oy_slider,
                            left_ox_slider,
                            left_oy_slider,
                            right_ox_slider,
                            right_oy_slider,
                            chin_ox_slider,
                            chin_oy_slider,
                            live_preview_image,
                            drag_canvas_html,
                            live_occlusion_badge,
                        ],
                        js="""(...args) => {
                            try {
                                if (window.__resetFaceDragOffsets) window.__resetFaceDragOffsets();
                            } catch(e) {}
                            return args;
                        }""",
                        show_progress="hidden",
                    )
                
                with gr.Column(scale=2):
                    gr.Markdown("### 2. Výsledok")
                    with gr.Row():
                        orig_identity_output = gr.Textbox(label="Rozpoznanie pôvodnej fotky", lines=4, placeholder="Zobrazí sa po spustení útoku.")
                        adv_identity_output = gr.Textbox(label="Rozpoznanie po útoku", lines=4, placeholder="Zobrazí sa po spustení útoku.")
                    status_output = gr.Textbox(label="Parametre a výsledok útoku", lines=5, placeholder="Vyber model a útok a stlač Spustiť útok.")
                    with gr.Row():
                        orig_face_output = gr.Image(label="Pôvodná tvár (112×112 px)", type="numpy")
                        adv_image_output = gr.Image(label="Tvár po útoku", type="numpy")
                        noise_image_output = gr.Image(label="Rozdiel (zosilnený) / tlačový hárok", type="numpy")
            
        # TAB 2: Zber dát
        with gr.TabItem("Zber dát a Finetuning"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 1. Pridanie tvárí do datasetu")
                    
                    with gr.Row():
                        input_type = gr.Radio(
                            ["Webkamera", "Nahrať súbor"], 
                            value="Webkamera", 
                            label="Zdroj obrázka",
                            scale=2
                        )
                        tab2_flip_cam_btn = gr.Button("↔️ Otočiť kameru L/P", variant="secondary", size="sm", scale=1)
                    
                    # Webkamera (streaming=False pre fotenie spúšťou, mirror=False bez zrkadlenia)
                    collect_webcam = gr.Image(
                        sources=["webcam"], 
                        streaming=False, 
                        type="pil",
                        webcam_options=gr.WebcamOptions(mirror=False),
                        label="Webkamera (odfoť sa a stlač Uložiť fotku)", 
                        visible=True
                    )
                    # Upload (schovaný defaultne)
                    collect_upload = gr.Image(
                        sources=["upload"], 
                        type="pil",
                        label="Nahrať fotku z disku", 
                        visible=False
                    )
                    
                    person_name_input = gr.Dropdown(
                        choices=get_dataset_classes(),
                        value="brandajsky_peter",
                        label="Meno osoby / profil (vyber existujúcu alebo napíš nové meno)",
                        allow_custom_value=True
                    )
                    save_btn = gr.Button("Uložiť fotku do datasetu", variant="primary")
                    
                    save_status = gr.Textbox(label="Status uloženia", lines=3)
                    cropped_preview = gr.Image(label="Náhľad orezanej tváre 112×112 (Čo model uložil)", type="numpy")
                    
                with gr.Column():
                    gr.Markdown("### 2. Dotrénovanie modelu")
                    gr.Markdown("Po pridaní fotiek môžeš dotrénovať model BenchmarkCNN.")
                    finetune_btn = gr.Button("Dotrénovať model", variant="primary")
                    finetune_status = gr.Textbox(label="Status Finetuningu", lines=4)

        # Event handlers
        def toggle_tab1_source(source):
            if source == "Webkamera":
                return (
                    gr.update(visible=True, value=None),
                    gr.update(visible=False, value=None),
                    gr.update(visible=False, value=None),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    None
                )
            else:
                return (
                    gr.update(visible=False, value=None),
                    gr.update(visible=True, value=None),
                    gr.update(visible=False, value=None),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    None
                )

        tab1_source_radio.change(
            fn=toggle_tab1_source,
            inputs=[tab1_source_radio],
            outputs=[tab1_webcam, tab1_upload, tab1_photo_preview, tab1_snap_msg, tab1_retake_btn, tab1_current_img],
            show_progress="hidden"
        )

        # Odfotenie cez webkameru: schová kameru, na jej mieste zobrazí odfotenú fotku
        def on_tab1_webcam_snap(img):
            if img is None:
                return gr.update(), gr.update(), gr.update(), gr.update(), None
            pil = to_pil_image(img)
            arr = np.array(pil) if pil is not None else np.array([])
            mean_val = float(arr.mean()) if arr.size > 0 else 0.0
            print(f"[DEBUG WEBCAM SNAP] size={getattr(pil, 'size', None)}, mean={mean_val:.2f}, min={arr.min() if arr.size>0 else 0}, max={arr.max() if arr.size>0 else 0}")

            if mean_val < 3.0:
                msg = "Záber z webkamery je celý čierny. Skontroluj krytku kamery a či ju nepoužíva iná aplikácia (Teams, Zoom), alebo nahraj fotku zo súboru."
                msg_update = gr.update(value=msg, visible=True)
            else:
                msg_update = gr.update(value="", visible=False)

            return (
                gr.update(visible=False),
                gr.update(value=pil, visible=True),
                msg_update,
                gr.update(visible=True),
                pil
            )

        tab1_webcam.change(
            fn=on_tab1_webcam_snap,
            inputs=[tab1_webcam],
            outputs=[tab1_webcam, tab1_photo_preview, tab1_snap_msg, tab1_retake_btn, tab1_current_img],
            show_progress="hidden"
        )

        # Nahratie fotky zo súboru: schová upload, na jeho mieste zobrazí vybratú fotku
        def on_tab1_upload_select(img):
            if img is None:
                return gr.update(), gr.update(), gr.update(), gr.update(), None
            pil = to_pil_image(img)
            return (
                gr.update(visible=False),
                gr.update(value=pil, visible=True),
                gr.update(value="", visible=False),
                gr.update(visible=True),
                pil
            )

        tab1_upload.change(
            fn=on_tab1_upload_select,
            inputs=[tab1_upload],
            outputs=[tab1_upload, tab1_photo_preview, tab1_snap_msg, tab1_retake_btn, tab1_current_img],
            show_progress="hidden"
        )

        # Tlačidlo: Odfoť sa znova / Zmeniť fotku
        def on_tab1_retake(source):
            if source == "Webkamera":
                return (
                    gr.update(visible=True, value=None),
                    gr.update(visible=False, value=None),
                    gr.update(visible=False, value=None),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    None
                )
            else:
                return (
                    gr.update(visible=False, value=None),
                    gr.update(visible=True, value=None),
                    gr.update(visible=False, value=None),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    None
                )

        tab1_retake_btn.click(
            fn=on_tab1_retake,
            inputs=[tab1_source_radio],
            outputs=[tab1_webcam, tab1_upload, tab1_photo_preview, tab1_snap_msg, tab1_retake_btn, tab1_current_img],
            show_progress="hidden"
        )

        tab1_flip_cam_btn.click(
            fn=None,
            inputs=[],
            outputs=[],
            js="() => { if (window.__toggleWebcamFlip) window.__toggleWebcamFlip(); }"
        )
        tab2_flip_cam_btn.click(
            fn=None,
            inputs=[],
            outputs=[],
            js="() => { if (window.__toggleWebcamFlip) window.__toggleWebcamFlip(); }"
        )

        identify_btn.click(
            fn=run_identify,
            inputs=[tab1_current_img, model_dropdown, ensemble_mode_dropdown],
            outputs=[orig_face_output, adv_image_output, noise_image_output, orig_identity_output, adv_identity_output, status_output]
        )

        ar_webcam_stream.stream(
            fn=render_ar_stream_frame,
            inputs=[
                ar_webcam_stream,
                constellation_pattern_dropdown,
                patch_radius_slider,
                strip_scale_slider,
                strip_ox_slider,
                strip_oy_slider,
                left_ox_slider,
                left_oy_slider,
                right_ox_slider,
                right_oy_slider,
                chin_ox_slider,
                chin_oy_slider,
            ],
            outputs=[ar_guide_output, ar_latest_pil],
            show_progress="hidden",
        )

        def on_ar_use_frame(latest_pil, model_nm, ens_mode):
            if latest_pil is None:
                return gr.update(), gr.update(), None, None, None, None, "Najprv zapni kameru v AR zameriavači.", "", ""
            o_face, a_img, n_img, o_id, a_id, st = run_identify(latest_pil, model_nm, ens_mode)
            return (
                gr.update(value=latest_pil, visible=True),
                gr.update(visible=True),
                latest_pil,
                o_face, a_img, n_img, o_id, a_id, st
            )

        ar_use_frame_btn.click(
            fn=on_ar_use_frame,
            inputs=[ar_latest_pil, model_dropdown, ensemble_mode_dropdown],
            outputs=[
                tab1_photo_preview,
                tab1_retake_btn,
                tab1_current_img,
                orig_face_output,
                adv_image_output,
                noise_image_output,
                orig_identity_output,
                adv_identity_output,
                status_output,
            ],
        )

        attack_btn.click(
            fn=run_attack,
            inputs=[
                tab1_current_img,
                model_dropdown,
                attack_dropdown,
                epsilon_slider,
                alpha_slider,
                iter_slider,
                patch_pos_dropdown,
                patch_size_dropdown,
                constellation_pattern_dropdown,
                patch_radius_slider,
                strip_scale_slider,
                hybrid_style_dropdown,
                ensemble_mode_dropdown,
                strip_ox_slider,
                strip_oy_slider,
                left_ox_slider,
                left_oy_slider,
                right_ox_slider,
                right_oy_slider,
                chin_ox_slider,
                chin_oy_slider,
            ],
            outputs=[orig_face_output, adv_image_output, noise_image_output, orig_identity_output, adv_identity_output, status_output],
            js="""(...args) => {
                try {
                    const offs = window.__getFaceDragOffsets ? window.__getFaceDragOffsets() : null;
                    if (offs) {
                        const n = args.length;
                        if (n >= 18) {
                            args[n - 8] = offs.strip_ox;
                            args[n - 7] = offs.strip_oy;
                            args[n - 6] = offs.left_ox;
                            args[n - 5] = offs.left_oy;
                            args[n - 4] = offs.right_ox;
                            args[n - 3] = offs.right_oy;
                            args[n - 2] = offs.chin_ox;
                            args[n - 1] = offs.chin_oy;
                        }
                    }
                } catch(e) {
                    console.error("attack_btn js error:", e);
                }
                return args;
            }"""
        )

        def save_from_tab1(img, name):
            status_msg, face_np = save_image_to_dataset(img, name)
            classes = get_dataset_classes()
            return status_msg, gr.update(choices=classes, value=name), gr.update(choices=classes)

        tab1_save_btn.click(
            fn=save_from_tab1,
            inputs=[tab1_current_img, tab1_person_name],
            outputs=[tab1_save_status, tab1_person_name, person_name_input]
        )

        # Tab 2 prepínanie zdroja
        def toggle_input(choice):
            if choice == "Webkamera":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)

        input_type.change(
            fn=toggle_input, 
            inputs=[input_type], 
            outputs=[collect_webcam, collect_upload]
        )
        
        # Tab 2 ukladanie fotky
        def unified_save(web_img, up_img, name, mode):
            img = web_img if mode == "Webkamera" else up_img
            status_msg, face_np = save_image_to_dataset(img, name)
            classes = get_dataset_classes()
            return status_msg, face_np, gr.update(choices=classes, value=name), gr.update(choices=classes)

        save_btn.click(
            fn=unified_save,
            inputs=[collect_webcam, collect_upload, person_name_input, input_type],
            outputs=[save_status, cropped_preview, person_name_input, tab1_person_name]
        )
        
        # Tab 2 finetuning
        finetune_btn.click(
            fn=handle_finetune,
            inputs=[],
            outputs=[finetune_status, model_dropdown]
        )

if __name__ == "__main__":
    # Vytvorenie zložky na dáta ak neexistuje
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    # Spustenie aplikácie na porte 7860 (default)
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, head=HEAD_SCRIPTS, css=custom_css)
