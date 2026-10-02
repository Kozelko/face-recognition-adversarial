import os
import sys
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
    ctypes.CDLL('/home/kozel/miniconda/envs/cnn-benchmark/lib/python3.11/site-packages/opencv_contrib_python.libs/libpng16-ef62451c.so.16.44.0', mode=ctypes.RTLD_GLOBAL)
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
from attacks.nasal_strip_patch import nasal_strip_attack, generate_printable_nasal_strip_sheet, create_nasal_strip_mask

# Globálne nastavenia
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    global models_dict
    if not models_dict:
        print("Načítavam modely do pamäte...")
        models_dict["FaceNet"] = FaceNetWrapper(device=DEVICE)
        models_dict["ArcFace"] = ArcFaceWrapper(device=DEVICE)
        models_dict["AdaFace"] = AdaFaceWrapper(device=DEVICE)
        bcnn = load_benchmark_cnn(DEVICE)
        if bcnn is not None:
            models_dict["BenchmarkCNN"] = bcnn
        
        bcnn_finetuned = load_finetuned_cnn(DEVICE)
        if bcnn_finetuned is not None:
            models_dict["BenchmarkCNN (Finetuned)"] = bcnn_finetuned
            print("Finetuned BenchmarkCNN úspešne načítaný.")
            
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
    lines = []
    
    clf = None
    if hasattr(model, "predict_identity"):
        clf = model.predict_identity(img_tensor)
        
    gallery = get_model_gallery(model, model_name)
    best_person = "Neznáma osoba"
    best_sim = -1.0
    scores = {}
    with torch.no_grad():
        emb = model(img_tensor)
        for person, cent in gallery.items():
            sim = F.cosine_similarity(emb, cent).item()
            readable = format_display_name(person)
            scores[readable] = sim
            if sim > best_sim:
                best_sim = sim
                best_person = readable
                
    if not is_attack:
        lines.append("🟢 STAV: ČISTÁ TVÁR (Pôvodná fotografia)")
        lines.append("─────────────────────────────────────────────")
        
        if clf:
            label = format_display_name(clf["raw_label"])
            conf = clf["confidence"] * 100.0
            lines.append(f"👤 Kto si podľa modelu: {label} (Istota siete: {conf:.1f} %)")
            sorted_probs = sorted(clf["probabilities"].items(), key=lambda x: x[1], reverse=True)
            p_str = " | ".join([f"{format_short_name(k)}: {v*100:.1f}%" for k, v in sorted_probs])
            lines.append(f"   ↳ Pravdepodobnosti tried: [{p_str}]")
            
        if best_sim >= 0.50:
            lines.append(f"🎯 Databáza (1:N): Zhoda s profilom '{best_person}'")
            lines.append(f"   ↳ Podobnosť: {best_sim:.4f} (Limit na overenie je ≥ 0.5000) ✅ VSTUP POVOLENÝ")
        else:
            lines.append(f"❓ Databáza (1:N): Neznáma tvár (Max. podobnosť: {best_sim:.4f} < 0.50) ❌ VSTUP ZAMIETNUTÝ")
        
    else:
        sim_val = sim_1to1 if sim_1to1 is not None else 1.0
        is_dodging = sim_val < 0.50
        
        if is_dodging:
            lines.append("🚨 VÝSLEDOK: BIOMETRIA OKLAMANÁ! (Útok bol úspešný)")
        else:
            lines.append("⚠️ VÝSLEDOK: BIOMETRIA ODOLALA (Útok zlyhal / nebol dosť silný)")
        lines.append("─────────────────────────────────────────────")
        
        lines.append(f"📉 1:1 Zhoda s čistou tvárou: {sim_val:.4f} (z pôvodných 1.0000)")
        if is_dodging:
            lines.append("   ↳ ✅ ÚSPECH: Zhoda padla POD prah 0.5000 ➔ Systém vyhlásil cudzinca!")
        else:
            lines.append("   ↳ ❌ ZLYHANIE: Zhoda zostala NAD prahom 0.5000 ➔ Systém stále spoznal tvár.")
            
        if clf:
            label = format_display_name(clf["raw_label"])
            conf = clf["confidence"] * 100.0
            is_peter = "Peter" in label
            if not is_peter:
                lines.append(f"🎭 ZÁMENA IDENTITY: Model si teraz myslí, že ide o '{label}' (Istota: {conf:.1f} %)")
            else:
                lines.append(f"👤 Klasifikátor: Model tipuje ({label}), istota klesla na {conf:.1f} %")
                
            sorted_probs = sorted(clf["probabilities"].items(), key=lambda x: x[1], reverse=True)
            p_str = " | ".join([f"{format_short_name(k)}: {v*100:.1f}%" for k, v in sorted_probs])
            lines.append(f"   ↳ Pravdepodobnosti tried: [{p_str}]")
            
        if best_sim < 0.50:
            lines.append(f"❓ Databáza (1:N): STRATA IDENTITY (Podobnosť s profilom klesla na {best_sim:.4f} < 0.50 ❌)")
        else:
            lines.append(f"🎯 Databáza (1:N): Zhoda s '{best_person}' ({best_sim:.4f} ≥ 0.50)")
            
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
        return None, None, "⚠️ Žiadny obrázok. Nahraj fotku alebo sa odfoť cez webkameru."
        
    # Kontrola čierneho/prázdneho záberu
    arr_check = np.array(pil_img)
    if arr_check.mean() < 3.0:
        return None, None, "⚠️ Webkamera odoslala čierny/prázdny záber! Skontroluj, či nemáš zakrytú webkameru fyzickou krytkou a stlač spúšť kamery 📷 znova."
        
    face_t_255, method = crop_face_from_image(pil_img)
    if face_t_255 is None:
        return None, None, "❌ Nepodarilo sa nájsť tvár na obrázku."
        
    t = (face_t_255.float() / 255.0 - 0.5) / 0.5
    if len(t.shape) == 3:
        t = t.unsqueeze(0)
    return t.to(DEVICE), method, None

def run_identify(image, model_name):
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
    
    status = f"🔍 Test biometrického rozpoznávania identity (BEZ ÚTOKU)\n"
    status += f"Cieľový model: {model_name} | Detekcia: {method} (112×112 px)\n"
    status += "Tvár je čistá. Pre otestovanie oklamania biometrie zvoľ útok a klikni na '⚡ Spustiť útok'."
    
    adv_id = "— Tvár je čistá (bez útoku). Pre otestovanie oklamania biometrie klikni na tlačidlo '⚡ Spustiť útok' vľavo. —"
    
    return face_np, face_np, None, clean_id, adv_id, status

# Interaktívny náhľad umiestnenia nálepiek & Drag and Drop plátno
_live_preview_cache = {
    "cached_img_id": None,
    "face_np": None,
    "landmarks": None,
    "mesh_info": None,
}

def extract_base_patch_coords(landmarks, mesh):
    if landmarks is None:
        return None
    lx, ly = landmarks[0]
    rx, ry = landmarks[1]
    nx, ny = landmarks[2]

    if mesh and "points_3d" in mesh and len(mesh["points_3d"]) > 197:
        pts = mesh["points_3d"]
        p195, p197 = pts[195], pts[197]
        strip_x = float(p195[0]) * 2.0
        strip_y = float(p195[1] * 0.70 + p197[1] * 0.30) * 2.0
    else:
        strip_x = ((lx + rx) * 0.5) * 2.0
        strip_y = ((ly + ry) * 0.5 + (ny - (ly + ry) * 0.5) * 0.42) * 2.0

    if mesh and "points_3d" in mesh and len(mesh["points_3d"]) > 205:
        left_x = float(mesh["points_3d"][205][0]) * 2.0
        left_y = float(mesh["points_3d"][205][1]) * 2.0
    else:
        left_x = (lx + (nx - lx) * 0.14) * 2.0
        left_y = (ly + (ny - ly) * 0.42) * 2.0

    if mesh and "points_3d" in mesh and len(mesh["points_3d"]) > 425:
        right_x = float(mesh["points_3d"][425][0]) * 2.0
        right_y = float(mesh["points_3d"][425][1]) * 2.0
    else:
        right_x = (rx - (rx - nx) * 0.14) * 2.0
        right_y = (ry + (ny - ry) * 0.42) * 2.0

    if mesh and "points_3d" in mesh and len(mesh["points_3d"]) > 175:
        chin_x = float(mesh["points_3d"][175][0]) * 2.0
        chin_y = float(mesh["points_3d"][175][1]) * 2.0
    else:
        chin_x = ((lx + rx) * 0.5) * 2.0
        chin_y = (ny + (ny - ly) * 1.50) * 2.0

    return {
        "strip": {"x": round(strip_x, 1), "y": round(strip_y, 1), "label": "👃 Páska na nos / Patch", "color": "#00e5ff", "type": "strip"},
        "left": {"x": round(left_x, 1), "y": round(left_y, 1), "label": "🟣 Ľavé líce", "color": "#d500f9", "type": "dot"},
        "right": {"x": round(right_x, 1), "y": round(right_y, 1), "label": "🟣 Pravé líce", "color": "#d500f9", "type": "dot"},
        "chin": {"x": round(chin_x, 1), "y": round(chin_y, 1), "label": "🟠 Brada", "color": "#ff9100", "type": "dot"},
    }

def get_active_handles_for_attack(attack_name, constellation_pattern):
    is_nasal = attack_name == "Športová páska na nos (Breathe Right)"
    is_leukoplast = attack_name == "Adversarial Patch (Leukoplast)"
    is_const = attack_name == "Constellation Patch (Pimple Patches)"
    is_hybrid = "Hybrid Patch" in attack_name

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
</style>
<script>
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

    function getOff(k, axis) {
        if (!state.handles || !state.handles[k] || !state.handles[k].active || !state.base[k]) return 0.0;
        const d = (state.handles[k][axis] - state.base[k][axis]) / 2.0;
        return Math.round(d * 2.0) / 2.0;
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

    for (const k of Object.keys(base)) {
        state.handles[k] = {
            x: Math.max(12, Math.min(212, base[k].x + (curOffsets[k + '_ox'] || 0) * 2)),
            y: Math.max(12, Math.min(212, base[k].y + (curOffsets[k + '_oy'] || 0) * 2)),
            label: base[k].label,
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
                ctx.fillStyle = isHover ? "rgba(0, 229, 255, 0.35)" : "rgba(0, 229, 255, 0.20)";
                ctx.strokeStyle = h.color;
                ctx.lineWidth = isHover ? 2.5 : 1.5;
                ctx.shadowColor = h.color;
                ctx.shadowBlur = isHover ? 10 : 4;

                const sw = 48, sh = 14, sr = 4;
                ctx.beginPath();
                ctx.roundRect(h.x - sw/2, h.y - sh/2, sw, sh, sr);
                ctx.fill();
                ctx.stroke();

                ctx.beginPath();
                ctx.arc(h.x, h.y, isHover ? 5.5 : 4, 0, Math.PI * 2);
                ctx.fillStyle = "#ffffff";
                ctx.fill();
                ctx.strokeStyle = h.color;
                ctx.lineWidth = 1.5;
                ctx.stroke();
            } else {
                ctx.fillStyle = isHover ? "rgba(213, 0, 249, 0.38)" : "rgba(213, 0, 249, 0.22)";
                ctx.strokeStyle = h.color;
                ctx.lineWidth = isHover ? 2.5 : 1.5;
                ctx.shadowColor = h.color;
                ctx.shadowBlur = isHover ? 10 : 4;

                const pr = isHover ? 13 : 11;
                ctx.beginPath();
                ctx.arc(h.x, h.y, pr, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();

                ctx.beginPath();
                ctx.arc(h.x, h.y, isHover ? 4.5 : 3.5, 0, Math.PI * 2);
                ctx.fillStyle = "#ffffff";
                ctx.fill();
                ctx.strokeStyle = h.color;
                ctx.lineWidth = 1.5;
                ctx.stroke();
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
            h.x = Math.max(12, Math.min(212, p.x + state.dragOffset.x));
            h.y = Math.max(12, Math.min(212, p.y + state.dragOffset.y));
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
                    tooltip.innerText = "👆 Chyť prvok myšou a posuň po tvári";
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
          <span>📷 Nahraj fotku alebo sa odfoť pre aktiváciu Drag & Drop plátna</span>
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
        👆 Chyť prvok myšou a posuň po tvári
      </div>
    </div>
    """

def render_live_placement_preview(
    image,
    attack_name,
    patch_pos,
    patch_size_choice,
    constellation_pattern,
    patch_radius,
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
        return None, build_drag_canvas_html(None, None, None, None, attack_name)

    try:
        img_id = (id(image), getattr(image, "size", None))
        if img_id != _live_preview_cache["cached_img_id"] or _live_preview_cache["face_np"] is None:
            face_tensor, method, err = get_face_tensor_from_image(image)
            if face_tensor is None:
                return None, build_drag_canvas_html(None, None, None, None, attack_name)
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
            return None, build_drag_canvas_html(None, None, None, None, attack_name)

        canvas = face_np.copy()
        sox = float(strip_ox) if strip_ox is not None else 0.0
        soy = float(strip_oy) if strip_oy is not None else 0.0
        lox = float(left_ox) if left_ox is not None else 0.0
        loy = float(left_oy) if left_oy is not None else 0.0
        rox = float(right_ox) if right_ox is not None else 0.0
        roy = float(right_oy) if right_oy is not None else 0.0
        cox = float(chin_ox) if chin_ox is not None else 0.0
        coy = float(chin_oy) if chin_oy is not None else 0.0

        # Športová páska na nos (Breathe Right)
        if attack_name == "Športová páska na nos (Breathe Right)":
            if mesh and "points_3d" in mesh and len(mesh["points_3d"]) > 197:
                pts = mesh["points_3d"]
                p195, p197 = pts[195], pts[197]
                cx, cy = float(p195[0]), float(p195[1] * 0.70 + p197[1] * 0.30)
            else:
                lx, ly = lms[0]
                rx, ry = lms[1]
                nx, ny = lms[2]
                cx = (lx + rx) * 0.5
                cy = (ly + ry) * 0.5 + (ny - (ly + ry) * 0.5) * 0.42

            cx = float(np.clip(cx + sox, 10.0, 102.0))
            cy = float(np.clip(cy + soy, 10.0, 102.0))
            dx = lms[1][0] - lms[0][0]
            dy = lms[1][1] - lms[0][1]
            roll = math.atan2(dy, dx)
            eye_dist = math.sqrt(dx**2 + dy**2)
            base_w = max(16.0, min(24.0, eye_dist * 0.45))
            yaw = mesh.get("yaw", 0.0) if mesh else 0.0

            if mesh and "points_3d" in mesh and len(mesh["points_3d"]) > 168:
                pts = mesh["points_3d"]
                p_bridge = pts[168] if len(pts) > 168 else pts[6]
                p_tip = pts[1] if len(pts) > 1 else pts[4]
                nv_x = p_tip[0] - p_bridge[0]
                nv_y = p_tip[1] - p_bridge[1]
                if abs(nv_x) + abs(nv_y) > 1e-4:
                    strip_angle = math.atan2(nv_y, nv_x) - math.pi * 0.5
                else:
                    strip_angle = roll
                y_rel = (cy - 50.0) / 20.0
                base_w = base_w * (1.0 + float(np.clip(y_rel * 0.15, -0.15, 0.20)))
            else:
                strip_angle = roll

            base_hc = base_w * 0.23
            base_hw = base_w * 0.29

            mask, relief, _, _ = create_nasal_strip_mask(112, (cy, cx), base_w, base_hc, base_hw, strip_angle, yaw)
            m_np = mask.squeeze().cpu().numpy()
            r_np = relief.squeeze().cpu().numpy()

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
            for s in specs:
                m, d, _, _ = create_elliptical_patch_mask(112, s["center"], s["rx"], s["ry"], s["angle"], shape=s.get("shape", "circle"))
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

        # Adversarial Patch (Leukoplast)
        elif attack_name == "Adversarial Patch (Leukoplast)":
            pos_map = {"Koreň nosa (nose_bridge)": "nose_bridge", "Líce (cheek)": "cheek", "Čelo (forehead)": "forehead"}
            size_map = {"Stredný (7x18 px / 22x8 mm)": (7, 18), "Malý (5x14 px / 18x6 mm)": (5, 14), "Veľký (9x22 px / 26x10 mm)": (9, 22)}
            c_pos = pos_map.get(patch_pos, "nose_bridge")
            c_sz = size_map.get(patch_size_choice, (7, 18))
            mask, relief, _ = get_patch_geometry(c_pos, 112, c_sz, lms, mesh, sox, soy)
            m_np = mask.squeeze().cpu().numpy()
            r_np = relief.squeeze().cpu().numpy()
            col = np.array([215, 172, 128], dtype=np.float32)
            for c in range(3):
                canvas[:, :, c] = np.clip((1.0 - m_np * 0.92) * canvas[:, :, c] + m_np * 0.92 * col[c] * r_np, 0, 255).astype(np.uint8)

        # Hybrid Patch (Páska na nos + Pimple Patches)
        elif "Hybrid Patch" in attack_name:
            dot_pat = "cheeks_chin" if "3 nálepky" in str(constellation_pattern) else ("1_cheek" if "1 nálepka" in str(constellation_pattern) else "2_cheeks")
            f_mask, f_relief, boxes = get_hybrid_geometry(
                landmarks=lms,
                mesh_info=mesh,
                img_size=112,
                dot_radius=float(patch_radius) if patch_radius else 4.5,
                dot_pattern=dot_pat,
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

        res_img = cv2.resize(canvas, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        base_coords = extract_base_patch_coords(lms, mesh)
        active_handles = get_active_handles_for_attack(attack_name, constellation_pattern)
        offsets_dict = {
            "strip_ox": sox, "strip_oy": soy,
            "left_ox": lox, "left_oy": loy,
            "right_ox": rox, "right_oy": roy,
            "chin_ox": cox, "chin_oy": coy,
        }
        drag_html = build_drag_canvas_html(res_img, base_coords, active_handles, offsets_dict, attack_name)
        return res_img, drag_html
    except Exception:
        return None, build_drag_canvas_html(None, None, None, None, attack_name)

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
    constellation_pattern="3 body: Nos + Obe líca",
    patch_radius=4.5,
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
        return None, None, None, "", "", "⚠️ Prosím, najprv nahrajte alebo odfoťte obrázok."
    
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

    print(f"🚀 [RUN ATTACK] {attack_name} na modeli {model_name} | Posuny: Páska=({sox}, {soy}), Ľavé=({lox}, {loy}), Pravé=({rox}, {roy}), Brada=({cox}, {coy})")

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
        style_map = {
            "Farebný vzor (RGB Adversarial Art - 100% prelomenie)": "adversarial_rgb",
            "Čierna športová (Athlete Black)": "athlete_black",
            "Karbón s logom (Pro Carbon Logo)": "pro_carbon_logo",
            "Telová klasická (Tan Fabric)": "tan_fabric",
            "Priesvitná silikónová (Clear)": "clear_silicone",
        }
        chosen_style = style_map.get(patch_position, "adversarial_rgb" if ("rgb" in str(patch_position).lower() or "farebn" in str(patch_position).lower()) else "athlete_black")
        adv_tensor, patch_crop, nasal_info = nasal_strip_attack(
            model=model,
            image=img_tensor,
            style=chosen_style,
            width_scale=1.0,
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
    elif "Hybrid Patch" in attack_name:
        h_iters = int(num_iter)
        if "3 nálepky" in str(constellation_pattern):
            dot_pat = "cheeks_chin"
        elif "1 nálepka" in str(constellation_pattern):
            dot_pat = "1_cheek"
        else:
            dot_pat = "2_cheeks"

        strip_style_map = {
            "Farebný vzor (RGB Adversarial Art - 100% prelomenie)": "adversarial_rgb",
            "Čierna športová (Athlete Black)": "athlete_black",
            "Karbón s logom (Pro Carbon Logo)": "pro_carbon_logo",
            "Telová klasická (Tan Fabric)": "tan_fabric",
            "Priesvitná silikónová (Clear)": "clear_silicone",
        }
        chosen_strip_style = strip_style_map.get(patch_position, "adversarial_rgb" if ("rgb" in str(patch_position).lower() or "farebn" in str(patch_position).lower()) else "athlete_black")

        adv_tensor, printable_sheet, hybrid_info = hybrid_patch_attack(
            model=model,
            image=img_tensor,
            strip_style=chosen_strip_style,
            dot_radius=float(patch_radius),
            dot_pattern=dot_pat,
            num_iter=h_iters,
            eot=True,
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
        
    # Adversariálny embedding
    with torch.no_grad():
        adv_emb = model(adv_tensor)
        
    similarity = F.cosine_similarity(orig_emb, adv_emb).item()
    
    # Konverzia späť na obrázky pre zobrazenie
    adv_img_np = denormalize(adv_tensor).squeeze().permute(1, 2, 0).cpu().numpy()
    
    if patch_info is not None:
        patch_np = ((patch_crop.detach().cpu().permute(1, 2, 0).numpy() + 1.0) * 0.5 * 255.0).clip(0, 255).astype("uint8")
        diff_np = cv2.resize(patch_np, (112, 112), interpolation=cv2.INTER_NEAREST)

        occ_pct = patch_info.get("face_occlusion_pct", 2.5)
        status_text = f"Metóda: Adversariálny leukoplast 2.0 ({patch_position})\n"
        status_text += f"Oklúzia tváre: iba {occ_pct:.1f} % | 1:1 Cosine Similarity: {similarity:.4f} (z pôvodných 1.0000)\n"
        status_text += f"📍 Aplikovaný posun: Leukoplast=({sox:+.1f}, {soy:+.1f}) px\n"
        if similarity < 0.5:
            status_text += "✅ Úspech: Model pod vplyvom leukoplastu stratil identitu tváre (zhoda klesla pod 50 %)."
        else:
            status_text += "⚠️ Útok zatiaľ neprekonal prah 0.50 (odporúča sa zvýšiť počet iterácií na 80-120)."
    elif nasal_info is not None:
        patch_np = ((patch_crop.detach().cpu().permute(1, 2, 0).numpy() + 1.0) * 0.5 * 255.0).clip(0, 255).astype("uint8")
        diff_np = cv2.resize(patch_np, (112, 112), interpolation=cv2.INTER_NEAREST)

        occ_pct = nasal_info.get("face_occlusion_pct", 2.4)
        style_name = nasal_info.get("style", "athlete_black")
        status_text = f"Metóda: Športová nosná páska 3D (Breathe Right - {style_name})\n"
        status_text += f"Oklúzia tváre: iba {occ_pct:.1f} % | 1:1 Cosine Similarity: {similarity:.4f} (z pôvodných 1.0000)\n"
        status_text += f"📍 Aplikovaný posun: Páska=({sox:+.1f}, {soy:+.1f}) px\n"
        if similarity < 0.5:
            status_text += "✅ Úspech: Model stratil identitu tváre pod vplyvom 3D nosnej pásky (zhoda klesla pod 50 %)."
        else:
            status_text += "⚠️ Útok zatiaľ neprekonal prah 0.50 (odporúča sa 80-120 iterácií pre ArcFace)."
    elif constellation_info is not None:
        patch_np = ((patch_crop.detach().cpu().permute(1, 2, 0).numpy() + 1.0) * 0.5 * 255.0).clip(0, 255).astype("uint8")
        diff_np = cv2.resize(patch_np, (112, 112), interpolation=cv2.INTER_NEAREST)

        occ_pct = constellation_info.get("face_occlusion_pct", 3.0)
        rad_px = constellation_info.get("radius_px", patch_radius)
        if "freckle" in constellation_pattern.lower():
            pat_label = "Adversariálne pehy / Freckles 3D"
        elif "star" in constellation_pattern.lower():
            pat_label = "Starface kozmetické hviezdičky 3D"
        else:
            pat_label = f"Constellation Patch 3D ({constellation_pattern})"
        status_text = f"Metóda: {pat_label}\n"
        status_text += f"Polomer bodu: {rad_px} px | Oklúzia tváre: iba {occ_pct:.1f} % | 1:1 Cosine Similarity: {similarity:.4f}\n"
        active = get_active_handles_for_attack(attack_name, constellation_pattern)
        off_parts = []
        if active["strip"]:
            off_parts.append(f"Nos=({sox:+.1f}, {soy:+.1f})")
        if active["left"]:
            off_parts.append(f"Ľavé líce=({lox:+.1f}, {loy:+.1f})")
        if active["right"]:
            off_parts.append(f"Pravé líce=({rox:+.1f}, {roy:+.1f})")
        if active["chin"]:
            off_parts.append(f"Brada=({cox:+.1f}, {coy:+.1f})")
        if off_parts:
            status_text += f"📍 Aplikované posuny: {' | '.join(off_parts)}\n"
        if similarity < 0.5:
            status_text += "✅ Úspech: Model stratil identitu tváre pri minimálnej oklúzii (zhoda klesla pod 50 %)."
        else:
            status_text += "⚠️ Útok zatiaľ neprekonal prah 0.50 (skús viac iterácií alebo 4 body)."
    elif hybrid_info is not None:
        diff_np = cv2.cvtColor(printable_sheet, cv2.COLOR_BGR2RGB)
        occ_pct = hybrid_info.get("face_occlusion_pct", 4.37)
        status_text = f"Metóda: Hybrid Patch (Leukoplast na nose + Nálepky na lícach)\n"
        status_text += f"Oklúzia tváre: iba {occ_pct:.1f} % | 1:1 Cosine Similarity: {similarity:.4f} (z pôvodných 1.0000)\n"
        active = get_active_handles_for_attack(attack_name, constellation_pattern)
        off_parts = []
        if active["strip"]:
            off_parts.append(f"Páska=({sox:+.1f}, {soy:+.1f})")
        if active["left"]:
            off_parts.append(f"Ľavé líce=({lox:+.1f}, {loy:+.1f})")
        if active["right"]:
            off_parts.append(f"Pravé líce=({rox:+.1f}, {roy:+.1f})")
        if active["chin"]:
            off_parts.append(f"Brada=({cox:+.1f}, {coy:+.1f})")
        if off_parts:
            status_text += f"📍 Aplikované posuny: {' | '.join(off_parts)}\n"
        if similarity < 0.5:
            status_text += "✅ Úspech: Model stratil identitu tváre (vpravo je vygenerovaný 300 DPI tlačový hárok)."
        else:
            status_text += "⚠️ Útok zatiaľ neprekonal prah 0.50 (odporúča sa 150-200 iterácií pre ArcFace)."
    else:
        # Zvýraznený šum pre klasické digitálne útoky (uint8 0..255)
        diff_np = (adv_tensor - img_tensor).squeeze().permute(1, 2, 0).cpu().numpy()
        diff_np = ((diff_np * 10 + 0.5).clip(0, 1) * 255.0).astype(np.uint8)
        
        status_text = f"Metóda: Digitálny útok {attack_name}\n"
        status_text += f"1:1 Cosine Similarity: {similarity:.4f} (Prah úspechu je < 0.5000)\n"
        if similarity < 0.5:
            status_text += "✅ Úspech: Model bol oklamaný digitálnym šumom (zhoda klesla pod 50 %)."
        else:
            status_text += "⚠️ Útok zlyhal: Podobnosť zostala nad prahom 50 %."
        
    # Vyhodnotenie identity po útoku
    adv_id = predict_person_identity(model, model_name, adv_tensor, is_attack=True, sim_1to1=similarity)
    
    # Pôvodná tvár a adversariálna tvár v uint8 [0, 255]
    orig_face_np = (denormalize(img_tensor).squeeze().permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    adv_img_np = (denormalize(adv_tensor).squeeze().permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        
    return orig_face_np, adv_img_np, diff_np, clean_id, adv_id, status_text

# Funkcie pre záložku 2: Zber dát
def save_image_to_dataset(image, person_name):
    if image is None:
        return "⚠️ Žiadny obrázok na uloženie. Najprv stlač spúšť kamery 📷 (alebo nahraj fotku).", None
    if not person_name or not str(person_name).strip():
        return "⚠️ Prosím, zadaj meno osoby (alebo vyber z existujúcich).", None
    
    pil_img = to_pil_image(image)
    if pil_img is None:
        return "⚠️ Nepodarilo sa spracovať obrázok.", None
        
    arr_check = np.array(pil_img)
    if arr_check.mean() < 3.0:
        return "⚠️ Webkamera odoslala čierny/prázdny záber! Skontroluj krytku kamery a odfoť sa znova.", None
        
    face_t_255, method = crop_face_from_image(pil_img)
    if face_t_255 is None:
        return "❌ Na fotke sa nepodarilo nájsť tvár!", None
        
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
    status_msg = f"✅ Tvár úspešne uložená!\n👤 Profil: {readable} ({person_name})\n🔍 Detekcia: {method}\n📁 Súbor: {filename}\n📊 Celkovo fotiek v profile: {count}"
    
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

# Gradio UI
with gr.Blocks(title="Face Recognition & Adversarial Attacks") as app:
    gr.Markdown("# Face Recognition & Adversarial Attacks Platform")
    gr.Markdown("Prototyp pre testovanie adversariálnych útokov a zber dát.")
    
    with gr.Tabs():
        # TAB 1: Testovanie útokov
        with gr.TabItem("Testovanie útokov"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Krok: Fotografia a Nastavenie útoku")
                    tab1_current_img = gr.State(None)

                    with gr.Row():
                        tab1_source_radio = gr.Radio(
                            ["Webkamera", "Nahrať súbor"],
                            value="Webkamera",
                            label="Zdroj fotografie",
                            scale=2
                        )
                        tab1_retake_btn = gr.Button("🔄 Odfoť znova", visible=False, variant="secondary", scale=1)

                    tab1_webcam = gr.Image(
                        sources=["webcam"],
                        streaming=False,
                        type="pil",
                        format="jpeg",
                        webcam_options=gr.WebcamOptions(mirror=False),
                        label="Webkamera (1. Namier tvár, 2. Klikni na spúšť 📷)",
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
                        label="📷 Tvoja fotografia (Pripravená na test)",
                        interactive=False,
                        visible=False,
                        height=210
                    )

                    tab1_snap_msg = gr.Markdown(visible=False)

                    with gr.Row():
                        model_dropdown = gr.Dropdown(
                            choices=list(models_dict.keys()), 
                            value="BenchmarkCNN", 
                            label="Cieľový Model"
                        )
                        attack_dropdown = gr.Dropdown(
                            choices=[
                                "FGSM", "PGD", "BIM", "MI-FGSM", "C&W",
                                "Adversarial Patch (Leukoplast)",
                                "Športová páska na nos (Breathe Right)",
                                "Constellation Patch (Pimple Patches)",
                                "Hybrid Patch (Páska na nos + Pimple Patches)"
                            ], 
                            value="PGD", 
                            label="Typ Útoku"
                        )

                    with gr.Row():
                        patch_pos_dropdown = gr.Dropdown(
                            choices=["Koreň nosa (nose_bridge)", "Líce (cheek)", "Čelo (forehead)"],
                            value="Koreň nosa (nose_bridge)",
                            label="Poloha / Štýl pásky",
                            allow_custom_value=True,
                            visible=False
                        )
                        constellation_pattern_dropdown = gr.Dropdown(
                            choices=[
                                "2 nálepky: Obe líca",
                                "3 nálepky: Obe líca + Brada",
                                "1 nálepka: Iba ľavé líce"
                            ],
                            value="2 nálepky: Obe líca",
                            label="Rozmiestnenie nálepiek",
                            allow_custom_value=True,
                            visible=False
                        )

                    with gr.Row():
                        patch_size_dropdown = gr.Dropdown(
                            choices=["Stredný (7x18 px / 22x8 mm)", "Malý (5x14 px / 18x6 mm)", "Veľký (9x22 px / 26x10 mm)"],
                            value="Stredný (7x18 px / 22x8 mm)",
                            label="Veľkosť leukoplastu",
                            allow_custom_value=True,
                            visible=False
                        )
                        patch_radius_slider = gr.Slider(
                            minimum=1.5,
                            maximum=8.0,
                            value=4.5,
                            step=0.5,
                            label="Veľkosť nálepky (Polomer px)",
                            visible=False
                        )

                    with gr.Group(visible=False) as live_preview_group:
                        gr.Markdown("#### 🎯 Drag & Drop umiestnenie na tvári")
                        drag_canvas_html = gr.HTML(elem_id="drag_canvas_container")
                        reset_offsets_btn = gr.Button("↺ Resetovať pozície do stredu", size="sm", variant="secondary")
                        live_preview_image = gr.Image(visible=False)
                        with gr.Group(elem_id="drag_sync_hidden_group"):
                            drag_sync_btn = gr.Button("sync_offsets", elem_id="drag_sync_btn")

                    with gr.Row():
                        attack_btn = gr.Button("⚡ Spustiť útok", variant="primary", size="lg")
                        identify_btn = gr.Button("🔍 Rozpoznať tvár", variant="secondary", size="lg")

                    with gr.Accordion("⚙️ Pokročilé parametre a posuvníky", open=False):
                        with gr.Row():
                            epsilon_slider = gr.Slider(minimum=1/255.0, maximum=32/255.0, value=8/255.0, step=1/255.0, label="Epsilon (sila šumu)", visible=True)
                            alpha_slider = gr.Slider(minimum=1/255.0, maximum=10/255.0, value=2/255.0, step=1/255.0, label="Alpha (krok)", visible=True)
                            iter_slider = gr.Slider(minimum=1, maximum=500, value=20, step=1, label="Počet iterácií", visible=True)

                        with gr.Group(visible=False) as offsets_accordion:
                            gr.Markdown("##### Manuálne jemné doladenie posunu prvkov (px)")
                            with gr.Row(visible=True) as strip_offset_row:
                                strip_ox_slider = gr.Slider(-25.0, 25.0, value=0.0, step=0.5, label="↔ Páska na nos / Leukoplast X (px)")
                                strip_oy_slider = gr.Slider(-25.0, 25.0, value=0.0, step=0.5, label="↕ Páska na nos / Leukoplast Y (px)")
                            with gr.Row(visible=False) as left_offset_row:
                                left_ox_slider = gr.Slider(-25.0, 25.0, value=0.0, step=0.5, label="↔ Ľavé líce X (px)")
                                left_oy_slider = gr.Slider(-25.0, 25.0, value=0.0, step=0.5, label="↕ Ľavé líce Y (px)")
                            with gr.Row(visible=False) as right_offset_row:
                                right_ox_slider = gr.Slider(-25.0, 25.0, value=0.0, step=0.5, label="↔ Pravé líce X (px)")
                                right_oy_slider = gr.Slider(-25.0, 25.0, value=0.0, step=0.5, label="↕ Pravé líce Y (px)")
                            with gr.Row(visible=False) as chin_offset_row:
                                chin_ox_slider = gr.Slider(-25.0, 25.0, value=0.0, step=0.5, label="↔ Brada X (px)")
                                chin_oy_slider = gr.Slider(-25.0, 25.0, value=0.0, step=0.5, label="↕ Brada Y (px)")

                    with gr.Accordion("💾 Uložiť fotku do datasetu", open=False):
                        with gr.Row():
                            tab1_person_name = gr.Dropdown(
                                choices=get_dataset_classes(),
                                value="brandajsky_peter",
                                label="Profil pre uloženie",
                                allow_custom_value=True
                            )
                            tab1_save_btn = gr.Button("💾 Uložiť fotku", variant="secondary")
                        tab1_save_status = gr.Textbox(label="Status uloženia", lines=2)

                    def update_attack_controls(selected_attack):
                        is_bounded = selected_attack in ["FGSM", "PGD", "BIM", "MI-FGSM"]
                        is_step = selected_attack in ["PGD", "BIM", "MI-FGSM"]
                        is_nasal = selected_attack == "Športová páska na nos (Breathe Right)"
                        is_leukoplast = selected_attack == "Adversarial Patch (Leukoplast)"
                        is_const = selected_attack == "Constellation Patch (Pimple Patches)"
                        is_hybrid = "Hybrid Patch" in selected_attack
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
                            default_iters = 150

                        show_leuko_size = is_leukoplast
                        show_dots_radius = is_const or is_hybrid

                        if is_nasal or is_hybrid:
                            pos_update = gr.update(
                                visible=True,
                                choices=[
                                    "Farebný vzor (RGB Adversarial Art - 100% prelomenie)",
                                    "Čierna športová (Athlete Black)",
                                    "Karbón s logom (Pro Carbon Logo)",
                                    "Telová klasická (Tan Fabric)",
                                    "Priesvitná silikónová (Clear)"
                                ],
                                value="Farebný vzor (RGB Adversarial Art - 100% prelomenie)",
                                label="Štýl športovej pásky na nos",
                                allow_custom_value=True
                            )
                        elif is_leukoplast:
                            pos_update = gr.update(
                                visible=True,
                                choices=["Koreň nosa (nose_bridge)", "Líce (cheek)", "Čelo (forehead)"],
                                value="Koreň nosa (nose_bridge)",
                                label="Poloha leukoplastu (pre Patch útok)",
                                allow_custom_value=True
                            )
                        else:
                            pos_update = gr.update(visible=False, allow_custom_value=True)

                        default_pattern = "2 nálepky: Obe líca" if is_hybrid else "3 body: Nos + Obe líca"
                        if is_hybrid:
                            pattern_update = gr.update(
                                visible=True,
                                choices=["2 nálepky: Obe líca", "3 nálepky: Obe líca + Brada", "1 nálepka: Iba ľavé líce"],
                                value="2 nálepky: Obe líca",
                                label="Rozmiestnenie nálepiek (pre Hybrid útok)",
                                allow_custom_value=True
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
                                allow_custom_value=True
                            )
                        else:
                            pattern_update = gr.update(visible=False, allow_custom_value=True)

                        active = get_active_handles_for_attack(selected_attack, default_pattern)

                        return [
                            gr.update(visible=is_bounded),
                            gr.update(visible=is_step),
                            gr.update(visible=is_iter, value=default_iters),
                            pos_update,
                            gr.update(visible=show_leuko_size),
                            pattern_update,
                            gr.update(visible=show_dots_radius),
                            gr.update(visible=is_physical),
                            gr.update(visible=active["strip"]),
                            gr.update(visible=active["left"]),
                            gr.update(visible=active["right"]),
                            gr.update(visible=active["chin"]),
                            gr.update(visible=is_physical),
                        ]

                    attack_dropdown.change(
                        fn=update_attack_controls,
                        inputs=[attack_dropdown],
                        outputs=[
                            epsilon_slider,
                            alpha_slider,
                            iter_slider,
                            patch_pos_dropdown,
                            patch_size_dropdown,
                            constellation_pattern_dropdown,
                            patch_radius_slider,
                            offsets_accordion,
                            strip_offset_row,
                            left_offset_row,
                            right_offset_row,
                            chin_offset_row,
                            live_preview_group,
                        ],
                        show_progress="hidden"
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
                        patch_pos_dropdown,
                        patch_size_dropdown,
                        constellation_pattern_dropdown,
                        attack_dropdown,
                        tab1_current_img,
                    ]:
                        comp.change(
                            fn=render_live_placement_preview,
                            inputs=live_preview_inputs,
                            outputs=[live_preview_image, drag_canvas_html],
                            show_progress="hidden",
                        )

                    def handle_drag_sync(img, attack_nm, p_pos, p_sz, c_pat, p_rad, strip_ox, strip_oy, left_ox, left_oy, right_ox, right_oy, chin_ox, chin_oy):
                        active = get_active_handles_for_attack(attack_nm, c_pat)
                        sox = float(strip_ox) if active["strip"] and strip_ox is not None else 0.0
                        soy = float(strip_oy) if active["strip"] and strip_oy is not None else 0.0
                        lox = float(left_ox) if active["left"] and left_ox is not None else 0.0
                        loy = float(left_oy) if active["left"] and left_oy is not None else 0.0
                        rox = float(right_ox) if active["right"] and right_ox is not None else 0.0
                        roy = float(right_oy) if active["right"] and right_oy is not None else 0.0
                        cox = float(chin_ox) if active["chin"] and chin_ox is not None else 0.0
                        coy = float(chin_oy) if active["chin"] and chin_oy is not None else 0.0
                        print(f"📍 [DRAG SYNC] {attack_nm} | strip=({sox}, {soy}), left=({lox}, {loy}), right=({rox}, {roy}), chin=({cox}, {coy})")

                        prev_img, drag_html = render_live_placement_preview(
                            image=img,
                            attack_name=attack_nm,
                            patch_pos=p_pos,
                            patch_size_choice=p_sz,
                            constellation_pattern=c_pat,
                            patch_radius=p_rad,
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
                        ]

                    def handle_reset_offsets(img, attack_nm, p_pos, p_sz, c_pat, p_rad, *rest):
                        prev_img, drag_html = render_live_placement_preview(
                            image=img,
                            attack_name=attack_nm,
                            patch_pos=p_pos,
                            patch_size_choice=p_sz,
                            constellation_pattern=c_pat,
                            patch_radius=p_rad,
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
                        ],
                        js="""(img, attack, p_pos, p_sz, c_pat, p_rad, ...rest) => {
                            try {
                                const offs = window.__getFaceDragOffsets ? window.__getFaceDragOffsets() : null;
                                if (offs) {
                                    return [img, attack, p_pos, p_sz, c_pat, p_rad, offs.strip_ox, offs.strip_oy, offs.left_ox, offs.left_oy, offs.right_ox, offs.right_oy, offs.chin_ox, offs.chin_oy];
                                }
                            } catch(e) {
                                console.error("drag_sync_btn js error:", e);
                            }
                            return [img, attack, p_pos, p_sz, c_pat, p_rad, ...rest];
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
                    gr.Markdown("### 2. Krok: Výsledok a Identifikácia")
                    with gr.Row():
                        orig_identity_output = gr.Textbox(label="Identita PRED útokom (Čistá tvár)", lines=4, placeholder="Po kliknutí na '⚡ Spustiť útok' sa tu zobrazí overenie tvojej identity...")
                        adv_identity_output = gr.Textbox(label="Identita PO útoku (Adversariálna tvár)", lines=4, placeholder="Po kliknutí na '⚡ Spustiť útok' sa tu zobrazí, ako útok oklamal biometriu...")
                    status_output = gr.Textbox(label="Vyhodnotenie biometrického útoku a parametre", lines=2, placeholder="Čaká na spustenie útoku (vyber model, útok a klikni na '⚡ Spustiť útok')...")
                    with gr.Row():
                        orig_face_output = gr.Image(label="👤 Pôvodná orezaná tvár (112×112 px)", type="numpy")
                        adv_image_output = gr.Image(label="🎭 Upravená tvár (Čo vidí model)", type="numpy")
                        noise_image_output = gr.Image(label="🔬 Zosilnený šum / Vzor nálepky na tlač", type="numpy")
            
        # TAB 2: Zber dát
        with gr.TabItem("Zber dát a Finetuning"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 1. Pridanie nových tvárí do datasetu")
                    
                    input_type = gr.Radio(
                        ["Webkamera", "Nahrať súbor"], 
                        value="Webkamera", 
                        label="Zdroj obrázka"
                    )
                    
                    # Webkamera (streaming=False pre fotenie spúšťou)
                    collect_webcam = gr.Image(
                        sources=["webcam"], 
                        streaming=False, 
                        type="pil",
                        label="Webkamera (1. Odfoť sa spúšťou 📷, 2. Klikni 'Uložiť fotku do datasetu')", 
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
                    save_btn = gr.Button("💾 Uložiť fotku do datasetu", variant="primary")
                    
                    save_status = gr.Textbox(label="Status uloženia", lines=3)
                    cropped_preview = gr.Image(label="Náhľad orezanej tváre 112×112 (Čo model uložil)", type="numpy")
                    
                with gr.Column():
                    gr.Markdown("### 2. Dotrénovanie (Fine-Tuning)")
                    gr.Markdown("Po odfotení nových tvárí klikni sem na pretrénovanie BenchmarkCNN modelu.")
                    finetune_btn = gr.Button("🚀 Spustiť Fine-Tuning", variant="primary")
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
                msg = "⚠️ **Záber z webkamery je úplne čierny (pixely = 0)!**\n- Skontroluj, či nemáš fyzickú krytku na webkamere (posuvník nad displejom).\n- Skontroluj, či kameru nepoužíva Teams, Zoom alebo iná appka.\n- Alebo prepni hore na **'Nahrať súbor'** a vyber fotku z disku."
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

        identify_btn.click(
            fn=run_identify,
            inputs=[tab1_current_img, model_dropdown],
            outputs=[orig_face_output, adv_image_output, noise_image_output, orig_identity_output, adv_identity_output, status_output]
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
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, head=HEAD_SCRIPTS)
