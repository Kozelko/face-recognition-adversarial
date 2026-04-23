import gradio as gr
import torch
import torch.nn.functional as F
import os
import cv2
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

# --- Globálne premenné a nastavenia ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_DIR = "data/custom_dataset"

# --- Načítanie modelov ---
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
            print("✅ Finetuned BenchmarkCNN úspešne načítaný.")
            
        print("Modely úspešne načítané.")

# Inicializuj modely pri štarte aplikácie
init_models()

# --- Funkcie pre záložku 1: Útoky ---
def run_attack(image, model_name, attack_name, epsilon, alpha, num_iter):
    if image is None:
        return None, None, "Prosím, nahrajte alebo odfoťte obrázok."
    
    # Príprava obrázka (Gradio vracia numpy array HxWxC v RGB)
    # Konvertujeme na PyTorch tensor 1xCxHxW v rozsahu [-1, 1]
    img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    img_tensor = (img_tensor - 0.5) / 0.5  # Normalizácia
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    
    model = models_dict.get(model_name)
    if model is None:
        return None, None, f"Model {model_name} nie je dostupný."
    
    # Pôvodný embedding
    with torch.no_grad():
        orig_emb = model(img_tensor)
        
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
        adv_tensor = cw_l2_attack_untargeted(model, img_tensor, max_iter=int(num_iter))
    else:
        return None, None, "Neznámy útok."
        
    # Adversariálny embedding
    with torch.no_grad():
        adv_emb = model(adv_tensor)
        
    similarity = F.cosine_similarity(orig_emb, adv_emb).item()
    
    # Konverzia späť na obrázky pre zobrazenie
    adv_img_np = denormalize(adv_tensor).squeeze().permute(1, 2, 0).cpu().numpy()
    
    # Zvýraznený šum
    diff_np = (adv_tensor - img_tensor).squeeze().permute(1, 2, 0).cpu().numpy()
    diff_np = (diff_np * 10 + 0.5).clip(0, 1)  # Zosilnené 10x
    
    status_text = f"✅ Cosine Similarity: {similarity:.4f}\n"
    if similarity < 0.5:
        status_text += "Útok bol úspešný! Model bol oklamaný."
    else:
        status_text += "Útok zlyhal. Podobnosť je stále vysoká."
        
    return adv_img_np, diff_np, status_text

# --- Funkcie pre záložku 2: Zber dát ---
def save_image_to_dataset(image, person_name):
    if image is None:
        return "⚠️ Žiadny obrázok na uloženie."
    if not person_name or person_name.strip() == "":
        return "⚠️ Prosím, zadaj meno osoby."
    
    person_name = person_name.strip().replace(" ", "_")
    person_dir = os.path.join(DATASET_DIR, person_name)
    os.makedirs(person_dir, exist_ok=True)
    
    # Vytvorenie unikátneho názvu
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}.jpg"
    filepath = os.path.join(person_dir, filename)
    
    # Gradio image je RGB numpy array, cv2 používa BGR
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Tu by sa ideálne zišlo pridať detekciu tváre (napr. MTCNN) a orezanie na 112x112
    # Pre zjednodušenie ukladáme zatiaľ ako 112x112 (rezize)
    img_resized = cv2.resize(img_bgr, (112, 112))
    
    cv2.imwrite(filepath, img_resized)
    
    count = len(os.listdir(person_dir))
    return f"✅ Fotka úspešne uložená do {filepath}. Osoba '{person_name}' má teraz {count} fotiek."

def handle_finetune(progress=gr.Progress()):
    success, msg = run_finetuning(dataset_dir=DATASET_DIR, epochs=15, lr=0.001, progress=progress)
    if success:
        # Skús znovu načítať nový model do pamäte
        bcnn_finetuned = load_finetuned_cnn(DEVICE)
        if bcnn_finetuned is not None:
            models_dict["BenchmarkCNN (Finetuned)"] = bcnn_finetuned
            return msg, gr.update(choices=list(models_dict.keys()), value="BenchmarkCNN (Finetuned)")
    return msg, gr.update()

# --- Vytvorenie Gradio UI ---
with gr.Blocks(title="Face Recognition & Adversarial Attacks") as app:
    gr.Markdown("# 🛡️ Face Recognition & Adversarial Attacks Platform")
    gr.Markdown("Prototyp pre testovanie adversariálnych útokov a zber dát.")
    
    with gr.Tabs():
        # TAB 1: Testovanie útokov
        with gr.TabItem("⚔️ Testovanie útokov"):
            with gr.Row():
                with gr.Column(scale=1):
                    # Vstup
                    input_image = gr.Image(sources=["upload", "webcam"], label="Vstupný obrázok (Webkamera / Upload)")
                    
                    # Nastavenia
                    model_dropdown = gr.Dropdown(
                        choices=list(models_dict.keys()), 
                        value="BenchmarkCNN", 
                        label="Cieľový Model"
                    )
                    attack_dropdown = gr.Dropdown(
                        choices=["FGSM", "PGD", "BIM", "MI-FGSM", "C&W"], 
                        value="PGD", 
                        label="Typ Útoku"
                    )
                    
                    epsilon_slider = gr.Slider(minimum=1/255.0, maximum=32/255.0, value=8/255.0, step=1/255.0, label="Epsilon (Sila šumu pre L_inf)")
                    alpha_slider = gr.Slider(minimum=1/255.0, maximum=10/255.0, value=2/255.0, step=1/255.0, label="Alpha (Krok pre PGD/BIM/MI-FGSM)")
                    iter_slider = gr.Slider(minimum=1, maximum=500, value=20, step=1, label="Počet iterácií (PGD=20, C&W=100+)")
                    
                    attack_btn = gr.Button("🚀 Spustiť útok", variant="primary")
                
                with gr.Column(scale=2):
                    # Výstup
                    status_output = gr.Textbox(label="Výsledok", lines=2)
                    with gr.Row():
                        adv_image_output = gr.Image(label="Adversariálny obrázok")
                        noise_image_output = gr.Image(label="Zosilnený šum (x10)")
            
            attack_btn.click(
                fn=run_attack,
                inputs=[input_image, model_dropdown, attack_dropdown, epsilon_slider, alpha_slider, iter_slider],
                outputs=[adv_image_output, noise_image_output, status_output]
            )
            
        # TAB 2: Zber dát
        with gr.TabItem("📸 Zber dát a Finetuning"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Pridanie nových tvárí do custom datasetu")
                    collect_image = gr.Image(sources=["webcam", "upload"], label="Webkamera")
                    person_name_input = gr.Textbox(label="Meno osoby (napr. janko_hrasko)", placeholder="Zadaj meno a stlač 'Uložiť'")
                    save_btn = gr.Button("💾 Uložiť fotku", variant="primary")
                    save_status = gr.Textbox(label="Status")
                    
                with gr.Column():
                    gr.Markdown("### Dotrénovanie (Fine-Tuning)")
                    gr.Markdown("""
                    Ak si nazbieral nové fotky, môžeš model naučiť rozpoznávať tieto nové tváre technikou **Transfer Learning / Fine-Tuning**.
                    
                    **Ako to funguje:**
                    1. Model sa nenačíta s náhodnými váhami, ale s už natrénovanými váhami (z tvojho checkpointu).
                    2. Posledná vrstva (klasifikátor), ktorá doteraz rozoznávala napr. 10000 ľudí, sa "odreže" a nahradí novou, ktorá rozpoznáva tvoje nové identity (napr. +1 nová osoba).
                    3. Ostatné vrstvy (Feature Extractor) sa zmrazia, takže model nezabudne "ako vyzerá tvár", len sa doučí spojiť nové črty s tvojím menom. Trvá to veľmi krátko (niekoľko minút).
                    """)
                    
                    # Tlačidlo pre spustenie trénovania
                    finetune_btn = gr.Button("🔄 Spustiť Fine-Tuning", variant="primary")
                    finetune_status = gr.Textbox(label="Status Finetuningu")
                    
            save_btn.click(
                fn=save_image_to_dataset,
                inputs=[collect_image, person_name_input],
                outputs=[save_status]
            )
            
            finetune_btn.click(
                fn=handle_finetune,
                inputs=[],
                outputs=[finetune_status, model_dropdown]
            )

if __name__ == "__main__":
    # Vytvorenie zložky na dáta ak neexistuje
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    # Spustenie aplikácie na porte 7860 (default)
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
