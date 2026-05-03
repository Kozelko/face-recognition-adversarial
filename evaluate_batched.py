import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import pandas as pd
import os
import random
import time

from models.wrappers import FaceNetWrapper, BenchmarkCNNWrapper, ArcFaceWrapper, AdaFaceWrapper
from attacks.fgsm import fgsm_attack_untargeted
from attacks.pgd import pgd_attack_untargeted
from attacks.bim import bim_attack_untargeted
from attacks.mifgsm import mifgsm_attack_untargeted
from attacks.cw import cw_l2_attack_untargeted

# --- Nastavenia ---
DATA_DIR = "data/processed/casia"
BATCH_SIZE = 32  # Znížené z 64 na 32 kvôli kapacite VRAM (6GB na RTX 3060) pri väčších modeloch ako ArcFace
NUM_TEST_IMAGES = 2000  # Finálny test pre diplomovku
THRESHOLD = 0.5         # Hranica cosine similarity pre oklamanie modelu
RESULTS_FILE = "results/batched_evaluation.csv"

# Parametre útokov
EPSILON = 8 / 255.0     # Štandard pre L_inf útoky na tváre/ImageNet
ALPHA = 2 / 255.0       # Štandardný krok
NUM_ITER = 20           # Štandard pre PGD-20 / BIM / MI-FGSM

def load_benchmark_cnn(device):
    checkpoint_path = "models/checkpoints/benchmark_cnn_best_run1.pth"
    if not os.path.exists(checkpoint_path):
        return None
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Snažíme sa nájsť počet tried z datasetu, ak zlyhá, dáme 10575 (casia default)
        try:
            num_classes = len(datasets.ImageFolder(DATA_DIR).classes)
        except:
            num_classes = 10575
        model = BenchmarkCNNWrapper(num_classes=num_classes, checkpoint_path=checkpoint_path, device=device)
        return model
    except Exception as e:
        print(f"Chyba pri načítaní BenchmarkCNN: {e}")
        return None

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Používam zariadenie: {device}")

    # 1. Príprava datasetu a Dataloaderu
    print("Načítavam dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    
    # Náhodný výber podmnožiny
    total_images = len(full_dataset)
    indices = random.sample(range(total_images), min(NUM_TEST_IMAGES, total_images))
    test_subset = Subset(full_dataset, indices)
    
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Veľkosť testovacej sady: {len(test_subset)} obrázkov (Batch size: {BATCH_SIZE})")

    # 2. Definícia modelov (Inicializujú sa postupne kvôli VRAM)
    model_creators = {
        "FaceNet": lambda: FaceNetWrapper(device=device),
        "ArcFace": lambda: ArcFaceWrapper(device=device),
        "AdaFace": lambda: AdaFaceWrapper(device=device),
        "BenchmarkCNN": lambda: load_benchmark_cnn(device)
    }

    # 3. Definícia útokov
    attacks = {
        "FGSM": lambda m, img: fgsm_attack_untargeted(m, img, epsilon=EPSILON),
        "PGD": lambda m, img: pgd_attack_untargeted(m, img, epsilon=EPSILON, alpha=ALPHA, num_iter=NUM_ITER),
        "BIM": lambda m, img: bim_attack_untargeted(m, img, epsilon=EPSILON, alpha=ALPHA, num_iter=NUM_ITER),
        "MI-FGSM": lambda m, img: mifgsm_attack_untargeted(m, img, epsilon=EPSILON, alpha=ALPHA, num_iter=NUM_ITER),
        "C&W": lambda m, img: cw_l2_attack_untargeted(m, img, max_iter=100) # 100 iterácií by malo s novými parametrami stačiť
    }

    # Príprava na ukladanie výsledkov
    results = []
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

    # 4. Samotná evaluácia po modeloch
    for model_name, creator in model_creators.items():
        print(f"\n{'='*50}")
        print(f"=== Inicializujem model: {model_name} ===")
        model = creator()
        
        if model is None:
            print(f"Model {model_name} sa nepodarilo načítať. Preskakujem.")
            continue
            
        # Prepni model do eval režimu
        model.eval()

        for attack_name, attack_fn in attacks.items():
            print(f"\n--- Spúšťam útok: {attack_name} na modeli {model_name} ---")
            start_time = time.time()
            
            total_images_processed = 0
            successful_attacks = 0  # Počet obrázkov, kde similarity klesla pod THRESHOLD
            avg_similarity_drop = 0.0
            avg_l2_norm = 0.0
            avg_linf_norm = 0.0

            for batch_idx, (images, labels) in enumerate(test_loader):
                images = images.to(device)
                
                # 1. Získaj pôvodný embedding
                with torch.no_grad():
                    orig_embs = model(images)
                
                # 2. Vytvor adversariálne obrázky
                adv_images = attack_fn(model, images)
                
                # 3. Získaj adversariálne embeddingy
                with torch.no_grad():
                    adv_embs = model(adv_images)
                
                # 4. Vypočítaj Cosine Similarity pre celý batch
                similarities = F.cosine_similarity(orig_embs, adv_embs)
                
                # Výpočet veľkosti šumu (perturbácie)
                perturbation = adv_images - images
                # L2 norma (celková energia šumu) - prepočítaná na batch
                l2 = torch.norm(perturbation.view(perturbation.size(0), -1), p=2, dim=1).sum().item()
                # L_inf norma (maximálna zmena pixelu)
                linf = torch.norm(perturbation.view(perturbation.size(0), -1), p=float('inf'), dim=1).sum().item()
                
                # 5. Vyhodnoť úspešnosť v batchi
                batch_success = (similarities < THRESHOLD).sum().item()
                successful_attacks += batch_success
                avg_similarity_drop += similarities.sum().item()
                avg_l2_norm += l2
                avg_linf_norm += linf
                total_images_processed += len(images)
                
                # Progres v konzole
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Spracovaných dávok: {batch_idx + 1}/{len(test_loader)}")

            end_time = time.time()
            elapsed = end_time - start_time
            
            # Výpočet finálnych metrík
            success_rate = (successful_attacks / total_images_processed) * 100
            mean_sim = avg_similarity_drop / total_images_processed
            mean_l2 = avg_l2_norm / total_images_processed
            mean_linf = avg_linf_norm / total_images_processed
            time_per_image = (elapsed / total_images_processed) * 1000 # v milisekundách
            
            print(f"-> Úspešnosť útoku (Success Rate): {success_rate:.2f}% (Similarity < {THRESHOLD})")
            print(f"-> Priemerná Cosine Sim. po útoku: {mean_sim:.4f}")
            print(f"-> Priemerný L2 šum: {mean_l2:.4f}, L_inf šum: {mean_linf:.4f}")
            print(f"-> Trvanie: {elapsed:.2f} s ({time_per_image:.2f} ms / obrázok)")

            # Ulož výsledok
            results.append({
                "Model": model_name,
                "Attack": attack_name,
                "Num_Images": total_images_processed,
                "Batch_Size": BATCH_SIZE,
                "Epsilon": EPSILON,
                "Num_Iter": NUM_ITER if attack_name != "FGSM" else 1,
                "Success_Rate_pct": round(success_rate, 2),
                "Mean_Similarity": round(mean_sim, 4),
                "Mean_L2_Noise": round(mean_l2, 4),
                "Mean_Linf_Noise": round(mean_linf, 4),
                "Time_Seconds": round(elapsed, 2),
                "Time_Per_Image_ms": round(time_per_image, 2)
            })
            
            # Priebežne ukladaj do CSV
            df = pd.DataFrame(results)
            df.to_csv(RESULTS_FILE, index=False)

        # Uvoľni pamäť GPU po každom modeli
        del model
        torch.cuda.empty_cache()

    print(f"\n✅ Evaluácia dokončená. Výsledky uložené do {RESULTS_FILE}")

if __name__ == "__main__":
    main()
