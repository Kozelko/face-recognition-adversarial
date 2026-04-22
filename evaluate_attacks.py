import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

from models.wrappers import FaceNetWrapper, BenchmarkCNNWrapper, ArcFaceWrapper, AdaFaceWrapper
from attacks.fgsm import fgsm_attack_untargeted
from attacks.pgd import pgd_attack_untargeted
from attacks.bim import bim_attack_untargeted
from attacks.mifgsm import mifgsm_attack_untargeted

def denormalize(tensor):
    # Konvertuje tenzor z rozsahu [-1, 1] späť na [0, 1] pre zobrazenie cez matplotlib.
    # Predpokladá normalizáciu: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    tensor = tensor * 0.5 + 0.5
    return torch.clamp(tensor, 0.0, 1.0)

def visualize_attack(orig_img, adv_img, diff_img, similarity, title="Útok", save_path="results/attack_visualization.png"):
    # Vykreslí pôvodný obrázok, adversariálny obrázok a vizualizuje aplikovaný šum.
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    orig_np = denormalize(orig_img).squeeze().permute(1, 2, 0).cpu().numpy()
    adv_np = denormalize(adv_img).squeeze().permute(1, 2, 0).cpu().numpy()
    
    # Zosilníme šum pre vizualizáciu (napr. 10x) aby bol lepšie viditeľný, posunieme do šedej (+0.5)
    diff_np = (adv_img - orig_img).squeeze().permute(1, 2, 0).cpu().numpy()
    diff_np = np.clip((diff_np * 10 + 0.5), 0, 1) 
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"{title} | Cosine Sim po útoku: {similarity:.4f}", fontsize=14)
    
    axes[0].imshow(orig_np)
    axes[0].set_title("Pôvodný obrázok")
    axes[0].axis('off')
    
    axes[1].imshow(adv_np)
    axes[1].set_title("Adversariálny obrázok")
    axes[1].axis('off')
    
    axes[2].imshow(diff_np)
    axes[2].set_title("Aplikovaný šum (Zosilnený 10x)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Vizualizácia uložená: {save_path}")

def load_benchmark_cnn(device):
    checkpoint_path = "models/checkpoints/benchmark_cnn_best_run1.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Nemám checkpoint {checkpoint_path}")
        return None
    try:
        # Pre prípad, ak model načítať priamo (často checkpoint obsahuje parametre)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        num_classes = checkpoint.get("num_classes", 10575)  # Fallback na CASIA-WebFace
        model = BenchmarkCNNWrapper(num_classes=num_classes, checkpoint_path=checkpoint_path, device=device)
        return model
    except Exception as e:
        print(f"Chyba pri načítaní BenchmarkCNN: {e}")
        return None

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Používam zariadenie: {device}\n")

    # 1. Príprava modelov
    models_dict = {}
    
    print("Načítavam model: FaceNet...")
    models_dict["FaceNet"] = FaceNetWrapper(device=device)
    
    print("Načítavam model: ArcFace...")
    models_dict["ArcFace"] = ArcFaceWrapper(device=device)
    
    print("Načítavam model: AdaFace...")
    models_dict["AdaFace"] = AdaFaceWrapper(device=device)
    
    print("Načítavam model: BenchmarkCNN...")
    bcnn = load_benchmark_cnn(device)
    if bcnn is not None:
        models_dict["BenchmarkCNN"] = bcnn

    # 2. Príprava obrázka
    try:
        from torchvision import datasets, transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        dataset = datasets.ImageFolder("data/processed/casia", transform=transform)
        image, label = dataset[0]
        image = image.unsqueeze(0).to(device)
        print("Načítaný skutočný obrázok z datasetu CASIA.")
    except Exception as e:
        print(f"Nepodarilo sa načítať dataset, použijem náhodný dummy obrázok. (Dôvod: {e})")
        image = torch.rand(1, 3, 112, 112).to(device) * 2 - 1

    # Definícia útokov
    epsilon = 16 / 255.0
    alpha = (2 / 255.0) * 2
    num_iter = 20
    
    attacks = {
        "FGSM": lambda m, img: fgsm_attack_untargeted(m, img, epsilon=epsilon),
        "PGD": lambda m, img: pgd_attack_untargeted(m, img, epsilon=epsilon, alpha=alpha, num_iter=num_iter),
        "BIM": lambda m, img: bim_attack_untargeted(m, img, epsilon=epsilon, alpha=alpha, num_iter=num_iter),
        "MI-FGSM": lambda m, img: mifgsm_attack_untargeted(m, img, epsilon=epsilon, alpha=alpha, num_iter=num_iter)
    }

    threshold = 0.5

    # 3. Spustenie útokov na každom modeli
    for model_name, model in models_dict.items():
        print(f"\n{'='*50}")
        print(f"=== Testujem model: {model_name} ===")
        print(f"{'='*50}")
        
        # Získame pôvodný embedding pre tento model
        with torch.no_grad():
            orig_emb = model(image)
            
        for attack_name, attack_fn in attacks.items():
            print(f"\n--- Spúšťam útok: {attack_name} ---")
            adv_image = attack_fn(model, image)
            
            with torch.no_grad():
                adv_emb = model(adv_image)
                
            similarity = F.cosine_similarity(orig_emb, adv_emb).item()
            print(f"Vzdialenosť po {attack_name} útoku (Cosine Sim): {similarity:.4f}")
            
            if similarity < threshold:
                print(f"✅ Útok {attack_name} bol ÚSPEŠNÝ! Podobnosť klesla pod {threshold}.")
            else:
                print(f"❌ Útok {attack_name} ZLYHAL. Podobnosť je stále príliš vysoká.")

            save_path = f"results/{attack_name.lower()}_test_{model_name.lower()}.png"
            visualize_attack(
                image, adv_image, adv_image - image, 
                similarity, 
                title=f"{model_name} - {attack_name} Dodging Attack", 
                save_path=save_path
            )

if __name__ == "__main__":
    main()
