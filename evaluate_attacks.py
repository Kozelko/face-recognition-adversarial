import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

from models.wrappers import FaceNetWrapper, BenchmarkCNNWrapper
from attacks.fgsm import fgsm_attack_untargeted
from attacks.pgd import pgd_attack_untargeted

def denormalize(tensor):
    """
    Konvertuje tenzor z rozsahu [-1, 1] späť na [0, 1] pre zobrazenie cez matplotlib.
    Predpokladá normalizáciu: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    """
    tensor = tensor * 0.5 + 0.5
    return torch.clamp(tensor, 0.0, 1.0)

def visualize_attack(orig_img, adv_img, diff_img, similarity, title="Útok", save_path="results/attack_visualization.png"):
    """
    Vykreslí pôvodný obrázok, adversariálny obrázok a vizualizuje aplikovaný šum.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    orig_np = denormalize(orig_img).squeeze().permute(1, 2, 0).cpu().numpy()
    adv_np = denormalize(adv_img).squeeze().permute(1, 2, 0).cpu().numpy()
    
    # Zosilníme šum pre vizualizáciu (napr. 10x) aby bol lepšie viditeľný, posunieme do šedej (+0.5)
    diff_np = (adv_img - orig_img).squeeze().permute(1, 2, 0).cpu().numpy()
    diff_np = np.clip((diff_np * 10 + 0.5), 0, 1) 
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"{title} | Cosine Similarity po útoku: {similarity:.4f}", fontsize=14)
    
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
    print(f"Vizualizácia útoku bola uložená do: {save_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Používam zariadenie: {device}\n")

    # 1. Načítanie modelu
    print("Načítavam model (FaceNet)...")
    model = FaceNetWrapper(device=device)
    model.eval()

    # 2. Príprava obrázka
    # Skúsime načítať reálny obrázok z tvojho datasetu (CASIA)
    try:
        from torchvision import datasets, transforms
        
        # Transformácia, ktorú používaš v train.py
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        dataset = datasets.ImageFolder("data/processed/casia", transform=transform)
        # Zoberieme prvý obrázok
        image, label = dataset[0]
        image = image.unsqueeze(0).to(device)
        print("Načítaný skutočný obrázok z datasetu CASIA.")
    except Exception as e:
        print(f"Nepodarilo sa načítať dataset, použijem náhodný dummy obrázok. (Dôvod: {e})")
        # Generujeme dummy obrázok s hodnotami priamo v rozsahu [-1, 1]
        image = torch.rand(1, 3, 112, 112).to(device) * 2 - 1

    # 3. Získame pôvodný embedding
    print("\nExtrahujem pôvodný embedding...")
    with torch.no_grad():
        orig_emb = model(image)

    # 4. Aplikácia FGSM útoku
    # Eps = 16/255 v rozsahu [-1, 1] je ekvivalent vizuálnej zmeny 8/255 v klasickom RGB
    epsilon = 16 / 255.0 
    print(f"Spúšťam FGSM útok (untargeted) s epsilon = {epsilon:.4f}...")
    
    adv_image = fgsm_attack_untargeted(model, image, epsilon=epsilon)

    # 5. Vyhodnotenie útoku
    print("\nExtrahujem embedding po útoku...")
    with torch.no_grad():
        adv_emb = model(adv_image)
        
    similarity = F.cosine_similarity(orig_emb, adv_emb).item()
    
    print(f"Vzdialenosť pred útokom (Cosine Sim): 1.0000")
    print(f"Vzdialenosť po útoku  (Cosine Sim): {similarity:.4f}")
    
    # Pre FaceNet je bežný threshold zhody (rovnaká osoba) niekde okolo 0.5 - 0.6.
    # Ak podobnosť klesne pod túto hranicu, systém tvár rozpozná ako "Neznámu" alebo ako niekoho iného.
    threshold = 0.5
    if similarity < threshold:
        print(f"Útok bol ÚSPEŠNÝ! Podobnosť klesla pod threshold zhody ({threshold}).")
    else:
        print(f"Útok ZLYHAL. Podobnosť je príliš vysoká, model je stále presvedčený, že ide o rovnakú osobu.")

    # 6. Vizualizácia FGSM
    visualize_attack(
        image, adv_image, adv_image - image, 
        similarity, 
        title="FaceNet - FGSM Dodging Attack", 
        save_path="results/fgsm_test_facenet.png"
    )

    # 7. Aplikácia PGD útoku
    # Použijeme menší krok (alpha) a 20 iterácií pre oveľa silnejší útok
    alpha = (2 / 255.0) * 2  # Ekvivalent 2/255 v rozsahu [-1, 1]
    num_iter = 20
    print(f"\nSpúšťam PGD útok (untargeted) s epsilon = {epsilon:.4f}, iterácií = {num_iter}...")
    
    adv_image_pgd = pgd_attack_untargeted(model, image, epsilon=epsilon, alpha=alpha, num_iter=num_iter)
    
    # 8. Vyhodnotenie PGD útoku
    print("\nExtrahujem embedding po PGD útoku...")
    with torch.no_grad():
        adv_emb_pgd = model(adv_image_pgd)
        
    similarity_pgd = F.cosine_similarity(orig_emb, adv_emb_pgd).item()
    print(f"Vzdialenosť po PGD útoku (Cosine Sim): {similarity_pgd:.4f}")
    
    if similarity_pgd < threshold:
        print(f"✅ PGD útok bol ÚSPEŠNÝ! Podobnosť klesla pod threshold zhody ({threshold}).")
    else:
        print(f"❌ PGD útok ZLYHAL. Podobnosť je stále príliš vysoká.")

    # 9. Vizualizácia PGD
    visualize_attack(
        image, adv_image_pgd, adv_image_pgd - image, 
        similarity_pgd, 
        title="FaceNet - PGD Dodging Attack", 
        save_path="results/pgd_test_facenet.png"
    )

if __name__ == "__main__":
    main()
