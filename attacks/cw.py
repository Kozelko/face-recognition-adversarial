import torch
import torch.nn as nn
import torch.optim as optim

def cw_l2_attack_untargeted(model, images, c=1e-4, kappa=0, max_iter=1000, learning_rate=0.01, device=None):
    """
    Carlini & Wagner L2 Untargeted Attack pre Face Recognition (prispôsobený).
    Namiesto klasifikácie (kde C&W minimalizuje logity správnej triedy) tu 
    minimalizujeme kosínusovú podobnosť voči pôvodnému embeddingu.
    """
    if device is None:
        device = images.device

    # C&W funguje v arctanh priestore, aby sme obišli orezávanie [-1, 1]
    # Obrázky preškálujeme z [-1, 1] do [0, 1] a potom do w (arctanh)
    imgs_01 = (images + 1.0) / 2.0
    # Aby sme predišli nekonečnu, orežeme trochu od okrajov
    imgs_01 = torch.clamp(imgs_01, 1e-5, 1.0 - 1e-5)
    # 1. Korektná inicializácia w (inverzná funkcia k 0.5*(tanh(w)+1))
    # w = atanh(2*x - 1)
    w = torch.atanh(torch.clamp(2.0 * imgs_01 - 1.0, min=-1.0 + 1e-6, max=1.0 - 1e-6)).to(device)
    w.requires_grad = True

    # Pôvodné embeddingy (cieľ, od ktorého sa chceme vzdialiť)
    with torch.no_grad():
        orig_embs = model(images).detach()

    # Väčšia hodnota C zabezpečí, že model bude mať vyššiu prioritu ako L2 vzdialenosť
    # Pre embeddingy potrebujeme vyššie C (aspoň 10.0), aby útok prekonal L2 penalizáciu
    c = 10.0 if c == 1e-4 else c 
    
    optimizer = optim.Adam([w], lr=learning_rate * 2)

    for step in range(max_iter):
        # 1. Transformácia z w späť do [-1, 1]
        adv_imgs_01 = 0.5 * (torch.tanh(w) + 1)
        adv_imgs = adv_imgs_01 * 2.0 - 1.0

        # 2. Vzdialenosť (L2 norma perturbácie)
        l2_dist = torch.sum((adv_imgs - images) ** 2, dim=(1, 2, 3))

        # 3. Strata (Loss) pre model
        adv_embs = model(adv_imgs)
        # Kosínusová podobnosť (chceme aby bola čo najmenšia/záporná)
        sims = torch.nn.functional.cosine_similarity(orig_embs, adv_embs)
        
        # Loss f = max(sims - kappa, 0) -> chceme stlačiť podobnosť pod hodnotu kappa (napr. 0.0)
        f_loss = torch.clamp(sims - kappa, min=0.0)

        # Celková strata: L2 + C * f_loss
        loss = l2_dist + c * f_loss
        loss = torch.sum(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Vráti finálne adversariálne obrázky
    with torch.no_grad():
        adv_imgs_01 = 0.5 * (torch.tanh(w) + 1)
        adv_imgs = adv_imgs_01 * 2.0 - 1.0
        
    return torch.clamp(adv_imgs, -1.0, 1.0)
