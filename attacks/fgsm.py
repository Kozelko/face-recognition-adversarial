import torch
import torch.nn.functional as F

def fgsm_attack_untargeted(model, image, epsilon=8/255):
        # FGSM (Fast Gradient Sign Method) untargeted attack pre tvárovú biometriu.
    # Snaží sa čo najviac vzdialiť adversariálny embedding od pôvodného embeddingu
    # (minimalizovať kosínusovú podobnosť).
    #
    # Predpokladá vstupné obrázky normalizované v rozsahu [-1, 1] (podľa tvojho datasetu).

    # Klonujeme vstup a pridáme veľmi jemný náhodný šum, aby sme rozbili symetriu.
    # Toto je kritický krok, pretože gradient kosínusovej podobnosti dvoch IDENTICKÝCH
    # normalizovaných vektorov je presne NULA. Bez šumu by sa obrázok vôbec nezmenil.
    noise = torch.empty_like(image).uniform_(-1e-3, 1e-3)
    adv_image = (image + noise).clamp(-1.0, 1.0).detach().requires_grad_(True)
    
    # 1. Pôvodný embedding referenčného obrázka (bez gradientov)
    with torch.no_grad():
        orig_emb = model(image).detach()
    
    # 2. Embedding pre perturbovaný obrázok (cez ktorý pretečú gradienty)
    adv_emb = model(adv_image)
    
    # 3. Stratová funkcia (Loss)
    # Chceme, aby sa adv_emb čo najviac líšil od orig_emb.
    # F.cosine_similarity vráti hodnotu blízko 1 pre rovnaké vektory.
    # PyTorch backward() počíta gradient smerom k RASTU stratovej funkcie.
    # Keďže chceme podobnosť ZNÍŽIŤ, našou stratou bude samotná podobnosť,
    # ktorú následne minimalizujeme (teda odčítame gradient).
    loss = F.cosine_similarity(orig_emb, adv_emb).mean()
    
    # 4. Spätný prechod (Backpropagation) pre výpočet gradientu obrázka
    model.zero_grad()
    loss.backward()
    
    # 5. Modifikácia obrázka (FGSM krok)
    # Odčítame smer gradientu, čím minimalizujeme našu "loss" (podobnosť).
    # Normalizovaný rozsah tvojich obrázkov je [-1, 1]. Epsilon na to musí byť prispôsobený.
    adv_image = adv_image - epsilon * adv_image.grad.sign()
    
    # 6. Orezanie späť do platného rozsahu [-1, 1]
    adv_image = torch.clamp(adv_image, -1.0, 1.0)
    
    return adv_image.detach()
