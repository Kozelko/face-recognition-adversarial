import torch
import torch.nn.functional as F

def pgd_attack_untargeted(model, image, epsilon=8/255, alpha=2/255, num_iter=10):
        # PGD (Projected Gradient Descent) untargeted attack pre tvárovú biometriu.
    # Iteratívne aplikuje šum v smere gradientu na minimalizáciu kosínusovej podobnosti.
    # Po každom kroku projektuje perturbáciu späť do L-infinity okolia (epsilon).
    #
    # Predpokladá vstupné obrázky normalizované v rozsahu [-1, 1].

    # Originálny obrázok si uložíme pre projekciu
    orig_image = image.clone().detach()
    
    # Pridáme počiatočný náhodný šum do okolia epsilon (Random Start)
    # Rozbíja symetriu a bráni uviaznutiu v lokálnom optime (alebo nulovom gradiente)
    noise = torch.empty_like(image).uniform_(-epsilon, epsilon)
    adv_image = (image + noise).clamp(-1.0, 1.0).detach().requires_grad_(True)
    
    # Pôvodný embedding (bez gradientov)
    with torch.no_grad():
        orig_emb = model(image).detach()
        
    for i in range(num_iter):
        adv_emb = model(adv_image)
        
        # Chceme znížiť podobnosť, takže ako "loss" berieme priamo podobnosť
        loss = F.cosine_similarity(orig_emb, adv_emb).mean()
        
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            # Aplikácia gradientu s veľkosťou kroku alpha
            adv_image = adv_image - alpha * adv_image.grad.sign()
            
            # Projekcia perturbácie do epsilon-okolia pôvodného obrázka (L-inf norma)
            eta = torch.clamp(adv_image - orig_image, min=-epsilon, max=epsilon)
            
            # Aplikácia projekcie a orezanie do rozsahu obrázka [-1, 1]
            adv_image = torch.clamp(orig_image + eta, min=-1.0, max=1.0)
            
        adv_image.requires_grad_(True)
        
    return adv_image.detach()
