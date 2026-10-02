import torch
import torch.nn.functional as F

def fgsm_attack_untargeted(model, image, epsilon=8/255):
    """
    FGSM untargeted útok pre tvárovú biometriu.
    Minimalizuje kosínusovú podobnosť voči pôvodnému embeddingu.
    """
    noise = torch.empty_like(image).uniform_(-1e-3, 1e-3)
    adv_image = (image + noise).clamp(-1.0, 1.0).detach().requires_grad_(True)
    
    with torch.no_grad():
        orig_emb = model(image).detach()
    
    adv_emb = model(adv_image)
    loss = F.cosine_similarity(orig_emb, adv_emb).mean()
    
    model.zero_grad()
    loss.backward()
    
    adv_image = adv_image - epsilon * adv_image.grad.sign()
    adv_image = torch.clamp(adv_image, -1.0, 1.0)
    
    return adv_image.detach()
