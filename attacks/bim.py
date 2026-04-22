import torch
import torch.nn.functional as F

def bim_attack_untargeted(model, image, epsilon=8/255, alpha=2/255, num_iter=10):
    """
    BIM (Basic Iterative Method) untargeted attack pre tvárovú biometriu.
    Podobné ako PGD, ale zvyčajne bez počiatočného náhodného šumu (random start).
    """
    orig_image = image.clone().detach()
    # Pridáme veľmi jemný šum na rozbitie symetrie (inak je gradient cosine similarity 0)
    noise = torch.empty_like(image).uniform_(-1e-3, 1e-3)
    adv_image = (image + noise).clamp(-1.0, 1.0).detach().requires_grad_(True)
    
    with torch.no_grad():
        orig_emb = model(image).detach()
        
    for i in range(num_iter):
        adv_emb = model(adv_image)
        
        loss = F.cosine_similarity(orig_emb, adv_emb).mean()
        
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            adv_image = adv_image - alpha * adv_image.grad.sign()
            
            eta = torch.clamp(adv_image - orig_image, min=-epsilon, max=epsilon)
            adv_image = torch.clamp(orig_image + eta, min=-1.0, max=1.0)
            
        adv_image.requires_grad_(True)
        
    return adv_image.detach()
