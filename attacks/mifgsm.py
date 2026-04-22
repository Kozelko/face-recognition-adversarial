import torch
import torch.nn.functional as F

def mifgsm_attack_untargeted(model, image, epsilon=8/255, alpha=2/255, num_iter=10, decay=1.0):
    """
    MI-FGSM (Momentum Iterative FGSM) untargeted attack pre tvárovú biometriu.
    Zahŕňa momentový člen na prekonanie lokálnych miním a stabilizáciu updatov.
    """
    orig_image = image.clone().detach()
    # Pridáme veľmi jemný šum na rozbitie symetrie (inak je gradient cosine similarity 0)
    noise = torch.empty_like(image).uniform_(-1e-3, 1e-3)
    adv_image = (image + noise).clamp(-1.0, 1.0).detach().requires_grad_(True)
    momentum = torch.zeros_like(image).detach()
    
    with torch.no_grad():
        orig_emb = model(image).detach()
        
    for i in range(num_iter):
        adv_emb = model(adv_image)
        
        loss = F.cosine_similarity(orig_emb, adv_emb).mean()
        
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            grad = adv_image.grad
            
            # L1 normovanie gradientu pre stabilný momentum
            grad_norm = torch.norm(grad, p=1, dim=[1, 2, 3], keepdim=True)
            grad_norm = torch.clamp(grad_norm, min=1e-8)
            grad = grad / grad_norm
            
            momentum = decay * momentum + grad
            
            adv_image = adv_image - alpha * momentum.sign()
            
            eta = torch.clamp(adv_image - orig_image, min=-epsilon, max=epsilon)
            adv_image = torch.clamp(orig_image + eta, min=-1.0, max=1.0)
            
        adv_image.requires_grad_(True)
        
    return adv_image.detach()
