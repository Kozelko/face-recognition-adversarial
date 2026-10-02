import torch
import torch.nn.functional as F

def pgd_attack_untargeted(model, image, epsilon=8/255, alpha=2/255, num_iter=10):
    """
    PGD untargeted útok pre tvárovú biometriu s L-inf projekciou.
    """
    orig_image = image.clone().detach()
    
    # Random start v okolí epsilon
    noise = torch.empty_like(image).uniform_(-epsilon, epsilon)
    adv_image = (image + noise).clamp(-1.0, 1.0).detach().requires_grad_(True)
    
    with torch.no_grad():
        orig_emb = model(image).detach()
        
    for _ in range(num_iter):
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
