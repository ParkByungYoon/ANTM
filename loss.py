import torch
import torch.nn.functional as F

def l1_distance_cross_entropy_loss(pred, gt):
    return torch.mean(F.cross_entropy(pred, gt, reduction='none') * torch.abs(gt - torch.argmax(pred, dim=1)))

def l2_distance_cross_entropy_loss(pred, gt):
    return torch.mean(F.cross_entropy(pred, gt, reduction='none') * torch.square(gt - torch.argmax(pred, dim=1)))

def focal_loss(pred,  gt, alpha=0.25, gamma=2):
    
    ce_loss = F.binary_cross_entropy(pred, gt, reduction="none")
    
    p_t = pred * gt + (1 - pred) * (1 - gt)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * gt + (1 - alpha) * (1 - gt)
        loss = alpha_t * loss
        
    return loss.mean()