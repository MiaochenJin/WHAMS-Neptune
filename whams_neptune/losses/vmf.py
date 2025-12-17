import torch

def von_mises_fisher_loss(n_pred: torch.Tensor, n_true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """von Mises-Fisher loss for directional data (kappa from ||n_pred||)."""
    kappa = torch.norm(n_pred, dim=1)
    logC = -kappa + torch.log((kappa + eps) / (1 - torch.exp(-2 * kappa) + 2 * eps))
    return -((n_true * n_pred).sum(dim=1) + logC).mean()

def angular_distance_loss(preds: torch.Tensor, labels: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """
    Calculates the angular distance loss. Assumes inputs are unit vectors.
    """
    # Normalize inputs to be safe
    preds_norm = torch.nn.functional.normalize(preds, p=2, dim=1)
    labels_norm = torch.nn.functional.normalize(labels, p=2, dim=1)
    
    # Cosine similarity is the dot product of normalized vectors
    cos_sim = (preds_norm * labels_norm).sum(dim=1)
    
    # Clamp to avoid numerical issues with acos
    cos_sim = torch.clamp(cos_sim, -1.0 + 1e-7, 1.0 - 1e-7)
    
    # Angle in radians
    angle = torch.acos(cos_sim)
    
    if reduction == 'mean':
        return angle.mean()
    elif reduction == 'sum':
        return angle.sum()
    elif reduction == 'none':
        return angle
    else:
        raise ValueError(f"Unknown reduction type: {reduction}")