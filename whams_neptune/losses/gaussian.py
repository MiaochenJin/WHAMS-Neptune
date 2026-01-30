import torch

def gaussian_nll_1d(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Computes the Negative Log Likelihood (NLL) for a 1D Gaussian distribution.

    Args:
        preds: Tensor of shape (batch_size, 2).
               preds[:, 0] is the predicted mean.
               preds[:, 1] is the raw predicted standard deviation (sigma).
        targets: Tensor of shape (batch_size,) containing the true values.
        eps: Small value for numerical stability.

    Returns:
        Scalar tensor representing the mean NLL over the batch.
    """
    pred_mean = preds[:, 0]
    raw_sigma = preds[:, 1]

    # Ensure sigma is positive
    sigma = torch.nn.functional.softplus(raw_sigma) + eps

    # Calculate squared difference
    squared_diff = (targets - pred_mean) ** 2

    # NLL = log(sigma) + (target - pred_mean)^2 / (2 * sigma^2)
    # Note: We omit the constant term 0.5 * log(2 * pi) as it doesn't affect optimization
    log_sigma = torch.log(sigma)
    nll = log_sigma + squared_diff / (2 * sigma ** 2)

    return nll.mean()

def isotropic_gaussian_nll_3d(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Computes the Negative Log Likelihood (NLL) for a 3D isotropic Gaussian distribution.
    
    Args:
        preds: Tensor of shape (batch_size, 4). 
               preds[:, :3] are the predicted means (x, y, z).
               preds[:, 3] is the raw predicted standard deviation (sigma).
        targets: Tensor of shape (batch_size, 3) containing the true (x, y, z) coordinates.
        eps: Small value for numerical stability.

    Returns:
        Scalar tensor representing the mean NLL over the batch.
    """
    pred_means = preds[:, :3]
    raw_sigma = preds[:, 3]
    
    # Ensure sigma is positive
    sigma = torch.nn.functional.softplus(raw_sigma) + eps
    
    # Calculate squared Euclidean distance: ||x_true - x_pred||^2
    squared_diff = torch.sum((targets - pred_means) ** 2, dim=1)
    
    # NLL = 3 * log(sigma) + (||x_true - x_pred||^2) / (2 * sigma^2)
    # Note: We omit the constant term 1.5 * log(2 * pi) as it doesn't affect optimization
    log_sigma = torch.log(sigma)
    nll = 3 * log_sigma + squared_diff / (2 * sigma ** 2)
    
    return nll.mean()

def mean_euclidean_distance(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the mean Euclidean distance between predicted means and targets.
    Useful for validation metrics.
    """
    pred_means = preds[:, :3]
    return torch.norm(targets - pred_means, dim=1).mean()

def mean_absolute_error_1d(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the mean absolute error for 1D predictions.
    Useful for validation metrics for energy reconstruction.

    Args:
        preds: Tensor of shape (batch_size, 2) where preds[:, 0] is the predicted mean.
        targets: Tensor of shape (batch_size,) containing the true values.

    Returns:
        Scalar tensor representing the mean absolute error.
    """
    pred_mean = preds[:, 0]
    return torch.abs(targets - pred_mean).mean()

def mean_squared_error_1d(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the mean squared error for 1D predictions.
    Useful for validation metrics for energy reconstruction.

    Args:
        preds: Tensor of shape (batch_size, 2) where preds[:, 0] is the predicted mean.
        targets: Tensor of shape (batch_size,) containing the true values.

    Returns:
        Scalar tensor representing the mean squared error.
    """
    pred_mean = preds[:, 0]
    return ((targets - pred_mean) ** 2).mean()

def mean_predicted_uncertainty_1d(preds: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Computes the mean predicted uncertainty (sigma) for 1D Gaussian predictions.

    Args:
        preds: Tensor of shape (batch_size, 2) where preds[:, 1] is the raw predicted sigma.
        eps: Small value for numerical stability.

    Returns:
        Scalar tensor representing the mean predicted uncertainty.
    """
    raw_sigma = preds[:, 1]
    sigma = torch.nn.functional.softplus(raw_sigma) + eps
    return sigma.mean()
