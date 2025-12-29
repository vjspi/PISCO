import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F

def l1_loss_from_difference(error):
    return torch.mean(torch.abs(error))

def l2_loss_from_difference(error):
    return torch.mean(torch.abs(error ** 2))

def huber_loss_from_difference(error, delta=1.0):
    quadratic_loss = 0.5 * (error ** 2)
    linear_loss = delta * (torch.abs(error) - 0.5 * delta)
    loss = torch.where(torch.abs(error) <= delta, quadratic_loss, linear_loss)
    return torch.mean(loss)


def loss_correlation(y1,y2):
    mean_orig = torch.mean(y1, dim=1)
    mean_fit = torch.mean(y2, dim=1)

    numerator = torch.sum(
        (y1 - mean_orig.unsqueeze(1))
        * (y2 - mean_fit.unsqueeze(1)),
        dim=1
    )
    denominator = torch.sqrt(
        torch.sum((y1 - mean_orig.unsqueeze(1)) ** 2, dim=1) *
        torch.sum((y2 - mean_fit.unsqueeze(1)) ** 2, dim=1)
    )
    return 1 - torch.mean(numerator / denominator)


# The proposed ⟂-Loss
def mag_error_l2(v1, v2):
    return (torch.abs(v1) - torch.abs(v2)) ** 2


def pdist_batch(target, est):
    # Find magnitudes
    # l_est = torch.linalg.norm(est, dim=-1)
    # l_tar = torch.linalg.norm(target, dim=-1)
    l_est = torch.abs(est)
    l_tar = torch.abs(target)
    # Find angle
    angle = torch.angle(est - target)

    # Initialize ploss with mag_error_l2 where l_est is less than threshold
    ploss = mag_error_l2(est, target)

    # Calculate cross product for the perpendicular part
    cross = target.real * est.imag - est.real * target.imag
    ploss_est_nonzero = abs(cross) / l_est

    # Handle angles > 90 degrees for non-zero estimated vectors
    angle_condition = abs(angle) > np.pi / 2.0
    compensation = l_tar + (l_tar - ploss_est_nonzero)

    # Apply the condition
    ploss[~angle_condition] = ploss_est_nonzero[~angle_condition]
    ploss[angle_condition] = compensation[angle_condition]

    return ploss + mag_error_l2(est, target)
def multivariate_normal_logpdf(x, mean, cov):
    """
    Compute the log probability density function of a multivariate normal distribution.

    Parameters:
    x (torch.Tensor): Input samples of shape (n_samples, n_features).
    mean (torch.Tensor): Mean vector of shape (n_features,).
    cov (torch.Tensor): Covariance matrix of shape (n_features, n_features).

    Returns:
    torch.Tensor: Log probability density of each sample.
    """
    n = x.shape[1]
    diff = x - mean
    inv_cov = torch.linalg.inv(cov)
    log_det_cov = torch.linalg.slogdet(cov)[1]

    exponent = -0.5 * torch.einsum('ij,ij->i', torch.mm(diff, inv_cov), diff)
    norm_constant = -0.5 * (n * torch.log(2 * torch.tensor(torch.pi)) + log_det_cov)

    return norm_constant + exponent

def distribution_matching_loss(vectors):
    """
    Compute a loss that measures how well the distribution of complex vectors
    matches a reference Gaussian distribution.

    Parameters:
    vectors (torch.Tensor): Array of shape (n_samples, vector_length) with complex vectors.

    Returns:
    torch.Tensor: Distribution matching loss.
    """
    # Flatten the complex vectors into a real-valued representation
    real_vectors = torch.cat([vectors.real, vectors.imag], dim=-1)

    # Compute mean and covariance of the real-valued representation
    mean = torch.mean(real_vectors, dim=0)
    cov = torch.cov(real_vectors.T)

    # Compute the log likelihood of the samples under the reference distribution
    log_likelihood = multivariate_normal_logpdf(real_vectors, mean, cov)

    # The loss can be the negative log likelihood (lower is better)
    loss = -log_likelihood

    return loss

def pdist_phase_aware_loss(X, Y):
    """
    Compute the phase_aware_loss function between all combinations of rows in matrices X and Y.

    Parameters:
    X (torch.Tensor): First complex-valued matrix with shape (M, N).
    Y (torch.Tensor): Second complex-valued matrix with shape (P, N).

    Returns:
    torch.Tensor: Loss matrix with shape (M, P) where each element (i, j) corresponds to the loss between row i of X and row j of Y.
    """
    # Ensure the inputs are complex tensors
    if not torch.is_complex(X) or not torch.is_complex(Y):
        raise ValueError("Input tensors must be complex.")

    # Get shapes
    M, N = X.shape
    P, _ = Y.shape

    # Compute the pairwise differences
    real_diff = X.real.unsqueeze(1) - Y.real.unsqueeze(0)
    imag_diff = X.imag.unsqueeze(1) - Y.imag.unsqueeze(0)

    # Compute the L2 norm of the differences
    real_norm = torch.norm(real_diff, p=2, dim=-1)
    imag_norm = torch.norm(imag_diff, p=2, dim=-1)

    # Sum the norms
    combined_norm = real_norm + imag_norm

    # Compute the L2 norm of the rows in X
    X_norm = torch.norm(X, p=2, dim=-1, keepdim=True)

    # Add a small epsilon to avoid division by zero
    epsilon = 1e-8

    # Compute the relative difference
    relative_difference = combined_norm / (X_norm + epsilon)

    return relative_difference