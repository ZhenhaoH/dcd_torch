"""
Utility functions for shape matching operations.
"""

import torch
from typing import Tuple, Optional


def compute_center_of_mass(positions: torch.Tensor, masses: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute center of mass for point sets.
    
    Args:
        positions: Positions of shape (..., N, 3) where N is number of particles
        masses: Optional masses of shape (..., N). If None, uniform masses are assumed.
        
    Returns:
        Center of mass of shape (..., 3)
    """
    if masses is None:
        return torch.mean(positions, dim=-2)
    else:
        # Weighted center of mass
        masses_expanded = masses.unsqueeze(-1)  # (..., N, 1)
        total_mass = torch.sum(masses, dim=-1, keepdim=True).unsqueeze(-1)  # (..., 1, 1)
        weighted_sum = torch.sum(positions * masses_expanded, dim=-2, keepdim=True)  # (..., 1, 3)
        return (weighted_sum / total_mass).squeeze(-2)  # (..., 3)


def polar_decomposition(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute polar decomposition A = R * S where R is rotation and S is symmetric positive definite.
    
    Args:
        A: Matrix of shape (..., 3, 3)
        
    Returns:
        Tuple of (R, S) where R is rotation matrix and S is symmetric matrix
    """
    # Use SVD: A = U * Σ * V^T
    U, sigma, Vt = torch.linalg.svd(A)
    
    # R = U * V^T
    R = torch.matmul(U, Vt)
    
    # Ensure proper rotation (determinant = 1, not -1)
    det_R = torch.linalg.det(R)
    
    # If determinant is negative, flip the last column of U
    mask = det_R < 0
    if mask.any():
        U_corrected = U.clone()
        U_corrected[mask, :, -1] *= -1
        R = torch.where(mask.unsqueeze(-1).unsqueeze(-1), 
                       torch.matmul(U_corrected, Vt), R)
    
    # S = V * Σ * V^T
    V = Vt.transpose(-2, -1)
    sigma_matrix = torch.diag_embed(sigma)
    S = torch.matmul(torch.matmul(V, sigma_matrix), Vt)
    
    return R, S


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to quaternion [w, x, y, z].
    
    Args:
        R: Rotation matrix of shape (..., 3, 3)
        
    Returns:
        Quaternion of shape (..., 4) with [w, x, y, z] convention
    """
    batch_shape = R.shape[:-2]
    R_flat = R.view(-1, 3, 3)
    batch_size = R_flat.shape[0]
    
    q = torch.zeros(batch_size, 4, dtype=R.dtype, device=R.device)
    
    trace = R_flat[:, 0, 0] + R_flat[:, 1, 1] + R_flat[:, 2, 2]
    
    # Case 1: trace > 0
    mask1 = trace > 0
    if mask1.any():
        s = torch.sqrt(trace[mask1] + 1.0) * 2  # s = 4 * w
        q[mask1, 0] = 0.25 * s  # w
        q[mask1, 1] = (R_flat[mask1, 2, 1] - R_flat[mask1, 1, 2]) / s  # x
        q[mask1, 2] = (R_flat[mask1, 0, 2] - R_flat[mask1, 2, 0]) / s  # y
        q[mask1, 3] = (R_flat[mask1, 1, 0] - R_flat[mask1, 0, 1]) / s  # z
    
    # Case 2: R[0,0] > R[1,1] and R[0,0] > R[2,2]
    mask2 = (~mask1) & (R_flat[:, 0, 0] > R_flat[:, 1, 1]) & (R_flat[:, 0, 0] > R_flat[:, 2, 2])
    if mask2.any():
        s = torch.sqrt(1.0 + R_flat[mask2, 0, 0] - R_flat[mask2, 1, 1] - R_flat[mask2, 2, 2]) * 2
        q[mask2, 0] = (R_flat[mask2, 2, 1] - R_flat[mask2, 1, 2]) / s  # w
        q[mask2, 1] = 0.25 * s  # x
        q[mask2, 2] = (R_flat[mask2, 0, 1] + R_flat[mask2, 1, 0]) / s  # y
        q[mask2, 3] = (R_flat[mask2, 0, 2] + R_flat[mask2, 2, 0]) / s  # z
    
    # Case 3: R[1,1] > R[2,2]
    mask3 = (~mask1) & (~mask2) & (R_flat[:, 1, 1] > R_flat[:, 2, 2])
    if mask3.any():
        s = torch.sqrt(1.0 + R_flat[mask3, 1, 1] - R_flat[mask3, 0, 0] - R_flat[mask3, 2, 2]) * 2
        q[mask3, 0] = (R_flat[mask3, 0, 2] - R_flat[mask3, 2, 0]) / s  # w
        q[mask3, 1] = (R_flat[mask3, 0, 1] + R_flat[mask3, 1, 0]) / s  # x
        q[mask3, 2] = 0.25 * s  # y
        q[mask3, 3] = (R_flat[mask3, 1, 2] + R_flat[mask3, 2, 1]) / s  # z
    
    # Case 4: else
    mask4 = (~mask1) & (~mask2) & (~mask3)
    if mask4.any():
        s = torch.sqrt(1.0 + R_flat[mask4, 2, 2] - R_flat[mask4, 0, 0] - R_flat[mask4, 1, 1]) * 2
        q[mask4, 0] = (R_flat[mask4, 1, 0] - R_flat[mask4, 0, 1]) / s  # w
        q[mask4, 1] = (R_flat[mask4, 0, 2] + R_flat[mask4, 2, 0]) / s  # x
        q[mask4, 2] = (R_flat[mask4, 1, 2] + R_flat[mask4, 2, 1]) / s  # y
        q[mask4, 3] = 0.25 * s  # z
    
    return q.view(*batch_shape, 4)


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion [w, x, y, z] to rotation matrix.
    
    Args:
        q: Quaternion of shape (..., 4) with [w, x, y, z] convention
        
    Returns:
        Rotation matrix of shape (..., 3, 3)
    """
    # Normalize quaternion
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True)
    
    w, x, y, z = q.unbind(-1)
    
    # Compute rotation matrix elements
    R = torch.stack([
        torch.stack([1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)], dim=-1),
        torch.stack([2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)], dim=-1),
        torch.stack([2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)], dim=-1)
    ], dim=-2)
    
    return R


def apply_transformation(positions: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """
    Apply rotation and translation to positions.
    
    Args:
        positions: Positions of shape (..., N, 3)
        rotation: Rotation matrix of shape (..., 3, 3)
        translation: Translation vector of shape (..., 3)
        
    Returns:
        Transformed positions of shape (..., N, 3)
    """
    # Apply rotation: R @ positions^T -> (..., 3, N) -> (..., N, 3)
    rotated = torch.matmul(rotation, positions.transpose(-2, -1)).transpose(-2, -1)
    
    # Add translation
    if translation.dim() == positions.dim() - 1:
        translation = translation.unsqueeze(-2)  # Add point dimension
    
    return rotated + translation
