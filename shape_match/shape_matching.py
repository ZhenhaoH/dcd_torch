"""
Based on Müller et al. "Meshless Deformations Based on Shape Matching" Section 3.3: Shape Matching
This implements the algorithm for finding optimal rotation and translation to match deformed positions to rest positions.
"""

import torch
from typing import Tuple, Optional
from .utils import compute_center_of_mass, polar_decomposition, apply_transformation


def optimal_rotation_translation(
    rest_positions: torch.Tensor,
    deformed_positions: torch.Tensor,
    masses: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        rest_positions: Rest positions of shape (..., N, 3)
        deformed_positions: Current deformed positions of shape (..., N, 3)
        masses: Optional masses of shape (..., N). If None, uniform masses assumed.
        
    Returns:
        Tuple of (rotation_matrix, translation_vector):
        - rotation_matrix: Optimal rotation matrix of shape (..., 3, 3)
        - translation_vector: Optimal translation vector of shape (..., 3)
    """
    # Step 1: Compute centers of mass
    rest_com = compute_center_of_mass(rest_positions, masses)
    deformed_com = compute_center_of_mass(deformed_positions, masses)
    
    # Step 2: Compute relative positions (subtract center of mass)
    rest_rel = rest_positions - rest_com.unsqueeze(-2)
    deformed_rel = deformed_positions - deformed_com.unsqueeze(-2)
    
    # Step 3: Compute the covariance matrix A_pq
    if masses is not None:
        # Weighted covariance matrix
        masses_expanded = masses.unsqueeze(-1)  # (..., N, 1)
        weighted_deformed = deformed_rel * masses_expanded  # (..., N, 3)
        # A_pq = sum_i m_i * p_i * q_i^T where p_i is deformed, q_i is rest
        A_pq = torch.matmul(weighted_deformed.transpose(-2, -1), rest_rel)  # (..., 3, 3)
    else:
        # Unweighted covariance matrix
        A_pq = torch.matmul(deformed_rel.transpose(-2, -1), rest_rel)  # (..., 3, 3)
    
    # Step 4: Extract optimal rotation using polar decomposition
    # A_pq = R * S where R is rotation, S is symmetric positive definite
    R, S = polar_decomposition(A_pq)
    
    # Step 5: Compute optimal translation
    # Translation aligns the centers of mass after rotation
    translation = deformed_com - torch.matmul(R, rest_com.unsqueeze(-1)).squeeze(-1)
    
    return R, translation


def shape_matching_transform(
    rest_positions: torch.Tensor,
    deformed_positions: torch.Tensor,
    masses: Optional[torch.Tensor] = None,
    stiffness: float = 1.0
) -> torch.Tensor:
    """
    Args:
        rest_positions: Rest positions of shape (..., N, 3)
        deformed_positions: Current positions of shape (..., N, 3)
        masses: Optional masses of shape (..., N)
        stiffness: Stiffness parameter in [0, 1] controlling how much to match
        
    Returns:
        Goal positions of shape (..., N, 3)
    """
    # Find optimal transformation
    R, t = optimal_rotation_translation(rest_positions, deformed_positions, masses)
    
    # Apply transformation to rest positions to get goal positions
    rest_com = compute_center_of_mass(rest_positions, masses)
    rest_rel = rest_positions - rest_com.unsqueeze(-2)
    
    # Transform rest positions: R * rest_rel + (rest_com + t)
    goal_positions = apply_transformation(rest_rel, R, rest_com + t)
    
    # Blend with current positions based on stiffness
    if stiffness < 1.0:
        goal_positions = stiffness * goal_positions + (1.0 - stiffness) * deformed_positions
    
    return goal_positions


def weighted_shape_matching(
    rest_positions: torch.Tensor,
    deformed_positions: torch.Tensor,
    masses: torch.Tensor,
    region_weights: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Weighted shape matching with optional region-based weighting.
    
    Args:
        rest_positions: Rest positions of shape (..., N, 3)
        deformed_positions: Current positions of shape (..., N, 3)
        masses: Masses of shape (..., N)
        region_weights: Optional region weights of shape (..., N)
        
    Returns:
        Tuple of (goal_positions, rotation_matrix, translation_vector)
    """
    effective_masses = masses
    if region_weights is not None:
        effective_masses = masses * region_weights
    
    R, t = optimal_rotation_translation(rest_positions, deformed_positions, effective_masses)
    goal_positions = shape_matching_transform(rest_positions, deformed_positions, effective_masses)
    
    return goal_positions, R, t


def shape_matching_loss(
    rest_positions: torch.Tensor,
    deformed_positions: torch.Tensor,
    masses: Optional[torch.Tensor] = None,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute shape matching loss between rest and deformed configurations.
    
    Args:
        rest_positions: Rest positions of shape (..., N, 3)
        deformed_positions: Current positions of shape (..., N, 3)
        masses: Optional masses of shape (..., N)
        reduction: Reduction method ('mean', 'sum', 'none')
        
    Returns:
        Loss tensor
    """
    # Compute goal positions
    goal_positions = shape_matching_transform(rest_positions, deformed_positions, masses)
    
    # Compute squared distance loss
    squared_distances = torch.sum((deformed_positions - goal_positions) ** 2, dim=-1)
    
    if masses is not None:
        squared_distances = squared_distances * masses
    
    if reduction == 'mean':
        return torch.mean(squared_distances)
    elif reduction == 'sum':
        return torch.sum(squared_distances)
    else:
        return squared_distances


class ShapeMatchingConstraint(torch.nn.Module):
    """
    PyTorch module for shape matching constraints.
    """
    
    def __init__(self, rest_positions: torch.Tensor, masses: Optional[torch.Tensor] = None, stiffness: float = 1.0):
        """
        Initialize shape matching constraint.
        
        Args:
            rest_positions: Rest positions of shape (N, 3)
            masses: Optional masses of shape (N,)
            stiffness: Stiffness parameter in [0, 1]
        """
        super().__init__()
        self.register_buffer('rest_positions', rest_positions)
        if masses is not None:
            self.register_buffer('masses', masses)
        else:
            self.masses = None
        self.stiffness = stiffness
    
    def forward(self, deformed_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply shape matching constraint.
        
        Args:
            deformed_positions: Current positions of shape (..., N, 3)
            
        Returns:
            Goal positions of shape (..., N, 3)
        """
        return shape_matching_transform(
            self.rest_positions, 
            deformed_positions, 
            self.masses, 
            self.stiffness
        )
    
    def compute_loss(self, deformed_positions: torch.Tensor) -> torch.Tensor:
        """
        Compute shape matching loss.
        
        Args:
            deformed_positions: Current positions of shape (..., N, 3)
            
        Returns:
            Loss tensor
        """
        return shape_matching_loss(
            self.rest_positions,
            deformed_positions,
            self.masses
        )
