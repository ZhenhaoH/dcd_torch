"""
Differentiable Shape Matching with PyTorch

Implementation of the shape matching algorithm from Müller et al. "Position Based Dynamics"
Section 3.3: Shape Matching

This module provides differentiable implementations that compute optimal rotation and translation
to match deformed positions to rest positions, with gradients available w.r.t. input nodes.
"""

from .shape_matching import (
    shape_matching_transform,
    shape_matching_loss,
    optimal_rotation_translation,
    weighted_shape_matching,
)

from .utils import (
    compute_center_of_mass,
    polar_decomposition,
    rotation_matrix_to_quaternion,
    quaternion_to_rotation_matrix,
)

__version__ = "1.0.0"

__all__ = [
    'shape_matching_transform',
    'shape_matching_loss', 
    'optimal_rotation_translation',
    'weighted_shape_matching',
    'compute_center_of_mass',
    'polar_decomposition',
    'rotation_matrix_to_quaternion',
    'quaternion_to_rotation_matrix',
]
