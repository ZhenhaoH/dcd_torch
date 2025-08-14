# Differentiable Shape Matching with PyTorch

This module implements the shape matching algorithm from **Müller et al. "Position Based Dynamics"** Section 3.3, with full differentiability support for gradient-based optimization.

## Overview

Shape matching is a technique used in physics-based simulation to maintain the shape of deformable objects. The algorithm finds the optimal rotation and translation that best matches deformed particle positions to their rest configuration.

## Algorithm (Section 3.3)

The shape matching algorithm works as follows:

1. **Center of Mass Calculation**: Compute centers of mass for both rest and deformed configurations
2. **Relative Positions**: Subtract center of mass to get relative positions
3. **Covariance Matrix**: Compute the covariance matrix A_pq between deformed and rest positions
4. **Polar Decomposition**: Extract optimal rotation R from A_pq = R·S using polar decomposition
5. **Goal Positions**: Transform rest positions using optimal R and translation

### Mathematical Formulation

Given:
- Rest positions: `q_i ∈ ℝ³` for particles i = 1...N
- Deformed positions: `p_i ∈ ℝ³`  
- Masses: `m_i > 0`

The algorithm computes:

```
# Centers of mass
q_cm = (∑ m_i q_i) / (∑ m_i)
p_cm = (∑ m_i p_i) / (∑ m_i)

# Relative positions  
q'_i = q_i - q_cm
p'_i = p_i - p_cm

# Covariance matrix
A_pq = ∑ m_i p'_i q'_i^T

# Polar decomposition: A_pq = R·S
R = optimal_rotation(A_pq)

# Goal positions
g_i = R q'_i + p_cm
```

## Features

- **Fully Differentiable**: All operations support automatic differentiation
- **Batched Processing**: Efficient batch operations for multiple objects
- **Weighted Masses**: Support for non-uniform particle masses
- **PyTorch Integration**: Native PyTorch module interface
- **Numerical Stability**: Robust polar decomposition implementation

## Usage

### Basic Shape Matching

```python
import torch
from shape_match import optimal_rotation_translation, shape_matching_transform

# Define rest and deformed positions
rest_positions = torch.randn(100, 3)      # (N, 3)
deformed_positions = torch.randn(100, 3)  # (N, 3)

# Find optimal transformation
rotation, translation = optimal_rotation_translation(rest_positions, deformed_positions)

# Compute goal positions  
goal_positions = shape_matching_transform(rest_positions, deformed_positions)
```

### With Gradients

```python
# Enable gradients for optimization
deformed_positions.requires_grad_(True)

# Compute loss
loss = shape_matching_loss(rest_positions, deformed_positions)

# Backpropagate
loss.backward()

# Access gradients
gradients = deformed_positions.grad
```

### Batched Processing

```python
# Multiple objects simultaneously
rest_batch = torch.randn(batch_size, N, 3)
deformed_batch = torch.randn(batch_size, N, 3)

# Batch processing
rotations, translations = optimal_rotation_translation(rest_batch, deformed_batch)
# rotations: (batch_size, 3, 3)
# translations: (batch_size, 3)
```

### Weighted Shape Matching

```python
# With particle masses
masses = torch.rand(N) + 0.1  # Avoid zero masses

rotation, translation = optimal_rotation_translation(
    rest_positions, deformed_positions, masses=masses
)
```

### PyTorch Module Interface

```python
from shape_match import ShapeMatchingConstraint

# Create constraint module
constraint = ShapeMatchingConstraint(rest_positions, masses=masses, stiffness=0.8)

# Use in forward pass
goal_positions = constraint(deformed_positions)

# Compute loss
loss = constraint.compute_loss(deformed_positions)
```

## API Reference

### Core Functions

- `optimal_rotation_translation()`: Compute optimal R and t
- `shape_matching_transform()`: Apply shape matching to get goal positions  
- `shape_matching_loss()`: Compute shape matching loss
- `weighted_shape_matching()`: Shape matching with region weights

### Utility Functions

- `compute_center_of_mass()`: Compute center of mass with optional weights
- `polar_decomposition()`: Robust polar decomposition R, S = polar(A)
- `rotation_matrix_to_quaternion()`: Convert rotation matrix to quaternion
- `quaternion_to_rotation_matrix()`: Convert quaternion to rotation matrix

### PyTorch Module

- `ShapeMatchingConstraint`: PyTorch nn.Module for shape matching constraints

## Applications

This implementation is useful for:

- **Physics-based Animation**: Maintaining object shapes in deformable body simulation
- **Machine Learning**: Differentiable physics in neural networks  
- **Robotics**: Shape-based control and planning
- **Computer Graphics**: Character animation and soft body dynamics
- **Optimization**: Gradient-based shape fitting and alignment

## Mathematical Properties

- **Optimal Rotation**: The computed rotation minimizes the sum of squared distances
- **Differentiability**: All operations preserve gradients through automatic differentiation
- **Orthogonality**: Rotation matrices maintain proper orthogonal structure (det(R) = 1)
- **Batch Efficiency**: Vectorized operations for high performance

## Dependencies

- PyTorch (≥ 1.9.0)
- Python (≥ 3.7)

## Example

See `example.py` for comprehensive usage examples including:
- Basic shape matching
- Gradient computation
- Batched processing  
- Weighted constraints
- Optimization loops

## References

1. Müller, M., Heidelberger, B., Teschner, M., & Gross, M. (2005). "Meshless deformations based on shape matching." ACM Transactions on Graphics, 24(3), 471-478.

2. Sorkine-Hornung, O., & Rabinovich, M. (2017). "Least-squares rigid motion using SVD." Technical Report.
