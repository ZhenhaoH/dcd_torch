# Differentiable Shape Matching in PyTorch

This module implements the shape matching algorithm from **Müller et al. "Meshless Deformations Based on Shape Matching"** with full differentiability support. The algorithm finds the optimal rotation and translation that best matches deformed particle positions to their rest configuration.

## Main Functions

- `optimal_rotation_translation(rest_positions, deformed_positions, masses=None)`
  - Computes optimal rotation matrix and translation vector
  - Inputs:
    - `rest_positions`: Rest positions of shape `(..., N, 3)`  
    - `deformed_positions`: Current positions of shape `(..., N, 3)`
    - `masses`: Optional masses of shape `(..., N)`, uniform if None
  - Returns: `(rotation_matrix, translation_vector)` of shapes `(..., 3, 3)` and `(..., 3)`

- `shape_matching_transform(rest_positions, deformed_positions, masses=None, stiffness=1.0)`
  - Applies shape matching to compute goal positions
  - `stiffness`: Controls how much to match (0=no change, 1=full match)
  - Returns: Goal positions of shape `(..., N, 3)`

## Quick Start

```python
import torch
from shape_match import optimal_rotation_translation, shape_matching_transform

torch.set_default_dtype(torch.float64)

# Define rest configuration (e.g., a square)
rest_positions = torch.tensor([
    [-1., -1., 0.],
    [1., -1., 0.], 
    [1., 1., 0.],
    [-1., 1., 0.]
])

# Deformed configuration (rotated and translated)
deformed_positions = torch.tensor([
    [0., 0., 0.],
    [1., 1., 0.],
    [0., 2., 0.],
    [-1., 1., 0.]
])

# Find optimal transformation
rotation, translation = optimal_rotation_translation(rest_positions, deformed_positions)
print('Rotation matrix:', rotation)
print('Translation:', translation)

# Compute goal positions for shape matching
goal_positions = shape_matching_transform(rest_positions, deformed_positions, stiffness=0.8)
print('Goal positions:', goal_positions)

# With gradients for optimization
deformed_positions.requires_grad_(True)
loss = torch.mean((goal_positions - deformed_positions) ** 2)
loss.backward()
print('Gradients:', deformed_positions.grad)
```

## Batched Processing

```python
# Multiple objects simultaneously  
batch_size, N = 10, 8
rest_batch = torch.randn(batch_size, N, 3)
deformed_batch = torch.randn(batch_size, N, 3)

# Batch shape matching
rotations, translations = optimal_rotation_translation(rest_batch, deformed_batch)
# rotations: (10, 3, 3), translations: (10, 3)

goal_batch = shape_matching_transform(rest_batch, deformed_batch)
# goal_batch: (10, 8, 3)
```

## With Masses

```python
# Weighted shape matching
masses = torch.tensor([1.0, 2.0, 1.0, 0.5])  # Different particle masses

rotation, translation = optimal_rotation_translation(
    rest_positions, deformed_positions, masses=masses
)

goal_positions = shape_matching_transform(
    rest_positions, deformed_positions, masses=masses, stiffness=1.0
)
```

## Reference

- Müller, M., Heidelberger, B., Teschner, M., & Gross, M. (2005). "Meshless deformations based on shape matching." ACM Transactions on Graphics, 24(3), 471-478.
