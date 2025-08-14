"""
Example usage of the differentiable shape matching algorithm.

This demonstrates how to use the shape matching implementation
for various scenarios including gradient computation.
"""

import torch
import numpy as np
from shape_matching import (
    optimal_rotation_translation,
    shape_matching_transform,
    shape_matching_loss,
    ShapeMatchingConstraint
)


def create_test_data(n_points: int = 10, batch_size: int = 1) -> tuple:
    """Create test data for shape matching."""
    torch.manual_seed(42)
    
    # Create rest positions (e.g., a simple 3D shape)
    if batch_size == 1:
        rest_positions = torch.randn(n_points, 3)
    else:
        rest_positions = torch.randn(batch_size, n_points, 3)
    
    # Create a known transformation
    angle = torch.tensor(0.3)  # 0.3 radians rotation
    axis = torch.tensor([0.0, 0.0, 1.0])  # Rotate around z-axis
    
    # Create rotation matrix for z-axis rotation
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    R_true = torch.tensor([
        [cos_a, -sin_a, 0.0],
        [sin_a,  cos_a, 0.0],
        [0.0,    0.0,   1.0]
    ])
    
    t_true = torch.tensor([0.5, -0.3, 0.1])  # Translation
    
    # Apply transformation to get deformed positions
    if batch_size == 1:
        deformed_positions = (R_true @ rest_positions.T).T + t_true
    else:
        R_true_batch = R_true.unsqueeze(0).expand(batch_size, -1, -1)
        t_true_batch = t_true.unsqueeze(0).expand(batch_size, -1)
        deformed_positions = torch.matmul(R_true_batch, rest_positions.transpose(-2, -1)).transpose(-2, -1)
        deformed_positions = deformed_positions + t_true_batch.unsqueeze(-2)
    
    return rest_positions, deformed_positions, R_true, t_true


def test_basic_shape_matching():
    """Test basic shape matching functionality."""
    print("Testing basic shape matching...")
    
    rest_pos, deformed_pos, R_true, t_true = create_test_data(n_points=20)
    
    # Find optimal transformation
    R_pred, t_pred = optimal_rotation_translation(rest_pos, deformed_pos)
    
    print(f"True rotation matrix:\n{R_true}")
    print(f"Predicted rotation matrix:\n{R_pred}")
    print(f"Rotation error: {torch.norm(R_pred - R_true):.6f}")
    
    print(f"True translation: {t_true}")
    print(f"Predicted translation: {t_pred}")
    print(f"Translation error: {torch.norm(t_pred - t_true):.6f}")


def test_gradient_computation():
    """Test gradient computation through shape matching."""
    print("\nTesting gradient computation...")
    
    rest_pos, deformed_pos, _, _ = create_test_data(n_points=15)
    
    # Make deformed positions require gradients
    deformed_pos.requires_grad_(True)
    
    # Compute shape matching loss
    loss = shape_matching_loss(rest_pos, deformed_pos)
    
    print(f"Shape matching loss: {loss.item():.6f}")
    
    # Compute gradients
    loss.backward()
    
    print(f"Gradient shape: {deformed_pos.grad.shape}")
    print(f"Gradient norm: {torch.norm(deformed_pos.grad):.6f}")
    print(f"Max gradient: {torch.max(torch.abs(deformed_pos.grad)):.6f}")


def test_batched_processing():
    """Test batched shape matching."""
    print("\nTesting batched processing...")
    
    batch_size = 5
    rest_pos, deformed_pos, _, _ = create_test_data(n_points=10, batch_size=batch_size)
    
    # Compute shape matching for entire batch
    R_batch, t_batch = optimal_rotation_translation(rest_pos, deformed_pos)
    
    print(f"Batch size: {batch_size}")
    print(f"Rotation matrices shape: {R_batch.shape}")
    print(f"Translation vectors shape: {t_batch.shape}")
    
    # Compute goal positions
    goal_pos = shape_matching_transform(rest_pos, deformed_pos)
    print(f"Goal positions shape: {goal_pos.shape}")


def test_weighted_shape_matching():
    """Test weighted shape matching with masses."""
    print("\nTesting weighted shape matching...")
    
    rest_pos, deformed_pos, _, _ = create_test_data(n_points=12)
    
    # Create random masses
    masses = torch.rand(12) + 0.1  # Avoid zero masses
    
    # Compare unweighted vs weighted
    R_unweighted, t_unweighted = optimal_rotation_translation(rest_pos, deformed_pos)
    R_weighted, t_weighted = optimal_rotation_translation(rest_pos, deformed_pos, masses)
    
    print(f"Unweighted rotation determinant: {torch.det(R_unweighted):.6f}")
    print(f"Weighted rotation determinant: {torch.det(R_weighted):.6f}")
    print(f"Rotation difference: {torch.norm(R_weighted - R_unweighted):.6f}")
    print(f"Translation difference: {torch.norm(t_weighted - t_unweighted):.6f}")


def test_pytorch_module():
    """Test PyTorch module interface."""
    print("\nTesting PyTorch module interface...")
    
    rest_pos, deformed_pos, _, _ = create_test_data(n_points=8)
    
    # Create shape matching constraint module
    constraint = ShapeMatchingConstraint(rest_pos, stiffness=0.8)
    
    # Test forward pass
    goal_pos = constraint(deformed_pos)
    print(f"Goal positions shape: {goal_pos.shape}")
    
    # Test loss computation
    loss = constraint.compute_loss(deformed_pos)
    print(f"Module loss: {loss.item():.6f}")
    
    # Test with gradients
    deformed_pos.requires_grad_(True)
    loss = constraint.compute_loss(deformed_pos)
    loss.backward()
    print(f"Gradient computed successfully: {deformed_pos.grad is not None}")


def test_optimization_example():
    """Demonstrate using shape matching in an optimization loop."""
    print("\nTesting optimization example...")
    
    rest_pos, target_pos, _, _ = create_test_data(n_points=10)
    
    # Initialize deformed positions with noise
    deformed_pos = rest_pos + 0.1 * torch.randn_like(rest_pos)
    deformed_pos.requires_grad_(True)
    
    # Create optimizer
    optimizer = torch.optim.Adam([deformed_pos], lr=0.01)
    
    print("Initial loss:", shape_matching_loss(rest_pos, deformed_pos).item())
    
    # Optimization loop
    for i in range(50):
        optimizer.zero_grad()
        loss = shape_matching_loss(rest_pos, deformed_pos)
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f"Step {i}: Loss = {loss.item():.6f}")
    
    print("Final loss:", shape_matching_loss(rest_pos, deformed_pos).item())


if __name__ == "__main__":
    print("=== Differentiable Shape Matching Tests ===\n")
    
    test_basic_shape_matching()
    test_gradient_computation()
    test_batched_processing()
    test_weighted_shape_matching()
    test_pytorch_module()
    test_optimization_example()
    
    print("\n=== All tests completed! ===")
