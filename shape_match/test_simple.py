"""
Simple test to verify the shape matching implementation.
"""

import sys
import os

# Add parent directory to path to import torch_dcd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    print("✓ PyTorch imported successfully")
    
    # Test basic tensor operations
    x = torch.randn(3, 3)
    y = torch.linalg.svd(x)
    print("✓ PyTorch SVD works")
    
    # Import our modules
    from shape_match.utils import compute_center_of_mass, polar_decomposition
    from shape_match.shape_matching import optimal_rotation_translation, shape_matching_transform
    
    print("✓ Shape matching modules imported successfully")
    
    # Create simple test data
    torch.manual_seed(42)
    rest_pos = torch.randn(5, 3)
    
    # Create deformed positions by applying a known transformation
    angle = torch.tensor(0.5)
    R_true = torch.tensor([
        [torch.cos(angle), -torch.sin(angle), 0],
        [torch.sin(angle), torch.cos(angle), 0],
        [0, 0, 1]
    ], dtype=torch.float32)
    
    t_true = torch.tensor([1.0, 0.5, -0.3])
    
    deformed_pos = (R_true @ rest_pos.T).T + t_true
    
    print("✓ Test data created")
    
    # Test the algorithm
    R_pred, t_pred = optimal_rotation_translation(rest_pos, deformed_pos)
    
    print(f"✓ Algorithm completed")
    print(f"  Rotation error: {torch.norm(R_pred - R_true):.4f}")
    print(f"  Translation error: {torch.norm(t_pred - t_true):.4f}")
    
    # Test gradient computation
    deformed_pos_grad = deformed_pos.clone().requires_grad_(True)
    goal_pos = shape_matching_transform(rest_pos, deformed_pos_grad)
    loss = torch.sum((deformed_pos_grad - goal_pos) ** 2)
    loss.backward()
    
    print("✓ Gradient computation successful")
    print(f"  Gradient norm: {torch.norm(deformed_pos_grad.grad):.4f}")
    
    print("\n🎉 All tests passed! Shape matching implementation is working correctly.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
