"""
Advanced example showing shape matching in optimization and physics simulation.
"""

import torch
import torch.nn as nn
from typing import List, Tuple
from shape_matching import ShapeMatchingConstraint, shape_matching_loss


class DeformableObject(nn.Module):
    """
    A deformable object that maintains its shape using shape matching constraints.
    """
    
    def __init__(self, rest_positions: torch.Tensor, masses: torch.Tensor = None, stiffness: float = 0.9):
        super().__init__()
        
        self.n_particles = rest_positions.shape[0]
        
        # Register rest positions and masses as buffers (non-trainable)
        self.register_buffer('rest_positions', rest_positions)
        if masses is not None:
            self.register_buffer('masses', masses)
        else:
            self.register_buffer('masses', torch.ones(self.n_particles))
            
        # Current positions (trainable parameters)
        self.positions = nn.Parameter(rest_positions.clone())
        
        # Velocities (for physics simulation)
        self.register_buffer('velocities', torch.zeros_like(rest_positions))
        
        # Shape matching constraint
        self.shape_constraint = ShapeMatchingConstraint(rest_positions, masses, stiffness)
        
        # Physical parameters
        self.damping = 0.99
        self.dt = 0.01
    
    def apply_external_forces(self, forces: torch.Tensor):
        """Apply external forces to the object."""
        # F = ma, so acceleration = F / m
        acceleration = forces / self.masses.unsqueeze(-1)
        
        # Update velocities
        self.velocities += acceleration * self.dt
        
        # Apply damping
        self.velocities *= self.damping
    
    def apply_shape_constraints(self):
        """Apply shape matching constraints."""
        # Get goal positions from shape matching
        goal_positions = self.shape_constraint(self.positions)
        
        # Move towards goal positions (constraint force)
        constraint_force = (goal_positions - self.positions) / (self.dt ** 2)
        constraint_acceleration = constraint_force / self.masses.unsqueeze(-1)
        
        # Update velocities with constraint forces
        self.velocities += constraint_acceleration * self.dt
    
    def step(self, external_forces: torch.Tensor = None):
        """Perform one simulation step."""
        # Apply external forces
        if external_forces is not None:
            self.apply_external_forces(external_forces)
        
        # Apply shape matching constraints
        self.apply_shape_constraints()
        
        # Update positions
        with torch.no_grad():
            self.positions.data += self.velocities * self.dt
    
    def compute_shape_energy(self) -> torch.Tensor:
        """Compute shape matching energy."""
        return self.shape_constraint.compute_loss(self.positions)


def create_test_object(object_type: str = "cube") -> Tuple[torch.Tensor, torch.Tensor]:
    """Create test objects with different shapes."""
    
    if object_type == "cube":
        # Create a cube with 8 vertices
        positions = torch.tensor([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # Bottom face
            [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],  # Top face
        ], dtype=torch.float32)
        masses = torch.ones(8)
        
    elif object_type == "tetrahedron":
        # Create a tetrahedron
        positions = torch.tensor([
            [0, 0, 0], [1, 0, 0], [0.5, 0.866, 0], [0.5, 0.289, 0.816]
        ], dtype=torch.float32)
        masses = torch.ones(4)
        
    elif object_type == "random":
        # Random point cloud
        torch.manual_seed(42)
        positions = torch.randn(12, 3)
        masses = torch.rand(12) + 0.5  # Random masses between 0.5 and 1.5
        
    else:
        raise ValueError(f"Unknown object type: {object_type}")
    
    return positions, masses


def simulate_deformation():
    """Simulate object deformation with shape matching."""
    print("Simulating deformable object with shape matching...")
    
    # Create object
    rest_pos, masses = create_test_object("cube")
    obj = DeformableObject(rest_pos, masses, stiffness=0.8)
    
    print(f"Object created with {obj.n_particles} particles")
    
    # Simulation parameters
    n_steps = 100
    
    # Apply initial deformation
    with torch.no_grad():
        obj.positions.data += 0.2 * torch.randn_like(obj.positions.data)
    
    initial_energy = obj.compute_shape_energy().item()
    print(f"Initial shape energy: {initial_energy:.4f}")
    
    # Simulation loop
    energies = []
    for step in range(n_steps):
        # Apply some external forces (e.g., gravity)
        gravity = torch.zeros_like(obj.positions)
        gravity[:, 2] = -0.1  # Downward force
        
        # Random perturbation
        if step < 20:
            perturbation = 0.05 * torch.randn_like(obj.positions)
            obj.step(gravity + perturbation)
        else:
            obj.step(gravity)
        
        # Record energy
        energy = obj.compute_shape_energy().item()
        energies.append(energy)
        
        if step % 20 == 0:
            print(f"Step {step}: Shape energy = {energy:.4f}")
    
    final_energy = energies[-1]
    print(f"Final shape energy: {final_energy:.4f}")
    print(f"Energy reduction: {((initial_energy - final_energy) / initial_energy * 100):.1f}%")


def optimize_shape_matching():
    """Use shape matching in an optimization problem."""
    print("\nOptimizing shape matching parameters...")
    
    # Create target and source shapes
    rest_pos, masses = create_test_object("tetrahedron")
    
    # Create a known deformation
    angle = torch.tensor(0.8)
    R_target = torch.tensor([
        [torch.cos(angle), -torch.sin(angle), 0],
        [torch.sin(angle), torch.cos(angle), 0],
        [0, 0, 1]
    ])
    t_target = torch.tensor([0.5, -0.2, 0.3])
    
    target_pos = (R_target @ rest_pos.T).T + t_target
    
    # Initialize with noisy positions
    current_pos = rest_pos + 0.3 * torch.randn_like(rest_pos)
    current_pos.requires_grad_(True)
    
    # Optimizer
    optimizer = torch.optim.Adam([current_pos], lr=0.01)
    
    print("Starting optimization...")
    
    for i in range(200):
        optimizer.zero_grad()
        
        # Shape matching loss
        shape_loss = shape_matching_loss(rest_pos, current_pos, masses)
        
        # Target matching loss
        target_loss = torch.sum((current_pos - target_pos) ** 2)
        
        # Combined loss
        total_loss = shape_loss + 0.1 * target_loss
        
        total_loss.backward()
        optimizer.step()
        
        if i % 50 == 0:
            print(f"Step {i}: Total loss = {total_loss.item():.4f}, "
                  f"Shape loss = {shape_loss.item():.4f}, "
                  f"Target loss = {target_loss.item():.4f}")
    
    print("Optimization completed!")


def demonstrate_batch_processing():
    """Demonstrate batched shape matching for multiple objects."""
    print("\nDemonstrating batch processing...")
    
    batch_size = 3
    n_particles = 6
    
    # Create batch of rest positions
    torch.manual_seed(123)
    rest_batch = torch.randn(batch_size, n_particles, 3)
    
    # Create different deformations for each object in batch
    deformed_batch = rest_batch.clone()
    
    for i in range(batch_size):
        # Apply different transformations to each object
        angle = torch.tensor(i * 0.5)
        R = torch.tensor([
            [torch.cos(angle), -torch.sin(angle), 0],
            [torch.sin(angle), torch.cos(angle), 0],
            [0, 0, 1]
        ])
        t = torch.tensor([i * 0.3, i * 0.2, i * 0.1])
        
        deformed_batch[i] = (R @ rest_batch[i].T).T + t
        
        # Add some noise
        deformed_batch[i] += 0.1 * torch.randn_like(deformed_batch[i])
    
    # Batch shape matching
    from shape_matching import optimal_rotation_translation
    
    rotations, translations = optimal_rotation_translation(rest_batch, deformed_batch)
    
    print(f"Processed batch of {batch_size} objects")
    print(f"Rotation matrices shape: {rotations.shape}")
    print(f"Translations shape: {translations.shape}")
    
    # Check if rotations are valid
    for i in range(batch_size):
        det = torch.det(rotations[i])
        print(f"Object {i}: Rotation determinant = {det:.4f}")


if __name__ == "__main__":
    print("=== Advanced Shape Matching Examples ===\n")
    
    simulate_deformation()
    optimize_shape_matching()
    demonstrate_batch_processing()
    
    print("\n=== Examples completed! ===")
