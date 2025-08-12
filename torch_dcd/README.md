### Torch Polytope Proximity and Distance

This module provides a PyTorch implementation of the LP-based polytope proximity solver and differentiable minimum distance/penetration magnitude between two convex polytopes. It mirrors the JAX reference in `dpax/polytopes.py` and uses a primal–dual interior-point (PDIP) method.

## API

- `problem_matrices(A1, b1, r1, Q1, A2, b2, r2, Q2)`
  - Build LP terms `(c, G, h)` for the proximity problem.
  - Inputs:
    - `A1, b1`: H-rep of polytope 1 in its body frame (`A1 x_B ≤ b1`)
    - `r1`: world position of polytope 1 (shape `(3,)`)
    - `Q1`: orientation of polytope 1, either a rotation matrix `(3,3)` or quaternion `[w, x, y, z]`
    - `A2, b2, r2, Q2`: same for polytope 2
  - Orientation semantics: `Q1`, `Q2` represent the rotation from body to world (W_R_B). Internally, we use `Q^T` to map world vectors into the body frame (consistent with the JAX implementation).

- `proximity(A1, b1, r1, Q1, A2, b2, r2, Q2)`
  - Solve the LP and return the optimal dilation scalar `alpha`.
  - Lower `alpha` implies greater overlap; `alpha ≤ 1` indicates collision.

- `proximity_autograd(A1, b1, r1, Q1, A2, b2, r2, Q2)`
  - Same as `proximity`, but with custom autograd backward based on the envelope theorem.
  - Gradients w.r.t. `(A, b, r, Q)` are computed from the optimal Lagrangian, without differentiating through PDIP iterations. If `Q` is a quaternion, gradients chain through the differentiable quaternion→matrix conversion.

- `min_distance(A1, b1, r1, Q1, A2, b2, r2, Q2)`
  - Compute `d = || r1 - r2 + (r2 - r1)/alpha ||`, where `alpha = proximity_autograd(...)`.
  - Interpreted as penetration depth when overlapping, or minimum separation distance when disjoint.
  - Fully differentiable w.r.t. inputs.

## Quick Start

```python
import torch
from torch_impl import proximity, min_distance

# Use float64 for robustness
torch.set_default_dtype(torch.float64)

def cube_hrep(a, device=None, dtype=torch.float64):
    I = torch.eye(3, device=device, dtype=dtype)
    A = torch.cat([I, -I], dim=0)  # (6,3)
    b = a * torch.ones(6, device=device, dtype=dtype)
    return A, b

# Two identical axis-aligned cubes with half-extent a
device = torch.device('cpu')
a = 1.0
A, b = cube_hrep(a, device=device)

r1 = torch.tensor([0.0, 0.0, 0.0], device=device)
r2 = torch.tensor([1.0, 0.0, 0.0], device=device)

# Orientation can be matrix (3x3) or quaternion [w,x,y,z]
R = torch.eye(3, device=device)
q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

# Either of these pairs is accepted:
alpha = proximity(A, b, r1, R, A, b, r2, R)
# alpha = proximity(A, b, r1, q, A, b, r2, q)
print('alpha =', float(alpha))  # <=1 => collision

# Minimum distance / penetration depth
r1 = r1.requires_grad_()
r2 = r2.requires_grad_()
d = min_distance(A, b, r1, R, A, b, r2, R)
print('d =', float(d))

# Gradients w.r.t positions
d.backward()
print('dd/dr1 =', r1.grad)
print('dd/dr2 =', r2.grad)
```

## Notes

- Dtypes/devices: The solver uses dense linear algebra; CPU float64 is recommended for stability. You can switch to float32/GPUs if desired.
- Autograd: Backward relies on the envelope theorem; gradients are stable even for large problems. For rotations, gradients are provided w.r.t. the rotation matrix; if you pass a quaternion input, gradients will chain back into the quaternion via the internal conversion.
- Non-smoothness: The LP objective is piecewise-smooth due to active-set changes. Directional finite differences are recommended when validating rotational gradients (e.g., via small quaternion deltas) rather than naive parameter perturbations.

## Reference

- PDIP reduction and predictor–corrector follow the same approach as the JAX reference (see `dpax/pdip_solver.py` and `dpax/polytopes.py`).
- Original method background: DCOL (https://arxiv.org/abs/2207.00669) for problem formulation; Boyd et al. codegen notes and Nocedal & Wright for PDIP details.

