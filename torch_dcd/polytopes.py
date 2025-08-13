import torch
from .solver import solve_lp


def _to_rotation_matrix(R_or_q: torch.Tensor) -> torch.Tensor:
    """Return 3x3 rotation matrix from matrix or quaternion input.

    Rotation is from body frame to world frame.

    - If input shape is (3,3), assume valid rotation matrix and return it.
    - If input is length-4 (quaternion [w, x, y, z]), normalize and convert to 3x3.
    """
    if R_or_q.dim() == 2 and R_or_q.shape == (3, 3):
        return R_or_q

    q = R_or_q.reshape(-1)
    if q.numel() != 4:
        raise ValueError("Rotation argument must be 3x3 matrix or length-4 quaternion [w,x,y,z].")
    q = q / torch.linalg.norm(q)
    w, x, y, z = q
    R = torch.stack([
        torch.stack([1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)]),
        torch.stack([2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)]),
        torch.stack([2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)])
    ])
    return R.to(dtype=R_or_q.dtype, device=R_or_q.device)


def problem_matrices(A1: torch.Tensor, b1: torch.Tensor, r1: torch.Tensor, Q1: torch.Tensor,
                     A2: torch.Tensor, b2: torch.Tensor, r2: torch.Tensor, Q2: torch.Tensor):
    """
    Build LP terms (c, G, h) for the polytope proximity problem.
    """
    device = A1.device
    dtype = A1.dtype

    c = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=dtype)

    R1 = _to_rotation_matrix(Q1)
    R2 = _to_rotation_matrix(Q2)

    G_top = torch.cat([A1 @ R1.t(), -b1.reshape(-1, 1)], dim=1)
    G_bot = torch.cat([A2 @ R2.t(), -b2.reshape(-1, 1)], dim=1)
    G = torch.cat([G_top, G_bot], dim=0)

    h_top = A1 @ (R1.t() @ r1)
    h_bot = A2 @ (R2.t() @ r2)
    h = torch.cat([h_top, h_bot], dim=0)

    return c, G, h


def proximity(A1: torch.Tensor, b1: torch.Tensor, r1: torch.Tensor, Q1: torch.Tensor,
              A2: torch.Tensor, b2: torch.Tensor, r2: torch.Tensor, Q2: torch.Tensor) -> torch.Tensor:
    c, G, h = problem_matrices(A1, b1, r1, Q1, A2, b2, r2, Q2)
    x, s, z = solve_lp(c, G, h)
    return x[3]


class _ProximityFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                A1: torch.Tensor, b1: torch.Tensor, r1: torch.Tensor, Q1: torch.Tensor,
                A2: torch.Tensor, b2: torch.Tensor, r2: torch.Tensor, Q2: torch.Tensor):
        c, G, h = problem_matrices(A1, b1, r1, Q1, A2, b2, r2, Q2)
        x, s, z = solve_lp(c, G, h)
        ctx.save_for_backward(x, s, z, A1, b1, r1, Q1, A2, b2, r2, Q2)
        ctx.n1 = A1.shape[0]
        return x[3]

    @staticmethod
    def backward(ctx, grad_alpha):
        x, s, z, A1, b1, r1, Q1, A2, b2, r2, Q2 = ctx.saved_tensors
        n1 = ctx.n1
        alpha = x[3]
        z1 = z[:n1]
        z2 = z[n1:]
        
        v1 = x[:3] - r1
        v2 = x[:3] - r2

        # Envelope theorem gradients
        gA1 = z1.unsqueeze(1) * (Q1.t() @ v1).unsqueeze(0)
        gb1 = -alpha * z1
        gr1 = -Q1 @ (A1.t() @ z1)
        gQ1 = torch.outer(v1, A1.t() @ z1)

        gA2 = z2.unsqueeze(1) * (Q2.t() @ v2).unsqueeze(0)
        gb2 = -alpha * z2
        gr2 = -Q2 @ (A2.t() @ z2)
        gQ2 = torch.outer(v2, A2.t() @ z2)

        # Chain rule
        gA1 = grad_alpha * gA1
        gb1 = grad_alpha * gb1
        gr1 = grad_alpha * gr1
        gQ1 = grad_alpha * gQ1

        gA2 = grad_alpha * gA2
        gb2 = grad_alpha * gb2
        gr2 = grad_alpha * gr2
        gQ2 = grad_alpha * gQ2

        return gA1, gb1, gr1, gQ1, gA2, gb2, gr2, gQ2


def proximity_autograd(A1: torch.Tensor, b1: torch.Tensor, r1: torch.Tensor, Q1: torch.Tensor,
                       A2: torch.Tensor, b2: torch.Tensor, r2: torch.Tensor, Q2: torch.Tensor) -> torch.Tensor:
    """Autograd-enabled proximity using envelope-theorem backward."""
    # Convert quaternion inputs (if provided) to rotation matrices so gradients
    # propagate through the conversion into the quaternion inputs.
    Q1m = _to_rotation_matrix(Q1)
    Q2m = _to_rotation_matrix(Q2)
    return _ProximityFn.apply(A1, b1, r1, Q1m, A2, b2, r2, Q2m)


def min_distance(A1: torch.Tensor, b1: torch.Tensor, r1: torch.Tensor, Q1: torch.Tensor,
                 A2: torch.Tensor, b2: torch.Tensor, r2: torch.Tensor, Q2: torch.Tensor) -> torch.Tensor:
    """Minimum distance or penetration magnitude between two polytopes.

    d = || r1 - r2 + (r2 - r1)/alpha || where alpha is the optimal proximity dilation.
    This equals penetration depth when overlapping, and separation distance when disjoint.
    """
    alpha = proximity_autograd(A1, b1, r1, Q1, A2, b2, r2, Q2)
    return torch.linalg.norm(r1 - r2 + (r2 - r1) / alpha)


# =========================
# Time-of-Impact (TOI) LP
# =========================

def toi_problem_matrices(
    A1: torch.Tensor, b1: torch.Tensor, r1: torch.Tensor, Q1: torch.Tensor, v1: torch.Tensor,
    A2: torch.Tensor, b2: torch.Tensor, r2: torch.Tensor, Q2: torch.Tensor, v2: torch.Tensor,
    dt: torch.Tensor | float,
):
    """
    Build LP terms (c, G, h) for the time-of-impact problem with linear motion:
        min  [0,0,0,1]^T [x; tau]
        s.t. A_i Q_i^T (x - r_i - v_i tau) <= b_i,  i=1,2
             0 <= tau <= dt
    Returns (c, G, h).
    """
    device = A1.device
    dtype = A1.dtype

    c = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=dtype)

    R1 = _to_rotation_matrix(Q1)
    R2 = _to_rotation_matrix(Q2)

    # Blocks for each polytope
    # G_i = [A_i R_i^T,  -A_i R_i^T v_i],  h_i = A_i R_i^T r_i + b_i
    G1 = torch.cat([A1 @ R1.t(), -(A1 @ R1.t() @ v1).reshape(-1, 1)], dim=1)
    G2 = torch.cat([A2 @ R2.t(), -(A2 @ R2.t() @ v2).reshape(-1, 1)], dim=1)
    h1 = A1 @ (R1.t() @ r1) + b1
    h2 = A2 @ (R2.t() @ r2) + b2

    # Time bounds: -tau <= 0,  tau <= dt
    G_time = torch.stack([
        torch.tensor([0.0, 0.0, 0.0, -1.0], device=device, dtype=dtype),
        torch.tensor([0.0, 0.0, 0.0,  1.0], device=device, dtype=dtype),
    ], dim=0)
    if not torch.is_tensor(dt):
        dt = torch.tensor(dt, device=device, dtype=dtype)
    h_time = torch.stack([torch.tensor(0.0, device=device, dtype=dtype), dt], dim=0)

    G = torch.cat([G1, G2, G_time], dim=0)
    h = torch.cat([h1, h2, h_time], dim=0)

    return c, G, h


def time_of_impact(
    A1: torch.Tensor, b1: torch.Tensor, r1: torch.Tensor, Q1: torch.Tensor, v1: torch.Tensor,
    A2: torch.Tensor, b2: torch.Tensor, r2: torch.Tensor, Q2: torch.Tensor, v2: torch.Tensor,
    dt: torch.Tensor | float,
) -> torch.Tensor:
    c, G, h = toi_problem_matrices(A1, b1, r1, Q1, v1, A2, b2, r2, Q2, v2, dt)
    y, s, z = solve_lp(c, G, h)
    return y[3]


class _TOIFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                A1: torch.Tensor, b1: torch.Tensor, r1: torch.Tensor, Q1: torch.Tensor, v1: torch.Tensor,
                A2: torch.Tensor, b2: torch.Tensor, r2: torch.Tensor, Q2: torch.Tensor, v2: torch.Tensor,
                dt: torch.Tensor):
        c, G, h = toi_problem_matrices(A1, b1, r1, Q1, v1, A2, b2, r2, Q2, v2, dt)
        y, s, z = solve_lp(c, G, h)
        ctx.save_for_backward(y, s, z, A1, b1, r1, Q1, v1, A2, b2, r2, Q2, v2, dt)
        ctx.n1 = A1.shape[0]
        ctx.n2 = A2.shape[0]
        return y[3]

    @staticmethod
    def backward(ctx, grad_tau):
        y, s, z, A1, b1, r1, Q1, v1, A2, b2, r2, Q2, v2, dt = ctx.saved_tensors
        n1 = ctx.n1
        # Partition duals
        z1 = z[:n1]
        z2 = z[n1:n1 + ctx.n2]
        z_tau_minus = z[-2]
        z_tau_plus = z[-1]

        x = y[:3]
        tau = y[3]

        # Common terms
        u1 = A1.t() @ z1
        u2 = A2.t() @ z2
        w1 = x - r1 - v1 * tau
        w2 = x - r2 - v2 * tau

        # Gradients via envelope theorem
        gA1 = z1.unsqueeze(1) * (Q1.t() @ w1).unsqueeze(0)
        gb1 = -z1
        gr1 = -Q1 @ u1
        gQ1 = torch.outer(w1, u1)
        gv1 = -tau * (Q1 @ u1)

        gA2 = z2.unsqueeze(1) * (Q2.t() @ w2).unsqueeze(0)
        gb2 = -z2
        gr2 = -Q2 @ u2
        gQ2 = torch.outer(w2, u2)
        gv2 = -tau * (Q2 @ u2)

        gdt = -z_tau_plus

        # Chain rule with upstream gradient
        gA1 = grad_tau * gA1
        gb1 = grad_tau * gb1
        gr1 = grad_tau * gr1
        gQ1 = grad_tau * gQ1
        gv1 = grad_tau * gv1

        gA2 = grad_tau * gA2
        gb2 = grad_tau * gb2
        gr2 = grad_tau * gr2
        gQ2 = grad_tau * gQ2
        gv2 = grad_tau * gv2

        gdt = grad_tau * gdt

        return gA1, gb1, gr1, gQ1, gv1, gA2, gb2, gr2, gQ2, gv2, gdt


def time_of_impact_autograd(
    A1: torch.Tensor, b1: torch.Tensor, r1: torch.Tensor, Q1: torch.Tensor, v1: torch.Tensor,
    A2: torch.Tensor, b2: torch.Tensor, r2: torch.Tensor, Q2: torch.Tensor, v2: torch.Tensor,
    dt: torch.Tensor | float,
) -> torch.Tensor:
    """Autograd-enabled TOI using envelope-theorem backward."""
    Q1m = _to_rotation_matrix(Q1)
    Q2m = _to_rotation_matrix(Q2)
    if not torch.is_tensor(dt):
        dt = torch.as_tensor(dt, dtype=A1.dtype, device=A1.device)
    return _TOIFn.apply(A1, b1, r1, Q1m, v1, A2, b2, r2, Q2m, v2, dt)
