import torch


def _ort_linesearch(x: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
    """Maximum alpha in (0, 1] such that x + alpha*dx >= 0 elementwise."""
    assert x.ndim == 1 and dx.ndim == 1 and x.shape == dx.shape
    device = x.device
    dtype = x.dtype
    cand = torch.where(dx < 0, -x / dx, torch.tensor(float("inf"), device=device, dtype=dtype))
    a = torch.min(torch.stack([torch.tensor(1.0, device=device, dtype=dtype), cand.min()]))
    return a


def _centering_params(s: torch.Tensor, z: torch.Tensor, ds_a: torch.Tensor, dz_a: torch.Tensor):
    mu = torch.dot(s, z) / s.numel()
    alpha = torch.min(
        torch.stack([
            _ort_linesearch(s, ds_a),
            _ort_linesearch(z, dz_a),
        ])
    )
    sigma = ((torch.dot(s + alpha * ds_a, z + alpha * dz_a) / torch.dot(s, z)) ** 3).clamp(min=0.0)
    return sigma, mu


@torch.no_grad()
def solve_lp(q: torch.Tensor, G: torch.Tensor, h: torch.Tensor,
                   max_iters: int = 20,
                   tol: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Primal-dual interior-point for:
        min  q^T x
        s.t. G x <= h

    Returns (x, s, z).
    """
    ns, nx = G.shape
    device = G.device
    dtype = G.dtype

    # Initialize with s, z > 0
    x = torch.zeros(nx, device=device, dtype=dtype)
    s = torch.ones(ns, device=device, dtype=dtype)
    z = torch.ones(ns, device=device, dtype=dtype)

    eye_nx = torch.eye(nx, device=device, dtype=dtype)

    for _ in range(max_iters):
        # residuals
        r1 = G.t() @ z + q
        r2 = s * z
        r3 = G @ x + s - h

        invSZ = z / s  # diag entries

        # G^T diag(invSZ) G
        GSG = G.t() @ (invSZ.unsqueeze(1) * G)
        max_elt = torch.max(GSG.abs())
        GSG = GSG + (1e-8 * max_elt) * eye_nx  # regularize

        # Solve affine step
        rhs_vec = -r1 + G.t() @ (invSZ * (-r3 + (r2 / z)))
        L = torch.linalg.cholesky(GSG)
        dx_a = torch.cholesky_solve(rhs_vec.unsqueeze(1), L, upper=False).squeeze(1)
        ds_a = -(G @ dx_a + r3)
        dz_a = -(r2 + z * ds_a) / s

        # corrector + centering
        sigma, mu = _centering_params(s, z, ds_a, dz_a)
        r2_corr = r2 - (sigma * mu - (ds_a * dz_a))
        rhs_vec = -r1 + G.t() @ (invSZ * (-r3 + (r2_corr / z)))
        dx = torch.cholesky_solve(rhs_vec.unsqueeze(1), L, upper=False).squeeze(1)
        ds = -(G @ dx + r3)
        dz = -(r2_corr + z * ds) / s

        # step and update
        alpha = 0.99 * torch.min(torch.stack([
            _ort_linesearch(s, ds),
            _ort_linesearch(z, dz),
        ]))
        x = x + alpha * dx
        s = s + alpha * ds
        z = z + alpha * dz

        # stopping
        duality_gap = torch.dot(s, z) / s.numel()
        eq_res = torch.linalg.norm(G @ x + s - h)
        if (duality_gap <= tol) and (eq_res <= tol):
            break

    return x, s, z
