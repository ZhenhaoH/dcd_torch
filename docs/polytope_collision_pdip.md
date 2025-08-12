# Polytope Collision via Primal–Dual Interior-Point (PDIP)

## Problem Setup

- Goal: minimize a scalar dilation `alpha` so a single point `x ∈ R^3` lies inside both polytopes when each is uniformly “inflated” by `alpha`.
- Decision variable: `x_hat = [x; alpha] ∈ R^4` where `x` is a 3D point and `alpha` is scalar.
- Polytope model: each body i has an H-rep in its body frame `A_i x_B <= b_i`, pose `(r_i, Q_i)` with `x_B = Q_i^T (x_W - r_i)`.
- Constraints: for each row of `A_i, b_i`, impose `A_i Q_i^T x - b_i alpha <= A_i Q_i^T r_i`. Stacking both bodies gives `G x_hat <= h`.
- Objective: `min c^T x_hat` with `c = [0, 0, 0, 1]` (only penalize `alpha`).
- Implementation: `dpax/polytopes.py::problem_matrices` builds `(c, G, h)`.

## Interpretation

- Collision test: the optimal `alpha*` is the minimum uniform dilation factor making the two polytopes intersect at some `x`.
- Sign convention: `alpha* ≤ 1` ⇒ already intersect (collision); `alpha* > 1` ⇒ separated; separation margin grows with `alpha*`.

## PDIP Solver (LP)

- Form: solve `min q^T x` s.t. `Gx <= h` (here `x = x_hat`, `q = c`).
- Slack/dual: introduce slack `s > 0` with `Gx + s = h` and dual `z > 0`.
- Residuals:
  - Dual: `r1 = G^T z + q`
  - Complementarity: `r2 = s ∘ z`
  - Primal: `r3 = Gx + s - h`
- KKT reduction: use Schur complement on `G^T diag(z/s) G` with small diagonal regularization for robustness, factor via Cholesky, solve for `dx`, then recover `ds`, `dz`.
- Predictor–corrector:
  - Affine step `(dx_a, ds_a, dz_a)`; duality gap `mu = (s·z)/m`.
  - Positivity-preserving step cap via a backtracking on `s + α ds`, `z + α dz`.
  - Centering `sigma = ( ((s + α ds_a)·(z + α dz_a)) / (s·z) )^3`, correct `r2` by `sigma*mu - ds_a ∘ dz_a`, and re-solve.
- Step update: `α = 0.99 * min(α_s, α_z)`; update `x, s, z`.
- Stopping: up to 20 iters; early stop when duality gap and primal residual are both below `1e-5`.
- Implementation: see `dpax/pdip_solver.py` (`solve_lp`, `pdip_step`, `centering_params`, `ort_linesearch`).

## Value and Gradients

- Value: `polytope_proximity(...)` builds `(c, G, h)`, solves the LP, and returns `alpha = x[3]`.
- Gradients (custom JVP):
  - Envelope theorem: gradient of the optimal value wrt problem data equals gradient of the Lagrangian evaluated at optimal `(x, z)` with `L = c^T x + z^T (Gx - h)`.
  - Since `c` is constant wrt data, differentiate `z^T (Gx - h)` while holding `(x, s, z)` at the PDIP solution.
  - Implemented via `jax.custom_jvp` using `polytope_lagrangian` and `jax.grad` for sensitivities of `(A1, b1, r1, Q1, A2, b2, r2, Q2)`.
- Implementation: `dpax/polytopes.py::polytope_proximity` and `.defjvp`.

## Example: Two Equal Cubes

The following example constructs two identical axis-aligned cubes (side length `2a`) using the H-representation:

- `A = [ I; -I ] ∈ R^{6×3}` (face normals ±x, ±y, ±z)
- `b = a · 1_6`

We test two placements:

- Overlapping: centers at `r1 = (0,0,0)`, `r2 = (1.0, 0, 0)` with `a = 1.0` — expect `alpha ≤ 1`.
- Separated: centers at `r1 = (0,0,0)`, `r2 = (2.5, 0, 0)` with `a = 1.0` — expect `alpha > 1`.

```python
# save as scripts/example_two_cubes.py to run
import jax.numpy as jnp
from dpax.polytopes import polytope_proximity

# cube of half-extent a
A = jnp.vstack([jnp.eye(3), -jnp.eye(3)])  # (6,3)
a = 1.0
b = a * jnp.ones(6)                        # (6,)

I = jnp.eye(3)

cases = {
    "overlap": {
        "r1": jnp.array([0.0, 0.0, 0.0]),
        "Q1": I,
        "r2": jnp.array([1.0, 0.0, 0.0]),
        "Q2": I,
    },
    "separate": {
        "r1": jnp.array([0.0, 0.0, 0.0]),
        "Q1": I,
        "r2": jnp.array([2.5, 0.0, 0.0]),
        "Q2": I,
    },
}

for name, cfg in cases.items():
    alpha = polytope_proximity(A, b, cfg["r1"], cfg["Q1"], A, b, cfg["r2"], cfg["Q2"])
    print(f"{name}: alpha = {float(alpha):.6f}  ->  {'collision' if alpha <= 1.0 else 'separated'}")
```

Notes:

- When the center separation equals `2a`, faces just touch and `alpha` should be near `1.0` (up to solver tolerances and regularization).
- Larger separations yield `alpha > 1`; smaller yield `alpha < 1`.
- For rotated cubes, replace `Q1`, `Q2` with the rotation matrices of each body.

## Running Locally

- Environment: see `requirements.txt` or `environment.yml` (Python 3.9, `jax`, `jaxlib`).
- Example: copy the snippet above into `scripts/example_two_cubes.py` and run:

```bash
python scripts/example_two_cubes.py
```

You should see `alpha` ≤ 1.0 for the overlapping case and `alpha` > 1.0 for the separated case.
