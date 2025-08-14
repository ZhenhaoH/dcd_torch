import os, sys
import torch
# Ensure local package import when running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from torch_dcd import time_of_impact, time_of_impact_autograd


def cube_hrep(a, device=None, dtype=torch.float64):
    I = torch.eye(3, device=device, dtype=dtype)
    A = torch.cat([I, -I], dim=0)  # (6,3)
    b = a * torch.ones(6, device=device, dtype=dtype)
    return A, b


def main():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cpu")

    # Two identical cubes of half-extent 1
    A, b = cube_hrep(1.0, device=device)
    R = torch.eye(3, device=device)

    r1 = torch.tensor([0.0, 0.0, 0.0], device=device)
    r2 = torch.tensor([3.0, 0.0, 0.0], device=device)
    v1 = torch.tensor([0.0, 0.0, 0.0], device=device)
    v2 = torch.tensor([-10.0, 0.0, 0.0], device=device)
    dt = 1.0

    # Expected TOI: initial center distance 3, need 2 to touch => (3-2)/10 = 0.1
    tau = time_of_impact(A, b, r1, R, v1, A, b, r2, R, v2, dt)
    print("tau =", float(tau))
    assert abs(float(tau) - 0.1) < 1e-5, "Unexpected TOI value"

    # Autograd wrt r1 and dt
    r1 = r1.clone().requires_grad_()
    dt_t = torch.tensor(dt, device=device, dtype=torch.float64, requires_grad=True)
    tau2 = time_of_impact_autograd(A, b, r1, R, v1, A, b, r2, R, v2, dt_t)
    tau2.backward()
    print("dtau/dr1 =", r1.grad)
    print("dtau/ddt =", dt_t.grad)


if __name__ == "__main__":
    main()
