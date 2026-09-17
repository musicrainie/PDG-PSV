# SPDX-License-Identifier: Apache-2.0
"""Minimal synthetic demonstration of the PDG-PSV Method core.

No EEG dataset, preprocessing pipeline, or experiment configuration is needed.
Run:
    python demo_minimal.py
"""

import torch

from pdg_psv_core import make_controller, specify, symmetric_infonce, warmup_boundary


def make_views(seed: int, batch: int = 32, dim: int = 64):
    g = torch.Generator().manual_seed(seed)
    z1 = 0.30 * torch.randn(batch, dim, generator=g)
    z2 = z1 + 0.06 * torch.randn(batch, dim, generator=g)
    return z1, z2


def main() -> None:
    # Illustrative warm-up: estimate frozen P/S boundaries from synthetic batches.
    warmup = {"P": [], "S": []}
    for seed in (1, 2, 3):
        q = specify(*make_views(seed))
        warmup["P"].append(q["P"].item())
        warmup["S"].append(q["S"].item())

    boundaries = {
        "P": warmup_boundary(warmup["P"], rho=0.90),
        "S": warmup_boundary(warmup["S"], rho=0.90),
        # V can be fixed; this toy value is chosen only for the synthetic demo.
        "V": 0.65,
    }

    # Simple illustrative coefficients; these are not a reproduction recipe.
    controller = make_controller(
        boundaries=boundaries,
        sigmas={"P": 1.0, "S": 1.0, "V": 1.0},
        etas={"P": 0.5, "S": 0.5, "V": 0.5},
        initial_lambdas={"P": 1.0, "S": 1.0, "V": 1.0},
        beta=0.9,
        lambda_min=0.0,
        lambda_max=5.0,
    )

    print("=== Frozen boundaries (illustrative) ===")
    for key, value in boundaries.items():
        print(f"t_{key} = {value:.6f}")

    print("\n=== Interval c: Specify -> Assess -> Enforce ===")
    print("lambda_c:", controller.lambdas())

    for seed in (10, 11, 12):
        z1, z2 = make_views(seed)
        q = specify(z1, z2)
        residual = controller.signed_residuals(q)
        scaled = controller.scaled_residuals(q)
        nce = symmetric_infonce(z1, z2)
        total = controller.primal_objective(nce, q)
        controller.observe_batch(q)
        print(
            "batch",
            seed,
            "q=", {k: round(v.item(), 6) for k, v in q.items()},
            "delta=", {k: round(v.item(), 6) for k, v in residual.items()},
            "tilde_delta=", {k: round(v.item(), 6) for k, v in scaled.items()},
            f"L={total.item():.6f}",
        )

    # Lambda changes only here, after interval-c observations are aggregated.
    update = controller.end_interval()
    print("\nr_c:", {k: round(v, 6) for k, v in update["interval_mean_scaled_residual"].items()})
    print("mhat_c:", {k: round(v, 6) for k, v in update["corrected_history"].items()})
    print("lambda_c:", update["lambda_before"])
    print("lambda_c+1:", update["lambda_after"])


if __name__ == "__main__":
    main()
