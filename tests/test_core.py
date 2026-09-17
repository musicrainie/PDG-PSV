# SPDX-License-Identifier: Apache-2.0
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pdg_psv_core import (  # noqa: E402
    ConditionState,
    PDGPSVController,
    make_controller,
    q_positive,
    q_structural,
    q_variance,
    specify,
    symmetric_infonce,
    warmup_boundary,
)


class TestPDGPSVCore(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.z1 = torch.randn(16, 12)
        self.z2 = self.z1 + 0.2 * torch.randn(16, 12)


    def test_infonce_is_finite(self):
        value = symmetric_infonce(self.z1, self.z2, temperature=0.1)
        self.assertTrue(torch.isfinite(value))

    def test_method_parameter_domains_are_guarded(self):
        with self.assertRaises(ValueError):
            symmetric_infonce(self.z1, self.z2, temperature=0.0)
        with self.assertRaises(ValueError):
            warmup_boundary([1.0, 2.0], rho=-0.1)
        with self.assertRaises(ValueError):
            ConditionState(boundary=0.0, sigma=-1.0, eta=0.1)
        with self.assertRaises(ValueError):
            ConditionState(boundary=0.0, sigma=1.0, eta=-0.1)

    def test_positive_identical_views_is_zero(self):
        self.assertAlmostEqual(q_positive(self.z1, self.z1).item(), 0.0, places=7)

    def test_variance_uses_raw_projection_scale(self):
        low_scale = q_variance(0.1 * self.z1, 0.1 * self.z2)
        high_scale = q_variance(3.0 * self.z1, 3.0 * self.z2)
        self.assertGreater(low_scale.item(), high_scale.item())

    def test_structural_is_finite_nonnegative(self):
        value = q_structural(self.z1, self.z2)
        self.assertTrue(torch.isfinite(value))
        self.assertGreaterEqual(value.item(), 0.0)

    def test_signed_residual_is_not_relu(self):
        q = specify(self.z1, self.z2)
        boundaries = {k: q[k].item() + 1.0 for k in ("P", "S", "V")}
        ctl = make_controller(
            boundaries,
            sigmas={"P": 1.0, "S": 1.0, "V": 1.0},
            etas={"P": 0.1, "S": 0.1, "V": 0.1},
        )
        residual = ctl.signed_residuals(q)
        self.assertTrue(all(v.item() < 0.0 for v in residual.values()))
        self.assertLess(ctl.primal_geometry_term(q).item(), 0.0)
        self.assertTrue(all(v.item() == 0.0 for v in ctl.positive_violations(q).values()))

    def test_lambda_is_fixed_until_end_interval(self):
        q = specify(self.z1, self.z2)
        ctl = make_controller(
            boundaries={k: q[k].item() - 0.1 for k in ("P", "S", "V")},
            sigmas={"P": 1.0, "S": 1.0, "V": 1.0},
            etas={"P": 0.5, "S": 0.5, "V": 0.5},
        )
        before = ctl.lambdas()
        ctl.observe_batch(q)
        self.assertEqual(before, ctl.lambdas())
        ctl.end_interval()
        self.assertNotEqual(before, ctl.lambdas())

    def test_bias_correction_first_active_update_equals_interval_mean(self):
        ctl = make_controller(
            boundaries={"P": 0.0, "S": 0.0, "V": 0.0},
            sigmas={"P": 1.0, "S": 1.0, "V": 1.0},
            etas={"P": 0.0, "S": 0.0, "V": 0.0},
            beta=0.9,
        )
        q = {k: torch.tensor(float(i + 1)) for i, k in enumerate(("P", "S", "V"))}
        ctl.observe_batch(q)
        update = ctl.end_interval()
        for key in ("P", "S", "V"):
            self.assertAlmostEqual(
                update["corrected_history"][key],
                update["interval_mean_scaled_residual"][key],
                places=7,
            )

    def test_inactive_condition_retains_state_and_lambda(self):
        states = {
            "P": ConditionState(boundary=0.0, sigma=1.0, eta=0.5, active=True),
            "S": ConditionState(boundary=0.0, sigma=1.0, eta=0.5, active=False),
            "V": ConditionState(boundary=0.0, sigma=1.0, eta=0.5, active=True),
        }
        ctl = PDGPSVController(states, beta=0.9)
        q = {"P": torch.tensor(1.0), "S": torch.tensor(3.0), "V": torch.tensor(2.0)}
        ctl.observe_batch(q)
        ctl.end_interval()
        self.assertEqual(states["S"].active_updates, 0)
        self.assertEqual(states["S"].ema, 0.0)
        self.assertEqual(states["S"].multiplier, 1.0)

    def test_projection_respects_bounds(self):
        ctl = make_controller(
            boundaries={"P": 0.0, "S": 0.0, "V": 0.0},
            sigmas={"P": 1.0, "S": 1.0, "V": 1.0},
            etas={"P": 100.0, "S": 100.0, "V": 100.0},
            lambda_min=0.0,
            lambda_max=5.0,
        )
        q = {"P": torch.tensor(10.0), "S": torch.tensor(10.0), "V": torch.tensor(10.0)}
        ctl.observe_batch(q)
        ctl.end_interval()
        self.assertEqual(ctl.lambdas(), {"P": 5.0, "S": 5.0, "V": 5.0})

    def test_static_mode_keeps_lambdas_and_history_fixed(self):
        ctl = make_controller(
            boundaries={"P": 0.0, "S": 0.0, "V": 0.0},
            sigmas={"P": 1.0, "S": 1.0, "V": 1.0},
            etas={"P": 1.0, "S": 1.0, "V": 1.0},
            dynamic=False,
        )
        q = {"P": torch.tensor(1.0), "S": torch.tensor(1.0), "V": torch.tensor(1.0)}
        ctl.observe_batch(q)
        ctl.end_interval()
        self.assertEqual(ctl.lambdas(), {"P": 1.0, "S": 1.0, "V": 1.0})
        self.assertTrue(all(ctl.conditions[k].active_updates == 0 for k in ("P", "S", "V")))

    def test_beta_zero_removes_carried_history(self):
        ctl = make_controller(
            boundaries={"P": 0.0, "S": 0.0, "V": 0.0},
            sigmas={"P": 1.0, "S": 1.0, "V": 1.0},
            etas={"P": 0.0, "S": 0.0, "V": 0.0},
            beta=0.0,
        )
        ctl.observe_batch({"P": torch.tensor(1.0), "S": torch.tensor(2.0), "V": torch.tensor(3.0)})
        ctl.end_interval()
        ctl.observe_batch({"P": torch.tensor(-4.0), "S": torch.tensor(-5.0), "V": torch.tensor(-6.0)})
        update = ctl.end_interval()
        self.assertEqual(update["corrected_history"], {"P": -4.0, "S": -5.0, "V": -6.0})


if __name__ == "__main__":
    unittest.main()
