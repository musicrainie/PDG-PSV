# SPDX-License-Identifier: Apache-2.0
"""Compact executable reference implementation of the PDG-PSV Method core.

This module is reviewer-facing: it mirrors the paper's Specify -> Assess ->
Enforce ordering and intentionally omits dataset loading, EEG preprocessing,
training recipes, and result-reproduction infrastructure.

This implementation contains the manuscript method core; exploratory alternatives
outside the reported method are intentionally omitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping

import torch
import torch.nn.functional as F

CONDITIONS = ("P", "S", "V")


def _check_pair(z1: torch.Tensor, z2: torch.Tensor) -> None:
    if z1.ndim != 2 or z2.ndim != 2:
        raise ValueError(f"z1/z2 must be [B, D], got {tuple(z1.shape)} and {tuple(z2.shape)}")
    if z1.shape != z2.shape:
        raise ValueError(f"z1/z2 shape mismatch: {tuple(z1.shape)} vs {tuple(z2.shape)}")
    if z1.shape[0] <= 1:
        raise ValueError(f"PDG-PSV batch statistics require B > 1, got {z1.shape[0]}")


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """Return all off-diagonal entries of a square matrix as one vector."""
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError(f"expected a square matrix, got {tuple(x.shape)}")
    n = x.shape[0]
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


# ---------------------------------------------------------------------------
# Base objective: symmetric NT-Xent / InfoNCE used by the selected path.
# ---------------------------------------------------------------------------
def symmetric_infonce(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Symmetric InfoNCE with each paired cross-view sample as the target."""
    _check_pair(z1, z2)
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be > 0")
    batch = z1.shape[0]
    z1n = F.normalize(z1, dim=-1)
    z2n = F.normalize(z2, dim=-1)
    diag = torch.eye(batch, dtype=torch.bool, device=z1.device)

    logits_11 = (z1n @ z1n.T) / float(temperature)
    logits_12 = (z1n @ z2n.T) / float(temperature)
    logits_21 = (z2n @ z1n.T) / float(temperature)
    logits_22 = (z2n @ z2n.T) / float(temperature)

    logits_11 = logits_11[~diag].view(batch, -1)
    logits_22 = logits_22[~diag].view(batch, -1)
    logits_1 = torch.cat([logits_12, logits_11], dim=1)
    logits_2 = torch.cat([logits_21, logits_22], dim=1)
    logits = torch.cat([logits_1, logits_2], dim=0)
    labels = torch.arange(batch, dtype=torch.long, device=z1.device).repeat(2)
    return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# Specify: the three monitored projected-space geometry observables.
# ---------------------------------------------------------------------------
def q_positive(z1: torch.Tensor, z2: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Eq. (1): normalized positive-pair squared distance q_P."""
    _check_pair(z1, z2)
    z1n = F.normalize(z1, dim=1, eps=float(eps))
    z2n = F.normalize(z2, dim=1, eps=float(eps))
    return (z1n - z2n).pow(2).sum(dim=1).mean()


def q_structural(z1: torch.Tensor, z2: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Eq. (2): within-view standardized cross-feature dependence q_S."""
    _check_pair(z1, z2)
    z1s = (z1 - z1.mean(dim=0)) / (z1.std(dim=0, unbiased=False) + float(eps))
    z2s = (z2 - z2.mean(dim=0)) / (z2.std(dim=0, unbiased=False) + float(eps))
    scale = float(z1.shape[0])
    c1 = z1s.T @ z1s / scale
    c2 = z2s.T @ z2s / scale
    offdiag_mean_sq = 0.5 * (_off_diagonal(c1).pow(2).mean() + _off_diagonal(c2).pow(2).mean())
    return torch.log1p(offdiag_mean_sq.clamp_min(0.0))


def q_variance(z1: torch.Tensor, z2: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Eq. (3): marginal-spread deficit q_V on *unnormalized* projections."""
    _check_pair(z1, z2)
    std1 = torch.sqrt(z1.var(dim=0, unbiased=False) + float(eps))
    std2 = torch.sqrt(z2.var(dim=0, unbiased=False) + float(eps))
    return 0.5 * (F.relu(1.0 - std1).mean() + F.relu(1.0 - std2).mean())


def specify(z1: torch.Tensor, z2: torch.Tensor, eps: float = 1e-4) -> Dict[str, torch.Tensor]:
    """Compute the selected P/S/V geometry observables for one batch."""
    return {
        "P": q_positive(z1, z2, eps=eps),
        "S": q_structural(z1, z2, eps=eps),
        "V": q_variance(z1, z2, eps=eps),
    }


def warmup_boundary(batch_observables: Iterable[float], rho: float) -> float:
    """Warm-up rule t_k = rho_k * mean_{b in W_k} q_{k,b}.

    The full training loop is responsible for supplying training-only warm-up
    observations and freezing the returned boundary before activation.
    """
    if float(rho) < 0.0:
        raise ValueError("rho must be >= 0")
    values = [float(v) for v in batch_observables]
    if not values:
        raise ValueError("warmup_boundary requires at least one batch observable")
    return float(rho) * (sum(values) / len(values))


# ---------------------------------------------------------------------------
# Assess + Enforce state.
# ---------------------------------------------------------------------------
@dataclass
class ConditionState:
    """State for one active geometry condition k in {P,S,V}."""

    boundary: float
    sigma: float
    eta: float
    multiplier: float = 1.0
    active: bool = True
    ema: float = 0.0
    active_updates: int = 0
    _interval_sum: float = field(default=0.0, repr=False)
    _interval_count: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if float(self.sigma) < 0.0:
            raise ValueError("sigma must be >= 0")
        if float(self.eta) < 0.0:
            raise ValueError("eta must be >= 0")

    def corrected_history(self, beta: float) -> float:
        if self.active_updates <= 0:
            return 0.0
        correction = max(1.0 - float(beta) ** int(self.active_updates), 1e-12)
        return float(self.ema) / correction


class PDGPSVController:
    """Minimal interval-wise PDG-PSV controller matching the paper's ordering.

    During an interval, ``primal_objective`` always uses the current multipliers.
    ``observe_batch`` only accumulates signed scaled residuals. Multipliers change
    only in ``end_interval``, so interval-c observations can affect lambda_{c+1}
    but cannot retroactively alter lambda_c.
    """

    def __init__(
        self,
        conditions: Mapping[str, ConditionState],
        *,
        beta: float = 0.9,
        lambda_min: float = 0.0,
        lambda_max: float = 5.0,
        dynamic: bool = True,
    ) -> None:
        if set(conditions) != set(CONDITIONS):
            raise ValueError(f"conditions must contain exactly {CONDITIONS}")
        if not 0.0 <= float(beta) < 1.0:
            raise ValueError("beta must be in [0, 1)")
        if float(lambda_min) < 0.0 or float(lambda_max) < float(lambda_min):
            raise ValueError("invalid multiplier bounds")
        self.conditions = {key: conditions[key] for key in CONDITIONS}
        self.beta = float(beta)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.dynamic = bool(dynamic)
        for key, state in self.conditions.items():
            if not self.lambda_min <= float(state.multiplier) <= self.lambda_max:
                raise ValueError(f"lambda_{key} is outside multiplier bounds")

    def lambdas(self) -> Dict[str, float]:
        return {key: float(self.conditions[key].multiplier) for key in CONDITIONS}

    def signed_residuals(self, q: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Eq. (5): delta_{k,b}=q_{k,b}-t_k for active conditions."""
        out: Dict[str, torch.Tensor] = {}
        for key in CONDITIONS:
            state = self.conditions[key]
            out[key] = q[key] - float(state.boundary) if state.active else torch.zeros_like(q[key])
        return out

    def scaled_residuals(self, q: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Eq. (5): tilde_delta_{k,b}=sigma_k * delta_{k,b}."""
        residual = self.signed_residuals(q)
        return {key: float(self.conditions[key].sigma) * residual[key] for key in CONDITIONS}

    def positive_violations(self, q: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Diagnostic [q-t]_+ values; these are NOT the signed primal terms."""
        return {key: F.relu(value) for key, value in self.signed_residuals(q).items()}

    def primal_geometry_term(self, q: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Eq. (7): sum_k lambda_{k,c} sigma_k (q_{k,b}-t_k)."""
        scaled = self.scaled_residuals(q)
        terms = [
            float(self.conditions[key].multiplier) * scaled[key]
            for key in CONDITIONS
            if self.conditions[key].active
        ]
        if not terms:
            return next(iter(q.values())).new_zeros(())
        return torch.stack(terms).sum()

    def primal_objective(self, infonce_loss: torch.Tensor, q: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """InfoNCE plus the signed PDG-PSV geometry term."""
        return infonce_loss + self.primal_geometry_term(q)

    def observe_batch(self, q: Mapping[str, torch.Tensor]) -> Dict[str, float]:
        """Accumulate scaled signed residuals without changing current multipliers."""
        scaled = self.scaled_residuals(q)
        snapshot: Dict[str, float] = {}
        for key in CONDITIONS:
            state = self.conditions[key]
            if state.active:
                value = float(scaled[key].detach().item())
                state._interval_sum += value
                state._interval_count += 1
                snapshot[key] = value
            else:
                snapshot[key] = 0.0
        return snapshot

    def end_interval(self) -> Dict[str, Dict[str, float]]:
        """Eqs. (6,9): aggregate -> EMA/bias correction -> projected lambda update."""
        before = self.lambdas()
        interval_mean: Dict[str, float] = {}
        corrected: Dict[str, float] = {}

        for key in CONDITIONS:
            state = self.conditions[key]
            if not state.active or state._interval_count <= 0:
                interval_mean[key] = 0.0
                corrected[key] = state.corrected_history(self.beta) if state.active else 0.0
                continue

            r_k = state._interval_sum / float(state._interval_count)
            interval_mean[key] = r_k
            if self.dynamic:
                state.ema = self.beta * float(state.ema) + (1.0 - self.beta) * r_k
                state.active_updates += 1
                corrected[key] = state.corrected_history(self.beta)
                new_lambda = float(state.multiplier) + float(state.eta) * corrected[key]
                state.multiplier = min(self.lambda_max, max(self.lambda_min, new_lambda))
            else:
                corrected[key] = state.corrected_history(self.beta)

        after = self.lambdas()
        for state in self.conditions.values():
            state._interval_sum = 0.0
            state._interval_count = 0

        return {
            "interval_mean_scaled_residual": interval_mean,
            "corrected_history": corrected,
            "lambda_before": before,
            "lambda_after": after,
        }


def make_controller(
    boundaries: Mapping[str, float],
    sigmas: Mapping[str, float],
    etas: Mapping[str, float],
    *,
    initial_lambdas: Mapping[str, float] | None = None,
    active: Mapping[str, bool] | None = None,
    beta: float = 0.9,
    lambda_min: float = 0.0,
    lambda_max: float = 5.0,
    dynamic: bool = True,
) -> PDGPSVController:
    """Convenience constructor using paper-facing P/S/V names."""
    initial_lambdas = initial_lambdas or {key: 1.0 for key in CONDITIONS}
    active = active or {key: True for key in CONDITIONS}
    states = {
        key: ConditionState(
            boundary=float(boundaries[key]),
            sigma=float(sigmas[key]),
            eta=float(etas[key]),
            multiplier=float(initial_lambdas[key]),
            active=bool(active[key]),
        )
        for key in CONDITIONS
    }
    return PDGPSVController(
        states,
        beta=beta,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        dynamic=dynamic,
    )
