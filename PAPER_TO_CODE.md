# Paper-to-code map

The implementation is organized in the same order as the manuscript Method. The reference code begins from projected representations `z1, z2`; the full EEG representation/training pipeline is outside this release.

| Manuscript Method | Reference implementation | What to inspect |
|---|---|---|
| Base objective: symmetric InfoNCE | `symmetric_infonce()` | paired cross-view target with within-view negatives |
| **Specify / Eq. (1)** `q_{P,b}` | `q_positive()` | row-L2 normalization, paired squared distance, batch mean |
| **Specify / Eq. (2)** `q_{S,b}` | `q_structural()` | per-dimension batch standardization, within-view feature matrix, squared off-diagonal dependence, `log1p` |
| **Specify / Eq. (3)** `q_{V,b}` | `q_variance()` | population variance on **unnormalized** `z`, `sqrt(var+eps)`, one-sided unit-spread deficit |
| **Specify / Eq. (4)** `G_k = E[q_{k,b}-t_k] <= 0` | `ConditionState.boundary`, `signed_residuals()` | the code represents the condition through the frozen boundary `t_k` and signed batch residual `q-t`; `warmup_boundary()` implements the warm-up boundary formula |
| Active set `A_c` | `ConditionState.active` | inactive conditions contribute no primal term and do not advance controller state |
| **Assess / Eq. (5)** `delta=q-t`, `tilde_delta=sigma*delta` | `signed_residuals()`, `scaled_residuals()` | sign is preserved; `sigma` scales magnitude without rectification |
| Positive-part violation is diagnostic only | `positive_violations()` | explicitly separated from the signed primal term |
| **Assess / Eq. (6)** interval aggregate `r_{k,c}` | `observe_batch()` → `end_interval()` | accumulate scaled signed residuals while current `lambda_c` remains fixed, then average at interval end |
| **Assess / Eq. (6)** EMA / active-step bias correction | `ConditionState.corrected_history()`, `end_interval()` | per-condition EMA and `1-beta^n` correction using active update count |
| **Enforce / Eq. (7)** signed primal objective | `primal_geometry_term()`, `primal_objective()` | `InfoNCE + sum lambda * sigma * (q - t)` |
| **Enforce / Eq. (8)** gradient meaning | `primal_geometry_term()` | target, multiplier and scale are constants w.r.t. current representation variables; residual sign does not reverse the current geometry gradient |
| **Enforce / Eq. (9)** projected multiplier update | `end_interval()` | `clip(lambda + eta * mhat, lambda_min, lambda_max)` |
| **Eq. (10)** temporal ordering | `PDGPSVController` docstring, `observe_batch()` → `end_interval()` | interval observations cannot retroactively change the multiplier used in that same interval |
| Static control | `PDGPSVController(dynamic=False)` | same geometry/boundaries, no recursive history/multiplier update |
| `beta=0` dynamic control | `PDGPSVController(beta=0.0)` | recursive signed feedback retained, carried EMA history removed |

## Upstream representation path

The manuscript defines the encoder/projector that produce `z1, z2`; this minimal release deliberately starts at those projected representations so that the reviewer-facing implementation stays focused on the PDG-PSV geometry/controller mechanics. The complete representation and training pipeline is deferred to the later reproduction release.
