# Event Filter Pipeline — Design & Implementation Notes

**File:** `event_filter_pipeline.py`  
**Notebook:** `Reweighting.ipynb`  
**Project:** UCC (Ultracentral Collisions) — TRENTo + Hydro Pipeline  
**Author:** Thiago Siqueira Domingues  

---

## Overview

This module implements an **online event filter** for the TRENTo initial-state generator. Its purpose is to bias the sampling of events toward ultracentral (UC) collisions — which are rare in minimum-bias runs — while keeping every accepted event statistically unbiased through **Inverse Probability Weighting (IPW)**.

The core idea: instead of running full hydrodynamics on every TRENTo event (expensive), accept all UC events, probabilistically keep a fraction of peripheral ones, and assign each accepted event a weight $w = 1/P_\text{acc}$ so that reweighted observables reproduce the minimum-bias distribution.

The companion notebook (`Reweighting.ipynb`) validates the entire pipeline on 1 million minimum-bias TRENTo events across multiple design points, without running hydro.

---

## Physics Motivation

In heavy-ion collisions, ultracentral events (small impact parameter, large initial entropy) drive key physics signatures such as differential radial flow ($v_0$). These events represent only ~1% of minimum-bias collisions. A naive minimum-bias run wastes ~99% of CPU cycles on peripheral events that contribute little to UC observables. The filter concentrates hydro resources on the region of interest while preserving statistical validity through weighting.

---

## Module Structure

### `AcceptancePolicy`

Defines the acceptance probability $P_\text{acc}(E)$ as a piecewise linear function of centrality percentile $C(E)$.

**Constructor parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `calibrator` | `TrentoCalibrator` | Loaded calibration object mapping entropy → centrality |
| `uc_cutoff_pct` | `float` | Centrality percentile below which all events are accepted (default `1.0`) |
| `transition_pct` | `float` | Percentile at which the ramp-down ends (default `10.0`) |
| `min_p_acc` | `float` | Minimum acceptance probability for peripheral events (default `0.05`) |

**Acceptance function:**

$$
P_\text{acc}(E) = \begin{cases}
1.0 & \text{if } C(E) \leq c_\text{UC} \\
1.0 + \frac{P_\text{min} - 1.0}{c_\text{trans} - c_\text{UC}} \cdot (C(E) - c_\text{UC}) & \text{if } c_\text{UC} < C(E) \leq c_\text{trans} \\
P_\text{min} & \text{if } C(E) > c_\text{trans}
\end{cases}
$$

The ramp is linear between the full-accept region and the plateau at $P_\text{min}$. With the defaults (`uc_cutoff=1%`, `transition=10%`, `min_p_acc=0.05`), all 0–1% events pass, 1–10% events are linearly down-sampled from 1.0 → 0.05, and events beyond 10% are accepted at 5%.

> **Note (open question in code):** The method signature currently takes raw entropy `E_val` and internally converts it to centrality via `calibrator.E_to_centrality`. An alternative design would take centrality percentile directly as input, which would decouple `AcceptancePolicy` from the calibrator. Worth considering if the policy is ever reused with pre-computed centralities.

> **Planned extensions:** Step-function and exponential-decay policies are noted as future options (commented out).

---

### `EventFilterPipeline`

Online event-by-event filter that wraps the acceptance policy and executes the Monte Carlo rejection step.

**Constructor:** takes a path to the calibration file, `uc_cutoff_pct`, `transition_pct`, and `min_p_acc`. Internally loads `TrentoCalibrator` and instantiates `AcceptancePolicy`.

> **Bug to fix:** The `__init__` method references `calibrator_path` and `min_p_acc` as bare names (not constructor arguments). The constructor signature needs updating to match the usage shown in `__main__`:
> ```python
> pipeline = EventFilterPipeline("trento_calibration.dp0.pkl", uc_cutoff_pct=1.0, transition_pct=10.0, min_p_acc=0.05)
> ```

#### `process_trento_event(event_id, total_entropy)`

Evaluates one TRENTo event and returns an acceptance decision.

**Algorithm:**
1. Compute $C(E)$ via the calibrator.
2. Evaluate $P_\text{acc}$ from the policy.
3. Draw a uniform random number using `np.random.default_rng(seed=event_id + 123456789)`.
4. Accept if $U < P_\text{acc}$; assign weight $w = 1/P_\text{acc}$ if accepted, $0$ if rejected.

**Returns a dict:**

| Key | Description |
|-----|-------------|
| `event_id` | Integer event identifier |
| `total_entropy` | Raw TRENTo output entropy |
| `centrality_pct` | Centrality percentile (design-point specific) |
| `p_acc` | Acceptance probability |
| `accepted` | Boolean: event passed to hydro |
| `weight` | Statistical weight ($1/P_\text{acc}$ or 0) |

**RNG seeding strategy:** Each event gets its own independent RNG seeded with `event_id + 123456789`. This ensures:
- **Reproducibility:** re-running any event gives the same decision.
- **Synchronization across design points:** the same event ID receives the same random number regardless of which design point is being processed, so acceptance correlations across designs reflect only differences in $P_\text{acc}(C)$, not noise.
- **Platform independence:** `np.random.default_rng` uses PCG64, which is deterministic across Python versions and operating systems (confirmed in notebook Cell 14).

---

### Utility Functions

#### `effective_size(observables, weights)`

Computes Kish's effective sample size:

$$N_\text{eff} = \frac{\left(\sum_i w_i\right)^2}{\sum_i w_i^2}
$$

This quantifies the statistical efficiency loss from weighting. $N_\text{eff}/N_\text{acc} = 1$ means no efficiency loss (uniform weights); values significantly below 1 indicate that a few high-weight events dominate the sample. The notebook (Cell 12–13) plots this per design point and resolves it with 1% centrality bins in the UC region.

---

#### `plot_event_distributions(events)`

Plots histograms of `total_entropy` and `centrality_pct` for a list of event dicts. Intended as a quick sanity check that the input distribution looks like minimum-bias PbPb.

---

#### `recover_min_bias_distribution(accepted_events)`

Estimates the weighted mean of `total_entropy` and `centrality_pct` for the accepted subsample:

$$\langle X \rangle_\text{MB} = \frac{\sum_i w_i X_i}{\sum_i w_i}$$

Returns scalar weighted means, **not** full distributions. This is used inside `calculate_synchronization_level` to summarize per-design distributions.

> **Limitation:** Returning scalar means is a coarse summary. The notebook (Cell 7, Cell 9) performs the more rigorous validation: it compares the full reweighted histogram against the original via chi-square and KS tests, confirming the IPW recovery is accurate.

---

#### `calculate_synchronization_level(accepted_events)`

Groups events by `design_point`, computes `recover_min_bias_distribution` per group, prints per-event acceptance status, and returns a dict of synchronization summaries.

**Requires** that each event dict contains a `design_point` key (not currently populated by `process_trento_event` — this needs to be added upstream when looping over design points).

---

#### `plot_acceptance_probability(synchronization_levels)`

Plots `min_bias_centrality` (the weighted-mean centrality, used here as a proxy axis) on the x-axis against design-point labels. 

> **Known issue:** This function plots the scalar weighted-mean centrality per design point, not the actual $P_\text{acc}(C)$ curve. The notebook (Cell 11) contains the correct version: it bins events by centrality and averages $P_\text{acc}$ within each bin to produce the proper curve.

---

#### `plot_effective_sample_size(synchronization_levels)`

Bar chart of $N_\text{eff} / N_\text{total}$ per design point, using the weighted-mean arrays from `synchronization_levels`.

> **Known issue:** `effective_size` is called with the scalar means (`min_bias_total_entropy`, `min_bias_centrality`) rather than the full arrays of observables and weights. The correct call should pass the per-event observable values and their corresponding weights.

---

## Validation (Notebook: `Reweighting.ipynb`)

The notebook performs a full end-to-end test without hydro:

| Cell | Test |
|------|------|
| 1–2 | Imports, configuration, TRENTo parallel event generation |
| 3 | Load generated events; compute design-specific centrality percentiles |
| 4 | Histograms of entropy, multiplicity, centrality |
| 5 | Impact-parameter synchronization: overlay of $b$ distributions across designs (should be identical) |
| 6 | Apply `AcceptancePolicy`; store acceptance flags and weights |
| 7 | Overlay original vs. reweighted entropy histograms |
| 8 | Cross-section accounting: $\sigma_\text{acc} = f_\text{acc} \cdot \pi b_\text{max}^2$ |
| 9 | Chi-square test: original vs. reweighted distributions |
| 10 | Acceptance correlation matrix across design points |
| 11 | $P_\text{acc}$ vs. centrality curve per design point |
| 12–13 | $N_\text{eff}$ per design, with fine 1%-wide bins in 0–10% region |
| 14 | RNG reproducibility: PCG64 determinism across platforms |
| 15 | Save filter summary to `.npz` |
| 16 | Scatter plot: accepted vs. rejected events in entropy space |

---

## Known Issues & TODOs

1. **`EventFilterPipeline.__init__` signature mismatch** — `calibrator_path` and `min_p_acc` are used but not declared as parameters. Fix before integration.

2. **`process_trento_event` should accept centrality directly** — Converting entropy to centrality inside the policy couples it to the calibrator unnecessarily. Consider passing `centrality_pct` as the input (noted in the code comment).

3. **`design_point` key missing from event dicts** — `calculate_synchronization_level` expects `event['design_point']`, but `process_trento_event` does not populate it. Add this field when building the event dict in the caller loop.

4. **`recover_min_bias_distribution` returns scalars, not histograms** — Fine for a summary metric, but `plot_acceptance_probability` and `plot_effective_sample_size` misuse these scalars as array inputs. The notebook versions of these plots are correct; the standalone functions need fixing to match.

5. **`List` import missing** — `plot_event_distributions`, `recover_min_bias_distribution`, and `calculate_synchronization_level` use `List[dict]` type hints but `from typing import List` is not imported.

6. **`TrentoCalibrator` not defined in this file** — The class is referenced but not shown here. Ensure the calibration module is imported and that the `.pkl` serialization format is stable across Python versions (the notebook Cell 14 notes that `np.random.default_rng` is cross-version safe; the same check is advisable for the calibrator pickle).

7. **RNG test case** — The code comment in `process_trento_event` notes the need for a well-defined test vector to verify PCG64 output across platforms. The notebook (Cell 14) addresses this qualitatively; a formal unit test with a known expected output would be a good addition.

---

## Dependencies

```
numpy
matplotlib
scipy        (KS test, chi-square — used in notebook)
seaborn      (notebook plots)
pandas       (notebook)
pathlib      (notebook)
```

The main script itself uses only `numpy` and `matplotlib`.
