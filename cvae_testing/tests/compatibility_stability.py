from __future__ import annotations

import math
import random
from statistics import mean
from typing import Dict, Mapping, Sequence, Tuple


LEGACY_STD_POLICY = "legacy_std_v1"
SIGN_CI_POLICY = "sign_ci_v2"
SUPPORTED_DECISION_POLICIES = {LEGACY_STD_POLICY, SIGN_CI_POLICY}

TOP1_CI_LOWER_TOLERANCE = -0.025
SPEARMAN_CI_LOWER_TOLERANCE = -0.025
GAP_PCT_CI_LOWER_TOLERANCE = -1.0

CATASTROPHIC_TOP1_UPLIFT_MIN = -0.05
CATASTROPHIC_SPEARMAN_UPLIFT_MIN = -0.05
CATASTROPHIC_GAP_PCT_REDUCTION_MIN = -2.0


def validate_decision_policy_version(value: object) -> str:
    policy = str(value or SIGN_CI_POLICY).strip()
    if policy not in SUPPORTED_DECISION_POLICIES:
        raise ValueError(
            f"decision_policy_version must be one of {sorted(SUPPORTED_DECISION_POLICIES)}, got {policy!r}"
        )
    return policy


def finite_values(values: Sequence[float]) -> list[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    clean = finite_values(values)
    if not clean:
        return 0.0, 0.0
    mu = float(mean(clean))
    var = float(sum((v - mu) ** 2 for v in clean) / len(clean))
    return mu, math.sqrt(max(var, 0.0))


def sign_inconsistency_count(values: Sequence[float]) -> int:
    clean = finite_values(values)
    pos = any(v > 1e-12 for v in clean)
    neg = any(v < -1e-12 for v in clean)
    return 1 if (pos and neg) else 0


def effective_positive_threshold(
    n_observations: int,
    *,
    min_improving_runs: int,
    min_positive_fraction: float,
) -> int:
    n = int(n_observations)
    if n <= 3:
        base = 2
    else:
        base = int(math.ceil(float(min_positive_fraction) * float(n)))
    return max(int(min_improving_runs), int(base))


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    level: float,
    reps: int,
    seed: int,
) -> Tuple[float, float]:
    clean = finite_values(values)
    if not clean:
        return float("nan"), float("nan")
    if len(clean) == 1 or int(reps) <= 0:
        value = float(clean[0])
        return value, value

    rng = random.Random(int(seed))
    n = len(clean)
    boot_means = []
    for _ in range(int(reps)):
        sample = [clean[rng.randrange(n)] for _idx in range(n)]
        boot_means.append(float(mean(sample)))
    boot_means.sort()

    alpha = max(0.0, min(1.0, 1.0 - float(level)))
    low_q = alpha / 2.0
    high_q = 1.0 - alpha / 2.0
    low_idx = int(math.floor(low_q * (len(boot_means) - 1)))
    high_idx = int(math.ceil(high_q * (len(boot_means) - 1)))
    return float(boot_means[low_idx]), float(boot_means[high_idx])


def evaluate_sign_ci_stability(
    *,
    metric_values: Mapping[str, Sequence[float]],
    metric_means: Mapping[str, float],
    min_improving_runs: int,
    min_positive_fraction: float,
    ci_level: float,
    ci_bootstrap_reps: int,
    ci_bootstrap_seed: int,
    ci_source: str,
    ci_lower_tolerances: Mapping[str, float],
    catastrophic_regression_breach: bool,
    regression_check_missing: bool,
) -> Dict[str, object]:
    n_observations = max((len(finite_values(v)) for v in metric_values.values()), default=0)
    positive_threshold = effective_positive_threshold(
        n_observations,
        min_improving_runs=int(min_improving_runs),
        min_positive_fraction=float(min_positive_fraction),
    )

    counts: Dict[str, int] = {}
    ci_low: Dict[str, float] = {}
    ci_high: Dict[str, float] = {}
    for idx, (metric, values) in enumerate(metric_values.items()):
        clean = finite_values(values)
        counts[metric] = int(sum(1 for v in clean if v > 0.0))
        lo, hi = bootstrap_mean_ci(
            clean,
            level=float(ci_level),
            reps=int(ci_bootstrap_reps),
            seed=int(ci_bootstrap_seed) + idx * 1009,
        )
        ci_low[metric] = float(lo)
        ci_high[metric] = float(hi)

    ci_source_value = str(ci_source)
    ci_hard_gate_applied = bool(not (ci_source_value == "seed_descriptive" and n_observations <= 3))
    mean_positive_pass = all(float(metric_means.get(metric, 0.0)) > 0.0 for metric in metric_values)
    positive_count_pass = all(int(counts.get(metric, 0)) >= int(positive_threshold) for metric in metric_values)
    ci_pass = True
    if ci_hard_gate_applied:
        ci_pass = all(
            float(ci_low.get(metric, float("-inf"))) >= float(ci_lower_tolerances.get(metric, 0.0))
            for metric in metric_values
        )

    stability_pass = bool(
        n_observations > 0
        and mean_positive_pass
        and positive_count_pass
        and ci_pass
        and not catastrophic_regression_breach
        and not regression_check_missing
    )

    out: Dict[str, object] = {
        "positive_observation_count": int(n_observations),
        "positive_observation_threshold": int(positive_threshold),
        "ci_source": ci_source_value,
        "ci_hard_gate_applied": int(ci_hard_gate_applied),
        "mean_positive_pass": int(mean_positive_pass),
        "positive_count_pass": int(positive_count_pass),
        "ci_pass": int(ci_pass),
        "sign_ci_stability_pass": int(stability_pass),
    }
    for metric in metric_values:
        out[f"{metric}_positive_count"] = int(counts.get(metric, 0))
        out[f"{metric}_ci_low"] = float(ci_low.get(metric, float("nan")))
        out[f"{metric}_ci_high"] = float(ci_high.get(metric, float("nan")))
    return out
