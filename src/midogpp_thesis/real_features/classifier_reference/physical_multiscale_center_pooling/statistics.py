"""Descriptive, conditional paired case-cluster uncertainty only."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

from ..downstream import balanced_accuracy
from ..protocol import ProtocolError
from .config import BootstrapConfig


def paired_case_cluster_bootstrap(
    prediction_rows: Sequence[Mapping[str, object]],
    *,
    config: BootstrapConfig,
) -> Mapping[str, object]:
    """Resample identical evaluation cases for selected policy and A."""

    import numpy as np  # type: ignore

    by_center_role: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in prediction_rows:
        by_center_role[(str(row["heldout_center"]), str(row["role"]))].append(row)
    centers = sorted({key[0] for key in by_center_role}, key=int)
    rng = np.random.default_rng(config.seed)
    deltas: list[float] = []
    attempts = 0
    rejected = 0
    while len(deltas) < config.valid_replicates and attempts < config.max_attempts:
        attempts += 1
        center_deltas: list[float] = []
        valid = True
        for center in centers:
            a_rows = by_center_role[(center, "canonical_a")]
            p_rows = by_center_role[(center, "selected_policy")]
            a_by_case = _by_case(a_rows)
            p_by_case = _by_case(p_rows)
            if set(a_by_case) != set(p_by_case):
                raise ProtocolError("Paired bootstrap case pools differ between policy and A.")
            cases = sorted(a_by_case)
            sampled = rng.choice(cases, size=len(cases), replace=True)
            a_eval = [row for case in sampled for row in a_by_case[str(case)]]
            p_eval = [row for case in sampled for row in p_by_case[str(case)]]
            labels = [int(row["label"]) for row in a_eval]
            if set(labels) != {0, 1}:
                valid = False
                break
            a_bacc = balanced_accuracy(labels, [int(row["prediction"]) for row in a_eval])
            p_bacc = balanced_accuracy(
                [int(row["label"]) for row in p_eval],
                [int(row["prediction"]) for row in p_eval],
            )
            center_deltas.append(p_bacc - a_bacc)
        if not valid:
            rejected += 1
            continue
        deltas.append(sum(center_deltas) / float(len(center_deltas)))
    if len(deltas) != config.valid_replicates:
        raise ProtocolError(
            f"Bootstrap produced {len(deltas)} valid replicates after {attempts} attempts."
        )
    lower, upper = np.quantile(np.asarray(deltas), [0.025, 0.975]).tolist()
    return {
        "schema_version": "midogpp_physical_multiscale_conditional_bootstrap_v1",
        "status": "PASS",
        "seed": config.seed,
        "valid_replicates": len(deltas),
        "attempted_replicates": attempts,
        "rejected_class_missing_replicates": rejected,
        "mean_delta": float(np.mean(deltas)),
        "percentile_2_5": float(lower),
        "percentile_97_5": float(upper),
        "interval_role": "conditional_paired_evaluation_case_interval",
        "conditions_on_fixed_fits_and_locked_selection": True,
        "covers_training_selection_uncertainty": False,
        "covers_new_center_uncertainty": False,
        "p_value_computed": False,
        "significance_decision_computed": False,
    }


def _by_case(rows: Sequence[Mapping[str, object]]) -> dict[str, list[Mapping[str, object]]]:
    out: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        out[str(row["case_id"])].append(row)
    return dict(out)
