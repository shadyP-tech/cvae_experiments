"""Frozen decoded-embedding scoring for the paired Variant-B audit."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence

from ...metrics import balanced_accuracy, macro_f1
from ...protocol import ProtocolError
from ....real_features.classifier_reference.classifiers import ClassifierSpec


LEGACY_REPLAY = "legacy_v2_seed_specific_one_epsilon"
FIXED_ONE_EPSILON = "fold_fixed_one_epsilon"
FIXED_ANTITHETIC = "fold_fixed_antithetic"
CONTROLLED_CANDIDATES = (FIXED_ONE_EPSILON, FIXED_ANTITHETIC)


@dataclass(frozen=True)
class DecodedScore:
    """One diagnostic decoded-mean score and its row-level evidence."""

    metric: Mapping[str, object]
    predictions: tuple[Mapping[str, object], ...]


def fit_frozen_classifier(x_fit: object, y_fit: object) -> object:
    """Fit the exact v2 real-embedding classifier without any selection."""

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    specification = classifier_spec()
    fitted = make_pipeline(
        StandardScaler(),
        LogisticRegression(**specification.to_sklearn_kwargs()),
    )
    fitted.fit(x_fit, y_fit)
    if any(int(value) >= specification.max_iter for value in fitted[-1].n_iter_):
        raise ProtocolError("Frozen Variant-B diagnostic classifier did not converge.")
    return fitted


def classifier_spec() -> ClassifierSpec:
    return ClassifierSpec(
        C=0.01,
        penalty="l2",
        solver="lbfgs",
        max_iter=5000,
        class_weight="balanced",
        random_state=23,
        threshold_policy="fixed_0_5",
    )


def score_decoded_mean(
    *,
    runtime: object,
    classifier: object,
    x_eval: object,
    y_eval: Sequence[int],
    sample_ids: Sequence[str],
    case_ids: Sequence[str],
    center: str,
    training_seed: int,
    candidate: str,
    real_reference_bacc: float,
    minimum_real_bacc: float,
) -> DecodedScore:
    """Decode posterior means, then score with the frozen real classifier."""

    import numpy as np
    import torch

    if float(real_reference_bacc) < float(minimum_real_bacc):
        raise ProtocolError("Variant-B real-reference denominator is below its floor.")
    model = getattr(runtime, "model", None)
    device_name = str(getattr(runtime, "device", ""))
    if model is None or not device_name:
        raise ProtocolError("Fixed-step runtime lacks a model or device.")
    y_np = np.asarray(y_eval, dtype=np.int64)
    if not (
        len(y_np) == len(sample_ids) == len(case_ids)
        and sorted(set(int(value) for value in y_np.tolist())) == [0, 1]
    ):
        raise ProtocolError("Decoded scoring arrays are misaligned or single-class.")
    x_tensor = torch.as_tensor(x_eval, dtype=torch.float32, device=device_name)
    y_tensor = torch.as_tensor(y_np, dtype=torch.long, device=device_name)
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(x_tensor, y_tensor)
        decoded = model.decode(mu, y_tensor).detach().cpu().numpy()
    predicted = [int(value) for value in classifier.predict(decoded)]
    metric = metric_row(
        center=center,
        training_seed=training_seed,
        candidate=candidate,
        truth=y_np.tolist(),
        predicted=predicted,
        real_reference_bacc=real_reference_bacc,
        minimum_real_bacc=minimum_real_bacc,
    )
    rows = tuple(
        {
            "schema_version": "midogpp_b_paired_reparameterization_prediction_v1",
            "center": str(center),
            "training_seed": int(training_seed),
            "candidate": str(candidate),
            "representation_role": "decode_mu",
            "sample_id": str(sample_id),
            "case_id": str(case_id),
            "y_true": int(truth),
            "y_pred": int(prediction),
            "eval_label_role": "final_diagnostic_scoring_and_decode_condition_only",
            "selection_source": "none",
            "oracle_eligible": False,
            "claim_scope": "diagnostic_only",
        }
        for sample_id, case_id, truth, prediction in zip(
            sample_ids,
            case_ids,
            y_np.tolist(),
            predicted,
            strict=True,
        )
    )
    return DecodedScore(metric=metric, predictions=rows)


def metric_row(
    *,
    center: str,
    training_seed: int,
    candidate: str,
    truth: Sequence[int],
    predicted: Sequence[int],
    real_reference_bacc: float,
    minimum_real_bacc: float,
) -> dict[str, object]:
    if len(truth) != len(predicted) or not truth:
        raise ProtocolError("Metric truth/prediction rows must be aligned and nonempty.")
    y_true = [int(value) for value in truth]
    y_pred = [int(value) for value in predicted]
    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred, strict=True))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred, strict=True))
    tn = sum(t == 0 and p == 0 for t, p in zip(y_true, y_pred, strict=True))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred, strict=True))
    positive_recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    bacc = balanced_accuracy(y_true, y_pred)
    expected_bacc = 0.5 * (positive_recall + specificity)
    if abs(bacc - expected_bacc) > 1e-12:
        raise ProtocolError("Balanced accuracy does not match confusion counts.")
    if float(real_reference_bacc) < float(minimum_real_bacc):
        raise ProtocolError("Real-reference BACC is below the denominator floor.")
    preservation = (bacc - 0.5) / (float(real_reference_bacc) - 0.5)
    return {
        "schema_version": "midogpp_b_paired_reparameterization_metric_v1",
        "center": str(center),
        "training_seed": int(training_seed),
        "candidate": str(candidate),
        "representation_role": "decode_mu",
        "bacc": float(bacc),
        "macro_f1": float(macro_f1(y_true, y_pred)),
        "positive_recall": float(positive_recall),
        "specificity": float(specificity),
        "preservation_ratio": float(preservation),
        "real_reference_bacc": float(real_reference_bacc),
        "tp": int(tp),
        "fn": int(fn),
        "tn": int(tn),
        "fp": int(fp),
        "n_positive": int(tp + fn),
        "n_negative": int(tn + fp),
        "eval_labels_used_for_fit": False,
        "eval_labels_used_for_selection": False,
        "eval_labels_used_for_scoring": True,
        "eval_labels_used_for_decode_condition": True,
        "selection_source": "none",
        "oracle_eligible": False,
        "claim_scope": "diagnostic_only",
    }


def prediction_digest(rows: Sequence[Mapping[str, object]]) -> str:
    """Match the immutable v2 endpoint digest copied into the new snapshot."""

    canonical = sorted(
        (
            {
                "sample_id": str(row["sample_id"]),
                "case_id": str(row["case_id"]),
                "y_true": str(row["y_true"]),
                "y_pred": str(row["y_pred"]),
            }
            for row in rows
        ),
        key=lambda row: row["sample_id"],
    )
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def paired_comparison_rows(
    metric_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Return antithetic-minus-one-epsilon deltas for the 12 paired cells."""

    controlled = [
        row for row in metric_rows if str(row.get("candidate")) in CONTROLLED_CANDIDATES
    ]
    by_key: dict[tuple[str, int], dict[str, Mapping[str, object]]] = {}
    for row in controlled:
        key = (str(row["center"]), int(row["training_seed"]))
        candidate = str(row["candidate"])
        if candidate in by_key.setdefault(key, {}):
            raise ProtocolError(f"Duplicate controlled metric row: {key}/{candidate}")
        by_key[key][candidate] = row
    if len(by_key) != 12:
        raise ProtocolError("Controlled metric coverage must contain exactly 12 pairs.")
    output: list[dict[str, object]] = []
    for (center, seed), pair in sorted(by_key.items()):
        if set(pair) != set(CONTROLLED_CANDIDATES):
            raise ProtocolError(f"Incomplete controlled pair for center={center}, seed={seed}.")
        baseline = pair[FIXED_ONE_EPSILON]
        proposed = pair[FIXED_ANTITHETIC]
        pair_ids = {
            str(row.get("pair_id", ""))
            for row in (baseline, proposed)
        }
        if len(pair_ids) != 1 or not next(iter(pair_ids)):
            raise ProtocolError(
                f"Controlled metrics do not share one bound pair ID for "
                f"center={center}, seed={seed}."
            )
        row: dict[str, object] = {
            "schema_version": "midogpp_b_paired_reparameterization_delta_v1",
            "pair_id": next(iter(pair_ids)),
            "center": center,
            "training_seed": seed,
            "baseline": FIXED_ONE_EPSILON,
            "candidate": FIXED_ANTITHETIC,
            "comparison_role": "controlled_common_random_numbers",
            "legacy_v2_included": False,
            "claim_scope": "diagnostic_only",
        }
        for metric in (
            "bacc",
            "macro_f1",
            "positive_recall",
            "specificity",
            "preservation_ratio",
        ):
            row[f"delta_{metric}"] = float(proposed[metric]) - float(baseline[metric])
        output.append(row)
    return output


def audit_decision(
    metric_rows: Sequence[Mapping[str, object]],
    *,
    thresholds: Mapping[str, float],
) -> dict[str, object]:
    """Evaluate only the controlled pair; legacy replay cannot affect the result."""

    candidate_summary = {
        candidate: _candidate_summary(metric_rows, candidate)
        for candidate in CONTROLLED_CANDIDATES
    }
    baseline = candidate_summary[FIXED_ONE_EPSILON]
    proposed = candidate_summary[FIXED_ANTITHETIC]
    gates = {
        "mean_preservation": (
            float(proposed["mean_preservation"])
            >= float(thresholds["mean_preservation_min"])
        ),
        "minimum_seed_mean_preservation": (
            float(proposed["minimum_seed_mean_preservation"])
            >= float(thresholds["minimum_seed_mean_preservation"])
        ),
        "seed_mean_preservation_range": (
            float(proposed["seed_mean_preservation_range"])
            <= float(thresholds["maximum_seed_mean_preservation_range"])
        ),
        "within_center_class_direction_seed_range": (
            float(proposed["maximum_within_center_class_direction_seed_range"])
            <= float(thresholds["maximum_within_center_class_direction_seed_range"])
        ),
        "mean_bacc_delta_vs_fixed": (
            float(proposed["mean_bacc"]) - float(baseline["mean_bacc"])
            >= float(thresholds["mean_bacc_delta_vs_fixed_min"])
        ),
        "mean_preservation_delta_vs_fixed": (
            float(proposed["mean_preservation"]) - float(baseline["mean_preservation"])
            >= float(thresholds["mean_preservation_delta_vs_fixed_min"])
        ),
    }
    passed = all(gates.values())
    return {
        "schema_version": "midogpp_b_paired_reparameterization_audit_decision_v1",
        "decision": (
            "LOW_NOISE_RECIPE_MEETS_DIAGNOSTIC_GATES"
            if passed
            else "LOW_NOISE_RECIPE_DOES_NOT_MEET_DIAGNOSTIC_GATES"
        ),
        "all_gates_pass": passed,
        "gates": gates,
        "thresholds": {key: float(value) for key, value in thresholds.items()},
        "candidate_summary": candidate_summary,
        "controlled_comparison": {
            "baseline": FIXED_ONE_EPSILON,
            "candidate": FIXED_ANTITHETIC,
            "mean_bacc_delta": float(proposed["mean_bacc"]) - float(baseline["mean_bacc"]),
            "mean_preservation_delta": (
                float(proposed["mean_preservation"])
                - float(baseline["mean_preservation"])
            ),
        },
        "legacy_v2_role": "exact_replay_validation_only",
        "legacy_v2_used_for_decision": False,
        "inference_scope": (
            "conditional initialization dispersion on centers 2,5,6,9 under one "
            "locked schedule and posterior-noise realization"
        ),
        "may_export_recipe_lock": False,
        "may_feed_expert_bank": False,
        "may_feed_generation": False,
        "may_feed_routing": False,
        "may_feed_downstream": False,
        "claim_scope": "diagnostic_only",
    }


def _candidate_summary(
    rows: Sequence[Mapping[str, object]],
    candidate: str,
) -> dict[str, object]:
    selected = [row for row in rows if str(row.get("candidate")) == candidate]
    if len(selected) != 12:
        raise ProtocolError(f"{candidate} must contain exactly 12 metric rows.")
    keys = {(str(row["center"]), int(row["training_seed"])) for row in selected}
    if len(keys) != 12:
        raise ProtocolError(f"{candidate} metric keys are duplicate or incomplete.")
    seeds = sorted({int(row["training_seed"]) for row in selected})
    seed_means = {
        seed: _mean(
            float(row["preservation_ratio"])
            for row in selected
            if int(row["training_seed"]) == seed
        )
        for seed in seeds
    }
    direction_ranges = []
    centers = sorted({str(row["center"]) for row in selected})
    for center in centers:
        center_rows = [row for row in selected if str(row["center"]) == center]
        for field in ("positive_recall", "specificity"):
            values = [float(row[field]) for row in center_rows]
            direction_ranges.append(max(values) - min(values))
    return {
        "n_cells": len(selected),
        "mean_bacc": _mean(float(row["bacc"]) for row in selected),
        "mean_macro_f1": _mean(float(row["macro_f1"]) for row in selected),
        "mean_preservation": _mean(
            float(row["preservation_ratio"]) for row in selected
        ),
        "mean_positive_recall": _mean(
            float(row["positive_recall"]) for row in selected
        ),
        "mean_specificity": _mean(float(row["specificity"]) for row in selected),
        "seed_mean_preservation": {str(key): value for key, value in seed_means.items()},
        "minimum_seed_mean_preservation": min(seed_means.values()),
        "seed_mean_preservation_range": max(seed_means.values()) - min(seed_means.values()),
        "maximum_within_center_class_direction_seed_range": max(direction_ranges),
    }


def _mean(values: Sequence[float] | object) -> float:
    materialized = [float(value) for value in values]  # type: ignore[arg-type]
    if not materialized:
        raise ProtocolError("Cannot average an empty diagnostic cell set.")
    return sum(materialized) / float(len(materialized))


__all__ = (
    "CONTROLLED_CANDIDATES",
    "DecodedScore",
    "FIXED_ANTITHETIC",
    "FIXED_ONE_EPSILON",
    "LEGACY_REPLAY",
    "audit_decision",
    "classifier_spec",
    "fit_frozen_classifier",
    "metric_row",
    "paired_comparison_rows",
    "prediction_digest",
    "score_decoded_mean",
)
