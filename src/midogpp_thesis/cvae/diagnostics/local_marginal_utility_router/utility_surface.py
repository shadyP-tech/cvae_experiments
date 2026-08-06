"""Scoring primitives for the sealed local marginal-utility surface.

This module is deliberately downstream of the prediction seal.  It accepts
predictions and an already capability-gated label mapping, but it cannot fit a
classifier, load a manifest, or construct a routing policy.  The separation
makes the central scientific quantity explicit::

    Y(H, q, e, s, g) = (BACC(w(+e)) - BACC(uniform)) / epsilon

where both terms are evaluated on the same query rows and seed cell.
"""

from __future__ import annotations

import json
import math
from typing import Mapping, Sequence

import numpy as np

from ...metrics import balanced_accuracy, macro_f1, spearman
from ...protocol import ProtocolError


METRIC_COLUMNS = (
    "schema_version",
    "phase",
    "row_role",
    "claim_role",
    "outer_target",
    "query_center",
    "action_id",
    "arm_role",
    "boosted_source",
    "training_seed",
    "generation_seed",
    "evaluation_row_count",
    "evaluation_class_0_count",
    "evaluation_class_1_count",
    "bacc",
    "macro_f1",
    "primary_metric",
    "macro_f1_role",
    "labels_used_for_scoring_only_after_global_prediction_seal",
    "seed_selection_performed",
)

MARGINAL_UTILITY_COLUMNS = (
    "schema_version",
    "row_role",
    "claim_role",
    "outer_target",
    "query_center",
    "source_center",
    "training_seed",
    "generation_seed",
    "epsilon",
    "control_bacc",
    "perturbed_bacc",
    "paired_bacc_delta",
    "marginal_bacc_utility",
    "control_macro_f1",
    "perturbed_macro_f1",
    "paired_macro_f1_delta_descriptive",
    "pairing_key",
    "support_labels_used",
    "target_H_labels_used",
    "seed_selection_performed",
    "oracle_eligible",
)

LEARNABILITY_PREDICTION_COLUMNS = (
    "schema_version",
    "row_role",
    "claim_role",
    "outer_target",
    "heldout_query_center",
    "source_center",
    "training_seed",
    "generation_seed",
    "observed_marginal_utility",
    "predicted_marginal_utility",
    "prediction_standard_error",
    "alpha",
    "train_query_centers_json",
    "heldout_query_excluded_from_fit",
    "heldout_query_excluded_from_source_role",
    "outer_target_excluded_from_fit",
    "seed_selection_performed",
)

LEARNABILITY_SUMMARY_COLUMNS = (
    "schema_version",
    "row_role",
    "claim_role",
    "outer_target",
    "heldout_query_center",
    "source_count",
    "seed_cell_count",
    "spearman_source_mean_utility",
    "spearman_source_mean_utility_defined",
    "top1_source_agreement",
    "normalized_oracle_gap",
    "rmse",
    "mean_prediction_standard_error",
    "diagnostic_only",
)


def score_sealed_development_predictions(
    store: object,
    *,
    labels_by_sample_id: Mapping[str, int],
    outer_target: str,
) -> tuple[dict[str, object], ...]:
    """Score one outer fold after the global prediction seal is verified.

    ``store`` follows the sibling package's compact prediction-store protocol:
    it exposes ``index_rows`` and ``slice_for(row)``.  Keeping this interface
    structural avoids coupling utility scoring to serialization details.
    """

    index_rows = tuple(getattr(store, "index_rows", ()))
    rows = tuple(
        row
        for row in index_rows
        if str(row.get("phase")) == "development_utility_surface"
        and str(row.get("outer_target")) == str(outer_target)
    )
    if not rows:
        raise ProtocolError("Local utility scoring fold has no sealed predictions.")

    output: list[dict[str, object]] = []
    seen: set[tuple[str, str, int, int]] = set()
    for row in rows:
        query = str(row.get("query_center"))
        action_id = str(row.get("action_id"))
        training_seed = _integer(row.get("training_seed"), "training seed")
        generation_seed = _integer(row.get("generation_seed"), "generation seed")
        key = (query, action_id, training_seed, generation_seed)
        if key in seen:
            raise ProtocolError("Local utility prediction cell is duplicated.")
        seen.add(key)

        sample_ids = _json_strings(row.get("evaluation_row_ids_json"))
        if not sample_ids or len(sample_ids) != len(set(sample_ids)):
            raise ProtocolError("Local utility evaluation identities are invalid.")
        try:
            labels = np.asarray(
                [int(labels_by_sample_id[sample_id]) for sample_id in sample_ids],
                dtype=np.uint8,
            )
        except KeyError as exc:
            raise ProtocolError(
                "Local utility labels do not cover a sealed prediction slice."
            ) from exc
        if set(int(value) for value in labels.tolist()) != {0, 1}:
            raise ProtocolError("Local utility evaluation slice lacks both classes.")

        predictions, _ = getattr(store, "slice_for")(row)
        predictions = np.asarray(predictions, dtype=np.uint8)
        if predictions.shape != labels.shape or not np.isin(predictions, [0, 1]).all():
            raise ProtocolError("Local utility predictions and labels do not align.")
        arm_role = str(row.get("arm_role"))
        boosted_source = str(row.get("boosted_source") or "")
        if (arm_role == "control") != (action_id == "control"):
            raise ProtocolError("Local utility control action binding drifted.")
        if arm_role == "source_perturbation" and not boosted_source:
            raise ProtocolError("Local utility perturbation lacks a boosted source.")

        output.append(
            {
                "schema_version": "midogpp_local_marginal_utility_metric_cell_v1",
                "phase": "development_utility_surface",
                "row_role": "consumed_validation_development_outcome",
                "claim_role": "diagnostic_learnability_only",
                "outer_target": str(outer_target),
                "query_center": query,
                "action_id": action_id,
                "arm_role": arm_role,
                "boosted_source": boosted_source,
                "training_seed": training_seed,
                "generation_seed": generation_seed,
                "evaluation_row_count": len(labels),
                "evaluation_class_0_count": int(np.sum(labels == 0)),
                "evaluation_class_1_count": int(np.sum(labels == 1)),
                "bacc": float(balanced_accuracy(labels.tolist(), predictions.tolist())),
                "macro_f1": float(macro_f1(labels.tolist(), predictions.tolist())),
                "primary_metric": "balanced_accuracy",
                "macro_f1_role": "secondary_descriptive_only",
                "labels_used_for_scoring_only_after_global_prediction_seal": True,
                "seed_selection_performed": False,
            }
        )
    return tuple(output)


def build_paired_marginal_utility_rows(
    metric_rows: Sequence[Mapping[str, object]],
    *,
    epsilon: float,
) -> tuple[dict[str, object], ...]:
    """Pair every source perturbation with its exact equal-union control."""

    eps = float(epsilon)
    if not math.isfinite(eps) or eps <= 0.0 or eps >= 1.0:
        raise ProtocolError("Local utility epsilon must lie strictly between zero and one.")
    rows = tuple(metric_rows)
    if not rows:
        raise ProtocolError("Local utility marginal surface cannot be empty.")

    by_cell: dict[tuple[str, str, int, int, str], Mapping[str, object]] = {}
    for row in rows:
        key = (
            str(row.get("outer_target")),
            str(row.get("query_center")),
            _integer(row.get("training_seed"), "training seed"),
            _integer(row.get("generation_seed"), "generation seed"),
            str(row.get("action_id")),
        )
        if key in by_cell:
            raise ProtocolError("Local utility metric cell is duplicated.")
        by_cell[key] = row

    output: list[dict[str, object]] = []
    for key, perturbed in sorted(by_cell.items()):
        outer, query, training_seed, generation_seed, action_id = key
        if action_id == "control":
            continue
        source = str(perturbed.get("boosted_source") or "")
        if action_id != f"boost_source_{source}" or not source:
            raise ProtocolError("Local utility perturbation action/source binding drifted.")
        control_key = (outer, query, training_seed, generation_seed, "control")
        control = by_cell.get(control_key)
        if control is None:
            raise ProtocolError("Local utility perturbation lacks its paired control.")
        control_bacc = _finite(control.get("bacc"), "control BACC")
        perturbed_bacc = _finite(perturbed.get("bacc"), "perturbed BACC")
        control_f1 = _finite(control.get("macro_f1"), "control macro-F1")
        perturbed_f1 = _finite(perturbed.get("macro_f1"), "perturbed macro-F1")
        delta = perturbed_bacc - control_bacc
        pairing_key = f"H={outer}|q={query}|s={training_seed}|g={generation_seed}"
        output.append(
            {
                "schema_version": "midogpp_local_marginal_utility_row_v1",
                "row_role": "paired_local_utility_outcome",
                "claim_role": "consumed_validation_diagnostic_only",
                "outer_target": outer,
                "query_center": query,
                "source_center": source,
                "training_seed": training_seed,
                "generation_seed": generation_seed,
                "epsilon": eps,
                "control_bacc": control_bacc,
                "perturbed_bacc": perturbed_bacc,
                "paired_bacc_delta": delta,
                "marginal_bacc_utility": delta / eps,
                "control_macro_f1": control_f1,
                "perturbed_macro_f1": perturbed_f1,
                "paired_macro_f1_delta_descriptive": perturbed_f1 - control_f1,
                "pairing_key": pairing_key,
                "support_labels_used": False,
                "target_H_labels_used": False,
                "seed_selection_performed": False,
                "oracle_eligible": False,
            }
        )
    if not output:
        raise ProtocolError("Local utility surface contains no perturbation pairs.")
    return tuple(output)


def summarize_loqdo_learnability(
    prediction_rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Compute utility-aligned diagnostics for each held-out query fold.

    Ranking is evaluated on source means so the nine retained seed cells are
    replicates, not nine opportunities to select a favorable seed.
    """

    rows = tuple(prediction_rows)
    groups: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        key = (str(row.get("outer_target")), str(row.get("heldout_query_center")))
        groups.setdefault(key, []).append(row)
    output: list[dict[str, object]] = []
    for (outer, query), group in sorted(groups.items()):
        by_source: dict[str, list[Mapping[str, object]]] = {}
        for row in group:
            by_source.setdefault(str(row.get("source_center")), []).append(row)
        if len(by_source) < 2:
            raise ProtocolError("LOQDO learnability fold needs at least two sources.")
        observed_means = {
            source: float(np.mean([_finite(row.get("observed_marginal_utility"), "observed utility") for row in source_rows]))
            for source, source_rows in by_source.items()
        }
        predicted_means = {
            source: float(np.mean([_finite(row.get("predicted_marginal_utility"), "predicted utility") for row in source_rows]))
            for source, source_rows in by_source.items()
        }
        ordered_sources = sorted(by_source)
        observed = [observed_means[source] for source in ordered_sources]
        predicted = [predicted_means[source] for source in ordered_sources]
        predicted_best = min(
            ordered_sources,
            key=lambda source: (-predicted_means[source], source),
        )
        observed_best = min(
            ordered_sources,
            key=lambda source: (-observed_means[source], source),
        )
        oracle_gain = max(observed)
        selected_gain = observed_means[predicted_best]
        observed_range = max(observed) - min(observed)
        normalized_gap = (
            (oracle_gain - selected_gain) / observed_range
            if observed_range > 0.0
            else 0.0
        )
        errors = np.asarray(
            [
                _finite(row.get("predicted_marginal_utility"), "predicted utility")
                - _finite(row.get("observed_marginal_utility"), "observed utility")
                for row in group
            ],
            dtype=float,
        )
        standard_errors = [
            _finite(row.get("prediction_standard_error"), "prediction standard error")
            for row in group
        ]
        rank_correlation = float(spearman(observed, predicted))
        rank_defined = math.isfinite(rank_correlation)
        output.append(
            {
                "schema_version": "midogpp_local_marginal_utility_learnability_summary_v1",
                "row_role": "inner_loqdo_heldout_query_diagnostic",
                "claim_role": "utility_learnability_not_routing_performance",
                "outer_target": outer,
                "heldout_query_center": query,
                "source_count": len(by_source),
                "seed_cell_count": len(group),
                # CSV tables require a finite numeric field.  Undefined
                # constant-rank folds use a zero sentinel paired with an
                # explicit false definition flag and are excluded from means.
                "spearman_source_mean_utility": (
                    rank_correlation if rank_defined else 0.0
                ),
                "spearman_source_mean_utility_defined": rank_defined,
                "top1_source_agreement": predicted_best == observed_best,
                "normalized_oracle_gap": float(normalized_gap),
                "rmse": float(np.sqrt(np.mean(np.square(errors)))),
                "mean_prediction_standard_error": float(np.mean(standard_errors)),
                "diagnostic_only": True,
            }
        )
    if not output:
        raise ProtocolError("LOQDO learnability summary cannot be empty.")
    return tuple(output)


def _json_strings(value: object) -> tuple[str, ...]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ProtocolError("Local utility prediction-index JSON is malformed.") from exc
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        raise ProtocolError("Local utility prediction-index row ids are invalid.")
    return tuple(parsed)


def _integer(value: object, role: str) -> int:
    if isinstance(value, bool):
        raise ProtocolError(f"Local utility {role} must be an integer.")
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"Local utility {role} must be an integer.") from exc
    return parsed


def _finite(value: object, role: str) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"Local utility {role} must be numeric.") from exc
    if not math.isfinite(parsed):
        raise ProtocolError(f"Local utility {role} must be finite.")
    return parsed


__all__ = (
    "LEARNABILITY_PREDICTION_COLUMNS",
    "LEARNABILITY_SUMMARY_COLUMNS",
    "MARGINAL_UTILITY_COLUMNS",
    "METRIC_COLUMNS",
    "build_paired_marginal_utility_rows",
    "score_sealed_development_predictions",
    "summarize_loqdo_learnability",
)
