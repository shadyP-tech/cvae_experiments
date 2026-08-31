"""Label-free diagnostics for understanding HARP v3 policy abstention."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
import math

import numpy as np

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from ...routing.harp_v3.metrics import (
    CASE_CONTRIBUTION_METRIC_NAME,
    PRIMARY_ESTIMAND,
)
from ...routing.harp_v3.serialization import decision_from_payload
from .contracts import ActionKind, PrelabelRouteSet


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ProtocolError("HARP v3 prelabel diagnostic values are invalid.")
    return {
        "count": int(len(array)),
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9, method="higher")),
        "maximum": float(np.max(array)),
    }


def build_prelabel_diagnostics(routes: PrelabelRouteSet) -> dict[str, object]:
    """Summarize every source-only veto before evaluation labels can open."""

    if not isinstance(routes, PrelabelRouteSet):
        raise ProtocolError("HARP v3 prelabel diagnostics require typed routes.")
    selected: Counter[str] = Counter()
    final_reasons: Counter[str] = Counter()
    action_rejections: Counter[str] = Counter()
    comparison_rejections: Counter[str] = Counter()
    action_eligibility: Counter[str] = Counter()
    geometry_ratios: defaultdict[str, list[float]] = defaultdict(list)
    geometry_percentiles: defaultdict[str, list[float]] = defaultdict(list)
    geometry_tail_probabilities: defaultdict[str, list[float]] = defaultdict(list)
    geometry_tail_floors: defaultdict[str, list[float]] = defaultdict(list)
    shrinkages: defaultdict[str, list[float]] = defaultdict(list)
    geometry_methods: set[str] = set()
    geometry_cardinality_rules: set[str] = set()
    exact_b = True

    for routed in routes.cases:
        decision = decision_from_payload(routed.decision_payload)
        if (
            decision.outer_target_id != routed.outer_target_id
            or decision.case_id != routed.case_id
            or decision.selected_kind.value
            != ("HXE" if routed.selected_kind is ActionKind.HXE else routed.selected_kind.value)
        ):
            raise ProtocolError("HARP v3 route/decision diagnostic binding drifted.")
        selected[routed.selected_kind.value] += 1
        final_reasons[routed.reason] += 1
        if routed.selected_kind is ActionKind.B:
            exact_b &= (
                routed.routed_probabilities.tobytes(order="C")
                == routed.baseline_probabilities.tobytes(order="C")
            )
        for audit in decision.action_audits:
            action_eligibility[
                f"{audit.action_kind.value}:{'eligible' if audit.eligible else 'rejected'}"
            ] += 1
            for reason in audit.rejection_reasons:
                action_rejections[str(reason)] += 1
            for score in audit.comparison_scores:
                comparison = score.comparison.value
                for reason in score.rejection_reasons:
                    comparison_rejections[f"{comparison}:{reason}"] += 1
                geometry_ratios[comparison].append(float(score.geometry.maximum_ratio))
                geometry_percentiles[comparison].append(
                    float(score.geometry.empirical_percentile)
                )
                geometry_tail_probabilities[comparison].append(
                    float(score.geometry.empirical_tail_probability)
                )
                geometry_tail_floors[comparison].append(
                    float(score.geometry.finite_sample_tail_floor)
                )
                shrinkages[comparison].append(
                    float(score.geometry.compatibility_shrinkage)
                )
                geometry_methods.add(str(score.geometry.calibration_method))
                geometry_cardinality_rules.add(
                    str(score.geometry.ensemble_cardinality_rule)
                )

    if not math.isfinite(float(len(routes.cases))) or not exact_b:
        raise ProtocolError("HARP v3 prelabel exact-B diagnostic failed.")
    body: dict[str, object] = {
        "schema_version": "midogpp_harp_v3_prelabel_rejection_diagnostics_v2",
        "route_hash": routes.route_hash,
        "case_count": len(routes.cases),
        "selected_action_counts": dict(sorted(selected.items())),
        "final_reason_counts": dict(sorted(final_reasons.items())),
        "action_eligibility_counts": dict(sorted(action_eligibility.items())),
        "action_rejection_reason_counts": dict(sorted(action_rejections.items())),
        "comparison_rejection_reason_counts": dict(
            sorted(comparison_rejections.items())
        ),
        "geometry_maximum_ratio": {
            key: _distribution(value) for key, value in sorted(geometry_ratios.items())
        },
        "geometry_empirical_percentile": {
            key: _distribution(value)
            for key, value in sorted(geometry_percentiles.items())
        },
        "geometry_empirical_tail_probability": {
            key: _distribution(value)
            for key, value in sorted(geometry_tail_probabilities.items())
        },
        "geometry_finite_sample_tail_floor": {
            key: _distribution(value)
            for key, value in sorted(geometry_tail_floors.items())
        },
        "compatibility_shrinkage": {
            key: _distribution(value) for key, value in sorted(shrinkages.items())
        },
        "geometry_calibration_methods": sorted(geometry_methods),
        "geometry_ensemble_cardinality_rules": sorted(
            geometry_cardinality_rules
        ),
        "formal_conformal_geometry_claimed": False,
        "exact_b_fallback_byte_identity": exact_b,
        "evaluation_labels_opened": False,
        "terminal_oracle_used": False,
        "utility_kind": "downstream_classifier_utility_not_NELBO",
        "primary_estimand": PRIMARY_ESTIMAND,
        "policy_gain_threshold_units": CASE_CONTRIBUTION_METRIC_NAME,
    }
    return {**body, "diagnostic_hash": canonical_hash(body)}


__all__ = ("build_prelabel_diagnostics",)
