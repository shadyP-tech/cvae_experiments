"""Human- and machine-readable reporting for the non-adoptive pilot."""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence


def decision_summary(
    decision_rows: Sequence[Mapping[str, object]],
    outer_results: Sequence[Mapping[str, object]],
    bootstrap: Mapping[str, object],
    *,
    profile_id: str,
) -> dict[str, object]:
    selections = Counter(str(row["selected_representation"]) for row in decision_rows)
    policy = [row for row in outer_results if row.get("role") == "selected_policy"]
    canonical = [row for row in outer_results if row.get("role") == "canonical_a"]
    mean_policy = sum(float(row["bacc"]) for row in policy) / float(len(policy))
    mean_a = sum(float(row["bacc"]) for row in canonical) / float(len(canonical))
    return {
        "schema_version": "midogpp_physical_multiscale_decision_summary_v1",
        "status": "COMPLETE",
        "profile_id": profile_id,
        "claim_scope": "real_feature_transfer_only",
        "evidence_role": (
            "complete_deterministic_representation_plus_classifier_"
            "pipeline_diagnostic"
        ),
        "non_adoptive": True,
        "representation_selection_frequency": dict(sorted(selections.items())),
        "equal_center_mean_policy_bacc": mean_policy,
        "equal_center_mean_canonical_a_bacc": mean_a,
        "paired_mean_delta": mean_policy - mean_a,
        "conditional_bootstrap": dict(bootstrap),
        "inner_delta_role": "optimistic_selection_statistic",
        "not_performance_estimate": True,
        "gate_is_statistical_test": False,
        "probabilities_calibrated": False,
        "covers_new_center_uncertainty": False,
        "bootstrap_conditions_on_fixed_fits_and_locked_selection": True,
        "global_representation_adoption_allowed": False,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
    }


def render_decision_report(summary: Mapping[str, object]) -> str:
    frequencies = summary["representation_selection_frequency"]
    profile_id = str(summary["profile_id"])
    title = (
        "# MIDOG++ Clipped-Bbox Annotation-Local Pooling Pilot v3"
        if profile_id
        == "physical_multiscale_clipped_bbox_annotation_local_pooling_pilot_v3"
        else "# MIDOG++ Physical Multiscale Center-Pooling Pilot"
    )
    return "\n".join(
        (
            title,
            "",
            "Status: complete non-adoptive Stage-10 diagnostic.",
            "",
            f"- Adaptive-policy mean BACC: `{float(summary['equal_center_mean_policy_bacc']):.6f}`",
            f"- Canonical-A mean BACC: `{float(summary['equal_center_mean_canonical_a_bacc']):.6f}`",
            f"- Paired mean delta: `{float(summary['paired_mean_delta']):+.6f}`",
            f"- Representation choices: `{frequencies}`",
            "",
            "The source-inner B/C-versus-A deltas are optimistic selection statistics,",
            "not unbiased performance estimates, and the fixed gate is not a statistical test.",
            "The paired case bootstrap is conditional on the observed centers, fixed fits,",
            "and locked selections; it does not quantify training, selection, calibration,",
            "probabilistic, or new-center uncertainty.",
            "",
            "The result estimates the complete deterministic representation-plus-classifier",
            "pipeline only. It does not",
            "establish a globally superior representation, routing quality, CVAE preservation,",
            "NELBO compatibility, deployment value, or downstream synthetic utility.",
            "",
        )
    )
