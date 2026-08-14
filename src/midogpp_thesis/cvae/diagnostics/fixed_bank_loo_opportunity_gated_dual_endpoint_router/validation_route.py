"""Reconstruct label-free plans, scoped route fits, gains, and decisions."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from .artifact_rows import row_payload
from .artifact_topology import (
    CASE_ACTION_SUFFICIENT_STAT_COUNT,
    CORRECTNESS_OBSERVATION_COUNT,
    DIRECTIONAL_SUPPORT_GAIN_COUNT,
    DONOR_PRIOR_COUNT,
    IDENTIFICATION_DECISION_COUNT,
    MODEL_FIT_COUNT,
    ROBUST_ARM_DECISION_COUNT,
)
from .artifact_writers import read_rows
from .correctness_proxy import build_label_free_features
from .donor_prior import compute_donor_priors
from .hashing import canonical_hash
from .reports import seal_payload
from .response_products import deduplicate_sufficient_stats
from .response_scoring import score_case_action_confusions
from .robust import build_endpoint_arms
from .runner_runtime import compute_route_job, exact_route_blas_scope
from .split_plans import build_whole_case_loo_plans, seal_whole_case_loo_plans


def reconstruct_route_products(
    root: Path,
    *,
    config: object,
    frame: object,
    surface: object,
    physical_prelabel_seal_hash: str,
    label_firewall: object,
) -> Mapping[str, object]:
    plans = build_whole_case_loo_plans(
        frame.rows, probability_surface_hash=str(surface.surface_hash)
    )
    plan_seal = seal_whole_case_loo_plans(
        plans, probability_surface_hash=str(surface.surface_hash)
    )
    features = build_label_free_features(surface)
    plan_rows = tuple(row.to_payload() for row in plans)
    feature_rows = tuple(row.to_payload() for row in features)
    if read_rows(root / "tables/whole_case_loo_plans.csv") != plan_rows:
        raise ProtocolError("Dual-endpoint LOO plan table is not reconstructive.")
    if read_rows(root / "tables/label_free_candidate_features.csv") != feature_rows:
        raise ProtocolError("Dual-endpoint feature table is not reconstructive.")
    persisted_plan = seal_payload(
        "fixed_bank_dual_endpoint_loo_plan_seal_v1",
        bindings={
            "physical_prelabel_seal_hash": physical_prelabel_seal_hash,
            "plans_hash": canonical_hash(plan_rows),
            "science_plan_seal_hash": plan_seal.plan_seal_hash,
        },
        plan_count=len(plan_rows),
        held_case_and_group_excluded=True,
        labels_used=False,
    )
    if read_json(root / "manifests/loo_plan_seal.json") != persisted_plan:
        raise ProtocolError("Dual-endpoint LOO plan seal is not reconstructive.")
    feature_seal = seal_payload(
        "fixed_bank_dual_endpoint_label_free_feature_seal_v1",
        bindings={
            "physical_prelabel_seal_hash": physical_prelabel_seal_hash,
            "loo_plan_seal_hash": persisted_plan["seal_hash"],
            "features_hash": canonical_hash(feature_rows),
        },
        feature_count=len(feature_rows),
        labels_used=False,
        feature_blocks_sealed_before_support_labels=True,
    )
    if read_json(root / "manifests/label_free_feature_seal.json") != feature_seal:
        raise ProtocolError("Dual-endpoint feature seal is not reconstructive.")

    donor_counts: list[object] = []
    donor_priors: list[object] = []
    priors_by_target: dict[str, tuple[object, ...]] = {}
    from .constants import CENTERS, candidate_sources

    for target in CENTERS:
        by_source: dict[str, tuple[object, ...]] = {}
        for source in candidate_sources(target):
            labels = label_firewall.open_donor_labels(target, source)
            counts = tuple(score_case_action_confusions(surface, labels))
            by_source[source] = counts
            donor_counts.extend(counts)
        priors = compute_donor_priors(by_source, heldout_center=target)
        priors_by_target[target] = priors
        donor_priors.extend(priors)

    jobs = []
    for plan in plans:
        labels = label_firewall.open_route_support_labels(
            plan.target_center, plan.case_id, plan_hash=plan.plan_hash
        )
        route_cases = {*plan.support_case_ids, plan.case_id}
        jobs.append(
            {
                "plan": plan,
                "support_labels": labels,
                "donor_priors": priors_by_target[plan.target_center],
                "route_features": tuple(
                    row
                    for row in features
                    if row.target_center == plan.target_center
                    and row.case_id in route_cases
                ),
            }
        )
    with exact_route_blas_scope(3):
        results = tuple(compute_route_job(surface, job) for job in jobs)
    route_counts = tuple(
        row for result in results for row in result.case_action_confusions
    )
    stats = deduplicate_sufficient_stats((*donor_counts, *route_counts))
    gains = tuple(
        row for result in results for row in result.directional_support_gains
    )
    identification = tuple(
        row for result in results for row in result.identification_decisions
    )
    robust = tuple(row for result in results for row in result.robust_arm_decisions)
    observations = tuple(
        _tag(row, "canonical")
        for result in results
        for row in result.correctness_observations_primary
    ) + tuple(
        _tag(row, "candidate_feature_block_permuted")
        for result in results
        for row in result.correctness_observations_permuted
    )
    models = tuple(
        _tag(row, "canonical")
        for result in results
        for row in result.model_fits_primary
    ) + tuple(
        _tag(row, "candidate_feature_block_permuted")
        for result in results
        for row in result.model_fits_permuted
    )
    expected_counts = (
        len(stats),
        len(observations),
        len(models),
        len(gains),
        len(donor_priors),
        len(identification),
        len(robust),
    )
    if expected_counts != (
        CASE_ACTION_SUFFICIENT_STAT_COUNT,
        CORRECTNESS_OBSERVATION_COUNT,
        MODEL_FIT_COUNT,
        DIRECTIONAL_SUPPORT_GAIN_COUNT,
        DONOR_PRIOR_COUNT,
        IDENTIFICATION_DECISION_COUNT,
        ROBUST_ARM_DECISION_COUNT,
    ):
        raise ProtocolError(f"Dual-endpoint replay topology drifted: {expected_counts}.")
    _compare(root, "case_action_confusions.csv", stats)
    _compare(root, "route_correctness_observations.csv", observations)
    _compare(root, "route_model_fits.csv", models)
    _compare(root, "directional_support_gains.csv", gains)
    _compare(root, "donor_priors.csv", donor_priors)
    _compare(root, "endpoint_arms.csv", build_endpoint_arms())
    _compare(root, "identification_decisions.csv", identification)
    _compare(root, "robust_arm_decisions.csv", robust)
    return {
        "plans": plans,
        "plan_seal": plan_seal,
        "persisted_plan_seal": persisted_plan,
        "features": features,
        "feature_seal": feature_seal,
        "case_action_sufficient_stats": stats,
        "directional_support_gains": gains,
        "donor_priors": tuple(donor_priors),
        "identification_decisions": identification,
        "robust_arm_decisions": robust,
        "route_results": results,
    }


def _tag(row: object, feature_surface_id: str) -> dict[str, object]:
    return {"feature_surface_id": feature_surface_id, **row_payload(row)}


def _compare(root: Path, filename: str, rows: Sequence[object]) -> None:
    expected = tuple(row_payload(row) for row in rows)
    if read_rows(root / "tables" / filename) != expected:
        raise ProtocolError(f"Dual-endpoint table is not reconstructive: {filename}.")


__all__ = ("reconstruct_route_products",)
