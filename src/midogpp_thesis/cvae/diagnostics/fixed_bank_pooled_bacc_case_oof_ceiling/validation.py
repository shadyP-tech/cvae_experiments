"""Byte-first, fail-closed semantic validation for a completed v2 bundle."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ...generation.contracts import EXPECTED_GENERATION_LOCK_HASH
from ...protocol import ProtocolError
from ...runtime.frozen_source_streams import load_frozen_source_streams
from ...runtime.label_free_action_predictions import load_global_prediction_seal
from .artifact_io import persist_or_validate_csv, read_json
from .bundle import assert_closed_world, validate_content_index
from .core_hashing import canonical_hash, require_sha256
from .experiment_contracts import (
    CENTERS,
    EXPECTED_CENTER_FOLD_COUNT,
    EXPECTED_MIXED_CLASS_CASE_COUNT,
    EXPECTED_NEGATIVE_ONLY_CASE_COUNT,
    EXPECTED_NULL_ACTION_COUNT,
    EXPECTED_POSITIVE_ONLY_CASE_COUNT,
    PERMUTATION_COUNT,
)


def validate_fixed_bank_pooled_bacc_case_oof_ceiling_bundle(
    root: Path, *, config: object
) -> Mapping[str, object]:
    """Validate the closed byte inventory before any scientific JSON is opened."""

    assert_closed_world(root, allow_incomplete=False, allow_pending_validation=True)
    content = validate_content_index(
        root, config_contract_hash=str(config.contract_hash)
    )

    protocol = read_json(root / "manifests/protocol_manifest.json")
    partition = read_json(root / "manifests/case_oof_partition.json")
    probability = read_json(root / "manifests/sealed_probability_surface.json")
    loco_statistics = read_json(
        root / "manifests/loco_sufficient_statistic_surfaces.json"
    )
    priors = read_json(
        root / "manifests/loco_global_and_pairwise_prior_seals.json"
    )
    support_statistics = read_json(
        root / "manifests/fold_support_sufficient_statistic_surfaces.json"
    )
    posteriors = read_json(root / "manifests/fold_posterior_seals.json")
    decisions = read_json(root / "manifests/fold_decisions.json")
    decision_seal = read_json(root / "manifests/all_fold_decisions_seal.json")
    null_seal = read_json(root / "manifests/permutation_null_decision_seal.json")
    evaluation_statistics = read_json(
        root / "manifests/evaluation_sufficient_statistic_surface.json"
    )
    evaluation = read_json(root / "manifests/ceiling_evaluation.json")
    capability = read_json(root / "reports/label_capability_report.json")
    leakage = read_json(root / "reports/leakage_report.json")
    publication = read_json(root / "reports/publication_decision.json")
    runtime = read_json(root / "reports/runtime_summary.json")
    preflight = read_json(root / "reports/workstation_preflight.json")
    prediction_phase = read_json(
        root / "reports/phase_01_global_prediction_seal_complete.json"
    )
    fold_metric_rows = _read_csv(root / "tables/pooled_oof_fold_metrics.csv")

    source_cache = load_frozen_source_streams(
        root,
        expected_config_hash=str(config.contract_hash),
        expected_generation_lock_hash=EXPECTED_GENERATION_LOCK_HASH,
    )
    predictions = load_global_prediction_seal(
        root,
        expected_config_hash=str(config.contract_hash),
        expected_source_lock_hash=source_cache.lock_hash,
        expected_partition_lock_hash=str(partition.get("partition_hash", "")),
        expected_target_cache_binding_hash=str(
            protocol.get("test_cache_binding_hash", "")
        ),
    )
    require_sha256(str(partition.get("partition_hash", "")), "partition_hash")
    require_sha256(str(probability.get("surface_hash", "")), "probability_surface_hash")
    null_path = root / "arrays/permutation_null_actions.npy"
    null_actions = np.load(null_path, allow_pickle=False)
    _semantic_rebuild(
        root,
        config=config,
        predictions=predictions,
        partition_payload=partition,
        probability_payload=probability,
        loco_statistics_payload=loco_statistics,
        priors_payload=priors,
        support_statistics_payload=support_statistics,
        posteriors_payload=posteriors,
        decisions_payload=decisions,
        decision_seal_payload=decision_seal,
        null_seal_payload=null_seal,
        null_actions=null_actions,
        evaluation_statistics_payload=evaluation_statistics,
        evaluation_payload=evaluation,
        capability_payload=capability,
    )
    _validate_closed_reports(
        config=config,
        source_cache=source_cache,
        predictions=predictions,
        protocol=protocol,
        probability=probability,
        capability=capability,
        leakage=leakage,
        publication=publication,
        runtime=runtime,
        preflight=preflight,
        prediction_phase=prediction_phase,
        evaluation=evaluation,
    )
    raw_priors = _list(priors.get("priors"), "priors")
    raw_posteriors = _list(posteriors.get("posteriors"), "posteriors")
    raw_decisions = _list(decisions.get("decisions"), "decisions")
    loco_surfaces = _list(loco_statistics.get("surfaces"), "LOCO statistic surfaces")
    support_surfaces = _list(
        support_statistics.get("surfaces"), "support statistic surfaces"
    )
    _validate_sufficient_statistic_surfaces(loco_surfaces, expected_count=9)
    _validate_sufficient_statistic_surfaces(
        support_surfaces, expected_count=EXPECTED_CENTER_FOLD_COUNT
    )
    _validate_sufficient_statistic_surfaces((evaluation_statistics,), expected_count=1)

    if (
        protocol.get("experiment_id") != config.experiment_id
        or protocol.get("config_contract_hash") != config.contract_hash
        or protocol.get("support_utility") != "pooled_exact_bacc"
        or protocol.get("uncertainty_unit") != "paired_whole_case_cluster"
        or protocol.get("per_case_bacc_defined_or_persisted") is not False
        or protocol.get("v1_output_or_scratch_consumed") is not False
        or partition.get("fold_count") != 5
        or len(partition.get("folds", [])) != EXPECTED_CENTER_FOLD_COUNT
        or partition.get("evaluation_case_coverage_exactly_once") is not True
        or probability.get("global_prediction_seal_hash") != predictions.seal_hash
        or probability.get("probability_store_hash") != predictions.store.store_hash
        or predictions.store.target_cache_binding_hash
        != protocol.get("test_cache_binding_hash")
        or len(raw_priors) != len(CENTERS)
        or len(raw_posteriors) != EXPECTED_CENTER_FOLD_COUNT
        or len(raw_decisions) != EXPECTED_CENTER_FOLD_COUNT
        or decision_seal.get("fold_decision_count") != EXPECTED_CENTER_FOLD_COUNT
        or decision_seal.get("all_fold_decisions_sealed_before_evaluation_labels")
        is not True
        or capability.get("status") != "PASS"
        or capability.get("evaluation_labels_opened") is not True
        or capability.get("fold_decision_count") != EXPECTED_CENTER_FOLD_COUNT
        or capability.get("sealed_null_action_count") != EXPECTED_NULL_ACTION_COUNT
        or not isinstance(capability.get("manifest_case_class_topology"), Mapping)
        or capability["manifest_case_class_topology"].get("mixed_class_case_count")
        != EXPECTED_MIXED_CLASS_CASE_COUNT
        or capability["manifest_case_class_topology"].get("negative_only_case_count")
        != EXPECTED_NEGATIVE_ONLY_CASE_COUNT
        or capability["manifest_case_class_topology"].get("positive_only_case_count")
        != EXPECTED_POSITIVE_ONLY_CASE_COUNT
        or leakage.get("status") != "PASS"
        or leakage.get("pooled_exact_bacc_used") is not True
        or leakage.get("per_case_bacc_used") is not False
        or leakage.get("v1_output_or_scratch_consumed") is not False
        or publication.get("decision")
        != "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"
        or publication.get("promotion_eligible") is not False
        or publication.get("may_feed_another_stage90") is not False
        or publication.get("scientific_result_hash")
        != evaluation.get("scientific_result_hash")
        or runtime.get("classifier_cell_count") != 729
        or runtime.get("scratch_root")
        != "/data/local/fixed_bank_pooled_bacc_case_oof_ceiling_v2"
        or runtime.get("v1_artifact_or_scratch_reused") is not False
        or len(fold_metric_rows) != EXPECTED_CENTER_FOLD_COUNT
        or null_actions.shape != (PERMUTATION_COUNT, EXPECTED_CENTER_FOLD_COUNT)
        or null_actions.dtype != np.uint8
        or null_actions.size != EXPECTED_NULL_ACTION_COUNT
        or np.any(null_actions > 8)
        or null_seal.get("action_array_member")
        != "arrays/permutation_null_actions.npy"
        or null_seal.get("action_array_sha256") != _sha256_file(null_path)
        or null_seal.get("action_array_value_sha256") != _sha256_array(null_actions)
        or null_seal.get("null_action_count") != EXPECTED_NULL_ACTION_COUNT
        or null_seal.get("observed_decision_seal_hash")
        != decision_seal.get("decision_seal_hash")
        or null_seal.get("config_contract_hash") != config.contract_hash
        or null_seal.get("sealed_before_evaluation_labels") is not True
        or null_seal.get("evaluation_labels_used_to_generate_actions") is not False
        or null_seal.get("permutation_baseline_B_fixed") is not True
        or null_seal.get("candidate_multiset_preserved") is not True
    ):
        raise ProtocolError("Pooled-BACC scientific bundle invariants drifted.")

    prior_targets = tuple(str(value.get("target_center")) for value in raw_priors)
    posterior_keys = tuple(
        (str(value.get("target_center")), int(value.get("fold_ordinal", -1)))
        for value in raw_posteriors
    )
    decision_keys = tuple(
        (str(value.get("target_center")), int(value.get("fold_ordinal", -1)))
        for value in raw_decisions
    )
    expected_keys = tuple((center, fold) for center in CENTERS for fold in range(5))
    if prior_targets != CENTERS or posterior_keys != expected_keys or decision_keys != expected_keys:
        raise ProtocolError("Pooled-BACC prior/posterior/decision order drifted.")
    for prior in raw_priors:
        _validate_direct_hash(prior, "prior_hash")
        if (
            prior.get("H_labels_used_in_G_H") is not False
            or prior.get("G_H_sealed_before_H_support_access") is not True
            or not isinstance(prior.get("pairwise_estimates"), list)
        ):
            raise ProtocolError("Pooled-BACC prior seal boundary drifted.")
    for posterior in raw_posteriors:
        _validate_direct_hash(posterior, "posterior_hash")
        if (
            posterior.get("evaluation_labels_used") is not False
            or posterior.get("uncertainty_unit") != "paired_whole_case_cluster"
        ):
            raise ProtocolError("Pooled-BACC posterior boundary drifted.")
    _validate_direct_hash(decision_seal, "decision_seal_hash")
    _validate_direct_hash(evaluation, "scientific_result_hash")
    if capability.get("all_decisions_seal_hash") != decision_seal.get("decision_seal_hash"):
        raise ProtocolError("Label capability differs from the observed decision seal.")
    null_hash = str(
        null_seal.get("permutation_decision_seal_hash", null_seal.get("plan_hash", ""))
    )
    require_sha256(null_hash, "null_decision_seal_hash")
    if capability.get("null_decision_seal_hash") != null_hash:
        raise ProtocolError("Label capability differs from the null-decision seal.")
    _assert_no_per_case_bacc_columns(
        (
            root / "tables/loco_case_action_sufficient_statistics.csv",
            root / "tables/fold_support_case_action_sufficient_statistics.csv",
            root / "tables/oof_evaluation_case_action_sufficient_statistics.csv",
        )
    )
    result = {
        "schema_version": "midogpp_pooled_bacc_case_oof_validation_v2",
        "status": "PASS",
        "content_hash": content["content_hash"],
        "config_contract_hash": config.contract_hash,
        "source_stream_lock_hash": source_cache.lock_hash,
        "global_prediction_seal_hash": predictions.seal_hash,
        "probability_surface_hash": probability["surface_hash"],
        "decision_seal_hash": decision_seal["decision_seal_hash"],
        "null_decision_seal_hash": null_hash,
        "loco_prior_count": len(raw_priors),
        "fold_posterior_count": len(raw_posteriors),
        "fold_decision_count": len(raw_decisions),
        "null_action_count": int(null_actions.size),
        "content_index_validated_before_scientific_members": True,
        "nonrepairing_validation": True,
        "closed_world": True,
        "pooled_exact_bacc": True,
        "paired_whole_case_cluster_uncertainty": True,
        "per_case_bacc_persisted": False,
        "terminal_consumed_test_diagnostic_only": True,
        "fresh_evidence": False,
        "promotion_eligible": False,
        "may_feed_another_stage90": False,
        "v1_artifact_or_scratch_consumed": False,
        "validator_reopened_scoped_labels_after_complete_preevaluation_seals": True,
        "validator_label_use": "reconstructive_nonselective_only",
        "raw_labels_persisted": False,
    }
    _validate_excluded_control_members(root, result)
    return result


def _semantic_rebuild(
    root: Path,
    *,
    config: object,
    predictions: object,
    partition_payload: Mapping[str, object],
    probability_payload: Mapping[str, object],
    loco_statistics_payload: Mapping[str, object],
    priors_payload: Mapping[str, object],
    support_statistics_payload: Mapping[str, object],
    posteriors_payload: Mapping[str, object],
    decisions_payload: Mapping[str, object],
    decision_seal_payload: Mapping[str, object],
    null_seal_payload: Mapping[str, object],
    null_actions: np.ndarray,
    evaluation_statistics_payload: Mapping[str, object],
    evaluation_payload: Mapping[str, object],
    capability_payload: Mapping[str, object],
) -> None:
    """Recompute every scientific layer instead of trusting coherent hashes."""

    from .case_partitions import CaseFold, CaseOOFPartition
    from .core_contracts import CaseIdentityRow
    from .decisions import DecisionConfig, make_fold_decision, seal_fold_decisions
    from .execution_adapter import seed_probability_rows
    from .inputs import load_label_free_test_frame
    from .label_capabilities import LabelCapabilityManager
    from .permutation_plan import (
        PermutationDecisionPlan,
        build_permutation_decision_plan,
    )
    from .pooled_evaluation import evaluate_statistics_seal
    from .pooled_metrics import (
        score_evaluation_statistics_after_preevaluation_seals,
        score_fold_support_statistics,
        score_loco_prior_statistics,
    )
    from .pooled_posterior import PosteriorConfig, fit_pooled_fold_posterior
    from .pooled_prior import PriorConfig, fit_pooled_loco_prior
    from .probability_surface import aggregate_exact_nine_probabilities

    seed_rows = seed_probability_rows(predictions)
    probabilities = aggregate_exact_nine_probabilities(seed_rows)
    expected_probability = {
        **probabilities.to_payload(),
        "global_prediction_seal_hash": predictions.seal_hash,
    }
    if dict(probability_payload) != expected_probability:
        raise ProtocolError(
            "Persisted pooled probabilities do not rebuild from the sealed prediction store."
        )
    _validate_csv_payloads(
        root / "tables/seed_probability_rows.csv",
        tuple(row.to_payload() for row in seed_rows),
    )
    _validate_csv_payloads(
        root / "tables/aggregated_probability_rows.csv",
        tuple(row.to_payload() for row in probabilities.rows),
    )

    partition = _rebuild_partition(partition_payload, CaseIdentityRow, CaseFold, CaseOOFPartition)
    manager = LabelCapabilityManager(
        config.test_manifest_path,
        load_label_free_test_frame(config),
        partition,
        global_prediction_seal_hash=predictions.seal_hash,
    )
    prior_config = PriorConfig(
        variance_floor=float(config.global_prior["variance_floor"]),
        confidence_multiplier=float(config.global_prior["confidence_multiplier"]),
        minimum_gain=float(config.global_prior["minimum_gain"]),
        tie_tolerance=float(config.global_prior["tie_tolerance"]),
    )
    loco_surface_list: list[object] = []
    prior_list: list[object] = []
    for center in CENTERS:
        labels = manager.open_loco_prior_labels(center)
        surface = score_loco_prior_statistics(
            probabilities, labels, target_center=center
        )
        prior = fit_pooled_loco_prior(center, surface, config=prior_config)
        loco_surface_list.append(surface)
        prior_list.append(prior)
    loco_surfaces = tuple(loco_surface_list)
    expected_priors = tuple(prior_list)
    persisted_loco_surfaces = _list(
        loco_statistics_payload.get("surfaces"), "LOCO statistic surfaces"
    )
    _require_recomputed_surface_payloads(
        loco_surfaces, persisted_loco_surfaces, role="LOCO"
    )
    observed_priors = _list(priors_payload.get("priors"), "priors")
    if tuple(value.to_payload() for value in expected_priors) != observed_priors:
        raise ProtocolError(
            "Pooled LOCO donor effects, variances, pairwise priors, or G_H selection drifted."
        )
    expected_loco_manifest = {
        "schema_version": "fixed_bank_pooled_bacc_all_loco_statistics_v2",
        "surface_count": len(loco_surfaces),
        "surfaces": [value.to_payload() for value in loco_surfaces],
        "per_case_bacc_stored": False,
        "sufficient_statistics_only": True,
    }
    expected_prior_manifest = {
        "schema_version": "fixed_bank_pooled_bacc_all_loco_priors_v2",
        "prior_count": len(expected_priors),
        "priors": [value.to_payload() for value in expected_priors],
        "all_G_H_and_pairwise_priors_sealed_before_H_support_access": True,
        "H_labels_used_in_G_H": False,
        "G_H_shared_across_H": False,
        "pooled_exact_bacc": True,
    }
    if (
        dict(loco_statistics_payload) != expected_loco_manifest
        or dict(priors_payload) != expected_prior_manifest
    ):
        raise ProtocolError("Pooled LOCO manifest closed schema drifted.")
    for prior in expected_priors:
        manager.record_loco_prior_seal(prior.target_center, prior.prior_hash)
    _validate_csv_payloads(
        root / "tables/loco_case_action_sufficient_statistics.csv",
        tuple(_statistic_table_rows(loco_surfaces)),
    )
    _validate_csv_payloads(
        root / "tables/loco_global_and_pairwise_priors.csv",
        tuple(_prior_table_rows(expected_priors)),
    )

    posterior_config = PosteriorConfig(
        variance_floor=float(config.posterior["variance_floor"]),
        confidence_multiplier=float(config.posterior["confidence_multiplier"]),
        minimum_gain=float(config.posterior["minimum_gain"]),
    )
    decision_config = DecisionConfig(
        minimum_gain=float(config.decision["minimum_gain"]),
        tie_tolerance=float(config.decision["tie_tolerance"]),
    )
    prior_by_target = {value.target_center: value for value in expected_priors}
    expected_posteriors: list[object] = []
    expected_decisions: list[object] = []
    support_surface_list: list[object] = []
    support_by_fold: dict[tuple[str, int], object] = {}
    for fold in partition.folds:
        prior = prior_by_target[fold.target_center]
        labels = manager.open_fold_support_labels(
            fold.target_center, fold.fold_ordinal
        )
        surface = score_fold_support_statistics(
            probabilities, labels, fold=fold, global_prior=prior
        )
        posterior = fit_pooled_fold_posterior(
            fold, surface, prior, config=posterior_config
        )
        decision = make_fold_decision(
            fold, posterior, prior, config=decision_config
        )
        expected_posteriors.append(posterior)
        expected_decisions.append(decision)
        support_surface_list.append(surface)
        support_by_fold[(fold.target_center, fold.fold_ordinal)] = surface
        manager.record_fold_decision(
            fold.target_center, fold.fold_ordinal, decision.decision_hash
        )
    support_surfaces = tuple(support_surface_list)
    persisted_support_surfaces = _list(
        support_statistics_payload.get("surfaces"), "support statistic surfaces"
    )
    _require_recomputed_surface_payloads(
        support_surfaces, persisted_support_surfaces, role="support"
    )
    if tuple(value.to_payload() for value in expected_posteriors) != _list(
        posteriors_payload.get("posteriors"), "posteriors"
    ):
        raise ProtocolError(
            "Pooled cluster contrasts or normal-normal posteriors failed semantic rebuild."
        )
    if tuple(value.to_payload() for value in expected_decisions) != _list(
        decisions_payload.get("decisions"), "decisions"
    ):
        raise ProtocolError("Pooled fold decisions failed semantic rebuild.")
    expected_support_manifest = {
        "schema_version": "fixed_bank_pooled_bacc_all_fold_support_statistics_v2",
        "surface_count": len(support_surfaces),
        "surfaces": [value.to_payload() for value in support_surfaces],
        "per_case_bacc_stored": False,
        "sufficient_statistics_only": True,
    }
    expected_posterior_manifest = {
        "schema_version": "fixed_bank_pooled_bacc_all_fold_posteriors_v2",
        "posterior_count": len(expected_posteriors),
        "posteriors": [value.to_payload() for value in expected_posteriors],
        "pooled_exact_bacc": True,
        "paired_whole_case_cluster_uncertainty": True,
        "evaluation_labels_used": False,
    }
    expected_decision_manifest = {
        "schema_version": "fixed_bank_pooled_bacc_all_fold_decisions_v2",
        "decision_count": len(expected_decisions),
        "decisions": [value.to_payload() for value in expected_decisions],
        "evaluation_labels_used": False,
    }
    if (
        dict(support_statistics_payload) != expected_support_manifest
        or dict(posteriors_payload) != expected_posterior_manifest
        or dict(decisions_payload) != expected_decision_manifest
    ):
        raise ProtocolError("Pooled support/posterior/decision manifest closed schema drifted.")
    decision_seal = seal_fold_decisions(
        expected_decisions, partition, probabilities
    )
    if decision_seal.to_payload() != dict(decision_seal_payload):
        raise ProtocolError("Pooled all-fold decision seal failed semantic rebuild.")
    _validate_csv_payloads(
        root / "tables/fold_support_case_action_sufficient_statistics.csv",
        tuple(_statistic_table_rows(support_surfaces)),
    )
    _validate_csv_payloads(
        root / "tables/fold_posteriors.csv",
        tuple(_posterior_table_rows(expected_posteriors)),
    )
    _validate_csv_payloads(
        root / "tables/fold_decisions.csv",
        tuple(value.to_payload() for value in expected_decisions),
    )

    permutation_plan = build_permutation_decision_plan(
        partition,
        probabilities,
        expected_priors,
        support_by_fold,
        posterior_config=posterior_config,
        decision_config=decision_config,
        permutation_seed=int(config.evaluation["permutation_seed"]),
        permutation_count=int(config.evaluation["permutation_count"]),
    )
    if not np.array_equal(permutation_plan.action_codes, null_actions):
        raise ProtocolError("Pooled null actions failed blocked-support semantic rebuild.")
    action_path = root / "arrays/permutation_null_actions.npy"
    expected_null = {
        **permutation_plan.to_payload(),
        "observed_decision_seal_hash": decision_seal.decision_seal_hash,
        "config_contract_hash": str(config.contract_hash),
        "action_array_member": "arrays/permutation_null_actions.npy",
        "action_array_sha256": _sha256_file(action_path),
        "action_array_value_sha256": _sha256_array(null_actions),
        "null_action_count": int(null_actions.size),
        "permutation_baseline_B_fixed": True,
        "candidate_multiset_preserved": True,
        "evaluation_utility_used_for_tie_break": False,
    }
    _assert_exact_mapping(
        null_seal_payload,
        expected_null,
        role="pooled null decision metadata",
    )
    manager.record_preevaluation_seals(
        decision_seal.decision_seal_hash,
        permutation_plan.plan_hash,
        decision_count=len(expected_decisions),
        permutation_count=permutation_plan.permutation_count,
        null_action_count=int(permutation_plan.action_codes.size),
    )
    evaluation_labels = manager.open_oof_evaluation_labels()
    evaluation_statistics = score_evaluation_statistics_after_preevaluation_seals(
        probabilities,
        evaluation_labels,
        decision_seal=decision_seal,
        permutation_plan=permutation_plan,
    )
    _require_recomputed_surface_payloads(
        (evaluation_statistics,), (evaluation_statistics_payload,), role="evaluation"
    )
    if dict(manager.access_report()) != dict(capability_payload):
        raise ProtocolError("Persisted label-capability event chain failed replay.")
    expected_evaluation = evaluate_statistics_seal(
        decision_seal,
        partition,
        evaluation_statistics,
        permutation_plan=permutation_plan,
        tie_tolerance=float(config.decision["tie_tolerance"]),
        confidence_level=float(config.evaluation["confidence_level"]),
    )
    if expected_evaluation.to_payload() != dict(evaluation_payload):
        raise ProtocolError(
            "Pooled center metrics, equal-center inference, or null summary failed semantic rebuild."
        )
    _validate_csv_payloads(
        root / "tables/oof_evaluation_case_action_sufficient_statistics.csv",
        tuple(_statistic_table_rows((evaluation_statistics,))),
    )
    _validate_evaluation_csvs(root, expected_evaluation)


def _rebuild_partition(
    payload: Mapping[str, object], identity_type: type, fold_type: type, partition_type: type
) -> object:
    identities = tuple(
        identity_type(
            target_center=str(row["target_center"]),
            case_id=str(row["case_id"]),
            sample_id=str(row["sample_id"]),
        )
        for row in _list(payload.get("identities"), "partition identities")
    )
    folds = tuple(
        fold_type(
            target_center=str(row["target_center"]),
            fold_ordinal=int(row["fold_ordinal"]),
            support_case_ids=tuple(str(value) for value in row["support_case_ids"]),
            evaluation_case_ids=tuple(str(value) for value in row["evaluation_case_ids"]),
        )
        for row in _list(payload.get("folds"), "partition folds")
    )
    partition = partition_type(
        identities=identities,
        folds=folds,
        partition_seed=int(payload["partition_seed"]),
        partition_hash=str(payload["partition_hash"]),
        fold_count=int(payload["fold_count"]),
    )
    expected = {
        **partition.to_payload(),
        "evaluation_case_coverage_exactly_once": True,
        "support_evaluation_disjoint": True,
        "target_expert_excluded": True,
        "label_free_partition": True,
    }
    # The typed payload already contains the four flags; this update documents
    # the closed-world comparison and remains idempotent.
    if dict(payload) != expected:
        raise ProtocolError("Persisted case partition failed typed semantic rebuild.")
    return partition


def _statistic_table_rows(surfaces: Sequence[object]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for ordinal, surface in enumerate(surfaces):
        for row in surface.rows:
            output.append(
                {
                    **row.to_payload(),
                    "surface_ordinal": ordinal,
                    "label_scope": surface.label_scope,
                    "statistics_surface_hash": surface.statistics_surface_hash,
                    "prerequisite_seal_hash": surface.prerequisite_seal_hash,
                }
            )
    return output


def _require_recomputed_surface_payloads(
    recomputed: Sequence[object],
    persisted: Sequence[Mapping[str, object]],
    *,
    role: str,
) -> None:
    expected = tuple(value.to_payload() for value in recomputed)
    if expected != tuple(dict(value) for value in persisted):
        raise ProtocolError(
            f"{role} sufficient statistics do not rebuild from canonical labels and probabilities."
        )


def _prior_table_rows(priors: Sequence[object]) -> list[dict[str, object]]:
    return [
        {
            "target_center": value.target_center,
            "global_action_id": value.global_action_id,
            "prior_hash": value.prior_hash,
            "pooled_prior_payload": json.dumps(
                value.to_payload(), sort_keys=True, separators=(",", ":")
            ),
        }
        for value in priors
    ]


def _posterior_table_rows(posteriors: Sequence[object]) -> list[dict[str, object]]:
    return [
        {
            "target_center": value.target_center,
            "fold_ordinal": value.fold_ordinal,
            "posterior_hash": value.posterior_hash,
            "pooled_posterior_payload": json.dumps(
                value.to_payload(), sort_keys=True, separators=(",", ":")
            ),
        }
        for value in posteriors
    ]


def _validate_evaluation_csvs(root: Path, evaluation: object) -> None:
    mappings = (
        ("tables/pooled_oof_fold_metrics.csv", ("fold_metric_rows", "fold_rows")),
        ("tables/pooled_oof_center_metrics.csv", ("center_metric_rows", "center_rows")),
        ("tables/equal_center_inference.csv", ("equal_center_inference_rows", "inference_rows")),
        ("tables/action_selection_metrics.csv", ("action_selection_rows",)),
        (
            "tables/permutation_null_summary.csv",
            ("permutation_null_summary_rows", "permutation_rows"),
        ),
    )
    for member, names in mappings:
        rows = None
        for name in names:
            candidate = getattr(evaluation, name, None)
            if candidate:
                rows = candidate
                break
        if rows is None:
            raise ProtocolError(f"Rebuilt evaluation lacks required rows: {names}.")
        _validate_csv_payloads(
            root / member, tuple(value.to_payload() for value in rows)
        )


def _validate_csv_payloads(
    path: Path, rows: Sequence[Mapping[str, object]]
) -> None:
    converted = tuple(_table_row(value) for value in rows)
    if not converted:
        raise ProtocolError(f"Semantic rebuild produced an empty table: {path}.")
    columns = tuple(converted[0])
    if any(tuple(row) != columns for row in converted):
        raise ProtocolError(f"Semantic rebuild produced drifting columns: {path}.")
    persist_or_validate_csv(path, converted, columns)


def _table_row(row: Mapping[str, object]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in row.items():
        output[str(key)] = (
            json.dumps(value, sort_keys=True, separators=(",", ":"))
            if isinstance(value, (Mapping, list, tuple))
            else value
        )
    return output


def _validate_sufficient_statistic_surfaces(
    surfaces: Sequence[Mapping[str, object]], *, expected_count: int
) -> None:
    if len(surfaces) != expected_count:
        raise ProtocolError("Pooled sufficient-statistic surface count drifted.")
    required = {
        "n_positive",
        "true_positive",
        "n_negative",
        "true_negative",
    }
    for surface in surfaces:
        rows = _list(surface.get("rows"), "sufficient-statistic rows")
        if not rows or surface.get("hard_threshold") != 0.5:
            raise ProtocolError("Pooled sufficient-statistic surface is malformed.")
        for row in rows:
            bacc_keys = {str(key) for key in row if "bacc" in str(key).lower()}
            if (
                not required.issubset(row)
                or bacc_keys.difference({"per_case_bacc_stored"})
                or row.get("per_case_bacc_stored") is not False
            ):
                raise ProtocolError("Per-case BACC entered a sufficient-statistic row.")
            if int(row["n_positive"]) + int(row["n_negative"]) <= 0:
                raise ProtocolError("Empty case statistic entered the v2 bundle.")


def _validate_direct_hash(payload: Mapping[str, object], field: str) -> None:
    unhashed = {str(key): value for key, value in payload.items() if key != field}
    if payload.get(field) != canonical_hash(unhashed):
        raise ProtocolError(f"Pooled-BACC {field} drifted.")


def _list(value: object, role: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list) or not all(isinstance(row, Mapping) for row in value):
        raise ProtocolError(f"Pooled-BACC {role} are malformed.")
    return tuple(value)


def _assert_no_per_case_bacc_columns(paths: Sequence[Path]) -> None:
    for path in paths:
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                fields = csv.DictReader(handle).fieldnames
        except OSError as exc:
            raise ProtocolError(f"Cannot read pooled statistic table: {path}.") from exc
        forbidden = (
            []
            if fields is None
            else [
                value
                for value in fields
                if "bacc" in value.lower() and value != "per_case_bacc_stored"
            ]
        )
        if fields is None or forbidden:
            raise ProtocolError("Per-case BACC column entered a statistic table.")


def _read_csv(path: Path) -> tuple[dict[str, str], ...]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return tuple(dict(row) for row in csv.DictReader(handle))
    except OSError as exc:
        raise ProtocolError(f"Cannot read pooled evaluation table: {path}.") from exc


def _validate_excluded_control_members(
    root: Path, reconstructed_validation: Mapping[str, object]
) -> None:
    """Authenticate mutable control members excluded from the scientific index."""

    from .reports import run_state_payload

    state = read_json(root / "reports/run_state.json")
    report_path = root / "reports/validation_report.json"
    if report_path.is_file():
        if (
            state != run_state_payload("COMPLETE", "COMPLETE")
            or read_json(report_path) != dict(reconstructed_validation)
        ):
            raise ProtocolError(
                "Completed pooled-BACC run-state or validation report drifted."
            )
        return
    expected_running = run_state_payload(
        "RUNNING", "CLOSED_WORLD_CONTENT_FIRST_VALIDATION"
    )
    if state != expected_running:
        raise ProtocolError(
            "Pre-report pooled-BACC validation requires the exact running phase."
        )


def _validate_closed_reports(
    *,
    config: object,
    source_cache: object,
    predictions: object,
    protocol: Mapping[str, object],
    probability: Mapping[str, object],
    capability: Mapping[str, object],
    leakage: Mapping[str, object],
    publication: Mapping[str, object],
    runtime: Mapping[str, object],
    preflight: Mapping[str, object],
    prediction_phase: Mapping[str, object],
    evaluation: Mapping[str, object],
) -> None:
    from .execution_adapter import runtime_summary_payload
    from .reports import (
        leakage_report_payload,
        protocol_manifest_payload,
        publication_decision_payload,
    )

    input_hashes = protocol.get("input_artifact_hashes")
    firewall = protocol.get("pre_gpu_firewall")
    if (
        not isinstance(input_hashes, Mapping)
        or set(input_hashes) != set(config.input_artifact_ids)
        or not isinstance(firewall, Mapping)
    ):
        raise ProtocolError("Pooled protocol manifest input/firewall schema drifted.")
    firewall_keys = {
        "status",
        "bank_lock_hash",
        "expert_count",
        "fresh_source_only_training",
        "evaluation_split",
        "manifest_sha256",
        "test_split_previously_consumed",
        "ledger_parent_sha256",
        "ledger_amendment_sha256",
        "fresh_evidence",
        "prior_stage90_output_consumed",
        "v1_output_or_scratch_consumed",
        "target_labels_opened",
        "target_expert_used",
        "gpu_work_authorized",
        "workspace_binding",
    }
    workspace_binding = firewall.get("workspace_binding")
    if (
        set(firewall) != firewall_keys
        or not isinstance(workspace_binding, Mapping)
        or set(workspace_binding)
        != {"status", "experiment_id", "output_artifact_id", "stage", "claim_scope"}
    ):
        raise ProtocolError("Pooled protocol firewall closed schema drifted.")
    expected_protocol = protocol_manifest_payload(
        config,
        input_artifact_hashes={str(key): str(value) for key, value in input_hashes.items()},
        cache_binding_hash=str(protocol.get("test_cache_binding_hash", "")),
        firewall=firewall,
    )
    _assert_exact_mapping(protocol, expected_protocol, role="protocol manifest")

    expected_prediction_phase = {
        "schema_version": "midogpp_pooled_bacc_global_prediction_phase_v2",
        "status": "COMPLETE_BEFORE_ANY_LABEL_ACCESS",
        "global_prediction_seal_hash": predictions.seal_hash,
        "prediction_store_hash": probability["probability_store_hash"],
        "probability_surface_hash": probability["surface_hash"],
        "cell_count": len(predictions.store.cells),
        "all_target_rows_predicted": True,
        "support_labels_opened": False,
        "evaluation_labels_opened": False,
        "v1_prediction_artifact_reused": False,
    }
    _assert_exact_mapping(
        prediction_phase,
        expected_prediction_phase,
        role="global prediction phase report",
    )

    preflight_keys = {
        "schema_version",
        "status",
        "generation_devices",
        "persistent_gpu_workers",
        "classifier_workers",
        "blas_threads_per_classifier_worker",
        "gpu_then_cpu_phase_order",
        "phase_disjoint_gpu_and_cpu_pools",
        "parent_cuda_initialized",
        "tf32_enabled",
        "amp_enabled",
        "scratch_preference",
        "available_cpu_affinity_count",
        "physical_ram_bytes",
        "disk_probe_path",
        "disk_free_bytes_at_launch",
        "thread_environment",
        "cuda_visible_devices",
        "package_versions",
        "gpus",
    }
    if set(preflight) != preflight_keys:
        raise ProtocolError("Pooled workstation preflight closed schema drifted.")
    staging = runtime.get("local_source_staging")
    if not isinstance(staging, Mapping):
        raise ProtocolError("Pooled runtime staging report is malformed.")
    required_staging = {"attempted", "used", "status", "workstation_preflight"}
    if (
        not required_staging.issubset(staging)
        or set(staging).difference(required_staging | {"failure"})
        or staging.get("workstation_preflight") != preflight
    ):
        raise ProtocolError("Pooled runtime staging/preflight schema drifted.")
    expected_runtime = runtime_summary_payload(
        source_cache=source_cache,
        prediction_capability=predictions,
        local_staging=staging,
    )
    _assert_exact_mapping(runtime, expected_runtime, role="runtime summary")
    expected_leakage = leakage_report_payload(
        prediction_seal_hash=predictions.seal_hash,
        prior_count=9,
        decision_count=EXPECTED_CENTER_FOLD_COUNT,
        null_action_count=EXPECTED_NULL_ACTION_COUNT,
        capability_report=capability,
    )
    _assert_exact_mapping(leakage, expected_leakage, role="leakage report")
    _assert_exact_mapping(
        publication,
        publication_decision_payload(evaluation),
        role="publication decision",
    )


def _assert_exact_mapping(
    observed: Mapping[str, object],
    expected: Mapping[str, object],
    *,
    role: str,
) -> None:
    if dict(observed) != dict(expected):
        raise ProtocolError(f"{role} closed schema or payload drifted.")


def _sha256_file(path: Path) -> str:
    from ...runtime.artifact_io import sha256_file

    return sha256_file(path)


def _sha256_array(values: np.ndarray) -> str:
    from ...runtime.artifact_io import sha256_array

    return sha256_array(values)


__all__ = ("validate_fixed_bank_pooled_bacc_case_oof_ceiling_bundle",)
