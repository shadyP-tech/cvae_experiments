"""Deterministic persistence split along label-capability phase boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from ...runtime.artifact_io import atomic_json, atomic_npy, read_json, sha256_array, sha256_file
from .artifact_io import persist_or_validate_csv, persist_or_validate_json
from .reports import protocol_manifest_payload, publication_decision_payload, run_state_payload


_PERMUTATION_DECISION_TIE_BREAK = (
    "lexicographic_action_id_no_evaluation_utility_access"
)


def persist_initial_surfaces(
    root: Path,
    *,
    config: object,
    provenance: Mapping[str, Mapping[str, object]],
    frame: object,
    firewall: Mapping[str, object],
    partition: object,
) -> None:
    input_hashes = {
        artifact_id: stable_hash(provenance[artifact_id])
        for artifact_id in getattr(config, "input_artifact_ids")
    }
    persist_or_validate_json(
        root / "manifests/protocol_manifest.json",
        protocol_manifest_payload(
            config,
            input_artifact_hashes=input_hashes,
            cache_binding_hash=str(frame.cache_binding_hash),
            firewall=firewall,
        ),
    )
    partition_payload = {
        "schema_version": "fixed_bank_label_aware_case_oof_partition_v1",
        "partition_seed": int(partition.partition_seed),
        "fold_count": int(partition.fold_count),
        "identities": [row.to_payload() for row in partition.identities],
        "folds": [fold.to_payload() for fold in partition.folds],
        "partition_hash": str(partition.partition_hash),
        "evaluation_case_coverage_exactly_once": True,
        "support_evaluation_disjoint": True,
        "target_expert_excluded": True,
    }
    persist_or_validate_json(root / "manifests/case_oof_partition.json", partition_payload)
    fold_rows: list[dict[str, object]] = []
    for fold in partition.folds:
        for role, cases in (("support", fold.support_case_ids), ("evaluation", fold.evaluation_case_ids)):
            for case_id in cases:
                fold_rows.append(
                    {
                        "target_center": fold.target_center,
                        "fold_ordinal": fold.fold_ordinal,
                        "fold_id": fold.fold_id,
                        "case_id": case_id,
                        "role": role,
                        "fold_hash": fold.fold_hash,
                        "partition_hash": partition.partition_hash,
                    }
                )
    _persist_rows(root / "tables/case_oof_partitions.csv", fold_rows)


def persist_probability_surface(
    root: Path,
    *,
    prediction_capability: object,
    seed_rows: Sequence[object],
    probabilities: object,
) -> None:
    _persist_rows(root / "tables/seed_probability_rows.csv", [_payload(row) for row in seed_rows])
    _persist_rows(root / "tables/aggregated_probability_rows.csv", [_payload(row) for row in probabilities.rows])
    persist_or_validate_json(
        root / "manifests/sealed_probability_surface.json",
        {
            "schema_version": "fixed_bank_label_aware_probability_surface_v1",
            "probability_store_hash": probabilities.probability_store_hash,
            "surface_hash": probabilities.surface_hash,
            "row_count": len(probabilities.rows),
            "rows": [_payload(row) for row in probabilities.rows],
            "global_prediction_seal_hash": prediction_capability.seal_hash,
            "predictions_globally_sealed_before_labels": True,
            "labels_readable_during_materialization": False,
        },
    )
    persist_or_validate_json(
        root / "reports/phase_01_global_prediction_seal_complete.json",
        {
            "schema_version": "midogpp_label_aware_global_prediction_phase_v1",
            "status": "COMPLETE_BEFORE_ANY_LABEL_ACCESS",
            "global_prediction_seal_hash": prediction_capability.seal_hash,
            "prediction_store_hash": probabilities.probability_store_hash,
            "probability_surface_hash": probabilities.surface_hash,
            "cell_count": len(prediction_capability.store.cells),
            "all_target_rows_predicted": True,
            "support_labels_opened": False,
            "evaluation_labels_opened": False,
        },
    )


def persist_postseal_results(
    root: Path,
    *,
    evaluation: object,
    capability_report: Mapping[str, object],
    leakage_report: Mapping[str, object],
    runtime_summary: Mapping[str, object],
) -> None:
    evaluation_payload = _payload(evaluation)
    persist_or_validate_json(root / "manifests/ceiling_evaluation.json", evaluation_payload)
    _persist_evaluation_tables(root, evaluation)
    persist_or_validate_json(root / "reports/label_capability_report.json", dict(capability_report))
    persist_or_validate_json(root / "reports/leakage_report.json", dict(leakage_report))
    persist_or_validate_json(root / "reports/publication_decision.json", publication_decision_payload(evaluation_payload))
    persist_or_validate_json(root / "reports/runtime_summary.json", dict(runtime_summary))


def persist_and_validate_loco_prior_seals(
    root: Path, priors: Sequence[object]
) -> Mapping[str, object]:
    prior_payloads = [_payload(value) for value in priors]
    payload = {
        "schema_version": "fixed_bank_label_aware_all_loco_priors_v1",
        "prior_count": len(prior_payloads),
        "priors": prior_payloads,
        "all_G_H_sealed_before_H_support_access": True,
        "H_labels_used_in_G_H": False,
        "G_H_shared_across_H": False,
    }
    path = root / "manifests/loco_global_prior_seals.json"
    persist_or_validate_json(path, payload)
    if read_json(path) != payload:
        raise ProtocolError("Durable LOCO prior seals changed before support access.")
    rows = [
        {
            "target_center": prior.target_center,
            "global_action_id": prior.global_action_id,
            "best_candidate_action_id": prior.best_candidate_action_id,
            "candidate_action_id": estimate.action_id,
            "other_center_count": estimate.other_center_count,
            "other_center_case_count": estimate.other_center_case_count,
            "shrunk_mean_gain_vs_b": estimate.shrunk_mean_gain_vs_b,
            "standard_error": estimate.standard_error,
            "lower_confidence_bound": estimate.lower_confidence_bound,
            "estimate_hash": estimate.estimate_hash,
            "prior_hash": prior.prior_hash,
        }
        for prior in priors
        for estimate in prior.estimates
    ]
    _persist_rows(root / "tables/loco_global_priors.csv", rows)
    return payload


def persist_and_validate_preevaluation_seals(
    root: Path,
    *,
    posteriors: Sequence[object],
    decisions: Sequence[object],
    decision_seal: object,
    permutation_seal: object,
    config_contract_hash: str,
) -> Mapping[str, object]:
    posterior_payloads = [_payload(value) for value in posteriors]
    decision_payloads = [_payload(value) for value in decisions]
    posterior_manifest = {
        "schema_version": "fixed_bank_label_aware_all_fold_posteriors_v1",
        "posterior_count": len(posterior_payloads),
        "posteriors": posterior_payloads,
        "exact_response_only": True,
        "smooth_response_used": False,
    }
    decision_manifest = {
        "schema_version": "fixed_bank_label_aware_all_fold_decisions_v1",
        "decision_count": len(decision_payloads),
        "decisions": decision_payloads,
        "evaluation_labels_used": False,
    }
    decision_seal_payload = _payload(decision_seal)
    persist_or_validate_json(root / "manifests/fold_posterior_seals.json", posterior_manifest)
    persist_or_validate_json(root / "manifests/fold_decisions.json", decision_manifest)
    persist_or_validate_json(root / "manifests/all_fold_decisions_seal.json", decision_seal_payload)

    actions = _permutation_action_array(permutation_seal)
    action_path = root / "arrays/permutation_null_actions.npy"
    if action_path.is_file():
        observed = np.load(action_path, allow_pickle=False)
        if observed.dtype != actions.dtype or observed.shape != actions.shape or not np.array_equal(observed, actions):
            raise ProtocolError("Existing permutation-null action array differs and will not be repaired.")
    else:
        atomic_npy(action_path, actions)
    permutation_payload = _payload(permutation_seal)
    if (
        permutation_payload.get("permutation_decision_tie_break")
        != _PERMUTATION_DECISION_TIE_BREAK
        or permutation_payload.get(
            "evaluation_utility_used_for_permutation_tie_break"
        )
        is not False
    ):
        raise ProtocolError("Permutation decision-plan tie boundary drifted.")
    permutation_payload["observed_decision_seal_hash"] = str(
        decision_seal_payload["decision_seal_hash"]
    )
    permutation_payload["config_contract_hash"] = str(config_contract_hash)
    permutation_payload["action_array_member"] = "arrays/permutation_null_actions.npy"
    permutation_payload["action_array_sha256"] = sha256_file(action_path)
    permutation_payload["action_array_value_sha256"] = sha256_array(actions)
    permutation_path = root / "manifests/permutation_null_decision_seal.json"
    persist_or_validate_json(permutation_path, permutation_payload)

    for path, expected in (
        (root / "manifests/fold_posterior_seals.json", posterior_manifest),
        (root / "manifests/fold_decisions.json", decision_manifest),
        (root / "manifests/all_fold_decisions_seal.json", decision_seal_payload),
        (permutation_path, permutation_payload),
    ):
        if read_json(path) != expected:
            raise ProtocolError("Durable pre-evaluation seal changed before label access.")
    reloaded_actions = np.load(action_path, allow_pickle=False)
    if (
        reloaded_actions.dtype != actions.dtype
        or reloaded_actions.shape != actions.shape
        or not np.array_equal(reloaded_actions, actions)
        or sha256_file(action_path) != permutation_payload["action_array_sha256"]
    ):
        raise ProtocolError("Durable permutation-null action bytes changed before label access.")

    posterior_rows = [
        {
            "target_center": posterior.target_center,
            "fold_ordinal": posterior.fold_ordinal,
            "candidate_action_id": estimate.action_id,
            "support_case_count": estimate.support_case_count,
            "prior_mean_gain_vs_g": estimate.prior_mean_gain_vs_g,
            "posterior_mean_gain_vs_g": estimate.posterior_mean_gain_vs_g,
            "standard_error": estimate.standard_error,
            "lower_confidence_bound": estimate.lower_confidence_bound,
            "estimate_hash": estimate.estimate_hash,
            "posterior_hash": posterior.posterior_hash,
        }
        for posterior in posteriors
        for estimate in posterior.estimates
    ]
    _persist_rows(root / "tables/fold_posteriors.csv", posterior_rows)
    _persist_rows(root / "tables/fold_decisions.csv", decision_payloads)
    return permutation_payload


def _permutation_action_array(value: object) -> np.ndarray:
    for name in ("action_codes", "action_ordinals", "actions", "null_action_ordinals"):
        raw = getattr(value, name, None)
        if raw is not None:
            array = np.asarray(raw)
            break
    else:
        raise TypeError("Permutation decision seal must expose compact action ordinals.")
    if array.ndim != 2 or array.shape[1] != 45 or array.dtype.kind not in "ui":
        raise ProtocolError("Permutation-null actions must be an integer [permutation,45] array.")
    if np.any(array < 0) or np.any(array > 8):
        raise ProtocolError("Permutation-null action ordinal escaped the nine-action menu.")
    return np.ascontiguousarray(array, dtype=np.uint8)


def persist_validation_report(root: Path, payload: Mapping[str, object]) -> None:
    persist_or_validate_json(root / "reports/validation_report.json", payload)


def write_run_state(root: Path, *, status: str, phase: str, error: str | None = None) -> None:
    atomic_json(root / "reports/run_state.json", run_state_payload(status, phase, error=error))


def _persist_evaluation_tables(root: Path, evaluation: object) -> None:
    case_rows = getattr(evaluation, "case_metric_rows", getattr(evaluation, "case_rows", ()))
    center_rows = getattr(evaluation, "center_metric_rows", getattr(evaluation, "center_rows", ()))
    action_selection_rows = getattr(evaluation, "action_selection_rows", ())
    permutation_rows = getattr(
        evaluation,
        "permutation_null_summary_rows",
        getattr(evaluation, "permutation_rows", ()),
    )
    if (
        not case_rows
        or not center_rows
        or len(action_selection_rows) != 20
        or not permutation_rows
    ):
        raise ValueError(
            "Ceiling evaluation must expose case, center, action-selection, and permutation tables."
        )
    _persist_rows(root / "tables/oof_case_metrics.csv", [_payload(row) for row in case_rows])
    _persist_rows(root / "tables/oof_center_metrics.csv", [_payload(row) for row in center_rows])
    _persist_rows(
        root / "tables/action_selection_metrics.csv",
        [_payload(row) for row in action_selection_rows],
    )
    _persist_rows(root / "tables/permutation_null_summary.csv", [_payload(row) for row in permutation_rows])


def _payload(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if hasattr(value, "to_payload"):
        raw = value.to_payload()
        if isinstance(raw, Mapping):
            return {str(key): _json_value(item) for key, item in raw.items()}
    raw_dict = getattr(value, "__dict__", None)
    if isinstance(raw_dict, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in raw_dict.items()
            if not str(key).startswith("_")
        }
    raise TypeError("Label-aware persisted object must be mapping-like.")


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "to_payload"):
        return _json_value(value.to_payload())
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    return value


def _persist_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    values = tuple(_table_row(row) for row in rows)
    if not values:
        raise ValueError(f"Cannot persist empty label-aware table: {path}.")
    columns = tuple(values[0])
    if any(tuple(row) != columns for row in values):
        raise ValueError(f"Label-aware table columns drifted: {path}.")
    persist_or_validate_csv(path, values, columns)


def _table_row(row: Mapping[str, object]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in row.items():
        converted = _json_value(value)
        output[str(key)] = (
            json.dumps(converted, sort_keys=True, separators=(",", ":"))
            if isinstance(converted, (Mapping, list, tuple))
            else converted
        )
    return output


__all__ = (
    "persist_and_validate_loco_prior_seals",
    "persist_and_validate_preevaluation_seals",
    "persist_initial_surfaces",
    "persist_postseal_results",
    "persist_probability_surface",
    "persist_validation_report",
    "write_run_state",
)
