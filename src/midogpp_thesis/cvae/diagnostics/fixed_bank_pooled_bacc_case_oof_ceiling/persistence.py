"""Non-repairing persistence split at every label-capability boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from ...runtime.artifact_io import atomic_json, atomic_npy, read_json, sha256_array, sha256_file
from .artifact_io import persist_or_validate_csv, persist_or_validate_json
from .experiment_contracts import EXPECTED_CENTER_FOLD_COUNT, EXPECTED_NULL_ACTION_COUNT, PERMUTATION_COUNT
from .reports import protocol_manifest_payload, publication_decision_payload, run_state_payload


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
        "schema_version": "fixed_bank_pooled_bacc_case_oof_partition_v2",
        "partition_seed": int(partition.partition_seed),
        "fold_count": int(partition.fold_count),
        "identities": [_payload(row) for row in partition.identities],
        "folds": [_payload(fold) for fold in partition.folds],
        "partition_hash": str(partition.partition_hash),
        "evaluation_case_coverage_exactly_once": True,
        "support_evaluation_disjoint": True,
        "target_expert_excluded": True,
        "label_free_partition": True,
    }
    persist_or_validate_json(root / "manifests/case_oof_partition.json", partition_payload)
    rows: list[dict[str, object]] = []
    for fold in partition.folds:
        for role, cases in (
            ("support", fold.support_case_ids),
            ("evaluation", fold.evaluation_case_ids),
        ):
            for case_id in cases:
                rows.append(
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
    _persist_rows(root / "tables/case_oof_partitions.csv", rows)


def persist_probability_surface(
    root: Path,
    *,
    prediction_capability: object,
    seed_rows: Sequence[object],
    probabilities: object,
) -> None:
    _persist_rows(
        root / "tables/seed_probability_rows.csv", [_payload(row) for row in seed_rows]
    )
    _persist_rows(
        root / "tables/aggregated_probability_rows.csv",
        [_payload(row) for row in probabilities.rows],
    )
    persist_or_validate_json(
        root / "manifests/sealed_probability_surface.json",
        {
            **_payload(probabilities),
            "global_prediction_seal_hash": prediction_capability.seal_hash,
        },
    )
    persist_or_validate_json(
        root / "reports/phase_01_global_prediction_seal_complete.json",
        {
            "schema_version": "midogpp_pooled_bacc_global_prediction_phase_v2",
            "status": "COMPLETE_BEFORE_ANY_LABEL_ACCESS",
            "global_prediction_seal_hash": prediction_capability.seal_hash,
            "prediction_store_hash": probabilities.probability_store_hash,
            "probability_surface_hash": probabilities.surface_hash,
            "cell_count": len(prediction_capability.store.cells),
            "all_target_rows_predicted": True,
            "support_labels_opened": False,
            "evaluation_labels_opened": False,
            "v1_prediction_artifact_reused": False,
        },
    )


def persist_and_validate_loco_prior_seals(
    root: Path,
    *,
    statistic_surfaces: Sequence[object],
    priors: Sequence[object],
) -> Mapping[str, object]:
    if len(statistic_surfaces) != 9 or len(priors) != 9:
        raise ProtocolError("Pooled-BACC requires nine LOCO statistic/prior surfaces.")
    statistics_payload = {
        "schema_version": "fixed_bank_pooled_bacc_all_loco_statistics_v2",
        "surface_count": len(statistic_surfaces),
        "surfaces": [_payload(value) for value in statistic_surfaces],
        "per_case_bacc_stored": False,
        "sufficient_statistics_only": True,
    }
    priors_payload = {
        "schema_version": "fixed_bank_pooled_bacc_all_loco_priors_v2",
        "prior_count": len(priors),
        "priors": [_payload(value) for value in priors],
        "all_G_H_and_pairwise_priors_sealed_before_H_support_access": True,
        "H_labels_used_in_G_H": False,
        "G_H_shared_across_H": False,
        "pooled_exact_bacc": True,
    }
    statistic_path = root / "manifests/loco_sufficient_statistic_surfaces.json"
    prior_path = root / "manifests/loco_global_and_pairwise_prior_seals.json"
    persist_or_validate_json(statistic_path, statistics_payload)
    persist_or_validate_json(prior_path, priors_payload)
    if read_json(statistic_path) != statistics_payload or read_json(prior_path) != priors_payload:
        raise ProtocolError("Durable pooled LOCO surfaces changed before support access.")
    _persist_rows(
        root / "tables/loco_case_action_sufficient_statistics.csv",
        _statistic_rows(statistic_surfaces),
    )
    _persist_rows(
        root / "tables/loco_global_and_pairwise_priors.csv",
        [_prior_table_row(value) for value in priors],
    )
    return priors_payload


def persist_and_validate_preevaluation_seals(
    root: Path,
    *,
    support_surfaces: Sequence[object],
    posteriors: Sequence[object],
    decisions: Sequence[object],
    decision_seal: object,
    permutation_seal: object,
    config_contract_hash: str,
) -> Mapping[str, object]:
    if not (
        len(support_surfaces)
        == len(posteriors)
        == len(decisions)
        == EXPECTED_CENTER_FOLD_COUNT
    ):
        raise ProtocolError("Pooled-BACC pre-evaluation inventory must contain 45 folds.")
    support_manifest = {
        "schema_version": "fixed_bank_pooled_bacc_all_fold_support_statistics_v2",
        "surface_count": len(support_surfaces),
        "surfaces": [_payload(value) for value in support_surfaces],
        "per_case_bacc_stored": False,
        "sufficient_statistics_only": True,
    }
    posterior_manifest = {
        "schema_version": "fixed_bank_pooled_bacc_all_fold_posteriors_v2",
        "posterior_count": len(posteriors),
        "posteriors": [_payload(value) for value in posteriors],
        "pooled_exact_bacc": True,
        "paired_whole_case_cluster_uncertainty": True,
        "evaluation_labels_used": False,
    }
    decision_manifest = {
        "schema_version": "fixed_bank_pooled_bacc_all_fold_decisions_v2",
        "decision_count": len(decisions),
        "decisions": [_payload(value) for value in decisions],
        "evaluation_labels_used": False,
    }
    decision_payload = _payload(decision_seal)
    persist_or_validate_json(
        root / "manifests/fold_support_sufficient_statistic_surfaces.json",
        support_manifest,
    )
    persist_or_validate_json(root / "manifests/fold_posterior_seals.json", posterior_manifest)
    persist_or_validate_json(root / "manifests/fold_decisions.json", decision_manifest)
    persist_or_validate_json(root / "manifests/all_fold_decisions_seal.json", decision_payload)

    actions = _permutation_action_array(permutation_seal)
    action_path = root / "arrays/permutation_null_actions.npy"
    if action_path.is_file():
        observed = np.load(action_path, allow_pickle=False)
        if (
            observed.dtype != actions.dtype
            or observed.shape != actions.shape
            or not np.array_equal(observed, actions)
        ):
            raise ProtocolError("Existing pooled null-action array differs and will not be repaired.")
    else:
        atomic_npy(action_path, actions)
    permutation_payload = _payload(permutation_seal)
    if (
        permutation_payload.get("sealed_before_evaluation_labels") is not True
        or permutation_payload.get("evaluation_labels_used_to_generate_actions") is not False
        or permutation_payload.get("baseline_action_permuted") is not False
        or permutation_payload.get("candidate_multiset_preserved_per_case") is not True
        or permutation_payload.get(
            "evaluation_utility_used_for_permutation_tie_break"
        )
        is not False
    ):
        raise ProtocolError("Pooled null-decision seal crossed the evaluation boundary.")
    permutation_payload.update(
        {
            "observed_decision_seal_hash": str(decision_payload["decision_seal_hash"]),
            "config_contract_hash": str(config_contract_hash),
            "action_array_member": "arrays/permutation_null_actions.npy",
            "action_array_sha256": sha256_file(action_path),
            "action_array_value_sha256": sha256_array(actions),
            "null_action_count": int(actions.size),
            "permutation_baseline_B_fixed": True,
            "candidate_multiset_preserved": True,
            "evaluation_utility_used_for_tie_break": False,
        }
    )
    permutation_path = root / "manifests/permutation_null_decision_seal.json"
    persist_or_validate_json(permutation_path, permutation_payload)
    for path, expected in (
        (root / "manifests/fold_support_sufficient_statistic_surfaces.json", support_manifest),
        (root / "manifests/fold_posterior_seals.json", posterior_manifest),
        (root / "manifests/fold_decisions.json", decision_manifest),
        (root / "manifests/all_fold_decisions_seal.json", decision_payload),
        (permutation_path, permutation_payload),
    ):
        if read_json(path) != expected:
            raise ProtocolError("Durable pooled pre-evaluation seal changed before label access.")
    reloaded = np.load(action_path, allow_pickle=False)
    if not np.array_equal(reloaded, actions) or sha256_file(action_path) != permutation_payload["action_array_sha256"]:
        raise ProtocolError("Durable pooled null-action bytes changed before label access.")
    _persist_rows(
        root / "tables/fold_support_case_action_sufficient_statistics.csv",
        _statistic_rows(support_surfaces),
    )
    _persist_rows(
        root / "tables/fold_posteriors.csv",
        [_posterior_table_row(value) for value in posteriors],
    )
    _persist_rows(
        root / "tables/fold_decisions.csv", [_payload(value) for value in decisions]
    )
    return permutation_payload


def persist_postseal_results(
    root: Path,
    *,
    evaluation_statistics: object,
    evaluation: object,
    capability_report: Mapping[str, object],
    leakage_report: Mapping[str, object],
    runtime_summary: Mapping[str, object],
) -> None:
    statistic_payload = _payload(evaluation_statistics)
    evaluation_payload = _payload(evaluation)
    persist_or_validate_json(
        root / "manifests/evaluation_sufficient_statistic_surface.json",
        statistic_payload,
    )
    persist_or_validate_json(root / "manifests/ceiling_evaluation.json", evaluation_payload)
    _persist_rows(
        root / "tables/oof_evaluation_case_action_sufficient_statistics.csv",
        _statistic_rows((evaluation_statistics,)),
    )
    _persist_evaluation_tables(root, evaluation)
    persist_or_validate_json(root / "reports/label_capability_report.json", dict(capability_report))
    persist_or_validate_json(root / "reports/leakage_report.json", dict(leakage_report))
    persist_or_validate_json(
        root / "reports/publication_decision.json",
        publication_decision_payload(evaluation_payload),
    )
    persist_or_validate_json(root / "reports/runtime_summary.json", dict(runtime_summary))


def persist_validation_report(root: Path, payload: Mapping[str, object]) -> None:
    persist_or_validate_json(root / "reports/validation_report.json", payload)


def write_run_state(
    root: Path, *, status: str, phase: str, error: str | None = None
) -> None:
    atomic_json(root / "reports/run_state.json", run_state_payload(status, phase, error=error))


def _permutation_action_array(value: object) -> np.ndarray:
    for name in ("action_codes", "action_ordinals", "actions", "null_action_ordinals"):
        raw = getattr(value, name, None)
        if raw is not None:
            array = np.asarray(raw)
            break
    else:
        raise TypeError("Pooled permutation seal must expose compact action ordinals.")
    if (
        array.shape != (PERMUTATION_COUNT, EXPECTED_CENTER_FOLD_COUNT)
        or array.dtype.kind not in "ui"
        or np.any(array < 0)
        or np.any(array > 8)
        or array.size != EXPECTED_NULL_ACTION_COUNT
    ):
        raise ProtocolError("Pooled null actions must be integer [10000,45] ordinals 0..8.")
    return np.ascontiguousarray(array, dtype=np.uint8)


def _statistic_rows(surfaces: Sequence[object]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for ordinal, surface in enumerate(surfaces):
        for row in getattr(surface, "rows"):
            output.append(
                {
                    **_payload(row),
                    "surface_ordinal": ordinal,
                    "label_scope": str(getattr(surface, "label_scope")),
                    "statistics_surface_hash": str(
                        getattr(surface, "statistics_surface_hash")
                    ),
                    "prerequisite_seal_hash": str(
                        getattr(surface, "prerequisite_seal_hash")
                    ),
                }
            )
    return output


def _prior_table_row(value: object) -> dict[str, object]:
    payload = _payload(value)
    return {
        "target_center": payload.get("target_center"),
        "global_action_id": payload.get("global_action_id"),
        "prior_hash": payload.get("prior_hash"),
        "pooled_prior_payload": json.dumps(payload, sort_keys=True, separators=(",", ":")),
    }


def _posterior_table_row(value: object) -> dict[str, object]:
    payload = _payload(value)
    return {
        "target_center": payload.get("target_center"),
        "fold_ordinal": payload.get("fold_ordinal"),
        "posterior_hash": payload.get("posterior_hash"),
        "pooled_posterior_payload": json.dumps(payload, sort_keys=True, separators=(",", ":")),
    }


def _persist_evaluation_tables(root: Path, evaluation: object) -> None:
    fold_rows = _rows_attr(evaluation, "fold_metric_rows", "fold_rows")
    center_rows = _rows_attr(evaluation, "center_metric_rows", "center_rows")
    inference_rows = _rows_attr(
        evaluation, "equal_center_inference_rows", "inference_rows"
    )
    action_rows = _rows_attr(evaluation, "action_selection_rows")
    permutation_rows = _rows_attr(
        evaluation, "permutation_null_summary_rows", "permutation_rows"
    )
    _persist_rows(root / "tables/pooled_oof_fold_metrics.csv", [_payload(row) for row in fold_rows])
    _persist_rows(root / "tables/pooled_oof_center_metrics.csv", [_payload(row) for row in center_rows])
    _persist_rows(root / "tables/equal_center_inference.csv", [_payload(row) for row in inference_rows])
    _persist_rows(root / "tables/action_selection_metrics.csv", [_payload(row) for row in action_rows])
    _persist_rows(root / "tables/permutation_null_summary.csv", [_payload(row) for row in permutation_rows])


def _rows_attr(value: object, *names: str) -> Sequence[object]:
    for name in names:
        rows = getattr(value, name, None)
        if rows:
            return rows
    raise ProtocolError(f"Pooled evaluation lacks table rows: {names}.")


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
    raise TypeError("Pooled-BACC persisted object must be mapping-like.")


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
        raise ProtocolError(f"Cannot persist empty pooled-BACC table: {path}.")
    columns = tuple(values[0])
    if any(tuple(row) != columns for row in values):
        raise ProtocolError(f"Pooled-BACC table columns drifted: {path}.")
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
