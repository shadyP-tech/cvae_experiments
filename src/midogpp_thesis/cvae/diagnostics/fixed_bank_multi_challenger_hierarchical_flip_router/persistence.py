"""Atomic, phase-bound persistence for the multi-challenger bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...runtime.artifact_io import atomic_json, read_json
from .actions import build_action_library
from .artifact_io import json_value, object_payload, persist_json, persist_rows
from .hashing import canonical_hash
from .reports import protocol_manifest_payload, run_state_payload
from .terminal_schema import (
    TERMINAL_TABLE_FIELDS,
    TERMINAL_TABLE_MEMBERS,
    canonical_terminal_rows,
)


TERMINAL_CHECKPOINT_MEMBER = "checkpoints/terminal_evaluation/sealed_result.json"
_TERMINAL_RESULT_KEYS = frozenset(
    {
        "terminal_case_confusions",
        "terminal_center_metrics",
        "terminal_contrasts",
        "router_identification_metrics",
        "permutation_metrics",
        "menu_oracle_metrics",
        "sealed_terminal_evaluation",
    }
)
_TERMINAL_CHECKPOINT_KEYS = frozenset(
    {
        "schema_version",
        "result",
        "capability_report",
        "leakage_report",
        "publication_decision",
        "runtime_summary",
        "raw_labels_persisted",
        "terminal_products_only",
        "checkpoint_hash",
    }
)
_FORBIDDEN_RAW_KEYS = frozenset(
    {"label", "labels", "ground_truth", "true_label", "image_path", "sample_path"}
)


def persist_initial_surfaces(
    root: Path,
    *,
    config: object,
    protocol: object,
    provenance: Mapping[str, Mapping[str, object]],
    frame: object,
    firewall: Mapping[str, object],
    partition: object,
) -> None:
    input_ids = tuple(getattr(config, "input_artifact_ids"))
    if set(provenance) != set(input_ids) or len(provenance) != 6:
        raise ProtocolError("Multi-challenger initial provenance must cover six inputs.")
    persist_json(
        root / "manifests/protocol_manifest.json",
        protocol_manifest_payload(
            config,
            protocol=protocol,
            input_artifact_hashes={
                artifact_id: canonical_hash(provenance[artifact_id])
                for artifact_id in input_ids
            },
            cache_binding_hash=str(getattr(frame, "cache_binding_hash")),
            firewall=firewall,
        ),
    )
    persist_json(
        root / "manifests/three_role_partition.json", object_payload(partition)
    )
    partition_rows = []
    for fold in getattr(partition, "folds"):
        for role, cases in (
            ("selection", fold.selection_case_ids),
            ("calibration", fold.calibration_case_ids),
            ("evaluation", fold.evaluation_case_ids),
        ):
            for case_id in cases:
                partition_rows.append(
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
    _persist_nonempty_rows(
        root / "tables/three_role_partitions.csv", partition_rows
    )
    actions = tuple(build_action_library())
    action_rows = [object_payload(action) for action in actions]
    _persist_nonempty_rows(root / "tables/action_library.csv", action_rows)
    by_target = {
        target: [
            object_payload(action)
            for action in actions
            if action.target_center == target
        ]
        for target in tuple(dict.fromkeys(action.target_center for action in actions))
    }
    unhashed = {
        "schema_version": "fixed_bank_multi_challenger_action_library_manifest_v1",
        "actions": action_rows,
        "action_count": len(actions),
        "physical_actions_per_target": 10,
        "target_expert_used": False,
        "labels_used": False,
        "previous_stage90_predictions_used": False,
    }
    persist_json(
        root / "manifests/action_library.json",
        {**unhashed, "action_library_hash": _stable_library_hash(by_target)},
    )


def persist_prelabel_surfaces(
    root: Path,
    *,
    prediction: object,
    seed_rows: Sequence[object],
    probability_surface: object,
    prelabel: object,
) -> None:
    _persist_nonempty_rows(
        root / "tables/seed_probability_rows.csv",
        [object_payload(row) for row in seed_rows],
    )
    _persist_nonempty_rows(
        root / "tables/aggregated_probability_rows.csv",
        [object_payload(row) for row in probability_surface.rows],
    )
    _persist_nonempty_rows(
        root / "tables/case_action_features.csv",
        [object_payload(row) for row in prelabel.features],
    )
    persist_json(
        root / "manifests/sealed_probability_surface.json",
        {
            "schema_version": "fixed_bank_multi_challenger_probability_seal_v1",
            "global_prediction_seal_hash": prediction.seal_hash,
            "probability_store_hash": probability_surface.probability_store_hash,
            "surface_hash": probability_surface.surface_hash,
            "row_count": len(probability_surface.rows),
            "seed_row_count": len(seed_rows),
            "exact_nine_ensemble_first": True,
            "labels_used": False,
        },
    )
    persist_json(
        root / "manifests/prelabel_feature_seal.json",
        {
            "schema_version": "fixed_bank_multi_challenger_feature_seal_v1",
            "prediction_seal_hash": prediction.seal_hash,
            "probability_surface_hash": prelabel.probability_surface_hash,
            "feature_surface_hash": prelabel.feature_surface_hash,
            "feature_count": len(prelabel.features),
            "sealed_before_label_capabilities": True,
            "feature_hyperparameters_selected_after_labels": False,
            "raw_labels_persisted": False,
        },
    )


def persist_fold_plans(root: Path, plans: Sequence[object]) -> None:
    payloads = [object_payload(plan) for plan in plans]
    unhashed = {
        "schema_version": "fixed_bank_multi_challenger_fold_plan_seals_v1",
        "plans": payloads,
        "plan_count": len(payloads),
        "held_evaluation_labels_used": False,
        "each_plan_invariant_to_held_evaluation_label_values": True,
    }
    persist_json(
        root / "manifests/fold_plan_seals.json",
        {**unhashed, "fold_plan_surface_hash": canonical_hash(unhashed)},
    )


def persist_donor_models(root: Path, donor_phase: object) -> None:
    contribution_rows = [dict(row) for row in donor_phase.contribution_rows]
    fit_rows = [dict(row) for row in donor_phase.fit_rows]
    _persist_nonempty_rows(
        root / "tables/directional_donor_responses.csv", contribution_rows
    )
    _persist_nonempty_rows(root / "tables/model_fits.csv", fit_rows)
    persist_json(
        root / "manifests/donor_model_seals.json",
        {
            "schema_version": "fixed_bank_multi_challenger_donor_model_seals_v1",
            "models": {
                key: dict(value)
                for key, value in sorted(donor_phase.model_seals.items())
            },
            "model_count": len(donor_phase.model_seals),
            "models_are_H_specific": True,
            "strict_H_q_e_exclusion": True,
            "heldout_H_labels_used": False,
        },
    )
    persist_json(
        root / "manifests/permutation_provenance_seal.json",
        donor_phase.permutation_provenance,
    )


def persist_fold_decisions(root: Path, phase: object) -> None:
    menus = [dict(row) for row in phase.menu_rows]
    calibrations = [dict(row) for row in phase.calibration_rows]
    scores = [dict(row) for row in phase.score_rows]
    decisions = [object_payload(row) for row in phase.decisions]
    _persist_nonempty_rows(root / "tables/candidate_menus.csv", menus)
    _persist_nonempty_rows(
        root / "tables/directional_calibrations.csv", calibrations
    )
    _persist_nonempty_rows(root / "tables/candidate_scores.csv", scores)
    _persist_nonempty_rows(root / "tables/method_decisions.csv", decisions)
    menu_unhashed = {
        "schema_version": "fixed_bank_multi_challenger_candidate_menu_seals_v1",
        "menu_count": len(menus),
        "menu_row_hashes": [str(row["row_hash"]) for row in menus],
        "selection_labels_used_only_for_fixed_B_ranked_menus": True,
        "held_evaluation_labels_used": False,
    }
    persist_json(
        root / "manifests/candidate_menu_seals.json",
        {**menu_unhashed, "menu_surface_hash": canonical_hash(menu_unhashed)},
    )
    calibration_unhashed = {
        "schema_version": "fixed_bank_multi_challenger_calibration_seals_v1",
        "calibration_count": len(calibrations),
        "calibration_row_hashes": [str(row["row_hash"]) for row in calibrations],
        "menu_bound": True,
        "shared_model_updated": False,
        "held_evaluation_labels_used": False,
    }
    persist_json(
        root / "manifests/calibration_seals.json",
        {
            **calibration_unhashed,
            "calibration_surface_hash": canonical_hash(calibration_unhashed),
        },
    )
    persist_json(
        root / "manifests/all_method_decisions_seal.json",
        {
            "schema_version": "fixed_bank_multi_challenger_all_decisions_v1",
            "decision_count": len(decisions),
            "fold_seals": {
                f"{key[0]}::{key[1]}": value
                for key, value in sorted(phase.fold_seal_hashes.items())
            },
            "fold_seal_count": len(phase.fold_seal_hashes),
            "decision_bundle_hash": phase.decision_bundle_hash,
            "each_fold_decision_without_its_held_evaluation_labels": True,
            "terminal_evaluation_labels_used": False,
        },
    )


def persist_terminal(
    root: Path,
    *,
    result: Mapping[str, object],
    capability_report: Mapping[str, object],
    leakage_report: Mapping[str, object],
    publication_decision: Mapping[str, object],
    runtime_summary: Mapping[str, object],
) -> None:
    for key, member in TERMINAL_TABLE_MEMBERS.items():
        rows = result.get(key)
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
            raise ProtocolError(f"Multi-challenger terminal table absent: {key}.")
        canonical_rows = canonical_terminal_rows(key, rows)
        persist_rows(
            root / member,
            canonical_rows,
            TERMINAL_TABLE_FIELDS[key],
        )
    sealed = result.get("sealed_terminal_evaluation")
    if not isinstance(sealed, Mapping):
        raise ProtocolError("Multi-challenger terminal seal is absent.")
    persist_json(root / "manifests/sealed_terminal_evaluation.json", sealed)
    persist_json(root / "reports/label_capability_report.json", capability_report)
    persist_json(root / "reports/leakage_report.json", leakage_report)
    persist_json(root / "reports/publication_decision.json", publication_decision)
    persist_json(root / "reports/runtime_summary.json", runtime_summary)


def persist_terminal_checkpoint(
    root: Path,
    *,
    result: Mapping[str, object],
    capability_report: Mapping[str, object],
    leakage_report: Mapping[str, object],
    publication_decision: Mapping[str, object],
    runtime_summary: Mapping[str, object],
) -> Mapping[str, object]:
    converted = json_value(result)
    if not isinstance(converted, Mapping):
        raise ProtocolError("Multi-challenger terminal checkpoint is malformed.")
    sealed = converted.get("sealed_terminal_evaluation")
    if (
        set(converted) != _TERMINAL_RESULT_KEYS
        or not isinstance(sealed, Mapping)
        or sealed.get("raw_labels_persisted") is not False
        or _contains_forbidden_raw_key(converted)
        or any(
            _contains_forbidden_raw_key(value)
            for value in (
                capability_report,
                leakage_report,
                publication_decision,
                runtime_summary,
            )
        )
    ):
        raise ProtocolError("Multi-challenger terminal checkpoint lacks its safe seal.")
    unhashed = {
        "schema_version": "fixed_bank_multi_challenger_terminal_checkpoint_v1",
        "result": dict(converted),
        "capability_report": json_value(capability_report),
        "leakage_report": json_value(leakage_report),
        "publication_decision": json_value(publication_decision),
        "runtime_summary": json_value(runtime_summary),
        "raw_labels_persisted": False,
        "terminal_products_only": True,
    }
    payload = {**unhashed, "checkpoint_hash": canonical_hash(unhashed)}
    persist_json(root / TERMINAL_CHECKPOINT_MEMBER, payload)
    return payload


def load_terminal_checkpoint(root: Path) -> Mapping[str, object]:
    path = root / TERMINAL_CHECKPOINT_MEMBER
    if path.is_symlink() or not path.is_file():
        raise ProtocolError("Multi-challenger terminal checkpoint is absent or unsafe.")
    payload = read_json(path)
    unhashed = {key: value for key, value in payload.items() if key != "checkpoint_hash"}
    if (
        set(payload) != _TERMINAL_CHECKPOINT_KEYS
        or payload.get("checkpoint_hash") != canonical_hash(unhashed)
        or payload.get("schema_version")
        != "fixed_bank_multi_challenger_terminal_checkpoint_v1"
        or payload.get("raw_labels_persisted") is not False
        or payload.get("terminal_products_only") is not True
        or not all(
            isinstance(payload.get(key), Mapping)
            for key in (
                "result",
                "capability_report",
                "leakage_report",
                "publication_decision",
                "runtime_summary",
            )
        )
        or set(payload["result"]) != _TERMINAL_RESULT_KEYS
        or _contains_forbidden_raw_key(payload)
    ):
        raise ProtocolError("Multi-challenger terminal checkpoint drifted.")
    return payload


def finalize_terminal_checkpoint(root: Path) -> None:
    payload = load_terminal_checkpoint(root)
    persist_terminal(
        root,
        result=payload["result"],
        capability_report=payload["capability_report"],
        leakage_report=payload["leakage_report"],
        publication_decision=payload["publication_decision"],
        runtime_summary=payload["runtime_summary"],
    )


def remove_validated_terminal_checkpoint(root: Path) -> None:
    path = root / TERMINAL_CHECKPOINT_MEMBER
    load_terminal_checkpoint(root)
    path.unlink()
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir() or any(parent.iterdir()):
        raise ProtocolError("Multi-challenger checkpoint directory is unsafe.")
    parent.rmdir()
    checkpoint_root = parent.parent
    if (
        checkpoint_root.is_symlink()
        or not checkpoint_root.is_dir()
        or any(checkpoint_root.iterdir())
    ):
        raise ProtocolError("Multi-challenger checkpoint root contains unknown members.")
    checkpoint_root.rmdir()


def write_run_state(
    root: Path,
    *,
    status: str,
    phase: str,
    error: str | None = None,
    error_class: str | None = None,
) -> None:
    path = root / "reports/run_state.json"
    if path.is_symlink():
        raise ProtocolError("Multi-challenger run-state path is a symlink.")
    atomic_json(
        path,
        run_state_payload(
            status, phase, error=error, error_class=error_class
        ),
    )


def persist_validation_report(root: Path, checks: Mapping[str, object]) -> None:
    from .fresh_process_validation import verify_attested_validation_checks

    validated = verify_attested_validation_checks(checks)
    persist_json(root / "reports/validation_report.json", validated)


def _persist_nonempty_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ProtocolError(f"Multi-challenger cannot persist an empty table: {path}.")
    fields = tuple(rows[0])
    if not fields or any(set(row) != set(fields) for row in rows):
        raise ProtocolError(f"Multi-challenger table schema drifted: {path}.")
    persist_rows(path, ({key: row[key] for key in fields} for row in rows), fields)


def _stable_library_hash(payload: object) -> str:
    from ....common.hashing import stable_hash

    return stable_hash(payload)


def _contains_forbidden_raw_key(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key) in _FORBIDDEN_RAW_KEYS or _contains_forbidden_raw_key(item)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list)):
        return any(_contains_forbidden_raw_key(item) for item in value)
    return False


__all__ = (
    "TERMINAL_CHECKPOINT_MEMBER",
    "finalize_terminal_checkpoint",
    "load_terminal_checkpoint",
    "persist_donor_models",
    "persist_fold_decisions",
    "persist_fold_plans",
    "persist_initial_surfaces",
    "persist_prelabel_surfaces",
    "persist_terminal",
    "persist_terminal_checkpoint",
    "persist_validation_report",
    "remove_validated_terminal_checkpoint",
    "write_run_state",
)
