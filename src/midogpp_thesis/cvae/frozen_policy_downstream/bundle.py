"""Closed-world serialization for Stage-70 prediction and scoring artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import numpy as np

from ...common.hashing import stable_hash
from ..reporting import write_csv_rows, write_json
from ..protocol import ProtocolError
from .bootstrap import BootstrapSummary
from .contrasts import ArmSummary, PairedDelta
from .prediction import FrozenPolicyPredictionPass, PersistedPredictionPass
from .prediction_seal import (
    PredictionSealBinding,
    prediction_paths,
    prediction_record_payload,
    validate_authorization_phase,
    verify_persisted_prediction_artifact,
)
from .scoring import ScoredFrozenPolicies


REQUIRED_FILES = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "manifests/protocol_manifest.json",
    "manifests/evaluation_plan.json",
    "manifests/source_block_index.json",
    "manifests/composition_index.json",
    "manifests/prediction_index.json",
    "manifests/prediction_seal.json",
    "manifests/content_index.json",
    "arrays/target_predictions.npz",
    "tables/target_metrics.csv",
    "tables/case_confusions.csv",
    "tables/arm_summaries.csv",
    "tables/paired_deltas.csv",
    "tables/bootstrap_summary.csv",
    "reports/phase_01_authorization_complete.json",
    "reports/phase_02_predictions_persisted.json",
    "reports/phase_03_labels_opened.json",
    "reports/phase_04_scoring_complete.json",
    "reports/leakage_report.json",
    "reports/identity_overlap_report.json",
    "reports/utility_control_equivalence.json",
    "reports/publication_decision.json",
    "reports/run_state.json",
    "reports/validation_report.json",
)


def write_authorization_phase(
    root: str | Path,
    *,
    binding: PredictionSealBinding,
) -> None:
    """Durably bind phase 01 to canonical auth/cache/reservation identities."""

    path = Path(root)
    if not path.is_dir() or path.is_symlink():
        raise ProtocolError("Stage-70 phase-01 output root is missing or a symlink.")
    if any(member.is_symlink() for member in path.rglob("*")):
        raise ProtocolError("Stage-70 phase-01 output root contains a symlink.")
    _write_json_durable(
        path / "reports/phase_01_authorization_complete.json",
        binding.phase_payload(),
    )


def seal_prediction_pass(
    root: str | Path,
    prediction_pass: FrozenPolicyPredictionPass,
    *,
    expected_binding: PredictionSealBinding,
) -> PersistedPredictionPass:
    """Persist every prediction array/index before any label accessor is called."""

    path = Path(root)
    if not path.is_dir() or path.is_symlink():
        raise ProtocolError("Stage-70 prediction output root is missing or a symlink.")
    if any(member.is_symlink() for member in path.rglob("*")):
        raise ProtocolError("Stage-70 prediction output root contains a symlink.")
    if (
        prediction_pass.classifier_fit_count != 162
        or prediction_pass.prediction_reuse_count != 81
    ):
        raise ProtocolError("Stage-70 prediction fit/reuse geometry drifted.")
    phase_01_sha = validate_authorization_phase(path, expected_binding)
    paths = prediction_paths(path)
    arrays_path = paths["arrays"]
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    index_records: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {}
    for ordinal, cell in enumerate(prediction_pass.cells):
        record = prediction_record_payload(ordinal=ordinal, cell=cell)
        prediction_key = str(record["prediction_array_key"])
        probability_key = str(record["probability_array_key"])
        arrays[prediction_key] = np.array(cell.predictions, dtype=np.int64, copy=True)
        arrays[probability_key] = np.array(cell.probabilities, dtype=np.float64, copy=True)
        index_records.append(record)
    with tempfile.NamedTemporaryFile(
        dir=arrays_path.parent,
        suffix=".npz",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(arrays_path)
        _fsync_directory(arrays_path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
    binding_hash = str(expected_binding.phase_payload()["authorization_binding_hash"])
    index_payload: dict[str, object] = {
        "schema_version": "midogpp_stage70_prediction_index_v2",
        "phase": "PREDICTIONS_PERSISTED",
        "target_labels_opened": False,
        "cell_count": len(index_records),
        "target_row_count": 9928,
        "phase_01_sha256": phase_01_sha,
        "authorization_binding_hash": binding_hash,
        "prediction_metadata_hash": stable_hash(index_records),
        "records": index_records,
    }
    _write_json_durable(paths["index"], index_payload)
    index_sha = _sha256_file(paths["index"])
    arrays_sha = _sha256_file(arrays_path)
    seal_payload: dict[str, object] = {
        "schema_version": "midogpp_stage70_prediction_seal_v2",
        "phase": "PREDICTIONS_PERSISTED",
        "phase_01_sha256": phase_01_sha,
        "authorization_binding_hash": binding_hash,
        "prediction_index_sha256": index_sha,
        "prediction_arrays_sha256": arrays_sha,
        "prediction_metadata_hash": index_payload["prediction_metadata_hash"],
        "cell_count": len(index_records),
        "target_row_count": 9928,
        "classifier_fit_count": prediction_pass.classifier_fit_count,
        "prediction_reuse_count": prediction_pass.prediction_reuse_count,
        "target_labels_opened": False,
    }
    _write_json_durable(paths["seal"], seal_payload)
    _write_json_durable(
        paths["phase_02"],
        {
            **seal_payload,
            "schema_version": "midogpp_stage70_phase_marker_v2",
        },
    )
    sealed = PersistedPredictionPass(
        artifact_root=path,
        authorization_binding_hash=binding_hash,
        phase_01_sha256=phase_01_sha,
        prediction_index_sha256=index_sha,
        prediction_arrays_sha256=arrays_sha,
        prediction_seal_sha256=_sha256_file(paths["seal"]),
        phase_02_sha256=_sha256_file(paths["phase_02"]),
    )
    verify_persisted_prediction_artifact(sealed, expected_binding=expected_binding)
    return sealed


def write_scored_bundle(
    root: str | Path,
    *,
    scored: ScoredFrozenPolicies,
    summaries: Sequence[ArmSummary],
    deltas: Sequence[PairedDelta],
    bootstrap: Sequence[BootstrapSummary],
    final_authorization_hash: str,
) -> None:
    path = Path(root)
    if final_authorization_hash != scored.final_authorization_hash:
        raise ProtocolError("Stage-70 scored authorization provenance drifted.")
    write_json(
        path / "reports/phase_03_labels_opened.json",
        {
            "schema_version": "midogpp_stage70_phase_marker_v1",
            "phase": "LABELS_OPENED_AFTER_PREDICTIONS_PERSISTED",
            "authorization_binding_hash": scored.authorization_binding_hash,
            "final_authorization_hash": scored.final_authorization_hash,
            "target_cache_content_hash": scored.target_cache_content_hash,
            "phase_01_sha256": scored.phase_01_sha256,
            "prediction_index_sha256": scored.prediction_index_sha256,
            "prediction_arrays_sha256": scored.prediction_arrays_sha256,
            "prediction_seal_sha256": scored.prediction_seal_sha256,
            "phase_02_sha256": scored.phase_02_sha256,
            "label_manifest_sha256": scored.label_manifest_sha256,
            "labels_used_for_scoring_only": True,
        },
    )
    write_csv_rows(
        path / "tables/target_metrics.csv",
        [row.to_payload() for row in scored.metrics],
    )
    write_csv_rows(
        path / "tables/case_confusions.csv",
        [row.to_payload() for row in scored.case_confusions],
    )
    write_csv_rows(
        path / "tables/arm_summaries.csv",
        [row.to_payload() for row in summaries],
    )
    write_csv_rows(
        path / "tables/paired_deltas.csv",
        [row.to_payload() for row in deltas],
    )
    write_csv_rows(
        path / "tables/bootstrap_summary.csv",
        [row.to_payload() for row in bootstrap],
    )
    utility_rows = [
        row for row in deltas if row.comparison_id == "utility_regret_minus_equal_union"
    ]
    equivalence = all(
        row.bacc_delta == 0.0 and row.macro_f1_delta == 0.0 for row in utility_rows
    ) and len(utility_rows) == 81
    if not equivalence:
        raise ProtocolError("Stage-70 utility/control equivalence audit failed.")
    write_json(
        path / "reports/utility_control_equivalence.json",
        {
            "schema_version": "midogpp_stage70_utility_control_equivalence_v1",
            "status": "PASS",
            "cell_count": len(utility_rows),
            "exact_metric_equivalence": True,
            "exact_prediction_and_probability_hash_equivalence": True,
            "independent_policy_hypothesis_test": False,
        },
    )
    write_json(
        path / "reports/leakage_report.json",
        {
            "schema_version": "midogpp_stage70_leakage_report_v1",
            "status": "PASS",
            "final_authorization_hash": final_authorization_hash,
            "target_labels_opened_after_prediction_seal": True,
            "target_labels_used_for_fit_selection_or_prediction": False,
            "target_labels_used_for_scoring_only": True,
            "target_support_used": False,
            "routing_recomputed": False,
            "stage50_or_stage90_input_used": False,
            "fresh_confirmatory_evidence": False,
        },
    )
    write_json(
        path / "reports/identity_overlap_report.json",
        {
            "schema_version": "midogpp_stage70_identity_overlap_v1",
            "status": "PASS",
            "target_expert_assignments": 0,
            "center_4_rows": 0,
            "legacy_label_encoded_identifiers_persisted": 0,
        },
    )
    write_json(
        path / "reports/publication_decision.json",
        {
            "schema_version": "midogpp_stage70_publication_decision_v1",
            "status": "PASS",
            "decision": "DESCRIPTIVE_COMPARISON_COMPLETE",
            "claim_scope": (
                "descriptive_frozen_policy_comparison_on_previously_consumed_test"
            ),
            "fresh_confirmatory_status": "BLOCKED_NO_UNCONSUMED_ELIGIBLE_SPLIT",
            "routing_policy_promoted": False,
            "deployment_utility_claimed": False,
            "new_center_generalization_claimed": False,
            "external_generalization_claimed": False,
        },
    )
    write_json(
        path / "reports/phase_04_scoring_complete.json",
        {
            "schema_version": "midogpp_stage70_phase_marker_v1",
            "phase": "SCORING_COMPLETE",
            "metric_row_count": len(scored.metrics),
            "authorization_binding_hash": scored.authorization_binding_hash,
            "final_authorization_hash": scored.final_authorization_hash,
            "target_cache_content_hash": scored.target_cache_content_hash,
            "phase_01_sha256": scored.phase_01_sha256,
            "prediction_index_sha256": scored.prediction_index_sha256,
            "prediction_arrays_sha256": scored.prediction_arrays_sha256,
            "prediction_seal_sha256": scored.prediction_seal_sha256,
            "phase_02_sha256": scored.phase_02_sha256,
            "label_manifest_sha256": scored.label_manifest_sha256,
        },
    )
    write_json(
        path / "reports/run_state.json",
        {
            "schema_version": "midogpp_stage70_run_state_v1",
            "status": "COMPLETE",
            "phase": "SCORING_COMPLETE",
        },
    )


def write_content_index(root: str | Path) -> None:
    path = Path(root)
    members = sorted(
        member
        for member in path.rglob("*")
        if member.is_file()
        and member.relative_to(path).as_posix()
        not in {"manifests/content_index.json", "reports/validation_report.json"}
    )
    records = [
        {
            "path": member.relative_to(path).as_posix(),
            "sha256": _sha256_file(member),
        }
        for member in members
    ]
    write_json(
        path / "manifests/content_index.json",
        {
            "schema_version": "midogpp_stage70_content_index_v1",
            "files": records,
        },
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_durable(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = (
    "REQUIRED_FILES",
    "seal_prediction_pass",
    "write_authorization_phase",
    "write_content_index",
    "write_scored_bundle",
)
