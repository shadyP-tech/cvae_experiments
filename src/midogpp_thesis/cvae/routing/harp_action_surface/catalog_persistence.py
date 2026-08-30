"""Atomic file and table persistence for HARP Stage-60 surface catalogs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
import os
from pathlib import Path

import numpy as np
import yaml

from ...protocol import ProtocolError
from ...runtime.artifact_io import atomic_json, atomic_npy
from ...runtime.harp_probability_menu import HarpPredictionMenuSeal
from ..harp_protocol.hashing import canonical_hash
from ..harp_stage60.config import HarpInputReadiness, HarpStage60Config
from ..harp_stage60.constants import ACTION_SURFACE
from .artifact_contract import (
    CONFIG_MEMBER,
    DIRECTIONAL_FEATURES_MEMBER,
    PROBABILITY_ARRAY_MEMBER,
    PROBABILITY_INDEX_MEMBER,
    PROTOCOL_MEMBER,
    PROVENANCE_MEMBER,
)
from .contracts import HarpDirectionalResponseSurface


def persist_prelabel_catalog_members(
    config: HarpStage60Config,
    readiness: HarpInputReadiness,
    menu: HarpPredictionMenuSeal,
    *,
    feature_payload: Mapping[str, object],
) -> Path:
    """Publish every label-free catalog member before a label edge exists."""

    menu.assert_valid()
    _write_resolved_config(config.artifact_root / CONFIG_MEMBER, config)
    provenance_unhashed: dict[str, object] = {
        "schema_version": "midogpp_harp_surface_input_provenance_v1",
        "surface": config.contract.surface,
        "experiment_id": config.experiment_id,
        "input_artifact_ids": list(config.input_artifact_ids),
        "config_contract_hash": config.contract_hash,
        "input_binding_sha256": readiness.input_binding_sha256,
        "reservation_sha256": readiness.reservation_sha256,
        "cache_binding_sha256": readiness.cache_binding_sha256,
        "manifest_sha256": readiness.manifest_sha256,
        "attestation_sha256": readiness.attestation_sha256,
        "menu_seal_hash": menu.seal_hash,
        "target_support_labels_used": False,
        "target_evaluation_labels_used": False,
        "stage50_artifacts_used": False,
        "stage90_artifacts_used": False,
        "consumed_test_rows_used": False,
    }
    atomic_json(
        config.artifact_root / PROVENANCE_MEMBER,
        {**provenance_unhashed, "provenance_hash": canonical_hash(provenance_unhashed)},
    )
    protocol_unhashed: dict[str, object] = {
        "schema_version": "midogpp_harp_surface_protocol_manifest_v1",
        "surface": config.contract.surface,
        "dataset_family": "MIDOG++",
        "config_contract_hash": config.contract_hash,
        "action_menu_hash": menu.action_menu_hash,
        "prediction_store_hash": menu.prediction_store_hash,
        "menu_seal_hash": menu.seal_hash,
        "strict_outer_target_query_candidate_exclusion": True,
        "outer_target_excluded_before_transform": True,
        "probability_endpoint": "exact_nine_seed_ensemble_per_sample",
        "seed_cells_may_feed_model": False,
        "case_equal_weighting_required": True,
        "source_labels_opened_after_global_prediction_seal": (
            config.contract == ACTION_SURFACE
        ),
        "target_support_labels_used": False,
        "target_evaluation_labels_used": False,
    }
    atomic_json(
        config.artifact_root / PROTOCOL_MEMBER,
        {**protocol_unhashed, "protocol_manifest_hash": canonical_hash(protocol_unhashed)},
    )
    probabilities = np.concatenate(
        [np.asarray(cell.probabilities, dtype=np.float32) for cell in menu.cells]
    ).astype(np.float32, copy=False)
    if probabilities.dtype != np.float32 or not np.isfinite(probabilities).all():
        raise ProtocolError("HARP probability catalog array is not finite float32.")
    array_path = config.artifact_root / PROBABILITY_ARRAY_MEMBER
    atomic_npy(array_path, probabilities)
    _write_probability_index(config.artifact_root / PROBABILITY_INDEX_MEMBER, menu)
    write_directional_feature_table(
        config.artifact_root / DIRECTIONAL_FEATURES_MEMBER, feature_payload
    )
    return array_path


def _write_resolved_config(path: Path, config: HarpStage60Config) -> None:
    payload = {
        "schema_version": "midogpp_harp_stage60_resolved_config_v1",
        "surface": config.contract.surface,
        "experiment": {
            "id": config.experiment_id,
            "artifact_root": str(config.artifact_root.resolve()),
            "output_artifact_id": config.output_artifact_id,
        },
        "inputs": {
            "artifact_ids": list(config.input_artifact_ids),
            "paths": {key: str(value.resolve()) for key, value in config.input_paths.items()},
        },
        "protocol": dict(config.protocol),
        "model": dict(config.model),
        "runtime": dict(config.runtime),
        "claim_boundary": dict(config.claim_boundary),
        "config_contract_hash": config.contract_hash,
    }
    atomic_text(path, yaml.safe_dump(payload, sort_keys=True))


def _write_probability_index(path: Path, menu: HarpPredictionMenuSeal) -> None:
    fieldnames = (
        "cell_ordinal",
        "surface_kind",
        "outer_target_id",
        "query_center_id",
        "selected_source_id",
        "action_id",
        "action_hash",
        "training_seed",
        "generation_seed",
        "array_offset",
        "row_count",
        "row_identity_sha256",
        "case_identity_sha256",
        "probability_bytes_sha256",
        "fit_provenance_hash",
        "cell_hash",
    )
    rows: list[dict[str, object]] = []
    offset = 0
    for ordinal, cell in enumerate(menu.cells):
        rows.append(
            {
                "cell_ordinal": ordinal,
                "surface_kind": cell.action.surface_kind,
                "outer_target_id": cell.action.outer_target_id,
                "query_center_id": cell.action.query_center_id,
                "selected_source_id": cell.action.selected_source_id or "",
                "action_id": cell.action.action_id,
                "action_hash": cell.action.action_hash,
                "training_seed": cell.training_seed,
                "generation_seed": cell.generation_seed,
                "array_offset": offset,
                "row_count": len(cell.probabilities),
                "row_identity_sha256": cell.row_identity_sha256,
                "case_identity_sha256": cell.case_identity_sha256,
                "probability_bytes_sha256": cell.probability_bytes_sha256,
                "fit_provenance_hash": cell.fit_provenance_hash,
                "cell_hash": cell.cell_hash,
            }
        )
        offset += len(cell.probabilities)
    atomic_csv(path, fieldnames, rows)


def write_directional_feature_table(
    path: Path, feature_payload: Mapping[str, object]
) -> None:
    raw_rows = feature_payload.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ProtocolError("HARP directional feature artifact is empty.")
    fieldnames = (
        "outer_target",
        "pseudo_query",
        "candidate_source",
        "inner_donor",
        "case_id",
        "sample_id",
        "action_lambda",
        "direction",
        "feature_names_json",
        "feature_values_json",
        "ensemble_receipt_hash",
        "case_weight_receipt_hash",
        "seed_count",
        "feature_hash",
    )
    rows = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise ProtocolError("HARP directional feature row is malformed.")
        rows.append(
            {
                "outer_target": raw.get("outer_target", ""),
                "pseudo_query": raw.get("pseudo_query", ""),
                "candidate_source": raw.get("candidate_source", ""),
                "inner_donor": raw.get("inner_donor") or "",
                "case_id": raw.get("case_id", ""),
                "sample_id": raw.get("sample_id", ""),
                "action_lambda": raw.get("action_lambda"),
                "direction": raw.get("direction", ""),
                "feature_names_json": json.dumps(
                    raw.get("feature_names", ()), separators=(",", ":")
                ),
                "feature_values_json": json.dumps(
                    raw.get("feature_values", ()), separators=(",", ":")
                ),
                "ensemble_receipt_hash": raw.get("ensemble_receipt_hash", ""),
                "case_weight_receipt_hash": raw.get("case_weight_receipt_hash", ""),
                "seed_count": raw.get("seed_count"),
                "feature_hash": raw.get("feature_hash", ""),
            }
        )
    atomic_csv(path, fieldnames, rows)


def write_directional_response_table(
    path: Path, surface: HarpDirectionalResponseSurface
) -> None:
    fieldnames = (
        "outer_target",
        "pseudo_query",
        "candidate_source",
        "inner_donor",
        "case_id",
        "sample_id",
        "action_lambda",
        "direction",
        "truth_class",
        "weighted_correctness_surrogate",
        "brier_delta",
        "log_loss_delta",
        "denominator_receipt_hash",
        "ensemble_receipt_hash",
        "case_weight_receipt_hash",
        "feature_hash",
        "outer_scoped_label_surface_hash",
        "response_hash",
    )
    rows = tuple(
        {
            "outer_target": row.outer_target,
            "pseudo_query": row.pseudo_query,
            "candidate_source": row.candidate_source,
            "inner_donor": row.inner_donor or "",
            "case_id": row.case_id,
            "sample_id": row.sample_id,
            "action_lambda": row.action_lambda,
            "direction": row.direction,
            "truth_class": row.truth_class,
            "weighted_correctness_surrogate": row.weighted_correctness_surrogate,
            "brier_delta": row.brier_delta,
            "log_loss_delta": row.log_loss_delta,
            "denominator_receipt_hash": row.denominator_receipt_hash,
            "ensemble_receipt_hash": row.ensemble_receipt_hash,
            "case_weight_receipt_hash": row.case_aggregation_receipt_hash,
            "feature_hash": row.feature_hash,
            "outer_scoped_label_surface_hash": row.label_surface_hash,
            "response_hash": row.response_hash,
        }
        for row in surface.rows
    )
    atomic_csv(path, fieldnames, rows)


def atomic_csv(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=tuple(fieldnames),
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except (OSError, csv.Error, ValueError) as exc:
        temporary.unlink(missing_ok=True)
        raise ProtocolError(f"Cannot publish HARP table: {path}.") from exc


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise ProtocolError(f"Cannot publish HARP text member: {path}.") from exc


__all__ = (
    "atomic_csv",
    "atomic_text",
    "persist_prelabel_catalog_members",
    "write_directional_feature_table",
    "write_directional_response_table",
)
