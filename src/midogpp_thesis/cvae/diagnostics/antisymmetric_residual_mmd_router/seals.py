"""Global pre-label capability seal for the 468 cross-fit prediction cells."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from ..mmd_kmm_router.contracts import ValidationRowIdentity
from .artifact_io import atomic_write_json, read_json, sha256_file
from .contracts import (
    ARM_ROLES,
    CENTERS,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_PREDICTION_CELL_COUNT,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
    row_identity_hash,
)
from .partitions import CrossfitSurface
from .prediction import (
    CROSSFIT_PREDICTION_ARRAY_MEMBER,
    CROSSFIT_PREDICTION_INDEX_MEMBER,
    CrossfitPredictionStore,
)

if TYPE_CHECKING:  # pragma: no cover
    from .config import AntisymmetricResidualMMDDiagnosticConfig


GLOBAL_CROSSFIT_PREDICTION_SEAL_MEMBER = (
    "manifests/global_case_prediction_seal.json"
)
GLOBAL_CROSSFIT_PREDICTION_SEAL_STATUS = (
    "COMPLETE_468_CROSSFIT_PREDICTIONS_BEFORE_ANY_EVALUATION_LABEL_ACCESS"
)


def _global_crossfit_prediction_seal_payload(
    config: "AntisymmetricResidualMMDDiagnosticConfig",
    crossfit: CrossfitSurface,
    plans: object,
    predictions: CrossfitPredictionStore,
    *,
    root: Path,
) -> Mapping[str, object]:
    plan_map, plan_lock_hash = _plan_surface(plans)
    expected_keys = tuple(
        (
            fold.fold_ordinal,
            fold.fold_id,
            fold.target_center,
            fold.heldout_case_id,
            training_seed,
            generation_seed,
            arm,
        )
        for fold in crossfit.folds
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
        for arm in ARM_ROLES
    )
    observed_keys = tuple(
        (
            _integer(row.get("fold_ordinal"), "fold ordinal"),
            str(row.get("fold_id")),
            str(row.get("target_center")),
            str(row.get("heldout_case_id")),
            _integer(row.get("training_seed"), "training seed"),
            _integer(row.get("generation_seed"), "generation seed"),
            str(row.get("arm_role")),
        )
        for row in predictions.index_rows
    )
    if (
        observed_keys != expected_keys
        or len(observed_keys) != EXPECTED_PREDICTION_CELL_COUNT
    ):
        raise ProtocolError("Antisymmetric global seal cell coverage drifted.")

    rows_by_fold = {
        fold.fold_id: [row.sample_id for row in fold.heldout_rows]
        for fold in crossfit.folds
    }
    row_hash_by_fold = {
        fold.fold_id: row_identity_hash(fold.heldout_rows)
        for fold in crossfit.folds
    }
    cells: list[dict[str, object]] = []
    for row in predictions.index_rows:
        ordinal = _integer(row["fold_ordinal"], "fold ordinal")
        fold = crossfit.folds[ordinal]
        plan = plan_map.get(fold.fold_id)
        if not isinstance(plan, Mapping):
            raise ProtocolError("Antisymmetric global seal lacks a fold plan.")
        if (
            row.get("config_contract_hash") != config.contract_hash
            or row.get("router_plan_lock_hash") != plan_lock_hash
            or row.get("fold_id") != fold.fold_id
            or row.get("fold_hash") != fold.fold_hash
            or row.get("heldout_case_id") != fold.heldout_case_id
            or _json_list(row.get("evaluation_row_ids_json"))
            != rows_by_fold[fold.fold_id]
            or row.get("evaluation_row_identity_hash")
            != row_hash_by_fold[fold.fold_id]
            or row.get("plan_hash") != plan.get("plan_hash")
            or row.get("router_support_row_identity_hash")
            != row_identity_hash(fold.router_support_rows)
            or not _truthy(row.get("classifier_converged"))
            or not _truthy(row.get("heldout_case_excluded_from_route"))
            or _truthy(row.get("labels_available_to_fit_or_predict"))
            or _truthy(row.get("support_labels_used"))
            or _truthy(row.get("seed_selection_performed"))
        ):
            raise ProtocolError(
                "Antisymmetric global seal cell escaped its heldout boundary."
            )
        cells.append(
            {
                "cell_ordinal": _integer(row["cell_ordinal"], "cell ordinal"),
                "fold_ordinal": ordinal,
                "fold_id": fold.fold_id,
                "fold_hash": fold.fold_hash,
                "target_center": fold.target_center,
                "heldout_case_id": fold.heldout_case_id,
                "arm_role": str(row["arm_role"]),
                "training_seed": _integer(row["training_seed"], "training seed"),
                "generation_seed": _integer(
                    row["generation_seed"], "generation seed"
                ),
                "evaluation_row_identity_hash": str(
                    row["evaluation_row_identity_hash"]
                ),
                "prediction_sha256": str(row["prediction_sha256"]),
                "probability_sha256": str(row["probability_sha256"]),
                "composition_hash": str(row["composition_hash"]),
                "classifier_config_hash": str(row["classifier_config_hash"]),
                "plan_hash": str(row["plan_hash"]),
                "control_fit_aliased": _truthy(row["control_fit_aliased"]),
            }
        )

    evaluation_ids = [
        row.sample_id for fold in crossfit.folds for row in fold.heldout_rows
    ]
    if len(evaluation_ids) != len(set(evaluation_ids)):
        raise ProtocolError("Antisymmetric global seal heldout rows duplicate.")
    unhashed: dict[str, object] = {
        "schema_version": (
            "midogpp_antisymmetric_residual_mmd_global_crossfit_prediction_seal_v1"
        ),
        "status": GLOBAL_CROSSFIT_PREDICTION_SEAL_STATUS,
        "config_contract_hash": config.contract_hash,
        "crossfit_surface_lock_hash": crossfit.lock_hash,
        "router_plan_lock_hash": plan_lock_hash,
        "validation_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "prediction_index_member": CROSSFIT_PREDICTION_INDEX_MEMBER,
        "prediction_index_sha256": sha256_file(
            root / CROSSFIT_PREDICTION_INDEX_MEMBER
        ),
        "prediction_arrays_member": CROSSFIT_PREDICTION_ARRAY_MEMBER,
        "prediction_arrays_sha256": sha256_file(
            root / CROSSFIT_PREDICTION_ARRAY_MEMBER
        ),
        "heldout_row_ids_by_fold": rows_by_fold,
        "heldout_row_identity_hash_by_fold": row_hash_by_fold,
        "fold_count": len(crossfit.folds),
        "target_count": len(CENTERS),
        "seed_cell_count_per_fold": len(TRAINING_SEEDS) * len(GENERATION_SEEDS),
        "arm_count_per_fold_seed": len(ARM_ROLES),
        "cell_count": len(cells),
        "unique_classifier_fit_count": predictions.unique_classifier_fit_count,
        "cells": cells,
        "all_evaluation_cases_held_out_exactly_once": True,
        "heldout_case_excluded_from_own_route": True,
        "all_predictions_persisted": True,
        "all_predictions_hashed": True,
        "support_labels_opened": False,
        "evaluation_labels_opened": False,
        "whole_label_column_loaded": False,
    }
    return {**unhashed, "seal_hash": stable_hash(unhashed)}


def build_global_crossfit_prediction_seal(
    config: "AntisymmetricResidualMMDDiagnosticConfig",
    crossfit: CrossfitSurface,
    plans: object,
    predictions: CrossfitPredictionStore,
    *,
    root: Path,
) -> Mapping[str, object]:
    payload = _global_crossfit_prediction_seal_payload(
        config, crossfit, plans, predictions, root=root
    )
    atomic_write_json(root / GLOBAL_CROSSFIT_PREDICTION_SEAL_MEMBER, payload)
    return payload


def validate_global_crossfit_prediction_seal(
    config: "AntisymmetricResidualMMDDiagnosticConfig",
    crossfit: CrossfitSurface,
    plans: object,
    predictions: CrossfitPredictionStore,
    *,
    root: Path,
) -> Mapping[str, object]:
    observed = read_json(root / GLOBAL_CROSSFIT_PREDICTION_SEAL_MEMBER)
    expected = _global_crossfit_prediction_seal_payload(
        config, crossfit, plans, predictions, root=root
    )
    if observed != expected:
        raise ProtocolError(
            "Antisymmetric global seal is not independently reconstructible."
        )
    return observed


def open_crossfit_evaluation_labels(
    config: "AntisymmetricResidualMMDDiagnosticConfig",
    crossfit: CrossfitSurface,
    *,
    root: Path,
) -> tuple[dict[str, int], Mapping[str, object]]:
    """Open only sealed heldout rows, after validating the global capability."""

    seal = read_json(root / GLOBAL_CROSSFIT_PREDICTION_SEAL_MEMBER)
    unhashed = {key: value for key, value in seal.items() if key != "seal_hash"}
    expected_ids = {
        fold.fold_id: [row.sample_id for row in fold.heldout_rows]
        for fold in crossfit.folds
    }
    expected_hashes = {
        fold.fold_id: row_identity_hash(fold.heldout_rows)
        for fold in crossfit.folds
    }
    if (
        seal.get("seal_hash") != stable_hash(unhashed)
        or seal.get("status") != GLOBAL_CROSSFIT_PREDICTION_SEAL_STATUS
        or seal.get("config_contract_hash") != config.contract_hash
        or seal.get("crossfit_surface_lock_hash") != crossfit.lock_hash
        or seal.get("validation_manifest_sha256") != EXPECTED_MANIFEST_SHA256
        or seal.get("cell_count") != EXPECTED_PREDICTION_CELL_COUNT
        or seal.get("prediction_index_sha256")
        != sha256_file(root / CROSSFIT_PREDICTION_INDEX_MEMBER)
        or seal.get("prediction_arrays_sha256")
        != sha256_file(root / CROSSFIT_PREDICTION_ARRAY_MEMBER)
        or seal.get("heldout_row_ids_by_fold") != expected_ids
        or seal.get("heldout_row_identity_hash_by_fold") != expected_hashes
        or seal.get("support_labels_opened") is not False
        or seal.get("evaluation_labels_opened") is not False
    ):
        raise ProtocolError(
            "Antisymmetric evaluation-label capability failed seal validation."
        )
    rows = tuple(row for fold in crossfit.folds for row in fold.heldout_rows)
    if (
        len({row.sample_id for row in rows}) != len(rows)
        or any(row.partition_role != "evaluation" for row in rows)
    ):
        raise ProtocolError("Antisymmetric label request contains non-heldout rows.")
    labels = _stream_labels(
        config.validation_manifest_path,
        rows,
        expected_sha256=EXPECTED_MANIFEST_SHA256,
    )
    by_sample = {
        row.sample_id: label for row, label in zip(rows, labels, strict=True)
    }
    label_hash_by_target: dict[str, str] = {}
    for target in CENTERS:
        target_rows = tuple(
            row
            for fold in crossfit.folds_by_target[target]
            for row in fold.heldout_rows
        )
        target_labels = tuple(by_sample[row.sample_id] for row in target_rows)
        if set(target_labels) != {0, 1}:
            raise ProtocolError(
                f"Antisymmetric target {target} lacks both evaluation classes."
            )
        label_hash_by_target[target] = stable_hash(
            {
                "target_center": target,
                "row_identity_hash": row_identity_hash(target_rows),
                "labels": list(target_labels),
                "prediction_seal_hash": seal["seal_hash"],
                "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            }
        )
    report: dict[str, object] = {
        "schema_version": (
            "midogpp_antisymmetric_residual_mmd_label_access_report_v1"
        ),
        "status": "OPENED_AFTER_GLOBAL_468_CELL_CROSSFIT_PREDICTION_SEAL",
        "prediction_seal_hash": seal["seal_hash"],
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "opened_row_count": len(rows),
        "opened_case_count": len(crossfit.folds),
        "opened_target_count": len(CENTERS),
        "label_vector_hash_by_target": label_hash_by_target,
        "support_label_count": 0,
        "train_label_count": 0,
        "test_label_count": 0,
        "excluded_center_label_count": 0,
        "whole_label_column_loaded": False,
        "labels_available_to_router": False,
        "labels_used_for_policy_or_hyperparameter_selection": False,
        "labels_used_for_scoring_only": True,
        "cross_fitted_transductive_diagnostic": True,
    }
    report["label_access_report_hash"] = stable_hash(report)
    return by_sample, report


def _stream_labels(
    manifest_path: Path,
    rows: Sequence[ValidationRowIdentity],
    *,
    expected_sha256: str,
) -> tuple[int, ...]:
    if sha256_file(manifest_path) != expected_sha256:
        raise ProtocolError("Antisymmetric validation manifest hash drifted.")
    expected_by_index = {row.manifest_row_index: row for row in rows}
    if len(expected_by_index) != len(rows):
        raise ProtocolError("Antisymmetric manifest row requests duplicate.")
    labels: dict[int, int] = {}
    try:
        handle = manifest_path.open(newline="", encoding="utf-8")
    except OSError as exc:
        raise ProtocolError("Cannot open antisymmetric scoring manifest.") from exc
    with handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "case_id", "center", "split", "label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ProtocolError("Antisymmetric scoring manifest lacks fields.")
        for index, raw in enumerate(reader):
            expected = expected_by_index.get(index)
            if expected is None:
                # Critically, this branch runs before the only label read.
                continue
            observed = (
                str(raw.get("sample_id", "")),
                str(raw.get("case_id", "")),
                str(raw.get("center", "")),
                str(raw.get("split", "")),
            )
            wanted = (
                expected.sample_id,
                expected.case_id,
                expected.center,
                expected.split,
            )
            if observed != wanted:
                raise ProtocolError(
                    "Antisymmetric scoring-manifest identity drifted."
                )
            try:
                value = int(str(raw["label"]).strip())
            except (TypeError, ValueError) as exc:
                raise ProtocolError("Antisymmetric evaluation label is invalid.") from exc
            if value not in (0, 1):
                raise ProtocolError("Antisymmetric evaluation label is not binary.")
            labels[index] = value
    if set(labels) != set(expected_by_index):
        raise ProtocolError("Antisymmetric evaluation-label coverage drifted.")
    return tuple(labels[row.manifest_row_index] for row in rows)


def _plan_surface(plans: object) -> tuple[Mapping[str, Mapping[str, object]], str]:
    raw = getattr(plans, "plans_by_fold", None)
    lock_hash = str(getattr(plans, "lock_hash", ""))
    if not isinstance(raw, Mapping) or not _is_hash(lock_hash):
        raise ProtocolError("Antisymmetric router-plan surface is malformed.")
    normalized = {
        str(key): value for key, value in raw.items() if isinstance(value, Mapping)
    }
    if len(normalized) != len(raw):
        raise ProtocolError("Antisymmetric router-plan rows are malformed.")
    return normalized, lock_hash


def _json_list(value: object) -> list[str]:
    try:
        payload = json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ProtocolError("Antisymmetric sealed row IDs are malformed.") from exc
    if not isinstance(payload, list) or any(not isinstance(item, str) for item in payload):
        raise ProtocolError("Antisymmetric sealed row IDs are invalid.")
    return payload


def _integer(value: object, role: str) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"Antisymmetric {role} is invalid.") from exc


def _truthy(value: object) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _is_hash(value: object) -> bool:
    text = str(value)
    return len(text) in {16, 64} and all(character in "0123456789abcdef" for character in text)


# Short aliases keep the runner surface parallel to earlier Stage-90 routers.
build_global_prediction_seal = build_global_crossfit_prediction_seal
validate_global_prediction_seal = validate_global_crossfit_prediction_seal
open_evaluation_labels = open_crossfit_evaluation_labels


__all__ = (
    "GLOBAL_CROSSFIT_PREDICTION_SEAL_MEMBER",
    "GLOBAL_CROSSFIT_PREDICTION_SEAL_STATUS",
    "build_global_crossfit_prediction_seal",
    "build_global_prediction_seal",
    "open_crossfit_evaluation_labels",
    "open_evaluation_labels",
    "validate_global_crossfit_prediction_seal",
    "validate_global_prediction_seal",
)
