"""Durable all-action seal created before any validation label access."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .artifact_io import atomic_write_json, read_json, sha256_file
from .config import EXPECTED_MANIFEST_SHA256
from .contracts import (
    CENTERS,
    EXPECTED_ACTION_COUNT_PER_TARGET,
    EXPECTED_CASE_OOF_FOLD_COUNT,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
    expected_action_ids,
)
from .prediction_store import (
    EXPECTED_PREDICTION_CELL_COUNT,
    PREDICTION_ARRAY_MEMBER,
    PREDICTION_INDEX_MEMBER,
)


GLOBAL_PREDICTION_SEAL_MEMBER = (
    "manifests/global_all_action_prediction_seal.json"
)
GLOBAL_PREDICTION_SEAL_STATUS = (
    "COMPLETE_ALL_B_U_G_S_P_HXE_CASE_OOF_PREDICTIONS_BEFORE_ANY_LABEL_ACCESS"
)


def build_global_prediction_seal(
    config: object,
    crossfit: object,
    plan: object,
    predictions: object,
    *,
    source_cache_lock_hash: str,
    root: Path,
) -> Mapping[str, object]:
    payload = _seal_payload(
        config,
        crossfit,
        plan,
        predictions,
        source_cache_lock_hash=source_cache_lock_hash,
        root=root,
    )
    atomic_write_json(root / GLOBAL_PREDICTION_SEAL_MEMBER, payload)
    return payload


def validate_global_prediction_seal(
    config: object,
    crossfit: object,
    plan: object,
    predictions: object,
    *,
    source_cache_lock_hash: str,
    root: Path,
) -> Mapping[str, object]:
    observed = read_json(root / GLOBAL_PREDICTION_SEAL_MEMBER)
    expected = _seal_payload(
        config,
        crossfit,
        plan,
        predictions,
        source_cache_lock_hash=source_cache_lock_hash,
        root=root,
    )
    if observed != expected:
        raise ProtocolError("Case-OOF global prediction seal drifted.")
    return observed


def _seal_payload(
    config: object,
    crossfit: object,
    plan: object,
    predictions: object,
    *,
    source_cache_lock_hash: str,
    root: Path,
) -> Mapping[str, object]:
    rows = tuple(getattr(predictions, "index_rows", ()))
    if (
        len(rows) != EXPECTED_PREDICTION_CELL_COUNT
        or not str(getattr(config, "contract_hash", ""))
        or not str(getattr(crossfit, "lock_hash", ""))
        or not str(getattr(plan, "lock_hash", ""))
        or not source_cache_lock_hash
    ):
        raise ProtocolError("Case-OOF seal inputs are incomplete.")
    fold_by_id = {fold.fold_id: fold for fold in crossfit.folds}
    cells: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for ordinal, row in enumerate(rows):
        fold_id = str(row["fold_id"])
        fold = fold_by_id.get(fold_id)
        target = str(row["target_center"])
        action_id = str(row["action_id"])
        key = (
            fold_id,
            action_id,
            int(row["training_seed"]),
            int(row["generation_seed"]),
        )
        if (
            fold is None
            or fold.target_center != target
            or action_id not in expected_action_ids(target)
            or key in seen
            or int(row["cell_ordinal"]) != ordinal
            or row["config_contract_hash"] != getattr(config, "contract_hash")
            or row["source_cache_lock_hash"] != source_cache_lock_hash
            or row["crossfit_fold_lock_hash"] != crossfit.lock_hash
            or row["router_plan_lock_hash"] != plan.lock_hash
            or str(row["fold_hash"]) != fold.fold_hash
            or str(row["action_hash"])
            != getattr(plan, "action")(target, action_id).action_hash
            or _truthy(row["labels_available_to_fit_or_predict"])
            or _truthy(row["support_labels_used"])
            or _truthy(row["evaluation_embeddings_used_for_route"])
            or _truthy(row["other_evaluation_embeddings_used_for_route"])
            or not _truthy(row["heldout_case_excluded_from_route"])
            or not _truthy(row["target_expert_excluded"])
            or _truthy(row["seed_selection_performed"])
            or _truthy(row["policy_selection_performed"])
            or _truthy(row["fallback_performed"])
            or not _truthy(row["classifier_converged"])
        ):
            raise ProtocolError("Case-OOF prediction escaped its seal boundary.")
        seen.add(key)
        cells.append(
            {
                "cell_ordinal": ordinal,
                "fold_id": fold_id,
                "fold_ordinal": int(row["fold_ordinal"]),
                "target_center": target,
                "heldout_case_id": str(row["heldout_case_id"]),
                "action_id": action_id,
                "training_seed": int(row["training_seed"]),
                "generation_seed": int(row["generation_seed"]),
                "evaluation_row_identity_hash": str(
                    row["evaluation_row_identity_hash"]
                ),
                "prediction_sha256": str(row["prediction_sha256"]),
                "probability_sha256": str(row["probability_sha256"]),
                "composition_hash": str(row["composition_hash"]),
                "action_hash": str(row["action_hash"]),
            }
        )
    expected = {
        (fold.fold_id, action_id, training_seed, generation_seed)
        for fold in crossfit.folds
        for action_id in expected_action_ids(fold.target_center)
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
    }
    if seen != expected or len(seen) != EXPECTED_PREDICTION_CELL_COUNT:
        raise ProtocolError("Case-OOF seal cell coverage drifted.")
    unhashed: dict[str, object] = {
        "schema_version": "midogpp_residual_topup_case_oof_global_prediction_seal_v1",
        "status": GLOBAL_PREDICTION_SEAL_STATUS,
        "config_contract_hash": str(getattr(config, "contract_hash")),
        "source_cache_lock_hash": source_cache_lock_hash,
        "crossfit_fold_lock_hash": str(getattr(crossfit, "lock_hash")),
        "router_plan_lock_hash": str(getattr(plan, "lock_hash")),
        "validation_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "prediction_array_member": PREDICTION_ARRAY_MEMBER,
        "prediction_array_sha256": sha256_file(root / PREDICTION_ARRAY_MEMBER),
        "prediction_index_member": PREDICTION_INDEX_MEMBER,
        "prediction_index_sha256": sha256_file(root / PREDICTION_INDEX_MEMBER),
        "fold_count": EXPECTED_CASE_OOF_FOLD_COUNT,
        "action_count_per_target": EXPECTED_ACTION_COUNT_PER_TARGET,
        "cell_count": len(cells),
        "cells": cells,
        "all_B_U_G_S_P_Hxe_actions_materialized": True,
        "all_predictions_hashed": True,
        "fixed_support_route_reused_across_target_folds": True,
        "other_evaluation_embeddings_used_for_route": False,
        "support_labels_opened": False,
        "evaluation_labels_opened": False,
        "whole_label_column_loaded": False,
        "selector_or_fallback_performed": False,
        "policy_update_performed": False,
    }
    return {**unhashed, "seal_hash": stable_hash(unhashed)}


def _truthy(value: object) -> bool:
    return value is True or str(value).strip().lower() == "true"


__all__ = (
    "GLOBAL_PREDICTION_SEAL_MEMBER",
    "GLOBAL_PREDICTION_SEAL_STATUS",
    "build_global_prediction_seal",
    "validate_global_prediction_seal",
)
