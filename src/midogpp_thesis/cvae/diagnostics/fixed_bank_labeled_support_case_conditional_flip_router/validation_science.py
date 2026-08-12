"""Reconstruct scientific surfaces from persisted rows and seals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from ...runtime.fixed_bank_a1_action_predictions import load_global_prediction_seal
from .actions import build_action_library
from .artifact_io import object_payload, read_rows
from .constants import CENTERS
from .partitions import CaseIdentityRow, build_three_role_partition
from .probability_surfaces import aggregate_exact_nine, build_prelabel_surface
from .products import SeedProbabilityRow
from .validation_science_replay import replay_label_aware_surfaces


def validate_scientific_surfaces(
    root: Path, *, config: object, frame: object
) -> Mapping[str, object]:
    protocol = read_json(root / "manifests/protocol_manifest.json")
    action_manifest = read_json(root / "manifests/action_library.json")
    partition_manifest = read_json(root / "manifests/three_role_partition.json")
    source_lock = read_json(root / "manifests/frozen_source_stream_lock.json")
    actions = _validate_actions(root, action_manifest)
    partition = _validate_partition(root, partition_manifest)
    prediction = load_global_prediction_seal(
        root,
        expected_config_hash=str(getattr(config, "contract_hash")),
        expected_partition_hash=partition.partition_hash,
        expected_source_lock_hash=str(source_lock["source_stream_lock_hash"]),
        expected_action_library_hash=str(action_manifest["action_library_hash"]),
        expected_target_cache_binding_hash=str(protocol["test_cache_binding_hash"]),
    )
    probability, prelabel = _validate_prelabel(root, prediction)
    label_aware = replay_label_aware_surfaces(
        root,
        config=config,
        frame=frame,
        partition=partition,
        prediction=prediction,
        probability_surface=probability,
        prelabel=prelabel,
    )
    return {
        "action_count": actions,
        "partition_fold_count": len(partition.folds),
        "prediction_cell_count": len(prediction.store.cells),
        "seed_probability_row_count": len(probability.rows) * 9,
        "aggregated_probability_row_count": len(probability.rows),
        "case_action_feature_count": len(prelabel.features),
        "probability_surface_hash": probability.surface_hash,
        "feature_surface_hash": prelabel.feature_surface_hash,
        **dict(label_aware),
        "scientific_reconstruction": "PASS",
    }


def _validate_actions(root: Path, manifest: Mapping[str, object]) -> int:
    expected = tuple(build_action_library())
    rows = read_rows(root / "tables/action_library.csv")
    observed = tuple(_parse_row(row) for row in rows)
    expected_payloads = tuple(object_payload(action) for action in expected)
    if observed != expected_payloads:
        raise ProtocolError("Flip-router action table is not reconstructive.")
    by_target = {
        target: [object_payload(action) for action in expected if action.target_center == target]
        for target in CENTERS
    }
    if (
        manifest.get("actions") != list(expected_payloads)
        or manifest.get("action_count") != len(expected)
        or manifest.get("physical_actions_per_target") != 10
        or manifest.get("action_library_hash") != stable_hash(by_target)
    ):
        raise ProtocolError("Flip-router action manifest drifted.")
    return len(expected)


def _validate_partition(root: Path, manifest: Mapping[str, object]):
    raw_identities = manifest.get("identities")
    if not isinstance(raw_identities, list):
        raise ProtocolError("Flip-router partition identities are absent.")
    identities = tuple(
        CaseIdentityRow(str(row["target_center"]), str(row["case_id"]), str(row["sample_id"]))
        for row in raw_identities
        if isinstance(row, Mapping)
    )
    rebuilt = build_three_role_partition(identities)
    if rebuilt.to_payload() != dict(manifest):
        raise ProtocolError("Flip-router three-role partition is not reconstructive.")
    expected_rows = []
    for fold in rebuilt.folds:
        for role, cases in (
            ("selection", fold.selection_case_ids),
            ("calibration", fold.calibration_case_ids),
            ("evaluation", fold.evaluation_case_ids),
        ):
            for case_id in cases:
                expected_rows.append({
                    "target_center": fold.target_center,
                    "fold_ordinal": fold.fold_ordinal,
                    "fold_id": fold.fold_id,
                    "case_id": case_id,
                    "role": role,
                    "fold_hash": fold.fold_hash,
                    "partition_hash": rebuilt.partition_hash,
                })
    if tuple(_parse_row(row) for row in read_rows(root / "tables/three_role_partitions.csv")) != tuple(expected_rows):
        raise ProtocolError("Flip-router partition table drifted.")
    return rebuilt


def _validate_prelabel(root: Path, prediction: object):
    raw_seed = read_rows(root / "tables/seed_probability_rows.csv")
    seed_rows = tuple(
        SeedProbabilityRow(
            str(row["target_center"]), str(row["case_id"]), str(row["sample_id"]), str(row["action_id"]),
            int(row["seed_pair_ordinal"]), float(row["probability"]), str(row["probability_store_hash"]),
        )
        for row in raw_seed
    )
    probability = aggregate_exact_nine(seed_rows)
    aggregate_rows = tuple(_parse_row(row) for row in read_rows(root / "tables/aggregated_probability_rows.csv"))
    if aggregate_rows != tuple(object_payload(row) for row in probability.rows):
        raise ProtocolError("Flip-router exact-nine aggregate table drifted.")
    probability_seal = read_json(root / "manifests/sealed_probability_surface.json")
    if (
        probability_seal.get("global_prediction_seal_hash") != prediction.seal_hash
        or probability_seal.get("probability_store_hash") != prediction.store.store_hash
        or probability_seal.get("surface_hash") != probability.surface_hash
        or probability_seal.get("row_count") != len(probability.rows)
        or probability_seal.get("seed_row_count") != len(seed_rows)
        or probability_seal.get("labels_used") is not False
    ):
        raise ProtocolError("Flip-router probability seal drifted.")
    prelabel = build_prelabel_surface(probability, prediction_seal_hash=prediction.seal_hash)
    feature_rows = tuple(_parse_row(row) for row in read_rows(root / "tables/case_action_features.csv"))
    if feature_rows != tuple(object_payload(row) for row in prelabel.features):
        raise ProtocolError("Flip-router case-action features are not reconstructive.")
    feature_seal = read_json(root / "manifests/prelabel_feature_seal.json")
    if (
        feature_seal.get("prediction_seal_hash") != prediction.seal_hash
        or feature_seal.get("probability_surface_hash") != probability.surface_hash
        or feature_seal.get("feature_surface_hash") != prelabel.feature_surface_hash
        or feature_seal.get("feature_count") != len(prelabel.features)
        or feature_seal.get("sealed_before_label_capabilities") is not True
    ):
        raise ProtocolError("Flip-router prelabel feature seal drifted.")
    return probability, prelabel


def _parse_row(row: Mapping[str, str]) -> dict[str, object]:
    output: dict[str, object] = {}
    integer_fields = {
        "fold_ordinal", "seed_pair_ordinal", "seed_pair_count", "feature_count",
        "n_positive", "n_negative", "tp_delta", "tn_delta", "delta_tp", "delta_tn",
        "training_row_count", "row_count", "flip_0to1_count", "flip_1to0_count",
    }
    float_fields = {
        "probability", "probability_mean", "probability_sd", "predicted_gain",
        "gain_standard_error", "lower_confidence_bound", "alpha", "variance_floor",
        "exact_gain", "runner_up_gain", "gamma_0to1", "gamma_1to0",
    }
    bool_fields = {
        "physical_fit_required", "target_expert_excluded", "seed_repetitions_selectable",
        "labels_used", "evaluation_labels_used", "held_evaluation_labels_in_plan",
        "plan_hash_invariant_to_held_evaluation_label_values",
        "fallback_to_b", "valid", "zero_intercept", "target_source_action_intercepts",
    }
    for key, value in row.items():
        if key in integer_fields:
            output[key] = int(value)
        elif key in float_fields:
            output[key] = float(value)
        elif key in bool_fields:
            output[key] = _bool(value)
        elif value == "" and key in {"selected_source", "geometry_id"}:
            output[key] = None
        elif value.startswith("[") or value.startswith("{") or value == "null":
            output[key] = json.loads(value)
        else:
            output[key] = value
    return output


def _bool(value: object) -> bool:
    if value is True or value == "True" or value == "true": return True
    if value is False or value == "False" or value == "false": return False
    raise ProtocolError(f"Flip-router boolean field is malformed: {value!r}.")


__all__ = ("validate_scientific_surfaces",)
