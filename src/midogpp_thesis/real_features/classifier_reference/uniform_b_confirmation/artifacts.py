"""Artifact utilities for prospective uniform-B confirmation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from ..downstream import balanced_accuracy
from ..protocol import ProtocolError
from ..uniform_b_replay.artifacts import (
    read_csv,
    read_json,
    sha256,
    validate_content_index,
    write_content_index,
)
from .cache import validate_uniform_b_test_cache
from .config import (
    CHECKPOINT_FILE_SHA256,
    MODEL_CONFIG_SHA256,
    MODEL_REF,
    MODEL_REVISION,
    PREPROCESSING_CONFIG_HASH,
    STATE_DICT_SHA256,
    UniformBConfirmationConfig,
    UniformBTestCacheConfig,
)


REQUIRED_FILES = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "manifests/frozen_protocol_snapshot.json",
    "manifests/protocol_manifest.json",
    "manifests/uniform_representation_lock.json",
    "manifests/source_lock_index.json",
    "manifests/content_index.json",
    "tables/source_lock_audit.csv",
    "tables/split_isolation_audit.csv",
    "tables/prospective_test_results.csv",
    "tables/prospective_test_predictions.csv",
    "tables/paired_center_comparison.csv",
    "tables/outer_fit_audit.csv",
    "reports/conditional_bootstrap.json",
    "reports/confirmation_summary.json",
    "reports/confirmation_report.md",
    "reports/leakage_provenance_report.json",
    "reports/runtime_summary.json",
    "reports/validation_report.json",
)


def confirmation_input_hashes(config: UniformBConfirmationConfig) -> dict[str, str]:
    paths = {
        "dataset_manifest": config.manifest_path,
        "canonical_train_cache": config.canonical_train_cache_path,
        "canonical_test_cache": config.canonical_test_cache_path,
        "source_b_alignment": config.source_train_b_cache_root / "manifests/row_alignment.json",
        "source_b_report": config.source_train_b_cache_root / "reports/cache_builder_report.json",
        "test_b_content_index": config.test_b_cache_root / "manifests/content_index.json",
        "test_b_alignment": config.test_b_cache_root / "manifests/row_alignment.json",
        "test_b_report": config.test_b_cache_root / "reports/cache_builder_report.json",
        "source_v3_protocol": config.source_v3_root / "manifests/protocol_manifest.json",
        "source_v3_lock_index": config.source_v3_root / "manifests/decision_lock_index.json",
        "retrospective_protocol": config.retrospective_root / "manifests/protocol_manifest.json",
        "retrospective_summary": config.retrospective_root / "reports/diagnostic_summary.json",
    }
    for center in config.heldout_centers:
        paths[f"source_b_center_{center}"] = (
            config.source_train_b_cache_root / "embeddings/by_center" / f"center_{center}.pt"
        )
        paths[f"test_b_center_{center}"] = (
            config.test_b_cache_root / "embeddings/by_center" / f"center_{center}.pt"
        )
        paths[f"source_lock_{center}"] = (
            config.source_v3_root / "manifests/decision_locks" / f"center_{center}.json"
        )
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ProtocolError(f"Uniform-B prospective inputs are incomplete: {missing}.")
    return {key: sha256(path) for key, path in sorted(paths.items())}


def validate_confirmation_inputs(config: UniformBConfirmationConfig) -> None:
    source_protocol = read_json(config.source_v3_root / "manifests/protocol_manifest.json")
    retrospective_protocol = read_json(
        config.retrospective_root / "manifests/protocol_manifest.json"
    )
    retrospective_summary = read_json(
        config.retrospective_root / "reports/diagnostic_summary.json"
    )
    if (
        source_protocol.get("status") != "PASS"
        or retrospective_protocol.get("status") != "PASS"
        or retrospective_protocol.get("study_design_informed_by_prior_target_scores") is not True
        or retrospective_summary.get("study_design_status") != "POSTHOC_DISCOVERY"
        or retrospective_summary.get("independent_confirmation") is not False
    ):
        raise ProtocolError("Uniform-B prospective source evidence is invalid.")
    cache_config = cache_config_from_confirmation(config)
    validate_uniform_b_test_cache(
        config.test_b_cache_root, expected_config=cache_config
    )
    validate_content_index(config.source_v3_root)
    validate_content_index(config.retrospective_root)


def cache_config_from_confirmation(
    config: UniformBConfirmationConfig,
) -> UniformBTestCacheConfig:
    return UniformBTestCacheConfig(
        name="uniform_b_v3_prospective_test_cache_v1",
        repo_root=Path("."),
        manifest_path=config.manifest_path,
        canonical_train_cache_path=config.canonical_train_cache_path,
        canonical_test_cache_path=config.canonical_test_cache_path,
        source_train_b_cache_root=config.source_train_b_cache_root,
        cache_root=config.test_b_cache_root,
        eligible_centers=config.heldout_centers,
        device="cuda",
        batch_size=32,
        experiment_seed=42,
        model_ref=MODEL_REF,
        model_revision=MODEL_REVISION,
        expected_model_config_sha256=MODEL_CONFIG_SHA256,
        expected_checkpoint_file_sha256=CHECKPOINT_FILE_SHA256,
        expected_state_dict_sha256=STATE_DICT_SHA256,
        expected_preprocessing_config_hash=PREPROCESSING_CONFIG_HASH,
        expected_runtime={"timm": "1.0.27", "torch": "2.6.0+cu124", "pillow": "12.0.0"},
    )


def prospective_paired_case_bootstrap(
    prediction_rows: Sequence[Mapping[str, object]],
    *,
    seed: int,
    valid_replicates: int,
    max_attempts: int,
) -> dict[str, object]:
    import numpy as np

    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in prediction_rows:
        grouped[(str(row["heldout_center"]), str(row["role"]))].append(row)
    centers = sorted({center for center, _role in grouped}, key=int)
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    attempts = rejected = 0
    while len(deltas) < valid_replicates and attempts < max_attempts:
        attempts += 1
        center_deltas = []
        valid = True
        for center in centers:
            a = _by_case(grouped[(center, "canonical_a")])
            b = _by_case(grouped[(center, "uniform_b")])
            if set(a) != set(b):
                raise ProtocolError("Uniform-B prospective paired case pools differ.")
            cases = sorted(a)
            sampled = rng.choice(cases, size=len(cases), replace=True)
            ar = [row for case in sampled for row in a[str(case)]]
            br = [row for case in sampled for row in b[str(case)]]
            labels = [int(row["label"]) for row in ar]
            if set(labels) != {0, 1}:
                valid = False
                break
            center_deltas.append(
                balanced_accuracy(
                    [int(row["label"]) for row in br],
                    [int(row["prediction"]) for row in br],
                )
                - balanced_accuracy(labels, [int(row["prediction"]) for row in ar])
            )
        if not valid:
            rejected += 1
            continue
        deltas.append(sum(center_deltas) / len(center_deltas))
    if len(deltas) != valid_replicates:
        raise ProtocolError("Uniform-B prospective bootstrap is incomplete.")
    lower, upper = np.quantile(np.asarray(deltas), [0.025, 0.975]).tolist()
    return {
        "schema_version": "midogpp_uniform_b_prospective_bootstrap_v1",
        "status": "PASS",
        "seed": seed,
        "valid_replicates": len(deltas),
        "attempted_replicates": attempts,
        "rejected_class_missing_replicates": rejected,
        "mean_delta": float(np.mean(deltas)),
        "percentile_2_5": float(lower),
        "percentile_97_5": float(upper),
        "interval_role": "prospective_within_center_paired_case_interval",
        "conditions_on_fixed_training_fits_and_classifier_locks": True,
        "covers_new_case_uncertainty_within_centers": True,
        "covers_new_center_uncertainty": False,
        "p_value_computed": False,
    }


def _by_case(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, list[Mapping[str, object]]]:
    out: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        out[str(row["case_id"])].append(row)
    return dict(out)


__all__ = [
    "REQUIRED_FILES",
    "confirmation_input_hashes",
    "prospective_paired_case_bootstrap",
    "read_csv",
    "read_json",
    "validate_confirmation_inputs",
    "validate_content_index",
    "write_content_index",
]
