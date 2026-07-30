"""Run the separately reviewed Uniform-B canonical reference experiment."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping

from midogpp_thesis.common.hashing import stable_hash

from ..artifacts import write_json
from ..matched_reference import MatchedReferenceConfig, run_matched_reference
from ..protocol import ProtocolError
from ..uniform_b_confirmation import (
    load_uniform_b_confirmation_config,
    validate_uniform_b_confirmation_bundle,
)
from .cache import validate_uniform_b_canonical_train_cache
from .config import (
    CONFIRMATION_PROTOCOL_SHA256,
    CONFIRMATION_SUMMARY_SHA256,
    PROMOTION_REVIEW_ID,
    REPRESENTATION_ID,
    UniformBCanonicalReferenceConfig,
    load_uniform_b_canonical_cache_config,
)
from .workspace_binding import validate_production_workspace_binding


def run_uniform_b_canonical_reference(
    config: UniformBCanonicalReferenceConfig,
) -> Path:
    validate_production_workspace_binding(config)
    cache_config = load_uniform_b_canonical_cache_config(
        Path("datasets/midogpp/configs/uniform_b_canonical_train_cache_v1.yaml")
    )
    validate_uniform_b_canonical_train_cache(
        config.feature_cache_path.parent.parent,
        expected_config=cache_config,
    )
    confirmation = _validated_confirmation(config)
    root = config.artifact_root
    for relative in ("manifests", "reports", "tables", "provenance"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    write_json(
        root / "manifests/uniform_b_canonical_representation_lock.json",
        _representation_lock(config, confirmation),
    )
    write_json(
        root / "manifests/promotion_review_snapshot.json",
        _review_snapshot(config, confirmation),
    )
    write_json(
        root / "reports/test_consumption_ledger.json",
        _test_consumption_ledger(confirmation),
    )

    run_matched_reference(
        MatchedReferenceConfig(
            name=config.name,
            artifact_root=root,
            manifest_path=config.manifest_path,
            feature_cache_path=config.feature_cache_path,
            heldout_centers=config.heldout_centers,
            experiment_seed=config.experiment_seed,
            classifier_seed=config.classifier_seed,
            expected_feature_dim=config.expected_feature_dim,
            classifier_specs=config.classifier_specs,
            allow_excluded_center_omission=True,
        )
    )
    summary = _reference_summary(root, confirmation)
    write_json(root / "reports/promotion_decision.json", summary)
    (root / "reports/promotion_report.md").write_text(
        "\n".join(
            [
                "# MIDOG++ Uniform-B Canonical Real-Feature Reference",
                "",
                "Decision: `PROMOTED_AS_NEW_CANONICAL_REFERENCE`.",
                "",
                f"- Representation: `{REPRESENTATION_ID}`",
                "- Canonical A retained: `true`",
                "- Phase-B test split consumed for representation adoption: `true`",
                "- Classifier tuning: fresh source-inner LODO on B train rows",
                "- Automatic downstream migration: `false`",
                "",
                "Later experiments may adopt this reference only through an explicit",
                "config and artifact-input migration. Existing A-based runs remain immutable.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _write_content_index(root)
    from .validation import validate_uniform_b_canonical_reference_bundle

    pending = validate_uniform_b_canonical_reference_bundle(
        root, config=config, allow_pending=True
    )
    write_json(
        root / "reports/validation_report.json",
        {
            "schema_version": "midogpp_uniform_b_canonical_reference_validation_v1",
            "status": "PASS",
            "validator": "validate_uniform_b_canonical_reference_bundle",
            "checks": pending,
        },
    )
    _write_content_index(root)
    validate_uniform_b_canonical_reference_bundle(root, config=config)
    return root


def _validated_confirmation(
    config: UniformBCanonicalReferenceConfig,
) -> Mapping[str, object]:
    confirmation_config = load_uniform_b_confirmation_config(
        config.confirmation_root / "config.resolved.yaml"
    )
    validate_uniform_b_confirmation_bundle(
        config.confirmation_root, config=confirmation_config
    )
    summary_path = config.confirmation_root / "reports/confirmation_summary.json"
    protocol_path = config.confirmation_root / "manifests/protocol_manifest.json"
    if (
        _sha256_file(summary_path) != CONFIRMATION_SUMMARY_SHA256
        or _sha256_file(protocol_path) != CONFIRMATION_PROTOCOL_SHA256
    ):
        raise ProtocolError("Uniform-B promotion confirmation hashes drifted.")
    summary = _read_json(summary_path)
    bootstrap = _mapping(summary, "conditional_bootstrap")
    if (
        summary.get("decision") != "CONFIRMED_WITHIN_CENTER"
        or summary.get("confirmation_passed") is not True
        or summary.get("strict_wins") != 9
        or float(summary.get("paired_mean_delta", 0.0)) < 0.02
        or float(summary.get("worst_center_delta", -1.0)) < -0.01
        or float(bootstrap.get("percentile_2_5", -1.0)) <= 0.0
    ):
        raise ProtocolError("Uniform-B promotion requires a passing Phase-B confirmation.")
    return summary


def _representation_lock(
    config: UniformBCanonicalReferenceConfig,
    confirmation: Mapping[str, object],
) -> dict[str, object]:
    payload = {
        "schema_version": "midogpp_uniform_b_canonical_representation_lock_v1",
        "representation_id": REPRESENTATION_ID,
        "feature_dim": config.expected_feature_dim,
        "pooling": "fixed_center_rows6to9_cols6to9",
        "promotion_review_id": PROMOTION_REVIEW_ID,
        "confirmation_decision": confirmation["decision"],
        "confirmation_summary_sha256": CONFIRMATION_SUMMARY_SHA256,
        "canonical_a_retained": True,
        "automatic_downstream_migration": False,
    }
    payload["representation_lock_hash"] = stable_hash(payload)
    return payload


def _review_snapshot(
    config: UniformBCanonicalReferenceConfig,
    confirmation: Mapping[str, object],
) -> dict[str, object]:
    payload = {
        "schema_version": "midogpp_uniform_b_promotion_review_v1",
        **dict(config.review),
        "observed_confirmation_mean_delta": confirmation["paired_mean_delta"],
        "observed_confirmation_strict_wins": confirmation["strict_wins"],
        "observed_confirmation_worst_center_delta": confirmation["worst_center_delta"],
        "observed_confirmation_bootstrap_lower": _mapping(
            confirmation, "conditional_bootstrap"
        )["percentile_2_5"],
        "review_effect": "authorizes_new_stage10_reference_only",
    }
    payload["review_hash"] = stable_hash(payload)
    return payload


def _test_consumption_ledger(
    confirmation: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_uniform_b_test_consumption_ledger_v1",
        "status": "CONSUMED_FOR_REPRESENTATION_ADOPTION",
        "split": "test",
        "row_count": 9928,
        "observed_centers": 9,
        "consumed_decision": confirmation["decision"],
        "may_be_reused_as_fresh_representation_selection_evidence": False,
        "may_be_reused_for_descriptive_locked-model_scoring": True,
        "new_center_uncertainty_covered": False,
        "external_dataset_uncertainty_covered": False,
    }


def _reference_summary(
    root: Path,
    confirmation: Mapping[str, object],
) -> dict[str, object]:
    with (root / "tables/classifier_tuned_source_results.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    mean_bacc = sum(float(row["heldout_bacc"]) for row in rows) / len(rows)
    return {
        "schema_version": "midogpp_uniform_b_canonical_promotion_decision_v1",
        "status": "PASS",
        "decision": "PROMOTED_AS_NEW_CANONICAL_REFERENCE",
        "representation_id": REPRESENTATION_ID,
        "promotion_review_id": PROMOTION_REVIEW_ID,
        "heldout_centers": len(rows),
        "source_inner_reference_mean_bacc": mean_bacc,
        "confirmation_mean_delta": confirmation["paired_mean_delta"],
        "confirmation_strict_wins": confirmation["strict_wins"],
        "test_split_consumed_for_representation_adoption": True,
        "classifier_locks_imported_from_diagnostics": False,
        "canonical_a_retained": True,
        "automatic_downstream_migration": False,
        "claim_scope": "real_feature_transfer_only",
        "new_center_generalization_claimed": False,
        "external_dataset_generalization_claimed": False,
    }


def _write_content_index(root: Path) -> None:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = str(path.relative_to(root))
        if relative == "manifests/content_index.json":
            continue
        files.append({"path": relative, "sha256": _sha256_file(path)})
    payload = {
        "schema_version": "midogpp_uniform_b_canonical_reference_content_index_v1",
        "files": files,
    }
    payload["content_hash"] = stable_hash(payload)
    write_json(root / "manifests/content_index.json", payload)


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError(f"Uniform-B canonical JSON must be an object: {path}.")
    return payload


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Uniform-B canonical payload lacks mapping {key!r}.")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
