"""Artifact utilities for the retrospective uniform-B replay."""

from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from midogpp_thesis.common.hashing import stable_hash

from ..artifacts import write_json
from ..downstream import balanced_accuracy
from ..protocol import ProtocolError
from .config import BootstrapConfig, UniformBReplayConfig


REQUIRED_FILES = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "manifests/frozen_protocol_snapshot.json",
    "manifests/protocol_manifest.json",
    "manifests/uniform_representation_lock.json",
    "manifests/source_lock_index.json",
    "manifests/content_index.json",
    "tables/source_lock_replay.csv",
    "tables/cache_alignment_audit.csv",
    "tables/uniform_b_outer_results.csv",
    "tables/uniform_b_outer_predictions.csv",
    "tables/paired_center_comparison.csv",
    "tables/outer_fit_audit.csv",
    "tables/canonical_a_replay.csv",
    "tables/v3_result_replay.csv",
    "reports/conditional_bootstrap.json",
    "reports/diagnostic_summary.json",
    "reports/diagnostic_report.md",
    "reports/leakage_provenance_report.json",
    "reports/runtime_summary.json",
    "reports/validation_report.json",
)


def input_hashes(config: UniformBReplayConfig) -> dict[str, str]:
    paths = {
        "source_v3_content_index": config.source_v3_root
        / "manifests/content_index.json",
        "source_v3_protocol": config.source_v3_root
        / "manifests/protocol_manifest.json",
        "source_v3_lock_index": config.source_v3_root
        / "manifests/decision_lock_index.json",
        "source_v3_validation": config.source_v3_root
        / "reports/validation_report.json",
        "b_alignment": config.b_cache_root / "manifests/row_alignment.json",
        "b_builder_report": config.b_cache_root / "reports/cache_builder_report.json",
        "canonical_reference_protocol": config.canonical_reference_root
        / "manifests/protocol_manifest.json",
        "canonical_reference_results": config.canonical_reference_root
        / "tables/classifier_tuned_source_results.csv",
        "canonical_reference_predictions": config.canonical_reference_root
        / "tables/classifier_tuned_predictions.csv",
    }
    for center in config.heldout_centers:
        paths[f"b_center_{center}"] = (
            config.b_cache_root
            / "embeddings/by_center"
            / f"center_{center}.pt"
        )
        paths[f"source_lock_{center}"] = (
            config.source_v3_root
            / "manifests/decision_locks"
            / f"center_{center}.json"
        )
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ProtocolError(f"Uniform-B inputs are incomplete: {missing}")
    return {key: sha256(path) for key, path in sorted(paths.items())}


def validate_source_bundle(config: UniformBReplayConfig) -> None:
    protocol = read_json(config.source_v3_root / "manifests/protocol_manifest.json")
    validation = read_json(config.source_v3_root / "reports/validation_report.json")
    lock_index = read_json(config.source_v3_root / "manifests/decision_lock_index.json")
    content = read_json(config.source_v3_root / "manifests/content_index.json")
    if (
        protocol.get("status") != "PASS"
        or protocol.get("protocol_hash") != config.source_protocol_hash
        or protocol.get("bundle_lock_hash") != config.source_bundle_lock_hash
        or validation.get("status") != "PASS"
        or validation.get("authoritative_bundle_verdict") is not True
        or lock_index.get("bundle_lock_hash") != config.source_bundle_lock_hash
        or lock_index.get("lock_count") != len(config.heldout_centers)
        or content.get("content_hash") != config.source_content_hash
    ):
        raise ProtocolError("Uniform-B source v3 bundle identity/status drifted.")
    validate_content_index(config.source_v3_root)


def paired_case_bootstrap(
    prediction_rows: Sequence[Mapping[str, object]],
    *,
    config: BootstrapConfig,
) -> dict[str, object]:
    import numpy as np  # type: ignore

    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in prediction_rows:
        grouped[(str(row["heldout_center"]), str(row["role"]))].append(row)
    centers = sorted({center for center, _role in grouped}, key=int)
    rng = np.random.default_rng(config.seed)
    deltas: list[float] = []
    attempts = rejected = 0
    while len(deltas) < config.valid_replicates and attempts < config.max_attempts:
        attempts += 1
        center_deltas: list[float] = []
        valid = True
        for center in centers:
            a = _by_case(grouped[(center, "canonical_a")])
            b = _by_case(grouped[(center, "uniform_b")])
            if set(a) != set(b):
                raise ProtocolError("Uniform-B paired bootstrap case pools differ.")
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
    if len(deltas) != config.valid_replicates:
        raise ProtocolError("Uniform-B bootstrap did not produce all valid replicates.")
    lower, upper = np.quantile(np.asarray(deltas), [0.025, 0.975]).tolist()
    return {
        "schema_version": "midogpp_uniform_b_conditional_bootstrap_v1",
        "status": "PASS",
        "seed": config.seed,
        "valid_replicates": len(deltas),
        "attempted_replicates": attempts,
        "rejected_class_missing_replicates": rejected,
        "mean_delta": float(np.mean(deltas)),
        "percentile_2_5": float(lower),
        "percentile_97_5": float(upper),
        "interval_role": "retrospective_conditional_paired_case_interval",
        "conditions_on_fixed_fits_and_imported_classifier_locks": True,
        "covers_representation_choice_uncertainty": False,
        "covers_new_center_uncertainty": False,
        "p_value_computed": False,
        "significance_decision_computed": False,
    }


def write_content_index(root: Path) -> None:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = str(path.relative_to(root))
        if relative == "manifests/content_index.json":
            continue
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    write_json(
        root / "manifests/content_index.json",
        {
            "schema_version": "midogpp_uniform_b_content_index_v1",
            "files": files,
            "content_hash": stable_hash(files),
        },
    )


def validate_content_index(root: Path) -> None:
    payload = read_json(root / "manifests/content_index.json")
    files = payload.get("files")
    if not isinstance(files, list):
        raise ProtocolError("Uniform-B content index is invalid.")
    expected = []
    for item in files:
        if not isinstance(item, Mapping):
            raise ProtocolError("Uniform-B content-index row is invalid.")
        path = root / str(item["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(item["size_bytes"])
            or sha256(path) != str(item["sha256"])
        ):
            raise ProtocolError(f"Uniform-B indexed file drifted: {path}")
        expected.append(dict(item))
    actual_paths = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "content_index.json"
    }
    if actual_paths != {str(item["path"]) for item in files}:
        raise ProtocolError("Uniform-B content-index coverage drifted.")
    if stable_hash(expected) != payload.get("content_hash"):
        raise ProtocolError("Uniform-B content hash drifted.")


def read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError(f"Expected a JSON object: {path}")
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _by_case(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, list[Mapping[str, object]]]:
    out: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        out[str(row["case_id"])].append(row)
    return dict(out)
