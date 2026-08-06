"""Materialize the fail-closed Stage-60 metadata compatibility artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from ...reporting import write_csv_rows, write_json
from .bundle import CONTENT_INDEX_MEMBERS, REQUIRED_FILES
from .config import MetadataCompatibilityConfig
from .locks import build_compatibility_lock, build_metadata_profile_lock
from .payloads import (
    compatibility_decision_payload,
    leakage_report_payload,
    protocol_manifest_payload,
    run_state_payload,
)
from .profiles import derive_metadata_profiles, metadata_profile_rows
from .scoring import derive_compatibility_scores
from .table_io import PROFILE_COLUMNS, SCORE_COLUMNS


def run_metadata_compatibility_lock(
    config: MetadataCompatibilityConfig,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    """Build the lock; ``artifact_root`` is a low-level test/orchestration override.

    Production callers must first enforce the canonical path with
    ``workspace_binding.validate_production_workspace_binding``; the CLI does so.
    """

    root = Path(artifact_root or config.artifact_root)
    for relative in ("manifests", "reports", "tables", "provenance"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    state_path = root / "reports/run_state.json"
    try:
        _assert_closed_world(root)
        if not (root / "config.resolved.yaml").is_file() or not (
            root / "provenance/input_artifacts.json"
        ).is_file():
            raise ProtocolError(
                "Metadata compatibility lock requires resolved config and input provenance."
            )
        if state_path.is_file() and _json(state_path).get("status") == "COMPLETE":
            from .validation import validate_metadata_compatibility_bundle

            validate_metadata_compatibility_bundle(root, config=config)
            return root
    except Exception:
        if state_path.exists():
            _write_state(root, "FAILED")
        raise

    _write_state(root, "RUNNING")
    try:
        profiles = derive_metadata_profiles(
            config.metadata_mapping_path,
            expected_sha256=config.expected_domain_mapping_sha256,
        )
        scores = derive_compatibility_scores(profiles)
        profile_lock = build_metadata_profile_lock(config, profiles)
        compatibility_lock = build_compatibility_lock(config, profile_lock, scores)

        write_csv_rows(
            root / "tables/metadata_profiles.csv",
            list(metadata_profile_rows(profiles)),
            columns=PROFILE_COLUMNS,
        )
        write_csv_rows(
            root / "tables/compatibility_scores.csv",
            [row.to_payload() for row in scores],
            columns=SCORE_COLUMNS,
        )
        write_json(root / "manifests/metadata_profile_lock.json", profile_lock)
        write_json(
            root / "manifests/compatibility_lock.json",
            compatibility_lock.to_payload(),
        )
        write_json(
            root / "manifests/protocol_manifest.json",
            protocol_manifest_payload(config, profile_lock, compatibility_lock),
        )
        write_json(
            root / "reports/compatibility_decision.json",
            compatibility_decision_payload(compatibility_lock),
        )
        write_json(root / "reports/leakage_report.json", leakage_report_payload())
        _write_content_index(root)
        _write_state(root, "COMPLETE")

        from .validation import validate_metadata_compatibility_bundle

        checks = validate_metadata_compatibility_bundle(
            root,
            config=config,
            allow_pending=True,
            _expected_profiles=profiles,
        )
        write_json(
            root / "reports/validation_report.json",
            {
                "schema_version": (
                    "midogpp_uniform_b_v2_metadata_compatibility_validation_v1"
                ),
                "status": "PASS",
                "validator": "validate_metadata_compatibility_bundle",
                "checks": checks,
            },
        )
        validate_metadata_compatibility_bundle(
            root,
            config=config,
            _expected_profiles=profiles,
        )
    except Exception:
        _write_state(root, "FAILED")
        raise
    return root


def _write_content_index(root: Path) -> None:
    records = []
    for relative in CONTENT_INDEX_MEMBERS:
        member = root / relative
        if not member.is_file():
            raise ProtocolError(
                f"Metadata compatibility content member is missing: {relative}."
            )
        records.append(
            {
                "relative_path": relative,
                "sha256": _sha256_file(member),
                "size_bytes": member.stat().st_size,
            }
        )
    payload: dict[str, object] = {
        "schema_version": "midogpp_uniform_b_v2_metadata_compatibility_content_v1",
        "records": records,
    }
    payload["content_hash"] = stable_hash(payload)
    write_json(root / "manifests/content_index.json", payload)


def _assert_closed_world(root: Path) -> None:
    actual = {
        member.relative_to(root).as_posix()
        for member in root.rglob("*")
        if member.is_file()
    }
    unexpected = sorted(actual.difference(REQUIRED_FILES))
    if unexpected:
        raise ProtocolError(
            f"Metadata compatibility artifact contains unexpected files: {unexpected}."
        )


def _write_state(root: Path, status: str) -> None:
    write_json(root / "reports/run_state.json", run_state_payload(status))


def _json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read metadata compatibility JSON: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"Metadata compatibility JSON must be an object: {path}.")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ("run_metadata_compatibility_lock",)
