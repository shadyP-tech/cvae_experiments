"""Reconstruction-based validation for the metadata compatibility bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .bundle import CONTENT_INDEX_MEMBERS, REQUIRED_FILES
from .config import MetadataCompatibilityConfig, load_metadata_compatibility_config
from .contracts import (
    CLAIM_SCOPE,
    DOMAIN_AXIS,
    DOMAIN_MAPPING_MEMBER,
    DOMAIN_MAPPING_SHA256,
    ELIGIBLE_CENTERS,
    EXPECTED_COMPATIBILITY_LOCK_HASH,
    EXPECTED_COMPATIBILITY_SCORE_TABLE_HASH,
    EXPECTED_METADATA_PROFILE_LOCK_HASH,
    EXPECTED_METADATA_PROFILE_TABLE_HASH,
    EXPECTED_PROFILE_COUNT,
    EXPECTED_SCORE_COUNT,
    EXPERIMENT_ID,
    INPUT_ARTIFACT_ID,
    MetadataProfile,
)
from .locks import (
    build_compatibility_lock,
    build_metadata_profile_lock,
    read_compatibility_lock,
    read_metadata_profile_lock,
)
from .payloads import (
    compatibility_decision_payload,
    leakage_report_payload,
    protocol_manifest_payload,
    run_state_payload,
)
from .profiles import derive_metadata_profiles
from .scoring import derive_compatibility_scores
from .table_io import (
    read_compatibility_scores_table,
    read_metadata_profiles_table,
)


def validate_metadata_compatibility_bundle(
    root: str | Path,
    *,
    config: MetadataCompatibilityConfig,
    allow_pending: bool = False,
    _expected_profiles: Mapping[str, MetadataProfile] | None = None,
) -> dict[str, object]:
    """Rebuild every semantic payload and reject any undeclared artifact member."""

    path = Path(root)
    required = set(REQUIRED_FILES)
    if allow_pending:
        required.remove("reports/validation_report.json")
    missing = sorted(relative for relative in required if not (path / relative).is_file())
    if missing:
        raise ProtocolError(f"Metadata compatibility artifact is incomplete: {missing}.")
    _validate_closed_world(path)
    if load_metadata_compatibility_config(path / "config.resolved.yaml") != config:
        raise ProtocolError("Metadata compatibility resolved config drifted.")
    validate_metadata_compatibility_provenance(path, config=config)

    reconstructed_profiles = derive_metadata_profiles(
        config.metadata_mapping_path,
        expected_sha256=config.expected_domain_mapping_sha256,
    )
    if _expected_profiles is not None and dict(_expected_profiles) != reconstructed_profiles:
        raise ProtocolError("Runner and validator metadata-profile derivations disagree.")
    expected_scores = derive_compatibility_scores(reconstructed_profiles)
    expected_profile_lock = build_metadata_profile_lock(config, reconstructed_profiles)
    expected_compatibility_lock = build_compatibility_lock(
        config,
        expected_profile_lock,
        expected_scores,
    )

    observed_profiles = read_metadata_profiles_table(path)
    if observed_profiles != reconstructed_profiles:
        raise ProtocolError("Metadata profile table drifted from the frozen mapping.")
    observed_scores = read_compatibility_scores_table(path)
    if observed_scores != expected_scores:
        raise ProtocolError("Metadata compatibility rows drifted from exact-match scoring.")

    observed_profile_lock = read_metadata_profile_lock(path)
    if observed_profile_lock != expected_profile_lock:
        raise ProtocolError("Metadata profile lock drifted from reconstructed profiles.")
    observed_compatibility_lock = read_compatibility_lock(path)
    if observed_compatibility_lock.to_payload() != expected_compatibility_lock.to_payload():
        raise ProtocolError("Metadata compatibility lock drifted from reconstructed scores.")

    _require_exact_payload(
        path / "manifests/protocol_manifest.json",
        protocol_manifest_payload(
            config,
            expected_profile_lock,
            expected_compatibility_lock,
        ),
        "protocol manifest",
    )
    _require_exact_payload(
        path / "reports/compatibility_decision.json",
        compatibility_decision_payload(expected_compatibility_lock),
        "compatibility decision",
    )
    _require_exact_payload(
        path / "reports/leakage_report.json",
        leakage_report_payload(),
        "leakage report",
    )
    _require_exact_payload(
        path / "reports/run_state.json",
        run_state_payload("COMPLETE"),
        "run state",
    )
    _validate_content_index(path)

    checks: dict[str, object] = {
        "status": "PASS",
        "domain_mapping_sha256": DOMAIN_MAPPING_SHA256,
        "metadata_profile_lock_hash": EXPECTED_METADATA_PROFILE_LOCK_HASH,
        "compatibility_lock_hash": EXPECTED_COMPATIBILITY_LOCK_HASH,
        "metadata_profile_table_hash": EXPECTED_METADATA_PROFILE_TABLE_HASH,
        "compatibility_score_table_hash": EXPECTED_COMPATIBILITY_SCORE_TABLE_HASH,
        "profile_count": EXPECTED_PROFILE_COUNT,
        "ordered_score_count": EXPECTED_SCORE_COUNT,
        "eligible_target_count": len(ELIGIBLE_CENTERS),
        "center_4_emitted": False,
        "target_expert_excluded": True,
        "scorer_uses_profile_values_only": True,
        "center_or_domain_ids_passed_to_scorer": False,
        "metadata_score_is_proxy_only": True,
        "ranking_performed": False,
        "selection_performed": False,
        "weighting_performed": False,
        "target_labels_used": False,
        "nelbo_computed": False,
        "true_utility_computed": False,
        "routing_quality_claimed": False,
    }
    if not allow_pending:
        _require_exact_payload(
            path / "reports/validation_report.json",
            {
                "schema_version": (
                    "midogpp_uniform_b_v2_metadata_compatibility_validation_v1"
                ),
                "status": "PASS",
                "validator": "validate_metadata_compatibility_bundle",
                "checks": checks,
            },
            "validation report",
        )
    return checks


def validate_metadata_compatibility_provenance(
    root: str | Path,
    *,
    config: MetadataCompatibilityConfig,
) -> None:
    output_root = Path(root)
    manifest = _json(output_root / "provenance/input_artifacts.json")
    allowed_manifest_fields = {
        "schema_version",
        "dataset_id",
        "experiment_id",
        "stage",
        "claim_scope",
        "selection_used_target_eval_artifacts",
        "input_artifacts",
        "repository_revision",
        "repository_dirty",
        "repository_status_hash",
    }
    if set(manifest) != allowed_manifest_fields:
        raise ProtocolError("Metadata compatibility provenance schema drifted.")
    expected_header = {
        "schema_version": "midogpp_input_artifacts_v2",
        "dataset_id": "midogpp",
        "experiment_id": EXPERIMENT_ID,
        "stage": "60_routing_and_composition",
        "claim_scope": CLAIM_SCOPE,
        "selection_used_target_eval_artifacts": False,
    }
    if any(manifest.get(key) != value for key, value in expected_header.items()):
        raise ProtocolError("Metadata compatibility provenance identity drifted.")
    if (
        not _is_hex(manifest.get("repository_revision"), length=40)
        or not isinstance(manifest.get("repository_dirty"), bool)
        or not _is_hex(manifest.get("repository_status_hash"), length=64)
    ):
        raise ProtocolError("Metadata compatibility repository provenance is malformed.")

    rows = manifest.get("input_artifacts")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], Mapping):
        raise ProtocolError("Metadata compatibility must have exactly one input artifact.")
    row = rows[0]
    allowed_row_fields = {
        "artifact_id",
        "resolved_path",
        "stage",
        "evidence_label",
        "claim_scope",
        "semantic_identities",
        "semantic_identities_are_file_hashes",
        "file_integrity",
        "exists",
    }
    if set(row) != allowed_row_fields:
        raise ProtocolError("Metadata compatibility input provenance fields drifted.")
    expected_root = config.metadata_mapping_path.parent.resolve()
    expected_semantic_identities = {
        "routing_metadata_source": "midogpp_domain_mapping_v1",
        "domain_axis": DOMAIN_AXIS,
        "domain_mapping_sha256": DOMAIN_MAPPING_SHA256,
    }
    if (
        row.get("artifact_id") != INPUT_ARTIFACT_ID
        or Path(str(row.get("resolved_path", ""))).resolve() != expected_root
        or row.get("stage") != "dataset_contract"
        or row.get("evidence_label") != "ROUTING_METADATA_INPUT_AUTHORIZED"
        or row.get("claim_scope") != "dataset_contract_and_split_provenance"
        or row.get("semantic_identities") != expected_semantic_identities
        or row.get("semantic_identities_are_file_hashes") is not False
        or row.get("exists") is not True
    ):
        raise ProtocolError("Metadata compatibility input provenance drifted.")

    integrity = row.get("file_integrity")
    if not isinstance(integrity, Mapping) or set(integrity) != {
        "status",
        "default_recording_algorithm",
        "files",
    }:
        raise ProtocolError("Metadata compatibility input integrity is malformed.")
    if (
        integrity.get("status") != "EXPECTED_FILE_HASHES_MATCH"
        or integrity.get("default_recording_algorithm") != "sha256"
    ):
        raise ProtocolError("Metadata compatibility input hash expectation was not enforced.")
    files = integrity.get("files")
    if not isinstance(files, list) or len(files) != 1 or not isinstance(files[0], Mapping):
        raise ProtocolError("Metadata compatibility input file inventory drifted.")
    item = files[0]
    if set(item) != {
        "path",
        "resolved_path",
        "exists",
        "expected",
        "size_bytes",
        "computed",
        "verification",
    }:
        raise ProtocolError("Metadata compatibility input file row schema drifted.")
    member = config.metadata_mapping_path.resolve()
    expected_digest = {"algorithm": "sha256", "digest": DOMAIN_MAPPING_SHA256}
    if (
        item.get("path") != DOMAIN_MAPPING_MEMBER
        or Path(str(item.get("resolved_path", ""))).resolve() != member
        or item.get("exists") is not True
        or item.get("expected") != expected_digest
        or item.get("computed") != {"sha256": DOMAIN_MAPPING_SHA256}
        or item.get("verification") != "MATCH"
        or not member.is_file()
        or item.get("size_bytes") != member.stat().st_size
        or _sha256_file(member) != DOMAIN_MAPPING_SHA256
    ):
        raise ProtocolError("Metadata compatibility input member drifted.")


def _validate_content_index(root: Path) -> None:
    payload = _json(root / "manifests/content_index.json")
    if set(payload) != {"schema_version", "records", "content_hash"}:
        raise ProtocolError("Metadata compatibility content-index schema drifted.")
    if (
        payload.get("schema_version")
        != "midogpp_uniform_b_v2_metadata_compatibility_content_v1"
        or payload.get("content_hash")
        != stable_hash({key: value for key, value in payload.items() if key != "content_hash"})
    ):
        raise ProtocolError("Metadata compatibility content-index identity drifted.")
    rows = payload.get("records")
    if not isinstance(rows, list) or len(rows) != len(CONTENT_INDEX_MEMBERS):
        raise ProtocolError("Metadata compatibility content-index coverage drifted.")
    observed_paths: list[str] = []
    for raw in rows:
        if not isinstance(raw, Mapping) or set(raw) != {
            "relative_path",
            "sha256",
            "size_bytes",
        }:
            raise ProtocolError("Metadata compatibility content-index row drifted.")
        relative = str(raw.get("relative_path", ""))
        member = _safe_member(root, relative)
        if (
            not member.is_file()
            or raw.get("sha256") != _sha256_file(member)
            or raw.get("size_bytes") != member.stat().st_size
        ):
            raise ProtocolError(f"Metadata compatibility content member drifted: {relative}.")
        observed_paths.append(relative)
    if tuple(observed_paths) != CONTENT_INDEX_MEMBERS:
        raise ProtocolError("Metadata compatibility content-index order drifted.")


def _validate_closed_world(root: Path) -> None:
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


def _require_exact_payload(path: Path, expected: Mapping[str, object], label: str) -> None:
    if _json(path) != dict(expected):
        raise ProtocolError(f"Metadata compatibility {label} drifted.")


def _json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read metadata compatibility JSON: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"Metadata compatibility JSON must be an object: {path}.")
    return payload


def _safe_member(root: Path, relative: str) -> Path:
    resolved_root = root.resolve()
    member = (resolved_root / relative).resolve()
    if member == resolved_root or not member.is_relative_to(resolved_root):
        raise ProtocolError("Metadata compatibility content path escapes its artifact root.")
    return member


def _is_hex(value: object, *, length: int) -> bool:
    rendered = str(value or "")
    return len(rendered) == length and all(
        character in "0123456789abcdef" for character in rendered
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = (
    "REQUIRED_FILES",
    "validate_metadata_compatibility_bundle",
    "validate_metadata_compatibility_provenance",
)
