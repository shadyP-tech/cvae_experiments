"""Exact seven-input provenance renderer for OE-PPUR v3.

The generic workspace renderer is intentionally not used for the authorized
configuration.  This renderer still consumes the canonical catalog as the
identity source, then independently hashes every declared provenance member.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
from typing import Any

from ...protocol import ProtocolError
from ....workspace import MidogppWorkspace, WorkspaceError
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.identity import (
    AUTHORIZATION_AMENDMENT_ARTIFACT_ID,
    CLAIM_SCOPE,
    DIRECT_INPUT_ARTIFACT_IDS,
    EXPERIMENT_ID,
    INPUT_RELATIVE_MEMBERS,
    SOURCE_SUPERVISION_ARTIFACT_ID,
)
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.hashing import (
    canonical_hash,
    require_sha256,
)
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.run_paths import (
    assert_no_symlink_chain,
)
from .durable_io import hash_unique_regular_file
from .paths import CanonicalPreparationPaths


@dataclass(frozen=True, slots=True)
class PreissuanceInputInventoryReceipt:
    existing_input_inventory_hash: str
    authorized_semantics_hash: str
    existing_input_count: int
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for role in (
            "existing_input_inventory_hash",
            "authorized_semantics_hash",
        ):
            object.__setattr__(
                self,
                role,
                require_sha256(getattr(self, role), role.replace("_", " ")),
            )
        if self.existing_input_count != 6:
            raise ProtocolError("OE-PPUR v3 preissuance input count drifted.")
        object.__setattr__(
            self,
            "receipt_hash",
            canonical_hash(
                {
                    "schema_version": "oe_ppur_v3_preissuance_input_inventory_v1",
                    "existing_input_inventory_hash": self.existing_input_inventory_hash,
                    "authorized_semantics_hash": self.authorized_semantics_hash,
                    "existing_input_count": 6,
                    "amendment_file_opened": False,
                    "target_labels_opened": False,
                }
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v3_preissuance_input_inventory_v1",
            "existing_input_inventory_hash": self.existing_input_inventory_hash,
            "authorized_semantics_hash": self.authorized_semantics_hash,
            "existing_input_count": 6,
            "amendment_file_opened": False,
            "target_labels_opened": False,
            "receipt_hash": self.receipt_hash,
        }


def build_exact_input_manifest(
    workspace: MidogppWorkspace,
    paths: CanonicalPreparationPaths,
    *,
    authorized_semantics: Mapping[str, Mapping[str, str]],
    prospective_amendment_bytes: bytes | None = None,
) -> dict[str, object]:
    """Build the sole provenance payload accepted by the v3 run admission."""

    if workspace.repo_root != paths.repository_root:
        raise ProtocolError("OE-PPUR v3 preparation workspace root drifted.")
    try:
        workspace.validate()
        experiment = workspace.get_experiment(EXPERIMENT_ID)
    except WorkspaceError as exc:
        raise ProtocolError("OE-PPUR v3 preparation workspace is invalid.") from exc
    if (
        experiment.stage != "90_oracles_and_diagnostics"
        or experiment.claim_scope != CLAIM_SCOPE
        or experiment.input_artifact_ids != DIRECT_INPUT_ARTIFACT_IDS
    ):
        raise ProtocolError("OE-PPUR v3 registry identity drifted.")

    live_semantics = _validate_authorized_semantics(authorized_semantics)
    binding_by_id = {row.artifact_id: row for row in paths.input_bindings}
    rows: list[dict[str, object]] = []
    for artifact_id in sorted(DIRECT_INPUT_ARTIFACT_IDS):
        prospective_amendment = (
            artifact_id == AUTHORIZATION_AMENDMENT_ARTIFACT_ID
            and prospective_amendment_bytes is not None
        )
        binding = binding_by_id[artifact_id]
        catalog_entry = workspace.artifacts[artifact_id]
        root = _artifact_root(binding.path, artifact_id=artifact_id)
        try:
            catalog_root = workspace.resolve_artifact(
                artifact_id,
                require_exists=not prospective_amendment,
            )
        except WorkspaceError as exc:
            raise ProtocolError(
                f"OE-PPUR v3 input is unavailable: {artifact_id}."
            ) from exc
        if root != catalog_root:
            raise ProtocolError(
                f"OE-PPUR v3 input escaped its catalog root: {artifact_id}."
            )
        assert_no_symlink_chain(
            root,
            allow_missing_leaf=prospective_amendment,
        )
        if root.is_symlink() or (
            not prospective_amendment and not root.is_dir()
        ):
            raise ProtocolError("OE-PPUR v3 provenance root is unsafe.")
        if prospective_amendment and (
            root.exists()
            or not root.parent.is_dir()
            or root.parent.is_symlink()
        ):
            raise ProtocolError("OE-PPUR v3 prospective amendment root is unsafe.")

        members = catalog_entry.provenance_files
        relative_member = INPUT_RELATIVE_MEMBERS[
            DIRECT_INPUT_ARTIFACT_IDS.index(artifact_id)
        ]
        if not members and relative_member:
            members = (relative_member,)
        file_rows = (
            tuple(
                _prospective_file_integrity_row(
                    root,
                    relative,
                    raw=prospective_amendment_bytes,
                    expected=catalog_entry.expected_file_hashes.get(relative),
                )
                for relative in members
            )
            if prospective_amendment
            else tuple(
                _file_integrity_row(
                    root,
                    relative,
                    expected=catalog_entry.expected_file_hashes.get(relative),
                )
                for relative in members
            )
        )
        expected_present = bool(catalog_entry.expected_file_hashes)
        semantic_identities = dict(catalog_entry.semantic_identities)
        semantic_identities.update(live_semantics.get(artifact_id, {}))
        rows.append(
            {
                "artifact_id": artifact_id,
                "resolved_path": root.as_posix(),
                "stage": catalog_entry.stage,
                "evidence_label": catalog_entry.evidence_label,
                "claim_scope": catalog_entry.claim_scope,
                "semantic_identities": semantic_identities,
                "semantic_identities_are_file_hashes": False,
                "file_integrity": {
                    "status": (
                        "EXPECTED_FILE_HASHES_MATCH"
                        if expected_present
                        else "HASHES_RECORDED_NO_EXPECTATIONS"
                        if file_rows
                        else "NO_PROVENANCE_FILES_DECLARED"
                    ),
                    "default_recording_algorithm": "sha256",
                    "files": list(file_rows),
                },
                "exists": True,
            }
        )
    return {
        "schema_version": "midogpp_input_artifacts_v2",
        "dataset_id": "midogpp",
        "experiment_id": EXPERIMENT_ID,
        "stage": "90_oracles_and_diagnostics",
        "claim_scope": CLAIM_SCOPE,
        "selection_used_target_eval_artifacts": False,
        "input_artifacts": rows,
        **_git_state(
            paths.repository_root,
            excluded_roots=(paths.artifact_root, paths.amendment_root),
        ),
    }


def validate_preissuance_input_inventory(
    workspace: MidogppWorkspace,
    paths: CanonicalPreparationPaths,
    *,
    authorized_semantics: Mapping[str, Mapping[str, str]],
) -> PreissuanceInputInventoryReceipt:
    """Hash every already-existing direct input before issuing input #7."""

    if workspace.repo_root != paths.repository_root:
        raise ProtocolError("OE-PPUR v3 preissuance workspace root drifted.")
    try:
        workspace.validate()
        experiment = workspace.get_experiment(EXPERIMENT_ID)
    except WorkspaceError as exc:
        raise ProtocolError("OE-PPUR v3 preissuance workspace is invalid.") from exc
    if (
        experiment.stage != "90_oracles_and_diagnostics"
        or experiment.claim_scope != CLAIM_SCOPE
        or experiment.input_artifact_ids != DIRECT_INPUT_ARTIFACT_IDS
    ):
        raise ProtocolError("OE-PPUR v3 preissuance registry identity drifted.")
    live_semantics = _validate_authorized_semantics(authorized_semantics)
    binding_by_id = {row.artifact_id: row for row in paths.input_bindings}
    rows: list[dict[str, object]] = []
    for artifact_id in DIRECT_INPUT_ARTIFACT_IDS[:6]:
        binding = binding_by_id[artifact_id]
        catalog_entry = workspace.artifacts[artifact_id]
        root = _artifact_root(binding.path, artifact_id=artifact_id)
        try:
            catalog_root = workspace.resolve_artifact(
                artifact_id,
                require_exists=True,
            )
        except WorkspaceError as exc:
            raise ProtocolError(
                f"OE-PPUR v3 preissuance input is unavailable: {artifact_id}."
            ) from exc
        if root != catalog_root:
            raise ProtocolError("OE-PPUR v3 preissuance input escaped its catalog root.")
        assert_no_symlink_chain(root)
        if root.is_symlink() or not root.is_dir():
            raise ProtocolError("OE-PPUR v3 preissuance provenance root is unsafe.")
        relative_member = INPUT_RELATIVE_MEMBERS[
            DIRECT_INPUT_ARTIFACT_IDS.index(artifact_id)
        ]
        members = catalog_entry.provenance_files or (
            (relative_member,) if relative_member else ()
        )
        rows.append(
            {
                "artifact_id": artifact_id,
                "resolved_path": root.as_posix(),
                "files": [
                    _file_integrity_row(
                        root,
                        relative,
                        expected=catalog_entry.expected_file_hashes.get(relative),
                    )
                    for relative in members
                ],
            }
        )
    return PreissuanceInputInventoryReceipt(
        existing_input_inventory_hash=canonical_hash(
            {
                "schema_version": "oe_ppur_v3_preissuance_existing_inputs_v1",
                "rows": rows,
            }
        ),
        authorized_semantics_hash=canonical_hash(live_semantics),
        existing_input_count=len(rows),
    )


def _validate_authorized_semantics(
    value: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping) or set(value) != {
        SOURCE_SUPERVISION_ARTIFACT_ID,
        AUTHORIZATION_AMENDMENT_ARTIFACT_ID,
    }:
        raise ProtocolError("OE-PPUR v3 live provenance inventory drifted.")
    normalized: dict[str, dict[str, str]] = {}
    for artifact_id, identities in value.items():
        if (
            not isinstance(identities, Mapping)
            or not identities
            or not all(
                isinstance(key, str)
                and key
                and isinstance(item, str)
                and item
                for key, item in identities.items()
            )
        ):
            raise ProtocolError("OE-PPUR v3 live provenance is malformed.")
        normalized[str(artifact_id)] = dict(identities)
    return normalized


def _artifact_root(path: Path, *, artifact_id: str) -> Path:
    relative = Path(
        INPUT_RELATIVE_MEMBERS[DIRECT_INPUT_ARTIFACT_IDS.index(artifact_id)]
    )
    root = Path(path)
    for _part in relative.parts:
        root = root.parent
    return root


def _file_integrity_row(
    root: Path,
    relative: str,
    *,
    expected: Any,
) -> dict[str, object]:
    member_relative = Path(relative)
    if (
        not relative
        or member_relative.is_absolute()
        or ".." in member_relative.parts
    ):
        raise ProtocolError("OE-PPUR v3 provenance member path is unsafe.")
    member = root / member_relative
    assert_no_symlink_chain(member)
    digest, size = hash_unique_regular_file(
        member,
        role=f"provenance member {relative}",
    )
    expected_payload = None
    verification = "RECORDED_NO_EXPECTATION"
    if expected is not None:
        algorithm = str(expected.algorithm)
        value = str(expected.digest)
        if algorithm != "sha256" or digest != value:
            raise ProtocolError("OE-PPUR v3 catalog file hash drifted.")
        expected_payload = {"algorithm": algorithm, "digest": value}
        verification = "MATCH"
    return {
        "path": relative,
        "resolved_path": member.as_posix(),
        "exists": True,
        "expected": expected_payload,
        "size_bytes": size,
        "computed": {"sha256": digest},
        "verification": verification,
    }


def _prospective_file_integrity_row(
    root: Path,
    relative: str,
    *,
    raw: bytes | None,
    expected: Any,
) -> dict[str, object]:
    if not isinstance(raw, bytes):
        raise ProtocolError("OE-PPUR v3 prospective amendment bytes are untyped.")
    member_relative = Path(relative)
    if (
        not relative
        or member_relative.is_absolute()
        or ".." in member_relative.parts
    ):
        raise ProtocolError("OE-PPUR v3 prospective member path is unsafe.")
    digest = hashlib.sha256(raw).hexdigest()
    expected_payload = None
    verification = "RECORDED_NO_EXPECTATION"
    if expected is not None:
        algorithm = str(expected.algorithm)
        value = str(expected.digest)
        if algorithm != "sha256" or digest != value:
            raise ProtocolError("OE-PPUR v3 prospective catalog hash drifted.")
        expected_payload = {"algorithm": algorithm, "digest": value}
        verification = "MATCH"
    return {
        "path": relative,
        "resolved_path": (root / member_relative).as_posix(),
        "exists": True,
        "expected": expected_payload,
        "size_bytes": len(raw),
        "computed": {"sha256": digest},
        "verification": verification,
    }


def _git_state(
    repository_root: Path,
    *,
    excluded_roots: tuple[Path, ...] = (),
) -> dict[str, object]:
    excluded: tuple[str, ...] = tuple(
        Path(root).relative_to(repository_root).as_posix().rstrip("/")
        for root in excluded_roots
    )
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        raw_status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        status = "".join(
            line
            for line in raw_status.splitlines(keepends=True)
            if not _status_line_is_excluded(line, excluded)
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return {
            "repository_revision": "unknown",
            "repository_dirty": None,
            "repository_status_hash": "unknown",
        }
    return {
        "repository_revision": revision or "unknown",
        "repository_dirty": bool(status.strip()),
        "repository_status_hash": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _status_line_is_excluded(line: str, prefixes: tuple[str, ...]) -> bool:
    if not prefixes or len(line) < 4:
        return False
    rendered = line[3:].strip()
    candidate = rendered.rsplit(" -> ", 1)[-1]
    return any(
        candidate == prefix or candidate.startswith(f"{prefix}/")
        for prefix in prefixes
    )


__all__ = (
    "PreissuanceInputInventoryReceipt",
    "build_exact_input_manifest",
    "validate_preissuance_input_inventory",
)
