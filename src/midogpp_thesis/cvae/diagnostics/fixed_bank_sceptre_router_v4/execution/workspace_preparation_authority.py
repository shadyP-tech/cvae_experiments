"""Pre-render workspace authority admission for SCEPTRE v4.

This adapter authenticates only the consumer-specific execution amendment.  It
must complete before the workspace resolves the source-inner surface, consumed
test cache, role-scoped manifest, or parent consumption ledger.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol

import yaml

from ....protocol import ProtocolError
from ..config import load_config
from ..experiment_contracts import (
    EXECUTION_AMENDMENT_ARTIFACT_ID,
    EXECUTION_AMENDMENT_FILENAME,
    INPUT_ARTIFACT_IDS,
)
from ..identity import EXPERIMENT_ID, file_sha256
from .inputs import canonical_execution_amendment_payload


class AuthorityMemberLike(Protocol):
    path: Path
    expected_sha256: str


AuthorityMemberResolver = Callable[[str, str], AuthorityMemberLike]


class SceptreV4WorkspaceAuthorityError(ValueError):
    """Raised before protected MIDOG++ inputs are rendered or hashed."""


@dataclass(frozen=True, slots=True)
class SceptreV4WorkspaceAuthorityReceipt:
    """Exact safe-file bytes authenticated before workspace rendering."""

    config_path: Path
    config_sha256: str
    amendment_path: Path
    amendment_sha256: str


def validate_workspace_preparation_authority(
    *,
    repo_root: Path,
    experiment_id: str,
    config_path: str | None,
    input_artifact_ids: Sequence[str],
    resolve_authority_member: AuthorityMemberResolver,
) -> SceptreV4WorkspaceAuthorityReceipt:
    """Authenticate the exact v4 execution amendment and nothing else."""

    if (
        experiment_id != EXPERIMENT_ID
        or tuple(input_artifact_ids) != INPUT_ARTIFACT_IDS
        or config_path is None
    ):
        raise SceptreV4WorkspaceAuthorityError(
            "SCEPTRE v4 workspace authority gate binding drifted."
        )
    config_file = _safe_repository_config(repo_root, config_path)
    raw = _read_yaml_mapping(config_file)
    _validate_unresolved_amendment_uri(raw)
    try:
        config = load_config(config_file)
    except ProtocolError as exc:
        raise SceptreV4WorkspaceAuthorityError(
            "SCEPTRE v4 checked-in execution config is not authentic."
        ) from exc
    if (
        config.experiment_id != EXPERIMENT_ID
        or config.input_artifact_ids != INPUT_ARTIFACT_IDS
        or not config.execution_authorized
    ):
        raise SceptreV4WorkspaceAuthorityError(
            "SCEPTRE v4 executable config authority drifted."
        )

    member = resolve_authority_member(
        EXECUTION_AMENDMENT_ARTIFACT_ID,
        EXECUTION_AMENDMENT_FILENAME,
    )
    amendment_path = Path(member.path)
    expected_sha256 = config.expected_execution_amendment_sha256
    if member.expected_sha256 != expected_sha256:
        raise SceptreV4WorkspaceAuthorityError(
            "SCEPTRE v4 catalog and config amendment hashes disagree."
        )
    if amendment_path.is_symlink() or not amendment_path.is_file():
        raise SceptreV4WorkspaceAuthorityError(
            "SCEPTRE v4 execution amendment bytes are absent or drifted."
        )
    amendment_sha256 = file_sha256(amendment_path)
    if amendment_sha256 != expected_sha256:
        raise SceptreV4WorkspaceAuthorityError(
            "SCEPTRE v4 execution amendment bytes are absent or drifted."
        )
    try:
        amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SceptreV4WorkspaceAuthorityError(
            "SCEPTRE v4 execution amendment is not readable canonical JSON."
        ) from exc
    if (
        not isinstance(amendment, dict)
        or amendment != canonical_execution_amendment_payload(config)
    ):
        raise SceptreV4WorkspaceAuthorityError(
            "SCEPTRE v4 consumer-specific execution amendment drifted."
        )
    return SceptreV4WorkspaceAuthorityReceipt(
        config_path=config_file,
        config_sha256=file_sha256(config_file),
        amendment_path=amendment_path,
        amendment_sha256=amendment_sha256,
    )


def _safe_repository_config(repo_root: Path, config_path: str) -> Path:
    root = repo_root.resolve()
    raw_path = Path(config_path)
    candidate = raw_path if raw_path.is_absolute() else root / raw_path
    if candidate.is_symlink():
        raise SceptreV4WorkspaceAuthorityError(
            "SCEPTRE v4 workspace config may not be a symlink."
        )
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise SceptreV4WorkspaceAuthorityError(
            "SCEPTRE v4 workspace config is absent or outside the repository."
        )
    return resolved


def _read_yaml_mapping(path: Path) -> Mapping[str, object]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise SceptreV4WorkspaceAuthorityError(
            "SCEPTRE v4 workspace config is not readable YAML."
        ) from exc
    if not isinstance(raw, Mapping):
        raise SceptreV4WorkspaceAuthorityError(
            "SCEPTRE v4 workspace config must be a mapping."
        )
    return raw


def _validate_unresolved_amendment_uri(raw: Mapping[str, object]) -> None:
    inputs = raw.get("inputs")
    if not isinstance(inputs, Mapping) or inputs.get(
        "execution_amendment_path"
    ) != (
        f"artifact://{EXECUTION_AMENDMENT_ARTIFACT_ID}/"
        f"{EXECUTION_AMENDMENT_FILENAME}"
    ):
        raise SceptreV4WorkspaceAuthorityError(
            "SCEPTRE v4 checked-in config must bind the exact amendment URI."
        )


__all__ = (
    "SceptreV4WorkspaceAuthorityReceipt",
    "SceptreV4WorkspaceAuthorityError",
    "validate_workspace_preparation_authority",
)
