"""Pre-render workspace execution-authority gate for HARP v1.

Only the catalog-pinned HARP amendment may be resolved here.  All scientific
inputs and the output root remain unopened until this adapter has authenticated
the checked-in config, exact amendment bytes, current source closure, and the
absence of a previously claimed single-use lease.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol

import yaml

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_bytes
from ...runtime.artifact_io import sha256_file
from .authorization import (
    EXECUTION_AMENDMENT_ARTIFACT_ID,
    EXECUTION_AMENDMENT_FILENAME,
    WORKSPACE_ARTIFACT_CATALOG_RELATIVE_PATH,
    WORKSPACE_CONFIG_RELATIVE_PATH,
    WORKSPACE_REGISTRY_RELATIVE_PATH,
    lease_path,
    validate_execution_amendment_payload,
    validate_workspace_registration_execution_projection,
    workspace_registration_execution_contract,
)
from .config import INPUT_ARTIFACT_IDS, load_config
from .identity import EXPERIMENT_ID


class AuthorityMemberLike(Protocol):
    path: Path
    expected_sha256: str


AuthorityMemberResolver = Callable[[str, str], AuthorityMemberLike]


class HarpV1WorkspaceAuthorityError(ValueError):
    """Raised before protected HARP inputs or output paths are resolved."""


@dataclass(frozen=True, slots=True)
class HarpV1WorkspaceAuthorityReceipt:
    """Exact config and amendment bytes authenticated before rendering."""

    config_path: Path
    config_sha256: str
    amendment_path: Path
    amendment_sha256: str
    workspace_registration_contract_hash: str
    registry_path: Path
    registry_sha256: str
    artifact_catalog_path: Path
    artifact_catalog_sha256: str


def validate_workspace_preparation_authority(
    *,
    repo_root: Path,
    experiment_id: str,
    config_path: str | None,
    input_artifact_ids: Sequence[str],
    registration_projection: Mapping[str, object] | None,
    resolve_authority_member: AuthorityMemberResolver,
) -> HarpV1WorkspaceAuthorityReceipt:
    """Authenticate one unconsumed, HARP-specific execution amendment."""

    if (
        experiment_id != EXPERIMENT_ID
        or tuple(input_artifact_ids) != INPUT_ARTIFACT_IDS
        or config_path != WORKSPACE_CONFIG_RELATIVE_PATH
    ):
        raise HarpV1WorkspaceAuthorityError(
            "HARP v1 workspace authority gate binding drifted."
        )
    _validate_registration_projection(registration_projection)

    repository = Path(repo_root).resolve()
    config_file = _safe_repository_config(repository, config_path)
    raw_config = _read_yaml_mapping(config_file)
    _validate_unresolved_amendment_uri(raw_config)
    try:
        config = load_config(config_file)
    except ProtocolError as exc:
        raise HarpV1WorkspaceAuthorityError(
            "HARP v1 checked-in execution config is not authentic."
        ) from exc
    if (
        config.experiment_id != EXPERIMENT_ID
        or config.input_artifact_ids != INPUT_ARTIFACT_IDS
        or not config.execution_authorized
    ):
        raise HarpV1WorkspaceAuthorityError(
            "HARP v1 executable config authority drifted."
        )

    (
        registration_hash,
        registry_path,
        registry_sha256,
        artifact_catalog_path,
        artifact_catalog_sha256,
    ) = _validate_workspace_registration(repository)

    # This check precedes even the authority-member resolution.  Once a lease
    # directory exists, deleting outputs cannot restore the authorization.
    lease = lease_path(repository)
    if lease.exists() or lease.is_symlink():
        raise HarpV1WorkspaceAuthorityError(
            "HARP v1 single-use execution authorization is exhausted."
        )
    if lease.parent.is_symlink() or (
        lease.parent.exists() and not lease.parent.is_dir()
    ):
        raise HarpV1WorkspaceAuthorityError(
            "HARP v1 authorization lease parent is unsafe."
        )

    member = resolve_authority_member(
        EXECUTION_AMENDMENT_ARTIFACT_ID,
        EXECUTION_AMENDMENT_FILENAME,
    )
    amendment_path = Path(member.path)
    expected_sha256 = config.expected_execution_amendment_sha256
    if (
        type(expected_sha256) is not str
        or member.expected_sha256 != expected_sha256
    ):
        raise HarpV1WorkspaceAuthorityError(
            "HARP v1 catalog and config amendment hashes disagree."
        )
    if amendment_path.is_symlink() or not amendment_path.is_file():
        raise HarpV1WorkspaceAuthorityError(
            "HARP v1 execution amendment bytes are absent or unsafe."
        )
    resolved_amendment = amendment_path.resolve()
    if not resolved_amendment.is_relative_to(repository):
        raise HarpV1WorkspaceAuthorityError(
            "HARP v1 execution amendment must remain inside the repository."
        )
    amendment_sha256 = sha256_file(resolved_amendment)
    if amendment_sha256 != expected_sha256:
        raise HarpV1WorkspaceAuthorityError(
            "HARP v1 execution amendment bytes are absent or drifted."
        )
    try:
        amendment = json.loads(resolved_amendment.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HarpV1WorkspaceAuthorityError(
            "HARP v1 execution amendment is not readable JSON."
        ) from exc
    try:
        validate_execution_amendment_payload(
            amendment,
            config,
            repo_root=repository,
        )
    except ProtocolError as exc:
        raise HarpV1WorkspaceAuthorityError(
            "HARP v1 consumer-specific execution amendment drifted."
        ) from exc
    return HarpV1WorkspaceAuthorityReceipt(
        config_path=config_file,
        config_sha256=sha256_file(config_file),
        amendment_path=resolved_amendment,
        amendment_sha256=amendment_sha256,
        workspace_registration_contract_hash=registration_hash,
        registry_path=registry_path,
        registry_sha256=registry_sha256,
        artifact_catalog_path=artifact_catalog_path,
        artifact_catalog_sha256=artifact_catalog_sha256,
    )


def _validate_workspace_registration(
    repo_root: Path,
) -> tuple[str, Path, str, Path, str]:
    """Independently authenticate executable registry and output projection."""

    registry_path = _safe_repository_member(
        repo_root,
        WORKSPACE_REGISTRY_RELATIVE_PATH,
        label="registry",
    )
    catalog_path = _safe_repository_member(
        repo_root,
        WORKSPACE_ARTIFACT_CATALOG_RELATIVE_PATH,
        label="artifact catalog",
    )
    registry = _read_yaml_file_mapping(registry_path, label="registry")
    catalog = _read_yaml_file_mapping(catalog_path, label="artifact catalog")

    experiments = registry.get("experiments")
    if (
        isinstance(experiments, (str, bytes))
        or not isinstance(experiments, Sequence)
    ):
        raise HarpV1WorkspaceAuthorityError(
            "HARP v1 workspace registry experiment inventory is malformed."
        )
    matches = tuple(
        row
        for row in experiments
        if isinstance(row, Mapping) and row.get("experiment_id") == EXPERIMENT_ID
    )
    if len(matches) != 1:
        raise HarpV1WorkspaceAuthorityError(
            "HARP v1 workspace registry must contain exactly one experiment entry."
        )
    experiment = matches[0]
    runner = experiment.get("runner")
    input_ids = experiment.get("input_artifact_ids")
    if (
        not isinstance(runner, Mapping)
        or set(runner)
        != {"preparation_authority_gate", "environment", "argv"}
        or isinstance(input_ids, (str, bytes))
        or not isinstance(input_ids, Sequence)
    ):
        raise HarpV1WorkspaceAuthorityError(
            "HARP v1 workspace registration execution contract drifted."
        )

    artifacts = catalog.get("artifacts")
    if (
        isinstance(artifacts, (str, bytes))
        or not isinstance(artifacts, Sequence)
    ):
        raise HarpV1WorkspaceAuthorityError(
            "HARP v1 artifact catalog inventory is malformed."
        )
    output_id = experiment.get("output_artifact_id")
    output_matches = tuple(
        row
        for row in artifacts
        if isinstance(row, Mapping) and row.get("artifact_id") == output_id
    )
    if len(output_matches) != 1:
        raise HarpV1WorkspaceAuthorityError(
            "HARP v1 output artifact catalog binding is not unique."
        )
    output = output_matches[0]

    observed: dict[str, object] = {
        "schema_version": workspace_registration_execution_contract()[
            "schema_version"
        ],
        "experiment_id": experiment.get("experiment_id"),
        "stage": experiment.get("stage"),
        "status": experiment.get("status"),
        "claim_scope": experiment.get("claim_scope"),
        "config_path": experiment.get("config_path"),
        "output_artifact_id": output_id,
        "output_canonical_path": output.get("canonical_path"),
        "input_artifact_ids": list(input_ids),
        "preparation_authority_gate": runner.get("preparation_authority_gate"),
        "run_recovery_strategy": runner.get("run_recovery_strategy"),
        "runner_argv": runner.get("argv"),
        "runner_environment": runner.get("environment"),
    }
    expected = workspace_registration_execution_contract()
    observed["workspace_registration_execution_contract_hash"] = expected[
        "workspace_registration_execution_contract_hash"
    ]
    if canonical_bytes(observed) != canonical_bytes(expected):
        raise HarpV1WorkspaceAuthorityError(
            "HARP v1 workspace registration execution contract drifted."
        )
    return (
        str(expected["workspace_registration_execution_contract_hash"]),
        registry_path,
        sha256_file(registry_path),
        catalog_path,
        sha256_file(catalog_path),
    )


def _validate_registration_projection(
    projection: Mapping[str, object] | None,
) -> str:
    """Authenticate the frozen entry through the shared pure validator."""

    try:
        return validate_workspace_registration_execution_projection(projection)
    except ProtocolError as exc:
        raise HarpV1WorkspaceAuthorityError(str(exc)) from exc


def _safe_repository_config(repo_root: Path, config_path: str) -> Path:
    raw_path = Path(config_path)
    candidate = raw_path if raw_path.is_absolute() else repo_root / raw_path
    if candidate.is_symlink():
        raise HarpV1WorkspaceAuthorityError(
            "HARP v1 workspace config may not be a symlink."
        )
    resolved = candidate.resolve()
    if not resolved.is_relative_to(repo_root) or not resolved.is_file():
        raise HarpV1WorkspaceAuthorityError(
            "HARP v1 workspace config is absent or outside the repository."
        )
    return resolved


def _safe_repository_member(
    repo_root: Path,
    relative: str,
    *,
    label: str,
) -> Path:
    candidate = repo_root / relative
    if candidate.is_symlink():
        raise HarpV1WorkspaceAuthorityError(
            f"HARP v1 workspace {label} may not be a symlink."
        )
    resolved = candidate.resolve()
    if not resolved.is_relative_to(repo_root) or not resolved.is_file():
        raise HarpV1WorkspaceAuthorityError(
            f"HARP v1 workspace {label} is absent or outside the repository."
        )
    return resolved


def _read_yaml_file_mapping(path: Path, *, label: str) -> Mapping[str, object]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise HarpV1WorkspaceAuthorityError(
            f"HARP v1 workspace {label} is not readable YAML."
        ) from exc
    if not isinstance(raw, Mapping):
        raise HarpV1WorkspaceAuthorityError(
            f"HARP v1 workspace {label} must be a mapping."
        )
    return raw


def _read_yaml_mapping(path: Path) -> Mapping[str, object]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise HarpV1WorkspaceAuthorityError(
            "HARP v1 workspace config is not readable YAML."
        ) from exc
    if not isinstance(raw, Mapping):
        raise HarpV1WorkspaceAuthorityError(
            "HARP v1 workspace config must be a mapping."
        )
    return raw


def _validate_unresolved_amendment_uri(raw: Mapping[str, object]) -> None:
    inputs = raw.get("inputs")
    expected = (
        f"artifact://{EXECUTION_AMENDMENT_ARTIFACT_ID}/"
        f"{EXECUTION_AMENDMENT_FILENAME}"
    )
    if not isinstance(inputs, Mapping) or inputs.get(
        "execution_amendment_path"
    ) != expected:
        raise HarpV1WorkspaceAuthorityError(
            "HARP v1 checked-in config must bind the exact amendment URI."
        )


__all__ = (
    "HarpV1WorkspaceAuthorityError",
    "HarpV1WorkspaceAuthorityReceipt",
    "validate_workspace_preparation_authority",
)
