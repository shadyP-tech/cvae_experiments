"""Pre-render workspace authority gate for optimized HARP v16."""

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


class HarpV16WorkspaceAuthorityError(ValueError):
    """Raised before protected HARP v16 paths are rendered."""


@dataclass(frozen=True, slots=True)
class HarpV16WorkspaceAuthorityReceipt:
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
) -> HarpV16WorkspaceAuthorityReceipt:
    if (
        experiment_id != EXPERIMENT_ID
        or tuple(input_artifact_ids) != INPUT_ARTIFACT_IDS
        or config_path != WORKSPACE_CONFIG_RELATIVE_PATH
    ):
        raise HarpV16WorkspaceAuthorityError("HARP v16 workspace authority gate binding drifted.")
    try:
        validate_workspace_registration_execution_projection(registration_projection)
    except ProtocolError as exc:
        raise HarpV16WorkspaceAuthorityError(str(exc)) from exc
    repository = Path(repo_root).resolve()
    config_file = _safe_repository_member(repository, config_path, label="config")
    raw_config = _read_yaml_mapping(config_file, label="config")
    expected_uri = f"artifact://{EXECUTION_AMENDMENT_ARTIFACT_ID}/{EXECUTION_AMENDMENT_FILENAME}"
    inputs = raw_config.get("inputs")
    if not isinstance(inputs, Mapping) or inputs.get("execution_amendment_path") != expected_uri:
        raise HarpV16WorkspaceAuthorityError("HARP v16 config must bind the exact amendment URI.")
    try:
        config = load_config(config_file)
    except ProtocolError as exc:
        raise HarpV16WorkspaceAuthorityError("HARP v16 checked-in execution config is not authentic.") from exc
    if not config.execution_authorized:
        raise HarpV16WorkspaceAuthorityError("HARP v16 checked-in config is not execution-authorized.")

    registration_hash, registry_path, catalog_path = _validate_workspace_registration(repository)
    lease = lease_path(repository)
    if lease.is_symlink():
        raise HarpV16WorkspaceAuthorityError("HARP v16 authorization lease is unsafe.")
    if lease.parent.is_symlink() or (lease.parent.exists() and not lease.parent.is_dir()):
        raise HarpV16WorkspaceAuthorityError("HARP v16 authorization lease parent is unsafe.")

    member = resolve_authority_member(EXECUTION_AMENDMENT_ARTIFACT_ID, EXECUTION_AMENDMENT_FILENAME)
    amendment_path = Path(member.path)
    expected_sha256 = config.expected_execution_amendment_sha256
    if type(expected_sha256) is not str or member.expected_sha256 != expected_sha256:
        raise HarpV16WorkspaceAuthorityError("HARP v16 catalog and config amendment hashes disagree.")
    if amendment_path.is_symlink() or not amendment_path.is_file():
        raise HarpV16WorkspaceAuthorityError("HARP v16 amendment bytes are absent or unsafe.")
    resolved = amendment_path.resolve()
    if not resolved.is_relative_to(repository) or sha256_file(resolved) != expected_sha256:
        raise HarpV16WorkspaceAuthorityError("HARP v16 amendment bytes are outside the repository or drifted.")
    try:
        amendment = json.loads(resolved.read_text(encoding="utf-8"))
        validated_amendment = validate_execution_amendment_payload(
            amendment, config, repo_root=repository
        )
        if lease.exists():
            from .authorization import validate_active_recovery_surface

            validate_active_recovery_surface(
                config,
                validated_amendment,
                repo_root=repository,
            )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ProtocolError) as exc:
        raise HarpV16WorkspaceAuthorityError("HARP v16 consumer-specific amendment drifted.") from exc
    return HarpV16WorkspaceAuthorityReceipt(
        config_path=config_file, config_sha256=sha256_file(config_file),
        amendment_path=resolved, amendment_sha256=expected_sha256,
        workspace_registration_contract_hash=registration_hash,
        registry_path=registry_path, registry_sha256=sha256_file(registry_path),
        artifact_catalog_path=catalog_path, artifact_catalog_sha256=sha256_file(catalog_path),
    )


def _validate_workspace_registration(repo_root: Path) -> tuple[str, Path, Path]:
    registry_path = _safe_repository_member(
        repo_root, WORKSPACE_REGISTRY_RELATIVE_PATH, label="registry"
    )
    catalog_path = _safe_repository_member(
        repo_root, WORKSPACE_ARTIFACT_CATALOG_RELATIVE_PATH, label="artifact catalog"
    )
    registry = _read_yaml_mapping(registry_path, label="registry")
    catalog = _read_yaml_mapping(catalog_path, label="artifact catalog")
    experiments = registry.get("experiments")
    if isinstance(experiments, (str, bytes)) or not isinstance(experiments, Sequence):
        raise HarpV16WorkspaceAuthorityError("HARP v16 registry inventory is malformed.")
    matches = tuple(
        row for row in experiments
        if isinstance(row, Mapping) and row.get("experiment_id") == EXPERIMENT_ID
    )
    if len(matches) != 1:
        raise HarpV16WorkspaceAuthorityError("HARP v16 registry entry is not unique.")
    experiment = matches[0]
    runner = experiment.get("runner")
    input_ids = experiment.get("input_artifact_ids")
    if (
        not isinstance(runner, Mapping)
        or set(runner) != {"preparation_authority_gate", "environment", "argv"}
        or isinstance(input_ids, (str, bytes))
        or not isinstance(input_ids, Sequence)
    ):
        raise HarpV16WorkspaceAuthorityError("HARP v16 registration execution contract drifted.")
    artifacts = catalog.get("artifacts")
    if isinstance(artifacts, (str, bytes)) or not isinstance(artifacts, Sequence):
        raise HarpV16WorkspaceAuthorityError("HARP v16 catalog inventory is malformed.")
    output_id = experiment.get("output_artifact_id")
    outputs = tuple(
        row for row in artifacts
        if isinstance(row, Mapping) and row.get("artifact_id") == output_id
    )
    if len(outputs) != 1:
        raise HarpV16WorkspaceAuthorityError("HARP v16 output binding is not unique.")
    expected = workspace_registration_execution_contract()
    observed: dict[str, object] = {
        "schema_version": expected["schema_version"],
        "experiment_id": experiment.get("experiment_id"), "stage": experiment.get("stage"),
        "status": experiment.get("status"), "claim_scope": experiment.get("claim_scope"),
        "config_path": experiment.get("config_path"), "output_artifact_id": output_id,
        "output_canonical_path": outputs[0].get("canonical_path"),
        "input_artifact_ids": list(input_ids),
        "preparation_authority_gate": runner.get("preparation_authority_gate"),
        "run_recovery_strategy": runner.get("run_recovery_strategy"),
        "runner_argv": runner.get("argv"), "runner_environment": runner.get("environment"),
        "workspace_registration_execution_contract_hash": expected[
            "workspace_registration_execution_contract_hash"
        ],
    }
    if canonical_bytes(observed) != canonical_bytes(expected):
        raise HarpV16WorkspaceAuthorityError("HARP v16 workspace registration execution contract drifted.")
    return str(expected["workspace_registration_execution_contract_hash"]), registry_path, catalog_path


def _safe_repository_member(repo_root: Path, relative: str, *, label: str) -> Path:
    candidate = repo_root / relative
    if candidate.is_symlink():
        raise HarpV16WorkspaceAuthorityError(f"HARP v16 {label} may not be a symlink.")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(repo_root) or not resolved.is_file():
        raise HarpV16WorkspaceAuthorityError(f"HARP v16 {label} is absent or outside repository.")
    return resolved


def _read_yaml_mapping(path: Path, *, label: str) -> Mapping[str, object]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise HarpV16WorkspaceAuthorityError(f"HARP v16 {label} is not readable YAML.") from exc
    if not isinstance(raw, Mapping):
        raise HarpV16WorkspaceAuthorityError(f"HARP v16 {label} must be a mapping.")
    return raw


__all__ = (
    "HarpV16WorkspaceAuthorityError", "HarpV16WorkspaceAuthorityReceipt",
    "validate_workspace_preparation_authority",
)
