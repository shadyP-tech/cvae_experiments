"""Closed-world authority gates that run before workspace input rendering.

The registry may select one of the identifiers in this module.  Registry data
never supplies a Python import path, callable, module name, or file path.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

from ..cvae.protocol import ProtocolError


SCEPTRE_V4_EXECUTION_AMENDMENT_GATE = (
    "sceptre_v4_consumed_test_execution_amendment_v1"
)
SCEPTRE_V4_EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_sceptre_router.v4"
)
SCEPTRE_V5_EXECUTION_AMENDMENT_GATE = (
    "sceptre_v5_consumed_test_execution_amendment_v1"
)
SCEPTRE_V5_EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_sceptre_router.v5"
)
HARP_V1_EXECUTION_AMENDMENT_GATE = (
    "harp_v1_consumed_test_execution_amendment_v1"
)
HARP_V1_EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_harp_router.v1"
)
HARP_V2_EXECUTION_AMENDMENT_GATE = (
    "harp_v2_consumed_test_execution_amendment_v1"
)
HARP_V2_EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_harp_router.v2"
)
HARP_V3_EXECUTION_AMENDMENT_GATE = (
    "harp_v3_consumed_test_execution_amendment_v1"
)
HARP_V3_EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_harp_router.v3"
)
KNOWN_PREPARATION_AUTHORITY_GATES = frozenset(
    {
        HARP_V1_EXECUTION_AMENDMENT_GATE,
        HARP_V2_EXECUTION_AMENDMENT_GATE,
        HARP_V3_EXECUTION_AMENDMENT_GATE,
        SCEPTRE_V4_EXECUTION_AMENDMENT_GATE,
        SCEPTRE_V5_EXECUTION_AMENDMENT_GATE,
    }
)
_AUTHORITY_MODULE_BY_GATE = {
    HARP_V1_EXECUTION_AMENDMENT_GATE: (
        "midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v1."
        "workspace_preparation_authority",
        "HarpV1WorkspaceAuthorityError",
    ),
    HARP_V2_EXECUTION_AMENDMENT_GATE: (
        "midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v2."
        "workspace_preparation_authority",
        "HarpV2WorkspaceAuthorityError",
    ),
    HARP_V3_EXECUTION_AMENDMENT_GATE: (
        "midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3."
        "workspace_preparation_authority",
        "HarpV3WorkspaceAuthorityError",
    ),
    SCEPTRE_V4_EXECUTION_AMENDMENT_GATE: (
        "midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v4."
        "execution.workspace_preparation_authority",
        "SceptreV4WorkspaceAuthorityError",
    ),
    SCEPTRE_V5_EXECUTION_AMENDMENT_GATE: (
        "midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v5."
        "execution.workspace_preparation_authority",
        "SceptreV5WorkspaceAuthorityError",
    ),
}
_HARP_VERSION_BY_GATE = {
    HARP_V1_EXECUTION_AMENDMENT_GATE: "v1",
    HARP_V2_EXECUTION_AMENDMENT_GATE: "v2",
    HARP_V3_EXECUTION_AMENDMENT_GATE: "v3",
}
_HARP_EXPERIMENT_BY_GATE = {
    HARP_V1_EXECUTION_AMENDMENT_GATE: HARP_V1_EXPERIMENT_ID,
    HARP_V2_EXECUTION_AMENDMENT_GATE: HARP_V2_EXPERIMENT_ID,
    HARP_V3_EXECUTION_AMENDMENT_GATE: HARP_V3_EXPERIMENT_ID,
}


class PreparationAuthorityError(ValueError):
    """Raised when a registered pre-render authority gate is not satisfied."""


@dataclass(frozen=True, slots=True)
class AuthorityMember:
    """One catalog-pinned authority file, and no scientific input surface."""

    path: Path
    expected_sha256: str


@dataclass(frozen=True, slots=True)
class PreparationAuthorityReceipt:
    """Immutable bytes binding carried across the pre-render call boundary."""

    gate_id: str
    experiment_id: str
    config_path: Path
    config_sha256: str
    authority_path: Path
    authority_sha256: str
    workspace_registration_contract_hash: str | None = None
    registry_path: Path | None = None
    registry_sha256: str | None = None
    artifact_catalog_path: Path | None = None
    artifact_catalog_sha256: str | None = None


AuthorityMemberResolver = Callable[[str, str], AuthorityMember]


def expected_workspace_registration_contract_hash(
    gate_id: str | None,
) -> str | None:
    """Reconstruct a consumer registration hash from a closed module table."""

    if gate_id == HARP_V1_EXECUTION_AMENDMENT_GATE:
        module_name = (
            "midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v1."
            "authorization"
        )
    elif gate_id == HARP_V2_EXECUTION_AMENDMENT_GATE:
        module_name = (
            "midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v2."
            "authorization"
        )
    elif gate_id == HARP_V3_EXECUTION_AMENDMENT_GATE:
        module_name = (
            "midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3."
            "authorization"
        )
    else:
        return None
    # Both names are source constants selected by exact gate equality; no
    # registry value is ever interpreted as a Python module or symbol.
    authority = import_module(module_name)
    contract = authority.workspace_registration_execution_contract()
    value = contract.get("workspace_registration_execution_contract_hash")
    if type(value) is not str:
        raise PreparationAuthorityError(
            f"HARP {_HARP_VERSION_BY_GATE[gate_id]} workspace registration "
            "contract hash is malformed."
        )
    return value


def validate_preparation_authority_extra_args(
    gate_id: str | None,
    extra_args: Sequence[str],
    *,
    force: bool = False,
) -> tuple[str, ...]:
    """Apply consumer-specific closed-world runner argument constraints."""

    normalized = tuple(extra_args)
    harp_version = _HARP_VERSION_BY_GATE.get(gate_id)
    if harp_version is not None:
        if force:
            raise PreparationAuthorityError(
                f"HARP {harp_version} workspace preparation and execution "
                "reject --force."
            )
        if normalized not in {(), ("--dry-run",)}:
            raise PreparationAuthorityError(
                f"HARP {harp_version} workspace execution accepts only no "
                "extra arguments or "
                "the exact '--dry-run' argument."
            )
    return normalized


def validate_preparation_authority_registration_projection(
    gate_id: str | None,
    registration_projection: Mapping[str, object] | None,
) -> str | None:
    """Validate a runnable consumer's frozen registration projection.

    Non-HARP consumers retain their existing behavior.  The HARP validator is
    imported from a closed module name and is the same pure function called by
    its pre-render authority gate.
    """

    if gate_id == HARP_V1_EXECUTION_AMENDMENT_GATE:
        module_name = (
            "midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v1."
            "authorization"
        )
    elif gate_id == HARP_V2_EXECUTION_AMENDMENT_GATE:
        module_name = (
            "midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v2."
            "authorization"
        )
    elif gate_id == HARP_V3_EXECUTION_AMENDMENT_GATE:
        module_name = (
            "midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3."
            "authorization"
        )
    else:
        return None
    authority = import_module(module_name)
    try:
        return authority.validate_workspace_registration_execution_projection(
            registration_projection
        )
    except ProtocolError as exc:
        raise PreparationAuthorityError(str(exc)) from exc


def validate_preparation_authority_gate_id(value: object) -> str | None:
    """Parse the optional registry value against the closed-world allow-list."""

    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise PreparationAuthorityError(
            "runner.preparation_authority_gate must be a non-empty string"
        )
    if value not in KNOWN_PREPARATION_AUTHORITY_GATES:
        raise PreparationAuthorityError(
            f"unknown runner.preparation_authority_gate {value!r}"
        )
    return value


def preparation_authority_registration_error(
    gate_id: str | None,
    *,
    experiment_id: str,
) -> str | None:
    """Return a closed-world registry-binding error, if any."""

    required = {
        HARP_V1_EXPERIMENT_ID: HARP_V1_EXECUTION_AMENDMENT_GATE,
        HARP_V2_EXPERIMENT_ID: HARP_V2_EXECUTION_AMENDMENT_GATE,
        HARP_V3_EXPERIMENT_ID: HARP_V3_EXECUTION_AMENDMENT_GATE,
        SCEPTRE_V4_EXPERIMENT_ID: SCEPTRE_V4_EXECUTION_AMENDMENT_GATE,
        SCEPTRE_V5_EXPERIMENT_ID: SCEPTRE_V5_EXECUTION_AMENDMENT_GATE,
    }.get(experiment_id)
    if required is not None and gate_id != required:
        return (
            f"{experiment_id}: runner.preparation_authority_gate must remain "
            f"{required!r}"
        )
    if gate_id == SCEPTRE_V4_EXECUTION_AMENDMENT_GATE and (
        experiment_id != SCEPTRE_V4_EXPERIMENT_ID
    ):
        return (
            f"{experiment_id}: runner.preparation_authority_gate "
            f"{gate_id!r} is bound only to {SCEPTRE_V4_EXPERIMENT_ID}"
        )
    harp_experiment_id = _HARP_EXPERIMENT_BY_GATE.get(gate_id)
    if harp_experiment_id is not None and experiment_id != harp_experiment_id:
        return (
            f"{experiment_id}: runner.preparation_authority_gate "
            f"{gate_id!r} is bound only to {harp_experiment_id}"
        )
    if gate_id == SCEPTRE_V5_EXECUTION_AMENDMENT_GATE and (
        experiment_id != SCEPTRE_V5_EXPERIMENT_ID
    ):
        return (
            f"{experiment_id}: runner.preparation_authority_gate "
            f"{gate_id!r} is bound only to {SCEPTRE_V5_EXPERIMENT_ID}"
        )
    return None


def enforce_preparation_authority(
    gate_id: str | None,
    *,
    repo_root: Path,
    experiment_id: str,
    config_path: str | None,
    input_artifact_ids: Sequence[str],
    registration_projection: Mapping[str, object] | None = None,
    resolve_authority_member: AuthorityMemberResolver,
) -> PreparationAuthorityReceipt | None:
    """Run the named gate without resolving any normal experiment input.

    Imports are intentionally selected by an exact branch.  This prevents a
    registry edit from turning workspace preparation into arbitrary code
    loading.
    """

    if gate_id is None:
        return None
    validate_preparation_authority_gate_id(gate_id)
    module_binding = _AUTHORITY_MODULE_BY_GATE.get(gate_id)
    if module_binding is None:  # defensive even though the allow-list ran above
        raise PreparationAuthorityError(
            f"unknown runner.preparation_authority_gate {gate_id!r}"
        )
    module_name, error_name = module_binding
    # The module name comes only from the closed table above. Registry content
    # can select an allow-listed gate but can never provide importable code.
    authority = import_module(module_name)
    authority_error = getattr(authority, error_name)
    try:
        call_kwargs = {
            "repo_root": repo_root,
            "experiment_id": experiment_id,
            "config_path": config_path,
            "input_artifact_ids": tuple(input_artifact_ids),
            "resolve_authority_member": resolve_authority_member,
        }
        if gate_id in _HARP_VERSION_BY_GATE:
            call_kwargs["registration_projection"] = registration_projection
        receipt = authority.validate_workspace_preparation_authority(
            **call_kwargs,
        )
    except authority_error as exc:
        raise PreparationAuthorityError(str(exc)) from exc
    return PreparationAuthorityReceipt(
        gate_id=gate_id,
        experiment_id=experiment_id,
        config_path=receipt.config_path,
        config_sha256=receipt.config_sha256,
        authority_path=receipt.amendment_path,
        authority_sha256=receipt.amendment_sha256,
        workspace_registration_contract_hash=getattr(
            receipt,
            "workspace_registration_contract_hash",
            None,
        ),
        registry_path=getattr(receipt, "registry_path", None),
        registry_sha256=getattr(receipt, "registry_sha256", None),
        artifact_catalog_path=getattr(receipt, "artifact_catalog_path", None),
        artifact_catalog_sha256=getattr(
            receipt,
            "artifact_catalog_sha256",
            None,
        ),
    )


__all__ = (
    "AuthorityMember",
    "HARP_V1_EXECUTION_AMENDMENT_GATE",
    "HARP_V1_EXPERIMENT_ID",
    "HARP_V2_EXECUTION_AMENDMENT_GATE",
    "HARP_V2_EXPERIMENT_ID",
    "HARP_V3_EXECUTION_AMENDMENT_GATE",
    "HARP_V3_EXPERIMENT_ID",
    "KNOWN_PREPARATION_AUTHORITY_GATES",
    "PreparationAuthorityError",
    "PreparationAuthorityReceipt",
    "SCEPTRE_V4_EXECUTION_AMENDMENT_GATE",
    "SCEPTRE_V5_EXECUTION_AMENDMENT_GATE",
    "enforce_preparation_authority",
    "expected_workspace_registration_contract_hash",
    "preparation_authority_registration_error",
    "validate_preparation_authority_gate_id",
    "validate_preparation_authority_extra_args",
    "validate_preparation_authority_registration_projection",
)
