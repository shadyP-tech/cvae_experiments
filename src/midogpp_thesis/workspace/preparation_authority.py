"""Closed-world authority gates that run before workspace input rendering.

The registry may select one of the identifiers in this module.  Registry data
never supplies a Python import path, callable, module name, or file path.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path


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
KNOWN_PREPARATION_AUTHORITY_GATES = frozenset(
    {
        SCEPTRE_V4_EXECUTION_AMENDMENT_GATE,
        SCEPTRE_V5_EXECUTION_AMENDMENT_GATE,
    }
)
_AUTHORITY_MODULE_BY_GATE = {
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


AuthorityMemberResolver = Callable[[str, str], AuthorityMember]


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
        receipt = authority.validate_workspace_preparation_authority(
            repo_root=repo_root,
            experiment_id=experiment_id,
            config_path=config_path,
            input_artifact_ids=tuple(input_artifact_ids),
            resolve_authority_member=resolve_authority_member,
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
    )


__all__ = (
    "AuthorityMember",
    "KNOWN_PREPARATION_AUTHORITY_GATES",
    "PreparationAuthorityError",
    "PreparationAuthorityReceipt",
    "SCEPTRE_V4_EXECUTION_AMENDMENT_GATE",
    "SCEPTRE_V5_EXECUTION_AMENDMENT_GATE",
    "enforce_preparation_authority",
    "preparation_authority_registration_error",
    "validate_preparation_authority_gate_id",
)
