"""Closed-world authority gates that run before workspace input rendering.

The registry may select one of the identifiers in this module.  Registry data
never supplies a Python import path, callable, module name, or file path.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


SCEPTRE_V4_EXECUTION_AMENDMENT_GATE = (
    "sceptre_v4_consumed_test_execution_amendment_v1"
)
SCEPTRE_V4_EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_sceptre_router.v4"
)
KNOWN_PREPARATION_AUTHORITY_GATES = frozenset(
    {SCEPTRE_V4_EXECUTION_AMENDMENT_GATE}
)


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

    required = (
        SCEPTRE_V4_EXECUTION_AMENDMENT_GATE
        if experiment_id == SCEPTRE_V4_EXPERIMENT_ID
        else None
    )
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
    if gate_id == SCEPTRE_V4_EXECUTION_AMENDMENT_GATE:
        from ..cvae.diagnostics.fixed_bank_sceptre_router_v4.execution import (
            workspace_preparation_authority as sceptre_v4_authority,
        )

        try:
            receipt = sceptre_v4_authority.validate_workspace_preparation_authority(
                repo_root=repo_root,
                experiment_id=experiment_id,
                config_path=config_path,
                input_artifact_ids=tuple(input_artifact_ids),
                resolve_authority_member=resolve_authority_member,
            )
        except sceptre_v4_authority.SceptreV4WorkspaceAuthorityError as exc:
            raise PreparationAuthorityError(str(exc)) from exc
        return PreparationAuthorityReceipt(
            gate_id=gate_id,
            experiment_id=experiment_id,
            config_path=receipt.config_path,
            config_sha256=receipt.config_sha256,
            authority_path=receipt.amendment_path,
            authority_sha256=receipt.amendment_sha256,
        )
    raise PreparationAuthorityError(
        f"unknown runner.preparation_authority_gate {gate_id!r}"
    )


__all__ = (
    "AuthorityMember",
    "KNOWN_PREPARATION_AUTHORITY_GATES",
    "PreparationAuthorityError",
    "PreparationAuthorityReceipt",
    "SCEPTRE_V4_EXECUTION_AMENDMENT_GATE",
    "enforce_preparation_authority",
    "preparation_authority_registration_error",
    "validate_preparation_authority_gate_id",
)
