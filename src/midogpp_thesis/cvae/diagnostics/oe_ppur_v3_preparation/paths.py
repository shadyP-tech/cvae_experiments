"""Canonical workspace paths for OE-PPUR v3 preparation and launch.

This module is intentionally outside the sealed scientific adapter.  It may
resolve catalog paths, but it cannot participate in source supervision or in
the label-free routing computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...protocol import ProtocolError
from ....workspace import MidogppWorkspace, WorkspaceError
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.execution.inputs import (
    ResolvedDirectInput,
    validate_exact_resolved_input_bindings,
)
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.identity import (
    AUTHORIZATION_AMENDMENT_FILENAME,
    DIRECT_INPUT_ARTIFACT_IDS,
    DIRECT_INPUT_ROLES,
    EXPECTED_INPUT_KINDS,
    INPUT_RELATIVE_MEMBERS,
    OUTPUT_ARTIFACT_ID,
)
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.lease_claim import (
    LEASE_DIRECTORY_NAME,
)
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.run_paths import (
    assert_no_symlink_chain,
    paths_overlap,
    validate_absolute_path,
)
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.workspace_binding import (
    assert_canonical_output_root,
)


DEFAULT_SCRATCH_ROOT = Path(
    "/data/local/fixed_bank_p_anchored_opportunity_equivalence_"
    "pairwise_primitive_utility_router_v3"
)


@dataclass(frozen=True, slots=True)
class CanonicalPreparationPaths:
    repository_root: Path
    artifact_root: Path
    scratch_root: Path
    lease_root: Path
    amendment_root: Path
    input_bindings: tuple[ResolvedDirectInput, ...]

    def __post_init__(self) -> None:
        repository = validate_absolute_path(
            self.repository_root, role="preparation repository root"
        )
        artifact = assert_canonical_output_root(self.artifact_root)
        scratch = validate_absolute_path(
            self.scratch_root, role="preparation scratch root"
        )
        amendment = validate_absolute_path(
            self.amendment_root, role="preparation amendment root"
        )
        bindings = validate_exact_resolved_input_bindings(self.input_bindings)
        lease = artifact.parent / LEASE_DIRECTORY_NAME
        if (
            self.lease_root != lease
            or repository.is_symlink()
            or not repository.is_dir()
            or paths_overlap(artifact, scratch)
            or paths_overlap(lease, scratch)
            or paths_overlap(amendment, artifact)
            or paths_overlap(amendment, scratch)
        ):
            raise ProtocolError("OE-PPUR v3 preparation path topology drifted.")
        object.__setattr__(self, "repository_root", repository)
        object.__setattr__(self, "artifact_root", artifact)
        object.__setattr__(self, "scratch_root", scratch)
        object.__setattr__(self, "lease_root", lease)
        object.__setattr__(self, "amendment_root", amendment)
        object.__setattr__(self, "input_bindings", bindings)

    @property
    def amendment_path(self) -> Path:
        return self.input_bindings[6].path


def resolve_canonical_preparation_paths(
    repository_root: str | Path,
    *,
    scratch_root: str | Path = DEFAULT_SCRATCH_ROOT,
    require_source: bool = True,
    require_amendment: bool = False,
) -> CanonicalPreparationPaths:
    """Resolve the seven direct inputs without using generic run preparation."""

    try:
        workspace = MidogppWorkspace.load(repository_root)
        workspace.validate()
        repository = workspace.repo_root
        artifact_root = workspace.resolve_artifact(
            OUTPUT_ARTIFACT_ID,
            for_output=True,
            require_exists=False,
        )
        roots = tuple(
            workspace.resolve_artifact(
                artifact_id,
                require_exists=(
                    True
                    if ordinal not in {3, 7}
                    else require_source
                    if ordinal == 3
                    else require_amendment
                ),
            )
            for ordinal, artifact_id in enumerate(
                DIRECT_INPUT_ARTIFACT_IDS, start=1
            )
        )
    except WorkspaceError as exc:
        raise ProtocolError("OE-PPUR v3 canonical workspace resolution failed.") from exc

    locations = tuple(
        root / relative if relative else root
        for root, relative in zip(roots, INPUT_RELATIVE_MEMBERS, strict=True)
    )
    amendment_root = roots[6]
    if locations[6].name != AUTHORIZATION_AMENDMENT_FILENAME:
        raise ProtocolError("OE-PPUR v3 canonical amendment filename drifted.")
    bindings = tuple(
        ResolvedDirectInput(role, artifact_id, kind, path)
        for role, artifact_id, kind, path in zip(
            DIRECT_INPUT_ROLES,
            DIRECT_INPUT_ARTIFACT_IDS,
            EXPECTED_INPUT_KINDS,
            locations,
            strict=True,
        )
    )
    scratch = validate_absolute_path(scratch_root, role="preparation scratch root")
    for value in (repository, artifact_root.parent, amendment_root.parent):
        assert_no_symlink_chain(value)
    assert_no_symlink_chain(artifact_root, allow_missing_leaf=True)
    assert_no_symlink_chain(scratch, allow_missing_leaf=True)
    assert_no_symlink_chain(amendment_root, allow_missing_leaf=True)
    lease = artifact_root.parent / LEASE_DIRECTORY_NAME
    assert_no_symlink_chain(lease, allow_missing_leaf=True)
    return CanonicalPreparationPaths(
        repository_root=repository,
        artifact_root=artifact_root,
        scratch_root=scratch,
        lease_root=lease,
        amendment_root=amendment_root,
        input_bindings=bindings,
    )


__all__ = (
    "CanonicalPreparationPaths",
    "DEFAULT_SCRATCH_ROOT",
    "resolve_canonical_preparation_paths",
)
