"""Catalog-only path resolution for HARP v13 preparation and activation.

The preparation command accepts a repository root, never user-selected input or
output paths.  Every scientific input and every publication destination is
projected from the canonical :class:`MidogppWorkspace` catalog and then checked
lexically before resolution through :class:`RepositoryBoundary`.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from ...protocol import ProtocolError
from ....workspace.runtime import ArtifactEntry, MidogppWorkspace, WorkspaceError
from .activation_paths import RepositoryBoundary
from .authorization import (
    EXECUTION_AMENDMENT_FILENAME,
    WORKSPACE_CONFIG_RELATIVE_PATH,
)
from .config import INPUT_ARTIFACT_IDS
from .identity import EXPERIMENT_ID, OUTPUT_ARTIFACT_ID


CANONICAL_TRAIN_CACHE_ARTIFACT_ID = (
    "midogpp_virchow2_uniform_b_canonical_train_cache_seed42"
)
CANONICAL_TEST_CACHE_ARTIFACT_ID = (
    "midogpp_virchow2_uniform_b_v2_descriptive_test_cache_seed42"
)
# Compatibility alias for lifecycle helpers that still call the target-test
# input the canonical cache.  New code must bind both explicit roots below.
CANONICAL_CACHE_ARTIFACT_ID = CANONICAL_TEST_CACHE_ARTIFACT_ID
CANONICAL_MANIFEST_ARTIFACT_ID = "midogpp_dataset_contract_annotation_patch_v1"
CANONICAL_MANIFEST_MEMBER = "manifest.csv"
PARENT_LEDGER_MEMBER = "reports/test_consumption_ledger.json"
DEVELOPMENT_MANIFEST_MEMBER = "index.json"
EVALUATION_RELEASE_MEMBER = "release.json"

# Durable output members for the source-train H/q/r cross-fit firewall.  The
# label-capability member is one closed manifest containing 72 individually
# hashed (outer H, held-out q) seals, rather than an unbounded path convention.
SOURCE_CROSSFIT_PHYSICAL_SURFACE_STORE = (
    "stores/source_crossfit_physical_surface"
)
SOURCE_CROSSFIT_PHYSICAL_SURFACE_SEAL_MEMBER = (
    "manifests/source_crossfit_physical_surface_seal.json"
)
SOURCE_CROSSFIT_EFFECTIVE_MENU_STORE = "stores/source_crossfit_effective_menu"
SOURCE_CROSSFIT_EFFECTIVE_MENU_SEAL_MEMBER = (
    "manifests/source_crossfit_effective_menu_seal.json"
)
SOURCE_FOLD_LABEL_CAPABILITY_SEALS_MEMBER = (
    "manifests/source_fold_label_capability_seals.json"
)
SOURCE_PRELABEL_Q_PREDICTIONS_STORE = "stores/source_prelabel_q_predictions"
SOURCE_PRELABEL_Q_PREDICTION_SEAL_MEMBER = (
    "manifests/source_prelabel_q_prediction_seal.json"
)
SOURCE_CROSSFIT_REQUIRED_OUTPUT_MEMBERS = (
    SOURCE_CROSSFIT_PHYSICAL_SURFACE_SEAL_MEMBER,
    SOURCE_CROSSFIT_EFFECTIVE_MENU_SEAL_MEMBER,
    SOURCE_FOLD_LABEL_CAPABILITY_SEALS_MEMBER,
    SOURCE_PRELABEL_Q_PREDICTION_SEAL_MEMBER,
    f"{SOURCE_CROSSFIT_PHYSICAL_SURFACE_STORE}/manifest.json",
    f"{SOURCE_CROSSFIT_PHYSICAL_SURFACE_STORE}/probabilities.npy",
    f"{SOURCE_CROSSFIT_PHYSICAL_SURFACE_STORE}/seed_dispersion.npy",
    f"{SOURCE_CROSSFIT_PHYSICAL_SURFACE_STORE}/compatibility.npy",
    f"{SOURCE_CROSSFIT_EFFECTIVE_MENU_STORE}/manifest.json",
    f"{SOURCE_CROSSFIT_EFFECTIVE_MENU_STORE}/arrays.npz",
    f"{SOURCE_PRELABEL_Q_PREDICTIONS_STORE}/manifest.json",
    f"{SOURCE_PRELABEL_Q_PREDICTIONS_STORE}/arrays.npz",
)

_EXPECTED_LOCATIONS = {
    INPUT_ARTIFACT_IDS[0]: (
        "canonical_path",
        "artifacts/midogpp/30_expert_bank/"
        "uniform_b_v2_routing_authorized_expert_bank_v1",
    ),
    INPUT_ARTIFACT_IDS[1]: (
        "canonical_path",
        "artifacts/midogpp/40_prior_and_generation/"
        "uniform_b_v2_generation_lock/v1",
    ),
    CANONICAL_TRAIN_CACHE_ARTIFACT_ID: (
        "canonical_path",
        "datasets/midogpp/derived/features/virchow2/"
        "uniform_b_canonical_reference_v1/seed42",
    ),
    CANONICAL_TEST_CACHE_ARTIFACT_ID: (
        "canonical_path",
        "datasets/midogpp/derived/features/virchow2/"
        "uniform_b_v2_descriptive_test_cache_v1/seed42",
    ),
    CANONICAL_MANIFEST_ARTIFACT_ID: (
        "physical_path",
        "datasets/midogpp/contract/annotation_patch_v1",
    ),
    INPUT_ARTIFACT_IDS[2]: (
        "canonical_path",
        "datasets/midogpp/derived/features/virchow2/"
        "harp_source_train_full_test_cache_v13",
    ),
    INPUT_ARTIFACT_IDS[3]: (
        "canonical_path",
        "datasets/midogpp/contract/harp_source_train_label_capability_v13",
    ),
    INPUT_ARTIFACT_IDS[4]: (
        "canonical_path",
        "datasets/midogpp/contract/harp_full_test_evaluation_release_v13",
    ),
    INPUT_ARTIFACT_IDS[5]: (
        "physical_path",
        "artifacts/midogpp/10_real_feature_reference/"
        "uniform_b_canonical_real_feature_reference_v1/seed42",
    ),
    INPUT_ARTIFACT_IDS[6]: (
        "physical_path",
        "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts/"
        "harp_router_v13",
    ),
    OUTPUT_ARTIFACT_ID: (
        "canonical_path",
        "artifacts/midogpp/90_oracles_and_diagnostics/"
        "uniform_b_v2_consumed_test_fixed_bank_harp_router/v13",
    ),
}


@dataclass(frozen=True, slots=True)
class HarpV13WorkspacePaths:
    """The only paths admitted by the HARP v13 workstation lifecycle."""

    repository_root: Path
    config_path: Path
    registry_path: Path
    catalog_path: Path
    expert_bank_root: Path
    generation_lock_root: Path
    canonical_train_cache_root: Path
    canonical_test_cache_root: Path
    canonical_manifest_path: Path
    parent_ledger_path: Path
    prepared_cache_root: Path
    development_manifest_path: Path
    evaluation_manifest_path: Path
    amendment_path: Path
    output_root: Path
    staging_root: Path
    transaction_path: Path
    lock_path: Path

    @property
    def canonical_cache_root(self) -> Path:
        """Legacy descriptive alias for the canonical target-test cache."""

        return self.canonical_test_cache_root

    def activation_kwargs(self) -> dict[str, Path]:
        """Return the exact prepared inputs accepted by activation planning."""

        return {
            "expert_bank_root": self.expert_bank_root,
            "generation_lock_root": self.generation_lock_root,
            "prepared_cache_root": self.prepared_cache_root,
            "development_manifest_path": self.development_manifest_path,
            "evaluation_manifest_path": self.evaluation_manifest_path,
            "parent_ledger_path": self.parent_ledger_path,
        }


def resolve_harp_v13_workspace_paths(
    repository_root: str | Path,
    require_prepared: bool,
) -> HarpV13WorkspacePaths:
    """Resolve the closed HARP v13 path set from the workspace catalog.

    Resolving the canonical scoring manifest performs pathname checks only.  It
    deliberately does not open, hash, parse, or inventory that file.
    """

    boundary = RepositoryBoundary.open(repository_root)
    try:
        workspace = MidogppWorkspace.load(boundary.lexical_root)
        workspace.validate()
    except WorkspaceError as exc:
        raise ProtocolError("HARP v13 workspace catalog is unavailable.") from exc
    if workspace.repo_root != boundary.resolved_root:
        raise ProtocolError("HARP v13 workspace repository identity drifted.")
    experiment = workspace.experiments.get(EXPERIMENT_ID)
    if (
        experiment is None
        or experiment.input_artifact_ids != INPUT_ARTIFACT_IDS
        or experiment.output_artifact_id != OUTPUT_ARTIFACT_ID
    ):
        raise ProtocolError("HARP v13 workspace registration drifted.")
    cache_entry = _entry(workspace, INPUT_ARTIFACT_IDS[2])
    if (
        cache_entry.semantic_identities.get("source_train_artifact_id")
        != CANONICAL_TRAIN_CACHE_ARTIFACT_ID
        or cache_entry.semantic_identities.get("target_test_artifact_id")
        != CANONICAL_TEST_CACHE_ARTIFACT_ID
    ):
        raise ProtocolError("HARP v13 composite cache lineage drifted.")

    registry = boundary.member(
        "experiments/midogpp/registry.yaml", label="registry", kind="file"
    )
    config = boundary.member(
        WORKSPACE_CONFIG_RELATIVE_PATH, label="registered config", kind="file"
    )
    catalog = boundary.member(
        "experiments/midogpp/artifact_catalog.yaml",
        label="artifact catalog",
        kind="file",
    )
    expert_bank = _catalog_root(
        workspace, boundary, INPUT_ARTIFACT_IDS[0], label="expert bank", kind="directory"
    )
    generation = _catalog_root(
        workspace,
        boundary,
        INPUT_ARTIFACT_IDS[1],
        label="generation lock",
        kind="directory",
    )
    canonical_train_cache = _catalog_root(
        workspace,
        boundary,
        CANONICAL_TRAIN_CACHE_ARTIFACT_ID,
        label="canonical source-train cache",
        kind="directory",
    )
    canonical_test_cache = _catalog_root(
        workspace,
        boundary,
        CANONICAL_TEST_CACHE_ARTIFACT_ID,
        label="canonical target-test cache",
        kind="directory",
    )
    canonical_manifest_root = _catalog_root(
        workspace,
        boundary,
        CANONICAL_MANIFEST_ARTIFACT_ID,
        label="canonical scoring contract",
        kind="directory",
    )
    canonical_manifest = boundary.path(
        canonical_manifest_root / CANONICAL_MANIFEST_MEMBER,
        label="canonical scoring manifest",
        kind="file",
    )
    parent_root = _catalog_root(
        workspace,
        boundary,
        INPUT_ARTIFACT_IDS[5],
        label="parent ledger artifact",
        kind="directory",
    )
    parent = boundary.path(
        parent_root / PARENT_LEDGER_MEMBER, label="parent ledger", kind="file"
    )

    prepared_cache = _catalog_destination(
        workspace,
        boundary,
        INPUT_ARTIFACT_IDS[2],
        label="prepared cache",
        require_prepared=require_prepared,
    )
    development_root = _catalog_destination(
        workspace,
        boundary,
        INPUT_ARTIFACT_IDS[3],
        label="development capability",
        require_prepared=require_prepared,
    )
    evaluation_root = _catalog_destination(
        workspace,
        boundary,
        INPUT_ARTIFACT_IDS[4],
        label="evaluation capability",
        require_prepared=require_prepared,
    )
    development = _destination_member(
        boundary,
        development_root,
        DEVELOPMENT_MANIFEST_MEMBER,
        label="development manifest",
        require_prepared=require_prepared,
    )
    evaluation = _destination_member(
        boundary,
        evaluation_root,
        EVALUATION_RELEASE_MEMBER,
        label="evaluation release descriptor",
        require_prepared=require_prepared,
    )

    amendment_root = _catalog_root(
        workspace,
        boundary,
        INPUT_ARTIFACT_IDS[6],
        label="amendment artifact",
        kind="directory",
    )
    amendment = boundary.path(
        amendment_root / EXECUTION_AMENDMENT_FILENAME,
        label="execution amendment",
        kind="optional",
    )
    output = _catalog_root(
        workspace,
        boundary,
        OUTPUT_ARTIFACT_ID,
        label="experiment output",
        kind="future",
    )

    # Internal preparation state is fixed adjacent to the catalog-owned cache;
    # callers cannot redirect it to another disk or a predecessor tree.
    state_parent = prepared_cache.parent
    staging = boundary.path(
        state_parent / ".harp_source_train_full_test_cache_v13.preparing",
        label="preparation staging root",
        kind=("directory" if os.path.lexists(state_parent / ".harp_source_train_full_test_cache_v13.preparing") else "future"),
    )
    transaction = boundary.path(
        state_parent / ".harp_source_train_full_test_cache_v13.transaction.json",
        label="preparation transaction",
        kind=("file" if os.path.lexists(state_parent / ".harp_source_train_full_test_cache_v13.transaction.json") else "future"),
    )
    lock = boundary.path(
        state_parent / ".harp_source_train_full_test_cache_v13.lock",
        label="preparation lock",
        kind=("file" if os.path.lexists(state_parent / ".harp_source_train_full_test_cache_v13.lock") else "future"),
    )
    paths = HarpV13WorkspacePaths(
        repository_root=boundary.resolved_root,
        config_path=config,
        registry_path=registry,
        catalog_path=catalog,
        expert_bank_root=expert_bank,
        generation_lock_root=generation,
        canonical_train_cache_root=canonical_train_cache,
        canonical_test_cache_root=canonical_test_cache,
        canonical_manifest_path=canonical_manifest,
        parent_ledger_path=parent,
        prepared_cache_root=prepared_cache,
        development_manifest_path=development,
        evaluation_manifest_path=evaluation,
        amendment_path=amendment,
        output_root=output,
        staging_root=staging,
        transaction_path=transaction,
        lock_path=lock,
    )
    _reject_overlaps(paths)
    return paths


def _entry(workspace: MidogppWorkspace, artifact_id: str) -> ArtifactEntry:
    try:
        return workspace.artifacts[artifact_id]
    except KeyError as exc:
        raise ProtocolError(f"HARP v13 catalog lacks {artifact_id}.") from exc


def _catalog_location(workspace: MidogppWorkspace, artifact_id: str) -> str:
    entry = _entry(workspace, artifact_id)
    try:
        field, expected = _EXPECTED_LOCATIONS[artifact_id]
    except KeyError as exc:  # pragma: no cover - closed module table
        raise ProtocolError("HARP v13 catalog resolver received an unknown identity.") from exc
    actual = getattr(entry, field)
    alternate = entry.physical_path if field == "canonical_path" else entry.canonical_path
    if actual != expected or alternate is not None:
        raise ProtocolError(f"HARP v13 catalog path drifted for {artifact_id}.")
    return expected


def _catalog_root(
    workspace: MidogppWorkspace,
    boundary: RepositoryBoundary,
    artifact_id: str,
    *,
    label: str,
    kind: str,
) -> Path:
    relative = _catalog_location(workspace, artifact_id)
    return boundary.member(relative, label=label, kind=kind)


def _catalog_destination(
    workspace: MidogppWorkspace,
    boundary: RepositoryBoundary,
    artifact_id: str,
    *,
    label: str,
    require_prepared: bool,
) -> Path:
    relative = _catalog_location(workspace, artifact_id)
    lexical = boundary.lexical_root / relative
    exists = os.path.lexists(lexical)
    if require_prepared and not exists:
        raise ProtocolError(f"HARP v13 {label} is not prepared.")
    return boundary.member(
        relative,
        label=label,
        kind=("directory" if exists else "future"),
    )


def _destination_member(
    boundary: RepositoryBoundary,
    root: Path,
    member: str,
    *,
    label: str,
    require_prepared: bool,
) -> Path:
    candidate = root / member
    exists = os.path.lexists(candidate)
    if require_prepared and not exists:
        raise ProtocolError(f"HARP v13 {label} is not prepared.")
    return boundary.path(
        candidate,
        label=label,
        kind=("file" if exists else "future"),
    )


def _reject_overlaps(paths: HarpV13WorkspacePaths) -> None:
    protected = (
        paths.expert_bank_root,
        paths.generation_lock_root,
        paths.canonical_train_cache_root,
        paths.canonical_test_cache_root,
        paths.canonical_manifest_path,
        paths.parent_ledger_path,
        paths.prepared_cache_root,
        paths.development_manifest_path.parent,
        paths.evaluation_manifest_path.parent,
        paths.amendment_path,
        paths.output_root,
        paths.staging_root,
        paths.transaction_path,
        paths.lock_path,
        paths.config_path,
    )
    if len(set(protected)) != len(protected):
        raise ProtocolError("HARP v13 catalog path identities overlap.")
    for destination in (
        paths.prepared_cache_root,
        paths.development_manifest_path.parent,
        paths.evaluation_manifest_path.parent,
        paths.staging_root,
    ):
        if any(
            destination == source or destination.is_relative_to(source)
            for source in (
                paths.expert_bank_root,
                paths.generation_lock_root,
                paths.canonical_train_cache_root,
                paths.canonical_test_cache_root,
                paths.canonical_manifest_path.parent,
                paths.parent_ledger_path.parent,
            )
        ):
            raise ProtocolError("HARP v13 preparation destination overlaps an input.")


__all__ = (
    "CANONICAL_CACHE_ARTIFACT_ID",
    "CANONICAL_TRAIN_CACHE_ARTIFACT_ID",
    "CANONICAL_TEST_CACHE_ARTIFACT_ID",
    "CANONICAL_MANIFEST_ARTIFACT_ID",
    "HarpV13WorkspacePaths",
    "SOURCE_CROSSFIT_EFFECTIVE_MENU_SEAL_MEMBER",
    "SOURCE_CROSSFIT_EFFECTIVE_MENU_STORE",
    "SOURCE_CROSSFIT_PHYSICAL_SURFACE_SEAL_MEMBER",
    "SOURCE_CROSSFIT_PHYSICAL_SURFACE_STORE",
    "SOURCE_CROSSFIT_REQUIRED_OUTPUT_MEMBERS",
    "SOURCE_FOLD_LABEL_CAPABILITY_SEALS_MEMBER",
    "SOURCE_PRELABEL_Q_PREDICTION_SEAL_MEMBER",
    "SOURCE_PRELABEL_Q_PREDICTIONS_STORE",
    "resolve_harp_v13_workspace_paths",
)
