"""Public catalog-bound preparation lifecycle for HARP v5 on the workstation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import os
from pathlib import Path
import shutil

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from ...runtime.artifact_io import read_json, sha256_file
from ....workspace.runtime import MidogppWorkspace, WorkspaceError
from . import authorization
from .config import INPUT_ARTIFACT_IDS
from .identity import EXPERIMENT_ID, PUBLICATION_STATUS, TERMINAL_DECISION
from .input_surfaces import (
    DEVELOPMENT_ROLE,
    EVALUATION_ROLE,
    _read_label_manifest,
)
from .preparation import (
    CANONICAL_CACHE_CONTENT_HASH,
    CANONICAL_CACHE_ROW_ORDER_HASH,
    CANONICAL_MANIFEST_SHA256,
    CANONICAL_PARENT_LEDGER_SHA256,
    PREPARATION_RECEIPT,
    HarpV5PreparedInputs,
    V5_PREPARATION_IDENTITY,
    _prepare_harp_consumed_test_inputs,
    _validate_final_prepared_cache,
    validate_canonical_label_blind_cache_identity,
)
from .preparation_transaction import (
    build_preparation_journal,
    commit_prepared_inputs,
    inspect_preparation_recovery,
    inventory_tree,
    recover_prepared_inputs,
    staging_destinations,
)
from .workspace_paths import (
    CANONICAL_CACHE_ARTIFACT_ID,
    CANONICAL_MANIFEST_ARTIFACT_ID,
    HarpV5WorkspacePaths,
    resolve_harp_v5_workspace_paths,
)


PREPARATION_CONFIRMATION = "PREPARE_HARP_V5_CONSUMED_TEST_INPUTS"
PREPARATION_PLAN_SCHEMA = "midogpp_harp_v5_workstation_preparation_plan_v1"
FaultInjector = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class HarpV5WorkstationPreparationPlan:
    paths: HarpV5WorkspacePaths
    canonical_cache_content_hash: str
    canonical_cache_row_order_hash: str
    expected_manifest_sha256: str
    expected_parent_ledger_sha256: str
    config_sha256: str
    registry_sha256: str
    catalog_sha256: str
    observed_state: str
    preparation_plan_hash: str

    def __post_init__(self) -> None:
        if self.preparation_plan_hash and canonical_hash(
            self._hash_payload()
        ) != self.preparation_plan_hash:
            raise ProtocolError("HARP v5 preparation plan identity drifted.")

    def _hash_payload(self) -> dict[str, object]:
        paths = self.paths
        return {
            "schema_version": PREPARATION_PLAN_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "repository_root": paths.repository_root.as_posix(),
            "canonical_source_artifact_id": CANONICAL_CACHE_ARTIFACT_ID,
            "canonical_manifest_artifact_id": CANONICAL_MANIFEST_ARTIFACT_ID,
            "canonical_cache_root": paths.canonical_cache_root.as_posix(),
            "canonical_manifest_path": paths.canonical_manifest_path.as_posix(),
            "expert_bank_root": paths.expert_bank_root.as_posix(),
            "generation_lock_root": paths.generation_lock_root.as_posix(),
            "parent_ledger_path": paths.parent_ledger_path.as_posix(),
            "prepared_cache_root": paths.prepared_cache_root.as_posix(),
            "development_manifest_path": paths.development_manifest_path.as_posix(),
            "evaluation_manifest_path": paths.evaluation_manifest_path.as_posix(),
            "amendment_path": paths.amendment_path.as_posix(),
            "output_root": paths.output_root.as_posix(),
            "canonical_cache_content_hash": self.canonical_cache_content_hash,
            "canonical_cache_row_order_hash": self.canonical_cache_row_order_hash,
            "expected_manifest_sha256": self.expected_manifest_sha256,
            "expected_parent_ledger_sha256": self.expected_parent_ledger_sha256,
            "config_sha256": self.config_sha256,
            "registry_sha256": self.registry_sha256,
            "catalog_sha256": self.catalog_sha256,
            "commit_order": ["cache", "development", "evaluation"],
            "publication_status": PUBLICATION_STATUS,
            "terminal_decision": TERMINAL_DECISION,
            "fresh_evidence": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {
            **self._hash_payload(),
            "preparation_plan_hash": self.preparation_plan_hash,
            "status": self.observed_state,
            "confirmation_required": PREPARATION_CONFIRMATION,
            "canonical_scoring_manifest_opened": False,
            "canonical_scoring_manifest_hashed": False,
            "filesystem_mutations": 0,
            "execution_amendment_created": False,
            "authorization_lease_claimed": False,
            "output_created": False,
            "execution_authorized": False,
            "may_feed_stage60_or_stage70": False,
            "may_feed_another_experiment": False,
        }


def plan_harp_v5_workstation_preparation(
    repository_root: str | Path,
) -> HarpV5WorkstationPreparationPlan:
    """Plan exact catalog outputs without opening scoring-manifest bytes."""

    paths = resolve_harp_v5_workspace_paths(repository_root, require_prepared=False)
    _require_preparation_only_surface(paths)
    try:
        workspace = MidogppWorkspace.load(paths.repository_root)
    except WorkspaceError as exc:
        raise ProtocolError("HARP v5 workspace cannot be loaded for preparation.") from exc
    experiment = workspace.experiments.get(EXPERIMENT_ID)
    if (
        experiment is None
        or experiment.status != "planned"
        or experiment.runnable
        or experiment.input_artifact_ids != INPUT_ARTIFACT_IDS
    ):
        raise ProtocolError("HARP v5 preparation requires the planned workspace state.")
    expected_manifest = _catalog_sha256(
        workspace,
        CANONICAL_MANIFEST_ARTIFACT_ID,
        "manifest.csv",
        label="canonical scoring manifest",
    )
    expected_parent = _catalog_sha256(
        workspace,
        INPUT_ARTIFACT_IDS[5],
        "reports/test_consumption_ledger.json",
        label="parent ledger",
    )
    cache_semantics = workspace.artifacts[
        CANONICAL_CACHE_ARTIFACT_ID
    ].semantic_identities
    content_hash = cache_semantics.get("content_hash")
    row_order_hash = cache_semantics.get("row_order_hash")
    if (
        content_hash != CANONICAL_CACHE_CONTENT_HASH
        or row_order_hash != CANONICAL_CACHE_ROW_ORDER_HASH
        or expected_manifest != CANONICAL_MANIFEST_SHA256
        or expected_parent != CANONICAL_PARENT_LEDGER_SHA256
    ):
        raise ProtocolError("HARP v5 catalog preparation identity drifted.")
    observed = _observed_state(paths)
    provisional = HarpV5WorkstationPreparationPlan(
        paths=paths,
        canonical_cache_content_hash=content_hash,
        canonical_cache_row_order_hash=row_order_hash,
        expected_manifest_sha256=expected_manifest,
        expected_parent_ledger_sha256=expected_parent,
        config_sha256=sha256_file(paths.config_path),
        registry_sha256=sha256_file(paths.registry_path),
        catalog_sha256=sha256_file(paths.catalog_path),
        observed_state=observed,
        preparation_plan_hash="",
    )
    return HarpV5WorkstationPreparationPlan(
        paths=provisional.paths,
        canonical_cache_content_hash=provisional.canonical_cache_content_hash,
        canonical_cache_row_order_hash=provisional.canonical_cache_row_order_hash,
        expected_manifest_sha256=provisional.expected_manifest_sha256,
        expected_parent_ledger_sha256=provisional.expected_parent_ledger_sha256,
        config_sha256=provisional.config_sha256,
        registry_sha256=provisional.registry_sha256,
        catalog_sha256=provisional.catalog_sha256,
        observed_state=provisional.observed_state,
        preparation_plan_hash=canonical_hash(provisional._hash_payload()),
    )


def prepare_harp_v5_workstation_inputs(
    plan: HarpV5WorkstationPreparationPlan,
    *,
    confirmation: str,
    _fault_injector: FaultInjector | None = None,
) -> HarpV5PreparedInputs:
    """Build in owned staging and publish only after an authenticated receipt."""

    if type(plan) is not HarpV5WorkstationPreparationPlan:
        raise ProtocolError("HARP v5 workstation preparation requires a typed plan.")
    if confirmation != PREPARATION_CONFIRMATION:
        raise ProtocolError("HARP v5 preparation confirmation is absent or drifted.")
    _verify_plan(plan)
    paths = plan.paths
    if os.path.lexists(paths.transaction_path):
        _validate_scientific_sources(plan)
        return recover_prepared_inputs(
            paths,
            expected_plan_hash=plan.preparation_plan_hash,
            fault_injector=_fault_injector,
        )
    state = _destination_state(paths)
    if state == (True, True, True):
        return _load_completed_prepared(plan)
    if any(state):
        raise ProtocolError(
            "HARP v5 unjournaled prepared destinations are partially committed."
        )
    _discard_unjournaled_staging(paths)
    try:
        paths.staging_root.mkdir(parents=False, exist_ok=False)
        _fsync_directory(paths.staging_root.parent)
        _inject(_fault_injector, "staging_created")
        stage_cache, stage_development, stage_evaluation = staging_destinations(paths)
        prepared = _prepare_staged_inputs(
            plan,
            cache_root=stage_cache,
            development_manifest_path=stage_development / "manifest.csv",
            evaluation_manifest_path=stage_evaluation / "manifest.csv",
        )
        journal = build_preparation_journal(plan, prepared)
    except Exception:
        _discard_unjournaled_staging(paths)
        raise
    return commit_prepared_inputs(
        journal,
        fault_injector=_fault_injector,
    )


def recover_harp_v5_workstation_preparation(
    repository_root: str | Path,
    *,
    confirmation: str,
    _fault_injector: FaultInjector | None = None,
) -> HarpV5PreparedInputs:
    if confirmation != PREPARATION_CONFIRMATION:
        raise ProtocolError("HARP v5 preparation confirmation is absent or drifted.")
    plan = plan_harp_v5_workstation_preparation(repository_root)
    if not os.path.lexists(plan.paths.transaction_path):
        raise ProtocolError("HARP v5 preparation has no recoverable transaction.")
    _validate_scientific_sources(plan)
    return recover_prepared_inputs(
        plan.paths,
        expected_plan_hash=plan.preparation_plan_hash,
        fault_injector=_fault_injector,
    )


def inspect_harp_v5_workstation_preparation(
    repository_root: str | Path,
) -> dict[str, object]:
    plan = plan_harp_v5_workstation_preparation(repository_root)
    recovery = inspect_preparation_recovery(plan.paths)
    payload = plan.to_payload()
    if recovery is not None:
        payload["recovery"] = recovery
    return payload


def _prepare_staged_inputs(
    plan: HarpV5WorkstationPreparationPlan,
    *,
    cache_root: Path,
    development_manifest_path: Path,
    evaluation_manifest_path: Path,
) -> HarpV5PreparedInputs:
    data = _prepare_harp_consumed_test_inputs(
        canonical_cache_root=plan.paths.canonical_cache_root,
        canonical_manifest_path=plan.paths.canonical_manifest_path,
        parent_ledger_path=plan.paths.parent_ledger_path,
        cache_root=cache_root,
        development_manifest_path=development_manifest_path,
        evaluation_manifest_path=evaluation_manifest_path,
        identity=V5_PREPARATION_IDENTITY,
        expected_manifest_sha256=plan.expected_manifest_sha256,
        expected_parent_ledger_sha256=plan.expected_parent_ledger_sha256,
    )
    return HarpV5PreparedInputs(
        cache_root=data.cache_root,
        development_manifest_path=data.development_manifest_path,
        evaluation_manifest_path=data.evaluation_manifest_path,
        cache_content_sha256=data.cache_content_sha256,
        development_manifest_sha256=data.development_manifest_sha256,
        evaluation_manifest_sha256=data.evaluation_manifest_sha256,
        parent_ledger_sha256=data.parent_ledger_sha256,
        partition_hash=data.partition_hash,
        preparation_receipt_hash=data.preparation_receipt_hash,
    )


def _load_completed_prepared(
    plan: HarpV5WorkstationPreparationPlan,
) -> HarpV5PreparedInputs:
    _validate_scientific_sources(plan)
    paths = resolve_harp_v5_workspace_paths(
        plan.paths.repository_root, require_prepared=True
    )
    if paths != plan.paths:
        raise ProtocolError("HARP v5 completed preparation paths drifted.")
    # Authenticate the label-free cache and preparation receipt before opening
    # either role manifest.  This is the idempotent equivalent of the original
    # label-free barrier.
    cache = _validate_final_prepared_cache(paths.prepared_cache_root)
    receipt = read_json(paths.prepared_cache_root / PREPARATION_RECEIPT)
    base = {key: value for key, value in receipt.items() if key != "receipt_hash"}
    if (
        receipt.get("receipt_hash") != canonical_hash(base)
        or receipt.get("schema_version")
        != V5_PREPARATION_IDENTITY.preparation_receipt_schema
        or receipt.get("status") != "PREPARED_INPUTS_NO_EXECUTION_AUTHORITY"
        or receipt.get("canonical_cache_content_hash")
        != plan.canonical_cache_content_hash
        or receipt.get("canonical_cache_row_order_hash")
        != plan.canonical_cache_row_order_hash
        or receipt.get("canonical_manifest_sha256")
        != plan.expected_manifest_sha256
        or receipt.get("parent_ledger_sha256")
        != plan.expected_parent_ledger_sha256
        or receipt.get("cache_fsynced_and_independently_validated_before_manifest_open")
        is not True
        or receipt.get("execution_amendment_created") is not False
        or receipt.get("execution_authorized") is not False
    ):
        raise ProtocolError("HARP v5 completed preparation receipt drifted.")
    development_sha = str(receipt.get("development_manifest_sha256"))
    evaluation_sha = str(receipt.get("evaluation_manifest_sha256"))
    if (
        sha256_file(paths.parent_ledger_path) != plan.expected_parent_ledger_sha256
        or sha256_file(paths.development_manifest_path) != development_sha
        or sha256_file(paths.evaluation_manifest_path) != evaluation_sha
        or inventory_tree(paths.development_manifest_path.parent)
        != {"manifest.csv": development_sha}
        or inventory_tree(paths.evaluation_manifest_path.parent)
        != {"manifest.csv": evaluation_sha}
    ):
        raise ProtocolError("HARP v5 completed prepared inputs drifted.")
    _read_label_manifest(
        paths.development_manifest_path,
        expected_sha256=development_sha,
        expected_role=DEVELOPMENT_ROLE,
        cache=cache,
    )
    _read_label_manifest(
        paths.evaluation_manifest_path,
        expected_sha256=evaluation_sha,
        expected_role=EVALUATION_ROLE,
        cache=cache,
    )
    return HarpV5PreparedInputs(
        cache_root=paths.prepared_cache_root,
        development_manifest_path=paths.development_manifest_path,
        evaluation_manifest_path=paths.evaluation_manifest_path,
        cache_content_sha256=cache.content_sha256,
        development_manifest_sha256=development_sha,
        evaluation_manifest_sha256=evaluation_sha,
        parent_ledger_sha256=plan.expected_parent_ledger_sha256,
        partition_hash=str(receipt["partition_hash"]),
        preparation_receipt_hash=str(receipt["receipt_hash"]),
    )


def _verify_plan(plan: HarpV5WorkstationPreparationPlan) -> None:
    if canonical_hash(plan._hash_payload()) != plan.preparation_plan_hash:
        raise ProtocolError("HARP v5 preparation plan changed after inspection.")
    current = resolve_harp_v5_workspace_paths(
        plan.paths.repository_root, require_prepared=False
    )
    if current != plan.paths:
        raise ProtocolError("HARP v5 workspace paths changed after preparation plan.")
    if (
        sha256_file(plan.paths.config_path) != plan.config_sha256
        or sha256_file(plan.paths.registry_path) != plan.registry_sha256
        or sha256_file(plan.paths.catalog_path) != plan.catalog_sha256
    ):
        raise ProtocolError("HARP v5 workspace metadata changed after preparation plan.")
    _require_preparation_only_surface(plan.paths)


def _validate_scientific_sources(
    plan: HarpV5WorkstationPreparationPlan,
) -> None:
    """Reject recovery or idempotent reuse after canonical-source drift."""

    source = validate_canonical_label_blind_cache_identity(
        plan.paths.canonical_cache_root
    )
    if (
        source.content_hash != plan.canonical_cache_content_hash
        or sha256_file(plan.paths.canonical_manifest_path)
        != plan.expected_manifest_sha256
        or sha256_file(plan.paths.parent_ledger_path)
        != plan.expected_parent_ledger_sha256
    ):
        raise ProtocolError(
            "HARP v5 canonical preparation sources changed before recovery."
        )
    read_json(plan.paths.parent_ledger_path)


def _observed_state(paths: HarpV5WorkspacePaths) -> str:
    state = _destination_state(paths)
    if os.path.lexists(paths.transaction_path):
        return "RECOVERY_REQUIRED_OR_COMPLETABLE"
    if state == (False, False, False):
        return "READY_FOR_EXPLICIT_PREPARATION_CONFIRMATION"
    if state == (True, True, True):
        return "PREPARED_INPUTS_CANDIDATE_REVALIDATION_REQUIRED"
    raise ProtocolError("HARP v5 unjournaled prepared destinations are partial.")


def _destination_state(paths: HarpV5WorkspacePaths) -> tuple[bool, bool, bool]:
    return (
        os.path.lexists(paths.prepared_cache_root),
        os.path.lexists(paths.development_manifest_path.parent),
        os.path.lexists(paths.evaluation_manifest_path.parent),
    )


def _require_preparation_only_surface(paths: HarpV5WorkspacePaths) -> None:
    if os.path.lexists(paths.amendment_path):
        raise ProtocolError("HARP v5 inputs cannot be prepared after amendment issuance.")
    if os.path.lexists(paths.output_root):
        raise ProtocolError("HARP v5 preparation refuses an existing output.")
    if os.path.lexists(authorization.lease_path(paths.repository_root)):
        raise ProtocolError("HARP v5 preparation refuses an existing lease.")


def _discard_unjournaled_staging(paths: HarpV5WorkspacePaths) -> None:
    root = paths.staging_root
    if not os.path.lexists(root):
        return
    if os.path.lexists(paths.transaction_path):
        raise ProtocolError("HARP v5 journaled staging requires exact recovery.")
    if not root.is_dir() or root.is_symlink():
        raise ProtocolError("HARP v5 unjournaled staging is unsafe.")
    entries = tuple(root.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise ProtocolError("HARP v5 unjournaled staging contains a symlink.")
    shutil.rmtree(root)
    _fsync_directory(root.parent)


def _catalog_sha256(
    workspace: MidogppWorkspace,
    artifact_id: str,
    member: str,
    *,
    label: str,
) -> str:
    try:
        expectation = workspace.artifacts[artifact_id].expected_file_hashes[member]
    except KeyError as exc:
        raise ProtocolError(f"HARP v5 catalog lacks the {label} hash.") from exc
    if expectation.algorithm != "sha256":
        raise ProtocolError(f"HARP v5 {label} hash algorithm drifted.")
    return expectation.digest


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _inject(fault_injector: FaultInjector | None, point: str) -> None:
    if fault_injector is not None:
        fault_injector(point)


__all__ = (
    "HarpV5WorkstationPreparationPlan",
    "PREPARATION_CONFIRMATION",
    "inspect_harp_v5_workstation_preparation",
    "plan_harp_v5_workstation_preparation",
    "prepare_harp_v5_workstation_inputs",
    "recover_harp_v5_workstation_preparation",
)
