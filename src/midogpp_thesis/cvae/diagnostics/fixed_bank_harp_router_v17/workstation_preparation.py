"""Public catalog-bound preparation lifecycle for HARP v17 on the workstation."""

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
    SOURCE_TRAIN_ROLE,
    EVALUATION_RELEASE_MEMBER,
    _read_evaluation_release_descriptor,
    _read_label_manifest,
)
from .preparation import (
    CANONICAL_CACHE_CONTENT_HASH,
    CANONICAL_CACHE_ROW_ORDER_HASH,
    CANONICAL_MANIFEST_SHA256,
    CANONICAL_PARENT_LEDGER_SHA256,
    CANONICAL_SOURCE_TRAIN_BUILDER_REPORT_SHA256,
    CANONICAL_SOURCE_TRAIN_CONTENT_INDEX_SHA256,
    CANONICAL_SOURCE_TRAIN_PROTOCOL_SHA256,
    CANONICAL_SOURCE_TRAIN_TENSOR_SHA256,
    CANONICAL_SOURCE_TRAIN_VALIDATION_REPORT_SHA256,
    PREPARATION_RECEIPT,
    HarpV17PreparedInputs,
    V17_PREPARATION_IDENTITY,
    _prepare_harp_consumed_test_inputs,
    _validate_final_prepared_cache,
    validate_canonical_label_blind_cache_identity,
)
from .preparation_source_train_cache import _validate_source_train_cache_identity
from .preparation_transaction import (
    build_preparation_journal,
    commit_prepared_inputs,
    inspect_preparation_recovery,
    inventory_tree,
    recover_prepared_inputs,
    staging_destinations,
)
from .workspace_paths import (
    CANONICAL_TEST_CACHE_ARTIFACT_ID,
    CANONICAL_TRAIN_CACHE_ARTIFACT_ID,
    CANONICAL_MANIFEST_ARTIFACT_ID,
    HarpV17WorkspacePaths,
    resolve_harp_v17_workspace_paths,
)


PREPARATION_CONFIRMATION = "PREPARE_HARP_V17_CONSUMED_TEST_INPUTS"
PREPARATION_PLAN_SCHEMA = "midogpp_harp_v17_workstation_preparation_plan_v1"
FaultInjector = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class HarpV17WorkstationPreparationPlan:
    paths: HarpV17WorkspacePaths
    canonical_cache_content_hash: str
    canonical_cache_row_order_hash: str
    canonical_source_train_member_sha256: Mapping[str, str]
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
            raise ProtocolError("HARP v17 preparation plan identity drifted.")

    def _hash_payload(self) -> dict[str, object]:
        paths = self.paths
        return {
            "schema_version": PREPARATION_PLAN_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "repository_root": paths.repository_root.as_posix(),
            "canonical_source_artifact_id": CANONICAL_TRAIN_CACHE_ARTIFACT_ID,
            "canonical_source_train_artifact_id": CANONICAL_TRAIN_CACHE_ARTIFACT_ID,
            "canonical_target_test_artifact_id": CANONICAL_TEST_CACHE_ARTIFACT_ID,
            "canonical_manifest_artifact_id": CANONICAL_MANIFEST_ARTIFACT_ID,
            "canonical_source_train_cache_root": paths.canonical_train_cache_root.as_posix(),
            "canonical_target_test_cache_root": paths.canonical_test_cache_root.as_posix(),
            "canonical_manifest_path": paths.canonical_manifest_path.as_posix(),
            "expert_bank_root": paths.expert_bank_root.as_posix(),
            "generation_lock_root": paths.generation_lock_root.as_posix(),
            "parent_ledger_path": paths.parent_ledger_path.as_posix(),
            "prepared_cache_root": paths.prepared_cache_root.as_posix(),
            "development_manifest_path": paths.development_manifest_path.as_posix(),
            "evaluation_manifest_path": paths.evaluation_manifest_path.as_posix(),
            "evaluation_artifact_kind": "sealed_label_free_release_descriptor",
            "amendment_path": paths.amendment_path.as_posix(),
            "output_root": paths.output_root.as_posix(),
            "canonical_cache_content_hash": self.canonical_cache_content_hash,
            "canonical_cache_row_order_hash": self.canonical_cache_row_order_hash,
            "canonical_source_train_member_sha256": dict(
                sorted(self.canonical_source_train_member_sha256.items())
            ),
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


def plan_harp_v17_workstation_preparation(
    repository_root: str | Path,
) -> HarpV17WorkstationPreparationPlan:
    """Plan exact catalog outputs without opening scoring-manifest bytes."""

    paths = resolve_harp_v17_workspace_paths(repository_root, require_prepared=False)
    _require_preparation_only_surface(paths)
    try:
        workspace = MidogppWorkspace.load(paths.repository_root)
    except WorkspaceError as exc:
        raise ProtocolError("HARP v17 workspace cannot be loaded for preparation.") from exc
    experiment = workspace.experiments.get(EXPERIMENT_ID)
    if (
        experiment is None
        or experiment.status != "planned"
        or experiment.runnable
        or experiment.input_artifact_ids != INPUT_ARTIFACT_IDS
    ):
        raise ProtocolError("HARP v17 preparation requires the planned workspace state.")
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
    source_train_members = {
        "embeddings/train.pt": _catalog_sha256(
            workspace,
            CANONICAL_TRAIN_CACHE_ARTIFACT_ID,
            "embeddings/train.pt",
            label="canonical source-train tensor",
        ),
        "manifests/frozen_cache_protocol.json": _catalog_sha256(
            workspace,
            CANONICAL_TRAIN_CACHE_ARTIFACT_ID,
            "manifests/frozen_cache_protocol.json",
            label="canonical source-train protocol",
        ),
        "manifests/content_index.json": _catalog_sha256(
            workspace,
            CANONICAL_TRAIN_CACHE_ARTIFACT_ID,
            "manifests/content_index.json",
            label="canonical source-train content index",
        ),
        "reports/cache_builder_report.json": _catalog_sha256(
            workspace,
            CANONICAL_TRAIN_CACHE_ARTIFACT_ID,
            "reports/cache_builder_report.json",
            label="canonical source-train builder report",
        ),
        "reports/validation_report.json": _catalog_sha256(
            workspace,
            CANONICAL_TRAIN_CACHE_ARTIFACT_ID,
            "reports/validation_report.json",
            label="canonical source-train validation report",
        ),
    }
    expected_source_train_members = {
        "embeddings/train.pt": CANONICAL_SOURCE_TRAIN_TENSOR_SHA256,
        "manifests/frozen_cache_protocol.json": CANONICAL_SOURCE_TRAIN_PROTOCOL_SHA256,
        "manifests/content_index.json": CANONICAL_SOURCE_TRAIN_CONTENT_INDEX_SHA256,
        "reports/cache_builder_report.json": CANONICAL_SOURCE_TRAIN_BUILDER_REPORT_SHA256,
        "reports/validation_report.json": CANONICAL_SOURCE_TRAIN_VALIDATION_REPORT_SHA256,
    }
    cache_semantics = workspace.artifacts[
        CANONICAL_TEST_CACHE_ARTIFACT_ID
    ].semantic_identities
    content_hash = cache_semantics.get("content_hash")
    row_order_hash = cache_semantics.get("row_order_hash")
    if (
        content_hash != CANONICAL_CACHE_CONTENT_HASH
        or row_order_hash != CANONICAL_CACHE_ROW_ORDER_HASH
        or source_train_members != expected_source_train_members
        or expected_manifest != CANONICAL_MANIFEST_SHA256
        or expected_parent != CANONICAL_PARENT_LEDGER_SHA256
    ):
        raise ProtocolError("HARP v17 catalog preparation identity drifted.")
    observed = _observed_state(paths)
    provisional = HarpV17WorkstationPreparationPlan(
        paths=paths,
        canonical_cache_content_hash=content_hash,
        canonical_cache_row_order_hash=row_order_hash,
        canonical_source_train_member_sha256=source_train_members,
        expected_manifest_sha256=expected_manifest,
        expected_parent_ledger_sha256=expected_parent,
        config_sha256=sha256_file(paths.config_path),
        registry_sha256=sha256_file(paths.registry_path),
        catalog_sha256=sha256_file(paths.catalog_path),
        observed_state=observed,
        preparation_plan_hash="",
    )
    return HarpV17WorkstationPreparationPlan(
        paths=provisional.paths,
        canonical_cache_content_hash=provisional.canonical_cache_content_hash,
        canonical_cache_row_order_hash=provisional.canonical_cache_row_order_hash,
        canonical_source_train_member_sha256=(
            provisional.canonical_source_train_member_sha256
        ),
        expected_manifest_sha256=provisional.expected_manifest_sha256,
        expected_parent_ledger_sha256=provisional.expected_parent_ledger_sha256,
        config_sha256=provisional.config_sha256,
        registry_sha256=provisional.registry_sha256,
        catalog_sha256=provisional.catalog_sha256,
        observed_state=provisional.observed_state,
        preparation_plan_hash=canonical_hash(provisional._hash_payload()),
    )


def prepare_harp_v17_workstation_inputs(
    plan: HarpV17WorkstationPreparationPlan,
    *,
    confirmation: str,
    _fault_injector: FaultInjector | None = None,
) -> HarpV17PreparedInputs:
    """Build in owned staging and publish only after an authenticated receipt."""

    if type(plan) is not HarpV17WorkstationPreparationPlan:
        raise ProtocolError("HARP v17 workstation preparation requires a typed plan.")
    if confirmation != PREPARATION_CONFIRMATION:
        raise ProtocolError("HARP v17 preparation confirmation is absent or drifted.")
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
            "HARP v17 unjournaled prepared destinations are partially committed."
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
            development_manifest_path=stage_development / "index.json",
            evaluation_manifest_path=stage_evaluation / EVALUATION_RELEASE_MEMBER,
        )
        journal = build_preparation_journal(plan, prepared)
    except Exception:
        _discard_unjournaled_staging(paths)
        raise
    return commit_prepared_inputs(
        journal,
        fault_injector=_fault_injector,
    )


def recover_harp_v17_workstation_preparation(
    repository_root: str | Path,
    *,
    confirmation: str,
    _fault_injector: FaultInjector | None = None,
) -> HarpV17PreparedInputs:
    if confirmation != PREPARATION_CONFIRMATION:
        raise ProtocolError("HARP v17 preparation confirmation is absent or drifted.")
    plan = plan_harp_v17_workstation_preparation(repository_root)
    if not os.path.lexists(plan.paths.transaction_path):
        raise ProtocolError("HARP v17 preparation has no recoverable transaction.")
    _validate_scientific_sources(plan)
    return recover_prepared_inputs(
        plan.paths,
        expected_plan_hash=plan.preparation_plan_hash,
        fault_injector=_fault_injector,
    )


def inspect_harp_v17_workstation_preparation(
    repository_root: str | Path,
) -> dict[str, object]:
    plan = plan_harp_v17_workstation_preparation(repository_root)
    recovery = inspect_preparation_recovery(plan.paths)
    payload = plan.to_payload()
    if recovery is not None:
        payload["recovery"] = recovery
    return payload


def _prepare_staged_inputs(
    plan: HarpV17WorkstationPreparationPlan,
    *,
    cache_root: Path,
    development_manifest_path: Path,
    evaluation_manifest_path: Path,
) -> HarpV17PreparedInputs:
    data = _prepare_harp_consumed_test_inputs(
        canonical_train_cache_root=plan.paths.canonical_train_cache_root,
        canonical_test_cache_root=plan.paths.canonical_test_cache_root,
        canonical_manifest_path=plan.paths.canonical_manifest_path,
        parent_ledger_path=plan.paths.parent_ledger_path,
        cache_root=cache_root,
        development_manifest_path=development_manifest_path,
        evaluation_manifest_path=evaluation_manifest_path,
        identity=V17_PREPARATION_IDENTITY,
        expected_manifest_sha256=plan.expected_manifest_sha256,
        expected_parent_ledger_sha256=plan.expected_parent_ledger_sha256,
    )
    return HarpV17PreparedInputs(
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
    plan: HarpV17WorkstationPreparationPlan,
) -> HarpV17PreparedInputs:
    _validate_scientific_sources(plan)
    paths = resolve_harp_v17_workspace_paths(
        plan.paths.repository_root, require_prepared=True
    )
    if paths != plan.paths:
        raise ProtocolError("HARP v17 completed preparation paths drifted.")
    # Authenticate the label-free cache and preparation receipt before opening
    # the source-train capability or evaluation release descriptor. This is
    # the idempotent equivalent of the original label-free barrier.
    cache = _validate_final_prepared_cache(paths.prepared_cache_root)
    receipt = read_json(paths.prepared_cache_root / PREPARATION_RECEIPT)
    base = {key: value for key, value in receipt.items() if key != "receipt_hash"}
    if (
        receipt.get("receipt_hash") != canonical_hash(base)
        or receipt.get("schema_version")
        != V17_PREPARATION_IDENTITY.preparation_receipt_schema
        or receipt.get("status") != "PREPARED_INPUTS_NO_EXECUTION_AUTHORITY"
        or receipt.get("canonical_source_train_tensor_sha256")
        != CANONICAL_SOURCE_TRAIN_TENSOR_SHA256
        or not _is_sha256(receipt.get("canonical_source_train_row_order_hash"))
        or receipt.get("canonical_target_test_cache_content_hash")
        != plan.canonical_cache_content_hash
        or receipt.get("canonical_target_test_row_order_hash")
        != plan.canonical_cache_row_order_hash
        or receipt.get("canonical_test_manifest_sha256")
        != plan.expected_manifest_sha256
        or receipt.get("parent_ledger_sha256")
        != plan.expected_parent_ledger_sha256
        or receipt.get("cache_fsynced_and_independently_validated_before_manifest_open")
        is not True
        or receipt.get("source_train_row_count") != 9648
        or receipt.get("source_train_case_count") != 216
        or receipt.get("source_train_label_capability_artifact_kind")
        != "center_sharded_source_train_label_capability"
        or receipt.get("source_train_label_capability_shard_count") != 9
        or receipt.get("source_train_label_capability_index_contains_labels")
        is not False
        or receipt.get("source_train_label_capability_state")
        != "SOURCE_TRAIN_CENTER_SCOPED_OPEN_AFTER_ALL_SOURCE_AND_TARGET_MENU_SEALS_AND_BANK_ATTESTATIONS"
        or receipt.get("source_train_label_fit_scope")
        != "all_216_train_cases_pooled_across_nine_known_centers"
        or receipt.get(
            "all_source_and_target_menus_must_seal_before_any_source_label_open"
        )
        is not True
        or receipt.get("candidate_expert_q_excluded") is not True
        or receipt.get(
            "bank_independence_attestation_required_per_source_and_target_context"
        )
        is not True
        or receipt.get("target_evaluation_row_count") != 9928
        or receipt.get("target_evaluation_case_count") != 218
        or receipt.get("test_development_case_count") != 0
        or receipt.get("execution_amendment_created") is not False
        or receipt.get("execution_authorized") is not False
    ):
        raise ProtocolError("HARP v17 completed preparation receipt drifted.")
    development_sha = str(receipt.get("development_manifest_sha256"))
    evaluation_sha = str(receipt.get("evaluation_manifest_sha256"))
    if (
        sha256_file(paths.parent_ledger_path) != plan.expected_parent_ledger_sha256
        or sha256_file(paths.development_manifest_path) != development_sha
        or sha256_file(paths.evaluation_manifest_path) != evaluation_sha
        or inventory_tree(paths.development_manifest_path.parent)
        != _source_capability_inventory(
            paths.development_manifest_path,
            index_sha256=development_sha,
        )
        or inventory_tree(paths.evaluation_manifest_path.parent)
        != {EVALUATION_RELEASE_MEMBER: evaluation_sha}
    ):
        raise ProtocolError("HARP v17 completed prepared inputs drifted.")
    _read_label_manifest(
        paths.development_manifest_path,
        expected_sha256=development_sha,
        expected_role=SOURCE_TRAIN_ROLE,
        cache=cache,
    )
    _read_evaluation_release_descriptor(
        paths.evaluation_manifest_path,
        expected_sha256=evaluation_sha,
        cache=cache,
    )
    return HarpV17PreparedInputs(
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


def _verify_plan(plan: HarpV17WorkstationPreparationPlan) -> None:
    if canonical_hash(plan._hash_payload()) != plan.preparation_plan_hash:
        raise ProtocolError("HARP v17 preparation plan changed after inspection.")
    current = resolve_harp_v17_workspace_paths(
        plan.paths.repository_root, require_prepared=False
    )
    if current != plan.paths:
        raise ProtocolError("HARP v17 workspace paths changed after preparation plan.")
    if (
        sha256_file(plan.paths.config_path) != plan.config_sha256
        or sha256_file(plan.paths.registry_path) != plan.registry_sha256
        or sha256_file(plan.paths.catalog_path) != plan.catalog_sha256
    ):
        raise ProtocolError("HARP v17 workspace metadata changed after preparation plan.")
    _require_preparation_only_surface(plan.paths)


def _validate_scientific_sources(
    plan: HarpV17WorkstationPreparationPlan,
) -> None:
    """Reject recovery or idempotent reuse after canonical-source drift."""

    _validate_source_train_cache_identity(plan.paths.canonical_train_cache_root)
    source = validate_canonical_label_blind_cache_identity(
        plan.paths.canonical_test_cache_root
    )
    if (
        source.content_hash != plan.canonical_cache_content_hash
        or plan.canonical_source_train_member_sha256
        != {
            "embeddings/train.pt": CANONICAL_SOURCE_TRAIN_TENSOR_SHA256,
            "manifests/frozen_cache_protocol.json": CANONICAL_SOURCE_TRAIN_PROTOCOL_SHA256,
            "manifests/content_index.json": CANONICAL_SOURCE_TRAIN_CONTENT_INDEX_SHA256,
            "reports/cache_builder_report.json": CANONICAL_SOURCE_TRAIN_BUILDER_REPORT_SHA256,
            "reports/validation_report.json": CANONICAL_SOURCE_TRAIN_VALIDATION_REPORT_SHA256,
        }
        or sha256_file(plan.paths.canonical_manifest_path)
        != plan.expected_manifest_sha256
        or sha256_file(plan.paths.parent_ledger_path)
        != plan.expected_parent_ledger_sha256
    ):
        raise ProtocolError(
            "HARP v17 canonical preparation sources changed before recovery."
        )
    read_json(plan.paths.parent_ledger_path)


def _observed_state(paths: HarpV17WorkspacePaths) -> str:
    state = _destination_state(paths)
    if os.path.lexists(paths.transaction_path):
        return "RECOVERY_REQUIRED_OR_COMPLETABLE"
    if state == (False, False, False):
        return "READY_FOR_EXPLICIT_PREPARATION_CONFIRMATION"
    if state == (True, True, True):
        return "PREPARED_INPUTS_CANDIDATE_REVALIDATION_REQUIRED"
    raise ProtocolError("HARP v17 unjournaled prepared destinations are partial.")


def _destination_state(paths: HarpV17WorkspacePaths) -> tuple[bool, bool, bool]:
    return (
        os.path.lexists(paths.prepared_cache_root),
        os.path.lexists(paths.development_manifest_path.parent),
        os.path.lexists(paths.evaluation_manifest_path.parent),
    )


def _require_preparation_only_surface(paths: HarpV17WorkspacePaths) -> None:
    if os.path.lexists(paths.amendment_path):
        raise ProtocolError("HARP v17 inputs cannot be prepared after amendment issuance.")
    if os.path.lexists(paths.output_root):
        raise ProtocolError("HARP v17 preparation refuses an existing output.")
    if os.path.lexists(authorization.lease_path(paths.repository_root)):
        raise ProtocolError("HARP v17 preparation refuses an existing lease.")


def _discard_unjournaled_staging(paths: HarpV17WorkspacePaths) -> None:
    root = paths.staging_root
    if not os.path.lexists(root):
        return
    if os.path.lexists(paths.transaction_path):
        raise ProtocolError("HARP v17 journaled staging requires exact recovery.")
    if not root.is_dir() or root.is_symlink():
        raise ProtocolError("HARP v17 unjournaled staging is unsafe.")
    entries = tuple(root.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise ProtocolError("HARP v17 unjournaled staging contains a symlink.")
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
        raise ProtocolError(f"HARP v17 catalog lacks the {label} hash.") from exc
    if expectation.algorithm != "sha256":
        raise ProtocolError(f"HARP v17 {label} hash algorithm drifted.")
    return expectation.digest


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _source_capability_inventory(
    index_path: Path,
    *,
    index_sha256: str,
) -> dict[str, str]:
    payload = read_json(index_path)
    raw_shards = payload.get("shards")
    if not isinstance(raw_shards, list):
        raise ProtocolError("HARP v17 source-label capability index is malformed.")
    inventory = {"index.json": index_sha256}
    for raw in raw_shards:
        if not isinstance(raw, Mapping):
            raise ProtocolError("HARP v17 source-label capability shard is malformed.")
        relative = str(raw.get("relative_path", ""))
        digest = raw.get("sha256")
        if relative in inventory or not _is_sha256(digest):
            raise ProtocolError("HARP v17 source-label capability shard drifted.")
        inventory[relative] = str(digest)
    return inventory


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
    "HarpV17WorkstationPreparationPlan",
    "PREPARATION_CONFIRMATION",
    "inspect_harp_v17_workstation_preparation",
    "plan_harp_v17_workstation_preparation",
    "prepare_harp_v17_workstation_inputs",
    "recover_harp_v17_workstation_preparation",
)
