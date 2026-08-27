"""Construct exact outer-center tasks and their scoped label delegations."""

from __future__ import annotations

import json
from pathlib import Path

from ..config import ScaleBPV2Config
from ..execution import OuterCenterTask
from ..identity import (
    CENTERS,
    EXPECTED_CASE_COUNT,
    EXPECTED_TEST_MANIFEST_SHA256,
    GovernanceError,
)
from ..input_contracts import LabelFreeTestFrame
from ..label_capabilities import (
    LabelCapabilityJournal,
    WorkerLabelDelegation,
    WorkerSupportScope,
)
from ..manifest_labels import ManifestLabelDecoder
from ..physical_memmaps import PhysicalMemmapBundle


TASK_PAYLOAD_SCHEMA = "scale_bp_v2_outer_science_task_payload_v1"


def build_outer_tasks(
    *,
    config: ScaleBPV2Config,
    root: Path,
    frame: LabelFreeTestFrame,
    decoder: ManifestLabelDecoder,
    journal: LabelCapabilityJournal,
    memmaps: PhysicalMemmapBundle,
    identity_index_path: Path,
    identity_hash: str,
    run_identity_hash: str,
    protocol_hash: str,
) -> tuple[tuple[OuterCenterTask, ...], dict[str, WorkerLabelDelegation]]:
    """Create one immutable task and one matching delegation per outer center."""

    tasks: list[OuterCenterTask] = []
    delegations: dict[str, WorkerLabelDelegation] = {}
    for center in CENTERS:
        center_rows = frame.rows_by_center[center]
        case_ids = tuple(sorted({row.case_id for row in center_rows}))
        route_scopes = tuple(
            _support_scope(decoder, center=center, case_id=case_id)
            for case_id in case_ids
        )
        task_id = f"scale-bp-v2:outer:{center}"
        donor_hash = decoder.donor_identity_hash(center)
        payload = {
            "schema_version": TASK_PAYLOAD_SCHEMA,
            "artifact_root": str(root),
            "physical_index_path": str(memmaps.index_path),
            "physical_index_hash": memmaps.index_hash,
            "label_identity_index_path": str(identity_index_path.resolve(strict=True)),
            "label_identity_hash": identity_hash,
            "manifest_path": str(config.test_manifest_path),
            "manifest_sha256": EXPECTED_TEST_MANIFEST_SHA256,
            "delegation_seed": {
                "parent_journal_id": journal.journal_id,
                "run_identity_hash": run_identity_hash,
                "task_id": task_id,
                "donor_identity_hash": donor_hash,
                "route_scopes": [scope.to_payload() for scope in route_scopes],
            },
            "scientific_contracts": {
                key: dict(value) for key, value in config.scientific_contracts.items()
            },
        }
        task = OuterCenterTask(
            target_center=center,
            case_ids=case_ids,
            memmaps=memmaps.references,
            protocol_hash=protocol_hash,
            payload_json=json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )
        delegation = journal.delegate_outer_worker(
            task_id=task_id,
            outer_center=center,
            task_hash=task.task_hash,
            manifest_path=config.test_manifest_path,
            manifest_sha256=EXPECTED_TEST_MANIFEST_SHA256,
            donor_identity_hash=donor_hash,
            route_scopes=route_scopes,
        )
        tasks.append(task)
        delegations[center] = delegation
    if sum(len(task.case_ids) for task in tasks) != EXPECTED_CASE_COUNT:
        raise GovernanceError("SCALE-BP v2 outer task case inventory drifted.")
    return tuple(tasks), delegations


def _support_scope(
    decoder: ManifestLabelDecoder, *, center: str, case_id: str
) -> WorkerSupportScope:
    support_hash, evaluation_hash = decoder.support_identity_hashes(center, case_id)
    return WorkerSupportScope(
        held_case_id=case_id,
        support_identity_hash=support_hash,
        evaluation_identity_hash=evaluation_hash,
    )


__all__ = ("TASK_PAYLOAD_SCHEMA", "build_outer_tasks")
