"""Thin one-H coordinator for the responsibility-scoped worker phases."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..execution.dtos import OuterCenterResult, OuterCenterTask
from ..identity import GovernanceError
from ..label_capabilities import DelegatedWorkerLabelJournal
from ..label_identity import load_label_identity_index
from ..manifest_labels import ManifestLabelDecoder
from ..physical_memmaps import open_mapped_physical_store
from .boundary import (
    EXPECTED_PHYSICAL_ROLES,
    parse_task_payload,
    validate_worker_inventory,
)
from .contracts import FinalRouteOutput
from .donor_phase import run_donor_phase
from .emission import emit_outer_center_result
from .route_phase import run_final_route


def coordinate_outer_center_science(
    task: OuterCenterTask,
    arrays: Mapping[str, np.memmap],
) -> OuterCenterResult:
    """Execute exactly one complete outer H without nested process pools."""

    if not isinstance(task, OuterCenterTask):
        raise GovernanceError("SCALE-BP v2 science worker received a foreign task.")
    parsed = parse_task_payload(task)
    if tuple(arrays) != EXPECTED_PHYSICAL_ROLES:
        raise GovernanceError("SCALE-BP v2 worker physical map inventory drifted.")
    store = open_mapped_physical_store(
        arrays,
        index_path=parsed.physical_index_path,
        expected_index_hash=parsed.physical_index_hash,
    )
    identity = load_label_identity_index(
        parsed.label_identity_index_path,
        expected_identity_hash=parsed.label_identity_hash,
    )
    validate_worker_inventory(task, store, identity, parsed)
    decoder = ManifestLabelDecoder(identity, parsed.manifest_path)
    journal = DelegatedWorkerLabelJournal(parsed.delegation)
    journal.verify_manifest_file()
    if decoder.donor_identity_hash(task.target_center) != (
        parsed.delegation.donor_identity_hash
    ):
        raise GovernanceError("SCALE-BP v2 delegated donor identity drifted.")

    donor_capability = journal.open_donor_scope()
    donor_labels = decoder.decode_donor(
        journal,
        donor_capability,
        outer_center=task.target_center,
    )
    donor_phase = run_donor_phase(
        task,
        store,
        journal,
        donor_capability,
        donor_labels,
        parsed.settings,
    )
    del donor_labels
    journal.close_donor_scope(donor_capability)

    route_outputs: list[FinalRouteOutput] = []
    for case_id in task.case_ids:
        support_capability = journal.open_support_scope(case_id)
        support_labels = decoder.decode_support(
            journal,
            support_capability,
            target_center=task.target_center,
            held_case_id=case_id,
        )
        route_outputs.append(
            run_final_route(
                task,
                store,
                journal,
                support_capability,
                support_labels,
                case_id=case_id,
                final_prior=donor_phase.final_prior,
                final_donor_model=donor_phase.final_model,
                admission=donor_phase.admission,
                settings=parsed.settings,
            )
        )
        del support_labels
        journal.close_support_scope(support_capability)

    return emit_outer_center_result(
        task,
        parsed,
        store,
        journal,
        donor_phase,
        route_outputs,
    )


__all__ = ("coordinate_outer_center_science",)
