from __future__ import annotations

import csv
import hashlib
import pickle

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.hashing import canonical_hash
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.identity import (
    CENTERS,
    EXPECTED_CASE_COUNT,
    EXPECTED_CASE_COUNTS_BY_CENTER,
    EXPECTED_TEST_MANIFEST_SHA256,
    FEATURE_DIM,
    GovernanceError,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.input_contracts import (
    LabelFreeTestFrame,
    TestRowIdentity as RowIdentity,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.label_capabilities import (
    CLOSED,
    TERMINAL_OPEN,
    LabelCapabilityJournal,
    WorkerCapabilityAudit,
    WorkerSupportScope,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.manifest_labels import ManifestLabelDecoder


def _digest(value: object) -> str:
    return canonical_hash({"value": value})


def test_parent_delegation_covers_all_nine_centers_and_218_routes(tmp_path) -> None:
    journal = LabelCapabilityJournal(_digest("run"))
    manifest_path = (tmp_path / "test.csv").resolve()
    manifest_path.write_text("unused", encoding="utf-8")

    total = 0
    for center, count in EXPECTED_CASE_COUNTS_BY_CENTER:
        scopes = tuple(
            WorkerSupportScope(
                held_case_id=f"{center}-case-{index:03d}",
                support_identity_hash=_digest((center, index, "support")),
                evaluation_identity_hash=_digest((center, index, "evaluation")),
            )
            for index in range(count)
        )
        delegation = journal.delegate_outer_worker(
            task_id=f"outer-{center}",
            outer_center=center,
            task_hash=_digest((center, "task")),
            manifest_path=manifest_path,
            manifest_sha256=EXPECTED_TEST_MANIFEST_SHA256,
            donor_identity_hash=_digest((center, "donor")),
            route_scopes=scopes,
        )
        audit = WorkerCapabilityAudit(
            delegation_hash=delegation.delegation_hash,
            worker_journal_id=_digest((center, "journal")),
            task_id=delegation.task_id,
            task_hash=delegation.task_hash,
            outer_center=center,
            donor_scope_count=1,
            support_scope_count=count,
            event_count=2 * (1 + count),
            event_log_hash=_digest((center, "events")),
            decision_fragment_hash=_digest((center, "decisions")),
        )
        journal.accept_worker_audit(delegation, audit)
        total += len(scopes)

    assert total == EXPECTED_CASE_COUNT
    journal.seal_decisions(
        decision_seal_hash=_digest("all-218-decisions"),
        route_count=EXPECTED_CASE_COUNT,
    )
    terminal = journal.open_terminal_scope(
        scope_id="terminal-once",
        terminal_identity_hash=_digest("terminal-identity"),
        decision_seal_hash=_digest("all-218-decisions"),
    )
    assert journal.phase == TERMINAL_OPEN
    journal.close_terminal_scope(terminal)
    audit = journal.audit_payload()
    assert journal.phase == CLOSED
    assert audit["delegated_worker_count"] == len(CENTERS)
    assert audit["accepted_worker_audit_count"] == len(CENTERS)
    assert audit["raw_labels_persisted"] is False
    assert audit["row_level_label_values_persisted"] is False


def _synthetic_manifest_and_frame(tmp_path, monkeypatch):
    import midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.manifest_labels as labels_module

    manifest = (tmp_path / "manifest.csv").resolve()
    raw_rows = []
    ordinal = 0
    for center, count in EXPECTED_CASE_COUNTS_BY_CENTER:
        for case_index in range(count):
            raw_rows.append(
                {
                    "sample_id": f"source-{ordinal}",
                    "case_id": f"center-{center}-case-{case_index:03d}",
                    "label": str((ordinal + case_index) % 2),
                    "center": center,
                    "split": "test",
                }
            )
            ordinal += 1
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    monkeypatch.setattr(labels_module, "EXPECTED_TEST_MANIFEST_SHA256", digest)
    monkeypatch.setattr(labels_module, "EXPECTED_TEST_ROW_COUNT", EXPECTED_CASE_COUNT)

    rows = tuple(
        RowIdentity(
            row_ordinal=index,
            manifest_row_index=index,
            sample_id=f"eval_{canonical_hash({'manifest_sha256': digest, 'contract_row_index': index})}",
            case_id=source["case_id"],
            center=source["center"],
            patient_slide_group_id=source["case_id"],
        )
        for index, source in enumerate(raw_rows)
    )
    by_center = {
        center: tuple(row for row in rows if row.center == center) for center in CENTERS
    }
    frame = LabelFreeTestFrame(
        np.zeros((EXPECTED_CASE_COUNT, FEATURE_DIM), dtype=np.float32),
        rows,
        by_center,
        {"cache_content_hash": "a" * 64, "row_order_hash": "b" * 64},
    )
    return manifest, frame


def test_manifest_decoder_obeys_donor_support_and_terminal_capabilities(
    tmp_path, monkeypatch
) -> None:
    manifest, frame = _synthetic_manifest_and_frame(tmp_path, monkeypatch)
    decoder = ManifestLabelDecoder(frame, manifest)
    journal = LabelCapabilityJournal(_digest("local-run"))

    donor_identity = decoder.donor_identity_hash("0")
    donor_capability = journal.open_donor_scope(
        scope_id="donor-0",
        outer_center="0",
        donor_centers=tuple(center for center in CENTERS if center != "0"),
        row_identity_hash=donor_identity,
    )
    donor_labels = decoder.decode_donor(
        journal, donor_capability, outer_center="0"
    )
    assert "0" not in donor_labels.labels_by_center_case
    assert set(donor_labels.labels_by_center_case) == set(CENTERS) - {"0"}
    with pytest.raises(TypeError):
        pickle.dumps(donor_labels)
    journal.close_donor_scope(donor_capability)
    with pytest.raises(GovernanceError):
        decoder.decode_donor(journal, donor_capability, outer_center="0")

    held = frame.rows_by_center["0"][0].case_id
    support_hash, evaluation_hash = decoder.support_identity_hashes("0", held)
    support_capability = journal.open_support_scope(
        scope_id="support-0-held",
        target_center="0",
        held_case_id=held,
        support_identity_hash=support_hash,
        evaluation_identity_hash=evaluation_hash,
    )
    support_labels = decoder.decode_support(
        journal,
        support_capability,
        target_center="0",
        held_case_id=held,
    )
    assert held not in support_labels.case_ids("0")
    assert support_labels.case_count == dict(EXPECTED_CASE_COUNTS_BY_CENTER)["0"] - 1
    journal.close_support_scope(support_capability)

    decision_hash = _digest("sealed-decisions")
    journal.seal_decisions(
        decision_seal_hash=decision_hash, route_count=EXPECTED_CASE_COUNT
    )
    terminal_capability = journal.open_terminal_scope(
        scope_id="terminal",
        terminal_identity_hash=decoder.terminal_identity_hash(),
        decision_seal_hash=decision_hash,
    )
    terminal = decoder.decode_terminal(journal, terminal_capability)
    assert terminal.labels.shape == (EXPECTED_CASE_COUNT,)
    assert set(terminal.centers.tolist()) == set(CENTERS)
    assert terminal.labels.flags.writeable is False
    with pytest.raises(TypeError):
        pickle.dumps(terminal)
    journal.close_terminal_scope(terminal_capability)
