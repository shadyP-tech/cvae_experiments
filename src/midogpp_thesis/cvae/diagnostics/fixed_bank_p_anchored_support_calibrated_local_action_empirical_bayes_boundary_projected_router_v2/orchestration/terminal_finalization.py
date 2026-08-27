"""Aggregate-only terminal scoring and immutable final artifact sealing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import time

from ..artifacts import persist_final_content_index
from ..artifacts.io import atomic_json
from ..capability_scoring import score_terminal_capability
from ..execution import MemmapReference, OuterCenterResult
from ..fresh_process_validation import require_two_fresh_process_attestations
from ..identity import P_METHOD_ID
from ..label_capabilities import LabelCapabilityJournal
from ..manifest_labels import ManifestLabelDecoder
from ..reports import (
    leakage_report_payload,
    publication_decision_payload,
    runtime_report_payload,
    validation_report_payload,
)
from ..run_state import mark_complete, read_run_state, transition_run
from ..terminal import persist_terminal_aggregate
from ..validation import validate_final_bundle
from ..workstation import canonical_workstation_payload
from .result_assembly import assemble_method_probabilities, read_route_chunk


@dataclass(frozen=True, slots=True)
class TerminalPhaseResult:
    terminal_payload: Mapping[str, object]
    final_journal: Mapping[str, object]
    elapsed_seconds: float


def score_terminal_phase(
    root: Path,
    *,
    results: Sequence[OuterCenterResult],
    journal: LabelCapabilityJournal,
    decoder: ManifestLabelDecoder,
    decision_seal_hash: str,
) -> TerminalPhaseResult:
    """Reload sealed decision bytes, open terminal scope, and persist aggregates."""

    started = time.monotonic()
    route_payloads = {
        result.target_center: read_route_chunk(root, result) for result in results
    }
    method_probabilities, probability_hashes = assemble_method_probabilities(
        route_payloads
    )
    terminal_capability = journal.open_terminal_scope(
        scope_id="terminal:aggregate-only",
        terminal_identity_hash=decoder.terminal_identity_hash(),
        decision_seal_hash=decision_seal_hash,
    )
    transition_run(
        root,
        "TERMINAL_OPEN",
        expected_phase="PRETERMINAL_ATTESTED",
        evidence_hash=terminal_capability.event_hash,
    )
    terminal_view = decoder.decode_terminal(journal, terminal_capability)
    terminal_aggregate = score_terminal_capability(
        journal,
        terminal_capability,
        terminal_view,
        method_probabilities,
        expected_probability_hashes=probability_hashes,
        protected_method_id=P_METHOD_ID,
        decision_seal_hash=decision_seal_hash,
    )
    del terminal_view
    journal.close_terminal_scope(terminal_capability)
    terminal_payload = persist_terminal_aggregate(root, terminal_aggregate)
    transition_run(
        root,
        "TERMINAL_SCORED",
        expected_phase="TERMINAL_OPEN",
        evidence_hash=str(terminal_payload["terminal_seal_hash"]),
    )
    return TerminalPhaseResult(
        terminal_payload=terminal_payload,
        final_journal=journal.audit_payload(),
        elapsed_seconds=time.monotonic() - started,
    )


def finalize_terminal_run(
    root: Path,
    *,
    protocol_hash: str,
    decision_seal_hash: str,
    preterminal: Mapping[str, object],
    terminal_phase: TerminalPhaseResult,
    results: Sequence[OuterCenterResult],
    phase_timings_seconds: Mapping[str, float],
    memmap_references: Sequence[MemmapReference],
) -> None:
    """Write reconstructive reports, seal the final index, and mark complete."""

    terminal_payload = terminal_phase.terminal_payload
    final_journal = terminal_phase.final_journal
    current_state = read_run_state(root)
    runtime = runtime_report_payload(
        run_state=current_state,
        workstation_plan=canonical_workstation_payload(),
        center_results=[result.to_payload() for result in results],
        phase_timings_seconds=phase_timings_seconds,
        memmap_reference_hashes=[row.reference_hash for row in memmap_references],
    )
    leakage = leakage_report_payload(
        protocol_hash=protocol_hash,
        preterminal_aggregate_seal_hash=str(preterminal["aggregate_seal_hash"]),
        decision_seal_hash=decision_seal_hash,
        preterminal_journal_hash=str(preterminal["label_capability_journal_hash"]),
        final_journal_hash=str(final_journal["audit_hash"]),
        terminal_seal_hash=str(terminal_payload["terminal_seal_hash"]),
    )
    publication = publication_decision_payload(
        terminal_seal_hash=str(terminal_payload["terminal_seal_hash"]),
        diagnostic_summary={
            "method_count": terminal_payload["method_count"],
            "comparisons_to_protected_p": terminal_payload[
                "comparisons_to_protected_p"
            ],
            "all_outer_admission_gates_passed": True,
            "consumed_test_surface": True,
        },
    )
    atomic_json(root / "reports/runtime.json", runtime)
    atomic_json(root / "reports/leakage.json", leakage)
    atomic_json(root / "reports/publication_decision.json", publication)

    final_index = persist_final_content_index(
        root,
        terminal_seal_hash=str(terminal_payload["terminal_seal_hash"]),
        terminal_metrics_hash=str(terminal_payload["terminal_metrics_hash"]),
        label_capability_journal=final_journal,
        required_members=(
            "reports/runtime.json",
            "reports/leakage.json",
            "reports/publication_decision.json",
        ),
    )
    transition_run(
        root,
        "FINAL_INDEX_SEALED",
        expected_phase="TERMINAL_SCORED",
        evidence_hash=str(final_index["aggregate_seal_hash"]),
    )
    final_attestation = require_two_fresh_process_attestations(root, phase="final")
    transition_run(
        root,
        "FINAL_ATTESTED",
        expected_phase="FINAL_INDEX_SEALED",
        evidence_hash=str(final_attestation["attestation_hash"]),
    )
    checks = validate_final_bundle(root, no_refit=True, require_fresh_attestation=True)
    validation = validation_report_payload(
        checks,
        fresh_process_attestation_hash=str(final_attestation["attestation_hash"]),
    )
    atomic_json(root / "reports/validation.json", validation)
    mark_complete(root, final_validation_hash=str(validation["validation_report_hash"]))


__all__ = (
    "TerminalPhaseResult",
    "finalize_terminal_run",
    "score_terminal_phase",
)
