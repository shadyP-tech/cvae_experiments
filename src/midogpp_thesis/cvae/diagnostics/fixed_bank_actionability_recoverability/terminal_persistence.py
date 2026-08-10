"""Terminal table/report publication and its non-replayable label commit."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from .artifact_serialization import payload, persist_payload, persist_rows
from .reports import publication_decision_payload


_TERMINAL_TABLES = {
    "tables/terminal_case_confusions.csv",
    "tables/terminal_center_metrics.csv",
    "tables/terminal_method_summary.csv",
    "tables/terminal_contrasts.csv",
    "tables/oracle_rank_metrics.csv",
    "tables/complementarity.csv",
    "tables/rank_stability.csv",
    "tables/permutation_metrics.csv",
}


def persist_label_capability_report(
    root: Path, value: Mapping[str, object]
) -> None:
    """Commit terminal label access before any terminal computation starts."""

    persist_payload(root / "reports/label_capability_report.json", value)
    if read_json(root / "reports/label_capability_report.json") != value:
        raise ProtocolError("Label-capability commit marker drifted.")


def persist_postseal_results(
    root: Path,
    *,
    evaluation: object,
    capability_report: Mapping[str, object],
    leakage_report: Mapping[str, object],
    runtime_summary: Mapping[str, object],
) -> None:
    evaluation_payload = payload(evaluation)
    for member, rows in _terminal_tables(evaluation).items():
        persist_rows(root / member, rows)
    persist_payload(root / "reports/label_capability_report.json", capability_report)
    persist_payload(root / "reports/leakage_report.json", leakage_report)
    persist_payload(
        root / "reports/publication_decision.json",
        publication_decision_payload(evaluation_payload),
    )
    persist_payload(root / "reports/runtime_summary.json", runtime_summary)
    # Publish last: this is the durable terminal-phase commit marker.
    persist_payload(
        root / "manifests/sealed_terminal_evaluation.json", evaluation_payload
    )


def _terminal_tables(
    evaluation: object,
) -> Mapping[str, Sequence[Mapping[str, object]]]:
    if hasattr(evaluation, "table_rows"):
        raw = getattr(evaluation, "table_rows")()
        if isinstance(raw, Mapping) and set(raw) == _TERMINAL_TABLES:
            return raw
    raise ProtocolError("Terminal evaluation table inventory drifted.")


__all__ = ("persist_label_capability_report", "persist_postseal_results")
