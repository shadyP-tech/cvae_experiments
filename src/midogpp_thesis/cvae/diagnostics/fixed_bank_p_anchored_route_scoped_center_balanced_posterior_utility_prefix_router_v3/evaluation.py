"""Terminal evaluation after the aggregate seal opens target labels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from .constants import (
    COMPOSED_POLICY_IDS,
    ENDPOINT_METHOD_IDS,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
)
from .contracts import BinaryLabel
from .hashing import canonical_hash, require_sha256
from .terminal_metrics import score_methods, selection_aware_center_sign_flip


@dataclass(frozen=True)
class TerminalResult:
    method_rows: tuple[Mapping[str, object], ...]
    center_rows: tuple[Mapping[str, object], ...]
    oracle_rows: tuple[Mapping[str, object], ...]
    terminal_seal_hash: str
    diagnostic_summary: Mapping[str, object]
    result_hash: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.terminal_seal_hash, "terminal_seal_hash")
        object.__setattr__(
            self,
            "result_hash",
            canonical_hash(
                {
                    "schema_version": "fixed_bank_cbpupr_terminal_result_v1",
                    "method_rows": list(self.method_rows),
                    "center_rows": list(self.center_rows),
                    "oracle_rows": list(self.oracle_rows),
                    "terminal_seal_hash": self.terminal_seal_hash,
                    "diagnostic_summary": dict(self.diagnostic_summary),
                    "raw_labels_persisted": False,
                }
            ),
        )


def evaluate_terminal(
    *,
    probabilities: Mapping[str, Mapping[str, Mapping[str, tuple[float, ...]]]],
    sample_ids: Mapping[str, Mapping[str, tuple[str, ...]]],
    labels: Sequence[BinaryLabel],
    aggregate_seal_hash: str,
    diagnostic_summary: Mapping[str, object],
) -> TerminalResult:
    require_sha256(aggregate_seal_hash, "aggregate_seal_hash")
    label_rows = tuple(labels)
    label_map = {row.key: row.value for row in label_rows}
    method_order = (*ENDPOINT_METHOD_IDS, *COMPOSED_POLICY_IDS)
    if (
        len(label_map) != len(label_rows)
        or {row.scope for row in label_rows} != {"target_terminal_after_aggregate_seal"}
        or tuple(probabilities) != method_order
    ):
        raise ProtocolError("CBPUPR terminal capability or method menu drifted.")
    method_rows, center_rows, oracle_rows, center_metrics = score_methods(
        probabilities,
        sample_ids,
        label_map,
        method_order=method_order,
    )
    selection_control = selection_aware_center_sign_flip(center_metrics)
    summary = dict(diagnostic_summary)
    summary.update(
        {
            "selection_aware_center_sign_flip": dict(selection_control),
            "selection_aware_observed_max_statistic": selection_control[
                "observed_max_statistic"
            ],
            "selection_aware_descriptive_randomization_p_value": (
                selection_control[
                    "selection_aware_descriptive_randomization_p_value"
                ]
            ),
            "formal_claim_authorized": False,
            "nominal_significance_claimed": False,
        }
    )
    seal_hash = canonical_hash(
        {
            "schema_version": "fixed_bank_cbpupr_terminal_seal_v1",
            "aggregate_seal_hash": aggregate_seal_hash,
            "label_identity_hash": canonical_hash([list(row.key) for row in label_rows]),
            "method_rows": list(method_rows),
            "selection_aware_center_sign_flip": dict(selection_control),
            "publication_status": PUBLICATION_STATUS,
            "terminal_decision": TERMINAL_DECISION,
        }
    )
    return TerminalResult(
        method_rows,
        center_rows,
        oracle_rows,
        seal_hash,
        summary,
    )


__all__ = ("TerminalResult", "evaluate_terminal")
