"""Small ordered-hash barriers for route and whole-policy decisions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ...protocol import ProtocolError
from .hashing import canonical_hash, require_sha256
from .policy_regret import CenterCandidatePolicy


def seal_case_decisions(
    case_hashes: Mapping[tuple[str, str], str],
) -> dict[str, object]:
    rows = tuple(sorted(((str(center), str(case), require_sha256(digest, "case_decision_hash")) for (center, case), digest in case_hashes.items())))
    if not rows or len({(center, case) for center, case, _digest in rows}) != len(rows):
        raise ProtocolError("PCSI-PARC case decision barrier drifted.")
    payload = {
        "schema_version": "fixed_bank_pcsi_parc_case_decision_barrier_v1",
        "case_decision_count": len(rows),
        "ordered_case_decision_hash": canonical_hash([list(row) for row in rows]),
        "terminal_labels_used": False,
    }
    return {**payload, "decision_barrier_hash": canonical_hash(payload)}


def seal_policy_menu(
    policies: Sequence[CenterCandidatePolicy],
    *,
    decision_barrier_hash: str,
) -> dict[str, object]:
    rows = tuple(sorted(policies, key=lambda row: (row.policy_id, row.center, row.geometry_id)))
    if not rows:
        raise ProtocolError("PCSI-PARC cannot seal an empty policy menu.")
    payload = {
        "schema_version": "fixed_bank_pcsi_parc_policy_menu_seal_v1",
        "decision_barrier_hash": require_sha256(decision_barrier_hash, "decision_barrier_hash"),
        "policy_count": len(rows),
        "ordered_policy_hash": canonical_hash(
            [
                [row.policy_id, row.center, row.geometry_id, row.policy_seal_hash]
                for row in rows
            ]
        ),
        "terminal_labels_used": False,
    }
    return {**payload, "policy_menu_seal_hash": canonical_hash(payload)}


__all__ = ("seal_case_decisions", "seal_policy_menu")
