"""Only capability-bearing label-to-science bridges for SCALE-BP v2.

The numerical utility and aggregate metric kernels deliberately remain small
and testable.  Production orchestration must enter them through this module so
an active donor, support, or post-seal terminal capability is checked at the
last possible moment before a label array reaches a numerical kernel.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from .label_capabilities import DONOR, SUPPORT, TERMINAL, LabelCapability
from .manifest_labels import ScopedCaseLabels, TerminalLabelVector
from .physical.contracts import ACTION_IDS
from .protocol import GovernanceError
from .terminal.contracts import TerminalAggregate
from .terminal.scoring import score_sealed_method_probabilities
from .utility.actions import ActionRectangle
from .utility.metrics import (
    CenterMetricDenominators,
    ScoredActionRectangle,
    center_denominators,
    score_action_rectangle,
)


class ActiveCapabilityJournal(Protocol):
    """Small common surface shared by the parent and spawned-worker journals."""

    def assert_active(
        self,
        capability: LabelCapability,
        *,
        kind: str,
        scope_id: str | None = None,
    ) -> None: ...


def scoped_center_denominators(
    journal: ActiveCapabilityJournal,
    capability: LabelCapability,
    scoped_labels: ScopedCaseLabels,
    *,
    center: object,
) -> CenterMetricDenominators:
    """Derive denominators without exposing the scoped arrays to orchestration."""

    _assert_preterminal_binding(journal, capability, scoped_labels)
    center_id = str(center)
    case_ids = scoped_labels.case_ids(center_id)
    return center_denominators(
        {
            case_id: scoped_labels.labels_for_case(center_id, case_id)
            for case_id in case_ids
        }
    )


def score_scoped_action_rectangle(
    journal: ActiveCapabilityJournal,
    capability: LabelCapability,
    scoped_labels: ScopedCaseLabels,
    rectangle: ActionRectangle,
) -> ScoredActionRectangle:
    """Score all six actions with denominators derived from the same scope."""

    _assert_preterminal_binding(journal, capability, scoped_labels)
    if not isinstance(rectangle, ActionRectangle):
        raise GovernanceError("SCALE-BP v2 utility scoring received a foreign rectangle.")
    case_labels = scoped_labels.labels_for_case(
        rectangle.target_center, rectangle.case_id
    )
    denominators = scoped_center_denominators(
        journal,
        capability,
        scoped_labels,
        center=rectangle.target_center,
    )
    scored = score_action_rectangle(
        rectangle,
        case_labels,
        denominators=denominators,
        label_scope_hash=capability.scope_hash,
    )
    if tuple(value.action_id for value in scored.values) != ACTION_IDS:
        raise GovernanceError("SCALE-BP v2 capability-scored action surface drifted.")
    return scored


def score_terminal_capability(
    journal: ActiveCapabilityJournal,
    capability: LabelCapability,
    terminal_labels: TerminalLabelVector,
    method_probabilities: Mapping[str, object],
    *,
    expected_probability_hashes: Mapping[str, str],
    protected_method_id: str,
    decision_seal_hash: str,
) -> TerminalAggregate:
    """Score sealed probabilities while a post-decision terminal token is active."""

    journal.assert_active(capability, kind=TERMINAL)
    if not isinstance(terminal_labels, TerminalLabelVector):
        raise GovernanceError("SCALE-BP v2 terminal scoring requires its label view.")
    if (
        terminal_labels.scope_hash != capability.scope_hash
        or capability.decision_seal_hash is None
        or capability.decision_seal_hash != str(decision_seal_hash)
    ):
        raise GovernanceError("SCALE-BP v2 terminal label/decision binding drifted.")
    return score_sealed_method_probabilities(
        method_probabilities,
        expected_probability_hashes=expected_probability_hashes,
        labels=terminal_labels.labels,
        centers=terminal_labels.centers,
        protected_method_id=str(protected_method_id),
        decision_seal_hash=str(decision_seal_hash),
    )


def _assert_preterminal_binding(
    journal: ActiveCapabilityJournal,
    capability: LabelCapability,
    scoped_labels: ScopedCaseLabels,
) -> None:
    if not isinstance(scoped_labels, ScopedCaseLabels):
        raise GovernanceError("SCALE-BP v2 utility scoring requires scoped labels.")
    if scoped_labels.kind not in {DONOR, SUPPORT}:
        raise GovernanceError("SCALE-BP v2 preterminal label kind drifted.")
    journal.assert_active(capability, kind=scoped_labels.kind)
    if capability.scope_hash != scoped_labels.scope_hash:
        raise GovernanceError("SCALE-BP v2 label capability/scope binding drifted.")


__all__ = (
    "ActiveCapabilityJournal",
    "score_scoped_action_rectangle",
    "score_terminal_capability",
    "scoped_center_denominators",
)
