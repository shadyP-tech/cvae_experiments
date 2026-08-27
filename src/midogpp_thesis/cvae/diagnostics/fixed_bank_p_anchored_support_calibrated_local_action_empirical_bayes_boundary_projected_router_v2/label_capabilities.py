"""Hash-scoped label capabilities for the one-shot SCALE-BP v2 lifecycle.

The journal never stores labels or row-level label values.  It records only
structural scope identifiers and full-width hashes, and enforces that terminal
evaluation labels cannot open before every decision is durably sealed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
from pathlib import Path

from .hashing import canonical_hash, require_sha256
from .identity import (
    CENTERS,
    EXPECTED_CASE_COUNT,
    EXPECTED_CASE_COUNTS_BY_CENTER,
    EXPECTED_TEST_MANIFEST_SHA256,
    EXPERIMENT_ID,
    GovernanceError,
)


JOURNAL_SCHEMA = "scale_bp_v2_label_capability_journal_v1"
EVENT_SCHEMA = "scale_bp_v2_label_capability_event_v1"
CAPABILITY_SCHEMA = "scale_bp_v2_label_capability_v1"
DELEGATION_SCHEMA = "scale_bp_v2_worker_label_delegation_v1"
WORKER_JOURNAL_SCHEMA = "scale_bp_v2_worker_label_journal_v1"
WORKER_EVENT_SCHEMA = "scale_bp_v2_worker_label_event_v1"
WORKER_AUDIT_SCHEMA = "scale_bp_v2_worker_label_audit_v1"

PRETERMINAL = "PRETERMINAL"
DECISIONS_SEALED = "DECISIONS_SEALED"
TERMINAL_OPEN = "TERMINAL_OPEN"
CLOSED = "CLOSED"

DONOR = "DONOR"
SUPPORT = "SUPPORT"
TERMINAL = "TERMINAL"


@dataclass(frozen=True, slots=True)
class LabelCapability:
    """Unforgeable-by-journal-comparison token for one active label scope."""

    journal_id: str
    kind: str
    scope_id: str
    scope_hash: str
    sequence: int
    event_hash: str
    decision_seal_hash: str | None = None

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": CAPABILITY_SCHEMA,
            "journal_id": self.journal_id,
            "kind": self.kind,
            "scope_id": self.scope_id,
            "scope_hash": self.scope_hash,
            "sequence": self.sequence,
            "event_hash": self.event_hash,
            "decision_seal_hash": self.decision_seal_hash,
        }


@dataclass(frozen=True, slots=True)
class WorkerSupportScope:
    """One parent-approved H\\c support/evaluation partition, represented by hashes."""

    held_case_id: str
    support_identity_hash: str
    evaluation_identity_hash: str
    scope_hash: str = field(init=False)

    def __post_init__(self) -> None:
        held = str(self.held_case_id)
        support = require_sha256(
            self.support_identity_hash, "delegated support identity hash"
        )
        evaluation = require_sha256(
            self.evaluation_identity_hash, "delegated evaluation identity hash"
        )
        if not held or support == evaluation:
            raise GovernanceError("SCALE-BP v2 delegated support scope drifted.")
        object.__setattr__(self, "held_case_id", held)
        object.__setattr__(self, "support_identity_hash", support)
        object.__setattr__(self, "evaluation_identity_hash", evaluation)
        object.__setattr__(
            self,
            "scope_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_v2_worker_support_scope_v1",
                    "held_case_id": held,
                    "support_identity_hash": support,
                    "evaluation_identity_hash": evaluation,
                    "support_evaluation_disjoint": True,
                }
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "held_case_id": self.held_case_id,
            "support_identity_hash": self.support_identity_hash,
            "evaluation_identity_hash": self.evaluation_identity_hash,
            "scope_hash": self.scope_hash,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "WorkerSupportScope":
        scope = cls(
            held_case_id=str(payload.get("held_case_id", "")),
            support_identity_hash=str(payload.get("support_identity_hash", "")),
            evaluation_identity_hash=str(
                payload.get("evaluation_identity_hash", "")
            ),
        )
        if dict(payload) != scope.to_payload():
            raise GovernanceError("SCALE-BP v2 delegated support payload drifted.")
        return scope


@dataclass(frozen=True, slots=True)
class WorkerLabelDelegation:
    """Plain-pickle capability binding one spawned task to exact label scopes."""

    parent_journal_id: str
    run_identity_hash: str
    task_id: str
    task_hash: str
    outer_center: str
    manifest_path: str
    manifest_sha256: str
    donor_identity_hash: str
    route_scopes: tuple[WorkerSupportScope, ...]
    delegation_hash: str = field(init=False)

    def __post_init__(self) -> None:
        parent = require_sha256(self.parent_journal_id, "parent journal id")
        run_hash = require_sha256(self.run_identity_hash, "delegated run identity")
        task_id = str(self.task_id)
        task_hash = require_sha256(self.task_hash, "delegated task hash")
        outer = str(self.outer_center)
        manifest = str(self.manifest_path)
        manifest_hash = require_sha256(
            self.manifest_sha256, "delegated manifest hash"
        )
        donor_hash = require_sha256(
            self.donor_identity_hash, "delegated donor identity hash"
        )
        scopes = tuple(self.route_scopes)
        expected_count = dict(EXPECTED_CASE_COUNTS_BY_CENTER).get(outer)
        if (
            not task_id
            or outer not in CENTERS
            or not Path(manifest).is_absolute()
            or manifest_hash != EXPECTED_TEST_MANIFEST_SHA256
            or expected_count is None
            or len(scopes) != expected_count
            or len({scope.held_case_id for scope in scopes}) != len(scopes)
            or len({scope.scope_hash for scope in scopes}) != len(scopes)
        ):
            raise GovernanceError("SCALE-BP v2 worker label delegation drifted.")
        object.__setattr__(self, "parent_journal_id", parent)
        object.__setattr__(self, "run_identity_hash", run_hash)
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "task_hash", task_hash)
        object.__setattr__(self, "outer_center", outer)
        object.__setattr__(self, "manifest_path", manifest)
        object.__setattr__(self, "manifest_sha256", manifest_hash)
        object.__setattr__(self, "donor_identity_hash", donor_hash)
        object.__setattr__(self, "route_scopes", scopes)
        object.__setattr__(self, "delegation_hash", canonical_hash(self._body()))

    def _body(self) -> dict[str, object]:
        return {
            "schema_version": DELEGATION_SCHEMA,
            "parent_journal_id": self.parent_journal_id,
            "run_identity_hash": self.run_identity_hash,
            "task_id": self.task_id,
            "task_hash": self.task_hash,
            "outer_center": self.outer_center,
            "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
            "donor_identity_hash": self.donor_identity_hash,
            "donor_centers": [center for center in CENTERS if center != self.outer_center],
            "route_scopes": [scope.to_payload() for scope in self.route_scopes],
            "raw_labels_embedded": False,
            "label_arrays_embedded": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._body(), "delegation_hash": self.delegation_hash}

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "WorkerLabelDelegation":
        raw_scopes = payload.get("route_scopes")
        if not isinstance(raw_scopes, list):
            raise GovernanceError("SCALE-BP v2 worker route scopes are malformed.")
        delegation = cls(
            parent_journal_id=str(payload.get("parent_journal_id", "")),
            run_identity_hash=str(payload.get("run_identity_hash", "")),
            task_id=str(payload.get("task_id", "")),
            task_hash=str(payload.get("task_hash", "")),
            outer_center=str(payload.get("outer_center", "")),
            manifest_path=str(payload.get("manifest_path", "")),
            manifest_sha256=str(payload.get("manifest_sha256", "")),
            donor_identity_hash=str(payload.get("donor_identity_hash", "")),
            route_scopes=tuple(
                WorkerSupportScope.from_payload(scope)
                for scope in raw_scopes
                if isinstance(scope, Mapping)
            ),
        )
        if dict(payload) != delegation.to_payload():
            raise GovernanceError("SCALE-BP v2 worker delegation payload drifted.")
        return delegation


@dataclass(frozen=True, slots=True)
class WorkerCapabilityAudit:
    """Compact, primitive-only result returned across the spawn boundary."""

    delegation_hash: str
    worker_journal_id: str
    task_id: str
    task_hash: str
    outer_center: str
    donor_scope_count: int
    support_scope_count: int
    event_count: int
    event_log_hash: str
    decision_fragment_hash: str
    audit_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for value, role in (
            (self.delegation_hash, "worker delegation hash"),
            (self.worker_journal_id, "worker journal id"),
            (self.task_hash, "worker task hash"),
            (self.event_log_hash, "worker event-log hash"),
            (self.decision_fragment_hash, "worker decision fragment hash"),
        ):
            require_sha256(value, role)
        expected = dict(EXPECTED_CASE_COUNTS_BY_CENTER).get(str(self.outer_center))
        if (
            not str(self.task_id)
            or expected is None
            or self.donor_scope_count != 1
            or self.support_scope_count != expected
            or self.event_count != 2 * (1 + expected)
        ):
            raise GovernanceError("SCALE-BP v2 worker capability audit drifted.")
        object.__setattr__(self, "audit_hash", canonical_hash(self._body()))

    def _body(self) -> dict[str, object]:
        return {
            "schema_version": WORKER_AUDIT_SCHEMA,
            "delegation_hash": self.delegation_hash,
            "worker_journal_id": self.worker_journal_id,
            "task_id": self.task_id,
            "task_hash": self.task_hash,
            "outer_center": self.outer_center,
            "donor_scope_count": self.donor_scope_count,
            "support_scope_count": self.support_scope_count,
            "event_count": self.event_count,
            "event_log_hash": self.event_log_hash,
            "decision_fragment_hash": self.decision_fragment_hash,
            "raw_labels_returned": False,
            "raw_labels_persisted": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._body(), "audit_hash": self.audit_hash}

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "WorkerCapabilityAudit":
        try:
            audit = cls(
                delegation_hash=str(payload["delegation_hash"]),
                worker_journal_id=str(payload["worker_journal_id"]),
                task_id=str(payload["task_id"]),
                task_hash=str(payload["task_hash"]),
                outer_center=str(payload["outer_center"]),
                donor_scope_count=int(payload["donor_scope_count"]),
                support_scope_count=int(payload["support_scope_count"]),
                event_count=int(payload["event_count"]),
                event_log_hash=str(payload["event_log_hash"]),
                decision_fragment_hash=str(payload["decision_fragment_hash"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise GovernanceError("SCALE-BP v2 worker audit is malformed.") from exc
        if dict(payload) != audit.to_payload():
            raise GovernanceError("SCALE-BP v2 worker audit payload drifted.")
        return audit


class LabelCapabilityJournal:
    """Single-process capability state machine with a deterministic hash chain."""

    def __init__(self, run_identity_hash: str, *, experiment_id: str = EXPERIMENT_ID):
        if experiment_id != EXPERIMENT_ID:
            raise GovernanceError("SCALE-BP v2 journal experiment identity drifted.")
        self._run_identity_hash = require_sha256(
            run_identity_hash, "label-journal run identity"
        )
        self._journal_id = canonical_hash(
            {
                "schema_version": JOURNAL_SCHEMA,
                "experiment_id": experiment_id,
                "run_identity_hash": self._run_identity_hash,
            }
        )
        self._phase = PRETERMINAL
        self._events: list[dict[str, object]] = []
        self._active: LabelCapability | None = None
        self._used_scope_ids: set[str] = set()
        self._closed_counts = {DONOR: 0, SUPPORT: 0, TERMINAL: 0}
        self._decision_seal_hash: str | None = None
        self._terminal_opened = False
        self._delegations: dict[str, WorkerLabelDelegation] = {}
        self._worker_audits: dict[str, WorkerCapabilityAudit] = {}

    @property
    def journal_id(self) -> str:
        return self._journal_id

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def active_capability(self) -> LabelCapability | None:
        return self._active

    @property
    def decision_seal_hash(self) -> str | None:
        return self._decision_seal_hash

    def open_donor_scope(
        self,
        *,
        scope_id: str,
        outer_center: str,
        donor_centers: Sequence[str],
        row_identity_hash: str,
    ) -> LabelCapability:
        """Open labels for donor centers only, excluding the outer center H."""

        self._require_preterminal_idle()
        if self._delegations:
            raise GovernanceError(
                "SCALE-BP v2 cannot mix parent-local and delegated labels."
            )
        outer = str(outer_center)
        donors = tuple(str(center) for center in donor_centers)
        expected = tuple(center for center in CENTERS if center != outer)
        if outer not in CENTERS or donors != expected:
            raise GovernanceError(
                "SCALE-BP v2 donor capability must be exact ordered J != H."
            )
        row_hash = require_sha256(row_identity_hash, "donor row-identity hash")
        scope = self._new_scope_id(scope_id)
        payload = {
            "outer_center": outer,
            "donor_centers": list(donors),
            "row_identity_hash": row_hash,
            "outer_center_labels_available": False,
            "target_expert_available": False,
        }
        return self._open(DONOR, scope, canonical_hash(payload), payload)

    def close_donor_scope(self, capability: LabelCapability) -> None:
        self._close(capability, DONOR)

    def open_support_scope(
        self,
        *,
        scope_id: str,
        target_center: str,
        held_case_id: str,
        support_identity_hash: str,
        evaluation_identity_hash: str,
    ) -> LabelCapability:
        """Open H\\c support labels while keeping held-case c labels closed."""

        self._require_preterminal_idle()
        if self._delegations:
            raise GovernanceError(
                "SCALE-BP v2 cannot mix parent-local and delegated labels."
            )
        target = str(target_center)
        held = str(held_case_id)
        support_hash = require_sha256(
            support_identity_hash, "support identity hash"
        )
        evaluation_hash = require_sha256(
            evaluation_identity_hash, "evaluation identity hash"
        )
        if target not in CENTERS or not held or support_hash == evaluation_hash:
            raise GovernanceError("SCALE-BP v2 support/evaluation scope drifted.")
        scope = self._new_scope_id(scope_id)
        payload = {
            "target_center": target,
            "held_case_id": held,
            "support_identity_hash": support_hash,
            "evaluation_identity_hash": evaluation_hash,
            "held_case_labels_available": False,
            "support_updates_global_state": False,
            "support_updates_source_experts": False,
            "support_tunes_hyperparameters": False,
        }
        return self._open(SUPPORT, scope, canonical_hash(payload), payload)

    def close_support_scope(self, capability: LabelCapability) -> None:
        self._close(capability, SUPPORT)

    def delegate_outer_worker(
        self,
        *,
        task_id: str,
        outer_center: str,
        task_hash: str,
        manifest_path: str | Path,
        manifest_sha256: str,
        donor_identity_hash: str,
        route_scopes: Sequence[WorkerSupportScope | Mapping[str, object]],
    ) -> WorkerLabelDelegation:
        """Grant exact hash-only H and H\\c scopes to one spawn worker."""

        self._require_preterminal_idle()
        if self._closed_counts[DONOR] or self._closed_counts[SUPPORT]:
            raise GovernanceError(
                "SCALE-BP v2 cannot mix parent-local and delegated labels."
            )
        normalized = tuple(
            scope
            if isinstance(scope, WorkerSupportScope)
            else WorkerSupportScope.from_payload(scope)
            for scope in route_scopes
        )
        delegation = WorkerLabelDelegation(
            parent_journal_id=self._journal_id,
            run_identity_hash=self._run_identity_hash,
            task_id=str(task_id),
            task_hash=str(task_hash),
            outer_center=str(outer_center),
            manifest_path=str(manifest_path),
            manifest_sha256=str(manifest_sha256),
            donor_identity_hash=str(donor_identity_hash),
            route_scopes=normalized,
        )
        if (
            delegation.outer_center in self._delegations
            or delegation.task_id
            in {existing.task_id for existing in self._delegations.values()}
        ):
            raise GovernanceError("SCALE-BP v2 worker delegation is duplicated.")
        self._delegations[delegation.outer_center] = delegation
        self._append_event(
            "DELEGATE_OUTER_WORKER",
            {
                "outer_center": delegation.outer_center,
                "task_id": delegation.task_id,
                "task_hash": delegation.task_hash,
                "delegation_hash": delegation.delegation_hash,
                "donor_identity_hash": delegation.donor_identity_hash,
                "support_scope_manifest_hash": canonical_hash(
                    [scope.to_payload() for scope in delegation.route_scopes]
                ),
                "support_scope_count": len(delegation.route_scopes),
                "raw_labels_embedded": False,
            },
        )
        return delegation

    def accept_worker_audit(
        self,
        delegation: WorkerLabelDelegation | Mapping[str, object],
        audit: WorkerCapabilityAudit | Mapping[str, object],
    ) -> None:
        """Validate and bind one compact child audit before aggregate sealing."""

        self._require_preterminal_idle()
        delegated = (
            delegation
            if isinstance(delegation, WorkerLabelDelegation)
            else WorkerLabelDelegation.from_payload(delegation)
        )
        observed = (
            audit
            if isinstance(audit, WorkerCapabilityAudit)
            else WorkerCapabilityAudit.from_payload(audit)
        )
        expected = self._delegations.get(delegated.outer_center)
        if (
            expected != delegated
            or delegated.parent_journal_id != self._journal_id
            or observed.delegation_hash != delegated.delegation_hash
            or observed.task_id != delegated.task_id
            or observed.task_hash != delegated.task_hash
            or observed.outer_center != delegated.outer_center
            or observed.support_scope_count != len(delegated.route_scopes)
            or delegated.outer_center in self._worker_audits
        ):
            raise GovernanceError("SCALE-BP v2 worker capability audit escaped scope.")
        self._worker_audits[delegated.outer_center] = observed
        self._append_event(
            "ACCEPT_OUTER_WORKER_AUDIT",
            {
                "outer_center": delegated.outer_center,
                "task_id": delegated.task_id,
                "delegation_hash": delegated.delegation_hash,
                "worker_audit_hash": observed.audit_hash,
                "worker_event_log_hash": observed.event_log_hash,
                "decision_fragment_hash": observed.decision_fragment_hash,
                "support_scope_count": observed.support_scope_count,
                "raw_labels_returned": False,
            },
        )

    def seal_decisions(self, *, decision_seal_hash: str, route_count: int) -> None:
        """Irreversibly bind all 218 decisions before terminal labels may open."""

        self._require_preterminal_idle()
        seal = require_sha256(decision_seal_hash, "preterminal decision seal")
        local_complete = (
            not self._delegations
            and self._closed_counts[DONOR] > 0
            and self._closed_counts[SUPPORT] > 0
        )
        delegated_complete = (
            set(self._delegations) == set(CENTERS)
            and set(self._worker_audits) == set(CENTERS)
            and sum(audit.support_scope_count for audit in self._worker_audits.values())
            == EXPECTED_CASE_COUNT
        )
        if (
            isinstance(route_count, bool)
            or int(route_count) != EXPECTED_CASE_COUNT
            or not (local_complete or delegated_complete)
        ):
            raise GovernanceError(
                "SCALE-BP v2 cannot seal incomplete preterminal label scopes."
            )
        self._decision_seal_hash = seal
        self._append_event(
            "SEAL_DECISIONS",
            {
                "decision_seal_hash": seal,
                "route_count": int(route_count),
                "donor_scope_count": self._closed_counts[DONOR],
                "support_scope_count": self._closed_counts[SUPPORT],
                "delegated_worker_count": len(self._worker_audits),
                "delegated_worker_audit_manifest_hash": (
                    canonical_hash(
                        {
                            center: self._worker_audits[center].to_payload()
                            for center in CENTERS
                        }
                    )
                    if delegated_complete
                    else None
                ),
                "terminal_labels_opened": False,
            },
        )
        self._phase = DECISIONS_SEALED

    def open_terminal_scope(
        self,
        *,
        scope_id: str,
        terminal_identity_hash: str,
        decision_seal_hash: str,
    ) -> LabelCapability:
        """Open terminal labels once and only for the sealed evaluation universe."""

        if self._phase != DECISIONS_SEALED or self._active is not None:
            raise GovernanceError(
                "SCALE-BP v2 terminal labels require an idle sealed journal."
            )
        supplied_seal = require_sha256(
            decision_seal_hash, "terminal decision-seal binding"
        )
        if (
            self._terminal_opened
            or supplied_seal != self._decision_seal_hash
        ):
            raise GovernanceError("SCALE-BP v2 terminal seal binding drifted.")
        terminal_hash = require_sha256(
            terminal_identity_hash, "terminal identity hash"
        )
        scope = self._new_scope_id(scope_id)
        payload = {
            "terminal_identity_hash": terminal_hash,
            "decision_seal_hash": supplied_seal,
            "may_update_preterminal_state": False,
            "may_reselect_actions": False,
        }
        capability = self._open(
            TERMINAL,
            scope,
            canonical_hash(payload),
            payload,
            decision_seal_hash=supplied_seal,
        )
        self._terminal_opened = True
        self._phase = TERMINAL_OPEN
        return capability

    def close_terminal_scope(self, capability: LabelCapability) -> None:
        self._close(capability, TERMINAL)
        self._phase = CLOSED

    def assert_active(
        self, capability: LabelCapability, *, kind: str, scope_id: str | None = None
    ) -> None:
        """Validate a capability immediately before a scoped label decoder runs."""

        if (
            self._active is None
            or capability != self._active
            or capability.journal_id != self._journal_id
            or capability.kind != kind
            or (scope_id is not None and capability.scope_id != str(scope_id))
        ):
            raise GovernanceError("SCALE-BP v2 label capability is not active.")

    def audit_payload(self) -> dict[str, object]:
        """Return a JSON-safe immutable-by-copy audit record with no labels."""

        body = {
            "schema_version": JOURNAL_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "journal_id": self._journal_id,
            "run_identity_hash": self._run_identity_hash,
            "phase": self._phase,
            "active_capability": (
                None if self._active is None else self._active.to_payload()
            ),
            "decision_seal_hash": self._decision_seal_hash,
            "event_count": len(self._events),
            "closed_scope_counts": dict(self._closed_counts),
            "delegated_worker_count": len(self._delegations),
            "accepted_worker_audit_count": len(self._worker_audits),
            "delegated_worker_audit_hashes": {
                center: audit.audit_hash
                for center, audit in self._worker_audits.items()
            },
            "events": [dict(event) for event in self._events],
            "raw_labels_persisted": False,
            "row_level_label_values_persisted": False,
            "historical_capability_state_imported": False,
        }
        return {**body, "audit_hash": canonical_hash(body)}

    # Short aliases keep lifecycle call sites readable without weakening scope.
    grant_donor = open_donor_scope
    close_donor = close_donor_scope
    grant_support = open_support_scope
    close_support = close_support_scope
    open_terminal = open_terminal_scope
    close_terminal = close_terminal_scope

    def _require_preterminal_idle(self) -> None:
        if self._phase != PRETERMINAL or self._active is not None:
            raise GovernanceError(
                "SCALE-BP v2 preterminal capability transition is out of order."
            )

    def _new_scope_id(self, value: object) -> str:
        scope = str(value)
        if not scope or scope in self._used_scope_ids:
            raise GovernanceError("SCALE-BP v2 label scope identity is invalid.")
        self._used_scope_ids.add(scope)
        return scope

    def _open(
        self,
        kind: str,
        scope_id: str,
        scope_hash: str,
        details: Mapping[str, object],
        *,
        decision_seal_hash: str | None = None,
    ) -> LabelCapability:
        event_hash = self._append_event(
            f"OPEN_{kind}",
            {
                "kind": kind,
                "scope_id": scope_id,
                "scope_hash": scope_hash,
                **dict(details),
            },
        )
        capability = LabelCapability(
            journal_id=self._journal_id,
            kind=kind,
            scope_id=scope_id,
            scope_hash=scope_hash,
            sequence=len(self._events),
            event_hash=event_hash,
            decision_seal_hash=decision_seal_hash,
        )
        self._active = capability
        return capability

    def _close(self, capability: LabelCapability, expected_kind: str) -> None:
        self.assert_active(capability, kind=expected_kind)
        self._append_event(
            f"CLOSE_{expected_kind}",
            {
                "kind": expected_kind,
                "scope_id": capability.scope_id,
                "scope_hash": capability.scope_hash,
                "open_event_hash": capability.event_hash,
            },
        )
        self._closed_counts[expected_kind] += 1
        self._active = None

    def _append_event(self, transition: str, payload: Mapping[str, object]) -> str:
        previous = self._events[-1]["event_hash"] if self._events else None
        body = {
            "schema_version": EVENT_SCHEMA,
            "journal_id": self._journal_id,
            "sequence": len(self._events) + 1,
            "transition": transition,
            "previous_event_hash": previous,
            **dict(payload),
            "raw_labels_persisted": False,
        }
        event_hash = canonical_hash(body)
        self._events.append({**body, "event_hash": event_hash})
        return event_hash


class DelegatedWorkerLabelJournal:
    """Task-bound child journal used inside one spawned outer-H process."""

    def __init__(
        self, delegation: WorkerLabelDelegation | Mapping[str, object]
    ) -> None:
        self._delegation = (
            delegation
            if isinstance(delegation, WorkerLabelDelegation)
            else WorkerLabelDelegation.from_payload(delegation)
        )
        self._journal_id = canonical_hash(
            {
                "schema_version": WORKER_JOURNAL_SCHEMA,
                "delegation_hash": self._delegation.delegation_hash,
                "task_hash": self._delegation.task_hash,
            }
        )
        self._events: list[dict[str, object]] = []
        self._active: LabelCapability | None = None
        self._donor_closed = False
        self._support_index = 0
        self._audit: WorkerCapabilityAudit | None = None
        self._manifest_verified = False

    @property
    def delegation(self) -> WorkerLabelDelegation:
        return self._delegation

    @property
    def journal_id(self) -> str:
        return self._journal_id

    @property
    def manifest_path(self) -> str:
        return self._delegation.manifest_path

    def assert_manifest_binding(self, path: str | Path, sha256: str) -> None:
        if (
            str(path) != self._delegation.manifest_path
            or require_sha256(sha256, "worker manifest binding")
            != self._delegation.manifest_sha256
        ):
            raise GovernanceError("SCALE-BP v2 worker manifest binding drifted.")

    def verify_manifest_file(self) -> str:
        """Rehash the immutable manifest inside the spawned process."""

        path = Path(self._delegation.manifest_path)
        if path.is_symlink() or not path.is_file():
            raise GovernanceError("SCALE-BP v2 worker manifest is absent or unsafe.")
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise GovernanceError("SCALE-BP v2 worker cannot hash manifest.") from exc
        observed = digest.hexdigest()
        self.assert_manifest_binding(path, observed)
        self._manifest_verified = True
        return observed

    def open_donor_scope(self) -> LabelCapability:
        if (
            self._audit is not None
            or self._active is not None
            or self._donor_closed
            or not self._manifest_verified
        ):
            raise GovernanceError("SCALE-BP v2 worker donor transition drifted.")
        donors = tuple(
            center for center in CENTERS if center != self._delegation.outer_center
        )
        return self._open(
            DONOR,
            f"worker:{self._delegation.outer_center}:donor",
            self._delegation.donor_identity_hash,
            {
                "outer_center": self._delegation.outer_center,
                "donor_centers": list(donors),
                "row_identity_hash": self._delegation.donor_identity_hash,
                "outer_center_labels_available": False,
            },
        )

    def close_donor_scope(self, capability: LabelCapability) -> None:
        self._close(capability, DONOR)
        self._donor_closed = True

    def open_support_scope(self, held_case_id: str) -> LabelCapability:
        if (
            self._audit is not None
            or not self._donor_closed
            or self._active is not None
            or self._support_index >= len(self._delegation.route_scopes)
        ):
            raise GovernanceError("SCALE-BP v2 worker support transition drifted.")
        approved = self._delegation.route_scopes[self._support_index]
        if str(held_case_id) != approved.held_case_id:
            raise GovernanceError("SCALE-BP v2 worker support order escaped scope.")
        return self._open(
            SUPPORT,
            f"worker:{self._delegation.outer_center}:support:{approved.held_case_id}",
            approved.scope_hash,
            {
                "target_center": self._delegation.outer_center,
                "held_case_id": approved.held_case_id,
                "support_identity_hash": approved.support_identity_hash,
                "evaluation_identity_hash": approved.evaluation_identity_hash,
                "held_case_labels_available": False,
                "support_updates_global_state": False,
                "support_updates_source_experts": False,
                "support_tunes_hyperparameters": False,
            },
        )

    def close_support_scope(self, capability: LabelCapability) -> None:
        self._close(capability, SUPPORT)
        self._support_index += 1

    def assert_active(
        self, capability: LabelCapability, *, kind: str, scope_id: str | None = None
    ) -> None:
        if (
            self._active is None
            or capability != self._active
            or capability.journal_id != self._journal_id
            or capability.kind != kind
            or (scope_id is not None and capability.scope_id != str(scope_id))
        ):
            raise GovernanceError("SCALE-BP v2 worker capability is not active.")

    def complete(self, *, decision_fragment_hash: str) -> WorkerCapabilityAudit:
        if (
            self._audit is not None
            or self._active is not None
            or not self._donor_closed
            or self._support_index != len(self._delegation.route_scopes)
        ):
            raise GovernanceError("SCALE-BP v2 worker label scopes are incomplete.")
        decision_hash = require_sha256(
            decision_fragment_hash, "worker decision fragment hash"
        )
        event_log = self._event_log_body()
        audit = WorkerCapabilityAudit(
            delegation_hash=self._delegation.delegation_hash,
            worker_journal_id=self._journal_id,
            task_id=self._delegation.task_id,
            task_hash=self._delegation.task_hash,
            outer_center=self._delegation.outer_center,
            donor_scope_count=1,
            support_scope_count=self._support_index,
            event_count=len(self._events),
            event_log_hash=canonical_hash(event_log),
            decision_fragment_hash=decision_hash,
        )
        self._audit = audit
        return audit

    def event_log_payload(self) -> dict[str, object]:
        if self._audit is None:
            raise GovernanceError("SCALE-BP v2 worker event log is not final.")
        body = self._event_log_body()
        return {**body, "event_log_hash": canonical_hash(body)}

    def _event_log_body(self) -> dict[str, object]:
        return {
            "schema_version": WORKER_JOURNAL_SCHEMA,
            "worker_journal_id": self._journal_id,
            "delegation_hash": self._delegation.delegation_hash,
            "task_id": self._delegation.task_id,
            "task_hash": self._delegation.task_hash,
            "outer_center": self._delegation.outer_center,
            "event_count": len(self._events),
            "events": [dict(event) for event in self._events],
            "raw_labels_persisted": False,
            "row_level_label_values_persisted": False,
        }

    def _open(
        self,
        kind: str,
        scope_id: str,
        scope_hash: str,
        details: Mapping[str, object],
    ) -> LabelCapability:
        event_hash = self._append_event(
            f"OPEN_{kind}",
            {
                "kind": kind,
                "scope_id": scope_id,
                "scope_hash": scope_hash,
                **dict(details),
            },
        )
        capability = LabelCapability(
            journal_id=self._journal_id,
            kind=kind,
            scope_id=scope_id,
            scope_hash=scope_hash,
            sequence=len(self._events),
            event_hash=event_hash,
        )
        self._active = capability
        return capability

    def _close(self, capability: LabelCapability, expected_kind: str) -> None:
        self.assert_active(capability, kind=expected_kind)
        self._append_event(
            f"CLOSE_{expected_kind}",
            {
                "kind": expected_kind,
                "scope_id": capability.scope_id,
                "scope_hash": capability.scope_hash,
                "open_event_hash": capability.event_hash,
            },
        )
        self._active = None

    def _append_event(self, transition: str, payload: Mapping[str, object]) -> str:
        previous = self._events[-1]["event_hash"] if self._events else None
        body = {
            "schema_version": WORKER_EVENT_SCHEMA,
            "worker_journal_id": self._journal_id,
            "delegation_hash": self._delegation.delegation_hash,
            "sequence": len(self._events) + 1,
            "transition": transition,
            "previous_event_hash": previous,
            **dict(payload),
            "raw_labels_persisted": False,
        }
        event_hash = canonical_hash(body)
        self._events.append({**body, "event_hash": event_hash})
        return event_hash


__all__ = (
    "CAPABILITY_SCHEMA",
    "CLOSED",
    "DECISIONS_SEALED",
    "DELEGATION_SCHEMA",
    "DelegatedWorkerLabelJournal",
    "DONOR",
    "EVENT_SCHEMA",
    "JOURNAL_SCHEMA",
    "LabelCapability",
    "LabelCapabilityJournal",
    "PRETERMINAL",
    "SUPPORT",
    "TERMINAL",
    "TERMINAL_OPEN",
    "WORKER_AUDIT_SCHEMA",
    "WORKER_EVENT_SCHEMA",
    "WORKER_JOURNAL_SCHEMA",
    "WorkerCapabilityAudit",
    "WorkerLabelDelegation",
    "WorkerSupportScope",
)
