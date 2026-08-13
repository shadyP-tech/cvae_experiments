"""Route-scoped label grants and the global 218-decision terminal barrier."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ...protocol import ProtocolError
from .constants import CENTERS, EXPECTED_TOTAL_CASE_COUNT, candidate_sources
from .hashing import canonical_hash, require_sha256
from .loo_plans import LooPlanSeal, WholeCaseLooPlan
from .products import BinaryLabel


@dataclass(frozen=True, order=True)
class LabelAccessEvent:
    role: str
    heldout_center: str | None
    case_id: str | None
    source: str | None
    row_count: int
    case_count: int
    identity_hash: str
    own_held_case_included: bool

    def to_payload(self) -> dict[str, object]:
        return {
            **self.__dict__,
            "raw_labels_persisted": False,
        }


class LabelCapabilityFirewall:
    """The only label opener for donor, route support, and terminal roles.

    ``label_loader`` is deliberately lazy: merely constructing the firewall
    and globally sealing all 218 plans does not read terminal label values.
    """

    def __init__(
        self,
        plan_seal: LooPlanSeal,
        label_loader: Callable[
            [frozenset[tuple[str, str, str]]], Sequence[object]
        ],
        *,
        expected_plan_count: int = EXPECTED_TOTAL_CASE_COUNT,
    ) -> None:
        if len(plan_seal.plans) != int(expected_plan_count):
            raise ProtocolError("DCSE label firewall requires the complete global plan seal.")
        self._plan_seal = plan_seal
        self._plans = MappingProxyType({plan.key: plan for plan in plan_seal.plans})
        self._sample_keys_by_case = MappingProxyType(
            {
                plan.key: tuple(
                    (plan.target_center, plan.case_id, sample_id)
                    for sample_id in plan.evaluation_sample_ids
                )
                for plan in plan_seal.plans
            }
        )
        self._label_loader = label_loader
        self._donor_opened: set[tuple[str, str]] = set()
        self._support_opened: set[tuple[str, str]] = set()
        self._decision_seals: dict[tuple[str, str], str] = {}
        self._events: list[LabelAccessEvent] = []
        self._terminal_opened = False
        self._aggregate_seal: Mapping[str, str] | None = None

    @property
    def plan_seal_hash(self) -> str:
        return self._plan_seal.plan_seal_hash

    @property
    def decision_seal_count(self) -> int:
        return len(self._decision_seals)

    def open_donor_labels(self, heldout_center: str, source: str) -> tuple[BinaryLabel, ...]:
        """Open all and only q outside {H,e}; never any H/e query labels."""

        heldout = str(heldout_center)
        candidate = str(source)
        key = (heldout, candidate)
        if (
            heldout not in CENTERS
            or candidate not in candidate_sources(heldout)
            or key in self._donor_opened
            or self._support_opened
            or self._terminal_opened
        ):
            raise ProtocolError("DCSE donor label capability opened out of order or twice.")
        legal_queries = set(CENTERS).difference((heldout, candidate))
        allowed_keys = frozenset(
            sample_key
            for (target, _case), sample_keys in self._sample_keys_by_case.items()
            if target in legal_queries
            for sample_key in sample_keys
        )
        rows = self._load_labels(
            f"donor::heldout_H={heldout}::source_e={candidate}", allowed_keys
        )
        if not rows or {row.target_center for row in rows} != legal_queries:
            raise ProtocolError("DCSE donor grant lacks one or more q outside {H,e}.")
        if any(row.target_center in {heldout, candidate} for row in rows):
            raise ProtocolError("DCSE donor grant leaked H or e query labels.")
        self._donor_opened.add(key)
        self._record("donor", heldout, None, candidate, rows, own=False)
        return rows

    # Explicit alias used in runtime orchestration.
    open_global_donor_labels = open_donor_labels

    def open_route_support_labels(
        self,
        target_center: str,
        case_id: str,
        *,
        plan_hash: str,
    ) -> tuple[BinaryLabel, ...]:
        """Open H-minus-c labels for exactly one route; grants never cross routes."""

        key = (str(target_center), str(case_id))
        try:
            plan = self._plans[key]
        except KeyError as exc:
            raise ProtocolError("DCSE route support requested for an unsealed plan.") from exc
        if (
            require_sha256(plan_hash, "plan_hash") != plan.plan_hash
            or key in self._support_opened
            or key in self._decision_seals
            or self._terminal_opened
        ):
            raise ProtocolError("DCSE route support capability opened out of order or twice.")
        allowed_keys = frozenset(
            sample_key
            for support_case in plan.support_case_ids
            for sample_key in self._sample_keys_by_case[
                (plan.target_center, support_case)
            ]
        )
        rows = self._load_labels(
            f"route_support::H={plan.target_center}::c={plan.case_id}",
            allowed_keys,
        )
        observed = {row.case_id for row in rows}
        if not rows or observed != set(plan.support_case_ids) or plan.case_id in observed:
            raise ProtocolError("DCSE route support grant is not exactly H-minus-c.")
        self._support_opened.add(key)
        self._record("route_support", plan.target_center, plan.case_id, None, rows, own=False)
        return rows

    def record_route_decision_seal(
        self,
        target_center: str,
        case_id: str,
        decision_seal_hash: str,
    ) -> None:
        key = (str(target_center), str(case_id))
        require_sha256(decision_seal_hash, "route_decision_seal_hash")
        if (
            key not in self._plans
            or key not in self._support_opened
            or key in self._decision_seals
            or self._terminal_opened
        ):
            raise ProtocolError("DCSE route decision seal is missing support authority or duplicated.")
        self._decision_seals[key] = decision_seal_hash

    def open_terminal_labels(self) -> tuple[BinaryLabel, ...]:
        """Open all labels only after every one of the 218 route seals exists."""

        expected = set(self._plans)
        if (
            set(self._decision_seals) != expected
            or set(self._support_opened) != expected
            or self._aggregate_seal is None
            or self._terminal_opened
        ):
            raise ProtocolError(
                "DCSE terminal labels require all 218 route seals and the "
                "persisted/read-back aggregate plan+decision seal."
            )
        allowed_keys = frozenset(
            sample_key
            for sample_keys in self._sample_keys_by_case.values()
            for sample_key in sample_keys
        )
        rows = self._load_labels("terminal_evaluation", allowed_keys)
        expected_cases = set(self._plans)
        observed_cases = {row.case_key for row in rows}
        if observed_cases != expected_cases:
            raise ProtocolError("DCSE terminal labels do not cover the sealed 218-case universe.")
        self._terminal_opened = True
        self._record("terminal_evaluation", None, None, None, rows, own=True)
        return rows

    def record_aggregate_plan_decision_seal(
        self,
        seal_hash: str,
        *,
        plan_seal_hash: str | None = None,
        decision_barrier_hash: str | None = None,
    ) -> None:
        """Bind the persisted/read-back global barrier before terminal access."""

        expected = set(self._plans)
        value = require_sha256(seal_hash, "aggregate_plan_decision_seal_hash")
        if (
            set(self._decision_seals) != expected
            or set(self._support_opened) != expected
            or self._aggregate_seal is not None
            or self._terminal_opened
        ):
            raise ProtocolError(
                "DCSE aggregate decision seal requires all 218 route seals exactly once."
            )
        barrier = self.decision_barrier_payload()
        current_plan_hash = self._plan_seal.plan_seal_hash
        current_barrier_hash = str(barrier["decision_barrier_hash"])
        if plan_seal_hash is not None and require_sha256(
            plan_seal_hash, "bound_plan_seal_hash"
        ) != current_plan_hash:
            raise ProtocolError("DCSE aggregate seal is bound to another global plan seal.")
        if decision_barrier_hash is not None and require_sha256(
            decision_barrier_hash, "bound_decision_barrier_hash"
        ) != current_barrier_hash:
            raise ProtocolError("DCSE aggregate seal is bound to another decision barrier.")
        self._aggregate_seal = MappingProxyType(
            {
                "aggregate_plan_decision_seal_hash": value,
                "plan_seal_hash": current_plan_hash,
                "decision_barrier_hash": current_barrier_hash,
            }
        )

    open_terminal_evaluation_labels = open_terminal_labels

    def report_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_dcse_label_capability_report_v1",
            "status": "PASS" if self._terminal_opened else "INCOMPLETE",
            "global_plan_seal_hash": self._plan_seal.plan_seal_hash,
            "global_plan_count": len(self._plans),
            "donor_grant_count": len(self._donor_opened),
            "route_support_grant_count": len(self._support_opened),
            "route_decision_seal_count": len(self._decision_seals),
            "aggregate_plan_decision_seal_hash": (
                None
                if self._aggregate_seal is None
                else self._aggregate_seal["aggregate_plan_decision_seal_hash"]
            ),
            "aggregate_plan_decision_seal": (
                None if self._aggregate_seal is None else dict(self._aggregate_seal)
            ),
            "aggregate_plan_decision_seal_recorded": (
                self._aggregate_seal is not None
            ),
            "terminal_opened": self._terminal_opened,
            "terminal_scoring_opened": self._terminal_opened,
            "events": [event.to_payload() for event in self._events],
            "all_218_plan_and_decision_seals_before_terminal_labels": (
                not self._terminal_opened or set(self._decision_seals) == set(self._plans)
            ),
            "aggregate_plan_decision_seal_before_terminal_labels": (
                not self._terminal_opened or self._aggregate_seal is not None
            ),
            "route_support_grants_are_H_minus_c": all(
                not event.own_held_case_included
                for event in self._events
                if event.role == "route_support"
            ),
            "raw_labels_persisted": False,
        }

    def decision_barrier_payload(self) -> dict[str, object]:
        if set(self._decision_seals) != set(self._plans):
            raise ProtocolError("DCSE decision barrier requires all 218 route seals.")
        payload = {
            "schema_version": "fixed_bank_dcse_all_218_decision_barrier_v1",
            "plan_seal_hash": self._plan_seal.plan_seal_hash,
            "route_count": len(self._plans),
            "decision_seals": [
                [target, case_id, seal]
                for (target, case_id), seal in sorted(
                    self._decision_seals.items(),
                    key=lambda item: (CENTERS.index(item[0][0]), item[0][1]),
                )
            ],
            "terminal_labels_used": False,
        }
        return {**payload, "decision_barrier_hash": canonical_hash(payload)}

    def _load_labels(
        self,
        scope: str,
        allowed_keys: frozenset[tuple[str, str, str]],
    ) -> tuple[BinaryLabel, ...]:
        if not allowed_keys:
            raise ProtocolError("DCSE label capability has an empty authorized identity set.")
        # The I/O boundary itself is scoped: a loader that cannot accept the
        # authorized identity set is not a valid DCSE capability provider.
        raw = tuple(self._label_loader(allowed_keys))
        rows: list[BinaryLabel] = []
        seen: set[tuple[str, str, str]] = set()
        for row in raw:
            key = (
                str(row.target_center),
                str(row.case_id),
                str(row.sample_id),
            )
            if key not in allowed_keys:
                raise ProtocolError(
                    "DCSE scoped label loader returned an unauthorized identity."
                )
            try:
                value = row.value
            except AttributeError:
                value = row.label
            label = BinaryLabel(
                key[0],
                key[1],
                key[2],
                int(value),
                scope,
            )
            if label.sample_key in seen:
                raise ProtocolError("DCSE label loader returned duplicate sample identities.")
            seen.add(label.sample_key)
            rows.append(label)
        if not rows:
            raise ProtocolError("DCSE label loader returned no labels.")
        if seen != allowed_keys:
            raise ProtocolError("DCSE label loader did not cover its exact authorized identities.")
        return tuple(rows)

    def _record(
        self,
        role: str,
        heldout: str | None,
        case_id: str | None,
        source: str | None,
        rows: Sequence[BinaryLabel],
        *,
        own: bool,
    ) -> None:
        identities = tuple(sorted(row.sample_key for row in rows))
        self._events.append(
            LabelAccessEvent(
                role,
                heldout,
                case_id,
                source,
                len(rows),
                len({row.case_key for row in rows}),
                canonical_hash([list(key) for key in identities]),
                own,
            )
        )


# Descriptive name retained for integration discoverability.
DirectionalShrinkageLabelCapabilityManager = LabelCapabilityFirewall


__all__ = (
    "DirectionalShrinkageLabelCapabilityManager",
    "LabelAccessEvent",
    "LabelCapabilityFirewall",
)
