"""Experiment-owned scoped label firewall and global 218-route barrier."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ...protocol import ProtocolError
from .constants import CENTERS, EXPECTED_TOTAL_CASE_COUNT, candidate_sources
from .hashing import canonical_hash, require_sha256
from .held_case_plans import HeldCasePlanSeal
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
        return {**self.__dict__, "raw_labels_persisted": False}


class DirectionalCorrectnessLabelFirewall:
    """The sole label opener for donor, H-minus-c, and terminal scopes."""

    def __init__(
        self,
        plan_seal: HeldCasePlanSeal,
        label_loader: Callable[
            [frozenset[tuple[str, str, str]]], Sequence[object]
        ],
        *,
        expected_plan_count: int = EXPECTED_TOTAL_CASE_COUNT,
    ) -> None:
        if len(plan_seal.plans) != int(expected_plan_count):
            raise ProtocolError(
                "Abstention-router label firewall requires the complete global plan seal."
            )
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
        self._expected_donor_keys = frozenset(
            (heldout, source)
            for heldout in CENTERS
            for source in candidate_sources(heldout)
        )
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

    def open_donor_labels(
        self, heldout_center: object, source: object
    ) -> tuple[BinaryLabel, ...]:
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
            raise ProtocolError(
                "Abstention-router donor label capability opened out of order or twice."
            )
        legal_queries = set(CENTERS).difference((heldout, candidate))
        allowed = frozenset(
            sample_key
            for (target, _case), sample_keys in self._sample_keys_by_case.items()
            if target in legal_queries
            for sample_key in sample_keys
        )
        rows = self._load_labels(
            f"donor::heldout_H={heldout}::source_e={candidate}", allowed
        )
        if {row.target_center for row in rows} != legal_queries:
            raise ProtocolError(
                "Abstention-router donor grant lacks q outside H and e."
            )
        self._donor_opened.add(key)
        self._record("donor", heldout, None, candidate, rows, own=False)
        return rows

    open_global_donor_labels = open_donor_labels

    def open_route_support_labels(
        self,
        target_center: object,
        case_id: object,
        *,
        plan_hash: str,
    ) -> tuple[BinaryLabel, ...]:
        key = (str(target_center), str(case_id))
        try:
            plan = self._plans[key]
        except KeyError as exc:
            raise ProtocolError(
                "Abstention-router support requested for an unsealed route."
            ) from exc
        if (
            require_sha256(plan_hash, "plan_hash") != plan.plan_hash
            or self._donor_opened != self._expected_donor_keys
            or key in self._support_opened
            or key in self._decision_seals
            or self._terminal_opened
        ):
            raise ProtocolError(
                "Abstention-router support requires all 72 donor grants and opens once."
            )
        allowed = frozenset(
            sample_key
            for support_case in plan.support_case_ids
            for sample_key in self._sample_keys_by_case[
                (plan.target_center, support_case)
            ]
        )
        rows = self._load_labels(
            f"route_support::H={plan.target_center}::c={plan.case_id}", allowed
        )
        observed = {row.case_id for row in rows}
        if observed != set(plan.support_case_ids) or plan.case_id in observed:
            raise ProtocolError(
                "Abstention-router support grant is not exactly H-minus-c."
            )
        self._support_opened.add(key)
        self._record("route_support", plan.target_center, plan.case_id, None, rows, own=False)
        return rows

    def record_route_decision_seal(
        self,
        target_center: object,
        case_id: object,
        decision_seal_hash: str,
    ) -> None:
        key = (str(target_center), str(case_id))
        value = require_sha256(decision_seal_hash, "route_decision_seal_hash")
        if (
            key not in self._plans
            or key not in self._support_opened
            or key in self._decision_seals
            or self._terminal_opened
        ):
            raise ProtocolError(
                "Abstention-router route seal lacks support authority or is duplicated."
            )
        self._decision_seals[key] = value

    def decision_barrier_payload(self) -> dict[str, object]:
        if set(self._decision_seals) != set(self._plans):
            raise ProtocolError(
                "Abstention-router decision barrier requires all 218 route seals."
            )
        payload = {
            "schema_version": "fixed_bank_cdca_all_218_decision_barrier_v1",
            "plan_seal_hash": self._plan_seal.plan_seal_hash,
            "route_count": len(self._plans),
            "decision_seals": [
                [target, case, value]
                for (target, case), value in sorted(
                    self._decision_seals.items(),
                    key=lambda item: (CENTERS.index(item[0][0]), item[0][1]),
                )
            ],
            "terminal_labels_used": False,
        }
        return {**payload, "decision_barrier_hash": canonical_hash(payload)}

    def record_aggregate_plan_decision_seal(
        self,
        seal_hash: str,
        *,
        plan_seal_hash: str,
        decision_barrier_hash: str,
    ) -> None:
        if (
            set(self._decision_seals) != set(self._plans)
            or set(self._support_opened) != set(self._plans)
            or self._aggregate_seal is not None
            or self._terminal_opened
        ):
            raise ProtocolError(
                "Abstention-router aggregate seal requires every route exactly once."
            )
        barrier = self.decision_barrier_payload()
        if require_sha256(plan_seal_hash, "bound_plan_seal_hash") != self.plan_seal_hash:
            raise ProtocolError("Abstention-router aggregate binds another plan seal.")
        if require_sha256(
            decision_barrier_hash, "bound_decision_barrier_hash"
        ) != barrier["decision_barrier_hash"]:
            raise ProtocolError("Abstention-router aggregate binds another decision barrier.")
        self._aggregate_seal = MappingProxyType(
            {
                "aggregate_plan_decision_seal_hash": require_sha256(
                    seal_hash, "aggregate_plan_decision_seal_hash"
                ),
                "plan_seal_hash": self.plan_seal_hash,
                "decision_barrier_hash": str(barrier["decision_barrier_hash"]),
            }
        )

    def open_terminal_labels(self) -> tuple[BinaryLabel, ...]:
        if (
            set(self._decision_seals) != set(self._plans)
            or set(self._support_opened) != set(self._plans)
            or self._aggregate_seal is None
            or self._terminal_opened
        ):
            raise ProtocolError(
                "Abstention-router terminal labels require all 218 routes and aggregate seal."
            )
        allowed = frozenset(
            sample_key
            for sample_keys in self._sample_keys_by_case.values()
            for sample_key in sample_keys
        )
        rows = self._load_labels("terminal_evaluation", allowed)
        if {(row.target_center, row.case_id) for row in rows} != set(self._plans):
            raise ProtocolError(
                "Abstention-router terminal labels do not cover all sealed cases."
            )
        self._terminal_opened = True
        self._record("terminal_evaluation", None, None, None, rows, own=True)
        return rows

    open_terminal_evaluation_labels = open_terminal_labels

    def report_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_cdca_label_capability_report_v1",
            "status": "PASS" if self._terminal_opened else "INCOMPLETE",
            "global_plan_seal_hash": self.plan_seal_hash,
            "global_plan_count": len(self._plans),
            "donor_grant_count": len(self._donor_opened),
            "all_72_donor_grants_before_route_support": (
                not self._support_opened
                or self._donor_opened == self._expected_donor_keys
            ),
            "route_support_grant_count": len(self._support_opened),
            "route_decision_seal_count": len(self._decision_seals),
            "aggregate_plan_decision_seal": (
                None if self._aggregate_seal is None else dict(self._aggregate_seal)
            ),
            "terminal_opened": self._terminal_opened,
            "terminal_scoring_opened": self._terminal_opened,
            "events": [event.to_payload() for event in self._events],
            "all_218_route_seals_before_terminal": (
                not self._terminal_opened
                or set(self._decision_seals) == set(self._plans)
            ),
            "aggregate_seal_before_terminal": (
                not self._terminal_opened or self._aggregate_seal is not None
            ),
            "raw_labels_persisted": False,
        }

    def _load_labels(
        self,
        scope: str,
        allowed_keys: frozenset[tuple[str, str, str]],
    ) -> tuple[BinaryLabel, ...]:
        if not allowed_keys:
            raise ProtocolError(
                "Abstention-router label capability has an empty identity set."
            )
        raw = tuple(self._label_loader(allowed_keys))
        rows: list[BinaryLabel] = []
        seen: set[tuple[str, str, str]] = set()
        for row in raw:
            key = (str(row.target_center), str(row.case_id), str(row.sample_id))
            if key not in allowed_keys or key in seen:
                raise ProtocolError(
                    "Abstention-router loader returned unauthorized or duplicate identity."
                )
            try:
                value = row.value
            except AttributeError:
                value = row.label
            rows.append(BinaryLabel(*key, int(value), scope))
            seen.add(key)
        if seen != allowed_keys:
            raise ProtocolError(
                "Abstention-router loader did not cover the exact authorized identities."
            )
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
        identities = tuple(sorted(row.key for row in rows))
        self._events.append(
            LabelAccessEvent(
                role,
                heldout,
                case_id,
                source,
                len(rows),
                len({(row.target_center, row.case_id) for row in rows}),
                canonical_hash([list(key) for key in identities]),
                own,
            )
        )


LabelCapabilityFirewall = DirectionalCorrectnessLabelFirewall


__all__ = (
    "DirectionalCorrectnessLabelFirewall",
    "LabelAccessEvent",
    "LabelCapabilityFirewall",
)
