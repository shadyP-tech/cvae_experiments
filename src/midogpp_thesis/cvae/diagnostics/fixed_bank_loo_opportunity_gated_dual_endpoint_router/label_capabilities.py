"""Experiment-owned label firewall and all-218 decision barrier."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from ...protocol import ProtocolError
from .constants import CENTERS, EXPECTED_TOTAL_CASE_COUNT, candidate_sources
from .hashing import canonical_hash, require_sha256
from .response_products import BinaryLabel
from .split_plans import LooPlanSeal


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


class DualEndpointLabelFirewall:
    """Sole decoder for q-not-{H,e}, H-minus-c, and terminal labels."""

    def __init__(
        self,
        plan_seal: LooPlanSeal,
        label_loader: Callable[[frozenset[tuple[str, str, str]]], Sequence[object]],
    ) -> None:
        if len(plan_seal.plans) != EXPECTED_TOTAL_CASE_COUNT:
            raise ProtocolError("OGDE label firewall requires the complete 218-plan seal.")
        self._seal = plan_seal
        self._plans = MappingProxyType({plan.key: plan for plan in plan_seal.plans})
        self._samples = MappingProxyType(
            {
                plan.key: tuple(
                    (plan.target_center, plan.case_id, sample_id)
                    for sample_id in plan.evaluation_sample_ids
                )
                for plan in plan_seal.plans
            }
        )
        self._loader = label_loader
        self._expected_donor = frozenset(
            (heldout, source)
            for heldout in CENTERS
            for source in candidate_sources(heldout)
        )
        self._donor_opened: set[tuple[str, str]] = set()
        self._support_opened: set[tuple[str, str]] = set()
        self._route_seals: dict[tuple[str, str], str] = {}
        self._aggregate_seal: str | None = None
        self._events: list[LabelAccessEvent] = []
        self._terminal_opened = False

    @property
    def plan_seal_hash(self) -> str:
        return self._seal.plan_seal_hash

    def _decode(
        self, role: str, allowed: frozenset[tuple[str, str, str]]
    ) -> tuple[BinaryLabel, ...]:
        raw = tuple(self._loader(allowed))
        labels = tuple(
            BinaryLabel(
                str(getattr(row, "target_center", getattr(row, "center", ""))),
                str(getattr(row, "case_id", "")),
                str(getattr(row, "sample_id", getattr(row, "evaluation_row_id", ""))),
                int(getattr(row, "value", getattr(row, "label", -1))),
                role,
            )
            for row in raw
        )
        if {row.key for row in labels} != set(allowed) or len(labels) != len(allowed):
            raise ProtocolError("OGDE label loader returned rows outside its exact capability.")
        return tuple(sorted(labels, key=lambda row: row.key))

    def _record(
        self,
        role: str,
        heldout: str | None,
        case: str | None,
        source: str | None,
        labels: Sequence[BinaryLabel],
        *,
        own: bool,
    ) -> None:
        identities = [list(row.key) for row in labels]
        self._events.append(
            LabelAccessEvent(
                role,
                heldout,
                case,
                source,
                len(labels),
                len({(row.target_center, row.case_id) for row in labels}),
                canonical_hash(identities),
                own,
            )
        )

    def open_donor_labels(self, heldout_center: object, source: object) -> tuple[BinaryLabel, ...]:
        heldout, candidate = str(heldout_center), str(source)
        key = (heldout, candidate)
        if key not in self._expected_donor or key in self._donor_opened or self._support_opened or self._terminal_opened:
            raise ProtocolError("OGDE donor label grant opened out of order or twice.")
        legal_queries = set(CENTERS).difference((heldout, candidate))
        allowed = frozenset(
            sample_key
            for (target, _case), sample_keys in self._samples.items()
            if target in legal_queries
            for sample_key in sample_keys
        )
        role = f"donor::heldout_H={heldout}::source_e={candidate}"
        labels = self._decode(role, allowed)
        if {row.target_center for row in labels} != legal_queries:
            raise ProtocolError("OGDE donor grant lacks exact q outside H and e.")
        self._donor_opened.add(key)
        self._record("donor", heldout, None, candidate, labels, own=False)
        return labels

    def open_route_support_labels(
        self, target_center: object, case_id: object, *, plan_hash: str
    ) -> tuple[BinaryLabel, ...]:
        key = (str(target_center), str(case_id))
        try:
            plan = self._plans[key]
        except KeyError as exc:
            raise ProtocolError("OGDE support labels requested for an unsealed route.") from exc
        if (
            require_sha256(plan_hash, "plan_hash") != plan.plan_hash
            or self._donor_opened != self._expected_donor
            or key in self._support_opened
            or key in self._route_seals
            or self._terminal_opened
        ):
            raise ProtocolError("OGDE support grant requires all donor grants and opens once.")
        allowed = frozenset(
            sample_key
            for support_case in plan.support_case_ids
            for sample_key in self._samples[(plan.target_center, support_case)]
        )
        role = f"route_support::H={plan.target_center}::c={plan.case_id}"
        labels = self._decode(role, allowed)
        if {row.case_id for row in labels} != set(plan.support_case_ids) or plan.case_id in {row.case_id for row in labels}:
            raise ProtocolError("OGDE support grant is not exactly H-minus-c.")
        self._support_opened.add(key)
        self._record("route_support", plan.target_center, plan.case_id, None, labels, own=False)
        return labels

    def record_route_decision_seal(
        self, target_center: object, case_id: object, decision_seal_hash: str
    ) -> None:
        key = (str(target_center), str(case_id))
        if key not in self._support_opened or key in self._route_seals or self._terminal_opened:
            raise ProtocolError("OGDE route seal lacks support authority or is duplicated.")
        self._route_seals[key] = require_sha256(decision_seal_hash, "route_decision_seal_hash")

    def decision_barrier_payload(self) -> dict[str, object]:
        if set(self._route_seals) != set(self._plans):
            raise ProtocolError("OGDE decision barrier requires all 218 route seals.")
        payload = {
            "schema_version": "fixed_bank_ogde_all_218_decision_barrier_v1",
            "plan_seal_hash": self.plan_seal_hash,
            "route_count": len(self._route_seals),
            "route_decision_seals": [
                [target, case, digest]
                for (target, case), digest in sorted(
                    self._route_seals.items(), key=lambda item: (CENTERS.index(item[0][0]), item[0][1])
                )
            ],
            "both_endpoints_and_compositions_sealed": True,
            "terminal_labels_used": False,
        }
        return {**payload, "decision_barrier_hash": canonical_hash(payload)}

    def record_aggregate_plan_decision_seal(
        self, aggregate_seal_hash: str, *, plan_seal_hash: str, decision_barrier_hash: str
    ) -> None:
        barrier = self.decision_barrier_payload()
        if (
            set(self._support_opened) != set(self._plans)
            or self._aggregate_seal is not None
            or self._terminal_opened
            or require_sha256(plan_seal_hash, "bound_plan_seal_hash") != self.plan_seal_hash
            or require_sha256(decision_barrier_hash, "bound_decision_barrier_hash") != barrier["decision_barrier_hash"]
        ):
            raise ProtocolError("OGDE aggregate seal does not bind the complete preterminal surface.")
        self._aggregate_seal = require_sha256(aggregate_seal_hash, "aggregate_plan_decision_seal_hash")

    def open_terminal_labels(self) -> tuple[BinaryLabel, ...]:
        if self._aggregate_seal is None or self._terminal_opened or set(self._route_seals) != set(self._plans):
            raise ProtocolError("OGDE terminal labels require the complete aggregate decision seal.")
        allowed = frozenset(sample_key for sample_keys in self._samples.values() for sample_key in sample_keys)
        labels = self._decode("terminal_evaluation", allowed)
        self._terminal_opened = True
        self._record("terminal_evaluation", None, None, None, labels, own=True)
        return labels

    def report_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_ogde_label_capability_report_v1",
            "status": "PASS" if self._terminal_opened else "INCOMPLETE",
            "plan_seal_hash": self.plan_seal_hash,
            "donor_grant_count": len(self._donor_opened),
            "route_support_grant_count": len(self._support_opened),
            "route_decision_seal_count": len(self._route_seals),
            "aggregate_seal_hash": self._aggregate_seal,
            "terminal_opened": self._terminal_opened,
            "events": [event.to_payload() for event in self._events],
        }


DirectionalCorrectnessLabelFirewall = DualEndpointLabelFirewall


__all__ = ("DirectionalCorrectnessLabelFirewall", "DualEndpointLabelFirewall", "LabelAccessEvent")
