"""Role-scoped label firewall for donor, H-c support, and terminal stages."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from ...protocol import ProtocolError
from .constants import CENTERS, EXPECTED_TOTAL_CASE_COUNT, candidate_sources
from .contracts import BinaryLabel
from .hashing import canonical_hash, require_sha256
from .outer_plans import OuterPlanSeal


PRETERMINAL_AGGREGATE_KEYS = frozenset(
    {
        "schema_version",
        "protocol_hash",
        "probability_surface_hash",
        "plan_seal_hash",
        "decision_barrier_hash",
        "utility_descriptor_hash",
        "fingerprint_hash",
        "target_posterior_model_hash",
        "target_posterior_prediction_hash",
        "route_posterior_ensemble_hash",
        "posterior_utility_prediction_hash",
        "donor_utility_row_hash",
        "donor_posterior_utility_hash",
        "envelope_calibration_hash",
        "utility_certificate_hash",
        "composition_hash",
        "policy_menu_hash",
        "terminal_labels_used",
        "aggregate_seal_hash",
    }
)


def validate_preterminal_aggregate_seal(
    aggregate_seal: Mapping[str, object],
    *,
    expected_plan_seal_hash: str,
    expected_decision_barrier_hash: str,
) -> str:
    payload = dict(aggregate_seal)
    submitted = require_sha256(
        str(payload.get("aggregate_seal_hash")), "aggregate_preterminal_seal_hash"
    )
    unhashed = {key: value for key, value in payload.items() if key != "aggregate_seal_hash"}
    if (
        set(payload) != PRETERMINAL_AGGREGATE_KEYS
        or payload.get("schema_version") != "fixed_bank_psscur_preterminal_aggregate_v1"
        or payload.get("plan_seal_hash")
        != require_sha256(expected_plan_seal_hash, "expected_plan_seal_hash")
        or payload.get("decision_barrier_hash")
        != require_sha256(expected_decision_barrier_hash, "expected_decision_barrier_hash")
        or payload.get("terminal_labels_used") is not False
        or canonical_hash(unhashed) != submitted
    ):
        raise ProtocolError("PSSCUR aggregate seal is not bound to its route barrier.")
    return submitted


@dataclass(frozen=True, order=True)
class LabelAccessEvent:
    role: str
    outer_target_center: str | None
    candidate_or_donor_center: str | None
    excluded_case_ids: tuple[str, ...]
    row_count: int
    case_count: int
    identity_hash: str

    def to_payload(self) -> dict[str, object]:
        return {
            **self.__dict__,
            "excluded_case_ids": list(self.excluded_case_ids),
            "raw_labels_persisted": False,
        }


class PSSCURLabelFirewall:
    """The sole label decoder; target evaluation labels open after route sealing."""

    def __init__(
        self,
        plan_seal: OuterPlanSeal,
        label_loader: Callable[[frozenset[tuple[str, str, str]], str], Sequence[object]],
    ) -> None:
        if plan_seal.strict_canonical_topology and len(plan_seal.outer_plans) != EXPECTED_TOTAL_CASE_COUNT:
            raise ProtocolError("PSSCUR canonical firewall requires all outer plans.")
        self._seal = plan_seal
        self._outer = MappingProxyType({plan.key: plan for plan in plan_seal.outer_plans})
        self._samples = MappingProxyType(
            {
                plan.key: tuple(
                    (plan.target_center, plan.case_id, sample_id)
                    for sample_id in plan.evaluation_sample_ids
                )
                for plan in plan_seal.outer_plans
            }
        )
        self._loader = label_loader
        self._expected_prior = frozenset(
            (heldout, source)
            for heldout in CENTERS
            for source in candidate_sources(heldout)
        )
        self._expected_donor = frozenset(
            (heldout, donor)
            for heldout in CENTERS
            for donor in CENTERS
            if donor != heldout
        )
        self._prior_opened: set[tuple[str, str]] = set()
        self._donor_opened: set[tuple[str, str]] = set()
        self._outer_support_opened: set[tuple[str, str]] = set()
        self._outer_state_seals: dict[tuple[str, str], str] = {}
        self._route_decision_seals: dict[tuple[str, str], str] = {}
        self._aggregate_seal: str | None = None
        self._terminal_opened = False
        self._events: list[LabelAccessEvent] = []

    @property
    def plan_seal_hash(self) -> str:
        return self._seal.seal_hash

    def _decode(
        self,
        role: str,
        allowed: frozenset[tuple[str, str, str]],
    ) -> tuple[BinaryLabel, ...]:
        raw = tuple(self._loader(allowed, role))
        labels = tuple(
            BinaryLabel(
                str(getattr(row, "center", getattr(row, "target_center", ""))),
                str(getattr(row, "case_id", "")),
                str(getattr(row, "sample_id", getattr(row, "evaluation_row_id", ""))),
                int(getattr(row, "value", getattr(row, "label", -1))),
                role,
            )
            for row in raw
        )
        if {row.key for row in labels} != set(allowed) or len(labels) != len(allowed):
            raise ProtocolError("PSSCUR label loader escaped its exact capability.")
        return tuple(sorted(labels, key=lambda row: row.key))

    def _record(
        self,
        role: str,
        outer: str | None,
        other: str | None,
        excluded: tuple[str, ...],
        labels: Sequence[BinaryLabel],
    ) -> None:
        self._events.append(
            LabelAccessEvent(
                role,
                outer,
                other,
                excluded,
                len(labels),
                len({(row.center, row.case_id) for row in labels}),
                canonical_hash([list(row.key) for row in labels]),
            )
        )

    def open_source_prior_labels(
        self, heldout_center: object, candidate_source: object
    ) -> tuple[BinaryLabel, ...]:
        heldout, source = str(heldout_center), str(candidate_source)
        key = (heldout, source)
        if (
            key not in self._expected_prior
            or key in self._prior_opened
            or self._outer_support_opened
            or self._terminal_opened
        ):
            raise ProtocolError("PSSCUR source-prior capability opened out of order.")
        legal = set(CENTERS).difference((heldout, source))
        allowed = frozenset(
            sample
            for (center, _case), samples in self._samples.items()
            if center in legal
            for sample in samples
        )
        labels = self._decode(f"source_prior::H={heldout}::e={source}", allowed)
        if {row.center for row in labels} != legal:
            raise ProtocolError("PSSCUR source-prior grant lacks exact legal centers.")
        self._prior_opened.add(key)
        self._record("source_prior", heldout, source, (), labels)
        return labels

    def open_crossing_donor_labels(
        self, outer_target_center: object, donor_center: object
    ) -> tuple[BinaryLabel, ...]:
        outer, donor = str(outer_target_center), str(donor_center)
        key = (outer, donor)
        if (
            key not in self._expected_donor
            or key in self._donor_opened
            or self._outer_support_opened
            or self._terminal_opened
        ):
            raise ProtocolError("PSSCUR donor capability opened out of order.")
        allowed = frozenset(
            sample
            for (center, _case), samples in self._samples.items()
            if center == donor
            for sample in samples
        )
        labels = self._decode(f"crossing_donor::outer_H={outer}::donor_J={donor}", allowed)
        if {row.center for row in labels} != {donor}:
            raise ProtocolError("PSSCUR donor grant escaped one donor center.")
        self._donor_opened.add(key)
        self._record("crossing_donor", outer, donor, (), labels)
        return labels

    def _require_donor_stage_complete(self) -> None:
        if self._prior_opened != self._expected_prior or self._donor_opened != self._expected_donor:
            raise ProtocolError("PSSCUR donor grants must finish before target support opens.")

    def open_outer_support_labels(
        self, target_center: object, case_id: object, *, plan_hash: str
    ) -> tuple[BinaryLabel, ...]:
        self._require_donor_stage_complete()
        key = (str(target_center), str(case_id))
        try:
            plan = self._outer[key]
        except KeyError as exc:
            raise ProtocolError("PSSCUR outer support requested for an unsealed route.") from exc
        if (
            require_sha256(plan_hash, "outer_plan_hash") != plan.plan_hash
            or key in self._outer_support_opened
            or self._terminal_opened
        ):
            raise ProtocolError("PSSCUR outer support capability duplicated or late.")
        allowed = frozenset(
            sample
            for support_case in plan.support_case_ids
            for sample in self._samples[(plan.target_center, support_case)]
        )
        labels = self._decode(
            f"outer_support::H={plan.target_center}::excluded_c={plan.case_id}",
            allowed,
        )
        if plan.case_id in {row.case_id for row in labels}:
            raise ProtocolError("PSSCUR held case entered its own endpoint fit.")
        self._outer_support_opened.add(key)
        self._record("outer_support", plan.target_center, None, (plan.case_id,), labels)
        return labels

    def record_outer_state_seal(
        self, target_center: object, case_id: object, state_hash: str
    ) -> None:
        key = (str(target_center), str(case_id))
        if key not in self._outer_support_opened or key in self._outer_state_seals:
            raise ProtocolError("PSSCUR endpoint state lacks authority or duplicates.")
        self._outer_state_seals[key] = require_sha256(state_hash, "outer_state_hash")

    def record_route_decision_seal(
        self, target_center: object, case_id: object, decision_hash: str
    ) -> None:
        key = (str(target_center), str(case_id))
        if (
            key not in self._outer_state_seals
            or key in self._route_decision_seals
            or self._terminal_opened
        ):
            raise ProtocolError("PSSCUR route decision lacks its H-c state seal.")
        self._route_decision_seals[key] = require_sha256(decision_hash, "route_decision_hash")

    def decision_barrier_payload(self) -> dict[str, object]:
        if set(self._outer_state_seals) != set(self._outer) or set(self._route_decision_seals) != set(self._outer):
            raise ProtocolError("PSSCUR route barrier requires every outer state and route.")
        payload = {
            "schema_version": "fixed_bank_psscur_decision_barrier_v1",
            "plan_seal_hash": self.plan_seal_hash,
            "outer_state_count": len(self._outer_state_seals),
            "route_decision_count": len(self._route_decision_seals),
            "outer_state_hash": canonical_hash(sorted(self._outer_state_seals.items())),
            "route_decision_hash": canonical_hash(sorted(self._route_decision_seals.items())),
            "double_exclusion_state_count": 0,
            "terminal_labels_used": False,
        }
        return {**payload, "decision_barrier_hash": canonical_hash(payload)}

    def record_aggregate_seal(self, aggregate_seal: Mapping[str, object]) -> None:
        if self._aggregate_seal is not None or self._terminal_opened:
            raise ProtocolError("PSSCUR aggregate seal duplicated or late.")
        barrier = self.decision_barrier_payload()
        self._aggregate_seal = validate_preterminal_aggregate_seal(
            aggregate_seal,
            expected_plan_seal_hash=self.plan_seal_hash,
            expected_decision_barrier_hash=str(barrier["decision_barrier_hash"]),
        )

    def open_terminal_labels(self) -> tuple[BinaryLabel, ...]:
        if self._aggregate_seal is None or self._terminal_opened:
            raise ProtocolError("PSSCUR terminal labels require a complete aggregate seal.")
        allowed = frozenset(sample for samples in self._samples.values() for sample in samples)
        labels = self._decode("terminal_evaluation", allowed)
        self._terminal_opened = True
        self._record("terminal_evaluation", None, None, (), labels)
        return labels

    def report_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_psscur_label_capability_report_v1",
            "status": "PASS" if self._terminal_opened else "INCOMPLETE",
            "plan_seal_hash": self.plan_seal_hash,
            "source_prior_grant_count": len(self._prior_opened),
            "crossing_donor_grant_count": len(self._donor_opened),
            "outer_support_grant_count": len(self._outer_support_opened),
            "outer_state_seal_count": len(self._outer_state_seals),
            "route_decision_seal_count": len(self._route_decision_seals),
            "double_exclusion_grant_count": 0,
            "aggregate_seal_hash": self._aggregate_seal,
            "terminal_opened": self._terminal_opened,
            "events": [event.to_payload() for event in self._events],
            "raw_labels_persisted": False,
        }


__all__ = (
    "LabelAccessEvent",
    "PSSCURLabelFirewall",
    "PRETERMINAL_AGGREGATE_KEYS",
    "validate_preterminal_aggregate_seal",
)
