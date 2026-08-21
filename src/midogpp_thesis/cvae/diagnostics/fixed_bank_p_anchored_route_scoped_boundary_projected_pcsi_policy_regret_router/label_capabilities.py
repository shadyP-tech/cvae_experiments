"""Role-scoped label capabilities and route-level noninterference ledger."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from ...protocol import ProtocolError
from .constants import (
    CENTERS,
    COMPOSED_POLICY_IDS,
    EXPECTED_DOUBLE_EXCLUSION_PAIR_COUNT,
    EXPECTED_POLICY_REPLAY_COUNT,
    EXPECTED_TOTAL_CASE_COUNT,
    PRIMARY_METHOD_ID,
    PROJECTION_GEOMETRY_ID,
    RAW_OBSERVED_MAX_METHOD_ID,
    UNPROJECTED_GEOMETRY_ID,
    candidate_sources,
)
from .contracts import BinaryLabel, LabelRoleRecord
from .hashing import canonical_hash, require_sha256
from .outer_plans import OuterPlanSeal


RACR_GEOMETRY_POLICY = {
    PROJECTION_GEOMETRY_ID: PRIMARY_METHOD_ID,
    UNPROJECTED_GEOMETRY_ID: RAW_OBSERVED_MAX_METHOD_ID,
}

PRETERMINAL_AGGREGATE_KEYS = frozenset(
    {
        "schema_version",
        "protocol_hash",
        "probability_surface_hash",
        "plan_seal_hash",
        "decision_barrier_hash",
        "donor_runtime_hash",
        "transport_hash",
        "policy_menu_hash",
        "policy_replay_hash",
        "authorization_hash",
        "final_prediction_hash",
        "workload_hash",
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
        str(payload.get("aggregate_seal_hash")),
        "aggregate_preterminal_seal_hash",
    )
    unhashed = {
        key: value
        for key, value in payload.items()
        if key != "aggregate_seal_hash"
    }
    digest_fields = PRETERMINAL_AGGREGATE_KEYS.difference(
        {"schema_version", "terminal_labels_used", "aggregate_seal_hash"}
    )
    if (
        set(payload) != PRETERMINAL_AGGREGATE_KEYS
        or payload.get("schema_version")
        != "fixed_bank_pcsi_racr_preterminal_aggregate_v1"
        or payload.get("plan_seal_hash")
        != require_sha256(expected_plan_seal_hash, "expected_plan_seal_hash")
        or payload.get("decision_barrier_hash")
        != require_sha256(
            expected_decision_barrier_hash,
            "expected_decision_barrier_hash",
        )
        or payload.get("terminal_labels_used") is not False
        or any(
            require_sha256(str(payload.get(key)), key) != payload.get(key)
            for key in digest_fields
        )
        or canonical_hash(unhashed) != submitted
    ):
        raise ProtocolError("PCSI-RACR aggregate seal escaped its barriers.")
    return submitted


@dataclass(frozen=True, order=True)
class LabelAccessEvent:
    role: str
    outer_target_center: str | None
    candidate_or_donor_center: str | None
    geometry_id: str | None
    route_case_id: str | None
    excluded_centers: tuple[str, ...]
    excluded_case_ids: tuple[str, ...]
    row_count: int
    case_count: int
    identity_hash: str

    def to_payload(self) -> dict[str, object]:
        return {
            **self.__dict__,
            "excluded_centers": list(self.excluded_centers),
            "excluded_case_ids": list(self.excluded_case_ids),
            "raw_labels_persisted": False,
        }


class PCSIRACRLabelFirewall:
    """The only raw-label decoder used by the route-scoped experiment."""

    def __init__(
        self,
        plan_seal: OuterPlanSeal,
        label_loader: Callable[
            [frozenset[tuple[str, str, str]], str], Sequence[object]
        ],
    ) -> None:
        if (
            plan_seal.strict_canonical_topology
            and len(plan_seal.outer_plans) != EXPECTED_TOTAL_CASE_COUNT
        ):
            raise ProtocolError("PCSI-RACR canonical firewall requires 218 plans.")
        if (
            len(plan_seal.double_exclusion_plans)
            != EXPECTED_DOUBLE_EXCLUSION_PAIR_COUNT
        ):
            raise ProtocolError("PCSI-RACR firewall requires all H/J plans.")
        self._seal = plan_seal
        self._outer = plan_seal.outer_by_key
        self._double = plan_seal.double_by_key
        self._samples = {
            plan.key: tuple(
                (plan.target_center, plan.case_id, sample_id)
                for sample_id in plan.evaluation_sample_ids
            )
            for plan in plan_seal.outer_plans
        }
        self._cases_by_center = {
            center: tuple(
                plan.case_id
                for plan in plan_seal.outer_plans
                if plan.target_center == center
            )
            for center in CENTERS
        }
        self._loader = label_loader
        self._expected_prior = frozenset(
            (heldout, source)
            for heldout in CENTERS
            for source in candidate_sources(heldout)
        )
        self._expected_donor = frozenset(
            (outer, donor)
            for outer in CENTERS
            for donor in CENTERS
            if donor != outer
        )
        self._expected_pseudo = frozenset(
            (geometry, outer, donor, case_id)
            for geometry in RACR_GEOMETRY_POLICY
            for outer, donor in self._double
            for case_id in self._cases_by_center[donor]
        )
        self._expected_target = frozenset(
            (policy, center, case_id)
            for policy in COMPOSED_POLICY_IDS
            for center in CENTERS
            for case_id in self._cases_by_center[center]
        )
        self._expected_calibration = frozenset(
            (geometry, center)
            for geometry in RACR_GEOMETRY_POLICY
            for center in CENTERS
        )
        self._prior_opened: set[tuple[str, str]] = set()
        self._donor_opened: set[tuple[str, str]] = set()
        self._outer_support_opened: set[tuple[str, str]] = set()
        self._outer_state_seals: dict[tuple[str, str], str] = {}
        self._target_case_seals: dict[tuple[str, str, str], str] = {}
        self._pseudo_case_seals: dict[tuple[str, str, str, str], str] = {}
        self._pre_evaluation_seal: str | None = None
        self._transport_hash: str | None = None
        self._match_hash: str | None = None
        self._pseudo_evaluation_opened: set[
            tuple[str, str, str, str]
        ] = set()
        self._replay_seals: dict[tuple[str, str, str, str], str] = {}
        self._calibration_seals: dict[tuple[str, str], str] = {}
        self._decision_seals: dict[tuple[str, str, str], str] = {}
        self._aggregate_seal: str | None = None
        self._terminal_opened = False
        self._events: list[LabelAccessEvent] = []
        self._role_ledger: list[LabelRoleRecord] = []

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
                str(
                    getattr(
                        row,
                        "sample_id",
                        getattr(row, "evaluation_row_id", ""),
                    )
                ),
                int(getattr(row, "value", getattr(row, "label", -1))),
                role,
            )
            for row in raw
        )
        if (
            {row.key for row in labels} != set(allowed)
            or len(labels) != len(allowed)
        ):
            raise ProtocolError("PCSI-RACR label loader escaped its capability.")
        return tuple(sorted(labels, key=lambda row: row.key))

    def _record(
        self,
        *,
        role: str,
        outer: str | None,
        other: str | None,
        geometry: str | None,
        route_case: str | None,
        phase: str,
        excluded_centers: tuple[str, ...],
        excluded_cases: tuple[str, ...],
        labels: Sequence[BinaryLabel],
        permitted: bool = True,
    ) -> None:
        identity_hash = canonical_hash([list(row.key) for row in labels])
        self._events.append(
            LabelAccessEvent(
                role,
                outer,
                other,
                geometry,
                route_case,
                excluded_centers,
                excluded_cases,
                len(labels),
                len({(row.center, row.case_id) for row in labels}),
                identity_hash,
            )
        )
        outer_value = outer or (labels[0].center if labels else CENTERS[0])
        route_value = route_case or "*"
        self._role_ledger.extend(
            LabelRoleRecord(
                canonical_hash([row.center, row.case_id, row.sample_id]),
                row.center,
                row.case_id,
                outer_value,
                route_value,
                role,
                phase,
                permitted,
            )
            for row in labels
        )

    def _record_forbidden_own_case(
        self,
        center: str,
        case_id: str,
        *,
        outer: str,
        role_prefix: str,
        phase: str,
    ) -> None:
        for role in (
            "endpoint_fit",
            "posterior_fit",
            "transport_candidate",
            "candidate_policy",
        ):
            for sample in self._samples[(center, case_id)]:
                self._role_ledger.append(
                    LabelRoleRecord(
                        canonical_hash(list(sample)),
                        center,
                        case_id,
                        outer,
                        case_id,
                        f"FORBIDDEN::{role_prefix}::{role}",
                        phase,
                        False,
                    )
                )

    def open_source_prior_labels(
        self, heldout_center: object, candidate_source: object
    ) -> tuple[BinaryLabel, ...]:
        heldout, source = str(heldout_center), str(candidate_source)
        key = heldout, source
        if (
            key not in self._expected_prior
            or key in self._prior_opened
            or self._outer_support_opened
            or self._terminal_opened
        ):
            raise ProtocolError("PCSI-RACR source-prior grant opened out of order.")
        legal = set(CENTERS).difference({heldout, source})
        allowed = frozenset(
            sample
            for (center, _case), samples in self._samples.items()
            if center in legal
            for sample in samples
        )
        role = f"source_prior::heldout={heldout}::source={source}"
        labels = self._decode(role, allowed)
        self._prior_opened.add(key)
        self._record(
            role=role,
            outer=heldout,
            other=source,
            geometry=None,
            route_case=None,
            phase="SOURCE_PRIOR",
            excluded_centers=tuple(sorted({heldout, source})),
            excluded_cases=(),
            labels=labels,
        )
        return labels

    def open_utility_donor_labels(
        self, outer_target_center: object, donor_center: object
    ) -> tuple[BinaryLabel, ...]:
        outer, donor = str(outer_target_center), str(donor_center)
        key = outer, donor
        if (
            key not in self._expected_donor
            or key in self._donor_opened
            or self._outer_support_opened
            or self._terminal_opened
        ):
            raise ProtocolError("PCSI-RACR donor-response grant opened out of order.")
        allowed = frozenset(
            sample
            for (center, _case), samples in self._samples.items()
            if center == donor
            for sample in samples
        )
        role = f"utility_donor::outer_H={outer}::donor_J={donor}"
        labels = self._decode(role, allowed)
        self._donor_opened.add(key)
        self._record(
            role=role,
            outer=outer,
            other=donor,
            geometry=None,
            route_case=None,
            phase="UTILITY_RESPONSE",
            excluded_centers=(outer,),
            excluded_cases=(),
            labels=labels,
        )
        return labels

    def open_crossing_donor_labels(
        self, outer_target_center: object, donor_center: object
    ) -> tuple[BinaryLabel, ...]:
        return self.open_utility_donor_labels(
            outer_target_center, donor_center
        )

    def _require_donor_stage_complete(self) -> None:
        if (
            self._prior_opened != self._expected_prior
            or self._donor_opened != self._expected_donor
        ):
            raise ProtocolError(
                "PCSI-RACR source-prior and donor roles are incomplete."
            )

    def open_outer_support_labels(
        self,
        target_center: object,
        case_id: object,
        *,
        plan_hash: str,
    ) -> tuple[BinaryLabel, ...]:
        self._require_donor_stage_complete()
        key = str(target_center), str(case_id)
        try:
            plan = self._outer[key]
        except KeyError as exc:
            raise ProtocolError("PCSI-RACR support route was not sealed.") from exc
        if (
            require_sha256(plan_hash, "outer_plan_hash") != plan.plan_hash
            or key in self._outer_support_opened
            or self._terminal_opened
        ):
            raise ProtocolError("PCSI-RACR support capability duplicated or late.")
        allowed = frozenset(
            sample
            for support_case in plan.support_case_ids
            for sample in self._samples[(plan.target_center, support_case)]
        )
        role = f"outer_support::H={plan.target_center}::excluded_c={plan.case_id}"
        labels = self._decode(role, allowed)
        if plan.case_id in {row.case_id for row in labels}:
            raise ProtocolError("PCSI-RACR held case entered its own support.")
        self._outer_support_opened.add(key)
        self._record(
            role=role,
            outer=plan.target_center,
            other=None,
            geometry=None,
            route_case=plan.case_id,
            phase="ROUTE_SUPPORT",
            excluded_centers=(),
            excluded_cases=(plan.case_id,),
            labels=labels,
        )
        self._record_forbidden_own_case(
            plan.target_center,
            plan.case_id,
            outer=plan.target_center,
            role_prefix="TARGET",
            phase="PRE_EVALUATION",
        )
        return labels

    def record_outer_state_seal(
        self, target_center: object, case_id: object, state_hash: str
    ) -> None:
        key = str(target_center), str(case_id)
        if (
            key not in self._outer_support_opened
            or key in self._outer_state_seals
        ):
            raise ProtocolError("PCSI-RACR endpoint state lacks its H\\c grant.")
        self._outer_state_seals[key] = require_sha256(
            state_hash, "outer_state_hash"
        )

    def _require_all_outer_states(self) -> None:
        if set(self._outer_state_seals) != set(self._outer):
            raise ProtocolError("PCSI-RACR requires all route state seals.")

    def record_target_case_policy_seal(
        self,
        target_center: object,
        case_id: object,
        policy_id: object,
        case_policy_hash: str,
    ) -> None:
        self._require_all_outer_states()
        key = str(policy_id), str(target_center), str(case_id)
        if key not in self._expected_target or key in self._target_case_seals:
            raise ProtocolError("PCSI-RACR target policy seal drifted.")
        self._target_case_seals[key] = require_sha256(
            case_policy_hash, "target_case_policy_hash"
        )

    def record_pseudo_case_policy_seal(
        self,
        outer_target_center: object,
        pseudo_target_center: object,
        geometry_id: object,
        case_id: object,
        case_policy_hash: str,
    ) -> None:
        self._require_all_outer_states()
        key = (
            str(geometry_id),
            str(outer_target_center),
            str(pseudo_target_center),
            str(case_id),
        )
        if (
            key not in self._expected_pseudo
            or key in self._pseudo_case_seals
            or self._pre_evaluation_seal is not None
        ):
            raise ProtocolError("PCSI-RACR pseudo case policy seal drifted.")
        self._pseudo_case_seals[key] = require_sha256(
            case_policy_hash, "pseudo_case_policy_hash"
        )
        self._record_forbidden_own_case(
            key[2],
            key[3],
            outer=key[1],
            role_prefix="PSEUDO",
            phase="PRE_EVALUATION",
        )

    def record_pre_evaluation_seal(
        self,
        seal_hash: str,
        *,
        transport_hash: str,
        match_hash: str,
    ) -> None:
        if (
            self._pre_evaluation_seal is not None
            or set(self._target_case_seals) != set(self._expected_target)
            or set(self._pseudo_case_seals) != set(self._expected_pseudo)
        ):
            raise ProtocolError("PCSI-RACR pre-evaluation barrier is incomplete.")
        self._pre_evaluation_seal = require_sha256(
            seal_hash, "pre_evaluation_seal_hash"
        )
        self._transport_hash = require_sha256(
            transport_hash, "transport_runtime_hash"
        )
        self._match_hash = require_sha256(match_hash, "descriptor_match_hash")

    def open_pseudo_evaluation_labels(
        self,
        outer_target_center: object,
        pseudo_target_center: object,
        geometry_id: object,
        case_id: object,
        *,
        policy_seal_hash: str,
        pre_evaluation_seal_hash: str,
    ) -> tuple[BinaryLabel, ...]:
        key = (
            str(geometry_id),
            str(outer_target_center),
            str(pseudo_target_center),
            str(case_id),
        )
        if (
            key not in self._expected_pseudo
            or key in self._pseudo_evaluation_opened
            or self._pre_evaluation_seal is None
            or require_sha256(
                pre_evaluation_seal_hash,
                "pre_evaluation_seal_hash",
            )
            != self._pre_evaluation_seal
            or require_sha256(
                policy_seal_hash, "pseudo_case_policy_hash"
            )
            != self._pseudo_case_seals[key]
            or self._terminal_opened
        ):
            raise ProtocolError("PCSI-RACR pseudo label grant escaped its seal.")
        allowed = frozenset(self._samples[(key[2], key[3])])
        role = (
            f"pseudo_evaluation::geometry={key[0]}::outer_H={key[1]}::"
            f"pseudo_J={key[2]}::case={key[3]}"
        )
        labels = self._decode(role, allowed)
        self._pseudo_evaluation_opened.add(key)
        self._record(
            role=role,
            outer=key[1],
            other=key[2],
            geometry=key[0],
            route_case=key[3],
            phase="PSEUDO_REPLAY",
            excluded_centers=(key[1],),
            excluded_cases=(),
            labels=labels,
        )
        return labels

    def record_pseudo_replay_seal(
        self,
        outer_target_center: object,
        pseudo_target_center: object,
        geometry_id: object,
        case_id: object,
        replay_hash: str,
    ) -> None:
        key = (
            str(geometry_id),
            str(outer_target_center),
            str(pseudo_target_center),
            str(case_id),
        )
        if (
            key not in self._pseudo_evaluation_opened
            or key in self._replay_seals
        ):
            raise ProtocolError("PCSI-RACR replay lacks its route grant.")
        self._replay_seals[key] = require_sha256(
            replay_hash, "pseudo_case_replay_hash"
        )

    def record_calibration_seal(
        self, geometry_id: object, outer_center: object, seal_hash: str
    ) -> None:
        key = str(geometry_id), str(outer_center)
        expected_replays = {
            route
            for route in self._expected_pseudo
            if route[0] == key[0] and route[1] == key[1]
        }
        if (
            key not in self._expected_calibration
            or key in self._calibration_seals
            or not expected_replays.issubset(self._replay_seals)
        ):
            raise ProtocolError("PCSI-RACR calibration seal is premature.")
        self._calibration_seals[key] = require_sha256(
            seal_hash, "route_calibration_hash"
        )

    def record_target_decision_seal(
        self,
        policy_id: object,
        target_center: object,
        case_id: object,
        decision_hash: str,
    ) -> None:
        key = str(policy_id), str(target_center), str(case_id)
        if (
            key not in self._expected_target
            or key in self._decision_seals
            or set(self._calibration_seals) != set(self._expected_calibration)
        ):
            raise ProtocolError("PCSI-RACR target decision seal drifted.")
        self._decision_seals[key] = require_sha256(
            decision_hash, "route_decision_hash"
        )

    def decision_barrier_payload(self) -> dict[str, object]:
        if (
            self._pre_evaluation_seal is None
            or set(self._pseudo_evaluation_opened) != set(self._expected_pseudo)
            or set(self._replay_seals) != set(self._expected_pseudo)
            or set(self._calibration_seals) != set(self._expected_calibration)
            or set(self._decision_seals) != set(self._expected_target)
        ):
            raise ProtocolError("PCSI-RACR decision barrier is incomplete.")
        payload = {
            "schema_version": "fixed_bank_pcsi_racr_decision_barrier_v1",
            "plan_seal_hash": self.plan_seal_hash,
            "pre_evaluation_seal_hash": self._pre_evaluation_seal,
            "transport_hash": self._transport_hash,
            "descriptor_match_hash": self._match_hash,
            "outer_state_count": len(self._outer_state_seals),
            "target_case_policy_seal_count": len(self._target_case_seals),
            "pseudo_case_policy_seal_count": len(self._pseudo_case_seals),
            "pseudo_evaluation_grant_count": len(
                self._pseudo_evaluation_opened
            ),
            "policy_replay_count": len(self._replay_seals),
            "calibration_count": len(self._calibration_seals),
            "target_decision_count": len(self._decision_seals),
            "policy_replay_hash": canonical_hash(
                sorted(self._replay_seals.items())
            ),
            "calibration_hash": canonical_hash(
                sorted(self._calibration_seals.items())
            ),
            "target_decision_hash": canonical_hash(
                sorted(self._decision_seals.items())
            ),
            "terminal_labels_used": False,
        }
        return {**payload, "decision_barrier_hash": canonical_hash(payload)}

    def record_aggregate_seal(
        self, aggregate_seal: Mapping[str, object]
    ) -> None:
        if self._aggregate_seal is not None or self._terminal_opened:
            raise ProtocolError("PCSI-RACR aggregate seal duplicated or late.")
        barrier = self.decision_barrier_payload()
        self._aggregate_seal = validate_preterminal_aggregate_seal(
            aggregate_seal,
            expected_plan_seal_hash=self.plan_seal_hash,
            expected_decision_barrier_hash=str(
                barrier["decision_barrier_hash"]
            ),
        )

    def open_terminal_labels(self) -> tuple[BinaryLabel, ...]:
        if self._aggregate_seal is None or self._terminal_opened:
            raise ProtocolError(
                "PCSI-RACR terminal labels require DecisionSeal."
            )
        allowed = frozenset(
            sample for samples in self._samples.values() for sample in samples
        )
        labels = self._decode("target_terminal_evaluation", allowed)
        self._terminal_opened = True
        by_case = {
            key: tuple(row for row in labels if (row.center, row.case_id) == key)
            for key in self._samples
        }
        for (center, case_id), rows in by_case.items():
            self._record(
                role=f"target_terminal::H={center}::case={case_id}",
                outer=center,
                other=None,
                geometry=None,
                route_case=case_id,
                phase="TARGET_TERMINAL",
                excluded_centers=(),
                excluded_cases=(),
                labels=rows,
            )
        return labels

    def report_payload(self) -> dict[str, object]:
        complete = bool(
            self._terminal_opened
            and (
                not self._seal.strict_canonical_topology
                or len(self._replay_seals) == EXPECTED_POLICY_REPLAY_COUNT
            )
        )
        return {
            "schema_version": "fixed_bank_pcsi_racr_label_capability_report_v1",
            "status": "PASS" if complete else "INCOMPLETE",
            "plan_seal_hash": self.plan_seal_hash,
            "source_prior_grant_count": len(self._prior_opened),
            "utility_donor_grant_count": len(self._donor_opened),
            "outer_support_grant_count": len(self._outer_support_opened),
            "outer_state_seal_count": len(self._outer_state_seals),
            "target_case_policy_seal_count": len(self._target_case_seals),
            "pseudo_case_policy_seal_count": len(self._pseudo_case_seals),
            "pseudo_evaluation_grant_count": len(
                self._pseudo_evaluation_opened
            ),
            "policy_replay_seal_count": len(self._replay_seals),
            "calibration_seal_count": len(self._calibration_seals),
            "target_decision_seal_count": len(self._decision_seals),
            "aggregate_seal_hash": self._aggregate_seal,
            "terminal_opened": self._terminal_opened,
            "events": [event.to_payload() for event in self._events],
            "label_role_ledger": [
                row.to_payload()
                for row in sorted(
                    self._role_ledger,
                    key=lambda row: (
                        row.outer_center,
                        row.route_case_id,
                        row.label_center,
                        row.label_case_id,
                        row.role,
                        row.label_identity_hash,
                    ),
                )
            ],
            "own_route_noninterference_required": True,
            "global_label_invariance_claimed": False,
            "raw_labels_persisted": False,
        }


__all__ = (
    "LabelAccessEvent",
    "RACR_GEOMETRY_POLICY",
    "PCSIRACRLabelFirewall",
    "PRETERMINAL_AGGREGATE_KEYS",
    "validate_preterminal_aggregate_seal",
)
