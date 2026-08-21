"""Role-scoped, fail-closed label capabilities for PCSI-PARC.

Labels are deliberately decoded more than once under disjoint roles. A donor
center may provide utility responses and later act as a pseudo evaluation
center, but the two grants are different immutable objects and the latter is
unavailable until every case decision plus the complete center policy has been
sealed. The runtime never receives a general-purpose label table.
"""

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
    UNPROJECTED_GEOMETRY_ID,
    UNPROJECTED_PARC_METHOD_ID,
    candidate_sources,
)
from .contracts import BinaryLabel
from .hashing import canonical_hash, require_sha256
from .outer_plans import OuterPlanSeal


PARC_GEOMETRY_POLICY = {
    PROJECTION_GEOMETRY_ID: PRIMARY_METHOD_ID,
    UNPROJECTED_GEOMETRY_ID: UNPROJECTED_PARC_METHOD_ID,
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
        str(payload.get("aggregate_seal_hash")), "aggregate_preterminal_seal_hash"
    )
    unhashed = {
        key: value for key, value in payload.items() if key != "aggregate_seal_hash"
    }
    digest_fields = set(PRETERMINAL_AGGREGATE_KEYS).difference(
        {"schema_version", "terminal_labels_used", "aggregate_seal_hash"}
    )
    if (
        set(payload) != PRETERMINAL_AGGREGATE_KEYS
        or payload.get("schema_version")
        != "fixed_bank_pcsi_parc_preterminal_aggregate_v1"
        or payload.get("plan_seal_hash")
        != require_sha256(expected_plan_seal_hash, "expected_plan_seal_hash")
        or payload.get("decision_barrier_hash")
        != require_sha256(
            expected_decision_barrier_hash, "expected_decision_barrier_hash"
        )
        or payload.get("terminal_labels_used") is not False
        or any(
            require_sha256(str(payload.get(key)), key) != payload.get(key)
            for key in digest_fields
        )
        or canonical_hash(unhashed) != submitted
    ):
        raise ProtocolError("PCSI-PARC aggregate seal is not bound to its barriers.")
    return submitted


@dataclass(frozen=True, order=True)
class LabelAccessEvent:
    role: str
    outer_target_center: str | None
    candidate_or_donor_center: str | None
    geometry_id: str | None
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


class PCSIPARCLabelFirewall:
    """Sole raw-label decoder for every training and evaluation role."""

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
            raise ProtocolError("PCSI-PARC canonical firewall requires all outer plans.")
        if len(plan_seal.double_exclusion_plans) != EXPECTED_DOUBLE_EXCLUSION_PAIR_COUNT:
            raise ProtocolError("PCSI-PARC firewall requires all ordered H/J plans.")
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
            (geometry, outer, pseudo)
            for geometry in PARC_GEOMETRY_POLICY
            for outer, pseudo in self._double
        )
        self._expected_target_centers = frozenset(
            (policy, center)
            for policy in COMPOSED_POLICY_IDS
            for center in CENTERS
        )
        self._prior_opened: set[tuple[str, str]] = set()
        self._donor_opened: set[tuple[str, str]] = set()
        self._outer_support_opened: set[tuple[str, str]] = set()
        self._outer_state_seals: dict[tuple[str, str], str] = {}
        self._target_case_seals: dict[tuple[str, str, str], str] = {}
        self._target_policy_seals: dict[tuple[str, str], str] = {}
        self._pseudo_case_seals: dict[tuple[str, str, str, str], str] = {}
        self._pseudo_policy_seals: dict[tuple[str, str, str], str] = {}
        self._pseudo_evaluation_opened: set[tuple[str, str, str]] = set()
        self._replay_seals: dict[tuple[str, str, str], str] = {}
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
        if {row.key for row in labels} != set(allowed) or len(labels) != len(allowed):
            raise ProtocolError("PCSI-PARC label loader escaped its exact capability.")
        return tuple(sorted(labels, key=lambda row: row.key))

    def _record(
        self,
        role: str,
        outer: str | None,
        other: str | None,
        geometry: str | None,
        excluded_centers: tuple[str, ...],
        excluded_cases: tuple[str, ...],
        labels: Sequence[BinaryLabel],
    ) -> None:
        self._events.append(
            LabelAccessEvent(
                role,
                outer,
                other,
                geometry,
                excluded_centers,
                excluded_cases,
                len(labels),
                len({(row.center, row.case_id) for row in labels}),
                canonical_hash([list(row.key) for row in labels]),
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
            raise ProtocolError("PCSI-PARC source-prior capability opened out of order.")
        legal = set(CENTERS).difference({heldout, source})
        allowed = frozenset(
            sample
            for (center, _case), samples in self._samples.items()
            if center in legal
            for sample in samples
        )
        role = f"source_prior::heldout={heldout}::source={source}"
        labels = self._decode(role, allowed)
        if {row.center for row in labels} != legal:
            raise ProtocolError("PCSI-PARC source-prior grant lacks exact legal centers.")
        self._prior_opened.add(key)
        self._record(
            role,
            heldout,
            source,
            None,
            tuple(sorted({heldout, source})),
            (),
            labels,
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
            raise ProtocolError("PCSI-PARC utility-donor capability opened out of order.")
        allowed = frozenset(
            sample
            for (center, _case), samples in self._samples.items()
            if center == donor
            for sample in samples
        )
        role = f"utility_donor::outer_H={outer}::donor_J={donor}"
        labels = self._decode(role, allowed)
        if {row.center for row in labels} != {donor}:
            raise ProtocolError("PCSI-PARC donor grant escaped one donor center.")
        self._donor_opened.add(key)
        self._record(role, outer, donor, None, (outer,), (), labels)
        return labels

    # Compatibility facade for package-local fresh-legacy helpers.
    def open_crossing_donor_labels(
        self, outer_target_center: object, donor_center: object
    ) -> tuple[BinaryLabel, ...]:
        return self.open_utility_donor_labels(outer_target_center, donor_center)

    def _require_donor_stage_complete(self) -> None:
        if (
            self._prior_opened != self._expected_prior
            or self._donor_opened != self._expected_donor
        ):
            raise ProtocolError(
                "PCSI-PARC prior and donor grants must finish before target support."
            )

    def open_outer_support_labels(
        self, target_center: object, case_id: object, *, plan_hash: str
    ) -> tuple[BinaryLabel, ...]:
        self._require_donor_stage_complete()
        key = str(target_center), str(case_id)
        try:
            plan = self._outer[key]
        except KeyError as exc:
            raise ProtocolError(
                "PCSI-PARC support requested for an unsealed H\\c route."
            ) from exc
        if (
            require_sha256(plan_hash, "outer_plan_hash") != plan.plan_hash
            or key in self._outer_support_opened
            or self._terminal_opened
        ):
            raise ProtocolError("PCSI-PARC outer support capability duplicated or late.")
        allowed = frozenset(
            sample
            for support_case in plan.support_case_ids
            for sample in self._samples[(plan.target_center, support_case)]
        )
        role = f"outer_support::H={plan.target_center}::excluded_c={plan.case_id}"
        labels = self._decode(role, allowed)
        if plan.case_id in {row.case_id for row in labels}:
            raise ProtocolError("PCSI-PARC held case entered its own endpoint fit.")
        self._outer_support_opened.add(key)
        self._record(role, plan.target_center, None, None, (), (plan.case_id,), labels)
        return labels

    def record_outer_state_seal(
        self, target_center: object, case_id: object, state_hash: str
    ) -> None:
        key = str(target_center), str(case_id)
        if key not in self._outer_support_opened or key in self._outer_state_seals:
            raise ProtocolError("PCSI-PARC endpoint state lacks its H\\c capability.")
        self._outer_state_seals[key] = require_sha256(state_hash, "outer_state_hash")

    def _require_all_outer_states(self) -> None:
        if set(self._outer_state_seals) != set(self._outer):
            raise ProtocolError("PCSI-PARC policies require all 218 H\\c state seals.")

    def record_target_case_policy_seal(
        self,
        target_center: object,
        case_id: object,
        policy_id: object,
        case_policy_hash: str,
    ) -> None:
        self._require_all_outer_states()
        center, case, policy = str(target_center), str(case_id), str(policy_id)
        key = policy, center, case
        if (
            policy not in COMPOSED_POLICY_IDS
            or (center, case) not in self._outer
            or key in self._target_case_seals
            or (policy, center) in self._target_policy_seals
            or self._terminal_opened
        ):
            raise ProtocolError("PCSI-PARC target case policy seal drifted.")
        self._target_case_seals[key] = require_sha256(
            case_policy_hash, "target_case_policy_hash"
        )

    def record_target_center_policy_seal(
        self, target_center: object, policy_id: object, policy_seal_hash: str
    ) -> None:
        center, policy = str(target_center), str(policy_id)
        key = policy, center
        expected_cases = set(self._cases_by_center.get(center, ()))
        observed_cases = {
            case
            for observed_policy, observed_center, case in self._target_case_seals
            if (observed_policy, observed_center) == key
        }
        if (
            key not in self._expected_target_centers
            or key in self._target_policy_seals
            or observed_cases != expected_cases
            or self._terminal_opened
        ):
            raise ProtocolError(
                "PCSI-PARC target center policy requires every case seal first."
            )
        self._target_policy_seals[key] = require_sha256(
            policy_seal_hash, "target_center_policy_hash"
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
        geometry, outer, pseudo, case = (
            str(geometry_id),
            str(outer_target_center),
            str(pseudo_target_center),
            str(case_id),
        )
        state = geometry, outer, pseudo
        key = (*state, case)
        plan = self._double.get((outer, pseudo))
        if (
            state not in self._expected_pseudo
            or plan is None
            or case not in plan.pseudo_case_ids
            or key in self._pseudo_case_seals
            or state in self._pseudo_policy_seals
            or self._pseudo_evaluation_opened
            or self._terminal_opened
        ):
            raise ProtocolError("PCSI-PARC pseudo case seal escaped its H/J plan.")
        self._pseudo_case_seals[key] = require_sha256(
            case_policy_hash, "pseudo_case_policy_hash"
        )

    def record_pseudo_policy_seal(
        self,
        outer_target_center: object,
        pseudo_target_center: object,
        geometry_id: object,
        policy_seal_hash: str,
    ) -> None:
        state = (
            str(geometry_id),
            str(outer_target_center),
            str(pseudo_target_center),
        )
        plan = self._double.get((state[1], state[2]))
        observed = {
            case
            for geometry, outer, pseudo, case in self._pseudo_case_seals
            if (geometry, outer, pseudo) == state
        }
        if (
            state not in self._expected_pseudo
            or plan is None
            or observed != set(plan.pseudo_case_ids)
            or state in self._pseudo_policy_seals
            or self._pseudo_evaluation_opened
            or self._terminal_opened
        ):
            raise ProtocolError(
                "PCSI-PARC pseudo policy requires all per-case decisions first."
            )
        self._pseudo_policy_seals[state] = require_sha256(
            policy_seal_hash, "pseudo_center_policy_hash"
        )

    def open_pseudo_evaluation_labels(
        self,
        outer_target_center: object,
        pseudo_target_center: object,
        geometry_id: object,
        *,
        policy_seal_hash: str,
    ) -> tuple[BinaryLabel, ...]:
        state = (
            str(geometry_id),
            str(outer_target_center),
            str(pseudo_target_center),
        )
        sealed = self._pseudo_policy_seals.get(state)
        if (
            state not in self._expected_pseudo
            or sealed is None
            or require_sha256(policy_seal_hash, "pseudo_center_policy_hash") != sealed
            or set(self._target_policy_seals) != set(self._expected_target_centers)
            or set(self._pseudo_policy_seals) != set(self._expected_pseudo)
            or state in self._pseudo_evaluation_opened
            or self._terminal_opened
        ):
            raise ProtocolError(
                "PCSI-PARC pseudo labels require every target and pseudo policy seal."
            )
        geometry, outer, pseudo = state
        allowed = frozenset(
            sample
            for case in self._cases_by_center[pseudo]
            for sample in self._samples[(pseudo, case)]
        )
        role = (
            f"pseudo_evaluation::geometry={geometry}::outer_H={outer}::pseudo_J={pseudo}"
        )
        labels = self._decode(role, allowed)
        self._pseudo_evaluation_opened.add(state)
        self._record(
            role,
            outer,
            pseudo,
            geometry,
            tuple(center for center in CENTERS if center in {outer, pseudo}),
            (),
            labels,
        )
        return labels

    def record_policy_replay_seal(
        self,
        outer_target_center: object,
        pseudo_target_center: object,
        geometry_id: object,
        replay_hash: str,
    ) -> None:
        state = (
            str(geometry_id),
            str(outer_target_center),
            str(pseudo_target_center),
        )
        if state not in self._pseudo_evaluation_opened or state in self._replay_seals:
            raise ProtocolError("PCSI-PARC replay seal lacks its scoped evaluation.")
        self._replay_seals[state] = require_sha256(replay_hash, "policy_replay_hash")

    def decision_barrier_payload(self) -> dict[str, object]:
        if (
            set(self._outer_state_seals) != set(self._outer)
            or set(self._target_policy_seals) != set(self._expected_target_centers)
            or set(self._pseudo_policy_seals) != set(self._expected_pseudo)
            or set(self._pseudo_evaluation_opened) != set(self._expected_pseudo)
            or set(self._replay_seals) != set(self._expected_pseudo)
        ):
            raise ProtocolError("PCSI-PARC decision barrier is incomplete.")
        payload = {
            "schema_version": "fixed_bank_pcsi_parc_decision_barrier_v1",
            "plan_seal_hash": self.plan_seal_hash,
            "outer_state_count": len(self._outer_state_seals),
            "target_case_policy_seal_count": len(self._target_case_seals),
            "target_center_policy_seal_count": len(self._target_policy_seals),
            "double_exclusion_pair_count": len(self._double),
            "double_exclusion_state_count": len(self._pseudo_policy_seals),
            "pseudo_case_policy_seal_count": len(self._pseudo_case_seals),
            "policy_replay_count": len(self._replay_seals),
            "outer_state_hash": canonical_hash(sorted(self._outer_state_seals.items())),
            "target_policy_hash": canonical_hash(
                sorted(self._target_policy_seals.items())
            ),
            "pseudo_policy_hash": canonical_hash(
                sorted(self._pseudo_policy_seals.items())
            ),
            "policy_replay_hash": canonical_hash(sorted(self._replay_seals.items())),
            "terminal_labels_used": False,
        }
        return {**payload, "decision_barrier_hash": canonical_hash(payload)}

    def record_aggregate_seal(self, aggregate_seal: Mapping[str, object]) -> None:
        if self._aggregate_seal is not None or self._terminal_opened:
            raise ProtocolError("PCSI-PARC aggregate seal duplicated or late.")
        barrier = self.decision_barrier_payload()
        self._aggregate_seal = validate_preterminal_aggregate_seal(
            aggregate_seal,
            expected_plan_seal_hash=self.plan_seal_hash,
            expected_decision_barrier_hash=str(barrier["decision_barrier_hash"]),
        )

    def open_terminal_labels(self) -> tuple[BinaryLabel, ...]:
        if self._aggregate_seal is None or self._terminal_opened:
            raise ProtocolError("PCSI-PARC terminal labels require the aggregate seal.")
        allowed = frozenset(
            sample for samples in self._samples.values() for sample in samples
        )
        labels = self._decode("terminal_evaluation", allowed)
        self._terminal_opened = True
        self._record("terminal_evaluation", None, None, None, (), (), labels)
        return labels

    def report_payload(self) -> dict[str, object]:
        canonical = self._seal.strict_canonical_topology
        complete = bool(
            self._terminal_opened
            and (
                not canonical
                or len(self._replay_seals) == EXPECTED_POLICY_REPLAY_COUNT
            )
        )
        return {
            "schema_version": "fixed_bank_pcsi_parc_label_capability_report_v1",
            "status": "PASS" if complete else "INCOMPLETE",
            "plan_seal_hash": self.plan_seal_hash,
            "source_prior_grant_count": len(self._prior_opened),
            "utility_donor_grant_count": len(self._donor_opened),
            "outer_support_grant_count": len(self._outer_support_opened),
            "outer_state_seal_count": len(self._outer_state_seals),
            "target_case_policy_seal_count": len(self._target_case_seals),
            "target_center_policy_seal_count": len(self._target_policy_seals),
            "double_exclusion_pair_count": len(self._double),
            "double_exclusion_state_count": len(self._pseudo_policy_seals),
            "pseudo_evaluation_grant_count": len(self._pseudo_evaluation_opened),
            "policy_replay_seal_count": len(self._replay_seals),
            "aggregate_seal_hash": self._aggregate_seal,
            "terminal_opened": self._terminal_opened,
            "events": [event.to_payload() for event in self._events],
            "raw_labels_persisted": False,
        }


__all__ = (
    "LabelAccessEvent",
    "PARC_GEOMETRY_POLICY",
    "PCSIPARCLabelFirewall",
    "PRETERMINAL_AGGREGATE_KEYS",
    "validate_preterminal_aggregate_seal",
)
