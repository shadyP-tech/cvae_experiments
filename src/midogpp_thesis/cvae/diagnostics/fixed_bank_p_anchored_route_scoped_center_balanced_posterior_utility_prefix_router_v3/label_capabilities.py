"""Role-scoped label firewall for CBPUPR's sealed phase chain."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from ...protocol import ProtocolError
from .constants import (
    CENTERS,
    COMPOSED_POLICY_IDS,
    EXPECTED_PSEUDO_ROUTE_COUNT,
    EXPECTED_TOTAL_CASE_COUNT,
)
from .contracts import BinaryLabel
from .hashing import canonical_hash, require_sha256
from .outer_plans import OuterPlanSeal


@dataclass(frozen=True, order=True)
class LabelAccessEvent:
    role: str
    outer_target_center: str | None
    target_center: str | None
    case_id: str | None
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


class CBPUPRLabelFirewall:
    """The only component allowed to decode raw manifest labels.

    The firewall records hashes and counts only.  Candidate, replay, decision,
    and persistence code receive scoped ``BinaryLabel`` tuples or aggregate
    seals, never a global label mapping.
    """

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
            raise ProtocolError("CBPUPR firewall requires all outer routes.")
        if len(plan_seal.double_exclusion_plans) != len(CENTERS) * (len(CENTERS) - 1):
            raise ProtocolError("CBPUPR firewall requires all ordered H/J plans.")
        self._seal = plan_seal
        self._loader = label_loader
        self._samples = {
            row.key: tuple(
                (row.target_center, row.case_id, sample_id)
                for sample_id in row.evaluation_sample_ids
            )
            for row in plan_seal.outer_plans
        }
        self._cases_by_center = {
            center: tuple(
                row.case_id for row in plan_seal.outer_plans if row.target_center == center
            )
            for center in CENTERS
        }
        self._outer_support_opened: set[tuple[str, str]] = set()
        self._prior_opened: set[tuple[str | None, str, str]] = set()
        self._candidate_seal: str | None = None
        self._pre_evaluation_seal: str | None = None
        self._pseudo_evaluation_opened: set[tuple[str, str, str]] = set()
        self._calibration_seal: str | None = None
        self._decision_seals: dict[tuple[str, str, str], str] = {}
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
        *,
        outer: str | None,
        target: str | None,
        case: str | None,
        excluded_centers: tuple[str, ...],
        excluded_cases: tuple[str, ...],
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
        if len(labels) != len(allowed) or {row.key for row in labels} != set(allowed):
            raise ProtocolError("CBPUPR label loader escaped its exact capability.")
        ordered = tuple(sorted(labels, key=lambda row: row.key))
        self._events.append(
            LabelAccessEvent(
                role,
                outer,
                target,
                case,
                excluded_centers,
                excluded_cases,
                len(ordered),
                len({(row.center, row.case_id) for row in ordered}),
                canonical_hash([list(row.key) for row in ordered]),
            )
        )
        return ordered

    def open_source_prior_labels(
        self,
        target_center: object,
        candidate_source: object,
        *,
        outer_excluded_center: object | None = None,
    ) -> tuple[BinaryLabel, ...]:
        target, source = str(target_center), str(candidate_source)
        outer = None if outer_excluded_center is None else str(outer_excluded_center)
        excluded = {target, source}
        if outer is not None:
            excluded.add(outer)
        key = outer, target, source
        if (
            target not in CENTERS
            or source not in CENTERS
            or source == target
            or (outer is not None and (outer not in CENTERS or outer == target))
            or (outer is not None and source == outer)
            or key in self._prior_opened
            or self._candidate_seal is not None
        ):
            raise ProtocolError("CBPUPR source-prior capability opened out of order.")
        allowed = frozenset(
            sample
            for (center, _case), samples in self._samples.items()
            if center not in excluded
            for sample in samples
        )
        role = (
            f"source_prior::target={target}::source={source}"
            if outer is None
            else f"source_prior::outer_H={outer}::J={target}::source={source}"
        )
        labels = self._decode(
            role,
            allowed,
            outer=outer,
            target=target,
            case=None,
            excluded_centers=tuple(sorted(excluded)),
            excluded_cases=(),
        )
        self._prior_opened.add(key)
        return labels

    def open_outer_support_labels(
        self, target_center: object, held_case_id: object
    ) -> tuple[BinaryLabel, ...]:
        center, held = str(target_center), str(held_case_id)
        key = center, held
        if (
            key not in self._samples
            or key in self._outer_support_opened
            or self._candidate_seal is not None
        ):
            raise ProtocolError("CBPUPR outer-support capability opened out of order.")
        allowed = frozenset(
            sample
            for (observed, case), samples in self._samples.items()
            if observed == center and case != held
            for sample in samples
        )
        role = f"outer_support::H={center}::excluded_c={held}"
        labels = self._decode(
            role,
            allowed,
            outer=center,
            target=center,
            case=held,
            excluded_centers=(),
            excluded_cases=(held,),
        )
        self._outer_support_opened.add(key)
        return labels

    def open_pseudo_support_labels(
        self, outer_target_center: object, pseudo_target_center: object, held_case_id: object
    ) -> tuple[BinaryLabel, ...]:
        del outer_target_center, pseudo_target_center, held_case_id
        raise ProtocolError(
            "CBPUPR forbids pseudo-support reopen/refit; use the sealed J-d posterior reference."
        )

    def seal_candidates(
        self,
        target_candidate_hashes: Mapping[tuple[str, str, str], str],
        pseudo_candidate_hashes: Mapping[tuple[str, str, str, str], str],
    ) -> str:
        target = dict(target_candidate_hashes)
        pseudo = dict(pseudo_candidate_hashes)
        expected_target = {
            (center, case, control)
            for center in CENTERS
            for case in self._cases_by_center[center]
            for control in ("IDENTITY", "WITHIN_CASE_CYCLIC_SHIFT")
        }
        expected_pseudo = {
            (outer, donor, case, control)
            for outer in CENTERS
            for donor in CENTERS
            if donor != outer
            for case in self._cases_by_center[donor]
            for control in ("IDENTITY", "WITHIN_CASE_CYCLIC_SHIFT")
        }
        expected_support_grants = set(self._samples)
        expected_prior_grants = {
            (None, target_center, source)
            for target_center in CENTERS
            for source in CENTERS
            if source != target_center
        } | {
            (outer, pseudo, source)
            for outer in CENTERS
            for pseudo in CENTERS
            if pseudo != outer
            for source in CENTERS
            if source not in {outer, pseudo}
        }
        if (
            self._candidate_seal is not None
            or self._outer_support_opened != expected_support_grants
            or len(self._outer_support_opened) != EXPECTED_TOTAL_CASE_COUNT
            or self._prior_opened != expected_prior_grants
            or len(self._prior_opened) != 576
            or set(target) != expected_target
            or set(pseudo) != expected_pseudo
            or len(target) != 2 * EXPECTED_TOTAL_CASE_COUNT
            or len(pseudo) != 2 * EXPECTED_PSEUDO_ROUTE_COUNT
            or any(require_sha256(value, "candidate_hash") != value for value in (*target.values(), *pseudo.values()))
        ):
            raise ProtocolError("CBPUPR candidate seal topology drifted.")
        self._candidate_seal = canonical_hash(
            {
                "schema_version": "fixed_bank_cbpupr_candidate_seal_v1",
                "plan_seal_hash": self._seal.seal_hash,
                "target": [[*key, target[key]] for key in sorted(target)],
                "pseudo": [[*key, pseudo[key]] for key in sorted(pseudo)],
                "terminal_labels_used": False,
            }
        )
        return self._candidate_seal

    def seal_pre_evaluation(self, transport_lineage_hash: object) -> str:
        if self._candidate_seal is None or self._pre_evaluation_seal is not None:
            raise ProtocolError("CBPUPR pre-evaluation seal opened out of order.")
        self._pre_evaluation_seal = canonical_hash(
            [self._candidate_seal, require_sha256(transport_lineage_hash, "transport_lineage_hash")]
        )
        return self._pre_evaluation_seal

    def open_pseudo_evaluation_labels(
        self, outer_target_center: object, pseudo_target_center: object, case_id: object
    ) -> tuple[BinaryLabel, ...]:
        outer, pseudo, case = str(outer_target_center), str(pseudo_target_center), str(case_id)
        key = outer, pseudo, case
        if (
            self._pre_evaluation_seal is None
            or key in self._pseudo_evaluation_opened
            or outer == pseudo
            or (pseudo, case) not in self._samples
            or self._calibration_seal is not None
        ):
            raise ProtocolError("CBPUPR pseudo-evaluation capability opened out of order.")
        role = f"PSEUDO_EVALUATION::H={outer}::J={pseudo}::excluded_d={case}"
        labels = self._decode(
            role,
            frozenset(self._samples[(pseudo, case)]),
            outer=outer,
            target=pseudo,
            case=case,
            excluded_centers=(outer,),
            excluded_cases=(),
        )
        self._pseudo_evaluation_opened.add(key)
        return labels

    def seal_replays_and_calibrations(
        self,
        replay_hash: object,
        utility_calibration_hash: object,
        policy_replay_diagnostic_hash: object,
    ) -> str:
        if (
            len(self._pseudo_evaluation_opened) != EXPECTED_PSEUDO_ROUTE_COUNT
            or self._calibration_seal is not None
        ):
            raise ProtocolError("CBPUPR calibration seal opened before all pseudo routes.")
        payload = [
            self._pre_evaluation_seal,
            require_sha256(replay_hash, "replay_hash"),
            require_sha256(utility_calibration_hash, "utility_calibration_hash"),
            require_sha256(
                policy_replay_diagnostic_hash, "policy_replay_diagnostic_hash"
            ),
            {"policy_replay_bias_used": False},
        ]
        self._calibration_seal = canonical_hash(payload)
        return self._calibration_seal

    def seal_target_decision(
        self, method_id: object, target_center: object, case_id: object, decision_hash: object
    ) -> None:
        key = str(method_id), str(target_center), str(case_id)
        if (
            self._calibration_seal is None
            or key[0] not in COMPOSED_POLICY_IDS
            or (key[1], key[2]) not in self._samples
            or key in self._decision_seals
        ):
            raise ProtocolError("CBPUPR target decision seal drifted.")
        self._decision_seals[key] = require_sha256(decision_hash, "decision_hash")

    def seal_aggregate(self, aggregate_hash: object) -> str:
        expected = {
            (method, center, case)
            for method in COMPOSED_POLICY_IDS
            for center in CENTERS
            for case in self._cases_by_center[center]
        }
        if set(self._decision_seals) != expected or self._aggregate_seal is not None:
            raise ProtocolError("CBPUPR aggregate seal lacks all decisions.")
        self._aggregate_seal = canonical_hash(
            [
                self._calibration_seal,
                [[*key, self._decision_seals[key]] for key in sorted(expected)],
                require_sha256(aggregate_hash, "aggregate_prediction_hash"),
            ]
        )
        return self._aggregate_seal

    def open_target_terminal_labels(self) -> tuple[BinaryLabel, ...]:
        if self._aggregate_seal is None or self._terminal_opened:
            raise ProtocolError("CBPUPR terminal labels opened before aggregate seal.")
        allowed = frozenset(sample for samples in self._samples.values() for sample in samples)
        labels = self._decode(
            "target_terminal_after_aggregate_seal",
            allowed,
            outer=None,
            target=None,
            case=None,
            excluded_centers=(),
            excluded_cases=(),
        )
        self._terminal_opened = True
        return labels

    def audit_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_cbpupr_label_access_audit_v1",
            "plan_seal_hash": self._seal.seal_hash,
            "event_count": len(self._events),
            "events": [row.to_payload() for row in self._events],
            "target_candidate_seal_complete": self._candidate_seal is not None,
            "pre_evaluation_seal_complete": self._pre_evaluation_seal is not None,
            "pseudo_evaluation_route_count": len(self._pseudo_evaluation_opened),
            "calibration_seal_complete": self._calibration_seal is not None,
            "decision_count": len(self._decision_seals),
            "aggregate_seal_complete": self._aggregate_seal is not None,
            "terminal_opened": self._terminal_opened,
            "raw_labels_persisted": False,
            "audit_hash": canonical_hash([row.to_payload() for row in self._events]),
        }


__all__ = ("CBPUPRLabelFirewall", "LabelAccessEvent")
