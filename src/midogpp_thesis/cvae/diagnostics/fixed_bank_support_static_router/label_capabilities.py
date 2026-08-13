"""Fail-closed, route-scoped access to consumed MIDOG++ labels."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterator, Mapping, Sequence

from ....data.contract.stage70_target_evaluation.contracts import evaluation_row_id
from ...protocol import ProtocolError
from ...runtime.artifact_io import sha256_file
from .constants import (
    CENTERS,
    OOF_FOLD_COUNT,
    a1_action_id,
    candidate_sources,
    source_from_action,
)
from .experiment_contracts import EXPECTED_MANIFEST_SHA256
from .hashing import canonical_hash, require_sha256, require_stable_hash
from .input_contracts import LabelFreeTestFrame, TestRowIdentity
from .partitions import CaseFold, CaseOOFPartition
from .products import (
    BinaryLabelRow,
    DecisionSeal,
    NullSelectionPlan,
    RouteDecision,
    StaticSelection,
)


@dataclass(frozen=True)
class FoldPlan:
    target_center: str
    fold_ordinal: int
    support_case_ids: tuple[str, ...]
    evaluation_case_ids: tuple[str, ...]
    fold_hash: str
    partition_hash: str
    probability_seal_hash: str
    plan_hash: str

    @classmethod
    def from_fold(
        cls,
        fold: CaseFold,
        *,
        partition_hash: str,
        probability_seal_hash: str,
    ) -> "FoldPlan":
        require_sha256(partition_hash, "partition_hash")
        require_stable_hash(probability_seal_hash, "probability_seal_hash")
        payload = {
            "schema_version": "fixed_bank_support_static_router_fold_plan_v1",
            "target_center": fold.target_center,
            "fold_ordinal": fold.fold_ordinal,
            "support_case_ids": list(fold.support_case_ids),
            "evaluation_case_ids": list(fold.evaluation_case_ids),
            "fold_hash": fold.fold_hash,
            "partition_hash": partition_hash,
            "probability_seal_hash": probability_seal_hash,
            "support_is_other_four_folds": True,
            "held_evaluation_labels_in_plan": False,
        }
        return cls(
            fold.target_center,
            fold.fold_ordinal,
            fold.support_case_ids,
            fold.evaluation_case_ids,
            fold.fold_hash,
            partition_hash,
            probability_seal_hash,
            canonical_hash(payload),
        )

    @property
    def route_key(self) -> tuple[str, int]:
        return self.target_center, self.fold_ordinal

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_support_static_router_fold_plan_v1",
            "target_center": self.target_center,
            "fold_ordinal": self.fold_ordinal,
            "support_case_ids": list(self.support_case_ids),
            "evaluation_case_ids": list(self.evaluation_case_ids),
            "fold_hash": self.fold_hash,
            "partition_hash": self.partition_hash,
            "probability_seal_hash": self.probability_seal_hash,
            "support_is_other_four_folds": True,
            "held_evaluation_labels_in_plan": False,
            "plan_hash": self.plan_hash,
        }


@dataclass(frozen=True)
class ScopedLabelGrant:
    role: str
    target_center: str | None
    fold_ordinal: int | None
    candidate_source: str | None
    labels: tuple[BinaryLabelRow, ...]
    row_identity_hash: str
    grant_hash: str

    def __post_init__(self) -> None:
        labels = tuple(self.labels)
        if not labels or len({row.sample_key for row in labels}) != len(labels):
            raise ProtocolError("Label grant must contain unique non-empty rows.")
        require_sha256(self.row_identity_hash, "row_identity_hash")
        require_sha256(self.grant_hash, "grant_hash")
        object.__setattr__(self, "labels", labels)

    def __iter__(self) -> Iterator[BinaryLabelRow]:
        return iter(self.labels)

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def case_keys(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted({row.case_key for row in self.labels}))

    def to_payload(self) -> dict[str, object]:
        """Persist capability evidence, never the raw label values."""

        return {
            "schema_version": "fixed_bank_support_static_router_scoped_label_grant_v1",
            "role": self.role,
            "target_center": self.target_center,
            "fold_ordinal": self.fold_ordinal,
            "candidate_source": self.candidate_source,
            "row_count": len(self.labels),
            "case_count": len(self.case_keys),
            "row_identity_hash": self.row_identity_hash,
            "grant_hash": self.grant_hash,
            "raw_labels_persisted": False,
        }


@dataclass(frozen=True)
class LabelAccessEvent:
    role: str
    target_center: str | None
    fold_ordinal: int | None
    candidate_source: str | None
    row_count: int
    case_count: int
    row_identity_hash: str
    grant_hash: str

    def to_payload(self) -> dict[str, object]:
        return {**self.__dict__, "raw_labels_persisted": False}


class LabelCapabilityManager:
    """The only manifest reader for G_static, S4, and route evaluation labels."""

    def __init__(
        self,
        manifest_path: Path,
        frame: LabelFreeTestFrame,
        partition: CaseOOFPartition,
        *,
        probability_seal_hash: str,
    ) -> None:
        require_stable_hash(probability_seal_hash, "probability_seal_hash")
        manifest_path = Path(manifest_path)
        if sha256_file(manifest_path) != EXPECTED_MANIFEST_SHA256:
            raise ProtocolError("S4 label manifest hash drifted.")
        frame_keys = {
            (row.center, row.case_id, row.evaluation_row_id) for row in frame.rows
        }
        partition_keys = {row.sample_key for row in partition.identities}
        if frame_keys != partition_keys:
            raise ProtocolError("S4 partition differs from the sealed label-free frame.")
        self._manifest_path = manifest_path
        self._manifest_sha256 = EXPECTED_MANIFEST_SHA256
        self._frame = frame
        self._partition = partition
        self._probability_seal_hash = probability_seal_hash
        self._plans: Mapping[tuple[str, int], FoldPlan] | None = None
        self._g_donor_grants: dict[tuple[str, str], ScopedLabelGrant] = {}
        self._g_selection_seals: dict[str, str] = {}
        self._support_grants: dict[tuple[str, int], ScopedLabelGrant] = {}
        self._route_decision_seals: dict[tuple[str, int], str] = {}
        self._null_selection_seals: dict[tuple[str, int], str] = {}
        self._aggregate_decision_seal_hash: str | None = None
        self._aggregate_null_plan_seal_hash: str | None = None
        self._evaluation_grants: dict[tuple[str, int], ScopedLabelGrant] = {}
        self._events: list[LabelAccessEvent] = []

    def seal_all_fold_plans(self) -> tuple[FoldPlan, ...]:
        if self._plans is not None or self._events:
            raise ProtocolError("Fold plans must be sealed before any label capability opens.")
        plans = tuple(
            FoldPlan.from_fold(
                fold,
                partition_hash=self._partition.partition_hash,
                probability_seal_hash=self._probability_seal_hash,
            )
            for fold in self._partition.folds
        )
        expected = len(CENTERS) * OOF_FOLD_COUNT
        if len(plans) != expected:
            raise ProtocolError("Fold-plan coverage drifted from 45 routes.")
        self._plans = MappingProxyType({row.route_key: row for row in plans})
        return plans

    def open_g_static_donor_labels(
        self, heldout_target: object, candidate_source: object
    ) -> ScopedLabelGrant:
        self._require_plans()
        target = str(heldout_target)
        source = str(candidate_source)
        key = (target, source)
        if (
            target not in CENTERS
            or source not in candidate_sources(target)
            or key in self._g_donor_grants
            or target in self._g_selection_seals
            or any(route[0] == target for route in self._support_grants)
        ):
            raise ProtocolError("G_static donor capability opened out of order.")
        rows = tuple(
            row for row in self._frame.rows if row.center not in {target, source}
        )
        grant = self._open_rows(
            rows,
            role="g_static_q_notin_H_e",
            target=target,
            fold=None,
            candidate_source=source,
        )
        if {label.target_center for label in grant.labels} != set(CENTERS) - {target, source}:
            raise ProtocolError("G_static label grant violates q not in {H,e}.")
        self._g_donor_grants[key] = grant
        return grant

    def g_static_donor_grant_seal(self, heldout_target: object) -> str:
        target = str(heldout_target)
        expected = tuple((target, source) for source in candidate_sources(target))
        if any(key not in self._g_donor_grants for key in expected):
            raise ProtocolError("G_static requires all eight candidate-specific donor grants.")
        return canonical_hash(
            {
                "schema_version": "fixed_bank_support_static_router_g_donor_grant_seal_v1",
                "heldout_target": target,
                "candidate_grant_hashes": [
                    [source, self._g_donor_grants[(target, source)].grant_hash]
                    for source in candidate_sources(target)
                ],
                "q_excludes_H_and_e": True,
                "same_H_support_labels_used": False,
            }
        )

    def record_g_static_selection(self, selection: StaticSelection) -> None:
        target = selection.target_center
        expected_prerequisite = self.g_static_donor_grant_seal(target)
        if (
            selection.method_id != "G_static"
            or selection.prerequisite_seal_hash != expected_prerequisite
            or target in self._g_selection_seals
            or any(route[0] == target for route in self._support_grants)
        ):
            raise ProtocolError("G_static selection seal is misbound or out of order.")
        self._g_selection_seals[target] = selection.selection_hash

    def record_g_static_selection_seal(
        self, heldout_target: object, selection_hash: str
    ) -> None:
        """Hash-only adapter; typed ``record_g_static_selection`` is preferred."""

        target = str(heldout_target)
        require_sha256(selection_hash, "g_static_selection_hash")
        self.g_static_donor_grant_seal(target)
        if target in self._g_selection_seals or any(
            route[0] == target for route in self._support_grants
        ):
            raise ProtocolError("G_static selection seal was reused or recorded late.")
        self._g_selection_seals[target] = selection_hash

    def open_fold_support_labels(
        self, target_center: object, fold_ordinal: int
    ) -> ScopedLabelGrant:
        plan = self._plan(target_center, fold_ordinal)
        key = plan.route_key
        if (
            plan.target_center not in self._g_selection_seals
            or key in self._support_grants
            or key in self._route_decision_seals
            or key in self._evaluation_grants
        ):
            raise ProtocolError("Fold support labels opened out of order.")
        support = set(plan.support_case_ids)
        rows = tuple(
            row
            for row in self._frame.rows_by_center[plan.target_center]
            if row.case_id in support
        )
        if {row.case_id for row in rows} != support:
            raise ProtocolError("Fold support label request has incomplete case coverage.")
        grant = self._open_rows(
            rows,
            role="same_H_other_four_fold_support",
            target=plan.target_center,
            fold=plan.fold_ordinal,
            candidate_source=None,
        )
        if any(label.case_id in set(plan.evaluation_case_ids) for label in grant.labels):
            raise ProtocolError("Fold support capability intersected its own evaluation cases.")
        self._support_grants[key] = grant
        return grant

    def record_route_decision(self, decision: RouteDecision) -> None:
        plan = self._plan(decision.target_center, decision.fold_ordinal)
        key = plan.route_key
        if (
            key not in self._support_grants
            or key in self._route_decision_seals
            or decision.fold_hash != plan.fold_hash
            or decision.support_case_ids != plan.support_case_ids
            or decision.evaluation_case_ids != plan.evaluation_case_ids
            or decision.probability_seal_hash != self._probability_seal_hash
            or decision.g_static.selection_hash
            != self._g_selection_seals.get(plan.target_center)
            or decision.s4.prerequisite_seal_hash != self._support_grants[key].grant_hash
        ):
            raise ProtocolError("Route decision seal is misbound to its fold capabilities.")
        self._route_decision_seals[key] = decision.route_decision_hash

    def record_route_decision_seal(
        self, target_center: object, fold_ordinal: int, decision_hash: str
    ) -> None:
        """Hash-only adapter for persistence layers that already validated the DTO."""

        plan = self._plan(target_center, fold_ordinal)
        key = plan.route_key
        require_sha256(decision_hash, "route_decision_hash")
        if key not in self._support_grants or key in self._route_decision_seals:
            raise ProtocolError("Route decision seal lacks its scoped support capability.")
        self._route_decision_seals[key] = decision_hash

    def record_route_null_selection(self, plan: NullSelectionPlan) -> None:
        fold_plan = self._plan(plan.target_center, plan.fold_ordinal)
        key = fold_plan.route_key
        if (
            key not in self._support_grants
            or key not in self._route_decision_seals
            or key in self._null_selection_seals
            or plan.fold_hash != fold_plan.fold_hash
            or plan.prerequisite_seal_hash != self._support_grants[key].grant_hash
        ):
            raise ProtocolError("Null selection plan is misbound or out of order.")
        self._null_selection_seals[key] = plan.plan_hash

    def record_route_null_selection_seal(
        self, target_center: object, fold_ordinal: int, null_selection_hash: str
    ) -> None:
        fold_plan = self._plan(target_center, fold_ordinal)
        key = fold_plan.route_key
        require_sha256(null_selection_hash, "null_selection_hash")
        if (
            key not in self._support_grants
            or key not in self._route_decision_seals
            or key in self._null_selection_seals
        ):
            raise ProtocolError("Null selection seal lacks its observed route decision.")
        self._null_selection_seals[key] = null_selection_hash

    def record_pre_evaluation_aggregate_seals(
        self,
        decision_seal: DecisionSeal,
        null_plan_seal_payload: Mapping[str, object],
    ) -> None:
        """Bind the two durable 45-route seals before any evaluation grant.

        Per-route records alone are not a global barrier: an evaluation case from
        one route may legally appear in another route's support set.  This method
        therefore verifies the complete aggregate decision and null-plan surfaces
        that the runner has already persisted before unlocking any evaluation
        labels.
        """

        self._require_plans()
        route_keys = tuple(
            (center, fold)
            for center in CENTERS
            for fold in range(OOF_FOLD_COUNT)
        )
        if (
            self._aggregate_decision_seal_hash is not None
            or self._aggregate_null_plan_seal_hash is not None
            or self._evaluation_grants
            or tuple(self._route_decision_seals) != route_keys
            or tuple(self._null_selection_seals) != route_keys
        ):
            raise ProtocolError(
                "Pre-evaluation aggregate seals require all 45 ordered route and null records exactly once."
            )
        if (
            decision_seal.partition_hash != self._partition.partition_hash
            or decision_seal.probability_seal_hash != self._probability_seal_hash
            or tuple(row.route_key for row in decision_seal.decisions) != route_keys
            or {
                row.route_key: row.route_decision_hash
                for row in decision_seal.decisions
            }
            != self._route_decision_seals
        ):
            raise ProtocolError(
                "Aggregate decision seal is not bound to the recorded 45-route surface."
            )
        require_sha256(decision_seal.decision_seal_hash, "decision_seal_hash")

        payload = dict(null_plan_seal_payload)
        unhashed = {
            key: value
            for key, value in payload.items()
            if key != "null_selection_plan_seal_hash"
        }
        null_hash = payload.get("null_selection_plan_seal_hash")
        expected_plan_hashes = [self._null_selection_seals[key] for key in route_keys]
        if not isinstance(null_hash, str):
            raise ProtocolError("Aggregate null-plan seal hash is absent.")
        require_sha256(null_hash, "null_selection_plan_seal_hash")
        if (
            null_hash != canonical_hash(unhashed)
            or payload.get("decision_seal_hash")
            != decision_seal.decision_seal_hash
            or payload.get("partition_hash") != self._partition.partition_hash
            or payload.get("route_plan_hashes") != expected_plan_hashes
            or payload.get("route_plan_count") != len(route_keys)
            or payload.get("sealed_before_any_route_evaluation_labels") is not True
            or payload.get("evaluation_labels_used") is not False
        ):
            raise ProtocolError(
                "Aggregate null-plan seal is not bound to the recorded 45-route surface."
            )
        self._aggregate_decision_seal_hash = decision_seal.decision_seal_hash
        self._aggregate_null_plan_seal_hash = null_hash

    def open_route_evaluation_labels(
        self, target_center: object, fold_ordinal: int
    ) -> ScopedLabelGrant:
        plan = self._plan(target_center, fold_ordinal)
        key = plan.route_key
        if (
            key not in self._route_decision_seals
            or key not in self._null_selection_seals
            or self._aggregate_decision_seal_hash is None
            or self._aggregate_null_plan_seal_hash is None
            or key in self._evaluation_grants
        ):
            raise ProtocolError(
                "Route evaluation labels require durable aggregate observed and null selection seals."
            )
        evaluation = set(plan.evaluation_case_ids)
        rows = tuple(
            row
            for row in self._frame.rows_by_center[plan.target_center]
            if row.case_id in evaluation
        )
        if {row.case_id for row in rows} != evaluation:
            raise ProtocolError("Route evaluation label request has incomplete case coverage.")
        grant = self._open_rows(
            rows,
            role="route_evaluation_after_observed_and_null_seals",
            target=plan.target_center,
            fold=plan.fold_ordinal,
            candidate_source=None,
        )
        self._evaluation_grants[key] = grant
        return grant

    open_fold_evaluation_labels = open_route_evaluation_labels

    def access_report(self) -> Mapping[str, object]:
        expected_routes = len(CENTERS) * OOF_FOLD_COUNT
        complete = len(self._evaluation_grants) == expected_routes
        payload = {
            "schema_version": "fixed_bank_support_static_router_label_capability_report_v1",
            "status": "PASS" if complete else "INCOMPLETE",
            "probability_seal_hash": self._probability_seal_hash,
            "partition_hash": self._partition.partition_hash,
            "fold_plan_count": 0 if self._plans is None else len(self._plans),
            "g_static_candidate_donor_grant_count": len(self._g_donor_grants),
            "g_static_selection_seal_count": len(self._g_selection_seals),
            "support_grant_count": len(self._support_grants),
            "route_decision_seal_count": len(self._route_decision_seals),
            "null_selection_seal_count": len(self._null_selection_seals),
            "null_route_plan_seal_count": len(self._null_selection_seals),
            "route_evaluation_grant_count": len(self._evaluation_grants),
            "pre_evaluation_aggregate_decision_seal_count": int(
                self._aggregate_decision_seal_hash is not None
            ),
            "pre_evaluation_aggregate_null_plan_seal_count": int(
                self._aggregate_null_plan_seal_hash is not None
            ),
            "pre_evaluation_aggregate_decision_seal_hash": self._aggregate_decision_seal_hash,
            "pre_evaluation_aggregate_null_plan_seal_hash": self._aggregate_null_plan_seal_hash,
            "all_route_and_null_aggregate_seals_recorded_before_evaluation_labels": (
                not self._evaluation_grants
                or (
                    self._aggregate_decision_seal_hash is not None
                    and self._aggregate_null_plan_seal_hash is not None
                )
            ),
            "every_route_decision_sealed_before_own_evaluation_labels": all(
                key in self._route_decision_seals for key in self._evaluation_grants
            ),
            "every_route_decision_excludes_own_evaluation_labels": all(
                key in self._route_decision_seals for key in self._evaluation_grants
            ),
            "every_null_selection_sealed_before_own_evaluation_labels": all(
                key in self._null_selection_seals for key in self._evaluation_grants
            ),
            "each_null_route_plan_sealed_before_own_evaluation_labels": all(
                key in self._null_selection_seals for key in self._evaluation_grants
            ),
            "events": [row.to_payload() for row in self._events],
            "raw_labels_persisted": False,
            "per_case_bacc_persisted": False,
            "evaluation_labels_used_for_decisions": False,
            "source_expert_updated": False,
            "shared_router_trained": False,
        }
        return MappingProxyType({**payload, "report_hash": canonical_hash(payload)})

    report_payload = access_report

    def _require_plans(self) -> None:
        if self._plans is None:
            raise ProtocolError("Labels require all probability-bound fold plans first.")

    def _plan(self, target_center: object, fold_ordinal: int) -> FoldPlan:
        self._require_plans()
        key = (str(target_center), int(fold_ordinal))
        try:
            return self._plans[key]  # type: ignore[index]
        except KeyError as exc:
            raise ProtocolError("Route fold plan is absent.") from exc

    def _open_rows(
        self,
        rows: Sequence[TestRowIdentity],
        *,
        role: str,
        target: str | None,
        fold: int | None,
        candidate_source: str | None,
    ) -> ScopedLabelGrant:
        requested = {row.manifest_row_index: row for row in rows}
        labels: list[BinaryLabelRow] = []
        try:
            handle = self._manifest_path.open(newline="", encoding="utf-8")
        except OSError as exc:
            raise ProtocolError("Cannot open scoped S4 label manifest.") from exc
        with handle:
            reader = csv.DictReader(handle)
            required = {"case_id", "center", "split", "label"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ProtocolError("Scoped S4 manifest fields drifted.")
            for index, raw in enumerate(reader):
                wanted = requested.get(index)
                if wanted is None:
                    continue
                if (
                    evaluation_row_id(EXPECTED_MANIFEST_SHA256, index)
                    != wanted.evaluation_row_id
                    or str(raw["case_id"]) != wanted.case_id
                    or str(raw["center"]) != wanted.center
                    or str(raw["split"]) != wanted.split
                ):
                    raise ProtocolError("Scoped S4 manifest identity drifted.")
                try:
                    value = int(raw["label"])
                except (TypeError, ValueError) as exc:
                    raise ProtocolError("Scoped S4 manifest label is malformed.") from exc
                labels.append(
                    BinaryLabelRow(wanted.center, wanted.case_id, wanted.evaluation_row_id, value)
                )
        if sha256_file(self._manifest_path) != self._manifest_sha256:
            raise ProtocolError("S4 manifest changed while a label capability was open.")
        if len(labels) != len(requested) or {row.sample_id for row in labels} != {
            row.evaluation_row_id for row in rows
        }:
            raise ProtocolError("Scoped S4 label coverage drifted.")
        ordered = tuple(sorted(labels))
        identities = [list(row.sample_key) for row in ordered]
        row_hash = canonical_hash(
            {
                "schema_version": "fixed_bank_support_static_router_label_rows_v1",
                "identities": identities,
            }
        )
        grant_hash = canonical_hash(
            {
                "schema_version": "fixed_bank_support_static_router_label_grant_v1",
                "role": role,
                "target_center": target,
                "fold_ordinal": fold,
                "candidate_source": candidate_source,
                "labels": [
                    [row.target_center, row.case_id, row.sample_id, row.value]
                    for row in ordered
                ],
                "raw_labels_persisted": False,
            }
        )
        grant = ScopedLabelGrant(
            role,
            target,
            fold,
            candidate_source,
            ordered,
            row_hash,
            grant_hash,
        )
        self._events.append(
            LabelAccessEvent(
                role,
                target,
                fold,
                candidate_source,
                len(ordered),
                len(grant.case_keys),
                row_hash,
                grant_hash,
            )
        )
        return grant


# Explicit package-specific spelling for callers that prefer it.
SupportStaticLabelCapabilityManager = LabelCapabilityManager


__all__ = (
    "FoldPlan",
    "LabelAccessEvent",
    "LabelCapabilityManager",
    "ScopedLabelGrant",
    "SupportStaticLabelCapabilityManager",
)
