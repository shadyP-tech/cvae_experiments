"""Deterministic whole-group target-local H\\c support folds."""

from __future__ import annotations

from dataclasses import dataclass, field

from .hashing import canonical_hash, require_sha256
from .identity import SUPPORT_FOLD_COUNT
from .protocol import ProtocolError
from .route_identity import RouteScopeWitness


@dataclass(frozen=True, slots=True, order=True)
class SupportMember:
    """One case-level support unit; labels are intentionally not a field."""

    member_id: str
    center_id: str
    case_id: str
    group_id: str
    patient_id: str
    slide_id: str
    sample_key_hash: str
    row_count: int
    member_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        member_id = str(self.member_id)
        center_id = str(self.center_id)
        case_id = str(self.case_id)
        group_id = str(self.group_id)
        patient_id = str(self.patient_id)
        slide_id = str(self.slide_id)
        sample_key_hash = require_sha256(
            self.sample_key_hash,
            "support-member sample-key hash",
        )
        row_count = int(self.row_count)
        if (
            not member_id
            or not center_id
            or not case_id
            or not group_id
            or not patient_id
            or not slide_id
            or row_count <= 0
        ):
            raise ProtocolError("SCALE-BP support-member identity drifted.")
        payload = {
            "schema_version": "scale_bp_support_member_v2",
            "member_id": member_id,
            "center_id": center_id,
            "case_id": case_id,
            "group_id": group_id,
            "patient_id": patient_id,
            "slide_id": slide_id,
            "sample_key_hash": sample_key_hash,
            "row_count": row_count,
            "labels_present": False,
        }
        object.__setattr__(self, "member_id", member_id)
        object.__setattr__(self, "center_id", center_id)
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "group_id", group_id)
        object.__setattr__(self, "patient_id", patient_id)
        object.__setattr__(self, "slide_id", slide_id)
        object.__setattr__(self, "sample_key_hash", sample_key_hash)
        object.__setattr__(self, "row_count", row_count)
        object.__setattr__(self, "member_hash", canonical_hash(payload))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "scale_bp_support_member_v2",
            "member_id": self.member_id,
            "center_id": self.center_id,
            "case_id": self.case_id,
            "group_id": self.group_id,
            "patient_id": self.patient_id,
            "slide_id": self.slide_id,
            "sample_key_hash": self.sample_key_hash,
            "row_count": self.row_count,
            "labels_present": False,
            "member_hash": self.member_hash,
        }


@dataclass(frozen=True, slots=True)
class SupportFold:
    fold_index: int
    member_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    patient_ids: tuple[str, ...]
    slide_ids: tuple[str, ...]
    row_count: int
    fold_hash: str = field(init=False)

    def __post_init__(self) -> None:
        fold_index = int(self.fold_index)
        member_ids = tuple(str(value) for value in self.member_ids)
        group_ids = tuple(str(value) for value in self.group_ids)
        patient_ids = tuple(str(value) for value in self.patient_ids)
        slide_ids = tuple(str(value) for value in self.slide_ids)
        row_count = int(self.row_count)
        if (
            fold_index < 0
            or not member_ids
            or member_ids != tuple(sorted(set(member_ids)))
            or not group_ids
            or group_ids != tuple(sorted(set(group_ids)))
            or not patient_ids
            or patient_ids != tuple(sorted(set(patient_ids)))
            or not slide_ids
            or slide_ids != tuple(sorted(set(slide_ids)))
            or row_count <= 0
        ):
            raise ProtocolError("SCALE-BP support-fold contract drifted.")
        payload = {
            "schema_version": "scale_bp_support_fold_v1",
            "fold_index": fold_index,
            "member_ids": member_ids,
            "group_ids": group_ids,
            "patient_ids": patient_ids,
            "slide_ids": slide_ids,
            "row_count": row_count,
        }
        object.__setattr__(self, "fold_index", fold_index)
        object.__setattr__(self, "member_ids", member_ids)
        object.__setattr__(self, "group_ids", group_ids)
        object.__setattr__(self, "patient_ids", patient_ids)
        object.__setattr__(self, "slide_ids", slide_ids)
        object.__setattr__(self, "row_count", row_count)
        object.__setattr__(self, "fold_hash", canonical_hash(payload))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "scale_bp_support_fold_v1",
            "fold_index": self.fold_index,
            "member_ids": self.member_ids,
            "group_ids": self.group_ids,
            "patient_ids": self.patient_ids,
            "slide_ids": self.slide_ids,
            "row_count": self.row_count,
            "fold_hash": self.fold_hash,
        }


@dataclass(frozen=True, slots=True)
class SupportFoldPlan:
    held_center: str
    held_case_id: str
    held_group_id: str
    held_patient_id: str
    held_slide_id: str
    route_scope_hash: str
    route_witness: RouteScopeWitness
    fold_count: int
    members: tuple[SupportMember, ...]
    folds: tuple[SupportFold, ...]
    plan_hash: str = field(init=False)

    def __post_init__(self) -> None:
        held_center = str(self.held_center)
        held_case_id = str(self.held_case_id)
        held_group_id = str(self.held_group_id)
        held_patient_id = str(self.held_patient_id)
        held_slide_id = str(self.held_slide_id)
        route_scope_hash = require_sha256(
            self.route_scope_hash, "route-scope hash"
        )
        fold_count = int(self.fold_count)
        members = tuple(self.members)
        member_bindings = tuple((row.member_id, row.member_hash) for row in members)
        folds = tuple(self.folds)
        witness = self.route_witness
        if (
            not held_center
            or not held_case_id
            or not held_group_id
            or not held_patient_id
            or not held_slide_id
            or not isinstance(witness, RouteScopeWitness)
            or held_center != witness.target_center
            or held_case_id != witness.held_case_id
            or held_group_id != witness.held_group_id
            or held_patient_id != witness.held_patient_id
            or held_slide_id != witness.held_slide_id
            or route_scope_hash != witness.witness_hash
            or fold_count != SUPPORT_FOLD_COUNT
            or len(folds) != fold_count
            or tuple(fold.fold_index for fold in folds) != tuple(range(fold_count))
            or not members
            or any(not isinstance(row, SupportMember) for row in members)
            or members != tuple(sorted(members, key=lambda row: row.member_id))
            or len({member_id for member_id, _ in member_bindings})
            != len(member_bindings)
            or len({member_hash for _, member_hash in member_bindings})
            != len(member_bindings)
        ):
            raise ProtocolError("SCALE-BP support-fold plan drifted.")
        expected_bindings = {row.case_id: row for row in witness.support_bindings}
        if (
            {row.case_id for row in members} != set(expected_bindings)
            or len({row.case_id for row in members}) != len(members)
            or any(
                row.center_id != binding.center
                or row.group_id != binding.group_id
                or row.patient_id != binding.patient_id
                or row.slide_id != binding.slide_id
                or row.row_count != binding.row_count
                or row.sample_key_hash != binding.sample_key_hash
                for row in members
                for binding in (expected_bindings[row.case_id],)
            )
        ):
            raise ProtocolError(
                "SCALE-BP support members drifted from the route identity witness."
            )
        if any(
            row.center_id != held_center
            or row.case_id == held_case_id
            or row.group_id == held_group_id
            or row.patient_id == held_patient_id
            or row.slide_id == held_slide_id
            for row in members
        ):
            raise ProtocolError("SCALE-BP held or cross-center identity entered the plan.")
        member_ids = tuple(member_id for fold in folds for member_id in fold.member_ids)
        group_ids = tuple(group_id for fold in folds for group_id in fold.group_ids)
        patient_ids = tuple(patient_id for fold in folds for patient_id in fold.patient_ids)
        slide_ids = tuple(slide_id for fold in folds for slide_id in fold.slide_ids)
        if (
            len(member_ids) != len(set(member_ids))
            or len(group_ids) != len(set(group_ids))
            or len(patient_ids) != len(set(patient_ids))
            or len(slide_ids) != len(set(slide_ids))
            or tuple(sorted(member_ids))
            != tuple(member_id for member_id, _ in member_bindings)
        ):
            raise ProtocolError("SCALE-BP support identity crosses fold boundaries.")
        payload = {
            "schema_version": "scale_bp_support_fold_plan_v2",
            "held_center": held_center,
            "held_case_id": held_case_id,
            "held_group_id": held_group_id,
            "held_patient_id": held_patient_id,
            "held_slide_id": held_slide_id,
            "route_scope_hash": route_scope_hash,
            "route_identity_inventory_hash": witness.identity_inventory.inventory_hash,
            "evaluation_sample_key_hash": witness.evaluation_binding.sample_key_hash,
            "support_sample_key_hash": witness.support_sample_key_hash,
            "fold_count": fold_count,
            "member_bindings": member_bindings,
            "fold_hashes": tuple(fold.fold_hash for fold in folds),
            "held_case_excluded": True,
            "held_group_excluded": True,
            "held_patient_excluded": True,
            "held_slide_excluded": True,
            "labels_used_for_assignment": False,
        }
        object.__setattr__(self, "held_center", held_center)
        object.__setattr__(self, "held_case_id", held_case_id)
        object.__setattr__(self, "held_group_id", held_group_id)
        object.__setattr__(self, "held_patient_id", held_patient_id)
        object.__setattr__(self, "held_slide_id", held_slide_id)
        object.__setattr__(self, "route_scope_hash", route_scope_hash)
        object.__setattr__(self, "fold_count", fold_count)
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "folds", folds)
        object.__setattr__(self, "plan_hash", canonical_hash(payload))

    @property
    def member_ids(self) -> tuple[str, ...]:
        return tuple(member_id for member_id, _ in self.member_bindings)

    @property
    def member_bindings(self) -> tuple[tuple[str, str], ...]:
        return tuple((row.member_id, row.member_hash) for row in self.members)

    @property
    def member_hashes(self) -> tuple[str, ...]:
        return tuple(member_hash for _, member_hash in self.member_bindings)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "scale_bp_support_fold_plan_v2",
            "held_center": self.held_center,
            "held_case_id": self.held_case_id,
            "held_group_id": self.held_group_id,
            "held_patient_id": self.held_patient_id,
            "held_slide_id": self.held_slide_id,
            "route_scope_hash": self.route_scope_hash,
            "route_identity_inventory_hash": (
                self.route_witness.identity_inventory.inventory_hash
            ),
            "evaluation_sample_key_hash": (
                self.route_witness.evaluation_binding.sample_key_hash
            ),
            "support_sample_key_hash": self.route_witness.support_sample_key_hash,
            "fold_count": self.fold_count,
            "members": tuple(row.to_payload() for row in self.members),
            "folds": tuple(fold.to_payload() for fold in self.folds),
            "held_case_excluded": True,
            "held_group_excluded": True,
            "held_patient_excluded": True,
            "held_slide_excluded": True,
            "labels_used_for_assignment": False,
            "plan_hash": self.plan_hash,
        }


def build_support_fold_plan(
    members: object,
    *,
    route_witness: RouteScopeWitness,
    fold_count: int = SUPPORT_FOLD_COUNT,
) -> SupportFoldPlan:
    """Assign whole groups using a deterministic row-balanced greedy schedule.

    The caller must provide an already resolved H\\c support population.  Any
    held-case, held-group, or cross-center poison is rejected rather than
    silently filtered, leaving an auditable boundary at the caller.
    """

    rows = tuple(members)  # type: ignore[arg-type]
    if not isinstance(route_witness, RouteScopeWitness):
        raise ProtocolError("SCALE-BP route witness is absent from support folds.")
    center = route_witness.target_center
    held_case = route_witness.held_case_id
    held_group = route_witness.held_group_id
    held_patient = route_witness.held_patient_id
    held_slide = route_witness.held_slide_id
    scope_hash = route_witness.witness_hash
    q = int(fold_count)
    if q != SUPPORT_FOLD_COUNT:
        raise ProtocolError("SCALE-BP support fold count is not frozen.")
    if not rows or any(not isinstance(row, SupportMember) for row in rows):
        raise ProtocolError("SCALE-BP support population is empty or malformed.")
    if len({row.member_id for row in rows}) != len(rows):
        raise ProtocolError("SCALE-BP support member identity is duplicated.")
    for row in rows:
        if row.center_id != center:
            raise ProtocolError("SCALE-BP support contains a cross-center member.")
        if row.case_id == held_case:
            raise ProtocolError("SCALE-BP held case entered target-local support.")
        if row.group_id == held_group:
            raise ProtocolError("SCALE-BP held group entered target-local support.")
        if row.patient_id == held_patient:
            raise ProtocolError("SCALE-BP held patient entered target-local support.")
        if row.slide_id == held_slide:
            raise ProtocolError("SCALE-BP held slide entered target-local support.")

    # Connected components prevent a patient or slide spanning multiple case
    # or group identifiers from being split across OOF folds.
    parents = list(range(len(rows)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    seen_identity: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        for identity in (
            ("group", row.group_id),
            ("patient", row.patient_id),
            ("slide", row.slide_id),
        ):
            if identity in seen_identity:
                union(index, seen_identity[identity])
            else:
                seen_identity[identity] = index
    grouped: dict[int, list[SupportMember]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(find(index), []).append(row)
    if len(grouped) < q:
        raise ProtocolError("SCALE-BP support has fewer whole groups than folds.")

    group_rows = []
    for group_members in grouped.values():
        ordered = tuple(sorted(group_members, key=lambda item: item.member_id))
        row_count = sum(member.row_count for member in ordered)
        component_id = canonical_hash(
            {
                "schema_version": "scale_bp_support_component_v1",
                "member_hashes": tuple(member.member_hash for member in ordered),
            }
        )
        tie_hash = canonical_hash(
            {
                "schema_version": "scale_bp_support_group_order_v1",
                "held_center": center,
                "held_case_id": held_case,
                "held_group_id": held_group,
                "held_patient_id": held_patient,
                "held_slide_id": held_slide,
                "route_scope_hash": scope_hash,
                "component_id": component_id,
                "member_hashes": tuple(member.member_hash for member in ordered),
            }
        )
        group_rows.append((component_id, ordered, row_count, tie_hash))
    group_rows.sort(key=lambda item: (-item[2], item[3], item[0]))

    fold_groups: list[list[tuple[str, tuple[SupportMember, ...], int, str]]] = [
        [] for _ in range(q)
    ]
    fold_loads = [0 for _ in range(q)]
    for group in group_rows:
        fold_index = min(
            range(q), key=lambda index: (fold_loads[index], len(fold_groups[index]), index)
        )
        fold_groups[fold_index].append(group)
        fold_loads[fold_index] += group[2]

    folds = []
    for fold_index, assigned in enumerate(fold_groups):
        member_ids = tuple(
            sorted(member.member_id for group in assigned for member in group[1])
        )
        group_ids = tuple(
            sorted({member.group_id for group in assigned for member in group[1]})
        )
        patient_ids = tuple(
            sorted({member.patient_id for group in assigned for member in group[1]})
        )
        slide_ids = tuple(
            sorted({member.slide_id for group in assigned for member in group[1]})
        )
        folds.append(
            SupportFold(
                fold_index,
                member_ids,
                group_ids,
                patient_ids,
                slide_ids,
                fold_loads[fold_index],
            )
        )
    return SupportFoldPlan(
        held_center=center,
        held_case_id=held_case,
        held_group_id=held_group,
        held_patient_id=held_patient,
        held_slide_id=held_slide,
        route_scope_hash=scope_hash,
        route_witness=route_witness,
        fold_count=q,
        members=tuple(sorted(rows, key=lambda row: row.member_id)),
        folds=tuple(folds),
    )


def fold_index_for_member(plan: SupportFoldPlan, member_id: str) -> int:
    identity = str(member_id)
    matches = tuple(fold.fold_index for fold in plan.folds if identity in fold.member_ids)
    if len(matches) != 1:
        raise ProtocolError("SCALE-BP support member does not resolve to exactly one fold.")
    return matches[0]


__all__ = (
    "SupportFold",
    "SupportFoldPlan",
    "SupportMember",
    "build_support_fold_plan",
    "fold_index_for_member",
)
