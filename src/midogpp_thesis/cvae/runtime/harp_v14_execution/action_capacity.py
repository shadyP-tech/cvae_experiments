"""Pre-scheduling capacity proof for every physical HARP v14 action.

HARP composes each class from an immutable prefix of every resident expert
stream.  The largest prefix is not determined by the final classifier budget:
the six-source cross-fit calibration menu assigns its complete residual tail
to one selected expert and therefore needs ``168 + 126 == 294`` rows from
that expert.  This module enumerates the physical action graph and proves that
the resident stream contract can satisfy every classwise window before GPU
generation or classifier work is scheduled.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from .crossfit_actions import (
    CROSSFIT_SURFACE,
    FoldConditionedActionSpec,
    build_fold_conditioned_action_menu,
)
from .physical_actions import HarpActionSpec, TARGET_SURFACE, build_target_action_menu
from .resident_stream_contracts import SOURCE_ROWS_PER_CLASS


ACTION_CAPACITY_CERTIFICATE_SCHEMA = (
    "midogpp_harp_v14_complete_action_capacity_certificate_v1"
)
ACTION_CAPACITY_ENUMERATION_SCHEMA = (
    "midogpp_harp_v14_action_capacity_enumeration_v1"
)
TARGET_MAX_REQUIRED_PER_CLASS = 256
SEVEN_SOURCE_MAX_REQUIRED_PER_CLASS = 270
SIX_SOURCE_MAX_REQUIRED_PER_CLASS = 294
GLOBAL_MAX_REQUIRED_PER_CLASS = SIX_SOURCE_MAX_REQUIRED_PER_CLASS


@dataclass(frozen=True, slots=True)
class ActionCapacityRequirement:
    """Maximum classwise prefix required by one immutable physical action."""

    surface_kind: str
    outer_target_id: str
    heldout_center_id: str | None
    current_query_center_id: str
    action_id: str
    selected_source_id: str | None
    source_count: int
    required_by_source: Mapping[str, int]
    maximum_required_per_class: int
    action_hash: str

    def __post_init__(self) -> None:
        normalized = {
            str(source): int(value) for source, value in self.required_by_source.items()
        }
        if (
            tuple(sorted(normalized)) != tuple(normalized)
            or len(normalized) != self.source_count
            or not normalized
            or any(value <= 0 for value in normalized.values())
            or max(normalized.values()) != self.maximum_required_per_class
        ):
            raise ProtocolError("HARP v14 action capacity requirement is malformed.")
        object.__setattr__(self, "required_by_source", MappingProxyType(normalized))

    def to_payload(self) -> dict[str, object]:
        return {
            "surface_kind": self.surface_kind,
            "outer_target_id": self.outer_target_id,
            "heldout_center_id": self.heldout_center_id,
            "current_query_center_id": self.current_query_center_id,
            "action_id": self.action_id,
            "selected_source_id": self.selected_source_id,
            "source_count": self.source_count,
            "required_by_source": dict(self.required_by_source),
            "maximum_required_per_class": self.maximum_required_per_class,
            "action_hash": self.action_hash,
        }


def requirement_for_action(
    action: HarpActionSpec | FoldConditionedActionSpec,
) -> ActionCapacityRequirement:
    """Derive the exact largest classwise source window for ``action``."""

    if not isinstance(action, (HarpActionSpec, FoldConditionedActionSpec)):
        raise ProtocolError("HARP v14 capacity audit requires a typed action.")
    geometry = action.geometry
    residual = action.residual_action
    if residual is None:
        required = {
            source: int(geometry.base_per_source) for source in geometry.source_order
        }
    else:
        required = {
            source: max(
                int(residual.windows_by_class[label][source].required_capacity)
                for label in geometry.class_labels
            )
            for source in geometry.source_order
        }
    heldout = (
        action.heldout_center_id
        if isinstance(action, FoldConditionedActionSpec)
        else None
    )
    query = (
        action.current_query_center_id
        if isinstance(action, FoldConditionedActionSpec)
        else action.query_center_id
    )
    surface = (
        CROSSFIT_SURFACE
        if isinstance(action, FoldConditionedActionSpec)
        else action.surface_kind
    )
    action_id = action.action_id
    if type(action_id) is not str or not action_id:
        raise ProtocolError("HARP v14 capacity audit action identity is absent.")
    return ActionCapacityRequirement(
        surface_kind=surface,
        outer_target_id=action.outer_target_id,
        heldout_center_id=heldout,
        current_query_center_id=query,
        action_id=action_id,
        selected_source_id=action.selected_source_id,
        source_count=len(action.source_order),
        required_by_source=required,
        maximum_required_per_class=max(required.values()),
        action_hash=action.action_hash,
    )


def validate_action_capacity(
    actions: Sequence[HarpActionSpec | FoldConditionedActionSpec],
    *,
    available_rows_per_class: int = SOURCE_ROWS_PER_CLASS,
) -> tuple[ActionCapacityRequirement, ...]:
    """Fail before scheduling if any action exceeds a resident stream prefix."""

    if (
        isinstance(available_rows_per_class, bool)
        or type(available_rows_per_class) is not int
        or available_rows_per_class <= 0
    ):
        raise ProtocolError("HARP v14 available source capacity is malformed.")
    rows = tuple(requirement_for_action(action) for action in actions)
    if not rows:
        raise ProtocolError("HARP v14 action capacity audit is empty.")
    failed = tuple(
        row
        for row in rows
        if row.maximum_required_per_class > available_rows_per_class
    )
    if failed:
        first = failed[0]
        raise ProtocolError(
            "HARP v14 action capacity certificate failed before scheduling: "
            f"{first.surface_kind} H={first.outer_target_id} "
            f"q={first.heldout_center_id} r={first.current_query_center_id} "
            f"action={first.action_id} requires "
            f"{first.maximum_required_per_class} rows/class but only "
            f"{available_rows_per_class} are available."
        )
    return rows


def enumerate_complete_action_capacity(
    outer_targets: Sequence[str] = CENTERS,
) -> tuple[ActionCapacityRequirement, ...]:
    """Enumerate every target and H/q/r source-crossfit action exactly once."""

    requested = _outer_targets(outer_targets)
    target = tuple(
        requirement_for_action(action)
        for outer in requested
        for action in build_target_action_menu(outer)
    )
    crossfit = tuple(
        requirement_for_action(action)
        for outer in requested
        for heldout in CENTERS
        if heldout != outer
        for query in CENTERS
        if query != outer
        for action in build_fold_conditioned_action_menu(outer, heldout, query)
    )
    rows = (*target, *crossfit)
    expected = len(requested) * (10 + 520)
    if len(rows) != expected or len({row.action_hash for row in rows}) != expected:
        raise ProtocolError("HARP v14 capacity action enumeration drifted.")
    return rows


def build_action_capacity_certificate(
    centers: Sequence[str] = CENTERS,
    *,
    stream_rows_per_class: int = SOURCE_ROWS_PER_CLASS,
) -> Mapping[str, object]:
    """Return a compact hash binding the complete feasible action enumeration."""

    requested = _outer_targets(centers)
    rows = enumerate_complete_action_capacity(requested)
    if (
        isinstance(stream_rows_per_class, bool)
        or type(stream_rows_per_class) is not int
        or stream_rows_per_class <= 0
    ):
        raise ProtocolError("HARP v14 available source capacity is malformed.")
    failed = tuple(
        row
        for row in rows
        if row.maximum_required_per_class > stream_rows_per_class
    )
    if failed:
        first = failed[0]
        raise ProtocolError(
            "HARP v14 complete capacity certificate failed before scheduling: "
            f"action {first.action_id} requires "
            f"{first.maximum_required_per_class} rows/class but only "
            f"{stream_rows_per_class} are available."
        )
    histogram = Counter(row.maximum_required_per_class for row in rows)
    target = tuple(row for row in rows if row.surface_kind == TARGET_SURFACE)
    prediction = tuple(
        row
        for row in rows
        if row.surface_kind == CROSSFIT_SURFACE
        and row.heldout_center_id == row.current_query_center_id
    )
    calibration = tuple(
        row
        for row in rows
        if row.surface_kind == CROSSFIT_SURFACE
        and row.heldout_center_id != row.current_query_center_id
    )
    maxima = {
        "target": max(row.maximum_required_per_class for row in target),
        "source_prediction_seven_source": max(
            row.maximum_required_per_class for row in prediction
        ),
        "source_calibration_six_source": max(
            row.maximum_required_per_class for row in calibration
        ),
    }
    expected_maxima = {
        "target": TARGET_MAX_REQUIRED_PER_CLASS,
        "source_prediction_seven_source": SEVEN_SOURCE_MAX_REQUIRED_PER_CLASS,
        "source_calibration_six_source": SIX_SOURCE_MAX_REQUIRED_PER_CLASS,
    }
    if (
        maxima != expected_maxima
        or max(maxima.values()) != GLOBAL_MAX_REQUIRED_PER_CLASS
    ):
        raise ProtocolError("HARP v14 capacity maxima drifted.")
    enumeration_payload = {
        "schema_version": ACTION_CAPACITY_ENUMERATION_SCHEMA,
        "requirements": [row.to_payload() for row in rows],
    }
    body = {
        "schema_version": ACTION_CAPACITY_CERTIFICATE_SCHEMA,
        "outer_target_ids": list(requested),
        "stream_rows_per_class": stream_rows_per_class,
        "required_rows_per_class_by_surface": maxima,
        "global_maximum_required_rows_per_class": GLOBAL_MAX_REQUIRED_PER_CLASS,
        "target_action_count": len(target),
        "source_prediction_action_count": len(prediction),
        "source_calibration_action_count": len(calibration),
        "enumerated_action_count": len(rows),
        "maximum_requirement_histogram": {
            str(value): histogram[value] for value in sorted(histogram)
        },
        "action_enumeration_hash": canonical_hash(enumeration_payload),
        "every_target_and_H_q_r_action_enumerated": True,
        "all_classwise_windows_feasible": True,
        "validated_before_gpu_or_classifier_scheduling": True,
        "labels_consumed": False,
    }
    return MappingProxyType({**body, "capacity_certificate_hash": canonical_hash(body)})


def validate_action_capacity_certificate(
    raw: object,
    *,
    centers: Sequence[str] = CENTERS,
    stream_rows_per_class: int = SOURCE_ROWS_PER_CLASS,
) -> Mapping[str, object]:
    """Reconstruct and validate a capacity certificate from typed action code."""

    if not isinstance(raw, Mapping):
        raise ProtocolError("HARP v14 action capacity certificate is malformed.")
    expected = dict(
        build_action_capacity_certificate(
            centers=centers,
            stream_rows_per_class=stream_rows_per_class,
        )
    )
    observed = dict(raw)
    if observed != expected:
        raise ProtocolError("HARP v14 action capacity certificate drifted.")
    return MappingProxyType(observed)


def _outer_targets(values: Sequence[str]) -> tuple[str, ...]:
    requested = tuple(str(value) for value in values)
    if (
        not requested
        or tuple(center for center in CENTERS if center in set(requested)) != requested
        or len(set(requested)) != len(requested)
    ):
        raise ProtocolError("HARP v14 capacity outer-target order is noncanonical.")
    return requested


__all__ = (
    "ACTION_CAPACITY_CERTIFICATE_SCHEMA",
    "ACTION_CAPACITY_ENUMERATION_SCHEMA",
    "ActionCapacityRequirement",
    "GLOBAL_MAX_REQUIRED_PER_CLASS",
    "SEVEN_SOURCE_MAX_REQUIRED_PER_CLASS",
    "SIX_SOURCE_MAX_REQUIRED_PER_CLASS",
    "TARGET_MAX_REQUIRED_PER_CLASS",
    "build_action_capacity_certificate",
    "enumerate_complete_action_capacity",
    "requirement_for_action",
    "validate_action_capacity",
    "validate_action_capacity_certificate",
)
