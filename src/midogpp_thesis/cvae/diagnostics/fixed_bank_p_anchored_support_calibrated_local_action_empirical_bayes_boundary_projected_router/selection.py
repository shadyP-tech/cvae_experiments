"""Direct conservative case-level action selection without a prefix model."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from .action_geometry import BoundaryProjection
from .hashing import canonical_hash, require_sha256
from .identity import ACTION_IDS, DIRECTIONS, TIE_TOLERANCE
from .protocol import ProtocolError
from .uncertainty import ActionEnvelope


@dataclass(frozen=True, slots=True)
class ActionCandidate:
    case_id: str
    projection: BoundaryProjection
    envelope: ActionEnvelope
    within_support: bool
    bank_viable: bool
    candidate_hash: str = field(init=False)

    def __post_init__(self) -> None:
        case_id = str(self.case_id)
        if (
            not case_id
            or self.projection.is_exact_p
            or self.projection.action_id != self.envelope.action_id
            or not isinstance(self.within_support, bool)
            or not isinstance(self.bank_viable, bool)
        ):
            raise ProtocolError("SCALE-BP action candidate drifted.")
        payload = {
            "schema_version": "scale_bp_action_candidate_v1",
            "case_id": case_id,
            "geometry_hash": self.projection.geometry_hash,
            "envelope_hash": self.envelope.envelope_hash,
            "within_support": self.within_support,
            "bank_viable": self.bank_viable,
        }
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "candidate_hash", canonical_hash(payload))

    @property
    def action_id(self) -> str:
        return self.projection.action_id

    @property
    def direction(self) -> str:
        return self.projection.direction

    @property
    def robustly_safe(self) -> bool:
        return (
            self.within_support
            and self.bank_viable
            and self.envelope.bacc_lower > TIE_TOLERANCE
            and self.envelope.brier_upper <= TIE_TOLERANCE
            and self.envelope.log_upper <= TIE_TOLERANCE
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "scale_bp_action_candidate_v1",
            "case_id": self.case_id,
            "geometry_hash": self.projection.geometry_hash,
            "envelope_hash": self.envelope.envelope_hash,
            "within_support": self.within_support,
            "bank_viable": self.bank_viable,
            "robustly_safe": self.robustly_safe,
            "candidate_hash": self.candidate_hash,
        }


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    case_id: str
    baseline_probability_hash: str
    selected_action_ids: tuple[str, ...]
    robust_bacc_lower: float
    brier_upper: float
    log_upper: float
    reason: str
    candidate_hashes: tuple[str, ...]
    decision_hash: str = field(init=False)

    def __post_init__(self) -> None:
        case_id = str(self.case_id)
        baseline_hash = require_sha256(
            self.baseline_probability_hash, "selection baseline probability hash"
        )
        selected = tuple(str(value) for value in self.selected_action_ids)
        candidate_hashes = tuple(str(value) for value in self.candidate_hashes)
        values = (
            float(self.robust_bacc_lower),
            float(self.brier_upper),
            float(self.log_upper),
        )
        reasons = {
            "SELECTED_ACTION",
            "SELECTED_DISJOINT_PAIR",
            "EXACT_P_NO_CANDIDATES",
            "EXACT_P_NO_ADMISSIBLE_ACTION",
        }
        if (
            not case_id
            or len(selected) > len(DIRECTIONS)
            or selected != tuple(sorted(set(selected)))
            or any(action_id not in ACTION_IDS for action_id in selected)
            or not all(math.isfinite(value) for value in values)
            or self.reason not in reasons
            or candidate_hashes != tuple(sorted(set(candidate_hashes)))
            or (not selected and values != (0.0, 0.0, 0.0))
            or (not selected and not self.reason.startswith("EXACT_P"))
            or (len(selected) == 1 and self.reason != "SELECTED_ACTION")
            or (len(selected) == 2 and self.reason != "SELECTED_DISJOINT_PAIR")
            or (
                selected
                and (
                    values[0] <= TIE_TOLERANCE
                    or values[1] > TIE_TOLERANCE
                    or values[2] > TIE_TOLERANCE
                )
            )
        ):
            raise ProtocolError("SCALE-BP selection decision drifted.")
        for digest in candidate_hashes:
            require_sha256(digest, "selection candidate hash")
        payload = {
            "schema_version": "scale_bp_selection_decision_v1",
            "case_id": case_id,
            "baseline_probability_hash": baseline_hash,
            "selected_action_ids": selected,
            "robust_bacc_lower": values[0],
            "brier_upper": values[1],
            "log_upper": values[2],
            "reason": self.reason,
            "candidate_hashes": candidate_hashes,
            "p_wins_tie_tolerance": TIE_TOLERANCE,
        }
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "baseline_probability_hash", baseline_hash)
        object.__setattr__(self, "selected_action_ids", selected)
        object.__setattr__(self, "robust_bacc_lower", values[0])
        object.__setattr__(self, "brier_upper", values[1])
        object.__setattr__(self, "log_upper", values[2])
        object.__setattr__(self, "candidate_hashes", candidate_hashes)
        object.__setattr__(self, "decision_hash", canonical_hash(payload))

    @property
    def is_exact_p(self) -> bool:
        return not self.selected_action_ids

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "scale_bp_selection_decision_v1",
            "case_id": self.case_id,
            "baseline_probability_hash": self.baseline_probability_hash,
            "selected_action_ids": self.selected_action_ids,
            "robust_bacc_lower": self.robust_bacc_lower,
            "brier_upper": self.brier_upper,
            "log_upper": self.log_upper,
            "reason": self.reason,
            "candidate_hashes": self.candidate_hashes,
            "p_wins_tie_tolerance": TIE_TOLERANCE,
            "decision_hash": self.decision_hash,
        }


def select_case_actions(
    *,
    case_id: str,
    baseline_probability_hash: str,
    candidates: object,
) -> SelectionDecision:
    """Select one action or one disjoint opposite-direction pair directly."""

    rows = tuple(candidates)  # type: ignore[arg-type]
    identity = str(case_id)
    baseline_hash = require_sha256(
        baseline_probability_hash, "selection baseline probability hash"
    )
    if any(not isinstance(row, ActionCandidate) for row in rows):
        raise ProtocolError("SCALE-BP selection candidate population drifted.")
    ordered = tuple(sorted(rows, key=lambda row: row.action_id))
    if len({row.action_id for row in ordered}) != len(ordered):
        raise ProtocolError("SCALE-BP case has duplicate action candidates.")
    if any(
        row.case_id != identity
        or row.projection.baseline_probability_hash != baseline_hash
        for row in ordered
    ):
        raise ProtocolError("SCALE-BP selection candidate lineage drifted.")
    candidate_hashes = tuple(sorted(row.candidate_hash for row in ordered))
    if not ordered:
        return SelectionDecision(
            identity,
            baseline_hash,
            (),
            0.0,
            0.0,
            0.0,
            "EXACT_P_NO_CANDIDATES",
            candidate_hashes,
        )

    eligible = tuple(row for row in ordered if row.robustly_safe)
    options: list[tuple[tuple[str, ...], float, float, float]] = []
    for row in eligible:
        options.append(
            (
                (row.action_id,),
                row.envelope.bacc_lower,
                row.envelope.brier_upper,
                row.envelope.log_upper,
            )
        )
    for index, left in enumerate(eligible):
        for right in eligible[index + 1 :]:
            if left.direction == right.direction or set(left.projection.crossing_indices).intersection(
                right.projection.crossing_indices
            ):
                continue
            bacc_lower = left.envelope.bacc_lower + right.envelope.bacc_lower
            brier_upper = left.envelope.brier_upper + right.envelope.brier_upper
            log_upper = left.envelope.log_upper + right.envelope.log_upper
            if (
                bacc_lower > TIE_TOLERANCE
                and brier_upper <= TIE_TOLERANCE
                and log_upper <= TIE_TOLERANCE
            ):
                options.append(
                    (
                        tuple(sorted((left.action_id, right.action_id))),
                        bacc_lower,
                        brier_upper,
                        log_upper,
                    )
                )
    if not options:
        return SelectionDecision(
            identity,
            baseline_hash,
            (),
            0.0,
            0.0,
            0.0,
            "EXACT_P_NO_ADMISSIBLE_ACTION",
            candidate_hashes,
        )

    best: tuple[tuple[str, ...], float, float, float] | None = None
    for option in sorted(options, key=lambda row: (len(row[0]), row[0])):
        if best is None or option[1] > best[1] + TIE_TOLERANCE:
            best = option
        elif abs(option[1] - best[1]) <= TIE_TOLERANCE and (
            len(option[0]), option[0]
        ) < (len(best[0]), best[0]):
            best = option
    assert best is not None
    return SelectionDecision(
        identity,
        baseline_hash,
        best[0],
        best[1],
        best[2],
        best[3],
        "SELECTED_ACTION" if len(best[0]) == 1 else "SELECTED_DISJOINT_PAIR",
        candidate_hashes,
    )


__all__ = ("ActionCandidate", "SelectionDecision", "select_case_actions")
