"""Frozen method adapters and direct replay-score selection."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from ..action_geometry import probability_hash
from ..controls import (
    CYCLIC_ACTION_IDENTITY,
    DONOR_ONLY,
    FULL_ENDPOINT_SENSITIVITY,
    LEGACY_SAME_RUN,
    LOCAL_ONLY,
    P_PROTECTED,
    SCALE_BP_PRIMARY,
    SUPPORT_LABEL_PERMUTATION,
)
from ..engine import CaseRouteRequest, CaseRouteResult
from ..hashing import canonical_hash, require_sha256
from ..identity import ACTION_IDS, DIRECTIONS, TIE_TOLERANCE
from ..protocol import ProtocolError


@dataclass(frozen=True, slots=True)
class ReplayActionScore:
    method_id: str
    action_id: str
    direction: str
    crossing_indices: tuple[int, ...]
    bacc_lower: float
    brier_upper: float
    log_upper: float
    within_support: bool
    bank_viable: bool
    source_hash: str
    score_hash: str = field(init=False)

    def __post_init__(self) -> None:
        crossing = tuple(int(value) for value in self.crossing_indices)
        values = (
            float(self.bacc_lower),
            float(self.brier_upper),
            float(self.log_upper),
        )
        source_hash = require_sha256(self.source_hash, "replay-score source hash")
        if (
            self.method_id not in {
                P_PROTECTED,
                SCALE_BP_PRIMARY,
                DONOR_ONLY,
                LOCAL_ONLY,
                LEGACY_SAME_RUN,
                SUPPORT_LABEL_PERMUTATION,
                CYCLIC_ACTION_IDENTITY,
                FULL_ENDPOINT_SENSITIVITY,
            }
            or self.action_id not in ACTION_IDS
            or self.direction not in DIRECTIONS
            or self.direction != self.action_id.split("::", 1)[1]
            or crossing != tuple(sorted(set(crossing)))
            or any(value < 0 for value in crossing)
            or not all(math.isfinite(value) for value in values)
            or type(self.within_support) is not bool
            or type(self.bank_viable) is not bool
        ):
            raise ProtocolError("SCALE-BP replay action score drifted.")
        payload = {
            "schema_version": "scale_bp_replay_action_score_v1",
            "method_id": self.method_id,
            "action_id": self.action_id,
            "direction": self.direction,
            "crossing_indices": crossing,
            "bacc_lower": values[0],
            "brier_upper": values[1],
            "log_upper": values[2],
            "within_support": self.within_support,
            "bank_viable": self.bank_viable,
            "source_hash": source_hash,
        }
        object.__setattr__(self, "crossing_indices", crossing)
        object.__setattr__(self, "bacc_lower", values[0])
        object.__setattr__(self, "brier_upper", values[1])
        object.__setattr__(self, "log_upper", values[2])
        object.__setattr__(self, "source_hash", source_hash)
        object.__setattr__(self, "score_hash", canonical_hash(payload))

    @property
    def opportunity(self) -> bool:
        return bool(self.crossing_indices)

    @property
    def robustly_safe(self) -> bool:
        return (
            self.opportunity
            and self.within_support
            and self.bank_viable
            and self.bacc_lower > TIE_TOLERANCE
            and self.brier_upper <= TIE_TOLERANCE
            and self.log_upper <= TIE_TOLERANCE
        )


@dataclass(frozen=True, slots=True)
class ReplayDecision:
    method_id: str
    case_id: str
    baseline_probability_hash: str
    selected_action_ids: tuple[str, ...]
    robust_bacc_lower: float
    brier_upper: float
    log_upper: float
    score_hashes: tuple[str, ...]
    reason: str
    decision_hash: str = field(init=False)

    def __post_init__(self) -> None:
        baseline_hash = require_sha256(
            self.baseline_probability_hash, "replay-decision baseline hash"
        )
        selected = tuple(str(value) for value in self.selected_action_ids)
        values = (
            float(self.robust_bacc_lower),
            float(self.brier_upper),
            float(self.log_upper),
        )
        score_hashes = tuple(str(value) for value in self.score_hashes)
        for digest in score_hashes:
            require_sha256(digest, "replay-decision score hash")
        if (
            not self.case_id
            or any(value not in ACTION_IDS for value in selected)
            or selected != tuple(sorted(set(selected)))
            or len(selected) > len(DIRECTIONS)
            or not all(math.isfinite(value) for value in values)
            or score_hashes != tuple(sorted(set(score_hashes)))
            or len(score_hashes) != len(ACTION_IDS)
            or (not selected and values != (0.0, 0.0, 0.0))
            or (selected and values[0] <= TIE_TOLERANCE)
            or (selected and values[1] > TIE_TOLERANCE)
            or (selected and values[2] > TIE_TOLERANCE)
            or (self.method_id == P_PROTECTED and selected)
        ):
            raise ProtocolError("SCALE-BP replay decision drifted.")
        payload = {
            "schema_version": "scale_bp_replay_decision_v1",
            "method_id": self.method_id,
            "case_id": self.case_id,
            "baseline_probability_hash": baseline_hash,
            "selected_action_ids": selected,
            "robust_bacc_lower": values[0],
            "brier_upper": values[1],
            "log_upper": values[2],
            "score_hashes": score_hashes,
            "reason": self.reason,
            "p_wins_tie_tolerance": TIE_TOLERANCE,
        }
        object.__setattr__(self, "baseline_probability_hash", baseline_hash)
        object.__setattr__(self, "selected_action_ids", selected)
        object.__setattr__(self, "robust_bacc_lower", values[0])
        object.__setattr__(self, "brier_upper", values[1])
        object.__setattr__(self, "log_upper", values[2])
        object.__setattr__(self, "score_hashes", score_hashes)
        object.__setattr__(self, "decision_hash", canonical_hash(payload))


def _empty_score(
    request: CaseRouteRequest,
    *,
    method_id: str,
    action_id: str,
) -> ReplayActionScore:
    return ReplayActionScore(
        method_id=method_id,
        action_id=action_id,
        direction=action_id.split("::", 1)[1],
        crossing_indices=(),
        bacc_lower=0.0,
        brier_upper=0.0,
        log_upper=0.0,
        within_support=False,
        bank_viable=False,
        source_hash=canonical_hash(
            {
                "schema_version": "scale_bp_no_opportunity_action_v1",
                "route_request_hash": request.request_hash,
                "method_id": method_id,
                "action_id": action_id,
            }
        ),
    )


def scores_from_route_result(
    request: CaseRouteRequest,
    result: CaseRouteResult,
    *,
    method_id: str,
) -> tuple[ReplayActionScore, ...]:
    """Adapt one actual engine replay into a complete six-action score menu."""

    if method_id not in {SCALE_BP_PRIMARY, SUPPORT_LABEL_PERMUTATION}:
        raise ProtocolError("SCALE-BP route-result adapter method drifted.")
    if result.request_hash != request.request_hash or result.case_id != request.case_id:
        raise ProtocolError("SCALE-BP route-result adapter lineage drifted.")
    candidates = {row.action_id: row for row in result.candidates}
    rows = []
    for action_id in ACTION_IDS:
        candidate = candidates.get(action_id)
        if candidate is None:
            rows.append(_empty_score(request, method_id=method_id, action_id=action_id))
            continue
        envelope = candidate.envelope
        rows.append(
            ReplayActionScore(
                method_id=method_id,
                action_id=action_id,
                direction=candidate.direction,
                crossing_indices=candidate.projection.crossing_indices,
                bacc_lower=envelope.bacc_lower,
                brier_upper=envelope.brier_upper,
                log_upper=envelope.log_upper,
                within_support=candidate.within_support,
                bank_viable=candidate.bank_viable,
                source_hash=candidate.candidate_hash,
            )
        )
    return tuple(rows)


def ablation_scores(
    request: CaseRouteRequest,
    result: CaseRouteResult,
    *,
    method_id: str,
) -> tuple[ReplayActionScore, ...]:
    """Freeze donor-only, local-only, and unshrunk legacy score semantics."""

    if method_id not in {DONOR_ONLY, LOCAL_ONLY, LEGACY_SAME_RUN}:
        raise ProtocolError("SCALE-BP replay ablation identity drifted.")
    by_input = {row.action_id: row for row in request.action_inputs}
    calibrations = {row.action_id: row for row in result.calibrations}
    estimates = {row.action_id: row for row in result.estimates}
    candidates = {row.action_id: row for row in result.candidates}
    if set(by_input) != set(calibrations) or set(by_input) != set(estimates):
        raise ProtocolError("SCALE-BP replay ablation lineage drifted.")
    rows = []
    for action_id in ACTION_IDS:
        action_input = by_input.get(action_id)
        if action_input is None:
            rows.append(_empty_score(request, method_id=method_id, action_id=action_id))
            continue
        calibration = calibrations[action_id]
        estimate = estimates[action_id]
        candidate = candidates[action_id]
        if method_id == DONOR_ONLY:
            point = action_input.donor_prediction.mean
            uncertainty = action_input.donor_prediction.between_center_standard_error
            residual = (0.0, 0.0, 0.0)
        elif method_id == LOCAL_ONLY:
            point = estimate.local_residual
            uncertainty = calibration.local_standard_error
            radius = result.selection_radius
            if radius is None:
                raise ProtocolError("SCALE-BP local-only radius is absent.")
            residual = radius.radius.as_tuple()
        else:
            point = action_input.donor_prediction.mean.plus(estimate.local_residual)
            uncertainty = None
            residual = (0.0, 0.0, 0.0)
        if uncertainty is None:
            bacc, brier, log = point.as_tuple()
        else:
            bacc = point.bacc_gain - uncertainty.bacc - residual[0]
            brier = point.brier_loss_delta + uncertainty.brier + residual[1]
            log = point.log_loss_delta + uncertainty.log + residual[2]
        rows.append(
            ReplayActionScore(
                method_id=method_id,
                action_id=action_id,
                direction=candidate.direction,
                crossing_indices=candidate.projection.crossing_indices,
                bacc_lower=bacc,
                brier_upper=brier,
                log_upper=log,
                within_support=calibration.within_support,
                bank_viable=calibration.bank_viable,
                source_hash=canonical_hash(
                    {
                        "schema_version": "scale_bp_ablation_score_source_v1",
                        "method_id": method_id,
                        "route_request_hash": request.request_hash,
                        "candidate_hash": candidate.candidate_hash,
                        "calibration_hash": calibration.calibration_hash,
                        "estimate_hash": estimate.estimate_hash,
                        "donor_prediction_hash": (
                            action_input.donor_prediction.prediction_hash
                        ),
                        "unshrunk_legacy": method_id == LEGACY_SAME_RUN,
                    }
                ),
            )
        )
    return tuple(rows)


def protected_scores(
    request: CaseRouteRequest,
    primary: tuple[ReplayActionScore, ...],
) -> tuple[ReplayActionScore, ...]:
    return tuple(
        ReplayActionScore(
            method_id=P_PROTECTED,
            action_id=row.action_id,
            direction=row.direction,
            crossing_indices=row.crossing_indices,
            bacc_lower=0.0,
            brier_upper=0.0,
            log_upper=0.0,
            within_support=False,
            bank_viable=False,
            source_hash=canonical_hash(
                {
                    "schema_version": "scale_bp_protected_score_source_v1",
                    "route_request_hash": request.request_hash,
                    "primary_score_hash": row.score_hash,
                }
            ),
        )
        for row in primary
    )


def relabel_scores(
    primary: tuple[ReplayActionScore, ...],
    *,
    method_id: str,
) -> tuple[ReplayActionScore, ...]:
    if method_id == FULL_ENDPOINT_SENSITIVITY:
        source_rows = primary
    elif method_id == CYCLIC_ACTION_IDENTITY:
        source_rows = primary[1:] + primary[:1]
    else:
        raise ProtocolError("SCALE-BP score relabel identity drifted.")
    return tuple(
        ReplayActionScore(
            method_id=method_id,
            action_id=target.action_id,
            direction=target.direction,
            crossing_indices=target.crossing_indices,
            bacc_lower=source.bacc_lower,
            brier_upper=source.brier_upper,
            log_upper=source.log_upper,
            within_support=source.within_support,
            bank_viable=source.bank_viable,
            source_hash=canonical_hash(
                {
                    "schema_version": "scale_bp_relabelled_score_source_v1",
                    "method_id": method_id,
                    "target_action_id": target.action_id,
                    "source_action_id": source.action_id,
                    "target_score_hash": target.score_hash,
                    "source_score_hash": source.score_hash,
                }
            ),
        )
        for target, source in zip(primary, source_rows, strict=True)
    )


def select_replay_actions(
    request: CaseRouteRequest,
    scores: tuple[ReplayActionScore, ...],
    *,
    method_id: str,
) -> ReplayDecision:
    if (
        tuple(row.action_id for row in scores) != ACTION_IDS
        or any(row.method_id != method_id for row in scores)
    ):
        raise ProtocolError("SCALE-BP replay score menu is incomplete.")
    score_hashes = tuple(sorted(row.score_hash for row in scores))
    baseline_hash = probability_hash(request.portfolio_probabilities)
    if method_id == P_PROTECTED:
        return ReplayDecision(
            method_id, request.case_id, baseline_hash, (), 0.0, 0.0, 0.0,
            score_hashes, "EXACT_P_PROTECTED",
        )
    eligible = tuple(row for row in scores if row.robustly_safe)
    options: list[tuple[tuple[str, ...], float, float, float]] = []
    for row in eligible:
        options.append(
            ((row.action_id,), row.bacc_lower, row.brier_upper, row.log_upper)
        )
    for index, left in enumerate(eligible):
        for right in eligible[index + 1 :]:
            if left.direction == right.direction or set(
                left.crossing_indices
            ).intersection(right.crossing_indices):
                continue
            option = (
                tuple(sorted((left.action_id, right.action_id))),
                left.bacc_lower + right.bacc_lower,
                left.brier_upper + right.brier_upper,
                left.log_upper + right.log_upper,
            )
            if (
                option[1] > TIE_TOLERANCE
                and option[2] <= TIE_TOLERANCE
                and option[3] <= TIE_TOLERANCE
            ):
                options.append(option)
    best: tuple[tuple[str, ...], float, float, float] | None = None
    for option in sorted(options, key=lambda row: (len(row[0]), row[0])):
        if best is None or option[1] > best[1] + TIE_TOLERANCE:
            best = option
        elif abs(option[1] - best[1]) <= TIE_TOLERANCE and (
            len(option[0]), option[0]
        ) < (len(best[0]), best[0]):
            best = option
    if best is None:
        return ReplayDecision(
            method_id, request.case_id, baseline_hash, (), 0.0, 0.0, 0.0,
            score_hashes, "EXACT_P_NO_ADMISSIBLE_ACTION",
        )
    return ReplayDecision(
        method_id=method_id,
        case_id=request.case_id,
        baseline_probability_hash=baseline_hash,
        selected_action_ids=best[0],
        robust_bacc_lower=best[1],
        brier_upper=best[2],
        log_upper=best[3],
        score_hashes=score_hashes,
        reason="SELECTED_ACTION" if len(best[0]) == 1 else "SELECTED_DISJOINT_PAIR",
    )


__all__ = (
    "ReplayActionScore",
    "ReplayDecision",
    "ablation_scores",
    "protected_scores",
    "relabel_scores",
    "scores_from_route_result",
    "select_replay_actions",
)
