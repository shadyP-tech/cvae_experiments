"""Nested donor-held calibration of the scalar PUMR abstention margin."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    ALTERNATIVE_METHOD_IDS,
    CENTERS,
    MARGIN_TIE_TOLERANCE,
    MARGIN_MIN,
    P_FALLBACK_MARGIN,
    ROBUST_MAD_SCALE,
    UTILITY_ZERO_TOLERANCE,
)
from .hashing import canonical_hash
from .utility_contracts import (
    DonorUtilityRow,
    InnerDonorReplay,
    MarginCalibration,
    PosteriorUtilityPrediction,
)


def calibrate_margin(
    *,
    outer_target_center: str,
    control_id: str,
    predictions: Sequence[PosteriorUtilityPrediction],
    donor_rows: Sequence[DonorUtilityRow],
) -> MarginCalibration:
    """Choose a margin on donors and audit it by inner leave-one-donor replay."""

    utilities = tuple(predictions)
    responses = tuple(donor_rows)
    donors = tuple(center for center in CENTERS if center != outer_target_center)
    if (
        outer_target_center not in CENTERS
        or not utilities
        or not responses
        or {row.target_center for row in utilities} != set(donors)
        or {row.outer_target_center for row in responses} != {outer_target_center}
        or {row.donor_center for row in responses} != set(donors)
        or {row.control_id for row in utilities} != {control_id}
        or {row.descriptor_hash for row in utilities}
        != {row.descriptor_hash for row in responses}
    ):
        raise ProtocolError("PUMR margin calibration scope drifted.")
    candidate_margins = _candidate_margins(utilities)
    inner: list[InnerDonorReplay] = []
    for held_donor in donors:
        training_donors = tuple(center for center in donors if center != held_donor)
        inner_candidates = _candidate_margins(
            row for row in utilities if row.target_center in training_donors
        )
        margin, _training = _choose_margin(
            inner_candidates,
            utilities,
            responses,
            allowed_donors=training_donors,
        )
        held = _evaluate_margin(
            margin,
            utilities,
            responses,
            allowed_donors=(held_donor,),
        )
        center_metrics = held["by_center"][held_donor]
        inner.append(
            InnerDonorReplay(
                outer_target_center,
                control_id,
                held_donor,
                margin,
                int(held["selected_action_count"]),
                center_metrics[0],
                center_metrics[1],
                center_metrics[2],
            )
        )
    selected_margin, final = _choose_margin(
        candidate_margins,
        utilities,
        responses,
        allowed_donors=donors,
    )
    authorized = selected_margin >= MARGIN_MIN and all(
        row.bacc_delta >= -UTILITY_ZERO_TOLERANCE
        and row.brier_delta <= UTILITY_ZERO_TOLERANCE
        and row.log_loss_delta <= UTILITY_ZERO_TOLERANCE
        for row in inner
    )
    return MarginCalibration(
        outer_target_center,
        control_id,
        selected_margin,
        authorized,
        candidate_margins,
        int(final["selected_action_count"]),
        float(final["median_bacc"]),
        float(final["median_brier"]),
        float(final["median_log_loss"]),
        tuple(inner),
        canonical_hash([row.to_payload() for row in sorted(utilities)]),
        canonical_hash([row.to_payload() for row in sorted(responses, key=lambda row: row.key)]),
    )


def select_prediction_for_direction(
    predictions: Sequence[PosteriorUtilityPrediction],
    *,
    margin: float,
    require_proper: bool,
) -> PosteriorUtilityPrediction | None:
    """Apply the frozen deterministic candidate rule for one direction."""

    rows = tuple(predictions)
    if (
        len(rows) != len(ALTERNATIVE_METHOD_IDS)
        or len({row.alternative for row in rows}) != len(ALTERNATIVE_METHOD_IDS)
        or len({row.direction for row in rows}) != 1
        or len({(row.target_center, row.case_id, row.control_id) for row in rows}) != 1
        or not np.isfinite(margin)
        or margin < 0.0
    ):
        raise ProtocolError("PUMR directional posterior candidate set drifted.")
    alternative_order = {
        alternative: index for index, alternative in enumerate(ALTERNATIVE_METHOD_IDS)
    }
    admissible = tuple(
        row
        for row in rows
        if row.crossing_count > 0
        and row.reliability_pass
        and row.robust_bacc_lower > margin + UTILITY_ZERO_TOLERANCE
        and (row.proper_safe or not require_proper)
    )
    return max(
        admissible,
        default=None,
        key=lambda row: (
            row.robust_bacc_lower,
            -alternative_order[row.alternative],
        ),
    )


def _candidate_margins(
    predictions: Iterable[PosteriorUtilityPrediction],
) -> tuple[float, ...]:
    values = {
        0.0,
        MARGIN_MIN,
        P_FALLBACK_MARGIN,
        *(
            float(row.robust_bacc_lower)
            for row in predictions
            if row.robust_bacc_lower > 0.0
            and row.robust_bacc_lower < P_FALLBACK_MARGIN
        ),
    }
    return tuple(sorted(values))


def _choose_margin(
    candidates: Sequence[float],
    predictions: Sequence[PosteriorUtilityPrediction],
    responses: Sequence[DonorUtilityRow],
    *,
    allowed_donors: Sequence[str],
) -> tuple[float, dict[str, object]]:
    evaluated = tuple(
        (
            float(margin),
            _evaluate_margin(
                float(margin),
                predictions,
                responses,
                allowed_donors=allowed_donors,
            ),
        )
        for margin in candidates
    )
    safe = tuple(
        row
        for row in evaluated
        if all(
            bacc >= -UTILITY_ZERO_TOLERANCE
            and brier <= UTILITY_ZERO_TOLERANCE
            and log_loss <= UTILITY_ZERO_TOLERANCE
            for bacc, brier, log_loss in row[1]["by_center"].values()
        )
    )
    pool = safe or tuple(
        row for row in evaluated if row[0] == P_FALLBACK_MARGIN
    )
    if not pool:
        raise ProtocolError("PUMR fallback margin is absent.")
    best = pool[0]
    for row in pool[1:]:
        objective_delta = float(row[1]["robust_objective"]) - float(
            best[1]["robust_objective"]
        )
        if objective_delta > MARGIN_TIE_TOLERANCE or (
            abs(objective_delta) <= MARGIN_TIE_TOLERANCE
            and (
                row[0] > best[0]
                or (
                    row[0] == best[0]
                    and int(row[1]["selected_action_count"])
                    < int(best[1]["selected_action_count"])
                )
            )
        ):
            best = row
    return best


def _evaluate_margin(
    margin: float,
    predictions: Sequence[PosteriorUtilityPrediction],
    responses: Sequence[DonorUtilityRow],
    *,
    allowed_donors: Sequence[str],
) -> dict[str, object]:
    allowed = set(allowed_donors)
    prediction_groups: dict[
        tuple[str, str, str], list[PosteriorUtilityPrediction]
    ] = {}
    for row in predictions:
        if row.target_center in allowed:
            prediction_groups.setdefault(
                (row.target_center, row.case_id, row.direction), []
            ).append(row)
    response_by_hash = {row.descriptor_hash: row for row in responses}
    by_center = {center: [0.0, 0.0, 0.0] for center in allowed_donors}
    selected_count = 0
    for (donor, _case, _direction), rows in sorted(prediction_groups.items()):
        selected = select_prediction_for_direction(
            rows, margin=margin, require_proper=True
        )
        if selected is None:
            continue
        response = response_by_hash[selected.descriptor_hash]
        if response.donor_center != donor:
            raise ProtocolError("PUMR donor response binding drifted.")
        by_center[donor][0] += response.bacc_contribution_delta
        by_center[donor][1] += response.brier_contribution_delta
        by_center[donor][2] += response.log_loss_contribution_delta
        selected_count += 1
    frozen = {
        center: (float(values[0]), float(values[1]), float(values[2]))
        for center, values in by_center.items()
    }
    bacc = np.asarray([values[0] for values in frozen.values()], dtype=np.float64)
    brier = np.asarray([values[1] for values in frozen.values()], dtype=np.float64)
    log_loss = np.asarray([values[2] for values in frozen.values()], dtype=np.float64)
    median_bacc = float(np.median(bacc))
    mad_bacc = float(np.median(np.abs(bacc - median_bacc)))
    return {
        "by_center": frozen,
        "selected_action_count": selected_count,
        "median_bacc": median_bacc,
        "median_brier": float(np.median(brier)),
        "median_log_loss": float(np.median(log_loss)),
        "robust_objective": median_bacc - ROBUST_MAD_SCALE * mad_bacc,
    }


__all__ = ("calibrate_margin", "select_prediction_for_direction")
