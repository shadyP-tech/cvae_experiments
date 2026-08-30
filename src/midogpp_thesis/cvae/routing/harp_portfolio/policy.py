"""Conservative HARP selection using only sealed probabilities and features."""

from __future__ import annotations

from collections import defaultdict
import math
from statistics import median
import struct
from typing import Sequence

from ...protocol import ProtocolError
from ..harp_action_model import LAMBDA_GRID, HarpActionScore
from .contracts import HarpConservativeAction, HarpPolicyConfig, HarpPortfolioDecision


MAD_SCALE = 1.4826


def _median_mad(values: tuple[float, ...]) -> tuple[float, float]:
    center = float(median(values))
    spread = MAD_SCALE * float(median(tuple(abs(value - center) for value in values)))
    return center, spread


def _conservative_action_with_rho(
    score: HarpActionScore,
    config: HarpPolicyConfig,
    *,
    rho: float,
) -> HarpConservativeAction:
    if not isinstance(score, HarpActionScore) or not isinstance(config, HarpPolicyConfig):
        raise ProtocolError("HARP policy requires typed score and config contracts.")
    # Compatibility can only weaken evidence: positive gains and negative loss
    # improvements shrink toward zero; harmful values are never attenuated.
    gains = tuple(value if value <= 0 else rho * value for value in score.gain_predictions)
    briers = tuple(value if value >= 0 else rho * value for value in score.brier_predictions)
    losses = tuple(value if value >= 0 else rho * value for value in score.log_loss_predictions)
    gain_center, gain_spread = _median_mad(gains)
    brier_center, brier_spread = _median_mad(briers)
    loss_center, loss_spread = _median_mad(losses)
    gain_lower = gain_center - config.kappa_gain * gain_spread
    brier_upper = brier_center + config.kappa_loss * brier_spread
    log_loss_upper = loss_center + config.kappa_loss * loss_spread
    maximum_leverage = max(score.leverages)
    reasons: list[str] = []
    if score.support.donor_count < config.min_donor_count:
        reasons.append("insufficient_donor_coverage")
    if score.support.paired_case_count < config.min_paired_case_count:
        reasons.append("insufficient_paired_cases")
    if score.support.truth_classes != (0, 1):
        reasons.append("both_source_truth_classes_not_covered")
    if maximum_leverage > config.max_leverage:
        reasons.append("excessive_leverage")
    if rho < config.min_compatibility_shrinkage:
        reasons.append("compatibility_abstention")
    if gain_lower <= config.gain_threshold:
        reasons.append("gain_lower_bound_not_positive")
    if brier_upper > config.brier_noninferiority_margin:
        reasons.append("brier_noninferiority_failed")
    if log_loss_upper > config.log_loss_noninferiority_margin:
        reasons.append("log_loss_noninferiority_failed")
    return HarpConservativeAction(score, gain_lower, brier_upper, log_loss_upper, maximum_leverage, not reasons, tuple(reasons))


def conservative_action(score: HarpActionScore, config: HarpPolicyConfig) -> HarpConservativeAction:
    if not isinstance(score, HarpActionScore) or not isinstance(config, HarpPolicyConfig):
        raise ProtocolError("HARP policy requires typed score and config contracts.")
    adjusted = _conservative_action_with_rho(
        score,
        config,
        rho=score.action.compatibility_shrinkage,
    )
    source_only = _conservative_action_with_rho(score, config, rho=1.0)
    # The support envelope is a one-way veto.  It cannot turn an action that
    # failed source-trained evidence into an authorized route.
    if adjusted.eligible and not source_only.eligible:
        return HarpConservativeAction(
            score,
            adjusted.gain_lower,
            adjusted.brier_upper,
            adjusted.log_loss_upper,
            adjusted.maximum_leverage,
            False,
            (*adjusted.rejection_reasons, "support_envelope_may_not_authorize"),
        )
    return adjusted


def _validate_complete_grid(
    block: tuple[HarpActionScore, ...],
    *,
    expected_lambdas: tuple[float, ...],
) -> None:
    by_source: dict[str, set[float]] = defaultdict(set)
    for row in block:
        by_source[row.action.candidate_source_id].add(row.action.lambda_value)
    if not by_source or any(
        tuple(sorted(values)) != expected_lambdas for values in by_source.values()
    ):
        raise ProtocolError("Every HARP candidate must cover the complete frozen lambda grid.")
    if len(block) != len(by_source) * len(expected_lambdas):
        raise ProtocolError("HARP portfolio contains duplicate candidate-lambda actions.")


def _select_harp_portfolio(
    scores: Sequence[HarpActionScore],
    *,
    config: HarpPolicyConfig,
    expected_lambdas: tuple[float, ...],
) -> tuple[HarpPortfolioDecision, ...]:
    rows = tuple(scores)
    if not rows or any(not isinstance(row, HarpActionScore) for row in rows):
        raise ProtocolError("HARP portfolio selection requires typed action scores.")
    grouped: dict[tuple[str, str, str], list[HarpActionScore]] = defaultdict(list)
    for row in rows:
        grouped[row.action.group_key].append(row)
    decisions: list[HarpPortfolioDecision] = []
    for group_key in sorted(grouped):
        block = tuple(grouped[group_key])
        _validate_complete_grid(block, expected_lambdas=expected_lambdas)
        reference_bytes = block[0].action.baseline_probability_bytes
        fallback_bytes = block[0].action.operational_fallback_probability_bytes
        assert isinstance(fallback_bytes, bytes)
        seals = {row.action.prediction_seal_hash for row in block}
        if (
            any(
                row.action.baseline_probability_bytes != reference_bytes
                or row.action.operational_fallback_probability_bytes != fallback_bytes
                for row in block
            )
            or len(seals) != 1
        ):
            raise ProtocolError(
                "HARP action group drifted in U-reference/B-fallback bytes or prediction seal."
            )
        source_candidates = tuple(
            _conservative_action_with_rho(row, config, rho=1.0)
            for row in block
        )
        source_eligible = tuple(row for row in source_candidates if row.eligible)
        receipts = {row.action.ensemble_receipt_hash for row in block}
        if len(receipts) != 1:
            raise ProtocolError("HARP action group drifted across exact-nine ensemble receipts.")
        ensemble_receipt_hash = next(iter(receipts))
        outer_target_id, case_id, sample_id = group_key
        if not source_eligible:
            decisions.append(HarpPortfolioDecision(outer_target_id, case_id, sample_id, fallback_bytes, fallback_bytes, None, None, False, "EXACT_B_FALLBACK_NO_ADMISSIBLE_ACTION", None, None, None, next(iter(seals)), ensemble_receipt_hash))
            continue
        source_winner = min(
            source_eligible,
            key=lambda row: (
                -row.gain_lower,
                row.brier_upper,
                row.log_loss_upper,
                row.score.action.lambda_value,
                row.score.action.candidate_source_id,
            ),
        )
        chosen = conservative_action(source_winner.score, config)
        if not chosen.eligible:
            decisions.append(
                HarpPortfolioDecision(
                    outer_target_id,
                    case_id,
                    sample_id,
                    fallback_bytes,
                    fallback_bytes,
                    None,
                    None,
                    False,
                    "EXACT_B_FALLBACK_SUPPORT_ENVELOPE_REJECTED_SOURCE_WINNER",
                    None,
                    None,
                    None,
                    next(iter(seals)),
                    ensemble_receipt_hash,
                )
            )
            continue
        action = chosen.score.action
        probability = (1.0 - action.lambda_value) * action.baseline_probability + action.lambda_value * action.expert_probability
        output_bytes = struct.pack("<d", float(probability))
        decisions.append(HarpPortfolioDecision(outer_target_id, case_id, sample_id, fallback_bytes, output_bytes, action.candidate_source_id, action.lambda_value, True, "CONSERVATIVE_ACTION_ADMITTED", chosen.gain_lower, chosen.brier_upper, chosen.log_loss_upper, action.prediction_seal_hash, ensemble_receipt_hash))
    return tuple(decisions)


def select_harp_portfolio(
    scores: Sequence[HarpActionScore],
    *,
    config: HarpPolicyConfig = HarpPolicyConfig(),
) -> tuple[HarpPortfolioDecision, ...]:
    """Select from the complete frozen predictive-lambda portfolio."""

    return _select_harp_portfolio(
        scores,
        config=config,
        expected_lambdas=LAMBDA_GRID,
    )


def select_harp_physical_portfolio(
    scores: Sequence[HarpActionScore],
    *,
    config: HarpPolicyConfig = HarpPolicyConfig(),
) -> tuple[HarpPortfolioDecision, ...]:
    """Select only physical Hxe endpoints with the identical gates and veto."""

    return _select_harp_portfolio(
        scores,
        config=config,
        expected_lambdas=(1.0,),
    )


__all__ = (
    "MAD_SCALE",
    "conservative_action",
    "select_harp_physical_portfolio",
    "select_harp_portfolio",
)
