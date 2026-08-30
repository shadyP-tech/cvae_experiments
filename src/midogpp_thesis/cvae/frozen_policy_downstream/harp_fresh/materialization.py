"""Complete label-free B/U/Hxe probability materialization for fresh targets."""

from __future__ import annotations

from collections.abc import Callable

from ...expert_bank.uniform_b_v2_promotion.contracts import (
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from ...protocol import ProtocolError
from ...runtime.harp_probability_menu import (
    DEFAULT_WORKSTATION_CONTRACT,
    HarpActionSpec,
    HarpPredictionCell,
    HarpPredictionMenuSeal,
    HarpWorkstationContract,
    build_all_target_actions,
    seal_harp_prediction_menu,
)
from .contracts import (
    HarpFreshPredictionOutput,
    HarpFreshTargetCache,
    HarpFreshTargetFrame,
)
from .policy import FrozenHarpPolicy


PredictionProvider = Callable[
    [HarpActionSpec, int, int, HarpFreshTargetFrame],
    HarpFreshPredictionOutput,
]


def materialize_harp_fresh_probability_menu(
    policy: FrozenHarpPolicy,
    cache: HarpFreshTargetCache,
    predictor: PredictionProvider,
    *,
    workstation: HarpWorkstationContract = DEFAULT_WORKSTATION_CONTRACT,
) -> HarpPredictionMenuSeal:
    """Build all 810 target/action/seed cells before policy inference."""

    if not isinstance(policy, FrozenHarpPolicy):
        raise ProtocolError("Fresh probability materialization requires a frozen HARP policy.")
    if not isinstance(cache, HarpFreshTargetCache) or not callable(predictor):
        raise ProtocolError("Fresh HARP materialization requires an admitted cache and predictor.")
    if policy.metadata.fresh_reservation_hash != cache.reservation.reservation_hash:
        raise ProtocolError("Fresh HARP policy/cache reservation binding drifted.")

    actions = build_all_target_actions()
    cells: list[HarpPredictionCell] = []
    for action in actions:
        frame = cache.frames_by_center[action.outer_target_id]
        for training_seed in TRAINING_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                output = predictor(action, training_seed, generation_seed, frame)
                if not isinstance(output, HarpFreshPredictionOutput):
                    raise ProtocolError("Fresh HARP predictor returned an untyped output.")
                if len(output.probabilities) != len(frame.row_ids):
                    raise ProtocolError("Fresh HARP predictor row coverage drifted.")
                cells.append(
                    HarpPredictionCell(
                        action=action,
                        training_seed=training_seed,
                        generation_seed=generation_seed,
                        row_ids=frame.row_ids,
                        case_ids=frame.case_ids,
                        probabilities=output.probabilities,
                        bank_hash=policy.metadata.bank_hash,
                        generation_lock_hash=policy.metadata.generation_lock_hash,
                        source_cache_hash=policy.metadata.source_cache_hash,
                        frame_hash=frame.frame_hash,
                        classifier_hash=policy.metadata.classifier_hash,
                        composition_hash=output.composition_hash,
                        scaler_state_hash=output.scaler_state_hash,
                    )
                )
    expected_cells = len(actions) * len(TRAINING_SEEDS) * len(GENERATION_SEEDS)
    if len(cells) != expected_cells:
        raise ProtocolError("Fresh HARP global probability inventory drifted.")
    seal = seal_harp_prediction_menu(actions, cells, workstation=workstation)
    seal.assert_valid()
    return seal


__all__ = ("PredictionProvider", "materialize_harp_fresh_probability_menu")
