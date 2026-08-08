"""Fast label-free validation of persisted prediction bindings."""

from __future__ import annotations

import json
from typing import Mapping

from ...generation.generation import derived_composition_seed
from ...protocol import ProtocolError
from .contracts import (
    BASE_ACTION_ID,
    CENTERS,
    GENERATION_SEEDS,
    GLOBAL_ACTION_ID,
    TRAINING_SEEDS,
    candidate_sources,
    expected_action_ids,
)
from .prediction_store import EXPECTED_PREDICTION_CELL_COUNT, PredictionStore


def validate_prediction_store_binding(
    store: PredictionStore,
    *,
    config: object,
    generation_lock_hash: str,
    source_cache: object,
    source_cache_lock_hash: str,
    plan: object,
    crossfit: object,
) -> None:
    """Bind every cell to a frozen fold, target action, stream, and row slice."""

    rows = tuple(store.index_rows)
    if len(rows) != EXPECTED_PREDICTION_CELL_COUNT:
        raise ProtocolError("Case-OOF prediction binding coverage drifted.")
    source_rows = {
        (
            str(row["source_center"]),
            int(row["training_seed"]),
            int(row["generation_seed"]),
        ): row
        for row in getattr(source_cache, "index_rows")
    }
    expected_keys: list[tuple[object, ...]] = []
    observed_keys: list[tuple[object, ...]] = []
    for target in CENTERS:
        for training_seed in TRAINING_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                for fold in crossfit.folds_by_target[target]:
                    for action_id in expected_action_ids(target):
                        expected_keys.append(
                            (
                                target,
                                training_seed,
                                generation_seed,
                                fold.fold_id,
                                action_id,
                            )
                        )

    distinct_fits: set[tuple[object, ...]] = set()
    aliases_by_action_seed: dict[tuple[object, ...], bool] = {}
    for row in rows:
        target = str(row["target_center"])
        training_seed = int(row["training_seed"])
        generation_seed = int(row["generation_seed"])
        fold_id = str(row["fold_id"])
        action_id = str(row["action_id"])
        observed_keys.append(
            (target, training_seed, generation_seed, fold_id, action_id)
        )
        try:
            fold = next(
                value
                for value in crossfit.folds_by_target[target]
                if value.fold_id == fold_id
            )
            action = getattr(plan, "action")(target, action_id)
        except (KeyError, StopIteration) as exc:
            raise ProtocolError("Case-OOF prediction fold/action is unknown.") from exc
        candidates = candidate_sources(target)
        expected_streams = {
            source: str(
                source_rows[(source, training_seed, generation_seed)]["stream_id"]
            )
            for source in candidates
        }
        expected_experts = {
            source: str(
                source_rows[(source, training_seed, generation_seed)][
                    "expert_lock_hash"
                ]
            )
            for source in candidates
        }
        expected_shuffle = {
            str(label): derived_composition_seed(
                generation_lock_hash=generation_lock_hash,
                target_center=target,
                training_seed=training_seed,
                generation_seed=generation_seed,
                class_label=label,
            )
            for label in (0, 1)
        }
        core = action.core_action
        expected_windows = (
            _base_windows(candidates)
            if core is None
            else core.to_payload()["windows_by_class"]
        )
        composition_hash = str(row["composition_hash"])
        fit_key = (target, training_seed, generation_seed, composition_hash)
        distinct_fits.add(fit_key)
        aliased = _truthy(row["fit_aliased_by_composition_hash"])
        alias_key = (target, training_seed, generation_seed, action_id)
        previous_alias = aliases_by_action_seed.setdefault(alias_key, aliased)
        if previous_alias != aliased:
            raise ProtocolError("Case-OOF composition alias flag drifted by fold.")
        if (
            row["config_contract_hash"] != getattr(config, "contract_hash")
            or row["generation_lock_hash"] != generation_lock_hash
            or row["source_cache_lock_hash"] != source_cache_lock_hash
            or row["crossfit_fold_lock_hash"] != crossfit.lock_hash
            or row["router_plan_lock_hash"] != plan.lock_hash
            or int(row["fold_ordinal"]) != fold.fold_ordinal
            or str(row["heldout_case_id"]) != fold.heldout_case_id
            or str(row["fold_hash"]) != fold.fold_hash
            or str(row["action_hash"]) != action.action_hash
            or str(row["core_action_hash"])
            != ("" if core is None else core.action_hash)
            or str(row["action_role"]) != action.policy_id
            or _parse_json(row["candidate_sources_json"]) != list(candidates)
            or _parse_json(row["source_stream_ids_json"]) != expected_streams
            or _parse_json(row["expert_lock_hashes_json"]) != expected_experts
            or int(row["base_per_source"]) != 128
            or int(row["base_total_per_class"]) != 1024
            or int(row["topup_total_per_class"])
            != action.topup_total_per_class
            or int(row["final_total_per_class"])
            != action.final_total_per_class
            or _parse_json(row["topup_counts_json"])
            != dict(action.topup_counts_by_source)
            or _parse_json(row["final_counts_by_class_json"])
            != {
                str(label): dict(action.final_counts_by_class[label])
                for label in (0, 1)
            }
            or _parse_json(row["windows_by_class_json"]) != expected_windows
            or _parse_json(row["shuffle_seed_by_class_json"]) != expected_shuffle
            or not _is_sha256(row["allocation_hash"])
            or not _is_sha256(row["window_hash"])
            or not _is_sha256(composition_hash)
            or not _is_sha256(row["composition_output_sha256"])
            or row["classifier_config_hash"]
            != getattr(config, "classifier").config_hash
            or not _positive_integer_list(row["classifier_n_iter_json"])
            or not _truthy(row["classifier_converged"])
            or _parse_json(row["evaluation_row_ids_json"])
            != [str(item.sample_id) for item in fold.heldout_rows]
            or str(row["evaluation_row_identity_hash"])
            != fold.heldout_row_identity_hash
            or _truthy(row["labels_available_to_fit_or_predict"])
            or _truthy(row["support_labels_used"])
            or _truthy(row["evaluation_embeddings_used_for_route"])
            or _truthy(row["other_evaluation_embeddings_used_for_route"])
            or not _truthy(row["heldout_case_excluded_from_route"])
            or not _truthy(row["target_expert_excluded"])
            or _truthy(row["global_excludes_target_and_query"])
            != (action_id == GLOBAL_ACTION_ID)
            or _truthy(row["seed_selection_performed"])
            or _truthy(row["policy_selection_performed"])
            or _truthy(row["fallback_performed"])
            or str(row["claim_role"])
            != "terminal_consumed_validation_case_oof_diagnostic_only"
        ):
            raise ProtocolError("Case-OOF prediction binding drifted.")
    if observed_keys != expected_keys:
        raise ProtocolError("Case-OOF action/fold/seed order drifted.")
    if len(distinct_fits) != store.unique_classifier_fit_count:
        raise ProtocolError("Case-OOF unique classifier-fit accounting drifted.")


def _base_windows(sources: tuple[str, ...]) -> dict[str, object]:
    return {
        str(label): {
            source: {
                "base": [0, 128],
                "topup": [128, 128],
                "base_count": 128,
                "topup_count": 0,
                "required_capacity": 128,
            }
            for source in sources
        }
        for label in (0, 1)
    }


def _parse_json(value: object) -> object:
    try:
        return json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ProtocolError("Case-OOF prediction JSON field is invalid.") from exc


def _truthy(value: object) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _positive_integer_list(value: object) -> bool:
    parsed = _parse_json(value)
    return (
        isinstance(parsed, list)
        and bool(parsed)
        and all(
            isinstance(item, int)
            and not isinstance(item, bool)
            and item > 0
            for item in parsed
        )
    )


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


__all__ = ("validate_prediction_store_binding",)
