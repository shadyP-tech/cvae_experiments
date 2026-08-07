"""Fast, label-free binding validation for materialized prediction cells."""

from __future__ import annotations

import json
from typing import Mapping

from ....common.hashing import stable_hash
from ...generation.generation import derived_composition_seed
from ...protocol import ProtocolError
from .contracts import (
    CENTERS,
    DEVELOPMENT_ACTION_IDS,
    EXPECTED_PREDICTION_CELL_COUNT,
    GENERATION_SEEDS,
    TARGET_ACTION_IDS,
    TRAINING_SEEDS,
    development_queries,
)
from .prediction_store import PredictionStore


def validate_prediction_store_binding(
    store: PredictionStore,
    *,
    config: object,
    generation_lock_hash: str,
    source_cache: object,
    source_cache_lock_hash: str,
    plans: object,
    partitions: object,
) -> None:
    """Bind every persisted cell to immutable plans, streams, and unlabeled rows."""

    if (
        len(store.index_rows) != EXPECTED_PREDICTION_CELL_COUNT
        or store.unique_classifier_fit_count != EXPECTED_PREDICTION_CELL_COUNT
    ):
        raise ProtocolError("Residual top-up prediction binding coverage drifted.")
    evaluation_by_center = getattr(partitions, "evaluation_rows_by_center")
    cache_rows = {
        (
            str(item["source_center"]),
            int(item["training_seed"]),
            int(item["generation_seed"]),
        ): item
        for item in getattr(source_cache, "index_rows")
    }
    expected_keys = []
    for outer in CENTERS:
        for query in development_queries(outer):
            for training_seed in TRAINING_SEEDS:
                for generation_seed in GENERATION_SEEDS:
                    for action in DEVELOPMENT_ACTION_IDS:
                        expected_keys.append(("development", outer, query, training_seed, generation_seed, action))
    for target in CENTERS:
        for training_seed in TRAINING_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                for action in TARGET_ACTION_IDS:
                    expected_keys.append(("target", target, target, training_seed, generation_seed, action))
    observed = []
    for row in store.index_rows:
        phase = str(row["phase"])
        outer = str(row["outer_target"])
        query = str(row["query_center"])
        action = str(row["action_id"])
        training_seed = int(row["training_seed"])
        generation_seed = int(row["generation_seed"])
        plan = plans.plan(
            phase=phase, outer_target=outer, query_center=query, action_id=action
        )
        payload = plan.get("action_payload")
        if not isinstance(payload, Mapping):
            raise ProtocolError("Residual top-up prediction plan payload drifted.")
        geometry = payload.get("geometry")
        if not isinstance(geometry, Mapping):
            raise ProtocolError("Residual top-up prediction geometry drifted.")
        candidates = tuple(str(value) for value in plan["candidate_sources"])
        try:
            expected_streams = {
                source: str(cache_rows[(source, training_seed, generation_seed)]["stream_id"])
                for source in candidates
            }
            expected_experts = {
                source: str(cache_rows[(source, training_seed, generation_seed)]["expert_lock_hash"])
                for source in candidates
            }
        except KeyError as exc:
            raise ProtocolError("Residual top-up prediction source grid drifted.") from exc
        expected_shuffle = {
            str(label): derived_composition_seed(
                generation_lock_hash=generation_lock_hash,
                target_center=query,
                training_seed=training_seed,
                generation_seed=generation_seed,
                class_label=label,
            )
            for label in (0, 1)
        }
        expected_rows = tuple(evaluation_by_center[query])
        observed.append((phase, outer, query, training_seed, generation_seed, action))
        if (
            row["config_contract_hash"] != getattr(config, "contract_hash")
            or row["generation_lock_hash"] != generation_lock_hash
            or row["source_cache_lock_hash"] != source_cache_lock_hash
            or row["router_plan_lock_hash"] != plans.lock_hash
            or row["plan_hash"] != plan["plan_hash"]
            or row["classifier_config_hash"] != getattr(config, "classifier").config_hash
            or str(row["arm_role"]) != str(plan["arm_role"])
            or str(row["budget_role"]) != str(plan["budget_role"])
            or _parse_json(row["candidate_sources_json"]) != list(candidates)
            or _parse_json(row["source_stream_ids_json"]) != expected_streams
            or _parse_json(row["expert_lock_hashes_json"]) != expected_experts
            or int(row["base_per_source"]) != int(geometry["base_per_source"])
            or int(row["base_total_per_class"]) != int(geometry["base_total_per_class"])
            or int(row["topup_total_per_class"]) != int(geometry["topup_total_per_class"])
            or int(row["final_total_per_class"]) != int(geometry["final_total_per_class"])
            or _parse_json(row["topup_counts_json"]) != payload["topup_counts"]
            or _parse_json(row["final_counts_by_class_json"]) != payload["final_counts_by_class"]
            or _parse_json(row["final_weights_by_class_json"]) != payload["final_weights_by_class"]
            or _parse_json(row["windows_by_class_json"]) != payload["windows_by_class"]
            or _parse_json(row["shuffle_seed_by_class_json"]) != expected_shuffle
            or str(row["action_hash"]) != str(payload["action_hash"])
            or str(row["allocation_hash"]) != str(payload["allocation_hash"])
            or str(row["window_hash"]) != str(payload["window_hash"])
            or len(str(row["composition_hash"])) != 64
            or len(str(row["composition_output_sha256"])) != 64
            or not _positive_integer_list(row["classifier_n_iter_json"])
            or _parse_json(row["evaluation_row_ids_json"]) != [item.sample_id for item in expected_rows]
            or row["evaluation_row_identity_hash"] != stable_hash([item.identity_payload() for item in expected_rows])
            or not _truthy(row["classifier_converged"])
            or _truthy(row["labels_available_to_fit_or_predict"])
            or _truthy(row["support_labels_used"])
            or _truthy(row["seed_selection_performed"])
            or not _truthy(row["target_expert_excluded"])
            or _truthy(row["outer_and_query_experts_excluded"]) != (phase == "development")
            or _truthy(row["fit_aliased_by_composition_hash"])
            or str(row["selection_source"]) != str(plan["selection_source"])
            or str(row["claim_role"]) != str(plan["claim_role"])
        ):
            raise ProtocolError("Residual top-up prediction binding drifted.")
    if observed != expected_keys:
        raise ProtocolError("Residual top-up prediction cell order drifted.")


def _parse_json(value: object) -> object:
    return json.loads(str(value))


def _truthy(value: object) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _positive_integer_list(value: object) -> bool:
    parsed = _parse_json(value)
    return (
        isinstance(parsed, list)
        and bool(parsed)
        and all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in parsed)
    )


__all__ = ("validate_prediction_store_binding",)
