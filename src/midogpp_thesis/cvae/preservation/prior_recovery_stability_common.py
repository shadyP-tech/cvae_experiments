"""Identity helpers shared by the stability runner and validator."""

from __future__ import annotations

from typing import Mapping, Sequence

from ...real_features.classifier_reference.artifacts import stable_hash
from ...real_features.classifier_reference.midogpp_real_feature_classifier import (
    RealFeatureFrame,
)
from ...real_features.classifier_reference.protocol import ProtocolError
from ...real_features.classifier_reference.schemas.midogpp import (
    MIDOGPP_ELIGIBLE_CENTERS,
    MIDOGPP_EXCLUDED_CENTERS,
)
from ..objectives import ISOTROPIC_OBJECTIVE
from .prior_recovery_classifier import (
    SOURCE_INNER_CLASSIFIER_GRID_HASH,
    source_inner_classifier_specs,
)
from .prior_recovery_common import (
    PRIOR_RECOVERY_METHOD,
    canonical_rows_hash,
    protocol_hash,
    selection_evidence_hash,
)
from .prior_recovery_config import (
    SourceInnerPriorRecoveryConfig,
    SourceInnerStabilityConfig,
    recipe_contract_hash,
    recipe_contract_payload,
    stability_contract_hash,
    stability_contract_payload,
)
from .prior_recovery_schema import STABILITY_PROTOCOL_SCHEMA


RNG_PAIRING_AUDIT_SCHEMA = "midogpp_prior_recovery_rng_pairing_audit_v1"
RNG_PAIRING_AUDIT_COLUMNS = (
    "schema_version",
    "outer_target_center",
    "inner_pseudo_target_center",
    "training_seed",
    "arm",
    "objective_id",
    "sampler_family",
    "representation_role",
    "generation_seed",
    "generation_class_counts",
    "training_key_hash",
    "initialization_hash",
    "stochastic_stream_hash",
    "training_rng_namespace_hash",
    "generation_noise_engine",
    "generation_noise_namespace_hash",
    "status",
)


def stability_runtime_protocol_hash(
    config: SourceInnerStabilityConfig,
    frame: RealFeatureFrame,
) -> str:
    return stable_hash(
        {
            "schema_version": "midogpp_prior_recovery_stability_runtime_protocol_v1",
            "name": config.name,
            "mode": config.mode,
            "stability_contract_hash": stability_contract_hash(config),
            "manifest_hash": frame.manifest_hash,
            "feature_cache_hash": frame.feature_cache_hash,
        }
    )


def child_source_inner_protocol(
    config: SourceInnerPriorRecoveryConfig,
    frame: RealFeatureFrame,
) -> dict[str, object]:
    specs = source_inner_classifier_specs(classifier_seed=23)
    runtime_hash = protocol_hash(config, frame)
    return {
        "schema_version": "midogpp_prior_recovery_source_inner_protocol_v1",
        "experiment_name": config.name,
        "method": PRIOR_RECOVERY_METHOD,
        "claim_scope": "cvae_recipe_lock_only",
        "claim_role": "cvae_recipe_lock",
        "heldout_centers": list(config.heldout_centers),
        "eligible_centers": list(frame.eligible_centers),
        "coverage_mode": (
            "complete"
            if config.heldout_centers
            == frame.eligible_centers
            == MIDOGPP_ELIGIBLE_CENTERS
            else "partial_test"
        ),
        "excluded_centers": list(MIDOGPP_EXCLUDED_CENTERS),
        "classifier_grid_hash": SOURCE_INNER_CLASSIFIER_GRID_HASH,
        "classifier_grid": [spec.to_payload() for spec in specs],
        "manifest_hash": frame.manifest_hash,
        "feature_cache_hash": frame.feature_cache_hash,
        "protocol_hash": runtime_hash,
        "recipe_contract": recipe_contract_payload(config),
        "recipe_contract_hash": recipe_contract_hash(config),
        "source_inner_labels_used_for_selection": True,
        "outer_target_rows_passed_to_training_or_selection": False,
        "target_eval_labels_used_for_selection": False,
        "target_eval_labels_used_for_scoring_only": False,
        "support_labels_used": False,
        "oracle_eligible": False,
        "may_feed_model_recipe": True,
        "may_feed_deployable_selection": False,
        "routing_performed": False,
        "composition_performed": False,
        "query_object": "none",
    }


def parent_stability_protocol(
    config: SourceInnerStabilityConfig,
    frame: RealFeatureFrame,
    *,
    child_protocols: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": STABILITY_PROTOCOL_SCHEMA,
        "experiment_name": config.name,
        "mode": config.mode,
        "method": PRIOR_RECOVERY_METHOD,
        "claim_scope": "cvae_recipe_lock_only",
        "claim_role": "training_seed_stability_recipe_lock",
        "heldout_centers": list(config.heldout_centers),
        "eligible_centers": list(frame.eligible_centers),
        "coverage_mode": (
            "complete"
            if config.heldout_centers
            == frame.eligible_centers
            == MIDOGPP_ELIGIBLE_CENTERS
            else "partial_test"
        ),
        "excluded_centers": list(MIDOGPP_EXCLUDED_CENTERS),
        "manifest_hash": frame.manifest_hash,
        "feature_cache_hash": frame.feature_cache_hash,
        "protocol_hash": stability_runtime_protocol_hash(config, frame),
        "stability_contract": stability_contract_payload(config),
        "stability_contract_hash": stability_contract_hash(config),
        "training_seeds": list(config.training_seeds),
        "generation_seeds": list(config.generation_seeds),
        "consensus_rule_id": config.consensus_rule_id,
        "child_protocol_hashes": {
            seed: str(protocol["protocol_hash"])
            for seed, protocol in child_protocols.items()
        },
        "child_recipe_contract_hashes": {
            seed: str(protocol["recipe_contract_hash"])
            for seed, protocol in child_protocols.items()
        },
        "deterministic_evidence_shared_across_training_seeds": True,
        "generation_noise_paired_by_generation_seed": True,
        "posterior_noise_paired_by_generation_seed": True,
        "training_rng_varied_only_by_training_seed": True,
        "generation_budget_policy": "source_count_per_class_no_rebalancing",
        "task_fisher_state_policy": "one_shared_state_per_outer_inner_fold",
        "source_inner_labels_used_for_selection": True,
        "outer_target_rows_passed_to_training_or_selection": False,
        "target_eval_labels_used_for_selection": False,
        "target_eval_labels_used_for_scoring_only": False,
        "support_labels_used": False,
        "oracle_eligible": False,
        "may_feed_model_recipe": True,
        "may_feed_deployable_selection": False,
        "routing_performed": False,
        "composition_performed": False,
        "query_object": "none",
    }


def filter_checkpoint_index(
    index: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    keys = {str(row.get("training_key_hash", "")) for row in rows}
    records = [
        dict(record)
        for record in _records(index)
        if str(record.get("training_key_hash", "")) in keys
    ]
    return {
        "schema_version": index.get("schema_version"),
        "n_unique_checkpoints": len(records),
        "records": records,
    }


def filter_task_fisher_index(
    index: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    keys = {
        str(row.get("task_fisher_state_hash", ""))
        for row in rows
        if str(row.get("task_fisher_state_hash", "")) not in {"", "none"}
    }
    records = [
        dict(record)
        for record in _records(index)
        if str(record.get("task_fisher_state_hash", "")) in keys
    ]
    return {
        "schema_version": index.get("schema_version"),
        "n_unique_states": len(records),
        "records": records,
    }


def seed_selection_evidence_hash(
    *,
    metric_rows: Sequence[Mapping[str, object]],
    nested_reference_rows: Sequence[Mapping[str, object]],
    nested_tuning_rows: Sequence[Mapping[str, object]],
    sampler_rows: Sequence[Mapping[str, object]],
    identity_rows: Sequence[Mapping[str, object]],
    child_protocol: Mapping[str, object],
    checkpoint_index: Mapping[str, object],
    task_fisher_index: Mapping[str, object],
    feature_frame_index: Mapping[str, object],
    rng_audit_rows: Sequence[Mapping[str, object]],
) -> str:
    base_hash = selection_evidence_hash(
        metric_rows=metric_rows,
        nested_reference_rows=nested_reference_rows,
        nested_tuning_rows=nested_tuning_rows,
        sampler_rows=sampler_rows,
        identity_rows=identity_rows,
        protocol_manifest=child_protocol,
        checkpoint_index=filter_checkpoint_index(checkpoint_index, metric_rows),
        task_fisher_index=filter_task_fisher_index(task_fisher_index, metric_rows),
        feature_frame_index=feature_frame_index,
    )
    return stable_hash(
        {
            "schema_version": "midogpp_prior_recovery_seed_evidence_with_rng_v1",
            "base_selection_evidence_hash": base_hash,
            "rng_pairing_audit_hash": canonical_rows_hash(rng_audit_rows),
        }
    )


def stability_selection_evidence_hash(
    *,
    metric_rows: Sequence[Mapping[str, object]],
    nested_reference_rows: Sequence[Mapping[str, object]],
    nested_tuning_rows: Sequence[Mapping[str, object]],
    sampler_rows: Sequence[Mapping[str, object]],
    identity_rows: Sequence[Mapping[str, object]],
    protocol_manifest: Mapping[str, object],
    checkpoint_index: Mapping[str, object],
    task_fisher_index: Mapping[str, object],
    feature_frame_index: Mapping[str, object],
    rng_audit_rows: Sequence[Mapping[str, object]],
) -> str:
    """Bind the parent selection identity to the recomputable RNG audit."""

    base_hash = selection_evidence_hash(
        metric_rows=metric_rows,
        nested_reference_rows=nested_reference_rows,
        nested_tuning_rows=nested_tuning_rows,
        sampler_rows=sampler_rows,
        identity_rows=identity_rows,
        protocol_manifest=protocol_manifest,
        checkpoint_index=checkpoint_index,
        task_fisher_index=task_fisher_index,
        feature_frame_index=feature_frame_index,
    )
    return stable_hash(
        {
            "schema_version": "midogpp_prior_recovery_stability_evidence_with_rng_v1",
            "base_selection_evidence_hash": base_hash,
            "rng_pairing_audit_hash": canonical_rows_hash(rng_audit_rows),
        }
    )


def derive_rng_pairing_audit(
    metric_rows: Sequence[Mapping[str, object]],
    *,
    checkpoint_index: Mapping[str, object],
) -> list[dict[str, object]]:
    """Materialize the training and evaluation RNG namespaces used by each row."""

    records: dict[str, Mapping[str, object]] = {}
    for record in _records(checkpoint_index):
        key = str(record.get("training_key_hash", ""))
        if not key or key in records:
            raise ProtocolError("Checkpoint index has ambiguous training RNG identity.")
        records[key] = record
    result: list[dict[str, object]] = []
    for metric in metric_rows:
        key = str(metric.get("training_key_hash", ""))
        record = records.get(key)
        training_key = record.get("training_key") if record is not None else None
        if not isinstance(record, Mapping) or not isinstance(training_key, Mapping):
            raise ProtocolError("Metric row has no checkpoint-backed RNG identity.")
        training_seed = int(metric["training_seed"])
        if int(training_key.get("training_seed", -1)) != training_seed:
            raise ProtocolError("Metric and checkpoint training seeds disagree.")
        role = str(metric["representation_role"])
        generation_seed = int(metric["generation_seed"])
        if role == "decode":
            engine = "deterministic_decode"
            status = "DETERMINISTIC"
        elif role == "posterior":
            engine = "torch.Generator.manual_seed"
            status = "PAIRED_BY_GENERATION_SEED"
        elif role == "prior":
            engine = "numpy.random.default_rng"
            status = "PAIRED_BY_GENERATION_SEED"
        else:
            raise ProtocolError(f"Unsupported representation RNG role: {role!r}.")
        noise_namespace = stable_hash(
            {
                "schema_version": "midogpp_prior_recovery_generation_noise_namespace_v1",
                "outer_target_center": str(metric["outer_target_center"]),
                "inner_pseudo_target_center": str(metric["inner_pseudo_target_center"]),
                "representation_role": role,
                "generation_seed": generation_seed,
                "generation_class_counts": str(metric["generation_class_counts"]),
                "engine": engine,
            }
        )
        result.append(
            {
                "schema_version": RNG_PAIRING_AUDIT_SCHEMA,
                "outer_target_center": str(metric["outer_target_center"]),
                "inner_pseudo_target_center": str(metric["inner_pseudo_target_center"]),
                "training_seed": training_seed,
                "arm": str(metric["arm"]),
                "objective_id": str(metric["objective_id"]),
                "sampler_family": str(metric["sampler_family"]),
                "representation_role": role,
                "generation_seed": generation_seed,
                "generation_class_counts": str(metric["generation_class_counts"]),
                "training_key_hash": key,
                "initialization_hash": str(record["initialization_hash"]),
                "stochastic_stream_hash": str(record["stochastic_stream_hash"]),
                "training_rng_namespace_hash": str(record["stochastic_pairing_hash"]),
                "generation_noise_engine": engine,
                "generation_noise_namespace_hash": noise_namespace,
                "status": status,
            }
        )
    return result


def validate_rng_pairing_audit(
    observed_rows: Sequence[Mapping[str, object]],
    *,
    metric_rows: Sequence[Mapping[str, object]],
    checkpoint_index: Mapping[str, object],
    training_seeds: Sequence[int],
) -> None:
    """Recompute pairing identities and prove varied training/common eval RNGs."""

    expected = derive_rng_pairing_audit(
        metric_rows,
        checkpoint_index=checkpoint_index,
    )
    if canonical_rows_hash(observed_rows) != canonical_rows_hash(expected):
        raise ProtocolError("RNG pairing audit does not recompute from metric provenance.")
    expected_seeds = {int(seed) for seed in training_seeds}
    training_by_fold: dict[
        tuple[str, str], dict[int, set[tuple[str, str]]]
    ] = {}
    noise_by_cell: dict[tuple[str, str, str, int], dict[str, set[object]]] = {}
    for row in observed_rows:
        if str(row["objective_id"]) != ISOTROPIC_OBJECTIVE:
            continue
        fold = (
            str(row["outer_target_center"]),
            str(row["inner_pseudo_target_center"]),
        )
        seed = int(row["training_seed"])
        training_by_fold.setdefault(fold, {}).setdefault(seed, set()).add(
            (
                str(row["initialization_hash"]),
                str(row["stochastic_stream_hash"]),
            )
        )
        role = str(row["representation_role"])
        if role not in {"prior", "posterior"}:
            continue
        cell = (*fold, role, int(row["generation_seed"]))
        grouped = noise_by_cell.setdefault(
            cell,
            {"seeds": set(), "namespaces": set()},
        )
        grouped["seeds"].add(seed)
        grouped["namespaces"].add(str(row["generation_noise_namespace_hash"]))
    for seed_map in training_by_fold.values():
        identities = {
            next(iter(values))
            for values in seed_map.values()
            if len(values) == 1
        }
        initialization_hashes = {identity[0] for identity in identities}
        stochastic_stream_hashes = {identity[1] for identity in identities}
        if (
            set(seed_map) != expected_seeds
            or any(len(values) != 1 for values in seed_map.values())
            or len(identities) != len(expected_seeds)
            or len(initialization_hashes) != len(expected_seeds)
            or len(stochastic_stream_hashes) != len(expected_seeds)
        ):
            raise ProtocolError("Training RNG identities are not distinct across seeds.")
    if (
        not training_by_fold
        or not noise_by_cell
        or any(
            grouped["seeds"] != expected_seeds
            or len(grouped["namespaces"]) != 1
            for grouped in noise_by_cell.values()
        )
    ):
        raise ProtocolError("Generation/posterior noise is not paired by generation seed.")


def _records(index: Mapping[str, object]) -> list[Mapping[str, object]]:
    records = index.get("records")
    if not isinstance(records, list) or not all(
        isinstance(record, Mapping) for record in records
    ):
        return []
    return list(records)
