"""Shared runtime and identity helpers for prior-recovery experiments."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from ...real_features.classifier_reference.artifacts import stable_hash
from ...real_features.classifier_reference.classifiers import ClassifierSpec
from ...real_features.classifier_reference.midogpp_real_feature_classifier import (
    RealFeatureFrame,
    load_midogpp_real_feature_frame,
)
from ...real_features.classifier_reference.protocol import ProtocolError
from ..feature_frame import ExpertFeatureFrame
from ..generation_samplers import (
    AggregatePosteriorSampler,
    STANDARD_SAMPLER,
    fit_aggregate_posterior_sampler,
    standard_normal_sampler,
)
from ..training import (
    TrainedCVAERuntime,
    TrainingKey,
    TrainingVariant,
    train_cvae,
    training_variant_hash,
)
from .prior_recovery_config import (
    OuterPriorRecoveryConfig,
    PriorRecoveryConfig,
    outer_decision_contract_hash,
    recipe_contract_hash,
)
from .prior_recovery_provenance import ProvenanceRecorder
from .representations import encode_posterior
from .runtime import EvaluationKey, GenerationKey, SamplerFitKey
from .scoring import chance_normalized_preservation
from .splits import row_hash
from .tuned_reference import TunedClassifierSpec


PRIOR_RECOVERY_METHOD = "eligible_source_inner_locked_prior_recovery_factorial_v1"
NO_TASK_FISHER_STATE = "none"


def load_frame(config: PriorRecoveryConfig) -> RealFeatureFrame:
    frame = load_midogpp_real_feature_frame(
        manifest_path=config.manifest_path,
        feature_cache_path=config.feature_cache_path,
        expected_feature_dim=config.expected_feature_dim,
    )
    missing = set(config.heldout_centers).difference(frame.eligible_centers)
    if missing:
        raise ProtocolError(f"Heldout centers absent from feature frame: {sorted(missing)}")
    return frame


def protocol_hash(
    config: PriorRecoveryConfig,
    frame: RealFeatureFrame,
    *,
    reference_protocol_hash: str = "none",
    selection_bundle_hash: str = "none",
    source_inner_protocol_hash: str = "none",
    frozen_reference_identity_hash: str = "none",
) -> str:
    payload = {
        "schema_version": "midogpp_prior_recovery_runtime_protocol_v1",
        "name": config.name,
        "mode": config.mode,
        "recipe_contract_hash": recipe_contract_hash(config),
        "manifest_hash": frame.manifest_hash,
        "feature_cache_hash": frame.feature_cache_hash,
        "reference_protocol_hash": reference_protocol_hash,
    }
    if isinstance(config, OuterPriorRecoveryConfig):
        payload["outer_decision_contract_hash"] = outer_decision_contract_hash(config)
        payload["selection_bundle_hash"] = selection_bundle_hash
        payload["source_inner_protocol_hash"] = source_inner_protocol_hash
        payload["frozen_reference_identity_hash"] = frozen_reference_identity_hash
    return stable_hash(payload)


def train_runtime(
    config: PriorRecoveryConfig,
    *,
    variant: TrainingVariant,
    frame: ExpertFeatureFrame,
    fit_centers: Sequence[str],
    source_ids: Sequence[str],
    x_fit: object,
    y_fit: Sequence[int],
    training_seed: int,
    runtime_protocol_hash: str,
    feature_cache_hash: str,
    manifest_hash: str,
    task_metric: object | None,
    objective_context_hash: str,
    recorder: ProvenanceRecorder,
    task_fisher_state_hash: str,
    classifier_spec_hash: str,
) -> TrainedCVAERuntime:
    import numpy as np

    pairing_hash = stable_hash(
        {
            "fit_centers": list(fit_centers),
            "fit_row_hash": row_hash(source_ids),
            "training_seed": int(training_seed),
            "frame_hash": frame.state_hash,
            "dataset_contract_hash": manifest_hash,
            "feature_cache_hash": feature_cache_hash,
            "backbone_output_frame_id": "virchow2:full_to_pca",
            "recipe_contract_hash": recipe_contract_hash(config),
            "code_version": config.code_version,
            "paired_variant": variant.stochastic_pairing_payload(),
        }
    )
    key = TrainingKey(
        fit_centers=tuple(str(center) for center in fit_centers),
        fit_row_hash=row_hash(source_ids),
        objective_id=variant.objective_id,
        training_seed=int(training_seed),
        frame_hash=frame.state_hash,
        dataset_contract_hash=manifest_hash,
        feature_cache_hash=feature_cache_hash,
        backbone_output_frame_id="virchow2:full_to_pca",
        protocol_hash=runtime_protocol_hash,
        code_version=config.code_version,
        variant_hash=training_variant_hash(variant),
        stochastic_pairing_hash=pairing_hash,
        objective_context_hash=objective_context_hash,
    )
    input_dim = int(np.asarray(x_fit).shape[1])
    runtime = recorder.load_runtime(
        training_key=key,
        variant=variant,
        input_dim=input_dim,
        task_fisher_state_hash=task_fisher_state_hash,
        classifier_spec_hash=classifier_spec_hash,
        device=config.device,
    )
    if runtime is None:
        runtime = train_cvae(
            x_fit,
            y_fit,
            variant=variant,
            training_key=key,
            task_metric=task_metric,
            device=config.device,
        )
    recorder.record_runtime(
        runtime,
        task_fisher_state_hash=task_fisher_state_hash,
        classifier_spec_hash=classifier_spec_hash,
    )
    return runtime


def fit_samplers(
    config: PriorRecoveryConfig,
    *,
    runtime: TrainedCVAERuntime,
    x_fit: object,
    y_fit: Sequence[int],
    source_ids: Sequence[str],
    families: Sequence[str],
) -> dict[str, AggregatePosteriorSampler]:
    mu, logvar = encode_posterior(runtime, x_fit, y_fit)
    output: dict[str, AggregatePosteriorSampler] = {}
    source_hash = row_hash(source_ids)
    for family in dict.fromkeys(families):
        output[family] = (
            standard_normal_sampler(latent_dim=runtime.model.latent_dim, source_row_hash=source_hash)
            if family == STANDARD_SAMPLER
            else fit_aggregate_posterior_sampler(
                mu,
                logvar,
                y_fit,
                family=family,
                source_row_hash=source_hash,
                min_class_count=config.sampler_min_class_count,
                max_condition_number=config.sampler_max_condition_number,
            )
        )
    return output


def sampler_fit_key_hash(runtime: TrainedCVAERuntime, sampler: AggregatePosteriorSampler) -> str:
    return stable_hash(
        {
            str(class_label): SamplerFitKey(
                checkpoint_hash=runtime.checkpoint_hash,
                source_row_hash=sampler.source_row_hash,
                class_label=class_label,
                sampler_rule=sampler.requested_family,
            ).hash
            for class_label in sorted(sampler.classes)
        }
    )


def generation_and_evaluation_hashes(
    *,
    runtime: TrainedCVAERuntime,
    sampler: AggregatePosteriorSampler,
    generation_seed: int,
    labels: Sequence[int],
    representation_role: str,
    classifier_spec_hash: str,
    eval_center: str,
    eval_ids: Sequence[str],
    runtime_protocol_hash: str,
) -> tuple[str, str]:
    counts = (
        sum(int(value) == 0 for value in labels),
        sum(int(value) == 1 for value in labels),
    )
    source_state_hash = (
        sampler.state_hash
        if representation_role == "prior"
        else stable_hash(
            {
                "checkpoint_hash": runtime.checkpoint_hash,
                "fit_row_hash": runtime.training_key.fit_row_hash,
                "role": representation_role,
            }
        )
    )
    generated_hash = GenerationKey(
        source_state_hash=source_state_hash,
        generation_seed=int(generation_seed),
        class_count_vector=counts,
        representation_role=representation_role,
    ).hash
    evaluation_hash = EvaluationKey(
        generated_artifact_hash=generated_hash,
        frozen_classifier_spec_hash=classifier_spec_hash,
        eval_center=str(eval_center),
        eval_row_hash=row_hash(eval_ids),
        metric_schema_version="chance_corrected_bacc_preservation_v1",
        protocol_hash=runtime_protocol_hash,
    ).hash
    return generated_hash, evaluation_hash


def classifier_spec(spec: TunedClassifierSpec) -> ClassifierSpec:
    parsed = ClassifierSpec(
        C=spec.C,
        penalty=spec.penalty,
        solver=spec.solver,
        max_iter=spec.max_iter,
        class_weight=spec.class_weight,
        random_state=spec.random_state,
        l1_ratio=spec.l1_ratio,
        threshold_policy=spec.threshold_policy,
    )
    if parsed.config_hash != spec.config_hash:
        raise ProtocolError("Imported classifier spec hash does not match its payload.")
    return parsed


def safe_ratio(generated_bacc: float, real_bacc: float, *, minimum_real_bacc: float) -> float:
    try:
        return chance_normalized_preservation(
            generated_bacc,
            real_bacc,
            minimum_real_bacc=minimum_real_bacc,
        )
    except ValueError:
        return math.nan


def canonical_rows_hash(rows: Sequence[Mapping[str, object]]) -> str:
    canonical: list[dict[str, str]] = []
    for row in rows:
        payload = dict(row)
        payload.pop("selection_bundle_hash", None)
        canonical.append(_tabular_row(payload))
    return stable_hash(sorted(canonical, key=lambda item: stable_hash(item)))


def _tabular_row(row: Mapping[str, object]) -> dict[str, str]:
    """Normalize a row exactly as a CSV write/read round trip exposes it."""

    return {
        str(key): "" if value is None else str(value)
        for key, value in row.items()
    }


def _tabular_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, str]]:
    normalized = [_tabular_row(row) for row in rows]
    return sorted(normalized, key=lambda item: stable_hash(item))


def selection_evidence_hash(
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
) -> str:
    metrics = []
    for row in metric_rows:
        payload = dict(row)
        payload["selection_bundle_hash"] = ""
        metrics.append(payload)
    return stable_hash(
        {
            "metric_rows": _tabular_rows(metrics),
            "nested_reference_rows": _tabular_rows(nested_reference_rows),
            "nested_tuning_rows": _tabular_rows(nested_tuning_rows),
            "sampler_rows": _tabular_rows(sampler_rows),
            "identity_rows": _tabular_rows(identity_rows),
            "protocol_manifest": dict(protocol_manifest),
            "checkpoint_index": dict(checkpoint_index),
            "task_fisher_index": dict(task_fisher_index),
            "feature_frame_index": dict(feature_frame_index),
        }
    )


def mean(values: Sequence[float] | object) -> float:
    items = [float(value) for value in values]  # type: ignore[arg-type]
    if not items:
        return math.nan
    return sum(items) / float(len(items))
