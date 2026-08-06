"""Fail-closed configuration for the Uniform-B v2 GenerationLock."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from ...common.hashing import stable_hash
from ...real_features.classifier_reference.classifiers import ClassifierSpec
from ..expert_bank.uniform_b_v2_promotion.contracts import legal_routing_sources
from ..protocol import ProtocolError
from .contracts import (
    CENTERS,
    CLAIM_SCOPE,
    COMPOSITION_SHUFFLE_NAMESPACE,
    EXPECTED_BANK_INDEX_SHA256,
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_CONTENT_HASH,
    EXPECTED_CONTENT_INDEX_SHA256,
    EXPECTED_CONTROL_LOCK_HASH,
    EXPECTED_CONTROL_LOCK_SHA256,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_SEEDS,
    OUTPUT_ARTIFACT_ID,
    REPLICATE_POLICY,
    SAMPLER_FAMILY,
    SOURCE_BUDGET_PER_CLASS,
    SOURCE_STREAM_NAMESPACE,
    TOTAL_PER_CLASS,
    TRAINING_SEEDS,
)


@dataclass(frozen=True)
class UniformBV2GenerationLockConfig:
    experiment_id: str
    name: str
    artifact_root: Path
    bank_root: Path
    bank_artifact_id: str
    expected_bank_lock_hash: str
    expected_control_lock_hash: str
    expected_bank_index_sha256: str
    expected_control_sha256: str
    expected_content_index_sha256: str
    expected_content_hash: str
    generation_contract: Mapping[str, object]
    model: Mapping[str, object]
    source_frame: Mapping[str, object]
    aggregate_prior: Mapping[str, object]
    deterministic_rng: Mapping[str, object]
    classifier: ClassifierSpec
    health_probe: Mapping[str, object]
    runtime: Mapping[str, object]
    claim_boundary: Mapping[str, object]

    @property
    def contract_hash(self) -> str:
        """Hash only the path- and runtime-independent scientific contract."""

        return stable_hash(
            {
                "experiment_id": self.experiment_id,
                "bank_artifact_id": self.bank_artifact_id,
                "expected_bank_lock_hash": self.expected_bank_lock_hash,
                "expected_control_lock_hash": self.expected_control_lock_hash,
                "expected_bank_index_sha256": self.expected_bank_index_sha256,
                "expected_control_sha256": self.expected_control_sha256,
                "expected_content_index_sha256": self.expected_content_index_sha256,
                "expected_content_hash": self.expected_content_hash,
                "generation_contract": dict(self.generation_contract),
                "model": dict(self.model),
                "source_frame": dict(self.source_frame),
                "aggregate_prior": dict(self.aggregate_prior),
                "deterministic_rng": dict(self.deterministic_rng),
                "classifier": self.classifier.to_payload(),
                "claim_boundary": dict(self.claim_boundary),
            }
        )

    @property
    def centers(self) -> tuple[str, ...]:
        return _strings(self.generation_contract.get("centers"))

    @property
    def training_seeds(self) -> tuple[int, ...]:
        return _ints(self.generation_contract.get("training_seeds"))

    @property
    def generation_seeds(self) -> tuple[int, ...]:
        return _ints(self.generation_contract.get("generation_seeds"))

    @property
    def health_samples_per_class(self) -> int:
        return int(self.health_probe.get("samples_per_class", 0))

    @property
    def runtime_device(self) -> str:
        return str(self.runtime.get("default_device", "cpu"))


def load_generation_lock_config(path: str | Path) -> UniformBV2GenerationLockConfig:
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ProtocolError("Uniform-B v2 GenerationLock config must be a mapping.")
    experiment = _mapping(payload, "experiment")
    inputs = _mapping(payload, "inputs")
    classifier_raw = _mapping(payload, "classifier")
    base = config_path.parent
    classifier = ClassifierSpec(
        family=str(classifier_raw.get("family", "")),
        C=float(classifier_raw.get("C", 0.0)),
        penalty=str(classifier_raw.get("penalty", "")),
        solver=str(classifier_raw.get("solver", "")),
        max_iter=int(classifier_raw.get("max_iter", 0)),
        class_weight=(
            None
            if classifier_raw.get("class_weight") is None
            else str(classifier_raw["class_weight"])
        ),
        random_state=int(classifier_raw.get("random_state", -1)),
        threshold_policy=str(classifier_raw.get("threshold_policy", "")),
        scaler_fit=str(classifier_raw.get("scaler_fit", "")),
    )
    config = UniformBV2GenerationLockConfig(
        experiment_id=str(experiment.get("id", "")),
        name=str(experiment.get("name", "")),
        artifact_root=_path(base, experiment.get("artifact_root")),
        bank_root=_path(base, inputs.get("bank_root")),
        bank_artifact_id=str(inputs.get("bank_artifact_id", "")),
        expected_bank_lock_hash=str(inputs.get("expected_bank_lock_hash", "")),
        expected_control_lock_hash=str(inputs.get("expected_control_lock_hash", "")),
        expected_bank_index_sha256=str(inputs.get("expected_bank_index_sha256", "")),
        expected_control_sha256=str(inputs.get("expected_control_sha256", "")),
        expected_content_index_sha256=str(inputs.get("expected_content_index_sha256", "")),
        expected_content_hash=str(inputs.get("expected_content_hash", "")),
        generation_contract=dict(_mapping(payload, "generation_contract")),
        model=dict(_mapping(payload, "model")),
        source_frame=dict(_mapping(payload, "source_frame")),
        aggregate_prior=dict(_mapping(payload, "aggregate_prior")),
        deterministic_rng=dict(_mapping(payload, "deterministic_rng")),
        classifier=classifier,
        health_probe=dict(_mapping(payload, "health_probe")),
        runtime=dict(_mapping(payload, "runtime")),
        claim_boundary=dict(_mapping(payload, "claim_boundary")),
    )
    _validate(config, classifier_raw=classifier_raw)
    return config


def _validate(
    config: UniformBV2GenerationLockConfig,
    *,
    classifier_raw: Mapping[str, object],
) -> None:
    exact = {
        "experiment_id": (config.experiment_id, EXPERIMENT_ID),
        "name": (config.name, EXPERIMENT_NAME),
        "bank_artifact_id": (config.bank_artifact_id, EXPERT_BANK_ARTIFACT_ID),
        "expected_bank_lock_hash": (config.expected_bank_lock_hash, EXPECTED_BANK_LOCK_HASH),
        "expected_control_lock_hash": (
            config.expected_control_lock_hash,
            EXPECTED_CONTROL_LOCK_HASH,
        ),
        "expected_bank_index_sha256": (
            config.expected_bank_index_sha256,
            EXPECTED_BANK_INDEX_SHA256,
        ),
        "expected_control_sha256": (
            config.expected_control_sha256,
            EXPECTED_CONTROL_LOCK_SHA256,
        ),
        "expected_content_index_sha256": (
            config.expected_content_index_sha256,
            EXPECTED_CONTENT_INDEX_SHA256,
        ),
        "expected_content_hash": (config.expected_content_hash, EXPECTED_CONTENT_HASH),
        "centers": (config.centers, CENTERS),
        "training_seeds": (config.training_seeds, TRAINING_SEEDS),
        "generation_seeds": (config.generation_seeds, GENERATION_SEEDS),
    }
    mismatch = [
        f"{key}: observed={observed!r}, expected={expected!r}"
        for key, (observed, expected) in exact.items()
        if observed != expected
    ]
    if mismatch:
        raise ProtocolError("Uniform-B v2 GenerationLock identity drifted: " + "; ".join(mismatch))

    generation = config.generation_contract
    expected_generation = {
        "class_labels": [0, 1],
        "class_budget_policy": "balanced_equal_per_class",
        "seed_pairing": "cartesian_product",
        "replicate_policy": REPLICATE_POLICY,
        "training_seed_policy": "use_matching_frozen_expert_per_replicate",
        "generation_seed_policy": "generate_every_declared_seed_per_training_seed",
        "total_per_class": TOTAL_PER_CLASS,
        "sources_per_target": 8,
        "source_budget_per_class": SOURCE_BUDGET_PER_CLASS,
        "budget_applies_independently_per_replicate": True,
        "source_budgets_split_across_seeds": False,
        "target_expert_excluded": True,
        "target_conditioned_source_weighting": False,
        "source_weighting": "equal_fixed_count",
        "no_expert_selection": True,
        "no_seed_selection": True,
        "expected_source_plan_rows": 81,
        "expected_target_replicate_rows": 81,
        "candidate_sources_by_target": {
            target: list(legal_routing_sources(target)) for target in CENTERS
        },
    }
    _require_values(generation, expected_generation, "generation contract")
    _require_values(
        config.model,
        {
            "family": "conditional_variational_autoencoder",
            "input_dim": 256,
            "hidden_dim": 1024,
            "latent_dim": 64,
            "num_hidden_layers": 3,
            "class_conditioning_dim": 2,
            "frozen_checkpoint_required": True,
            "training_allowed": False,
        },
        "model",
    )
    _require_values(
        config.source_frame,
        {
            "family": "source_specific_pca",
            "model_space_dim": 256,
            "reconstructed_embedding_dim": 3840,
            "inverse_transform_required": True,
            "one_frame_per_source_center": True,
            "fit_scope": "source_center_rows_only",
            "refit_allowed": False,
        },
        "source frame",
    )
    _require_values(
        config.aggregate_prior,
        {
            "family": SAMPLER_FAMILY,
            "covariance": "full",
            "class_conditional": True,
            "fit_scope": "source_center_rows_only",
            "min_class_count": 64,
            "max_condition_number": 1_000_000.0,
            "partial_class_fallback_allowed": False,
            "refit_allowed": False,
        },
        "aggregate prior",
    )
    namespaces = _mapping(config.deterministic_rng, "namespaces")
    _require_values(
        config.deterministic_rng,
        {
            "seed_derivation": "sha256_first_unsigned_64_bit",
            "global_rng_forbidden": True,
            "latent_draw_key_fields": [
                "namespace",
                "bank_lock_hash",
                "expert_lock_hash",
                "generation_seed",
                "class_label",
            ],
            "equal_union_shuffle_key_fields": [
                "namespace",
                "generation_lock_hash",
                "target_center",
                "training_seed",
                "generation_seed",
                "class_label",
            ],
        },
        "deterministic RNG",
    )
    _require_values(
        namespaces,
        {
            "latent_draw": SOURCE_STREAM_NAMESPACE,
            "equal_union_shuffle": COMPOSITION_SHUFFLE_NAMESPACE,
            "health_probe": "uniform_b_v2_generation_lock.health_probe.v1",
        },
        "deterministic RNG namespaces",
    )
    expected_classifier = ClassifierSpec(
        C=0.01,
        penalty="l2",
        solver="lbfgs",
        max_iter=3000,
        class_weight=None,
        random_state=23,
        threshold_policy="predict",
        scaler_fit="synthetic_train_only",
    )
    if config.classifier != expected_classifier:
        raise ProtocolError("Uniform-B v2 GenerationLock classifier drifted.")
    _require_values(
        classifier_raw,
        {
            "scaler": "sklearn.preprocessing.StandardScaler",
            "fit_in_stage_40": False,
        },
        "classifier execution contract",
    )
    _require_values(
        config.health_probe,
        {
            "samples_per_class": 1,
            "scope": "every_source_training_seed_generation_seed_and_class",
            "expected_rows": 162,
            "require_finite_latents": True,
            "require_finite_decoder_outputs": True,
            "require_finite_reconstructed_embeddings": True,
            "require_model_space_shape": [1, 256],
            "require_reconstructed_embedding_shape": [1, 3840],
        },
        "health probe",
    )
    _require_values(
        config.runtime,
        {
            "one_expert_in_memory_at_a_time": True,
            "no_training": True,
            "no_sampler_refit": True,
            "no_frame_refit": True,
            "no_classifier_fit": True,
        },
        "runtime",
    )
    _require_values(
        config.claim_boundary,
        {
            "strict_claim_firewall": True,
            "claim_scope": CLAIM_SCOPE,
            "lock_only": True,
            "may_feed_deployable_selection": True,
            "source_only_frozen_state": True,
            "target_data_used": False,
            "target_support_used": False,
            "target_labels_used": False,
            "target_evaluation_labels_used": False,
            "routing_evidence_computed": False,
            "routing_quality_claimed": False,
            "nelbo_computed": False,
            "expert_selection_performed": False,
            "source_weighting_learned": False,
            "classifier_fit_performed": False,
            "downstream_utility_computed": False,
            "stage20_bacc_reused_as_stage40_result": False,
            "eight_source_control_scored": False,
        },
        "claim boundary",
    )
    if config.runtime_device != "cpu" and not config.runtime_device.startswith("cuda:"):
        raise ProtocolError("GenerationLock runtime device must be cpu or explicit cuda:N.")
    health_device = str(config.health_probe.get("device", ""))
    if health_device != config.runtime_device:
        raise ProtocolError("GenerationLock health and runtime devices must match.")
    if str(config.artifact_root).startswith("output:") and config.artifact_root.name != OUTPUT_ARTIFACT_ID:
        raise ProtocolError("Unexpected Uniform-B v2 GenerationLock output identity.")


def _require_values(
    observed: Mapping[str, object],
    expected: Mapping[str, object],
    label: str,
) -> None:
    mismatch = [
        f"{key}: observed={observed.get(key)!r}, expected={value!r}"
        for key, value in expected.items()
        if observed.get(key) != value
    ]
    if mismatch:
        raise ProtocolError(f"Uniform-B v2 {label} drifted: " + "; ".join(mismatch))


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"GenerationLock config section {key!r} must be a mapping.")
    return value


def _ints(value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ProtocolError("GenerationLock config expected an integer list.")
    return tuple(int(item) for item in value)


def _strings(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ProtocolError("GenerationLock config expected a string list.")
    return tuple(str(item) for item in value)


def _path(base: Path, value: object) -> Path:
    rendered = str(value or "")
    if not rendered:
        raise ProtocolError("GenerationLock config path is empty.")
    if rendered.startswith(("artifact://", "output://")):
        return Path(rendered)
    path = Path(rendered)
    return path if path.is_absolute() else (base / path).resolve()


__all__ = ("UniformBV2GenerationLockConfig", "load_generation_lock_config")
