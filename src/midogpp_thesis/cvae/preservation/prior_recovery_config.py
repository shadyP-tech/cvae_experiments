"""Configuration contracts for source-inner prior recovery and outer preservation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

from ...real_features.classifier_reference.artifacts import stable_hash
from ...real_features.classifier_reference.protocol import ProtocolError
from ...real_features.classifier_reference.schemas.midogpp import MIDOGPP_ELIGIBLE_CENTERS
from ..generation_samplers import DIAGONAL_SAMPLER, FULL_SAMPLER, STANDARD_SAMPLER
from ..objectives import ISOTROPIC_OBJECTIVE, TASK_FISHER_OBJECTIVE
from ..training import TrainingVariant
from .prior_recovery_classifier import SOURCE_INNER_CLASSIFIER_GRID_HASH


SOURCE_INNER_EXPERIMENT = "virchow2_cvae_midogpp_prior_recovery_source_inner_v1"
SOURCE_INNER_STABILITY_EXPERIMENT = (
    "virchow2_cvae_midogpp_prior_recovery_source_inner_training_seed_stability_v1"
)
OUTER_EXPERIMENT = "virchow2_cvae_midogpp_prior_recovery_outer_v1"
SAMPLER_FALLBACK_POLICY = "full_to_diagonal_to_standard_per_class"
SAMPLER_VIABILITY_POLICY = "require_requested_family_both_classes"
STABILITY_TRAINING_SEEDS = (17, 42, 101)
STABILITY_CONSENSUS_RULE = (
    "unanimous_conditional_family_d_only_if_all_d_else_c_v1"
)


@dataclass(frozen=True)
class PriorRecoveryConfig:
    name: str
    artifact_root: Path
    manifest_path: Path
    feature_cache_path: Path
    heldout_centers: tuple[str, ...]
    expected_feature_dim: int
    pca_dim: int
    selection_training_seed: int
    generation_seeds: tuple[int, ...]
    device: str
    isotropic_variant: TrainingVariant
    task_fisher_variant: TrainingVariant
    sampler_min_class_count: int
    sampler_max_condition_number: float
    sampler_families: tuple[str, ...]
    sampler_fallback_policy: str
    sampler_viability_policy: str
    gate_min_ratio_improvement: float
    gate_min_inner_wins: int
    sampler_tie_margin: float
    task_increment_min_ratio: float
    safety_max_bacc_regression: float
    minimum_real_bacc: float
    code_version: str

    @property
    def mode(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class SourceInnerPriorRecoveryConfig(PriorRecoveryConfig):
    @property
    def mode(self) -> str:
        return "source_inner"


@dataclass(frozen=True)
class SourceInnerStabilityConfig(PriorRecoveryConfig):
    training_seeds: tuple[int, ...]
    consensus_rule_id: str
    child_code_version: str

    @property
    def mode(self) -> str:
        return "source_inner_training_seed_stability"


@dataclass(frozen=True)
class OuterPriorRecoveryConfig(PriorRecoveryConfig):
    reference_artifact_root: Path
    recipe_lock_artifact_root: Path
    training_seeds: tuple[int, ...]
    positive_claim_min_ratio: float
    positive_claim_min_center_wins: int

    @property
    def mode(self) -> str:
        return "outer"


def recipe_contract_payload(config: PriorRecoveryConfig) -> dict[str, object]:
    """Return every field allowed to influence source-inner recipe selection."""

    return {
        "schema_version": "midogpp_prior_recovery_recipe_contract_v1",
        "heldout_centers": list(config.heldout_centers),
        "expected_feature_dim": config.expected_feature_dim,
        "pca_dim": config.pca_dim,
        "selection_training_seed": config.selection_training_seed,
        "generation_seeds": list(config.generation_seeds),
        "device": config.device,
        "isotropic_variant": config.isotropic_variant.to_payload(),
        "task_fisher_variant": config.task_fisher_variant.to_payload(),
        "sampler_min_class_count": config.sampler_min_class_count,
        "sampler_max_condition_number": config.sampler_max_condition_number,
        "sampler_families": list(config.sampler_families),
        "sampler_fallback_policy": config.sampler_fallback_policy,
        "sampler_viability_policy": config.sampler_viability_policy,
        "gate_min_ratio_improvement": config.gate_min_ratio_improvement,
        "gate_min_inner_wins": config.gate_min_inner_wins,
        "sampler_tie_margin": config.sampler_tie_margin,
        "task_increment_min_ratio": config.task_increment_min_ratio,
        "safety_max_bacc_regression": config.safety_max_bacc_regression,
        "minimum_real_bacc": config.minimum_real_bacc,
        "classifier_grid_hash": SOURCE_INNER_CLASSIFIER_GRID_HASH,
        "classifier_policy": {
            "C": 0.01,
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 5000,
            "class_weight_candidates": [None, "balanced"],
            "threshold_policy": "predict",
            "classifier_seed": 23,
            "freeze_source": "stage10_source_inner_all_outer_centers_selected_C_0_01",
            "selection_used_stage20_outer_or_inner_metrics": False,
        },
        "code_version": config.code_version,
    }


def recipe_contract_hash(config: PriorRecoveryConfig) -> str:
    return stable_hash(recipe_contract_payload(config))


def scalar_source_inner_config(
    config: SourceInnerStabilityConfig,
    *,
    training_seed: int,
) -> SourceInnerPriorRecoveryConfig:
    """Derive the exact scalar-v1 computational contract for one seed."""

    seed = int(training_seed)
    if seed not in config.training_seeds:
        raise ProtocolError(f"Training seed {seed} is not in the stability panel.")
    return SourceInnerPriorRecoveryConfig(
        name=SOURCE_INNER_EXPERIMENT,
        artifact_root=config.artifact_root,
        manifest_path=config.manifest_path,
        feature_cache_path=config.feature_cache_path,
        heldout_centers=config.heldout_centers,
        expected_feature_dim=config.expected_feature_dim,
        pca_dim=config.pca_dim,
        selection_training_seed=seed,
        generation_seeds=config.generation_seeds,
        device=config.device,
        isotropic_variant=config.isotropic_variant,
        task_fisher_variant=config.task_fisher_variant,
        sampler_min_class_count=config.sampler_min_class_count,
        sampler_max_condition_number=config.sampler_max_condition_number,
        sampler_families=config.sampler_families,
        sampler_fallback_policy=config.sampler_fallback_policy,
        sampler_viability_policy=config.sampler_viability_policy,
        gate_min_ratio_improvement=config.gate_min_ratio_improvement,
        gate_min_inner_wins=config.gate_min_inner_wins,
        sampler_tie_margin=config.sampler_tie_margin,
        task_increment_min_ratio=config.task_increment_min_ratio,
        safety_max_bacc_regression=config.safety_max_bacc_regression,
        minimum_real_bacc=config.minimum_real_bacc,
        code_version=config.child_code_version,
    )


def common_source_inner_design_payload(
    config: SourceInnerStabilityConfig,
) -> dict[str, object]:
    """Return frozen scientific choices, excluding only the varied training seed."""

    scalar = scalar_source_inner_config(
        config,
        training_seed=config.training_seeds[0],
    )
    payload = recipe_contract_payload(scalar)
    payload.pop("selection_training_seed")
    return {
        "schema_version": "midogpp_prior_recovery_common_source_inner_design_v1",
        **payload,
    }


def common_source_inner_design_hash(config: SourceInnerStabilityConfig) -> str:
    return stable_hash(common_source_inner_design_payload(config))


def stability_contract_payload(
    config: SourceInnerStabilityConfig,
) -> dict[str, object]:
    child_hashes = {
        str(seed): recipe_contract_hash(
            scalar_source_inner_config(config, training_seed=seed)
        )
        for seed in config.training_seeds
    }
    return {
        "schema_version": "midogpp_prior_recovery_training_seed_stability_contract_v1",
        "heldout_centers": list(config.heldout_centers),
        "training_seeds": list(config.training_seeds),
        "selection_training_seed": config.selection_training_seed,
        "generation_seeds": list(config.generation_seeds),
        "consensus_rule_id": config.consensus_rule_id,
        "common_design": common_source_inner_design_payload(config),
        "common_design_hash": common_source_inner_design_hash(config),
        "child_recipe_contract_hashes": child_hashes,
        "parent_code_version": config.code_version,
        "child_code_version": config.child_code_version,
    }


def stability_contract_hash(config: SourceInnerStabilityConfig) -> str:
    return stable_hash(stability_contract_payload(config))


def outer_decision_contract_payload(config: OuterPriorRecoveryConfig) -> dict[str, object]:
    return {
        "positive_claim_min_ratio": config.positive_claim_min_ratio,
        "positive_claim_min_center_wins": config.positive_claim_min_center_wins,
        "safety_max_bacc_regression": config.safety_max_bacc_regression,
    }


def outer_decision_contract_hash(config: OuterPriorRecoveryConfig) -> str:
    return stable_hash(outer_decision_contract_payload(config))


def load_prior_recovery_config(
    path: str | Path,
    *,
    expected_mode: str | None = None,
) -> SourceInnerPriorRecoveryConfig | SourceInnerStabilityConfig | OuterPriorRecoveryConfig:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("Prior-recovery configs require PyYAML.") from exc
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ProtocolError("Prior-recovery config must be a mapping.")
    experiment = _mapping(payload.get("experiment"), "experiment")
    inputs = _mapping(payload.get("inputs"), "inputs")
    run = _mapping(payload.get("run", {}), "run")
    model = _mapping(payload.get("model", {}), "model")
    sampler = _mapping(payload.get("sampler", {}), "sampler")
    decisions = _mapping(payload.get("decisions", {}), "decisions")
    name = str(experiment.get("name", ""))
    default_mode = (
        "source_inner"
        if name == SOURCE_INNER_EXPERIMENT
        else (
            "source_inner_training_seed_stability"
            if name == SOURCE_INNER_STABILITY_EXPERIMENT
            else "outer"
        )
    )
    mode = str(experiment.get("mode") or default_mode)
    if mode not in {"source_inner", "source_inner_training_seed_stability", "outer"}:
        raise ProtocolError(f"Unsupported prior-recovery mode: {mode!r}")
    if expected_mode and mode != expected_mode:
        raise ProtocolError(f"Expected prior-recovery mode {expected_mode!r}, got {mode!r}")
    heldout_raw = run.get("heldout_centers", "all")
    heldouts = (
        MIDOGPP_ELIGIBLE_CENTERS
        if str(heldout_raw).lower() == "all"
        else tuple(str(value) for value in heldout_raw)
    )
    unknown = set(heldouts).difference(MIDOGPP_ELIGIBLE_CENTERS)
    if unknown:
        raise ProtocolError(f"Unknown or quarantined heldout centers: {sorted(unknown)}")
    base_variant = TrainingVariant(
        hidden_dim=int(model.get("hidden_dim", 512)),
        latent_dim=int(model.get("latent_dim", 32)),
        num_hidden_layers=int(model.get("num_hidden_layers", 2)),
        train_epochs=int(model.get("train_epochs", 100)),
        batch_size=int(model.get("batch_size", 128)),
        learning_rate=float(model.get("learning_rate", 1e-3)),
        weight_decay=float(model.get("weight_decay", 1e-4)),
        beta_final=float(model.get("beta_final", 1e-3)),
        kl_warmup_epochs=int(model.get("kl_warmup_epochs", 25)),
    )
    alpha = float(model.get("task_fisher_alpha", 1.0))
    families = tuple(str(value) for value in sampler.get("families", ()))
    base = config_path.parent
    common: dict[str, object] = {
        "name": name,
        "artifact_root": _path(base, str(experiment["artifact_root"])),
        "manifest_path": _path(base, str(inputs["manifest_path"])),
        "feature_cache_path": _path(base, str(inputs["feature_cache_path"])),
        "heldout_centers": tuple(heldouts),
        "expected_feature_dim": int(run.get("expected_feature_dim", 2560)),
        "pca_dim": int(model.get("pca_dim", 128)),
        "selection_training_seed": int(run.get("selection_training_seed", 42)),
        "generation_seeds": _int_tuple(run.get("generation_seeds", (17, 42, 101))),
        "device": str(run.get("device", "cpu")),
        "isotropic_variant": replace(base_variant, objective_id=ISOTROPIC_OBJECTIVE, alpha=0.0),
        "task_fisher_variant": replace(base_variant, objective_id=TASK_FISHER_OBJECTIVE, alpha=alpha),
        "sampler_min_class_count": int(sampler.get("min_class_count", 64)),
        "sampler_max_condition_number": float(sampler.get("max_condition_number", 1e6)),
        "sampler_families": families,
        "sampler_fallback_policy": str(sampler.get("fallback", "")),
        "sampler_viability_policy": str(sampler.get("viability", "")),
        "gate_min_ratio_improvement": float(decisions.get("gate_min_ratio_improvement", 0.05)),
        "gate_min_inner_wins": int(decisions.get("gate_min_inner_wins", 6)),
        "sampler_tie_margin": float(decisions.get("sampler_tie_margin", 0.01)),
        "task_increment_min_ratio": float(decisions.get("task_increment_min_ratio", 0.01)),
        "safety_max_bacc_regression": float(decisions.get("safety_max_bacc_regression", 0.01)),
        "minimum_real_bacc": float(decisions.get("minimum_real_bacc", 0.55)),
        "code_version": str(experiment.get("code_version", "prior_recovery_v2_resume")),
    }
    if mode == "source_inner":
        config: SourceInnerPriorRecoveryConfig | SourceInnerStabilityConfig | OuterPriorRecoveryConfig = SourceInnerPriorRecoveryConfig(**common)
    elif mode == "source_inner_training_seed_stability":
        config = SourceInnerStabilityConfig(
            **common,
            training_seeds=_int_tuple(run.get("training_seeds", STABILITY_TRAINING_SEEDS)),
            consensus_rule_id=str(
                decisions.get("training_seed_consensus_rule", STABILITY_CONSENSUS_RULE)
            ),
            child_code_version=str(
                experiment.get("child_code_version", "prior_recovery_v2_resume")
            ),
        )
    else:
        config = OuterPriorRecoveryConfig(
            **common,
            reference_artifact_root=_path(base, str(inputs["reference_artifact_root"])),
            recipe_lock_artifact_root=_path(base, str(inputs["recipe_lock_artifact_root"])),
            training_seeds=_int_tuple(run.get("training_seeds", (17, 42, 101))),
            positive_claim_min_ratio=float(decisions.get("positive_claim_min_ratio", 0.80)),
            positive_claim_min_center_wins=int(decisions.get("positive_claim_min_center_wins", 7)),
        )
    _validate(config)
    return config


def _validate(config: PriorRecoveryConfig) -> None:
    if config.heldout_centers != MIDOGPP_ELIGIBLE_CENTERS:
        raise ProtocolError("Production prior-recovery configs require exact nine-center coverage.")
    if not config.generation_seeds:
        raise ProtocolError("Prior recovery requires nonempty generation seeds.")
    if isinstance(config, OuterPriorRecoveryConfig) and not config.training_seeds:
        raise ProtocolError("Outer prior recovery requires nonempty training seeds.")
    if isinstance(config, SourceInnerStabilityConfig):
        if config.training_seeds != STABILITY_TRAINING_SEEDS:
            raise ProtocolError(
                "Training-seed stability requires ordered seeds (17, 42, 101)."
            )
        if config.generation_seeds != STABILITY_TRAINING_SEEDS:
            raise ProtocolError(
                "Training-seed stability requires ordered generation seeds (17, 42, 101)."
            )
        if config.selection_training_seed != config.training_seeds[0]:
            raise ProtocolError(
                "Stability selection_training_seed must equal the first panel seed."
            )
        if config.consensus_rule_id != STABILITY_CONSENSUS_RULE:
            raise ProtocolError("Training-seed stability consensus rule drifted.")
        if config.child_code_version != "prior_recovery_v2_resume":
            raise ProtocolError("Training-seed stability child code version drifted from v1.")
    if config.sampler_min_class_count <= 0:
        raise ProtocolError("Sampler class count must be positive.")
    if config.pca_dim != 128:
        raise ProtocolError("Production prior recovery fixes PCA to 128 dimensions; it is not a sweep.")
    if config.gate_min_inner_wins < 1:
        raise ProtocolError("gate_min_inner_wins must be positive.")
    if config.task_fisher_variant.alpha != 1.0:
        raise ProtocolError("Task-Fisher alpha is predeclared and locked to 1.0 in v1.")
    expected_families = (STANDARD_SAMPLER, DIAGONAL_SAMPLER, FULL_SAMPLER)
    if config.sampler_families != expected_families:
        raise ProtocolError("Prior recovery requires the exact standard/diagonal/full sampler order.")
    if config.sampler_fallback_policy != SAMPLER_FALLBACK_POLICY:
        raise ProtocolError("Prior-recovery sampler fallback policy drifted.")
    if config.sampler_viability_policy != SAMPLER_VIABILITY_POLICY:
        raise ProtocolError("Prior-recovery sampler viability policy drifted.")


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{name} must be a mapping.")
    return value


def _path(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _int_tuple(value: object) -> tuple[int, ...]:
    if isinstance(value, str):
        values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    else:
        values = tuple(int(item) for item in value)  # type: ignore[union-attr]
    if len(set(values)) != len(values):
        raise ProtocolError("Seed lists must not contain duplicates.")
    return values
