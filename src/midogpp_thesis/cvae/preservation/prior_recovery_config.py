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


SOURCE_INNER_EXPERIMENT = "virchow2_cvae_midogpp_prior_recovery_source_inner_v1"
OUTER_EXPERIMENT = "virchow2_cvae_midogpp_prior_recovery_outer_v1"
CANONICAL_CLASSIFIER_GRID_HASH = "16a7a1183ea3f65b"
SAMPLER_FALLBACK_POLICY = "full_to_diagonal_to_standard_per_class"
SAMPLER_VIABILITY_POLICY = "require_requested_family_both_classes"


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
        "classifier_grid_hash": CANONICAL_CLASSIFIER_GRID_HASH,
        "code_version": config.code_version,
    }


def recipe_contract_hash(config: PriorRecoveryConfig) -> str:
    return stable_hash(recipe_contract_payload(config))


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
) -> SourceInnerPriorRecoveryConfig | OuterPriorRecoveryConfig:
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
    mode = str(experiment.get("mode") or ("source_inner" if name == SOURCE_INNER_EXPERIMENT else "outer"))
    if mode not in {"source_inner", "outer"}:
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
        "code_version": str(experiment.get("code_version", "prior_recovery_v1")),
    }
    if mode == "source_inner":
        config: SourceInnerPriorRecoveryConfig | OuterPriorRecoveryConfig = SourceInnerPriorRecoveryConfig(**common)
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
    if config.sampler_min_class_count <= 0 or config.pca_dim <= 0:
        raise ProtocolError("Sampler class count and PCA dimension must be positive.")
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
