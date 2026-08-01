"""Fail-closed config for fresh BG training and P0/Pq generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Mapping

from ....common.hashing import stable_hash
from ....real_features.classifier_reference.schemas.midogpp import (
    MIDOGPP_ELIGIBLE_CENTERS,
)
from ...protocol import ProtocolError
from .contracts import (
    CLAIM_SCOPE,
    MODE,
    OUTPUT_ARTIFACT_ID,
    PRIORS,
    STUDY_NAME,
    STUDY_VERSION,
    UNIFORM_B_FEATURE_HASH,
    UNIFORM_B_INPUT_ARTIFACT_ID,
)


@dataclass(frozen=True)
class UniformBResampledPriorConfig:
    name: str
    mode: str
    study_version: str
    artifact_root: Path
    code_version: str
    manifest_path: Path
    feature_cache_path: Path
    feature_cache_artifact_id: str
    expected_feature_cache_hash: str
    heldout_centers: tuple[str, ...]
    expected_feature_dim: int
    training_seeds: tuple[int, ...]
    generation_seeds: tuple[int, ...]
    device: str
    training_arm: str
    priors: tuple[str, ...]
    block_frame: str
    pca_output_dim: int
    hidden_dim: int
    latent_dim: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    beta_final: float
    gradient_clip_norm: float
    warmup_steps: int
    total_steps: int
    geco_target_slack: float
    geco_ema_decay: float
    geco_dual_step_size: float
    geco_initial_multiplier: float
    geco_minimum_multiplier: float
    geco_maximum_multiplier: float
    ratio_crossfit_folds: int
    ratio_classifier_c: float
    ratio_classifier_max_iter: int
    ratio_lambda: float
    acceptance_floor: float
    proposal_multiplier: int
    min_ratio_auc: float
    min_log_loss_gain: float
    min_acceptance_rate: float
    min_ess_ratio: float
    base_generation_per_class: int
    classifier_c: float
    classifier_seed: int
    min_effective_rank_ratio: float
    min_pairwise_distance_ratio: float
    max_pairwise_distance_ratio: float
    max_source_mean_regression: float
    claim_scope: str
    inner_labels_scoring_only: bool
    target_support_labels_for_selection: bool
    fresh_bg_training_required: bool
    existing_checkpoint_input_allowed: bool
    may_feed_recipe_selection: bool
    may_feed_deployable_selection: bool
    separate_promotion_artifact_required: bool

    @property
    def contract_hash(self) -> str:
        payload = asdict(self)
        for key in ("artifact_root", "manifest_path", "feature_cache_path"):
            payload.pop(key)
        return stable_hash(payload)


def load_uniform_b_resampled_prior_config(
    path: str | Path,
) -> UniformBResampledPriorConfig:
    import yaml

    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ProtocolError("Resampled-prior config must be a mapping.")
    experiment = _mapping(payload.get("experiment"), "experiment")
    inputs = _mapping(payload.get("inputs"), "inputs")
    run = _mapping(payload.get("run"), "run")
    model = _mapping(payload.get("model"), "model")
    geco = _mapping(payload.get("geco"), "geco")
    ratio = _mapping(payload.get("posterior_ratio"), "posterior_ratio")
    generation = _mapping(payload.get("generation"), "generation")
    classifier = _mapping(payload.get("classifier"), "classifier")
    diversity = _mapping(payload.get("diversity_gates"), "diversity_gates")
    decision = _mapping(payload.get("decision_gates"), "decision_gates")
    claim = _mapping(payload.get("claim_boundary"), "claim_boundary")
    base = config_path.parent
    heldout_raw = run.get("heldout_centers", "all")
    heldout = (
        MIDOGPP_ELIGIBLE_CENTERS
        if str(heldout_raw).lower() == "all"
        else tuple(str(value) for value in heldout_raw)  # type: ignore[union-attr]
    )
    config = UniformBResampledPriorConfig(
        name=str(experiment.get("name", "")),
        mode=str(experiment.get("mode", "")),
        study_version=str(experiment.get("study_version", "")),
        artifact_root=_path(base, _required(experiment, "artifact_root")),
        code_version=str(experiment.get("code_version", "")),
        manifest_path=_path(base, _required(inputs, "manifest_path")),
        feature_cache_path=_path(base, _required(inputs, "feature_cache_path")),
        feature_cache_artifact_id=str(inputs.get("feature_cache_artifact_id", "")),
        expected_feature_cache_hash=str(inputs.get("expected_feature_cache_hash", "")),
        heldout_centers=tuple(heldout),
        expected_feature_dim=int(run.get("expected_feature_dim", 0)),
        training_seeds=_int_tuple(run.get("training_seeds", ())),
        generation_seeds=_int_tuple(run.get("generation_seeds", ())),
        device=str(run.get("device", "")),
        training_arm=str(model.get("training_arm", "")),
        priors=tuple(str(value) for value in model.get("priors", ())),
        block_frame=str(model.get("block_frame", "")),
        pca_output_dim=int(model.get("pca_output_dim", 0)),
        hidden_dim=int(model.get("hidden_dim", 0)),
        latent_dim=int(model.get("latent_dim", 0)),
        batch_size=int(model.get("batch_size", 0)),
        learning_rate=float(model.get("learning_rate", float("nan"))),
        weight_decay=float(model.get("weight_decay", float("nan"))),
        beta_final=float(model.get("beta_final", float("nan"))),
        gradient_clip_norm=float(model.get("gradient_clip_norm", float("nan"))),
        warmup_steps=int(model.get("warmup_steps", 0)),
        total_steps=int(model.get("total_steps", 0)),
        geco_target_slack=float(geco.get("target_slack", float("nan"))),
        geco_ema_decay=float(geco.get("ema_decay", float("nan"))),
        geco_dual_step_size=float(geco.get("dual_step_size", float("nan"))),
        geco_initial_multiplier=float(geco.get("initial_multiplier", float("nan"))),
        geco_minimum_multiplier=float(geco.get("minimum_multiplier", float("nan"))),
        geco_maximum_multiplier=float(geco.get("maximum_multiplier", float("nan"))),
        ratio_crossfit_folds=int(ratio.get("crossfit_folds", 0)),
        ratio_classifier_c=float(ratio.get("classifier_c", float("nan"))),
        ratio_classifier_max_iter=int(ratio.get("classifier_max_iter", 0)),
        ratio_lambda=float(ratio.get("lambda", float("nan"))),
        acceptance_floor=float(ratio.get("acceptance_floor", float("nan"))),
        proposal_multiplier=int(ratio.get("proposal_multiplier", 0)),
        min_ratio_auc=float(ratio.get("min_crossfit_auc", float("nan"))),
        min_log_loss_gain=float(ratio.get("min_log_loss_gain", float("nan"))),
        min_acceptance_rate=float(ratio.get("min_acceptance_rate", float("nan"))),
        min_ess_ratio=float(ratio.get("min_ess_ratio", float("nan"))),
        base_generation_per_class=int(generation.get("base_per_class", 0)),
        classifier_c=float(classifier.get("C", float("nan"))),
        classifier_seed=int(classifier.get("seed", -1)),
        min_effective_rank_ratio=float(diversity.get("min_effective_rank_ratio", float("nan"))),
        min_pairwise_distance_ratio=float(diversity.get("min_pairwise_distance_ratio", float("nan"))),
        max_pairwise_distance_ratio=float(diversity.get("max_pairwise_distance_ratio", float("nan"))),
        max_source_mean_regression=float(decision.get("max_source_mean_regression", float("nan"))),
        claim_scope=str(claim.get("claim_scope", "")),
        inner_labels_scoring_only=_required_bool(claim, "inner_labels_scoring_only"),
        target_support_labels_for_selection=_required_bool(claim, "target_support_labels_for_selection"),
        fresh_bg_training_required=_required_bool(claim, "fresh_bg_training_required"),
        existing_checkpoint_input_allowed=_required_bool(claim, "existing_checkpoint_input_allowed"),
        may_feed_recipe_selection=_required_bool(claim, "may_feed_recipe_selection"),
        may_feed_deployable_selection=_required_bool(claim, "may_feed_deployable_selection"),
        separate_promotion_artifact_required=_required_bool(claim, "separate_promotion_artifact_required"),
    )
    _validate(config)
    return config


def _validate(config: UniformBResampledPriorConfig) -> None:
    exact = {
        "name": (config.name, STUDY_NAME),
        "mode": (config.mode, MODE),
        "study_version": (config.study_version, STUDY_VERSION),
        "feature_cache_artifact_id": (config.feature_cache_artifact_id, UNIFORM_B_INPUT_ARTIFACT_ID),
        "expected_feature_cache_hash": (config.expected_feature_cache_hash, UNIFORM_B_FEATURE_HASH),
        "heldout_centers": (config.heldout_centers, MIDOGPP_ELIGIBLE_CENTERS),
        "expected_feature_dim": (config.expected_feature_dim, 3840),
        "training_seeds": (config.training_seeds, (17, 42, 101)),
        "generation_seeds": (config.generation_seeds, (17, 42, 101)),
        "training_arm": (config.training_arm, "BG"),
        "priors": (config.priors, PRIORS),
        "block_frame": (config.block_frame, "b_block_pca96_32"),
        "pca_output_dim": (config.pca_output_dim, 128),
        "claim_scope": (config.claim_scope, CLAIM_SCOPE),
        "inner_labels_scoring_only": (config.inner_labels_scoring_only, True),
        "target_support_labels_for_selection": (config.target_support_labels_for_selection, False),
        "fresh_bg_training_required": (config.fresh_bg_training_required, True),
        "existing_checkpoint_input_allowed": (config.existing_checkpoint_input_allowed, False),
        "may_feed_recipe_selection": (config.may_feed_recipe_selection, False),
        "may_feed_deployable_selection": (config.may_feed_deployable_selection, False),
        "separate_promotion_artifact_required": (config.separate_promotion_artifact_required, True),
    }
    mismatches = [
        f"{name}: observed={observed!r}, expected={expected!r}"
        for name, (observed, expected) in exact.items()
        if observed != expected
    ]
    if mismatches:
        raise ProtocolError("Resampled-prior config violates locked values: " + "; ".join(mismatches))
    numeric = (
        config.learning_rate, config.weight_decay, config.beta_final,
        config.gradient_clip_norm, config.geco_target_slack,
        config.geco_ema_decay, config.geco_dual_step_size,
        config.ratio_classifier_c, config.ratio_lambda,
        config.acceptance_floor, config.min_ratio_auc,
        config.min_log_loss_gain, config.min_acceptance_rate,
        config.min_ess_ratio, config.classifier_c,
        config.min_effective_rank_ratio, config.min_pairwise_distance_ratio,
        config.max_pairwise_distance_ratio, config.max_source_mean_regression,
    )
    if not all(math.isfinite(value) for value in numeric):
        raise ProtocolError("Resampled-prior config contains a nonfinite value.")
    if not (
        0 < config.warmup_steps < config.total_steps
        and config.batch_size > 0 and config.batch_size % 2 == 0
        and config.hidden_dim > 0 and config.latent_dim > 0
        and config.ratio_crossfit_folds >= 3
        and config.ratio_classifier_max_iter > 0
        and config.proposal_multiplier >= 2
        and config.base_generation_per_class > 1
        and config.classifier_seed >= 0
        and config.learning_rate > 0.0
        and config.weight_decay >= 0.0
        and config.gradient_clip_norm > 0.0
        and config.geco_target_slack > 0.0
        and 0.0 <= config.geco_ema_decay < 1.0
        and config.geco_dual_step_size > 0.0
        and config.ratio_classifier_c > 0.0
        and config.ratio_lambda >= 0.0
        and 0.0 < config.acceptance_floor < 1.0
        and 0.5 <= config.min_ratio_auc < 1.0
        and config.min_log_loss_gain >= 0.0
        and 0.0 < config.min_acceptance_rate <= 1.0
        and 0.0 < config.min_ess_ratio <= 1.0
        and config.classifier_c > 0.0
        and 0.0 < config.min_pairwise_distance_ratio <= config.max_pairwise_distance_ratio
        and config.max_source_mean_regression >= 0.0
    ):
        raise ProtocolError("Resampled-prior config ranges are invalid.")
    forbidden = " ".join((str(config.manifest_path), str(config.feature_cache_path))).lower()
    if any(token in forbidden for token in ("task_geometry_source_inner", "90_oracles", "snapshot", "quarantine")):
        raise ProtocolError("Completed/quarantined experiment inputs are forbidden.")
    if config.artifact_root.name != OUTPUT_ARTIFACT_ID and str(config.artifact_root).startswith("output://"):
        raise ProtocolError("Unexpected resampled-prior output identity.")


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Config section {name!r} must be a mapping.")
    return value


def _required(mapping: Mapping[str, object], key: str) -> str:
    value = str(mapping.get(key, "")).strip()
    if not value:
        raise ProtocolError(f"Missing required config value: {key}")
    return value


def _required_bool(mapping: Mapping[str, object], key: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise ProtocolError(f"Config value {key!r} must be boolean.")
    return value


def _path(base: Path, value: str) -> Path:
    if value.startswith(("artifact://", "output://")):
        return Path(value)
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _int_tuple(value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ProtocolError("Expected an integer list.")
    return tuple(int(item) for item in value)


__all__ = ("UniformBResampledPriorConfig", "load_uniform_b_resampled_prior_config")
