"""Fail-closed v2 configuration for improving the Uniform-B CVAE prior."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Mapping

from ....common.hashing import stable_hash
from ....real_features.classifier_reference.schemas.midogpp import MIDOGPP_ELIGIBLE_CENTERS
from ...generation_samplers import FULL_SAMPLER
from ...protocol import ProtocolError
from .contracts import (
    ARMS, CLAIM_SCOPE, FRAME, MODE, OUTPUT_ARTIFACT_ID, STUDY_NAME,
    STUDY_VERSION, UNIFORM_B_FEATURE_HASH, UNIFORM_B_INPUT_ARTIFACT_ID,
)


@dataclass(frozen=True)
class OptimizedPriorConfig:
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
    training_seeds: tuple[int, ...]
    generation_seeds: tuple[int, ...]
    device: str
    block_frame: str
    pca_output_dim: int
    hidden_dim: int
    latent_dim: int
    num_hidden_layers: int
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
    sampler_family: str
    sampler_min_class_count: int
    sampler_max_condition_number: float
    arms: tuple[str, ...]
    total_generation_per_class: int
    classifier_c: float
    classifier_seed: int
    target_prior_bacc: float
    required_posterior_ceiling: float
    runtime_scoring_workers: int
    runtime_training_devices: tuple[str, ...]
    claim_scope: str
    inner_labels_scoring_only: bool
    target_support_labels_for_selection: bool
    may_feed_deployable_selection: bool

    @property
    def contract_hash(self) -> str:
        payload = asdict(self)
        for key in (
            "artifact_root", "manifest_path", "feature_cache_path",
            "runtime_scoring_workers", "runtime_training_devices",
        ):
            payload.pop(key)
        return stable_hash(payload)


def load_optimized_prior_config(path: str | Path) -> OptimizedPriorConfig:
    import yaml

    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ProtocolError("Optimized-prior config must be a mapping.")
    experiment = _mapping(payload, "experiment")
    inputs = _mapping(payload, "inputs")
    run = _mapping(payload, "run")
    model = _mapping(payload, "model")
    geco = _mapping(payload, "geco")
    prior = _mapping(payload, "aggregate_prior")
    generation = _mapping(payload, "generation")
    classifier = _mapping(payload, "classifier")
    gates = _mapping(payload, "decision_gates")
    runtime = _mapping(payload, "runtime")
    claim = _mapping(payload, "claim_boundary")
    base = config_path.parent
    centers_raw = run.get("heldout_centers", "all")
    centers = (
        MIDOGPP_ELIGIBLE_CENTERS
        if str(centers_raw).lower() == "all"
        else tuple(str(value) for value in centers_raw)  # type: ignore[union-attr]
    )
    config = OptimizedPriorConfig(
        name=str(experiment.get("name", "")),
        mode=str(experiment.get("mode", "")),
        study_version=str(experiment.get("study_version", "")),
        artifact_root=_path(base, str(experiment.get("artifact_root", ""))),
        code_version=str(experiment.get("code_version", "")),
        manifest_path=_path(base, str(inputs.get("manifest_path", ""))),
        feature_cache_path=_path(base, str(inputs.get("feature_cache_path", ""))),
        feature_cache_artifact_id=str(inputs.get("feature_cache_artifact_id", "")),
        expected_feature_cache_hash=str(inputs.get("expected_feature_cache_hash", "")),
        heldout_centers=tuple(centers),
        training_seeds=_ints(run.get("training_seeds", ())),
        generation_seeds=_ints(run.get("generation_seeds", ())),
        device=str(run.get("device", "")),
        block_frame=str(model.get("block_frame", "")),
        pca_output_dim=int(model.get("pca_output_dim", 0)),
        hidden_dim=int(model.get("hidden_dim", 0)),
        latent_dim=int(model.get("latent_dim", 0)),
        num_hidden_layers=int(model.get("num_hidden_layers", 0)),
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
        sampler_family=str(prior.get("family", "")),
        sampler_min_class_count=int(prior.get("min_class_count", 0)),
        sampler_max_condition_number=float(prior.get("max_condition_number", float("nan"))),
        arms=tuple(str(value) for value in generation.get("arms", ())),
        total_generation_per_class=int(generation.get("total_per_class", 0)),
        classifier_c=float(classifier.get("C", float("nan"))),
        classifier_seed=int(classifier.get("seed", -1)),
        target_prior_bacc=float(gates.get("target_prior_bacc", float("nan"))),
        required_posterior_ceiling=float(gates.get("required_posterior_ceiling", float("nan"))),
        runtime_scoring_workers=int(runtime.get("scoring_workers", 12)),
        runtime_training_devices=tuple(str(value) for value in runtime.get("training_devices", ("cuda:0", "cuda:1"))),
        claim_scope=str(claim.get("claim_scope", "")),
        inner_labels_scoring_only=_bool(claim, "inner_labels_scoring_only"),
        target_support_labels_for_selection=_bool(claim, "target_support_labels_for_selection"),
        may_feed_deployable_selection=_bool(claim, "may_feed_deployable_selection"),
    )
    _validate(config)
    return config


def _validate(config: OptimizedPriorConfig) -> None:
    exact = {
        "name": (config.name, STUDY_NAME),
        "mode": (config.mode, MODE),
        "study_version": (config.study_version, STUDY_VERSION),
        "feature_cache_artifact_id": (config.feature_cache_artifact_id, UNIFORM_B_INPUT_ARTIFACT_ID),
        "expected_feature_cache_hash": (config.expected_feature_cache_hash, UNIFORM_B_FEATURE_HASH),
        "heldout_centers": (config.heldout_centers, MIDOGPP_ELIGIBLE_CENTERS),
        "training_seeds": (config.training_seeds, (17, 42, 101)),
        "generation_seeds": (config.generation_seeds, (17, 42, 101)),
        "block_frame": (config.block_frame, FRAME),
        "pca_output_dim": (config.pca_output_dim, 256),
        "num_hidden_layers": (config.num_hidden_layers, 3),
        "sampler_family": (config.sampler_family, FULL_SAMPLER),
        "arms": (config.arms, ARMS),
        "claim_scope": (config.claim_scope, CLAIM_SCOPE),
        "inner_labels_scoring_only": (config.inner_labels_scoring_only, True),
        "target_support_labels_for_selection": (config.target_support_labels_for_selection, False),
        "may_feed_deployable_selection": (config.may_feed_deployable_selection, False),
    }
    mismatch = [
        f"{key}: observed={got!r}, expected={want!r}"
        for key, (got, want) in exact.items() if got != want
    ]
    if mismatch:
        raise ProtocolError("Optimized-prior config violates locked values: " + "; ".join(mismatch))
    finite = (
        config.learning_rate, config.weight_decay, config.beta_final,
        config.gradient_clip_norm, config.geco_target_slack,
        config.geco_ema_decay, config.geco_dual_step_size,
        config.geco_initial_multiplier, config.geco_minimum_multiplier,
        config.geco_maximum_multiplier, config.sampler_max_condition_number,
        config.classifier_c, config.target_prior_bacc,
        config.required_posterior_ceiling,
    )
    if not all(math.isfinite(value) for value in finite):
        raise ProtocolError("Optimized-prior config contains nonfinite values.")
    if not (
        config.hidden_dim >= 512 and config.latent_dim >= 32
        and config.batch_size >= 128 and config.batch_size % 2 == 0
        and 0 < config.warmup_steps < config.total_steps
        and config.total_steps >= 2000
        and config.learning_rate > 0 and config.weight_decay >= 0
        and config.gradient_clip_norm > 0
        and config.geco_target_slack > 0
        and 0 <= config.geco_ema_decay < 1
        and config.geco_dual_step_size > 0
        and 0 < config.geco_minimum_multiplier <= config.geco_initial_multiplier <= config.geco_maximum_multiplier
        and config.sampler_min_class_count >= config.latent_dim
        and config.sampler_max_condition_number >= 1e3
        and config.total_generation_per_class >= 7
        and config.classifier_c > 0 and config.classifier_seed >= 0
        and config.target_prior_bacc >= 0.7
        and config.required_posterior_ceiling > config.target_prior_bacc
        and 1 <= config.runtime_scoring_workers <= 24
        and config.runtime_training_devices
    ):
        raise ProtocolError("Optimized-prior config ranges are invalid.")
    forbidden = " ".join((str(config.manifest_path), str(config.feature_cache_path))).lower()
    if any(token in forbidden for token in ("task_geometry_source_inner", "90_oracles", "snapshot", "quarantine")):
        raise ProtocolError("Completed or diagnostic experiment inputs are forbidden.")
    if str(config.artifact_root).startswith("output:") and config.artifact_root.name != OUTPUT_ARTIFACT_ID:
        raise ProtocolError("Unexpected v2 output identity.")


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Config section {key!r} must be a mapping.")
    return value


def _bool(payload: Mapping[str, object], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ProtocolError(f"Config value {key!r} must be boolean.")
    return value


def _ints(value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ProtocolError("Expected an integer list.")
    return tuple(int(item) for item in value)


def _path(base: Path, value: str) -> Path:
    if not value:
        raise ProtocolError("Required path is empty.")
    if value.startswith(("artifact://", "output://")):
        return Path(value)
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


__all__ = ("OptimizedPriorConfig", "load_optimized_prior_config")
