"""Fail-closed configuration for the Uniform-B task-geometry study."""

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
    ARMS,
    CLAIM_SCOPE,
    COMPOSITION_MODES,
    MODE,
    OUTPUT_ARTIFACT_ID,
    STUDY_NAME,
    STUDY_VERSION,
    UNIFORM_B_FEATURE_HASH,
    UNIFORM_B_INPUT_ARTIFACT_ID,
)


@dataclass(frozen=True)
class UniformBTaskGeometryConfig:
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
    arms: tuple[str, ...]
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
    task_start_step: int
    total_steps: int
    geco_target_slack: float
    geco_ema_decay: float
    geco_dual_step_size: float
    geco_initial_multiplier: float
    geco_minimum_multiplier: float
    geco_maximum_multiplier: float
    geco_target_uses_inner_or_outer_data: bool
    crossfit_folds: int
    nystrom_components: int
    nystrom_gamma: float
    teacher_c: float
    teacher_max_iter: int
    hessian_ridge: float
    hessian_eigenfloor: float
    reference_per_class: int
    mmd_bandwidth_multipliers: tuple[float, ...]
    cdf_grid_quantiles: tuple[float, ...]
    cdf_temperature: float
    mmd_weight: float
    margin_weight: float
    gradient_weight: float
    task_weight: float
    reference_rows_are_out_of_fold: bool
    base_generation_per_class: int
    composition_modes: tuple[str, ...]
    classifier_c: float
    classifier_seed: int
    min_effective_rank_ratio: float
    min_pairwise_distance_ratio: float
    max_pairwise_distance_ratio: float
    claim_scope: str
    inner_labels_scoring_only: bool
    target_support_labels_for_selection: bool
    may_change_existing_consensus_locks: bool
    may_feed_recipe_selection: bool
    may_feed_deployable_selection: bool
    separate_promotion_artifact_required: bool

    @property
    def contract_hash(self) -> str:
        payload = asdict(self)
        for key in ("artifact_root", "manifest_path", "feature_cache_path"):
            payload.pop(key)
        return stable_hash(payload)


def load_uniform_b_task_geometry_config(
    path: str | Path,
) -> UniformBTaskGeometryConfig:
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Uniform-B task-geometry configs require PyYAML.") from exc

    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ProtocolError("Uniform-B task-geometry config must be a mapping.")
    experiment = _mapping(payload.get("experiment"), "experiment")
    inputs = _mapping(payload.get("inputs"), "inputs")
    run = _mapping(payload.get("run"), "run")
    model = _mapping(payload.get("model"), "model")
    geco = _mapping(payload.get("geco"), "geco")
    task = _mapping(payload.get("task_geometry"), "task_geometry")
    generation = _mapping(payload.get("generation"), "generation")
    classifier = _mapping(payload.get("classifier"), "classifier")
    diversity = _mapping(payload.get("diversity_gates"), "diversity_gates")
    claim = _mapping(payload.get("claim_boundary"), "claim_boundary")
    base = config_path.parent
    heldout_raw = run.get("heldout_centers", "all")
    heldout = (
        MIDOGPP_ELIGIBLE_CENTERS
        if str(heldout_raw).lower() == "all"
        else tuple(str(value) for value in heldout_raw)  # type: ignore[union-attr]
    )
    config = UniformBTaskGeometryConfig(
        name=str(experiment.get("name", "")),
        mode=str(experiment.get("mode", "")),
        study_version=str(experiment.get("study_version", "")),
        artifact_root=_path(base, _required(experiment, "artifact_root")),
        code_version=str(experiment.get("code_version", "")),
        manifest_path=_path(base, _required(inputs, "manifest_path")),
        feature_cache_path=_path(base, _required(inputs, "feature_cache_path")),
        feature_cache_artifact_id=str(inputs.get("feature_cache_artifact_id", "")),
        expected_feature_cache_hash=str(
            inputs.get("expected_feature_cache_hash", "")
        ),
        heldout_centers=tuple(heldout),
        expected_feature_dim=int(run.get("expected_feature_dim", 0)),
        training_seeds=_int_tuple(run.get("training_seeds", ())),
        generation_seeds=_int_tuple(run.get("generation_seeds", ())),
        device=str(run.get("device", "")),
        arms=tuple(str(value) for value in model.get("arms", ())),
        block_frame=str(model.get("block_frame", "")),
        pca_output_dim=int(model.get("pca_output_dim", 0)),
        hidden_dim=int(model.get("hidden_dim", 0)),
        latent_dim=int(model.get("latent_dim", 0)),
        batch_size=int(model.get("batch_size", 0)),
        learning_rate=float(model.get("learning_rate", float("nan"))),
        weight_decay=float(model.get("weight_decay", float("nan"))),
        beta_final=float(model.get("beta_final", float("nan"))),
        gradient_clip_norm=float(
            model.get("gradient_clip_norm", float("nan"))
        ),
        warmup_steps=int(model.get("warmup_steps", 0)),
        task_start_step=int(model.get("task_start_step", 0)),
        total_steps=int(model.get("total_steps", 0)),
        geco_target_slack=float(geco.get("target_slack", float("nan"))),
        geco_ema_decay=float(geco.get("ema_decay", float("nan"))),
        geco_dual_step_size=float(geco.get("dual_step_size", float("nan"))),
        geco_initial_multiplier=float(
            geco.get("initial_multiplier", float("nan"))
        ),
        geco_minimum_multiplier=float(
            geco.get("minimum_multiplier", float("nan"))
        ),
        geco_maximum_multiplier=float(
            geco.get("maximum_multiplier", float("nan"))
        ),
        geco_target_uses_inner_or_outer_data=_required_bool(
            geco,
            "target_uses_inner_or_outer_data",
        ),
        crossfit_folds=int(task.get("crossfit_folds", 0)),
        nystrom_components=int(task.get("nystrom_components", 0)),
        nystrom_gamma=float(task.get("nystrom_gamma", float("nan"))),
        teacher_c=float(task.get("teacher_c", float("nan"))),
        teacher_max_iter=int(task.get("teacher_max_iter", 0)),
        hessian_ridge=float(task.get("hessian_ridge", float("nan"))),
        hessian_eigenfloor=float(
            task.get("hessian_eigenfloor", float("nan"))
        ),
        reference_per_class=int(task.get("reference_per_class", 0)),
        mmd_bandwidth_multipliers=_float_tuple(
            task.get("mmd_bandwidth_multipliers", ())
        ),
        cdf_grid_quantiles=_float_tuple(task.get("cdf_grid_quantiles", ())),
        cdf_temperature=float(task.get("cdf_temperature", float("nan"))),
        mmd_weight=float(task.get("mmd_weight", float("nan"))),
        margin_weight=float(task.get("margin_weight", float("nan"))),
        gradient_weight=float(task.get("gradient_weight", float("nan"))),
        task_weight=float(task.get("task_weight", float("nan"))),
        reference_rows_are_out_of_fold=_required_bool(
            task,
            "reference_rows_are_out_of_fold",
        ),
        base_generation_per_class=int(
            generation.get("base_per_class", 0)
        ),
        composition_modes=tuple(
            str(value) for value in generation.get("modes", ())
        ),
        classifier_c=float(classifier.get("C", float("nan"))),
        classifier_seed=int(classifier.get("seed", -1)),
        min_effective_rank_ratio=float(
            diversity.get("min_effective_rank_ratio", float("nan"))
        ),
        min_pairwise_distance_ratio=float(
            diversity.get("min_pairwise_distance_ratio", float("nan"))
        ),
        max_pairwise_distance_ratio=float(
            diversity.get("max_pairwise_distance_ratio", float("nan"))
        ),
        claim_scope=str(claim.get("claim_scope", "")),
        inner_labels_scoring_only=_required_bool(
            claim,
            "inner_labels_scoring_only",
        ),
        target_support_labels_for_selection=_required_bool(
            claim,
            "target_support_labels_for_selection",
        ),
        may_change_existing_consensus_locks=_required_bool(
            claim,
            "may_change_existing_consensus_locks",
        ),
        may_feed_recipe_selection=_required_bool(
            claim,
            "may_feed_recipe_selection",
        ),
        may_feed_deployable_selection=_required_bool(
            claim,
            "may_feed_deployable_selection",
        ),
        separate_promotion_artifact_required=_required_bool(
            claim,
            "separate_promotion_artifact_required",
        ),
    )
    _validate(config)
    return config


def _validate(config: UniformBTaskGeometryConfig) -> None:
    exact: dict[str, tuple[object, object]] = {
        "name": (config.name, STUDY_NAME),
        "mode": (config.mode, MODE),
        "study_version": (config.study_version, STUDY_VERSION),
        "feature_cache_artifact_id": (
            config.feature_cache_artifact_id,
            UNIFORM_B_INPUT_ARTIFACT_ID,
        ),
        "expected_feature_cache_hash": (
            config.expected_feature_cache_hash,
            UNIFORM_B_FEATURE_HASH,
        ),
        "heldout_centers": (config.heldout_centers, MIDOGPP_ELIGIBLE_CENTERS),
        "expected_feature_dim": (config.expected_feature_dim, 3840),
        "training_seeds": (config.training_seeds, (17, 42, 101)),
        "generation_seeds": (config.generation_seeds, (17, 42, 101)),
        "arms": (config.arms, ARMS),
        "block_frame": (config.block_frame, "b_block_pca96_32"),
        "pca_output_dim": (config.pca_output_dim, 128),
        "composition_modes": (config.composition_modes, COMPOSITION_MODES),
        "claim_scope": (config.claim_scope, CLAIM_SCOPE),
        "inner_labels_scoring_only": (config.inner_labels_scoring_only, True),
        "target_support_labels_for_selection": (
            config.target_support_labels_for_selection,
            False,
        ),
        "geco_target_uses_inner_or_outer_data": (
            config.geco_target_uses_inner_or_outer_data,
            False,
        ),
        "reference_rows_are_out_of_fold": (
            config.reference_rows_are_out_of_fold,
            True,
        ),
        "may_change_existing_consensus_locks": (
            config.may_change_existing_consensus_locks,
            False,
        ),
        "may_feed_recipe_selection": (config.may_feed_recipe_selection, False),
        "may_feed_deployable_selection": (
            config.may_feed_deployable_selection,
            False,
        ),
        "separate_promotion_artifact_required": (
            config.separate_promotion_artifact_required,
            True,
        ),
    }
    mismatches = [
        f"{name}: observed={observed!r}, expected={expected!r}"
        for name, (observed, expected) in exact.items()
        if observed != expected
    ]
    if mismatches:
        raise ProtocolError(
            "Uniform-B task-geometry config violates locked values: "
            + "; ".join(mismatches)
        )
    if not (
        0 < config.warmup_steps < config.task_start_step < config.total_steps
        and config.batch_size > 0
        and config.batch_size % 2 == 0
        and config.hidden_dim > 0
        and config.latent_dim > 0
        and config.crossfit_folds == 3
        and config.nystrom_components > 0
        and config.reference_per_class > 0
        and config.base_generation_per_class > 0
        and config.classifier_seed >= 0
    ):
        raise ProtocolError("Uniform-B task-geometry sizes/steps are invalid.")
    numeric = (
        config.learning_rate,
        config.weight_decay,
        config.beta_final,
        config.gradient_clip_norm,
        config.geco_target_slack,
        config.geco_ema_decay,
        config.geco_dual_step_size,
        config.nystrom_gamma,
        config.teacher_c,
        config.hessian_ridge,
        config.hessian_eigenfloor,
        config.cdf_temperature,
        config.mmd_weight,
        config.margin_weight,
        config.gradient_weight,
        config.task_weight,
        config.classifier_c,
        config.min_effective_rank_ratio,
        config.min_pairwise_distance_ratio,
        config.max_pairwise_distance_ratio,
    )
    if not all(math.isfinite(value) for value in numeric):
        raise ProtocolError("Uniform-B task-geometry numeric value is nonfinite.")
    if (
        config.learning_rate <= 0.0
        or config.weight_decay < 0.0
        or config.beta_final < 0.0
        or config.gradient_clip_norm <= 0.0
        or config.geco_target_slack <= 0.0
        or not 0.0 <= config.geco_ema_decay < 1.0
        or config.geco_dual_step_size <= 0.0
        or config.nystrom_gamma <= 0.0
        or config.teacher_c <= 0.0
        or config.hessian_ridge <= 0.0
        or config.hessian_eigenfloor <= 0.0
        or config.cdf_temperature <= 0.0
        or min(config.mmd_weight, config.margin_weight, config.gradient_weight) < 0.0
        or config.task_weight <= 0.0
        or config.classifier_c <= 0.0
        or not (
            0.0 < config.min_pairwise_distance_ratio
            <= config.max_pairwise_distance_ratio
        )
    ):
        raise ProtocolError("Uniform-B task-geometry numeric range is invalid.")
    if (
        not config.mmd_bandwidth_multipliers
        or any(value <= 0.0 for value in config.mmd_bandwidth_multipliers)
        or not config.cdf_grid_quantiles
        or any(not 0.0 < value < 1.0 for value in config.cdf_grid_quantiles)
        or tuple(sorted(config.cdf_grid_quantiles))
        != config.cdf_grid_quantiles
    ):
        raise ProtocolError("Task-geometry grid/bandwidth contract is invalid.")
    forbidden_text = " ".join(
        [str(config.feature_cache_path), str(config.manifest_path)]
    ).lower()
    if any(token in forbidden_text for token in ("90_oracles", "snapshot", "quarantine")):
        raise ProtocolError("Stage-90/quarantined inputs are forbidden.")
    if config.artifact_root.name != OUTPUT_ARTIFACT_ID and str(
        config.artifact_root
    ).startswith("output://"):
        raise ProtocolError("Unexpected output artifact identity.")


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


def _float_tuple(value: object) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ProtocolError("Expected a numeric list.")
    return tuple(float(item) for item in value)


__all__ = (
    "UniformBTaskGeometryConfig",
    "load_uniform_b_task_geometry_config",
)
