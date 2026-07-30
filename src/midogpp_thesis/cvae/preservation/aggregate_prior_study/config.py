"""Fail-closed config contract for the independent-source v3 study."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from ....real_features.classifier_reference.artifacts import stable_hash
from ....real_features.classifier_reference.protocol import ProtocolError
from ....real_features.classifier_reference.schemas.midogpp import (
    MIDOGPP_ELIGIBLE_CENTERS,
)
from .contracts import (
    ARMS,
    CLAIM_SCOPE,
    MODE,
    PRIMARY_ARM,
    STUDY_NAME,
    STUDY_VERSION,
    arm_contract,
)


TRAINING_SEEDS = (17, 42, 101)
GENERATION_SEEDS = (17, 42, 101)


@dataclass(frozen=True)
class AggregatePriorStudyConfig:
    name: str
    mode: str
    study_version: str
    artifact_root: Path
    code_version: str
    manifest_path: Path
    feature_cache_path: Path
    heldout_centers: tuple[str, ...]
    expected_feature_dim: int
    training_seeds: tuple[int, ...]
    generation_seeds: tuple[int, ...]
    device: str
    pca_dim: int
    latent_dim: int
    hidden_dim: int
    num_hidden_layers: int
    warmup_epochs: int
    continuation_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    beta_final: float
    kl_warmup_epochs: int
    arms: tuple[str, ...]
    n_components: int
    mixture_rank: int
    weight_floor: float
    variance_floor: float
    covariance_shrinkage: float
    refit_interval_epochs: int
    final_stabilization_epochs: int
    minimum_component_rows: int
    minimum_component_cases: int
    maximum_condition_number: float
    prior_fitting: str
    optimizer_updates_prior_parameters: bool
    mixture_rate_semantics: str
    geco_target_policy: str
    geco_target_slack: float
    geco_ema_decay: float
    geco_dual_step_size: float
    geco_initial_multiplier: float
    geco_minimum_multiplier: float
    geco_maximum_multiplier: float
    geco_uses_inner_or_outer_data: bool
    generation_policy: str
    generation_per_class: int
    same_budget_and_rng_across_arms: bool
    source_or_target_prevalence_used: bool
    inverse_transform_to_common_frame: bool
    primary_arm: str
    min_mean_bacc_delta_vs_sf: float
    min_inner_wins: int
    max_worst_inner_regression: float
    max_training_seed_range: float
    max_posterior_bacc_regression: float
    min_prior_posterior_gap_reduction: float
    require_positive_delta_vs_kf: bool
    require_positive_delta_vs_sg: bool
    equal_weight_sources_then_inner_centers: bool
    claim_scope: str
    may_feed_recipe_selection: bool
    may_feed_deployable_selection: bool
    separate_promotion_artifact_required: bool

    @property
    def train_epochs(self) -> int:
        return self.warmup_epochs + self.continuation_epochs

    @property
    def contract_hash(self) -> str:
        payload = asdict(self)
        # Run-local filesystem roots are operational provenance, not method
        # identity. The actual input content is bound separately by manifest
        # and feature-cache hashes in every source-expert training key.
        for key in ("artifact_root", "manifest_path", "feature_cache_path"):
            payload.pop(key)
        payload["arm_contract"] = arm_contract()
        return stable_hash(payload)


def load_aggregate_prior_study_config(
    path: str | Path,
) -> AggregatePriorStudyConfig:
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Aggregate-prior configs require PyYAML.") from exc

    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ProtocolError("Aggregate-prior config must be a mapping.")
    experiment = _mapping(payload.get("experiment"), "experiment")
    inputs = _mapping(payload.get("inputs"), "inputs")
    run = _mapping(payload.get("run"), "run")
    model = _mapping(payload.get("model"), "model")
    prior = _mapping(payload.get("prior"), "prior")
    geco = _mapping(payload.get("geco"), "geco")
    generation = _mapping(payload.get("generation"), "generation")
    decisions = _mapping(payload.get("decisions"), "decisions")
    claim = _mapping(payload.get("claim_boundary"), "claim_boundary")

    heldout_raw = run.get("heldout_centers", "all")
    heldout = (
        MIDOGPP_ELIGIBLE_CENTERS
        if str(heldout_raw).lower() == "all"
        else tuple(str(value) for value in heldout_raw)  # type: ignore[union-attr]
    )
    base = config_path.parent
    config = AggregatePriorStudyConfig(
        name=str(experiment.get("name", "")),
        mode=str(experiment.get("mode", "")),
        study_version=str(experiment.get("study_version", "")),
        artifact_root=_path(base, _required(experiment, "artifact_root")),
        code_version=str(experiment.get("code_version", "")),
        manifest_path=_path(base, _required(inputs, "manifest_path")),
        feature_cache_path=_path(base, _required(inputs, "feature_cache_path")),
        heldout_centers=tuple(heldout),
        expected_feature_dim=int(run.get("expected_feature_dim", 0)),
        training_seeds=_int_tuple(run.get("training_seeds", ())),
        generation_seeds=_int_tuple(run.get("generation_seeds", ())),
        device=str(run.get("device", "")),
        pca_dim=int(model.get("pca_dim", 0)),
        latent_dim=int(model.get("latent_dim", 0)),
        hidden_dim=int(model.get("hidden_dim", 0)),
        num_hidden_layers=int(model.get("num_hidden_layers", 0)),
        warmup_epochs=int(model.get("warmup_epochs", 0)),
        continuation_epochs=int(model.get("continuation_epochs", 0)),
        batch_size=int(model.get("batch_size", 0)),
        learning_rate=float(model.get("learning_rate", float("nan"))),
        weight_decay=float(model.get("weight_decay", float("nan"))),
        gradient_clip_norm=float(
            model.get("gradient_clip_norm", float("nan"))
        ),
        beta_final=float(model.get("beta_final", float("nan"))),
        kl_warmup_epochs=int(model.get("kl_warmup_epochs", 0)),
        arms=tuple(str(value) for value in prior.get("arms", ())),
        n_components=int(prior.get("n_components", 0)),
        mixture_rank=int(prior.get("rank", 0)),
        weight_floor=float(prior.get("weight_floor", float("nan"))),
        variance_floor=float(prior.get("variance_floor", float("nan"))),
        covariance_shrinkage=float(
            prior.get("covariance_shrinkage", float("nan"))
        ),
        refit_interval_epochs=int(prior.get("refit_interval_epochs", 0)),
        final_stabilization_epochs=int(
            prior.get("final_stabilization_epochs", 0)
        ),
        minimum_component_rows=int(prior.get("minimum_component_rows", 0)),
        minimum_component_cases=int(prior.get("minimum_component_cases", 0)),
        maximum_condition_number=float(
            prior.get("maximum_condition_number", float("nan"))
        ),
        prior_fitting=str(prior.get("fitting", "")),
        optimizer_updates_prior_parameters=_required_bool(
            prior,
            "optimizer_updates_prior_parameters",
        ),
        mixture_rate_semantics=str(prior.get("rate_semantics", "")),
        geco_target_policy=str(geco.get("target_policy", "")),
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
        geco_uses_inner_or_outer_data=_required_bool(
            geco,
            "target_uses_inner_or_outer_data",
        ),
        generation_policy=str(generation.get("policy", "")),
        generation_per_class=int(generation.get("per_source_per_class", 0)),
        same_budget_and_rng_across_arms=_required_bool(
            generation,
            "same_budget_and_rng_across_arms",
        ),
        source_or_target_prevalence_used=_required_bool(
            generation,
            "source_or_target_prevalence_used",
        ),
        inverse_transform_to_common_frame=_required_bool(
            generation,
            "inverse_transform_to_common_virchow2_frame",
        ),
        primary_arm=str(decisions.get("primary_arm", "")),
        min_mean_bacc_delta_vs_sf=float(
            decisions.get("min_mean_bacc_delta_vs_sf", float("nan"))
        ),
        min_inner_wins=int(decisions.get("min_inner_wins", 0)),
        max_worst_inner_regression=float(
            decisions.get("max_worst_inner_regression", float("nan"))
        ),
        max_training_seed_range=float(
            decisions.get("max_training_seed_range", float("nan"))
        ),
        max_posterior_bacc_regression=float(
            decisions.get("max_posterior_bacc_regression", float("nan"))
        ),
        min_prior_posterior_gap_reduction=float(
            decisions.get(
                "min_prior_posterior_gap_reduction",
                float("nan"),
            )
        ),
        require_positive_delta_vs_kf=_required_bool(
            decisions,
            "require_positive_delta_vs_kf",
        ),
        require_positive_delta_vs_sg=_required_bool(
            decisions,
            "require_positive_delta_vs_sg",
        ),
        equal_weight_sources_then_inner_centers=_required_bool(
            decisions,
            "equal_weight_sources_then_inner_centers",
        ),
        claim_scope=str(claim.get("claim_scope", "")),
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
    _validate(config, claim=claim)
    return config


def _validate(
    config: AggregatePriorStudyConfig,
    *,
    claim: Mapping[str, object],
) -> None:
    exact = {
        "name": (config.name, STUDY_NAME),
        "mode": (config.mode, MODE),
        "study_version": (config.study_version, STUDY_VERSION),
        "heldout_centers": (
            config.heldout_centers,
            MIDOGPP_ELIGIBLE_CENTERS,
        ),
        "expected_feature_dim": (config.expected_feature_dim, 2560),
        "training_seeds": (config.training_seeds, TRAINING_SEEDS),
        "generation_seeds": (config.generation_seeds, GENERATION_SEEDS),
        "pca_dim": (config.pca_dim, 128),
        "latent_dim": (config.latent_dim, 32),
        "hidden_dim": (config.hidden_dim, 512),
        "num_hidden_layers": (config.num_hidden_layers, 2),
        "warmup_epochs": (config.warmup_epochs, 25),
        "continuation_epochs": (config.continuation_epochs, 75),
        "batch_size": (config.batch_size, 128),
        "learning_rate": (config.learning_rate, 0.001),
        "weight_decay": (config.weight_decay, 0.0001),
        "gradient_clip_norm": (config.gradient_clip_norm, 5.0),
        "beta_final": (config.beta_final, 0.001),
        "kl_warmup_epochs": (config.kl_warmup_epochs, 25),
        "arms": (config.arms, ARMS),
        "n_components": (config.n_components, 2),
        "mixture_rank": (config.mixture_rank, 2),
        "weight_floor": (config.weight_floor, 0.05),
        "variance_floor": (config.variance_floor, 0.0001),
        "covariance_shrinkage": (config.covariance_shrinkage, 0.10),
        "refit_interval_epochs": (config.refit_interval_epochs, 5),
        "final_stabilization_epochs": (config.final_stabilization_epochs, 5),
        "minimum_component_rows": (config.minimum_component_rows, 8),
        "minimum_component_cases": (config.minimum_component_cases, 2),
        "maximum_condition_number": (config.maximum_condition_number, 1e6),
        "prior_fitting": (
            config.prior_fitting,
            "deterministic_source_aggregate_coordinate_updates",
        ),
        "optimizer_updates_prior_parameters": (
            config.optimizer_updates_prior_parameters,
            False,
        ),
        "mixture_rate_semantics": (
            config.mixture_rate_semantics,
            "mixture_KL_upper_bound_not_exact_NELBO",
        ),
        "geco_target_policy": (
            config.geco_target_policy,
            "source_warmup_mean_mse_times_slack",
        ),
        "geco_target_slack": (config.geco_target_slack, 1.05),
        "geco_ema_decay": (config.geco_ema_decay, 0.99),
        "geco_dual_step_size": (config.geco_dual_step_size, 0.001),
        "geco_initial_multiplier": (config.geco_initial_multiplier, 1.0),
        "geco_minimum_multiplier": (config.geco_minimum_multiplier, 1e-6),
        "geco_maximum_multiplier": (config.geco_maximum_multiplier, 1e6),
        "geco_uses_inner_or_outer_data": (
            config.geco_uses_inner_or_outer_data,
            False,
        ),
        "generation_policy": (
            config.generation_policy,
            "fixed_balanced_per_source",
        ),
        "generation_per_class": (config.generation_per_class, 256),
        "same_budget_and_rng_across_arms": (
            config.same_budget_and_rng_across_arms,
            True,
        ),
        "source_or_target_prevalence_used": (
            config.source_or_target_prevalence_used,
            False,
        ),
        "inverse_transform_to_common_frame": (
            config.inverse_transform_to_common_frame,
            True,
        ),
        "primary_arm": (config.primary_arm, PRIMARY_ARM),
        "min_mean_bacc_delta_vs_sf": (
            config.min_mean_bacc_delta_vs_sf,
            0.02,
        ),
        "min_inner_wins": (config.min_inner_wins, 6),
        "max_worst_inner_regression": (
            config.max_worst_inner_regression,
            -0.01,
        ),
        "max_training_seed_range": (config.max_training_seed_range, 0.05),
        "max_posterior_bacc_regression": (
            config.max_posterior_bacc_regression,
            -0.01,
        ),
        "min_prior_posterior_gap_reduction": (
            config.min_prior_posterior_gap_reduction,
            0.01,
        ),
        "require_positive_delta_vs_kf": (
            config.require_positive_delta_vs_kf,
            True,
        ),
        "require_positive_delta_vs_sg": (
            config.require_positive_delta_vs_sg,
            True,
        ),
        "equal_weight_sources_then_inner_centers": (
            config.equal_weight_sources_then_inner_centers,
            True,
        ),
        "claim_scope": (config.claim_scope, CLAIM_SCOPE),
        "may_feed_recipe_selection": (
            config.may_feed_recipe_selection,
            False,
        ),
        "may_feed_deployable_selection": (
            config.may_feed_deployable_selection,
            False,
        ),
        "separate_promotion_artifact_required": (
            config.separate_promotion_artifact_required,
            True,
        ),
    }
    drift = {
        key: {"observed": observed, "expected": expected}
        for key, (observed, expected) in exact.items()
        if observed != expected
    }
    if drift:
        raise ProtocolError(f"Aggregate-prior v3 config drifted: {drift}")
    if not config.code_version or not config.device:
        raise ProtocolError("Aggregate-prior v3 requires code_version and device.")
    if (
        claim.get("allowed")
        != "fully nested independent-source prior-mismatch study evidence only"
        or claim.get("forbidden")
        != (
            "recipe adoption, outer-target preservation, expert-bank input, "
            "generation evidence, routing, expert selection, NELBO "
            "compatibility, or downstream utility"
        )
        or claim.get("outer_target_rows_used") is not False
        or claim.get("inner_rows_used_for_fit") is not False
        or claim.get("inner_labels_used_for_scoring_only") is not True
        or claim.get("independent_source_experts_required") is not True
    ):
        raise ProtocolError("Aggregate-prior v3 claim firewall drifted.")


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{name} must be a mapping.")
    return value


def _required(payload: Mapping[str, object], key: str) -> str:
    value = str(payload.get(key, ""))
    if not value:
        raise ProtocolError(f"Missing required config key: {key}.")
    return value


def _required_bool(payload: Mapping[str, object], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ProtocolError(f"Config key {key} must be boolean.")
    return value


def _path(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _int_tuple(value: object) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in value)  # type: ignore[union-attr]
    except (TypeError, ValueError) as exc:
        raise ProtocolError("Seed lists must contain integers.") from exc
    if len(values) != len(set(values)):
        raise ProtocolError("Seed lists must not contain duplicates.")
    return values
