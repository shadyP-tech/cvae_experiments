"""Frozen config contracts for the additive Stage-20 source-inner v2 studies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ....real_features.classifier_reference.artifacts import stable_hash
from ....real_features.classifier_reference.protocol import ProtocolError
from ....real_features.classifier_reference.schemas.midogpp import (
    MIDOGPP_ELIGIBLE_CENTERS,
)
from ...objectives import ISOTROPIC_OBJECTIVE, TASK_FISHER_OBJECTIVE
from ...latent_priors import (
    ACTIVE_UNIT_THRESHOLD,
    CLASS_SEPARATION_THRESHOLD,
    PRIOR_SATURATION_THRESHOLD,
)
from .contracts import (
    FISHER_ALPHAS,
    FISHER_SHRINKAGE_MODE,
    LEARNED_PRIOR_MODE,
    LEARNED_PRIOR_MODEL_FAMILY,
    LEARNED_CONDITIONAL_DIAGONAL_PRIOR,
    PRIOR_ARMS,
    SOURCE_INNER_STUDY_VERSION,
    STANDARD_MODEL_FAMILY,
    STANDARD_NORMAL_PRIOR,
    StudyTrainingVariant,
)


LEARNED_PRIOR_STUDY_NAME = (
    "virchow2_cvae_midogpp_learned_conditional_prior_source_inner_v2"
)
FISHER_SHRINKAGE_STUDY_NAME = (
    "virchow2_cvae_midogpp_task_fisher_shrinkage_source_inner_v2"
)
STUDY_PANEL_SEEDS = (17, 42, 101)
SOURCE_EMPIRICAL_BUDGET_POLICY = "source_empirical_class_counts_from_y_fit"

STANDARD_PRIOR_FAMILY = "standard_normal"
EX_POST_DIAGONAL_PRIOR_FAMILY = "class_conditional_diagonal_total_moment"
LEARNED_PRIOR_FAMILY = "learned_class_conditional_diagonal_gaussian"
PRIOR_LOGVAR_PARAMETERIZATION = "bounded_tanh"
PRIOR_INITIALIZATION = "exact_standard_normal"
RAW_FISHER_FIT_SCOPE = "shared_per_outer_inner"
ALPHA_ZERO_POLICY = "literal_isotropic_metric_none"

_FIXED_MODEL = {
    "expected_feature_dim": 2560,
    "pca_dim": 128,
    "hidden_dim": 512,
    "latent_dim": 32,
    "num_hidden_layers": 2,
    "train_epochs": 100,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "beta_final": 1e-3,
    "kl_warmup_epochs": 25,
    "network_gradient_clip_norm": 5.0,
}


@dataclass(frozen=True)
class SourceInnerStudyConfig:
    name: str
    mode: str
    study_version: str
    artifact_root: Path
    manifest_path: Path
    feature_cache_path: Path
    heldout_centers: tuple[str, ...]
    expected_feature_dim: int
    pca_dim: int
    training_seeds: tuple[int, ...]
    generation_seeds: tuple[int, ...]
    device: str
    hidden_dim: int
    latent_dim: int
    num_hidden_layers: int
    train_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    beta_final: float
    kl_warmup_epochs: int
    network_gradient_clip_norm: float
    generation_budget_policy: str
    minimum_real_bacc: float
    code_version: str

    def training_variant(
        self,
        *,
        model_family: str,
        prior_family: str,
        alpha: float = 0.0,
        raw_fisher_state_hash: str = "none",
        objective_context_hash: str = "none",
    ) -> StudyTrainingVariant:
        """Build one exact variant while preserving the fixed base recipe."""

        resolved_alpha = float(alpha)
        if self.mode == LEARNED_PRIOR_MODE:
            allowed_pairs = {
                (STANDARD_MODEL_FAMILY, STANDARD_NORMAL_PRIOR),
                (
                    LEARNED_PRIOR_MODEL_FAMILY,
                    LEARNED_CONDITIONAL_DIAGONAL_PRIOR,
                ),
            }
            if (str(model_family), str(prior_family)) not in allowed_pairs:
                raise ProtocolError(
                    "Learned-prior training variants must be the A or E model/prior pair."
                )
            objective_id = ISOTROPIC_OBJECTIVE
        else:
            if (
                str(model_family) != STANDARD_MODEL_FAMILY
                or str(prior_family) != STANDARD_NORMAL_PRIOR
            ):
                raise ProtocolError(
                    "Fisher-shrinkage variants fix the standard model and prior."
                )
            if resolved_alpha not in FISHER_ALPHAS:
                raise ProtocolError("Fisher-shrinkage variant alpha is outside the panel.")
            objective_id = (
                ISOTROPIC_OBJECTIVE
                if resolved_alpha == 0.0
                else TASK_FISHER_OBJECTIVE
            )
        return StudyTrainingVariant(
            study_mode=self.mode,
            study_version=self.study_version,
            model_family=str(model_family),
            prior_family=str(prior_family),
            objective_id=objective_id,
            alpha=resolved_alpha,
            raw_fisher_state_hash=str(raw_fisher_state_hash),
            objective_context_hash=str(objective_context_hash),
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            num_hidden_layers=self.num_hidden_layers,
            train_epochs=self.train_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            beta_final=self.beta_final,
            kl_warmup_epochs=self.kl_warmup_epochs,
            network_gradient_clip_norm=self.network_gradient_clip_norm,
            prior_learning_rate_multiplier=float(
                getattr(self, "prior_optimizer_learning_rate_multiplier", 1.0)
            ),
            prior_weight_decay=float(
                getattr(self, "prior_optimizer_weight_decay", 0.0)
            ),
            prior_gradient_clip_norm=float(
                getattr(self, "prior_gradient_clip_norm", 5.0)
            ),
        )


@dataclass(frozen=True)
class LearnedConditionalPriorStudyConfig(SourceInnerStudyConfig):
    arms: tuple[str, ...]
    standard_prior_family: str
    ex_post_prior_family: str
    learned_prior_family: str
    prior_logvar_parameterization: str
    prior_logvar_bound: float
    prior_initialization: str
    prior_optimizer_learning_rate_multiplier: float
    prior_optimizer_weight_decay: float
    prior_gradient_clip_norm: float
    e_vs_a_min_mean_delta: float
    e_vs_c_min_mean_delta: float
    min_inner_wins: int
    safety_max_bacc_regression: float


@dataclass(frozen=True)
class TaskFisherShrinkageStudyConfig(SourceInnerStudyConfig):
    alphas: tuple[float, ...]
    raw_fisher_fit_scope: str
    alpha_zero_policy: str
    fisher_min_mean_delta: float
    min_inner_wins: int
    tie_margin: float
    safety_max_bacc_regression: float


def load_source_inner_study_config(
    path: str | Path,
    *,
    expected_mode: str | None = None,
) -> LearnedConditionalPriorStudyConfig | TaskFisherShrinkageStudyConfig:
    """Load and fail-closed validate one production v2 study config."""

    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - project dependency
        raise RuntimeError("Source-inner study configs require PyYAML.") from exc

    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ProtocolError("Source-inner study config must be a mapping.")
    experiment = _mapping(payload.get("experiment"), "experiment")
    inputs = _mapping(payload.get("inputs"), "inputs")
    run = _mapping(payload.get("run"), "run")
    model = _mapping(payload.get("model"), "model")
    generation = _mapping(payload.get("generation"), "generation")
    decisions = _mapping(payload.get("decisions"), "decisions")
    claim_boundary = _mapping(payload.get("claim_boundary"), "claim_boundary")

    mode = str(experiment.get("mode", ""))
    if mode not in {LEARNED_PRIOR_MODE, FISHER_SHRINKAGE_MODE}:
        raise ProtocolError(f"Unsupported source-inner study mode: {mode!r}")
    if expected_mode is not None and mode != expected_mode:
        raise ProtocolError(
            f"Expected source-inner study mode {expected_mode!r}, got {mode!r}."
        )
    _validate_claim_boundary(claim_boundary, mode=mode)

    heldout_raw = run.get("heldout_centers", "all")
    heldout_centers = (
        MIDOGPP_ELIGIBLE_CENTERS
        if str(heldout_raw).lower() == "all"
        else tuple(str(value) for value in heldout_raw)  # type: ignore[union-attr]
    )
    base = config_path.parent
    common: dict[str, object] = {
        "name": str(experiment.get("name", "")),
        "mode": mode,
        "study_version": str(experiment.get("study_version", "")),
        "artifact_root": _path(base, _required_string(experiment, "artifact_root")),
        "manifest_path": _path(base, _required_string(inputs, "manifest_path")),
        "feature_cache_path": _path(
            base, _required_string(inputs, "feature_cache_path")
        ),
        "heldout_centers": tuple(heldout_centers),
        "expected_feature_dim": int(
            run.get("expected_feature_dim", _FIXED_MODEL["expected_feature_dim"])
        ),
        "pca_dim": int(model.get("pca_dim", _FIXED_MODEL["pca_dim"])),
        "training_seeds": _int_tuple(run.get("training_seeds", STUDY_PANEL_SEEDS)),
        "generation_seeds": _int_tuple(
            run.get("generation_seeds", STUDY_PANEL_SEEDS)
        ),
        "device": str(run.get("device", "cpu")),
        "hidden_dim": int(model.get("hidden_dim", _FIXED_MODEL["hidden_dim"])),
        "latent_dim": int(model.get("latent_dim", _FIXED_MODEL["latent_dim"])),
        "num_hidden_layers": int(
            model.get("num_hidden_layers", _FIXED_MODEL["num_hidden_layers"])
        ),
        "train_epochs": int(
            model.get("train_epochs", _FIXED_MODEL["train_epochs"])
        ),
        "batch_size": int(model.get("batch_size", _FIXED_MODEL["batch_size"])),
        "learning_rate": float(
            model.get("learning_rate", _FIXED_MODEL["learning_rate"])
        ),
        "weight_decay": float(
            model.get("weight_decay", _FIXED_MODEL["weight_decay"])
        ),
        "beta_final": float(model.get("beta_final", _FIXED_MODEL["beta_final"])),
        "kl_warmup_epochs": int(
            model.get("kl_warmup_epochs", _FIXED_MODEL["kl_warmup_epochs"])
        ),
        "network_gradient_clip_norm": float(
            model.get(
                "gradient_clip_norm", _FIXED_MODEL["network_gradient_clip_norm"]
            )
        ),
        "generation_budget_policy": str(generation.get("budget_policy", "")),
        "minimum_real_bacc": float(
            decisions.get("minimum_real_bacc", float("nan"))
        ),
        "code_version": str(experiment.get("code_version", "")),
    }

    if mode == LEARNED_PRIOR_MODE:
        objective = _mapping(payload.get("objective"), "objective")
        if (
            objective.get("family") != ISOTROPIC_OBJECTIVE
            or objective.get("fixed_across_arms") is not True
        ):
            raise ProtocolError(
                "Learned-prior study must fix the stochastic isotropic objective."
            )
        prior = _mapping(payload.get("prior"), "prior")
        config: LearnedConditionalPriorStudyConfig | TaskFisherShrinkageStudyConfig
        config = LearnedConditionalPriorStudyConfig(
            **common,
            arms=tuple(str(value) for value in prior.get("arms", ())),
            standard_prior_family=str(prior.get("standard_family", "")),
            ex_post_prior_family=str(prior.get("ex_post_family", "")),
            learned_prior_family=str(prior.get("learned_family", "")),
            prior_logvar_parameterization=str(
                prior.get("logvar_parameterization", "")
            ),
            prior_logvar_bound=float(prior.get("logvar_bound", float("nan"))),
            prior_initialization=str(prior.get("initialization", "")),
            prior_optimizer_learning_rate_multiplier=float(
                prior.get("optimizer_learning_rate_multiplier", float("nan"))
            ),
            prior_optimizer_weight_decay=float(
                prior.get("optimizer_weight_decay", float("nan"))
            ),
            prior_gradient_clip_norm=float(
                prior.get("prior_gradient_clip_norm", float("nan"))
            ),
            e_vs_a_min_mean_delta=float(
                decisions.get("e_vs_a_min_mean_delta", float("nan"))
            ),
            e_vs_c_min_mean_delta=float(
                decisions.get("e_vs_c_min_mean_delta", float("nan"))
            ),
            min_inner_wins=int(decisions.get("min_inner_wins", 0)),
            safety_max_bacc_regression=float(
                decisions.get("safety_max_bacc_regression", float("nan"))
            ),
        )
    else:
        prior = _mapping(payload.get("prior"), "prior")
        if (
            prior.get("family") != STANDARD_NORMAL_PRIOR
            or prior.get("fixed_across_alphas") is not True
        ):
            raise ProtocolError(
                "Fisher-shrinkage study must fix the standard-normal prior."
            )
        objective = _mapping(payload.get("objective"), "objective")
        config = TaskFisherShrinkageStudyConfig(
            **common,
            alphas=_float_tuple(objective.get("alphas", ())),
            raw_fisher_fit_scope=str(objective.get("raw_fisher_fit_scope", "")),
            alpha_zero_policy=str(objective.get("alpha_zero_policy", "")),
            fisher_min_mean_delta=float(
                decisions.get("fisher_min_mean_delta", float("nan"))
            ),
            min_inner_wins=int(decisions.get("min_inner_wins", 0)),
            tie_margin=float(decisions.get("tie_margin", float("nan"))),
            safety_max_bacc_regression=float(
                decisions.get("safety_max_bacc_regression", float("nan"))
            ),
        )
    _validate(config)
    return config


def load_prior_study_config(path: str | Path) -> LearnedConditionalPriorStudyConfig:
    config = load_source_inner_study_config(path, expected_mode=LEARNED_PRIOR_MODE)
    if not isinstance(config, LearnedConditionalPriorStudyConfig):  # pragma: no cover
        raise ProtocolError("Loaded study config has the wrong prior-study type.")
    return config


def load_fisher_study_config(path: str | Path) -> TaskFisherShrinkageStudyConfig:
    config = load_source_inner_study_config(path, expected_mode=FISHER_SHRINKAGE_MODE)
    if not isinstance(config, TaskFisherShrinkageStudyConfig):  # pragma: no cover
        raise ProtocolError("Loaded study config has the wrong Fisher-study type.")
    return config


def decision_contract_payload(config: SourceInnerStudyConfig) -> dict[str, object]:
    if isinstance(config, LearnedConditionalPriorStudyConfig):
        return {
            "schema_version": "midogpp_learned_prior_decision_contract_v2",
            "e_vs_a_min_mean_delta": config.e_vs_a_min_mean_delta,
            "e_vs_a_comparator": ">=",
            "e_vs_c_min_mean_delta": config.e_vs_c_min_mean_delta,
            "e_vs_c_comparator": ">",
            "min_inner_wins": config.min_inner_wins,
            "win_comparator": ">0",
            "safety_max_bacc_regression": config.safety_max_bacc_regression,
            "minimum_real_bacc": config.minimum_real_bacc,
            "generation_aggregation": "mean_within_inner_before_center_comparison",
            "training_seed_consensus": "all_three_exact",
        }
    if isinstance(config, TaskFisherShrinkageStudyConfig):
        return {
            "schema_version": "midogpp_fisher_shrinkage_decision_contract_v2",
            "fisher_min_mean_delta": config.fisher_min_mean_delta,
            "fisher_comparator": ">",
            "min_inner_wins": config.min_inner_wins,
            "win_comparator": ">0",
            "tie_margin": config.tie_margin,
            "tie_comparator": "<=",
            "tie_break": "smallest_alpha",
            "safety_max_bacc_regression": config.safety_max_bacc_regression,
            "minimum_real_bacc": config.minimum_real_bacc,
            "generation_aggregation": "mean_within_inner_before_center_comparison",
            "training_seed_consensus": "exact_same_nonzero_alpha_all_three",
        }
    raise ProtocolError("Unsupported source-inner study config type.")


def decision_contract_hash(config: SourceInnerStudyConfig) -> str:
    return stable_hash(decision_contract_payload(config))


def study_contract_payload(config: SourceInnerStudyConfig) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "midogpp_source_inner_study_contract_v2",
        "name": config.name,
        "mode": config.mode,
        "study_version": config.study_version,
        "heldout_centers": list(config.heldout_centers),
        "device": config.device,
        "expected_feature_dim": config.expected_feature_dim,
        "pca_dim": config.pca_dim,
        "training_seeds": list(config.training_seeds),
        "generation_seeds": list(config.generation_seeds),
        "base_model": {
            "hidden_dim": config.hidden_dim,
            "latent_dim": config.latent_dim,
            "num_hidden_layers": config.num_hidden_layers,
            "train_epochs": config.train_epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "beta_final": config.beta_final,
            "kl_warmup_epochs": config.kl_warmup_epochs,
            "network_gradient_clip_norm": config.network_gradient_clip_norm,
        },
        "generation_budget_policy": config.generation_budget_policy,
        "minimum_real_bacc": config.minimum_real_bacc,
        "claim_boundary": {
            "claim_scope": "cvae_source_inner_study_only",
            "target_evaluation_data_used": False,
            "may_change_existing_consensus_locks": False,
            "may_feed_recipe_selection": False,
            "may_feed_deployable_selection": False,
        },
        "decision_contract": decision_contract_payload(config),
        "code_version": config.code_version,
    }
    if isinstance(config, LearnedConditionalPriorStudyConfig):
        payload["fixed_axis"] = {
            "objective_family": ISOTROPIC_OBJECTIVE,
            "fixed_across_arms": True,
        }
        payload["prior"] = {
            "arms": list(config.arms),
            "standard_family": config.standard_prior_family,
            "ex_post_family": config.ex_post_prior_family,
            "learned_family": config.learned_prior_family,
            "logvar_parameterization": config.prior_logvar_parameterization,
            "logvar_bound": config.prior_logvar_bound,
            "initialization": config.prior_initialization,
            "optimizer_learning_rate_multiplier": (
                config.prior_optimizer_learning_rate_multiplier
            ),
            "optimizer_weight_decay": config.prior_optimizer_weight_decay,
            "prior_gradient_clip_norm": config.prior_gradient_clip_norm,
        }
        payload["learned_prior_diagnostics"] = {
            "posterior_scope": "source_fit_rows_only",
            "active_unit_definition": (
                "0.5*sum_y Var[(mu_q-mu_p_y)/sigma_p_y] per dimension"
            ),
            "active_unit_threshold": ACTIVE_UNIT_THRESHOLD,
            "class_separation_definition": (
                "latent_normalized_symmetric_KL_between_class_priors"
            ),
            "class_separation_threshold": CLASS_SEPARATION_THRESHOLD,
            "saturation_definition": "any_abs_effective_logvar_greater_equal_threshold",
            "saturation_threshold": PRIOR_SATURATION_THRESHOLD,
            "eligibility_rule": "finite_and_not_saturated_and_active_unit_count_gt_zero",
            "kl_audit": "per_class_per_dimension_analytic_q_to_p",
            "trajectory_audit": "per_epoch_prior_location_and_logvar_ranges",
        }
    elif isinstance(config, TaskFisherShrinkageStudyConfig):
        payload["fixed_axis"] = {
            "prior_family": STANDARD_NORMAL_PRIOR,
            "fixed_across_alphas": True,
        }
        payload["objective"] = {
            "alphas": list(config.alphas),
            "raw_fisher_fit_scope": config.raw_fisher_fit_scope,
            "alpha_zero_policy": config.alpha_zero_policy,
        }
    return payload


def study_contract_hash(config: SourceInnerStudyConfig) -> str:
    return stable_hash(study_contract_payload(config))


def _validate(config: SourceInnerStudyConfig) -> None:
    expected_name = (
        LEARNED_PRIOR_STUDY_NAME
        if config.mode == LEARNED_PRIOR_MODE
        else FISHER_SHRINKAGE_STUDY_NAME
    )
    if config.name != expected_name:
        raise ProtocolError(
            f"Source-inner study name must be {expected_name!r} for mode {config.mode!r}."
        )
    if config.study_version != SOURCE_INNER_STUDY_VERSION:
        raise ProtocolError("Source-inner studies require study_version='v2'.")
    if not config.code_version:
        raise ProtocolError("Source-inner studies require an explicit code_version.")
    if config.heldout_centers != MIDOGPP_ELIGIBLE_CENTERS:
        raise ProtocolError("Source-inner studies require exact ordered nine-center coverage.")
    if config.training_seeds != STUDY_PANEL_SEEDS:
        raise ProtocolError("Source-inner studies require training seeds (17, 42, 101).")
    if config.generation_seeds != STUDY_PANEL_SEEDS:
        raise ProtocolError("Source-inner studies require generation seeds (17, 42, 101).")
    if config.generation_budget_policy != SOURCE_EMPIRICAL_BUDGET_POLICY:
        raise ProtocolError("Generation must use source empirical class counts from y_fit.")
    if config.minimum_real_bacc != 0.55:
        raise ProtocolError("Source-inner studies require minimum_real_bacc=0.55.")
    observed_model = {
        "expected_feature_dim": config.expected_feature_dim,
        "pca_dim": config.pca_dim,
        "hidden_dim": config.hidden_dim,
        "latent_dim": config.latent_dim,
        "num_hidden_layers": config.num_hidden_layers,
        "train_epochs": config.train_epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "beta_final": config.beta_final,
        "kl_warmup_epochs": config.kl_warmup_epochs,
        "network_gradient_clip_norm": config.network_gradient_clip_norm,
    }
    if observed_model != _FIXED_MODEL:
        raise ProtocolError("Source-inner v2 base CVAE hyperparameters drifted.")
    if isinstance(config, LearnedConditionalPriorStudyConfig):
        if config.arms != PRIOR_ARMS:
            raise ProtocolError("Learned-prior study arms must be (A, C-diag, E).")
        if (
            config.standard_prior_family != STANDARD_PRIOR_FAMILY
            or config.ex_post_prior_family != EX_POST_DIAGONAL_PRIOR_FAMILY
            or config.learned_prior_family != LEARNED_PRIOR_FAMILY
        ):
            raise ProtocolError("Learned-prior family identities drifted.")
        if (
            config.prior_logvar_parameterization != PRIOR_LOGVAR_PARAMETERIZATION
            or config.prior_logvar_bound != 6.0
            or config.prior_initialization != PRIOR_INITIALIZATION
            or config.prior_optimizer_learning_rate_multiplier != 1.0
            or config.prior_optimizer_weight_decay != 0.0
            or config.prior_gradient_clip_norm != 5.0
        ):
            raise ProtocolError("Learned conditional-prior optimization contract drifted.")
        if (
            config.e_vs_a_min_mean_delta != 0.05
            or config.e_vs_c_min_mean_delta != 0.01
            or config.min_inner_wins != 6
            or config.safety_max_bacc_regression != 0.01
        ):
            raise ProtocolError("Learned-prior decision thresholds drifted.")
    elif isinstance(config, TaskFisherShrinkageStudyConfig):
        if config.alphas != FISHER_ALPHAS:
            raise ProtocolError("Fisher-shrinkage alphas must be (0, .05, .10, .25).")
        if (
            config.raw_fisher_fit_scope != RAW_FISHER_FIT_SCOPE
            or config.alpha_zero_policy != ALPHA_ZERO_POLICY
        ):
            raise ProtocolError("Fisher-shrinkage state-sharing contract drifted.")
        if (
            config.fisher_min_mean_delta != 0.01
            or config.min_inner_wins != 6
            or config.tie_margin != 0.01
            or config.safety_max_bacc_regression != 0.01
        ):
            raise ProtocolError("Fisher-shrinkage decision thresholds drifted.")
    else:  # pragma: no cover - guarded by the loader's constructors
        raise ProtocolError("Unknown source-inner study config type.")


def _validate_claim_boundary(claim: Mapping[str, object], *, mode: str) -> None:
    expected_allowed = {
        LEARNED_PRIOR_MODE: (
            "fully nested source-inner learned-prior study evidence only"
        ),
        FISHER_SHRINKAGE_MODE: (
            "fully nested source-inner Task-Fisher shrinkage study evidence only"
        ),
    }[mode]
    expected_forbidden = (
        "recipe adoption, outer-target preservation, expert-bank input, "
        "generation evidence, routing, expert selection, NELBO compatibility, "
        "or downstream utility"
    )
    if (
        claim.get("allowed") != expected_allowed
        or claim.get("forbidden") != expected_forbidden
        or claim.get("target_evaluation_data_used") is not False
        or claim.get("may_change_existing_consensus_locks") is not False
        or claim.get("may_feed_recipe_selection") is not False
        or claim.get("may_feed_deployable_selection") is not False
    ):
        raise ProtocolError("Source-inner v2 claim boundary drifted or became adoptive.")


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{name} must be a mapping.")
    return value


def _required_string(payload: Mapping[str, object], key: str) -> str:
    value = str(payload.get(key, ""))
    if not value:
        raise ProtocolError(f"Missing required config key: {key}.")
    return value


def _path(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _int_tuple(value: object) -> tuple[int, ...]:
    try:
        values = (
            tuple(int(part.strip()) for part in value.split(",") if part.strip())
            if isinstance(value, str)
            else tuple(int(item) for item in value)  # type: ignore[union-attr]
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolError("Seed lists must contain integers.") from exc
    if len(set(values)) != len(values):
        raise ProtocolError("Seed lists must not contain duplicates.")
    return values


def _float_tuple(value: object) -> tuple[float, ...]:
    try:
        values = tuple(float(item) for item in value)  # type: ignore[union-attr]
    except (TypeError, ValueError) as exc:
        raise ProtocolError("Alpha lists must contain numbers.") from exc
    if len(set(values)) != len(values):
        raise ProtocolError("Alpha lists must not contain duplicates.")
    return values
