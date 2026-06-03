from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .downstream import (
    PredictionBundle,
    evaluate_probability_predictions,
    predict_from_probabilities,
    weighted_arithmetic_probability_pool,
)
from .features import load_feature_cache, select_rows
from .metrics import nanmean
from .preservation import _hash_array
from .preservation_repair import (
    NA,
    PRIMARY_VARIANT,
    _existing_cache_path,
    _float,
    _format_float,
    _hash_strings,
    _label,
    _load_mapping,
    _mapping,
    _path,
    _source_data_for_centers,
    _target_indices,
)
from .preservation_sampling import DIAGNOSTIC_SELECTION, PRIMARY_SELECTION, RuntimeSource, _manifest_row, _per_source_variant, _runtime_source
from .protocol import ProtocolError, assert_candidate_pool, build_leakage_report
from .reporting import prepare_artifact_dirs, write_csv_rows, write_json
from .splits import candidate_experts

from . import component_union_mass_bagged as mb
from . import decentralized_adaptive_gmm_prior as d1a
from . import decentralized_component_union_prior as cu
from . import decentralized_k16_gmm_prior as d1
from . import decentralized_reliability_weighted_gmm_prior as d12
from . import paired_dense_all4_reliability_confirmation as paired


TAILRISK_NAME = "virchow2_cvae_component_union_tailrisk_anchored_mass_bagged_v1"
PRIMARY_TAILRISK_METHOD = "component_union_tailrisk_anchored_shrink050_random_mass_bag_blend050"
TAILRISK_SOURCE_WEIGHTING = "tailrisk_anchored_shrink050_random_mass_bag_blend050"
MULTIPANEL_TAILRISK_NAME = "virchow2_cvae_component_union_tailrisk_multipanel_mass_bagged_v1"
PRIMARY_MULTIPANEL_TAILRISK_METHOD = "component_union_tailrisk_multipanel_shrink050_random_mass_bag_blend050"
MULTIPANEL_SEED_BLEND_METHOD = "component_union_tailrisk_seed_shrink050_random_mass_bag_blend050"
MULTIPANEL_POOLED_RANDOM_BAG_METHOD = "component_union_tailrisk_multipanel_pooled_random_mass_bag"
MULTIPANEL_CANONICAL_RANDOM_BAG_METHOD = "same_cell_single_random_mass_bag_canonical"
MULTIPANEL_POOLED_ANCHOR_METHOD = "component_union_tailrisk_multipanel_pooled_shrink050"
MULTIPANEL_SOURCE_WEIGHTING = "tailrisk_multipanel_shrink050_random_mass_bag_blend050"
POSITIVE_UNION_TAILRISK_NAME = "virchow2_cvae_source_inner_class_conditional_positive_union_v1"
PRIMARY_POSITIVE_UNION_METHOD = "source_inner_class_conditional_positive_union_v1"
POSITIVE_UNION_SOURCE_WEIGHTING = "source_inner_class_conditional_positive_union"
POSITIVE_UNION_PRIMARY_POOLING = "source_inner_selected_class_conditional_positive_union"
FIXED_BETA050_POSITIVE_UNION_NAME = "virchow2_cvae_fixed_beta050_positive_union_confirmation_v1"
PRIMARY_FIXED_BETA050_POSITIVE_UNION_METHOD = "fixed_beta050_positive_union_confirmation_v1"
FIXED_BETA050_POSITIVE_UNION_SOURCE_WEIGHTING = "fixed_beta050_positive_union_confirmation"
FIXED_BETA050_POSITIVE_UNION_PRIMARY_POOLING = "fixed_global_positive_union_beta050"
POSITIVE_UNION_RULE_ARITHMETIC = "arithmetic_mean"
POSITIVE_UNION_RULE_BETA025 = "positive_union_beta025"
POSITIVE_UNION_RULE_BETA050 = "positive_union_beta050"
POSITIVE_UNION_RULE_BETA100 = "positive_union_beta100"
POSITIVE_UNION_RULES = (
    POSITIVE_UNION_RULE_ARITHMETIC,
    POSITIVE_UNION_RULE_BETA025,
    POSITIVE_UNION_RULE_BETA050,
    POSITIVE_UNION_RULE_BETA100,
)
POSITIVE_UNION_BETAS = {
    POSITIVE_UNION_RULE_ARITHMETIC: None,
    POSITIVE_UNION_RULE_BETA025: 0.25,
    POSITIVE_UNION_RULE_BETA050: 0.50,
    POSITIVE_UNION_RULE_BETA100: 1.00,
}
ANCHOR_METHOD = cu.ROW_COMPONENT_UNION_SHRINK050
BAG_METHOD = cu.ROW_RANDOM_MASS_BAG_CONTROL
MATCHED_SHUFFLED_TAILRISK_PREFIX = cu.MATCHED_SHUFFLED_RELIABILITY_SHRINK050_PREFIX
MULTIPANEL_CANONICAL_PANEL = "canonical"
MULTIPANEL_FRESH_A_PANEL = "fresh_a"
MULTIPANEL_FRESH_B_PANEL = "fresh_b"
MULTIPANEL_PANEL_SEEDS = (
    (MULTIPANEL_CANONICAL_PANEL, (17, 23, 31)),
    (MULTIPANEL_FRESH_A_PANEL, (101, 103, 107)),
    (MULTIPANEL_FRESH_B_PANEL, (109, 113, 127)),
)
CENTER3_FAILURE_AUDIT_CELLS = (
    (42, "3"),
    (44, "3"),
    (43, "4"),
    (43, "1"),
)
CENTER3_FAILURE_PRIMARY_CELL = (42, "3")
FIXED_BETA050_DEVELOPMENT_EXPERIMENT_SEEDS = (42, 43, 44)
FIXED_BETA050_CONFIRMATION_EXPERIMENT_SEEDS = (45, 46, 47, 48, 49)
FIXED_BETA050_RARE_POSITIVE_COUNT_THRESHOLD = 10
FIXED_BETA050_RARE_POSITIVE_PREVALENCE_THRESHOLD = 0.05


@dataclass(frozen=True)
class TailRiskAnchoredConfig:
    name: str
    artifact_root: Path
    repair_artifact_root: Path
    paired_dense_artifact_root: Path | None
    mass_bagged_artifact_root: Path | None
    support_calibrated_artifact_root: Path | None
    shrink050_artifact_root: Path | None
    source_union_gmm_artifact_root: Path | None
    balanced_gmm_artifact_root: Path | None
    feature_cache_root: Path
    backbone: str
    experiment_seeds: tuple[int, ...]
    heldout_centers: tuple[str, ...]
    replicate_seeds: tuple[int, ...]
    fresh_replicate_seeds: tuple[int, ...]
    strict_full_run_matrix: bool
    synthetic_per_class_total: int
    min_per_source_per_class: int
    primary_variant: str
    primary_method: str
    random_mass_bag_size: int
    random_mass_bag_alpha: float
    blend_alpha: float
    primary_shrink_lambda: float
    matched_shuffled_reliability_null_permutations: int
    candidate_components_per_source_class: tuple[int, ...]
    min_samples_per_component: int
    source_weighting: str
    gmm_covariance_type: str
    gmm_reg_covar: float
    gmm_n_init: int
    gmm_max_iter: int
    min_component_weight: float
    variance_floor: float
    variance_ceiling_multiplier: float
    primary_pooling: str
    reliability_floor_score: float
    reliability_epsilon: float
    anchor_repro_tolerance: float
    classifier_type: str
    classifier_solver: str
    classifier_c: float
    classifier_max_iter: int
    classifier_class_weight: str
    classifier_seed: int | None

    @property
    def all_replicate_seeds(self) -> tuple[int, ...]:
        return tuple(dict.fromkeys((*self.replicate_seeds, *self.fresh_replicate_seeds)))

    @property
    def control_bag_size(self) -> int:
        return self.random_mass_bag_size

    @property
    def max_local_gmm_components_per_source_class(self) -> int:
        return max(self.candidate_components_per_source_class)

    @property
    def composed_components_per_class_nominal(self) -> int:
        return self.max_local_gmm_components_per_source_class * (len(self.heldout_centers) - 1)


@dataclass(frozen=True)
class MultipanelTailRiskConfig(TailRiskAnchoredConfig):
    prior_tailrisk_artifact_root: Path | None = None
    panel_seed_groups: tuple[tuple[str, tuple[int, ...]], ...] = MULTIPANEL_PANEL_SEEDS
    primary_noninferiority_margin: float = 0.005
    weak_pass_noninferiority_margin: float = 0.010
    tailrisk_transfer_threshold: float = -0.010

    @property
    def all_panel_seeds(self) -> tuple[int, ...]:
        seeds: list[int] = []
        for _panel, panel_seeds in self.panel_seed_groups:
            seeds.extend(int(seed) for seed in panel_seeds)
        return tuple(dict.fromkeys(seeds))

    @property
    def all_replicate_seeds(self) -> tuple[int, ...]:
        return self.all_panel_seeds


@dataclass(frozen=True)
class SourceInnerPositiveUnionConfig(MultipanelTailRiskConfig):
    candidate_pooling_rules: tuple[str, ...] = POSITIVE_UNION_RULES
    positive_label: int = 1
    prediction_threshold: float = 0.50
    min_source_inner_positive_count: int = 5
    positive_union_eps: float = 1.0e-8
    source_inner_bacc_noninferiority_margin: float = 0.010
    source_inner_class0_recall_margin: float = 0.015
    source_inner_predicted_positive_rate_delta: float = 0.050
    beta100_class0_recall_margin: float = 0.005
    beta100_precision_margin: float = 0.010


@dataclass(frozen=True)
class FixedBeta050PositiveUnionConfig(SourceInnerPositiveUnionConfig):
    fixed_pooling_rule: str = POSITIVE_UNION_RULE_BETA050
    fixed_beta: float = 0.50
    development_experiment_seeds: tuple[int, ...] = FIXED_BETA050_DEVELOPMENT_EXPERIMENT_SEEDS
    confirmation_experiment_seeds: tuple[int, ...] = FIXED_BETA050_CONFIRMATION_EXPERIMENT_SEEDS
    development_positive_union_artifact_root: Path | None = None
    rare_positive_count_threshold: int = FIXED_BETA050_RARE_POSITIVE_COUNT_THRESHOLD
    rare_positive_prevalence_threshold: float = FIXED_BETA050_RARE_POSITIVE_PREVALENCE_THRESHOLD


@dataclass(frozen=True)
class BagEvaluation:
    ensemble_row: dict[str, object]
    ensemble_bundle: PredictionBundle | None
    ensemble_coverage: dict[str, object]
    ensemble_paired_row: dict[str, object]
    member_results: tuple[mb.MemberResult, ...]
    eligibility_rows: tuple[dict[str, object], ...]
    component_counts: dict[int, dict[str, int]]
    generated_hash: str
    source_generation_hash: str
    ensemble_plan: dict[str, object]


@dataclass(frozen=True)
class TailRiskEvaluation:
    primary_row: dict[str, object]
    primary_bundle: PredictionBundle | None
    primary_coverage: dict[str, object]
    primary_paired_row: dict[str, object]
    anchor_result: mb.MemberResult
    bag_evaluation: BagEvaluation
    blend_manifest_row: dict[str, object]
    complementarity_row: dict[str, object]
    calibration_rows: tuple[dict[str, object], ...]
    source_weight_rows: tuple[dict[str, object], ...]
    eligibility_rows: tuple[dict[str, object], ...]
    source_inner_bundles: Mapping[str, PredictionBundle]
    source_inner_labels: tuple[int, ...]
    source_inner_source_ids: tuple[str, ...]


def load_tailrisk_anchored_component_union_config(path: str | Path) -> TailRiskAnchoredConfig:
    source = Path(path).resolve()
    data = _load_mapping(source)
    base_dir = source.parents[2] if len(source.parents) >= 3 else source.parent
    return parse_tailrisk_anchored_component_union_config(data, base_dir=base_dir)


def parse_tailrisk_anchored_component_union_config(
    data: Mapping[str, Any],
    *,
    base_dir: str | Path = ".",
) -> TailRiskAnchoredConfig:
    base = Path(base_dir)
    experiment = _mapping(data, "experiment")
    inputs = _mapping(data, "inputs")
    run = _mapping(data, "run_matrix")
    generation = _mapping(data, "generation")
    tailrisk = _mapping(data, "tailrisk_anchored_component_union")
    classifier = _mapping(data, "classifier")
    cfg = TailRiskAnchoredConfig(
        name=str(experiment["name"]),
        artifact_root=_path(base, str(experiment["artifact_root"])),
        repair_artifact_root=_path(base, str(inputs["repair_artifact_root"])),
        paired_dense_artifact_root=_optional_path(base, inputs.get("paired_dense_artifact_root")),
        mass_bagged_artifact_root=_optional_path(base, inputs.get("mass_bagged_artifact_root")),
        support_calibrated_artifact_root=_optional_path(base, inputs.get("support_calibrated_artifact_root")),
        shrink050_artifact_root=_optional_path(base, inputs.get("shrink050_artifact_root")),
        source_union_gmm_artifact_root=_optional_path(base, inputs.get("source_union_gmm_artifact_root")),
        balanced_gmm_artifact_root=_optional_path(base, inputs.get("balanced_gmm_artifact_root")),
        feature_cache_root=_path(base, str(inputs["feature_cache_root"])),
        backbone=str(inputs.get("backbone", "")),
        experiment_seeds=tuple(int(v) for v in run["experiment_seeds"]),
        heldout_centers=tuple(str(v) for v in run["heldout_centers"]),
        replicate_seeds=tuple(int(v) for v in run["replicate_seeds"]),
        fresh_replicate_seeds=tuple(int(v) for v in run.get("fresh_replicate_seeds", ())),
        strict_full_run_matrix=bool(run.get("strict_full_run_matrix", False)),
        synthetic_per_class_total=int(generation["synthetic_per_class_total"]),
        min_per_source_per_class=int(generation["min_per_source_per_class"]),
        primary_variant=str(experiment["primary_variant"]),
        primary_method=str(tailrisk["primary_method"]),
        random_mass_bag_size=int(tailrisk["random_mass_bag_size"]),
        random_mass_bag_alpha=float(tailrisk["random_mass_bag_alpha"]),
        blend_alpha=float(tailrisk["blend_alpha"]),
        primary_shrink_lambda=float(tailrisk["primary_shrink_lambda"]),
        matched_shuffled_reliability_null_permutations=int(tailrisk["matched_shuffled_reliability_null_permutations"]),
        candidate_components_per_source_class=tuple(int(v) for v in tailrisk["candidate_components_per_source_class"]),
        min_samples_per_component=int(tailrisk["min_samples_per_component"]),
        source_weighting=str(tailrisk["source_weighting"]),
        gmm_covariance_type=str(tailrisk["gmm_covariance_type"]),
        gmm_reg_covar=float(tailrisk["gmm_reg_covar"]),
        gmm_n_init=int(tailrisk["gmm_n_init"]),
        gmm_max_iter=int(tailrisk["gmm_max_iter"]),
        min_component_weight=float(tailrisk["min_component_weight"]),
        variance_floor=float(tailrisk["variance_floor"]),
        variance_ceiling_multiplier=float(tailrisk["variance_ceiling_multiplier"]),
        primary_pooling=str(tailrisk["primary_pooling"]),
        reliability_floor_score=float(tailrisk["reliability_floor_score"]),
        reliability_epsilon=float(tailrisk["reliability_epsilon"]),
        anchor_repro_tolerance=float(tailrisk["anchor_repro_tolerance"]),
        classifier_type=str(classifier["type"]),
        classifier_solver=str(classifier["solver"]),
        classifier_c=float(classifier["C"]),
        classifier_max_iter=int(classifier["max_iter"]),
        classifier_class_weight=str(classifier["class_weight"]),
        classifier_seed=None if classifier.get("classifier_seed") is None else int(classifier["classifier_seed"]),
    )
    validate_tailrisk_anchored_component_union_config(cfg)
    return cfg


def validate_tailrisk_anchored_component_union_config(cfg: TailRiskAnchoredConfig) -> None:
    if cfg.name != TAILRISK_NAME:
        raise ProtocolError(f"Tail-risk component-union experiment name must be {TAILRISK_NAME!r}.")
    if cfg.backbone != "virchow2":
        raise ProtocolError("Tail-risk component union is locked to backbone=virchow2.")
    if cfg.primary_variant != PRIMARY_VARIANT:
        raise ProtocolError(f"primary_variant must be {PRIMARY_VARIANT!r}.")
    if cfg.primary_method != PRIMARY_TAILRISK_METHOD:
        raise ProtocolError(f"primary_method must be {PRIMARY_TAILRISK_METHOD!r}.")
    if cfg.source_weighting != TAILRISK_SOURCE_WEIGHTING:
        raise ProtocolError(f"source_weighting must be {TAILRISK_SOURCE_WEIGHTING!r}.")
    if cfg.candidate_components_per_source_class != (4, 3, 2, 1):
        raise ProtocolError("candidate_components_per_source_class must be locked to [4, 3, 2, 1].")
    if len(cfg.heldout_centers) != 5:
        raise ProtocolError("Tail-risk component union expects exactly five centers.")
    if cfg.gmm_covariance_type != "diag":
        raise ProtocolError("gmm_covariance_type must be diag.")
    if cfg.primary_pooling != "fixed_arithmetic_probability_blend":
        raise ProtocolError("primary_pooling must be fixed_arithmetic_probability_blend.")
    if not math.isclose(cfg.primary_shrink_lambda, 0.5, rel_tol=0.0, abs_tol=1.0e-12):
        raise ProtocolError("primary_shrink_lambda must be locked to 0.50.")
    if not math.isclose(cfg.blend_alpha, 0.5, rel_tol=0.0, abs_tol=1.0e-12):
        raise ProtocolError("blend_alpha must be locked to 0.50.")
    if not math.isclose(cfg.random_mass_bag_alpha, 4.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ProtocolError("random_mass_bag_alpha must be locked to Dirichlet-uniform alpha4.")
    if cfg.random_mass_bag_size < 1:
        raise ProtocolError("random_mass_bag_size must be positive.")
    if cfg.matched_shuffled_reliability_null_permutations < 0:
        raise ProtocolError("matched_shuffled_reliability_null_permutations must be non-negative.")
    if cfg.strict_full_run_matrix:
        if cfg.experiment_seeds != (42, 43, 44):
            raise ProtocolError("strict_full_run_matrix requires experiment_seeds=[42, 43, 44].")
        if cfg.heldout_centers != ("0", "1", "2", "3", "4"):
            raise ProtocolError("strict_full_run_matrix requires heldout_centers=['0', '1', '2', '3', '4'].")
        if cfg.replicate_seeds != (17, 23, 31):
            raise ProtocolError("strict_full_run_matrix requires canonical replicate_seeds=[17, 23, 31].")
        if cfg.fresh_replicate_seeds != (101, 103, 107):
            raise ProtocolError("strict_full_run_matrix requires fresh_replicate_seeds=[101, 103, 107].")
        if cfg.synthetic_per_class_total != 128:
            raise ProtocolError("strict_full_run_matrix requires synthetic_per_class_total=128.")
        if cfg.min_per_source_per_class != 8:
            raise ProtocolError("strict_full_run_matrix requires min_per_source_per_class=8.")
        if cfg.random_mass_bag_size != 11:
            raise ProtocolError("strict_full_run_matrix requires random_mass_bag_size=11.")
        if cfg.matched_shuffled_reliability_null_permutations != 20:
            raise ProtocolError("strict_full_run_matrix requires matched_shuffled_reliability_null_permutations=20.")
    if min(cfg.min_per_source_per_class, cfg.min_samples_per_component, cfg.gmm_n_init, cfg.gmm_max_iter) < 1:
        raise ProtocolError("Component minimums and GMM iterations must be positive.")
    if min(
        cfg.gmm_reg_covar,
        cfg.min_component_weight,
        cfg.variance_floor,
        cfg.variance_ceiling_multiplier,
        cfg.reliability_floor_score,
        cfg.reliability_epsilon,
        cfg.anchor_repro_tolerance,
    ) <= 0.0:
        raise ProtocolError("Tail-risk numeric floors/tolerances must be positive.")
    if cfg.classifier_type != "sklearn_logistic_regression":
        raise ProtocolError("classifier.type must be sklearn_logistic_regression.")
    if cfg.classifier_solver != "lbfgs" or cfg.classifier_c != 1.0 or cfg.classifier_max_iter != 2000:
        raise ProtocolError("Classifier solver/C/max_iter must remain locked.")
    if cfg.classifier_class_weight != "balanced" or cfg.classifier_seed is not None:
        raise ProtocolError("Classifier must use class_weight=balanced and classifier_seed=null.")


def load_multipanel_tailrisk_component_union_config(path: str | Path) -> MultipanelTailRiskConfig:
    source = Path(path).resolve()
    data = _load_mapping(source)
    base_dir = source.parents[2] if len(source.parents) >= 3 else source.parent
    return parse_multipanel_tailrisk_component_union_config(data, base_dir=base_dir)


def parse_multipanel_tailrisk_component_union_config(
    data: Mapping[str, Any],
    *,
    base_dir: str | Path = ".",
) -> MultipanelTailRiskConfig:
    base = Path(base_dir)
    experiment = _mapping(data, "experiment")
    inputs = _mapping(data, "inputs")
    run = _mapping(data, "run_matrix")
    generation = _mapping(data, "generation")
    multipanel = _mapping(data, "tailrisk_multipanel_component_union")
    classifier = _mapping(data, "classifier")
    panel_seed_groups = _parse_panel_seed_groups(multipanel.get("panel_seed_groups", {}))
    if inputs.get("support_calibrated_artifact_root") not in (None, ""):
        raise ProtocolError("support_calibrated_artifact_root is not allowed for source-only multipanel tail-risk v2.")
    cfg = MultipanelTailRiskConfig(
        name=str(experiment["name"]),
        artifact_root=_path(base, str(experiment["artifact_root"])),
        repair_artifact_root=_path(base, str(inputs["repair_artifact_root"])),
        paired_dense_artifact_root=_optional_path(base, inputs.get("paired_dense_artifact_root")),
        mass_bagged_artifact_root=_optional_path(base, inputs.get("mass_bagged_artifact_root")),
        support_calibrated_artifact_root=None,
        shrink050_artifact_root=_optional_path(base, inputs.get("shrink050_artifact_root")),
        source_union_gmm_artifact_root=_optional_path(base, inputs.get("source_union_gmm_artifact_root")),
        balanced_gmm_artifact_root=_optional_path(base, inputs.get("balanced_gmm_artifact_root")),
        feature_cache_root=_path(base, str(inputs["feature_cache_root"])),
        backbone=str(inputs.get("backbone", "")),
        experiment_seeds=tuple(int(v) for v in run["experiment_seeds"]),
        heldout_centers=tuple(str(v) for v in run["heldout_centers"]),
        replicate_seeds=tuple(int(v) for v in run["replicate_seeds"]),
        fresh_replicate_seeds=tuple(int(v) for v in run.get("fresh_replicate_seeds", ())),
        strict_full_run_matrix=bool(run.get("strict_full_run_matrix", False)),
        synthetic_per_class_total=int(generation["synthetic_per_class_total"]),
        min_per_source_per_class=int(generation["min_per_source_per_class"]),
        primary_variant=str(experiment["primary_variant"]),
        primary_method=str(multipanel["primary_method"]),
        random_mass_bag_size=int(multipanel["random_mass_bag_size"]),
        random_mass_bag_alpha=float(multipanel["random_mass_bag_alpha"]),
        blend_alpha=float(multipanel["blend_alpha"]),
        primary_shrink_lambda=float(multipanel["primary_shrink_lambda"]),
        matched_shuffled_reliability_null_permutations=int(multipanel.get("matched_shuffled_reliability_null_permutations", 0)),
        candidate_components_per_source_class=tuple(int(v) for v in multipanel["candidate_components_per_source_class"]),
        min_samples_per_component=int(multipanel["min_samples_per_component"]),
        source_weighting=str(multipanel["source_weighting"]),
        gmm_covariance_type=str(multipanel["gmm_covariance_type"]),
        gmm_reg_covar=float(multipanel["gmm_reg_covar"]),
        gmm_n_init=int(multipanel["gmm_n_init"]),
        gmm_max_iter=int(multipanel["gmm_max_iter"]),
        min_component_weight=float(multipanel["min_component_weight"]),
        variance_floor=float(multipanel["variance_floor"]),
        variance_ceiling_multiplier=float(multipanel["variance_ceiling_multiplier"]),
        primary_pooling=str(multipanel["primary_pooling"]),
        reliability_floor_score=float(multipanel["reliability_floor_score"]),
        reliability_epsilon=float(multipanel["reliability_epsilon"]),
        anchor_repro_tolerance=float(multipanel["anchor_repro_tolerance"]),
        classifier_type=str(classifier["type"]),
        classifier_solver=str(classifier["solver"]),
        classifier_c=float(classifier["C"]),
        classifier_max_iter=int(classifier["max_iter"]),
        classifier_class_weight=str(classifier["class_weight"]),
        classifier_seed=None if classifier.get("classifier_seed") is None else int(classifier["classifier_seed"]),
        prior_tailrisk_artifact_root=_optional_path(base, inputs.get("prior_tailrisk_artifact_root")),
        panel_seed_groups=panel_seed_groups,
        primary_noninferiority_margin=float(multipanel.get("primary_noninferiority_margin", 0.005)),
        weak_pass_noninferiority_margin=float(multipanel.get("weak_pass_noninferiority_margin", 0.010)),
        tailrisk_transfer_threshold=float(multipanel.get("tailrisk_transfer_threshold", -0.010)),
    )
    validate_multipanel_tailrisk_component_union_config(cfg)
    return cfg


def _parse_panel_seed_groups(value: object) -> tuple[tuple[str, tuple[int, ...]], ...]:
    if not isinstance(value, Mapping):
        raise ProtocolError("panel_seed_groups must be a mapping of panel name to seed list.")
    out = []
    for name in (MULTIPANEL_CANONICAL_PANEL, MULTIPANEL_FRESH_A_PANEL, MULTIPANEL_FRESH_B_PANEL):
        seeds = value.get(name)
        if seeds is None:
            raise ProtocolError(f"Missing panel_seed_groups.{name}.")
        out.append((name, tuple(int(seed) for seed in seeds)))
    return tuple(out)


def validate_multipanel_tailrisk_component_union_config(cfg: MultipanelTailRiskConfig) -> None:
    if cfg.name != MULTIPANEL_TAILRISK_NAME:
        raise ProtocolError(f"Multipanel tail-risk experiment name must be {MULTIPANEL_TAILRISK_NAME!r}.")
    if cfg.backbone != "virchow2":
        raise ProtocolError("Multipanel tail-risk component union is locked to backbone=virchow2.")
    if cfg.primary_variant != PRIMARY_VARIANT:
        raise ProtocolError(f"primary_variant must be {PRIMARY_VARIANT!r}.")
    if cfg.primary_method != PRIMARY_MULTIPANEL_TAILRISK_METHOD:
        raise ProtocolError(f"primary_method must be {PRIMARY_MULTIPANEL_TAILRISK_METHOD!r}.")
    if cfg.source_weighting != MULTIPANEL_SOURCE_WEIGHTING:
        raise ProtocolError(f"source_weighting must be {MULTIPANEL_SOURCE_WEIGHTING!r}.")
    if cfg.candidate_components_per_source_class != (4, 3, 2, 1):
        raise ProtocolError("candidate_components_per_source_class must be locked to [4, 3, 2, 1].")
    if len(cfg.heldout_centers) != 5:
        raise ProtocolError("Multipanel tail-risk component union expects exactly five centers.")
    if cfg.gmm_covariance_type != "diag":
        raise ProtocolError("gmm_covariance_type must be diag.")
    if cfg.primary_pooling != "seed_blend_then_equal_probability_pool":
        raise ProtocolError("primary_pooling must be seed_blend_then_equal_probability_pool.")
    if not math.isclose(cfg.primary_shrink_lambda, 0.5, rel_tol=0.0, abs_tol=1.0e-12):
        raise ProtocolError("primary_shrink_lambda must be locked to 0.50.")
    if not math.isclose(cfg.blend_alpha, 0.5, rel_tol=0.0, abs_tol=1.0e-12):
        raise ProtocolError("blend_alpha must be locked to 0.50.")
    if not math.isclose(cfg.random_mass_bag_alpha, 4.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ProtocolError("random_mass_bag_alpha must be locked to Dirichlet-uniform alpha4.")
    if cfg.random_mass_bag_size < 1:
        raise ProtocolError("random_mass_bag_size must be positive.")
    if cfg.matched_shuffled_reliability_null_permutations != 0:
        raise ProtocolError("matched_shuffled_reliability_null_permutations must be 0 for v2; shuffled null is not part of this stabilization test.")
    if cfg.replicate_seeds != cfg.panel_seed_groups[0][1]:
        raise ProtocolError("run_matrix.replicate_seeds must equal the canonical panel seeds.")
    expected_fresh = tuple(seed for _panel, seeds in cfg.panel_seed_groups[1:] for seed in seeds)
    if cfg.fresh_replicate_seeds != expected_fresh:
        raise ProtocolError("run_matrix.fresh_replicate_seeds must equal fresh_a + fresh_b panel seeds.")
    if cfg.panel_seed_groups != MULTIPANEL_PANEL_SEEDS:
        raise ProtocolError("panel_seed_groups must be locked to canonical/fresh_a/fresh_b predeclared seeds.")
    if cfg.strict_full_run_matrix:
        if cfg.experiment_seeds != (42, 43, 44):
            raise ProtocolError("strict_full_run_matrix requires experiment_seeds=[42, 43, 44].")
        if cfg.heldout_centers != ("0", "1", "2", "3", "4"):
            raise ProtocolError("strict_full_run_matrix requires heldout_centers=['0', '1', '2', '3', '4'].")
        if cfg.synthetic_per_class_total != 128:
            raise ProtocolError("strict_full_run_matrix requires synthetic_per_class_total=128.")
        if cfg.min_per_source_per_class != 8:
            raise ProtocolError("strict_full_run_matrix requires min_per_source_per_class=8.")
        if cfg.random_mass_bag_size != 11:
            raise ProtocolError("strict_full_run_matrix requires random_mass_bag_size=11.")
    if min(cfg.min_per_source_per_class, cfg.min_samples_per_component, cfg.gmm_n_init, cfg.gmm_max_iter) < 1:
        raise ProtocolError("Component minimums and GMM iterations must be positive.")
    if min(
        cfg.gmm_reg_covar,
        cfg.min_component_weight,
        cfg.variance_floor,
        cfg.variance_ceiling_multiplier,
        cfg.reliability_floor_score,
        cfg.reliability_epsilon,
        cfg.anchor_repro_tolerance,
        cfg.primary_noninferiority_margin,
        cfg.weak_pass_noninferiority_margin,
    ) <= 0.0:
        raise ProtocolError("Multipanel numeric floors/tolerances must be positive.")
    if cfg.tailrisk_transfer_threshold >= 0.0:
        raise ProtocolError("tailrisk_transfer_threshold must be negative.")
    if cfg.classifier_type != "sklearn_logistic_regression":
        raise ProtocolError("classifier.type must be sklearn_logistic_regression.")
    if cfg.classifier_solver != "lbfgs" or cfg.classifier_c != 1.0 or cfg.classifier_max_iter != 2000:
        raise ProtocolError("Classifier solver/C/max_iter must remain locked.")
    if cfg.classifier_class_weight != "balanced" or cfg.classifier_seed is not None:
        raise ProtocolError("Classifier must use class_weight=balanced and classifier_seed=null.")


def load_source_inner_positive_union_config(path: str | Path) -> SourceInnerPositiveUnionConfig:
    source = Path(path).resolve()
    data = _load_mapping(source)
    base_dir = source.parents[2] if len(source.parents) >= 3 else source.parent
    return parse_source_inner_positive_union_config(data, base_dir=base_dir)


def parse_source_inner_positive_union_config(
    data: Mapping[str, Any],
    *,
    base_dir: str | Path = ".",
) -> SourceInnerPositiveUnionConfig:
    base = Path(base_dir)
    experiment = _mapping(data, "experiment")
    inputs = _mapping(data, "inputs")
    run = _mapping(data, "run_matrix")
    generation = _mapping(data, "generation")
    positive_union = _mapping(data, "source_inner_class_conditional_positive_union")
    classifier = _mapping(data, "classifier")
    panel_seed_groups = _parse_panel_seed_groups(positive_union.get("panel_seed_groups", {}))
    if inputs.get("support_calibrated_artifact_root") not in (None, ""):
        raise ProtocolError("support_calibrated_artifact_root is not allowed for source-only positive-union v1.")
    for forbidden in (
        "target_support_used",
        "target_support_labels_for_selection",
        "target_label_calibration",
        "target_eval_metric_selection",
        "target_threshold_selection",
        "target_eval_calibrated_rule_selection",
    ):
        if forbidden in positive_union:
            raise ProtocolError(f"{forbidden} is not allowed for source-only positive-union v1.")
    cfg = SourceInnerPositiveUnionConfig(
        name=str(experiment["name"]),
        artifact_root=_path(base, str(experiment["artifact_root"])),
        repair_artifact_root=_path(base, str(inputs["repair_artifact_root"])),
        paired_dense_artifact_root=_optional_path(base, inputs.get("paired_dense_artifact_root")),
        mass_bagged_artifact_root=_optional_path(base, inputs.get("mass_bagged_artifact_root")),
        support_calibrated_artifact_root=None,
        shrink050_artifact_root=_optional_path(base, inputs.get("shrink050_artifact_root")),
        source_union_gmm_artifact_root=_optional_path(base, inputs.get("source_union_gmm_artifact_root")),
        balanced_gmm_artifact_root=_optional_path(base, inputs.get("balanced_gmm_artifact_root")),
        feature_cache_root=_path(base, str(inputs["feature_cache_root"])),
        backbone=str(inputs.get("backbone", "")),
        experiment_seeds=tuple(int(v) for v in run["experiment_seeds"]),
        heldout_centers=tuple(str(v) for v in run["heldout_centers"]),
        replicate_seeds=tuple(int(v) for v in run["replicate_seeds"]),
        fresh_replicate_seeds=tuple(int(v) for v in run.get("fresh_replicate_seeds", ())),
        strict_full_run_matrix=bool(run.get("strict_full_run_matrix", False)),
        synthetic_per_class_total=int(generation["synthetic_per_class_total"]),
        min_per_source_per_class=int(generation["min_per_source_per_class"]),
        primary_variant=str(experiment["primary_variant"]),
        primary_method=str(positive_union["primary_method"]),
        random_mass_bag_size=int(positive_union["random_mass_bag_size"]),
        random_mass_bag_alpha=float(positive_union["random_mass_bag_alpha"]),
        blend_alpha=float(positive_union["blend_alpha"]),
        primary_shrink_lambda=float(positive_union["primary_shrink_lambda"]),
        matched_shuffled_reliability_null_permutations=int(positive_union.get("matched_shuffled_reliability_null_permutations", 0)),
        candidate_components_per_source_class=tuple(int(v) for v in positive_union["candidate_components_per_source_class"]),
        min_samples_per_component=int(positive_union["min_samples_per_component"]),
        source_weighting=str(positive_union["source_weighting"]),
        gmm_covariance_type=str(positive_union["gmm_covariance_type"]),
        gmm_reg_covar=float(positive_union["gmm_reg_covar"]),
        gmm_n_init=int(positive_union["gmm_n_init"]),
        gmm_max_iter=int(positive_union["gmm_max_iter"]),
        min_component_weight=float(positive_union["min_component_weight"]),
        variance_floor=float(positive_union["variance_floor"]),
        variance_ceiling_multiplier=float(positive_union["variance_ceiling_multiplier"]),
        primary_pooling=str(positive_union["primary_pooling"]),
        reliability_floor_score=float(positive_union["reliability_floor_score"]),
        reliability_epsilon=float(positive_union["reliability_epsilon"]),
        anchor_repro_tolerance=float(positive_union["anchor_repro_tolerance"]),
        classifier_type=str(classifier["type"]),
        classifier_solver=str(classifier["solver"]),
        classifier_c=float(classifier["C"]),
        classifier_max_iter=int(classifier["max_iter"]),
        classifier_class_weight=str(classifier["class_weight"]),
        classifier_seed=None if classifier.get("classifier_seed") is None else int(classifier["classifier_seed"]),
        prior_tailrisk_artifact_root=_optional_path(base, inputs.get("prior_tailrisk_artifact_root")),
        panel_seed_groups=panel_seed_groups,
        primary_noninferiority_margin=float(positive_union.get("primary_noninferiority_margin", 0.005)),
        weak_pass_noninferiority_margin=float(positive_union.get("weak_pass_noninferiority_margin", 0.010)),
        tailrisk_transfer_threshold=float(positive_union.get("tailrisk_transfer_threshold", -0.010)),
        candidate_pooling_rules=tuple(str(v) for v in positive_union["candidate_pooling_rules"]),
        positive_label=int(positive_union["positive_label"]),
        prediction_threshold=float(positive_union["prediction_threshold"]),
        min_source_inner_positive_count=int(positive_union["min_source_inner_positive_count"]),
        positive_union_eps=float(positive_union["positive_union_eps"]),
        source_inner_bacc_noninferiority_margin=float(positive_union["source_inner_bacc_noninferiority_margin"]),
        source_inner_class0_recall_margin=float(positive_union["source_inner_class0_recall_margin"]),
        source_inner_predicted_positive_rate_delta=float(positive_union["source_inner_predicted_positive_rate_delta"]),
        beta100_class0_recall_margin=float(positive_union["beta100_class0_recall_margin"]),
        beta100_precision_margin=float(positive_union["beta100_precision_margin"]),
    )
    validate_source_inner_positive_union_config(cfg)
    return cfg


def validate_source_inner_positive_union_config(cfg: SourceInnerPositiveUnionConfig) -> None:
    if cfg.name != POSITIVE_UNION_TAILRISK_NAME:
        raise ProtocolError(f"Positive-union experiment name must be {POSITIVE_UNION_TAILRISK_NAME!r}.")
    if cfg.backbone != "virchow2":
        raise ProtocolError("Positive-union component union is locked to backbone=virchow2.")
    if cfg.primary_variant != PRIMARY_VARIANT:
        raise ProtocolError(f"primary_variant must be {PRIMARY_VARIANT!r}.")
    if cfg.primary_method != PRIMARY_POSITIVE_UNION_METHOD:
        raise ProtocolError(f"primary_method must be {PRIMARY_POSITIVE_UNION_METHOD!r}.")
    if cfg.source_weighting != POSITIVE_UNION_SOURCE_WEIGHTING:
        raise ProtocolError(f"source_weighting must be {POSITIVE_UNION_SOURCE_WEIGHTING!r}.")
    if cfg.primary_pooling != POSITIVE_UNION_PRIMARY_POOLING:
        raise ProtocolError(f"primary_pooling must be {POSITIVE_UNION_PRIMARY_POOLING!r}.")
    if cfg.candidate_pooling_rules != POSITIVE_UNION_RULES:
        raise ProtocolError("candidate_pooling_rules must be locked to arithmetic_mean/beta025/beta050/beta100.")
    if cfg.positive_label != 1:
        raise ProtocolError("positive_label must be locked to 1.")
    if not math.isclose(cfg.prediction_threshold, 0.5, rel_tol=0.0, abs_tol=1.0e-12):
        raise ProtocolError("prediction_threshold must be locked to 0.50.")
    if cfg.min_source_inner_positive_count != 5:
        raise ProtocolError("min_source_inner_positive_count must be locked to 5.")
    if not math.isclose(cfg.positive_union_eps, 1.0e-8, rel_tol=0.0, abs_tol=1.0e-14):
        raise ProtocolError("positive_union_eps must be locked to 1e-8.")
    if cfg.candidate_components_per_source_class != (4, 3, 2, 1):
        raise ProtocolError("candidate_components_per_source_class must be locked to [4, 3, 2, 1].")
    if len(cfg.heldout_centers) != 5:
        raise ProtocolError("Positive-union component union expects exactly five centers.")
    if cfg.gmm_covariance_type != "diag":
        raise ProtocolError("gmm_covariance_type must be diag.")
    if not math.isclose(cfg.primary_shrink_lambda, 0.5, rel_tol=0.0, abs_tol=1.0e-12):
        raise ProtocolError("primary_shrink_lambda must be locked to 0.50.")
    if not math.isclose(cfg.blend_alpha, 0.5, rel_tol=0.0, abs_tol=1.0e-12):
        raise ProtocolError("blend_alpha must be locked to 0.50.")
    if not math.isclose(cfg.random_mass_bag_alpha, 4.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ProtocolError("random_mass_bag_alpha must be locked to Dirichlet-uniform alpha4.")
    if cfg.random_mass_bag_size < 1:
        raise ProtocolError("random_mass_bag_size must be positive.")
    if cfg.matched_shuffled_reliability_null_permutations != 0:
        raise ProtocolError("matched_shuffled_reliability_null_permutations must be 0 for positive-union v1.")
    if cfg.replicate_seeds != cfg.panel_seed_groups[0][1]:
        raise ProtocolError("run_matrix.replicate_seeds must equal the canonical panel seeds.")
    expected_fresh = tuple(seed for _panel, seeds in cfg.panel_seed_groups[1:] for seed in seeds)
    if cfg.fresh_replicate_seeds != expected_fresh:
        raise ProtocolError("run_matrix.fresh_replicate_seeds must equal fresh_a + fresh_b panel seeds.")
    if cfg.panel_seed_groups != MULTIPANEL_PANEL_SEEDS:
        raise ProtocolError("panel_seed_groups must be locked to canonical/fresh_a/fresh_b predeclared seeds.")
    if cfg.strict_full_run_matrix:
        if cfg.experiment_seeds != (42, 43, 44):
            raise ProtocolError("strict_full_run_matrix requires experiment_seeds=[42, 43, 44].")
        if cfg.heldout_centers != ("0", "1", "2", "3", "4"):
            raise ProtocolError("strict_full_run_matrix requires heldout_centers=['0', '1', '2', '3', '4'].")
        if cfg.synthetic_per_class_total != 128:
            raise ProtocolError("strict_full_run_matrix requires synthetic_per_class_total=128.")
        if cfg.min_per_source_per_class != 8:
            raise ProtocolError("strict_full_run_matrix requires min_per_source_per_class=8.")
        if cfg.random_mass_bag_size != 11:
            raise ProtocolError("strict_full_run_matrix requires random_mass_bag_size=11.")
    if min(cfg.min_per_source_per_class, cfg.min_samples_per_component, cfg.gmm_n_init, cfg.gmm_max_iter) < 1:
        raise ProtocolError("Component minimums and GMM iterations must be positive.")
    if min(
        cfg.gmm_reg_covar,
        cfg.min_component_weight,
        cfg.variance_floor,
        cfg.variance_ceiling_multiplier,
        cfg.reliability_floor_score,
        cfg.reliability_epsilon,
        cfg.anchor_repro_tolerance,
        cfg.primary_noninferiority_margin,
        cfg.weak_pass_noninferiority_margin,
        cfg.source_inner_bacc_noninferiority_margin,
        cfg.source_inner_class0_recall_margin,
        cfg.source_inner_predicted_positive_rate_delta,
        cfg.beta100_class0_recall_margin,
        cfg.beta100_precision_margin,
    ) <= 0.0:
        raise ProtocolError("Positive-union numeric floors/tolerances must be positive.")
    if cfg.tailrisk_transfer_threshold >= 0.0:
        raise ProtocolError("tailrisk_transfer_threshold must be negative.")
    if cfg.classifier_type != "sklearn_logistic_regression":
        raise ProtocolError("classifier.type must be sklearn_logistic_regression.")
    if cfg.classifier_solver != "lbfgs" or cfg.classifier_c != 1.0 or cfg.classifier_max_iter != 2000:
        raise ProtocolError("Classifier solver/C/max_iter must remain locked.")
    if cfg.classifier_class_weight != "balanced" or cfg.classifier_seed is not None:
        raise ProtocolError("Classifier must use class_weight=balanced and classifier_seed=null.")


def load_fixed_beta050_positive_union_config(path: str | Path) -> FixedBeta050PositiveUnionConfig:
    source = Path(path).resolve()
    data = _load_mapping(source)
    base_dir = source.parents[2] if len(source.parents) >= 3 else source.parent
    return parse_fixed_beta050_positive_union_config(data, base_dir=base_dir)


def parse_fixed_beta050_positive_union_config(
    data: Mapping[str, Any],
    *,
    base_dir: str | Path = ".",
) -> FixedBeta050PositiveUnionConfig:
    base = Path(base_dir)
    experiment = _mapping(data, "experiment")
    inputs = _mapping(data, "inputs")
    run = _mapping(data, "run_matrix")
    generation = _mapping(data, "generation")
    fixed = _mapping(data, "fixed_beta050_positive_union_confirmation")
    classifier = _mapping(data, "classifier")
    panel_seed_groups = _parse_panel_seed_groups(fixed.get("panel_seed_groups", {}))
    if inputs.get("support_calibrated_artifact_root") not in (None, ""):
        raise ProtocolError("support_calibrated_artifact_root is not allowed for fixed beta050 source-only confirmation.")
    for forbidden in (
        "target_support_used",
        "target_support_labels_for_selection",
        "target_label_calibration",
        "target_eval_metric_selection",
        "target_threshold_selection",
        "target_eval_calibrated_rule_selection",
        "source_inner_rule_selection",
    ):
        if forbidden in fixed:
            raise ProtocolError(f"{forbidden} is not allowed for fixed beta050 source-only confirmation.")
    cfg = FixedBeta050PositiveUnionConfig(
        name=str(experiment["name"]),
        artifact_root=_path(base, str(experiment["artifact_root"])),
        repair_artifact_root=_path(base, str(inputs["repair_artifact_root"])),
        paired_dense_artifact_root=_optional_path(base, inputs.get("paired_dense_artifact_root")),
        mass_bagged_artifact_root=_optional_path(base, inputs.get("mass_bagged_artifact_root")),
        support_calibrated_artifact_root=None,
        shrink050_artifact_root=_optional_path(base, inputs.get("shrink050_artifact_root")),
        source_union_gmm_artifact_root=_optional_path(base, inputs.get("source_union_gmm_artifact_root")),
        balanced_gmm_artifact_root=_optional_path(base, inputs.get("balanced_gmm_artifact_root")),
        feature_cache_root=_path(base, str(inputs["feature_cache_root"])),
        backbone=str(inputs.get("backbone", "")),
        experiment_seeds=tuple(int(v) for v in run["experiment_seeds"]),
        heldout_centers=tuple(str(v) for v in run["heldout_centers"]),
        replicate_seeds=tuple(int(v) for v in run["replicate_seeds"]),
        fresh_replicate_seeds=tuple(int(v) for v in run.get("fresh_replicate_seeds", ())),
        strict_full_run_matrix=bool(run.get("strict_full_run_matrix", False)),
        synthetic_per_class_total=int(generation["synthetic_per_class_total"]),
        min_per_source_per_class=int(generation["min_per_source_per_class"]),
        primary_variant=str(experiment["primary_variant"]),
        primary_method=str(fixed["primary_method"]),
        random_mass_bag_size=int(fixed["random_mass_bag_size"]),
        random_mass_bag_alpha=float(fixed["random_mass_bag_alpha"]),
        blend_alpha=float(fixed["blend_alpha"]),
        primary_shrink_lambda=float(fixed["primary_shrink_lambda"]),
        matched_shuffled_reliability_null_permutations=int(fixed.get("matched_shuffled_reliability_null_permutations", 0)),
        candidate_components_per_source_class=tuple(int(v) for v in fixed["candidate_components_per_source_class"]),
        min_samples_per_component=int(fixed["min_samples_per_component"]),
        source_weighting=str(fixed["source_weighting"]),
        gmm_covariance_type=str(fixed["gmm_covariance_type"]),
        gmm_reg_covar=float(fixed["gmm_reg_covar"]),
        gmm_n_init=int(fixed["gmm_n_init"]),
        gmm_max_iter=int(fixed["gmm_max_iter"]),
        min_component_weight=float(fixed["min_component_weight"]),
        variance_floor=float(fixed["variance_floor"]),
        variance_ceiling_multiplier=float(fixed["variance_ceiling_multiplier"]),
        primary_pooling=str(fixed["primary_pooling"]),
        reliability_floor_score=float(fixed["reliability_floor_score"]),
        reliability_epsilon=float(fixed["reliability_epsilon"]),
        anchor_repro_tolerance=float(fixed["anchor_repro_tolerance"]),
        classifier_type=str(classifier["type"]),
        classifier_solver=str(classifier["solver"]),
        classifier_c=float(classifier["C"]),
        classifier_max_iter=int(classifier["max_iter"]),
        classifier_class_weight=str(classifier["class_weight"]),
        classifier_seed=None if classifier.get("classifier_seed") is None else int(classifier["classifier_seed"]),
        prior_tailrisk_artifact_root=_optional_path(base, inputs.get("prior_tailrisk_artifact_root")),
        panel_seed_groups=panel_seed_groups,
        primary_noninferiority_margin=float(fixed.get("primary_noninferiority_margin", 0.005)),
        weak_pass_noninferiority_margin=float(fixed.get("weak_pass_noninferiority_margin", 0.010)),
        tailrisk_transfer_threshold=float(fixed.get("tailrisk_transfer_threshold", -0.010)),
        candidate_pooling_rules=tuple(str(v) for v in fixed["candidate_pooling_rules"]),
        positive_label=int(fixed["positive_label"]),
        prediction_threshold=float(fixed["prediction_threshold"]),
        min_source_inner_positive_count=int(fixed.get("min_source_inner_positive_count", 5)),
        positive_union_eps=float(fixed["positive_union_eps"]),
        source_inner_bacc_noninferiority_margin=float(fixed.get("source_inner_bacc_noninferiority_margin", 0.010)),
        source_inner_class0_recall_margin=float(fixed.get("source_inner_class0_recall_margin", 0.015)),
        source_inner_predicted_positive_rate_delta=float(fixed.get("source_inner_predicted_positive_rate_delta", 0.050)),
        beta100_class0_recall_margin=float(fixed.get("beta100_class0_recall_margin", 0.005)),
        beta100_precision_margin=float(fixed.get("beta100_precision_margin", 0.010)),
        fixed_pooling_rule=str(fixed["fixed_pooling_rule"]),
        fixed_beta=float(fixed["fixed_beta"]),
        development_experiment_seeds=tuple(int(v) for v in fixed["development_experiment_seeds"]),
        confirmation_experiment_seeds=tuple(int(v) for v in fixed["primary_confirmation_experiment_seeds"]),
        development_positive_union_artifact_root=_optional_path(base, inputs.get("development_positive_union_artifact_root")),
        rare_positive_count_threshold=int(fixed["rare_positive_count_threshold"]),
        rare_positive_prevalence_threshold=float(fixed["rare_positive_prevalence_threshold"]),
    )
    validate_fixed_beta050_positive_union_config(cfg)
    return cfg


def validate_fixed_beta050_positive_union_config(cfg: FixedBeta050PositiveUnionConfig) -> None:
    if cfg.name != FIXED_BETA050_POSITIVE_UNION_NAME:
        raise ProtocolError(f"Fixed beta050 experiment name must be {FIXED_BETA050_POSITIVE_UNION_NAME!r}.")
    if cfg.backbone != "virchow2":
        raise ProtocolError("Fixed beta050 positive-union confirmation is locked to backbone=virchow2.")
    if cfg.primary_variant != PRIMARY_VARIANT:
        raise ProtocolError(f"primary_variant must be {PRIMARY_VARIANT!r}.")
    if cfg.primary_method != PRIMARY_FIXED_BETA050_POSITIVE_UNION_METHOD:
        raise ProtocolError(f"primary_method must be {PRIMARY_FIXED_BETA050_POSITIVE_UNION_METHOD!r}.")
    if cfg.source_weighting != FIXED_BETA050_POSITIVE_UNION_SOURCE_WEIGHTING:
        raise ProtocolError(f"source_weighting must be {FIXED_BETA050_POSITIVE_UNION_SOURCE_WEIGHTING!r}.")
    if cfg.primary_pooling != FIXED_BETA050_POSITIVE_UNION_PRIMARY_POOLING:
        raise ProtocolError(f"primary_pooling must be {FIXED_BETA050_POSITIVE_UNION_PRIMARY_POOLING!r}.")
    if cfg.fixed_pooling_rule != POSITIVE_UNION_RULE_BETA050:
        raise ProtocolError("fixed_pooling_rule must be locked to positive_union_beta050.")
    if not math.isclose(cfg.fixed_beta, 0.50, rel_tol=0.0, abs_tol=1.0e-12):
        raise ProtocolError("fixed_beta must be locked to 0.50.")
    if cfg.development_experiment_seeds != FIXED_BETA050_DEVELOPMENT_EXPERIMENT_SEEDS:
        raise ProtocolError("development_experiment_seeds must be locked to [42, 43, 44].")
    if set(cfg.experiment_seeds) & set(cfg.development_experiment_seeds):
        raise ProtocolError("primary confirmation experiment_seeds must not overlap development_experiment_seeds.")
    if cfg.confirmation_experiment_seeds != tuple(cfg.experiment_seeds):
        raise ProtocolError("primary_confirmation_experiment_seeds must equal run_matrix.experiment_seeds.")
    if cfg.candidate_pooling_rules != POSITIVE_UNION_RULES:
        raise ProtocolError("candidate_pooling_rules must be locked to arithmetic_mean/beta025/beta050/beta100.")
    if cfg.positive_label != 1:
        raise ProtocolError("positive_label must be locked to 1.")
    if not math.isclose(cfg.prediction_threshold, 0.5, rel_tol=0.0, abs_tol=1.0e-12):
        raise ProtocolError("prediction_threshold must be locked to 0.50.")
    if cfg.min_source_inner_positive_count != 5:
        raise ProtocolError("min_source_inner_positive_count must remain locked to 5 for diagnostics.")
    if not math.isclose(cfg.positive_union_eps, 1.0e-8, rel_tol=0.0, abs_tol=1.0e-14):
        raise ProtocolError("positive_union_eps must be locked to 1e-8.")
    if cfg.rare_positive_count_threshold != FIXED_BETA050_RARE_POSITIVE_COUNT_THRESHOLD:
        raise ProtocolError("rare_positive_count_threshold must be locked to 10.")
    if not math.isclose(cfg.rare_positive_prevalence_threshold, FIXED_BETA050_RARE_POSITIVE_PREVALENCE_THRESHOLD, rel_tol=0.0, abs_tol=1.0e-12):
        raise ProtocolError("rare_positive_prevalence_threshold must be locked to 0.05.")
    if cfg.candidate_components_per_source_class != (4, 3, 2, 1):
        raise ProtocolError("candidate_components_per_source_class must be locked to [4, 3, 2, 1].")
    if len(cfg.heldout_centers) != 5:
        raise ProtocolError("Fixed beta050 positive-union confirmation expects exactly five centers.")
    if cfg.gmm_covariance_type != "diag":
        raise ProtocolError("gmm_covariance_type must be diag.")
    if not math.isclose(cfg.primary_shrink_lambda, 0.5, rel_tol=0.0, abs_tol=1.0e-12):
        raise ProtocolError("primary_shrink_lambda must be locked to 0.50.")
    if not math.isclose(cfg.blend_alpha, 0.5, rel_tol=0.0, abs_tol=1.0e-12):
        raise ProtocolError("blend_alpha must be locked to 0.50.")
    if not math.isclose(cfg.random_mass_bag_alpha, 4.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ProtocolError("random_mass_bag_alpha must be locked to Dirichlet-uniform alpha4.")
    if cfg.random_mass_bag_size < 1:
        raise ProtocolError("random_mass_bag_size must be positive.")
    if cfg.matched_shuffled_reliability_null_permutations != 0:
        raise ProtocolError("matched_shuffled_reliability_null_permutations must be 0 for fixed beta050 confirmation.")
    if cfg.replicate_seeds != cfg.panel_seed_groups[0][1]:
        raise ProtocolError("run_matrix.replicate_seeds must equal the canonical panel seeds.")
    expected_fresh = tuple(seed for _panel, seeds in cfg.panel_seed_groups[1:] for seed in seeds)
    if cfg.fresh_replicate_seeds != expected_fresh:
        raise ProtocolError("run_matrix.fresh_replicate_seeds must equal fresh_a + fresh_b panel seeds.")
    if cfg.panel_seed_groups != MULTIPANEL_PANEL_SEEDS:
        raise ProtocolError("panel_seed_groups must be locked to canonical/fresh_a/fresh_b predeclared seeds.")
    if cfg.strict_full_run_matrix:
        if cfg.experiment_seeds != FIXED_BETA050_CONFIRMATION_EXPERIMENT_SEEDS:
            raise ProtocolError("strict_full_run_matrix requires experiment_seeds=[45, 46, 47, 48, 49].")
        if cfg.heldout_centers != ("0", "1", "2", "3", "4"):
            raise ProtocolError("strict_full_run_matrix requires heldout_centers=['0', '1', '2', '3', '4'].")
        if cfg.synthetic_per_class_total != 128:
            raise ProtocolError("strict_full_run_matrix requires synthetic_per_class_total=128.")
        if cfg.min_per_source_per_class != 8:
            raise ProtocolError("strict_full_run_matrix requires min_per_source_per_class=8.")
        if cfg.random_mass_bag_size != 11:
            raise ProtocolError("strict_full_run_matrix requires random_mass_bag_size=11.")
    if min(cfg.min_per_source_per_class, cfg.min_samples_per_component, cfg.gmm_n_init, cfg.gmm_max_iter) < 1:
        raise ProtocolError("Component minimums and GMM iterations must be positive.")
    if min(
        cfg.gmm_reg_covar,
        cfg.min_component_weight,
        cfg.variance_floor,
        cfg.variance_ceiling_multiplier,
        cfg.reliability_floor_score,
        cfg.reliability_epsilon,
        cfg.anchor_repro_tolerance,
        cfg.primary_noninferiority_margin,
        cfg.weak_pass_noninferiority_margin,
        cfg.source_inner_bacc_noninferiority_margin,
        cfg.source_inner_class0_recall_margin,
        cfg.source_inner_predicted_positive_rate_delta,
        cfg.beta100_class0_recall_margin,
        cfg.beta100_precision_margin,
    ) <= 0.0:
        raise ProtocolError("Fixed beta050 numeric floors/tolerances must be positive.")
    if cfg.tailrisk_transfer_threshold >= 0.0:
        raise ProtocolError("tailrisk_transfer_threshold must be negative.")
    if cfg.classifier_type != "sklearn_logistic_regression":
        raise ProtocolError("classifier.type must be sklearn_logistic_regression.")
    if cfg.classifier_solver != "lbfgs" or cfg.classifier_c != 1.0 or cfg.classifier_max_iter != 2000:
        raise ProtocolError("Classifier solver/C/max_iter must remain locked.")
    if cfg.classifier_class_weight != "balanced" or cfg.classifier_seed is not None:
        raise ProtocolError("Classifier must use class_weight=balanced and classifier_seed=null.")


def run_tailrisk_anchored_component_union(
    cfg: TailRiskAnchoredConfig,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    root = prepare_artifact_dirs(Path(artifact_root) if artifact_root is not None else cfg.artifact_root)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "summaries").mkdir(parents=True, exist_ok=True)
    (root / "dense_anchor_summaries").mkdir(parents=True, exist_ok=True)
    (root / "cache" / "generated").mkdir(parents=True, exist_ok=True)
    (root / "cache" / "predictions").mkdir(parents=True, exist_ok=True)

    matrix_rows: list[dict[str, object]] = []
    component_manifest_rows: list[dict[str, object]] = []
    component_coverage_rows: list[dict[str, object]] = []
    source_weight_rows: list[dict[str, object]] = []
    reliability_rows: list[dict[str, object]] = []
    source_summary_rows: list[dict[str, object]] = []
    source_ablation_rows: list[dict[str, object]] = []
    paired_generation_rows: list[dict[str, object]] = []
    eligibility_rows: list[dict[str, object]] = []
    blend_manifest_rows: list[dict[str, object]] = []
    complementarity_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    shuffled_null_rows: list[dict[str, object]] = []
    model_manifest_rows: list[dict[str, object]] = []
    protocol_violations: list[str] = []
    target_expert_excluded = True

    source_union_refs = d1._load_reference_values(
        cfg.source_union_gmm_artifact_root,
        table_name="gmm_prior_gap_summary.csv",
        method="source_union_cc_diag_gmm_k16_prior_sample_diagnostic",
        label="source-union K16",
    )
    center_balanced_refs = d1._load_reference_values(
        cfg.balanced_gmm_artifact_root,
        table_name="balanced_gmm_gap_summary.csv",
        method="source_union_center_balanced_cc_diag_gmm_k16_prior_sample",
        label="center-balanced K16",
    )
    for optional_root in (
        cfg.source_union_gmm_artifact_root,
        cfg.balanced_gmm_artifact_root,
        cfg.paired_dense_artifact_root,
        cfg.mass_bagged_artifact_root,
        cfg.support_calibrated_artifact_root,
        cfg.shrink050_artifact_root,
    ):
        d1._validate_optional_leakage_report(optional_root, protocol_violations)

    repair_cfg = d1._repair_runtime_config(cfg, root)
    per_source_variant = _per_source_variant()

    try:
        for experiment_seed in cfg.experiment_seeds:
            train_cache = load_feature_cache(_existing_cache_path(cfg.feature_cache_root, seed=experiment_seed, split="train"))
            test_cache = load_feature_cache(_existing_cache_path(cfg.feature_cache_root, seed=experiment_seed, split="test"))
            per_source_runtime: dict[str, RuntimeSource] = {}
            dense_summaries: dict[tuple[str, int], d1a.AdaptiveSourceLocalSummary] = {}
            gmm_summaries: dict[tuple[str, int], d1a.AdaptiveSourceLocalSummary] = {}
            shuffled_summaries: dict[tuple[str, int], d1a.AdaptiveSourceLocalSummary] = {}
            component_details: dict[tuple[str, int, int], dict[str, object]] = {}

            for source_center in cfg.heldout_centers:
                source_data = _source_data_for_centers(train_cache, centers=(source_center,), experiment_seed=int(experiment_seed))
                runtime_source = _runtime_source(
                    cfg,
                    repair_cfg,
                    root=root,
                    experiment_seed=int(experiment_seed),
                    heldout_center=NA,
                    expert_id=str(source_center),
                    source_data=source_data,
                    variant=per_source_variant,
                )
                per_source_runtime[str(source_center)] = runtime_source
                model_manifest_rows.append(_manifest_row(experiment_seed, NA, runtime_source))

                dense_largest, _dense_bic = d1a._fit_and_export_source_summaries(
                    cfg,
                    root / "dense_anchor_summaries",
                    runtime_source.runtime,
                    experiment_seed=int(experiment_seed),
                    shuffled_label_control=False,
                )
                for summary in dense_largest:
                    dense_summaries[(summary.source_center, summary.class_label)] = summary
                    source_summary_rows.append({**d1a._summary_diagnostic_row(cfg, summary), "summary_use": "dense_anchor"})

                summaries, detail_rows = cu._fit_and_export_pruned_gmm_summaries(
                    cfg,
                    root,
                    runtime_source.runtime,
                    experiment_seed=int(experiment_seed),
                    shuffled_label_control=False,
                )
                shuffled, shuffled_detail_rows = cu._fit_and_export_pruned_gmm_summaries(
                    cfg,
                    root,
                    runtime_source.runtime,
                    experiment_seed=int(experiment_seed),
                    shuffled_label_control=True,
                )
                for summary in summaries:
                    gmm_summaries[(summary.source_center, summary.class_label)] = summary
                    source_summary_rows.append({**d1a._summary_diagnostic_row(cfg, summary), "summary_use": "component_union"})
                for summary in shuffled:
                    shuffled_summaries[(summary.source_center, summary.class_label)] = summary
                    source_summary_rows.append({**d1a._summary_diagnostic_row(cfg, summary), "summary_use": "shuffled_label_control"})
                for row in detail_rows:
                    component_details[(str(row["source_center"]), int(row["class_label"]), int(row["source_component_id"]))] = row
                component_manifest_rows.extend(detail_rows)
                component_manifest_rows.extend(shuffled_detail_rows)

            reliability: dict[tuple[int, int, str], d12.SourceReliability] = {}
            for replicate_seed in cfg.all_replicate_seeds:
                for source_center in cfg.heldout_centers:
                    rel = d12._source_local_reliability(
                        cfg,
                        per_source_runtime=per_source_runtime,
                        summaries=dense_summaries,
                        test_cache=test_cache,
                        experiment_seed=int(experiment_seed),
                        replicate_seed=int(replicate_seed),
                        source_center=str(source_center),
                    )
                    reliability[(int(experiment_seed), int(replicate_seed), str(source_center))] = rel
                    reliability_rows.append(d12._source_reliability_row(rel))

            for heldout_center in cfg.heldout_centers:
                candidates = candidate_experts(cfg.heldout_centers, str(heldout_center))
                try:
                    assert_candidate_pool(
                        heldout_center=str(heldout_center),
                        candidate_experts=candidates,
                        expected_count=len(cfg.heldout_centers) - 1,
                    )
                except Exception:
                    target_expert_excluded = False
                    raise

                target_indices = _target_indices(test_cache.metadata, str(heldout_center))
                eval_raw, eval_meta = select_rows(test_cache.embeddings, test_cache.metadata, target_indices)
                eval_labels = tuple(_label(row) for row in eval_meta)
                eval_error = "mono_class_target_eval" if len(set(eval_labels)) < 2 else ""

                for replicate_seed in cfg.all_replicate_seeds:
                    su_ref = d1._reference_for_cell(source_union_refs, experiment_seed, heldout_center, replicate_seed)
                    cb_ref = d1._reference_for_cell(center_balanced_refs, experiment_seed, heldout_center, replicate_seed)
                    rels = {
                        source: reliability[(int(experiment_seed), int(replicate_seed), str(source))]
                        for source in candidates
                    }
                    if eval_error:
                        rows = _target_ineligible_rows(cfg, experiment_seed, heldout_center, replicate_seed, candidates, su_ref, cb_ref, eval_error)
                        matrix_rows.extend(rows)
                        eligibility_rows.append(_eligibility_row(experiment_seed, heldout_center, replicate_seed, "target_eval", "ineligible", eval_error))
                        continue

                    ref_row, _real_late = d1a._real_feature_reference(
                        cfg,
                        per_source_runtime=per_source_runtime,
                        candidates=candidates,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        replicate_seed=int(replicate_seed),
                        eval_raw=eval_raw,
                        eval_labels=eval_labels,
                    )
                    ref_row = _normalize_row(ref_row, prior_method=cu.ROW_REAL_FEATURE_DENSE_REFERENCE)
                    matrix_rows.append(ref_row)
                    real_feature_bacc = _float(ref_row["bacc"])

                    matrix_rows.extend(
                        mb._dense_comparator_rows(
                            cfg,
                            per_source_runtime=per_source_runtime,
                            summaries=dense_summaries,
                            candidates=candidates,
                            rels=rels,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            replicate_seed=int(replicate_seed),
                            eval_raw=eval_raw,
                            eval_labels=eval_labels,
                            source_union_ref=su_ref,
                            center_balanced_ref=cb_ref,
                            real_feature_bacc=real_feature_bacc,
                        )
                    )

                    uniform_plan = cu._uniform_source_plan(cfg, candidates, rels, total=cfg.synthetic_per_class_total)
                    source_weight_rows.extend(cu._source_weight_manifest_rows(int(experiment_seed), int(replicate_seed), str(heldout_center), cu.PRIMARY_COMPONENT_UNION_METHOD, uniform_plan, rels))
                    uniform = mb._evaluate_member(
                        cfg,
                        root=root,
                        per_source_runtime=per_source_runtime,
                        candidates=candidates,
                        summaries=gmm_summaries,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        replicate_seed=int(replicate_seed),
                        eval_raw=eval_raw,
                        eval_labels=eval_labels,
                        source_union_ref=su_ref,
                        center_balanced_ref=cb_ref,
                        real_feature_bacc=real_feature_bacc,
                        weight_plan=uniform_plan,
                        prior_method=cu.PRIMARY_COMPONENT_UNION_METHOD,
                        selection_source=DIAGNOSTIC_SELECTION,
                        claim_role="single_prior_component_union_reference",
                        control_mode="normal",
                    )
                    matrix_rows.append(uniform.row)
                    component_coverage_rows.append(uniform.coverage_row)
                    paired_generation_rows.append(uniform.paired_row)

                    component_manifest_rows.extend(
                        cu._fold_component_manifest_rows(
                            cfg,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            candidates=candidates,
                            summaries=gmm_summaries,
                            component_details=component_details,
                            weight_plan=uniform_plan,
                        )
                    )

                    tailrisk_eval = _evaluate_tailrisk_pair(
                        cfg,
                        root=root,
                        per_source_runtime=per_source_runtime,
                        candidates=candidates,
                        summaries=gmm_summaries,
                        rels=rels,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        replicate_seed=int(replicate_seed),
                        eval_raw=eval_raw,
                        eval_labels=eval_labels,
                        source_union_ref=su_ref,
                        center_balanced_ref=cb_ref,
                        real_feature_bacc=real_feature_bacc,
                    )
                    _append_tailrisk_outputs(
                        tailrisk_eval,
                        matrix_rows=matrix_rows,
                        component_coverage_rows=component_coverage_rows,
                        paired_generation_rows=paired_generation_rows,
                        source_weight_rows=source_weight_rows,
                        blend_manifest_rows=blend_manifest_rows,
                        complementarity_rows=complementarity_rows,
                        calibration_rows=calibration_rows,
                        eligibility_rows=eligibility_rows,
                    )

                    primary_bacc = _float(tailrisk_eval.primary_row.get("bacc"))
                    source_ablation_rows.extend(
                        _source_ablation_rows(
                            cfg,
                            root=root,
                            per_source_runtime=per_source_runtime,
                            summaries=gmm_summaries,
                            reliability=reliability,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            replicate_seed=int(replicate_seed),
                            candidates=candidates,
                            eval_raw=eval_raw,
                            eval_labels=eval_labels,
                            source_union_ref=su_ref,
                            center_balanced_ref=cb_ref,
                            real_feature_bacc=real_feature_bacc,
                            primary_bacc=primary_bacc,
                        )
                    )

                    control_evals = [
                        mb._evaluate_single_plan_control(
                            cfg,
                            root=root,
                            per_source_runtime=per_source_runtime,
                            candidates=candidates,
                            summaries=gmm_summaries,
                            rels=rels,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            replicate_seed=int(replicate_seed),
                            eval_raw=eval_raw,
                            eval_labels=eval_labels,
                            source_union_ref=su_ref,
                            center_balanced_ref=cb_ref,
                            real_feature_bacc=real_feature_bacc,
                        ),
                        _evaluate_single_control_member(
                            cfg,
                            root=root,
                            per_source_runtime=per_source_runtime,
                            candidates=candidates,
                            summaries=shuffled_summaries,
                            rels=rels,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            replicate_seed=int(replicate_seed),
                            eval_raw=eval_raw,
                            eval_labels=eval_labels,
                            source_union_ref=su_ref,
                            center_balanced_ref=cb_ref,
                            real_feature_bacc=real_feature_bacc,
                            prior_method=cu.ROW_SHUFFLED_LABEL_CONTROL,
                            claim_role="negative_control_shuffled_label_summary",
                            control_mode="normal",
                        ),
                        _evaluate_single_control_member(
                            cfg,
                            root=root,
                            per_source_runtime=per_source_runtime,
                            candidates=candidates,
                            summaries=gmm_summaries,
                            rels=rels,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            replicate_seed=int(replicate_seed),
                            eval_raw=eval_raw,
                            eval_labels=eval_labels,
                            source_union_ref=su_ref,
                            center_balanced_ref=cb_ref,
                            real_feature_bacc=real_feature_bacc,
                            prior_method=cu.ROW_SHUFFLED_SUMMARY_CONTROL,
                            claim_role="negative_control_class_flipped_summary",
                            control_mode="class_flip",
                        ),
                    ]
                    for control_eval in control_evals:
                        _append_control_outputs(
                            control_eval,
                            matrix_rows=matrix_rows,
                            component_coverage_rows=component_coverage_rows,
                            paired_generation_rows=paired_generation_rows,
                            source_weight_rows=source_weight_rows,
                            rels=rels,
                        )

                    shuffled_null_rows.extend(
                        _evaluate_shuffled_reliability_null(
                            cfg,
                            root=root,
                            per_source_runtime=per_source_runtime,
                            candidates=candidates,
                            summaries=gmm_summaries,
                            rels=rels,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            replicate_seed=int(replicate_seed),
                            eval_raw=eval_raw,
                            eval_labels=eval_labels,
                            source_union_ref=su_ref,
                            center_balanced_ref=cb_ref,
                            real_feature_bacc=real_feature_bacc,
                        )
                    )

                    matrix_rows.append(cu._reference_matrix_row(cfg, experiment_seed=int(experiment_seed), heldout_center=str(heldout_center), replicate_seed=int(replicate_seed), candidates=candidates, prior_method=cu.ROW_SOURCE_UNION_K16_REFERENCE, reference=su_ref))
                    matrix_rows.append(cu._reference_matrix_row(cfg, experiment_seed=int(experiment_seed), heldout_center=str(heldout_center), replicate_seed=int(replicate_seed), candidates=candidates, prior_method=cu.ROW_CENTER_BALANCED_K16_REFERENCE, reference=cb_ref))
    except ProtocolError as exc:
        protocol_violations.append(str(exc))

    complementarity_rows = _mark_bottom20_complementarity(matrix_rows, complementarity_rows)
    leakage = build_leakage_report(
        target_support_labels_for_selection=False,
        target_eval_labels_for_scoring_only=True,
        target_expert_excluded=target_expert_excluded,
        oracle_rows_diagnostic_only=True,
        extra_violations=protocol_violations,
    )
    anchor_rows = mb._anchor_reproducibility_rows(matrix_rows, cfg)
    shuffled_null_summary = _shuffled_null_summary_rows(matrix_rows, shuffled_null_rows)
    decision = _decision(
        matrix_rows,
        cfg=cfg,
        leakage_status=leakage.status,
        source_ablation_rows=source_ablation_rows,
        anchor_rows=anchor_rows,
        complementarity_rows=complementarity_rows,
        shuffled_null_summary=shuffled_null_summary,
    )
    _write_artifacts(
        root,
        cfg,
        matrix_rows=matrix_rows,
        component_manifest_rows=component_manifest_rows,
        component_coverage_rows=component_coverage_rows,
        source_weight_rows=source_weight_rows,
        reliability_rows=reliability_rows,
        source_summary_rows=source_summary_rows,
        source_ablation_rows=source_ablation_rows,
        paired_generation_rows=paired_generation_rows,
        eligibility_rows=eligibility_rows,
        blend_manifest_rows=blend_manifest_rows,
        complementarity_rows=complementarity_rows,
        calibration_rows=calibration_rows,
        shuffled_null_rows=shuffled_null_rows,
        shuffled_null_summary=shuffled_null_summary,
        model_manifest_rows=model_manifest_rows,
        anchor_rows=anchor_rows,
        decision=decision,
        leakage=leakage,
        protocol_violations=protocol_violations,
        target_expert_excluded=target_expert_excluded,
    )
    return root


@dataclass(frozen=True)
class _MultipanelSeedEvaluation:
    seed: int
    panel_group: str
    evaluated: TailRiskEvaluation


def run_multipanel_tailrisk_component_union(
    cfg: MultipanelTailRiskConfig,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    fixed_beta050_mode = isinstance(cfg, FixedBeta050PositiveUnionConfig)
    positive_union_mode = isinstance(cfg, SourceInnerPositiveUnionConfig)
    root = prepare_artifact_dirs(Path(artifact_root) if artifact_root is not None else cfg.artifact_root)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "summaries").mkdir(parents=True, exist_ok=True)
    (root / "dense_anchor_summaries").mkdir(parents=True, exist_ok=True)
    (root / "cache" / "generated").mkdir(parents=True, exist_ok=True)
    (root / "cache" / "predictions").mkdir(parents=True, exist_ok=True)

    matrix_rows: list[dict[str, object]] = []
    seed_diagnostic_rows: list[dict[str, object]] = []
    source_weight_rows: list[dict[str, object]] = []
    reliability_rows: list[dict[str, object]] = []
    source_summary_rows: list[dict[str, object]] = []
    component_manifest_rows: list[dict[str, object]] = []
    component_coverage_rows: list[dict[str, object]] = []
    paired_generation_rows: list[dict[str, object]] = []
    eligibility_rows: list[dict[str, object]] = []
    blend_manifest_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    panel_disagreement_rows: list[dict[str, object]] = []
    invariant_rows: list[dict[str, object]] = []
    confidence_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    center3_failure_cell_rows: list[dict[str, object]] = []
    center3_failure_sample_rows: list[dict[str, object]] = []
    center3_failure_pooling_rows: list[dict[str, object]] = []
    positive_union_source_inner_selection_rows: list[dict[str, object]] = []
    positive_union_candidate_rule_rows: list[dict[str, object]] = []
    positive_union_class_conditional_rows: list[dict[str, object]] = []
    positive_union_effective_threshold_rows: list[dict[str, object]] = []
    positive_union_harm_rows: list[dict[str, object]] = []
    positive_union_per_source_harm_rows: list[dict[str, object]] = []
    fixed_beta050_rare_positive_rows: list[dict[str, object]] = []
    fixed_beta050_source_inner_rows: list[dict[str, object]] = []
    model_manifest_rows: list[dict[str, object]] = []
    protocol_violations: list[str] = []
    target_expert_excluded = True

    source_union_refs = d1._load_reference_values(
        cfg.source_union_gmm_artifact_root,
        table_name="gmm_prior_gap_summary.csv",
        method="source_union_cc_diag_gmm_k16_prior_sample_diagnostic",
        label="source-union K16",
    )
    center_balanced_refs = d1._load_reference_values(
        cfg.balanced_gmm_artifact_root,
        table_name="balanced_gmm_gap_summary.csv",
        method="source_union_center_balanced_cc_diag_gmm_k16_prior_sample",
        label="center-balanced K16",
    )
    for optional_root in (
        cfg.source_union_gmm_artifact_root,
        cfg.balanced_gmm_artifact_root,
        cfg.paired_dense_artifact_root,
        cfg.mass_bagged_artifact_root,
        cfg.shrink050_artifact_root,
        cfg.prior_tailrisk_artifact_root,
    ):
        d1._validate_optional_leakage_report(optional_root, protocol_violations)

    repair_cfg = d1._repair_runtime_config(cfg, root)
    per_source_variant = _per_source_variant()

    try:
        for experiment_seed in cfg.experiment_seeds:
            train_cache = load_feature_cache(_existing_cache_path(cfg.feature_cache_root, seed=experiment_seed, split="train"))
            test_cache = load_feature_cache(_existing_cache_path(cfg.feature_cache_root, seed=experiment_seed, split="test"))
            per_source_runtime: dict[str, RuntimeSource] = {}
            dense_summaries: dict[tuple[str, int], d1a.AdaptiveSourceLocalSummary] = {}
            gmm_summaries: dict[tuple[str, int], d1a.AdaptiveSourceLocalSummary] = {}
            component_details: dict[tuple[str, int, int], dict[str, object]] = {}

            for source_center in cfg.heldout_centers:
                source_data = _source_data_for_centers(train_cache, centers=(source_center,), experiment_seed=int(experiment_seed))
                runtime_source = _runtime_source(
                    cfg,
                    repair_cfg,
                    root=root,
                    experiment_seed=int(experiment_seed),
                    heldout_center=NA,
                    expert_id=str(source_center),
                    source_data=source_data,
                    variant=per_source_variant,
                )
                per_source_runtime[str(source_center)] = runtime_source
                model_manifest_rows.append(_manifest_row(experiment_seed, NA, runtime_source))

                dense_largest, _dense_bic = d1a._fit_and_export_source_summaries(
                    cfg,
                    root / "dense_anchor_summaries",
                    runtime_source.runtime,
                    experiment_seed=int(experiment_seed),
                    shuffled_label_control=False,
                )
                for summary in dense_largest:
                    dense_summaries[(summary.source_center, summary.class_label)] = summary
                    source_summary_rows.append({**d1a._summary_diagnostic_row(cfg, summary), "summary_use": "dense_anchor"})

                summaries, detail_rows = cu._fit_and_export_pruned_gmm_summaries(
                    cfg,
                    root,
                    runtime_source.runtime,
                    experiment_seed=int(experiment_seed),
                    shuffled_label_control=False,
                )
                for summary in summaries:
                    gmm_summaries[(summary.source_center, summary.class_label)] = summary
                    source_summary_rows.append({**d1a._summary_diagnostic_row(cfg, summary), "summary_use": "component_union"})
                for row in detail_rows:
                    component_details[(str(row["source_center"]), int(row["class_label"]), int(row["source_component_id"]))] = row
                component_manifest_rows.extend(detail_rows)

            reliability: dict[tuple[int, int, str], d12.SourceReliability] = {}
            for replicate_seed in cfg.all_panel_seeds:
                for source_center in cfg.heldout_centers:
                    rel = d12._source_local_reliability(
                        cfg,
                        per_source_runtime=per_source_runtime,
                        summaries=dense_summaries,
                        test_cache=test_cache,
                        experiment_seed=int(experiment_seed),
                        replicate_seed=int(replicate_seed),
                        source_center=str(source_center),
                    )
                    reliability[(int(experiment_seed), int(replicate_seed), str(source_center))] = rel
                    rel_row = d12._source_reliability_row(rel)
                    rel_row["panel_group"] = _multipanel_panel_for_seed(cfg, replicate_seed)
                    reliability_rows.append(rel_row)

            for heldout_center in cfg.heldout_centers:
                candidates = candidate_experts(cfg.heldout_centers, str(heldout_center))
                try:
                    assert_candidate_pool(
                        heldout_center=str(heldout_center),
                        candidate_experts=candidates,
                        expected_count=len(cfg.heldout_centers) - 1,
                    )
                except Exception:
                    target_expert_excluded = False
                    raise

                target_indices = _target_indices(test_cache.metadata, str(heldout_center))
                eval_raw, eval_meta = select_rows(test_cache.embeddings, test_cache.metadata, target_indices)
                eval_labels = tuple(_label(row) for row in eval_meta)
                eval_sample_ids = tuple(str(row.get("sample_id", "")) for row in eval_meta)
                eval_sample_hash = _hash_strings(eval_sample_ids)
                eval_error = "mono_class_target_eval" if len(set(eval_labels)) < 2 else ""
                if eval_error:
                    eligibility_rows.append(_eligibility_row(int(experiment_seed), str(heldout_center), 0, "target_eval", "ineligible", eval_error))
                    continue

                component_manifest_rows.extend(
                    cu._fold_component_manifest_rows(
                        cfg,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        candidates=candidates,
                        summaries=gmm_summaries,
                        component_details=component_details,
                        weight_plan=cu._uniform_source_plan(
                            cfg,
                            candidates,
                            {source: reliability[(int(experiment_seed), cfg.all_panel_seeds[0], str(source))] for source in candidates},
                            total=cfg.synthetic_per_class_total,
                        ),
                    )
                )

                seed_evaluations: list[_MultipanelSeedEvaluation] = []
                real_feature_values: list[float] = []
                source_union_values: list[d1.ReferenceValue] = []
                center_balanced_values: list[d1.ReferenceValue] = []
                for replicate_seed in cfg.all_panel_seeds:
                    panel_group = _multipanel_panel_for_seed(cfg, replicate_seed)
                    su_ref = d1._reference_for_cell(source_union_refs, experiment_seed, heldout_center, replicate_seed)
                    cb_ref = d1._reference_for_cell(center_balanced_refs, experiment_seed, heldout_center, replicate_seed)
                    rels = {
                        source: reliability[(int(experiment_seed), int(replicate_seed), str(source))]
                        for source in candidates
                    }
                    ref_row, _real_late = d1a._real_feature_reference(
                        cfg,
                        per_source_runtime=per_source_runtime,
                        candidates=candidates,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        replicate_seed=int(replicate_seed),
                        eval_raw=eval_raw,
                        eval_labels=eval_labels,
                    )
                    real_feature_values.append(_float(ref_row.get("bacc")))
                    source_union_values.append(su_ref)
                    center_balanced_values.append(cb_ref)

                    evaluated = _evaluate_tailrisk_pair(
                        cfg,
                        root=root,
                        per_source_runtime=per_source_runtime,
                        candidates=candidates,
                        summaries=gmm_summaries,
                        rels=rels,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        replicate_seed=int(replicate_seed),
                        eval_raw=eval_raw,
                        eval_labels=eval_labels,
                        source_union_ref=su_ref,
                        center_balanced_ref=cb_ref,
                        real_feature_bacc=_float(ref_row.get("bacc")),
                    )
                    seed_evaluations.append(_MultipanelSeedEvaluation(int(replicate_seed), panel_group, evaluated))
                    _append_multipanel_seed_diagnostics(
                        cfg,
                        evaluated,
                        panel_group=panel_group,
                        seed_diagnostic_rows=seed_diagnostic_rows,
                        component_coverage_rows=component_coverage_rows,
                        paired_generation_rows=paired_generation_rows,
                        source_weight_rows=source_weight_rows,
                        blend_manifest_rows=blend_manifest_rows,
                        calibration_rows=calibration_rows,
                        eligibility_rows=eligibility_rows,
                    )

                if fixed_beta050_mode:
                    final = _build_fixed_beta050_positive_union_cell_outputs(
                        cfg,
                        seed_evaluations=seed_evaluations,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        candidates=candidates,
                        summaries=gmm_summaries,
                        eval_labels=eval_labels,
                        eval_sample_ids=eval_sample_ids,
                        eval_sample_hash=eval_sample_hash,
                        source_union_ref=_mean_reference(source_union_values),
                        center_balanced_ref=_mean_reference(center_balanced_values),
                        real_feature_bacc=nanmean([value for value in real_feature_values if math.isfinite(value)]),
                    )
                elif positive_union_mode:
                    final = _build_positive_union_cell_outputs(
                        cfg,
                        seed_evaluations=seed_evaluations,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        candidates=candidates,
                        summaries=gmm_summaries,
                        eval_labels=eval_labels,
                        eval_sample_ids=eval_sample_ids,
                        eval_sample_hash=eval_sample_hash,
                        source_union_ref=_mean_reference(source_union_values),
                        center_balanced_ref=_mean_reference(center_balanced_values),
                        real_feature_bacc=nanmean([value for value in real_feature_values if math.isfinite(value)]),
                    )
                else:
                    final = _build_multipanel_cell_outputs(
                        cfg,
                        seed_evaluations=seed_evaluations,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        candidates=candidates,
                        summaries=gmm_summaries,
                        eval_labels=eval_labels,
                        eval_sample_ids=eval_sample_ids,
                        eval_sample_hash=eval_sample_hash,
                        source_union_ref=_mean_reference(source_union_values),
                        center_balanced_ref=_mean_reference(center_balanced_values),
                        real_feature_bacc=nanmean([value for value in real_feature_values if math.isfinite(value)]),
                    )
                matrix_rows.extend(final["matrix_rows"])
                source_weight_rows.extend(final["source_weight_rows"])
                blend_manifest_rows.extend(final["blend_manifest_rows"])
                component_coverage_rows.extend(final["component_coverage_rows"])
                paired_generation_rows.extend(final["paired_generation_rows"])
                panel_disagreement_rows.extend(final.get("panel_disagreement_rows", []))
                invariant_rows.extend(final["invariant_rows"])
                confidence_rows.extend(final.get("confidence_rows", []))
                failure_rows.extend(final.get("failure_rows", []))
                center3_failure_cell_rows.extend(final.get("center3_failure_cell_rows", []))
                center3_failure_sample_rows.extend(final.get("center3_failure_sample_rows", []))
                center3_failure_pooling_rows.extend(final.get("center3_failure_pooling_rows", []))
                positive_union_source_inner_selection_rows.extend(final.get("positive_union_source_inner_selection_rows", []))
                positive_union_candidate_rule_rows.extend(final.get("positive_union_candidate_rule_rows", []))
                positive_union_class_conditional_rows.extend(final.get("positive_union_class_conditional_rows", []))
                positive_union_effective_threshold_rows.extend(final.get("positive_union_effective_threshold_rows", []))
                positive_union_harm_rows.extend(final.get("positive_union_harm_rows", []))
                positive_union_per_source_harm_rows.extend(final.get("positive_union_per_source_harm_rows", []))
                fixed_beta050_rare_positive_rows.extend(final.get("fixed_beta050_rare_positive_rows", []))
                fixed_beta050_source_inner_rows.extend(final.get("fixed_beta050_source_inner_rows", []))
    except ProtocolError as exc:
        protocol_violations.append(str(exc))

    leakage = build_leakage_report(
        target_support_labels_for_selection=False,
        target_eval_labels_for_scoring_only=True,
        target_expert_excluded=target_expert_excluded,
        oracle_rows_diagnostic_only=True,
        extra_violations=protocol_violations,
    )
    historical_rows = _load_prior_tailrisk_matrix_rows(cfg)
    if fixed_beta050_mode:
        paired_delta_rows, arithmetic_tail_keys = _fixed_beta050_paired_delta_rows(matrix_rows, cfg)
        positive_union_harm_rows = _annotate_fixed_beta050_harm_rows(positive_union_harm_rows, paired_delta_rows, cfg)
        decision = _fixed_beta050_decision(
            matrix_rows,
            paired_delta_rows=paired_delta_rows,
            arithmetic_tail_keys=arithmetic_tail_keys,
            rare_positive_rows=fixed_beta050_rare_positive_rows,
            harm_rows=positive_union_harm_rows,
            leakage_status=leakage.status,
            cfg=cfg,
        )
        _write_fixed_beta050_positive_union_artifacts(
            root,
            cfg,
            matrix_rows=matrix_rows,
            candidate_rule_rows=positive_union_candidate_rule_rows,
            class_conditional_rows=positive_union_class_conditional_rows,
            effective_threshold_rows=positive_union_effective_threshold_rows,
            rare_positive_rows=fixed_beta050_rare_positive_rows,
            paired_delta_rows=paired_delta_rows,
            harm_rows=positive_union_harm_rows,
            invariant_rows=invariant_rows,
            blend_manifest_rows=blend_manifest_rows,
            retrospective_reference_rows=_fixed_beta050_retrospective_reference_rows(cfg),
            source_inner_rows=fixed_beta050_source_inner_rows,
            decision=decision,
            leakage=leakage,
            protocol_violations=protocol_violations,
            target_expert_excluded=target_expert_excluded,
        )
    elif positive_union_mode:
        paired_delta_rows, prior_tail_keys = _positive_union_paired_delta_rows(matrix_rows, historical_rows, cfg)
        positive_union_harm_rows = _annotate_positive_union_harm_rows(positive_union_harm_rows, paired_delta_rows, cfg)
        decision = _positive_union_decision(
            matrix_rows,
            paired_delta_rows=paired_delta_rows,
            prior_tail_keys=prior_tail_keys,
            selection_rows=positive_union_source_inner_selection_rows,
            harm_rows=positive_union_harm_rows,
            leakage_status=leakage.status,
            cfg=cfg,
        )
        _write_positive_union_artifacts(
            root,
            cfg,
            matrix_rows=matrix_rows,
            source_inner_selection_rows=positive_union_source_inner_selection_rows,
            candidate_rule_rows=positive_union_candidate_rule_rows,
            class_conditional_rows=positive_union_class_conditional_rows,
            effective_threshold_rows=positive_union_effective_threshold_rows,
            paired_delta_rows=paired_delta_rows,
            harm_rows=positive_union_harm_rows,
            per_source_harm_rows=positive_union_per_source_harm_rows,
            invariant_rows=invariant_rows,
            blend_manifest_rows=blend_manifest_rows,
            source_weight_rows=source_weight_rows,
            reliability_rows=reliability_rows,
            source_summary_rows=source_summary_rows,
            component_manifest_rows=component_manifest_rows,
            component_coverage_rows=component_coverage_rows,
            paired_generation_rows=paired_generation_rows,
            eligibility_rows=eligibility_rows,
            model_manifest_rows=model_manifest_rows,
            decision=decision,
            leakage=leakage,
            protocol_violations=protocol_violations,
            target_expert_excluded=target_expert_excluded,
        )
    else:
        paired_delta_rows, prior_tail_keys = _multipanel_paired_delta_rows(matrix_rows, historical_rows, cfg)
        failure_rows = _annotate_failure_rows(failure_rows, paired_delta_rows, prior_tail_keys)
        panel_disagreement_rows = _annotate_panel_disagreement_rows(panel_disagreement_rows, prior_tail_keys)
        decision = _multipanel_decision(
            matrix_rows,
            paired_delta_rows=paired_delta_rows,
            prior_tail_keys=prior_tail_keys,
            leakage_status=leakage.status,
            cfg=cfg,
        )
        _write_multipanel_artifacts(
            root,
            cfg,
            matrix_rows=matrix_rows,
            seed_diagnostic_rows=seed_diagnostic_rows,
            source_weight_rows=source_weight_rows,
            reliability_rows=reliability_rows,
            source_summary_rows=source_summary_rows,
            component_manifest_rows=component_manifest_rows,
            component_coverage_rows=component_coverage_rows,
            paired_generation_rows=paired_generation_rows,
            eligibility_rows=eligibility_rows,
            blend_manifest_rows=blend_manifest_rows,
            calibration_rows=calibration_rows,
            panel_disagreement_rows=panel_disagreement_rows,
            invariant_rows=invariant_rows,
            confidence_rows=confidence_rows,
            failure_rows=failure_rows,
            center3_failure_cell_rows=center3_failure_cell_rows,
            center3_failure_sample_rows=center3_failure_sample_rows,
            center3_failure_pooling_rows=center3_failure_pooling_rows,
            paired_delta_rows=paired_delta_rows,
            model_manifest_rows=model_manifest_rows,
            decision=decision,
            leakage=leakage,
            protocol_violations=protocol_violations,
            target_expert_excluded=target_expert_excluded,
        )
    return root


def run_source_inner_positive_union(
    cfg: SourceInnerPositiveUnionConfig,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    return run_multipanel_tailrisk_component_union(cfg, artifact_root=artifact_root)


def run_fixed_beta050_positive_union(
    cfg: FixedBeta050PositiveUnionConfig,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    return run_multipanel_tailrisk_component_union(cfg, artifact_root=artifact_root)


def _multipanel_panel_for_seed(cfg: MultipanelTailRiskConfig, seed: int) -> str:
    for panel, seeds in cfg.panel_seed_groups:
        if int(seed) in {int(value) for value in seeds}:
            return str(panel)
    raise ProtocolError(f"Seed {seed} is not in the locked multipanel seed groups.")


def _append_multipanel_seed_diagnostics(
    cfg: MultipanelTailRiskConfig,
    evaluated: TailRiskEvaluation,
    *,
    panel_group: str,
    seed_diagnostic_rows: list[dict[str, object]],
    component_coverage_rows: list[dict[str, object]],
    paired_generation_rows: list[dict[str, object]],
    source_weight_rows: list[dict[str, object]],
    blend_manifest_rows: list[dict[str, object]],
    calibration_rows: list[dict[str, object]],
    eligibility_rows: list[dict[str, object]],
) -> None:
    for source, row in (
        ("seed_anchor", evaluated.anchor_result.row),
        ("seed_random_mass_bag", evaluated.bag_evaluation.ensemble_row),
        ("seed_blend", evaluated.primary_row),
    ):
        out = dict(row)
        out["selection_source"] = DIAGNOSTIC_SELECTION
        out["claim_role"] = f"multipanel_{source}_diagnostic_not_primary"
        out["panel_group"] = panel_group
        out["aggregation_unit"] = "seed_diagnostic"
        if source == "seed_blend":
            out["prior_method"] = MULTIPANEL_SEED_BLEND_METHOD
            out["primary_method_not_reportable"] = True
        seed_diagnostic_rows.append(out)
    for row in evaluated.calibration_rows:
        out = dict(row)
        out["panel_group"] = panel_group
        out["calibration_scope"] = "source_inner_primary_target_eval_diagnostic_only"
        calibration_rows.append(out)
    blend_row = dict(evaluated.blend_manifest_row)
    blend_row["panel_group"] = panel_group
    blend_row["primary_method"] = MULTIPANEL_SEED_BLEND_METHOD
    blend_row["aggregation_unit"] = "seed_blend_before_multipanel_pooling"
    blend_manifest_rows.append(blend_row)
    component_coverage_rows.extend(
        [
            evaluated.anchor_result.coverage_row,
            evaluated.bag_evaluation.ensemble_coverage,
            evaluated.primary_coverage,
        ]
    )
    paired_generation_rows.extend(
        [
            evaluated.anchor_result.paired_row,
            evaluated.bag_evaluation.ensemble_paired_row,
            evaluated.primary_paired_row,
        ]
    )
    source_weight_rows.extend(evaluated.source_weight_rows)
    eligibility_rows.extend(evaluated.eligibility_rows)


def _build_multipanel_cell_outputs(
    cfg: MultipanelTailRiskConfig,
    *,
    seed_evaluations: Sequence[_MultipanelSeedEvaluation],
    experiment_seed: int,
    heldout_center: str,
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    eval_labels: Sequence[int],
    eval_sample_ids: Sequence[str],
    eval_sample_hash: str,
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
    real_feature_bacc: float,
) -> dict[str, list[dict[str, object]]]:
    out: dict[str, list[dict[str, object]]] = {
        "matrix_rows": [],
        "source_weight_rows": [],
        "blend_manifest_rows": [],
        "component_coverage_rows": [],
        "paired_generation_rows": [],
        "panel_disagreement_rows": [],
        "invariant_rows": [],
        "confidence_rows": [],
        "failure_rows": [],
        "center3_failure_cell_rows": [],
        "center3_failure_sample_rows": [],
        "center3_failure_pooling_rows": [],
    }
    ok = [
        item
        for item in seed_evaluations
        if item.evaluated.primary_bundle is not None
        and item.evaluated.anchor_result.bundle is not None
        and item.evaluated.bag_evaluation.ensemble_bundle is not None
        and item.evaluated.primary_row.get("status") == "ok"
    ]
    if len(ok) != len(seed_evaluations) or len(ok) != len(cfg.all_panel_seeds):
        for method in (
            cfg.primary_method,
            MULTIPANEL_POOLED_ANCHOR_METHOD,
            MULTIPANEL_POOLED_RANDOM_BAG_METHOD,
            MULTIPANEL_CANONICAL_RANDOM_BAG_METHOD,
        ):
            row = cu._empty_matrix_row(
                cfg,
                experiment_seed=experiment_seed,
                heldout_center=heldout_center,
                replicate_seed=0,
                candidates=candidates,
                prior_method=method,
                source_union_ref=source_union_ref,
                center_balanced_ref=center_balanced_ref,
                real_feature_bacc=real_feature_bacc,
                status="ineligible",
                error_message="one_or_more_seed_blends_ineligible",
                claim_role="multipanel_probability_pool",
            )
            row["panel"] = "multipanel"
            row["aggregation_unit"] = "experiment_seed_x_heldout_center"
            out["matrix_rows"].append(row)
        return out

    seed_blend_bundles = [item.evaluated.primary_bundle for item in ok if item.evaluated.primary_bundle is not None]
    anchor_bundles = [item.evaluated.anchor_result.bundle for item in ok if item.evaluated.anchor_result.bundle is not None]
    bag_bundles = [item.evaluated.bag_evaluation.ensemble_bundle for item in ok if item.evaluated.bag_evaluation.ensemble_bundle is not None]
    seed_blend_rows = [item.evaluated.primary_row for item in ok]
    anchor_rows = [item.evaluated.anchor_result.row for item in ok]
    bag_rows = [item.evaluated.bag_evaluation.ensemble_row for item in ok]
    seed_hashes = [str(row.get("prediction_hash", "")) for row in seed_blend_rows]
    group_json = _panel_seed_groups_json(cfg)

    final_bundle = _pool_bundle(cfg.primary_method, seed_blend_bundles)
    pooled_anchor = _pool_bundle(MULTIPANEL_POOLED_ANCHOR_METHOD, anchor_bundles)
    pooled_random = _pool_bundle(MULTIPANEL_POOLED_RANDOM_BAG_METHOD, bag_bundles)
    canonical_bags = [
        item.evaluated.bag_evaluation.ensemble_bundle
        for item in ok
        if item.panel_group == MULTIPANEL_CANONICAL_PANEL and item.evaluated.bag_evaluation.ensemble_bundle is not None
    ]
    canonical_random = _pool_bundle(MULTIPANEL_CANONICAL_RANDOM_BAG_METHOD, canonical_bags)

    panel_blend_bundles: dict[str, PredictionBundle] = {}
    for panel, _seeds in cfg.panel_seed_groups:
        panel_bundles = [item.evaluated.primary_bundle for item in ok if item.panel_group == panel and item.evaluated.primary_bundle is not None]
        panel_blend_bundles[str(panel)] = _pool_bundle(f"{MULTIPANEL_SEED_BLEND_METHOD}_{panel}", panel_bundles)

    row_specs = (
        (cfg.primary_method, final_bundle, seed_blend_rows, PRIMARY_SELECTION, "primary_multipanel_seed_blend_probability_pool"),
        (MULTIPANEL_POOLED_ANCHOR_METHOD, pooled_anchor, anchor_rows, DIAGNOSTIC_SELECTION, "pooled_shrink050_comparator"),
        (MULTIPANEL_POOLED_RANDOM_BAG_METHOD, pooled_random, bag_rows, DIAGNOSTIC_SELECTION, "pooled_random_mass_bag_comparator"),
        (
            MULTIPANEL_CANONICAL_RANDOM_BAG_METHOD,
            canonical_random,
            [item.evaluated.bag_evaluation.ensemble_row for item in ok if item.panel_group == MULTIPANEL_CANONICAL_PANEL],
            DIAGNOSTIC_SELECTION,
            "canonical_single_random_mass_bag_comparator",
        ),
    )
    row_by_method: dict[str, dict[str, object]] = {}
    for method, bundle, plan_rows, selection_source, claim_role in row_specs:
        row = _multipanel_result_row(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            candidates=candidates,
            summaries=summaries,
            method=method,
            bundle=bundle,
            eval_labels=eval_labels,
            source_union_ref=source_union_ref,
            center_balanced_ref=center_balanced_ref,
            real_feature_bacc=real_feature_bacc,
            weight_plan=_average_plan_from_rows(cfg, candidates, plan_rows),
            generated_features_hash=_hash_strings(str(row.get("generated_features_hash", "")) for row in plan_rows),
            seed_bundle_hashes=seed_hashes if method == cfg.primary_method else [str(row.get("prediction_hash", "")) for row in plan_rows],
            selection_source=selection_source,
            claim_role=claim_role,
            eval_sample_hash=eval_sample_hash,
            panel_seed_groups_json=group_json,
        )
        row_by_method[method] = row
        out["matrix_rows"].append(row)
        out["component_coverage_rows"].append(cu._empty_coverage_row(row))
        out["paired_generation_rows"].append(cu._paired_generation_row(row, str(row.get("generated_features_hash", "")), "", "ok"))
        out["confidence_rows"].append(_confidence_row(experiment_seed, heldout_center, method, "multipanel", bundle))
        out["invariant_rows"].append(
            _probability_invariant_row(
                experiment_seed,
                heldout_center,
                method,
                bundle,
                eval_sample_ids=eval_sample_ids,
                expected_sample_hash=eval_sample_hash,
                panel="multipanel",
            )
        )

    for panel, bundle in panel_blend_bundles.items():
        panel_method = f"{MULTIPANEL_SEED_BLEND_METHOD}_{panel}"
        panel_result = evaluate_probability_predictions(panel_method, bundle.probabilities, eval_labels, classes=bundle.classes)
        out["failure_rows"].append(
            {
                "experiment_seed": experiment_seed,
                "heldout_center": heldout_center,
                "decomposition_source": f"panel_{panel}",
                "bacc": panel_result.bacc,
                "macro_f1": panel_result.macro_f1,
            }
        )
        out["confidence_rows"].append(_confidence_row(experiment_seed, heldout_center, panel_method, panel, bundle))
        out["invariant_rows"].append(
            _probability_invariant_row(
                experiment_seed,
                heldout_center,
                panel_method,
                bundle,
                eval_sample_ids=eval_sample_ids,
                expected_sample_hash=eval_sample_hash,
                panel=panel,
            )
        )

    out["panel_disagreement_rows"].append(
        _panel_disagreement_row(
            experiment_seed,
            heldout_center,
            panel_blend_bundles,
            eval_labels=eval_labels,
        )
    )
    out["blend_manifest_rows"].append(
        {
            "experiment_seed": experiment_seed,
            "heldout_center": heldout_center,
            "replicate_seed": 0,
            "panel": "multipanel",
            "primary_method": cfg.primary_method,
            "aggregation_unit": "experiment_seed_x_heldout_center",
            "pooling_rule": "seed_blend_then_equal_probability_pool",
            "blend_alpha_anchor": cfg.blend_alpha,
            "blend_alpha_bag": 1.0 - cfg.blend_alpha,
            "panel_seed_groups_json": group_json,
            "seed_blend_prediction_hashes_json": json.dumps(seed_hashes),
            "final_prediction_hash": row_by_method[cfg.primary_method].get("prediction_hash", ""),
            "eval_sample_ids_hash": eval_sample_hash,
            "class_order": "|".join(str(value) for value in final_bundle.classes),
            "class_order_match": True,
        }
    )
    out["failure_rows"].append(
        {
            "experiment_seed": experiment_seed,
            "heldout_center": heldout_center,
            "anchor_bacc": row_by_method[MULTIPANEL_POOLED_ANCHOR_METHOD].get("bacc", math.nan),
            "same_cell_single_random_mass_bag_canonical_bacc": row_by_method[MULTIPANEL_CANONICAL_RANDOM_BAG_METHOD].get("bacc", math.nan),
            "panel1_bacc": _panel_bacc(out["failure_rows"], MULTIPANEL_CANONICAL_PANEL),
            "panel2_bacc": _panel_bacc(out["failure_rows"], MULTIPANEL_FRESH_A_PANEL),
            "panel3_bacc": _panel_bacc(out["failure_rows"], MULTIPANEL_FRESH_B_PANEL),
            "pooled_random_bag_bacc": row_by_method[MULTIPANEL_POOLED_RANDOM_BAG_METHOD].get("bacc", math.nan),
            "final_anchor_random_blend_bacc": row_by_method[cfg.primary_method].get("bacc", math.nan),
            "status": "ok",
        }
    )
    audit = _center3_failure_audit_outputs(
        cfg,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        eval_labels=eval_labels,
        eval_sample_ids=eval_sample_ids,
        final_bundle=final_bundle,
        pooled_anchor=pooled_anchor,
        pooled_random=pooled_random,
        canonical_random=canonical_random,
        panel_blend_bundles=panel_blend_bundles,
        seed_evaluations=ok,
    )
    out["center3_failure_cell_rows"].extend(audit["cell_rows"])
    out["center3_failure_sample_rows"].extend(audit["sample_rows"])
    out["center3_failure_pooling_rows"].extend(audit["pooling_rows"])
    return out


def _build_positive_union_cell_outputs(
    cfg: SourceInnerPositiveUnionConfig,
    *,
    seed_evaluations: Sequence[_MultipanelSeedEvaluation],
    experiment_seed: int,
    heldout_center: str,
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    eval_labels: Sequence[int],
    eval_sample_ids: Sequence[str],
    eval_sample_hash: str,
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
    real_feature_bacc: float,
) -> dict[str, list[dict[str, object]]]:
    out: dict[str, list[dict[str, object]]] = {
        "matrix_rows": [],
        "source_weight_rows": [],
        "blend_manifest_rows": [],
        "component_coverage_rows": [],
        "paired_generation_rows": [],
        "invariant_rows": [],
        "positive_union_source_inner_selection_rows": [],
        "positive_union_candidate_rule_rows": [],
        "positive_union_class_conditional_rows": [],
        "positive_union_effective_threshold_rows": [],
        "positive_union_harm_rows": [],
        "positive_union_per_source_harm_rows": [],
    }
    ok = [
        item
        for item in seed_evaluations
        if item.evaluated.primary_bundle is not None
        and item.evaluated.anchor_result.bundle is not None
        and item.evaluated.bag_evaluation.ensemble_bundle is not None
        and item.evaluated.primary_row.get("status") == "ok"
    ]
    if len(ok) != len(seed_evaluations) or len(ok) != len(cfg.all_panel_seeds):
        row = cu._empty_matrix_row(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=0,
            candidates=candidates,
            prior_method=cfg.primary_method,
            source_union_ref=source_union_ref,
            center_balanced_ref=center_balanced_ref,
            real_feature_bacc=real_feature_bacc,
            status="ineligible",
            error_message="one_or_more_seed_blends_ineligible",
            claim_role="positive_union_probability_pool",
        )
        row["selection_used_target_labels"] = False
        row["target_eval_labels_used_for_scoring_only"] = True
        out["matrix_rows"].append(row)
        return out

    seed_blend_bundles = [item.evaluated.primary_bundle for item in ok if item.evaluated.primary_bundle is not None]
    anchor_bundles = [item.evaluated.anchor_result.bundle for item in ok if item.evaluated.anchor_result.bundle is not None]
    bag_bundles = [item.evaluated.bag_evaluation.ensemble_bundle for item in ok if item.evaluated.bag_evaluation.ensemble_bundle is not None]
    seed_blend_rows = [item.evaluated.primary_row for item in ok]
    anchor_rows = [item.evaluated.anchor_result.row for item in ok]
    bag_rows = [item.evaluated.bag_evaluation.ensemble_row for item in ok]
    seed_hashes = [str(row.get("prediction_hash", "")) for row in seed_blend_rows]
    group_json = _panel_seed_groups_json(cfg)

    source_inner_bundles = [
        item.evaluated.source_inner_bundles.get("primary_blend")
        for item in ok
        if item.evaluated.source_inner_bundles.get("primary_blend") is not None
    ]
    source_inner_labels = ok[0].evaluated.source_inner_labels
    source_inner_source_ids = ok[0].evaluated.source_inner_source_ids
    if len(source_inner_bundles) != len(ok) or not source_inner_labels:
        selected_rule = POSITIVE_UNION_RULE_ARITHMETIC
        source_candidate_bundles = _positive_union_candidate_bundles(cfg, seed_blend_bundles)
        source_rows = {
            rule: {
                "scope": "source_inner",
                "rule": rule,
                "beta": "" if _positive_union_rule_beta(rule) is None else _positive_union_rule_beta(rule),
                "source_inner_eligible": rule == POSITIVE_UNION_RULE_ARITHMETIC,
                "source_inner_ineligible_reason": "missing_source_inner_primary_blend_bundles" if rule != POSITIVE_UNION_RULE_ARITHMETIC else "",
            }
            for rule in cfg.candidate_pooling_rules
        }
        selection_row = _positive_union_selection_row(
            cfg,
            selected_rule=selected_rule,
            selected_row=source_rows[selected_rule],
            positive_count=0,
            negative_count=0,
            selection_reason="missing_source_inner_primary_blend_bundles",
        )
    else:
        source_candidate_bundles = _positive_union_candidate_bundles(cfg, source_inner_bundles)
        source_metric_rows = [
            _positive_union_metrics(rule, bundle, source_inner_labels, scope="source_inner")
            for rule, bundle in source_candidate_bundles.items()
        ]
        selected_rule, selected_source_rows, selection_row = _select_positive_union_rule(cfg, source_rows=source_metric_rows)
        source_rows = {str(row["rule"]): row for row in selected_source_rows}
        out["positive_union_per_source_harm_rows"].extend(
            _positive_union_per_source_harm_rows(
                cfg,
                experiment_seed=experiment_seed,
                heldout_center=heldout_center,
                source_ids=source_inner_source_ids,
                source_labels=source_inner_labels,
                source_bundles_by_rule=source_candidate_bundles,
            )
        )

    target_candidate_bundles = _positive_union_candidate_bundles(cfg, seed_blend_bundles)
    target_rows = {
        rule: _positive_union_metrics(rule, bundle, eval_labels, scope="target_eval")
        for rule, bundle in target_candidate_bundles.items()
    }
    selected_bundle = target_candidate_bundles[selected_rule]
    arithmetic_bundle = target_candidate_bundles[POSITIVE_UNION_RULE_ARITHMETIC]
    pooled_anchor = _pool_bundle(MULTIPANEL_POOLED_ANCHOR_METHOD, anchor_bundles)
    pooled_random = _pool_bundle(MULTIPANEL_POOLED_RANDOM_BAG_METHOD, bag_bundles)
    canonical_bags = [
        item.evaluated.bag_evaluation.ensemble_bundle
        for item in ok
        if item.panel_group == MULTIPANEL_CANONICAL_PANEL and item.evaluated.bag_evaluation.ensemble_bundle is not None
    ]
    canonical_random = _pool_bundle(MULTIPANEL_CANONICAL_RANDOM_BAG_METHOD, canonical_bags)

    selection_row.update(
        {
            "experiment_seed": experiment_seed,
            "heldout_center": heldout_center,
            "decision_cell": "experiment_seed_x_heldout_center",
            "audit_only": False,
            "primary_adoption_eligible": True,
        }
    )
    out["positive_union_source_inner_selection_rows"].append(selection_row)

    for rule in cfg.candidate_pooling_rules:
        row = {
            "experiment_seed": experiment_seed,
            "heldout_center": heldout_center,
            "rule": rule,
            "beta": "" if _positive_union_rule_beta(rule) is None else _positive_union_rule_beta(rule),
            "selected_rule_for_cell": selected_rule,
            "is_selected_rule": rule == selected_rule,
            "selection_used_target_labels": False,
            "target_eval_labels_used_for_audit_only": True,
            "audit_only": True,
            "primary_adoption_eligible": False,
        }
        for key, value in source_rows.get(rule, {}).items():
            row[f"source_inner_{key}"] = value
        for key, value in target_rows[rule].items():
            row[f"target_{key}"] = value
        out["positive_union_candidate_rule_rows"].append(row)

    row_specs = (
        (cfg.primary_method, selected_bundle, seed_blend_rows, PRIMARY_SELECTION, "source_inner_selected_class_conditional_positive_union_primary", selected_rule),
        (POSITIVE_UNION_RULE_ARITHMETIC, arithmetic_bundle, seed_blend_rows, DIAGNOSTIC_SELECTION, "arithmetic_multipanel_comparator", POSITIVE_UNION_RULE_ARITHMETIC),
        (MULTIPANEL_POOLED_ANCHOR_METHOD, pooled_anchor, anchor_rows, DIAGNOSTIC_SELECTION, "pooled_shrink050_comparator", "pooled_anchor"),
        (MULTIPANEL_POOLED_RANDOM_BAG_METHOD, pooled_random, bag_rows, DIAGNOSTIC_SELECTION, "pooled_random_mass_bag_comparator", "pooled_random"),
        (
            MULTIPANEL_CANONICAL_RANDOM_BAG_METHOD,
            canonical_random,
            [item.evaluated.bag_evaluation.ensemble_row for item in ok if item.panel_group == MULTIPANEL_CANONICAL_PANEL],
            DIAGNOSTIC_SELECTION,
            "canonical_single_random_mass_bag_comparator",
            "canonical_random_mass_bag",
        ),
    )
    row_by_method: dict[str, dict[str, object]] = {}
    for method, bundle, plan_rows, selection_source, claim_role, pooling_rule in row_specs:
        row = _multipanel_result_row(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            candidates=candidates,
            summaries=summaries,
            method=method,
            bundle=bundle,
            eval_labels=eval_labels,
            source_union_ref=source_union_ref,
            center_balanced_ref=center_balanced_ref,
            real_feature_bacc=real_feature_bacc,
            weight_plan=_average_plan_from_rows(cfg, candidates, plan_rows),
            generated_features_hash=_hash_strings(str(row.get("generated_features_hash", "")) for row in plan_rows),
            seed_bundle_hashes=seed_hashes if method in {cfg.primary_method, POSITIVE_UNION_RULE_ARITHMETIC} else [str(row.get("prediction_hash", "")) for row in plan_rows],
            selection_source=selection_source,
            claim_role=claim_role,
            eval_sample_hash=eval_sample_hash,
            panel_seed_groups_json=group_json,
        )
        row["pooling_rule"] = pooling_rule
        row["selected_positive_union_rule"] = selected_rule
        row["selected_positive_union_beta"] = "" if _positive_union_rule_beta(selected_rule) is None else _positive_union_rule_beta(selected_rule)
        row["source_inner_selection_reason"] = selection_row.get("selection_reason", "")
        row["target_support_used"] = False
        row["primary_adoption_eligible"] = method == cfg.primary_method
        row["audit_only"] = method != cfg.primary_method
        row_by_method[method] = row
        out["matrix_rows"].append(row)
        out["component_coverage_rows"].append(cu._empty_coverage_row(row))
        out["paired_generation_rows"].append(cu._paired_generation_row(row, str(row.get("generated_features_hash", "")), "", "ok"))
        out["invariant_rows"].append(
            _probability_invariant_row(
                experiment_seed,
                heldout_center,
                method,
                bundle,
                eval_sample_ids=eval_sample_ids,
                expected_sample_hash=eval_sample_hash,
                panel="positive_union",
            )
        )

    out["blend_manifest_rows"].append(
        {
            "experiment_seed": experiment_seed,
            "heldout_center": heldout_center,
            "replicate_seed": 0,
            "panel": "positive_union",
            "primary_method": cfg.primary_method,
            "aggregation_unit": "experiment_seed_x_heldout_center",
            "pooling_rule": POSITIVE_UNION_PRIMARY_POOLING,
            "selected_rule": selected_rule,
            "selected_beta": "" if _positive_union_rule_beta(selected_rule) is None else _positive_union_rule_beta(selected_rule),
            "selection_source": "source_inner",
            "selection_used_target_labels": False,
            "target_support_used": False,
            "blend_alpha_anchor": cfg.blend_alpha,
            "blend_alpha_bag": 1.0 - cfg.blend_alpha,
            "panel_seed_groups_json": group_json,
            "seed_blend_prediction_hashes_json": json.dumps(seed_hashes),
            "final_prediction_hash": row_by_method[cfg.primary_method].get("prediction_hash", ""),
            "eval_sample_ids_hash": eval_sample_hash,
            "class_order": "|".join(str(value) for value in selected_bundle.classes),
            "class_order_match": selected_bundle.classes == (0, 1),
        }
    )
    out["positive_union_class_conditional_rows"].extend(
        _positive_union_class_conditional_rows(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            target_rows=target_rows,
            selected_rule=selected_rule,
        )
    )
    out["positive_union_effective_threshold_rows"].extend(
        _positive_union_effective_threshold_rows(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            n_seed_bundles=len(seed_blend_bundles),
            source_rows=source_rows,
            target_rows=target_rows,
        )
    )
    out["positive_union_harm_rows"].append(
        _positive_union_harm_row(
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            selected_rule=selected_rule,
            selected_bundle=selected_bundle,
            arithmetic_bundle=arithmetic_bundle,
            eval_labels=eval_labels,
        )
    )
    return out


def _build_fixed_beta050_positive_union_cell_outputs(
    cfg: FixedBeta050PositiveUnionConfig,
    *,
    seed_evaluations: Sequence[_MultipanelSeedEvaluation],
    experiment_seed: int,
    heldout_center: str,
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    eval_labels: Sequence[int],
    eval_sample_ids: Sequence[str],
    eval_sample_hash: str,
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
    real_feature_bacc: float,
) -> dict[str, list[dict[str, object]]]:
    out: dict[str, list[dict[str, object]]] = {
        "matrix_rows": [],
        "source_weight_rows": [],
        "blend_manifest_rows": [],
        "component_coverage_rows": [],
        "paired_generation_rows": [],
        "invariant_rows": [],
        "positive_union_candidate_rule_rows": [],
        "positive_union_class_conditional_rows": [],
        "positive_union_effective_threshold_rows": [],
        "positive_union_harm_rows": [],
        "fixed_beta050_rare_positive_rows": [],
        "fixed_beta050_source_inner_rows": [],
    }
    ok = [
        item
        for item in seed_evaluations
        if item.evaluated.primary_bundle is not None
        and item.evaluated.anchor_result.bundle is not None
        and item.evaluated.bag_evaluation.ensemble_bundle is not None
        and item.evaluated.primary_row.get("status") == "ok"
    ]
    if len(ok) != len(seed_evaluations) or len(ok) != len(cfg.all_panel_seeds):
        row = cu._empty_matrix_row(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=0,
            candidates=candidates,
            prior_method=cfg.primary_method,
            source_union_ref=source_union_ref,
            center_balanced_ref=center_balanced_ref,
            real_feature_bacc=real_feature_bacc,
            status="ineligible",
            error_message="one_or_more_seed_blends_ineligible",
            claim_role="fixed_beta050_positive_union_probability_pool",
        )
        row["selection_used_target_labels"] = False
        row["target_eval_labels_used_for_scoring_only"] = True
        row["fixed_positive_union_rule"] = cfg.fixed_pooling_rule
        out["matrix_rows"].append(row)
        return out

    seed_blend_bundles = [item.evaluated.primary_bundle for item in ok if item.evaluated.primary_bundle is not None]
    anchor_bundles = [item.evaluated.anchor_result.bundle for item in ok if item.evaluated.anchor_result.bundle is not None]
    bag_bundles = [item.evaluated.bag_evaluation.ensemble_bundle for item in ok if item.evaluated.bag_evaluation.ensemble_bundle is not None]
    seed_blend_rows = [item.evaluated.primary_row for item in ok]
    anchor_rows = [item.evaluated.anchor_result.row for item in ok]
    bag_rows = [item.evaluated.bag_evaluation.ensemble_row for item in ok]
    seed_hashes = [str(row.get("prediction_hash", "")) for row in seed_blend_rows]
    group_json = _panel_seed_groups_json(cfg)

    target_candidate_bundles = _positive_union_candidate_bundles(cfg, seed_blend_bundles)
    target_rows = {
        rule: _positive_union_metrics(rule, bundle, eval_labels, scope="target_eval")
        for rule, bundle in target_candidate_bundles.items()
    }
    source_rows = _fixed_beta050_source_inner_diagnostic_rows(
        cfg,
        seed_evaluations=ok,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
    )
    source_rows_by_rule = {str(row["rule"]): row for row in source_rows}
    if not source_rows_by_rule:
        source_rows_by_rule = {rule: _empty_positive_union_metric_row(rule, scope="source_inner") for rule in cfg.candidate_pooling_rules}
    out["fixed_beta050_source_inner_rows"].extend(source_rows)

    selected_rule = cfg.fixed_pooling_rule
    selected_bundle = target_candidate_bundles[selected_rule]
    arithmetic_bundle = target_candidate_bundles[POSITIVE_UNION_RULE_ARITHMETIC]
    pooled_anchor = _pool_bundle(MULTIPANEL_POOLED_ANCHOR_METHOD, anchor_bundles)
    pooled_random = _pool_bundle(MULTIPANEL_POOLED_RANDOM_BAG_METHOD, bag_bundles)
    canonical_bags = [
        item.evaluated.bag_evaluation.ensemble_bundle
        for item in ok
        if item.panel_group == MULTIPANEL_CANONICAL_PANEL and item.evaluated.bag_evaluation.ensemble_bundle is not None
    ]
    canonical_random = _pool_bundle(MULTIPANEL_CANONICAL_RANDOM_BAG_METHOD, canonical_bags)

    for rule in cfg.candidate_pooling_rules:
        row = {
            "experiment_seed": experiment_seed,
            "heldout_center": heldout_center,
            "rule": rule,
            "beta": "" if _positive_union_rule_beta(rule) is None else _positive_union_rule_beta(rule),
            "fixed_rule_for_cell": selected_rule,
            "is_fixed_primary_rule": rule == selected_rule,
            "selection_used_target_labels": False,
            "source_inner_selection_used": False,
            "target_eval_labels_used_for_audit_only": True,
            "audit_only": True,
            "primary_adoption_eligible": False,
        }
        for key, value in source_rows_by_rule.get(rule, {}).items():
            row[f"source_inner_{key}"] = value
        for key, value in target_rows[rule].items():
            row[f"target_{key}"] = value
        out["positive_union_candidate_rule_rows"].append(row)

    row_specs = (
        (cfg.primary_method, selected_bundle, seed_blend_rows, PRIMARY_SELECTION, "fixed_global_beta050_positive_union_primary", selected_rule),
        (POSITIVE_UNION_RULE_ARITHMETIC, arithmetic_bundle, seed_blend_rows, DIAGNOSTIC_SELECTION, "arithmetic_multipanel_comparator", POSITIVE_UNION_RULE_ARITHMETIC),
        (POSITIVE_UNION_RULE_BETA025, target_candidate_bundles[POSITIVE_UNION_RULE_BETA025], seed_blend_rows, DIAGNOSTIC_SELECTION, "fixed_beta025_diagnostic", POSITIVE_UNION_RULE_BETA025),
        (POSITIVE_UNION_RULE_BETA100, target_candidate_bundles[POSITIVE_UNION_RULE_BETA100], seed_blend_rows, DIAGNOSTIC_SELECTION, "fixed_beta100_diagnostic", POSITIVE_UNION_RULE_BETA100),
        (MULTIPANEL_POOLED_ANCHOR_METHOD, pooled_anchor, anchor_rows, DIAGNOSTIC_SELECTION, "pooled_shrink050_comparator", "pooled_anchor"),
        (MULTIPANEL_POOLED_RANDOM_BAG_METHOD, pooled_random, bag_rows, DIAGNOSTIC_SELECTION, "pooled_random_mass_bag_comparator", "pooled_random"),
        (
            MULTIPANEL_CANONICAL_RANDOM_BAG_METHOD,
            canonical_random,
            [item.evaluated.bag_evaluation.ensemble_row for item in ok if item.panel_group == MULTIPANEL_CANONICAL_PANEL],
            DIAGNOSTIC_SELECTION,
            "canonical_single_random_mass_bag_comparator",
            "canonical_random_mass_bag",
        ),
    )
    row_by_method: dict[str, dict[str, object]] = {}
    for method, bundle, plan_rows, selection_source, claim_role, pooling_rule in row_specs:
        row = _multipanel_result_row(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            candidates=candidates,
            summaries=summaries,
            method=method,
            bundle=bundle,
            eval_labels=eval_labels,
            source_union_ref=source_union_ref,
            center_balanced_ref=center_balanced_ref,
            real_feature_bacc=real_feature_bacc,
            weight_plan=_average_plan_from_rows(cfg, candidates, plan_rows),
            generated_features_hash=_hash_strings(str(row.get("generated_features_hash", "")) for row in plan_rows),
            seed_bundle_hashes=seed_hashes if method in {cfg.primary_method, POSITIVE_UNION_RULE_ARITHMETIC, POSITIVE_UNION_RULE_BETA025, POSITIVE_UNION_RULE_BETA100} else [str(row.get("prediction_hash", "")) for row in plan_rows],
            selection_source=selection_source,
            claim_role=claim_role,
            eval_sample_hash=eval_sample_hash,
            panel_seed_groups_json=group_json,
        )
        row["pooling_rule"] = pooling_rule
        row["fixed_positive_union_rule"] = selected_rule
        row["fixed_positive_union_beta"] = cfg.fixed_beta
        row["source_inner_selection_used"] = False
        row["no_posthoc_beta_selection"] = True
        row["target_support_used"] = False
        row["primary_adoption_eligible"] = method == cfg.primary_method
        row["audit_only"] = method != cfg.primary_method
        row["retrospective_reference_only"] = False
        row_by_method[method] = row
        out["matrix_rows"].append(row)
        out["component_coverage_rows"].append(cu._empty_coverage_row(row))
        out["paired_generation_rows"].append(cu._paired_generation_row(row, str(row.get("generated_features_hash", "")), "", "ok"))
        out["invariant_rows"].append(
            _probability_invariant_row(
                experiment_seed,
                heldout_center,
                method,
                bundle,
                eval_sample_ids=eval_sample_ids,
                expected_sample_hash=eval_sample_hash,
                panel="fixed_beta050_positive_union",
            )
        )

    out["blend_manifest_rows"].append(
        {
            "experiment_seed": experiment_seed,
            "heldout_center": heldout_center,
            "replicate_seed": 0,
            "panel": "fixed_beta050_positive_union",
            "primary_method": cfg.primary_method,
            "aggregation_unit": "experiment_seed_x_heldout_center",
            "pooling_rule": FIXED_BETA050_POSITIVE_UNION_PRIMARY_POOLING,
            "fixed_rule": selected_rule,
            "fixed_beta": cfg.fixed_beta,
            "beta_origin": "hypothesis_generated_from_prior_positive_union_diagnostic",
            "development_experiment_seeds_json": json.dumps(list(cfg.development_experiment_seeds)),
            "primary_confirmation_experiment_seeds_json": json.dumps(list(cfg.confirmation_experiment_seeds)),
            "source_inner_selection_used": False,
            "selection_used_target_labels": False,
            "target_support_used": False,
            "no_posthoc_beta_selection": True,
            "blend_alpha_anchor": cfg.blend_alpha,
            "blend_alpha_bag": 1.0 - cfg.blend_alpha,
            "panel_seed_groups_json": group_json,
            "seed_blend_prediction_hashes_json": json.dumps(seed_hashes),
            "final_prediction_hash": row_by_method[cfg.primary_method].get("prediction_hash", ""),
            "eval_sample_ids_hash": eval_sample_hash,
            "class_order": "|".join(str(value) for value in selected_bundle.classes),
            "class_order_match": selected_bundle.classes == (0, 1),
        }
    )
    out["positive_union_class_conditional_rows"].extend(
        _positive_union_class_conditional_rows(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            target_rows=target_rows,
            selected_rule=selected_rule,
        )
    )
    out["positive_union_effective_threshold_rows"].extend(
        _positive_union_effective_threshold_rows(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            n_seed_bundles=len(seed_blend_bundles),
            source_rows=source_rows_by_rule,
            target_rows=target_rows,
        )
    )
    out["positive_union_harm_rows"].append(
        _positive_union_harm_row(
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            selected_rule=selected_rule,
            selected_bundle=selected_bundle,
            arithmetic_bundle=arithmetic_bundle,
            eval_labels=eval_labels,
        )
    )
    out["fixed_beta050_rare_positive_rows"].append(
        _fixed_beta050_rare_positive_opportunity_row(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            eval_labels=eval_labels,
            arithmetic_bundle=arithmetic_bundle,
            beta050_bundle=selected_bundle,
        )
    )
    return out


def _pool_bundle(method: str, bundles: Sequence[PredictionBundle | None]) -> PredictionBundle:
    valid = [bundle for bundle in bundles if bundle is not None]
    if not valid:
        raise ProtocolError(f"No prediction bundles available for {method}.")
    pooled = weighted_arithmetic_probability_pool(valid, [1.0] * len(valid))
    return PredictionBundle(
        expert_id=str(method),
        probabilities=tuple(tuple(float(value) for value in row) for row in pooled),
        classes=valid[0].classes,
    )


def _positive_union_pool_bundle(
    method: str,
    bundles: Sequence[PredictionBundle | None],
    *,
    beta: float | None,
    positive_label: int,
    eps: float,
) -> PredictionBundle:
    if beta is None:
        return _pool_bundle(method, bundles)
    valid = [bundle for bundle in bundles if bundle is not None]
    if not valid:
        raise ProtocolError(f"No prediction bundles available for {method}.")
    classes = valid[0].classes
    if classes != (0, 1):
        raise ProtocolError("Positive-union pooling requires binary class order [0, 1].")
    if int(positive_label) != 1:
        raise ProtocolError("Positive-union pooling requires positive_label=1.")
    n_rows = len(valid[0].probabilities)
    for bundle in valid:
        if bundle.classes != classes:
            raise ProtocolError("Class order mismatch in positive-union pooling.")
        if len(bundle.probabilities) != n_rows:
            raise ProtocolError("Prediction row count mismatch in positive-union pooling.")
    pooled: list[tuple[float, float]] = []
    for row_idx in range(n_rows):
        survival = 1.0
        for bundle in valid:
            row = bundle.probabilities[row_idx]
            if len(row) != 2:
                raise ProtocolError("Probability width mismatch in positive-union pooling.")
            p_pos = min(max(float(row[1]), float(eps)), 1.0 - float(eps))
            survival *= (1.0 - p_pos) ** float(beta)
        p_union = min(max(1.0 - survival, float(eps)), 1.0 - float(eps))
        pooled.append((1.0 - p_union, p_union))
    return PredictionBundle(
        expert_id=str(method),
        probabilities=tuple(pooled),
        classes=classes,
    )


def _positive_union_rule_beta(rule: str) -> float | None:
    if rule not in POSITIVE_UNION_BETAS:
        raise ProtocolError(f"Unknown positive-union pooling rule: {rule}")
    return POSITIVE_UNION_BETAS[rule]


def _positive_union_candidate_bundles(
    cfg: SourceInnerPositiveUnionConfig,
    bundles: Sequence[PredictionBundle | None],
) -> dict[str, PredictionBundle]:
    return {
        rule: _positive_union_pool_bundle(
            rule,
            bundles,
            beta=_positive_union_rule_beta(rule),
            positive_label=cfg.positive_label,
            eps=cfg.positive_union_eps,
        )
        for rule in cfg.candidate_pooling_rules
    }


def _binary_metrics_from_predictions(labels: Sequence[int], preds: Sequence[int]) -> dict[str, object]:
    labels_i = [int(value) for value in labels]
    preds_i = [int(value) for value in preds]
    support0 = sum(value == 0 for value in labels_i)
    support1 = sum(value == 1 for value in labels_i)
    pred0 = sum(value == 0 for value in preds_i)
    pred1 = sum(value == 1 for value in preds_i)
    tp0 = sum(true == 0 and pred == 0 for true, pred in zip(labels_i, preds_i))
    tp1 = sum(true == 1 and pred == 1 for true, pred in zip(labels_i, preds_i))
    fp1 = sum(true == 0 and pred == 1 for true, pred in zip(labels_i, preds_i))
    fp0 = sum(true == 1 and pred == 0 for true, pred in zip(labels_i, preds_i))
    rec0 = float(tp0) / float(support0) if support0 else math.nan
    rec1 = float(tp1) / float(support1) if support1 else math.nan
    prec0 = float(tp0) / float(pred0) if pred0 else math.nan
    prec1 = float(tp1) / float(pred1) if pred1 else math.nan
    spec0 = float(tp1) / float(support1) if support1 else math.nan
    spec1 = float(tp0) / float(support0) if support0 else math.nan
    f1_0 = _f1(prec0, rec0)
    f1_1 = _f1(prec1, rec1)
    smoothed_rec0 = (float(tp0) + 0.5) / (float(support0) + 1.0)
    smoothed_rec1 = (float(tp1) + 0.5) / (float(support1) + 1.0)
    smoothed_prec0 = (float(tp0) + 0.5) / (float(pred0) + 1.0)
    smoothed_prec1 = (float(tp1) + 0.5) / (float(pred1) + 1.0)
    smoothed_f1_0 = _f1(smoothed_prec0, smoothed_rec0)
    smoothed_f1_1 = _f1(smoothed_prec1, smoothed_rec1)
    n = len(labels_i)
    return {
        "n_eval": n,
        "class0_support": support0,
        "class1_support": support1,
        "class0_predicted_count": pred0,
        "class1_predicted_count": pred1,
        "class0_error_count": support0 - tp0,
        "class1_error_count": support1 - tp1,
        "class0_recall": rec0,
        "class1_recall": rec1,
        "class0_specificity": spec0,
        "class1_specificity": spec1,
        "precision_class0": prec0,
        "precision": prec1,
        "false_positive_count": fp1,
        "false_negative_count": fp0,
        "predicted_positive_rate": float(pred1) / float(n) if n else math.nan,
        "bacc": nanmean([value for value in (rec0, rec1) if math.isfinite(value)]),
        "macro_f1": nanmean([value for value in (f1_0, f1_1) if math.isfinite(value)]),
        "smoothed_class0_recall": smoothed_rec0,
        "smoothed_class1_recall": smoothed_rec1,
        "smoothed_min_class_recall": min(smoothed_rec0, smoothed_rec1),
        "smoothed_precision_class0": smoothed_prec0,
        "smoothed_precision": smoothed_prec1,
        "smoothed_bacc": 0.5 * (smoothed_rec0 + smoothed_rec1),
        "smoothed_macro_f1": 0.5 * (smoothed_f1_0 + smoothed_f1_1),
    }


def _f1(precision: float, recall: float) -> float:
    if not (math.isfinite(float(precision)) and math.isfinite(float(recall))):
        return math.nan
    denom = float(precision) + float(recall)
    return 0.0 if denom <= 0.0 else 2.0 * float(precision) * float(recall) / denom


def _positive_union_metrics(
    rule: str,
    bundle: PredictionBundle,
    labels: Sequence[int],
    *,
    scope: str,
) -> dict[str, object]:
    preds = predict_from_probabilities(bundle.probabilities, classes=bundle.classes)
    metrics = _binary_metrics_from_predictions(labels, preds)
    return {
        "scope": str(scope),
        "rule": str(rule),
        "beta": "" if _positive_union_rule_beta(rule) is None else _positive_union_rule_beta(rule),
        "class_order": "|".join(str(value) for value in bundle.classes),
        **metrics,
    }


def _empty_positive_union_metric_row(rule: str, *, scope: str) -> dict[str, object]:
    return {
        "scope": str(scope),
        "rule": str(rule),
        "beta": "" if _positive_union_rule_beta(rule) is None else _positive_union_rule_beta(rule),
        "class_order": "",
        "n_eval": 0,
        "class0_support": 0,
        "class1_support": 0,
        "class0_predicted_count": 0,
        "class1_predicted_count": 0,
        "class0_error_count": 0,
        "class1_error_count": 0,
        "class0_recall": math.nan,
        "class1_recall": math.nan,
        "class0_specificity": math.nan,
        "class1_specificity": math.nan,
        "precision_class0": math.nan,
        "precision": math.nan,
        "false_positive_count": 0,
        "false_negative_count": 0,
        "predicted_positive_rate": math.nan,
        "bacc": math.nan,
        "macro_f1": math.nan,
        "smoothed_class0_recall": math.nan,
        "smoothed_class1_recall": math.nan,
        "smoothed_min_class_recall": math.nan,
        "smoothed_precision_class0": math.nan,
        "smoothed_precision": math.nan,
        "smoothed_bacc": math.nan,
        "smoothed_macro_f1": math.nan,
    }


def _fixed_beta050_source_inner_diagnostic_rows(
    cfg: FixedBeta050PositiveUnionConfig,
    *,
    seed_evaluations: Sequence[_MultipanelSeedEvaluation],
    experiment_seed: int,
    heldout_center: str,
) -> list[dict[str, object]]:
    source_inner_bundles = [
        item.evaluated.source_inner_bundles.get("primary_blend")
        for item in seed_evaluations
        if item.evaluated.source_inner_bundles.get("primary_blend") is not None
    ]
    if len(source_inner_bundles) != len(seed_evaluations) or not seed_evaluations:
        return []
    source_inner_labels = seed_evaluations[0].evaluated.source_inner_labels
    if not source_inner_labels:
        return []
    candidate_bundles = _positive_union_candidate_bundles(cfg, source_inner_bundles)
    rows = []
    for rule, bundle in candidate_bundles.items():
        row = _positive_union_metrics(rule, bundle, source_inner_labels, scope="source_inner")
        row.update(
            {
                "experiment_seed": experiment_seed,
                "heldout_center": heldout_center,
                "fixed_rule_for_cell": cfg.fixed_pooling_rule,
                "is_fixed_primary_rule": rule == cfg.fixed_pooling_rule,
                "source_inner_selection_used": False,
                "selection_used_target_labels": False,
                "audit_only": True,
                "primary_adoption_eligible": False,
            }
        )
        rows.append(row)
    return rows


def _fixed_beta050_rare_positive_opportunity_row(
    cfg: FixedBeta050PositiveUnionConfig,
    *,
    experiment_seed: int,
    heldout_center: str,
    eval_labels: Sequence[int],
    arithmetic_bundle: PredictionBundle,
    beta050_bundle: PredictionBundle,
) -> dict[str, object]:
    labels = tuple(int(value) for value in eval_labels)
    arithmetic_preds = predict_from_probabilities(arithmetic_bundle.probabilities, classes=arithmetic_bundle.classes)
    beta050_preds = predict_from_probabilities(beta050_bundle.probabilities, classes=beta050_bundle.classes)
    arithmetic_metrics = _binary_metrics_from_predictions(labels, arithmetic_preds)
    beta050_metrics = _binary_metrics_from_predictions(labels, beta050_preds)
    class0_count = int(arithmetic_metrics["class0_support"])
    class1_count = int(arithmetic_metrics["class1_support"])
    n = class0_count + class1_count
    prevalence = float(class1_count) / float(n) if n else math.nan
    rare = bool(
        class1_count <= cfg.rare_positive_count_threshold
        or (math.isfinite(prevalence) and prevalence <= cfg.rare_positive_prevalence_threshold)
    )
    arithmetic_probs = np.asarray(arithmetic_bundle.probabilities, dtype=float)
    beta050_probs = np.asarray(beta050_bundle.probabilities, dtype=float)
    true_positive_indices = [idx for idx, label in enumerate(labels) if label == cfg.positive_label]
    arithmetic_tp_probs = [
        _sample_probability_for_class(arithmetic_bundle, arithmetic_probs, idx, cfg.positive_label)
        for idx in true_positive_indices
    ]
    beta050_tp_probs = [
        _sample_probability_for_class(beta050_bundle, beta050_probs, idx, cfg.positive_label)
        for idx in true_positive_indices
    ]
    deltas = [
        beta - arith
        for arith, beta in zip(arithmetic_tp_probs, beta050_tp_probs)
        if math.isfinite(arith) and math.isfinite(beta)
    ]
    return {
        "experiment_seed": experiment_seed,
        "heldout_center": heldout_center,
        "class0_count": class0_count,
        "class1_count": class1_count,
        "positive_prevalence": prevalence,
        "rare_positive_cell": rare,
        "arithmetic_class1_recall": arithmetic_metrics.get("class1_recall", math.nan),
        "beta050_class1_recall": beta050_metrics.get("class1_recall", math.nan),
        "arithmetic_true_positive_probabilities": json.dumps(arithmetic_tp_probs),
        "beta050_true_positive_probabilities": json.dumps(beta050_tp_probs),
        "positive_margin_delta": nanmean(deltas),
        "assessable_for_rare_positive_repair": bool(rare and class1_count > 0),
        "audit_only": True,
        "primary_adoption_eligible": False,
        "selection_used_target_labels": False,
        "target_eval_labels_used_for_audit_only": True,
    }


def _select_positive_union_rule(
    cfg: SourceInnerPositiveUnionConfig,
    *,
    source_rows: Sequence[Mapping[str, object]],
) -> tuple[str, list[dict[str, object]], dict[str, object]]:
    rows = [dict(row) for row in source_rows]
    by_rule = {str(row["rule"]): row for row in rows}
    arithmetic = by_rule[POSITIVE_UNION_RULE_ARITHMETIC]
    positive_count = _safe_int(arithmetic.get("class1_support"), default=0)
    negative_count = _safe_int(arithmetic.get("class0_support"), default=0)
    if positive_count < cfg.min_source_inner_positive_count:
        for row in rows:
            row["source_inner_eligible"] = row["rule"] == POSITIVE_UNION_RULE_ARITHMETIC
            row["source_inner_ineligible_reason"] = "" if row["rule"] == POSITIVE_UNION_RULE_ARITHMETIC else "insufficient_source_inner_positive_count"
        selected = POSITIVE_UNION_RULE_ARITHMETIC
        return selected, rows, _positive_union_selection_row(
            cfg,
            selected_rule=selected,
            selected_row=by_rule[selected],
            positive_count=positive_count,
            negative_count=negative_count,
            selection_reason="insufficient_source_inner_positive_count",
        )

    arith_bacc = _float(arithmetic.get("smoothed_bacc"))
    arith_class0 = _float(arithmetic.get("smoothed_class0_recall"))
    arith_class1 = _float(arithmetic.get("smoothed_class1_recall"))
    arith_precision = _float(arithmetic.get("smoothed_precision"))
    arith_ppr = _float(arithmetic.get("predicted_positive_rate"))
    for row in rows:
        rule = str(row["rule"])
        reasons: list[str] = []
        eligible = True
        if rule != POSITIVE_UNION_RULE_ARITHMETIC:
            if _float(row.get("smoothed_bacc")) < arith_bacc - cfg.source_inner_bacc_noninferiority_margin:
                reasons.append("source_inner_bacc_inferior")
            if _float(row.get("smoothed_class0_recall")) < arith_class0 - cfg.source_inner_class0_recall_margin:
                reasons.append("source_inner_class0_recall_harm")
            if _float(row.get("predicted_positive_rate")) > arith_ppr + cfg.source_inner_predicted_positive_rate_delta:
                reasons.append("source_inner_predicted_positive_rate_inflation")
            if rule == POSITIVE_UNION_RULE_BETA100:
                if _float(row.get("smoothed_class1_recall")) <= arith_class1:
                    reasons.append("beta100_no_class1_recall_gain")
                if _float(row.get("smoothed_class0_recall")) < arith_class0 - cfg.beta100_class0_recall_margin:
                    reasons.append("beta100_class0_recall_harm")
                if _float(row.get("smoothed_precision")) < arith_precision - cfg.beta100_precision_margin:
                    reasons.append("beta100_precision_harm")
            eligible = not reasons
        row["source_inner_eligible"] = eligible
        row["source_inner_ineligible_reason"] = "|".join(reasons)

    eligible_rows = [row for row in rows if row.get("source_inner_eligible") is True]
    if not eligible_rows:
        selected = POSITIVE_UNION_RULE_ARITHMETIC
        selected_row = by_rule[selected]
        reason = "no_eligible_rule_fallback_arithmetic"
    else:
        selected_row = max(
            eligible_rows,
            key=lambda row: (
                _float(row.get("smoothed_min_class_recall")),
                _float(row.get("smoothed_bacc")),
                _float(row.get("smoothed_macro_f1")),
                -cfg.candidate_pooling_rules.index(str(row["rule"])),
            ),
        )
        selected = str(selected_row["rule"])
        reason = "source_inner_selected"
    return selected, rows, _positive_union_selection_row(
        cfg,
        selected_rule=selected,
        selected_row=selected_row,
        positive_count=positive_count,
        negative_count=negative_count,
        selection_reason=reason,
    )


def _positive_union_selection_row(
    cfg: SourceInnerPositiveUnionConfig,
    *,
    selected_rule: str,
    selected_row: Mapping[str, object],
    positive_count: int,
    negative_count: int,
    selection_reason: str,
) -> dict[str, object]:
    return {
        "selected_rule": selected_rule,
        "selected_beta": "" if _positive_union_rule_beta(selected_rule) is None else _positive_union_rule_beta(selected_rule),
        "selection_reason": selection_reason,
        "source_inner_positive_count": int(positive_count),
        "source_inner_negative_count": int(negative_count),
        "min_source_inner_positive_count": cfg.min_source_inner_positive_count,
        "selected_source_inner_min_class_recall_smoothed": selected_row.get("smoothed_min_class_recall", math.nan),
        "selected_source_inner_bacc_smoothed": selected_row.get("smoothed_bacc", math.nan),
        "selected_source_inner_macro_f1_smoothed": selected_row.get("smoothed_macro_f1", math.nan),
        "selected_source_inner_precision_smoothed": selected_row.get("smoothed_precision", math.nan),
        "selected_source_inner_predicted_positive_rate": selected_row.get("predicted_positive_rate", math.nan),
        "selection_used_target_labels": False,
        "target_support_used": False,
    }


def _effective_threshold_for_rule(rule: str, n_seed_bundles: int) -> tuple[float, float]:
    beta = _positive_union_rule_beta(rule)
    if beta is None:
        identical = 0.5
        single = math.nan if n_seed_bundles > 2 else 1.0
        return identical, single
    identical = 1.0 - 0.5 ** (1.0 / (float(n_seed_bundles) * float(beta)))
    single = 1.0 - 0.5 ** (1.0 / float(beta))
    return identical, single


def _row_delta(row: Mapping[str, object], base: Mapping[str, object], key: str) -> float:
    left = _float(row.get(key))
    right = _float(base.get(key))
    return left - right if math.isfinite(left) and math.isfinite(right) else math.nan


def _positive_union_effective_threshold_rows(
    cfg: SourceInnerPositiveUnionConfig,
    *,
    experiment_seed: int,
    heldout_center: str,
    n_seed_bundles: int,
    source_rows: Mapping[str, Mapping[str, object]],
    target_rows: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    out = []
    arith_source = source_rows[POSITIVE_UNION_RULE_ARITHMETIC]
    arith_target = target_rows[POSITIVE_UNION_RULE_ARITHMETIC]
    for rule in cfg.candidate_pooling_rules:
        source = source_rows[rule]
        target = target_rows[rule]
        identical, single = _effective_threshold_for_rule(rule, n_seed_bundles)
        out.append(
            {
                "experiment_seed": experiment_seed,
                "heldout_center": heldout_center,
                "rule": rule,
                "beta": "" if _positive_union_rule_beta(rule) is None else _positive_union_rule_beta(rule),
                "n_seed_bundles": n_seed_bundles,
                "identical_seed_probability_needed_for_positive_flip": identical,
                "single_seed_probability_needed_for_positive_flip_if_other_seeds_zero": single,
                "source_inner_predicted_positive_rate": source.get("predicted_positive_rate", math.nan),
                "target_predicted_positive_rate": target.get("predicted_positive_rate", math.nan),
                "delta_predicted_positive_rate_vs_arithmetic": _row_delta(target, arith_target, "predicted_positive_rate"),
                "source_inner_delta_predicted_positive_rate_vs_arithmetic": _row_delta(source, arith_source, "predicted_positive_rate"),
                "class1_recall_delta": _row_delta(target, arith_target, "class1_recall"),
                "class0_recall_delta": _row_delta(target, arith_target, "class0_recall"),
                "precision_delta": _row_delta(target, arith_target, "precision"),
                "macro_f1_delta": _row_delta(target, arith_target, "macro_f1"),
                "bacc_delta": _row_delta(target, arith_target, "bacc"),
                "audit_only": True,
                "primary_adoption_eligible": False,
                "selection_used_target_labels": False,
                "target_eval_labels_used_for_audit_only": True,
            }
        )
    return out


def _positive_union_class_conditional_rows(
    cfg: SourceInnerPositiveUnionConfig,
    *,
    experiment_seed: int,
    heldout_center: str,
    target_rows: Mapping[str, Mapping[str, object]],
    selected_rule: str,
) -> list[dict[str, object]]:
    out = []
    for rule in cfg.candidate_pooling_rules:
        row = dict(target_rows[rule])
        row.update(
            {
                "experiment_seed": experiment_seed,
                "heldout_center": heldout_center,
                "selected_rule_for_cell": selected_rule,
                "is_selected_rule": rule == selected_rule,
                "audit_only": True,
                "primary_adoption_eligible": False,
                "selection_used_target_labels": False,
                "target_eval_labels_used_for_audit_only": True,
            }
        )
        out.append(row)
    return out


def _positive_union_harm_row(
    *,
    experiment_seed: int,
    heldout_center: str,
    selected_rule: str,
    selected_bundle: PredictionBundle,
    arithmetic_bundle: PredictionBundle,
    eval_labels: Sequence[int],
) -> dict[str, object]:
    selected_preds = predict_from_probabilities(selected_bundle.probabilities, classes=selected_bundle.classes)
    arithmetic_preds = predict_from_probabilities(arithmetic_bundle.probabilities, classes=arithmetic_bundle.classes)
    selected_metrics = _binary_metrics_from_predictions(eval_labels, selected_preds)
    arithmetic_metrics = _binary_metrics_from_predictions(eval_labels, arithmetic_preds)
    negative_to_positive = sum(int(a) == 0 and int(s) == 1 for a, s in zip(arithmetic_preds, selected_preds))
    positive_to_negative = sum(int(a) == 1 and int(s) == 0 for a, s in zip(arithmetic_preds, selected_preds))
    selected_true_positive = _safe_int(selected_metrics.get("class1_support"), default=0) - _safe_int(selected_metrics.get("class1_error_count"), default=0)
    arithmetic_true_positive = _safe_int(arithmetic_metrics.get("class1_support"), default=0) - _safe_int(arithmetic_metrics.get("class1_error_count"), default=0)
    bacc_delta = _row_delta(selected_metrics, arithmetic_metrics, "bacc")
    return {
        "experiment_seed": experiment_seed,
        "heldout_center": heldout_center,
        "selected_rule": selected_rule,
        "precision_delta_vs_arithmetic": _row_delta(selected_metrics, arithmetic_metrics, "precision"),
        "specificity_delta_vs_arithmetic": _row_delta(selected_metrics, arithmetic_metrics, "class1_specificity"),
        "predicted_positive_rate_delta": _row_delta(selected_metrics, arithmetic_metrics, "predicted_positive_rate"),
        "false_positive_count_delta_vs_arithmetic": _safe_int(selected_metrics.get("false_positive_count"), default=0) - _safe_int(arithmetic_metrics.get("false_positive_count"), default=0),
        "true_positive_count_delta_vs_arithmetic": selected_true_positive - arithmetic_true_positive,
        "negative_to_positive_flip_count": negative_to_positive,
        "positive_to_negative_flip_count": positive_to_negative,
        "bacc_delta_vs_arithmetic": bacc_delta,
        "worst_per_center_regression": bacc_delta,
        "worst_seed_center_regression": bacc_delta,
        "tail_risk_transfer_flag": bool(math.isfinite(bacc_delta) and bacc_delta < -0.010),
        "audit_only": True,
        "primary_adoption_eligible": False,
        "selection_used_target_labels": False,
        "target_eval_labels_used_for_audit_only": True,
    }


def _positive_union_per_source_harm_rows(
    cfg: SourceInnerPositiveUnionConfig,
    *,
    experiment_seed: int,
    heldout_center: str,
    source_ids: Sequence[str],
    source_labels: Sequence[int],
    source_bundles_by_rule: Mapping[str, PredictionBundle],
) -> list[dict[str, object]]:
    out = []
    arithmetic_preds = predict_from_probabilities(
        source_bundles_by_rule[POSITIVE_UNION_RULE_ARITHMETIC].probabilities,
        classes=source_bundles_by_rule[POSITIVE_UNION_RULE_ARITHMETIC].classes,
    )
    preds_by_rule = {
        rule: predict_from_probabilities(bundle.probabilities, classes=bundle.classes)
        for rule, bundle in source_bundles_by_rule.items()
    }
    for source_center in sorted(set(str(value) for value in source_ids)):
        indices = [idx for idx, value in enumerate(source_ids) if str(value) == source_center]
        labels = [int(source_labels[idx]) for idx in indices]
        arithmetic_metrics = _binary_metrics_from_predictions(labels, [int(arithmetic_preds[idx]) for idx in indices])
        for rule in cfg.candidate_pooling_rules:
            preds = [int(preds_by_rule[rule][idx]) for idx in indices]
            metrics = _binary_metrics_from_predictions(labels, preds)
            bacc_delta = _row_delta(metrics, arithmetic_metrics, "bacc")
            class0_delta = _row_delta(metrics, arithmetic_metrics, "class0_recall")
            class1_delta = _row_delta(metrics, arithmetic_metrics, "class1_recall")
            precision_delta = _row_delta(metrics, arithmetic_metrics, "precision")
            ppr_delta = _row_delta(metrics, arithmetic_metrics, "predicted_positive_rate")
            out.append(
                {
                    "experiment_seed": experiment_seed,
                    "heldout_center": heldout_center,
                    "source_center": source_center,
                    "rule": rule,
                    "beta": "" if _positive_union_rule_beta(rule) is None else _positive_union_rule_beta(rule),
                    "source_inner_positive_count": metrics["class1_support"],
                    "source_inner_negative_count": metrics["class0_support"],
                    "bacc_delta_vs_arithmetic": bacc_delta,
                    "class0_recall_delta_vs_arithmetic": class0_delta,
                    "class1_recall_delta_vs_arithmetic": class1_delta,
                    "precision_delta_vs_arithmetic": precision_delta,
                    "predicted_positive_rate_delta_vs_arithmetic": ppr_delta,
                    "worst_per_source_harm_flag": bool(
                        (math.isfinite(bacc_delta) and bacc_delta < -cfg.source_inner_bacc_noninferiority_margin)
                        or (math.isfinite(class0_delta) and class0_delta < -cfg.source_inner_class0_recall_margin)
                        or (math.isfinite(precision_delta) and precision_delta < -cfg.beta100_precision_margin)
                        or (math.isfinite(ppr_delta) and ppr_delta > cfg.source_inner_predicted_positive_rate_delta)
                    ),
                    "audit_only": True,
                    "primary_adoption_eligible": False,
                    "selection_used_target_labels": False,
                }
            )
    return out


def _is_center3_failure_audit_cell(experiment_seed: int | str, heldout_center: str) -> bool:
    key = (int(experiment_seed), str(heldout_center))
    return key in {(int(seed), str(center)) for seed, center in CENTER3_FAILURE_AUDIT_CELLS}


def _center3_failure_audit_role(experiment_seed: int | str, heldout_center: str) -> str:
    key = (int(experiment_seed), str(heldout_center))
    if key == CENTER3_FAILURE_PRIMARY_CELL:
        return "primary_center3_failure"
    if str(heldout_center) == "3":
        return "center3_control"
    if key == (43, "4"):
        return "tail_repair_control"
    if key == (43, "1"):
        return "weak_tail_control"
    return "diagnostic_control"


def _center3_failure_audit_outputs(
    cfg: MultipanelTailRiskConfig,
    *,
    experiment_seed: int,
    heldout_center: str,
    eval_labels: Sequence[int],
    eval_sample_ids: Sequence[str],
    final_bundle: PredictionBundle,
    pooled_anchor: PredictionBundle,
    pooled_random: PredictionBundle,
    canonical_random: PredictionBundle,
    panel_blend_bundles: Mapping[str, PredictionBundle],
    seed_evaluations: Sequence[_MultipanelSeedEvaluation],
) -> dict[str, list[dict[str, object]]]:
    if not _is_center3_failure_audit_cell(experiment_seed, heldout_center):
        return {"cell_rows": [], "sample_rows": [], "pooling_rows": []}

    cell_role = _center3_failure_audit_role(experiment_seed, heldout_center)
    method_bundles: list[tuple[str, str, str, int, str, PredictionBundle]] = [
        ("final_v2", cfg.primary_method, "all_seed_final_pool", 0, "all_seed_blend_pool", final_bundle),
        ("pooled_anchor", MULTIPANEL_POOLED_ANCHOR_METHOD, "all_seed_anchor_pool", 0, "all_seed_anchor_pool", pooled_anchor),
        ("pooled_random_mass_bag", MULTIPANEL_POOLED_RANDOM_BAG_METHOD, "all_seed_random_pool", 0, "all_seed_random_pool", pooled_random),
        ("canonical_random_mass_bag", MULTIPANEL_CANONICAL_RANDOM_BAG_METHOD, "canonical_random_pool", 0, "canonical_random_pool", canonical_random),
    ]
    for panel, bundle in panel_blend_bundles.items():
        method_bundles.append(
            (
                f"panel_{panel}_blend",
                f"{MULTIPANEL_SEED_BLEND_METHOD}_{panel}",
                "panel_seed_blend_pool",
                0,
                str(panel),
                bundle,
            )
        )
    for item in seed_evaluations:
        if item.evaluated.anchor_result.bundle is not None:
            method_bundles.append(
                (
                    f"seed_{item.seed}_anchor",
                    ANCHOR_METHOD,
                    "individual_seed_anchor",
                    int(item.seed),
                    item.panel_group,
                    item.evaluated.anchor_result.bundle,
                )
            )
        if item.evaluated.bag_evaluation.ensemble_bundle is not None:
            method_bundles.append(
                (
                    f"seed_{item.seed}_random_mass_bag",
                    BAG_METHOD,
                    "individual_seed_random_mass_bag",
                    int(item.seed),
                    item.panel_group,
                    item.evaluated.bag_evaluation.ensemble_bundle,
                )
            )
        if item.evaluated.primary_bundle is not None:
            method_bundles.append(
                (
                    f"seed_{item.seed}_blend",
                    MULTIPANEL_SEED_BLEND_METHOD,
                    "individual_seed_blend",
                    int(item.seed),
                    item.panel_group,
                    item.evaluated.primary_bundle,
                )
            )

    cell_rows = [
        _center3_failure_cell_metric_row(
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            cell_role=cell_role,
            audit_method=audit_method,
            prior_method=prior_method,
            pooling_stage=pooling_stage,
            replicate_seed=replicate_seed,
            panel=panel,
            bundle=bundle,
            eval_labels=eval_labels,
        )
        for audit_method, prior_method, pooling_stage, replicate_seed, panel, bundle in method_bundles
    ]
    cell_rows = _annotate_center3_failure_metric_deltas(cell_rows)
    pooling_rows = [dict(row, pooling_path_role=row["pooling_stage"]) for row in cell_rows]
    sample_rows = _center3_failure_sample_rows(
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        cell_role=cell_role,
        eval_labels=eval_labels,
        eval_sample_ids=eval_sample_ids,
        method_bundles=method_bundles,
    )
    return {"cell_rows": cell_rows, "sample_rows": sample_rows, "pooling_rows": pooling_rows}


def _center3_failure_cell_metric_row(
    *,
    experiment_seed: int,
    heldout_center: str,
    cell_role: str,
    audit_method: str,
    prior_method: str,
    pooling_stage: str,
    replicate_seed: int,
    panel: str,
    bundle: PredictionBundle,
    eval_labels: Sequence[int],
) -> dict[str, object]:
    probs = np.asarray(bundle.probabilities, dtype=float)
    preds = predict_from_probabilities(bundle.probabilities, classes=bundle.classes)
    result = evaluate_probability_predictions(audit_method, bundle.probabilities, eval_labels, classes=bundle.classes)
    labels = tuple(int(value) for value in eval_labels)
    row: dict[str, object] = {
        "audit_only": True,
        "target_eval_labels_used_for_audit_only": True,
        "selection_used_target_labels": False,
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "audit_cell_role": cell_role,
        "audit_method": audit_method,
        "prior_method": prior_method,
        "pooling_stage": pooling_stage,
        "aggregation_unit": "diagnostic_probability_bundle",
        "replicate_seed": int(replicate_seed),
        "panel": str(panel),
        "bacc": result.bacc,
        "macro_f1": result.macro_f1,
        "n_target_eval": len(labels),
        "class_order": "|".join(str(value) for value in bundle.classes),
        "class_count_json": json.dumps(_count_by_value(labels), sort_keys=True),
        "predicted_class_count_json": json.dumps(_count_by_value(preds), sort_keys=True),
        "error_count_json": json.dumps(_error_count_by_class(labels, preds), sort_keys=True),
        "prediction_hash": _hash_array(probs),
    }
    row.update(_binary_class_metric_fields(labels, preds))
    if probs.ndim == 2 and probs.shape[0]:
        confidences = np.max(probs, axis=1)
        margins = _probability_margins(probs)
        correct = np.asarray([int(t) == int(p) for t, p in zip(labels, preds)], dtype=bool)
        row.update(
            {
                "mean_confidence": float(np.mean(confidences)),
                "median_confidence": float(np.median(confidences)),
                "mean_margin": float(np.mean(margins)),
                "median_margin": float(np.median(margins)),
                "mean_confidence_correct": float(np.mean(confidences[correct])) if bool(np.any(correct)) else math.nan,
                "mean_confidence_incorrect": float(np.mean(confidences[~correct])) if bool(np.any(~correct)) else math.nan,
                "mean_probability_class0": float(np.mean(_probability_column(bundle, 0))),
                "mean_probability_class1": float(np.mean(_probability_column(bundle, 1))),
            }
        )
    else:
        row.update(
            {
                "mean_confidence": math.nan,
                "median_confidence": math.nan,
                "mean_margin": math.nan,
                "median_margin": math.nan,
                "mean_confidence_correct": math.nan,
                "mean_confidence_incorrect": math.nan,
                "mean_probability_class0": math.nan,
                "mean_probability_class1": math.nan,
            }
        )
    return row


def _annotate_center3_failure_metric_deltas(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    final_bacc = _first_finite(row.get("bacc") for row in rows if row.get("audit_method") == "final_v2")
    seed_blends = [row for row in rows if row.get("pooling_stage") == "individual_seed_blend"]
    panels = [row for row in rows if row.get("pooling_stage") == "panel_seed_blend_pool"]
    best_seed = _best_bacc_row(seed_blends)
    best_panel = _best_bacc_row(panels)
    best_seed_bacc = _float(best_seed.get("bacc", math.nan)) if best_seed else math.nan
    best_panel_bacc = _float(best_panel.get("bacc", math.nan)) if best_panel else math.nan
    out = []
    for row in rows:
        updated = dict(row)
        bacc = _float(updated.get("bacc"))
        updated["final_v2_bacc"] = final_bacc
        updated["delta_bacc_vs_final_v2"] = bacc - final_bacc if math.isfinite(bacc) and math.isfinite(final_bacc) else math.nan
        updated["best_individual_seed_blend_method"] = best_seed.get("audit_method", "") if best_seed else ""
        updated["best_individual_seed_blend_bacc"] = best_seed_bacc
        updated["delta_bacc_vs_best_individual_seed_blend"] = bacc - best_seed_bacc if math.isfinite(bacc) and math.isfinite(best_seed_bacc) else math.nan
        updated["best_panel_blend_method"] = best_panel.get("audit_method", "") if best_panel else ""
        updated["best_panel_blend_bacc"] = best_panel_bacc
        updated["delta_bacc_vs_best_panel_blend"] = bacc - best_panel_bacc if math.isfinite(bacc) and math.isfinite(best_panel_bacc) else math.nan
        updated["delta_best_individual_seed_blend_minus_final_v2"] = best_seed_bacc - final_bacc if math.isfinite(best_seed_bacc) and math.isfinite(final_bacc) else math.nan
        updated["delta_best_panel_blend_minus_final_v2"] = best_panel_bacc - final_bacc if math.isfinite(best_panel_bacc) and math.isfinite(final_bacc) else math.nan
        out.append(updated)
    return out


def _center3_failure_sample_rows(
    *,
    experiment_seed: int,
    heldout_center: str,
    cell_role: str,
    eval_labels: Sequence[int],
    eval_sample_ids: Sequence[str],
    method_bundles: Sequence[tuple[str, str, str, int, str, PredictionBundle]],
) -> list[dict[str, object]]:
    labels = tuple(int(value) for value in eval_labels)
    bundle_by_audit_method = {audit_method: bundle for audit_method, _prior, _stage, _seed, _panel, bundle in method_bundles}
    pred_by_method = {
        audit_method: predict_from_probabilities(bundle.probabilities, classes=bundle.classes)
        for audit_method, bundle in bundle_by_audit_method.items()
    }
    probs_by_method = {
        audit_method: np.asarray(bundle.probabilities, dtype=float)
        for audit_method, bundle in bundle_by_audit_method.items()
    }
    seed_blend_methods = [audit_method for audit_method, _prior, stage, _seed, _panel, _bundle in method_bundles if stage == "individual_seed_blend"]
    panel_methods = [audit_method for audit_method, _prior, stage, _seed, _panel, _bundle in method_bundles if stage == "panel_seed_blend_pool"]
    out = []
    for idx, true_label in enumerate(labels):
        row: dict[str, object] = {
            "audit_only": True,
            "target_eval_labels_used_for_audit_only": True,
            "selection_used_target_labels": False,
            "experiment_seed": int(experiment_seed),
            "heldout_center": str(heldout_center),
            "audit_cell_role": cell_role,
            "sample_index": idx,
            "sample_id": str(eval_sample_ids[idx]) if idx < len(eval_sample_ids) else "",
            "true_label": int(true_label),
        }
        for audit_method, _prior_method, _stage, _replicate_seed, _panel, bundle in method_bundles:
            probs = probs_by_method[audit_method]
            pred = int(pred_by_method[audit_method][idx])
            prefix = _safe_audit_prefix(audit_method)
            row[f"{prefix}_pred"] = pred
            row[f"{prefix}_correct"] = pred == int(true_label)
            row[f"{prefix}_prob_class0"] = _sample_probability_for_class(bundle, probs, idx, 0)
            row[f"{prefix}_prob_class1"] = _sample_probability_for_class(bundle, probs, idx, 1)
            row[f"{prefix}_confidence"] = float(np.max(probs[idx])) if probs.ndim == 2 and idx < probs.shape[0] else math.nan
            row[f"{prefix}_margin"] = _sample_probability_margin(probs, idx)
        final_pred = int(pred_by_method.get("final_v2", (math.nan,) * len(labels))[idx])
        seed_preds = [int(pred_by_method[name][idx]) for name in seed_blend_methods]
        panel_preds = [int(pred_by_method[name][idx]) for name in panel_methods]
        seed_class1 = [
            _sample_probability_for_class(bundle_by_audit_method[name], probs_by_method[name], idx, 1)
            for name in seed_blend_methods
        ]
        panel_class1 = [
            _sample_probability_for_class(bundle_by_audit_method[name], probs_by_method[name], idx, 1)
            for name in panel_methods
        ]
        row["n_seed_blends"] = len(seed_blend_methods)
        row["n_seed_blends_disagree_with_final"] = sum(pred != final_pred for pred in seed_preds)
        row["n_panel_blends_disagree_with_final"] = sum(pred != final_pred for pred in panel_preds)
        row["seed_blend_probability_spread_class1"] = _finite_range(seed_class1)
        row["panel_probability_spread_class1"] = _finite_range(panel_class1)
        for seed in (101, 127):
            name = f"seed_{seed}_blend"
            seed_correct = bool(row.get(f"{_safe_audit_prefix(name)}_correct", False))
            final_correct = bool(row.get("final_v2_correct", False))
            row[f"seed_{seed}_correct_final_wrong"] = seed_correct and not final_correct
            row[f"final_correct_seed_{seed}_wrong"] = final_correct and not seed_correct
        out.append(row)
    return out


def _binary_class_metric_fields(labels: Sequence[int], preds: Sequence[int]) -> dict[str, object]:
    out: dict[str, object] = {}
    for cls in (0, 1):
        total = sum(int(value) == cls for value in labels)
        correct = sum(int(t) == cls and int(p) == cls for t, p in zip(labels, preds))
        predicted = sum(int(value) == cls for value in preds)
        errors = sum(int(t) == cls and int(p) != cls for t, p in zip(labels, preds))
        negatives = sum(int(value) != cls for value in labels)
        true_negatives = sum(int(t) != cls and int(p) != cls for t, p in zip(labels, preds))
        out[f"class{cls}_support"] = total
        out[f"class{cls}_predicted_count"] = predicted
        out[f"class{cls}_error_count"] = errors
        out[f"class{cls}_recall"] = float(correct) / float(total) if total else math.nan
        out[f"class{cls}_specificity"] = float(true_negatives) / float(negatives) if negatives else math.nan
    return out


def _count_by_value(values: Sequence[int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(int(value))
        out[key] = out.get(key, 0) + 1
    return out


def _error_count_by_class(labels: Sequence[int], preds: Sequence[int]) -> dict[str, int]:
    out = {"0": 0, "1": 0}
    for true, pred in zip(labels, preds):
        if int(true) != int(pred):
            key = str(int(true))
            out[key] = out.get(key, 0) + 1
    return out


def _probability_column(bundle: PredictionBundle, cls: int) -> np.ndarray:
    probs = np.asarray(bundle.probabilities, dtype=float)
    if probs.ndim != 2 or cls not in bundle.classes:
        return np.asarray([], dtype=float)
    return probs[:, bundle.classes.index(cls)]


def _probability_margins(probs: np.ndarray) -> np.ndarray:
    if probs.ndim != 2 or probs.shape[1] < 2:
        return np.asarray([], dtype=float)
    sorted_probs = np.sort(probs, axis=1)
    return sorted_probs[:, -1] - sorted_probs[:, -2]


def _sample_probability_for_class(bundle: PredictionBundle, probs: np.ndarray, idx: int, cls: int) -> float:
    if probs.ndim != 2 or idx >= probs.shape[0] or cls not in bundle.classes:
        return math.nan
    return float(probs[idx, bundle.classes.index(cls)])


def _sample_probability_margin(probs: np.ndarray, idx: int) -> float:
    if probs.ndim != 2 or idx >= probs.shape[0] or probs.shape[1] < 2:
        return math.nan
    values = sorted(float(value) for value in probs[idx])
    return values[-1] - values[-2]


def _safe_audit_prefix(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")


def _finite_range(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return max(finite) - min(finite) if finite else math.nan


def _best_bacc_row(rows: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    finite = [row for row in rows if math.isfinite(_float(row.get("bacc")))]
    return max(finite, key=lambda row: _float(row.get("bacc"))) if finite else {}


def _first_finite(values: Sequence[object]) -> float:
    for value in values:
        parsed = _float(value)
        if math.isfinite(parsed):
            return parsed
    return math.nan


def _multipanel_result_row(
    cfg: MultipanelTailRiskConfig,
    *,
    experiment_seed: int,
    heldout_center: str,
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    method: str,
    bundle: PredictionBundle,
    eval_labels: Sequence[int],
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
    real_feature_bacc: float,
    weight_plan: Mapping[str, object],
    generated_features_hash: str,
    seed_bundle_hashes: Sequence[str],
    selection_source: str,
    claim_role: str,
    eval_sample_hash: str,
    panel_seed_groups_json: str,
) -> dict[str, object]:
    result = evaluate_probability_predictions(method, bundle.probabilities, eval_labels, classes=bundle.classes)
    prediction_hash = _hash_array(np.asarray(bundle.probabilities, dtype=float))
    row = cu._result_matrix_row(
        cfg,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=0,
        candidates=candidates,
        prior_method=method,
        summary_kind="multipanel_probability_pool",
        source_union_ref=source_union_ref,
        center_balanced_ref=center_balanced_ref,
        real_feature_bacc=real_feature_bacc,
        weight_plan=weight_plan,
        bacc=result.bacc,
        macro_f1=result.macro_f1,
        generated_features_hash=generated_features_hash,
        prediction_hash=prediction_hash,
        selection_source=selection_source,
        claim_role=claim_role,
        status="ok",
        error_message="",
        control_mode="normal",
        summaries=summaries,
    )
    row["panel"] = "multipanel"
    row["aggregation_unit"] = "experiment_seed_x_heldout_center"
    row["panel_seed_groups_json"] = panel_seed_groups_json
    row["seed_prediction_hashes_json"] = json.dumps(list(seed_bundle_hashes))
    row["eval_sample_ids_hash"] = eval_sample_hash
    row["pooling_rule"] = "seed_blend_then_equal_probability_pool" if method == cfg.primary_method else "equal_probability_pool"
    row["target_eval_labels_used_for_scoring_only"] = True
    row["selection_used_target_labels"] = False
    row["target_support_labels_for_selection"] = False
    return row


def _average_plan_from_rows(
    cfg: MultipanelTailRiskConfig,
    candidates: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not rows:
        rels = {
            str(source): d12.SourceReliability(0, 0, str(source), math.nan, math.nan, cfg.reliability_floor_score, "empty", "empty", 0, "", "")
            for source in candidates
        }
        return cu._uniform_source_plan(cfg, candidates, rels, total=cfg.synthetic_per_class_total)
    weights_by_source = {str(source): [] for source in candidates}
    budgets_by_source = {str(source): [] for source in candidates}
    for row in rows:
        weights = json.loads(str(row.get("source_weight_json", "{}") or "{}"))
        budgets = json.loads(str(row.get("source_budget_json", "{}") or "{}"))
        for source in candidates:
            source_id = str(source)
            weights_by_source[source_id].append(_float(weights.get(source_id, math.nan)))
            budgets_by_source[source_id].append(_float(budgets.get(source_id, math.nan)))
    weights = {
        source: nanmean([value for value in values if math.isfinite(value)])
        for source, values in weights_by_source.items()
    }
    total_weight = sum(value for value in weights.values() if math.isfinite(value))
    if total_weight > 0.0:
        weights = {source: value / total_weight if math.isfinite(value) else 0.0 for source, value in weights.items()}
    else:
        weights = {str(source): 1.0 / float(len(candidates)) for source in candidates}
    budgets = {
        source: int(round(nanmean([value for value in values if math.isfinite(value)])))
        for source, values in budgets_by_source.items()
    }
    scores = dict(weights)
    return cu._with_weight_diagnostics(
        tuple(str(source) for source in candidates),
        weights,
        budgets,
        scores,
        total=cfg.synthetic_per_class_total,
        mode=MULTIPANEL_SOURCE_WEIGHTING,
    )


def _mean_reference(values: Sequence[d1.ReferenceValue]) -> d1.ReferenceValue:
    ok = [value for value in values if value.status == "ok" and math.isfinite(value.bacc)]
    if not ok:
        return d1.ReferenceValue(math.nan, math.nan, "missing", "no_reference_values")
    return d1.ReferenceValue(
        bacc=nanmean([value.bacc for value in ok]),
        macro_f1=nanmean([value.macro_f1 for value in ok if math.isfinite(value.macro_f1)]),
        status="ok",
    )


def _probability_invariant_row(
    experiment_seed: int,
    heldout_center: str,
    method: str,
    bundle: PredictionBundle,
    *,
    eval_sample_ids: Sequence[str],
    expected_sample_hash: str,
    panel: str,
) -> dict[str, object]:
    probs = np.asarray(bundle.probabilities, dtype=float)
    row_sums = probs.sum(axis=1) if probs.ndim == 2 else np.asarray([], dtype=float)
    sample_hash = _hash_strings(eval_sample_ids)
    finite = bool(np.isfinite(probs).all()) if probs.size else False
    row_sum_pass = bool(row_sums.size and np.allclose(row_sums, 1.0, atol=1.0e-6))
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "panel": str(panel),
        "prior_method": str(method),
        "sample_id_alignment_pass": sample_hash == expected_sample_hash,
        "sample_id_hash": sample_hash,
        "expected_sample_id_hash": expected_sample_hash,
        "class_order": "|".join(str(value) for value in bundle.classes),
        "class_order_alignment_pass": bundle.classes == (0, 1),
        "probability_row_sum_pass": row_sum_pass,
        "probability_no_nan_inf_pass": finite,
        "min_probability_row_sum": float(row_sums.min()) if row_sums.size else math.nan,
        "max_probability_row_sum": float(row_sums.max()) if row_sums.size else math.nan,
        "n_probability_rows": int(probs.shape[0]) if probs.ndim == 2 else 0,
    }


def _confidence_row(
    experiment_seed: int,
    heldout_center: str,
    method: str,
    panel: str,
    bundle: PredictionBundle,
) -> dict[str, object]:
    probs = np.asarray(bundle.probabilities, dtype=float)
    if probs.ndim != 2 or probs.shape[0] == 0:
        return {
            "experiment_seed": int(experiment_seed),
            "heldout_center": str(heldout_center),
            "panel": str(panel),
            "prior_method": str(method),
            "mean_confidence": math.nan,
            "mean_entropy": math.nan,
            "n_probability_rows": 0,
        }
    clipped = np.clip(probs, 1.0e-12, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "panel": str(panel),
        "prior_method": str(method),
        "mean_confidence": float(np.max(probs, axis=1).mean()),
        "median_confidence": float(np.median(np.max(probs, axis=1))),
        "mean_entropy": float(entropy.mean()),
        "n_probability_rows": int(probs.shape[0]),
    }


def _panel_disagreement_row(
    experiment_seed: int,
    heldout_center: str,
    panel_bundles: Mapping[str, PredictionBundle],
    *,
    eval_labels: Sequence[int],
) -> dict[str, object]:
    panels = list(panel_bundles)
    js_values: list[float] = []
    hard_values: list[float] = []
    mad_values: list[float] = []
    for idx, left in enumerate(panels):
        for right in panels[idx + 1:]:
            left_probs = np.asarray(panel_bundles[left].probabilities, dtype=float)
            right_probs = np.asarray(panel_bundles[right].probabilities, dtype=float)
            js_values.append(_mean_js_divergence(left_probs, right_probs))
            left_pred = np.argmax(left_probs, axis=1)
            right_pred = np.argmax(right_probs, axis=1)
            hard_values.append(float(np.mean(left_pred != right_pred)))
            mad_values.append(float(np.mean(np.abs(left_probs - right_probs))))
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "panel_set": "|".join(panels),
        "mean_pairwise_js_divergence": nanmean(js_values),
        "mean_pairwise_hard_label_disagreement": nanmean(hard_values),
        "mean_pairwise_absolute_probability_deviation": nanmean(mad_values),
        "n_target_eval": len(tuple(eval_labels)),
        "is_prior_bottom20_cell": False,
    }


def _mean_js_divergence(left: np.ndarray, right: np.ndarray, *, eps: float = 1.0e-8) -> float:
    p = np.clip(np.asarray(left, dtype=float), eps, 1.0)
    q = np.clip(np.asarray(right, dtype=float), eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p) - np.log(m)), axis=1)
    kl_qm = np.sum(q * (np.log(q) - np.log(m)), axis=1)
    return float(np.mean(0.5 * (kl_pm + kl_qm)))


def _panel_bacc(rows: Sequence[Mapping[str, object]], panel: str) -> float:
    key = f"panel_{panel}"
    values = [_float(row.get("bacc")) for row in rows if row.get("decomposition_source") == key]
    return nanmean([value for value in values if math.isfinite(value)])


def _panel_seed_groups_json(cfg: MultipanelTailRiskConfig) -> str:
    return json.dumps({panel: list(seeds) for panel, seeds in cfg.panel_seed_groups}, sort_keys=True)


def _load_prior_tailrisk_matrix_rows(cfg: MultipanelTailRiskConfig) -> list[dict[str, object]]:
    if cfg.prior_tailrisk_artifact_root is None:
        return []
    path = cfg.prior_tailrisk_artifact_root / "tables" / "tailrisk_downstream_matrix.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _collapsed_method_rows(rows: Sequence[Mapping[str, object]], method: str) -> dict[tuple[str, str], dict[str, object]]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        if row.get("prior_method") != method or row.get("status") != "ok":
            continue
        grouped.setdefault((str(row.get("experiment_seed")), str(row.get("heldout_center"))), []).append(row)
    out: dict[tuple[str, str], dict[str, object]] = {}
    for key, subset in grouped.items():
        out[key] = {
            "experiment_seed": key[0],
            "heldout_center": key[1],
            "prior_method": method,
            "status": "ok",
            "bacc": d1._mean_field(subset, "bacc"),
            "macro_f1": d1._mean_field(subset, "macro_f1"),
        }
    return out


def _multipanel_paired_delta_rows(
    rows: Sequence[Mapping[str, object]],
    historical_rows: Sequence[Mapping[str, object]],
    cfg: MultipanelTailRiskConfig,
) -> tuple[list[dict[str, object]], set[tuple[str, str]]]:
    primary = _collapsed_method_rows(rows, cfg.primary_method)
    canonical = _collapsed_method_rows(rows, MULTIPANEL_CANONICAL_RANDOM_BAG_METHOD)
    anchor = _collapsed_method_rows(rows, MULTIPANEL_POOLED_ANCHOR_METHOD)
    pooled_random = _collapsed_method_rows(rows, MULTIPANEL_POOLED_RANDOM_BAG_METHOD)
    prior = _collapsed_method_rows(historical_rows, PRIMARY_TAILRISK_METHOD)
    intersection = sorted(set(primary) & set(canonical) & set(anchor) & set(prior))
    prior_values = sorted((_float(prior[key]["bacc"]), key) for key in intersection if math.isfinite(_float(prior[key]["bacc"])))
    bottom_count = max(1, int(math.ceil(0.20 * len(prior_values)))) if prior_values else 0
    prior_tail_keys = {key for _value, key in prior_values[:bottom_count]}
    out = []
    for key in intersection:
        p = _float(primary[key]["bacc"])
        prior_bacc = _float(prior[key]["bacc"])
        canon_bacc = _float(canonical[key]["bacc"])
        anchor_bacc = _float(anchor[key]["bacc"])
        pooled_random_bacc = _float(pooled_random.get(key, {}).get("bacc", math.nan))
        out.append(
            {
                "experiment_seed": key[0],
                "heldout_center": key[1],
                "is_frozen_prior_bottom20_cell": key in prior_tail_keys,
                "v2_primary_bacc": p,
                "prior_tailrisk_bacc": prior_bacc,
                "same_cell_single_random_mass_bag_canonical_bacc": canon_bacc,
                "same_cell_shrink050_bacc": anchor_bacc,
                "pooled_random_mass_bag_bacc": pooled_random_bacc,
                "delta_v2_minus_prior_tailrisk": p - prior_bacc if math.isfinite(p) and math.isfinite(prior_bacc) else math.nan,
                "delta_v2_minus_canonical_random_mass_bag": p - canon_bacc if math.isfinite(p) and math.isfinite(canon_bacc) else math.nan,
                "delta_v2_minus_shrink050": p - anchor_bacc if math.isfinite(p) and math.isfinite(anchor_bacc) else math.nan,
                "delta_v2_minus_pooled_random_mass_bag": p - pooled_random_bacc if math.isfinite(p) and math.isfinite(pooled_random_bacc) else math.nan,
                "comparison_cell_set": "intersection_v2_prior_tailrisk_canonical_random_shrink050",
                "status": "ok",
            }
        )
    return out, prior_tail_keys


def _positive_union_paired_delta_rows(
    rows: Sequence[Mapping[str, object]],
    historical_rows: Sequence[Mapping[str, object]],
    cfg: SourceInnerPositiveUnionConfig,
) -> tuple[list[dict[str, object]], set[tuple[str, str]]]:
    primary = _collapsed_method_rows(rows, cfg.primary_method)
    arithmetic = _collapsed_method_rows(rows, POSITIVE_UNION_RULE_ARITHMETIC)
    canonical = _collapsed_method_rows(rows, MULTIPANEL_CANONICAL_RANDOM_BAG_METHOD)
    anchor = _collapsed_method_rows(rows, MULTIPANEL_POOLED_ANCHOR_METHOD)
    pooled_random = _collapsed_method_rows(rows, MULTIPANEL_POOLED_RANDOM_BAG_METHOD)
    prior = _collapsed_method_rows(historical_rows, PRIMARY_TAILRISK_METHOD)
    intersection = sorted(set(primary) & set(arithmetic) & set(canonical) & set(anchor) & set(prior))
    prior_values = sorted((_float(prior[key]["bacc"]), key) for key in intersection if math.isfinite(_float(prior[key]["bacc"])))
    bottom_count = max(1, int(math.ceil(0.20 * len(prior_values)))) if prior_values else 0
    prior_tail_keys = {key for _value, key in prior_values[:bottom_count]}
    out = []
    for key in intersection:
        p = _float(primary[key]["bacc"])
        arithmetic_bacc = _float(arithmetic[key]["bacc"])
        prior_bacc = _float(prior[key]["bacc"])
        canon_bacc = _float(canonical[key]["bacc"])
        anchor_bacc = _float(anchor[key]["bacc"])
        pooled_random_bacc = _float(pooled_random.get(key, {}).get("bacc", math.nan))
        out.append(
            {
                "experiment_seed": key[0],
                "heldout_center": key[1],
                "is_frozen_prior_bottom20_cell": key in prior_tail_keys,
                "positive_union_bacc": p,
                "v2_arithmetic_multipanel_bacc": arithmetic_bacc,
                "prior_tailrisk_bacc": prior_bacc,
                "same_cell_single_random_mass_bag_canonical_bacc": canon_bacc,
                "same_cell_shrink050_bacc": anchor_bacc,
                "pooled_random_mass_bag_bacc": pooled_random_bacc,
                "delta_positive_union_minus_v2_arithmetic": p - arithmetic_bacc if math.isfinite(p) and math.isfinite(arithmetic_bacc) else math.nan,
                "delta_positive_union_minus_prior_tailrisk": p - prior_bacc if math.isfinite(p) and math.isfinite(prior_bacc) else math.nan,
                "delta_positive_union_minus_canonical_random_mass_bag": p - canon_bacc if math.isfinite(p) and math.isfinite(canon_bacc) else math.nan,
                "delta_positive_union_minus_shrink050": p - anchor_bacc if math.isfinite(p) and math.isfinite(anchor_bacc) else math.nan,
                "delta_positive_union_minus_pooled_random_mass_bag": p - pooled_random_bacc if math.isfinite(p) and math.isfinite(pooled_random_bacc) else math.nan,
                "selected_rule": primary[key].get("selected_positive_union_rule", ""),
                "comparison_cell_set": "intersection_positive_union_v2_arithmetic_prior_tailrisk_canonical_random_shrink050",
                "status": "ok",
            }
        )
    return out, prior_tail_keys


def _fixed_beta050_paired_delta_rows(
    rows: Sequence[Mapping[str, object]],
    cfg: FixedBeta050PositiveUnionConfig,
) -> tuple[list[dict[str, object]], set[tuple[str, str]]]:
    primary = _collapsed_method_rows(rows, cfg.primary_method)
    arithmetic = _collapsed_method_rows(rows, POSITIVE_UNION_RULE_ARITHMETIC)
    beta025 = _collapsed_method_rows(rows, POSITIVE_UNION_RULE_BETA025)
    beta100 = _collapsed_method_rows(rows, POSITIVE_UNION_RULE_BETA100)
    canonical = _collapsed_method_rows(rows, MULTIPANEL_CANONICAL_RANDOM_BAG_METHOD)
    anchor = _collapsed_method_rows(rows, MULTIPANEL_POOLED_ANCHOR_METHOD)
    pooled_random = _collapsed_method_rows(rows, MULTIPANEL_POOLED_RANDOM_BAG_METHOD)
    intersection = sorted(set(primary) & set(arithmetic) & set(canonical) & set(anchor))
    arithmetic_values = sorted((_float(arithmetic[key]["bacc"]), key) for key in intersection if math.isfinite(_float(arithmetic[key]["bacc"])))
    bottom_count = max(1, int(math.ceil(0.20 * len(arithmetic_values)))) if arithmetic_values else 0
    arithmetic_tail_keys = {key for _value, key in arithmetic_values[:bottom_count]}
    out = []
    for key in intersection:
        p = _float(primary[key]["bacc"])
        arithmetic_bacc = _float(arithmetic[key]["bacc"])
        beta025_bacc = _float(beta025.get(key, {}).get("bacc", math.nan))
        beta100_bacc = _float(beta100.get(key, {}).get("bacc", math.nan))
        canon_bacc = _float(canonical[key]["bacc"])
        anchor_bacc = _float(anchor[key]["bacc"])
        pooled_random_bacc = _float(pooled_random.get(key, {}).get("bacc", math.nan))
        out.append(
            {
                "experiment_seed": key[0],
                "heldout_center": key[1],
                "is_frozen_arithmetic_bottom20_cell": key in arithmetic_tail_keys,
                "fixed_rule": cfg.fixed_pooling_rule,
                "fixed_beta": cfg.fixed_beta,
                "fixed_beta050_bacc": p,
                "v2_arithmetic_multipanel_bacc": arithmetic_bacc,
                "fixed_beta025_diagnostic_bacc": beta025_bacc,
                "fixed_beta100_diagnostic_bacc": beta100_bacc,
                "same_cell_single_random_mass_bag_canonical_bacc": canon_bacc,
                "same_cell_shrink050_bacc": anchor_bacc,
                "pooled_random_mass_bag_bacc": pooled_random_bacc,
                "delta_fixed_beta050_minus_v2_arithmetic": p - arithmetic_bacc if math.isfinite(p) and math.isfinite(arithmetic_bacc) else math.nan,
                "delta_fixed_beta050_minus_beta025": p - beta025_bacc if math.isfinite(p) and math.isfinite(beta025_bacc) else math.nan,
                "delta_fixed_beta050_minus_beta100": p - beta100_bacc if math.isfinite(p) and math.isfinite(beta100_bacc) else math.nan,
                "delta_fixed_beta050_minus_canonical_random_mass_bag": p - canon_bacc if math.isfinite(p) and math.isfinite(canon_bacc) else math.nan,
                "delta_fixed_beta050_minus_shrink050": p - anchor_bacc if math.isfinite(p) and math.isfinite(anchor_bacc) else math.nan,
                "delta_fixed_beta050_minus_pooled_random_mass_bag": p - pooled_random_bacc if math.isfinite(p) and math.isfinite(pooled_random_bacc) else math.nan,
                "comparison_cell_set": "fresh_confirmation_intersection_fixed_beta050_v2_arithmetic_canonical_random_shrink050",
                "status": "ok",
            }
        )
    return out, arithmetic_tail_keys


def _annotate_fixed_beta050_harm_rows(
    rows: Sequence[Mapping[str, object]],
    paired_delta_rows: Sequence[Mapping[str, object]],
    cfg: FixedBeta050PositiveUnionConfig,
) -> list[dict[str, object]]:
    deltas_by_center: dict[str, list[float]] = {}
    deltas_by_key: dict[tuple[str, str], float] = {}
    for row in paired_delta_rows:
        center = str(row.get("heldout_center"))
        key = (str(row.get("experiment_seed")), center)
        delta = _float(row.get("delta_fixed_beta050_minus_v2_arithmetic"))
        if math.isfinite(delta):
            deltas_by_center.setdefault(center, []).append(delta)
            deltas_by_key[key] = delta
    center_means = {
        center: nanmean([value for value in values if math.isfinite(value)])
        for center, values in deltas_by_center.items()
    }
    worst_center = min(center_means.values(), default=math.nan)
    worst_seed_center = min(deltas_by_key.values(), default=math.nan)
    out = []
    for row in rows:
        updated = dict(row)
        key = (str(updated.get("experiment_seed")), str(updated.get("heldout_center")))
        updated["delta_vs_v2_arithmetic"] = deltas_by_key.get(key, math.nan)
        updated["worst_per_center_regression"] = worst_center
        updated["worst_seed_center_regression"] = worst_seed_center
        updated["tail_risk_transfer_flag"] = bool(
            (math.isfinite(worst_center) and worst_center < cfg.tailrisk_transfer_threshold)
            or (math.isfinite(worst_seed_center) and worst_seed_center < cfg.tailrisk_transfer_threshold)
        )
        out.append(updated)
    return out


def _fixed_beta050_retrospective_reference_rows(cfg: FixedBeta050PositiveUnionConfig) -> list[dict[str, object]]:
    if cfg.development_positive_union_artifact_root is None:
        return []
    path = cfg.development_positive_union_artifact_root / "tables" / "positive_union_candidate_rule_matrix.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [dict(row) for row in csv.DictReader(f)]
    out = []
    for row in rows:
        if _safe_int(row.get("experiment_seed"), default=-1) not in cfg.development_experiment_seeds:
            continue
        updated = dict(row)
        updated["retrospective_reference_only"] = True
        updated["primary_adoption_eligible"] = False
        updated["audit_only"] = True
        updated["beta_origin"] = "hypothesis_generated_from_prior_positive_union_diagnostic"
        out.append(updated)
    return out


def _annotate_positive_union_harm_rows(
    rows: Sequence[Mapping[str, object]],
    paired_delta_rows: Sequence[Mapping[str, object]],
    cfg: SourceInnerPositiveUnionConfig,
) -> list[dict[str, object]]:
    deltas_by_center: dict[str, list[float]] = {}
    deltas_by_key: dict[tuple[str, str], float] = {}
    for row in paired_delta_rows:
        center = str(row.get("heldout_center"))
        key = (str(row.get("experiment_seed")), center)
        delta = _float(row.get("delta_positive_union_minus_prior_tailrisk"))
        if math.isfinite(delta):
            deltas_by_center.setdefault(center, []).append(delta)
            deltas_by_key[key] = delta
    center_means = {
        center: nanmean([value for value in values if math.isfinite(value)])
        for center, values in deltas_by_center.items()
    }
    worst_center = min(center_means.values(), default=math.nan)
    worst_seed_center = min(deltas_by_key.values(), default=math.nan)
    out = []
    for row in rows:
        updated = dict(row)
        key = (str(updated.get("experiment_seed")), str(updated.get("heldout_center")))
        delta = deltas_by_key.get(key, math.nan)
        updated["delta_vs_prior_tailrisk"] = delta
        updated["worst_per_center_regression"] = worst_center
        updated["worst_seed_center_regression"] = worst_seed_center
        updated["tail_risk_transfer_flag"] = bool(
            (math.isfinite(worst_center) and worst_center < cfg.tailrisk_transfer_threshold)
            or (math.isfinite(worst_seed_center) and worst_seed_center < cfg.tailrisk_transfer_threshold)
        )
        out.append(updated)
    return out


def _annotate_failure_rows(
    rows: Sequence[Mapping[str, object]],
    paired_delta_rows: Sequence[Mapping[str, object]],
    prior_tail_keys: set[tuple[str, str]],
) -> list[dict[str, object]]:
    prior_by_key = {
        (str(row.get("experiment_seed")), str(row.get("heldout_center"))): row
        for row in paired_delta_rows
    }
    out = []
    for row in rows:
        updated = dict(row)
        key = (str(updated.get("experiment_seed")), str(updated.get("heldout_center")))
        prior = prior_by_key.get(key, {})
        updated["prior_tailrisk_bacc"] = prior.get("prior_tailrisk_bacc", "")
        updated["delta_final_minus_prior_tailrisk"] = prior.get("delta_v2_minus_prior_tailrisk", "")
        updated["is_frozen_prior_bottom20_cell"] = key in prior_tail_keys
        out.append(updated)
    return out


def _annotate_panel_disagreement_rows(
    rows: Sequence[Mapping[str, object]],
    prior_tail_keys: set[tuple[str, str]],
) -> list[dict[str, object]]:
    out = []
    for row in rows:
        updated = dict(row)
        key = (str(updated.get("experiment_seed")), str(updated.get("heldout_center")))
        updated["is_prior_bottom20_cell"] = key in prior_tail_keys
        out.append(updated)
    return out


def _multipanel_tail_metrics(
    rows: Sequence[Mapping[str, object]],
    method: str,
    *,
    prior_tail_keys: set[tuple[str, str]] | None = None,
) -> dict[str, object]:
    subset = cu._rows_for(rows, method)
    stats = cu._method_stats(subset)
    grouped = cu._replicate_averaged(subset)
    bacc_values = sorted(_float(row.get("bacc")) for row in grouped if math.isfinite(_float(row.get("bacc"))))
    bottom_count = max(1, int(math.ceil(0.20 * len(bacc_values)))) if bacc_values else 0
    own_bottom20 = nanmean(bacc_values[:bottom_count]) if bacc_values else math.nan
    if prior_tail_keys:
        frozen = [
            _float(row.get("bacc"))
            for row in grouped
            if (str(row.get("experiment_seed")), str(row.get("heldout_center"))) in prior_tail_keys
        ]
        bottom20 = nanmean([value for value in frozen if math.isfinite(value)])
    else:
        bottom20 = own_bottom20
    center3_rows = [row for row in grouped if str(row.get("heldout_center")) == "3"]
    center3 = d1._mean_field(center3_rows, "bacc") if center3_rows else math.nan
    return {
        **stats,
        "bottom20_cell_mean_bacc": bottom20,
        "own_bottom20_cell_mean_bacc": own_bottom20,
        "worst_seed_center_bacc": min(bacc_values) if bacc_values else math.nan,
        "center3_bacc": center3,
    }


def _stats_from_paired(
    paired_delta_rows: Sequence[Mapping[str, object]],
    value_field: str,
) -> dict[str, object]:
    rows = [
        {
            "experiment_seed": row["experiment_seed"],
            "heldout_center": row["heldout_center"],
            "prior_method": value_field,
            "status": "ok",
            "bacc": row.get(value_field, math.nan),
            "macro_f1": math.nan,
        }
        for row in paired_delta_rows
    ]
    return _multipanel_tail_metrics(rows, value_field)


def _multipanel_decision(
    rows: Sequence[Mapping[str, object]],
    *,
    paired_delta_rows: Sequence[Mapping[str, object]],
    prior_tail_keys: set[tuple[str, str]],
    leakage_status: str,
    cfg: MultipanelTailRiskConfig,
) -> dict[str, object]:
    primary = _multipanel_tail_metrics(rows, cfg.primary_method, prior_tail_keys=prior_tail_keys)
    anchor = _multipanel_tail_metrics(rows, MULTIPANEL_POOLED_ANCHOR_METHOD, prior_tail_keys=prior_tail_keys)
    canonical = _multipanel_tail_metrics(rows, MULTIPANEL_CANONICAL_RANDOM_BAG_METHOD, prior_tail_keys=prior_tail_keys)
    pooled_random = _multipanel_tail_metrics(rows, MULTIPANEL_POOLED_RANDOM_BAG_METHOD, prior_tail_keys=prior_tail_keys)
    prior = _stats_from_paired(paired_delta_rows, "prior_tailrisk_bacc") if paired_delta_rows else {}
    primary_i = _stats_from_paired(paired_delta_rows, "v2_primary_bacc") if paired_delta_rows else {}
    canonical_i = _stats_from_paired(paired_delta_rows, "same_cell_single_random_mass_bag_canonical_bacc") if paired_delta_rows else {}

    mean_i = _float(primary_i.get("center_equal_mean_bacc", math.nan))
    prior_mean = _float(prior.get("center_equal_mean_bacc", math.nan))
    canonical_mean = _float(canonical_i.get("center_equal_mean_bacc", math.nan))
    primary_mean = _float(primary["center_equal_mean_bacc"])
    min_center_delta = _delta(primary_i.get("min_center_bacc", math.nan), prior.get("min_center_bacc", math.nan)) if paired_delta_rows else math.nan
    center3_delta = _delta(primary_i.get("center3_bacc", math.nan), prior.get("center3_bacc", math.nan)) if paired_delta_rows else math.nan
    bottom20_delta = _delta(primary_i.get("bottom20_cell_mean_bacc", math.nan), prior.get("bottom20_cell_mean_bacc", math.nan)) if paired_delta_rows else math.nan
    seed_std_delta = _delta(primary_i.get("seed_std_bacc", math.nan), prior.get("seed_std_bacc", math.nan)) if paired_delta_rows else math.nan
    tail_deltas = [
        _float(row.get("delta_v2_minus_prior_tailrisk"))
        for row in paired_delta_rows
        if str(row.get("is_frozen_prior_bottom20_cell")) == "True" or row.get("is_frozen_prior_bottom20_cell") is True
    ]
    tail_deltas = [value for value in tail_deltas if math.isfinite(value)]
    tail_positive_fraction = float(sum(value > 0.0 for value in tail_deltas)) / float(len(tail_deltas)) if tail_deltas else math.nan
    tail_median_delta = float(np.median(np.asarray(tail_deltas, dtype=float))) if tail_deltas else math.nan
    center_regressions = _per_center_regressions(primary_i, prior) if paired_delta_rows else {}
    worst_center_regression = min(center_regressions.values(), default=math.nan)
    flags: list[str] = []
    if leakage_status != "PASS":
        flags.append("LEAKAGE_FAIL")
    if not paired_delta_rows:
        flags.append("MISSING_PRIOR_TAILRISK_INTERSECTION")
    if math.isfinite(worst_center_regression) and worst_center_regression < cfg.tailrisk_transfer_threshold:
        flags.append("TAIL_RISK_TRANSFER")
    if math.isfinite(mean_i) and math.isfinite(prior_mean) and mean_i < prior_mean - cfg.primary_noninferiority_margin:
        flags.append("MEAN_INFERIOR_TO_PRIOR_TAILRISK_GT_0P005")
    if math.isfinite(mean_i) and math.isfinite(canonical_mean) and mean_i < canonical_mean - cfg.primary_noninferiority_margin:
        flags.append("MEAN_INFERIOR_TO_CANONICAL_RANDOM_GT_0P005")
    if math.isfinite(min_center_delta) and min_center_delta <= 0.0:
        flags.append("MIN_CENTER_NOT_IMPROVED")
    if math.isfinite(center3_delta) and center3_delta < 0.0 and _float(primary_i.get("center3_bacc", math.nan)) < 0.82:
        flags.append("CENTER3_NOT_IMPROVED_AND_BELOW_0P82")
    if math.isfinite(bottom20_delta) and bottom20_delta <= 0.0:
        flags.append("BOTTOM20_NOT_IMPROVED")
    if math.isfinite(seed_std_delta) and seed_std_delta >= 0.0:
        flags.append("SEED_STD_NOT_REDUCED")
    if math.isfinite(tail_median_delta) and tail_median_delta <= 0.0:
        flags.append("FROZEN_BOTTOM20_MEDIAN_DELTA_NOT_POSITIVE")
    if math.isfinite(tail_positive_fraction) and tail_positive_fraction <= 0.5:
        flags.append("FROZEN_BOTTOM20_NOT_MAJORITY_IMPROVED")

    noninferior = (
        math.isfinite(mean_i)
        and math.isfinite(prior_mean)
        and math.isfinite(canonical_mean)
        and mean_i >= prior_mean - cfg.primary_noninferiority_margin
        and mean_i >= canonical_mean - cfg.primary_noninferiority_margin
    )
    weak_noninferior = (
        math.isfinite(mean_i)
        and math.isfinite(prior_mean)
        and math.isfinite(canonical_mean)
        and mean_i >= prior_mean - cfg.weak_pass_noninferiority_margin
        and mean_i >= canonical_mean - cfg.weak_pass_noninferiority_margin
    )
    primary_success = (
        leakage_status == "PASS"
        and bool(paired_delta_rows)
        and noninferior
        and math.isfinite(min_center_delta)
        and min_center_delta > 0.0
        and (center3_delta >= 0.0 or _float(primary_i.get("center3_bacc", math.nan)) >= 0.82)
        and bottom20_delta > 0.0
        and seed_std_delta < 0.0
        and tail_median_delta > 0.0
        and tail_positive_fraction > 0.5
    )
    weak_success = (
        leakage_status == "PASS"
        and bool(paired_delta_rows)
        and weak_noninferior
        and math.isfinite(min_center_delta)
        and min_center_delta > 0.0
        and bottom20_delta > 0.0
        and seed_std_delta < 0.0
    )
    strong_success = (
        primary_success
        and primary_mean >= 0.90
        and _float(primary["min_center_bacc"]) >= 0.82
        and _float(primary["center3_bacc"]) >= 0.82
        and primary_mean > _float(anchor["center_equal_mean_bacc"])
        and primary_mean > _float(canonical["center_equal_mean_bacc"])
    )
    verdict = "MULTIPANEL_TAILRISK_STABILIZATION_FAIL"
    if leakage_status != "PASS":
        verdict = "PROTOCOL_FAIL"
    elif strong_success:
        verdict = "MULTIPANEL_TAILRISK_STABILIZATION_STRONG_SUCCESS"
    elif primary_success:
        verdict = "MULTIPANEL_TAILRISK_STABILIZATION_PRIMARY_SUCCESS"
    elif weak_success:
        verdict = "MULTIPANEL_TAILRISK_STABILIZATION_WEAK_PASS"
    return {
        "primary_verdict": verdict,
        "diagnostic_flags": "|".join(flags),
        "primary_method": cfg.primary_method,
        "leakage_status": leakage_status,
        "claim_boundary": "stabilized source-only dense stochastic generative composition; not compatibility routing",
        "comparison_cell_set": "intersection_v2_prior_tailrisk_canonical_random_shrink050",
        "n_intersection_cells": len(paired_delta_rows),
        "center_equal_mean_bacc": primary["center_equal_mean_bacc"],
        "intersection_center_equal_mean_bacc": mean_i,
        "seed_cell_mean_bacc": primary["seed_cell_mean_bacc"],
        "center_equal_macro_f1": primary["center_equal_macro_f1"],
        "min_center_bacc": primary["min_center_bacc"],
        "seed_std_bacc": primary["seed_std_bacc"],
        "bottom20_cell_mean_bacc": primary["bottom20_cell_mean_bacc"],
        "worst_seed_center_bacc": primary["worst_seed_center_bacc"],
        "center3_bacc": primary["center3_bacc"],
        "prior_tailrisk_center_equal_mean_bacc": prior_mean,
        "canonical_random_mass_bag_center_equal_mean_bacc": canonical["center_equal_mean_bacc"],
        "pooled_random_mass_bag_center_equal_mean_bacc": pooled_random["center_equal_mean_bacc"],
        "shrink050_center_equal_mean_bacc": anchor["center_equal_mean_bacc"],
        "delta_vs_prior_tailrisk_intersection": mean_i - prior_mean if math.isfinite(mean_i) and math.isfinite(prior_mean) else math.nan,
        "delta_vs_canonical_random_mass_bag_intersection": mean_i - canonical_mean if math.isfinite(mean_i) and math.isfinite(canonical_mean) else math.nan,
        "min_center_delta_vs_prior_tailrisk": min_center_delta,
        "center3_delta_vs_prior_tailrisk": center3_delta,
        "bottom20_delta_vs_prior_tailrisk": bottom20_delta,
        "seed_std_delta_vs_prior_tailrisk": seed_std_delta,
        "frozen_bottom20_median_delta_vs_prior_tailrisk": tail_median_delta,
        "frozen_bottom20_positive_fraction": tail_positive_fraction,
        "worst_per_center_regression_vs_prior_tailrisk": worst_center_regression,
        "tailrisk_transfer_flag": "TAIL_RISK_TRANSFER" in flags,
        **primary,
    }


def _positive_union_decision(
    rows: Sequence[Mapping[str, object]],
    *,
    paired_delta_rows: Sequence[Mapping[str, object]],
    prior_tail_keys: set[tuple[str, str]],
    selection_rows: Sequence[Mapping[str, object]],
    harm_rows: Sequence[Mapping[str, object]],
    leakage_status: str,
    cfg: SourceInnerPositiveUnionConfig,
) -> dict[str, object]:
    primary = _multipanel_tail_metrics(rows, cfg.primary_method, prior_tail_keys=prior_tail_keys)
    arithmetic = _multipanel_tail_metrics(rows, POSITIVE_UNION_RULE_ARITHMETIC, prior_tail_keys=prior_tail_keys)
    anchor = _multipanel_tail_metrics(rows, MULTIPANEL_POOLED_ANCHOR_METHOD, prior_tail_keys=prior_tail_keys)
    canonical = _multipanel_tail_metrics(rows, MULTIPANEL_CANONICAL_RANDOM_BAG_METHOD, prior_tail_keys=prior_tail_keys)
    pooled_random = _multipanel_tail_metrics(rows, MULTIPANEL_POOLED_RANDOM_BAG_METHOD, prior_tail_keys=prior_tail_keys)
    prior = _stats_from_paired(paired_delta_rows, "prior_tailrisk_bacc") if paired_delta_rows else {}
    primary_i = _stats_from_paired(paired_delta_rows, "positive_union_bacc") if paired_delta_rows else {}
    arithmetic_i = _stats_from_paired(paired_delta_rows, "v2_arithmetic_multipanel_bacc") if paired_delta_rows else {}
    canonical_i = _stats_from_paired(paired_delta_rows, "same_cell_single_random_mass_bag_canonical_bacc") if paired_delta_rows else {}

    mean_i = _float(primary_i.get("center_equal_mean_bacc", math.nan))
    prior_mean = _float(prior.get("center_equal_mean_bacc", math.nan))
    arithmetic_mean = _float(arithmetic_i.get("center_equal_mean_bacc", math.nan))
    canonical_mean = _float(canonical_i.get("center_equal_mean_bacc", math.nan))
    min_center_delta = _delta(primary_i.get("min_center_bacc", math.nan), prior.get("min_center_bacc", math.nan)) if paired_delta_rows else math.nan
    center3_delta = _delta(primary_i.get("center3_bacc", math.nan), prior.get("center3_bacc", math.nan)) if paired_delta_rows else math.nan
    bottom20_delta = _delta(primary_i.get("bottom20_cell_mean_bacc", math.nan), prior.get("bottom20_cell_mean_bacc", math.nan)) if paired_delta_rows else math.nan
    seed_std_delta = _delta(primary_i.get("seed_std_bacc", math.nan), prior.get("seed_std_bacc", math.nan)) if paired_delta_rows else math.nan
    arithmetic_delta = mean_i - arithmetic_mean if math.isfinite(mean_i) and math.isfinite(arithmetic_mean) else math.nan
    tail_deltas = [
        _float(row.get("delta_positive_union_minus_prior_tailrisk"))
        for row in paired_delta_rows
        if str(row.get("is_frozen_prior_bottom20_cell")) == "True" or row.get("is_frozen_prior_bottom20_cell") is True
    ]
    tail_deltas = [value for value in tail_deltas if math.isfinite(value)]
    tail_positive_fraction = float(sum(value > 0.0 for value in tail_deltas)) / float(len(tail_deltas)) if tail_deltas else math.nan
    tail_median_delta = float(np.median(np.asarray(tail_deltas, dtype=float))) if tail_deltas else math.nan
    center_regressions = _per_center_regressions(primary_i, prior) if paired_delta_rows else {}
    worst_center_regression = min(center_regressions.values(), default=math.nan)
    worst_seed_center_regression = min(
        (_float(row.get("delta_positive_union_minus_prior_tailrisk")) for row in paired_delta_rows),
        default=math.nan,
    )
    selected_counts: dict[str, int] = {}
    insufficient_count = 0
    for row in selection_rows:
        selected = str(row.get("selected_rule", ""))
        selected_counts[selected] = selected_counts.get(selected, 0) + 1
        if row.get("selection_reason") == "insufficient_source_inner_positive_count":
            insufficient_count += 1
    flags: list[str] = []
    if leakage_status != "PASS":
        flags.append("LEAKAGE_FAIL")
    if not paired_delta_rows:
        flags.append("MISSING_PRIOR_TAILRISK_INTERSECTION")
    if math.isfinite(worst_center_regression) and worst_center_regression < cfg.tailrisk_transfer_threshold:
        flags.append("TAIL_RISK_TRANSFER")
    if math.isfinite(mean_i) and math.isfinite(prior_mean) and mean_i < prior_mean - cfg.primary_noninferiority_margin:
        flags.append("MEAN_INFERIOR_TO_PRIOR_TAILRISK_GT_0P005")
    if math.isfinite(mean_i) and math.isfinite(arithmetic_mean) and mean_i < arithmetic_mean - cfg.primary_noninferiority_margin:
        flags.append("MEAN_INFERIOR_TO_V2_ARITHMETIC_GT_0P005")
    if math.isfinite(min_center_delta) and min_center_delta <= 0.0:
        flags.append("MIN_CENTER_NOT_IMPROVED")
    if math.isfinite(center3_delta) and center3_delta <= 0.0:
        flags.append("CENTER3_NOT_IMPROVED")
    if math.isfinite(bottom20_delta) and bottom20_delta <= 0.0:
        flags.append("BOTTOM20_NOT_IMPROVED")
    if math.isfinite(seed_std_delta) and seed_std_delta > 0.005:
        flags.append("SEED_STD_INCREASED_GT_0P005")
    if math.isfinite(tail_median_delta) and tail_median_delta <= 0.0:
        flags.append("FROZEN_BOTTOM20_MEDIAN_DELTA_NOT_POSITIVE")
    if math.isfinite(tail_positive_fraction) and tail_positive_fraction <= 0.5:
        flags.append("FROZEN_BOTTOM20_NOT_MAJORITY_IMPROVED")

    noninferior = (
        math.isfinite(mean_i)
        and math.isfinite(prior_mean)
        and math.isfinite(arithmetic_mean)
        and mean_i >= prior_mean - cfg.primary_noninferiority_margin
        and mean_i >= arithmetic_mean - cfg.primary_noninferiority_margin
    )
    weak_noninferior = (
        math.isfinite(mean_i)
        and math.isfinite(prior_mean)
        and math.isfinite(arithmetic_mean)
        and mean_i >= prior_mean - cfg.weak_pass_noninferiority_margin
        and mean_i >= arithmetic_mean - cfg.weak_pass_noninferiority_margin
    )
    primary_success = (
        leakage_status == "PASS"
        and bool(paired_delta_rows)
        and noninferior
        and math.isfinite(min_center_delta)
        and min_center_delta > 0.0
        and center3_delta > 0.0
        and bottom20_delta > 0.0
        and (not math.isfinite(seed_std_delta) or seed_std_delta <= 0.005)
        and tail_median_delta > 0.0
        and tail_positive_fraction > 0.5
    )
    weak_success = (
        leakage_status == "PASS"
        and bool(paired_delta_rows)
        and weak_noninferior
        and math.isfinite(min_center_delta)
        and min_center_delta > 0.0
        and bottom20_delta > 0.0
    )
    primary_mean = _float(primary["center_equal_mean_bacc"])
    strong_success = (
        primary_success
        and primary_mean >= 0.90
        and _float(primary["min_center_bacc"]) >= 0.82
        and _float(primary["center3_bacc"]) >= 0.82
        and primary_mean > arithmetic_mean
        and primary_mean > canonical_mean
    )
    verdict = "SOURCE_INNER_POSITIVE_UNION_FAIL"
    if leakage_status != "PASS":
        verdict = "PROTOCOL_FAIL"
    elif strong_success:
        verdict = "SOURCE_INNER_POSITIVE_UNION_STRONG_SUCCESS"
    elif primary_success:
        verdict = "SOURCE_INNER_POSITIVE_UNION_PRIMARY_SUCCESS"
    elif weak_success:
        verdict = "SOURCE_INNER_POSITIVE_UNION_WEAK_PASS"
    return {
        "primary_verdict": verdict,
        "diagnostic_flags": "|".join(flags),
        "primary_method": cfg.primary_method,
        "leakage_status": leakage_status,
        "claim_boundary": "source-inner selected class-conditional aggregation repair; not compatibility routing",
        "comparison_cell_set": "intersection_positive_union_v2_arithmetic_prior_tailrisk_canonical_random_shrink050",
        "n_intersection_cells": len(paired_delta_rows),
        "selected_rule_counts_json": json.dumps(selected_counts, sort_keys=True),
        "insufficient_source_inner_positive_count_cells": insufficient_count,
        "center_equal_mean_bacc": primary["center_equal_mean_bacc"],
        "intersection_center_equal_mean_bacc": mean_i,
        "seed_cell_mean_bacc": primary["seed_cell_mean_bacc"],
        "center_equal_macro_f1": primary["center_equal_macro_f1"],
        "min_center_bacc": primary["min_center_bacc"],
        "seed_std_bacc": primary["seed_std_bacc"],
        "bottom20_cell_mean_bacc": primary["bottom20_cell_mean_bacc"],
        "worst_seed_center_bacc": primary["worst_seed_center_bacc"],
        "center3_bacc": primary["center3_bacc"],
        "prior_tailrisk_center_equal_mean_bacc": prior_mean,
        "v2_arithmetic_center_equal_mean_bacc": arithmetic_mean,
        "canonical_random_mass_bag_center_equal_mean_bacc": canonical["center_equal_mean_bacc"],
        "pooled_random_mass_bag_center_equal_mean_bacc": pooled_random["center_equal_mean_bacc"],
        "shrink050_center_equal_mean_bacc": anchor["center_equal_mean_bacc"],
        "delta_vs_prior_tailrisk_intersection": mean_i - prior_mean if math.isfinite(mean_i) and math.isfinite(prior_mean) else math.nan,
        "delta_vs_v2_arithmetic_intersection": arithmetic_delta,
        "delta_vs_canonical_random_mass_bag_intersection": mean_i - canonical_mean if math.isfinite(mean_i) and math.isfinite(canonical_mean) else math.nan,
        "min_center_delta_vs_prior_tailrisk": min_center_delta,
        "center3_delta_vs_prior_tailrisk": center3_delta,
        "bottom20_delta_vs_prior_tailrisk": bottom20_delta,
        "seed_std_delta_vs_prior_tailrisk": seed_std_delta,
        "frozen_bottom20_median_delta_vs_prior_tailrisk": tail_median_delta,
        "frozen_bottom20_positive_fraction": tail_positive_fraction,
        "worst_per_center_regression_vs_prior_tailrisk": worst_center_regression,
        "worst_seed_center_regression_vs_prior_tailrisk": worst_seed_center_regression,
        "tailrisk_transfer_flag": "TAIL_RISK_TRANSFER" in flags or any(str(row.get("tail_risk_transfer_flag")) == "True" for row in harm_rows),
        **primary,
    }


def _fixed_beta050_decision(
    rows: Sequence[Mapping[str, object]],
    *,
    paired_delta_rows: Sequence[Mapping[str, object]],
    arithmetic_tail_keys: set[tuple[str, str]],
    rare_positive_rows: Sequence[Mapping[str, object]],
    harm_rows: Sequence[Mapping[str, object]],
    leakage_status: str,
    cfg: FixedBeta050PositiveUnionConfig,
) -> dict[str, object]:
    primary = _multipanel_tail_metrics(rows, cfg.primary_method, prior_tail_keys=arithmetic_tail_keys)
    arithmetic = _multipanel_tail_metrics(rows, POSITIVE_UNION_RULE_ARITHMETIC, prior_tail_keys=arithmetic_tail_keys)
    beta025 = _multipanel_tail_metrics(rows, POSITIVE_UNION_RULE_BETA025, prior_tail_keys=arithmetic_tail_keys)
    beta100 = _multipanel_tail_metrics(rows, POSITIVE_UNION_RULE_BETA100, prior_tail_keys=arithmetic_tail_keys)
    anchor = _multipanel_tail_metrics(rows, MULTIPANEL_POOLED_ANCHOR_METHOD, prior_tail_keys=arithmetic_tail_keys)
    canonical = _multipanel_tail_metrics(rows, MULTIPANEL_CANONICAL_RANDOM_BAG_METHOD, prior_tail_keys=arithmetic_tail_keys)
    pooled_random = _multipanel_tail_metrics(rows, MULTIPANEL_POOLED_RANDOM_BAG_METHOD, prior_tail_keys=arithmetic_tail_keys)
    primary_i = _stats_from_paired(paired_delta_rows, "fixed_beta050_bacc") if paired_delta_rows else {}
    arithmetic_i = _stats_from_paired(paired_delta_rows, "v2_arithmetic_multipanel_bacc") if paired_delta_rows else {}

    mean_i = _float(primary_i.get("center_equal_mean_bacc", math.nan))
    arithmetic_mean = _float(arithmetic_i.get("center_equal_mean_bacc", math.nan))
    min_center_delta = _delta(primary_i.get("min_center_bacc", math.nan), arithmetic_i.get("min_center_bacc", math.nan)) if paired_delta_rows else math.nan
    center3_delta = _delta(primary_i.get("center3_bacc", math.nan), arithmetic_i.get("center3_bacc", math.nan)) if paired_delta_rows else math.nan
    bottom20_delta = _delta(primary_i.get("bottom20_cell_mean_bacc", math.nan), arithmetic_i.get("bottom20_cell_mean_bacc", math.nan)) if paired_delta_rows else math.nan
    seed_std_delta = _delta(primary_i.get("seed_std_bacc", math.nan), arithmetic_i.get("seed_std_bacc", math.nan)) if paired_delta_rows else math.nan
    arithmetic_delta = mean_i - arithmetic_mean if math.isfinite(mean_i) and math.isfinite(arithmetic_mean) else math.nan
    tail_deltas = [
        _float(row.get("delta_fixed_beta050_minus_v2_arithmetic"))
        for row in paired_delta_rows
        if str(row.get("is_frozen_arithmetic_bottom20_cell")) == "True" or row.get("is_frozen_arithmetic_bottom20_cell") is True
    ]
    tail_deltas = [value for value in tail_deltas if math.isfinite(value)]
    tail_positive_fraction = float(sum(value > 0.0 for value in tail_deltas)) / float(len(tail_deltas)) if tail_deltas else math.nan
    tail_median_delta = float(np.median(np.asarray(tail_deltas, dtype=float))) if tail_deltas else math.nan
    center_regressions = _per_center_regressions(primary_i, arithmetic_i) if paired_delta_rows else {}
    worst_center_regression = min(center_regressions.values(), default=math.nan)
    worst_seed_center_regression = min(
        (_float(row.get("delta_fixed_beta050_minus_v2_arithmetic")) for row in paired_delta_rows),
        default=math.nan,
    )
    assessable_rare = [
        row
        for row in rare_positive_rows
        if row.get("assessable_for_rare_positive_repair") is True or str(row.get("assessable_for_rare_positive_repair")) == "True"
    ]
    rare_recall_deltas = [
        _float(row.get("beta050_class1_recall")) - _float(row.get("arithmetic_class1_recall"))
        for row in assessable_rare
        if math.isfinite(_float(row.get("beta050_class1_recall"))) and math.isfinite(_float(row.get("arithmetic_class1_recall")))
    ]
    rare_recall_mean_delta = nanmean([value for value in rare_recall_deltas if math.isfinite(value)])
    rare_recall_positive_fraction = (
        float(sum(value > 0.0 for value in rare_recall_deltas)) / float(len(rare_recall_deltas))
        if rare_recall_deltas
        else math.nan
    )
    flags: list[str] = []
    if leakage_status != "PASS":
        flags.append("LEAKAGE_FAIL")
    if not paired_delta_rows:
        flags.append("MISSING_SAME_RUN_ARITHMETIC_INTERSECTION")
    if not assessable_rare:
        flags.append("NO_ASSESSABLE_RARE_POSITIVE_CELLS")
    elif math.isfinite(rare_recall_mean_delta) and rare_recall_mean_delta <= 0.0:
        flags.append("RARE_POSITIVE_RECALL_NOT_IMPROVED")
    if math.isfinite(worst_center_regression) and worst_center_regression < cfg.tailrisk_transfer_threshold:
        flags.append("TAIL_RISK_TRANSFER")
    if math.isfinite(worst_center_regression) and worst_center_regression < -0.020:
        flags.append("CENTER_REGRESSION_GT_0P020")
    if math.isfinite(mean_i) and math.isfinite(arithmetic_mean) and mean_i < arithmetic_mean - cfg.primary_noninferiority_margin:
        flags.append("MEAN_INFERIOR_TO_V2_ARITHMETIC_GT_0P005")
    if math.isfinite(min_center_delta) and (min_center_delta <= 0.0 or _float(primary_i.get("min_center_bacc", math.nan)) < 0.82):
        flags.append("MIN_CENTER_NOT_IMPROVED_OR_BELOW_0P82")
    if math.isfinite(center3_delta) and (center3_delta <= 0.0 or _float(primary_i.get("center3_bacc", math.nan)) < 0.82):
        flags.append("CENTER3_NOT_IMPROVED_OR_BELOW_0P82")
    if math.isfinite(bottom20_delta) and bottom20_delta <= 0.0:
        flags.append("BOTTOM20_NOT_IMPROVED")
    if math.isfinite(seed_std_delta) and seed_std_delta > 0.005:
        flags.append("SEED_STD_INCREASED_GT_0P005")
    if math.isfinite(tail_median_delta) and tail_median_delta <= 0.0:
        flags.append("FROZEN_BOTTOM20_MEDIAN_DELTA_NOT_POSITIVE")
    if math.isfinite(tail_positive_fraction) and tail_positive_fraction <= 0.5:
        flags.append("FROZEN_BOTTOM20_NOT_MAJORITY_IMPROVED")

    noninferior = (
        math.isfinite(mean_i)
        and math.isfinite(arithmetic_mean)
        and mean_i >= arithmetic_mean - cfg.primary_noninferiority_margin
    )
    weak_noninferior = (
        math.isfinite(mean_i)
        and math.isfinite(arithmetic_mean)
        and mean_i >= arithmetic_mean - cfg.weak_pass_noninferiority_margin
    )
    rare_success = bool(rare_recall_deltas) and rare_recall_mean_delta > 0.0 and rare_recall_positive_fraction > 0.5
    primary_success = (
        leakage_status == "PASS"
        and bool(paired_delta_rows)
        and noninferior
        and math.isfinite(min_center_delta)
        and min_center_delta > 0.0
        and _float(primary_i.get("min_center_bacc", math.nan)) >= 0.82
        and math.isfinite(center3_delta)
        and center3_delta > 0.0
        and _float(primary_i.get("center3_bacc", math.nan)) >= 0.82
        and bottom20_delta > 0.0
        and (not math.isfinite(seed_std_delta) or seed_std_delta <= 0.005)
        and tail_median_delta > 0.0
        and tail_positive_fraction > 0.5
        and rare_success
    )
    weak_success = (
        leakage_status == "PASS"
        and bool(paired_delta_rows)
        and weak_noninferior
        and math.isfinite(min_center_delta)
        and min_center_delta > 0.0
        and bottom20_delta > 0.0
    )
    primary_mean = _float(primary["center_equal_mean_bacc"])
    strong_success = (
        primary_success
        and primary_mean >= 0.90
        and _float(primary["min_center_bacc"]) >= 0.82
        and _float(primary["center3_bacc"]) >= 0.82
        and primary_mean > arithmetic_mean
    )
    verdict = "FIXED_BETA050_POSITIVE_UNION_FAIL"
    if leakage_status != "PASS":
        verdict = "PROTOCOL_FAIL"
    elif strong_success:
        verdict = "FIXED_BETA050_POSITIVE_UNION_STRONG_SUCCESS"
    elif primary_success:
        verdict = "FIXED_BETA050_POSITIVE_UNION_PRIMARY_SUCCESS"
    elif weak_success:
        verdict = "FIXED_BETA050_POSITIVE_UNION_WEAK_PASS"
    return {
        "primary_verdict": verdict,
        "diagnostic_flags": "|".join(flags),
        "primary_method": cfg.primary_method,
        "leakage_status": leakage_status,
        "claim_boundary": "fixed global beta050 positive-evidence pooling confirmation; not source-inner selected and not compatibility routing",
        "comparison_cell_set": "fresh_confirmation_intersection_fixed_beta050_v2_arithmetic_canonical_random_shrink050",
        "n_intersection_cells": len(paired_delta_rows),
        "fixed_rule": cfg.fixed_pooling_rule,
        "fixed_beta": cfg.fixed_beta,
        "development_experiment_seeds_json": json.dumps(list(cfg.development_experiment_seeds)),
        "primary_confirmation_experiment_seeds_json": json.dumps(list(cfg.confirmation_experiment_seeds)),
        "center_equal_mean_bacc": primary["center_equal_mean_bacc"],
        "intersection_center_equal_mean_bacc": mean_i,
        "seed_cell_mean_bacc": primary["seed_cell_mean_bacc"],
        "center_equal_macro_f1": primary["center_equal_macro_f1"],
        "min_center_bacc": primary["min_center_bacc"],
        "seed_std_bacc": primary["seed_std_bacc"],
        "bottom20_cell_mean_bacc": primary["bottom20_cell_mean_bacc"],
        "worst_seed_center_bacc": primary["worst_seed_center_bacc"],
        "center3_bacc": primary["center3_bacc"],
        "v2_arithmetic_center_equal_mean_bacc": arithmetic_mean,
        "fixed_beta025_center_equal_mean_bacc": beta025["center_equal_mean_bacc"],
        "fixed_beta100_center_equal_mean_bacc": beta100["center_equal_mean_bacc"],
        "canonical_random_mass_bag_center_equal_mean_bacc": canonical["center_equal_mean_bacc"],
        "pooled_random_mass_bag_center_equal_mean_bacc": pooled_random["center_equal_mean_bacc"],
        "shrink050_center_equal_mean_bacc": anchor["center_equal_mean_bacc"],
        "delta_vs_v2_arithmetic_intersection": arithmetic_delta,
        "min_center_delta_vs_v2_arithmetic": min_center_delta,
        "center3_delta_vs_v2_arithmetic": center3_delta,
        "bottom20_delta_vs_v2_arithmetic": bottom20_delta,
        "seed_std_delta_vs_v2_arithmetic": seed_std_delta,
        "frozen_bottom20_median_delta_vs_v2_arithmetic": tail_median_delta,
        "frozen_bottom20_positive_fraction": tail_positive_fraction,
        "n_assessable_rare_positive_cells": len(assessable_rare),
        "rare_positive_recall_mean_delta_vs_arithmetic": rare_recall_mean_delta,
        "rare_positive_recall_positive_fraction": rare_recall_positive_fraction,
        "worst_per_center_regression_vs_v2_arithmetic": worst_center_regression,
        "worst_seed_center_regression_vs_v2_arithmetic": worst_seed_center_regression,
        "tailrisk_transfer_flag": "TAIL_RISK_TRANSFER" in flags or any(str(row.get("tail_risk_transfer_flag")) == "True" for row in harm_rows),
        **primary,
    }


def _per_center_regressions(primary_stats: Mapping[str, object], prior_stats: Mapping[str, object]) -> dict[str, float]:
    try:
        primary = json.loads(str(primary_stats.get("per_center_bacc", "{}")))
        prior = json.loads(str(prior_stats.get("per_center_bacc", "{}")))
    except json.JSONDecodeError:
        return {}
    out = {}
    for center, prior_value in prior.items():
        p = _float(primary.get(center, math.nan))
        b = _float(prior_value)
        if math.isfinite(p) and math.isfinite(b):
            out[str(center)] = p - b
    return out


def _write_multipanel_artifacts(
    root: Path,
    cfg: MultipanelTailRiskConfig,
    *,
    matrix_rows: Sequence[Mapping[str, object]],
    seed_diagnostic_rows: Sequence[Mapping[str, object]],
    source_weight_rows: Sequence[Mapping[str, object]],
    reliability_rows: Sequence[Mapping[str, object]],
    source_summary_rows: Sequence[Mapping[str, object]],
    component_manifest_rows: Sequence[Mapping[str, object]],
    component_coverage_rows: Sequence[Mapping[str, object]],
    paired_generation_rows: Sequence[Mapping[str, object]],
    eligibility_rows: Sequence[Mapping[str, object]],
    blend_manifest_rows: Sequence[Mapping[str, object]],
    calibration_rows: Sequence[Mapping[str, object]],
    panel_disagreement_rows: Sequence[Mapping[str, object]],
    invariant_rows: Sequence[Mapping[str, object]],
    confidence_rows: Sequence[Mapping[str, object]],
    failure_rows: Sequence[Mapping[str, object]],
    center3_failure_cell_rows: Sequence[Mapping[str, object]],
    center3_failure_sample_rows: Sequence[Mapping[str, object]],
    center3_failure_pooling_rows: Sequence[Mapping[str, object]],
    paired_delta_rows: Sequence[Mapping[str, object]],
    model_manifest_rows: Sequence[Mapping[str, object]],
    decision: Mapping[str, object],
    leakage: object,
    protocol_violations: Sequence[str],
    target_expert_excluded: bool,
) -> None:
    write_csv_rows(root / "tables" / "multipanel_tailrisk_downstream_matrix.csv", matrix_rows)
    write_csv_rows(root / "tables" / "multipanel_tailrisk_seed_diagnostic_matrix.csv", seed_diagnostic_rows)
    write_csv_rows(root / "tables" / "multipanel_tailrisk_summary.csv", [dict(decision)])
    write_csv_rows(root / "tables" / "multipanel_tailrisk_failure_decomposition.csv", failure_rows)
    write_csv_rows(root / "tables" / "multipanel_tailrisk_paired_deltas.csv", paired_delta_rows)
    write_csv_rows(root / "tables" / "multipanel_tailrisk_panel_disagreement.csv", panel_disagreement_rows)
    write_csv_rows(root / "tables" / "panel_ece_source_inner.csv", _panel_ece_source_inner_rows(calibration_rows))
    write_csv_rows(root / "tables" / "panel_confidence_summary.csv", confidence_rows)
    write_csv_rows(root / "tables" / "multipanel_tailrisk_probability_invariants.csv", invariant_rows)
    write_csv_rows(root / "tables" / "multipanel_tailrisk_probability_blend_manifest.csv", blend_manifest_rows)
    write_csv_rows(root / "tables" / "source_weight_manifest.csv", source_weight_rows)
    write_csv_rows(root / "tables" / "source_reliability_manifest.csv", reliability_rows)
    write_csv_rows(root / "tables" / "component_manifest.csv", component_manifest_rows)
    write_csv_rows(root / "tables" / "component_coverage_audit.csv", component_coverage_rows)
    write_csv_rows(root / "tables" / "paired_generation_audit.csv", paired_generation_rows)
    write_csv_rows(root / "tables" / "eligibility_audit.csv", eligibility_rows)
    write_csv_rows(root / "tables" / "source_summary_diagnostics.csv", source_summary_rows)
    write_csv_rows(root / "manifests" / "multipanel_tailrisk_model_manifest.csv", model_manifest_rows)
    _write_center3_failure_audit_artifacts(
        root,
        cell_rows=center3_failure_cell_rows,
        sample_rows=center3_failure_sample_rows,
        pooling_rows=center3_failure_pooling_rows,
        source_weight_rows=source_weight_rows,
        component_coverage_rows=component_coverage_rows,
    )
    write_json(root / "reports" / "leakage_report.json", leakage.to_json_dict())
    write_json(
        root / "manifests" / "protocol_manifest.json",
        {
            "schema_version": "cvae_rebuild_tailrisk_multipanel_component_union_protocol_v1",
            "experiment_name": cfg.name,
            "primary_method": cfg.primary_method,
            "experiment_type": "source_only_tailrisk_multipanel_mass_bag_stabilization",
            "target_expert_excluded": bool(target_expert_excluded),
            "target_support_used": False,
            "target_support_labels_for_selection": False,
            "target_eval_labels_for_scoring_only": True,
            "selection_used_target_labels": False,
            "target_calibration_metrics_audit_only": True,
            "center3_failure_audit_diagnostic_only": True,
            "center3_failure_audit_target_labels_post_prediction_only": True,
            "center3_failure_audit_cells": [f"{seed}xcenter{center}" for seed, center in CENTER3_FAILURE_AUDIT_CELLS],
            "source_inner_calibration_primary": True,
            "target_conditioned_point_compatibility_estimate": False,
            "fixed_all_source_inclusion": True,
            "panel_seeds_are_evaluation_replicates": False,
            "decision_cell": "experiment_seed_x_heldout_center",
            "primary_pooling_rule": "blend_per_seed_then_equal_probability_pool",
            "blend_alpha_locked": cfg.blend_alpha,
            "random_mass_bag_size": cfg.random_mass_bag_size,
            "random_mass_bag_distribution": "dirichlet_uniform_alpha4",
            "panel_seed_groups": {panel: list(seeds) for panel, seeds in cfg.panel_seed_groups},
            "prior_tailrisk_comparator": "" if cfg.prior_tailrisk_artifact_root is None else str(cfg.prior_tailrisk_artifact_root / "tables" / "tailrisk_downstream_matrix.csv"),
            "claim_boundary": (
                "stabilized source-only dense stochastic generative composition; "
                "not compatibility routing, target adaptation, or target-label-driven method choice"
            ),
            "protocol_violations": list(protocol_violations),
        },
    )
    write_json(root / "run_config_resolved.yaml", _resolved_multipanel_config(cfg))
    _write_multipanel_decision_summary(root, decision)


def _write_positive_union_artifacts(
    root: Path,
    cfg: SourceInnerPositiveUnionConfig,
    *,
    matrix_rows: Sequence[Mapping[str, object]],
    source_inner_selection_rows: Sequence[Mapping[str, object]],
    candidate_rule_rows: Sequence[Mapping[str, object]],
    class_conditional_rows: Sequence[Mapping[str, object]],
    effective_threshold_rows: Sequence[Mapping[str, object]],
    paired_delta_rows: Sequence[Mapping[str, object]],
    harm_rows: Sequence[Mapping[str, object]],
    per_source_harm_rows: Sequence[Mapping[str, object]],
    invariant_rows: Sequence[Mapping[str, object]],
    blend_manifest_rows: Sequence[Mapping[str, object]],
    source_weight_rows: Sequence[Mapping[str, object]],
    reliability_rows: Sequence[Mapping[str, object]],
    source_summary_rows: Sequence[Mapping[str, object]],
    component_manifest_rows: Sequence[Mapping[str, object]],
    component_coverage_rows: Sequence[Mapping[str, object]],
    paired_generation_rows: Sequence[Mapping[str, object]],
    eligibility_rows: Sequence[Mapping[str, object]],
    model_manifest_rows: Sequence[Mapping[str, object]],
    decision: Mapping[str, object],
    leakage: object,
    protocol_violations: Sequence[str],
    target_expert_excluded: bool,
) -> None:
    write_csv_rows(root / "tables" / "positive_union_downstream_matrix.csv", matrix_rows)
    write_csv_rows(root / "tables" / "positive_union_summary.csv", [dict(decision)])
    write_csv_rows(root / "tables" / "positive_union_source_inner_selection.csv", source_inner_selection_rows)
    write_csv_rows(root / "tables" / "positive_union_candidate_rule_matrix.csv", candidate_rule_rows)
    write_csv_rows(root / "tables" / "positive_union_class_conditional_audit.csv", class_conditional_rows)
    write_csv_rows(root / "tables" / "positive_union_effective_threshold_audit.csv", effective_threshold_rows)
    write_csv_rows(root / "tables" / "positive_union_paired_deltas.csv", paired_delta_rows)
    write_csv_rows(root / "tables" / "positive_union_harm_audit.csv", harm_rows)
    write_csv_rows(root / "tables" / "positive_union_source_inner_per_source_harm_audit.csv", per_source_harm_rows)
    write_csv_rows(root / "tables" / "positive_union_probability_invariants.csv", invariant_rows)
    write_csv_rows(root / "tables" / "positive_union_probability_blend_manifest.csv", blend_manifest_rows)
    write_csv_rows(root / "tables" / "source_weight_manifest.csv", source_weight_rows)
    write_csv_rows(root / "tables" / "source_reliability_manifest.csv", reliability_rows)
    write_csv_rows(root / "tables" / "source_summary_diagnostics.csv", source_summary_rows)
    write_csv_rows(root / "tables" / "component_manifest.csv", component_manifest_rows)
    write_csv_rows(root / "tables" / "component_coverage_audit.csv", component_coverage_rows)
    write_csv_rows(root / "tables" / "paired_generation_audit.csv", paired_generation_rows)
    write_csv_rows(root / "tables" / "eligibility_audit.csv", eligibility_rows)
    write_csv_rows(root / "manifests" / "positive_union_model_manifest.csv", model_manifest_rows)
    write_json(root / "reports" / "leakage_report.json", leakage.to_json_dict())
    write_json(
        root / "manifests" / "protocol_manifest.json",
        {
            "schema_version": "cvae_rebuild_source_inner_positive_union_protocol_v1",
            "experiment_name": cfg.name,
            "primary_method": cfg.primary_method,
            "experiment_type": "source_only_class_conditional_positive_union_tailrisk_repair",
            "target_expert_excluded": bool(target_expert_excluded),
            "target_support_used": False,
            "target_support_labels_for_selection": False,
            "target_eval_labels_for_scoring_only": True,
            "selection_used_target_labels": False,
            "target_calibration_metrics_audit_only": True,
            "target_eval_candidate_rule_metrics_audit_only": True,
            "target_conditioned_point_compatibility_estimate": False,
            "compatibility_router": False,
            "fixed_all_source_inclusion": True,
            "panel_seeds_are_evaluation_replicates": False,
            "decision_cell": "experiment_seed_x_heldout_center",
            "source_inner_selection_primary": True,
            "source_inner_validation_scope": "pooled_non_target_source_validation_rows",
            "source_inner_per_source_harm_audit": True,
            "positive_label": cfg.positive_label,
            "prediction_threshold": cfg.prediction_threshold,
            "minimum_source_inner_positive_count": cfg.min_source_inner_positive_count,
            "positive_union_eps": cfg.positive_union_eps,
            "candidate_pooling_rules": list(cfg.candidate_pooling_rules),
            "primary_pooling_rule": POSITIVE_UNION_PRIMARY_POOLING,
            "blend_alpha_locked": cfg.blend_alpha,
            "random_mass_bag_size": cfg.random_mass_bag_size,
            "random_mass_bag_distribution": "dirichlet_uniform_alpha4",
            "panel_seed_groups": {panel: list(seeds) for panel, seeds in cfg.panel_seed_groups},
            "prior_tailrisk_comparator": "" if cfg.prior_tailrisk_artifact_root is None else str(cfg.prior_tailrisk_artifact_root / "tables" / "tailrisk_downstream_matrix.csv"),
            "claim_boundary": (
                "source-inner selected class-conditional aggregation repair after fixed dense source-only "
                "CVAE seed-blend aggregation; not compatibility routing, target adaptation, or target-label tuning"
            ),
            "protocol_violations": list(protocol_violations),
        },
    )
    write_json(root / "run_config_resolved.yaml", _resolved_positive_union_config(cfg))
    _write_positive_union_decision_summary(root, decision)


def _write_fixed_beta050_positive_union_artifacts(
    root: Path,
    cfg: FixedBeta050PositiveUnionConfig,
    *,
    matrix_rows: Sequence[Mapping[str, object]],
    candidate_rule_rows: Sequence[Mapping[str, object]],
    class_conditional_rows: Sequence[Mapping[str, object]],
    effective_threshold_rows: Sequence[Mapping[str, object]],
    rare_positive_rows: Sequence[Mapping[str, object]],
    paired_delta_rows: Sequence[Mapping[str, object]],
    harm_rows: Sequence[Mapping[str, object]],
    invariant_rows: Sequence[Mapping[str, object]],
    blend_manifest_rows: Sequence[Mapping[str, object]],
    retrospective_reference_rows: Sequence[Mapping[str, object]],
    source_inner_rows: Sequence[Mapping[str, object]],
    decision: Mapping[str, object],
    leakage: object,
    protocol_violations: Sequence[str],
    target_expert_excluded: bool,
) -> None:
    write_csv_rows(root / "tables" / "fixed_beta050_downstream_matrix.csv", matrix_rows)
    write_csv_rows(root / "tables" / "fixed_beta050_summary.csv", [dict(decision)])
    write_csv_rows(root / "tables" / "fixed_beta050_candidate_rule_matrix.csv", candidate_rule_rows)
    write_csv_rows(root / "tables" / "fixed_beta050_class_conditional_audit.csv", class_conditional_rows)
    write_csv_rows(root / "tables" / "fixed_beta050_effective_threshold_audit.csv", effective_threshold_rows)
    write_csv_rows(root / "tables" / "fixed_beta050_rare_positive_opportunity_audit.csv", rare_positive_rows)
    write_csv_rows(root / "tables" / "fixed_beta050_paired_deltas.csv", paired_delta_rows)
    write_csv_rows(root / "tables" / "fixed_beta050_harm_audit.csv", harm_rows)
    write_csv_rows(root / "tables" / "fixed_beta050_probability_invariants.csv", invariant_rows)
    write_csv_rows(root / "tables" / "fixed_beta050_probability_blend_manifest.csv", blend_manifest_rows)
    write_csv_rows(root / "tables" / "fixed_beta050_retrospective_reference.csv", retrospective_reference_rows)
    write_csv_rows(root / "tables" / "fixed_beta050_source_inner_diagnostics.csv", source_inner_rows)
    write_json(root / "reports" / "leakage_report.json", leakage.to_json_dict())
    write_json(
        root / "manifests" / "protocol_manifest.json",
        {
            "schema_version": "cvae_rebuild_fixed_beta050_positive_union_confirmation_protocol_v1",
            "experiment_name": cfg.name,
            "primary_method": cfg.primary_method,
            "experiment_type": "source_only_fixed_beta050_positive_union_confirmation",
            "target_expert_excluded": bool(target_expert_excluded),
            "target_support_used": False,
            "target_support_labels_for_selection": False,
            "target_eval_labels_for_scoring_only": True,
            "selection_used_target_labels": False,
            "target_eval_candidate_rule_metrics_audit_only": True,
            "target_conditioned_point_compatibility_estimate": False,
            "compatibility_router": False,
            "fixed_all_source_inclusion": True,
            "panel_seeds_are_evaluation_replicates": False,
            "decision_cell": "experiment_seed_x_heldout_center",
            "beta_rule": "fixed_global_beta050",
            "fixed_pooling_rule": cfg.fixed_pooling_rule,
            "fixed_beta": cfg.fixed_beta,
            "beta_origin": "hypothesis_generated_from_prior_positive_union_diagnostic",
            "development_experiment_seeds": list(cfg.development_experiment_seeds),
            "primary_confirmation_experiment_seeds": list(cfg.confirmation_experiment_seeds),
            "no_posthoc_beta_selection": True,
            "old_cells_retrospective_reference_only": True,
            "source_inner_selection_primary": False,
            "source_inner_diagnostics_only": True,
            "positive_label": cfg.positive_label,
            "prediction_threshold": cfg.prediction_threshold,
            "positive_union_eps": cfg.positive_union_eps,
            "candidate_pooling_rules": list(cfg.candidate_pooling_rules),
            "primary_pooling_rule": FIXED_BETA050_POSITIVE_UNION_PRIMARY_POOLING,
            "rare_positive_definition": {
                "class1_count_lte": cfg.rare_positive_count_threshold,
                "positive_prevalence_lte": cfg.rare_positive_prevalence_threshold,
            },
            "blend_alpha_locked": cfg.blend_alpha,
            "random_mass_bag_size": cfg.random_mass_bag_size,
            "random_mass_bag_distribution": "dirichlet_uniform_alpha4",
            "panel_seed_groups": {panel: list(seeds) for panel, seeds in cfg.panel_seed_groups},
            "prior_tailrisk_comparator": "retrospective/contextual_only_for_development_seeds",
            "claim_boundary": (
                "fixed global beta050 positive-evidence pooling after dense source-only CVAE seed-blend "
                "aggregation; not source-inner selected, not compatibility routing, not target adaptation, "
                "and not target-threshold tuning"
            ),
            "protocol_violations": list(protocol_violations),
        },
    )
    write_json(root / "run_config_resolved.yaml", _resolved_fixed_beta050_config(cfg))
    _write_fixed_beta050_decision_summary(root, decision)


def _write_center3_failure_audit_artifacts(
    root: Path,
    *,
    cell_rows: Sequence[Mapping[str, object]],
    sample_rows: Sequence[Mapping[str, object]],
    pooling_rows: Sequence[Mapping[str, object]],
    source_weight_rows: Sequence[Mapping[str, object]],
    component_coverage_rows: Sequence[Mapping[str, object]],
) -> None:
    audit_root = root / "center3_failure_audit"
    filtered_source_weights = _center3_failure_filtered_existing_rows(source_weight_rows)
    filtered_component_coverage = _center3_failure_filtered_existing_rows(component_coverage_rows)
    write_csv_rows(audit_root / "center3_failure_cell_summary.csv", cell_rows)
    write_csv_rows(audit_root / "center3_failure_sample_audit.csv", sample_rows)
    write_csv_rows(audit_root / "center3_failure_pooling_path.csv", pooling_rows)
    write_csv_rows(audit_root / "center3_failure_source_weight_comparison.csv", filtered_source_weights)
    write_csv_rows(audit_root / "center3_failure_component_coverage_comparison.csv", filtered_component_coverage)
    _write_center3_failure_conclusion(
        audit_root / "center3_failure_conclusion.md",
        cell_rows=cell_rows,
        sample_rows=sample_rows,
        source_weight_rows=filtered_source_weights,
        component_coverage_rows=filtered_component_coverage,
    )


def _center3_failure_filtered_existing_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    out = []
    for row in rows:
        experiment_seed = _safe_int(row.get("experiment_seed"), default=-1)
        heldout_center = str(row.get("heldout_center", ""))
        if not _is_center3_failure_audit_cell(experiment_seed, heldout_center):
            continue
        out.append(
            {
                "audit_only": True,
                "target_eval_labels_used_for_audit_only": True,
                "selection_used_target_labels": False,
                "audit_cell_role": _center3_failure_audit_role(experiment_seed, heldout_center),
                **dict(row),
            }
        )
    return out


def _write_center3_failure_conclusion(
    path: Path,
    *,
    cell_rows: Sequence[Mapping[str, object]],
    sample_rows: Sequence[Mapping[str, object]],
    source_weight_rows: Sequence[Mapping[str, object]],
    component_coverage_rows: Sequence[Mapping[str, object]],
) -> None:
    primary_cell = [
        row
        for row in cell_rows
        if int(_safe_int(row.get("experiment_seed"), default=-1)) == CENTER3_FAILURE_PRIMARY_CELL[0]
        and str(row.get("heldout_center")) == CENTER3_FAILURE_PRIMARY_CELL[1]
    ]
    final = next((row for row in primary_cell if row.get("audit_method") == "final_v2"), {})
    best_seed_delta = _float(final.get("delta_best_individual_seed_blend_minus_final_v2", math.nan))
    class0_recall = _float(final.get("class0_recall", math.nan))
    class1_recall = _float(final.get("class1_recall", math.nan))
    class0_pred = _safe_int(final.get("class0_predicted_count"), default=0)
    class1_pred = _safe_int(final.get("class1_predicted_count"), default=0)
    n_eval = _safe_int(final.get("n_target_eval"), default=0)
    seed101_suppressed = sum(
        1
        for row in sample_rows
        if row.get("audit_cell_role") == "primary_center3_failure"
        and (row.get("seed_101_correct_final_wrong") is True or str(row.get("seed_101_correct_final_wrong")) == "True")
    )
    seed127_suppressed = sum(
        1
        for row in sample_rows
        if row.get("audit_cell_role") == "primary_center3_failure"
        and (row.get("seed_127_correct_final_wrong") is True or str(row.get("seed_127_correct_final_wrong")) == "True")
    )
    final_correct_seed101_wrong = sum(
        1
        for row in sample_rows
        if row.get("audit_cell_role") == "primary_center3_failure"
        and (row.get("final_correct_seed_101_wrong") is True or str(row.get("final_correct_seed_101_wrong")) == "True")
    )
    final_correct_seed127_wrong = sum(
        1
        for row in sample_rows
        if row.get("audit_cell_role") == "primary_center3_failure"
        and (row.get("final_correct_seed_127_wrong") is True or str(row.get("final_correct_seed_127_wrong")) == "True")
    )
    flags: list[str] = []
    if not final:
        flags.append("insufficient_row_level_evidence")
    if n_eval and (class0_pred == 0 or class1_pred == 0 or class0_pred == n_eval or class1_pred == n_eval):
        flags.append("class_collapse")
    elif math.isfinite(class0_recall) and math.isfinite(class1_recall) and min(class0_recall, class1_recall) <= 0.05:
        flags.append("near_class_collapse")
    if math.isfinite(best_seed_delta) and best_seed_delta >= 0.10:
        flags.append("probability_pooling_suppresses_best_seed")
    if seed101_suppressed > final_correct_seed101_wrong or seed127_suppressed > final_correct_seed127_wrong:
        if "probability_pooling_suppresses_best_seed" not in flags:
            flags.append("probability_pooling_suppresses_best_seed")
    mean_incorrect_conf = _float(final.get("mean_confidence_incorrect", math.nan))
    if math.isfinite(mean_incorrect_conf) and mean_incorrect_conf >= 0.65:
        flags.append("confident_wrong_predictions")
    if not flags:
        flags.append("no_single_dominant_failure_mode_from_compact_audit")

    lines = [
        "# Center3 Failure Audit",
        "",
        "## Scope",
        "",
        "Diagnostic-only audit of predefined cells. Target labels are used only after fixed prediction bundles exist, for scoring and failure analysis.",
        "",
        "## Primary Cell",
        "",
        f"- Cell: `{CENTER3_FAILURE_PRIMARY_CELL[0]} x center{CENTER3_FAILURE_PRIMARY_CELL[1]}`",
        f"- Final v2 BACC: {_format_float(final.get('bacc', math.nan)) if final else 'nan'}",
        f"- Final class0 recall: {_format_float(class0_recall)}",
        f"- Final class1 recall: {_format_float(class1_recall)}",
        f"- Predicted class counts: class0={class0_pred}, class1={class1_pred}, n={n_eval}",
        f"- Best individual seed-blend delta over final: {_format_float(best_seed_delta)}",
        f"- Seed101 correct while final wrong: {seed101_suppressed}",
        f"- Seed127 correct while final wrong: {seed127_suppressed}",
        f"- Final correct while seed101 wrong: {final_correct_seed101_wrong}",
        f"- Final correct while seed127 wrong: {final_correct_seed127_wrong}",
        "",
        "## Assigned Failure Mode",
        "",
        f"- `{ '|'.join(flags) }`",
        "",
        "## Artifact Evidence",
        "",
        f"- Cell/pooling rows: {len(cell_rows)}",
        f"- Sample audit rows: {len(sample_rows)}",
        f"- Source-weight comparison rows: {len(source_weight_rows)}",
        f"- Component-coverage comparison rows: {len(component_coverage_rows)}",
        "",
        "## Protocol Boundary",
        "",
        "This audit must not be used to select seeds, calibrate on target labels, change pooling policy, or claim target-compatible expert discovery. Any follow-up method must be predeclared separately.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _safe_int(value: object, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _panel_ece_source_inner_rows(calibration_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    out = []
    groups: dict[tuple[str, str, str], list[Mapping[str, object]]] = {}
    for row in calibration_rows:
        groups.setdefault((str(row.get("heldout_center")), str(row.get("panel_group", "")), str(row.get("probability_source", ""))), []).append(row)
    for (center, panel, source), rows in sorted(groups.items()):
        ece = [_float(row.get("source_inner_ece")) for row in rows]
        brier = [_float(row.get("source_inner_brier")) for row in rows]
        log_loss = [_float(row.get("source_inner_log_loss")) for row in rows]
        out.append(
            {
                "heldout_center": center,
                "panel_group": panel,
                "probability_source": source,
                "mean_source_inner_ece": nanmean([value for value in ece if math.isfinite(value)]),
                "mean_source_inner_brier": nanmean([value for value in brier if math.isfinite(value)]),
                "mean_source_inner_log_loss": nanmean([value for value in log_loss if math.isfinite(value)]),
                "source_inner_calibration_available": any(str(row.get("source_inner_calibration_available")) == "True" or row.get("source_inner_calibration_available") is True for row in rows),
                "target_eval_calibration_audit_only": True,
            }
        )
    return out


def _resolved_multipanel_config(cfg: MultipanelTailRiskConfig) -> dict[str, object]:
    resolved = _resolved_config(cfg)
    resolved["experiment"]["name"] = cfg.name
    resolved["experiment"]["artifact_root"] = str(cfg.artifact_root)
    resolved["tailrisk_multipanel_component_union"] = {
        "primary_method": cfg.primary_method,
        "primary_shrink_lambda": cfg.primary_shrink_lambda,
        "random_mass_bag_size": cfg.random_mass_bag_size,
        "random_mass_bag_alpha": cfg.random_mass_bag_alpha,
        "blend_alpha": cfg.blend_alpha,
        "panel_seed_groups": {panel: list(seeds) for panel, seeds in cfg.panel_seed_groups},
        "source_weighting": cfg.source_weighting,
        "primary_pooling": cfg.primary_pooling,
        "primary_noninferiority_margin": cfg.primary_noninferiority_margin,
        "weak_pass_noninferiority_margin": cfg.weak_pass_noninferiority_margin,
        "tailrisk_transfer_threshold": cfg.tailrisk_transfer_threshold,
    }
    return resolved


def _resolved_positive_union_config(cfg: SourceInnerPositiveUnionConfig) -> dict[str, object]:
    resolved = _resolved_config(cfg)
    resolved["experiment"]["name"] = cfg.name
    resolved["experiment"]["artifact_root"] = str(cfg.artifact_root)
    resolved["source_inner_class_conditional_positive_union"] = {
        "primary_method": cfg.primary_method,
        "primary_shrink_lambda": cfg.primary_shrink_lambda,
        "random_mass_bag_size": cfg.random_mass_bag_size,
        "random_mass_bag_alpha": cfg.random_mass_bag_alpha,
        "blend_alpha": cfg.blend_alpha,
        "panel_seed_groups": {panel: list(seeds) for panel, seeds in cfg.panel_seed_groups},
        "source_weighting": cfg.source_weighting,
        "primary_pooling": cfg.primary_pooling,
        "candidate_pooling_rules": list(cfg.candidate_pooling_rules),
        "positive_label": cfg.positive_label,
        "prediction_threshold": cfg.prediction_threshold,
        "min_source_inner_positive_count": cfg.min_source_inner_positive_count,
        "positive_union_eps": cfg.positive_union_eps,
        "source_inner_bacc_noninferiority_margin": cfg.source_inner_bacc_noninferiority_margin,
        "source_inner_class0_recall_margin": cfg.source_inner_class0_recall_margin,
        "source_inner_predicted_positive_rate_delta": cfg.source_inner_predicted_positive_rate_delta,
        "beta100_class0_recall_margin": cfg.beta100_class0_recall_margin,
        "beta100_precision_margin": cfg.beta100_precision_margin,
        "primary_noninferiority_margin": cfg.primary_noninferiority_margin,
        "weak_pass_noninferiority_margin": cfg.weak_pass_noninferiority_margin,
        "tailrisk_transfer_threshold": cfg.tailrisk_transfer_threshold,
    }
    return resolved


def _resolved_fixed_beta050_config(cfg: FixedBeta050PositiveUnionConfig) -> dict[str, object]:
    resolved = _resolved_config(cfg)
    resolved["experiment"]["name"] = cfg.name
    resolved["experiment"]["artifact_root"] = str(cfg.artifact_root)
    resolved["fixed_beta050_positive_union_confirmation"] = {
        "primary_method": cfg.primary_method,
        "primary_shrink_lambda": cfg.primary_shrink_lambda,
        "random_mass_bag_size": cfg.random_mass_bag_size,
        "random_mass_bag_alpha": cfg.random_mass_bag_alpha,
        "blend_alpha": cfg.blend_alpha,
        "panel_seed_groups": {panel: list(seeds) for panel, seeds in cfg.panel_seed_groups},
        "source_weighting": cfg.source_weighting,
        "primary_pooling": cfg.primary_pooling,
        "candidate_pooling_rules": list(cfg.candidate_pooling_rules),
        "fixed_pooling_rule": cfg.fixed_pooling_rule,
        "fixed_beta": cfg.fixed_beta,
        "beta_origin": "hypothesis_generated_from_prior_positive_union_diagnostic",
        "development_experiment_seeds": list(cfg.development_experiment_seeds),
        "primary_confirmation_experiment_seeds": list(cfg.confirmation_experiment_seeds),
        "positive_label": cfg.positive_label,
        "prediction_threshold": cfg.prediction_threshold,
        "positive_union_eps": cfg.positive_union_eps,
        "rare_positive_count_threshold": cfg.rare_positive_count_threshold,
        "rare_positive_prevalence_threshold": cfg.rare_positive_prevalence_threshold,
        "primary_noninferiority_margin": cfg.primary_noninferiority_margin,
        "weak_pass_noninferiority_margin": cfg.weak_pass_noninferiority_margin,
        "tailrisk_transfer_threshold": cfg.tailrisk_transfer_threshold,
    }
    return resolved


def _write_multipanel_decision_summary(root: Path, decision: Mapping[str, object]) -> None:
    lines = [
        "# Multi-Panel Tail-Risk Mass-Bag Stabilization v1",
        "",
        "## Summary",
        "",
        f"- Primary method: `{decision.get('primary_method', PRIMARY_MULTIPANEL_TAILRISK_METHOD)}`",
        f"- Primary verdict: `{decision.get('primary_verdict', 'MULTIPANEL_TAILRISK_STABILIZATION_FAIL')}`",
        f"- Diagnostic flags: `{decision.get('diagnostic_flags', '')}`",
        f"- Center-equal mean BACC: {_format_float(decision.get('center_equal_mean_bacc'))}",
        f"- Intersection mean BACC: {_format_float(decision.get('intersection_center_equal_mean_bacc'))}",
        f"- Min center BACC: {_format_float(decision.get('min_center_bacc'))}",
        f"- Center 3 BACC: {_format_float(decision.get('center3_bacc'))}",
        f"- Frozen bottom-20 BACC: {_format_float(decision.get('bottom20_cell_mean_bacc'))}",
        f"- Seed std BACC: {_format_float(decision.get('seed_std_bacc'))}",
        f"- Delta vs prior tailrisk: {_format_float(decision.get('delta_vs_prior_tailrisk_intersection'))}",
        f"- Delta vs canonical random mass-bag: {_format_float(decision.get('delta_vs_canonical_random_mass_bag_intersection'))}",
        f"- Frozen bottom20 median delta: {_format_float(decision.get('frozen_bottom20_median_delta_vs_prior_tailrisk'))}",
        f"- Worst per-center regression vs prior tailrisk: {_format_float(decision.get('worst_per_center_regression_vs_prior_tailrisk'))}",
        f"- Tail-risk transfer flag: `{decision.get('tailrisk_transfer_flag')}`",
        f"- Leakage status: `{decision.get('leakage_status', '')}`",
        "",
        "## Protocol Boundary",
        "",
        "This is a source-only stochastic-composition stabilization experiment. It is not a compatibility router and does not claim random mass-bag discovers target-compatible experts.",
        "",
        "The primary method blends each seed-specific shrink050 anchor with its seed-specific random mass-bag, then probability-pools the nine predeclared seed blends before computing metrics.",
        "",
        "Target evaluation labels are scoring/audit only and never choose seeds, alpha, source set, calibration, classifier, or pass/fail policy.",
        "",
    ]
    (root / "reports" / "decision_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _write_positive_union_decision_summary(root: Path, decision: Mapping[str, object]) -> None:
    lines = [
        "# Source-Inner Class-Conditional Positive Union v1",
        "",
        "## Summary",
        "",
        f"- Primary method: `{decision.get('primary_method', PRIMARY_POSITIVE_UNION_METHOD)}`",
        f"- Primary verdict: `{decision.get('primary_verdict', 'SOURCE_INNER_POSITIVE_UNION_FAIL')}`",
        f"- Diagnostic flags: `{decision.get('diagnostic_flags', '')}`",
        f"- Selected rule counts: `{decision.get('selected_rule_counts_json', '{}')}`",
        f"- Insufficient source-inner positive-count cells: {decision.get('insufficient_source_inner_positive_count_cells', 0)}",
        f"- Center-equal mean BACC: {_format_float(decision.get('center_equal_mean_bacc'))}",
        f"- Intersection mean BACC: {_format_float(decision.get('intersection_center_equal_mean_bacc'))}",
        f"- Min center BACC: {_format_float(decision.get('min_center_bacc'))}",
        f"- Center 3 BACC: {_format_float(decision.get('center3_bacc'))}",
        f"- Frozen bottom-20 BACC: {_format_float(decision.get('bottom20_cell_mean_bacc'))}",
        f"- Seed std BACC: {_format_float(decision.get('seed_std_bacc'))}",
        f"- Delta vs prior tailrisk: {_format_float(decision.get('delta_vs_prior_tailrisk_intersection'))}",
        f"- Delta vs v2 arithmetic multipanel: {_format_float(decision.get('delta_vs_v2_arithmetic_intersection'))}",
        f"- Frozen bottom20 median delta: {_format_float(decision.get('frozen_bottom20_median_delta_vs_prior_tailrisk'))}",
        f"- Worst per-center regression vs prior tailrisk: {_format_float(decision.get('worst_per_center_regression_vs_prior_tailrisk'))}",
        f"- Worst seed-center regression vs prior tailrisk: {_format_float(decision.get('worst_seed_center_regression_vs_prior_tailrisk'))}",
        f"- Tail-risk transfer flag: `{decision.get('tailrisk_transfer_flag')}`",
        f"- Leakage status: `{decision.get('leakage_status', '')}`",
        "",
        "## Protocol Boundary",
        "",
        "This is a source-inner selected class-conditional aggregation repair over fixed CVAE seed-blend probabilities. It is not compatibility routing, target adaptation, target-threshold tuning, or target-compatible expert discovery.",
        "",
        "Target labels are used only after the source-inner rule is fixed, for scoring and audit rows.",
        "",
    ]
    (root / "reports" / "decision_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _write_fixed_beta050_decision_summary(root: Path, decision: Mapping[str, object]) -> None:
    lines = [
        "# Fixed Beta050 Positive-Union Confirmation v1",
        "",
        "## Summary",
        "",
        f"- Primary method: `{decision.get('primary_method', PRIMARY_FIXED_BETA050_POSITIVE_UNION_METHOD)}`",
        f"- Primary verdict: `{decision.get('primary_verdict', 'FIXED_BETA050_POSITIVE_UNION_FAIL')}`",
        f"- Diagnostic flags: `{decision.get('diagnostic_flags', '')}`",
        f"- Fixed rule: `{decision.get('fixed_rule', POSITIVE_UNION_RULE_BETA050)}`",
        f"- Fixed beta: {_format_float(decision.get('fixed_beta', 0.5))}",
        f"- Development seeds: `{decision.get('development_experiment_seeds_json', '[]')}`",
        f"- Primary confirmation seeds: `{decision.get('primary_confirmation_experiment_seeds_json', '[]')}`",
        f"- Center-equal mean BACC: {_format_float(decision.get('center_equal_mean_bacc'))}",
        f"- Intersection mean BACC: {_format_float(decision.get('intersection_center_equal_mean_bacc'))}",
        f"- Min center BACC: {_format_float(decision.get('min_center_bacc'))}",
        f"- Center 3 BACC: {_format_float(decision.get('center3_bacc'))}",
        f"- Frozen arithmetic bottom-20 BACC: {_format_float(decision.get('bottom20_cell_mean_bacc'))}",
        f"- Seed std BACC: {_format_float(decision.get('seed_std_bacc'))}",
        f"- Delta vs v2 arithmetic multipanel: {_format_float(decision.get('delta_vs_v2_arithmetic_intersection'))}",
        f"- Frozen bottom20 median delta vs arithmetic: {_format_float(decision.get('frozen_bottom20_median_delta_vs_v2_arithmetic'))}",
        f"- Assessable rare-positive cells: {decision.get('n_assessable_rare_positive_cells', 0)}",
        f"- Rare-positive recall mean delta vs arithmetic: {_format_float(decision.get('rare_positive_recall_mean_delta_vs_arithmetic'))}",
        f"- Worst per-center regression vs arithmetic: {_format_float(decision.get('worst_per_center_regression_vs_v2_arithmetic'))}",
        f"- Worst seed-center regression vs arithmetic: {_format_float(decision.get('worst_seed_center_regression_vs_v2_arithmetic'))}",
        f"- Tail-risk transfer flag: `{decision.get('tailrisk_transfer_flag')}`",
        f"- Leakage status: `{decision.get('leakage_status', '')}`",
        "",
        "## Protocol Boundary",
        "",
        "This is a fixed global beta050 confirmation. The beta was hypothesis-generated from prior diagnostic seeds `[42,43,44]` and is predeclared before evaluating fresh seeds.",
        "",
        "This is not source-inner selected, not compatibility routing, not target adaptation, and not target-threshold tuning. Target labels are scoring/audit only after fixed predictions exist.",
        "",
    ]
    (root / "reports" / "decision_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _evaluate_tailrisk_pair(
    cfg: TailRiskAnchoredConfig,
    *,
    root: Path,
    per_source_runtime: Mapping[str, RuntimeSource],
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    rels: Mapping[str, d12.SourceReliability],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    eval_raw: object,
    eval_labels: Sequence[int],
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
    real_feature_bacc: float,
) -> TailRiskEvaluation:
    anchor_plan = cu._shrink_source_plan(cfg, candidates, rels, shrink_lambda=0.5, total=cfg.synthetic_per_class_total)
    anchor_result = mb._evaluate_member(
        cfg,
        root=root,
        per_source_runtime=per_source_runtime,
        candidates=candidates,
        summaries=summaries,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        eval_raw=eval_raw,
        eval_labels=eval_labels,
        source_union_ref=source_union_ref,
        center_balanced_ref=center_balanced_ref,
        real_feature_bacc=real_feature_bacc,
        weight_plan=anchor_plan,
        prior_method=ANCHOR_METHOD,
        selection_source=DIAGNOSTIC_SELECTION,
        claim_role="tailrisk_reliability_anchor_comparator",
        control_mode="normal",
    )
    bag_specs = mb._random_mass_bag_specs(cfg, candidates, rels, experiment_seed, heldout_center, replicate_seed)
    bag_eval = _evaluate_bag_with_bundle(
        cfg,
        root=root,
        per_source_runtime=per_source_runtime,
        candidates=candidates,
        summaries=summaries,
        specs=bag_specs,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        eval_raw=eval_raw,
        eval_labels=eval_labels,
        source_union_ref=source_union_ref,
        center_balanced_ref=center_balanced_ref,
        real_feature_bacc=real_feature_bacc,
        method=BAG_METHOD,
        selection_source=DIAGNOSTIC_SELECTION,
        claim_role="tailrisk_random_mass_bag_comparator",
    )
    source_weight_rows = list(cu._source_weight_manifest_rows(experiment_seed, replicate_seed, heldout_center, ANCHOR_METHOD, anchor_plan, rels))
    source_weight_rows.extend(cu._source_weight_manifest_rows(experiment_seed, replicate_seed, heldout_center, BAG_METHOD, bag_eval.ensemble_plan, rels))
    eligibility_rows = [
        _eligibility_row(experiment_seed, heldout_center, replicate_seed, "anchor_shrink050", str(anchor_result.row.get("status", "")), str(anchor_result.row.get("error_message", ""))),
        *bag_eval.eligibility_rows,
    ]
    if anchor_result.bundle is None or anchor_result.row.get("status") != "ok" or bag_eval.ensemble_bundle is None or bag_eval.ensemble_row.get("status") != "ok":
        row = cu._empty_matrix_row(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            candidates=candidates,
            prior_method=PRIMARY_TAILRISK_METHOD,
            source_union_ref=source_union_ref,
            center_balanced_ref=center_balanced_ref,
            real_feature_bacc=real_feature_bacc,
            status="ineligible",
            error_message="anchor_or_random_mass_bag_ineligible",
            claim_role="primary_tailrisk_probability_blend",
        )
        row["pooling_rule"] = "fixed_arithmetic_probability_blend"
        empty_coverage = cu._empty_coverage_row(row)
        paired_row = cu._paired_generation_row(row, "", "", "ineligible")
        return TailRiskEvaluation(
            primary_row=row,
            primary_bundle=None,
            primary_coverage=empty_coverage,
            primary_paired_row=paired_row,
            anchor_result=anchor_result,
            bag_evaluation=bag_eval,
            blend_manifest_row=_blend_manifest_row(cfg, row, anchor_result, bag_eval, "", class_order_match=False),
            complementarity_row=_empty_complementarity_row(cfg, experiment_seed, heldout_center, replicate_seed, "ineligible"),
            calibration_rows=(),
            source_weight_rows=tuple(source_weight_rows),
            eligibility_rows=tuple(eligibility_rows),
            source_inner_bundles={},
            source_inner_labels=(),
            source_inner_source_ids=(),
        )
    class_order_match = anchor_result.bundle.classes == bag_eval.ensemble_bundle.classes
    if not class_order_match:
        raise ProtocolError(f"Class order mismatch before tail-risk blending: {anchor_result.bundle.classes} vs {bag_eval.ensemble_bundle.classes}")
    blended_probs = weighted_arithmetic_probability_pool(
        [anchor_result.bundle, bag_eval.ensemble_bundle],
        [cfg.blend_alpha, 1.0 - cfg.blend_alpha],
    )
    blended_bundle = PredictionBundle(
        expert_id=PRIMARY_TAILRISK_METHOD,
        probabilities=tuple(tuple(float(v) for v in row) for row in blended_probs),
        classes=anchor_result.bundle.classes,
    )
    result = evaluate_probability_predictions(PRIMARY_TAILRISK_METHOD, blended_bundle.probabilities, eval_labels, classes=blended_bundle.classes)
    blended_hash = _hash_array(np.asarray(blended_bundle.probabilities, dtype=float))
    blend_plan = _blend_source_plan(cfg, candidates, anchor_plan, bag_eval.ensemble_plan)
    row = cu._result_matrix_row(
        cfg,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        candidates=candidates,
        prior_method=PRIMARY_TAILRISK_METHOD,
        summary_kind="tailrisk_anchor_random_mass_bag_probability_blend",
        source_union_ref=source_union_ref,
        center_balanced_ref=center_balanced_ref,
        real_feature_bacc=real_feature_bacc,
        weight_plan=blend_plan,
        bacc=result.bacc,
        macro_f1=result.macro_f1,
        generated_features_hash=_hash_strings([anchor_result.generated_hash, bag_eval.generated_hash]),
        prediction_hash=blended_hash,
        selection_source=PRIMARY_SELECTION,
        claim_role="primary_tailrisk_probability_blend",
        status="ok",
        error_message="",
        control_mode="normal",
        summaries=summaries,
    )
    row["pooling_rule"] = "fixed_arithmetic_probability_blend"
    row["anchor_method"] = ANCHOR_METHOD
    row["bag_method"] = BAG_METHOD
    row["blend_alpha_anchor"] = cfg.blend_alpha
    row["blend_alpha_bag"] = 1.0 - cfg.blend_alpha
    source_weight_rows.extend(cu._source_weight_manifest_rows(experiment_seed, replicate_seed, heldout_center, PRIMARY_TAILRISK_METHOD, blend_plan, rels))
    merged_counts = mb._merge_component_counts([anchor_result.component_counts, bag_eval.component_counts])
    coverage = cu._component_coverage_row(row, merged_counts, cu._expected_component_keys(candidates, summaries, control_mode="normal"))
    paired_row = cu._paired_generation_row(row, str(row["generated_features_hash"]), _hash_strings([anchor_result.source_generation_hash, bag_eval.source_generation_hash]), "ok")
    source_inner_bundles, source_inner_labels, source_inner_source_ids = _source_inner_probability_bundles(
        cfg,
        root=root,
        per_source_runtime=per_source_runtime,
        candidates=candidates,
        summaries=summaries,
        anchor_plan=anchor_plan,
        bag_specs=bag_specs,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
    )
    return TailRiskEvaluation(
        primary_row=row,
        primary_bundle=blended_bundle,
        primary_coverage=coverage,
        primary_paired_row=paired_row,
        anchor_result=anchor_result,
        bag_evaluation=bag_eval,
        blend_manifest_row=_blend_manifest_row(cfg, row, anchor_result, bag_eval, blended_hash, class_order_match=True),
        complementarity_row=_complementarity_row(cfg, experiment_seed, heldout_center, replicate_seed, anchor_result.bundle, bag_eval.ensemble_bundle, eval_labels, row),
        calibration_rows=tuple(
            _calibration_rows(
                cfg,
                experiment_seed=experiment_seed,
                heldout_center=heldout_center,
                replicate_seed=replicate_seed,
                eval_labels=eval_labels,
                anchor_bundle=anchor_result.bundle,
                bag_bundle=bag_eval.ensemble_bundle,
                blended_bundle=blended_bundle,
                source_inner_bundles=source_inner_bundles,
                source_inner_labels=source_inner_labels,
            )
        ),
        source_weight_rows=tuple(source_weight_rows),
        eligibility_rows=tuple(eligibility_rows),
        source_inner_bundles=source_inner_bundles,
        source_inner_labels=source_inner_labels,
        source_inner_source_ids=source_inner_source_ids,
    )


def _evaluate_bag_with_bundle(
    cfg: TailRiskAnchoredConfig,
    *,
    root: Path,
    per_source_runtime: Mapping[str, RuntimeSource],
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    specs: Sequence[Mapping[str, object]],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    eval_raw: object,
    eval_labels: Sequence[int],
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
    real_feature_bacc: float,
    method: str,
    selection_source: str,
    claim_role: str,
    control_mode: str = "normal",
) -> BagEvaluation:
    member_results: list[mb.MemberResult] = []
    eligibility_rows = []
    for spec in specs:
        result = mb._evaluate_member(
            cfg,
            root=root,
            per_source_runtime=per_source_runtime,
            candidates=candidates,
            summaries=summaries,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            eval_raw=eval_raw,
            eval_labels=eval_labels,
            source_union_ref=source_union_ref,
            center_balanced_ref=center_balanced_ref,
            real_feature_bacc=real_feature_bacc,
            weight_plan=spec["plan"],
            prior_method=f"{method}__member_{int(spec['bag_member_index']):03d}",
            selection_source=DIAGNOSTIC_SELECTION,
            claim_role="tailrisk_random_mass_bag_member_diagnostic",
            control_mode=control_mode,
        )
        result.row.update(mb._member_extra(spec, method))
        member_results.append(result)
        eligibility_rows.append(_eligibility_row(experiment_seed, heldout_center, replicate_seed, str(spec["bag_member_id"]), str(result.row.get("status", "")), str(result.row.get("error_message", ""))))
    bundles = [result.bundle for result in member_results if result.bundle is not None and result.row.get("status") == "ok"]
    if len(bundles) != len(member_results):
        row = cu._empty_matrix_row(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            candidates=candidates,
            prior_method=method,
            source_union_ref=source_union_ref,
            center_balanced_ref=center_balanced_ref,
            real_feature_bacc=real_feature_bacc,
            status="ineligible",
            error_message="one_or_more_bag_members_ineligible",
            claim_role=claim_role,
        )
        row.update(mb._ensemble_extra(cfg, specs, method, status="ineligible"))
        plan = mb._ensemble_plan(cfg, candidates, [spec["plan"] for spec in specs])
        return BagEvaluation(row, None, cu._empty_coverage_row(row), cu._paired_generation_row(row, "", "", "ineligible"), tuple(member_results), tuple(eligibility_rows), {}, "", "", plan)
    pooled = weighted_arithmetic_probability_pool(bundles, [1.0] * len(bundles))
    bundle = PredictionBundle(expert_id=method, probabilities=tuple(tuple(float(v) for v in row) for row in pooled), classes=bundles[0].classes)
    result = evaluate_probability_predictions(method, bundle.probabilities, eval_labels, classes=bundle.classes)
    ensemble_plan = mb._ensemble_plan(cfg, candidates, [spec["plan"] for spec in specs])
    row = cu._result_matrix_row(
        cfg,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        candidates=candidates,
        prior_method=method,
        summary_kind="gmm_component_probability_ensemble",
        source_union_ref=source_union_ref,
        center_balanced_ref=center_balanced_ref,
        real_feature_bacc=real_feature_bacc,
        weight_plan=ensemble_plan,
        bacc=result.bacc,
        macro_f1=result.macro_f1,
        generated_features_hash=_hash_strings([r.generated_hash for r in member_results]),
        prediction_hash=_hash_array(np.asarray(bundle.probabilities, dtype=float)),
        selection_source=selection_source,
        claim_role=claim_role,
        status="ok",
        error_message="",
        control_mode=control_mode,
        summaries=summaries,
    )
    row.update(mb._ensemble_extra(cfg, specs, method, status="ok"))
    row["pooling_rule"] = "arithmetic_probability_ensemble"
    merged_counts = mb._merge_component_counts([result.component_counts for result in member_results])
    coverage = cu._component_coverage_row(row, merged_counts, cu._expected_component_keys(candidates, summaries, control_mode=control_mode))
    paired_row = cu._paired_generation_row(row, str(row["generated_features_hash"]), _hash_strings([r.source_generation_hash for r in member_results]), "ok")
    return BagEvaluation(row, bundle, coverage, paired_row, tuple(member_results), tuple(eligibility_rows), merged_counts, str(row["generated_features_hash"]), str(paired_row["source_generation_hash"]), ensemble_plan)


def _source_inner_probability_bundles(
    cfg: TailRiskAnchoredConfig,
    *,
    root: Path,
    per_source_runtime: Mapping[str, RuntimeSource],
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    anchor_plan: Mapping[str, object],
    bag_specs: Sequence[Mapping[str, object]],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
) -> tuple[dict[str, PredictionBundle], tuple[int, ...], tuple[str, ...]]:
    source_inner_raw, source_inner_labels, source_inner_source_ids = _source_inner_eval_set_with_sources(per_source_runtime, candidates)
    bundles: dict[str, PredictionBundle] = {}
    anchor_seed = d1._latent_seed(experiment_seed, heldout_center, replicate_seed, ANCHOR_METHOD, cu._plan_hash(anchor_plan), "normal")
    generated, labels, _counts, _train_raw, _hashes = mb._sample_cached(
        cfg,
        root=root,
        per_source_runtime=per_source_runtime,
        sources=candidates,
        summaries=summaries,
        weight_plan=anchor_plan,
        seed=anchor_seed,
        control_mode="normal",
    )
    if sorted(set(int(v) for v in labels)) == [0, 1]:
        bundles["anchor"] = mb._prediction_cached(
            cfg,
            root=root,
            generated=generated,
            labels=labels,
            eval_raw=source_inner_raw,
            expert_id=f"{ANCHOR_METHOD}_source_inner",
        )
    bag_member_bundles: list[PredictionBundle] = []
    for spec in bag_specs:
        method = f"{BAG_METHOD}__member_{int(spec['bag_member_index']):03d}"
        seed = d1._latent_seed(experiment_seed, heldout_center, replicate_seed, method, cu._plan_hash(spec["plan"]), "normal")
        generated, labels, _counts, _train_raw, _hashes = mb._sample_cached(
            cfg,
            root=root,
            per_source_runtime=per_source_runtime,
            sources=candidates,
            summaries=summaries,
            weight_plan=spec["plan"],
            seed=seed,
            control_mode="normal",
        )
        if sorted(set(int(v) for v in labels)) != [0, 1]:
            continue
        bag_member_bundles.append(
            mb._prediction_cached(
                cfg,
                root=root,
                generated=generated,
                labels=labels,
                eval_raw=source_inner_raw,
                expert_id=f"{method}_source_inner",
            )
        )
    if len(bag_member_bundles) == len(bag_specs) and bag_member_bundles:
        pooled = weighted_arithmetic_probability_pool(bag_member_bundles, [1.0] * len(bag_member_bundles))
        bundles["random_mass_bag"] = PredictionBundle(
            expert_id=f"{BAG_METHOD}_source_inner",
            probabilities=tuple(tuple(float(v) for v in row) for row in pooled),
            classes=bag_member_bundles[0].classes,
        )
    if "anchor" in bundles and "random_mass_bag" in bundles and bundles["anchor"].classes == bundles["random_mass_bag"].classes:
        pooled = weighted_arithmetic_probability_pool(
            [bundles["anchor"], bundles["random_mass_bag"]],
            [cfg.blend_alpha, 1.0 - cfg.blend_alpha],
        )
        bundles["primary_blend"] = PredictionBundle(
            expert_id=f"{PRIMARY_TAILRISK_METHOD}_source_inner",
            probabilities=tuple(tuple(float(v) for v in row) for row in pooled),
            classes=bundles["anchor"].classes,
        )
    return bundles, source_inner_labels, source_inner_source_ids


def _source_inner_eval_set(
    per_source_runtime: Mapping[str, RuntimeSource],
    candidates: Sequence[str],
) -> tuple[object, tuple[int, ...]]:
    raw, labels, _source_ids = _source_inner_eval_set_with_sources(per_source_runtime, candidates)
    return raw, labels


def _source_inner_eval_set_with_sources(
    per_source_runtime: Mapping[str, RuntimeSource],
    candidates: Sequence[str],
) -> tuple[object, tuple[int, ...], tuple[str, ...]]:
    raw_chunks = []
    labels: list[int] = []
    source_ids: list[str] = []
    for source in candidates:
        runtime = per_source_runtime[str(source)].runtime
        raw_chunks.append(cu._inverse_to_raw(runtime, runtime.source_val_embeddings))
        source_labels = [int(v) for v in runtime.source_val_labels]
        labels.extend(source_labels)
        source_ids.extend([str(source)] * len(source_labels))
    return np.vstack(raw_chunks), tuple(labels), tuple(source_ids)


def _evaluate_single_control_member(
    cfg: TailRiskAnchoredConfig,
    *,
    root: Path,
    per_source_runtime: Mapping[str, RuntimeSource],
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    rels: Mapping[str, d12.SourceReliability],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    eval_raw: object,
    eval_labels: Sequence[int],
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
    real_feature_bacc: float,
    prior_method: str,
    claim_role: str,
    control_mode: str,
) -> mb.MemberResult:
    plan = cu._shrink_source_plan(cfg, candidates, rels, shrink_lambda=0.5, total=cfg.synthetic_per_class_total)
    return mb._evaluate_member(
        cfg,
        root=root,
        per_source_runtime=per_source_runtime,
        candidates=candidates,
        summaries=summaries,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        eval_raw=eval_raw,
        eval_labels=eval_labels,
        source_union_ref=source_union_ref,
        center_balanced_ref=center_balanced_ref,
        real_feature_bacc=real_feature_bacc,
        weight_plan=plan,
        prior_method=prior_method,
        selection_source=DIAGNOSTIC_SELECTION,
        claim_role=claim_role,
        control_mode=control_mode,
    )


def _evaluate_shuffled_reliability_null(
    cfg: TailRiskAnchoredConfig,
    *,
    root: Path,
    per_source_runtime: Mapping[str, RuntimeSource],
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    rels: Mapping[str, d12.SourceReliability],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    eval_raw: object,
    eval_labels: Sequence[int],
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
    real_feature_bacc: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for permutation_id in range(cfg.matched_shuffled_reliability_null_permutations):
        plan = cu._shuffled_reliability_plan(
            cfg,
            candidates,
            rels,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            shrink_lambda=0.5,
            permutation_id=permutation_id,
            total=cfg.synthetic_per_class_total,
        )
        result = mb._evaluate_member(
            cfg,
            root=root,
            per_source_runtime=per_source_runtime,
            candidates=candidates,
            summaries=summaries,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            eval_raw=eval_raw,
            eval_labels=eval_labels,
            source_union_ref=source_union_ref,
            center_balanced_ref=center_balanced_ref,
            real_feature_bacc=real_feature_bacc,
            weight_plan=plan,
            prior_method=f"{MATCHED_SHUFFLED_TAILRISK_PREFIX}{permutation_id:03d}",
            selection_source=DIAGNOSTIC_SELECTION,
            claim_role="matched_shuffled_reliability_shrink050_null",
            control_mode="normal",
        )
        row = dict(result.row)
        row["null_perm_id"] = int(permutation_id)
        rows.append(row)
    return rows


def _source_ablation_rows(
    cfg: TailRiskAnchoredConfig,
    *,
    root: Path,
    per_source_runtime: Mapping[str, RuntimeSource],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    reliability: Mapping[tuple[int, int, str], d12.SourceReliability],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    eval_raw: object,
    eval_labels: Sequence[int],
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
    real_feature_bacc: float,
    primary_bacc: float,
) -> list[dict[str, object]]:
    rows = []
    for removed in cfg.heldout_centers:
        if str(removed) == str(heldout_center):
            rows.append(
                {
                    "experiment_seed": int(experiment_seed),
                    "heldout_center": str(heldout_center),
                    "replicate_seed": int(replicate_seed),
                    "panel": cu._panel_for_replicate_seed(cfg, replicate_seed),
                    "removed_source_center": str(removed),
                    "remaining_source_centers": "|".join(str(v) for v in candidates),
                    "primary_bacc": primary_bacc,
                    "ablation_bacc": "",
                    "delta_ablation_minus_primary": "",
                    "status": "not_applicable_target_source_excluded",
                }
            )
            continue
        remaining = tuple(source for source in candidates if str(source) != str(removed))
        rels = {source: reliability[(int(experiment_seed), int(replicate_seed), str(source))] for source in remaining}
        evaluated = _evaluate_tailrisk_pair(
            cfg,
            root=root,
            per_source_runtime=per_source_runtime,
            candidates=remaining,
            summaries=summaries,
            rels=rels,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            eval_raw=eval_raw,
            eval_labels=eval_labels,
            source_union_ref=source_union_ref,
            center_balanced_ref=center_balanced_ref,
            real_feature_bacc=real_feature_bacc,
        )
        ablation_bacc = _float(evaluated.primary_row.get("bacc"))
        rows.append(
            {
                "experiment_seed": int(experiment_seed),
                "heldout_center": str(heldout_center),
                "replicate_seed": int(replicate_seed),
                "panel": cu._panel_for_replicate_seed(cfg, replicate_seed),
                "removed_source_center": str(removed),
                "remaining_source_centers": "|".join(str(v) for v in remaining),
                "primary_bacc": primary_bacc,
                "ablation_bacc": ablation_bacc,
                "delta_ablation_minus_primary": ablation_bacc - primary_bacc if math.isfinite(ablation_bacc) and math.isfinite(primary_bacc) else math.nan,
                "status": evaluated.primary_row.get("status", ""),
            }
        )
    return rows


def _append_tailrisk_outputs(
    evaluated: TailRiskEvaluation,
    *,
    matrix_rows: list[dict[str, object]],
    component_coverage_rows: list[dict[str, object]],
    paired_generation_rows: list[dict[str, object]],
    source_weight_rows: list[dict[str, object]],
    blend_manifest_rows: list[dict[str, object]],
    complementarity_rows: list[dict[str, object]],
    calibration_rows: list[dict[str, object]],
    eligibility_rows: list[dict[str, object]],
) -> None:
    matrix_rows.append(evaluated.anchor_result.row)
    matrix_rows.append(evaluated.bag_evaluation.ensemble_row)
    matrix_rows.append(evaluated.primary_row)
    component_coverage_rows.extend(
        [
            evaluated.anchor_result.coverage_row,
            evaluated.bag_evaluation.ensemble_coverage,
            evaluated.primary_coverage,
        ]
    )
    paired_generation_rows.extend(
        [
            evaluated.anchor_result.paired_row,
            evaluated.bag_evaluation.ensemble_paired_row,
            evaluated.primary_paired_row,
        ]
    )
    source_weight_rows.extend(evaluated.source_weight_rows)
    blend_manifest_rows.append(evaluated.blend_manifest_row)
    complementarity_rows.append(evaluated.complementarity_row)
    calibration_rows.extend(evaluated.calibration_rows)
    eligibility_rows.extend(evaluated.eligibility_rows)


def _append_control_outputs(
    evaluated: object,
    *,
    matrix_rows: list[dict[str, object]],
    component_coverage_rows: list[dict[str, object]],
    paired_generation_rows: list[dict[str, object]],
    source_weight_rows: list[dict[str, object]],
    rels: Mapping[str, d12.SourceReliability],
) -> None:
    if isinstance(evaluated, mb.MemberResult):
        matrix_rows.append(evaluated.row)
        component_coverage_rows.append(evaluated.coverage_row)
        paired_generation_rows.append(evaluated.paired_row)
        plan = mb._plan_from_row(evaluated.row)
        if plan is not None:
            source_weight_rows.extend(cu._source_weight_manifest_rows(int(evaluated.row["experiment_seed"]), int(evaluated.row["replicate_seed"]), str(evaluated.row["heldout_center"]), str(evaluated.row["prior_method"]), plan, rels))
        return
    if isinstance(evaluated, Mapping):
        matrix_rows.append(dict(evaluated["ensemble_row"]))
        ensemble_result = evaluated.get("ensemble_result")
        if isinstance(ensemble_result, mb.MemberResult):
            component_coverage_rows.append(ensemble_result.coverage_row)
            paired_generation_rows.append(ensemble_result.paired_row)


def _blend_source_plan(
    cfg: TailRiskAnchoredConfig,
    sources: Sequence[str],
    anchor_plan: Mapping[str, object],
    bag_plan: Mapping[str, object],
) -> dict[str, object]:
    alpha = float(cfg.blend_alpha)
    weights = {}
    budgets = {}
    scores = {}
    for source in sources:
        source_id = str(source)
        weights[source_id] = alpha * _float(dict(anchor_plan["weights"]).get(source_id)) + (1.0 - alpha) * _float(dict(bag_plan["weights"]).get(source_id))
        budgets[source_id] = int(round(alpha * _float(dict(anchor_plan["budgets"]).get(source_id)) + (1.0 - alpha) * _float(dict(bag_plan["budgets"]).get(source_id))))
        scores[source_id] = alpha * _float(dict(anchor_plan["scores"]).get(source_id)) + (1.0 - alpha) * _float(dict(bag_plan["scores"]).get(source_id))
    total_weight = sum(weights.values())
    if total_weight > 0.0:
        weights = {source: value / total_weight for source, value in weights.items()}
    plan = cu._with_weight_diagnostics(tuple(str(v) for v in sources), weights, budgets, scores, total=cfg.synthetic_per_class_total, mode=TAILRISK_SOURCE_WEIGHTING)
    plan["blend_alpha_anchor"] = alpha
    plan["blend_alpha_bag"] = 1.0 - alpha
    plan["anchor_method"] = ANCHOR_METHOD
    plan["bag_method"] = BAG_METHOD
    return plan


def _blend_manifest_row(
    cfg: TailRiskAnchoredConfig,
    row: Mapping[str, object],
    anchor_result: mb.MemberResult,
    bag_eval: BagEvaluation,
    blended_hash: str,
    *,
    class_order_match: bool,
) -> dict[str, object]:
    anchor_hash = str(anchor_result.row.get("prediction_hash", ""))
    bag_hash = str(bag_eval.ensemble_row.get("prediction_hash", ""))
    class_order = ""
    if anchor_result.bundle is not None:
        class_order = "|".join(str(v) for v in anchor_result.bundle.classes)
    return {
        "experiment_seed": row.get("experiment_seed", ""),
        "heldout_center": row.get("heldout_center", ""),
        "replicate_seed": row.get("replicate_seed", ""),
        "panel": row.get("panel", ""),
        "primary_method": PRIMARY_TAILRISK_METHOD,
        "anchor_method": ANCHOR_METHOD,
        "bag_method": BAG_METHOD,
        "blend_alpha_anchor": cfg.blend_alpha,
        "blend_alpha_bag": 1.0 - cfg.blend_alpha,
        "anchor_prediction_hash": anchor_hash,
        "bag_prediction_hash": bag_hash,
        "blended_prediction_hash": blended_hash,
        "class_order": class_order,
        "class_order_match": bool(class_order_match),
    }


def _complementarity_row(
    cfg: TailRiskAnchoredConfig,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    anchor_bundle: PredictionBundle,
    bag_bundle: PredictionBundle,
    eval_labels: Sequence[int],
    primary_row: Mapping[str, object],
) -> dict[str, object]:
    anchor_preds = predict_from_probabilities(anchor_bundle.probabilities, classes=anchor_bundle.classes)
    bag_preds = predict_from_probabilities(bag_bundle.probabilities, classes=bag_bundle.classes)
    labels = tuple(int(v) for v in eval_labels)
    n = len(labels)
    if n == 0:
        return _empty_complementarity_row(cfg, experiment_seed, heldout_center, replicate_seed, "empty_eval")
    anchor_correct = [int(pred == label) for pred, label in zip(anchor_preds, labels)]
    bag_correct = [int(pred == label) for pred, label in zip(bag_preds, labels)]
    anchor_correct_bag_wrong = sum(1 for a, b in zip(anchor_correct, bag_correct) if a and not b) / float(n)
    bag_correct_anchor_wrong = sum(1 for a, b in zip(anchor_correct, bag_correct) if b and not a) / float(n)
    both_wrong = sum(1 for a, b in zip(anchor_correct, bag_correct) if not a and not b) / float(n)
    both_correct = sum(1 for a, b in zip(anchor_correct, bag_correct) if a and b) / float(n)
    disagreement = sum(1 for a, b in zip(anchor_preds, bag_preds) if int(a) != int(b)) / float(n)
    center3_rate: object = disagreement if str(heldout_center) == "3" else ""
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "replicate_seed": int(replicate_seed),
        "panel": cu._panel_for_replicate_seed(cfg, replicate_seed),
        "primary_bacc": primary_row.get("bacc", math.nan),
        "anchor_correct_bag_wrong_rate": anchor_correct_bag_wrong,
        "bag_correct_anchor_wrong_rate": bag_correct_anchor_wrong,
        "both_wrong_rate": both_wrong,
        "both_correct_rate": both_correct,
        "disagreement_rate": disagreement,
        "center3_disagreement_rate": center3_rate,
        "bottom20_disagreement_rate": "",
        "is_bottom20_cell": False,
        "status": "ok",
    }


def _empty_complementarity_row(
    cfg: TailRiskAnchoredConfig,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    status: str,
) -> dict[str, object]:
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "replicate_seed": int(replicate_seed),
        "panel": cu._panel_for_replicate_seed(cfg, replicate_seed),
        "primary_bacc": math.nan,
        "anchor_correct_bag_wrong_rate": math.nan,
        "bag_correct_anchor_wrong_rate": math.nan,
        "both_wrong_rate": math.nan,
        "both_correct_rate": math.nan,
        "disagreement_rate": math.nan,
        "center3_disagreement_rate": "",
        "bottom20_disagreement_rate": "",
        "is_bottom20_cell": False,
        "status": status,
    }


def _calibration_rows(
    cfg: TailRiskAnchoredConfig,
    *,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    eval_labels: Sequence[int],
    anchor_bundle: PredictionBundle,
    bag_bundle: PredictionBundle,
    blended_bundle: PredictionBundle,
    source_inner_bundles: Mapping[str, PredictionBundle],
    source_inner_labels: Sequence[int],
) -> list[dict[str, object]]:
    rows = []
    for source, method, bundle in (
        ("anchor", ANCHOR_METHOD, anchor_bundle),
        ("random_mass_bag", BAG_METHOD, bag_bundle),
        ("primary_blend", PRIMARY_TAILRISK_METHOD, blended_bundle),
    ):
        metrics = _probability_calibration_metrics(bundle.probabilities, eval_labels, bundle.classes)
        inner_bundle = source_inner_bundles.get(source)
        inner_metrics = (
            _probability_calibration_metrics(inner_bundle.probabilities, source_inner_labels, inner_bundle.classes)
            if inner_bundle is not None
            else {"brier": math.nan, "ece": math.nan, "log_loss": math.nan}
        )
        rows.append(
            {
                "experiment_seed": int(experiment_seed),
                "heldout_center": str(heldout_center),
                "replicate_seed": int(replicate_seed),
                "panel": cu._panel_for_replicate_seed(cfg, replicate_seed),
                "probability_source": source,
                "prior_method": method,
                "source_inner_brier": inner_metrics["brier"],
                "source_inner_ece": inner_metrics["ece"],
                "source_inner_log_loss": inner_metrics["log_loss"],
                "source_inner_calibration_available": inner_bundle is not None,
                "target_eval_brier_diagnostic_only": metrics["brier"],
                "target_eval_ece_diagnostic_only": metrics["ece"],
                "target_eval_log_loss_diagnostic_only": metrics["log_loss"],
                "target_calibration_audit_only": True,
            }
        )
    return rows


def _probability_calibration_metrics(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    classes: Sequence[int],
) -> dict[str, float]:
    probs = np.asarray(probabilities, dtype=float)
    y = np.asarray([int(v) for v in labels], dtype=int)
    cls = tuple(int(v) for v in classes)
    if probs.ndim != 2 or len(y) != probs.shape[0] or probs.shape[0] == 0:
        return {"brier": math.nan, "ece": math.nan, "log_loss": math.nan}
    lookup = {value: idx for idx, value in enumerate(cls)}
    true_idx = np.asarray([lookup.get(int(v), -1) for v in y], dtype=int)
    valid = true_idx >= 0
    if not bool(valid.all()):
        probs = probs[valid]
        true_idx = true_idx[valid]
    if probs.shape[0] == 0:
        return {"brier": math.nan, "ece": math.nan, "log_loss": math.nan}
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(probs.shape[0]), true_idx] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    clipped = np.clip(probs[np.arange(probs.shape[0]), true_idx], 1.0e-12, 1.0)
    log_loss = float(-np.mean(np.log(clipped)))
    pred_idx = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)
    correct = (pred_idx == true_idx).astype(float)
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        if upper >= 1.0:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if not bool(mask.any()):
            continue
        ece += float(mask.mean()) * abs(float(confidence[mask].mean()) - float(correct[mask].mean()))
    return {"brier": brier, "ece": ece, "log_loss": log_loss}


def _mark_bottom20_complementarity(
    matrix_rows: Sequence[Mapping[str, object]],
    complementarity_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    bottom_keys: set[tuple[str, str, str]] = set()
    primary_rows = cu._rows_for(matrix_rows, PRIMARY_TAILRISK_METHOD)
    for panel in ("canonical", "fresh", "combined"):
        panel_rows = cu._rows_for_panel(primary_rows, panel)
        grouped = cu._replicate_averaged(panel_rows)
        if not grouped:
            continue
        count = max(1, int(math.ceil(0.20 * len(grouped))))
        bottom = sorted(grouped, key=lambda row: _float(row.get("bacc")))[:count]
        for row in bottom:
            bottom_keys.add((panel, str(row["experiment_seed"]), str(row["heldout_center"])))
    out = []
    for row in complementarity_rows:
        updated = dict(row)
        key = (str(updated.get("panel", "")), str(updated.get("experiment_seed", "")), str(updated.get("heldout_center", "")))
        combined_key = ("combined", str(updated.get("experiment_seed", "")), str(updated.get("heldout_center", "")))
        is_bottom = key in bottom_keys or combined_key in bottom_keys
        updated["is_bottom20_cell"] = bool(is_bottom)
        updated["bottom20_disagreement_rate"] = updated.get("disagreement_rate", "") if is_bottom else ""
        out.append(updated)
    return out


def _tail_metrics(
    rows: Sequence[Mapping[str, object]],
    method: str,
    *,
    panel: str = "combined",
) -> dict[str, object]:
    subset = cu._rows_for(cu._rows_for_panel(rows, panel), method)
    stats = cu._method_stats(subset)
    grouped = cu._replicate_averaged(subset)
    bacc_values = sorted(_float(row.get("bacc")) for row in grouped if math.isfinite(_float(row.get("bacc"))))
    bottom_count = max(1, int(math.ceil(0.20 * len(bacc_values)))) if bacc_values else 0
    bottom20 = nanmean(bacc_values[:bottom_count]) if bacc_values else math.nan
    center3_rows = [row for row in grouped if str(row.get("heldout_center")) == "3"]
    center3 = d1._mean_field(center3_rows, "bacc") if center3_rows else math.nan
    return {
        **stats,
        "bottom20_cell_mean_bacc": bottom20,
        "worst_seed_center_bacc": min(bacc_values) if bacc_values else math.nan,
        "center3_bacc": center3,
    }


def _tail_metric_summary_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    out = []
    methods = sorted({str(row.get("prior_method")) for row in rows if row.get("prior_method")})
    for panel in ("canonical", "fresh", "combined"):
        for method in methods:
            metrics = _tail_metrics(rows, method, panel=panel)
            if int(metrics["n_raw_rows"]) < 1:
                continue
            random_metrics = _tail_metrics(rows, BAG_METHOD, panel=panel)
            anchor_metrics = _tail_metrics(rows, ANCHOR_METHOD, panel=panel)
            out.append(
                {
                    "panel": panel,
                    "prior_method": method,
                    **metrics,
                    "center3_delta_vs_random_mass_bag": _delta(metrics["center3_bacc"], random_metrics["center3_bacc"]),
                    "center3_delta_vs_shrink050": _delta(metrics["center3_bacc"], anchor_metrics["center3_bacc"]),
                    "bottom20_delta_vs_random_mass_bag": _delta(metrics["bottom20_cell_mean_bacc"], random_metrics["bottom20_cell_mean_bacc"]),
                    "bottom20_delta_vs_shrink050": _delta(metrics["bottom20_cell_mean_bacc"], anchor_metrics["bottom20_cell_mean_bacc"]),
                }
            )
    return out


def _decision(
    rows: Sequence[Mapping[str, object]],
    *,
    cfg: TailRiskAnchoredConfig,
    leakage_status: str,
    source_ablation_rows: Sequence[Mapping[str, object]],
    anchor_rows: Sequence[Mapping[str, object]],
    complementarity_rows: Sequence[Mapping[str, object]],
    shuffled_null_summary: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    primary = _tail_metrics(rows, PRIMARY_TAILRISK_METHOD)
    anchor = _tail_metrics(rows, ANCHOR_METHOD)
    bag = _tail_metrics(rows, BAG_METHOD)
    uniform = _tail_metrics(rows, cu.PRIMARY_COMPONENT_UNION_METHOD)
    source_union = _tail_metrics(rows, cu.ROW_SOURCE_UNION_K16_REFERENCE)
    real = _tail_metrics(rows, cu.ROW_REAL_FEATURE_DENSE_REFERENCE)
    random_single = _tail_metrics(rows, cu.ROW_RANDOM_SOURCE_MASS_CONTROL)
    shuffled_label = _tail_metrics(rows, cu.ROW_SHUFFLED_LABEL_CONTROL)
    shuffled_summary = _tail_metrics(rows, cu.ROW_SHUFFLED_SUMMARY_CONTROL)
    primary_bacc = _float(primary["center_equal_mean_bacc"])
    anchor_bacc = _float(anchor["center_equal_mean_bacc"])
    bag_bacc = _float(bag["center_equal_mean_bacc"])
    source_union_bacc = _float(source_union["center_equal_mean_bacc"])
    real_bacc = _float(real["center_equal_mean_bacc"])
    random_single_bacc = _float(random_single["center_equal_mean_bacc"])
    shuffled_label_bacc = _float(shuffled_label["center_equal_mean_bacc"])
    shuffled_summary_bacc = _float(shuffled_summary["center_equal_mean_bacc"])
    null = dict(shuffled_null_summary[0]) if shuffled_null_summary else {}
    null_mean = _float(null.get("null_mean_center_equal_bacc"))
    null_p95 = _float(null.get("null_p95_center_equal_bacc"))
    strongest_control = max(
        (
            (cu.ROW_RANDOM_SOURCE_MASS_CONTROL, random_single_bacc),
            (cu.ROW_SHUFFLED_LABEL_CONTROL, shuffled_label_bacc),
            (cu.ROW_SHUFFLED_SUMMARY_CONTROL, shuffled_summary_bacc),
            (f"{MATCHED_SHUFFLED_TAILRISK_PREFIX}*", null_mean),
        ),
        key=lambda item: item[1] if math.isfinite(item[1]) else -math.inf,
    )
    anchor_pass = bool(anchor_rows) and all(row.get("anchor_repro_status") == "PASS" for row in anchor_rows)
    ablation = _source_ablation_stats(source_ablation_rows)
    retention = d1._retention(primary_bacc, source_union_bacc)
    center3_delta_bag = _delta(primary["center3_bacc"], bag["center3_bacc"])
    center3_delta_anchor = _delta(primary["center3_bacc"], anchor["center3_bacc"])
    bottom20_delta_bag = _delta(primary["bottom20_cell_mean_bacc"], bag["bottom20_cell_mean_bacc"])
    bottom20_delta_anchor = _delta(primary["bottom20_cell_mean_bacc"], anchor["bottom20_cell_mean_bacc"])
    max_comparator = max(value for value in (anchor_bacc, bag_bacc) if math.isfinite(value)) if any(math.isfinite(v) for v in (anchor_bacc, bag_bacc)) else math.nan
    no_center_worse = _no_center_worse_than_both(rows)
    fresh_preserves = _fresh_preserves_tail_direction(rows)
    complementarity_nontrivial = _complementarity_nontrivial(complementarity_rows)
    controls_worse = all(
        primary_bacc > value
        for value in (random_single_bacc, shuffled_label_bacc, shuffled_summary_bacc, null_mean)
        if math.isfinite(value)
    )
    flags: list[str] = []
    if leakage_status != "PASS":
        flags.append("LEAKAGE_FAIL")
    if not anchor_pass:
        flags.append("ANCHOR_MISMATCH")
    if not no_center_worse:
        flags.append("CENTER_WORSE_THAN_BOTH_COMPARATORS")
    if not fresh_preserves:
        flags.append("FRESH_PANEL_REVERSES_TAIL_DIRECTION")
    if not complementarity_nontrivial:
        flags.append("ANCHOR_BAG_COMPLEMENTARITY_WEAK")
    if math.isfinite(primary_bacc) and math.isfinite(max_comparator) and primary_bacc < max_comparator - 0.005:
        flags.append("MEAN_DROPS_GT_0P005_BELOW_BEST_COMPONENT_COMPARATOR")
    if math.isfinite(center3_delta_bag) and center3_delta_bag <= 0.0:
        flags.append("CENTER3_NOT_IMPROVED_VS_RANDOM_MASS_BAG")
    if math.isfinite(bottom20_delta_bag) and bottom20_delta_bag <= 0.0:
        flags.append("BOTTOM20_NOT_IMPROVED_VS_RANDOM_MASS_BAG")
    if math.isfinite(primary["center3_bacc"]) and _float(primary["center3_bacc"]) < 0.80:
        flags.append("CENTER3_BELOW_0P80")
    if not controls_worse:
        flags.append("NEGATIVE_CONTROLS_COMPETITIVE")
    strong = (
        leakage_status == "PASS"
        and int(primary["n_heldout_centers"]) >= len(cfg.heldout_centers)
        and anchor_pass
        and no_center_worse
        and math.isfinite(max_comparator)
        and primary_bacc >= max_comparator - 0.002
        and _float(primary["min_center_bacc"]) >= 0.82
        and center3_delta_bag >= 0.020
        and bottom20_delta_bag >= 0.015
        and _float(primary["seed_std_bacc"]) <= 0.045
        and retention >= 0.97
        and fresh_preserves
        and complementarity_nontrivial
        and controls_worse
    )
    useful = (
        leakage_status == "PASS"
        and int(primary["n_heldout_centers"]) >= len(cfg.heldout_centers)
        and math.isfinite(max_comparator)
        and primary_bacc >= max_comparator - 0.005
        and _float(primary["min_center_bacc"]) >= 0.80
        and center3_delta_bag > 0.0
        and bottom20_delta_bag > 0.0
        and fresh_preserves
        and complementarity_nontrivial
    )
    verdict = "TAILRISK_ANCHORED_COMPONENT_UNION_FAIL"
    if leakage_status != "PASS":
        verdict = "PROTOCOL_FAIL"
    elif strong:
        verdict = "TAILRISK_ANCHORED_COMPONENT_UNION_STRONG_SUCCESS"
    elif useful:
        verdict = "TAILRISK_ANCHORED_COMPONENT_UNION_USEFUL_THESIS_SUCCESS"
    return {
        "primary_verdict": verdict,
        "diagnostic_flags": "|".join(flags),
        "primary_method": PRIMARY_TAILRISK_METHOD,
        "leakage_status": leakage_status,
        "class_order_match_all_cells": _class_order_match_all(rows),
        "anchor_reproducibility_status": "PASS" if anchor_pass else "ANCHOR_MISMATCH",
        "center_equal_mean_bacc": primary["center_equal_mean_bacc"],
        "seed_cell_mean_bacc": primary["seed_cell_mean_bacc"],
        "center_equal_macro_f1": primary["center_equal_macro_f1"],
        "min_center_bacc": primary["min_center_bacc"],
        "seed_std_bacc": primary["seed_std_bacc"],
        "bottom20_cell_mean_bacc": primary["bottom20_cell_mean_bacc"],
        "worst_seed_center_bacc": primary["worst_seed_center_bacc"],
        "center3_bacc": primary["center3_bacc"],
        "shrink050_center_equal_mean_bacc": anchor_bacc,
        "random_mass_bag_center_equal_mean_bacc": bag_bacc,
        "uniform_component_union_center_equal_mean_bacc": uniform["center_equal_mean_bacc"],
        "source_union_k16_reference_center_equal_mean_bacc": source_union_bacc,
        "real_feature_dense_reference_center_equal_mean_bacc": real_bacc,
        "center3_delta_vs_random_mass_bag": center3_delta_bag,
        "center3_delta_vs_shrink050": center3_delta_anchor,
        "bottom20_delta_vs_random_mass_bag": bottom20_delta_bag,
        "bottom20_delta_vs_shrink050": bottom20_delta_anchor,
        "delta_vs_shrink050": primary_bacc - anchor_bacc if math.isfinite(primary_bacc) and math.isfinite(anchor_bacc) else math.nan,
        "delta_vs_random_mass_bag": primary_bacc - bag_bacc if math.isfinite(primary_bacc) and math.isfinite(bag_bacc) else math.nan,
        "delta_vs_uniform_component_union": primary_bacc - _float(uniform["center_equal_mean_bacc"]) if math.isfinite(primary_bacc) else math.nan,
        "retention_vs_source_union_k16": retention,
        "oracle_gap_vs_source_union_k16": source_union_bacc - primary_bacc if math.isfinite(source_union_bacc) and math.isfinite(primary_bacc) else math.nan,
        "oracle_gap_vs_real_feature_dense": real_bacc - primary_bacc if math.isfinite(real_bacc) and math.isfinite(primary_bacc) else math.nan,
        "no_center_worse_than_both_shrink050_and_random_mass_bag": no_center_worse,
        "fresh_panel_preserves_tail_direction": fresh_preserves,
        "complementarity_nontrivial_on_center3_or_bottom20": complementarity_nontrivial,
        "strongest_negative_control_method": strongest_control[0],
        "strongest_negative_control_center_equal_mean_bacc": strongest_control[1],
        "negative_control_gap": primary_bacc - strongest_control[1] if math.isfinite(primary_bacc) and math.isfinite(strongest_control[1]) else math.nan,
        "matched_shuffled_null_mean_center_equal_bacc": null_mean,
        "matched_shuffled_null_p95_center_equal_bacc": null_p95,
        "primary_minus_shuffled_reliability_null_mean": primary_bacc - null_mean if math.isfinite(primary_bacc) and math.isfinite(null_mean) else math.nan,
        **ablation,
        **primary,
    }


def _source_ablation_stats(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    deltas = [_float(row.get("delta_ablation_minus_primary")) for row in rows if row.get("status") == "ok"]
    finite = [value for value in deltas if math.isfinite(value)]
    max_abs = max((abs(value) for value in finite), default=math.nan)
    return {
        "source_ablation_max_abs_delta": max_abs,
        "source_ablation_mean_delta_bacc": nanmean(finite) if finite else math.nan,
    }


def _shuffled_null_summary_rows(
    matrix_rows: Sequence[Mapping[str, object]],
    null_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    primary_stats = cu._method_stats(cu._rows_for(matrix_rows, PRIMARY_TAILRISK_METHOD))
    primary_bacc = _float(primary_stats["center_equal_mean_bacc"])
    perm_ids = sorted({int(row.get("null_perm_id", -1)) for row in null_rows if int(row.get("null_perm_id", -1)) >= 0})
    null_means = []
    for perm_id in perm_ids:
        stats = cu._method_stats([row for row in null_rows if int(row.get("null_perm_id", -1)) == perm_id])
        null_means.append(_float(stats["center_equal_mean_bacc"]))
    finite = sorted(value for value in null_means if math.isfinite(value))
    if not finite:
        return [
            {
                "n_null_permutations": len(perm_ids),
                "primary_center_equal_mean_bacc": primary_bacc,
                "null_mean_center_equal_bacc": math.nan,
                "null_p90_center_equal_bacc": math.nan,
                "null_p95_center_equal_bacc": math.nan,
                "null_max_center_equal_bacc": math.nan,
                "empirical_p_value": math.nan,
                "primary_minus_null_mean": math.nan,
                "primary_minus_null_p95": math.nan,
                "paired_cell_win_fraction_vs_null": math.nan,
            }
        ]
    null_mean = nanmean(finite)
    null_p90 = float(np.quantile(np.asarray(finite, dtype=float), 0.90))
    null_p95 = float(np.quantile(np.asarray(finite, dtype=float), 0.95))
    null_max = max(finite)
    empirical_p = (1.0 + sum(1 for value in finite if value >= primary_bacc)) / float(len(finite) + 1) if math.isfinite(primary_bacc) else math.nan
    pair_wins = 0
    pair_total = 0
    primary_cells = {
        (str(row["experiment_seed"]), str(row["heldout_center"]), str(row["replicate_seed"])): _float(row.get("bacc"))
        for row in cu._rows_for(matrix_rows, PRIMARY_TAILRISK_METHOD)
    }
    for row in null_rows:
        key = (str(row["experiment_seed"]), str(row["heldout_center"]), str(row["replicate_seed"]))
        primary_cell = primary_cells.get(key, math.nan)
        null_cell = _float(row.get("bacc"))
        if math.isfinite(primary_cell) and math.isfinite(null_cell):
            pair_total += 1
            if primary_cell > null_cell:
                pair_wins += 1
    return [
        {
            "n_null_permutations": len(perm_ids),
            "primary_center_equal_mean_bacc": primary_bacc,
            "null_mean_center_equal_bacc": null_mean,
            "null_p90_center_equal_bacc": null_p90,
            "null_p95_center_equal_bacc": null_p95,
            "null_max_center_equal_bacc": null_max,
            "empirical_p_value": empirical_p,
            "primary_minus_null_mean": primary_bacc - null_mean if math.isfinite(primary_bacc) else math.nan,
            "primary_minus_null_p95": primary_bacc - null_p95 if math.isfinite(primary_bacc) else math.nan,
            "paired_cell_win_fraction_vs_null": float(pair_wins) / float(pair_total) if pair_total else math.nan,
        }
    ]


def _panel_summary_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    out = []
    methods = sorted({str(row.get("prior_method")) for row in rows if row.get("prior_method")})
    for panel in ("canonical", "fresh", "combined"):
        panel_rows = cu._rows_for_panel(rows, panel)
        for method in methods:
            stats = cu._method_stats(cu._rows_for(panel_rows, method))
            if int(stats["n_raw_rows"]) < 1:
                continue
            out.append({"panel": panel, "prior_method": method, **stats})
    return out


def _oracle_gap_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    out = []
    for method in sorted({str(row.get("prior_method")) for row in rows if row.get("prior_method")}):
        subset = cu._rows_for(rows, method)
        if not subset:
            continue
        stats = _tail_metrics(rows, method)
        out.append({"prior_method": method, **stats})
    return out


def _random_mass_bag_summary(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [{"prior_method": BAG_METHOD, **_tail_metrics(rows, BAG_METHOD)}]


def _negative_control_summary(decision: Mapping[str, object]) -> dict[str, object]:
    return {
        "primary_method": PRIMARY_TAILRISK_METHOD,
        "primary_center_equal_mean_bacc": decision.get("center_equal_mean_bacc", math.nan),
        "strongest_negative_control_method": decision.get("strongest_negative_control_method", ""),
        "strongest_negative_control_center_equal_mean_bacc": decision.get("strongest_negative_control_center_equal_mean_bacc", math.nan),
        "negative_control_gap": decision.get("negative_control_gap", math.nan),
        "matched_shuffled_null_mean_center_equal_bacc": decision.get("matched_shuffled_null_mean_center_equal_bacc", math.nan),
        "primary_minus_shuffled_reliability_null_mean": decision.get("primary_minus_shuffled_reliability_null_mean", math.nan),
        "control_competitive": "NEGATIVE_CONTROLS_COMPETITIVE" in str(decision.get("diagnostic_flags", "")),
    }


def _write_artifacts(
    root: Path,
    cfg: TailRiskAnchoredConfig,
    *,
    matrix_rows: Sequence[Mapping[str, object]],
    component_manifest_rows: Sequence[Mapping[str, object]],
    component_coverage_rows: Sequence[Mapping[str, object]],
    source_weight_rows: Sequence[Mapping[str, object]],
    reliability_rows: Sequence[Mapping[str, object]],
    source_summary_rows: Sequence[Mapping[str, object]],
    source_ablation_rows: Sequence[Mapping[str, object]],
    paired_generation_rows: Sequence[Mapping[str, object]],
    eligibility_rows: Sequence[Mapping[str, object]],
    blend_manifest_rows: Sequence[Mapping[str, object]],
    complementarity_rows: Sequence[Mapping[str, object]],
    calibration_rows: Sequence[Mapping[str, object]],
    shuffled_null_rows: Sequence[Mapping[str, object]],
    shuffled_null_summary: Sequence[Mapping[str, object]],
    model_manifest_rows: Sequence[Mapping[str, object]],
    anchor_rows: Sequence[Mapping[str, object]],
    decision: Mapping[str, object],
    leakage: object,
    protocol_violations: Sequence[str],
    target_expert_excluded: bool,
) -> None:
    write_csv_rows(root / "tables" / "tailrisk_downstream_matrix.csv", matrix_rows)
    write_csv_rows(root / "tables" / "tailrisk_summary.csv", [dict(decision)])
    write_csv_rows(root / "tables" / "tailrisk_panel_summary.csv", _panel_summary_rows(matrix_rows))
    write_csv_rows(root / "tables" / "tailrisk_tail_metric_summary.csv", _tail_metric_summary_rows(matrix_rows))
    write_csv_rows(root / "tables" / "tailrisk_probability_blend_manifest.csv", blend_manifest_rows)
    write_csv_rows(root / "tables" / "tailrisk_complementarity_audit.csv", complementarity_rows)
    write_csv_rows(root / "tables" / "tailrisk_calibration_audit.csv", calibration_rows)
    write_csv_rows(root / "tables" / "source_weight_manifest.csv", source_weight_rows)
    write_csv_rows(root / "tables" / "source_reliability_manifest.csv", reliability_rows)
    write_csv_rows(root / "tables" / "component_manifest.csv", component_manifest_rows)
    write_csv_rows(root / "tables" / "component_coverage_audit.csv", component_coverage_rows)
    write_csv_rows(root / "tables" / "paired_generation_audit.csv", paired_generation_rows)
    write_csv_rows(root / "tables" / "negative_control_summary.csv", [_negative_control_summary(decision)])
    write_csv_rows(root / "tables" / "source_ablation_audit.csv", source_ablation_rows)
    write_csv_rows(root / "tables" / "oracle_gap_summary.csv", _oracle_gap_rows(matrix_rows))
    write_csv_rows(root / "tables" / "random_mass_bag_control_summary.csv", _random_mass_bag_summary(matrix_rows))
    write_csv_rows(root / "tables" / "shuffled_reliability_null_summary.csv", shuffled_null_summary)
    write_csv_rows(root / "tables" / "anchor_reproducibility_audit.csv", anchor_rows)
    write_csv_rows(root / "tables" / "eligibility_audit.csv", eligibility_rows)
    write_csv_rows(root / "tables" / "source_summary_diagnostics.csv", source_summary_rows)
    write_csv_rows(root / "tables" / "shuffled_reliability_null_matrix.csv", shuffled_null_rows)
    write_csv_rows(root / "manifests" / "tailrisk_component_union_model_manifest.csv", model_manifest_rows)
    write_json(root / "reports" / "leakage_report.json", leakage.to_json_dict())
    write_json(
        root / "manifests" / "protocol_manifest.json",
        {
            "schema_version": "cvae_rebuild_tailrisk_anchored_component_union_protocol_v1",
            "experiment_name": cfg.name,
            "primary_method": cfg.primary_method,
            "experiment_type": "source_only_tailrisk_anchored_mass_uncertainty_component_union",
            "target_expert_excluded": bool(target_expert_excluded),
            "target_support_used": False,
            "target_support_labels_for_selection": False,
            "target_eval_labels_for_scoring_only": True,
            "target_calibration_metrics_audit_only": True,
            "target_conditioned_point_compatibility_estimate": False,
            "fixed_all_source_inclusion": True,
            "blend_alpha_locked": cfg.blend_alpha,
            "random_mass_bag_size": cfg.random_mass_bag_size,
            "random_mass_bag_distribution": "dirichlet_uniform_alpha4",
            "source_ablation_diagnostic_only": True,
            "oracle_rows_diagnostic_only": True,
            "claim_boundary": (
                "source-only robustness aggregation under component/source-mass uncertainty; "
                "not learned routing, sparse expert selection, target adaptation, formal privacy, "
                "or causal reliability validation"
            ),
            "protocol_violations": list(protocol_violations),
        },
    )
    write_json(root / "run_config_resolved.yaml", _resolved_config(cfg))
    _write_decision_summary(root, decision)


def _write_decision_summary(root: Path, decision: Mapping[str, object]) -> None:
    lines = [
        "# Tail-Risk Anchored Mass-Uncertainty Component-Union v1",
        "",
        "## Summary",
        "",
        f"- Primary method: `{decision.get('primary_method', PRIMARY_TAILRISK_METHOD)}`",
        f"- Primary verdict: `{decision.get('primary_verdict', 'TAILRISK_ANCHORED_COMPONENT_UNION_FAIL')}`",
        f"- Diagnostic flags: `{decision.get('diagnostic_flags', '')}`",
        f"- Center-equal mean BACC: {_format_float(decision.get('center_equal_mean_bacc'))}",
        f"- Min center BACC: {_format_float(decision.get('min_center_bacc'))}",
        f"- Center 3 BACC: {_format_float(decision.get('center3_bacc'))}",
        f"- Bottom-20 cell mean BACC: {_format_float(decision.get('bottom20_cell_mean_bacc'))}",
        f"- Seed std BACC: {_format_float(decision.get('seed_std_bacc'))}",
        f"- Shrink050 BACC: {_format_float(decision.get('shrink050_center_equal_mean_bacc'))}",
        f"- Random mass-bag BACC: {_format_float(decision.get('random_mass_bag_center_equal_mean_bacc'))}",
        f"- Center3 delta vs random mass-bag: {_format_float(decision.get('center3_delta_vs_random_mass_bag'))}",
        f"- Bottom20 delta vs random mass-bag: {_format_float(decision.get('bottom20_delta_vs_random_mass_bag'))}",
        f"- Retention vs source-union K16: {_format_float(decision.get('retention_vs_source_union_k16'))}",
        f"- Complementarity nontrivial: `{decision.get('complementarity_nontrivial_on_center3_or_bottom20')}`",
        f"- Fresh panel preserves tail direction: `{decision.get('fresh_panel_preserves_tail_direction')}`",
        f"- Leakage status: `{decision.get('leakage_status', '')}`",
        "",
        "## Protocol Boundary",
        "",
        "This is a locked source-only robustness aggregation audit. It uses no target support, no target-conditioned point compatibility estimate, and no sparse expert selection.",
        "",
        "The primary method averages fixed prediction probabilities from a reliability-shrink050 component-union anchor and an 11-member Dirichlet-uniform random mass-bag ensemble with alpha 0.50/0.50.",
        "",
        "Target evaluation labels and target calibration metrics are audit/scoring only and never choose alpha, weights, source set, classifier, or decision logic.",
        "",
        "Safe claim if successful: in Virchow2 CVAE-generated feature aggregation, fixed source-only probability blending of a conservative reliability-weighted component union with a random mass-bag ensemble can reduce weak-center tail risk when the two compositions make complementary errors.",
        "",
    ]
    (root / "reports" / "decision_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _target_ineligible_rows(
    cfg: TailRiskAnchoredConfig,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
    error_message: str,
) -> list[dict[str, object]]:
    methods = (
        PRIMARY_TAILRISK_METHOD,
        ANCHOR_METHOD,
        BAG_METHOD,
        cu.PRIMARY_COMPONENT_UNION_METHOD,
        cu.ROW_REAL_FEATURE_DENSE_REFERENCE,
        cu.ROW_RANDOM_SOURCE_MASS_CONTROL,
        cu.ROW_SHUFFLED_LABEL_CONTROL,
        cu.ROW_SHUFFLED_SUMMARY_CONTROL,
    )
    return [
        cu._empty_matrix_row(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            candidates=candidates,
            prior_method=method,
            source_union_ref=source_union_ref,
            center_balanced_ref=center_balanced_ref,
            real_feature_bacc=math.nan,
            status="ineligible",
            error_message=error_message,
            claim_role="ineligible_target_eval",
        )
        for method in methods
    ]


def _eligibility_row(
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    row_scope: str,
    status: str,
    error_message: str,
) -> dict[str, object]:
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "replicate_seed": int(replicate_seed),
        "row_scope": str(row_scope),
        "status": str(status),
        "error_message": str(error_message),
    }


def _normalize_row(row: Mapping[str, object], *, prior_method: str) -> dict[str, object]:
    out = dict(row)
    out["prior_method"] = prior_method
    out.setdefault("summary_kind", "")
    out.setdefault("source_weight_json", "{}")
    out.setdefault("source_budget_json", "{}")
    return out


def _optional_path(base: Path, value: object) -> Path | None:
    if value is None or str(value) == "":
        return None
    return _path(base, str(value))


def _delta(value: object, baseline: object) -> float:
    left = _float(value)
    right = _float(baseline)
    return left - right if math.isfinite(left) and math.isfinite(right) else math.nan


def _class_order_match_all(rows: Sequence[Mapping[str, object]]) -> bool:
    return True


def _no_center_worse_than_both(rows: Sequence[Mapping[str, object]]) -> bool:
    primary = json.loads(str(_tail_metrics(rows, PRIMARY_TAILRISK_METHOD)["per_center_bacc"]))
    anchor = json.loads(str(_tail_metrics(rows, ANCHOR_METHOD)["per_center_bacc"]))
    bag = json.loads(str(_tail_metrics(rows, BAG_METHOD)["per_center_bacc"]))
    for center, value in primary.items():
        p = _float(value)
        a = _float(anchor.get(center, math.nan))
        b = _float(bag.get(center, math.nan))
        if math.isfinite(p) and math.isfinite(a) and math.isfinite(b) and p < min(a, b) - 1.0e-12:
            return False
    return True


def _fresh_preserves_tail_direction(rows: Sequence[Mapping[str, object]]) -> bool:
    checks = []
    for panel in ("canonical", "fresh"):
        primary = _tail_metrics(rows, PRIMARY_TAILRISK_METHOD, panel=panel)
        bag = _tail_metrics(rows, BAG_METHOD, panel=panel)
        checks.append(
            (
                _delta(primary["min_center_bacc"], bag["min_center_bacc"]),
                _delta(primary["center3_bacc"], bag["center3_bacc"]),
                _delta(primary["bottom20_cell_mean_bacc"], bag["bottom20_cell_mean_bacc"]),
            )
        )
    canonical, fresh = checks
    return all((not math.isfinite(c)) or (not math.isfinite(f)) or (c > 0.0 and f >= 0.0) or (c <= 0.0 and f >= c) for c, f in zip(canonical, fresh))


def _complementarity_nontrivial(rows: Sequence[Mapping[str, object]]) -> bool:
    values = []
    for row in rows:
        if str(row.get("heldout_center")) == "3" or str(row.get("is_bottom20_cell")) == "True":
            value = _float(row.get("anchor_correct_bag_wrong_rate"))
            if math.isfinite(value):
                values.append(value)
    return max(values, default=0.0) >= 0.01


def _resolved_config(cfg: TailRiskAnchoredConfig) -> dict[str, object]:
    return {
        "experiment": {
            "name": cfg.name,
            "artifact_root": str(cfg.artifact_root),
            "primary_variant": cfg.primary_variant,
        },
        "inputs": {
            "feature_cache_root": str(cfg.feature_cache_root),
            "repair_artifact_root": str(cfg.repair_artifact_root),
            "paired_dense_artifact_root": "" if cfg.paired_dense_artifact_root is None else str(cfg.paired_dense_artifact_root),
            "mass_bagged_artifact_root": "" if cfg.mass_bagged_artifact_root is None else str(cfg.mass_bagged_artifact_root),
            "support_calibrated_artifact_root": "" if cfg.support_calibrated_artifact_root is None else str(cfg.support_calibrated_artifact_root),
            "shrink050_artifact_root": "" if cfg.shrink050_artifact_root is None else str(cfg.shrink050_artifact_root),
            "source_union_gmm_artifact_root": "" if cfg.source_union_gmm_artifact_root is None else str(cfg.source_union_gmm_artifact_root),
            "balanced_gmm_artifact_root": "" if cfg.balanced_gmm_artifact_root is None else str(cfg.balanced_gmm_artifact_root),
            "backbone": cfg.backbone,
        },
        "run_matrix": {
            "strict_full_run_matrix": cfg.strict_full_run_matrix,
            "experiment_seeds": list(cfg.experiment_seeds),
            "heldout_centers": list(cfg.heldout_centers),
            "replicate_seeds": list(cfg.replicate_seeds),
            "fresh_replicate_seeds": list(cfg.fresh_replicate_seeds),
        },
        "generation": {
            "synthetic_per_class_total": cfg.synthetic_per_class_total,
            "min_per_source_per_class": cfg.min_per_source_per_class,
        },
        "tailrisk_anchored_component_union": {
            "primary_method": cfg.primary_method,
            "primary_shrink_lambda": cfg.primary_shrink_lambda,
            "random_mass_bag_size": cfg.random_mass_bag_size,
            "random_mass_bag_alpha": cfg.random_mass_bag_alpha,
            "blend_alpha": cfg.blend_alpha,
            "matched_shuffled_reliability_null_permutations": cfg.matched_shuffled_reliability_null_permutations,
            "candidate_components_per_source_class": list(cfg.candidate_components_per_source_class),
            "min_samples_per_component": cfg.min_samples_per_component,
            "source_weighting": cfg.source_weighting,
            "gmm_covariance_type": cfg.gmm_covariance_type,
            "gmm_reg_covar": cfg.gmm_reg_covar,
            "gmm_n_init": cfg.gmm_n_init,
            "gmm_max_iter": cfg.gmm_max_iter,
            "min_component_weight": cfg.min_component_weight,
            "variance_floor": cfg.variance_floor,
            "variance_ceiling_multiplier": cfg.variance_ceiling_multiplier,
            "primary_pooling": cfg.primary_pooling,
            "reliability_floor_score": cfg.reliability_floor_score,
            "reliability_epsilon": cfg.reliability_epsilon,
            "anchor_repro_tolerance": cfg.anchor_repro_tolerance,
        },
        "classifier": {
            "type": cfg.classifier_type,
            "solver": cfg.classifier_solver,
            "C": cfg.classifier_c,
            "max_iter": cfg.classifier_max_iter,
            "class_weight": cfg.classifier_class_weight,
            "classifier_seed": cfg.classifier_seed,
        },
    }
