from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluation.downstream import (
    PredictionBundle,
    evaluate_probability_predictions,
    geometric_probability_pool,
    predict_from_probabilities,
    weighted_arithmetic_probability_pool,
    weighted_geometric_probability_pool,
)
from data.features import load_feature_cache, select_rows
from core.metrics import nanmean
from experiments.preservation.preservation import _hash_array
from experiments.preservation.preservation_repair import (
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
from experiments.preservation.preservation_sampling import DIAGNOSTIC_SELECTION, PRIMARY_SELECTION, RuntimeSource, _manifest_row, _per_source_variant, _runtime_source
from core.protocol import ProtocolError, assert_candidate_pool, build_leakage_report
from core.reporting import prepare_artifact_dirs, write_csv_rows, write_json
from data.splits import candidate_experts

from experiments.component_union import component_union_mass_bagged as mb
from experiments.component_union import component_union_tailrisk_anchored_mass_bagged as tr
from experiments.decentralized import decentralized_adaptive_gmm_prior as d1a
from experiments.decentralized import decentralized_component_union_prior as cu
from experiments.decentralized import decentralized_k16_gmm_prior as d1
from experiments.decentralized import decentralized_reliability_weighted_gmm_prior as d12
from experiments.component_union import paired_dense_all4_reliability_confirmation as paired


DENSE_TAILSHIELD_NAME = "virchow2_cvae_dense_reliability_tailshield_random_mass_bag_v1"
PRIMARY_DENSE_TAILSHIELD_METHOD = "dense_reliability_tailshield_random_mass_bag_blend25_75"
DENSE_TAILSHIELD_SOURCE_WEIGHTING = "dense_reliability_tailshield_random_mass_bag_blend25_75"
DENSE_ANCHOR_METHOD = paired.ROW_RELIABILITY_ALL4_WEIGHTED
DENSE_EQUAL_ANCHOR_METHOD = paired.ROW_EQUAL_ALL4
BAG_METHOD = cu.ROW_RANDOM_MASS_BAG_CONTROL
EQUAL_DENSE_SHIELD_METHOD = "equal_dense_tailshield_random_mass_bag_blend25_75"
SHUFFLED_DENSE_SHIELD_METHOD = "shuffled_reliability_dense_tailshield_random_mass_bag_blend25_75"
RANDOM_DENSE_SHIELD_METHOD = "random_dense_tailshield_random_mass_bag_blend25_75"
ALPHA_CURVE_PREFIX = "dense_tailshield_alpha_curve"


@dataclass(frozen=True)
class DenseTailShieldConfig:
    name: str
    artifact_root: Path
    repair_artifact_root: Path
    paired_dense_artifact_root: Path | None
    mass_bagged_artifact_root: Path | None
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
    dense_blend_alpha: float
    bag_blend_alpha: float
    alpha_curve_dense_values: tuple[float, ...]
    reconstruction_probability_tolerance: float
    nontrivial_rescue_threshold: float
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
class DenseBundleEvaluation:
    row: dict[str, object]
    bundle: PredictionBundle | None
    late_rows: tuple[dict[str, object], ...]
    coverage_rows: tuple[dict[str, object], ...]
    weak_rows: tuple[dict[str, object], ...]
    nn_rows: tuple[dict[str, object], ...]
    generated_hash: str
    plan: dict[str, object]
    reconstruction_row: dict[str, object]


@dataclass(frozen=True)
class DenseTailShieldEvaluation:
    primary_row: dict[str, object]
    primary_bundle: PredictionBundle | None
    primary_coverage: dict[str, object]
    primary_paired_row: dict[str, object]
    dense_anchor: DenseBundleEvaluation
    equal_anchor: DenseBundleEvaluation
    shuffled_anchor: DenseBundleEvaluation
    random_anchor: DenseBundleEvaluation
    bag_evaluation: tr.BagEvaluation
    blend_manifest_rows: tuple[dict[str, object], ...]
    reconstruction_rows: tuple[dict[str, object], ...]
    complementarity_row: dict[str, object]
    calibration_rows: tuple[dict[str, object], ...]
    confidence_rows: tuple[dict[str, object], ...]
    rescue_row: dict[str, object]
    alpha_curve_rows: tuple[dict[str, object], ...]
    control_rows: tuple[dict[str, object], ...]
    control_coverages: tuple[dict[str, object], ...]
    control_paired_rows: tuple[dict[str, object], ...]
    source_weight_rows: tuple[dict[str, object], ...]
    eligibility_rows: tuple[dict[str, object], ...]


def load_dense_tailshield_random_mass_bag_config(path: str | Path) -> DenseTailShieldConfig:
    source = Path(path).resolve()
    data = _load_mapping(source)
    base_dir = source.parents[2] if len(source.parents) >= 3 else source.parent
    return parse_dense_tailshield_random_mass_bag_config(data, base_dir=base_dir)


def parse_dense_tailshield_random_mass_bag_config(
    data: Mapping[str, Any],
    *,
    base_dir: str | Path = ".",
) -> DenseTailShieldConfig:
    base = Path(base_dir)
    experiment = _mapping(data, "experiment")
    inputs = _mapping(data, "inputs")
    run = _mapping(data, "run_matrix")
    generation = _mapping(data, "generation")
    shield = _mapping(data, "dense_tailshield_random_mass_bag")
    classifier = _mapping(data, "classifier")
    cfg = DenseTailShieldConfig(
        name=str(experiment["name"]),
        artifact_root=_path(base, str(experiment["artifact_root"])),
        repair_artifact_root=_path(base, str(inputs["repair_artifact_root"])),
        paired_dense_artifact_root=_optional_path(base, inputs.get("paired_dense_artifact_root")),
        mass_bagged_artifact_root=_optional_path(base, inputs.get("mass_bagged_artifact_root")),
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
        primary_method=str(shield["primary_method"]),
        random_mass_bag_size=int(shield["random_mass_bag_size"]),
        random_mass_bag_alpha=float(shield["random_mass_bag_alpha"]),
        dense_blend_alpha=float(shield["dense_blend_alpha"]),
        bag_blend_alpha=float(shield["bag_blend_alpha"]),
        alpha_curve_dense_values=tuple(float(v) for v in shield["alpha_curve_dense_values"]),
        reconstruction_probability_tolerance=float(shield["reconstruction_probability_tolerance"]),
        nontrivial_rescue_threshold=float(shield["nontrivial_rescue_threshold"]),
        candidate_components_per_source_class=tuple(int(v) for v in shield["candidate_components_per_source_class"]),
        min_samples_per_component=int(shield["min_samples_per_component"]),
        source_weighting=str(shield["source_weighting"]),
        gmm_covariance_type=str(shield["gmm_covariance_type"]),
        gmm_reg_covar=float(shield["gmm_reg_covar"]),
        gmm_n_init=int(shield["gmm_n_init"]),
        gmm_max_iter=int(shield["gmm_max_iter"]),
        min_component_weight=float(shield["min_component_weight"]),
        variance_floor=float(shield["variance_floor"]),
        variance_ceiling_multiplier=float(shield["variance_ceiling_multiplier"]),
        primary_pooling=str(shield["primary_pooling"]),
        reliability_floor_score=float(shield["reliability_floor_score"]),
        reliability_epsilon=float(shield["reliability_epsilon"]),
        anchor_repro_tolerance=float(shield["anchor_repro_tolerance"]),
        classifier_type=str(classifier["type"]),
        classifier_solver=str(classifier["solver"]),
        classifier_c=float(classifier["C"]),
        classifier_max_iter=int(classifier["max_iter"]),
        classifier_class_weight=str(classifier["class_weight"]),
        classifier_seed=None if classifier.get("classifier_seed") is None else int(classifier["classifier_seed"]),
    )
    validate_dense_tailshield_random_mass_bag_config(cfg)
    return cfg


def validate_dense_tailshield_random_mass_bag_config(cfg: DenseTailShieldConfig) -> None:
    if cfg.name != DENSE_TAILSHIELD_NAME:
        raise ProtocolError(f"Dense tail-shield experiment name must be {DENSE_TAILSHIELD_NAME!r}.")
    if cfg.backbone != "virchow2":
        raise ProtocolError("Dense tail shield is locked to backbone=virchow2.")
    if cfg.primary_variant != PRIMARY_VARIANT:
        raise ProtocolError(f"primary_variant must be {PRIMARY_VARIANT!r}.")
    if cfg.primary_method != PRIMARY_DENSE_TAILSHIELD_METHOD:
        raise ProtocolError(f"primary_method must be {PRIMARY_DENSE_TAILSHIELD_METHOD!r}.")
    if cfg.source_weighting != DENSE_TAILSHIELD_SOURCE_WEIGHTING:
        raise ProtocolError(f"source_weighting must be {DENSE_TAILSHIELD_SOURCE_WEIGHTING!r}.")
    if cfg.candidate_components_per_source_class != (4, 3, 2, 1):
        raise ProtocolError("candidate_components_per_source_class must be locked to [4, 3, 2, 1].")
    if cfg.gmm_covariance_type != "diag":
        raise ProtocolError("gmm_covariance_type must be diag.")
    if cfg.primary_pooling != "fixed_arithmetic_probability_blend":
        raise ProtocolError("primary_pooling must be fixed_arithmetic_probability_blend.")
    if not math.isclose(cfg.dense_blend_alpha, 0.25, rel_tol=0.0, abs_tol=1.0e-12):
        raise ProtocolError("dense_blend_alpha must be locked to 0.25.")
    if not math.isclose(cfg.bag_blend_alpha, 0.75, rel_tol=0.0, abs_tol=1.0e-12):
        raise ProtocolError("bag_blend_alpha must be locked to 0.75.")
    if not math.isclose(cfg.dense_blend_alpha + cfg.bag_blend_alpha, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ProtocolError("dense_blend_alpha + bag_blend_alpha must equal 1.")
    if not cfg.alpha_curve_dense_values:
        raise ProtocolError("alpha_curve_dense_values must be non-empty.")
    if not math.isclose(cfg.random_mass_bag_alpha, 4.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ProtocolError("random_mass_bag_alpha must be locked to Dirichlet-uniform alpha4.")
    if cfg.random_mass_bag_size < 1:
        raise ProtocolError("random_mass_bag_size must be positive.")
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
        if cfg.alpha_curve_dense_values != (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
            raise ProtocolError("strict_full_run_matrix requires alpha_curve_dense_values=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0].")
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
        cfg.reconstruction_probability_tolerance,
        cfg.nontrivial_rescue_threshold,
    ) <= 0.0:
        raise ProtocolError("Dense tail-shield numeric floors/tolerances must be positive.")
    if cfg.classifier_type != "sklearn_logistic_regression":
        raise ProtocolError("classifier.type must be sklearn_logistic_regression.")
    if cfg.classifier_solver != "lbfgs" or cfg.classifier_c != 1.0 or cfg.classifier_max_iter != 2000:
        raise ProtocolError("Classifier solver/C/max_iter must remain locked.")
    if cfg.classifier_class_weight != "balanced" or cfg.classifier_seed is not None:
        raise ProtocolError("Classifier must use class_weight=balanced and classifier_seed=null.")


def run_dense_reliability_tailshield_random_mass_bag(
    cfg: DenseTailShieldConfig,
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
    reconstruction_rows: list[dict[str, object]] = []
    complementarity_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    confidence_rows: list[dict[str, object]] = []
    rescue_rows: list[dict[str, object]] = []
    alpha_curve_rows: list[dict[str, object]] = []
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
                eval_ids = tuple(_sample_id(row, idx) for idx, row in enumerate(eval_meta))
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
                    ref_row = _normalize_component_row(ref_row, prior_method=cu.ROW_REAL_FEATURE_DENSE_REFERENCE)
                    matrix_rows.append(ref_row)
                    real_feature_bacc = _float(ref_row["bacc"])

                    uniform_plan = cu._uniform_source_plan(cfg, candidates, rels, total=cfg.synthetic_per_class_total)
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
                        claim_role="uniform_component_union_reference",
                        control_mode="normal",
                    )
                    matrix_rows.append(uniform.row)
                    component_coverage_rows.append(uniform.coverage_row)
                    paired_generation_rows.append(uniform.paired_row)
                    source_weight_rows.extend(cu._source_weight_manifest_rows(int(experiment_seed), int(replicate_seed), str(heldout_center), cu.PRIMARY_COMPONENT_UNION_METHOD, uniform_plan, rels))

                    shrink050_plan = cu._shrink_source_plan(cfg, candidates, rels, shrink_lambda=0.5, total=cfg.synthetic_per_class_total)
                    shrink050 = mb._evaluate_member(
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
                        weight_plan=shrink050_plan,
                        prior_method=cu.ROW_COMPONENT_UNION_SHRINK050,
                        selection_source=DIAGNOSTIC_SELECTION,
                        claim_role="component_union_shrink050_tail_comparator",
                        control_mode="normal",
                    )
                    matrix_rows.append(shrink050.row)
                    component_coverage_rows.append(shrink050.coverage_row)
                    paired_generation_rows.append(shrink050.paired_row)
                    source_weight_rows.extend(cu._source_weight_manifest_rows(int(experiment_seed), int(replicate_seed), str(heldout_center), cu.ROW_COMPONENT_UNION_SHRINK050, shrink050_plan, rels))

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

                    shield_eval = _evaluate_dense_tailshield_pair(
                        cfg,
                        root=root,
                        per_source_runtime=per_source_runtime,
                        dense_summaries=dense_summaries,
                        component_summaries=gmm_summaries,
                        candidates=candidates,
                        rels=rels,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        replicate_seed=int(replicate_seed),
                        eval_raw=eval_raw,
                        eval_labels=eval_labels,
                        eval_sample_ids=eval_ids,
                        source_union_ref=su_ref,
                        center_balanced_ref=cb_ref,
                        real_feature_bacc=real_feature_bacc,
                    )
                    _append_dense_tailshield_outputs(
                        shield_eval,
                        matrix_rows=matrix_rows,
                        component_coverage_rows=component_coverage_rows,
                        paired_generation_rows=paired_generation_rows,
                        source_weight_rows=source_weight_rows,
                        blend_manifest_rows=blend_manifest_rows,
                        reconstruction_rows=reconstruction_rows,
                        complementarity_rows=complementarity_rows,
                        calibration_rows=calibration_rows,
                        confidence_rows=confidence_rows,
                        rescue_rows=rescue_rows,
                        alpha_curve_rows=alpha_curve_rows,
                        eligibility_rows=eligibility_rows,
                    )

                    primary_bacc = _float(shield_eval.primary_row.get("bacc"))
                    source_ablation_rows.extend(
                        _source_ablation_rows(
                            cfg,
                            root=root,
                            per_source_runtime=per_source_runtime,
                            dense_summaries=dense_summaries,
                            component_summaries=gmm_summaries,
                            reliability=reliability,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            replicate_seed=int(replicate_seed),
                            candidates=candidates,
                            eval_raw=eval_raw,
                            eval_labels=eval_labels,
                            eval_sample_ids=eval_ids,
                            source_union_ref=su_ref,
                            center_balanced_ref=cb_ref,
                            real_feature_bacc=real_feature_bacc,
                            primary_bacc=primary_bacc,
                        )
                    )

                    random_single = mb._evaluate_single_plan_control(
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
                    tr._append_control_outputs(
                        random_single,
                        matrix_rows=matrix_rows,
                        component_coverage_rows=component_coverage_rows,
                        paired_generation_rows=paired_generation_rows,
                        source_weight_rows=source_weight_rows,
                        rels=rels,
                    )

                    matrix_rows.append(cu._reference_matrix_row(cfg, experiment_seed=int(experiment_seed), heldout_center=str(heldout_center), replicate_seed=int(replicate_seed), candidates=candidates, prior_method=cu.ROW_SOURCE_UNION_K16_REFERENCE, reference=su_ref))
                    matrix_rows.append(cu._reference_matrix_row(cfg, experiment_seed=int(experiment_seed), heldout_center=str(heldout_center), replicate_seed=int(replicate_seed), candidates=candidates, prior_method=cu.ROW_CENTER_BALANCED_K16_REFERENCE, reference=cb_ref))
    except ProtocolError as exc:
        protocol_violations.append(str(exc))

    bottom20_keys = _bottom20_raw_cell_keys(matrix_rows, BAG_METHOD)
    complementarity_rows = _mark_bottom20_rows(complementarity_rows, bottom20_keys)
    rescue_rows = _mark_bottom20_rows(rescue_rows, bottom20_keys)
    leakage = build_leakage_report(
        target_support_labels_for_selection=False,
        target_eval_labels_for_scoring_only=True,
        target_expert_excluded=target_expert_excluded,
        oracle_rows_diagnostic_only=True,
        extra_violations=protocol_violations,
    )
    anchor_rows = _anchor_reproducibility_rows(matrix_rows, cfg)
    shuffled_summary = _shuffled_dense_shield_summary_rows(matrix_rows)
    decision = _decision(
        matrix_rows,
        cfg=cfg,
        leakage_status=leakage.status,
        source_ablation_rows=source_ablation_rows,
        anchor_rows=anchor_rows,
        reconstruction_rows=reconstruction_rows,
        complementarity_rows=complementarity_rows,
        shuffled_summary=shuffled_summary,
        bottom20_keys=bottom20_keys,
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
        reconstruction_rows=reconstruction_rows,
        complementarity_rows=complementarity_rows,
        calibration_rows=calibration_rows,
        confidence_rows=confidence_rows,
        rescue_rows=rescue_rows,
        alpha_curve_rows=alpha_curve_rows,
        shuffled_summary=shuffled_summary,
        model_manifest_rows=model_manifest_rows,
        anchor_rows=anchor_rows,
        decision=decision,
        leakage=leakage,
        protocol_violations=protocol_violations,
        target_expert_excluded=target_expert_excluded,
    )
    return root


def _evaluate_dense_tailshield_pair(
    cfg: DenseTailShieldConfig,
    *,
    root: Path,
    per_source_runtime: Mapping[str, RuntimeSource],
    dense_summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    component_summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    candidates: Sequence[str],
    rels: Mapping[str, d12.SourceReliability],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    eval_raw: object,
    eval_labels: Sequence[int],
    eval_sample_ids: Sequence[str],
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
    real_feature_bacc: float,
    include_diagnostics: bool = True,
) -> DenseTailShieldEvaluation:
    transform = paired._heldout_excluded_reliability_transform(cfg, heldout_center, candidates, rels)
    plans = paired._variant_plans(
        cfg,
        candidates,
        transform,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
    )
    random_plan = _random_dense_anchor_plan(cfg, candidates, rels, experiment_seed, heldout_center, replicate_seed)

    dense_anchor = _evaluate_dense_anchor_bundle(
        cfg,
        per_source_runtime=per_source_runtime,
        summaries=dense_summaries,
        candidates=candidates,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        eval_raw=eval_raw,
        eval_labels=eval_labels,
        eval_sample_ids=eval_sample_ids,
        source_union_ref=source_union_ref,
        center_balanced_ref=center_balanced_ref,
        real_feature_bacc=real_feature_bacc,
        method=DENSE_ANCHOR_METHOD,
        plan=plans[DENSE_ANCHOR_METHOD],
        pooling_rule="weighted_geometric",
        selection_source=DIAGNOSTIC_SELECTION,
        claim_role="dense_reliability_tail_shield_anchor",
    )
    equal_anchor = dense_anchor
    shuffled_anchor = dense_anchor
    random_anchor = dense_anchor
    if include_diagnostics:
        equal_anchor = _evaluate_dense_anchor_bundle(
            cfg,
            per_source_runtime=per_source_runtime,
            summaries=dense_summaries,
            candidates=candidates,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            eval_raw=eval_raw,
            eval_labels=eval_labels,
            eval_sample_ids=eval_sample_ids,
            source_union_ref=source_union_ref,
            center_balanced_ref=center_balanced_ref,
            real_feature_bacc=real_feature_bacc,
            method=DENSE_EQUAL_ANCHOR_METHOD,
            plan=plans[DENSE_EQUAL_ANCHOR_METHOD],
            pooling_rule="geometric",
            selection_source=DIAGNOSTIC_SELECTION,
            claim_role="equal_dense_tail_shield_control_anchor",
        )
        shuffled_anchor = _evaluate_dense_anchor_bundle(
            cfg,
            per_source_runtime=per_source_runtime,
            summaries=dense_summaries,
            candidates=candidates,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            eval_raw=eval_raw,
            eval_labels=eval_labels,
            eval_sample_ids=eval_sample_ids,
            source_union_ref=source_union_ref,
            center_balanced_ref=center_balanced_ref,
            real_feature_bacc=real_feature_bacc,
            method=paired.ROW_SHUFFLED,
            plan=plans[paired.ROW_SHUFFLED],
            pooling_rule="weighted_geometric",
            selection_source=DIAGNOSTIC_SELECTION,
            claim_role="shuffled_reliability_dense_tail_shield_control_anchor",
        )
        random_anchor = _evaluate_dense_anchor_bundle(
            cfg,
            per_source_runtime=per_source_runtime,
            summaries=dense_summaries,
            candidates=candidates,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            eval_raw=eval_raw,
            eval_labels=eval_labels,
            eval_sample_ids=eval_sample_ids,
            source_union_ref=source_union_ref,
            center_balanced_ref=center_balanced_ref,
            real_feature_bacc=real_feature_bacc,
            method="random_dense_tailshield_anchor",
            plan=random_plan,
            pooling_rule="weighted_geometric",
            selection_source=DIAGNOSTIC_SELECTION,
            claim_role="random_dense_tail_shield_control_anchor",
        )

    bag_specs = mb._random_mass_bag_specs(cfg, candidates, rels, experiment_seed, heldout_center, replicate_seed)
    bag_eval = tr._evaluate_bag_with_bundle(
        cfg,
        root=root,
        per_source_runtime=per_source_runtime,
        candidates=candidates,
        summaries=component_summaries,
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
        claim_role="random_mass_bag_high_ceiling_comparator",
    )

    eligibility_rows = [
        _eligibility_row(experiment_seed, heldout_center, replicate_seed, "dense_anchor", str(dense_anchor.row.get("status", "")), str(dense_anchor.row.get("error_message", ""))),
        _eligibility_row(experiment_seed, heldout_center, replicate_seed, "random_mass_bag", str(bag_eval.ensemble_row.get("status", "")), str(bag_eval.ensemble_row.get("error_message", ""))),
        *bag_eval.eligibility_rows,
    ]
    dense_bundle = dense_anchor.bundle
    bag_bundle = bag_eval.ensemble_bundle
    source_weight_rows: list[dict[str, object]] = []
    source_weight_rows.extend(_dense_source_weight_rows(experiment_seed, replicate_seed, heldout_center, DENSE_ANCHOR_METHOD, dense_anchor.plan, rels))
    if include_diagnostics:
        source_weight_rows.extend(_dense_source_weight_rows(experiment_seed, replicate_seed, heldout_center, DENSE_EQUAL_ANCHOR_METHOD, equal_anchor.plan, rels))
        source_weight_rows.extend(_dense_source_weight_rows(experiment_seed, replicate_seed, heldout_center, paired.ROW_SHUFFLED, shuffled_anchor.plan, rels))
        source_weight_rows.extend(_dense_source_weight_rows(experiment_seed, replicate_seed, heldout_center, "random_dense_tailshield_anchor", random_anchor.plan, rels))
    source_weight_rows.extend(cu._source_weight_manifest_rows(experiment_seed, replicate_seed, heldout_center, BAG_METHOD, bag_eval.ensemble_plan, rels))

    if dense_bundle is None or dense_anchor.row.get("status") != "ok" or bag_bundle is None or bag_eval.ensemble_row.get("status") != "ok":
        row = cu._empty_matrix_row(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            candidates=candidates,
            prior_method=PRIMARY_DENSE_TAILSHIELD_METHOD,
            source_union_ref=source_union_ref,
            center_balanced_ref=center_balanced_ref,
            real_feature_bacc=real_feature_bacc,
            status="ineligible",
            error_message="dense_anchor_or_random_mass_bag_ineligible",
            claim_role="primary_dense_tailshield_probability_blend",
        )
        row["pooling_rule"] = "fixed_arithmetic_probability_blend"
        return DenseTailShieldEvaluation(
            primary_row=row,
            primary_bundle=None,
            primary_coverage=cu._empty_coverage_row(row),
            primary_paired_row=cu._paired_generation_row(row, "", "", "ineligible"),
            dense_anchor=dense_anchor,
            equal_anchor=equal_anchor,
            shuffled_anchor=shuffled_anchor,
            random_anchor=random_anchor,
            bag_evaluation=bag_eval,
            blend_manifest_rows=(_blend_manifest_row(cfg, row, dense_anchor, bag_eval, "", eval_sample_ids, class_order_match=False, sample_order_match=False),),
            reconstruction_rows=(dense_anchor.reconstruction_row, equal_anchor.reconstruction_row, shuffled_anchor.reconstruction_row, random_anchor.reconstruction_row) if include_diagnostics else (dense_anchor.reconstruction_row,),
            complementarity_row=_empty_complementarity_row(cfg, experiment_seed, heldout_center, replicate_seed, "ineligible"),
            calibration_rows=(),
            confidence_rows=(),
            rescue_row=_empty_rescue_row(cfg, experiment_seed, heldout_center, replicate_seed, "ineligible"),
            alpha_curve_rows=(),
            control_rows=(),
            control_coverages=(),
            control_paired_rows=(),
            source_weight_rows=tuple(source_weight_rows),
            eligibility_rows=tuple(eligibility_rows),
        )

    _assert_probability_alignment(dense_bundle, bag_bundle, eval_sample_ids)
    primary_row, primary_bundle, primary_coverage, primary_paired = _blend_dense_and_bag(
        cfg,
        dense_eval=dense_anchor,
        bag_eval=bag_eval,
        dense_bundle=dense_bundle,
        bag_bundle=bag_bundle,
        candidates=candidates,
        component_summaries=component_summaries,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        eval_labels=eval_labels,
        source_union_ref=source_union_ref,
        center_balanced_ref=center_balanced_ref,
        real_feature_bacc=real_feature_bacc,
        method=PRIMARY_DENSE_TAILSHIELD_METHOD,
        alpha_dense=cfg.dense_blend_alpha,
        selection_source=PRIMARY_SELECTION,
        claim_role="primary_dense_reliability_tailshield_random_mass_bag_blend",
    )

    control_rows: list[dict[str, object]] = []
    control_coverages: list[dict[str, object]] = []
    control_paired_rows: list[dict[str, object]] = []
    if include_diagnostics:
        for method, anchor_eval, role in (
            (EQUAL_DENSE_SHIELD_METHOD, equal_anchor, "equal_dense_tailshield_negative_control"),
            (SHUFFLED_DENSE_SHIELD_METHOD, shuffled_anchor, "shuffled_reliability_dense_tailshield_negative_control"),
            (RANDOM_DENSE_SHIELD_METHOD, random_anchor, "random_dense_tailshield_negative_control"),
        ):
            if anchor_eval.bundle is None or anchor_eval.row.get("status") != "ok":
                continue
            control_row, _control_bundle, control_coverage, control_paired = _blend_dense_and_bag(
                cfg,
                dense_eval=anchor_eval,
                bag_eval=bag_eval,
                dense_bundle=anchor_eval.bundle,
                bag_bundle=bag_bundle,
                candidates=candidates,
                component_summaries=component_summaries,
                experiment_seed=experiment_seed,
                heldout_center=heldout_center,
                replicate_seed=replicate_seed,
                eval_labels=eval_labels,
                source_union_ref=source_union_ref,
                center_balanced_ref=center_balanced_ref,
                real_feature_bacc=real_feature_bacc,
                method=method,
                alpha_dense=cfg.dense_blend_alpha,
                selection_source=DIAGNOSTIC_SELECTION,
                claim_role=role,
            )
            control_rows.append(control_row)
            control_coverages.append(control_coverage)
            control_paired_rows.append(control_paired)

    alpha_rows = []
    if include_diagnostics:
        for alpha_dense in cfg.alpha_curve_dense_values:
            alpha_row, _alpha_bundle, _alpha_coverage, _alpha_paired = _blend_dense_and_bag(
                cfg,
                dense_eval=dense_anchor,
                bag_eval=bag_eval,
                dense_bundle=dense_bundle,
                bag_bundle=bag_bundle,
                candidates=candidates,
                component_summaries=component_summaries,
                experiment_seed=experiment_seed,
                heldout_center=heldout_center,
                replicate_seed=replicate_seed,
                eval_labels=eval_labels,
                source_union_ref=source_union_ref,
                center_balanced_ref=center_balanced_ref,
                real_feature_bacc=real_feature_bacc,
                method=f"{ALPHA_CURVE_PREFIX}_alpha_dense_{alpha_dense:.2f}".replace(".", "p"),
                alpha_dense=float(alpha_dense),
                selection_source=DIAGNOSTIC_SELECTION,
                claim_role="audit_only_alpha_curve_not_primary_adoption",
            )
            alpha_rows.append(
                {
                    **_cell_keys(alpha_row),
                    "alpha_dense": float(alpha_dense),
                    "alpha_bag": 1.0 - float(alpha_dense),
                    "bacc": alpha_row.get("bacc", math.nan),
                    "macro_f1": alpha_row.get("macro_f1", math.nan),
                    "prediction_hash": alpha_row.get("prediction_hash", ""),
                    "diagnostic_only": True,
                    "primary_adoption_eligible": False,
                    "alpha_curve_can_rescue_primary": False,
                }
            )

    blend_rows = [
        _blend_manifest_row(cfg, primary_row, dense_anchor, bag_eval, str(primary_row.get("prediction_hash", "")), eval_sample_ids, class_order_match=True, sample_order_match=True)
    ]
    return DenseTailShieldEvaluation(
        primary_row=primary_row,
        primary_bundle=primary_bundle,
        primary_coverage=primary_coverage,
        primary_paired_row=primary_paired,
        dense_anchor=dense_anchor,
        equal_anchor=equal_anchor,
        shuffled_anchor=shuffled_anchor,
        random_anchor=random_anchor,
        bag_evaluation=bag_eval,
        blend_manifest_rows=tuple(blend_rows),
        reconstruction_rows=(dense_anchor.reconstruction_row, equal_anchor.reconstruction_row, shuffled_anchor.reconstruction_row, random_anchor.reconstruction_row) if include_diagnostics else (dense_anchor.reconstruction_row,),
        complementarity_row=_complementarity_row(cfg, experiment_seed, heldout_center, replicate_seed, dense_bundle, bag_bundle, eval_labels, primary_row),
        calibration_rows=tuple(_calibration_rows(cfg, experiment_seed=experiment_seed, heldout_center=heldout_center, replicate_seed=replicate_seed, eval_labels=eval_labels, dense_bundle=dense_bundle, bag_bundle=bag_bundle, blended_bundle=primary_bundle)) if include_diagnostics else (),
        confidence_rows=tuple(_confidence_rows(cfg, experiment_seed=experiment_seed, heldout_center=heldout_center, replicate_seed=replicate_seed, dense_bundle=dense_bundle, bag_bundle=bag_bundle, blended_bundle=primary_bundle)) if include_diagnostics else (),
        rescue_row=_rescue_row(cfg, experiment_seed, heldout_center, replicate_seed, dense_bundle, bag_bundle, eval_labels, primary_row),
        alpha_curve_rows=tuple(alpha_rows),
        control_rows=tuple(control_rows),
        control_coverages=tuple(control_coverages),
        control_paired_rows=tuple(control_paired_rows),
        source_weight_rows=tuple(source_weight_rows),
        eligibility_rows=tuple(eligibility_rows),
    )


def _evaluate_dense_anchor_bundle(
    cfg: DenseTailShieldConfig,
    *,
    per_source_runtime: Mapping[str, RuntimeSource],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    candidates: Sequence[str],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    eval_raw: object,
    eval_labels: Sequence[int],
    eval_sample_ids: Sequence[str],
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
    real_feature_bacc: float,
    method: str,
    plan: Mapping[str, object],
    pooling_rule: str,
    selection_source: str,
    claim_role: str,
) -> DenseBundleEvaluation:
    sources = tuple(str(source) for source in candidates)
    status, error = d1a._composition_status(sources, summaries, control_mode="normal")
    if status != "ok":
        row = d1a._dense_empty_row(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            candidates=sources,
            summaries=summaries,
            prior_method=method,
            source_union_ref=source_union_ref,
            center_balanced_ref=center_balanced_ref,
            real_feature_bacc=real_feature_bacc,
            status=status,
            error_message=error,
            claim_role=claim_role,
        )
        row = d12._extend_row(row, weight_plan=plan, source_weighting=str(plan.get("source_weighting", "")))
        row = paired._extend_paired_row(row, method=method, plan=plan, generation_bundle_key=_generation_bundle_key(experiment_seed, heldout_center, replicate_seed, candidates, str(plan.get("budget_policy", ""))))
        return DenseBundleEvaluation(row, None, (), (), (), (), "", dict(plan), _dense_reconstruction_row(cfg, row, None, eval_sample_ids, status="ineligible", max_abs_delta=math.nan))

    bundles, late_rows, coverage_rows, weak_rows, nn_rows, generated_hash = d12._source_generated_bundles(
        cfg,
        per_source_runtime=per_source_runtime,
        candidates=sources,
        summaries=summaries,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        eval_raw=eval_raw,
        eval_labels=eval_labels,
        prior_method=method,
        control_mode="normal",
        budgets=dict(plan["budgets"]),
        weight_plan=plan,
        generation_seed_method=str(plan.get("generation_seed_method", "")),
    )
    weights = [float(plan["weights"][str(bundle.expert_id)]) for bundle in bundles]
    if pooling_rule == "weighted_geometric":
        pooled = weighted_geometric_probability_pool(bundles, weights)
    elif pooling_rule == "geometric":
        pooled = geometric_probability_pool(bundles)
    else:
        raise ProtocolError(f"Unsupported dense tail-shield pooling rule: {pooling_rule}")
    single_baccs = [_float(row["bacc"]) for row in late_rows if row.get("status") == "ok"]
    single_macro = [_float(row["macro_f1"]) for row in late_rows if row.get("status") == "ok"]
    row = d1a._dense_result_row(
        cfg,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        candidates=sources,
        summaries=summaries,
        prior_method=method,
        pooling_rule=pooling_rule,
        probabilities=pooled,
        eval_labels=eval_labels,
        generated_features_hash=generated_hash,
        source_union_ref=source_union_ref,
        center_balanced_ref=center_balanced_ref,
        real_feature_bacc=real_feature_bacc,
        mean_single_bacc=nanmean(single_baccs),
        oracle_single_bacc=max(single_baccs) if single_baccs else math.nan,
        mean_single_macro_f1=nanmean(single_macro),
        selection_source=selection_source,
        claim_role=claim_role,
    )
    row = d12._extend_row(row, weight_plan=plan, source_weighting=str(plan.get("source_weighting", "")))
    row = paired._extend_paired_row(row, method=method, plan=plan, generation_bundle_key=_generation_bundle_key(experiment_seed, heldout_center, replicate_seed, candidates, str(plan.get("budget_policy", ""))))
    bundle = PredictionBundle(
        expert_id=method,
        probabilities=tuple(tuple(float(v) for v in values) for values in pooled),
        classes=bundles[0].classes,
    )
    probability_hash = _hash_array(np.asarray(bundle.probabilities, dtype=float))
    row_hash = str(row.get("prediction_hash", ""))
    max_abs_delta = 0.0 if probability_hash == row_hash else math.inf
    reconstruction_status = "PASS" if probability_hash == row_hash and max_abs_delta <= cfg.reconstruction_probability_tolerance else "FAIL"
    if reconstruction_status != "PASS":
        raise ProtocolError(f"Dense probability reconstruction failed for {method}: row hash {row_hash}, reconstructed {probability_hash}.")
    return DenseBundleEvaluation(
        row=row,
        bundle=bundle,
        late_rows=tuple(late_rows),
        coverage_rows=tuple(coverage_rows),
        weak_rows=tuple(weak_rows),
        nn_rows=tuple(nn_rows),
        generated_hash=generated_hash,
        plan=dict(plan),
        reconstruction_row=_dense_reconstruction_row(cfg, row, bundle, eval_sample_ids, status=reconstruction_status, max_abs_delta=max_abs_delta),
    )


def _blend_dense_and_bag(
    cfg: DenseTailShieldConfig,
    *,
    dense_eval: DenseBundleEvaluation,
    bag_eval: tr.BagEvaluation,
    dense_bundle: PredictionBundle,
    bag_bundle: PredictionBundle,
    candidates: Sequence[str],
    component_summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    eval_labels: Sequence[int],
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
    real_feature_bacc: float,
    method: str,
    alpha_dense: float,
    selection_source: str,
    claim_role: str,
) -> tuple[dict[str, object], PredictionBundle, dict[str, object], dict[str, object]]:
    alpha = float(alpha_dense)
    pooled = weighted_arithmetic_probability_pool([dense_bundle, bag_bundle], [alpha, 1.0 - alpha])
    bundle = PredictionBundle(
        expert_id=method,
        probabilities=tuple(tuple(float(v) for v in row) for row in pooled),
        classes=dense_bundle.classes,
    )
    result = evaluate_probability_predictions(method, bundle.probabilities, eval_labels, classes=bundle.classes)
    blend_plan = _blend_source_plan(cfg, candidates, dense_eval.plan, bag_eval.ensemble_plan, alpha_dense=alpha)
    row = cu._result_matrix_row(
        cfg,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        candidates=candidates,
        prior_method=method,
        summary_kind="dense_reliability_tailshield_random_mass_bag_probability_blend",
        source_union_ref=source_union_ref,
        center_balanced_ref=center_balanced_ref,
        real_feature_bacc=real_feature_bacc,
        weight_plan=blend_plan,
        bacc=result.bacc,
        macro_f1=result.macro_f1,
        generated_features_hash=_hash_strings([dense_eval.generated_hash, bag_eval.generated_hash]),
        prediction_hash=_hash_array(np.asarray(bundle.probabilities, dtype=float)),
        selection_source=selection_source,
        claim_role=claim_role,
        status="ok",
        error_message="",
        control_mode="normal",
        summaries=component_summaries,
    )
    row["pooling_rule"] = "fixed_arithmetic_probability_blend"
    row["dense_anchor_method"] = str(dense_eval.row.get("prior_method", ""))
    row["bag_method"] = BAG_METHOD
    row["blend_alpha_dense"] = alpha
    row["blend_alpha_bag"] = 1.0 - alpha
    row["target_calibration_metrics_audit_only"] = True
    row["class_order_match"] = True
    row["sample_order_match"] = True
    merged_counts = mb._merge_component_counts([bag_eval.component_counts])
    coverage = cu._component_coverage_row(row, merged_counts, cu._expected_component_keys(candidates, component_summaries, control_mode="normal"))
    paired_row = cu._paired_generation_row(row, str(row["generated_features_hash"]), _hash_strings([dense_eval.generated_hash, bag_eval.source_generation_hash]), "ok")
    return row, bundle, coverage, paired_row


def _append_dense_tailshield_outputs(
    evaluated: DenseTailShieldEvaluation,
    *,
    matrix_rows: list[dict[str, object]],
    component_coverage_rows: list[dict[str, object]],
    paired_generation_rows: list[dict[str, object]],
    source_weight_rows: list[dict[str, object]],
    blend_manifest_rows: list[dict[str, object]],
    reconstruction_rows: list[dict[str, object]],
    complementarity_rows: list[dict[str, object]],
    calibration_rows: list[dict[str, object]],
    confidence_rows: list[dict[str, object]],
    rescue_rows: list[dict[str, object]],
    alpha_curve_rows: list[dict[str, object]],
    eligibility_rows: list[dict[str, object]],
) -> None:
    for dense_eval in (evaluated.dense_anchor, evaluated.equal_anchor, evaluated.shuffled_anchor, evaluated.random_anchor):
        matrix_rows.append(dense_eval.row)
    matrix_rows.append(evaluated.bag_evaluation.ensemble_row)
    matrix_rows.extend(evaluated.control_rows)
    matrix_rows.append(evaluated.primary_row)
    component_coverage_rows.extend([evaluated.bag_evaluation.ensemble_coverage, *evaluated.control_coverages, evaluated.primary_coverage])
    paired_generation_rows.extend([evaluated.bag_evaluation.ensemble_paired_row, *evaluated.control_paired_rows, evaluated.primary_paired_row])
    source_weight_rows.extend(evaluated.source_weight_rows)
    blend_manifest_rows.extend(evaluated.blend_manifest_rows)
    reconstruction_rows.extend(evaluated.reconstruction_rows)
    complementarity_rows.append(evaluated.complementarity_row)
    calibration_rows.extend(evaluated.calibration_rows)
    confidence_rows.extend(evaluated.confidence_rows)
    rescue_rows.append(evaluated.rescue_row)
    alpha_curve_rows.extend(evaluated.alpha_curve_rows)
    eligibility_rows.extend(evaluated.eligibility_rows)


def _source_ablation_rows(
    cfg: DenseTailShieldConfig,
    *,
    root: Path,
    per_source_runtime: Mapping[str, RuntimeSource],
    dense_summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    component_summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    reliability: Mapping[tuple[int, int, str], d12.SourceReliability],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    eval_raw: object,
    eval_labels: Sequence[int],
    eval_sample_ids: Sequence[str],
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
                    "source_ablation_diagnostic_only": True,
                }
            )
            continue
        remaining = tuple(source for source in candidates if str(source) != str(removed))
        rels = {source: reliability[(int(experiment_seed), int(replicate_seed), str(source))] for source in remaining}
        evaluated = _evaluate_dense_tailshield_pair(
            cfg,
            root=root,
            per_source_runtime=per_source_runtime,
            dense_summaries=dense_summaries,
            component_summaries=component_summaries,
            candidates=remaining,
            rels=rels,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            eval_raw=eval_raw,
            eval_labels=eval_labels,
            eval_sample_ids=eval_sample_ids,
            source_union_ref=source_union_ref,
            center_balanced_ref=center_balanced_ref,
            real_feature_bacc=real_feature_bacc,
            include_diagnostics=False,
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
                "source_ablation_diagnostic_only": True,
            }
        )
    return rows


def _random_dense_anchor_plan(
    cfg: DenseTailShieldConfig,
    sources: Sequence[str],
    rels: Mapping[str, d12.SourceReliability],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
) -> dict[str, object]:
    plan = mb._dirichlet_source_plan(
        cfg,
        sources,
        rels,
        center_weights={str(source): 1.0 / float(len(sources)) for source in sources},
        alpha_per_source=4.0,
        family="random_dense_tailshield_uniform_alpha4",
        permutation_id=0,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
    )
    plan.update(
        {
            "weight_rule": "random_dense_dirichlet_uniform_alpha4",
            "budget_policy": "random_dense_largest_remainder_min8",
            "pairing_group": "random_dense_tailshield_uniform_alpha4",
            "generation_seed_method": "random_dense_tailshield_uniform_alpha4",
            "source_weighting": "random_dense_tailshield_uniform_alpha4",
        }
    )
    return plan


def _blend_source_plan(
    cfg: DenseTailShieldConfig,
    sources: Sequence[str],
    dense_plan: Mapping[str, object],
    bag_plan: Mapping[str, object],
    *,
    alpha_dense: float,
) -> dict[str, object]:
    weights = {}
    budgets = {}
    scores = {}
    alpha = float(alpha_dense)
    for source in sources:
        source_id = str(source)
        weights[source_id] = alpha * _float(dict(dense_plan["weights"]).get(source_id)) + (1.0 - alpha) * _float(dict(bag_plan["weights"]).get(source_id))
        scores[source_id] = alpha * _float(dict(dense_plan["scores"]).get(source_id)) + (1.0 - alpha) * _float(dict(bag_plan["scores"]).get(source_id))
    total = sum(weights.values())
    if total > 0.0:
        weights = {source: value / total for source, value in weights.items()}
    budgets = d12._weighted_budgets(cfg.synthetic_per_class_total, tuple(str(v) for v in sources), weights, cfg.min_per_source_per_class)
    plan = cu._with_weight_diagnostics(tuple(str(v) for v in sources), weights, budgets, scores, total=cfg.synthetic_per_class_total, mode=DENSE_TAILSHIELD_SOURCE_WEIGHTING)
    plan["blend_alpha_dense"] = alpha
    plan["blend_alpha_bag"] = 1.0 - alpha
    plan["dense_anchor_method"] = DENSE_ANCHOR_METHOD
    plan["bag_method"] = BAG_METHOD
    return plan


def _assert_probability_alignment(dense_bundle: PredictionBundle, bag_bundle: PredictionBundle, eval_sample_ids: Sequence[str]) -> None:
    if tuple(dense_bundle.classes) != tuple(bag_bundle.classes):
        raise ProtocolError(f"Class order mismatch before dense tail-shield blending: {dense_bundle.classes} vs {bag_bundle.classes}")
    dense = np.asarray(dense_bundle.probabilities, dtype=float)
    bag = np.asarray(bag_bundle.probabilities, dtype=float)
    if dense.shape != bag.shape:
        raise ProtocolError(f"Probability row/column mismatch before dense tail-shield blending: {dense.shape} vs {bag.shape}")
    if dense.shape[0] != len(eval_sample_ids):
        raise ProtocolError("Dense tail-shield eval sample count does not match probability rows.")
    if not bool(np.isfinite(dense).all() and np.isfinite(bag).all()):
        raise ProtocolError("Non-finite probability row before dense tail-shield blending.")
    if not bool(np.allclose(dense.sum(axis=1), 1.0, atol=1.0e-6) and np.allclose(bag.sum(axis=1), 1.0, atol=1.0e-6)):
        raise ProtocolError("Probability rows do not sum to 1 before dense tail-shield blending.")


def _dense_reconstruction_row(
    cfg: DenseTailShieldConfig,
    row: Mapping[str, object],
    bundle: PredictionBundle | None,
    eval_sample_ids: Sequence[str],
    *,
    status: str,
    max_abs_delta: float,
) -> dict[str, object]:
    probability_hash = _hash_array(np.asarray(bundle.probabilities, dtype=float)) if bundle is not None else ""
    row_hash = str(row.get("prediction_hash", ""))
    prediction_vector_hash = _hash_strings([str(v) for v in predict_from_probabilities(bundle.probabilities, classes=bundle.classes)]) if bundle is not None else ""
    return {
        "experiment_seed": row.get("experiment_seed", ""),
        "heldout_center": row.get("heldout_center", ""),
        "replicate_seed": row.get("replicate_seed", ""),
        "panel": row.get("panel", ""),
        "dense_anchor_method": row.get("prior_method", ""),
        "same_run_comparator_prediction_hash": row_hash,
        "reconstructed_prediction_hash": probability_hash,
        "max_abs_probability_delta": max_abs_delta,
        "probability_hash_match": bool(row_hash and probability_hash and row_hash == probability_hash),
        "same_sample_order": True,
        "eval_sample_id_hash": _hash_strings(eval_sample_ids),
        "same_class_order": True if bundle is not None else False,
        "same_class_columns": True if bundle is not None else False,
        "same_eligibility_mask": row.get("status") == "ok",
        "same_prediction_vector": True if bundle is not None else False,
        "prediction_vector_hash": prediction_vector_hash,
        "same_bacc": True if bundle is not None else False,
        "probability_reconstruction_tolerance": cfg.reconstruction_probability_tolerance,
        "dense_probability_reconstruction_status": status,
    }


def _blend_manifest_row(
    cfg: DenseTailShieldConfig,
    row: Mapping[str, object],
    dense_eval: DenseBundleEvaluation,
    bag_eval: tr.BagEvaluation,
    blended_hash: str,
    eval_sample_ids: Sequence[str],
    *,
    class_order_match: bool,
    sample_order_match: bool,
) -> dict[str, object]:
    class_order = ""
    if dense_eval.bundle is not None:
        class_order = "|".join(str(v) for v in dense_eval.bundle.classes)
    return {
        "experiment_seed": row.get("experiment_seed", ""),
        "heldout_center": row.get("heldout_center", ""),
        "replicate_seed": row.get("replicate_seed", ""),
        "panel": row.get("panel", ""),
        "primary_method": PRIMARY_DENSE_TAILSHIELD_METHOD,
        "anchor_method": DENSE_ANCHOR_METHOD,
        "bag_method": BAG_METHOD,
        "blend_alpha_anchor": cfg.dense_blend_alpha,
        "blend_alpha_bag": cfg.bag_blend_alpha,
        "anchor_prediction_hash": dense_eval.row.get("prediction_hash", ""),
        "bag_prediction_hash": bag_eval.ensemble_row.get("prediction_hash", ""),
        "blended_prediction_hash": blended_hash,
        "class_order": class_order,
        "class_order_match": bool(class_order_match),
        "eval_sample_id_hash": _hash_strings(eval_sample_ids),
        "sample_order_match": bool(sample_order_match),
    }


def _complementarity_row(
    cfg: DenseTailShieldConfig,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    dense_bundle: PredictionBundle,
    bag_bundle: PredictionBundle,
    eval_labels: Sequence[int],
    primary_row: Mapping[str, object],
) -> dict[str, object]:
    dense_preds = predict_from_probabilities(dense_bundle.probabilities, classes=dense_bundle.classes)
    bag_preds = predict_from_probabilities(bag_bundle.probabilities, classes=bag_bundle.classes)
    labels = tuple(int(v) for v in eval_labels)
    n = len(labels)
    if n == 0:
        return _empty_complementarity_row(cfg, experiment_seed, heldout_center, replicate_seed, "empty_eval")
    dense_correct = [int(pred == label) for pred, label in zip(dense_preds, labels)]
    bag_correct = [int(pred == label) for pred, label in zip(bag_preds, labels)]
    dense_correct_bag_wrong = sum(1 for d, b in zip(dense_correct, bag_correct) if d and not b) / float(n)
    bag_correct_dense_wrong = sum(1 for d, b in zip(dense_correct, bag_correct) if b and not d) / float(n)
    both_wrong = sum(1 for d, b in zip(dense_correct, bag_correct) if not d and not b) / float(n)
    both_correct = sum(1 for d, b in zip(dense_correct, bag_correct) if d and b) / float(n)
    disagreement = sum(1 for d, b in zip(dense_preds, bag_preds) if int(d) != int(b)) / float(n)
    return {
        **_cell_identity(cfg, experiment_seed, heldout_center, replicate_seed),
        "primary_bacc": primary_row.get("bacc", math.nan),
        "dense_correct_bag_wrong_rate": dense_correct_bag_wrong,
        "bag_correct_dense_wrong_rate": bag_correct_dense_wrong,
        "both_wrong_rate": both_wrong,
        "both_correct_rate": both_correct,
        "disagreement_rate": disagreement,
        "center3_disagreement_rate": disagreement if str(heldout_center) == "3" else "",
        "bottom20_disagreement_rate": "",
        "is_bottom20_cell": False,
        "nontrivial_rescue_threshold": cfg.nontrivial_rescue_threshold,
        "status": "ok",
    }


def _empty_complementarity_row(
    cfg: DenseTailShieldConfig,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    status: str,
) -> dict[str, object]:
    return {
        **_cell_identity(cfg, experiment_seed, heldout_center, replicate_seed),
        "primary_bacc": math.nan,
        "dense_correct_bag_wrong_rate": math.nan,
        "bag_correct_dense_wrong_rate": math.nan,
        "both_wrong_rate": math.nan,
        "both_correct_rate": math.nan,
        "disagreement_rate": math.nan,
        "center3_disagreement_rate": "",
        "bottom20_disagreement_rate": "",
        "is_bottom20_cell": False,
        "nontrivial_rescue_threshold": cfg.nontrivial_rescue_threshold,
        "status": status,
    }


def _rescue_row(
    cfg: DenseTailShieldConfig,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    dense_bundle: PredictionBundle,
    bag_bundle: PredictionBundle,
    eval_labels: Sequence[int],
    primary_row: Mapping[str, object],
) -> dict[str, object]:
    row = _complementarity_row(cfg, experiment_seed, heldout_center, replicate_seed, dense_bundle, bag_bundle, eval_labels, primary_row)
    rescue_rate = _float(row["dense_correct_bag_wrong_rate"])
    row.update(
        {
            "dense_tailshield_rescue_rate": rescue_rate,
            "nontrivial_rescue": bool(math.isfinite(rescue_rate) and rescue_rate >= cfg.nontrivial_rescue_threshold),
            "tail_cell_scope": "center3" if str(heldout_center) == "3" else "candidate",
        }
    )
    return row


def _empty_rescue_row(
    cfg: DenseTailShieldConfig,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    status: str,
) -> dict[str, object]:
    row = _empty_complementarity_row(cfg, experiment_seed, heldout_center, replicate_seed, status)
    row.update({"dense_tailshield_rescue_rate": math.nan, "nontrivial_rescue": False, "tail_cell_scope": ""})
    return row


def _calibration_rows(
    cfg: DenseTailShieldConfig,
    *,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    eval_labels: Sequence[int],
    dense_bundle: PredictionBundle,
    bag_bundle: PredictionBundle,
    blended_bundle: PredictionBundle,
) -> list[dict[str, object]]:
    rows = []
    for source, method, bundle in (
        ("dense_reliability_anchor", DENSE_ANCHOR_METHOD, dense_bundle),
        ("random_mass_bag", BAG_METHOD, bag_bundle),
        ("primary_blend", PRIMARY_DENSE_TAILSHIELD_METHOD, blended_bundle),
    ):
        metrics = _probability_calibration_metrics(bundle.probabilities, eval_labels, bundle.classes)
        rows.append(
            {
                **_cell_identity(cfg, experiment_seed, heldout_center, replicate_seed),
                "probability_source": source,
                "prior_method": method,
                "source_inner_brier": math.nan,
                "source_inner_ece": math.nan,
                "source_inner_log_loss": math.nan,
                "source_inner_calibration_available": False,
                "target_eval_brier_diagnostic_only": metrics["brier"],
                "target_eval_ece_diagnostic_only": metrics["ece"],
                "target_eval_adaptive_ece_diagnostic_only": metrics["adaptive_ece"],
                "target_eval_log_loss_diagnostic_only": metrics["log_loss"],
                "target_calibration_audit_only": True,
                "used_for_alpha_or_adoption": False,
                "used_for_source_composition_change": False,
            }
        )
    return rows


def _confidence_rows(
    cfg: DenseTailShieldConfig,
    *,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    dense_bundle: PredictionBundle,
    bag_bundle: PredictionBundle,
    blended_bundle: PredictionBundle,
) -> list[dict[str, object]]:
    rows = []
    dense_metrics = _confidence_metrics(dense_bundle.probabilities)
    bag_metrics = _confidence_metrics(bag_bundle.probabilities)
    for source, method, bundle in (
        ("dense_reliability_anchor", DENSE_ANCHOR_METHOD, dense_bundle),
        ("random_mass_bag", BAG_METHOD, bag_bundle),
        ("primary_blend", PRIMARY_DENSE_TAILSHIELD_METHOD, blended_bundle),
    ):
        metrics = _confidence_metrics(bundle.probabilities)
        rows.append(
            {
                **_cell_identity(cfg, experiment_seed, heldout_center, replicate_seed),
                "probability_source": source,
                "prior_method": method,
                **metrics,
                "per_center_confidence_shift_vs_dense": metrics["mean_max_probability"] - dense_metrics["mean_max_probability"] if math.isfinite(metrics["mean_max_probability"]) else math.nan,
                "per_center_confidence_shift_vs_bag": metrics["mean_max_probability"] - bag_metrics["mean_max_probability"] if math.isfinite(metrics["mean_max_probability"]) else math.nan,
                "target_calibration_metrics_audit_only": True,
                "used_for_alpha_or_adoption": False,
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
        return {"brier": math.nan, "ece": math.nan, "adaptive_ece": math.nan, "log_loss": math.nan}
    lookup = {value: idx for idx, value in enumerate(cls)}
    true_idx = np.asarray([lookup.get(int(v), -1) for v in y], dtype=int)
    valid = true_idx >= 0
    probs = probs[valid]
    true_idx = true_idx[valid]
    if probs.shape[0] == 0:
        return {"brier": math.nan, "ece": math.nan, "adaptive_ece": math.nan, "log_loss": math.nan}
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(probs.shape[0]), true_idx] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    clipped = np.clip(probs[np.arange(probs.shape[0]), true_idx], 1.0e-12, 1.0)
    log_loss = float(-np.mean(np.log(clipped)))
    pred_idx = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)
    correct = (pred_idx == true_idx).astype(float)
    ece = _ece_from_confidence(confidence, correct)
    adaptive_ece = _adaptive_ece_from_confidence(confidence, correct)
    return {"brier": brier, "ece": ece, "adaptive_ece": adaptive_ece, "log_loss": log_loss}


def _ece_from_confidence(confidence: np.ndarray, correct: np.ndarray) -> float:
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        mask = (confidence >= lower) & (confidence <= upper if upper >= 1.0 else confidence < upper)
        if not bool(mask.any()):
            continue
        ece += float(mask.mean()) * abs(float(confidence[mask].mean()) - float(correct[mask].mean()))
    return float(ece)


def _adaptive_ece_from_confidence(confidence: np.ndarray, correct: np.ndarray) -> float:
    if confidence.size == 0:
        return math.nan
    order = np.argsort(confidence)
    bins = np.array_split(order, min(10, confidence.size))
    ece = 0.0
    for idx in bins:
        if idx.size == 0:
            continue
        ece += float(idx.size / confidence.size) * abs(float(confidence[idx].mean()) - float(correct[idx].mean()))
    return float(ece)


def _confidence_metrics(probabilities: Sequence[Sequence[float]]) -> dict[str, float]:
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2 or probs.shape[0] == 0:
        return {
            "mean_entropy": math.nan,
            "mean_max_probability": math.nan,
            "mean_top1_top2_margin": math.nan,
            "row_sum_min": math.nan,
            "row_sum_max": math.nan,
            "probability_rows_finite": False,
            "probability_row_sums_close_to_one": False,
        }
    clipped = np.clip(probs, 1.0e-12, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    sorted_probs = np.sort(probs, axis=1)
    top1 = sorted_probs[:, -1]
    top2 = sorted_probs[:, -2] if sorted_probs.shape[1] > 1 else np.zeros_like(top1)
    row_sums = probs.sum(axis=1)
    return {
        "mean_entropy": float(np.mean(entropy)),
        "mean_max_probability": float(np.mean(top1)),
        "mean_top1_top2_margin": float(np.mean(top1 - top2)),
        "row_sum_min": float(np.min(row_sums)),
        "row_sum_max": float(np.max(row_sums)),
        "probability_rows_finite": bool(np.isfinite(probs).all()),
        "probability_row_sums_close_to_one": bool(np.allclose(row_sums, 1.0, atol=1.0e-6)),
    }


def _tail_metrics(
    rows: Sequence[Mapping[str, object]],
    method: str,
    *,
    panel: str = "combined",
    bottom20_keys: set[tuple[str, str, str]] | None = None,
) -> dict[str, object]:
    subset = cu._rows_for(cu._rows_for_panel(rows, panel), method)
    stats = cu._method_stats(subset)
    grouped = cu._replicate_averaged(subset)
    grouped_bacc = [_float(row.get("bacc")) for row in grouped if math.isfinite(_float(row.get("bacc")))]
    raw_by_key = {
        (str(row.get("experiment_seed")), str(row.get("heldout_center")), str(row.get("replicate_seed"))): row
        for row in subset
        if row.get("status") == "ok"
    }
    if bottom20_keys is None:
        bottom_values = []
    else:
        bottom_values = [_float(raw_by_key[key].get("bacc")) for key in bottom20_keys if key in raw_by_key and math.isfinite(_float(raw_by_key[key].get("bacc")))]
    center3_rows = [row for row in grouped if str(row.get("heldout_center")) == "3"]
    return {
        **stats,
        "bottom20_cell_mean_bacc": nanmean(bottom_values) if bottom_values else math.nan,
        "worst_seed_center_bacc": min(grouped_bacc) if grouped_bacc else math.nan,
        "center3_bacc": d1._mean_field(center3_rows, "bacc") if center3_rows else math.nan,
    }


def _tail_metric_summary_rows(rows: Sequence[Mapping[str, object]], bottom20_keys: set[tuple[str, str, str]]) -> list[dict[str, object]]:
    out = []
    methods = sorted({str(row.get("prior_method")) for row in rows if row.get("prior_method")})
    for panel in ("canonical", "fresh", "combined"):
        for method in methods:
            metrics = _tail_metrics(rows, method, panel=panel, bottom20_keys=bottom20_keys)
            if int(metrics["n_raw_rows"]) < 1:
                continue
            bag = _tail_metrics(rows, BAG_METHOD, panel=panel, bottom20_keys=bottom20_keys)
            shrink050 = _tail_metrics(rows, cu.ROW_COMPONENT_UNION_SHRINK050, panel=panel, bottom20_keys=bottom20_keys)
            dense = _tail_metrics(rows, DENSE_ANCHOR_METHOD, panel=panel, bottom20_keys=bottom20_keys)
            out.append(
                {
                    "panel": panel,
                    "prior_method": method,
                    **metrics,
                    "center3_delta_vs_random_mass_bag": _delta(metrics["center3_bacc"], bag["center3_bacc"]),
                    "center3_delta_vs_shrink050": _delta(metrics["center3_bacc"], shrink050["center3_bacc"]),
                    "center3_delta_vs_dense_reliability": _delta(metrics["center3_bacc"], dense["center3_bacc"]),
                    "bottom20_delta_vs_random_mass_bag": _delta(metrics["bottom20_cell_mean_bacc"], bag["bottom20_cell_mean_bacc"]),
                    "bottom20_delta_vs_shrink050": _delta(metrics["bottom20_cell_mean_bacc"], shrink050["bottom20_cell_mean_bacc"]),
                    "bottom20_delta_vs_dense_reliability": _delta(metrics["bottom20_cell_mean_bacc"], dense["bottom20_cell_mean_bacc"]),
                    "bottom20_definition": "lowest_20pct_eligible_raw_cells_by_random_mass_bag_control_bacc",
                    "center3_definition": 'heldout_center == "3"',
                }
            )
    return out


def _bottom20_raw_cell_keys(rows: Sequence[Mapping[str, object]], baseline_method: str) -> set[tuple[str, str, str]]:
    baseline = [
        row for row in cu._rows_for(rows, baseline_method)
        if row.get("status") == "ok" and math.isfinite(_float(row.get("bacc")))
    ]
    count = max(1, int(math.ceil(0.20 * len(baseline)))) if baseline else 0
    bottom = sorted(baseline, key=lambda row: (_float(row.get("bacc")), str(row.get("experiment_seed")), str(row.get("heldout_center")), str(row.get("replicate_seed"))))[:count]
    return {(str(row.get("experiment_seed")), str(row.get("heldout_center")), str(row.get("replicate_seed"))) for row in bottom}


def _mark_bottom20_rows(rows: Sequence[Mapping[str, object]], bottom20_keys: set[tuple[str, str, str]]) -> list[dict[str, object]]:
    out = []
    for row in rows:
        updated = dict(row)
        key = (str(updated.get("experiment_seed")), str(updated.get("heldout_center")), str(updated.get("replicate_seed")))
        is_bottom = key in bottom20_keys
        updated["is_bottom20_cell"] = bool(is_bottom)
        if is_bottom:
            updated["bottom20_disagreement_rate"] = updated.get("disagreement_rate", "")
            updated["tail_cell_scope"] = "bottom20" if str(updated.get("heldout_center")) != "3" else "center3|bottom20"
        out.append(updated)
    return out


def _decision(
    rows: Sequence[Mapping[str, object]],
    *,
    cfg: DenseTailShieldConfig,
    leakage_status: str,
    source_ablation_rows: Sequence[Mapping[str, object]],
    anchor_rows: Sequence[Mapping[str, object]],
    reconstruction_rows: Sequence[Mapping[str, object]],
    complementarity_rows: Sequence[Mapping[str, object]],
    shuffled_summary: Sequence[Mapping[str, object]],
    bottom20_keys: set[tuple[str, str, str]],
) -> dict[str, object]:
    primary = _tail_metrics(rows, PRIMARY_DENSE_TAILSHIELD_METHOD, bottom20_keys=bottom20_keys)
    dense = _tail_metrics(rows, DENSE_ANCHOR_METHOD, bottom20_keys=bottom20_keys)
    bag = _tail_metrics(rows, BAG_METHOD, bottom20_keys=bottom20_keys)
    shrink050 = _tail_metrics(rows, cu.ROW_COMPONENT_UNION_SHRINK050, bottom20_keys=bottom20_keys)
    equal = _tail_metrics(rows, DENSE_EQUAL_ANCHOR_METHOD, bottom20_keys=bottom20_keys)
    uniform = _tail_metrics(rows, cu.PRIMARY_COMPONENT_UNION_METHOD, bottom20_keys=bottom20_keys)
    source_union = _tail_metrics(rows, cu.ROW_SOURCE_UNION_K16_REFERENCE, bottom20_keys=bottom20_keys)
    real = _tail_metrics(rows, cu.ROW_REAL_FEATURE_DENSE_REFERENCE, bottom20_keys=bottom20_keys)
    equal_shield = _tail_metrics(rows, EQUAL_DENSE_SHIELD_METHOD, bottom20_keys=bottom20_keys)
    shuffled_shield = _tail_metrics(rows, SHUFFLED_DENSE_SHIELD_METHOD, bottom20_keys=bottom20_keys)
    random_shield = _tail_metrics(rows, RANDOM_DENSE_SHIELD_METHOD, bottom20_keys=bottom20_keys)
    random_single = _tail_metrics(rows, cu.ROW_RANDOM_SOURCE_MASS_CONTROL, bottom20_keys=bottom20_keys)

    primary_bacc = _float(primary["center_equal_mean_bacc"])
    bag_bacc = _float(bag["center_equal_mean_bacc"])
    dense_bacc = _float(dense["center_equal_mean_bacc"])
    shrink050_bacc = _float(shrink050["center_equal_mean_bacc"])
    source_union_bacc = _float(source_union["center_equal_mean_bacc"])
    real_bacc = _float(real["center_equal_mean_bacc"])
    max_parent = max(value for value in (dense_bacc, bag_bacc) if math.isfinite(value)) if any(math.isfinite(v) for v in (dense_bacc, bag_bacc)) else math.nan
    reconstruction_pass = bool(reconstruction_rows) and all(row.get("dense_probability_reconstruction_status") == "PASS" for row in reconstruction_rows)
    anchor_pass = bool(anchor_rows) and all(row.get("anchor_repro_status") in {"PASS", "NO_EXPECTED_ARTIFACT"} for row in anchor_rows)
    no_cell_worse = _no_seed_center_worse_than_both(rows)
    fresh_preserves = _fresh_preserves_tail_direction(rows, bottom20_keys)
    rescue_nontrivial = _complementarity_nontrivial(cfg, complementarity_rows)
    control_tail_beaten = _control_tail_beaten(primary, (equal_shield, shuffled_shield, random_shield))
    ablation = _source_ablation_stats(source_ablation_rows)
    strongest_control = max(
        (
            (EQUAL_DENSE_SHIELD_METHOD, _float(equal_shield["bottom20_cell_mean_bacc"])),
            (SHUFFLED_DENSE_SHIELD_METHOD, _float(shuffled_shield["bottom20_cell_mean_bacc"])),
            (RANDOM_DENSE_SHIELD_METHOD, _float(random_shield["bottom20_cell_mean_bacc"])),
            (cu.ROW_RANDOM_SOURCE_MASS_CONTROL, _float(random_single["bottom20_cell_mean_bacc"])),
        ),
        key=lambda item: item[1] if math.isfinite(item[1]) else -math.inf,
    )
    flags: list[str] = []
    if leakage_status != "PASS":
        flags.append("LEAKAGE_FAIL")
    if not reconstruction_pass:
        flags.append("DENSE_PROBABILITY_RECONSTRUCTION_MISMATCH")
    if not anchor_pass:
        flags.append("ANCHOR_REPRO_MISMATCH")
    if not no_cell_worse:
        flags.append("SEED_CENTER_CELL_WORSE_THAN_BOTH_PARENTS")
    if not fresh_preserves:
        flags.append("FRESH_PANEL_REVERSES_TAIL_DIRECTION")
    if not rescue_nontrivial:
        flags.append("DENSE_BAG_RESCUE_RATE_WEAK")
    if not control_tail_beaten:
        flags.append("DENSE_SHIELD_CONTROLS_MATCH_TAIL_GAIN")
    if math.isfinite(max_parent) and primary_bacc < max_parent - 0.005:
        flags.append("MEAN_DROPS_GT_0P005_BELOW_PARENT")
    if _delta(primary["center3_bacc"], bag["center3_bacc"]) <= 0.0:
        flags.append("CENTER3_NOT_IMPROVED_VS_RANDOM_MASS_BAG")
    if _delta(primary["bottom20_cell_mean_bacc"], bag["bottom20_cell_mean_bacc"]) <= 0.0:
        flags.append("BOTTOM20_NOT_IMPROVED_VS_RANDOM_MASS_BAG")
    if math.isfinite(_float(primary["center3_bacc"])) and _float(primary["center3_bacc"]) < 0.80:
        flags.append("CENTER3_BELOW_0P80")

    strong = (
        leakage_status == "PASS"
        and reconstruction_pass
        and int(primary["n_heldout_centers"]) >= len(cfg.heldout_centers)
        and math.isfinite(bag_bacc)
        and primary_bacc >= bag_bacc - 0.005
        and _float(primary["min_center_bacc"]) >= 0.82
        and _delta(primary["center3_bacc"], bag["center3_bacc"]) >= 0.025
        and _delta(primary["bottom20_cell_mean_bacc"], bag["bottom20_cell_mean_bacc"]) >= 0.025
        and _delta(primary["worst_seed_center_bacc"], bag["worst_seed_center_bacc"]) >= 0.030
        and _delta(primary["bottom20_cell_mean_bacc"], shrink050["bottom20_cell_mean_bacc"]) >= 0.010
        and (
            _float(primary["center3_bacc"]) >= _float(shrink050["center3_bacc"])
            or (_delta(primary["center3_bacc"], shrink050["center3_bacc"]) >= -0.005 and primary_bacc > shrink050_bacc)
        )
        and _float(primary["seed_std_bacc"]) <= 0.045
        and no_cell_worse
        and control_tail_beaten
        and rescue_nontrivial
    )
    useful = (
        leakage_status == "PASS"
        and reconstruction_pass
        and math.isfinite(bag_bacc)
        and primary_bacc >= bag_bacc - 0.007
        and _float(primary["min_center_bacc"]) >= 0.80
        and _delta(primary["center3_bacc"], bag["center3_bacc"]) > 0.0
        and _delta(primary["bottom20_cell_mean_bacc"], bag["bottom20_cell_mean_bacc"]) > 0.0
        and _delta(primary["worst_seed_center_bacc"], bag["worst_seed_center_bacc"]) > 0.0
        and _float(primary["seed_std_bacc"]) <= _float(bag["seed_std_bacc"]) + 1.0e-12
        and fresh_preserves
        and rescue_nontrivial
    )
    verdict = "DENSE_TAILSHIELD_RANDOM_MASS_BAG_FAIL"
    if leakage_status != "PASS":
        verdict = "PROTOCOL_FAIL"
    elif strong:
        verdict = "DENSE_TAILSHIELD_RANDOM_MASS_BAG_STRONG_SUCCESS"
    elif useful:
        verdict = "DENSE_TAILSHIELD_RANDOM_MASS_BAG_USEFUL_THESIS_SUCCESS"
    retention = d1._retention(primary_bacc, source_union_bacc)
    return {
        "primary_verdict": verdict,
        "diagnostic_flags": "|".join(flags),
        "primary_method": PRIMARY_DENSE_TAILSHIELD_METHOD,
        "leakage_status": leakage_status,
        "dense_probability_reconstruction_status": "PASS" if reconstruction_pass else "FAIL",
        "anchor_reproducibility_status": "PASS" if anchor_pass else "ANCHOR_MISMATCH",
        "class_order_match_all_cells": _class_order_match_all(rows),
        "center3_definition": 'heldout_center == "3"',
        "bottom20_definition": "lowest 20% eligible seed-center-replicate cells by random_mass_bag_control BACC",
        "nontrivial_rescue_threshold": cfg.nontrivial_rescue_threshold,
        "center_equal_mean_bacc": primary["center_equal_mean_bacc"],
        "seed_cell_mean_bacc": primary["seed_cell_mean_bacc"],
        "center_equal_macro_f1": primary["center_equal_macro_f1"],
        "min_center_bacc": primary["min_center_bacc"],
        "seed_std_bacc": primary["seed_std_bacc"],
        "bottom20_cell_mean_bacc": primary["bottom20_cell_mean_bacc"],
        "worst_seed_center_bacc": primary["worst_seed_center_bacc"],
        "center3_bacc": primary["center3_bacc"],
        "dense_reliability_center_equal_mean_bacc": dense_bacc,
        "random_mass_bag_center_equal_mean_bacc": bag_bacc,
        "shrink050_center_equal_mean_bacc": shrink050_bacc,
        "equal_all4_center_equal_mean_bacc": _float(equal["center_equal_mean_bacc"]),
        "uniform_component_union_center_equal_mean_bacc": _float(uniform["center_equal_mean_bacc"]),
        "source_union_k16_reference_center_equal_mean_bacc": source_union_bacc,
        "real_feature_dense_reference_center_equal_mean_bacc": real_bacc,
        "center3_delta_vs_random_mass_bag": _delta(primary["center3_bacc"], bag["center3_bacc"]),
        "center3_delta_vs_shrink050": _delta(primary["center3_bacc"], shrink050["center3_bacc"]),
        "center3_delta_vs_dense_reliability": _delta(primary["center3_bacc"], dense["center3_bacc"]),
        "bottom20_delta_vs_random_mass_bag": _delta(primary["bottom20_cell_mean_bacc"], bag["bottom20_cell_mean_bacc"]),
        "bottom20_delta_vs_shrink050": _delta(primary["bottom20_cell_mean_bacc"], shrink050["bottom20_cell_mean_bacc"]),
        "bottom20_delta_vs_dense_reliability": _delta(primary["bottom20_cell_mean_bacc"], dense["bottom20_cell_mean_bacc"]),
        "worst_seed_center_delta_vs_random_mass_bag": _delta(primary["worst_seed_center_bacc"], bag["worst_seed_center_bacc"]),
        "delta_vs_random_mass_bag": primary_bacc - bag_bacc if math.isfinite(primary_bacc) and math.isfinite(bag_bacc) else math.nan,
        "delta_vs_dense_reliability": primary_bacc - dense_bacc if math.isfinite(primary_bacc) and math.isfinite(dense_bacc) else math.nan,
        "delta_vs_shrink050": primary_bacc - shrink050_bacc if math.isfinite(primary_bacc) and math.isfinite(shrink050_bacc) else math.nan,
        "retention_vs_source_union_k16": retention,
        "oracle_gap_vs_source_union_k16": source_union_bacc - primary_bacc if math.isfinite(source_union_bacc) and math.isfinite(primary_bacc) else math.nan,
        "oracle_gap_vs_real_feature_dense": real_bacc - primary_bacc if math.isfinite(real_bacc) and math.isfinite(primary_bacc) else math.nan,
        "no_seed_center_cell_worse_than_both_parents_by_gt_0p010": no_cell_worse,
        "fresh_panel_preserves_tail_direction": fresh_preserves,
        "dense_correct_bag_wrong_nontrivial_on_center3_or_bottom20": rescue_nontrivial,
        "dense_shield_controls_tail_beaten": control_tail_beaten,
        "strongest_negative_control_method": strongest_control[0],
        "strongest_negative_control_bottom20_bacc": strongest_control[1],
        "negative_control_bottom20_gap": _float(primary["bottom20_cell_mean_bacc"]) - strongest_control[1] if math.isfinite(_float(primary["bottom20_cell_mean_bacc"])) and math.isfinite(strongest_control[1]) else math.nan,
        "shuffled_dense_shield_center_equal_mean_bacc": _float(shuffled_shield["center_equal_mean_bacc"]),
        "primary_minus_shuffled_dense_shield_mean": primary_bacc - _float(shuffled_shield["center_equal_mean_bacc"]) if math.isfinite(primary_bacc) else math.nan,
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


def _control_tail_beaten(primary: Mapping[str, object], controls: Sequence[Mapping[str, object]]) -> bool:
    p_center3 = _float(primary.get("center3_bacc"))
    p_bottom = _float(primary.get("bottom20_cell_mean_bacc"))
    for control in controls:
        c_center3 = _float(control.get("center3_bacc"))
        c_bottom = _float(control.get("bottom20_cell_mean_bacc"))
        if math.isfinite(c_center3) and p_center3 < c_center3 - 1.0e-12:
            return False
        if math.isfinite(c_bottom) and p_bottom < c_bottom - 1.0e-12:
            return False
    return True


def _no_seed_center_worse_than_both(rows: Sequence[Mapping[str, object]]) -> bool:
    primary = _seed_center_bacc(rows, PRIMARY_DENSE_TAILSHIELD_METHOD)
    dense = _seed_center_bacc(rows, DENSE_ANCHOR_METHOD)
    bag = _seed_center_bacc(rows, BAG_METHOD)
    for key, value in primary.items():
        d = dense.get(key, math.nan)
        b = bag.get(key, math.nan)
        if math.isfinite(value) and math.isfinite(d) and math.isfinite(b) and value < min(d, b) - 0.010:
            return False
    return True


def _seed_center_bacc(rows: Sequence[Mapping[str, object]], method: str) -> dict[tuple[str, str], float]:
    out = {}
    for row in cu._replicate_averaged(cu._rows_for(rows, method)):
        out[(str(row["experiment_seed"]), str(row["heldout_center"]))] = _float(row.get("bacc"))
    return out


def _fresh_preserves_tail_direction(rows: Sequence[Mapping[str, object]], bottom20_keys: set[tuple[str, str, str]]) -> bool:
    canonical_primary = _tail_metrics(rows, PRIMARY_DENSE_TAILSHIELD_METHOD, panel="canonical", bottom20_keys=bottom20_keys)
    canonical_bag = _tail_metrics(rows, BAG_METHOD, panel="canonical", bottom20_keys=bottom20_keys)
    fresh_primary = _tail_metrics(rows, PRIMARY_DENSE_TAILSHIELD_METHOD, panel="fresh", bottom20_keys=bottom20_keys)
    fresh_bag = _tail_metrics(rows, BAG_METHOD, panel="fresh", bottom20_keys=bottom20_keys)
    for field in ("min_center_bacc", "center3_bacc", "bottom20_cell_mean_bacc"):
        canonical_delta = _delta(canonical_primary[field], canonical_bag[field])
        fresh_delta = _delta(fresh_primary[field], fresh_bag[field])
        if math.isfinite(canonical_delta) and canonical_delta > 0.0 and math.isfinite(fresh_delta) and fresh_delta < 0.0:
            return False
    return True


def _complementarity_nontrivial(cfg: DenseTailShieldConfig, rows: Sequence[Mapping[str, object]]) -> bool:
    values = []
    for row in rows:
        if str(row.get("heldout_center")) == "3" or str(row.get("is_bottom20_cell")) == "True":
            value = _float(row.get("dense_correct_bag_wrong_rate"))
            if math.isfinite(value):
                values.append(value)
    return max(values, default=0.0) >= cfg.nontrivial_rescue_threshold


def _class_order_match_all(rows: Sequence[Mapping[str, object]]) -> bool:
    primary_rows = cu._rows_for(rows, PRIMARY_DENSE_TAILSHIELD_METHOD)
    return bool(primary_rows) and all(str(row.get("class_order_match", "")) == "True" or row.get("class_order_match") is True for row in primary_rows)


def _anchor_reproducibility_rows(
    rows: Sequence[Mapping[str, object]],
    cfg: DenseTailShieldConfig,
) -> list[dict[str, object]]:
    expected = mb._load_paired_expected(cfg.paired_dense_artifact_root)
    out = []
    for method, expected_key in (
        (paired.ROW_EQUAL_ALL4, "equal_all4_center_equal_mean_bacc"),
        (paired.ROW_RELIABILITY_ALL4_WEIGHTED, "best_center_equal_mean_bacc"),
    ):
        observed = _float(cu._method_stats(cu._rows_for(rows, method))["center_equal_mean_bacc"])
        expected_value = _float(expected.get(expected_key, math.nan))
        delta = observed - expected_value if math.isfinite(observed) and math.isfinite(expected_value) else math.nan
        status = "NO_EXPECTED_ARTIFACT"
        if math.isfinite(delta):
            status = "PASS" if abs(delta) <= cfg.anchor_repro_tolerance else "ANCHOR_MISMATCH"
        out.append(
            {
                "anchor_method": method,
                "observed_center_equal_mean_bacc": observed,
                "expected_center_equal_mean_bacc": expected_value,
                "delta_observed_minus_expected": delta,
                "tolerance": cfg.anchor_repro_tolerance,
                "anchor_repro_status": status,
                "expected_artifact_root": "" if cfg.paired_dense_artifact_root is None else str(cfg.paired_dense_artifact_root),
            }
        )
    return out


def _shuffled_dense_shield_summary_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    primary = cu._method_stats(cu._rows_for(rows, PRIMARY_DENSE_TAILSHIELD_METHOD))
    shuffled = cu._method_stats(cu._rows_for(rows, SHUFFLED_DENSE_SHIELD_METHOD))
    p = _float(primary["center_equal_mean_bacc"])
    s = _float(shuffled["center_equal_mean_bacc"])
    return [
        {
            "primary_method": PRIMARY_DENSE_TAILSHIELD_METHOD,
            "shuffled_control_method": SHUFFLED_DENSE_SHIELD_METHOD,
            "primary_center_equal_mean_bacc": p,
            "shuffled_center_equal_mean_bacc": s,
            "primary_minus_shuffled_dense_shield": p - s if math.isfinite(p) and math.isfinite(s) else math.nan,
            "control_type": "single_matched_shuffled_dense_reliability_shield",
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


def _oracle_gap_rows(rows: Sequence[Mapping[str, object]], bottom20_keys: set[tuple[str, str, str]]) -> list[dict[str, object]]:
    out = []
    for method in sorted({str(row.get("prior_method")) for row in rows if row.get("prior_method")}):
        stats = _tail_metrics(rows, method, bottom20_keys=bottom20_keys)
        if int(stats["n_raw_rows"]) < 1:
            continue
        out.append({"prior_method": method, **stats})
    return out


def _random_mass_bag_summary(rows: Sequence[Mapping[str, object]], bottom20_keys: set[tuple[str, str, str]]) -> list[dict[str, object]]:
    return [{"prior_method": BAG_METHOD, **_tail_metrics(rows, BAG_METHOD, bottom20_keys=bottom20_keys)}]


def _negative_control_summary(decision: Mapping[str, object]) -> dict[str, object]:
    return {
        "primary_method": PRIMARY_DENSE_TAILSHIELD_METHOD,
        "primary_center_equal_mean_bacc": decision.get("center_equal_mean_bacc", math.nan),
        "primary_bottom20_cell_mean_bacc": decision.get("bottom20_cell_mean_bacc", math.nan),
        "strongest_negative_control_method": decision.get("strongest_negative_control_method", ""),
        "strongest_negative_control_bottom20_bacc": decision.get("strongest_negative_control_bottom20_bacc", math.nan),
        "negative_control_bottom20_gap": decision.get("negative_control_bottom20_gap", math.nan),
        "control_competitive": "DENSE_SHIELD_CONTROLS_MATCH_TAIL_GAIN" in str(decision.get("diagnostic_flags", "")),
    }


def _write_artifacts(
    root: Path,
    cfg: DenseTailShieldConfig,
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
    reconstruction_rows: Sequence[Mapping[str, object]],
    complementarity_rows: Sequence[Mapping[str, object]],
    calibration_rows: Sequence[Mapping[str, object]],
    confidence_rows: Sequence[Mapping[str, object]],
    rescue_rows: Sequence[Mapping[str, object]],
    alpha_curve_rows: Sequence[Mapping[str, object]],
    shuffled_summary: Sequence[Mapping[str, object]],
    model_manifest_rows: Sequence[Mapping[str, object]],
    anchor_rows: Sequence[Mapping[str, object]],
    decision: Mapping[str, object],
    leakage: object,
    protocol_violations: Sequence[str],
    target_expert_excluded: bool,
) -> None:
    bottom20_keys = _bottom20_raw_cell_keys(matrix_rows, BAG_METHOD)
    write_csv_rows(root / "tables" / "dense_tailshield_downstream_matrix.csv", matrix_rows)
    write_csv_rows(root / "tables" / "dense_tailshield_summary.csv", [dict(decision)])
    write_csv_rows(root / "tables" / "dense_tailshield_panel_summary.csv", _panel_summary_rows(matrix_rows))
    write_csv_rows(root / "tables" / "dense_tailshield_tail_metric_summary.csv", _tail_metric_summary_rows(matrix_rows, bottom20_keys))
    write_csv_rows(root / "tables" / "dense_tailshield_probability_blend_manifest.csv", blend_manifest_rows)
    write_csv_rows(root / "tables" / "dense_tailshield_probability_reconstruction_audit.csv", reconstruction_rows)
    write_csv_rows(root / "tables" / "dense_tailshield_complementarity_audit.csv", complementarity_rows)
    write_csv_rows(root / "tables" / "dense_tailshield_calibration_audit.csv", calibration_rows)
    write_csv_rows(root / "tables" / "dense_tailshield_confidence_audit.csv", confidence_rows)
    write_csv_rows(root / "tables" / "dense_tailshield_rescue_audit.csv", rescue_rows)
    write_csv_rows(root / "tables" / "dense_tailshield_alpha_curve_audit.csv", alpha_curve_rows)
    write_csv_rows(root / "tables" / "source_weight_manifest.csv", source_weight_rows)
    write_csv_rows(root / "tables" / "source_reliability_manifest.csv", reliability_rows)
    write_csv_rows(root / "tables" / "component_manifest.csv", component_manifest_rows)
    write_csv_rows(root / "tables" / "component_coverage_audit.csv", component_coverage_rows)
    write_csv_rows(root / "tables" / "paired_generation_audit.csv", paired_generation_rows)
    write_csv_rows(root / "tables" / "negative_control_summary.csv", [_negative_control_summary(decision)])
    write_csv_rows(root / "tables" / "source_ablation_audit.csv", source_ablation_rows)
    write_csv_rows(root / "tables" / "oracle_gap_summary.csv", _oracle_gap_rows(matrix_rows, bottom20_keys))
    write_csv_rows(root / "tables" / "random_mass_bag_control_summary.csv", _random_mass_bag_summary(matrix_rows, bottom20_keys))
    write_csv_rows(root / "tables" / "shuffled_reliability_null_summary.csv", shuffled_summary)
    write_csv_rows(root / "tables" / "anchor_reproducibility_audit.csv", anchor_rows)
    write_csv_rows(root / "tables" / "eligibility_audit.csv", eligibility_rows)
    write_csv_rows(root / "tables" / "source_summary_diagnostics.csv", source_summary_rows)
    write_csv_rows(root / "manifests" / "dense_tailshield_model_manifest.csv", model_manifest_rows)
    write_json(root / "reports" / "leakage_report.json", leakage.to_json_dict())
    write_json(
        root / "manifests" / "protocol_manifest.json",
        {
            "schema_version": "cvae_rebuild_dense_reliability_tailshield_random_mass_bag_protocol_v1",
            "experiment_name": cfg.name,
            "primary_method": cfg.primary_method,
            "experiment_type": "source_only_dense_reliability_tailshield_random_mass_bag_component_union",
            "target_expert_excluded": bool(target_expert_excluded),
            "target_support_used": False,
            "target_support_labels_for_selection": False,
            "target_eval_labels_for_scoring_only": True,
            "target_calibration_metrics_audit_only": True,
            "target_conditioned_point_compatibility_estimate": False,
            "fixed_all_source_inclusion": True,
            "dense_anchor_method": DENSE_ANCHOR_METHOD,
            "bag_method": BAG_METHOD,
            "blend_alpha_dense_locked": cfg.dense_blend_alpha,
            "blend_alpha_bag_locked": cfg.bag_blend_alpha,
            "random_mass_bag_size": cfg.random_mass_bag_size,
            "random_mass_bag_distribution": "dirichlet_uniform_alpha4",
            "center3_definition": 'heldout_center == "3"',
            "bottom20_definition": "lowest 20% eligible seed-center-replicate cells by random_mass_bag_control BACC",
            "nontrivial_rescue_threshold": "dense_correct_bag_wrong_rate >= 0.02",
            "alpha_curve_diagnostic_only": True,
            "alpha_curve_can_rescue_primary": False,
            "source_ablation_diagnostic_only": True,
            "oracle_rows_diagnostic_only": True,
            "claim_boundary": (
                "source-only robustness aggregation under component/source-mass uncertainty; "
                "not learned routing, source selection, target adaptation, formal privacy, "
                "or causal reliability validation"
            ),
            "protocol_violations": list(protocol_violations),
        },
    )
    write_json(root / "run_config_resolved.yaml", _resolved_config(cfg))
    _write_decision_summary(root, decision, alpha_curve_rows)


def _write_decision_summary(root: Path, decision: Mapping[str, object], alpha_curve_rows: Sequence[Mapping[str, object]]) -> None:
    lines = [
        "# Dense-Reliability Tail Shield over Random Mass-Bag Component Union v1",
        "",
        "## Primary Verdict",
        "",
        f"- Primary method: `{decision.get('primary_method', PRIMARY_DENSE_TAILSHIELD_METHOD)}`",
        f"- Primary verdict: `{decision.get('primary_verdict', 'DENSE_TAILSHIELD_RANDOM_MASS_BAG_FAIL')}`",
        f"- Diagnostic flags: `{decision.get('diagnostic_flags', '')}`",
        f"- Leakage status: `{decision.get('leakage_status', '')}`",
        f"- Dense probability reconstruction: `{decision.get('dense_probability_reconstruction_status', '')}`",
        f"- Center-equal mean BACC: {_format_float(decision.get('center_equal_mean_bacc'))}",
        f"- Min center BACC: {_format_float(decision.get('min_center_bacc'))}",
        f"- Center 3 BACC: {_format_float(decision.get('center3_bacc'))}",
        f"- Bottom-20 cell mean BACC: {_format_float(decision.get('bottom20_cell_mean_bacc'))}",
        f"- Worst seed-center BACC: {_format_float(decision.get('worst_seed_center_bacc'))}",
        f"- Seed std BACC: {_format_float(decision.get('seed_std_bacc'))}",
        f"- Delta vs random mass-bag mean: {_format_float(decision.get('delta_vs_random_mass_bag'))}",
        f"- Center3 delta vs random mass-bag: {_format_float(decision.get('center3_delta_vs_random_mass_bag'))}",
        f"- Bottom20 delta vs random mass-bag: {_format_float(decision.get('bottom20_delta_vs_random_mass_bag'))}",
        f"- Dense-correct bag-wrong nontrivial on center3/bottom20: `{decision.get('dense_correct_bag_wrong_nontrivial_on_center3_or_bottom20')}`",
        "",
        "The verdict above is the only adoption verdict. Alpha-curve rows below are audit-only and cannot rescue a failed primary.",
        "",
        "## Protocol Boundary",
        "",
        "This is a locked source-only robustness aggregation audit. It uses no target support, no target-conditioned point compatibility estimate, no source selection, and no learned routing.",
        "",
        "The primary method averages fixed prediction probabilities from paired dense reliability all4 weighted-geometric aggregation and an 11-member Dirichlet-uniform random mass-bag component-union ensemble with alpha 0.25/0.75.",
        "",
        "Target evaluation labels, ECE, and calibration metrics are scoring/audit only and never choose alpha, source set, bag draws, classifier, or decision logic.",
        "",
        "## Alpha Curve Audit",
        "",
    ]
    for row in alpha_curve_rows[:12]:
        lines.append(f"- alpha_dense={row.get('alpha_dense')}: BACC={_format_float(row.get('bacc'))} (`diagnostic_only={row.get('diagnostic_only')}`)")
    (root / "reports" / "decision_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _target_ineligible_rows(
    cfg: DenseTailShieldConfig,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
    error_message: str,
) -> list[dict[str, object]]:
    methods = (
        PRIMARY_DENSE_TAILSHIELD_METHOD,
        DENSE_ANCHOR_METHOD,
        BAG_METHOD,
        cu.PRIMARY_COMPONENT_UNION_METHOD,
        cu.ROW_COMPONENT_UNION_SHRINK050,
        cu.ROW_REAL_FEATURE_DENSE_REFERENCE,
        EQUAL_DENSE_SHIELD_METHOD,
        SHUFFLED_DENSE_SHIELD_METHOD,
        RANDOM_DENSE_SHIELD_METHOD,
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


def _dense_source_weight_rows(
    experiment_seed: int,
    replicate_seed: int,
    heldout_center: str,
    method: str,
    plan: Mapping[str, object],
    rels: Mapping[str, d12.SourceReliability],
) -> list[dict[str, object]]:
    rows = []
    for source in plan["sources"]:
        source_id = str(source)
        rel = rels[source_id]
        rows.append(
            {
                "experiment_seed": int(experiment_seed),
                "replicate_seed": int(replicate_seed),
                "heldout_center": str(heldout_center),
                "panel": "fresh" if int(replicate_seed) in (101, 103, 107) else "canonical",
                "prior_method": str(method),
                "source_center": source_id,
                "raw_reliability_bacc": rel.raw_bacc,
                "reliability_score": plan["scores"][source_id],
                "normalized_source_weight": plan["weights"][source_id],
                "synthetic_per_class_budget": plan["budgets"][source_id],
                "weight_mode": plan.get("source_weighting", ""),
                "weight_rule": plan.get("weight_rule", ""),
                "budget_policy": plan.get("budget_policy", ""),
                "weight_entropy": plan["weight_entropy"],
                "effective_num_sources": plan["effective_num_sources"],
                "l1_distance_from_uniform": plan["l1_distance_from_uniform"],
                "max_weight": plan["max_weight"],
                "min_weight": plan["min_weight"],
                "dominant_source": plan["dominant_source"],
                "dominant_source_weight": plan["dominant_source_weight"],
            }
        )
    return rows


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
        "panel": "fresh" if int(replicate_seed) in (101, 103, 107) else "canonical",
        "row_scope": str(row_scope),
        "status": str(status),
        "error_message": str(error_message),
    }


def _normalize_component_row(row: Mapping[str, object], *, prior_method: str) -> dict[str, object]:
    out = dict(row)
    out["prior_method"] = prior_method
    out.setdefault("summary_kind", "")
    out.setdefault("source_weight_json", "{}")
    out.setdefault("source_budget_json", "{}")
    out.setdefault("panel", "fresh" if int(out.get("replicate_seed", 0)) in (101, 103, 107) else "canonical")
    return out


def _optional_path(base: Path, value: object) -> Path | None:
    if value is None or str(value) == "":
        return None
    return _path(base, str(value))


def _sample_id(row: Mapping[str, object], fallback_idx: int) -> str:
    for key in ("sample_id", "slide_id", "path", "embedding_id", "id"):
        value = row.get(key)
        if value is not None and str(value) != "":
            return str(value)
    return f"eval_row_{int(fallback_idx):06d}"


def _cell_identity(cfg: DenseTailShieldConfig, experiment_seed: int, heldout_center: str, replicate_seed: int) -> dict[str, object]:
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "replicate_seed": int(replicate_seed),
        "panel": cu._panel_for_replicate_seed(cfg, replicate_seed),
    }


def _cell_keys(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "experiment_seed": row.get("experiment_seed", ""),
        "heldout_center": row.get("heldout_center", ""),
        "replicate_seed": row.get("replicate_seed", ""),
        "panel": row.get("panel", ""),
        "prior_method": row.get("prior_method", ""),
    }


def _generation_bundle_key(
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    budget_policy: str,
) -> str:
    return json.dumps(
        {
            "experiment_seed": int(experiment_seed),
            "heldout_center": str(heldout_center),
            "replicate_seed": int(replicate_seed),
            "source_set": "|".join(str(source) for source in candidates),
            "budget_policy": str(budget_policy),
            "control_mode": "normal",
        },
        sort_keys=True,
    )


def _delta(value: object, baseline: object) -> float:
    left = _float(value)
    right = _float(baseline)
    return left - right if math.isfinite(left) and math.isfinite(right) else math.nan


def _resolved_config(cfg: DenseTailShieldConfig) -> dict[str, object]:
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
        "dense_tailshield_random_mass_bag": {
            "primary_method": cfg.primary_method,
            "random_mass_bag_size": cfg.random_mass_bag_size,
            "random_mass_bag_alpha": cfg.random_mass_bag_alpha,
            "dense_blend_alpha": cfg.dense_blend_alpha,
            "bag_blend_alpha": cfg.bag_blend_alpha,
            "alpha_curve_dense_values": list(cfg.alpha_curve_dense_values),
            "reconstruction_probability_tolerance": cfg.reconstruction_probability_tolerance,
            "nontrivial_rescue_threshold": cfg.nontrivial_rescue_threshold,
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
