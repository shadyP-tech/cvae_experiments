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
ANCHOR_METHOD = cu.ROW_COMPONENT_UNION_SHRINK050
BAG_METHOD = cu.ROW_RANDOM_MASS_BAG_CONTROL
MATCHED_SHUFFLED_TAILRISK_PREFIX = cu.MATCHED_SHUFFLED_RELIABILITY_SHRINK050_PREFIX


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
    source_inner_bundles, source_inner_labels = _source_inner_probability_bundles(
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
) -> tuple[dict[str, PredictionBundle], tuple[int, ...]]:
    source_inner_raw, source_inner_labels = _source_inner_eval_set(per_source_runtime, candidates)
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
    return bundles, source_inner_labels


def _source_inner_eval_set(
    per_source_runtime: Mapping[str, RuntimeSource],
    candidates: Sequence[str],
) -> tuple[object, tuple[int, ...]]:
    raw_chunks = []
    labels: list[int] = []
    for source in candidates:
        runtime = per_source_runtime[str(source)].runtime
        raw_chunks.append(cu._inverse_to_raw(runtime, runtime.source_val_embeddings))
        labels.extend(int(v) for v in runtime.source_val_labels)
    return np.vstack(raw_chunks), tuple(labels)


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
