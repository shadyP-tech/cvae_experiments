from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .downstream import evaluate_probability_predictions, weighted_arithmetic_probability_pool
from .features import load_feature_cache, select_rows
from .metrics import nanmean, spearman
from .preservation import _hash_array
from .preservation_repair import (
    NA,
    POOL_PER_SOURCE,
    POOL_SOURCE_UNION,
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
)
from .preservation_sampling import DIAGNOSTIC_SELECTION, PRIMARY_SELECTION, UNION_VARIANT, RuntimeSource, _manifest_row, _per_source_variant, _runtime_source
from .protocol import ProtocolError, assert_candidate_pool, assert_support_eval_disjoint, build_leakage_report
from .reporting import prepare_artifact_dirs, write_csv_rows, write_json
from .splits import candidate_experts
from .support_nelbo import SupportScore, calibrate, rank_support_scores

from . import decentralized_adaptive_gmm_prior as d1a
from . import decentralized_component_union_prior as cu
from . import decentralized_k16_gmm_prior as d1
from . import decentralized_reliability_weighted_gmm_prior as d12
from . import decentralized_support_nelbo_reliability_gmm_prior as snr


SUPPORT_CALIBRATED_COMPONENT_UNION_NAME = "virchow2_cvae_support8_calibrated_component_union_prior_v1"
PRIMARY_SUPPORT_CALIBRATED_COMPONENT_UNION_METHOD = "support8_calibrated_component_union_softmax_shrink050"
ROW_SUPPORT16_PRIMARY_EVAL = "support16_calibrated_component_union_softmax_shrink050_primary_eval_diagnostic"
ROW_SUPPORT32_PRIMARY_EVAL = "support32_calibrated_component_union_softmax_shrink050_primary_eval_diagnostic"
ROW_SUPPORT8_FIXED_EVAL = "support8_calibrated_component_union_softmax_shrink050_fixed_eval_diagnostic"
ROW_SUPPORT16_FIXED_EVAL = "support16_calibrated_component_union_softmax_shrink050_fixed_eval_diagnostic"
ROW_SUPPORT32_FIXED_EVAL = "support32_calibrated_component_union_softmax_shrink050_fixed_eval_diagnostic"
ROW_MATCHED_SHUFFLED_SUPPORT_PREFIX = "support8_calibrated_component_union_shuffled_support_shrink050_perm"
ROW_RANDOM_MASS_BAG_CONTROL = "support_calibrated_component_union_random_mass_bag_control"
ROW_SINGLE_SOURCE_ALIGNMENT = "support_calibrated_component_union_single_source_alignment_reference"
ROW_REAL_FEATURE_DENSE_REFERENCE = "real_source_embedding_classifier_dense_support_calibrated_reference"
ROW_SOURCE_UNION_K16_REFERENCE = "source_union_cc_diag_gmm_k16_prior_sample_support_calibrated_reference"
ROW_UNIFORM_COMPONENT_UNION = cu.PRIMARY_COMPONENT_UNION_METHOD
ROW_RELIABILITY_SHRINK050 = cu.ROW_COMPONENT_UNION_SHRINK050
PROTOCOL_WORDING = (
    "This is a target-support compatibility calibration audit. It uses an unlabeled, disjoint target "
    "support set to estimate source compatibility and held-out target evaluation labels only for final scoring. "
    "It is not source-only, not metadata routing, not hard expert selection, and not a formal privacy claim."
)


@dataclass(frozen=True)
class SupportCalibratedComponentUnionConfig:
    name: str
    artifact_root: Path
    repair_artifact_root: Path
    source_union_gmm_artifact_root: Path | None
    balanced_gmm_artifact_root: Path | None
    feature_cache_root: Path
    backbone: str
    experiment_seeds: tuple[int, ...]
    heldout_centers: tuple[str, ...]
    replicate_seeds: tuple[int, ...]
    support_seeds: tuple[int, ...]
    strict_full_run_matrix: bool
    support_size: int
    support_size_diagnostics: tuple[int, ...]
    nested_support_max_size: int
    synthetic_per_class_total: int
    budget_diagnostic_per_class_total: int | None
    min_per_source_per_class: int
    primary_variant: str
    primary_method: str
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
    support_nelbo_tau: float
    support_shrink_lambda: float
    reliability_floor_score: float
    shrink_lambdas: tuple[float, ...]
    primary_shrink_lambda: float | None
    matched_shuffled_support_null_permutations: int
    matched_shuffled_reliability_null_permutations: int
    random_mass_bag_control_size: int
    paired_dense_artifact_root: Path | None
    anchor_repro_tolerance: float
    prototype_candidate_counts_per_source_class: tuple[int, ...]
    prototype_min_samples_per_component: int
    prototype_variance_floor: float
    classifier_type: str
    classifier_solver: str
    classifier_c: float
    classifier_max_iter: int
    classifier_class_weight: str
    classifier_seed: int | None

    @property
    def max_local_gmm_components_per_source_class(self) -> int:
        return max(self.candidate_components_per_source_class)

    @property
    def composed_components_per_class_nominal(self) -> int:
        return self.max_local_gmm_components_per_source_class * (len(self.heldout_centers) - 1)

    @property
    def fresh_replicate_seeds(self) -> tuple[int, ...]:
        return tuple()

    @property
    def all_replicate_seeds(self) -> tuple[int, ...]:
        return self.replicate_seeds


@dataclass(frozen=True)
class NestedSupportEvalSplit:
    heldout_center: str
    support_size: int
    support_seed: int
    eval_mode: str
    support_indices: tuple[int, ...]
    eval_indices: tuple[int, ...]
    support_sample_ids: tuple[str, ...]
    eval_sample_ids: tuple[str, ...]
    support_eval_split_id: str
    parent_support32_split_id: str
    support_labels_used: bool = False


def load_support_calibrated_component_union_config(path: str | Path) -> SupportCalibratedComponentUnionConfig:
    source = Path(path).resolve()
    data = _load_mapping(source)
    base_dir = source.parents[2] if len(source.parents) >= 3 else source.parent
    return parse_support_calibrated_component_union_config(data, base_dir=base_dir)


def parse_support_calibrated_component_union_config(
    data: Mapping[str, Any],
    *,
    base_dir: str | Path = ".",
) -> SupportCalibratedComponentUnionConfig:
    base = Path(base_dir)
    experiment = _mapping(data, "experiment")
    inputs = _mapping(data, "inputs")
    run = _mapping(data, "run_matrix")
    generation = _mapping(data, "generation")
    prior = _mapping(data, "support_calibrated_component_union_prior")
    classifier = _mapping(data, "classifier")
    cfg = SupportCalibratedComponentUnionConfig(
        name=str(experiment["name"]),
        artifact_root=_path(base, str(experiment["artifact_root"])),
        repair_artifact_root=_path(base, str(inputs["repair_artifact_root"])),
        source_union_gmm_artifact_root=_optional_path(base, inputs.get("source_union_gmm_artifact_root")),
        balanced_gmm_artifact_root=_optional_path(base, inputs.get("balanced_gmm_artifact_root")),
        paired_dense_artifact_root=_optional_path(base, inputs.get("paired_dense_artifact_root")),
        feature_cache_root=_path(base, str(inputs["feature_cache_root"])),
        backbone=str(inputs.get("backbone", "")),
        experiment_seeds=tuple(int(v) for v in run["experiment_seeds"]),
        heldout_centers=tuple(str(v) for v in run["heldout_centers"]),
        replicate_seeds=tuple(int(v) for v in run["replicate_seeds"]),
        support_seeds=tuple(int(v) for v in run["support_seeds"]),
        strict_full_run_matrix=bool(run.get("strict_full_run_matrix", False)),
        support_size=int(run["support_size"]),
        support_size_diagnostics=tuple(int(v) for v in run["support_size_diagnostics"]),
        nested_support_max_size=int(run["nested_support_max_size"]),
        synthetic_per_class_total=int(generation["synthetic_per_class_total"]),
        budget_diagnostic_per_class_total=None,
        min_per_source_per_class=int(generation["min_per_source_per_class"]),
        primary_variant=str(experiment["primary_variant"]),
        primary_method=str(prior["primary_method"]),
        candidate_components_per_source_class=tuple(int(v) for v in prior["candidate_components_per_source_class"]),
        min_samples_per_component=int(prior["min_samples_per_component"]),
        source_weighting=str(prior["source_weighting"]),
        gmm_covariance_type=str(prior["gmm_covariance_type"]),
        gmm_reg_covar=float(prior["gmm_reg_covar"]),
        gmm_n_init=int(prior["gmm_n_init"]),
        gmm_max_iter=int(prior["gmm_max_iter"]),
        min_component_weight=float(prior["min_component_weight"]),
        variance_floor=float(prior["variance_floor"]),
        variance_ceiling_multiplier=float(prior["variance_ceiling_multiplier"]),
        primary_pooling=str(prior["primary_pooling"]),
        support_nelbo_tau=float(prior["support_nelbo_tau"]),
        support_shrink_lambda=float(prior["support_shrink_lambda"]),
        reliability_floor_score=float(prior["reliability_floor_score"]),
        shrink_lambdas=tuple(float(v) for v in prior.get("shrink_lambdas", (0.25, 0.5))),
        primary_shrink_lambda=float(prior["support_shrink_lambda"]),
        matched_shuffled_support_null_permutations=int(prior["matched_shuffled_support_null_permutations"]),
        matched_shuffled_reliability_null_permutations=0,
        random_mass_bag_control_size=int(prior["random_mass_bag_control_size"]),
        anchor_repro_tolerance=float(prior.get("anchor_repro_tolerance", 1.0e-4)),
        prototype_candidate_counts_per_source_class=tuple(int(v) for v in prior.get("prototype_candidate_counts_per_source_class", prior["candidate_components_per_source_class"])),
        prototype_min_samples_per_component=int(prior.get("prototype_min_samples_per_component", prior["min_samples_per_component"])),
        prototype_variance_floor=float(prior.get("prototype_variance_floor", prior["variance_floor"])),
        classifier_type=str(classifier["type"]),
        classifier_solver=str(classifier["solver"]),
        classifier_c=float(classifier["C"]),
        classifier_max_iter=int(classifier["max_iter"]),
        classifier_class_weight=str(classifier["class_weight"]),
        classifier_seed=None if classifier.get("classifier_seed") is None else int(classifier["classifier_seed"]),
    )
    validate_support_calibrated_component_union_config(cfg)
    return cfg


def validate_support_calibrated_component_union_config(cfg: SupportCalibratedComponentUnionConfig) -> None:
    if cfg.name != SUPPORT_CALIBRATED_COMPONENT_UNION_NAME:
        raise ProtocolError(f"Support-calibrated component-union name must be {SUPPORT_CALIBRATED_COMPONENT_UNION_NAME!r}.")
    if cfg.backbone != "virchow2":
        raise ProtocolError("Support-calibrated component-union is locked to backbone=virchow2.")
    if cfg.primary_variant != PRIMARY_VARIANT:
        raise ProtocolError(f"primary_variant must be {PRIMARY_VARIANT!r}.")
    if cfg.primary_method != PRIMARY_SUPPORT_CALIBRATED_COMPONENT_UNION_METHOD:
        raise ProtocolError(f"primary_method must be {PRIMARY_SUPPORT_CALIBRATED_COMPONENT_UNION_METHOD!r}.")
    if cfg.strict_full_run_matrix:
        if cfg.experiment_seeds != (42, 43, 44):
            raise ProtocolError("strict_full_run_matrix requires experiment_seeds=[42, 43, 44].")
        if cfg.heldout_centers != ("0", "1", "2", "3", "4"):
            raise ProtocolError("strict_full_run_matrix requires heldout_centers=['0', '1', '2', '3', '4'].")
        if cfg.replicate_seeds != (17, 23, 31) or cfg.support_seeds != (17, 23, 31):
            raise ProtocolError("strict_full_run_matrix requires support/replicate seeds [17, 23, 31].")
    if cfg.replicate_seeds != cfg.support_seeds:
        raise ProtocolError("support_seeds must equal replicate_seeds for paired support/generation cells.")
    if cfg.support_size != 8:
        raise ProtocolError("Primary support_size must be locked to 8.")
    if cfg.support_size_diagnostics != (16, 32):
        raise ProtocolError("support_size_diagnostics must be locked to [16, 32].")
    if cfg.nested_support_max_size != 32:
        raise ProtocolError("nested_support_max_size must be locked to 32.")
    if cfg.strict_full_run_matrix:
        if cfg.synthetic_per_class_total != 128:
            raise ProtocolError("strict_full_run_matrix requires synthetic_per_class_total=128.")
        if cfg.min_per_source_per_class != 8:
            raise ProtocolError("strict_full_run_matrix requires min_per_source_per_class=8.")
    if cfg.synthetic_per_class_total < 1 or cfg.min_per_source_per_class < 1:
        raise ProtocolError("Synthetic budget and min_per_source_per_class must be positive.")
    if (len(cfg.heldout_centers) - 1) * cfg.min_per_source_per_class > cfg.synthetic_per_class_total:
        raise ProtocolError("min_per_source_per_class cannot exceed the per-class source budget.")
    if cfg.candidate_components_per_source_class != (4, 3, 2, 1):
        raise ProtocolError("candidate_components_per_source_class must be locked to [4, 3, 2, 1].")
    if cfg.min_samples_per_component != 12:
        raise ProtocolError("min_samples_per_component must be locked to 12.")
    if cfg.source_weighting != "support_calibrated_component_union_softmax_shrink050":
        raise ProtocolError("source_weighting must be support_calibrated_component_union_softmax_shrink050.")
    if cfg.gmm_covariance_type != "diag":
        raise ProtocolError("gmm_covariance_type must be diag.")
    if cfg.primary_pooling != "pooled_raw_logistic":
        raise ProtocolError("primary_pooling must be pooled_raw_logistic.")
    if cfg.min_component_weight != 0.02:
        raise ProtocolError("min_component_weight must be locked to 0.02.")
    if cfg.variance_floor != 1.0e-5:
        raise ProtocolError("variance_floor must be locked to 1e-5.")
    if cfg.variance_ceiling_multiplier != 16.0:
        raise ProtocolError("variance_ceiling_multiplier must be locked to 16.0.")
    if cfg.support_nelbo_tau != 1.0:
        raise ProtocolError("support_nelbo_tau must be locked to 1.0.")
    if cfg.support_shrink_lambda != 0.5:
        raise ProtocolError("support_shrink_lambda must be locked to 0.50.")
    if cfg.matched_shuffled_support_null_permutations < 1:
        raise ProtocolError("matched_shuffled_support_null_permutations must be positive.")
    if cfg.strict_full_run_matrix and cfg.matched_shuffled_support_null_permutations != 20:
        raise ProtocolError("strict_full_run_matrix requires matched_shuffled_support_null_permutations=20.")
    if cfg.random_mass_bag_control_size < 1:
        raise ProtocolError("random_mass_bag_control_size must be positive.")
    if cfg.strict_full_run_matrix and cfg.random_mass_bag_control_size != 11:
        raise ProtocolError("strict_full_run_matrix requires random_mass_bag_control_size=11.")
    if min(
        cfg.gmm_reg_covar,
        cfg.min_component_weight,
        cfg.variance_floor,
        cfg.variance_ceiling_multiplier,
        cfg.reliability_floor_score,
        cfg.support_nelbo_tau,
        cfg.support_shrink_lambda,
    ) <= 0.0:
        raise ProtocolError("Support-calibrated component-union numeric floors must be positive.")
    if cfg.classifier_type != "sklearn_logistic_regression":
        raise ProtocolError("classifier.type must be sklearn_logistic_regression.")
    if cfg.classifier_solver != "lbfgs" or cfg.classifier_c != 1.0 or cfg.classifier_max_iter != 2000:
        raise ProtocolError("Classifier solver/C/max_iter must remain locked.")
    if cfg.classifier_class_weight != "balanced" or cfg.classifier_seed is not None:
        raise ProtocolError("Classifier must use class_weight=balanced and classifier_seed=null.")


def run_support_calibrated_component_union_prior(
    cfg: SupportCalibratedComponentUnionConfig,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    root = prepare_artifact_dirs(Path(artifact_root) if artifact_root is not None else cfg.artifact_root)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "summaries").mkdir(parents=True, exist_ok=True)

    matrix_rows: list[dict[str, object]] = []
    component_manifest_rows: list[dict[str, object]] = []
    component_coverage_rows: list[dict[str, object]] = []
    source_summary_rows: list[dict[str, object]] = []
    source_weight_rows: list[dict[str, object]] = []
    support_weight_rows: list[dict[str, object]] = []
    support_score_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    paired_generation_rows: list[dict[str, object]] = []
    mass_alignment_rows: list[dict[str, object]] = []
    real_feature_rows: list[dict[str, object]] = []
    model_manifest_rows: list[dict[str, object]] = []
    eligibility_rows: list[dict[str, object]] = []
    protocol_violations: list[str] = []
    target_expert_excluded = True

    source_union_refs = d1._load_reference_values(
        cfg.source_union_gmm_artifact_root,
        table_name="gmm_prior_gap_summary.csv",
        method="source_union_cc_diag_gmm_k16_prior_sample_diagnostic",
        label="source-union K16",
    )
    d1._validate_optional_leakage_report(cfg.source_union_gmm_artifact_root, protocol_violations)

    repair_cfg = d1._repair_runtime_config(cfg, root)
    per_source_variant = _per_source_variant()

    try:
        for experiment_seed in cfg.experiment_seeds:
            train_cache = load_feature_cache(_existing_cache_path(cfg.feature_cache_root, seed=experiment_seed, split="train"))
            test_cache = load_feature_cache(_existing_cache_path(cfg.feature_cache_root, seed=experiment_seed, split="test"))
            per_source_runtime: dict[str, RuntimeSource] = {}
            gmm_summaries: dict[tuple[str, int], d1a.AdaptiveSourceLocalSummary] = {}
            component_details: dict[tuple[str, int, int], dict[str, object]] = {}
            support_calibration: dict[str, object] = {}

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
                support_calibration[str(source_center)] = snr._source_nelbo_calibration(runtime_source.runtime, str(source_center))

                summaries, detail_rows = cu._fit_and_export_pruned_gmm_summaries(
                    cfg,
                    root,
                    runtime_source.runtime,
                    experiment_seed=int(experiment_seed),
                    shuffled_label_control=False,
                )
                for summary in summaries:
                    gmm_summaries[(summary.source_center, summary.class_label)] = summary
                    source_summary_rows.append(d1a._summary_diagnostic_row(cfg, summary))
                for row in detail_rows:
                    component_details[(str(row["source_center"]), int(row["class_label"]), int(row["source_component_id"]))] = row
                component_manifest_rows.extend(detail_rows)

            reliability: dict[tuple[int, int, str], d12.SourceReliability] = {}
            for replicate_seed in cfg.replicate_seeds:
                for source_center in cfg.heldout_centers:
                    rel = d12._source_local_reliability(
                        cfg,
                        per_source_runtime=per_source_runtime,
                        summaries=gmm_summaries,
                        test_cache=test_cache,
                        experiment_seed=int(experiment_seed),
                        replicate_seed=int(replicate_seed),
                        source_center=str(source_center),
                    )
                    reliability[(int(experiment_seed), int(replicate_seed), str(source_center))] = rel

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

                for replicate_seed in cfg.replicate_seeds:
                    support_seed = int(replicate_seed)
                    rels = {
                        source: reliability[(int(experiment_seed), int(replicate_seed), str(source))]
                        for source in candidates
                    }
                    splits = nested_unlabeled_support_eval_splits(
                        test_cache.metadata,
                        heldout_center=str(heldout_center),
                        support_seed=support_seed,
                        support_sizes=(cfg.support_size, *cfg.support_size_diagnostics),
                        max_support_size=cfg.nested_support_max_size,
                    )
                    split_by_key = {(split.support_size, split.eval_mode): split for split in splits}
                    split_rows.extend(_split_manifest_rows(splits, experiment_seed, replicate_seed))
                    support_scores_by_size: dict[int, tuple[SupportScore, ...]] = {}
                    support_plans_by_size: dict[int, dict[str, object]] = {}
                    support_raw_by_size: dict[int, object] = {}
                    for support_size in (cfg.support_size, *cfg.support_size_diagnostics):
                        split = split_by_key[(int(support_size), "primary_style")]
                        support_raw, _support_meta = select_rows(test_cache.embeddings, test_cache.metadata, split.support_indices)
                        support_raw_by_size[int(support_size)] = support_raw
                        scores = _support_scores(
                            cfg,
                            per_source_runtime=per_source_runtime,
                            support_calibration=support_calibration,
                            candidates=candidates,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            support_seed=support_seed,
                            support_size=int(support_size),
                            support_raw=support_raw,
                        )
                        ranked = tuple(rank_support_scores(scores, eligible_count=len(candidates)))
                        plan = _support_shrink_plan(
                            cfg,
                            candidates,
                            ranked,
                            total=cfg.synthetic_per_class_total,
                        )
                        support_scores_by_size[int(support_size)] = ranked
                        support_plans_by_size[int(support_size)] = plan
                        support_score_rows.extend(_support_score_manifest_rows(ranked, split))
                        support_weight_rows.extend(_support_weight_manifest_rows(experiment_seed, replicate_seed, heldout_center, split, plan))

                    primary_plan = support_plans_by_size[cfg.support_size]
                    uniform_plan = _uniform_source_plan(cfg, candidates, total=cfg.synthetic_per_class_total)
                    shrink050_plan = _reliability_shrink_plan(cfg, candidates, rels, shrink_lambda=0.5, total=cfg.synthetic_per_class_total)
                    component_manifest_rows.extend(
                        cu._fold_component_manifest_rows(
                            cfg,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            candidates=candidates,
                            summaries=gmm_summaries,
                            component_details=component_details,
                            weight_plan=primary_plan,
                        )
                    )

                    primary_split = split_by_key[(cfg.support_size, "primary_style")]
                    eval_contexts = _evaluation_contexts(cfg, split_by_key)
                    for eval_context in eval_contexts:
                        split = eval_context["split"]
                        support_size = int(split.support_size)
                        plan = support_plans_by_size[support_size]
                        method = str(eval_context["method"])
                        source_weight_rows.extend(
                            _source_weight_manifest_rows(
                                experiment_seed,
                                replicate_seed,
                                heldout_center,
                                split,
                                method,
                                plan,
                                rels,
                            )
                        )
                        eval_raw, eval_meta = select_rows(test_cache.embeddings, test_cache.metadata, split.eval_indices)
                        eval_labels = tuple(_label(row) for row in eval_meta)
                        eval_error = "mono_class_target_eval_after_support_split" if len(set(eval_labels)) < 2 else ""
                        su_ref = d1._reference_for_cell(source_union_refs, experiment_seed, heldout_center, replicate_seed)
                        if eval_error:
                            row = _support_empty_row(
                                cfg,
                                experiment_seed=int(experiment_seed),
                                heldout_center=str(heldout_center),
                                replicate_seed=int(replicate_seed),
                                candidates=candidates,
                                prior_method=method,
                                source_union_ref=su_ref,
                                status="ineligible",
                                error_message=eval_error,
                                claim_role=str(eval_context["claim_role"]),
                            )
                            matrix_rows.append(_extend_support_row(row, split=split, plan=plan))
                            if method == cfg.primary_method:
                                eligibility_rows.append(_eligibility_row(row, split))
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
                        ref_row = cu._component_extend_row(ref_row, source_weighting="not_applicable")
                        ref_row["prior_method"] = ROW_REAL_FEATURE_DENSE_REFERENCE
                        ref_row = _extend_support_row(ref_row, split=split, plan=None)
                        matrix_rows.append(ref_row)
                        real_feature_rows.append(ref_row)
                        real_feature_bacc = _float(ref_row["bacc"])

                        row, coverage, weak, _nn, paired = cu._evaluate_gmm_component_union(
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
                            center_balanced_ref=d1._missing_reference(),
                            real_feature_bacc=real_feature_bacc,
                            weight_plan=plan,
                            prior_method=method,
                            selection_source=PRIMARY_SELECTION if method == cfg.primary_method else DIAGNOSTIC_SELECTION,
                            claim_role=str(eval_context["claim_role"]),
                        )
                        row = _extend_support_row(row, split=split, plan=plan)
                        matrix_rows.append(row)
                        component_coverage_rows.append(_extend_support_row(coverage, split=split, plan=plan))
                        paired_generation_rows.append(_extend_support_row(paired, split=split, plan=plan))
                        if method == cfg.primary_method:
                            eligibility_rows.append(_eligibility_row(row, split))

                        single_rows = _single_source_alignment_rows(
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
                            real_feature_bacc=real_feature_bacc,
                            split=split,
                        )
                        matrix_rows.extend(single_rows)
                        mass_alignment_rows.extend(_mass_alignment_rows(single_rows, plan, split))

                    eval_raw, eval_meta = select_rows(test_cache.embeddings, test_cache.metadata, primary_split.eval_indices)
                    eval_labels = tuple(_label(row) for row in eval_meta)
                    if len(set(eval_labels)) < 2:
                        continue
                    su_ref = d1._reference_for_cell(source_union_refs, experiment_seed, heldout_center, replicate_seed)
                    real_feature_bacc = _real_feature_bacc_for(matrix_rows, experiment_seed, heldout_center, replicate_seed, primary_split)
                    for method, plan, role in (
                        (ROW_UNIFORM_COMPONENT_UNION, uniform_plan, "diagnostic_uniform_component_union"),
                        (ROW_RELIABILITY_SHRINK050, shrink050_plan, "diagnostic_reliability_shrink050_component_union"),
                    ):
                        source_weight_rows.extend(_source_weight_manifest_rows(experiment_seed, replicate_seed, heldout_center, primary_split, method, plan, rels))
                        row, coverage, _weak, _nn, paired = cu._evaluate_gmm_component_union(
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
                            center_balanced_ref=d1._missing_reference(),
                            real_feature_bacc=real_feature_bacc,
                            weight_plan=plan,
                            prior_method=method,
                            selection_source=DIAGNOSTIC_SELECTION,
                            claim_role=role,
                        )
                        matrix_rows.append(_extend_support_row(row, split=primary_split, plan=plan))
                        component_coverage_rows.append(_extend_support_row(coverage, split=primary_split, plan=plan))
                        paired_generation_rows.append(_extend_support_row(paired, split=primary_split, plan=plan))

                    random_bag_row, random_coverage, random_paired, random_weight_rows = _evaluate_random_mass_bag_control(
                        cfg,
                        root=root,
                        per_source_runtime=per_source_runtime,
                        candidates=candidates,
                        summaries=gmm_summaries,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        replicate_seed=int(replicate_seed),
                        split=primary_split,
                        eval_raw=eval_raw,
                        eval_labels=eval_labels,
                        source_union_ref=su_ref,
                        real_feature_bacc=real_feature_bacc,
                    )
                    matrix_rows.append(random_bag_row)
                    component_coverage_rows.append(random_coverage)
                    paired_generation_rows.append(random_paired)
                    source_weight_rows.extend(random_weight_rows)

                    for permutation_id in range(cfg.matched_shuffled_support_null_permutations):
                        method = _matched_shuffled_support_method(permutation_id)
                        null_plan = _matched_shuffled_support_plan(
                            cfg,
                            candidates,
                            support_scores_by_size[cfg.support_size],
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            support_seed=support_seed,
                            permutation_id=permutation_id,
                            total=cfg.synthetic_per_class_total,
                        )
                        source_weight_rows.extend(_source_weight_manifest_rows(experiment_seed, replicate_seed, heldout_center, primary_split, method, null_plan, rels))
                        row, coverage, _weak, _nn, paired = cu._evaluate_gmm_component_union(
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
                            center_balanced_ref=d1._missing_reference(),
                            real_feature_bacc=real_feature_bacc,
                            weight_plan=null_plan,
                            prior_method=method,
                            selection_source=DIAGNOSTIC_SELECTION,
                            claim_role="matched_shuffled_support_null",
                        )
                        matrix_rows.append(_extend_support_row(row, split=primary_split, plan=null_plan))
                        component_coverage_rows.append(_extend_support_row(coverage, split=primary_split, plan=null_plan))
                        paired_generation_rows.append(_extend_support_row(paired, split=primary_split, plan=null_plan))

                    matrix_rows.append(_extend_support_row(_source_union_reference_row(cfg, experiment_seed, heldout_center, replicate_seed, candidates, su_ref), split=primary_split, plan=None))
    except ProtocolError as exc:
        protocol_violations.append(str(exc))

    leakage = build_leakage_report(
        target_support_labels_for_selection=False,
        target_eval_labels_for_scoring_only=True,
        target_expert_excluded=target_expert_excluded,
        oracle_rows_diagnostic_only=True,
        extra_violations=protocol_violations,
    )
    decision = _decision(matrix_rows, cfg, leakage_status=leakage.status, mass_alignment_rows=mass_alignment_rows)
    _write_artifacts(
        root,
        cfg,
        matrix_rows=matrix_rows,
        component_manifest_rows=component_manifest_rows,
        source_summary_rows=source_summary_rows,
        component_coverage_rows=component_coverage_rows,
        source_weight_rows=source_weight_rows,
        support_weight_rows=support_weight_rows,
        support_score_rows=support_score_rows,
        split_rows=split_rows,
        paired_generation_rows=paired_generation_rows,
        mass_alignment_rows=mass_alignment_rows,
        real_feature_rows=real_feature_rows,
        model_manifest_rows=model_manifest_rows,
        eligibility_rows=eligibility_rows,
        decision=decision,
        leakage_status=leakage.status,
        protocol_violations=protocol_violations,
        target_expert_excluded=target_expert_excluded,
    )
    return root


def nested_unlabeled_support_eval_splits(
    metadata: Sequence[Mapping[str, object]],
    *,
    heldout_center: str,
    support_seed: int,
    support_sizes: Sequence[int],
    max_support_size: int,
    center_key: str = "center",
    sample_id_key: str = "sample_id",
) -> tuple[NestedSupportEvalSplit, ...]:
    target_indices = [
        idx
        for idx, row in enumerate(metadata)
        if _row_center(row, center_key=center_key) == str(heldout_center)
    ]
    if len(target_indices) <= int(max_support_size):
        raise ProtocolError(
            f"Need more than {max_support_size} target samples for nested support/eval split; got {len(target_indices)}."
        )
    rng = random.Random(int(support_seed))
    shuffled = list(target_indices)
    rng.shuffle(shuffled)
    parent_order = tuple(shuffled[: int(max_support_size)])
    parent_support = tuple(sorted(parent_order))
    parent_eval = tuple(sorted(idx for idx in target_indices if idx not in set(parent_support)))
    parent_id = f"target{heldout_center}_seed{support_seed}_nested_unlabeled_k{max_support_size}"
    out: list[NestedSupportEvalSplit] = []
    for support_size in support_sizes:
        support = tuple(sorted(parent_order[: int(support_size)]))
        primary_eval = tuple(sorted(idx for idx in target_indices if idx not in set(support)))
        for eval_mode, eval_indices in (("primary_style", primary_eval), ("fixed_support32", parent_eval)):
            support_ids = tuple(_sample_id(metadata[idx], idx, sample_id_key=sample_id_key) for idx in support)
            eval_ids = tuple(_sample_id(metadata[idx], idx, sample_id_key=sample_id_key) for idx in eval_indices)
            assert_support_eval_disjoint(support_ids, eval_ids)
            out.append(
                NestedSupportEvalSplit(
                    heldout_center=str(heldout_center),
                    support_size=int(support_size),
                    support_seed=int(support_seed),
                    eval_mode=eval_mode,
                    support_indices=support,
                    eval_indices=eval_indices,
                    support_sample_ids=support_ids,
                    eval_sample_ids=eval_ids,
                    support_eval_split_id=f"{parent_id}_support{support_size}_{eval_mode}",
                    parent_support32_split_id=parent_id,
                    support_labels_used=False,
                )
            )
    return tuple(out)


def _support_scores(
    cfg: SupportCalibratedComponentUnionConfig,
    *,
    per_source_runtime: Mapping[str, RuntimeSource],
    support_calibration: Mapping[str, object],
    candidates: Sequence[str],
    experiment_seed: int,
    heldout_center: str,
    support_seed: int,
    support_size: int,
    support_raw: object,
) -> tuple[SupportScore, ...]:
    scores: list[SupportScore] = []
    for source in candidates:
        runtime = per_source_runtime[str(source)].runtime
        vals = snr._marginal_nelbo_values(runtime, support_raw, already_in_frame=False)
        raw = nanmean(vals)
        scores.append(
            SupportScore(
                experiment_seed=int(experiment_seed),
                heldout_center=str(heldout_center),
                support_seed=int(support_seed),
                support_size=int(support_size),
                expert_id=str(source),
                raw_support_nelbo=float(raw),
                calibrated_support_nelbo=float(calibrate(float(raw), support_calibration[str(source)])),
            )
        )
    return tuple(scores)


def _support_shrink_plan(
    cfg: SupportCalibratedComponentUnionConfig,
    sources: Sequence[str],
    ranked_scores: Sequence[SupportScore],
    *,
    total: int,
) -> dict[str, object]:
    sources_tuple = tuple(str(source) for source in sources)
    score_by_source = {row.expert_id: row for row in ranked_scores}
    logits = [-float(score_by_source[source].calibrated_support_nelbo) / float(cfg.support_nelbo_tau) for source in sources_tuple]
    support_values = _softmax(logits)
    support_weights = {source: float(weight) for source, weight in zip(sources_tuple, support_values)}
    uniform = 1.0 / float(len(sources_tuple))
    weights = {
        source: (1.0 - float(cfg.support_shrink_lambda)) * uniform + float(cfg.support_shrink_lambda) * support_weights[source]
        for source in sources_tuple
    }
    return _plan_from_weights(
        cfg,
        sources_tuple,
        weights,
        scores={source: -float(score_by_source[source].calibrated_support_nelbo) for source in sources_tuple},
        total=total,
        mode=f"support_softmax_tau{cfg.support_nelbo_tau:g}_shrink{cfg.support_shrink_lambda:.2f}",
        extra={
            "support_weights": support_weights,
            "raw_support_nelbo": {source: float(score_by_source[source].raw_support_nelbo) for source in sources_tuple},
            "calibrated_support_nelbo": {source: float(score_by_source[source].calibrated_support_nelbo) for source in sources_tuple},
            "support_nelbo_tau": float(cfg.support_nelbo_tau),
            "support_shrink_lambda": float(cfg.support_shrink_lambda),
        },
    )


def _matched_shuffled_support_plan(
    cfg: SupportCalibratedComponentUnionConfig,
    sources: Sequence[str],
    ranked_scores: Sequence[SupportScore],
    *,
    experiment_seed: int,
    heldout_center: str,
    support_seed: int,
    permutation_id: int,
    total: int,
) -> dict[str, object]:
    sources_tuple = tuple(str(source) for source in sources)
    score_by_source = {row.expert_id: row for row in ranked_scores}
    shuffled = list(sources_tuple)
    shuffle_seed = d1._latent_seed(experiment_seed, heldout_center, support_seed, "support_calibrated_shuffled_support", int(permutation_id))
    rng = np.random.default_rng(shuffle_seed)
    rng.shuffle(shuffled)
    pseudo_scores = [
        SupportScore(
            experiment_seed=int(experiment_seed),
            heldout_center=str(heldout_center),
            support_seed=int(support_seed),
            support_size=ranked_scores[0].support_size,
            expert_id=source,
            raw_support_nelbo=float(score_by_source[shuffled[idx]].raw_support_nelbo),
            calibrated_support_nelbo=float(score_by_source[shuffled[idx]].calibrated_support_nelbo),
        )
        for idx, source in enumerate(sources_tuple)
    ]
    plan = _support_shrink_plan(cfg, sources_tuple, pseudo_scores, total=total)
    plan.update(
        {
            "component_union_weight_mode": f"shuffled_support_shrink_{cfg.support_shrink_lambda:.2f}",
            "control_permutation_id": int(permutation_id),
            "shuffle_seed": int(shuffle_seed),
            "shuffle_mapping": {source: shuffled[idx] for idx, source in enumerate(sources_tuple)},
        }
    )
    return plan


def _matched_shuffled_support_method(permutation_id: int) -> str:
    return f"{ROW_MATCHED_SHUFFLED_SUPPORT_PREFIX}{int(permutation_id):03d}"


def _is_matched_shuffled_support_method(method: object) -> bool:
    return str(method).startswith(ROW_MATCHED_SHUFFLED_SUPPORT_PREFIX)


def _uniform_source_plan(
    cfg: SupportCalibratedComponentUnionConfig,
    sources: Sequence[str],
    *,
    total: int,
) -> dict[str, object]:
    sources_tuple = tuple(str(source) for source in sources)
    weights = {source: 1.0 / float(len(sources_tuple)) for source in sources_tuple}
    return _plan_from_weights(cfg, sources_tuple, weights, scores=weights, total=total, mode="uniform_source_mass")


def _reliability_shrink_plan(
    cfg: SupportCalibratedComponentUnionConfig,
    sources: Sequence[str],
    rels: Mapping[str, d12.SourceReliability],
    *,
    shrink_lambda: float,
    total: int,
) -> dict[str, object]:
    sources_tuple = tuple(str(source) for source in sources)
    raw = {source: d12._linear_reliability_score(rels[source].raw_bacc, cfg.reliability_floor_score) for source in sources_tuple}
    raw_total = sum(raw.values())
    reliability = {source: raw[source] / raw_total for source in sources_tuple}
    uniform = 1.0 / float(len(sources_tuple))
    weights = {source: (1.0 - float(shrink_lambda)) * uniform + float(shrink_lambda) * reliability[source] for source in sources_tuple}
    plan = _plan_from_weights(cfg, sources_tuple, weights, scores=raw, total=total, mode=f"reliability_shrink_{float(shrink_lambda):.2f}")
    plan["shrink_lambda"] = float(shrink_lambda)
    return plan


def _random_dirichlet_plan(
    cfg: SupportCalibratedComponentUnionConfig,
    sources: Sequence[str],
    *,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    permutation_id: int,
    total: int,
) -> dict[str, object]:
    sources_tuple = tuple(str(source) for source in sources)
    center = np.full(len(sources_tuple), 1.0 / float(len(sources_tuple)), dtype=float)
    alpha = center * 4.0 * float(len(sources_tuple))
    rng = np.random.default_rng(d1._latent_seed(experiment_seed, heldout_center, replicate_seed, "support_calibrated_random_mass_bag", int(permutation_id)))
    values = rng.dirichlet(alpha)
    weights = {source: float(weight) for source, weight in zip(sources_tuple, values)}
    plan = _plan_from_weights(cfg, sources_tuple, weights, scores=weights, total=total, mode=f"random_mass_bag_uniform_alpha4_perm{int(permutation_id):03d}")
    plan.update({"dirichlet_alpha_per_source": 4.0, "control_permutation_id": int(permutation_id)})
    return plan


def _plan_from_weights(
    cfg: SupportCalibratedComponentUnionConfig,
    sources: Sequence[str],
    weights: Mapping[str, float],
    *,
    scores: Mapping[str, float],
    total: int,
    mode: str,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    sources_tuple = tuple(str(source) for source in sources)
    norm = _normalize_weights(sources_tuple, weights)
    budgets, floor_binding = _constrained_weighted_budgets(int(total), sources_tuple, norm, cfg.min_per_source_per_class)
    plan = cu._with_weight_diagnostics(sources_tuple, norm, budgets, scores, total=total, mode=mode)
    plan.update(
        {
            "pre_round_weight": norm,
            "post_round_weight": {source: float(budgets[source]) / float(total) for source in sources_tuple},
            "floor_binding": floor_binding,
            "floor_binding_count": sum(1 for value in floor_binding.values() if value),
        }
    )
    if extra:
        plan.update(dict(extra))
    return plan


def _constrained_weighted_budgets(
    total: int,
    sources: Sequence[str],
    weights: Mapping[str, float],
    minimum: int,
) -> tuple[dict[str, int], dict[str, bool]]:
    sources_tuple = tuple(str(source) for source in sources)
    if len(sources_tuple) * int(minimum) > int(total):
        raise ProtocolError("Minimum per-source budget exceeds total synthetic budget.")
    exact = {source: float(weights[source]) * float(total) for source in sources_tuple}
    floor_binding = {source: exact[source] < float(minimum) for source in sources_tuple}
    if not any(floor_binding.values()):
        budgets = {source: int(math.floor(exact[source])) for source in sources_tuple}
        leftover = int(total) - sum(budgets.values())
        ordered = sorted(sources_tuple, key=lambda source: (-(exact[source] - math.floor(exact[source])), source))
        for source in ordered[:leftover]:
            budgets[source] += 1
        return budgets, floor_binding

    bound = {source for source, value in floor_binding.items() if value}
    budgets = {source: int(minimum) for source in bound}
    free = [source for source in sources_tuple if source not in bound]
    remaining = int(total) - sum(budgets.values())
    free_weight = sum(float(weights[source]) for source in free)
    if not free:
        return budgets, floor_binding
    free_exact = {source: (float(weights[source]) / free_weight) * float(remaining) for source in free}
    for source in free:
        budgets[source] = max(int(minimum), int(math.floor(free_exact[source])))
    while sum(budgets.values()) > int(total):
        adjustable = sorted((source for source in free if budgets[source] > int(minimum)), key=lambda source: (-budgets[source], source))
        if not adjustable:
            raise ProtocolError("Constrained budget allocation failed under minimum floor.")
        budgets[adjustable[0]] -= 1
    leftover = int(total) - sum(budgets.values())
    ordered = sorted(free, key=lambda source: (-(free_exact[source] - math.floor(free_exact[source])), source))
    idx = 0
    while leftover > 0:
        budgets[ordered[idx % len(ordered)]] += 1
        leftover -= 1
        idx += 1
    if sum(budgets.values()) != int(total):
        raise ProtocolError("Constrained weighted budget allocation failed to sum to total.")
    return budgets, floor_binding


def _normalize_weights(sources: Sequence[str], weights: Mapping[str, float]) -> dict[str, float]:
    values = {str(source): float(weights[str(source)]) for source in sources}
    total = sum(values.values())
    if not math.isfinite(total) or total <= 0.0:
        raise ProtocolError("Source weights must sum to a positive finite value.")
    return {source: value / total for source, value in values.items()}


def _softmax(values: Sequence[float]) -> tuple[float, ...]:
    arr = np.asarray([float(value) for value in values], dtype=float)
    if np.any(~np.isfinite(arr)):
        raise ProtocolError("Support softmax received non-finite values.")
    shifted = arr - float(np.max(arr))
    exp_vals = np.exp(shifted)
    denom = float(exp_vals.sum())
    if not math.isfinite(denom) or denom <= 0.0:
        raise ProtocolError("Support softmax failed.")
    return tuple(float(value / denom) for value in exp_vals)


def _evaluation_contexts(
    cfg: SupportCalibratedComponentUnionConfig,
    split_by_key: Mapping[tuple[int, str], NestedSupportEvalSplit],
) -> list[dict[str, object]]:
    contexts = [
        {
            "split": split_by_key[(cfg.support_size, "primary_style")],
            "method": cfg.primary_method,
            "claim_role": "primary_support8_target_support_calibrated_component_union",
        }
    ]
    for support_size in (cfg.support_size, *cfg.support_size_diagnostics):
        for eval_mode in ("fixed_support32",):
            contexts.append(
                {
                    "split": split_by_key[(int(support_size), eval_mode)],
                    "method": _support_size_method(int(support_size), eval_mode),
                    "claim_role": "diagnostic_support_size_fixed_eval_sensitivity",
                }
            )
        if int(support_size) != cfg.support_size:
            contexts.append(
                {
                    "split": split_by_key[(int(support_size), "primary_style")],
                    "method": _support_size_method(int(support_size), "primary_style"),
                    "claim_role": "diagnostic_support_size_primary_style_sensitivity",
                }
            )
    return contexts


def _support_size_method(support_size: int, eval_mode: str) -> str:
    if int(support_size) == 8 and eval_mode == "fixed_support32":
        return ROW_SUPPORT8_FIXED_EVAL
    if int(support_size) == 16 and eval_mode == "primary_style":
        return ROW_SUPPORT16_PRIMARY_EVAL
    if int(support_size) == 16 and eval_mode == "fixed_support32":
        return ROW_SUPPORT16_FIXED_EVAL
    if int(support_size) == 32 and eval_mode == "primary_style":
        return ROW_SUPPORT32_PRIMARY_EVAL
    if int(support_size) == 32 and eval_mode == "fixed_support32":
        return ROW_SUPPORT32_FIXED_EVAL
    raise ProtocolError(f"Unsupported support-size diagnostic: support_size={support_size}, eval_mode={eval_mode}")


def _evaluate_random_mass_bag_control(
    cfg: SupportCalibratedComponentUnionConfig,
    *,
    root: Path,
    per_source_runtime: Mapping[str, RuntimeSource],
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    split: NestedSupportEvalSplit,
    eval_raw: object,
    eval_labels: Sequence[int],
    source_union_ref: d1.ReferenceValue,
    real_feature_bacc: float,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], list[dict[str, object]]]:
    sources = tuple(str(source) for source in candidates)
    bundles = []
    component_count_items = []
    generated_hashes: list[str] = []
    source_hashes_all: list[str] = []
    weight_rows: list[dict[str, object]] = []
    plans = []
    for idx in range(cfg.random_mass_bag_control_size):
        plan = _random_dirichlet_plan(
            cfg,
            sources,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            permutation_id=idx,
            total=cfg.synthetic_per_class_total,
        )
        plans.append(plan)
        method = f"{ROW_RANDOM_MASS_BAG_CONTROL}__member_{idx:03d}"
        weight_rows.extend(_source_weight_manifest_rows(experiment_seed, replicate_seed, heldout_center, split, method, plan, {}))
        seed = d1._latent_seed(experiment_seed, heldout_center, replicate_seed, method, cu._plan_hash(plan), "normal")
        generated, labels, component_counts, _source_train_raw, source_hashes = cu._sample_gmm_component_union_cached(
            cfg,
            root=root,
            per_source_runtime=per_source_runtime,
            sources=sources,
            summaries=summaries,
            weight_plan=plan,
            seed=seed,
            control_mode="normal",
        )
        bundle = cu._prediction_cached(
            cfg,
            root=root,
            generated=generated,
            labels=labels,
            eval_raw=eval_raw,
            expert_id=cu.POOL_COMPONENT_UNION,
        )
        bundles.append(bundle)
        component_count_items.append(component_counts)
        generated_hashes.append(_hash_array(generated))
        source_hashes_all.extend(source_hashes)
    pooled = weighted_arithmetic_probability_pool(bundles, [1.0] * len(bundles))
    result = evaluate_probability_predictions(ROW_RANDOM_MASS_BAG_CONTROL, pooled, eval_labels, classes=bundles[0].classes)
    ensemble_plan = _ensemble_plan(cfg, sources, plans)
    row = cu._result_matrix_row(
        cfg,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        candidates=sources,
        prior_method=ROW_RANDOM_MASS_BAG_CONTROL,
        summary_kind="gmm_component_probability_ensemble",
        source_union_ref=source_union_ref,
        center_balanced_ref=d1._missing_reference(),
        real_feature_bacc=real_feature_bacc,
        weight_plan=ensemble_plan,
        bacc=result.bacc,
        macro_f1=result.macro_f1,
        generated_features_hash=_hash_strings(generated_hashes),
        prediction_hash=_hash_array(np.asarray(pooled, dtype=float)),
        selection_source=DIAGNOSTIC_SELECTION,
        claim_role="negative_control_random_mass_bag",
        status="ok",
        error_message="",
        control_mode="normal",
        summaries=summaries,
    )
    row.update(
        {
            "pooling_rule": "arithmetic_probability_ensemble",
            "bag_size": int(cfg.random_mass_bag_control_size),
            "random_mass_sampling_distribution": "dirichlet_uniform_alpha4",
        }
    )
    row = _extend_support_row(row, split=split, plan=ensemble_plan)
    coverage = _extend_support_row(cu._component_coverage_row(row, cu._merge_component_counts(component_count_items), cu._expected_component_keys(sources, summaries, control_mode="normal")), split=split, plan=ensemble_plan)
    paired = _extend_support_row(cu._paired_generation_row(row, str(row["generated_features_hash"]), _hash_strings(source_hashes_all), "ok"), split=split, plan=ensemble_plan)
    return row, coverage, paired, weight_rows


def _ensemble_plan(
    cfg: SupportCalibratedComponentUnionConfig,
    sources: Sequence[str],
    plans: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    sources_tuple = tuple(str(source) for source in sources)
    weights = {
        source: nanmean([_float(dict(plan["weights"]).get(source)) for plan in plans])
        for source in sources_tuple
    }
    total = sum(weights.values())
    weights = {source: value / total for source, value in weights.items()}
    return _plan_from_weights(
        cfg,
        sources_tuple,
        weights,
        scores=weights,
        total=cfg.synthetic_per_class_total,
        mode=f"random_mass_bag_{len(plans)}_probability_ensemble",
    )


def _single_source_alignment_rows(
    cfg: SupportCalibratedComponentUnionConfig,
    *,
    root: Path,
    per_source_runtime: Mapping[str, RuntimeSource],
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    eval_raw: object,
    eval_labels: Sequence[int],
    source_union_ref: d1.ReferenceValue,
    real_feature_bacc: float,
    split: NestedSupportEvalSplit,
) -> list[dict[str, object]]:
    rows = []
    for source in candidates:
        plan = _uniform_source_plan(cfg, (str(source),), total=cfg.synthetic_per_class_total)
        row, _coverage, _weak, _nn, _paired = cu._evaluate_gmm_component_union(
            cfg,
            root=root,
            per_source_runtime=per_source_runtime,
            candidates=(str(source),),
            summaries=summaries,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            eval_raw=eval_raw,
            eval_labels=eval_labels,
            source_union_ref=source_union_ref,
            center_balanced_ref=d1._missing_reference(),
            real_feature_bacc=real_feature_bacc,
            weight_plan=plan,
            prior_method=ROW_SINGLE_SOURCE_ALIGNMENT,
            selection_source=DIAGNOSTIC_SELECTION,
            claim_role="mass_alignment_single_source_oracle_audit",
        )
        row["expert_id"] = str(source)
        row["expert_pool_type"] = POOL_PER_SOURCE
        rows.append(_extend_support_row(row, split=split, plan=plan))
    return rows


def _mass_alignment_rows(
    single_source_rows: Sequence[Mapping[str, object]],
    plan: Mapping[str, object],
    split: NestedSupportEvalSplit,
) -> list[dict[str, object]]:
    ok = [row for row in single_source_rows if row.get("status") == "ok"]
    bacc_by_source = {str(row["expert_id"]): _float(row["bacc"]) for row in ok}
    weights = {str(source): _float(weight) for source, weight in dict(plan["weights"]).items()}
    if not bacc_by_source:
        return []
    bacc_rank = _descending_rank(bacc_by_source)
    weight_rank = _descending_rank(weights)
    oracle = max(bacc_by_source, key=lambda source: (bacc_by_source[source], source))
    bottom = min(bacc_by_source, key=lambda source: (bacc_by_source[source], source))
    top2_weight = set(sorted(weights, key=lambda source: (-weights[source], source))[:2])
    alignment = spearman([weights[source] for source in sorted(bacc_by_source)], [bacc_by_source[source] for source in sorted(bacc_by_source)])
    top1_match = int(max(weights, key=lambda source: (weights[source], source)) == oracle)
    top2_contains = int(oracle in top2_weight)
    out = []
    for row in ok:
        source = str(row["expert_id"])
        out.append(
            {
                "experiment_seed": row.get("experiment_seed", ""),
                "heldout_center": row.get("heldout_center", ""),
                "support_seed": split.support_seed,
                "replicate_seed": row.get("replicate_seed", ""),
                "eval_mode": split.eval_mode,
                "source_center": source,
                "support_weight": weights[source],
                "single_source_bacc": bacc_by_source[source],
                "single_source_rank": bacc_rank[source],
                "weight_rank": weight_rank[source],
                "mass_on_oracle_source": weights[oracle],
                "mass_on_bottom_source": weights[bottom],
                "top1_weight_matches_oracle": top1_match,
                "top2_weight_contains_oracle": top2_contains,
                "spearman_weight_vs_single_source_bacc": alignment,
            }
        )
    return out


def _descending_rank(values: Mapping[str, float]) -> dict[str, int]:
    return {key: idx + 1 for idx, key in enumerate(sorted(values, key=lambda key: (-float(values[key]), key)))}


def _source_weight_manifest_rows(
    experiment_seed: int,
    replicate_seed: int,
    heldout_center: str,
    split: NestedSupportEvalSplit,
    method: str,
    plan: Mapping[str, object],
    rels: Mapping[str, d12.SourceReliability],
) -> list[dict[str, object]]:
    rows = []
    for source in plan["sources"]:
        source_id = str(source)
        rel = rels.get(source_id)
        rows.append(
            {
                "experiment_seed": int(experiment_seed),
                "replicate_seed": int(replicate_seed),
                "heldout_center": str(heldout_center),
                "support_seed": int(split.support_seed),
                "support_size": int(split.support_size),
                "eval_mode": split.eval_mode,
                "support_eval_split_id": split.support_eval_split_id,
                "prior_method": str(method),
                "source_center": source_id,
                "raw_reliability_bacc": "" if rel is None else rel.raw_bacc,
                "reliability_score": "" if rel is None else rel.reliability_score,
                "raw_support_nelbo": dict(plan.get("raw_support_nelbo", {})).get(source_id, ""),
                "calibrated_support_nelbo": dict(plan.get("calibrated_support_nelbo", {})).get(source_id, ""),
                "support_softmax_weight": dict(plan.get("support_weights", {})).get(source_id, ""),
                "pre_round_weight": plan["pre_round_weight"][source_id],
                "normalized_source_weight": plan["weights"][source_id],
                "post_round_budget": plan["budgets"][source_id],
                "synthetic_per_class_budget": plan["budgets"][source_id],
                "post_round_weight": plan["post_round_weight"][source_id],
                "floor_binding": int(bool(plan["floor_binding"][source_id])),
                "floor_binding_count": plan["floor_binding_count"],
                "weight_mode": plan.get("component_union_weight_mode", ""),
                "weight_entropy": plan["weight_entropy"],
                "effective_num_sources": plan["effective_num_sources"],
                "l1_distance_from_uniform": plan["l1_distance_from_uniform"],
                "max_weight": plan["max_weight"],
                "min_weight": plan["min_weight"],
                "dominant_source": plan["dominant_source"],
                "dominant_source_weight": plan["dominant_source_weight"],
                "support_nelbo_tau": plan.get("support_nelbo_tau", ""),
                "support_shrink_lambda": plan.get("support_shrink_lambda", ""),
                "control_permutation_id": plan.get("control_permutation_id", ""),
                "shuffle_seed": plan.get("shuffle_seed", ""),
                "shuffle_mapping_json": json.dumps(dict(plan.get("shuffle_mapping", {})), sort_keys=True),
            }
        )
    return rows


def _support_weight_manifest_rows(
    experiment_seed: int,
    replicate_seed: int,
    heldout_center: str,
    split: NestedSupportEvalSplit,
    plan: Mapping[str, object],
) -> list[dict[str, object]]:
    rows = []
    for source in plan["sources"]:
        source_id = str(source)
        rows.append(
            {
                "experiment_seed": int(experiment_seed),
                "replicate_seed": int(replicate_seed),
                "heldout_center": str(heldout_center),
                "support_seed": int(split.support_seed),
                "support_size": int(split.support_size),
                "support_eval_split_id": split.support_eval_split_id,
                "source_center": source_id,
                "raw_support_nelbo": plan["raw_support_nelbo"][source_id],
                "calibrated_support_nelbo": plan["calibrated_support_nelbo"][source_id],
                "support_softmax_weight": plan["support_weights"][source_id],
                "support_shrink_weight": plan["weights"][source_id],
                "synthetic_per_class_budget": plan["budgets"][source_id],
                "support_nelbo_tau": plan["support_nelbo_tau"],
                "support_shrink_lambda": plan["support_shrink_lambda"],
            }
        )
    return rows


def _support_score_manifest_rows(scores: Sequence[SupportScore], split: NestedSupportEvalSplit) -> list[dict[str, object]]:
    rows = []
    for score in scores:
        row = score.to_csv_row()
        row.update(
            {
                "support_eval_split_id": split.support_eval_split_id,
                "parent_support32_split_id": split.parent_support32_split_id,
                "support_labels_used": int(split.support_labels_used),
            }
        )
        rows.append(row)
    return rows


def _split_manifest_rows(
    splits: Sequence[NestedSupportEvalSplit],
    experiment_seed: int,
    replicate_seed: int,
) -> list[dict[str, object]]:
    return [
        {
            "experiment_seed": int(experiment_seed),
            "replicate_seed": int(replicate_seed),
            "heldout_center": split.heldout_center,
            "support_seed": split.support_seed,
            "support_size": split.support_size,
            "eval_mode": split.eval_mode,
            "support_eval_split_id": split.support_eval_split_id,
            "parent_support32_split_id": split.parent_support32_split_id,
            "support_labels_used": int(split.support_labels_used),
            "support_size_actual": len(split.support_indices),
            "n_target_eval": len(split.eval_indices),
            "support_sample_id_hash": _hash_strings(split.support_sample_ids),
            "eval_sample_id_hash": _hash_strings(split.eval_sample_ids),
            "nested_support_diagnostics": 1,
            "fixed_eval_support_size_diagnostics": int(split.eval_mode == "fixed_support32"),
        }
        for split in splits
    ]


def _extend_support_row(
    row: Mapping[str, object],
    *,
    split: NestedSupportEvalSplit,
    plan: Mapping[str, object] | None,
) -> dict[str, object]:
    out = dict(row)
    out.update(
        {
            "support_seed": int(split.support_seed),
            "support_size": int(split.support_size),
            "eval_mode": split.eval_mode,
            "support_eval_split_id": split.support_eval_split_id,
            "parent_support32_split_id": split.parent_support32_split_id,
            "support_labels_used": int(split.support_labels_used),
            "n_target_eval_after_support": len(split.eval_indices),
            "nested_support_diagnostics": 1,
            "fixed_eval_support_size_diagnostics": int(split.eval_mode == "fixed_support32"),
        }
    )
    if plan is not None:
        out.update(
            {
                "support_weight_json": json.dumps(dict(plan.get("support_weights", {})), sort_keys=True),
                "raw_support_nelbo_json": json.dumps(dict(plan.get("raw_support_nelbo", {})), sort_keys=True),
                "calibrated_support_nelbo_json": json.dumps(dict(plan.get("calibrated_support_nelbo", {})), sort_keys=True),
                "pre_round_weight_json": json.dumps(dict(plan.get("pre_round_weight", {})), sort_keys=True),
                "post_round_weight_json": json.dumps(dict(plan.get("post_round_weight", {})), sort_keys=True),
                "floor_binding_json": json.dumps(dict(plan.get("floor_binding", {})), sort_keys=True),
                "floor_binding_count": plan.get("floor_binding_count", ""),
                "support_nelbo_tau": plan.get("support_nelbo_tau", ""),
                "support_shrink_lambda": plan.get("support_shrink_lambda", ""),
            }
        )
    return out


def _support_empty_row(
    cfg: SupportCalibratedComponentUnionConfig,
    *,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    prior_method: str,
    source_union_ref: d1.ReferenceValue,
    status: str,
    error_message: str,
    claim_role: str,
) -> dict[str, object]:
    return cu._empty_matrix_row(
        cfg,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        candidates=candidates,
        prior_method=prior_method,
        source_union_ref=source_union_ref,
        center_balanced_ref=d1._missing_reference(),
        real_feature_bacc=math.nan,
        status=status,
        error_message=error_message,
        claim_role=claim_role,
    )


def _source_union_reference_row(
    cfg: SupportCalibratedComponentUnionConfig,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    reference: d1.ReferenceValue,
) -> dict[str, object]:
    row = cu._reference_matrix_row(
        cfg,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        candidates=candidates,
        prior_method=ROW_SOURCE_UNION_K16_REFERENCE,
        reference=reference,
    )
    row.update(
        {
            "expert_id": POOL_SOURCE_UNION,
            "expert_pool_type": POOL_SOURCE_UNION,
            "variant_id": UNION_VARIANT,
            "reference_eval_scope": "external_full_target_eval_diagnostic",
        }
    )
    return row


def _real_feature_bacc_for(
    rows: Sequence[Mapping[str, object]],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    split: NestedSupportEvalSplit,
) -> float:
    for row in rows:
        if (
            row.get("prior_method") == ROW_REAL_FEATURE_DENSE_REFERENCE
            and str(row.get("experiment_seed")) == str(experiment_seed)
            and str(row.get("heldout_center")) == str(heldout_center)
            and str(row.get("replicate_seed")) == str(replicate_seed)
            and str(row.get("support_eval_split_id")) == split.support_eval_split_id
        ):
            return _float(row.get("bacc"))
    return math.nan


def _decision(
    rows: Sequence[Mapping[str, object]],
    cfg: SupportCalibratedComponentUnionConfig,
    *,
    leakage_status: str,
    mass_alignment_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    primary_stats = cu._method_stats(cu._rows_for(rows, cfg.primary_method))
    uniform_stats = cu._method_stats(cu._rows_for(rows, ROW_UNIFORM_COMPONENT_UNION))
    shrink050_stats = cu._method_stats(cu._rows_for(rows, ROW_RELIABILITY_SHRINK050))
    random_bag_stats = cu._method_stats(cu._rows_for(rows, ROW_RANDOM_MASS_BAG_CONTROL))
    source_union_stats = cu._method_stats(cu._rows_for(rows, ROW_SOURCE_UNION_K16_REFERENCE))
    real_stats = cu._method_stats(_primary_scope_rows(cu._rows_for(rows, ROW_REAL_FEATURE_DENSE_REFERENCE), cfg))
    primary_bacc = _float(primary_stats["center_equal_mean_bacc"])
    delta_uniform = primary_bacc - _float(uniform_stats["center_equal_mean_bacc"])
    delta_shrink050 = primary_bacc - _float(shrink050_stats["center_equal_mean_bacc"])
    delta_random_bag = primary_bacc - _float(random_bag_stats["center_equal_mean_bacc"])
    null_summary = _matched_shuffled_support_null_summary(rows, cfg)
    mass_summary = _mass_alignment_summary(mass_alignment_rows)
    retention = d1._retention(primary_bacc, _float(source_union_stats["center_equal_mean_bacc"]))
    oracle_gap = _float(source_union_stats["center_equal_mean_bacc"]) - primary_bacc
    floor_binding_count = _primary_floor_binding_count(rows, cfg)
    strong = (
        leakage_status == "PASS"
        and int(primary_stats["n_heldout_centers"]) >= len(cfg.heldout_centers)
        and delta_uniform >= 0.010
        and (delta_shrink050 >= 0.005 or _float(primary_stats["min_center_bacc"]) - _float(shrink050_stats["min_center_bacc"]) >= 0.020)
        and delta_random_bag >= 0.005
        and _float(primary_stats["min_center_bacc"]) >= 0.82
        and _float(primary_stats["seed_std_bacc"]) <= 0.045
        and retention >= 0.97
        and oracle_gap <= 0.005
        and _float(null_summary["primary_minus_null_p95"]) > 0.0
        and _float(null_summary["empirical_p_value"]) <= 0.10
        and _float(null_summary["primary_minus_null_mean"]) >= 0.005
        and _float(null_summary["paired_cell_win_fraction_vs_null"]) >= 0.60
        and _float(mass_summary["mass_alignment_spearman"]) > 0.0
        and _float(mass_summary["top2_weight_contains_oracle"]) >= 0.60
        and int(floor_binding_count) == 0
    )
    useful = (
        leakage_status == "PASS"
        and int(primary_stats["n_heldout_centers"]) >= len(cfg.heldout_centers)
        and delta_uniform > 0.0
        and delta_shrink050 > 0.0
        and delta_random_bag >= -0.002
        and _float(primary_stats["min_center_bacc"]) >= 0.80
        and _float(primary_stats["seed_std_bacc"]) <= 0.055
        and retention >= 0.94
        and _float(null_summary["primary_minus_null_p95"]) > 0.0
        and _float(null_summary["primary_minus_null_mean"]) >= 0.005
    )
    verdict = "SUPPORT_CALIBRATED_COMPONENT_UNION_FAIL"
    if leakage_status != "PASS":
        verdict = "PROTOCOL_FAIL"
    elif strong:
        verdict = "SUPPORT_CALIBRATED_COMPONENT_UNION_STRONG_SUCCESS"
    elif useful:
        verdict = "SUPPORT_CALIBRATED_COMPONENT_UNION_USEFUL_THESIS_SUCCESS"
    flags = []
    if math.isfinite(delta_uniform) and delta_uniform < 0.010:
        flags.append("DELTA_VS_UNIFORM_BELOW_0P010")
    if math.isfinite(delta_shrink050) and delta_shrink050 < 0.005:
        flags.append("DELTA_VS_SHRINK050_BELOW_0P005")
    if math.isfinite(delta_random_bag) and delta_random_bag < -0.002:
        flags.append("RANDOM_MASS_BAG_COMPETITIVE")
    if math.isfinite(_float(null_summary["primary_minus_null_p95"])) and _float(null_summary["primary_minus_null_p95"]) <= 0.0:
        flags.append("MATCHED_SHUFFLED_SUPPORT_NULL_COMPETITIVE")
    if math.isfinite(_float(mass_summary["mass_alignment_spearman"])) and _float(mass_summary["mass_alignment_spearman"]) <= 0.0:
        flags.append("MASS_ALIGNMENT_NON_POSITIVE")
    if int(floor_binding_count) != 0:
        flags.append("PRIMARY_FLOOR_BINDING_NONZERO")
    return {
        "primary_verdict": verdict,
        "diagnostic_flags": "|".join(flags),
        "leakage_status": leakage_status,
        "primary_method": cfg.primary_method,
        "center_equal_mean_bacc": primary_stats["center_equal_mean_bacc"],
        "seed_cell_mean_bacc": primary_stats["seed_cell_mean_bacc"],
        "center_equal_macro_f1": primary_stats["center_equal_macro_f1"],
        "min_center_bacc": primary_stats["min_center_bacc"],
        "seed_std_bacc": primary_stats["seed_std_bacc"],
        "primary_vs_uniform_delta": delta_uniform,
        "primary_vs_shrink050_delta": delta_shrink050,
        "primary_vs_random_mass_bag_delta": delta_random_bag,
        "uniform_component_union_center_equal_mean_bacc": uniform_stats["center_equal_mean_bacc"],
        "shrink050_center_equal_mean_bacc": shrink050_stats["center_equal_mean_bacc"],
        "random_mass_bag_center_equal_mean_bacc": random_bag_stats["center_equal_mean_bacc"],
        "source_union_k16_center_equal_mean_bacc": source_union_stats["center_equal_mean_bacc"],
        "real_feature_dense_center_equal_mean_bacc": real_stats["center_equal_mean_bacc"],
        "retention_vs_source_union_k16": retention,
        "oracle_gap_vs_source_union_k16": oracle_gap,
        "oracle_gap_vs_real_feature_dense": _float(real_stats["center_equal_mean_bacc"]) - primary_bacc,
        "nested_support_diagnostics": True,
        "fixed_eval_support_size_diagnostics": True,
        "floor_binding_count": floor_binding_count,
        **null_summary,
        **mass_summary,
        **primary_stats,
    }


def _matched_shuffled_support_null_summary(
    rows: Sequence[Mapping[str, object]],
    cfg: SupportCalibratedComponentUnionConfig,
) -> dict[str, object]:
    primary = cu._rows_for(rows, cfg.primary_method)
    primary_stats = cu._method_stats(primary)
    primary_bacc = _float(primary_stats["center_equal_mean_bacc"])
    null_rows = [row for row in rows if _is_matched_shuffled_support_method(row.get("prior_method"))]
    methods = sorted({str(row.get("prior_method")) for row in null_rows})
    unique_patterns = {
        "|".join([str(row.get("prior_method", "")), str(row.get("shuffle_mapping_json", ""))])
        for row in null_rows
        if row.get("status") == "ok"
    }
    null_means = [_float(cu._method_stats(cu._rows_for(rows, method))["center_equal_mean_bacc"]) for method in methods]
    null_means = [value for value in null_means if math.isfinite(value)]
    null_mean = nanmean(null_means) if null_means else math.nan
    null_p95 = float(np.quantile(null_means, 0.95)) if null_means else math.nan
    empirical_p = (
        (1.0 + float(sum(value >= primary_bacc for value in null_means))) / (float(len(null_means)) + 1.0)
        if null_means and math.isfinite(primary_bacc)
        else math.nan
    )
    primary_by_cell = {
        (str(row.get("experiment_seed")), str(row.get("heldout_center")), str(row.get("replicate_seed"))): _float(row.get("bacc"))
        for row in primary
    }
    pair_wins = 0
    pair_total = 0
    paired_deltas = []
    null_by_cell: dict[tuple[str, str, str], list[float]] = {}
    for row in null_rows:
        key = (str(row.get("experiment_seed")), str(row.get("heldout_center")), str(row.get("replicate_seed")))
        null_bacc = _float(row.get("bacc"))
        primary_cell = primary_by_cell.get(key, math.nan)
        if math.isfinite(null_bacc):
            null_by_cell.setdefault(key, []).append(null_bacc)
        if math.isfinite(null_bacc) and math.isfinite(primary_cell):
            pair_total += 1
            if primary_cell > null_bacc:
                pair_wins += 1
    for key, values in null_by_cell.items():
        primary_cell = primary_by_cell.get(key, math.nan)
        null_cell_mean = nanmean(values)
        if math.isfinite(primary_cell) and math.isfinite(null_cell_mean):
            paired_deltas.append(primary_cell - null_cell_mean)
    return {
        "n_null_permutations": len(null_means),
        "effective_unique_null_patterns": len(unique_patterns),
        "primary_center_equal_mean_bacc": primary_bacc,
        "null_mean_center_equal_bacc": null_mean,
        "null_p95_center_equal_bacc": null_p95,
        "null_max_center_equal_bacc": max(null_means, default=math.nan),
        "empirical_p_value": empirical_p,
        "primary_minus_null_mean": primary_bacc - null_mean if math.isfinite(primary_bacc) and math.isfinite(null_mean) else math.nan,
        "primary_minus_null_p95": primary_bacc - null_p95 if math.isfinite(primary_bacc) and math.isfinite(null_p95) else math.nan,
        "paired_cell_mean_delta_vs_null_mean": nanmean(paired_deltas) if paired_deltas else math.nan,
        "paired_cell_win_fraction_vs_null": float(pair_wins) / float(pair_total) if pair_total else math.nan,
    }


def _mass_alignment_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    primary = [row for row in rows if row.get("eval_mode") == "primary_style" and str(row.get("support_size")) == "8"]
    cell_seen = set()
    spearmans = []
    top1 = []
    top2 = []
    for row in primary:
        key = (row.get("experiment_seed"), row.get("heldout_center"), row.get("support_seed"), row.get("replicate_seed"), row.get("eval_mode"))
        if key in cell_seen:
            continue
        cell_seen.add(key)
        spearmans.append(_float(row.get("spearman_weight_vs_single_source_bacc")))
        top1.append(_float(row.get("top1_weight_matches_oracle")))
        top2.append(_float(row.get("top2_weight_contains_oracle")))
    return {
        "mass_alignment_spearman": nanmean([v for v in spearmans if math.isfinite(v)]) if spearmans else math.nan,
        "top1_weight_matches_oracle": nanmean([v for v in top1 if math.isfinite(v)]) if top1 else math.nan,
        "top2_weight_contains_oracle": nanmean([v for v in top2 if math.isfinite(v)]) if top2 else math.nan,
    }


def _primary_floor_binding_count(rows: Sequence[Mapping[str, object]], cfg: SupportCalibratedComponentUnionConfig) -> int:
    count = 0
    for row in rows:
        if row.get("prior_method") != cfg.primary_method or row.get("status") != "ok":
            continue
        try:
            bindings = json.loads(str(row.get("floor_binding_json", "{}")))
        except Exception:
            bindings = {}
        count += sum(1 for value in bindings.values() if bool(value))
    return count


def _support_size_sensitivity_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    specs = [
        (8, "primary_style", PRIMARY_SUPPORT_CALIBRATED_COMPONENT_UNION_METHOD, True),
        (8, "fixed_support32", ROW_SUPPORT8_FIXED_EVAL, False),
        (16, "primary_style", ROW_SUPPORT16_PRIMARY_EVAL, False),
        (16, "fixed_support32", ROW_SUPPORT16_FIXED_EVAL, False),
        (32, "primary_style", ROW_SUPPORT32_PRIMARY_EVAL, False),
        (32, "fixed_support32", ROW_SUPPORT32_FIXED_EVAL, False),
    ]
    out = []
    for support_size, eval_mode, method, primary in specs:
        subset = [row for row in rows if row.get("prior_method") == method and str(row.get("eval_mode")) == eval_mode]
        out.append(
            {
                "support_size": support_size,
                "eval_mode": eval_mode,
                "prior_method": method,
                "is_primary_claim": int(primary),
                **cu._method_stats(subset),
            }
        )
    return out


def _matched_shuffled_support_cell_delta_rows(
    rows: Sequence[Mapping[str, object]],
    cfg: SupportCalibratedComponentUnionConfig,
) -> list[dict[str, object]]:
    primary_by_cell = {
        (str(row.get("experiment_seed")), str(row.get("heldout_center")), str(row.get("replicate_seed"))): _float(row.get("bacc"))
        for row in cu._rows_for(rows, cfg.primary_method)
    }
    out = []
    for row in rows:
        method = str(row.get("prior_method"))
        if not _is_matched_shuffled_support_method(method) or row.get("status") != "ok":
            continue
        key = (str(row.get("experiment_seed")), str(row.get("heldout_center")), str(row.get("replicate_seed")))
        primary = primary_by_cell.get(key, math.nan)
        null = _float(row.get("bacc"))
        out.append(
            {
                "experiment_seed": key[0],
                "heldout_center": key[1],
                "replicate_seed": key[2],
                "support_seed": row.get("support_seed", ""),
                "support_size": row.get("support_size", ""),
                "eval_mode": row.get("eval_mode", ""),
                "null_perm_id": row.get("control_permutation_id", ""),
                "primary_bacc": primary,
                "null_bacc": null,
                "delta_primary_minus_null": primary - null if math.isfinite(primary) and math.isfinite(null) else math.nan,
            }
        )
    return out


def _oracle_gap_rows(rows: Sequence[Mapping[str, object]], cfg: SupportCalibratedComponentUnionConfig) -> list[dict[str, object]]:
    primary = _float(cu._method_stats(cu._rows_for(rows, cfg.primary_method))["center_equal_mean_bacc"])
    source_union = _float(cu._method_stats(cu._rows_for(rows, ROW_SOURCE_UNION_K16_REFERENCE))["center_equal_mean_bacc"])
    real = _float(cu._method_stats(_primary_scope_rows(cu._rows_for(rows, ROW_REAL_FEATURE_DENSE_REFERENCE), cfg))["center_equal_mean_bacc"])
    return [
        {
            "primary_method": cfg.primary_method,
            "primary_center_equal_mean_bacc": primary,
            "source_union_k16_center_equal_mean_bacc": source_union,
            "real_feature_dense_center_equal_mean_bacc": real,
            "retention_vs_source_union_k16": d1._retention(primary, source_union),
            "oracle_gap_vs_source_union_k16": source_union - primary if math.isfinite(source_union) and math.isfinite(primary) else math.nan,
            "oracle_gap_vs_real_feature_dense": real - primary if math.isfinite(real) and math.isfinite(primary) else math.nan,
            "source_union_reference_eval_scope": "external_full_target_eval_diagnostic",
        }
    ]


def _primary_scope_rows(
    rows: Sequence[Mapping[str, object]],
    cfg: SupportCalibratedComponentUnionConfig,
) -> list[Mapping[str, object]]:
    return [
        row for row in rows
        if str(row.get("support_size")) == str(cfg.support_size)
        and row.get("eval_mode") == "primary_style"
    ]


def _eligibility_row(row: Mapping[str, object], split: NestedSupportEvalSplit) -> dict[str, object]:
    return {
        "experiment_seed": row.get("experiment_seed", ""),
        "heldout_center": row.get("heldout_center", ""),
        "support_seed": split.support_seed,
        "replicate_seed": row.get("replicate_seed", ""),
        "support_size": split.support_size,
        "eval_mode": split.eval_mode,
        "prior_method": row.get("prior_method", ""),
        "eligible": int(row.get("status") == "ok"),
        "status": row.get("status", ""),
        "error_message": row.get("error_message", ""),
    }


def _write_artifacts(
    root: Path,
    cfg: SupportCalibratedComponentUnionConfig,
    *,
    matrix_rows: Sequence[Mapping[str, object]],
    component_manifest_rows: Sequence[Mapping[str, object]],
    source_summary_rows: Sequence[Mapping[str, object]],
    component_coverage_rows: Sequence[Mapping[str, object]],
    source_weight_rows: Sequence[Mapping[str, object]],
    support_weight_rows: Sequence[Mapping[str, object]],
    support_score_rows: Sequence[Mapping[str, object]],
    split_rows: Sequence[Mapping[str, object]],
    paired_generation_rows: Sequence[Mapping[str, object]],
    mass_alignment_rows: Sequence[Mapping[str, object]],
    real_feature_rows: Sequence[Mapping[str, object]],
    model_manifest_rows: Sequence[Mapping[str, object]],
    eligibility_rows: Sequence[Mapping[str, object]],
    decision: Mapping[str, object],
    leakage_status: str,
    protocol_violations: Sequence[str],
    target_expert_excluded: bool,
) -> None:
    null_rows = [row for row in matrix_rows if _is_matched_shuffled_support_method(row.get("prior_method"))]
    write_csv_rows(root / "tables" / "support_calibrated_component_union_downstream_matrix.csv", matrix_rows)
    write_csv_rows(root / "tables" / "support_calibrated_component_union_summary.csv", [dict(decision)])
    write_csv_rows(root / "tables" / "support_size_sensitivity_summary.csv", _support_size_sensitivity_rows(matrix_rows))
    write_csv_rows(root / "tables" / "matched_shuffled_support_null_matrix.csv", null_rows)
    write_csv_rows(root / "tables" / "matched_shuffled_support_null_summary.csv", [_null_summary_output(decision)])
    write_csv_rows(root / "tables" / "matched_shuffled_support_cell_delta_summary.csv", _matched_shuffled_support_cell_delta_rows(matrix_rows, cfg))
    write_csv_rows(root / "tables" / "oracle_gap_summary.csv", _oracle_gap_rows(matrix_rows, cfg))
    write_csv_rows(root / "tables" / "eligibility_audit.csv", eligibility_rows)
    write_csv_rows(root / "tables" / "support_eval_split_manifest.csv", split_rows)
    write_csv_rows(root / "tables" / "support_nelbo_score_manifest.csv", support_score_rows)
    write_csv_rows(root / "tables" / "support_weight_manifest.csv", support_weight_rows)
    write_csv_rows(root / "tables" / "source_weight_manifest.csv", source_weight_rows)
    write_csv_rows(root / "tables" / "component_manifest.csv", component_manifest_rows)
    write_csv_rows(root / "tables" / "source_summary_diagnostics.csv", source_summary_rows)
    write_csv_rows(root / "tables" / "component_coverage_audit.csv", component_coverage_rows)
    write_csv_rows(root / "tables" / "paired_generation_audit.csv", paired_generation_rows)
    write_csv_rows(root / "tables" / "mass_alignment_to_single_source_oracle.csv", mass_alignment_rows)
    write_csv_rows(root / "tables" / "real_feature_reference_matrix.csv", real_feature_rows)
    write_csv_rows(root / "manifests" / "support_calibrated_component_union_model_manifest.csv", model_manifest_rows)
    leakage = build_leakage_report(
        target_support_labels_for_selection=False,
        target_eval_labels_for_scoring_only=True,
        target_expert_excluded=target_expert_excluded,
        oracle_rows_diagnostic_only=True,
        extra_violations=protocol_violations,
    )
    write_json(root / "reports" / "leakage_report.json", leakage.to_json_dict())
    write_json(
        root / "manifests" / "protocol_manifest.json",
        {
            "schema_version": "cvae_rebuild_support_calibrated_component_union_protocol_manifest_v1",
            "experiment_name": cfg.name,
            "experiment_type": "target_support_compatibility_calibrated_component_union",
            "primary_variant": cfg.primary_variant,
            "primary_method": cfg.primary_method,
            "support_size": cfg.support_size,
            "support_size_diagnostics": list(cfg.support_size_diagnostics),
            "nested_support_max_size": cfg.nested_support_max_size,
            "nested_support_diagnostics": True,
            "fixed_eval_support_size_diagnostics": True,
            "support_labels_for_selection": False,
            "target_eval_labels_for_scoring_only": True,
            "target_expert_excluded": target_expert_excluded,
            "support_nelbo_tau": cfg.support_nelbo_tau,
            "support_shrink_lambda": cfg.support_shrink_lambda,
            "matched_shuffled_support_null_permutations": cfg.matched_shuffled_support_null_permutations,
            "random_mass_bag_control_size": cfg.random_mass_bag_control_size,
            "oracle_rows_diagnostic_only": True,
            "source_union_reference_eval_scope": "external_full_target_eval_diagnostic",
            "protocol_wording": PROTOCOL_WORDING,
        },
    )
    write_json(root / "run_config_resolved.yaml", _resolved_config(cfg))
    _write_decision_summary(root, decision, leakage_status=leakage_status)


def _null_summary_output(decision: Mapping[str, object]) -> dict[str, object]:
    fields = (
        "n_null_permutations",
        "effective_unique_null_patterns",
        "primary_center_equal_mean_bacc",
        "null_mean_center_equal_bacc",
        "null_p95_center_equal_bacc",
        "null_max_center_equal_bacc",
        "empirical_p_value",
        "primary_minus_null_mean",
        "primary_minus_null_p95",
        "paired_cell_mean_delta_vs_null_mean",
        "paired_cell_win_fraction_vs_null",
    )
    return {field: decision.get(field, math.nan) for field in fields}


def _write_decision_summary(root: Path, decision: Mapping[str, object], *, leakage_status: str) -> None:
    text = "\n".join(
        [
            "# Support-Calibrated Component-Union Prior v1",
            "",
            "## Summary",
            "",
            f"- Primary method: `{decision.get('primary_method', '')}`",
            f"- Primary verdict: `{decision.get('primary_verdict', '')}`",
            f"- Diagnostic flags: `{decision.get('diagnostic_flags', '')}`",
            f"- Center-equal mean BACC: {_format_float(decision.get('center_equal_mean_bacc'))}",
            f"- Seed-cell mean BACC: {_format_float(decision.get('seed_cell_mean_bacc'))}",
            f"- Center-equal macro-F1: {_format_float(decision.get('center_equal_macro_f1'))}",
            f"- Min center BACC: {_format_float(decision.get('min_center_bacc'))}",
            f"- Seed std BACC: {_format_float(decision.get('seed_std_bacc'))}",
            f"- Primary vs uniform delta: {_format_float(decision.get('primary_vs_uniform_delta'))}",
            f"- Primary vs shrink050 delta: {_format_float(decision.get('primary_vs_shrink050_delta'))}",
            f"- Primary vs random mass bag delta: {_format_float(decision.get('primary_vs_random_mass_bag_delta'))}",
            f"- Retention vs source-union K16: {_format_float(decision.get('retention_vs_source_union_k16'))}",
            f"- Oracle gap vs source-union K16: {_format_float(decision.get('oracle_gap_vs_source_union_k16'))}",
            f"- Primary minus shuffled-support null mean: {_format_float(decision.get('primary_minus_null_mean'))}",
            f"- Primary minus shuffled-support null p95: {_format_float(decision.get('primary_minus_null_p95'))}",
            f"- Matched shuffled-support empirical p-value: {_format_float(decision.get('empirical_p_value'))}",
            f"- Paired-cell win fraction vs null: {_format_float(decision.get('paired_cell_win_fraction_vs_null'))}",
            f"- Mass-alignment Spearman: {_format_float(decision.get('mass_alignment_spearman'))}",
            f"- Top2 weighted sources contain oracle: {_format_float(decision.get('top2_weight_contains_oracle'))}",
            f"- nested_support_diagnostics: `{decision.get('nested_support_diagnostics', True)}`",
            f"- fixed_eval_support_size_diagnostics: `{decision.get('fixed_eval_support_size_diagnostics', True)}`",
            f"- floor_binding_count: {decision.get('floor_binding_count', '')}",
            f"- Leakage status: `{leakage_status}`",
            "",
            "## Protocol Boundary",
            "",
            PROTOCOL_WORDING,
            "",
            "The primary support8 row is the only adoption-eligible row.",
            "Support16/support32 and fixed-eval rows are diagnostic-only and cannot rescue the primary claim.",
            "Target support is unlabeled; target evaluation labels are final scoring only.",
            "",
            "## Supported Claim If Successful",
            "",
            "Unlabeled target support can estimate compatibility well enough to calibrate dense generative expert composition, improving downstream utility without target-evaluation labels.",
            "",
        ]
    )
    path = root / "reports" / "decision_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _resolved_config(cfg: SupportCalibratedComponentUnionConfig) -> dict[str, object]:
    return {
        "name": cfg.name,
        "artifact_root": str(cfg.artifact_root),
        "feature_cache_root": str(cfg.feature_cache_root),
        "repair_artifact_root": str(cfg.repair_artifact_root),
        "source_union_gmm_artifact_root": "" if cfg.source_union_gmm_artifact_root is None else str(cfg.source_union_gmm_artifact_root),
        "backbone": cfg.backbone,
        "experiment_seeds": list(cfg.experiment_seeds),
        "heldout_centers": list(cfg.heldout_centers),
        "replicate_seeds": list(cfg.replicate_seeds),
        "support_seeds": list(cfg.support_seeds),
        "support_size": cfg.support_size,
        "support_size_diagnostics": list(cfg.support_size_diagnostics),
        "nested_support_max_size": cfg.nested_support_max_size,
        "synthetic_per_class_total": cfg.synthetic_per_class_total,
        "min_per_source_per_class": cfg.min_per_source_per_class,
        "primary_method": cfg.primary_method,
        "support_nelbo_tau": cfg.support_nelbo_tau,
        "support_shrink_lambda": cfg.support_shrink_lambda,
        "matched_shuffled_support_null_permutations": cfg.matched_shuffled_support_null_permutations,
        "random_mass_bag_control_size": cfg.random_mass_bag_control_size,
    }


def _optional_path(base: Path, value: object) -> Path | None:
    if value is None or str(value) == "":
        return None
    return _path(base, str(value))


def _row_center(row: Mapping[str, object], *, center_key: str) -> str:
    if center_key in row:
        return str(row[center_key])
    if "magnification" in row:
        return str(row["magnification"])
    raise ProtocolError(f"Metadata row missing center key {center_key!r}.")


def _sample_id(row: Mapping[str, object], idx: int, *, sample_id_key: str) -> str:
    value = row.get(sample_id_key, "")
    return str(value) if str(value) else f"row_{idx}"
