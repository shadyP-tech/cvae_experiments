from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from features import load_feature_cache, select_rows
from metrics import nanmean
from preservation_repair import (
    NA,
    PRIMARY_VARIANT,
    _existing_cache_path,
    _float,
    _format_float,
    _label,
    _load_mapping,
    _mapping,
    _path,
    _source_data_for_centers,
    _target_indices,
)
from preservation_sampling import DIAGNOSTIC_SELECTION, PRIMARY_SELECTION, RuntimeSource, _manifest_row, _per_source_variant, _runtime_source
from protocol import ProtocolError, assert_candidate_pool, build_leakage_report
from reporting import prepare_artifact_dirs, write_csv_rows, write_protocol_finalization
from splits import candidate_experts

import component_union_mass_bagged as mb
import decentralized_adaptive_gmm_prior as d1a
import decentralized_component_union_prior as cu
import decentralized_k16_gmm_prior as d1
import decentralized_reliability_weighted_gmm_prior as d12
import paired_dense_all4_reliability_confirmation as paired
import dense_reliability_tailshield_random_mass_bag as tail


HARMFUL_SUPPRESSION_NAME = "virchow2_cvae_source_inner_harmful_source_suppression_random_mass_bag_v1"
PRIMARY_HARMFUL_SUPPRESSION_METHOD = "source_inner_harm_suppressed_random_mass_bag_component_union_v1"
HARMFUL_SUPPRESSION_SOURCE_WEIGHTING = "source_inner_harm_suppressed_random_mass_bag_component_union"
ROW_SHUFFLED_HARMFULNESS_CONTROL = "source_inner_harm_suppression_shuffled_harmfulness_control"
ROW_RANDOM_MATCHED_SUPPRESSION_CONTROL = "source_inner_harm_suppression_random_matched_control"
ROW_INVERSE_HARMFULNESS_CONTROL = "source_inner_harm_suppression_inverse_harmfulness_control"
ROW_HARD_EXCLUSION_DIAGNOSTIC = "source_inner_harm_suppression_hard_exclusion_diagnostic"
PROTOCOL_WORDING = (
    "This is a locked source-only harmful-source suppression audit. Source-inner labels are used "
    "only for non-target pseudo-target validation. The heldout target expert, target support, "
    "target evaluation labels, and target evaluation metrics are not visible to the suppressor."
)


@dataclass(frozen=True)
class HarmfulSourceSuppressionConfig:
    name: str
    artifact_root: Path
    repair_artifact_root: Path
    paired_dense_artifact_root: Path | None
    dense_tailshield_artifact_root: Path | None
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
    dirichlet_total_concentration: float
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
    min_harmfulness_observations: int
    moderate_hit_rate_min: float
    moderate_gain_min: float
    moderate_helpful_loss_max: float
    severe_hit_rate_min: float
    severe_gain_min: float
    severe_helpful_loss_max: float
    max_suppressed_sources: int
    suppression_rate_low: float
    suppression_rate_high: float
    oracle_harm_delta_threshold: float
    classifier_type: str
    classifier_solver: str
    classifier_c: float
    classifier_max_iter: int
    classifier_class_weight: str
    classifier_seed: int | None
    skip_nearest_neighbor_audit: bool

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


def load_harmful_source_suppression_config(path: str | Path) -> HarmfulSourceSuppressionConfig:
    source = Path(path).resolve()
    data = _load_mapping(source)
    base_dir = source.parents[2] if len(source.parents) >= 3 else source.parent
    return parse_harmful_source_suppression_config(data, base_dir=base_dir)


def parse_harmful_source_suppression_config(
    data: Mapping[str, Any],
    *,
    base_dir: str | Path = ".",
) -> HarmfulSourceSuppressionConfig:
    base = Path(base_dir)
    experiment = _mapping(data, "experiment")
    inputs = _mapping(data, "inputs")
    run = _mapping(data, "run_matrix")
    generation = _mapping(data, "generation")
    suppression = _mapping(data, "harmful_source_suppression")
    classifier = _mapping(data, "classifier")
    memory_raw = data.get("memory", {})
    if not isinstance(memory_raw, Mapping):
        raise ProtocolError("memory must be a mapping when provided.")
    memory = memory_raw
    if "support_size" in run or "support_seeds" in run:
        raise ProtocolError("Harmful-source suppression v1 must not configure or consume target support rows.")
    cfg = HarmfulSourceSuppressionConfig(
        name=str(experiment["name"]),
        artifact_root=_path(base, str(experiment["artifact_root"])),
        repair_artifact_root=_path(base, str(inputs["repair_artifact_root"])),
        paired_dense_artifact_root=_optional_path(base, inputs.get("paired_dense_artifact_root")),
        dense_tailshield_artifact_root=_optional_path(base, inputs.get("dense_tailshield_artifact_root")),
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
        primary_method=str(suppression["primary_method"]),
        random_mass_bag_size=int(suppression["random_mass_bag_size"]),
        random_mass_bag_alpha=float(suppression["random_mass_bag_alpha"]),
        dirichlet_total_concentration=float(suppression["dirichlet_total_concentration"]),
        candidate_components_per_source_class=tuple(int(v) for v in suppression["candidate_components_per_source_class"]),
        min_samples_per_component=int(suppression["min_samples_per_component"]),
        source_weighting=str(suppression["source_weighting"]),
        gmm_covariance_type=str(suppression["gmm_covariance_type"]),
        gmm_reg_covar=float(suppression["gmm_reg_covar"]),
        gmm_n_init=int(suppression["gmm_n_init"]),
        gmm_max_iter=int(suppression["gmm_max_iter"]),
        min_component_weight=float(suppression["min_component_weight"]),
        variance_floor=float(suppression["variance_floor"]),
        variance_ceiling_multiplier=float(suppression["variance_ceiling_multiplier"]),
        primary_pooling=str(suppression["primary_pooling"]),
        reliability_floor_score=float(suppression["reliability_floor_score"]),
        reliability_epsilon=float(suppression["reliability_epsilon"]),
        anchor_repro_tolerance=float(suppression["anchor_repro_tolerance"]),
        min_harmfulness_observations=int(suppression["min_harmfulness_observations"]),
        moderate_hit_rate_min=float(suppression["moderate_hit_rate_min"]),
        moderate_gain_min=float(suppression["moderate_gain_min"]),
        moderate_helpful_loss_max=float(suppression["moderate_helpful_loss_max"]),
        severe_hit_rate_min=float(suppression["severe_hit_rate_min"]),
        severe_gain_min=float(suppression["severe_gain_min"]),
        severe_helpful_loss_max=float(suppression["severe_helpful_loss_max"]),
        max_suppressed_sources=int(suppression["max_suppressed_sources"]),
        suppression_rate_low=float(suppression["suppression_rate_low"]),
        suppression_rate_high=float(suppression["suppression_rate_high"]),
        oracle_harm_delta_threshold=float(suppression["oracle_harm_delta_threshold"]),
        classifier_type=str(classifier["type"]),
        classifier_solver=str(classifier["solver"]),
        classifier_c=float(classifier["C"]),
        classifier_max_iter=int(classifier["max_iter"]),
        classifier_class_weight=str(classifier["class_weight"]),
        classifier_seed=None if classifier.get("classifier_seed") is None else int(classifier["classifier_seed"]),
        skip_nearest_neighbor_audit=bool(memory.get("skip_nearest_neighbor_audit", True)),
    )
    validate_harmful_source_suppression_config(cfg)
    return cfg


def validate_harmful_source_suppression_config(cfg: HarmfulSourceSuppressionConfig) -> None:
    if cfg.name != HARMFUL_SUPPRESSION_NAME:
        raise ProtocolError(f"Harmful-source suppression experiment name must be {HARMFUL_SUPPRESSION_NAME!r}.")
    if cfg.backbone != "virchow2":
        raise ProtocolError("Harmful-source suppression is locked to backbone=virchow2.")
    if cfg.primary_variant != PRIMARY_VARIANT:
        raise ProtocolError(f"primary_variant must be {PRIMARY_VARIANT!r}.")
    if cfg.primary_method != PRIMARY_HARMFUL_SUPPRESSION_METHOD:
        raise ProtocolError(f"primary_method must be {PRIMARY_HARMFUL_SUPPRESSION_METHOD!r}.")
    if cfg.source_weighting != HARMFUL_SUPPRESSION_SOURCE_WEIGHTING:
        raise ProtocolError(f"source_weighting must be {HARMFUL_SUPPRESSION_SOURCE_WEIGHTING!r}.")
    if cfg.candidate_components_per_source_class != (4, 3, 2, 1):
        raise ProtocolError("candidate_components_per_source_class must be locked to [4, 3, 2, 1].")
    if len(cfg.heldout_centers) != 5:
        raise ProtocolError("Harmful-source suppression expects exactly five centers.")
    if cfg.gmm_covariance_type != "diag":
        raise ProtocolError("gmm_covariance_type must be diag.")
    if cfg.primary_pooling != "arithmetic_probability_ensemble":
        raise ProtocolError("primary_pooling must be arithmetic_probability_ensemble.")
    if not math.isclose(cfg.random_mass_bag_alpha, 4.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ProtocolError("random_mass_bag_alpha must be locked to Dirichlet-uniform alpha4.")
    if not math.isclose(cfg.dirichlet_total_concentration, 16.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ProtocolError("dirichlet_total_concentration must be locked to 16.")
    if cfg.max_suppressed_sources != 2:
        raise ProtocolError("max_suppressed_sources must be locked to 2.")
    if cfg.min_harmfulness_observations != 6:
        raise ProtocolError("min_harmfulness_observations must be locked to 6.")
    if not math.isclose(cfg.suppression_rate_low, 0.05, rel_tol=0.0, abs_tol=1.0e-12):
        raise ProtocolError("suppression_rate_low must be locked to 0.05.")
    if not math.isclose(cfg.suppression_rate_high, 0.80, rel_tol=0.0, abs_tol=1.0e-12):
        raise ProtocolError("suppression_rate_high must be locked to 0.80.")
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
    if min(cfg.synthetic_per_class_total, cfg.min_per_source_per_class, cfg.min_samples_per_component, cfg.gmm_n_init, cfg.gmm_max_iter) < 1:
        raise ProtocolError("Budget, component minimums, and GMM iteration settings must be positive.")
    if min(
        cfg.gmm_reg_covar,
        cfg.min_component_weight,
        cfg.variance_floor,
        cfg.variance_ceiling_multiplier,
        cfg.reliability_floor_score,
        cfg.reliability_epsilon,
    ) <= 0.0:
        raise ProtocolError("GMM, variance, and reliability floors must be positive.")
    if cfg.classifier_type != "sklearn_logistic_regression":
        raise ProtocolError("classifier.type must be sklearn_logistic_regression.")
    if cfg.classifier_solver != "lbfgs" or cfg.classifier_c != 1.0 or cfg.classifier_max_iter != 2000:
        raise ProtocolError("Classifier solver/C/max_iter must remain locked.")
    if cfg.classifier_class_weight != "balanced" or cfg.classifier_seed is not None:
        raise ProtocolError("Classifier must use class_weight=balanced and classifier_seed=null.")
    if not cfg.skip_nearest_neighbor_audit:
        raise ProtocolError("Harmful-source suppression v1 must skip nearest-neighbor audit for memory safety.")


def run_harmful_source_suppression(
    cfg: HarmfulSourceSuppressionConfig,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    root = prepare_artifact_dirs(Path(artifact_root) if artifact_root is not None else cfg.artifact_root)
    for rel in ("checkpoints", "summaries", "dense_anchor_summaries", "cache/generated", "cache/predictions"):
        (root / rel).mkdir(parents=True, exist_ok=True)

    matrix_rows: list[dict[str, object]] = []
    bag_member_rows: list[dict[str, object]] = []
    component_manifest_rows: list[dict[str, object]] = []
    component_coverage_rows: list[dict[str, object]] = []
    source_weight_rows: list[dict[str, object]] = []
    source_mass_bag_rows: list[dict[str, object]] = []
    reliability_rows: list[dict[str, object]] = []
    source_summary_rows: list[dict[str, object]] = []
    source_inner_rows: list[dict[str, object]] = []
    harmfulness_summary_rows: list[dict[str, object]] = []
    suppression_rows: list[dict[str, object]] = []
    signal_rows: list[dict[str, object]] = []
    realized_mass_rows: list[dict[str, object]] = []
    source_ablation_rows: list[dict[str, object]] = []
    paired_generation_rows: list[dict[str, object]] = []
    eligibility_rows: list[dict[str, object]] = []
    weak_rows: list[dict[str, object]] = []
    nn_rows: list[dict[str, object]] = []
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
        cfg.dense_tailshield_artifact_root,
    ):
        d1._validate_optional_leakage_report(optional_root, protocol_violations)

    repair_cfg = d1._repair_runtime_config(cfg, root)
    per_source_variant = _per_source_variant()
    print(
        f"[harmful_suppression] start artifact={root} seeds={list(cfg.experiment_seeds)} "
        f"centers={list(cfg.heldout_centers)} reps={list(cfg.all_replicate_seeds)} "
        f"skip_nearest_neighbor_audit={cfg.skip_nearest_neighbor_audit}",
        flush=True,
    )

    try:
        for experiment_seed in cfg.experiment_seeds:
            print(f"[harmful_suppression] seed_start experiment_seed={experiment_seed}", flush=True)
            train_cache = load_feature_cache(_existing_cache_path(cfg.feature_cache_root, seed=experiment_seed, split="train"))
            test_cache = load_feature_cache(_existing_cache_path(cfg.feature_cache_root, seed=experiment_seed, split="test"))
            per_source_runtime: dict[str, RuntimeSource] = {}
            dense_summaries: dict[tuple[str, int], d1a.AdaptiveSourceLocalSummary] = {}
            gmm_summaries: dict[tuple[str, int], d1a.AdaptiveSourceLocalSummary] = {}
            shuffled_summaries: dict[tuple[str, int], d1a.AdaptiveSourceLocalSummary] = {}
            component_details: dict[tuple[str, int, int], dict[str, object]] = {}

            for source_center in cfg.heldout_centers:
                print(f"[harmful_suppression] fit_source_summaries seed={experiment_seed} source={source_center}", flush=True)
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
                    row = d12._source_reliability_row(rel)
                    row["panel"] = _panel_for_replicate_seed(cfg, replicate_seed)
                    reliability_rows.append(row)

            for heldout_center in cfg.heldout_centers:
                print(f"[harmful_suppression] heldout_start seed={experiment_seed} heldout={heldout_center}", flush=True)
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

                source_eval_by_center = _source_eval_cache(test_cache, candidates)
                source_eval_error = _source_eval_error(source_eval_by_center)
                suppression_plan, inner_rows, harm_rows, signal = _source_inner_suppression_plan(
                    cfg,
                    root=root,
                    per_source_runtime=per_source_runtime,
                    summaries=gmm_summaries,
                    reliability=reliability,
                    candidates=candidates,
                    source_eval_by_center=source_eval_by_center,
                    source_eval_error=source_eval_error,
                    experiment_seed=int(experiment_seed),
                    heldout_center=str(heldout_center),
                )
                source_inner_rows.extend(inner_rows)
                harmfulness_summary_rows.extend(harm_rows)
                suppression_rows.extend(_suppression_manifest_rows(cfg, experiment_seed, heldout_center, candidates, suppression_plan, signal))
                signal_rows.extend(_source_inner_signal_rows(cfg, experiment_seed, heldout_center, candidates, suppression_plan, signal, gmm_summaries, reliability))
                print(
                    f"[harmful_suppression] suppression_plan seed={experiment_seed} heldout={heldout_center} "
                    f"suppressed={sum(1 for row in suppression_plan['ranked'] if float(row['multiplier']) < 1.0)}",
                    flush=True,
                )

                target_indices = _target_indices(test_cache.metadata, str(heldout_center))
                eval_raw, eval_meta = select_rows(test_cache.embeddings, test_cache.metadata, target_indices)
                eval_labels = tuple(_label(row) for row in eval_meta)
                eval_error = "mono_class_target_eval" if len(set(eval_labels)) < 2 else ""

                for replicate_seed in cfg.all_replicate_seeds:
                    print(
                        f"[harmful_suppression] cell_start seed={experiment_seed} heldout={heldout_center} rep={replicate_seed}",
                        flush=True,
                    )
                    panel = _panel_for_replicate_seed(cfg, replicate_seed)
                    su_ref = d1._reference_for_cell(source_union_refs, experiment_seed, heldout_center, replicate_seed)
                    cb_ref = d1._reference_for_cell(center_balanced_refs, experiment_seed, heldout_center, replicate_seed)
                    rels = {source: reliability[(int(experiment_seed), int(replicate_seed), str(source))] for source in candidates}
                    if eval_error:
                        rows = _target_ineligible_rows(cfg, experiment_seed, heldout_center, replicate_seed, candidates, su_ref, cb_ref, eval_error)
                        matrix_rows.extend(rows)
                        eligibility_rows.append(mb._eligibility_row(experiment_seed, heldout_center, replicate_seed, "target_eval", "ineligible", eval_error))
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
                    ref_row = mb._normalize_row(ref_row, prior_method=cu.ROW_REAL_FEATURE_DENSE_REFERENCE)
                    ref_row["panel"] = panel
                    matrix_rows.append(ref_row)
                    real_feature_bacc = _float(ref_row["bacc"])

                    dense_rows = mb._dense_comparator_rows(
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
                    for row in dense_rows:
                        row["panel"] = panel
                    matrix_rows.extend(dense_rows)

                    for method, plan in (
                        (cu.PRIMARY_COMPONENT_UNION_METHOD, cu._uniform_source_plan(cfg, candidates, rels, total=cfg.synthetic_per_class_total)),
                        (cu.ROW_COMPONENT_UNION_SHRINK050, cu._shrink_source_plan(cfg, candidates, rels, shrink_lambda=0.5, total=cfg.synthetic_per_class_total)),
                    ):
                        source_weight_rows.extend(cu._source_weight_manifest_rows(int(experiment_seed), int(replicate_seed), str(heldout_center), method, plan, rels, panel=panel))
                        row, coverage, weak, nn, paired_row = cu._evaluate_gmm_component_union(
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
                            weight_plan=plan,
                            prior_method=method,
                            selection_source=DIAGNOSTIC_SELECTION,
                            claim_role="same_run_component_union_reference",
                        )
                        matrix_rows.append(row)
                        component_coverage_rows.append(coverage)
                        paired_generation_rows.append(paired_row)
                        if weak:
                            weak_rows.append(weak)
                        if nn:
                            nn_rows.append(nn)

                    primary_specs = _suppressed_random_mass_bag_specs(
                        cfg,
                        candidates,
                        rels,
                        suppression_plan=suppression_plan,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        replicate_seed=int(replicate_seed),
                        method=PRIMARY_HARMFUL_SUPPRESSION_METHOD,
                        family="source_inner_harm_suppressed",
                    )
                    component_manifest_rows.extend(
                        cu._fold_component_manifest_rows(
                            cfg,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            candidates=candidates,
                            summaries=gmm_summaries,
                            component_details=component_details,
                            weight_plan=primary_specs[0]["plan"],
                        )
                    )
                    primary_eval = _evaluate_and_extend(
                        cfg,
                        root=root,
                        per_source_runtime=per_source_runtime,
                        candidates=candidates,
                        summaries=gmm_summaries,
                        specs=primary_specs,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        replicate_seed=int(replicate_seed),
                        eval_raw=eval_raw,
                        eval_labels=eval_labels,
                        source_union_ref=su_ref,
                        center_balanced_ref=cb_ref,
                        real_feature_bacc=real_feature_bacc,
                        method=PRIMARY_HARMFUL_SUPPRESSION_METHOD,
                        selection_source=PRIMARY_SELECTION,
                        claim_role="primary_source_inner_harm_suppressed_probability_ensemble",
                        matrix_rows=matrix_rows,
                        bag_member_rows=bag_member_rows,
                        component_coverage_rows=component_coverage_rows,
                        paired_generation_rows=paired_generation_rows,
                        weak_rows=weak_rows,
                        nn_rows=nn_rows,
                        source_weight_rows=source_weight_rows,
                        source_mass_bag_rows=source_mass_bag_rows,
                        rels=rels,
                    )
                    eligibility_rows.extend(primary_eval["eligibility_rows"])
                    realized_mass_rows.extend(_realized_bag_mass_rows(cfg, experiment_seed, heldout_center, replicate_seed, primary_specs, suppression_plan))
                    primary_bacc = _float(primary_eval["ensemble_row"].get("bacc"))

                    control_specs = {
                        cu.ROW_RANDOM_MASS_BAG_CONTROL: _plain_random_bag_specs(cfg, candidates, rels, int(experiment_seed), str(heldout_center), int(replicate_seed)),
                        mb.ROW_RANDOM_SINGLE_MASS_CONTROL: _random_single_mass_specs(cfg, candidates, rels, int(experiment_seed), str(heldout_center), int(replicate_seed)),
                        ROW_SHUFFLED_HARMFULNESS_CONTROL: _suppressed_random_mass_bag_specs(
                            cfg,
                            candidates,
                            rels,
                            suppression_plan=_shuffled_suppression_plan(cfg, candidates, suppression_plan, experiment_seed, heldout_center),
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            replicate_seed=int(replicate_seed),
                            method=ROW_SHUFFLED_HARMFULNESS_CONTROL,
                            family="shuffled_harmfulness_suppression",
                        ),
                        ROW_RANDOM_MATCHED_SUPPRESSION_CONTROL: _suppressed_random_mass_bag_specs(
                            cfg,
                            candidates,
                            rels,
                            suppression_plan=_random_matched_suppression_plan(cfg, candidates, suppression_plan, experiment_seed, heldout_center),
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            replicate_seed=int(replicate_seed),
                            method=ROW_RANDOM_MATCHED_SUPPRESSION_CONTROL,
                            family="random_matched_suppression",
                        ),
                        ROW_INVERSE_HARMFULNESS_CONTROL: _suppressed_random_mass_bag_specs(
                            cfg,
                            candidates,
                            rels,
                            suppression_plan=_inverse_suppression_plan(cfg, candidates, suppression_plan, signal),
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            replicate_seed=int(replicate_seed),
                            method=ROW_INVERSE_HARMFULNESS_CONTROL,
                            family="inverse_harmfulness_suppression",
                        ),
                    }
                    for method, specs in control_specs.items():
                        evaluated = _evaluate_and_extend(
                            cfg,
                            root=root,
                            per_source_runtime=per_source_runtime,
                            candidates=candidates,
                            summaries=gmm_summaries,
                            specs=specs,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            replicate_seed=int(replicate_seed),
                            eval_raw=eval_raw,
                            eval_labels=eval_labels,
                            source_union_ref=su_ref,
                            center_balanced_ref=cb_ref,
                            real_feature_bacc=real_feature_bacc,
                            method=method,
                            selection_source=DIAGNOSTIC_SELECTION,
                            claim_role="negative_control" if method != cu.ROW_RANDOM_MASS_BAG_CONTROL else "unsuppressed_random_mass_bag_comparator",
                            matrix_rows=matrix_rows,
                            bag_member_rows=bag_member_rows,
                            component_coverage_rows=component_coverage_rows,
                            paired_generation_rows=paired_generation_rows,
                            weak_rows=weak_rows,
                            nn_rows=nn_rows,
                            source_weight_rows=source_weight_rows,
                            source_mass_bag_rows=source_mass_bag_rows,
                            rels=rels,
                        )
                        eligibility_rows.extend(evaluated["eligibility_rows"])

                    for method, summaries, control_mode in (
                        (cu.ROW_SHUFFLED_LABEL_CONTROL, shuffled_summaries, "normal"),
                        (cu.ROW_SHUFFLED_SUMMARY_CONTROL, gmm_summaries, "class_flip"),
                    ):
                        evaluated = _evaluate_and_extend(
                            cfg,
                            root=root,
                            per_source_runtime=per_source_runtime,
                            candidates=candidates,
                            summaries=summaries,
                            specs=primary_specs,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            replicate_seed=int(replicate_seed),
                            eval_raw=eval_raw,
                            eval_labels=eval_labels,
                            source_union_ref=su_ref,
                            center_balanced_ref=cb_ref,
                            real_feature_bacc=real_feature_bacc,
                            method=method,
                            selection_source=DIAGNOSTIC_SELECTION,
                            claim_role="negative_control",
                            matrix_rows=matrix_rows,
                            bag_member_rows=bag_member_rows,
                            component_coverage_rows=component_coverage_rows,
                            paired_generation_rows=paired_generation_rows,
                            weak_rows=weak_rows,
                            nn_rows=nn_rows,
                            source_weight_rows=source_weight_rows,
                            source_mass_bag_rows=source_mass_bag_rows,
                            rels=rels,
                            control_mode=control_mode,
                        )
                        eligibility_rows.extend(evaluated["eligibility_rows"])

                    hard_sources = tuple(row["source"] for row in suppression_plan["ranked"] if float(row["multiplier"]) <= 0.25)[:1]
                    if hard_sources:
                        hard_candidates = tuple(source for source in candidates if source not in set(hard_sources))
                        hard_specs = _plain_random_bag_specs(cfg, hard_candidates, {source: rels[source] for source in hard_candidates}, int(experiment_seed), str(heldout_center), int(replicate_seed), family="hard_exclusion")
                        evaluated = _evaluate_and_extend(
                            cfg,
                            root=root,
                            per_source_runtime=per_source_runtime,
                            candidates=hard_candidates,
                            summaries=gmm_summaries,
                            specs=hard_specs,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            replicate_seed=int(replicate_seed),
                            eval_raw=eval_raw,
                            eval_labels=eval_labels,
                            source_union_ref=su_ref,
                            center_balanced_ref=cb_ref,
                            real_feature_bacc=real_feature_bacc,
                            method=ROW_HARD_EXCLUSION_DIAGNOSTIC,
                            selection_source=DIAGNOSTIC_SELECTION,
                            claim_role="hard_exclusion_diagnostic_only",
                            matrix_rows=matrix_rows,
                            bag_member_rows=bag_member_rows,
                            component_coverage_rows=component_coverage_rows,
                            paired_generation_rows=paired_generation_rows,
                            weak_rows=weak_rows,
                            nn_rows=nn_rows,
                            source_weight_rows=source_weight_rows,
                            source_mass_bag_rows=source_mass_bag_rows,
                            rels={source: rels[source] for source in hard_candidates},
                        )
                        eligibility_rows.extend(evaluated["eligibility_rows"])

                    source_ablation_rows.extend(
                        _target_random_bag_ablation_rows(
                            cfg,
                            root=root,
                            per_source_runtime=per_source_runtime,
                            summaries=gmm_summaries,
                            rels=rels,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            replicate_seed=int(replicate_seed),
                            candidates=candidates,
                            eval_raw=eval_raw,
                            eval_labels=eval_labels,
                            source_union_ref=su_ref,
                            center_balanced_ref=cb_ref,
                            real_feature_bacc=real_feature_bacc,
                        )
                    )
                    matrix_rows.append(cu._reference_matrix_row(cfg, experiment_seed=int(experiment_seed), heldout_center=str(heldout_center), replicate_seed=int(replicate_seed), candidates=candidates, prior_method=cu.ROW_SOURCE_UNION_K16_REFERENCE, reference=su_ref))
                    matrix_rows.append(cu._reference_matrix_row(cfg, experiment_seed=int(experiment_seed), heldout_center=str(heldout_center), replicate_seed=int(replicate_seed), candidates=candidates, prior_method=cu.ROW_CENTER_BALANCED_K16_REFERENCE, reference=cb_ref))
                    print(
                        f"[harmful_suppression] cell_done seed={experiment_seed} heldout={heldout_center} rep={replicate_seed} "
                        f"primary_bacc={_format_float(primary_bacc)}",
                        flush=True,
                    )
                print(f"[harmful_suppression] heldout_done seed={experiment_seed} heldout={heldout_center}", flush=True)
            print(f"[harmful_suppression] seed_done experiment_seed={experiment_seed}", flush=True)
    except ProtocolError as exc:
        protocol_violations.append(str(exc))

    bottom20_keys = tail._bottom20_raw_cell_keys(matrix_rows, cu.ROW_RANDOM_MASS_BAG_CONTROL)
    alignment_rows = _target_oracle_alignment_rows(cfg, suppression_rows, source_ablation_rows)
    leakage = build_leakage_report(
        target_support_labels_for_selection=False,
        target_eval_labels_for_scoring_only=True,
        target_expert_excluded=target_expert_excluded,
        oracle_rows_diagnostic_only=True,
        extra_violations=protocol_violations,
    )
    decision = _decision(
        cfg,
        matrix_rows,
        leakage_status=leakage.status,
        bottom20_keys=bottom20_keys,
        suppression_rows=suppression_rows,
        alignment_rows=alignment_rows,
    )
    print(
        f"[harmful_suppression] writing_artifacts matrix_rows={len(matrix_rows)} bag_member_rows={len(bag_member_rows)} "
        f"source_inner_rows={len(source_inner_rows)}",
        flush=True,
    )
    _write_artifacts(
        root,
        cfg,
        matrix_rows=matrix_rows,
        bag_member_rows=bag_member_rows,
        component_manifest_rows=component_manifest_rows,
        component_coverage_rows=component_coverage_rows,
        source_weight_rows=source_weight_rows,
        source_mass_bag_rows=source_mass_bag_rows,
        reliability_rows=reliability_rows,
        source_summary_rows=source_summary_rows,
        source_inner_rows=source_inner_rows,
        harmfulness_summary_rows=harmfulness_summary_rows,
        suppression_rows=suppression_rows,
        signal_rows=signal_rows,
        realized_mass_rows=realized_mass_rows,
        source_ablation_rows=source_ablation_rows,
        alignment_rows=alignment_rows,
        paired_generation_rows=paired_generation_rows,
        eligibility_rows=eligibility_rows,
        weak_rows=weak_rows,
        nn_rows=nn_rows,
        model_manifest_rows=model_manifest_rows,
        decision=decision,
        leakage=leakage,
        protocol_violations=protocol_violations,
        target_expert_excluded=target_expert_excluded,
        bottom20_keys=bottom20_keys,
    )
    print(f"[harmful_suppression] done artifact={root}", flush=True)
    return root


def _optional_path(base: Path, value: object) -> Path | None:
    if value is None or str(value) == "":
        return None
    return _path(base, str(value))


def _panel_for_replicate_seed(cfg: HarmfulSourceSuppressionConfig, replicate_seed: object) -> str:
    return "fresh" if int(replicate_seed) in set(cfg.fresh_replicate_seeds) else "canonical"


def _source_eval_cache(test_cache: object, centers: Sequence[str]) -> dict[str, tuple[object, tuple[int, ...]]]:
    out = {}
    for center in centers:
        indices = _target_indices(test_cache.metadata, str(center))
        raw, meta = select_rows(test_cache.embeddings, test_cache.metadata, indices)
        out[str(center)] = (raw, tuple(_label(row) for row in meta))
    return out


def _source_eval_error(source_eval_by_center: Mapping[str, tuple[object, Sequence[int]]]) -> str:
    for center, (_raw, labels) in source_eval_by_center.items():
        if len(set(int(v) for v in labels)) < 2:
            return f"mono_class_source_inner_eval_center_{center}"
    return ""


def _source_inner_suppression_plan(
    cfg: HarmfulSourceSuppressionConfig,
    *,
    root: Path,
    per_source_runtime: Mapping[str, RuntimeSource],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    reliability: Mapping[tuple[int, int, str], d12.SourceReliability],
    candidates: Sequence[str],
    source_eval_by_center: Mapping[str, tuple[object, Sequence[int]]],
    source_eval_error: str,
    experiment_seed: int,
    heldout_center: str,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], dict[str, dict[str, object]]]:
    if source_eval_error:
        plan = _empty_suppression_plan(candidates, source_eval_error)
        return plan, [], _harmfulness_summary_rows(cfg, experiment_seed, heldout_center, candidates, {}, plan), {}
    observation_rows: list[dict[str, object]] = []
    by_source: dict[str, list[float]] = {str(source): [] for source in candidates}
    for replicate_seed in cfg.replicate_seeds:
        for pseudo_target in candidates:
            pseudo_sources = tuple(source for source in candidates if str(source) != str(pseudo_target))
            eval_raw, eval_labels = source_eval_by_center[str(pseudo_target)]
            rels_base = {source: reliability[(int(experiment_seed), int(replicate_seed), str(source))] for source in pseudo_sources}
            baseline = mb._evaluate_bag(
                cfg,
                root=root,
                per_source_runtime=per_source_runtime,
                candidates=pseudo_sources,
                summaries=summaries,
                specs=_plain_random_bag_specs(cfg, pseudo_sources, rels_base, experiment_seed, heldout_center, replicate_seed, family=f"source_inner_baseline_p{pseudo_target}"),
                experiment_seed=experiment_seed,
                heldout_center=heldout_center,
                replicate_seed=replicate_seed,
                eval_raw=eval_raw,
                eval_labels=eval_labels,
                source_union_ref=d1._missing_reference(),
                center_balanced_ref=d1._missing_reference(),
                real_feature_bacc=math.nan,
                method=f"source_inner_baseline_pseudo_{pseudo_target}",
                selection_source=DIAGNOSTIC_SELECTION,
                claim_role="source_inner_baseline_not_target",
            )
            baseline_bacc = _float(baseline["ensemble_row"].get("bacc"))
            for removed in pseudo_sources:
                remaining = tuple(source for source in pseudo_sources if str(source) != str(removed))
                rels_remaining = {source: reliability[(int(experiment_seed), int(replicate_seed), str(source))] for source in remaining}
                removed_eval = mb._evaluate_bag(
                    cfg,
                    root=root,
                    per_source_runtime=per_source_runtime,
                    candidates=remaining,
                    summaries=summaries,
                    specs=_plain_random_bag_specs(cfg, remaining, rels_remaining, experiment_seed, heldout_center, replicate_seed, family=f"source_inner_minus_{removed}_p{pseudo_target}"),
                    experiment_seed=experiment_seed,
                    heldout_center=heldout_center,
                    replicate_seed=replicate_seed,
                    eval_raw=eval_raw,
                    eval_labels=eval_labels,
                    source_union_ref=d1._missing_reference(),
                    center_balanced_ref=d1._missing_reference(),
                    real_feature_bacc=math.nan,
                    method=f"source_inner_minus_{removed}_pseudo_{pseudo_target}",
                    selection_source=DIAGNOSTIC_SELECTION,
                    claim_role="source_inner_leave_one_source_not_target",
                )
                removed_bacc = _float(removed_eval["ensemble_row"].get("bacc"))
                delta = removed_bacc - baseline_bacc if math.isfinite(removed_bacc) and math.isfinite(baseline_bacc) else math.nan
                if math.isfinite(delta):
                    by_source[str(removed)].append(delta)
                observation_rows.append(
                    {
                        "experiment_seed": int(experiment_seed),
                        "heldout_center": str(heldout_center),
                        "replicate_seed": int(replicate_seed),
                        "pseudo_target_source": str(pseudo_target),
                        "candidate_harmful_source": str(removed),
                        "baseline_sources": "|".join(pseudo_sources),
                        "remaining_sources": "|".join(remaining),
                        "baseline_bacc": baseline_bacc,
                        "removed_bacc": removed_bacc,
                        "delta_remove_r_on_p": delta,
                        "source_inner_target_labels_only": True,
                        "heldout_target_rows_used": False,
                        "status": removed_eval["ensemble_row"].get("status", ""),
                    }
                )
    signal = _harmfulness_signal(cfg, candidates, by_source)
    plan = _suppression_plan_from_signal(cfg, candidates, signal)
    return plan, observation_rows, _harmfulness_summary_rows(cfg, experiment_seed, heldout_center, candidates, signal, plan), signal


def _harmfulness_signal(
    cfg: HarmfulSourceSuppressionConfig,
    candidates: Sequence[str],
    by_source: Mapping[str, Sequence[float]],
) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for source in candidates:
        values = [float(v) for v in by_source.get(str(source), []) if math.isfinite(float(v))]
        n = len(values)
        mean = nanmean(values) if values else math.nan
        std = d1._std(values)
        sem = std / math.sqrt(float(n)) if n > 1 and math.isfinite(std) else math.nan
        harmful_gain = nanmean([max(v, 0.0) for v in values]) if values else math.nan
        hit_rate = float(sum(v >= 0.020 for v in values)) / float(n) if n else math.nan
        helpful_loss = nanmean([max(-v, 0.0) for v in values]) if values else math.nan
        net = harmful_gain - 0.5 * helpful_loss if math.isfinite(harmful_gain) and math.isfinite(helpful_loss) else math.nan
        out[str(source)] = {
            "source": str(source),
            "n_harmfulness_observations": n,
            "delta_mean": mean,
            "delta_std": std,
            "delta_sem": sem,
            "delta_ci_low": mean - 1.96 * sem if math.isfinite(mean) and math.isfinite(sem) else math.nan,
            "delta_ci_high": mean + 1.96 * sem if math.isfinite(mean) and math.isfinite(sem) else math.nan,
            "harmful_gain_mean": harmful_gain,
            "harmful_hit_rate_020": hit_rate,
            "helpful_loss_mean": helpful_loss,
            "net_harm_score": net,
            "suppression_decision_stability_across_reps": hit_rate,
            "suppression_ineligible": n < cfg.min_harmfulness_observations,
        }
    return out


def _suppression_plan_from_signal(
    cfg: HarmfulSourceSuppressionConfig,
    candidates: Sequence[str],
    signal: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    rows = []
    for source in candidates:
        item = dict(signal.get(str(source), {}))
        multiplier = 1.0
        severity = "none"
        if not bool(item.get("suppression_ineligible", True)):
            hit = _float(item.get("harmful_hit_rate_020"))
            gain = _float(item.get("harmful_gain_mean"))
            loss = _float(item.get("helpful_loss_mean"))
            if hit >= cfg.severe_hit_rate_min and gain >= cfg.severe_gain_min and loss <= cfg.severe_helpful_loss_max:
                multiplier = 0.25
                severity = "severe"
            elif hit >= cfg.moderate_hit_rate_min and gain >= cfg.moderate_gain_min and loss <= cfg.moderate_helpful_loss_max:
                multiplier = 0.50
                severity = "moderate"
        rows.append({"source": str(source), "candidate_multiplier": multiplier, "candidate_severity": severity, **item})
    selected = sorted(
        [row for row in rows if float(row["candidate_multiplier"]) < 1.0],
        key=lambda row: (-_float(row.get("net_harm_score")), str(row.get("source"))),
    )[: cfg.max_suppressed_sources]
    selected_sources = {str(row["source"]) for row in selected}
    ranked = []
    multipliers: dict[str, float] = {}
    severity_by_source: dict[str, str] = {}
    for row in rows:
        source = str(row["source"])
        multiplier = float(row["candidate_multiplier"]) if source in selected_sources else 1.0
        severity = str(row["candidate_severity"]) if source in selected_sources else "none"
        multipliers[source] = multiplier
        severity_by_source[source] = severity
        ranked.append({**row, "multiplier": multiplier, "severity": severity, "selected_for_suppression": source in selected_sources})
    base = {source: 1.0 / float(len(candidates)) for source in (str(v) for v in candidates)}
    adjusted = {source: base[source] * multipliers[source] for source in base}
    total = sum(adjusted.values())
    masses = {source: adjusted[source] / total for source in adjusted}
    return {
        "multipliers": multipliers,
        "severity": severity_by_source,
        "center_masses": masses,
        "ranked": ranked,
        "suppressed_source_count": sum(1 for value in multipliers.values() if value < 1.0),
        "severe_source_count": sum(1 for value in severity_by_source.values() if value == "severe"),
        "moderate_source_count": sum(1 for value in severity_by_source.values() if value == "moderate"),
        "suppression_rate": float(sum(1 for value in multipliers.values() if value < 1.0)) / float(len(multipliers)) if multipliers else math.nan,
    }


def _empty_suppression_plan(candidates: Sequence[str], reason: str) -> dict[str, object]:
    sources = tuple(str(source) for source in candidates)
    masses = {source: 1.0 / float(len(sources)) for source in sources}
    return {
        "multipliers": {source: 1.0 for source in sources},
        "severity": {source: "none" for source in sources},
        "center_masses": masses,
        "ranked": [{"source": source, "multiplier": 1.0, "severity": "none", "selected_for_suppression": False, "suppression_ineligible": True, "ineligible_reason": reason} for source in sources],
        "suppressed_source_count": 0,
        "severe_source_count": 0,
        "moderate_source_count": 0,
        "suppression_rate": 0.0,
    }


def _plain_random_bag_specs(
    cfg: HarmfulSourceSuppressionConfig,
    sources: Sequence[str],
    rels: Mapping[str, d12.SourceReliability],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    *,
    family: str = "random_mass_bag_uniform_alpha4",
) -> list[dict[str, object]]:
    center = {str(source): 1.0 / float(len(sources)) for source in sources}
    return _dirichlet_bag_specs(cfg, sources, rels, center, experiment_seed, heldout_center, replicate_seed, method=cu.ROW_RANDOM_MASS_BAG_CONTROL, family=family)


def _random_single_mass_specs(
    cfg: HarmfulSourceSuppressionConfig,
    sources: Sequence[str],
    rels: Mapping[str, d12.SourceReliability],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
) -> list[dict[str, object]]:
    center = {str(source): 1.0 / float(len(sources)) for source in sources}
    spec = _dirichlet_bag_specs(cfg, sources, rels, center, experiment_seed, heldout_center, replicate_seed, method=mb.ROW_RANDOM_SINGLE_MASS_CONTROL, family="random_single_mass_alpha1", size=1)[0]
    spec["method"] = mb.ROW_RANDOM_SINGLE_MASS_CONTROL
    spec["bag_member_id"] = "random_single_mass_alpha1_perm000"
    return [spec]


def _suppressed_random_mass_bag_specs(
    cfg: HarmfulSourceSuppressionConfig,
    sources: Sequence[str],
    rels: Mapping[str, d12.SourceReliability],
    *,
    suppression_plan: Mapping[str, object],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    method: str,
    family: str,
) -> list[dict[str, object]]:
    return _dirichlet_bag_specs(
        cfg,
        sources,
        rels,
        {str(source): float(dict(suppression_plan["center_masses"])[str(source)]) for source in sources},
        experiment_seed,
        heldout_center,
        replicate_seed,
        method=method,
        family=family,
    )


def _dirichlet_bag_specs(
    cfg: HarmfulSourceSuppressionConfig,
    sources: Sequence[str],
    rels: Mapping[str, d12.SourceReliability],
    center_weights: Mapping[str, float],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    *,
    method: str,
    family: str,
    size: int | None = None,
) -> list[dict[str, object]]:
    sources_tuple = tuple(str(source) for source in sources)
    n = int(size if size is not None else cfg.random_mass_bag_size)
    specs = []
    for idx in range(n):
        alpha_per_source = cfg.dirichlet_total_concentration / float(len(sources_tuple))
        plan = mb._dirichlet_source_plan(
            cfg,
            sources_tuple,
            rels,
            center_weights=center_weights,
            alpha_per_source=alpha_per_source,
            family=family,
            permutation_id=idx,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
        )
        member_id = f"{family}_perm{idx:03d}"
        plan.update({"bag_member_id": member_id, "dirichlet_total_concentration": cfg.dirichlet_total_concentration})
        specs.append({"bag_member_id": member_id, "bag_member_index": idx, "bag_member_family": family, "method": f"{method}__member_{idx:03d}", "plan": plan})
    return specs


def _shuffled_suppression_plan(
    cfg: HarmfulSourceSuppressionConfig,
    candidates: Sequence[str],
    plan: Mapping[str, object],
    experiment_seed: object,
    heldout_center: object,
) -> dict[str, object]:
    sources = tuple(str(source) for source in candidates)
    multipliers = [float(dict(plan["multipliers"])[source]) for source in sources]
    severities = [str(dict(plan["severity"])[source]) for source in sources]
    rng = random.Random(d1._latent_seed(experiment_seed, heldout_center, "shuffled_harmfulness_assignment"))
    paired_values = list(zip(multipliers, severities))
    rng.shuffle(paired_values)
    return _plan_from_assigned_values(candidates, paired_values)


def _random_matched_suppression_plan(
    cfg: HarmfulSourceSuppressionConfig,
    candidates: Sequence[str],
    plan: Mapping[str, object],
    experiment_seed: object,
    heldout_center: object,
) -> dict[str, object]:
    sources = tuple(str(source) for source in candidates)
    selected_values = [(float(dict(plan["multipliers"])[source]), str(dict(plan["severity"])[source])) for source in sources if float(dict(plan["multipliers"])[source]) < 1.0]
    values = [(1.0, "none") for _ in sources]
    rng = random.Random(d1._latent_seed(experiment_seed, heldout_center, "random_matched_suppression"))
    positions = list(range(len(sources)))
    rng.shuffle(positions)
    for pos, selected in zip(positions, selected_values):
        values[pos] = selected
    return _plan_from_assigned_values(candidates, values)


def _inverse_suppression_plan(
    cfg: HarmfulSourceSuppressionConfig,
    candidates: Sequence[str],
    plan: Mapping[str, object],
    signal: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    suppressed_values = sorted(
        [(float(dict(plan["multipliers"])[source]), str(dict(plan["severity"])[source])) for source in (str(v) for v in candidates) if float(dict(plan["multipliers"])[source]) < 1.0],
        key=lambda item: item[0],
    )
    ordered_sources = sorted((str(source) for source in candidates), key=lambda source: (_float(signal.get(source, {}).get("net_harm_score")), source))
    assigned = {source: (1.0, "none") for source in ordered_sources}
    for source, value in zip(ordered_sources, suppressed_values):
        assigned[source] = value
    return _plan_from_assigned_values(candidates, [assigned[str(source)] for source in candidates])


def _plan_from_assigned_values(candidates: Sequence[str], values: Sequence[tuple[float, str]]) -> dict[str, object]:
    sources = tuple(str(source) for source in candidates)
    multipliers = {source: float(value[0]) for source, value in zip(sources, values)}
    severity = {source: str(value[1]) for source, value in zip(sources, values)}
    base = {source: 1.0 / float(len(sources)) for source in sources}
    adjusted = {source: base[source] * multipliers[source] for source in sources}
    total = sum(adjusted.values())
    masses = {source: adjusted[source] / total for source in sources}
    return {
        "multipliers": multipliers,
        "severity": severity,
        "center_masses": masses,
        "ranked": [{"source": source, "multiplier": multipliers[source], "severity": severity[source], "selected_for_suppression": multipliers[source] < 1.0} for source in sources],
        "suppressed_source_count": sum(1 for value in multipliers.values() if value < 1.0),
        "severe_source_count": sum(1 for value in severity.values() if value == "severe"),
        "moderate_source_count": sum(1 for value in severity.values() if value == "moderate"),
        "suppression_rate": float(sum(1 for value in multipliers.values() if value < 1.0)) / float(len(sources)) if sources else math.nan,
    }


def _evaluate_and_extend(
    cfg: HarmfulSourceSuppressionConfig,
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
    matrix_rows: list[dict[str, object]],
    bag_member_rows: list[dict[str, object]],
    component_coverage_rows: list[dict[str, object]],
    paired_generation_rows: list[dict[str, object]],
    weak_rows: list[dict[str, object]],
    nn_rows: list[dict[str, object]],
    source_weight_rows: list[dict[str, object]],
    source_mass_bag_rows: list[dict[str, object]],
    rels: Mapping[str, d12.SourceReliability],
    control_mode: str = "normal",
) -> dict[str, object]:
    evaluated = mb._evaluate_bag(
        cfg,
        root=root,
        per_source_runtime=per_source_runtime,
        candidates=candidates,
        summaries=summaries,
        specs=specs,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        eval_raw=eval_raw,
        eval_labels=eval_labels,
        source_union_ref=source_union_ref,
        center_balanced_ref=center_balanced_ref,
        real_feature_bacc=real_feature_bacc,
        method=method,
        selection_source=selection_source,
        claim_role=claim_role,
        control_mode=control_mode,
    )
    mb._extend_run_outputs(
        evaluated,
        matrix_rows,
        bag_member_rows,
        component_coverage_rows,
        paired_generation_rows,
        weak_rows,
        nn_rows,
        source_weight_rows,
        source_mass_bag_rows,
        rels,
    )
    return evaluated


def _target_random_bag_ablation_rows(
    cfg: HarmfulSourceSuppressionConfig,
    *,
    root: Path,
    per_source_runtime: Mapping[str, RuntimeSource],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    rels: Mapping[str, d12.SourceReliability],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    eval_raw: object,
    eval_labels: Sequence[int],
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
    real_feature_bacc: float,
) -> list[dict[str, object]]:
    baseline = mb._evaluate_bag(
        cfg,
        root=root,
        per_source_runtime=per_source_runtime,
        candidates=candidates,
        summaries=summaries,
        specs=_plain_random_bag_specs(cfg, candidates, rels, experiment_seed, heldout_center, replicate_seed),
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        eval_raw=eval_raw,
        eval_labels=eval_labels,
        source_union_ref=source_union_ref,
        center_balanced_ref=center_balanced_ref,
        real_feature_bacc=real_feature_bacc,
        method="target_ablation_baseline_random_mass_bag",
        selection_source=DIAGNOSTIC_SELECTION,
        claim_role="target_ablation_baseline_diagnostic_only",
    )
    baseline_bacc = _float(baseline["ensemble_row"].get("bacc"))
    rows = []
    for removed in cfg.heldout_centers:
        if str(removed) == str(heldout_center):
            rows.append(
                {
                    "experiment_seed": int(experiment_seed),
                    "heldout_center": str(heldout_center),
                    "replicate_seed": int(replicate_seed),
                    "panel": _panel_for_replicate_seed(cfg, replicate_seed),
                    "removed_source_center": str(removed),
                    "remaining_source_centers": "|".join(str(v) for v in candidates),
                    "primary_bacc": baseline_bacc,
                    "ablation_bacc": "",
                    "delta_ablation_minus_primary": "",
                    "status": "not_applicable_target_source_excluded",
                    "target_ablation_diagnostic_only": True,
                    "ablation_reference_method": cu.ROW_RANDOM_MASS_BAG_CONTROL,
                }
            )
            continue
        remaining = tuple(source for source in candidates if str(source) != str(removed))
        remaining_rels = {source: rels[source] for source in remaining}
        evaluated = mb._evaluate_bag(
            cfg,
            root=root,
            per_source_runtime=per_source_runtime,
            candidates=remaining,
            summaries=summaries,
            specs=_plain_random_bag_specs(cfg, remaining, remaining_rels, experiment_seed, heldout_center, replicate_seed, family=f"target_ablation_minus_{removed}"),
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            eval_raw=eval_raw,
            eval_labels=eval_labels,
            source_union_ref=source_union_ref,
            center_balanced_ref=center_balanced_ref,
            real_feature_bacc=real_feature_bacc,
            method=f"target_ablation_minus_{removed}",
            selection_source=DIAGNOSTIC_SELECTION,
            claim_role="target_ablation_diagnostic_only",
        )
        ablation_bacc = _float(evaluated["ensemble_row"].get("bacc"))
        rows.append(
            {
                "experiment_seed": int(experiment_seed),
                "heldout_center": str(heldout_center),
                "replicate_seed": int(replicate_seed),
                "panel": _panel_for_replicate_seed(cfg, replicate_seed),
                "removed_source_center": str(removed),
                "remaining_source_centers": "|".join(str(v) for v in remaining),
                "primary_bacc": baseline_bacc,
                "ablation_bacc": ablation_bacc,
                "delta_ablation_minus_primary": ablation_bacc - baseline_bacc if math.isfinite(ablation_bacc) and math.isfinite(baseline_bacc) else math.nan,
                "status": evaluated["ensemble_row"].get("status", ""),
                "target_ablation_diagnostic_only": True,
                "ablation_reference_method": cu.ROW_RANDOM_MASS_BAG_CONTROL,
            }
        )
    return rows


def _target_ineligible_rows(
    cfg: HarmfulSourceSuppressionConfig,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
    error_message: str,
) -> list[dict[str, object]]:
    methods = (
        PRIMARY_HARMFUL_SUPPRESSION_METHOD,
        cu.ROW_RANDOM_MASS_BAG_CONTROL,
        mb.ROW_RANDOM_SINGLE_MASS_CONTROL,
        cu.PRIMARY_COMPONENT_UNION_METHOD,
        cu.ROW_COMPONENT_UNION_SHRINK050,
        paired.ROW_RELIABILITY_ALL4_WEIGHTED,
        paired.ROW_EQUAL_ALL4,
        ROW_SHUFFLED_HARMFULNESS_CONTROL,
        ROW_RANDOM_MATCHED_SUPPRESSION_CONTROL,
        ROW_INVERSE_HARMFULNESS_CONTROL,
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


def _harmfulness_summary_rows(
    cfg: HarmfulSourceSuppressionConfig,
    experiment_seed: int,
    heldout_center: str,
    candidates: Sequence[str],
    signal: Mapping[str, Mapping[str, object]],
    plan: Mapping[str, object],
) -> list[dict[str, object]]:
    rows = []
    multipliers = dict(plan.get("multipliers", {}))
    severity = dict(plan.get("severity", {}))
    for source in candidates:
        item = dict(signal.get(str(source), {}))
        rows.append(
            {
                "experiment_seed": int(experiment_seed),
                "heldout_center": str(heldout_center),
                "candidate_source": str(source),
                "multiplier": multipliers.get(str(source), 1.0),
                "severity": severity.get(str(source), "none"),
                "selected_for_suppression": float(multipliers.get(str(source), 1.0)) < 1.0,
                **item,
            }
        )
    return rows


def _suppression_manifest_rows(
    cfg: HarmfulSourceSuppressionConfig,
    experiment_seed: int,
    heldout_center: str,
    candidates: Sequence[str],
    plan: Mapping[str, object],
    signal: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    rows = []
    masses = dict(plan["center_masses"])
    multipliers = dict(plan["multipliers"])
    severity = dict(plan["severity"])
    for source in candidates:
        item = dict(signal.get(str(source), {}))
        rows.append(
            {
                "experiment_seed": int(experiment_seed),
                "heldout_center": str(heldout_center),
                "source_center": str(source),
                "source_inner_multiplier": multipliers[str(source)],
                "suppression_severity": severity[str(source)],
                "selected_for_suppression": float(multipliers[str(source)]) < 1.0,
                "intended_source_mass": masses[str(source)],
                "suppressed_source_count_per_cell": plan["suppressed_source_count"],
                "severe_source_count_per_cell": plan["severe_source_count"],
                "moderate_source_count_per_cell": plan["moderate_source_count"],
                "suppression_rate": plan["suppression_rate"],
                "mean_effective_entropy_of_source_weights": _entropy(list(float(v) for v in masses.values())),
                "target_eval_metric_used_for_suppression": False,
                **item,
            }
        )
    return rows


def _source_inner_signal_rows(
    cfg: HarmfulSourceSuppressionConfig,
    experiment_seed: int,
    heldout_center: str,
    candidates: Sequence[str],
    plan: Mapping[str, object],
    signal: Mapping[str, Mapping[str, object]],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    reliability: Mapping[tuple[int, int, str], d12.SourceReliability],
) -> list[dict[str, object]]:
    rows = []
    for source in candidates:
        entropies = [_float(getattr(summary, "component_entropy", math.nan)) for key, summary in summaries.items() if key[0] == str(source)]
        rel_values = [
            reliability[(int(experiment_seed), int(rep), str(source))].reliability_score
            for rep in cfg.replicate_seeds
            if (int(experiment_seed), int(rep), str(source)) in reliability
        ]
        rows.append(
            {
                "experiment_seed": int(experiment_seed),
                "heldout_center": str(heldout_center),
                "source_center": str(source),
                "component_entropy_mean": nanmean(entropies) if entropies else math.nan,
                "source_reliability_mean": nanmean(rel_values) if rel_values else math.nan,
                "source_reliability_std": d1._std(rel_values),
                "source_contribution_dominance_pre_suppression": 1.0 / float(len(candidates)),
                "source_contribution_dominance_post_suppression": dict(plan["center_masses"])[str(source)],
                **dict(signal.get(str(source), {})),
            }
        )
    return rows


def _realized_bag_mass_rows(
    cfg: HarmfulSourceSuppressionConfig,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    specs: Sequence[Mapping[str, object]],
    plan: Mapping[str, object],
) -> list[dict[str, object]]:
    rows = []
    sources = tuple(dict(plan["center_masses"]).keys())
    budgets_by_source: dict[str, list[int]] = {source: [] for source in sources}
    weights_by_source: dict[str, list[float]] = {source: [] for source in sources}
    for spec in specs:
        weights = dict(spec["plan"]["weights"])
        budgets = dict(spec["plan"]["budgets"])
        for source in sources:
            weights_by_source[source].append(float(weights[source]))
            budgets_by_source[source].append(int(budgets[source]))
    for source in sources:
        weights = weights_by_source[source]
        budgets = budgets_by_source[source]
        intended = float(dict(plan["center_masses"])[source])
        severity = str(dict(plan["severity"])[source])
        severe = severity == "severe"
        floor_binding = any(int(v) <= cfg.min_per_source_per_class for v in budgets)
        rows.append(
            {
                "experiment_seed": int(experiment_seed),
                "heldout_center": str(heldout_center),
                "replicate_seed": int(replicate_seed),
                "panel": _panel_for_replicate_seed(cfg, replicate_seed),
                "source_center": source,
                "suppression_severity": severity,
                "intended_source_mass": intended,
                "realized_mean_source_mass": nanmean(weights),
                "realized_max_source_mass": max(weights, default=math.nan),
                "realized_min_source_mass": min(weights, default=math.nan),
                "realized_std_source_mass": d1._std(weights),
                "realized_budget_per_source": nanmean(budgets),
                "floor_binding": floor_binding,
                "severe_source_realized_mass_mean": nanmean(weights) if severe else "",
                "severe_source_realized_mass_max": max(weights, default=math.nan) if severe else "",
            }
        )
    return rows


def _target_oracle_alignment_rows(
    cfg: HarmfulSourceSuppressionConfig,
    suppression_rows: Sequence[Mapping[str, object]],
    source_ablation_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    suppression_by_key = {
        (str(row.get("experiment_seed")), str(row.get("heldout_center")), str(row.get("source_center"))): row
        for row in suppression_rows
    }
    rows = []
    for ablation in source_ablation_rows:
        if ablation.get("status") != "ok":
            continue
        key = (str(ablation.get("experiment_seed")), str(ablation.get("heldout_center")), str(ablation.get("removed_source_center")))
        suppression = suppression_by_key.get(key, {})
        delta = _float(ablation.get("delta_ablation_minus_primary"))
        oracle_harmful = math.isfinite(delta) and delta >= cfg.oracle_harm_delta_threshold
        oracle_helpful = math.isfinite(delta) and delta <= -cfg.oracle_harm_delta_threshold
        suppressed = str(suppression.get("selected_for_suppression")) == "True" or suppression.get("selected_for_suppression") is True
        rows.append(
            {
                "experiment_seed": ablation.get("experiment_seed"),
                "heldout_center": ablation.get("heldout_center"),
                "replicate_seed": ablation.get("replicate_seed"),
                "panel": ablation.get("panel", ""),
                "source_center": ablation.get("removed_source_center"),
                "suppressed_by_source_inner": suppressed,
                "source_inner_multiplier": suppression.get("source_inner_multiplier", ""),
                "suppression_severity": suppression.get("suppression_severity", ""),
                "target_ablation_delta_remove_source": delta,
                "target_oracle_harmful": oracle_harmful,
                "target_oracle_helpful": oracle_helpful,
                "suppressed_source_oracle_harmful": bool(suppressed and oracle_harmful),
                "suppressed_source_oracle_helpful": bool(suppressed and oracle_helpful),
                "unsuppressed_source_oracle_harmful": bool((not suppressed) and oracle_harmful),
                "target_oracle_alignment_audit_only": True,
            }
        )
    summary = _alignment_summary_fields(rows)
    for row in rows:
        row.update(summary)
    return rows


def _alignment_summary_fields(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    suppressed = [row for row in rows if str(row.get("suppressed_by_source_inner")) == "True" or row.get("suppressed_by_source_inner") is True]
    unsuppressed = [row for row in rows if row not in suppressed]
    return {
        "suppressed_source_oracle_harmful_rate": _rate(suppressed, "target_oracle_harmful"),
        "suppressed_source_oracle_helpful_rate": _rate(suppressed, "target_oracle_helpful"),
        "unsuppressed_source_oracle_harmful_rate": _rate(unsuppressed, "target_oracle_harmful"),
        "target_oracle_alignment_positive": _rate(suppressed, "target_oracle_harmful") > _rate(suppressed, "target_oracle_helpful") if suppressed else False,
    }


def _rate(rows: Sequence[Mapping[str, object]], field: str) -> float:
    if not rows:
        return math.nan
    return float(sum(bool(row.get(field)) for row in rows)) / float(len(rows))


def _decision(
    cfg: HarmfulSourceSuppressionConfig,
    rows: Sequence[Mapping[str, object]],
    *,
    leakage_status: str,
    bottom20_keys: set[tuple[str, str, str]],
    suppression_rows: Sequence[Mapping[str, object]],
    alignment_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    primary = tail._tail_metrics(rows, PRIMARY_HARMFUL_SUPPRESSION_METHOD, bottom20_keys=bottom20_keys)
    random_bag = tail._tail_metrics(rows, cu.ROW_RANDOM_MASS_BAG_CONTROL, bottom20_keys=bottom20_keys)
    shrink050 = tail._tail_metrics(rows, cu.ROW_COMPONENT_UNION_SHRINK050, bottom20_keys=bottom20_keys)
    dense = tail._tail_metrics(rows, paired.ROW_RELIABILITY_ALL4_WEIGHTED, bottom20_keys=bottom20_keys)
    source_union = tail._tail_metrics(rows, cu.ROW_SOURCE_UNION_K16_REFERENCE, bottom20_keys=bottom20_keys)
    control_methods = (ROW_SHUFFLED_HARMFULNESS_CONTROL, ROW_RANDOM_MATCHED_SUPPRESSION_CONTROL, ROW_INVERSE_HARMFULNESS_CONTROL, cu.ROW_SHUFFLED_LABEL_CONTROL, cu.ROW_SHUFFLED_SUMMARY_CONTROL)
    controls = {method: tail._tail_metrics(rows, method, bottom20_keys=bottom20_keys) for method in control_methods}
    primary_bacc = _float(primary.get("center_equal_mean_bacc"))
    bag_bacc = _float(random_bag.get("center_equal_mean_bacc"))
    source_union_bacc = _float(source_union.get("center_equal_mean_bacc"))
    suppression_rates = [_float(row.get("suppression_rate")) for row in suppression_rows if math.isfinite(_float(row.get("suppression_rate")))]
    suppression_rate = nanmean(suppression_rates) if suppression_rates else math.nan
    alignment = _alignment_summary_fields(alignment_rows)
    center3_delta = _delta(primary.get("center3_bacc"), random_bag.get("center3_bacc"))
    bottom20_delta = _delta(primary.get("bottom20_cell_mean_bacc"), random_bag.get("bottom20_cell_mean_bacc"))
    worst_delta = _delta(primary.get("worst_seed_center_bacc"), random_bag.get("worst_seed_center_bacc"))
    mean_delta = primary_bacc - bag_bacc if math.isfinite(primary_bacc) and math.isfinite(bag_bacc) else math.nan
    no_bad_cell = _no_seed_center_worse_than_bag(rows, primary, bottom20_keys)
    controls_beaten = all(_float(primary.get("bottom20_cell_mean_bacc")) > _float(stats.get("bottom20_cell_mean_bacc")) for stats in controls.values() if math.isfinite(_float(stats.get("bottom20_cell_mean_bacc"))))
    suppression_collapsed = math.isfinite(suppression_rate) and (suppression_rate <= cfg.suppression_rate_low or suppression_rate >= cfg.suppression_rate_high)
    strong = (
        leakage_status == "PASS"
        and int(primary.get("n_heldout_centers", 0)) >= len(cfg.heldout_centers)
        and mean_delta >= -0.005
        and _float(primary.get("min_center_bacc")) >= 0.82
        and center3_delta >= 0.025
        and bottom20_delta >= 0.025
        and worst_delta >= 0.030
        and _float(primary.get("seed_std_bacc")) <= min(0.045, _float(random_bag.get("seed_std_bacc")))
        and no_bad_cell
        and controls_beaten
        and not suppression_collapsed
        and bool(alignment["target_oracle_alignment_positive"])
    )
    useful = (
        leakage_status == "PASS"
        and mean_delta >= -0.007
        and (center3_delta > 0.0 or bottom20_delta > 0.0)
        and worst_delta > 0.0
        and controls_beaten
        and not suppression_collapsed
    )
    flags = []
    if leakage_status != "PASS":
        flags.append("LEAKAGE_FAIL")
    if mean_delta < -0.005:
        flags.append("MEAN_DROP_VS_RANDOM_MASS_BAG")
    if _float(primary.get("min_center_bacc")) < 0.80:
        flags.append("MIN_CENTER_BELOW_0P80")
    if center3_delta <= 0.0:
        flags.append("CENTER3_NOT_IMPROVED")
    if bottom20_delta <= 0.0:
        flags.append("BOTTOM20_NOT_IMPROVED")
    if suppression_collapsed:
        flags.append("SUPPRESSION_RATE_COLLAPSED")
    if not controls_beaten:
        flags.append("NEGATIVE_CONTROLS_MATCH_TAIL_GAIN")
    if not bool(alignment["target_oracle_alignment_positive"]):
        flags.append("TARGET_ORACLE_ALIGNMENT_NOT_POSITIVE")
    verdict = "HARMFUL_SOURCE_SUPPRESSION_FAIL"
    if leakage_status != "PASS":
        verdict = "PROTOCOL_FAIL"
    elif strong:
        verdict = "HARMFUL_SOURCE_SUPPRESSION_STRONG_SUCCESS"
    elif useful:
        verdict = "HARMFUL_SOURCE_SUPPRESSION_USEFUL_THESIS_SUCCESS"
    return {
        "primary_verdict": verdict,
        "diagnostic_flags": "|".join(flags),
        "primary_method": PRIMARY_HARMFUL_SUPPRESSION_METHOD,
        "leakage_status": leakage_status,
        "center_equal_mean_bacc": primary.get("center_equal_mean_bacc"),
        "seed_cell_mean_bacc": primary.get("seed_cell_mean_bacc"),
        "center_equal_macro_f1": primary.get("center_equal_macro_f1"),
        "min_center_bacc": primary.get("min_center_bacc"),
        "seed_std_bacc": primary.get("seed_std_bacc"),
        "bottom20_cell_mean_bacc": primary.get("bottom20_cell_mean_bacc"),
        "worst_seed_center_bacc": primary.get("worst_seed_center_bacc"),
        "center3_bacc": primary.get("center3_bacc"),
        "random_mass_bag_center_equal_mean_bacc": bag_bacc,
        "delta_vs_random_mass_bag": mean_delta,
        "center3_delta_vs_random_mass_bag": center3_delta,
        "bottom20_delta_vs_random_mass_bag": bottom20_delta,
        "worst_seed_center_delta_vs_random_mass_bag": worst_delta,
        "bottom20_delta_vs_shrink050": _delta(primary.get("bottom20_cell_mean_bacc"), shrink050.get("bottom20_cell_mean_bacc")),
        "center3_delta_vs_shrink050": _delta(primary.get("center3_bacc"), shrink050.get("center3_bacc")),
        "dense_reliability_center_equal_mean_bacc": dense.get("center_equal_mean_bacc"),
        "retention_vs_source_union_k16": d1._retention(primary_bacc, source_union_bacc),
        "oracle_gap_vs_source_union_k16": source_union_bacc - primary_bacc if math.isfinite(source_union_bacc) and math.isfinite(primary_bacc) else math.nan,
        "suppression_rate": suppression_rate,
        "suppression_collapsed": suppression_collapsed,
        "controls_beaten_on_tail": controls_beaten,
        "no_seed_center_worse_than_random_bag_gt_0p030": no_bad_cell,
        **alignment,
        **primary,
    }


def _no_seed_center_worse_than_bag(
    rows: Sequence[Mapping[str, object]],
    primary_stats: Mapping[str, object],
    bottom20_keys: set[tuple[str, str, str]],
) -> bool:
    primary_rows = {
        (str(row.get("experiment_seed")), str(row.get("heldout_center")), str(row.get("replicate_seed"))): _float(row.get("bacc"))
        for row in cu._rows_for(rows, PRIMARY_HARMFUL_SUPPRESSION_METHOD)
        if row.get("status") == "ok"
    }
    bag_rows = {
        (str(row.get("experiment_seed")), str(row.get("heldout_center")), str(row.get("replicate_seed"))): _float(row.get("bacc"))
        for row in cu._rows_for(rows, cu.ROW_RANDOM_MASS_BAG_CONTROL)
        if row.get("status") == "ok"
    }
    primary_mean = _float(primary_stats.get("center_equal_mean_bacc"))
    bag_mean = _float(tail._tail_metrics(rows, cu.ROW_RANDOM_MASS_BAG_CONTROL, bottom20_keys=bottom20_keys).get("center_equal_mean_bacc"))
    tail_positive = _delta(primary_stats.get("bottom20_cell_mean_bacc"), tail._tail_metrics(rows, cu.ROW_RANDOM_MASS_BAG_CONTROL, bottom20_keys=bottom20_keys).get("bottom20_cell_mean_bacc")) > 0.0
    for key, bag_value in bag_rows.items():
        primary_value = primary_rows.get(key, math.nan)
        if math.isfinite(primary_value) and math.isfinite(bag_value) and primary_value < bag_value - 0.030:
            return bool(math.isfinite(primary_mean) and math.isfinite(bag_mean) and primary_mean > bag_mean and tail_positive)
    return True


def _delta(a: object, b: object) -> float:
    av = _float(a)
    bv = _float(b)
    return av - bv if math.isfinite(av) and math.isfinite(bv) else math.nan


def _entropy(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if float(v) > 0.0]
    total = sum(finite)
    if total <= 0.0:
        return math.nan
    probs = [value / total for value in finite]
    return float(-sum(p * math.log(p) for p in probs))


def _write_artifacts(
    root: Path,
    cfg: HarmfulSourceSuppressionConfig,
    *,
    matrix_rows: Sequence[Mapping[str, object]],
    bag_member_rows: Sequence[Mapping[str, object]],
    component_manifest_rows: Sequence[Mapping[str, object]],
    component_coverage_rows: Sequence[Mapping[str, object]],
    source_weight_rows: Sequence[Mapping[str, object]],
    source_mass_bag_rows: Sequence[Mapping[str, object]],
    reliability_rows: Sequence[Mapping[str, object]],
    source_summary_rows: Sequence[Mapping[str, object]],
    source_inner_rows: Sequence[Mapping[str, object]],
    harmfulness_summary_rows: Sequence[Mapping[str, object]],
    suppression_rows: Sequence[Mapping[str, object]],
    signal_rows: Sequence[Mapping[str, object]],
    realized_mass_rows: Sequence[Mapping[str, object]],
    source_ablation_rows: Sequence[Mapping[str, object]],
    alignment_rows: Sequence[Mapping[str, object]],
    paired_generation_rows: Sequence[Mapping[str, object]],
    eligibility_rows: Sequence[Mapping[str, object]],
    weak_rows: Sequence[Mapping[str, object]],
    nn_rows: Sequence[Mapping[str, object]],
    model_manifest_rows: Sequence[Mapping[str, object]],
    decision: Mapping[str, object],
    leakage: object,
    protocol_violations: Sequence[str],
    target_expert_excluded: bool,
    bottom20_keys: set[tuple[str, str, str]],
) -> None:
    write_csv_rows(root / "tables" / "harmful_source_suppression_downstream_matrix.csv", matrix_rows)
    write_csv_rows(root / "tables" / "harmful_source_suppression_summary.csv", [dict(decision)])
    write_csv_rows(root / "tables" / "harmful_source_suppression_panel_summary.csv", tail._panel_summary_rows(matrix_rows))
    write_csv_rows(root / "tables" / "harmful_source_suppression_tail_metric_summary.csv", tail._tail_metric_summary_rows(matrix_rows, bottom20_keys))
    write_csv_rows(root / "tables" / "source_inner_harmfulness_matrix.csv", source_inner_rows)
    write_csv_rows(root / "tables" / "source_inner_harmfulness_summary.csv", harmfulness_summary_rows)
    write_csv_rows(root / "tables" / "source_inner_suppression_manifest.csv", suppression_rows)
    write_csv_rows(root / "tables" / "source_inner_signal_audit.csv", signal_rows)
    write_csv_rows(root / "tables" / "realized_bag_mass_audit.csv", realized_mass_rows)
    write_csv_rows(root / "tables" / "harmfulness_target_oracle_alignment_audit.csv", alignment_rows)
    write_csv_rows(root / "tables" / "source_weight_manifest.csv", source_weight_rows)
    write_csv_rows(root / "tables" / "source_mass_bag_manifest.csv", source_mass_bag_rows)
    write_csv_rows(root / "tables" / "source_reliability_manifest.csv", reliability_rows)
    write_csv_rows(root / "tables" / "component_manifest.csv", component_manifest_rows)
    write_csv_rows(root / "tables" / "component_coverage_audit.csv", component_coverage_rows)
    write_csv_rows(root / "tables" / "source_ablation_audit.csv", source_ablation_rows)
    write_csv_rows(root / "tables" / "paired_generation_audit.csv", paired_generation_rows)
    write_csv_rows(root / "tables" / "negative_control_summary.csv", [_negative_control_summary(decision)])
    write_csv_rows(root / "tables" / "oracle_gap_summary.csv", tail._oracle_gap_rows(matrix_rows, bottom20_keys))
    write_csv_rows(root / "tables" / "eligibility_audit.csv", eligibility_rows)
    write_csv_rows(root / "tables" / "source_summary_diagnostics.csv", source_summary_rows)
    write_csv_rows(root / "tables" / "weak_source_audit.csv", weak_rows)
    if cfg.skip_nearest_neighbor_audit:
        write_csv_rows(
            root / "tables" / "nearest_neighbor_memorization_audit.csv",
            [{"audit_skipped": True, "reason": "skip_nearest_neighbor_audit"}],
        )
    else:
        write_csv_rows(root / "tables" / "nearest_neighbor_memorization_audit.csv", nn_rows)
    write_csv_rows(root / "tables" / "mass_bag_member_matrix.csv", bag_member_rows)
    write_csv_rows(root / "manifests" / "harmful_source_suppression_model_manifest.csv", model_manifest_rows)
    write_protocol_finalization(
        root,
        leakage_report=leakage.to_json_dict(),
        protocol_manifest={
            "schema_version": "cvae_rebuild_source_inner_harmful_source_suppression_protocol_v1",
            "experiment_name": cfg.name,
            "primary_method": cfg.primary_method,
            "experiment_type": "source_only_harmful_source_suppression_random_mass_bag_component_union",
            "target_expert_excluded": bool(target_expert_excluded),
            "target_support_used": False,
            "target_support_labels_for_selection": False,
            "target_eval_labels_for_scoring_only": True,
            "heldout_target_rows_used_for_source_inner_scoring": False,
            "source_inner_uses_non_target_source_eval_rows": True,
            "source_inner_harmfulness_aggregation": "experiment_seed_x_heldout_center_over_pseudo_target_x_canonical_replicate_seed",
            "bottom20_definition": "lowest 20% eligible seed-center-replicate cells by unsuppressed random_mass_bag_control BACC",
            "center3_definition": 'heldout_center == "3"',
            "target_ablation_alignment_audit_only": True,
            "target_ablation_alignment_cannot_change_thresholds_weights_adoption_or_selection": True,
            "nearest_neighbor_memorization_audit_skipped": bool(cfg.skip_nearest_neighbor_audit),
            "nearest_neighbor_memorization_audit_skip_reason": "memory_safety" if cfg.skip_nearest_neighbor_audit else "",
            "hard_exclusion_diagnostic_only": True,
            "suppression_rate_low": cfg.suppression_rate_low,
            "suppression_rate_high": cfg.suppression_rate_high,
            "claim_boundary": (
                "source-inner leave-one-source diagnostics for robust source-only component composition; "
                "not target-conditioned routing, target adaptation, learned routing, or post-hoc source removal"
            ),
            "protocol_wording": PROTOCOL_WORDING,
            "protocol_violations": list(protocol_violations),
        },
        resolved_config=_resolved_config(cfg),
    )
    _write_decision_summary(root, decision)


def _negative_control_summary(decision: Mapping[str, object]) -> dict[str, object]:
    return {
        "primary_method": PRIMARY_HARMFUL_SUPPRESSION_METHOD,
        "primary_center_equal_mean_bacc": decision.get("center_equal_mean_bacc", math.nan),
        "delta_vs_random_mass_bag": decision.get("delta_vs_random_mass_bag", math.nan),
        "center3_delta_vs_random_mass_bag": decision.get("center3_delta_vs_random_mass_bag", math.nan),
        "bottom20_delta_vs_random_mass_bag": decision.get("bottom20_delta_vs_random_mass_bag", math.nan),
        "controls_beaten_on_tail": decision.get("controls_beaten_on_tail", False),
        "suppression_collapsed": decision.get("suppression_collapsed", False),
    }


def _write_decision_summary(root: Path, decision: Mapping[str, object]) -> None:
    lines = [
        "# Source-Only Harmful-Source Suppression over Random Mass-Bag Component Union v1",
        "",
        "## Primary Verdict",
        "",
        f"- Primary method: `{decision.get('primary_method', PRIMARY_HARMFUL_SUPPRESSION_METHOD)}`",
        f"- Primary verdict: `{decision.get('primary_verdict', 'HARMFUL_SOURCE_SUPPRESSION_FAIL')}`",
        f"- Diagnostic flags: `{decision.get('diagnostic_flags', '')}`",
        f"- Leakage status: `{decision.get('leakage_status', '')}`",
        f"- Center-equal mean BACC: {_format_float(decision.get('center_equal_mean_bacc'))}",
        f"- Min center BACC: {_format_float(decision.get('min_center_bacc'))}",
        f"- Center3 BACC: {_format_float(decision.get('center3_bacc'))}",
        f"- Bottom20 cell mean BACC: {_format_float(decision.get('bottom20_cell_mean_bacc'))}",
        f"- Worst seed-center BACC: {_format_float(decision.get('worst_seed_center_bacc'))}",
        f"- Seed std BACC: {_format_float(decision.get('seed_std_bacc'))}",
        f"- Delta vs random mass-bag: {_format_float(decision.get('delta_vs_random_mass_bag'))}",
        f"- Center3 delta vs random mass-bag: {_format_float(decision.get('center3_delta_vs_random_mass_bag'))}",
        f"- Bottom20 delta vs random mass-bag: {_format_float(decision.get('bottom20_delta_vs_random_mass_bag'))}",
        f"- Worst seed-center delta vs random mass-bag: {_format_float(decision.get('worst_seed_center_delta_vs_random_mass_bag'))}",
        f"- Suppression rate: {_format_float(decision.get('suppression_rate'))}",
        f"- Suppressed source oracle harmful rate: {_format_float(decision.get('suppressed_source_oracle_harmful_rate'))}",
        f"- Suppressed source oracle helpful rate: {_format_float(decision.get('suppressed_source_oracle_helpful_rate'))}",
        f"- Unsuppressed source oracle harmful rate: {_format_float(decision.get('unsuppressed_source_oracle_harmful_rate'))}",
        "",
        "## Protocol Boundary",
        "",
        PROTOCOL_WORDING,
        "",
        "Target-ablation alignment is evaluated after primary scoring. It cannot change suppression rules, thresholds, source weights, adoption labels, or primary method selection.",
        "",
        "Useful thesis success is robustness evidence, not full final-method adoption.",
        "",
        "Do not claim target-conditioned routing, target adaptation, learned routing, formal privacy, or post-hoc source removal.",
        "",
    ]
    (root / "reports" / "decision_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _resolved_config(cfg: HarmfulSourceSuppressionConfig) -> dict[str, object]:
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
            "dense_tailshield_artifact_root": "" if cfg.dense_tailshield_artifact_root is None else str(cfg.dense_tailshield_artifact_root),
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
        "harmful_source_suppression": {
            "primary_method": cfg.primary_method,
            "random_mass_bag_size": cfg.random_mass_bag_size,
            "random_mass_bag_alpha": cfg.random_mass_bag_alpha,
            "dirichlet_total_concentration": cfg.dirichlet_total_concentration,
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
            "min_harmfulness_observations": cfg.min_harmfulness_observations,
            "moderate_hit_rate_min": cfg.moderate_hit_rate_min,
            "moderate_gain_min": cfg.moderate_gain_min,
            "moderate_helpful_loss_max": cfg.moderate_helpful_loss_max,
            "severe_hit_rate_min": cfg.severe_hit_rate_min,
            "severe_gain_min": cfg.severe_gain_min,
            "severe_helpful_loss_max": cfg.severe_helpful_loss_max,
            "max_suppressed_sources": cfg.max_suppressed_sources,
            "suppression_rate_low": cfg.suppression_rate_low,
            "suppression_rate_high": cfg.suppression_rate_high,
            "oracle_harm_delta_threshold": cfg.oracle_harm_delta_threshold,
        },
        "classifier": {
            "type": cfg.classifier_type,
            "solver": cfg.classifier_solver,
            "C": cfg.classifier_c,
            "max_iter": cfg.classifier_max_iter,
            "class_weight": cfg.classifier_class_weight,
            "classifier_seed": cfg.classifier_seed,
        },
        "memory": {
            "skip_nearest_neighbor_audit": cfg.skip_nearest_neighbor_audit,
        },
    }
