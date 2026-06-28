from __future__ import annotations

import json
import math
import random
import resource
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from downstream import (
    PredictionBundle,
    evaluate_probability_predictions,
    predict_from_probabilities,
)
from config_sections import experiment_config_sections
from features import load_feature_cache, select_rows
from metrics import nanmean
from preservation import _hash_array
from preservation_repair import (
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
from preservation_sampling import DIAGNOSTIC_SELECTION, PRIMARY_SELECTION, RuntimeSource, _manifest_row, _per_source_variant, _runtime_source
from protocol import ProtocolError, assert_candidate_pool, assert_support_eval_disjoint, build_leakage_report
from reporting import prepare_artifact_dirs, write_csv_rows, write_protocol_finalization
from splits import candidate_experts
from support_policy_rows import (
    candidate_policy_coverage_rows,
    candidate_policy_matrix_rows,
    candidate_policy_paired_rows,
    policy_source_row_bundle,
)
from support_split_rows import scoped_unlabeled_support_split_rows

import component_union_mass_bagged as mb
import component_union_tailrisk_anchored_mass_bagged as tr
import decentralized_adaptive_gmm_prior as d1a
import decentralized_component_union_prior as cu
import decentralized_k16_gmm_prior as d1
import decentralized_reliability_weighted_gmm_prior as d12
import decentralized_support_nelbo_reliability_gmm_prior as snr
import dense_reliability_tailshield_random_mass_bag as dense
import paired_dense_all4_reliability_confirmation as paired
import support_calibrated_component_union_prior as support_cal


TARGET_SUPPORT_RISK_GATE_NAME = "virchow2_cvae_target_support32_regime_risk_gated_component_union_v1"
PRIMARY_RISK_GATED_METHOD = "target_support32_regime_risk_gated_random_bag_tail_safe_policy_v1"
RISK_GATE_SOURCE_WEIGHTING = "target_support32_regime_risk_policy_gate"

POLICY_RANDOM_BAG = "random_mass_bag_component_union"
POLICY_SHRINK050 = "reliability_shrink050_component_union"
POLICY_DENSE = "paired_dense_reliability_all4_weighted_geom"

ROW_ALWAYS_RANDOM_BAG = "always_random_mass_bag_component_union"
ROW_ALWAYS_SHRINK050 = "always_reliability_shrink050_component_union"
ROW_ALWAYS_DENSE = "always_paired_dense_reliability_all4_weighted_geom"
ROW_RANDOM_GATE_CONTROL = "target_support_regime_risk_random_gate_matched_selection_rate"
ROW_SHUFFLED_LABEL_GATE_CONTROL = "target_support_regime_risk_shuffled_source_inner_label_gate"
ROW_PERMUTED_FEATURE_GATE_CONTROL = "target_support_regime_risk_permuted_support_feature_gate"
ROW_ORACLE_BEST_POLICY = "target_support_regime_risk_oracle_best_policy_diagnostic"
ROW_SUPPORT8_DIAGNOSTIC = "target_support8_regime_risk_gate_diagnostic"
ROW_SUPPORT16_DIAGNOSTIC = "target_support16_regime_risk_gate_diagnostic"
ROW_THRESHOLD_SENSITIVITY_PREFIX = "target_support32_regime_risk_threshold_sensitivity"

COMPACT_FEATURES = (
    "random_bag_vs_shrink050_disagreement",
    "random_bag_vs_dense_reliability_disagreement",
    "shrink050_vs_dense_reliability_disagreement",
    "random_bag_vote_entropy",
    "random_bag_mean_entropy",
    "random_bag_top1_margin",
    "support_nelbo_ood_score",
    "support_nelbo_best_gap",
)

PROTOCOL_WORDING = (
    "This is a target-support regime-risk policy selection audit. It uses unlabeled, disjoint "
    "target support to choose among fixed composition policies and held-out target evaluation labels "
    "only for final scoring. It is not source-only routing, expert-level compatibility recovery, "
    "metadata routing, target adaptation, formal privacy, or causal reliability validation."
)


@dataclass(frozen=True)
class TargetSupportRiskGateConfig:
    name: str
    artifact_root: Path
    repair_artifact_root: Path
    source_union_gmm_artifact_root: Path | None
    balanced_gmm_artifact_root: Path | None
    feature_cache_root: Path
    backbone: str
    experiment_seeds: tuple[int, ...]
    heldout_centers: tuple[str, ...]
    support_seeds: tuple[int, ...]
    strict_full_run_matrix: bool
    support_size: int
    support_size_diagnostics: tuple[int, ...]
    nested_support_max_size: int
    synthetic_per_class_total: int
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
    reliability_floor_score: float
    reliability_epsilon: float
    support_nelbo_tau: float
    random_mass_bag_size: int
    random_mass_bag_alpha: float
    risk_low_threshold: float
    risk_high_threshold: float
    threshold_sensitivity_pairs: tuple[tuple[float, float], ...]
    min_gate_train_episodes: int
    tail_risk_bacc_threshold: float
    safer_policy_gain_threshold: float
    gate_c: float
    reconstruction_probability_tolerance: float
    classifier_type: str
    classifier_solver: str
    classifier_c: float
    classifier_max_iter: int
    classifier_class_weight: str
    classifier_seed: int | None
    skip_nearest_neighbor_audit: bool

    @property
    def replicate_seeds(self) -> tuple[int, ...]:
        return self.support_seeds

    @property
    def fresh_replicate_seeds(self) -> tuple[int, ...]:
        return tuple()

    @property
    def all_replicate_seeds(self) -> tuple[int, ...]:
        return self.support_seeds

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
class PolicyBundleSet:
    random_bag: tr.BagEvaluation
    shrink050: mb.MemberResult
    dense_reliability: dense.DenseBundleEvaluation
    real_feature_row: dict[str, object] | None


@dataclass(frozen=True)
class GateModel:
    status: str
    risk_probability: float
    selected_policy: str
    coefficients_json: str
    intercept: float
    train_episode_count: int
    train_positive_count: int
    train_negative_count: int
    fallback_reason: str
    control_kind: str = "primary"


def load_target_support_regime_risk_gate_config(path: str | Path) -> TargetSupportRiskGateConfig:
    source = Path(path).resolve()
    data = _load_mapping(source)
    base_dir = source.parents[2] if len(source.parents) >= 3 else source.parent
    return parse_target_support_regime_risk_gate_config(data, base_dir=base_dir)


def parse_target_support_regime_risk_gate_config(
    data: Mapping[str, Any],
    *,
    base_dir: str | Path = ".",
) -> TargetSupportRiskGateConfig:
    base = Path(base_dir)
    sections = experiment_config_sections(data)
    experiment = sections.experiment
    inputs = sections.inputs
    run = sections.run_matrix
    generation = sections.generation
    gate = _mapping(data, "target_support_regime_risk_gate")
    classifier = sections.classifier
    memory_raw = data.get("memory", {})
    if not isinstance(memory_raw, Mapping):
        raise ProtocolError("memory must be a mapping when provided.")
    cfg = TargetSupportRiskGateConfig(
        name=str(experiment["name"]),
        artifact_root=_path(base, str(experiment["artifact_root"])),
        repair_artifact_root=_path(base, str(inputs["repair_artifact_root"])),
        source_union_gmm_artifact_root=_optional_path(base, inputs.get("source_union_gmm_artifact_root")),
        balanced_gmm_artifact_root=_optional_path(base, inputs.get("balanced_gmm_artifact_root")),
        feature_cache_root=_path(base, str(inputs["feature_cache_root"])),
        backbone=str(inputs.get("backbone", "")),
        experiment_seeds=tuple(int(v) for v in run["experiment_seeds"]),
        heldout_centers=tuple(str(v) for v in run["heldout_centers"]),
        support_seeds=tuple(int(v) for v in run["support_seeds"]),
        strict_full_run_matrix=bool(run.get("strict_full_run_matrix", False)),
        support_size=int(run["support_size"]),
        support_size_diagnostics=tuple(int(v) for v in run["support_size_diagnostics"]),
        nested_support_max_size=int(run["nested_support_max_size"]),
        synthetic_per_class_total=int(generation["synthetic_per_class_total"]),
        min_per_source_per_class=int(generation["min_per_source_per_class"]),
        primary_variant=str(experiment["primary_variant"]),
        primary_method=str(gate["primary_method"]),
        candidate_components_per_source_class=tuple(int(v) for v in gate["candidate_components_per_source_class"]),
        min_samples_per_component=int(gate["min_samples_per_component"]),
        source_weighting=str(gate["source_weighting"]),
        gmm_covariance_type=str(gate["gmm_covariance_type"]),
        gmm_reg_covar=float(gate["gmm_reg_covar"]),
        gmm_n_init=int(gate["gmm_n_init"]),
        gmm_max_iter=int(gate["gmm_max_iter"]),
        min_component_weight=float(gate["min_component_weight"]),
        variance_floor=float(gate["variance_floor"]),
        variance_ceiling_multiplier=float(gate["variance_ceiling_multiplier"]),
        primary_pooling=str(gate["primary_pooling"]),
        reliability_floor_score=float(gate["reliability_floor_score"]),
        reliability_epsilon=float(gate["reliability_epsilon"]),
        support_nelbo_tau=float(gate["support_nelbo_tau"]),
        random_mass_bag_size=int(gate["random_mass_bag_size"]),
        random_mass_bag_alpha=float(gate["random_mass_bag_alpha"]),
        risk_low_threshold=float(gate["risk_low_threshold"]),
        risk_high_threshold=float(gate["risk_high_threshold"]),
        threshold_sensitivity_pairs=tuple((float(pair[0]), float(pair[1])) for pair in gate["threshold_sensitivity_pairs"]),
        min_gate_train_episodes=int(gate["min_gate_train_episodes"]),
        tail_risk_bacc_threshold=float(gate["tail_risk_bacc_threshold"]),
        safer_policy_gain_threshold=float(gate["safer_policy_gain_threshold"]),
        gate_c=float(gate["gate_c"]),
        reconstruction_probability_tolerance=float(gate.get("reconstruction_probability_tolerance", 1.0e-6)),
        classifier_type=str(classifier["type"]),
        classifier_solver=str(classifier["solver"]),
        classifier_c=float(classifier["C"]),
        classifier_max_iter=int(classifier["max_iter"]),
        classifier_class_weight=str(classifier["class_weight"]),
        classifier_seed=None if classifier.get("classifier_seed") is None else int(classifier["classifier_seed"]),
        skip_nearest_neighbor_audit=bool(memory_raw.get("skip_nearest_neighbor_audit", True)),
    )
    validate_target_support_regime_risk_gate_config(cfg)
    return cfg


def validate_target_support_regime_risk_gate_config(cfg: TargetSupportRiskGateConfig) -> None:
    if cfg.name != TARGET_SUPPORT_RISK_GATE_NAME:
        raise ProtocolError(f"Target-support risk gate name must be {TARGET_SUPPORT_RISK_GATE_NAME!r}.")
    if cfg.backbone != "virchow2":
        raise ProtocolError("Target-support risk gate is locked to backbone=virchow2.")
    if cfg.primary_variant != PRIMARY_VARIANT:
        raise ProtocolError(f"primary_variant must be {PRIMARY_VARIANT!r}.")
    if cfg.primary_method != PRIMARY_RISK_GATED_METHOD:
        raise ProtocolError(f"primary_method must be {PRIMARY_RISK_GATED_METHOD!r}.")
    if cfg.source_weighting != RISK_GATE_SOURCE_WEIGHTING:
        raise ProtocolError(f"source_weighting must be {RISK_GATE_SOURCE_WEIGHTING!r}.")
    if cfg.heldout_centers != ("0", "1", "2", "3", "4"):
        raise ProtocolError("heldout_centers must be locked to ['0', '1', '2', '3', '4'].")
    if cfg.support_size != 32:
        raise ProtocolError("Primary support_size must be locked to 32.")
    if cfg.support_size_diagnostics != (8, 16):
        raise ProtocolError("support_size_diagnostics must be locked to [8, 16].")
    if cfg.nested_support_max_size != 32:
        raise ProtocolError("nested_support_max_size must be locked to 32.")
    if cfg.candidate_components_per_source_class != (4, 3, 2, 1):
        raise ProtocolError("candidate_components_per_source_class must be locked to [4, 3, 2, 1].")
    if cfg.gmm_covariance_type != "diag":
        raise ProtocolError("gmm_covariance_type must be diag.")
    if cfg.primary_pooling != "policy_level_gate":
        raise ProtocolError("primary_pooling must be policy_level_gate.")
    if not math.isclose(cfg.random_mass_bag_alpha, 4.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ProtocolError("random_mass_bag_alpha must be Dirichlet-uniform alpha4.")
    if not math.isclose(cfg.risk_low_threshold, 0.60, rel_tol=0.0, abs_tol=1.0e-12):
        raise ProtocolError("risk_low_threshold must be locked to 0.60.")
    if not math.isclose(cfg.risk_high_threshold, 0.75, rel_tol=0.0, abs_tol=1.0e-12):
        raise ProtocolError("risk_high_threshold must be locked to 0.75.")
    if cfg.threshold_sensitivity_pairs != ((0.50, 0.70), (0.60, 0.75), (0.70, 0.85)):
        raise ProtocolError("threshold_sensitivity_pairs must be locked to [[0.50,0.70],[0.60,0.75],[0.70,0.85]].")
    if cfg.min_gate_train_episodes != 8:
        raise ProtocolError("min_gate_train_episodes must be locked to 8.")
    if not math.isclose(cfg.tail_risk_bacc_threshold, 0.80, rel_tol=0.0, abs_tol=1.0e-12):
        raise ProtocolError("tail_risk_bacc_threshold must be locked to 0.80.")
    if not math.isclose(cfg.safer_policy_gain_threshold, 0.025, rel_tol=0.0, abs_tol=1.0e-12):
        raise ProtocolError("safer_policy_gain_threshold must be locked to 0.025.")
    if not math.isclose(cfg.gate_c, 0.25, rel_tol=0.0, abs_tol=1.0e-12):
        raise ProtocolError("gate_c must be locked to 0.25.")
    if cfg.strict_full_run_matrix:
        if cfg.experiment_seeds != (42, 43, 44):
            raise ProtocolError("strict_full_run_matrix requires experiment_seeds=[42,43,44].")
        if cfg.support_seeds != (17, 23, 31):
            raise ProtocolError("strict_full_run_matrix requires support_seeds=[17,23,31].")
        if cfg.synthetic_per_class_total != 128:
            raise ProtocolError("strict_full_run_matrix requires synthetic_per_class_total=128.")
        if cfg.min_per_source_per_class != 8:
            raise ProtocolError("strict_full_run_matrix requires min_per_source_per_class=8.")
        if cfg.random_mass_bag_size != 11:
            raise ProtocolError("strict_full_run_matrix requires random_mass_bag_size=11.")
    if min(cfg.synthetic_per_class_total, cfg.min_per_source_per_class, cfg.min_samples_per_component, cfg.gmm_n_init, cfg.gmm_max_iter) < 1:
        raise ProtocolError("Budgets, component minimums, and GMM settings must be positive.")
    if min(
        cfg.gmm_reg_covar,
        cfg.min_component_weight,
        cfg.variance_floor,
        cfg.variance_ceiling_multiplier,
        cfg.reliability_floor_score,
        cfg.reliability_epsilon,
        cfg.support_nelbo_tau,
        cfg.random_mass_bag_alpha,
        cfg.reconstruction_probability_tolerance,
    ) <= 0.0:
        raise ProtocolError("Numeric floors/tolerances must be positive.")
    if cfg.classifier_type != "sklearn_logistic_regression":
        raise ProtocolError("classifier.type must be sklearn_logistic_regression.")
    if cfg.classifier_solver != "lbfgs" or cfg.classifier_c != 1.0 or cfg.classifier_max_iter != 2000:
        raise ProtocolError("Classifier solver/C/max_iter must remain locked.")
    if cfg.classifier_class_weight != "balanced" or cfg.classifier_seed is not None:
        raise ProtocolError("Classifier must use class_weight=balanced and classifier_seed=null.")
    if not cfg.skip_nearest_neighbor_audit:
        raise ProtocolError("Target-support risk gate v1 must skip nearest-neighbor audit for memory safety.")


def run_target_support_regime_risk_gated_component_union(
    cfg: TargetSupportRiskGateConfig,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    root = prepare_artifact_dirs(Path(artifact_root) if artifact_root is not None else cfg.artifact_root)
    for rel in ("checkpoints", "summaries", "dense_anchor_summaries", "cache/generated", "cache/predictions"):
        (root / rel).mkdir(parents=True, exist_ok=True)

    matrix_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    source_inner_rows: list[dict[str, object]] = []
    lopo_rows: list[dict[str, object]] = []
    feature_ablation_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    probability_manifest_rows: list[dict[str, object]] = []
    random_bag_manifest_rows: list[dict[str, object]] = []
    component_manifest_rows: list[dict[str, object]] = []
    component_coverage_rows: list[dict[str, object]] = []
    paired_generation_rows: list[dict[str, object]] = []
    eligibility_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
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
    for optional_root in (cfg.source_union_gmm_artifact_root, cfg.balanced_gmm_artifact_root):
        d1._validate_optional_leakage_report(optional_root, protocol_violations)

    repair_cfg = d1._repair_runtime_config(cfg, root)
    per_source_variant = _per_source_variant()

    try:
        for experiment_seed in cfg.experiment_seeds:
            print(f"[risk_gate] seed_start experiment_seed={experiment_seed}", flush=True)
            train_cache = load_feature_cache(_existing_cache_path(cfg.feature_cache_root, seed=experiment_seed, split="train"))
            test_cache = load_feature_cache(_existing_cache_path(cfg.feature_cache_root, seed=experiment_seed, split="test"))
            per_source_runtime: dict[str, RuntimeSource] = {}
            dense_summaries: dict[tuple[str, int], d1a.AdaptiveSourceLocalSummary] = {}
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
                model_rows.append(_manifest_row(experiment_seed, NA, runtime_source))
                support_calibration[str(source_center)] = snr._source_nelbo_calibration(runtime_source.runtime, str(source_center))

                dense_largest, _dense_bic = d1a._fit_and_export_source_summaries(
                    cfg,
                    root / "dense_anchor_summaries",
                    runtime_source.runtime,
                    experiment_seed=int(experiment_seed),
                    shuffled_label_control=False,
                )
                for summary in dense_largest:
                    dense_summaries[(summary.source_center, summary.class_label)] = summary

                summaries, detail_rows = cu._fit_and_export_pruned_gmm_summaries(
                    cfg,
                    root,
                    runtime_source.runtime,
                    experiment_seed=int(experiment_seed),
                    shuffled_label_control=False,
                )
                for summary in summaries:
                    gmm_summaries[(summary.source_center, summary.class_label)] = summary
                for row in detail_rows:
                    component_details[(str(row["source_center"]), int(row["class_label"]), int(row["source_component_id"]))] = row
                component_manifest_rows.extend(detail_rows)

            reliability: dict[tuple[int, int, str], d12.SourceReliability] = {}
            for support_seed in cfg.support_seeds:
                for source_center in cfg.heldout_centers:
                    reliability[(int(experiment_seed), int(support_seed), str(source_center))] = d12._source_local_reliability(
                        cfg,
                        per_source_runtime=per_source_runtime,
                        summaries=dense_summaries,
                        test_cache=test_cache,
                        experiment_seed=int(experiment_seed),
                        replicate_seed=int(support_seed),
                        source_center=str(source_center),
                    )

            for heldout_center in cfg.heldout_centers:
                print(f"[risk_gate] heldout_start seed={experiment_seed} heldout={heldout_center}", flush=True)
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

                source_inner = _source_inner_training_rows(
                    cfg,
                    root=root,
                    test_cache=test_cache,
                    per_source_runtime=per_source_runtime,
                    dense_summaries=dense_summaries,
                    gmm_summaries=gmm_summaries,
                    support_calibration=support_calibration,
                    reliability=reliability,
                    experiment_seed=int(experiment_seed),
                    real_heldout_center=str(heldout_center),
                    source_pool=candidates,
                )
                source_inner_rows.extend(source_inner)
                gate = _fit_gate(cfg, source_inner, control_kind="primary")
                shuffled_gate = _fit_gate(cfg, _shuffle_risk_labels(source_inner, experiment_seed, heldout_center), control_kind="shuffled_label")
                permuted_gate = _fit_gate(cfg, _permute_feature_values(source_inner, experiment_seed, heldout_center), control_kind="permuted_features")
                model_rows.extend(_gate_model_rows(cfg, experiment_seed, heldout_center, gate, shuffled_gate, permuted_gate))
                lopo_rows.extend(_lopo_audit_rows(cfg, source_inner, experiment_seed, heldout_center))
                feature_ablation_rows.extend(_feature_ablation_rows(cfg, source_inner, experiment_seed, heldout_center))

                for support_seed in cfg.support_seeds:
                    print(f"[risk_gate] cell_start seed={experiment_seed} heldout={heldout_center} support_seed={support_seed}", flush=True)
                    rels = {source: reliability[(int(experiment_seed), int(support_seed), str(source))] for source in candidates}
                    splits = support_cal.nested_unlabeled_support_eval_splits(
                        test_cache.metadata,
                        heldout_center=str(heldout_center),
                        support_seed=int(support_seed),
                        support_sizes=(*cfg.support_size_diagnostics, cfg.support_size),
                        max_support_size=cfg.nested_support_max_size,
                    )
                    split_rows.extend(_split_manifest_rows(splits, experiment_seed, support_seed, "target"))
                    split_by_key = {(split.support_size, split.eval_mode): split for split in splits}
                    primary_split = split_by_key[(cfg.support_size, "primary_style")]

                    eval_raw, eval_meta = select_rows(test_cache.embeddings, test_cache.metadata, primary_split.eval_indices)
                    eval_labels = tuple(_label(row) for row in eval_meta)
                    eval_sample_ids = tuple(_sample_id(row, idx) for idx, row in enumerate(eval_meta))
                    if len(set(eval_labels)) < 2:
                        row = _empty_policy_row(cfg, experiment_seed, heldout_center, support_seed, candidates, cfg.primary_method, "mono_class_target_eval_after_support32")
                        matrix_rows.append(row)
                        eligibility_rows.append(_eligibility_row(experiment_seed, heldout_center, support_seed, cfg.primary_method, "ineligible", "mono_class_target_eval_after_support32"))
                        continue

                    ref_row, _real_late = d1a._real_feature_reference(
                        cfg,
                        per_source_runtime=per_source_runtime,
                        candidates=candidates,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        replicate_seed=int(support_seed),
                        eval_raw=eval_raw,
                        eval_labels=eval_labels,
                    )
                    ref_row = mb._normalize_row(ref_row, prior_method=cu.ROW_REAL_FEATURE_DENSE_REFERENCE)
                    ref_row["support_seed"] = int(support_seed)
                    ref_row["support_size"] = cfg.support_size
                    ref_row["selection_source"] = DIAGNOSTIC_SELECTION
                    matrix_rows.append(ref_row)
                    real_feature_bacc = _float(ref_row["bacc"])
                    su_ref = d1._reference_for_cell(source_union_refs, experiment_seed, heldout_center, support_seed)
                    cb_ref = d1._reference_for_cell(center_balanced_refs, experiment_seed, heldout_center, support_seed)

                    policies = _candidate_policy_bundles(
                        cfg,
                        root=root,
                        per_source_runtime=per_source_runtime,
                        dense_summaries=dense_summaries,
                        gmm_summaries=gmm_summaries,
                        candidates=candidates,
                        rels=rels,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        support_seed=int(support_seed),
                        eval_raw=eval_raw,
                        eval_labels=eval_labels,
                        eval_sample_ids=eval_sample_ids,
                        source_union_ref=su_ref,
                        center_balanced_ref=cb_ref,
                        real_feature_bacc=real_feature_bacc,
                        include_real_feature=False,
                    )
                    for method, row in _candidate_matrix_rows(policies).items():
                        row = dict(row)
                        row["support_seed"] = int(support_seed)
                        row["support_size"] = cfg.support_size
                        row["selection_source"] = DIAGNOSTIC_SELECTION
                        matrix_rows.append(row)
                    component_coverage_rows.extend(_candidate_coverage_rows(policies, support_seed, cfg.support_size))
                    paired_generation_rows.extend(_candidate_paired_rows(policies, support_seed, cfg.support_size))
                    random_bag_manifest_rows.extend(_random_bag_manifest_rows(policies.random_bag, experiment_seed, heldout_center, support_seed, POLICY_RANDOM_BAG))
                    probability_manifest_rows.extend(_probability_manifest_rows(policies, experiment_seed, heldout_center, support_seed, "target_eval"))

                    component_manifest_rows.extend(
                        cu._fold_component_manifest_rows(
                            cfg,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            candidates=candidates,
                            summaries=gmm_summaries,
                            component_details=component_details,
                            weight_plan=policies.random_bag.ensemble_plan,
                        )
                    )

                    support_features_by_size: dict[int, dict[str, float]] = {}
                    for support_size in (*cfg.support_size_diagnostics, cfg.support_size):
                        split = split_by_key[(int(support_size), "primary_style")]
                        support_raw, _support_meta = select_rows(test_cache.embeddings, test_cache.metadata, split.support_indices)
                        support_ids = tuple(_sample_id(test_cache.metadata[idx], idx) for idx in split.support_indices)
                        support_policy = _candidate_policy_bundles(
                            cfg,
                            root=root,
                            per_source_runtime=per_source_runtime,
                            dense_summaries=dense_summaries,
                            gmm_summaries=gmm_summaries,
                            candidates=candidates,
                            rels=rels,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            support_seed=int(support_seed),
                            eval_raw=support_raw,
                            eval_labels=_dummy_binary_labels(len(split.support_indices)),
                            eval_sample_ids=support_ids,
                            source_union_ref=su_ref,
                            center_balanced_ref=cb_ref,
                            real_feature_bacc=math.nan,
                            include_real_feature=False,
                        )
                        scores = _support_scores(cfg, per_source_runtime, support_calibration, candidates, experiment_seed, heldout_center, support_seed, support_size, support_raw)
                        features = _regime_features(support_policy, scores)
                        support_features_by_size[int(support_size)] = features
                        feature_rows.append(
                            {
                                "row_scope": "target_support",
                                "experiment_seed": int(experiment_seed),
                                "heldout_center": str(heldout_center),
                                "support_seed": int(support_seed),
                                "support_size": int(support_size),
                                "support_eval_split_id": split.support_eval_split_id,
                                "support_labels_used": False,
                                "center_id_used_as_feature": False,
                                **features,
                            }
                        )

                    primary_features = support_features_by_size[cfg.support_size]
                    primary_decision = _predict_gate(cfg, gate, primary_features)
                    selected_row = _selected_policy_row(
                        cfg,
                        policies,
                        primary_decision.selected_policy,
                        experiment_seed,
                        heldout_center,
                        support_seed,
                        candidates,
                        su_ref,
                        cb_ref,
                        real_feature_bacc,
                        method=cfg.primary_method,
                        selection_source=PRIMARY_SELECTION,
                        claim_role="primary_target_support32_regime_risk_policy_gate",
                        risk_probability=primary_decision.risk_probability,
                        gate_status=primary_decision.status,
                    )
                    matrix_rows.append(selected_row)
                    selection_rows.append(_selection_row(experiment_seed, heldout_center, support_seed, cfg.support_size, primary_decision, selected_row))
                    eligibility_rows.append(_eligibility_row(experiment_seed, heldout_center, support_seed, cfg.primary_method, str(selected_row.get("status", "")), str(selected_row.get("error_message", ""))))

                    for diag_size, method in ((8, ROW_SUPPORT8_DIAGNOSTIC), (16, ROW_SUPPORT16_DIAGNOSTIC)):
                        diag_decision = _predict_gate(cfg, gate, support_features_by_size[diag_size])
                        diag_row = _selected_policy_row(
                            cfg,
                            policies,
                            diag_decision.selected_policy,
                            experiment_seed,
                            heldout_center,
                            support_seed,
                            candidates,
                            su_ref,
                            cb_ref,
                            real_feature_bacc,
                            method=method,
                            selection_source=DIAGNOSTIC_SELECTION,
                            claim_role=f"support{diag_size}_diagnostic_gate_not_primary_adoption",
                            risk_probability=diag_decision.risk_probability,
                            gate_status=diag_decision.status,
                        )
                        diag_row["support_size"] = diag_size
                        diag_row["diagnostic_only"] = True
                        matrix_rows.append(diag_row)

                    for control_gate, method in (
                        (shuffled_gate, ROW_SHUFFLED_LABEL_GATE_CONTROL),
                        (permuted_gate, ROW_PERMUTED_FEATURE_GATE_CONTROL),
                    ):
                        control_decision = _predict_gate(cfg, control_gate, primary_features)
                        matrix_rows.append(
                            _selected_policy_row(
                                cfg,
                                policies,
                                control_decision.selected_policy,
                                experiment_seed,
                                heldout_center,
                                support_seed,
                                candidates,
                                su_ref,
                                cb_ref,
                                real_feature_bacc,
                                method=method,
                                selection_source=DIAGNOSTIC_SELECTION,
                                claim_role="negative_control_gate",
                                risk_probability=control_decision.risk_probability,
                                gate_status=control_decision.status,
                            )
                        )

                    for low, high in cfg.threshold_sensitivity_pairs:
                        sens_decision = _predict_gate(cfg, gate, primary_features, thresholds=(low, high))
                        method = f"{ROW_THRESHOLD_SENSITIVITY_PREFIX}_low_{low:.2f}_high_{high:.2f}".replace(".", "p")
                        sens_row = _selected_policy_row(
                            cfg,
                            policies,
                            sens_decision.selected_policy,
                            experiment_seed,
                            heldout_center,
                            support_seed,
                            candidates,
                            su_ref,
                            cb_ref,
                            real_feature_bacc,
                            method=method,
                            selection_source=DIAGNOSTIC_SELECTION,
                            claim_role="threshold_sensitivity_diagnostic_only",
                            risk_probability=sens_decision.risk_probability,
                            gate_status=sens_decision.status,
                        )
                        sens_row.update({"threshold_low": low, "threshold_high": high, "diagnostic_only": True})
                        matrix_rows.append(sens_row)

                    oracle_policy = _oracle_best_policy(policies)
                    matrix_rows.append(
                        _selected_policy_row(
                            cfg,
                            policies,
                            oracle_policy,
                            experiment_seed,
                            heldout_center,
                            support_seed,
                            candidates,
                            su_ref,
                            cb_ref,
                            real_feature_bacc,
                            method=ROW_ORACLE_BEST_POLICY,
                            selection_source=DIAGNOSTIC_SELECTION,
                            claim_role="oracle_best_policy_diagnostic_only",
                            risk_probability=math.nan,
                            gate_status="oracle_diagnostic_only",
                        )
                    )
                    matrix_rows.append(cu._reference_matrix_row(cfg, experiment_seed=int(experiment_seed), heldout_center=str(heldout_center), replicate_seed=int(support_seed), candidates=candidates, prior_method=cu.ROW_SOURCE_UNION_K16_REFERENCE, reference=su_ref))
                    matrix_rows.append(cu._reference_matrix_row(cfg, experiment_seed=int(experiment_seed), heldout_center=str(heldout_center), replicate_seed=int(support_seed), candidates=candidates, prior_method=cu.ROW_CENTER_BALANCED_K16_REFERENCE, reference=cb_ref))
                    runtime_rows.append(_runtime_row(experiment_seed, heldout_center, support_seed, root))

    except ProtocolError as exc:
        protocol_violations.append(str(exc))

    matrix_rows.extend(_random_gate_control_rows(cfg, matrix_rows))
    bottom20_keys = dense._bottom20_raw_cell_keys(matrix_rows, ROW_ALWAYS_RANDOM_BAG)
    tail_rows = _tail_metric_summary_rows(matrix_rows, bottom20_keys)
    oracle_rows = _oracle_gap_rows(matrix_rows)
    negative_rows = _negative_control_rows(matrix_rows, bottom20_keys)
    target_oracle_rows = _target_oracle_audit_rows(matrix_rows)
    decision = _decision(matrix_rows, tail_rows, lopo_rows, target_oracle_rows, cfg, protocol_violations)
    leakage = build_leakage_report(
        target_support_labels_for_selection=False,
        target_eval_labels_for_scoring_only=True,
        target_expert_excluded=target_expert_excluded,
        oracle_rows_diagnostic_only=True,
        extra_violations=protocol_violations,
    )
    _write_artifacts(
        root,
        cfg,
        matrix_rows=matrix_rows,
        split_rows=split_rows,
        feature_rows=feature_rows,
        source_inner_rows=source_inner_rows,
        lopo_rows=lopo_rows,
        feature_ablation_rows=feature_ablation_rows,
        model_rows=model_rows,
        selection_rows=selection_rows,
        probability_manifest_rows=probability_manifest_rows,
        random_bag_manifest_rows=random_bag_manifest_rows,
        component_manifest_rows=component_manifest_rows,
        component_coverage_rows=component_coverage_rows,
        paired_generation_rows=paired_generation_rows,
        negative_rows=negative_rows,
        oracle_rows=oracle_rows,
        target_oracle_rows=target_oracle_rows,
        eligibility_rows=eligibility_rows,
        runtime_rows=runtime_rows,
        tail_rows=tail_rows,
        decision=decision,
        leakage=leakage,
        protocol_violations=protocol_violations,
        target_expert_excluded=target_expert_excluded,
    )
    return root


def _source_inner_training_rows(
    cfg: TargetSupportRiskGateConfig,
    *,
    root: Path,
    test_cache: object,
    per_source_runtime: Mapping[str, RuntimeSource],
    dense_summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    gmm_summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    support_calibration: Mapping[str, object],
    reliability: Mapping[tuple[int, int, str], d12.SourceReliability],
    experiment_seed: int,
    real_heldout_center: str,
    source_pool: Sequence[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pseudo_target in source_pool:
        pseudo_sources = tuple(source for source in source_pool if str(source) != str(pseudo_target))
        for support_seed in cfg.support_seeds:
            split = support_cal.nested_unlabeled_support_eval_splits(
                test_cache.metadata,
                heldout_center=str(pseudo_target),
                support_seed=int(support_seed),
                support_sizes=(cfg.support_size,),
                max_support_size=cfg.nested_support_max_size,
            )[0]
            support_raw, _support_meta = select_rows(test_cache.embeddings, test_cache.metadata, split.support_indices)
            eval_raw, eval_meta = select_rows(test_cache.embeddings, test_cache.metadata, split.eval_indices)
            eval_labels = tuple(_label(row) for row in eval_meta)
            if len(set(eval_labels)) < 2:
                rows.append(
                    {
                        "experiment_seed": experiment_seed,
                        "heldout_center": real_heldout_center,
                        "pseudo_target_center": str(pseudo_target),
                        "support_seed": int(support_seed),
                        "status": "ineligible",
                        "error_message": "mono_class_source_inner_eval",
                    }
                )
                continue
            support_ids = tuple(_sample_id(test_cache.metadata[idx], idx) for idx in split.support_indices)
            eval_ids = tuple(_sample_id(row, idx) for idx, row in enumerate(eval_meta))
            rels = {source: reliability[(experiment_seed, int(support_seed), str(source))] for source in pseudo_sources}
            real_feature_bacc = math.nan
            policies_eval = _candidate_policy_bundles(
                cfg,
                root=root,
                per_source_runtime=per_source_runtime,
                dense_summaries=dense_summaries,
                gmm_summaries=gmm_summaries,
                candidates=pseudo_sources,
                rels=rels,
                experiment_seed=experiment_seed,
                heldout_center=str(pseudo_target),
                support_seed=int(support_seed),
                eval_raw=eval_raw,
                eval_labels=eval_labels,
                eval_sample_ids=eval_ids,
                source_union_ref=d1._missing_reference(),
                center_balanced_ref=d1._missing_reference(),
                real_feature_bacc=real_feature_bacc,
                include_real_feature=False,
            )
            policies_support = _candidate_policy_bundles(
                cfg,
                root=root,
                per_source_runtime=per_source_runtime,
                dense_summaries=dense_summaries,
                gmm_summaries=gmm_summaries,
                candidates=pseudo_sources,
                rels=rels,
                experiment_seed=experiment_seed,
                heldout_center=str(pseudo_target),
                support_seed=int(support_seed),
                eval_raw=support_raw,
                eval_labels=_dummy_binary_labels(len(split.support_indices)),
                eval_sample_ids=support_ids,
                source_union_ref=d1._missing_reference(),
                center_balanced_ref=d1._missing_reference(),
                real_feature_bacc=math.nan,
                include_real_feature=False,
            )
            scores = _support_scores(cfg, per_source_runtime, support_calibration, pseudo_sources, experiment_seed, pseudo_target, support_seed, cfg.support_size, support_raw)
            features = _regime_features(policies_support, scores)
            random_bacc = _policy_bacc(policies_eval, POLICY_RANDOM_BAG)
            shrink_bacc = _policy_bacc(policies_eval, POLICY_SHRINK050)
            dense_bacc = _policy_bacc(policies_eval, POLICY_DENSE)
            best_safe = max(shrink_bacc, dense_bacc)
            risk = (random_bacc < cfg.tail_risk_bacc_threshold) or (best_safe - random_bacc >= cfg.safer_policy_gain_threshold)
            rows.append(
                {
                    "experiment_seed": experiment_seed,
                    "heldout_center": real_heldout_center,
                    "pseudo_target_center": str(pseudo_target),
                    "support_seed": int(support_seed),
                    "support_size": cfg.support_size,
                    "source_pool_json": json.dumps(list(pseudo_sources)),
                    "status": "ok",
                    "error_message": "",
                    "support_labels_used": False,
                    "target_eval_labels_used": False,
                    "random_bag_bacc": random_bacc,
                    "shrink050_bacc": shrink_bacc,
                    "dense_reliability_bacc": dense_bacc,
                    "best_safer_policy_bacc": best_safe,
                    "random_bag_tail_risk": int(bool(risk)),
                    "best_policy": _oracle_best_policy(policies_eval),
                    **features,
                }
            )
    return rows


def _candidate_policy_bundles(
    cfg: TargetSupportRiskGateConfig,
    *,
    root: Path,
    per_source_runtime: Mapping[str, RuntimeSource],
    dense_summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    gmm_summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    candidates: Sequence[str],
    rels: Mapping[str, d12.SourceReliability],
    experiment_seed: int,
    heldout_center: str,
    support_seed: int,
    eval_raw: object,
    eval_labels: Sequence[int],
    eval_sample_ids: Sequence[str],
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
    real_feature_bacc: float,
    include_real_feature: bool,
) -> PolicyBundleSet:
    transform = paired._heldout_excluded_reliability_transform(cfg, heldout_center, candidates, rels)
    plans = paired._variant_plans(
        cfg,
        candidates,
        transform,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=support_seed,
    )
    dense_eval = dense._evaluate_dense_anchor_bundle(
        cfg,
        per_source_runtime=per_source_runtime,
        summaries=dense_summaries,
        candidates=candidates,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=support_seed,
        eval_raw=eval_raw,
        eval_labels=eval_labels,
        eval_sample_ids=eval_sample_ids,
        source_union_ref=source_union_ref,
        center_balanced_ref=center_balanced_ref,
        real_feature_bacc=real_feature_bacc,
        method=ROW_ALWAYS_DENSE,
        plan=plans[paired.ROW_RELIABILITY_ALL4_WEIGHTED],
        pooling_rule="weighted_geometric",
        selection_source=DIAGNOSTIC_SELECTION,
        claim_role="always_dense_reliability_policy",
    )
    shrink_plan = cu._shrink_source_plan(cfg, candidates, rels, shrink_lambda=0.5, total=cfg.synthetic_per_class_total)
    shrink_eval = mb._evaluate_member(
        cfg,
        root=root,
        per_source_runtime=per_source_runtime,
        candidates=candidates,
        summaries=gmm_summaries,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=support_seed,
        eval_raw=eval_raw,
        eval_labels=eval_labels,
        source_union_ref=source_union_ref,
        center_balanced_ref=center_balanced_ref,
        real_feature_bacc=real_feature_bacc,
        weight_plan=shrink_plan,
        prior_method=ROW_ALWAYS_SHRINK050,
        selection_source=DIAGNOSTIC_SELECTION,
        claim_role="always_shrink050_policy",
        control_mode="normal",
    )
    bag_specs = mb._random_mass_bag_specs(cfg, candidates, rels, experiment_seed, heldout_center, support_seed)
    bag_eval = tr._evaluate_bag_with_bundle(
        cfg,
        root=root,
        per_source_runtime=per_source_runtime,
        candidates=candidates,
        summaries=gmm_summaries,
        specs=bag_specs,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=support_seed,
        eval_raw=eval_raw,
        eval_labels=eval_labels,
        source_union_ref=source_union_ref,
        center_balanced_ref=center_balanced_ref,
        real_feature_bacc=real_feature_bacc,
        method=ROW_ALWAYS_RANDOM_BAG,
        selection_source=DIAGNOSTIC_SELECTION,
        claim_role="always_random_mass_bag_policy",
    )
    real_feature_row = None
    if include_real_feature:
        real_feature_row, _ = d1a._real_feature_reference(
            cfg,
            per_source_runtime=per_source_runtime,
            candidates=candidates,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=support_seed,
            eval_raw=eval_raw,
            eval_labels=eval_labels,
        )
    return PolicyBundleSet(
        random_bag=bag_eval,
        shrink050=shrink_eval,
        dense_reliability=dense_eval,
        real_feature_row=real_feature_row,
    )


def _candidate_matrix_rows(policies: PolicyBundleSet) -> dict[str, dict[str, object]]:
    return candidate_policy_matrix_rows(
        policies,
        random_method=ROW_ALWAYS_RANDOM_BAG,
        shrink_method=ROW_ALWAYS_SHRINK050,
        dense_method=ROW_ALWAYS_DENSE,
    )


def _candidate_coverage_rows(policies: PolicyBundleSet, support_seed: int, support_size: int) -> list[dict[str, object]]:
    return candidate_policy_coverage_rows(policies, support_seed, support_size)


def _candidate_paired_rows(policies: PolicyBundleSet, support_seed: int, support_size: int) -> list[dict[str, object]]:
    return candidate_policy_paired_rows(policies, support_seed, support_size)


def _support_scores(
    cfg: TargetSupportRiskGateConfig,
    per_source_runtime: Mapping[str, RuntimeSource],
    support_calibration: Mapping[str, object],
    candidates: Sequence[str],
    experiment_seed: int,
    heldout_center: str,
    support_seed: int,
    support_size: int,
    support_raw: object,
):
    return support_cal._support_scores(
        cfg,
        per_source_runtime=per_source_runtime,
        support_calibration=support_calibration,
        candidates=candidates,
        experiment_seed=experiment_seed,
        heldout_center=str(heldout_center),
        support_seed=int(support_seed),
        support_size=int(support_size),
        support_raw=support_raw,
    )


def _regime_features(policies: PolicyBundleSet, support_scores: Sequence[object]) -> dict[str, float]:
    random_bundle = policies.random_bag.ensemble_bundle
    shrink_bundle = policies.shrink050.bundle
    dense_bundle = policies.dense_reliability.bundle
    if random_bundle is None or shrink_bundle is None or dense_bundle is None:
        return {feature: math.nan for feature in COMPACT_FEATURES}
    _assert_bundle_alignment((random_bundle, shrink_bundle, dense_bundle))
    random_preds = predict_from_probabilities(random_bundle.probabilities, classes=random_bundle.classes)
    shrink_preds = predict_from_probabilities(shrink_bundle.probabilities, classes=shrink_bundle.classes)
    dense_preds = predict_from_probabilities(dense_bundle.probabilities, classes=dense_bundle.classes)
    score_values = sorted(
        float(score.calibrated_support_nelbo)
        for score in support_scores
        if math.isfinite(float(score.calibrated_support_nelbo))
    )
    best_gap = (score_values[1] - score_values[0]) if len(score_values) >= 2 else math.nan
    return {
        "random_bag_vs_shrink050_disagreement": _disagreement(random_preds, shrink_preds),
        "random_bag_vs_dense_reliability_disagreement": _disagreement(random_preds, dense_preds),
        "shrink050_vs_dense_reliability_disagreement": _disagreement(shrink_preds, dense_preds),
        "random_bag_vote_entropy": _bag_vote_entropy(policies.random_bag),
        "random_bag_mean_entropy": _mean_entropy(random_bundle),
        "random_bag_top1_margin": _mean_top1_margin(random_bundle),
        "support_nelbo_ood_score": nanmean(score_values),
        "support_nelbo_best_gap": best_gap,
    }


def _fit_gate(cfg: TargetSupportRiskGateConfig, rows: Sequence[Mapping[str, object]], *, control_kind: str) -> GateModel:
    ok_rows = [row for row in rows if row.get("status") == "ok" and _row_features_finite(row)]
    y = np.asarray([int(row["random_bag_tail_risk"]) for row in ok_rows], dtype=int)
    if len(ok_rows) < cfg.min_gate_train_episodes:
        return _fallback_gate("too_few_source_inner_episodes", len(ok_rows), y, control_kind)
    if len(set(int(v) for v in y.tolist())) < 2:
        return _fallback_gate("single_risk_class", len(ok_rows), y, control_kind)
    try:
        from sklearn.linear_model import LogisticRegression  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Target-support risk gate requires scikit-learn.") from exc
    x = np.asarray([[float(row[feature]) for feature in COMPACT_FEATURES] for row in ok_rows], dtype=float)
    model = LogisticRegression(solver="liblinear", C=cfg.gate_c, class_weight="balanced", random_state=0)
    model.fit(x, y)
    return GateModel(
        status="trained",
        risk_probability=math.nan,
        selected_policy=POLICY_SHRINK050,
        coefficients_json=json.dumps({feature: float(value) for feature, value in zip(COMPACT_FEATURES, model.coef_[0].tolist())}, sort_keys=True),
        intercept=float(model.intercept_[0]),
        train_episode_count=len(ok_rows),
        train_positive_count=int(y.sum()),
        train_negative_count=int(len(y) - y.sum()),
        fallback_reason="",
        control_kind=control_kind,
    )


def _predict_gate(
    cfg: TargetSupportRiskGateConfig,
    gate: GateModel,
    features: Mapping[str, float],
    *,
    thresholds: tuple[float, float] | None = None,
) -> GateModel:
    low, high = thresholds or (cfg.risk_low_threshold, cfg.risk_high_threshold)
    if gate.status != "trained" or not _feature_values_finite(features):
        return GateModel(
            status="fallback",
            risk_probability=math.nan,
            selected_policy=POLICY_SHRINK050,
            coefficients_json=gate.coefficients_json,
            intercept=gate.intercept,
            train_episode_count=gate.train_episode_count,
            train_positive_count=gate.train_positive_count,
            train_negative_count=gate.train_negative_count,
            fallback_reason=gate.fallback_reason or "nonfinite_target_support_features",
            control_kind=gate.control_kind,
        )
    coefs = json.loads(gate.coefficients_json)
    logit = gate.intercept + sum(float(coefs[feature]) * float(features[feature]) for feature in COMPACT_FEATURES)
    risk_prob = 1.0 / (1.0 + math.exp(-logit))
    if risk_prob < low:
        policy = POLICY_RANDOM_BAG
    elif risk_prob < high:
        policy = POLICY_SHRINK050
    else:
        policy = POLICY_DENSE
    return GateModel(
        status="trained",
        risk_probability=risk_prob,
        selected_policy=policy,
        coefficients_json=gate.coefficients_json,
        intercept=gate.intercept,
        train_episode_count=gate.train_episode_count,
        train_positive_count=gate.train_positive_count,
        train_negative_count=gate.train_negative_count,
        fallback_reason="",
        control_kind=gate.control_kind,
    )


def _fallback_gate(reason: str, n_rows: int, y: np.ndarray, control_kind: str) -> GateModel:
    return GateModel(
        status="fallback",
        risk_probability=math.nan,
        selected_policy=POLICY_SHRINK050,
        coefficients_json="{}",
        intercept=math.nan,
        train_episode_count=int(n_rows),
        train_positive_count=int(y.sum()) if y.size else 0,
        train_negative_count=int(y.size - y.sum()) if y.size else 0,
        fallback_reason=reason,
        control_kind=control_kind,
    )


def _selected_policy_row(
    cfg: TargetSupportRiskGateConfig,
    policies: PolicyBundleSet,
    policy: str,
    experiment_seed: int,
    heldout_center: str,
    support_seed: int,
    candidates: Sequence[str],
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
    real_feature_bacc: float,
    *,
    method: str,
    selection_source: str,
    claim_role: str,
    risk_probability: float,
    gate_status: str,
) -> dict[str, object]:
    source = _policy_source_row_bundle(policies, policy)
    row = dict(source[0])
    bundle = source[1]
    if row.get("status") != "ok" or bundle is None:
        out = cu._empty_matrix_row(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=support_seed,
            candidates=candidates,
            prior_method=method,
            source_union_ref=source_union_ref,
            center_balanced_ref=center_balanced_ref,
            real_feature_bacc=real_feature_bacc,
            status="ineligible",
            error_message=f"selected_policy_ineligible:{policy}",
            claim_role=claim_role,
        )
    else:
        out = dict(row)
        out.update(
            {
                "prior_method": method,
                "prediction_hash": _hash_array(np.asarray(bundle.probabilities, dtype=float)),
                "selection_source": selection_source,
                "claim_role": claim_role,
                "status": "ok",
                "error_message": "",
            }
        )
    out.update(
        {
            "support_seed": int(support_seed),
            "support_size": cfg.support_size,
            "selected_policy": policy,
            "gate_status": gate_status,
            "risk_probability": risk_probability,
            "target_support_labels_used": False,
            "target_eval_labels_used_for_scoring_only": True,
            "oracle_policy_used_for_selection": False,
            "center_id_used_as_feature": False,
        }
    )
    return out


def _policy_source_row_bundle(policies: PolicyBundleSet, policy: str) -> tuple[dict[str, object], PredictionBundle | None]:
    return policy_source_row_bundle(
        policies,
        policy,
        random_policy=POLICY_RANDOM_BAG,
        shrink_policy=POLICY_SHRINK050,
        dense_policy=POLICY_DENSE,
    )


def _policy_bacc(policies: PolicyBundleSet, policy: str) -> float:
    row, _bundle = _policy_source_row_bundle(policies, policy)
    return _float(row.get("bacc"))


def _oracle_best_policy(policies: PolicyBundleSet) -> str:
    values = {
        POLICY_RANDOM_BAG: _policy_bacc(policies, POLICY_RANDOM_BAG),
        POLICY_SHRINK050: _policy_bacc(policies, POLICY_SHRINK050),
        POLICY_DENSE: _policy_bacc(policies, POLICY_DENSE),
    }
    return max(values, key=lambda key: (values[key] if math.isfinite(values[key]) else -math.inf, key))


def _assert_bundle_alignment(bundles: Sequence[PredictionBundle]) -> None:
    if not bundles:
        raise ProtocolError("No bundles to align.")
    classes = bundles[0].classes
    n_rows = len(bundles[0].probabilities)
    for bundle in bundles:
        if bundle.classes != classes:
            raise ProtocolError("Candidate policy class order mismatch.")
        if len(bundle.probabilities) != n_rows:
            raise ProtocolError("Candidate policy row count mismatch.")


def _dummy_binary_labels(n_rows: int) -> tuple[int, ...]:
    if n_rows < 2:
        raise ProtocolError("Need at least two support rows for dummy binary labels.")
    return tuple(idx % 2 for idx in range(n_rows))


def _disagreement(a: Sequence[int], b: Sequence[int]) -> float:
    if len(a) != len(b) or not a:
        return math.nan
    return sum(1 for x, y in zip(a, b) if int(x) != int(y)) / float(len(a))


def _mean_entropy(bundle: PredictionBundle) -> float:
    values = []
    for row in bundle.probabilities:
        entropy = -sum(float(p) * math.log(max(float(p), 1.0e-12)) for p in row)
        values.append(entropy / math.log(len(row)))
    return nanmean(values)


def _mean_top1_margin(bundle: PredictionBundle) -> float:
    margins = []
    for row in bundle.probabilities:
        vals = sorted((float(v) for v in row), reverse=True)
        margins.append(vals[0] - vals[1] if len(vals) >= 2 else math.nan)
    return nanmean(margins)


def _bag_vote_entropy(bag: tr.BagEvaluation) -> float:
    bundles = [result.bundle for result in bag.member_results if result.bundle is not None]
    if not bundles:
        return math.nan
    _assert_bundle_alignment(bundles)
    member_preds = [predict_from_probabilities(bundle.probabilities, classes=bundle.classes) for bundle in bundles]
    entropies = []
    for row_idx in range(len(member_preds[0])):
        counts: dict[int, int] = {}
        for preds in member_preds:
            counts[int(preds[row_idx])] = counts.get(int(preds[row_idx]), 0) + 1
        probs = [count / float(len(member_preds)) for count in counts.values()]
        entropies.append(-sum(p * math.log(max(p, 1.0e-12)) for p in probs) / math.log(2.0))
    return nanmean(entropies)


def _row_features_finite(row: Mapping[str, object]) -> bool:
    return all(math.isfinite(_float(row.get(feature))) for feature in COMPACT_FEATURES)


def _feature_values_finite(features: Mapping[str, float]) -> bool:
    return all(math.isfinite(float(features.get(feature, math.nan))) for feature in COMPACT_FEATURES)


def _shuffle_risk_labels(rows: Sequence[Mapping[str, object]], experiment_seed: int, heldout_center: str) -> list[dict[str, object]]:
    out = [dict(row) for row in rows]
    labels = [row.get("random_bag_tail_risk") for row in out if row.get("status") == "ok"]
    rng = random.Random(d1._latent_seed(experiment_seed, heldout_center, 0, "risk_gate_shuffle_labels"))
    rng.shuffle(labels)
    idx = 0
    for row in out:
        if row.get("status") == "ok":
            row["random_bag_tail_risk"] = labels[idx]
            idx += 1
    return out


def _permute_feature_values(rows: Sequence[Mapping[str, object]], experiment_seed: int, heldout_center: str) -> list[dict[str, object]]:
    out = [dict(row) for row in rows]
    rng = random.Random(d1._latent_seed(experiment_seed, heldout_center, 0, "risk_gate_permute_features"))
    for feature in COMPACT_FEATURES:
        values = [row.get(feature) for row in out if row.get("status") == "ok"]
        rng.shuffle(values)
        idx = 0
        for row in out:
            if row.get("status") == "ok":
                row[feature] = values[idx]
                idx += 1
    return out


def _lopo_audit_rows(
    cfg: TargetSupportRiskGateConfig,
    rows: Sequence[Mapping[str, object]],
    experiment_seed: int,
    heldout_center: str,
) -> list[dict[str, object]]:
    out = []
    for control_name, source_rows in (
        ("all_features", list(rows)),
        ("shuffled_labels", _shuffle_risk_labels(rows, experiment_seed, heldout_center)),
        ("permuted_features", _permute_feature_values(rows, experiment_seed, heldout_center)),
    ):
        preds = []
        labels = []
        selected_minus_random = []
        selected_minus_best = []
        for pseudo in sorted({str(row.get("pseudo_target_center")) for row in source_rows if row.get("status") == "ok"}):
            train = [row for row in source_rows if row.get("status") == "ok" and str(row.get("pseudo_target_center")) != pseudo]
            test = [row for row in source_rows if row.get("status") == "ok" and str(row.get("pseudo_target_center")) == pseudo]
            gate = _fit_gate(cfg, train, control_kind=f"lopo_{control_name}")
            for row in test:
                decision = _predict_gate(cfg, gate, {feature: _float(row.get(feature)) for feature in COMPACT_FEATURES})
                labels.append(int(row.get("random_bag_tail_risk", 0)))
                preds.append(0.5 if not math.isfinite(decision.risk_probability) else decision.risk_probability)
                selected_bacc = _policy_bacc_from_row(row, decision.selected_policy)
                random_bacc = _float(row.get("random_bag_bacc"))
                best_bacc = max(_float(row.get("random_bag_bacc")), _float(row.get("shrink050_bacc")), _float(row.get("dense_reliability_bacc")))
                if math.isfinite(selected_bacc) and math.isfinite(random_bacc):
                    selected_minus_random.append(selected_bacc - random_bacc)
                if math.isfinite(selected_bacc) and math.isfinite(best_bacc):
                    selected_minus_best.append(selected_bacc - best_bacc)
        auc = _safe_auc(labels, preds)
        recall = _risk_recall(labels, preds, threshold=cfg.risk_low_threshold)
        out.append(
            {
                "experiment_seed": int(experiment_seed),
                "heldout_center": str(heldout_center),
                "feature_group": control_name,
                "n_train_episodes": sum(1 for row in source_rows if row.get("status") == "ok"),
                "n_positive_risk": sum(1 for row in source_rows if row.get("status") == "ok" and int(row.get("random_bag_tail_risk", 0)) == 1),
                "n_negative_risk": sum(1 for row in source_rows if row.get("status") == "ok" and int(row.get("random_bag_tail_risk", 0)) == 0),
                "leave_one_pseudo_center_out_risk_auc": auc,
                "leave_one_pseudo_center_out_risk_recall": recall,
                "leave_one_pseudo_center_out_selected_minus_random_bag_bacc": nanmean(selected_minus_random),
                "leave_one_pseudo_center_out_selected_minus_best_available_bacc": nanmean(selected_minus_best),
                "lopo_adoption_gate": control_name == "all_features",
            }
        )
    return out


def _feature_ablation_rows(
    cfg: TargetSupportRiskGateConfig,
    rows: Sequence[Mapping[str, object]],
    experiment_seed: int,
    heldout_center: str,
) -> list[dict[str, object]]:
    groups = {
        "hard_disagreement_only": (
            "random_bag_vs_shrink050_disagreement",
            "random_bag_vs_dense_reliability_disagreement",
            "shrink050_vs_dense_reliability_disagreement",
        ),
        "probability_uncertainty_only": (
            "random_bag_vote_entropy",
            "random_bag_mean_entropy",
            "random_bag_top1_margin",
        ),
        "support_nelbo_only": (
            "support_nelbo_ood_score",
            "support_nelbo_best_gap",
        ),
        "all_features": COMPACT_FEATURES,
    }
    out = []
    for group, features in groups.items():
        projected = []
        for row in rows:
            updated = dict(row)
            if row.get("status") == "ok":
                for feature in COMPACT_FEATURES:
                    if feature not in features:
                        updated[feature] = 0.0
            projected.append(updated)
        audit = _lopo_audit_rows(cfg, projected, experiment_seed, heldout_center)[0]
        audit["feature_group"] = group
        out.append(audit)
    return out


def _policy_bacc_from_row(row: Mapping[str, object], policy: str) -> float:
    if policy == POLICY_RANDOM_BAG:
        return _float(row.get("random_bag_bacc"))
    if policy == POLICY_SHRINK050:
        return _float(row.get("shrink050_bacc"))
    if policy == POLICY_DENSE:
        return _float(row.get("dense_reliability_bacc"))
    return math.nan


def _safe_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    if len(set(labels)) < 2 or len(labels) != len(scores):
        return math.nan
    try:
        from sklearn.metrics import roc_auc_score  # type: ignore
    except ModuleNotFoundError:
        return math.nan
    return float(roc_auc_score(labels, scores))


def _risk_recall(labels: Sequence[int], scores: Sequence[float], *, threshold: float) -> float:
    positives = [idx for idx, label in enumerate(labels) if int(label) == 1]
    if not positives:
        return math.nan
    hits = sum(1 for idx in positives if float(scores[idx]) >= threshold)
    return hits / float(len(positives))


def _gate_model_rows(
    cfg: TargetSupportRiskGateConfig,
    experiment_seed: int,
    heldout_center: str,
    *gates: GateModel,
) -> list[dict[str, object]]:
    return [
        {
            "experiment_seed": int(experiment_seed),
            "heldout_center": str(heldout_center),
            "control_kind": gate.control_kind,
            "gate_status": gate.status,
            "fallback_reason": gate.fallback_reason,
            "train_episode_count": gate.train_episode_count,
            "train_positive_count": gate.train_positive_count,
            "train_negative_count": gate.train_negative_count,
            "gate_c": cfg.gate_c,
            "compact_feature_set_json": json.dumps(list(COMPACT_FEATURES)),
            "coefficients_json": gate.coefficients_json,
            "intercept": gate.intercept,
            "center_id_used_as_feature": False,
        }
        for gate in gates
    ]


def _selection_row(
    experiment_seed: int,
    heldout_center: str,
    support_seed: int,
    support_size: int,
    decision: GateModel,
    selected_row: Mapping[str, object],
) -> dict[str, object]:
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "support_seed": int(support_seed),
        "support_size": int(support_size),
        "risk_probability": decision.risk_probability,
        "selected_policy": decision.selected_policy,
        "gate_status": decision.status,
        "fallback_reason": decision.fallback_reason,
        "selected_target_bacc": selected_row.get("bacc", math.nan),
        "target_eval_labels_used_for_scoring_only": True,
        "target_support_labels_used": False,
    }


def _probability_manifest_rows(
    policies: PolicyBundleSet,
    experiment_seed: int,
    heldout_center: str,
    support_seed: int,
    eval_scope: str,
) -> list[dict[str, object]]:
    rows = []
    for policy, row, bundle in (
        (POLICY_RANDOM_BAG, policies.random_bag.ensemble_row, policies.random_bag.ensemble_bundle),
        (POLICY_SHRINK050, policies.shrink050.row, policies.shrink050.bundle),
        (POLICY_DENSE, policies.dense_reliability.row, policies.dense_reliability.bundle),
    ):
        rows.append(
            {
                "experiment_seed": int(experiment_seed),
                "heldout_center": str(heldout_center),
                "support_seed": int(support_seed),
                "eval_scope": eval_scope,
                "candidate_policy": policy,
                "status": row.get("status", ""),
                "prediction_hash": row.get("prediction_hash", ""),
                "class_order": "" if bundle is None else json.dumps(list(bundle.classes)),
                "n_probability_rows": 0 if bundle is None else len(bundle.probabilities),
            }
        )
    return rows


def _random_bag_manifest_rows(
    bag: tr.BagEvaluation,
    experiment_seed: int,
    heldout_center: str,
    support_seed: int,
    candidate_policy: str,
) -> list[dict[str, object]]:
    rows = []
    for result in bag.member_results:
        row = result.row
        weights = json.loads(str(row.get("source_weight_json", "{}") or "{}"))
        source_weight_mode = str(row.get("source_weighting", ""))
        control_permutation_id = row.get("control_permutation_id", "")
        try:
            permutation_id = int(control_permutation_id)
        except (TypeError, ValueError):
            permutation_id = None
        bag_seed = ""
        if source_weight_mode.startswith("random_mass_bag_uniform_alpha4_perm") and permutation_id is not None:
            bag_seed = d1._latent_seed(experiment_seed, heldout_center, support_seed, "random_mass_bag_uniform_alpha4", permutation_id)
        plan = mb._plan_from_row(row)
        mass_prior_hash = "" if plan is None else cu._plan_hash(plan)
        for source, mass in weights.items():
            rows.append(
                {
                    "experiment_seed": int(experiment_seed),
                    "heldout_center": str(heldout_center),
                    "support_seed": int(support_seed),
                    "candidate_policy": candidate_policy,
                    "bag_member_id": row.get("bag_member_id", ""),
                    "bag_seed": bag_seed,
                    "latent_sample_seed": row.get("latent_sample_seed", ""),
                    "source_weight_mode": source_weight_mode,
                    "control_permutation_id": control_permutation_id,
                    "source_id": source,
                    "source_mass": mass,
                    "mass_prior_hash": mass_prior_hash,
                    "generated_pool_hash": result.generated_hash,
                    "probability_bundle_hash": row.get("prediction_hash", ""),
                }
            )
    return rows


def _split_manifest_rows(splits: Sequence[object], experiment_seed: int, support_seed: int, scope: str) -> list[dict[str, object]]:
    return scoped_unlabeled_support_split_rows(splits, experiment_seed, support_seed, scope)


def _empty_policy_row(
    cfg: TargetSupportRiskGateConfig,
    experiment_seed: int,
    heldout_center: str,
    support_seed: int,
    candidates: Sequence[str],
    method: str,
    error: str,
) -> dict[str, object]:
    row = cu._empty_matrix_row(
        cfg,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=support_seed,
        candidates=candidates,
        prior_method=method,
        source_union_ref=d1._missing_reference(),
        center_balanced_ref=d1._missing_reference(),
        real_feature_bacc=math.nan,
        status="ineligible",
        error_message=error,
        claim_role="primary_target_support32_regime_risk_policy_gate",
    )
    row["support_seed"] = int(support_seed)
    return row


def _eligibility_row(
    experiment_seed: int,
    heldout_center: str,
    support_seed: int,
    scope: str,
    status: str,
    error: str,
) -> dict[str, object]:
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "support_seed": int(support_seed),
        "scope": str(scope),
        "status": str(status),
        "error_message": str(error),
    }


def _random_gate_control_rows(cfg: TargetSupportRiskGateConfig, rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    primary = cu._rows_for(rows, cfg.primary_method)
    if not primary:
        return []
    counts: dict[str, int] = {}
    for row in primary:
        counts[str(row.get("selected_policy", ""))] = counts.get(str(row.get("selected_policy", "")), 0) + 1
    policies = [POLICY_RANDOM_BAG, POLICY_SHRINK050, POLICY_DENSE]
    weights = [counts.get(policy, 0) / float(len(primary)) for policy in policies]
    if sum(weights) <= 0:
        weights = [1.0 / 3.0] * 3
    out = []
    for row in primary:
        seed = d1._latent_seed(row.get("experiment_seed", 0), row.get("heldout_center", ""), row.get("support_seed", 0), "random_gate_matched")
        rng = np.random.default_rng(seed)
        selected = str(rng.choice(policies, p=np.asarray(weights) / sum(weights)))
        source_method = {POLICY_RANDOM_BAG: ROW_ALWAYS_RANDOM_BAG, POLICY_SHRINK050: ROW_ALWAYS_SHRINK050, POLICY_DENSE: ROW_ALWAYS_DENSE}[selected]
        matches = [
            cand for cand in rows
            if cand.get("prior_method") == source_method
            and str(cand.get("experiment_seed")) == str(row.get("experiment_seed"))
            and str(cand.get("heldout_center")) == str(row.get("heldout_center"))
            and str(cand.get("replicate_seed")) == str(row.get("replicate_seed"))
        ]
        if not matches:
            continue
        control = dict(matches[0])
        control.update(
            {
                "prior_method": ROW_RANDOM_GATE_CONTROL,
                "selection_source": DIAGNOSTIC_SELECTION,
                "claim_role": "negative_control_random_gate_matched_selection_rate",
                "selected_policy": selected,
                "diagnostic_only": True,
                "oracle_policy_used_for_selection": False,
            }
        )
        out.append(control)
    return out


def _tail_metric_summary_rows(rows: Sequence[Mapping[str, object]], bottom20_keys: set[tuple[str, str, str]]) -> list[dict[str, object]]:
    out = []
    methods = sorted({str(row.get("prior_method")) for row in rows if row.get("prior_method")})
    random_metrics = dense._tail_metrics(rows, ROW_ALWAYS_RANDOM_BAG, bottom20_keys=bottom20_keys)
    for method in methods:
        metrics = dense._tail_metrics(rows, method, bottom20_keys=bottom20_keys)
        if int(metrics.get("n_raw_rows", 0)) < 1:
            continue
        out.append(
            {
                "prior_method": method,
                **metrics,
                "center3_delta_vs_random_mass_bag": _delta(metrics.get("center3_bacc"), random_metrics.get("center3_bacc")),
                "bottom20_delta_vs_random_mass_bag": _delta(metrics.get("bottom20_cell_mean_bacc"), random_metrics.get("bottom20_cell_mean_bacc")),
                "worst_seed_center_delta_vs_random_mass_bag": _delta(metrics.get("worst_seed_center_bacc"), random_metrics.get("worst_seed_center_bacc")),
                "bottom20_definition": "lowest_20pct_eligible_raw_cells_by_always_random_mass_bag_bacc",
                "center3_definition": 'heldout_center == "3"',
            }
        )
    return out


def _oracle_gap_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    primary = {(_key(row)): row for row in cu._rows_for(rows, PRIMARY_RISK_GATED_METHOD)}
    oracle = {(_key(row)): row for row in cu._rows_for(rows, ROW_ORACLE_BEST_POLICY)}
    out = []
    for key, row in primary.items():
        oracle_row = oracle.get(key)
        out.append(
            {
                "experiment_seed": key[0],
                "heldout_center": key[1],
                "support_seed": key[2],
                "primary_bacc": row.get("bacc", math.nan),
                "oracle_best_policy_bacc": "" if oracle_row is None else oracle_row.get("bacc", math.nan),
                "oracle_gap": "" if oracle_row is None else _delta(oracle_row.get("bacc"), row.get("bacc")),
                "oracle_policy_diagnostic_only": True,
            }
        )
    return out


def _negative_control_rows(rows: Sequence[Mapping[str, object]], bottom20_keys: set[tuple[str, str, str]]) -> list[dict[str, object]]:
    primary = dense._tail_metrics(rows, PRIMARY_RISK_GATED_METHOD, bottom20_keys=bottom20_keys)
    out = []
    for method in (ROW_RANDOM_GATE_CONTROL, ROW_SHUFFLED_LABEL_GATE_CONTROL, ROW_PERMUTED_FEATURE_GATE_CONTROL):
        metrics = dense._tail_metrics(rows, method, bottom20_keys=bottom20_keys)
        out.append(
            {
                "primary_method": PRIMARY_RISK_GATED_METHOD,
                "control_method": method,
                "primary_center_equal_mean_bacc": primary.get("center_equal_mean_bacc"),
                "control_center_equal_mean_bacc": metrics.get("center_equal_mean_bacc"),
                "primary_minus_control_bacc": _delta(primary.get("center_equal_mean_bacc"), metrics.get("center_equal_mean_bacc")),
                "primary_bottom20_minus_control": _delta(primary.get("bottom20_cell_mean_bacc"), metrics.get("bottom20_cell_mean_bacc")),
                "diagnostic_only": True,
            }
        )
    return out


def _target_oracle_audit_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    out = []
    for primary in cu._rows_for(rows, PRIMARY_RISK_GATED_METHOD):
        key = _key(primary)
        random_rows = [row for row in rows if _key(row) == key and row.get("prior_method") == ROW_ALWAYS_RANDOM_BAG]
        oracle_rows = [row for row in rows if _key(row) == key and row.get("prior_method") == ROW_ORACLE_BEST_POLICY]
        if not random_rows or not oracle_rows:
            continue
        random_bacc = _float(random_rows[0].get("bacc"))
        oracle_bacc = _float(oracle_rows[0].get("bacc"))
        target_risk = random_bacc < 0.80 or oracle_bacc - random_bacc >= 0.025
        selected_conservative = primary.get("selected_policy") in (POLICY_SHRINK050, POLICY_DENSE)
        out.append(
            {
                "experiment_seed": key[0],
                "heldout_center": key[1],
                "support_seed": key[2],
                "random_mass_bag_bacc": random_bacc,
                "oracle_best_policy_bacc": oracle_bacc,
                "target_oracle_random_bag_tail_risk": int(target_risk),
                "selected_policy": primary.get("selected_policy", ""),
                "selected_conservative_policy": int(selected_conservative),
                "risk_detected_correctly": int(bool(target_risk) == bool(selected_conservative)),
                "diagnostic_only": True,
            }
        )
    return out


def _decision(
    rows: Sequence[Mapping[str, object]],
    tail_rows: Sequence[Mapping[str, object]],
    lopo_rows: Sequence[Mapping[str, object]],
    target_oracle_rows: Sequence[Mapping[str, object]],
    cfg: TargetSupportRiskGateConfig,
    protocol_violations: Sequence[str],
) -> dict[str, object]:
    by_method = {row["prior_method"]: row for row in tail_rows}
    primary = by_method.get(cfg.primary_method, {})
    random_metrics = by_method.get(ROW_ALWAYS_RANDOM_BAG, {})
    control_methods = (ROW_RANDOM_GATE_CONTROL, ROW_SHUFFLED_LABEL_GATE_CONTROL, ROW_PERMUTED_FEATURE_GATE_CONTROL)
    control_metrics = [by_method[method] for method in control_methods if method in by_method]
    lopo_primary = [row for row in lopo_rows if row.get("feature_group") == "all_features"]
    lopo_controls = [row for row in lopo_rows if row.get("feature_group") in {"shuffled_labels", "permuted_features"}]
    lopo_auc = nanmean([_float(row.get("leave_one_pseudo_center_out_risk_auc")) for row in lopo_primary])
    control_auc = nanmean([_float(row.get("leave_one_pseudo_center_out_risk_auc")) for row in lopo_controls])
    selection_rows = cu._rows_for(rows, cfg.primary_method)
    rates = _selection_rates(selection_rows)
    target_recall = _target_oracle_risk_recall(target_oracle_rows)
    represented_centers = {str(row.get("heldout_center")) for row in selection_rows if str(row.get("heldout_center", ""))}
    centers_ok = (not cfg.strict_full_run_matrix) or len(represented_centers) == len(cfg.heldout_centers)
    lopo_ok = math.isfinite(lopo_auc) and (not math.isfinite(control_auc) or lopo_auc > control_auc)
    gate_not_collapsed = max(
        rates["selected_random_mass_bag_rate"],
        rates["selected_shrink050_rate"],
        rates["selected_dense_reliability_rate"],
    ) < 0.90
    seed_std_ok = (
        _float(primary.get("seed_std_bacc")) <= _float(random_metrics.get("seed_std_bacc"))
        and _float(primary.get("seed_std_bacc")) <= 0.045
    )
    control_bottom20_max = _max_metric(control_metrics, "bottom20_cell_mean_bacc")
    control_worst_max = _max_metric(control_metrics, "worst_seed_center_bacc")
    control_center3_max = _max_metric(control_metrics, "center3_bacc")
    control_tail_clear = (
        _float(primary.get("bottom20_cell_mean_bacc")) > control_bottom20_max
        and _float(primary.get("worst_seed_center_bacc")) > control_worst_max
        and _float(primary.get("center3_bacc")) > control_center3_max
    )
    control_tail_not_matched = (
        _float(primary.get("bottom20_cell_mean_bacc")) > control_bottom20_max
        or _float(primary.get("worst_seed_center_bacc")) > control_worst_max
        or _float(primary.get("center3_bacc")) > control_center3_max
    )
    flags = []
    if protocol_violations:
        flags.append("PROTOCOL_VIOLATION")
    if not centers_ok:
        flags.append("MISSING_CENTER")
    if not lopo_ok:
        flags.append("LOPO_GATE_NOT_PREDICTIVE")
    if not gate_not_collapsed:
        flags.append("GATE_COLLAPSED_TO_ONE_POLICY")
    if not control_tail_not_matched:
        flags.append("TARGET_SUPPORT_CONTROLS_MATCH_PRIMARY")
    if math.isfinite(target_recall) and target_recall < 0.60:
        flags.append("TARGET_ORACLE_RISK_RECALL_BELOW_0P60")
    if not seed_std_ok:
        flags.append("SEED_STD_GATE_FAILED")
    if _float(primary.get("min_center_bacc")) < 0.80:
        flags.append("MIN_CENTER_BELOW_0P80")
    if _delta(primary.get("center_equal_mean_bacc"), random_metrics.get("center_equal_mean_bacc")) < -0.010:
        flags.append("MEAN_DROP_GT_0P010")
    useful = (
        not protocol_violations
        and centers_ok
        and _delta(primary.get("center_equal_mean_bacc"), random_metrics.get("center_equal_mean_bacc")) >= -0.007
        and _float(primary.get("min_center_bacc")) >= 0.80
        and lopo_ok
        and gate_not_collapsed
        and control_tail_not_matched
        and (
            _delta(primary.get("bottom20_cell_mean_bacc"), random_metrics.get("bottom20_cell_mean_bacc")) > 0.0
            or _delta(primary.get("worst_seed_center_bacc"), random_metrics.get("worst_seed_center_bacc")) > 0.0
        )
    )
    strong = (
        useful
        and _delta(primary.get("center_equal_mean_bacc"), random_metrics.get("center_equal_mean_bacc")) >= -0.005
        and _float(primary.get("min_center_bacc")) >= 0.82
        and _delta(primary.get("center3_bacc"), random_metrics.get("center3_bacc")) >= 0.020
        and _delta(primary.get("bottom20_cell_mean_bacc"), random_metrics.get("bottom20_cell_mean_bacc")) >= 0.030
        and _delta(primary.get("worst_seed_center_bacc"), random_metrics.get("worst_seed_center_bacc")) >= 0.100
        and seed_std_ok
        and control_tail_clear
        and math.isfinite(target_recall)
        and target_recall >= 0.60
    )
    verdict = "STRONG_SUCCESS" if strong else ("USEFUL_THESIS_SUCCESS" if useful else "DIAGNOSTIC_ONLY")
    return {
        "primary_method": cfg.primary_method,
        "primary_verdict": verdict,
        "diagnostic_flags": "|".join(flags) if flags else "NONE",
        **{key: primary.get(key, math.nan) for key in (
            "center_equal_mean_bacc",
            "seed_cell_mean_bacc",
            "center_equal_macro_f1",
            "min_center_bacc",
            "seed_std_bacc",
            "bottom20_cell_mean_bacc",
            "worst_seed_center_bacc",
            "center3_bacc",
        )},
        "random_mass_bag_center_equal_mean_bacc": random_metrics.get("center_equal_mean_bacc", math.nan),
        "delta_vs_random_mass_bag": _delta(primary.get("center_equal_mean_bacc"), random_metrics.get("center_equal_mean_bacc")),
        "bottom20_delta_vs_random_mass_bag": _delta(primary.get("bottom20_cell_mean_bacc"), random_metrics.get("bottom20_cell_mean_bacc")),
        "worst_seed_center_delta_vs_random_mass_bag": _delta(primary.get("worst_seed_center_bacc"), random_metrics.get("worst_seed_center_bacc")),
        "lopo_gate_auc": lopo_auc,
        "lopo_control_auc": control_auc,
        "lopo_gate_verdict": "PASS" if math.isfinite(lopo_auc) and (not math.isfinite(control_auc) or lopo_auc > control_auc) else "FAIL",
        "target_oracle_risk_recall": target_recall,
        "n_centers_represented": len(represented_centers),
        "control_tail_clear": bool(control_tail_clear),
        "control_tail_not_matched": bool(control_tail_not_matched),
        "max_control_bottom20_cell_mean_bacc": control_bottom20_max,
        "max_control_worst_seed_center_bacc": control_worst_max,
        "max_control_center3_bacc": control_center3_max,
        **rates,
    }


def _max_metric(rows: Sequence[Mapping[str, object]], key: str) -> float:
    values = [_float(row.get(key)) for row in rows]
    finite = [value for value in values if math.isfinite(value)]
    return max(finite) if finite else -math.inf


def _target_oracle_risk_recall(rows: Sequence[Mapping[str, object]]) -> float:
    risky = [row for row in rows if _float(row.get("target_oracle_random_bag_tail_risk")) == 1.0]
    if not risky:
        return math.nan
    detected = sum(1 for row in risky if _float(row.get("selected_conservative_policy")) == 1.0)
    return detected / float(len(risky))


def _selection_rates(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    total = len(rows)
    if not total:
        return {
            "selected_random_mass_bag_rate": math.nan,
            "selected_shrink050_rate": math.nan,
            "selected_dense_reliability_rate": math.nan,
            "fallback_rate": math.nan,
            "untrained_gate_rate": math.nan,
        }
    return {
        "selected_random_mass_bag_rate": sum(1 for row in rows if row.get("selected_policy") == POLICY_RANDOM_BAG) / total,
        "selected_shrink050_rate": sum(1 for row in rows if row.get("selected_policy") == POLICY_SHRINK050) / total,
        "selected_dense_reliability_rate": sum(1 for row in rows if row.get("selected_policy") == POLICY_DENSE) / total,
        "fallback_rate": sum(1 for row in rows if row.get("gate_status") == "fallback") / total,
        "untrained_gate_rate": sum(1 for row in rows if row.get("gate_status") != "trained") / total,
    }


def _runtime_row(experiment_seed: int, heldout_center: str, support_seed: int, root: Path) -> dict[str, object]:
    maxrss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "support_seed": int(support_seed),
        "max_rss_kb": int(maxrss_kb),
        "max_rss_gb": float(maxrss_kb) / 1024.0 / 1024.0,
        "target_max_rss_gb": 1.5,
        "artifact_root": str(root),
    }


from target_support_risk_gate_artifacts import (
    _protocol_manifest,
    _resolved_config,
    _threshold_sensitivity_rows,
    _write_artifacts,
    _write_decision_summary,
)


def _key(row: Mapping[str, object]) -> tuple[str, str, str]:
    return (str(row.get("experiment_seed")), str(row.get("heldout_center")), str(row.get("replicate_seed", row.get("support_seed"))))


def _delta(a: object, b: object) -> float:
    left = _float(a)
    right = _float(b)
    return left - right if math.isfinite(left) and math.isfinite(right) else math.nan


def _sample_id(row: Mapping[str, object], idx: int) -> str:
    value = row.get("sample_id", "")
    return str(value) if str(value) else f"row_{idx}"


def _optional_path(base: Path, value: object) -> Path | None:
    if value is None or str(value) == "":
        return None
    return _path(base, str(value))
