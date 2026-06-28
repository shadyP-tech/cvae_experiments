from __future__ import annotations

import json
import math
import random
import resource
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluation.downstream import PredictionBundle, evaluate_probability_predictions, predict_from_probabilities
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
)
from experiments.preservation.preservation_sampling import DIAGNOSTIC_SELECTION, PRIMARY_SELECTION, RuntimeSource, _manifest_row, _per_source_variant, _runtime_source
from core.protocol import ProtocolError, assert_candidate_pool, assert_support_eval_disjoint
from core.reporting import prepare_artifact_dirs, write_csv_rows, write_json
from data.splits import candidate_experts

from experiments.component_union import component_union_mass_bagged as mb
from experiments.decentralized import decentralized_adaptive_gmm_prior as d1a
from experiments.decentralized import decentralized_component_union_prior as cu
from experiments.decentralized import decentralized_k16_gmm_prior as d1
from experiments.decentralized import decentralized_reliability_weighted_gmm_prior as d12
from experiments.support_selection import support_calibrated_component_union_prior as support_cal
from experiments.support_selection import target_support_regime_risk_gated_component_union as risk


LABELED_SUPPORT_POLICY_CALIBRATION_NAME = "virchow2_cvae_labeled_support16_random_vs_dense_policy_calibration_v1"
PRIMARY_LABELED_SUPPORT_POLICY_METHOD = "labeled_support16_random_default_dense_switch_v1"
LABELED_SUPPORT_SOURCE_WEIGHTING = "labeled_support16_random_default_dense_switch"

POLICY_RANDOM_BAG = risk.POLICY_RANDOM_BAG
POLICY_DENSE = risk.POLICY_DENSE
POLICY_SHRINK050 = risk.POLICY_SHRINK050

ROW_ALWAYS_RANDOM_BAG = risk.ROW_ALWAYS_RANDOM_BAG
ROW_ALWAYS_DENSE = risk.ROW_ALWAYS_DENSE
ROW_ALWAYS_SHRINK050 = risk.ROW_ALWAYS_SHRINK050
ROW_RANDOM_SINGLE_MASS = mb.ROW_RANDOM_SINGLE_MASS_CONTROL
ROW_ORACLE_BEST_POLICY = "labeled_support16_oracle_best_random_vs_dense_policy_diagnostic"
ROW_SHUFFLED_SUPPORT_LABEL_CONTROL = "labeled_support16_shuffled_support_label_dense_switch_control"
ROW_OFF_TARGET_SUPPORT_CONTROL = "labeled_support16_off_target_labeled_support_dense_switch_control"
ROW_RANDOM_SWITCH_MATCHED_RATE = "labeled_support16_random_dense_switch_matched_rate_control"
ROW_RANDOM_DEFAULT_CONTROL = "labeled_support16_random_default_conservative_switch_control"
ROW_SUPPORT_SIZE_DIAGNOSTIC_PREFIX = "labeled_support_policy_size"
ROW_COMMON_EVAL_DIAGNOSTIC_PREFIX = "labeled_support_policy_common_eval_size"

PROTOCOL_WORDING = (
    "This is a Tier 2 few-shot target-local utility calibration audit. Class-balanced labeled "
    "target support is allowed only to score already-fixed candidate policies. It does not train "
    "classifiers, modify generation, tune thresholds, change candidate policies, choose support "
    "size, or use target-evaluation labels for selection."
)


@dataclass(frozen=True)
class LabeledNestedSupportEvalSplit:
    heldout_center: str
    support_size: int
    support_seed: int
    eval_mode: str
    support_indices: tuple[int, ...]
    eval_indices: tuple[int, ...]
    support_sample_ids: tuple[str, ...]
    eval_sample_ids: tuple[str, ...]
    support_labels: tuple[int, ...]
    support_eval_split_id: str
    parent_support32_split_id: str
    support_labels_used: bool = True
    class_balanced_support: bool = True


@dataclass(frozen=True)
class LabeledSupportPolicyCalibrationConfig:
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
    primary_labeled_support_size: int
    diagnostic_labeled_support_sizes: tuple[int, ...]
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
    random_mass_bag_size: int
    random_mass_bag_alpha: float
    primary_switch_quantum: float
    support_quantum_by_size: Mapping[int, float]
    classifier_type: str
    classifier_solver: str
    classifier_c: float
    classifier_max_iter: int
    classifier_class_weight: str
    classifier_seed: int | None
    skip_nearest_neighbor_audit: bool

    @property
    def support_size(self) -> int:
        return self.primary_labeled_support_size

    @property
    def support_size_diagnostics(self) -> tuple[int, ...]:
        return self.diagnostic_labeled_support_sizes

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

    @property
    def reconstruction_probability_tolerance(self) -> float:
        return 1.0e-6


def load_labeled_support_policy_calibration_config(path: str | Path) -> LabeledSupportPolicyCalibrationConfig:
    source = Path(path).resolve()
    data = _load_mapping(source)
    base_dir = source.parents[2] if len(source.parents) >= 3 else source.parent
    return parse_labeled_support_policy_calibration_config(data, base_dir=base_dir)


def parse_labeled_support_policy_calibration_config(
    data: Mapping[str, Any],
    *,
    base_dir: str | Path = ".",
) -> LabeledSupportPolicyCalibrationConfig:
    base = Path(base_dir)
    experiment = _mapping(data, "experiment")
    inputs = _mapping(data, "inputs")
    run = _mapping(data, "run_matrix")
    generation = _mapping(data, "generation")
    labeled = _mapping(data, "labeled_support_policy_calibration")
    classifier = _mapping(data, "classifier")
    memory_raw = data.get("memory", {})
    if not isinstance(memory_raw, Mapping):
        raise ProtocolError("memory must be a mapping when provided.")
    quantum_raw = labeled.get("support_quantum_by_size", {8: 0.125, 16: 0.0625, 32: 0.03125})
    if not isinstance(quantum_raw, Mapping):
        raise ProtocolError("support_quantum_by_size must be a mapping.")
    cfg = LabeledSupportPolicyCalibrationConfig(
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
        primary_labeled_support_size=int(run["primary_labeled_support_size"]),
        diagnostic_labeled_support_sizes=tuple(int(v) for v in run["diagnostic_labeled_support_sizes"]),
        nested_support_max_size=int(run["nested_support_max_size"]),
        synthetic_per_class_total=int(generation["synthetic_per_class_total"]),
        min_per_source_per_class=int(generation["min_per_source_per_class"]),
        primary_variant=str(experiment["primary_variant"]),
        primary_method=str(labeled["primary_method"]),
        candidate_components_per_source_class=tuple(int(v) for v in labeled["candidate_components_per_source_class"]),
        min_samples_per_component=int(labeled["min_samples_per_component"]),
        source_weighting=str(labeled["source_weighting"]),
        gmm_covariance_type=str(labeled["gmm_covariance_type"]),
        gmm_reg_covar=float(labeled["gmm_reg_covar"]),
        gmm_n_init=int(labeled["gmm_n_init"]),
        gmm_max_iter=int(labeled["gmm_max_iter"]),
        min_component_weight=float(labeled["min_component_weight"]),
        variance_floor=float(labeled["variance_floor"]),
        variance_ceiling_multiplier=float(labeled["variance_ceiling_multiplier"]),
        primary_pooling=str(labeled["primary_pooling"]),
        reliability_floor_score=float(labeled["reliability_floor_score"]),
        reliability_epsilon=float(labeled["reliability_epsilon"]),
        random_mass_bag_size=int(labeled["random_mass_bag_size"]),
        random_mass_bag_alpha=float(labeled["random_mass_bag_alpha"]),
        primary_switch_quantum=float(labeled["primary_switch_quantum"]),
        support_quantum_by_size={int(k): float(v) for k, v in quantum_raw.items()},
        classifier_type=str(classifier["type"]),
        classifier_solver=str(classifier["solver"]),
        classifier_c=float(classifier["C"]),
        classifier_max_iter=int(classifier["max_iter"]),
        classifier_class_weight=str(classifier["class_weight"]),
        classifier_seed=None if classifier.get("classifier_seed") is None else int(classifier["classifier_seed"]),
        skip_nearest_neighbor_audit=bool(memory_raw.get("skip_nearest_neighbor_audit", True)),
    )
    validate_labeled_support_policy_calibration_config(cfg)
    return cfg


def validate_labeled_support_policy_calibration_config(cfg: LabeledSupportPolicyCalibrationConfig) -> None:
    if cfg.name != LABELED_SUPPORT_POLICY_CALIBRATION_NAME:
        raise ProtocolError(f"Labeled support policy calibration name must be {LABELED_SUPPORT_POLICY_CALIBRATION_NAME!r}.")
    if cfg.backbone != "virchow2":
        raise ProtocolError("Labeled support policy calibration is locked to backbone=virchow2.")
    if cfg.primary_variant != PRIMARY_VARIANT:
        raise ProtocolError(f"primary_variant must be {PRIMARY_VARIANT!r}.")
    if cfg.primary_method != PRIMARY_LABELED_SUPPORT_POLICY_METHOD:
        raise ProtocolError(f"primary_method must be {PRIMARY_LABELED_SUPPORT_POLICY_METHOD!r}.")
    if cfg.source_weighting != LABELED_SUPPORT_SOURCE_WEIGHTING:
        raise ProtocolError(f"source_weighting must be {LABELED_SUPPORT_SOURCE_WEIGHTING!r}.")
    if cfg.heldout_centers != ("0", "1", "2", "3", "4"):
        raise ProtocolError("heldout_centers must be locked to ['0', '1', '2', '3', '4'].")
    if cfg.primary_labeled_support_size != 16:
        raise ProtocolError("primary_labeled_support_size must be locked to 16.")
    if cfg.diagnostic_labeled_support_sizes != (8, 32):
        raise ProtocolError("diagnostic_labeled_support_sizes must be locked to [8, 32].")
    if cfg.nested_support_max_size != 32:
        raise ProtocolError("nested_support_max_size must be locked to 32.")
    if cfg.candidate_components_per_source_class != (4, 3, 2, 1):
        raise ProtocolError("candidate_components_per_source_class must be locked to [4, 3, 2, 1].")
    if cfg.gmm_covariance_type != "diag":
        raise ProtocolError("gmm_covariance_type must be diag.")
    if cfg.primary_pooling != "labeled_support_random_default_dense_switch":
        raise ProtocolError("primary_pooling must be labeled_support_random_default_dense_switch.")
    if not math.isclose(cfg.random_mass_bag_alpha, 4.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ProtocolError("random_mass_bag_alpha must be Dirichlet-uniform alpha4.")
    expected_quanta = {8: 0.125, 16: 0.0625, 32: 0.03125}
    if dict(cfg.support_quantum_by_size) != expected_quanta:
        raise ProtocolError("support_quantum_by_size must be locked to support8=0.125, support16=0.0625, support32=0.03125.")
    if not math.isclose(cfg.primary_switch_quantum, 0.0625, rel_tol=0.0, abs_tol=1.0e-12):
        raise ProtocolError("primary_switch_quantum must be locked to 0.0625.")
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
        cfg.random_mass_bag_alpha,
    ) <= 0.0:
        raise ProtocolError("Numeric floors/tolerances must be positive.")
    if cfg.classifier_type != "sklearn_logistic_regression":
        raise ProtocolError("classifier.type must be sklearn_logistic_regression.")
    if cfg.classifier_solver != "lbfgs" or cfg.classifier_c != 1.0 or cfg.classifier_max_iter != 2000:
        raise ProtocolError("Classifier solver/C/max_iter must remain locked.")
    if cfg.classifier_class_weight != "balanced" or cfg.classifier_seed is not None:
        raise ProtocolError("Classifier must use class_weight=balanced and classifier_seed=null.")
    if not cfg.skip_nearest_neighbor_audit:
        raise ProtocolError("Labeled support policy calibration v1 must skip nearest-neighbor audit for memory safety.")


def run_labeled_support_policy_calibration(
    cfg: LabeledSupportPolicyCalibrationConfig,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    root = prepare_artifact_dirs(Path(artifact_root) if artifact_root is not None else cfg.artifact_root)
    for rel in ("checkpoints", "summaries", "dense_anchor_summaries", "cache/generated", "cache/predictions"):
        (root / rel).mkdir(parents=True, exist_ok=True)

    matrix_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    policy_score_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    switch_event_rows: list[dict[str, object]] = []
    probability_manifest_rows: list[dict[str, object]] = []
    random_bag_manifest_rows: list[dict[str, object]] = []
    utility_alignment_rows: list[dict[str, object]] = []
    quantization_rows: list[dict[str, object]] = []
    common_eval_rows: list[dict[str, object]] = []
    target_oracle_rows: list[dict[str, object]] = []
    eligibility_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    component_manifest_rows: list[dict[str, object]] = []
    component_coverage_rows: list[dict[str, object]] = []
    paired_generation_rows: list[dict[str, object]] = []
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
            print(f"[labeled_support] seed_start experiment_seed={experiment_seed}", flush=True)
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
                print(f"[labeled_support] heldout_start seed={experiment_seed} heldout={heldout_center}", flush=True)
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

                for support_seed in cfg.support_seeds:
                    print(f"[labeled_support] cell_start seed={experiment_seed} heldout={heldout_center} support_seed={support_seed}", flush=True)
                    rels = {source: reliability[(int(experiment_seed), int(support_seed), str(source))] for source in candidates}
                    split_sizes = (8, 16, 32)
                    try:
                        splits = nested_labeled_support_eval_splits(
                            test_cache.metadata,
                            heldout_center=str(heldout_center),
                            support_seed=int(support_seed),
                            support_sizes=split_sizes,
                            max_support_size=cfg.nested_support_max_size,
                        )
                    except ProtocolError as exc:
                        message = f"labeled_support_split_ineligible:{exc}"
                        matrix_rows.append(_empty_policy_row(cfg, experiment_seed, heldout_center, support_seed, candidates, cfg.primary_method, message))
                        for requested_size in split_sizes:
                            eligibility_rows.append(
                                _eligibility_row(
                                    experiment_seed,
                                    heldout_center,
                                    support_seed,
                                    f"labeled_support{requested_size}_split",
                                    "ineligible",
                                    message,
                                )
                            )
                        print(
                            f"[labeled_support] cell_ineligible seed={experiment_seed} heldout={heldout_center} support_seed={support_seed} reason={message}",
                            flush=True,
                        )
                        continue
                    split_rows.extend(_split_manifest_rows(splits, experiment_seed, support_seed, "target"))
                    split_by_key = {(split.support_size, split.eval_mode): split for split in splits}
                    for requested_size in split_sizes:
                        if (requested_size, "primary_style") not in split_by_key:
                            eligibility_rows.append(
                                _eligibility_row(
                                    experiment_seed,
                                    heldout_center,
                                    support_seed,
                                    f"labeled_support{requested_size}_split",
                                    "ineligible",
                                    f"insufficient_class_balanced_target_samples_for_support{requested_size}_with_disjoint_eval",
                                )
                            )
                    primary_split = split_by_key.get((cfg.primary_labeled_support_size, "primary_style"))
                    if primary_split is None:
                        matrix_rows.append(_empty_policy_row(cfg, experiment_seed, heldout_center, support_seed, candidates, cfg.primary_method, "primary_support16_split_ineligible"))
                        eligibility_rows.append(_eligibility_row(experiment_seed, heldout_center, support_seed, cfg.primary_method, "ineligible", "primary_support16_split_ineligible"))
                        continue
                    su_ref = d1._reference_for_cell(source_union_refs, experiment_seed, heldout_center, support_seed)
                    cb_ref = d1._reference_for_cell(center_balanced_refs, experiment_seed, heldout_center, support_seed)

                    target_policy_by_size: dict[int, risk.PolicyBundleSet] = {}
                    support_policy_by_size: dict[int, risk.PolicyBundleSet] = {}
                    support_scores_by_size: dict[int, dict[str, float]] = {}

                    for support_size in split_sizes:
                        eval_split = split_by_key.get((support_size, "primary_style"))
                        if eval_split is None:
                            continue
                        eval_raw, eval_meta = select_rows(test_cache.embeddings, test_cache.metadata, eval_split.eval_indices)
                        eval_labels = tuple(_label(row) for row in eval_meta)
                        eval_ids = tuple(_sample_id(row, idx) for idx, row in zip(eval_split.eval_indices, eval_meta))
                        if len(set(eval_labels)) < 2:
                            eligibility_rows.append(_eligibility_row(experiment_seed, heldout_center, support_seed, f"target_eval_support{support_size}", "ineligible", "mono_class_target_eval_after_support"))
                            continue
                        real_feature_row, _real_late = d1a._real_feature_reference(
                            cfg,
                            per_source_runtime=per_source_runtime,
                            candidates=candidates,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            replicate_seed=int(support_seed),
                            eval_raw=eval_raw,
                            eval_labels=eval_labels,
                        )
                        real_feature_bacc = _float(real_feature_row["bacc"])
                        policies_eval = risk._candidate_policy_bundles(
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
                            eval_sample_ids=eval_ids,
                            source_union_ref=su_ref,
                            center_balanced_ref=cb_ref,
                            real_feature_bacc=real_feature_bacc,
                            include_real_feature=False,
                        )
                        target_policy_by_size[support_size] = policies_eval
                        probability_manifest_rows.extend(risk._probability_manifest_rows(policies_eval, experiment_seed, heldout_center, support_seed, f"target_eval_support{support_size}_primary_style"))
                        if support_size == cfg.primary_labeled_support_size:
                            for method, row in risk._candidate_matrix_rows(policies_eval).items():
                                out = dict(row)
                                out["support_seed"] = int(support_seed)
                                out["support_size"] = int(support_size)
                                out["eval_mode"] = "primary_style"
                                out["selection_source"] = DIAGNOSTIC_SELECTION
                                matrix_rows.append(out)
                            random_single = mb._evaluate_single_plan_control(
                                cfg,
                                root=root,
                                per_source_runtime=per_source_runtime,
                                candidates=candidates,
                                summaries=gmm_summaries,
                                rels=rels,
                                experiment_seed=int(experiment_seed),
                                heldout_center=str(heldout_center),
                                replicate_seed=int(support_seed),
                                eval_raw=eval_raw,
                                eval_labels=eval_labels,
                                source_union_ref=su_ref,
                                center_balanced_ref=cb_ref,
                                real_feature_bacc=real_feature_bacc,
                            )
                            if random_single.get("ensemble_row"):
                                row = dict(random_single["ensemble_row"])
                                row["support_seed"] = int(support_seed)
                                row["support_size"] = int(support_size)
                                row["eval_mode"] = "primary_style"
                                row["diagnostic_only"] = True
                                matrix_rows.append(row)
                            real_feature_row = mb._normalize_row(real_feature_row, prior_method=cu.ROW_REAL_FEATURE_DENSE_REFERENCE)
                            real_feature_row["support_seed"] = int(support_seed)
                            real_feature_row["support_size"] = int(support_size)
                            real_feature_row["eval_mode"] = "primary_style"
                            real_feature_row["selection_source"] = DIAGNOSTIC_SELECTION
                            matrix_rows.append(real_feature_row)
                            matrix_rows.append(cu._reference_matrix_row(cfg, experiment_seed=int(experiment_seed), heldout_center=str(heldout_center), replicate_seed=int(support_seed), candidates=candidates, prior_method=cu.ROW_SOURCE_UNION_K16_REFERENCE, reference=su_ref))
                            matrix_rows.append(cu._reference_matrix_row(cfg, experiment_seed=int(experiment_seed), heldout_center=str(heldout_center), replicate_seed=int(support_seed), candidates=candidates, prior_method=cu.ROW_CENTER_BALANCED_K16_REFERENCE, reference=cb_ref))
                            component_coverage_rows.extend(risk._candidate_coverage_rows(policies_eval, support_seed, support_size))
                            paired_generation_rows.extend(risk._candidate_paired_rows(policies_eval, support_seed, support_size))
                            random_bag_manifest_rows.extend(risk._random_bag_manifest_rows(policies_eval.random_bag, experiment_seed, heldout_center, support_seed, POLICY_RANDOM_BAG))
                            component_manifest_rows.extend(
                                cu._fold_component_manifest_rows(
                                    cfg,
                                    experiment_seed=int(experiment_seed),
                                    heldout_center=str(heldout_center),
                                    candidates=candidates,
                                    summaries=gmm_summaries,
                                    component_details=component_details,
                                    weight_plan=policies_eval.random_bag.ensemble_plan,
                                )
                            )

                    for support_size in split_sizes:
                        split = split_by_key.get((support_size, "primary_style"))
                        if split is None:
                            continue
                        support_raw, support_meta = select_rows(test_cache.embeddings, test_cache.metadata, split.support_indices)
                        support_labels = tuple(_label(row) for row in support_meta)
                        support_ids = tuple(_sample_id(test_cache.metadata[idx], idx) for idx in split.support_indices)
                        policies_support = risk._candidate_policy_bundles(
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
                            eval_labels=support_labels,
                            eval_sample_ids=support_ids,
                            source_union_ref=d1._missing_reference(),
                            center_balanced_ref=d1._missing_reference(),
                            real_feature_bacc=math.nan,
                            include_real_feature=False,
                        )
                        support_policy_by_size[support_size] = policies_support
                        support_scores = {
                            POLICY_RANDOM_BAG: risk._policy_bacc(policies_support, POLICY_RANDOM_BAG),
                            POLICY_DENSE: risk._policy_bacc(policies_support, POLICY_DENSE),
                            POLICY_SHRINK050: risk._policy_bacc(policies_support, POLICY_SHRINK050),
                        }
                        support_scores_by_size[support_size] = support_scores
                        for policy, bacc in support_scores.items():
                            policy_score_rows.append(
                                {
                                    "experiment_seed": int(experiment_seed),
                                    "heldout_center": str(heldout_center),
                                    "support_seed": int(support_seed),
                                    "support_size": int(support_size),
                                    "eval_scope": "labeled_target_support",
                                    "candidate_policy": policy,
                                    "support_bacc": bacc,
                                    "support_labels_used_for_policy_scoring": True,
                                    "support_labels_used_to_train_classifier": False,
                                    "support_labels_used_to_modify_generation": False,
                                    "target_eval_labels_used_for_selection": False,
                                }
                            )
                        quantization_rows.append(_quantization_row(split, cfg.support_quantum_by_size[support_size], support_scores))
                        probability_manifest_rows.extend(risk._probability_manifest_rows(policies_support, experiment_seed, heldout_center, support_seed, f"support_scoring_support{support_size}"))

                    if cfg.primary_labeled_support_size not in target_policy_by_size or cfg.primary_labeled_support_size not in support_scores_by_size:
                        matrix_rows.append(_empty_policy_row(cfg, experiment_seed, heldout_center, support_seed, candidates, cfg.primary_method, "primary_policy_or_support_scores_missing"))
                        continue

                    primary_scores = support_scores_by_size[cfg.primary_labeled_support_size]
                    primary_decision = _support_decision(primary_scores[POLICY_RANDOM_BAG], primary_scores[POLICY_DENSE], cfg.primary_switch_quantum)
                    primary_policies = target_policy_by_size[cfg.primary_labeled_support_size]
                    selected = _selected_policy_row(
                        cfg,
                        primary_policies,
                        primary_decision["selected_policy"],
                        experiment_seed,
                        heldout_center,
                        support_seed,
                        candidates,
                        su_ref,
                        cb_ref,
                        method=cfg.primary_method,
                        selection_source=PRIMARY_SELECTION,
                        claim_role="primary_labeled_support16_random_default_dense_switch",
                        support_size=cfg.primary_labeled_support_size,
                        eval_mode="primary_style",
                        decision=primary_decision,
                    )
                    matrix_rows.append(selected)
                    selection_rows.append(_selection_row(experiment_seed, heldout_center, support_seed, cfg.primary_labeled_support_size, primary_decision, selected))
                    eligibility_rows.append(_eligibility_row(experiment_seed, heldout_center, support_seed, cfg.primary_method, str(selected.get("status", "")), str(selected.get("error_message", ""))))

                    oracle_policy = _oracle_best_random_dense_policy(primary_policies)
                    oracle_row = _selected_policy_row(
                        cfg,
                        primary_policies,
                        oracle_policy,
                        experiment_seed,
                        heldout_center,
                        support_seed,
                        candidates,
                        su_ref,
                        cb_ref,
                        method=ROW_ORACLE_BEST_POLICY,
                        selection_source=DIAGNOSTIC_SELECTION,
                        claim_role="oracle_best_random_vs_dense_policy_diagnostic_only",
                        support_size=cfg.primary_labeled_support_size,
                        eval_mode="primary_style",
                        decision={"selected_policy": oracle_policy, "switch_to_dense": oracle_policy == POLICY_DENSE, "selection_rule": "oracle_diagnostic_only"},
                    )
                    oracle_row["oracle_policy_used_for_selection"] = True
                    oracle_row["diagnostic_only"] = True
                    matrix_rows.append(oracle_row)

                    shuffled_decision = _shuffled_support_label_decision(
                        cfg,
                        support_policy_by_size[cfg.primary_labeled_support_size],
                        primary_split.support_labels,
                        experiment_seed,
                        heldout_center,
                        support_seed,
                    )
                    matrix_rows.append(_selected_policy_row(cfg, primary_policies, shuffled_decision["selected_policy"], experiment_seed, heldout_center, support_seed, candidates, su_ref, cb_ref, method=ROW_SHUFFLED_SUPPORT_LABEL_CONTROL, selection_source=DIAGNOSTIC_SELECTION, claim_role="negative_control_shuffled_support_label_switch", support_size=16, eval_mode="primary_style", decision=shuffled_decision))

                    try:
                        off_target_decision = _off_target_support_decision(
                            cfg,
                            root=root,
                            test_cache=test_cache,
                            per_source_runtime=per_source_runtime,
                            dense_summaries=dense_summaries,
                            gmm_summaries=gmm_summaries,
                            candidates=candidates,
                            rels=rels,
                            experiment_seed=int(experiment_seed),
                            real_heldout_center=str(heldout_center),
                            support_seed=int(support_seed),
                        )
                    except ProtocolError as exc:
                        off_target_decision = {
                            "selected_policy": POLICY_RANDOM_BAG,
                            "switch_to_dense": False,
                            "random_support_bacc": math.nan,
                            "dense_support_bacc": math.nan,
                            "dense_minus_random_support_bacc": math.nan,
                            "support_quantum": cfg.primary_switch_quantum,
                            "selection_rule": f"off_target_labeled_support_control_ineligible:{exc}",
                        }
                        eligibility_rows.append(
                            _eligibility_row(
                                experiment_seed,
                                heldout_center,
                                support_seed,
                                ROW_OFF_TARGET_SUPPORT_CONTROL,
                                "ineligible_control_fallback_random",
                                str(exc),
                            )
                        )
                    matrix_rows.append(_selected_policy_row(cfg, primary_policies, off_target_decision["selected_policy"], experiment_seed, heldout_center, support_seed, candidates, su_ref, cb_ref, method=ROW_OFF_TARGET_SUPPORT_CONTROL, selection_source=DIAGNOSTIC_SELECTION, claim_role="negative_control_off_target_labeled_support_switch", support_size=16, eval_mode="primary_style", decision=off_target_decision))

                    switch_event_rows.append(_switch_event_row(experiment_seed, heldout_center, support_seed, primary_scores, primary_decision, primary_policies))
                    utility_alignment_rows.extend(_utility_alignment_rows(experiment_seed, heldout_center, support_seed, primary_scores, primary_policies))
                    target_oracle_rows.append(_target_oracle_row(experiment_seed, heldout_center, support_seed, primary_decision, primary_policies))

                    for support_size in (8, 32):
                        if support_size in target_policy_by_size and support_size in support_scores_by_size:
                            decision = _support_decision(support_scores_by_size[support_size][POLICY_RANDOM_BAG], support_scores_by_size[support_size][POLICY_DENSE], cfg.support_quantum_by_size[support_size])
                            method = f"{ROW_SUPPORT_SIZE_DIAGNOSTIC_PREFIX}{support_size}_random_default_dense_switch_diagnostic"
                            row = _selected_policy_row(cfg, target_policy_by_size[support_size], decision["selected_policy"], experiment_seed, heldout_center, support_seed, candidates, su_ref, cb_ref, method=method, selection_source=DIAGNOSTIC_SELECTION, claim_role="support_size_specific_eval_diagnostic_only", support_size=support_size, eval_mode="primary_style", decision=decision)
                            row["diagnostic_only"] = True
                            matrix_rows.append(row)
                    for support_size in (8, 16, 32):
                        if 32 in target_policy_by_size and support_size in support_scores_by_size:
                            decision = _support_decision(support_scores_by_size[support_size][POLICY_RANDOM_BAG], support_scores_by_size[support_size][POLICY_DENSE], cfg.support_quantum_by_size[support_size])
                            method = f"{ROW_COMMON_EVAL_DIAGNOSTIC_PREFIX}{support_size}_random_default_dense_switch_diagnostic"
                            row = _selected_policy_row(cfg, target_policy_by_size[32], decision["selected_policy"], experiment_seed, heldout_center, support_seed, candidates, su_ref, cb_ref, method=method, selection_source=DIAGNOSTIC_SELECTION, claim_role="common_eval_excluding_support32_diagnostic_only", support_size=support_size, eval_mode="fixed_support32", decision=decision)
                            row["diagnostic_only"] = True
                            common_eval_rows.append(_common_eval_audit_row(row, support_scores_by_size[support_size], decision, target_policy_by_size[32]))
                    runtime_rows.append(_runtime_row(experiment_seed, heldout_center, support_seed, root))
                    print(
                        f"[labeled_support] cell_done seed={experiment_seed} heldout={heldout_center} support_seed={support_seed} selected={primary_decision['selected_policy']} bacc={_format_float(selected.get('bacc'))}",
                        flush=True,
                    )

    except ProtocolError as exc:
        protocol_violations.append(str(exc))

    matrix_rows.extend(_random_switch_control_rows(cfg, matrix_rows, per_center=False))
    matrix_rows.extend(_random_switch_control_rows(cfg, matrix_rows, per_center=True))
    bottom20_keys = _bottom20_keys(matrix_rows)
    tail_rows = risk._tail_metric_summary_rows(matrix_rows, bottom20_keys)
    summary = _decision(matrix_rows, tail_rows, utility_alignment_rows, target_oracle_rows, cfg, protocol_violations)
    negative_rows = _negative_control_rows(matrix_rows, bottom20_keys, cfg)
    oracle_rows = _oracle_gap_rows(matrix_rows, cfg)
    alignment_summary = _alignment_summary(utility_alignment_rows)
    for row in utility_alignment_rows:
        row.update(alignment_summary)
    leakage = _labeled_support_leakage_report(
        protocol_violations=protocol_violations,
        target_expert_excluded=target_expert_excluded,
        support_eval_disjoint=True,
    )
    _write_artifacts(
        root,
        cfg,
        matrix_rows=matrix_rows,
        split_rows=split_rows,
        policy_score_rows=policy_score_rows,
        selection_rows=selection_rows,
        switch_event_rows=switch_event_rows,
        probability_manifest_rows=probability_manifest_rows,
        random_bag_manifest_rows=random_bag_manifest_rows,
        utility_alignment_rows=utility_alignment_rows,
        quantization_rows=quantization_rows,
        common_eval_rows=common_eval_rows,
        negative_rows=negative_rows,
        oracle_rows=oracle_rows,
        target_oracle_rows=target_oracle_rows,
        eligibility_rows=eligibility_rows,
        runtime_rows=runtime_rows,
        component_manifest_rows=component_manifest_rows,
        component_coverage_rows=component_coverage_rows,
        paired_generation_rows=paired_generation_rows,
        tail_rows=tail_rows,
        summary=summary,
        leakage=leakage,
        protocol_violations=protocol_violations,
        target_expert_excluded=target_expert_excluded,
    )
    return root


def nested_labeled_support_eval_splits(
    metadata: Sequence[Mapping[str, object]],
    *,
    heldout_center: str,
    support_seed: int,
    support_sizes: Sequence[int],
    max_support_size: int,
    center_key: str = "center",
    label_key: str = "label",
    sample_id_key: str = "sample_id",
) -> tuple[LabeledNestedSupportEvalSplit, ...]:
    if max_support_size % 2 != 0:
        raise ProtocolError("max_support_size must be class-balanced and even.")
    requested_sizes = tuple(sorted({int(value) for value in support_sizes}))
    target_by_label: dict[int, list[int]] = {0: [], 1: []}
    for idx, row in enumerate(metadata):
        if _row_center(row, center_key=center_key) == str(heldout_center):
            label = int(row[label_key])
            if label in target_by_label:
                target_by_label[label].append(idx)
    min_class_count = min(len(values) for values in target_by_label.values())
    feasible_sizes = tuple(size for size in requested_sizes if size % 2 == 0 and min_class_count > (size // 2))
    if not feasible_sizes:
        smallest = min(requested_sizes) if requested_sizes else max_support_size
        raise ProtocolError(f"Need more than {smallest // 2} target samples per class for labeled support split.")
    parent_size = max_support_size if max_support_size in feasible_sizes else max(feasible_sizes)
    per_class_parent = parent_size // 2
    rng = random.Random(int(support_seed))
    parent_by_label: dict[int, tuple[int, ...]] = {}
    for label, values in target_by_label.items():
        shuffled = list(values)
        rng.shuffle(shuffled)
        parent_by_label[label] = tuple(shuffled[:per_class_parent])
    parent_support = tuple(sorted(parent_by_label[0] + parent_by_label[1]))
    target_indices = tuple(sorted(target_by_label[0] + target_by_label[1]))
    parent_eval = tuple(idx for idx in target_indices if idx not in set(parent_support))
    parent_id = f"target{heldout_center}_seed{support_seed}_nested_labeled_k{parent_size}"
    out: list[LabeledNestedSupportEvalSplit] = []
    for support_size in requested_sizes:
        if support_size not in feasible_sizes:
            continue
        per_class = int(support_size) // 2
        support = tuple(sorted(parent_by_label[0][:per_class] + parent_by_label[1][:per_class]))
        primary_eval = tuple(idx for idx in target_indices if idx not in set(support))
        eval_modes: list[tuple[str, tuple[int, ...]]] = [("primary_style", primary_eval)]
        if max_support_size in feasible_sizes:
            eval_modes.append(("fixed_support32", parent_eval))
        for eval_mode, eval_indices in eval_modes:
            support_ids = tuple(_sample_id(metadata[idx], idx, sample_id_key=sample_id_key) for idx in support)
            eval_ids = tuple(_sample_id(metadata[idx], idx, sample_id_key=sample_id_key) for idx in eval_indices)
            assert_support_eval_disjoint(support_ids, eval_ids)
            support_labels = tuple(int(metadata[idx][label_key]) for idx in support)
            if support_labels.count(0) != support_labels.count(1):
                raise ProtocolError("Labeled support split is not class-balanced.")
            out.append(
                LabeledNestedSupportEvalSplit(
                    heldout_center=str(heldout_center),
                    support_size=int(support_size),
                    support_seed=int(support_seed),
                    eval_mode=eval_mode,
                    support_indices=support,
                    eval_indices=eval_indices,
                    support_sample_ids=support_ids,
                    eval_sample_ids=eval_ids,
                    support_labels=support_labels,
                    support_eval_split_id=f"{parent_id}_support{support_size}_{eval_mode}",
                    parent_support32_split_id=parent_id,
                )
            )
    return tuple(out)


def _support_decision(random_support_bacc: float, dense_support_bacc: float, quantum: float) -> dict[str, object]:
    delta = _delta(dense_support_bacc, random_support_bacc)
    switch = bool(math.isfinite(delta) and delta >= float(quantum))
    selected = POLICY_DENSE if switch else POLICY_RANDOM_BAG
    return {
        "selected_policy": selected,
        "switch_to_dense": switch,
        "random_support_bacc": random_support_bacc,
        "dense_support_bacc": dense_support_bacc,
        "dense_minus_random_support_bacc": delta,
        "support_quantum": float(quantum),
        "selection_rule": "dense_if_dense_support_bacc_ge_random_plus_one_quantum",
    }


def _selected_policy_row(
    cfg: LabeledSupportPolicyCalibrationConfig,
    policies: risk.PolicyBundleSet,
    policy: str,
    experiment_seed: int,
    heldout_center: str,
    support_seed: int,
    candidates: Sequence[str],
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
    *,
    method: str,
    selection_source: str,
    claim_role: str,
    support_size: int,
    eval_mode: str,
    decision: Mapping[str, object],
) -> dict[str, object]:
    row, bundle = risk._policy_source_row_bundle(policies, policy)
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
            real_feature_bacc=math.nan,
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
            "support_size": int(support_size),
            "eval_mode": str(eval_mode),
            "selected_policy": policy,
            "switch_to_dense": int(bool(decision.get("switch_to_dense", False))),
            "random_support_bacc": decision.get("random_support_bacc", math.nan),
            "dense_support_bacc": decision.get("dense_support_bacc", math.nan),
            "dense_minus_random_support_bacc": decision.get("dense_minus_random_support_bacc", math.nan),
            "support_quantum": decision.get("support_quantum", math.nan),
            "selection_rule": decision.get("selection_rule", ""),
            "target_support_labels_used_for_policy_selection": True,
            "support_labels_used_to_train_classifier": False,
            "support_labels_used_to_modify_generation": False,
            "target_eval_labels_used_for_scoring_only": True,
            "target_eval_labels_used_for_selection": False,
            "oracle_policy_used_for_selection": False,
        }
    )
    return out


def _shuffled_support_label_decision(
    cfg: LabeledSupportPolicyCalibrationConfig,
    support_policies: risk.PolicyBundleSet,
    support_labels: Sequence[int],
    experiment_seed: int,
    heldout_center: str,
    support_seed: int,
) -> dict[str, object]:
    labels = list(int(v) for v in support_labels)
    rng = random.Random(d1._latent_seed(experiment_seed, heldout_center, support_seed, "labeled_support_shuffle_labels"))
    rng.shuffle(labels)
    random_bundle = support_policies.random_bag.ensemble_bundle
    dense_bundle = support_policies.dense_reliability.bundle
    random_bacc = _bundle_bacc(random_bundle, labels)
    dense_bacc = _bundle_bacc(dense_bundle, labels)
    decision = _support_decision(random_bacc, dense_bacc, cfg.primary_switch_quantum)
    decision["selection_rule"] = "shuffled_support_label_control"
    return decision


def _off_target_support_decision(
    cfg: LabeledSupportPolicyCalibrationConfig,
    *,
    root: Path,
    test_cache: object,
    per_source_runtime: Mapping[str, RuntimeSource],
    dense_summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    gmm_summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    candidates: Sequence[str],
    rels: Mapping[str, d12.SourceReliability],
    experiment_seed: int,
    real_heldout_center: str,
    support_seed: int,
) -> dict[str, object]:
    off_target = sorted(str(source) for source in candidates)[0]
    splits = nested_labeled_support_eval_splits(
        test_cache.metadata,
        heldout_center=off_target,
        support_seed=support_seed,
        support_sizes=(cfg.primary_labeled_support_size,),
        max_support_size=cfg.nested_support_max_size,
    )
    split = [item for item in splits if item.eval_mode == "primary_style"][0]
    support_raw, support_meta = select_rows(test_cache.embeddings, test_cache.metadata, split.support_indices)
    support_labels = tuple(_label(row) for row in support_meta)
    support_ids = tuple(_sample_id(test_cache.metadata[idx], idx) for idx in split.support_indices)
    support_policies = risk._candidate_policy_bundles(
        cfg,
        root=root,
        per_source_runtime=per_source_runtime,
        dense_summaries=dense_summaries,
        gmm_summaries=gmm_summaries,
        candidates=candidates,
        rels=rels,
        experiment_seed=int(experiment_seed),
        heldout_center=str(real_heldout_center),
        support_seed=int(support_seed),
        eval_raw=support_raw,
        eval_labels=support_labels,
        eval_sample_ids=support_ids,
        source_union_ref=d1._missing_reference(),
        center_balanced_ref=d1._missing_reference(),
        real_feature_bacc=math.nan,
        include_real_feature=False,
    )
    decision = _support_decision(risk._policy_bacc(support_policies, POLICY_RANDOM_BAG), risk._policy_bacc(support_policies, POLICY_DENSE), cfg.primary_switch_quantum)
    decision["selection_rule"] = f"off_target_labeled_support_control:{off_target}"
    decision["off_target_support_center"] = off_target
    return decision


def _random_switch_control_rows(cfg: LabeledSupportPolicyCalibrationConfig, rows: Sequence[Mapping[str, object]], *, per_center: bool) -> list[dict[str, object]]:
    primary_rows = cu._rows_for(rows, cfg.primary_method)
    if not primary_rows:
        return []
    method = ROW_RANDOM_DEFAULT_CONTROL if per_center else ROW_RANDOM_SWITCH_MATCHED_RATE
    out = []
    groups: dict[str, list[Mapping[str, object]]] = {}
    for row in primary_rows:
        key = str(row.get("heldout_center")) if per_center else "global"
        groups.setdefault(key, []).append(row)
    dense_rates = {
        key: sum(1 for row in value if str(row.get("selected_policy")) == POLICY_DENSE) / float(len(value))
        for key, value in groups.items()
    }
    for row in primary_rows:
        key = str(row.get("heldout_center")) if per_center else "global"
        seed = d1._latent_seed(row.get("experiment_seed", 0), row.get("heldout_center", ""), row.get("support_seed", 0), method)
        rng = random.Random(seed)
        select_dense = rng.random() < dense_rates.get(key, 0.0)
        source_method = ROW_ALWAYS_DENSE if select_dense else ROW_ALWAYS_RANDOM_BAG
        source = _matching_row(rows, row, source_method)
        if source is None:
            continue
        control = dict(source)
        control.update(
            {
                "prior_method": method,
                "selection_source": DIAGNOSTIC_SELECTION,
                "claim_role": "negative_control_random_switch_matched_primary_switch_rate",
                "selected_policy": POLICY_DENSE if select_dense else POLICY_RANDOM_BAG,
                "switch_to_dense": int(select_dense),
                "diagnostic_only": True,
                "random_switch_rate_matched_scope": key,
                "oracle_policy_used_for_selection": False,
            }
        )
        out.append(control)
    return out


def _matching_row(rows: Sequence[Mapping[str, object]], key_row: Mapping[str, object], method: str) -> dict[str, object] | None:
    for row in rows:
        if (
            row.get("prior_method") == method
            and str(row.get("experiment_seed")) == str(key_row.get("experiment_seed"))
            and str(row.get("heldout_center")) == str(key_row.get("heldout_center"))
            and str(row.get("replicate_seed", row.get("support_seed"))) == str(key_row.get("replicate_seed", key_row.get("support_seed")))
            and str(row.get("support_size", "16")) == str(key_row.get("support_size", "16"))
        ):
            return dict(row)
    return None


def _bundle_bacc(bundle: PredictionBundle | None, labels: Sequence[int]) -> float:
    if bundle is None:
        return math.nan
    result = evaluate_probability_predictions("support_score", bundle.probabilities, labels, classes=bundle.classes)
    return float(result.bacc)


def _oracle_best_random_dense_policy(policies: risk.PolicyBundleSet) -> str:
    random_bacc = risk._policy_bacc(policies, POLICY_RANDOM_BAG)
    dense_bacc = risk._policy_bacc(policies, POLICY_DENSE)
    return POLICY_DENSE if dense_bacc > random_bacc else POLICY_RANDOM_BAG


def _switch_event_row(
    experiment_seed: int,
    heldout_center: str,
    support_seed: int,
    support_scores: Mapping[str, float],
    decision: Mapping[str, object],
    target_policies: risk.PolicyBundleSet,
) -> dict[str, object]:
    random_bacc = risk._policy_bacc(target_policies, POLICY_RANDOM_BAG)
    dense_bacc = risk._policy_bacc(target_policies, POLICY_DENSE)
    selected_bacc = dense_bacc if decision["selected_policy"] == POLICY_DENSE else random_bacc
    oracle_bacc = max(random_bacc, dense_bacc)
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "support_seed": int(support_seed),
        "random_support_bacc": support_scores.get(POLICY_RANDOM_BAG, math.nan),
        "dense_support_bacc": support_scores.get(POLICY_DENSE, math.nan),
        "dense_minus_random_support_bacc": _delta(support_scores.get(POLICY_DENSE), support_scores.get(POLICY_RANDOM_BAG)),
        "support16_quantum": 0.0625,
        "selected_policy": decision.get("selected_policy", ""),
        "switch_to_dense": bool(decision.get("switch_to_dense", False)),
        "target_bacc_random": random_bacc,
        "target_bacc_dense": dense_bacc,
        "target_bacc_selected": selected_bacc,
        "target_bacc_oracle": oracle_bacc,
        "selected_minus_random": _delta(selected_bacc, random_bacc),
        "selected_minus_dense": _delta(selected_bacc, dense_bacc),
        "selected_minus_oracle": _delta(selected_bacc, oracle_bacc),
        "support_margin_over_random": _delta(support_scores.get(POLICY_DENSE), support_scores.get(POLICY_RANDOM_BAG)),
    }


def _utility_alignment_rows(
    experiment_seed: int,
    heldout_center: str,
    support_seed: int,
    support_scores: Mapping[str, float],
    target_policies: risk.PolicyBundleSet,
) -> list[dict[str, object]]:
    out = []
    for policy in (POLICY_RANDOM_BAG, POLICY_DENSE):
        out.append(
            {
                "experiment_seed": int(experiment_seed),
                "heldout_center": str(heldout_center),
                "support_seed": int(support_seed),
                "candidate_policy": policy,
                "support_bacc": support_scores.get(policy, math.nan),
                "target_bacc": risk._policy_bacc(target_policies, policy),
            }
        )
    return out


def _alignment_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    support_values = [_float(row.get("support_bacc")) for row in rows]
    target_values = [_float(row.get("target_bacc")) for row in rows]
    spearman = _spearman(support_values, target_values)
    by_cell: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in rows:
        key = (str(row.get("experiment_seed")), str(row.get("heldout_center")), str(row.get("support_seed")))
        by_cell.setdefault(key, {})[str(row.get("candidate_policy"))] = (_float(row.get("support_bacc")), _float(row.get("target_bacc")))  # type: ignore[assignment]
    scores = []
    for values in by_cell.values():
        if POLICY_RANDOM_BAG not in values or POLICY_DENSE not in values:
            continue
        random_support, random_target = values[POLICY_RANDOM_BAG]  # type: ignore[misc]
        dense_support, dense_target = values[POLICY_DENSE]  # type: ignore[misc]
        support_delta = dense_support - random_support
        target_delta = dense_target - random_target
        if not (math.isfinite(support_delta) and math.isfinite(target_delta)):
            continue
        if target_delta == 0.0 or support_delta == 0.0:
            scores.append(0.5)
        else:
            scores.append(1.0 if (target_delta > 0) == (support_delta > 0) else 0.0)
    return {
        "aggregate_spearman_support_bacc_vs_target_bacc": spearman,
        "within_cell_pairwise_policy_auc": nanmean(scores),
    }


def _target_oracle_row(
    experiment_seed: int,
    heldout_center: str,
    support_seed: int,
    decision: Mapping[str, object],
    target_policies: risk.PolicyBundleSet,
) -> dict[str, object]:
    random_bacc = risk._policy_bacc(target_policies, POLICY_RANDOM_BAG)
    dense_bacc = risk._policy_bacc(target_policies, POLICY_DENSE)
    dense_favored = dense_bacc > random_bacc
    switched = decision.get("selected_policy") == POLICY_DENSE
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "support_seed": int(support_seed),
        "random_target_bacc": random_bacc,
        "dense_target_bacc": dense_bacc,
        "target_oracle_dense_favored": int(dense_favored),
        "selected_policy": decision.get("selected_policy", ""),
        "switch_to_dense": int(switched),
        "dense_switch_true_positive": int(switched and dense_favored),
        "dense_switch_false_positive": int(switched and not dense_favored),
        "dense_switch_false_negative": int((not switched) and dense_favored),
        "diagnostic_only": True,
    }


def _quantization_row(split: LabeledNestedSupportEvalSplit, quantum: float, support_scores: Mapping[str, float]) -> dict[str, object]:
    return {
        "heldout_center": split.heldout_center,
        "support_seed": int(split.support_seed),
        "support_size": int(split.support_size),
        "eval_mode": split.eval_mode,
        "support_size_total": len(split.support_indices),
        "support_count_class0": split.support_labels.count(0),
        "support_count_class1": split.support_labels.count(1),
        "bacc_quantum": float(quantum),
        "random_support_bacc": support_scores.get(POLICY_RANDOM_BAG, math.nan),
        "dense_support_bacc": support_scores.get(POLICY_DENSE, math.nan),
        "dense_minus_random_support_bacc": _delta(support_scores.get(POLICY_DENSE), support_scores.get(POLICY_RANDOM_BAG)),
        "class_balanced_support": split.class_balanced_support,
    }


def _common_eval_audit_row(
    row: Mapping[str, object],
    support_scores: Mapping[str, float],
    decision: Mapping[str, object],
    target_policies: risk.PolicyBundleSet,
) -> dict[str, object]:
    return {
        "experiment_seed": row.get("experiment_seed", ""),
        "heldout_center": row.get("heldout_center", ""),
        "support_seed": row.get("support_seed", ""),
        "support_size": row.get("support_size", ""),
        "common_eval_definition": "target_eval_excluding_support32",
        "selected_policy": decision.get("selected_policy", ""),
        "switch_to_dense": int(bool(decision.get("switch_to_dense", False))),
        "random_support_bacc": support_scores.get(POLICY_RANDOM_BAG, math.nan),
        "dense_support_bacc": support_scores.get(POLICY_DENSE, math.nan),
        "target_bacc_random": risk._policy_bacc(target_policies, POLICY_RANDOM_BAG),
        "target_bacc_dense": risk._policy_bacc(target_policies, POLICY_DENSE),
        "target_bacc_selected": row.get("bacc", math.nan),
        "diagnostic_only": True,
    }


def _selection_row(
    experiment_seed: int,
    heldout_center: str,
    support_seed: int,
    support_size: int,
    decision: Mapping[str, object],
    selected_row: Mapping[str, object],
) -> dict[str, object]:
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "support_seed": int(support_seed),
        "support_size": int(support_size),
        "selected_policy": decision.get("selected_policy", ""),
        "switch_to_dense": int(bool(decision.get("switch_to_dense", False))),
        "random_support_bacc": decision.get("random_support_bacc", math.nan),
        "dense_support_bacc": decision.get("dense_support_bacc", math.nan),
        "support_quantum": decision.get("support_quantum", math.nan),
        "selected_target_bacc": selected_row.get("bacc", math.nan),
        "target_support_labels_used_for_policy_selection": True,
        "target_eval_labels_used_for_scoring_only": True,
    }


def _split_manifest_rows(
    splits: Sequence[LabeledNestedSupportEvalSplit],
    experiment_seed: int,
    support_seed: int,
    scope: str,
) -> list[dict[str, object]]:
    return [
        {
            "experiment_seed": int(experiment_seed),
            "support_seed": int(support_seed),
            "heldout_center": split.heldout_center,
            "support_size": split.support_size,
            "eval_mode": split.eval_mode,
            "split_scope": scope,
            "support_eval_split_id": split.support_eval_split_id,
            "parent_support32_split_id": split.parent_support32_split_id,
            "support_labels_used": int(split.support_labels_used),
            "support_size_actual": len(split.support_indices),
            "support_count_class0": split.support_labels.count(0),
            "support_count_class1": split.support_labels.count(1),
            "class_balanced_support": int(split.class_balanced_support),
            "n_target_eval": len(split.eval_indices),
            "support_sample_id_hash": _hash_strings(split.support_sample_ids),
            "eval_sample_id_hash": _hash_strings(split.eval_sample_ids),
            "support_eval_disjoint": 1,
            "size_specific_eval_exclusion": int(split.eval_mode == "primary_style"),
            "common_eval_excluding_support32": int(split.eval_mode == "fixed_support32"),
        }
        for split in splits
    ]


def _empty_policy_row(
    cfg: LabeledSupportPolicyCalibrationConfig,
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
        claim_role="primary_labeled_support16_random_default_dense_switch",
    )
    row["support_seed"] = int(support_seed)
    row["support_size"] = cfg.primary_labeled_support_size
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


def _negative_control_rows(rows: Sequence[Mapping[str, object]], bottom20_keys: set[tuple[str, str, str]], cfg: LabeledSupportPolicyCalibrationConfig) -> list[dict[str, object]]:
    primary = risk.dense._tail_metrics(rows, cfg.primary_method, bottom20_keys=bottom20_keys)
    out = []
    for method in (ROW_SHUFFLED_SUPPORT_LABEL_CONTROL, ROW_OFF_TARGET_SUPPORT_CONTROL, ROW_RANDOM_SWITCH_MATCHED_RATE, ROW_RANDOM_DEFAULT_CONTROL):
        metrics = risk.dense._tail_metrics(rows, method, bottom20_keys=bottom20_keys)
        out.append(
            {
                "primary_method": cfg.primary_method,
                "control_method": method,
                "primary_center_equal_mean_bacc": primary.get("center_equal_mean_bacc"),
                "control_center_equal_mean_bacc": metrics.get("center_equal_mean_bacc"),
                "primary_minus_control_bacc": _delta(primary.get("center_equal_mean_bacc"), metrics.get("center_equal_mean_bacc")),
                "primary_bottom20_minus_control": _delta(primary.get("bottom20_cell_mean_bacc"), metrics.get("bottom20_cell_mean_bacc")),
                "diagnostic_only": True,
            }
        )
    return out


def _oracle_gap_rows(rows: Sequence[Mapping[str, object]], cfg: LabeledSupportPolicyCalibrationConfig) -> list[dict[str, object]]:
    primary = {_key(row): row for row in cu._rows_for(rows, cfg.primary_method)}
    oracle = {_key(row): row for row in cu._rows_for(rows, ROW_ORACLE_BEST_POLICY)}
    random_rows = {_key(row): row for row in cu._rows_for(rows, ROW_ALWAYS_RANDOM_BAG)}
    out = []
    for key, row in primary.items():
        oracle_row = oracle.get(key)
        random_row = random_rows.get(key)
        oracle_gap = math.nan if oracle_row is None else _delta(oracle_row.get("bacc"), row.get("bacc"))
        random_gap = math.nan if oracle_row is None or random_row is None else _delta(oracle_row.get("bacc"), random_row.get("bacc"))
        reduction = math.nan
        if math.isfinite(oracle_gap) and math.isfinite(random_gap) and random_gap > 0:
            reduction = (random_gap - oracle_gap) / random_gap
        out.append(
            {
                "experiment_seed": key[0],
                "heldout_center": key[1],
                "support_seed": key[2],
                "primary_bacc": row.get("bacc", math.nan),
                "random_mass_bag_bacc": "" if random_row is None else random_row.get("bacc", math.nan),
                "oracle_best_random_dense_bacc": "" if oracle_row is None else oracle_row.get("bacc", math.nan),
                "selected_minus_random_bacc": "" if random_row is None else _delta(row.get("bacc"), random_row.get("bacc")),
                "oracle_gap": oracle_gap,
                "random_oracle_gap": random_gap,
                "oracle_gap_reduction_vs_random": reduction,
                "oracle_policy_diagnostic_only": True,
            }
        )
    return out


def _decision(
    rows: Sequence[Mapping[str, object]],
    tail_rows: Sequence[Mapping[str, object]],
    utility_alignment_rows: Sequence[Mapping[str, object]],
    target_oracle_rows: Sequence[Mapping[str, object]],
    cfg: LabeledSupportPolicyCalibrationConfig,
    protocol_violations: Sequence[str],
) -> dict[str, object]:
    by_method = {row["prior_method"]: row for row in tail_rows}
    primary = by_method.get(cfg.primary_method, {})
    random_metrics = by_method.get(ROW_ALWAYS_RANDOM_BAG, {})
    controls = [by_method[method] for method in (ROW_SHUFFLED_SUPPORT_LABEL_CONTROL, ROW_OFF_TARGET_SUPPORT_CONTROL, ROW_RANDOM_SWITCH_MATCHED_RATE, ROW_RANDOM_DEFAULT_CONTROL) if method in by_method]
    alignment = _alignment_summary(utility_alignment_rows)
    target_recall = _dense_switch_recall(target_oracle_rows)
    target_precision = _dense_switch_precision(target_oracle_rows)
    rates = _selection_rates(cu._rows_for(rows, cfg.primary_method))
    oracle_reductions = [_float(row.get("oracle_gap_reduction_vs_random")) for row in _oracle_gap_rows(rows, cfg)]
    control_bottom20_max = _max_metric(controls, "bottom20_cell_mean_bacc")
    control_worst_max = _max_metric(controls, "worst_seed_center_bacc")
    control_tail_not_matched = (
        _float(primary.get("bottom20_cell_mean_bacc")) > control_bottom20_max
        or _float(primary.get("worst_seed_center_bacc")) > control_worst_max
    )
    flags = []
    if protocol_violations:
        flags.append("PROTOCOL_VIOLATION")
    if _float(alignment.get("within_cell_pairwise_policy_auc")) <= 0.55:
        flags.append("SUPPORT16_POLICY_AUC_LOW")
    if not control_tail_not_matched:
        flags.append("SUPPORT_LABEL_CONTROLS_MATCH_PRIMARY")
    if _delta(primary.get("center_equal_mean_bacc"), random_metrics.get("center_equal_mean_bacc")) < -0.015:
        flags.append("MEAN_DROP_GT_0P015")
    if _float(primary.get("min_center_bacc")) < 0.80:
        flags.append("MIN_CENTER_BELOW_0P80")
    useful = (
        not protocol_violations
        and _delta(primary.get("center_equal_mean_bacc"), random_metrics.get("center_equal_mean_bacc")) >= -0.010
        and _float(primary.get("min_center_bacc")) >= 0.80
        and _float(alignment.get("within_cell_pairwise_policy_auc")) > 0.55
        and nanmean(oracle_reductions) > 0.0
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
        and _float(primary.get("seed_std_bacc")) <= _float(random_metrics.get("seed_std_bacc"))
        and _float(primary.get("seed_std_bacc")) <= 0.045
        and _float(alignment.get("within_cell_pairwise_policy_auc")) > 0.60
        and math.isfinite(target_recall)
        and target_recall >= 0.50
        and nanmean(oracle_reductions) >= 0.25
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
        "center3_delta_vs_random_mass_bag": _delta(primary.get("center3_bacc"), random_metrics.get("center3_bacc")),
        "aggregate_spearman_support_bacc_vs_target_bacc": alignment.get("aggregate_spearman_support_bacc_vs_target_bacc"),
        "within_cell_pairwise_policy_auc": alignment.get("within_cell_pairwise_policy_auc"),
        "dense_switch_precision_against_target_oracle": target_precision,
        "dense_switch_recall_against_target_oracle": target_recall,
        "oracle_gap_reduction_vs_random": nanmean(oracle_reductions),
        "control_tail_not_matched": bool(control_tail_not_matched),
        "max_control_bottom20_cell_mean_bacc": control_bottom20_max,
        "max_control_worst_seed_center_bacc": control_worst_max,
        **rates,
    }


def _selection_rates(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    total = len(rows)
    if not total:
        return {"selected_random_mass_bag_rate": math.nan, "selected_dense_reliability_rate": math.nan, "dense_switch_rate": math.nan}
    dense_count = sum(1 for row in rows if row.get("selected_policy") == POLICY_DENSE)
    return {
        "selected_random_mass_bag_rate": sum(1 for row in rows if row.get("selected_policy") == POLICY_RANDOM_BAG) / total,
        "selected_dense_reliability_rate": dense_count / total,
        "dense_switch_rate": dense_count / total,
    }


def _dense_switch_recall(rows: Sequence[Mapping[str, object]]) -> float:
    dense_favored = [row for row in rows if _float(row.get("target_oracle_dense_favored")) == 1.0]
    if not dense_favored:
        return math.nan
    return sum(1 for row in dense_favored if _float(row.get("switch_to_dense")) == 1.0) / float(len(dense_favored))


def _dense_switch_precision(rows: Sequence[Mapping[str, object]]) -> float:
    switched = [row for row in rows if _float(row.get("switch_to_dense")) == 1.0]
    if not switched:
        return math.nan
    return sum(1 for row in switched if _float(row.get("target_oracle_dense_favored")) == 1.0) / float(len(switched))


def _bottom20_keys(rows: Sequence[Mapping[str, object]]) -> set[tuple[str, str, str]]:
    return risk.dense._bottom20_raw_cell_keys(rows, ROW_ALWAYS_RANDOM_BAG)


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


def _write_artifacts(
    root: Path,
    cfg: LabeledSupportPolicyCalibrationConfig,
    *,
    matrix_rows: Sequence[Mapping[str, object]],
    split_rows: Sequence[Mapping[str, object]],
    policy_score_rows: Sequence[Mapping[str, object]],
    selection_rows: Sequence[Mapping[str, object]],
    switch_event_rows: Sequence[Mapping[str, object]],
    probability_manifest_rows: Sequence[Mapping[str, object]],
    random_bag_manifest_rows: Sequence[Mapping[str, object]],
    utility_alignment_rows: Sequence[Mapping[str, object]],
    quantization_rows: Sequence[Mapping[str, object]],
    common_eval_rows: Sequence[Mapping[str, object]],
    negative_rows: Sequence[Mapping[str, object]],
    oracle_rows: Sequence[Mapping[str, object]],
    target_oracle_rows: Sequence[Mapping[str, object]],
    eligibility_rows: Sequence[Mapping[str, object]],
    runtime_rows: Sequence[Mapping[str, object]],
    component_manifest_rows: Sequence[Mapping[str, object]],
    component_coverage_rows: Sequence[Mapping[str, object]],
    paired_generation_rows: Sequence[Mapping[str, object]],
    tail_rows: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
    leakage: Mapping[str, object],
    protocol_violations: Sequence[str],
    target_expert_excluded: bool,
) -> None:
    write_csv_rows(root / "tables" / "labeled_support_policy_downstream_matrix.csv", matrix_rows)
    write_csv_rows(root / "tables" / "labeled_support_policy_summary.csv", [dict(summary)])
    write_csv_rows(root / "tables" / "labeled_support_tail_metric_summary.csv", tail_rows)
    write_csv_rows(root / "tables" / "labeled_support_split_manifest.csv", split_rows)
    write_csv_rows(root / "tables" / "labeled_support_policy_score_matrix.csv", policy_score_rows)
    write_csv_rows(root / "tables" / "labeled_support_policy_selection_manifest.csv", selection_rows)
    write_csv_rows(root / "tables" / "policy_switch_event_table.csv", switch_event_rows)
    write_csv_rows(root / "tables" / "candidate_policy_probability_manifest.csv", probability_manifest_rows)
    write_csv_rows(root / "tables" / "random_bag_manifest.csv", random_bag_manifest_rows)
    write_csv_rows(root / "tables" / "support_to_target_utility_alignment.csv", utility_alignment_rows)
    write_csv_rows(root / "tables" / "support_size_quantization_audit.csv", quantization_rows)
    write_csv_rows(root / "tables" / "support_size_common_eval_audit.csv", common_eval_rows)
    write_csv_rows(root / "tables" / "negative_control_summary.csv", negative_rows)
    write_csv_rows(root / "tables" / "oracle_policy_gap_summary.csv", oracle_rows)
    write_csv_rows(root / "tables" / "labeled_support_target_oracle_audit.csv", target_oracle_rows)
    write_csv_rows(root / "tables" / "eligibility_audit.csv", eligibility_rows)
    write_csv_rows(root / "tables" / "runtime_memory_audit.csv", runtime_rows)
    write_csv_rows(root / "tables" / "component_manifest.csv", component_manifest_rows)
    write_csv_rows(root / "tables" / "component_coverage_audit.csv", component_coverage_rows)
    write_csv_rows(root / "tables" / "paired_generation_audit.csv", paired_generation_rows)
    write_json(root / "reports" / "leakage_report.json", leakage)
    write_json(root / "manifests" / "protocol_manifest.json", _protocol_manifest(cfg, target_expert_excluded, protocol_violations))
    write_json(root / "run_config_resolved.yaml", _resolved_config(cfg))
    _write_decision_summary(root, summary, leakage_status=str(leakage.get("status", "")))


def _labeled_support_leakage_report(
    *,
    protocol_violations: Sequence[str],
    target_expert_excluded: bool,
    support_eval_disjoint: bool,
) -> dict[str, object]:
    violations = list(str(v) for v in protocol_violations)
    if not target_expert_excluded:
        violations.append("target_expert_not_excluded")
    if not support_eval_disjoint:
        violations.append("support_eval_overlap")
    return {
        "schema_version": "cvae_rebuild_labeled_support_tier2_leakage_report_v1",
        "status": "PASS" if not violations else "FAIL",
        "violations": violations,
        "protocol_tier": "tier2_labeled_target_support_calibration",
        "target_support_labels_for_policy_selection": True,
        "target_eval_labels_for_scoring_only": True,
        "support_eval_disjoint": bool(support_eval_disjoint),
        "target_expert_excluded": bool(target_expert_excluded),
        "support_labels_do_not_train_classifiers": True,
        "support_labels_do_not_modify_generation": True,
        "support_labels_do_not_tune_hyperparameters": True,
        "oracle_rows_diagnostic_only": True,
    }


def _protocol_manifest(cfg: LabeledSupportPolicyCalibrationConfig, target_expert_excluded: bool, protocol_violations: Sequence[str]) -> dict[str, object]:
    return {
        "schema_version": "cvae_rebuild_labeled_support_random_vs_dense_policy_calibration_protocol_v1",
        "experiment_name": cfg.name,
        "protocol_tier": "tier2_labeled_target_support_calibration",
        "experiment_type": "few_shot_target_local_utility_calibration",
        "primary_method": cfg.primary_method,
        "primary_variant": cfg.primary_variant,
        "target_support_labels_for_policy_selection": True,
        "target_eval_labels_for_scoring_only": True,
        "support_eval_disjoint": True,
        "class_balanced_support": True,
        "support_labels_do_not_train_classifiers": True,
        "support_labels_do_not_modify_generation": True,
        "support_labels_do_not_tune_hyperparameters": True,
        "support_labels_do_not_choose_support_size": True,
        "support_labels_do_not_change_candidate_policies": True,
        "target_expert_excluded": target_expert_excluded,
        "adoption_eligible_policies": [POLICY_RANDOM_BAG, POLICY_DENSE],
        "diagnostic_policies": [POLICY_SHRINK050, ROW_RANDOM_SINGLE_MASS, cu.ROW_SOURCE_UNION_K16_REFERENCE, cu.ROW_REAL_FEATURE_DENSE_REFERENCE],
        "primary_labeled_support_size": cfg.primary_labeled_support_size,
        "diagnostic_labeled_support_sizes": list(cfg.diagnostic_labeled_support_sizes),
        "primary_switch_quantum": cfg.primary_switch_quantum,
        "support_quantum_by_size": {str(key): value for key, value in cfg.support_quantum_by_size.items()},
        "oracle_rows_diagnostic_only": True,
        "support8_support32_common_eval_diagnostic_only": True,
        "skip_nearest_neighbor_audit": cfg.skip_nearest_neighbor_audit,
        "protocol_violations": list(protocol_violations),
        "protocol_wording": PROTOCOL_WORDING,
    }


def _write_decision_summary(root: Path, decision: Mapping[str, object], *, leakage_status: str) -> None:
    lines = [
        "# Labeled Support16 Random-vs-Dense Policy Calibration v1",
        "",
        "## Primary Verdict",
        "",
        f"- Primary method: `{decision.get('primary_method', '')}`",
        f"- Primary verdict: `{decision.get('primary_verdict', '')}`",
        f"- Diagnostic flags: `{decision.get('diagnostic_flags', '')}`",
        f"- Leakage status: `{leakage_status}`",
        f"- Center-equal mean BACC: {_format_float(decision.get('center_equal_mean_bacc'))}",
        f"- Min center BACC: {_format_float(decision.get('min_center_bacc'))}",
        f"- Center3 BACC: {_format_float(decision.get('center3_bacc'))}",
        f"- Bottom20 mean BACC: {_format_float(decision.get('bottom20_cell_mean_bacc'))}",
        f"- Worst seed-center BACC: {_format_float(decision.get('worst_seed_center_bacc'))}",
        f"- Delta vs random mass-bag: {_format_float(decision.get('delta_vs_random_mass_bag'))}",
        f"- Bottom20 delta vs random mass-bag: {_format_float(decision.get('bottom20_delta_vs_random_mass_bag'))}",
        f"- Within-cell random-vs-dense policy AUC: {_format_float(decision.get('within_cell_pairwise_policy_auc'))}",
        f"- Aggregate Spearman support-vs-target BACC: {_format_float(decision.get('aggregate_spearman_support_bacc_vs_target_bacc'))}",
        f"- Dense switch precision vs target oracle: {_format_float(decision.get('dense_switch_precision_against_target_oracle'))}",
        f"- Dense switch recall vs target oracle: {_format_float(decision.get('dense_switch_recall_against_target_oracle'))}",
        f"- Oracle gap reduction vs random: {_format_float(decision.get('oracle_gap_reduction_vs_random'))}",
        f"- selected_random_mass_bag_rate: {_format_float(decision.get('selected_random_mass_bag_rate'))}",
        f"- selected_dense_reliability_rate: {_format_float(decision.get('selected_dense_reliability_rate'))}",
        "",
        "## Protocol Boundary",
        "",
        PROTOCOL_WORDING,
        "",
        "Support8/support32 and common-eval rows are diagnostic-only and cannot rescue a failed support16 primary.",
        "Oracle rows are diagnostic-only and cannot affect policy selection, thresholds, candidate policies, or adoption.",
        "",
        "## Supported Claim If Successful",
        "",
        "A small class-balanced labeled target-support set can detect when a high-mean random mass-bag composition is unsafe and switch to a dense reliability policy, improving weak-regime robustness without retraining source experts or using target-evaluation labels.",
        "",
    ]
    path = root / "reports" / "decision_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _resolved_config(cfg: LabeledSupportPolicyCalibrationConfig) -> dict[str, object]:
    return {
        "name": cfg.name,
        "artifact_root": str(cfg.artifact_root),
        "experiment_seeds": list(cfg.experiment_seeds),
        "heldout_centers": list(cfg.heldout_centers),
        "support_seeds": list(cfg.support_seeds),
        "primary_labeled_support_size": cfg.primary_labeled_support_size,
        "diagnostic_labeled_support_sizes": list(cfg.diagnostic_labeled_support_sizes),
        "synthetic_per_class_total": cfg.synthetic_per_class_total,
        "random_mass_bag_size": cfg.random_mass_bag_size,
        "primary_switch_quantum": cfg.primary_switch_quantum,
    }


def _key(row: Mapping[str, object]) -> tuple[str, str, str]:
    return (str(row.get("experiment_seed")), str(row.get("heldout_center")), str(row.get("replicate_seed", row.get("support_seed"))))


def _delta(a: object, b: object) -> float:
    left = _float(a)
    right = _float(b)
    return left - right if math.isfinite(left) and math.isfinite(right) else math.nan


def _max_metric(rows: Sequence[Mapping[str, object]], key: str) -> float:
    values = [_float(row.get(key)) for row in rows]
    finite = [value for value in values if math.isfinite(value)]
    return max(finite) if finite else -math.inf


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    pairs = [(float(a), float(b)) for a, b in zip(left, right) if math.isfinite(float(a)) and math.isfinite(float(b))]
    if len(pairs) < 3:
        return math.nan
    try:
        from scipy.stats import spearmanr  # type: ignore
        value, _p = spearmanr([a for a, _b in pairs], [b for _a, b in pairs])
        return float(value)
    except Exception:
        return _pearson(_ranks([a for a, _b in pairs]), _ranks([b for _a, b in pairs]))


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    for rank, idx in enumerate(order):
        ranks[idx] = float(rank)
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return math.nan
    lx = np.asarray(left, dtype=float)
    rx = np.asarray(right, dtype=float)
    if float(np.std(lx)) <= 0.0 or float(np.std(rx)) <= 0.0:
        return math.nan
    return float(np.corrcoef(lx, rx)[0, 1])


def _sample_id(row: Mapping[str, object], idx: int, *, sample_id_key: str = "sample_id") -> str:
    value = row.get(sample_id_key, "")
    return str(value) if str(value) else f"row_{idx}"


def _row_center(row: Mapping[str, object], *, center_key: str) -> str:
    value = row.get(center_key, row.get("center_id", ""))
    return str(value)


def _optional_path(base: Path, value: object) -> Path | None:
    if value is None or str(value) == "":
        return None
    return _path(base, str(value))
