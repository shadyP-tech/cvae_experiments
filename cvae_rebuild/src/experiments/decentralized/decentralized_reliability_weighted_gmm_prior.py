from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from downstream import (
    PredictionBundle,
    evaluate_probability_predictions,
    fit_locked_logistic_classifier,
    geometric_probability_pool,
    weighted_geometric_probability_pool,
)
from features import load_feature_cache, select_rows
from metrics import nanmean
from preservation import _hash_array
from preservation_repair import (
    NA,
    POOL_PER_SOURCE,
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
    _to_numpy,
)
from preservation_sampling import DIAGNOSTIC_SELECTION, PRIMARY_SELECTION, RuntimeSource, _manifest_row, _per_source_variant, _runtime_source
from protocol import ProtocolError, assert_candidate_pool, build_leakage_report
from reporting import prepare_artifact_dirs, write_csv_rows, write_protocol_finalization
from source_union_gmm_prior import _nearest_neighbor_row
from splits import candidate_experts

import decentralized_adaptive_gmm_prior as d1a
import decentralized_k16_gmm_prior as d1


RELIABILITY_NAME = "virchow2_cvae_decentralized_reliability_weighted_gmm_prior_v1"
PRIMARY_RELIABILITY_METHOD = "decentralized_exported_adaptive_k_source_reliability_weighted_geom"
ROW_EQUAL_REFERENCE = "decentralized_exported_adaptive_k_equal_geom_reference"
ROW_POOL_ONLY = "decentralized_exported_adaptive_k_source_reliability_pool_only_geom"
ROW_BUDGET_ONLY = "decentralized_exported_adaptive_k_source_reliability_budget_only_geom"
ROW_TOP3 = "decentralized_exported_adaptive_k_source_reliability_top3_geom_diagnostic"
ROW_SOFTMAX = "decentralized_exported_adaptive_k_source_reliability_softmax_tau1_geom_diagnostic"
ROW_BIC = d1a.ROW_BIC
ROW_SINGLE_MEAN = d1a.ROW_SINGLE_MEAN
ROW_SINGLE_ORACLE = d1a.ROW_SINGLE_ORACLE
ROW_SOURCE_UNION_K16_REFERENCE = d1a.ROW_SOURCE_UNION_K16_REFERENCE
ROW_CENTER_BALANCED_K16_REFERENCE = d1a.ROW_CENTER_BALANCED_K16_REFERENCE
ROW_REAL_FEATURE_DENSE_REFERENCE = d1a.ROW_REAL_FEATURE_DENSE_REFERENCE
ROW_SHUFFLED_SUMMARY_CONTROL = "decentralized_reliability_shuffled_summary_control"
ROW_SHUFFLED_LABEL_CONTROL = "decentralized_reliability_shuffled_label_control"
POOL_DECENTRALIZED = d1a.POOL_DECENTRALIZED
PROTOCOL_WORDING = (
    "This is a data-minimizing, raw-data-free source-local latent summary-exchange protocol. "
    "It is not a formal differential privacy claim. Exported latent summary statistics may still "
    "contain distributional information derived from private data."
)


@dataclass(frozen=True)
class DecentralizedReliabilityWeightedConfig:
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
    primary_pooling: str
    reliability_floor_score: float
    softmax_tau: float
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


@dataclass(frozen=True)
class SourceReliability:
    experiment_seed: int
    replicate_seed: int
    source_center: str
    raw_bacc: float
    macro_f1: float
    reliability_score: float
    reliability_status: str
    error_message: str
    n_eval: int
    generated_features_hash: str
    prediction_hash: str


def load_decentralized_reliability_weighted_gmm_prior_config(path: str | Path) -> DecentralizedReliabilityWeightedConfig:
    source = Path(path).resolve()
    data = _load_mapping(source)
    base_dir = source.parents[2] if len(source.parents) >= 3 else source.parent
    return parse_decentralized_reliability_weighted_gmm_prior_config(data, base_dir=base_dir)


def parse_decentralized_reliability_weighted_gmm_prior_config(
    data: Mapping[str, Any],
    *,
    base_dir: str | Path = ".",
) -> DecentralizedReliabilityWeightedConfig:
    base = Path(base_dir)
    experiment = _mapping(data, "experiment")
    inputs = _mapping(data, "inputs")
    run = _mapping(data, "run_matrix")
    generation = _mapping(data, "generation")
    gmm = _mapping(data, "reliability_weighted_gmm_prior")
    classifier = _mapping(data, "classifier")
    cfg = DecentralizedReliabilityWeightedConfig(
        name=str(experiment["name"]),
        artifact_root=_path(base, str(experiment["artifact_root"])),
        repair_artifact_root=_path(base, str(inputs["repair_artifact_root"])),
        source_union_gmm_artifact_root=_optional_path(base, inputs.get("source_union_gmm_artifact_root")),
        balanced_gmm_artifact_root=_optional_path(base, inputs.get("balanced_gmm_artifact_root")),
        feature_cache_root=_path(base, str(inputs["feature_cache_root"])),
        backbone=str(inputs.get("backbone", "")),
        experiment_seeds=tuple(int(v) for v in run["experiment_seeds"]),
        heldout_centers=tuple(str(v) for v in run["heldout_centers"]),
        replicate_seeds=tuple(int(v) for v in run["replicate_seeds"]),
        synthetic_per_class_total=int(generation["synthetic_per_class_total"]),
        min_per_source_per_class=int(generation["min_per_source_per_class"]),
        primary_variant=str(experiment["primary_variant"]),
        primary_method=str(gmm["primary_method"]),
        candidate_components_per_source_class=tuple(int(v) for v in gmm["candidate_components_per_source_class"]),
        min_samples_per_component=int(gmm["min_samples_per_component"]),
        source_weighting=str(gmm["source_weighting"]),
        gmm_covariance_type=str(gmm["gmm_covariance_type"]),
        gmm_reg_covar=float(gmm["gmm_reg_covar"]),
        gmm_n_init=int(gmm["gmm_n_init"]),
        gmm_max_iter=int(gmm["gmm_max_iter"]),
        min_component_weight=float(gmm["min_component_weight"]),
        variance_floor=float(gmm["variance_floor"]),
        primary_pooling=str(gmm["primary_pooling"]),
        reliability_floor_score=float(gmm["reliability_floor_score"]),
        softmax_tau=float(gmm["softmax_tau"]),
        classifier_type=str(classifier["type"]),
        classifier_solver=str(classifier["solver"]),
        classifier_c=float(classifier["C"]),
        classifier_max_iter=int(classifier["max_iter"]),
        classifier_class_weight=str(classifier["class_weight"]),
        classifier_seed=None if classifier.get("classifier_seed") is None else int(classifier["classifier_seed"]),
    )
    validate_decentralized_reliability_weighted_gmm_prior_config(cfg)
    return cfg


def validate_decentralized_reliability_weighted_gmm_prior_config(cfg: DecentralizedReliabilityWeightedConfig) -> None:
    if cfg.name != RELIABILITY_NAME:
        raise ProtocolError(f"D1.2 experiment name must be {RELIABILITY_NAME!r}.")
    if cfg.backbone != "virchow2":
        raise ProtocolError("D1.2 is locked to backbone=virchow2.")
    if cfg.primary_variant != PRIMARY_VARIANT:
        raise ProtocolError(f"primary_variant must be {PRIMARY_VARIANT!r}.")
    if cfg.primary_method != PRIMARY_RELIABILITY_METHOD:
        raise ProtocolError(f"primary_method must be {PRIMARY_RELIABILITY_METHOD!r}.")
    if cfg.candidate_components_per_source_class != (4, 3, 2, 1):
        raise ProtocolError("candidate_components_per_source_class must be locked to [4, 3, 2, 1].")
    if len(cfg.heldout_centers) != 5:
        raise ProtocolError("D1.2 composition expects exactly five centers, giving four source experts per fold.")
    if cfg.source_weighting != "source_local_reliability":
        raise ProtocolError("source_weighting must be source_local_reliability.")
    if cfg.gmm_covariance_type != "diag":
        raise ProtocolError("gmm_covariance_type must be diag.")
    if cfg.primary_pooling != "weighted_geometric":
        raise ProtocolError("primary_pooling must be weighted_geometric.")
    if cfg.synthetic_per_class_total != 128:
        raise ProtocolError("synthetic_per_class_total must be locked to 128.")
    if cfg.min_per_source_per_class != 8:
        raise ProtocolError("min_per_source_per_class must be locked to 8.")
    if min(cfg.min_samples_per_component, cfg.gmm_n_init, cfg.gmm_max_iter) < 1:
        raise ProtocolError("GMM counts and iteration settings must be positive.")
    if min(cfg.gmm_reg_covar, cfg.min_component_weight, cfg.variance_floor, cfg.reliability_floor_score, cfg.softmax_tau) <= 0.0:
        raise ProtocolError("GMM/reliability floors and softmax_tau must be positive.")
    if cfg.classifier_type != "sklearn_logistic_regression":
        raise ProtocolError("classifier.type must be sklearn_logistic_regression.")
    if cfg.classifier_solver != "lbfgs" or cfg.classifier_c != 1.0 or cfg.classifier_max_iter != 2000:
        raise ProtocolError("Classifier solver/C/max_iter must remain locked.")
    if cfg.classifier_class_weight != "balanced" or cfg.classifier_seed is not None:
        raise ProtocolError("Classifier must use class_weight=balanced and classifier_seed=null.")


def run_decentralized_reliability_weighted_gmm_prior(
    cfg: DecentralizedReliabilityWeightedConfig,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    root = prepare_artifact_dirs(Path(artifact_root) if artifact_root is not None else cfg.artifact_root)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "summaries").mkdir(parents=True, exist_ok=True)

    matrix_rows: list[dict[str, object]] = []
    gap_rows: list[dict[str, object]] = []
    summary_manifest_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    reliability_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    rank_rows: list[dict[str, object]] = []
    centerwise_rows: list[dict[str, object]] = []
    late_rows: list[dict[str, object]] = []
    real_feature_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
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
    d1._validate_optional_leakage_report(cfg.source_union_gmm_artifact_root, protocol_violations)
    d1._validate_optional_leakage_report(cfg.balanced_gmm_artifact_root, protocol_violations)

    repair_cfg = d1._repair_runtime_config(cfg, root)
    per_source_variant = _per_source_variant()

    try:
        for experiment_seed in cfg.experiment_seeds:
            train_cache = load_feature_cache(_existing_cache_path(cfg.feature_cache_root, seed=experiment_seed, split="train"))
            test_cache = load_feature_cache(_existing_cache_path(cfg.feature_cache_root, seed=experiment_seed, split="test"))
            per_source_runtime: dict[str, RuntimeSource] = {}
            largest_summaries: dict[tuple[str, int], d1a.AdaptiveSourceLocalSummary] = {}
            bic_summaries: dict[tuple[str, int], d1a.AdaptiveSourceLocalSummary] = {}
            shuffled_summaries: dict[tuple[str, int], d1a.AdaptiveSourceLocalSummary] = {}
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

                largest, bic = d1a._fit_and_export_source_summaries(
                    cfg,
                    root,
                    runtime_source.runtime,
                    experiment_seed=int(experiment_seed),
                    shuffled_label_control=False,
                )
                shuffled, _ = d1a._fit_and_export_source_summaries(
                    cfg,
                    root,
                    runtime_source.runtime,
                    experiment_seed=int(experiment_seed),
                    shuffled_label_control=True,
                )
                for summary in largest:
                    largest_summaries[(summary.source_center, summary.class_label)] = summary
                    summary_manifest_rows.append(d1a._summary_manifest_row(summary))
                    diagnostic_rows.append(d1a._summary_diagnostic_row(cfg, summary))
                for summary in bic:
                    bic_summaries[(summary.source_center, summary.class_label)] = summary
                    diagnostic_rows.append(d1a._summary_diagnostic_row(cfg, summary))
                for summary in shuffled:
                    shuffled_summaries[(summary.source_center, summary.class_label)] = summary
                    diagnostic_rows.append(d1a._summary_diagnostic_row(cfg, summary))

            reliability: dict[tuple[int, int, str], SourceReliability] = {}
            for replicate_seed in cfg.replicate_seeds:
                for source_center in cfg.heldout_centers:
                    rel = _source_local_reliability(
                        cfg,
                        per_source_runtime=per_source_runtime,
                        summaries=largest_summaries,
                        test_cache=test_cache,
                        experiment_seed=int(experiment_seed),
                        replicate_seed=int(replicate_seed),
                        source_center=str(source_center),
                    )
                    reliability[(int(experiment_seed), int(replicate_seed), str(source_center))] = rel
                    reliability_rows.append(_source_reliability_row(rel))

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

                for replicate_seed in cfg.replicate_seeds:
                    su_ref = d1._reference_for_cell(source_union_refs, experiment_seed, heldout_center, replicate_seed)
                    cb_ref = d1._reference_for_cell(center_balanced_refs, experiment_seed, heldout_center, replicate_seed)
                    rels = {source: reliability[(int(experiment_seed), int(replicate_seed), str(source))] for source in candidates}
                    primary_plan = _weight_plan(cfg, candidates, rels, mode="linear")
                    weight_rows.extend(_weight_manifest_rows(experiment_seed, replicate_seed, heldout_center, primary_plan, rels))

                    if eval_error:
                        matrix_rows.extend(
                            _ineligible_rows(
                                cfg,
                                experiment_seed=int(experiment_seed),
                                heldout_center=str(heldout_center),
                                replicate_seed=int(replicate_seed),
                                candidates=candidates,
                                source_union_ref=su_ref,
                                center_balanced_ref=cb_ref,
                                status="ineligible",
                                error_message=eval_error,
                            )
                        )
                        continue

                    ref_row, real_late = d1a._real_feature_reference(
                        cfg,
                        per_source_runtime=per_source_runtime,
                        candidates=candidates,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        replicate_seed=int(replicate_seed),
                        eval_raw=eval_raw,
                        eval_labels=eval_labels,
                    )
                    real_feature_rows.append(_extend_row(ref_row))
                    matrix_rows.append(_extend_row(ref_row))
                    late_rows.extend(_extend_rows(real_late))

                    equal_plan = _uniform_weight_plan(cfg, candidates, rels)
                    equal_rows, equal_late, coverage, weak, nn = _evaluate_weighted_variant(
                        cfg,
                        per_source_runtime=per_source_runtime,
                        candidates=candidates,
                        summaries=largest_summaries,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        replicate_seed=int(replicate_seed),
                        eval_raw=eval_raw,
                        eval_labels=eval_labels,
                        source_union_ref=su_ref,
                        center_balanced_ref=cb_ref,
                        real_feature_bacc=_float(ref_row["bacc"]),
                        weight_plan=equal_plan,
                        prior_method=ROW_EQUAL_REFERENCE,
                        pooling_rule="geometric",
                        selection_source=DIAGNOSTIC_SELECTION,
                        claim_role="d1_1_equal_adaptive_geom_reference",
                    )
                    matrix_rows.extend(equal_rows)
                    late_rows.extend(equal_late)
                    coverage_rows.extend(coverage)
                    weak_rows.extend(weak)
                    nn_rows.extend(nn)
                    rank_rows.extend(_rank_diagnostic_rows(experiment_seed, replicate_seed, heldout_center, equal_late, rels))

                    primary_rows, primary_late, coverage, weak, nn = _evaluate_weighted_variant(
                        cfg,
                        per_source_runtime=per_source_runtime,
                        candidates=candidates,
                        summaries=largest_summaries,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        replicate_seed=int(replicate_seed),
                        eval_raw=eval_raw,
                        eval_labels=eval_labels,
                        source_union_ref=su_ref,
                        center_balanced_ref=cb_ref,
                        real_feature_bacc=_float(ref_row["bacc"]),
                        weight_plan=primary_plan,
                        prior_method=PRIMARY_RELIABILITY_METHOD,
                        pooling_rule="weighted_geometric",
                        selection_source=PRIMARY_SELECTION,
                        claim_role="primary_reliability_weighted_composition",
                    )
                    matrix_rows.extend(primary_rows)
                    late_rows.extend(primary_late)
                    coverage_rows.extend(coverage)
                    weak_rows.extend(weak)
                    nn_rows.extend(nn)

                    _append_single_source_references(
                        cfg,
                        matrix_rows,
                        primary_late,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        replicate_seed=int(replicate_seed),
                        candidates=candidates,
                        summaries=largest_summaries,
                        source_union_ref=su_ref,
                        center_balanced_ref=cb_ref,
                        real_feature_bacc=_float(ref_row["bacc"]),
                    )

                    for method, plan, pooling_rule, role in (
                        (ROW_POOL_ONLY, _pool_only_plan(equal_plan, primary_plan), "weighted_geometric", "diagnostic_reliability_pool_only"),
                        (ROW_BUDGET_ONLY, _budget_only_plan(equal_plan, primary_plan), "geometric", "diagnostic_reliability_budget_only"),
                        (ROW_TOP3, _topk_plan(cfg, primary_plan, rels, k=3), "weighted_geometric", "diagnostic_reliability_top3"),
                        (ROW_SOFTMAX, _weight_plan(cfg, candidates, rels, mode="softmax"), "weighted_geometric", "diagnostic_reliability_softmax_tau1"),
                    ):
                        rows, late, coverage, weak, nn = _evaluate_weighted_variant(
                            cfg,
                            per_source_runtime=per_source_runtime,
                            candidates=tuple(plan["sources"]),
                            summaries=largest_summaries,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            replicate_seed=int(replicate_seed),
                            eval_raw=eval_raw,
                            eval_labels=eval_labels,
                            source_union_ref=su_ref,
                            center_balanced_ref=cb_ref,
                            real_feature_bacc=_float(ref_row["bacc"]),
                            weight_plan=plan,
                            prior_method=method,
                            pooling_rule=pooling_rule,
                            selection_source=DIAGNOSTIC_SELECTION,
                            claim_role=role,
                        )
                        matrix_rows.extend(rows)
                        late_rows.extend(late)
                        coverage_rows.extend(coverage)
                        weak_rows.extend(weak)
                        nn_rows.extend(nn)

                    rows, late, coverage, weak, nn = d1a._evaluate_bic_diagnostic(
                        cfg,
                        per_source_runtime=per_source_runtime,
                        candidates=candidates,
                        summaries=bic_summaries,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        replicate_seed=int(replicate_seed),
                        eval_raw=eval_raw,
                        eval_labels=eval_labels,
                        source_union_ref=su_ref,
                        center_balanced_ref=cb_ref,
                        real_feature_bacc=_float(ref_row["bacc"]),
                    )
                    matrix_rows.extend(_rename_rows(rows, d1a.ROW_BIC, ROW_BIC))
                    late_rows.extend(_extend_rows(late))
                    coverage_rows.extend(coverage)
                    weak_rows.extend(weak)
                    nn_rows.extend(nn)

                    for prior_method, summaries, control_mode in (
                        (ROW_SHUFFLED_SUMMARY_CONTROL, largest_summaries, "class_flip"),
                        (ROW_SHUFFLED_LABEL_CONTROL, shuffled_summaries, "normal"),
                    ):
                        rows, late, coverage, weak, nn = _evaluate_weighted_variant(
                            cfg,
                            per_source_runtime=per_source_runtime,
                            candidates=candidates,
                            summaries=summaries,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            replicate_seed=int(replicate_seed),
                            eval_raw=eval_raw,
                            eval_labels=eval_labels,
                            source_union_ref=su_ref,
                            center_balanced_ref=cb_ref,
                            real_feature_bacc=_float(ref_row["bacc"]),
                            weight_plan=primary_plan,
                            prior_method=prior_method,
                            pooling_rule="weighted_geometric",
                            selection_source=DIAGNOSTIC_SELECTION,
                            claim_role="negative_control",
                            control_mode=control_mode,
                        )
                        matrix_rows.extend(rows)
                        late_rows.extend(late)
                        coverage_rows.extend(coverage)
                        weak_rows.extend(weak)
                        nn_rows.extend(nn)

                    matrix_rows.append(_extend_row(d1a._reference_matrix_row(
                        cfg,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        replicate_seed=int(replicate_seed),
                        candidates=candidates,
                        prior_method=ROW_SOURCE_UNION_K16_REFERENCE,
                        reference=su_ref,
                    )))
                    matrix_rows.append(_extend_row(d1a._reference_matrix_row(
                        cfg,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        replicate_seed=int(replicate_seed),
                        candidates=candidates,
                        prior_method=ROW_CENTER_BALANCED_K16_REFERENCE,
                        reference=cb_ref,
                    )))
    except ProtocolError as exc:
        protocol_violations.append(str(exc))

    leakage = build_leakage_report(
        target_support_labels_for_selection=False,
        target_eval_labels_for_scoring_only=True,
        target_expert_excluded=target_expert_excluded,
        oracle_rows_diagnostic_only=True,
        extra_violations=protocol_violations,
    )
    _populate_negative_control_gaps(matrix_rows)
    gap_rows = [dict(row) for row in matrix_rows if row.get("status") == "ok"]
    centerwise_rows = _centerwise_delta_rows(matrix_rows)
    decision = _decision(
        matrix_rows,
        cfg,
        leakage_status=leakage.status,
        diagnostic_rows=diagnostic_rows,
        reliability_rows=reliability_rows,
        weight_rows=weight_rows,
        rank_rows=rank_rows,
        centerwise_rows=centerwise_rows,
    )
    _write_artifacts(
        root,
        cfg,
        matrix_rows=matrix_rows,
        gap_rows=gap_rows,
        summary_manifest_rows=summary_manifest_rows,
        diagnostic_rows=diagnostic_rows,
        reliability_rows=reliability_rows,
        weight_rows=weight_rows,
        rank_rows=rank_rows,
        centerwise_rows=centerwise_rows,
        late_rows=late_rows,
        real_feature_rows=real_feature_rows,
        coverage_rows=coverage_rows,
        weak_rows=weak_rows,
        nn_rows=nn_rows,
        model_manifest_rows=model_manifest_rows,
        decision=decision,
        leakage_status=leakage.status,
        protocol_violations=protocol_violations,
        target_expert_excluded=target_expert_excluded,
    )
    return root


def _optional_path(base: Path, value: object) -> Path | None:
    if value is None or str(value) == "":
        return None
    return _path(base, str(value))


def _source_local_reliability(
    cfg: DecentralizedReliabilityWeightedConfig,
    *,
    per_source_runtime: Mapping[str, RuntimeSource],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    test_cache: object,
    experiment_seed: int,
    replicate_seed: int,
    source_center: str,
) -> SourceReliability:
    target_indices = _target_indices(test_cache.metadata, str(source_center))
    eval_raw, eval_meta = select_rows(test_cache.embeddings, test_cache.metadata, target_indices)
    eval_labels = tuple(_label(row) for row in eval_meta)
    if len(set(eval_labels)) < 2:
        return _neutral_reliability(experiment_seed, replicate_seed, source_center, len(eval_labels), "mono_class_source_reliability_eval")
    status, error = d1a._composition_status((str(source_center),), summaries, control_mode="normal")
    if status != "ok":
        return _neutral_reliability(experiment_seed, replicate_seed, source_center, len(eval_labels), error)
    runtime = per_source_runtime[str(source_center)].runtime
    budget = cfg.synthetic_per_class_total // (len(cfg.heldout_centers) - 1)
    seed = d1._latent_seed(experiment_seed, replicate_seed, source_center, "source_local_reliability")
    generated, labels, _counts = d1a._sample_source_from_summaries(
        cfg,
        runtime,
        summaries,
        source_center=str(source_center),
        budget_per_class=int(budget),
        seed=seed,
        control_mode="normal",
    )
    eval_x = runtime.frame.transform(_to_numpy(eval_raw))
    bundle = fit_locked_logistic_classifier(
        generated,
        labels,
        eval_x,
        classifier_seed=cfg.classifier_seed,
        expert_id=str(source_center),
        class_weight=cfg.classifier_class_weight,
    )
    result = evaluate_probability_predictions("source_local_reliability", bundle.probabilities, eval_labels)
    return SourceReliability(
        experiment_seed=int(experiment_seed),
        replicate_seed=int(replicate_seed),
        source_center=str(source_center),
        raw_bacc=float(result.bacc),
        macro_f1=float(result.macro_f1),
        reliability_score=_linear_reliability_score(float(result.bacc), cfg.reliability_floor_score),
        reliability_status="ok",
        error_message="",
        n_eval=len(eval_labels),
        generated_features_hash=_hash_array(generated),
        prediction_hash=_hash_array(bundle.probabilities),
    )


def _neutral_reliability(experiment_seed: int, replicate_seed: int, source_center: str, n_eval: int, error_message: str) -> SourceReliability:
    return SourceReliability(
        experiment_seed=int(experiment_seed),
        replicate_seed=int(replicate_seed),
        source_center=str(source_center),
        raw_bacc=0.5,
        macro_f1=math.nan,
        reliability_score=0.05,
        reliability_status="neutral_fallback",
        error_message=str(error_message),
        n_eval=int(n_eval),
        generated_features_hash="",
        prediction_hash="",
    )


def _linear_reliability_score(raw_bacc: float, floor: float) -> float:
    clipped = min(max(float(raw_bacc), 0.5), 1.0)
    return max((clipped - 0.5) / 0.5, float(floor))


def _source_reliability_row(rel: SourceReliability) -> dict[str, object]:
    return {
        "experiment_seed": rel.experiment_seed,
        "replicate_seed": rel.replicate_seed,
        "source_center": rel.source_center,
        "raw_reliability_bacc": rel.raw_bacc,
        "source_reliability_macro_f1": rel.macro_f1,
        "reliability_score": rel.reliability_score,
        "reliability_status": rel.reliability_status,
        "error_message": rel.error_message,
        "n_source_eval": rel.n_eval,
        "generated_features_hash": rel.generated_features_hash,
        "prediction_hash": rel.prediction_hash,
    }


def _weight_plan(
    cfg: DecentralizedReliabilityWeightedConfig,
    sources: Sequence[str],
    rels: Mapping[str, SourceReliability],
    *,
    mode: str,
) -> dict[str, object]:
    sources_tuple = tuple(str(source) for source in sources)
    if mode == "softmax":
        raw = np.asarray([rels[source].raw_bacc for source in sources_tuple], dtype=float)
        shifted = (raw - float(np.max(raw))) / float(cfg.softmax_tau)
        values = np.exp(shifted)
        weights = values / float(values.sum())
        scores = {source: float(value) for source, value in zip(sources_tuple, values)}
    else:
        scores = {source: _linear_reliability_score(rels[source].raw_bacc, cfg.reliability_floor_score) for source in sources_tuple}
        total = sum(scores.values())
        weights = np.asarray([scores[source] / total for source in sources_tuple], dtype=float)
    weights_dict = {source: float(weight) for source, weight in zip(sources_tuple, weights)}
    budgets = _weighted_budgets(cfg.synthetic_per_class_total, sources_tuple, weights_dict, cfg.min_per_source_per_class)
    return _with_weight_diagnostics(sources_tuple, weights_dict, budgets, scores)


def _uniform_weight_plan(
    cfg: DecentralizedReliabilityWeightedConfig,
    sources: Sequence[str],
    rels: Mapping[str, SourceReliability],
) -> dict[str, object]:
    sources_tuple = tuple(str(source) for source in sources)
    weight = 1.0 / float(len(sources_tuple))
    scores = {source: rels[source].reliability_score for source in sources_tuple}
    budgets = {source: int(value) for source, value in zip(sources_tuple, d1._balanced_counts(cfg.synthetic_per_class_total, len(sources_tuple)))}
    return _with_weight_diagnostics(sources_tuple, {source: weight for source in sources_tuple}, budgets, scores)


def _pool_only_plan(equal_plan: Mapping[str, object], primary_plan: Mapping[str, object]) -> dict[str, object]:
    return _with_weight_diagnostics(
        tuple(str(v) for v in primary_plan["sources"]),
        dict(primary_plan["weights"]),
        dict(equal_plan["budgets"]),
        dict(primary_plan["scores"]),
    )


def _budget_only_plan(equal_plan: Mapping[str, object], primary_plan: Mapping[str, object]) -> dict[str, object]:
    return _with_weight_diagnostics(
        tuple(str(v) for v in primary_plan["sources"]),
        dict(equal_plan["weights"]),
        dict(primary_plan["budgets"]),
        dict(primary_plan["scores"]),
    )


def _topk_plan(
    cfg: DecentralizedReliabilityWeightedConfig,
    primary_plan: Mapping[str, object],
    rels: Mapping[str, SourceReliability],
    *,
    k: int,
) -> dict[str, object]:
    ordered = sorted(tuple(str(v) for v in primary_plan["sources"]), key=lambda source: (-float(primary_plan["weights"][source]), source))
    sources = tuple(ordered[: int(k)])
    return _weight_plan(cfg, sources, {source: rels[source] for source in sources}, mode="linear")


def _with_weight_diagnostics(
    sources: Sequence[str],
    weights: Mapping[str, float],
    budgets: Mapping[str, int],
    scores: Mapping[str, float],
) -> dict[str, object]:
    weights_values = [float(weights[source]) for source in sources]
    entropy = -sum(value * math.log(value) for value in weights_values if value > 0.0)
    uniform = 1.0 / float(len(sources))
    dominant = max(sources, key=lambda source: (float(weights[source]), str(source)))
    return {
        "sources": tuple(str(source) for source in sources),
        "weights": {str(source): float(weights[source]) for source in sources},
        "budgets": {str(source): int(budgets[source]) for source in sources},
        "scores": {str(source): float(scores[source]) for source in sources},
        "weight_entropy": float(entropy),
        "effective_num_sources": float(math.exp(entropy)),
        "l1_distance_from_uniform": float(sum(abs(float(weights[source]) - uniform) for source in sources)),
        "max_weight": float(max(weights_values)),
        "min_weight": float(min(weights_values)),
        "dominant_source": str(dominant),
        "dominant_source_weight": float(weights[dominant]),
    }


def _weighted_budgets(total: int, sources: Sequence[str], weights: Mapping[str, float], minimum: int) -> dict[str, int]:
    if len(sources) * int(minimum) > int(total):
        raise ProtocolError("Minimum per-source budget exceeds total synthetic budget.")
    remaining = int(total) - len(sources) * int(minimum)
    exact = {source: float(weights[source]) * remaining for source in sources}
    budgets = {source: int(minimum) + int(math.floor(exact[source])) for source in sources}
    leftover = int(total) - sum(budgets.values())
    ordered = sorted(sources, key=lambda source: (-(exact[source] - math.floor(exact[source])), str(source)))
    for source in ordered[:leftover]:
        budgets[source] += 1
    if sum(budgets.values()) != int(total):
        raise ProtocolError("Weighted synthetic budget allocation failed to sum to total.")
    return budgets


def _evaluate_weighted_variant(
    cfg: DecentralizedReliabilityWeightedConfig,
    *,
    per_source_runtime: Mapping[str, RuntimeSource],
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    eval_raw: object,
    eval_labels: Sequence[int],
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
    real_feature_bacc: float,
    weight_plan: Mapping[str, object],
    prior_method: str,
    pooling_rule: str,
    selection_source: str,
    claim_role: str,
    control_mode: str = "normal",
    generation_seed_method: str | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    sources = tuple(str(source) for source in candidates)
    status, error = d1a._composition_status(sources, summaries, control_mode=control_mode)
    if status != "ok":
        row = d1a._dense_empty_row(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            candidates=sources,
            summaries=summaries,
            prior_method=prior_method,
            source_union_ref=source_union_ref,
            center_balanced_ref=center_balanced_ref,
            real_feature_bacc=real_feature_bacc,
            status=status,
            error_message=error,
            claim_role=claim_role,
        )
        return [_extend_row(row, weight_plan=weight_plan, source_weighting=cfg.source_weighting)], [], [], [], []
    bundles, single_rows, coverage_rows, weak_rows, nn_rows, generated_hash = _source_generated_bundles(
        cfg,
        per_source_runtime=per_source_runtime,
        candidates=sources,
        summaries=summaries,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        eval_raw=eval_raw,
        eval_labels=eval_labels,
        prior_method=prior_method,
        control_mode=control_mode,
        budgets=dict(weight_plan["budgets"]),
        weight_plan=weight_plan,
        generation_seed_method=generation_seed_method,
    )
    weights = [float(weight_plan["weights"][str(bundle.expert_id)]) for bundle in bundles]
    if pooling_rule == "weighted_geometric":
        pooled = weighted_geometric_probability_pool(bundles, weights)
    elif pooling_rule == "geometric":
        pooled = geometric_probability_pool(bundles)
    else:
        raise ProtocolError(f"Unsupported D1.2 pooling rule: {pooling_rule}")
    single_baccs = [_float(row["bacc"]) for row in single_rows if row.get("status") == "ok"]
    single_macro = [_float(row["macro_f1"]) for row in single_rows if row.get("status") == "ok"]
    row = d1a._dense_result_row(
        cfg,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        candidates=sources,
        summaries=summaries,
        prior_method=prior_method,
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
    row = _extend_row(row, weight_plan=weight_plan, source_weighting=_source_weighting_for_method(prior_method))
    return [row], single_rows, coverage_rows, weak_rows, nn_rows


def _source_generated_bundles(
    cfg: DecentralizedReliabilityWeightedConfig,
    *,
    per_source_runtime: Mapping[str, RuntimeSource],
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    eval_raw: object,
    eval_labels: Sequence[int],
    prior_method: str,
    control_mode: str,
    budgets: Mapping[str, int],
    weight_plan: Mapping[str, object],
    generation_seed_method: str | None = None,
) -> tuple[list[PredictionBundle], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], str]:
    bundles: list[PredictionBundle] = []
    late_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    weak_rows: list[dict[str, object]] = []
    nn_rows: list[dict[str, object]] = []
    generated_hashes: list[str] = []
    for source_center in candidates:
        runtime = per_source_runtime[str(source_center)].runtime
        budget_per_class = int(budgets[str(source_center)])
        if generation_seed_method is None:
            latent_seed = d1._latent_seed(experiment_seed, heldout_center, replicate_seed, prior_method, source_center, control_mode)
        else:
            latent_seed = d1._latent_seed(
                experiment_seed,
                heldout_center,
                replicate_seed,
                generation_seed_method,
                source_center,
                budget_per_class,
                control_mode,
            )
        generated, labels, counts = d1a._sample_source_from_summaries(
            cfg,
            runtime,
            summaries,
            source_center=str(source_center),
            budget_per_class=budget_per_class,
            seed=latent_seed,
            control_mode=control_mode,
        )
        eval_x = runtime.frame.transform(_to_numpy(eval_raw))
        bundle = fit_locked_logistic_classifier(
            generated,
            labels,
            eval_x,
            classifier_seed=cfg.classifier_seed,
            expert_id=str(source_center),
            class_weight=cfg.classifier_class_weight,
        )
        result = evaluate_probability_predictions(prior_method, bundle.probabilities, eval_labels)
        generated_hash = _hash_array(generated)
        prediction_hash = _hash_array(bundle.probabilities)
        generated_hashes.append(generated_hash)
        row = d1a._base_matrix_row(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            candidates=candidates,
            summaries=summaries,
            expert_id=str(source_center),
            expert_pool_type=POOL_PER_SOURCE,
            prior_method=prior_method,
            pooling_rule="single_source",
            source_union_ref=d1._missing_reference(),
            center_balanced_ref=d1._missing_reference(),
            control_mode=control_mode,
        )
        row.update(
            {
                "source_weighting": _source_weighting_for_method(prior_method),
                "synthetic_per_class_total": budget_per_class,
                "synthetic_per_class_per_source_json": json.dumps({str(source_center): budget_per_class}, sort_keys=True),
                "bacc": result.bacc,
                "macro_f1": result.macro_f1,
                "generated_features_hash": generated_hash,
                "prediction_hash": prediction_hash,
                "selection_source": DIAGNOSTIC_SELECTION,
                "status": "ok",
                "claim_role": "single_source_component_for_dense_aggregation",
            }
        )
        row = _extend_row(row, weight_plan=weight_plan, source_weighting=_source_weighting_for_method(prior_method))
        late_rows.append(row)
        if _float(row["bacc"]) < 0.75:
            weak_rows.append(d1a._weak_row(row))
        coverage_rows.append(d1a._coverage_row(row, counts, candidates=candidates, summaries=summaries, control_mode=control_mode))
        nn_rows.append(_nearest_neighbor_row(row, generated, runtime.source_train_embeddings))
        bundles.append(bundle)
    aggregate_hash = _hash_strings(generated_hashes)
    return bundles, late_rows, coverage_rows, weak_rows, nn_rows, aggregate_hash


def _source_weighting_for_method(method: str) -> str:
    if method == ROW_EQUAL_REFERENCE:
        return "equal_source_mass"
    if method == ROW_BUDGET_ONLY:
        return "source_local_reliability_budget_only"
    if method == ROW_POOL_ONLY:
        return "source_local_reliability_pool_only"
    if method == ROW_TOP3:
        return "source_local_reliability_top3"
    if method == ROW_SOFTMAX:
        return "source_local_reliability_softmax_tau1"
    return "source_local_reliability"


def _append_single_source_references(
    cfg: DecentralizedReliabilityWeightedConfig,
    matrix_rows: list[dict[str, object]],
    single_rows: Sequence[Mapping[str, object]],
    *,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
    real_feature_bacc: float,
) -> None:
    baccs = [_float(row["bacc"]) for row in single_rows if row.get("status") == "ok"]
    macros = [_float(row["macro_f1"]) for row in single_rows if row.get("status") == "ok"]
    mean_single = nanmean(baccs)
    oracle_single = max(baccs) if baccs else math.nan
    for method, bacc, macro_f1, role in (
        (ROW_SINGLE_MEAN, mean_single, nanmean(macros), "single_source_mean_reference"),
        (ROW_SINGLE_ORACLE, oracle_single, max(macros) if macros else math.nan, "diagnostic_only_oracle_reference"),
    ):
        row = d1a._aggregate_reference_row(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            candidates=candidates,
            summaries=summaries,
            prior_method=method,
            bacc=bacc,
            macro_f1=macro_f1,
            source_union_ref=source_union_ref,
            center_balanced_ref=center_balanced_ref,
            real_feature_bacc=real_feature_bacc,
            mean_single_bacc=mean_single,
            oracle_single_bacc=oracle_single,
            claim_role=role,
        )
        matrix_rows.append(_extend_row(row))


def _rank_diagnostic_rows(
    experiment_seed: int,
    replicate_seed: int,
    heldout_center: str,
    equal_late_rows: Sequence[Mapping[str, object]],
    rels: Mapping[str, SourceReliability],
) -> list[dict[str, object]]:
    source_rows = [row for row in equal_late_rows if row.get("pooling_rule") == "single_source" and row.get("status") == "ok"]
    utilities = {str(row["expert_id"]): _float(row.get("bacc")) for row in source_rows}
    sources = tuple(sorted(utilities))
    reliability_values = {source: rels[source].raw_bacc for source in sources}
    utility_values = {source: utilities[source] for source in sources}
    rel_ranks = _ranks(reliability_values)
    util_ranks = _ranks(utility_values)
    spearman = _spearman(reliability_values, utility_values)
    return [
        {
            "experiment_seed": int(experiment_seed),
            "replicate_seed": int(replicate_seed),
            "heldout_center": str(heldout_center),
            "source_center": source,
            "source_reliability_bacc": reliability_values[source],
            "target_single_source_bacc": utility_values[source],
            "source_reliability_rank": rel_ranks[source],
            "target_utility_rank": util_ranks[source],
            "cell_spearman": spearman,
        }
        for source in sources
    ]


def _ranks(values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (-float(item[1]), item[0]))
    ranks: dict[str, float] = {}
    idx = 0
    while idx < len(ordered):
        end = idx + 1
        while end < len(ordered) and math.isclose(float(ordered[end][1]), float(ordered[idx][1])):
            end += 1
        rank = (idx + 1 + end) / 2.0
        for pos in range(idx, end):
            ranks[ordered[pos][0]] = rank
        idx = end
    return ranks


def _spearman(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = [key for key in left if key in right and math.isfinite(float(left[key])) and math.isfinite(float(right[key]))]
    if len(keys) < 2:
        return math.nan
    left_rank = _ranks({key: float(left[key]) for key in keys})
    right_rank = _ranks({key: float(right[key]) for key in keys})
    xs = [left_rank[key] for key in keys]
    ys = [right_rank[key] for key in keys]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0.0 or den_y == 0.0:
        return math.nan
    return float(num / (den_x * den_y))


def _weight_manifest_rows(
    experiment_seed: int,
    replicate_seed: int,
    heldout_center: str,
    plan: Mapping[str, object],
    rels: Mapping[str, SourceReliability],
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
                "source_center": source_id,
                "raw_reliability_bacc": rel.raw_bacc,
                "reliability_score": plan["scores"][source_id],
                "normalized_reliability_weight": plan["weights"][source_id],
                "synthetic_per_class_budget": plan["budgets"][source_id],
                "reliability_status": rel.reliability_status,
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


def _extend_row(
    row: Mapping[str, object],
    *,
    weight_plan: Mapping[str, object] | None = None,
    source_weighting: str | None = None,
) -> dict[str, object]:
    out = dict(row)
    if source_weighting is not None:
        out["source_weighting"] = source_weighting
    if weight_plan is None:
        out.update(
            {
                "reliability_weight_json": "{}",
                "reliability_score_json": "{}",
                "reliability_budget_per_class_json": "{}",
                "reliability_weight_entropy": math.nan,
                "effective_num_sources": math.nan,
                "l1_distance_from_uniform": math.nan,
                "max_reliability_weight": math.nan,
                "min_reliability_weight": math.nan,
                "dominant_source": "",
                "dominant_source_weight": math.nan,
                "delta_vs_d1_1_equal_adaptive_geom": math.nan,
            }
        )
        return out
    out.update(
        {
            "synthetic_per_class_per_source_json": json.dumps(dict(weight_plan["budgets"]), sort_keys=True),
            "reliability_weight_json": json.dumps(dict(weight_plan["weights"]), sort_keys=True),
            "reliability_score_json": json.dumps(dict(weight_plan["scores"]), sort_keys=True),
            "reliability_budget_per_class_json": json.dumps(dict(weight_plan["budgets"]), sort_keys=True),
            "reliability_weight_entropy": weight_plan["weight_entropy"],
            "effective_num_sources": weight_plan["effective_num_sources"],
            "l1_distance_from_uniform": weight_plan["l1_distance_from_uniform"],
            "max_reliability_weight": weight_plan["max_weight"],
            "min_reliability_weight": weight_plan["min_weight"],
            "dominant_source": weight_plan["dominant_source"],
            "dominant_source_weight": weight_plan["dominant_source_weight"],
            "delta_vs_d1_1_equal_adaptive_geom": math.nan,
        }
    )
    return out


def _extend_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [_extend_row(row) for row in rows]


def _rename_rows(rows: Sequence[Mapping[str, object]], old: str, new: str) -> list[dict[str, object]]:
    out = []
    for row in rows:
        copied = _extend_row(row)
        if copied.get("prior_method") == old:
            copied["prior_method"] = new
        out.append(copied)
    return out


def _ineligible_rows(
    cfg: DecentralizedReliabilityWeightedConfig,
    *,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
    status: str,
    error_message: str,
) -> list[dict[str, object]]:
    empty: dict[tuple[str, int], d1a.AdaptiveSourceLocalSummary] = {}
    rows = []
    for method, role in (
        (PRIMARY_RELIABILITY_METHOD, "primary_reliability_weighted_composition"),
        (ROW_EQUAL_REFERENCE, "d1_1_equal_adaptive_geom_reference"),
        (ROW_POOL_ONLY, "diagnostic_reliability_pool_only"),
        (ROW_BUDGET_ONLY, "diagnostic_reliability_budget_only"),
        (ROW_TOP3, "diagnostic_reliability_top3"),
        (ROW_SOFTMAX, "diagnostic_reliability_softmax_tau1"),
        (ROW_BIC, "diagnostic_bic_selected_source_local_k"),
        (ROW_SINGLE_MEAN, "single_source_mean_reference"),
        (ROW_SINGLE_ORACLE, "diagnostic_only_oracle_reference"),
        (ROW_SHUFFLED_SUMMARY_CONTROL, "negative_control"),
        (ROW_SHUFFLED_LABEL_CONTROL, "negative_control"),
    ):
        row = d1a._dense_empty_row(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            candidates=candidates,
            summaries=empty,
            prior_method=method,
            source_union_ref=source_union_ref,
            center_balanced_ref=center_balanced_ref,
            real_feature_bacc=math.nan,
            status=status,
            error_message=error_message,
            claim_role=role,
        )
        rows.append(_extend_row(row))
    rows.append(_extend_row(d1a._reference_matrix_row(
        cfg,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        candidates=candidates,
        prior_method=ROW_SOURCE_UNION_K16_REFERENCE,
        reference=source_union_ref,
    )))
    rows.append(_extend_row(d1a._reference_matrix_row(
        cfg,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        candidates=candidates,
        prior_method=ROW_CENTER_BALANCED_K16_REFERENCE,
        reference=center_balanced_ref,
    )))
    return rows


def _populate_negative_control_gaps(rows: list[dict[str, object]]) -> None:
    controls: dict[tuple[str, str, str], float] = {}
    for row in rows:
        if row.get("prior_method") not in {ROW_SHUFFLED_SUMMARY_CONTROL, ROW_SHUFFLED_LABEL_CONTROL}:
            continue
        key = (str(row.get("experiment_seed")), str(row.get("heldout_center")), str(row.get("replicate_seed")))
        value = _float(row.get("bacc"))
        if math.isfinite(value):
            controls[key] = max(controls.get(key, -math.inf), value)
    equal: dict[tuple[str, str, str], float] = {}
    for row in rows:
        if row.get("prior_method") != ROW_EQUAL_REFERENCE:
            continue
        key = (str(row.get("experiment_seed")), str(row.get("heldout_center")), str(row.get("replicate_seed")))
        equal[key] = _float(row.get("bacc"))
    for row in rows:
        key = (str(row.get("experiment_seed")), str(row.get("heldout_center")), str(row.get("replicate_seed")))
        if row.get("prior_method") == PRIMARY_RELIABILITY_METHOD:
            control = controls.get(key, math.nan)
            value = _float(row.get("bacc"))
            if math.isfinite(value) and math.isfinite(control):
                row["negative_control_gap"] = value - control
            baseline = equal.get(key, math.nan)
            if math.isfinite(value) and math.isfinite(baseline):
                row["delta_vs_d1_1_equal_adaptive_geom"] = value - baseline


def _centerwise_delta_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    primary = _grouped_cell_means(d1a._rows_for(rows, PRIMARY_RELIABILITY_METHOD))
    equal = _grouped_cell_means(d1a._rows_for(rows, ROW_EQUAL_REFERENCE))
    by_center: dict[str, list[float]] = {}
    by_seed: dict[str, list[float]] = {}
    for key, value in primary.items():
        baseline = equal.get(key, math.nan)
        delta = value - baseline if math.isfinite(value) and math.isfinite(baseline) else math.nan
        seed, center = key
        if math.isfinite(delta):
            by_center.setdefault(center, []).append(delta)
            by_seed.setdefault(seed, []).append(delta)
    rows_out = [
        {"axis": "center", "id": key, "delta_vs_d1_1_equal_adaptive_geom": nanmean(values), "n_cells": len(values)}
        for key, values in sorted(by_center.items())
    ]
    rows_out.extend(
        {"axis": "seed", "id": key, "delta_vs_d1_1_equal_adaptive_geom": nanmean(values), "n_cells": len(values)}
        for key, values in sorted(by_seed.items())
    )
    return rows_out


def _grouped_cell_means(rows: Sequence[Mapping[str, object]]) -> dict[tuple[str, str], float]:
    groups: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        if row.get("status") == "ok":
            groups.setdefault((str(row["experiment_seed"]), str(row["heldout_center"])), []).append(row)
    return {key: d1._mean_field(values, "bacc") for key, values in groups.items()}


def _decision(
    rows: Sequence[Mapping[str, object]],
    cfg: DecentralizedReliabilityWeightedConfig,
    *,
    leakage_status: str,
    diagnostic_rows: Sequence[Mapping[str, object]],
    reliability_rows: Sequence[Mapping[str, object]],
    weight_rows: Sequence[Mapping[str, object]],
    rank_rows: Sequence[Mapping[str, object]],
    centerwise_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    primary_all = d1a._rows_for(rows, PRIMARY_RELIABILITY_METHOD, include_non_ok=True)
    primary = d1a._rows_for(rows, PRIMARY_RELIABILITY_METHOD)
    equal = d1a._rows_for(rows, ROW_EQUAL_REFERENCE)
    controls = [row for row in rows if row.get("prior_method") in {ROW_SHUFFLED_SUMMARY_CONTROL, ROW_SHUFFLED_LABEL_CONTROL} and row.get("status") == "ok"]
    single_mean = d1a._rows_for(rows, ROW_SINGLE_MEAN)
    single_oracle = d1a._rows_for(rows, ROW_SINGLE_ORACLE)
    real_feature = d1a._rows_for(rows, ROW_REAL_FEATURE_DENSE_REFERENCE)
    source_union = d1a._rows_for(rows, ROW_SOURCE_UNION_K16_REFERENCE)
    center_balanced = d1a._rows_for(rows, ROW_CENTER_BALANCED_K16_REFERENCE)
    stats = d1a._primary_stats(primary)
    equal_stats = d1a._primary_stats(equal)
    single_mean_stats = d1a._primary_stats(single_mean)
    single_oracle_stats = d1a._primary_stats(single_oracle)
    real_stats = d1a._primary_stats(real_feature)
    source_union_stats = d1a._primary_stats(source_union)
    center_balanced_stats = d1a._primary_stats(center_balanced)
    control_stats = d1a._primary_stats(controls)
    intervention = d1a._adaptive_intervention_stats(diagnostic_rows, cfg)
    delta_vs_equal = _float(stats["center_equal_mean_bacc"]) - _float(equal_stats["center_equal_mean_bacc"])
    delta_vs_mean_single = _float(stats["center_equal_mean_bacc"]) - _float(single_mean_stats["center_equal_mean_bacc"])
    delta_vs_oracle_single = _float(stats["center_equal_mean_bacc"]) - _float(single_oracle_stats["center_equal_mean_bacc"])
    delta_vs_real = _float(stats["center_equal_mean_bacc"]) - _float(real_stats["center_equal_mean_bacc"])
    retention_source_union = d1._retention(_float(stats["center_equal_mean_bacc"]), _float(source_union_stats["center_equal_mean_bacc"]))
    retention_center_balanced = d1._retention(_float(stats["center_equal_mean_bacc"]), _float(center_balanced_stats["center_equal_mean_bacc"]))
    center_deltas = {str(row["id"]): _float(row["delta_vs_d1_1_equal_adaptive_geom"]) for row in centerwise_rows if row.get("axis") == "center"}
    seed_deltas = {str(row["id"]): _float(row["delta_vs_d1_1_equal_adaptive_geom"]) for row in centerwise_rows if row.get("axis") == "seed"}
    centers_beating = sum(1 for value in center_deltas.values() if math.isfinite(value) and value > 0.0)
    seeds_beating = sum(1 for value in seed_deltas.values() if math.isfinite(value) and value > 0.0)
    weight_stats = _weight_diagnostics(weight_rows, reliability_rows)
    spearman_values = [_float(row.get("cell_spearman")) for row in rank_rows]
    spearman = nanmean([value for value in spearman_values if math.isfinite(value)])
    fit_ineligible = any(row.get("status") == "ineligible_component_fit" for row in primary_all)
    negative_control_gap = _float(stats["center_equal_mean_bacc"]) - _float(control_stats["center_equal_mean_bacc"])
    pass_refs = (
        (not source_union or retention_source_union >= 0.95)
        and (not center_balanced or retention_center_balanced >= 0.95)
    )
    primary_pass = (
        leakage_status == "PASS"
        and not fit_ineligible
        and int(stats["n_heldout_centers"]) >= len(cfg.heldout_centers)
        and _float(stats["center_equal_mean_bacc"]) >= 0.85
        and _float(stats["min_center_mean_bacc"]) >= 0.75
        and _float(stats["seed_std_bacc"]) <= 0.06
        and delta_vs_equal >= 0.01
        and centers_beating >= 4
        and seeds_beating >= 2
        and delta_vs_mean_single > 0.0
        and pass_refs
        and negative_control_gap >= 0.03
    )
    partial = (
        leakage_status == "PASS"
        and not fit_ineligible
        and int(stats["n_heldout_centers"]) >= len(cfg.heldout_centers)
        and (delta_vs_equal >= 0.01 or (_float(stats["min_center_mean_bacc"]) - _float(equal_stats["min_center_mean_bacc"])) > 0.0)
    )
    negative = (
        leakage_status == "PASS"
        and not fit_ineligible
        and (not math.isfinite(delta_vs_equal) or delta_vs_equal <= 0.0 or (math.isfinite(delta_vs_oracle_single) and delta_vs_oracle_single <= 0.0))
    )
    verdict = "D1_2_FAIL"
    if leakage_status != "PASS":
        verdict = "PROTOCOL_FAIL"
    elif fit_ineligible:
        verdict = "INELIGIBLE"
    elif primary_pass:
        verdict = "D1_2_PASS"
    elif partial:
        verdict = "D1_2_PARTIAL_EVIDENCE"
    elif negative:
        verdict = "D1_2_NEGATIVE_EVIDENCE"
    elif int(stats["n_heldout_centers"]) < len(cfg.heldout_centers):
        verdict = "TARGET_EVAL_INSUFFICIENT"
    flags = []
    if fit_ineligible:
        flags.append("INELIGIBLE_COMPONENT_FIT")
    if math.isfinite(delta_vs_equal) and delta_vs_equal < 0.01:
        flags.append("DELTA_VS_D1_1_BELOW_0P01")
    if centers_beating < 4:
        flags.append("CENTER_CONSISTENCY_BELOW_4_OF_5")
    if seeds_beating < 2:
        flags.append("SEED_CONSISTENCY_BELOW_2_OF_3")
    if math.isfinite(retention_source_union) and retention_source_union < 0.95:
        flags.append("SOURCE_UNION_RETENTION_BELOW_0P95")
    if math.isfinite(retention_center_balanced) and retention_center_balanced < 0.95:
        flags.append("CENTER_BALANCED_RETENTION_BELOW_0P95")
    if math.isfinite(delta_vs_oracle_single) and delta_vs_oracle_single <= 0.0:
        flags.append("DOES_NOT_BEAT_SINGLE_SOURCE_ORACLE_ADAPTIVE_K")
    if math.isfinite(negative_control_gap) and negative_control_gap < 0.03:
        flags.append("NEGATIVE_CONTROL_GAP_BELOW_0P03")
    return {
        "primary_verdict": verdict,
        "diagnostic_flags": "|".join(flags),
        "leakage_status": leakage_status,
        "primary_method": PRIMARY_RELIABILITY_METHOD,
        "center_equal_mean_bacc": stats["center_equal_mean_bacc"],
        "center_equal_macro_f1": stats["center_equal_macro_f1"],
        "min_center_bacc": stats["min_center_mean_bacc"],
        "seed_std_bacc": stats["seed_std_bacc"],
        "delta_vs_d1_1_equal_adaptive_geom": delta_vs_equal,
        "delta_vs_mean_single_source_adaptive_k": delta_vs_mean_single,
        "delta_vs_single_source_oracle_adaptive_k": delta_vs_oracle_single,
        "retention_vs_source_union_k16": retention_source_union,
        "retention_vs_center_balanced_k16": retention_center_balanced,
        "delta_vs_real_source_embedding_dense_reference": delta_vs_real,
        "negative_control_gap": negative_control_gap,
        "centerwise_delta_vs_d1_1_json": json.dumps(center_deltas, sort_keys=True),
        "seedwise_delta_vs_d1_1_json": json.dumps(seed_deltas, sort_keys=True),
        "centers_beating_d1_1_equal": centers_beating,
        "seeds_beating_d1_1_equal": seeds_beating,
        "source_reliability_vs_target_single_source_spearman": spearman,
        "d1_1_equal_adaptive_geom_center_equal_mean_bacc": equal_stats["center_equal_mean_bacc"],
        "mean_single_source_adaptive_k_center_equal_mean_bacc": single_mean_stats["center_equal_mean_bacc"],
        "single_source_oracle_adaptive_k_center_equal_mean_bacc": single_oracle_stats["center_equal_mean_bacc"],
        "source_union_k16_reference_center_equal_mean_bacc": source_union_stats["center_equal_mean_bacc"],
        "center_balanced_k16_reference_center_equal_mean_bacc": center_balanced_stats["center_equal_mean_bacc"],
        "real_feature_dense_reference_center_equal_mean_bacc": real_stats["center_equal_mean_bacc"],
        "negative_control_center_equal_mean_bacc": control_stats["center_equal_mean_bacc"],
        "eligible_heldout_centers": stats["n_heldout_centers"],
        "eligible_seed_center_cells": stats["n_decision_cells"],
        **weight_stats,
        **intervention,
        **stats,
    }


def _weight_diagnostics(weight_rows: Sequence[Mapping[str, object]], reliability_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    cells: dict[tuple[str, str, str], list[Mapping[str, object]]] = {}
    for row in weight_rows:
        cells.setdefault((str(row["experiment_seed"]), str(row["replicate_seed"]), str(row["heldout_center"])), []).append(row)
    entropies = [_float(values[0].get("weight_entropy")) for values in cells.values() if values]
    eff = [_float(values[0].get("effective_num_sources")) for values in cells.values() if values]
    l1 = [_float(values[0].get("l1_distance_from_uniform")) for values in cells.values() if values]
    max_weights = [_float(row.get("normalized_reliability_weight")) for row in weight_rows]
    min_weights = [_float(row.get("normalized_reliability_weight")) for row in weight_rows]
    dominant_fraction = (
        sum(1 for values in cells.values() if values and _float(values[0].get("max_weight")) >= 0.5) / float(len(cells))
        if cells else math.nan
    )
    fallback_rows = [row for row in reliability_rows if row.get("reliability_status") == "neutral_fallback"]
    fallback_by_source: dict[str, int] = {}
    fallback_by_seed: dict[str, int] = {}
    for row in fallback_rows:
        fallback_by_source[str(row["source_center"])] = fallback_by_source.get(str(row["source_center"]), 0) + 1
        fallback_by_seed[str(row["experiment_seed"])] = fallback_by_seed.get(str(row["experiment_seed"]), 0) + 1
    rel_bacc = [_float(row.get("raw_reliability_bacc")) for row in reliability_rows if row.get("reliability_status") == "ok"]
    return {
        "mean_effective_num_sources": nanmean([value for value in eff if math.isfinite(value)]),
        "mean_reliability_weight_entropy": nanmean([value for value in entropies if math.isfinite(value)]),
        "mean_l1_distance_from_uniform": nanmean([value for value in l1 if math.isfinite(value)]),
        "max_weight_per_fold": max([value for value in max_weights if math.isfinite(value)], default=math.nan),
        "min_weight_per_fold": min([value for value in min_weights if math.isfinite(value)], default=math.nan),
        "dominant_source_fraction": dominant_fraction,
        "neutral_reliability_fallback_count": len(fallback_rows),
        "neutral_reliability_fallback_fraction": len(fallback_rows) / float(len(reliability_rows)) if reliability_rows else math.nan,
        "fallback_by_source_center_json": json.dumps(fallback_by_source, sort_keys=True),
        "fallback_by_seed_json": json.dumps(fallback_by_seed, sort_keys=True),
        "min_source_reliability_bacc": min([value for value in rel_bacc if math.isfinite(value)], default=math.nan),
    }


def _write_artifacts(
    root: Path,
    cfg: DecentralizedReliabilityWeightedConfig,
    *,
    matrix_rows: Sequence[Mapping[str, object]],
    gap_rows: Sequence[Mapping[str, object]],
    summary_manifest_rows: Sequence[Mapping[str, object]],
    diagnostic_rows: Sequence[Mapping[str, object]],
    reliability_rows: Sequence[Mapping[str, object]],
    weight_rows: Sequence[Mapping[str, object]],
    rank_rows: Sequence[Mapping[str, object]],
    centerwise_rows: Sequence[Mapping[str, object]],
    late_rows: Sequence[Mapping[str, object]],
    real_feature_rows: Sequence[Mapping[str, object]],
    coverage_rows: Sequence[Mapping[str, object]],
    weak_rows: Sequence[Mapping[str, object]],
    nn_rows: Sequence[Mapping[str, object]],
    model_manifest_rows: Sequence[Mapping[str, object]],
    decision: Mapping[str, object],
    leakage_status: str,
    protocol_violations: Sequence[str],
    target_expert_excluded: bool,
) -> None:
    write_csv_rows(root / "tables" / "decentralized_reliability_downstream_matrix.csv", matrix_rows, columns=_matrix_columns())
    write_csv_rows(root / "tables" / "decentralized_reliability_gap_summary.csv", gap_rows, columns=_matrix_columns())
    write_csv_rows(root / "tables" / "decentralized_reliability_summary.csv", [dict(decision)])
    write_csv_rows(root / "tables" / "source_reliability_manifest.csv", reliability_rows)
    write_csv_rows(root / "tables" / "reliability_weight_manifest.csv", weight_rows)
    write_csv_rows(root / "tables" / "source_reliability_rank_vs_target_utility.csv", rank_rows)
    write_csv_rows(root / "tables" / "centerwise_delta_summary.csv", centerwise_rows)
    write_csv_rows(root / "tables" / "late_aggregation_matrix.csv", late_rows, columns=_matrix_columns())
    write_csv_rows(root / "tables" / "real_feature_reference_matrix.csv", real_feature_rows, columns=_matrix_columns())
    write_csv_rows(root / "tables" / "generated_component_coverage_audit.csv", coverage_rows)
    write_csv_rows(root / "tables" / "weak_source_audit.csv", weak_rows)
    write_csv_rows(root / "tables" / "nearest_neighbor_memorization_audit.csv", nn_rows)
    write_csv_rows(root / "tables" / "negative_control_summary.csv", [_negative_control_summary(decision)])
    write_csv_rows(root / "tables" / "exported_source_summary_manifest.csv", summary_manifest_rows, columns=d1a._summary_manifest_columns())
    write_csv_rows(root / "tables" / "source_summary_diagnostics.csv", diagnostic_rows, columns=d1a._diagnostic_columns())
    write_csv_rows(root / "manifests" / "decentralized_reliability_prior_model_manifest.csv", model_manifest_rows)
    leakage = build_leakage_report(
        target_support_labels_for_selection=False,
        target_eval_labels_for_scoring_only=True,
        target_expert_excluded=target_expert_excluded,
        oracle_rows_diagnostic_only=True,
        extra_violations=protocol_violations,
    )
    write_protocol_finalization(
        root,
        leakage_report=leakage.to_json_dict(),
        protocol_manifest={
            "schema_version": "cvae_rebuild_decentralized_reliability_weighted_gmm_prior_protocol_manifest_v1",
            "experiment_name": cfg.name,
            "experiment_type": "source_local_reliability_weighted_decentralized_composition",
            "primary_variant": cfg.primary_variant,
            "primary_method": cfg.primary_method,
            "target_support_labels_for_selection": False,
            "target_eval_labels_for_scoring_only": True,
            "target_expert_excluded": target_expert_excluded,
            "exported_source_summaries_are_target_agnostic": True,
            "source_reliability_manifest_has_no_heldout_center": True,
            "fold_weight_manifest_excludes_heldout_center": True,
            "raw_source_embedding_pooling_for_prior_fit": False,
            "adaptive_k_selection_uses_source_local_fit_statistics_only": True,
            "source_reliability_uses_source_local_eval_only": True,
            "source_union_references_diagnostic_only": True,
            "oracle_rows_diagnostic_only": True,
            "protocol_wording": PROTOCOL_WORDING,
            "claim_boundary": (
                "source-local reliability-weighted decentralized composition only; no target-specific "
                "compatibility routing claim, no metadata-routing claim, no support-NELBO downstream claim, "
                "and no formal privacy claim"
            ),
        },
        resolved_config=_resolved_config(cfg),
    )
    _write_decision_summary(root, decision, leakage_status=leakage_status)


def _matrix_columns() -> tuple[str, ...]:
    return d1a._matrix_columns() + (
        "reliability_weight_json",
        "reliability_score_json",
        "reliability_budget_per_class_json",
        "reliability_weight_entropy",
        "effective_num_sources",
        "l1_distance_from_uniform",
        "max_reliability_weight",
        "min_reliability_weight",
        "dominant_source",
        "dominant_source_weight",
        "delta_vs_d1_1_equal_adaptive_geom",
    )


def _negative_control_summary(decision: Mapping[str, object]) -> dict[str, object]:
    return {
        "primary_method": PRIMARY_RELIABILITY_METHOD,
        "control_methods": f"{ROW_SHUFFLED_SUMMARY_CONTROL}|{ROW_SHUFFLED_LABEL_CONTROL}",
        "primary_center_equal_mean_bacc": decision.get("center_equal_mean_bacc", math.nan),
        "control_center_equal_mean_bacc": decision.get("negative_control_center_equal_mean_bacc", math.nan),
        "negative_control_gap": decision.get("negative_control_gap", math.nan),
        "control_competitive": _float(decision.get("negative_control_gap")) < 0.03,
    }


def _write_decision_summary(root: Path, decision: Mapping[str, object], *, leakage_status: str) -> None:
    text = "\n".join(
        [
            "# D1.2: Source-Local Reliability-Weighted Decentralized Composition",
            "",
            "## Summary",
            "",
            f"- Primary method: `{PRIMARY_RELIABILITY_METHOD}`",
            f"- Primary verdict: `{decision.get('primary_verdict', 'D1_2_FAIL')}`",
            f"- Diagnostic flags: `{decision.get('diagnostic_flags', '')}`",
            f"- Center-equal mean BACC: {_format_float(decision.get('center_equal_mean_bacc'))}",
            f"- Center-equal macro-F1: {_format_float(decision.get('center_equal_macro_f1'))}",
            f"- Min center BACC: {_format_float(decision.get('min_center_bacc'))}",
            f"- Seed std BACC: {_format_float(decision.get('seed_std_bacc'))}",
            f"- Delta vs D1.1 equal adaptive geom: {_format_float(decision.get('delta_vs_d1_1_equal_adaptive_geom'))}",
            f"- Centers beating D1.1 equal: {decision.get('centers_beating_d1_1_equal', '')}",
            f"- Seeds beating D1.1 equal: {decision.get('seeds_beating_d1_1_equal', '')}",
            f"- Delta vs mean single-source adaptive K: {_format_float(decision.get('delta_vs_mean_single_source_adaptive_k'))}",
            f"- Retention vs source-union K16: {_format_float(decision.get('retention_vs_source_union_k16'))}",
            f"- Retention vs center-balanced K16: {_format_float(decision.get('retention_vs_center_balanced_k16'))}",
            f"- Delta vs real-feature dense reference: {_format_float(decision.get('delta_vs_real_source_embedding_dense_reference'))}",
            f"- Negative-control gap: {_format_float(decision.get('negative_control_gap'))}",
            f"- Source reliability vs target single-source Spearman: {_format_float(decision.get('source_reliability_vs_target_single_source_spearman'))}",
            f"- Mean effective source count: {_format_float(decision.get('mean_effective_num_sources'))}",
            f"- Mean L1 distance from uniform: {_format_float(decision.get('mean_l1_distance_from_uniform'))}",
            f"- Neutral reliability fallback count: {decision.get('neutral_reliability_fallback_count', '')}",
            f"- Leakage status: `{leakage_status}`",
            "",
            "## Protocol Boundary",
            "",
            PROTOCOL_WORDING,
            "",
            "This is source-local reliability-weighted decentralized composition.",
            "It is not a target-specific compatibility-routing result.",
            "It does not prove metadata routing or support-NELBO routing.",
            "The source-union K16 rows are centralized diagnostic references only.",
            "",
            "## Supported Claim If PASS",
            "",
            "Source-local generation-preservation reliability improves decentralized adaptive latent-summary composition under a raw-data-free summary-exchange protocol.",
            "",
        ]
    )
    path = root / "reports" / "decision_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _resolved_config(cfg: DecentralizedReliabilityWeightedConfig) -> dict[str, object]:
    return {
        "name": cfg.name,
        "artifact_root": str(cfg.artifact_root),
        "repair_artifact_root": str(cfg.repair_artifact_root),
        "source_union_gmm_artifact_root": "" if cfg.source_union_gmm_artifact_root is None else str(cfg.source_union_gmm_artifact_root),
        "balanced_gmm_artifact_root": "" if cfg.balanced_gmm_artifact_root is None else str(cfg.balanced_gmm_artifact_root),
        "feature_cache_root": str(cfg.feature_cache_root),
        "backbone": cfg.backbone,
        "experiment_seeds": list(cfg.experiment_seeds),
        "heldout_centers": list(cfg.heldout_centers),
        "replicate_seeds": list(cfg.replicate_seeds),
        "synthetic_per_class_total": cfg.synthetic_per_class_total,
        "min_per_source_per_class": cfg.min_per_source_per_class,
        "primary_variant": cfg.primary_variant,
        "primary_method": cfg.primary_method,
        "candidate_components_per_source_class": list(cfg.candidate_components_per_source_class),
        "min_samples_per_component": cfg.min_samples_per_component,
        "source_weighting": cfg.source_weighting,
        "gmm_covariance_type": cfg.gmm_covariance_type,
        "gmm_reg_covar": cfg.gmm_reg_covar,
        "gmm_n_init": cfg.gmm_n_init,
        "gmm_max_iter": cfg.gmm_max_iter,
        "min_component_weight": cfg.min_component_weight,
        "variance_floor": cfg.variance_floor,
        "primary_pooling": cfg.primary_pooling,
        "reliability_floor_score": cfg.reliability_floor_score,
        "softmax_tau": cfg.softmax_tau,
        "classifier": {
            "type": cfg.classifier_type,
            "solver": cfg.classifier_solver,
            "C": cfg.classifier_c,
            "max_iter": cfg.classifier_max_iter,
            "class_weight": cfg.classifier_class_weight,
            "classifier_seed": cfg.classifier_seed,
        },
    }
