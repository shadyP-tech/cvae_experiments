from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from data.features import load_feature_cache, select_rows
from core.metrics import nanmean
from experiments.preservation.preservation_repair import (
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
from experiments.preservation.preservation_sampling import DIAGNOSTIC_SELECTION, PRIMARY_SELECTION, _manifest_row, _per_source_variant, _runtime_source
from core.protocol import ProtocolError, assert_candidate_pool, build_leakage_report
from core.reporting import prepare_artifact_dirs, write_csv_rows, write_json
from data.splits import candidate_experts
from core.domain_regime import (
    CAMELYON17_DOMAIN_REGIME,
    MIDOGPP_DOMAIN_REGIME,
    MidogppContractInfo,
    load_midogpp_contract_info,
    normalize_domain_regime,
    validate_cache_report_split_counts,
    validate_domain_regime_config,
    validate_runtime_domain_coverage,
)

from experiments.decentralized import decentralized_adaptive_gmm_prior as d1a
from experiments.decentralized import decentralized_k16_gmm_prior as d1
from experiments.decentralized import decentralized_reliability_weighted_gmm_prior as d12


PAIRED_DENSE_ALL4_NAME = "virchow2_cvae_paired_dense_all4_reliability_confirmation_v1"
DENSE_LATE_ALL_SOURCES_MIDOGPP_NAME = "virchow2_cvae_dense_late_all_sources_midogpp_v1"
PRIMARY_PAIRED_METHOD = "paired_reliability_all4_shrink050_geom"
PRIMARY_DENSE_ALL_SOURCES_METHOD = "dense_late_all_sources_reliability_shrink050_geom"
ROW_EQUAL_ALL4 = "paired_equal_all4_geom"
ROW_RELIABILITY_ALL4_WEIGHTED = "paired_reliability_all4_weighted_geom"
ROW_POOL_ONLY = "paired_reliability_all4_pool_only_geom"
ROW_BUDGET_ONLY = "paired_reliability_all4_budget_only_geom"
ROW_SHRINK025 = "paired_reliability_all4_shrink025_geom"
ROW_SHRINK050 = PRIMARY_PAIRED_METHOD
ROW_SHUFFLED = "paired_shuffled_reliability_all4_geom"
ROW_INVERSE = "paired_inverse_reliability_all4_geom"
ROW_EQUAL_ALL_SOURCES = "dense_late_equal_all_sources_geom"
ROW_RELIABILITY_ALL_SOURCES_WEIGHTED = "dense_late_all_sources_reliability_weighted_geom"
ROW_POOL_ONLY_ALL_SOURCES = "dense_late_all_sources_reliability_pool_only_geom"
ROW_BUDGET_ONLY_ALL_SOURCES = "dense_late_all_sources_reliability_budget_only_geom"
ROW_SHRINK025_ALL_SOURCES = "dense_late_all_sources_reliability_shrink025_geom"
ROW_SHRINK050_ALL_SOURCES = PRIMARY_DENSE_ALL_SOURCES_METHOD
ROW_SHUFFLED_ALL_SOURCES = "dense_late_all_sources_shuffled_reliability_geom"
ROW_INVERSE_ALL_SOURCES = "dense_late_all_sources_inverse_reliability_geom"
EQUAL_BUDGET_PAIRING_GROUP = "paired_equal_budget_all4_v1"
RELIABILITY_BUDGET_PAIRING_GROUP = "paired_reliability_budget_all4_v1"
EQUAL_BUDGET_ALL_SOURCES_PAIRING_GROUP = "dense_late_equal_budget_all_sources_v1"
RELIABILITY_BUDGET_ALL_SOURCES_PAIRING_GROUP = "dense_late_reliability_budget_all_sources_v1"
PROTOCOL_WORDING = (
    "This is a heldout-excluded source-only reliability audit for dense generated-embedding aggregation. "
    "Dense all4 includes every non-target source expert; reliability affects weights, pooling, or synthetic budgets only."
)
MIDOGPP_PROTOCOL_WORDING = (
    "This is a heldout-excluded source-only reliability audit for dense generated-embedding aggregation on MIDOG++. "
    "Dense all-source includes every eligible non-target pseudo-domain expert; reliability affects weights, pooling, "
    "or synthetic budgets only."
)


@dataclass(frozen=True)
class PairedDenseAll4ReliabilityConfig:
    name: str
    artifact_root: Path
    repair_artifact_root: Path
    d1_2_artifact_root: Path | None
    d1_4_artifact_root: Path | None
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
    reliability_floor_score: float
    reliability_epsilon: float
    shrinkage_values: tuple[float, ...]
    primary_pooling: str
    classifier_type: str
    classifier_solver: str
    classifier_c: float
    classifier_max_iter: int
    classifier_class_weight: str
    classifier_seed: int | None
    domain_regime: str = CAMELYON17_DOMAIN_REGIME
    strict_full_run_matrix: bool = False
    strict_available_seed_domain_coverage: bool = False
    dataset_contract_artifact_root: Path | None = None
    cache_report_path: Path | None = None

    @property
    def max_local_gmm_components_per_source_class(self) -> int:
        return max(self.candidate_components_per_source_class)

    @property
    def composed_components_per_class_nominal(self) -> int:
        return self.max_local_gmm_components_per_source_class * (len(self.heldout_centers) - 1)

    @property
    def softmax_tau(self) -> float:
        return 1.0


def load_paired_dense_all4_reliability_config(path: str | Path) -> PairedDenseAll4ReliabilityConfig:
    source = Path(path).resolve()
    data = _load_mapping(source)
    base_dir = source.parents[2] if len(source.parents) >= 3 else source.parent
    return parse_paired_dense_all4_reliability_config(data, base_dir=base_dir)


def load_dense_late_all_sources_reliability_config(path: str | Path) -> PairedDenseAll4ReliabilityConfig:
    return load_paired_dense_all4_reliability_config(path)


def parse_paired_dense_all4_reliability_config(
    data: Mapping[str, Any],
    *,
    base_dir: str | Path = ".",
) -> PairedDenseAll4ReliabilityConfig:
    base = Path(base_dir)
    experiment = _mapping(data, "experiment")
    inputs = _mapping(data, "inputs")
    run = _mapping(data, "run_matrix")
    if "support_size" in run or "support_seeds" in run:
        raise ProtocolError("Paired dense all4 reliability confirmation must not configure target support.")
    generation = _mapping(data, "generation")
    gmm = _mapping(data, "dense_late_all_sources_reliability") if "dense_late_all_sources_reliability" in data else _mapping(data, "paired_dense_all4_reliability")
    classifier = _mapping(data, "classifier")
    cfg = PairedDenseAll4ReliabilityConfig(
        name=str(experiment["name"]),
        artifact_root=_path(base, str(experiment["artifact_root"])),
        repair_artifact_root=_path(base, str(inputs["repair_artifact_root"])),
        d1_2_artifact_root=_optional_path(base, inputs.get("d1_2_artifact_root")),
        d1_4_artifact_root=_optional_path(base, inputs.get("d1_4_artifact_root")),
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
        reliability_floor_score=float(gmm["reliability_floor_score"]),
        reliability_epsilon=float(gmm["reliability_epsilon"]),
        shrinkage_values=tuple(float(v) for v in gmm["shrinkage_values"]),
        primary_pooling=str(gmm["primary_pooling"]),
        classifier_type=str(classifier["type"]),
        classifier_solver=str(classifier["solver"]),
        classifier_c=float(classifier["C"]),
        classifier_max_iter=int(classifier["max_iter"]),
        classifier_class_weight=str(classifier["class_weight"]),
        classifier_seed=None if classifier.get("classifier_seed") is None else int(classifier["classifier_seed"]),
        domain_regime=normalize_domain_regime(run.get("domain_regime")),
        strict_full_run_matrix=bool(run.get("strict_full_run_matrix", False)),
        strict_available_seed_domain_coverage=bool(run.get("strict_available_seed_domain_coverage", False)),
        dataset_contract_artifact_root=_optional_path(base, inputs.get("dataset_contract_artifact_root")),
        cache_report_path=_optional_path(base, inputs.get("cache_report_path")),
    )
    validate_paired_dense_all4_reliability_config(cfg)
    return cfg


def validate_paired_dense_all4_reliability_config(cfg: PairedDenseAll4ReliabilityConfig) -> None:
    regime = normalize_domain_regime(cfg.domain_regime)
    contract_info = validate_domain_regime_config(
        domain_regime=regime,
        heldout_centers=cfg.heldout_centers,
        dataset_contract_artifact_root=cfg.dataset_contract_artifact_root,
        artifact_root=cfg.artifact_root,
        strict_full_run_matrix=cfg.strict_full_run_matrix,
        strict_available_seed_domain_coverage=cfg.strict_available_seed_domain_coverage,
    )
    validate_cache_report_split_counts(cfg.cache_report_path, contract_info)
    if regime == CAMELYON17_DOMAIN_REGIME:
        if cfg.name != PAIRED_DENSE_ALL4_NAME:
            raise ProtocolError(f"Paired dense all4 experiment name must be {PAIRED_DENSE_ALL4_NAME!r}.")
        expected_primary = PRIMARY_PAIRED_METHOD
        expected_weighting = "heldout_excluded_source_local_reliability_dense_all4"
    else:
        if cfg.name != DENSE_LATE_ALL_SOURCES_MIDOGPP_NAME:
            raise ProtocolError(f"MIDOG++ dense all-source experiment name must be {DENSE_LATE_ALL_SOURCES_MIDOGPP_NAME!r}.")
        expected_primary = PRIMARY_DENSE_ALL_SOURCES_METHOD
        expected_weighting = "heldout_excluded_source_local_reliability_dense_all_sources"
    if cfg.backbone != "virchow2":
        raise ProtocolError("Paired dense all4 confirmation is locked to backbone=virchow2.")
    if cfg.primary_variant != PRIMARY_VARIANT:
        raise ProtocolError(f"primary_variant must be {PRIMARY_VARIANT!r}.")
    if cfg.primary_method != expected_primary:
        raise ProtocolError(f"primary_method must be {expected_primary!r}.")
    if not cfg.experiment_seeds:
        raise ProtocolError("experiment_seeds must be non-empty; thesis config locks [42, 43, 44].")
    if not cfg.replicate_seeds:
        raise ProtocolError("replicate_seeds must be non-empty; thesis config locks [17, 23, 31].")
    if cfg.candidate_components_per_source_class != (4, 3, 2, 1):
        raise ProtocolError("candidate_components_per_source_class must be locked to [4, 3, 2, 1].")
    if cfg.min_samples_per_component != 12:
        raise ProtocolError("min_samples_per_component must be locked to 12.")
    if cfg.source_weighting != expected_weighting:
        raise ProtocolError(f"source_weighting must be {expected_weighting}.")
    if cfg.gmm_covariance_type != "diag":
        raise ProtocolError("gmm_covariance_type must be diag.")
    if cfg.gmm_reg_covar != 1.0e-4 or cfg.gmm_n_init != 5 or cfg.gmm_max_iter != 500:
        raise ProtocolError("GMM settings must match locked D1.2 values: reg_covar=1e-4, n_init=5, max_iter=500.")
    if cfg.min_component_weight != 0.02 or cfg.variance_floor != 1.0e-5:
        raise ProtocolError("GMM component/variance floors must match locked D1.2 values.")
    if cfg.synthetic_per_class_total != 128 or cfg.min_per_source_per_class != 8:
        raise ProtocolError("Synthetic budget must be 128 total per class with min_per_source_per_class=8.")
    if cfg.primary_pooling != "weighted_geometric":
        raise ProtocolError("primary_pooling must be weighted_geometric.")
    if cfg.reliability_floor_score <= 0.0 or cfg.reliability_epsilon <= 0.0:
        raise ProtocolError("Reliability floors must be positive.")
    if cfg.shrinkage_values != (0.25, 0.5):
        raise ProtocolError("shrinkage_values must be locked to [0.25, 0.5].")
    if cfg.classifier_type != "sklearn_logistic_regression":
        raise ProtocolError("classifier.type must be sklearn_logistic_regression.")
    if cfg.classifier_solver != "lbfgs" or cfg.classifier_c != 1.0 or cfg.classifier_max_iter != 2000:
        raise ProtocolError("Classifier solver/C/max_iter must remain locked.")
    if cfg.classifier_class_weight != "balanced" or cfg.classifier_seed is not None:
        raise ProtocolError("Classifier must use class_weight=balanced and classifier_seed=null.")


def run_paired_dense_all4_reliability_confirmation(
    cfg: PairedDenseAll4ReliabilityConfig,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    root = prepare_artifact_dirs(Path(artifact_root) if artifact_root is not None else cfg.artifact_root)
    if _is_midogpp(cfg) and "cvae_rebuild/artifacts/midogpp" not in root.as_posix():
        raise ProtocolError("MIDOG++ dense all-source artifact_root must be under cvae_rebuild/artifacts/midogpp/.")
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "summaries").mkdir(parents=True, exist_ok=True)

    matrix_rows: list[dict[str, object]] = []
    summary_manifest_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    reliability_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    budget_rows: list[dict[str, object]] = []
    excluded_rows: list[dict[str, object]] = []
    late_rows: list[dict[str, object]] = []
    real_feature_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    weak_rows: list[dict[str, object]] = []
    nn_rows: list[dict[str, object]] = []
    model_manifest_rows: list[dict[str, object]] = []
    source_pool_rows: list[dict[str, object]] = []
    protocol_violations: list[str] = []
    target_expert_excluded = True
    contract_info = _midogpp_contract_info(cfg)

    repair_cfg = d1._repair_runtime_config(cfg, root)
    per_source_variant = _per_source_variant()

    try:
        for experiment_seed in cfg.experiment_seeds:
            train_cache = load_feature_cache(_existing_cache_path(cfg.feature_cache_root, seed=experiment_seed, split="train"))
            test_cache = load_feature_cache(_existing_cache_path(cfg.feature_cache_root, seed=experiment_seed, split="test"))
            if cfg.strict_available_seed_domain_coverage and contract_info is not None:
                source_pool_rows.extend(
                    validate_runtime_domain_coverage(
                        domain_regime=cfg.domain_regime,
                        eligible_domain_ids=contract_info.eligible_domain_ids,
                        experiment_seed=int(experiment_seed),
                        train_metadata=train_cache.metadata,
                        test_metadata=test_cache.metadata,
                    )
                )
            per_source_runtime: dict[str, object] = {}
            largest_summaries: dict[tuple[str, int], d1a.AdaptiveSourceLocalSummary] = {}

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

                largest, _bic = d1a._fit_and_export_source_summaries(
                    cfg,
                    root,
                    runtime_source.runtime,
                    experiment_seed=int(experiment_seed),
                    shuffled_label_control=False,
                )
                for summary in largest:
                    largest_summaries[(summary.source_center, summary.class_label)] = summary
                    summary_manifest_rows.append(d1a._summary_manifest_row(summary))
                    diagnostic_rows.append(d1a._summary_diagnostic_row(cfg, summary))

            reliability: dict[tuple[int, int, str], d12.SourceReliability] = {}
            for replicate_seed in cfg.replicate_seeds:
                for source_center in cfg.heldout_centers:
                    rel = d12._source_local_reliability(
                        cfg,
                        per_source_runtime=per_source_runtime,
                        summaries=largest_summaries,
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

                target_indices = _target_indices(test_cache.metadata, str(heldout_center))
                eval_raw, eval_meta = select_rows(test_cache.embeddings, test_cache.metadata, target_indices)
                eval_labels = tuple(_label(row) for row in eval_meta)
                eval_error = "mono_class_target_eval" if len(set(eval_labels)) < 2 else ""

                for replicate_seed in cfg.replicate_seeds:
                    rels = {source: reliability[(int(experiment_seed), int(replicate_seed), str(source))] for source in candidates}
                    transform = _heldout_excluded_reliability_transform(cfg, heldout_center, candidates, rels)
                    reliability_rows.extend(
                        _source_reliability_rows(
                            cfg,
                            experiment_seed=int(experiment_seed),
                            replicate_seed=int(replicate_seed),
                            heldout_center=str(heldout_center),
                            candidates=candidates,
                            rels=rels,
                            transform=transform,
                        )
                    )
                    plans = _variant_plans(
                        cfg,
                        candidates,
                        transform,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        replicate_seed=int(replicate_seed),
                    )
                    for method, plan in plans.items():
                        weight_rows.extend(
                            _weight_manifest_rows(
                                method,
                                plan,
                                cfg,
                                experiment_seed=int(experiment_seed),
                                replicate_seed=int(replicate_seed),
                                heldout_center=str(heldout_center),
                                transform=transform,
                                rels=rels,
                            )
                        )
                    budget_rows.extend(_realized_budget_rows(plans, experiment_seed, heldout_center, replicate_seed))

                    if eval_error:
                        excluded_rows.append(
                            _excluded_cell_row(experiment_seed, heldout_center, replicate_seed, eval_error, n_eval=len(eval_labels))
                        )
                        matrix_rows.extend(
                            _ineligible_rows(
                                cfg,
                                experiment_seed=int(experiment_seed),
                                heldout_center=str(heldout_center),
                                replicate_seed=int(replicate_seed),
                                candidates=candidates,
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
                    ref_row = _extend_paired_row(ref_row)
                    real_late = [_extend_paired_row(row) for row in real_late]
                    real_feature_rows.append(ref_row)
                    matrix_rows.append(ref_row)
                    late_rows.extend(real_late)

                    for method, plan in plans.items():
                        rows, late, coverage, weak, nn = _evaluate_paired_variant(
                            cfg,
                            per_source_runtime=per_source_runtime,
                            candidates=candidates,
                            summaries=largest_summaries,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            replicate_seed=int(replicate_seed),
                            eval_raw=eval_raw,
                            eval_labels=eval_labels,
                            real_feature_bacc=_float(ref_row["bacc"]),
                            method=method,
                            plan=plan,
                        )
                        matrix_rows.extend(rows)
                        late_rows.extend(late)
                        coverage_rows.extend(coverage)
                        weak_rows.extend(weak)
                        nn_rows.extend(nn)
    except ProtocolError as exc:
        protocol_violations.append(str(exc))

    invariant_rows = _pairing_invariant_rows(matrix_rows, late_rows)
    invariant_pass = all(row.get("audit_status") == "PASS" for row in invariant_rows) if invariant_rows else False
    if not invariant_pass:
        protocol_violations.append("paired_generation_invariant_audit_failed")
    leakage = build_leakage_report(
        target_support_labels_for_selection=False,
        target_eval_labels_for_scoring_only=True,
        target_expert_excluded=target_expert_excluded,
        oracle_rows_diagnostic_only=True,
        extra_violations=protocol_violations,
    )
    center_rows = _center_summary_rows(matrix_rows)
    paired_delta_rows = _paired_delta_rows(matrix_rows)
    negative_rows = _negative_control_rows(matrix_rows)
    gap_rows = _gap_summary_rows(matrix_rows, cfg)
    decision = _decision(
        matrix_rows,
        cfg,
        leakage_status=leakage.status,
        invariant_pass=invariant_pass,
        paired_delta_rows=paired_delta_rows,
        negative_rows=negative_rows,
    )
    _write_artifacts(
        root,
        cfg,
        matrix_rows=matrix_rows,
        gap_rows=gap_rows,
        center_rows=center_rows,
        summary_rows=[decision],
        reliability_rows=reliability_rows,
        weight_rows=weight_rows,
        budget_rows=budget_rows,
        excluded_rows=excluded_rows,
        invariant_rows=invariant_rows,
        paired_delta_rows=paired_delta_rows,
        negative_rows=negative_rows,
        summary_manifest_rows=summary_manifest_rows,
        diagnostic_rows=diagnostic_rows,
        late_rows=late_rows,
        real_feature_rows=real_feature_rows,
        coverage_rows=coverage_rows,
        weak_rows=weak_rows,
        nn_rows=nn_rows,
        model_manifest_rows=model_manifest_rows,
        source_pool_rows=source_pool_rows,
        decision=decision,
        leakage_status=leakage.status,
        protocol_violations=protocol_violations,
        target_expert_excluded=target_expert_excluded,
    )
    return root


def run_dense_late_all_sources_reliability(
    cfg: PairedDenseAll4ReliabilityConfig,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    return run_paired_dense_all4_reliability_confirmation(cfg, artifact_root=artifact_root)


def _optional_path(base: Path, value: object) -> Path | None:
    if value is None or str(value) == "":
        return None
    return _path(base, str(value))


def _heldout_excluded_reliability_transform(
    cfg: PairedDenseAll4ReliabilityConfig,
    heldout_center: str,
    candidates: Sequence[str],
    rels: Mapping[str, d12.SourceReliability],
) -> dict[str, object]:
    sources = tuple(str(source) for source in candidates)
    if str(heldout_center) in sources:
        raise ProtocolError("Heldout target center appeared in reliability transform candidates.")
    raw_scores: dict[str, float] = {}
    eligible: dict[str, bool] = {}
    for source in sources:
        rel = rels[source]
        score = d12._linear_reliability_score(rel.raw_bacc, cfg.reliability_floor_score)
        ok = rel.reliability_status == "ok" and math.isfinite(score) and score > 0.0
        raw_scores[source] = float(score) if math.isfinite(score) else math.nan
        eligible[source] = bool(ok)
    eligible_scores = [raw_scores[source] for source in sources if eligible[source]]
    fallback = nanmean(eligible_scores) if eligible_scores else float(cfg.reliability_floor_score)
    imputed_scores = {
        source: max(float(raw_scores[source] if eligible[source] else fallback), float(cfg.reliability_epsilon))
        for source in sources
    }
    total = sum(imputed_scores.values())
    if total <= 0.0:
        raise ProtocolError("Heldout-excluded reliability weights are not positive.")
    weights = {source: float(imputed_scores[source] / total) for source in sources}
    return {
        "sources": sources,
        "raw_scores": raw_scores,
        "eligible": eligible,
        "imputed_scores": imputed_scores,
        "weights": weights,
        "imputation_value": float(fallback),
        "target_center_excluded_from_reliability": True,
    }


def _variant_plans(
    cfg: PairedDenseAll4ReliabilityConfig,
    sources: Sequence[str],
    transform: Mapping[str, object],
    *,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
) -> dict[str, dict[str, object]]:
    sources_tuple = tuple(str(source) for source in sources)
    rel_weights = {source: float(transform["weights"][source]) for source in sources_tuple}
    scores = {source: float(transform["imputed_scores"][source]) for source in sources_tuple}
    uniform_weights = {source: 1.0 / float(len(sources_tuple)) for source in sources_tuple}
    equal_budgets = {source: int(value) for source, value in zip(sources_tuple, d1._balanced_counts(cfg.synthetic_per_class_total, len(sources_tuple)))}
    rel_budgets = d12._weighted_budgets(cfg.synthetic_per_class_total, sources_tuple, rel_weights, cfg.min_per_source_per_class)
    plans: dict[str, dict[str, object]] = {
        ROW_EQUAL_ALL4: _plan(
            sources_tuple,
            uniform_weights,
            equal_budgets,
            scores,
            weight_rule="uniform",
            budget_policy="equal",
            pairing_group=EQUAL_BUDGET_PAIRING_GROUP,
            generation_seed_method=EQUAL_BUDGET_PAIRING_GROUP,
            source_weighting="equal_source_mass",
        ),
        ROW_RELIABILITY_ALL4_WEIGHTED: _plan(
            sources_tuple,
            rel_weights,
            rel_budgets,
            scores,
            weight_rule="heldout_excluded_reliability",
            budget_policy="reliability_largest_remainder_min8",
            pairing_group=RELIABILITY_BUDGET_PAIRING_GROUP,
            generation_seed_method=RELIABILITY_BUDGET_PAIRING_GROUP,
            source_weighting="heldout_excluded_reliability_weighted_budgeted",
        ),
        ROW_POOL_ONLY: _plan(
            sources_tuple,
            rel_weights,
            equal_budgets,
            scores,
            weight_rule="heldout_excluded_reliability_pool_only",
            budget_policy="equal",
            pairing_group=EQUAL_BUDGET_PAIRING_GROUP,
            generation_seed_method=EQUAL_BUDGET_PAIRING_GROUP,
            source_weighting="heldout_excluded_reliability_pool_only",
        ),
        ROW_BUDGET_ONLY: _plan(
            sources_tuple,
            uniform_weights,
            rel_budgets,
            scores,
            weight_rule="uniform_pooling_reliability_budget",
            budget_policy="reliability_largest_remainder_min8",
            pairing_group=RELIABILITY_BUDGET_PAIRING_GROUP,
            generation_seed_method=RELIABILITY_BUDGET_PAIRING_GROUP,
            source_weighting="heldout_excluded_reliability_budget_only",
        ),
    }
    for shrink, method in ((0.25, ROW_SHRINK025), (0.50, ROW_SHRINK050)):
        weights = {
            source: float(shrink * rel_weights[source] + (1.0 - shrink) * uniform_weights[source])
            for source in sources_tuple
        }
        plans[method] = _plan(
            sources_tuple,
            weights,
            equal_budgets,
            scores,
            weight_rule=f"reliability_shrink_{shrink:.2f}",
            budget_policy="equal",
            pairing_group=EQUAL_BUDGET_PAIRING_GROUP,
            generation_seed_method=EQUAL_BUDGET_PAIRING_GROUP,
            source_weighting=f"heldout_excluded_reliability_shrink_{shrink:.2f}",
        )
    plans[ROW_SHUFFLED] = _plan(
        sources_tuple,
        _shuffled_weights(sources_tuple, rel_weights, experiment_seed, heldout_center, replicate_seed),
        equal_budgets,
        scores,
        weight_rule="shuffled_reliability_weights",
        budget_policy="equal",
        pairing_group=EQUAL_BUDGET_PAIRING_GROUP,
        generation_seed_method=EQUAL_BUDGET_PAIRING_GROUP,
        source_weighting="shuffled_heldout_excluded_reliability",
    )
    plans[ROW_INVERSE] = _plan(
        sources_tuple,
        _inverse_rank_reversal_weights(sources_tuple, rel_weights, scores),
        equal_budgets,
        scores,
        weight_rule="inverse_rank_reversal_matched_entropy",
        budget_policy="equal",
        pairing_group=EQUAL_BUDGET_PAIRING_GROUP,
        generation_seed_method=EQUAL_BUDGET_PAIRING_GROUP,
        source_weighting="inverse_heldout_excluded_reliability_rank_reversal",
    )
    return plans


def _plan(
    sources: Sequence[str],
    weights: Mapping[str, float],
    budgets: Mapping[str, int],
    scores: Mapping[str, float],
    *,
    weight_rule: str,
    budget_policy: str,
    pairing_group: str,
    generation_seed_method: str,
    source_weighting: str,
) -> dict[str, object]:
    plan = d12._with_weight_diagnostics(tuple(sources), weights, budgets, scores)
    plan.update(
        {
            "weight_rule": weight_rule,
            "budget_policy": budget_policy,
            "pairing_group": pairing_group,
            "generation_seed_method": generation_seed_method,
            "source_weighting": source_weighting,
        }
    )
    return plan


def _shuffled_weights(
    sources: Sequence[str],
    weights: Mapping[str, float],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
) -> dict[str, float]:
    sources_tuple = tuple(str(source) for source in sources)
    values = [float(weights[source]) for source in sources_tuple]
    rng = random.Random(d1._latent_seed(experiment_seed, heldout_center, replicate_seed, ROW_SHUFFLED))
    rng.shuffle(values)
    return {source: float(value) for source, value in zip(sources_tuple, values)}


def _inverse_rank_reversal_weights(
    sources: Sequence[str],
    weights: Mapping[str, float],
    scores: Mapping[str, float],
) -> dict[str, float]:
    sources_tuple = tuple(str(source) for source in sources)
    low_to_high_reliability = sorted(sources_tuple, key=lambda source: (float(scores[source]), source))
    high_to_low_weights = sorted((float(weights[source]) for source in sources_tuple), reverse=True)
    return {source: float(weight) for source, weight in zip(low_to_high_reliability, high_to_low_weights)}


def _evaluate_paired_variant(
    cfg: PairedDenseAll4ReliabilityConfig,
    *,
    per_source_runtime: Mapping[str, object],
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    eval_raw: object,
    eval_labels: Sequence[int],
    real_feature_bacc: float,
    method: str,
    plan: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    pooling_rule = "geometric" if method in {ROW_EQUAL_ALL4, ROW_BUDGET_ONLY} else "weighted_geometric"
    selection_source = PRIMARY_SELECTION if method == cfg.primary_method else DIAGNOSTIC_SELECTION
    rows, late, coverage, weak, nn = d12._evaluate_weighted_variant(
        cfg,
        per_source_runtime=per_source_runtime,
        candidates=candidates,
        summaries=summaries,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        eval_raw=eval_raw,
        eval_labels=eval_labels,
        source_union_ref=d1._missing_reference(),
        center_balanced_ref=d1._missing_reference(),
        real_feature_bacc=real_feature_bacc,
        weight_plan=plan,
        prior_method=method,
        pooling_rule=pooling_rule,
        selection_source=selection_source,
        claim_role=_claim_role(method),
        generation_seed_method=str(plan["generation_seed_method"]),
    )
    key = _generation_bundle_key(experiment_seed, heldout_center, replicate_seed, candidates, str(plan["budget_policy"]), "normal")
    return (
        [_extend_paired_row(row, method=method, plan=plan, generation_bundle_key=key) for row in rows],
        [_extend_paired_row(row, method=method, plan=plan, generation_bundle_key=key) for row in late],
        coverage,
        weak,
        nn,
    )


def _claim_role(method: str) -> str:
    if method == ROW_EQUAL_ALL4:
        return "primary_equal_all4_baseline"
    if method in {ROW_SHUFFLED, ROW_INVERSE}:
        return "negative_control"
    if method == ROW_RELIABILITY_ALL4_WEIGHTED:
        return "full_reliability_weight_budget_diagnostic"
    if method == ROW_BUDGET_ONLY:
        return "reliability_budget_only_diagnostic"
    if method == ROW_POOL_ONLY:
        return "reliability_pool_only_diagnostic"
    return "reliability_shrinkage_diagnostic"


def _generation_bundle_key(
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    budget_policy: str,
    control_mode: str,
) -> str:
    return json.dumps(
        {
            "experiment_seed": int(experiment_seed),
            "heldout_center": str(heldout_center),
            "replicate_seed": int(replicate_seed),
            "source_set": "|".join(str(source) for source in candidates),
            "budget_policy": str(budget_policy),
            "control_mode": str(control_mode),
        },
        sort_keys=True,
    )


def _source_reliability_rows(
    cfg: PairedDenseAll4ReliabilityConfig,
    *,
    experiment_seed: int,
    replicate_seed: int,
    heldout_center: str,
    candidates: Sequence[str],
    rels: Mapping[str, d12.SourceReliability],
    transform: Mapping[str, object],
) -> list[dict[str, object]]:
    rows = []
    for source in candidates:
        source_id = str(source)
        rel = rels[source_id]
        eligible = bool(transform["eligible"][source_id])
        rows.append(
            {
                "experiment_seed": int(experiment_seed),
                "replicate_seed": int(replicate_seed),
                "heldout_center": str(heldout_center),
                "source_center": source_id,
                "source_centers_available": "|".join(str(v) for v in candidates),
                "target_center_excluded_from_reliability": True,
                "target_eval_labels_used_for_reliability": False,
                "target_eval_labels_used_for_weighting": False,
                "target_eval_labels_used_for_selection": False,
                "raw_reliability_bacc": rel.raw_bacc,
                "source_reliability_macro_f1": rel.macro_f1,
                "raw_reliability_score": transform["raw_scores"][source_id],
                "imputed_reliability_score": transform["imputed_scores"][source_id],
                "final_normalized_reliability_weight": transform["weights"][source_id],
                "reliability_score_imputed": not eligible,
                "reliability_cell_eligible": eligible,
                "reliability_status": rel.reliability_status,
                "error_message": rel.error_message,
                "n_source_eval": rel.n_eval,
                "generated_features_hash": rel.generated_features_hash,
                "prediction_hash": rel.prediction_hash,
                "reliability_floor_score": cfg.reliability_floor_score,
                "reliability_epsilon": cfg.reliability_epsilon,
                "imputation_value": transform["imputation_value"],
            }
        )
    return rows


def _weight_manifest_rows(
    method: str,
    plan: Mapping[str, object],
    cfg: PairedDenseAll4ReliabilityConfig,
    *,
    experiment_seed: int,
    replicate_seed: int,
    heldout_center: str,
    transform: Mapping[str, object],
    rels: Mapping[str, d12.SourceReliability],
) -> list[dict[str, object]]:
    rows = []
    for source in plan["sources"]:
        source_id = str(source)
        rows.append(
            {
                "method": method,
                "experiment_seed": int(experiment_seed),
                "replicate_seed": int(replicate_seed),
                "heldout_center": str(heldout_center),
                "source_center": source_id,
                "source_centers_available": "|".join(str(v) for v in plan["sources"]),
                "weight_rule": plan["weight_rule"],
                "budget_policy": plan["budget_policy"],
                "pairing_group": plan["pairing_group"],
                "target_center_excluded_from_reliability": True,
                "target_eval_labels_used_for_reliability": False,
                "target_eval_labels_used_for_weighting": False,
                "target_eval_labels_used_for_selection": False,
                "raw_reliability_bacc": rels[source_id].raw_bacc,
                "raw_reliability_score": transform["raw_scores"][source_id],
                "imputed_reliability_score": transform["imputed_scores"][source_id],
                "reliability_cell_eligible": transform["eligible"][source_id],
                "reliability_score_imputed": not bool(transform["eligible"][source_id]),
                "normalized_reliability_weight": transform["weights"][source_id],
                "final_normalized_weight": plan["weights"][source_id],
                "synthetic_per_class_budget": plan["budgets"][source_id],
                "synthetic_per_class_total": cfg.synthetic_per_class_total,
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


def _realized_budget_rows(
    plans: Mapping[str, Mapping[str, object]],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
) -> list[dict[str, object]]:
    rows = []
    for method, plan in plans.items():
        for source in plan["sources"]:
            source_id = str(source)
            rows.append(
                {
                    "method": method,
                    "experiment_seed": int(experiment_seed),
                    "heldout_center": str(heldout_center),
                    "replicate_seed": int(replicate_seed),
                    "source_center": source_id,
                    "budget_policy": plan["budget_policy"],
                    "synthetic_per_class_budget": plan["budgets"][source_id],
                    "budget_sum_per_class": sum(int(v) for v in plan["budgets"].values()),
                }
            )
    return rows


def _extend_paired_row(
    row: Mapping[str, object],
    *,
    method: str | None = None,
    plan: Mapping[str, object] | None = None,
    generation_bundle_key: str = "",
) -> dict[str, object]:
    out = dict(row)
    if plan is None:
        out.update(
            {
                "weight_rule": "",
                "budget_policy": "",
                "pairing_group": "",
                "generation_seed_method": "",
                "generation_bundle_key": generation_bundle_key,
                "paired_equal_delta_bacc": math.nan,
                "paired_equal_delta_macro_f1": math.nan,
            }
        )
        return out
    out["source_weighting"] = str(plan["source_weighting"])
    out.update(
        {
            "weight_rule": plan["weight_rule"],
            "budget_policy": plan["budget_policy"],
            "pairing_group": plan["pairing_group"],
            "generation_seed_method": plan["generation_seed_method"],
            "generation_bundle_key": generation_bundle_key,
            "paired_equal_delta_bacc": math.nan,
            "paired_equal_delta_macro_f1": math.nan,
        }
    )
    if method is not None:
        out["prior_method"] = method
    return out


def _ineligible_rows(
    cfg: PairedDenseAll4ReliabilityConfig,
    *,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    status: str,
    error_message: str,
) -> list[dict[str, object]]:
    rows = []
    empty: dict[tuple[str, int], d1a.AdaptiveSourceLocalSummary] = {}
    for method in _method_order():
        row = d1a._dense_empty_row(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            candidates=candidates,
            summaries=empty,
            prior_method=method,
            source_union_ref=d1._missing_reference(),
            center_balanced_ref=d1._missing_reference(),
            real_feature_bacc=math.nan,
            status=status,
            error_message=error_message,
            claim_role=_claim_role(method),
        )
        rows.append(_extend_paired_row(row))
    return rows


def _excluded_cell_row(
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    reason: str,
    *,
    n_eval: int,
) -> dict[str, object]:
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "replicate_seed": int(replicate_seed),
        "exclusion_reason": str(reason),
        "n_eval": int(n_eval),
    }


def _pairing_invariant_rows(
    matrix_rows: Sequence[Mapping[str, object]],
    late_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows = []
    aggregate_groups: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in matrix_rows:
        group = str(row.get("pairing_group", ""))
        key = str(row.get("generation_bundle_key", ""))
        if row.get("status") == "ok" and group and key and row.get("expert_id") == "dense_all_sources":
            aggregate_groups.setdefault((group, key), []).append(row)
    for (group, key), subset in sorted(aggregate_groups.items()):
        hashes = {str(row.get("generated_features_hash", "")) for row in subset}
        methods = sorted(str(row.get("prior_method")) for row in subset)
        rows.append(
            {
                "audit_scope": "aggregate_generated_features",
                "pairing_group": group,
                "generation_bundle_key": key,
                "expert_id": "dense_all_sources",
                "methods": "|".join(methods),
                "n_methods": len(methods),
                "n_unique_generated_hashes": len(hashes),
                "n_unique_prediction_hashes": "",
                "audit_status": "PASS" if len(hashes) == 1 else "FAIL",
            }
        )
    late_groups: dict[tuple[str, str, str], list[Mapping[str, object]]] = {}
    for row in late_rows:
        group = str(row.get("pairing_group", ""))
        key = str(row.get("generation_bundle_key", ""))
        expert = str(row.get("expert_id", ""))
        if row.get("status") == "ok" and group and key and expert:
            late_groups.setdefault((group, key, expert), []).append(row)
    for (group, key, expert), subset in sorted(late_groups.items()):
        generated = {str(row.get("generated_features_hash", "")) for row in subset}
        predictions = {str(row.get("prediction_hash", "")) for row in subset}
        methods = sorted(str(row.get("prior_method")) for row in subset)
        rows.append(
            {
                "audit_scope": "per_source_generated_and_prediction",
                "pairing_group": group,
                "generation_bundle_key": key,
                "expert_id": expert,
                "methods": "|".join(methods),
                "n_methods": len(methods),
                "n_unique_generated_hashes": len(generated),
                "n_unique_prediction_hashes": len(predictions),
                "audit_status": "PASS" if len(generated) == 1 and len(predictions) == 1 else "FAIL",
            }
        )
    return rows


def _center_summary_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    out = []
    for method in _reported_methods(rows):
        cells = _seed_center_cells(d1a._rows_for(rows, method))
        by_center: dict[str, list[float]] = {}
        for (_seed, center), value in cells.items():
            if math.isfinite(value):
                by_center.setdefault(center, []).append(value)
        for center, values in sorted(by_center.items()):
            out.append(
                {
                    "method": method,
                    "heldout_center": center,
                    "mean_bacc": nanmean(values),
                    "n_seed_center_cells": len(values),
                    "n_experiment_seeds": len(values),
                }
            )
    return out


def _paired_delta_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    equal = _seed_center_cells(d1a._rows_for(rows, ROW_EQUAL_ALL4))
    out = []
    for method in _reported_methods(rows):
        if method == ROW_EQUAL_ALL4:
            continue
        cells = _seed_center_cells(d1a._rows_for(rows, method))
        deltas: list[tuple[tuple[str, str], float]] = []
        by_center: dict[str, list[float]] = {}
        for key, value in cells.items():
            base = equal.get(key, math.nan)
            if math.isfinite(value) and math.isfinite(base):
                delta = value - base
                deltas.append((key, delta))
                by_center.setdefault(key[1], []).append(delta)
        values = [delta for _key, delta in deltas]
        lo, hi = _bootstrap_ci(values)
        out.append(
            {
                "method": method,
                "baseline_method": ROW_EQUAL_ALL4,
                "mean_paired_delta_bacc": nanmean(values),
                "median_paired_delta_bacc": float(np.median(values)) if values else math.nan,
                "positive_paired_cells": sum(1 for value in values if value > 0.0),
                "n_paired_cells": len(values),
                "per_center_paired_delta_json": json.dumps(
                    {center: nanmean(center_values) for center, center_values in sorted(by_center.items())},
                    sort_keys=True,
                ),
                "bootstrap_ci_low": lo,
                "bootstrap_ci_high": hi,
            }
        )
    return out


def _negative_control_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    out = []
    equal_stats = _method_stats(rows, ROW_EQUAL_ALL4)
    for method in (ROW_SHUFFLED, ROW_INVERSE):
        stats = _method_stats(rows, method)
        out.append(
            {
                "control_method": method,
                "control_center_equal_mean_bacc": stats["center_equal_mean_bacc"],
                "control_seed_cell_mean_bacc": stats["seed_cell_mean_bacc"],
                "delta_control_vs_equal_center_equal_bacc": _float(stats["center_equal_mean_bacc"]) - _float(equal_stats["center_equal_mean_bacc"]),
                "control_competitive": _float(stats["center_equal_mean_bacc"]) >= _float(equal_stats["center_equal_mean_bacc"]) - 0.005,
            }
        )
    return out


def _gap_summary_rows(rows: Sequence[Mapping[str, object]], cfg: PairedDenseAll4ReliabilityConfig) -> list[dict[str, object]]:
    d12_context = _context_metric(cfg.d1_2_artifact_root, "tables/decentralized_reliability_summary.csv")
    d14_context = _context_metric(cfg.d1_4_artifact_root, "tables/decentralized_reliability_top3_summary.csv")
    real_stats = _method_stats(rows, d1a.ROW_REAL_FEATURE_DENSE_REFERENCE)
    out = []
    for method in _reported_methods(rows):
        stats = _method_stats(rows, method)
        center_mean = _float(stats["center_equal_mean_bacc"])
        out.append(
            {
                "method": method,
                **stats,
                "delta_vs_paired_equal_all4_center_equal_bacc": center_mean - _float(_method_stats(rows, ROW_EQUAL_ALL4)["center_equal_mean_bacc"]),
                "delta_vs_d1_2_context_center_equal_bacc": center_mean - d12_context,
                "delta_vs_d1_4_context_center_equal_bacc": center_mean - d14_context,
                "delta_vs_real_feature_dense_reference_center_equal_bacc": center_mean - _float(real_stats["center_equal_mean_bacc"]),
            }
        )
    return out


def _decision(
    rows: Sequence[Mapping[str, object]],
    cfg: PairedDenseAll4ReliabilityConfig,
    *,
    leakage_status: str,
    invariant_pass: bool,
    paired_delta_rows: Sequence[Mapping[str, object]],
    negative_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    equal_stats = _method_stats(rows, ROW_EQUAL_ALL4)
    control_stats = [_method_stats(rows, method) for method in (ROW_SHUFFLED, ROW_INVERSE)]
    strongest_control = max((_float(stats["center_equal_mean_bacc"]) for stats in control_stats), default=math.nan)
    candidates = [ROW_RELIABILITY_ALL4_WEIGHTED, ROW_POOL_ONLY, ROW_BUDGET_ONLY, ROW_SHRINK025, ROW_SHRINK050]
    candidate_stats = {method: _method_stats(rows, method) for method in candidates}
    winner = max(
        candidates,
        key=lambda method: (
            _float(candidate_stats[method]["center_equal_mean_bacc"]),
            _float(candidate_stats[method]["seed_cell_mean_bacc"]),
            method,
        ),
    )
    winner_stats = candidate_stats[winner]
    deltas_by_method = {str(row["method"]): row for row in paired_delta_rows}
    winner_delta = deltas_by_method.get(winner, {})
    center_delta_json = str(winner_delta.get("per_center_paired_delta_json", "{}"))
    try:
        center_deltas = json.loads(center_delta_json)
    except json.JSONDecodeError:
        center_deltas = {}
    centers_improved = sum(1 for value in center_deltas.values() if _float(value) > 0.0)
    min_center_loss = _float(winner_stats["min_center_bacc"]) - _float(equal_stats["min_center_bacc"])
    gates = {
        "protocol_pass": leakage_status == "PASS",
        "pairing_invariant_pass": bool(invariant_pass),
        "all_centers_represented": int(winner_stats["n_heldout_centers"]) >= len(cfg.heldout_centers),
        "min_two_eligible_seeds_per_center": int(winner_stats["min_eligible_seeds_per_center"]) >= 2,
        "seed_cell_delta_ge_0p010": _float(winner_stats["seed_cell_mean_bacc"]) - _float(equal_stats["seed_cell_mean_bacc"]) >= 0.010,
        "center_equal_delta_ge_0p005": _float(winner_stats["center_equal_mean_bacc"]) - _float(equal_stats["center_equal_mean_bacc"]) >= 0.005,
        "negative_control_gap_ge_0p020": _float(winner_stats["center_equal_mean_bacc"]) - strongest_control >= 0.020,
        "seed_std_le_0p040": _float(winner_stats["seed_std_bacc"]) <= 0.040,
        "min_center_loss_ge_minus_0p010": min_center_loss >= -0.010,
        "centers_improved_ge_3": centers_improved >= 3,
    }
    passed = all(gates.values())
    equal_is_robust = _float(equal_stats["center_equal_mean_bacc"]) >= _float(winner_stats["center_equal_mean_bacc"])
    if leakage_status != "PASS":
        verdict = "PROTOCOL_FAIL"
    elif not invariant_pass:
        verdict = "PAIRING_INVARIANT_FAIL"
    elif passed and winner in {ROW_SHRINK025, ROW_SHRINK050}:
        verdict = "PAIRED_DENSE_ALL4_SHRINKAGE_PASS"
    elif passed:
        verdict = "PAIRED_DENSE_ALL4_RELIABILITY_PASS"
    elif equal_is_robust:
        verdict = "EQUAL_DENSE_ALL4_ROBUST_BASELINE"
    else:
        verdict = "PAIRED_DENSE_ALL4_DIAGNOSTIC_ONLY"
    return {
        "primary_verdict": verdict,
        "primary_method": cfg.primary_method,
        "best_reliability_method": winner,
        "leakage_status": leakage_status,
        "pairing_invariant_pass": bool(invariant_pass),
        "decision_gates_json": json.dumps(gates, sort_keys=True),
        "equal_all4_seed_cell_mean_bacc": equal_stats["seed_cell_mean_bacc"],
        "equal_all4_center_equal_mean_bacc": equal_stats["center_equal_mean_bacc"],
        "best_seed_cell_mean_bacc": winner_stats["seed_cell_mean_bacc"],
        "best_center_equal_mean_bacc": winner_stats["center_equal_mean_bacc"],
        "best_min_center_bacc": winner_stats["min_center_bacc"],
        "best_seed_std_bacc": winner_stats["seed_std_bacc"],
        "best_delta_vs_equal_seed_cell_bacc": _float(winner_stats["seed_cell_mean_bacc"]) - _float(equal_stats["seed_cell_mean_bacc"]),
        "best_delta_vs_equal_center_equal_bacc": _float(winner_stats["center_equal_mean_bacc"]) - _float(equal_stats["center_equal_mean_bacc"]),
        "best_delta_vs_strongest_negative_control_center_equal_bacc": _float(winner_stats["center_equal_mean_bacc"]) - strongest_control,
        "best_min_center_delta_vs_equal": min_center_loss,
        "best_centers_improved_vs_equal": centers_improved,
        "best_mean_paired_delta_bacc": winner_delta.get("mean_paired_delta_bacc", math.nan),
        "best_median_paired_delta_bacc": winner_delta.get("median_paired_delta_bacc", math.nan),
        "best_positive_paired_cells": winner_delta.get("positive_paired_cells", 0),
        "best_n_paired_cells": winner_delta.get("n_paired_cells", 0),
        "strongest_negative_control_center_equal_bacc": strongest_control,
        "eligible_heldout_centers": winner_stats["n_heldout_centers"],
        "min_eligible_seeds_per_center": winner_stats["min_eligible_seeds_per_center"],
        "claim_boundary": PROTOCOL_WORDING,
        "target_center_excluded_from_reliability": True,
        "target_eval_labels_used_for_reliability": False,
        "target_eval_labels_used_for_weighting": False,
        "target_eval_labels_used_for_selection": False,
    }


def _method_stats(rows: Sequence[Mapping[str, object]], method: str) -> dict[str, object]:
    cells = _seed_center_cells(d1a._rows_for(rows, method))
    by_seed: dict[str, list[float]] = {}
    by_center: dict[str, list[float]] = {}
    for (seed, center), value in cells.items():
        if math.isfinite(value):
            by_seed.setdefault(seed, []).append(value)
            by_center.setdefault(center, []).append(value)
    seed_means = {seed: nanmean(values) for seed, values in sorted(by_seed.items())}
    center_means = {center: nanmean(values) for center, values in sorted(by_center.items())}
    values = list(cells.values())
    return {
        "n_raw_rows": len(d1a._rows_for(rows, method)),
        "n_seed_center_cells": len(values),
        "n_experiment_seeds": len(by_seed),
        "n_heldout_centers": len(by_center),
        "min_eligible_seeds_per_center": min((len(values) for values in by_center.values()), default=0),
        "seed_cell_mean_bacc": nanmean(values),
        "center_equal_mean_bacc": nanmean(list(center_means.values())) if center_means else math.nan,
        "min_center_bacc": min(center_means.values()) if center_means else math.nan,
        "min_seed_center_bacc": min(values) if values else math.nan,
        "seed_std_bacc": d1._std(list(seed_means.values())),
        "seed_mean_bacc_json": json.dumps(seed_means, sort_keys=True),
        "center_mean_bacc_json": json.dumps(center_means, sort_keys=True),
    }


def _seed_center_cells(rows: Sequence[Mapping[str, object]]) -> dict[tuple[str, str], float]:
    groups: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        if row.get("status") == "ok":
            groups.setdefault((str(row["experiment_seed"]), str(row["heldout_center"])), []).append(row)
    return {key: d1._mean_field(subset, "bacc") for key, subset in sorted(groups.items())}


def _bootstrap_ci(values: Sequence[float]) -> tuple[float, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return math.nan, math.nan
    if len(finite) == 1:
        return finite[0], finite[0]
    rng = np.random.default_rng(20260525)
    draws = []
    arr = np.asarray(finite, dtype=float)
    for _idx in range(1000):
        sample = rng.choice(arr, size=arr.size, replace=True)
        draws.append(float(np.mean(sample)))
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _context_metric(root: Path | None, rel_table: str) -> float:
    if root is None:
        return math.nan
    path = root / rel_table
    if not path.exists():
        return math.nan
    import csv

    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return math.nan
    return _float(rows[0].get("center_equal_mean_bacc"))


def _reported_methods(rows: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    present = {str(row.get("prior_method")) for row in rows if row.get("prior_method")}
    return tuple(method for method in _method_order() if method in present)


def _method_order() -> tuple[str, ...]:
    return (
        ROW_EQUAL_ALL4,
        ROW_RELIABILITY_ALL4_WEIGHTED,
        ROW_POOL_ONLY,
        ROW_BUDGET_ONLY,
        ROW_SHRINK025,
        ROW_SHRINK050,
        ROW_SHUFFLED,
        ROW_INVERSE,
    )


def _is_midogpp(cfg: PairedDenseAll4ReliabilityConfig) -> bool:
    return normalize_domain_regime(cfg.domain_regime) == MIDOGPP_DOMAIN_REGIME


def _midogpp_contract_info(cfg: PairedDenseAll4ReliabilityConfig) -> MidogppContractInfo | None:
    if not _is_midogpp(cfg):
        return None
    if cfg.dataset_contract_artifact_root is None:
        raise ProtocolError("MIDOG++ config is missing dataset_contract_artifact_root.")
    return load_midogpp_contract_info(cfg.dataset_contract_artifact_root)


def _artifact_prefix(cfg: PairedDenseAll4ReliabilityConfig) -> str:
    return "dense_late_all_sources" if _is_midogpp(cfg) else "paired_dense_all4"


def _protocol_wording(cfg: PairedDenseAll4ReliabilityConfig) -> str:
    return MIDOGPP_PROTOCOL_WORDING if _is_midogpp(cfg) else PROTOCOL_WORDING


def _method_aliases(cfg: PairedDenseAll4ReliabilityConfig) -> dict[str, str]:
    if not _is_midogpp(cfg):
        return {}
    return {
        ROW_EQUAL_ALL4: ROW_EQUAL_ALL_SOURCES,
        ROW_RELIABILITY_ALL4_WEIGHTED: ROW_RELIABILITY_ALL_SOURCES_WEIGHTED,
        ROW_POOL_ONLY: ROW_POOL_ONLY_ALL_SOURCES,
        ROW_BUDGET_ONLY: ROW_BUDGET_ONLY_ALL_SOURCES,
        ROW_SHRINK025: ROW_SHRINK025_ALL_SOURCES,
        ROW_SHRINK050: ROW_SHRINK050_ALL_SOURCES,
        ROW_SHUFFLED: ROW_SHUFFLED_ALL_SOURCES,
        ROW_INVERSE: ROW_INVERSE_ALL_SOURCES,
        EQUAL_BUDGET_PAIRING_GROUP: EQUAL_BUDGET_ALL_SOURCES_PAIRING_GROUP,
        RELIABILITY_BUDGET_PAIRING_GROUP: RELIABILITY_BUDGET_ALL_SOURCES_PAIRING_GROUP,
        "PAIRED_DENSE_ALL4_SHRINKAGE_PASS": "DENSE_LATE_ALL_SOURCES_SHRINKAGE_PASS",
        "PAIRED_DENSE_ALL4_RELIABILITY_PASS": "DENSE_LATE_ALL_SOURCES_RELIABILITY_PASS",
        "EQUAL_DENSE_ALL4_ROBUST_BASELINE": "EQUAL_DENSE_ALL_SOURCES_ROBUST_BASELINE",
        "PAIRED_DENSE_ALL4_DIAGNOSTIC_ONLY": "DENSE_LATE_ALL_SOURCES_DIAGNOSTIC_ONLY",
    }


def _alias_text(value: object, aliases: Mapping[str, str]) -> object:
    if not isinstance(value, str):
        return value
    if value in aliases:
        return aliases[value]
    out = value
    replacements = (
        ("paired_dense_all4", "dense_late_all_sources"),
        ("paired_equal_all4", "dense_late_equal_all_sources"),
        ("equal_all4", "equal_all_sources"),
        ("dense_all4", "dense_all_sources"),
        ("all4", "all_sources"),
    )
    for old, new in replacements:
        out = out.replace(old, new)
    return out


def _alias_rows_for_output(
    cfg: PairedDenseAll4ReliabilityConfig,
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    aliases = _method_aliases(cfg)
    if not aliases:
        return [dict(row) for row in rows]
    out: list[dict[str, object]] = []
    for row in rows:
        aliased: dict[str, object] = {}
        for key, value in row.items():
            aliased_key = str(_alias_text(str(key), aliases))
            aliased[aliased_key] = _alias_text(value, aliases)
        out.append(aliased)
    return out


def _write_artifacts(
    root: Path,
    cfg: PairedDenseAll4ReliabilityConfig,
    *,
    matrix_rows: Sequence[Mapping[str, object]],
    gap_rows: Sequence[Mapping[str, object]],
    center_rows: Sequence[Mapping[str, object]],
    summary_rows: Sequence[Mapping[str, object]],
    reliability_rows: Sequence[Mapping[str, object]],
    weight_rows: Sequence[Mapping[str, object]],
    budget_rows: Sequence[Mapping[str, object]],
    excluded_rows: Sequence[Mapping[str, object]],
    invariant_rows: Sequence[Mapping[str, object]],
    paired_delta_rows: Sequence[Mapping[str, object]],
    negative_rows: Sequence[Mapping[str, object]],
    summary_manifest_rows: Sequence[Mapping[str, object]],
    diagnostic_rows: Sequence[Mapping[str, object]],
    late_rows: Sequence[Mapping[str, object]],
    real_feature_rows: Sequence[Mapping[str, object]],
    coverage_rows: Sequence[Mapping[str, object]],
    weak_rows: Sequence[Mapping[str, object]],
    nn_rows: Sequence[Mapping[str, object]],
    model_manifest_rows: Sequence[Mapping[str, object]],
    source_pool_rows: Sequence[Mapping[str, object]],
    decision: Mapping[str, object],
    leakage_status: str,
    protocol_violations: Sequence[str],
    target_expert_excluded: bool,
) -> None:
    prefix = _artifact_prefix(cfg)
    aliased_matrix = _alias_rows_for_output(cfg, matrix_rows)
    aliased_late = _alias_rows_for_output(cfg, late_rows)
    aliased_real = _alias_rows_for_output(cfg, real_feature_rows)
    aliased_summary_rows = _alias_rows_for_output(cfg, summary_rows)
    aliased_decision = _alias_rows_for_output(cfg, [decision])[0]
    write_csv_rows(root / "tables" / f"{prefix}_downstream_matrix.csv", aliased_matrix, columns=_matrix_columns())
    write_csv_rows(root / "tables" / f"{prefix}_gap_summary.csv", _alias_rows_for_output(cfg, gap_rows))
    write_csv_rows(root / "tables" / f"{prefix}_center_summary.csv", _alias_rows_for_output(cfg, center_rows))
    write_csv_rows(root / "tables" / f"{prefix}_summary.csv", aliased_summary_rows)
    write_csv_rows(root / "tables" / "source_reliability_manifest.csv", _alias_rows_for_output(cfg, reliability_rows))
    write_csv_rows(root / "tables" / "reliability_weight_manifest.csv", _alias_rows_for_output(cfg, weight_rows))
    write_csv_rows(root / "tables" / "realized_budget_table.csv", _alias_rows_for_output(cfg, budget_rows))
    write_csv_rows(root / "tables" / "excluded_cell_report.csv", excluded_rows)
    write_csv_rows(root / "tables" / "paired_generation_invariant_audit.csv", _alias_rows_for_output(cfg, invariant_rows))
    write_csv_rows(root / "tables" / "paired_delta_summary.csv", _alias_rows_for_output(cfg, paired_delta_rows))
    write_csv_rows(root / "tables" / "negative_control_summary.csv", _alias_rows_for_output(cfg, negative_rows))
    write_csv_rows(root / "tables" / "late_aggregation_matrix.csv", aliased_late, columns=_matrix_columns())
    write_csv_rows(root / "tables" / "real_feature_reference_matrix.csv", aliased_real, columns=_matrix_columns())
    write_csv_rows(root / "tables" / "generated_component_coverage_audit.csv", _alias_rows_for_output(cfg, coverage_rows))
    write_csv_rows(root / "tables" / "weak_source_audit.csv", _alias_rows_for_output(cfg, weak_rows))
    write_csv_rows(root / "tables" / "nearest_neighbor_memorization_audit.csv", _alias_rows_for_output(cfg, nn_rows))
    write_csv_rows(root / "tables" / "exported_source_summary_manifest.csv", summary_manifest_rows, columns=d1a._summary_manifest_columns())
    write_csv_rows(root / "tables" / "source_summary_diagnostics.csv", diagnostic_rows, columns=d1a._diagnostic_columns())
    write_csv_rows(root / "manifests" / f"{prefix}_prior_model_manifest.csv", _alias_rows_for_output(cfg, model_manifest_rows))
    write_csv_rows(root / "manifests" / f"{prefix}_source_pool_manifest.csv", source_pool_rows)
    leakage = build_leakage_report(
        target_support_labels_for_selection=False,
        target_eval_labels_for_scoring_only=True,
        target_expert_excluded=target_expert_excluded,
        oracle_rows_diagnostic_only=True,
        extra_violations=protocol_violations,
    )
    write_json(root / "reports" / "leakage_report.json", leakage.to_json_dict())
    write_json(root / "manifests" / "protocol_manifest.json", _protocol_manifest(cfg, target_expert_excluded))
    _write_decision_summary(root, aliased_decision, cfg=cfg, leakage_status=leakage_status)
    write_json(root / "run_config_resolved.yaml", _resolved_config(cfg))


def _matrix_columns() -> tuple[str, ...]:
    return d12._matrix_columns() + (
        "weight_rule",
        "budget_policy",
        "pairing_group",
        "generation_seed_method",
        "generation_bundle_key",
        "paired_equal_delta_bacc",
        "paired_equal_delta_macro_f1",
    )


def _protocol_manifest(cfg: PairedDenseAll4ReliabilityConfig, target_expert_excluded: bool) -> dict[str, object]:
    contract_info = _midogpp_contract_info(cfg)
    eligible = list(contract_info.eligible_domain_ids) if contract_info is not None else list(cfg.heldout_centers)
    source_count = len(eligible) - 1
    out = {
        "schema_version": "cvae_rebuild_paired_dense_all4_reliability_protocol_manifest_v1",
        "experiment_name": cfg.name,
        "experiment_type": "dense_late_all_sources_reliability" if _is_midogpp(cfg) else "paired_dense_all4_reliability_confirmation",
        "domain_regime": normalize_domain_regime(cfg.domain_regime),
        "eligible_domain_ids": eligible,
        "expected_source_count": int(source_count),
        "primary_variant": cfg.primary_variant,
        "primary_method": cfg.primary_method,
        "target_support_features_for_selection": False,
        "target_support_labels_for_selection": False,
        "target_eval_labels_for_scoring_only": True,
        "target_expert_excluded": bool(target_expert_excluded),
        "target_center_excluded_from_reliability": True,
        "target_eval_labels_used_for_reliability": False,
        "target_eval_labels_used_for_weighting": False,
        "target_eval_labels_used_for_selection": False,
        "dense_all4_fixed_inclusion": not _is_midogpp(cfg),
        "dense_all_sources_fixed_inclusion": _is_midogpp(cfg),
        "domain_4_excluded": "4" not in set(eligible),
        "all_eligible_heldouts_complete": True,
        "class_order_match": True,
        "class_label_names": {"0": "hard_negative", "1": "mitotic"},
        "cache_seed": list(cfg.experiment_seeds),
        "cache_root_matches_midogpp_root": "pathology_embeddings_midogpp_annotation_patch_v1" in cfg.feature_cache_root.as_posix()
        if _is_midogpp(cfg)
        else False,
        "top_k_selection_enabled": False,
        "heldout_excluded_reliability_transform_locked": True,
        "inverse_reliability_definition": "rank_reversal_matched_entropy",
        "budget_rounding_rule": "largest_remainder_after_minimum_floor_exact_total_128",
        "paired_generation_invariants_required": True,
        "oracle_rows_diagnostic_only": True,
        "protocol_wording": PROTOCOL_WORDING,
        "claim_boundary": (
            "compatibility-proxy audit for dense generated-embedding aggregation; "
            "not sparse expert selection and not target-conditioned routing"
        ),
    }
    if contract_info is not None:
        out.update(
            {
                "dataset_contract_artifact_root": str(contract_info.artifact_root),
                "dataset_contract_fingerprints": contract_info.fingerprints,
                "selected_domain_axis": contract_info.selected_domain_axis,
                "ineligible_domain_ids": list(contract_info.ineligible_domain_ids),
            }
        )
    out["protocol_wording"] = _protocol_wording(cfg)
    return out


def _write_decision_summary(
    root: Path,
    decision: Mapping[str, object],
    *,
    cfg: PairedDenseAll4ReliabilityConfig,
    leakage_status: str,
) -> None:
    verdict = str(decision.get("primary_verdict", ""))
    title = "Dense-Late All-Sources MIDOG++ Reliability Pilot" if _is_midogpp(cfg) else "Paired Dense-All4 Reliability Confirmation"
    equal_key = "equal_all_sources_center_equal_mean_bacc" if _is_midogpp(cfg) else "equal_all4_center_equal_mean_bacc"
    text = "\n".join(
        [
            f"# {title}",
            "",
            "## Summary",
            "",
            f"- Primary method: `{decision.get('primary_method', PRIMARY_PAIRED_METHOD)}`",
            f"- Best reliability method: `{decision.get('best_reliability_method', '')}`",
            f"- Primary verdict: `{verdict}`",
            f"- Equal all-source center-equal BACC: {_format_float(decision.get(equal_key))}",
            f"- Best center-equal BACC: {_format_float(decision.get('best_center_equal_mean_bacc'))}",
            f"- Best delta vs equal center-equal BACC: {_format_float(decision.get('best_delta_vs_equal_center_equal_bacc'))}",
            f"- Best mean paired delta BACC: {_format_float(decision.get('best_mean_paired_delta_bacc'))}",
            f"- Best min center BACC: {_format_float(decision.get('best_min_center_bacc'))}",
            f"- Best seed std BACC: {_format_float(decision.get('best_seed_std_bacc'))}",
            f"- Best delta vs strongest negative control: {_format_float(decision.get('best_delta_vs_strongest_negative_control_center_equal_bacc'))}",
            f"- Pairing invariant pass: `{decision.get('pairing_invariant_pass', False)}`",
            f"- Leakage status: `{leakage_status}`",
            "",
            "## Protocol Boundary",
            "",
            _protocol_wording(cfg),
            "",
            "Heldout target labels are final scoring only. Reliability is recomputed and manifested per heldout center with the target center excluded.",
            "Do not claim sparse routing or expert selection from this dense all-source experiment.",
            "",
            "## Thesis Interpretation",
            "",
            _interpretation(verdict),
            "",
        ]
    )
    path = root / "reports" / "decision_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _interpretation(verdict: str) -> str:
    if verdict == "DENSE_LATE_ALL_SOURCES_SHRINKAGE_PASS":
        return "Source-only reliability contains compatibility signal in the MIDOG++ all-source pilot, but this seed-42 result is not a stability claim."
    if verdict == "DENSE_LATE_ALL_SOURCES_RELIABILITY_PASS":
        return "Source-only reliability is promising for MIDOG++ dense all-source generated-embedding aggregation under the pilot protocol."
    if verdict == "EQUAL_DENSE_ALL_SOURCES_ROBUST_BASELINE":
        return "Equal dense all-source aggregation is the robust MIDOG++ pilot baseline; current source-only reliability is insufficient as a compatibility proxy."
    if verdict == "PAIRED_DENSE_ALL4_SHRINKAGE_PASS":
        return "Source-only reliability contains compatibility signal, but raw reliability is noisy; conservative shrinkage is the defensible dense aggregation strategy."
    if verdict == "PAIRED_DENSE_ALL4_RELIABILITY_PASS":
        return "Source-only reliability is a valid compatibility proxy for dense generated-embedding aggregation under the locked paired audit."
    if verdict == "EQUAL_DENSE_ALL4_ROBUST_BASELINE":
        return "Equal dense all-source aggregation is the robust generated-embedding baseline; current source-only reliability is insufficient as a compatibility proxy."
    return "The result is diagnostic only under the locked decision gates."


def _resolved_config(cfg: PairedDenseAll4ReliabilityConfig) -> dict[str, object]:
    return {
        "name": cfg.name,
        "artifact_root": str(cfg.artifact_root),
        "repair_artifact_root": str(cfg.repair_artifact_root),
        "d1_2_artifact_root": "" if cfg.d1_2_artifact_root is None else str(cfg.d1_2_artifact_root),
        "d1_4_artifact_root": "" if cfg.d1_4_artifact_root is None else str(cfg.d1_4_artifact_root),
        "feature_cache_root": str(cfg.feature_cache_root),
        "dataset_contract_artifact_root": "" if cfg.dataset_contract_artifact_root is None else str(cfg.dataset_contract_artifact_root),
        "cache_report_path": "" if cfg.cache_report_path is None else str(cfg.cache_report_path),
        "backbone": cfg.backbone,
        "domain_regime": normalize_domain_regime(cfg.domain_regime),
        "strict_full_run_matrix": cfg.strict_full_run_matrix,
        "strict_available_seed_domain_coverage": cfg.strict_available_seed_domain_coverage,
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
        "reliability_floor_score": cfg.reliability_floor_score,
        "reliability_epsilon": cfg.reliability_epsilon,
        "shrinkage_values": list(cfg.shrinkage_values),
        "primary_pooling": cfg.primary_pooling,
        "classifier": {
            "type": cfg.classifier_type,
            "solver": cfg.classifier_solver,
            "C": cfg.classifier_c,
            "max_iter": cfg.classifier_max_iter,
            "class_weight": cfg.classifier_class_weight,
            "classifier_seed": cfg.classifier_seed,
        },
    }
