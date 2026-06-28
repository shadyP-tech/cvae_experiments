from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.downstream import (
    PredictionBundle,
    evaluate_probability_predictions,
    fit_locked_logistic_classifier,
    geometric_probability_pool,
)
from data.features import load_feature_cache, select_rows
from core.metrics import nanmean
from experiments.preservation.preservation import _hash_array
from experiments.preservation.preservation_repair import (
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
from experiments.preservation.preservation_sampling import DIAGNOSTIC_SELECTION, PRIMARY_SELECTION, _manifest_row, _per_source_variant, _runtime_source
from core.protocol import ProtocolError, assert_candidate_pool, build_leakage_report
from core.reporting import prepare_artifact_dirs, write_csv_rows, write_json
from experiments.source_union.source_union_gmm_prior import _nearest_neighbor_row
from data.splits import candidate_experts

from experiments.decentralized import decentralized_adaptive_gmm_prior as d1a
from experiments.decentralized import decentralized_k16_gmm_prior as d1
from experiments.decentralized import decentralized_reliability_weighted_gmm_prior as d12
from experiments.decentralized import decentralized_reliability_top3_gmm_prior as d14


SOURCE_INNER_TRANSFER_NAME = "virchow2_cvae_decentralized_source_inner_transfer_top3_gmm_prior_v1"
PRIMARY_SOURCE_INNER_TRANSFER_METHOD = "decentralized_source_inner_transfer_top3_geom_confirmation"
ROW_EQUAL_ALL4 = "decentralized_equal_all4_geom_reference"
ROW_RELIABILITY_TOP3_REFERENCE = "decentralized_reliability_top3_geom_reference"
ROW_RELIABILITY_ALL4 = "decentralized_reliability_all4_weighted_geom_reference"
ROW_INDIVIDUAL_TRANSFER_TOP3 = "decentralized_source_inner_individual_transfer_top3_geom_diagnostic"
ROW_TRANSFER_TOP2 = "decentralized_source_inner_transfer_top2_geom_diagnostic"
ROW_TRANSFER_TOP4 = "decentralized_source_inner_transfer_top4_geom_diagnostic"
ROW_DROP_ONE_MEAN = "exhaustive_drop_one_top3_mean_reference"
ROW_DROP_ONE_ORACLE = "exhaustive_drop_one_top3_oracle_reference"
ROW_SINGLE_MEAN = "per_source_adaptive_k_single_expert_mean_reference"
ROW_SINGLE_ORACLE = "per_source_adaptive_k_single_expert_oracle_reference"
ROW_REAL_FEATURE_DENSE_REFERENCE = "real_source_embedding_classifier_dense_reference"
ROW_SHUFFLED_SCORE_CONTROL = "decentralized_source_inner_transfer_shuffled_score_control"
ROW_SHUFFLED_SUMMARY_CONTROL = "decentralized_source_inner_transfer_shuffled_summary_control"
ROW_SHUFFLED_LABEL_CONTROL = "decentralized_source_inner_transfer_shuffled_label_control"
PROTOCOL_WORDING = (
    "Source-inner transfer scores may be computed locally or shared only as aggregate score matrices. "
    "No raw source embeddings, source images, or heldout target data are shared. This remains a "
    "data-minimizing summary/score-exchange protocol, not a formal differential privacy claim."
)
DROP_ONE_CLAIM_BOUNDARY = (
    "Top-3 over four candidates is drop-one source selection; claims are limited to small-pool "
    "sparse source exclusion."
)


@dataclass(frozen=True)
class DecentralizedSourceInnerTransferTop3Config:
    name: str
    artifact_root: Path
    repair_artifact_root: Path
    feature_cache_root: Path
    backbone: str
    experiment_seeds: tuple[int, ...]
    heldout_centers: tuple[str, ...]
    replicate_seeds: tuple[int, ...]
    synthetic_per_class_total: int
    min_per_source_per_class: int
    top_k_sources: int
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
    classifier_type: str
    classifier_solver: str
    classifier_c: float
    classifier_max_iter: int
    classifier_class_weight: str
    classifier_seed: int | None

    @property
    def max_local_gmm_components_per_source_class(self) -> int:
        return max(self.candidate_components_per_source_class)


@dataclass(frozen=True)
class SubsetScore:
    experiment_seed: int
    heldout_center: str
    replicate_seed: int
    subset_sources: tuple[str, ...]
    dropped_source: str
    subset_size: int
    min_pseudo_bacc: float
    mean_pseudo_bacc: float
    mean_pseudo_macro_f1: float
    pseudo_target_count: int
    selected_by_source_inner: bool = False
    selected_by_shuffled_score: bool = False

    @property
    def sort_key(self) -> tuple[float, float, float, str]:
        return (
            float(self.min_pseudo_bacc),
            float(self.mean_pseudo_bacc),
            float(self.mean_pseudo_macro_f1),
            _drop_tie_value(self.dropped_source),
        )


def load_decentralized_source_inner_transfer_top3_gmm_prior_config(
    path: str | Path,
) -> DecentralizedSourceInnerTransferTop3Config:
    source = Path(path).resolve()
    data = _load_mapping(source)
    base_dir = source.parents[2] if len(source.parents) >= 3 else source.parent
    return parse_decentralized_source_inner_transfer_top3_gmm_prior_config(data, base_dir=base_dir)


def parse_decentralized_source_inner_transfer_top3_gmm_prior_config(
    data: Mapping[str, Any],
    *,
    base_dir: str | Path = ".",
) -> DecentralizedSourceInnerTransferTop3Config:
    base = Path(base_dir)
    experiment = _mapping(data, "experiment")
    inputs = _mapping(data, "inputs")
    run = _mapping(data, "run_matrix")
    if "support_size" in run or "support_seeds" in run:
        raise ProtocolError("D1.5 primary must not configure or consume target support rows.")
    generation = _mapping(data, "generation")
    gmm = _mapping(data, "source_inner_transfer_top3_gmm_prior")
    classifier = _mapping(data, "classifier")
    cfg = DecentralizedSourceInnerTransferTop3Config(
        name=str(experiment["name"]),
        artifact_root=_path(base, str(experiment["artifact_root"])),
        repair_artifact_root=_path(base, str(inputs["repair_artifact_root"])),
        feature_cache_root=_path(base, str(inputs["feature_cache_root"])),
        backbone=str(inputs.get("backbone", "")),
        experiment_seeds=tuple(int(v) for v in run["experiment_seeds"]),
        heldout_centers=tuple(str(v) for v in run["heldout_centers"]),
        replicate_seeds=tuple(int(v) for v in run["replicate_seeds"]),
        synthetic_per_class_total=int(generation["synthetic_per_class_total"]),
        min_per_source_per_class=int(generation["min_per_source_per_class"]),
        top_k_sources=int(generation["top_k_sources"]),
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
        classifier_type=str(classifier["type"]),
        classifier_solver=str(classifier["solver"]),
        classifier_c=float(classifier["C"]),
        classifier_max_iter=int(classifier["max_iter"]),
        classifier_class_weight=str(classifier["class_weight"]),
        classifier_seed=None if classifier.get("classifier_seed") is None else int(classifier["classifier_seed"]),
    )
    validate_decentralized_source_inner_transfer_top3_gmm_prior_config(cfg)
    return cfg


def validate_decentralized_source_inner_transfer_top3_gmm_prior_config(
    cfg: DecentralizedSourceInnerTransferTop3Config,
) -> None:
    if cfg.name != SOURCE_INNER_TRANSFER_NAME:
        raise ProtocolError(f"D1.5 experiment name must be {SOURCE_INNER_TRANSFER_NAME!r}.")
    if cfg.backbone != "virchow2":
        raise ProtocolError("D1.5 is locked to backbone=virchow2.")
    if cfg.primary_variant != PRIMARY_VARIANT:
        raise ProtocolError(f"primary_variant must be {PRIMARY_VARIANT!r}.")
    if cfg.primary_method != PRIMARY_SOURCE_INNER_TRANSFER_METHOD:
        raise ProtocolError(f"primary_method must be {PRIMARY_SOURCE_INNER_TRANSFER_METHOD!r}.")
    if cfg.candidate_components_per_source_class != (4, 3, 2, 1):
        raise ProtocolError("candidate_components_per_source_class must be locked to [4, 3, 2, 1].")
    if len(cfg.heldout_centers) != 5:
        raise ProtocolError("D1.5 composition expects exactly five centers, giving four source experts per fold.")
    if cfg.top_k_sources != 3:
        raise ProtocolError("D1.5 top_k_sources must be locked to 3.")
    if cfg.source_weighting != "source_inner_transfer_top3":
        raise ProtocolError("source_weighting must be source_inner_transfer_top3.")
    if cfg.gmm_covariance_type != "diag":
        raise ProtocolError("gmm_covariance_type must be diag.")
    if cfg.primary_pooling != "geometric":
        raise ProtocolError("D1.5 primary_pooling must be geometric.")
    if cfg.synthetic_per_class_total != 128 or cfg.min_per_source_per_class != 8:
        raise ProtocolError("D1.5 synthetic budget must be 128 total with min_per_source_per_class=8.")
    if min(cfg.min_samples_per_component, cfg.gmm_n_init, cfg.gmm_max_iter) < 1:
        raise ProtocolError("GMM counts and iteration settings must be positive.")
    if min(cfg.gmm_reg_covar, cfg.min_component_weight, cfg.variance_floor, cfg.reliability_floor_score) <= 0.0:
        raise ProtocolError("GMM and reliability floors must be positive.")
    if cfg.classifier_type != "sklearn_logistic_regression":
        raise ProtocolError("classifier.type must be sklearn_logistic_regression.")
    if cfg.classifier_solver != "lbfgs" or cfg.classifier_c != 1.0 or cfg.classifier_max_iter != 2000:
        raise ProtocolError("Classifier solver/C/max_iter must remain locked.")
    if cfg.classifier_class_weight != "balanced" or cfg.classifier_seed is not None:
        raise ProtocolError("Classifier must use class_weight=balanced and classifier_seed=null.")


def run_decentralized_source_inner_transfer_top3_gmm_prior(
    cfg: DecentralizedSourceInnerTransferTop3Config,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    root = prepare_artifact_dirs(Path(artifact_root) if artifact_root is not None else cfg.artifact_root)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "summaries").mkdir(parents=True, exist_ok=True)

    matrix_rows: list[dict[str, object]] = []
    summary_manifest_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    reliability_rows: list[dict[str, object]] = []
    source_inner_rows: list[dict[str, object]] = []
    subset_score_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    drop_frequency_rows: list[dict[str, object]] = []
    drop_one_target_rows: list[dict[str, object]] = []
    late_rows: list[dict[str, object]] = []
    real_feature_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    weak_rows: list[dict[str, object]] = []
    nn_rows: list[dict[str, object]] = []
    model_manifest_rows: list[dict[str, object]] = []
    protocol_violations: list[str] = []
    target_expert_excluded = True

    repair_cfg = d1._repair_runtime_config(cfg, root)
    per_source_variant = _per_source_variant()

    try:
        for experiment_seed in cfg.experiment_seeds:
            train_cache = load_feature_cache(_existing_cache_path(cfg.feature_cache_root, seed=experiment_seed, split="train"))
            test_cache = load_feature_cache(_existing_cache_path(cfg.feature_cache_root, seed=experiment_seed, split="test"))
            per_source_runtime: dict[str, object] = {}
            largest_summaries: dict[tuple[str, int], d1a.AdaptiveSourceLocalSummary] = {}
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

                largest, _bic = d1a._fit_and_export_source_summaries(
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
                for summary in shuffled:
                    shuffled_summaries[(summary.source_center, summary.class_label)] = summary
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

                source_eval_by_center = _source_eval_cache(test_cache, candidates)
                source_eval_error = _source_eval_error(source_eval_by_center)

                for replicate_seed in cfg.replicate_seeds:
                    rels = {source: reliability[(int(experiment_seed), int(replicate_seed), str(source))] for source in candidates}
                    all4_weighted = d12._weight_plan(cfg, candidates, rels, mode="linear")
                    equal_all4 = _equal_plan(cfg, candidates, score_label="equal_all4")
                    reliability_top3 = _equal_plan(
                        cfg,
                        d14._select_topk_reliable(candidates, rels, k=cfg.top_k_sources),
                        score_label="source_local_reliability_top3",
                    )

                    if eval_error or source_eval_error:
                        matrix_rows.extend(
                            _ineligible_rows(
                                cfg,
                                experiment_seed=int(experiment_seed),
                                heldout_center=str(heldout_center),
                                replicate_seed=int(replicate_seed),
                                candidates=candidates,
                                status="ineligible",
                                error_message=eval_error or source_eval_error,
                            )
                        )
                        continue

                    subset_scores, transfer_rows = _source_inner_subset_scores(
                        cfg,
                        per_source_runtime=per_source_runtime,
                        summaries=largest_summaries,
                        candidates=candidates,
                        source_eval_by_center=source_eval_by_center,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        replicate_seed=int(replicate_seed),
                        subset_size=cfg.top_k_sources,
                        prior_method=PRIMARY_SOURCE_INNER_TRANSFER_METHOD,
                    )
                    primary_subset = _best_subset(subset_scores)
                    shuffled_subset = _shuffled_score_subset(
                        subset_scores,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        replicate_seed=int(replicate_seed),
                    )
                    individual_subset = _individual_transfer_subset(
                        cfg,
                        transfer_rows,
                        candidates=candidates,
                        k=cfg.top_k_sources,
                    )
                    top2_subset = _best_subset(
                        _source_inner_subset_scores(
                            cfg,
                            per_source_runtime=per_source_runtime,
                            summaries=largest_summaries,
                            candidates=candidates,
                            source_eval_by_center=source_eval_by_center,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            replicate_seed=int(replicate_seed),
                            subset_size=2,
                            prior_method=ROW_TRANSFER_TOP2,
                        )[0]
                    )
                    top4_subset = _best_subset(
                        _source_inner_subset_scores(
                            cfg,
                            per_source_runtime=per_source_runtime,
                            summaries=largest_summaries,
                            candidates=candidates,
                            source_eval_by_center=source_eval_by_center,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            replicate_seed=int(replicate_seed),
                            subset_size=4,
                            prior_method=ROW_TRANSFER_TOP4,
                        )[0]
                    )
                    source_inner_rows.extend(transfer_rows)
                    subset_score_rows.extend(
                        _subset_score_rows(
                            subset_scores,
                            selected_subset=primary_subset.subset_sources,
                            shuffled_subset=shuffled_subset.subset_sources,
                        )
                    )

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
                    ref_row = _extend_source_inner_row(ref_row)
                    ref_row["prior_method"] = ROW_REAL_FEATURE_DENSE_REFERENCE
                    real_late = [_extend_source_inner_row(row) for row in real_late]
                    for row in real_late:
                        row["prior_method"] = ROW_REAL_FEATURE_DENSE_REFERENCE
                    real_feature_rows.append(ref_row)
                    matrix_rows.append(ref_row)
                    late_rows.extend(real_late)

                    equal_rows, equal_late, coverage, weak, nn = _evaluate_variant(
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
                        weight_plan=equal_all4,
                        prior_method=ROW_EQUAL_ALL4,
                        pooling_rule="geometric",
                        source_weighting="equal_source_mass",
                        selection_source=DIAGNOSTIC_SELECTION,
                        claim_role="equal_all4_reference",
                    )
                    matrix_rows.extend(equal_rows)
                    late_rows.extend(equal_late)
                    coverage_rows.extend(coverage)
                    weak_rows.extend(weak)
                    nn_rows.extend(nn)

                    all4_rows, all4_late, coverage, weak, nn = _evaluate_variant(
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
                        weight_plan=all4_weighted,
                        prior_method=ROW_RELIABILITY_ALL4,
                        pooling_rule="weighted_geometric",
                        source_weighting="source_local_reliability_all4_weighted",
                        selection_source=DIAGNOSTIC_SELECTION,
                        claim_role="reliability_all4_weighted_reference",
                    )
                    matrix_rows.extend(all4_rows)
                    late_rows.extend(all4_late)
                    coverage_rows.extend(coverage)
                    weak_rows.extend(weak)
                    nn_rows.extend(nn)

                    target_subset_rows: list[dict[str, object]] = []
                    target_subset_audit_rows: list[dict[str, object]] = []
                    for subset in _drop_one_subsets(candidates, cfg.top_k_sources):
                        subset_plan = _equal_plan(cfg, subset, score_label="exhaustive_drop_one")
                        rows, subset_late, coverage, weak, nn = _evaluate_variant(
                            cfg,
                            per_source_runtime=per_source_runtime,
                            candidates=subset,
                            summaries=largest_summaries,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            replicate_seed=int(replicate_seed),
                            eval_raw=eval_raw,
                            eval_labels=eval_labels,
                            real_feature_bacc=_float(ref_row["bacc"]),
                            weight_plan=subset_plan,
                            prior_method="exhaustive_drop_one_top3_subset_diagnostic",
                            pooling_rule="geometric",
                            source_weighting="exhaustive_drop_one_top3_equal",
                            selection_source=DIAGNOSTIC_SELECTION,
                            claim_role="drop_one_subset_target_utility_diagnostic",
                        )
                        row = rows[0]
                        target_subset_rows.append(row)
                        audit_row = _drop_one_target_row(row, subset, candidates, subset_scores)
                        target_subset_audit_rows.append(audit_row)
                        drop_one_target_rows.append(audit_row)
                        late_rows.extend(subset_late)
                        coverage_rows.extend(coverage)
                        weak_rows.extend(weak)
                        nn_rows.extend(nn)

                    mean_oracle_rows = _drop_one_mean_oracle_rows(
                        cfg,
                        target_subset_rows,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        replicate_seed=int(replicate_seed),
                        candidates=candidates,
                        summaries=largest_summaries,
                        real_feature_bacc=_float(ref_row["bacc"]),
                    )
                    matrix_rows.extend(mean_oracle_rows)

                    diagnostics = (
                        (
                            PRIMARY_SOURCE_INNER_TRANSFER_METHOD,
                            primary_subset.subset_sources,
                            PRIMARY_SELECTION,
                            "source_inner_transfer_top3_equal",
                            "primary_source_inner_transfer_drop_one_composition",
                        ),
                        (
                            ROW_RELIABILITY_TOP3_REFERENCE,
                            tuple(reliability_top3["sources"]),
                            DIAGNOSTIC_SELECTION,
                            "source_local_reliability_top3_equal",
                            "reliability_top3_reference",
                        ),
                        (
                            ROW_INDIVIDUAL_TRANSFER_TOP3,
                            individual_subset,
                            DIAGNOSTIC_SELECTION,
                            "source_inner_individual_transfer_top3_equal",
                            "individual_source_inner_transfer_top3_diagnostic",
                        ),
                        (
                            ROW_TRANSFER_TOP2,
                            top2_subset.subset_sources,
                            DIAGNOSTIC_SELECTION,
                            "source_inner_transfer_top2_equal",
                            "source_inner_transfer_top2_diagnostic",
                        ),
                        (
                            ROW_TRANSFER_TOP4,
                            top4_subset.subset_sources,
                            DIAGNOSTIC_SELECTION,
                            "source_inner_transfer_top4_equal",
                            "source_inner_transfer_top4_diagnostic",
                        ),
                        (
                            ROW_SHUFFLED_SCORE_CONTROL,
                            shuffled_subset.subset_sources,
                            DIAGNOSTIC_SELECTION,
                            "shuffled_source_inner_transfer_score_top3_equal",
                            "negative_control",
                        ),
                    )
                    primary_eval_row = None
                    for method, subset, selection_source, source_weighting, role in diagnostics:
                        rows, subset_late, coverage, weak, nn = _evaluate_variant(
                            cfg,
                            per_source_runtime=per_source_runtime,
                            candidates=tuple(subset),
                            summaries=largest_summaries,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            replicate_seed=int(replicate_seed),
                            eval_raw=eval_raw,
                            eval_labels=eval_labels,
                            real_feature_bacc=_float(ref_row["bacc"]),
                            weight_plan=_equal_plan(cfg, tuple(subset), score_label=source_weighting),
                            prior_method=method,
                            pooling_rule="geometric",
                            source_weighting=source_weighting,
                            selection_source=selection_source,
                            claim_role=role,
                        )
                        if method == PRIMARY_SOURCE_INNER_TRANSFER_METHOD:
                            primary_eval_row = rows[0]
                        matrix_rows.extend(rows)
                        late_rows.extend(subset_late)
                        coverage_rows.extend(coverage)
                        weak_rows.extend(weak)
                        nn_rows.extend(nn)

                    _append_single_source_references(
                        cfg,
                        matrix_rows,
                        equal_late,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        replicate_seed=int(replicate_seed),
                        candidates=candidates,
                        summaries=largest_summaries,
                        real_feature_bacc=_float(ref_row["bacc"]),
                    )

                    selected_oracle = _oracle_source(equal_late)
                    selection_rows.append(
                        _selection_row(
                            primary_subset,
                            equal_late,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            replicate_seed=int(replicate_seed),
                            selection_rule="source_inner_transfer_top3",
                        )
                    )
                    drop_frequency_rows.append(
                        _drop_frequency_row(
                            primary_subset,
                            target_subset_audit_rows,
                            subset_scores,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            replicate_seed=int(replicate_seed),
                            selected_oracle_source=selected_oracle,
                        )
                    )

                    for prior_method, summaries, control_mode in (
                        (ROW_SHUFFLED_SUMMARY_CONTROL, largest_summaries, "class_flip"),
                        (ROW_SHUFFLED_LABEL_CONTROL, shuffled_summaries, "normal"),
                    ):
                        rows, subset_late, coverage, weak, nn = _evaluate_variant(
                            cfg,
                            per_source_runtime=per_source_runtime,
                            candidates=primary_subset.subset_sources,
                            summaries=summaries,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            replicate_seed=int(replicate_seed),
                            eval_raw=eval_raw,
                            eval_labels=eval_labels,
                            real_feature_bacc=_float(ref_row["bacc"]),
                            weight_plan=_equal_plan(cfg, primary_subset.subset_sources, score_label="control"),
                            prior_method=prior_method,
                            pooling_rule="geometric",
                            source_weighting="source_inner_transfer_top3_equal",
                            selection_source=DIAGNOSTIC_SELECTION,
                            claim_role="negative_control",
                            control_mode=control_mode,
                        )
                        matrix_rows.extend(rows)
                        late_rows.extend(subset_late)
                        coverage_rows.extend(coverage)
                        weak_rows.extend(weak)
                        nn_rows.extend(nn)
                    if primary_eval_row is not None:
                        _annotate_drop_one_oracle_gap(drop_one_target_rows, primary_eval_row, primary_subset)
    except ProtocolError as exc:
        protocol_violations.append(str(exc))

    leakage = build_leakage_report(
        target_support_labels_for_selection=False,
        target_eval_labels_for_scoring_only=True,
        target_expert_excluded=target_expert_excluded,
        oracle_rows_diagnostic_only=True,
        extra_violations=protocol_violations,
    )
    _populate_deltas(matrix_rows)
    _populate_drop_frequencies(drop_frequency_rows)
    gap_rows = [dict(row) for row in matrix_rows if row.get("status") == "ok"]
    centerwise_rows = _centerwise_delta_rows(matrix_rows)
    decision = _decision(
        matrix_rows,
        cfg,
        leakage_status=leakage.status,
        selection_rows=selection_rows,
        centerwise_rows=centerwise_rows,
        subset_score_rows=subset_score_rows,
        drop_one_target_rows=drop_one_target_rows,
    )
    _write_artifacts(
        root,
        cfg,
        matrix_rows=matrix_rows,
        gap_rows=gap_rows,
        summary_manifest_rows=summary_manifest_rows,
        diagnostic_rows=diagnostic_rows,
        reliability_rows=reliability_rows,
        source_inner_rows=source_inner_rows,
        subset_score_rows=subset_score_rows,
        selection_rows=selection_rows,
        drop_frequency_rows=drop_frequency_rows,
        drop_one_target_rows=drop_one_target_rows,
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


def _equal_plan(
    cfg: DecentralizedSourceInnerTransferTop3Config,
    sources: Sequence[str],
    *,
    score_label: str,
) -> dict[str, object]:
    sources_tuple = tuple(str(source) for source in sources)
    weight = 1.0 / float(len(sources_tuple))
    weights = {source: weight for source in sources_tuple}
    budgets = {source: int(value) for source, value in zip(sources_tuple, d1._balanced_counts(cfg.synthetic_per_class_total, len(sources_tuple)))}
    scores = {source: 1.0 for source in sources_tuple}
    plan = d12._with_weight_diagnostics(sources_tuple, weights, budgets, scores)
    plan["score_label"] = score_label
    return plan


def _drop_one_subsets(candidates: Sequence[str], k: int) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(sorted(subset)) for subset in combinations(tuple(str(v) for v in candidates), int(k)))


def _source_inner_subset_scores(
    cfg: DecentralizedSourceInnerTransferTop3Config,
    *,
    per_source_runtime: Mapping[str, object],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    candidates: Sequence[str],
    source_eval_by_center: Mapping[str, tuple[object, Sequence[int]]],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    subset_size: int,
    prior_method: str,
) -> tuple[list[SubsetScore], list[dict[str, object]]]:
    subset_scores: list[SubsetScore] = []
    transfer_rows: list[dict[str, object]] = []
    for subset in _drop_one_subsets(candidates, int(subset_size)):
        plan = _equal_plan(cfg, subset, score_label=f"source_inner_transfer_top{subset_size}")
        pseudo_baccs: list[float] = []
        pseudo_macros: list[float] = []
        for pseudo_target in candidates:
            used_sources = tuple(source for source in subset if source != str(pseudo_target))
            if not used_sources:
                continue
            eval_raw, eval_labels = source_eval_by_center[str(pseudo_target)]
            bundles = []
            for source in used_sources:
                bundle, generated_hash, prediction_hash, single_bacc, single_macro = _source_inner_bundle(
                    cfg,
                    per_source_runtime=per_source_runtime,
                    summaries=summaries,
                    source_center=source,
                    eval_raw=eval_raw,
                    eval_labels=eval_labels,
                    budget_per_class=int(plan["budgets"][source]),
                    experiment_seed=int(experiment_seed),
                    heldout_center=str(heldout_center),
                    replicate_seed=int(replicate_seed),
                    prior_method=prior_method,
                    control_mode="normal",
                )
                transfer_rows.append(
                    {
                        "experiment_seed": int(experiment_seed),
                        "heldout_center": str(heldout_center),
                        "replicate_seed": int(replicate_seed),
                        "subset_size": int(subset_size),
                        "selected_subset": "|".join(subset),
                        "dropped_source": _dropped_source(candidates, subset),
                        "source_expert": str(source),
                        "pseudo_target_source": str(pseudo_target),
                        "source_expert_used_in_subset": True,
                        "source_expert_equals_pseudo_target": str(source) == str(pseudo_target),
                        "source_inner_single_bacc": single_bacc,
                        "source_inner_single_macro_f1": single_macro,
                        "budget_per_class": int(plan["budgets"][source]),
                        "generated_features_hash": generated_hash,
                        "prediction_hash": prediction_hash,
                    }
                )
                bundles.append(bundle)
            pooled = geometric_probability_pool(bundles)
            result = evaluate_probability_predictions(prior_method, pooled, eval_labels)
            pseudo_baccs.append(float(result.bacc))
            pseudo_macros.append(float(result.macro_f1))
        subset_scores.append(
            SubsetScore(
                experiment_seed=int(experiment_seed),
                heldout_center=str(heldout_center),
                replicate_seed=int(replicate_seed),
                subset_sources=subset,
                dropped_source=_dropped_source(candidates, subset),
                subset_size=int(subset_size),
                min_pseudo_bacc=min(pseudo_baccs) if pseudo_baccs else math.nan,
                mean_pseudo_bacc=nanmean(pseudo_baccs),
                mean_pseudo_macro_f1=nanmean(pseudo_macros),
                pseudo_target_count=len(pseudo_baccs),
            )
        )
    return subset_scores, transfer_rows


def _source_inner_bundle(
    cfg: DecentralizedSourceInnerTransferTop3Config,
    *,
    per_source_runtime: Mapping[str, object],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    source_center: str,
    eval_raw: object,
    eval_labels: Sequence[int],
    budget_per_class: int,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    prior_method: str,
    control_mode: str,
) -> tuple[PredictionBundle, str, str, float, float]:
    runtime = per_source_runtime[str(source_center)].runtime
    latent_seed = d1._latent_seed(
        experiment_seed,
        heldout_center,
        replicate_seed,
        "d1_5_source_inner_scoring",
        source_center,
        int(budget_per_class),
        control_mode,
    )
    generated, labels, _counts = d1a._sample_source_from_summaries(
        cfg,
        runtime,
        summaries,
        source_center=str(source_center),
        budget_per_class=int(budget_per_class),
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
    return bundle, _hash_array(generated), _hash_array(bundle.probabilities), result.bacc, result.macro_f1


def _best_subset(subsets: Sequence[SubsetScore]) -> SubsetScore:
    if not subsets:
        raise ProtocolError("No source-inner subsets were scored.")
    return max(subsets, key=lambda row: row.sort_key)


def _shuffled_score_subset(
    subsets: Sequence[SubsetScore],
    *,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
) -> SubsetScore:
    ordered = list(subsets)
    scores = [(row.min_pseudo_bacc, row.mean_pseudo_bacc, row.mean_pseudo_macro_f1) for row in ordered]
    rng = random.Random(d1._latent_seed(experiment_seed, heldout_center, replicate_seed, "source_inner_shuffled_score_control"))
    rng.shuffle(scores)
    shuffled = [
        SubsetScore(
            row.experiment_seed,
            row.heldout_center,
            row.replicate_seed,
            row.subset_sources,
            row.dropped_source,
            row.subset_size,
            score[0],
            score[1],
            score[2],
            row.pseudo_target_count,
        )
        for row, score in zip(ordered, scores)
    ]
    return _best_subset(shuffled)


def _individual_transfer_subset(
    cfg: DecentralizedSourceInnerTransferTop3Config,
    transfer_rows: Sequence[Mapping[str, object]],
    *,
    candidates: Sequence[str],
    k: int,
) -> tuple[str, ...]:
    by_source: dict[str, list[float]] = {}
    for row in transfer_rows:
        if int(row.get("subset_size", 0)) != int(cfg.top_k_sources):
            continue
        source = str(row.get("source_expert"))
        value = _float(row.get("source_inner_single_bacc"))
        if source and math.isfinite(value):
            by_source.setdefault(source, []).append(value)
    ranked = sorted(
        (str(source) for source in candidates),
        key=lambda source: (-nanmean(by_source.get(source, [])), source),
    )
    return tuple(ranked[: int(k)])


def _dropped_source(candidates: Sequence[str], subset: Sequence[str]) -> str:
    dropped = sorted(set(str(v) for v in candidates).difference(str(v) for v in subset))
    return dropped[0] if dropped else ""


def _drop_tie_value(source: str) -> float:
    if not source:
        return 0.0
    try:
        return -float(source)
    except ValueError:
        return float("-inf")


def _evaluate_variant(
    cfg: DecentralizedSourceInnerTransferTop3Config,
    *,
    source_weighting: str,
    **kwargs: Any,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    kwargs.setdefault("source_union_ref", d1._missing_reference())
    kwargs.setdefault("center_balanced_ref", d1._missing_reference())
    kwargs.setdefault("generation_seed_method", "d1_5_target_eval_shared_generation")
    rows, late, coverage, weak, nn = d12._evaluate_weighted_variant(cfg, **kwargs)
    return (
        [_extend_source_inner_row(row, source_weighting=source_weighting) for row in rows],
        [_extend_source_inner_row(row, source_weighting=source_weighting) for row in late],
        coverage,
        weak,
        nn,
    )


def _extend_source_inner_row(row: Mapping[str, object], *, source_weighting: str | None = None) -> dict[str, object]:
    out = dict(row)
    if out.get("prior_method") == d1a.ROW_REAL_FEATURE_DENSE_REFERENCE:
        out["prior_method"] = ROW_REAL_FEATURE_DENSE_REFERENCE
    if source_weighting is not None:
        out["source_weighting"] = source_weighting
    out.update(
        {
            "delta_vs_equal_all4": out.get("delta_vs_equal_all4", math.nan),
            "delta_vs_reliability_top3": out.get("delta_vs_reliability_top3", math.nan),
            "delta_vs_exhaustive_drop_one_mean": out.get("delta_vs_exhaustive_drop_one_mean", math.nan),
            "strongest_negative_control_gap": out.get("strongest_negative_control_gap", math.nan),
            "shuffled_score_control_gap": out.get("shuffled_score_control_gap", math.nan),
            "selected_subset_target_oracle_gap": out.get("selected_subset_target_oracle_gap", math.nan),
        }
    )
    return out


def _append_single_source_references(
    cfg: DecentralizedSourceInnerTransferTop3Config,
    matrix_rows: list[dict[str, object]],
    single_rows: Sequence[Mapping[str, object]],
    *,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
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
            source_union_ref=d1._missing_reference(),
            center_balanced_ref=d1._missing_reference(),
            real_feature_bacc=real_feature_bacc,
            mean_single_bacc=mean_single,
            oracle_single_bacc=oracle_single,
            claim_role=role,
        )
        matrix_rows.append(_extend_source_inner_row(row))


def _drop_one_mean_oracle_rows(
    cfg: DecentralizedSourceInnerTransferTop3Config,
    subset_rows: Sequence[Mapping[str, object]],
    *,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    real_feature_bacc: float,
) -> list[dict[str, object]]:
    ok = [row for row in subset_rows if row.get("status") == "ok"]
    baccs = [_float(row.get("bacc")) for row in ok]
    macros = [_float(row.get("macro_f1")) for row in ok]
    mean_bacc = nanmean(baccs)
    oracle_bacc = max(baccs) if baccs else math.nan
    mean_macro = nanmean(macros)
    oracle_macro = max(macros) if macros else math.nan
    rows = []
    for method, bacc, macro_f1, role in (
        (ROW_DROP_ONE_MEAN, mean_bacc, mean_macro, "exhaustive_drop_one_mean_reference"),
        (ROW_DROP_ONE_ORACLE, oracle_bacc, oracle_macro, "diagnostic_only_drop_one_oracle_reference"),
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
            source_union_ref=d1._missing_reference(),
            center_balanced_ref=d1._missing_reference(),
            real_feature_bacc=real_feature_bacc,
            mean_single_bacc=mean_bacc,
            oracle_single_bacc=oracle_bacc,
            claim_role=role,
        )
        rows.append(_extend_source_inner_row(row))
    return rows


def _oracle_source(single_rows: Sequence[Mapping[str, object]]) -> str:
    rows = [
        row for row in single_rows
        if row.get("pooling_rule") == "single_source" and row.get("status") == "ok" and math.isfinite(_float(row.get("bacc")))
    ]
    if not rows:
        return ""
    return str(max(rows, key=lambda row: (_float(row.get("bacc")), str(row.get("expert_id")))).get("expert_id"))


def _selection_row(
    subset: SubsetScore,
    single_rows: Sequence[Mapping[str, object]],
    *,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    selection_rule: str,
) -> dict[str, object]:
    oracle = _oracle_source(single_rows)
    selected = tuple(str(source) for source in subset.subset_sources)
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "replicate_seed": int(replicate_seed),
        "selection_rule": selection_rule,
        "selected_sources": "|".join(selected),
        "selected_source_count": len(selected),
        "dropped_source": subset.dropped_source,
        "oracle_source": oracle,
        "oracle_in_selected_top3": int(bool(oracle and oracle in selected)),
        "top3_downstream_oracle_containment": int(bool(oracle and oracle in selected)),
        "source_inner_min_pseudo_bacc": subset.min_pseudo_bacc,
        "source_inner_mean_pseudo_bacc": subset.mean_pseudo_bacc,
        "source_inner_mean_pseudo_macro_f1": subset.mean_pseudo_macro_f1,
        "subset_size": subset.subset_size,
    }


def _subset_score_rows(
    subsets: Sequence[SubsetScore],
    *,
    selected_subset: Sequence[str],
    shuffled_subset: Sequence[str],
) -> list[dict[str, object]]:
    selected = tuple(str(v) for v in selected_subset)
    shuffled = tuple(str(v) for v in shuffled_subset)
    ranked = _rank_subset_scores(subsets)
    rows = []
    for subset in subsets:
        rows.append(
            {
                "experiment_seed": subset.experiment_seed,
                "heldout_center": subset.heldout_center,
                "replicate_seed": subset.replicate_seed,
                "subset_size": subset.subset_size,
                "selected_sources": "|".join(subset.subset_sources),
                "dropped_source": subset.dropped_source,
                "min_pseudo_target_bacc": subset.min_pseudo_bacc,
                "mean_pseudo_target_bacc": subset.mean_pseudo_bacc,
                "mean_pseudo_target_macro_f1": subset.mean_pseudo_macro_f1,
                "pseudo_target_count": subset.pseudo_target_count,
                "source_inner_score_rank": ranked.get(subset.subset_sources, math.nan),
                "selected_by_source_inner": int(subset.subset_sources == selected),
                "selected_by_shuffled_score_control": int(subset.subset_sources == shuffled),
            }
        )
    return rows


def _rank_subset_scores(subsets: Sequence[SubsetScore]) -> dict[tuple[str, ...], int]:
    ordered = sorted(subsets, key=lambda row: row.sort_key, reverse=True)
    return {row.subset_sources: idx for idx, row in enumerate(ordered, start=1)}


def _drop_one_target_row(
    row: Mapping[str, object],
    subset: Sequence[str],
    candidates: Sequence[str],
    subset_scores: Sequence[SubsetScore],
) -> dict[str, object]:
    subset_tuple = tuple(str(v) for v in subset)
    source_score = next((score for score in subset_scores if score.subset_sources == subset_tuple), None)
    return {
        "experiment_seed": row["experiment_seed"],
        "heldout_center": row["heldout_center"],
        "replicate_seed": row["replicate_seed"],
        "selected_sources": "|".join(subset_tuple),
        "dropped_source": _dropped_source(candidates, subset_tuple),
        "target_subset_bacc": row.get("bacc", math.nan),
        "target_subset_macro_f1": row.get("macro_f1", math.nan),
        "source_inner_min_pseudo_bacc": source_score.min_pseudo_bacc if source_score else math.nan,
        "source_inner_mean_pseudo_bacc": source_score.mean_pseudo_bacc if source_score else math.nan,
        "source_inner_mean_pseudo_macro_f1": source_score.mean_pseudo_macro_f1 if source_score else math.nan,
        "selected_by_source_inner": 0,
        "target_subset_oracle": 0,
    }


def _annotate_drop_one_oracle_gap(
    drop_one_target_rows: list[dict[str, object]],
    primary_row: Mapping[str, object],
    primary_subset: SubsetScore,
) -> None:
    matching = [
        row for row in drop_one_target_rows
        if str(row.get("experiment_seed")) == str(primary_row.get("experiment_seed"))
        and str(row.get("heldout_center")) == str(primary_row.get("heldout_center"))
        and str(row.get("replicate_seed")) == str(primary_row.get("replicate_seed"))
    ]
    if not matching:
        return
    oracle_value = max(_float(row.get("target_subset_bacc")) for row in matching)
    selected_key = "|".join(primary_subset.subset_sources)
    for row in matching:
        row["selected_by_source_inner"] = int(row.get("selected_sources") == selected_key)
        row["target_subset_oracle"] = int(math.isclose(_float(row.get("target_subset_bacc")), oracle_value))
        row["selected_subset_target_oracle_gap"] = _float(primary_row.get("bacc")) - oracle_value


def _drop_frequency_row(
    subset: SubsetScore,
    target_subset_rows: Sequence[Mapping[str, object]],
    subset_scores: Sequence[SubsetScore],
    *,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    selected_oracle_source: str,
) -> dict[str, object]:
    drop_rank_target = _drop_target_ranks(target_subset_rows)
    drop_rank_source = _drop_source_inner_ranks(subset_scores)
    return {
        "heldout_center": str(heldout_center),
        "experiment_seed": int(experiment_seed),
        "replicate_seed": int(replicate_seed),
        "selected_top3_subset": "|".join(subset.subset_sources),
        "dropped_source": subset.dropped_source,
        "dropped_source_frequency": 1,
        "dropped_source_target_utility_rank": drop_rank_target.get(subset.dropped_source, math.nan),
        "dropped_source_source_inner_score_rank": drop_rank_source.get(subset.dropped_source, math.nan),
        "single_source_oracle": selected_oracle_source,
    }


def _populate_drop_frequencies(rows: list[dict[str, object]]) -> None:
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (str(row.get("heldout_center")), str(row.get("dropped_source")))
        counts[key] = counts.get(key, 0) + 1
    for row in rows:
        key = (str(row.get("heldout_center")), str(row.get("dropped_source")))
        row["dropped_source_frequency"] = counts.get(key, 0)


def _drop_target_ranks(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    scored = [
        (str(row.get("dropped_source")), _float(row.get("target_subset_bacc")))
        for row in rows
        if math.isfinite(_float(row.get("target_subset_bacc")))
    ]
    ordered = sorted(scored, key=lambda item: (item[1], item[0]), reverse=True)
    return {drop: idx for idx, (drop, _value) in enumerate(ordered, start=1)}


def _drop_source_inner_ranks(subsets: Sequence[SubsetScore]) -> dict[str, int]:
    ordered = sorted(subsets, key=lambda row: row.sort_key, reverse=True)
    return {row.dropped_source: idx for idx, row in enumerate(ordered, start=1)}


def _ineligible_rows(
    cfg: DecentralizedSourceInnerTransferTop3Config,
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
    for method, role in _methods():
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
            claim_role=role,
        )
        rows.append(_extend_source_inner_row(row))
    return rows


def _methods() -> tuple[tuple[str, str], ...]:
    return (
        (PRIMARY_SOURCE_INNER_TRANSFER_METHOD, "primary_source_inner_transfer_drop_one_composition"),
        (ROW_EQUAL_ALL4, "equal_all4_reference"),
        (ROW_RELIABILITY_TOP3_REFERENCE, "reliability_top3_reference"),
        (ROW_RELIABILITY_ALL4, "reliability_all4_weighted_reference"),
        (ROW_INDIVIDUAL_TRANSFER_TOP3, "individual_source_inner_transfer_top3_diagnostic"),
        (ROW_TRANSFER_TOP2, "source_inner_transfer_top2_diagnostic"),
        (ROW_TRANSFER_TOP4, "source_inner_transfer_top4_diagnostic"),
        (ROW_DROP_ONE_MEAN, "exhaustive_drop_one_mean_reference"),
        (ROW_DROP_ONE_ORACLE, "diagnostic_only_drop_one_oracle_reference"),
        (ROW_SINGLE_MEAN, "single_source_mean_reference"),
        (ROW_SINGLE_ORACLE, "diagnostic_only_oracle_reference"),
        (ROW_REAL_FEATURE_DENSE_REFERENCE, "real_feature_transfer_ceiling_reference"),
        (ROW_SHUFFLED_SCORE_CONTROL, "negative_control"),
        (ROW_SHUFFLED_SUMMARY_CONTROL, "negative_control"),
        (ROW_SHUFFLED_LABEL_CONTROL, "negative_control"),
    )


def _populate_deltas(rows: list[dict[str, object]]) -> None:
    refs: dict[str, dict[tuple[str, str, str], float]] = {
        ROW_EQUAL_ALL4: {},
        ROW_RELIABILITY_TOP3_REFERENCE: {},
        ROW_DROP_ONE_MEAN: {},
        ROW_SHUFFLED_SCORE_CONTROL: {},
    }
    controls: dict[tuple[str, str, str], float] = {}
    control_methods = {ROW_SHUFFLED_SCORE_CONTROL, ROW_SHUFFLED_SUMMARY_CONTROL, ROW_SHUFFLED_LABEL_CONTROL}
    drop_oracle: dict[tuple[str, str, str], float] = {}
    for row in rows:
        key = (str(row.get("experiment_seed")), str(row.get("heldout_center")), str(row.get("replicate_seed")))
        value = _float(row.get("bacc"))
        if not math.isfinite(value):
            continue
        method = str(row.get("prior_method"))
        if method in refs:
            refs[method][key] = value
        if method in control_methods:
            controls[key] = max(controls.get(key, -math.inf), value)
        if method == ROW_DROP_ONE_ORACLE:
            drop_oracle[key] = value
    for row in rows:
        if row.get("prior_method") != PRIMARY_SOURCE_INNER_TRANSFER_METHOD:
            continue
        key = (str(row.get("experiment_seed")), str(row.get("heldout_center")), str(row.get("replicate_seed")))
        value = _float(row.get("bacc"))
        for field, method in (
            ("delta_vs_equal_all4", ROW_EQUAL_ALL4),
            ("delta_vs_reliability_top3", ROW_RELIABILITY_TOP3_REFERENCE),
            ("delta_vs_exhaustive_drop_one_mean", ROW_DROP_ONE_MEAN),
            ("shuffled_score_control_gap", ROW_SHUFFLED_SCORE_CONTROL),
        ):
            baseline = refs[method].get(key, math.nan)
            if math.isfinite(value) and math.isfinite(baseline):
                row[field] = value - baseline
        control = controls.get(key, math.nan)
        if math.isfinite(value) and math.isfinite(control):
            row["strongest_negative_control_gap"] = value - control
            row["negative_control_gap"] = value - control
        oracle = drop_oracle.get(key, math.nan)
        if math.isfinite(value) and math.isfinite(oracle):
            row["selected_subset_target_oracle_gap"] = value - oracle


def _replicate_averaged(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        groups.setdefault((str(row["experiment_seed"]), str(row["heldout_center"])), []).append(row)
    return [
        {
            "experiment_seed": seed,
            "heldout_center": center,
            "bacc": d1._mean_field(subset, "bacc"),
            "macro_f1": d1._mean_field(subset, "macro_f1"),
        }
        for (seed, center), subset in sorted(groups.items())
    ]


def _method_stats(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    ok = [row for row in rows if row.get("status") == "ok"]
    grouped = _replicate_averaged(ok)
    by_seed: dict[str, list[Mapping[str, object]]] = {}
    by_center: dict[str, list[Mapping[str, object]]] = {}
    for row in grouped:
        by_seed.setdefault(str(row["experiment_seed"]), []).append(row)
        by_center.setdefault(str(row["heldout_center"]), []).append(row)
    seed_means = {seed: d1._mean_field(values, "bacc") for seed, values in sorted(by_seed.items())}
    center_means = {center: d1._mean_field(values, "bacc") for center, values in sorted(by_center.items())}
    return {
        "n_ok_rows": len(ok),
        "eligible_seed_center_cells": len(grouped),
        "eligible_seed_center_replicate_rows": len(ok),
        "n_experiment_seeds": len(by_seed),
        "n_heldout_centers": len(by_center),
        "center_equal_mean_bacc": nanmean(center_means.values()) if center_means else math.nan,
        "center_equal_macro_f1": _center_equal_mean(grouped, "macro_f1"),
        "seed_equal_mean_bacc": nanmean(seed_means.values()) if seed_means else math.nan,
        "replicate_row_mean_bacc": d1._mean_field(ok, "bacc"),
        "seed_std_bacc": d1._std(list(seed_means.values())),
        "min_center_bacc": min(center_means.values()) if center_means else math.nan,
        "min_cell_bacc": d1._min_field(grouped, "bacc"),
        "per_center_bacc": json.dumps(center_means, sort_keys=True),
        "per_seed_bacc": json.dumps(seed_means, sort_keys=True),
    }


def _center_equal_mean(rows: Sequence[Mapping[str, object]], field: str) -> float:
    by_center: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        by_center.setdefault(str(row["heldout_center"]), []).append(row)
    return nanmean([d1._mean_field(values, field) for values in by_center.values()]) if by_center else math.nan


def _rows_for(rows: Sequence[Mapping[str, object]], method: str) -> list[Mapping[str, object]]:
    return [row for row in rows if row.get("prior_method") == method and row.get("status") == "ok"]


def _cell_means(rows: Sequence[Mapping[str, object]], field: str = "bacc") -> dict[tuple[str, str], float]:
    groups: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        if row.get("status") == "ok":
            groups.setdefault((str(row["experiment_seed"]), str(row["heldout_center"])), []).append(row)
    return {key: d1._mean_field(values, field) for key, values in groups.items()}


def _centerwise_delta_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    primary = _cell_means(_rows_for(rows, PRIMARY_SOURCE_INNER_TRANSFER_METHOD))
    equal = _cell_means(_rows_for(rows, ROW_EQUAL_ALL4))
    by_center: dict[str, list[float]] = {}
    by_seed: dict[str, list[float]] = {}
    for key, value in primary.items():
        baseline = equal.get(key, math.nan)
        seed, center = key
        if math.isfinite(value) and math.isfinite(baseline):
            delta = value - baseline
            by_center.setdefault(center, []).append(delta)
            by_seed.setdefault(seed, []).append(delta)
    out = [
        {"axis": "center", "id": key, "delta_vs_equal_all4": nanmean(values), "n_cells": len(values)}
        for key, values in sorted(by_center.items())
    ]
    out.extend(
        {"axis": "seed", "id": key, "delta_vs_equal_all4": nanmean(values), "n_cells": len(values)}
        for key, values in sorted(by_seed.items())
    )
    return out


def _decision(
    rows: Sequence[Mapping[str, object]],
    cfg: DecentralizedSourceInnerTransferTop3Config,
    *,
    leakage_status: str,
    selection_rows: Sequence[Mapping[str, object]],
    centerwise_rows: Sequence[Mapping[str, object]],
    subset_score_rows: Sequence[Mapping[str, object]],
    drop_one_target_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    primary_stats = _method_stats(_rows_for(rows, PRIMARY_SOURCE_INNER_TRANSFER_METHOD))
    equal_stats = _method_stats(_rows_for(rows, ROW_EQUAL_ALL4))
    reliability_stats = _method_stats(_rows_for(rows, ROW_RELIABILITY_TOP3_REFERENCE))
    drop_mean_stats = _method_stats(_rows_for(rows, ROW_DROP_ONE_MEAN))
    single_mean_stats = _method_stats(_rows_for(rows, ROW_SINGLE_MEAN))
    single_oracle_stats = _method_stats(_rows_for(rows, ROW_SINGLE_ORACLE))
    real_stats = _method_stats(_rows_for(rows, ROW_REAL_FEATURE_DENSE_REFERENCE))
    controls = {
        ROW_SHUFFLED_SCORE_CONTROL: _method_stats(_rows_for(rows, ROW_SHUFFLED_SCORE_CONTROL)),
        ROW_SHUFFLED_SUMMARY_CONTROL: _method_stats(_rows_for(rows, ROW_SHUFFLED_SUMMARY_CONTROL)),
        ROW_SHUFFLED_LABEL_CONTROL: _method_stats(_rows_for(rows, ROW_SHUFFLED_LABEL_CONTROL)),
    }
    strongest_control_method, strongest_control_bacc = _strongest_control(controls)
    delta_equal = _float(primary_stats["center_equal_mean_bacc"]) - _float(equal_stats["center_equal_mean_bacc"])
    delta_reliability = _float(primary_stats["center_equal_mean_bacc"]) - _float(reliability_stats["center_equal_mean_bacc"])
    delta_drop_mean = _float(primary_stats["center_equal_mean_bacc"]) - _float(drop_mean_stats["center_equal_mean_bacc"])
    delta_mean_single = _float(primary_stats["center_equal_mean_bacc"]) - _float(single_mean_stats["center_equal_mean_bacc"])
    delta_oracle_single = _float(primary_stats["center_equal_mean_bacc"]) - _float(single_oracle_stats["center_equal_mean_bacc"])
    delta_real = _float(primary_stats["center_equal_mean_bacc"]) - _float(real_stats["center_equal_mean_bacc"])
    negative_gap = _float(primary_stats["center_equal_mean_bacc"]) - strongest_control_bacc
    shuffled_score_gap = _float(primary_stats["center_equal_mean_bacc"]) - _float(controls[ROW_SHUFFLED_SCORE_CONTROL]["center_equal_mean_bacc"])
    source_target_spearman = _source_inner_target_subset_spearman(subset_score_rows, drop_one_target_rows)
    center_deltas = {str(row["id"]): _float(row["delta_vs_equal_all4"]) for row in centerwise_rows if row.get("axis") == "center"}
    seed_deltas = {str(row["id"]): _float(row["delta_vs_equal_all4"]) for row in centerwise_rows if row.get("axis") == "seed"}
    centers_beating = sum(1 for value in center_deltas.values() if math.isfinite(value) and value > 0.0)
    seeds_beating = sum(1 for value in seed_deltas.values() if math.isfinite(value) and value > 0.0)
    containment = nanmean([_float(row.get("top3_downstream_oracle_containment")) for row in selection_rows])
    oracle_gap = nanmean([
        _float(row.get("selected_subset_target_oracle_gap"))
        for row in drop_one_target_rows
        if str(row.get("selected_by_source_inner")) == "1"
    ])
    weak_stability_improved = _float(primary_stats["min_center_bacc"]) > _float(equal_stats["min_center_bacc"])
    pass_rule = (
        leakage_status == "PASS"
        and int(primary_stats["n_heldout_centers"]) >= len(cfg.heldout_centers)
        and int(primary_stats["eligible_seed_center_cells"]) >= 14
        and _float(primary_stats["center_equal_mean_bacc"]) >= 0.85
        and _float(primary_stats["min_center_bacc"]) >= 0.78
        and _float(primary_stats["seed_std_bacc"]) <= 0.05
        and delta_equal >= 0.01
        and delta_reliability >= 0.01
        and delta_drop_mean >= 0.01
        and centers_beating >= 4
        and seeds_beating >= 2
        and source_target_spearman >= 0.20
        and containment >= 0.85
        and negative_gap >= 0.03
        and shuffled_score_gap >= 0.02
        and bool(_rows_for(rows, ROW_REAL_FEATURE_DENSE_REFERENCE))
    )
    thesis_partial = (
        leakage_status == "PASS"
        and delta_equal >= 0.005
        and delta_reliability >= 0.005
        and source_target_spearman >= 0.20
        and shuffled_score_gap >= 0.01
        and weak_stability_improved
    )
    diagnostic_only = (
        leakage_status == "PASS"
        and source_target_spearman >= 0.20
        and delta_equal < 0.005
    )
    verdict = "D1_5_FAIL"
    if leakage_status != "PASS":
        verdict = "PROTOCOL_FAIL"
    elif pass_rule:
        verdict = "D1_5_PASS"
    elif thesis_partial:
        verdict = "D1_5_THESIS_PARTIAL"
    elif diagnostic_only:
        verdict = "D1_5_DIAGNOSTIC_ONLY"

    return {
        "primary_verdict": verdict,
        "leakage_status": leakage_status,
        "primary_method": PRIMARY_SOURCE_INNER_TRANSFER_METHOD,
        "center_equal_mean_bacc": primary_stats["center_equal_mean_bacc"],
        "center_equal_macro_f1": primary_stats["center_equal_macro_f1"],
        "seed_equal_mean_bacc": primary_stats["seed_equal_mean_bacc"],
        "replicate_row_mean_bacc": primary_stats["replicate_row_mean_bacc"],
        "min_center_bacc": primary_stats["min_center_bacc"],
        "seed_std_bacc": primary_stats["seed_std_bacc"],
        "eligible_seed_center_cells": primary_stats["eligible_seed_center_cells"],
        "eligible_seed_center_replicate_rows": primary_stats["eligible_seed_center_replicate_rows"],
        "delta_vs_equal_all4": delta_equal,
        "delta_vs_reliability_top3": delta_reliability,
        "delta_vs_exhaustive_drop_one_mean": delta_drop_mean,
        "delta_vs_mean_single_source_adaptive_k": delta_mean_single,
        "delta_vs_single_source_oracle_adaptive_k": delta_oracle_single,
        "delta_vs_real_feature_dense_reference": delta_real,
        "source_inner_score_vs_target_subset_utility_spearman": source_target_spearman,
        "top3_downstream_oracle_containment": containment,
        "selected_subset_target_oracle_gap": oracle_gap,
        "strongest_negative_control_method": strongest_control_method,
        "strongest_negative_control_center_equal_mean_bacc": strongest_control_bacc,
        "strongest_negative_control_gap": negative_gap,
        "shuffled_score_control_gap": shuffled_score_gap,
        "centerwise_delta_vs_equal_all4_json": json.dumps(center_deltas, sort_keys=True),
        "seedwise_delta_vs_equal_all4_json": json.dumps(seed_deltas, sort_keys=True),
        "centers_beating_equal_all4": centers_beating,
        "seeds_beating_equal_all4": seeds_beating,
        "equal_all4_center_equal_mean_bacc": equal_stats["center_equal_mean_bacc"],
        "reliability_top3_center_equal_mean_bacc": reliability_stats["center_equal_mean_bacc"],
        "exhaustive_drop_one_mean_center_equal_mean_bacc": drop_mean_stats["center_equal_mean_bacc"],
        "mean_single_source_adaptive_k_center_equal_mean_bacc": single_mean_stats["center_equal_mean_bacc"],
        "single_source_oracle_adaptive_k_center_equal_mean_bacc": single_oracle_stats["center_equal_mean_bacc"],
        "real_feature_dense_reference_center_equal_mean_bacc": real_stats["center_equal_mean_bacc"],
        "shuffled_score_control_center_equal_mean_bacc": controls[ROW_SHUFFLED_SCORE_CONTROL]["center_equal_mean_bacc"],
        "weak_center_stability_improved_vs_equal_all4": weak_stability_improved,
        **primary_stats,
    }


def _strongest_control(control_by_method: Mapping[str, Mapping[str, object]]) -> tuple[str, float]:
    scored = [
        (method, _float(stats.get("center_equal_mean_bacc")))
        for method, stats in control_by_method.items()
        if math.isfinite(_float(stats.get("center_equal_mean_bacc")))
    ]
    if not scored:
        return "", math.nan
    return max(scored, key=lambda item: (item[1], item[0]))


def _source_inner_target_subset_spearman(
    subset_score_rows: Sequence[Mapping[str, object]],
    target_rows: Sequence[Mapping[str, object]],
) -> float:
    score_by_key = {
        (
            str(row.get("experiment_seed")),
            str(row.get("heldout_center")),
            str(row.get("replicate_seed")),
            str(row.get("selected_sources")),
        ): _float(row.get("min_pseudo_target_bacc"))
        for row in subset_score_rows
    }
    xs: list[float] = []
    ys: list[float] = []
    for row in target_rows:
        key = (
            str(row.get("experiment_seed")),
            str(row.get("heldout_center")),
            str(row.get("replicate_seed")),
            str(row.get("selected_sources")),
        )
        x = score_by_key.get(key, math.nan)
        y = _float(row.get("target_subset_bacc"))
        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)
    return _spearman(xs, ys)


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return math.nan
    rx = _ranks([x for x, _y in pairs])
    ry = _ranks([y for _x, y in pairs])
    return _pearson(rx, ry)


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for pos in range(i, j + 1):
            ranks[order[pos]] = avg
        i = j + 1
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return math.nan
    mean_x = sum(xs) / float(len(xs))
    mean_y = sum(ys) / float(len(ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0.0 or var_y <= 0.0:
        return math.nan
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / math.sqrt(var_x * var_y)


def _write_artifacts(
    root: Path,
    cfg: DecentralizedSourceInnerTransferTop3Config,
    *,
    matrix_rows: Sequence[Mapping[str, object]],
    gap_rows: Sequence[Mapping[str, object]],
    summary_manifest_rows: Sequence[Mapping[str, object]],
    diagnostic_rows: Sequence[Mapping[str, object]],
    reliability_rows: Sequence[Mapping[str, object]],
    source_inner_rows: Sequence[Mapping[str, object]],
    subset_score_rows: Sequence[Mapping[str, object]],
    selection_rows: Sequence[Mapping[str, object]],
    drop_frequency_rows: Sequence[Mapping[str, object]],
    drop_one_target_rows: Sequence[Mapping[str, object]],
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
    write_csv_rows(root / "tables" / "decentralized_source_inner_transfer_downstream_matrix.csv", matrix_rows, columns=_matrix_columns())
    write_csv_rows(root / "tables" / "decentralized_source_inner_transfer_gap_summary.csv", gap_rows, columns=_matrix_columns())
    write_csv_rows(root / "tables" / "decentralized_source_inner_transfer_summary.csv", [dict(decision)])
    write_csv_rows(root / "tables" / "source_reliability_manifest.csv", reliability_rows)
    write_csv_rows(root / "tables" / "source_inner_transfer_matrix.csv", source_inner_rows)
    write_csv_rows(root / "tables" / "source_inner_subset_score_manifest.csv", subset_score_rows)
    write_csv_rows(root / "tables" / "source_inner_top3_selection_manifest.csv", selection_rows)
    write_csv_rows(root / "tables" / "source_drop_frequency_summary.csv", drop_frequency_rows)
    write_csv_rows(root / "tables" / "drop_one_subset_target_utility_matrix.csv", drop_one_target_rows)
    write_csv_rows(root / "tables" / "centerwise_delta_summary.csv", centerwise_rows)
    write_csv_rows(root / "tables" / "late_aggregation_matrix.csv", late_rows, columns=_matrix_columns())
    write_csv_rows(root / "tables" / "real_feature_reference_matrix.csv", real_feature_rows, columns=_matrix_columns())
    write_csv_rows(root / "tables" / "generated_component_coverage_audit.csv", coverage_rows)
    write_csv_rows(root / "tables" / "weak_source_audit.csv", weak_rows)
    write_csv_rows(root / "tables" / "nearest_neighbor_memorization_audit.csv", nn_rows)
    write_csv_rows(root / "tables" / "negative_control_summary.csv", [_negative_control_summary(decision)])
    write_csv_rows(root / "tables" / "exported_source_summary_manifest.csv", summary_manifest_rows, columns=d1a._summary_manifest_columns())
    write_csv_rows(root / "tables" / "source_summary_diagnostics.csv", diagnostic_rows, columns=d1a._diagnostic_columns())
    write_csv_rows(root / "manifests" / "decentralized_source_inner_transfer_prior_model_manifest.csv", model_manifest_rows)
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
            "schema_version": "cvae_rebuild_decentralized_source_inner_transfer_protocol_manifest_v1",
            "experiment_name": cfg.name,
            "experiment_type": "source_inner_off_diagonal_transfer_drop_one_confirmation",
            "primary_variant": cfg.primary_variant,
            "primary_method": cfg.primary_method,
            "top_k_sources": cfg.top_k_sources,
            "target_support_labels_for_selection": False,
            "target_support_features_for_selection": False,
            "target_eval_labels_for_scoring_only": True,
            "target_expert_excluded": target_expert_excluded,
            "heldout_target_rows_used_for_source_inner_scoring": False,
            "source_inner_uses_non_target_source_eval_rows": True,
            "method_comparison_uses_method_invariant_generation_seed": True,
            "exported_source_summaries_are_target_agnostic": True,
            "raw_source_embedding_pooling_for_prior_fit": False,
            "adaptive_k_selection_uses_source_local_fit_statistics_only": True,
            "oracle_rows_diagnostic_only": True,
            "protocol_wording": PROTOCOL_WORDING,
            "drop_one_claim_boundary": DROP_ONE_CLAIM_BOUNDARY,
            "claim_boundary": (
                "source-inner off-diagonal transfer drop-one confirmation only; no target-conditioned routing, "
                "no metadata-routing claim, no large-pool sparse MoErging claim, and no formal privacy claim"
            ),
        },
    )
    _write_decision_summary(root, decision, leakage_status=leakage_status)
    write_json(root / "run_config_resolved.yaml", _resolved_config(cfg))


def _matrix_columns() -> tuple[str, ...]:
    return d12._matrix_columns() + (
        "delta_vs_equal_all4",
        "delta_vs_reliability_top3",
        "delta_vs_exhaustive_drop_one_mean",
        "strongest_negative_control_gap",
        "shuffled_score_control_gap",
        "selected_subset_target_oracle_gap",
    )


def _negative_control_summary(decision: Mapping[str, object]) -> dict[str, object]:
    return {
        "primary_method": PRIMARY_SOURCE_INNER_TRANSFER_METHOD,
        "control_methods": f"{ROW_SHUFFLED_SCORE_CONTROL}|{ROW_SHUFFLED_SUMMARY_CONTROL}|{ROW_SHUFFLED_LABEL_CONTROL}",
        "primary_center_equal_mean_bacc": decision.get("center_equal_mean_bacc", math.nan),
        "strongest_negative_control_method": decision.get("strongest_negative_control_method", ""),
        "strongest_negative_control_center_equal_mean_bacc": decision.get("strongest_negative_control_center_equal_mean_bacc", math.nan),
        "strongest_negative_control_gap": decision.get("strongest_negative_control_gap", math.nan),
        "shuffled_score_control_gap": decision.get("shuffled_score_control_gap", math.nan),
        "control_competitive": _float(decision.get("strongest_negative_control_gap")) < 0.03,
    }


def _write_decision_summary(root: Path, decision: Mapping[str, object], *, leakage_status: str) -> None:
    text = "\n".join(
        [
            "# D1.5: Source-Inner Off-Diagonal Transfer Drop-One Confirmation",
            "",
            "## Summary",
            "",
            f"- Primary method: `{PRIMARY_SOURCE_INNER_TRANSFER_METHOD}`",
            f"- Primary verdict: `{decision.get('primary_verdict', 'D1_5_FAIL')}`",
            f"- Center-equal mean BACC: {_format_float(decision.get('center_equal_mean_bacc'))}",
            f"- Center-equal macro-F1: {_format_float(decision.get('center_equal_macro_f1'))}",
            f"- Seed-equal mean BACC: {_format_float(decision.get('seed_equal_mean_bacc'))}",
            f"- Replicate-row mean BACC: {_format_float(decision.get('replicate_row_mean_bacc'))}",
            f"- Min center BACC: {_format_float(decision.get('min_center_bacc'))}",
            f"- Seed std BACC: {_format_float(decision.get('seed_std_bacc'))}",
            f"- Eligible seed-center cells: {decision.get('eligible_seed_center_cells')}",
            f"- Delta vs equal all4: {_format_float(decision.get('delta_vs_equal_all4'))}",
            f"- Delta vs reliability top3: {_format_float(decision.get('delta_vs_reliability_top3'))}",
            f"- Delta vs exhaustive drop-one mean: {_format_float(decision.get('delta_vs_exhaustive_drop_one_mean'))}",
            f"- Source-inner score vs target subset utility Spearman: {_format_float(decision.get('source_inner_score_vs_target_subset_utility_spearman'))}",
            f"- Top-3 downstream oracle containment: {_format_float(decision.get('top3_downstream_oracle_containment'))}",
            f"- Selected subset target oracle gap: {_format_float(decision.get('selected_subset_target_oracle_gap'))}",
            f"- Strongest negative-control gap: {_format_float(decision.get('strongest_negative_control_gap'))}",
            f"- Shuffled-score control gap: {_format_float(decision.get('shuffled_score_control_gap'))}",
            f"- Leakage status: `{leakage_status}`",
            "",
            "## Protocol Boundary",
            "",
            PROTOCOL_WORDING,
            "",
            DROP_ONE_CLAIM_BOUNDARY,
            "",
            "This is source-inner transfer selection, not target-conditioned routing.",
            "It does not consume target support features. Target labels are final scoring only.",
            "For target evaluation, rows with the same source, budget, fold, replicate, and control mode use the same synthetic generation seed.",
            "",
            "## Supported Claim If PASS",
            "",
            "Source-inner off-diagonal transfer can provide source-only evidence for small-pool sparse source exclusion in decentralized generated-embedding composition.",
            "",
        ]
    )
    path = root / "reports" / "decision_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _resolved_config(cfg: DecentralizedSourceInnerTransferTop3Config) -> dict[str, object]:
    return {
        "name": cfg.name,
        "artifact_root": str(cfg.artifact_root),
        "repair_artifact_root": str(cfg.repair_artifact_root),
        "feature_cache_root": str(cfg.feature_cache_root),
        "backbone": cfg.backbone,
        "experiment_seeds": list(cfg.experiment_seeds),
        "heldout_centers": list(cfg.heldout_centers),
        "replicate_seeds": list(cfg.replicate_seeds),
        "synthetic_per_class_total": cfg.synthetic_per_class_total,
        "min_per_source_per_class": cfg.min_per_source_per_class,
        "top_k_sources": cfg.top_k_sources,
        "primary_variant": cfg.primary_variant,
        "primary_method": cfg.primary_method,
        "source_weighting": cfg.source_weighting,
        "candidate_components_per_source_class": list(cfg.candidate_components_per_source_class),
        "gmm_covariance_type": cfg.gmm_covariance_type,
        "gmm_reg_covar": cfg.gmm_reg_covar,
        "gmm_n_init": cfg.gmm_n_init,
        "gmm_max_iter": cfg.gmm_max_iter,
        "min_component_weight": cfg.min_component_weight,
        "variance_floor": cfg.variance_floor,
        "reliability_floor_score": cfg.reliability_floor_score,
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
