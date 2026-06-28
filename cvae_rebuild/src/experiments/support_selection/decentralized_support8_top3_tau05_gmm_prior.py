from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from features import load_feature_cache
from metrics import nanmean
from preservation_repair import (
    NA,
    PRIMARY_VARIANT,
    _float,
    _format_float,
    _label,
    _load_mapping,
    _mapping,
    _path,
    _source_data_for_centers,
)
from preservation_sampling import DIAGNOSTIC_SELECTION, PRIMARY_SELECTION, _manifest_row, _per_source_variant, _runtime_source
from protocol import ProtocolError, assert_candidate_pool, build_leakage_report
from reporting import prepare_artifact_dirs, write_csv_rows, write_json, write_protocol_finalization
from splits import candidate_experts
from support_nelbo import SupportScore, rank_support_scores, ranking_alignment

import decentralized_adaptive_gmm_prior as d1a
import decentralized_k16_gmm_prior as d1
import decentralized_reliability_weighted_gmm_prior as d12
import decentralized_support_nelbo_reliability_gmm_prior as d13


SUPPORT8_TOP3_TAU05_NAME = "virchow2_cvae_decentralized_support8_top3_tau05_gmm_prior_v1"
PRIMARY_SUPPORT8_TOP3_TAU05_METHOD = "decentralized_support8_top3_tau05_support_nelbo_x_reliability_geom"
ROW_D1_2_SUPPORT8_REFERENCE = "decentralized_support8_d1_2_reliability_all4_geom_reference"
ROW_EQUAL_SUPPORT8_REFERENCE = "decentralized_support8_equal_all4_geom_reference"
ROW_SUPPORT_ONLY_TOP3 = "decentralized_support8_support_nelbo_only_top3_tau05_geom"
ROW_SINGLE_MEAN = "per_source_support8_adaptive_k_single_expert_mean_reference"
ROW_SINGLE_ORACLE = "per_source_support8_adaptive_k_single_expert_oracle_reference"
ROW_REAL_FEATURE_SUPPORT8 = "real_source_embedding_classifier_dense_support8_reference"
ROW_ALL4_TAU05 = "decentralized_support8_all4_tau05_support_nelbo_x_reliability_geom"
ROW_TOP3_TAU1 = "decentralized_support8_top3_tau1_support_nelbo_x_reliability_geom"
ROW_ALL4_TAU1 = "decentralized_support8_all4_tau1_support_nelbo_x_reliability_geom"
ROW_RELIABILITY_TOP3 = "decentralized_support8_reliability_only_top3_geom_diagnostic"
ROW_SUPPORT_ONLY_TOP3_EQUAL_BUDGET = "decentralized_support8_top3_tau05_support_nelbo_only_equal_budget_geom"
ROW_SHUFFLED_SUPPORT_CONTROL = "decentralized_support8_shuffled_support_top3_tau05_control"
ROW_SHUFFLED_SUMMARY_CONTROL = "decentralized_support8_shuffled_summary_top3_tau05_control"
ROW_SHUFFLED_LABEL_CONTROL = "decentralized_support8_shuffled_label_top3_tau05_control"
ROW_D13_PRIMARY_CONTEXT = "decentralized_support32_d1_3_primary_context"
ROW_D13_TOP3_CONTEXT = "decentralized_support32_d1_3_top3_context"
PROTOCOL_WORDING = (
    "This is a data-minimizing, raw-data-free source-local latent summary-exchange protocol with "
    "a locked support-size-8/top-3/tau-0.5 target-support NELBO signal. It is not a formal "
    "differential privacy claim. Exported latent summary statistics may still contain "
    "distributional information derived from private data."
)


@dataclass(frozen=True)
class DecentralizedSupport8Top3Tau05Config:
    name: str
    artifact_root: Path
    repair_artifact_root: Path
    d1_3_artifact_root: Path | None
    feature_cache_root: Path
    backbone: str
    experiment_seeds: tuple[int, ...]
    heldout_centers: tuple[str, ...]
    replicate_seeds: tuple[int, ...]
    support_seeds: tuple[int, ...]
    support_size: int
    align_support_and_generation_seed: bool
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
    support_nelbo_tau: float
    top_k_sources: int
    support_alpha: float
    reliability_alpha: float
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
    def softmax_tau(self) -> float:
        return 1.0


def load_decentralized_support8_top3_tau05_gmm_prior_config(path: str | Path) -> DecentralizedSupport8Top3Tau05Config:
    source = Path(path).resolve()
    data = _load_mapping(source)
    base_dir = source.parents[2] if len(source.parents) >= 3 else source.parent
    return parse_decentralized_support8_top3_tau05_gmm_prior_config(data, base_dir=base_dir)


def parse_decentralized_support8_top3_tau05_gmm_prior_config(
    data: Mapping[str, Any],
    *,
    base_dir: str | Path = ".",
) -> DecentralizedSupport8Top3Tau05Config:
    base = Path(base_dir)
    experiment = _mapping(data, "experiment")
    inputs = _mapping(data, "inputs")
    run = _mapping(data, "run_matrix")
    generation = _mapping(data, "generation")
    gmm = _mapping(data, "support8_top3_tau05_gmm_prior")
    classifier = _mapping(data, "classifier")
    cfg = DecentralizedSupport8Top3Tau05Config(
        name=str(experiment["name"]),
        artifact_root=_path(base, str(experiment["artifact_root"])),
        repair_artifact_root=_path(base, str(inputs["repair_artifact_root"])),
        d1_3_artifact_root=_optional_path(base, inputs.get("d1_3_artifact_root")),
        feature_cache_root=_path(base, str(inputs["feature_cache_root"])),
        backbone=str(inputs.get("backbone", "")),
        experiment_seeds=tuple(int(v) for v in run["experiment_seeds"]),
        heldout_centers=tuple(str(v) for v in run["heldout_centers"]),
        replicate_seeds=tuple(int(v) for v in run["replicate_seeds"]),
        support_seeds=tuple(int(v) for v in run["support_seeds"]),
        support_size=int(run["support_size"]),
        align_support_and_generation_seed=bool(run["align_support_and_generation_seed"]),
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
        support_nelbo_tau=float(gmm["support_nelbo_tau"]),
        top_k_sources=int(gmm["top_k_sources"]),
        support_alpha=float(gmm["support_alpha"]),
        reliability_alpha=float(gmm["reliability_alpha"]),
        classifier_type=str(classifier["type"]),
        classifier_solver=str(classifier["solver"]),
        classifier_c=float(classifier["C"]),
        classifier_max_iter=int(classifier["max_iter"]),
        classifier_class_weight=str(classifier["class_weight"]),
        classifier_seed=None if classifier.get("classifier_seed") is None else int(classifier["classifier_seed"]),
    )
    validate_decentralized_support8_top3_tau05_gmm_prior_config(cfg)
    return cfg


def validate_decentralized_support8_top3_tau05_gmm_prior_config(cfg: DecentralizedSupport8Top3Tau05Config) -> None:
    if cfg.name != SUPPORT8_TOP3_TAU05_NAME:
        raise ProtocolError(f"D1.3.1 experiment name must be {SUPPORT8_TOP3_TAU05_NAME!r}.")
    if cfg.backbone != "virchow2":
        raise ProtocolError("D1.3.1 is locked to backbone=virchow2.")
    if cfg.primary_variant != PRIMARY_VARIANT:
        raise ProtocolError(f"primary_variant must be {PRIMARY_VARIANT!r}.")
    if cfg.primary_method != PRIMARY_SUPPORT8_TOP3_TAU05_METHOD:
        raise ProtocolError(f"primary_method must be {PRIMARY_SUPPORT8_TOP3_TAU05_METHOD!r}.")
    if cfg.candidate_components_per_source_class != (4, 3, 2, 1):
        raise ProtocolError("candidate_components_per_source_class must be locked to [4, 3, 2, 1].")
    if len(cfg.heldout_centers) != 5:
        raise ProtocolError("D1.3.1 composition expects exactly five centers, giving four source experts per fold.")
    if cfg.replicate_seeds != cfg.support_seeds or not cfg.align_support_and_generation_seed:
        raise ProtocolError("D1.3.1 requires support_seeds == replicate_seeds with aligned support/generation seeds.")
    if cfg.support_size != 8:
        raise ProtocolError("D1.3.1 support_size must be locked to 8.")
    if abs(cfg.support_nelbo_tau - 0.5) > 1.0e-12:
        raise ProtocolError("D1.3.1 support_nelbo_tau must be locked to 0.5.")
    if cfg.top_k_sources != 3:
        raise ProtocolError("D1.3.1 top_k_sources must be locked to 3.")
    if cfg.source_weighting != "support_nelbo_x_source_local_reliability_top3":
        raise ProtocolError("source_weighting must be support_nelbo_x_source_local_reliability_top3.")
    if cfg.gmm_covariance_type != "diag":
        raise ProtocolError("gmm_covariance_type must be diag.")
    if cfg.primary_pooling != "weighted_geometric":
        raise ProtocolError("primary_pooling must be weighted_geometric.")
    if cfg.synthetic_per_class_total != 128 or cfg.min_per_source_per_class != 8:
        raise ProtocolError("D1.3.1 synthetic budget must be 128 total with min_per_source_per_class=8.")
    if min(cfg.min_samples_per_component, cfg.gmm_n_init, cfg.gmm_max_iter) < 1:
        raise ProtocolError("GMM counts and iteration settings must be positive.")
    if min(
        cfg.gmm_reg_covar,
        cfg.min_component_weight,
        cfg.variance_floor,
        cfg.reliability_floor_score,
        cfg.support_nelbo_tau,
        cfg.support_alpha,
        cfg.reliability_alpha,
    ) <= 0.0:
        raise ProtocolError("GMM, reliability, support-NELBO tau, and alpha values must be positive.")
    if cfg.classifier_type != "sklearn_logistic_regression":
        raise ProtocolError("classifier.type must be sklearn_logistic_regression.")
    if cfg.classifier_solver != "lbfgs" or cfg.classifier_c != 1.0 or cfg.classifier_max_iter != 2000:
        raise ProtocolError("Classifier solver/C/max_iter must remain locked.")
    if cfg.classifier_class_weight != "balanced" or cfg.classifier_seed is not None:
        raise ProtocolError("Classifier must use class_weight=balanced and classifier_seed=null.")


def run_decentralized_support8_top3_tau05_gmm_prior(
    cfg: DecentralizedSupport8Top3Tau05Config,
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
    support_score_rows: list[dict[str, object]] = []
    support_weight_rows: list[dict[str, object]] = []
    combined_weight_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    alignment_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
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
            train_cache = load_feature_cache(d12._existing_cache_path(cfg.feature_cache_root, seed=experiment_seed, split="train"))
            test_cache = load_feature_cache(d12._existing_cache_path(cfg.feature_cache_root, seed=experiment_seed, split="test"))
            per_source_runtime: dict[str, object] = {}
            largest_summaries: dict[tuple[str, int], d1a.AdaptiveSourceLocalSummary] = {}
            shuffled_summaries: dict[tuple[str, int], d1a.AdaptiveSourceLocalSummary] = {}
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
                support_calibration[str(source_center)] = d13._source_nelbo_calibration(runtime_source.runtime, str(source_center))

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

                for replicate_seed in cfg.replicate_seeds:
                    support_seed = int(replicate_seed)
                    support_context, eval_raw, eval_labels, support_raw, split_error = d13._support_eval_context(
                        cfg,
                        test_cache=test_cache,
                        heldout_center=str(heldout_center),
                        support_size=cfg.support_size,
                        support_seed=support_seed,
                    )
                    support_context = dict(support_context)
                    support_context.update({"experiment_seed": int(experiment_seed), "replicate_seed": int(replicate_seed)})
                    split_rows.append(support_context)
                    rels = {
                        source: reliability[(int(experiment_seed), int(replicate_seed), str(source))]
                        for source in candidates
                    }

                    if split_error:
                        matrix_rows.extend(
                            _ineligible_rows(
                                cfg,
                                experiment_seed=int(experiment_seed),
                                heldout_center=str(heldout_center),
                                replicate_seed=int(replicate_seed),
                                candidates=candidates,
                                support_context=support_context,
                                status="ineligible",
                                error_message=split_error,
                            )
                        )
                        continue

                    support_scores = d13._support_scores(
                        cfg,
                        per_source_runtime=per_source_runtime,
                        support_calibration=support_calibration,
                        candidates=candidates,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        support_seed=support_seed,
                        support_size=cfg.support_size,
                        support_raw=support_raw,
                    )
                    ranked_scores = tuple(rank_support_scores(support_scores, eligible_count=len(candidates)))

                    all4_tau05_plan = d13._combined_weight_plan(
                        cfg,
                        candidates,
                        rels,
                        ranked_scores,
                        tau=0.5,
                        use_calibrated=True,
                        include_reliability=True,
                    )
                    primary_plan = _topk_existing_plan(cfg, all4_tau05_plan, ranked_scores, k=cfg.top_k_sources)
                    support_score_rows.extend(d13._support_score_rows(ranked_scores, support_context))
                    support_weight_rows.extend(d13._support_weight_manifest_rows(ranked_scores, support_context, primary_plan))
                    combined_weight_rows.extend(d13._combined_weight_manifest_rows(support_context, primary_plan, rels))

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
                    ref_row = _rename_real_feature_reference(ref_row)
                    real_late = [_rename_real_feature_reference(row) for row in real_late]
                    ref_row = _extend_support8_row(ref_row, support_context=support_context)
                    real_late = [_extend_support8_row(row, support_context=support_context) for row in real_late]
                    real_feature_rows.append(ref_row)
                    matrix_rows.append(ref_row)
                    late_rows.extend(real_late)

                    equal_plan = d12._uniform_weight_plan(cfg, candidates, rels)
                    equal_rows, equal_late, coverage, weak, nn = _evaluate_support8_variant(
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
                        weight_plan=equal_plan,
                        prior_method=ROW_EQUAL_SUPPORT8_REFERENCE,
                        pooling_rule="geometric",
                        support_context=support_context,
                        source_weighting="equal_source_mass",
                        selection_source=DIAGNOSTIC_SELECTION,
                        claim_role="matched_equal_all4_support8_reference",
                    )
                    matrix_rows.extend(equal_rows)
                    late_rows.extend(equal_late)
                    coverage_rows.extend(coverage)
                    weak_rows.extend(weak)
                    nn_rows.extend(nn)

                    reliability_plan = d12._weight_plan(cfg, candidates, rels, mode="linear")
                    rel_rows, rel_late, coverage, weak, nn = _evaluate_support8_variant(
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
                        weight_plan=reliability_plan,
                        prior_method=ROW_D1_2_SUPPORT8_REFERENCE,
                        pooling_rule="weighted_geometric",
                        support_context=support_context,
                        source_weighting="source_local_reliability_all4",
                        selection_source=DIAGNOSTIC_SELECTION,
                        claim_role="matched_d1_2_reliability_all4_support8_reference",
                    )
                    matrix_rows.extend(rel_rows)
                    late_rows.extend(rel_late)
                    coverage_rows.extend(coverage)
                    weak_rows.extend(weak)
                    nn_rows.extend(nn)

                    primary_rows, primary_late, coverage, weak, nn = _evaluate_support8_variant(
                        cfg,
                        per_source_runtime=per_source_runtime,
                        candidates=tuple(primary_plan["sources"]),
                        summaries=largest_summaries,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        replicate_seed=int(replicate_seed),
                        eval_raw=eval_raw,
                        eval_labels=eval_labels,
                        real_feature_bacc=_float(ref_row["bacc"]),
                        weight_plan=primary_plan,
                        prior_method=PRIMARY_SUPPORT8_TOP3_TAU05_METHOD,
                        pooling_rule="weighted_geometric",
                        support_context=support_context,
                        source_weighting="support_nelbo_x_source_local_reliability_top3_tau05",
                        selection_source=PRIMARY_SELECTION,
                        claim_role="primary_locked_support8_top3_tau05_confirmation",
                    )
                    matrix_rows.extend(primary_rows)
                    late_rows.extend(primary_late)
                    coverage_rows.extend(coverage)
                    weak_rows.extend(weak)
                    nn_rows.extend(nn)

                    _append_support8_single_source_references(
                        cfg,
                        matrix_rows,
                        equal_late,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        replicate_seed=int(replicate_seed),
                        candidates=candidates,
                        summaries=largest_summaries,
                        real_feature_bacc=_float(ref_row["bacc"]),
                        support_context=support_context,
                    )
                    alignment_rows.append(
                        _alignment_row_support8(
                            ranked_scores,
                            equal_late,
                            primary_rows,
                            selected_sources=tuple(primary_plan["sources"]),
                            support_context=support_context,
                        )
                    )
                    selection_rows.append(_selected_source_row(primary_plan, equal_late, support_context))
                    d13._attach_downstream_to_support_scores(support_score_rows, support_context, equal_late)

                    support_only_all4 = d13._combined_weight_plan(
                        cfg,
                        candidates,
                        rels,
                        ranked_scores,
                        tau=0.5,
                        use_calibrated=True,
                        include_reliability=False,
                    )
                    support_only_top3 = _topk_existing_plan(cfg, support_only_all4, ranked_scores, k=cfg.top_k_sources)
                    support_only_equal_budget = _uniform_selected_plan(cfg, tuple(support_only_top3["sources"]), support_only_top3)
                    tau1_all4 = d13._combined_weight_plan(
                        cfg,
                        candidates,
                        rels,
                        ranked_scores,
                        tau=1.0,
                        use_calibrated=True,
                        include_reliability=True,
                    )
                    reliability_top3 = _topk_existing_plan(cfg, reliability_plan, ranked_scores, k=cfg.top_k_sources)
                    shuffled_support = _topk_existing_plan(
                        cfg,
                        d13._shuffled_support_plan(cfg, candidates, rels, ranked_scores, replicate_seed),
                        ranked_scores,
                        k=cfg.top_k_sources,
                    )
                    diagnostics = (
                        (
                            ROW_SUPPORT_ONLY_TOP3,
                            support_only_top3,
                            "weighted_geometric",
                            "support_nelbo_only_top3_tau05_reference",
                            "support_nelbo_only_top3_tau05",
                        ),
                        (
                            ROW_ALL4_TAU05,
                            all4_tau05_plan,
                            "weighted_geometric",
                            "support_nelbo_x_reliability_all4_tau05_ablation",
                            "support_nelbo_x_source_local_reliability_all4_tau05",
                        ),
                        (
                            ROW_TOP3_TAU1,
                            _topk_existing_plan(cfg, tau1_all4, ranked_scores, k=cfg.top_k_sources),
                            "weighted_geometric",
                            "support_nelbo_x_reliability_top3_tau1_ablation",
                            "support_nelbo_x_source_local_reliability_top3_tau1",
                        ),
                        (
                            ROW_ALL4_TAU1,
                            tau1_all4,
                            "weighted_geometric",
                            "support_nelbo_x_reliability_all4_tau1_ablation",
                            "support_nelbo_x_source_local_reliability_all4_tau1",
                        ),
                        (
                            ROW_RELIABILITY_TOP3,
                            reliability_top3,
                            "weighted_geometric",
                            "reliability_only_top3_diagnostic",
                            "source_local_reliability_top3",
                        ),
                        (
                            ROW_SUPPORT_ONLY_TOP3_EQUAL_BUDGET,
                            support_only_equal_budget,
                            "geometric",
                            "support_nelbo_only_top3_equal_budget_ablation",
                            "support_nelbo_only_top3_equal_budget",
                        ),
                        (
                            ROW_SHUFFLED_SUPPORT_CONTROL,
                            shuffled_support,
                            "weighted_geometric",
                            "negative_control",
                            "shuffled_support_nelbo_top3_tau05_control",
                        ),
                    )
                    for method, plan, pooling_rule, role, weighting in diagnostics:
                        rows, late, coverage, weak, nn = _evaluate_support8_variant(
                            cfg,
                            per_source_runtime=per_source_runtime,
                            candidates=tuple(plan["sources"]),
                            summaries=largest_summaries,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            replicate_seed=int(replicate_seed),
                            eval_raw=eval_raw,
                            eval_labels=eval_labels,
                            real_feature_bacc=_float(ref_row["bacc"]),
                            weight_plan=plan,
                            prior_method=method,
                            pooling_rule=pooling_rule,
                            support_context=support_context,
                            source_weighting=weighting,
                            selection_source=DIAGNOSTIC_SELECTION,
                            claim_role=role,
                        )
                        matrix_rows.extend(rows)
                        late_rows.extend(late)
                        coverage_rows.extend(coverage)
                        weak_rows.extend(weak)
                        nn_rows.extend(nn)

                    for prior_method, summaries, control_mode in (
                        (ROW_SHUFFLED_SUMMARY_CONTROL, largest_summaries, "class_flip"),
                        (ROW_SHUFFLED_LABEL_CONTROL, shuffled_summaries, "normal"),
                    ):
                        rows, late, coverage, weak, nn = _evaluate_support8_variant(
                            cfg,
                            per_source_runtime=per_source_runtime,
                            candidates=tuple(primary_plan["sources"]),
                            summaries=summaries,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            replicate_seed=int(replicate_seed),
                            eval_raw=eval_raw,
                            eval_labels=eval_labels,
                            real_feature_bacc=_float(ref_row["bacc"]),
                            weight_plan=primary_plan,
                            prior_method=prior_method,
                            pooling_rule="weighted_geometric",
                            support_context=support_context,
                            source_weighting="support_nelbo_x_source_local_reliability_top3_tau05",
                            selection_source=DIAGNOSTIC_SELECTION,
                            claim_role="negative_control",
                            control_mode=control_mode,
                        )
                        matrix_rows.extend(rows)
                        late_rows.extend(late)
                        coverage_rows.extend(coverage)
                        weak_rows.extend(weak)
                        nn_rows.extend(nn)
    except ProtocolError as exc:
        protocol_violations.append(str(exc))

    context_rows, context_top3 = _load_d1_3_context(cfg.d1_3_artifact_root)
    matrix_rows.extend(context_rows)
    leakage = build_leakage_report(
        target_support_labels_for_selection=False,
        target_eval_labels_for_scoring_only=True,
        target_expert_excluded=target_expert_excluded,
        oracle_rows_diagnostic_only=True,
        extra_violations=protocol_violations,
    )
    _populate_support8_deltas(matrix_rows)
    gap_rows = [dict(row) for row in matrix_rows if row.get("status") == "ok"]
    centerwise_rows = _centerwise_delta_rows(matrix_rows)
    stability_rows = _selection_stability_rows(selection_rows)
    decision = _decision(
        matrix_rows,
        cfg,
        leakage_status=leakage.status,
        support_score_rows=support_score_rows,
        support_weight_rows=support_weight_rows,
        combined_weight_rows=combined_weight_rows,
        split_rows=split_rows,
        alignment_rows=alignment_rows,
        centerwise_rows=centerwise_rows,
        selection_rows=stability_rows,
        d1_3_top3_context_containment=context_top3,
    )
    _write_artifacts(
        root,
        cfg,
        matrix_rows=matrix_rows,
        gap_rows=gap_rows,
        summary_manifest_rows=summary_manifest_rows,
        diagnostic_rows=diagnostic_rows,
        reliability_rows=reliability_rows,
        support_score_rows=support_score_rows,
        support_weight_rows=support_weight_rows,
        combined_weight_rows=combined_weight_rows,
        split_rows=split_rows,
        alignment_rows=alignment_rows,
        selection_rows=stability_rows,
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


def _topk_existing_plan(
    cfg: DecentralizedSupport8Top3Tau05Config,
    plan: Mapping[str, object],
    ranked_scores: Sequence[SupportScore],
    *,
    k: int,
) -> dict[str, object]:
    calibrated = {row.expert_id: float(row.calibrated_support_nelbo) for row in ranked_scores}
    ordered = sorted(
        tuple(str(v) for v in plan["sources"]),
        key=lambda source: (-float(plan["scores"][source]), calibrated.get(source, math.inf), source),
    )
    sources = tuple(ordered[: int(k)])
    raw_scores = {source: float(plan["scores"][source]) for source in sources}
    total = sum(raw_scores.values())
    if not math.isfinite(total) or total <= 0.0:
        raise ProtocolError("Top-k combined source scores are not positive finite values.")
    weights = {source: raw_scores[source] / total for source in sources}
    budgets = d12._weighted_budgets(cfg.synthetic_per_class_total, sources, weights, cfg.min_per_source_per_class)
    out = d12._with_weight_diagnostics(sources, weights, budgets, raw_scores)
    for key in ("support_weights", "support_scores", "raw_support_nelbo", "calibrated_support_nelbo", "reliability_scores"):
        values = dict(plan.get(key, {}))
        selected = {source: values[source] for source in sources if source in values}
        if key == "support_weights":
            denom = sum(float(v) for v in selected.values())
            if denom > 0.0:
                selected = {source: float(value) / denom for source, value in selected.items()}
        out[key] = selected
    out.update(
        {
            "use_calibrated_support_nelbo": bool(plan.get("use_calibrated_support_nelbo", True)),
            "include_reliability": bool(plan.get("include_reliability", True)),
            "support_nelbo_tau": float(plan.get("support_nelbo_tau", cfg.support_nelbo_tau)),
        }
    )
    return out


def _uniform_selected_plan(
    cfg: DecentralizedSupport8Top3Tau05Config,
    sources: Sequence[str],
    source_plan: Mapping[str, object],
) -> dict[str, object]:
    sources_tuple = tuple(str(source) for source in sources)
    if not sources_tuple:
        raise ProtocolError("Cannot build an equal selected-source plan without sources.")
    weight = 1.0 / float(len(sources_tuple))
    weights = {source: weight for source in sources_tuple}
    scores = {source: float(source_plan["scores"][source]) for source in sources_tuple}
    budgets = {source: int(value) for source, value in zip(sources_tuple, d1._balanced_counts(cfg.synthetic_per_class_total, len(sources_tuple)))}
    out = d12._with_weight_diagnostics(sources_tuple, weights, budgets, scores)
    for key in ("support_weights", "support_scores", "raw_support_nelbo", "calibrated_support_nelbo", "reliability_scores"):
        values = dict(source_plan.get(key, {}))
        out[key] = {source: values[source] for source in sources_tuple if source in values}
    out.update(
        {
            "use_calibrated_support_nelbo": bool(source_plan.get("use_calibrated_support_nelbo", True)),
            "include_reliability": bool(source_plan.get("include_reliability", False)),
            "support_nelbo_tau": float(source_plan.get("support_nelbo_tau", cfg.support_nelbo_tau)),
        }
    )
    return out


def _evaluate_support8_variant(
    cfg: DecentralizedSupport8Top3Tau05Config,
    **kwargs: Any,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    rows, late, coverage, weak, nn = d13._evaluate_support_variant(cfg, **kwargs)
    return [_extend_support8_row(row) for row in rows], [_extend_support8_row(row) for row in late], coverage, weak, nn


def _extend_support8_row(
    row: Mapping[str, object],
    *,
    support_context: Mapping[str, object] | None = None,
) -> dict[str, object]:
    out = d13._extend_support_row(row, support_context=support_context) if support_context is not None else dict(row)
    out.update(
        {
            "delta_vs_d1_2_reliability_support8_reference": math.nan,
            "delta_vs_equal_support8_reference": math.nan,
            "strongest_negative_control_gap": math.nan,
            "shuffled_support_control_gap": math.nan,
        }
    )
    return out


def _rename_real_feature_reference(row: Mapping[str, object]) -> dict[str, object]:
    out = dict(row)
    if out.get("prior_method") in {d1a.ROW_REAL_FEATURE_DENSE_REFERENCE, d13.ROW_REAL_FEATURE_DENSE_REFERENCE}:
        out["prior_method"] = ROW_REAL_FEATURE_SUPPORT8
    return out


def _append_support8_single_source_references(
    cfg: DecentralizedSupport8Top3Tau05Config,
    matrix_rows: list[dict[str, object]],
    single_rows: Sequence[Mapping[str, object]],
    *,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    real_feature_bacc: float,
    support_context: Mapping[str, object],
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
        matrix_rows.append(_extend_support8_row(row, support_context=support_context))


def _alignment_row_support8(
    ranked_scores: Sequence[SupportScore],
    equal_late_rows: Sequence[Mapping[str, object]],
    primary_rows: Sequence[Mapping[str, object]],
    *,
    selected_sources: Sequence[str],
    support_context: Mapping[str, object],
) -> dict[str, object]:
    downstream = {
        str(row["expert_id"]): _float(row.get("bacc"))
        for row in equal_late_rows
        if row.get("pooling_rule") == "single_source" and row.get("status") == "ok"
    }
    method_baccs = {"all4_geom": _float(primary_rows[0].get("bacc")) if primary_rows else math.nan}
    alignment = ranking_alignment(ranked_scores=ranked_scores, downstream_bacc_by_expert=downstream, method_baccs=method_baccs)
    oracle = _oracle_source(equal_late_rows)
    selected = set(str(source) for source in selected_sources)
    return {
        "experiment_seed": ranked_scores[0].experiment_seed if ranked_scores else "",
        "heldout_center": support_context.get("heldout_center", ""),
        "support_seed": support_context.get("support_seed", ""),
        "replicate_seed": support_context.get("replicate_seed", support_context.get("support_seed", "")),
        "support_size": support_context.get("support_size", ""),
        "support_eval_split_id": support_context.get("support_eval_split_id", ""),
        "top1_downstream_oracle_hit": alignment["top1_downstream_oracle_hit"],
        "top2_downstream_oracle_containment": alignment["top2_oracle_containment"],
        "top3_downstream_oracle_containment": float(str(oracle) in selected) if oracle else math.nan,
        "spearman_support_nelbo_vs_downstream_utility": alignment["spearman_support_nelbo_vs_downstream_bacc"],
        "downstream_oracle_gap": alignment["oracle_gap_all4"],
    }


def _oracle_source(single_rows: Sequence[Mapping[str, object]]) -> str:
    rows = [
        row for row in single_rows
        if row.get("pooling_rule") == "single_source" and row.get("status") == "ok" and math.isfinite(_float(row.get("bacc")))
    ]
    if not rows:
        return ""
    return str(max(rows, key=lambda row: (_float(row.get("bacc")), str(row.get("expert_id")))).get("expert_id"))


def _selected_source_row(
    plan: Mapping[str, object],
    single_rows: Sequence[Mapping[str, object]],
    context: Mapping[str, object],
) -> dict[str, object]:
    selected = tuple(str(source) for source in plan["sources"])
    oracle = _oracle_source(single_rows)
    return {
        "experiment_seed": context.get("experiment_seed", ""),
        "heldout_center": context.get("heldout_center", ""),
        "support_seed": context.get("support_seed", ""),
        "replicate_seed": context.get("replicate_seed", context.get("support_seed", "")),
        "support_size": context.get("support_size", ""),
        "support_eval_split_id": context.get("support_eval_split_id", ""),
        "top3_selected_sources_by_fold_seed": "|".join(selected),
        "selected_source_1": selected[0] if len(selected) > 0 else "",
        "selected_source_2": selected[1] if len(selected) > 1 else "",
        "selected_source_3": selected[2] if len(selected) > 2 else "",
        "oracle_source": oracle,
        "oracle_in_selected_top3": int(bool(oracle and oracle in selected)),
    }


def _selection_stability_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row.get("experiment_seed")), str(row.get("heldout_center"))), []).append(row)
    group_jaccard: dict[tuple[str, str], float] = {}
    for key, subset in grouped.items():
        sets = [set(str(row.get("top3_selected_sources_by_fold_seed", "")).split("|")) - {""} for row in subset]
        if len(sets) < 2:
            group_jaccard[key] = 1.0 if sets else math.nan
            continue
        vals = []
        for left, right in combinations(sets, 2):
            denom = len(left | right)
            vals.append(len(left & right) / float(denom) if denom else math.nan)
        group_jaccard[key] = nanmean([value for value in vals if math.isfinite(value)])
    out = []
    for row in rows:
        key = (str(row.get("experiment_seed")), str(row.get("heldout_center")))
        copied = dict(row)
        copied["top3_selection_jaccard_across_support_seeds"] = group_jaccard.get(key, math.nan)
        out.append(copied)
    return out


def _ineligible_rows(
    cfg: DecentralizedSupport8Top3Tau05Config,
    *,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    support_context: Mapping[str, object],
    status: str,
    error_message: str,
) -> list[dict[str, object]]:
    rows = []
    empty: dict[tuple[str, int], d1a.AdaptiveSourceLocalSummary] = {}
    for method, role in _support8_methods():
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
        rows.append(_extend_support8_row(row, support_context=support_context))
    return rows


def _support8_methods() -> tuple[tuple[str, str], ...]:
    return (
        (PRIMARY_SUPPORT8_TOP3_TAU05_METHOD, "primary_locked_support8_top3_tau05_confirmation"),
        (ROW_D1_2_SUPPORT8_REFERENCE, "matched_d1_2_reliability_all4_support8_reference"),
        (ROW_EQUAL_SUPPORT8_REFERENCE, "matched_equal_all4_support8_reference"),
        (ROW_SUPPORT_ONLY_TOP3, "support_nelbo_only_top3_tau05_reference"),
        (ROW_SINGLE_MEAN, "single_source_mean_reference"),
        (ROW_SINGLE_ORACLE, "diagnostic_only_oracle_reference"),
        (ROW_REAL_FEATURE_SUPPORT8, "real_feature_transfer_ceiling_reference"),
        (ROW_ALL4_TAU05, "support_nelbo_x_reliability_all4_tau05_ablation"),
        (ROW_TOP3_TAU1, "support_nelbo_x_reliability_top3_tau1_ablation"),
        (ROW_ALL4_TAU1, "support_nelbo_x_reliability_all4_tau1_ablation"),
        (ROW_RELIABILITY_TOP3, "reliability_only_top3_diagnostic"),
        (ROW_SUPPORT_ONLY_TOP3_EQUAL_BUDGET, "support_nelbo_only_top3_equal_budget_ablation"),
        (ROW_SHUFFLED_SUPPORT_CONTROL, "negative_control"),
        (ROW_SHUFFLED_SUMMARY_CONTROL, "negative_control"),
        (ROW_SHUFFLED_LABEL_CONTROL, "negative_control"),
    )


def _populate_support8_deltas(rows: list[dict[str, object]]) -> None:
    d12_ref: dict[tuple[str, str, str], float] = {}
    equal_ref: dict[tuple[str, str, str], float] = {}
    shuffled_support: dict[tuple[str, str, str], float] = {}
    controls: dict[tuple[str, str, str], float] = {}
    control_methods = {ROW_SHUFFLED_SUPPORT_CONTROL, ROW_SHUFFLED_SUMMARY_CONTROL, ROW_SHUFFLED_LABEL_CONTROL}
    for row in rows:
        key = (str(row.get("experiment_seed")), str(row.get("heldout_center")), str(row.get("replicate_seed")))
        value = _float(row.get("bacc"))
        if not math.isfinite(value):
            continue
        if row.get("prior_method") == ROW_D1_2_SUPPORT8_REFERENCE:
            d12_ref[key] = value
        elif row.get("prior_method") == ROW_EQUAL_SUPPORT8_REFERENCE:
            equal_ref[key] = value
        elif row.get("prior_method") == ROW_SHUFFLED_SUPPORT_CONTROL:
            shuffled_support[key] = value
            controls[key] = max(controls.get(key, -math.inf), value)
        elif row.get("prior_method") in control_methods:
            controls[key] = max(controls.get(key, -math.inf), value)
    for row in rows:
        if row.get("prior_method") != PRIMARY_SUPPORT8_TOP3_TAU05_METHOD:
            continue
        key = (str(row.get("experiment_seed")), str(row.get("heldout_center")), str(row.get("replicate_seed")))
        value = _float(row.get("bacc"))
        d12_value = d12_ref.get(key, math.nan)
        equal_value = equal_ref.get(key, math.nan)
        control_value = controls.get(key, math.nan)
        shuffled_value = shuffled_support.get(key, math.nan)
        if math.isfinite(value) and math.isfinite(d12_value):
            row["delta_vs_d1_2_reliability_support8_reference"] = value - d12_value
        if math.isfinite(value) and math.isfinite(equal_value):
            row["delta_vs_equal_support8_reference"] = value - equal_value
        if math.isfinite(value) and math.isfinite(control_value):
            row["strongest_negative_control_gap"] = value - control_value
            row["negative_control_gap"] = value - control_value
        if math.isfinite(value) and math.isfinite(shuffled_value):
            row["shuffled_support_control_gap"] = value - shuffled_value


def _grouped_cell_means(rows: Sequence[Mapping[str, object]]) -> dict[tuple[str, str], float]:
    groups: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        if row.get("status") == "ok":
            groups.setdefault((str(row["experiment_seed"]), str(row["heldout_center"])), []).append(row)
    return {key: d1._mean_field(values, "bacc") for key, values in groups.items()}


def _centerwise_delta_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    primary = _grouped_cell_means(d1a._rows_for(rows, PRIMARY_SUPPORT8_TOP3_TAU05_METHOD))
    d12_ref = _grouped_cell_means(d1a._rows_for(rows, ROW_D1_2_SUPPORT8_REFERENCE))
    equal_ref = _grouped_cell_means(d1a._rows_for(rows, ROW_EQUAL_SUPPORT8_REFERENCE))
    by_center: dict[str, list[float]] = {}
    by_seed: dict[str, list[float]] = {}
    by_center_equal: dict[str, list[float]] = {}
    for key, value in primary.items():
        baseline = d12_ref.get(key, math.nan)
        equal = equal_ref.get(key, math.nan)
        seed, center = key
        if math.isfinite(value) and math.isfinite(baseline):
            delta = value - baseline
            by_center.setdefault(center, []).append(delta)
            by_seed.setdefault(seed, []).append(delta)
        if math.isfinite(value) and math.isfinite(equal):
            by_center_equal.setdefault(center, []).append(value - equal)
    rows_out = [
        {
            "axis": "center",
            "id": key,
            "delta_vs_d1_2_reliability_support8_reference": nanmean(values),
            "delta_vs_equal_support8_reference": nanmean(by_center_equal.get(key, [])),
            "n_cells": len(values),
        }
        for key, values in sorted(by_center.items())
    ]
    rows_out.extend(
        {
            "axis": "seed",
            "id": key,
            "delta_vs_d1_2_reliability_support8_reference": nanmean(values),
            "delta_vs_equal_support8_reference": math.nan,
            "n_cells": len(values),
        }
        for key, values in sorted(by_seed.items())
    )
    return rows_out


def _decision(
    rows: Sequence[Mapping[str, object]],
    cfg: DecentralizedSupport8Top3Tau05Config,
    *,
    leakage_status: str,
    support_score_rows: Sequence[Mapping[str, object]],
    support_weight_rows: Sequence[Mapping[str, object]],
    combined_weight_rows: Sequence[Mapping[str, object]],
    split_rows: Sequence[Mapping[str, object]],
    alignment_rows: Sequence[Mapping[str, object]],
    centerwise_rows: Sequence[Mapping[str, object]],
    selection_rows: Sequence[Mapping[str, object]],
    d1_3_top3_context_containment: float,
) -> dict[str, object]:
    primary = d1a._rows_for(rows, PRIMARY_SUPPORT8_TOP3_TAU05_METHOD)
    d12_ref = d1a._rows_for(rows, ROW_D1_2_SUPPORT8_REFERENCE)
    equal = d1a._rows_for(rows, ROW_EQUAL_SUPPORT8_REFERENCE)
    single_mean = d1a._rows_for(rows, ROW_SINGLE_MEAN)
    single_oracle = d1a._rows_for(rows, ROW_SINGLE_ORACLE)
    real_feature = d1a._rows_for(rows, ROW_REAL_FEATURE_SUPPORT8)
    stats = d1a._primary_stats(primary)
    d12_stats = d1a._primary_stats(d12_ref)
    equal_stats = d1a._primary_stats(equal)
    single_mean_stats = d1a._primary_stats(single_mean)
    single_oracle_stats = d1a._primary_stats(single_oracle)
    real_stats = d1a._primary_stats(real_feature)
    control_by_method = {
        ROW_SHUFFLED_SUPPORT_CONTROL: d1a._primary_stats(d1a._rows_for(rows, ROW_SHUFFLED_SUPPORT_CONTROL)),
        ROW_SHUFFLED_SUMMARY_CONTROL: d1a._primary_stats(d1a._rows_for(rows, ROW_SHUFFLED_SUMMARY_CONTROL)),
        ROW_SHUFFLED_LABEL_CONTROL: d1a._primary_stats(d1a._rows_for(rows, ROW_SHUFFLED_LABEL_CONTROL)),
    }
    strongest_control_method, strongest_control_bacc = d13._strongest_control(control_by_method)
    delta_vs_d12 = _float(stats["center_equal_mean_bacc"]) - _float(d12_stats["center_equal_mean_bacc"])
    delta_vs_equal = _float(stats["center_equal_mean_bacc"]) - _float(equal_stats["center_equal_mean_bacc"])
    delta_vs_mean_single = _float(stats["center_equal_mean_bacc"]) - _float(single_mean_stats["center_equal_mean_bacc"])
    delta_vs_oracle_single = _float(stats["center_equal_mean_bacc"]) - _float(single_oracle_stats["center_equal_mean_bacc"])
    delta_vs_real = _float(stats["center_equal_mean_bacc"]) - _float(real_stats["center_equal_mean_bacc"])
    strongest_negative_control_gap = _float(stats["center_equal_mean_bacc"]) - strongest_control_bacc
    shuffled_support_gap = _float(stats["center_equal_mean_bacc"]) - _float(
        control_by_method[ROW_SHUFFLED_SUPPORT_CONTROL]["center_equal_mean_bacc"]
    )
    center_deltas = {str(row["id"]): _float(row["delta_vs_d1_2_reliability_support8_reference"]) for row in centerwise_rows if row.get("axis") == "center"}
    seed_deltas = {str(row["id"]): _float(row["delta_vs_d1_2_reliability_support8_reference"]) for row in centerwise_rows if row.get("axis") == "seed"}
    centers_beating = sum(1 for value in center_deltas.values() if math.isfinite(value) and value > 0.0)
    seeds_beating = sum(1 for value in seed_deltas.values() if math.isfinite(value) and value > 0.0)
    spearman_values = [_float(row.get("spearman_support_nelbo_vs_downstream_utility")) for row in alignment_rows]
    spearman = nanmean([value for value in spearman_values if math.isfinite(value)])
    top1 = nanmean([_float(row.get("top1_downstream_oracle_hit")) for row in alignment_rows])
    top2 = nanmean([_float(row.get("top2_downstream_oracle_containment")) for row in alignment_rows])
    top3 = nanmean([_float(row.get("top3_downstream_oracle_containment")) for row in alignment_rows])
    oracle_gap = nanmean([_float(row.get("downstream_oracle_gap")) for row in alignment_rows])
    seedwise_spearman = d13._axis_means(alignment_rows, "experiment_seed", "spearman_support_nelbo_vs_downstream_utility")
    centerwise_spearman = d13._axis_means(alignment_rows, "heldout_center", "spearman_support_nelbo_vs_downstream_utility")
    weight_stats = _combined_weight_diagnostics(combined_weight_rows)
    selection_jaccard = nanmean([_float(row.get("top3_selection_jaccard_across_support_seeds")) for row in selection_rows])
    oracle_selected_rate = nanmean([_float(row.get("oracle_in_selected_top3")) for row in selection_rows])
    primary_center_equal = _float(stats["center_equal_mean_bacc"])
    primary_min_center = _float(stats["min_center_mean_bacc"])
    primary_seed_std = _float(stats["seed_std_bacc"])
    eligible_primary_cells = len(primary)
    pass_rule = (
        leakage_status == "PASS"
        and int(stats["n_heldout_centers"]) >= len(cfg.heldout_centers)
        and eligible_primary_cells >= 42
        and primary_center_equal >= 0.87
        and primary_min_center >= 0.80
        and primary_seed_std <= 0.06
        and delta_vs_d12 >= 0.01
        and delta_vs_equal >= 0.01
        and centers_beating >= 4
        and seeds_beating >= 2
        and spearman >= 0.20
        and top3 >= 0.85
        and strongest_negative_control_gap >= 0.03
        and shuffled_support_gap >= 0.03
    )
    thesis_partial = (
        leakage_status == "PASS"
        and primary_center_equal >= 0.85
        and primary_min_center >= 0.75
        and delta_vs_d12 >= 0.01
        and delta_vs_equal >= 0.01
        and strongest_negative_control_gap >= 0.01
        and shuffled_support_gap >= 0.01
        and math.isfinite(d1_3_top3_context_containment)
        and top3 > d1_3_top3_context_containment
    )
    weak_pass = (
        leakage_status == "PASS"
        and delta_vs_d12 >= 0.01
        and delta_vs_equal >= 0.01
        and primary_min_center >= _float(d12_stats["min_center_mean_bacc"])
    )
    diagnostic_only = (
        leakage_status == "PASS"
        and (delta_vs_d12 > 0.0 or delta_vs_equal > 0.0)
        and math.isfinite(shuffled_support_gap)
        and shuffled_support_gap < 0.03
    )
    verdict = "D1_3_1_FAIL"
    if leakage_status != "PASS":
        verdict = "PROTOCOL_FAIL"
    elif pass_rule:
        verdict = "D1_3_1_PASS"
    elif thesis_partial:
        verdict = "D1_3_1_THESIS_PARTIAL"
    elif weak_pass:
        verdict = "D1_3_1_WEAK_PASS"
    elif diagnostic_only:
        verdict = "D1_3_1_DIAGNOSTIC_ONLY"
    flags = []
    if eligible_primary_cells < 42:
        flags.append("ELIGIBLE_PRIMARY_SUPPORT8_CELLS_BELOW_42")
    if math.isfinite(delta_vs_d12) and delta_vs_d12 < 0.01:
        flags.append("DELTA_VS_D1_2_SUPPORT8_BELOW_0P01")
    if math.isfinite(delta_vs_equal) and delta_vs_equal < 0.01:
        flags.append("DELTA_VS_EQUAL_SUPPORT8_BELOW_0P01")
    if centers_beating < 4:
        flags.append("CENTER_CONSISTENCY_BELOW_4_OF_5")
    if seeds_beating < 2:
        flags.append("SEED_CONSISTENCY_BELOW_2_OF_3")
    if math.isfinite(spearman) and spearman < 0.20:
        flags.append("SUPPORT_NELBO_SPEARMAN_BELOW_0P20")
    if math.isfinite(top3) and top3 < 0.85:
        flags.append("TOP3_ORACLE_CONTAINMENT_BELOW_0P85")
    if math.isfinite(strongest_negative_control_gap) and strongest_negative_control_gap < 0.03:
        flags.append("STRONGEST_NEGATIVE_CONTROL_GAP_BELOW_0P03")
    if math.isfinite(shuffled_support_gap) and shuffled_support_gap < 0.03:
        flags.append("SHUFFLED_SUPPORT_CONTROL_COMPETITIVE")
    return {
        "primary_verdict": verdict,
        "diagnostic_flags": "|".join(flags),
        "leakage_status": leakage_status,
        "primary_method": PRIMARY_SUPPORT8_TOP3_TAU05_METHOD,
        "center_equal_mean_bacc": stats["center_equal_mean_bacc"],
        "center_equal_macro_f1": stats["center_equal_macro_f1"],
        "min_center_bacc": stats["min_center_mean_bacc"],
        "seed_std_bacc": stats["seed_std_bacc"],
        "eligible_primary_support8_cells": eligible_primary_cells,
        "delta_vs_d1_2_reliability_support8_reference": delta_vs_d12,
        "delta_vs_equal_support8_reference": delta_vs_equal,
        "delta_vs_mean_single_source_adaptive_k": delta_vs_mean_single,
        "delta_vs_single_source_oracle_adaptive_k": delta_vs_oracle_single,
        "delta_vs_real_feature_dense_support8_reference": delta_vs_real,
        "top1_downstream_oracle_hit": top1,
        "top2_downstream_oracle_containment": top2,
        "top3_downstream_oracle_containment": top3,
        "spearman_support_nelbo_vs_downstream_utility": spearman,
        "seedwise_spearman_support_nelbo_vs_downstream_utility_json": json.dumps(seedwise_spearman, sort_keys=True),
        "centerwise_spearman_support_nelbo_vs_downstream_utility_json": json.dumps(centerwise_spearman, sort_keys=True),
        "downstream_oracle_gap": oracle_gap,
        "strongest_negative_control_method": strongest_control_method,
        "strongest_negative_control_gap": strongest_negative_control_gap,
        "strongest_negative_control_center_equal_mean_bacc": strongest_control_bacc,
        "shuffled_support_control_gap": shuffled_support_gap,
        "shuffled_support_control_center_equal_mean_bacc": control_by_method[ROW_SHUFFLED_SUPPORT_CONTROL]["center_equal_mean_bacc"],
        "centerwise_delta_vs_d1_2_support8_json": json.dumps(center_deltas, sort_keys=True),
        "seedwise_delta_vs_d1_2_support8_json": json.dumps(seed_deltas, sort_keys=True),
        "centers_beating_d1_2_support8": centers_beating,
        "seeds_beating_d1_2_support8": seeds_beating,
        "mean_top3_selection_jaccard": selection_jaccard,
        "oracle_in_selected_top3_rate": oracle_selected_rate,
        "d1_3_support32_top3_context_top3_containment": d1_3_top3_context_containment,
        "d1_2_reliability_support8_center_equal_mean_bacc": d12_stats["center_equal_mean_bacc"],
        "equal_support8_center_equal_mean_bacc": equal_stats["center_equal_mean_bacc"],
        "mean_single_source_adaptive_k_center_equal_mean_bacc": single_mean_stats["center_equal_mean_bacc"],
        "single_source_oracle_adaptive_k_center_equal_mean_bacc": single_oracle_stats["center_equal_mean_bacc"],
        "real_feature_dense_support8_center_equal_mean_bacc": real_stats["center_equal_mean_bacc"],
        "support_score_rows": len(support_score_rows),
        "support_weight_rows": len(support_weight_rows),
        "combined_weight_rows": len(combined_weight_rows),
        "support_eval_split_rows": len(split_rows),
        **weight_stats,
        **stats,
    }


def _combined_weight_diagnostics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    cells: dict[tuple[str, str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        cells.setdefault((str(row["experiment_seed"]), str(row["heldout_center"]), str(row["replicate_seed"])), []).append(row)
    entropies = [_float(values[0].get("weight_entropy")) for values in cells.values() if values]
    eff = [_float(values[0].get("effective_num_sources")) for values in cells.values() if values]
    l1_uniform = [_float(values[0].get("l1_distance_from_uniform")) for values in cells.values() if values]
    weights = [_float(row.get("combined_weight")) for row in rows]
    return {
        "mean_effective_num_sources": nanmean([value for value in eff if math.isfinite(value)]),
        "mean_weight_entropy": nanmean([value for value in entropies if math.isfinite(value)]),
        "mean_l1_distance_from_uniform": nanmean([value for value in l1_uniform if math.isfinite(value)]),
        "max_weight_per_fold": max([value for value in weights if math.isfinite(value)], default=math.nan),
        "min_weight_per_fold": min([value for value in weights if math.isfinite(value)], default=math.nan),
    }


def _load_d1_3_context(root: Path | None) -> tuple[list[dict[str, object]], float]:
    if root is None:
        return _missing_context_rows(), math.nan
    matrix_path = root / "tables" / "decentralized_support_nelbo_reliability_downstream_matrix.csv"
    rows: list[dict[str, object]] = []
    if matrix_path.exists():
        with matrix_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if str(row.get("support_size")) != "32":
                    continue
                if row.get("prior_method") == d13.PRIMARY_SUPPORT_RELIABILITY_METHOD:
                    copied = dict(row)
                    copied["prior_method"] = ROW_D13_PRIMARY_CONTEXT
                elif row.get("prior_method") == d13.ROW_TOP3:
                    copied = dict(row)
                    copied["prior_method"] = ROW_D13_TOP3_CONTEXT
                else:
                    continue
                copied["claim_role"] = "historical_d1_3_context_only"
                copied["selection_source"] = DIAGNOSTIC_SELECTION
                rows.append(copied)
    if not rows:
        rows = _missing_context_rows()
    align_path = root / "tables" / "support_nelbo_alignment_matrix.csv"
    top3 = math.nan
    if align_path.exists():
        with align_path.open(newline="", encoding="utf-8") as f:
            vals = [
                _float(row.get("top3_downstream_oracle_containment"))
                for row in csv.DictReader(f)
                if str(row.get("support_size")) == "32"
            ]
        top3 = nanmean([value for value in vals if math.isfinite(value)])
    return rows, top3


def _missing_context_rows() -> list[dict[str, object]]:
    return [
        {"prior_method": ROW_D13_PRIMARY_CONTEXT, "status": "missing_context_reference", "claim_role": "historical_d1_3_context_only"},
        {"prior_method": ROW_D13_TOP3_CONTEXT, "status": "missing_context_reference", "claim_role": "historical_d1_3_context_only"},
    ]


def _write_artifacts(
    root: Path,
    cfg: DecentralizedSupport8Top3Tau05Config,
    *,
    matrix_rows: Sequence[Mapping[str, object]],
    gap_rows: Sequence[Mapping[str, object]],
    summary_manifest_rows: Sequence[Mapping[str, object]],
    diagnostic_rows: Sequence[Mapping[str, object]],
    reliability_rows: Sequence[Mapping[str, object]],
    support_score_rows: Sequence[Mapping[str, object]],
    support_weight_rows: Sequence[Mapping[str, object]],
    combined_weight_rows: Sequence[Mapping[str, object]],
    split_rows: Sequence[Mapping[str, object]],
    alignment_rows: Sequence[Mapping[str, object]],
    selection_rows: Sequence[Mapping[str, object]],
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
    write_csv_rows(root / "tables" / "decentralized_support8_top3_tau05_downstream_matrix.csv", matrix_rows, columns=_matrix_columns())
    write_csv_rows(root / "tables" / "decentralized_support8_top3_tau05_gap_summary.csv", gap_rows, columns=_matrix_columns())
    write_csv_rows(root / "tables" / "decentralized_support8_top3_tau05_summary.csv", [dict(decision)])
    write_csv_rows(root / "tables" / "support_eval_split_manifest.csv", split_rows)
    write_csv_rows(root / "tables" / "support_nelbo_scores.csv", support_score_rows)
    write_csv_rows(root / "tables" / "support_nelbo_weight_manifest.csv", support_weight_rows)
    write_csv_rows(root / "tables" / "combined_weight_manifest.csv", combined_weight_rows)
    write_csv_rows(root / "tables" / "top3_selection_stability.csv", selection_rows)
    write_csv_rows(root / "tables" / "support_nelbo_alignment_matrix.csv", alignment_rows)
    write_csv_rows(root / "tables" / "source_reliability_manifest.csv", reliability_rows)
    write_csv_rows(root / "tables" / "centerwise_delta_summary.csv", centerwise_rows)
    write_csv_rows(root / "tables" / "late_aggregation_matrix.csv", late_rows, columns=_matrix_columns())
    write_csv_rows(root / "tables" / "real_feature_reference_matrix.csv", real_feature_rows, columns=_matrix_columns())
    write_csv_rows(root / "tables" / "negative_control_summary.csv", [_negative_control_summary(decision)])
    write_csv_rows(root / "tables" / "generated_component_coverage_audit.csv", coverage_rows)
    write_csv_rows(root / "tables" / "weak_source_audit.csv", weak_rows)
    write_csv_rows(root / "tables" / "nearest_neighbor_memorization_audit.csv", nn_rows)
    write_csv_rows(root / "tables" / "exported_source_summary_manifest.csv", summary_manifest_rows, columns=d1a._summary_manifest_columns())
    write_csv_rows(root / "tables" / "source_summary_diagnostics.csv", diagnostic_rows, columns=d1a._diagnostic_columns())
    write_csv_rows(root / "manifests" / "decentralized_support8_top3_tau05_prior_model_manifest.csv", model_manifest_rows)
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
            "schema_version": "cvae_rebuild_decentralized_support8_top3_tau05_protocol_manifest_v1",
            "experiment_name": cfg.name,
            "experiment_type": "locked_support8_top3_tau05_target_conditioned_support_nelbo_x_reliability_composition",
            "primary_variant": cfg.primary_variant,
            "primary_method": cfg.primary_method,
            "support_size": cfg.support_size,
            "support_nelbo_tau": cfg.support_nelbo_tau,
            "top_k_sources": cfg.top_k_sources,
            "target_support_labels_for_selection": False,
            "target_eval_labels_for_scoring_only": True,
            "support_eval_disjoint": True,
            "target_expert_excluded": target_expert_excluded,
            "exported_source_summaries_are_target_agnostic": True,
            "raw_source_embedding_pooling_for_prior_fit": False,
            "adaptive_k_selection_uses_source_local_fit_statistics_only": True,
            "source_reliability_uses_source_local_eval_only": True,
            "support_nelbo_uses_unlabeled_target_support_only": True,
            "decision_baselines_recomputed_on_support8_excluded_eval_subset": True,
            "oracle_rows_diagnostic_only": True,
            "context_support32_rows_decision_excluded": True,
            "protocol_wording": PROTOCOL_WORDING,
            "claim_boundary": (
                "bounded support-size-8/top-3/tau-0.5 support-NELBO x reliability composition; no metadata-routing claim, "
                "no formal privacy claim, no centralized source-union deployability claim, and no general support-NELBO claim"
            ),
        },
        resolved_config=_resolved_config(cfg),
    )
    _write_decision_summary(root, decision, leakage_status=leakage_status)


def _matrix_columns() -> tuple[str, ...]:
    return d13._matrix_columns() + (
        "delta_vs_d1_2_reliability_support8_reference",
        "delta_vs_equal_support8_reference",
        "strongest_negative_control_gap",
        "shuffled_support_control_gap",
    )


def _negative_control_summary(decision: Mapping[str, object]) -> dict[str, object]:
    return {
        "primary_method": PRIMARY_SUPPORT8_TOP3_TAU05_METHOD,
        "control_methods": f"{ROW_SHUFFLED_SUPPORT_CONTROL}|{ROW_SHUFFLED_SUMMARY_CONTROL}|{ROW_SHUFFLED_LABEL_CONTROL}",
        "primary_center_equal_mean_bacc": decision.get("center_equal_mean_bacc", math.nan),
        "strongest_negative_control_method": decision.get("strongest_negative_control_method", ""),
        "strongest_negative_control_center_equal_mean_bacc": decision.get("strongest_negative_control_center_equal_mean_bacc", math.nan),
        "strongest_negative_control_gap": decision.get("strongest_negative_control_gap", math.nan),
        "shuffled_support_control_center_equal_mean_bacc": decision.get("shuffled_support_control_center_equal_mean_bacc", math.nan),
        "shuffled_support_control_gap": decision.get("shuffled_support_control_gap", math.nan),
        "negative_control_gap": decision.get("strongest_negative_control_gap", math.nan),
        "control_competitive": _float(decision.get("strongest_negative_control_gap")) < 0.03,
    }


def _write_decision_summary(root: Path, decision: Mapping[str, object], *, leakage_status: str) -> None:
    text = "\n".join(
        [
            "# D1.3.1: Locked Support-Size-8 Top-3 Tau-0.5 Confirmation",
            "",
            "## Summary",
            "",
            f"- Primary method: `{PRIMARY_SUPPORT8_TOP3_TAU05_METHOD}`",
            f"- Primary verdict: `{decision.get('primary_verdict', 'D1_3_1_FAIL')}`",
            f"- Diagnostic flags: `{decision.get('diagnostic_flags', '')}`",
            f"- Center-equal mean BACC: {_format_float(decision.get('center_equal_mean_bacc'))}",
            f"- Center-equal macro-F1: {_format_float(decision.get('center_equal_macro_f1'))}",
            f"- Min center BACC: {_format_float(decision.get('min_center_bacc'))}",
            f"- Seed std BACC: {_format_float(decision.get('seed_std_bacc'))}",
            f"- Delta vs D1.2 support8 reliability reference: {_format_float(decision.get('delta_vs_d1_2_reliability_support8_reference'))}",
            f"- Delta vs equal support8 reference: {_format_float(decision.get('delta_vs_equal_support8_reference'))}",
            f"- Support-NELBO vs downstream Spearman: {_format_float(decision.get('spearman_support_nelbo_vs_downstream_utility'))}",
            f"- Top-3 downstream oracle containment: {_format_float(decision.get('top3_downstream_oracle_containment'))}",
            f"- Strongest negative-control gap: {_format_float(decision.get('strongest_negative_control_gap'))}",
            f"- Shuffled-support control gap: {_format_float(decision.get('shuffled_support_control_gap'))}",
            f"- Mean top-3 selection Jaccard: {_format_float(decision.get('mean_top3_selection_jaccard'))}",
            f"- Oracle in selected top-3 rate: {_format_float(decision.get('oracle_in_selected_top3_rate'))}",
            f"- Leakage status: `{leakage_status}`",
            "",
            "## Protocol Boundary",
            "",
            PROTOCOL_WORDING,
            "",
            "Support-size-8/top-3/tau-0.5 was predeclared for D1.3.1 from prior D1.3 diagnostics.",
            "This is not an in-run hyperparameter search and support-32 context rows do not affect the decision rule.",
            "Support labels are not used for weighting or selection. Target eval labels are scoring-only.",
            "This is not metadata routing and it is not a formal privacy result.",
            "",
            "## Supported Claim If PASS",
            "",
            "Support-size-8, low-temperature, top-3 support-NELBO x reliability composition improves decentralized generated-embedding downstream utility over matched reliability-only and equal dense baselines, with evidence that the support-NELBO signal is not replaceable by shuffled support weights.",
            "",
        ]
    )
    path = root / "reports" / "decision_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _resolved_config(cfg: DecentralizedSupport8Top3Tau05Config) -> dict[str, object]:
    return {
        "name": cfg.name,
        "artifact_root": str(cfg.artifact_root),
        "repair_artifact_root": str(cfg.repair_artifact_root),
        "d1_3_artifact_root": "" if cfg.d1_3_artifact_root is None else str(cfg.d1_3_artifact_root),
        "feature_cache_root": str(cfg.feature_cache_root),
        "backbone": cfg.backbone,
        "experiment_seeds": list(cfg.experiment_seeds),
        "heldout_centers": list(cfg.heldout_centers),
        "replicate_seeds": list(cfg.replicate_seeds),
        "support_seeds": list(cfg.support_seeds),
        "support_size": cfg.support_size,
        "align_support_and_generation_seed": cfg.align_support_and_generation_seed,
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
        "support_nelbo_tau": cfg.support_nelbo_tau,
        "top_k_sources": cfg.top_k_sources,
        "support_alpha": cfg.support_alpha,
        "reliability_alpha": cfg.reliability_alpha,
        "classifier": {
            "type": cfg.classifier_type,
            "solver": cfg.classifier_solver,
            "C": cfg.classifier_c,
            "max_iter": cfg.classifier_max_iter,
            "class_weight": cfg.classifier_class_weight,
            "classifier_seed": cfg.classifier_seed,
        },
    }
