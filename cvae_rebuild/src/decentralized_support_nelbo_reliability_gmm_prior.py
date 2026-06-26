from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from config_sections import experiment_config_sections
from features import load_feature_cache, select_rows
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
    _to_numpy,
)
from preservation_sampling import DIAGNOSTIC_SELECTION, PRIMARY_SELECTION, _manifest_row, _per_source_variant, _runtime_source
from protocol import ProtocolError, assert_candidate_pool, build_leakage_report
from reporting import prepare_artifact_dirs, write_csv_rows, write_json, write_protocol_finalization
from splits import candidate_experts, random_unlabeled_support_eval_split
from support_nelbo import SupportScore, calibration_stats, calibrate, rank_support_scores, ranking_alignment

import decentralized_adaptive_gmm_prior as d1a
import decentralized_k16_gmm_prior as d1
import decentralized_reliability_weighted_gmm_prior as d12


SUPPORT_RELIABILITY_NAME = "virchow2_cvae_decentralized_support_nelbo_reliability_gmm_prior_v1"
PRIMARY_SUPPORT_RELIABILITY_METHOD = "decentralized_exported_adaptive_k_support_nelbo_x_reliability_weighted_geom"
ROW_SUPPORT_ONLY = "decentralized_exported_adaptive_k_support_nelbo_only_weighted_geom"
ROW_D1_2_REFERENCE = "decentralized_exported_adaptive_k_source_reliability_weighted_geom_support_eval_reference"
ROW_EQUAL_REFERENCE = "decentralized_exported_adaptive_k_equal_geom_support_eval_reference"
ROW_TOP3 = "decentralized_exported_adaptive_k_support_nelbo_x_reliability_top3_geom_diagnostic"
ROW_SUPPORT_ONLY_TOP3 = "decentralized_exported_adaptive_k_support_nelbo_only_top3_geom_diagnostic"
ROW_UNCALIBRATED = "decentralized_exported_adaptive_k_support_nelbo_uncalibrated_weighted_geom_diagnostic"
ROW_TAU05 = "decentralized_support_nelbo_tau05_x_reliability_geom_diagnostic"
ROW_TAU20 = "decentralized_support_nelbo_tau20_x_reliability_geom_diagnostic"
ROW_SUPPORT_SIZE8 = "decentralized_support_nelbo_support_size8_x_reliability_geom_diagnostic"
ROW_SUPPORT_SIZE16 = "decentralized_support_nelbo_support_size16_x_reliability_geom_diagnostic"
ROW_SUPPORT_SIZE64 = "decentralized_support_nelbo_support_size64_x_reliability_geom_diagnostic"
ROW_SINGLE_MEAN = d1a.ROW_SINGLE_MEAN
ROW_SINGLE_ORACLE = d1a.ROW_SINGLE_ORACLE
ROW_REAL_FEATURE_DENSE_REFERENCE = "real_source_embedding_classifier_dense_support_eval_reference"
ROW_SHUFFLED_SUPPORT_CONTROL = "decentralized_support_nelbo_shuffled_support_control"
ROW_SHUFFLED_SUMMARY_CONTROL = "decentralized_support_nelbo_shuffled_summary_control"
ROW_SHUFFLED_LABEL_CONTROL = "decentralized_support_nelbo_shuffled_label_control"
PROTOCOL_WORDING = (
    "This is a data-minimizing, raw-data-free source-local latent summary-exchange protocol with "
    "an unlabeled target-support compatibility signal. It is not a formal differential privacy claim. "
    "Exported latent summary statistics may still contain distributional information derived from private data."
)


@dataclass(frozen=True)
class DecentralizedSupportNelboReliabilityConfig:
    name: str
    artifact_root: Path
    repair_artifact_root: Path
    feature_cache_root: Path
    backbone: str
    experiment_seeds: tuple[int, ...]
    heldout_centers: tuple[str, ...]
    replicate_seeds: tuple[int, ...]
    support_seeds: tuple[int, ...]
    support_size: int
    support_size_diagnostics: tuple[int, ...]
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
    tau_diagnostics: tuple[float, ...]
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


def load_decentralized_support_nelbo_reliability_gmm_prior_config(
    path: str | Path,
) -> DecentralizedSupportNelboReliabilityConfig:
    source = Path(path).resolve()
    data = _load_mapping(source)
    base_dir = source.parents[2] if len(source.parents) >= 3 else source.parent
    return parse_decentralized_support_nelbo_reliability_gmm_prior_config(data, base_dir=base_dir)


def parse_decentralized_support_nelbo_reliability_gmm_prior_config(
    data: Mapping[str, Any],
    *,
    base_dir: str | Path = ".",
) -> DecentralizedSupportNelboReliabilityConfig:
    base = Path(base_dir)
    sections = experiment_config_sections(data)
    experiment = sections.experiment
    inputs = sections.inputs
    run = sections.run_matrix
    generation = sections.generation
    gmm = _mapping(data, "support_nelbo_reliability_gmm_prior")
    classifier = sections.classifier
    cfg = DecentralizedSupportNelboReliabilityConfig(
        name=str(experiment["name"]),
        artifact_root=_path(base, str(experiment["artifact_root"])),
        repair_artifact_root=_path(base, str(inputs["repair_artifact_root"])),
        feature_cache_root=_path(base, str(inputs["feature_cache_root"])),
        backbone=str(inputs.get("backbone", "")),
        experiment_seeds=tuple(int(v) for v in run["experiment_seeds"]),
        heldout_centers=tuple(str(v) for v in run["heldout_centers"]),
        replicate_seeds=tuple(int(v) for v in run["replicate_seeds"]),
        support_seeds=tuple(int(v) for v in run["support_seeds"]),
        support_size=int(run["support_size"]),
        support_size_diagnostics=tuple(int(v) for v in run.get("support_size_diagnostics", ())),
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
        tau_diagnostics=tuple(float(v) for v in gmm.get("tau_diagnostics", ())),
        support_alpha=float(gmm["support_alpha"]),
        reliability_alpha=float(gmm["reliability_alpha"]),
        classifier_type=str(classifier["type"]),
        classifier_solver=str(classifier["solver"]),
        classifier_c=float(classifier["C"]),
        classifier_max_iter=int(classifier["max_iter"]),
        classifier_class_weight=str(classifier["class_weight"]),
        classifier_seed=None if classifier.get("classifier_seed") is None else int(classifier["classifier_seed"]),
    )
    validate_decentralized_support_nelbo_reliability_gmm_prior_config(cfg)
    return cfg


def validate_decentralized_support_nelbo_reliability_gmm_prior_config(
    cfg: DecentralizedSupportNelboReliabilityConfig,
) -> None:
    if cfg.name != SUPPORT_RELIABILITY_NAME:
        raise ProtocolError(f"D1.3 experiment name must be {SUPPORT_RELIABILITY_NAME!r}.")
    if cfg.backbone != "virchow2":
        raise ProtocolError("D1.3 is locked to backbone=virchow2.")
    if cfg.primary_variant != PRIMARY_VARIANT:
        raise ProtocolError(f"primary_variant must be {PRIMARY_VARIANT!r}.")
    if cfg.primary_method != PRIMARY_SUPPORT_RELIABILITY_METHOD:
        raise ProtocolError(f"primary_method must be {PRIMARY_SUPPORT_RELIABILITY_METHOD!r}.")
    if cfg.candidate_components_per_source_class != (4, 3, 2, 1):
        raise ProtocolError("candidate_components_per_source_class must be locked to [4, 3, 2, 1].")
    if len(cfg.heldout_centers) != 5:
        raise ProtocolError("D1.3 composition expects exactly five centers, giving four source experts per fold.")
    if cfg.replicate_seeds != cfg.support_seeds or not cfg.align_support_and_generation_seed:
        raise ProtocolError("D1.3 requires support_seeds == replicate_seeds with aligned support/generation seeds.")
    if cfg.support_size != 32:
        raise ProtocolError("Primary D1.3 support_size must be locked to 32.")
    if tuple(cfg.support_size_diagnostics) != (8, 16, 64):
        raise ProtocolError("support_size_diagnostics must be locked to [8, 16, 64].")
    if cfg.source_weighting != "support_nelbo_x_source_local_reliability":
        raise ProtocolError("source_weighting must be support_nelbo_x_source_local_reliability.")
    if cfg.gmm_covariance_type != "diag":
        raise ProtocolError("gmm_covariance_type must be diag.")
    if cfg.primary_pooling != "weighted_geometric":
        raise ProtocolError("primary_pooling must be weighted_geometric.")
    if cfg.synthetic_per_class_total != 128 or cfg.min_per_source_per_class != 8:
        raise ProtocolError("D1.3 synthetic budget must be 128 total with min_per_source_per_class=8.")
    if set(round(v, 6) for v in cfg.tau_diagnostics) != {0.5, 2.0}:
        raise ProtocolError("tau_diagnostics must contain [0.5, 2.0].")
    if min(cfg.min_samples_per_component, cfg.gmm_n_init, cfg.gmm_max_iter) < 1:
        raise ProtocolError("GMM counts and iteration settings must be positive.")
    floors = (
        cfg.gmm_reg_covar,
        cfg.min_component_weight,
        cfg.variance_floor,
        cfg.reliability_floor_score,
        cfg.support_nelbo_tau,
        cfg.support_alpha,
        cfg.reliability_alpha,
    )
    if min(floors) <= 0.0:
        raise ProtocolError("GMM, reliability, support-NELBO tau, and alpha values must be positive.")
    if cfg.classifier_type != "sklearn_logistic_regression":
        raise ProtocolError("classifier.type must be sklearn_logistic_regression.")
    if cfg.classifier_solver != "lbfgs" or cfg.classifier_c != 1.0 or cfg.classifier_max_iter != 2000:
        raise ProtocolError("Classifier solver/C/max_iter must remain locked.")
    if cfg.classifier_class_weight != "balanced" or cfg.classifier_seed is not None:
        raise ProtocolError("Classifier must use class_weight=balanced and classifier_seed=null.")


def run_decentralized_support_nelbo_reliability_gmm_prior(
    cfg: DecentralizedSupportNelboReliabilityConfig,
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
    missing_ref = d1._missing_reference()

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
                support_calibration[str(source_center)] = _source_nelbo_calibration(runtime_source.runtime, str(source_center))

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

                for support_size in (cfg.support_size,) + cfg.support_size_diagnostics:
                    for replicate_seed in cfg.replicate_seeds:
                        support_seed = int(replicate_seed)
                        support_context, eval_raw, eval_labels, support_raw, split_error = _support_eval_context(
                            cfg,
                            test_cache=test_cache,
                            heldout_center=str(heldout_center),
                            support_size=int(support_size),
                            support_seed=support_seed,
                        )
                        support_context = dict(support_context)
                        support_context.update(
                            {
                                "experiment_seed": int(experiment_seed),
                                "replicate_seed": int(replicate_seed),
                            }
                        )
                        split_rows.append(support_context)
                        rels = {
                            source: reliability[(int(experiment_seed), int(replicate_seed), str(source))]
                            for source in candidates
                        }
                        if split_error:
                            methods = _primary_methods_for_support_size(cfg, int(support_size))
                            matrix_rows.extend(
                                _ineligible_rows(
                                    cfg,
                                    experiment_seed=int(experiment_seed),
                                    heldout_center=str(heldout_center),
                                    replicate_seed=int(replicate_seed),
                                    candidates=candidates,
                                    methods=methods,
                                    support_context=support_context,
                                    status="ineligible",
                                    error_message=split_error,
                                )
                            )
                            continue

                        support_scores = _support_scores(
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
                        ranked_scores = tuple(rank_support_scores(support_scores, eligible_count=len(candidates)))

                        primary_plan = _combined_weight_plan(
                            cfg,
                            candidates,
                            rels,
                            ranked_scores,
                            tau=cfg.support_nelbo_tau,
                            use_calibrated=True,
                            include_reliability=True,
                        )
                        support_score_rows.extend(_support_score_rows(ranked_scores, support_context))
                        support_weight_rows.extend(_support_weight_manifest_rows(ranked_scores, support_context, primary_plan))
                        combined_weight_rows.extend(_combined_weight_manifest_rows(support_context, primary_plan, rels))

                        if int(support_size) != cfg.support_size:
                            method = _support_size_method(int(support_size))
                            rows, late, coverage, weak, nn = _evaluate_support_variant(
                                cfg,
                                per_source_runtime=per_source_runtime,
                                candidates=candidates,
                                summaries=largest_summaries,
                                experiment_seed=int(experiment_seed),
                                heldout_center=str(heldout_center),
                                replicate_seed=int(replicate_seed),
                                eval_raw=eval_raw,
                                eval_labels=eval_labels,
                                weight_plan=primary_plan,
                                prior_method=method,
                                pooling_rule="weighted_geometric",
                                support_context=support_context,
                                source_weighting="support_nelbo_x_source_local_reliability",
                                selection_source=DIAGNOSTIC_SELECTION,
                                claim_role="diagnostic_support_size_sensitivity",
                            )
                            matrix_rows.extend(rows)
                            late_rows.extend(late)
                            coverage_rows.extend(coverage)
                            weak_rows.extend(weak)
                            nn_rows.extend(nn)
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
                        ref_row = _rename_real_feature_reference(ref_row)
                        real_late = [_rename_real_feature_reference(row) for row in real_late]
                        ref_row = _extend_support_row(ref_row, support_context=support_context)
                        real_late = [_extend_support_row(row, support_context=support_context) for row in real_late]
                        real_feature_rows.append(ref_row)
                        matrix_rows.append(ref_row)
                        late_rows.extend(real_late)

                        equal_plan = d12._uniform_weight_plan(cfg, candidates, rels)
                        equal_rows, equal_late, coverage, weak, nn = _evaluate_support_variant(
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
                            prior_method=ROW_EQUAL_REFERENCE,
                            pooling_rule="geometric",
                            support_context=support_context,
                            source_weighting="equal_source_mass",
                            selection_source=DIAGNOSTIC_SELECTION,
                            claim_role="d1_1_equal_adaptive_geom_support_eval_reference",
                        )
                        matrix_rows.extend(equal_rows)
                        late_rows.extend(equal_late)
                        coverage_rows.extend(coverage)
                        weak_rows.extend(weak)
                        nn_rows.extend(nn)

                        reliability_plan = d12._weight_plan(cfg, candidates, rels, mode="linear")
                        rel_rows, rel_late, coverage, weak, nn = _evaluate_support_variant(
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
                            prior_method=ROW_D1_2_REFERENCE,
                            pooling_rule="weighted_geometric",
                            support_context=support_context,
                            source_weighting="source_local_reliability",
                            selection_source=DIAGNOSTIC_SELECTION,
                            claim_role="d1_2_reliability_support_eval_reference",
                        )
                        matrix_rows.extend(rel_rows)
                        late_rows.extend(rel_late)
                        coverage_rows.extend(coverage)
                        weak_rows.extend(weak)
                        nn_rows.extend(nn)

                        primary_rows, primary_late, coverage, weak, nn = _evaluate_support_variant(
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
                            weight_plan=primary_plan,
                            prior_method=PRIMARY_SUPPORT_RELIABILITY_METHOD,
                            pooling_rule="weighted_geometric",
                            support_context=support_context,
                            source_weighting="support_nelbo_x_source_local_reliability",
                            selection_source=PRIMARY_SELECTION,
                            claim_role="primary_target_conditioned_support_nelbo_x_reliability_composition",
                        )
                        matrix_rows.extend(primary_rows)
                        late_rows.extend(primary_late)
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
                            support_context=support_context,
                        )
                        alignment = _alignment_row(
                            ranked_scores,
                            equal_late,
                            primary_rows,
                            support_context=support_context,
                        )
                        alignment_rows.append(alignment)
                        _attach_downstream_to_support_scores(support_score_rows, support_context, equal_late)

                        diagnostics = (
                            (
                                ROW_SUPPORT_ONLY,
                                _combined_weight_plan(
                                    cfg,
                                    candidates,
                                    rels,
                                    ranked_scores,
                                    tau=cfg.support_nelbo_tau,
                                    use_calibrated=True,
                                    include_reliability=False,
                                ),
                                "weighted_geometric",
                                "support_nelbo_only_weighted_diagnostic",
                                "support_nelbo_only",
                            ),
                            (
                                ROW_TOP3,
                                _topk_plan(
                                    cfg,
                                    primary_plan,
                                    rels,
                                    ranked_scores,
                                    k=3,
                                    include_reliability=True,
                                ),
                                "weighted_geometric",
                                "support_nelbo_x_reliability_top3_diagnostic",
                                "support_nelbo_x_source_local_reliability_top3",
                            ),
                            (
                                ROW_SUPPORT_ONLY_TOP3,
                                _topk_plan(
                                    cfg,
                                    _combined_weight_plan(
                                        cfg,
                                        candidates,
                                        rels,
                                        ranked_scores,
                                        tau=cfg.support_nelbo_tau,
                                        use_calibrated=True,
                                        include_reliability=False,
                                    ),
                                    rels,
                                    ranked_scores,
                                    k=3,
                                    include_reliability=False,
                                ),
                                "weighted_geometric",
                                "support_nelbo_only_top3_diagnostic",
                                "support_nelbo_only_top3",
                            ),
                            (
                                ROW_UNCALIBRATED,
                                _combined_weight_plan(
                                    cfg,
                                    candidates,
                                    rels,
                                    ranked_scores,
                                    tau=cfg.support_nelbo_tau,
                                    use_calibrated=False,
                                    include_reliability=True,
                                ),
                                "weighted_geometric",
                                "uncalibrated_support_nelbo_diagnostic",
                                "uncalibrated_support_nelbo_x_source_local_reliability",
                            ),
                            (
                                ROW_TAU05,
                                _combined_weight_plan(
                                    cfg,
                                    candidates,
                                    rels,
                                    ranked_scores,
                                    tau=0.5,
                                    use_calibrated=True,
                                    include_reliability=True,
                                ),
                                "weighted_geometric",
                                "tau_sensitivity_diagnostic",
                                "support_nelbo_tau05_x_source_local_reliability",
                            ),
                            (
                                ROW_TAU20,
                                _combined_weight_plan(
                                    cfg,
                                    candidates,
                                    rels,
                                    ranked_scores,
                                    tau=2.0,
                                    use_calibrated=True,
                                    include_reliability=True,
                                ),
                                "weighted_geometric",
                                "tau_sensitivity_diagnostic",
                                "support_nelbo_tau20_x_source_local_reliability",
                            ),
                            (
                                ROW_SHUFFLED_SUPPORT_CONTROL,
                                _shuffled_support_plan(cfg, candidates, rels, ranked_scores, replicate_seed),
                                "weighted_geometric",
                                "negative_control",
                                "shuffled_support_nelbo_control",
                            ),
                        )
                        for method, plan, pooling_rule, role, weighting in diagnostics:
                            rows, late, coverage, weak, nn = _evaluate_support_variant(
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
                            rows, late, coverage, weak, nn = _evaluate_support_variant(
                                cfg,
                                per_source_runtime=per_source_runtime,
                                candidates=candidates,
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
                                source_weighting="support_nelbo_x_source_local_reliability",
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

    leakage = build_leakage_report(
        target_support_labels_for_selection=False,
        target_eval_labels_for_scoring_only=True,
        target_expert_excluded=target_expert_excluded,
        oracle_rows_diagnostic_only=True,
        extra_violations=protocol_violations,
    )
    _populate_delta_and_control_gaps(matrix_rows)
    gap_rows = [dict(row) for row in matrix_rows if row.get("status") == "ok"]
    centerwise_rows = _centerwise_delta_rows(matrix_rows)
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


def _source_nelbo_calibration(runtime: object, expert_id: str) -> object:
    vals = _marginal_nelbo_values(runtime, runtime.source_val_embeddings, already_in_frame=True)
    return calibration_stats(expert_id, vals)


def _marginal_nelbo_values(runtime: object, x: object, *, already_in_frame: bool) -> tuple[float, ...]:
    import torch  # type: ignore

    arr = _to_numpy(x) if already_in_frame else runtime.frame.transform(_to_numpy(x))
    with torch.no_grad():
        values = runtime.model.marginal_nelbo(torch.as_tensor(np.asarray(arr, dtype=np.float32), dtype=torch.float32))
    return tuple(float(v) for v in values.detach().cpu().numpy().tolist())


def _support_eval_context(
    cfg: DecentralizedSupportNelboReliabilityConfig,
    *,
    test_cache: object,
    heldout_center: str,
    support_size: int,
    support_seed: int,
) -> tuple[dict[str, object], object | None, tuple[int, ...], object | None, str]:
    try:
        split = random_unlabeled_support_eval_split(
            test_cache.metadata,
            heldout_center=str(heldout_center),
            support_size=int(support_size),
            support_seed=int(support_seed),
        )
        support_raw, _support_meta = select_rows(test_cache.embeddings, test_cache.metadata, split.support_indices)
        eval_raw, eval_meta = select_rows(test_cache.embeddings, test_cache.metadata, split.eval_indices)
        eval_labels = tuple(_label(row) for row in eval_meta)
        label_counts = {int(label): eval_labels.count(int(label)) for label in set(eval_labels)}
        min_class_count = min(label_counts.values()) if label_counts else 0
        ineligible = "mono_class_target_eval_after_support_split" if len(set(eval_labels)) < 2 else ""
        context = {
            "heldout_center": str(heldout_center),
            "support_size": int(support_size),
            "support_seed": int(support_seed),
            "support_size_requested": split.support_size_requested,
            "support_size_actual": split.support_size_actual,
            "support_eval_split_id": split.support_eval_split_id,
            "support_labels_used": int(split.support_labels_used),
            "n_target_eval_after_support": len(eval_labels),
            "support_eval_min_class_count": int(min_class_count),
            "support_eval_ineligible_reason": ineligible,
            "support_sample_id_hash": d12._hash_strings(split.support_sample_ids),
            "eval_sample_id_hash": d12._hash_strings(split.eval_sample_ids),
        }
        return context, eval_raw, eval_labels, support_raw, ineligible
    except Exception as exc:
        context = {
            "heldout_center": str(heldout_center),
            "support_size": int(support_size),
            "support_seed": int(support_seed),
            "support_size_requested": int(support_size),
            "support_size_actual": 0,
            "support_eval_split_id": f"target{heldout_center}_seed{support_seed}_random_unlabeled_k{support_size}",
            "support_labels_used": 0,
            "n_target_eval_after_support": 0,
            "support_eval_min_class_count": 0,
            "support_eval_ineligible_reason": str(exc),
            "support_sample_id_hash": "",
            "eval_sample_id_hash": "",
        }
        return context, None, tuple(), None, str(exc)


def _support_scores(
    cfg: DecentralizedSupportNelboReliabilityConfig,
    *,
    per_source_runtime: Mapping[str, object],
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
        vals = _marginal_nelbo_values(runtime, support_raw, already_in_frame=False)
        raw = nanmean(vals)
        score = SupportScore(
            experiment_seed=int(experiment_seed),
            heldout_center=str(heldout_center),
            support_seed=int(support_seed),
            support_size=int(support_size),
            expert_id=str(source),
            raw_support_nelbo=float(raw),
            calibrated_support_nelbo=float(calibrate(float(raw), support_calibration[str(source)])),
        )
        scores.append(score)
    return tuple(scores)


def _combined_weight_plan(
    cfg: DecentralizedSupportNelboReliabilityConfig,
    sources: Sequence[str],
    rels: Mapping[str, d12.SourceReliability],
    ranked_scores: Sequence[SupportScore],
    *,
    tau: float,
    use_calibrated: bool,
    include_reliability: bool,
) -> dict[str, object]:
    sources_tuple = tuple(str(source) for source in sources)
    score_by_source = {row.expert_id: row for row in ranked_scores}
    support_logits = []
    for source in sources_tuple:
        row = score_by_source[source]
        value = row.calibrated_support_nelbo if use_calibrated else row.raw_support_nelbo
        support_logits.append(-float(value) / float(tau))
    support_weights_arr = _softmax(support_logits)
    support_weights = {source: float(weight) for source, weight in zip(sources_tuple, support_weights_arr)}
    reliability_scores = {source: float(rels[source].reliability_score) for source in sources_tuple}
    raw_combined = {}
    for source in sources_tuple:
        support_component = float(support_weights[source]) ** float(cfg.support_alpha)
        reliability_component = float(reliability_scores[source]) ** float(cfg.reliability_alpha) if include_reliability else 1.0
        raw_combined[source] = support_component * reliability_component
    total = sum(raw_combined.values())
    if not math.isfinite(total) or total <= 0.0:
        raise ProtocolError("Support-NELBO combined source weights are not positive finite values.")
    weights = {source: float(raw_combined[source] / total) for source in sources_tuple}
    budgets = d12._weighted_budgets(cfg.synthetic_per_class_total, sources_tuple, weights, cfg.min_per_source_per_class)
    plan = d12._with_weight_diagnostics(sources_tuple, weights, budgets, raw_combined)
    plan.update(
        {
            "support_weights": support_weights,
            "support_scores": {source: float(support_logits[idx]) for idx, source in enumerate(sources_tuple)},
            "raw_support_nelbo": {source: float(score_by_source[source].raw_support_nelbo) for source in sources_tuple},
            "calibrated_support_nelbo": {source: float(score_by_source[source].calibrated_support_nelbo) for source in sources_tuple},
            "reliability_scores": reliability_scores,
            "use_calibrated_support_nelbo": bool(use_calibrated),
            "include_reliability": bool(include_reliability),
            "support_nelbo_tau": float(tau),
        }
    )
    return plan


def _softmax(values: Sequence[float]) -> tuple[float, ...]:
    vals = np.asarray([float(v) for v in values], dtype=float)
    if np.any(~np.isfinite(vals)):
        raise ProtocolError("Support-NELBO softmax received non-finite values.")
    shifted = vals - float(np.max(vals))
    exp_vals = np.exp(shifted)
    denom = float(exp_vals.sum())
    if not math.isfinite(denom) or denom <= 0.0:
        raise ProtocolError("Support-NELBO softmax failed.")
    return tuple(float(v / denom) for v in exp_vals)


def _topk_plan(
    cfg: DecentralizedSupportNelboReliabilityConfig,
    primary_plan: Mapping[str, object],
    rels: Mapping[str, d12.SourceReliability],
    ranked_scores: Sequence[SupportScore],
    *,
    k: int,
    include_reliability: bool,
) -> dict[str, object]:
    ordered = sorted(tuple(str(v) for v in primary_plan["sources"]), key=lambda source: (-float(primary_plan["weights"][source]), source))
    sources = tuple(ordered[: int(k)])
    return _combined_weight_plan(
        cfg,
        sources,
        {source: rels[source] for source in sources},
        tuple(row for row in ranked_scores if row.expert_id in sources),
        tau=float(primary_plan["support_nelbo_tau"]),
        use_calibrated=bool(primary_plan["use_calibrated_support_nelbo"]),
        include_reliability=include_reliability,
    )


def _shuffled_support_plan(
    cfg: DecentralizedSupportNelboReliabilityConfig,
    candidates: Sequence[str],
    rels: Mapping[str, d12.SourceReliability],
    ranked_scores: Sequence[SupportScore],
    replicate_seed: int,
) -> dict[str, object]:
    plan = _combined_weight_plan(
        cfg,
        candidates,
        rels,
        ranked_scores,
        tau=cfg.support_nelbo_tau,
        use_calibrated=True,
        include_reliability=True,
    )
    sources = tuple(str(v) for v in plan["sources"])
    shuffled = list(plan["support_weights"].values())
    rng = random.Random(int(replicate_seed) + 13013)
    rng.shuffle(shuffled)
    support_weights = {source: float(weight) for source, weight in zip(sources, shuffled)}
    raw_combined = {
        source: (support_weights[source] ** float(cfg.support_alpha)) * (rels[source].reliability_score ** float(cfg.reliability_alpha))
        for source in sources
    }
    total = sum(raw_combined.values())
    weights = {source: float(raw_combined[source] / total) for source in sources}
    budgets = d12._weighted_budgets(cfg.synthetic_per_class_total, sources, weights, cfg.min_per_source_per_class)
    out = d12._with_weight_diagnostics(sources, weights, budgets, raw_combined)
    out.update(
        {
            "support_weights": support_weights,
            "support_scores": dict(plan.get("support_scores", {})),
            "raw_support_nelbo": dict(plan.get("raw_support_nelbo", {})),
            "calibrated_support_nelbo": dict(plan.get("calibrated_support_nelbo", {})),
            "reliability_scores": dict(plan.get("reliability_scores", {})),
            "use_calibrated_support_nelbo": bool(plan.get("use_calibrated_support_nelbo", True)),
            "include_reliability": bool(plan.get("include_reliability", True)),
            "support_nelbo_tau": float(plan.get("support_nelbo_tau", cfg.support_nelbo_tau)),
        }
    )
    return out


def _evaluate_support_variant(
    cfg: DecentralizedSupportNelboReliabilityConfig,
    *,
    per_source_runtime: Mapping[str, object],
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    eval_raw: object,
    eval_labels: Sequence[int],
    weight_plan: Mapping[str, object],
    prior_method: str,
    pooling_rule: str,
    support_context: Mapping[str, object],
    source_weighting: str,
    selection_source: str,
    claim_role: str,
    real_feature_bacc: float = math.nan,
    control_mode: str = "normal",
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
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
        weight_plan=weight_plan,
        prior_method=prior_method,
        pooling_rule=pooling_rule,
        selection_source=selection_source,
        claim_role=claim_role,
        control_mode=control_mode,
    )
    rows = [_extend_support_row(row, weight_plan=weight_plan, support_context=support_context, source_weighting=source_weighting) for row in rows]
    late = [_extend_support_row(row, weight_plan=weight_plan, support_context=support_context, source_weighting=source_weighting) for row in late]
    return rows, late, coverage, weak, nn


def _append_single_source_references(
    cfg: DecentralizedSupportNelboReliabilityConfig,
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
    temp: list[dict[str, object]] = []
    d12._append_single_source_references(
        cfg,
        temp,
        single_rows,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        candidates=candidates,
        summaries=summaries,
        source_union_ref=d1._missing_reference(),
        center_balanced_ref=d1._missing_reference(),
        real_feature_bacc=real_feature_bacc,
    )
    for row in temp:
        matrix_rows.append(_extend_support_row(row, support_context=support_context))


def _rename_real_feature_reference(row: Mapping[str, object]) -> dict[str, object]:
    out = dict(row)
    if out.get("prior_method") == d1a.ROW_REAL_FEATURE_DENSE_REFERENCE:
        out["prior_method"] = ROW_REAL_FEATURE_DENSE_REFERENCE
    return out


def _extend_support_row(
    row: Mapping[str, object],
    *,
    weight_plan: Mapping[str, object] | None = None,
    support_context: Mapping[str, object] | None = None,
    source_weighting: str | None = None,
) -> dict[str, object]:
    out = dict(row)
    if source_weighting is not None:
        out["source_weighting"] = source_weighting
    support_context = support_context or {}
    out.update(
        {
            "support_size": support_context.get("support_size", ""),
            "support_seed": support_context.get("support_seed", ""),
            "support_eval_split_id": support_context.get("support_eval_split_id", ""),
            "support_labels_used": support_context.get("support_labels_used", 0),
            "n_target_eval_after_support": support_context.get("n_target_eval_after_support", ""),
            "support_eval_min_class_count": support_context.get("support_eval_min_class_count", ""),
            "support_eval_ineligible_reason": support_context.get("support_eval_ineligible_reason", ""),
            "delta_vs_d1_2_reliability_support_eval_reference": math.nan,
            "delta_vs_equal_support_eval_reference": math.nan,
        }
    )
    if weight_plan is None:
        out.update(
            {
                "support_weight_json": "{}",
                "support_score_json": "{}",
                "raw_support_nelbo_json": "{}",
                "calibrated_support_nelbo_json": "{}",
                "combined_weight_json": "{}",
                "combined_budget_per_class_json": "{}",
                "support_nelbo_tau": math.nan,
                "mean_l1_distance_from_reliability_only": math.nan,
            }
        )
        return out
    rel_weights = dict(weight_plan.get("reliability_weights", {}))
    if not rel_weights:
        rel_weights = dict(weight_plan.get("weights", {}))
    out.update(
        {
            "synthetic_per_class_per_source_json": json.dumps(dict(weight_plan["budgets"]), sort_keys=True),
            "support_weight_json": json.dumps(dict(weight_plan.get("support_weights", {})), sort_keys=True),
            "support_score_json": json.dumps(dict(weight_plan.get("support_scores", {})), sort_keys=True),
            "raw_support_nelbo_json": json.dumps(dict(weight_plan.get("raw_support_nelbo", {})), sort_keys=True),
            "calibrated_support_nelbo_json": json.dumps(dict(weight_plan.get("calibrated_support_nelbo", {})), sort_keys=True),
            "combined_weight_json": json.dumps(dict(weight_plan["weights"]), sort_keys=True),
            "combined_budget_per_class_json": json.dumps(dict(weight_plan["budgets"]), sort_keys=True),
            "support_nelbo_tau": weight_plan.get("support_nelbo_tau", math.nan),
            "effective_num_sources": weight_plan["effective_num_sources"],
            "mean_l1_distance_from_reliability_only": _l1_between(dict(weight_plan["weights"]), rel_weights),
        }
    )
    return out


def _l1_between(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = set(left).intersection(right)
    if not keys:
        return math.nan
    return float(sum(abs(float(left[key]) - float(right[key])) for key in keys))


def _support_score_rows(scores: Sequence[SupportScore], context: Mapping[str, object]) -> list[dict[str, object]]:
    rows = []
    for score in scores:
        row = score.to_csv_row()
        row.update(
            {
                "support_eval_split_id": context.get("support_eval_split_id", ""),
                "support_labels_used": context.get("support_labels_used", 0),
            }
        )
        rows.append(row)
    return rows


def _support_weight_manifest_rows(
    ranked_scores: Sequence[SupportScore],
    context: Mapping[str, object],
    plan: Mapping[str, object],
) -> list[dict[str, object]]:
    rows = []
    rank_by_source = {row.expert_id: row.candidate_rank for row in ranked_scores}
    for source in plan["sources"]:
        source_id = str(source)
        rows.append(
            {
                "experiment_seed": ranked_scores[0].experiment_seed,
                "heldout_center": ranked_scores[0].heldout_center,
                "support_seed": ranked_scores[0].support_seed,
                "replicate_seed": ranked_scores[0].support_seed,
                "support_size": ranked_scores[0].support_size,
                "support_eval_split_id": context.get("support_eval_split_id", ""),
                "source_center": source_id,
                "candidate_rank": rank_by_source[source_id],
                "raw_support_nelbo": plan["raw_support_nelbo"][source_id],
                "calibrated_support_nelbo": plan["calibrated_support_nelbo"][source_id],
                "support_nelbo_weight": plan["support_weights"][source_id],
                "support_nelbo_tau": plan["support_nelbo_tau"],
                "use_calibrated_support_nelbo": int(bool(plan["use_calibrated_support_nelbo"])),
            }
        )
    return rows


def _combined_weight_manifest_rows(
    context: Mapping[str, object],
    plan: Mapping[str, object],
    rels: Mapping[str, d12.SourceReliability],
) -> list[dict[str, object]]:
    rows = []
    for source in plan["sources"]:
        source_id = str(source)
        rel = rels[source_id]
        rows.append(
            {
                "experiment_seed": rel.experiment_seed,
                "heldout_center": context.get("heldout_center", ""),
                "support_seed": context.get("support_seed", ""),
                "replicate_seed": rel.replicate_seed,
                "support_size": context.get("support_size", ""),
                "support_eval_split_id": context.get("support_eval_split_id", ""),
                "source_center": source_id,
                "raw_reliability_bacc": rel.raw_bacc,
                "reliability_score": rel.reliability_score,
                "support_nelbo_weight": plan.get("support_weights", {}).get(source_id, math.nan),
                "combined_score": plan["scores"][source_id],
                "combined_weight": plan["weights"][source_id],
                "synthetic_per_class_budget": plan["budgets"][source_id],
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


def _attach_downstream_to_support_scores(
    support_score_rows: list[dict[str, object]],
    context: Mapping[str, object],
    single_rows: Sequence[Mapping[str, object]],
) -> None:
    downstream = {str(row["expert_id"]): _float(row.get("bacc")) for row in single_rows if row.get("pooling_rule") == "single_source"}
    key = (
        str(single_rows[0].get("experiment_seed")) if single_rows else "",
        str(context.get("heldout_center")),
        str(context.get("support_seed")),
        str(context.get("support_size")),
        str(context.get("support_eval_split_id")),
    )
    for row in support_score_rows:
        row_key = (
            str(row.get("experiment_seed")),
            str(row.get("heldout_center")),
            str(row.get("support_seed")),
            str(row.get("support_size")),
            str(row.get("support_eval_split_id")),
        )
        if row_key == key and str(row.get("expert_id")) in downstream:
            row["downstream_bacc"] = downstream[str(row["expert_id"])]


def _alignment_row(
    ranked_scores: Sequence[SupportScore],
    equal_late_rows: Sequence[Mapping[str, object]],
    primary_rows: Sequence[Mapping[str, object]],
    *,
    support_context: Mapping[str, object],
) -> dict[str, object]:
    downstream = {
        str(row["expert_id"]): _float(row.get("bacc"))
        for row in equal_late_rows
        if row.get("pooling_rule") == "single_source" and row.get("status") == "ok"
    }
    method_bacc = {
        "all4_geom": _float(primary_rows[0].get("bacc")) if primary_rows else math.nan,
    }
    alignment = ranking_alignment(ranked_scores=ranked_scores, downstream_bacc_by_expert=downstream, method_baccs=method_bacc)
    return {
        "experiment_seed": ranked_scores[0].experiment_seed if ranked_scores else "",
        "heldout_center": support_context.get("heldout_center", ""),
        "support_seed": support_context.get("support_seed", ""),
        "replicate_seed": support_context.get("support_seed", ""),
        "support_size": support_context.get("support_size", ""),
        "support_eval_split_id": support_context.get("support_eval_split_id", ""),
        "top1_downstream_oracle_hit": alignment["top1_downstream_oracle_hit"],
        "top2_downstream_oracle_containment": alignment["top2_oracle_containment"],
        "top3_downstream_oracle_containment": alignment["top3_oracle_containment"],
        "spearman_support_nelbo_vs_downstream_utility": alignment["spearman_support_nelbo_vs_downstream_bacc"],
        "downstream_oracle_gap": alignment["oracle_gap_all4"],
    }


def _primary_methods_for_support_size(cfg: DecentralizedSupportNelboReliabilityConfig, support_size: int) -> tuple[tuple[str, str], ...]:
    if int(support_size) != cfg.support_size:
        return ((_support_size_method(int(support_size)), "diagnostic_support_size_sensitivity"),)
    return (
        (PRIMARY_SUPPORT_RELIABILITY_METHOD, "primary_target_conditioned_support_nelbo_x_reliability_composition"),
        (ROW_SUPPORT_ONLY, "support_nelbo_only_weighted_diagnostic"),
        (ROW_D1_2_REFERENCE, "d1_2_reliability_support_eval_reference"),
        (ROW_EQUAL_REFERENCE, "d1_1_equal_adaptive_geom_support_eval_reference"),
        (ROW_TOP3, "support_nelbo_x_reliability_top3_diagnostic"),
        (ROW_SUPPORT_ONLY_TOP3, "support_nelbo_only_top3_diagnostic"),
        (ROW_UNCALIBRATED, "uncalibrated_support_nelbo_diagnostic"),
        (ROW_TAU05, "tau_sensitivity_diagnostic"),
        (ROW_TAU20, "tau_sensitivity_diagnostic"),
        (ROW_SINGLE_MEAN, "single_source_mean_reference"),
        (ROW_SINGLE_ORACLE, "diagnostic_only_oracle_reference"),
        (ROW_REAL_FEATURE_DENSE_REFERENCE, "real_feature_transfer_ceiling_reference"),
        (ROW_SHUFFLED_SUPPORT_CONTROL, "negative_control"),
        (ROW_SHUFFLED_SUMMARY_CONTROL, "negative_control"),
        (ROW_SHUFFLED_LABEL_CONTROL, "negative_control"),
    )


def _support_size_method(support_size: int) -> str:
    return {
        8: ROW_SUPPORT_SIZE8,
        16: ROW_SUPPORT_SIZE16,
        64: ROW_SUPPORT_SIZE64,
    }.get(int(support_size), f"decentralized_support_nelbo_support_size{support_size}_x_reliability_geom_diagnostic")


def _ineligible_rows(
    cfg: DecentralizedSupportNelboReliabilityConfig,
    *,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    methods: Sequence[tuple[str, str]],
    support_context: Mapping[str, object],
    status: str,
    error_message: str,
) -> list[dict[str, object]]:
    rows = []
    empty: dict[tuple[str, int], d1a.AdaptiveSourceLocalSummary] = {}
    for method, role in methods:
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
        rows.append(_extend_support_row(row, support_context=support_context))
    return rows


def _populate_delta_and_control_gaps(rows: list[dict[str, object]]) -> None:
    controls: dict[tuple[str, str, str, str], float] = {}
    d12_ref: dict[tuple[str, str, str, str], float] = {}
    equal_ref: dict[tuple[str, str, str, str], float] = {}
    for row in rows:
        key = (
            str(row.get("experiment_seed")),
            str(row.get("heldout_center")),
            str(row.get("replicate_seed")),
            str(row.get("support_size")),
        )
        value = _float(row.get("bacc"))
        if row.get("prior_method") in {ROW_SHUFFLED_SUPPORT_CONTROL, ROW_SHUFFLED_SUMMARY_CONTROL, ROW_SHUFFLED_LABEL_CONTROL} and math.isfinite(value):
            controls[key] = max(controls.get(key, -math.inf), value)
        if row.get("prior_method") == ROW_D1_2_REFERENCE:
            d12_ref[key] = value
        if row.get("prior_method") == ROW_EQUAL_REFERENCE:
            equal_ref[key] = value
    for row in rows:
        key = (
            str(row.get("experiment_seed")),
            str(row.get("heldout_center")),
            str(row.get("replicate_seed")),
            str(row.get("support_size")),
        )
        value = _float(row.get("bacc"))
        if row.get("prior_method") == PRIMARY_SUPPORT_RELIABILITY_METHOD:
            control = controls.get(key, math.nan)
            if math.isfinite(value) and math.isfinite(control):
                row["negative_control_gap"] = value - control
            baseline = d12_ref.get(key, math.nan)
            if math.isfinite(value) and math.isfinite(baseline):
                row["delta_vs_d1_2_reliability_support_eval_reference"] = value - baseline
            equal = equal_ref.get(key, math.nan)
            if math.isfinite(value) and math.isfinite(equal):
                row["delta_vs_equal_support_eval_reference"] = value - equal


def _centerwise_delta_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    primary = _grouped_cell_means(d1a._rows_for(rows, PRIMARY_SUPPORT_RELIABILITY_METHOD))
    d12_ref = _grouped_cell_means(d1a._rows_for(rows, ROW_D1_2_REFERENCE))
    by_center: dict[str, list[float]] = {}
    by_seed: dict[str, list[float]] = {}
    for key, value in primary.items():
        baseline = d12_ref.get(key, math.nan)
        delta = value - baseline if math.isfinite(value) and math.isfinite(baseline) else math.nan
        seed, center = key
        if math.isfinite(delta):
            by_center.setdefault(center, []).append(delta)
            by_seed.setdefault(seed, []).append(delta)
    rows_out = [
        {"axis": "center", "id": key, "delta_vs_d1_2_reliability_support_eval_reference": nanmean(values), "n_cells": len(values)}
        for key, values in sorted(by_center.items())
    ]
    rows_out.extend(
        {"axis": "seed", "id": key, "delta_vs_d1_2_reliability_support_eval_reference": nanmean(values), "n_cells": len(values)}
        for key, values in sorted(by_seed.items())
    )
    return rows_out


def _grouped_cell_means(rows: Sequence[Mapping[str, object]]) -> dict[tuple[str, str], float]:
    groups: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        if row.get("status") == "ok" and int(float(str(row.get("support_size", 0) or 0))) == 32:
            groups.setdefault((str(row["experiment_seed"]), str(row["heldout_center"])), []).append(row)
    return {key: d1._mean_field(values, "bacc") for key, values in groups.items()}


def _decision(
    rows: Sequence[Mapping[str, object]],
    cfg: DecentralizedSupportNelboReliabilityConfig,
    *,
    leakage_status: str,
    support_score_rows: Sequence[Mapping[str, object]],
    support_weight_rows: Sequence[Mapping[str, object]],
    combined_weight_rows: Sequence[Mapping[str, object]],
    split_rows: Sequence[Mapping[str, object]],
    alignment_rows: Sequence[Mapping[str, object]],
    centerwise_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    primary = d1a._rows_for(rows, PRIMARY_SUPPORT_RELIABILITY_METHOD)
    d12_ref = d1a._rows_for(rows, ROW_D1_2_REFERENCE)
    equal = d1a._rows_for(rows, ROW_EQUAL_REFERENCE)
    single_mean = d1a._rows_for(rows, ROW_SINGLE_MEAN)
    single_oracle = d1a._rows_for(rows, ROW_SINGLE_ORACLE)
    real_feature = d1a._rows_for(rows, ROW_REAL_FEATURE_DENSE_REFERENCE)
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
    strongest_control_method, strongest_control_bacc = _strongest_control(control_by_method)
    delta_vs_d12 = _float(stats["center_equal_mean_bacc"]) - _float(d12_stats["center_equal_mean_bacc"])
    delta_vs_equal = _float(stats["center_equal_mean_bacc"]) - _float(equal_stats["center_equal_mean_bacc"])
    delta_vs_mean_single = _float(stats["center_equal_mean_bacc"]) - _float(single_mean_stats["center_equal_mean_bacc"])
    delta_vs_oracle_single = _float(stats["center_equal_mean_bacc"]) - _float(single_oracle_stats["center_equal_mean_bacc"])
    delta_vs_real = _float(stats["center_equal_mean_bacc"]) - _float(real_stats["center_equal_mean_bacc"])
    negative_control_gap = _float(stats["center_equal_mean_bacc"]) - strongest_control_bacc
    shuffled_support_gap = _float(stats["center_equal_mean_bacc"]) - _float(
        control_by_method[ROW_SHUFFLED_SUPPORT_CONTROL]["center_equal_mean_bacc"]
    )
    center_deltas = {str(row["id"]): _float(row["delta_vs_d1_2_reliability_support_eval_reference"]) for row in centerwise_rows if row.get("axis") == "center"}
    seed_deltas = {str(row["id"]): _float(row["delta_vs_d1_2_reliability_support_eval_reference"]) for row in centerwise_rows if row.get("axis") == "seed"}
    centers_beating = sum(1 for value in center_deltas.values() if math.isfinite(value) and value > 0.0)
    seeds_beating = sum(1 for value in seed_deltas.values() if math.isfinite(value) and value > 0.0)
    alignment_primary = [row for row in alignment_rows if int(float(str(row.get("support_size", 0) or 0))) == cfg.support_size]
    spearman_values = [_float(row.get("spearman_support_nelbo_vs_downstream_utility")) for row in alignment_primary]
    spearman = nanmean([value for value in spearman_values if math.isfinite(value)])
    top1 = nanmean([_float(row.get("top1_downstream_oracle_hit")) for row in alignment_primary])
    top2 = nanmean([_float(row.get("top2_downstream_oracle_containment")) for row in alignment_primary])
    oracle_gap = nanmean([_float(row.get("downstream_oracle_gap")) for row in alignment_primary])
    seedwise_spearman = _axis_means(alignment_primary, "experiment_seed", "spearman_support_nelbo_vs_downstream_utility")
    centerwise_spearman = _axis_means(alignment_primary, "heldout_center", "spearman_support_nelbo_vs_downstream_utility")
    negative_spearman_axes = sum(1 for value in list(seedwise_spearman.values()) + list(centerwise_spearman.values()) if math.isfinite(value) and value < 0.0)
    weight_stats = _combined_weight_diagnostics(combined_weight_rows)
    support_size_valid = {
        str(size): sum(
            1
            for row in rows
            if row.get("prior_method") == _support_size_method(int(size)) and row.get("status") == "ok"
        )
        for size in cfg.support_size_diagnostics
    }
    pass_rule = (
        leakage_status == "PASS"
        and int(stats["n_heldout_centers"]) >= len(cfg.heldout_centers)
        and _float(stats["center_equal_mean_bacc"]) >= 0.85
        and _float(stats["min_center_mean_bacc"]) >= 0.75
        and _float(stats["seed_std_bacc"]) <= 0.06
        and delta_vs_d12 >= 0.01
        and centers_beating >= 4
        and seeds_beating >= 2
        and spearman >= 0.20
        and negative_spearman_axes <= 1
        and top2 >= 0.70
        and negative_control_gap >= 0.03
    )
    partial = leakage_status == "PASS" and (delta_vs_d12 >= 0.01 or (_float(stats["min_center_mean_bacc"]) - _float(d12_stats["min_center_mean_bacc"])) > 0.0)
    diagnostic_only = leakage_status == "PASS" and spearman >= 0.20 and not pass_rule
    negative = leakage_status == "PASS" and (not math.isfinite(delta_vs_d12) or delta_vs_d12 <= 0.0)
    verdict = "D1_3_FAIL"
    if leakage_status != "PASS":
        verdict = "PROTOCOL_FAIL"
    elif pass_rule:
        verdict = "D1_3_PASS"
    elif partial:
        verdict = "D1_3_PARTIAL_EVIDENCE"
    elif diagnostic_only:
        verdict = "D1_3_DIAGNOSTIC_ONLY"
    elif negative:
        verdict = "D1_3_NEGATIVE_EVIDENCE"
    flags = []
    if math.isfinite(delta_vs_d12) and delta_vs_d12 < 0.01:
        flags.append("DELTA_VS_D1_2_BELOW_0P01")
    if centers_beating < 4:
        flags.append("CENTER_CONSISTENCY_BELOW_4_OF_5")
    if seeds_beating < 2:
        flags.append("SEED_CONSISTENCY_BELOW_2_OF_3")
    if math.isfinite(spearman) and spearman < 0.20:
        flags.append("SUPPORT_NELBO_SPEARMAN_BELOW_0P20")
    if negative_spearman_axes > 1:
        flags.append("NEGATIVE_SPEARMAN_AXIS_INSTABILITY")
    if math.isfinite(top2) and top2 < 0.70:
        flags.append("TOP2_ORACLE_CONTAINMENT_BELOW_0P70")
    if math.isfinite(negative_control_gap) and negative_control_gap < 0.03:
        flags.append("NEGATIVE_CONTROL_GAP_BELOW_0P03")
    if math.isfinite(shuffled_support_gap) and shuffled_support_gap < 0.03:
        flags.append("SHUFFLED_SUPPORT_CONTROL_COMPETITIVE")
    return {
        "primary_verdict": verdict,
        "diagnostic_flags": "|".join(flags),
        "leakage_status": leakage_status,
        "primary_method": PRIMARY_SUPPORT_RELIABILITY_METHOD,
        "center_equal_mean_bacc": stats["center_equal_mean_bacc"],
        "center_equal_macro_f1": stats["center_equal_macro_f1"],
        "min_center_bacc": stats["min_center_mean_bacc"],
        "seed_std_bacc": stats["seed_std_bacc"],
        "delta_vs_d1_2_reliability_support_eval_reference": delta_vs_d12,
        "delta_vs_equal_support_eval_reference": delta_vs_equal,
        "delta_vs_mean_single_source_adaptive_k": delta_vs_mean_single,
        "delta_vs_single_source_oracle_adaptive_k": delta_vs_oracle_single,
        "delta_vs_real_feature_dense_support_eval_reference": delta_vs_real,
        "negative_control_gap": negative_control_gap,
        "strongest_negative_control_method": strongest_control_method,
        "strongest_negative_control_center_equal_mean_bacc": strongest_control_bacc,
        "shuffled_support_control_center_equal_mean_bacc": control_by_method[ROW_SHUFFLED_SUPPORT_CONTROL]["center_equal_mean_bacc"],
        "shuffled_support_control_gap": shuffled_support_gap,
        "shuffled_summary_control_center_equal_mean_bacc": control_by_method[ROW_SHUFFLED_SUMMARY_CONTROL]["center_equal_mean_bacc"],
        "shuffled_label_control_center_equal_mean_bacc": control_by_method[ROW_SHUFFLED_LABEL_CONTROL]["center_equal_mean_bacc"],
        "top1_downstream_oracle_hit": top1,
        "top2_downstream_oracle_containment": top2,
        "spearman_support_nelbo_vs_downstream_utility": spearman,
        "seedwise_spearman_support_nelbo_vs_downstream_utility_json": json.dumps(seedwise_spearman, sort_keys=True),
        "centerwise_spearman_support_nelbo_vs_downstream_utility_json": json.dumps(centerwise_spearman, sort_keys=True),
        "downstream_oracle_gap": oracle_gap,
        "centerwise_delta_vs_d1_2_json": json.dumps(center_deltas, sort_keys=True),
        "seedwise_delta_vs_d1_2_json": json.dumps(seed_deltas, sort_keys=True),
        "centers_beating_d1_2_reliability": centers_beating,
        "seeds_beating_d1_2_reliability": seeds_beating,
        "d1_2_reliability_support_eval_center_equal_mean_bacc": d12_stats["center_equal_mean_bacc"],
        "equal_support_eval_center_equal_mean_bacc": equal_stats["center_equal_mean_bacc"],
        "mean_single_source_adaptive_k_center_equal_mean_bacc": single_mean_stats["center_equal_mean_bacc"],
        "single_source_oracle_adaptive_k_center_equal_mean_bacc": single_oracle_stats["center_equal_mean_bacc"],
        "real_feature_dense_support_eval_center_equal_mean_bacc": real_stats["center_equal_mean_bacc"],
        "negative_control_center_equal_mean_bacc": strongest_control_bacc,
        "support_size_diagnostic_num_valid_cells_json": json.dumps(support_size_valid, sort_keys=True),
        "support_score_rows": len(support_score_rows),
        "support_weight_rows": len(support_weight_rows),
        "combined_weight_rows": len(combined_weight_rows),
        "support_eval_split_rows": len(split_rows),
        **weight_stats,
        **stats,
    }


def _axis_means(rows: Sequence[Mapping[str, object]], axis: str, field: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        value = _float(row.get(field))
        if math.isfinite(value):
            grouped.setdefault(str(row.get(axis)), []).append(value)
    return {key: nanmean(values) for key, values in sorted(grouped.items())}


def _strongest_control(control_by_method: Mapping[str, Mapping[str, object]]) -> tuple[str, float]:
    scored = [
        (method, _float(stats.get("center_equal_mean_bacc")))
        for method, stats in control_by_method.items()
        if math.isfinite(_float(stats.get("center_equal_mean_bacc")))
    ]
    if not scored:
        return "", math.nan
    return max(scored, key=lambda item: (item[1], item[0]))


def _combined_weight_diagnostics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    cells: dict[tuple[str, str, str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        if int(float(str(row.get("support_size", 0) or 0))) != 32:
            continue
        cells.setdefault((str(row["experiment_seed"]), str(row["heldout_center"]), str(row["replicate_seed"]), str(row["support_size"])), []).append(row)
    entropies = [_float(values[0].get("weight_entropy")) for values in cells.values() if values]
    eff = [_float(values[0].get("effective_num_sources")) for values in cells.values() if values]
    l1_uniform = [_float(values[0].get("l1_distance_from_uniform")) for values in cells.values() if values]
    max_weights = [_float(row.get("combined_weight")) for row in rows if int(float(str(row.get("support_size", 0) or 0))) == 32]
    min_weights = [_float(row.get("combined_weight")) for row in rows if int(float(str(row.get("support_size", 0) or 0))) == 32]
    return {
        "mean_effective_num_sources": nanmean([value for value in eff if math.isfinite(value)]),
        "mean_weight_entropy": nanmean([value for value in entropies if math.isfinite(value)]),
        "mean_l1_distance_from_uniform": nanmean([value for value in l1_uniform if math.isfinite(value)]),
        "max_weight_per_fold": max([value for value in max_weights if math.isfinite(value)], default=math.nan),
        "min_weight_per_fold": min([value for value in min_weights if math.isfinite(value)], default=math.nan),
    }


from decentralized_support_nelbo_reliability_artifacts import (
    _matrix_columns,
    _negative_control_summary,
    _resolved_config,
    _write_artifacts,
    _write_decision_summary,
)
