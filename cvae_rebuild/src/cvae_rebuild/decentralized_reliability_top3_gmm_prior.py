from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

from .features import load_feature_cache, select_rows
from .metrics import nanmean
from .preservation_repair import (
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
from .preservation_sampling import DIAGNOSTIC_SELECTION, PRIMARY_SELECTION, _manifest_row, _per_source_variant, _runtime_source
from .protocol import ProtocolError, assert_candidate_pool, build_leakage_report
from .reporting import prepare_artifact_dirs, write_csv_rows, write_json
from .splits import candidate_experts

from . import decentralized_adaptive_gmm_prior as d1a
from . import decentralized_k16_gmm_prior as d1
from . import decentralized_reliability_weighted_gmm_prior as d12
from . import decentralized_support8_top3_tau05_gmm_prior as d131


RELIABILITY_TOP3_NAME = "virchow2_cvae_decentralized_reliability_top3_gmm_prior_v1"
PRIMARY_RELIABILITY_TOP3_METHOD = "decentralized_reliability_top3_geom_confirmation"
ROW_RELIABILITY_ALL4 = "decentralized_reliability_all4_weighted_geom_reference"
ROW_EQUAL_ALL4 = "decentralized_equal_all4_geom_reference"
ROW_RELIABILITY_TOP3_WEIGHTED = "decentralized_reliability_top3_weighted_geom_diagnostic"
ROW_RELIABILITY_TOP2 = "decentralized_reliability_top2_geom_diagnostic"
ROW_RELIABILITY_TOP4 = "decentralized_reliability_top4_geom_diagnostic"
ROW_SINGLE_MEAN = "per_source_adaptive_k_single_expert_mean_reference"
ROW_SINGLE_ORACLE = "per_source_adaptive_k_single_expert_oracle_reference"
ROW_REAL_FEATURE_DENSE_REFERENCE = "real_source_embedding_classifier_dense_reference"
ROW_SHUFFLED_RELIABILITY_CONTROL = "decentralized_reliability_top3_shuffled_reliability_control"
ROW_RANDOM_SOURCE_DROP_CONTROL = "decentralized_reliability_top3_random_source_drop_control"
ROW_SHUFFLED_SUMMARY_CONTROL = "decentralized_reliability_top3_shuffled_summary_control"
ROW_SHUFFLED_LABEL_CONTROL = "decentralized_reliability_top3_shuffled_label_control"
ROW_SUPPORT8_RELIABILITY_TOP3_CONTEXT = "decentralized_support8_reliability_top3_geom_matched_d1_3_1_reference"
ROW_SUPPORT8_D1_3_1_PRIMARY_CONTEXT = "decentralized_support8_d1_3_1_primary_context"
ROW_SUPPORT8_D1_3_1_SHUFFLED_SUPPORT_CONTEXT = "decentralized_support8_d1_3_1_shuffled_support_context"
PROTOCOL_WORDING = (
    "This is a data-minimizing, raw-data-free source-local latent summary-exchange protocol. "
    "It uses source-local generation-preservation reliability for sparse top-3 composition. "
    "It is not a target-conditioned compatibility-routing or formal differential privacy claim."
)


@dataclass(frozen=True)
class DecentralizedReliabilityTop3Config:
    name: str
    artifact_root: Path
    repair_artifact_root: Path
    d1_3_1_artifact_root: Path | None
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

    @property
    def softmax_tau(self) -> float:
        return 1.0


def load_decentralized_reliability_top3_gmm_prior_config(path: str | Path) -> DecentralizedReliabilityTop3Config:
    source = Path(path).resolve()
    data = _load_mapping(source)
    base_dir = source.parents[2] if len(source.parents) >= 3 else source.parent
    return parse_decentralized_reliability_top3_gmm_prior_config(data, base_dir=base_dir)


def parse_decentralized_reliability_top3_gmm_prior_config(
    data: Mapping[str, Any],
    *,
    base_dir: str | Path = ".",
) -> DecentralizedReliabilityTop3Config:
    base = Path(base_dir)
    experiment = _mapping(data, "experiment")
    inputs = _mapping(data, "inputs")
    run = _mapping(data, "run_matrix")
    if "support_size" in run or "support_seeds" in run:
        raise ProtocolError("D1.4 primary must not configure or consume target support rows.")
    generation = _mapping(data, "generation")
    gmm = _mapping(data, "reliability_top3_gmm_prior")
    classifier = _mapping(data, "classifier")
    cfg = DecentralizedReliabilityTop3Config(
        name=str(experiment["name"]),
        artifact_root=_path(base, str(experiment["artifact_root"])),
        repair_artifact_root=_path(base, str(inputs["repair_artifact_root"])),
        d1_3_1_artifact_root=_optional_path(base, inputs.get("d1_3_1_artifact_root")),
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
    validate_decentralized_reliability_top3_gmm_prior_config(cfg)
    return cfg


def validate_decentralized_reliability_top3_gmm_prior_config(cfg: DecentralizedReliabilityTop3Config) -> None:
    if cfg.name != RELIABILITY_TOP3_NAME:
        raise ProtocolError(f"D1.4 experiment name must be {RELIABILITY_TOP3_NAME!r}.")
    if cfg.backbone != "virchow2":
        raise ProtocolError("D1.4 is locked to backbone=virchow2.")
    if cfg.primary_variant != PRIMARY_VARIANT:
        raise ProtocolError(f"primary_variant must be {PRIMARY_VARIANT!r}.")
    if cfg.primary_method != PRIMARY_RELIABILITY_TOP3_METHOD:
        raise ProtocolError(f"primary_method must be {PRIMARY_RELIABILITY_TOP3_METHOD!r}.")
    if cfg.candidate_components_per_source_class != (4, 3, 2, 1):
        raise ProtocolError("candidate_components_per_source_class must be locked to [4, 3, 2, 1].")
    if len(cfg.heldout_centers) != 5:
        raise ProtocolError("D1.4 composition expects exactly five centers, giving four source experts per fold.")
    if cfg.top_k_sources != 3:
        raise ProtocolError("D1.4 top_k_sources must be locked to 3.")
    if cfg.source_weighting != "source_local_reliability_top3":
        raise ProtocolError("source_weighting must be source_local_reliability_top3.")
    if cfg.gmm_covariance_type != "diag":
        raise ProtocolError("gmm_covariance_type must be diag.")
    if cfg.primary_pooling != "geometric":
        raise ProtocolError("D1.4 primary_pooling must be geometric.")
    if cfg.synthetic_per_class_total != 128 or cfg.min_per_source_per_class != 8:
        raise ProtocolError("D1.4 synthetic budget must be 128 total with min_per_source_per_class=8.")
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


def run_decentralized_reliability_top3_gmm_prior(
    cfg: DecentralizedReliabilityTop3Config,
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

                for replicate_seed in cfg.replicate_seeds:
                    rels = {source: reliability[(int(experiment_seed), int(replicate_seed), str(source))] for source in candidates}
                    all4_weighted = d12._weight_plan(cfg, candidates, rels, mode="linear")
                    equal_all4 = d12._uniform_weight_plan(cfg, candidates, rels)
                    selected = _select_topk_reliable(candidates, rels, k=cfg.top_k_sources)
                    primary_plan = _equal_selected_plan(cfg, selected, rels)
                    weighted_top3 = _weighted_selected_plan(cfg, selected, rels)
                    top2_plan = _equal_selected_plan(cfg, _select_topk_reliable(candidates, rels, k=2), rels)
                    top4_plan = _equal_selected_plan(cfg, _select_topk_reliable(candidates, rels, k=4), rels)
                    shuffled_reliability_plan = _shuffled_reliability_plan(cfg, candidates, rels, experiment_seed, heldout_center, replicate_seed)
                    random_drop_plan = _random_source_drop_plan(cfg, candidates, rels, experiment_seed, heldout_center, replicate_seed)

                    if eval_error:
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
                    ref_row = _extend_top3_row(ref_row)
                    real_late = [_extend_top3_row(row) for row in real_late]
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

                    primary_rows, primary_late, coverage, weak, nn = _evaluate_variant(
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
                        prior_method=PRIMARY_RELIABILITY_TOP3_METHOD,
                        pooling_rule="geometric",
                        source_weighting="source_local_reliability_top3_equal",
                        selection_source=PRIMARY_SELECTION,
                        claim_role="primary_reliability_top3_sparse_composition",
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
                    )
                    selection_rows.append(
                        _selection_row(
                            primary_plan,
                            equal_late,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            replicate_seed=int(replicate_seed),
                            rels=rels,
                            selection_rule="source_local_reliability_top3",
                        )
                    )

                    diagnostics = (
                        (
                            ROW_RELIABILITY_TOP3_WEIGHTED,
                            weighted_top3,
                            "weighted_geometric",
                            "source_local_reliability_top3_weighted",
                            "reliability_top3_weighted_diagnostic",
                        ),
                        (
                            ROW_RELIABILITY_TOP2,
                            top2_plan,
                            "geometric",
                            "source_local_reliability_top2_equal",
                            "reliability_top2_diagnostic",
                        ),
                        (
                            ROW_RELIABILITY_TOP4,
                            top4_plan,
                            "geometric",
                            "source_local_reliability_top4_equal",
                            "reliability_top4_diagnostic",
                        ),
                        (
                            ROW_SHUFFLED_RELIABILITY_CONTROL,
                            shuffled_reliability_plan,
                            "geometric",
                            "shuffled_source_local_reliability_top3",
                            "negative_control",
                        ),
                        (
                            ROW_RANDOM_SOURCE_DROP_CONTROL,
                            random_drop_plan,
                            "geometric",
                            "random_drop_one_source_top3",
                            "negative_control",
                        ),
                    )
                    for method, plan, pooling_rule, weighting, role in diagnostics:
                        rows, late, coverage, weak, nn = _evaluate_variant(
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
                        rows, late, coverage, weak, nn = _evaluate_variant(
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
                            pooling_rule="geometric",
                            source_weighting="source_local_reliability_top3_equal",
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

    context_rows = _load_d1_3_1_context(cfg.d1_3_1_artifact_root)
    matrix_rows.extend(context_rows)
    leakage = build_leakage_report(
        target_support_labels_for_selection=False,
        target_eval_labels_for_scoring_only=True,
        target_expert_excluded=target_expert_excluded,
        oracle_rows_diagnostic_only=True,
        extra_violations=protocol_violations,
    )
    _populate_deltas(matrix_rows)
    gap_rows = [dict(row) for row in matrix_rows if row.get("status") == "ok"]
    centerwise_rows = _centerwise_delta_rows(matrix_rows)
    stability_rows = _selection_stability_rows(selection_rows)
    decision = _decision(
        matrix_rows,
        cfg,
        leakage_status=leakage.status,
        reliability_rows=reliability_rows,
        selection_rows=stability_rows,
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


def _select_topk_reliable(
    sources: Sequence[str],
    rels: Mapping[str, d12.SourceReliability],
    *,
    k: int,
) -> tuple[str, ...]:
    ordered = sorted(
        tuple(str(source) for source in sources),
        key=lambda source: (-float(rels[source].reliability_score), -float(rels[source].raw_bacc), source),
    )
    return tuple(ordered[: int(k)])


def _equal_selected_plan(
    cfg: DecentralizedReliabilityTop3Config,
    sources: Sequence[str],
    rels: Mapping[str, d12.SourceReliability],
) -> dict[str, object]:
    sources_tuple = tuple(str(source) for source in sources)
    weight = 1.0 / float(len(sources_tuple))
    weights = {source: weight for source in sources_tuple}
    budgets = {source: int(value) for source, value in zip(sources_tuple, d1._balanced_counts(cfg.synthetic_per_class_total, len(sources_tuple)))}
    scores = {source: float(rels[source].reliability_score) for source in sources_tuple}
    return d12._with_weight_diagnostics(sources_tuple, weights, budgets, scores)


def _weighted_selected_plan(
    cfg: DecentralizedReliabilityTop3Config,
    sources: Sequence[str],
    rels: Mapping[str, d12.SourceReliability],
) -> dict[str, object]:
    scores = {source: float(rels[source].reliability_score) for source in sources}
    total = sum(scores.values())
    if total <= 0.0:
        raise ProtocolError("Selected reliability scores are not positive.")
    weights = {source: float(scores[source] / total) for source in sources}
    budgets = d12._weighted_budgets(cfg.synthetic_per_class_total, tuple(sources), weights, cfg.min_per_source_per_class)
    return d12._with_weight_diagnostics(tuple(sources), weights, budgets, scores)


def _shuffled_reliability_plan(
    cfg: DecentralizedReliabilityTop3Config,
    sources: Sequence[str],
    rels: Mapping[str, d12.SourceReliability],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
) -> dict[str, object]:
    sources_tuple = tuple(str(source) for source in sources)
    values = [(rels[source].reliability_score, rels[source].raw_bacc) for source in sources_tuple]
    rng = random.Random(d1._latent_seed(experiment_seed, heldout_center, replicate_seed, "shuffled_reliability_control"))
    rng.shuffle(values)
    pseudo = {
        source: d12.SourceReliability(
            rels[source].experiment_seed,
            rels[source].replicate_seed,
            source,
            float(raw_bacc),
            rels[source].macro_f1,
            float(score),
            rels[source].reliability_status,
            rels[source].error_message,
            rels[source].n_eval,
            rels[source].generated_features_hash,
            rels[source].prediction_hash,
        )
        for source, (score, raw_bacc) in zip(sources_tuple, values)
    }
    return _equal_selected_plan(cfg, _select_topk_reliable(sources_tuple, pseudo, k=cfg.top_k_sources), pseudo)


def _random_source_drop_plan(
    cfg: DecentralizedReliabilityTop3Config,
    sources: Sequence[str],
    rels: Mapping[str, d12.SourceReliability],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
) -> dict[str, object]:
    sources_tuple = tuple(str(source) for source in sources)
    rng = random.Random(d1._latent_seed(experiment_seed, heldout_center, replicate_seed, "random_source_drop_control"))
    selected = tuple(sorted(rng.sample(list(sources_tuple), cfg.top_k_sources)))
    return _equal_selected_plan(cfg, selected, rels)


def _evaluate_variant(
    cfg: DecentralizedReliabilityTop3Config,
    *,
    source_weighting: str,
    **kwargs: Any,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    kwargs.setdefault("source_union_ref", d1._missing_reference())
    kwargs.setdefault("center_balanced_ref", d1._missing_reference())
    rows, late, coverage, weak, nn = d12._evaluate_weighted_variant(cfg, **kwargs)
    return (
        [_extend_top3_row(row, source_weighting=source_weighting) for row in rows],
        [_extend_top3_row(row, source_weighting=source_weighting) for row in late],
        coverage,
        weak,
        nn,
    )


def _extend_top3_row(row: Mapping[str, object], *, source_weighting: str | None = None) -> dict[str, object]:
    out = dict(row)
    if out.get("prior_method") == d1a.ROW_REAL_FEATURE_DENSE_REFERENCE:
        out["prior_method"] = ROW_REAL_FEATURE_DENSE_REFERENCE
    if source_weighting is not None:
        out["source_weighting"] = source_weighting
    out.update(
        {
            "delta_vs_reliability_all4_weighted": out.get("delta_vs_reliability_all4_weighted", math.nan),
            "delta_vs_equal_all4": out.get("delta_vs_equal_all4", math.nan),
            "delta_vs_d1_3_1_primary_context": out.get("delta_vs_d1_3_1_primary_context", math.nan),
            "strongest_negative_control_gap": out.get("strongest_negative_control_gap", math.nan),
            "shuffled_reliability_control_gap": out.get("shuffled_reliability_control_gap", math.nan),
            "random_source_drop_control_gap": out.get("random_source_drop_control_gap", math.nan),
        }
    )
    return out


def _append_single_source_references(
    cfg: DecentralizedReliabilityTop3Config,
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
        matrix_rows.append(_extend_top3_row(row))


def _oracle_source(single_rows: Sequence[Mapping[str, object]]) -> str:
    rows = [
        row for row in single_rows
        if row.get("pooling_rule") == "single_source" and row.get("status") == "ok" and math.isfinite(_float(row.get("bacc")))
    ]
    if not rows:
        return ""
    return str(max(rows, key=lambda row: (_float(row.get("bacc")), str(row.get("expert_id")))).get("expert_id"))


def _selection_row(
    plan: Mapping[str, object],
    single_rows: Sequence[Mapping[str, object]],
    *,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    rels: Mapping[str, d12.SourceReliability],
    selection_rule: str,
) -> dict[str, object]:
    selected = tuple(str(source) for source in plan["sources"])
    oracle = _oracle_source(single_rows)
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "replicate_seed": int(replicate_seed),
        "selection_rule": selection_rule,
        "selected_sources": "|".join(selected),
        "selected_source_count": len(selected),
        "oracle_source": oracle,
        "oracle_in_selected_top3": int(bool(oracle and oracle in selected)),
        "top3_downstream_oracle_containment": int(bool(oracle and oracle in selected)),
        "synthetic_per_class_budget_json": json.dumps(dict(plan["budgets"]), sort_keys=True),
        "selection_weight_json": json.dumps(dict(plan["weights"]), sort_keys=True),
        "selected_source_histogram_json": json.dumps({source: 1 for source in selected}, sort_keys=True),
        **{
            f"rank{idx}_source_center": source
            for idx, source in enumerate(selected, start=1)
        },
        **{
            f"rank{idx}_raw_reliability_bacc": rels[source].raw_bacc
            for idx, source in enumerate(selected, start=1)
        },
        **{
            f"rank{idx}_reliability_score": rels[source].reliability_score
            for idx, source in enumerate(selected, start=1)
        },
    }


def _selection_stability_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row.get("experiment_seed")), str(row.get("heldout_center"))), []).append(row)
    jaccard_by_group: dict[tuple[str, str], float] = {}
    for key, subset in grouped.items():
        sets = [set(str(row.get("selected_sources", "")).split("|")) - {""} for row in subset]
        vals = []
        for left, right in combinations(sets, 2):
            denom = len(left | right)
            vals.append(len(left & right) / float(denom) if denom else math.nan)
        jaccard_by_group[key] = nanmean([value for value in vals if math.isfinite(value)]) if vals else 1.0
    out = []
    for row in rows:
        key = (str(row.get("experiment_seed")), str(row.get("heldout_center")))
        copied = dict(row)
        copied["top3_selection_jaccard_across_replicates"] = jaccard_by_group.get(key, math.nan)
        out.append(copied)
    return out


def _ineligible_rows(
    cfg: DecentralizedReliabilityTop3Config,
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
        rows.append(_extend_top3_row(row))
    return rows


def _methods() -> tuple[tuple[str, str], ...]:
    return (
        (PRIMARY_RELIABILITY_TOP3_METHOD, "primary_reliability_top3_sparse_composition"),
        (ROW_RELIABILITY_ALL4, "reliability_all4_weighted_reference"),
        (ROW_EQUAL_ALL4, "equal_all4_reference"),
        (ROW_RELIABILITY_TOP3_WEIGHTED, "reliability_top3_weighted_diagnostic"),
        (ROW_RELIABILITY_TOP2, "reliability_top2_diagnostic"),
        (ROW_RELIABILITY_TOP4, "reliability_top4_diagnostic"),
        (ROW_SINGLE_MEAN, "single_source_mean_reference"),
        (ROW_SINGLE_ORACLE, "diagnostic_only_oracle_reference"),
        (ROW_REAL_FEATURE_DENSE_REFERENCE, "real_feature_transfer_ceiling_reference"),
        (ROW_SHUFFLED_RELIABILITY_CONTROL, "negative_control"),
        (ROW_RANDOM_SOURCE_DROP_CONTROL, "negative_control"),
        (ROW_SHUFFLED_SUMMARY_CONTROL, "negative_control"),
        (ROW_SHUFFLED_LABEL_CONTROL, "negative_control"),
    )


def _load_d1_3_1_context(root: Path | None) -> list[dict[str, object]]:
    if root is None:
        return _missing_context_rows()
    matrix_path = root / "tables" / "decentralized_support8_top3_tau05_downstream_matrix.csv"
    if not matrix_path.exists():
        return _missing_context_rows()
    mapping = {
        d131.ROW_RELIABILITY_TOP3: ROW_SUPPORT8_RELIABILITY_TOP3_CONTEXT,
        d131.PRIMARY_SUPPORT8_TOP3_TAU05_METHOD: ROW_SUPPORT8_D1_3_1_PRIMARY_CONTEXT,
        d131.ROW_SHUFFLED_SUPPORT_CONTROL: ROW_SUPPORT8_D1_3_1_SHUFFLED_SUPPORT_CONTEXT,
    }
    rows = []
    with matrix_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            method = mapping.get(str(row.get("prior_method")))
            if method is None:
                continue
            copied = _extend_top3_row(row)
            copied["prior_method"] = method
            copied["claim_role"] = "historical_d1_3_1_support8_context_only"
            copied["selection_source"] = DIAGNOSTIC_SELECTION
            rows.append(copied)
    return rows or _missing_context_rows()


def _missing_context_rows() -> list[dict[str, object]]:
    return [
        {"prior_method": ROW_SUPPORT8_RELIABILITY_TOP3_CONTEXT, "status": "missing_context_reference", "claim_role": "historical_d1_3_1_support8_context_only"},
        {"prior_method": ROW_SUPPORT8_D1_3_1_PRIMARY_CONTEXT, "status": "missing_context_reference", "claim_role": "historical_d1_3_1_support8_context_only"},
        {"prior_method": ROW_SUPPORT8_D1_3_1_SHUFFLED_SUPPORT_CONTEXT, "status": "missing_context_reference", "claim_role": "historical_d1_3_1_support8_context_only"},
    ]


def _populate_deltas(rows: list[dict[str, object]]) -> None:
    refs: dict[str, dict[tuple[str, str, str], float]] = {
        ROW_RELIABILITY_ALL4: {},
        ROW_EQUAL_ALL4: {},
        ROW_SHUFFLED_RELIABILITY_CONTROL: {},
        ROW_RANDOM_SOURCE_DROP_CONTROL: {},
    }
    controls: dict[tuple[str, str, str], float] = {}
    control_methods = {
        ROW_SHUFFLED_RELIABILITY_CONTROL,
        ROW_RANDOM_SOURCE_DROP_CONTROL,
        ROW_SHUFFLED_SUMMARY_CONTROL,
        ROW_SHUFFLED_LABEL_CONTROL,
    }
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
    for row in rows:
        if row.get("prior_method") != PRIMARY_RELIABILITY_TOP3_METHOD:
            continue
        key = (str(row.get("experiment_seed")), str(row.get("heldout_center")), str(row.get("replicate_seed")))
        value = _float(row.get("bacc"))
        for field, method in (
            ("delta_vs_reliability_all4_weighted", ROW_RELIABILITY_ALL4),
            ("delta_vs_equal_all4", ROW_EQUAL_ALL4),
            ("shuffled_reliability_control_gap", ROW_SHUFFLED_RELIABILITY_CONTROL),
            ("random_source_drop_control_gap", ROW_RANDOM_SOURCE_DROP_CONTROL),
        ):
            baseline = refs[method].get(key, math.nan)
            if math.isfinite(value) and math.isfinite(baseline):
                row[field] = value - baseline
        control = controls.get(key, math.nan)
        if math.isfinite(value) and math.isfinite(control):
            row["strongest_negative_control_gap"] = value - control
            row["negative_control_gap"] = value - control


def _grouped_cell_means(rows: Sequence[Mapping[str, object]]) -> dict[tuple[str, str], float]:
    groups: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        if row.get("status") == "ok":
            groups.setdefault((str(row["experiment_seed"]), str(row["heldout_center"])), []).append(row)
    return {key: d1._mean_field(values, "bacc") for key, values in groups.items()}


def _centerwise_delta_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    primary = _grouped_cell_means(d1a._rows_for(rows, PRIMARY_RELIABILITY_TOP3_METHOD))
    rel_all4 = _grouped_cell_means(d1a._rows_for(rows, ROW_RELIABILITY_ALL4))
    equal = _grouped_cell_means(d1a._rows_for(rows, ROW_EQUAL_ALL4))
    by_center: dict[str, list[float]] = {}
    by_seed: dict[str, list[float]] = {}
    by_center_equal: dict[str, list[float]] = {}
    for key, value in primary.items():
        baseline = rel_all4.get(key, math.nan)
        equal_value = equal.get(key, math.nan)
        seed, center = key
        if math.isfinite(value) and math.isfinite(baseline):
            delta = value - baseline
            by_center.setdefault(center, []).append(delta)
            by_seed.setdefault(seed, []).append(delta)
        if math.isfinite(value) and math.isfinite(equal_value):
            by_center_equal.setdefault(center, []).append(value - equal_value)
    rows_out = [
        {
            "axis": "center",
            "id": key,
            "delta_vs_reliability_all4_weighted": nanmean(values),
            "delta_vs_equal_all4": nanmean(by_center_equal.get(key, [])),
            "n_cells": len(values),
        }
        for key, values in sorted(by_center.items())
    ]
    rows_out.extend(
        {
            "axis": "seed",
            "id": key,
            "delta_vs_reliability_all4_weighted": nanmean(values),
            "delta_vs_equal_all4": math.nan,
            "n_cells": len(values),
        }
        for key, values in sorted(by_seed.items())
    )
    return rows_out


def _decision(
    rows: Sequence[Mapping[str, object]],
    cfg: DecentralizedReliabilityTop3Config,
    *,
    leakage_status: str,
    reliability_rows: Sequence[Mapping[str, object]],
    selection_rows: Sequence[Mapping[str, object]],
    centerwise_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    primary = d1a._rows_for(rows, PRIMARY_RELIABILITY_TOP3_METHOD)
    rel_all4 = d1a._rows_for(rows, ROW_RELIABILITY_ALL4)
    equal = d1a._rows_for(rows, ROW_EQUAL_ALL4)
    single_mean = d1a._rows_for(rows, ROW_SINGLE_MEAN)
    single_oracle = d1a._rows_for(rows, ROW_SINGLE_ORACLE)
    real_feature = d1a._rows_for(rows, ROW_REAL_FEATURE_DENSE_REFERENCE)
    d131_primary = d1a._rows_for(rows, ROW_SUPPORT8_D1_3_1_PRIMARY_CONTEXT)
    controls = {
        ROW_SHUFFLED_RELIABILITY_CONTROL: d1a._primary_stats(d1a._rows_for(rows, ROW_SHUFFLED_RELIABILITY_CONTROL)),
        ROW_RANDOM_SOURCE_DROP_CONTROL: d1a._primary_stats(d1a._rows_for(rows, ROW_RANDOM_SOURCE_DROP_CONTROL)),
        ROW_SHUFFLED_SUMMARY_CONTROL: d1a._primary_stats(d1a._rows_for(rows, ROW_SHUFFLED_SUMMARY_CONTROL)),
        ROW_SHUFFLED_LABEL_CONTROL: d1a._primary_stats(d1a._rows_for(rows, ROW_SHUFFLED_LABEL_CONTROL)),
    }
    stats = d1a._primary_stats(primary)
    rel_all4_stats = d1a._primary_stats(rel_all4)
    equal_stats = d1a._primary_stats(equal)
    single_mean_stats = d1a._primary_stats(single_mean)
    single_oracle_stats = d1a._primary_stats(single_oracle)
    real_stats = d1a._primary_stats(real_feature)
    d131_stats = d1a._primary_stats(d131_primary)
    strongest_control_method, strongest_control_bacc = _strongest_control(controls)
    delta_vs_rel_all4 = _float(stats["center_equal_mean_bacc"]) - _float(rel_all4_stats["center_equal_mean_bacc"])
    delta_vs_equal = _float(stats["center_equal_mean_bacc"]) - _float(equal_stats["center_equal_mean_bacc"])
    delta_vs_d131 = _float(stats["center_equal_mean_bacc"]) - _float(d131_stats["center_equal_mean_bacc"])
    delta_vs_mean_single = _float(stats["center_equal_mean_bacc"]) - _float(single_mean_stats["center_equal_mean_bacc"])
    delta_vs_oracle_single = _float(stats["center_equal_mean_bacc"]) - _float(single_oracle_stats["center_equal_mean_bacc"])
    delta_vs_real = _float(stats["center_equal_mean_bacc"]) - _float(real_stats["center_equal_mean_bacc"])
    negative_control_gap = _float(stats["center_equal_mean_bacc"]) - strongest_control_bacc
    shuffled_reliability_gap = _float(stats["center_equal_mean_bacc"]) - _float(controls[ROW_SHUFFLED_RELIABILITY_CONTROL]["center_equal_mean_bacc"])
    random_drop_gap = _float(stats["center_equal_mean_bacc"]) - _float(controls[ROW_RANDOM_SOURCE_DROP_CONTROL]["center_equal_mean_bacc"])
    shuffled_reliability_min = _float(controls[ROW_SHUFFLED_RELIABILITY_CONTROL]["min_center_mean_bacc"])
    center_deltas = {str(row["id"]): _float(row["delta_vs_reliability_all4_weighted"]) for row in centerwise_rows if row.get("axis") == "center"}
    seed_deltas = {str(row["id"]): _float(row["delta_vs_reliability_all4_weighted"]) for row in centerwise_rows if row.get("axis") == "seed"}
    centers_beating = sum(1 for value in center_deltas.values() if math.isfinite(value) and value > 0.0)
    seeds_beating = sum(1 for value in seed_deltas.values() if math.isfinite(value) and value > 0.0)
    top3_containment = nanmean([_float(row.get("top3_downstream_oracle_containment")) for row in selection_rows])
    selection_jaccard = nanmean([_float(row.get("top3_selection_jaccard_across_replicates")) for row in selection_rows])
    selected_hist = _selected_source_histogram(selection_rows)
    eligible_rows = len(primary)
    weight_stats = _selection_weight_diagnostics(selection_rows, reliability_rows)
    pass_rule = (
        leakage_status == "PASS"
        and int(stats["n_heldout_centers"]) >= len(cfg.heldout_centers)
        and eligible_rows >= 45
        and _float(stats["center_equal_mean_bacc"]) >= 0.85
        and _float(stats["min_center_mean_bacc"]) >= 0.78
        and _float(stats["seed_std_bacc"]) <= 0.05
        and delta_vs_rel_all4 >= 0.02
        and delta_vs_equal >= 0.02
        and centers_beating >= 4
        and seeds_beating >= 2
        and delta_vs_mean_single > 0.0
        and negative_control_gap >= 0.03
        and shuffled_reliability_gap >= 0.02
        and _float(stats["min_center_mean_bacc"]) >= shuffled_reliability_min
        and real_feature
    )
    thesis_partial = (
        leakage_status == "PASS"
        and _float(stats["center_equal_mean_bacc"]) >= 0.83
        and _float(stats["min_center_mean_bacc"]) >= 0.75
        and delta_vs_rel_all4 >= 0.01
        and delta_vs_equal >= 0.01
        and negative_control_gap >= 0.01
        and shuffled_reliability_gap >= 0.01
        and random_drop_gap >= 0.01
    )
    diagnostic_only = (
        leakage_status == "PASS"
        and (delta_vs_rel_all4 > 0.0 or delta_vs_equal > 0.0)
        and (shuffled_reliability_gap < 0.02 or random_drop_gap < 0.02)
    )
    verdict = "D1_4_FAIL"
    if leakage_status != "PASS":
        verdict = "PROTOCOL_FAIL"
    elif pass_rule:
        verdict = "D1_4_PASS"
    elif thesis_partial:
        verdict = "D1_4_THESIS_PARTIAL"
    elif diagnostic_only:
        verdict = "D1_4_DIAGNOSTIC_ONLY"
    flags = []
    if eligible_rows < 45:
        flags.append("ELIGIBLE_SEED_CENTER_CELLS_BELOW_45")
    if math.isfinite(delta_vs_rel_all4) and delta_vs_rel_all4 < 0.02:
        flags.append("DELTA_VS_RELIABILITY_ALL4_BELOW_0P02")
    if math.isfinite(delta_vs_equal) and delta_vs_equal < 0.02:
        flags.append("DELTA_VS_EQUAL_ALL4_BELOW_0P02")
    if centers_beating < 4:
        flags.append("CENTER_CONSISTENCY_BELOW_4_OF_5")
    if seeds_beating < 2:
        flags.append("SEED_CONSISTENCY_BELOW_2_OF_3")
    if math.isfinite(negative_control_gap) and negative_control_gap < 0.03:
        flags.append("NEGATIVE_CONTROL_GAP_BELOW_0P03")
    if math.isfinite(shuffled_reliability_gap) and shuffled_reliability_gap < 0.02:
        flags.append("SHUFFLED_RELIABILITY_CONTROL_COMPETITIVE")
    if math.isfinite(random_drop_gap) and random_drop_gap < 0.02:
        flags.append("RANDOM_SOURCE_DROP_CONTROL_COMPETITIVE")
    return {
        "primary_verdict": verdict,
        "diagnostic_flags": "|".join(flags),
        "leakage_status": leakage_status,
        "primary_method": PRIMARY_RELIABILITY_TOP3_METHOD,
        "center_equal_mean_bacc": stats["center_equal_mean_bacc"],
        "center_equal_macro_f1": stats["center_equal_macro_f1"],
        "min_center_bacc": stats["min_center_mean_bacc"],
        "seed_std_bacc": stats["seed_std_bacc"],
        "delta_vs_reliability_all4_weighted": delta_vs_rel_all4,
        "delta_vs_equal_all4": delta_vs_equal,
        "delta_vs_d1_3_1_primary_context": delta_vs_d131,
        "delta_vs_mean_single_source_adaptive_k": delta_vs_mean_single,
        "delta_vs_single_source_oracle_adaptive_k": delta_vs_oracle_single,
        "delta_vs_real_feature_dense_reference": delta_vs_real,
        "top3_downstream_oracle_containment": top3_containment,
        "oracle_in_selected_top3_rate": top3_containment,
        "mean_top3_selection_jaccard": selection_jaccard,
        "negative_control_gap": negative_control_gap,
        "strongest_negative_control_method": strongest_control_method,
        "strongest_negative_control_center_equal_mean_bacc": strongest_control_bacc,
        "shuffled_reliability_control_gap": shuffled_reliability_gap,
        "random_source_drop_control_gap": random_drop_gap,
        "eligible_seed_center_cells": eligible_rows,
        "selected_source_histogram_json": json.dumps(selected_hist, sort_keys=True),
        "centerwise_delta_vs_reliability_all4_json": json.dumps(center_deltas, sort_keys=True),
        "seedwise_delta_vs_reliability_all4_json": json.dumps(seed_deltas, sort_keys=True),
        "centers_beating_reliability_all4": centers_beating,
        "seeds_beating_reliability_all4": seeds_beating,
        "reliability_all4_center_equal_mean_bacc": rel_all4_stats["center_equal_mean_bacc"],
        "equal_all4_center_equal_mean_bacc": equal_stats["center_equal_mean_bacc"],
        "mean_single_source_adaptive_k_center_equal_mean_bacc": single_mean_stats["center_equal_mean_bacc"],
        "single_source_oracle_adaptive_k_center_equal_mean_bacc": single_oracle_stats["center_equal_mean_bacc"],
        "real_feature_dense_reference_center_equal_mean_bacc": real_stats["center_equal_mean_bacc"],
        "d1_3_1_primary_context_center_equal_mean_bacc": d131_stats["center_equal_mean_bacc"],
        "shuffled_reliability_control_center_equal_mean_bacc": controls[ROW_SHUFFLED_RELIABILITY_CONTROL]["center_equal_mean_bacc"],
        "random_source_drop_control_center_equal_mean_bacc": controls[ROW_RANDOM_SOURCE_DROP_CONTROL]["center_equal_mean_bacc"],
        **weight_stats,
        **stats,
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


def _selection_weight_diagnostics(
    selection_rows: Sequence[Mapping[str, object]],
    reliability_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    fallback_rows = [row for row in reliability_rows if row.get("reliability_status") == "neutral_fallback"]
    rel_bacc = [_float(row.get("raw_reliability_bacc")) for row in reliability_rows if row.get("reliability_status") == "ok"]
    return {
        "mean_effective_num_sources": 3.0 if selection_rows else math.nan,
        "mean_reliability_weight_entropy": math.log(3.0) if selection_rows else math.nan,
        "mean_l1_distance_from_uniform": 0.0 if selection_rows else math.nan,
        "neutral_reliability_fallback_count": len(fallback_rows),
        "neutral_reliability_fallback_fraction": len(fallback_rows) / float(len(reliability_rows)) if reliability_rows else math.nan,
        "min_source_reliability_bacc": min([value for value in rel_bacc if math.isfinite(value)], default=math.nan),
    }


def _selected_source_histogram(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for source in str(row.get("selected_sources", "")).split("|"):
            if source:
                counts[source] = counts.get(source, 0) + 1
    return counts


def _write_artifacts(
    root: Path,
    cfg: DecentralizedReliabilityTop3Config,
    *,
    matrix_rows: Sequence[Mapping[str, object]],
    gap_rows: Sequence[Mapping[str, object]],
    summary_manifest_rows: Sequence[Mapping[str, object]],
    diagnostic_rows: Sequence[Mapping[str, object]],
    reliability_rows: Sequence[Mapping[str, object]],
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
    write_csv_rows(root / "tables" / "decentralized_reliability_top3_downstream_matrix.csv", matrix_rows, columns=_matrix_columns())
    write_csv_rows(root / "tables" / "decentralized_reliability_top3_gap_summary.csv", gap_rows, columns=_matrix_columns())
    write_csv_rows(root / "tables" / "decentralized_reliability_top3_summary.csv", [dict(decision)])
    write_csv_rows(root / "tables" / "source_reliability_manifest.csv", reliability_rows)
    write_csv_rows(root / "tables" / "reliability_top3_selection_manifest.csv", selection_rows)
    write_csv_rows(root / "tables" / "top3_selection_stability.csv", selection_rows)
    write_csv_rows(root / "tables" / "centerwise_delta_summary.csv", centerwise_rows)
    write_csv_rows(root / "tables" / "late_aggregation_matrix.csv", late_rows, columns=_matrix_columns())
    write_csv_rows(root / "tables" / "real_feature_reference_matrix.csv", real_feature_rows, columns=_matrix_columns())
    write_csv_rows(root / "tables" / "generated_component_coverage_audit.csv", coverage_rows)
    write_csv_rows(root / "tables" / "weak_source_audit.csv", weak_rows)
    write_csv_rows(root / "tables" / "nearest_neighbor_memorization_audit.csv", nn_rows)
    write_csv_rows(root / "tables" / "negative_control_summary.csv", [_negative_control_summary(decision)])
    write_csv_rows(root / "tables" / "exported_source_summary_manifest.csv", summary_manifest_rows, columns=d1a._summary_manifest_columns())
    write_csv_rows(root / "tables" / "source_summary_diagnostics.csv", diagnostic_rows, columns=d1a._diagnostic_columns())
    write_csv_rows(root / "manifests" / "decentralized_reliability_top3_prior_model_manifest.csv", model_manifest_rows)
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
            "schema_version": "cvae_rebuild_decentralized_reliability_top3_protocol_manifest_v1",
            "experiment_name": cfg.name,
            "experiment_type": "source_local_reliability_top3_sparse_decentralized_composition",
            "primary_variant": cfg.primary_variant,
            "primary_method": cfg.primary_method,
            "top_k_sources": cfg.top_k_sources,
            "target_support_labels_for_selection": False,
            "target_support_features_for_selection": False,
            "target_eval_labels_for_scoring_only": True,
            "target_expert_excluded": target_expert_excluded,
            "exported_source_summaries_are_target_agnostic": True,
            "raw_source_embedding_pooling_for_prior_fit": False,
            "adaptive_k_selection_uses_source_local_fit_statistics_only": True,
            "source_reliability_uses_source_local_eval_only": True,
            "support8_context_rows_decision_excluded": True,
            "oracle_rows_diagnostic_only": True,
            "protocol_wording": PROTOCOL_WORDING,
            "claim_boundary": (
                "source-local reliability-based sparse composition only; no target-conditioned compatibility routing claim, "
                "no support-NELBO claim, no metadata-routing claim, and no formal privacy claim"
            ),
        },
    )
    _write_decision_summary(root, decision, leakage_status=leakage_status)
    write_json(root / "run_config_resolved.yaml", _resolved_config(cfg))


def _matrix_columns() -> tuple[str, ...]:
    return d12._matrix_columns() + (
        "delta_vs_reliability_all4_weighted",
        "delta_vs_equal_all4",
        "delta_vs_d1_3_1_primary_context",
        "strongest_negative_control_gap",
        "shuffled_reliability_control_gap",
        "random_source_drop_control_gap",
    )


def _negative_control_summary(decision: Mapping[str, object]) -> dict[str, object]:
    return {
        "primary_method": PRIMARY_RELIABILITY_TOP3_METHOD,
        "control_methods": (
            f"{ROW_SHUFFLED_RELIABILITY_CONTROL}|{ROW_RANDOM_SOURCE_DROP_CONTROL}|"
            f"{ROW_SHUFFLED_SUMMARY_CONTROL}|{ROW_SHUFFLED_LABEL_CONTROL}"
        ),
        "primary_center_equal_mean_bacc": decision.get("center_equal_mean_bacc", math.nan),
        "strongest_negative_control_method": decision.get("strongest_negative_control_method", ""),
        "strongest_negative_control_center_equal_mean_bacc": decision.get("strongest_negative_control_center_equal_mean_bacc", math.nan),
        "negative_control_gap": decision.get("negative_control_gap", math.nan),
        "shuffled_reliability_control_gap": decision.get("shuffled_reliability_control_gap", math.nan),
        "random_source_drop_control_gap": decision.get("random_source_drop_control_gap", math.nan),
        "control_competitive": _float(decision.get("negative_control_gap")) < 0.03,
    }


def _write_decision_summary(root: Path, decision: Mapping[str, object], *, leakage_status: str) -> None:
    text = "\n".join(
        [
            "# D1.4: Locked Reliability-Only Top-3 Decentralized Composition Confirmation",
            "",
            "## Summary",
            "",
            f"- Primary method: `{PRIMARY_RELIABILITY_TOP3_METHOD}`",
            f"- Primary verdict: `{decision.get('primary_verdict', 'D1_4_FAIL')}`",
            f"- Diagnostic flags: `{decision.get('diagnostic_flags', '')}`",
            f"- Center-equal mean BACC: {_format_float(decision.get('center_equal_mean_bacc'))}",
            f"- Center-equal macro-F1: {_format_float(decision.get('center_equal_macro_f1'))}",
            f"- Min center BACC: {_format_float(decision.get('min_center_bacc'))}",
            f"- Seed std BACC: {_format_float(decision.get('seed_std_bacc'))}",
            f"- Delta vs reliability all4 weighted: {_format_float(decision.get('delta_vs_reliability_all4_weighted'))}",
            f"- Delta vs equal all4: {_format_float(decision.get('delta_vs_equal_all4'))}",
            f"- Top-3 downstream oracle containment: {_format_float(decision.get('top3_downstream_oracle_containment'))}",
            f"- Negative-control gap: {_format_float(decision.get('negative_control_gap'))}",
            f"- Shuffled-reliability control gap: {_format_float(decision.get('shuffled_reliability_control_gap'))}",
            f"- Random source-drop control gap: {_format_float(decision.get('random_source_drop_control_gap'))}",
            f"- Mean top-3 selection Jaccard: {_format_float(decision.get('mean_top3_selection_jaccard'))}",
            f"- Leakage status: `{leakage_status}`",
            "",
            "## Protocol Boundary",
            "",
            PROTOCOL_WORDING,
            "",
            "This is reliability-based sparse composition, not target-conditioned routing.",
            "It does not consume target support features. Target labels are final scoring only.",
            "Support8 D1.3.1 rows are historical context only and do not affect the D1.4 decision rule.",
            "",
            "## Supported Claim If PASS",
            "",
            "Source-local generation-preservation reliability can sparsify decentralized adaptive source-summary composition and improve heldout generated-embedding downstream utility without raw source pooling.",
            "",
        ]
    )
    path = root / "reports" / "decision_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _resolved_config(cfg: DecentralizedReliabilityTop3Config) -> dict[str, object]:
    return {
        "name": cfg.name,
        "artifact_root": str(cfg.artifact_root),
        "repair_artifact_root": str(cfg.repair_artifact_root),
        "d1_3_1_artifact_root": "" if cfg.d1_3_1_artifact_root is None else str(cfg.d1_3_1_artifact_root),
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
        "classifier": {
            "type": cfg.classifier_type,
            "solver": cfg.classifier_solver,
            "C": cfg.classifier_c,
            "max_iter": cfg.classifier_max_iter,
            "class_weight": cfg.classifier_class_weight,
            "classifier_seed": cfg.classifier_seed,
        },
    }
