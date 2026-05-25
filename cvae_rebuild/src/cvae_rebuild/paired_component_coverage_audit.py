from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .downstream import (
    PredictionBundle,
    evaluate_probability_predictions,
    fit_locked_logistic_classifier,
    weighted_geometric_probability_pool,
)
from .features import load_feature_cache, select_rows
from .metrics import nanmean
from .preservation import _hash_array
from .preservation_repair import (
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
from .preservation_sampling import DIAGNOSTIC_SELECTION, PRIMARY_SELECTION, _manifest_row, _per_source_variant, _runtime_source
from .protocol import ProtocolError, assert_candidate_pool, build_leakage_report
from .reporting import prepare_artifact_dirs, write_csv_rows, write_json
from .source_union_gmm_prior import _nearest_neighbor_row
from .splits import candidate_experts

from . import decentralized_adaptive_gmm_prior as d1a
from . import decentralized_k16_gmm_prior as d1
from . import decentralized_reliability_weighted_gmm_prior as d12
from . import paired_dense_all4_reliability_confirmation as paired


COMPONENT_COVERAGE_AUDIT_NAME = "virchow2_cvae_paired_component_coverage_audit_v1"
ROW_RELIABILITY_MULTINOMIAL128_REFERENCE = "paired_reliability_all4_weighted_multinomial128_geom_reference"
ROW_RELIABILITY_STRATIFIED128 = "paired_reliability_all4_weighted_component_stratified128_geom"
ROW_EQUAL_STRATIFIED128 = "paired_equal_all4_component_stratified128_geom"
ROW_RELIABILITY_MULTINOMIAL256 = "paired_reliability_all4_weighted_multinomial256_geom_diagnostic"
ROW_RELIABILITY_STRATIFIED256 = "paired_reliability_all4_weighted_component_stratified256_geom_diagnostic"
SAMPLING_MULTINOMIAL = "multinomial"
SAMPLING_STRATIFIED = "stratified_largest_remainder"
PROTOCOL_WORDING = (
    "This is a CVAE sampling-fidelity audit for dense all4 generated-embedding aggregation. "
    "Dense all4 inclusion, heldout-excluded reliability weights, reliability budgets, pooling, "
    "classifier settings, backbone, and LOQDO protocol stay fixed; only source-local GMM "
    "component sampling changes."
)


@dataclass(frozen=True)
class PairedComponentCoverageAuditConfig:
    name: str
    artifact_root: Path
    repair_artifact_root: Path
    paired_reliability_artifact_root: Path | None
    feature_cache_root: Path
    backbone: str
    experiment_seeds: tuple[int, ...]
    heldout_centers: tuple[str, ...]
    replicate_seeds: tuple[int, ...]
    synthetic_per_class_total: int
    diagnostic_synthetic_per_class_total: int
    min_per_source_per_class: int
    primary_variant: str
    primary_method: str
    candidate_components_per_source_class: tuple[int, ...]
    min_samples_per_component: int
    source_weighting: str
    component_sampling_rules: tuple[str, ...]
    gmm_covariance_type: str
    gmm_reg_covar: float
    gmm_n_init: int
    gmm_max_iter: int
    min_component_weight: float
    variance_floor: float
    reliability_floor_score: float
    reliability_epsilon: float
    primary_pooling: str
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
    def shrinkage_values(self) -> tuple[float, ...]:
        return (0.25, 0.5)

    @property
    def softmax_tau(self) -> float:
        return 1.0


def load_paired_component_coverage_audit_config(path: str | Path) -> PairedComponentCoverageAuditConfig:
    source = Path(path).resolve()
    data = _load_mapping(source)
    base_dir = source.parents[2] if len(source.parents) >= 3 else source.parent
    return parse_paired_component_coverage_audit_config(data, base_dir=base_dir)


def parse_paired_component_coverage_audit_config(
    data: Mapping[str, Any],
    *,
    base_dir: str | Path = ".",
) -> PairedComponentCoverageAuditConfig:
    base = Path(base_dir)
    experiment = _mapping(data, "experiment")
    inputs = _mapping(data, "inputs")
    run = _mapping(data, "run_matrix")
    if any(key in run for key in ("support_size", "support_seeds", "top_k_sources")):
        raise ProtocolError("Paired component coverage audit must not configure target support or top-k selection.")
    generation = _mapping(data, "generation")
    audit = _mapping(data, "paired_component_coverage_audit")
    classifier = _mapping(data, "classifier")
    cfg = PairedComponentCoverageAuditConfig(
        name=str(experiment["name"]),
        artifact_root=_path(base, str(experiment["artifact_root"])),
        repair_artifact_root=_path(base, str(inputs["repair_artifact_root"])),
        paired_reliability_artifact_root=_optional_path(base, inputs.get("paired_reliability_artifact_root")),
        feature_cache_root=_path(base, str(inputs["feature_cache_root"])),
        backbone=str(inputs.get("backbone", "")),
        experiment_seeds=tuple(int(v) for v in run["experiment_seeds"]),
        heldout_centers=tuple(str(v) for v in run["heldout_centers"]),
        replicate_seeds=tuple(int(v) for v in run["replicate_seeds"]),
        synthetic_per_class_total=int(generation["synthetic_per_class_total"]),
        diagnostic_synthetic_per_class_total=int(generation["diagnostic_synthetic_per_class_total"]),
        min_per_source_per_class=int(generation["min_per_source_per_class"]),
        primary_variant=str(experiment["primary_variant"]),
        primary_method=str(audit["primary_method"]),
        candidate_components_per_source_class=tuple(int(v) for v in audit["candidate_components_per_source_class"]),
        min_samples_per_component=int(audit["min_samples_per_component"]),
        source_weighting=str(audit["source_weighting"]),
        component_sampling_rules=tuple(str(v) for v in audit["component_sampling_rules"]),
        gmm_covariance_type=str(audit["gmm_covariance_type"]),
        gmm_reg_covar=float(audit["gmm_reg_covar"]),
        gmm_n_init=int(audit["gmm_n_init"]),
        gmm_max_iter=int(audit["gmm_max_iter"]),
        min_component_weight=float(audit["min_component_weight"]),
        variance_floor=float(audit["variance_floor"]),
        reliability_floor_score=float(audit["reliability_floor_score"]),
        reliability_epsilon=float(audit["reliability_epsilon"]),
        primary_pooling=str(audit["primary_pooling"]),
        classifier_type=str(classifier["type"]),
        classifier_solver=str(classifier["solver"]),
        classifier_c=float(classifier["C"]),
        classifier_max_iter=int(classifier["max_iter"]),
        classifier_class_weight=str(classifier["class_weight"]),
        classifier_seed=None if classifier.get("classifier_seed") is None else int(classifier["classifier_seed"]),
    )
    validate_paired_component_coverage_audit_config(cfg)
    return cfg


def validate_paired_component_coverage_audit_config(cfg: PairedComponentCoverageAuditConfig) -> None:
    if cfg.name != COMPONENT_COVERAGE_AUDIT_NAME:
        raise ProtocolError(f"Paired component coverage audit name must be {COMPONENT_COVERAGE_AUDIT_NAME!r}.")
    if cfg.backbone != "virchow2":
        raise ProtocolError("Paired component coverage audit is locked to backbone=virchow2.")
    if cfg.primary_variant != PRIMARY_VARIANT:
        raise ProtocolError(f"primary_variant must be {PRIMARY_VARIANT!r}.")
    if cfg.primary_method != ROW_RELIABILITY_STRATIFIED128:
        raise ProtocolError(f"primary_method must be {ROW_RELIABILITY_STRATIFIED128!r}.")
    if not cfg.experiment_seeds or not cfg.replicate_seeds:
        raise ProtocolError("Experiment and replicate seeds must be non-empty.")
    if cfg.heldout_centers != ("0", "1", "2", "3", "4"):
        raise ProtocolError("heldout_centers must be locked to centers 0..4.")
    if cfg.synthetic_per_class_total != 128 or cfg.diagnostic_synthetic_per_class_total != 256:
        raise ProtocolError("Component coverage audit must use primary budget 128/class and diagnostic budget 256/class.")
    if cfg.min_per_source_per_class != 8:
        raise ProtocolError("min_per_source_per_class must be locked to 8.")
    if cfg.candidate_components_per_source_class != (4, 3, 2, 1):
        raise ProtocolError("candidate_components_per_source_class must be locked to [4, 3, 2, 1].")
    if cfg.min_samples_per_component != 12:
        raise ProtocolError("min_samples_per_component must be locked to 12.")
    if cfg.source_weighting != "heldout_excluded_source_local_reliability_dense_all4":
        raise ProtocolError("source_weighting must be heldout_excluded_source_local_reliability_dense_all4.")
    if cfg.component_sampling_rules != (SAMPLING_MULTINOMIAL, SAMPLING_STRATIFIED):
        raise ProtocolError("component_sampling_rules must be [multinomial, stratified_largest_remainder].")
    if cfg.gmm_covariance_type != "diag":
        raise ProtocolError("gmm_covariance_type must be diag.")
    if cfg.gmm_reg_covar != 1.0e-4 or cfg.gmm_n_init != 5 or cfg.gmm_max_iter != 500:
        raise ProtocolError("GMM settings must match locked D1.2 values: reg_covar=1e-4, n_init=5, max_iter=500.")
    if cfg.min_component_weight != 0.02 or cfg.variance_floor != 1.0e-5:
        raise ProtocolError("GMM component and variance floors must match locked D1.2 values.")
    if cfg.reliability_floor_score <= 0.0 or cfg.reliability_epsilon <= 0.0:
        raise ProtocolError("Reliability floors must be positive.")
    if cfg.primary_pooling != "weighted_geometric":
        raise ProtocolError("primary_pooling must be weighted_geometric.")
    if cfg.classifier_type != "sklearn_logistic_regression":
        raise ProtocolError("classifier.type must be sklearn_logistic_regression.")
    if cfg.classifier_solver != "lbfgs" or cfg.classifier_c != 1.0 or cfg.classifier_max_iter != 2000:
        raise ProtocolError("Classifier solver/C/max_iter must remain locked.")
    if cfg.classifier_class_weight != "balanced" or cfg.classifier_seed is not None:
        raise ProtocolError("Classifier must use class_weight=balanced and classifier_seed=null.")


def run_paired_component_coverage_audit(
    cfg: PairedComponentCoverageAuditConfig,
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
    weight_rows: list[dict[str, object]] = []
    budget_rows: list[dict[str, object]] = []
    excluded_rows: list[dict[str, object]] = []
    late_rows: list[dict[str, object]] = []
    real_feature_rows: list[dict[str, object]] = []
    source_coverage_rows: list[dict[str, object]] = []
    aggregate_coverage_rows: list[dict[str, object]] = []
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
                    transform = paired._heldout_excluded_reliability_transform(cfg, heldout_center, candidates, rels)
                    reliability_rows.extend(
                        paired._source_reliability_rows(
                            cfg,
                            experiment_seed=int(experiment_seed),
                            replicate_seed=int(replicate_seed),
                            heldout_center=str(heldout_center),
                            candidates=candidates,
                            rels=rels,
                            transform=transform,
                        )
                    )
                    plans = _variant_plans(cfg, candidates, transform)
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
                            paired._excluded_cell_row(experiment_seed, heldout_center, replicate_seed, eval_error, n_eval=len(eval_labels))
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
                    ref_row = _extend_audit_row(ref_row)
                    real_late = [_extend_audit_row(row) for row in real_late]
                    real_feature_rows.append(ref_row)
                    matrix_rows.append(ref_row)
                    late_rows.extend(real_late)

                    for method, plan in plans.items():
                        rows, late, source_cov, aggregate_cov, weak, nn = _evaluate_component_variant(
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
                        source_coverage_rows.extend(source_cov)
                        aggregate_coverage_rows.extend(aggregate_cov)
                        weak_rows.extend(weak)
                        nn_rows.extend(nn)
    except ProtocolError as exc:
        protocol_violations.append(str(exc))

    invariant_rows = _pairing_invariant_rows(matrix_rows)
    invariant_pass = all(row.get("audit_status") == "PASS" for row in invariant_rows) if invariant_rows else False
    if not invariant_pass:
        protocol_violations.append("component_sampling_pairing_audit_failed")
    leakage = build_leakage_report(
        target_support_labels_for_selection=False,
        target_eval_labels_for_scoring_only=True,
        target_expert_excluded=target_expert_excluded,
        oracle_rows_diagnostic_only=True,
        extra_violations=protocol_violations,
    )
    center_rows = _center_summary_rows(matrix_rows)
    paired_delta_rows = _paired_delta_rows(matrix_rows)
    gap_rows = _gap_summary_rows(matrix_rows, aggregate_coverage_rows)
    decision = _decision(
        matrix_rows,
        aggregate_coverage_rows,
        cfg,
        leakage_status=leakage.status,
        invariant_pass=invariant_pass,
        paired_delta_rows=paired_delta_rows,
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
        summary_manifest_rows=summary_manifest_rows,
        diagnostic_rows=diagnostic_rows,
        late_rows=late_rows,
        real_feature_rows=real_feature_rows,
        source_coverage_rows=source_coverage_rows,
        aggregate_coverage_rows=aggregate_coverage_rows,
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


def _variant_plans(
    cfg: PairedComponentCoverageAuditConfig,
    sources: Sequence[str],
    transform: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    sources_tuple = tuple(str(source) for source in sources)
    rel_weights = {source: float(transform["weights"][source]) for source in sources_tuple}
    scores = {source: float(transform["imputed_scores"][source]) for source in sources_tuple}
    uniform_weights = {source: 1.0 / float(len(sources_tuple)) for source in sources_tuple}
    equal_budgets_128 = {
        source: int(value)
        for source, value in zip(sources_tuple, d1._balanced_counts(cfg.synthetic_per_class_total, len(sources_tuple)))
    }
    rel_budgets_128 = d12._weighted_budgets(
        cfg.synthetic_per_class_total,
        sources_tuple,
        rel_weights,
        cfg.min_per_source_per_class,
    )
    rel_budgets_256 = d12._weighted_budgets(
        cfg.diagnostic_synthetic_per_class_total,
        sources_tuple,
        rel_weights,
        cfg.min_per_source_per_class,
    )
    return {
        ROW_RELIABILITY_MULTINOMIAL128_REFERENCE: _plan(
            sources_tuple,
            rel_weights,
            rel_budgets_128,
            scores,
            synthetic_total=cfg.synthetic_per_class_total,
            component_sampling_rule=SAMPLING_MULTINOMIAL,
            weight_rule="heldout_excluded_reliability",
            budget_policy="reliability_largest_remainder_min8_total128",
            pairing_group="paired_component_coverage_reliability_budget128_v1",
            source_weighting="heldout_excluded_reliability_weighted_budgeted",
        ),
        ROW_RELIABILITY_STRATIFIED128: _plan(
            sources_tuple,
            rel_weights,
            rel_budgets_128,
            scores,
            synthetic_total=cfg.synthetic_per_class_total,
            component_sampling_rule=SAMPLING_STRATIFIED,
            weight_rule="heldout_excluded_reliability",
            budget_policy="reliability_largest_remainder_min8_total128",
            pairing_group="paired_component_coverage_reliability_budget128_v1",
            source_weighting="heldout_excluded_reliability_weighted_budgeted_component_stratified",
        ),
        ROW_EQUAL_STRATIFIED128: _plan(
            sources_tuple,
            uniform_weights,
            equal_budgets_128,
            scores,
            synthetic_total=cfg.synthetic_per_class_total,
            component_sampling_rule=SAMPLING_STRATIFIED,
            weight_rule="uniform",
            budget_policy="equal_total128",
            pairing_group="paired_component_coverage_equal_budget128_v1",
            source_weighting="equal_source_mass_component_stratified",
        ),
        ROW_RELIABILITY_MULTINOMIAL256: _plan(
            sources_tuple,
            rel_weights,
            rel_budgets_256,
            scores,
            synthetic_total=cfg.diagnostic_synthetic_per_class_total,
            component_sampling_rule=SAMPLING_MULTINOMIAL,
            weight_rule="heldout_excluded_reliability",
            budget_policy="reliability_largest_remainder_min8_total256",
            pairing_group="diagnostic_component_coverage_reliability_budget256_v1",
            source_weighting="heldout_excluded_reliability_weighted_budgeted_budget256_diagnostic",
        ),
        ROW_RELIABILITY_STRATIFIED256: _plan(
            sources_tuple,
            rel_weights,
            rel_budgets_256,
            scores,
            synthetic_total=cfg.diagnostic_synthetic_per_class_total,
            component_sampling_rule=SAMPLING_STRATIFIED,
            weight_rule="heldout_excluded_reliability",
            budget_policy="reliability_largest_remainder_min8_total256",
            pairing_group="diagnostic_component_coverage_reliability_budget256_v1",
            source_weighting="heldout_excluded_reliability_weighted_budgeted_component_stratified_budget256_diagnostic",
        ),
    }


def _plan(
    sources: Sequence[str],
    weights: Mapping[str, float],
    budgets: Mapping[str, int],
    scores: Mapping[str, float],
    *,
    synthetic_total: int,
    component_sampling_rule: str,
    weight_rule: str,
    budget_policy: str,
    pairing_group: str,
    source_weighting: str,
) -> dict[str, object]:
    plan = d12._with_weight_diagnostics(tuple(sources), weights, budgets, scores)
    plan.update(
        {
            "synthetic_per_class_total": int(synthetic_total),
            "component_sampling_rule": str(component_sampling_rule),
            "weight_rule": str(weight_rule),
            "budget_policy": str(budget_policy),
            "pairing_group": str(pairing_group),
            "generation_seed_method": f"{pairing_group}_{component_sampling_rule}",
            "source_weighting": str(source_weighting),
        }
    )
    return plan


def _evaluate_component_variant(
    cfg: PairedComponentCoverageAuditConfig,
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
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    sources = tuple(str(source) for source in candidates)
    status, error = d1a._composition_status(sources, summaries, control_mode="normal")
    cfg_total = replace(cfg, synthetic_per_class_total=int(plan["synthetic_per_class_total"]))
    key = _sampling_rule_pair_key(experiment_seed, heldout_center, replicate_seed, sources, plan)
    if status != "ok":
        row = d1a._dense_empty_row(
            cfg_total,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            candidates=sources,
            summaries=summaries,
            prior_method=method,
            source_union_ref=d1._missing_reference(),
            center_balanced_ref=d1._missing_reference(),
            real_feature_bacc=real_feature_bacc,
            status=status,
            error_message=error,
            claim_role=_claim_role(method),
        )
        row = d12._extend_row(row, weight_plan=plan, source_weighting=str(plan["source_weighting"]))
        return [_extend_audit_row(row, method=method, plan=plan, sampling_rule_pair_key=key)], [], [], [], [], []

    bundles, single_rows, source_cov, aggregate_counts, weak_rows, nn_rows, generated_hash = _source_generated_bundles_with_sampling(
        cfg_total,
        per_source_runtime=per_source_runtime,
        candidates=sources,
        summaries=summaries,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        eval_raw=eval_raw,
        eval_labels=eval_labels,
        prior_method=method,
        plan=plan,
    )
    weights = [float(plan["weights"][str(bundle.expert_id)]) for bundle in bundles]
    pooled = weighted_geometric_probability_pool(bundles, weights)
    single_baccs = [_float(row["bacc"]) for row in single_rows if row.get("status") == "ok"]
    single_macro = [_float(row["macro_f1"]) for row in single_rows if row.get("status") == "ok"]
    row = d1a._dense_result_row(
        cfg_total,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        candidates=sources,
        summaries=summaries,
        prior_method=method,
        pooling_rule="weighted_geometric",
        probabilities=pooled,
        eval_labels=eval_labels,
        generated_features_hash=generated_hash,
        source_union_ref=d1._missing_reference(),
        center_balanced_ref=d1._missing_reference(),
        real_feature_bacc=real_feature_bacc,
        mean_single_bacc=nanmean(single_baccs),
        oracle_single_bacc=max(single_baccs) if single_baccs else math.nan,
        mean_single_macro_f1=nanmean(single_macro),
        selection_source=PRIMARY_SELECTION if method == cfg.primary_method else DIAGNOSTIC_SELECTION,
        claim_role=_claim_role(method),
    )
    row = d12._extend_row(row, weight_plan=plan, source_weighting=str(plan["source_weighting"]))
    aggregate_row = _aggregate_component_coverage_row(
        row,
        aggregate_counts,
        cfg_total,
        candidates=sources,
        summaries=summaries,
        budgets={str(k): int(v) for k, v in plan["budgets"].items()},
        component_sampling_rule=str(plan["component_sampling_rule"]),
        sampling_rule_pair_key=key,
    )
    return (
        [_extend_audit_row(row, method=method, plan=plan, sampling_rule_pair_key=key)],
        [_extend_audit_row(row, method=method, plan=plan, sampling_rule_pair_key=key) for row in single_rows],
        [_extend_coverage_row(row, plan=plan, sampling_rule_pair_key=key) for row in source_cov],
        [aggregate_row],
        weak_rows,
        nn_rows,
    )


def _source_generated_bundles_with_sampling(
    cfg: PairedComponentCoverageAuditConfig,
    *,
    per_source_runtime: Mapping[str, object],
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    eval_raw: object,
    eval_labels: Sequence[int],
    prior_method: str,
    plan: Mapping[str, object],
) -> tuple[list[PredictionBundle], list[dict[str, object]], list[dict[str, object]], dict[int, dict[str, int]], list[dict[str, object]], list[dict[str, object]], str]:
    bundles: list[PredictionBundle] = []
    late_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    weak_rows: list[dict[str, object]] = []
    nn_rows: list[dict[str, object]] = []
    generated_hashes: list[str] = []
    aggregate_counts: dict[int, dict[str, int]] = {0: {}, 1: {}}
    for source_center in candidates:
        runtime = per_source_runtime[str(source_center)].runtime
        budget_per_class = int(plan["budgets"][str(source_center)])
        latent_seed = d1._latent_seed(
            experiment_seed,
            heldout_center,
            replicate_seed,
            str(plan["generation_seed_method"]),
            source_center,
            budget_per_class,
            str(plan["component_sampling_rule"]),
        )
        generated, labels, counts = _sample_source_from_summaries_with_rule(
            cfg,
            runtime,
            summaries,
            source_center=str(source_center),
            budget_per_class=budget_per_class,
            seed=latent_seed,
            control_mode="normal",
            component_sampling_rule=str(plan["component_sampling_rule"]),
        )
        for cls, cls_counts in counts.items():
            aggregate_counts.setdefault(int(cls), {}).update({str(key): int(value) for key, value in cls_counts.items()})
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
        )
        row.update(
            {
                "source_weighting": str(plan["source_weighting"]),
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
        row = d12._extend_row(row, weight_plan=plan, source_weighting=str(plan["source_weighting"]))
        late_rows.append(row)
        if _float(row["bacc"]) < 0.75:
            weak_rows.append(d1a._weak_row(row))
        coverage_rows.append(d1a._coverage_row(row, counts, candidates=candidates, summaries=summaries, control_mode="normal"))
        nn_rows.append(_nearest_neighbor_row(row, generated, runtime.source_train_embeddings))
        bundles.append(bundle)
    return bundles, late_rows, coverage_rows, aggregate_counts, weak_rows, nn_rows, _hash_strings(generated_hashes)


def _sample_source_from_summaries_with_rule(
    cfg: PairedComponentCoverageAuditConfig,
    runtime: object,
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    *,
    source_center: str,
    budget_per_class: int,
    seed: int,
    control_mode: str,
    component_sampling_rule: str,
) -> tuple[object, tuple[int, ...], dict[int, dict[str, int]]]:
    if component_sampling_rule == SAMPLING_MULTINOMIAL:
        return d1a._sample_source_from_summaries(
            cfg,
            runtime,
            summaries,
            source_center=source_center,
            budget_per_class=budget_per_class,
            seed=seed,
            control_mode=control_mode,
        )
    if component_sampling_rule != SAMPLING_STRATIFIED:
        raise ProtocolError(f"Unsupported component_sampling_rule: {component_sampling_rule}")
    rng = np.random.default_rng(int(seed))
    chunks = []
    labels = []
    component_counts: dict[int, dict[str, int]] = {}
    for label_cls in (0, 1):
        summary_cls = 1 - int(label_cls) if control_mode == "class_flip" else int(label_cls)
        summary = summaries[(str(source_center), int(summary_cls))]
        z_np, counts = _sample_latents_stratified_largest_remainder(
            summary,
            rng,
            int(budget_per_class),
            variance_floor=cfg.variance_floor,
            min_component_weight=cfg.min_component_weight,
        )
        decoded, _ = d1a._decode_latents(runtime, z_np, [int(label_cls)] * int(budget_per_class))
        chunks.append(decoded)
        labels.extend([int(label_cls)] * int(budget_per_class))
        component_counts[int(label_cls)] = {f"{source_center}:{key}": int(value) for key, value in counts.items()}
    return np.vstack(chunks), tuple(labels), component_counts


def _sample_latents_stratified_largest_remainder(
    summary: d1a.AdaptiveSourceLocalSummary,
    rng: object,
    n_samples: int,
    *,
    variance_floor: float,
    min_component_weight: float,
) -> tuple[object, dict[int, int]]:
    weights = d1._normalized_weights(summary.weights)
    counts = _stratified_largest_remainder_component_counts(weights, int(n_samples), min_component_weight=float(min_component_weight))
    components = np.asarray([component for component, count in sorted(counts.items()) for _idx in range(int(count))], dtype=int)
    if components.size != int(n_samples):
        raise ProtocolError("Stratified component allocation failed to preserve the requested sample count.")
    rng.shuffle(components)
    means = np.asarray(summary.means, dtype=np.float32)[components]
    variances = np.asarray(summary.diag_vars, dtype=np.float32)[components]
    eps = rng.normal(size=means.shape).astype(np.float32)
    z_np = means + np.sqrt(np.maximum(variances, float(variance_floor))).astype(np.float32) * eps
    return z_np, {int(k): int(v) for k, v in counts.items() if int(v) > 0}


def _stratified_largest_remainder_component_counts(
    weights: Sequence[float],
    n_samples: int,
    *,
    min_component_weight: float,
) -> dict[int, int]:
    values = d1._normalized_weights(weights)
    active = [idx for idx, value in enumerate(values) if float(value) >= float(min_component_weight)]
    if not active:
        active = list(range(int(values.shape[0])))
    counts = {int(idx): 0 for idx in range(int(values.shape[0]))}
    if int(n_samples) <= 0:
        return counts
    if int(n_samples) < len(active):
        ordered = sorted(active, key=lambda idx: (-float(values[idx]), int(idx)))
        for idx in ordered[: int(n_samples)]:
            counts[int(idx)] = 1
        return counts
    for idx in active:
        counts[int(idx)] = 1
    remaining = int(n_samples) - len(active)
    if remaining <= 0:
        return counts
    active_weights = np.asarray([float(values[idx]) for idx in active], dtype=float)
    active_weights = active_weights / float(active_weights.sum())
    exact = {int(idx): float(weight) * remaining for idx, weight in zip(active, active_weights)}
    for idx in active:
        counts[int(idx)] += int(math.floor(exact[int(idx)]))
    leftover = int(n_samples) - sum(counts.values())
    ordered = sorted(active, key=lambda idx: (-(exact[int(idx)] - math.floor(exact[int(idx)])), int(idx)))
    for idx in ordered[:leftover]:
        counts[int(idx)] += 1
    return counts


def _aggregate_component_coverage_row(
    row: Mapping[str, object],
    component_counts: Mapping[int, Mapping[str, int]],
    cfg: PairedComponentCoverageAuditConfig,
    *,
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    budgets: Mapping[str, int],
    component_sampling_rule: str,
    sampling_rule_pair_key: str,
) -> dict[str, object]:
    sampled = {f"{cls}:{component}" for cls, counts in component_counts.items() for component, count in counts.items() if int(count) > 0}
    active_keys: set[str] = set()
    sampled_keys: set[str] = set()
    active_mass = 0.0
    sampled_mass = 0.0
    zero_budget_excluded = 0
    zero_budget_excluded_mass = 0.0
    infeasible_source_class_budgets = 0
    min_budget = math.inf
    for cls in (0, 1):
        total_budget = float(sum(int(budgets[str(source)]) for source in candidates))
        for source in candidates:
            source_id = str(source)
            budget = int(budgets[source_id])
            min_budget = min(min_budget, float(budget))
            summary = summaries.get((source_id, int(cls)))
            if summary is None:
                continue
            weights = d1._normalized_weights(summary.weights)
            active = [idx for idx, weight in enumerate(weights) if float(weight) >= cfg.min_component_weight]
            if budget <= 0:
                zero_budget_excluded += len(active)
                zero_budget_excluded_mass += sum(float(weights[idx]) for idx in active)
                continue
            if budget < len(active):
                infeasible_source_class_budgets += 1
            source_mass = float(budget) / total_budget if total_budget > 0 else 0.0
            for component_idx in active:
                key = f"{cls}:{source_id}:{component_idx}"
                mass = source_mass * float(weights[component_idx])
                active_keys.add(key)
                active_mass += mass
                if key in sampled:
                    sampled_keys.add(key)
                    sampled_mass += mass
    unsampled = sorted(active_keys.difference(sampled_keys))
    active_count = len(active_keys)
    sampled_count = len(sampled_keys)
    unsampled_mass = active_mass - sampled_mass
    return {
        "experiment_seed": row["experiment_seed"],
        "heldout_center": row["heldout_center"],
        "replicate_seed": row["replicate_seed"],
        "expert_id": row["expert_id"],
        "expert_pool_type": row["expert_pool_type"],
        "variant_id": row["variant_id"],
        "prior_method": row["prior_method"],
        "component_sampling_rule": component_sampling_rule,
        "sampling_rule_pair_key": sampling_rule_pair_key,
        "synthetic_per_class_total": row["synthetic_per_class_total"],
        "synthetic_per_class_per_source_json": row["synthetic_per_class_per_source_json"],
        "active_component_count": active_count,
        "sampled_component_count": sampled_count,
        "unsampled_component_count": len(unsampled),
        "component_count_coverage": sampled_count / float(active_count) if active_count else math.nan,
        "active_component_weight_mass": active_mass,
        "sampled_component_weight_mass": sampled_mass,
        "unsampled_component_weight_mass": max(0.0, unsampled_mass),
        "component_weight_mass_coverage": sampled_mass / active_mass if active_mass > 0.0 else math.nan,
        "min_source_class_budget": int(min_budget) if math.isfinite(min_budget) else "",
        "num_source_class_budgets_below_active_components": int(infeasible_source_class_budgets),
        "components_excluded_due_to_zero_source_class_budget": int(zero_budget_excluded),
        "component_weight_mass_excluded_due_to_zero_source_class_budget": float(zero_budget_excluded_mass),
        "generated_component_counts_json": json.dumps({str(cls): dict(values) for cls, values in component_counts.items()}, sort_keys=True),
        "unsampled_active_components": "|".join(unsampled),
        "latent_component_undersampled": bool(unsampled),
    }


def _weight_manifest_rows(
    method: str,
    plan: Mapping[str, object],
    cfg: PairedComponentCoverageAuditConfig,
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
                "component_sampling_rule": plan["component_sampling_rule"],
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
                "synthetic_per_class_total": plan["synthetic_per_class_total"],
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
                    "component_sampling_rule": plan["component_sampling_rule"],
                    "synthetic_per_class_budget": plan["budgets"][source_id],
                    "synthetic_per_class_total": plan["synthetic_per_class_total"],
                    "budget_sum_per_class": sum(int(v) for v in plan["budgets"].values()),
                }
            )
    return rows


def _extend_audit_row(
    row: Mapping[str, object],
    *,
    method: str | None = None,
    plan: Mapping[str, object] | None = None,
    sampling_rule_pair_key: str = "",
) -> dict[str, object]:
    out = dict(row)
    if plan is None:
        out.update(
            {
                "weight_rule": "",
                "budget_policy": "",
                "pairing_group": "",
                "generation_seed_method": "",
                "component_sampling_rule": "",
                "sampling_rule_pair_key": sampling_rule_pair_key,
                "paired_reference_delta_bacc": math.nan,
                "paired_reference_delta_macro_f1": math.nan,
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
            "component_sampling_rule": plan["component_sampling_rule"],
            "sampling_rule_pair_key": sampling_rule_pair_key,
            "paired_reference_delta_bacc": math.nan,
            "paired_reference_delta_macro_f1": math.nan,
        }
    )
    if method is not None:
        out["prior_method"] = method
    return out


def _extend_coverage_row(row: Mapping[str, object], *, plan: Mapping[str, object], sampling_rule_pair_key: str) -> dict[str, object]:
    out = dict(row)
    out["component_sampling_rule"] = plan["component_sampling_rule"]
    out["sampling_rule_pair_key"] = sampling_rule_pair_key
    out["synthetic_per_class_total"] = plan["synthetic_per_class_total"]
    return out


def _ineligible_rows(
    cfg: PairedComponentCoverageAuditConfig,
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
        rows.append(_extend_audit_row(row))
    return rows


def _sampling_rule_pair_key(
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    plan: Mapping[str, object],
) -> str:
    return json.dumps(
        {
            "experiment_seed": int(experiment_seed),
            "heldout_center": str(heldout_center),
            "replicate_seed": int(replicate_seed),
            "source_set": "|".join(str(source) for source in candidates),
            "source_weight_json": json.dumps(dict(plan["weights"]), sort_keys=True),
            "source_budget_json": json.dumps(dict(plan["budgets"]), sort_keys=True),
            "synthetic_per_class_total": int(plan["synthetic_per_class_total"]),
        },
        sort_keys=True,
    )


def _pairing_invariant_rows(matrix_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[Mapping[str, object]]] = {}
    for row in matrix_rows:
        key = str(row.get("sampling_rule_pair_key", ""))
        if row.get("status") == "ok" and key and row.get("expert_id") == "dense_all_sources":
            groups.setdefault(key, []).append(row)
    rows = []
    for key, subset in sorted(groups.items()):
        methods = sorted(str(row.get("prior_method")) for row in subset)
        rules = sorted(str(row.get("component_sampling_rule")) for row in subset)
        pairing_groups = {str(row.get("pairing_group", "")) for row in subset}
        source_sets = {str(row.get("included_source_centers", "")) for row in subset}
        budgets = {str(row.get("synthetic_per_class_per_source_json", "")) for row in subset}
        weights = {str(row.get("reliability_weight_json", "")) for row in subset}
        expected_paired_rules = any("reliability_budget" in group for group in pairing_groups)
        paired_rules_present = {SAMPLING_MULTINOMIAL, SAMPLING_STRATIFIED}.issubset(set(rules))
        status = (
            "PASS"
            if len(source_sets) == 1
            and len(budgets) == 1
            and len(weights) == 1
            and (not expected_paired_rules or paired_rules_present)
            else "FAIL"
        )
        rows.append(
            {
                "audit_scope": "sampling_rule_pair",
                "sampling_rule_pair_key": key,
                "pairing_groups": "|".join(sorted(pairing_groups)),
                "methods": "|".join(methods),
                "component_sampling_rules": "|".join(rules),
                "n_methods": len(methods),
                "n_unique_source_sets": len(source_sets),
                "n_unique_budget_json": len(budgets),
                "n_unique_weight_json": len(weights),
                "n_unique_generated_hashes": len({str(row.get("generated_features_hash", "")) for row in subset}),
                "n_unique_prediction_hashes": len({str(row.get("prediction_hash", "")) for row in subset}),
                "audit_status": status,
            }
        )
    return rows


def _center_summary_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    out = []
    for method in _reported_methods(rows):
        cells = paired._seed_center_cells(d1a._rows_for(rows, method))
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
    baseline = paired._seed_center_cells(d1a._rows_for(rows, ROW_RELIABILITY_MULTINOMIAL128_REFERENCE))
    out = []
    for method in _reported_methods(rows):
        if method == ROW_RELIABILITY_MULTINOMIAL128_REFERENCE:
            continue
        cells = paired._seed_center_cells(d1a._rows_for(rows, method))
        values: list[float] = []
        by_center: dict[str, list[float]] = {}
        for key, value in cells.items():
            base = baseline.get(key, math.nan)
            if math.isfinite(value) and math.isfinite(base):
                delta = value - base
                values.append(delta)
                by_center.setdefault(key[1], []).append(delta)
        lo, hi = paired._bootstrap_ci(values)
        out.append(
            {
                "method": method,
                "baseline_method": ROW_RELIABILITY_MULTINOMIAL128_REFERENCE,
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


def _gap_summary_rows(
    matrix_rows: Sequence[Mapping[str, object]],
    aggregate_coverage_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    real_stats = paired._method_stats(matrix_rows, d1a.ROW_REAL_FEATURE_DENSE_REFERENCE)
    coverage = _coverage_stats_by_method(aggregate_coverage_rows)
    out = []
    for method in _reported_methods(matrix_rows):
        stats = paired._method_stats(matrix_rows, method)
        cov = coverage.get(method, {})
        out.append(
            {
                "method": method,
                **stats,
                "delta_vs_multinomial128_reference_center_equal_bacc": _float(stats["center_equal_mean_bacc"])
                - _float(paired._method_stats(matrix_rows, ROW_RELIABILITY_MULTINOMIAL128_REFERENCE)["center_equal_mean_bacc"]),
                "delta_vs_real_feature_dense_reference_center_equal_bacc": _float(stats["center_equal_mean_bacc"])
                - _float(real_stats["center_equal_mean_bacc"]),
                "mean_unsampled_component_count": cov.get("mean_unsampled_component_count", math.nan),
                "mean_unsampled_component_weight_mass": cov.get("mean_unsampled_component_weight_mass", math.nan),
                "mean_component_count_coverage": cov.get("mean_component_count_coverage", math.nan),
                "mean_component_weight_mass_coverage": cov.get("mean_component_weight_mass_coverage", math.nan),
            }
        )
    return out


def _decision(
    matrix_rows: Sequence[Mapping[str, object]],
    aggregate_coverage_rows: Sequence[Mapping[str, object]],
    cfg: PairedComponentCoverageAuditConfig,
    *,
    leakage_status: str,
    invariant_pass: bool,
    paired_delta_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    baseline = paired._method_stats(matrix_rows, ROW_RELIABILITY_MULTINOMIAL128_REFERENCE)
    stratified = paired._method_stats(matrix_rows, ROW_RELIABILITY_STRATIFIED128)
    delta_by_method = {str(row["method"]): row for row in paired_delta_rows}
    strat_delta = delta_by_method.get(ROW_RELIABILITY_STRATIFIED128, {})
    center_deltas = json.loads(str(strat_delta.get("per_center_paired_delta_json", "{}") or "{}"))
    centers_improved = sum(1 for value in center_deltas.values() if _float(value) > 0.0)
    coverage = _coverage_stats_by_method(aggregate_coverage_rows)
    base_cov = coverage.get(ROW_RELIABILITY_MULTINOMIAL128_REFERENCE, {})
    strat_cov = coverage.get(ROW_RELIABILITY_STRATIFIED128, {})
    min_center_loss = _float(stratified["min_center_bacc"]) - _float(baseline["min_center_bacc"])
    gates = {
        "protocol_pass": leakage_status == "PASS",
        "pairing_invariant_pass": bool(invariant_pass),
        "all_centers_represented": int(stratified["n_heldout_centers"]) >= len(cfg.heldout_centers),
        "min_two_eligible_seeds_per_center": int(stratified["min_eligible_seeds_per_center"]) >= 2,
        "center_equal_delta_ge_0p015": _float(stratified["center_equal_mean_bacc"]) - _float(baseline["center_equal_mean_bacc"]) >= 0.015,
        "seed_cell_delta_ge_0p010": _float(stratified["seed_cell_mean_bacc"]) - _float(baseline["seed_cell_mean_bacc"]) >= 0.010,
        "min_center_loss_ge_minus_0p010": min_center_loss >= -0.010,
        "seed_std_le_0p040": _float(stratified["seed_std_bacc"]) <= 0.040,
        "centers_improved_ge_3": centers_improved >= 3,
        "unsampled_count_lower": _float(strat_cov.get("mean_unsampled_component_count")) < _float(base_cov.get("mean_unsampled_component_count")),
        "unsampled_weight_mass_lower": _float(strat_cov.get("mean_unsampled_component_weight_mass")) < _float(base_cov.get("mean_unsampled_component_weight_mass")),
    }
    diagnostic256 = paired._method_stats(matrix_rows, ROW_RELIABILITY_STRATIFIED256)
    budget_headroom = _float(diagnostic256["center_equal_mean_bacc"]) > _float(baseline["center_equal_mean_bacc"]) + 0.015
    high_coverage = (
        _float(base_cov.get("mean_component_count_coverage")) >= 0.97
        and _float(base_cov.get("min_component_count_coverage")) >= 0.93
        and _float(base_cov.get("mean_unsampled_component_weight_mass")) <= 0.03
    )
    if leakage_status != "PASS":
        verdict = "PROTOCOL_FAIL"
    elif not invariant_pass:
        verdict = "COMPONENT_COVERAGE_PAIRING_FAIL"
    elif all(gates.values()):
        verdict = "PAIRED_COMPONENT_COVERAGE_STRATIFIED_PASS"
    elif budget_headroom:
        verdict = "SYNTHETIC_BUDGET_HEADROOM_DIAGNOSTIC_ONLY"
    else:
        verdict = "COMPONENT_COVERAGE_NOT_BOTTLENECK"
    return {
        "primary_verdict": verdict,
        "primary_method": cfg.primary_method,
        "baseline_method": ROW_RELIABILITY_MULTINOMIAL128_REFERENCE,
        "leakage_status": leakage_status,
        "pairing_invariant_pass": bool(invariant_pass),
        "decision_gates_json": json.dumps(gates, sort_keys=True),
        "baseline_seed_cell_mean_bacc": baseline["seed_cell_mean_bacc"],
        "baseline_center_equal_mean_bacc": baseline["center_equal_mean_bacc"],
        "stratified_seed_cell_mean_bacc": stratified["seed_cell_mean_bacc"],
        "stratified_center_equal_mean_bacc": stratified["center_equal_mean_bacc"],
        "stratified_min_center_bacc": stratified["min_center_bacc"],
        "stratified_seed_std_bacc": stratified["seed_std_bacc"],
        "stratified_delta_vs_baseline_seed_cell_bacc": _float(stratified["seed_cell_mean_bacc"]) - _float(baseline["seed_cell_mean_bacc"]),
        "stratified_delta_vs_baseline_center_equal_bacc": _float(stratified["center_equal_mean_bacc"]) - _float(baseline["center_equal_mean_bacc"]),
        "stratified_min_center_delta_vs_baseline": min_center_loss,
        "stratified_centers_improved_vs_baseline": centers_improved,
        "stratified_mean_paired_delta_bacc": strat_delta.get("mean_paired_delta_bacc", math.nan),
        "stratified_median_paired_delta_bacc": strat_delta.get("median_paired_delta_bacc", math.nan),
        "stratified_positive_paired_cells": strat_delta.get("positive_paired_cells", 0),
        "stratified_n_paired_cells": strat_delta.get("n_paired_cells", 0),
        "baseline_mean_component_count_coverage": base_cov.get("mean_component_count_coverage", math.nan),
        "baseline_min_component_count_coverage": base_cov.get("min_component_count_coverage", math.nan),
        "baseline_mean_component_weight_mass_coverage": base_cov.get("mean_component_weight_mass_coverage", math.nan),
        "baseline_mean_unsampled_component_count": base_cov.get("mean_unsampled_component_count", math.nan),
        "baseline_mean_unsampled_component_weight_mass": base_cov.get("mean_unsampled_component_weight_mass", math.nan),
        "stratified_mean_unsampled_component_count": strat_cov.get("mean_unsampled_component_count", math.nan),
        "stratified_mean_unsampled_component_weight_mass": strat_cov.get("mean_unsampled_component_weight_mass", math.nan),
        "high_baseline_coverage_falsification_context": bool(high_coverage),
        "budget256_headroom_diagnostic": bool(budget_headroom),
        "claim_boundary": PROTOCOL_WORDING,
        "target_center_excluded_from_reliability": True,
        "target_eval_labels_used_for_reliability": False,
        "target_eval_labels_used_for_weighting": False,
        "target_eval_labels_used_for_selection": False,
    }


def _coverage_stats_by_method(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for method in {str(row.get("prior_method")) for row in rows if row.get("prior_method")}:
        subset = [row for row in rows if str(row.get("prior_method")) == method]
        out[method] = {
            "mean_unsampled_component_count": nanmean([_float(row.get("unsampled_component_count")) for row in subset]),
            "mean_unsampled_component_weight_mass": nanmean([_float(row.get("unsampled_component_weight_mass")) for row in subset]),
            "mean_component_count_coverage": nanmean([_float(row.get("component_count_coverage")) for row in subset]),
            "min_component_count_coverage": min((_float(row.get("component_count_coverage")) for row in subset), default=math.nan),
            "mean_component_weight_mass_coverage": nanmean([_float(row.get("component_weight_mass_coverage")) for row in subset]),
        }
    return out


def _reported_methods(rows: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    present = {str(row.get("prior_method")) for row in rows if row.get("prior_method")}
    return tuple(method for method in _method_order() if method in present)


def _method_order() -> tuple[str, ...]:
    return (
        ROW_RELIABILITY_MULTINOMIAL128_REFERENCE,
        ROW_RELIABILITY_STRATIFIED128,
        ROW_EQUAL_STRATIFIED128,
        ROW_RELIABILITY_MULTINOMIAL256,
        ROW_RELIABILITY_STRATIFIED256,
    )


def _claim_role(method: str) -> str:
    if method == ROW_RELIABILITY_MULTINOMIAL128_REFERENCE:
        return "current_best_reliability_weighted_multinomial_reference"
    if method == ROW_RELIABILITY_STRATIFIED128:
        return "primary_component_stratified_sampling_audit"
    if method == ROW_EQUAL_STRATIFIED128:
        return "equal_all4_component_stratified_sampling_control"
    return "synthetic_budget_256_diagnostic_only"


def _write_artifacts(
    root: Path,
    cfg: PairedComponentCoverageAuditConfig,
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
    summary_manifest_rows: Sequence[Mapping[str, object]],
    diagnostic_rows: Sequence[Mapping[str, object]],
    late_rows: Sequence[Mapping[str, object]],
    real_feature_rows: Sequence[Mapping[str, object]],
    source_coverage_rows: Sequence[Mapping[str, object]],
    aggregate_coverage_rows: Sequence[Mapping[str, object]],
    weak_rows: Sequence[Mapping[str, object]],
    nn_rows: Sequence[Mapping[str, object]],
    model_manifest_rows: Sequence[Mapping[str, object]],
    decision: Mapping[str, object],
    leakage_status: str,
    protocol_violations: Sequence[str],
    target_expert_excluded: bool,
) -> None:
    write_csv_rows(root / "tables" / "paired_component_coverage_downstream_matrix.csv", matrix_rows, columns=_matrix_columns())
    write_csv_rows(root / "tables" / "paired_component_coverage_gap_summary.csv", gap_rows)
    write_csv_rows(root / "tables" / "paired_component_coverage_center_summary.csv", center_rows)
    write_csv_rows(root / "tables" / "paired_component_coverage_summary.csv", summary_rows)
    write_csv_rows(root / "tables" / "source_reliability_manifest.csv", reliability_rows)
    write_csv_rows(root / "tables" / "reliability_weight_manifest.csv", weight_rows)
    write_csv_rows(root / "tables" / "realized_budget_table.csv", budget_rows)
    write_csv_rows(root / "tables" / "excluded_cell_report.csv", excluded_rows)
    write_csv_rows(root / "tables" / "component_sampling_pairing_audit.csv", invariant_rows)
    write_csv_rows(root / "tables" / "paired_delta_summary.csv", paired_delta_rows)
    write_csv_rows(root / "tables" / "late_aggregation_matrix.csv", late_rows, columns=_matrix_columns())
    write_csv_rows(root / "tables" / "real_feature_reference_matrix.csv", real_feature_rows, columns=_matrix_columns())
    write_csv_rows(root / "tables" / "generated_component_coverage_audit.csv", source_coverage_rows)
    write_csv_rows(root / "tables" / "aggregate_component_coverage_audit.csv", aggregate_coverage_rows)
    write_csv_rows(root / "tables" / "weak_source_audit.csv", weak_rows)
    write_csv_rows(root / "tables" / "nearest_neighbor_memorization_audit.csv", nn_rows)
    write_csv_rows(root / "tables" / "exported_source_summary_manifest.csv", summary_manifest_rows, columns=d1a._summary_manifest_columns())
    write_csv_rows(root / "tables" / "source_summary_diagnostics.csv", diagnostic_rows, columns=d1a._diagnostic_columns())
    write_csv_rows(root / "manifests" / "paired_component_coverage_prior_model_manifest.csv", model_manifest_rows)
    leakage = build_leakage_report(
        target_support_labels_for_selection=False,
        target_eval_labels_for_scoring_only=True,
        target_expert_excluded=target_expert_excluded,
        oracle_rows_diagnostic_only=True,
        extra_violations=protocol_violations,
    )
    write_json(root / "reports" / "leakage_report.json", leakage.to_json_dict())
    write_json(root / "manifests" / "protocol_manifest.json", _protocol_manifest(cfg, target_expert_excluded))
    _write_decision_summary(root, decision, leakage_status=leakage_status)
    write_json(root / "run_config_resolved.yaml", _resolved_config(cfg))


def _matrix_columns() -> tuple[str, ...]:
    return d12._matrix_columns() + (
        "weight_rule",
        "budget_policy",
        "pairing_group",
        "generation_seed_method",
        "component_sampling_rule",
        "sampling_rule_pair_key",
        "paired_reference_delta_bacc",
        "paired_reference_delta_macro_f1",
    )


def _protocol_manifest(cfg: PairedComponentCoverageAuditConfig, target_expert_excluded: bool) -> dict[str, object]:
    return {
        "schema_version": "cvae_rebuild_paired_component_coverage_audit_protocol_manifest_v1",
        "experiment_name": cfg.name,
        "experiment_type": "paired_component_coverage_audit",
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
        "dense_all4_fixed_inclusion": True,
        "top_k_selection_enabled": False,
        "component_sampling_rules": list(cfg.component_sampling_rules),
        "primary_budget_per_class": cfg.synthetic_per_class_total,
        "diagnostic_budget_per_class": cfg.diagnostic_synthetic_per_class_total,
        "coverage_denominator_uses_realized_source_class_budget": True,
        "weighted_component_mass_coverage_enabled": True,
        "stratified_sampler_interpretation": (
            "Tests whether forced mode coverage helps more than faithful multinomial mixture sampling "
            "at fixed budget; it is not assumed mathematically superior."
        ),
        "oracle_rows_diagnostic_only": True,
        "protocol_wording": PROTOCOL_WORDING,
        "claim_boundary": (
            "CVAE sampling-fidelity audit for dense reliability-weighted generated-embedding aggregation; "
            "not sparse routing, not expert selection, and not target-conditioned calibration"
        ),
    }


def _write_decision_summary(root: Path, decision: Mapping[str, object], *, leakage_status: str) -> None:
    verdict = str(decision.get("primary_verdict", ""))
    text = "\n".join(
        [
            "# Paired Component Coverage Audit",
            "",
            "## Summary",
            "",
            f"- Primary method: `{decision.get('primary_method', ROW_RELIABILITY_STRATIFIED128)}`",
            f"- Baseline method: `{decision.get('baseline_method', ROW_RELIABILITY_MULTINOMIAL128_REFERENCE)}`",
            f"- Primary verdict: `{verdict}`",
            f"- Baseline center-equal BACC: {_format_float(decision.get('baseline_center_equal_mean_bacc'))}",
            f"- Stratified center-equal BACC: {_format_float(decision.get('stratified_center_equal_mean_bacc'))}",
            f"- Stratified delta vs baseline center-equal BACC: {_format_float(decision.get('stratified_delta_vs_baseline_center_equal_bacc'))}",
            f"- Stratified mean paired delta BACC: {_format_float(decision.get('stratified_mean_paired_delta_bacc'))}",
            f"- Baseline mean component-count coverage: {_format_float(decision.get('baseline_mean_component_count_coverage'))}",
            f"- Baseline mean component-mass coverage: {_format_float(decision.get('baseline_mean_component_weight_mass_coverage'))}",
            f"- Baseline mean unsampled component mass: {_format_float(decision.get('baseline_mean_unsampled_component_weight_mass'))}",
            f"- Stratified mean unsampled component mass: {_format_float(decision.get('stratified_mean_unsampled_component_weight_mass'))}",
            f"- Budget-256 headroom diagnostic: `{decision.get('budget256_headroom_diagnostic', False)}`",
            f"- High baseline coverage context: `{decision.get('high_baseline_coverage_falsification_context', False)}`",
            f"- Pairing invariant pass: `{decision.get('pairing_invariant_pass', False)}`",
            f"- Leakage status: `{leakage_status}`",
            "",
            "## Protocol Boundary",
            "",
            PROTOCOL_WORDING,
            "",
            "Target evaluation labels are final scoring only. Reliability is heldout-excluded and source-only. "
            "The 256-budget rows are diagnostic-only.",
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
    if verdict == "PAIRED_COMPONENT_COVERAGE_STRATIFIED_PASS":
        return (
            "Component-stratified source-local GMM sampling improves reliability-weighted dense aggregation "
            "by increasing aggregate coverage of source-local mixture modes under fixed budget."
        )
    if verdict == "SYNTHETIC_BUDGET_HEADROOM_DIAGNOSTIC_ONLY":
        return (
            "More synthetic budget may help, but the 256 rows are not adoption-eligible without a separate "
            "budget-controlled confirmation."
        )
    if verdict == "COMPONENT_COVERAGE_NOT_BOTTLENECK":
        return (
            "Simple GMM component undercoverage is unlikely to explain the remaining generated-feature gap; "
            "freeze reliability-weighted dense all4 as the best confirmed setup."
        )
    return "The result is not thesis-claimable until protocol and pairing audits pass."


def _resolved_config(cfg: PairedComponentCoverageAuditConfig) -> dict[str, object]:
    return {
        "name": cfg.name,
        "artifact_root": str(cfg.artifact_root),
        "repair_artifact_root": str(cfg.repair_artifact_root),
        "paired_reliability_artifact_root": "" if cfg.paired_reliability_artifact_root is None else str(cfg.paired_reliability_artifact_root),
        "feature_cache_root": str(cfg.feature_cache_root),
        "backbone": cfg.backbone,
        "experiment_seeds": list(cfg.experiment_seeds),
        "heldout_centers": list(cfg.heldout_centers),
        "replicate_seeds": list(cfg.replicate_seeds),
        "synthetic_per_class_total": cfg.synthetic_per_class_total,
        "diagnostic_synthetic_per_class_total": cfg.diagnostic_synthetic_per_class_total,
        "min_per_source_per_class": cfg.min_per_source_per_class,
        "primary_variant": cfg.primary_variant,
        "primary_method": cfg.primary_method,
        "candidate_components_per_source_class": list(cfg.candidate_components_per_source_class),
        "min_samples_per_component": cfg.min_samples_per_component,
        "source_weighting": cfg.source_weighting,
        "component_sampling_rules": list(cfg.component_sampling_rules),
        "gmm_covariance_type": cfg.gmm_covariance_type,
        "gmm_reg_covar": cfg.gmm_reg_covar,
        "gmm_n_init": cfg.gmm_n_init,
        "gmm_max_iter": cfg.gmm_max_iter,
        "min_component_weight": cfg.min_component_weight,
        "variance_floor": cfg.variance_floor,
        "reliability_floor_score": cfg.reliability_floor_score,
        "reliability_epsilon": cfg.reliability_epsilon,
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
