from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .downstream import evaluate_probability_predictions, fit_locked_logistic_classifier
from .features import load_feature_cache, select_rows
from .metrics import nanmean
from .preservation import _hash_array
from .preservation_repair import (
    NA,
    POOL_PER_SOURCE,
    POOL_SOURCE_UNION,
    PRIMARY_VARIANT,
    VariantRuntime,
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
from .preservation_sampling import DIAGNOSTIC_SELECTION, PRIMARY_SELECTION, UNION_VARIANT, RuntimeSource, _manifest_row, _per_source_variant, _runtime_source
from .prior_calibration import _decode_latents
from .protocol import ProtocolError, assert_candidate_pool, build_leakage_report
from .reporting import prepare_artifact_dirs, write_csv_rows, write_json
from .source_union_gmm_prior import _nearest_neighbor_row
from .splits import candidate_experts

from . import decentralized_adaptive_gmm_prior as d1a
from . import decentralized_k16_gmm_prior as d1
from . import decentralized_reliability_weighted_gmm_prior as d12


COMPONENT_UNION_NAME = "virchow2_cvae_decentralized_component_union_prior_v1"
COMPONENT_UNION_SHRINK025_V2_NAME = "virchow2_cvae_decentralized_component_union_reliability_shrink025_v2"
PRIMARY_COMPONENT_UNION_METHOD = "decentralized_component_union_uniform_gmm"
ROW_COMPONENT_UNION_SHRINK025 = "decentralized_component_union_reliability_shrink025"
ROW_COMPONENT_UNION_SHRINK050 = "decentralized_component_union_reliability_shrink050"
ROW_PROTOTYPE_UNION = "decentralized_prototype_union_uniform"
ROW_COMPONENT_UNION_BUDGET256 = "decentralized_component_union_uniform_gmm_budget256_diagnostic"
ROW_EQUAL_ALL4_REFERENCE = "decentralized_exported_adaptive_k_equal_geom_reference"
ROW_RELIABILITY_ALL4_REFERENCE = "decentralized_exported_adaptive_k_source_reliability_weighted_geom_reference"
ROW_SINGLE_MEAN = "per_source_component_union_single_expert_mean_reference"
ROW_SINGLE_ORACLE = "per_source_component_union_single_expert_oracle_reference"
ROW_SOURCE_UNION_K16_REFERENCE = "source_union_cc_diag_gmm_k16_prior_sample_reference"
ROW_CENTER_BALANCED_K16_REFERENCE = "source_union_center_balanced_cc_diag_gmm_k16_prior_sample_reference"
ROW_REAL_FEATURE_DENSE_REFERENCE = "real_source_embedding_classifier_dense_reference"
ROW_SHUFFLED_SUMMARY_CONTROL = "decentralized_component_union_shuffled_summary_control"
ROW_SHUFFLED_LABEL_CONTROL = "decentralized_component_union_shuffled_label_control"
ROW_SHUFFLED_RELIABILITY_CONTROL = "decentralized_component_union_shuffled_reliability_control"
MATCHED_SHUFFLED_RELIABILITY_PREFIX = "decentralized_component_union_shuffled_reliability_shrink025_perm"
ROW_RANDOM_SOURCE_MASS_CONTROL = "decentralized_component_union_random_source_mass_control"
POOL_COMPONENT_UNION = "decentralized_component_union_raw_pool"
GMM_SUMMARY_SCHEMA_VERSION = "decentralized_source_local_cc_diag_gmm_component_union_summary_v1"
PROTOTYPE_SCHEMA_VERSION = "decentralized_source_local_prototype_codebook_summary_v1"
PROTOCOL_WORDING = (
    "This is a data-minimizing, raw-data-free source-local latent summary-exchange protocol. "
    "It is not a formal differential privacy claim. Exported latent summaries may still contain "
    "distributional information derived from private data."
)


@dataclass(frozen=True)
class ComponentUnionConfig:
    name: str
    artifact_root: Path
    repair_artifact_root: Path
    d1_2_artifact_root: Path | None
    source_union_gmm_artifact_root: Path | None
    balanced_gmm_artifact_root: Path | None
    feature_cache_root: Path
    backbone: str
    experiment_seeds: tuple[int, ...]
    heldout_centers: tuple[str, ...]
    replicate_seeds: tuple[int, ...]
    strict_full_run_matrix: bool
    synthetic_per_class_total: int
    budget_diagnostic_per_class_total: int | None
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
    shrink_lambdas: tuple[float, ...]
    matched_shuffled_reliability_null_permutations: int
    prototype_candidate_counts_per_source_class: tuple[int, ...]
    prototype_min_samples_per_component: int
    prototype_variance_floor: float
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
class PrototypeSummary:
    experiment_seed: int
    source_center: str
    class_label: int
    selected_k: int
    selected_k_reason: str
    candidate_fit_status_json: str
    weights: object
    means: object
    diag_vars: object
    assigned_counts: tuple[int, ...]
    source_class_count: int
    min_component_weight: float
    min_assigned_samples: int
    min_diag_var: float
    max_diag_var: float
    component_entropy: float
    all_finite: bool
    summary_path: Path
    summary_hash: str
    fit_row_ids_hash: str
    parameter_hash: str
    expert_config_hash: str
    status: str
    error_message: str


def load_decentralized_component_union_prior_config(path: str | Path) -> ComponentUnionConfig:
    source = Path(path).resolve()
    data = _load_mapping(source)
    base_dir = source.parents[2] if len(source.parents) >= 3 else source.parent
    return parse_decentralized_component_union_prior_config(data, base_dir=base_dir)


def parse_decentralized_component_union_prior_config(
    data: Mapping[str, Any],
    *,
    base_dir: str | Path = ".",
) -> ComponentUnionConfig:
    base = Path(base_dir)
    experiment = _mapping(data, "experiment")
    inputs = _mapping(data, "inputs")
    run = _mapping(data, "run_matrix")
    generation = _mapping(data, "generation")
    prior = _mapping(data, "component_union_prior")
    classifier = _mapping(data, "classifier")
    budget_diag = generation.get("budget_diagnostic_per_class_total")
    cfg = ComponentUnionConfig(
        name=str(experiment["name"]),
        artifact_root=_path(base, str(experiment["artifact_root"])),
        repair_artifact_root=_path(base, str(inputs["repair_artifact_root"])),
        d1_2_artifact_root=_optional_path(base, inputs.get("d1_2_artifact_root")),
        source_union_gmm_artifact_root=_optional_path(base, inputs.get("source_union_gmm_artifact_root")),
        balanced_gmm_artifact_root=_optional_path(base, inputs.get("balanced_gmm_artifact_root")),
        feature_cache_root=_path(base, str(inputs["feature_cache_root"])),
        backbone=str(inputs.get("backbone", "")),
        experiment_seeds=tuple(int(v) for v in run["experiment_seeds"]),
        heldout_centers=tuple(str(v) for v in run["heldout_centers"]),
        replicate_seeds=tuple(int(v) for v in run["replicate_seeds"]),
        strict_full_run_matrix=bool(run.get("strict_full_run_matrix", False)),
        synthetic_per_class_total=int(generation["synthetic_per_class_total"]),
        budget_diagnostic_per_class_total=None if budget_diag is None else int(budget_diag),
        min_per_source_per_class=int(generation["min_per_source_per_class"]),
        primary_variant=str(experiment["primary_variant"]),
        primary_method=str(prior["primary_method"]),
        candidate_components_per_source_class=tuple(int(v) for v in prior["candidate_components_per_source_class"]),
        min_samples_per_component=int(prior["min_samples_per_component"]),
        source_weighting=str(prior["source_weighting"]),
        gmm_covariance_type=str(prior["gmm_covariance_type"]),
        gmm_reg_covar=float(prior["gmm_reg_covar"]),
        gmm_n_init=int(prior["gmm_n_init"]),
        gmm_max_iter=int(prior["gmm_max_iter"]),
        min_component_weight=float(prior["min_component_weight"]),
        variance_floor=float(prior["variance_floor"]),
        variance_ceiling_multiplier=float(prior["variance_ceiling_multiplier"]),
        primary_pooling=str(prior["primary_pooling"]),
        reliability_floor_score=float(prior["reliability_floor_score"]),
        shrink_lambdas=tuple(float(v) for v in prior["shrink_lambdas"]),
        matched_shuffled_reliability_null_permutations=int(prior.get("matched_shuffled_reliability_null_permutations", 0)),
        prototype_candidate_counts_per_source_class=tuple(int(v) for v in prior["prototype_candidate_counts_per_source_class"]),
        prototype_min_samples_per_component=int(prior["prototype_min_samples_per_component"]),
        prototype_variance_floor=float(prior["prototype_variance_floor"]),
        classifier_type=str(classifier["type"]),
        classifier_solver=str(classifier["solver"]),
        classifier_c=float(classifier["C"]),
        classifier_max_iter=int(classifier["max_iter"]),
        classifier_class_weight=str(classifier["class_weight"]),
        classifier_seed=None if classifier.get("classifier_seed") is None else int(classifier["classifier_seed"]),
    )
    validate_decentralized_component_union_prior_config(cfg)
    return cfg


def validate_decentralized_component_union_prior_config(cfg: ComponentUnionConfig) -> None:
    primary_by_name = {
        COMPONENT_UNION_NAME: PRIMARY_COMPONENT_UNION_METHOD,
        COMPONENT_UNION_SHRINK025_V2_NAME: ROW_COMPONENT_UNION_SHRINK025,
    }
    if cfg.name not in primary_by_name:
        raise ProtocolError(
            "Component-union experiment name must be one of "
            f"{tuple(primary_by_name)!r}."
        )
    if cfg.backbone != "virchow2":
        raise ProtocolError("Component-union prior is locked to backbone=virchow2.")
    if cfg.primary_variant != PRIMARY_VARIANT:
        raise ProtocolError(f"primary_variant must be {PRIMARY_VARIANT!r}.")
    if cfg.primary_method != primary_by_name[cfg.name]:
        raise ProtocolError(f"primary_method must be {primary_by_name[cfg.name]!r} for {cfg.name}.")
    if cfg.candidate_components_per_source_class != (4, 3, 2, 1):
        raise ProtocolError("candidate_components_per_source_class must be locked to [4, 3, 2, 1].")
    if cfg.prototype_candidate_counts_per_source_class != (4, 3, 2, 1):
        raise ProtocolError("prototype_candidate_counts_per_source_class must be locked to [4, 3, 2, 1].")
    if len(cfg.heldout_centers) != 5:
        raise ProtocolError("Component-union composition expects exactly five centers.")
    if cfg.strict_full_run_matrix:
        if cfg.experiment_seeds != (42, 43, 44):
            raise ProtocolError("strict_full_run_matrix requires experiment_seeds=[42, 43, 44].")
        if cfg.heldout_centers != ("0", "1", "2", "3", "4"):
            raise ProtocolError("strict_full_run_matrix requires heldout_centers=['0', '1', '2', '3', '4'].")
        if cfg.replicate_seeds != (17, 23, 31):
            raise ProtocolError("strict_full_run_matrix requires replicate_seeds=[17, 23, 31].")
    if cfg.synthetic_per_class_total != 128:
        raise ProtocolError("synthetic_per_class_total must be locked to 128 for the primary row.")
    if cfg.budget_diagnostic_per_class_total not in (None, 256):
        raise ProtocolError("budget_diagnostic_per_class_total may be null or 256 only.")
    if cfg.name == COMPONENT_UNION_SHRINK025_V2_NAME and cfg.budget_diagnostic_per_class_total is not None:
        raise ProtocolError("Shrink025 v2 must not run the budget-256 diagnostic.")
    if cfg.min_per_source_per_class != 8:
        raise ProtocolError("min_per_source_per_class must be locked to 8.")
    if cfg.source_weighting != "uniform_source_component_union":
        raise ProtocolError("source_weighting must be uniform_source_component_union.")
    if cfg.gmm_covariance_type != "diag":
        raise ProtocolError("gmm_covariance_type must be diag.")
    if cfg.primary_pooling != "pooled_raw_logistic":
        raise ProtocolError("primary_pooling must be pooled_raw_logistic.")
    if cfg.shrink_lambdas != (0.25, 0.5):
        raise ProtocolError("shrink_lambdas must be locked to [0.25, 0.5].")
    if cfg.name == COMPONENT_UNION_SHRINK025_V2_NAME:
        if cfg.matched_shuffled_reliability_null_permutations < 1:
            raise ProtocolError("Shrink025 v2 requires matched shuffled-reliability null permutations.")
        if cfg.strict_full_run_matrix and cfg.matched_shuffled_reliability_null_permutations != 20:
            raise ProtocolError("Strict shrink025 v2 requires exactly 20 matched shuffled-reliability null permutations.")
    if min(
        cfg.min_samples_per_component,
        cfg.prototype_min_samples_per_component,
        cfg.gmm_n_init,
        cfg.gmm_max_iter,
    ) < 1:
        raise ProtocolError("Component/prototype minimums and GMM iterations must be positive.")
    if min(
        cfg.gmm_reg_covar,
        cfg.min_component_weight,
        cfg.variance_floor,
        cfg.variance_ceiling_multiplier,
        cfg.reliability_floor_score,
        cfg.prototype_variance_floor,
    ) <= 0.0:
        raise ProtocolError("Component-union numeric floors must be positive.")
    if cfg.classifier_type != "sklearn_logistic_regression":
        raise ProtocolError("classifier.type must be sklearn_logistic_regression.")
    if cfg.classifier_solver != "lbfgs" or cfg.classifier_c != 1.0 or cfg.classifier_max_iter != 2000:
        raise ProtocolError("Classifier solver/C/max_iter must remain locked.")
    if cfg.classifier_class_weight != "balanced" or cfg.classifier_seed is not None:
        raise ProtocolError("Classifier must use class_weight=balanced and classifier_seed=null.")


def run_decentralized_component_union_prior(
    cfg: ComponentUnionConfig,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    root = prepare_artifact_dirs(Path(artifact_root) if artifact_root is not None else cfg.artifact_root)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "summaries").mkdir(parents=True, exist_ok=True)
    (root / "prototypes").mkdir(parents=True, exist_ok=True)

    matrix_rows: list[dict[str, object]] = []
    gap_rows: list[dict[str, object]] = []
    component_manifest_rows: list[dict[str, object]] = []
    source_summary_rows: list[dict[str, object]] = []
    prototype_manifest_rows: list[dict[str, object]] = []
    component_coverage_rows: list[dict[str, object]] = []
    source_weight_rows: list[dict[str, object]] = []
    reliability_rows: list[dict[str, object]] = []
    source_ablation_rows: list[dict[str, object]] = []
    paired_generation_rows: list[dict[str, object]] = []
    weak_rows: list[dict[str, object]] = []
    nn_rows: list[dict[str, object]] = []
    real_feature_rows: list[dict[str, object]] = []
    late_rows: list[dict[str, object]] = []
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
            gmm_summaries: dict[tuple[str, int], d1a.AdaptiveSourceLocalSummary] = {}
            shuffled_summaries: dict[tuple[str, int], d1a.AdaptiveSourceLocalSummary] = {}
            prototype_summaries: dict[tuple[str, int], PrototypeSummary] = {}
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
                model_manifest_rows.append(_manifest_row(experiment_seed, NA, runtime_source))

                summaries, detail_rows = _fit_and_export_pruned_gmm_summaries(
                    cfg,
                    root,
                    runtime_source.runtime,
                    experiment_seed=int(experiment_seed),
                    shuffled_label_control=False,
                )
                shuffled, shuffled_detail_rows = _fit_and_export_pruned_gmm_summaries(
                    cfg,
                    root,
                    runtime_source.runtime,
                    experiment_seed=int(experiment_seed),
                    shuffled_label_control=True,
                )
                prototypes, prototype_rows = _fit_and_export_prototype_summaries(
                    cfg,
                    root,
                    runtime_source.runtime,
                    experiment_seed=int(experiment_seed),
                )
                for summary in summaries:
                    gmm_summaries[(summary.source_center, summary.class_label)] = summary
                    source_summary_rows.append(d1a._summary_diagnostic_row(cfg, summary))
                for summary in shuffled:
                    shuffled_summaries[(summary.source_center, summary.class_label)] = summary
                    source_summary_rows.append(d1a._summary_diagnostic_row(cfg, summary))
                for row in detail_rows:
                    component_details[(str(row["source_center"]), int(row["class_label"]), int(row["source_component_id"]))] = row
                component_manifest_rows.extend(detail_rows)
                component_manifest_rows.extend(shuffled_detail_rows)
                for summary in prototypes:
                    prototype_summaries[(summary.source_center, summary.class_label)] = summary
                prototype_manifest_rows.extend(prototype_rows)

            reliability: dict[tuple[int, int, str], d12.SourceReliability] = {}
            for replicate_seed in cfg.replicate_seeds:
                for source_center in cfg.heldout_centers:
                    rel = d12._source_local_reliability(
                        cfg,
                        per_source_runtime=per_source_runtime,
                        summaries=gmm_summaries,
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
                    su_ref = d1._reference_for_cell(source_union_refs, experiment_seed, heldout_center, replicate_seed)
                    cb_ref = d1._reference_for_cell(center_balanced_refs, experiment_seed, heldout_center, replicate_seed)
                    rels = {
                        source: reliability[(int(experiment_seed), int(replicate_seed), str(source))]
                        for source in candidates
                    }
                    uniform_plan = _uniform_source_plan(cfg, candidates, rels, total=cfg.synthetic_per_class_total)
                    shrink025_plan = _shrink_source_plan(cfg, candidates, rels, shrink_lambda=0.25, total=cfg.synthetic_per_class_total)
                    shrink050_plan = _shrink_source_plan(cfg, candidates, rels, shrink_lambda=0.5, total=cfg.synthetic_per_class_total)
                    primary_plan = _primary_source_plan(cfg, candidates, rels, total=cfg.synthetic_per_class_total)
                    shuffled_reliability_plan = _shuffled_reliability_plan(
                        cfg,
                        candidates,
                        rels,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        replicate_seed=int(replicate_seed),
                        shrink_lambda=0.5,
                        total=cfg.synthetic_per_class_total,
                    )
                    matched_shuffled_plans = [
                        (
                            _matched_shuffled_reliability_method(permutation_id),
                            _shuffled_reliability_plan(
                                cfg,
                                candidates,
                                rels,
                                experiment_seed=int(experiment_seed),
                                heldout_center=str(heldout_center),
                                replicate_seed=int(replicate_seed),
                                shrink_lambda=0.25,
                                permutation_id=permutation_id,
                                total=cfg.synthetic_per_class_total,
                            ),
                        )
                        for permutation_id in range(cfg.matched_shuffled_reliability_null_permutations)
                    ]
                    random_source_plan = _random_source_mass_plan(
                        cfg,
                        candidates,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        replicate_seed=int(replicate_seed),
                        total=cfg.synthetic_per_class_total,
                    )
                    for method, plan in (
                        (PRIMARY_COMPONENT_UNION_METHOD, uniform_plan),
                        (ROW_COMPONENT_UNION_SHRINK025, shrink025_plan),
                        (ROW_COMPONENT_UNION_SHRINK050, shrink050_plan),
                        (ROW_SHUFFLED_RELIABILITY_CONTROL, shuffled_reliability_plan),
                        (ROW_RANDOM_SOURCE_MASS_CONTROL, random_source_plan),
                    ):
                        source_weight_rows.extend(_source_weight_manifest_rows(experiment_seed, replicate_seed, heldout_center, method, plan, rels))
                    for method, plan in matched_shuffled_plans:
                        source_weight_rows.extend(_source_weight_manifest_rows(experiment_seed, replicate_seed, heldout_center, method, plan, rels))
                    component_manifest_rows.extend(
                        _fold_component_manifest_rows(
                            cfg,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            candidates=candidates,
                            summaries=gmm_summaries,
                            component_details=component_details,
                            weight_plan=primary_plan,
                        )
                    )

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
                    ref_row = _component_extend_row(ref_row, source_weighting="not_applicable")
                    matrix_rows.append(ref_row)
                    real_feature_rows.append(ref_row)
                    late_rows.extend(_component_extend_rows(real_late))
                    real_feature_bacc = _float(ref_row["bacc"])

                    d1_equal_rows, d1_equal_late, coverage, weak, nn = d12._evaluate_weighted_variant(
                        cfg,
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
                        weight_plan=uniform_plan,
                        prior_method=d12.ROW_EQUAL_REFERENCE,
                        pooling_rule="geometric",
                        selection_source=DIAGNOSTIC_SELECTION,
                        claim_role="d1_1_equal_adaptive_late_geom_reference",
                    )
                    matrix_rows.extend(_rename_component_rows(d1_equal_rows, d12.ROW_EQUAL_REFERENCE, ROW_EQUAL_ALL4_REFERENCE))
                    late_rows.extend(_component_extend_rows(d1_equal_late))
                    component_coverage_rows.extend(coverage)
                    weak_rows.extend(weak)
                    nn_rows.extend(nn)

                    d12_rows, d12_late, coverage, weak, nn = d12._evaluate_weighted_variant(
                        cfg,
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
                        weight_plan=d12._weight_plan(cfg, candidates, rels, mode="linear"),
                        prior_method=d12.PRIMARY_RELIABILITY_METHOD,
                        pooling_rule="weighted_geometric",
                        selection_source=DIAGNOSTIC_SELECTION,
                        claim_role="d1_2_reliability_all4_late_geom_reference",
                    )
                    matrix_rows.extend(_rename_component_rows(d12_rows, d12.PRIMARY_RELIABILITY_METHOD, ROW_RELIABILITY_ALL4_REFERENCE))
                    late_rows.extend(_component_extend_rows(d12_late))
                    component_coverage_rows.extend(coverage)
                    weak_rows.extend(weak)
                    nn_rows.extend(nn)

                    primary_row: dict[str, object] | None = None
                    for method, plan, role in (
                        (PRIMARY_COMPONENT_UNION_METHOD, uniform_plan, "diagnostic_uniform_component_union"),
                        (ROW_COMPONENT_UNION_SHRINK025, shrink025_plan, "diagnostic_reliability_shrink025_component_union"),
                        (ROW_COMPONENT_UNION_SHRINK050, shrink050_plan, "diagnostic_reliability_shrink050_component_union"),
                        (ROW_SHUFFLED_RELIABILITY_CONTROL, shuffled_reliability_plan, "negative_control"),
                        (ROW_RANDOM_SOURCE_MASS_CONTROL, random_source_plan, "negative_control"),
                    ):
                        row, coverage_row, weak_row, nn_row, paired_row = _evaluate_gmm_component_union(
                            cfg,
                            per_source_runtime=per_source_runtime,
                            candidates=candidates,
                            summaries=gmm_summaries,
                            rels=rels,
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
                            selection_source=_selection_source_for_method(cfg, method),
                            claim_role=_claim_role_for_method(cfg, method, role),
                        )
                        matrix_rows.append(row)
                        component_coverage_rows.append(coverage_row)
                        if weak_row:
                            weak_rows.append(weak_row)
                        if nn_row:
                            nn_rows.append(nn_row)
                        paired_generation_rows.append(paired_row)
                        if _is_primary_method(cfg, method):
                            primary_row = row

                    if primary_row is None:
                        raise ProtocolError(f"Primary method {cfg.primary_method!r} was not evaluated.")

                    for method, plan in matched_shuffled_plans:
                        row, coverage_row, weak_row, nn_row, paired_row = _evaluate_gmm_component_union(
                            cfg,
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
                            claim_role="matched_shuffled_reliability_null",
                        )
                        matrix_rows.append(row)
                        component_coverage_rows.append(coverage_row)
                        if weak_row:
                            weak_rows.append(weak_row)
                        if nn_row:
                            nn_rows.append(nn_row)
                        paired_generation_rows.append(paired_row)

                    for method, summaries, control_mode in (
                        (ROW_SHUFFLED_SUMMARY_CONTROL, gmm_summaries, "class_flip"),
                        (ROW_SHUFFLED_LABEL_CONTROL, shuffled_summaries, "normal"),
                    ):
                        row, coverage_row, weak_row, nn_row, paired_row = _evaluate_gmm_component_union(
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
                            real_feature_bacc=real_feature_bacc,
                            weight_plan=uniform_plan,
                            prior_method=method,
                            selection_source=DIAGNOSTIC_SELECTION,
                            claim_role="negative_control",
                            control_mode=control_mode,
                        )
                        matrix_rows.append(row)
                        component_coverage_rows.append(coverage_row)
                        if weak_row:
                            weak_rows.append(weak_row)
                        if nn_row:
                            nn_rows.append(nn_row)
                        paired_generation_rows.append(paired_row)

                    prototype_row, coverage_row, weak_row, nn_row, paired_row = _evaluate_prototype_union(
                        cfg,
                        per_source_runtime=per_source_runtime,
                        candidates=candidates,
                        summaries=prototype_summaries,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        replicate_seed=int(replicate_seed),
                        eval_raw=eval_raw,
                        eval_labels=eval_labels,
                        source_union_ref=su_ref,
                        center_balanced_ref=cb_ref,
                        real_feature_bacc=real_feature_bacc,
                        weight_plan=uniform_plan,
                    )
                    matrix_rows.append(prototype_row)
                    component_coverage_rows.append(coverage_row)
                    if weak_row:
                        weak_rows.append(weak_row)
                    if nn_row:
                        nn_rows.append(nn_row)
                    paired_generation_rows.append(paired_row)

                    if cfg.budget_diagnostic_per_class_total is not None:
                        budget_plan = _uniform_source_plan(cfg, candidates, rels, total=cfg.budget_diagnostic_per_class_total)
                        row, coverage_row, weak_row, nn_row, paired_row = _evaluate_gmm_component_union(
                            cfg,
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
                            weight_plan=budget_plan,
                            prior_method=ROW_COMPONENT_UNION_BUDGET256,
                            selection_source=DIAGNOSTIC_SELECTION,
                            claim_role="diagnostic_budget256_component_union",
                        )
                        matrix_rows.append(row)
                        component_coverage_rows.append(coverage_row)
                        if weak_row:
                            weak_rows.append(weak_row)
                        if nn_row:
                            nn_rows.append(nn_row)
                        paired_generation_rows.append(paired_row)

                    source_ablation_rows.extend(
                        _evaluate_source_ablation_diagnostics(
                            cfg,
                            per_source_runtime=per_source_runtime,
                            all_centers=cfg.heldout_centers,
                            candidates=candidates,
                            summaries=gmm_summaries,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            replicate_seed=int(replicate_seed),
                            eval_raw=eval_raw,
                            eval_labels=eval_labels,
                            primary_bacc=_float(primary_row.get("bacc")),
                        )
                    )

                    _append_single_source_references(
                        cfg,
                        matrix_rows,
                        late_rows,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        replicate_seed=int(replicate_seed),
                        candidates=candidates,
                        summaries=gmm_summaries,
                        source_union_ref=su_ref,
                        center_balanced_ref=cb_ref,
                        real_feature_bacc=real_feature_bacc,
                    )
                    matrix_rows.append(_reference_matrix_row(cfg, experiment_seed=int(experiment_seed), heldout_center=str(heldout_center), replicate_seed=int(replicate_seed), candidates=candidates, prior_method=ROW_SOURCE_UNION_K16_REFERENCE, reference=su_ref))
                    matrix_rows.append(_reference_matrix_row(cfg, experiment_seed=int(experiment_seed), heldout_center=str(heldout_center), replicate_seed=int(replicate_seed), candidates=candidates, prior_method=ROW_CENTER_BALANCED_K16_REFERENCE, reference=cb_ref))
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
    decision = _decision(
        matrix_rows,
        cfg,
        leakage_status=leakage.status,
        source_ablation_rows=source_ablation_rows,
    )
    _write_artifacts(
        root,
        cfg,
        matrix_rows=matrix_rows,
        gap_rows=gap_rows,
        component_manifest_rows=component_manifest_rows,
        source_summary_rows=source_summary_rows,
        prototype_manifest_rows=prototype_manifest_rows,
        component_coverage_rows=component_coverage_rows,
        source_weight_rows=source_weight_rows,
        reliability_rows=reliability_rows,
        source_ablation_rows=source_ablation_rows,
        paired_generation_rows=paired_generation_rows,
        weak_rows=weak_rows,
        nn_rows=nn_rows,
        real_feature_rows=real_feature_rows,
        late_rows=late_rows,
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


def _matched_shuffled_reliability_method(permutation_id: int) -> str:
    return f"{MATCHED_SHUFFLED_RELIABILITY_PREFIX}{int(permutation_id):03d}"


def _is_matched_shuffled_reliability_method(method: object) -> bool:
    return str(method).startswith(MATCHED_SHUFFLED_RELIABILITY_PREFIX)


def _is_primary_method(cfg: ComponentUnionConfig, method: str) -> bool:
    return str(method) == cfg.primary_method


def _selection_source_for_method(cfg: ComponentUnionConfig, method: str) -> str:
    return PRIMARY_SELECTION if _is_primary_method(cfg, method) else DIAGNOSTIC_SELECTION


def _claim_role_for_method(cfg: ComponentUnionConfig, method: str, default_role: str) -> str:
    if not _is_primary_method(cfg, method):
        return default_role
    if method == ROW_COMPONENT_UNION_SHRINK025:
        return "primary_component_union_reliability_shrink025_confirmation_audit"
    return "primary_component_level_prior_composition"


def _primary_source_plan(
    cfg: ComponentUnionConfig,
    sources: Sequence[str],
    rels: Mapping[str, d12.SourceReliability],
    *,
    total: int,
) -> dict[str, object]:
    if cfg.primary_method == PRIMARY_COMPONENT_UNION_METHOD:
        return _uniform_source_plan(cfg, sources, rels, total=total)
    if cfg.primary_method == ROW_COMPONENT_UNION_SHRINK025:
        return _shrink_source_plan(cfg, sources, rels, shrink_lambda=0.25, total=total)
    raise ProtocolError(f"Unsupported component-union primary_method={cfg.primary_method!r}.")


def _fit_and_export_pruned_gmm_summaries(
    cfg: ComponentUnionConfig,
    root: Path,
    runtime: VariantRuntime,
    *,
    experiment_seed: int,
    shuffled_label_control: bool,
) -> tuple[tuple[d1a.AdaptiveSourceLocalSummary, ...], list[dict[str, object]]]:
    import torch  # type: ignore

    source_centers = {str(v) for v in runtime.source_train_centers}
    if len(source_centers) != 1 or runtime.expert_id not in source_centers:
        raise ProtocolError("Source-local component summaries must be fitted from exactly one source center.")
    x_np = np.asarray(runtime.source_train_embeddings, dtype=np.float32)
    y_np = np.asarray(runtime.source_train_labels, dtype=int)
    fit_labels = y_np.copy()
    if shuffled_label_control:
        rng = np.random.default_rng(d1._latent_seed(experiment_seed, runtime.expert_id, "component_union_shuffled_label_summary"))
        rng.shuffle(fit_labels)
    with torch.no_grad():
        x = torch.as_tensor(x_np, dtype=torch.float32)
        y = torch.as_tensor(y_np, dtype=torch.long)
        mu, _logvar = runtime.model.encode(x, y)
    mu_np = mu.detach().cpu().numpy()
    summaries: list[d1a.AdaptiveSourceLocalSummary] = []
    component_rows: list[dict[str, object]] = []
    for cls in (0, 1):
        positions = np.flatnonzero(fit_labels == int(cls))
        cls_mu = mu_np[positions]
        candidates = _fit_pruned_gmm_candidates(
            cfg,
            cls_mu,
            experiment_seed=experiment_seed,
            source_center=runtime.expert_id,
            class_label=int(cls),
            shuffled_label_control=shuffled_label_control,
        )
        valid = [candidate for candidate in candidates if candidate["status"] == "ok"]
        selected = valid[0] if valid else None
        status_json = json.dumps(_candidate_status_payload(candidates), sort_keys=True)
        summary = d1a._build_summary(
            cfg,
            root,
            runtime,
            experiment_seed=experiment_seed,
            class_label=int(cls),
            positions=positions,
            candidate=selected,
            selection_rule="largest_viable",
            selected_k_reason="largest_source_only_viable_k_with_assigned_count_pruning" if selected else "no_viable_source_local_k",
            candidate_fit_status_json=status_json,
            shuffled_label_control=shuffled_label_control,
        )
        summaries.append(summary)
        component_rows.extend(_source_component_rows(cfg, summary, selected, shuffled_label_control=shuffled_label_control))
    return tuple(summaries), component_rows


def _fit_pruned_gmm_candidates(
    cfg: ComponentUnionConfig,
    cls_mu: object,
    *,
    experiment_seed: int,
    source_center: str,
    class_label: int,
    shuffled_label_control: bool,
) -> list[dict[str, object]]:
    from sklearn.mixture import GaussianMixture  # type: ignore

    x = np.asarray(cls_mu, dtype=float)
    n = int(x.shape[0])
    empirical_var = np.var(x, axis=0, ddof=0) if x.ndim == 2 and n else np.asarray([], dtype=float)
    variance_ceiling = np.maximum.reduce(
        [
            empirical_var * float(cfg.variance_ceiling_multiplier),
            np.full_like(empirical_var, float(cfg.gmm_reg_covar) * float(cfg.variance_ceiling_multiplier)),
            np.full_like(empirical_var, float(cfg.variance_floor)),
        ]
    )
    out: list[dict[str, object]] = []
    for k in cfg.candidate_components_per_source_class:
        errors: list[str] = []
        min_count = int(k) * int(cfg.min_samples_per_component)
        if n < min_count:
            errors.append(f"source_class_count<{min_count}")
        weights = np.asarray([], dtype=float)
        means = np.empty((0, x.shape[1] if x.ndim == 2 else 0), dtype=float)
        diag_vars = np.empty_like(means)
        assigned_counts = np.zeros(int(k), dtype=int)
        converged = False
        n_iter = 0
        score = math.nan
        bic = math.nan
        if not errors:
            if int(k) == 1:
                weights, means, diag_vars, score, bic = d1a._empirical_k1_params(x, variance_floor=cfg.variance_floor)
                assigned_counts = np.asarray([n], dtype=int)
                converged = True
                n_iter = 1
            else:
                gmm = GaussianMixture(
                    n_components=int(k),
                    covariance_type="diag",
                    reg_covar=cfg.gmm_reg_covar,
                    n_init=cfg.gmm_n_init,
                    max_iter=cfg.gmm_max_iter,
                    random_state=d1._latent_seed(
                        experiment_seed,
                        source_center,
                        class_label,
                        int(k),
                        "component_union_local_gmm",
                        shuffled_label_control,
                    ),
                )
                gmm.fit(x)
                assignments = np.asarray(gmm.predict(x), dtype=int)
                assigned_counts = np.bincount(assignments, minlength=int(k)).astype(int)
                weights = np.asarray(gmm.weights_, dtype=float)
                means = np.asarray(gmm.means_, dtype=float)
                diag_vars = np.maximum(np.asarray(gmm.covariances_, dtype=float), cfg.variance_floor)
                converged = bool(gmm.converged_)
                n_iter = int(gmm.n_iter_)
                score = float(gmm.score(x))
                bic = float(gmm.bic(x))
                if not converged:
                    errors.append("gmm_converged=false")
        finite = bool(np.isfinite(weights).all() and np.isfinite(means).all() and np.isfinite(diag_vars).all())
        effective = int(np.sum(weights >= cfg.min_component_weight)) if weights.size else 0
        min_weight = float(np.min(weights)) if weights.size else math.nan
        min_diag_var = float(np.min(diag_vars)) if diag_vars.size else math.nan
        max_diag_var = float(np.max(diag_vars)) if diag_vars.size else math.nan
        min_assigned = int(np.min(assigned_counts)) if assigned_counts.size else 0
        above_ceiling = int(np.sum(diag_vars > variance_ceiling[None, :])) if diag_vars.size and variance_ceiling.size else 0
        if not errors and effective != int(k):
            errors.append(f"effective_component_count!={int(k)}")
        if not errors and min_assigned < cfg.min_samples_per_component:
            errors.append(f"min_assigned_samples<{cfg.min_samples_per_component}")
        if not errors and (not math.isfinite(min_weight) or min_weight < cfg.min_component_weight):
            errors.append(f"min_component_weight<{cfg.min_component_weight}")
        if not errors and not finite:
            errors.append("nonfinite_summary_parameter")
        if not errors and (not math.isfinite(min_diag_var) or min_diag_var < cfg.variance_floor):
            errors.append(f"diag_var<{cfg.variance_floor}")
        if not errors and above_ceiling > 0:
            errors.append("diag_var_above_source_class_empirical_ceiling")
        out.append(
            {
                "k": int(k),
                "status": "ok" if not errors else "ineligible_component_fit",
                "error_message": "|".join(errors),
                "weights": weights,
                "means": means,
                "diag_vars": diag_vars,
                "assigned_counts": assigned_counts,
                "variance_ceiling": variance_ceiling,
                "effective_component_count": effective,
                "min_component_weight": min_weight,
                "min_assigned_samples": min_assigned,
                "min_diag_var": min_diag_var,
                "max_diag_var": max_diag_var,
                "num_variances_above_ceiling": above_ceiling,
                "component_entropy": d1._entropy(weights),
                "all_finite": finite,
                "gmm_converged": converged,
                "gmm_n_iter": n_iter,
                "source_train_log_likelihood": score,
                "bic": bic,
            }
        )
    return out


def _candidate_status_payload(candidates: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "k": int(candidate["k"]),
            "status": candidate["status"],
            "error_message": candidate["error_message"],
            "bic": candidate["bic"],
            "effective_component_count": candidate["effective_component_count"],
            "min_component_weight": candidate["min_component_weight"],
            "min_assigned_samples": candidate.get("min_assigned_samples", 0),
            "min_diag_var": candidate["min_diag_var"],
            "max_diag_var": candidate.get("max_diag_var", math.nan),
            "num_variances_above_ceiling": candidate.get("num_variances_above_ceiling", 0),
        }
        for candidate in candidates
    ]


def _source_component_rows(
    cfg: ComponentUnionConfig,
    summary: d1a.AdaptiveSourceLocalSummary,
    candidate: Mapping[str, object] | None,
    *,
    shuffled_label_control: bool,
) -> list[dict[str, object]]:
    if candidate is None or summary.status != "ok":
        return [
            {
                "row_scope": "source_local_export",
                "experiment_seed": int(summary.experiment_seed),
                "heldout_center": "",
                "source_center": summary.source_center,
                "class_label": int(summary.class_label),
                "source_component_id": "",
                "selected_k": int(summary.selected_k),
                "component_weight_local": "",
                "component_weight_after_source_normalization": "",
                "assigned_posterior_samples": "",
                "source_class_count": int(summary.source_class_count),
                "min_assigned_samples_per_component": cfg.min_samples_per_component,
                "min_component_weight": cfg.min_component_weight,
                "variance_floor": cfg.variance_floor,
                "variance_ceiling_multiplier": cfg.variance_ceiling_multiplier,
                "component_hash": "",
                "summary_hash": summary.summary_hash,
                "summary_status": summary.status,
                "summary_error_message": summary.error_message,
                "shuffled_label_control": bool(shuffled_label_control),
            }
        ]
    weights = d1._normalized_weights(summary.weights)
    assigned = [int(v) for v in np.asarray(candidate.get("assigned_counts", []), dtype=int).tolist()]
    ceilings = np.asarray(candidate.get("variance_ceiling", []), dtype=float)
    rows: list[dict[str, object]] = []
    for idx, local_weight in enumerate(weights):
        mean = np.asarray(summary.means, dtype=float)[idx]
        var = np.asarray(summary.diag_vars, dtype=float)[idx]
        rows.append(
            {
                "row_scope": "source_local_export",
                "experiment_seed": int(summary.experiment_seed),
                "heldout_center": "",
                "source_center": summary.source_center,
                "class_label": int(summary.class_label),
                "source_component_id": int(idx),
                "selected_k": int(summary.selected_k),
                "component_weight_local": float(local_weight),
                "component_weight_after_source_normalization": "",
                "assigned_posterior_samples": assigned[idx] if idx < len(assigned) else "",
                "source_class_count": int(summary.source_class_count),
                "min_assigned_samples_per_component": cfg.min_samples_per_component,
                "min_component_weight": cfg.min_component_weight,
                "variance_floor": cfg.variance_floor,
                "variance_ceiling_multiplier": cfg.variance_ceiling_multiplier,
                "component_min_diag_var": float(np.min(var)),
                "component_max_diag_var": float(np.max(var)),
                "variance_ceiling_min": float(np.min(ceilings)) if ceilings.size else math.nan,
                "variance_ceiling_max": float(np.max(ceilings)) if ceilings.size else math.nan,
                "num_dimensions_above_variance_ceiling": int(np.sum(var > ceilings)) if ceilings.size else 0,
                "component_hash": d1._hash_array(d1._flatten_payload([mean, var, np.asarray([local_weight], dtype=float)])),
                "summary_hash": summary.summary_hash,
                "summary_status": summary.status,
                "summary_error_message": summary.error_message,
                "shuffled_label_control": bool(shuffled_label_control),
            }
        )
    return rows


def _fit_and_export_prototype_summaries(
    cfg: ComponentUnionConfig,
    root: Path,
    runtime: VariantRuntime,
    *,
    experiment_seed: int,
) -> tuple[tuple[PrototypeSummary, ...], list[dict[str, object]]]:
    import torch  # type: ignore

    source_centers = {str(v) for v in runtime.source_train_centers}
    if len(source_centers) != 1 or runtime.expert_id not in source_centers:
        raise ProtocolError("Source-local prototypes must be fitted from exactly one source center.")
    x_np = np.asarray(runtime.source_train_embeddings, dtype=np.float32)
    y_np = np.asarray(runtime.source_train_labels, dtype=int)
    with torch.no_grad():
        x = torch.as_tensor(x_np, dtype=torch.float32)
        y = torch.as_tensor(y_np, dtype=torch.long)
        mu, _logvar = runtime.model.encode(x, y)
    mu_np = mu.detach().cpu().numpy()
    summaries: list[PrototypeSummary] = []
    rows: list[dict[str, object]] = []
    for cls in (0, 1):
        positions = np.flatnonzero(y_np == int(cls))
        candidates = _fit_prototype_candidates(
            cfg,
            mu_np[positions],
            experiment_seed=experiment_seed,
            source_center=runtime.expert_id,
            class_label=int(cls),
        )
        valid = [candidate for candidate in candidates if candidate["status"] == "ok"]
        selected = valid[0] if valid else None
        status_json = json.dumps(_candidate_status_payload(candidates), sort_keys=True)
        summary = _build_prototype_summary(
            cfg,
            root,
            runtime,
            experiment_seed=experiment_seed,
            class_label=int(cls),
            positions=positions,
            candidate=selected,
            selected_k_reason="largest_source_only_viable_prototype_codebook" if selected else "no_viable_source_local_prototype_codebook",
            candidate_fit_status_json=status_json,
        )
        summaries.append(summary)
        rows.extend(_prototype_manifest_rows(cfg, summary))
    return tuple(summaries), rows


def _fit_prototype_candidates(
    cfg: ComponentUnionConfig,
    cls_mu: object,
    *,
    experiment_seed: int,
    source_center: str,
    class_label: int,
) -> list[dict[str, object]]:
    from sklearn.cluster import KMeans  # type: ignore

    x = np.asarray(cls_mu, dtype=float)
    n = int(x.shape[0])
    out: list[dict[str, object]] = []
    for k in cfg.prototype_candidate_counts_per_source_class:
        errors: list[str] = []
        min_count = int(k) * int(cfg.prototype_min_samples_per_component)
        if n < min_count:
            errors.append(f"source_class_count<{min_count}")
        weights = np.asarray([], dtype=float)
        means = np.empty((0, x.shape[1] if x.ndim == 2 else 0), dtype=float)
        diag_vars = np.empty_like(means)
        assigned_counts = np.zeros(int(k), dtype=int)
        if not errors:
            if int(k) == 1:
                assigned = np.zeros(n, dtype=int)
                means = x.mean(axis=0, keepdims=True)
            else:
                model = KMeans(
                    n_clusters=int(k),
                    n_init=10,
                    random_state=d1._latent_seed(experiment_seed, source_center, class_label, int(k), "prototype_codebook"),
                )
                assigned = np.asarray(model.fit_predict(x), dtype=int)
                means = np.asarray(model.cluster_centers_, dtype=float)
            assigned_counts = np.bincount(assigned, minlength=int(k)).astype(int)
            weights = assigned_counts.astype(float) / float(max(n, 1))
            diag_vars = np.zeros_like(means, dtype=float)
            for idx in range(int(k)):
                cluster = x[assigned == idx]
                if cluster.shape[0] == 0:
                    diag_vars[idx] = cfg.prototype_variance_floor
                else:
                    diag_vars[idx] = np.maximum(cluster.var(axis=0, ddof=0), cfg.prototype_variance_floor)
        finite = bool(np.isfinite(weights).all() and np.isfinite(means).all() and np.isfinite(diag_vars).all())
        min_assigned = int(np.min(assigned_counts)) if assigned_counts.size else 0
        min_weight = float(np.min(weights)) if weights.size else math.nan
        min_diag_var = float(np.min(diag_vars)) if diag_vars.size else math.nan
        if not errors and min_assigned < cfg.prototype_min_samples_per_component:
            errors.append(f"min_assigned_samples<{cfg.prototype_min_samples_per_component}")
        if not errors and (not math.isfinite(min_weight) or min_weight <= 0.0):
            errors.append("nonpositive_prototype_weight")
        if not errors and not finite:
            errors.append("nonfinite_prototype_parameter")
        out.append(
            {
                "k": int(k),
                "status": "ok" if not errors else "ineligible_component_fit",
                "error_message": "|".join(errors),
                "weights": weights,
                "means": means,
                "diag_vars": diag_vars,
                "assigned_counts": assigned_counts,
                "effective_component_count": int(np.sum(weights > 0.0)) if weights.size else 0,
                "min_component_weight": min_weight,
                "min_assigned_samples": min_assigned,
                "min_diag_var": min_diag_var,
                "max_diag_var": float(np.max(diag_vars)) if diag_vars.size else math.nan,
                "num_variances_above_ceiling": 0,
                "component_entropy": d1._entropy(weights),
                "all_finite": finite,
                "gmm_converged": True,
                "gmm_n_iter": 1,
                "source_train_log_likelihood": math.nan,
                "bic": math.nan,
            }
        )
    return out


def _build_prototype_summary(
    cfg: ComponentUnionConfig,
    root: Path,
    runtime: VariantRuntime,
    *,
    experiment_seed: int,
    class_label: int,
    positions: object,
    candidate: Mapping[str, object] | None,
    selected_k_reason: str,
    candidate_fit_status_json: str,
) -> PrototypeSummary:
    weights = np.asarray(candidate["weights"], dtype=float) if candidate else np.asarray([], dtype=float)
    means = np.asarray(candidate["means"], dtype=float) if candidate else np.empty((0, int(runtime.model.latent_dim)), dtype=float)
    diag_vars = np.asarray(candidate["diag_vars"], dtype=float) if candidate else np.empty((0, int(runtime.model.latent_dim)), dtype=float)
    assigned = tuple(int(v) for v in np.asarray(candidate.get("assigned_counts", []), dtype=int).tolist()) if candidate else tuple()
    selected_k = int(candidate["k"]) if candidate else 0
    status = "ok" if candidate else "ineligible_component_fit"
    error_message = "" if candidate else "no_viable_source_local_prototype_codebook"
    parameter_hash = d1._hash_array(d1._flatten_payload([weights, means, diag_vars])) if weights.size else ""
    summary_path = root / "prototypes" / f"seed_{int(experiment_seed)}" / f"source_{runtime.expert_id}" / f"class_{int(class_label)}_prototype_summary.npz"
    summary_hash = ""
    if weights.size:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            summary_path,
            weights=weights,
            means=means,
            diag_vars=diag_vars,
            assigned_counts=np.asarray(assigned, dtype=int),
            source_class_count=np.asarray([int(np.asarray(positions).size)], dtype=int),
            selected_k=np.asarray([selected_k], dtype=int),
            candidate_fit_status_json=np.asarray([candidate_fit_status_json]),
            schema_version=np.asarray([PROTOTYPE_SCHEMA_VERSION]),
            expert_config_hash=np.asarray([d1._expert_config_hash(runtime)]),
        )
        summary_hash = d1._file_sha256(summary_path)
    return PrototypeSummary(
        experiment_seed=int(experiment_seed),
        source_center=runtime.expert_id,
        class_label=int(class_label),
        selected_k=selected_k,
        selected_k_reason=str(selected_k_reason),
        candidate_fit_status_json=str(candidate_fit_status_json),
        weights=weights,
        means=means,
        diag_vars=diag_vars,
        assigned_counts=assigned,
        source_class_count=int(np.asarray(positions).size),
        min_component_weight=float(candidate["min_component_weight"]) if candidate else math.nan,
        min_assigned_samples=int(candidate["min_assigned_samples"]) if candidate else 0,
        min_diag_var=float(candidate["min_diag_var"]) if candidate else math.nan,
        max_diag_var=float(candidate["max_diag_var"]) if candidate else math.nan,
        component_entropy=float(candidate["component_entropy"]) if candidate else math.nan,
        all_finite=bool(candidate["all_finite"]) if candidate else False,
        summary_path=summary_path,
        summary_hash=summary_hash,
        fit_row_ids_hash=d1._hash_strings([runtime.source_train_sample_ids[int(pos)] for pos in np.asarray(positions)]),
        parameter_hash=parameter_hash,
        expert_config_hash=d1._expert_config_hash(runtime),
        status=status,
        error_message=error_message,
    )


def _prototype_manifest_rows(cfg: ComponentUnionConfig, summary: PrototypeSummary) -> list[dict[str, object]]:
    if summary.status != "ok":
        return [
            {
                "experiment_seed": int(summary.experiment_seed),
                "source_center": summary.source_center,
                "class_label": int(summary.class_label),
                "prototype_id": "",
                "selected_k": int(summary.selected_k),
                "selected_k_reason": summary.selected_k_reason,
                "attempted_k_json": json.dumps(list(cfg.prototype_candidate_counts_per_source_class)),
                "candidate_fit_status_json": summary.candidate_fit_status_json,
                "prototype_weight_local": "",
                "assigned_posterior_samples": "",
                "source_class_count": int(summary.source_class_count),
                "summary_path": str(summary.summary_path),
                "summary_hash": summary.summary_hash,
                "status": summary.status,
                "error_message": summary.error_message,
            }
        ]
    rows = []
    weights = d1._normalized_weights(summary.weights)
    for idx, weight in enumerate(weights):
        rows.append(
            {
                "experiment_seed": int(summary.experiment_seed),
                "source_center": summary.source_center,
                "class_label": int(summary.class_label),
                "prototype_id": int(idx),
                "selected_k": int(summary.selected_k),
                "selected_k_reason": summary.selected_k_reason,
                "attempted_k_json": json.dumps(list(cfg.prototype_candidate_counts_per_source_class)),
                "candidate_fit_status_json": summary.candidate_fit_status_json,
                "prototype_weight_local": float(weight),
                "assigned_posterior_samples": summary.assigned_counts[idx] if idx < len(summary.assigned_counts) else "",
                "source_class_count": int(summary.source_class_count),
                "prototype_mean_hash": _hash_array(np.asarray(summary.means, dtype=float)[idx]),
                "prototype_diag_var_hash": _hash_array(np.asarray(summary.diag_vars, dtype=float)[idx]),
                "summary_path": str(summary.summary_path),
                "summary_hash": summary.summary_hash,
                "status": summary.status,
                "error_message": summary.error_message,
            }
        )
    return rows


def _evaluate_gmm_component_union(
    cfg: ComponentUnionConfig,
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
    selection_source: str,
    claim_role: str,
    control_mode: str = "normal",
) -> tuple[dict[str, object], dict[str, object], dict[str, object] | None, dict[str, object] | None, dict[str, object]]:
    sources = tuple(str(source) for source in candidates)
    status, error = d1a._composition_status(sources, summaries, control_mode=control_mode)
    if status != "ok":
        row = _empty_matrix_row(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            candidates=sources,
            prior_method=prior_method,
            source_union_ref=source_union_ref,
            center_balanced_ref=center_balanced_ref,
            real_feature_bacc=real_feature_bacc,
            status=status,
            error_message=error,
            claim_role=claim_role,
        )
        return row, _empty_coverage_row(row), None, None, _paired_generation_row(row, "", "", "ineligible")
    seed = d1._latent_seed(experiment_seed, heldout_center, replicate_seed, prior_method, _plan_hash(weight_plan), control_mode)
    generated, labels, component_counts, source_train_raw, source_hashes = _sample_gmm_component_union_raw(
        cfg,
        per_source_runtime=per_source_runtime,
        sources=sources,
        summaries=summaries,
        weight_plan=weight_plan,
        seed=seed,
        control_mode=control_mode,
    )
    bundle = fit_locked_logistic_classifier(
        generated,
        labels,
        _to_numpy(eval_raw),
        classifier_seed=cfg.classifier_seed,
        expert_id=POOL_COMPONENT_UNION,
        class_weight=cfg.classifier_class_weight,
    )
    result = evaluate_probability_predictions(prior_method, bundle.probabilities, eval_labels)
    generated_hash = _hash_array(generated)
    prediction_hash = _hash_array(bundle.probabilities)
    row = _result_matrix_row(
        cfg,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        candidates=sources,
        prior_method=prior_method,
        summary_kind="gmm_component",
        source_union_ref=source_union_ref,
        center_balanced_ref=center_balanced_ref,
        real_feature_bacc=real_feature_bacc,
        weight_plan=weight_plan,
        bacc=result.bacc,
        macro_f1=result.macro_f1,
        generated_features_hash=generated_hash,
        prediction_hash=prediction_hash,
        selection_source=selection_source,
        claim_role=claim_role,
        status="ok",
        error_message="",
        control_mode=control_mode,
        summaries=summaries,
    )
    coverage = _component_coverage_row(row, component_counts, _expected_component_keys(sources, summaries, control_mode=control_mode))
    weak = _weak_row(row) if _float(row["bacc"]) < 0.75 else None
    nn = _nearest_neighbor_row(row, generated, source_train_raw)
    paired = _paired_generation_row(row, generated_hash, _hash_strings(source_hashes), "ok")
    return row, coverage, weak, nn, paired


def _evaluate_prototype_union(
    cfg: ComponentUnionConfig,
    *,
    per_source_runtime: Mapping[str, RuntimeSource],
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], PrototypeSummary],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    eval_raw: object,
    eval_labels: Sequence[int],
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
    real_feature_bacc: float,
    weight_plan: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object] | None, dict[str, object] | None, dict[str, object]]:
    sources = tuple(str(source) for source in candidates)
    status, error = _prototype_status(sources, summaries)
    if status != "ok":
        row = _empty_matrix_row(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            candidates=sources,
            prior_method=ROW_PROTOTYPE_UNION,
            source_union_ref=source_union_ref,
            center_balanced_ref=center_balanced_ref,
            real_feature_bacc=real_feature_bacc,
            status=status,
            error_message=error,
            claim_role="diagnostic_prototype_union",
            summary_kind="prototype_codebook",
        )
        return row, _empty_coverage_row(row), None, None, _paired_generation_row(row, "", "", "ineligible")
    seed = d1._latent_seed(experiment_seed, heldout_center, replicate_seed, ROW_PROTOTYPE_UNION, _plan_hash(weight_plan))
    generated, labels, component_counts, source_train_raw, source_hashes = _sample_prototype_union_raw(
        cfg,
        per_source_runtime=per_source_runtime,
        sources=sources,
        summaries=summaries,
        weight_plan=weight_plan,
        seed=seed,
    )
    bundle = fit_locked_logistic_classifier(
        generated,
        labels,
        _to_numpy(eval_raw),
        classifier_seed=cfg.classifier_seed,
        expert_id=POOL_COMPONENT_UNION,
        class_weight=cfg.classifier_class_weight,
    )
    result = evaluate_probability_predictions(ROW_PROTOTYPE_UNION, bundle.probabilities, eval_labels)
    generated_hash = _hash_array(generated)
    prediction_hash = _hash_array(bundle.probabilities)
    row = _result_matrix_row(
        cfg,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        candidates=sources,
        prior_method=ROW_PROTOTYPE_UNION,
        summary_kind="prototype_codebook",
        source_union_ref=source_union_ref,
        center_balanced_ref=center_balanced_ref,
        real_feature_bacc=real_feature_bacc,
        weight_plan=weight_plan,
        bacc=result.bacc,
        macro_f1=result.macro_f1,
        generated_features_hash=generated_hash,
        prediction_hash=prediction_hash,
        selection_source=DIAGNOSTIC_SELECTION,
        claim_role="diagnostic_prototype_union",
        status="ok",
        error_message="",
        control_mode="normal",
        summaries=None,
    )
    coverage = _component_coverage_row(row, component_counts, _expected_prototype_keys(sources, summaries))
    weak = _weak_row(row) if _float(row["bacc"]) < 0.75 else None
    nn = _nearest_neighbor_row(row, generated, source_train_raw)
    paired = _paired_generation_row(row, generated_hash, _hash_strings(source_hashes), "ok")
    return row, coverage, weak, nn, paired


def _sample_gmm_component_union_raw(
    cfg: ComponentUnionConfig,
    *,
    per_source_runtime: Mapping[str, RuntimeSource],
    sources: Sequence[str],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    weight_plan: Mapping[str, object],
    seed: int,
    control_mode: str,
) -> tuple[object, tuple[int, ...], dict[int, dict[str, int]], object, list[str]]:
    rng = np.random.default_rng(int(seed))
    chunks = []
    labels: list[int] = []
    source_train_chunks = []
    source_hashes: list[str] = []
    component_counts: dict[int, dict[str, int]] = {0: {}, 1: {}}
    budgets = {str(k): int(v) for k, v in dict(weight_plan["budgets"]).items()}
    for source in sources:
        runtime = per_source_runtime[str(source)].runtime
        source_train_raw = _inverse_to_raw(runtime, runtime.source_train_embeddings)
        source_train_chunks.append(source_train_raw)
        for label_cls in (0, 1):
            summary_cls = 1 - int(label_cls) if control_mode == "class_flip" else int(label_cls)
            summary = summaries[(str(source), int(summary_cls))]
            budget = int(budgets[str(source)])
            z_np, counts = d1a._sample_latents(summary, rng, budget, variance_floor=cfg.variance_floor)
            decoded, _ = _decode_latents(runtime, z_np, [int(label_cls)] * budget)
            raw = _inverse_to_raw(runtime, decoded)
            chunks.append(raw)
            labels.extend([int(label_cls)] * budget)
            for component, count in counts.items():
                component_counts[int(label_cls)][f"{source}:{component}"] = int(count)
        source_hashes.append(_hash_array(chunks[-2]))
        source_hashes.append(_hash_array(chunks[-1]))
    return np.vstack(chunks), tuple(labels), component_counts, np.vstack(source_train_chunks), source_hashes


def _sample_prototype_union_raw(
    cfg: ComponentUnionConfig,
    *,
    per_source_runtime: Mapping[str, RuntimeSource],
    sources: Sequence[str],
    summaries: Mapping[tuple[str, int], PrototypeSummary],
    weight_plan: Mapping[str, object],
    seed: int,
) -> tuple[object, tuple[int, ...], dict[int, dict[str, int]], object, list[str]]:
    rng = np.random.default_rng(int(seed))
    chunks = []
    labels: list[int] = []
    source_train_chunks = []
    source_hashes: list[str] = []
    component_counts: dict[int, dict[str, int]] = {0: {}, 1: {}}
    budgets = {str(k): int(v) for k, v in dict(weight_plan["budgets"]).items()}
    for source in sources:
        runtime = per_source_runtime[str(source)].runtime
        source_train_raw = _inverse_to_raw(runtime, runtime.source_train_embeddings)
        source_train_chunks.append(source_train_raw)
        for label_cls in (0, 1):
            summary = summaries[(str(source), int(label_cls))]
            budget = int(budgets[str(source)])
            z_np, counts = _sample_prototype_latents(summary, rng, budget, variance_floor=cfg.prototype_variance_floor)
            decoded, _ = _decode_latents(runtime, z_np, [int(label_cls)] * budget)
            raw = _inverse_to_raw(runtime, decoded)
            chunks.append(raw)
            labels.extend([int(label_cls)] * budget)
            for component, count in counts.items():
                component_counts[int(label_cls)][f"{source}:{component}"] = int(count)
        source_hashes.append(_hash_array(chunks[-2]))
        source_hashes.append(_hash_array(chunks[-1]))
    return np.vstack(chunks), tuple(labels), component_counts, np.vstack(source_train_chunks), source_hashes


def _sample_prototype_latents(
    summary: PrototypeSummary,
    rng: object,
    n_samples: int,
    *,
    variance_floor: float,
) -> tuple[object, dict[int, int]]:
    weights = d1._normalized_weights(summary.weights)
    components = rng.choice(np.arange(weights.shape[0]), size=int(n_samples), replace=True, p=weights)
    means = np.asarray(summary.means, dtype=np.float32)[components]
    variances = np.asarray(summary.diag_vars, dtype=np.float32)[components]
    eps = rng.normal(size=means.shape).astype(np.float32)
    z_np = means + np.sqrt(np.maximum(variances, float(variance_floor))).astype(np.float32) * eps
    unique, counts = np.unique(components, return_counts=True)
    return z_np, {int(k): int(v) for k, v in zip(unique, counts)}


def _inverse_to_raw(runtime: VariantRuntime, frame_embeddings: object) -> object:
    frame_x = np.asarray(frame_embeddings, dtype=float)
    scaled = runtime.frame.pca.inverse_transform(frame_x)
    return runtime.frame.scaler.inverse_transform(scaled)


def _uniform_source_plan(
    cfg: ComponentUnionConfig,
    sources: Sequence[str],
    rels: Mapping[str, d12.SourceReliability],
    *,
    total: int,
) -> dict[str, object]:
    sources_tuple = tuple(str(source) for source in sources)
    weights = {source: 1.0 / float(len(sources_tuple)) for source in sources_tuple}
    scores = {source: rels[source].reliability_score for source in sources_tuple}
    budgets = {source: int(value) for source, value in zip(sources_tuple, d1._balanced_counts(int(total), len(sources_tuple)))}
    return _with_weight_diagnostics(sources_tuple, weights, budgets, scores, total=total, mode="uniform_source_mass")


def _shrink_source_plan(
    cfg: ComponentUnionConfig,
    sources: Sequence[str],
    rels: Mapping[str, d12.SourceReliability],
    *,
    shrink_lambda: float,
    total: int,
) -> dict[str, object]:
    sources_tuple = tuple(str(source) for source in sources)
    raw = {source: d12._linear_reliability_score(rels[source].raw_bacc, cfg.reliability_floor_score) for source in sources_tuple}
    total_raw = sum(raw.values())
    reliability = {source: raw[source] / total_raw for source in sources_tuple}
    uniform = 1.0 / float(len(sources_tuple))
    weights = {source: (1.0 - float(shrink_lambda)) * uniform + float(shrink_lambda) * reliability[source] for source in sources_tuple}
    scores = {source: raw[source] for source in sources_tuple}
    budgets = d12._weighted_budgets(int(total), sources_tuple, weights, cfg.min_per_source_per_class)
    return _with_weight_diagnostics(sources_tuple, weights, budgets, scores, total=total, mode=f"reliability_shrink_{shrink_lambda:.2f}")


def _shuffled_reliability_plan(
    cfg: ComponentUnionConfig,
    sources: Sequence[str],
    rels: Mapping[str, d12.SourceReliability],
    *,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    shrink_lambda: float = 0.5,
    permutation_id: int | None = None,
    total: int,
) -> dict[str, object]:
    sources_tuple = tuple(str(source) for source in sources)
    seed_parts = [experiment_seed, heldout_center, replicate_seed, "component_union_shuffled_reliability"]
    if permutation_id is not None:
        seed_parts.extend([f"lambda_{float(shrink_lambda):.2f}", int(permutation_id)])
    shuffle_seed = d1._latent_seed(*seed_parts)
    rng = np.random.default_rng(shuffle_seed)
    shuffled = list(sources_tuple)
    rng.shuffle(shuffled)
    raw = {
        source: d12._linear_reliability_score(rels[shuffled[idx]].raw_bacc, cfg.reliability_floor_score)
        for idx, source in enumerate(sources_tuple)
    }
    total_raw = sum(raw.values())
    reliability = {source: raw[source] / total_raw for source in sources_tuple}
    uniform = 1.0 / float(len(sources_tuple))
    weights = {
        source: (1.0 - float(shrink_lambda)) * uniform + float(shrink_lambda) * reliability[source]
        for source in sources_tuple
    }
    budgets = d12._weighted_budgets(int(total), sources_tuple, weights, cfg.min_per_source_per_class)
    plan = _with_weight_diagnostics(
        sources_tuple,
        weights,
        budgets,
        raw,
        total=total,
        mode=f"shuffled_reliability_shrink_{float(shrink_lambda):.2f}",
    )
    plan.update(
        {
            "shrink_lambda": float(shrink_lambda),
            "control_permutation_id": "" if permutation_id is None else int(permutation_id),
            "shuffle_seed": int(shuffle_seed),
            "shuffle_mapping": {source: shuffled[idx] for idx, source in enumerate(sources_tuple)},
        }
    )
    return plan


def _random_source_mass_plan(
    cfg: ComponentUnionConfig,
    sources: Sequence[str],
    *,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    total: int,
) -> dict[str, object]:
    sources_tuple = tuple(str(source) for source in sources)
    rng = np.random.default_rng(d1._latent_seed(experiment_seed, heldout_center, replicate_seed, "component_union_random_source_mass"))
    values = rng.dirichlet(np.ones(len(sources_tuple)))
    weights = {source: float(weight) for source, weight in zip(sources_tuple, values)}
    budgets = d12._weighted_budgets(int(total), sources_tuple, weights, cfg.min_per_source_per_class)
    scores = {source: weights[source] for source in sources_tuple}
    return _with_weight_diagnostics(sources_tuple, weights, budgets, scores, total=total, mode="random_source_mass")


def _with_weight_diagnostics(
    sources: Sequence[str],
    weights: Mapping[str, float],
    budgets: Mapping[str, int],
    scores: Mapping[str, float],
    *,
    total: int,
    mode: str,
) -> dict[str, object]:
    base = d12._with_weight_diagnostics(sources, weights, budgets, scores)
    base.update({"synthetic_per_class_total": int(total), "component_union_weight_mode": str(mode)})
    return base


def _evaluate_source_ablation_diagnostics(
    cfg: ComponentUnionConfig,
    *,
    per_source_runtime: Mapping[str, RuntimeSource],
    all_centers: Sequence[str],
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    rels: Mapping[str, d12.SourceReliability],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    eval_raw: object,
    eval_labels: Sequence[int],
    primary_bacc: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    candidate_set = {str(source) for source in candidates}
    for removed in all_centers:
        if str(removed) not in candidate_set:
            rows.append(
                {
                    "experiment_seed": int(experiment_seed),
                    "heldout_center": str(heldout_center),
                    "replicate_seed": int(replicate_seed),
                    "removed_source_center": str(removed),
                    "remaining_source_centers": "|".join(str(v) for v in candidates),
                    "primary_bacc": primary_bacc,
                    "ablation_bacc": "",
                    "delta_ablation_minus_primary": "",
                    "status": "not_applicable_target_source_excluded",
                }
            )
            continue
        remaining = tuple(source for source in candidates if str(source) != str(removed))
        if len(remaining) < 1:
            continue
        plan = _primary_source_plan(
            cfg,
            remaining,
            {source: rels[source] for source in remaining},
            total=cfg.synthetic_per_class_total,
        )
        plan["component_union_weight_mode"] = f"source_ablation_{plan['component_union_weight_mode']}"
        row, _coverage, _weak, _nn, _paired = _evaluate_gmm_component_union(
            cfg,
            per_source_runtime=per_source_runtime,
            candidates=remaining,
            summaries=summaries,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            eval_raw=eval_raw,
            eval_labels=eval_labels,
            source_union_ref=d1._missing_reference(),
            center_balanced_ref=d1._missing_reference(),
            real_feature_bacc=math.nan,
            weight_plan=plan,
            prior_method=f"source_ablation_minus_{removed}",
            selection_source=DIAGNOSTIC_SELECTION,
            claim_role="diagnostic_source_ablation_not_selection",
        )
        ablation_bacc = _float(row.get("bacc"))
        rows.append(
            {
                "experiment_seed": int(experiment_seed),
                "heldout_center": str(heldout_center),
                "replicate_seed": int(replicate_seed),
                "removed_source_center": str(removed),
                "remaining_source_centers": "|".join(remaining),
                "primary_bacc": primary_bacc,
                "ablation_bacc": ablation_bacc,
                "delta_ablation_minus_primary": ablation_bacc - primary_bacc if math.isfinite(ablation_bacc) and math.isfinite(primary_bacc) else math.nan,
                "status": row.get("status", ""),
            }
        )
    return rows


def _fold_component_manifest_rows(
    cfg: ComponentUnionConfig,
    *,
    experiment_seed: int,
    heldout_center: str,
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    component_details: Mapping[tuple[str, int, int], Mapping[str, object]],
    weight_plan: Mapping[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    source_weights = {str(k): float(v) for k, v in dict(weight_plan["weights"]).items()}
    for cls in (0, 1):
        composed_id = 0
        for source in candidates:
            summary = summaries.get((str(source), int(cls)))
            if summary is None or summary.status != "ok":
                rows.append(
                    {
                        "row_scope": "fold_union_prior",
                        "experiment_seed": int(experiment_seed),
                        "heldout_center": str(heldout_center),
                        "source_center": str(source),
                        "class_label": int(cls),
                        "source_component_id": "",
                        "composed_component_id": composed_id,
                        "summary_status": "missing_summary" if summary is None else summary.status,
                        "summary_error_message": f"missing_summary_source_{source}_class_{cls}" if summary is None else summary.error_message,
                    }
                )
                continue
            weights = d1._normalized_weights(summary.weights)
            for idx, local_weight in enumerate(weights):
                detail = dict(component_details.get((str(source), int(cls), int(idx)), {}))
                rows.append(
                    {
                        **detail,
                        "row_scope": "fold_union_prior",
                        "experiment_seed": int(experiment_seed),
                        "heldout_center": str(heldout_center),
                        "source_center": str(source),
                        "class_label": int(cls),
                        "source_component_id": int(idx),
                        "composed_component_id": int(composed_id),
                        "component_weight_local": float(local_weight),
                        "component_weight_after_source_normalization": float(source_weights[str(source)] * float(local_weight)),
                        "summary_hash": summary.summary_hash,
                        "summary_status": summary.status,
                        "summary_error_message": summary.error_message,
                    }
                )
                composed_id += 1
    return rows


def _result_matrix_row(
    cfg: ComponentUnionConfig,
    *,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    prior_method: str,
    summary_kind: str,
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
    real_feature_bacc: float,
    weight_plan: Mapping[str, object],
    bacc: float,
    macro_f1: float,
    generated_features_hash: str,
    prediction_hash: str,
    selection_source: str,
    claim_role: str,
    status: str,
    error_message: str,
    control_mode: str,
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary] | None,
) -> dict[str, object]:
    stats = _summary_stats(cfg, summaries, candidates, control_mode=control_mode) if summaries is not None else _prototype_like_stats()
    total = int(weight_plan.get("synthetic_per_class_total", cfg.synthetic_per_class_total))
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "expert_id": POOL_COMPONENT_UNION,
        "expert_pool_type": POOL_COMPONENT_UNION,
        "variant_id": PRIMARY_VARIANT,
        "prior_method": str(prior_method),
        "summary_kind": str(summary_kind),
        "gmm_components": stats["min_composed_components_per_class"],
        "effective_gmm_components": stats["min_composed_components_per_class"],
        "max_local_gmm_components_per_source_class": cfg.max_local_gmm_components_per_source_class,
        "composed_components_per_class_actual": stats["composed_components_per_class_actual"],
        "source_weighting": weight_plan.get("component_union_weight_mode", cfg.source_weighting),
        "pooling_rule": cfg.primary_pooling,
        "replicate_seed": int(replicate_seed),
        "latent_sample_seed": d1._latent_seed(experiment_seed, heldout_center, replicate_seed, prior_method, _plan_hash(weight_plan), control_mode),
        "included_source_centers": "|".join(str(v) for v in candidates),
        "num_included_sources": len(candidates),
        "synthetic_per_class_total": total,
        "synthetic_per_class_per_source_json": json.dumps(dict(weight_plan["budgets"]), sort_keys=True),
        "bacc": bacc,
        "macro_f1": macro_f1,
        "source_union_k16_bacc": source_union_ref.bacc,
        "center_balanced_k16_bacc": center_balanced_ref.bacc,
        "real_feature_dense_bacc": real_feature_bacc,
        "retention_vs_source_union_k16": d1._retention(bacc, source_union_ref.bacc),
        "retention_vs_center_balanced_k16": d1._retention(bacc, center_balanced_ref.bacc),
        "oracle_gap_vs_source_union_k16": source_union_ref.bacc - bacc if math.isfinite(source_union_ref.bacc) and math.isfinite(bacc) else math.nan,
        "oracle_gap_vs_real_feature_dense": real_feature_bacc - bacc if math.isfinite(real_feature_bacc) and math.isfinite(bacc) else math.nan,
        "delta_vs_real_source_embedding_dense_reference": bacc - real_feature_bacc if math.isfinite(real_feature_bacc) else math.nan,
        "negative_control_gap": math.nan,
        "selected_k_histogram_json": stats["selected_k_histogram_json"],
        "min_selected_k": stats["min_selected_k"],
        "mean_selected_k": stats["mean_selected_k"],
        "pct_source_class_summaries_not_k4": stats["pct_source_class_summaries_not_k4"],
        "adaptive_k_intervention_active": stats["adaptive_k_intervention_active"],
        "generated_features_hash": generated_features_hash,
        "prediction_hash": prediction_hash,
        "composed_prior_hash": _summary_set_hash(summaries, candidates, control_mode=control_mode) if summaries is not None else "",
        "summary_set_hash": _summary_set_hash(summaries, candidates, control_mode=control_mode) if summaries is not None else "",
        "source_weight_json": json.dumps(dict(weight_plan["weights"]), sort_keys=True),
        "source_budget_json": json.dumps(dict(weight_plan["budgets"]), sort_keys=True),
        "source_weight_entropy": weight_plan["weight_entropy"],
        "effective_num_sources": weight_plan["effective_num_sources"],
        "l1_distance_from_uniform": weight_plan["l1_distance_from_uniform"],
        "dominant_source": weight_plan["dominant_source"],
        "dominant_source_weight": weight_plan["dominant_source_weight"],
        "shrink_lambda": weight_plan.get("shrink_lambda", ""),
        "control_permutation_id": weight_plan.get("control_permutation_id", ""),
        "shuffle_seed": weight_plan.get("shuffle_seed", ""),
        "shuffle_mapping_json": json.dumps(dict(weight_plan.get("shuffle_mapping", {})), sort_keys=True),
        "selection_source": selection_source,
        "status": status,
        "error_message": error_message,
        "claim_role": claim_role,
    }


def _empty_matrix_row(
    cfg: ComponentUnionConfig,
    *,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    prior_method: str,
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
    real_feature_bacc: float,
    status: str,
    error_message: str,
    claim_role: str,
    summary_kind: str = "gmm_component",
) -> dict[str, object]:
    rels = {str(source): d12.SourceReliability(int(experiment_seed), int(replicate_seed), str(source), math.nan, math.nan, cfg.reliability_floor_score, "empty", str(error_message), 0, "", "") for source in candidates}
    plan = _uniform_source_plan(cfg, candidates, rels, total=cfg.synthetic_per_class_total)
    row = _result_matrix_row(
        cfg,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        candidates=candidates,
        prior_method=prior_method,
        summary_kind=summary_kind,
        source_union_ref=source_union_ref,
        center_balanced_ref=center_balanced_ref,
        real_feature_bacc=real_feature_bacc,
        weight_plan=plan,
        bacc=math.nan,
        macro_f1=math.nan,
        generated_features_hash="",
        prediction_hash="",
        selection_source=DIAGNOSTIC_SELECTION,
        claim_role=claim_role,
        status=status,
        error_message=error_message,
        control_mode="normal",
        summaries={},
    )
    row["bacc"] = ""
    row["macro_f1"] = ""
    return row


def _reference_matrix_row(
    cfg: ComponentUnionConfig,
    *,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    prior_method: str,
    reference: d1.ReferenceValue,
) -> dict[str, object]:
    row = _empty_matrix_row(
        cfg,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        candidates=candidates,
        prior_method=prior_method,
        source_union_ref=reference if prior_method == ROW_SOURCE_UNION_K16_REFERENCE else d1._missing_reference(),
        center_balanced_ref=reference if prior_method == ROW_CENTER_BALANCED_K16_REFERENCE else d1._missing_reference(),
        real_feature_bacc=math.nan,
        status=reference.status,
        error_message=reference.error_message,
        claim_role="centralized_reference_upper_bound_not_decentralized",
    )
    row.update(
        {
            "expert_id": POOL_SOURCE_UNION,
            "expert_pool_type": POOL_SOURCE_UNION,
            "variant_id": UNION_VARIANT,
            "bacc": reference.bacc if reference.status == "ok" else "",
            "macro_f1": reference.macro_f1 if reference.status == "ok" else "",
        }
    )
    return row


def _ineligible_rows(
    cfg: ComponentUnionConfig,
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
    rows = [
        _empty_matrix_row(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            candidates=candidates,
            prior_method=method,
            source_union_ref=source_union_ref,
            center_balanced_ref=center_balanced_ref,
            real_feature_bacc=math.nan,
            status=status,
            error_message=error_message,
            claim_role=role,
            summary_kind=summary_kind,
        )
        for method, role, summary_kind in (
            (
                PRIMARY_COMPONENT_UNION_METHOD,
                _claim_role_for_method(cfg, PRIMARY_COMPONENT_UNION_METHOD, "diagnostic_uniform_component_union"),
                "gmm_component",
            ),
            (
                ROW_COMPONENT_UNION_SHRINK025,
                _claim_role_for_method(cfg, ROW_COMPONENT_UNION_SHRINK025, "diagnostic_reliability_shrink025_component_union"),
                "gmm_component",
            ),
            (ROW_COMPONENT_UNION_SHRINK050, "diagnostic_reliability_shrink050_component_union", "gmm_component"),
            (ROW_PROTOTYPE_UNION, "diagnostic_prototype_union", "prototype_codebook"),
            (ROW_SHUFFLED_SUMMARY_CONTROL, "negative_control", "gmm_component"),
            (ROW_SHUFFLED_LABEL_CONTROL, "negative_control", "gmm_component"),
            (ROW_SHUFFLED_RELIABILITY_CONTROL, "negative_control", "gmm_component"),
            (ROW_RANDOM_SOURCE_MASS_CONTROL, "negative_control", "gmm_component"),
        )
    ]
    rows.append(_reference_matrix_row(cfg, experiment_seed=experiment_seed, heldout_center=heldout_center, replicate_seed=replicate_seed, candidates=candidates, prior_method=ROW_SOURCE_UNION_K16_REFERENCE, reference=source_union_ref))
    rows.append(_reference_matrix_row(cfg, experiment_seed=experiment_seed, heldout_center=heldout_center, replicate_seed=replicate_seed, candidates=candidates, prior_method=ROW_CENTER_BALANCED_K16_REFERENCE, reference=center_balanced_ref))
    return rows


def _append_single_source_references(
    cfg: ComponentUnionConfig,
    matrix_rows: list[dict[str, object]],
    late_rows: Sequence[Mapping[str, object]],
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
    source_rows = [
        row for row in late_rows
        if row.get("experiment_seed") == int(experiment_seed)
        and row.get("heldout_center") == str(heldout_center)
        and row.get("replicate_seed") == int(replicate_seed)
        and row.get("prior_method") == d12.ROW_EQUAL_REFERENCE
        and row.get("pooling_rule") == "single_source"
        and str(row.get("expert_id")) in {str(source) for source in candidates}
        and row.get("status") == "ok"
    ]
    baccs = [_float(row["bacc"]) for row in source_rows]
    macros = [_float(row["macro_f1"]) for row in source_rows]
    mean_single = nanmean(baccs)
    oracle_single = max(baccs) if baccs else math.nan
    for method, bacc, macro_f1, role in (
        (ROW_SINGLE_MEAN, mean_single, nanmean(macros), "single_source_mean_reference"),
        (ROW_SINGLE_ORACLE, oracle_single, max(macros) if macros else math.nan, "diagnostic_only_oracle_reference"),
    ):
        row = _empty_matrix_row(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            candidates=candidates,
            prior_method=method,
            source_union_ref=source_union_ref,
            center_balanced_ref=center_balanced_ref,
            real_feature_bacc=real_feature_bacc,
            status="ok",
            error_message="",
            claim_role=role,
        )
        row.update(
            {
                "expert_id": "single_source_reference",
                "expert_pool_type": POOL_PER_SOURCE,
                "bacc": bacc,
                "macro_f1": macro_f1,
                "real_feature_dense_bacc": real_feature_bacc,
                "retention_vs_source_union_k16": d1._retention(bacc, source_union_ref.bacc),
                "retention_vs_center_balanced_k16": d1._retention(bacc, center_balanced_ref.bacc),
                "delta_vs_real_source_embedding_dense_reference": bacc - real_feature_bacc if math.isfinite(bacc) and math.isfinite(real_feature_bacc) else math.nan,
            }
        )
        matrix_rows.append(row)


def _component_coverage_row(
    row: Mapping[str, object],
    component_counts: Mapping[int, Mapping[str, int]],
    expected: set[str],
) -> dict[str, object]:
    sampled = {f"{cls}:{component}" for cls, counts in component_counts.items() for component in counts}
    counts = [int(v) for values in component_counts.values() for v in values.values()]
    total = float(sum(counts))
    fractions = [value / total for value in counts] if total else []
    entropy = -sum(p * math.log(p) for p in fractions if p > 0.0)
    unsampled = sorted(expected.difference(sampled))
    source_totals: dict[str, int] = {}
    for values in component_counts.values():
        for key, count in values.items():
            source = str(key).split(":", 1)[0]
            source_totals[source] = source_totals.get(source, 0) + int(count)
    return {
        "experiment_seed": row["experiment_seed"],
        "heldout_center": row["heldout_center"],
        "expert_id": row["expert_id"],
        "expert_pool_type": row["expert_pool_type"],
        "variant_id": row["variant_id"],
        "prior_method": row["prior_method"],
        "summary_kind": row.get("summary_kind", ""),
        "replicate_seed": row["replicate_seed"],
        "generated_component_counts_json": json.dumps({str(cls): dict(values) for cls, values in component_counts.items()}, sort_keys=True),
        "generated_source_counts_json": json.dumps(source_totals, sort_keys=True),
        "num_active_components_unsampled": len(unsampled),
        "unsampled_active_components": "|".join(unsampled),
        "min_generated_samples_per_active_component": 0 if unsampled else (min(counts) if counts else 0),
        "component_weight_entropy": entropy,
        "component_mass_covered_by_generated_samples": 1.0 - (len(unsampled) / float(len(expected))) if expected else math.nan,
        "latent_component_undersampled": bool(unsampled),
        "status": row.get("status", ""),
    }


def _empty_coverage_row(row: Mapping[str, object]) -> dict[str, object]:
    return _component_coverage_row(row, {}, set())


def _weak_row(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "experiment_seed": row["experiment_seed"],
        "heldout_center": row["heldout_center"],
        "expert_id": row["expert_id"],
        "expert_pool_type": row["expert_pool_type"],
        "variant_id": row["variant_id"],
        "prior_method": row["prior_method"],
        "replicate_seed": row["replicate_seed"],
        "bacc": row.get("bacc", ""),
        "macro_f1": row.get("macro_f1", ""),
        "weak_cell_warning": bool(_float(row.get("bacc")) < 0.75),
        "status": row.get("status", ""),
    }


def _paired_generation_row(row: Mapping[str, object], generated_hash: str, source_hash: str, status: str) -> dict[str, object]:
    invariant_key = _hash_strings(
        [
            str(row.get("experiment_seed", "")),
            str(row.get("heldout_center", "")),
            str(row.get("replicate_seed", "")),
            str(row.get("included_source_centers", "")),
            str(row.get("prior_method", "")),
            str(row.get("synthetic_per_class_total", "")),
            str(row.get("source_budget_json", "")),
            str(row.get("summary_set_hash", "")),
        ]
    )
    return {
        "experiment_seed": row.get("experiment_seed", ""),
        "heldout_center": row.get("heldout_center", ""),
        "replicate_seed": row.get("replicate_seed", ""),
        "prior_method": row.get("prior_method", ""),
        "summary_kind": row.get("summary_kind", ""),
        "included_source_centers": row.get("included_source_centers", ""),
        "synthetic_per_class_total": row.get("synthetic_per_class_total", ""),
        "source_budget_json": row.get("source_budget_json", ""),
        "summary_set_hash": row.get("summary_set_hash", ""),
        "paired_generation_invariant_key": invariant_key,
        "generated_features_hash": generated_hash,
        "source_generation_hash": source_hash,
        "status": status,
    }


def _expected_component_keys(
    sources: Sequence[str],
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    *,
    control_mode: str,
) -> set[str]:
    expected: set[str] = set()
    for cls in (0, 1):
        summary_cls = 1 - cls if control_mode == "class_flip" else cls
        for source in sources:
            summary = summaries.get((str(source), int(summary_cls)))
            if summary is None or summary.status != "ok":
                continue
            for component in range(int(summary.selected_k)):
                expected.add(f"{cls}:{source}:{component}")
    return expected


def _expected_prototype_keys(
    sources: Sequence[str],
    summaries: Mapping[tuple[str, int], PrototypeSummary],
) -> set[str]:
    expected: set[str] = set()
    for cls in (0, 1):
        for source in sources:
            summary = summaries.get((str(source), int(cls)))
            if summary is None or summary.status != "ok":
                continue
            for component in range(int(summary.selected_k)):
                expected.add(f"{cls}:{source}:{component}")
    return expected


def _prototype_status(sources: Sequence[str], summaries: Mapping[tuple[str, int], PrototypeSummary]) -> tuple[str, str]:
    errors: list[str] = []
    for source in sources:
        for cls in (0, 1):
            summary = summaries.get((str(source), int(cls)))
            if summary is None:
                errors.append(f"missing_prototype_source_{source}_class_{cls}")
                continue
            if summary.status != "ok":
                errors.append(f"source_{source}_class_{cls}:{summary.error_message or summary.status}")
            if int(summary.selected_k) < 1:
                errors.append(f"source_{source}_class_{cls}_selected_k<1")
    if errors:
        return "ineligible_component_fit", "|".join(sorted(set(errors)))
    return "ok", ""


def _summary_stats(
    cfg: ComponentUnionConfig,
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary],
    candidates: Sequence[str],
    *,
    control_mode: str,
) -> dict[str, object]:
    stats = d1a._composition_stats(cfg, summaries, candidates, control_mode=control_mode)
    return dict(stats)


def _prototype_like_stats() -> dict[str, object]:
    return {
        "min_composed_components_per_class": math.nan,
        "composed_components_per_class_actual": "{}",
        "selected_k_histogram_json": "{}",
        "min_selected_k": math.nan,
        "mean_selected_k": math.nan,
        "pct_source_class_summaries_not_k4": math.nan,
        "adaptive_k_intervention_active": False,
    }


def _summary_set_hash(
    summaries: Mapping[tuple[str, int], d1a.AdaptiveSourceLocalSummary] | None,
    candidates: Sequence[str],
    *,
    control_mode: str,
) -> str:
    if summaries is None:
        return ""
    return d1a._summary_set_hash(summaries, candidates, control_mode=control_mode)


def _plan_hash(plan: Mapping[str, object]) -> str:
    return _hash_strings(
        [
            json.dumps(dict(plan["weights"]), sort_keys=True),
            json.dumps(dict(plan["budgets"]), sort_keys=True),
            str(plan.get("synthetic_per_class_total", "")),
            str(plan.get("component_union_weight_mode", "")),
            str(plan.get("shrink_lambda", "")),
            str(plan.get("control_permutation_id", "")),
            json.dumps(dict(plan.get("shuffle_mapping", {})), sort_keys=True),
        ]
    )


def _source_weight_manifest_rows(
    experiment_seed: int,
    replicate_seed: int,
    heldout_center: str,
    method: str,
    plan: Mapping[str, object],
    rels: Mapping[str, d12.SourceReliability],
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
                "prior_method": str(method),
                "source_center": source_id,
                "raw_reliability_bacc": rel.raw_bacc,
                "reliability_score": plan["scores"][source_id],
                "normalized_source_weight": plan["weights"][source_id],
                "synthetic_per_class_budget": plan["budgets"][source_id],
                "weight_mode": plan.get("component_union_weight_mode", ""),
                "weight_entropy": plan["weight_entropy"],
                "effective_num_sources": plan["effective_num_sources"],
                "l1_distance_from_uniform": plan["l1_distance_from_uniform"],
                "max_weight": plan["max_weight"],
                "min_weight": plan["min_weight"],
                "dominant_source": plan["dominant_source"],
                "dominant_source_weight": plan["dominant_source_weight"],
                "shrink_lambda": plan.get("shrink_lambda", ""),
                "control_permutation_id": plan.get("control_permutation_id", ""),
                "shuffle_seed": plan.get("shuffle_seed", ""),
                "shuffle_mapping_json": json.dumps(dict(plan.get("shuffle_mapping", {})), sort_keys=True),
            }
        )
    return rows


def _component_extend_row(row: Mapping[str, object], *, source_weighting: str | None = None) -> dict[str, object]:
    out = d12._extend_row(row, source_weighting=source_weighting)
    out.setdefault("summary_kind", "")
    out.setdefault("source_weight_json", out.get("reliability_weight_json", "{}"))
    out.setdefault("source_budget_json", out.get("reliability_budget_per_class_json", "{}"))
    out.setdefault("source_weight_entropy", out.get("reliability_weight_entropy", math.nan))
    return out


def _component_extend_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [_component_extend_row(row) for row in rows]


def _rename_component_rows(rows: Sequence[Mapping[str, object]], old: str, new: str) -> list[dict[str, object]]:
    out = []
    for row in rows:
        copied = _component_extend_row(row)
        if copied.get("prior_method") == old:
            copied["prior_method"] = new
        out.append(copied)
    return out


def _populate_negative_control_gaps(rows: list[dict[str, object]]) -> None:
    controls: dict[tuple[str, str, str], float] = {}
    control_methods = {
        ROW_SHUFFLED_SUMMARY_CONTROL,
        ROW_SHUFFLED_LABEL_CONTROL,
        ROW_SHUFFLED_RELIABILITY_CONTROL,
        ROW_RANDOM_SOURCE_MASS_CONTROL,
    }
    for row in rows:
        if row.get("prior_method") not in control_methods:
            continue
        key = (str(row.get("experiment_seed")), str(row.get("heldout_center")), str(row.get("replicate_seed")))
        value = _float(row.get("bacc"))
        if math.isfinite(value):
            controls[key] = max(controls.get(key, -math.inf), value)
    for row in rows:
        if row.get("prior_method") != PRIMARY_COMPONENT_UNION_METHOD:
            continue
        key = (str(row.get("experiment_seed")), str(row.get("heldout_center")), str(row.get("replicate_seed")))
        control = controls.get(key, math.nan)
        value = _float(row.get("bacc"))
        if math.isfinite(value) and math.isfinite(control):
            row["negative_control_gap"] = value - control


def _decision(
    rows: Sequence[Mapping[str, object]],
    cfg: ComponentUnionConfig,
    *,
    leakage_status: str,
    source_ablation_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    primary_all = _rows_for(rows, PRIMARY_COMPONENT_UNION_METHOD, include_non_ok=True)
    primary = _rows_for(rows, PRIMARY_COMPONENT_UNION_METHOD)
    equal = _rows_for(rows, ROW_EQUAL_ALL4_REFERENCE)
    rel_all4 = _rows_for(rows, ROW_RELIABILITY_ALL4_REFERENCE)
    prototype = _rows_for(rows, ROW_PROTOTYPE_UNION)
    source_union = _rows_for(rows, ROW_SOURCE_UNION_K16_REFERENCE)
    center_balanced = _rows_for(rows, ROW_CENTER_BALANCED_K16_REFERENCE)
    real_feature = _rows_for(rows, ROW_REAL_FEATURE_DENSE_REFERENCE)
    controls_by_method = {
        method: _method_stats(_rows_for(rows, method))
        for method in (
            ROW_SHUFFLED_SUMMARY_CONTROL,
            ROW_SHUFFLED_LABEL_CONTROL,
            ROW_SHUFFLED_RELIABILITY_CONTROL,
            ROW_RANDOM_SOURCE_MASS_CONTROL,
        )
    }
    strongest_control_method, strongest_control_stats = max(
        controls_by_method.items(),
        key=lambda item: (_float(item[1]["center_equal_mean_bacc"]), item[0]),
    )
    stats = _method_stats(primary)
    equal_stats = _method_stats(equal)
    rel_stats = _method_stats(rel_all4)
    prototype_stats = _method_stats(prototype)
    source_union_stats = _method_stats(source_union)
    center_balanced_stats = _method_stats(center_balanced)
    real_stats = _method_stats(real_feature)
    ablation = _source_ablation_stats(source_ablation_rows)
    fit_ineligible = any(row.get("status") == "ineligible_component_fit" for row in primary_all)
    primary_bacc = _float(stats["center_equal_mean_bacc"])
    delta_vs_d12 = primary_bacc - _float(rel_stats["center_equal_mean_bacc"])
    delta_vs_equal = primary_bacc - _float(equal_stats["center_equal_mean_bacc"])
    retention_source_union = d1._retention(primary_bacc, _float(source_union_stats["center_equal_mean_bacc"]))
    retention_center_balanced = d1._retention(primary_bacc, _float(center_balanced_stats["center_equal_mean_bacc"]))
    negative_control_gap = primary_bacc - _float(strongest_control_stats["center_equal_mean_bacc"])
    strong_success = (
        leakage_status == "PASS"
        and not fit_ineligible
        and int(stats["n_heldout_centers"]) >= len(cfg.heldout_centers)
        and delta_vs_d12 >= 0.015
        and delta_vs_equal >= 0.010
        and _float(stats["min_center_bacc"]) >= 0.80
        and _float(stats["seed_std_bacc"]) <= 0.04
        and retention_source_union >= 0.97
        and negative_control_gap > 0.0
    )
    useful_success = (
        leakage_status == "PASS"
        and not fit_ineligible
        and int(stats["n_heldout_centers"]) >= len(cfg.heldout_centers)
        and retention_source_union >= 0.94
        and delta_vs_d12 > 0.0
        and delta_vs_equal > 0.0
        and _float(stats["min_center_bacc"]) >= 0.80
        and _float(stats["seed_std_bacc"]) <= 0.04
        and not bool(ablation["source_ablation_dominance_flag"])
    )
    verdict = "COMPONENT_UNION_FAIL"
    if leakage_status != "PASS":
        verdict = "PROTOCOL_FAIL"
    elif fit_ineligible:
        verdict = "INELIGIBLE"
    elif strong_success:
        verdict = "COMPONENT_UNION_STRONG_SUCCESS"
    elif useful_success:
        verdict = "COMPONENT_UNION_USEFUL_THESIS_SUCCESS"
    elif int(stats["n_heldout_centers"]) < len(cfg.heldout_centers):
        verdict = "TARGET_EVAL_INSUFFICIENT"

    flags = []
    if fit_ineligible:
        flags.append("INELIGIBLE_COMPONENT_FIT")
    if math.isfinite(delta_vs_d12) and delta_vs_d12 < 0.015:
        flags.append("DELTA_VS_D1_2_BELOW_0P015")
    if math.isfinite(delta_vs_equal) and delta_vs_equal < 0.010:
        flags.append("DELTA_VS_EQUAL_ALL4_BELOW_0P010")
    if math.isfinite(retention_source_union) and retention_source_union < 0.97:
        flags.append("SOURCE_UNION_RETENTION_BELOW_STRONG_0P97")
    if math.isfinite(retention_source_union) and retention_source_union < 0.94:
        flags.append("SOURCE_UNION_RETENTION_BELOW_USEFUL_0P94")
    if math.isfinite(negative_control_gap) and negative_control_gap <= 0.0:
        flags.append("NEGATIVE_CONTROL_COMPETITIVE")
    if bool(ablation["source_ablation_dominance_flag"]):
        flags.append("SOURCE_ABLATION_DOMINANCE")

    return {
        "primary_verdict": verdict,
        "diagnostic_flags": "|".join(flags),
        "leakage_status": leakage_status,
        "primary_method": PRIMARY_COMPONENT_UNION_METHOD,
        "center_equal_mean_bacc": stats["center_equal_mean_bacc"],
        "seed_cell_mean_bacc": stats["seed_cell_mean_bacc"],
        "center_equal_macro_f1": stats["center_equal_macro_f1"],
        "min_center_bacc": stats["min_center_bacc"],
        "seed_std_bacc": stats["seed_std_bacc"],
        "delta_vs_d1_2_reliability_all4": delta_vs_d12,
        "delta_vs_equal_all4": delta_vs_equal,
        "retention_vs_source_union_k16": retention_source_union,
        "retention_vs_center_balanced_k16": retention_center_balanced,
        "delta_vs_real_source_embedding_dense_reference": primary_bacc - _float(real_stats["center_equal_mean_bacc"]),
        "negative_control_gap": negative_control_gap,
        "strongest_negative_control_method": strongest_control_method,
        "strongest_negative_control_center_equal_mean_bacc": strongest_control_stats["center_equal_mean_bacc"],
        "d1_2_reliability_all4_center_equal_mean_bacc": rel_stats["center_equal_mean_bacc"],
        "equal_all4_center_equal_mean_bacc": equal_stats["center_equal_mean_bacc"],
        "prototype_union_center_equal_mean_bacc": prototype_stats["center_equal_mean_bacc"],
        "source_union_k16_reference_center_equal_mean_bacc": source_union_stats["center_equal_mean_bacc"],
        "center_balanced_k16_reference_center_equal_mean_bacc": center_balanced_stats["center_equal_mean_bacc"],
        "real_feature_dense_reference_center_equal_mean_bacc": real_stats["center_equal_mean_bacc"],
        "eligible_heldout_centers": stats["n_heldout_centers"],
        "eligible_seed_center_cells": stats["n_decision_cells"],
        **ablation,
        **stats,
    }


def _method_stats(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    grouped = _replicate_averaged(rows)
    by_seed: dict[str, list[Mapping[str, object]]] = {}
    by_center: dict[str, list[Mapping[str, object]]] = {}
    for row in grouped:
        by_seed.setdefault(str(row["experiment_seed"]), []).append(row)
        by_center.setdefault(str(row["heldout_center"]), []).append(row)
    seed_means = [d1._mean_field(values, "bacc") for values in by_seed.values()]
    center_bacc = {center: d1._mean_field(values, "bacc") for center, values in sorted(by_center.items())}
    return {
        "n_raw_rows": len(rows),
        "n_decision_cells": len(grouped),
        "n_experiment_seeds": len(by_seed),
        "n_heldout_centers": len(by_center),
        "min_eligible_seeds_per_center": min((len({str(row["experiment_seed"]) for row in values}) for values in by_center.values()), default=0),
        "center_equal_mean_bacc": nanmean(list(center_bacc.values())) if center_bacc else math.nan,
        "seed_cell_mean_bacc": d1._mean_field(grouped, "bacc"),
        "center_equal_macro_f1": _center_equal_mean(grouped, "macro_f1"),
        "seed_std_bacc": d1._std(seed_means),
        "min_center_bacc": min(center_bacc.values()) if center_bacc else math.nan,
        "min_cell_bacc": d1._min_field(grouped, "bacc"),
        "per_center_bacc": json.dumps(center_bacc, sort_keys=True),
        "per_seed_bacc": json.dumps({seed: d1._mean_field(values, "bacc") for seed, values in sorted(by_seed.items())}, sort_keys=True),
    }


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
        for (seed, center), subset in groups.items()
    ]


def _center_equal_mean(rows: Sequence[Mapping[str, object]], field: str) -> float:
    by_center: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        by_center.setdefault(str(row["heldout_center"]), []).append(row)
    return nanmean([d1._mean_field(values, field) for values in by_center.values()])


def _rows_for(rows: Sequence[Mapping[str, object]], method: str, *, include_non_ok: bool = False) -> list[Mapping[str, object]]:
    return [
        row for row in rows
        if row.get("prior_method") == method
        and (include_non_ok or row.get("status") == "ok")
    ]


def _source_ablation_stats(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    deltas = [_float(row.get("delta_ablation_minus_primary")) for row in rows if row.get("status") == "ok"]
    finite = [value for value in deltas if math.isfinite(value)]
    max_gain = max(finite, default=math.nan)
    max_drop = abs(min(finite, default=math.nan)) if finite else math.nan
    dominance = (math.isfinite(max_drop) and max_drop > 0.08) or (math.isfinite(max_gain) and max_gain > 0.03)
    return {
        "max_source_ablation_drop_bacc": max_drop,
        "max_source_ablation_gain_bacc": max_gain,
        "mean_source_ablation_delta_bacc": nanmean(finite) if finite else math.nan,
        "source_ablation_dominance_flag": bool(dominance),
    }


def _write_artifacts(
    root: Path,
    cfg: ComponentUnionConfig,
    *,
    matrix_rows: Sequence[Mapping[str, object]],
    gap_rows: Sequence[Mapping[str, object]],
    component_manifest_rows: Sequence[Mapping[str, object]],
    source_summary_rows: Sequence[Mapping[str, object]],
    prototype_manifest_rows: Sequence[Mapping[str, object]],
    component_coverage_rows: Sequence[Mapping[str, object]],
    source_weight_rows: Sequence[Mapping[str, object]],
    reliability_rows: Sequence[Mapping[str, object]],
    source_ablation_rows: Sequence[Mapping[str, object]],
    paired_generation_rows: Sequence[Mapping[str, object]],
    weak_rows: Sequence[Mapping[str, object]],
    nn_rows: Sequence[Mapping[str, object]],
    real_feature_rows: Sequence[Mapping[str, object]],
    late_rows: Sequence[Mapping[str, object]],
    model_manifest_rows: Sequence[Mapping[str, object]],
    decision: Mapping[str, object],
    leakage_status: str,
    protocol_violations: Sequence[str],
    target_expert_excluded: bool,
) -> None:
    write_csv_rows(root / "tables" / "component_union_downstream_matrix.csv", matrix_rows)
    write_csv_rows(root / "tables" / "component_union_gap_summary.csv", gap_rows)
    write_csv_rows(root / "tables" / "component_union_summary.csv", [dict(decision)])
    write_csv_rows(root / "tables" / "component_manifest.csv", component_manifest_rows)
    write_csv_rows(root / "tables" / "source_summary_diagnostics.csv", source_summary_rows)
    write_csv_rows(root / "tables" / "prototype_manifest.csv", prototype_manifest_rows)
    write_csv_rows(root / "tables" / "component_coverage_audit.csv", component_coverage_rows)
    write_csv_rows(root / "tables" / "source_weight_manifest.csv", source_weight_rows)
    write_csv_rows(root / "tables" / "source_reliability_manifest.csv", reliability_rows)
    write_csv_rows(root / "tables" / "source_ablation_audit.csv", source_ablation_rows)
    write_csv_rows(root / "tables" / "paired_generation_audit.csv", paired_generation_rows)
    write_csv_rows(root / "tables" / "weak_source_audit.csv", weak_rows)
    write_csv_rows(root / "tables" / "nearest_neighbor_memorization_audit.csv", nn_rows)
    write_csv_rows(root / "tables" / "real_feature_reference_matrix.csv", real_feature_rows)
    write_csv_rows(root / "tables" / "late_aggregation_reference_matrix.csv", late_rows)
    write_csv_rows(root / "tables" / "negative_control_summary.csv", [_negative_control_summary(decision)])
    write_csv_rows(root / "manifests" / "decentralized_component_union_prior_model_manifest.csv", model_manifest_rows)
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
            "schema_version": "cvae_rebuild_decentralized_component_union_prior_protocol_manifest_v1",
            "experiment_name": cfg.name,
            "experiment_type": "decentralized_component_level_generative_expert_composition",
            "primary_variant": cfg.primary_variant,
            "primary_method": cfg.primary_method,
            "target_support_labels_for_selection": False,
            "target_eval_labels_for_scoring_only": True,
            "target_expert_excluded": target_expert_excluded,
            "fixed_all_source_inclusion": True,
            "tests_target_conditioned_routing": False,
            "tests_composition_granularity": True,
            "exported_source_summaries_are_target_agnostic": True,
            "raw_source_embedding_pooling_for_prior_fit": False,
            "pooled_classifier_frame": "raw_embedding_frame_after_source_inverse_pca",
            "source_union_references_diagnostic_only": True,
            "source_ablation_diagnostic_only": True,
            "oracle_rows_diagnostic_only": True,
            "protocol_wording": PROTOCOL_WORDING,
            "claim_boundary": (
                "component-level prior composition audit only; no target-specific compatibility routing claim, "
                "no support-NELBO downstream claim, and no formal privacy claim"
            ),
        },
    )
    _write_decision_summary(root, decision, leakage_status=leakage_status)
    write_json(root / "run_config_resolved.yaml", _resolved_config(cfg))


def _negative_control_summary(decision: Mapping[str, object]) -> dict[str, object]:
    return {
        "primary_method": PRIMARY_COMPONENT_UNION_METHOD,
        "control_methods": "|".join(
            [
                ROW_SHUFFLED_SUMMARY_CONTROL,
                ROW_SHUFFLED_LABEL_CONTROL,
                ROW_SHUFFLED_RELIABILITY_CONTROL,
                ROW_RANDOM_SOURCE_MASS_CONTROL,
            ]
        ),
        "primary_center_equal_mean_bacc": decision.get("center_equal_mean_bacc", math.nan),
        "strongest_negative_control_method": decision.get("strongest_negative_control_method", ""),
        "strongest_negative_control_center_equal_mean_bacc": decision.get("strongest_negative_control_center_equal_mean_bacc", math.nan),
        "negative_control_gap": decision.get("negative_control_gap", math.nan),
        "control_competitive": "NEGATIVE_CONTROL_COMPETITIVE" in str(decision.get("diagnostic_flags", "")),
    }


def _write_decision_summary(root: Path, decision: Mapping[str, object], *, leakage_status: str) -> None:
    text = "\n".join(
        [
            "# Decentralized Component-Level Generative Expert Composition",
            "",
            "## Summary",
            "",
            f"- Primary method: `{PRIMARY_COMPONENT_UNION_METHOD}`",
            f"- Primary verdict: `{decision.get('primary_verdict', 'COMPONENT_UNION_FAIL')}`",
            f"- Diagnostic flags: `{decision.get('diagnostic_flags', '')}`",
            f"- Center-equal mean BACC: {_format_float(decision.get('center_equal_mean_bacc'))}",
            f"- Seed-cell mean BACC: {_format_float(decision.get('seed_cell_mean_bacc'))}",
            f"- Center-equal macro-F1: {_format_float(decision.get('center_equal_macro_f1'))}",
            f"- Min center BACC: {_format_float(decision.get('min_center_bacc'))}",
            f"- Seed std BACC: {_format_float(decision.get('seed_std_bacc'))}",
            f"- Delta vs D1.2 reliability all4: {_format_float(decision.get('delta_vs_d1_2_reliability_all4'))}",
            f"- Delta vs equal all4: {_format_float(decision.get('delta_vs_equal_all4'))}",
            f"- Retention vs source-union K16: {_format_float(decision.get('retention_vs_source_union_k16'))}",
            f"- Retention vs center-balanced K16: {_format_float(decision.get('retention_vs_center_balanced_k16'))}",
            f"- Delta vs real-feature dense reference: {_format_float(decision.get('delta_vs_real_source_embedding_dense_reference'))}",
            f"- Negative-control gap: {_format_float(decision.get('negative_control_gap'))}",
            f"- Max source-ablation drop: {_format_float(decision.get('max_source_ablation_drop_bacc'))}",
            f"- Max source-ablation gain: {_format_float(decision.get('max_source_ablation_gain_bacc'))}",
            f"- Leakage status: `{leakage_status}`",
            "",
            "## Protocol Boundary",
            "",
            PROTOCOL_WORDING,
            "",
            "This experiment does not test target-conditioned routing.",
            "It tests whether decentralized generative composition should operate at component/prototype granularity rather than whole-source granularity.",
            "The routing decision is fixed: use all non-heldout source experts.",
            "Target evaluation labels are used only for final scoring.",
            "",
            "## Supported Claim If Successful",
            "",
            "Source-local component union preserves intra-source multimodality and improves decentralized generated-embedding utility over source-level dense aggregation without target support or raw data sharing.",
            "",
        ]
    )
    path = root / "reports" / "decision_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _resolved_config(cfg: ComponentUnionConfig) -> dict[str, object]:
    return {
        "name": cfg.name,
        "artifact_root": str(cfg.artifact_root),
        "repair_artifact_root": str(cfg.repair_artifact_root),
        "d1_2_artifact_root": "" if cfg.d1_2_artifact_root is None else str(cfg.d1_2_artifact_root),
        "source_union_gmm_artifact_root": "" if cfg.source_union_gmm_artifact_root is None else str(cfg.source_union_gmm_artifact_root),
        "balanced_gmm_artifact_root": "" if cfg.balanced_gmm_artifact_root is None else str(cfg.balanced_gmm_artifact_root),
        "feature_cache_root": str(cfg.feature_cache_root),
        "backbone": cfg.backbone,
        "experiment_seeds": list(cfg.experiment_seeds),
        "heldout_centers": list(cfg.heldout_centers),
        "replicate_seeds": list(cfg.replicate_seeds),
        "synthetic_per_class_total": cfg.synthetic_per_class_total,
        "budget_diagnostic_per_class_total": cfg.budget_diagnostic_per_class_total,
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
        "variance_ceiling_multiplier": cfg.variance_ceiling_multiplier,
        "primary_pooling": cfg.primary_pooling,
        "reliability_floor_score": cfg.reliability_floor_score,
        "shrink_lambdas": list(cfg.shrink_lambdas),
        "prototype_candidate_counts_per_source_class": list(cfg.prototype_candidate_counts_per_source_class),
        "prototype_min_samples_per_component": cfg.prototype_min_samples_per_component,
        "prototype_variance_floor": cfg.prototype_variance_floor,
        "classifier": {
            "type": cfg.classifier_type,
            "solver": cfg.classifier_solver,
            "C": cfg.classifier_c,
            "max_iter": cfg.classifier_max_iter,
            "class_weight": cfg.classifier_class_weight,
            "classifier_seed": cfg.classifier_seed,
        },
    }
