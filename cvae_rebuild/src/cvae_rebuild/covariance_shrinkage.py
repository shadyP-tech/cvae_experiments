from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .downstream import evaluate_probability_predictions, fit_locked_logistic_classifier
from .features import load_feature_cache, select_rows
from .metrics import nanmean
from .preservation import _hash_array
from .preservation_repair import (
    NA,
    POOL_PER_SOURCE,
    POOL_SOURCE_UNION,
    PRIMARY_VARIANT,
    RepairConfig,
    SourceProbeConfig,
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
from .preservation_sampling import (
    DIAGNOSTIC_SELECTION,
    PRIMARY_SELECTION,
    UNION_VARIANT,
    RuntimeSource,
    _manifest_row,
    _per_source_variant,
    _runtime_source,
    _union_variant,
)
from .prior_calibration import (
    ROW_DIAG_PRIOR,
    ROW_FULL_COV_PRIOR,
    ROW_STANDARD_PRIOR,
    ROW_UNION_DIAG_PRIOR,
    _balanced_labels,
    _decode_latents,
    _standard_z,
)
from .protocol import ProtocolError, assert_candidate_pool, build_leakage_report
from .reporting import prepare_artifact_dirs, write_csv_rows, write_json
from .splits import candidate_experts


SHRINKAGE_STABILITY_NAME = "virchow2_cvae_covariance_shrinkage_stability_v1"
CONFIRMATION_ARTIFACT_NAME = "virchow2_cvae_covariance_prior_confirmation_v1"
VIABILITY_AUDIT_NAME = "virchow2_cvae_covariance_prior_viability_audit_v1"
PRIMARY_SHRINKAGE_METHOD = "cvae_cc_cov_diag_shrinkage075_prior_sample"
ROW_SHRINKAGE075 = PRIMARY_SHRINKAGE_METHOD
ROW_SHRINKAGE050 = "cvae_cc_cov_diag_shrinkage050_prior_sample_diagnostic"
ROW_SHRINKAGE090 = "cvae_cc_cov_diag_shrinkage090_prior_sample_diagnostic"
ROW_ALPHA010_REFERENCE = "cvae_cc_cov_shrinkage010_prior_reference"
ROW_COVARIANCE_CONFIRMATION_PRIOR = "cvae_cc_cov_shrinkage_prior_sample"
ROW_DIAG_REFERENCE = "cvae_cc_diag_aggregate_prior_reference"
ROW_ROLES = (
    ROW_STANDARD_PRIOR,
    ROW_DIAG_REFERENCE,
    ROW_ALPHA010_REFERENCE,
    ROW_SHRINKAGE050,
    ROW_SHRINKAGE075,
    ROW_SHRINKAGE090,
)


@dataclass(frozen=True)
class CovarianceShrinkageConfig:
    name: str
    artifact_root: Path
    repair_artifact_root: Path
    sampling_artifact_root: Path
    prior_calibration_artifact_root: Path
    covariance_confirmation_artifact_root: Path
    covariance_viability_artifact_root: Path
    feature_cache_root: Path
    experiment_seeds: tuple[int, ...]
    heldout_centers: tuple[str, ...]
    replicate_seeds: tuple[int, ...]
    synthetic_per_class_total: int
    primary_variant: str
    min_decision_cells: int
    primary_method: str
    primary_covariance_shrinkage_alpha: float
    diagnostic_covariance_shrinkage_alphas: tuple[float, ...]
    reference_covariance_shrinkage_alpha: float
    diagonal_reference_alpha: float
    covariance_shrinkage_alpha: float
    covariance_eigenvalue_floor: float
    full_cov_min_records_per_class: int
    fallback_if_under_ranked: str
    standard_prior_repro_abs_tol_bacc: float
    diag_prior_repro_abs_tol_bacc: float
    full_cov_diagnostic_repro_abs_tol_bacc: float
    classifier_type: str
    classifier_solver: str
    classifier_c: float
    classifier_max_iter: int
    classifier_class_weight: str
    classifier_seed: int | None


@dataclass(frozen=True)
class ImportedReference:
    reference_real_budget_bacc: float
    variant_real_budget_bacc: float
    source_utility_stratum_reference: str
    imported_standard_prior_bacc: float
    imported_diag_prior_bacc: float
    imported_full_cov_diagnostic_bacc: float
    imported_total_prior_cvae_gap: float
    imported_diag_prior_gap: float
    imported_full_cov_diagnostic_gap: float
    source_budget_index_hash: str
    decision_cell_id: str
    decision_cell_set_hash: str


@dataclass(frozen=True)
class CovarianceClassStats:
    class_label: int
    n_records: int
    mean: object
    sigma_emp: object
    sigma_diag: object
    sigma_psd: object
    factor: object
    fallback_used: bool
    fallback_reason: str
    trace_before_shrinkage: float
    trace_after_shrinkage: float
    condition_number_after_clip: float
    num_eigenvalues_clipped: int
    min_eigenvalue_before_clip: float
    min_eigenvalue_after_clip: float
    max_eigenvalue_after_clip: float
    mean_diag_variance: float
    mean_offdiag_abs_correlation: float
    covariance_effective_rank: float
    offdiag_frobenius_ratio: float
    trace_ratio_vs_diag: float


@dataclass(frozen=True)
class CovariancePriorParameters:
    classes: dict[int, CovarianceClassStats]
    labels: tuple[int, ...]
    prior_fit_row_ids_hash: str
    prior_fit_feature_hash: str
    prior_fit_label_hash: str
    prior_parameter_hash: str
    parameter_rows: tuple[dict[str, object], ...]
    fallback_rows: tuple[dict[str, object], ...]
    status: str
    error_message: str


def load_covariance_shrinkage_config(path: str | Path) -> CovarianceShrinkageConfig:
    source = Path(path).resolve()
    data = _load_mapping(source)
    base_dir = source.parents[2] if len(source.parents) >= 3 else source.parent
    return parse_covariance_shrinkage_config(data, base_dir=base_dir)


def parse_covariance_shrinkage_config(data: Mapping[str, Any], *, base_dir: str | Path = ".") -> CovarianceShrinkageConfig:
    base = Path(base_dir)
    experiment = _mapping(data, "experiment")
    inputs = _mapping(data, "inputs")
    run = _mapping(data, "run_matrix")
    generation = _mapping(data, "generation")
    prior = _mapping(data, "covariance_shrinkage")
    classifier = _mapping(data, "classifier")
    cfg = CovarianceShrinkageConfig(
        name=str(experiment["name"]),
        artifact_root=_path(base, str(experiment["artifact_root"])),
        repair_artifact_root=_path(base, str(inputs["repair_artifact_root"])),
        sampling_artifact_root=_path(base, str(inputs["sampling_artifact_root"])),
        prior_calibration_artifact_root=_path(base, str(inputs["prior_calibration_artifact_root"])),
        covariance_confirmation_artifact_root=_path(base, str(inputs["covariance_confirmation_artifact_root"])),
        covariance_viability_artifact_root=_path(base, str(inputs["covariance_viability_artifact_root"])),
        feature_cache_root=_path(base, str(inputs["feature_cache_root"])),
        experiment_seeds=tuple(int(v) for v in run["experiment_seeds"]),
        heldout_centers=tuple(str(v) for v in run["heldout_centers"]),
        replicate_seeds=tuple(int(v) for v in run["replicate_seeds"]),
        synthetic_per_class_total=int(generation["synthetic_per_class_total"]),
        primary_variant=str(experiment["primary_variant"]),
        min_decision_cells=int(experiment.get("min_decision_cells", 9)),
        primary_method=str(prior["primary_method"]),
        primary_covariance_shrinkage_alpha=float(prior["primary_covariance_shrinkage_alpha"]),
        diagnostic_covariance_shrinkage_alphas=tuple(float(v) for v in prior["diagnostic_covariance_shrinkage_alphas"]),
        reference_covariance_shrinkage_alpha=float(prior["reference_covariance_shrinkage_alpha"]),
        diagonal_reference_alpha=float(prior["diagonal_reference_alpha"]),
        covariance_shrinkage_alpha=float(prior["primary_covariance_shrinkage_alpha"]),
        covariance_eigenvalue_floor=float(prior["covariance_eigenvalue_floor"]),
        full_cov_min_records_per_class=int(prior["full_cov_min_records_per_class"]),
        fallback_if_under_ranked=str(prior["fallback_if_under_ranked"]),
        standard_prior_repro_abs_tol_bacc=float(prior["standard_prior_repro_abs_tol_bacc"]),
        diag_prior_repro_abs_tol_bacc=float(prior["diag_prior_repro_abs_tol_bacc"]),
        full_cov_diagnostic_repro_abs_tol_bacc=float(prior["alpha010_repro_abs_tol_bacc"]),
        classifier_type=str(classifier["type"]),
        classifier_solver=str(classifier["solver"]),
        classifier_c=float(classifier["C"]),
        classifier_max_iter=int(classifier["max_iter"]),
        classifier_class_weight=str(classifier["class_weight"]),
        classifier_seed=None if classifier.get("classifier_seed") is None else int(classifier["classifier_seed"]),
    )
    validate_covariance_shrinkage_config(cfg)
    return cfg


def validate_covariance_shrinkage_config(cfg: CovarianceShrinkageConfig) -> None:
    if cfg.name != SHRINKAGE_STABILITY_NAME:
        raise ProtocolError(f"Covariance shrinkage experiment name must be {SHRINKAGE_STABILITY_NAME!r}.")
    if cfg.covariance_confirmation_artifact_root.name != CONFIRMATION_ARTIFACT_NAME:
        raise ProtocolError(f"covariance_confirmation_artifact_root must point to {CONFIRMATION_ARTIFACT_NAME!r}.")
    if cfg.covariance_viability_artifact_root.name != VIABILITY_AUDIT_NAME:
        raise ProtocolError(f"covariance_viability_artifact_root must point to {VIABILITY_AUDIT_NAME!r}.")
    if cfg.primary_variant != PRIMARY_VARIANT:
        raise ProtocolError(f"primary_variant must be {PRIMARY_VARIANT!r}.")
    if cfg.primary_method != PRIMARY_SHRINKAGE_METHOD:
        raise ProtocolError(f"primary_method must be {PRIMARY_SHRINKAGE_METHOD!r}.")
    if cfg.synthetic_per_class_total != 128:
        raise ProtocolError("synthetic_per_class_total must be locked to 128.")
    if cfg.min_decision_cells != 9:
        raise ProtocolError("min_decision_cells must be locked to the frozen 9-cell diagnostic population.")
    if not math.isclose(cfg.primary_covariance_shrinkage_alpha, 0.75, rel_tol=0.0, abs_tol=1.0e-12):
        raise ProtocolError("primary_covariance_shrinkage_alpha must be exactly 0.75.")
    if cfg.diagnostic_covariance_shrinkage_alphas != (0.50, 0.90):
        raise ProtocolError("diagnostic_covariance_shrinkage_alphas must be exactly [0.50, 0.90].")
    if not math.isclose(cfg.reference_covariance_shrinkage_alpha, 0.10, rel_tol=0.0, abs_tol=1.0e-12):
        raise ProtocolError("reference_covariance_shrinkage_alpha must be exactly 0.10.")
    if not math.isclose(cfg.diagonal_reference_alpha, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ProtocolError("diagonal_reference_alpha must be exactly 1.0.")
    if cfg.covariance_eigenvalue_floor != 1.0e-4:
        raise ProtocolError("covariance_eigenvalue_floor must be exactly 1.0e-4.")
    if cfg.full_cov_min_records_per_class != 32:
        raise ProtocolError("full_cov_min_records_per_class must be locked to 32.")
    if cfg.fallback_if_under_ranked != "diag":
        raise ProtocolError("fallback_if_under_ranked must be diag.")
    if min(cfg.standard_prior_repro_abs_tol_bacc, cfg.diag_prior_repro_abs_tol_bacc, cfg.full_cov_diagnostic_repro_abs_tol_bacc) <= 0.0:
        raise ProtocolError("Reproduction tolerances must be positive.")
    if cfg.classifier_type != "sklearn_logistic_regression":
        raise ProtocolError("classifier.type must be sklearn_logistic_regression.")
    if cfg.classifier_solver != "lbfgs" or cfg.classifier_c != 1.0 or cfg.classifier_max_iter != 2000:
        raise ProtocolError("Classifier solver/C/max_iter must remain locked.")
    if cfg.classifier_class_weight != "balanced" or cfg.classifier_seed is not None:
        raise ProtocolError("Classifier must use class_weight=balanced and classifier_seed=null.")


def _all_alphas(cfg: CovarianceShrinkageConfig) -> tuple[float, ...]:
    values = (
        cfg.diagonal_reference_alpha,
        cfg.reference_covariance_shrinkage_alpha,
        *cfg.diagnostic_covariance_shrinkage_alphas,
        cfg.primary_covariance_shrinkage_alpha,
    )
    return tuple(sorted({float(value) for value in values}))


def _cfg_with_alpha(cfg: CovarianceShrinkageConfig, alpha: float) -> CovarianceShrinkageConfig:
    return replace(cfg, covariance_shrinkage_alpha=float(alpha))


def run_covariance_shrinkage_stability(cfg: CovarianceShrinkageConfig, *, artifact_root: str | Path | None = None) -> Path:
    root = prepare_artifact_dirs(Path(artifact_root) if artifact_root is not None else cfg.artifact_root)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    matrix_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    fallback_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    protocol_violations: list[str] = []
    target_expert_excluded = True

    try:
        _validate_imported_artifacts(cfg)
        references, decision_cell_set_hash = _load_imported_references(cfg)
    except ProtocolError as exc:
        protocol_violations.append(str(exc))
        _write_artifacts(
            root,
            cfg,
            matrix_rows=[],
            gap_rows=[],
            parameter_rows=[],
            fallback_rows=[],
            low_stratum_rows=[],
            source_pool_rows=[],
            manifest_rows=[],
            decision=_decision([], cfg, leakage_status="FAIL", decision_cell_set_hash=""),
            protocol_violations=protocol_violations,
            target_expert_excluded=True,
        )
        return root

    repair_cfg = _repair_runtime_config(cfg, root)
    per_source_variant = _per_source_variant()
    union_variant = _union_variant()

    try:
        for experiment_seed in cfg.experiment_seeds:
            train_cache = load_feature_cache(_existing_cache_path(cfg.feature_cache_root, seed=experiment_seed, split="train"))
            test_cache = load_feature_cache(_existing_cache_path(cfg.feature_cache_root, seed=experiment_seed, split="test"))
            per_source_data = {
                center: _source_data_for_centers(train_cache, centers=(center,), experiment_seed=int(experiment_seed))
                for center in cfg.heldout_centers
            }
            per_source_runtime: dict[str, RuntimeSource] = {}
            for expert_id, source_data in per_source_data.items():
                per_source_runtime[str(expert_id)] = _runtime_source(
                    cfg,
                    repair_cfg,
                    root=root,
                    experiment_seed=int(experiment_seed),
                    heldout_center=NA,
                    expert_id=str(expert_id),
                    source_data=source_data,
                    variant=per_source_variant,
                )
                manifest_rows.append(_manifest_row(experiment_seed, NA, per_source_runtime[str(expert_id)]))

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

                union_data = _source_data_for_centers(train_cache, centers=candidates, experiment_seed=int(experiment_seed))
                union_runtime = _runtime_source(
                    cfg,
                    repair_cfg,
                    root=root,
                    experiment_seed=int(experiment_seed),
                    heldout_center=str(heldout_center),
                    expert_id=POOL_SOURCE_UNION,
                    source_data=union_data,
                    variant=union_variant,
                )
                manifest_rows.append(_manifest_row(experiment_seed, str(heldout_center), union_runtime))

                target_indices = _target_indices(test_cache.metadata, str(heldout_center))
                eval_raw, eval_meta = select_rows(test_cache.embeddings, test_cache.metadata, target_indices)
                eval_labels = tuple(_label(row) for row in eval_meta)
                eval_error = "mono_class_target_eval" if len(set(eval_labels)) < 2 else ""

                for expert_id in candidates:
                    rows, params, fallbacks = _evaluate_runtime(
                        cfg,
                        references=references,
                        runtime=per_source_runtime[str(expert_id)].runtime,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        eval_raw=eval_raw,
                        eval_labels=eval_labels,
                        eval_error=eval_error,
                    )
                    matrix_rows.extend(rows)
                    parameter_rows.extend(params)
                    fallback_rows.extend(fallbacks)

                rows, params, fallbacks = _evaluate_runtime(
                    cfg,
                    references=references,
                    runtime=union_runtime.runtime,
                    experiment_seed=int(experiment_seed),
                    heldout_center=str(heldout_center),
                    eval_raw=eval_raw,
                    eval_labels=eval_labels,
                    eval_error=eval_error,
                )
                matrix_rows.extend(rows)
                parameter_rows.extend(params)
                fallback_rows.extend(fallbacks)
    except ProtocolError as exc:
        protocol_violations.append(str(exc))

    _augment_paired_baseline_fields(matrix_rows)
    gap_rows = _gap_rows(matrix_rows)
    leakage = build_leakage_report(
        target_support_labels_for_selection=False,
        target_eval_labels_for_scoring_only=True,
        target_expert_excluded=target_expert_excluded,
        oracle_rows_diagnostic_only=True,
        extra_violations=protocol_violations,
    )
    decision = _decision(gap_rows, cfg, leakage_status=leakage.status, decision_cell_set_hash=decision_cell_set_hash)
    _write_artifacts(
        root,
        cfg,
        matrix_rows=matrix_rows,
        gap_rows=gap_rows,
        parameter_rows=parameter_rows,
        fallback_rows=fallback_rows,
        low_stratum_rows=_low_stratum_rows(gap_rows),
        source_pool_rows=_source_pool_rows(gap_rows),
        manifest_rows=manifest_rows,
        decision=decision,
        protocol_violations=protocol_violations,
        target_expert_excluded=target_expert_excluded,
    )
    return root


def _evaluate_runtime(
    cfg: CovarianceShrinkageConfig,
    *,
    references: Mapping[tuple[object, ...], ImportedReference],
    runtime: VariantRuntime,
    experiment_seed: int,
    heldout_center: str,
    eval_raw: object,
    eval_labels: Sequence[int],
    eval_error: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    fallback_rows: list[dict[str, object]] = []
    source_error = "" if set(int(v) for v in runtime.source_train_labels) == {0, 1} else "mono_class_source_train"
    error = eval_error or source_error
    if error:
        for seed in cfg.replicate_seeds:
            try:
                ref = _reference_for_runtime(references, runtime, experiment_seed, heldout_center, int(seed))
            except ProtocolError:
                ref = _empty_reference(experiment_seed, heldout_center, runtime)
            rows.extend(_ineligible_rows(cfg, runtime, experiment_seed, heldout_center, int(seed), ref, error))
        return rows, parameter_rows, fallback_rows

    eval_x = runtime.frame.transform(_to_numpy(eval_raw))
    params_by_alpha: dict[float, CovariancePriorParameters] = {}
    for alpha in _all_alphas(cfg):
        alpha_cfg = _cfg_with_alpha(cfg, alpha)
        params = _fit_covariance_prior_parameters(alpha_cfg, runtime, experiment_seed=experiment_seed, heldout_center=heldout_center)
        params_by_alpha[float(alpha)] = params
        parameter_rows.extend(params.parameter_rows)
        fallback_rows.extend(params.fallback_rows)
    primary_params = params_by_alpha[float(cfg.primary_covariance_shrinkage_alpha)]
    if primary_params.status != "ok":
        for seed in cfg.replicate_seeds:
            ref = _reference_for_runtime(references, runtime, experiment_seed, heldout_center, int(seed))
            rows.extend(_ineligible_rows(cfg, runtime, experiment_seed, heldout_center, int(seed), ref, primary_params.error_message))
        return rows, parameter_rows, fallback_rows

    for seed in cfg.replicate_seeds:
        ref = _reference_for_runtime(references, runtime, experiment_seed, heldout_center, int(seed))
        rows.extend(_evaluate_prior_methods(cfg, runtime, params_by_alpha, ref, experiment_seed, heldout_center, int(seed), eval_x, eval_labels))
    return rows, parameter_rows, fallback_rows


def _evaluate_prior_methods(
    cfg: CovarianceShrinkageConfig,
    runtime: VariantRuntime,
    params_by_alpha: Mapping[float, CovariancePriorParameters],
    ref: ImportedReference,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    eval_x: object,
    eval_labels: Sequence[int],
) -> list[dict[str, object]]:
    primary_selection = runtime.variant.selection_source if runtime.variant.expert_pool_type == POOL_PER_SOURCE else DIAGNOSTIC_SELECTION
    specs: list[tuple[str, str, float, str, str]] = [
        (ROW_STANDARD_PRIOR, "standard_normal", cfg.diagonal_reference_alpha, DIAGNOSTIC_SELECTION, "class_count_matched"),
        (ROW_DIAG_REFERENCE, "diag_aggregate", cfg.diagonal_reference_alpha, DIAGNOSTIC_SELECTION, "class_count_matched"),
        (
            ROW_ALPHA010_REFERENCE,
            "covariance_shrinkage",
            cfg.reference_covariance_shrinkage_alpha,
            DIAGNOSTIC_SELECTION,
            "class_count_matched",
        ),
        (ROW_SHRINKAGE050, "covariance_shrinkage", 0.50, DIAGNOSTIC_SELECTION, "class_count_matched"),
        (ROW_SHRINKAGE075, "covariance_shrinkage", cfg.primary_covariance_shrinkage_alpha, primary_selection, "class_count_matched"),
        (ROW_SHRINKAGE090, "covariance_shrinkage", 0.90, DIAGNOSTIC_SELECTION, "class_count_matched"),
    ]
    rows = []
    for row_role, method, alpha, selection_source, budget_match_type in specs:
        alpha_cfg = _cfg_with_alpha(cfg, alpha)
        params = params_by_alpha[float(alpha)]
        generated, labels = _sample_features(alpha_cfg, runtime, params, method=method, seed=replicate_seed)
        rows.append(
            _evaluate_generated(
                alpha_cfg,
                runtime,
                params,
                ref,
                experiment_seed,
                heldout_center,
                row_role=row_role,
                prior_method=row_role,
                replicate_seed=replicate_seed,
                generated=generated,
                labels=labels,
                eval_x=eval_x,
                eval_labels=eval_labels,
                selection_source=selection_source,
                budget_match_type=budget_match_type,
            )
        )
    return rows


def _evaluate_generated(
    cfg: CovarianceShrinkageConfig,
    runtime: VariantRuntime,
    params: CovariancePriorParameters,
    ref: ImportedReference,
    experiment_seed: int,
    heldout_center: str,
    *,
    row_role: str,
    prior_method: str,
    replicate_seed: int,
    generated: object,
    labels: Sequence[int],
    eval_x: object,
    eval_labels: Sequence[int],
    selection_source: str,
    budget_match_type: str,
) -> dict[str, object]:
    bundle = fit_locked_logistic_classifier(
        generated,
        labels,
        eval_x,
        classifier_seed=cfg.classifier_seed,
        expert_id=runtime.expert_id,
        class_weight=cfg.classifier_class_weight,
    )
    result = evaluate_probability_predictions(row_role, bundle.probabilities, eval_labels)
    return _covariance_row(
        cfg,
        runtime,
        params,
        ref,
        experiment_seed,
        heldout_center,
        row_role=row_role,
        prior_method=prior_method,
        replicate_seed=replicate_seed,
        generated_features_hash=_hash_array(generated),
        prediction_hash=_hash_array(bundle.probabilities),
        bacc=result.bacc,
        macro_f1=result.macro_f1,
        selection_source=selection_source,
        status="ok",
        error_message="",
        budget_match_type=budget_match_type,
    )


def _covariance_row(
    cfg: CovarianceShrinkageConfig,
    runtime: VariantRuntime,
    params: CovariancePriorParameters,
    ref: ImportedReference,
    experiment_seed: int,
    heldout_center: str,
    *,
    row_role: str,
    prior_method: str,
    replicate_seed: int | str,
    generated_features_hash: str,
    prediction_hash: str,
    bacc: float | str,
    macro_f1: float | str,
    selection_source: str,
    status: str,
    error_message: str,
    budget_match_type: str,
) -> dict[str, object]:
    bacc_value = _float(bacc)
    total_gap = ref.variant_real_budget_bacc - bacc_value if math.isfinite(bacc_value) else math.nan
    standard_delta = bacc_value - ref.imported_standard_prior_bacc if math.isfinite(bacc_value) else math.nan
    diag_delta = bacc_value - ref.imported_diag_prior_bacc if math.isfinite(bacc_value) else math.nan
    health = _parameter_health(params)
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "expert_id": runtime.expert_id,
        "expert_pool_type": runtime.variant.expert_pool_type,
        "variant_id": runtime.variant.variant_id,
        "row_role": row_role,
        "prior_method": prior_method,
        "replicate_seed": replicate_seed,
        "decision_cell_id": ref.decision_cell_id,
        "decision_cell_set_hash": ref.decision_cell_set_hash,
        "source_utility_stratum_reference": ref.source_utility_stratum_reference,
        "variant_real_stratum": _variant_real_stratum(ref.variant_real_budget_bacc),
        "variant_real_budget_bacc": ref.variant_real_budget_bacc,
        "imported_standard_prior_bacc": ref.imported_standard_prior_bacc,
        "imported_diag_prior_bacc": ref.imported_diag_prior_bacc,
        "imported_full_cov_diagnostic_bacc": ref.imported_full_cov_diagnostic_bacc,
        "rerun_standard_prior_bacc": NA,
        "rerun_diag_prior_bacc": NA,
        "rerun_alpha010_prior_bacc": NA,
        "bacc": bacc,
        "macro_f1": macro_f1,
        "delta_bacc_vs_standard_prior": standard_delta,
        "delta_bacc_vs_diag_prior": diag_delta,
        "delta_bacc_vs_alpha010": math.nan,
        "total_shrinkage_prior_gap": total_gap,
        "preservation_ratio": bacc_value / ref.variant_real_budget_bacc if math.isfinite(bacc_value) and math.isfinite(ref.variant_real_budget_bacc) and ref.variant_real_budget_bacc > 0 else math.nan,
        "clipped_preservation_gap": max(0.0, total_gap) if math.isfinite(total_gap) else math.nan,
        "gap_reduction_vs_standard_prior": ref.imported_total_prior_cvae_gap - total_gap if math.isfinite(total_gap) else math.nan,
        "gap_reduction_vs_diag_prior": ref.imported_diag_prior_gap - total_gap if math.isfinite(total_gap) else math.nan,
        "gap_reduction_vs_alpha010": math.nan,
        "covariance_shrinkage_alpha": cfg.covariance_shrinkage_alpha,
        "covariance_eigenvalue_floor": cfg.covariance_eigenvalue_floor,
        "covariance_fallback_used": health["covariance_fallback_used"],
        "fallback_reason": health["fallback_reason"],
        "prior_fit_row_ids_hash": params.prior_fit_row_ids_hash,
        "prior_parameter_hash": params.prior_parameter_hash,
        "generated_features_hash": generated_features_hash,
        "prediction_hash": prediction_hash,
        "selection_source": selection_source,
        "status": status,
        "error_message": error_message,
        "budget_match_type": budget_match_type,
        **{key: health[key] for key in _health_columns()},
    }


def _ineligible_rows(
    cfg: CovarianceShrinkageConfig,
    runtime: VariantRuntime,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    ref: ImportedReference,
    error_message: str,
) -> list[dict[str, object]]:
    empty = CovariancePriorParameters(
        classes={},
        labels=(),
        prior_fit_row_ids_hash="",
        prior_fit_feature_hash="",
        prior_fit_label_hash="",
        prior_parameter_hash="",
        parameter_rows=(),
        fallback_rows=(),
        status="ineligible",
        error_message=error_message,
    )
    rows = []
    for row_role in ROW_ROLES:
        rows.append(
            _covariance_row(
                cfg,
                runtime,
                empty,
                ref,
                experiment_seed,
                heldout_center,
                row_role=row_role,
                prior_method=row_role,
                replicate_seed=replicate_seed,
                generated_features_hash="",
                prediction_hash="",
                bacc="",
                macro_f1="",
                selection_source=runtime.variant.selection_source if row_role == ROW_SHRINKAGE075 else DIAGNOSTIC_SELECTION,
                status="ineligible",
                error_message=error_message,
                budget_match_type="class_count_matched",
            )
        )
    return rows


def _fit_covariance_prior_parameters(
    cfg: CovarianceShrinkageConfig,
    runtime: VariantRuntime,
    *,
    experiment_seed: int,
    heldout_center: str,
) -> CovariancePriorParameters:
    import numpy as np  # type: ignore
    import torch  # type: ignore

    y_np = np.asarray(runtime.source_train_labels, dtype=int)
    missing = [cls for cls in (0, 1) if int((y_np == cls).sum()) < 1]
    prior_fit_ids_hash = _hash_strings(runtime.source_train_sample_ids)
    prior_fit_feature_hash = _hash_array(runtime.source_train_embeddings)
    prior_fit_label_hash = _hash_strings([str(v) for v in y_np.tolist()])
    if missing:
        return CovariancePriorParameters(
            classes={},
            labels=tuple(int(v) for v in y_np.tolist()),
            prior_fit_row_ids_hash=prior_fit_ids_hash,
            prior_fit_feature_hash=prior_fit_feature_hash,
            prior_fit_label_hash=prior_fit_label_hash,
            prior_parameter_hash="",
            parameter_rows=(),
            fallback_rows=(),
            status="ineligible",
            error_message=f"insufficient_source_class_records:{','.join(str(v) for v in missing)}",
        )
    with torch.no_grad():
        x = torch.as_tensor(np.asarray(runtime.source_train_embeddings, dtype=np.float32), dtype=torch.float32)
        y = torch.as_tensor(y_np, dtype=torch.long)
        mu, logvar = runtime.model.encode(x, y)
    mu_np = mu.detach().cpu().numpy()
    post_var = np.exp(logvar.detach().cpu().numpy())
    classes: dict[int, CovarianceClassStats] = {}
    parameter_rows: list[dict[str, object]] = []
    fallback_rows: list[dict[str, object]] = []
    hash_payload = []
    for cls in (0, 1):
        mask = y_np == cls
        cls_mu = mu_np[mask]
        cls_post_var = post_var[mask]
        mean = cls_mu.mean(axis=0)
        if cls_mu.shape[0] >= 2:
            cov_mu = np.cov(cls_mu, rowvar=False, ddof=1)
        else:
            cov_mu = np.zeros((cls_mu.shape[1], cls_mu.shape[1]), dtype=float)
        sigma_emp = np.asarray(cov_mu, dtype=float) + np.diag(cls_post_var.mean(axis=0))
        sigma_emp = (sigma_emp + sigma_emp.T) / 2.0
        sigma_diag = np.diag(np.maximum(np.diag(sigma_emp), cfg.covariance_eigenvalue_floor))
        fallback_used = int(mask.sum()) < cfg.full_cov_min_records_per_class
        fallback_reason = f"n_class_records<{cfg.full_cov_min_records_per_class}" if fallback_used else ""
        if fallback_used:
            sigma_psd = sigma_diag
            factor = np.diag(np.sqrt(np.diag(sigma_diag)))
            health = _covariance_health(sigma_emp, sigma_diag, cfg.covariance_eigenvalue_floor, fallback_used=True)
        else:
            sigma_psd, factor, health = _stabilized_covariance_psd(
                sigma_emp,
                alpha=cfg.covariance_shrinkage_alpha,
                eigenvalue_floor=cfg.covariance_eigenvalue_floor,
            )
        stats = CovarianceClassStats(
            class_label=int(cls),
            n_records=int(mask.sum()),
            mean=mean,
            sigma_emp=sigma_emp,
            sigma_diag=sigma_diag,
            sigma_psd=sigma_psd,
            factor=factor,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            **health,
        )
        classes[int(cls)] = stats
        hash_payload.extend([mean, sigma_emp, sigma_psd])
        parameter_rows.append(
            {
                "experiment_seed": int(experiment_seed),
                "heldout_center": str(heldout_center),
                "expert_id": runtime.expert_id,
                "expert_pool_type": runtime.variant.expert_pool_type,
                "variant_id": runtime.variant.variant_id,
                "class_label": int(cls),
                "n_records": int(mask.sum()),
                "prior_fit_row_ids_hash": prior_fit_ids_hash,
                "prior_fit_feature_hash": prior_fit_feature_hash,
                "prior_fit_label_hash": prior_fit_label_hash,
                "prior_mean_hash": _hash_array(mean),
                "prior_parameter_hash": "",
                "covariance_shrinkage_alpha": cfg.covariance_shrinkage_alpha,
                "covariance_eigenvalue_floor": cfg.covariance_eigenvalue_floor,
                "covariance_fallback_used": bool(fallback_used),
                "fallback_reason": fallback_reason,
                **{key: (int(stats.n_records) if key == "class_min_n_encoded_records" else getattr(stats, key)) for key in _health_columns()},
            }
        )
        fallback_rows.append(
            {
                "experiment_seed": int(experiment_seed),
                "heldout_center": str(heldout_center),
                "expert_id": runtime.expert_id,
                "expert_pool_type": runtime.variant.expert_pool_type,
                "variant_id": runtime.variant.variant_id,
                "class_label": int(cls),
                "n_records": int(mask.sum()),
                "covariance_fallback_used": bool(fallback_used),
                "fallback_reason": fallback_reason,
            }
        )
    parameter_hash = _hash_array(np.concatenate([np.ravel(v) for v in hash_payload]))
    parameter_rows = [{**row, "prior_parameter_hash": parameter_hash} for row in parameter_rows]
    return CovariancePriorParameters(
        classes=classes,
        labels=tuple(int(v) for v in y_np.tolist()),
        prior_fit_row_ids_hash=prior_fit_ids_hash,
        prior_fit_feature_hash=prior_fit_feature_hash,
        prior_fit_label_hash=prior_fit_label_hash,
        prior_parameter_hash=parameter_hash,
        parameter_rows=tuple(parameter_rows),
        fallback_rows=tuple(fallback_rows),
        status="ok",
        error_message="",
    )


def _sample_features(
    cfg: CovarianceShrinkageConfig,
    runtime: VariantRuntime,
    params: CovariancePriorParameters,
    *,
    method: str,
    seed: int,
) -> tuple[object, tuple[int, ...]]:
    import numpy as np  # type: ignore

    if method == "standard_normal":
        return _decode_latents(runtime, _standard_z(runtime, cfg.synthetic_per_class_total, seed), _balanced_labels(cfg.synthetic_per_class_total))
    rng = np.random.default_rng(int(seed))
    chunks = []
    labels = _balanced_labels(cfg.synthetic_per_class_total)
    for cls in (0, 1):
        stats = params.classes[int(cls)]
        eps = rng.normal(size=(cfg.synthetic_per_class_total, int(runtime.model.latent_dim))).astype(np.float32)
        if method == "diag_aggregate":
            z_np = np.asarray(stats.mean, dtype=np.float32) + eps * np.sqrt(np.diag(stats.sigma_diag)).astype(np.float32)
        elif method == "covariance_shrinkage":
            z_np = np.asarray(stats.mean, dtype=np.float32) + eps @ np.asarray(stats.factor, dtype=np.float32).T
        else:
            raise ProtocolError(f"Unknown covariance prior method: {method}")
        decoded, _labels = _decode_latents(runtime, z_np, [int(cls)] * cfg.synthetic_per_class_total)
        chunks.append(decoded)
    return np.vstack(chunks), labels


def _stabilized_covariance_psd(cov: object, *, alpha: float, eigenvalue_floor: float) -> tuple[object, object, dict[str, float | int]]:
    import numpy as np  # type: ignore

    sigma = np.asarray(cov, dtype=float)
    sigma = (sigma + sigma.T) / 2.0
    diag = np.diag(np.diag(sigma))
    shrunk = (1.0 - float(alpha)) * sigma + float(alpha) * diag
    eigvals, eigvecs = np.linalg.eigh(shrunk)
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    eigvecs = _canonical_eigenvector_signs(eigvecs)
    clipped = np.maximum(eigvals, float(eigenvalue_floor))
    psd = (eigvecs * clipped) @ eigvecs.T
    factor = eigvecs @ np.diag(np.sqrt(clipped))
    health = _covariance_health(sigma, psd, eigenvalue_floor, fallback_used=False, eigvals_before=eigvals, eigvals_after=clipped, trace_after=float(np.trace(shrunk)))
    return psd, factor, health


def _canonical_eigenvector_signs(eigvecs: object) -> object:
    import numpy as np  # type: ignore

    out = np.asarray(eigvecs, dtype=float).copy()
    for idx in range(out.shape[1]):
        col = out[:, idx]
        pivot = int(np.argmax(np.abs(col)))
        if col[pivot] < 0:
            out[:, idx] *= -1.0
    return out


def _covariance_health(
    sigma_emp: object,
    sigma_after: object,
    eigenvalue_floor: float,
    *,
    fallback_used: bool,
    eigvals_before: object | None = None,
    eigvals_after: object | None = None,
    trace_after: float | None = None,
) -> dict[str, float | int]:
    import numpy as np  # type: ignore

    before = np.linalg.eigvalsh(np.asarray(sigma_emp, dtype=float)) if eigvals_before is None else np.asarray(eigvals_before, dtype=float)
    after = np.linalg.eigvalsh(np.asarray(sigma_after, dtype=float)) if eigvals_after is None else np.asarray(eigvals_after, dtype=float)
    after = np.maximum(after, float(eigenvalue_floor))
    total = float(after.sum())
    effective_rank = 0.0 if total <= 0 else float((total ** 2) / max(float(np.sum(after ** 2)), 1.0e-12))
    diag = np.diag(np.asarray(sigma_emp, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.sqrt(np.outer(np.maximum(diag, 0.0), np.maximum(diag, 0.0)))
        corr = np.divide(np.asarray(sigma_emp, dtype=float), denom, out=np.zeros_like(np.asarray(sigma_emp, dtype=float)), where=denom > 0)
    offdiag = corr[~np.eye(corr.shape[0], dtype=bool)] if corr.ndim == 2 else np.asarray([])
    after_matrix = np.asarray(sigma_after, dtype=float)
    diag_after = np.diag(np.diag(after_matrix))
    offdiag_after = after_matrix - diag_after
    after_norm = float(np.linalg.norm(after_matrix, ord="fro"))
    trace_after_value = float(np.trace(after_matrix)) if trace_after is None else float(trace_after)
    trace_diag = float(np.trace(diag_after))
    return {
        "trace_before_shrinkage": float(np.trace(np.asarray(sigma_emp, dtype=float))),
        "trace_after_shrinkage": trace_after_value,
        "condition_number_after_clip": float(np.max(after) / max(float(np.min(after)), float(eigenvalue_floor))),
        "num_eigenvalues_clipped": int(np.sum(before < float(eigenvalue_floor))) if not fallback_used else int(np.sum(after <= float(eigenvalue_floor))),
        "min_eigenvalue_before_clip": float(np.min(before)),
        "min_eigenvalue_after_clip": float(np.min(after)),
        "max_eigenvalue_after_clip": float(np.max(after)),
        "mean_diag_variance": float(np.mean(diag)),
        "mean_offdiag_abs_correlation": float(np.mean(np.abs(offdiag))) if offdiag.size else 0.0,
        "covariance_effective_rank": effective_rank,
        "offdiag_frobenius_ratio": float(np.linalg.norm(offdiag_after, ord="fro") / after_norm) if after_norm > 0.0 else 0.0,
        "trace_ratio_vs_diag": trace_after_value / trace_diag if trace_diag > 0.0 else math.nan,
    }


def _load_imported_references(cfg: CovarianceShrinkageConfig) -> tuple[dict[tuple[object, ...], ImportedReference], str]:
    sampling = _load_sampling_baselines(cfg)
    prior_rows = _load_prior_calibration_baselines(cfg)
    covariance_rows = _load_covariance_confirmation_baselines(cfg)
    decision_hashes = {ref["decision_cell_set_hash"] for ref in sampling.values()}
    prior_hashes = {row["decision_cell_set_hash"] for row in prior_rows.values()}
    covariance_hashes = {row["decision_cell_set_hash"] for row in covariance_rows.values()}
    if (
        len(decision_hashes) != 1
        or len(prior_hashes) != 1
        or len(covariance_hashes) != 1
        or decision_hashes != prior_hashes
        or decision_hashes != covariance_hashes
    ):
        raise ProtocolError("Decision-cell hash mismatch between sampling, prior-calibration, and covariance-confirmation artifacts.")
    decision_hash = next(iter(decision_hashes))
    out = {}
    for key, ref in sampling.items():
        diag_role = ROW_UNION_DIAG_PRIOR if key[3] == POOL_SOURCE_UNION else ROW_DIAG_PRIOR
        diag = prior_rows.get((key, diag_role))
        full = covariance_rows.get(key)
        if diag is None or full is None:
            raise ProtocolError(f"Missing imported diagonal prior or alpha010 covariance-confirmation reference for {key}.")
        out[key] = ImportedReference(
            reference_real_budget_bacc=float(ref["reference_real_budget_bacc"]),
            variant_real_budget_bacc=float(ref["variant_real_budget_bacc"]),
            source_utility_stratum_reference=str(ref["source_utility_stratum_reference"]),
            imported_standard_prior_bacc=float(ref["imported_standard_prior_bacc"]),
            imported_diag_prior_bacc=float(diag["calibrated_prior_bacc"]),
            imported_full_cov_diagnostic_bacc=float(full["bacc"]),
            imported_total_prior_cvae_gap=float(ref["imported_total_prior_cvae_gap"]),
            imported_diag_prior_gap=float(diag["total_calibrated_prior_cvae_gap"]),
            imported_full_cov_diagnostic_gap=float(full["total_covariance_prior_gap"]),
            source_budget_index_hash=str(ref["source_budget_index_hash"]),
            decision_cell_id=str(ref["decision_cell_id"]),
            decision_cell_set_hash=decision_hash,
        )
    if not out:
        raise ProtocolError("Imported references are empty.")
    return out, decision_hash


def _validate_imported_artifacts(cfg: CovarianceShrinkageConfig) -> None:
    required = (
        cfg.sampling_artifact_root / "reports" / "leakage_report.json",
        cfg.prior_calibration_artifact_root / "reports" / "leakage_report.json",
        cfg.covariance_confirmation_artifact_root / "reports" / "leakage_report.json",
        cfg.covariance_confirmation_artifact_root / "tables" / "covariance_prior_gap_summary.csv",
        cfg.covariance_viability_artifact_root / "reports" / "leakage_report.json",
        cfg.covariance_viability_artifact_root / "tables" / "conditional_viability_cells.csv",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ProtocolError(f"Missing imported shrinkage reference artifacts: {missing}")
    for path in required:
        if path.name != "leakage_report.json":
            continue
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("status") != "PASS":
            raise ProtocolError(f"Imported leakage report is not PASS: {path}")


def _load_sampling_baselines(cfg: CovarianceShrinkageConfig) -> dict[tuple[object, ...], dict[str, object]]:
    path = cfg.sampling_artifact_root / "tables" / "sampling_gap_summary.csv"
    if not path.exists():
        raise ProtocolError(f"Missing sampling gap summary: {path}")
    required = {
        "experiment_seed",
        "heldout_center",
        "expert_id",
        "expert_pool_type",
        "variant_id",
        "replicate_seed",
        "posterior_temperature",
        "prior_scale",
        "selection_source",
        "status",
        "reference_real_budget_bacc",
        "variant_real_budget_bacc",
        "source_utility_stratum_reference",
        "prior_bacc",
        "total_prior_cvae_gap",
        "source_budget_index_hash",
    }
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ProtocolError(f"Sampling gap summary is missing fields: {sorted(missing)}")
        for row in reader:
            if row.get("status") != "ok":
                continue
            if row.get("variant_id") not in {PRIMARY_VARIANT, UNION_VARIANT}:
                continue
            if str(row.get("posterior_temperature")) != "1.0" or str(row.get("prior_scale")) != "1.0":
                continue
            rows.append(row)
    decision_ids = sorted(
        {
            _decision_cell_id(row["experiment_seed"], row["heldout_center"], row["expert_id"])
            for row in rows
            if row["expert_pool_type"] == POOL_PER_SOURCE
            and row["variant_id"] == PRIMARY_VARIANT
            and row["selection_source"] == PRIMARY_SELECTION
            and row["source_utility_stratum_reference"] in {"medium", "high"}
        }
    )
    if not decision_ids:
        raise ProtocolError("Sampling artifact did not contain a primary decision-cell set.")
    decision_hash = _hash_strings(decision_ids)
    out: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        key = _reference_key(
            row["experiment_seed"],
            row["heldout_center"],
            row["expert_id"],
            row["expert_pool_type"],
            row["variant_id"],
            row["replicate_seed"],
        )
        out[key] = {
            "reference_real_budget_bacc": float(row["reference_real_budget_bacc"]),
            "variant_real_budget_bacc": float(row["variant_real_budget_bacc"]),
            "source_utility_stratum_reference": str(row["source_utility_stratum_reference"]),
            "imported_standard_prior_bacc": float(row["prior_bacc"]),
            "imported_total_prior_cvae_gap": float(row["total_prior_cvae_gap"]),
            "source_budget_index_hash": str(row["source_budget_index_hash"]),
            "decision_cell_id": _decision_cell_id(row["experiment_seed"], row["heldout_center"], row["expert_id"]),
            "decision_cell_set_hash": decision_hash,
        }
    return out


def _load_prior_calibration_baselines(cfg: CovarianceShrinkageConfig) -> dict[tuple[object, ...], dict[str, object]]:
    path = cfg.prior_calibration_artifact_root / "tables" / "calibrated_prior_gap_summary.csv"
    if not path.exists():
        raise ProtocolError(f"Missing calibrated prior gap summary: {path}")
    required = {
        "experiment_seed",
        "heldout_center",
        "expert_id",
        "expert_pool_type",
        "variant_id",
        "replicate_seed",
        "row_role",
        "decision_cell_set_hash",
        "calibrated_prior_bacc",
        "total_calibrated_prior_cvae_gap",
        "status",
    }
    out: dict[tuple[object, ...], dict[str, object]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ProtocolError(f"Calibrated prior gap summary is missing fields: {sorted(missing)}")
        for row in reader:
            if row.get("status") != "ok":
                continue
            if row.get("variant_id") not in {PRIMARY_VARIANT, UNION_VARIANT}:
                continue
            if row.get("row_role") not in {ROW_DIAG_PRIOR, ROW_UNION_DIAG_PRIOR, ROW_FULL_COV_PRIOR}:
                continue
            key = _reference_key(
                row["experiment_seed"],
                row["heldout_center"],
                row["expert_id"],
                row["expert_pool_type"],
                row["variant_id"],
                row["replicate_seed"],
            )
            out[(key, str(row["row_role"]))] = dict(row)
    return out


def _load_covariance_confirmation_baselines(cfg: CovarianceShrinkageConfig) -> dict[tuple[object, ...], dict[str, object]]:
    path = cfg.covariance_confirmation_artifact_root / "tables" / "covariance_prior_gap_summary.csv"
    if not path.exists():
        raise ProtocolError(f"Missing covariance prior gap summary: {path}")
    required = {
        "experiment_seed",
        "heldout_center",
        "expert_id",
        "expert_pool_type",
        "variant_id",
        "replicate_seed",
        "row_role",
        "decision_cell_set_hash",
        "bacc",
        "total_covariance_prior_gap",
        "status",
    }
    out: dict[tuple[object, ...], dict[str, object]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ProtocolError(f"Covariance prior gap summary is missing fields: {sorted(missing)}")
        for row in reader:
            if row.get("status") != "ok":
                continue
            if row.get("variant_id") not in {PRIMARY_VARIANT, UNION_VARIANT}:
                continue
            if row.get("row_role") != ROW_COVARIANCE_CONFIRMATION_PRIOR:
                continue
            key = _reference_key(
                row["experiment_seed"],
                row["heldout_center"],
                row["expert_id"],
                row["expert_pool_type"],
                row["variant_id"],
                row["replicate_seed"],
            )
            out[key] = dict(row)
    if not out:
        raise ProtocolError("Covariance prior confirmation artifact did not contain alpha010 reference rows.")
    return out


def _reference_for_runtime(
    references: Mapping[tuple[object, ...], ImportedReference],
    runtime: VariantRuntime,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
) -> ImportedReference:
    key = _reference_key(
        experiment_seed,
        heldout_center,
        runtime.expert_id,
        runtime.variant.expert_pool_type,
        runtime.variant.variant_id,
        replicate_seed,
    )
    ref = references.get(key)
    if ref is None:
        raise ProtocolError(f"Missing frozen covariance-prior reference for {key}.")
    return ref


def _reference_key(
    experiment_seed: object,
    heldout_center: object,
    expert_id: object,
    expert_pool_type: object,
    variant_id: object,
    replicate_seed: object,
) -> tuple[object, ...]:
    return tuple(str(v) for v in (experiment_seed, heldout_center, expert_id, expert_pool_type, variant_id, replicate_seed))


def _decision_cell_id(experiment_seed: object, heldout_center: object, expert_id: object) -> str:
    return f"{experiment_seed}|{heldout_center}|{expert_id}"


def _variant_real_stratum(value: object) -> str:
    bacc = _float(value)
    if not math.isfinite(bacc):
        return ""
    if bacc >= 0.80:
        return "variant_real_high"
    if bacc >= 0.75:
        return "variant_real_viable"
    if bacc >= 0.65:
        return "variant_real_borderline"
    return "variant_real_weak"


def _empty_reference(experiment_seed: int, heldout_center: str, runtime: VariantRuntime) -> ImportedReference:
    return ImportedReference(
        reference_real_budget_bacc=math.nan,
        variant_real_budget_bacc=math.nan,
        source_utility_stratum_reference="",
        imported_standard_prior_bacc=math.nan,
        imported_diag_prior_bacc=math.nan,
        imported_full_cov_diagnostic_bacc=math.nan,
        imported_total_prior_cvae_gap=math.nan,
        imported_diag_prior_gap=math.nan,
        imported_full_cov_diagnostic_gap=math.nan,
        source_budget_index_hash=NA,
        decision_cell_id=_decision_cell_id(experiment_seed, heldout_center, runtime.expert_id),
        decision_cell_set_hash="",
    )


def _gap_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [dict(row) for row in rows if row.get("status") == "ok"]


def _augment_paired_baseline_fields(rows: list[dict[str, object]]) -> None:
    ok = [row for row in rows if row.get("status") == "ok"]
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in ok:
        key = (row["experiment_seed"], row["heldout_center"], row["expert_id"], row["expert_pool_type"], row["replicate_seed"])
        grouped.setdefault(key, []).append(row)
    for subset in grouped.values():
        standard = _find_method(subset, ROW_STANDARD_PRIOR)
        diag = _find_method(subset, ROW_DIAG_REFERENCE)
        alpha010 = _find_method(subset, ROW_ALPHA010_REFERENCE)
        if not (standard and diag and alpha010):
            continue
        standard_bacc = _float(standard["bacc"])
        diag_bacc = _float(diag["bacc"])
        alpha010_bacc = _float(alpha010["bacc"])
        variant_real = _float(alpha010["variant_real_budget_bacc"])
        standard_gap = variant_real - standard_bacc
        diag_gap = variant_real - diag_bacc
        alpha010_gap = variant_real - alpha010_bacc
        for row in subset:
            bacc = _float(row["bacc"])
            total_gap = variant_real - bacc
            row["rerun_standard_prior_bacc"] = standard_bacc
            row["rerun_diag_prior_bacc"] = diag_bacc
            row["rerun_alpha010_prior_bacc"] = alpha010_bacc
            row["delta_bacc_vs_standard_prior"] = bacc - standard_bacc
            row["delta_bacc_vs_diag_prior"] = bacc - diag_bacc
            row["delta_bacc_vs_alpha010"] = bacc - alpha010_bacc
            row["total_shrinkage_prior_gap"] = total_gap
            row["preservation_ratio"] = bacc / variant_real if math.isfinite(bacc) and math.isfinite(variant_real) and variant_real > 0 else math.nan
            row["clipped_preservation_gap"] = max(0.0, total_gap) if math.isfinite(total_gap) else math.nan
            row["gap_reduction_vs_standard_prior"] = standard_gap - total_gap
            row["gap_reduction_vs_diag_prior"] = diag_gap - total_gap
            row["gap_reduction_vs_alpha010"] = alpha010_gap - total_gap


def _find_method(rows: Sequence[Mapping[str, object]], method: str) -> Mapping[str, object] | None:
    for row in rows:
        if row.get("row_role") == method:
            return row
    return None


def _decision(
    rows: Sequence[Mapping[str, object]],
    cfg: CovarianceShrinkageConfig,
    *,
    leakage_status: str,
    decision_cell_set_hash: str,
) -> dict[str, object]:
    primary = _decision_rows(rows, method=ROW_SHRINKAGE075, pool_type=POOL_PER_SOURCE)
    alpha010 = _decision_rows(rows, method=ROW_ALPHA010_REFERENCE, pool_type=POOL_PER_SOURCE, diagnostic=True)
    alpha090 = _decision_rows(rows, method=ROW_SHRINKAGE090, pool_type=POOL_PER_SOURCE, diagnostic=True)
    original_9 = _original_9_rows(rows, ROW_SHRINKAGE075)
    stats = _prior_stats(primary)
    alpha010_stats = _prior_stats(alpha010)
    alpha090_stats = _prior_stats(alpha090)
    original_stats = _prior_stats(original_9)
    numeric_core_pass = _numeric_core_pass(stats, alpha010_stats=alpha010_stats, leakage_status=leakage_status)
    fallback_used = _primary_fallback_used(primary)
    diag_preferred = (
        _float(stats["mean_delta_bacc_vs_diag_prior"]) < 0.02
        or _float(stats["covariance_beats_diag_cell_fraction"]) < 0.75
        or _float(stats["covariance_beats_diag_center_fraction"]) < 0.80
    )
    verdict = "PRIOR_STILL_UNSTABLE"
    if leakage_status != "PASS" or _baseline_repro_failed(rows, cfg) or int(stats["n_decision_cells"]) < int(cfg.min_decision_cells):
        verdict = "PROTOCOL_FAIL"
    elif numeric_core_pass and not fallback_used:
        if _float(stats["mean_bacc"]) >= _float(alpha010_stats["mean_bacc"]):
            verdict = "SHRINKAGE075_DOMINANCE_PASS_DIAGNOSTIC"
        else:
            verdict = "SHRINKAGE075_STABILITY_PASS_DIAGNOSTIC"
    elif numeric_core_pass and fallback_used:
        verdict = "SHRINKAGE075_HYBRID_PASS_DIAGNOSTIC"
    elif diag_preferred:
        verdict = "DIAGONAL_AGGREGATE_PRIOR_PREFERRED"
    elif _numeric_core_pass(alpha090_stats, alpha010_stats=alpha010_stats, leakage_status=leakage_status):
        verdict = "ALPHA090_DIAGNOSTIC_LEAD"
    elif _source_pool_passes(rows, alpha010_stats=alpha010_stats):
        verdict = "SOURCE_POOL_ONLY"
    elif _float(stats["mean_delta_bacc_vs_alpha010"]) >= -0.015 or _float(stats["mean_delta_bacc_vs_standard_prior"]) >= 0.05:
        verdict = "SHRINKAGE075_PARTIAL"

    flags = []
    if _center3_still_weak(primary):
        flags.append("CENTER3_STILL_WEAK")
    if _float(stats["worst_delta_vs_diag_prior"]) < -0.03:
        flags.append("NEGATIVE_TAIL_VS_DIAGONAL")
    if _float(stats["mean_delta_bacc_vs_alpha010"]) < 0.0:
        flags.append("ALPHA010_OUTPERFORMS_PRIMARY")
    if _float(stats["mean_delta_bacc_vs_diag_prior"]) < 0.0:
        flags.append("DIAGONAL_OUTPERFORMS_PRIMARY")
    if fallback_used:
        flags.append("FALLBACK_USED_IN_PRIMARY")
    if _source_pool_passes(rows, alpha010_stats=alpha010_stats):
        flags.append("SOURCE_POOL_STRONG")
    if int(original_stats["n_decision_cells"]) != 9:
        flags.append("ORIGINAL_9_STRESS_SCHEMA_WARNING")
    if _float(original_stats["mean_bacc"]) < 0.75 or _float(original_stats["worst_delta_vs_diag_prior"]) < -0.03:
        flags.append("ORIGINAL_9_STRESS_FAIL")

    return {
        "primary_verdict": verdict,
        "diagnostic_flags": "|".join(flags),
        "decision_cell_set_hash": decision_cell_set_hash,
        **stats,
        "alpha010_mean_bacc": alpha010_stats["mean_bacc"],
        "alpha010_prior_preservation_failure_count": alpha010_stats["prior_preservation_failure_count"],
        "alpha010_worst_delta_vs_diag": alpha010_stats["worst_delta_vs_diag_prior"],
        "alpha010_center3_mean_bacc": alpha010_stats["center3_mean_bacc"],
        "alpha010_center3_min_cell_bacc": alpha010_stats["center3_min_cell_bacc"],
        "original_9_decision_cells": original_stats["n_decision_cells"],
        "original_9_mean_bacc": original_stats["mean_bacc"],
        "original_9_worst_delta_vs_diag": original_stats["worst_delta_vs_diag_prior"],
    }


def _decision_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    method: str,
    pool_type: str,
    diagnostic: bool = False,
) -> list[Mapping[str, object]]:
    return [
        row for row in rows
        if row.get("row_role") == method
        and row.get("expert_pool_type") == pool_type
        and row.get("variant_id") == (PRIMARY_VARIANT if pool_type == POOL_PER_SOURCE else UNION_VARIANT)
        and row.get("status") == "ok"
        and _float(row.get("variant_real_budget_bacc", math.nan)) >= 0.80
        and (diagnostic or row.get("selection_source") == PRIMARY_SELECTION)
    ]


def _original_9_rows(rows: Sequence[Mapping[str, object]], method: str) -> list[Mapping[str, object]]:
    return [
        row for row in rows
        if row.get("row_role") == method
        and row.get("expert_pool_type") == POOL_PER_SOURCE
        and row.get("variant_id") == PRIMARY_VARIANT
        and row.get("status") == "ok"
        and row.get("source_utility_stratum_reference") in {"medium", "high"}
        and row.get("selection_source") == PRIMARY_SELECTION
    ]


def _prior_stats(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    grouped = _replicate_averaged(rows)
    by_seed: dict[str, list[Mapping[str, object]]] = {}
    centers = set()
    experts = set()
    center_counts: dict[str, int] = {}
    for row in grouped:
        by_seed.setdefault(str(row["experiment_seed"]), []).append(row)
        centers.add(str(row["heldout_center"]))
        experts.add(str(row["expert_id"]))
        center_counts[str(row["heldout_center"])] = center_counts.get(str(row["heldout_center"]), 0) + 1
    seed_bacc = [_center_equal_from_grouped(values, "bacc") for values in by_seed.values()]
    center_delta = _per_center_mean(grouped, "delta_bacc_vs_diag_prior")
    center_bacc = _per_center_mean(grouped, "bacc")
    center_gap = _per_center_mean(grouped, "clipped_preservation_gap")
    center_ratio = _per_center_mean(grouped, "preservation_ratio")
    center_alpha010 = _per_center_mean(grouped, "delta_bacc_vs_alpha010")
    return {
        "n_raw_rows": len(rows),
        "n_decision_cells": len(grouped),
        "n_experiment_seeds": len(by_seed),
        "n_heldout_centers": len(centers),
        "n_experts": len(experts),
        "min_cells_per_center": min(center_counts.values()) if center_counts else 0,
        "mean_bacc": _center_equal_from_grouped(grouped, "bacc"),
        "min_cell_bacc": _min_field(grouped, "bacc"),
        "mean_total_shrinkage_prior_gap": _center_equal_from_grouped(grouped, "total_shrinkage_prior_gap"),
        "mean_clipped_preservation_gap": _center_equal_from_grouped(grouped, "clipped_preservation_gap"),
        "mean_preservation_ratio": _center_equal_from_grouped(grouped, "preservation_ratio"),
        "mean_delta_bacc_vs_standard_prior": _center_equal_from_grouped(grouped, "delta_bacc_vs_standard_prior"),
        "mean_delta_bacc_vs_diag_prior": _center_equal_from_grouped(grouped, "delta_bacc_vs_diag_prior"),
        "mean_delta_bacc_vs_alpha010": _center_equal_from_grouped(grouped, "delta_bacc_vs_alpha010"),
        "mean_gap_reduction_vs_standard_prior": _mean_field(grouped, "gap_reduction_vs_standard_prior"),
        "mean_gap_reduction_vs_diag_prior": _mean_field(grouped, "gap_reduction_vs_diag_prior"),
        "shrinkage_prior_seed_std": _std(seed_bacc),
        "per_seed_bacc": json.dumps({seed: _mean_field(values, "bacc") for seed, values in sorted(by_seed.items())}, sort_keys=True),
        "per_center_bacc": json.dumps(center_bacc, sort_keys=True),
        "per_center_total_shrinkage_prior_gap": json.dumps(_per_center_mean(grouped, "total_shrinkage_prior_gap"), sort_keys=True),
        "per_center_delta_bacc_vs_diag_prior": json.dumps(center_delta, sort_keys=True),
        "per_center_delta_bacc_vs_alpha010": json.dumps(center_alpha010, sort_keys=True),
        "covariance_beats_diag_cell_fraction": _beats_diag_fraction(grouped),
        "covariance_beats_diag_center_fraction": _beats_diag_center_fraction(center_delta),
        "worst_delta_vs_diag_prior": _min_field(grouped, "delta_bacc_vs_diag_prior"),
        "min_center_delta_vs_diag_prior": min(center_delta.values()) if center_delta else math.nan,
        "min_center_mean_bacc": min(center_bacc.values()) if center_bacc else math.nan,
        "n_cells_bacc_lt_060": sum(1 for row in grouped if _float(row["bacc"]) < 0.60),
        "n_cells_gap_gt_008": sum(1 for row in grouped if _float(row["clipped_preservation_gap"]) > 0.08),
        "prior_preservation_failure_count": _prior_preservation_failure_count(grouped),
        "center3_mean_bacc": _center_value(center_bacc, "3"),
        "center3_min_cell_bacc": _center_min(grouped, "3", "bacc"),
        "n_center3_failures": sum(
            1 for row in grouped
            if str(row.get("heldout_center")) == "3" and (_float(row["bacc"]) < 0.60 or _float(row["clipped_preservation_gap"]) > 0.08)
        ),
        "original_9_decision_cells": len({row["decision_cell_id"] for row in grouped if row.get("source_utility_stratum_reference") in {"medium", "high"}}),
        "original_9_mean_bacc": _mean_field([row for row in grouped if row.get("source_utility_stratum_reference") in {"medium", "high"}], "bacc"),
        "original_9_worst_delta_vs_diag": _min_field([row for row in grouped if row.get("source_utility_stratum_reference") in {"medium", "high"}], "delta_bacc_vs_diag_prior"),
    }


def _replicate_averaged(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["experiment_seed"]), str(row["heldout_center"]), str(row["expert_id"])), []).append(row)
    fields = (
        "bacc",
        "total_shrinkage_prior_gap",
        "preservation_ratio",
        "clipped_preservation_gap",
        "delta_bacc_vs_standard_prior",
        "delta_bacc_vs_diag_prior",
        "delta_bacc_vs_alpha010",
        "gap_reduction_vs_standard_prior",
        "gap_reduction_vs_diag_prior",
        "gap_reduction_vs_alpha010",
    )
    out = []
    for (seed, center, expert), subset in groups.items():
        row = {"experiment_seed": seed, "heldout_center": center, "expert_id": expert}
        row["decision_cell_id"] = str(subset[0].get("decision_cell_id", _decision_cell_id(seed, center, expert)))
        row["source_utility_stratum_reference"] = str(subset[0].get("source_utility_stratum_reference", ""))
        row.update({field: _mean_field(subset, field) for field in fields})
        row["covariance_fallback_used"] = any(str(v.get("covariance_fallback_used")) == "True" for v in subset)
        out.append(row)
    return out


def _numeric_core_pass(
    stats: Mapping[str, object],
    *,
    alpha010_stats: Mapping[str, object],
    leakage_status: str,
) -> bool:
    return (
        _float(stats["n_decision_cells"]) >= 30
        and _float(stats["n_heldout_centers"]) >= 5
        and _min_cells_per_center(stats) >= 3
        and _float(stats["mean_bacc"]) >= 0.85
        and _float(stats["mean_clipped_preservation_gap"]) <= 0.08
        and _float(stats["mean_preservation_ratio"]) >= 0.92
        and _float(stats["shrinkage_prior_seed_std"]) <= 0.07
        and _float(stats["mean_delta_bacc_vs_standard_prior"]) >= 0.05
        and _float(stats["mean_delta_bacc_vs_diag_prior"]) >= 0.02
        and _float(stats["mean_delta_bacc_vs_alpha010"]) >= -0.015
        and _float(stats["covariance_beats_diag_cell_fraction"]) >= 0.75
        and _float(stats["covariance_beats_diag_center_fraction"]) >= 0.80
        and _float(stats["worst_delta_vs_diag_prior"]) >= -0.03
        and _float(stats["min_center_delta_vs_diag_prior"]) >= -0.02
        and leakage_status == "PASS"
        and _float(stats["min_cell_bacc"]) >= 0.60
        and _float(stats["min_center_mean_bacc"]) >= 0.75
        and int(stats["prior_preservation_failure_count"]) <= int(alpha010_stats["prior_preservation_failure_count"])
    )


def _standard_prior_already_passes(stats: Mapping[str, object]) -> bool:
    return (
        _float(stats["mean_bacc"]) >= 0.75
        and _float(stats["mean_total_shrinkage_prior_gap"]) <= 0.08
        and _float(stats["shrinkage_prior_seed_std"]) <= 0.07
    )


def _baseline_repro_failed(rows: Sequence[Mapping[str, object]], cfg: CovarianceShrinkageConfig) -> bool:
    for row in rows:
        if row.get("status") != "ok":
            continue
        bacc = _float(row["bacc"])
        if row["row_role"] == ROW_STANDARD_PRIOR and abs(bacc - _float(row["imported_standard_prior_bacc"])) > cfg.standard_prior_repro_abs_tol_bacc:
            return True
        if row["row_role"] == ROW_DIAG_REFERENCE and abs(bacc - _float(row["imported_diag_prior_bacc"])) > cfg.diag_prior_repro_abs_tol_bacc:
            return True
        if row["row_role"] == ROW_ALPHA010_REFERENCE and abs(bacc - _float(row["imported_full_cov_diagnostic_bacc"])) > cfg.full_cov_diagnostic_repro_abs_tol_bacc:
            return True
    return False


def _primary_fallback_used(rows: Sequence[Mapping[str, object]]) -> bool:
    return any(str(row.get("covariance_fallback_used")) == "True" for row in rows)


def _cell_collapse(rows: Sequence[Mapping[str, object]]) -> bool:
    return any(_float(row["bacc"]) < 0.60 for row in rows)


def _center_collapse(rows: Sequence[Mapping[str, object]]) -> bool:
    grouped = _replicate_averaged(rows)
    return any(value < 0.60 for value in _per_center_mean(grouped, "bacc").values())


def _source_pool_passes(rows: Sequence[Mapping[str, object]], *, alpha010_stats: Mapping[str, object]) -> bool:
    stats = _prior_stats(_decision_rows(rows, method=ROW_SHRINKAGE075, pool_type=POOL_SOURCE_UNION, diagnostic=True))
    return (
        _float(stats["mean_bacc"]) >= 0.85
        and _float(stats["mean_clipped_preservation_gap"]) <= 0.08
        and _float(stats["shrinkage_prior_seed_std"]) <= 0.07
        and _float(stats["mean_delta_bacc_vs_standard_prior"]) >= 0.05
        and _float(stats["mean_delta_bacc_vs_diag_prior"]) >= 0.02
        and _float(stats["min_center_mean_bacc"]) >= 0.75
    )


def _json_center_collapse(payload: object) -> bool:
    try:
        values = json.loads(str(payload)).values()
    except Exception:
        return False
    return any(float(value) < 0.60 for value in values)


def _beats_diag_fraction(rows: Sequence[Mapping[str, object]]) -> float:
    finite = [_float(row["delta_bacc_vs_diag_prior"]) for row in rows if math.isfinite(_float(row["delta_bacc_vs_diag_prior"]))]
    return nanmean([1.0 if value > 0.0 else 0.0 for value in finite]) if finite else math.nan


def _beats_diag_center_fraction(center_delta: Mapping[str, float]) -> float:
    values = [float(value) for value in center_delta.values() if math.isfinite(float(value))]
    return nanmean([1.0 if value > 0.0 else 0.0 for value in values]) if values else math.nan


def _prior_preservation_failure_count(rows: Sequence[Mapping[str, object]]) -> int:
    return sum(
        1 for row in rows
        if _float(row.get("bacc", math.nan)) < 0.60 or _float(row.get("clipped_preservation_gap", math.nan)) > 0.08
    )


def _center_value(values: Mapping[str, float], center: str) -> float:
    return float(values[center]) if center in values else math.nan


def _center_min(rows: Sequence[Mapping[str, object]], center: str, field: str) -> float:
    return _min_field([row for row in rows if str(row.get("heldout_center")) == str(center)], field)


def _center3_still_weak(rows: Sequence[Mapping[str, object]]) -> bool:
    grouped = _replicate_averaged(rows)
    center3 = [row for row in grouped if str(row.get("heldout_center")) == "3"]
    return bool(center3) and (_mean_field(center3, "bacc") < 0.75 or _min_field(center3, "bacc") < 0.60)


def _center_equal_from_grouped(rows: Sequence[Mapping[str, object]], field: str) -> float:
    seed_center: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        seed_center.setdefault((str(row["experiment_seed"]), str(row["heldout_center"])), []).append(row)
    by_seed: dict[str, list[float]] = {}
    for (seed, _center), subset in seed_center.items():
        by_seed.setdefault(seed, []).append(_mean_field(subset, field))
    return nanmean([nanmean(values) for values in by_seed.values()]) if by_seed else math.nan


def _min_cells_per_center(stats: Mapping[str, object]) -> int:
    return int(_float(stats.get("min_cells_per_center", 0)))


def _mean_field(rows: Sequence[Mapping[str, object]], field: str) -> float:
    return nanmean([_float(row[field]) for row in rows if field in row and str(row.get(field, "")) not in {"", NA}])


def _min_field(rows: Sequence[Mapping[str, object]], field: str) -> float:
    values = [_float(row[field]) for row in rows if field in row and math.isfinite(_float(row[field]))]
    return min(values) if values else math.nan


def _std(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if len(finite) < 2:
        return 0.0
    avg = sum(finite) / float(len(finite))
    return math.sqrt(sum((value - avg) ** 2 for value in finite) / float(len(finite)))


def _per_center_mean(rows: Sequence[Mapping[str, object]], field: str) -> dict[str, float]:
    groups: dict[str, list[float]] = {}
    for row in rows:
        value = _float(row[field])
        if math.isfinite(value):
            groups.setdefault(str(row["heldout_center"]), []).append(value)
    return {center: nanmean(values) for center, values in sorted(groups.items())}


def _parameter_health(params: CovariancePriorParameters) -> dict[str, object]:
    if not params.classes:
        return {
            "covariance_fallback_used": False,
            "fallback_reason": "",
            **{key: math.nan for key in _health_columns()},
        }
    rows = list(params.classes.values())
    reasons = sorted({row.fallback_reason for row in rows if row.fallback_reason})
    out: dict[str, object] = {
        "covariance_fallback_used": any(row.fallback_used for row in rows),
        "fallback_reason": "|".join(reasons),
    }
    for key in _health_columns():
        if key == "class_min_n_encoded_records":
            continue
        values = [float(getattr(row, key)) for row in rows]
        if key == "num_eigenvalues_clipped":
            out[key] = int(sum(values))
        else:
            out[key] = nanmean(values)
    out["class_min_n_encoded_records"] = min(int(row.n_records) for row in rows)
    return out


def _health_columns() -> tuple[str, ...]:
    return (
        "trace_before_shrinkage",
        "trace_after_shrinkage",
        "condition_number_after_clip",
        "num_eigenvalues_clipped",
        "min_eigenvalue_before_clip",
        "min_eigenvalue_after_clip",
        "max_eigenvalue_after_clip",
        "mean_diag_variance",
        "mean_offdiag_abs_correlation",
        "covariance_effective_rank",
        "offdiag_frobenius_ratio",
        "trace_ratio_vs_diag",
        "class_min_n_encoded_records",
    )


def _low_stratum_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [
        dict(row) for row in rows
        if row.get("row_role") == ROW_SHRINKAGE075
        and row.get("expert_pool_type") == POOL_PER_SOURCE
        and _float(row.get("variant_real_budget_bacc", math.nan)) < 0.80
    ]


def _source_pool_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    subset = _decision_rows(rows, method=ROW_SHRINKAGE075, pool_type=POOL_SOURCE_UNION, diagnostic=True)
    return [{"expert_pool_type": POOL_SOURCE_UNION, "prior_method": ROW_SHRINKAGE075, "selection_source": DIAGNOSTIC_SELECTION, **_prior_stats(subset)}]


def _repair_runtime_config(cfg: CovarianceShrinkageConfig, root: Path) -> RepairConfig:
    return RepairConfig(
        name="virchow2_cvae_preservation_repair_v1",
        artifact_root=root,
        feature_cache_root=cfg.feature_cache_root,
        experiment_seeds=cfg.experiment_seeds,
        heldout_centers=cfg.heldout_centers,
        replicate_seeds=cfg.replicate_seeds,
        synthetic_per_class_total=cfg.synthetic_per_class_total,
        primary_variant=cfg.primary_variant,
        min_decision_rows=cfg.min_decision_cells,
        variants=(_per_source_variant(), _union_variant()),
        source_probe=SourceProbeConfig(
            type="torch_linear_classifier",
            optimizer="adamw",
            learning_rate=0.001,
            weight_decay=0.0001,
            epochs=1,
            batch_size=128,
            class_weight="balanced",
            early_stopping=False,
        ),
        classifier_type=cfg.classifier_type,
        classifier_solver=cfg.classifier_solver,
        classifier_c=cfg.classifier_c,
        classifier_max_iter=cfg.classifier_max_iter,
        classifier_class_weight=cfg.classifier_class_weight,
        classifier_seed=cfg.classifier_seed,
    )


def _alpha_comparison_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    out = []
    for method in ROW_ROLES:
        subset = [
            row for row in rows
            if row.get("row_role") == method
            and row.get("expert_pool_type") == POOL_PER_SOURCE
            and row.get("status") == "ok"
            and _float(row.get("variant_real_budget_bacc", math.nan)) >= 0.80
        ]
        if not subset:
            continue
        out.append({"prior_method": method, **_prior_stats(subset)})
    return out


def _high_real_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    primary = [
        row for row in rows
        if row.get("row_role") == ROW_SHRINKAGE075
        and row.get("expert_pool_type") == POOL_PER_SOURCE
        and row.get("status") == "ok"
        and _float(row.get("variant_real_budget_bacc", math.nan)) >= 0.80
    ]
    return {"population": "conditional_high_real_cells", "prior_method": ROW_SHRINKAGE075, **_prior_stats(primary)}


def _original_9_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    primary = [
        row for row in rows
        if row.get("row_role") == ROW_SHRINKAGE075
        and row.get("expert_pool_type") == POOL_PER_SOURCE
        and row.get("status") == "ok"
        and row.get("source_utility_stratum_reference") in {"medium", "high"}
    ]
    stats = _prior_stats(primary)
    ids = sorted({str(row.get("decision_cell_id", "")) for row in primary if row.get("decision_cell_id")})
    missing_warning = len(ids) != 9
    return {
        "population": "original_9_decision_cells_stress",
        "prior_method": ROW_SHRINKAGE075,
        "n_original_cells": len(ids),
        "original_9_stress_schema_warning": missing_warning,
        "decision_cell_ids": json.dumps(ids),
        **stats,
    }


def _variant_real_stratum_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    out = []
    for stratum in ("variant_real_high", "variant_real_viable", "variant_real_borderline", "variant_real_weak"):
        subset = [
            row for row in rows
            if row.get("row_role") == ROW_SHRINKAGE075
            and row.get("expert_pool_type") == POOL_PER_SOURCE
            and row.get("status") == "ok"
            and row.get("variant_real_stratum") == stratum
        ]
        out.append({"variant_real_stratum": stratum, **_prior_stats(subset)})
    return out


def _covariance_health_by_alpha(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    out = []
    for method in (ROW_ALPHA010_REFERENCE, ROW_SHRINKAGE050, ROW_SHRINKAGE075, ROW_SHRINKAGE090, ROW_DIAG_REFERENCE):
        subset = [row for row in rows if row.get("row_role") == method and row.get("status") == "ok"]
        if not subset:
            continue
        row = {
            "prior_method": method,
            "covariance_shrinkage_alpha": _mean_field(subset, "covariance_shrinkage_alpha"),
        }
        for field in _health_columns():
            row[field] = _mean_field(subset, field)
        out.append(row)
    return out


def _fallback_stability_rows(
    rows: Sequence[Mapping[str, object]],
    fallback_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    primary = [
        row for row in rows
        if row.get("row_role") == ROW_SHRINKAGE075
        and row.get("status") == "ok"
        and str(row.get("covariance_fallback_used")) == "True"
    ]
    out = [dict(row) for row in fallback_rows]
    for row in primary:
        out.append(
            {
                "experiment_seed": row.get("experiment_seed"),
                "heldout_center": row.get("heldout_center"),
                "expert_id": row.get("expert_id"),
                "expert_pool_type": row.get("expert_pool_type"),
                "variant_id": row.get("variant_id"),
                "prior_method": row.get("prior_method"),
                "bacc": row.get("bacc"),
                "variant_real_budget_bacc": row.get("variant_real_budget_bacc"),
                "fallback_reason": row.get("fallback_reason"),
                "covariance_fallback_used": row.get("covariance_fallback_used"),
            }
        )
    return out


def _write_artifacts(
    root: Path,
    cfg: CovarianceShrinkageConfig,
    *,
    matrix_rows: Sequence[Mapping[str, object]],
    gap_rows: Sequence[Mapping[str, object]],
    parameter_rows: Sequence[Mapping[str, object]],
    fallback_rows: Sequence[Mapping[str, object]],
    low_stratum_rows: Sequence[Mapping[str, object]],
    source_pool_rows: Sequence[Mapping[str, object]],
    manifest_rows: Sequence[Mapping[str, object]],
    decision: Mapping[str, object],
    protocol_violations: Sequence[str],
    target_expert_excluded: bool,
) -> None:
    write_csv_rows(root / "tables" / "shrinkage_prior_downstream_matrix.csv", matrix_rows, columns=_matrix_columns())
    write_csv_rows(root / "tables" / "shrinkage_prior_gap_summary.csv", gap_rows)
    write_csv_rows(root / "tables" / "shrinkage_alpha_comparison.csv", _alpha_comparison_rows(gap_rows))
    write_csv_rows(root / "tables" / "high_real_viability_summary.csv", [_high_real_summary(gap_rows)])
    write_csv_rows(root / "tables" / "original_9_stress_summary.csv", [_original_9_summary(gap_rows)])
    write_csv_rows(root / "tables" / "variant_real_stratum_summary.csv", _variant_real_stratum_rows(gap_rows))
    write_csv_rows(root / "tables" / "covariance_health_by_alpha.csv", _covariance_health_by_alpha(gap_rows))
    write_csv_rows(root / "tables" / "fallback_stability_audit.csv", _fallback_stability_rows(gap_rows, fallback_rows))
    write_csv_rows(root / "tables" / "source_pool_shrinkage_summary.csv", source_pool_rows)
    write_csv_rows(root / "manifests" / "covariance_shrinkage_model_manifest.csv", manifest_rows)
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
            "schema_version": "cvae_rebuild_covariance_shrinkage_stability_protocol_manifest_v1",
            "experiment_name": cfg.name,
            "experiment_type": "covariance_shrinkage_stability_diagnostic",
            "primary_variant": cfg.primary_variant,
            "primary_method": cfg.primary_method,
            "decision_cell_set_hash": decision.get("decision_cell_set_hash", ""),
            "row_roles": list(ROW_ROLES),
            "target_support_labels_for_selection": False,
            "target_eval_labels_for_scoring_only": True,
            "target_expert_excluded": target_expert_excluded,
            "source_union_diagnostic_only": True,
            "claim_boundary": "covariance-shrinkage sampled-feature utility diagnostic only; no routing or formal privacy claim",
        },
    )
    _write_decision_summary(root, decision, leakage_status=leakage.status)
    write_json(root / "run_config_resolved.yaml", _resolved_config(cfg))


def _matrix_columns() -> tuple[str, ...]:
    return (
        "experiment_seed",
        "heldout_center",
        "expert_id",
        "expert_pool_type",
        "variant_id",
        "row_role",
        "prior_method",
        "replicate_seed",
        "decision_cell_id",
        "decision_cell_set_hash",
        "source_utility_stratum_reference",
        "variant_real_stratum",
        "variant_real_budget_bacc",
        "imported_standard_prior_bacc",
        "imported_diag_prior_bacc",
        "imported_full_cov_diagnostic_bacc",
        "rerun_standard_prior_bacc",
        "rerun_diag_prior_bacc",
        "rerun_alpha010_prior_bacc",
        "bacc",
        "macro_f1",
        "delta_bacc_vs_standard_prior",
        "delta_bacc_vs_diag_prior",
        "delta_bacc_vs_alpha010",
        "total_shrinkage_prior_gap",
        "preservation_ratio",
        "clipped_preservation_gap",
        "gap_reduction_vs_standard_prior",
        "gap_reduction_vs_diag_prior",
        "gap_reduction_vs_alpha010",
        "covariance_shrinkage_alpha",
        "covariance_eigenvalue_floor",
        "covariance_fallback_used",
        "fallback_reason",
        "prior_fit_row_ids_hash",
        "prior_parameter_hash",
        "generated_features_hash",
        "prediction_hash",
        "selection_source",
        "status",
        "error_message",
        "budget_match_type",
        *_health_columns(),
    )


def _write_decision_summary(root: Path, decision: Mapping[str, object], *, leakage_status: str) -> None:
    text = "\n".join(
        [
            "# Virchow2-CVAE Covariance Shrinkage Stability v1",
            "",
            "## Summary",
            "",
            f"- Primary method: `{PRIMARY_SHRINKAGE_METHOD}`",
            f"- Primary verdict: `{decision.get('primary_verdict', 'PRIOR_STILL_UNSTABLE')}`",
            f"- Diagnostic flags: `{decision.get('diagnostic_flags', '')}`",
            f"- Center-equal mean BACC: {_format_float(decision.get('mean_bacc'))}",
            f"- Mean clipped preservation gap: {_format_float(decision.get('mean_clipped_preservation_gap'))}",
            f"- Mean preservation ratio: {_format_float(decision.get('mean_preservation_ratio'))}",
            f"- Delta BACC vs standard prior: {_format_float(decision.get('mean_delta_bacc_vs_standard_prior'))}",
            f"- Delta BACC vs diagonal prior: {_format_float(decision.get('mean_delta_bacc_vs_diag_prior'))}",
            f"- Delta BACC vs alpha010: {_format_float(decision.get('mean_delta_bacc_vs_alpha010'))}",
            f"- Covariance beats diagonal cell fraction: {_format_float(decision.get('covariance_beats_diag_cell_fraction'))}",
            f"- Worst delta vs diagonal prior: {_format_float(decision.get('worst_delta_vs_diag_prior'))}",
            f"- Prior preservation failures: {decision.get('prior_preservation_failure_count', 0)}",
            f"- Alpha010 prior preservation failures: {decision.get('alpha010_prior_preservation_failure_count', 0)}",
            f"- High-real decision cells: {decision.get('n_decision_cells', 0)}",
            f"- Original stress decision cells found: {decision.get('original_9_decision_cells', 0)}",
            f"- Decision-cell hash: `{decision.get('decision_cell_set_hash', '')}`",
            f"- Leakage status: `{leakage_status}`",
            "",
            "## Interpretation Guide",
            "",
            "- `SHRINKAGE075_DOMINANCE_PASS_DIAGNOSTIC`: alpha075 improves mean and stability.",
            "- `SHRINKAGE075_STABILITY_PASS_DIAGNOSTIC`: alpha075 slightly lowers mean but repairs tail collapse.",
            "- `DIAGONAL_AGGREGATE_PRIOR_PREFERRED`: alpha075 behaves like diagonal or does not beat it safely.",
            "- `PRIOR_STILL_UNSTABLE`: alpha075 remains unstable.",
            "",
            "## Claim Boundary",
            "",
            "This slice diagnoses source-only diagonal-target covariance shrinkage for sampled-feature downstream utility.",
            "It does not evaluate routing, support-NELBO selection, metadata selection, top-k composition, or formal privacy.",
            "",
            "PASS does not unlock routing directly.",
            "PASS unlocks broader sampled-feature utility confirmation.",
            "Routing is reconsidered only after sampled-feature utility is stable across centers, seeds, and source strata.",
            "",
        ]
    )
    path = root / "reports" / "decision_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _resolved_config(cfg: CovarianceShrinkageConfig) -> dict[str, object]:
    return {
        "name": cfg.name,
        "artifact_root": str(cfg.artifact_root),
        "repair_artifact_root": str(cfg.repair_artifact_root),
        "sampling_artifact_root": str(cfg.sampling_artifact_root),
        "prior_calibration_artifact_root": str(cfg.prior_calibration_artifact_root),
        "covariance_confirmation_artifact_root": str(cfg.covariance_confirmation_artifact_root),
        "covariance_viability_artifact_root": str(cfg.covariance_viability_artifact_root),
        "feature_cache_root": str(cfg.feature_cache_root),
        "experiment_seeds": list(cfg.experiment_seeds),
        "heldout_centers": list(cfg.heldout_centers),
        "replicate_seeds": list(cfg.replicate_seeds),
        "synthetic_per_class_total": cfg.synthetic_per_class_total,
        "primary_variant": cfg.primary_variant,
        "min_decision_cells": cfg.min_decision_cells,
        "primary_method": cfg.primary_method,
        "primary_covariance_shrinkage_alpha": cfg.primary_covariance_shrinkage_alpha,
        "diagnostic_covariance_shrinkage_alphas": list(cfg.diagnostic_covariance_shrinkage_alphas),
        "reference_covariance_shrinkage_alpha": cfg.reference_covariance_shrinkage_alpha,
        "diagonal_reference_alpha": cfg.diagonal_reference_alpha,
        "covariance_eigenvalue_floor": cfg.covariance_eigenvalue_floor,
        "full_cov_min_records_per_class": cfg.full_cov_min_records_per_class,
        "fallback_if_under_ranked": cfg.fallback_if_under_ranked,
        "standard_prior_repro_abs_tol_bacc": cfg.standard_prior_repro_abs_tol_bacc,
        "diag_prior_repro_abs_tol_bacc": cfg.diag_prior_repro_abs_tol_bacc,
        "full_cov_diagnostic_repro_abs_tol_bacc": cfg.full_cov_diagnostic_repro_abs_tol_bacc,
        "classifier_type": cfg.classifier_type,
        "classifier_solver": cfg.classifier_solver,
        "classifier_c": cfg.classifier_c,
        "classifier_max_iter": cfg.classifier_max_iter,
        "classifier_class_weight": cfg.classifier_class_weight,
        "classifier_seed": cfg.classifier_seed,
    }
