from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
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
from experiments.preservation.preservation_sampling import (
    DIAGNOSTIC_SELECTION,
    PRIMARY_SELECTION,
    UNION_VARIANT,
    RuntimeSource,
    _manifest_row,
    _per_source_variant,
    _runtime_source,
)
from experiments.prior_diagnostics.prior_calibration import _decode_latents
from core.protocol import ProtocolError, assert_candidate_pool, build_leakage_report
from core.reporting import prepare_artifact_dirs, write_csv_rows, write_json
from experiments.source_union.source_union_gmm_prior import _nearest_neighbor_row, _read_required_csv
from data.splits import candidate_experts


DECENTRALIZED_K16_NAME = "virchow2_cvae_decentralized_k16_gmm_prior_v1"
PRIMARY_DECENTRALIZED_METHOD = "decentralized_exported_k4x4_cc_diag_gmm_k16_late_geom"
ROW_ARITH = "decentralized_exported_k4x4_cc_diag_gmm_k16_late_arith"
ROW_SUPPORT_NELBO = "decentralized_exported_k4x4_cc_diag_gmm_k16_support_nelbo_weighted_geom_diagnostic"
ROW_SINGLE_MEAN = "per_source_exported_k4_cc_diag_gmm_single_expert_mean_reference"
ROW_SINGLE_ORACLE = "per_source_exported_k4_cc_diag_gmm_single_expert_oracle_reference"
ROW_SOURCE_UNION_K16_REFERENCE = "source_union_cc_diag_gmm_k16_prior_sample_reference"
ROW_CENTER_BALANCED_K16_REFERENCE = "source_union_center_balanced_cc_diag_gmm_k16_prior_sample_reference"
ROW_REAL_FEATURE_DENSE_REFERENCE = "real_source_embedding_classifier_dense_reference"
ROW_SHUFFLED_SUMMARY_CONTROL = "decentralized_k16_shuffled_summary_control"
ROW_SHUFFLED_LABEL_CONTROL = "decentralized_k16_shuffled_label_control"
POOL_DECENTRALIZED = "decentralized_source_summary"
SUMMARY_SCHEMA_VERSION = "decentralized_source_local_cc_diag_gmm_k4_summary_v1"
PROTOCOL_WORDING = (
    "This is a data-minimizing, raw-data-free summary-exchange protocol. "
    "It is not a formal differential privacy claim. Exported latent summary "
    "statistics may still contain distributional information derived from private data."
)


@dataclass(frozen=True)
class DecentralizedK16GmmConfig:
    name: str
    artifact_root: Path
    repair_artifact_root: Path
    sampling_artifact_root: Path | None
    prior_calibration_artifact_root: Path | None
    covariance_confirmation_artifact_root: Path | None
    source_union_gmm_artifact_root: Path | None
    balanced_gmm_artifact_root: Path | None
    feature_cache_root: Path
    backbone: str
    experiment_seeds: tuple[int, ...]
    heldout_centers: tuple[str, ...]
    replicate_seeds: tuple[int, ...]
    synthetic_per_class_total: int
    primary_variant: str
    primary_method: str
    local_gmm_components_per_source_class: int
    composed_components_per_class: int
    source_weighting: str
    gmm_covariance_type: str
    gmm_reg_covar: float
    gmm_n_init: int
    gmm_max_iter: int
    min_count_for_k4: int
    min_component_weight: float
    variance_floor: float
    primary_pooling: str
    support_nelbo_enabled: bool
    classifier_type: str
    classifier_solver: str
    classifier_c: float
    classifier_max_iter: int
    classifier_class_weight: str
    classifier_seed: int | None


@dataclass(frozen=True)
class SourceLocalSummary:
    experiment_seed: int
    source_center: str
    class_label: int
    weights: object
    means: object
    diag_vars: object
    source_class_count: int
    effective_component_count: int
    min_component_weight: float
    min_diag_var: float
    component_entropy: float
    all_finite: bool
    gmm_converged: bool
    gmm_n_iter: int
    source_train_log_likelihood: float
    source_inner_bic: float
    summary_path: Path
    summary_hash: str
    fit_row_ids_hash: str
    parameter_hash: str
    expert_config_hash: str
    status: str
    error_message: str
    shuffled_label_control: bool


@dataclass(frozen=True)
class ReferenceValue:
    bacc: float
    macro_f1: float
    status: str
    error_message: str = ""


def load_decentralized_k16_gmm_prior_config(path: str | Path) -> DecentralizedK16GmmConfig:
    source = Path(path).resolve()
    data = _load_mapping(source)
    base_dir = source.parents[2] if len(source.parents) >= 3 else source.parent
    return parse_decentralized_k16_gmm_prior_config(data, base_dir=base_dir)


def parse_decentralized_k16_gmm_prior_config(
    data: Mapping[str, Any],
    *,
    base_dir: str | Path = ".",
) -> DecentralizedK16GmmConfig:
    base = Path(base_dir)
    experiment = _mapping(data, "experiment")
    inputs = _mapping(data, "inputs")
    run = _mapping(data, "run_matrix")
    generation = _mapping(data, "generation")
    gmm = _mapping(data, "decentralized_k16_prior")
    support = data.get("support_nelbo_diagnostic", {})
    if support is None:
        support = {}
    if not isinstance(support, Mapping):
        raise ProtocolError("support_nelbo_diagnostic must be a mapping when provided.")
    classifier = _mapping(data, "classifier")
    cfg = DecentralizedK16GmmConfig(
        name=str(experiment["name"]),
        artifact_root=_path(base, str(experiment["artifact_root"])),
        repair_artifact_root=_path(base, str(inputs["repair_artifact_root"])),
        sampling_artifact_root=_optional_path(base, inputs.get("sampling_artifact_root")),
        prior_calibration_artifact_root=_optional_path(base, inputs.get("prior_calibration_artifact_root")),
        covariance_confirmation_artifact_root=_optional_path(base, inputs.get("covariance_confirmation_artifact_root")),
        source_union_gmm_artifact_root=_optional_path(base, inputs.get("source_union_gmm_artifact_root")),
        balanced_gmm_artifact_root=_optional_path(base, inputs.get("balanced_gmm_artifact_root")),
        feature_cache_root=_path(base, str(inputs["feature_cache_root"])),
        backbone=str(inputs.get("backbone", "")),
        experiment_seeds=tuple(int(v) for v in run["experiment_seeds"]),
        heldout_centers=tuple(str(v) for v in run["heldout_centers"]),
        replicate_seeds=tuple(int(v) for v in run["replicate_seeds"]),
        synthetic_per_class_total=int(generation["synthetic_per_class_total"]),
        primary_variant=str(experiment["primary_variant"]),
        primary_method=str(gmm["primary_method"]),
        local_gmm_components_per_source_class=int(gmm["local_gmm_components_per_source_class"]),
        composed_components_per_class=int(gmm["composed_components_per_class"]),
        source_weighting=str(gmm["source_weighting"]),
        gmm_covariance_type=str(gmm["gmm_covariance_type"]),
        gmm_reg_covar=float(gmm["gmm_reg_covar"]),
        gmm_n_init=int(gmm["gmm_n_init"]),
        gmm_max_iter=int(gmm["gmm_max_iter"]),
        min_count_for_k4=int(gmm["min_count_for_k4"]),
        min_component_weight=float(gmm["min_component_weight"]),
        variance_floor=float(gmm["variance_floor"]),
        primary_pooling=str(gmm["primary_pooling"]),
        support_nelbo_enabled=bool(support.get("enabled", False)),
        classifier_type=str(classifier["type"]),
        classifier_solver=str(classifier["solver"]),
        classifier_c=float(classifier["C"]),
        classifier_max_iter=int(classifier["max_iter"]),
        classifier_class_weight=str(classifier["class_weight"]),
        classifier_seed=None if classifier.get("classifier_seed") is None else int(classifier["classifier_seed"]),
    )
    validate_decentralized_k16_gmm_prior_config(cfg)
    return cfg


def validate_decentralized_k16_gmm_prior_config(cfg: DecentralizedK16GmmConfig) -> None:
    if cfg.name != DECENTRALIZED_K16_NAME:
        raise ProtocolError(f"Decentralized K16 experiment name must be {DECENTRALIZED_K16_NAME!r}.")
    if cfg.backbone != "virchow2":
        raise ProtocolError("D1 is locked to backbone=virchow2.")
    if cfg.primary_variant != PRIMARY_VARIANT:
        raise ProtocolError(f"primary_variant must be {PRIMARY_VARIANT!r} for independent source experts.")
    if cfg.primary_method != PRIMARY_DECENTRALIZED_METHOD:
        raise ProtocolError(f"primary_method must be {PRIMARY_DECENTRALIZED_METHOD!r}.")
    if len(cfg.heldout_centers) != 5:
        raise ProtocolError("D1 K16 composition expects exactly five centers, giving four source experts per fold.")
    if cfg.local_gmm_components_per_source_class != 4:
        raise ProtocolError("local_gmm_components_per_source_class must be locked to 4.")
    if cfg.composed_components_per_class != 16:
        raise ProtocolError("composed_components_per_class must be locked to 16.")
    expected_components = cfg.local_gmm_components_per_source_class * (len(cfg.heldout_centers) - 1)
    if cfg.composed_components_per_class != expected_components:
        raise ProtocolError("composed_components_per_class must equal K4 times the four non-target sources.")
    if cfg.source_weighting != "equal_source_mass":
        raise ProtocolError("source_weighting must be equal_source_mass.")
    if cfg.gmm_covariance_type != "diag":
        raise ProtocolError("gmm_covariance_type must be diag.")
    if cfg.primary_pooling != "geometric":
        raise ProtocolError("primary_pooling must be geometric.")
    if cfg.synthetic_per_class_total != 128:
        raise ProtocolError("synthetic_per_class_total must be locked to 128.")
    if min(cfg.gmm_reg_covar, cfg.min_component_weight, cfg.variance_floor) <= 0.0:
        raise ProtocolError("GMM regularization, component weight, and variance floors must be positive.")
    if cfg.gmm_n_init < 1 or cfg.gmm_max_iter < 1 or cfg.min_count_for_k4 < cfg.local_gmm_components_per_source_class:
        raise ProtocolError("GMM n_init/max_iter must be positive and min_count_for_k4 must support K4.")
    if cfg.classifier_type != "sklearn_logistic_regression":
        raise ProtocolError("classifier.type must be sklearn_logistic_regression.")
    if cfg.classifier_solver != "lbfgs" or cfg.classifier_c != 1.0 or cfg.classifier_max_iter != 2000:
        raise ProtocolError("Classifier solver/C/max_iter must remain locked.")
    if cfg.classifier_class_weight != "balanced" or cfg.classifier_seed is not None:
        raise ProtocolError("Classifier must use class_weight=balanced and classifier_seed=null.")


def run_decentralized_k16_gmm_prior(
    cfg: DecentralizedK16GmmConfig,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    root = prepare_artifact_dirs(Path(artifact_root) if artifact_root is not None else cfg.artifact_root)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "summaries").mkdir(parents=True, exist_ok=True)

    matrix_rows: list[dict[str, object]] = []
    gap_rows: list[dict[str, object]] = []
    summary_manifest_rows: list[dict[str, object]] = []
    composition_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    late_rows: list[dict[str, object]] = []
    real_feature_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    weak_rows: list[dict[str, object]] = []
    nn_rows: list[dict[str, object]] = []
    model_manifest_rows: list[dict[str, object]] = []
    protocol_violations: list[str] = []
    target_expert_excluded = True

    source_union_refs = _load_reference_values(
        cfg.source_union_gmm_artifact_root,
        table_name="gmm_prior_gap_summary.csv",
        method="source_union_cc_diag_gmm_k16_prior_sample_diagnostic",
        label="source-union K16",
    )
    center_balanced_refs = _load_reference_values(
        cfg.balanced_gmm_artifact_root,
        table_name="balanced_gmm_gap_summary.csv",
        method="source_union_center_balanced_cc_diag_gmm_k16_prior_sample",
        label="center-balanced K16",
    )
    _validate_optional_leakage_report(cfg.source_union_gmm_artifact_root, protocol_violations)
    _validate_optional_leakage_report(cfg.balanced_gmm_artifact_root, protocol_violations)

    repair_cfg = _repair_runtime_config(cfg, root)
    per_source_variant = _per_source_variant()

    try:
        for experiment_seed in cfg.experiment_seeds:
            train_cache = load_feature_cache(_existing_cache_path(cfg.feature_cache_root, seed=experiment_seed, split="train"))
            test_cache = load_feature_cache(_existing_cache_path(cfg.feature_cache_root, seed=experiment_seed, split="test"))
            per_source_runtime: dict[str, RuntimeSource] = {}
            source_summaries: dict[tuple[str, int], SourceLocalSummary] = {}
            shuffled_summaries: dict[tuple[str, int], SourceLocalSummary] = {}
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

                fitted = _fit_and_export_source_summaries(
                    cfg,
                    root,
                    runtime_source.runtime,
                    experiment_seed=int(experiment_seed),
                    shuffled_label_control=False,
                )
                shuffled = _fit_and_export_source_summaries(
                    cfg,
                    root,
                    runtime_source.runtime,
                    experiment_seed=int(experiment_seed),
                    shuffled_label_control=True,
                )
                for summary in fitted:
                    source_summaries[(summary.source_center, summary.class_label)] = summary
                    summary_manifest_rows.append(_summary_manifest_row(summary))
                    diagnostic_rows.append(_summary_diagnostic_row(cfg, summary))
                for summary in shuffled:
                    shuffled_summaries[(summary.source_center, summary.class_label)] = summary
                    diagnostic_rows.append(_summary_diagnostic_row(cfg, summary))

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
                if len(candidates) * cfg.local_gmm_components_per_source_class != cfg.composed_components_per_class:
                    raise ProtocolError("Fold does not compose exactly K16 from source-local K4 summaries.")

                composition_rows.extend(
                    _composition_manifest_rows(
                        cfg,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        candidates=candidates,
                        summaries=source_summaries,
                    )
                )

                target_indices = _target_indices(test_cache.metadata, str(heldout_center))
                eval_raw, eval_meta = select_rows(test_cache.embeddings, test_cache.metadata, target_indices)
                eval_labels = tuple(_label(row) for row in eval_meta)
                eval_error = "mono_class_target_eval" if len(set(eval_labels)) < 2 else ""

                for replicate_seed in cfg.replicate_seeds:
                    su_ref = _reference_for_cell(source_union_refs, experiment_seed, heldout_center, replicate_seed)
                    cb_ref = _reference_for_cell(center_balanced_refs, experiment_seed, heldout_center, replicate_seed)
                    if eval_error:
                        ineligible = _ineligible_rows(
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
                        matrix_rows.extend(ineligible)
                        continue

                    ref_row, real_late, _real_bundles = _real_feature_reference(
                        cfg,
                        per_source_runtime=per_source_runtime,
                        candidates=candidates,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        replicate_seed=int(replicate_seed),
                        eval_raw=eval_raw,
                        eval_labels=eval_labels,
                    )
                    real_feature_rows.append(ref_row)
                    matrix_rows.append(ref_row)
                    late_rows.extend(real_late)

                    rows, late, coverage, weak, nn = _evaluate_primary_and_references(
                        cfg,
                        per_source_runtime=per_source_runtime,
                        candidates=candidates,
                        summaries=source_summaries,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        replicate_seed=int(replicate_seed),
                        eval_raw=eval_raw,
                        eval_labels=eval_labels,
                        source_union_ref=su_ref,
                        center_balanced_ref=cb_ref,
                        real_feature_bacc=_float(ref_row["bacc"]),
                    )
                    matrix_rows.extend(rows)
                    late_rows.extend(late)
                    coverage_rows.extend(coverage)
                    weak_rows.extend(weak)
                    nn_rows.extend(nn)

                    rows, late, coverage, weak, nn = _evaluate_control(
                        cfg,
                        per_source_runtime=per_source_runtime,
                        candidates=candidates,
                        summaries=source_summaries,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        replicate_seed=int(replicate_seed),
                        eval_raw=eval_raw,
                        eval_labels=eval_labels,
                        source_union_ref=su_ref,
                        center_balanced_ref=cb_ref,
                        real_feature_bacc=_float(ref_row["bacc"]),
                        prior_method=ROW_SHUFFLED_SUMMARY_CONTROL,
                        control_mode="class_flip",
                    )
                    matrix_rows.extend(rows)
                    late_rows.extend(late)
                    coverage_rows.extend(coverage)
                    weak_rows.extend(weak)
                    nn_rows.extend(nn)

                    rows, late, coverage, weak, nn = _evaluate_control(
                        cfg,
                        per_source_runtime=per_source_runtime,
                        candidates=candidates,
                        summaries=shuffled_summaries,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        replicate_seed=int(replicate_seed),
                        eval_raw=eval_raw,
                        eval_labels=eval_labels,
                        source_union_ref=su_ref,
                        center_balanced_ref=cb_ref,
                        real_feature_bacc=_float(ref_row["bacc"]),
                        prior_method=ROW_SHUFFLED_LABEL_CONTROL,
                        control_mode="normal",
                    )
                    matrix_rows.extend(rows)
                    late_rows.extend(late)
                    coverage_rows.extend(coverage)
                    weak_rows.extend(weak)
                    nn_rows.extend(nn)

                    matrix_rows.append(
                        _support_nelbo_disabled_row(
                            cfg,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            replicate_seed=int(replicate_seed),
                            candidates=candidates,
                            source_union_ref=su_ref,
                            center_balanced_ref=cb_ref,
                            real_feature_bacc=_float(ref_row["bacc"]),
                        )
                    )
                    matrix_rows.append(
                        _reference_matrix_row(
                            cfg,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            replicate_seed=int(replicate_seed),
                            candidates=candidates,
                            prior_method=ROW_SOURCE_UNION_K16_REFERENCE,
                            reference=su_ref,
                        )
                    )
                    matrix_rows.append(
                        _reference_matrix_row(
                            cfg,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            replicate_seed=int(replicate_seed),
                            candidates=candidates,
                            prior_method=ROW_CENTER_BALANCED_K16_REFERENCE,
                            reference=cb_ref,
                        )
                    )
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
    decision = _decision(matrix_rows, cfg, leakage_status=leakage.status)
    _write_artifacts(
        root,
        cfg,
        matrix_rows=matrix_rows,
        gap_rows=gap_rows,
        summary_manifest_rows=summary_manifest_rows,
        composition_rows=composition_rows,
        diagnostic_rows=diagnostic_rows,
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


def _fit_and_export_source_summaries(
    cfg: DecentralizedK16GmmConfig,
    root: Path,
    runtime: VariantRuntime,
    *,
    experiment_seed: int,
    shuffled_label_control: bool,
) -> tuple[SourceLocalSummary, ...]:
    import numpy as np  # type: ignore
    import torch  # type: ignore
    from sklearn.mixture import GaussianMixture  # type: ignore

    source_centers = {str(v) for v in runtime.source_train_centers}
    if len(source_centers) != 1 or runtime.expert_id not in source_centers:
        raise ProtocolError("Source-local summaries must be fitted from exactly one source center.")
    x_np = np.asarray(runtime.source_train_embeddings, dtype=np.float32)
    y_np = np.asarray(runtime.source_train_labels, dtype=int)
    fit_labels = y_np.copy()
    if shuffled_label_control:
        rng = np.random.default_rng(_latent_seed(experiment_seed, runtime.expert_id, "shuffled_label_summary"))
        rng.shuffle(fit_labels)
    with torch.no_grad():
        x = torch.as_tensor(x_np, dtype=torch.float32)
        y = torch.as_tensor(y_np, dtype=torch.long)
        mu, _logvar = runtime.model.encode(x, y)
    mu_np = mu.detach().cpu().numpy()
    out: list[SourceLocalSummary] = []
    for cls in (0, 1):
        positions = np.flatnonzero(fit_labels == int(cls))
        base_error = ""
        status = "ok"
        if int(positions.size) < cfg.min_count_for_k4:
            status = "ineligible_component_fit"
            base_error = f"source_class_count<{cfg.min_count_for_k4}"
        elif int(positions.size) < cfg.local_gmm_components_per_source_class:
            status = "ineligible_component_fit"
            base_error = f"source_class_count<{cfg.local_gmm_components_per_source_class}"
        weights = np.asarray([], dtype=float)
        means = np.empty((0, int(runtime.model.latent_dim)), dtype=float)
        diag_vars = np.empty((0, int(runtime.model.latent_dim)), dtype=float)
        converged = False
        n_iter = 0
        score = math.nan
        bic = math.nan
        if status == "ok":
            gmm = GaussianMixture(
                n_components=cfg.local_gmm_components_per_source_class,
                covariance_type="diag",
                reg_covar=cfg.gmm_reg_covar,
                n_init=cfg.gmm_n_init,
                max_iter=cfg.gmm_max_iter,
                random_state=_latent_seed(
                    experiment_seed,
                    runtime.expert_id,
                    cls,
                    "local_k4_gmm",
                    shuffled_label_control,
                ),
            )
            cls_mu = mu_np[positions]
            gmm.fit(cls_mu)
            weights = np.asarray(gmm.weights_, dtype=float)
            means = np.asarray(gmm.means_, dtype=float)
            diag_vars = np.asarray(gmm.covariances_, dtype=float)
            converged = bool(gmm.converged_)
            n_iter = int(gmm.n_iter_)
            score = float(gmm.score(cls_mu))
            bic = float(gmm.bic(cls_mu))
        finite = bool(np.isfinite(weights).all() and np.isfinite(means).all() and np.isfinite(diag_vars).all())
        effective = int(np.sum(weights >= cfg.min_component_weight)) if weights.size else 0
        min_weight = float(np.min(weights)) if weights.size else math.nan
        min_diag_var = float(np.min(diag_vars)) if diag_vars.size else math.nan
        entropy = _entropy(weights)
        errors = [base_error] if base_error else []
        if status == "ok" and not converged:
            errors.append("gmm_converged=false")
        if status == "ok" and effective != cfg.local_gmm_components_per_source_class:
            errors.append(f"effective_component_count!={cfg.local_gmm_components_per_source_class}")
        if status == "ok" and (not math.isfinite(min_weight) or min_weight < cfg.min_component_weight):
            errors.append(f"min_component_weight<{cfg.min_component_weight}")
        if status == "ok" and (not finite):
            errors.append("nonfinite_summary_parameter")
        if status == "ok" and (not math.isfinite(min_diag_var) or min_diag_var < cfg.variance_floor):
            errors.append(f"diag_var<{cfg.variance_floor}")
        if errors:
            status = "ineligible_component_fit"
        clipped_vars = np.maximum(diag_vars, cfg.variance_floor) if diag_vars.size else diag_vars
        parameter_hash = _hash_array(_flatten_payload([weights, means, clipped_vars])) if weights.size else ""
        summary_path = _summary_path(root, experiment_seed, runtime.expert_id, cls, shuffled_label_control=shuffled_label_control)
        summary_hash = ""
        if weights.size:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                summary_path,
                weights=weights,
                means=means,
                diag_vars=clipped_vars,
                source_class_count=np.asarray([int(positions.size)], dtype=int),
                schema_version=np.asarray([SUMMARY_SCHEMA_VERSION]),
                expert_config_hash=np.asarray([_expert_config_hash(runtime)]),
            )
            summary_hash = _file_sha256(summary_path)
            seedless = _seedless_summary_path(root, runtime.expert_id, cls, shuffled_label_control=shuffled_label_control)
            seedless.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                seedless,
                weights=weights,
                means=means,
                diag_vars=clipped_vars,
                source_class_count=np.asarray([int(positions.size)], dtype=int),
                schema_version=np.asarray([SUMMARY_SCHEMA_VERSION]),
                expert_config_hash=np.asarray([_expert_config_hash(runtime)]),
            )
        out.append(
            SourceLocalSummary(
                experiment_seed=int(experiment_seed),
                source_center=runtime.expert_id,
                class_label=int(cls),
                weights=weights,
                means=means,
                diag_vars=clipped_vars,
                source_class_count=int(positions.size),
                effective_component_count=effective,
                min_component_weight=min_weight,
                min_diag_var=min_diag_var,
                component_entropy=entropy,
                all_finite=finite,
                gmm_converged=converged,
                gmm_n_iter=n_iter,
                source_train_log_likelihood=score,
                source_inner_bic=bic,
                summary_path=summary_path,
                summary_hash=summary_hash,
                fit_row_ids_hash=_hash_strings([runtime.source_train_sample_ids[int(pos)] for pos in positions]),
                parameter_hash=parameter_hash,
                expert_config_hash=_expert_config_hash(runtime),
                status=status,
                error_message="|".join(errors),
                shuffled_label_control=bool(shuffled_label_control),
            )
        )
    return tuple(out)


def _summary_path(root: Path, seed: int, source_center: str, class_label: int, *, shuffled_label_control: bool) -> Path:
    suffix = "_shuffled_label_control" if shuffled_label_control else ""
    return root / "summaries" / f"seed_{int(seed)}" / f"source_{source_center}" / f"class_{int(class_label)}_k4{suffix}_summary.npz"


def _seedless_summary_path(root: Path, source_center: str, class_label: int, *, shuffled_label_control: bool) -> Path:
    suffix = "_shuffled_label_control" if shuffled_label_control else ""
    return root / "summaries" / f"source_{source_center}" / f"class_{int(class_label)}_k4{suffix}_summary.npz"


def _summary_manifest_row(summary: SourceLocalSummary) -> dict[str, object]:
    return {
        "experiment_seed": int(summary.experiment_seed),
        "source_center": summary.source_center,
        "class_label": int(summary.class_label),
        "component_id": "|".join(str(idx) for idx in range(_component_count(summary))),
        "component_weight_local": json.dumps(_as_float_list(summary.weights)),
        "latent_mean_hash": _hash_array(summary.means) if _component_count(summary) else "",
        "latent_diag_var_hash": _hash_array(summary.diag_vars) if _component_count(summary) else "",
        "source_class_count": int(summary.source_class_count),
        "expert_config_hash": summary.expert_config_hash,
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "summary_path": str(summary.summary_path),
        "summary_hash": summary.summary_hash,
        "status": summary.status,
        "error_message": summary.error_message,
    }


def _summary_diagnostic_row(cfg: DecentralizedK16GmmConfig, summary: SourceLocalSummary) -> dict[str, object]:
    return {
        "experiment_seed": int(summary.experiment_seed),
        "source_center": summary.source_center,
        "class_label": int(summary.class_label),
        "local_gmm_components": cfg.local_gmm_components_per_source_class,
        "source_class_count": int(summary.source_class_count),
        "effective_component_count": int(summary.effective_component_count),
        "min_component_weight": summary.min_component_weight,
        "min_required_component_weight": cfg.min_component_weight,
        "variance_floor": cfg.variance_floor,
        "min_diag_var": summary.min_diag_var,
        "all_finite": bool(summary.all_finite),
        "component_entropy": summary.component_entropy,
        "gmm_converged": bool(summary.gmm_converged),
        "gmm_n_iter": int(summary.gmm_n_iter),
        "source_train_log_likelihood": summary.source_train_log_likelihood,
        "source_inner_bic": summary.source_inner_bic,
        "fit_row_ids_hash": summary.fit_row_ids_hash,
        "parameter_hash": summary.parameter_hash,
        "summary_path": str(summary.summary_path),
        "summary_hash": summary.summary_hash,
        "shuffled_label_control": bool(summary.shuffled_label_control),
        "status": summary.status,
        "error_message": summary.error_message,
    }


def _composition_manifest_rows(
    cfg: DecentralizedK16GmmConfig,
    *,
    experiment_seed: int,
    heldout_center: str,
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], SourceLocalSummary],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    source_mass = 1.0 / float(len(candidates))
    for cls in (0, 1):
        composed_id = 0
        for source_center in candidates:
            summary = summaries.get((str(source_center), int(cls)))
            if summary is None:
                rows.append(
                    _invalid_composition_row(
                        experiment_seed=experiment_seed,
                        heldout_center=heldout_center,
                        candidates=candidates,
                        class_label=cls,
                        composed_component_id=composed_id,
                        source_center=str(source_center),
                        summary_status="missing_summary",
                        summary_error_message=f"missing_summary_source_{source_center}_class_{cls}",
                    )
                )
                continue
            if summary.status != "ok":
                rows.append(
                    _invalid_composition_row(
                        experiment_seed=experiment_seed,
                        heldout_center=heldout_center,
                        candidates=candidates,
                        class_label=cls,
                        composed_component_id=composed_id,
                        source_center=str(source_center),
                        summary_status=summary.status,
                        summary_error_message=summary.error_message,
                    )
                )
                continue
            try:
                weights = _normalized_weights(summary.weights)
            except ProtocolError as exc:
                rows.append(
                    _invalid_composition_row(
                        experiment_seed=experiment_seed,
                        heldout_center=heldout_center,
                        candidates=candidates,
                        class_label=cls,
                        composed_component_id=composed_id,
                        source_center=str(source_center),
                        summary_status="invalid_summary_weights",
                        summary_error_message=str(exc),
                    )
                )
                continue
            for component_idx, local_weight in enumerate(weights):
                rows.append(
                    {
                        "experiment_seed": int(experiment_seed),
                        "heldout_center": str(heldout_center),
                        "included_source_centers": "|".join(str(v) for v in candidates),
                        "class_label": int(cls),
                        "composed_component_id": composed_id,
                        "source_center": str(source_center),
                        "source_component_id": int(component_idx),
                        "component_weight_local": float(local_weight),
                        "component_weight_after_equal_source_normalization": float(source_mass * local_weight),
                        "summary_hash": summary.summary_hash,
                        "summary_status": summary.status,
                        "summary_error_message": summary.error_message,
                    }
                )
                composed_id += 1
    return rows


def _invalid_composition_row(
    *,
    experiment_seed: int,
    heldout_center: str,
    candidates: Sequence[str],
    class_label: int,
    composed_component_id: int,
    source_center: str,
    summary_status: str,
    summary_error_message: str,
) -> dict[str, object]:
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "included_source_centers": "|".join(str(v) for v in candidates),
        "class_label": int(class_label),
        "composed_component_id": int(composed_component_id),
        "source_center": str(source_center),
        "source_component_id": "",
        "component_weight_local": "",
        "component_weight_after_equal_source_normalization": "",
        "summary_hash": "",
        "summary_status": str(summary_status),
        "summary_error_message": str(summary_error_message),
    }


def _evaluate_primary_and_references(
    cfg: DecentralizedK16GmmConfig,
    *,
    per_source_runtime: Mapping[str, RuntimeSource],
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], SourceLocalSummary],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    eval_raw: object,
    eval_labels: Sequence[int],
    source_union_ref: ReferenceValue,
    center_balanced_ref: ReferenceValue,
    real_feature_bacc: float,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    status, error = _composition_status(cfg, candidates, summaries)
    if status != "ok":
        rows = _composition_ineligible_rows(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            candidates=candidates,
            source_union_ref=source_union_ref,
            center_balanced_ref=center_balanced_ref,
            real_feature_bacc=real_feature_bacc,
            status=status,
            error_message=error,
        )
        return rows, [], [], [], []
    bundles, single_rows, coverage_rows, weak_rows, nn_rows, generated_hash = _source_generated_bundles(
        cfg,
        per_source_runtime=per_source_runtime,
        candidates=candidates,
        summaries=summaries,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        eval_raw=eval_raw,
        eval_labels=eval_labels,
        prior_method=PRIMARY_DECENTRALIZED_METHOD,
        control_mode="normal",
    )
    pooled_geom = geometric_probability_pool(bundles)
    pooled_arith = _arithmetic_probability_pool(bundles)
    single_baccs = [_float(row["bacc"]) for row in single_rows if row.get("status") == "ok"]
    single_macro = [_float(row["macro_f1"]) for row in single_rows if row.get("status") == "ok"]
    mean_single = nanmean(single_baccs)
    oracle_single = max(single_baccs) if single_baccs else math.nan
    rows = [
        _dense_result_row(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            candidates=candidates,
            prior_method=PRIMARY_DECENTRALIZED_METHOD,
            pooling_rule="geometric",
            probabilities=pooled_geom,
            eval_labels=eval_labels,
            generated_features_hash=generated_hash,
            source_union_ref=source_union_ref,
            center_balanced_ref=center_balanced_ref,
            real_feature_bacc=real_feature_bacc,
            mean_single_bacc=mean_single,
            oracle_single_bacc=oracle_single,
            mean_single_macro_f1=nanmean(single_macro),
            selection_source=PRIMARY_SELECTION,
            claim_role="primary_preservation_test",
        ),
        _dense_result_row(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            candidates=candidates,
            prior_method=ROW_ARITH,
            pooling_rule="arithmetic",
            probabilities=pooled_arith,
            eval_labels=eval_labels,
            generated_features_hash=generated_hash,
            source_union_ref=source_union_ref,
            center_balanced_ref=center_balanced_ref,
            real_feature_bacc=real_feature_bacc,
            mean_single_bacc=mean_single,
            oracle_single_bacc=oracle_single,
            mean_single_macro_f1=nanmean(single_macro),
            selection_source=DIAGNOSTIC_SELECTION,
            claim_role="diagnostic_pooling_rule",
        ),
        _aggregate_reference_row(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            candidates=candidates,
            prior_method=ROW_SINGLE_MEAN,
            bacc=mean_single,
            macro_f1=nanmean(single_macro),
            source_union_ref=source_union_ref,
            center_balanced_ref=center_balanced_ref,
            real_feature_bacc=real_feature_bacc,
            mean_single_bacc=mean_single,
            oracle_single_bacc=oracle_single,
            claim_role="single_source_mean_reference",
        ),
        _aggregate_reference_row(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            candidates=candidates,
            prior_method=ROW_SINGLE_ORACLE,
            bacc=oracle_single,
            macro_f1=max(single_macro) if single_macro else math.nan,
            source_union_ref=source_union_ref,
            center_balanced_ref=center_balanced_ref,
            real_feature_bacc=real_feature_bacc,
            mean_single_bacc=mean_single,
            oracle_single_bacc=oracle_single,
            claim_role="diagnostic_only_oracle_reference",
        ),
    ]
    return rows, single_rows, coverage_rows, weak_rows, nn_rows


def _evaluate_control(
    cfg: DecentralizedK16GmmConfig,
    *,
    per_source_runtime: Mapping[str, RuntimeSource],
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], SourceLocalSummary],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    eval_raw: object,
    eval_labels: Sequence[int],
    source_union_ref: ReferenceValue,
    center_balanced_ref: ReferenceValue,
    real_feature_bacc: float,
    prior_method: str,
    control_mode: str,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    status, error = _composition_status(cfg, candidates, summaries)
    if status != "ok":
        row = _dense_empty_row(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            candidates=candidates,
            prior_method=prior_method,
            source_union_ref=source_union_ref,
            center_balanced_ref=center_balanced_ref,
            real_feature_bacc=real_feature_bacc,
            status=status,
            error_message=error,
            claim_role="negative_control",
        )
        return [row], [], [], [], []
    bundles, single_rows, coverage_rows, weak_rows, nn_rows, generated_hash = _source_generated_bundles(
        cfg,
        per_source_runtime=per_source_runtime,
        candidates=candidates,
        summaries=summaries,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        eval_raw=eval_raw,
        eval_labels=eval_labels,
        prior_method=prior_method,
        control_mode=control_mode,
    )
    pooled = geometric_probability_pool(bundles)
    single_baccs = [_float(row["bacc"]) for row in single_rows if row.get("status") == "ok"]
    row = _dense_result_row(
        cfg,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        candidates=candidates,
        prior_method=prior_method,
        pooling_rule="geometric",
        probabilities=pooled,
        eval_labels=eval_labels,
        generated_features_hash=generated_hash,
        source_union_ref=source_union_ref,
        center_balanced_ref=center_balanced_ref,
        real_feature_bacc=real_feature_bacc,
        mean_single_bacc=nanmean(single_baccs),
        oracle_single_bacc=max(single_baccs) if single_baccs else math.nan,
        mean_single_macro_f1=nanmean([_float(row["macro_f1"]) for row in single_rows if row.get("status") == "ok"]),
        selection_source=DIAGNOSTIC_SELECTION,
        claim_role="negative_control",
    )
    return [row], single_rows, coverage_rows, weak_rows, nn_rows


def _source_generated_bundles(
    cfg: DecentralizedK16GmmConfig,
    *,
    per_source_runtime: Mapping[str, RuntimeSource],
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], SourceLocalSummary],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    eval_raw: object,
    eval_labels: Sequence[int],
    prior_method: str,
    control_mode: str,
) -> tuple[
    list[PredictionBundle],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    str,
]:
    import numpy as np  # type: ignore

    budgets = _balanced_counts(cfg.synthetic_per_class_total, len(candidates))
    bundles: list[PredictionBundle] = []
    late_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    weak_rows: list[dict[str, object]] = []
    nn_rows: list[dict[str, object]] = []
    generated_hashes: list[str] = []
    for source_center, budget_per_class in zip(candidates, budgets):
        runtime = per_source_runtime[str(source_center)].runtime
        latent_seed = _latent_seed(experiment_seed, heldout_center, replicate_seed, prior_method, source_center, control_mode)
        generated, labels, counts = _sample_source_from_summaries(
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
        generated_hash = _hash_array(generated)
        prediction_hash = _hash_array(bundle.probabilities)
        generated_hashes.append(generated_hash)
        row = {
            "experiment_seed": int(experiment_seed),
            "heldout_center": str(heldout_center),
            "expert_id": str(source_center),
            "expert_pool_type": POOL_PER_SOURCE,
            "variant_id": PRIMARY_VARIANT,
            "prior_method": prior_method,
            "gmm_components": cfg.local_gmm_components_per_source_class,
            "effective_gmm_components": _source_effective_components(summaries, source_center, control_mode=control_mode),
            "local_gmm_components_per_source_class": cfg.local_gmm_components_per_source_class,
            "composed_components_per_class": cfg.composed_components_per_class,
            "source_weighting": cfg.source_weighting,
            "pooling_rule": "single_source",
            "replicate_seed": int(replicate_seed),
            "latent_sample_seed": int(latent_seed),
            "included_source_centers": "|".join(str(v) for v in candidates),
            "num_included_sources": len(candidates),
            "synthetic_per_class_total": int(budget_per_class),
            "synthetic_per_class_per_source_json": json.dumps({str(source_center): int(budget_per_class)}, sort_keys=True),
            "bacc": result.bacc,
            "macro_f1": result.macro_f1,
            "source_union_k16_bacc": math.nan,
            "center_balanced_k16_bacc": math.nan,
            "real_feature_dense_bacc": math.nan,
            "mean_single_source_k4_bacc": math.nan,
            "oracle_single_source_k4_bacc": math.nan,
            "delta_vs_mean_single_source_k4": math.nan,
            "delta_vs_single_source_oracle_reference": math.nan,
            "retention_vs_source_union_k16": math.nan,
            "retention_vs_center_balanced_k16": math.nan,
            "delta_vs_real_source_embedding_dense_reference": math.nan,
            "negative_control_gap": math.nan,
            "generated_features_hash": generated_hash,
            "prediction_hash": prediction_hash,
            "composed_prior_hash": _summary_set_hash(summaries, candidates, control_mode=control_mode),
            "summary_set_hash": _summary_set_hash(summaries, candidates, control_mode=control_mode),
            "selection_source": DIAGNOSTIC_SELECTION,
            "status": "ok",
            "error_message": "",
            "claim_role": "single_source_component_for_dense_aggregation",
        }
        late_rows.append(row)
        if _float(row["bacc"]) < 0.75:
            weak_rows.append(_weak_row(row))
        coverage_rows.append(_coverage_row(cfg, row, counts, candidates=candidates))
        nn_rows.append(_nearest_neighbor_row(row, generated, runtime.source_train_embeddings))
        bundles.append(bundle)
    aggregate_hash = _hash_strings(generated_hashes)
    return bundles, late_rows, coverage_rows, weak_rows, nn_rows, aggregate_hash


def _sample_source_from_summaries(
    cfg: DecentralizedK16GmmConfig,
    runtime: VariantRuntime,
    summaries: Mapping[tuple[str, int], SourceLocalSummary],
    *,
    source_center: str,
    budget_per_class: int,
    seed: int,
    control_mode: str,
) -> tuple[object, tuple[int, ...], dict[int, dict[str, int]]]:
    import numpy as np  # type: ignore

    rng = np.random.default_rng(int(seed))
    chunks = []
    labels = []
    component_counts: dict[int, dict[str, int]] = {}
    for label_cls in (0, 1):
        summary_cls = 1 - int(label_cls) if control_mode == "class_flip" else int(label_cls)
        summary = summaries[(str(source_center), int(summary_cls))]
        z_np, counts = _sample_latents(summary, rng, int(budget_per_class), variance_floor=cfg.variance_floor)
        decoded, _ = _decode_latents(runtime, z_np, [int(label_cls)] * int(budget_per_class))
        chunks.append(decoded)
        labels.extend([int(label_cls)] * int(budget_per_class))
        component_counts[int(label_cls)] = {f"{source_center}:{key}": int(value) for key, value in counts.items()}
    return np.vstack(chunks), tuple(labels), component_counts


def _sample_latents(
    summary: SourceLocalSummary,
    rng: object,
    n_samples: int,
    *,
    variance_floor: float,
) -> tuple[object, dict[int, int]]:
    import numpy as np  # type: ignore

    weights = _normalized_weights(summary.weights)
    components = rng.choice(np.arange(weights.shape[0]), size=int(n_samples), replace=True, p=weights)
    means = np.asarray(summary.means, dtype=np.float32)[components]
    variances = np.asarray(summary.diag_vars, dtype=np.float32)[components]
    eps = rng.normal(size=means.shape).astype(np.float32)
    z_np = means + np.sqrt(np.maximum(variances, float(variance_floor))).astype(np.float32) * eps
    unique, counts = np.unique(components, return_counts=True)
    return z_np, {int(k): int(v) for k, v in zip(unique, counts)}


def _real_feature_reference(
    cfg: DecentralizedK16GmmConfig,
    *,
    per_source_runtime: Mapping[str, RuntimeSource],
    candidates: Sequence[str],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    eval_raw: object,
    eval_labels: Sequence[int],
) -> tuple[dict[str, object], list[dict[str, object]], list[PredictionBundle]]:
    bundles: list[PredictionBundle] = []
    late_rows: list[dict[str, object]] = []
    for source_center in candidates:
        runtime = per_source_runtime[str(source_center)].runtime
        eval_x = runtime.frame.transform(_to_numpy(eval_raw))
        bundle = fit_locked_logistic_classifier(
            runtime.source_train_embeddings,
            runtime.source_train_labels,
            eval_x,
            classifier_seed=cfg.classifier_seed,
            expert_id=str(source_center),
            class_weight=cfg.classifier_class_weight,
        )
        result = evaluate_probability_predictions(ROW_REAL_FEATURE_DENSE_REFERENCE, bundle.probabilities, eval_labels)
        late_rows.append(
            {
                "experiment_seed": int(experiment_seed),
                "heldout_center": str(heldout_center),
                "expert_id": str(source_center),
                "expert_pool_type": POOL_PER_SOURCE,
                "variant_id": PRIMARY_VARIANT,
                "prior_method": ROW_REAL_FEATURE_DENSE_REFERENCE,
                "gmm_components": 0,
                "effective_gmm_components": 0,
                "local_gmm_components_per_source_class": 0,
                "composed_components_per_class": 0,
                "source_weighting": "not_applicable",
                "pooling_rule": "single_source_real_feature",
                "replicate_seed": int(replicate_seed),
                "latent_sample_seed": NA,
                "included_source_centers": "|".join(str(v) for v in candidates),
                "num_included_sources": len(candidates),
                "synthetic_per_class_total": 0,
                "synthetic_per_class_per_source_json": "{}",
                "bacc": result.bacc,
                "macro_f1": result.macro_f1,
                "source_union_k16_bacc": math.nan,
                "center_balanced_k16_bacc": math.nan,
                "real_feature_dense_bacc": math.nan,
                "mean_single_source_k4_bacc": math.nan,
                "oracle_single_source_k4_bacc": math.nan,
                "delta_vs_mean_single_source_k4": math.nan,
                "delta_vs_single_source_oracle_reference": math.nan,
                "retention_vs_source_union_k16": math.nan,
                "retention_vs_center_balanced_k16": math.nan,
                "delta_vs_real_source_embedding_dense_reference": math.nan,
                "negative_control_gap": math.nan,
                "generated_features_hash": "",
                "prediction_hash": _hash_array(bundle.probabilities),
                "composed_prior_hash": "",
                "summary_set_hash": "",
                "selection_source": DIAGNOSTIC_SELECTION,
                "status": "ok",
                "error_message": "",
                "claim_role": "real_feature_single_source_reference",
            }
        )
        bundles.append(bundle)
    pooled = geometric_probability_pool(bundles)
    result = evaluate_probability_predictions(ROW_REAL_FEATURE_DENSE_REFERENCE, pooled, eval_labels)
    row = _base_matrix_row(
        cfg,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        candidates=candidates,
        expert_id="dense_all_sources",
        expert_pool_type=POOL_DECENTRALIZED,
        prior_method=ROW_REAL_FEATURE_DENSE_REFERENCE,
        pooling_rule="geometric",
        source_union_ref=_missing_reference(),
        center_balanced_ref=_missing_reference(),
    )
    row.update(
        {
            "gmm_components": 0,
            "effective_gmm_components": 0,
            "local_gmm_components_per_source_class": 0,
            "composed_components_per_class": 0,
            "source_weighting": "not_applicable",
            "synthetic_per_class_total": 0,
            "synthetic_per_class_per_source_json": "{}",
            "bacc": result.bacc,
            "macro_f1": result.macro_f1,
            "real_feature_dense_bacc": result.bacc,
            "prediction_hash": _hash_array(pooled),
            "selection_source": DIAGNOSTIC_SELECTION,
            "status": "ok",
            "claim_role": "real_feature_transfer_ceiling_reference",
        }
    )
    return row, late_rows, bundles


def _dense_result_row(
    cfg: DecentralizedK16GmmConfig,
    *,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    prior_method: str,
    pooling_rule: str,
    probabilities: Sequence[Sequence[float]],
    eval_labels: Sequence[int],
    generated_features_hash: str,
    source_union_ref: ReferenceValue,
    center_balanced_ref: ReferenceValue,
    real_feature_bacc: float,
    mean_single_bacc: float,
    oracle_single_bacc: float,
    mean_single_macro_f1: float,
    selection_source: str,
    claim_role: str,
) -> dict[str, object]:
    result = evaluate_probability_predictions(prior_method, probabilities, eval_labels)
    row = _base_matrix_row(
        cfg,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        candidates=candidates,
        expert_id="dense_all_sources",
        expert_pool_type=POOL_DECENTRALIZED,
        prior_method=prior_method,
        pooling_rule=pooling_rule,
        source_union_ref=source_union_ref,
        center_balanced_ref=center_balanced_ref,
    )
    row.update(
        {
            "bacc": result.bacc,
            "macro_f1": result.macro_f1,
            "real_feature_dense_bacc": real_feature_bacc,
            "mean_single_source_k4_bacc": mean_single_bacc,
            "oracle_single_source_k4_bacc": oracle_single_bacc,
            "delta_vs_mean_single_source_k4": result.bacc - mean_single_bacc if math.isfinite(mean_single_bacc) else math.nan,
            "delta_vs_single_source_oracle_reference": result.bacc - oracle_single_bacc if math.isfinite(oracle_single_bacc) else math.nan,
            "retention_vs_source_union_k16": _retention(result.bacc, source_union_ref.bacc),
            "retention_vs_center_balanced_k16": _retention(result.bacc, center_balanced_ref.bacc),
            "delta_vs_real_source_embedding_dense_reference": result.bacc - real_feature_bacc if math.isfinite(real_feature_bacc) else math.nan,
            "generated_features_hash": generated_features_hash,
            "prediction_hash": _hash_array(probabilities),
            "selection_source": selection_source,
            "status": "ok",
            "claim_role": claim_role,
        }
    )
    return row


def _aggregate_reference_row(
    cfg: DecentralizedK16GmmConfig,
    *,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    prior_method: str,
    bacc: float,
    macro_f1: float,
    source_union_ref: ReferenceValue,
    center_balanced_ref: ReferenceValue,
    real_feature_bacc: float,
    mean_single_bacc: float,
    oracle_single_bacc: float,
    claim_role: str,
) -> dict[str, object]:
    row = _base_matrix_row(
        cfg,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        candidates=candidates,
        expert_id="single_source_reference",
        expert_pool_type=POOL_PER_SOURCE,
        prior_method=prior_method,
        pooling_rule="reference",
        source_union_ref=source_union_ref,
        center_balanced_ref=center_balanced_ref,
    )
    row.update(
        {
            "bacc": bacc,
            "macro_f1": macro_f1,
            "real_feature_dense_bacc": real_feature_bacc,
            "mean_single_source_k4_bacc": mean_single_bacc,
            "oracle_single_source_k4_bacc": oracle_single_bacc,
            "delta_vs_mean_single_source_k4": bacc - mean_single_bacc if math.isfinite(bacc) and math.isfinite(mean_single_bacc) else math.nan,
            "delta_vs_single_source_oracle_reference": bacc - oracle_single_bacc if math.isfinite(bacc) and math.isfinite(oracle_single_bacc) else math.nan,
            "retention_vs_source_union_k16": _retention(bacc, source_union_ref.bacc),
            "retention_vs_center_balanced_k16": _retention(bacc, center_balanced_ref.bacc),
            "delta_vs_real_source_embedding_dense_reference": bacc - real_feature_bacc if math.isfinite(bacc) and math.isfinite(real_feature_bacc) else math.nan,
            "selection_source": DIAGNOSTIC_SELECTION,
            "status": "ok",
            "claim_role": claim_role,
        }
    )
    return row


def _base_matrix_row(
    cfg: DecentralizedK16GmmConfig,
    *,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    expert_id: str,
    expert_pool_type: str,
    prior_method: str,
    pooling_rule: str,
    source_union_ref: ReferenceValue,
    center_balanced_ref: ReferenceValue,
) -> dict[str, object]:
    budgets = _budget_json(cfg, candidates)
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "expert_id": str(expert_id),
        "expert_pool_type": str(expert_pool_type),
        "variant_id": PRIMARY_VARIANT,
        "prior_method": prior_method,
        "gmm_components": cfg.composed_components_per_class,
        "effective_gmm_components": cfg.composed_components_per_class,
        "local_gmm_components_per_source_class": cfg.local_gmm_components_per_source_class,
        "composed_components_per_class": cfg.composed_components_per_class,
        "source_weighting": cfg.source_weighting,
        "pooling_rule": pooling_rule,
        "replicate_seed": int(replicate_seed),
        "latent_sample_seed": _latent_seed(experiment_seed, heldout_center, replicate_seed, prior_method),
        "included_source_centers": "|".join(str(v) for v in candidates),
        "num_included_sources": len(candidates),
        "synthetic_per_class_total": cfg.synthetic_per_class_total,
        "synthetic_per_class_per_source_json": budgets,
        "bacc": "",
        "macro_f1": "",
        "source_union_k16_bacc": source_union_ref.bacc,
        "center_balanced_k16_bacc": center_balanced_ref.bacc,
        "real_feature_dense_bacc": math.nan,
        "mean_single_source_k4_bacc": math.nan,
        "oracle_single_source_k4_bacc": math.nan,
        "delta_vs_mean_single_source_k4": math.nan,
        "delta_vs_single_source_oracle_reference": math.nan,
        "retention_vs_source_union_k16": math.nan,
        "retention_vs_center_balanced_k16": math.nan,
        "delta_vs_real_source_embedding_dense_reference": math.nan,
        "negative_control_gap": math.nan,
        "generated_features_hash": "",
        "prediction_hash": "",
        "composed_prior_hash": "",
        "summary_set_hash": "",
        "selection_source": DIAGNOSTIC_SELECTION,
        "status": "",
        "error_message": "",
        "claim_role": "",
    }


def _dense_empty_row(
    cfg: DecentralizedK16GmmConfig,
    *,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    prior_method: str,
    source_union_ref: ReferenceValue,
    center_balanced_ref: ReferenceValue,
    real_feature_bacc: float,
    status: str,
    error_message: str,
    claim_role: str,
) -> dict[str, object]:
    row = _base_matrix_row(
        cfg,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        candidates=candidates,
        expert_id="dense_all_sources",
        expert_pool_type=POOL_DECENTRALIZED,
        prior_method=prior_method,
        pooling_rule="geometric",
        source_union_ref=source_union_ref,
        center_balanced_ref=center_balanced_ref,
    )
    row.update(
        {
            "real_feature_dense_bacc": real_feature_bacc,
            "status": status,
            "error_message": error_message,
            "claim_role": claim_role,
        }
    )
    return row


def _composition_ineligible_rows(
    cfg: DecentralizedK16GmmConfig,
    *,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    source_union_ref: ReferenceValue,
    center_balanced_ref: ReferenceValue,
    real_feature_bacc: float,
    status: str,
    error_message: str,
) -> list[dict[str, object]]:
    rows = []
    for method, role in (
        (PRIMARY_DECENTRALIZED_METHOD, "primary_preservation_test"),
        (ROW_ARITH, "diagnostic_pooling_rule"),
        (ROW_SINGLE_MEAN, "single_source_mean_reference"),
        (ROW_SINGLE_ORACLE, "diagnostic_only_oracle_reference"),
    ):
        rows.append(
            _dense_empty_row(
                cfg,
                experiment_seed=experiment_seed,
                heldout_center=heldout_center,
                replicate_seed=replicate_seed,
                candidates=candidates,
                prior_method=method,
                source_union_ref=source_union_ref,
                center_balanced_ref=center_balanced_ref,
                real_feature_bacc=real_feature_bacc,
                status=status,
                error_message=error_message,
                claim_role=role,
            )
        )
    return rows


def _ineligible_rows(
    cfg: DecentralizedK16GmmConfig,
    *,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    source_union_ref: ReferenceValue,
    center_balanced_ref: ReferenceValue,
    status: str,
    error_message: str,
) -> list[dict[str, object]]:
    rows = []
    for method, role in (
        (PRIMARY_DECENTRALIZED_METHOD, "primary_preservation_test"),
        (ROW_ARITH, "diagnostic_pooling_rule"),
        (ROW_SUPPORT_NELBO, "diagnostic_disabled"),
        (ROW_SINGLE_MEAN, "single_source_mean_reference"),
        (ROW_SINGLE_ORACLE, "diagnostic_only_oracle_reference"),
        (ROW_SHUFFLED_SUMMARY_CONTROL, "negative_control"),
        (ROW_SHUFFLED_LABEL_CONTROL, "negative_control"),
    ):
        rows.append(
            _dense_empty_row(
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
            )
        )
    rows.append(
        _reference_matrix_row(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            candidates=candidates,
            prior_method=ROW_SOURCE_UNION_K16_REFERENCE,
            reference=source_union_ref,
        )
    )
    rows.append(
        _reference_matrix_row(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            candidates=candidates,
            prior_method=ROW_CENTER_BALANCED_K16_REFERENCE,
            reference=center_balanced_ref,
        )
    )
    return rows


def _support_nelbo_disabled_row(
    cfg: DecentralizedK16GmmConfig,
    *,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    source_union_ref: ReferenceValue,
    center_balanced_ref: ReferenceValue,
    real_feature_bacc: float,
) -> dict[str, object]:
    status = "diagnostic_not_implemented" if cfg.support_nelbo_enabled else "diagnostic_disabled"
    message = (
        "support_nelbo_diagnostic_enabled_but_no_support_split_implementation"
        if cfg.support_nelbo_enabled
        else "support_nelbo_diagnostic.enabled=false"
    )
    return _dense_empty_row(
        cfg,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        candidates=candidates,
        prior_method=ROW_SUPPORT_NELBO,
        source_union_ref=source_union_ref,
        center_balanced_ref=center_balanced_ref,
        real_feature_bacc=real_feature_bacc,
        status=status,
        error_message=message,
        claim_role="support_nelbo_weighting_diagnostic_only",
    )


def _reference_matrix_row(
    cfg: DecentralizedK16GmmConfig,
    *,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    prior_method: str,
    reference: ReferenceValue,
) -> dict[str, object]:
    expert_id = POOL_SOURCE_UNION
    row = _base_matrix_row(
        cfg,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        candidates=candidates,
        expert_id=expert_id,
        expert_pool_type=POOL_SOURCE_UNION,
        prior_method=prior_method,
        pooling_rule="reference",
        source_union_ref=reference if prior_method == ROW_SOURCE_UNION_K16_REFERENCE else _missing_reference(),
        center_balanced_ref=reference if prior_method == ROW_CENTER_BALANCED_K16_REFERENCE else _missing_reference(),
    )
    row.update(
        {
            "variant_id": UNION_VARIANT,
            "bacc": reference.bacc if reference.status == "ok" else "",
            "macro_f1": reference.macro_f1 if reference.status == "ok" else "",
            "selection_source": DIAGNOSTIC_SELECTION,
            "status": reference.status,
            "error_message": reference.error_message,
            "claim_role": "centralized_reference_upper_bound_not_decentralized",
        }
    )
    return row


def _composition_status(
    cfg: DecentralizedK16GmmConfig,
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], SourceLocalSummary],
) -> tuple[str, str]:
    errors: list[str] = []
    for source_center in candidates:
        for cls in (0, 1):
            summary = summaries.get((str(source_center), int(cls)))
            if summary is None:
                errors.append(f"missing_summary_source_{source_center}_class_{cls}")
                continue
            if summary.status != "ok":
                errors.append(f"source_{source_center}_class_{cls}:{summary.error_message or summary.status}")
            if summary.effective_component_count != cfg.local_gmm_components_per_source_class:
                errors.append(f"source_{source_center}_class_{cls}_effective_component_count!={cfg.local_gmm_components_per_source_class}")
    if errors:
        return "ineligible_component_fit", "|".join(sorted(set(errors)))
    return "ok", ""


def _source_effective_components(
    summaries: Mapping[tuple[str, int], SourceLocalSummary],
    source_center: str,
    *,
    control_mode: str,
) -> int:
    values = []
    for cls in (0, 1):
        summary_cls = 1 - cls if control_mode == "class_flip" else cls
        summary = summaries.get((str(source_center), int(summary_cls)))
        if summary is not None:
            values.append(int(summary.effective_component_count))
    return min(values) if values else 0


def _coverage_row(
    cfg: DecentralizedK16GmmConfig,
    row: Mapping[str, object],
    component_counts: Mapping[int, Mapping[str, int]],
    *,
    candidates: Sequence[str],
) -> dict[str, object]:
    expected = {
        f"{cls}:{source}:{component}"
        for cls in (0, 1)
        for source in candidates
        for component in range(cfg.local_gmm_components_per_source_class)
    }
    sampled = {f"{cls}:{component}" for cls, counts in component_counts.items() for component in counts}
    counts = [int(v) for values in component_counts.values() for v in values.values()]
    total = float(sum(counts))
    fractions = [value / total for value in counts] if total else []
    entropy = -sum(p * math.log(p) for p in fractions if p > 0.0)
    unsampled = sorted(expected.difference(sampled))
    return {
        "experiment_seed": row["experiment_seed"],
        "heldout_center": row["heldout_center"],
        "expert_id": row["expert_id"],
        "expert_pool_type": row["expert_pool_type"],
        "variant_id": row["variant_id"],
        "prior_method": row["prior_method"],
        "replicate_seed": row["replicate_seed"],
        "generated_component_counts_json": json.dumps({str(cls): dict(values) for cls, values in component_counts.items()}, sort_keys=True),
        "num_active_components_unsampled": len(unsampled),
        "unsampled_active_components": "|".join(unsampled),
        "min_generated_samples_per_active_component": 0 if unsampled else (min(counts) if counts else 0),
        "component_weight_entropy": entropy,
        "component_mass_covered_by_generated_samples": 1.0 - (len(unsampled) / float(len(expected))) if expected else math.nan,
        "latent_component_undersampled": bool(unsampled),
    }


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


def _load_reference_values(
    artifact_root: Path | None,
    *,
    table_name: str,
    method: str,
    label: str,
) -> dict[tuple[str, str, str], ReferenceValue]:
    if artifact_root is None:
        return {}
    path = artifact_root / "tables" / table_name
    if not path.exists():
        return {}
    required = {
        "experiment_seed",
        "heldout_center",
        "replicate_seed",
        "prior_method",
        "bacc",
        "macro_f1",
        "status",
    }
    try:
        rows = _read_required_csv(path, required, label)
    except ProtocolError:
        return {}
    out: dict[tuple[str, str, str], ReferenceValue] = {}
    for row in rows:
        if row.get("prior_method") != method or row.get("status") != "ok":
            continue
        key = (str(row["experiment_seed"]), str(row["heldout_center"]), str(row["replicate_seed"]))
        out[key] = ReferenceValue(bacc=float(row["bacc"]), macro_f1=float(row["macro_f1"]), status="ok")
    return out


def _reference_for_cell(
    references: Mapping[tuple[str, str, str], ReferenceValue],
    experiment_seed: object,
    heldout_center: object,
    replicate_seed: object,
) -> ReferenceValue:
    return references.get(
        (str(experiment_seed), str(heldout_center), str(replicate_seed)),
        ReferenceValue(math.nan, math.nan, "missing_reference", "reference_artifact_missing_or_cell_absent"),
    )


def _missing_reference() -> ReferenceValue:
    return ReferenceValue(math.nan, math.nan, "missing_reference", "reference_not_applicable")


def _validate_optional_leakage_report(root: Path | None, protocol_violations: list[str]) -> None:
    if root is None:
        return
    path = root / "reports" / "leakage_report.json"
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        protocol_violations.append(f"Could not read optional reference leakage report {path}: {exc}")
        return
    if payload.get("status") != "PASS":
        protocol_violations.append(f"Optional reference leakage report is not PASS: {path}")


def _populate_negative_control_gaps(rows: list[dict[str, object]]) -> None:
    controls: dict[tuple[str, str, str], float] = {}
    for row in rows:
        if row.get("prior_method") not in {ROW_SHUFFLED_SUMMARY_CONTROL, ROW_SHUFFLED_LABEL_CONTROL}:
            continue
        key = (str(row.get("experiment_seed")), str(row.get("heldout_center")), str(row.get("replicate_seed")))
        value = _float(row.get("bacc"))
        if math.isfinite(value):
            controls[key] = max(controls.get(key, -math.inf), value)
    for row in rows:
        if row.get("prior_method") != PRIMARY_DECENTRALIZED_METHOD:
            continue
        key = (str(row.get("experiment_seed")), str(row.get("heldout_center")), str(row.get("replicate_seed")))
        control = controls.get(key, math.nan)
        value = _float(row.get("bacc"))
        if math.isfinite(value) and math.isfinite(control):
            row["negative_control_gap"] = value - control


def _decision(
    rows: Sequence[Mapping[str, object]],
    cfg: DecentralizedK16GmmConfig,
    *,
    leakage_status: str,
) -> dict[str, object]:
    primary_all = _rows_for(rows, PRIMARY_DECENTRALIZED_METHOD, include_non_ok=True)
    primary = _rows_for(rows, PRIMARY_DECENTRALIZED_METHOD)
    controls = [row for row in rows if row.get("prior_method") in {ROW_SHUFFLED_SUMMARY_CONTROL, ROW_SHUFFLED_LABEL_CONTROL} and row.get("status") == "ok"]
    single_mean = _rows_for(rows, ROW_SINGLE_MEAN)
    single_oracle = _rows_for(rows, ROW_SINGLE_ORACLE)
    real_feature = _rows_for(rows, ROW_REAL_FEATURE_DENSE_REFERENCE)
    source_union = _rows_for(rows, ROW_SOURCE_UNION_K16_REFERENCE)
    center_balanced = _rows_for(rows, ROW_CENTER_BALANCED_K16_REFERENCE)
    stats = _primary_stats(primary)
    single_mean_stats = _primary_stats(single_mean)
    single_oracle_stats = _primary_stats(single_oracle)
    real_stats = _primary_stats(real_feature)
    source_union_stats = _primary_stats(source_union)
    center_balanced_stats = _primary_stats(center_balanced)
    control_stats = _primary_stats(controls)
    fit_ineligible = any(row.get("status") == "ineligible_component_fit" for row in primary_all)
    negative_control_competitive = (
        math.isfinite(_float(control_stats["center_equal_mean_bacc"]))
        and math.isfinite(_float(stats["center_equal_mean_bacc"]))
        and _float(stats["center_equal_mean_bacc"]) - _float(control_stats["center_equal_mean_bacc"]) <= 0.05
    )
    retention_source_union = _retention(_float(stats["center_equal_mean_bacc"]), _float(source_union_stats["center_equal_mean_bacc"]))
    retention_center_balanced = _retention(
        _float(stats["center_equal_mean_bacc"]),
        _float(center_balanced_stats["center_equal_mean_bacc"]),
    )
    delta_vs_mean_single = _float(stats["center_equal_mean_bacc"]) - _float(single_mean_stats["center_equal_mean_bacc"])
    delta_vs_oracle_single = _float(stats["center_equal_mean_bacc"]) - _float(single_oracle_stats["center_equal_mean_bacc"])
    delta_vs_real = _float(stats["center_equal_mean_bacc"]) - _float(real_stats["center_equal_mean_bacc"])
    references_ok = bool(source_union) or bool(center_balanced)
    pass_refs = (
        (not source_union or retention_source_union >= 0.95)
        and (not center_balanced or retention_center_balanced >= 0.95)
    )
    primary_pass = (
        leakage_status == "PASS"
        and not fit_ineligible
        and int(stats["n_heldout_centers"]) >= len(cfg.heldout_centers)
        and int(stats["min_eligible_seeds_per_center"]) >= 1
        and _float(stats["center_equal_mean_bacc"]) >= 0.85
        and _float(stats["min_center_mean_bacc"]) >= 0.75
        and _float(stats["seed_std_bacc"]) <= 0.06
        and pass_refs
        and delta_vs_mean_single > 0.0
        and not negative_control_competitive
    )
    partial = (
        leakage_status == "PASS"
        and not fit_ineligible
        and delta_vs_mean_single > 0.0
        and _float(stats["min_center_mean_bacc"]) > _float(single_mean_stats["min_center_mean_bacc"])
    )
    negative = (
        leakage_status == "PASS"
        and not fit_ineligible
        and not primary_pass
        and (
            (math.isfinite(retention_source_union) and retention_source_union < 0.95)
            or (math.isfinite(retention_center_balanced) and retention_center_balanced < 0.95)
        )
        and math.isfinite(delta_vs_oracle_single)
        and delta_vs_oracle_single <= 0.0
    )
    verdict = "D1_FAIL"
    if leakage_status != "PASS":
        verdict = "PROTOCOL_FAIL"
    elif fit_ineligible:
        verdict = "INELIGIBLE"
    elif primary_pass:
        verdict = "D1_PASS"
    elif partial:
        verdict = "D1_PARTIAL_EVIDENCE"
    elif negative:
        verdict = "D1_NEGATIVE_EVIDENCE"
    elif int(stats["n_heldout_centers"]) < len(cfg.heldout_centers):
        verdict = "TARGET_EVAL_INSUFFICIENT"

    flags = []
    if fit_ineligible:
        flags.append("INELIGIBLE_COMPONENT_FIT")
    if negative_control_competitive:
        flags.append("NEGATIVE_CONTROL_COMPETITIVE")
    if not references_ok:
        flags.append("CENTRALIZED_REFERENCE_MISSING")
    if not real_feature:
        flags.append("REAL_FEATURE_REFERENCE_MISSING")
    if math.isfinite(retention_source_union) and retention_source_union < 0.95:
        flags.append("SOURCE_UNION_RETENTION_BELOW_0P95")
    if math.isfinite(retention_center_balanced) and retention_center_balanced < 0.95:
        flags.append("CENTER_BALANCED_RETENTION_BELOW_0P95")
    if math.isfinite(delta_vs_oracle_single) and delta_vs_oracle_single <= 0.0:
        flags.append("DOES_NOT_BEAT_SINGLE_SOURCE_ORACLE")

    return {
        "primary_verdict": verdict,
        "diagnostic_flags": "|".join(flags),
        "leakage_status": leakage_status,
        "primary_method": PRIMARY_DECENTRALIZED_METHOD,
        "center_equal_mean_bacc": stats["center_equal_mean_bacc"],
        "center_equal_macro_f1": stats["center_equal_macro_f1"],
        "min_center_bacc": stats["min_center_mean_bacc"],
        "seed_std_bacc": stats["seed_std_bacc"],
        "delta_vs_mean_single_source_k4": delta_vs_mean_single,
        "delta_vs_single_source_oracle_reference": delta_vs_oracle_single,
        "retention_vs_source_union_k16": retention_source_union,
        "retention_vs_center_balanced_k16": retention_center_balanced,
        "delta_vs_real_source_embedding_dense_reference": delta_vs_real,
        "negative_control_gap": _float(stats["center_equal_mean_bacc"]) - _float(control_stats["center_equal_mean_bacc"]),
        "mean_single_source_k4_center_equal_mean_bacc": single_mean_stats["center_equal_mean_bacc"],
        "single_source_oracle_center_equal_mean_bacc": single_oracle_stats["center_equal_mean_bacc"],
        "source_union_k16_reference_center_equal_mean_bacc": source_union_stats["center_equal_mean_bacc"],
        "center_balanced_k16_reference_center_equal_mean_bacc": center_balanced_stats["center_equal_mean_bacc"],
        "real_feature_dense_reference_center_equal_mean_bacc": real_stats["center_equal_mean_bacc"],
        "negative_control_center_equal_mean_bacc": control_stats["center_equal_mean_bacc"],
        **stats,
    }


def _primary_stats(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    grouped = _replicate_averaged(rows)
    by_seed: dict[str, list[Mapping[str, object]]] = {}
    by_center: dict[str, list[Mapping[str, object]]] = {}
    for row in grouped:
        by_seed.setdefault(str(row["experiment_seed"]), []).append(row)
        by_center.setdefault(str(row["heldout_center"]), []).append(row)
    seed_means = [_mean_field(values, "bacc") for values in by_seed.values()]
    center_bacc = {center: _mean_field(values, "bacc") for center, values in sorted(by_center.items())}
    return {
        "n_raw_rows": len(rows),
        "n_decision_cells": len(grouped),
        "n_experiment_seeds": len(by_seed),
        "n_heldout_centers": len(by_center),
        "min_eligible_seeds_per_center": min((len({str(row["experiment_seed"]) for row in values}) for values in by_center.values()), default=0),
        "center_equal_mean_bacc": nanmean(seed_means) if seed_means else math.nan,
        "center_equal_macro_f1": _center_equal_mean(grouped, "macro_f1"),
        "seed_std_bacc": _std(seed_means),
        "min_center_mean_bacc": min(center_bacc.values()) if center_bacc else math.nan,
        "min_cell_bacc": _min_field(grouped, "bacc"),
        "per_center_bacc": json.dumps(center_bacc, sort_keys=True),
        "per_seed_bacc": json.dumps({seed: _mean_field(values, "bacc") for seed, values in sorted(by_seed.items())}, sort_keys=True),
    }


def _replicate_averaged(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        groups.setdefault((str(row["experiment_seed"]), str(row["heldout_center"])), []).append(row)
    out: list[dict[str, object]] = []
    for (seed, center), subset in groups.items():
        out.append(
            {
                "experiment_seed": seed,
                "heldout_center": center,
                "bacc": _mean_field(subset, "bacc"),
                "macro_f1": _mean_field(subset, "macro_f1"),
            }
        )
    return out


def _rows_for(rows: Sequence[Mapping[str, object]], method: str, *, include_non_ok: bool = False) -> list[Mapping[str, object]]:
    return [
        row for row in rows
        if row.get("prior_method") == method
        and (include_non_ok or row.get("status") == "ok")
    ]


def _center_equal_mean(rows: Sequence[Mapping[str, object]], field: str) -> float:
    by_seed: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        by_seed.setdefault(str(row["experiment_seed"]), []).append(row)
    return nanmean([_mean_field(values, field) for values in by_seed.values()]) if by_seed else math.nan


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


def _write_artifacts(
    root: Path,
    cfg: DecentralizedK16GmmConfig,
    *,
    matrix_rows: Sequence[Mapping[str, object]],
    gap_rows: Sequence[Mapping[str, object]],
    summary_manifest_rows: Sequence[Mapping[str, object]],
    composition_rows: Sequence[Mapping[str, object]],
    diagnostic_rows: Sequence[Mapping[str, object]],
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
    write_csv_rows(root / "tables" / "decentralized_k16_downstream_matrix.csv", matrix_rows, columns=_matrix_columns())
    write_csv_rows(root / "tables" / "decentralized_k16_gap_summary.csv", gap_rows, columns=_matrix_columns())
    write_csv_rows(root / "tables" / "decentralized_k16_summary.csv", [dict(decision)])
    write_csv_rows(root / "tables" / "exported_source_summary_manifest.csv", summary_manifest_rows, columns=_summary_manifest_columns())
    write_csv_rows(root / "tables" / "composed_prior_component_manifest.csv", composition_rows, columns=_composition_columns())
    write_csv_rows(root / "tables" / "source_summary_diagnostics.csv", diagnostic_rows, columns=_diagnostic_columns())
    write_csv_rows(root / "tables" / "late_aggregation_matrix.csv", late_rows, columns=_matrix_columns())
    write_csv_rows(root / "tables" / "real_feature_reference_matrix.csv", real_feature_rows, columns=_matrix_columns())
    write_csv_rows(root / "tables" / "generated_component_coverage_audit.csv", coverage_rows)
    write_csv_rows(root / "tables" / "weak_source_audit.csv", weak_rows)
    write_csv_rows(root / "tables" / "nearest_neighbor_memorization_audit.csv", nn_rows)
    write_csv_rows(root / "tables" / "negative_control_summary.csv", [_negative_control_summary(decision)])
    write_csv_rows(root / "manifests" / "decentralized_k16_prior_model_manifest.csv", model_manifest_rows)
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
            "schema_version": "cvae_rebuild_decentralized_k16_gmm_prior_protocol_manifest_v1",
            "experiment_name": cfg.name,
            "experiment_type": "decentralized_k16_prior_composition_preservation_test",
            "primary_variant": cfg.primary_variant,
            "primary_method": cfg.primary_method,
            "target_support_labels_for_selection": False,
            "target_eval_labels_for_scoring_only": True,
            "target_expert_excluded": target_expert_excluded,
            "exported_source_summaries_are_target_agnostic": True,
            "composition_manifests_are_fold_specific": True,
            "raw_source_embedding_pooling_for_prior_fit": False,
            "support_nelbo_weighting_primary": False,
            "support_nelbo_weighting_diagnostic_enabled": cfg.support_nelbo_enabled,
            "source_union_references_diagnostic_only": True,
            "oracle_rows_diagnostic_only": True,
            "protocol_wording": PROTOCOL_WORDING,
            "claim_boundary": (
                "decentralized prior-composition preservation test only; no target-specific "
                "compatibility routing claim, no support-NELBO downstream claim, and no formal privacy claim"
            ),
        },
    )
    _write_decision_summary(root, decision, leakage_status=leakage_status)
    write_json(root / "run_config_resolved.yaml", _resolved_config(cfg))


def _matrix_columns() -> tuple[str, ...]:
    return (
        "experiment_seed",
        "heldout_center",
        "expert_id",
        "expert_pool_type",
        "variant_id",
        "prior_method",
        "gmm_components",
        "effective_gmm_components",
        "local_gmm_components_per_source_class",
        "composed_components_per_class",
        "source_weighting",
        "pooling_rule",
        "replicate_seed",
        "latent_sample_seed",
        "included_source_centers",
        "num_included_sources",
        "synthetic_per_class_total",
        "synthetic_per_class_per_source_json",
        "bacc",
        "macro_f1",
        "source_union_k16_bacc",
        "center_balanced_k16_bacc",
        "real_feature_dense_bacc",
        "mean_single_source_k4_bacc",
        "oracle_single_source_k4_bacc",
        "delta_vs_mean_single_source_k4",
        "delta_vs_single_source_oracle_reference",
        "retention_vs_source_union_k16",
        "retention_vs_center_balanced_k16",
        "delta_vs_real_source_embedding_dense_reference",
        "negative_control_gap",
        "generated_features_hash",
        "prediction_hash",
        "composed_prior_hash",
        "summary_set_hash",
        "selection_source",
        "status",
        "error_message",
        "claim_role",
    )


def _summary_manifest_columns() -> tuple[str, ...]:
    return (
        "experiment_seed",
        "source_center",
        "class_label",
        "component_id",
        "component_weight_local",
        "latent_mean_hash",
        "latent_diag_var_hash",
        "source_class_count",
        "expert_config_hash",
        "summary_schema_version",
        "summary_path",
        "summary_hash",
        "status",
        "error_message",
    )


def _composition_columns() -> tuple[str, ...]:
    return (
        "experiment_seed",
        "heldout_center",
        "included_source_centers",
        "class_label",
        "composed_component_id",
        "source_center",
        "source_component_id",
        "component_weight_local",
        "component_weight_after_equal_source_normalization",
        "summary_hash",
        "summary_status",
        "summary_error_message",
    )


def _diagnostic_columns() -> tuple[str, ...]:
    return (
        "experiment_seed",
        "source_center",
        "class_label",
        "local_gmm_components",
        "source_class_count",
        "effective_component_count",
        "min_component_weight",
        "min_required_component_weight",
        "variance_floor",
        "min_diag_var",
        "all_finite",
        "component_entropy",
        "gmm_converged",
        "gmm_n_iter",
        "source_train_log_likelihood",
        "source_inner_bic",
        "fit_row_ids_hash",
        "parameter_hash",
        "summary_path",
        "summary_hash",
        "shuffled_label_control",
        "status",
        "error_message",
    )


def _negative_control_summary(decision: Mapping[str, object]) -> dict[str, object]:
    return {
        "primary_method": PRIMARY_DECENTRALIZED_METHOD,
        "control_methods": f"{ROW_SHUFFLED_SUMMARY_CONTROL}|{ROW_SHUFFLED_LABEL_CONTROL}",
        "primary_center_equal_mean_bacc": decision.get("center_equal_mean_bacc", math.nan),
        "control_center_equal_mean_bacc": decision.get("negative_control_center_equal_mean_bacc", math.nan),
        "negative_control_gap": decision.get("negative_control_gap", math.nan),
        "control_competitive": "NEGATIVE_CONTROL_COMPETITIVE" in str(decision.get("diagnostic_flags", "")),
    }


def _write_decision_summary(root: Path, decision: Mapping[str, object], *, leakage_status: str) -> None:
    text = "\n".join(
        [
            "# D1: Decentralized K16 Prior-Composition Preservation Test",
            "",
            "## Summary",
            "",
            f"- Primary method: `{PRIMARY_DECENTRALIZED_METHOD}`",
            f"- Primary verdict: `{decision.get('primary_verdict', 'D1_FAIL')}`",
            f"- Diagnostic flags: `{decision.get('diagnostic_flags', '')}`",
            f"- Center-equal mean BACC: {_format_float(decision.get('center_equal_mean_bacc'))}",
            f"- Center-equal macro-F1: {_format_float(decision.get('center_equal_macro_f1'))}",
            f"- Min center BACC: {_format_float(decision.get('min_center_bacc'))}",
            f"- Seed std BACC: {_format_float(decision.get('seed_std_bacc'))}",
            f"- Delta vs mean single-source K4: {_format_float(decision.get('delta_vs_mean_single_source_k4'))}",
            f"- Retention vs source-union K16: {_format_float(decision.get('retention_vs_source_union_k16'))}",
            f"- Retention vs center-balanced K16: {_format_float(decision.get('retention_vs_center_balanced_k16'))}",
            f"- Delta vs real-feature dense reference: {_format_float(decision.get('delta_vs_real_source_embedding_dense_reference'))}",
            f"- Negative-control gap: {_format_float(decision.get('negative_control_gap'))}",
            f"- Leakage status: `{leakage_status}`",
            "",
            "## Protocol Boundary",
            "",
            PROTOCOL_WORDING,
            "",
            "This is decentralized prior composition plus dense output aggregation.",
            "It is not a target-specific compatibility-routing result.",
            "It does not prove support-NELBO improves downstream utility.",
            "The source-union K16 rows are centralized diagnostic references only.",
            "",
            "## Supported Claim If PASS",
            "",
            "Source-local latent GMM summaries can preserve most of the centralized Virchow2 K16 CVAE prior's downstream utility under a raw-data-free summary-exchange protocol with dense expert aggregation.",
            "",
        ]
    )
    path = root / "reports" / "decision_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _resolved_config(cfg: DecentralizedK16GmmConfig) -> dict[str, object]:
    return {
        "name": cfg.name,
        "artifact_root": str(cfg.artifact_root),
        "repair_artifact_root": str(cfg.repair_artifact_root),
        "sampling_artifact_root": "" if cfg.sampling_artifact_root is None else str(cfg.sampling_artifact_root),
        "prior_calibration_artifact_root": "" if cfg.prior_calibration_artifact_root is None else str(cfg.prior_calibration_artifact_root),
        "covariance_confirmation_artifact_root": "" if cfg.covariance_confirmation_artifact_root is None else str(cfg.covariance_confirmation_artifact_root),
        "source_union_gmm_artifact_root": "" if cfg.source_union_gmm_artifact_root is None else str(cfg.source_union_gmm_artifact_root),
        "balanced_gmm_artifact_root": "" if cfg.balanced_gmm_artifact_root is None else str(cfg.balanced_gmm_artifact_root),
        "feature_cache_root": str(cfg.feature_cache_root),
        "backbone": cfg.backbone,
        "experiment_seeds": list(cfg.experiment_seeds),
        "heldout_centers": list(cfg.heldout_centers),
        "replicate_seeds": list(cfg.replicate_seeds),
        "synthetic_per_class_total": cfg.synthetic_per_class_total,
        "primary_variant": cfg.primary_variant,
        "primary_method": cfg.primary_method,
        "local_gmm_components_per_source_class": cfg.local_gmm_components_per_source_class,
        "composed_components_per_class": cfg.composed_components_per_class,
        "source_weighting": cfg.source_weighting,
        "gmm_covariance_type": cfg.gmm_covariance_type,
        "gmm_reg_covar": cfg.gmm_reg_covar,
        "gmm_n_init": cfg.gmm_n_init,
        "gmm_max_iter": cfg.gmm_max_iter,
        "min_count_for_k4": cfg.min_count_for_k4,
        "min_component_weight": cfg.min_component_weight,
        "variance_floor": cfg.variance_floor,
        "primary_pooling": cfg.primary_pooling,
        "support_nelbo_diagnostic": {"enabled": cfg.support_nelbo_enabled},
        "classifier": {
            "type": cfg.classifier_type,
            "solver": cfg.classifier_solver,
            "C": cfg.classifier_c,
            "max_iter": cfg.classifier_max_iter,
            "class_weight": cfg.classifier_class_weight,
            "classifier_seed": cfg.classifier_seed,
        },
    }


def _repair_runtime_config(cfg: DecentralizedK16GmmConfig, root: Path) -> RepairConfig:
    return RepairConfig(
        name="virchow2_cvae_preservation_repair_v1",
        artifact_root=root,
        feature_cache_root=cfg.feature_cache_root,
        experiment_seeds=cfg.experiment_seeds,
        heldout_centers=cfg.heldout_centers,
        replicate_seeds=cfg.replicate_seeds,
        synthetic_per_class_total=cfg.synthetic_per_class_total,
        primary_variant=PRIMARY_VARIANT,
        min_decision_rows=10,
        variants=(_per_source_variant(),),
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


def _arithmetic_probability_pool(bundles: Sequence[PredictionBundle]) -> tuple[tuple[float, ...], ...]:
    if not bundles:
        raise ProtocolError("At least one prediction bundle is required.")
    classes = bundles[0].classes
    n_rows = len(bundles[0].probabilities)
    for bundle in bundles:
        if bundle.classes != classes:
            raise ProtocolError("Class order mismatch in arithmetic pooling.")
        if len(bundle.probabilities) != n_rows:
            raise ProtocolError("Prediction row count mismatch in arithmetic pooling.")
    out = []
    for row_idx in range(n_rows):
        values = [0.0 for _ in classes]
        for bundle in bundles:
            for cls_idx, prob in enumerate(bundle.probabilities[row_idx]):
                values[cls_idx] += float(prob)
        denom = sum(values)
        out.append(tuple((value / denom) if denom else (1.0 / len(values)) for value in values))
    return tuple(out)


def _normalized_weights(weights: object):
    import numpy as np  # type: ignore

    w = np.asarray(weights, dtype=float)
    total = float(w.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ProtocolError("Component weights must sum to a positive finite value.")
    return w / total


def _entropy(weights: object) -> float:
    try:
        w = _normalized_weights(weights)
    except Exception:
        return math.nan
    return float(-sum(float(p) * math.log(float(p)) for p in w if float(p) > 0.0))


def _component_count(summary: SourceLocalSummary) -> int:
    try:
        import numpy as np  # type: ignore

        return int(np.asarray(summary.weights).shape[0])
    except Exception:
        return 0


def _as_float_list(values: object) -> list[float]:
    try:
        import numpy as np  # type: ignore

        return [float(v) for v in np.ravel(np.asarray(values, dtype=float)).tolist()]
    except Exception:
        return []


def _flatten_payload(values: Sequence[object]) -> object:
    import numpy as np  # type: ignore

    if not values:
        return np.asarray([], dtype=float)
    return np.concatenate([np.ravel(np.asarray(value, dtype=float)) for value in values])


def _summary_set_hash(
    summaries: Mapping[tuple[str, int], SourceLocalSummary],
    candidates: Sequence[str],
    *,
    control_mode: str,
) -> str:
    parts = []
    for source_center in candidates:
        for cls in (0, 1):
            summary_cls = 1 - cls if control_mode == "class_flip" else cls
            summary = summaries.get((str(source_center), int(summary_cls)))
            parts.append("" if summary is None else summary.summary_hash)
    return _hash_strings(parts)


def _budget_json(cfg: DecentralizedK16GmmConfig, candidates: Sequence[str]) -> str:
    budgets = _balanced_counts(cfg.synthetic_per_class_total, len(candidates))
    return json.dumps({str(source): int(budget) for source, budget in zip(candidates, budgets)}, sort_keys=True)


def _balanced_counts(total: int, n_groups: int) -> list[int]:
    base = int(total) // int(n_groups)
    rem = int(total) % int(n_groups)
    return [base + (1 if idx < rem else 0) for idx in range(int(n_groups))]


def _retention(value: object, reference: object) -> float:
    val = _float(value)
    ref = _float(reference)
    return val / ref if math.isfinite(val) and math.isfinite(ref) and ref > 0.0 else math.nan


def _expert_config_hash(runtime: VariantRuntime) -> str:
    return _hash_strings([json.dumps(runtime.variant.__dict__, sort_keys=True)])


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _latent_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)
