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
)
from features import load_feature_cache, select_rows
from metrics import nanmean
from preservation import _hash_array
from preservation_repair import (
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
from preservation_sampling import (
    DIAGNOSTIC_SELECTION,
    PRIMARY_SELECTION,
    UNION_VARIANT,
    RuntimeSource,
    _manifest_row,
    _per_source_variant,
    _runtime_source,
)
from prior_calibration import _decode_latents
from protocol import ProtocolError, assert_candidate_pool, build_leakage_report
from reporting import prepare_artifact_dirs, write_csv_rows, write_protocol_finalization
from source_union_gmm_prior import _nearest_neighbor_row
from splits import candidate_experts

import decentralized_k16_gmm_prior as d1


ADAPTIVE_NAME = "virchow2_cvae_decentralized_adaptive_gmm_prior_v1"
PRIMARY_ADAPTIVE_METHOD = "decentralized_exported_adaptive_k_cc_diag_gmm_late_geom"
ROW_ARITH = "decentralized_exported_adaptive_k_cc_diag_gmm_late_arith"
ROW_BIC = "decentralized_exported_bic_selected_cc_diag_gmm_late_geom"
ROW_SINGLE_MEAN = "per_source_exported_adaptive_k_cc_diag_gmm_single_expert_mean_reference"
ROW_SINGLE_ORACLE = "per_source_exported_adaptive_k_cc_diag_gmm_single_expert_oracle_reference"
ROW_SOURCE_UNION_K16_REFERENCE = "source_union_cc_diag_gmm_k16_prior_sample_reference"
ROW_CENTER_BALANCED_K16_REFERENCE = "source_union_center_balanced_cc_diag_gmm_k16_prior_sample_reference"
ROW_REAL_FEATURE_DENSE_REFERENCE = "real_source_embedding_classifier_dense_reference"
ROW_SHUFFLED_SUMMARY_CONTROL = "decentralized_adaptive_k_shuffled_summary_control"
ROW_SHUFFLED_LABEL_CONTROL = "decentralized_adaptive_k_shuffled_label_control"
POOL_DECENTRALIZED = "decentralized_source_summary"
SUMMARY_SCHEMA_VERSION = "decentralized_source_local_cc_diag_gmm_adaptive_summary_v1"
PROTOCOL_WORDING = (
    "This is a data-minimizing, raw-data-free source-local latent summary-exchange protocol. "
    "It is not a formal differential privacy claim. Exported latent summary statistics may still "
    "contain distributional information derived from private data."
)


@dataclass(frozen=True)
class DecentralizedAdaptiveGmmConfig:
    name: str
    artifact_root: Path
    repair_artifact_root: Path
    strict_d1_artifact_root: Path | None
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
    bic_method: str
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
class AdaptiveSourceLocalSummary:
    experiment_seed: int
    source_center: str
    class_label: int
    selection_rule: str
    selected_k: int
    selected_k_reason: str
    candidate_fit_status_json: str
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


def load_decentralized_adaptive_gmm_prior_config(path: str | Path) -> DecentralizedAdaptiveGmmConfig:
    source = Path(path).resolve()
    data = _load_mapping(source)
    base_dir = source.parents[2] if len(source.parents) >= 3 else source.parent
    return parse_decentralized_adaptive_gmm_prior_config(data, base_dir=base_dir)


def parse_decentralized_adaptive_gmm_prior_config(
    data: Mapping[str, Any],
    *,
    base_dir: str | Path = ".",
) -> DecentralizedAdaptiveGmmConfig:
    base = Path(base_dir)
    experiment = _mapping(data, "experiment")
    inputs = _mapping(data, "inputs")
    run = _mapping(data, "run_matrix")
    generation = _mapping(data, "generation")
    gmm = _mapping(data, "adaptive_gmm_prior")
    classifier = _mapping(data, "classifier")
    cfg = DecentralizedAdaptiveGmmConfig(
        name=str(experiment["name"]),
        artifact_root=_path(base, str(experiment["artifact_root"])),
        repair_artifact_root=_path(base, str(inputs["repair_artifact_root"])),
        strict_d1_artifact_root=_optional_path(base, inputs.get("strict_d1_artifact_root")),
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
        bic_method=str(gmm.get("bic_method", ROW_BIC)),
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
        classifier_type=str(classifier["type"]),
        classifier_solver=str(classifier["solver"]),
        classifier_c=float(classifier["C"]),
        classifier_max_iter=int(classifier["max_iter"]),
        classifier_class_weight=str(classifier["class_weight"]),
        classifier_seed=None if classifier.get("classifier_seed") is None else int(classifier["classifier_seed"]),
    )
    validate_decentralized_adaptive_gmm_prior_config(cfg)
    return cfg


def validate_decentralized_adaptive_gmm_prior_config(cfg: DecentralizedAdaptiveGmmConfig) -> None:
    if cfg.name != ADAPTIVE_NAME:
        raise ProtocolError(f"Adaptive decentralized experiment name must be {ADAPTIVE_NAME!r}.")
    if cfg.backbone != "virchow2":
        raise ProtocolError("D1.1 is locked to backbone=virchow2.")
    if cfg.primary_variant != PRIMARY_VARIANT:
        raise ProtocolError(f"primary_variant must be {PRIMARY_VARIANT!r}.")
    if cfg.primary_method != PRIMARY_ADAPTIVE_METHOD:
        raise ProtocolError(f"primary_method must be {PRIMARY_ADAPTIVE_METHOD!r}.")
    if cfg.bic_method != ROW_BIC:
        raise ProtocolError(f"bic_method must be {ROW_BIC!r}.")
    if cfg.candidate_components_per_source_class != (4, 3, 2, 1):
        raise ProtocolError("candidate_components_per_source_class must be locked to [4, 3, 2, 1].")
    if len(cfg.heldout_centers) != 5:
        raise ProtocolError("D1.1 composition expects exactly five centers, giving four source experts per fold.")
    if cfg.min_samples_per_component < 1:
        raise ProtocolError("min_samples_per_component must be positive.")
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
    if cfg.gmm_n_init < 1 or cfg.gmm_max_iter < 1:
        raise ProtocolError("GMM n_init/max_iter must be positive.")
    if cfg.classifier_type != "sklearn_logistic_regression":
        raise ProtocolError("classifier.type must be sklearn_logistic_regression.")
    if cfg.classifier_solver != "lbfgs" or cfg.classifier_c != 1.0 or cfg.classifier_max_iter != 2000:
        raise ProtocolError("Classifier solver/C/max_iter must remain locked.")
    if cfg.classifier_class_weight != "balanced" or cfg.classifier_seed is not None:
        raise ProtocolError("Classifier must use class_weight=balanced and classifier_seed=null.")


def run_decentralized_adaptive_gmm_prior(
    cfg: DecentralizedAdaptiveGmmConfig,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    root = prepare_artifact_dirs(Path(artifact_root) if artifact_root is not None else cfg.artifact_root)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "summaries").mkdir(parents=True, exist_ok=True)

    matrix_rows: list[dict[str, object]] = []
    summary_manifest_rows: list[dict[str, object]] = []
    composition_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    intervention_rows: list[dict[str, object]] = []
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
            largest_summaries: dict[tuple[str, int], AdaptiveSourceLocalSummary] = {}
            bic_summaries: dict[tuple[str, int], AdaptiveSourceLocalSummary] = {}
            shuffled_summaries: dict[tuple[str, int], AdaptiveSourceLocalSummary] = {}
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

                largest, bic = _fit_and_export_source_summaries(
                    cfg,
                    root,
                    runtime_source.runtime,
                    experiment_seed=int(experiment_seed),
                    shuffled_label_control=False,
                )
                shuffled, _shuffled_bic = _fit_and_export_source_summaries(
                    cfg,
                    root,
                    runtime_source.runtime,
                    experiment_seed=int(experiment_seed),
                    shuffled_label_control=True,
                )
                for summary in largest:
                    largest_summaries[(summary.source_center, summary.class_label)] = summary
                    summary_manifest_rows.append(_summary_manifest_row(summary))
                    diagnostic_rows.append(_summary_diagnostic_row(cfg, summary))
                for summary in bic:
                    bic_summaries[(summary.source_center, summary.class_label)] = summary
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

                composition_rows.extend(
                    _composition_manifest_rows(
                        cfg,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        candidates=candidates,
                        summaries=largest_summaries,
                    )
                )
                intervention_rows.extend(
                    _intervention_rows(
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        candidates=candidates,
                        summaries=largest_summaries,
                    )
                )

                target_indices = _target_indices(test_cache.metadata, str(heldout_center))
                eval_raw, eval_meta = select_rows(test_cache.embeddings, test_cache.metadata, target_indices)
                eval_labels = tuple(_label(row) for row in eval_meta)
                eval_error = "mono_class_target_eval" if len(set(eval_labels)) < 2 else ""

                for replicate_seed in cfg.replicate_seeds:
                    su_ref = d1._reference_for_cell(source_union_refs, experiment_seed, heldout_center, replicate_seed)
                    cb_ref = d1._reference_for_cell(center_balanced_refs, experiment_seed, heldout_center, replicate_seed)
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

                    ref_row, real_late = _real_feature_reference(
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
                        summaries=largest_summaries,
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

                    rows, late, coverage, weak, nn = _evaluate_bic_diagnostic(
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
                    matrix_rows.extend(rows)
                    late_rows.extend(late)
                    coverage_rows.extend(coverage)
                    weak_rows.extend(weak)
                    nn_rows.extend(nn)

                    for prior_method, summaries, control_mode in (
                        (ROW_SHUFFLED_SUMMARY_CONTROL, largest_summaries, "class_flip"),
                        (ROW_SHUFFLED_LABEL_CONTROL, shuffled_summaries, "normal"),
                    ):
                        rows, late, coverage, weak, nn = _evaluate_control(
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
                            prior_method=prior_method,
                            control_mode=control_mode,
                        )
                        matrix_rows.extend(rows)
                        late_rows.extend(late)
                        coverage_rows.extend(coverage)
                        weak_rows.extend(weak)
                        nn_rows.extend(nn)

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
    strict_d1 = _load_strict_d1_summary(cfg.strict_d1_artifact_root)
    decision = _decision(
        matrix_rows,
        cfg,
        leakage_status=leakage.status,
        diagnostic_rows=diagnostic_rows,
        strict_d1=strict_d1,
    )
    _write_artifacts(
        root,
        cfg,
        matrix_rows=matrix_rows,
        gap_rows=gap_rows,
        summary_manifest_rows=summary_manifest_rows,
        composition_rows=composition_rows,
        diagnostic_rows=diagnostic_rows,
        intervention_rows=intervention_rows,
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
    cfg: DecentralizedAdaptiveGmmConfig,
    root: Path,
    runtime: VariantRuntime,
    *,
    experiment_seed: int,
    shuffled_label_control: bool,
) -> tuple[tuple[AdaptiveSourceLocalSummary, ...], tuple[AdaptiveSourceLocalSummary, ...]]:
    import torch  # type: ignore
    from sklearn.mixture import GaussianMixture  # type: ignore

    source_centers = {str(v) for v in runtime.source_train_centers}
    if len(source_centers) != 1 or runtime.expert_id not in source_centers:
        raise ProtocolError("Source-local summaries must be fitted from exactly one source center.")
    x_np = np.asarray(runtime.source_train_embeddings, dtype=np.float32)
    y_np = np.asarray(runtime.source_train_labels, dtype=int)
    fit_labels = y_np.copy()
    if shuffled_label_control:
        rng = np.random.default_rng(d1._latent_seed(experiment_seed, runtime.expert_id, "adaptive_shuffled_label_summary"))
        rng.shuffle(fit_labels)
    with torch.no_grad():
        x = torch.as_tensor(x_np, dtype=torch.float32)
        y = torch.as_tensor(y_np, dtype=torch.long)
        mu, _logvar = runtime.model.encode(x, y)
    mu_np = mu.detach().cpu().numpy()

    largest_out: list[AdaptiveSourceLocalSummary] = []
    bic_out: list[AdaptiveSourceLocalSummary] = []
    for cls in (0, 1):
        positions = np.flatnonzero(fit_labels == int(cls))
        cls_mu = mu_np[positions]
        candidates = _fit_candidate_summaries(
            cfg,
            cls_mu,
            experiment_seed=experiment_seed,
            source_center=runtime.expert_id,
            class_label=int(cls),
            shuffled_label_control=shuffled_label_control,
        )
        valid = [candidate for candidate in candidates if candidate["status"] == "ok"]
        largest = valid[0] if valid else None
        bic = min(valid, key=lambda item: (float(item["bic"]), -int(item["k"]))) if valid else None
        status_json = json.dumps(_candidate_status_payload(candidates), sort_keys=True)
        largest_out.append(
            _build_summary(
                cfg,
                root,
                runtime,
                experiment_seed=experiment_seed,
                class_label=int(cls),
                positions=positions,
                candidate=largest,
                selection_rule="largest_viable",
                selected_k_reason="largest_source_only_viable_k" if largest else "no_viable_source_local_k",
                candidate_fit_status_json=status_json,
                shuffled_label_control=shuffled_label_control,
            )
        )
        bic_out.append(
            _build_summary(
                cfg,
                root,
                runtime,
                experiment_seed=experiment_seed,
                class_label=int(cls),
                positions=positions,
                candidate=bic,
                selection_rule="bic_selected",
                selected_k_reason="lowest_source_local_bic" if bic else "no_viable_source_local_k",
                candidate_fit_status_json=status_json,
                shuffled_label_control=shuffled_label_control,
            )
        )
    return tuple(largest_out), tuple(bic_out)


def _fit_candidate_summaries(
    cfg: DecentralizedAdaptiveGmmConfig,
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
    out: list[dict[str, object]] = []
    for k in cfg.candidate_components_per_source_class:
        errors: list[str] = []
        min_count = int(k) * int(cfg.min_samples_per_component)
        if n < min_count:
            errors.append(f"source_class_count<{min_count}")
        weights = np.asarray([], dtype=float)
        means = np.empty((0, x.shape[1] if x.ndim == 2 else 0), dtype=float)
        diag_vars = np.empty_like(means)
        converged = False
        n_iter = 0
        score = math.nan
        bic = math.nan
        if not errors:
            if int(k) == 1:
                weights, means, diag_vars, score, bic = _empirical_k1_params(x, variance_floor=cfg.variance_floor)
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
                        "adaptive_local_gmm",
                        shuffled_label_control,
                    ),
                )
                gmm.fit(x)
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
        if not errors and effective != int(k):
            errors.append(f"effective_component_count!={int(k)}")
        if not errors and (not math.isfinite(min_weight) or min_weight < cfg.min_component_weight):
            errors.append(f"min_component_weight<{cfg.min_component_weight}")
        if not errors and not finite:
            errors.append("nonfinite_summary_parameter")
        if not errors and (not math.isfinite(min_diag_var) or min_diag_var < cfg.variance_floor):
            errors.append(f"diag_var<{cfg.variance_floor}")
        out.append(
            {
                "k": int(k),
                "status": "ok" if not errors else "ineligible_component_fit",
                "error_message": "|".join(errors),
                "weights": weights,
                "means": means,
                "diag_vars": diag_vars,
                "effective_component_count": effective,
                "min_component_weight": min_weight,
                "min_diag_var": min_diag_var,
                "component_entropy": d1._entropy(weights),
                "all_finite": finite,
                "gmm_converged": converged,
                "gmm_n_iter": n_iter,
                "source_train_log_likelihood": score,
                "bic": bic,
            }
        )
    return out


def _empirical_k1_params(x: object, *, variance_floor: float) -> tuple[object, object, object, float, float]:
    values = np.asarray(x, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        return np.asarray([], dtype=float), np.empty((0, 0), dtype=float), np.empty((0, 0), dtype=float), math.nan, math.nan
    mean = values.mean(axis=0, keepdims=True)
    var = np.maximum(values.var(axis=0, ddof=0, keepdims=True), float(variance_floor))
    diff = values - mean
    log_probs = -0.5 * (
        np.sum(np.log(2.0 * math.pi * var), axis=1)
        + np.sum((diff * diff) / var, axis=1)
    )
    total_log_likelihood = float(np.sum(log_probs))
    score = total_log_likelihood / float(values.shape[0])
    n_params = 2 * int(values.shape[1])
    bic = -2.0 * total_log_likelihood + float(n_params) * math.log(float(values.shape[0]))
    return np.asarray([1.0], dtype=float), mean, var, score, bic


def _build_summary(
    cfg: DecentralizedAdaptiveGmmConfig,
    root: Path,
    runtime: VariantRuntime,
    *,
    experiment_seed: int,
    class_label: int,
    positions: object,
    candidate: Mapping[str, object] | None,
    selection_rule: str,
    selected_k_reason: str,
    candidate_fit_status_json: str,
    shuffled_label_control: bool,
) -> AdaptiveSourceLocalSummary:
    weights = np.asarray(candidate["weights"], dtype=float) if candidate else np.asarray([], dtype=float)
    means = np.asarray(candidate["means"], dtype=float) if candidate else np.empty((0, int(runtime.model.latent_dim)), dtype=float)
    diag_vars = np.asarray(candidate["diag_vars"], dtype=float) if candidate else np.empty((0, int(runtime.model.latent_dim)), dtype=float)
    selected_k = int(candidate["k"]) if candidate else 0
    status = "ok" if candidate else "ineligible_component_fit"
    error_message = "" if candidate else "no_viable_source_local_k"
    clipped_vars = np.maximum(diag_vars, cfg.variance_floor) if diag_vars.size else diag_vars
    parameter_hash = d1._hash_array(d1._flatten_payload([weights, means, clipped_vars])) if weights.size else ""
    summary_path = _summary_path(
        root,
        experiment_seed,
        runtime.expert_id,
        class_label,
        selection_rule=selection_rule,
        shuffled_label_control=shuffled_label_control,
    )
    summary_hash = ""
    if weights.size:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            summary_path,
            weights=weights,
            means=means,
            diag_vars=clipped_vars,
            source_class_count=np.asarray([int(np.asarray(positions).size)], dtype=int),
            selected_k=np.asarray([selected_k], dtype=int),
            selection_rule=np.asarray([selection_rule]),
            candidate_components=np.asarray(cfg.candidate_components_per_source_class, dtype=int),
            candidate_fit_status_json=np.asarray([candidate_fit_status_json]),
            schema_version=np.asarray([SUMMARY_SCHEMA_VERSION]),
            expert_config_hash=np.asarray([d1._expert_config_hash(runtime)]),
        )
        summary_hash = d1._file_sha256(summary_path)
        seedless = _seedless_summary_path(
            root,
            runtime.expert_id,
            class_label,
            selection_rule=selection_rule,
            shuffled_label_control=shuffled_label_control,
        )
        seedless.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            seedless,
            weights=weights,
            means=means,
            diag_vars=clipped_vars,
            source_class_count=np.asarray([int(np.asarray(positions).size)], dtype=int),
            selected_k=np.asarray([selected_k], dtype=int),
            selection_rule=np.asarray([selection_rule]),
            candidate_components=np.asarray(cfg.candidate_components_per_source_class, dtype=int),
            candidate_fit_status_json=np.asarray([candidate_fit_status_json]),
            schema_version=np.asarray([SUMMARY_SCHEMA_VERSION]),
            expert_config_hash=np.asarray([d1._expert_config_hash(runtime)]),
        )
    return AdaptiveSourceLocalSummary(
        experiment_seed=int(experiment_seed),
        source_center=runtime.expert_id,
        class_label=int(class_label),
        selection_rule=str(selection_rule),
        selected_k=selected_k,
        selected_k_reason=str(selected_k_reason),
        candidate_fit_status_json=str(candidate_fit_status_json),
        weights=weights,
        means=means,
        diag_vars=clipped_vars,
        source_class_count=int(np.asarray(positions).size),
        effective_component_count=int(candidate["effective_component_count"]) if candidate else 0,
        min_component_weight=float(candidate["min_component_weight"]) if candidate else math.nan,
        min_diag_var=float(candidate["min_diag_var"]) if candidate else math.nan,
        component_entropy=float(candidate["component_entropy"]) if candidate else math.nan,
        all_finite=bool(candidate["all_finite"]) if candidate else False,
        gmm_converged=bool(candidate["gmm_converged"]) if candidate else False,
        gmm_n_iter=int(candidate["gmm_n_iter"]) if candidate else 0,
        source_train_log_likelihood=float(candidate["source_train_log_likelihood"]) if candidate else math.nan,
        source_inner_bic=float(candidate["bic"]) if candidate else math.nan,
        summary_path=summary_path,
        summary_hash=summary_hash,
        fit_row_ids_hash=d1._hash_strings([runtime.source_train_sample_ids[int(pos)] for pos in np.asarray(positions)]),
        parameter_hash=parameter_hash,
        expert_config_hash=d1._expert_config_hash(runtime),
        status=status,
        error_message=error_message,
        shuffled_label_control=bool(shuffled_label_control),
    )


def _candidate_status_payload(candidates: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "k": int(candidate["k"]),
            "status": candidate["status"],
            "error_message": candidate["error_message"],
            "bic": candidate["bic"],
            "effective_component_count": candidate["effective_component_count"],
            "min_component_weight": candidate["min_component_weight"],
            "min_diag_var": candidate["min_diag_var"],
        }
        for candidate in candidates
    ]


def _summary_path(
    root: Path,
    seed: int,
    source_center: str,
    class_label: int,
    *,
    selection_rule: str,
    shuffled_label_control: bool,
) -> Path:
    suffix = "_shuffled_label_control" if shuffled_label_control else ""
    return (
        root
        / "summaries"
        / f"seed_{int(seed)}"
        / f"source_{source_center}"
        / f"class_{int(class_label)}_adaptive_{selection_rule}{suffix}_summary.npz"
    )


def _seedless_summary_path(root: Path, source_center: str, class_label: int, *, selection_rule: str, shuffled_label_control: bool) -> Path:
    suffix = "_shuffled_label_control" if shuffled_label_control else ""
    return root / "summaries" / f"source_{source_center}" / f"class_{int(class_label)}_adaptive_{selection_rule}{suffix}_summary.npz"


def _summary_manifest_row(summary: AdaptiveSourceLocalSummary) -> dict[str, object]:
    return {
        "experiment_seed": int(summary.experiment_seed),
        "source_center": summary.source_center,
        "class_label": int(summary.class_label),
        "selection_rule": summary.selection_rule,
        "selected_k": int(summary.selected_k),
        "selected_k_reason": summary.selected_k_reason,
        "attempted_k_json": json.dumps([4, 3, 2, 1]),
        "candidate_fit_status_json": summary.candidate_fit_status_json,
        "component_id": "|".join(str(idx) for idx in range(_component_count(summary))),
        "component_weight_local": json.dumps(d1._as_float_list(summary.weights)),
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


def _summary_diagnostic_row(cfg: DecentralizedAdaptiveGmmConfig, summary: AdaptiveSourceLocalSummary) -> dict[str, object]:
    return {
        "experiment_seed": int(summary.experiment_seed),
        "source_center": summary.source_center,
        "class_label": int(summary.class_label),
        "selection_rule": summary.selection_rule,
        "selected_k": int(summary.selected_k),
        "selected_k_reason": summary.selected_k_reason,
        "attempted_k_json": json.dumps(list(cfg.candidate_components_per_source_class)),
        "candidate_fit_status_json": summary.candidate_fit_status_json,
        "source_class_count": int(summary.source_class_count),
        "effective_component_count": int(summary.effective_component_count),
        "min_component_weight": summary.min_component_weight,
        "min_required_component_weight": cfg.min_component_weight,
        "min_samples_per_component": cfg.min_samples_per_component,
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
    cfg: DecentralizedAdaptiveGmmConfig,
    *,
    experiment_seed: int,
    heldout_center: str,
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], AdaptiveSourceLocalSummary],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    source_mass = 1.0 / float(len(candidates))
    class_counts = _composed_counts_by_class(summaries, candidates, control_mode="normal")
    for cls in (0, 1):
        composed_id = 0
        for source_center in candidates:
            summary = summaries.get((str(source_center), int(cls)))
            if summary is None or summary.status != "ok":
                rows.append(
                    _invalid_composition_row(
                        experiment_seed=experiment_seed,
                        heldout_center=heldout_center,
                        candidates=candidates,
                        class_label=cls,
                        composed_component_id=composed_id,
                        source_center=str(source_center),
                        summary_status="missing_summary" if summary is None else summary.status,
                        summary_error_message=(
                            f"missing_summary_source_{source_center}_class_{cls}"
                            if summary is None
                            else summary.error_message
                        ),
                    )
                )
                continue
            weights = d1._normalized_weights(summary.weights)
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
                        "selection_rule": summary.selection_rule,
                        "selected_k": int(summary.selected_k),
                        "component_count_after_composition": int(class_counts.get(int(cls), 0)),
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
        "selection_rule": "",
        "selected_k": "",
        "component_count_after_composition": "",
        "component_weight_local": "",
        "component_weight_after_equal_source_normalization": "",
        "summary_hash": "",
        "summary_status": str(summary_status),
        "summary_error_message": str(summary_error_message),
    }


def _intervention_rows(
    *,
    experiment_seed: int,
    heldout_center: str,
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], AdaptiveSourceLocalSummary],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    class_counts = _composed_counts_by_class(summaries, candidates, control_mode="normal")
    source_mass = 1.0 / float(len(candidates))
    for cls in (0, 1):
        for source_center in candidates:
            summary = summaries.get((str(source_center), int(cls)))
            rows.append(
                {
                    "experiment_seed": int(experiment_seed),
                    "heldout_center": str(heldout_center),
                    "class_label": int(cls),
                    "source_center": str(source_center),
                    "selected_k": "" if summary is None else int(summary.selected_k),
                    "source_class_count": "" if summary is None else int(summary.source_class_count),
                    "selected_k_reason": "missing_summary" if summary is None else summary.selected_k_reason,
                    "component_count_after_composition": int(class_counts.get(int(cls), 0)),
                    "sample_mass_assigned": float(source_mass),
                }
            )
    return rows


def _evaluate_primary_and_references(
    cfg: DecentralizedAdaptiveGmmConfig,
    *,
    per_source_runtime: Mapping[str, RuntimeSource],
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], AdaptiveSourceLocalSummary],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    eval_raw: object,
    eval_labels: Sequence[int],
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
    real_feature_bacc: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    status, error = _composition_status(candidates, summaries, control_mode="normal")
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
        prior_method=PRIMARY_ADAPTIVE_METHOD,
        control_mode="normal",
    )
    pooled_geom = geometric_probability_pool(bundles)
    pooled_arith = d1._arithmetic_probability_pool(bundles)
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
            summaries=summaries,
            prior_method=PRIMARY_ADAPTIVE_METHOD,
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
            summaries=summaries,
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
            summaries=summaries,
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
            summaries=summaries,
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


def _evaluate_bic_diagnostic(
    cfg: DecentralizedAdaptiveGmmConfig,
    *,
    per_source_runtime: Mapping[str, RuntimeSource],
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], AdaptiveSourceLocalSummary],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    eval_raw: object,
    eval_labels: Sequence[int],
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
    real_feature_bacc: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    status, error = _composition_status(candidates, summaries, control_mode="normal")
    if status != "ok":
        row = _dense_empty_row(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            candidates=candidates,
            summaries=summaries,
            prior_method=ROW_BIC,
            source_union_ref=source_union_ref,
            center_balanced_ref=center_balanced_ref,
            real_feature_bacc=real_feature_bacc,
            status=status,
            error_message=error,
            claim_role="diagnostic_bic_selected_source_local_k",
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
        prior_method=ROW_BIC,
        control_mode="normal",
    )
    pooled = geometric_probability_pool(bundles)
    single_baccs = [_float(row["bacc"]) for row in single_rows if row.get("status") == "ok"]
    row = _dense_result_row(
        cfg,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        candidates=candidates,
        summaries=summaries,
        prior_method=ROW_BIC,
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
        claim_role="diagnostic_bic_selected_source_local_k",
    )
    return [row], single_rows, coverage_rows, weak_rows, nn_rows


def _evaluate_control(
    cfg: DecentralizedAdaptiveGmmConfig,
    *,
    per_source_runtime: Mapping[str, RuntimeSource],
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], AdaptiveSourceLocalSummary],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    eval_raw: object,
    eval_labels: Sequence[int],
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
    real_feature_bacc: float,
    prior_method: str,
    control_mode: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    status, error = _composition_status(candidates, summaries, control_mode=control_mode)
    if status != "ok":
        row = _dense_empty_row(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            candidates=candidates,
            summaries=summaries,
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
        summaries=summaries,
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
    cfg: DecentralizedAdaptiveGmmConfig,
    *,
    per_source_runtime: Mapping[str, RuntimeSource],
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], AdaptiveSourceLocalSummary],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    eval_raw: object,
    eval_labels: Sequence[int],
    prior_method: str,
    control_mode: str,
) -> tuple[list[PredictionBundle], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], str]:
    budgets = d1._balanced_counts(cfg.synthetic_per_class_total, len(candidates))
    bundles: list[PredictionBundle] = []
    late_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    weak_rows: list[dict[str, object]] = []
    nn_rows: list[dict[str, object]] = []
    generated_hashes: list[str] = []
    for source_center, budget_per_class in zip(candidates, budgets):
        runtime = per_source_runtime[str(source_center)].runtime
        latent_seed = d1._latent_seed(experiment_seed, heldout_center, replicate_seed, prior_method, source_center, control_mode)
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
        row = _base_matrix_row(
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
                "synthetic_per_class_total": int(budget_per_class),
                "synthetic_per_class_per_source_json": json.dumps({str(source_center): int(budget_per_class)}, sort_keys=True),
                "bacc": result.bacc,
                "macro_f1": result.macro_f1,
                "generated_features_hash": generated_hash,
                "prediction_hash": prediction_hash,
                "selection_source": DIAGNOSTIC_SELECTION,
                "status": "ok",
                "claim_role": "single_source_component_for_dense_aggregation",
            }
        )
        late_rows.append(row)
        if _float(row["bacc"]) < 0.75:
            weak_rows.append(_weak_row(row))
        coverage_rows.append(_coverage_row(row, counts, candidates=candidates, summaries=summaries, control_mode=control_mode))
        nn_rows.append(_nearest_neighbor_row(row, generated, runtime.source_train_embeddings))
        bundles.append(bundle)
    aggregate_hash = _hash_strings(generated_hashes)
    return bundles, late_rows, coverage_rows, weak_rows, nn_rows, aggregate_hash


def _sample_source_from_summaries(
    cfg: DecentralizedAdaptiveGmmConfig,
    runtime: VariantRuntime,
    summaries: Mapping[tuple[str, int], AdaptiveSourceLocalSummary],
    *,
    source_center: str,
    budget_per_class: int,
    seed: int,
    control_mode: str,
) -> tuple[object, tuple[int, ...], dict[int, dict[str, int]]]:
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
    summary: AdaptiveSourceLocalSummary,
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


def _real_feature_reference(
    cfg: DecentralizedAdaptiveGmmConfig,
    *,
    per_source_runtime: Mapping[str, RuntimeSource],
    candidates: Sequence[str],
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    eval_raw: object,
    eval_labels: Sequence[int],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    bundles: list[PredictionBundle] = []
    late_rows: list[dict[str, object]] = []
    empty_summaries: dict[tuple[str, int], AdaptiveSourceLocalSummary] = {}
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
        row = _base_matrix_row(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            candidates=candidates,
            summaries=empty_summaries,
            expert_id=str(source_center),
            expert_pool_type=POOL_PER_SOURCE,
            prior_method=ROW_REAL_FEATURE_DENSE_REFERENCE,
            pooling_rule="single_source_real_feature",
            source_union_ref=d1._missing_reference(),
            center_balanced_ref=d1._missing_reference(),
        )
        row.update(
            {
                "gmm_components": 0,
                "effective_gmm_components": 0,
                "max_local_gmm_components_per_source_class": 0,
                "composed_components_per_class_actual": "{}",
                "source_weighting": "not_applicable",
                "synthetic_per_class_total": 0,
                "synthetic_per_class_per_source_json": "{}",
                "bacc": result.bacc,
                "macro_f1": result.macro_f1,
                "prediction_hash": _hash_array(bundle.probabilities),
                "selection_source": DIAGNOSTIC_SELECTION,
                "status": "ok",
                "claim_role": "real_feature_single_source_reference",
            }
        )
        late_rows.append(row)
        bundles.append(bundle)
    pooled = geometric_probability_pool(bundles)
    result = evaluate_probability_predictions(ROW_REAL_FEATURE_DENSE_REFERENCE, pooled, eval_labels)
    row = _base_matrix_row(
        cfg,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        candidates=candidates,
        summaries=empty_summaries,
        expert_id="dense_all_sources",
        expert_pool_type=POOL_DECENTRALIZED,
        prior_method=ROW_REAL_FEATURE_DENSE_REFERENCE,
        pooling_rule="geometric",
        source_union_ref=d1._missing_reference(),
        center_balanced_ref=d1._missing_reference(),
    )
    row.update(
        {
            "gmm_components": 0,
            "effective_gmm_components": 0,
            "max_local_gmm_components_per_source_class": 0,
            "composed_components_per_class_actual": "{}",
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
    return row, late_rows


def _dense_result_row(
    cfg: DecentralizedAdaptiveGmmConfig,
    *,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], AdaptiveSourceLocalSummary],
    prior_method: str,
    pooling_rule: str,
    probabilities: Sequence[Sequence[float]],
    eval_labels: Sequence[int],
    generated_features_hash: str,
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
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
        summaries=summaries,
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
            "mean_single_source_adaptive_k_bacc": mean_single_bacc,
            "oracle_single_source_adaptive_k_bacc": oracle_single_bacc,
            "delta_vs_mean_single_source_adaptive_k": result.bacc - mean_single_bacc if math.isfinite(mean_single_bacc) else math.nan,
            "delta_vs_single_source_oracle_adaptive_k": result.bacc - oracle_single_bacc if math.isfinite(oracle_single_bacc) else math.nan,
            "retention_vs_source_union_k16": d1._retention(result.bacc, source_union_ref.bacc),
            "retention_vs_center_balanced_k16": d1._retention(result.bacc, center_balanced_ref.bacc),
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
    cfg: DecentralizedAdaptiveGmmConfig,
    *,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], AdaptiveSourceLocalSummary],
    prior_method: str,
    bacc: float,
    macro_f1: float,
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
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
        summaries=summaries,
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
            "mean_single_source_adaptive_k_bacc": mean_single_bacc,
            "oracle_single_source_adaptive_k_bacc": oracle_single_bacc,
            "delta_vs_mean_single_source_adaptive_k": bacc - mean_single_bacc if math.isfinite(bacc) and math.isfinite(mean_single_bacc) else math.nan,
            "delta_vs_single_source_oracle_adaptive_k": bacc - oracle_single_bacc if math.isfinite(bacc) and math.isfinite(oracle_single_bacc) else math.nan,
            "retention_vs_source_union_k16": d1._retention(bacc, source_union_ref.bacc),
            "retention_vs_center_balanced_k16": d1._retention(bacc, center_balanced_ref.bacc),
            "delta_vs_real_source_embedding_dense_reference": bacc - real_feature_bacc if math.isfinite(bacc) and math.isfinite(real_feature_bacc) else math.nan,
            "selection_source": DIAGNOSTIC_SELECTION,
            "status": "ok",
            "claim_role": claim_role,
        }
    )
    return row


def _base_matrix_row(
    cfg: DecentralizedAdaptiveGmmConfig,
    *,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], AdaptiveSourceLocalSummary],
    expert_id: str,
    expert_pool_type: str,
    prior_method: str,
    pooling_rule: str,
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
    control_mode: str = "normal",
) -> dict[str, object]:
    stats = _composition_stats(cfg, summaries, candidates, control_mode=control_mode)
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "expert_id": str(expert_id),
        "expert_pool_type": str(expert_pool_type),
        "variant_id": PRIMARY_VARIANT,
        "prior_method": prior_method,
        "gmm_components": stats["min_composed_components_per_class"],
        "effective_gmm_components": stats["min_composed_components_per_class"],
        "max_local_gmm_components_per_source_class": cfg.max_local_gmm_components_per_source_class,
        "composed_components_per_class_actual": stats["composed_components_per_class_actual"],
        "source_weighting": cfg.source_weighting,
        "pooling_rule": pooling_rule,
        "replicate_seed": int(replicate_seed),
        "latent_sample_seed": d1._latent_seed(experiment_seed, heldout_center, replicate_seed, prior_method),
        "included_source_centers": "|".join(str(v) for v in candidates),
        "num_included_sources": len(candidates),
        "synthetic_per_class_total": cfg.synthetic_per_class_total,
        "synthetic_per_class_per_source_json": _budget_json(cfg, candidates),
        "bacc": "",
        "macro_f1": "",
        "source_union_k16_bacc": source_union_ref.bacc,
        "center_balanced_k16_bacc": center_balanced_ref.bacc,
        "real_feature_dense_bacc": math.nan,
        "mean_single_source_adaptive_k_bacc": math.nan,
        "oracle_single_source_adaptive_k_bacc": math.nan,
        "delta_vs_mean_single_source_adaptive_k": math.nan,
        "delta_vs_single_source_oracle_adaptive_k": math.nan,
        "retention_vs_source_union_k16": math.nan,
        "retention_vs_center_balanced_k16": math.nan,
        "delta_vs_real_source_embedding_dense_reference": math.nan,
        "negative_control_gap": math.nan,
        "selected_k_histogram_json": stats["selected_k_histogram_json"],
        "min_selected_k": stats["min_selected_k"],
        "mean_selected_k": stats["mean_selected_k"],
        "pct_source_class_summaries_not_k4": stats["pct_source_class_summaries_not_k4"],
        "adaptive_k_intervention_active": stats["adaptive_k_intervention_active"],
        "generated_features_hash": "",
        "prediction_hash": "",
        "composed_prior_hash": _summary_set_hash(summaries, candidates, control_mode=control_mode),
        "summary_set_hash": _summary_set_hash(summaries, candidates, control_mode=control_mode),
        "selection_source": DIAGNOSTIC_SELECTION,
        "status": "",
        "error_message": "",
        "claim_role": "",
    }


def _dense_empty_row(
    cfg: DecentralizedAdaptiveGmmConfig,
    *,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], AdaptiveSourceLocalSummary],
    prior_method: str,
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
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
        summaries=summaries,
        expert_id="dense_all_sources",
        expert_pool_type=POOL_DECENTRALIZED,
        prior_method=prior_method,
        pooling_rule="geometric",
        source_union_ref=source_union_ref,
        center_balanced_ref=center_balanced_ref,
    )
    row.update({"real_feature_dense_bacc": real_feature_bacc, "status": status, "error_message": error_message, "claim_role": claim_role})
    return row


def _composition_ineligible_rows(
    cfg: DecentralizedAdaptiveGmmConfig,
    *,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    source_union_ref: d1.ReferenceValue,
    center_balanced_ref: d1.ReferenceValue,
    real_feature_bacc: float,
    status: str,
    error_message: str,
) -> list[dict[str, object]]:
    empty: dict[tuple[str, int], AdaptiveSourceLocalSummary] = {}
    return [
        _dense_empty_row(
            cfg,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            replicate_seed=replicate_seed,
            candidates=candidates,
            summaries=empty,
            prior_method=method,
            source_union_ref=source_union_ref,
            center_balanced_ref=center_balanced_ref,
            real_feature_bacc=real_feature_bacc,
            status=status,
            error_message=error_message,
            claim_role=role,
        )
        for method, role in (
            (PRIMARY_ADAPTIVE_METHOD, "primary_preservation_test"),
            (ROW_ARITH, "diagnostic_pooling_rule"),
            (ROW_SINGLE_MEAN, "single_source_mean_reference"),
            (ROW_SINGLE_ORACLE, "diagnostic_only_oracle_reference"),
        )
    ]


def _ineligible_rows(
    cfg: DecentralizedAdaptiveGmmConfig,
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
    empty: dict[tuple[str, int], AdaptiveSourceLocalSummary] = {}
    rows = [
        _dense_empty_row(
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
        for method, role in (
            (PRIMARY_ADAPTIVE_METHOD, "primary_preservation_test"),
            (ROW_ARITH, "diagnostic_pooling_rule"),
            (ROW_BIC, "diagnostic_bic_selected_source_local_k"),
            (ROW_SINGLE_MEAN, "single_source_mean_reference"),
            (ROW_SINGLE_ORACLE, "diagnostic_only_oracle_reference"),
            (ROW_SHUFFLED_SUMMARY_CONTROL, "negative_control"),
            (ROW_SHUFFLED_LABEL_CONTROL, "negative_control"),
        )
    ]
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


def _reference_matrix_row(
    cfg: DecentralizedAdaptiveGmmConfig,
    *,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    candidates: Sequence[str],
    prior_method: str,
    reference: d1.ReferenceValue,
) -> dict[str, object]:
    empty: dict[tuple[str, int], AdaptiveSourceLocalSummary] = {}
    row = _base_matrix_row(
        cfg,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        replicate_seed=replicate_seed,
        candidates=candidates,
        summaries=empty,
        expert_id=POOL_SOURCE_UNION,
        expert_pool_type=POOL_SOURCE_UNION,
        prior_method=prior_method,
        pooling_rule="reference",
        source_union_ref=reference if prior_method == ROW_SOURCE_UNION_K16_REFERENCE else d1._missing_reference(),
        center_balanced_ref=reference if prior_method == ROW_CENTER_BALANCED_K16_REFERENCE else d1._missing_reference(),
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
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], AdaptiveSourceLocalSummary],
    *,
    control_mode: str,
) -> tuple[str, str]:
    errors: list[str] = []
    for source_center in candidates:
        for cls in (0, 1):
            summary_cls = 1 - cls if control_mode == "class_flip" else cls
            summary = summaries.get((str(source_center), int(summary_cls)))
            if summary is None:
                errors.append(f"missing_summary_source_{source_center}_class_{summary_cls}")
                continue
            if summary.status != "ok":
                errors.append(f"source_{source_center}_class_{summary_cls}:{summary.error_message or summary.status}")
            if int(summary.selected_k) < 1:
                errors.append(f"source_{source_center}_class_{summary_cls}_selected_k<1")
    if errors:
        return "ineligible_component_fit", "|".join(sorted(set(errors)))
    return "ok", ""


def _coverage_row(
    row: Mapping[str, object],
    component_counts: Mapping[int, Mapping[str, int]],
    *,
    candidates: Sequence[str],
    summaries: Mapping[tuple[str, int], AdaptiveSourceLocalSummary],
    control_mode: str,
) -> dict[str, object]:
    expected = set()
    for cls in (0, 1):
        summary_cls = 1 - cls if control_mode == "class_flip" else cls
        for source in candidates:
            summary = summaries.get((str(source), int(summary_cls)))
            if summary is None:
                continue
            for component in range(int(summary.selected_k)):
                expected.add(f"{cls}:{source}:{component}")
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
        if row.get("prior_method") != PRIMARY_ADAPTIVE_METHOD:
            continue
        key = (str(row.get("experiment_seed")), str(row.get("heldout_center")), str(row.get("replicate_seed")))
        control = controls.get(key, math.nan)
        value = _float(row.get("bacc"))
        if math.isfinite(value) and math.isfinite(control):
            row["negative_control_gap"] = value - control


def _decision(
    rows: Sequence[Mapping[str, object]],
    cfg: DecentralizedAdaptiveGmmConfig,
    *,
    leakage_status: str,
    diagnostic_rows: Sequence[Mapping[str, object]],
    strict_d1: Mapping[str, object],
) -> dict[str, object]:
    primary_all = _rows_for(rows, PRIMARY_ADAPTIVE_METHOD, include_non_ok=True)
    primary = _rows_for(rows, PRIMARY_ADAPTIVE_METHOD)
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
    intervention = _adaptive_intervention_stats(diagnostic_rows, cfg)
    fit_ineligible = any(row.get("status") == "ineligible_component_fit" for row in primary_all)
    negative_control_competitive = (
        math.isfinite(_float(control_stats["center_equal_mean_bacc"]))
        and math.isfinite(_float(stats["center_equal_mean_bacc"]))
        and _float(stats["center_equal_mean_bacc"]) - _float(control_stats["center_equal_mean_bacc"]) <= 0.05
    )
    retention_source_union = d1._retention(_float(stats["center_equal_mean_bacc"]), _float(source_union_stats["center_equal_mean_bacc"]))
    retention_center_balanced = d1._retention(_float(stats["center_equal_mean_bacc"]), _float(center_balanced_stats["center_equal_mean_bacc"]))
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
        and int(stats["n_heldout_centers"]) >= len(cfg.heldout_centers)
        and delta_vs_mean_single > 0.0
    )
    negative = (
        leakage_status == "PASS"
        and not fit_ineligible
        and math.isfinite(delta_vs_oracle_single)
        and delta_vs_oracle_single <= 0.0
    )
    verdict = "D1_1_FAIL"
    if leakage_status != "PASS":
        verdict = "PROTOCOL_FAIL"
    elif fit_ineligible:
        verdict = "INELIGIBLE"
    elif primary_pass:
        verdict = "D1_1_PASS"
    elif partial:
        verdict = "D1_1_PARTIAL_EVIDENCE"
    elif negative:
        verdict = "D1_1_NEGATIVE_EVIDENCE"
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
        flags.append("DOES_NOT_BEAT_SINGLE_SOURCE_ORACLE_ADAPTIVE_K")

    return {
        "primary_verdict": verdict,
        "diagnostic_flags": "|".join(flags),
        "leakage_status": leakage_status,
        "primary_method": PRIMARY_ADAPTIVE_METHOD,
        "center_equal_mean_bacc": stats["center_equal_mean_bacc"],
        "center_equal_macro_f1": stats["center_equal_macro_f1"],
        "min_center_bacc": stats["min_center_mean_bacc"],
        "seed_std_bacc": stats["seed_std_bacc"],
        "delta_vs_mean_single_source_adaptive_k": delta_vs_mean_single,
        "delta_vs_single_source_oracle_adaptive_k": delta_vs_oracle_single,
        "retention_vs_source_union_k16": retention_source_union,
        "retention_vs_center_balanced_k16": retention_center_balanced,
        "delta_vs_real_source_embedding_dense_reference": delta_vs_real,
        "negative_control_gap": _float(stats["center_equal_mean_bacc"]) - _float(control_stats["center_equal_mean_bacc"]),
        "mean_single_source_adaptive_k_center_equal_mean_bacc": single_mean_stats["center_equal_mean_bacc"],
        "single_source_oracle_adaptive_k_center_equal_mean_bacc": single_oracle_stats["center_equal_mean_bacc"],
        "source_union_k16_reference_center_equal_mean_bacc": source_union_stats["center_equal_mean_bacc"],
        "center_balanced_k16_reference_center_equal_mean_bacc": center_balanced_stats["center_equal_mean_bacc"],
        "real_feature_dense_reference_center_equal_mean_bacc": real_stats["center_equal_mean_bacc"],
        "negative_control_center_equal_mean_bacc": control_stats["center_equal_mean_bacc"],
        "eligible_heldout_centers": stats["n_heldout_centers"],
        "eligible_seed_center_cells": stats["n_decision_cells"],
        "strict_d1_primary_verdict": strict_d1.get("primary_verdict", "missing_reference"),
        "strict_d1_center_equal_mean_bacc": strict_d1.get("center_equal_mean_bacc", math.nan),
        "strict_d1_diagnostic_flags": strict_d1.get("diagnostic_flags", "missing_reference"),
        **intervention,
        **stats,
    }


def _primary_stats(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
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
        "center_equal_mean_bacc": nanmean(seed_means) if seed_means else math.nan,
        "center_equal_macro_f1": d1._center_equal_mean(grouped, "macro_f1"),
        "seed_std_bacc": d1._std(seed_means),
        "min_center_mean_bacc": min(center_bacc.values()) if center_bacc else math.nan,
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


def _rows_for(rows: Sequence[Mapping[str, object]], method: str, *, include_non_ok: bool = False) -> list[Mapping[str, object]]:
    return [
        row for row in rows
        if row.get("prior_method") == method
        and (include_non_ok or row.get("status") == "ok")
    ]


def _adaptive_intervention_stats(rows: Sequence[Mapping[str, object]], cfg: DecentralizedAdaptiveGmmConfig) -> dict[str, object]:
    selected = [
        int(row["selected_k"])
        for row in rows
        if row.get("selection_rule") == "largest_viable"
        and str(row.get("shuffled_label_control")) == "False"
        and row.get("status") == "ok"
        and str(row.get("selected_k", "")) != ""
    ]
    hist = {str(k): selected.count(k) for k in cfg.candidate_components_per_source_class}
    pct_not_k4 = (
        sum(1 for value in selected if value != cfg.max_local_gmm_components_per_source_class) / float(len(selected))
        if selected else math.nan
    )
    return {
        "min_selected_k": min(selected) if selected else math.nan,
        "mean_selected_k": nanmean([float(value) for value in selected]) if selected else math.nan,
        "selected_k_histogram_json": json.dumps(hist, sort_keys=True),
        "pct_source_class_summaries_not_k4": pct_not_k4,
        "adaptive_k_intervention_active": bool(any(value != cfg.max_local_gmm_components_per_source_class for value in selected)),
    }


def _load_strict_d1_summary(root: Path | None) -> dict[str, object]:
    if root is None:
        return {}
    path = root / "tables" / "decentralized_k16_summary.csv"
    if not path.exists():
        return {}
    import csv

    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return dict(rows[0]) if rows else {}


def _write_artifacts(
    root: Path,
    cfg: DecentralizedAdaptiveGmmConfig,
    *,
    matrix_rows: Sequence[Mapping[str, object]],
    gap_rows: Sequence[Mapping[str, object]],
    summary_manifest_rows: Sequence[Mapping[str, object]],
    composition_rows: Sequence[Mapping[str, object]],
    diagnostic_rows: Sequence[Mapping[str, object]],
    intervention_rows: Sequence[Mapping[str, object]],
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
    write_csv_rows(root / "tables" / "decentralized_adaptive_downstream_matrix.csv", matrix_rows, columns=_matrix_columns())
    write_csv_rows(root / "tables" / "decentralized_adaptive_gap_summary.csv", gap_rows, columns=_matrix_columns())
    write_csv_rows(root / "tables" / "decentralized_adaptive_summary.csv", [dict(decision)])
    write_csv_rows(root / "tables" / "exported_source_summary_manifest.csv", summary_manifest_rows, columns=_summary_manifest_columns())
    write_csv_rows(root / "tables" / "composed_prior_component_manifest.csv", composition_rows, columns=_composition_columns())
    write_csv_rows(root / "tables" / "source_summary_diagnostics.csv", diagnostic_rows, columns=_diagnostic_columns())
    write_csv_rows(root / "tables" / "adaptive_k_intervention_audit.csv", intervention_rows, columns=_intervention_columns())
    write_csv_rows(root / "tables" / "late_aggregation_matrix.csv", late_rows, columns=_matrix_columns())
    write_csv_rows(root / "tables" / "real_feature_reference_matrix.csv", real_feature_rows, columns=_matrix_columns())
    write_csv_rows(root / "tables" / "generated_component_coverage_audit.csv", coverage_rows)
    write_csv_rows(root / "tables" / "weak_source_audit.csv", weak_rows)
    write_csv_rows(root / "tables" / "nearest_neighbor_memorization_audit.csv", nn_rows)
    write_csv_rows(root / "tables" / "negative_control_summary.csv", [_negative_control_summary(decision)])
    write_csv_rows(root / "manifests" / "decentralized_adaptive_prior_model_manifest.csv", model_manifest_rows)
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
            "schema_version": "cvae_rebuild_decentralized_adaptive_gmm_prior_protocol_manifest_v1",
            "experiment_name": cfg.name,
            "experiment_type": "adaptive_source_local_latent_summary_preservation_test",
            "primary_variant": cfg.primary_variant,
            "primary_method": cfg.primary_method,
            "bic_method": cfg.bic_method,
            "target_support_labels_for_selection": False,
            "target_eval_labels_for_scoring_only": True,
            "target_expert_excluded": target_expert_excluded,
            "exported_source_summaries_are_target_agnostic": True,
            "composition_manifests_are_fold_specific": True,
            "raw_source_embedding_pooling_for_prior_fit": False,
            "adaptive_k_selection_uses_source_local_fit_statistics_only": True,
            "source_union_references_diagnostic_only": True,
            "oracle_rows_diagnostic_only": True,
            "protocol_wording": PROTOCOL_WORDING,
            "claim_boundary": (
                "adaptive decentralized prior-composition preservation test only; no target-specific "
                "compatibility routing claim, no metadata-routing claim, no support-NELBO downstream claim, "
                "and no formal privacy claim"
            ),
        },
        resolved_config=_resolved_config(cfg),
    )
    _write_decision_summary(root, decision, leakage_status=leakage_status)


def _matrix_columns() -> tuple[str, ...]:
    return (
        "experiment_seed", "heldout_center", "expert_id", "expert_pool_type", "variant_id",
        "prior_method", "gmm_components", "effective_gmm_components",
        "max_local_gmm_components_per_source_class", "composed_components_per_class_actual",
        "source_weighting", "pooling_rule", "replicate_seed", "latent_sample_seed",
        "included_source_centers", "num_included_sources", "synthetic_per_class_total",
        "synthetic_per_class_per_source_json", "bacc", "macro_f1", "source_union_k16_bacc",
        "center_balanced_k16_bacc", "real_feature_dense_bacc",
        "mean_single_source_adaptive_k_bacc", "oracle_single_source_adaptive_k_bacc",
        "delta_vs_mean_single_source_adaptive_k", "delta_vs_single_source_oracle_adaptive_k",
        "retention_vs_source_union_k16", "retention_vs_center_balanced_k16",
        "delta_vs_real_source_embedding_dense_reference", "negative_control_gap",
        "selected_k_histogram_json", "min_selected_k", "mean_selected_k",
        "pct_source_class_summaries_not_k4", "adaptive_k_intervention_active",
        "generated_features_hash", "prediction_hash", "composed_prior_hash", "summary_set_hash",
        "selection_source", "status", "error_message", "claim_role",
    )


def _summary_manifest_columns() -> tuple[str, ...]:
    return (
        "experiment_seed", "source_center", "class_label", "selection_rule", "selected_k",
        "selected_k_reason", "attempted_k_json", "candidate_fit_status_json", "component_id",
        "component_weight_local", "latent_mean_hash", "latent_diag_var_hash", "source_class_count",
        "expert_config_hash", "summary_schema_version", "summary_path", "summary_hash", "status",
        "error_message",
    )


def _composition_columns() -> tuple[str, ...]:
    return (
        "experiment_seed", "heldout_center", "included_source_centers", "class_label",
        "composed_component_id", "source_center", "source_component_id", "selection_rule",
        "selected_k", "component_count_after_composition", "component_weight_local",
        "component_weight_after_equal_source_normalization", "summary_hash", "summary_status",
        "summary_error_message",
    )


def _diagnostic_columns() -> tuple[str, ...]:
    return (
        "experiment_seed", "source_center", "class_label", "selection_rule", "selected_k",
        "selected_k_reason", "attempted_k_json", "candidate_fit_status_json", "source_class_count",
        "effective_component_count", "min_component_weight", "min_required_component_weight",
        "min_samples_per_component", "variance_floor", "min_diag_var", "all_finite",
        "component_entropy", "gmm_converged", "gmm_n_iter", "source_train_log_likelihood",
        "source_inner_bic", "fit_row_ids_hash", "parameter_hash", "summary_path", "summary_hash",
        "shuffled_label_control", "status", "error_message",
    )


def _intervention_columns() -> tuple[str, ...]:
    return (
        "experiment_seed", "heldout_center", "class_label", "source_center", "selected_k",
        "source_class_count", "selected_k_reason", "component_count_after_composition",
        "sample_mass_assigned",
    )


def _negative_control_summary(decision: Mapping[str, object]) -> dict[str, object]:
    return {
        "primary_method": PRIMARY_ADAPTIVE_METHOD,
        "control_methods": f"{ROW_SHUFFLED_SUMMARY_CONTROL}|{ROW_SHUFFLED_LABEL_CONTROL}",
        "primary_center_equal_mean_bacc": decision.get("center_equal_mean_bacc", math.nan),
        "control_center_equal_mean_bacc": decision.get("negative_control_center_equal_mean_bacc", math.nan),
        "negative_control_gap": decision.get("negative_control_gap", math.nan),
        "control_competitive": "NEGATIVE_CONTROL_COMPETITIVE" in str(decision.get("diagnostic_flags", "")),
    }


def _write_decision_summary(root: Path, decision: Mapping[str, object], *, leakage_status: str) -> None:
    actual_k16 = (
        not bool(decision.get("adaptive_k_intervention_active"))
        and math.isfinite(_float(decision.get("min_selected_k")))
        and int(_float(decision.get("min_selected_k"))) == 4
    )
    text = "\n".join(
        [
            "# D1.1: Adaptive Source-Local Latent Summary Preservation Test",
            "",
            "## Summary",
            "",
            f"- Primary method: `{PRIMARY_ADAPTIVE_METHOD}`",
            f"- Primary verdict: `{decision.get('primary_verdict', 'D1_1_FAIL')}`",
            f"- Diagnostic flags: `{decision.get('diagnostic_flags', '')}`",
            f"- Center-equal mean BACC: {_format_float(decision.get('center_equal_mean_bacc'))}",
            f"- Center-equal macro-F1: {_format_float(decision.get('center_equal_macro_f1'))}",
            f"- Min center BACC: {_format_float(decision.get('min_center_bacc'))}",
            f"- Seed std BACC: {_format_float(decision.get('seed_std_bacc'))}",
            f"- Delta vs mean single-source adaptive K: {_format_float(decision.get('delta_vs_mean_single_source_adaptive_k'))}",
            f"- Retention vs source-union K16: {_format_float(decision.get('retention_vs_source_union_k16'))}",
            f"- Retention vs center-balanced K16: {_format_float(decision.get('retention_vs_center_balanced_k16'))}",
            f"- Delta vs real-feature dense reference: {_format_float(decision.get('delta_vs_real_source_embedding_dense_reference'))}",
            f"- Negative-control gap: {_format_float(decision.get('negative_control_gap'))}",
            f"- Adaptive-K intervention active: {'yes' if decision.get('adaptive_k_intervention_active') else 'no'}",
            f"- Percent source-class summaries not selecting K4: {_format_float(decision.get('pct_source_class_summaries_not_k4'))}",
            f"- Selected-K histogram: `{decision.get('selected_k_histogram_json', '{}')}`",
            f"- Interpretation: `{'actual_k16_replay' if actual_k16 else 'adaptive_k_composition'}`",
            f"- Leakage status: `{leakage_status}`",
            "",
            "## Protocol Boundary",
            "",
            PROTOCOL_WORDING,
            "",
            "This is decentralized prior composition plus dense output aggregation.",
            "It is not a target-specific compatibility-routing result.",
            "It does not prove metadata routing or support-NELBO routing.",
            "The source-union K16 rows are centralized diagnostic references only.",
            "",
            "## Supported Claim If PASS",
            "",
            "Source-local adaptive latent summaries can preserve most centralized Virchow2 K16 utility under a raw-data-free summary-exchange protocol with dense expert aggregation.",
            "",
        ]
    )
    path = root / "reports" / "decision_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _resolved_config(cfg: DecentralizedAdaptiveGmmConfig) -> dict[str, object]:
    return {
        "name": cfg.name,
        "artifact_root": str(cfg.artifact_root),
        "repair_artifact_root": str(cfg.repair_artifact_root),
        "strict_d1_artifact_root": "" if cfg.strict_d1_artifact_root is None else str(cfg.strict_d1_artifact_root),
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
        "bic_method": cfg.bic_method,
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
        "classifier": {
            "type": cfg.classifier_type,
            "solver": cfg.classifier_solver,
            "C": cfg.classifier_c,
            "max_iter": cfg.classifier_max_iter,
            "class_weight": cfg.classifier_class_weight,
            "classifier_seed": cfg.classifier_seed,
        },
    }


def _component_count(summary: AdaptiveSourceLocalSummary) -> int:
    return int(np.asarray(summary.weights).shape[0]) if np.asarray(summary.weights).size else 0


def _composition_stats(
    cfg: DecentralizedAdaptiveGmmConfig,
    summaries: Mapping[tuple[str, int], AdaptiveSourceLocalSummary],
    candidates: Sequence[str],
    *,
    control_mode: str,
) -> dict[str, object]:
    values = []
    class_counts = _composed_counts_by_class(summaries, candidates, control_mode=control_mode)
    for cls in (0, 1):
        summary_cls = 1 - cls if control_mode == "class_flip" else cls
        for source in candidates:
            summary = summaries.get((str(source), int(summary_cls)))
            if summary is not None and summary.status == "ok":
                values.append(int(summary.selected_k))
    hist = {str(k): values.count(k) for k in cfg.candidate_components_per_source_class}
    pct_not_k4 = (
        sum(1 for value in values if value != cfg.max_local_gmm_components_per_source_class) / float(len(values))
        if values else math.nan
    )
    return {
        "min_composed_components_per_class": min(class_counts.values()) if class_counts else 0,
        "composed_components_per_class_actual": json.dumps({str(k): int(v) for k, v in class_counts.items()}, sort_keys=True),
        "selected_k_histogram_json": json.dumps(hist, sort_keys=True),
        "min_selected_k": min(values) if values else math.nan,
        "mean_selected_k": nanmean([float(value) for value in values]) if values else math.nan,
        "pct_source_class_summaries_not_k4": pct_not_k4,
        "adaptive_k_intervention_active": bool(any(value != cfg.max_local_gmm_components_per_source_class for value in values)),
    }


def _composed_counts_by_class(
    summaries: Mapping[tuple[str, int], AdaptiveSourceLocalSummary],
    candidates: Sequence[str],
    *,
    control_mode: str,
) -> dict[int, int]:
    out: dict[int, int] = {}
    for cls in (0, 1):
        total = 0
        summary_cls = 1 - cls if control_mode == "class_flip" else cls
        for source in candidates:
            summary = summaries.get((str(source), int(summary_cls)))
            if summary is not None and summary.status == "ok":
                total += int(summary.selected_k)
        out[int(cls)] = total
    return out


def _summary_set_hash(
    summaries: Mapping[tuple[str, int], AdaptiveSourceLocalSummary],
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


def _budget_json(cfg: DecentralizedAdaptiveGmmConfig, candidates: Sequence[str]) -> str:
    budgets = d1._balanced_counts(cfg.synthetic_per_class_total, len(candidates))
    return json.dumps({str(source): int(budget) for source, budget in zip(candidates, budgets)}, sort_keys=True)
