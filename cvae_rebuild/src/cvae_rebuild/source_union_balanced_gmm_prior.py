from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
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
from .prior_calibration import _balanced_labels, _decode_latents
from .protocol import ProtocolError, assert_candidate_pool, build_leakage_report
from .reporting import prepare_artifact_dirs, write_csv_rows, write_json
from .source_union_gmm_prior import (
    ImportedReference,
    _ci95,
    _ci95_low,
    _empty_reference,
    _load_imported_references,
    _nearest_neighbor_row,
    _read_required_csv,
    _reference_for_runtime,
    _reference_key,
)
from .splits import candidate_experts


BALANCED_GMM_NAME = "virchow2_cvae_source_union_center_balanced_gmm_prior_v1"
PRIMARY_BALANCED_METHOD = "source_union_center_balanced_cc_diag_gmm_k16_prior_sample"
ROW_VANILLA_K16_REFERENCE = "source_union_cc_diag_gmm_k16_prior_sample_reference"
ROW_BALANCED_K8 = "source_union_center_balanced_cc_diag_gmm_k8_prior_sample_diagnostic"
ROW_BALANCED_K24 = "source_union_center_balanced_cc_diag_gmm_k24_prior_sample_diagnostic"
ROW_CENTER_STRATIFIED_K4X4 = "source_center_stratified_cc_diag_gmm_k4x4_prior_sample_diagnostic"
ROW_SHUFFLED_LABEL_CONTROL = "source_union_center_balanced_cc_diag_gmm_k16_shuffled_label_control_diagnostic"
ROW_PER_SOURCE_BALANCED_K16 = "per_source_center_balanced_cc_diag_gmm_k16_prior_sample_diagnostic"
SOURCE_UNION_GMM_ARTIFACT_NAME = "virchow2_cvae_source_union_gmm_prior_v1"


@dataclass(frozen=True)
class SourceUnionBalancedGmmConfig:
    name: str
    artifact_root: Path
    repair_artifact_root: Path
    sampling_artifact_root: Path
    prior_calibration_artifact_root: Path
    covariance_confirmation_artifact_root: Path
    source_union_gmm_artifact_root: Path
    feature_cache_root: Path
    experiment_seeds: tuple[int, ...]
    heldout_centers: tuple[str, ...]
    replicate_seeds: tuple[int, ...]
    synthetic_per_class_total: int
    primary_variant: str
    primary_method: str
    gmm_components: int
    gmm_covariance_type: str
    gmm_reg_covar: float
    gmm_n_init: int
    gmm_max_iter: int
    gmm_weight_floor: float
    min_source_center_class_count: int
    min_effective_gmm_components: int
    balanced_fit_samples_per_center_class: int
    max_center_class_replacement_rate: float
    mean_center_class_replacement_rate: float
    posterior_noise_scale: float
    diagnostic_gmm_components: tuple[int, ...]
    classifier_type: str
    classifier_solver: str
    classifier_c: float
    classifier_max_iter: int
    classifier_class_weight: str
    classifier_seed: int | None


@dataclass(frozen=True)
class VanillaK16Reference:
    bacc: float
    macro_f1: float
    clipped_preservation_gap: float
    preservation_ratio: float


@dataclass(frozen=True)
class GmmClassStats:
    class_label: int
    center: str
    class_train_count: int
    weights: object
    means: object
    covariances: object
    effective_gmm_components: int
    min_component_weight: float
    num_components_below_weight_floor: int
    num_components_covariance_clipped: int
    gmm_converged: bool
    gmm_n_iter: int
    source_train_log_likelihood: float
    source_inner_bic: float


@dataclass(frozen=True)
class BalancedGmmParameters:
    classes: dict[int, GmmClassStats]
    center_classes: dict[tuple[int, str], GmmClassStats]
    gmm_fit_row_ids_hash: str
    gmm_parameter_hash: str
    diagnostics_rows: tuple[dict[str, object], ...]
    balance_rows: tuple[dict[str, object], ...]
    status: str
    error_message: str
    gmm_components: int
    strategy: str
    shuffled_label_control: bool
    source_center_fit_counts_json: str
    source_center_sample_counts_json: str
    fit_sample_replacement_rate: float
    max_center_class_replacement_rate: float
    mean_center_class_replacement_rate: float


def load_source_union_balanced_gmm_prior_config(path: str | Path) -> SourceUnionBalancedGmmConfig:
    source = Path(path).resolve()
    data = _load_mapping(source)
    base_dir = source.parents[2] if len(source.parents) >= 3 else source.parent
    return parse_source_union_balanced_gmm_prior_config(data, base_dir=base_dir)


def parse_source_union_balanced_gmm_prior_config(
    data: Mapping[str, Any],
    *,
    base_dir: str | Path = ".",
) -> SourceUnionBalancedGmmConfig:
    base = Path(base_dir)
    experiment = _mapping(data, "experiment")
    inputs = _mapping(data, "inputs")
    run = _mapping(data, "run_matrix")
    generation = _mapping(data, "generation")
    gmm = _mapping(data, "balanced_gmm_prior")
    classifier = _mapping(data, "classifier")
    cfg = SourceUnionBalancedGmmConfig(
        name=str(experiment["name"]),
        artifact_root=_path(base, str(experiment["artifact_root"])),
        repair_artifact_root=_path(base, str(inputs["repair_artifact_root"])),
        sampling_artifact_root=_path(base, str(inputs["sampling_artifact_root"])),
        prior_calibration_artifact_root=_path(base, str(inputs["prior_calibration_artifact_root"])),
        covariance_confirmation_artifact_root=_path(base, str(inputs["covariance_confirmation_artifact_root"])),
        source_union_gmm_artifact_root=_path(base, str(inputs["source_union_gmm_artifact_root"])),
        feature_cache_root=_path(base, str(inputs["feature_cache_root"])),
        experiment_seeds=tuple(int(v) for v in run["experiment_seeds"]),
        heldout_centers=tuple(str(v) for v in run["heldout_centers"]),
        replicate_seeds=tuple(int(v) for v in run["replicate_seeds"]),
        synthetic_per_class_total=int(generation["synthetic_per_class_total"]),
        primary_variant=str(experiment["primary_variant"]),
        primary_method=str(gmm["primary_method"]),
        gmm_components=int(gmm["gmm_components"]),
        gmm_covariance_type=str(gmm["gmm_covariance_type"]),
        gmm_reg_covar=float(gmm["gmm_reg_covar"]),
        gmm_n_init=int(gmm["gmm_n_init"]),
        gmm_max_iter=int(gmm["gmm_max_iter"]),
        gmm_weight_floor=float(gmm["gmm_weight_floor"]),
        min_source_center_class_count=int(gmm["min_source_center_class_count"]),
        min_effective_gmm_components=int(gmm["min_effective_gmm_components"]),
        balanced_fit_samples_per_center_class=int(gmm["balanced_fit_samples_per_center_class"]),
        max_center_class_replacement_rate=float(gmm["max_center_class_replacement_rate"]),
        mean_center_class_replacement_rate=float(gmm["mean_center_class_replacement_rate"]),
        posterior_noise_scale=float(gmm["posterior_noise_scale"]),
        diagnostic_gmm_components=tuple(int(v) for v in gmm.get("diagnostic_gmm_components", [8, 24])),
        classifier_type=str(classifier["type"]),
        classifier_solver=str(classifier["solver"]),
        classifier_c=float(classifier["C"]),
        classifier_max_iter=int(classifier["max_iter"]),
        classifier_class_weight=str(classifier["class_weight"]),
        classifier_seed=None if classifier.get("classifier_seed") is None else int(classifier["classifier_seed"]),
    )
    validate_source_union_balanced_gmm_prior_config(cfg)
    return cfg


def validate_source_union_balanced_gmm_prior_config(cfg: SourceUnionBalancedGmmConfig) -> None:
    if cfg.name != BALANCED_GMM_NAME:
        raise ProtocolError(f"Balanced source-union GMM experiment name must be {BALANCED_GMM_NAME!r}.")
    if cfg.primary_variant != UNION_VARIANT:
        raise ProtocolError(f"primary_variant must be {UNION_VARIANT!r}.")
    if cfg.primary_method != PRIMARY_BALANCED_METHOD:
        raise ProtocolError(f"primary_method must be {PRIMARY_BALANCED_METHOD!r}.")
    if cfg.source_union_gmm_artifact_root.name != SOURCE_UNION_GMM_ARTIFACT_NAME:
        raise ProtocolError(f"source_union_gmm_artifact_root must point to {SOURCE_UNION_GMM_ARTIFACT_NAME!r}.")
    if cfg.gmm_components != 16:
        raise ProtocolError("gmm_components must be locked to 16 for the primary method.")
    if cfg.gmm_covariance_type != "diag":
        raise ProtocolError("gmm_covariance_type must be diag.")
    if not math.isclose(cfg.posterior_noise_scale, 0.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ProtocolError("posterior_noise_scale must be 0.0 for the primary method.")
    if cfg.synthetic_per_class_total != 128:
        raise ProtocolError("synthetic_per_class_total must be locked to 128.")
    if cfg.diagnostic_gmm_components != (8, 24):
        raise ProtocolError("diagnostic_gmm_components must be exactly [8, 24].")
    if cfg.gmm_reg_covar <= 0.0 or cfg.gmm_n_init < 1 or cfg.gmm_max_iter < 1:
        raise ProtocolError("GMM regularization, n_init, and max_iter must be positive.")
    if min(cfg.gmm_weight_floor, cfg.max_center_class_replacement_rate, cfg.mean_center_class_replacement_rate) < 0.0:
        raise ProtocolError("GMM thresholds must be non-negative.")
    if min(cfg.min_source_center_class_count, cfg.min_effective_gmm_components, cfg.balanced_fit_samples_per_center_class) < 1:
        raise ProtocolError("GMM adequacy and balance counts must be positive.")
    if cfg.classifier_type != "sklearn_logistic_regression":
        raise ProtocolError("classifier.type must be sklearn_logistic_regression.")
    if cfg.classifier_solver != "lbfgs" or cfg.classifier_c != 1.0 or cfg.classifier_max_iter != 2000:
        raise ProtocolError("Classifier solver/C/max_iter must remain locked.")
    if cfg.classifier_class_weight != "balanced" or cfg.classifier_seed is not None:
        raise ProtocolError("Classifier must use class_weight=balanced and classifier_seed=null.")


def run_source_union_balanced_gmm_prior(
    cfg: SourceUnionBalancedGmmConfig,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    root = prepare_artifact_dirs(Path(artifact_root) if artifact_root is not None else cfg.artifact_root)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    matrix_rows: list[dict[str, object]] = []
    diagnostics_rows: list[dict[str, object]] = []
    balance_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    weak_rows: list[dict[str, object]] = []
    nn_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    protocol_violations: list[str] = []
    target_expert_excluded = True

    try:
        _validate_imported_artifacts(cfg)
        references, decision_cell_set_hash = _load_imported_references(cfg)
        vanilla_refs = _load_vanilla_k16_references(cfg)
    except ProtocolError as exc:
        protocol_violations.append(str(exc))
        leakage = _leakage(protocol_violations, target_expert_excluded=True)
        _write_artifacts(
            root,
            cfg,
            matrix_rows=[],
            gap_rows=[],
            diagnostics_rows=[],
            balance_rows=[],
            component_rows=[],
            weak_rows=[],
            nn_rows=[],
            manifest_rows=[],
            decision=_decision([], cfg, leakage_status=leakage.status, decision_cell_set_hash=""),
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

                rows, diag, balance, components, weak, nn = _evaluate_runtime(
                    cfg,
                    references=references,
                    vanilla_refs=vanilla_refs,
                    runtime=union_runtime.runtime,
                    experiment_seed=int(experiment_seed),
                    heldout_center=str(heldout_center),
                    eval_raw=eval_raw,
                    eval_labels=eval_labels,
                    eval_error=eval_error,
                    include_primary=True,
                )
                matrix_rows.extend(rows)
                diagnostics_rows.extend(diag)
                balance_rows.extend(balance)
                component_rows.extend(components)
                weak_rows.extend(weak)
                nn_rows.extend(nn)

                for expert_id in candidates:
                    rows, diag, balance, components, weak, nn = _evaluate_runtime(
                        cfg,
                        references=references,
                        vanilla_refs=vanilla_refs,
                        runtime=per_source_runtime[str(expert_id)].runtime,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        eval_raw=eval_raw,
                        eval_labels=eval_labels,
                        eval_error=eval_error,
                        include_primary=False,
                    )
                    matrix_rows.extend(rows)
                    diagnostics_rows.extend(diag)
                    balance_rows.extend(balance)
                    component_rows.extend(components)
                    weak_rows.extend(weak)
                    nn_rows.extend(nn)
    except ProtocolError as exc:
        protocol_violations.append(str(exc))

    gap_rows = [dict(row) for row in matrix_rows if row.get("status") == "ok"]
    leakage = _leakage(protocol_violations, target_expert_excluded=target_expert_excluded)
    decision = _decision(matrix_rows, cfg, leakage_status=leakage.status, decision_cell_set_hash=decision_cell_set_hash)
    _write_artifacts(
        root,
        cfg,
        matrix_rows=matrix_rows,
        gap_rows=gap_rows,
        diagnostics_rows=diagnostics_rows,
        balance_rows=balance_rows,
        component_rows=component_rows,
        weak_rows=weak_rows,
        nn_rows=nn_rows,
        manifest_rows=manifest_rows,
        decision=decision,
        protocol_violations=protocol_violations,
        target_expert_excluded=target_expert_excluded,
    )
    return root


def _evaluate_runtime(
    cfg: SourceUnionBalancedGmmConfig,
    *,
    references: Mapping[tuple[object, ...], ImportedReference],
    vanilla_refs: Mapping[tuple[object, ...], VanillaK16Reference],
    runtime: VariantRuntime,
    experiment_seed: int,
    heldout_center: str,
    eval_raw: object,
    eval_labels: Sequence[int],
    eval_error: str,
    include_primary: bool,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    rows: list[dict[str, object]] = []
    diagnostics_rows: list[dict[str, object]] = []
    balance_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    weak_rows: list[dict[str, object]] = []
    nn_rows: list[dict[str, object]] = []
    source_error = "" if set(int(v) for v in runtime.source_train_labels) == {0, 1} else "mono_class_source_train"
    error = eval_error or source_error
    specs = _method_specs(include_primary=include_primary)
    if error:
        for seed in cfg.replicate_seeds:
            ref = _safe_reference(references, runtime, experiment_seed, heldout_center, int(seed))
            vanilla = _safe_vanilla(vanilla_refs, runtime, experiment_seed, heldout_center, int(seed))
            rows.extend(_ineligible_rows(cfg, runtime, ref, vanilla, experiment_seed, heldout_center, int(seed), specs, error))
        return rows, diagnostics_rows, balance_rows, component_rows, weak_rows, nn_rows

    eval_x = runtime.frame.transform(_to_numpy(eval_raw))
    params_by_key: dict[tuple[str, int, bool], BalancedGmmParameters] = {}
    for spec in specs:
        key = (str(spec["strategy"]), int(spec["gmm_components"]), bool(spec["shuffled_label_control"]))
        if key in params_by_key:
            continue
        params = _fit_parameters(
            cfg,
            runtime,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            gmm_components=int(spec["gmm_components"]),
            strategy=str(spec["strategy"]),
            shuffled_label_control=bool(spec["shuffled_label_control"]),
        )
        params_by_key[key] = params
        diagnostics_rows.extend(params.diagnostics_rows)
        balance_rows.extend(params.balance_rows)

    for seed in cfg.replicate_seeds:
        ref = _reference_for_runtime(references, runtime, experiment_seed, heldout_center, int(seed))
        vanilla = _safe_vanilla(vanilla_refs, runtime, experiment_seed, heldout_center, int(seed))
        for spec in specs:
            method = str(spec["prior_method"])
            if method == ROW_VANILLA_K16_REFERENCE:
                row = _reference_row(cfg, runtime, ref, vanilla, experiment_seed, heldout_center, int(seed), method)
                rows.append(row)
                continue
            key = (str(spec["strategy"]), int(spec["gmm_components"]), bool(spec["shuffled_label_control"]))
            params = params_by_key[key]
            latent_seed = _latent_seed(experiment_seed, heldout_center, runtime.expert_id, method, seed)
            if params.status != "ok":
                rows.append(
                    _balanced_row(
                        cfg,
                        runtime,
                        params,
                        ref,
                        vanilla,
                        experiment_seed,
                        heldout_center,
                        prior_method=method,
                        replicate_seed=int(seed),
                        latent_sample_seed=latent_seed,
                        generated_features_hash="",
                        prediction_hash="",
                        bacc="",
                        macro_f1="",
                        selection_source=str(spec["selection_source"]),
                        status=params.status,
                        error_message=params.error_message,
                    )
                )
                continue
            generated, labels, component_counts = _sample_features(
                cfg,
                runtime,
                params,
                seed=latent_seed,
            )
            row = _evaluate_generated(
                cfg,
                runtime,
                params,
                ref,
                vanilla,
                experiment_seed,
                heldout_center,
                prior_method=method,
                replicate_seed=int(seed),
                latent_sample_seed=latent_seed,
                generated=generated,
                labels=labels,
                eval_x=eval_x,
                eval_labels=eval_labels,
                selection_source=str(spec["selection_source"]),
            )
            coverage = _component_coverage_row(row, component_counts, params)
            row.update(
                {
                    "generated_component_counts_json": coverage["generated_component_counts_json"],
                    "num_active_components_unsampled": coverage["num_active_components_unsampled"],
                    "min_generated_samples_per_active_component": coverage["min_generated_samples_per_active_component"],
                    "component_weight_entropy": coverage["component_weight_entropy"],
                    "max_component_weight": coverage["max_component_weight"],
                    "latent_component_undersampled": coverage["latent_component_undersampled"],
                }
            )
            rows.append(row)
            component_rows.append(coverage)
            if _float(row["bacc"]) < 0.75:
                weak_rows.append(_weak_cell_row(row))
            nn_rows.append(_nearest_neighbor_row(row, generated, runtime.source_train_embeddings))
    return rows, diagnostics_rows, balance_rows, component_rows, weak_rows, nn_rows


def _method_specs(*, include_primary: bool) -> list[dict[str, object]]:
    if include_primary:
        return [
            {
                "prior_method": PRIMARY_BALANCED_METHOD,
                "gmm_components": 16,
                "strategy": "center_balanced",
                "shuffled_label_control": False,
                "selection_source": PRIMARY_SELECTION,
            },
            {
                "prior_method": ROW_VANILLA_K16_REFERENCE,
                "gmm_components": 16,
                "strategy": "reference_only",
                "shuffled_label_control": False,
                "selection_source": DIAGNOSTIC_SELECTION,
            },
            {
                "prior_method": ROW_BALANCED_K8,
                "gmm_components": 8,
                "strategy": "center_balanced",
                "shuffled_label_control": False,
                "selection_source": DIAGNOSTIC_SELECTION,
            },
            {
                "prior_method": ROW_BALANCED_K24,
                "gmm_components": 24,
                "strategy": "center_balanced",
                "shuffled_label_control": False,
                "selection_source": DIAGNOSTIC_SELECTION,
            },
            {
                "prior_method": ROW_CENTER_STRATIFIED_K4X4,
                "gmm_components": 4,
                "strategy": "center_stratified",
                "shuffled_label_control": False,
                "selection_source": DIAGNOSTIC_SELECTION,
            },
            {
                "prior_method": ROW_SHUFFLED_LABEL_CONTROL,
                "gmm_components": 16,
                "strategy": "center_balanced",
                "shuffled_label_control": True,
                "selection_source": DIAGNOSTIC_SELECTION,
            },
        ]
    return [
        {
            "prior_method": ROW_PER_SOURCE_BALANCED_K16,
            "gmm_components": 16,
            "strategy": "center_balanced",
            "shuffled_label_control": False,
            "selection_source": DIAGNOSTIC_SELECTION,
        }
    ]


def _fit_parameters(
    cfg: SourceUnionBalancedGmmConfig,
    runtime: VariantRuntime,
    *,
    experiment_seed: int,
    heldout_center: str,
    gmm_components: int,
    strategy: str,
    shuffled_label_control: bool,
) -> BalancedGmmParameters:
    import numpy as np  # type: ignore
    import torch  # type: ignore
    from sklearn.mixture import GaussianMixture  # type: ignore

    x_np = np.asarray(runtime.source_train_embeddings, dtype=np.float32)
    y_np = np.asarray(runtime.source_train_labels, dtype=int)
    centers = tuple(str(v) for v in getattr(runtime, "source_train_centers", ()))
    if len(centers) != len(y_np):
        return _empty_params(gmm_components, strategy, shuffled_label_control, "missing_source_train_centers")
    if str(heldout_center) in set(centers):
        raise ProtocolError(f"Held-out target center {heldout_center} entered source-union GMM fitting.")

    with torch.no_grad():
        x = torch.as_tensor(x_np, dtype=torch.float32)
        y = torch.as_tensor(y_np, dtype=torch.long)
        mu, _logvar = runtime.model.encode(x, y)
    mu_np = mu.detach().cpu().numpy()
    fit_labels = y_np.copy()
    if shuffled_label_control:
        rng = np.random.default_rng(_latent_seed(experiment_seed, heldout_center, runtime.expert_id, strategy, gmm_components, "shuffle"))
        rng.shuffle(fit_labels)

    source_centers = sorted(set(centers))
    fit_ids_hash = _hash_strings(runtime.source_train_sample_ids)
    classes: dict[int, GmmClassStats] = {}
    center_classes: dict[tuple[int, str], GmmClassStats] = {}
    diagnostics: list[dict[str, object]] = []
    balance_rows: list[dict[str, object]] = []
    parameter_payload = []
    status = "ok"
    errors: list[str] = []
    replacement_rates: list[float] = []
    fit_counts: dict[str, int] = {}
    sample_counts: dict[str, int] = {}

    for cls in (0, 1):
        if strategy == "center_stratified":
            for center in source_centers:
                positions, balance = _balanced_positions_for_center_class(
                    cfg,
                    y_np=fit_labels,
                    centers=centers,
                    cls=cls,
                    center=center,
                    experiment_seed=experiment_seed,
                    heldout_center=heldout_center,
                    runtime=runtime,
                    strategy=strategy,
                    gmm_components=gmm_components,
                )
                replacement_rates.append(float(balance["replacement_rate"]))
                balance_rows.append(balance)
                fit_counts[f"{center}:{cls}"] = int(balance["available_count"])
                sample_counts[f"{center}:{cls}"] = int(balance["sampled_count"])
                if int(balance["available_count"]) < cfg.min_source_center_class_count:
                    status = "ineligible"
                    errors.append(f"center_{center}_class_{cls}_count<{cfg.min_source_center_class_count}")
                    continue
                if int(balance["sampled_count"]) < int(gmm_components):
                    status = "ineligible"
                    errors.append(f"center_{center}_class_{cls}_fit_count<{gmm_components}")
                    continue
                stats = _fit_single_gmm(
                    cfg,
                    mu_np[positions],
                    class_label=cls,
                    center=center,
                    gmm_components=gmm_components,
                    random_state=_latent_seed(experiment_seed, heldout_center, runtime.expert_id, strategy, gmm_components, cls, center),
                )
                center_classes[(int(cls), str(center))] = stats
                diagnostics.append(_diagnostic_row(experiment_seed, heldout_center, runtime, stats, strategy, shuffled_label_control))
                parameter_payload.extend([stats.weights, stats.means, stats.covariances])
                if not stats.gmm_converged:
                    status = "gmm_fit_fail"
                    errors.append(f"center_{center}_class_{cls}_gmm_converged=false")
                elif stats.effective_gmm_components < min(cfg.min_effective_gmm_components, gmm_components):
                    status = "gmm_component_collapse"
                    errors.append(f"center_{center}_class_{cls}_effective_components<{min(cfg.min_effective_gmm_components, gmm_components)}")
            continue

        sampled_positions: list[int] = []
        for center in source_centers:
            positions, balance = _balanced_positions_for_center_class(
                cfg,
                y_np=fit_labels,
                centers=centers,
                cls=cls,
                center=center,
                experiment_seed=experiment_seed,
                heldout_center=heldout_center,
                runtime=runtime,
                strategy=strategy,
                gmm_components=gmm_components,
            )
            replacement_rates.append(float(balance["replacement_rate"]))
            balance_rows.append(balance)
            fit_counts[f"{center}:{cls}"] = int(balance["available_count"])
            sample_counts[f"{center}:{cls}"] = int(balance["sampled_count"])
            if int(balance["available_count"]) < cfg.min_source_center_class_count:
                status = "ineligible"
                errors.append(f"center_{center}_class_{cls}_count<{cfg.min_source_center_class_count}")
            sampled_positions.extend(int(v) for v in positions)
        if not sampled_positions:
            continue
        if len(sampled_positions) < int(gmm_components):
            status = "ineligible"
            errors.append(f"class_{cls}_fit_count<{gmm_components}")
            continue
        stats = _fit_single_gmm(
            cfg,
            mu_np[np.asarray(sampled_positions, dtype=int)],
            class_label=cls,
            center="source_union_balanced",
            gmm_components=gmm_components,
            random_state=_latent_seed(experiment_seed, heldout_center, runtime.expert_id, strategy, gmm_components, cls),
        )
        classes[int(cls)] = stats
        diagnostics.append(_diagnostic_row(experiment_seed, heldout_center, runtime, stats, strategy, shuffled_label_control))
        parameter_payload.extend([stats.weights, stats.means, stats.covariances])
        if not stats.gmm_converged:
            status = "gmm_fit_fail"
            errors.append(f"class_{cls}_gmm_converged=false")
        elif stats.effective_gmm_components < min(cfg.min_effective_gmm_components, gmm_components):
            status = "gmm_component_collapse"
            errors.append(f"class_{cls}_effective_components<{min(cfg.min_effective_gmm_components, gmm_components)}")

    max_replacement = max(replacement_rates) if replacement_rates else math.nan
    mean_replacement = nanmean(replacement_rates) if replacement_rates else math.nan
    parameter_hash = _hash_array(_flatten_payload(parameter_payload)) if parameter_payload else ""
    return BalancedGmmParameters(
        classes=classes,
        center_classes=center_classes,
        gmm_fit_row_ids_hash=fit_ids_hash,
        gmm_parameter_hash=parameter_hash,
        diagnostics_rows=tuple(diagnostics),
        balance_rows=tuple(balance_rows),
        status=status,
        error_message="|".join(sorted(set(errors))),
        gmm_components=int(gmm_components),
        strategy=strategy,
        shuffled_label_control=bool(shuffled_label_control),
        source_center_fit_counts_json=json.dumps(fit_counts, sort_keys=True),
        source_center_sample_counts_json=json.dumps(sample_counts, sort_keys=True),
        fit_sample_replacement_rate=mean_replacement,
        max_center_class_replacement_rate=max_replacement,
        mean_center_class_replacement_rate=mean_replacement,
    )


def _empty_params(gmm_components: int, strategy: str, shuffled: bool, error: str) -> BalancedGmmParameters:
    return BalancedGmmParameters(
        classes={},
        center_classes={},
        gmm_fit_row_ids_hash="",
        gmm_parameter_hash="",
        diagnostics_rows=(),
        balance_rows=(),
        status="ineligible",
        error_message=error,
        gmm_components=int(gmm_components),
        strategy=strategy,
        shuffled_label_control=bool(shuffled),
        source_center_fit_counts_json="{}",
        source_center_sample_counts_json="{}",
        fit_sample_replacement_rate=math.nan,
        max_center_class_replacement_rate=math.nan,
        mean_center_class_replacement_rate=math.nan,
    )


def _balanced_positions_for_center_class(
    cfg: SourceUnionBalancedGmmConfig,
    *,
    y_np: object,
    centers: Sequence[str],
    cls: int,
    center: str,
    experiment_seed: int,
    heldout_center: str,
    runtime: VariantRuntime,
    strategy: str,
    gmm_components: int,
) -> tuple[object, dict[str, object]]:
    import numpy as np  # type: ignore

    y = np.asarray(y_np, dtype=int)
    center_np = np.asarray([str(v) for v in centers])
    pool = np.flatnonzero((y == int(cls)) & (center_np == str(center)))
    available = int(pool.size)
    target = int(cfg.balanced_fit_samples_per_center_class)
    replace = available < target
    rng = np.random.default_rng(_latent_seed(experiment_seed, heldout_center, runtime.expert_id, strategy, gmm_components, cls, center, "fit_pool"))
    if available == 0:
        selected = np.asarray([], dtype=int)
    else:
        selected = rng.choice(pool, size=target, replace=replace)
    replacement_rate = max(0.0, (target - available) / float(target)) if target > 0 else math.nan
    return selected, {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "expert_id": runtime.expert_id,
        "expert_pool_type": runtime.variant.expert_pool_type,
        "variant_id": runtime.variant.variant_id,
        "source_center": str(center),
        "class_label": int(cls),
        "strategy": strategy,
        "gmm_components": int(gmm_components),
        "available_count": available,
        "sampled_count": int(selected.size),
        "sampled_with_replacement": bool(replace),
        "replacement_rate": replacement_rate,
        "selected_fit_ids_hash": _hash_strings([runtime.source_train_sample_ids[int(pos)] for pos in selected]) if selected.size else "",
    }


def _fit_single_gmm(
    cfg: SourceUnionBalancedGmmConfig,
    fit_mu: object,
    *,
    class_label: int,
    center: str,
    gmm_components: int,
    random_state: int,
) -> GmmClassStats:
    import numpy as np  # type: ignore
    from sklearn.mixture import GaussianMixture  # type: ignore

    fit_mu_np = np.asarray(fit_mu, dtype=float)
    gmm = GaussianMixture(
        n_components=int(gmm_components),
        covariance_type="diag",
        reg_covar=float(cfg.gmm_reg_covar),
        n_init=int(cfg.gmm_n_init),
        max_iter=int(cfg.gmm_max_iter),
        random_state=int(random_state),
    )
    gmm.fit(fit_mu_np)
    weights = np.asarray(gmm.weights_, dtype=float)
    covariances = np.asarray(gmm.covariances_, dtype=float)
    return GmmClassStats(
        class_label=int(class_label),
        center=str(center),
        class_train_count=int(fit_mu_np.shape[0]),
        weights=weights,
        means=np.asarray(gmm.means_, dtype=float),
        covariances=covariances,
        effective_gmm_components=int(np.sum(weights >= cfg.gmm_weight_floor)),
        min_component_weight=float(np.min(weights)),
        num_components_below_weight_floor=int(np.sum(weights < cfg.gmm_weight_floor)),
        num_components_covariance_clipped=int(np.sum(np.any(covariances <= cfg.gmm_reg_covar * 1.000001, axis=1))),
        gmm_converged=bool(gmm.converged_),
        gmm_n_iter=int(gmm.n_iter_),
        source_train_log_likelihood=float(gmm.score(fit_mu_np)),
        source_inner_bic=float(gmm.bic(fit_mu_np)),
    )


def _diagnostic_row(
    experiment_seed: int,
    heldout_center: str,
    runtime: VariantRuntime,
    stats: GmmClassStats,
    strategy: str,
    shuffled_label_control: bool,
) -> dict[str, object]:
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "expert_id": runtime.expert_id,
        "expert_pool_type": runtime.variant.expert_pool_type,
        "variant_id": runtime.variant.variant_id,
        "class_label": stats.class_label,
        "source_center": stats.center,
        "strategy": strategy,
        "gmm_components": int(stats.weights.shape[0]),
        "shuffled_label_control": bool(shuffled_label_control),
        "class_train_count": stats.class_train_count,
        "effective_gmm_components": stats.effective_gmm_components,
        "min_component_weight": stats.min_component_weight,
        "num_components_below_weight_floor": stats.num_components_below_weight_floor,
        "num_components_covariance_clipped": stats.num_components_covariance_clipped,
        "gmm_converged": stats.gmm_converged,
        "gmm_n_iter": stats.gmm_n_iter,
        "source_train_log_likelihood": stats.source_train_log_likelihood,
        "source_inner_bic": stats.source_inner_bic,
        "status": "ok" if stats.gmm_converged else "gmm_fit_fail",
        "error_message": "" if stats.gmm_converged else "gmm_converged=false",
    }


def _sample_features(
    cfg: SourceUnionBalancedGmmConfig,
    runtime: VariantRuntime,
    params: BalancedGmmParameters,
    *,
    seed: int,
) -> tuple[object, tuple[int, ...], dict[int, dict[str, int]]]:
    import numpy as np  # type: ignore

    rng = np.random.default_rng(int(seed))
    labels = _balanced_labels(cfg.synthetic_per_class_total)
    chunks = []
    component_counts: dict[int, dict[str, int]] = {}
    if params.strategy == "center_stratified":
        centers = sorted({center for (_cls, center) in params.center_classes})
        for cls in (0, 1):
            per_center = _balanced_counts(cfg.synthetic_per_class_total, len(centers))
            class_chunks = []
            component_counts[int(cls)] = {}
            for center, n_samples in zip(centers, per_center):
                stats = params.center_classes[(int(cls), str(center))]
                z_np, counts = _sample_from_stats(cfg, rng, stats, int(n_samples))
                decoded, _ = _decode_latents(runtime, z_np, [int(cls)] * int(n_samples))
                class_chunks.append(decoded)
                for component, value in counts.items():
                    component_counts[int(cls)][f"{center}:{component}"] = int(value)
            chunks.append(np.vstack(class_chunks))
        return np.vstack(chunks), labels, component_counts

    for cls in (0, 1):
        stats = params.classes[int(cls)]
        z_np, counts = _sample_from_stats(cfg, rng, stats, cfg.synthetic_per_class_total)
        decoded, _ = _decode_latents(runtime, z_np, [int(cls)] * cfg.synthetic_per_class_total)
        chunks.append(decoded)
        component_counts[int(cls)] = {str(k): int(v) for k, v in counts.items()}
    return np.vstack(chunks), labels, component_counts


def _balanced_counts(total: int, n_groups: int) -> list[int]:
    base = int(total) // int(n_groups)
    rem = int(total) % int(n_groups)
    return [base + (1 if idx < rem else 0) for idx in range(int(n_groups))]


def _sample_from_stats(
    cfg: SourceUnionBalancedGmmConfig,
    rng: object,
    stats: GmmClassStats,
    n_samples: int,
) -> tuple[object, dict[int, int]]:
    import numpy as np  # type: ignore

    weights = np.asarray(stats.weights, dtype=float)
    components = rng.choice(np.arange(weights.shape[0]), size=int(n_samples), replace=True, p=weights / weights.sum())
    means = np.asarray(stats.means, dtype=np.float32)[components]
    variances = np.asarray(stats.covariances, dtype=np.float32)[components]
    eps = rng.normal(size=means.shape).astype(np.float32)
    z_np = means + np.sqrt(np.maximum(variances, cfg.gmm_reg_covar)).astype(np.float32) * eps
    unique, counts = np.unique(components, return_counts=True)
    return z_np, {int(k): int(v) for k, v in zip(unique, counts)}


def _evaluate_generated(
    cfg: SourceUnionBalancedGmmConfig,
    runtime: VariantRuntime,
    params: BalancedGmmParameters,
    ref: ImportedReference,
    vanilla: VanillaK16Reference,
    experiment_seed: int,
    heldout_center: str,
    *,
    prior_method: str,
    replicate_seed: int,
    latent_sample_seed: int,
    generated: object,
    labels: Sequence[int],
    eval_x: object,
    eval_labels: Sequence[int],
    selection_source: str,
) -> dict[str, object]:
    bundle = fit_locked_logistic_classifier(
        generated,
        labels,
        eval_x,
        classifier_seed=cfg.classifier_seed,
        expert_id=runtime.expert_id,
        class_weight=cfg.classifier_class_weight,
    )
    result = evaluate_probability_predictions(prior_method, bundle.probabilities, eval_labels)
    return _balanced_row(
        cfg,
        runtime,
        params,
        ref,
        vanilla,
        experiment_seed,
        heldout_center,
        prior_method=prior_method,
        replicate_seed=replicate_seed,
        latent_sample_seed=latent_sample_seed,
        generated_features_hash=_hash_array(generated),
        prediction_hash=_hash_array(bundle.probabilities),
        bacc=result.bacc,
        macro_f1=result.macro_f1,
        selection_source=selection_source,
        status="ok",
        error_message="",
    )


def _reference_row(
    cfg: SourceUnionBalancedGmmConfig,
    runtime: VariantRuntime,
    ref: ImportedReference,
    vanilla: VanillaK16Reference,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    prior_method: str,
) -> dict[str, object]:
    params = _empty_params(16, "reference_only", False, "")
    return _balanced_row(
        cfg,
        runtime,
        params,
        ref,
        vanilla,
        experiment_seed,
        heldout_center,
        prior_method=prior_method,
        replicate_seed=replicate_seed,
        latent_sample_seed=_latent_seed(experiment_seed, heldout_center, runtime.expert_id, prior_method, replicate_seed),
        generated_features_hash="",
        prediction_hash="",
        bacc=vanilla.bacc,
        macro_f1=vanilla.macro_f1,
        selection_source=DIAGNOSTIC_SELECTION,
        status="ok",
        error_message="",
    )


def _balanced_row(
    cfg: SourceUnionBalancedGmmConfig,
    runtime: VariantRuntime,
    params: BalancedGmmParameters,
    ref: ImportedReference,
    vanilla: VanillaK16Reference,
    experiment_seed: int,
    heldout_center: str,
    *,
    prior_method: str,
    replicate_seed: int,
    latent_sample_seed: int,
    generated_features_hash: str,
    prediction_hash: str,
    bacc: float | str,
    macro_f1: float | str,
    selection_source: str,
    status: str,
    error_message: str,
) -> dict[str, object]:
    bacc_value = _float(bacc)
    total_gap = ref.real_feature_bacc - bacc_value if math.isfinite(bacc_value) else math.nan
    clipped_gap = max(0.0, total_gap) if math.isfinite(total_gap) else math.nan
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "expert_id": runtime.expert_id,
        "expert_pool_type": runtime.variant.expert_pool_type,
        "variant_id": runtime.variant.variant_id,
        "prior_method": prior_method,
        "gmm_components": int(params.gmm_components),
        "effective_gmm_components": _effective_components(params),
        "replicate_seed": int(replicate_seed),
        "latent_sample_seed": int(latent_sample_seed),
        "source_center_balance_strategy": params.strategy,
        "source_center_fit_counts_json": params.source_center_fit_counts_json,
        "source_center_sample_counts_json": params.source_center_sample_counts_json,
        "fit_sample_replacement_rate": params.fit_sample_replacement_rate,
        "max_center_class_replacement_rate": params.max_center_class_replacement_rate,
        "mean_center_class_replacement_rate": params.mean_center_class_replacement_rate,
        "bacc": bacc,
        "macro_f1": macro_f1,
        "real_feature_bacc": ref.real_feature_bacc,
        "decode_mu_bacc": ref.decode_mu_bacc,
        "posterior_bacc": ref.posterior_bacc,
        "empirical_mu_bacc": ref.empirical_mu_bacc,
        "standard_prior_bacc": ref.standard_prior_bacc,
        "diag_prior_bacc": ref.diag_prior_bacc,
        "alpha010_prior_bacc": ref.alpha010_prior_bacc,
        "vanilla_k16_prior_bacc": vanilla.bacc,
        "delta_bacc_vs_standard": bacc_value - ref.standard_prior_bacc if math.isfinite(bacc_value) else math.nan,
        "delta_bacc_vs_diag": bacc_value - ref.diag_prior_bacc if math.isfinite(bacc_value) else math.nan,
        "delta_bacc_vs_alpha010": bacc_value - ref.alpha010_prior_bacc if math.isfinite(bacc_value) else math.nan,
        "delta_bacc_vs_empirical_mu": bacc_value - ref.empirical_mu_bacc if math.isfinite(bacc_value) else math.nan,
        "delta_bacc_vs_vanilla_k16": bacc_value - vanilla.bacc if math.isfinite(bacc_value) else math.nan,
        "total_balanced_gmm_prior_gap": total_gap,
        "clipped_preservation_gap": clipped_gap,
        "preservation_ratio": bacc_value / ref.real_feature_bacc if math.isfinite(bacc_value) and ref.real_feature_bacc > 0 else math.nan,
        "weak_cell_warning": bool(math.isfinite(bacc_value) and bacc_value < 0.75),
        "hard_cell_fail": bool(math.isfinite(bacc_value) and bacc_value < 0.60),
        "gmm_fit_row_ids_hash": params.gmm_fit_row_ids_hash,
        "gmm_parameter_hash": params.gmm_parameter_hash,
        "generated_features_hash": generated_features_hash,
        "prediction_hash": prediction_hash,
        "selection_source": selection_source,
        "status": status,
        "error_message": error_message,
        "decision_cell_id": ref.decision_cell_id,
        "decision_cell_set_hash": ref.decision_cell_set_hash,
        "source_utility_stratum_reference": ref.source_utility_stratum_reference,
        "variant_real_budget_bacc": ref.variant_real_budget_bacc,
    }


def _effective_components(params: BalancedGmmParameters) -> int | str:
    values = [stats.effective_gmm_components for stats in params.classes.values()]
    values.extend(stats.effective_gmm_components for stats in params.center_classes.values())
    return min(values) if values else ""


def _component_coverage_row(
    row: Mapping[str, object],
    component_counts: Mapping[int, Mapping[str, int]],
    params: BalancedGmmParameters,
) -> dict[str, object]:
    active_components = _active_component_keys(params)
    sampled = {f"{cls}:{component}" for cls, counts in component_counts.items() for component in counts}
    counts = [int(v) for class_counts in component_counts.values() for v in class_counts.values()]
    total = float(sum(counts))
    fractions = [value / total for value in counts] if total else []
    entropy = -sum(p * math.log(p) for p in fractions if p > 0.0)
    unsampled = sorted(active_components.difference(sampled))
    min_per_active = 0 if unsampled else (min(counts) if counts else 0)
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
        "min_generated_samples_per_active_component": min_per_active,
        "component_weight_entropy": entropy,
        "max_component_weight": _max_component_weight(params),
        "latent_component_undersampled": bool(unsampled),
    }


def _active_component_keys(params: BalancedGmmParameters) -> set[str]:
    out: set[str] = set()
    for cls, stats in params.classes.items():
        for idx, weight in enumerate(stats.weights):
            if float(weight) >= 1.0e-12:
                out.add(f"{cls}:{idx}")
    for (cls, center), stats in params.center_classes.items():
        for idx, weight in enumerate(stats.weights):
            if float(weight) >= 1.0e-12:
                out.add(f"{cls}:{center}:{idx}")
    return out


def _max_component_weight(params: BalancedGmmParameters) -> float:
    values = []
    for stats in list(params.classes.values()) + list(params.center_classes.values()):
        values.extend(float(value) for value in stats.weights)
    return max(values) if values else math.nan


def _weak_cell_row(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "experiment_seed": row["experiment_seed"],
        "heldout_center": row["heldout_center"],
        "expert_id": row["expert_id"],
        "expert_pool_type": row["expert_pool_type"],
        "variant_id": row["variant_id"],
        "prior_method": row["prior_method"],
        "replicate_seed": row["replicate_seed"],
        "bacc": row["bacc"],
        "macro_f1": row["macro_f1"],
        "weak_cell_warning": row["weak_cell_warning"],
        "hard_cell_fail": row["hard_cell_fail"],
        "delta_bacc_vs_vanilla_k16": row["delta_bacc_vs_vanilla_k16"],
        "clipped_preservation_gap": row["clipped_preservation_gap"],
        "status": row["status"],
    }


def _ineligible_rows(
    cfg: SourceUnionBalancedGmmConfig,
    runtime: VariantRuntime,
    ref: ImportedReference,
    vanilla: VanillaK16Reference,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    specs: Sequence[Mapping[str, object]],
    error_message: str,
) -> list[dict[str, object]]:
    rows = []
    for spec in specs:
        params = _empty_params(int(spec["gmm_components"]), str(spec["strategy"]), bool(spec["shuffled_label_control"]), error_message)
        rows.append(
            _balanced_row(
                cfg,
                runtime,
                params,
                ref,
                vanilla,
                experiment_seed,
                heldout_center,
                prior_method=str(spec["prior_method"]),
                replicate_seed=int(replicate_seed),
                latent_sample_seed=_latent_seed(experiment_seed, heldout_center, runtime.expert_id, str(spec["prior_method"]), replicate_seed),
                generated_features_hash="",
                prediction_hash="",
                bacc="",
                macro_f1="",
                selection_source=str(spec["selection_source"]),
                status="ineligible",
                error_message=error_message,
            )
        )
    return rows


def _load_vanilla_k16_references(cfg: SourceUnionBalancedGmmConfig) -> dict[tuple[object, ...], VanillaK16Reference]:
    path = cfg.source_union_gmm_artifact_root / "tables" / "gmm_prior_gap_summary.csv"
    required = {
        "experiment_seed",
        "heldout_center",
        "expert_id",
        "expert_pool_type",
        "variant_id",
        "replicate_seed",
        "prior_method",
        "bacc",
        "macro_f1",
        "clipped_preservation_gap",
        "preservation_ratio",
        "status",
    }
    rows = _read_required_csv(path, required, "Source-union GMM v1 gap summary")
    out: dict[tuple[object, ...], VanillaK16Reference] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        if row.get("expert_pool_type") != POOL_SOURCE_UNION or row.get("variant_id") != UNION_VARIANT:
            continue
        if row.get("prior_method") != "source_union_cc_diag_gmm_k16_prior_sample_diagnostic":
            continue
        key = _reference_key(
            row["experiment_seed"],
            row["heldout_center"],
            row["expert_id"],
            row["expert_pool_type"],
            row["variant_id"],
            row["replicate_seed"],
        )
        out[key] = VanillaK16Reference(
            bacc=float(row["bacc"]),
            macro_f1=float(row["macro_f1"]),
            clipped_preservation_gap=float(row["clipped_preservation_gap"]),
            preservation_ratio=float(row["preservation_ratio"]),
        )
    if not out:
        raise ProtocolError("Source-union GMM artifact did not contain vanilla K16 diagnostic references.")
    return out


def _validate_imported_artifacts(cfg: SourceUnionBalancedGmmConfig) -> None:
    required = (
        cfg.sampling_artifact_root / "reports" / "leakage_report.json",
        cfg.prior_calibration_artifact_root / "reports" / "leakage_report.json",
        cfg.covariance_confirmation_artifact_root / "reports" / "leakage_report.json",
        cfg.source_union_gmm_artifact_root / "reports" / "leakage_report.json",
        cfg.source_union_gmm_artifact_root / "tables" / "gmm_prior_gap_summary.csv",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ProtocolError(f"Missing imported balanced-GMM reference artifacts: {missing}")
    for path in required:
        if path.name != "leakage_report.json":
            continue
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("status") != "PASS":
            raise ProtocolError(f"Imported leakage report is not PASS: {path}")


def _safe_reference(
    references: Mapping[tuple[object, ...], ImportedReference],
    runtime: VariantRuntime,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
) -> ImportedReference:
    try:
        return _reference_for_runtime(references, runtime, experiment_seed, heldout_center, replicate_seed)
    except ProtocolError:
        return _empty_reference(experiment_seed, heldout_center, runtime)


def _safe_vanilla(
    vanilla_refs: Mapping[tuple[object, ...], VanillaK16Reference],
    runtime: VariantRuntime,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
) -> VanillaK16Reference:
    key = _reference_key(
        experiment_seed,
        heldout_center,
        runtime.expert_id,
        runtime.variant.expert_pool_type,
        runtime.variant.variant_id,
        replicate_seed,
    )
    return vanilla_refs.get(key, VanillaK16Reference(math.nan, math.nan, math.nan, math.nan))


def _decision(
    rows: Sequence[Mapping[str, object]],
    cfg: SourceUnionBalancedGmmConfig,
    *,
    leakage_status: str,
    decision_cell_set_hash: str,
) -> dict[str, object]:
    primary_all = _rows_for(rows, PRIMARY_BALANCED_METHOD, POOL_SOURCE_UNION, include_non_ok=True)
    primary = [row for row in primary_all if row.get("status") == "ok"]
    control = _rows_for(rows, ROW_SHUFFLED_LABEL_CONTROL, POOL_SOURCE_UNION)
    vanilla = _rows_for(rows, ROW_VANILLA_K16_REFERENCE, POOL_SOURCE_UNION)
    k24 = _rows_for(rows, ROW_BALANCED_K24, POOL_SOURCE_UNION)
    stratified = _rows_for(rows, ROW_CENTER_STRATIFIED_K4X4, POOL_SOURCE_UNION)
    per_source = _rows_for(rows, ROW_PER_SOURCE_BALANCED_K16, POOL_PER_SOURCE)
    stats = _union_stats(primary)
    vanilla_stats = _union_stats(vanilla)
    control_stats = _union_stats(control)
    k24_stats = _union_stats(k24)
    stratified_stats = _union_stats(stratified)
    per_source_stats = _per_source_stats(per_source)
    numeric_pass = _primary_pass(stats, leakage_status=leakage_status)
    gmm_fit_ineligible = any(
        row.get("status") in {"ineligible", "gmm_fit_fail", "gmm_component_collapse"}
        and row.get("error_message") != "mono_class_target_eval"
        for row in primary_all
    )
    target_eval_ineligible = any(row.get("error_message") == "mono_class_target_eval" for row in primary_all)
    negative_control_competitive = _negative_control_competitive(stats, control_stats)
    verdict = "BALANCED_GMM_PRIOR_FAIL"
    if leakage_status != "PASS":
        verdict = "PROTOCOL_FAIL"
    elif gmm_fit_ineligible:
        verdict = "GMM_FIT_INELIGIBLE"
    elif int(stats["n_decision_cells"]) < 14 or int(stats["n_heldout_centers"]) < len(cfg.heldout_centers) or int(stats["min_eligible_seeds_per_center"]) < 2:
        verdict = "TARGET_EVAL_INSUFFICIENT"
    elif negative_control_competitive:
        verdict = "NEGATIVE_CONTROL_FAIL"
    elif numeric_pass:
        verdict = "SOURCE_UNION_CENTER_BALANCED_K16_0P90_PASS_DIAGNOSTIC"
    elif (
        _float(stats["center_equal_mean_bacc"]) >= 0.88
        and _float(stats["mean_delta_bacc_vs_standard"]) >= 0.15
        and _float(stats["mean_delta_bacc_vs_diag"]) >= 0.04
        and _float(stats["mean_delta_bacc_vs_alpha010"]) >= 0.01
        and _float(stats["mean_delta_bacc_vs_vanilla_k16"]) >= 0.0
    ):
        verdict = "SOURCE_UNION_CENTER_BALANCED_K16_PARTIAL"
    elif _float(stratified_stats["center_equal_mean_bacc"]) >= 0.90 and _float(stats["center_equal_mean_bacc"]) < 0.90:
        verdict = "CENTER_STRATIFIED_DIAGNOSTIC_LEAD"
    elif _float(vanilla_stats["center_equal_mean_bacc"]) > _float(stats["center_equal_mean_bacc"]):
        verdict = "VANILLA_K16_PREFERRED"
    elif _float(stats["empirical_mu_mean_bacc"]) >= 0.90 and _float(stats["center_equal_mean_bacc"]) < 0.90:
        verdict = "GMM_APPROXIMATION_FAIL"

    flags = []
    if target_eval_ineligible:
        flags.append("TARGET_EVAL_INELIGIBLE_CELLS")
    if bool(stats["weak_cell_warning"]):
        flags.append("WEAK_CELL_WARNING")
    if _float(stats["center3_delta_vs_vanilla_k16"]) < 0.0 or _center_mean(stats, "3") < 0.85:
        flags.append("CENTER3_STILL_WEAK")
    if _float(stats["center4_delta_vs_vanilla_k16"]) < 0.0 or _center_mean(stats, "4") < 0.85:
        flags.append("CENTER4_STILL_WEAK")
    if _float(k24_stats["center_equal_mean_bacc"]) > _float(stats["center_equal_mean_bacc"]):
        flags.append("K24_DIAGNOSTIC_LEAD")
    if _float(stratified_stats["center_equal_mean_bacc"]) > _float(stats["center_equal_mean_bacc"]):
        flags.append("CENTER_STRATIFIED_DIAGNOSTIC_LEAD")
    if _float(per_source_stats["center_equal_mean_bacc"]) < _float(stats["center_equal_mean_bacc"]) - 0.05:
        flags.append("PER_SOURCE_STILL_WEAK")
    if negative_control_competitive:
        flags.append("NEGATIVE_CONTROL_COMPETITIVE")
    if _float(stats["max_center_class_replacement_rate"]) > 0.0:
        flags.append("SOURCE_CENTER_REPLACEMENT_USED")
    if bool(stats["latent_component_undersampled"]):
        flags.append("LATENT_COMPONENT_UNDERSAMPLED")

    return {
        "primary_verdict": verdict,
        "diagnostic_flags": "|".join(flags),
        "decision_cell_set_hash": decision_cell_set_hash,
        "leakage_status": leakage_status,
        "vanilla_k16_center_equal_mean_bacc": vanilla_stats["center_equal_mean_bacc"],
        "negative_control_center_equal_mean_bacc": control_stats["center_equal_mean_bacc"],
        "k24_center_equal_mean_bacc": k24_stats["center_equal_mean_bacc"],
        "center_stratified_center_equal_mean_bacc": stratified_stats["center_equal_mean_bacc"],
        "per_source_center_equal_mean_bacc": per_source_stats["center_equal_mean_bacc"],
        **stats,
    }


def _primary_pass(stats: Mapping[str, object], *, leakage_status: str) -> bool:
    return (
        int(stats["n_decision_cells"]) >= 14
        and int(stats["n_heldout_centers"]) >= 5
        and int(stats["min_eligible_seeds_per_center"]) >= 2
        and _float(stats["center_equal_mean_bacc"]) >= 0.90
        and _float(stats["macro_f1_mean"]) >= 0.88
        and _float(stats["seed_std_bacc"]) <= 0.06
        and _float(stats["min_center_mean_bacc"]) >= 0.85
        and _float(stats["min_cell_bacc"]) >= 0.60
        and _float(stats["mean_clipped_preservation_gap"]) <= 0.05
        and _float(stats["mean_preservation_ratio"]) >= 0.95
        and _float(stats["mean_delta_bacc_vs_standard"]) >= 0.15
        and _float(stats["mean_delta_bacc_vs_diag"]) >= 0.04
        and _float(stats["mean_delta_bacc_vs_alpha010"]) >= 0.01
        and _float(stats["mean_delta_bacc_vs_empirical_mu"]) >= -0.01
        and _float(stats["mean_delta_bacc_vs_vanilla_k16"]) >= 0.005
        and _float(stats["paired_delta_vs_vanilla_k16_ci95_low"]) >= -0.005
        and _float(stats["weak_center_mean_delta_vs_vanilla_k16"]) >= 0.02
        and int(stats["num_cells_below_075"]) <= int(stats["vanilla_k16_num_cells_below_075"])
        and _float(stats["min_center_mean_bacc_delta_vs_vanilla_k16"]) >= 0.0
        and _float(stats["center3_delta_vs_vanilla_k16"]) >= 0.0
        and _float(stats["center4_delta_vs_vanilla_k16"]) >= 0.0
        and _float(stats["max_center_class_replacement_rate"]) <= 0.50
        and _float(stats["mean_center_class_replacement_rate"]) <= 0.25
        and not bool(stats["negative_control_competitive"])
        and leakage_status == "PASS"
    )


def _negative_control_competitive(stats: Mapping[str, object], control_stats: Mapping[str, object]) -> bool:
    control = _float(control_stats["center_equal_mean_bacc"])
    primary = _float(stats["center_equal_mean_bacc"])
    return math.isfinite(control) and (control >= 0.80 or primary - control <= 0.10)


def _rows_for(
    rows: Sequence[Mapping[str, object]],
    method: str,
    pool_type: str,
    *,
    include_non_ok: bool = False,
) -> list[Mapping[str, object]]:
    return [
        row for row in rows
        if row.get("prior_method") == method
        and row.get("expert_pool_type") == pool_type
        and (include_non_ok or row.get("status") == "ok")
    ]


def _union_stats(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    grouped = _replicate_averaged_union(rows)
    by_seed: dict[str, list[Mapping[str, object]]] = {}
    by_center: dict[str, list[Mapping[str, object]]] = {}
    for row in grouped:
        by_seed.setdefault(str(row["experiment_seed"]), []).append(row)
        by_center.setdefault(str(row["heldout_center"]), []).append(row)
    seed_means = [_mean_field(values, "bacc") for values in by_seed.values()]
    center_bacc = {center: _mean_field(values, "bacc") for center, values in sorted(by_center.items())}
    center_delta = {center: _mean_field(values, "delta_bacc_vs_vanilla_k16") for center, values in sorted(by_center.items())}
    vanilla_below = sum(1 for row in grouped if _float(row["vanilla_k16_prior_bacc"]) < 0.75)
    below = sum(1 for row in grouped if _float(row["bacc"]) < 0.75)
    weak_deltas = [
        _float(row["delta_bacc_vs_vanilla_k16"])
        for row in grouped
        if _float(row["vanilla_k16_prior_bacc"]) < 0.85 or str(row["heldout_center"]) in {"3", "4"}
    ]
    return {
        "n_raw_rows": len(rows),
        "n_decision_cells": len(grouped),
        "n_experiment_seeds": len(by_seed),
        "n_heldout_centers": len(by_center),
        "min_eligible_seeds_per_center": min((len({str(row["experiment_seed"]) for row in values}) for values in by_center.values()), default=0),
        "center_equal_mean_bacc": nanmean(seed_means) if seed_means else math.nan,
        "macro_f1_mean": _center_equal_mean(grouped, "macro_f1"),
        "seed_std_bacc": _std(seed_means),
        "min_center_mean_bacc": min(center_bacc.values()) if center_bacc else math.nan,
        "min_cell_bacc": _min_field(grouped, "bacc"),
        "mean_clipped_preservation_gap": _center_equal_mean(grouped, "clipped_preservation_gap"),
        "mean_preservation_ratio": _center_equal_mean(grouped, "preservation_ratio"),
        "mean_delta_bacc_vs_standard": _center_equal_mean(grouped, "delta_bacc_vs_standard"),
        "mean_delta_bacc_vs_diag": _center_equal_mean(grouped, "delta_bacc_vs_diag"),
        "mean_delta_bacc_vs_alpha010": _center_equal_mean(grouped, "delta_bacc_vs_alpha010"),
        "mean_delta_bacc_vs_empirical_mu": _center_equal_mean(grouped, "delta_bacc_vs_empirical_mu"),
        "mean_delta_bacc_vs_vanilla_k16": _center_equal_mean(grouped, "delta_bacc_vs_vanilla_k16"),
        "paired_delta_vs_vanilla_k16_ci95": _ci95([_float(row["delta_bacc_vs_vanilla_k16"]) for row in grouped]),
        "paired_delta_vs_vanilla_k16_ci95_low": _ci95_low([_float(row["delta_bacc_vs_vanilla_k16"]) for row in grouped]),
        "weak_center_mean_delta_vs_vanilla_k16": nanmean(weak_deltas) if weak_deltas else math.nan,
        "min_center_mean_bacc_delta_vs_vanilla_k16": min(center_delta.values()) if center_delta else math.nan,
        "center3_delta_vs_vanilla_k16": center_delta.get("3", math.nan),
        "center4_delta_vs_vanilla_k16": center_delta.get("4", math.nan),
        "num_cells_below_075": below,
        "vanilla_k16_num_cells_below_075": vanilla_below,
        "max_center_class_replacement_rate": _max_field(grouped, "max_center_class_replacement_rate"),
        "mean_center_class_replacement_rate": _center_equal_mean(grouped, "mean_center_class_replacement_rate"),
        "negative_control_competitive": False,
        "weak_cell_warning": any(_float(row["bacc"]) < 0.75 for row in grouped),
        "latent_component_undersampled": any(str(row.get("latent_component_undersampled", "False")) == "True" for row in grouped),
        "per_center_bacc": json.dumps(center_bacc, sort_keys=True),
        "per_seed_bacc": json.dumps({seed: _mean_field(values, "bacc") for seed, values in sorted(by_seed.items())}, sort_keys=True),
        "real_feature_ceiling": _center_equal_mean(grouped, "real_feature_bacc"),
        "empirical_mu_or_codebook_ceiling": _center_equal_mean(grouped, "empirical_mu_bacc"),
        "vanilla_k16_gmm_ceiling": _center_equal_mean(grouped, "vanilla_k16_prior_bacc"),
        "center_balanced_k16_gmm_ceiling": _center_equal_mean(grouped, "bacc"),
        "empirical_mu_mean_bacc": _center_equal_mean(grouped, "empirical_mu_bacc"),
    }


def _per_source_stats(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    grouped = _replicate_averaged_per_source(rows)
    by_seed: dict[str, list[Mapping[str, object]]] = {}
    for row in grouped:
        by_seed.setdefault(str(row["experiment_seed"]), []).append(row)
    seed_means = [_center_equal_from_expert_cells(values, "bacc") for values in by_seed.values()]
    return {
        "n_raw_rows": len(rows),
        "n_decision_cells": len(grouped),
        "center_equal_mean_bacc": nanmean(seed_means) if seed_means else math.nan,
        "seed_std_bacc": _std(seed_means),
        "min_cell_bacc": _min_field(grouped, "bacc"),
    }


def _replicate_averaged_union(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["experiment_seed"]), str(row["heldout_center"])), []).append(row)
    fields = (
        "bacc",
        "macro_f1",
        "real_feature_bacc",
        "empirical_mu_bacc",
        "vanilla_k16_prior_bacc",
        "delta_bacc_vs_standard",
        "delta_bacc_vs_diag",
        "delta_bacc_vs_alpha010",
        "delta_bacc_vs_empirical_mu",
        "delta_bacc_vs_vanilla_k16",
        "clipped_preservation_gap",
        "preservation_ratio",
        "max_center_class_replacement_rate",
        "mean_center_class_replacement_rate",
    )
    out = []
    for (seed, center), subset in groups.items():
        row = {"experiment_seed": seed, "heldout_center": center}
        row.update({field: _mean_field(subset, field) for field in fields})
        row["latent_component_undersampled"] = any(str(v.get("latent_component_undersampled", "False")) == "True" for v in subset)
        out.append(row)
    return out


def _replicate_averaged_per_source(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["experiment_seed"]), str(row["heldout_center"]), str(row["expert_id"])), []).append(row)
    out = []
    for (seed, center, expert), subset in groups.items():
        row = {"experiment_seed": seed, "heldout_center": center, "expert_id": expert}
        row["bacc"] = _mean_field(subset, "bacc")
        row["macro_f1"] = _mean_field(subset, "macro_f1")
        out.append(row)
    return out


def _center_equal_from_expert_cells(rows: Sequence[Mapping[str, object]], field: str) -> float:
    by_center: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        by_center.setdefault(str(row["heldout_center"]), []).append(row)
    return nanmean([_mean_field(values, field) for values in by_center.values()]) if by_center else math.nan


def _center_equal_mean(rows: Sequence[Mapping[str, object]], field: str) -> float:
    by_seed: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        by_seed.setdefault(str(row["experiment_seed"]), []).append(row)
    seed_values = [_mean_field(values, field) for values in by_seed.values()]
    return nanmean(seed_values) if seed_values else math.nan


def _center_mean(stats: Mapping[str, object], center: str) -> float:
    try:
        payload = json.loads(str(stats.get("per_center_bacc", "{}")))
        return float(payload.get(str(center), math.nan))
    except Exception:
        return math.nan


def _mean_field(rows: Sequence[Mapping[str, object]], field: str) -> float:
    return nanmean([_float(row[field]) for row in rows if field in row and str(row.get(field, "")) not in {"", NA}])


def _min_field(rows: Sequence[Mapping[str, object]], field: str) -> float:
    values = [_float(row[field]) for row in rows if field in row and math.isfinite(_float(row[field]))]
    return min(values) if values else math.nan


def _max_field(rows: Sequence[Mapping[str, object]], field: str) -> float:
    values = [_float(row[field]) for row in rows if field in row and math.isfinite(_float(row[field]))]
    return max(values) if values else math.nan


def _std(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if len(finite) < 2:
        return 0.0
    avg = sum(finite) / float(len(finite))
    return math.sqrt(sum((value - avg) ** 2 for value in finite) / float(len(finite)))


def _flatten_payload(values: Sequence[object]) -> object:
    import numpy as np  # type: ignore

    if not values:
        return np.asarray([], dtype=float)
    return np.concatenate([np.ravel(np.asarray(value, dtype=float)) for value in values])


def _latent_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _repair_runtime_config(cfg: SourceUnionBalancedGmmConfig, root: Path) -> RepairConfig:
    return RepairConfig(
        name="virchow2_cvae_preservation_repair_v1",
        artifact_root=root,
        feature_cache_root=cfg.feature_cache_root,
        experiment_seeds=cfg.experiment_seeds,
        heldout_centers=cfg.heldout_centers,
        replicate_seeds=cfg.replicate_seeds,
        synthetic_per_class_total=cfg.synthetic_per_class_total,
        primary_variant=PRIMARY_VARIANT,
        min_decision_rows=9,
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


def _leakage(protocol_violations: Sequence[str], *, target_expert_excluded: bool):
    return build_leakage_report(
        target_support_labels_for_selection=False,
        target_eval_labels_for_scoring_only=True,
        target_expert_excluded=target_expert_excluded,
        oracle_rows_diagnostic_only=True,
        extra_violations=protocol_violations,
    )


def _write_artifacts(
    root: Path,
    cfg: SourceUnionBalancedGmmConfig,
    *,
    matrix_rows: Sequence[Mapping[str, object]],
    gap_rows: Sequence[Mapping[str, object]],
    diagnostics_rows: Sequence[Mapping[str, object]],
    balance_rows: Sequence[Mapping[str, object]],
    component_rows: Sequence[Mapping[str, object]],
    weak_rows: Sequence[Mapping[str, object]],
    nn_rows: Sequence[Mapping[str, object]],
    manifest_rows: Sequence[Mapping[str, object]],
    decision: Mapping[str, object],
    protocol_violations: Sequence[str],
    target_expert_excluded: bool,
) -> None:
    write_csv_rows(root / "tables" / "balanced_gmm_downstream_matrix.csv", matrix_rows, columns=_matrix_columns())
    write_csv_rows(root / "tables" / "balanced_gmm_gap_summary.csv", gap_rows)
    write_csv_rows(root / "tables" / "source_union_balanced_gmm_summary.csv", [dict(decision)])
    write_csv_rows(root / "tables" / "source_center_balance_audit.csv", balance_rows)
    write_csv_rows(root / "tables" / "gmm_component_diagnostics.csv", diagnostics_rows)
    write_csv_rows(root / "tables" / "generated_component_coverage_audit.csv", component_rows)
    write_csv_rows(root / "tables" / "weak_cell_audit.csv", weak_rows)
    write_csv_rows(root / "tables" / "nearest_neighbor_memorization_audit.csv", nn_rows)
    write_csv_rows(root / "tables" / "negative_control_summary.csv", [_negative_control_summary(decision)])
    write_csv_rows(root / "tables" / "per_source_balanced_gmm_diagnostic_summary.csv", [_per_source_summary(decision)])
    write_csv_rows(root / "manifests" / "balanced_gmm_prior_model_manifest.csv", manifest_rows)
    leakage = _leakage(protocol_violations, target_expert_excluded=target_expert_excluded)
    write_json(root / "reports" / "leakage_report.json", leakage.to_json_dict())
    write_json(
        root / "manifests" / "protocol_manifest.json",
        {
            "schema_version": "cvae_rebuild_source_union_center_balanced_gmm_prior_protocol_manifest_v1",
            "experiment_name": cfg.name,
            "experiment_type": "source_union_center_balanced_gmm_prior_diagnostic",
            "primary_variant": cfg.primary_variant,
            "primary_method": cfg.primary_method,
            "target_support_labels_for_selection": False,
            "target_eval_labels_for_scoring_only": True,
            "target_expert_excluded": target_expert_excluded,
            "primary_population_does_not_filter_on_variant_real_budget_bacc": True,
            "claim_boundary": "source-union center-balanced sampled-feature utility diagnostic only; no routing, decentralized per-source expert selection, or formal privacy claim",
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
        "prior_method",
        "gmm_components",
        "effective_gmm_components",
        "replicate_seed",
        "latent_sample_seed",
        "source_center_balance_strategy",
        "source_center_fit_counts_json",
        "source_center_sample_counts_json",
        "fit_sample_replacement_rate",
        "max_center_class_replacement_rate",
        "mean_center_class_replacement_rate",
        "generated_component_counts_json",
        "num_active_components_unsampled",
        "min_generated_samples_per_active_component",
        "component_weight_entropy",
        "max_component_weight",
        "latent_component_undersampled",
        "bacc",
        "macro_f1",
        "real_feature_bacc",
        "decode_mu_bacc",
        "posterior_bacc",
        "empirical_mu_bacc",
        "standard_prior_bacc",
        "diag_prior_bacc",
        "alpha010_prior_bacc",
        "vanilla_k16_prior_bacc",
        "delta_bacc_vs_standard",
        "delta_bacc_vs_diag",
        "delta_bacc_vs_alpha010",
        "delta_bacc_vs_empirical_mu",
        "delta_bacc_vs_vanilla_k16",
        "total_balanced_gmm_prior_gap",
        "clipped_preservation_gap",
        "preservation_ratio",
        "weak_cell_warning",
        "hard_cell_fail",
        "gmm_fit_row_ids_hash",
        "gmm_parameter_hash",
        "generated_features_hash",
        "prediction_hash",
        "selection_source",
        "status",
        "error_message",
        "decision_cell_id",
        "decision_cell_set_hash",
        "source_utility_stratum_reference",
        "variant_real_budget_bacc",
    )


def _negative_control_summary(decision: Mapping[str, object]) -> dict[str, object]:
    return {
        "primary_method": PRIMARY_BALANCED_METHOD,
        "control_method": ROW_SHUFFLED_LABEL_CONTROL,
        "primary_center_equal_mean_bacc": decision.get("center_equal_mean_bacc", math.nan),
        "control_center_equal_mean_bacc": decision.get("negative_control_center_equal_mean_bacc", math.nan),
        "negative_control_competitive": "NEGATIVE_CONTROL_COMPETITIVE" in str(decision.get("diagnostic_flags", "")),
    }


def _per_source_summary(decision: Mapping[str, object]) -> dict[str, object]:
    return {
        "prior_method": ROW_PER_SOURCE_BALANCED_K16,
        "center_equal_mean_bacc": decision.get("per_source_center_equal_mean_bacc", math.nan),
    }


def _write_decision_summary(root: Path, decision: Mapping[str, object], *, leakage_status: str) -> None:
    text = "\n".join(
        [
            "# Virchow2-CVAE Source-Union Center-Balanced K16 GMM Prior v1",
            "",
            "## Summary",
            "",
            f"- Primary method: `{PRIMARY_BALANCED_METHOD}`",
            f"- Primary verdict: `{decision.get('primary_verdict', 'BALANCED_GMM_PRIOR_FAIL')}`",
            f"- Diagnostic flags: `{decision.get('diagnostic_flags', '')}`",
            f"- Center-equal mean BACC: {_format_float(decision.get('center_equal_mean_bacc'))}",
            f"- Macro-F1 mean: {_format_float(decision.get('macro_f1_mean'))}",
            f"- Seed std BACC: {_format_float(decision.get('seed_std_bacc'))}",
            f"- Min center mean BACC: {_format_float(decision.get('min_center_mean_bacc'))}",
            f"- Delta BACC vs vanilla K16: {_format_float(decision.get('mean_delta_bacc_vs_vanilla_k16'))}",
            f"- Weak-center delta vs vanilla K16: {_format_float(decision.get('weak_center_mean_delta_vs_vanilla_k16'))}",
            f"- Paired delta vs vanilla K16 CI95: `{decision.get('paired_delta_vs_vanilla_k16_ci95', '')}`",
            f"- Leakage status: `{leakage_status}`",
            "",
            "## Ceilings",
            "",
            f"- Real-feature ceiling: {_format_float(decision.get('real_feature_ceiling'))}",
            f"- Empirical-mu/codebook ceiling: {_format_float(decision.get('empirical_mu_or_codebook_ceiling'))}",
            f"- Vanilla K16 GMM ceiling: {_format_float(decision.get('vanilla_k16_gmm_ceiling'))}",
            f"- Center-balanced K16 GMM ceiling: {_format_float(decision.get('center_balanced_k16_gmm_ceiling'))}",
            "",
            "## Claim Boundary",
            "",
            "This adaptive diagnostic tests source-union center-balanced sampled-feature utility only.",
            "It does not provide formal differential privacy.",
            "It does not evaluate metadata routing.",
            "It does not evaluate support-NELBO routing.",
            "It does not evaluate top-k composition.",
            "It does not evaluate decentralized per-source expert selection.",
            "",
        ]
    )
    path = root / "reports" / "decision_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _resolved_config(cfg: SourceUnionBalancedGmmConfig) -> dict[str, object]:
    return {
        "name": cfg.name,
        "artifact_root": str(cfg.artifact_root),
        "repair_artifact_root": str(cfg.repair_artifact_root),
        "sampling_artifact_root": str(cfg.sampling_artifact_root),
        "prior_calibration_artifact_root": str(cfg.prior_calibration_artifact_root),
        "covariance_confirmation_artifact_root": str(cfg.covariance_confirmation_artifact_root),
        "source_union_gmm_artifact_root": str(cfg.source_union_gmm_artifact_root),
        "feature_cache_root": str(cfg.feature_cache_root),
        "experiment_seeds": list(cfg.experiment_seeds),
        "heldout_centers": list(cfg.heldout_centers),
        "replicate_seeds": list(cfg.replicate_seeds),
        "synthetic_per_class_total": cfg.synthetic_per_class_total,
        "primary_variant": cfg.primary_variant,
        "primary_method": cfg.primary_method,
        "gmm_components": cfg.gmm_components,
        "gmm_covariance_type": cfg.gmm_covariance_type,
        "gmm_reg_covar": cfg.gmm_reg_covar,
        "gmm_n_init": cfg.gmm_n_init,
        "gmm_max_iter": cfg.gmm_max_iter,
        "gmm_weight_floor": cfg.gmm_weight_floor,
        "min_source_center_class_count": cfg.min_source_center_class_count,
        "min_effective_gmm_components": cfg.min_effective_gmm_components,
        "balanced_fit_samples_per_center_class": cfg.balanced_fit_samples_per_center_class,
        "max_center_class_replacement_rate": cfg.max_center_class_replacement_rate,
        "mean_center_class_replacement_rate": cfg.mean_center_class_replacement_rate,
        "posterior_noise_scale": cfg.posterior_noise_scale,
        "diagnostic_gmm_components": list(cfg.diagnostic_gmm_components),
        "classifier": {
            "type": cfg.classifier_type,
            "solver": cfg.classifier_solver,
            "C": cfg.classifier_c,
            "max_iter": cfg.classifier_max_iter,
            "class_weight": cfg.classifier_class_weight,
            "classifier_seed": cfg.classifier_seed,
        },
    }
