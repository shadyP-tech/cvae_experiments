from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.metrics import balanced_accuracy, macro_f1
from core.protocol import ProtocolError, build_leakage_report
from core.reporting import prepare_artifact_dirs, write_csv_rows, write_json
from evaluation.downstream import evaluate_probability_predictions, fit_locked_logistic_classifier
from experiments.midogpp.midogpp_preservation_gate import (
    DEFAULT_ARTIFACT_ROOT as DEFAULT_PRESERVATION_GATE_ARTIFACT_ROOT,
)
from experiments.midogpp.midogpp_preservation_gate import (
    PRIMARY_VARIANT,
    REAL_PCA128_REFERENCE,
)
from experiments.midogpp.midogpp_preservation_sanity import (
    AUDIT_COLUMNS,
    BALANCED_CONTROL,
    CONTROL_ORDER,
    DEFAULT_FEATURE_CACHE_PATH,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_SIGNAL_SPLIT_MANIFEST_PATH,
    DECODE_MU,
    POSTERIOR_SAMPLE,
    VALID_STATUS,
    CVAERuntime,
    ManifestRow,
    SplitSpec,
    VariantConfig,
    _aggregate_rows,
    _assert_cache_alignment,
    _assert_virchow2_cache,
    _bool_text,
    _decode_mu,
    _empty_metric_row,
    _evaluate_representation,
    _finite,
    _fmt,
    _identity_audit_rows,
    _load_feature_cache,
    _mapping,
    _parse_variant,
    _path,
    _posterior_sample,
    _read_split_manifest,
    _read_train_manifest,
    _split_counts,
    _split_status,
    _stable_seed,
    _to_numpy,
    _train_runtime,
)


EXPERIMENT_NAME = "virchow2_cvae_midogpp_pca128_posthoc_gmm_prior_v1"
DEFAULT_ARTIFACT_ROOT = "cvae_rebuild/artifacts/midogpp/virchow2_cvae_midogpp_pca128_posthoc_gmm_prior_v1"

PRIMARY_METHOD = "gmm_on_posterior_mu_class_conditional_diag"
GAUSSIAN_METHOD = "gaussian_on_posterior_mu_class_conditional_diag"
GLOBAL_GMM_METHOD = "gmm_on_posterior_mu_global_diag"
LABEL_PERMUTED_GMM_CONTROL = "label_permuted_gmm_on_posterior_mu_control"
CLASS_PRIOR_SHUFFLED_CONTROL = "class_prior_shuffled_gmm_control"
RANDOM_LATENT_CONTROL = "random_standard_normal_latent_decode_control"
DECODE_MU_DIAGNOSTIC = DECODE_MU
POSTERIOR_SAMPLE_DIAGNOSTIC = POSTERIOR_SAMPLE
GATE_REQUIRED_LABEL = "GMM_FEASIBILITY_ALLOWED_NEXT"
CONDITION_WARNING_LABEL = "LATENT_CLASS_SIGNAL_DOMINATES_CONDITION_WARNING"

METRIC_COLUMNS = (
    "aggregation_level",
    "control_name",
    "domain_name",
    "split_seed",
    "variant_id",
    "representation_role",
    "method_role",
    "adoption_eligible",
    "diagnostic_only",
    "model_type",
    "eligible_seed_count",
    "valid_seed_count",
    "decision_status",
    "n_fit",
    "n_eval",
    "n_fit_pos",
    "n_fit_neg",
    "n_eval_pos",
    "n_eval_neg",
    "n_fit_cases",
    "n_eval_cases",
    "n_generated",
    "generated_pos",
    "generated_neg",
    "bacc",
    "bacc_std",
    "bacc_min",
    "macro_f1",
    "precision_pos",
    "recall_pos",
    "f1_pos",
    "support_pos",
    "support_neg",
    "ci_low",
    "ci_high",
    "ci_method",
    "above_chance",
    "near_chance",
    "converged",
    "status",
    "error_message",
    "preservation_ratio_vs_real_pca128",
)

GMM_DIAGNOSTIC_COLUMNS = (
    "control_name",
    "domain_name",
    "split_seed",
    "variant_id",
    "method_role",
    "class_label",
    "fit_n",
    "selected_k",
    "covariance_type",
    "reg_covar",
    "converged",
    "n_iter",
    "effective_components",
    "min_component_weight",
    "max_component_weight",
    "min_variance",
    "max_variance",
    "condition_number",
    "fallback_used",
    "status",
    "error_message",
)

PREDICTION_COLUMNS = (
    "control_name",
    "domain_name",
    "split_seed",
    "variant_id",
    "representation_role",
    "method_role",
    "sample_id",
    "case_id",
    "feature_row_index",
    "y_true",
    "y_pred",
    "prob_pos",
)

MODEL_MANIFEST_COLUMNS = (
    "control_name",
    "domain_name",
    "split_seed",
    "variant_id",
    "method_role",
    "pca_dim",
    "latent_dim",
    "posterior_source",
    "class_prior_policy",
    "synthetic_budget",
    "generation_seed",
    "classifier_seed",
    "fit_row_hash",
    "eval_row_hash",
    "generated_feature_hash",
    "prediction_hash",
    "preservation_gate_artifact_root",
    "preservation_gate_hash",
)


@dataclass(frozen=True)
class MidogPPPosthocGMMPriorConfig:
    name: str
    artifact_root: Path
    manifest_path: Path
    feature_cache_path: Path
    signal_split_manifest_path: Path
    preservation_gate_artifact_root: Path
    positive_label: int
    controls: tuple[str, ...]
    primary_variant: VariantConfig
    split_seeds: tuple[int, ...] | None
    allow_npz_test_cache: bool
    min_fit: int
    min_eval: int
    min_fit_pos: int
    min_fit_neg: int
    min_eval_pos: int
    min_eval_neg: int
    min_fit_cases: int
    min_eval_cases: int
    primary_method: str
    class_prior_policy: str
    synthetic_budget: int | None
    gmm_components: int
    gmm_covariance_type: str
    gmm_reg_covar: float
    gmm_n_init: int
    gmm_max_iter: int
    min_per_class_n: int
    min_samples_per_component: int
    fallback_to_single_gaussian: bool
    no_rejection: bool
    generation_seeds: tuple[int, ...]
    classifier_seeds: tuple[int, ...]
    bootstrap_reps: int
    bootstrap_seed: int
    real_gate_min_bacc: float
    ci_low_threshold: float
    gate_required_label: str
    allow_condition_warning: bool


def load_midogpp_posthoc_gmm_prior_config(path: str | Path) -> MidogPPPosthocGMMPriorConfig:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise ProtocolError("Loading MIDOG++ post-hoc GMM prior configs requires PyYAML.") from exc
    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ProtocolError(f"Config must be a mapping: {path}")
    base_dir = source.parents[2] if len(source.parents) >= 3 else source.parent
    return parse_midogpp_posthoc_gmm_prior_config(payload, base_dir=base_dir)


def parse_midogpp_posthoc_gmm_prior_config(
    data: Mapping[str, object],
    *,
    base_dir: str | Path = ".",
) -> MidogPPPosthocGMMPriorConfig:
    base = Path(base_dir)
    experiment = _mapping(data.get("experiment"), "experiment")
    inputs = _mapping(data.get("inputs"), "inputs")
    run = _mapping(data.get("run"), "run")
    thresholds = _mapping(data.get("validity_thresholds"), "validity_thresholds", allow_empty=True)
    gmm = _mapping(data.get("gmm_prior"), "gmm_prior")
    downstream = _mapping(data.get("downstream"), "downstream")
    bootstrap = _mapping(data.get("bootstrap"), "bootstrap", allow_empty=True)
    decisions = _mapping(data.get("decision_thresholds"), "decision_thresholds", allow_empty=True)
    variant_payload = data.get("primary_variant") or {
        "variant_id": PRIMARY_VARIANT,
        "pca_dim": 128,
        "latent_dim": 32,
    }
    gate_root = inputs.get("preservation_gate_artifact_root", DEFAULT_PRESERVATION_GATE_ARTIFACT_ROOT)
    cfg = MidogPPPosthocGMMPriorConfig(
        name=str(experiment.get("name", "")),
        artifact_root=_path(base, str(experiment.get("artifact_root", DEFAULT_ARTIFACT_ROOT))),
        manifest_path=_path(base, str(inputs.get("manifest_path", DEFAULT_MANIFEST_PATH))),
        feature_cache_path=_path(base, str(inputs.get("feature_cache_path", DEFAULT_FEATURE_CACHE_PATH))),
        signal_split_manifest_path=_path(base, str(inputs.get("signal_split_manifest_path", DEFAULT_SIGNAL_SPLIT_MANIFEST_PATH))),
        preservation_gate_artifact_root=_path(base, str(gate_root)),
        positive_label=int(data.get("positive_label", 1)),
        controls=tuple(str(v) for v in run.get("controls", (BALANCED_CONTROL,))),
        primary_variant=_parse_variant(variant_payload),
        split_seeds=None
        if run.get("split_seeds") in (None, "", "all")
        else tuple(int(v) for v in run.get("split_seeds", ())),
        allow_npz_test_cache=bool(inputs.get("allow_npz_test_cache", False)),
        min_fit=int(thresholds.get("min_fit", 20)),
        min_eval=int(thresholds.get("min_eval", 10)),
        min_fit_pos=int(thresholds.get("min_fit_pos", 10)),
        min_fit_neg=int(thresholds.get("min_fit_neg", 10)),
        min_eval_pos=int(thresholds.get("min_eval_pos", 5)),
        min_eval_neg=int(thresholds.get("min_eval_neg", 5)),
        min_fit_cases=int(thresholds.get("min_fit_cases", 3)),
        min_eval_cases=int(thresholds.get("min_eval_cases", 2)),
        primary_method=str(run.get("primary_method", PRIMARY_METHOD)),
        class_prior_policy=str(run.get("class_prior_policy", "balanced")),
        synthetic_budget=None if run.get("synthetic_budget") in (None, "", "match_fit") else int(run.get("synthetic_budget", 0)),
        gmm_components=int(gmm.get("components", 2)),
        gmm_covariance_type=str(gmm.get("covariance_type", "diag")),
        gmm_reg_covar=float(gmm.get("reg_covar", 1.0e-4)),
        gmm_n_init=int(gmm.get("n_init", 1)),
        gmm_max_iter=int(gmm.get("max_iter", 100)),
        min_per_class_n=int(gmm.get("min_per_class_n", 4)),
        min_samples_per_component=int(gmm.get("min_samples_per_component", 2)),
        fallback_to_single_gaussian=bool(gmm.get("fallback_to_single_gaussian", True)),
        no_rejection=bool(gmm.get("no_rejection", True)),
        generation_seeds=tuple(int(v) for v in downstream.get("generation_seeds", (13,))),
        classifier_seeds=tuple(int(v) for v in downstream.get("classifier_seeds", (17,))),
        bootstrap_reps=int(bootstrap.get("reps", 1000)),
        bootstrap_seed=int(bootstrap.get("seed", 1337)),
        real_gate_min_bacc=float(decisions.get("real_gate_min_bacc", decisions.get("gate_min_bacc", 0.60))),
        ci_low_threshold=float(decisions.get("ci_low_threshold", 0.50)),
        gate_required_label=str(run.get("gate_required_label", GATE_REQUIRED_LABEL)),
        allow_condition_warning=bool(run.get("allow_condition_warning", True)),
    )
    validate_midogpp_posthoc_gmm_prior_config(cfg)
    return cfg


def validate_midogpp_posthoc_gmm_prior_config(cfg: MidogPPPosthocGMMPriorConfig) -> None:
    if cfg.name != EXPERIMENT_NAME:
        raise ProtocolError(f"Unexpected experiment.name={cfg.name!r}; expected {EXPERIMENT_NAME!r}.")
    unknown = sorted(set(cfg.controls) - set(CONTROL_ORDER))
    if unknown:
        raise ProtocolError(f"Unknown controls: {unknown}")
    if cfg.primary_variant.variant_id != PRIMARY_VARIANT:
        raise ProtocolError(f"primary_variant.variant_id must remain {PRIMARY_VARIANT!r}.")
    if (cfg.primary_variant.pca_dim, cfg.primary_variant.latent_dim) != (128, 32):
        raise ProtocolError("pca128 post-hoc GMM audit requires pca_dim=128 and latent_dim=32.")
    if cfg.primary_variant.hidden_dim != 512 or cfg.primary_variant.num_hidden_layers != 2:
        raise ProtocolError("CVAE architecture is locked to hidden_dim=512 and two hidden layers.")
    if not math.isclose(cfg.primary_variant.beta_final, 0.001):
        raise ProtocolError("CVAE beta_final is locked to 0.001.")
    if cfg.primary_method != PRIMARY_METHOD:
        raise ProtocolError(f"primary_method must remain {PRIMARY_METHOD!r}.")
    if cfg.class_prior_policy not in {"balanced", "fit_derived"}:
        raise ProtocolError("class_prior_policy must be 'balanced' or 'fit_derived'.")
    if cfg.gmm_covariance_type != "diag":
        raise ProtocolError("Only diag covariance is locked for this feasibility audit.")
    if cfg.gmm_components < 1:
        raise ProtocolError("gmm_prior.components must be >= 1.")
    if cfg.min_samples_per_component < 1:
        raise ProtocolError("gmm_prior.min_samples_per_component must be >= 1.")
    if not cfg.fallback_to_single_gaussian:
        raise ProtocolError("fallback_to_single_gaussian must remain true.")
    if not cfg.no_rejection:
        raise ProtocolError("no_rejection must remain true for the locked first audit.")
    if not cfg.generation_seeds or not cfg.classifier_seeds:
        raise ProtocolError("generation_seeds and classifier_seeds must be non-empty and locked.")


def run_midogpp_posthoc_gmm_prior(
    cfg: MidogPPPosthocGMMPriorConfig,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    import numpy as np  # type: ignore

    gate = _load_and_validate_gate(cfg)
    root = prepare_artifact_dirs(Path(artifact_root) if artifact_root is not None else cfg.artifact_root)
    manifest_rows = _read_train_manifest(cfg.manifest_path, positive_label=cfg.positive_label)
    cache = _load_feature_cache(cfg.feature_cache_path)
    _assert_virchow2_cache(cache, path=cfg.feature_cache_path, allow_npz_test_cache=cfg.allow_npz_test_cache)
    embeddings = np.asarray(_to_numpy(cache.embeddings), dtype=float)
    _assert_cache_alignment(manifest_rows, cache)
    split_specs = _read_split_manifest(cfg.signal_split_manifest_path, manifest_rows, controls=cfg.controls)
    if cfg.split_seeds is not None:
        allowed = set(int(seed) for seed in cfg.split_seeds)
        split_specs = tuple(spec for spec in split_specs if int(spec.split_seed) in allowed)
    if not split_specs:
        raise ProtocolError("No matching MIDOG++ signal-control splits were found.")
    _assert_gate_provenance(cfg, gate, manifest_rows, split_specs)

    metric_rows: list[dict[str, object]] = []
    negative_rows: list[dict[str, object]] = []
    gmm_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    model_manifest_rows: list[dict[str, object]] = []

    for spec in split_specs:
        audit = _identity_audit_rows(manifest_rows, spec)
        audit_rows.extend(audit)
        status, error = _split_status(cfg, manifest_rows, spec, audit)  # type: ignore[arg-type]
        counts = _split_counts(manifest_rows, spec.fit_idx, spec.eval_idx)
        if status != VALID_STATUS:
            metric_rows.append(
                _metric_failure_row(spec, counts, PRIMARY_METHOD, status=status, error_message=error)
            )
            continue

        x_fit_raw = embeddings[list(spec.fit_idx)]
        y_fit = np.asarray([manifest_rows[idx].label for idx in spec.fit_idx], dtype=int)
        x_eval_raw = embeddings[list(spec.eval_idx)]
        y_eval = np.asarray([manifest_rows[idx].label for idx in spec.eval_idx], dtype=int)
        runtime = _train_runtime(
            cfg,  # type: ignore[arg-type]
            spec,
            cfg.primary_variant,
            x_fit_raw=x_fit_raw,
            x_eval_raw=x_eval_raw,
            y_fit=y_fit,
            condition_mode="real_labels",
        )

        real_row, real_preds = _evaluate_representation(
            cfg,  # type: ignore[arg-type]
            spec,
            rows=manifest_rows,
            x_fit=runtime.fit_x,
            y_fit=y_fit,
            x_eval=runtime.eval_x,
            y_eval=y_eval,
            variant_id=PRIMARY_VARIANT,
            representation_role=REAL_PCA128_REFERENCE,
            counts=counts,
            prediction_prefix="",
        )
        metric_rows.append(_posthoc_metric_row(real_row, REAL_PCA128_REFERENCE, "real_reference", diagnostic_only=True))
        prediction_rows.extend(_posthoc_prediction_rows(real_preds, "real_reference"))

        decoded, _ = _decode_mu(runtime, runtime.fit_x, y_fit)
        decode_row, decode_preds = _evaluate_representation(
            cfg,  # type: ignore[arg-type]
            spec,
            rows=manifest_rows,
            x_fit=decoded,
            y_fit=y_fit,
            x_eval=runtime.eval_x,
            y_eval=y_eval,
            variant_id=PRIMARY_VARIANT,
            representation_role=DECODE_MU_DIAGNOSTIC,
            counts=counts,
            prediction_prefix="",
        )
        metric_rows.append(_posthoc_metric_row(decode_row, DECODE_MU_DIAGNOSTIC, "reconstruction_diagnostic", diagnostic_only=True))
        prediction_rows.extend(_posthoc_prediction_rows(decode_preds, "reconstruction_diagnostic"))

        posterior = _posterior_sample(runtime, runtime.fit_x, y_fit, seed=_stable_seed(spec, PRIMARY_VARIANT, "posthoc_posterior"))
        posterior_row, posterior_preds = _evaluate_representation(
            cfg,  # type: ignore[arg-type]
            spec,
            rows=manifest_rows,
            x_fit=posterior,
            y_fit=y_fit,
            x_eval=runtime.eval_x,
            y_eval=y_eval,
            variant_id=PRIMARY_VARIANT,
            representation_role=POSTERIOR_SAMPLE_DIAGNOSTIC,
            counts=counts,
            prediction_prefix="",
        )
        metric_rows.append(_posthoc_metric_row(posterior_row, POSTERIOR_SAMPLE_DIAGNOSTIC, "posterior_diagnostic", diagnostic_only=True))
        prediction_rows.extend(_posthoc_prediction_rows(posterior_preds, "posterior_diagnostic"))

        mu = _posterior_mu(runtime, runtime.fit_x, y_fit)
        synthetic_labels = _synthetic_labels(cfg, y_fit)
        for generation_seed in cfg.generation_seeds:
            gaussian_x, gaussian_y, gaussian_diag = _sample_class_gaussian(
                cfg, spec, runtime, mu, y_fit, synthetic_labels, seed=int(generation_seed)
            )
            gmm_rows.extend(gaussian_diag)
            _append_synthetic_eval(
                cfg,
                spec,
                manifest_rows,
                runtime,
                gaussian_x,
                gaussian_y,
                y_eval,
                method_role=GAUSSIAN_METHOD,
                counts=counts,
                generation_seed=int(generation_seed),
                metric_rows=metric_rows,
                prediction_rows=prediction_rows,
                model_manifest_rows=model_manifest_rows,
                gate=gate,
            )

            gmm_x, gmm_y, diag = _sample_class_gmm(
                cfg, spec, runtime, mu, y_fit, synthetic_labels, seed=int(generation_seed)
            )
            gmm_rows.extend(diag)
            _append_synthetic_eval(
                cfg,
                spec,
                manifest_rows,
                runtime,
                gmm_x,
                gmm_y,
                y_eval,
                method_role=PRIMARY_METHOD,
                counts=counts,
                generation_seed=int(generation_seed),
                metric_rows=metric_rows,
                prediction_rows=prediction_rows,
                model_manifest_rows=model_manifest_rows,
                gate=gate,
            )

            global_x, global_y, global_diag = _sample_global_gmm(
                cfg, spec, runtime, mu, synthetic_labels, seed=int(generation_seed)
            )
            gmm_rows.extend(global_diag)
            _append_synthetic_eval(
                cfg,
                spec,
                manifest_rows,
                runtime,
                global_x,
                global_y,
                y_eval,
                method_role=GLOBAL_GMM_METHOD,
                counts=counts,
                generation_seed=int(generation_seed),
                metric_rows=negative_rows,
                prediction_rows=prediction_rows,
                model_manifest_rows=model_manifest_rows,
                gate=gate,
                diagnostic_only=True,
            )

            permuted_y = np.asarray(y_fit, dtype=int)[_rng(_stable_seed(spec, generation_seed, "label_permute")).permutation(len(y_fit))]
            perm_x, perm_y, perm_diag = _sample_class_gmm(
                cfg, spec, runtime, mu, permuted_y, synthetic_labels, seed=_stable_seed(spec, generation_seed, "permuted_gmm")
            )
            gmm_rows.extend(_with_status(perm_diag, LABEL_PERMUTED_GMM_CONTROL))
            _append_synthetic_eval(
                cfg,
                spec,
                manifest_rows,
                runtime,
                perm_x,
                perm_y,
                y_eval,
                method_role=LABEL_PERMUTED_GMM_CONTROL,
                counts=counts,
                generation_seed=int(generation_seed),
                metric_rows=negative_rows,
                prediction_rows=prediction_rows,
                model_manifest_rows=model_manifest_rows,
                gate=gate,
                diagnostic_only=True,
            )

            shuffled_labels = np.asarray(synthetic_labels, dtype=int)[
                _rng(_stable_seed(spec, generation_seed, "class_prior_shuffle")).permutation(len(synthetic_labels))
            ]
            _append_synthetic_eval(
                cfg,
                spec,
                manifest_rows,
                runtime,
                gmm_x,
                shuffled_labels,
                y_eval,
                method_role=CLASS_PRIOR_SHUFFLED_CONTROL,
                counts=counts,
                generation_seed=int(generation_seed),
                metric_rows=negative_rows,
                prediction_rows=prediction_rows,
                model_manifest_rows=model_manifest_rows,
                gate=gate,
                diagnostic_only=True,
            )

            random_x, random_y = _sample_random_latent(runtime, synthetic_labels, seed=_stable_seed(spec, generation_seed, "random_latent"))
            _append_synthetic_eval(
                cfg,
                spec,
                manifest_rows,
                runtime,
                random_x,
                random_y,
                y_eval,
                method_role=RANDOM_LATENT_CONTROL,
                counts=counts,
                generation_seed=int(generation_seed),
                metric_rows=negative_rows,
                prediction_rows=prediction_rows,
                model_manifest_rows=model_manifest_rows,
                gate=gate,
                diagnostic_only=True,
            )

    metric_all = metric_rows + _aggregate_posthoc_rows(metric_rows)
    negative_all = negative_rows + _aggregate_posthoc_rows(negative_rows)
    _attach_preservation_ratios(metric_all + negative_all)
    leakage = _leakage_report(audit_rows)
    decision_labels = _decision_labels(metric_all, negative_all, leakage)

    write_csv_rows(root / "tables" / "posthoc_gmm_prior_metrics.csv", metric_all, METRIC_COLUMNS)
    write_csv_rows(root / "tables" / "gmm_parameter_diagnostics.csv", gmm_rows, GMM_DIAGNOSTIC_COLUMNS)
    write_csv_rows(root / "tables" / "negative_control_metrics.csv", negative_all, METRIC_COLUMNS)
    write_csv_rows(root / "tables" / "identity_overlap_audit.csv", audit_rows, AUDIT_COLUMNS)
    write_csv_rows(root / "tables" / "predictions.csv", prediction_rows, PREDICTION_COLUMNS)
    write_csv_rows(root / "manifests" / "model_manifest.csv", model_manifest_rows, MODEL_MANIFEST_COLUMNS)
    write_json(root / "manifests" / "protocol_manifest.json", _protocol_manifest(cfg, cache, manifest_rows, split_specs, gate))
    write_json(root / "reports" / "leakage_report.json", leakage)
    _write_decision_report(root / "reports" / "decision_report.md", cfg, decision_labels, leakage, metric_all, negative_all, gate)
    return root


def _load_and_validate_gate(cfg: MidogPPPosthocGMMPriorConfig) -> dict[str, object]:
    root = cfg.preservation_gate_artifact_root
    decision_path = root / "reports" / "decision_report.md"
    leakage_path = root / "reports" / "leakage_report.json"
    manifest_path = root / "manifests" / "protocol_manifest.json"
    for path in (decision_path, leakage_path, manifest_path):
        if not path.exists():
            raise ProtocolError(f"preservation_gate_missing_required_artifact: {path}")
    decision_text = decision_path.read_text(encoding="utf-8")
    labels = _parse_decision_labels(decision_text)
    if cfg.gate_required_label not in labels:
        raise ProtocolError(f"preservation_gate_missing_{cfg.gate_required_label}")
    if CONDITION_WARNING_LABEL in labels and not cfg.allow_condition_warning:
        raise ProtocolError(f"preservation_gate_condition_warning_present: {CONDITION_WARNING_LABEL}")
    leakage = json.loads(leakage_path.read_text(encoding="utf-8"))
    if leakage.get("status") != "PASS":
        raise ProtocolError(f"preservation_gate_leakage_not_pass: {leakage.get('status')}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "root": str(root),
        "decision_labels": labels,
        "condition_warning_present": CONDITION_WARNING_LABEL in labels,
        "leakage": leakage,
        "manifest": manifest,
        "artifact_hash": _hash_paths((decision_path, leakage_path, manifest_path)),
    }


def _assert_gate_provenance(
    cfg: MidogPPPosthocGMMPriorConfig,
    gate: Mapping[str, object],
    rows: Sequence[ManifestRow],
    splits: Sequence[SplitSpec],
) -> None:
    manifest = gate.get("manifest")
    if not isinstance(manifest, Mapping):
        raise ProtocolError("preservation_gate_protocol_manifest_invalid")
    checks = {
        "manifest_path": str(cfg.manifest_path),
        "feature_cache_path": str(cfg.feature_cache_path),
        "signal_split_manifest_path": str(cfg.signal_split_manifest_path),
    }
    for key, expected in checks.items():
        actual = str(manifest.get(key, ""))
        if actual != expected:
            raise ProtocolError(f"preservation_gate_provenance_mismatch: {key} gate={actual!r} config={expected!r}")
    if int(manifest.get("train_rows", -1)) != len(rows):
        raise ProtocolError("preservation_gate_train_row_count_mismatch")
    if int(manifest.get("split_count", -1)) < len(splits):
        raise ProtocolError("preservation_gate_split_count_mismatch")


def _parse_decision_labels(text: str) -> tuple[str, ...]:
    for line in text.splitlines():
        if "Decision labels:" not in line:
            continue
        raw = line.split("Decision labels:", 1)[1].strip().strip("`").strip()
        if raw.startswith("`") and raw.endswith("`"):
            raw = raw[1:-1]
        return tuple(part.strip(" `") for part in raw.split(",") if part.strip(" `"))
    return ()


def _posterior_mu(runtime: CVAERuntime, x: object, labels: Sequence[int]) -> Any:
    import numpy as np  # type: ignore
    import torch  # type: ignore

    with torch.no_grad():
        xt = torch.as_tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32)
        yt = torch.as_tensor(np.asarray(labels, dtype=np.int64), dtype=torch.long)
        mu, _ = runtime.model.encode(xt, yt)
    return mu.detach().cpu().numpy()


def _synthetic_labels(cfg: MidogPPPosthocGMMPriorConfig, y_fit: Sequence[int]) -> tuple[int, ...]:
    n_total = len(y_fit) if cfg.synthetic_budget is None else int(cfg.synthetic_budget)
    if n_total < 2:
        raise ProtocolError("synthetic_budget must allow both binary classes.")
    if cfg.class_prior_policy == "balanced":
        n_pos = n_total // 2
    else:
        n_pos = int(round(n_total * (sum(1 for v in y_fit if int(v) == 1) / float(len(y_fit)))))
        n_pos = min(max(n_pos, 1), n_total - 1)
    labels = [1] * n_pos + [0] * (n_total - n_pos)
    return tuple(sorted(labels))


def _sample_class_gaussian(
    cfg: MidogPPPosthocGMMPriorConfig,
    spec: SplitSpec,
    runtime: CVAERuntime,
    mu: object,
    y_fit: Sequence[int],
    synthetic_labels: Sequence[int],
    *,
    seed: int,
) -> tuple[Any, tuple[int, ...], list[dict[str, object]]]:
    import numpy as np  # type: ignore

    rng = _rng(seed)
    mu_np = np.asarray(mu, dtype=float)
    y_np = np.asarray(y_fit, dtype=int)
    samples = []
    labels = []
    diagnostics = []
    for cls in (0, 1):
        class_mu = mu_np[y_np == cls]
        _assert_min_class_n(cfg, cls, len(class_mu))
        mean = class_mu.mean(axis=0)
        var = np.var(class_mu, axis=0) + float(cfg.gmm_reg_covar)
        n = sum(1 for label in synthetic_labels if int(label) == cls)
        cls_samples = rng.normal(loc=mean, scale=np.sqrt(var), size=(n, mu_np.shape[1]))
        samples.append(cls_samples)
        labels.extend([cls] * n)
        diagnostics.append(_diag_row(spec, PRIMARY_VARIANT, GAUSSIAN_METHOD, cls, len(class_mu), 1, var, [1.0], True, 0, False))
    z = np.vstack(samples) if samples else np.zeros((0, runtime.model.latent_dim))
    return _decode_latents(runtime, z, labels), tuple(labels), diagnostics


def _sample_class_gmm(
    cfg: MidogPPPosthocGMMPriorConfig,
    spec: SplitSpec,
    runtime: CVAERuntime,
    mu: object,
    y_fit: Sequence[int],
    synthetic_labels: Sequence[int],
    *,
    seed: int,
) -> tuple[Any, tuple[int, ...], list[dict[str, object]]]:
    import numpy as np  # type: ignore
    from sklearn.mixture import GaussianMixture  # type: ignore

    rng = _rng(seed)
    mu_np = np.asarray(mu, dtype=float)
    y_np = np.asarray(y_fit, dtype=int)
    samples = []
    labels = []
    diagnostics = []
    for cls in (0, 1):
        class_mu = mu_np[y_np == cls]
        _assert_min_class_n(cfg, cls, len(class_mu))
        requested_k = min(int(cfg.gmm_components), len(class_mu) // int(cfg.min_samples_per_component))
        k = max(1, requested_k)
        fallback = k < int(cfg.gmm_components)
        n = sum(1 for label in synthetic_labels if int(label) == cls)
        if k == 1:
            mean = class_mu.mean(axis=0)
            var = np.var(class_mu, axis=0) + float(cfg.gmm_reg_covar)
            cls_samples = rng.normal(loc=mean, scale=np.sqrt(var), size=(n, mu_np.shape[1]))
            diagnostics.append(_diag_row(spec, PRIMARY_VARIANT, PRIMARY_METHOD, cls, len(class_mu), 1, var, [1.0], True, 0, fallback))
        else:
            model = GaussianMixture(
                n_components=k,
                covariance_type=cfg.gmm_covariance_type,
                reg_covar=float(cfg.gmm_reg_covar),
                n_init=int(cfg.gmm_n_init),
                max_iter=int(cfg.gmm_max_iter),
                random_state=int(seed) + cls,
            )
            model.fit(class_mu)
            cls_samples, _ = model.sample(n)
            diagnostics.append(
                _diag_row(
                    spec,
                    PRIMARY_VARIANT,
                    PRIMARY_METHOD,
                    cls,
                    len(class_mu),
                    k,
                    model.covariances_,
                    model.weights_,
                    bool(model.converged_),
                    int(model.n_iter_),
                    fallback,
                )
            )
        samples.append(cls_samples)
        labels.extend([cls] * n)
    z = np.vstack(samples) if samples else np.zeros((0, runtime.model.latent_dim))
    return _decode_latents(runtime, z, labels), tuple(labels), diagnostics


def _sample_global_gmm(
    cfg: MidogPPPosthocGMMPriorConfig,
    spec: SplitSpec,
    runtime: CVAERuntime,
    mu: object,
    synthetic_labels: Sequence[int],
    *,
    seed: int,
) -> tuple[Any, tuple[int, ...], list[dict[str, object]]]:
    import numpy as np  # type: ignore
    from sklearn.mixture import GaussianMixture  # type: ignore

    mu_np = np.asarray(mu, dtype=float)
    k = max(1, min(int(cfg.gmm_components), len(mu_np) // int(cfg.min_samples_per_component)))
    if k == 1:
        rng = _rng(seed)
        var = np.var(mu_np, axis=0) + float(cfg.gmm_reg_covar)
        z = rng.normal(loc=mu_np.mean(axis=0), scale=np.sqrt(var), size=(len(synthetic_labels), mu_np.shape[1]))
        diag = [_diag_row(spec, PRIMARY_VARIANT, GLOBAL_GMM_METHOD, -1, len(mu_np), 1, var, [1.0], True, 0, True)]
    else:
        model = GaussianMixture(
            n_components=k,
            covariance_type=cfg.gmm_covariance_type,
            reg_covar=float(cfg.gmm_reg_covar),
            n_init=int(cfg.gmm_n_init),
            max_iter=int(cfg.gmm_max_iter),
            random_state=int(seed),
        )
        model.fit(mu_np)
        z, _ = model.sample(len(synthetic_labels))
        diag = [_diag_row(spec, PRIMARY_VARIANT, GLOBAL_GMM_METHOD, -1, len(mu_np), k, model.covariances_, model.weights_, bool(model.converged_), int(model.n_iter_), k < cfg.gmm_components)]
    return _decode_latents(runtime, z, synthetic_labels), tuple(int(v) for v in synthetic_labels), diag


def _sample_random_latent(runtime: CVAERuntime, synthetic_labels: Sequence[int], *, seed: int) -> tuple[Any, tuple[int, ...]]:
    import numpy as np  # type: ignore

    z = _rng(seed).normal(size=(len(synthetic_labels), runtime.model.latent_dim))
    return _decode_latents(runtime, z, synthetic_labels), tuple(int(v) for v in synthetic_labels)


def _decode_latents(runtime: CVAERuntime, z: object, labels: Sequence[int]) -> Any:
    import numpy as np  # type: ignore
    import torch  # type: ignore

    z_np = np.asarray(z, dtype=np.float32)
    y_np = np.asarray(labels, dtype=np.int64)
    with torch.no_grad():
        zt = torch.as_tensor(z_np, dtype=torch.float32)
        yt = torch.as_tensor(y_np, dtype=torch.long)
        return runtime.model.decode(zt, yt).detach().cpu().numpy()


def _append_synthetic_eval(
    cfg: MidogPPPosthocGMMPriorConfig,
    spec: SplitSpec,
    rows: Sequence[ManifestRow],
    runtime: CVAERuntime,
    synthetic_x: object,
    synthetic_y: Sequence[int],
    y_eval: Sequence[int],
    *,
    method_role: str,
    counts: Mapping[str, int],
    generation_seed: int,
    metric_rows: list[dict[str, object]],
    prediction_rows: list[dict[str, object]],
    model_manifest_rows: list[dict[str, object]],
    gate: Mapping[str, object],
    diagnostic_only: bool = False,
) -> None:
    import numpy as np  # type: ignore

    x_syn = np.asarray(synthetic_x, dtype=float)
    y_syn = np.asarray(synthetic_y, dtype=int)
    if x_syn.ndim != 2 or x_syn.shape[1] != runtime.eval_x.shape[1] or not np.all(np.isfinite(x_syn)):
        raise ProtocolError(f"generated_embedding_shape_or_nan_failed: {method_role}")
    for classifier_seed in cfg.classifier_seeds:
        bundle = fit_locked_logistic_classifier(
            x_syn,
            y_syn,
            runtime.eval_x,
            classifier_seed=int(classifier_seed),
            expert_id=method_role,
            class_weight="balanced",
        )
        result = evaluate_probability_predictions(method_role, bundle.probabilities, y_eval, classes=bundle.classes)
        preds = [1 if row[1] >= 0.5 else 0 for row in bundle.probabilities]
        metric_rows.append(
            _metric_row_from_result(
                spec,
                counts,
                method_role,
                result.bacc,
                result.macro_f1,
                y_eval,
                preds,
                n_generated=len(y_syn),
                generated_pos=int(np.sum(y_syn == 1)),
                generated_neg=int(np.sum(y_syn == 0)),
                diagnostic_only=diagnostic_only,
            )
        )
        for pos, idx in enumerate(spec.eval_idx):
            row = rows[idx]
            prediction_rows.append(
                {
                    "control_name": spec.control_name,
                    "domain_name": spec.domain_name,
                    "split_seed": int(spec.split_seed),
                    "variant_id": PRIMARY_VARIANT,
                    "representation_role": method_role,
                    "method_role": method_role,
                    "sample_id": row.sample_id,
                    "case_id": row.case_id,
                    "feature_row_index": int(row.feature_row_index),
                    "y_true": int(y_eval[pos]),
                    "y_pred": int(preds[pos]),
                    "prob_pos": float(bundle.probabilities[pos][1]),
                }
            )
        model_manifest_rows.append(
            {
                "control_name": spec.control_name,
                "domain_name": spec.domain_name,
                "split_seed": int(spec.split_seed),
                "variant_id": PRIMARY_VARIANT,
                "method_role": method_role,
                "pca_dim": int(runtime.fit_x.shape[1]),
                "latent_dim": int(runtime.model.latent_dim),
                "posterior_source": "posterior_mu_fit_rows_only",
                "class_prior_policy": cfg.class_prior_policy,
                "synthetic_budget": int(len(y_syn)),
                "generation_seed": int(generation_seed),
                "classifier_seed": int(classifier_seed),
                "fit_row_hash": _row_hash(rows, spec.fit_idx),
                "eval_row_hash": _row_hash(rows, spec.eval_idx),
                "generated_feature_hash": _array_hash(x_syn),
                "prediction_hash": _hash_text(json.dumps(bundle.probabilities)),
                "preservation_gate_artifact_root": gate.get("root", ""),
                "preservation_gate_hash": gate.get("artifact_hash", ""),
            }
        )


def _metric_row_from_result(
    spec: SplitSpec,
    counts: Mapping[str, int],
    method_role: str,
    bacc: float,
    mf1: float,
    y_eval: Sequence[int],
    preds: Sequence[int],
    *,
    n_generated: int,
    generated_pos: int,
    generated_neg: int,
    diagnostic_only: bool,
) -> dict[str, object]:
    precision, recall, f1 = _positive_metrics(y_eval, preds)
    return {
        "aggregation_level": "seed",
        "control_name": spec.control_name,
        "domain_name": spec.domain_name,
        "split_seed": int(spec.split_seed),
        "variant_id": PRIMARY_VARIANT,
        "representation_role": method_role,
        "method_role": method_role,
        "adoption_eligible": _bool_text((not diagnostic_only) and method_role == PRIMARY_METHOD),
        "diagnostic_only": _bool_text(diagnostic_only or method_role != PRIMARY_METHOD),
        "model_type": "locked_logistic_regression",
        "eligible_seed_count": "",
        "valid_seed_count": "",
        "decision_status": "",
        **dict(counts),
        "n_generated": int(n_generated),
        "generated_pos": int(generated_pos),
        "generated_neg": int(generated_neg),
        "bacc": float(bacc),
        "bacc_std": math.nan,
        "bacc_min": math.nan,
        "macro_f1": float(mf1),
        "precision_pos": precision,
        "recall_pos": recall,
        "f1_pos": f1,
        "support_pos": sum(1 for v in y_eval if int(v) == 1),
        "support_neg": sum(1 for v in y_eval if int(v) == 0),
        "ci_low": math.nan,
        "ci_high": math.nan,
        "ci_method": "seed_rows",
        "above_chance": _bool_text(float(bacc) > 0.5),
        "near_chance": _bool_text(float(bacc) <= 0.55),
        "converged": "true",
        "status": VALID_STATUS,
        "error_message": "",
        "preservation_ratio_vs_real_pca128": math.nan,
    }


def _posthoc_metric_row(row: Mapping[str, object], representation_role: str, method_role: str, *, diagnostic_only: bool) -> dict[str, object]:
    out = {key: row.get(key, "") for key in METRIC_COLUMNS}
    out.update(row)
    out.update(
        {
            "representation_role": representation_role,
            "method_role": method_role,
            "adoption_eligible": "false",
            "diagnostic_only": _bool_text(diagnostic_only),
            "n_generated": "",
            "generated_pos": "",
            "generated_neg": "",
            "preservation_ratio_vs_real_pca128": math.nan,
        }
    )
    return out


def _posthoc_prediction_rows(rows: Sequence[Mapping[str, object]], method_role: str) -> list[dict[str, object]]:
    out = []
    for row in rows:
        item = {key: row.get(key, "") for key in PREDICTION_COLUMNS}
        item.update(row)
        item["method_role"] = method_role
        out.append(item)
    return out


def _metric_failure_row(spec: SplitSpec, counts: Mapping[str, int], method_role: str, *, status: str, error_message: str) -> dict[str, object]:
    row = _empty_metric_row(spec, variant_id=PRIMARY_VARIANT, representation_role=method_role, counts=counts, status=status, error_message=error_message)
    return _posthoc_metric_row(row, method_role, method_role, diagnostic_only=False)


def _aggregate_posthoc_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, object, object, object], list[Mapping[str, object]]] = {}
    for row in rows:
        if row.get("aggregation_level") != "seed" or row.get("status") != VALID_STATUS:
            continue
        key = (row.get("control_name"), row.get("domain_name"), row.get("variant_id"), row.get("method_role"))
        grouped.setdefault(key, []).append(row)
    out = []
    for (control, domain, variant, method), values in grouped.items():
        base = dict(values[0])
        baccs = [float(row.get("bacc")) for row in values if _finite(row.get("bacc"))]
        base.update(
            {
                "aggregation_level": "summary",
                "control_name": control,
                "domain_name": domain,
                "variant_id": variant,
                "representation_role": method,
                "method_role": method,
                "eligible_seed_count": len(values),
                "valid_seed_count": len(baccs),
                "split_seed": "",
                "bacc": _mean(baccs),
                "bacc_std": _std(baccs),
                "bacc_min": min(baccs) if baccs else math.nan,
                "macro_f1": _mean([float(row.get("macro_f1")) for row in values if _finite(row.get("macro_f1"))]),
                "status": VALID_STATUS if baccs else "no_valid_seeds",
            }
        )
        out.append(base)
    return out


def _attach_preservation_ratios(rows: Sequence[dict[str, object]]) -> None:
    refs = {
        (row.get("control_name"), row.get("domain_name"), row.get("aggregation_level")): row
        for row in rows
        if row.get("method_role") == "real_reference"
    }
    for row in rows:
        ref = refs.get((row.get("control_name"), row.get("domain_name"), row.get("aggregation_level")))
        if ref is None or row.get("method_role") == "real_reference":
            continue
        row["preservation_ratio_vs_real_pca128"] = _chance_corrected_ratio(row.get("bacc"), ref.get("bacc"))


def _decision_labels(metrics: Sequence[Mapping[str, object]], negatives: Sequence[Mapping[str, object]], leakage: Mapping[str, object]) -> list[str]:
    if leakage.get("status") != "PASS":
        return ["LEAKAGE_OR_ALIGNMENT_FAILURE_SUSPECT", "POSTHOC_GMM_PRIOR_NOT_ADOPTION_ELIGIBLE"]
    primary = [row for row in metrics if row.get("aggregation_level") == "summary" and row.get("method_role") == PRIMARY_METHOD and row.get("status") == VALID_STATUS]
    controls = [row for row in negatives if row.get("aggregation_level") == "summary" and row.get("status") == VALID_STATUS]
    labels = ["POSTHOC_GMM_ON_POSTERIOR_MU_FEASIBILITY_AUDIT"]
    if primary:
        labels.append("POSTHOC_GMM_PRIOR_UTILITY_REPORTED")
    else:
        labels.append("POSTHOC_GMM_PRIOR_INSUFFICIENT_VALID_SPLITS")
    if controls:
        labels.append("NEGATIVE_CONTROLS_REPORTED")
    labels.append("NO_ROUTING_OR_MOERGING_CLAIM")
    labels.append("NO_CONTROLLABLE_CLASS_CONDITIONAL_GENERATION_CLAIM")
    return labels


def _leakage_report(audit_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    overlap_failures = [row for row in audit_rows if int(row.get("overlap_count", 0)) > 0]
    report = build_leakage_report(
        target_support_labels_for_selection=False,
        target_eval_labels_for_scoring_only=True,
        target_expert_excluded=True,
        oracle_rows_diagnostic_only=True,
        extra_violations=["identity_overlap_failure"] if overlap_failures else (),
    ).to_json_dict()
    report.update(
        {
            "identity_overlap_failure_count": len(overlap_failures),
            "posterior_mu_fit_scope": "fit_rows_only",
            "gmm_fit_scope": "fit_rows_only",
            "class_prior_source": "locked_config_not_eval_distribution",
            "eval_labels_scoring_only": True,
            "routing_or_moerging_claim": False,
        }
    )
    return report


def _protocol_manifest(
    cfg: MidogPPPosthocGMMPriorConfig,
    cache: Any,
    rows: Sequence[ManifestRow],
    splits: Sequence[SplitSpec],
    gate: Mapping[str, object],
) -> dict[str, object]:
    feature_dim = int(getattr(cache.embeddings, "shape", [0, 0])[1])
    if feature_dim == 0:
        feature_dim = int(_to_numpy(cache.embeddings).shape[1])
    return {
        "schema_version": "midogpp_pca128_posthoc_gmm_on_posterior_mu_protocol_v1",
        "experiment_name": cfg.name,
        "manifest_path": str(cfg.manifest_path),
        "feature_cache_path": str(cfg.feature_cache_path),
        "signal_split_manifest_path": str(cfg.signal_split_manifest_path),
        "preservation_gate_artifact_root": str(cfg.preservation_gate_artifact_root),
        "preservation_gate_hash": gate.get("artifact_hash", ""),
        "preservation_gate_decision_labels": list(gate.get("decision_labels", ())),
        "train_rows": int(len(rows)),
        "split_count": int(len(splits)),
        "feature_dim": feature_dim,
        "feature_extractor": dict(cache.feature_extractor),
        "primary_method": cfg.primary_method,
        "posterior_source": "encoder_mu_fit_rows_only",
        "class_prior_policy": cfg.class_prior_policy,
        "synthetic_budget": "match_fit" if cfg.synthetic_budget is None else int(cfg.synthetic_budget),
        "generation_seeds": list(cfg.generation_seeds),
        "classifier_seeds": list(cfg.classifier_seeds),
        "gmm_components": int(cfg.gmm_components),
        "gmm_covariance_type": cfg.gmm_covariance_type,
        "gmm_reg_covar": float(cfg.gmm_reg_covar),
        "gmm_no_rejection": bool(cfg.no_rejection),
        "pca_scaler_fit_scope": "fit_rows_only",
        "cvae_training_scope": "fit_rows_only",
        "gmm_fit_scope": "fit_rows_only",
        "held_out_eval_role": "final_scoring_only",
        "claim_boundary": {
            "allowed": "Fit-only post-hoc latent sampler over pca128 encoder posterior means can recover useful synthetic PCA128 embeddings if metrics pass.",
            "forbidden": "No trained-prior, routing, MoErging, metadata-control, controllable class-conditional generation, or learned-prior-superiority claim.",
        },
    }


def _write_decision_report(
    path: Path,
    cfg: MidogPPPosthocGMMPriorConfig,
    labels: Sequence[str],
    leakage: Mapping[str, object],
    metrics: Sequence[Mapping[str, object]],
    negatives: Sequence[Mapping[str, object]],
    gate: Mapping[str, object],
) -> None:
    lines = [
        "# MIDOG++ pca128 Post-hoc GMM-on-Posterior-Mu Feasibility Audit",
        "",
        f"- Decision labels: `{', '.join(labels)}`",
        f"- Leakage status: `{leakage.get('status')}`",
        f"- Preservation gate labels: `{', '.join(str(v) for v in gate.get('decision_labels', ()))}`",
        "- Primary method: `gmm_on_posterior_mu_class_conditional_diag`.",
        "- This is a fitted latent-sampler feasibility audit only.",
        "- It is not routing, MoErging, metadata compatibility, learned-prior superiority, or controllable class-conditional generation evidence.",
        "",
        "## Summary Metrics",
        "",
        "| Method | BACC | Macro-F1 | Ratio vs real | Status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in metrics:
        if row.get("aggregation_level") == "summary":
            lines.append(
                f"| {row.get('method_role')} | {_fmt(row.get('bacc'))} | {_fmt(row.get('macro_f1'))} | "
                f"{_fmt(row.get('preservation_ratio_vs_real_pca128'))} | {row.get('status')} |"
            )
    lines.extend(["", "## Negative Controls", "", "| Method | BACC | Macro-F1 | Status |", "| --- | ---: | ---: | --- |"])
    for row in negatives:
        if row.get("aggregation_level") == "summary":
            lines.append(f"| {row.get('method_role')} | {_fmt(row.get('bacc'))} | {_fmt(row.get('macro_f1'))} | {row.get('status')} |")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- `decode_mu` and `posterior_sample` are diagnostic because they use encoder latents from real rows.",
            "- GMM rows model encoder posterior means, not the full aggregated posterior or the trained CVAE prior.",
            "- Held-out labels are final scoring only and must not select methods, budgets, seeds, priors, thresholds, or classifier settings.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _diag_row(
    spec: SplitSpec,
    variant_id: str,
    method_role: str,
    class_label: int,
    fit_n: int,
    k: int,
    covariance: object,
    weights: Sequence[float],
    converged: bool,
    n_iter: int,
    fallback: bool,
) -> dict[str, object]:
    import numpy as np  # type: ignore

    cov = np.asarray(covariance, dtype=float)
    weight_values = [float(value) for value in np.asarray(weights, dtype=float).reshape(-1).tolist()]
    if cov.ndim == 0:
        flat = cov.reshape(1)
    else:
        flat = cov.reshape(-1)
    finite = flat[np.isfinite(flat)]
    min_var = float(np.min(finite)) if finite.size else math.nan
    max_var = float(np.max(finite)) if finite.size else math.nan
    return {
        "control_name": spec.control_name,
        "domain_name": spec.domain_name,
        "split_seed": int(spec.split_seed),
        "variant_id": variant_id,
        "method_role": method_role,
        "class_label": int(class_label),
        "fit_n": int(fit_n),
        "selected_k": int(k),
        "covariance_type": "diag",
        "reg_covar": "",
        "converged": _bool_text(converged),
        "n_iter": int(n_iter),
        "effective_components": int(sum(1 for weight in weight_values if float(weight) > 0.01)),
        "min_component_weight": min(weight_values) if weight_values else math.nan,
        "max_component_weight": max(weight_values) if weight_values else math.nan,
        "min_variance": min_var,
        "max_variance": max_var,
        "condition_number": (max_var / min_var) if _finite(max_var) and _finite(min_var) and min_var > 0 else math.nan,
        "fallback_used": _bool_text(fallback),
        "status": VALID_STATUS if converged else "gmm_not_converged",
        "error_message": "",
    }


def _with_status(rows: Sequence[Mapping[str, object]], method_role: str) -> list[dict[str, object]]:
    out = []
    for row in rows:
        item = dict(row)
        item["method_role"] = method_role
        out.append(item)
    return out


def _assert_min_class_n(cfg: MidogPPPosthocGMMPriorConfig, cls: int, n: int) -> None:
    if int(n) < int(cfg.min_per_class_n):
        raise ProtocolError(f"too_few_fit_rows_for_class_{cls}: n={n} minimum={cfg.min_per_class_n}")


def _positive_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> tuple[float, float, float]:
    tp = sum(1 for truth, pred in zip(y_true, y_pred) if int(truth) == 1 and int(pred) == 1)
    fp = sum(1 for truth, pred in zip(y_true, y_pred) if int(truth) == 0 and int(pred) == 1)
    fn = sum(1 for truth, pred in zip(y_true, y_pred) if int(truth) == 1 and int(pred) == 0)
    precision = float(tp) / float(tp + fp) if (tp + fp) else 0.0
    recall = float(tp) / float(tp + fn) if (tp + fn) else 0.0
    f1 = float(2 * precision * recall) / float(precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _rng(seed: int) -> Any:
    import numpy as np  # type: ignore

    return np.random.default_rng(int(seed) % (2**32))


def _mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if _finite(v)]
    return sum(vals) / len(vals) if vals else math.nan


def _std(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if _finite(v)]
    if len(vals) < 2:
        return 0.0 if vals else math.nan
    mean = _mean(vals)
    return math.sqrt(sum((value - mean) ** 2 for value in vals) / float(len(vals) - 1))


def _chance_corrected_ratio(value: object, reference: object) -> float:
    if not _finite(value) or not _finite(reference):
        return math.nan
    denom = float(reference) - 0.5
    if denom <= 0.0:
        return math.nan
    return (float(value) - 0.5) / denom


def _row_hash(rows: Sequence[ManifestRow], indices: Sequence[int]) -> str:
    payload = "|".join(f"{idx}:{rows[idx].sample_id}:{rows[idx].case_id}:{rows[idx].label}" for idx in indices)
    return _hash_text(payload)


def _array_hash(value: object) -> str:
    import numpy as np  # type: ignore

    arr = np.asarray(value, dtype=float)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash_paths(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()
