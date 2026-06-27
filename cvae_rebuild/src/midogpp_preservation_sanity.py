from __future__ import annotations

import csv
import hashlib
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from sklearn.exceptions import ConvergenceWarning  # type: ignore

from feature_frame import ExpertFeatureFrame, fit_expert_frame
from metrics import balanced_accuracy, macro_f1, nanmean
from models import ClassConditionedCVAE
from protocol import ProtocolError, build_leakage_report
from reporting import prepare_artifact_dirs, write_csv_rows, write_json


EXPERIMENT_NAME = "virchow2_cvae_midogpp_preservation_sanity_v1"
DEFAULT_ARTIFACT_ROOT = "cvae_rebuild/artifacts/midogpp/virchow2_cvae_midogpp_preservation_sanity_v1"
DEFAULT_MANIFEST_PATH = "datasets/midogpp/artifacts/midogpp_annotation_patch_v1/manifest.csv"
DEFAULT_FEATURE_CACHE_PATH = "sail/artifacts/pathology_embeddings/midogpp/virchow2/seed42/embeddings/train.pt"
DEFAULT_SIGNAL_SPLIT_MANIFEST_PATH = "sail/artifacts/midogpp_virchow2_real_feature_signal_controls/tables/split_manifest.csv"
DEFAULT_SIGNAL_DECISION_REPORT_PATH = "sail/artifacts/midogpp_virchow2_real_feature_signal_controls/reports/decision_report.md"

VALID_STATUS = "valid"
LOGISTIC_MODEL_TYPE = "logistic_regression"
PRIMARY_VARIANT = "pca64_beta001"
DIAGNOSTIC_VARIANT = "pca128_beta001"
RAW_VARIANT = "raw"

POOLED_CONTROL = "pooled_case_disjoint_control"
BALANCED_CONTROL = "pooled_tumor_class_balanced_case_split"
WITHIN_TUMOR_CONTROL = "within_tumor_case_disjoint_control"
CONTROL_ORDER = (POOLED_CONTROL, BALANCED_CONTROL, WITHIN_TUMOR_CONTROL)

REAL_RAW_REFERENCE = "real_raw_reference"
REAL_FRAME_REFERENCE = "real_frame_reference"
DECODE_MU = "decode_mu_fit_to_real_eval"
POSTERIOR_SAMPLE = "posterior_sample_fit_to_real_eval"
PRIOR_SAMPLE = "prior_sample_fit_to_real_eval"
REAL_LABEL_PERMUTATION = "real_frame_label_permutation_control"
REAL_FEATURE_ROW_SHUFFLE = "real_frame_feature_label_row_shuffle_control"
CVAE_CONDITION_LABEL_PERMUTATION = "cvae_condition_label_permutation"
CVAE_DECODED_ROW_SHUFFLE = "cvae_decoded_feature_label_row_shuffle_control"
NEGATIVE_ROLES = {
    REAL_LABEL_PERMUTATION,
    REAL_FEATURE_ROW_SHUFFLE,
    CVAE_CONDITION_LABEL_PERMUTATION,
    CVAE_DECODED_ROW_SHUFFLE,
}


PRESERVATION_COLUMNS = (
    "aggregation_level",
    "control_name",
    "domain_name",
    "split_seed",
    "variant_id",
    "representation_role",
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
    "preservation_ratio_vs_real_frame",
)

GAP_COLUMNS = (
    "control_name",
    "domain_name",
    "variant_id",
    "representation_role",
    "real_raw_bacc",
    "real_frame_bacc",
    "variant_bacc",
    "pca_frame_preservation_ratio",
    "preservation_ratio_vs_real_frame",
    "bacc_gap_vs_real_frame",
    "status",
)

SUITABILITY_COLUMNS = (
    "criterion",
    "value",
    "threshold",
    "passed",
    "evidence",
)

RECON_COLUMNS = (
    "control_name",
    "domain_name",
    "split_seed",
    "variant_id",
    "representation_role",
    "n_rows",
    "recon_mse_per_dim",
    "recon_rmse_per_dim",
    "kl_per_latent_dim",
    "mu_norm_mean",
    "logvar_mean",
    "status",
)

TRAINING_COLUMNS = (
    "control_name",
    "domain_name",
    "split_seed",
    "variant_id",
    "condition_mode",
    "epoch",
    "beta",
    "train_recon_mse_per_dim",
    "train_kl_per_latent_dim",
    "train_total_loss",
)

AUDIT_COLUMNS = (
    "control_name",
    "domain_name",
    "split_seed",
    "identity_field",
    "overlap_count",
    "overlap_preview",
    "status",
)

PREDICTION_COLUMNS = (
    "control_name",
    "domain_name",
    "split_seed",
    "variant_id",
    "representation_role",
    "sample_id",
    "case_id",
    "feature_row_index",
    "tumor_type",
    "scanner_model",
    "lab_or_origin",
    "species",
    "y_true",
    "y_pred",
    "prob_pos",
)

MODEL_MANIFEST_COLUMNS = (
    "control_name",
    "domain_name",
    "split_seed",
    "variant_id",
    "condition_mode",
    "input_dim",
    "effective_pca_dim",
    "latent_dim",
    "hidden_dim",
    "train_epochs",
    "beta_final",
    "kl_warmup_epochs",
    "n_fit",
    "fit_sample_hash",
)


@dataclass(frozen=True)
class VariantConfig:
    variant_id: str
    pca_dim: int
    latent_dim: int
    hidden_dim: int = 512
    num_hidden_layers: int = 2
    train_epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    beta_final: float = 0.001
    kl_warmup_epochs: int = 25


@dataclass(frozen=True)
class MidogPPPreservationSanityConfig:
    name: str
    artifact_root: Path
    manifest_path: Path
    feature_cache_path: Path
    signal_split_manifest_path: Path
    signal_decision_report_path: Path | None
    positive_label: int
    controls: tuple[str, ...]
    variants: tuple[VariantConfig, ...]
    primary_variant: str
    split_seeds: tuple[int, ...] | None
    bootstrap_reps: int
    bootstrap_seed: int
    allow_npz_test_cache: bool
    min_fit: int
    min_eval: int
    min_fit_pos: int
    min_fit_neg: int
    min_eval_pos: int
    min_eval_neg: int
    min_fit_cases: int
    min_eval_cases: int
    real_gate_min_bacc: float
    cvae_gate_min_bacc: float
    ci_low_threshold: float
    preservation_pass_ratio: float
    preservation_strong_ratio: float
    within_tumor_min_above_fraction: float


@dataclass(frozen=True)
class ManifestRow:
    manifest_row_index: int
    feature_row_index: int
    sample_id: str
    case_id: str
    label: int
    split: str
    metadata: Mapping[str, str]


@dataclass(frozen=True)
class SplitSpec:
    control_name: str
    domain_name: str
    split_seed: int
    fit_idx: tuple[int, ...]
    eval_idx: tuple[int, ...]


@dataclass(frozen=True)
class FeatureCache:
    embeddings: Any
    metadata: tuple[Mapping[str, object], ...]
    feature_extractor: Mapping[str, object]


@dataclass(frozen=True)
class CVAERuntime:
    variant: VariantConfig
    frame: ExpertFeatureFrame
    model: ClassConditionedCVAE
    fit_x: Any
    eval_x: Any
    fit_y: tuple[int, ...]
    condition_mode: str
    training_rows: tuple[dict[str, object], ...]


def load_midogpp_preservation_sanity_config(path: str | Path) -> MidogPPPreservationSanityConfig:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise ProtocolError("Loading MIDOG++ preservation sanity configs requires PyYAML.") from exc
    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ProtocolError(f"Config must be a mapping: {path}")
    base_dir = source.parents[2] if len(source.parents) >= 3 else source.parent
    return parse_midogpp_preservation_sanity_config(payload, base_dir=base_dir)


def parse_midogpp_preservation_sanity_config(
    data: Mapping[str, object],
    *,
    base_dir: str | Path = ".",
) -> MidogPPPreservationSanityConfig:
    base = Path(base_dir)
    experiment = _mapping(data.get("experiment"), "experiment")
    inputs = _mapping(data.get("inputs"), "inputs")
    run = _mapping(data.get("run"), "run", allow_empty=True)
    bootstrap = _mapping(data.get("bootstrap"), "bootstrap", allow_empty=True)
    thresholds = _mapping(data.get("validity_thresholds"), "validity_thresholds", allow_empty=True)
    decisions = _mapping(data.get("decision_thresholds"), "decision_thresholds", allow_empty=True)
    variants_payload = data.get("variants") or []
    variants = tuple(_parse_variant(item) for item in variants_payload) if variants_payload else _default_variants()
    signal_report = inputs.get("signal_decision_report_path", DEFAULT_SIGNAL_DECISION_REPORT_PATH)
    cfg = MidogPPPreservationSanityConfig(
        name=str(experiment.get("name", "")),
        artifact_root=_path(base, str(experiment.get("artifact_root", DEFAULT_ARTIFACT_ROOT))),
        manifest_path=_path(base, str(inputs.get("manifest_path", DEFAULT_MANIFEST_PATH))),
        feature_cache_path=_path(base, str(inputs.get("feature_cache_path", DEFAULT_FEATURE_CACHE_PATH))),
        signal_split_manifest_path=_path(
            base,
            str(inputs.get("signal_split_manifest_path", DEFAULT_SIGNAL_SPLIT_MANIFEST_PATH)),
        ),
        signal_decision_report_path=None if signal_report in ("", None) else _path(base, str(signal_report)),
        positive_label=int(data.get("positive_label", 1)),
        controls=tuple(str(v) for v in run.get("controls", CONTROL_ORDER)),
        variants=variants,
        primary_variant=str(run.get("primary_variant", PRIMARY_VARIANT)),
        split_seeds=None
        if run.get("split_seeds") in (None, "", "all")
        else tuple(int(v) for v in run.get("split_seeds", ())),
        bootstrap_reps=int(bootstrap.get("reps", 1000)),
        bootstrap_seed=int(bootstrap.get("seed", 1337)),
        allow_npz_test_cache=bool(inputs.get("allow_npz_test_cache", False)),
        min_fit=int(thresholds.get("min_fit", 20)),
        min_eval=int(thresholds.get("min_eval", 10)),
        min_fit_pos=int(thresholds.get("min_fit_pos", 10)),
        min_fit_neg=int(thresholds.get("min_fit_neg", 10)),
        min_eval_pos=int(thresholds.get("min_eval_pos", 5)),
        min_eval_neg=int(thresholds.get("min_eval_neg", 5)),
        min_fit_cases=int(thresholds.get("min_fit_cases", 3)),
        min_eval_cases=int(thresholds.get("min_eval_cases", 2)),
        real_gate_min_bacc=float(decisions.get("real_gate_min_bacc", 0.60)),
        cvae_gate_min_bacc=float(decisions.get("cvae_gate_min_bacc", 0.60)),
        ci_low_threshold=float(decisions.get("ci_low_threshold", 0.50)),
        preservation_pass_ratio=float(decisions.get("preservation_pass_ratio", 0.80)),
        preservation_strong_ratio=float(decisions.get("preservation_strong_ratio", 0.90)),
        within_tumor_min_above_fraction=float(decisions.get("within_tumor_min_above_fraction", 0.60)),
    )
    validate_midogpp_preservation_sanity_config(cfg)
    return cfg


def validate_midogpp_preservation_sanity_config(cfg: MidogPPPreservationSanityConfig) -> None:
    if cfg.name != EXPERIMENT_NAME:
        raise ProtocolError(f"Unexpected experiment.name={cfg.name!r}; expected {EXPERIMENT_NAME!r}.")
    controls = set(cfg.controls)
    unknown = sorted(controls - set(CONTROL_ORDER))
    if unknown:
        raise ProtocolError(f"Unknown controls: {unknown}")
    if cfg.primary_variant != PRIMARY_VARIANT:
        raise ProtocolError(f"Primary variant must remain {PRIMARY_VARIANT!r}.")
    variant_ids = {variant.variant_id for variant in cfg.variants}
    if PRIMARY_VARIANT not in variant_ids:
        raise ProtocolError(f"variants must include primary variant {PRIMARY_VARIANT!r}.")
    for variant in cfg.variants:
        if variant.variant_id == PRIMARY_VARIANT and (variant.pca_dim != 64 or variant.latent_dim != 16):
            raise ProtocolError("pca64_beta001 must use pca_dim=64 and latent_dim=16.")
        if variant.variant_id == DIAGNOSTIC_VARIANT and (variant.pca_dim != 128 or variant.latent_dim != 32):
            raise ProtocolError("pca128_beta001 must use pca_dim=128 and latent_dim=32.")
        if variant.num_hidden_layers != 2 or variant.hidden_dim != 512:
            raise ProtocolError("CVAE architecture is locked to hidden_dim=512 and two hidden layers.")
        if not math.isclose(variant.beta_final, 0.001):
            raise ProtocolError("CVAE beta_final is locked to 0.001.")


def run_midogpp_preservation_sanity(
    cfg: MidogPPPreservationSanityConfig,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    import numpy as np  # type: ignore

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
        raise ProtocolError("No matching SAIL signal-control splits were found.")

    metric_rows: list[dict[str, object]] = []
    negative_rows: list[dict[str, object]] = []
    reconstruction_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    model_manifest_rows: list[dict[str, object]] = []

    for spec in split_specs:
        audit = _identity_audit_rows(manifest_rows, spec)
        audit_rows.extend(audit)
        status, error = _split_status(cfg, manifest_rows, spec, audit)
        counts = _split_counts(manifest_rows, spec.fit_idx, spec.eval_idx)
        if status != VALID_STATUS:
            metric_rows.append(
                _empty_metric_row(
                    spec,
                    variant_id=PRIMARY_VARIANT,
                    representation_role=REAL_FRAME_REFERENCE,
                    counts=counts,
                    status=status,
                    error_message=error,
                )
            )
            continue

        x_fit_raw = embeddings[list(spec.fit_idx)]
        y_fit = np.asarray([manifest_rows[idx].label for idx in spec.fit_idx], dtype=int)
        x_eval_raw = embeddings[list(spec.eval_idx)]
        y_eval = [manifest_rows[idx].label for idx in spec.eval_idx]

        raw_row, raw_preds = _evaluate_representation(
            cfg,
            spec,
            rows=manifest_rows,
            x_fit=x_fit_raw,
            y_fit=y_fit,
            x_eval=x_eval_raw,
            y_eval=y_eval,
            variant_id=RAW_VARIANT,
            representation_role=REAL_RAW_REFERENCE,
            counts=counts,
            prediction_prefix="",
        )
        metric_rows.append(raw_row)
        prediction_rows.extend(raw_preds)

        for variant in cfg.variants:
            runtime = _train_runtime(
                cfg,
                spec,
                variant,
                x_fit_raw=x_fit_raw,
                x_eval_raw=x_eval_raw,
                y_fit=y_fit,
                condition_mode="real_labels",
            )
            training_rows.extend(runtime.training_rows)
            model_manifest_rows.append(_model_manifest_row(spec, runtime, manifest_rows))

            frame_row, frame_preds = _evaluate_representation(
                cfg,
                spec,
                rows=manifest_rows,
                x_fit=runtime.fit_x,
                y_fit=y_fit,
                x_eval=runtime.eval_x,
                y_eval=y_eval,
                variant_id=variant.variant_id,
                representation_role=REAL_FRAME_REFERENCE,
                counts=counts,
                prediction_prefix="",
            )
            metric_rows.append(frame_row)
            prediction_rows.extend(frame_preds)

            for role, neg_x, neg_y in _real_frame_negative_inputs(cfg, spec, runtime.fit_x, y_fit):
                neg_row, neg_preds = _evaluate_representation(
                    cfg,
                    spec,
                    rows=manifest_rows,
                    x_fit=neg_x,
                    y_fit=neg_y,
                    x_eval=runtime.eval_x,
                    y_eval=y_eval,
                    variant_id=variant.variant_id,
                    representation_role=role,
                    counts=counts,
                    prediction_prefix="",
                )
                negative_rows.append(neg_row)
                prediction_rows.extend(neg_preds)

            decoded, recon = _decode_mu(runtime, runtime.fit_x, y_fit)
            reconstruction_rows.append(_reconstruction_row(spec, variant, DECODE_MU, recon, n_rows=len(y_fit)))
            decode_row, decode_preds = _evaluate_representation(
                cfg,
                spec,
                rows=manifest_rows,
                x_fit=decoded,
                y_fit=y_fit,
                x_eval=runtime.eval_x,
                y_eval=y_eval,
                variant_id=variant.variant_id,
                representation_role=DECODE_MU,
                counts=counts,
                prediction_prefix="",
            )
            metric_rows.append(decode_row)
            prediction_rows.extend(decode_preds)

            posterior = _posterior_sample(runtime, runtime.fit_x, y_fit, seed=_stable_seed(spec, variant.variant_id, "posterior"))
            posterior_row, posterior_preds = _evaluate_representation(
                cfg,
                spec,
                rows=manifest_rows,
                x_fit=posterior,
                y_fit=y_fit,
                x_eval=runtime.eval_x,
                y_eval=y_eval,
                variant_id=variant.variant_id,
                representation_role=POSTERIOR_SAMPLE,
                counts=counts,
                prediction_prefix="",
            )
            metric_rows.append(posterior_row)
            prediction_rows.extend(posterior_preds)

            prior_x, prior_y = _prior_sample(runtime, y_fit, seed=_stable_seed(spec, variant.variant_id, "prior"))
            prior_row, prior_preds = _evaluate_representation(
                cfg,
                spec,
                rows=manifest_rows,
                x_fit=prior_x,
                y_fit=prior_y,
                x_eval=runtime.eval_x,
                y_eval=y_eval,
                variant_id=variant.variant_id,
                representation_role=PRIOR_SAMPLE,
                counts=counts,
                prediction_prefix="",
            )
            metric_rows.append(prior_row)
            prediction_rows.extend(prior_preds)

            shuffled_decoded = np.asarray(decoded, dtype=float)[
                _rng(_stable_seed(spec, variant.variant_id, "decoded_shuffle")).permutation(len(decoded))
            ]
            decoded_neg_row, decoded_neg_preds = _evaluate_representation(
                cfg,
                spec,
                rows=manifest_rows,
                x_fit=shuffled_decoded,
                y_fit=y_fit,
                x_eval=runtime.eval_x,
                y_eval=y_eval,
                variant_id=variant.variant_id,
                representation_role=CVAE_DECODED_ROW_SHUFFLE,
                counts=counts,
                prediction_prefix="",
            )
            negative_rows.append(decoded_neg_row)
            prediction_rows.extend(decoded_neg_preds)

            if variant.variant_id == cfg.primary_variant:
                perm_runtime = _train_runtime(
                    cfg,
                    spec,
                    variant,
                    x_fit_raw=x_fit_raw,
                    x_eval_raw=x_eval_raw,
                    y_fit=y_fit,
                    condition_mode="permuted_labels",
                )
                training_rows.extend(perm_runtime.training_rows)
                model_manifest_rows.append(_model_manifest_row(spec, perm_runtime, manifest_rows))
                perm_labels = _permuted_labels(cfg, spec, y_fit, "permuted_labels")
                perm_decoded, perm_recon = _decode_mu(perm_runtime, perm_runtime.fit_x, perm_labels)
                reconstruction_rows.append(
                    _reconstruction_row(spec, variant, CVAE_CONDITION_LABEL_PERMUTATION, perm_recon, n_rows=len(y_fit))
                )
                perm_row, perm_preds = _evaluate_representation(
                    cfg,
                    spec,
                    rows=manifest_rows,
                    x_fit=perm_decoded,
                    y_fit=y_fit,
                    x_eval=perm_runtime.eval_x,
                    y_eval=y_eval,
                    variant_id=variant.variant_id,
                    representation_role=CVAE_CONDITION_LABEL_PERMUTATION,
                    counts=counts,
                    prediction_prefix="",
                )
                negative_rows.append(perm_row)
                prediction_rows.extend(perm_preds)

    summary_rows = _aggregate_rows(metric_rows)
    negative_summary_rows = _aggregate_rows(negative_rows)
    all_metric_rows = metric_rows + summary_rows
    all_negative_rows = negative_rows + negative_summary_rows
    _attach_preservation_ratios(all_metric_rows)
    gap_rows = _gap_rows(all_metric_rows)
    prior = _prior_signal_status(cfg.signal_decision_report_path)
    suitability_rows, real_suitable = _real_baseline_suitability(cfg, all_metric_rows, all_negative_rows, prior)
    leakage = _leakage_report(audit_rows, all_negative_rows)
    decision_labels = _decision_labels(cfg, all_metric_rows, all_negative_rows, leakage, real_suitable, prior)
    write_csv_rows(root / "tables" / "preservation_metrics.csv", all_metric_rows, PRESERVATION_COLUMNS)
    write_csv_rows(root / "tables" / "variant_gap_summary.csv", gap_rows, GAP_COLUMNS)
    write_csv_rows(root / "tables" / "real_baseline_suitability.csv", suitability_rows, SUITABILITY_COLUMNS)
    write_csv_rows(root / "tables" / "reconstruction_diagnostics.csv", reconstruction_rows, RECON_COLUMNS)
    write_csv_rows(root / "tables" / "training_diagnostics.csv", training_rows, TRAINING_COLUMNS)
    write_csv_rows(root / "tables" / "negative_control_metrics.csv", all_negative_rows, PRESERVATION_COLUMNS)
    write_csv_rows(root / "tables" / "identity_overlap_audit.csv", audit_rows, AUDIT_COLUMNS)
    write_csv_rows(root / "tables" / "predictions.csv", prediction_rows, PREDICTION_COLUMNS)
    write_csv_rows(root / "manifests" / "model_manifest.csv", model_manifest_rows, MODEL_MANIFEST_COLUMNS)
    write_json(root / "manifests" / "protocol_manifest.json", _protocol_manifest(cfg, cache, manifest_rows, split_specs, prior))
    write_json(root / "reports" / "leakage_report.json", leakage)
    _write_decision_report(root / "reports" / "decision_report.md", cfg, decision_labels, leakage, suitability_rows, all_metric_rows, all_negative_rows, prior)
    return root


def _parse_variant(data: object) -> VariantConfig:
    item = _mapping(data, "variant")
    return VariantConfig(
        variant_id=str(item["variant_id"]),
        pca_dim=int(item["pca_dim"]),
        latent_dim=int(item["latent_dim"]),
        hidden_dim=int(item.get("hidden_dim", 512)),
        num_hidden_layers=int(item.get("num_hidden_layers", 2)),
        train_epochs=int(item.get("train_epochs", 100)),
        batch_size=int(item.get("batch_size", 128)),
        learning_rate=float(item.get("learning_rate", 1.0e-3)),
        weight_decay=float(item.get("weight_decay", 1.0e-4)),
        beta_final=float(item.get("beta_final", 0.001)),
        kl_warmup_epochs=int(item.get("kl_warmup_epochs", 25)),
    )


def _default_variants() -> tuple[VariantConfig, ...]:
    return (
        VariantConfig(PRIMARY_VARIANT, pca_dim=64, latent_dim=16),
        VariantConfig(DIAGNOSTIC_VARIANT, pca_dim=128, latent_dim=32),
    )


def _mapping(value: object, name: str, *, allow_empty: bool = False) -> Mapping[str, Any]:
    if value is None and allow_empty:
        return {}
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{name} must be a mapping.")
    return value


def _path(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _read_train_manifest(path: Path, *, positive_label: int) -> tuple[ManifestRow, ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ProtocolError(f"Empty MIDOG++ manifest: {path}")
        required = {"sample_id", "case_id", "label", "split"}
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ProtocolError(f"MIDOG++ manifest missing required fields: {missing}")
        rows: list[ManifestRow] = []
        for manifest_idx, row in enumerate(reader):
            split = str(row.get("split", "")).strip().lower()
            if split != "train":
                continue
            raw_label = int(float(str(row.get("label", "")).strip()))
            feature_idx = len(rows)
            rows.append(
                ManifestRow(
                    manifest_row_index=int(manifest_idx),
                    feature_row_index=int(feature_idx),
                    sample_id=_required(row, "sample_id", manifest_idx),
                    case_id=_required(row, "case_id", manifest_idx),
                    label=1 if raw_label == int(positive_label) else 0,
                    split=split,
                    metadata={str(k): str(v).strip() for k, v in row.items()},
                )
            )
    if not rows:
        raise ProtocolError(f"MIDOG++ manifest has no split=train rows: {path}")
    return tuple(rows)


def _required(row: Mapping[str, object], key: str, idx: int) -> str:
    value = str(row.get(key, "")).strip()
    if not value:
        raise ProtocolError(f"Manifest row {idx} missing {key}.")
    return value


def _load_feature_cache(path: Path) -> FeatureCache:
    if path.suffix == ".npz":
        import numpy as np  # type: ignore

        payload = np.load(path, allow_pickle=True)
        if "metadata_json" in payload:
            metadata = json.loads(str(payload["metadata_json"].item()))
        else:
            metadata = payload["metadata"].tolist()
        feature_extractor = (
            json.loads(str(payload["feature_extractor_json"].item()))
            if "feature_extractor_json" in payload
            else {"loader": "npz_test_or_lightweight_cache"}
        )
        return FeatureCache(payload["embeddings"], tuple(dict(row) for row in metadata), feature_extractor)
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Loading torch feature caches requires torch.") from exc
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ProtocolError(f"Feature cache payload must be a mapping: {path}")
    return FeatureCache(
        embeddings=payload["embeddings"],
        metadata=tuple(dict(row) for row in payload.get("metadata", ())),
        feature_extractor=payload.get("feature_extractor", {}) if isinstance(payload.get("feature_extractor", {}), Mapping) else {},
    )


def _assert_virchow2_cache(cache: FeatureCache, *, path: Path, allow_npz_test_cache: bool) -> None:
    if allow_npz_test_cache and path.suffix == ".npz":
        return
    text = " ".join(
        str(cache.feature_extractor.get(key, ""))
        for key in ("backbone_type", "model_ref", "feature_extractor_name", "loader")
    ).lower()
    if "virchow2" not in text:
        raise ProtocolError(f"cache_alignment_failed: feature cache does not declare Virchow2 provenance: {path}")


def _assert_cache_alignment(rows: Sequence[ManifestRow], cache: FeatureCache) -> None:
    if len(rows) != len(cache.metadata):
        raise ProtocolError(f"cache_alignment_failed: train rows={len(rows)} cache rows={len(cache.metadata)}")
    for idx, (row, meta) in enumerate(zip(rows, cache.metadata)):
        cache_sample = str(meta.get("sample_id", "")).strip()
        if cache_sample and row.sample_id != cache_sample:
            raise ProtocolError(f"cache_alignment_failed: row {idx} sample_id manifest={row.sample_id!r} cache={cache_sample!r}")
        if "label" in meta and int(float(str(meta["label"]))) != int(row.label):
            raise ProtocolError(f"cache_alignment_failed: row {idx} label manifest={row.label!r} cache={meta['label']!r}")


def _to_numpy(value: object) -> object:
    import numpy as np  # type: ignore

    try:
        import torch  # type: ignore
    except ModuleNotFoundError:
        torch = None  # type: ignore
    if torch is not None and hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _read_split_manifest(
    path: Path,
    rows: Sequence[ManifestRow],
    *,
    controls: Sequence[str],
) -> tuple[SplitSpec, ...]:
    grouped: dict[tuple[str, str, int], dict[str, list[int]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ProtocolError(f"Empty split manifest: {path}")
        required = {"control_name", "domain_name", "split_seed", "subset", "feature_row_index", "sample_id"}
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ProtocolError(f"Split manifest missing required fields: {missing}")
        for item in reader:
            control = str(item["control_name"])
            if control not in controls:
                continue
            subset = str(item["subset"])
            if subset not in {"fit", "eval"}:
                continue
            feature_idx = int(item["feature_row_index"])
            if feature_idx < 0 or feature_idx >= len(rows):
                raise ProtocolError(f"Split manifest feature_row_index out of bounds: {feature_idx}")
            sample_id = str(item.get("sample_id", "")).strip()
            if sample_id and sample_id != rows[feature_idx].sample_id:
                raise ProtocolError(
                    f"Split manifest row mismatch for feature_row_index={feature_idx}: "
                    f"{sample_id!r} != {rows[feature_idx].sample_id!r}"
                )
            key = (control, str(item.get("domain_name", "")).strip(), int(item["split_seed"]))
            grouped.setdefault(key, {"fit": [], "eval": []})[subset].append(feature_idx)
    specs = [
        SplitSpec(control, domain, seed, tuple(sorted(set(values["fit"]))), tuple(sorted(set(values["eval"]))))
        for (control, domain, seed), values in sorted(grouped.items(), key=lambda item: (CONTROL_ORDER.index(item[0][0]), item[0][1], item[0][2]))
    ]
    return tuple(specs)


def _identity_audit_rows(rows: Sequence[ManifestRow], spec: SplitSpec) -> list[dict[str, object]]:
    out = []
    for field in ("sample_id", "case_id", "image_path", "feature_row_index"):
        fit = _identity_set(rows, spec.fit_idx, field)
        eval_ = _identity_set(rows, spec.eval_idx, field)
        overlap = sorted(fit.intersection(eval_))
        out.append(
            {
                "control_name": spec.control_name,
                "domain_name": spec.domain_name,
                "split_seed": int(spec.split_seed),
                "identity_field": field,
                "overlap_count": int(len(overlap)),
                "overlap_preview": "|".join(overlap[:10]),
                "status": "PASS" if not overlap else "FAIL",
            }
        )
    return out


def _identity_set(rows: Sequence[ManifestRow], indices: Sequence[int], field: str) -> set[str]:
    out = set()
    for idx in indices:
        row = rows[idx]
        if field == "sample_id":
            value = row.sample_id
        elif field == "case_id":
            value = row.case_id
        elif field == "image_path":
            value = str(row.metadata.get("image_path", "")).strip()
        elif field == "feature_row_index":
            value = str(row.feature_row_index)
        else:
            value = ""
        if value:
            out.add(value)
    return out


def _split_status(
    cfg: MidogPPPreservationSanityConfig,
    rows: Sequence[ManifestRow],
    spec: SplitSpec,
    audit_rows: Sequence[Mapping[str, object]],
) -> tuple[str, str]:
    for audit in audit_rows:
        if int(audit.get("overlap_count", 0)) > 0:
            return f"protocol_failed_{audit['identity_field']}_overlap", str(audit.get("overlap_preview", ""))
    counts = _split_counts(rows, spec.fit_idx, spec.eval_idx)
    checks = (
        ("n_fit", cfg.min_fit, "invalid_too_few_fit"),
        ("n_eval", cfg.min_eval, "invalid_too_few_eval"),
        ("n_fit_pos", cfg.min_fit_pos, "invalid_too_few_fit_pos"),
        ("n_fit_neg", cfg.min_fit_neg, "invalid_too_few_fit_neg"),
        ("n_eval_pos", cfg.min_eval_pos, "invalid_too_few_eval_pos"),
        ("n_eval_neg", cfg.min_eval_neg, "invalid_too_few_eval_neg"),
        ("n_fit_cases", cfg.min_fit_cases, "invalid_too_few_fit_cases"),
        ("n_eval_cases", cfg.min_eval_cases, "invalid_too_few_eval_cases"),
    )
    for key, minimum, status in checks:
        if int(counts[key]) < int(minimum):
            return status, f"{key}={counts[key]} minimum={minimum}"
    return VALID_STATUS, ""


def _split_counts(rows: Sequence[ManifestRow], fit_idx: Sequence[int], eval_idx: Sequence[int]) -> dict[str, int]:
    fit_labels = [rows[idx].label for idx in fit_idx]
    eval_labels = [rows[idx].label for idx in eval_idx]
    return {
        "n_fit": len(fit_idx),
        "n_eval": len(eval_idx),
        "n_fit_pos": sum(1 for value in fit_labels if value == 1),
        "n_fit_neg": sum(1 for value in fit_labels if value == 0),
        "n_eval_pos": sum(1 for value in eval_labels if value == 1),
        "n_eval_neg": sum(1 for value in eval_labels if value == 0),
        "n_fit_cases": len({rows[idx].case_id for idx in fit_idx}),
        "n_eval_cases": len({rows[idx].case_id for idx in eval_idx}),
    }


def _train_runtime(
    cfg: MidogPPPreservationSanityConfig,
    spec: SplitSpec,
    variant: VariantConfig,
    *,
    x_fit_raw: object,
    x_eval_raw: object,
    y_fit: Sequence[int],
    condition_mode: str,
) -> CVAERuntime:
    import numpy as np  # type: ignore
    import torch  # type: ignore

    seed = _stable_seed(spec, variant.variant_id, condition_mode)
    torch.manual_seed(seed)
    frame = fit_expert_frame(
        expert_id=f"{spec.control_name}:{spec.domain_name}:{spec.split_seed}:{variant.variant_id}",
        source_train_embeddings=np.asarray(x_fit_raw, dtype=float),
        requested_dim=int(variant.pca_dim),
    )
    fit_x = np.asarray(frame.transform(np.asarray(x_fit_raw, dtype=float)), dtype=np.float32)
    eval_x = np.asarray(frame.transform(np.asarray(x_eval_raw, dtype=float)), dtype=np.float32)
    fit_y = tuple(int(v) for v in y_fit)
    train_y = fit_y if condition_mode == "real_labels" else tuple(int(v) for v in _permuted_labels(cfg, spec, fit_y, condition_mode))
    model = ClassConditionedCVAE(
        input_dim=int(fit_x.shape[1]),
        hidden_dim=int(variant.hidden_dim),
        latent_dim=int(variant.latent_dim),
        n_classes=2,
        num_hidden_layers=int(variant.num_hidden_layers),
    )
    training_rows = _train_cvae(cfg, spec, variant, model, fit_x, train_y, condition_mode=condition_mode, seed=seed)
    return CVAERuntime(
        variant=variant,
        frame=frame,
        model=model,
        fit_x=fit_x,
        eval_x=eval_x,
        fit_y=fit_y,
        condition_mode=condition_mode,
        training_rows=tuple(training_rows),
    )


def _train_cvae(
    cfg: MidogPPPreservationSanityConfig,
    spec: SplitSpec,
    variant: VariantConfig,
    model: ClassConditionedCVAE,
    fit_x: object,
    fit_y: Sequence[int],
    *,
    condition_mode: str,
    seed: int,
) -> list[dict[str, object]]:
    del cfg
    import numpy as np  # type: ignore
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore
    from torch.nn.utils import clip_grad_norm_  # type: ignore
    from torch.utils.data import DataLoader, TensorDataset  # type: ignore

    x = torch.as_tensor(np.asarray(fit_x, dtype=np.float32))
    y = torch.as_tensor(np.asarray(fit_y, dtype=np.int64))
    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=int(variant.batch_size),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=float(variant.learning_rate), weight_decay=float(variant.weight_decay))
    rows = []
    for epoch in range(1, int(variant.train_epochs) + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            mu, logvar = model.encode(xb, yb)
            decoded = model.decode(mu, yb)
            recon = F.mse_loss(decoded, xb, reduction="none").mean(dim=1).mean()
            kl = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1) / float(model.latent_dim)).mean()
            beta = _beta_for_epoch(variant, epoch)
            loss = recon + (beta * kl)
            loss.backward()
            clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        rows.append(
            {
                "control_name": spec.control_name,
                "domain_name": spec.domain_name,
                "split_seed": int(spec.split_seed),
                "variant_id": variant.variant_id,
                "condition_mode": condition_mode,
                "epoch": int(epoch),
                "beta": _beta_for_epoch(variant, epoch),
                "train_recon_mse_per_dim": float(recon.detach().cpu()),
                "train_kl_per_latent_dim": float(kl.detach().cpu()),
                "train_total_loss": float(loss.detach().cpu()),
            }
        )
    model.eval()
    return rows


def _beta_for_epoch(variant: VariantConfig, epoch: int) -> float:
    warmup = max(1, int(variant.kl_warmup_epochs))
    return float(variant.beta_final) * min(1.0, float(epoch) / float(warmup))


def _decode_mu(runtime: CVAERuntime, x: object, labels: Sequence[int]) -> tuple[object, dict[str, float]]:
    import numpy as np  # type: ignore
    import torch  # type: ignore

    x_np = np.asarray(x, dtype=np.float32)
    y_np = np.asarray(labels, dtype=np.int64)
    with torch.no_grad():
        xt = torch.as_tensor(x_np, dtype=torch.float32)
        yt = torch.as_tensor(y_np, dtype=torch.long)
        mu, logvar = runtime.model.encode(xt, yt)
        decoded = runtime.model.decode(mu, yt).detach().cpu().numpy()
    return decoded, _reconstruction_diagnostics(x_np, decoded, mu, logvar)


def _posterior_sample(runtime: CVAERuntime, x: object, labels: Sequence[int], *, seed: int) -> object:
    import numpy as np  # type: ignore
    import torch  # type: ignore

    torch_gen = torch.Generator(device="cpu").manual_seed(int(seed))
    x_np = np.asarray(x, dtype=np.float32)
    y_np = np.asarray(labels, dtype=np.int64)
    with torch.no_grad():
        xt = torch.as_tensor(x_np, dtype=torch.float32)
        yt = torch.as_tensor(y_np, dtype=torch.long)
        mu, logvar = runtime.model.encode(xt, yt)
        noise = torch.randn(mu.shape, generator=torch_gen, dtype=mu.dtype, device=mu.device)
        z = mu + (noise * torch.exp(0.5 * logvar))
        return runtime.model.decode(z, yt).detach().cpu().numpy()


def _prior_sample(runtime: CVAERuntime, labels: Sequence[int], *, seed: int) -> tuple[object, tuple[int, ...]]:
    import numpy as np  # type: ignore
    import torch  # type: ignore

    y = []
    for cls in (0, 1):
        y.extend([cls] * sum(1 for value in labels if int(value) == cls))
    torch_gen = torch.Generator(device="cpu").manual_seed(int(seed))
    with torch.no_grad():
        z = torch.randn((len(y), runtime.model.latent_dim), generator=torch_gen, dtype=torch.float32)
        yt = torch.as_tensor(np.asarray(y, dtype=np.int64), dtype=torch.long)
        decoded = runtime.model.decode(z, yt).detach().cpu().numpy()
    return decoded, tuple(int(v) for v in y)


def _reconstruction_diagnostics(x: object, decoded: object, mu: object, logvar: object) -> dict[str, float]:
    import numpy as np  # type: ignore
    import torch  # type: ignore

    x_np = np.asarray(x, dtype=np.float32)
    decoded_np = np.asarray(decoded, dtype=np.float32)
    mse = np.mean((decoded_np - x_np) ** 2)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1) / float(mu.shape[1])
    return {
        "recon_mse_per_dim": float(mse),
        "recon_rmse_per_dim": float(math.sqrt(float(mse))),
        "kl_per_latent_dim": float(kl.mean().detach().cpu()),
        "mu_norm_mean": float(torch.norm(mu, dim=1).mean().detach().cpu()),
        "logvar_mean": float(logvar.mean().detach().cpu()),
    }


def _evaluate_representation(
    cfg: MidogPPPreservationSanityConfig,
    spec: SplitSpec,
    *,
    rows: Sequence[ManifestRow],
    x_fit: object,
    y_fit: Sequence[int],
    x_eval: object,
    y_eval: Sequence[int],
    variant_id: str,
    representation_role: str,
    counts: Mapping[str, int],
    prediction_prefix: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    del prediction_prefix
    base = _empty_metric_row(
        spec,
        variant_id=variant_id,
        representation_role=representation_role,
        counts=counts,
        status=VALID_STATUS,
        error_message="",
    )
    try:
        probabilities, predictions, converged = _fit_predict(x_fit, y_fit, x_eval, seed=int(spec.split_seed))
    except Exception as exc:
        failed = dict(base)
        failed.update({"status": "model_failed_convergence", "error_message": str(exc), "converged": "false"})
        return failed, []
    if not converged:
        failed = dict(base)
        failed.update({"status": "model_failed_convergence", "error_message": "sklearn convergence warning", "converged": "false"})
        return failed, []
    metrics = _classification_metrics(y_eval, predictions)
    ci_low, ci_high, ci_method = _case_cluster_bacc_ci(
        y_eval,
        predictions,
        [rows[idx].case_id for idx in spec.eval_idx],
        reps=cfg.bootstrap_reps,
        seed=_stable_seed(spec, variant_id, representation_role, "ci", cfg.bootstrap_seed),
    )
    above = _above_chance(metrics["bacc"], ci_low, cfg)
    near = _near_chance(ci_low, ci_high, metrics["recall_pos"])
    out = dict(base)
    out.update(
        {
            **metrics,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "ci_method": ci_method,
            "above_chance": _bool_text(above),
            "near_chance": _bool_text(near),
            "converged": "true",
            "status": VALID_STATUS,
            "error_message": "",
        }
    )
    preds = []
    for pos, idx in enumerate(spec.eval_idx):
        row = rows[idx]
        preds.append(
            {
                "control_name": spec.control_name,
                "domain_name": spec.domain_name,
                "split_seed": int(spec.split_seed),
                "variant_id": variant_id,
                "representation_role": representation_role,
                "sample_id": row.sample_id,
                "case_id": row.case_id,
                "feature_row_index": int(row.feature_row_index),
                "tumor_type": row.metadata.get("tumor_type", ""),
                "scanner_model": row.metadata.get("scanner_model", ""),
                "lab_or_origin": row.metadata.get("lab_or_origin", ""),
                "species": row.metadata.get("species", ""),
                "y_true": int(y_eval[pos]),
                "y_pred": int(predictions[pos]),
                "prob_pos": float(probabilities[pos]),
            }
        )
    return out, preds


def _fit_predict(x_fit: object, y_fit: Sequence[int], x_eval: object, *, seed: int) -> tuple[list[float], list[int], bool]:
    import numpy as np  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore

    x_train = np.asarray(x_fit, dtype=float)
    y_train = np.asarray(y_fit, dtype=int)
    x_target = np.asarray(x_eval, dtype=float)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_target = scaler.transform(x_target)
    clf = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=5000,
        random_state=int(seed),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(x_train, y_train)
    converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
    classes = tuple(int(value) for value in clf.classes_.tolist())
    if classes != (0, 1):
        raise ProtocolError(f"class order must be (0, 1), got {classes}")
    prob = clf.predict_proba(x_target)[:, 1]
    pred = [1 if float(value) >= 0.5 else 0 for value in prob.tolist()]
    return [float(value) for value in prob.tolist()], pred, converged


def _classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
    tp = sum(1 for truth, pred in zip(y_true, y_pred) if int(truth) == 1 and int(pred) == 1)
    fp = sum(1 for truth, pred in zip(y_true, y_pred) if int(truth) == 0 and int(pred) == 1)
    fn = sum(1 for truth, pred in zip(y_true, y_pred) if int(truth) == 1 and int(pred) == 0)
    precision = float(tp) / float(tp + fp) if (tp + fp) else 0.0
    recall = float(tp) / float(tp + fn) if (tp + fn) else 0.0
    f1 = float(2 * precision * recall) / float(precision + recall) if (precision + recall) else 0.0
    return {
        "bacc": balanced_accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "precision_pos": precision,
        "recall_pos": recall,
        "f1_pos": f1,
    }


def _case_cluster_bacc_ci(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    case_ids: Sequence[str],
    *,
    reps: int,
    seed: int,
) -> tuple[float, float, str]:
    import numpy as np  # type: ignore

    if len(y_true) != len(case_ids) or not case_ids or any(str(case).strip() == "" for case in case_ids):
        return _row_level_bacc_ci(y_true, y_pred, reps=reps, seed=seed) + ("row_level_fallback_missing_case_id",)
    by_case: dict[str, list[int]] = {}
    for idx, case_id in enumerate(case_ids):
        by_case.setdefault(str(case_id), []).append(idx)
    cases = sorted(by_case)
    if len(cases) < 2 or int(reps) <= 0:
        return math.nan, math.nan, "case_cluster"
    rng = _rng(seed)
    values = []
    for _ in range(int(reps)):
        sampled_cases = rng.choice(cases, size=len(cases), replace=True)
        sampled_idx = [idx for case in sampled_cases for idx in by_case[str(case)]]
        sampled_y = [int(y_true[idx]) for idx in sampled_idx]
        if len(set(sampled_y)) < 2:
            continue
        values.append(balanced_accuracy(sampled_y, [int(y_pred[idx]) for idx in sampled_idx]))
    if not values:
        return math.nan, math.nan, "case_cluster"
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)), "case_cluster"


def _row_level_bacc_ci(y_true: Sequence[int], y_pred: Sequence[int], *, reps: int, seed: int) -> tuple[float, float]:
    import numpy as np  # type: ignore

    if len(set(int(value) for value in y_true)) < 2 or int(reps) <= 0:
        return math.nan, math.nan
    rng = _rng(seed)
    values = []
    for _ in range(int(reps)):
        sampled = rng.choice(range(len(y_true)), size=len(y_true), replace=True)
        sampled_y = [int(y_true[idx]) for idx in sampled]
        if len(set(sampled_y)) < 2:
            continue
        values.append(balanced_accuracy(sampled_y, [int(y_pred[idx]) for idx in sampled]))
    if not values:
        return math.nan, math.nan
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def _empty_metric_row(
    spec: SplitSpec,
    *,
    variant_id: str,
    representation_role: str,
    counts: Mapping[str, int],
    status: str,
    error_message: str,
) -> dict[str, object]:
    return {
        "aggregation_level": "seed",
        "control_name": spec.control_name,
        "domain_name": spec.domain_name,
        "split_seed": int(spec.split_seed),
        "variant_id": variant_id,
        "representation_role": representation_role,
        "model_type": LOGISTIC_MODEL_TYPE,
        "eligible_seed_count": "",
        "valid_seed_count": "",
        "decision_status": "",
        **dict(counts),
        "bacc": math.nan,
        "bacc_std": "",
        "bacc_min": "",
        "macro_f1": math.nan,
        "precision_pos": math.nan,
        "recall_pos": math.nan,
        "f1_pos": math.nan,
        "support_pos": counts.get("n_eval_pos", ""),
        "support_neg": counts.get("n_eval_neg", ""),
        "ci_low": math.nan,
        "ci_high": math.nan,
        "ci_method": "",
        "above_chance": "",
        "near_chance": "",
        "converged": "false",
        "status": status,
        "error_message": error_message,
        "preservation_ratio_vs_real_frame": "",
    }


def _aggregate_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(
            (
                str(row.get("control_name", "")),
                str(row.get("domain_name", "")),
                str(row.get("variant_id", "")),
                str(row.get("representation_role", "")),
            ),
            [],
        ).append(row)
    out = []
    for (_control, _domain, _variant, _role), group in sorted(grouped.items()):
        valid = [row for row in group if str(row.get("status")) == VALID_STATUS]
        base = dict(group[0])
        bacc_values = [_float_or_nan(row.get("bacc")) for row in valid]
        mean_bacc = nanmean(bacc_values)
        ci_low = _nanmin([row.get("ci_low") for row in valid])
        ci_high = _nanmax([row.get("ci_high") for row in valid])
        recall = nanmean([_float_or_nan(row.get("recall_pos")) for row in valid])
        above = bool(_finite(mean_bacc) and _finite(ci_low) and mean_bacc >= 0.60 and ci_low > 0.50)
        near = _near_chance(ci_low, ci_high, recall)
        base.update(
            {
                "aggregation_level": "summary",
                "split_seed": "",
                "eligible_seed_count": len(group),
                "valid_seed_count": len(valid),
                "decision_status": _decision_status(valid, above=above, near=near),
                "n_fit": "",
                "n_eval": "",
                "n_fit_pos": "",
                "n_fit_neg": "",
                "n_eval_pos": "",
                "n_eval_neg": "",
                "n_fit_cases": "",
                "n_eval_cases": "",
                "bacc": mean_bacc,
                "bacc_std": _nanstd(bacc_values),
                "bacc_min": _nanmin(bacc_values),
                "macro_f1": nanmean([_float_or_nan(row.get("macro_f1")) for row in valid]),
                "precision_pos": nanmean([_float_or_nan(row.get("precision_pos")) for row in valid]),
                "recall_pos": recall,
                "f1_pos": nanmean([_float_or_nan(row.get("f1_pos")) for row in valid]),
                "support_pos": "",
                "support_neg": "",
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ci_method": "case_cluster_conservative_seed_aggregate",
                "above_chance": _bool_text(above),
                "near_chance": _bool_text(near),
                "converged": _bool_text(bool(valid) and len(valid) == len(group)),
                "status": VALID_STATUS if valid else "insufficient_valid_seeds",
                "error_message": "" if valid else "no valid seeds",
            }
        )
        out.append(base)
    return out


def _decision_status(valid: Sequence[Mapping[str, object]], *, above: bool, near: bool) -> str:
    if not valid:
        return "INSUFFICIENT_VALID_SPLITS"
    if above:
        return "ABOVE_CHANCE"
    if near:
        return "NEAR_CHANCE"
    return "WEAK_OR_UNSTABLE"


def _attach_preservation_ratios(rows: Sequence[dict[str, object]]) -> None:
    summary = {
        (row["control_name"], row["domain_name"], row["variant_id"], row["representation_role"]): row
        for row in rows
        if row.get("aggregation_level") == "summary"
    }
    for row in rows:
        if row.get("aggregation_level") != "summary":
            continue
        role = str(row.get("representation_role"))
        if role in {REAL_RAW_REFERENCE, REAL_FRAME_REFERENCE} or role in NEGATIVE_ROLES:
            continue
        real = summary.get((row["control_name"], row["domain_name"], row["variant_id"], REAL_FRAME_REFERENCE))
        if real is None:
            continue
        row["preservation_ratio_vs_real_frame"] = _chance_corrected_ratio(row.get("bacc"), real.get("bacc"))


def _gap_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    summary = [row for row in rows if row.get("aggregation_level") == "summary"]
    by_key = {
        (row["control_name"], row["domain_name"], row["variant_id"], row["representation_role"]): row
        for row in summary
    }
    raw_by_control = {
        (row["control_name"], row["domain_name"]): row
        for row in summary
        if row.get("variant_id") == RAW_VARIANT and row.get("representation_role") == REAL_RAW_REFERENCE
    }
    out = []
    for row in summary:
        role = str(row.get("representation_role"))
        if role in {REAL_RAW_REFERENCE, REAL_FRAME_REFERENCE} or role in NEGATIVE_ROLES:
            continue
        raw = raw_by_control.get((row["control_name"], row["domain_name"]))
        frame = by_key.get((row["control_name"], row["domain_name"], row["variant_id"], REAL_FRAME_REFERENCE))
        frame_bacc = _float_or_nan(frame.get("bacc")) if frame is not None else math.nan
        raw_bacc = _float_or_nan(raw.get("bacc")) if raw is not None else math.nan
        variant_bacc = _float_or_nan(row.get("bacc"))
        out.append(
            {
                "control_name": row["control_name"],
                "domain_name": row["domain_name"],
                "variant_id": row["variant_id"],
                "representation_role": role,
                "real_raw_bacc": raw_bacc,
                "real_frame_bacc": frame_bacc,
                "variant_bacc": variant_bacc,
                "pca_frame_preservation_ratio": _chance_corrected_ratio(frame_bacc, raw_bacc),
                "preservation_ratio_vs_real_frame": _chance_corrected_ratio(variant_bacc, frame_bacc),
                "bacc_gap_vs_real_frame": frame_bacc - variant_bacc if _finite(frame_bacc) and _finite(variant_bacc) else math.nan,
                "status": row.get("status", ""),
            }
        )
    return out


def _real_baseline_suitability(
    cfg: MidogPPPreservationSanityConfig,
    metrics: Sequence[Mapping[str, object]],
    negatives: Sequence[Mapping[str, object]],
    prior: Mapping[str, object],
) -> tuple[list[dict[str, object]], bool]:
    rows = []
    balanced = _summary(metrics, BALANCED_CONTROL, "", PRIMARY_VARIANT, REAL_FRAME_REFERENCE)
    balanced_bacc = _float_or_nan(balanced.get("bacc")) if balanced else math.nan
    balanced_ci = _float_or_nan(balanced.get("ci_low")) if balanced else math.nan
    real_negatives = _required_negative_rows(
        negatives,
        variant_id=PRIMARY_VARIANT,
        roles=(REAL_LABEL_PERMUTATION, REAL_FEATURE_ROW_SHUFFLE),
    )
    real_neg_above = [row for row in real_negatives if str(row.get("above_chance")) == "true"]
    real_negatives_ok = (
        bool(real_negatives)
        and all(str(row.get("near_chance")) == "true" for row in real_negatives)
        and not real_neg_above
    )
    within = [
        row
        for row in metrics
        if row.get("aggregation_level") == "summary"
        and row.get("control_name") == WITHIN_TUMOR_CONTROL
        and row.get("variant_id") == PRIMARY_VARIANT
        and row.get("representation_role") == REAL_FRAME_REFERENCE
        and row.get("status") == VALID_STATUS
    ]
    within_above = [row for row in within if row.get("above_chance") == "true"]
    within_fraction = float(len(within_above)) / float(len(within)) if within else 0.0
    prior_shortcut = bool(prior.get("shortcut_suspect"))
    checks = (
        (
            "tumor_balanced_real_frame_bacc",
            balanced_bacc,
            f">= {cfg.real_gate_min_bacc}",
            _finite(balanced_bacc) and balanced_bacc >= cfg.real_gate_min_bacc,
            f"control={BALANCED_CONTROL} variant={PRIMARY_VARIANT}",
        ),
        (
            "tumor_balanced_real_frame_ci_low",
            balanced_ci,
            f"> {cfg.ci_low_threshold}",
            _finite(balanced_ci) and balanced_ci > cfg.ci_low_threshold,
            f"control={BALANCED_CONTROL} variant={PRIMARY_VARIANT}",
        ),
        (
            "negative_controls_near_chance",
            len(real_negatives),
            "required real-frame negatives present and near chance",
            real_negatives_ok,
            "|".join(
                f"{row.get('representation_role')}={row.get('near_chance')}"
                for row in real_negatives
            )
            or "|".join(str(row.get("representation_role")) for row in real_neg_above[:10]),
        ),
        (
            "within_tumor_real_frame_above_fraction",
            within_fraction,
            f">= {cfg.within_tumor_min_above_fraction}",
            bool(within) and within_fraction >= cfg.within_tumor_min_above_fraction,
            f"{len(within_above)}/{len(within)} valid within-tumor summaries above chance",
        ),
        (
            "prior_signal_control_not_shortcut_suspect",
            int(prior_shortcut),
            "0",
            not prior_shortcut,
            str(prior.get("path", "")),
        ),
    )
    for criterion, value, threshold, passed, evidence in checks:
        rows.append(
            {
                "criterion": criterion,
                "value": value,
                "threshold": threshold,
                "passed": _bool_text(bool(passed)),
                "evidence": evidence,
            }
        )
    return rows, all(row["passed"] == "true" for row in rows)


def _decision_labels(
    cfg: MidogPPPreservationSanityConfig,
    metrics: Sequence[Mapping[str, object]],
    negatives: Sequence[Mapping[str, object]],
    leakage: Mapping[str, object],
    real_suitable: bool,
    prior: Mapping[str, object],
) -> list[str]:
    if leakage.get("status") != "PASS" or _negative_above_chance(negatives):
        return ["LEAKAGE_OR_ALIGNMENT_FAILURE_SUSPECT"]
    labels = []
    if not real_suitable:
        labels.append("REAL_FEATURE_SHORTCUT_BASELINE_UNSUITABLE_FOR_CVAE_GATE")
    primary = _summary(metrics, BALANCED_CONTROL, "", PRIMARY_VARIANT, DECODE_MU)
    real_frame = _summary(metrics, BALANCED_CONTROL, "", PRIMARY_VARIANT, REAL_FRAME_REFERENCE)
    real_raw = _summary(metrics, BALANCED_CONTROL, "", RAW_VARIANT, REAL_RAW_REFERENCE)
    if primary is None or real_frame is None:
        labels.append("INSUFFICIENT_VALID_SPLITS")
        return labels
    required_negatives_near = _required_primary_negatives_near(negatives)
    if not required_negatives_near:
        labels.append("INSUFFICIENT_VALID_SPLITS")
        return labels
    decode_bacc = _float_or_nan(primary.get("bacc"))
    decode_ci = _float_or_nan(primary.get("ci_low"))
    real_bacc = _float_or_nan(real_frame.get("bacc"))
    ratio = _chance_corrected_ratio(decode_bacc, real_bacc)
    raw_bacc = _float_or_nan(real_raw.get("bacc")) if real_raw else math.nan
    frame_ratio = _chance_corrected_ratio(real_bacc, raw_bacc)
    if _finite(frame_ratio) and frame_ratio < cfg.preservation_pass_ratio:
        labels.append("PCA_FRAME_BOTTLENECK")
    decode_pass = (
        real_suitable
        and _finite(decode_bacc)
        and _finite(decode_ci)
        and decode_bacc >= cfg.cvae_gate_min_bacc
        and decode_ci > cfg.ci_low_threshold
        and _finite(ratio)
        and ratio >= cfg.preservation_pass_ratio
        and required_negatives_near
    )
    if decode_pass:
        labels.append("CVAE_PRESERVATION_SANITY_PASS")
        if ratio >= cfg.preservation_strong_ratio:
            labels.append("CVAE_PRESERVATION_STRONG")
    elif real_suitable:
        labels.append("CVAE_RECONSTRUCTION_BOTTLENECK")
    posterior = _summary(metrics, BALANCED_CONTROL, "", PRIMARY_VARIANT, POSTERIOR_SAMPLE)
    prior_row = _summary(metrics, BALANCED_CONTROL, "", PRIMARY_VARIANT, PRIOR_SAMPLE)
    sampling_ratios = [
        _chance_corrected_ratio(row.get("bacc"), real_bacc)
        for row in (posterior, prior_row)
        if row is not None
    ]
    if decode_pass and any(_finite(value) and value < cfg.preservation_pass_ratio for value in sampling_ratios):
        labels.append("CVAE_SAMPLING_BOTTLENECK")
    if bool(prior.get("shortcut_suspect")) and "REAL_FEATURE_SHORTCUT_BASELINE_UNSUITABLE_FOR_CVAE_GATE" not in labels:
        labels.append("REAL_FEATURE_SHORTCUT_BASELINE_UNSUITABLE_FOR_CVAE_GATE")
    return labels or ["INSUFFICIENT_VALID_SPLITS"]


def _summary(
    rows: Sequence[Mapping[str, object]],
    control: str,
    domain: str,
    variant: str,
    role: str,
) -> Mapping[str, object] | None:
    for row in rows:
        if (
            row.get("aggregation_level") == "summary"
            and row.get("control_name") == control
            and row.get("domain_name") == domain
            and row.get("variant_id") == variant
            and row.get("representation_role") == role
        ):
            return row
    return None


def _negative_above_chance(rows: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    return [
        row
        for row in rows
        if row.get("aggregation_level") == "summary"
        and str(row.get("above_chance")) == "true"
        and str(row.get("representation_role")) in NEGATIVE_ROLES
    ]


def _required_primary_negatives_near(rows: Sequence[Mapping[str, object]]) -> bool:
    required = _required_negative_rows(
        rows,
        variant_id=PRIMARY_VARIANT,
        roles=(
            REAL_LABEL_PERMUTATION,
            REAL_FEATURE_ROW_SHUFFLE,
            CVAE_CONDITION_LABEL_PERMUTATION,
            CVAE_DECODED_ROW_SHUFFLE,
        ),
    )
    return len(required) == 4 and all(str(row.get("near_chance")) == "true" for row in required)


def _required_negative_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    variant_id: str,
    roles: Sequence[str],
) -> list[Mapping[str, object]]:
    out = []
    for role in roles:
        row = next(
            (
                item
                for item in rows
                if item.get("aggregation_level") == "summary"
                and item.get("variant_id") == variant_id
                and item.get("representation_role") == role
            ),
            None,
        )
        if row is not None:
            out.append(row)
    return out


def _leakage_report(
    audit_rows: Sequence[Mapping[str, object]],
    negative_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    overlap_failures = [row for row in audit_rows if int(row.get("overlap_count", 0)) > 0]
    negative_above = _negative_above_chance(negative_rows)
    extra = []
    if overlap_failures:
        extra.append("identity_overlap_failure")
    if negative_above:
        extra.append("negative_control_above_chance")
    report = build_leakage_report(
        target_support_labels_for_selection=False,
        target_eval_labels_for_scoring_only=True,
        target_expert_excluded=True,
        oracle_rows_diagnostic_only=True,
        extra_violations=extra,
    ).to_json_dict()
    report.update(
        {
            "identity_overlap_failure_count": len(overlap_failures),
            "negative_controls_above_chance_count": len(negative_above),
            "fit_only_pca": True,
            "fit_only_cvae_training": True,
            "fixed_threshold": 0.5,
            "metadata_success_signal": False,
        }
    )
    return report


def _protocol_manifest(
    cfg: MidogPPPreservationSanityConfig,
    cache: FeatureCache,
    rows: Sequence[ManifestRow],
    splits: Sequence[SplitSpec],
    prior: Mapping[str, object],
) -> dict[str, object]:
    feature_dim = int(getattr(cache.embeddings, "shape", [0, 0])[1])
    if feature_dim == 0:
        feature_dim = int(_to_numpy(cache.embeddings).shape[1])
    return {
        "schema_version": "midogpp_virchow2_cvae_preservation_sanity_protocol_v1",
        "experiment_name": cfg.name,
        "manifest_path": str(cfg.manifest_path),
        "feature_cache_path": str(cfg.feature_cache_path),
        "signal_split_manifest_path": str(cfg.signal_split_manifest_path),
        "train_rows": int(len(rows)),
        "split_count": int(len(splits)),
        "feature_dim": int(feature_dim),
        "feature_extractor": dict(cache.feature_extractor),
        "positive_label": int(cfg.positive_label),
        "cache_building_in_scope": False,
        "split_scope": "train_cache_signal_controls",
        "pca_scaler_fit_scope": "fit_rows_only",
        "cvae_training_scope": "fit_rows_only",
        "eval_labels_role": "scoring_only",
        "threshold_policy": "fixed_0.5_classifier_rule_not_calibrated_probability",
        "primary_gate": f"{PRIMARY_VARIANT}+{DECODE_MU}",
        "diagnostic_variants": [variant.variant_id for variant in cfg.variants if variant.variant_id != cfg.primary_variant],
        "prior_signal_control_status": dict(prior),
        "claim_boundary": {
            "allowed": "CVAE preservation mechanics under MIDOG++ signal-control splits if real comparator is suitable.",
            "forbidden": "No routing, expert-selection, metadata-compatibility, or multiaxis LODO CVAE preservation claim.",
        },
    }


def _write_decision_report(
    path: Path,
    cfg: MidogPPPreservationSanityConfig,
    decision_labels: Sequence[str],
    leakage: Mapping[str, object],
    suitability_rows: Sequence[Mapping[str, object]],
    metrics: Sequence[Mapping[str, object]],
    negatives: Sequence[Mapping[str, object]],
    prior: Mapping[str, object],
) -> None:
    del cfg
    lines = [
        "# MIDOG++ Virchow2 CVAE Preservation Sanity Check",
        "",
        f"- Decision labels: `{', '.join(decision_labels)}`",
        f"- Leakage status: `{leakage.get('status')}`",
        f"- Prior signal-control shortcut suspect: `{_bool_text(bool(prior.get('shortcut_suspect')))}`",
        "- Eval labels are scoring-only; CVAE and PCA are fit on fit rows only.",
        "- This is not routing or expert selection.",
        "",
        "## Real Baseline Suitability",
        "",
        "| Criterion | Value | Threshold | Passed |",
        "| --- | ---: | --- | --- |",
    ]
    for row in suitability_rows:
        lines.append(f"| {row['criterion']} | {_fmt(row['value'])} | {row['threshold']} | {row['passed']} |")
    lines.extend(
        [
            "",
            "## Primary Rows",
            "",
            "| Control | Variant | Role | BACC | CI low | Ratio | Status |",
            "| --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for role in (REAL_RAW_REFERENCE, REAL_FRAME_REFERENCE, DECODE_MU, POSTERIOR_SAMPLE, PRIOR_SAMPLE):
        variant = RAW_VARIANT if role == REAL_RAW_REFERENCE else PRIMARY_VARIANT
        row = _summary(metrics, BALANCED_CONTROL, "", variant, role)
        if row is not None:
            lines.append(
                f"| {BALANCED_CONTROL} | {variant} | {role} | {_fmt(row.get('bacc'))} | "
                f"{_fmt(row.get('ci_low'))} | {_fmt(row.get('preservation_ratio_vs_real_frame'))} | {row.get('status')} |"
            )
    lines.extend(["", "## Negative Controls", "", "| Role | Variant | BACC | CI low | Above chance |", "| --- | --- | ---: | ---: | --- |"])
    for row in negatives:
        if row.get("aggregation_level") == "summary":
            lines.append(
                f"| {row.get('representation_role')} | {row.get('variant_id')} | {_fmt(row.get('bacc'))} | "
                f"{_fmt(row.get('ci_low'))} | {row.get('above_chance')} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _real_frame_negative_inputs(
    cfg: MidogPPPreservationSanityConfig,
    spec: SplitSpec,
    fit_x: object,
    fit_y: Sequence[int],
) -> tuple[tuple[str, object, object], ...]:
    import numpy as np  # type: ignore

    x = np.asarray(fit_x, dtype=float)
    y = np.asarray(fit_y, dtype=int)
    rng = _rng(_stable_seed(spec, "real_negative", cfg.bootstrap_seed))
    return (
        (REAL_LABEL_PERMUTATION, x, rng.permutation(y)),
        (REAL_FEATURE_ROW_SHUFFLE, x[rng.permutation(len(x))], y),
    )


def _permuted_labels(
    cfg: MidogPPPreservationSanityConfig,
    spec: SplitSpec,
    fit_y: Sequence[int],
    salt: str,
) -> Any:
    import numpy as np  # type: ignore

    return np.asarray(_rng(_stable_seed(spec, salt, cfg.bootstrap_seed)).permutation(np.asarray(fit_y, dtype=int)), dtype=int)


def _reconstruction_row(
    spec: SplitSpec,
    variant: VariantConfig,
    role: str,
    diagnostics: Mapping[str, float],
    *,
    n_rows: int,
) -> dict[str, object]:
    return {
        "control_name": spec.control_name,
        "domain_name": spec.domain_name,
        "split_seed": int(spec.split_seed),
        "variant_id": variant.variant_id,
        "representation_role": role,
        "n_rows": int(n_rows),
        **dict(diagnostics),
        "status": VALID_STATUS,
    }


def _model_manifest_row(spec: SplitSpec, runtime: CVAERuntime, rows: Sequence[ManifestRow]) -> dict[str, object]:
    return {
        "control_name": spec.control_name,
        "domain_name": spec.domain_name,
        "split_seed": int(spec.split_seed),
        "variant_id": runtime.variant.variant_id,
        "condition_mode": runtime.condition_mode,
        "input_dim": int(runtime.fit_x.shape[1]),
        "effective_pca_dim": int(runtime.frame.effective_dim),
        "latent_dim": int(runtime.variant.latent_dim),
        "hidden_dim": int(runtime.variant.hidden_dim),
        "train_epochs": int(runtime.variant.train_epochs),
        "beta_final": float(runtime.variant.beta_final),
        "kl_warmup_epochs": int(runtime.variant.kl_warmup_epochs),
        "n_fit": int(len(spec.fit_idx)),
        "fit_sample_hash": _hash_strings(rows[idx].sample_id for idx in spec.fit_idx),
    }


def _prior_signal_status(path: Path | None) -> dict[str, object]:
    if path is None:
        return {"provided": False, "shortcut_suspect": False, "path": ""}
    if not path.exists():
        return {"provided": False, "shortcut_suspect": False, "path": str(path)}
    text = path.read_text(encoding="utf-8")
    return {
        "provided": True,
        "shortcut_suspect": "POOLED_SIGNAL_DOMAIN_SHORTCUT_SUSPECT" in text,
        "path": str(path),
    }


def _above_chance(bacc: object, ci_low: object, cfg: MidogPPPreservationSanityConfig) -> bool:
    return _finite(bacc) and _finite(ci_low) and float(bacc) >= cfg.real_gate_min_bacc and float(ci_low) > cfg.ci_low_threshold


def _near_chance(ci_low: object, ci_high: object, recall_pos: object) -> bool:
    return (
        _finite(ci_low)
        and _finite(ci_high)
        and float(ci_low) <= 0.50 <= float(ci_high)
    ) or (_finite(recall_pos) and float(recall_pos) <= 0.0)


def _chance_corrected_ratio(value: object, reference: object) -> float:
    v = _float_or_nan(value)
    ref = _float_or_nan(reference)
    denom = ref - 0.5
    if not _finite(v) or not _finite(ref) or denom <= 0.0:
        return math.nan
    return float((v - 0.5) / denom)


def _stable_seed(*parts: object) -> int:
    text = "|".join(_seed_part(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _seed_part(part: object) -> str:
    if isinstance(part, SplitSpec):
        return f"{part.control_name}:{part.domain_name}:{part.split_seed}"
    return str(part)


def _rng(seed: int) -> Any:
    import numpy as np  # type: ignore

    return np.random.default_rng(int(seed))


def _hash_strings(values: Sequence[str] | Any) -> str:
    h = hashlib.sha256()
    for value in values:
        h.update(str(value).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _float_or_nan(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _nanmin(values: Sequence[object]) -> float:
    vals = [_float_or_nan(value) for value in values if _finite(value)]
    return min(vals) if vals else math.nan


def _nanmax(values: Sequence[object]) -> float:
    vals = [_float_or_nan(value) for value in values if _finite(value)]
    return max(vals) if vals else math.nan


def _nanstd(values: Sequence[object]) -> float:
    import numpy as np  # type: ignore

    vals = [_float_or_nan(value) for value in values if _finite(value)]
    return float(np.std(vals, ddof=0)) if vals else math.nan


def _bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def _fmt(value: object) -> str:
    return f"{float(value):.4f}" if _finite(value) else "NA"
