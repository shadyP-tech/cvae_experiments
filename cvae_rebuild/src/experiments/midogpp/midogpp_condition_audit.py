from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.midogpp.midogpp_preservation_sanity import (
    BALANCED_CONTROL,
    CONTROL_ORDER,
    DEFAULT_FEATURE_CACHE_PATH,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_SIGNAL_SPLIT_MANIFEST_PATH,
    LOGISTIC_MODEL_TYPE,
    VALID_STATUS,
    WITHIN_TUMOR_CONTROL,
    CVAERuntime,
    FeatureCache,
    ManifestRow,
    SplitSpec,
    VariantConfig,
    _aggregate_rows,
    _assert_cache_alignment,
    _assert_virchow2_cache,
    _bool_text,
    _chance_corrected_ratio,
    _evaluate_representation,
    _finite,
    _float_or_nan,
    _fmt,
    _hash_strings,
    _identity_audit_rows,
    _load_feature_cache,
    _mapping,
    _path,
    _read_split_manifest,
    _read_train_manifest,
    _reconstruction_diagnostics,
    _permuted_labels,
    _split_counts,
    _split_status,
    _to_numpy,
    _train_runtime,
)
from core.protocol import ProtocolError, build_leakage_report
from core.reporting import prepare_artifact_dirs, write_csv_rows, write_json


EXPERIMENT_NAME = "virchow2_cvae_midogpp_preservation_condition_audit_v1"
DEFAULT_ARTIFACT_ROOT = "cvae_rebuild/artifacts/midogpp/virchow2_cvae_midogpp_preservation_condition_audit_v1"

FULL_DIM_VARIANT = "full_dim"
REAL_FULL_DIM_REFERENCE = "real_full_dim_reference"
REAL_PCA64_REFERENCE = "real_pca64_reference"
REAL_PCA128_REFERENCE = "real_pca128_reference"
REAL_PCA256_REFERENCE = "real_pca256_reference"
PCA_REFERENCE_BY_DIM = {
    64: REAL_PCA64_REFERENCE,
    128: REAL_PCA128_REFERENCE,
    256: REAL_PCA256_REFERENCE,
}

TRUE_TRAIN_TRUE_ENCODE_TRUE_DECODE = "true_train_true_encode_true_decode"
PERMUTED_TRAIN_TRUE_ENCODE_TRUE_DECODE = "permuted_train_true_encode_true_decode"
TRUE_TRAIN_TRUE_ENCODE_PERMUTED_DECODE = "true_train_true_encode_permuted_decode"
PERMUTED_TRAIN_PERMUTED_ENCODE_PERMUTED_DECODE = "permuted_train_permuted_encode_permuted_decode"
CONDITION_ROW_IDS = (
    TRUE_TRAIN_TRUE_ENCODE_TRUE_DECODE,
    PERMUTED_TRAIN_TRUE_ENCODE_TRUE_DECODE,
    TRUE_TRAIN_TRUE_ENCODE_PERMUTED_DECODE,
    PERMUTED_TRAIN_PERMUTED_ENCODE_PERMUTED_DECODE,
)

TRUE_LABELS = "true"
PERMUTED_LABELS = "permuted"
PERMUTED_TRAIN_CONDITION_MODE = "permuted_train_labels"

METRIC_COLUMNS = (
    "aggregation_level",
    "control_name",
    "domain_name",
    "split_seed",
    "variant_id",
    "representation_role",
    "condition_row_id",
    "train_condition_labels",
    "encode_condition_labels",
    "decode_condition_labels",
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
    "chance_corrected_ratio_vs_true_condition",
)

RECON_COLUMNS = (
    "control_name",
    "domain_name",
    "split_seed",
    "variant_id",
    "representation_role",
    "condition_row_id",
    "train_condition_labels",
    "encode_condition_labels",
    "decode_condition_labels",
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
    "train_condition_labels",
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
    "condition_row_id",
    "train_condition_labels",
    "encode_condition_labels",
    "decode_condition_labels",
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
    "train_condition_labels",
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

DECODER_AUDIT_COLUMNS = (
    "aggregation_level",
    "control_name",
    "domain_name",
    "split_seed",
    "variant_id",
    "n_rows",
    "label_swap_l2",
    "label_swap_mse",
    "label_swap_cosine_distance",
    "real_class_centroid_l2",
    "decoded_class_centroid_l2",
    "reconstruction_mse",
    "label_swap_l2_over_real_class_centroid_l2",
    "label_swap_mse_over_reconstruction_mse",
    "decoded_class_centroid_l2_over_real_class_centroid_l2",
    "weak_conditioning",
    "status",
)


@dataclass(frozen=True)
class MidogPPConditionAuditConfig:
    name: str
    artifact_root: Path
    manifest_path: Path
    feature_cache_path: Path
    signal_split_manifest_path: Path
    positive_label: int
    controls: tuple[str, ...]
    variants: tuple[VariantConfig, ...]
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
    ci_low_threshold: float
    close_bacc_delta: float
    close_ratio_threshold: float
    weak_conditioning_ratio_threshold: float


def load_midogpp_condition_audit_config(path: str | Path) -> MidogPPConditionAuditConfig:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise ProtocolError("Loading MIDOG++ condition audit configs requires PyYAML.") from exc
    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ProtocolError(f"Config must be a mapping: {path}")
    base_dir = source.parents[2] if len(source.parents) >= 3 else source.parent
    return parse_midogpp_condition_audit_config(payload, base_dir=base_dir)


def parse_midogpp_condition_audit_config(
    data: Mapping[str, object],
    *,
    base_dir: str | Path = ".",
) -> MidogPPConditionAuditConfig:
    base = Path(base_dir)
    experiment = _mapping(data.get("experiment"), "experiment")
    inputs = _mapping(data.get("inputs"), "inputs")
    run = _mapping(data.get("run"), "run", allow_empty=True)
    bootstrap = _mapping(data.get("bootstrap"), "bootstrap", allow_empty=True)
    thresholds = _mapping(data.get("validity_thresholds"), "validity_thresholds", allow_empty=True)
    decisions = _mapping(data.get("decision_thresholds"), "decision_thresholds", allow_empty=True)
    variants_payload = data.get("variants") or []
    variants = tuple(_parse_variant(item) for item in variants_payload) if variants_payload else _default_variants()
    cfg = MidogPPConditionAuditConfig(
        name=str(experiment.get("name", "")),
        artifact_root=_path(base, str(experiment.get("artifact_root", DEFAULT_ARTIFACT_ROOT))),
        manifest_path=_path(base, str(inputs.get("manifest_path", DEFAULT_MANIFEST_PATH))),
        feature_cache_path=_path(base, str(inputs.get("feature_cache_path", DEFAULT_FEATURE_CACHE_PATH))),
        signal_split_manifest_path=_path(
            base,
            str(inputs.get("signal_split_manifest_path", DEFAULT_SIGNAL_SPLIT_MANIFEST_PATH)),
        ),
        positive_label=int(data.get("positive_label", 1)),
        controls=tuple(str(v) for v in run.get("controls", CONTROL_ORDER)),
        variants=variants,
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
        ci_low_threshold=float(decisions.get("ci_low_threshold", 0.50)),
        close_bacc_delta=float(decisions.get("close_bacc_delta", 0.03)),
        close_ratio_threshold=float(decisions.get("close_ratio_threshold", 0.90)),
        weak_conditioning_ratio_threshold=float(decisions.get("weak_conditioning_ratio_threshold", 0.25)),
    )
    validate_midogpp_condition_audit_config(cfg)
    return cfg


def validate_midogpp_condition_audit_config(cfg: MidogPPConditionAuditConfig) -> None:
    if cfg.name != EXPERIMENT_NAME:
        raise ProtocolError(f"Unexpected experiment.name={cfg.name!r}; expected {EXPERIMENT_NAME!r}.")
    unknown_controls = sorted(set(cfg.controls) - set(CONTROL_ORDER))
    if unknown_controls:
        raise ProtocolError(f"Unknown controls: {unknown_controls}")
    expected = {
        "pca64_beta001": (64, 16),
        "pca128_beta001": (128, 32),
        "pca256_beta001": (256, 64),
    }
    seen = {variant.variant_id: variant for variant in cfg.variants}
    missing = sorted(set(expected) - set(seen))
    if missing:
        raise ProtocolError(f"Condition audit variants missing required diagnostic variants: {missing}")
    for variant_id, (pca_dim, latent_dim) in expected.items():
        variant = seen[variant_id]
        if variant.pca_dim != pca_dim or variant.latent_dim != latent_dim:
            raise ProtocolError(f"{variant_id} must use pca_dim={pca_dim} and latent_dim={latent_dim}.")
    for variant in cfg.variants:
        if variant.hidden_dim != 512 or variant.num_hidden_layers != 2:
            raise ProtocolError("CVAE architecture is locked to hidden_dim=512 and two hidden layers.")
        if not math.isclose(float(variant.beta_final), 0.001):
            raise ProtocolError("CVAE beta_final is locked to 0.001.")
    if any("PASS" in condition for condition in CONDITION_ROW_IDS):
        raise ProtocolError("Condition rows must not encode preservation PASS semantics.")


def run_midogpp_condition_audit(
    cfg: MidogPPConditionAuditConfig,
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

    pca_rows: list[dict[str, object]] = []
    condition_rows: list[dict[str, object]] = []
    reconstruction_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    model_manifest_rows: list[dict[str, object]] = []
    decoder_rows: list[dict[str, object]] = []

    for spec in split_specs:
        audit = _identity_audit_rows(manifest_rows, spec)
        audit_rows.extend(audit)
        status, error = _split_status(cfg, manifest_rows, spec, audit)
        counts = _split_counts(manifest_rows, spec.fit_idx, spec.eval_idx)
        if status != VALID_STATUS:
            empty = _metric_context(
                _empty_metric(spec, FULL_DIM_VARIANT, REAL_FULL_DIM_REFERENCE, counts, status, error),
                condition_row_id="",
                train_condition_labels="",
                encode_condition_labels="",
                decode_condition_labels="",
            )
            pca_rows.append(empty)
            continue

        x_fit_raw = embeddings[list(spec.fit_idx)]
        y_fit = np.asarray([manifest_rows[idx].label for idx in spec.fit_idx], dtype=int)
        x_eval_raw = embeddings[list(spec.eval_idx)]
        y_eval = [manifest_rows[idx].label for idx in spec.eval_idx]

        full_row, full_preds = _evaluate_representation(
            cfg,
            spec,
            rows=manifest_rows,
            x_fit=x_fit_raw,
            y_fit=y_fit,
            x_eval=x_eval_raw,
            y_eval=y_eval,
            variant_id=FULL_DIM_VARIANT,
            representation_role=REAL_FULL_DIM_REFERENCE,
            counts=counts,
            prediction_prefix="",
        )
        full_row = _metric_context(
            full_row,
            condition_row_id="",
            train_condition_labels="",
            encode_condition_labels="",
            decode_condition_labels="",
        )
        pca_rows.append(full_row)
        prediction_rows.extend(
            _prediction_context(
                full_preds,
                condition_row_id="",
                train_condition_labels="",
                encode_condition_labels="",
                decode_condition_labels="",
            )
        )

        for variant in cfg.variants:
            true_runtime = _train_runtime(
                cfg,
                spec,
                variant,
                x_fit_raw=x_fit_raw,
                x_eval_raw=x_eval_raw,
                y_fit=y_fit,
                condition_mode="real_labels",
            )
            training_rows.extend(_training_context(true_runtime.training_rows, TRUE_LABELS))
            model_manifest_rows.append(_model_manifest_row(spec, true_runtime, manifest_rows, TRUE_LABELS))

            pca_role = PCA_REFERENCE_BY_DIM.get(int(variant.pca_dim), f"real_pca{variant.pca_dim}_reference")
            pca_row, pca_preds = _evaluate_representation(
                cfg,
                spec,
                rows=manifest_rows,
                x_fit=true_runtime.fit_x,
                y_fit=y_fit,
                x_eval=true_runtime.eval_x,
                y_eval=y_eval,
                variant_id=variant.variant_id,
                representation_role=pca_role,
                counts=counts,
                prediction_prefix="",
            )
            pca_row = _metric_context(
                pca_row,
                condition_row_id="",
                train_condition_labels="",
                encode_condition_labels="",
                decode_condition_labels="",
            )
            pca_rows.append(pca_row)
            prediction_rows.extend(
                _prediction_context(
                    pca_preds,
                    condition_row_id="",
                    train_condition_labels="",
                    encode_condition_labels="",
                    decode_condition_labels="",
                )
            )

            permuted_fit_labels = _permuted_condition_labels(cfg, spec, variant, y_fit)
            permuted_runtime = _train_runtime(
                cfg,
                spec,
                variant,
                x_fit_raw=x_fit_raw,
                x_eval_raw=x_eval_raw,
                y_fit=y_fit,
                condition_mode=PERMUTED_TRAIN_CONDITION_MODE,
            )
            training_rows.extend(_training_context(permuted_runtime.training_rows, PERMUTED_LABELS))
            model_manifest_rows.append(_model_manifest_row(spec, permuted_runtime, manifest_rows, PERMUTED_LABELS))

            for condition in _condition_rows(true_runtime, permuted_runtime, y_fit, permuted_fit_labels):
                decoded, recon = _decode_with_condition_labels(
                    condition["runtime"],
                    condition["runtime"].fit_x,
                    condition["encode_labels"],
                    condition["decode_labels"],
                )
                recon_row = _reconstruction_row(
                    spec,
                    condition["runtime"].variant,
                    str(condition["condition_row_id"]),
                    str(condition["train_condition_labels"]),
                    str(condition["encode_condition_labels"]),
                    str(condition["decode_condition_labels"]),
                    recon,
                    n_rows=len(y_fit),
                )
                reconstruction_rows.append(recon_row)
                metric_row, preds = _evaluate_representation(
                    cfg,
                    spec,
                    rows=manifest_rows,
                    x_fit=decoded,
                    y_fit=y_fit,
                    x_eval=condition["runtime"].eval_x,
                    y_eval=y_eval,
                    variant_id=condition["runtime"].variant.variant_id,
                    representation_role=str(condition["condition_row_id"]),
                    counts=counts,
                    prediction_prefix="",
                )
                metric_row = _metric_context(
                    metric_row,
                    condition_row_id=str(condition["condition_row_id"]),
                    train_condition_labels=str(condition["train_condition_labels"]),
                    encode_condition_labels=str(condition["encode_condition_labels"]),
                    decode_condition_labels=str(condition["decode_condition_labels"]),
                )
                condition_rows.append(metric_row)
                prediction_rows.extend(
                    _prediction_context(
                        preds,
                        condition_row_id=str(condition["condition_row_id"]),
                        train_condition_labels=str(condition["train_condition_labels"]),
                        encode_condition_labels=str(condition["encode_condition_labels"]),
                        decode_condition_labels=str(condition["decode_condition_labels"]),
                    )
                )

            decoder_rows.append(_decoder_label_conditioning_row(cfg, spec, true_runtime, y_fit))

    pca_summary = _aggregate_rows(pca_rows)
    condition_summary = _aggregate_rows(condition_rows)
    decoder_summary = _aggregate_decoder_rows(cfg, decoder_rows)
    pca_all = _normalize_metric_rows(pca_rows + pca_summary)
    condition_all = _normalize_metric_rows(condition_rows + condition_summary)
    _attach_condition_ratios(condition_all)
    decoder_all = decoder_rows + decoder_summary
    leakage = _leakage_report(audit_rows)
    decision_labels = _decision_labels(cfg, pca_all, condition_all, decoder_all, leakage)
    downstream_rows = pca_all + condition_all

    write_csv_rows(root / "tables" / "pca_capacity_audit.csv", pca_all, METRIC_COLUMNS)
    write_csv_rows(root / "tables" / "condition_permutation_matrix.csv", condition_all, METRIC_COLUMNS)
    write_csv_rows(root / "tables" / "decoder_label_conditioning_audit.csv", decoder_all, DECODER_AUDIT_COLUMNS)
    write_csv_rows(root / "tables" / "downstream_condition_metrics.csv", downstream_rows, METRIC_COLUMNS)
    write_csv_rows(root / "tables" / "reconstruction_diagnostics.csv", reconstruction_rows, RECON_COLUMNS)
    write_csv_rows(root / "tables" / "training_diagnostics.csv", training_rows, TRAINING_COLUMNS)
    write_csv_rows(root / "tables" / "identity_overlap_audit.csv", audit_rows, AUDIT_COLUMNS)
    write_csv_rows(root / "tables" / "predictions.csv", prediction_rows, PREDICTION_COLUMNS)
    write_csv_rows(root / "manifests" / "model_manifest.csv", model_manifest_rows, MODEL_MANIFEST_COLUMNS)
    write_json(root / "manifests" / "protocol_manifest.json", _protocol_manifest(cfg, cache, manifest_rows, split_specs))
    write_json(root / "reports" / "leakage_report.json", leakage)
    _write_decision_report(root / "reports" / "decision_report.md", cfg, decision_labels, leakage, pca_all, condition_all, decoder_all)
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
        VariantConfig("pca64_beta001", pca_dim=64, latent_dim=16),
        VariantConfig("pca128_beta001", pca_dim=128, latent_dim=32),
        VariantConfig("pca256_beta001", pca_dim=256, latent_dim=64),
    )


def _empty_metric(
    spec: SplitSpec,
    variant_id: str,
    role: str,
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
        "representation_role": role,
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
        "chance_corrected_ratio_vs_true_condition": "",
    }


def _metric_context(
    row: Mapping[str, object],
    *,
    condition_row_id: str,
    train_condition_labels: str,
    encode_condition_labels: str,
    decode_condition_labels: str,
) -> dict[str, object]:
    out = dict(row)
    out.update(
        {
            "condition_row_id": condition_row_id,
            "train_condition_labels": train_condition_labels,
            "encode_condition_labels": encode_condition_labels,
            "decode_condition_labels": decode_condition_labels,
            "chance_corrected_ratio_vs_true_condition": out.get("chance_corrected_ratio_vs_true_condition", ""),
        }
    )
    return out


def _prediction_context(
    rows: Sequence[Mapping[str, object]],
    *,
    condition_row_id: str,
    train_condition_labels: str,
    encode_condition_labels: str,
    decode_condition_labels: str,
) -> list[dict[str, object]]:
    out = []
    for row in rows:
        item = dict(row)
        item.update(
            {
                "condition_row_id": condition_row_id,
                "train_condition_labels": train_condition_labels,
                "encode_condition_labels": encode_condition_labels,
                "decode_condition_labels": decode_condition_labels,
            }
        )
        out.append(item)
    return out


def _training_context(rows: Sequence[Mapping[str, object]], train_condition_labels: str) -> list[dict[str, object]]:
    out = []
    for row in rows:
        item = dict(row)
        item["train_condition_labels"] = train_condition_labels
        item["condition_mode"] = train_condition_labels
        out.append(item)
    return out


def _condition_rows(
    true_runtime: CVAERuntime,
    permuted_runtime: CVAERuntime,
    y_fit: Sequence[int],
    permuted_labels: Sequence[int],
) -> tuple[dict[str, object], ...]:
    y_true = tuple(int(value) for value in y_fit)
    y_perm = tuple(int(value) for value in permuted_labels)
    return (
        {
            "condition_row_id": TRUE_TRAIN_TRUE_ENCODE_TRUE_DECODE,
            "runtime": true_runtime,
            "train_condition_labels": TRUE_LABELS,
            "encode_condition_labels": TRUE_LABELS,
            "decode_condition_labels": TRUE_LABELS,
            "encode_labels": y_true,
            "decode_labels": y_true,
        },
        {
            "condition_row_id": PERMUTED_TRAIN_TRUE_ENCODE_TRUE_DECODE,
            "runtime": permuted_runtime,
            "train_condition_labels": PERMUTED_LABELS,
            "encode_condition_labels": TRUE_LABELS,
            "decode_condition_labels": TRUE_LABELS,
            "encode_labels": y_true,
            "decode_labels": y_true,
        },
        {
            "condition_row_id": TRUE_TRAIN_TRUE_ENCODE_PERMUTED_DECODE,
            "runtime": true_runtime,
            "train_condition_labels": TRUE_LABELS,
            "encode_condition_labels": TRUE_LABELS,
            "decode_condition_labels": PERMUTED_LABELS,
            "encode_labels": y_true,
            "decode_labels": y_perm,
        },
        {
            "condition_row_id": PERMUTED_TRAIN_PERMUTED_ENCODE_PERMUTED_DECODE,
            "runtime": permuted_runtime,
            "train_condition_labels": PERMUTED_LABELS,
            "encode_condition_labels": PERMUTED_LABELS,
            "decode_condition_labels": PERMUTED_LABELS,
            "encode_labels": y_perm,
            "decode_labels": y_perm,
        },
    )


def _permuted_condition_labels(
    cfg: MidogPPConditionAuditConfig,
    spec: SplitSpec,
    variant: VariantConfig,
    y_fit: Sequence[int],
) -> Any:
    del variant
    return _permuted_labels(cfg, spec, y_fit, PERMUTED_TRAIN_CONDITION_MODE)


def _decode_with_condition_labels(
    runtime: CVAERuntime,
    x: object,
    encode_labels: Sequence[int],
    decode_labels: Sequence[int],
) -> tuple[object, dict[str, float]]:
    import numpy as np  # type: ignore
    import torch  # type: ignore

    x_np = np.asarray(x, dtype=np.float32)
    encode_np = np.asarray(encode_labels, dtype=np.int64)
    decode_np = np.asarray(decode_labels, dtype=np.int64)
    if len(encode_np) != len(x_np) or len(decode_np) != len(x_np):
        raise ProtocolError("Condition label vectors must align with fit rows.")
    with torch.no_grad():
        xt = torch.as_tensor(x_np, dtype=torch.float32)
        encode_t = torch.as_tensor(encode_np, dtype=torch.long)
        decode_t = torch.as_tensor(decode_np, dtype=torch.long)
        mu, logvar = runtime.model.encode(xt, encode_t)
        decoded = runtime.model.decode(mu, decode_t).detach().cpu().numpy()
    return decoded, _reconstruction_diagnostics(x_np, decoded, mu, logvar)


def _decoder_label_conditioning_row(
    cfg: MidogPPConditionAuditConfig,
    spec: SplitSpec,
    runtime: CVAERuntime,
    y_fit: Sequence[int],
) -> dict[str, object]:
    import numpy as np  # type: ignore
    import torch  # type: ignore

    x_np = np.asarray(runtime.fit_x, dtype=np.float32)
    y_np = np.asarray(y_fit, dtype=np.int64)
    with torch.no_grad():
        xt = torch.as_tensor(x_np, dtype=torch.float32)
        yt = torch.as_tensor(y_np, dtype=torch.long)
        mu, _logvar = runtime.model.encode(xt, yt)
        zeros = torch.zeros(len(x_np), dtype=torch.long)
        ones = torch.ones(len(x_np), dtype=torch.long)
        dec0 = runtime.model.decode(mu, zeros).detach().cpu().numpy()
        dec1 = runtime.model.decode(mu, ones).detach().cpu().numpy()
        dec_true = runtime.model.decode(mu, yt).detach().cpu().numpy()

    diff = np.asarray(dec1, dtype=float) - np.asarray(dec0, dtype=float)
    label_swap_l2 = float(np.linalg.norm(diff, axis=1).mean()) if len(diff) else math.nan
    label_swap_mse = float(np.mean(diff**2)) if len(diff) else math.nan
    label_swap_cosine = _mean_cosine_distance(dec0, dec1)
    real_centroid = _class_centroid_l2(x_np, y_np)
    decoded_centroid = _class_centroid_l2(dec_true, y_np)
    recon_mse = float(np.mean((np.asarray(dec_true, dtype=float) - np.asarray(x_np, dtype=float)) ** 2))
    l2_ratio = _safe_ratio(label_swap_l2, real_centroid)
    mse_ratio = _safe_ratio(label_swap_mse, recon_mse)
    decoded_ratio = _safe_ratio(decoded_centroid, real_centroid)
    weak = (
        _finite(l2_ratio)
        and _finite(mse_ratio)
        and _finite(decoded_ratio)
        and l2_ratio < float(cfg.weak_conditioning_ratio_threshold)
        and mse_ratio < float(cfg.weak_conditioning_ratio_threshold)
        and decoded_ratio < float(cfg.weak_conditioning_ratio_threshold)
    )
    return {
        "aggregation_level": "seed",
        "control_name": spec.control_name,
        "domain_name": spec.domain_name,
        "split_seed": int(spec.split_seed),
        "variant_id": runtime.variant.variant_id,
        "n_rows": int(len(x_np)),
        "label_swap_l2": label_swap_l2,
        "label_swap_mse": label_swap_mse,
        "label_swap_cosine_distance": label_swap_cosine,
        "real_class_centroid_l2": real_centroid,
        "decoded_class_centroid_l2": decoded_centroid,
        "reconstruction_mse": recon_mse,
        "label_swap_l2_over_real_class_centroid_l2": l2_ratio,
        "label_swap_mse_over_reconstruction_mse": mse_ratio,
        "decoded_class_centroid_l2_over_real_class_centroid_l2": decoded_ratio,
        "weak_conditioning": _bool_text(weak),
        "status": VALID_STATUS,
    }


def _mean_cosine_distance(a: object, b: object) -> float:
    import numpy as np  # type: ignore

    a_np = np.asarray(a, dtype=float)
    b_np = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a_np, axis=1) * np.linalg.norm(b_np, axis=1)
    valid = denom > 0
    if not bool(np.any(valid)):
        return math.nan
    cosine = np.sum(a_np[valid] * b_np[valid], axis=1) / denom[valid]
    return float(np.mean(1.0 - cosine))


def _class_centroid_l2(x: object, y: Sequence[int]) -> float:
    import numpy as np  # type: ignore

    x_np = np.asarray(x, dtype=float)
    y_np = np.asarray(y, dtype=int)
    if not bool(np.any(y_np == 0)) or not bool(np.any(y_np == 1)):
        return math.nan
    c0 = x_np[y_np == 0].mean(axis=0)
    c1 = x_np[y_np == 1].mean(axis=0)
    return float(np.linalg.norm(c1 - c0))


def _safe_ratio(value: object, reference: object) -> float:
    v = _float_or_nan(value)
    ref = _float_or_nan(reference)
    if not _finite(v) or not _finite(ref) or ref <= 0.0:
        return math.nan
    return float(v / ref)


def _reconstruction_row(
    spec: SplitSpec,
    variant: VariantConfig,
    condition_row_id: str,
    train_condition_labels: str,
    encode_condition_labels: str,
    decode_condition_labels: str,
    diagnostics: Mapping[str, float],
    *,
    n_rows: int,
) -> dict[str, object]:
    return {
        "control_name": spec.control_name,
        "domain_name": spec.domain_name,
        "split_seed": int(spec.split_seed),
        "variant_id": variant.variant_id,
        "representation_role": condition_row_id,
        "condition_row_id": condition_row_id,
        "train_condition_labels": train_condition_labels,
        "encode_condition_labels": encode_condition_labels,
        "decode_condition_labels": decode_condition_labels,
        "n_rows": int(n_rows),
        **dict(diagnostics),
        "status": VALID_STATUS,
    }


def _model_manifest_row(
    spec: SplitSpec,
    runtime: CVAERuntime,
    rows: Sequence[ManifestRow],
    train_condition_labels: str,
) -> dict[str, object]:
    return {
        "control_name": spec.control_name,
        "domain_name": spec.domain_name,
        "split_seed": int(spec.split_seed),
        "variant_id": runtime.variant.variant_id,
        "train_condition_labels": train_condition_labels,
        "condition_mode": train_condition_labels,
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


def _normalize_metric_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    out = []
    for row in rows:
        item = dict(row)
        for key in (
            "condition_row_id",
            "train_condition_labels",
            "encode_condition_labels",
            "decode_condition_labels",
            "chance_corrected_ratio_vs_true_condition",
        ):
            item.setdefault(key, "")
        out.append(item)
    return out


def _attach_condition_ratios(rows: Sequence[dict[str, object]]) -> None:
    summary = {
        (row["control_name"], row["domain_name"], row["variant_id"], row["condition_row_id"]): row
        for row in rows
        if row.get("aggregation_level") == "summary"
    }
    for row in rows:
        if row.get("aggregation_level") != "summary":
            continue
        condition = str(row.get("condition_row_id", ""))
        if not condition or condition == TRUE_TRAIN_TRUE_ENCODE_TRUE_DECODE:
            continue
        true_row = summary.get((row["control_name"], row["domain_name"], row["variant_id"], TRUE_TRAIN_TRUE_ENCODE_TRUE_DECODE))
        if true_row is None:
            continue
        row["chance_corrected_ratio_vs_true_condition"] = _chance_corrected_ratio(row.get("bacc"), true_row.get("bacc"))


def _aggregate_decoder_rows(
    cfg: MidogPPConditionAuditConfig,
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(
            (
                str(row.get("control_name", "")),
                str(row.get("domain_name", "")),
                str(row.get("variant_id", "")),
            ),
            [],
        ).append(row)
    out = []
    for (_control, _domain, _variant), group in sorted(grouped.items()):
        valid = [row for row in group if row.get("status") == VALID_STATUS]
        base = dict(group[0])
        threshold = float(cfg.weak_conditioning_ratio_threshold)
        l2_ratio = _mean([row.get("label_swap_l2_over_real_class_centroid_l2") for row in valid])
        mse_ratio = _mean([row.get("label_swap_mse_over_reconstruction_mse") for row in valid])
        decoded_ratio = _mean([row.get("decoded_class_centroid_l2_over_real_class_centroid_l2") for row in valid])
        weak = (
            _finite(l2_ratio)
            and _finite(mse_ratio)
            and _finite(decoded_ratio)
            and l2_ratio < threshold
            and mse_ratio < threshold
            and decoded_ratio < threshold
        )
        base.update(
            {
                "aggregation_level": "summary",
                "split_seed": "",
                "n_rows": "",
                "label_swap_l2": _mean([row.get("label_swap_l2") for row in valid]),
                "label_swap_mse": _mean([row.get("label_swap_mse") for row in valid]),
                "label_swap_cosine_distance": _mean([row.get("label_swap_cosine_distance") for row in valid]),
                "real_class_centroid_l2": _mean([row.get("real_class_centroid_l2") for row in valid]),
                "decoded_class_centroid_l2": _mean([row.get("decoded_class_centroid_l2") for row in valid]),
                "reconstruction_mse": _mean([row.get("reconstruction_mse") for row in valid]),
                "label_swap_l2_over_real_class_centroid_l2": l2_ratio,
                "label_swap_mse_over_reconstruction_mse": mse_ratio,
                "decoded_class_centroid_l2_over_real_class_centroid_l2": decoded_ratio,
                "weak_conditioning": _bool_text(weak),
                "status": VALID_STATUS if valid else "insufficient_valid_seeds",
            }
        )
        out.append(base)
    return out


def _mean(values: Sequence[object]) -> float:
    vals = [_float_or_nan(value) for value in values if _finite(value)]
    return float(sum(vals) / len(vals)) if vals else math.nan


def _leakage_report(audit_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    overlap_failures = [row for row in audit_rows if int(row.get("overlap_count", 0)) > 0]
    extra = ["identity_overlap_failure"] if overlap_failures else []
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
            "fit_only_pca": True,
            "fit_only_cvae_training": True,
            "eval_labels_scoring_only": True,
            "fixed_threshold": 0.5,
            "metadata_success_signal": False,
            "preservation_pass_emitted": False,
        }
    )
    return report


def _decision_labels(
    cfg: MidogPPConditionAuditConfig,
    pca_rows: Sequence[Mapping[str, object]],
    condition_rows: Sequence[Mapping[str, object]],
    decoder_rows: Sequence[Mapping[str, object]],
    leakage: Mapping[str, object],
) -> list[str]:
    if leakage.get("status") != "PASS":
        return ["LEAKAGE_OR_ALIGNMENT_FAILURE_SUSPECT"]
    labels = ["CONDITION_AUDIT_PROTOCOL_CLEAN"]

    pca64 = _summary_metric(pca_rows, BALANCED_CONTROL, "", "pca64_beta001", REAL_PCA64_REFERENCE)
    if pca64 is None or not _above_threshold(pca64, cfg):
        labels.append("PCA64_CAPACITY_BOTTLENECK")
    for variant_id, role, label in (
        ("pca128_beta001", REAL_PCA128_REFERENCE, "PCA128_CAPACITY_RECOVERY"),
        ("pca256_beta001", REAL_PCA256_REFERENCE, "PCA256_CAPACITY_RECOVERY"),
    ):
        row = _summary_metric(pca_rows, BALANCED_CONTROL, "", variant_id, role)
        if row is not None and _above_threshold(row, cfg):
            labels.append(label)

    permuted_above = [
        row
        for row in condition_rows
        if row.get("aggregation_level") == "summary"
        and row.get("control_name") == BALANCED_CONTROL
        and str(row.get("condition_row_id")) != TRUE_TRAIN_TRUE_ENCODE_TRUE_DECODE
        and _finite(row.get("ci_low"))
        and float(row.get("ci_low")) > cfg.ci_low_threshold
        and _finite(row.get("bacc"))
        and float(row.get("bacc")) > cfg.real_gate_min_bacc
    ]
    if permuted_above:
        labels.append("CONDITION_PERMUTATION_CONTROL_REPRODUCED")

    if _latent_signal_dominates_condition(cfg, condition_rows):
        labels.append("LATENT_CLASS_SIGNAL_DOMINATES_CONDITION")

    weak_primary = _summary_decoder(decoder_rows, BALANCED_CONTROL, "", "pca64_beta001")
    if weak_primary is not None and str(weak_primary.get("weak_conditioning")) == "true":
        labels.append("DECODER_LABEL_CONDITIONING_WEAK")

    return _without_preservation_pass(labels)


def _above_threshold(row: Mapping[str, object], cfg: MidogPPConditionAuditConfig) -> bool:
    return (
        _finite(row.get("bacc"))
        and _finite(row.get("ci_low"))
        and float(row.get("bacc")) >= cfg.real_gate_min_bacc
        and float(row.get("ci_low")) > cfg.ci_low_threshold
    )


def _above_chance_condition(row: Mapping[str, object], cfg: MidogPPConditionAuditConfig) -> bool:
    return (
        _finite(row.get("bacc"))
        and _finite(row.get("ci_low"))
        and float(row.get("bacc")) > cfg.real_gate_min_bacc
        and float(row.get("ci_low")) > cfg.ci_low_threshold
    )


def _latent_signal_dominates_condition(
    cfg: MidogPPConditionAuditConfig,
    rows: Sequence[Mapping[str, object]],
) -> bool:
    for variant_id in ("pca64_beta001", "pca128_beta001", "pca256_beta001"):
        perm_train = _summary_condition(rows, BALANCED_CONTROL, "", variant_id, PERMUTED_TRAIN_TRUE_ENCODE_TRUE_DECODE)
        if perm_train is not None and _above_chance_condition(perm_train, cfg):
            return True
        true_row = _summary_condition(rows, BALANCED_CONTROL, "", variant_id, TRUE_TRAIN_TRUE_ENCODE_TRUE_DECODE)
        perm_decode = _summary_condition(rows, BALANCED_CONTROL, "", variant_id, TRUE_TRAIN_TRUE_ENCODE_PERMUTED_DECODE)
        if true_row is None or perm_decode is None:
            continue
        true_bacc = _float_or_nan(true_row.get("bacc"))
        perm_bacc = _float_or_nan(perm_decode.get("bacc"))
        ratio = _chance_corrected_ratio(perm_bacc, true_bacc)
        if _finite(true_bacc) and _finite(perm_bacc) and abs(perm_bacc - true_bacc) <= cfg.close_bacc_delta:
            return True
        if _finite(ratio) and ratio >= cfg.close_ratio_threshold:
            return True
    return False


def _summary_metric(
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


def _summary_condition(
    rows: Sequence[Mapping[str, object]],
    control: str,
    domain: str,
    variant: str,
    condition_row_id: str,
) -> Mapping[str, object] | None:
    for row in rows:
        if (
            row.get("aggregation_level") == "summary"
            and row.get("control_name") == control
            and row.get("domain_name") == domain
            and row.get("variant_id") == variant
            and row.get("condition_row_id") == condition_row_id
        ):
            return row
    return None


def _summary_decoder(
    rows: Sequence[Mapping[str, object]],
    control: str,
    domain: str,
    variant: str,
) -> Mapping[str, object] | None:
    for row in rows:
        if (
            row.get("aggregation_level") == "summary"
            and row.get("control_name") == control
            and row.get("domain_name") == domain
            and row.get("variant_id") == variant
        ):
            return row
    return None


def _without_preservation_pass(labels: Sequence[str]) -> list[str]:
    forbidden = ("PRESERVATION_PASS", "PRESERVATION_SANITY_PASS", "CVAE_PRESERVATION")
    out = []
    for label in labels:
        if any(token in str(label) for token in forbidden):
            continue
        out.append(str(label))
    return out or ["INSUFFICIENT_VALID_SPLITS"]


def _protocol_manifest(
    cfg: MidogPPConditionAuditConfig,
    cache: FeatureCache,
    rows: Sequence[ManifestRow],
    splits: Sequence[SplitSpec],
) -> dict[str, object]:
    feature_dim = int(getattr(cache.embeddings, "shape", [0, 0])[1])
    if feature_dim == 0:
        feature_dim = int(_to_numpy(cache.embeddings).shape[1])
    return {
        "schema_version": "midogpp_virchow2_cvae_condition_audit_protocol_v1",
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
        "real_full_dim_reference": "raw Virchow2 features with downstream StandardScaler only; no full-dimensional CVAE.",
        "condition_rows": [
            {
                "condition_row_id": TRUE_TRAIN_TRUE_ENCODE_TRUE_DECODE,
                "train_condition_labels": TRUE_LABELS,
                "encode_condition_labels": TRUE_LABELS,
                "decode_condition_labels": TRUE_LABELS,
            },
            {
                "condition_row_id": PERMUTED_TRAIN_TRUE_ENCODE_TRUE_DECODE,
                "train_condition_labels": PERMUTED_LABELS,
                "encode_condition_labels": TRUE_LABELS,
                "decode_condition_labels": TRUE_LABELS,
            },
            {
                "condition_row_id": TRUE_TRAIN_TRUE_ENCODE_PERMUTED_DECODE,
                "train_condition_labels": TRUE_LABELS,
                "encode_condition_labels": TRUE_LABELS,
                "decode_condition_labels": PERMUTED_LABELS,
            },
            {
                "condition_row_id": PERMUTED_TRAIN_PERMUTED_ENCODE_PERMUTED_DECODE,
                "train_condition_labels": PERMUTED_LABELS,
                "encode_condition_labels": PERMUTED_LABELS,
                "decode_condition_labels": PERMUTED_LABELS,
            },
        ],
        "diagnostic_only": True,
        "preservation_pass_emitted": False,
        "pca_promotion_allowed": False,
        "claim_boundary": {
            "allowed": "PCA capacity and CVAE condition-use mechanics under existing MIDOG++ signal-control splits.",
            "forbidden": "No preservation PASS, pca128 promotion, routing, expert-selection, or composition claim.",
        },
    }


def _write_decision_report(
    path: Path,
    cfg: MidogPPConditionAuditConfig,
    decision_labels: Sequence[str],
    leakage: Mapping[str, object],
    pca_rows: Sequence[Mapping[str, object]],
    condition_rows: Sequence[Mapping[str, object]],
    decoder_rows: Sequence[Mapping[str, object]],
) -> None:
    del cfg
    lines = [
        "# MIDOG++ CVAE Condition and PCA Capacity Audit",
        "",
        f"- Decision labels: `{', '.join(decision_labels)}`",
        f"- Leakage status: `{leakage.get('status')}`",
        "- Diagnostic-only: no CVAE preservation PASS, pca128 promotion, routing, or composition claim is emitted.",
        "- `real_full_dim_reference` is raw Virchow2 plus downstream StandardScaler only, not a full-dimensional CVAE.",
        "- Eval labels are scoring-only; PCA, CVAE training, and downstream classifier fitting use fit rows only.",
        "",
        "## PCA Capacity",
        "",
        "| Control | Variant | Role | BACC | CI low | Status |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for variant, role in (
        (FULL_DIM_VARIANT, REAL_FULL_DIM_REFERENCE),
        ("pca64_beta001", REAL_PCA64_REFERENCE),
        ("pca128_beta001", REAL_PCA128_REFERENCE),
        ("pca256_beta001", REAL_PCA256_REFERENCE),
    ):
        row = _summary_metric(pca_rows, BALANCED_CONTROL, "", variant, role)
        if row is not None:
            lines.append(
                f"| {BALANCED_CONTROL} | {variant} | {role} | {_fmt(row.get('bacc'))} | "
                f"{_fmt(row.get('ci_low'))} | {row.get('status')} |"
            )
    lines.extend(
        [
            "",
            "## Condition Matrix",
            "",
            "| Variant | Condition row | BACC | CI low | Ratio vs true |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in condition_rows:
        if row.get("aggregation_level") == "summary" and row.get("control_name") == BALANCED_CONTROL:
            lines.append(
                f"| {row.get('variant_id')} | {row.get('condition_row_id')} | {_fmt(row.get('bacc'))} | "
                f"{_fmt(row.get('ci_low'))} | {_fmt(row.get('chance_corrected_ratio_vs_true_condition'))} |"
            )
    lines.extend(
        [
            "",
            "## Decoder Label Conditioning",
            "",
            "| Variant | L2/centroid | MSE/recon | Decoded centroid/real centroid | Weak |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in decoder_rows:
        if row.get("aggregation_level") == "summary" and row.get("control_name") == BALANCED_CONTROL:
            lines.append(
                f"| {row.get('variant_id')} | {_fmt(row.get('label_swap_l2_over_real_class_centroid_l2'))} | "
                f"{_fmt(row.get('label_swap_mse_over_reconstruction_mse'))} | "
                f"{_fmt(row.get('decoded_class_centroid_l2_over_real_class_centroid_l2'))} | {row.get('weak_conditioning')} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
