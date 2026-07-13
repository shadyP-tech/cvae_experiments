from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..metrics import balanced_accuracy, macro_f1
from ..reporting import prepare_artifact_dirs, write_csv_rows, write_json
from .sanity import (
    AUDIT_COLUMNS,
    DEFAULT_FEATURE_CACHE_PATH,
    DEFAULT_MANIFEST_PATH,
    DECODE_MU,
    MODEL_MANIFEST_COLUMNS,
    POSTERIOR_SAMPLE,
    PRIOR_SAMPLE,
    RECON_COLUMNS,
    TRAINING_COLUMNS,
    VALID_STATUS,
    ManifestRow,
    SplitSpec,
    VariantConfig,
    _assert_cache_alignment,
    _assert_virchow2_cache,
    _bool_text,
    _case_cluster_bacc_ci,
    _decode_mu,
    _empty_metric_row,
    _finite,
    _identity_audit_rows,
    _load_feature_cache,
    _mapping,
    _parse_variant,
    _path,
    _posterior_sample,
    _prior_sample,
    _read_train_manifest,
    _reconstruction_row,
    _split_counts,
    _stable_seed,
    _to_numpy,
    _train_runtime,
)
from .condition_audit import REAL_PCA128_REFERENCE
from ..protocol import ProtocolError
from .tuned_reference import (
    TunedClassifierReference,
    TunedClassifierSpec,
    load_tuned_classifier_reference,
)


EXPERIMENT_NAME = "virchow2_cvae_midogpp_tuned_classifier_preservation_v1"
DEFAULT_ARTIFACT_ROOT = (
    "artifacts/midogpp/20_cvae_preservation/"
    "virchow2_cvae_midogpp_tuned_classifier_preservation_v1/seed42"
)
DEFAULT_REFERENCE_ROOT = (
    "artifacts/midogpp/10_real_feature_reference/"
    "real_feature_threshold_both_annotation_patch_xyxy_virchow2_seed42"
)
PRIMARY_VARIANT = "pca128_beta001"
CONTROL_NAME = "heldout_center_lodo"
REAL_TUNED_REFERENCE = "imported_real_virchow2_tuned_reference"

METRIC_COLUMNS = (
    "aggregation_level",
    "control_name",
    "heldout_center",
    "split_seed",
    "variant_id",
    "representation_role",
    "claim_role",
    "model_type",
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
    "real_tuned_bacc",
    "real_tuned_macro_f1",
    "preservation_ratio_vs_real_tuned",
    "ratio_status",
    "selected_classifier_config_hash",
    "selected_classifier_spec",
    "reference_artifact_root",
    "reference_protocol_hash",
    "reference_manifest_hash",
    "reference_feature_cache_hash",
    "selection_source",
    "target_eval_labels_used_for_scoring_only",
    "selection_used_target_labels",
    "fit_used_target_center",
    "claim_scope",
)

PREDICTION_COLUMNS = (
    "control_name",
    "heldout_center",
    "split_seed",
    "variant_id",
    "representation_role",
    "sample_id",
    "case_id",
    "feature_row_index",
    "y_true",
    "y_pred",
    "prob_pos",
    "selected_classifier_config_hash",
)

REFERENCE_COLUMNS = (
    "heldout_center",
    "heldout_bacc",
    "heldout_macro_f1",
    "n_train",
    "n_eval",
    "selected_classifier_config_hash",
    "selected_classifier_spec",
    "feature_cache_hash",
    "manifest_hash",
)


@dataclass(frozen=True)
class MidogPPTunedClassifierPreservationConfig:
    name: str
    artifact_root: Path
    manifest_path: Path
    feature_cache_path: Path
    real_feature_reference_artifact_root: Path
    positive_label: int
    variant: VariantConfig
    experiment_seed: int
    heldout_centers: tuple[str, ...] | None
    allow_npz_test_cache: bool
    min_fit: int
    min_eval: int
    min_fit_pos: int
    min_fit_neg: int
    min_eval_pos: int
    min_eval_neg: int
    min_fit_cases: int
    min_eval_cases: int
    bootstrap_reps: int
    bootstrap_seed: int
    real_reference_min_bacc_for_ratio: float
    ci_low_threshold: float
    expected_reference_manifest_hash: str | None
    expected_reference_feature_cache_hash: str | None


def load_midogpp_tuned_classifier_preservation_config(
    path: str | Path,
) -> MidogPPTunedClassifierPreservationConfig:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise ProtocolError("Loading MIDOG++ tuned-classifier preservation configs requires PyYAML.") from exc
    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ProtocolError(f"Config must be a mapping: {path}")
    base_dir = source.parents[2] if len(source.parents) >= 3 else source.parent
    return parse_midogpp_tuned_classifier_preservation_config(payload, base_dir=base_dir)


def parse_midogpp_tuned_classifier_preservation_config(
    data: Mapping[str, object],
    *,
    base_dir: str | Path = ".",
) -> MidogPPTunedClassifierPreservationConfig:
    base = Path(base_dir)
    experiment = _mapping(data.get("experiment"), "experiment")
    inputs = _mapping(data.get("inputs"), "inputs")
    run = _mapping(data.get("run"), "run", allow_empty=True)
    thresholds = _mapping(data.get("validity_thresholds"), "validity_thresholds", allow_empty=True)
    bootstrap = _mapping(data.get("bootstrap"), "bootstrap", allow_empty=True)
    reference = _mapping(data.get("reference_validation"), "reference_validation", allow_empty=True)
    variant_payload = data.get("variant") or {
        "variant_id": PRIMARY_VARIANT,
        "pca_dim": 128,
        "latent_dim": 32,
    }
    heldout = run.get("heldout_centers")
    cfg = MidogPPTunedClassifierPreservationConfig(
        name=str(experiment.get("name", "")),
        artifact_root=_path(base, str(experiment.get("artifact_root", DEFAULT_ARTIFACT_ROOT))),
        manifest_path=_path(base, str(inputs.get("manifest_path", DEFAULT_MANIFEST_PATH))),
        feature_cache_path=_path(base, str(inputs.get("feature_cache_path", DEFAULT_FEATURE_CACHE_PATH))),
        real_feature_reference_artifact_root=_path(
            base,
            str(inputs.get("real_feature_reference_artifact_root", DEFAULT_REFERENCE_ROOT)),
        ),
        positive_label=int(data.get("positive_label", 1)),
        variant=_parse_variant(variant_payload),
        experiment_seed=int(run.get("experiment_seed", 42)),
        heldout_centers=None if heldout in (None, "", "all") else tuple(str(v) for v in heldout),
        allow_npz_test_cache=bool(inputs.get("allow_npz_test_cache", False)),
        min_fit=int(thresholds.get("min_fit", 20)),
        min_eval=int(thresholds.get("min_eval", 10)),
        min_fit_pos=int(thresholds.get("min_fit_pos", 10)),
        min_fit_neg=int(thresholds.get("min_fit_neg", 10)),
        min_eval_pos=int(thresholds.get("min_eval_pos", 5)),
        min_eval_neg=int(thresholds.get("min_eval_neg", 5)),
        min_fit_cases=int(thresholds.get("min_fit_cases", 3)),
        min_eval_cases=int(thresholds.get("min_eval_cases", 2)),
        bootstrap_reps=int(bootstrap.get("reps", 1000)),
        bootstrap_seed=int(bootstrap.get("seed", 1337)),
        real_reference_min_bacc_for_ratio=float(reference.get("real_reference_min_bacc_for_ratio", 0.55)),
        ci_low_threshold=float(reference.get("ci_low_threshold", 0.50)),
        expected_reference_manifest_hash=_optional_str(reference.get("expected_manifest_hash")),
        expected_reference_feature_cache_hash=_optional_str(reference.get("expected_feature_cache_hash")),
    )
    validate_midogpp_tuned_classifier_preservation_config(cfg)
    return cfg


def validate_midogpp_tuned_classifier_preservation_config(
    cfg: MidogPPTunedClassifierPreservationConfig,
) -> None:
    if cfg.name != EXPERIMENT_NAME:
        raise ProtocolError(f"Unexpected experiment.name={cfg.name!r}; expected {EXPERIMENT_NAME!r}.")
    if cfg.variant.variant_id != PRIMARY_VARIANT:
        raise ProtocolError(f"variant.variant_id must remain {PRIMARY_VARIANT!r}.")
    if (cfg.variant.pca_dim, cfg.variant.latent_dim) != (128, 32):
        raise ProtocolError("Tuned-classifier preservation is locked to pca_dim=128 and latent_dim=32.")
    if cfg.variant.hidden_dim != 512 or cfg.variant.num_hidden_layers != 2:
        raise ProtocolError("CVAE architecture is locked to hidden_dim=512 and two hidden layers.")
    if not math.isclose(cfg.variant.beta_final, 0.001):
        raise ProtocolError("CVAE beta_final is locked to 0.001.")
    if cfg.real_reference_min_bacc_for_ratio <= 0.5:
        raise ProtocolError("real_reference_min_bacc_for_ratio must be above chance.")


def run_midogpp_tuned_classifier_preservation(
    cfg: MidogPPTunedClassifierPreservationConfig,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    import numpy as np  # type: ignore

    reference = load_tuned_classifier_reference(
        cfg.real_feature_reference_artifact_root,
        expected_manifest_hash=cfg.expected_reference_manifest_hash,
        expected_feature_cache_hash=cfg.expected_reference_feature_cache_hash,
        required_centers=cfg.heldout_centers,
    )
    heldout_centers = cfg.heldout_centers or reference.heldout_centers
    root = prepare_artifact_dirs(Path(artifact_root) if artifact_root is not None else cfg.artifact_root)
    manifest_rows = _read_train_manifest(cfg.manifest_path, positive_label=cfg.positive_label)
    cache = _load_feature_cache(cfg.feature_cache_path)
    _assert_virchow2_cache(cache, path=cfg.feature_cache_path, allow_npz_test_cache=cfg.allow_npz_test_cache)
    _assert_cache_alignment(manifest_rows, cache)
    embeddings = np.asarray(_to_numpy(cache.embeddings), dtype=float)
    split_specs = _heldout_center_splits(manifest_rows, heldout_centers, cfg.experiment_seed)

    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    reconstruction_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    for spec in split_specs:
        ref = reference.rows_by_center[spec.domain_name]
        audit = _identity_audit_rows(manifest_rows, spec)
        audit_rows.extend(audit)
        counts = _split_counts(manifest_rows, spec.fit_idx, spec.eval_idx)
        status, error = _split_status(cfg, counts)
        if status != VALID_STATUS:
            metric_rows.append(_failure_row(cfg, spec, ref, counts, REAL_PCA128_REFERENCE, status, error))
            continue

        x_fit_raw = embeddings[list(spec.fit_idx)]
        y_fit = np.asarray([manifest_rows[idx].label for idx in spec.fit_idx], dtype=int)
        x_eval_raw = embeddings[list(spec.eval_idx)]
        y_eval = [manifest_rows[idx].label for idx in spec.eval_idx]
        runtime = _train_runtime(
            cfg,  # type: ignore[arg-type]
            spec,
            cfg.variant,
            x_fit_raw=x_fit_raw,
            x_eval_raw=x_eval_raw,
            y_fit=y_fit,
            condition_mode="real_labels",
        )
        training_rows.extend(runtime.training_rows)

        real_row, real_preds = _score_representation(
            cfg,
            spec,
            manifest_rows,
            x_fit=runtime.fit_x,
            y_fit=y_fit,
            x_eval=runtime.eval_x,
            y_eval=y_eval,
            role=REAL_PCA128_REFERENCE,
            counts=counts,
            reference=reference,
        )
        metric_rows.append(real_row)
        prediction_rows.extend(real_preds)

        decoded, recon = _decode_mu(runtime, runtime.fit_x, y_fit)
        reconstruction_rows.append(_reconstruction_row(spec, cfg.variant, DECODE_MU, recon, n_rows=len(y_fit)))
        decode_row, decode_preds = _score_representation(
            cfg,
            spec,
            manifest_rows,
            x_fit=decoded,
            y_fit=y_fit,
            x_eval=runtime.eval_x,
            y_eval=y_eval,
            role=DECODE_MU,
            counts=counts,
            reference=reference,
        )
        metric_rows.append(decode_row)
        prediction_rows.extend(decode_preds)

        posterior = _posterior_sample(
            runtime,
            runtime.fit_x,
            y_fit,
            seed=_stable_seed(spec, cfg.variant.variant_id, "tuned_preservation_posterior"),
        )
        posterior_row, posterior_preds = _score_representation(
            cfg,
            spec,
            manifest_rows,
            x_fit=posterior,
            y_fit=y_fit,
            x_eval=runtime.eval_x,
            y_eval=y_eval,
            role=POSTERIOR_SAMPLE,
            counts=counts,
            reference=reference,
        )
        metric_rows.append(posterior_row)
        prediction_rows.extend(posterior_preds)

        prior_x, prior_y = _prior_sample(
            runtime,
            y_fit,
            seed=_stable_seed(spec, cfg.variant.variant_id, "tuned_preservation_standard_prior"),
        )
        prior_row, prior_preds = _score_representation(
            cfg,
            spec,
            manifest_rows,
            x_fit=prior_x,
            y_fit=prior_y,
            x_eval=runtime.eval_x,
            y_eval=y_eval,
            role=PRIOR_SAMPLE,
            counts=counts,
            reference=reference,
        )
        metric_rows.append(prior_row)
        prediction_rows.extend(prior_preds)

    all_metrics = metric_rows + _aggregate_metric_rows(metric_rows)
    leakage = _leakage_report(audit_rows)
    decision_labels = _decision_labels(cfg, all_metrics, leakage)
    write_csv_rows(root / "tables" / "tuned_preservation_metrics.csv", all_metrics, METRIC_COLUMNS)
    write_csv_rows(root / "tables" / "imported_real_tuned_reference.csv", _reference_rows(reference), REFERENCE_COLUMNS)
    write_csv_rows(root / "tables" / "reconstruction_diagnostics.csv", reconstruction_rows, RECON_COLUMNS)
    write_csv_rows(root / "tables" / "training_diagnostics.csv", training_rows, TRAINING_COLUMNS)
    write_csv_rows(root / "tables" / "identity_overlap_audit.csv", audit_rows, AUDIT_COLUMNS)
    write_csv_rows(root / "tables" / "predictions.csv", prediction_rows, PREDICTION_COLUMNS)
    write_json(root / "manifests" / "protocol_manifest.json", _protocol_manifest(cfg, reference, manifest_rows, split_specs))
    write_json(root / "reports" / "leakage_report.json", leakage)
    _write_decision_report(root / "reports" / "decision_report.md", cfg, reference, decision_labels, all_metrics, leakage)
    return root


def _heldout_center_splits(
    rows: Sequence[ManifestRow],
    heldout_centers: Sequence[str],
    experiment_seed: int,
) -> tuple[SplitSpec, ...]:
    out = []
    centers = {str(row.metadata.get("center", "")).strip() for row in rows}
    for center in heldout_centers:
        center = str(center)
        if center not in centers:
            raise ProtocolError(f"Held-out center {center!r} is absent from manifest train rows.")
        fit_idx = tuple(idx for idx, row in enumerate(rows) if str(row.metadata.get("center", "")).strip() != center)
        eval_idx = tuple(idx for idx, row in enumerate(rows) if str(row.metadata.get("center", "")).strip() == center)
        out.append(
            SplitSpec(
                control_name=CONTROL_NAME,
                domain_name=center,
                split_seed=int(experiment_seed),
                fit_idx=fit_idx,
                eval_idx=eval_idx,
            )
        )
    return tuple(out)


def _split_status(
    cfg: MidogPPTunedClassifierPreservationConfig,
    counts: Mapping[str, int],
) -> tuple[str, str]:
    checks = (
        ("n_fit", cfg.min_fit),
        ("n_eval", cfg.min_eval),
        ("n_fit_pos", cfg.min_fit_pos),
        ("n_fit_neg", cfg.min_fit_neg),
        ("n_eval_pos", cfg.min_eval_pos),
        ("n_eval_neg", cfg.min_eval_neg),
        ("n_fit_cases", cfg.min_fit_cases),
        ("n_eval_cases", cfg.min_eval_cases),
    )
    for key, minimum in checks:
        if int(counts.get(key, 0)) < int(minimum):
            return "invalid_split", f"{key}={counts.get(key, 0)} below minimum {minimum}"
    return VALID_STATUS, ""


def _score_representation(
    cfg: MidogPPTunedClassifierPreservationConfig,
    spec: SplitSpec,
    rows: Sequence[ManifestRow],
    *,
    x_fit: object,
    y_fit: Sequence[int],
    x_eval: object,
    y_eval: Sequence[int],
    role: str,
    counts: Mapping[str, int],
    reference: TunedClassifierReference,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    ref = reference.rows_by_center[spec.domain_name]
    base = _metric_base(cfg, spec, ref, counts, role)
    try:
        probabilities, predictions, converged, n_iter = _fit_predict_with_tuned_spec(
            x_fit,
            y_fit,
            x_eval,
            ref.selected_classifier_spec,
        )
    except Exception as exc:
        failed = dict(base)
        failed.update({"status": "model_failed_convergence", "error_message": str(exc), "converged": "false"})
        return failed, []
    if not converged:
        failed = dict(base)
        failed.update({"status": "model_failed_convergence", "error_message": f"n_iter={list(n_iter)}", "converged": "false"})
        return failed, []
    metrics = _classification_metrics(y_eval, predictions)
    ci_low, ci_high, ci_method = _case_cluster_bacc_ci(
        y_eval,
        predictions,
        [rows[idx].case_id for idx in spec.eval_idx],
        reps=cfg.bootstrap_reps,
        seed=_stable_seed(spec, cfg.variant.variant_id, role, "tuned_ci", cfg.bootstrap_seed),
    )
    ratio, ratio_status = _preservation_ratio(cfg, metrics["bacc"], ref.bacc, ci_low)
    out = dict(base)
    out.update(
        {
            **metrics,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "ci_method": ci_method,
            "above_chance": _bool_text(_finite(metrics["bacc"]) and _finite(ci_low) and metrics["bacc"] > 0.5 and ci_low > 0.5),
            "near_chance": _bool_text(not (_finite(ci_low) and ci_low > 0.5)),
            "converged": "true",
            "status": VALID_STATUS,
            "error_message": "",
            "preservation_ratio_vs_real_tuned": ratio,
            "ratio_status": ratio_status,
        }
    )
    preds = []
    for pos, idx in enumerate(spec.eval_idx):
        row = rows[idx]
        preds.append(
            {
                "control_name": spec.control_name,
                "heldout_center": spec.domain_name,
                "split_seed": int(spec.split_seed),
                "variant_id": cfg.variant.variant_id,
                "representation_role": role,
                "sample_id": row.sample_id,
                "case_id": row.case_id,
                "feature_row_index": int(row.feature_row_index),
                "y_true": int(y_eval[pos]),
                "y_pred": int(predictions[pos]),
                "prob_pos": float(probabilities[pos]),
                "selected_classifier_config_hash": ref.selected_classifier_config_hash,
            }
        )
    return out, preds


def _fit_predict_with_tuned_spec(
    x_fit: object,
    y_fit: Sequence[int],
    x_eval: object,
    spec: TunedClassifierSpec,
) -> tuple[list[float], list[int], bool, tuple[int, ...]]:
    import numpy as np  # type: ignore
    from sklearn.exceptions import ConvergenceWarning  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore
    import warnings

    x_train = np.asarray(x_fit, dtype=float)
    y_train = np.asarray(y_fit, dtype=int)
    x_target = np.asarray(x_eval, dtype=float)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_target = scaler.transform(x_target)
    kwargs: dict[str, object] = {
        "C": float(spec.C),
        "penalty": spec.penalty,
        "solver": spec.solver,
        "max_iter": int(spec.max_iter),
        "class_weight": spec.class_weight,
        "random_state": int(spec.random_state),
    }
    if spec.l1_ratio is not None:
        kwargs["l1_ratio"] = float(spec.l1_ratio)
    clf = LogisticRegression(**kwargs)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(x_train, y_train)
    classes = tuple(int(value) for value in clf.classes_.tolist())
    if classes != (0, 1):
        raise ProtocolError(f"class order must be (0, 1), got {classes}")
    proba = clf.predict_proba(x_target)[:, 1]
    if spec.threshold_policy == "fixed_0_5":
        pred = [1 if float(value) >= 0.5 else 0 for value in proba.tolist()]
    else:
        pred = [int(value) for value in clf.predict(x_target).tolist()]
    n_iter = tuple(int(value) for value in getattr(clf, "n_iter_", ()))
    converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
    return [float(value) for value in proba.tolist()], pred, bool(converged), n_iter


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


def _metric_base(
    cfg: MidogPPTunedClassifierPreservationConfig,
    spec: SplitSpec,
    ref: Any,
    counts: Mapping[str, int],
    role: str,
) -> dict[str, object]:
    row = _empty_metric_row(
        spec,
        variant_id=cfg.variant.variant_id,
        representation_role=role,
        counts=counts,
        status=VALID_STATUS,
        error_message="",
    )
    selected = ref.selected_classifier_spec
    row.update(
        {
            "heldout_center": spec.domain_name,
            "claim_role": "real_pca128_context" if role == REAL_PCA128_REFERENCE else "cvae_preservation",
            "real_tuned_bacc": ref.bacc,
            "real_tuned_macro_f1": ref.macro_f1,
            "preservation_ratio_vs_real_tuned": "",
            "ratio_status": "",
            "selected_classifier_config_hash": ref.selected_classifier_config_hash,
            "selected_classifier_spec": json.dumps(dict(selected.payload), sort_keys=True),
            "reference_artifact_root": str(cfg.real_feature_reference_artifact_root),
            "reference_protocol_hash": "",
            "reference_manifest_hash": ref.manifest_hash,
            "reference_feature_cache_hash": ref.feature_cache_hash,
            "selection_source": "imported_source_inner_lodo",
            "target_eval_labels_used_for_scoring_only": "true",
            "selection_used_target_labels": "false",
            "fit_used_target_center": "false",
            "claim_scope": "cvae_preservation_only",
        }
    )
    return row


def _failure_row(
    cfg: MidogPPTunedClassifierPreservationConfig,
    spec: SplitSpec,
    ref: Any,
    counts: Mapping[str, int],
    role: str,
    status: str,
    error: str,
) -> dict[str, object]:
    row = _metric_base(cfg, spec, ref, counts, role)
    row.update({"status": status, "error_message": error, "converged": "false"})
    return row


def _preservation_ratio(
    cfg: MidogPPTunedClassifierPreservationConfig,
    bacc: object,
    real_bacc: object,
    ci_low: object,
) -> tuple[float | str, str]:
    if not (_finite(bacc) and _finite(real_bacc)):
        return "", "invalid_nonfinite"
    real = float(real_bacc)
    if real < cfg.real_reference_min_bacc_for_ratio:
        return "", "invalid_reference_near_chance"
    if _finite(ci_low) and float(ci_low) <= cfg.ci_low_threshold:
        return "", "diagnostic_ci_crosses_chance"
    denom = real - 0.5
    if denom <= 0.0:
        return "", "invalid_reference_denominator"
    return (float(bacc) - 0.5) / denom, "ok"


def _aggregate_metric_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row.get("variant_id", "")), str(row.get("representation_role", ""))), []).append(row)
    out = []
    for (_variant, _role), group in sorted(grouped.items()):
        valid = [row for row in group if str(row.get("status")) == VALID_STATUS]
        base = dict(group[0])
        bacc_values = [_float(row.get("bacc")) for row in valid]
        ratio_values = [_float(row.get("preservation_ratio_vs_real_tuned")) for row in valid if row.get("ratio_status") == "ok"]
        base.update(
            {
                "aggregation_level": "summary",
                "heldout_center": "",
                "split_seed": "",
                "n_fit": "",
                "n_eval": "",
                "n_fit_pos": "",
                "n_fit_neg": "",
                "n_eval_pos": "",
                "n_eval_neg": "",
                "n_fit_cases": "",
                "n_eval_cases": "",
                "bacc": _mean(bacc_values),
                "bacc_std": _std(bacc_values),
                "bacc_min": _min(bacc_values),
                "macro_f1": _mean([_float(row.get("macro_f1")) for row in valid]),
                "precision_pos": _mean([_float(row.get("precision_pos")) for row in valid]),
                "recall_pos": _mean([_float(row.get("recall_pos")) for row in valid]),
                "f1_pos": _mean([_float(row.get("f1_pos")) for row in valid]),
                "support_pos": "",
                "support_neg": "",
                "ci_low": _min([_float(row.get("ci_low")) for row in valid]),
                "ci_high": _max([_float(row.get("ci_high")) for row in valid]),
                "ci_method": "case_cluster_conservative_center_equal_aggregate",
                "above_chance": "",
                "near_chance": "",
                "converged": _bool_text(bool(valid) and len(valid) == len(group)),
                "status": VALID_STATUS if valid else "insufficient_valid_centers",
                "error_message": "" if valid else "no valid centers",
                "real_tuned_bacc": _mean([_float(row.get("real_tuned_bacc")) for row in valid]),
                "real_tuned_macro_f1": _mean([_float(row.get("real_tuned_macro_f1")) for row in valid]),
                "preservation_ratio_vs_real_tuned": _mean(ratio_values),
                "ratio_status": "ok" if ratio_values else "no_valid_ratios",
            }
        )
        out.append(base)
    return out


def _reference_rows(reference: TunedClassifierReference) -> list[dict[str, object]]:
    rows = []
    for center in reference.heldout_centers:
        row = reference.rows_by_center[center]
        rows.append(
            {
                "heldout_center": row.heldout_center,
                "heldout_bacc": row.bacc,
                "heldout_macro_f1": row.macro_f1,
                "n_train": row.n_train,
                "n_eval": row.n_eval,
                "selected_classifier_config_hash": row.selected_classifier_config_hash,
                "selected_classifier_spec": json.dumps(dict(row.selected_classifier_spec.payload), sort_keys=True),
                "feature_cache_hash": row.feature_cache_hash,
                "manifest_hash": row.manifest_hash,
            }
        )
    return rows


def _leakage_report(audit_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    failed = [row for row in audit_rows if str(row.get("status")) != "PASS"]
    return {
        "schema_version": "midogpp_tuned_classifier_preservation_leakage_v1",
        "status": "PASS" if not failed else "FAIL",
        "fit_used_target_center": False,
        "selection_used_target_labels": False,
        "target_eval_labels_used_for_scoring_only": True,
        "sample_overlap_failures": len(failed),
    }


def _protocol_manifest(
    cfg: MidogPPTunedClassifierPreservationConfig,
    reference: TunedClassifierReference,
    rows: Sequence[ManifestRow],
    splits: Sequence[SplitSpec],
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_tuned_classifier_preservation_protocol_v1",
        "experiment_name": cfg.name,
        "artifact_identity": "cvae_tuned_classifier_preservation",
        "claim_scope": "cvae_preservation_only",
        "manifest_path": str(cfg.manifest_path),
        "feature_cache_path": str(cfg.feature_cache_path),
        "real_feature_reference_artifact_root": str(cfg.real_feature_reference_artifact_root),
        "reference_protocol_hash": reference.protocol.get("protocol_hash", ""),
        "reference_manifest_hash": reference.protocol.get("manifest_hash", ""),
        "reference_feature_cache_hash": reference.protocol.get("feature_cache_hash", ""),
        "heldout_centers": [spec.domain_name for spec in splits],
        "train_rows": len(rows),
        "split_count": len(splits),
        "variant": {
            "variant_id": cfg.variant.variant_id,
            "pca_dim": cfg.variant.pca_dim,
            "latent_dim": cfg.variant.latent_dim,
            "hidden_dim": cfg.variant.hidden_dim,
            "num_hidden_layers": cfg.variant.num_hidden_layers,
            "train_epochs": cfg.variant.train_epochs,
            "beta_final": cfg.variant.beta_final,
            "kl_warmup_epochs": cfg.variant.kl_warmup_epochs,
        },
        "classifier_specs_by_center": {
            center: dict(reference.rows_by_center[center].selected_classifier_spec.payload)
            for center in reference.heldout_centers
        },
        "posterior_seed_policy": "stable_seed(split, variant_id, role)",
        "prior_label_budget_policy": "match_fit_label_counts",
        "selection_used_target_labels": False,
        "fit_used_target_center": False,
        "target_eval_labels_used_for_scoring_only": True,
        "is_router": False,
        "generated_embeddings_used": True,
        "cvae_checkpoint_used": True,
    }


def _decision_labels(
    cfg: MidogPPTunedClassifierPreservationConfig,
    rows: Sequence[Mapping[str, object]],
    leakage: Mapping[str, object],
) -> list[str]:
    labels = ["CLAIM_SCOPE_CVAE_PRESERVATION_ONLY", "TUNED_CLASSIFIER_SPECS_IMPORTED"]
    if leakage.get("status") != "PASS":
        return ["LEAKAGE_OR_ALIGNMENT_FAILURE_SUSPECT", "DO_NOT_INTERPRET"]
    summary = [row for row in rows if row.get("aggregation_level") == "summary"]
    decode = next((row for row in summary if row.get("representation_role") == DECODE_MU), None)
    if decode and decode.get("ratio_status") == "ok":
        ratio = _float(decode.get("preservation_ratio_vs_real_tuned"))
        if _finite(ratio) and float(ratio) >= 0.80:
            labels.append("DECODE_PRESERVATION_RATIO_PASS")
        else:
            labels.append("DECODE_PRESERVATION_RATIO_WEAK")
    else:
        labels.append("DECODE_PRESERVATION_RATIO_INVALID")
    labels.append("NO_ROUTING_OR_SYNTHETIC_UTILITY_CLAIM")
    return labels


def _write_decision_report(
    path: Path,
    cfg: MidogPPTunedClassifierPreservationConfig,
    reference: TunedClassifierReference,
    labels: Sequence[str],
    metrics: Sequence[Mapping[str, object]],
    leakage: Mapping[str, object],
) -> None:
    summary = [row for row in metrics if row.get("aggregation_level") == "summary"]
    lines = [
        "# MIDOG++ Tuned-Classifier CVAE Preservation v1",
        "",
        f"Decision labels: `{', '.join(labels)}`",
        "",
        f"- Leakage status: `{leakage.get('status')}`",
        f"- Real tuned reference root: `{reference.root}`",
        f"- Reference protocol hash: `{reference.protocol.get('protocol_hash', '')}`",
        f"- Variant: `{cfg.variant.variant_id}`",
        "",
        "| Representation | Mean BACC | Mean macro-F1 | Mean preservation ratio vs real tuned | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for row in summary:
        lines.append(
            "| "
            f"{row.get('representation_role')} | {_fmt(row.get('bacc'))} | {_fmt(row.get('macro_f1'))} | "
            f"{_fmt(row.get('preservation_ratio_vs_real_tuned'))} | {row.get('status')} |"
        )
    lines.extend(
        [
            "",
            "This artifact is a CVAE preservation surface only. It does not claim routing, expert selection, metadata compatibility, NELBO compatibility, controllable generation, or downstream synthetic utility.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _mean(values: Sequence[float]) -> float:
    valid = [float(value) for value in values if math.isfinite(float(value))]
    return sum(valid) / float(len(valid)) if valid else math.nan


def _std(values: Sequence[float]) -> float:
    valid = [float(value) for value in values if math.isfinite(float(value))]
    if not valid:
        return math.nan
    mean = _mean(valid)
    return math.sqrt(sum((value - mean) ** 2 for value in valid) / float(len(valid)))


def _min(values: Sequence[float]) -> float:
    valid = [float(value) for value in values if math.isfinite(float(value))]
    return min(valid) if valid else math.nan


def _max(values: Sequence[float]) -> float:
    valid = [float(value) for value in values if math.isfinite(float(value))]
    return max(valid) if valid else math.nan


def _fmt(value: object) -> str:
    number = _float(value)
    return "" if not math.isfinite(number) else f"{number:.6f}"


def _optional_str(value: object) -> str | None:
    if value in (None, ""):
        return None
    return str(value)
