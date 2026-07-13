"""MIDOG++ Virchow2 real-feature multi-axis learnability diagnostic."""

from __future__ import annotations

import csv
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from sklearn.exceptions import ConvergenceWarning  # type: ignore

from .features import FeatureCache, load_feature_cache
from .metrics import balanced_accuracy, macro_f1, nanmean
from .protocol import ProtocolError, bool_text


EXPERIMENT_NAME = "midogpp_virchow2_real_feature_multiaxis_baseline"
DEFAULT_ARTIFACTS_ROOT = (
    "artifacts/midogpp/10_real_feature_reference/"
    "midogpp_virchow2_real_feature_multiaxis_baseline/v1"
)
POSITIVE_LABEL_DEFAULT = 1
LOGISTIC_MODEL_TYPE = "logistic_regression"
MLP_MODEL_TYPE = "mlp"
VALID_STATUS = "valid"
UNKNOWN_VALUES = {"", "unknown", "unk", "na", "n/a", "none", "null"}

PER_AXIS_DOMAIN_COLUMNS = (
    "axis",
    "axis_role",
    "axis_high_confounding",
    "heldout_domain_id",
    "heldout_domain_name",
    "n_source",
    "n_eval",
    "n_source_pos",
    "n_source_neg",
    "n_eval_pos",
    "n_eval_neg",
    "bacc",
    "macro_f1",
    "precision_pos",
    "recall_pos",
    "f1_pos",
    "support_pos",
    "support_neg",
    "ci_low",
    "ci_high",
    "model_type",
    "model_seed",
    "converged",
    "status",
    "error_message",
)

AXIS_SUMMARY_COLUMNS = (
    "axis",
    "axis_role",
    "axis_high_confounding",
    "axis_scope",
    "model_type",
    "eligible_folds",
    "valid_folds",
    "valid_fraction",
    "decision_valid",
    "decision_status",
    "domain_equal_mean_bacc",
    "domain_equal_mean_macro_f1",
    "ci_overlap_050_count",
    "near_chance",
    "global_failure_gate_axis",
)

DOMAIN_AXIS_COUNT_COLUMNS = (
    "axis",
    "axis_role",
    "axis_high_confounding",
    "domain_id",
    "domain_name",
    "split",
    "n_rows",
    "n_pos",
    "n_neg",
    "unknown_value",
)

DOMAIN_METADATA_COLUMNS = (
    "axis",
    "axis_role",
    "axis_high_confounding",
    "domain_id",
    "domain_name",
    "n_train_rows",
    "n_train_pos",
    "n_train_neg",
)

OVERLAP_COLUMNS = (
    "axis",
    "heldout_domain_id",
    "heldout_domain_name",
    "sample_overlap_count",
    "case_overlap_count",
    "sample_overlap_preview",
    "case_overlap_preview",
)

PREDICTION_COLUMNS = (
    "axis",
    "heldout_domain_id",
    "heldout_domain_name",
    "model_type",
    "model_seed",
    "sample_id",
    "case_id",
    "y_true",
    "y_pred",
    "prob_pos",
)


@dataclass(frozen=True)
class AxisSpec:
    name: str
    role: str
    fields: tuple[str, ...]
    high_confounding: bool = False
    stress_test: bool = False
    global_failure_gate_axis: bool = True


DEFAULT_AXES: tuple[AxisSpec, ...] = (
    AxisSpec("tumor_type", "primary", ("tumor_type",)),
    AxisSpec("scanner_model", "secondary", ("scanner_model",)),
    AxisSpec("lab_or_origin", "secondary", ("lab_or_origin",), high_confounding=True),
    AxisSpec(
        "species",
        "secondary_descriptive",
        ("species",),
        high_confounding=True,
        global_failure_gate_axis=False,
    ),
    AxisSpec(
        "tumor_type|lab_or_origin|scanner_model",
        "stress_test",
        ("tumor_type", "lab_or_origin", "scanner_model"),
        high_confounding=True,
        stress_test=True,
        global_failure_gate_axis=False,
    ),
)


@dataclass(frozen=True)
class MidogPPMultiAxisConfig:
    manifest_path: str
    feature_cache_path: str
    artifacts_root: str = DEFAULT_ARTIFACTS_ROOT
    positive_label: int = POSITIVE_LABEL_DEFAULT
    label_mapping: Mapping[str, str] | None = None
    axes: tuple[AxisSpec, ...] = DEFAULT_AXES
    min_source: int = 20
    min_eval: int = 10
    min_source_pos: int = 10
    min_source_neg: int = 10
    min_eval_pos: int = 5
    min_eval_neg: int = 5
    min_valid_domains: int = 2
    min_valid_fold_fraction: float = 0.70
    mlp_min_source: int = 100
    mlp_min_source_pos: int = 20
    mlp_min_source_neg: int = 20
    mlp_seeds: tuple[int, ...] = (42, 43, 44)
    bootstrap_reps: int = 1000
    bootstrap_seed: int = 1337
    allow_npz_test_cache: bool = False


@dataclass(frozen=True)
class MidogPPMultiAxisResult:
    output_paths: Mapping[str, Path]
    decision_labels: tuple[str, ...]


@dataclass(frozen=True)
class _ManifestRow:
    row_index: int
    sample_id: str
    case_id: str
    label: int
    split: str
    metadata: Mapping[str, str]


def load_midogpp_multiaxis_config(path: Path) -> MidogPPMultiAxisConfig:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise ProtocolError("Loading MIDOG++ multi-axis YAML configs requires PyYAML.") from exc
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ProtocolError(f"Config must be a mapping: {path}")
    experiment = _mapping(payload.get("experiment"), "experiment")
    if str(experiment.get("name", "")) != EXPERIMENT_NAME:
        raise ProtocolError(f"Unexpected experiment.name: {experiment.get('name')!r}")
    inputs = _mapping(payload.get("inputs"), "inputs")
    thresholds = _mapping(payload.get("validity_thresholds", {}), "validity_thresholds", allow_empty=True)
    mlp = _mapping(payload.get("mlp_sensitivity", {}), "mlp_sensitivity", allow_empty=True)
    bootstrap = _mapping(payload.get("bootstrap", {}), "bootstrap", allow_empty=True)
    output = _mapping(payload.get("output", {}), "output", allow_empty=True)
    return MidogPPMultiAxisConfig(
        manifest_path=str(inputs.get("manifest_path", "")),
        feature_cache_path=str(inputs.get("feature_cache_path", "")),
        artifacts_root=str(output.get("artifacts_root", DEFAULT_ARTIFACTS_ROOT)),
        positive_label=int(payload.get("positive_label", POSITIVE_LABEL_DEFAULT)),
        label_mapping={
            str(k): str(v) for k, v in _mapping(payload.get("label_mapping", {}), "label_mapping", allow_empty=True).items()
        }
        or None,
        min_source=int(thresholds.get("min_source", 20)),
        min_eval=int(thresholds.get("min_eval", 10)),
        min_source_pos=int(thresholds.get("min_source_pos", 10)),
        min_source_neg=int(thresholds.get("min_source_neg", 10)),
        min_eval_pos=int(thresholds.get("min_eval_pos", 5)),
        min_eval_neg=int(thresholds.get("min_eval_neg", 5)),
        min_valid_domains=int(thresholds.get("min_valid_domains", 2)),
        min_valid_fold_fraction=float(thresholds.get("min_valid_fold_fraction", 0.70)),
        mlp_min_source=int(mlp.get("min_source", 100)),
        mlp_min_source_pos=int(mlp.get("min_source_pos", 20)),
        mlp_min_source_neg=int(mlp.get("min_source_neg", 20)),
        mlp_seeds=tuple(int(v) for v in mlp.get("seeds", [42, 43, 44])),
        bootstrap_reps=int(bootstrap.get("reps", 1000)),
        bootstrap_seed=int(bootstrap.get("seed", 1337)),
        allow_npz_test_cache=bool(inputs.get("allow_npz_test_cache", False)),
    )


def run_midogpp_multiaxis_baseline(
    *,
    config: MidogPPMultiAxisConfig,
    repo_root: Path,
) -> MidogPPMultiAxisResult:
    artifact_root = _resolve_path(repo_root, config.artifacts_root)
    tables_dir = artifact_root / "tables"
    manifests_dir = artifact_root / "manifests"
    reports_dir = artifact_root / "reports"
    for directory in (tables_dir, manifests_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest_path = _resolve_path(repo_root, config.manifest_path)
    feature_cache_path = _resolve_path(repo_root, config.feature_cache_path)
    manifest_rows = _read_manifest(manifest_path, positive_label=config.positive_label)
    train_rows = tuple(row for row in manifest_rows if row.split == "train")
    if not train_rows:
        raise ProtocolError(f"MIDOG++ manifest has no split=train rows: {manifest_path}")
    cache = load_feature_cache(feature_cache_path)
    _assert_virchow2_cache(cache, path=feature_cache_path, allow_npz_test_cache=config.allow_npz_test_cache)
    embeddings = _to_numpy(cache.embeddings)
    feature_dim = int(embeddings.shape[1])
    _assert_cache_alignment(train_rows, cache)

    domain_maps = _domain_maps(train_rows, config.axes)
    count_rows = _domain_axis_count_rows(manifest_rows, config.axes, config.positive_label, domain_maps)
    metadata_rows = _domain_metadata_rows(train_rows, config.axes, config.positive_label, domain_maps)
    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    overlap_rows: list[dict[str, object]] = []
    for axis in config.axes:
        axis_map = domain_maps[axis.name]
        for domain_name, domain_id in axis_map.items():
            logistic_row, logistic_predictions, overlap = _evaluate_fold_model(
                config=config,
                axis=axis,
                domain_id=domain_id,
                domain_name=domain_name,
                rows=train_rows,
                embeddings=embeddings,
                model_type=LOGISTIC_MODEL_TYPE,
                model_seed=None,
            )
            fold_rows.append(logistic_row)
            prediction_rows.extend(logistic_predictions)
            overlap_rows.append(overlap)
            for seed in config.mlp_seeds:
                mlp_row, mlp_predictions, _mlp_overlap = _evaluate_fold_model(
                    config=config,
                    axis=axis,
                    domain_id=domain_id,
                    domain_name=domain_name,
                    rows=train_rows,
                    embeddings=embeddings,
                    model_type=MLP_MODEL_TYPE,
                    model_seed=int(seed),
                )
                fold_rows.append(mlp_row)
                prediction_rows.extend(mlp_predictions)

    summary_rows = _axis_summary_rows(fold_rows, config)
    leakage = _leakage_report(fold_rows, overlap_rows)
    protocol_manifest = _protocol_manifest(
        config=config,
        manifest_path=manifest_path,
        feature_cache_path=feature_cache_path,
        feature_dim=feature_dim,
        train_rows=len(train_rows),
        cache=cache,
    )
    domain_axis_manifest = _domain_axis_manifest(config.axes, domain_maps)

    output_paths = {
        "per_axis_domain_metrics": tables_dir / "per_axis_domain_metrics.csv",
        "axis_summary": tables_dir / "axis_summary.csv",
        "domain_axis_counts": tables_dir / "domain_axis_counts.csv",
        "domain_metadata_map": tables_dir / "domain_metadata_map.csv",
        "source_target_identity_overlap": tables_dir / "source_target_identity_overlap.csv",
        "predictions": tables_dir / "predictions.csv",
        "domain_axis_manifest": manifests_dir / "domain_axis_manifest.json",
        "protocol_manifest": manifests_dir / "protocol_manifest.json",
        "leakage_report": reports_dir / "leakage_report.json",
        "per_axis_decision_report": reports_dir / "per_axis_decision_report.md",
        "decision_report": reports_dir / "decision_report.md",
    }
    _write_csv(output_paths["per_axis_domain_metrics"], PER_AXIS_DOMAIN_COLUMNS, fold_rows)
    _write_csv(output_paths["axis_summary"], AXIS_SUMMARY_COLUMNS, summary_rows)
    _write_csv(output_paths["domain_axis_counts"], DOMAIN_AXIS_COUNT_COLUMNS, count_rows)
    _write_csv(output_paths["domain_metadata_map"], DOMAIN_METADATA_COLUMNS, metadata_rows)
    _write_csv(output_paths["source_target_identity_overlap"], OVERLAP_COLUMNS, overlap_rows)
    _write_csv(output_paths["predictions"], PREDICTION_COLUMNS, prediction_rows)
    _write_json(output_paths["domain_axis_manifest"], domain_axis_manifest)
    _write_json(output_paths["protocol_manifest"], protocol_manifest)
    _write_json(output_paths["leakage_report"], leakage)
    _write_axis_report(output_paths["per_axis_decision_report"], summary_rows, fold_rows)
    decision_labels = _decision_labels(summary_rows)
    _write_decision_report(output_paths["decision_report"], summary_rows, leakage, decision_labels)
    return MidogPPMultiAxisResult(output_paths=output_paths, decision_labels=tuple(decision_labels))


def _read_manifest(path: Path, *, positive_label: int) -> tuple[_ManifestRow, ...]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ProtocolError(f"Empty MIDOG++ manifest: {path}")
        required = {"sample_id", "case_id", "label", "split", "scanner_model", "tumor_type", "lab_or_origin", "species"}
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ProtocolError(f"MIDOG++ manifest missing required columns: {missing}")
        rows = []
        for idx, row in enumerate(reader):
            label_raw = str(row.get("label", "")).strip()
            try:
                raw_label = int(float(label_raw))
            except ValueError as exc:
                raise ProtocolError(f"Invalid label in manifest row {idx}: {label_raw!r}") from exc
            rows.append(
                _ManifestRow(
                    row_index=idx,
                    sample_id=_clean_required(row.get("sample_id"), "sample_id", idx),
                    case_id=_clean_required(row.get("case_id"), "case_id", idx),
                    label=1 if raw_label == int(positive_label) else 0,
                    split=str(row.get("split", "")).strip().lower(),
                    metadata={str(k): str(v).strip() for k, v in row.items()},
                )
            )
    return tuple(rows)


def _assert_virchow2_cache(cache: FeatureCache, *, path: Path, allow_npz_test_cache: bool) -> None:
    extractor = cache.feature_extractor if isinstance(cache.feature_extractor, Mapping) else {}
    text = " ".join(
        str(extractor.get(key, "")) for key in ("backbone_type", "model_ref", "feature_extractor_name", "loader")
    ).lower()
    if "virchow2" in text:
        return
    if allow_npz_test_cache and str(path).endswith(".npz"):
        return
    raise ProtocolError(
        f"cache_alignment_failed: feature cache does not declare Virchow2 provenance: {path}"
    )


def _assert_cache_alignment(rows: Sequence[_ManifestRow], cache: FeatureCache) -> None:
    metadata = tuple(cache.metadata)
    if len(rows) != len(metadata):
        raise ProtocolError(
            f"cache_alignment_failed: train manifest rows={len(rows)} cache rows={len(metadata)}"
        )
    for idx, (row, meta) in enumerate(zip(rows, metadata)):
        cache_sample = str(meta.get("sample_id", "")).strip()
        if row.sample_id != cache_sample:
            raise ProtocolError(
                f"cache_alignment_failed: row {idx} sample_id manifest={row.sample_id!r} cache={cache_sample!r}"
            )
        cache_label = meta.get("label")
        if cache_label is not None and int(float(str(cache_label))) != int(row.label):
            raise ProtocolError(
                f"cache_alignment_failed: row {idx} label manifest={row.label!r} cache={cache_label!r}"
            )


def _evaluate_fold_model(
    *,
    config: MidogPPMultiAxisConfig,
    axis: AxisSpec,
    domain_id: int,
    domain_name: str,
    rows: Sequence[_ManifestRow],
    embeddings: Any,
    model_type: str,
    model_seed: int | None,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    import numpy as np  # type: ignore

    values = [_axis_value(row, axis) for row in rows]
    source_idx = [idx for idx, value in enumerate(values) if value is not None and value != domain_name]
    eval_idx = [idx for idx, value in enumerate(values) if value == domain_name]
    counts = _fold_counts(rows, source_idx, eval_idx)
    status, error = _fold_status(config, rows, source_idx, eval_idx, model_type=model_type)
    overlap = _overlap_row(axis, domain_id, domain_name, rows, source_idx, eval_idx)
    base = {
        "axis": axis.name,
        "axis_role": axis.role,
        "axis_high_confounding": bool_text(axis.high_confounding),
        "heldout_domain_id": int(domain_id),
        "heldout_domain_name": domain_name,
        **counts,
        "bacc": math.nan,
        "macro_f1": math.nan,
        "precision_pos": math.nan,
        "recall_pos": math.nan,
        "f1_pos": math.nan,
        "support_pos": counts["n_eval_pos"],
        "support_neg": counts["n_eval_neg"],
        "ci_low": math.nan,
        "ci_high": math.nan,
        "model_type": model_type,
        "model_seed": "" if model_seed is None else int(model_seed),
        "converged": "false",
        "status": status,
        "error_message": error,
    }
    if status != VALID_STATUS:
        return base, [], overlap

    x_source = np.asarray(embeddings, dtype=float)[source_idx]
    y_source = np.asarray([rows[idx].label for idx in source_idx], dtype=int)
    x_eval = np.asarray(embeddings, dtype=float)[eval_idx]
    y_eval = [int(rows[idx].label) for idx in eval_idx]
    try:
        probabilities, predictions, converged = _fit_predict(
            x_source=x_source,
            y_source=y_source,
            x_eval=x_eval,
            model_type=model_type,
            seed=model_seed,
        )
    except Exception as exc:
        failed = dict(base)
        failed.update({"status": "model_failed_convergence", "error_message": str(exc), "converged": "false"})
        return failed, [], overlap
    if not converged:
        failed = dict(base)
        failed.update({"status": "model_failed_convergence", "error_message": "sklearn convergence warning", "converged": "false"})
        return failed, [], overlap

    metric_values = _classification_metrics(y_eval, predictions)
    ci_low, ci_high = _bootstrap_bacc_ci(
        y_eval,
        predictions,
        reps=config.bootstrap_reps,
        seed=_stable_seed(config.bootstrap_seed, axis.name, domain_name, model_type, str(model_seed or "")),
    )
    out = dict(base)
    out.update(
        {
            **metric_values,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "converged": "true",
            "status": VALID_STATUS,
            "error_message": "",
        }
    )
    prediction_rows = [
        {
            "axis": axis.name,
            "heldout_domain_id": int(domain_id),
            "heldout_domain_name": domain_name,
            "model_type": model_type,
            "model_seed": "" if model_seed is None else int(model_seed),
            "sample_id": rows[idx].sample_id,
            "case_id": rows[idx].case_id,
            "y_true": y_eval[pos],
            "y_pred": int(predictions[pos]),
            "prob_pos": float(probabilities[pos]),
        }
        for pos, idx in enumerate(eval_idx)
    ]
    return out, prediction_rows, overlap


def _fold_status(
    config: MidogPPMultiAxisConfig,
    rows: Sequence[_ManifestRow],
    source_idx: Sequence[int],
    eval_idx: Sequence[int],
    *,
    model_type: str,
) -> tuple[str, str]:
    sample_overlap = set(rows[idx].sample_id for idx in source_idx).intersection(rows[idx].sample_id for idx in eval_idx)
    if sample_overlap:
        return "protocol_failed_sample_overlap", _preview(sample_overlap)
    case_overlap = set(rows[idx].case_id for idx in source_idx).intersection(rows[idx].case_id for idx in eval_idx)
    if case_overlap:
        return "protocol_failed_case_overlap", _preview(case_overlap)
    counts = _fold_counts(rows, source_idx, eval_idx)
    for key, status in (
        ("n_source", "invalid_too_few_source"),
        ("n_eval", "invalid_too_few_eval"),
        ("n_source_pos", "invalid_too_few_source_pos"),
        ("n_source_neg", "invalid_too_few_source_neg"),
        ("n_eval_pos", "invalid_too_few_eval_pos"),
        ("n_eval_neg", "invalid_too_few_eval_neg"),
    ):
        minimum = {
            "n_source": config.min_source,
            "n_eval": config.min_eval,
            "n_source_pos": config.min_source_pos,
            "n_source_neg": config.min_source_neg,
            "n_eval_pos": config.min_eval_pos,
            "n_eval_neg": config.min_eval_neg,
        }[key]
        if int(counts[key]) < int(minimum):
            return status, f"{key}={counts[key]} minimum={minimum}"
    if model_type == MLP_MODEL_TYPE:
        if (
            int(counts["n_source"]) < int(config.mlp_min_source)
            or int(counts["n_source_pos"]) < int(config.mlp_min_source_pos)
            or int(counts["n_source_neg"]) < int(config.mlp_min_source_neg)
        ):
            return (
                "skipped_insufficient_support_for_mlp",
                (
                    f"n_source={counts['n_source']} n_source_pos={counts['n_source_pos']} "
                    f"n_source_neg={counts['n_source_neg']}"
                ),
            )
    return VALID_STATUS, ""


def _fit_predict(
    *,
    x_source: Any,
    y_source: Any,
    x_eval: Any,
    model_type: str,
    seed: int | None,
) -> tuple[list[float], list[int], bool]:
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.neural_network import MLPClassifier  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore

    scaler = StandardScaler()
    x_source_scaled = scaler.fit_transform(x_source)
    x_eval_scaled = scaler.transform(x_eval)
    if model_type == LOGISTIC_MODEL_TYPE:
        clf = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=2000,
            random_state=0,
        )
    elif model_type == MLP_MODEL_TYPE:
        clf = MLPClassifier(
            hidden_layer_sizes=(128,),
            alpha=1.0e-4,
            random_state=int(seed if seed is not None else 0),
            max_iter=2000,
            early_stopping=False,
        )
    else:
        raise ProtocolError(f"Unknown model_type={model_type!r}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(x_source_scaled, y_source)
    converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
    classes = tuple(int(value) for value in clf.classes_.tolist())
    if classes != (0, 1):
        raise ProtocolError(f"class order must be (0, 1), got {classes}")
    proba = clf.predict_proba(x_eval_scaled)[:, 1]
    pred = clf.predict(x_eval_scaled)
    return [float(value) for value in proba.tolist()], [int(value) for value in pred.tolist()], converged


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


def _bootstrap_bacc_ci(y_true: Sequence[int], y_pred: Sequence[int], *, reps: int, seed: int) -> tuple[float, float]:
    import numpy as np  # type: ignore

    pos = [idx for idx, value in enumerate(y_true) if int(value) == 1]
    neg = [idx for idx, value in enumerate(y_true) if int(value) == 0]
    if not pos or not neg or int(reps) <= 0:
        return math.nan, math.nan
    rng = np.random.default_rng(int(seed))
    values = []
    for _ in range(int(reps)):
        sampled = list(rng.choice(pos, size=len(pos), replace=True)) + list(rng.choice(neg, size=len(neg), replace=True))
        values.append(balanced_accuracy([y_true[idx] for idx in sampled], [y_pred[idx] for idx in sampled]))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def _axis_summary_rows(rows: Sequence[Mapping[str, object]], config: MidogPPMultiAxisConfig) -> list[dict[str, object]]:
    out = []
    by_axis_model: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    axis_by_name = {axis.name: axis for axis in config.axes}
    for row in rows:
        by_axis_model.setdefault((str(row["axis"]), str(row["model_type"])), []).append(row)
    for (axis_name, model_type), group in sorted(by_axis_model.items()):
        axis = axis_by_name[axis_name]
        eligible = len([row for row in group if not str(row.get("status", "")).startswith("skipped_")])
        valid = [row for row in group if str(row.get("status")) == VALID_STATUS]
        valid_fraction = float(len(valid)) / float(eligible) if eligible else 0.0
        decision_valid = len(valid) >= int(config.min_valid_domains) and valid_fraction >= float(config.min_valid_fold_fraction)
        ci_overlap = [
            row
            for row in valid
            if _finite(row.get("ci_low")) and _finite(row.get("ci_high")) and float(row["ci_low"]) <= 0.5 <= float(row["ci_high"])
        ]
        mean_bacc = nanmean([row.get("bacc") for row in valid])
        mean_macro = nanmean([row.get("macro_f1") for row in valid])
        most_ci_overlap = len(ci_overlap) > (len(valid) / 2.0) if valid else False
        near_chance = bool(decision_valid and _finite(mean_bacc) and mean_bacc <= 0.55 and most_ci_overlap)
        if model_type != LOGISTIC_MODEL_TYPE:
            decision_status = "SENSITIVITY_ONLY"
        elif not decision_valid:
            decision_status = "INSUFFICIENT_VALID_FOLDS"
        elif near_chance:
            decision_status = "NEAR_CHANCE"
        else:
            decision_status = "TRANSFER_SIGNAL_PRESENT"
        out.append(
            {
                "axis": axis_name,
                "axis_role": axis.role,
                "axis_high_confounding": bool_text(axis.high_confounding),
                "axis_scope": "stress_test" if axis.stress_test else "simple",
                "model_type": model_type,
                "eligible_folds": int(eligible),
                "valid_folds": int(len(valid)),
                "valid_fraction": valid_fraction,
                "decision_valid": bool_text(decision_valid),
                "decision_status": decision_status,
                "domain_equal_mean_bacc": mean_bacc,
                "domain_equal_mean_macro_f1": mean_macro,
                "ci_overlap_050_count": int(len(ci_overlap)),
                "near_chance": bool_text(near_chance),
                "global_failure_gate_axis": bool_text(axis.global_failure_gate_axis and model_type == LOGISTIC_MODEL_TYPE),
            }
        )
    return out


def _decision_labels(summary_rows: Sequence[Mapping[str, object]]) -> list[str]:
    labels: list[str] = []
    logistic = [row for row in summary_rows if str(row.get("model_type")) == LOGISTIC_MODEL_TYPE]
    gate_axes = [row for row in logistic if str(row.get("global_failure_gate_axis")) == "true" and str(row.get("decision_valid")) == "true"]
    if not gate_axes:
        labels.append("NO_DECISION_VALID_SIMPLE_GATE_AXES")
    elif all(str(row.get("near_chance")) == "true" for row in gate_axes):
        labels.append("ALL_DECISION_VALID_SIMPLE_GATE_AXES_NEAR_CHANCE")
    elif any(str(row.get("decision_status")) == "TRANSFER_SIGNAL_PRESENT" for row in gate_axes):
        labels.append("REAL_FEATURE_TRANSFER_SIGNAL_PRESENT")
    if any(str(row.get("axis_scope")) == "stress_test" and str(row.get("decision_status")) == "INSUFFICIENT_VALID_FOLDS" for row in logistic):
        labels.append("COMPOSITE_STRESS_TEST_INSUFFICIENT_VALID_FOLDS")
    return labels


def _domain_maps(rows: Sequence[_ManifestRow], axes: Sequence[AxisSpec]) -> dict[str, dict[str, int]]:
    out = {}
    for axis in axes:
        values = sorted({value for row in rows if (value := _axis_value(row, axis)) is not None})
        out[axis.name] = {value: idx for idx, value in enumerate(values)}
    return out


def _domain_axis_count_rows(
    rows: Sequence[_ManifestRow],
    axes: Sequence[AxisSpec],
    positive_label: int,
    domain_maps: Mapping[str, Mapping[str, int]],
) -> list[dict[str, object]]:
    del positive_label
    out = []
    for axis in axes:
        values = sorted({(_axis_value(row, axis) or "__UNKNOWN__") for row in rows})
        for value in values:
            domain_rows = [row for row in rows if (_axis_value(row, axis) or "__UNKNOWN__") == value]
            for split in sorted({row.split for row in domain_rows}):
                split_rows = [row for row in domain_rows if row.split == split]
                unknown = value == "__UNKNOWN__"
                out.append(
                    {
                        "axis": axis.name,
                        "axis_role": axis.role,
                        "axis_high_confounding": bool_text(axis.high_confounding),
                        "domain_id": "" if unknown else domain_maps[axis.name][value],
                        "domain_name": "" if unknown else value,
                        "split": split,
                        "n_rows": len(split_rows),
                        "n_pos": sum(1 for row in split_rows if row.label == 1),
                        "n_neg": sum(1 for row in split_rows if row.label == 0),
                        "unknown_value": bool_text(unknown),
                    }
                )
    return out


def _domain_metadata_rows(
    rows: Sequence[_ManifestRow],
    axes: Sequence[AxisSpec],
    positive_label: int,
    domain_maps: Mapping[str, Mapping[str, int]],
) -> list[dict[str, object]]:
    del positive_label
    out = []
    for axis in axes:
        for domain_name, domain_id in domain_maps[axis.name].items():
            domain_rows = [row for row in rows if _axis_value(row, axis) == domain_name]
            out.append(
                {
                    "axis": axis.name,
                    "axis_role": axis.role,
                    "axis_high_confounding": bool_text(axis.high_confounding),
                    "domain_id": int(domain_id),
                    "domain_name": domain_name,
                    "n_train_rows": len(domain_rows),
                    "n_train_pos": sum(1 for row in domain_rows if row.label == 1),
                    "n_train_neg": sum(1 for row in domain_rows if row.label == 0),
                }
            )
    return out


def _axis_value(row: _ManifestRow, axis: AxisSpec) -> str | None:
    parts = []
    for field in axis.fields:
        value = _clean_axis_value(row.metadata.get(field, ""))
        if value is None:
            return None
        parts.append(value)
    return "|".join(parts)


def _clean_axis_value(value: object) -> str | None:
    text = str(value if value is not None else "").strip()
    if text.lower() in UNKNOWN_VALUES:
        return None
    return text


def _fold_counts(
    rows: Sequence[_ManifestRow],
    source_idx: Sequence[int],
    eval_idx: Sequence[int],
) -> dict[str, int]:
    source_labels = [rows[idx].label for idx in source_idx]
    eval_labels = [rows[idx].label for idx in eval_idx]
    return {
        "n_source": len(source_idx),
        "n_eval": len(eval_idx),
        "n_source_pos": sum(1 for value in source_labels if value == 1),
        "n_source_neg": sum(1 for value in source_labels if value == 0),
        "n_eval_pos": sum(1 for value in eval_labels if value == 1),
        "n_eval_neg": sum(1 for value in eval_labels if value == 0),
    }


def _overlap_row(
    axis: AxisSpec,
    domain_id: int,
    domain_name: str,
    rows: Sequence[_ManifestRow],
    source_idx: Sequence[int],
    eval_idx: Sequence[int],
) -> dict[str, object]:
    source_samples = {rows[idx].sample_id for idx in source_idx}
    eval_samples = {rows[idx].sample_id for idx in eval_idx}
    source_cases = {rows[idx].case_id for idx in source_idx}
    eval_cases = {rows[idx].case_id for idx in eval_idx}
    sample_overlap = sorted(source_samples.intersection(eval_samples))
    case_overlap = sorted(source_cases.intersection(eval_cases))
    return {
        "axis": axis.name,
        "heldout_domain_id": int(domain_id),
        "heldout_domain_name": domain_name,
        "sample_overlap_count": len(sample_overlap),
        "case_overlap_count": len(case_overlap),
        "sample_overlap_preview": "|".join(sample_overlap[:10]),
        "case_overlap_preview": "|".join(case_overlap[:10]),
    }


def _leakage_report(rows: Sequence[Mapping[str, object]], overlap_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    protocol_rows = [
        row
        for row in rows
        if str(row.get("status")) in {"protocol_failed_sample_overlap", "protocol_failed_case_overlap"}
    ]
    return {
        "status": "PASS" if not protocol_rows else "FOLD_PROTOCOL_FAILURES_REPORTED",
        "protocol_failed_fold_count": len(protocol_rows),
        "sample_overlap_fold_count": sum(1 for row in rows if str(row.get("status")) == "protocol_failed_sample_overlap"),
        "case_overlap_fold_count": sum(1 for row in rows if str(row.get("status")) == "protocol_failed_case_overlap"),
        "overlap_rows": [dict(row) for row in overlap_rows],
        "target_labels_used_for_fitting": False,
        "target_labels_used_for_scoring_only": True,
    }


def _protocol_manifest(
    *,
    config: MidogPPMultiAxisConfig,
    manifest_path: Path,
    feature_cache_path: Path,
    feature_dim: int,
    train_rows: int,
    cache: FeatureCache,
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_virchow2_real_feature_multiaxis_protocol_v1",
        "experiment_name": EXPERIMENT_NAME,
        "manifest_path": str(manifest_path),
        "feature_cache_path": str(feature_cache_path),
        "train_rows": int(train_rows),
        "feature_dim": int(feature_dim),
        "feature_extractor": dict(cache.feature_extractor),
        "positive_label": int(config.positive_label),
        "label_mapping": dict(config.label_mapping or {"0": "negative", "1": "mitotic_positive"}),
        "split_scope": "train_only",
        "target_domain_rows_used_for_fitting": False,
        "target_domain_labels_used_for_scoring_only": True,
        "cache_building_in_scope": False,
        "validity_thresholds": {
            "min_source": config.min_source,
            "min_eval": config.min_eval,
            "min_source_pos": config.min_source_pos,
            "min_source_neg": config.min_source_neg,
            "min_eval_pos": config.min_eval_pos,
            "min_eval_neg": config.min_eval_neg,
            "min_valid_domains": config.min_valid_domains,
            "min_valid_fold_fraction": config.min_valid_fold_fraction,
        },
        "mlp_sensitivity_gate": {
            "min_source": config.mlp_min_source,
            "min_source_pos": config.mlp_min_source_pos,
            "min_source_neg": config.mlp_min_source_neg,
            "early_stopping": False,
            "seeds": list(config.mlp_seeds),
        },
    }


def _domain_axis_manifest(axes: Sequence[AxisSpec], domain_maps: Mapping[str, Mapping[str, int]]) -> dict[str, object]:
    return {
        "axes": [
            {
                "axis": axis.name,
                "axis_role": axis.role,
                "fields": list(axis.fields),
                "axis_high_confounding": axis.high_confounding,
                "stress_test": axis.stress_test,
                "global_failure_gate_axis": axis.global_failure_gate_axis,
                "domains": [
                    {"domain_id": int(domain_id), "domain_name": domain_name}
                    for domain_name, domain_id in domain_maps[axis.name].items()
                ],
            }
            for axis in axes
        ]
    }


def _write_axis_report(path: Path, summary_rows: Sequence[Mapping[str, object]], fold_rows: Sequence[Mapping[str, object]]) -> None:
    lines = ["# MIDOG++ Virchow2 Multi-Axis Per-Axis Decisions", ""]
    for row in summary_rows:
        if str(row.get("model_type")) != LOGISTIC_MODEL_TYPE:
            continue
        lines.append(
            (
                f"- `{row['axis']}` ({row['axis_role']}): {row['decision_status']}; "
                f"valid={row['valid_folds']}/{row['eligible_folds']}, "
                f"mean BACC={_fmt(row['domain_equal_mean_bacc'])}, "
                f"near_chance={row['near_chance']}, "
                f"high_confounding={row['axis_high_confounding']}"
            )
        )
    failed = [row for row in fold_rows if str(row.get("status")).startswith("protocol_failed_")]
    if failed:
        lines.extend(["", "## Protocol-Failed Folds", ""])
        for row in failed:
            lines.append(
                f"- `{row['axis']}` domain `{row['heldout_domain_name']}`: {row['status']} ({row['error_message']})"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_decision_report(
    path: Path,
    summary_rows: Sequence[Mapping[str, object]],
    leakage: Mapping[str, object],
    decision_labels: Sequence[str],
) -> None:
    lines = [
        "# MIDOG++ Virchow2 Real-Feature Multi-Axis Learnability Baseline",
        "",
        f"- Decision labels: `{', '.join(decision_labels) if decision_labels else 'NONE'}`",
        f"- Leakage report status: `{leakage.get('status')}`",
        "- Metadata axes define heldout-domain queries only; success is measured by downstream heldout utility.",
        "- Composite tumor/lab/scanner domains are stress-test diagnostics and cannot prove global Virchow2 failure alone.",
        "- Lab/origin and species are confounded metadata-shift diagnostics, not isolated causal effects.",
        "",
        "## Logistic Axis Summary",
        "",
        "| Axis | Role | Decision | Valid folds | Mean BACC | Near chance |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in summary_rows:
        if str(row.get("model_type")) != LOGISTIC_MODEL_TYPE:
            continue
        lines.append(
            (
                f"| {row['axis']} | {row['axis_role']} | {row['decision_status']} | "
                f"{row['valid_folds']}/{row['eligible_folds']} | {_fmt(row['domain_equal_mean_bacc'])} | "
                f"{row['near_chance']} |"
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mapping(value: object, name: str, *, allow_empty: bool = False) -> Mapping[str, Any]:
    if value is None and allow_empty:
        return {}
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{name} must be a mapping")
    return value


def _resolve_path(repo_root: Path, raw: str) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else repo_root / path


def _to_numpy(values: object) -> Any:
    import numpy as np  # type: ignore

    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ProtocolError(f"Feature cache embeddings must be 2D, got shape={array.shape}")
    return array


def _clean_required(value: object, field: str, idx: int) -> str:
    text = str(value if value is not None else "").strip()
    if not text:
        raise ProtocolError(f"Manifest row {idx} missing required {field}")
    return text


def _preview(values: Sequence[str] | set[str]) -> str:
    return "|".join(sorted(str(value) for value in values)[:10])


def _csv_value(value: object) -> object:
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, bool):
        return bool_text(value)
    return value


def _fmt(value: object) -> str:
    if not _finite(value):
        return "nan"
    return f"{float(value):.4f}"


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _stable_seed(*parts: object) -> int:
    import hashlib

    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:12], 16)
