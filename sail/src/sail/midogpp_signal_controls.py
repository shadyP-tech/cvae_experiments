"""MIDOG++ Virchow2 real-feature signal-control diagnostic."""

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
from .metrics import balanced_accuracy, macro_f1, nanmean, nanmin, nanstd
from .midogpp_multiaxis import (
    MLP_MODEL_TYPE,
    UNKNOWN_VALUES,
    VALID_STATUS,
    _assert_cache_alignment,
    _assert_virchow2_cache,
    _clean_axis_value,
    _classification_metrics,
    _csv_value,
    _finite,
    _fmt,
    _mapping,
    _ManifestRow,
    _read_manifest,
    _resolve_path,
    _stable_seed,
    _to_numpy,
    _write_csv,
    _write_json,
)
from .protocol import ProtocolError, bool_text


EXPERIMENT_NAME = "midogpp_virchow2_real_feature_signal_controls"
DEFAULT_ARTIFACTS_ROOT = "sail/artifacts/midogpp_virchow2_real_feature_signal_controls"
LOGISTIC_MODEL_TYPE = "logistic_regression"
POOLED_CONTROL = "pooled_case_disjoint_control"
BALANCED_CONTROL = "pooled_tumor_class_balanced_case_split"
WITHIN_TUMOR_CONTROL = "within_tumor_case_disjoint_control"
LABEL_PERMUTATION_CONTROL = "label_permutation_control"
FEATURE_ROW_SHUFFLE_CONTROL = "feature_label_row_shuffle_control"
REAL_CONTROLS = {POOLED_CONTROL, BALANCED_CONTROL, WITHIN_TUMOR_CONTROL}
NEGATIVE_CONTROLS = {LABEL_PERMUTATION_CONTROL, FEATURE_ROW_SHUFFLE_CONTROL}


METRIC_COLUMNS = (
    "aggregation_level",
    "control_name",
    "domain_name",
    "linked_real_control",
    "split_seed",
    "model_type",
    "model_seed",
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
    "n_fit_pos_cases",
    "n_fit_neg_cases",
    "n_eval_pos_cases",
    "n_eval_neg_cases",
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
    "strong",
    "near_chance",
    "converged",
    "status",
    "error_message",
    "bacc_delta_vs_real",
)

STRATIFIED_COLUMNS = (
    "control_name",
    "domain_name",
    "split_seed",
    "model_type",
    "model_seed",
    "stratify_axis",
    "stratum_value",
    "n_eval",
    "n_eval_pos",
    "n_eval_neg",
    "bacc",
    "macro_f1",
    "precision_pos",
    "recall_pos",
    "f1_pos",
    "status",
    "error_message",
)

SPLIT_MANIFEST_COLUMNS = (
    "control_name",
    "domain_name",
    "split_seed",
    "subset",
    "row_index",
    "feature_row_index",
    "sample_id",
    "case_id",
    "image_path",
    "label",
    "tumor_type",
    "scanner_model",
    "lab_or_origin",
    "species",
)

IDENTITY_AUDIT_COLUMNS = (
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
    "model_type",
    "model_seed",
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


@dataclass(frozen=True)
class MidogPPSignalControlsConfig:
    manifest_path: str
    feature_cache_path: str
    artifacts_root: str = DEFAULT_ARTIFACTS_ROOT
    positive_label: int = 1
    label_mapping: Mapping[str, str] | None = None
    split_seeds: tuple[int, ...] = (42, 43, 44, 45, 46)
    eval_fraction: float = 0.20
    min_fit: int = 20
    min_eval: int = 10
    min_fit_pos: int = 10
    min_fit_neg: int = 10
    min_eval_pos: int = 5
    min_eval_neg: int = 5
    min_fit_pos_cases: int = 3
    min_fit_neg_cases: int = 3
    min_eval_pos_cases: int = 2
    min_eval_neg_cases: int = 2
    mlp_min_fit: int = 100
    mlp_min_fit_pos: int = 20
    mlp_min_fit_neg: int = 20
    mlp_seeds: tuple[int, ...] = (42, 43, 44)
    bootstrap_reps: int = 1000
    bootstrap_seed: int = 1337
    allow_npz_test_cache: bool = False
    prior_lodo_axis_summary_path: str | None = (
        "sail/artifacts/midogpp_virchow2_real_feature_multiaxis_baseline/tables/axis_summary.csv"
    )


@dataclass(frozen=True)
class MidogPPSignalControlsResult:
    output_paths: Mapping[str, Path]
    decision_labels: tuple[str, ...]


@dataclass(frozen=True)
class _SplitSpec:
    control_name: str
    split_seed: int
    fit_idx: tuple[int, ...]
    eval_idx: tuple[int, ...]
    domain_name: str = ""
    linked_real_control: str = ""


def load_midogpp_signal_controls_config(path: Path) -> MidogPPSignalControlsConfig:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise ProtocolError("Loading MIDOG++ signal-control YAML configs requires PyYAML.") from exc
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ProtocolError(f"Config must be a mapping: {path}")
    experiment = _mapping(payload.get("experiment"), "experiment")
    if str(experiment.get("name", "")) != EXPERIMENT_NAME:
        raise ProtocolError(f"Unexpected experiment.name: {experiment.get('name')!r}")
    inputs = _mapping(payload.get("inputs"), "inputs")
    split = _mapping(payload.get("split"), "split", allow_empty=True)
    thresholds = _mapping(payload.get("validity_thresholds", {}), "validity_thresholds", allow_empty=True)
    mlp = _mapping(payload.get("mlp_sensitivity", {}), "mlp_sensitivity", allow_empty=True)
    bootstrap = _mapping(payload.get("bootstrap", {}), "bootstrap", allow_empty=True)
    output = _mapping(payload.get("output", {}), "output", allow_empty=True)
    prior = _mapping(payload.get("prior_lodo", {}), "prior_lodo", allow_empty=True)
    prior_path = prior.get("axis_summary_path", MidogPPSignalControlsConfig.prior_lodo_axis_summary_path)
    return MidogPPSignalControlsConfig(
        manifest_path=str(inputs.get("manifest_path", "")),
        feature_cache_path=str(inputs.get("feature_cache_path", "")),
        artifacts_root=str(output.get("artifacts_root", DEFAULT_ARTIFACTS_ROOT)),
        positive_label=int(payload.get("positive_label", 1)),
        label_mapping={
            str(k): str(v) for k, v in _mapping(payload.get("label_mapping", {}), "label_mapping", allow_empty=True).items()
        }
        or None,
        split_seeds=tuple(int(v) for v in split.get("seeds", [42, 43, 44, 45, 46])),
        eval_fraction=float(split.get("eval_fraction", 0.20)),
        min_fit=int(thresholds.get("min_fit", 20)),
        min_eval=int(thresholds.get("min_eval", 10)),
        min_fit_pos=int(thresholds.get("min_fit_pos", 10)),
        min_fit_neg=int(thresholds.get("min_fit_neg", 10)),
        min_eval_pos=int(thresholds.get("min_eval_pos", 5)),
        min_eval_neg=int(thresholds.get("min_eval_neg", 5)),
        min_fit_pos_cases=int(thresholds.get("min_fit_pos_cases", 3)),
        min_fit_neg_cases=int(thresholds.get("min_fit_neg_cases", 3)),
        min_eval_pos_cases=int(thresholds.get("min_eval_pos_cases", 2)),
        min_eval_neg_cases=int(thresholds.get("min_eval_neg_cases", 2)),
        mlp_min_fit=int(mlp.get("min_fit", 100)),
        mlp_min_fit_pos=int(mlp.get("min_fit_pos", 20)),
        mlp_min_fit_neg=int(mlp.get("min_fit_neg", 20)),
        mlp_seeds=tuple(int(v) for v in mlp.get("seeds", [42, 43, 44])),
        bootstrap_reps=int(bootstrap.get("reps", 1000)),
        bootstrap_seed=int(bootstrap.get("seed", 1337)),
        allow_npz_test_cache=bool(inputs.get("allow_npz_test_cache", False)),
        prior_lodo_axis_summary_path=None if prior_path in ("", None) else str(prior_path),
    )


def run_midogpp_signal_controls(
    *,
    config: MidogPPSignalControlsConfig,
    repo_root: Path,
) -> MidogPPSignalControlsResult:
    artifact_root = _resolve_path(repo_root, config.artifacts_root)
    tables_dir = artifact_root / "tables"
    manifests_dir = artifact_root / "manifests"
    reports_dir = artifact_root / "reports"
    for directory in (tables_dir, manifests_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest_path = _resolve_path(repo_root, config.manifest_path)
    feature_cache_path = _resolve_path(repo_root, config.feature_cache_path)
    manifest_rows = _read_manifest(manifest_path, positive_label=config.positive_label)
    _assert_signal_manifest_fields(manifest_rows)
    train_rows = tuple(row for row in manifest_rows if row.split == "train")
    if not train_rows:
        raise ProtocolError(f"MIDOG++ manifest has no split=train rows: {manifest_path}")
    cache = load_feature_cache(feature_cache_path)
    _assert_virchow2_cache(cache, path=feature_cache_path, allow_npz_test_cache=config.allow_npz_test_cache)
    embeddings = _to_numpy(cache.embeddings)
    feature_dim = int(embeddings.shape[1])
    _assert_cache_alignment(train_rows, cache)

    split_specs = _build_split_specs(config, train_rows)
    identity_rows: list[dict[str, object]] = []
    split_manifest_rows: list[dict[str, object]] = []
    metric_seed_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    stratified_rows: list[dict[str, object]] = []

    for spec in split_specs:
        identity_rows.extend(_identity_audit_rows(train_rows, spec))
        split_manifest_rows.extend(_split_manifest_rows(train_rows, spec))
        model_jobs: list[tuple[str, int | None]] = [(LOGISTIC_MODEL_TYPE, int(spec.split_seed))]
        if spec.control_name in REAL_CONTROLS:
            model_jobs.extend((MLP_MODEL_TYPE, int(seed)) for seed in config.mlp_seeds)
        for model_type, model_seed in model_jobs:
            metric_row, preds = _evaluate_signal_split(
                config=config,
                rows=train_rows,
                embeddings=embeddings,
                spec=spec,
                model_type=model_type,
                model_seed=model_seed,
            )
            metric_seed_rows.append(metric_row)
            prediction_rows.extend(preds)
            if spec.control_name in REAL_CONTROLS and str(metric_row["status"]) == VALID_STATUS:
                stratified_rows.extend(_stratified_metric_rows(train_rows, preds, metric_row))

    control_rows, domain_rows, negative_rows = _summary_tables(metric_seed_rows)
    _attach_negative_deltas(control_rows, negative_rows)
    prior_lodo = _prior_lodo_status(repo_root, config)
    decision_labels = _decision_labels(control_rows, domain_rows, negative_rows, prior_lodo)
    leakage = _leakage_report(identity_rows, control_rows, domain_rows, negative_rows)
    protocol_manifest = _protocol_manifest(
        config=config,
        manifest_path=manifest_path,
        feature_cache_path=feature_cache_path,
        feature_dim=feature_dim,
        train_rows=len(train_rows),
        cache=cache,
        prior_lodo=prior_lodo,
    )

    output_paths = {
        "control_metrics": tables_dir / "control_metrics.csv",
        "domain_control_metrics": tables_dir / "domain_control_metrics.csv",
        "negative_control_metrics": tables_dir / "negative_control_metrics.csv",
        "stratified_metrics": tables_dir / "stratified_metrics.csv",
        "split_manifest": tables_dir / "split_manifest.csv",
        "identity_overlap_audit": tables_dir / "identity_overlap_audit.csv",
        "predictions": tables_dir / "predictions.csv",
        "protocol_manifest": manifests_dir / "protocol_manifest.json",
        "leakage_report": reports_dir / "leakage_report.json",
        "decision_report": reports_dir / "decision_report.md",
    }
    _write_csv(output_paths["control_metrics"], METRIC_COLUMNS, control_rows)
    _write_csv(output_paths["domain_control_metrics"], METRIC_COLUMNS, domain_rows)
    _write_csv(output_paths["negative_control_metrics"], METRIC_COLUMNS, negative_rows)
    _write_csv(output_paths["stratified_metrics"], STRATIFIED_COLUMNS, stratified_rows)
    _write_csv(output_paths["split_manifest"], SPLIT_MANIFEST_COLUMNS, split_manifest_rows)
    _write_csv(output_paths["identity_overlap_audit"], IDENTITY_AUDIT_COLUMNS, identity_rows)
    _write_csv(output_paths["predictions"], PREDICTION_COLUMNS, prediction_rows)
    _write_json(output_paths["protocol_manifest"], protocol_manifest)
    _write_json(output_paths["leakage_report"], leakage)
    _write_decision_report(output_paths["decision_report"], control_rows, domain_rows, negative_rows, leakage, decision_labels, prior_lodo)
    return MidogPPSignalControlsResult(output_paths=output_paths, decision_labels=tuple(decision_labels))


def _assert_signal_manifest_fields(rows: Sequence[_ManifestRow]) -> None:
    required = {"image_path", "tumor_type", "scanner_model", "lab_or_origin", "species"}
    missing = sorted(field for field in required if any(str(row.metadata.get(field, "")).strip() == "" for row in rows))
    if missing:
        raise ProtocolError(f"MIDOG++ signal-control manifest has missing required values: {missing}")


def _build_split_specs(config: MidogPPSignalControlsConfig, rows: Sequence[_ManifestRow]) -> list[_SplitSpec]:
    all_idx = tuple(range(len(rows)))
    out: list[_SplitSpec] = []
    for seed in config.split_seeds:
        fit_idx, eval_idx = _case_disjoint_split(rows, all_idx, config=config, seed=int(seed))
        out.append(_SplitSpec(POOLED_CONTROL, int(seed), fit_idx, eval_idx))
        balanced_fit, balanced_eval = _tumor_class_balanced_indices(rows, fit_idx, eval_idx, seed=int(seed))
        out.append(_SplitSpec(BALANCED_CONTROL, int(seed), balanced_fit, balanced_eval))
        out.append(_SplitSpec(LABEL_PERMUTATION_CONTROL, int(seed), fit_idx, eval_idx, linked_real_control=POOLED_CONTROL))
        out.append(_SplitSpec(FEATURE_ROW_SHUFFLE_CONTROL, int(seed), fit_idx, eval_idx, linked_real_control=POOLED_CONTROL))
    for tumor in _tumor_values(rows):
        tumor_idx = tuple(idx for idx, row in enumerate(rows) if _tumor_value(row) == tumor)
        for seed in config.split_seeds:
            fit_idx, eval_idx = _case_disjoint_split(rows, tumor_idx, config=config, seed=_stable_seed(seed, tumor))
            out.append(_SplitSpec(WITHIN_TUMOR_CONTROL, int(seed), fit_idx, eval_idx, domain_name=tumor))
    return out


def _case_disjoint_split(
    rows: Sequence[_ManifestRow],
    indices: Sequence[int],
    *,
    config: MidogPPSignalControlsConfig,
    seed: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    import numpy as np  # type: ignore

    if not indices:
        return (), ()
    grouped = _indices_by_case(rows, indices)
    cases = sorted(grouped)
    rng = np.random.default_rng(int(seed))
    best: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    best_score = -1
    for attempt in range(100):
        eval_cases: set[str] = set()
        pure_pos = [case for case in cases if _case_labels(rows, grouped[case]) == {1}]
        pure_neg = [case for case in cases if _case_labels(rows, grouped[case]) == {0}]
        mixed = [case for case in cases if case not in pure_pos and case not in pure_neg]
        for bucket, min_eval, min_fit in (
            (pure_pos, config.min_eval_pos_cases, config.min_fit_pos_cases),
            (pure_neg, config.min_eval_neg_cases, config.min_fit_neg_cases),
        ):
            shuffled = list(bucket)
            rng.shuffle(shuffled)
            target = max(int(math.ceil(len(shuffled) * float(config.eval_fraction))), int(min_eval))
            target = min(target, max(0, len(shuffled) - int(min_fit)))
            eval_cases.update(shuffled[:target])
        shuffled_mixed = list(mixed)
        rng.shuffle(shuffled_mixed)
        target_eval_rows = int(math.ceil(len(indices) * float(config.eval_fraction)))
        for case in shuffled_mixed:
            if sum(len(grouped[item]) for item in eval_cases) >= target_eval_rows:
                break
            eval_cases.add(case)
        if not eval_cases and cases:
            eval_cases.add(str(rng.choice(cases)))
        fit_idx = tuple(idx for case in cases if case not in eval_cases for idx in grouped[case])
        eval_idx = tuple(idx for case in cases if case in eval_cases for idx in grouped[case])
        score = _validity_score(config, rows, fit_idx, eval_idx)
        if score > best_score:
            best = (fit_idx, eval_idx)
            best_score = score
        if _split_status(config, rows, fit_idx, eval_idx, model_type=LOGISTIC_MODEL_TYPE)[0] == VALID_STATUS:
            return fit_idx, eval_idx
        rng = np.random.default_rng(_stable_seed(seed, attempt, "retry"))
    return best if best is not None else ((), ())


def _tumor_class_balanced_indices(
    rows: Sequence[_ManifestRow],
    fit_idx: Sequence[int],
    eval_idx: Sequence[int],
    *,
    seed: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return (
        _downsample_tumor_class_cells(rows, fit_idx, seed=_stable_seed(seed, "fit")),
        _downsample_tumor_class_cells(rows, eval_idx, seed=_stable_seed(seed, "eval")),
    )


def _downsample_tumor_class_cells(rows: Sequence[_ManifestRow], indices: Sequence[int], *, seed: int) -> tuple[int, ...]:
    import numpy as np  # type: ignore

    cells: dict[tuple[str, int], list[int]] = {}
    for idx in indices:
        tumor = _tumor_value(rows[idx])
        if tumor is None:
            continue
        cells.setdefault((tumor, int(rows[idx].label)), []).append(int(idx))
    tumors = sorted({tumor for tumor, _label in cells})
    required = [(tumor, label) for tumor in tumors for label in (0, 1)]
    if not required or any(key not in cells for key in required):
        return ()
    cell_min = min(len(cells[key]) for key in required)
    if cell_min <= 0:
        return ()
    rng = np.random.default_rng(int(seed))
    selected: list[int] = []
    for key in required:
        values = list(cells[key])
        rng.shuffle(values)
        selected.extend(values[:cell_min])
    return tuple(sorted(selected))


def _evaluate_signal_split(
    *,
    config: MidogPPSignalControlsConfig,
    rows: Sequence[_ManifestRow],
    embeddings: Any,
    spec: _SplitSpec,
    model_type: str,
    model_seed: int | None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    import numpy as np  # type: ignore

    counts = _split_counts(rows, spec.fit_idx, spec.eval_idx)
    status, error = _split_status(config, rows, spec.fit_idx, spec.eval_idx, model_type=model_type)
    base = {
        "aggregation_level": "seed",
        "control_name": spec.control_name,
        "domain_name": spec.domain_name,
        "linked_real_control": spec.linked_real_control,
        "split_seed": int(spec.split_seed),
        "model_type": model_type,
        "model_seed": "" if model_seed is None else int(model_seed),
        "eligible_seed_count": "",
        "valid_seed_count": "",
        "decision_status": "",
        **counts,
        "bacc": math.nan,
        "bacc_std": "",
        "bacc_min": "",
        "macro_f1": math.nan,
        "precision_pos": math.nan,
        "recall_pos": math.nan,
        "f1_pos": math.nan,
        "support_pos": counts["n_eval_pos"],
        "support_neg": counts["n_eval_neg"],
        "ci_low": math.nan,
        "ci_high": math.nan,
        "ci_method": "",
        "above_chance": "",
        "strong": "",
        "near_chance": "",
        "converged": "false",
        "status": status,
        "error_message": error,
        "bacc_delta_vs_real": "",
    }
    if status != VALID_STATUS:
        return base, []

    x_fit = np.asarray(embeddings, dtype=float)[list(spec.fit_idx)]
    y_fit = np.asarray([rows[idx].label for idx in spec.fit_idx], dtype=int)
    x_eval = np.asarray(embeddings, dtype=float)[list(spec.eval_idx)]
    y_eval = [int(rows[idx].label) for idx in spec.eval_idx]
    rng = np.random.default_rng(_stable_seed(config.bootstrap_seed, spec.control_name, spec.split_seed, model_type, model_seed or ""))
    if spec.control_name == LABEL_PERMUTATION_CONTROL:
        y_fit = np.asarray(rng.permutation(y_fit), dtype=int)
    elif spec.control_name == FEATURE_ROW_SHUFFLE_CONTROL:
        x_fit = np.asarray(x_fit[rng.permutation(len(x_fit))], dtype=float)
    try:
        probabilities, predictions, converged = _fit_predict(
            x_fit=x_fit,
            y_fit=y_fit,
            x_eval=x_eval,
            model_type=model_type,
            seed=int(model_seed or spec.split_seed),
        )
    except Exception as exc:
        failed = dict(base)
        failed.update({"status": "model_failed_convergence", "error_message": str(exc), "converged": "false"})
        return failed, []
    if not converged:
        failed = dict(base)
        failed.update({"status": "model_failed_convergence", "error_message": "sklearn convergence warning", "converged": "false"})
        return failed, []

    metric_values = _classification_metrics(y_eval, predictions)
    case_ids = [rows[idx].case_id for idx in spec.eval_idx]
    ci_low, ci_high, ci_method = _case_cluster_bacc_ci(
        y_eval,
        predictions,
        case_ids,
        reps=config.bootstrap_reps,
        seed=_stable_seed(config.bootstrap_seed, spec.control_name, spec.domain_name, spec.split_seed, model_type, model_seed or ""),
    )
    out = dict(base)
    out.update(
        {
            **metric_values,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "ci_method": ci_method,
            "converged": "true",
            "status": VALID_STATUS,
            "error_message": "",
        }
    )
    prediction_rows = [
        {
            "control_name": spec.control_name,
            "domain_name": spec.domain_name,
            "split_seed": int(spec.split_seed),
            "model_type": model_type,
            "model_seed": "" if model_seed is None else int(model_seed),
            "sample_id": rows[idx].sample_id,
            "case_id": rows[idx].case_id,
            "feature_row_index": int(idx),
            "tumor_type": _tumor_value(rows[idx]) or "",
            "scanner_model": _axis_metadata(rows[idx], "scanner_model"),
            "lab_or_origin": _axis_metadata(rows[idx], "lab_or_origin"),
            "species": _axis_metadata(rows[idx], "species"),
            "y_true": y_eval[pos],
            "y_pred": int(predictions[pos]),
            "prob_pos": float(probabilities[pos]),
        }
        for pos, idx in enumerate(spec.eval_idx)
    ]
    return out, prediction_rows


def _fit_predict(
    *,
    x_fit: Any,
    y_fit: Any,
    x_eval: Any,
    model_type: str,
    seed: int,
) -> tuple[list[float], list[int], bool]:
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.neural_network import MLPClassifier  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore

    scaler = StandardScaler()
    x_fit_scaled = scaler.fit_transform(x_fit)
    x_eval_scaled = scaler.transform(x_eval)
    if model_type == LOGISTIC_MODEL_TYPE:
        clf = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=5000,
            random_state=int(seed),
        )
    elif model_type == MLP_MODEL_TYPE:
        clf = MLPClassifier(
            hidden_layer_sizes=(128,),
            alpha=1.0e-4,
            random_state=int(seed),
            max_iter=2000,
            early_stopping=False,
        )
    else:
        raise ProtocolError(f"Unknown model_type={model_type!r}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(x_fit_scaled, y_fit)
    converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
    classes = tuple(int(value) for value in clf.classes_.tolist())
    if classes != (0, 1):
        raise ProtocolError(f"class order must be (0, 1), got {classes}")
    proba = clf.predict_proba(x_eval_scaled)[:, 1]
    pred = [1 if float(value) >= 0.5 else 0 for value in proba.tolist()]
    return [float(value) for value in proba.tolist()], pred, converged


def _split_status(
    config: MidogPPSignalControlsConfig,
    rows: Sequence[_ManifestRow],
    fit_idx: Sequence[int],
    eval_idx: Sequence[int],
    *,
    model_type: str,
) -> tuple[str, str]:
    for audit in _identity_audit_rows(rows, _SplitSpec("_status", 0, tuple(fit_idx), tuple(eval_idx))):
        if int(audit["overlap_count"]) > 0:
            field = str(audit["identity_field"])
            return f"protocol_failed_{field}_overlap", str(audit["overlap_preview"])
    counts = _split_counts(rows, fit_idx, eval_idx)
    for key, status, minimum in (
        ("n_fit", "invalid_too_few_fit", config.min_fit),
        ("n_eval", "invalid_too_few_eval", config.min_eval),
        ("n_fit_pos", "invalid_too_few_fit_pos", config.min_fit_pos),
        ("n_fit_neg", "invalid_too_few_fit_neg", config.min_fit_neg),
        ("n_eval_pos", "invalid_too_few_eval_pos", config.min_eval_pos),
        ("n_eval_neg", "invalid_too_few_eval_neg", config.min_eval_neg),
        ("n_fit_pos_cases", "invalid_too_few_fit_pos_cases", config.min_fit_pos_cases),
        ("n_fit_neg_cases", "invalid_too_few_fit_neg_cases", config.min_fit_neg_cases),
        ("n_eval_pos_cases", "invalid_too_few_eval_pos_cases", config.min_eval_pos_cases),
        ("n_eval_neg_cases", "invalid_too_few_eval_neg_cases", config.min_eval_neg_cases),
    ):
        if int(counts[key]) < int(minimum):
            return status, f"{key}={counts[key]} minimum={minimum}"
    if model_type == MLP_MODEL_TYPE:
        if (
            int(counts["n_fit"]) < int(config.mlp_min_fit)
            or int(counts["n_fit_pos"]) < int(config.mlp_min_fit_pos)
            or int(counts["n_fit_neg"]) < int(config.mlp_min_fit_neg)
        ):
            return (
                "skipped_insufficient_support_for_mlp",
                f"n_fit={counts['n_fit']} n_fit_pos={counts['n_fit_pos']} n_fit_neg={counts['n_fit_neg']}",
            )
    return VALID_STATUS, ""


def _split_counts(rows: Sequence[_ManifestRow], fit_idx: Sequence[int], eval_idx: Sequence[int]) -> dict[str, int]:
    fit_labels = [rows[idx].label for idx in fit_idx]
    eval_labels = [rows[idx].label for idx in eval_idx]
    fit_pos_cases = {rows[idx].case_id for idx in fit_idx if rows[idx].label == 1}
    fit_neg_cases = {rows[idx].case_id for idx in fit_idx if rows[idx].label == 0}
    eval_pos_cases = {rows[idx].case_id for idx in eval_idx if rows[idx].label == 1}
    eval_neg_cases = {rows[idx].case_id for idx in eval_idx if rows[idx].label == 0}
    return {
        "n_fit": len(fit_idx),
        "n_eval": len(eval_idx),
        "n_fit_pos": sum(1 for value in fit_labels if value == 1),
        "n_fit_neg": sum(1 for value in fit_labels if value == 0),
        "n_eval_pos": sum(1 for value in eval_labels if value == 1),
        "n_eval_neg": sum(1 for value in eval_labels if value == 0),
        "n_fit_cases": len({rows[idx].case_id for idx in fit_idx}),
        "n_eval_cases": len({rows[idx].case_id for idx in eval_idx}),
        "n_fit_pos_cases": len(fit_pos_cases),
        "n_fit_neg_cases": len(fit_neg_cases),
        "n_eval_pos_cases": len(eval_pos_cases),
        "n_eval_neg_cases": len(eval_neg_cases),
    }


def _validity_score(
    config: MidogPPSignalControlsConfig,
    rows: Sequence[_ManifestRow],
    fit_idx: Sequence[int],
    eval_idx: Sequence[int],
) -> int:
    counts = _split_counts(rows, fit_idx, eval_idx)
    checks = (
        counts["n_fit"] >= config.min_fit,
        counts["n_eval"] >= config.min_eval,
        counts["n_fit_pos"] >= config.min_fit_pos,
        counts["n_fit_neg"] >= config.min_fit_neg,
        counts["n_eval_pos"] >= config.min_eval_pos,
        counts["n_eval_neg"] >= config.min_eval_neg,
        counts["n_fit_pos_cases"] >= config.min_fit_pos_cases,
        counts["n_fit_neg_cases"] >= config.min_fit_neg_cases,
        counts["n_eval_pos_cases"] >= config.min_eval_pos_cases,
        counts["n_eval_neg_cases"] >= config.min_eval_neg_cases,
    )
    return sum(1 for value in checks if value)


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
        low, high = _row_level_bacc_ci(y_true, y_pred, reps=reps, seed=seed)
        return low, high, "row_level_fallback_missing_case_id"
    by_case: dict[str, list[int]] = {}
    for idx, case_id in enumerate(case_ids):
        by_case.setdefault(str(case_id), []).append(idx)
    cases = sorted(by_case)
    if len(cases) < 2 or int(reps) <= 0:
        return math.nan, math.nan, "case_cluster"
    rng = np.random.default_rng(int(seed))
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
    rng = np.random.default_rng(int(seed))
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


def _summary_tables(seed_rows: Sequence[Mapping[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    control_seed = [
        dict(row)
        for row in seed_rows
        if str(row.get("control_name")) in {POOLED_CONTROL, BALANCED_CONTROL}
    ]
    domain_seed = [dict(row) for row in seed_rows if str(row.get("control_name")) == WITHIN_TUMOR_CONTROL]
    negative_seed = [dict(row) for row in seed_rows if str(row.get("control_name")) in NEGATIVE_CONTROLS]
    control_rows = control_seed + _aggregate_rows(control_seed)
    domain_rows = domain_seed + _aggregate_rows(domain_seed)
    negative_rows = negative_seed + _aggregate_rows(negative_seed)
    return control_rows, domain_rows, negative_rows


def _aggregate_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        key = (
            str(row.get("control_name", "")),
            str(row.get("domain_name", "")),
            str(row.get("linked_real_control", "")),
            str(row.get("model_type", "")),
        )
        grouped.setdefault(key, []).append(row)
    out = []
    for (control, domain, linked, model_type), group in sorted(grouped.items()):
        valid = [row for row in group if str(row.get("status")) == VALID_STATUS]
        values = [row.get("bacc") for row in valid]
        ci_low = nanmin([row.get("ci_low") for row in valid])
        ci_high = _nanmax([row.get("ci_high") for row in valid])
        mean_bacc = nanmean(values)
        mean_f1 = nanmean([row.get("f1_pos") for row in valid])
        mean_recall = nanmean([row.get("recall_pos") for row in valid])
        above = bool(_finite(mean_bacc) and _finite(ci_low) and mean_bacc > 0.60 and ci_low > 0.50)
        strong = bool(_finite(mean_bacc) and _finite(mean_f1) and mean_bacc > 0.70 and mean_f1 > 0.50)
        near = bool((_finite(ci_low) and _finite(ci_high) and ci_low <= 0.50 <= ci_high) or (_finite(mean_recall) and mean_recall <= 0.0))
        status = _summary_decision_status(valid, above=above, strong=strong, near=near)
        base = dict(group[0])
        base.update(
            {
                "aggregation_level": "summary",
                "split_seed": "",
                "model_seed": "",
                "eligible_seed_count": len(group),
                "valid_seed_count": len(valid),
                "decision_status": status,
                "n_fit": "",
                "n_eval": "",
                "n_fit_pos": "",
                "n_fit_neg": "",
                "n_eval_pos": "",
                "n_eval_neg": "",
                "n_fit_cases": "",
                "n_eval_cases": "",
                "n_fit_pos_cases": "",
                "n_fit_neg_cases": "",
                "n_eval_pos_cases": "",
                "n_eval_neg_cases": "",
                "bacc": mean_bacc,
                "bacc_std": nanstd(values),
                "bacc_min": nanmin(values),
                "macro_f1": nanmean([row.get("macro_f1") for row in valid]),
                "precision_pos": nanmean([row.get("precision_pos") for row in valid]),
                "recall_pos": mean_recall,
                "f1_pos": mean_f1,
                "support_pos": "",
                "support_neg": "",
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ci_method": "case_cluster_conservative_seed_aggregate",
                "above_chance": bool_text(above),
                "strong": bool_text(strong),
                "near_chance": bool_text(near),
                "converged": bool_text(bool(valid) and len(valid) == len(group)),
                "status": VALID_STATUS if valid else "insufficient_valid_seeds",
                "error_message": "" if valid else "no valid seeds",
                "linked_real_control": linked,
            }
        )
        out.append(base)
    return out


def _summary_decision_status(valid: Sequence[Mapping[str, object]], *, above: bool, strong: bool, near: bool) -> str:
    if not valid:
        return "INSUFFICIENT_VALID_CONTROLS"
    if strong:
        return "STRONG"
    if above:
        return "ABOVE_CHANCE"
    if near:
        return "NEAR_CHANCE"
    return "WEAK_OR_UNSTABLE"


def _attach_negative_deltas(control_rows: Sequence[dict[str, object]], negative_rows: Sequence[dict[str, object]]) -> None:
    real_summary = {
        (str(row.get("control_name")), str(row.get("model_type"))): row
        for row in control_rows
        if str(row.get("aggregation_level")) == "summary"
    }
    for row in negative_rows:
        if str(row.get("aggregation_level")) != "summary":
            continue
        linked = str(row.get("linked_real_control") or POOLED_CONTROL)
        real = real_summary.get((linked, str(row.get("model_type"))))
        if real is None or not _finite(real.get("bacc")) or not _finite(row.get("bacc")):
            continue
        row["bacc_delta_vs_real"] = float(real["bacc"]) - float(row["bacc"])


def _decision_labels(
    control_rows: Sequence[Mapping[str, object]],
    domain_rows: Sequence[Mapping[str, object]],
    negative_rows: Sequence[Mapping[str, object]],
    prior_lodo: Mapping[str, object],
) -> list[str]:
    negative_summaries = _summary_for(negative_rows, model_type=LOGISTIC_MODEL_TYPE)
    if any(str(row.get("above_chance")) == "true" for row in negative_summaries):
        return ["LEAKAGE_OR_ALIGNMENT_FAILURE_SUSPECT"]
    control_summaries = _summary_for(control_rows, model_type=LOGISTIC_MODEL_TYPE)
    domain_summaries = _summary_for(domain_rows, model_type=LOGISTIC_MODEL_TYPE)
    pooled = _find_summary(control_summaries, POOLED_CONTROL)
    balanced = _find_summary(control_summaries, BALANCED_CONTROL)
    if pooled is None or balanced is None or not domain_summaries:
        return ["INSUFFICIENT_VALID_CONTROLS"]
    pooled_above = str(pooled.get("above_chance")) == "true"
    balanced_above = str(balanced.get("above_chance")) == "true"
    pooled_strong = str(pooled.get("strong")) == "true"
    balanced_strong = str(balanced.get("strong")) == "true"
    valid_domains = [row for row in domain_summaries if str(row.get("status")) == VALID_STATUS]
    domain_above = [row for row in valid_domains if str(row.get("above_chance")) == "true"]
    domain_strong = [row for row in valid_domains if str(row.get("strong")) == "true"]
    most_domains_above = bool(valid_domains) and (len(domain_above) / len(valid_domains)) >= 0.60
    most_domains_strong = bool(valid_domains) and (len(domain_strong) / len(valid_domains)) >= 0.60
    all_negative_near = all(str(row.get("near_chance")) == "true" for row in negative_summaries) if negative_summaries else False
    if pooled_above and balanced_above and most_domains_above and all_negative_near:
        labels = ["CACHE_SIGNAL_PRESENT"]
        if pooled_strong and balanced_strong and most_domains_strong and bool(prior_lodo.get("near_chance_transfer")):
            labels.append("CROSS_DOMAIN_TRANSFER_FAILURE")
        else:
            labels.append("CACHE_SIGNAL_PRESENT_LODO_FAILURE_COMPATIBLE")
        return labels
    if pooled_above and (not balanced_above or not most_domains_above) and all_negative_near:
        return ["POOLED_SIGNAL_DOMAIN_SHORTCUT_SUSPECT"]
    if most_domains_above and not pooled_above and not balanced_above and all_negative_near:
        return ["DOMAIN_LOCAL_SIGNAL_ONLY"]
    real_near = (
        str(pooled.get("near_chance")) == "true"
        and str(balanced.get("near_chance")) == "true"
        and bool(valid_domains)
        and all(str(row.get("near_chance")) == "true" for row in valid_domains)
    )
    if real_near and all_negative_near:
        return ["NO_REAL_FEATURE_SIGNAL_DETECTED"]
    return ["INSUFFICIENT_VALID_CONTROLS"]


def _summary_for(rows: Sequence[Mapping[str, object]], *, model_type: str) -> list[Mapping[str, object]]:
    return [row for row in rows if str(row.get("aggregation_level")) == "summary" and str(row.get("model_type")) == model_type]


def _find_summary(rows: Sequence[Mapping[str, object]], control_name: str) -> Mapping[str, object] | None:
    for row in rows:
        if str(row.get("control_name")) == control_name:
            return row
    return None


def _prior_lodo_status(repo_root: Path, config: MidogPPSignalControlsConfig) -> dict[str, object]:
    if not config.prior_lodo_axis_summary_path:
        return {"provided": False, "near_chance_transfer": False, "path": ""}
    path = _resolve_path(repo_root, config.prior_lodo_axis_summary_path)
    if not path.exists():
        return {"provided": False, "near_chance_transfer": False, "path": str(path)}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    gate_rows = [
        row
        for row in rows
        if row.get("model_type") == LOGISTIC_MODEL_TYPE
        and row.get("global_failure_gate_axis") == "true"
        and row.get("decision_valid") == "true"
    ]
    near = bool(gate_rows) and all(row.get("near_chance") == "true" for row in gate_rows)
    return {"provided": True, "near_chance_transfer": near, "path": str(path), "gate_axis_count": len(gate_rows)}


def _leakage_report(
    identity_rows: Sequence[Mapping[str, object]],
    control_rows: Sequence[Mapping[str, object]],
    domain_rows: Sequence[Mapping[str, object]],
    negative_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    overlap_failures = [row for row in identity_rows if int(row.get("overlap_count", 0)) > 0]
    negative_above = [
        row
        for row in _summary_for(negative_rows, model_type=LOGISTIC_MODEL_TYPE)
        if str(row.get("above_chance")) == "true"
    ]
    status = "PASS"
    if overlap_failures:
        status = "SPLIT_PROTOCOL_FAILURES_REPORTED"
    if negative_above:
        status = "LEAKAGE_OR_ALIGNMENT_FAILURE_SUSPECT"
    failed_metric_rows = [
        row
        for row in list(control_rows) + list(domain_rows) + list(negative_rows)
        if str(row.get("aggregation_level")) == "seed" and str(row.get("status", "")).startswith("protocol_failed_")
    ]
    return {
        "status": status,
        "identity_overlap_failure_count": len(overlap_failures),
        "protocol_failed_metric_row_count": len(failed_metric_rows),
        "negative_controls_above_chance_count": len(negative_above),
        "fit_only_standardization": True,
        "fixed_threshold": 0.5,
        "threshold_tuned_on_eval": False,
        "eval_labels_used_for_scoring_only": True,
        "target_labels_used_for_fitting": False,
    }


def _protocol_manifest(
    *,
    config: MidogPPSignalControlsConfig,
    manifest_path: Path,
    feature_cache_path: Path,
    feature_dim: int,
    train_rows: int,
    cache: FeatureCache,
    prior_lodo: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_virchow2_real_feature_signal_controls_protocol_v1",
        "experiment_name": EXPERIMENT_NAME,
        "manifest_path": str(manifest_path),
        "feature_cache_path": str(feature_cache_path),
        "train_rows": int(train_rows),
        "feature_dim": int(feature_dim),
        "feature_extractor": dict(cache.feature_extractor),
        "positive_label": int(config.positive_label),
        "label_mapping": dict(config.label_mapping or {"0": "negative", "1": "mitotic_positive"}),
        "split_scope": "train_only",
        "cache_building_in_scope": False,
        "threshold_policy": "fixed_0.5_classifier_rule_not_calibrated_probability",
        "primary_aggregation": "mean_std_min_over_seed_level_metrics",
        "bootstrap_ci": "case_cluster_eval_cases_with_replacement",
        "split": {"seeds": list(config.split_seeds), "eval_fraction": float(config.eval_fraction)},
        "validity_thresholds": {
            "min_fit": config.min_fit,
            "min_eval": config.min_eval,
            "min_fit_pos": config.min_fit_pos,
            "min_fit_neg": config.min_fit_neg,
            "min_eval_pos": config.min_eval_pos,
            "min_eval_neg": config.min_eval_neg,
            "min_fit_pos_cases": config.min_fit_pos_cases,
            "min_fit_neg_cases": config.min_fit_neg_cases,
            "min_eval_pos_cases": config.min_eval_pos_cases,
            "min_eval_neg_cases": config.min_eval_neg_cases,
        },
        "logistic_regression": {
            "class_weight": "balanced",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 5000,
            "random_state": "split_seed",
        },
        "mlp_sensitivity_gate": {
            "min_fit": config.mlp_min_fit,
            "min_fit_pos": config.mlp_min_fit_pos,
            "min_fit_neg": config.mlp_min_fit_neg,
            "hidden_layer_sizes": [128],
            "alpha": 1.0e-4,
            "max_iter": 2000,
            "early_stopping": False,
            "seeds": list(config.mlp_seeds),
        },
        "negative_controls": {
            LABEL_PERMUTATION_CONTROL: "permute fit labels only; eval labels remain real",
            FEATURE_ROW_SHUFFLE_CONTROL: "permute fit feature rows relative to fit labels only; eval rows remain real",
        },
        "prior_lodo": dict(prior_lodo),
        "claim_boundary": {
            "allowed": "Real Virchow2 train-cache binary class signal under pooled, tumor-balanced, and within-tumor controls.",
            "forbidden": "No CVAE preservation, metadata-routing utility, or target-domain routing claim.",
        },
    }


def _write_decision_report(
    path: Path,
    control_rows: Sequence[Mapping[str, object]],
    domain_rows: Sequence[Mapping[str, object]],
    negative_rows: Sequence[Mapping[str, object]],
    leakage: Mapping[str, object],
    decision_labels: Sequence[str],
    prior_lodo: Mapping[str, object],
) -> None:
    lines = [
        "# MIDOG++ Virchow2 Real-Feature Signal Controls",
        "",
        f"- Decision labels: `{', '.join(decision_labels) if decision_labels else 'NONE'}`",
        f"- Leakage report status: `{leakage.get('status')}`",
        f"- Prior LODO summary linked: `{bool_text(bool(prior_lodo.get('provided')))}; near_chance_transfer={bool_text(bool(prior_lodo.get('near_chance_transfer')))}`",
        "- Metadata is used only for split construction and stratified reporting.",
        "- Threshold is fixed at 0.5 as a classifier rule, not as a calibrated probability claim.",
        "",
        "## Control Summary",
        "",
        "| Control | Model | Decision | Valid seeds | Mean BACC | CI low | Positive F1 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in _summary_for(control_rows, model_type=LOGISTIC_MODEL_TYPE):
        lines.append(
            f"| {row['control_name']} | {row['model_type']} | {row['decision_status']} | "
            f"{row['valid_seed_count']}/{row['eligible_seed_count']} | {_fmt(row['bacc'])} | {_fmt(row['ci_low'])} | {_fmt(row['f1_pos'])} |"
        )
    lines.extend(["", "## Negative Controls", "", "| Control | Decision | Mean BACC | Delta vs real |", "| --- | --- | ---: | ---: |"])
    for row in _summary_for(negative_rows, model_type=LOGISTIC_MODEL_TYPE):
        lines.append(
            f"| {row['control_name']} | {row['decision_status']} | {_fmt(row['bacc'])} | {_fmt(row.get('bacc_delta_vs_real'))} |"
        )
    domain_summaries = _summary_for(domain_rows, model_type=LOGISTIC_MODEL_TYPE)
    if domain_summaries:
        lines.extend(["", "## Within-Tumor Summary", "", "| Tumor type | Decision | Mean BACC | CI low |", "| --- | --- | ---: | ---: |"])
        for row in domain_summaries:
            lines.append(f"| {row['domain_name']} | {row['decision_status']} | {_fmt(row['bacc'])} | {_fmt(row['ci_low'])} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _identity_audit_rows(rows: Sequence[_ManifestRow], spec: _SplitSpec) -> list[dict[str, object]]:
    out = []
    for field in ("sample_id", "case_id", "image_path", "feature_row_index", "source_object_proxy"):
        fit_values = _identity_set(rows, spec.fit_idx, field)
        eval_values = _identity_set(rows, spec.eval_idx, field)
        overlap = sorted(fit_values.intersection(eval_values))
        out.append(
            {
                "control_name": spec.control_name,
                "domain_name": spec.domain_name,
                "split_seed": int(spec.split_seed),
                "identity_field": field,
                "overlap_count": len(overlap),
                "overlap_preview": "|".join(overlap[:10]),
                "status": "PASS" if not overlap else "FAIL",
            }
        )
    return out


def _identity_set(rows: Sequence[_ManifestRow], indices: Sequence[int], field: str) -> set[str]:
    values = set()
    for idx in indices:
        if field == "sample_id":
            value = rows[idx].sample_id
        elif field == "case_id":
            value = rows[idx].case_id
        elif field == "image_path":
            value = str(rows[idx].metadata.get("image_path", "")).strip()
        elif field == "feature_row_index":
            value = str(idx)
        elif field == "source_object_proxy":
            value = _source_object_proxy(rows[idx])
        else:
            value = ""
        if value:
            values.add(value)
    return values


def _source_object_proxy(row: _ManifestRow) -> str:
    fields = (
        "case_id",
        "annotation_id",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "patch_center_x",
        "patch_center_y",
    )
    parts = [str(row.metadata.get(field, "")).strip() for field in fields]
    compact = [part for part in parts if part]
    return "|".join(compact) if len(compact) > 1 else row.sample_id


def _split_manifest_rows(rows: Sequence[_ManifestRow], spec: _SplitSpec) -> list[dict[str, object]]:
    out = []
    for subset, indices in (("fit", spec.fit_idx), ("eval", spec.eval_idx)):
        for idx in indices:
            row = rows[idx]
            out.append(
                {
                    "control_name": spec.control_name,
                    "domain_name": spec.domain_name,
                    "split_seed": int(spec.split_seed),
                    "subset": subset,
                    "row_index": int(row.row_index),
                    "feature_row_index": int(idx),
                    "sample_id": row.sample_id,
                    "case_id": row.case_id,
                    "image_path": row.metadata.get("image_path", ""),
                    "label": int(row.label),
                    "tumor_type": _tumor_value(row) or "",
                    "scanner_model": _axis_metadata(row, "scanner_model"),
                    "lab_or_origin": _axis_metadata(row, "lab_or_origin"),
                    "species": _axis_metadata(row, "species"),
                }
            )
    return out


def _stratified_metric_rows(
    rows: Sequence[_ManifestRow],
    predictions: Sequence[Mapping[str, object]],
    metric_row: Mapping[str, object],
) -> list[dict[str, object]]:
    del rows
    out = []
    for axis in ("scanner_model", "lab_or_origin", "species"):
        values = sorted({str(row.get(axis, "")).strip() for row in predictions if str(row.get(axis, "")).strip()})
        for value in values:
            group = [row for row in predictions if str(row.get(axis, "")).strip() == value]
            y_true = [int(row["y_true"]) for row in group]
            y_pred = [int(row["y_pred"]) for row in group]
            if len(set(y_true)) < 2:
                status = "invalid_mono_class_eval"
                metrics = {"bacc": math.nan, "macro_f1": math.nan, "precision_pos": math.nan, "recall_pos": math.nan, "f1_pos": math.nan}
                error = "stratum has fewer than two eval classes"
            else:
                status = VALID_STATUS
                metrics = _classification_metrics(y_true, y_pred)
                error = ""
            out.append(
                {
                    "control_name": metric_row["control_name"],
                    "domain_name": metric_row["domain_name"],
                    "split_seed": metric_row["split_seed"],
                    "model_type": metric_row["model_type"],
                    "model_seed": metric_row["model_seed"],
                    "stratify_axis": axis,
                    "stratum_value": value,
                    "n_eval": len(group),
                    "n_eval_pos": sum(1 for item in y_true if item == 1),
                    "n_eval_neg": sum(1 for item in y_true if item == 0),
                    **metrics,
                    "status": status,
                    "error_message": error,
                }
            )
    return out


def _indices_by_case(rows: Sequence[_ManifestRow], indices: Sequence[int]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for idx in indices:
        out.setdefault(rows[idx].case_id, []).append(int(idx))
    return out


def _case_labels(rows: Sequence[_ManifestRow], indices: Sequence[int]) -> set[int]:
    return {int(rows[idx].label) for idx in indices}


def _tumor_values(rows: Sequence[_ManifestRow]) -> tuple[str, ...]:
    return tuple(sorted({value for row in rows if (value := _tumor_value(row)) is not None}))


def _tumor_value(row: _ManifestRow) -> str | None:
    return _axis_metadata_or_none(row, "tumor_type")


def _axis_metadata(row: _ManifestRow, field: str) -> str:
    return _axis_metadata_or_none(row, field) or ""


def _axis_metadata_or_none(row: _ManifestRow, field: str) -> str | None:
    value = str(row.metadata.get(field, "")).strip()
    if value.lower() in UNKNOWN_VALUES:
        return None
    return _clean_axis_value(value)


def _nanmax(values: Sequence[object]) -> float:
    vals = []
    for value in values:
        try:
            if value in ("", None):
                continue
            fval = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fval):
            vals.append(fval)
    return max(vals) if vals else math.nan
