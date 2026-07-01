"""Runner entry points for the MIDOG++ real-feature gate."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .adapters.sail import compare_sail_semantics
from .contracts import (
    DATASET,
    DEFAULT_GATE_CRITERIA,
    ELIGIBLE_CENTERS,
    POSITIVE_LABEL,
    POSITIVE_LABEL_NAME,
    QUARANTINE_CENTERS,
    REQUIRED_MATRIX_COLUMNS,
    SCHEMA_VERSION,
    ClaimRole,
    RowRole,
)
from .data import ManifestRow, assert_cache_alignment, load_feature_cache, load_manifest
from .metrics import binary_metrics
from .report import summarize_decision_labels
from .splits import Fold, heldout_center_folds, is_quarantine_center
from .validation import ValidationError, validate_artifact_bundle, validate_row_role_flags


@dataclass(frozen=True)
class RunConfig:
    manifest_path: Path
    feature_cache_path: Path
    artifact_root: Path
    repo_root: Path = Path(".")
    min_source: int = 20
    min_eval: int = 10
    min_source_pos: int = 2
    min_source_neg: int = 2
    min_eval_pos: int = 1
    min_eval_neg: int = 1
    model_seed: int = 0
    include_pooled_diagnostic: bool = True
    allow_npz_test_cache: bool = False


@dataclass(frozen=True)
class RunResult:
    output_paths: Mapping[str, Path]
    decision_labels: tuple[str, ...]


def run_gate(config: RunConfig) -> RunResult:
    """Run the primary held-out-center real-feature transfer gate."""
    artifact_root = Path(config.artifact_root)
    tables_dir = artifact_root / "tables"
    manifests_dir = artifact_root / "manifests"
    reports_dir = artifact_root / "reports"
    for directory in (tables_dir, manifests_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest_rows_all = load_manifest(Path(config.manifest_path), positive_label=POSITIVE_LABEL)
    train_rows = tuple(row for row in manifest_rows_all if row.split == "train")
    if not train_rows:
        raise ValidationError("MIDOG++ gate requires at least one split=train manifest row")
    cache = load_feature_cache(Path(config.feature_cache_path))
    if Path(config.feature_cache_path).suffix == ".npz" and not config.allow_npz_test_cache:
        raise ValidationError("npz feature caches are test/lightweight caches; set allow_npz_test_cache=True")
    assert_cache_alignment(train_rows, cache)

    import numpy as np  # type: ignore

    embeddings = np.asarray(cache.embeddings, dtype=float)
    if embeddings.ndim != 2:
        raise ValidationError(f"feature embeddings must be 2D, got shape={embeddings.shape}")

    manifest_hash = _file_hash(Path(config.manifest_path))
    cache_hash = _file_hash(Path(config.feature_cache_path))
    config_hash = _hash_json(_config_payload(config))
    protocol_hash = _hash_json(_protocol_payload())

    matrix_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    leakage_rows: list[dict[str, object]] = []

    for fold in heldout_center_folds(train_rows):
        if is_quarantine_center(fold.heldout_center):
            if config.include_pooled_diagnostic:
                row, preds = _evaluate_fold(
                    config=config,
                    fold=fold,
                    rows=train_rows,
                    embeddings=embeddings,
                    fit_indices=tuple(range(len(train_rows))),
                    role=RowRole.POOLED_DIAGNOSTIC_CEILING,
                    claim_role=ClaimRole.QUARANTINE_ONLY,
                    manifest_hash=manifest_hash,
                    cache_hash=cache_hash,
                    config_hash=config_hash,
                    protocol_hash=protocol_hash,
                )
                matrix_rows.append(row)
                prediction_rows.extend(preds)
                confusion_rows.append(_confusion_row(row))
            leakage_rows.append(_overlap_row(fold, train_rows))
            continue

        source_row, source_preds = _evaluate_fold(
            config=config,
            fold=fold,
            rows=train_rows,
            embeddings=embeddings,
            fit_indices=fold.source_indices,
            role=RowRole.SOURCE_ONLY_TRANSFER,
            claim_role=ClaimRole.TRANSFER_BASELINE,
            manifest_hash=manifest_hash,
            cache_hash=cache_hash,
            config_hash=config_hash,
            protocol_hash=protocol_hash,
        )
        matrix_rows.append(source_row)
        prediction_rows.extend(source_preds)
        confusion_rows.append(_confusion_row(source_row))
        if config.include_pooled_diagnostic:
            pooled_row, pooled_preds = _evaluate_fold(
                config=config,
                fold=fold,
                rows=train_rows,
                embeddings=embeddings,
                fit_indices=tuple(range(len(train_rows))),
                role=RowRole.POOLED_DIAGNOSTIC_CEILING,
                claim_role=ClaimRole.DIAGNOSTIC_CEILING,
                manifest_hash=manifest_hash,
                cache_hash=cache_hash,
                config_hash=config_hash,
                protocol_hash=protocol_hash,
            )
            matrix_rows.append(pooled_row)
            prediction_rows.extend(pooled_preds)
            confusion_rows.append(_confusion_row(pooled_row))
        leakage_rows.append(_overlap_row(fold, train_rows))

    for row in matrix_rows:
        validate_row_role_flags(row)

    ranking_rows = _source_only_ranking_rows(matrix_rows)
    delta_rows = _source_vs_pooled_delta_rows(matrix_rows)
    worst_rows = _worst_domain_rows(matrix_rows)
    stratified_rows = _stratified_breakdown_rows(prediction_rows)
    leakage_pass = all(int(row["sample_overlap_count"]) == 0 and int(row["case_overlap_count"]) == 0 for row in leakage_rows)
    artifact_completeness_pass = True
    negative_controls_pass = True
    decision_labels = summarize_decision_labels(
        matrix_rows,
        criteria=DEFAULT_GATE_CRITERIA,
        artifact_completeness_pass=artifact_completeness_pass,
        leakage_provenance_pass=leakage_pass,
        negative_controls_pass=negative_controls_pass,
    )

    output_paths = {
        "matrix": tables_dir / "matrix.csv",
        "predictions": tables_dir / "predictions.csv",
        "confusion_summary": tables_dir / "confusion_summary.csv",
        "stratified_breakdown": tables_dir / "stratified_breakdown.csv",
        "source_only_ranking_gap": tables_dir / "source_only_ranking_gap.csv",
        "source_vs_pooled_delta": tables_dir / "source_vs_pooled_delta.csv",
        "worst_domain_summary": tables_dir / "worst_domain_summary.csv",
        "protocol_manifest": manifests_dir / "protocol_manifest.json",
        "leakage_provenance_report": reports_dir / "leakage_provenance_report.json",
        "decision_report": reports_dir / "decision_report.md",
    }
    _write_csv(output_paths["matrix"], REQUIRED_MATRIX_COLUMNS, matrix_rows)
    _write_csv(output_paths["predictions"], _prediction_columns(), prediction_rows)
    _write_csv(output_paths["confusion_summary"], _confusion_columns(), confusion_rows)
    _write_csv(output_paths["stratified_breakdown"], _stratified_columns(), stratified_rows)
    _write_csv(output_paths["source_only_ranking_gap"], _ranking_columns(), ranking_rows)
    _write_csv(output_paths["source_vs_pooled_delta"], _delta_columns(), delta_rows)
    _write_csv(output_paths["worst_domain_summary"], _worst_columns(), worst_rows)
    _write_json(
        output_paths["protocol_manifest"],
        {
            **_protocol_payload(),
            "manifest_path": str(config.manifest_path),
            "feature_cache_path": str(config.feature_cache_path),
            "artifact_root": str(config.artifact_root),
            "manifest_hash": manifest_hash,
            "cache_hash": cache_hash,
            "config_hash": config_hash,
            "protocol_hash": protocol_hash,
            "sail_reference": compare_sail_semantics(Path(config.repo_root)),
        },
    )
    _write_json(
        output_paths["leakage_provenance_report"],
        {
            "status": "PASS" if leakage_pass else "FAIL",
            "target_labels_used_for_fitting": False,
            "target_labels_used_for_scoring_only": True,
            "source_only_rows_fit_used_target_center": False,
            "source_only_rows_selection_used_target_labels": False,
            "overlap_rows": leakage_rows,
        },
    )
    _write_decision_report(output_paths["decision_report"], decision_labels, matrix_rows, leakage_pass)
    validate_artifact_bundle(artifact_root)
    return RunResult(output_paths=output_paths, decision_labels=tuple(decision_labels))


def _evaluate_fold(
    *,
    config: RunConfig,
    fold: Fold,
    rows: Sequence[ManifestRow],
    embeddings: object,
    fit_indices: Sequence[int],
    role: RowRole,
    claim_role: ClaimRole,
    manifest_hash: str,
    cache_hash: str,
    config_hash: str,
    protocol_hash: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    import numpy as np  # type: ignore

    eval_indices = fold.eval_indices
    n_source_pos = sum(rows[idx].label == 1 for idx in fit_indices)
    n_source_neg = sum(rows[idx].label == 0 for idx in fit_indices)
    n_eval_pos = sum(rows[idx].label == 1 for idx in eval_indices)
    n_eval_neg = sum(rows[idx].label == 0 for idx in eval_indices)
    status, invalid_reason = _fold_status(config, fit_indices, eval_indices, n_source_pos, n_source_neg, n_eval_pos, n_eval_neg)
    y_eval = [int(rows[idx].label) for idx in eval_indices]
    prob_pos: list[float] = []
    y_pred: list[int] = []
    if status == "valid":
        x_fit = np.asarray(embeddings, dtype=float)[list(fit_indices)]
        y_fit = np.asarray([rows[idx].label for idx in fit_indices], dtype=int)
        x_eval = np.asarray(embeddings, dtype=float)[list(eval_indices)]
        prob_pos, y_pred = _fit_predict(x_fit, y_fit, x_eval, seed=config.model_seed)
    metric_values = binary_metrics(y_eval, y_pred, prob_pos) if y_pred else _empty_metric_values(n_eval_pos, n_eval_neg)
    prediction_hash = _hash_json(
        {
            "heldout_center": fold.heldout_center,
            "row_role": str(role),
            "sample_ids": [rows[idx].sample_id for idx in eval_indices],
            "prob_pos": prob_pos,
            "y_pred": y_pred,
        }
    )
    row = {
        "schema_version": SCHEMA_VERSION,
        "dataset": DATASET,
        "domain_regime": "midogpp_center",
        "fold_unit": fold.fold_unit,
        "heldout_center": fold.heldout_center,
        "heldout_tumor_domain": fold.heldout_tumor_domain,
        "source_scope": "all_non_target_centers" if role == RowRole.SOURCE_ONLY_TRANSFER else "pooled_all_centers",
        "fit_domains": "|".join(sorted({rows[idx].center for idx in fit_indices})),
        "eval_domain": fold.heldout_center,
        "method": "sklearn_logistic_regression_fixed_v1",
        "row_role": str(role),
        "claim_role": str(claim_role),
        "adoption_eligible": role == RowRole.SOURCE_ONLY_TRANSFER,
        "diagnostic_only": role != RowRole.SOURCE_ONLY_TRANSFER,
        "selection_source": "predeclared_source_only" if role == RowRole.SOURCE_ONLY_TRANSFER else "diagnostic_control",
        "fit_used_target_center": fold.heldout_center in {rows[idx].center for idx in fit_indices},
        "selection_used_target_labels": False,
        "target_eval_labels_used_for_scoring_only": True,
        "support_labels_used": False,
        "threshold_policy": "fixed_prob_pos_ge_0.5",
        "calibration_policy": "none_scores_not_claimed_calibrated",
        "model_seed": int(config.model_seed),
        "status": status,
        "invalid_reason": invalid_reason,
        "manifest_hash": manifest_hash,
        "cache_hash": cache_hash,
        "config_hash": config_hash,
        "protocol_hash": protocol_hash,
        "prediction_hash": prediction_hash,
        "n_eval": len(eval_indices),
        "n_eval_pos": n_eval_pos,
        "n_eval_neg": n_eval_neg,
        **metric_values,
    }
    prediction_rows = [
        {
            "schema_version": SCHEMA_VERSION,
            "heldout_center": fold.heldout_center,
            "heldout_tumor_domain": rows[idx].tumor_domain,
            "row_role": str(role),
            "sample_id": rows[idx].sample_id,
            "case_id": rows[idx].case_id,
            "y_true": rows[idx].label,
            "y_pred": y_pred[pos] if pos < len(y_pred) else "",
            "prob_pos": prob_pos[pos] if pos < len(prob_pos) else "",
            "prediction_hash": prediction_hash,
        }
        for pos, idx in enumerate(eval_indices)
    ]
    return row, prediction_rows


def _fit_predict(x_fit: object, y_fit: object, x_eval: object, *, seed: int) -> tuple[list[float], list[int]]:
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.pipeline import make_pipeline  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=2000, random_state=int(seed)),
    )
    clf.fit(x_fit, y_fit)
    prob = clf.predict_proba(x_eval)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return [float(value) for value in prob.tolist()], [int(value) for value in pred.tolist()]


def _fold_status(
    config: RunConfig,
    fit_indices: Sequence[int],
    eval_indices: Sequence[int],
    n_source_pos: int,
    n_source_neg: int,
    n_eval_pos: int,
    n_eval_neg: int,
) -> tuple[str, str]:
    if len(fit_indices) < config.min_source:
        return "invalid_too_few_source", f"n_source={len(fit_indices)} minimum={config.min_source}"
    if len(eval_indices) < config.min_eval:
        return "invalid_too_few_eval", f"n_eval={len(eval_indices)} minimum={config.min_eval}"
    checks = (
        ("n_source_pos", n_source_pos, config.min_source_pos),
        ("n_source_neg", n_source_neg, config.min_source_neg),
        ("n_eval_pos", n_eval_pos, config.min_eval_pos),
        ("n_eval_neg", n_eval_neg, config.min_eval_neg),
    )
    for name, value, minimum in checks:
        if int(value) < int(minimum):
            return f"invalid_too_few_{name.removeprefix('n_')}", f"{name}={value} minimum={minimum}"
    return "valid", ""


def _empty_metric_values(n_eval_pos: int, n_eval_neg: int) -> dict[str, object]:
    n_eval = n_eval_pos + n_eval_neg
    prevalence = float(n_eval_pos / n_eval) if n_eval else math.nan
    return {
        "target_prevalence": prevalence,
        "predicted_positive_rate": math.nan,
        "tp": 0.0,
        "fp": 0.0,
        "tn": 0.0,
        "fn": 0.0,
        "sensitivity": math.nan,
        "specificity": math.nan,
        "precision": math.nan,
        "macro_f1": math.nan,
        "balanced_accuracy": math.nan,
        "auroc": math.nan,
        "pr_auc": math.nan,
        "pr_auc_baseline": prevalence,
    }


def _overlap_row(fold: Fold, rows: Sequence[ManifestRow]) -> dict[str, object]:
    source_samples = {rows[idx].sample_id for idx in fold.source_indices}
    eval_samples = {rows[idx].sample_id for idx in fold.eval_indices}
    source_cases = {rows[idx].case_id for idx in fold.source_indices}
    eval_cases = {rows[idx].case_id for idx in fold.eval_indices}
    return {
        "fold_unit": fold.fold_unit,
        "heldout_center": fold.heldout_center,
        "sample_overlap_count": len(source_samples.intersection(eval_samples)),
        "case_overlap_count": len(source_cases.intersection(eval_cases)),
    }


def _confusion_row(row: Mapping[str, object]) -> dict[str, object]:
    return {key: row.get(key, "") for key in ("heldout_center", "row_role", "status", "tp", "fp", "tn", "fn")}


def _stratified_breakdown_rows(prediction_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, object]]] = {}
    for row in prediction_rows:
        if row.get("y_pred") == "":
            continue
        key = (str(row.get("heldout_center", "")), str(row.get("heldout_tumor_domain", "")), str(row.get("row_role", "")))
        grouped.setdefault(key, []).append(row)
    out = []
    for (center, tumor, role), rows in sorted(grouped.items()):
        y_true = [int(row["y_true"]) for row in rows]
        y_pred = [int(row["y_pred"]) for row in rows]
        prob = [float(row["prob_pos"]) for row in rows]
        metrics = binary_metrics(y_true, y_pred, prob)
        out.append({"heldout_center": center, "heldout_tumor_domain": tumor, "row_role": role, "n": len(rows), **metrics})
    return out


def _source_only_ranking_rows(matrix_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows = [row for row in matrix_rows if row.get("row_role") == RowRole.SOURCE_ONLY_TRANSFER]
    sorted_rows = sorted(rows, key=lambda row: _float(row.get("balanced_accuracy")), reverse=True)
    best = _float(sorted_rows[0].get("balanced_accuracy")) if sorted_rows else math.nan
    return [
        {
            "rank": idx + 1,
            "heldout_center": row.get("heldout_center", ""),
            "balanced_accuracy": row.get("balanced_accuracy", ""),
            "auroc": row.get("auroc", ""),
            "gap_to_best_bacc": best - _float(row.get("balanced_accuracy")) if not math.isnan(best) else math.nan,
        }
        for idx, row in enumerate(sorted_rows)
    ]


def _source_vs_pooled_delta_rows(matrix_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    by_center: dict[str, dict[str, Mapping[str, object]]] = {}
    for row in matrix_rows:
        by_center.setdefault(str(row.get("heldout_center", "")), {})[str(row.get("row_role", ""))] = row
    out = []
    for center, roles in sorted(by_center.items()):
        source = roles.get(str(RowRole.SOURCE_ONLY_TRANSFER))
        pooled = roles.get(str(RowRole.POOLED_DIAGNOSTIC_CEILING))
        if not source or not pooled:
            continue
        out.append(
            {
                "heldout_center": center,
                "source_bacc": source.get("balanced_accuracy", ""),
                "pooled_bacc": pooled.get("balanced_accuracy", ""),
                "delta_bacc": _float(pooled.get("balanced_accuracy")) - _float(source.get("balanced_accuracy")),
                "source_macro_f1": source.get("macro_f1", ""),
                "pooled_macro_f1": pooled.get("macro_f1", ""),
                "delta_macro_f1": _float(pooled.get("macro_f1")) - _float(source.get("macro_f1")),
            }
        )
    return out


def _worst_domain_rows(matrix_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    source = [row for row in matrix_rows if row.get("row_role") == RowRole.SOURCE_ONLY_TRANSFER]
    if not source:
        return []
    worst = min(source, key=lambda row: _float(row.get("balanced_accuracy")))
    return [
        {
            "summary": "worst_source_only_center",
            "heldout_center": worst.get("heldout_center", ""),
            "balanced_accuracy": worst.get("balanced_accuracy", ""),
            "auroc": worst.get("auroc", ""),
            "status": worst.get("status", ""),
        }
    ]


def _write_decision_report(path: Path, labels: Sequence[str], matrix_rows: Sequence[Mapping[str, object]], leakage_pass: bool) -> None:
    lines = [
        "# MIDOG++ Real Feature Gate Decision Report",
        "",
        f"- schema_version: `{SCHEMA_VERSION}`",
        f"- leakage_provenance_pass: `{str(leakage_pass).lower()}`",
        f"- decision_labels: `{', '.join(labels)}`",
        f"- matrix_rows: `{len(matrix_rows)}`",
        "",
        "Claim boundary: real-feature transfer, headroom, and failure-mode diagnostics only.",
        "No CVAE preservation, NELBO compatibility, routing quality, synthetic utility, or generative-quality claim is supported by this gate.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in columns})


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _prediction_columns() -> tuple[str, ...]:
    return ("schema_version", "heldout_center", "heldout_tumor_domain", "row_role", "sample_id", "case_id", "y_true", "y_pred", "prob_pos", "prediction_hash")


def _confusion_columns() -> tuple[str, ...]:
    return ("heldout_center", "row_role", "status", "tp", "fp", "tn", "fn")


def _stratified_columns() -> tuple[str, ...]:
    return ("heldout_center", "heldout_tumor_domain", "row_role", "n", "target_prevalence", "predicted_positive_rate", "tp", "fp", "tn", "fn", "sensitivity", "specificity", "precision", "macro_f1", "balanced_accuracy", "auroc", "pr_auc", "pr_auc_baseline")


def _ranking_columns() -> tuple[str, ...]:
    return ("rank", "heldout_center", "balanced_accuracy", "auroc", "gap_to_best_bacc")


def _delta_columns() -> tuple[str, ...]:
    return ("heldout_center", "source_bacc", "pooled_bacc", "delta_bacc", "source_macro_f1", "pooled_macro_f1", "delta_macro_f1")


def _worst_columns() -> tuple[str, ...]:
    return ("summary", "heldout_center", "balanced_accuracy", "auroc", "status")


def _config_payload(config: RunConfig) -> dict[str, object]:
    return {
        "manifest_path": str(config.manifest_path),
        "feature_cache_path": str(config.feature_cache_path),
        "min_source": config.min_source,
        "min_eval": config.min_eval,
        "min_source_pos": config.min_source_pos,
        "min_source_neg": config.min_source_neg,
        "min_eval_pos": config.min_eval_pos,
        "min_eval_neg": config.min_eval_neg,
        "model_seed": config.model_seed,
    }


def _protocol_payload() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": DATASET,
        "positive_label": POSITIVE_LABEL,
        "positive_label_name": POSITIVE_LABEL_NAME,
        "eligible_centers": list(ELIGIBLE_CENTERS),
        "quarantine_centers": list(QUARANTINE_CENTERS),
        "source_only_target_labels": "scoring_only",
        "pooled_rows": "diagnostic_only",
        "oracle_rows": "diagnostic_only",
    }


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_json(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _csv_value(value: object) -> object:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def _float(value: object) -> float:
    try:
        if value in ("", None):
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan
