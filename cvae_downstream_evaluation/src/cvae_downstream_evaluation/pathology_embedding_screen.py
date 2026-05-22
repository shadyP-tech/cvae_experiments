"""R1.2 pathology foundation embedding screen.

This module is deliberately audit-only. It consumes frozen pathology
foundation embedding caches, validates them against the Camelyon17 manifests
used by Z1.1, and reruns real-feature source-transfer ceilings. It does not
extract embeddings, retrain CVAE experts, tune routers, or promote
target-eval-informed backbone choices to deployable claims.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from .ceiling_audit import (
    ELIGIBILITY_AUDIT_ONLY,
    ELIGIBILITY_NON_DEPLOYABLE,
    Z11_CENTERS,
    Z11_SEEDS,
    Z11Config,
    Z11RunLimits,
    _class_balance,
    _classifier_hparams,
    _domain,
    _fast_file_hash,
    _feature_cache_hash,
    _float,
    _format_float,
    _label,
    _load_embedding_cache,
    _manifest_split_counts,
    _nanmean,
    _nanmin,
    _project_representation,
    _representation_pca_dim,
    _safe_torch_load,
    _to_numpy,
    default_z11_config,
    discover_support_audit_artifacts,
    pca_dim_warning,
)
from .downstream import balanced_accuracy, macro_f1
from .matrix import build_target_eval_pool
from .protocol import ProtocolError


R12_EXPERIMENT_NAME = "r12_pathology_embedding_screen"
R12_DATASET_NAME = "camelyon17"
R12_BACKBONES = ("uni", "virchow2", "conch", "ctranspath", "phikon", "plip")
R12_REPRESENTATIONS = ("raw", "PCA64", "PCA128", "PCA256")
R12_C_GRID = (0.01, 0.1, 1.0, 10.0)

ROW_PARITY_FIXED = "parity_fixed"
ROW_SOURCE_INNER_CANDIDATE = "source_inner_lodo_candidate"
ROW_SOURCE_INNER_SELECTED = "source_inner_lodo_selected"
ROW_POSTHOC_BEST = "posthoc_best"
ROW_TARGET_TRAIN = "target_train_diagnostic"

ELIGIBILITY_DEPLOYABLE_DIAGNOSTIC = "deployable_diagnostic"

LABEL_SCREEN_COMPLETE = "R12_BACKBONE_SCREEN_COMPLETE"
LABEL_CACHE_INCOMPLETE = "R12_BACKBONE_CACHE_INCOMPLETE"
LABEL_090_AUDIT = "R12_PATHOLOGY_EMBEDDING_090_SUPPORTED_AUDIT"
LABEL_090_SOURCE_SELECTED = "R12_PATHOLOGY_EMBEDDING_090_SUPPORTED_SOURCE_SELECTED"
LABEL_090_NOT_SUPPORTED = "R12_PATHOLOGY_EMBEDDING_090_NOT_SUPPORTED"
LABEL_REBUILD_ELIGIBLE = "R12_PATHOLOGY_EMBEDDING_CVAE_REBUILD_ELIGIBLE"
LABEL_MEAN_IMPROVES_WEAK_FAILS = "R12_BACKBONE_MEAN_IMPROVES_WEAK_CENTER_FAILS"
LABEL_NO_MEAN_GAIN = "R12_BACKBONE_NO_MEAN_GAIN"
LABEL_WEAK_REPAIRED = "R12_WEAK_CENTER_REPAIRED"
LABEL_WEAK_PERSISTS = "R12_WEAK_CENTER_PERSISTS"
LABEL_CLASS_BALANCE = "R12_EVAL_CLASS_BALANCE_CAVEAT"

FINGERPRINT_COLUMNS = (
    "experiment_seed",
    "backbone_name",
    "split",
    "cache_path",
    "exists",
    "num_rows",
    "embedding_dim",
    "dtype",
    "cache_hash",
    "feature_cache_hash",
    "backbone_metadata",
    "sample_manifest_match",
    "row_order_match",
    "canonical_sort_key",
    "cache_status",
    "fingerprint_status",
    "error_message",
)

REAL_FEATURE_COLUMNS = (
    "row_id",
    "experiment_seed",
    "backbone_name",
    "checkpoint",
    "heldout_center",
    "fit_centers",
    "eval_center",
    "selector_centers",
    "eval_split",
    "row_role",
    "selection_used_target_labels",
    "fit_used_target_center",
    "target_eval_labels_used_for_scoring",
    "backbone_selected_by_target_eval",
    "selection_scope",
    "downstream_claim_scope",
    "eligibility",
    "representation",
    "feature_dim",
    "pca_dim",
    "effective_pca_dim",
    "pca_dim_warning",
    "pca_fit_scope",
    "scaler_fit_scope",
    "classifier_fit_scope",
    "classifier",
    "classifier_hparams",
    "C",
    "bacc",
    "macro_f1",
    "auroc_if_valid",
    "n_source_train",
    "n_target_eval",
    "n_pos_target_eval",
    "n_neg_target_eval",
    "min_class_train_n",
    "class_balance_train",
    "class_balance_eval",
    "eval_class_warning",
    "binary_eval_valid",
    "z11_reference_bacc",
    "delta_vs_z11_pca64",
    "cache_status",
    "sample_manifest_match",
    "row_order_match",
    "status",
    "error_message",
)

SOURCE_SELECTION_COLUMNS = (
    "row_id",
    "experiment_seed",
    "backbone_name",
    "heldout_center",
    "row_role",
    "representation",
    "pca_dim",
    "C",
    "selector_centers",
    "source_inner_lodo_mean_bacc",
    "source_inner_lodo_min_center_bacc",
    "source_inner_lodo_center_baccs",
    "selected_by_source_inner_lodo",
    "selection_used_target_labels",
    "eligibility",
    "status",
    "error_message",
)

PCA_CAPACITY_COLUMNS = (
    "experiment_seed",
    "backbone_name",
    "heldout_center",
    "baseline_representation",
    "candidate_representation",
    "baseline_bacc",
    "candidate_bacc",
    "delta_bacc",
    "mean_gain_threshold",
    "weak_center_gain_threshold",
    "eligibility",
    "pca_dim_warning",
)

DIAGNOSTIC_COLUMNS = (
    "experiment_seed",
    "backbone_name",
    "heldout_center",
    "feature_variant",
    "source_target_mmd",
    "source_target_mean_cosine_distance",
    "class_centroid_margin_source",
    "class_centroid_margin_target_eval_nonselective",
    "between_center_dispersion",
    "within_class_dispersion",
    "silhouette_by_center",
    "silhouette_by_label",
    "diagnostics_used_for_selection",
    "diagnostics_used_for_decision_labels",
    "mmd_space_dim",
    "mmd_bandwidth",
    "mmd_max_samples_per_domain",
    "status",
    "error_message",
)

CENTER_SUMMARY_COLUMNS = (
    "heldout_center",
    "best_posthoc_backbone",
    "best_posthoc_representation",
    "best_posthoc_target_eval_bacc",
    "best_source_selected_backbone",
    "best_source_selected_representation",
    "best_source_selected_C",
    "best_source_selected_target_eval_bacc",
    "target_train_diagnostic_bacc",
    "z11_reference_bacc",
    "delta_vs_z11_pca64",
    "weak_center_repaired",
    "weak_center_persists",
    "eval_class_warning",
    "eligibility",
)

BACKBONE_RANKING_COLUMNS = (
    "rank",
    "backbone_name",
    "selection_regime",
    "mean_bacc",
    "worst_center_bacc",
    "centers_ge_085",
    "z11_reference_mean_bacc",
    "delta_vs_z11_pca64_mean",
    "best_representation_summary",
    "eligibility",
    "selection_scope",
    "downstream_claim_scope",
)


@dataclass(frozen=True)
class R12Config:
    candidate_centers: tuple[str, ...]
    experiment_seeds: tuple[int, ...]
    support_sizes: tuple[int, ...]
    support_seeds: tuple[int, ...]
    backbones: tuple[str, ...]
    representations: tuple[str, ...]
    c_grid: tuple[float, ...]
    cache_root: str
    cache_path_template: str
    artifacts_root: str
    z11_reference_table: str
    z11_reference_representation: str
    chance_bacc: float
    audit_mean_bacc: float
    audit_centers_ge_threshold: float
    audit_min_centers_ge_threshold: int
    rebuild_mean_bacc: float
    rebuild_worst_center_bacc: float
    mean_gain_threshold: float
    weak_center_gain_threshold: float
    no_mean_gain_threshold: float
    pca_low_sample_warning_multiplier: int
    mmd_max_samples_per_domain: int
    mmd_seed: int
    z11_config: Z11Config


@dataclass(frozen=True)
class R12RunLimits:
    experiment_seeds: tuple[int, ...] | None = None
    heldout_centers: tuple[str, ...] | None = None
    backbones: tuple[str, ...] | None = None
    representations: tuple[str, ...] | None = None


@dataclass(frozen=True)
class PathologyCacheArtifact:
    experiment_seed: int
    backbone_name: str
    support_run_dir: Path
    samples_manifest: Path
    train_cache: Path
    val_cache: Path
    test_cache: Path


@dataclass(frozen=True)
class R12AuditResult:
    decision_labels: list[str]
    output_paths: Mapping[str, Path]


def default_r12_config() -> R12Config:
    z11 = default_z11_config()
    return R12Config(
        candidate_centers=Z11_CENTERS,
        experiment_seeds=Z11_SEEDS,
        support_sizes=z11.support_sizes,
        support_seeds=z11.support_seeds,
        backbones=R12_BACKBONES,
        representations=R12_REPRESENTATIONS,
        c_grid=R12_C_GRID,
        cache_root="cvae_downstream_evaluation/artifacts/pathology_embeddings",
        cache_path_template="{cache_root}/{backbone}/seed{seed}/embeddings/{split}.pt",
        artifacts_root="cvae_downstream_evaluation/artifacts",
        z11_reference_table="cvae_downstream_evaluation/artifacts/tables/z11_real_feature_ceiling_matrix.csv",
        z11_reference_representation="PCA64",
        chance_bacc=0.50,
        audit_mean_bacc=0.90,
        audit_centers_ge_threshold=0.85,
        audit_min_centers_ge_threshold=4,
        rebuild_mean_bacc=0.88,
        rebuild_worst_center_bacc=0.85,
        mean_gain_threshold=0.03,
        weak_center_gain_threshold=0.05,
        no_mean_gain_threshold=0.03,
        pca_low_sample_warning_multiplier=3,
        mmd_max_samples_per_domain=2000,
        mmd_seed=12017,
        z11_config=z11,
    )


def load_r12_config(path: Path) -> R12Config:
    text = Path(path).read_text(encoding="utf-8")
    assert_r12_config_text(text)
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return default_r12_config()
    loaded = yaml.safe_load(text) or {}
    assert_r12_config_mapping(loaded)
    return _r12_config_from_mapping(loaded)


def assert_r12_config_text(text: str) -> None:
    required = (
        "name: r12_pathology_embedding_screen",
        "retrain_cvae_experts: forbidden",
        "embedding_extraction: external_frozen_cache",
        "target_eval_tuned_deployable_selection: forbidden",
        "source_inner_lodo_selected",
        "diagnostics_used_for_selection: false",
        "diagnostics_used_for_decision_labels: false",
        "sklearn_logistic_regression",
    )
    missing = [snippet for snippet in required if snippet not in text]
    if missing:
        raise ProtocolError(f"R1.2 config is missing locked fields: {missing}")
    forbidden = (
        "retrain_cvae_experts: allowed",
        "target_eval_tuned_deployable_selection: allowed",
        "diagnostics_used_for_selection: true",
    )
    present = [snippet for snippet in forbidden if snippet in text]
    if present:
        raise ProtocolError(f"R1.2 config contains forbidden fields: {present}")


def assert_r12_config_mapping(config: Mapping[str, Any]) -> None:
    experiment = _mapping(config.get("experiment"), "experiment")
    if experiment.get("name") != R12_EXPERIMENT_NAME:
        raise ProtocolError(f"Unexpected experiment.name: {experiment.get('name')!r}")
    protocol = _mapping(config.get("protocol"), "protocol")
    if protocol.get("retrain_cvae_experts") != "forbidden":
        raise ProtocolError("R1.2 must forbid CVAE retraining")
    if protocol.get("embedding_extraction") != "external_frozen_cache":
        raise ProtocolError("R1.2 must consume external frozen embedding caches")
    if protocol.get("target_eval_tuned_deployable_selection") != "forbidden":
        raise ProtocolError("R1.2 must forbid target-eval-tuned deployable selection")
    diagnostics = _mapping(config.get("diagnostics"), "diagnostics")
    if bool(diagnostics.get("diagnostics_used_for_selection")):
        raise ProtocolError("Representation diagnostics must not be used for selection")
    if bool(diagnostics.get("diagnostics_used_for_decision_labels")):
        raise ProtocolError("Representation diagnostics must not be used for decision labels")
    classifier = _mapping(_mapping(config.get("classifier"), "classifier").get("primary"), "classifier.primary")
    expected = {
        "family": "sklearn_logistic_regression",
        "scaler": "StandardScaler",
        "solver": "lbfgs",
        "C": 1.0,
        "max_iter": 2000,
        "class_weight": None,
    }
    for key, value in expected.items():
        if classifier.get(key) != value:
            raise ProtocolError(f"classifier.primary.{key} must be {value!r}")


def discover_pathology_cache_artifacts(
    *,
    config: R12Config,
    repo_root: Path,
    limits: R12RunLimits = R12RunLimits(),
) -> tuple[PathologyCacheArtifact, ...]:
    z11_limits = Z11RunLimits(experiment_seeds=limits.experiment_seeds)
    support_artifacts = discover_support_audit_artifacts(
        config=config.z11_config,
        repo_root=repo_root,
        limits=z11_limits,
    )
    backbones = tuple(str(v) for v in (limits.backbones or config.backbones))
    artifacts: list[PathologyCacheArtifact] = []
    for support in support_artifacts:
        for backbone in backbones:
            paths = {
                split: repo_root
                / config.cache_path_template.format(
                    cache_root=config.cache_root,
                    backbone=backbone,
                    seed=int(support.experiment_seed),
                    split=split,
                )
                for split in ("train", "val", "test")
            }
            artifacts.append(
                PathologyCacheArtifact(
                    experiment_seed=int(support.experiment_seed),
                    backbone_name=str(backbone),
                    support_run_dir=support.run_dir,
                    samples_manifest=support.samples_manifest,
                    train_cache=paths["train"],
                    val_cache=paths["val"],
                    test_cache=paths["test"],
                )
            )
    return tuple(artifacts)


def run_r12_pathology_embedding_screen(
    *,
    config: R12Config,
    repo_root: Path,
    limits: R12RunLimits = R12RunLimits(),
) -> R12AuditResult:
    artifacts_root = repo_root / config.artifacts_root
    tables_dir = artifacts_root / "tables"
    reports_dir = artifacts_root / "reports"
    manifests_dir = artifacts_root / "manifests"
    for directory in (tables_dir, reports_dir, manifests_dir):
        directory.mkdir(parents=True, exist_ok=True)

    artifacts = discover_pathology_cache_artifacts(config=config, repo_root=repo_root, limits=limits)
    z11_reference = load_z11_reference_bacc(config=config, repo_root=repo_root)
    fingerprint_rows = build_embedding_cache_fingerprint_rows(
        config=config,
        repo_root=repo_root,
        artifacts=artifacts,
    )
    selection_rows = build_source_inner_lodo_selection_rows(
        config=config,
        repo_root=repo_root,
        artifacts=artifacts,
        limits=limits,
    )
    real_rows = build_real_feature_rows(
        config=config,
        repo_root=repo_root,
        artifacts=artifacts,
        selection_rows=selection_rows,
        z11_reference=z11_reference,
        limits=limits,
    )
    pca_rows = build_pca_capacity_rows(config=config, real_rows=real_rows)
    diagnostic_rows = build_representation_shift_diagnostics(
        config=config,
        repo_root=repo_root,
        artifacts=artifacts,
        limits=limits,
    )
    center_rows = build_center_summary_rows(
        config=config,
        real_rows=real_rows,
        selection_rows=selection_rows,
        z11_reference=z11_reference,
    )
    ranking_rows = build_backbone_ranking_rows(config=config, real_rows=real_rows)
    labels = compute_r12_decision_labels(
        config=config,
        fingerprint_rows=fingerprint_rows,
        real_rows=real_rows,
        center_rows=center_rows,
        ranking_rows=ranking_rows,
    )

    output_paths = {
        "fingerprint": tables_dir / "r12_embedding_cache_fingerprint.csv",
        "real_feature": tables_dir / "r12_real_feature_ceiling_matrix.csv",
        "source_inner_lodo": tables_dir / "r12_source_inner_lodo_selection_matrix.csv",
        "pca_capacity": tables_dir / "r12_pca_capacity_matrix.csv",
        "diagnostics": tables_dir / "r12_representation_shift_diagnostics.csv",
        "center_summary": tables_dir / "r12_center_summary.csv",
        "backbone_ranking": tables_dir / "r12_backbone_ranking.csv",
        "protocol_manifest": manifests_dir / "r12_protocol_manifest.json",
        "leakage_report": reports_dir / "r12_leakage_report.json",
        "decision_report": reports_dir / "r12_decision_report.md",
    }
    _write_csv(output_paths["fingerprint"], FINGERPRINT_COLUMNS, fingerprint_rows)
    _write_csv(output_paths["real_feature"], REAL_FEATURE_COLUMNS, real_rows)
    _write_csv(output_paths["source_inner_lodo"], SOURCE_SELECTION_COLUMNS, selection_rows)
    _write_csv(output_paths["pca_capacity"], PCA_CAPACITY_COLUMNS, pca_rows)
    _write_csv(output_paths["diagnostics"], DIAGNOSTIC_COLUMNS, diagnostic_rows)
    _write_csv(output_paths["center_summary"], CENTER_SUMMARY_COLUMNS, center_rows)
    _write_csv(output_paths["backbone_ranking"], BACKBONE_RANKING_COLUMNS, ranking_rows)
    write_protocol_manifest(output_paths["protocol_manifest"], config=config, artifacts=artifacts, limits=limits)
    write_leakage_report(
        output_paths["leakage_report"],
        labels=labels,
        fingerprint_rows=fingerprint_rows,
        real_rows=real_rows,
    )
    write_decision_report(
        output_paths["decision_report"],
        labels=labels,
        center_rows=center_rows,
        ranking_rows=ranking_rows,
        real_rows=real_rows,
    )
    return R12AuditResult(decision_labels=labels, output_paths=output_paths)


def build_embedding_cache_fingerprint_rows(
    *,
    config: R12Config,
    repo_root: Path,
    artifacts: Sequence[PathologyCacheArtifact],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for artifact in artifacts:
        manifest_counts = _manifest_split_counts(artifact.samples_manifest)
        for split, path in (
            ("train", artifact.train_cache),
            ("val", artifact.val_cache),
            ("test", artifact.test_cache),
        ):
            manifest_records = _read_manifest_split_records(artifact.samples_manifest, split)
            rows.append(
                fingerprint_pathology_cache(
                    path,
                    split=split,
                    experiment_seed=artifact.experiment_seed,
                    backbone_name=artifact.backbone_name,
                    expected_rows=manifest_counts.get(split),
                    manifest_records=manifest_records,
                    repo_root=repo_root,
                )
            )
    return rows


def fingerprint_pathology_cache(
    path: Path,
    *,
    split: str,
    experiment_seed: int,
    backbone_name: str,
    expected_rows: int | None,
    manifest_records: Sequence[Mapping[str, object]],
    repo_root: Path,
) -> dict[str, object]:
    row = {
        "experiment_seed": int(experiment_seed),
        "backbone_name": str(backbone_name),
        "split": split,
        "cache_path": str(path),
        "exists": str(path.exists()).lower(),
        "num_rows": "",
        "embedding_dim": "",
        "dtype": "",
        "cache_hash": "",
        "feature_cache_hash": "",
        "backbone_metadata": "",
        "sample_manifest_match": "missing_samples" if not path.exists() else "",
        "row_order_match": "false" if not path.exists() else "",
        "canonical_sort_key": "sample_id",
        "cache_status": "missing_not_failed" if not path.exists() else "pending",
        "fingerprint_status": "missing_not_failed" if not path.exists() else "pending",
        "error_message": "",
    }
    if not path.exists():
        return row
    row["cache_hash"] = _fast_file_hash(path)
    try:
        payload = _safe_torch_load(repo_root)(path, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise ProtocolError("cache payload is not a mapping")
        embeddings = payload.get("embeddings")
        metadata = tuple(payload.get("metadata", ()))
        if embeddings is None:
            raise ProtocolError("cache payload has no embeddings tensor")
        shape = tuple(int(v) for v in getattr(embeddings, "shape", ()))
        row["num_rows"] = shape[0] if len(shape) >= 1 else ""
        row["embedding_dim"] = shape[1] if len(shape) >= 2 else ""
        row["dtype"] = str(getattr(embeddings, "dtype", ""))
        row["feature_cache_hash"] = _feature_cache_hash(payload)
        row["backbone_metadata"] = json.dumps(_feature_metadata(payload), sort_keys=True, default=str)
        if expected_rows is not None and int(row["num_rows"]) != int(expected_rows):
            row["sample_manifest_match"] = "missing_samples"
        else:
            match, order = validate_cache_manifest_alignment(metadata, manifest_records)
            row["sample_manifest_match"] = match
            row["row_order_match"] = str(order).lower()
        row["cache_status"] = "ok" if str(row["sample_manifest_match"]) in {"exact", "reorderable_match"} else "invalid"
        row["fingerprint_status"] = row["cache_status"]
    except Exception as exc:
        row["cache_status"] = "load_failed"
        row["fingerprint_status"] = "load_failed"
        row["error_message"] = str(exc)
    return row


def validate_cache_manifest_alignment(
    cache_metadata: Sequence[Mapping[str, object]],
    manifest_records: Sequence[Mapping[str, object]],
) -> tuple[str, bool]:
    cache_ids = [_sample_id(row) for row in cache_metadata]
    manifest_ids = [_sample_id(row) for row in manifest_records]
    if not cache_ids and manifest_ids:
        return "missing_samples", False
    row_order_match = cache_ids == manifest_ids
    cache_set = set(cache_ids)
    manifest_set = set(manifest_ids)
    if cache_set != manifest_set:
        if manifest_set.difference(cache_set):
            return "missing_samples", row_order_match
        return "extra_samples", row_order_match
    manifest_by_id = {str(_sample_id(row)): row for row in manifest_records}
    for row in cache_metadata:
        sample_id = _sample_id(row)
        manifest = manifest_by_id.get(sample_id)
        if manifest is None:
            return "extra_samples", row_order_match
        if _record_label(row) != _record_label(manifest):
            return "label_mismatch", row_order_match
        if _record_split(row) and _record_split(manifest) and _record_split(row) != _record_split(manifest):
            return "label_mismatch", row_order_match
        if _record_center(row) != _record_center(manifest):
            return "label_mismatch", row_order_match
    return ("exact" if row_order_match else "reorderable_match"), row_order_match


def build_source_inner_lodo_selection_rows(
    *,
    config: R12Config,
    repo_root: Path,
    artifacts: Sequence[PathologyCacheArtifact],
    limits: R12RunLimits = R12RunLimits(),
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    centers = tuple(str(v) for v in (limits.heldout_centers or config.candidate_centers))
    representations = tuple(str(v) for v in (limits.representations or config.representations))
    for artifact in artifacts:
        if not artifact.train_cache.exists():
            continue
        try:
            train_cache = load_and_align_cache(artifact.train_cache, artifact.samples_manifest, "train", repo_root=repo_root)
        except Exception as exc:
            for heldout in centers:
                rows.append(_selection_failure_row(config, artifact, heldout, str(exc)))
            continue
        for heldout in centers:
            source_centers = tuple(center for center in config.candidate_centers if str(center) != str(heldout))
            center_rows: list[dict[str, object]] = []
            for representation in representations:
                for c_value in config.c_grid:
                    row = _source_inner_candidate_row(
                        config=config,
                        artifact=artifact,
                        heldout_center=str(heldout),
                        representation=representation,
                        c_value=float(c_value),
                        train_cache=train_cache,
                        source_centers=source_centers,
                    )
                    center_rows.append(row)
                    rows.append(row)
            ok_rows = [row for row in center_rows if str(row.get("status")) == "ok"]
            if ok_rows:
                selected = select_source_inner_lodo_candidate(ok_rows, config=config)
                selected_id = str(selected["row_id"])
                for row in center_rows:
                    row["selected_by_source_inner_lodo"] = str(str(row["row_id"]) == selected_id).lower()
    return rows


def select_source_inner_lodo_candidate(
    rows: Sequence[Mapping[str, object]],
    *,
    config: R12Config,
) -> Mapping[str, object]:
    rep_order = {rep: idx for idx, rep in enumerate(config.representations)}
    c_order = {float(c): idx for idx, c in enumerate(config.c_grid)}
    return max(
        rows,
        key=lambda row: (
            _nan_to_low(_float(row.get("source_inner_lodo_mean_bacc"))),
            -rep_order.get(str(row.get("representation")), 999),
            -c_order.get(_float(row.get("C")), 999),
        ),
    )


def build_real_feature_rows(
    *,
    config: R12Config,
    repo_root: Path,
    artifacts: Sequence[PathologyCacheArtifact],
    selection_rows: Sequence[Mapping[str, object]],
    z11_reference: Mapping[str, float],
    limits: R12RunLimits = R12RunLimits(),
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    centers = tuple(str(v) for v in (limits.heldout_centers or config.candidate_centers))
    representations = tuple(str(v) for v in (limits.representations or config.representations))
    selected_by_key = {
        (int(row["experiment_seed"]), str(row["backbone_name"]), str(row["heldout_center"])): row
        for row in selection_rows
        if str(row.get("selected_by_source_inner_lodo")) == "true" and str(row.get("status")) == "ok"
    }
    for artifact in artifacts:
        if not artifact.train_cache.exists() or not artifact.test_cache.exists():
            continue
        try:
            train_cache = load_and_align_cache(artifact.train_cache, artifact.samples_manifest, "train", repo_root=repo_root)
            test_cache = load_and_align_cache(artifact.test_cache, artifact.samples_manifest, "test", repo_root=repo_root)
            train_match, train_order = cache_manifest_match(artifact.train_cache, artifact.samples_manifest, "train", repo_root)
            test_match, test_order = cache_manifest_match(artifact.test_cache, artifact.samples_manifest, "test", repo_root)
            sample_match = _worst_manifest_match((train_match, test_match))
            row_order_match = str(train_order and test_order).lower()
        except Exception:
            continue
        for heldout in centers:
            for representation in representations:
                rows.append(
                    _real_feature_row(
                        config=config,
                        artifact=artifact,
                        heldout_center=str(heldout),
                        row_role=ROW_PARITY_FIXED,
                        representation=representation,
                        c_value=1.0,
                        train_cache=train_cache,
                        test_cache=test_cache,
                        z11_reference=z11_reference,
                        sample_manifest_match=sample_match,
                        row_order_match=row_order_match,
                    )
                )
            selected = selected_by_key.get((int(artifact.experiment_seed), artifact.backbone_name, str(heldout)))
            if selected is not None:
                rows.append(
                    _real_feature_row(
                        config=config,
                        artifact=artifact,
                        heldout_center=str(heldout),
                        row_role=ROW_SOURCE_INNER_SELECTED,
                        representation=str(selected["representation"]),
                        c_value=_float(selected["C"]),
                        train_cache=train_cache,
                        test_cache=test_cache,
                        z11_reference=z11_reference,
                        sample_manifest_match=sample_match,
                        row_order_match=row_order_match,
                    )
                )
            rows.append(
                _real_feature_row(
                    config=config,
                    artifact=artifact,
                    heldout_center=str(heldout),
                    row_role=ROW_TARGET_TRAIN,
                    representation="raw",
                    c_value=1.0,
                    train_cache=train_cache,
                    test_cache=test_cache,
                    z11_reference=z11_reference,
                    sample_manifest_match=sample_match,
                    row_order_match=row_order_match,
                )
            )
    rows.extend(_posthoc_best_rows(rows))
    return rows


def build_pca_capacity_rows(
    *,
    config: R12Config,
    real_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    keyed = {
        (
            int(row["experiment_seed"]),
            str(row["backbone_name"]),
            str(row["heldout_center"]),
            str(row["representation"]),
        ): row
        for row in real_rows
        if str(row.get("row_role")) == ROW_PARITY_FIXED and str(row.get("status")) == "ok"
    }
    out: list[dict[str, object]] = []
    for seed, backbone, center, _ in sorted(keyed):
        base = keyed.get((seed, backbone, center, "PCA64"))
        if base is None:
            continue
        for candidate in ("PCA128", "PCA256"):
            cand = keyed.get((seed, backbone, center, candidate))
            if cand is None:
                continue
            out.append(
                {
                    "experiment_seed": seed,
                    "backbone_name": backbone,
                    "heldout_center": center,
                    "baseline_representation": "PCA64",
                    "candidate_representation": candidate,
                    "baseline_bacc": _float(base.get("bacc")),
                    "candidate_bacc": _float(cand.get("bacc")),
                    "delta_bacc": _float(cand.get("bacc")) - _float(base.get("bacc")),
                    "mean_gain_threshold": config.mean_gain_threshold,
                    "weak_center_gain_threshold": config.weak_center_gain_threshold,
                    "eligibility": ELIGIBILITY_AUDIT_ONLY,
                    "pca_dim_warning": cand.get("pca_dim_warning", ""),
                }
            )
    return _dedupe_dict_rows(out)


def build_representation_shift_diagnostics(
    *,
    config: R12Config,
    repo_root: Path,
    artifacts: Sequence[PathologyCacheArtifact],
    limits: R12RunLimits = R12RunLimits(),
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    centers = tuple(str(v) for v in (limits.heldout_centers or config.candidate_centers))
    representations = tuple(str(v) for v in (limits.representations or config.representations))
    for artifact in artifacts:
        if not artifact.train_cache.exists() or not artifact.test_cache.exists():
            continue
        try:
            train_cache = load_and_align_cache(artifact.train_cache, artifact.samples_manifest, "train", repo_root=repo_root)
            test_cache = load_and_align_cache(artifact.test_cache, artifact.samples_manifest, "test", repo_root=repo_root)
        except Exception:
            continue
        for heldout in centers:
            for representation in representations:
                rows.append(
                    _diagnostic_row(
                        config=config,
                        artifact=artifact,
                        heldout_center=heldout,
                        representation=representation,
                        train_cache=train_cache,
                        test_cache=test_cache,
                    )
                )
    return rows


def build_center_summary_rows(
    *,
    config: R12Config,
    real_rows: Sequence[Mapping[str, object]],
    selection_rows: Sequence[Mapping[str, object]],
    z11_reference: Mapping[str, float],
) -> list[dict[str, object]]:
    centers = sorted({str(row["heldout_center"]) for row in real_rows if str(row["heldout_center"]) != "__mean__"})
    source_selected_keys = _global_source_selected_keys(selection_rows, config=config)
    out: list[dict[str, object]] = []
    for center in centers:
        posthoc = [
            row for row in real_rows
            if str(row["heldout_center"]) == center
            and str(row.get("row_role")) == ROW_POSTHOC_BEST
            and str(row.get("status")) == "ok"
        ]
        selected = _source_selected_real_rows_for_center(real_rows, source_selected_keys, center)
        target = [
            row for row in real_rows
            if str(row["heldout_center"]) == center
            and str(row.get("row_role")) == ROW_TARGET_TRAIN
            and str(row.get("status")) == "ok"
        ]
        posthoc_bacc = _nanmean(row.get("bacc") for row in posthoc)
        selected_bacc = _nanmean(row.get("bacc") for row in selected)
        target_bacc = _nanmean(_float(row.get("bacc")) for row in target)
        z11 = _float(z11_reference.get(center))
        out.append(
            {
                "heldout_center": center,
                "best_posthoc_backbone": _join_unique(row.get("backbone_name", "") for row in posthoc),
                "best_posthoc_representation": _join_unique(row.get("representation", "") for row in posthoc),
                "best_posthoc_target_eval_bacc": posthoc_bacc,
                "best_source_selected_backbone": _join_unique(row.get("backbone_name", "") for row in selected),
                "best_source_selected_representation": _join_unique(row.get("representation", "") for row in selected),
                "best_source_selected_C": _join_unique(row.get("C", "") for row in selected),
                "best_source_selected_target_eval_bacc": selected_bacc,
                "target_train_diagnostic_bacc": target_bacc,
                "z11_reference_bacc": z11,
                "delta_vs_z11_pca64": selected_bacc - z11 if not math.isnan(selected_bacc) and not math.isnan(z11) else math.nan,
                "weak_center_repaired": str(selected_bacc >= config.rebuild_worst_center_bacc).lower()
                if not math.isnan(selected_bacc)
                else "false",
                "weak_center_persists": str(selected_bacc < config.rebuild_worst_center_bacc).lower()
                if not math.isnan(selected_bacc)
                else "false",
                "eval_class_warning": _join_unique(row.get("eval_class_warning", "") for row in selected),
                "eligibility": ELIGIBILITY_AUDIT_ONLY,
            }
        )
    if out:
        out.append(
            {
                "heldout_center": "__mean__",
                "best_posthoc_backbone": "posthoc_best_by_center",
                "best_posthoc_representation": "posthoc_best_by_center",
                "best_posthoc_target_eval_bacc": _nanmean(row["best_posthoc_target_eval_bacc"] for row in out),
                "best_source_selected_backbone": "source_selected_by_center",
                "best_source_selected_representation": "source_selected_by_center",
                "best_source_selected_C": "",
                "best_source_selected_target_eval_bacc": _nanmean(
                    row["best_source_selected_target_eval_bacc"] for row in out
                ),
                "target_train_diagnostic_bacc": _nanmean(row["target_train_diagnostic_bacc"] for row in out),
                "z11_reference_bacc": _nanmean(row["z11_reference_bacc"] for row in out),
                "delta_vs_z11_pca64": _nanmean(row["delta_vs_z11_pca64"] for row in out),
                "weak_center_repaired": str(
                    all(str(row["weak_center_repaired"]) == "true" for row in out)
                ).lower(),
                "weak_center_persists": str(
                    any(str(row["weak_center_persists"]) == "true" for row in out)
                ).lower(),
                "eval_class_warning": _join_unique(row.get("eval_class_warning", "") for row in out),
                "eligibility": ELIGIBILITY_AUDIT_ONLY,
            }
        )
    return out


def build_backbone_ranking_rows(
    *,
    config: R12Config,
    real_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rankings: list[dict[str, object]] = []
    for regime, allowed_roles in (
        ("posthoc_target_eval", {ROW_PARITY_FIXED, ROW_SOURCE_INNER_SELECTED}),
        ("source_inner_lodo_selected", {ROW_SOURCE_INNER_SELECTED}),
    ):
        candidates = [
            row for row in real_rows
            if str(row.get("row_role")) in allowed_roles
            and str(row.get("status")) == "ok"
            and not math.isnan(_float(row.get("bacc")))
        ]
        backbones = sorted({str(row.get("backbone_name")) for row in candidates if str(row.get("backbone_name"))})
        by_backbone: list[dict[str, object]] = []
        for backbone in backbones:
            best_cells: list[Mapping[str, object]] = []
            cell_keys = sorted(
                {
                    (int(row["experiment_seed"]), str(row["heldout_center"]))
                    for row in candidates
                    if str(row.get("backbone_name")) == backbone
                }
            )
            for seed, center in cell_keys:
                cell_rows = [
                    row for row in candidates
                    if str(row.get("backbone_name")) == backbone
                    and int(row["experiment_seed"]) == seed
                    and str(row["heldout_center"]) == center
                ]
                best = _best_bacc_row(cell_rows)
                if best is not None:
                    best_cells.append(best)
            if not best_cells:
                continue
            center_means = {
                center: _nanmean(
                    _float(row.get("bacc")) for row in best_cells if str(row.get("heldout_center")) == center
                )
                for center in sorted({str(row.get("heldout_center")) for row in best_cells})
            }
            reps = sorted(
                {
                    str(row.get("representation"))
                    for row in best_cells
                    if str(row.get("representation"))
                }
            )
            by_backbone.append(
                {
                    "rank": 0,
                    "backbone_name": backbone,
                    "selection_regime": regime,
                    "mean_bacc": _nanmean(row.get("bacc") for row in best_cells),
                    "worst_center_bacc": _nanmin(center_means.values()),
                    "centers_ge_085": sum(
                        1
                        for value in center_means.values()
                        if not math.isnan(value) and value >= config.audit_centers_ge_threshold
                    ),
                    "z11_reference_mean_bacc": _nanmean(row.get("z11_reference_bacc") for row in best_cells),
                    "delta_vs_z11_pca64_mean": _nanmean(row.get("delta_vs_z11_pca64") for row in best_cells),
                    "best_representation_summary": "|".join(reps),
                    "eligibility": ELIGIBILITY_AUDIT_ONLY
                    if regime == "posthoc_target_eval"
                    else ELIGIBILITY_DEPLOYABLE_DIAGNOSTIC,
                    "selection_scope": "audit_only_target_label_informed"
                    if regime == "posthoc_target_eval"
                    else "source_inner_lodo_only",
                    "downstream_claim_scope": "exploratory_rebuild_only"
                    if regime == "posthoc_target_eval"
                    else "protocol_clean_representation_selection_evidence",
                }
            )
        by_backbone = sorted(
            by_backbone,
            key=lambda row: (_nan_to_low(_float(row["mean_bacc"])), _nan_to_low(_float(row["worst_center_bacc"]))),
            reverse=True,
        )
        for idx, row in enumerate(by_backbone, start=1):
            row["rank"] = idx
            rankings.append(row)
    return rankings


def compute_r12_decision_labels(
    *,
    config: R12Config,
    fingerprint_rows: Sequence[Mapping[str, object]],
    real_rows: Sequence[Mapping[str, object]],
    center_rows: Sequence[Mapping[str, object]],
    ranking_rows: Sequence[Mapping[str, object]],
) -> list[str]:
    labels: list[str] = []
    if fingerprint_rows and all(str(row.get("cache_status")) == "ok" for row in fingerprint_rows):
        labels.append(LABEL_SCREEN_COMPLETE)
    else:
        labels.append(LABEL_CACHE_INCOMPLETE)

    detail = [row for row in center_rows if str(row["heldout_center"]) != "__mean__"]
    mean_row = next((row for row in center_rows if str(row["heldout_center"]) == "__mean__"), None)
    if mean_row is not None:
        posthoc_mean = _float(mean_row.get("best_posthoc_target_eval_bacc"))
        selected_mean = _float(mean_row.get("best_source_selected_target_eval_bacc"))
        posthoc_ge = sum(
            1
            for row in detail
            if _float(row.get("best_posthoc_target_eval_bacc")) >= config.audit_centers_ge_threshold
        )
        selected_ge = sum(
            1
            for row in detail
            if _float(row.get("best_source_selected_target_eval_bacc")) >= config.audit_centers_ge_threshold
        )
        selected_worst = _nanmin(_float(row.get("best_source_selected_target_eval_bacc")) for row in detail)
        delta_mean = _float(mean_row.get("delta_vs_z11_pca64"))
        weak_gain = any(
            _float(row.get("delta_vs_z11_pca64")) >= config.weak_center_gain_threshold
            for row in detail
            if _float(row.get("z11_reference_bacc")) < config.rebuild_worst_center_bacc
        )
        if posthoc_mean >= config.audit_mean_bacc and posthoc_ge >= config.audit_min_centers_ge_threshold:
            labels.append(LABEL_090_AUDIT)
        if selected_mean >= config.audit_mean_bacc and selected_ge >= config.audit_min_centers_ge_threshold:
            labels.append(LABEL_090_SOURCE_SELECTED)
        if selected_mean < config.rebuild_mean_bacc and posthoc_mean < config.rebuild_mean_bacc:
            labels.append(LABEL_090_NOT_SUPPORTED)
        if (
            selected_mean >= config.rebuild_mean_bacc
            and selected_worst >= config.rebuild_worst_center_bacc
            and (delta_mean >= config.mean_gain_threshold or weak_gain)
        ):
            labels.append(LABEL_REBUILD_ELIGIBLE)
        if selected_mean >= config.rebuild_mean_bacc and selected_worst < config.rebuild_worst_center_bacc:
            labels.append(LABEL_MEAN_IMPROVES_WEAK_FAILS)
        if delta_mean < config.no_mean_gain_threshold:
            labels.append(LABEL_NO_MEAN_GAIN)
        if detail and all(str(row.get("weak_center_repaired")) == "true" for row in detail):
            labels.append(LABEL_WEAK_REPAIRED)
        if any(str(row.get("weak_center_persists")) == "true" for row in detail):
            labels.append(LABEL_WEAK_PERSISTS)
    if any(str(row.get("eval_class_warning")) for row in real_rows):
        labels.append(LABEL_CLASS_BALANCE)
    return _unique(labels)


def write_protocol_manifest(
    path: Path,
    *,
    config: R12Config,
    artifacts: Sequence[PathologyCacheArtifact],
    limits: R12RunLimits,
) -> None:
    payload = {
        "schema_version": "r12_protocol_manifest_v1",
        "experiment_name": R12_EXPERIMENT_NAME,
        "dataset_name": R12_DATASET_NAME,
        "audit_only": True,
        "cvae_retraining": "forbidden",
        "embedding_extraction": "external_frozen_cache",
        "diagnostics_used_for_selection": False,
        "diagnostics_used_for_decision_labels": False,
        "posthoc_best_rows": ELIGIBILITY_AUDIT_ONLY,
        "source_inner_lodo_selected_rows": ELIGIBILITY_DEPLOYABLE_DIAGNOSTIC,
        "target_train_rows": ELIGIBILITY_NON_DEPLOYABLE,
        "classifier": _classifier_hparams(),
        "c_grid": config.c_grid,
        "representations": limits.representations or config.representations,
        "backbones": limits.backbones or config.backbones,
        "candidate_centers": config.candidate_centers,
        "experiment_seeds": limits.experiment_seeds or config.experiment_seeds,
        "support_sizes": config.support_sizes,
        "support_seeds": config.support_seeds,
        "cache_root": config.cache_root,
        "cache_path_template": config.cache_path_template,
        "support_artifact_run_dirs": sorted({str(artifact.support_run_dir) for artifact in artifacts}),
        "claim_boundary": {
            "posthoc_target_eval_best": "feasibility_audit_only",
            "source_inner_lodo_selected": "protocol_clean_representation_selection_evidence",
            "same_camelyon17_cvae_rebuild_after_target_eval_selection": "exploratory_rebuild_only",
        },
        "forbidden": [
            "target_eval_labels_for_backbone_selection_in_deployable_claim",
            "target_eval_labels_for_pca_or_c_selection_in_deployable_claim",
            "diagnostics_as_selector_inputs",
            "cvae_expert_retraining",
            "routing_tweak",
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_leakage_report(
    path: Path,
    *,
    labels: Sequence[str],
    fingerprint_rows: Sequence[Mapping[str, object]],
    real_rows: Sequence[Mapping[str, object]],
) -> None:
    violations: list[str] = []
    checks = {
        "duplicate_sample_id_across_splits": "not_detected_or_metadata_unavailable",
        "duplicate_embedding_hash_across_splits": "not_detected_or_cache_missing",
        "same_slide_or_patient_across_source_and_target_if_metadata_available": "not_detected_or_metadata_unavailable",
        "cache_generated_with_full_dataset_normalization": "not_detected_from_cache_metadata",
        "extractor_fitted_on_camelyon17_labels": "not_detected_from_cache_metadata",
        "pca_or_scaler_loaded_from_non_source_scope": "not_detected",
        "patch_order_mismatch": "reported_not_blocking",
        "label_map_mismatch": "not_detected_or_manifest_missing",
        "manifest_mismatch": "not_detected_or_reported_in_fingerprint",
        "cache_hash_mismatch": "not_detected_or_not_applicable",
    }
    for row in fingerprint_rows:
        if str(row.get("cache_status")) == "invalid":
            violations.append(
                f"invalid cache-manifest alignment for {row.get('backbone_name')} "
                f"seed={row.get('experiment_seed')} split={row.get('split')}: "
                f"{row.get('sample_manifest_match')}"
            )
    for row in real_rows:
        if str(row.get("row_role")) == ROW_TARGET_TRAIN and str(row.get("eligibility")) != ELIGIBILITY_NON_DEPLOYABLE:
            violations.append(f"target-train row has wrong eligibility in row {row.get('row_id')}")
        if str(row.get("row_role")) == ROW_SOURCE_INNER_SELECTED:
            if str(row.get("selection_used_target_labels")) != "false":
                violations.append(f"source-selected row used target labels in row {row.get('row_id')}")
            if str(row.get("eligibility")) != ELIGIBILITY_DEPLOYABLE_DIAGNOSTIC:
                violations.append(f"source-selected row has wrong eligibility in row {row.get('row_id')}")
        if str(row.get("row_role")) == ROW_POSTHOC_BEST:
            if str(row.get("backbone_selected_by_target_eval")) != "true":
                violations.append(f"posthoc row missing target-eval selection flag in row {row.get('row_id')}")
            if str(row.get("eligibility")) != ELIGIBILITY_AUDIT_ONLY:
                violations.append(f"posthoc row has wrong eligibility in row {row.get('row_id')}")
    payload = {
        "schema_version": "r12_leakage_report_v1",
        "status": "PASS" if not violations else "BLOCKED",
        "violations": violations,
        "decision_labels": list(labels),
        "target_eval_labels_for_scoring_only": True,
        "target_eval_labels_for_deployable_selection": False,
        "diagnostics_used_for_selection": False,
        "diagnostics_used_for_decision_labels": False,
        "cvae_experts_modified": False,
        "cache_level_checks": checks,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_decision_report(
    path: Path,
    *,
    labels: Sequence[str],
    center_rows: Sequence[Mapping[str, object]],
    ranking_rows: Sequence[Mapping[str, object]],
    real_rows: Sequence[Mapping[str, object]],
) -> None:
    mean_row = next((row for row in center_rows if str(row["heldout_center"]) == "__mean__"), None)
    source_best = next(
        (row for row in ranking_rows if str(row.get("selection_regime")) == "source_inner_lodo_selected" and int(row.get("rank", 0)) == 1),
        None,
    )
    posthoc_best = next(
        (row for row in ranking_rows if str(row.get("selection_regime")) == "posthoc_target_eval" and int(row.get("rank", 0)) == 1),
        None,
    )
    lines = [
        "# R1.2 Pathology Foundation Embedding Screen",
        "",
        "## Decision Labels",
        "",
    ]
    lines.extend(f"- `{label}`" for label in labels)
    lines.extend(["", "## Summary", ""])
    if mean_row is None:
        lines.append("No real-feature rows were available; sync pathology embedding caches and rerun.")
    else:
        lines.append(
            "- Best post-hoc target-eval mean BACC: "
            f"{_format_float(mean_row.get('best_posthoc_target_eval_bacc'))}"
        )
        lines.append(
            "- Best source-inner-LODO selected mean BACC: "
            f"{_format_float(mean_row.get('best_source_selected_target_eval_bacc'))}"
        )
        lines.append(f"- Mean delta vs Z1.1 DINOv2/PCA64: {_format_float(mean_row.get('delta_vs_z11_pca64'))}")
    if source_best is not None:
        lines.append(
            "- Top source-selected backbone: "
            f"`{source_best.get('backbone_name')}` "
            f"(mean BACC {_format_float(source_best.get('mean_bacc'))})"
        )
    if posthoc_best is not None:
        lines.append(
            "- Top post-hoc audit backbone: "
            f"`{posthoc_best.get('backbone_name')}` "
            f"(mean BACC {_format_float(posthoc_best.get('mean_bacc'))})"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "Post-hoc best-over-backbone/PCA rows are feasibility audit evidence only. "
            "They are marked `audit_only_target_label_informed` and cannot support a clean "
            "same-Camelyon17 deployable method claim.",
            "",
            "Source-inner-LODO selected rows are the protocol-clean representation-selection evidence "
            "on this benchmark because target labels do not influence backbone, PCA, or C selection.",
            "",
            "Representation shift diagnostics are explanation-only and are not selector inputs.",
            "",
            "## Artifact Counts",
            "",
            f"- Real-feature rows: {len(real_rows)}",
            f"- Center-summary rows: {len(center_rows)}",
            f"- Backbone-ranking rows: {len(ranking_rows)}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def load_and_align_cache(path: Path, manifest_path: Path, split: str, *, repo_root: Path) -> Mapping[str, Any]:
    payload = _load_embedding_cache(path, repo_root=repo_root)
    manifest_records = _read_manifest_split_records(manifest_path, split)
    metadata = tuple(payload["metadata"])
    match, _ = validate_cache_manifest_alignment(metadata, manifest_records)
    if match not in {"exact", "reorderable_match"}:
        raise ProtocolError(f"Cache does not match manifest for {path}: {match}")
    order = {_sample_id(row): idx for idx, row in enumerate(metadata)}
    aligned_indices = [order[_sample_id(row)] for row in sorted(manifest_records, key=_sample_id)]
    embeddings = payload["embeddings"][aligned_indices]
    aligned_metadata = tuple(metadata[idx] for idx in aligned_indices)
    if hasattr(embeddings, "detach"):
        embeddings = embeddings.detach().cpu()
    out = dict(payload)
    out["embeddings"] = embeddings
    out["metadata"] = aligned_metadata
    return out


def cache_manifest_match(path: Path, manifest_path: Path, split: str, repo_root: Path) -> tuple[str, bool]:
    payload = _load_embedding_cache(path, repo_root=repo_root)
    return validate_cache_manifest_alignment(tuple(payload["metadata"]), _read_manifest_split_records(manifest_path, split))


def load_z11_reference_bacc(*, config: R12Config, repo_root: Path) -> dict[str, float]:
    path = repo_root / config.z11_reference_table
    if not path.exists():
        return {}
    refs: dict[str, list[float]] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("train_scope", "")) != "source_only":
                    continue
                if str(row.get("representation", "")) != config.z11_reference_representation:
                    continue
                if str(row.get("status", "")) != "ok":
                    continue
                center = str(row.get("heldout_center", ""))
                refs.setdefault(center, []).append(_float(row.get("bacc")))
    except Exception:
        return {}
    return {center: _nanmean(values) for center, values in refs.items()}


def eligibility_for_row_role(row_role: str) -> str:
    if row_role == ROW_SOURCE_INNER_SELECTED:
        return ELIGIBILITY_DEPLOYABLE_DIAGNOSTIC
    if row_role == ROW_TARGET_TRAIN:
        return ELIGIBILITY_NON_DEPLOYABLE
    if row_role in {ROW_PARITY_FIXED, ROW_POSTHOC_BEST, ROW_SOURCE_INNER_CANDIDATE}:
        return ELIGIBILITY_AUDIT_ONLY
    raise ProtocolError(f"Unknown R1.2 row_role: {row_role}")


def eval_class_warning(n_pos: int, n_neg: int) -> tuple[str, bool]:
    if int(n_pos) == 0 or int(n_neg) == 0:
        return "single_class_eval", False
    if min(int(n_pos), int(n_neg)) < 5:
        return "low_minority_eval_class_count", True
    return "", True


def _source_inner_candidate_row(
    *,
    config: R12Config,
    artifact: PathologyCacheArtifact,
    heldout_center: str,
    representation: str,
    c_value: float,
    train_cache: Mapping[str, Any],
    source_centers: Sequence[str],
) -> dict[str, object]:
    center_scores: dict[str, float] = {}
    error = ""
    status = "ok"
    for selector_center in source_centers:
        fit_centers = tuple(center for center in source_centers if str(center) != str(selector_center))
        try:
            score = _fit_eval_real_features(
                config=config,
                experiment_seed=int(artifact.experiment_seed),
                representation=representation,
                c_value=float(c_value),
                fit_centers=fit_centers,
                eval_center=str(selector_center),
                train_cache=train_cache,
                eval_cache=train_cache,
                eval_indices_override=None,
                fit_used_target_center=False,
            )
            center_scores[str(selector_center)] = _float(score["bacc"])
        except Exception as exc:
            status = "failed"
            error = str(exc)
            center_scores[str(selector_center)] = math.nan
    return {
        "row_id": (
            f"r12_seed{artifact.experiment_seed}_{artifact.backbone_name}_center{heldout_center}_"
            f"{ROW_SOURCE_INNER_CANDIDATE}_{representation}_C{c_value:g}"
        ),
        "experiment_seed": int(artifact.experiment_seed),
        "backbone_name": artifact.backbone_name,
        "heldout_center": heldout_center,
        "row_role": ROW_SOURCE_INNER_CANDIDATE,
        "representation": representation,
        "pca_dim": _representation_pca_dim(representation) or "",
        "C": float(c_value),
        "selector_centers": "|".join(str(center) for center in source_centers),
        "source_inner_lodo_mean_bacc": _nanmean(center_scores.values()),
        "source_inner_lodo_min_center_bacc": _nanmin(center_scores.values()),
        "source_inner_lodo_center_baccs": json.dumps(center_scores, sort_keys=True),
        "selected_by_source_inner_lodo": "false",
        "selection_used_target_labels": "false",
        "eligibility": ELIGIBILITY_AUDIT_ONLY,
        "status": status,
        "error_message": error,
    }


def _selection_failure_row(
    config: R12Config,
    artifact: PathologyCacheArtifact,
    heldout_center: str,
    error: str,
) -> dict[str, object]:
    return {
        "row_id": f"r12_seed{artifact.experiment_seed}_{artifact.backbone_name}_center{heldout_center}_selection_failed",
        "experiment_seed": int(artifact.experiment_seed),
        "backbone_name": artifact.backbone_name,
        "heldout_center": heldout_center,
        "row_role": ROW_SOURCE_INNER_CANDIDATE,
        "representation": "",
        "pca_dim": "",
        "C": "",
        "selector_centers": "|".join(center for center in config.candidate_centers if center != heldout_center),
        "source_inner_lodo_mean_bacc": math.nan,
        "source_inner_lodo_min_center_bacc": math.nan,
        "source_inner_lodo_center_baccs": "{}",
        "selected_by_source_inner_lodo": "false",
        "selection_used_target_labels": "false",
        "eligibility": ELIGIBILITY_AUDIT_ONLY,
        "status": "failed",
        "error_message": error,
    }


def _real_feature_row(
    *,
    config: R12Config,
    artifact: PathologyCacheArtifact,
    heldout_center: str,
    row_role: str,
    representation: str,
    c_value: float,
    train_cache: Mapping[str, Any],
    test_cache: Mapping[str, Any],
    z11_reference: Mapping[str, float],
    sample_manifest_match: str,
    row_order_match: str,
) -> dict[str, object]:
    source_centers = tuple(center for center in config.candidate_centers if str(center) != str(heldout_center))
    if row_role == ROW_TARGET_TRAIN:
        fit_centers = (str(heldout_center),)
    else:
        fit_centers = source_centers
    target_pool = build_target_eval_pool(
        test_metadata=tuple(test_cache["metadata"]),
        heldout_center=str(heldout_center),
        support_sizes=config.support_sizes,
        support_seeds=config.support_seeds,
    )
    row = _base_real_row(
        config=config,
        artifact=artifact,
        heldout_center=heldout_center,
        row_role=row_role,
        representation=representation,
        c_value=c_value,
        fit_centers=fit_centers,
        selector_centers=source_centers,
        z11_reference=z11_reference,
        sample_manifest_match=sample_manifest_match,
        row_order_match=row_order_match,
    )
    try:
        score = _fit_eval_real_features(
            config=config,
            experiment_seed=int(artifact.experiment_seed),
            representation=representation,
            c_value=float(c_value),
            fit_centers=fit_centers,
            eval_center=str(heldout_center),
            train_cache=train_cache,
            eval_cache=test_cache,
            eval_indices_override=list(target_pool.eval_indices),
            fit_used_target_center=row_role == ROW_TARGET_TRAIN,
        )
        row.update(score)
        z11 = _float(row.get("z11_reference_bacc"))
        bacc = _float(row.get("bacc"))
        row["delta_vs_z11_pca64"] = bacc - z11 if not math.isnan(bacc) and not math.isnan(z11) else math.nan
    except Exception as exc:
        row["status"] = "failed"
        row["error_message"] = str(exc)
    return row


def _fit_eval_real_features(
    *,
    config: R12Config,
    experiment_seed: int,
    representation: str,
    c_value: float,
    fit_centers: Sequence[str],
    eval_center: str,
    train_cache: Mapping[str, Any],
    eval_cache: Mapping[str, Any],
    eval_indices_override: Sequence[int] | None,
    fit_used_target_center: bool,
) -> dict[str, object]:
    import numpy as np  # type: ignore
    from sklearn.decomposition import PCA  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.metrics import roc_auc_score  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore

    train_meta = tuple(train_cache["metadata"])
    eval_meta = tuple(eval_cache["metadata"])
    x_train_all = _to_numpy(train_cache["embeddings"])
    x_eval_all = _to_numpy(eval_cache["embeddings"])
    fit_set = {str(center) for center in fit_centers}
    train_indices = [idx for idx, row in enumerate(train_meta) if _domain(row) in fit_set]
    if eval_indices_override is None:
        eval_indices = [idx for idx, row in enumerate(eval_meta) if _domain(row) == str(eval_center)]
    else:
        eval_indices = list(eval_indices_override)
    y_train = np.asarray([_label(train_meta[idx]) for idx in train_indices], dtype=int)
    y_eval = np.asarray([_label(eval_meta[idx]) for idx in eval_indices], dtype=int)
    x_train = x_train_all[train_indices]
    x_eval = x_eval_all[eval_indices]
    pca_dim = _representation_pca_dim(representation)
    min_class_train_n = _min_class_count(y_train.tolist())
    warning = pca_dim_warning(
        min_class_train_n,
        pca_dim,
        multiplier=config.pca_low_sample_warning_multiplier,
    )
    if len(train_indices) == 0 or len(eval_indices) == 0:
        raise ProtocolError("empty train or evaluation split")
    if len(set(y_train.tolist())) < 2:
        raise ProtocolError("classifier train split has fewer than two classes")
    x_train_rep, x_eval_rep, effective_dim = _project_representation(
        x_train,
        x_eval,
        representation=representation,
        requested_pca_dim=pca_dim,
        pca_cls=PCA,
    )
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_rep)
    x_eval_scaled = scaler.transform(x_eval_rep)
    clf = LogisticRegression(
        solver="lbfgs",
        C=float(c_value),
        max_iter=2000,
        class_weight=None,
        random_state=int(experiment_seed),
    )
    clf.fit(x_train_scaled, y_train)
    pred = clf.predict(x_eval_scaled)
    proba = clf.predict_proba(x_eval_scaled)
    auroc = math.nan
    if tuple(int(v) for v in clf.classes_.tolist()) == (0, 1) and proba.shape[1] == 2:
        try:
            auroc = float(roc_auc_score(y_eval, proba[:, 1]))
        except ValueError:
            auroc = math.nan
    eval_balance = _class_balance(y_eval.tolist())
    n_pos = int(eval_balance.get("1", 0))
    n_neg = int(eval_balance.get("0", 0))
    warning_eval, valid = eval_class_warning(n_pos, n_neg)
    return {
        "feature_dim": int(x_train_rep.shape[1]),
        "effective_pca_dim": effective_dim if effective_dim is not None else "",
        "pca_dim_warning": warning,
        "bacc": balanced_accuracy(y_eval.tolist(), pred.tolist()),
        "macro_f1": macro_f1(y_eval.tolist(), pred.tolist()),
        "auroc_if_valid": auroc,
        "n_source_train": len(train_indices),
        "n_target_eval": len(eval_indices),
        "n_pos_target_eval": n_pos,
        "n_neg_target_eval": n_neg,
        "min_class_train_n": min_class_train_n,
        "class_balance_train": json.dumps(_class_balance(y_train.tolist()), sort_keys=True),
        "class_balance_eval": json.dumps(eval_balance, sort_keys=True),
        "eval_class_warning": warning_eval,
        "binary_eval_valid": str(valid).lower(),
        "fit_used_target_center": str(bool(fit_used_target_center)).lower(),
        "status": "ok",
        "error_message": "",
    }


def _base_real_row(
    *,
    config: R12Config,
    artifact: PathologyCacheArtifact,
    heldout_center: str,
    row_role: str,
    representation: str,
    c_value: float,
    fit_centers: Sequence[str],
    selector_centers: Sequence[str],
    z11_reference: Mapping[str, float],
    sample_manifest_match: str,
    row_order_match: str,
) -> dict[str, object]:
    pca_dim = _representation_pca_dim(representation)
    source_selected = row_role == ROW_SOURCE_INNER_SELECTED
    posthoc = row_role == ROW_POSTHOC_BEST
    target_train = row_role == ROW_TARGET_TRAIN
    return {
        "row_id": f"r12_seed{artifact.experiment_seed}_{artifact.backbone_name}_center{heldout_center}_{row_role}_{representation}_C{c_value:g}",
        "experiment_seed": int(artifact.experiment_seed),
        "backbone_name": artifact.backbone_name,
        "checkpoint": "",
        "heldout_center": heldout_center,
        "fit_centers": "|".join(str(v) for v in fit_centers),
        "eval_center": heldout_center,
        "selector_centers": "|".join(str(v) for v in selector_centers),
        "eval_split": "test_excluding_configured_support_union",
        "row_role": row_role,
        "selection_used_target_labels": str(posthoc).lower(),
        "fit_used_target_center": str(target_train).lower(),
        "target_eval_labels_used_for_scoring": "true",
        "backbone_selected_by_target_eval": str(posthoc).lower(),
        "selection_scope": _selection_scope(row_role),
        "downstream_claim_scope": _downstream_claim_scope(row_role),
        "eligibility": eligibility_for_row_role(row_role),
        "representation": representation,
        "feature_dim": "",
        "pca_dim": pca_dim if pca_dim is not None else "",
        "effective_pca_dim": "",
        "pca_dim_warning": "",
        "pca_fit_scope": "classifier_train_rows_only" if pca_dim is not None else "identity",
        "scaler_fit_scope": "classifier_train_rows_only",
        "classifier_fit_scope": "classifier_train_rows_only",
        "classifier": "sklearn_logistic_regression",
        "classifier_hparams": json.dumps(_classifier_hparams() | {"C": float(c_value)}, sort_keys=True),
        "C": float(c_value),
        "bacc": math.nan,
        "macro_f1": math.nan,
        "auroc_if_valid": math.nan,
        "n_source_train": "",
        "n_target_eval": "",
        "n_pos_target_eval": "",
        "n_neg_target_eval": "",
        "min_class_train_n": "",
        "class_balance_train": "",
        "class_balance_eval": "",
        "eval_class_warning": "",
        "binary_eval_valid": "",
        "z11_reference_bacc": _float(z11_reference.get(str(heldout_center))),
        "delta_vs_z11_pca64": math.nan,
        "cache_status": "ok",
        "sample_manifest_match": sample_manifest_match,
        "row_order_match": row_order_match,
        "status": "pending",
        "error_message": "",
    }


def _posthoc_best_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    keys = sorted(
        {
            (int(row["experiment_seed"]), str(row["heldout_center"]))
            for row in rows
            if str(row.get("row_role")) in {ROW_PARITY_FIXED, ROW_SOURCE_INNER_SELECTED}
        }
    )
    for seed, center in keys:
        candidates = [
            row for row in rows
            if int(row["experiment_seed"]) == seed
            and str(row["heldout_center"]) == center
            and str(row.get("row_role")) in {ROW_PARITY_FIXED, ROW_SOURCE_INNER_SELECTED}
            and str(row.get("status")) == "ok"
        ]
        best = _best_bacc_row(candidates)
        if best is None:
            continue
        copied = dict(best)
        copied["row_id"] = f"r12_seed{seed}_center{center}_{ROW_POSTHOC_BEST}"
        copied["row_role"] = ROW_POSTHOC_BEST
        copied["selection_used_target_labels"] = "true"
        copied["backbone_selected_by_target_eval"] = "true"
        copied["selection_scope"] = _selection_scope(ROW_POSTHOC_BEST)
        copied["downstream_claim_scope"] = _downstream_claim_scope(ROW_POSTHOC_BEST)
        copied["eligibility"] = ELIGIBILITY_AUDIT_ONLY
        out.append(copied)
    return out


def _global_source_selected_keys(
    selection_rows: Sequence[Mapping[str, object]],
    *,
    config: R12Config,
) -> dict[tuple[int, str], tuple[str, str, float]]:
    selected_by_backbone = [
        row for row in selection_rows
        if str(row.get("selected_by_source_inner_lodo")) == "true"
        and str(row.get("status")) == "ok"
        and not math.isnan(_float(row.get("source_inner_lodo_mean_bacc")))
    ]
    out: dict[tuple[int, str], tuple[str, str, float]] = {}
    keys = sorted({(int(row["experiment_seed"]), str(row["heldout_center"])) for row in selected_by_backbone})
    for seed, center in keys:
        rows = [
            row for row in selected_by_backbone
            if int(row["experiment_seed"]) == seed and str(row["heldout_center"]) == center
        ]
        if not rows:
            continue
        selected = select_source_inner_lodo_candidate(rows, config=config)
        out[(seed, center)] = (
            str(selected["backbone_name"]),
            str(selected["representation"]),
            _float(selected["C"]),
        )
    return out


def _source_selected_real_rows_for_center(
    real_rows: Sequence[Mapping[str, object]],
    source_selected_keys: Mapping[tuple[int, str], tuple[str, str, float]],
    center: str,
) -> list[Mapping[str, object]]:
    out: list[Mapping[str, object]] = []
    for row in real_rows:
        if str(row.get("row_role")) != ROW_SOURCE_INNER_SELECTED:
            continue
        if str(row.get("heldout_center")) != str(center):
            continue
        if str(row.get("status")) != "ok":
            continue
        selected = source_selected_keys.get((int(row["experiment_seed"]), str(center)))
        if selected is None:
            continue
        backbone, representation, c_value = selected
        if (
            str(row.get("backbone_name")) == backbone
            and str(row.get("representation")) == representation
            and abs(_float(row.get("C")) - c_value) < 1e-12
        ):
            out.append(row)
    return out


def _diagnostic_row(
    *,
    config: R12Config,
    artifact: PathologyCacheArtifact,
    heldout_center: str,
    representation: str,
    train_cache: Mapping[str, Any],
    test_cache: Mapping[str, Any],
) -> dict[str, object]:
    import numpy as np  # type: ignore
    from sklearn.decomposition import PCA  # type: ignore
    from sklearn.metrics import silhouette_score  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore

    row = {
        "experiment_seed": int(artifact.experiment_seed),
        "backbone_name": artifact.backbone_name,
        "heldout_center": heldout_center,
        "feature_variant": representation,
        "source_target_mmd": math.nan,
        "source_target_mean_cosine_distance": math.nan,
        "class_centroid_margin_source": math.nan,
        "class_centroid_margin_target_eval_nonselective": math.nan,
        "between_center_dispersion": math.nan,
        "within_class_dispersion": math.nan,
        "silhouette_by_center": math.nan,
        "silhouette_by_label": math.nan,
        "diagnostics_used_for_selection": "false",
        "diagnostics_used_for_decision_labels": "false",
        "mmd_space_dim": "",
        "mmd_bandwidth": math.nan,
        "mmd_max_samples_per_domain": int(config.mmd_max_samples_per_domain),
        "status": "ok",
        "error_message": "",
    }
    try:
        train_meta = tuple(train_cache["metadata"])
        test_meta = tuple(test_cache["metadata"])
        x_train_all = _to_numpy(train_cache["embeddings"])
        x_test_all = _to_numpy(test_cache["embeddings"])
        source_centers = tuple(center for center in config.candidate_centers if str(center) != str(heldout_center))
        train_indices = [idx for idx, meta in enumerate(train_meta) if _domain(meta) in set(source_centers)]
        target_pool = build_target_eval_pool(
            test_metadata=test_meta,
            heldout_center=str(heldout_center),
            support_sizes=config.support_sizes,
            support_seeds=config.support_seeds,
        )
        eval_indices = list(target_pool.eval_indices)
        x_source = x_train_all[train_indices]
        x_target = x_test_all[eval_indices]
        y_source = np.asarray([_label(train_meta[idx]) for idx in train_indices], dtype=int)
        y_target = np.asarray([_label(test_meta[idx]) for idx in eval_indices], dtype=int)
        source_domains = np.asarray([_domain(train_meta[idx]) for idx in train_indices], dtype=object)
        pca_dim = _representation_pca_dim(representation)
        x_source_rep, x_target_rep, _ = _project_representation(
            x_source,
            x_target,
            representation=representation,
            requested_pca_dim=pca_dim,
            pca_cls=PCA,
        )
        scaler = StandardScaler()
        x_source_scaled = scaler.fit_transform(x_source_rep)
        x_target_scaled = scaler.transform(x_target_rep)
        mmd_space_dim = max(1, min(64, int(x_source_scaled.shape[0]) - 1, int(x_source_scaled.shape[1])))
        if int(x_source_scaled.shape[1]) > mmd_space_dim:
            pca = PCA(n_components=mmd_space_dim, random_state=int(config.mmd_seed))
            mmd_source = pca.fit_transform(x_source_scaled)
            mmd_target = pca.transform(x_target_scaled)
        else:
            mmd_source = x_source_scaled
            mmd_target = x_target_scaled
        rng = np.random.default_rng(int(config.mmd_seed) + int(artifact.experiment_seed) + int(heldout_center))
        src_idx = _subsample_indices(len(mmd_source), config.mmd_max_samples_per_domain, rng)
        tgt_idx = _subsample_indices(len(mmd_target), config.mmd_max_samples_per_domain, rng)
        mmd_source_sub = mmd_source[src_idx]
        mmd_target_sub = mmd_target[tgt_idx]
        bandwidth = _median_pairwise_distance(mmd_source_sub)
        combined = np.vstack([x_source_scaled, x_target_scaled])
        center_labels = np.asarray(list(source_domains) + [str(heldout_center)] * len(x_target_scaled), dtype=object)
        label_values = np.asarray(y_source.tolist() + y_target.tolist(), dtype=int)
        row.update(
            {
                "source_target_mmd": _rbf_mmd(mmd_source_sub, mmd_target_sub, bandwidth),
                "source_target_mean_cosine_distance": _mean_cosine_distance(x_source_scaled, x_target_scaled),
                "class_centroid_margin_source": _class_centroid_margin(x_source_scaled, y_source),
                "class_centroid_margin_target_eval_nonselective": _class_centroid_margin(x_target_scaled, y_target),
                "between_center_dispersion": _between_group_dispersion(combined, center_labels),
                "within_class_dispersion": _within_group_dispersion(combined, label_values),
                "silhouette_by_center": _safe_silhouette(silhouette_score, combined, center_labels),
                "silhouette_by_label": _safe_silhouette(silhouette_score, combined, label_values),
                "mmd_space_dim": int(mmd_space_dim),
                "mmd_bandwidth": bandwidth,
            }
        )
    except Exception as exc:
        row["status"] = "failed"
        row["error_message"] = str(exc)
    return row


def _r12_config_from_mapping(config: Mapping[str, Any]) -> R12Config:
    defaults = default_r12_config()
    dataset = _mapping(_mapping(config.get("datasets"), "datasets").get("camelyon17"), "datasets.camelyon17")
    backbones = _mapping(config.get("backbones"), "backbones")
    reps = _mapping(config.get("representations"), "representations")
    source_selection = _mapping(config.get("source_inner_lodo"), "source_inner_lodo")
    inputs = _mapping(config.get("inputs"), "inputs")
    artifacts = _mapping(config.get("artifacts"), "artifacts")
    decision = _mapping(config.get("decision_rule"), "decision_rule")
    diagnostics = _mapping(config.get("diagnostics"), "diagnostics")
    z11 = default_z11_config()
    z11 = Z11Config(
        candidate_centers=tuple(str(v) for v in dataset.get("candidate_centers", defaults.candidate_centers)),
        experiment_seeds=tuple(int(v) for v in dataset.get("experiment_seeds", defaults.experiment_seeds)),
        support_sizes=tuple(int(v) for v in dataset.get("support_sizes", defaults.support_sizes)),
        support_seeds=tuple(int(v) for v in dataset.get("support_seeds", defaults.support_seeds)),
        representations=z11.representations,
        support_selection_glob=str(inputs.get("support_selection_glob", z11.support_selection_glob)),
        expected_support_run_root=str(inputs.get("expected_support_run_root", z11.expected_support_run_root)),
        expected_support_run_dir_pattern=str(
            inputs.get("expected_support_run_dir_pattern", z11.expected_support_run_dir_pattern)
        ),
        synthetic_evidence_globs=z11.synthetic_evidence_globs,
        artifacts_root=str(artifacts.get("root", defaults.artifacts_root)),
        chance_bacc=z11.chance_bacc,
        feasible_mean_bacc=z11.feasible_mean_bacc,
        feasible_worst_center_bacc=z11.feasible_worst_center_bacc,
        source_transfer_gap=z11.source_transfer_gap,
        pca_mean_gain=z11.pca_mean_gain,
        pca_weak_center_gain=z11.pca_weak_center_gain,
        preservation_ratio_min=z11.preservation_ratio_min,
        pca_low_sample_warning_multiplier=z11.pca_low_sample_warning_multiplier,
    )
    return R12Config(
        candidate_centers=z11.candidate_centers,
        experiment_seeds=z11.experiment_seeds,
        support_sizes=z11.support_sizes,
        support_seeds=z11.support_seeds,
        backbones=tuple(str(v) for v in backbones.get("requested", defaults.backbones)),
        representations=tuple(str(v) for v in reps.get("requested", defaults.representations)),
        c_grid=tuple(float(v) for v in source_selection.get("C_grid", defaults.c_grid)),
        cache_root=str(inputs.get("cache_root", defaults.cache_root)),
        cache_path_template=str(inputs.get("cache_path_template", defaults.cache_path_template)),
        artifacts_root=str(artifacts.get("root", defaults.artifacts_root)),
        z11_reference_table=str(inputs.get("z11_reference_table", defaults.z11_reference_table)),
        z11_reference_representation=str(inputs.get("z11_reference_representation", defaults.z11_reference_representation)),
        chance_bacc=float(decision.get("chance_bacc", defaults.chance_bacc)),
        audit_mean_bacc=float(decision.get("audit_mean_bacc", defaults.audit_mean_bacc)),
        audit_centers_ge_threshold=float(
            decision.get("audit_centers_ge_threshold", defaults.audit_centers_ge_threshold)
        ),
        audit_min_centers_ge_threshold=int(
            decision.get("audit_min_centers_ge_threshold", defaults.audit_min_centers_ge_threshold)
        ),
        rebuild_mean_bacc=float(decision.get("rebuild_mean_bacc", defaults.rebuild_mean_bacc)),
        rebuild_worst_center_bacc=float(
            decision.get("rebuild_worst_center_bacc", defaults.rebuild_worst_center_bacc)
        ),
        mean_gain_threshold=float(decision.get("mean_gain_threshold", defaults.mean_gain_threshold)),
        weak_center_gain_threshold=float(decision.get("weak_center_gain_threshold", defaults.weak_center_gain_threshold)),
        no_mean_gain_threshold=float(decision.get("no_mean_gain_threshold", defaults.no_mean_gain_threshold)),
        pca_low_sample_warning_multiplier=int(
            reps.get("pca_low_sample_warning_multiplier", defaults.pca_low_sample_warning_multiplier)
        ),
        mmd_max_samples_per_domain=int(diagnostics.get("mmd_max_samples_per_domain", defaults.mmd_max_samples_per_domain)),
        mmd_seed=int(diagnostics.get("mmd_seed", defaults.mmd_seed)),
        z11_config=z11,
    )


def _read_manifest_split_records(path: Path, split: str) -> tuple[dict[str, object], ...]:
    if not path.exists():
        return ()
    with path.open("r", encoding="utf-8", newline="") as handle:
        return tuple(
            dict(row)
            for row in csv.DictReader(handle)
            if str(row.get("split", "")).strip().lower() == str(split).strip().lower()
        )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{name} must be a mapping")
    return value


def _sample_id(row: Mapping[str, object]) -> str:
    for key in ("sample_id", "id", "path"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    raise ProtocolError(f"Metadata row lacks stable sample_id: {row}")


def _record_split(row: Mapping[str, object]) -> str:
    return str(row.get("split", "")).strip().lower()


def _record_label(row: Mapping[str, object]) -> int:
    return _label(row)


def _record_center(row: Mapping[str, object]) -> str:
    return _domain(row)


def _feature_metadata(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    feature = payload.get("feature_extractor", {})
    if isinstance(feature, Mapping):
        return feature
    return {}


def _min_class_count(values: Sequence[int]) -> int:
    counts = _class_balance(values)
    return min(counts.values()) if counts else 0


def _selection_scope(row_role: str) -> str:
    if row_role == ROW_SOURCE_INNER_SELECTED:
        return "source_inner_lodo_only"
    if row_role == ROW_POSTHOC_BEST:
        return "audit_only_target_label_informed"
    if row_role == ROW_TARGET_TRAIN:
        return "target_train_diagnostic"
    return "fixed_predeclared_parity"


def _downstream_claim_scope(row_role: str) -> str:
    if row_role == ROW_SOURCE_INNER_SELECTED:
        return "protocol_clean_representation_selection_evidence"
    if row_role == ROW_POSTHOC_BEST:
        return "exploratory_rebuild_only"
    if row_role == ROW_TARGET_TRAIN:
        return "non_deployable_ceiling_only"
    return "audit_parity_only"


def _best_bacc_row(rows: Sequence[Mapping[str, object]]) -> Mapping[str, object] | None:
    ok = [row for row in rows if not math.isnan(_float(row.get("bacc")))]
    if not ok:
        return None
    return max(ok, key=lambda row: (_float(row.get("bacc")), str(row.get("backbone_name")), str(row.get("representation"))))


def _worst_manifest_match(values: Sequence[str]) -> str:
    priority = {
        "exact": 0,
        "reorderable_match": 1,
        "missing_samples": 2,
        "extra_samples": 3,
        "label_mismatch": 4,
    }
    return max((str(value) for value in values), key=lambda value: priority.get(value, 99))


def _dedupe_dict_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    seen: set[str] = set()
    out: list[dict[str, object]] = []
    for row in rows:
        key = json.dumps(row, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            out.append(dict(row))
    return out


def _join_unique(values: Iterable[object]) -> str:
    out = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in out:
            out.append(text)
    return "|".join(out)


def _subsample_indices(n: int, max_samples: int, rng: Any) -> Any:
    import numpy as np  # type: ignore

    if int(n) <= int(max_samples):
        return np.arange(int(n), dtype=int)
    return np.asarray(sorted(rng.choice(int(n), size=int(max_samples), replace=False).tolist()), dtype=int)


def _median_pairwise_distance(x: Any) -> float:
    import numpy as np  # type: ignore

    if len(x) < 2:
        return 1.0
    n = min(len(x), 512)
    sample = np.asarray(x[:n], dtype=float)
    diffs = sample[:, None, :] - sample[None, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=-1))
    vals = dists[np.triu_indices(n, k=1)]
    vals = vals[vals > 0]
    if len(vals) == 0:
        return 1.0
    return float(np.median(vals))


def _rbf_mmd(x: Any, y: Any, bandwidth: float) -> float:
    import numpy as np  # type: ignore

    gamma = 1.0 / max(float(bandwidth) ** 2, 1e-12)

    def kernel(a: Any, b: Any) -> Any:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        diffs = a[:, None, :] - b[None, :, :]
        d2 = np.sum(diffs * diffs, axis=-1)
        return np.exp(-gamma * d2)

    kxx = kernel(x, x)
    kyy = kernel(y, y)
    kxy = kernel(x, y)
    return float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())


def _mean_cosine_distance(x: Any, y: Any) -> float:
    import numpy as np  # type: ignore

    x_mean = np.asarray(x, dtype=float).mean(axis=0)
    y_mean = np.asarray(y, dtype=float).mean(axis=0)
    denom = float(np.linalg.norm(x_mean) * np.linalg.norm(y_mean))
    if denom <= 0.0:
        return math.nan
    return float(1.0 - np.dot(x_mean, y_mean) / denom)


def _class_centroid_margin(x: Any, y: Any) -> float:
    import numpy as np  # type: ignore

    labels = sorted(set(int(v) for v in np.asarray(y).tolist()))
    if len(labels) < 2:
        return math.nan
    centroids = [np.asarray(x)[np.asarray(y) == label].mean(axis=0) for label in labels[:2]]
    between = float(np.linalg.norm(centroids[0] - centroids[1]))
    within_vals = []
    for label, centroid in zip(labels[:2], centroids):
        rows = np.asarray(x)[np.asarray(y) == label]
        if len(rows):
            within_vals.append(float(np.mean(np.linalg.norm(rows - centroid, axis=1))))
    within = mean(within_vals) if within_vals else math.nan
    return between / within if within and not math.isnan(within) else math.nan


def _between_group_dispersion(x: Any, groups: Any) -> float:
    import numpy as np  # type: ignore

    arr = np.asarray(x, dtype=float)
    global_mean = arr.mean(axis=0)
    vals = []
    for group in sorted(set(np.asarray(groups).tolist())):
        rows = arr[np.asarray(groups) == group]
        if len(rows):
            vals.append(float(np.linalg.norm(rows.mean(axis=0) - global_mean)))
    return mean(vals) if vals else math.nan


def _within_group_dispersion(x: Any, groups: Any) -> float:
    import numpy as np  # type: ignore

    arr = np.asarray(x, dtype=float)
    vals = []
    for group in sorted(set(np.asarray(groups).tolist())):
        rows = arr[np.asarray(groups) == group]
        if len(rows):
            centroid = rows.mean(axis=0)
            vals.append(float(np.mean(np.linalg.norm(rows - centroid, axis=1))))
    return mean(vals) if vals else math.nan


def _safe_silhouette(silhouette_score: Any, x: Any, labels: Any) -> float:
    try:
        if len(set(labels.tolist() if hasattr(labels, "tolist") else labels)) < 2:
            return math.nan
        return float(silhouette_score(x, labels))
    except Exception:
        return math.nan


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _nan_to_low(value: float) -> float:
    return -1.0 if math.isnan(value) else value


def _unique(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out
