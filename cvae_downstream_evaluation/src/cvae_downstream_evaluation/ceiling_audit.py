"""Z1.1 current-setup ceiling audit.

This module is deliberately audit-only. It consumes frozen Camelyon17 support
artifacts when they are available, computes real-feature ceiling diagnostics,
and joins optional predeclared C6.3 evidence without changing CVAE experts,
routing rules, generation settings, or deployable method eligibility.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from .downstream import balanced_accuracy, macro_f1
from .matrix import build_target_eval_pool
from .protocol import ProtocolError


Z11_EXPERIMENT_NAME = "z11_current_setup_ceiling_audit"
Z11_DATASET_NAME = "camelyon17"
Z11_CENTERS = ("0", "1", "2", "3", "4")
Z11_SEEDS = (42, 43, 44)
Z11_REPRESENTATIONS = ("raw", "PCA64", "PCA64_reconstruction", "PCA128", "PCA256")
Z11_PREDECLARED_REPRESENTATION = "PCA64"

ELIGIBILITY_DEPLOYABLE = "deployable"
ELIGIBILITY_AUDIT_ONLY = "audit_only"
ELIGIBILITY_NON_DEPLOYABLE = "non_deployable"

LABEL_IDENTITY_PASS = "Z11_IDENTITY_REPLAY_PASS"
LABEL_IDENTITY_INCOMPLETE = "Z11_IDENTITY_REPLAY_INCOMPLETE"
LABEL_FEASIBLE = "Z11_CURRENT_SETUP_090_FEASIBLE"
LABEL_NOT_SUPPORTED = "Z11_CURRENT_SETUP_090_NOT_SUPPORTED"
LABEL_PCA_BOTTLENECK = "Z11_PCA_CAPACITY_BOTTLENECK"
LABEL_PCA_NO_GAIN = "Z11_PCA_CAPACITY_NO_GAIN"
LABEL_SOURCE_TRANSFER = "Z11_SOURCE_TRANSFER_BOTTLENECK"
LABEL_CVAE_PRESERVATION = "Z11_CVAE_PRESERVATION_BOTTLENECK"
LABEL_WEAK_CENTER = "Z11_WEAK_CENTER_DOMAIN_SHIFT_BOTTLENECK"
LABEL_SYNTHETIC_MISSING = "Z11_SYNTHETIC_EVIDENCE_MISSING"

FINGERPRINT_COLUMNS = (
    "experiment_seed",
    "split",
    "tensor_path",
    "exists",
    "num_rows",
    "embedding_dim",
    "dtype",
    "sha256_or_fast_hash",
    "feature_cache_hash",
    "matches_manifest",
    "fingerprint_status",
    "error_message",
)

REAL_FEATURE_COLUMNS = (
    "row_id",
    "experiment_seed",
    "heldout_center",
    "train_centers",
    "eval_center",
    "eval_split",
    "train_scope",
    "representation",
    "feature_dim",
    "pca_dim",
    "effective_pca_dim",
    "pca_dim_warning",
    "projection_fit_scope",
    "scaler_fit_scope",
    "classifier",
    "classifier_hparams",
    "uses_target_train_labels",
    "uses_target_support_labels",
    "uses_target_eval_labels_for_training",
    "uses_target_eval_labels_for_scoring",
    "eligibility",
    "bacc",
    "macro_f1",
    "auroc_if_valid",
    "n_train",
    "n_eval",
    "min_class_train_n",
    "class_balance_train",
    "class_balance_eval",
    "status",
    "error_message",
)

PCA_CAPACITY_COLUMNS = (
    "experiment_seed",
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

SYNTHETIC_PRESERVATION_COLUMNS = (
    "heldout_center",
    "real_feature_reference_bacc",
    "c63_oracle_bacc",
    "c63_geometric_bacc",
    "chance_bacc",
    "oracle_preservation_ratio",
    "geometric_preservation_ratio",
    "real_minus_c63_gap",
    "matched_cell_count",
    "missing_cell_count",
    "evidence_status",
)

CENTER_SUMMARY_COLUMNS = (
    "heldout_center",
    "best_posthoc_source_only_representation",
    "best_posthoc_source_only_bacc",
    "predeclared_or_sourceval_selected_representation",
    "predeclared_or_sourceval_selected_bacc",
    "target_train_diagnostic_bacc",
    "target_minus_source_gap",
    "pca_capacity_best_gain",
    "weak_center_bottleneck",
    "source_transfer_bottleneck",
    "eligibility",
)


@dataclass(frozen=True)
class Z11Config:
    candidate_centers: tuple[str, ...]
    experiment_seeds: tuple[int, ...]
    support_sizes: tuple[int, ...]
    support_seeds: tuple[int, ...]
    representations: tuple[str, ...]
    support_selection_glob: str
    expected_support_run_root: str
    expected_support_run_dir_pattern: str
    synthetic_evidence_globs: tuple[str, ...]
    artifacts_root: str
    chance_bacc: float
    feasible_mean_bacc: float
    feasible_worst_center_bacc: float
    source_transfer_gap: float
    pca_mean_gain: float
    pca_weak_center_gain: float
    preservation_ratio_min: float
    pca_low_sample_warning_multiplier: int


@dataclass(frozen=True)
class Z11RunLimits:
    experiment_seeds: tuple[int, ...] | None = None
    heldout_centers: tuple[str, ...] | None = None
    representations: tuple[str, ...] | None = None


@dataclass(frozen=True)
class SupportAuditArtifact:
    experiment_seed: int
    run_dir: Path
    train_cache: Path
    val_cache: Path
    test_cache: Path
    samples_manifest: Path
    config_resolved: Path
    support_selection_path: Path


@dataclass(frozen=True)
class Z11AuditResult:
    decision_labels: list[str]
    output_paths: Mapping[str, Path]


def default_z11_config() -> Z11Config:
    return Z11Config(
        candidate_centers=Z11_CENTERS,
        experiment_seeds=Z11_SEEDS,
        support_sizes=(4, 8, 16, 32),
        support_seeds=(17, 23, 31),
        representations=Z11_REPRESENTATIONS,
        support_selection_glob=(
            "cvae_testing/outputs/camelyon17/"
            "camelyon17_support_estimated_utility_routing_v2/"
            "support_utility_v2_seed*/reports/support_response_sample_selections.csv"
        ),
        expected_support_run_root=(
            "cvae_testing/outputs/camelyon17/"
            "camelyon17_support_estimated_utility_routing_v2"
        ),
        expected_support_run_dir_pattern="support_utility_v2_seed{seed}",
        synthetic_evidence_globs=(
            "cvae_downstream_evaluation/artifacts/tables/*c63*.csv",
            "cvae_downstream_evaluation/artifacts/**/c63*.csv",
            "cvae_testing/results/comparison_tables/**/*c63*.csv",
        ),
        artifacts_root="cvae_downstream_evaluation/artifacts",
        chance_bacc=0.50,
        feasible_mean_bacc=0.90,
        feasible_worst_center_bacc=0.85,
        source_transfer_gap=0.05,
        pca_mean_gain=0.02,
        pca_weak_center_gain=0.03,
        preservation_ratio_min=0.70,
        pca_low_sample_warning_multiplier=3,
    )


def load_z11_config(path: Path) -> Z11Config:
    text = Path(path).read_text(encoding="utf-8")
    assert_z11_config_text(text)
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return default_z11_config()
    loaded = yaml.safe_load(text) or {}
    assert_z11_config_mapping(loaded)
    return _z11_config_from_mapping(loaded)


def assert_z11_config_text(text: str) -> None:
    required = (
        "name: z11_current_setup_ceiling_audit",
        "retrain_cvae_experts: forbidden",
        "target_train_rows: non_deployable",
        "source_only_rows: audit_only",
        "PCA64_reconstruction",
        "sklearn_logistic_regression",
    )
    missing = [snippet for snippet in required if snippet not in text]
    if missing:
        raise ProtocolError(f"Z1.1 config is missing locked fields: {missing}")
    forbidden = (
        "target_eval_tuned_deployable_selection: allowed",
        "retrain_cvae_experts: allowed",
        "hyperparameter_tuning: allowed",
    )
    present = [snippet for snippet in forbidden if snippet in text]
    if present:
        raise ProtocolError(f"Z1.1 config contains forbidden fields: {present}")


def assert_z11_config_mapping(config: Mapping[str, Any]) -> None:
    experiment = _mapping(config.get("experiment"), "experiment")
    if experiment.get("name") != Z11_EXPERIMENT_NAME:
        raise ProtocolError(f"Unexpected experiment.name: {experiment.get('name')!r}")
    if experiment.get("stage") != "current_setup_ceiling_audit":
        raise ProtocolError("experiment.stage must be current_setup_ceiling_audit")

    protocol = _mapping(config.get("protocol"), "protocol")
    if protocol.get("retrain_cvae_experts") != "forbidden":
        raise ProtocolError("Z1.1 must forbid CVAE retraining")
    if protocol.get("target_train_rows") != ELIGIBILITY_NON_DEPLOYABLE:
        raise ProtocolError("target_train_rows must be non_deployable")
    if protocol.get("source_only_rows") != ELIGIBILITY_AUDIT_ONLY:
        raise ProtocolError("source_only_rows must be audit_only")

    classifier = _mapping(_mapping(config.get("classifier"), "classifier").get("primary"), "classifier.primary")
    expected = {
        "family": "sklearn_logistic_regression",
        "scaler": "StandardScaler",
        "solver": "lbfgs",
        "C": 1.0,
        "max_iter": 2000,
        "class_weight": None,
        "hyperparameter_tuning": "forbidden",
    }
    for key, value in expected.items():
        if classifier.get(key) != value:
            raise ProtocolError(f"classifier.primary.{key} must be {value!r}")

    reps = tuple(str(value) for value in _mapping(config.get("representations"), "representations").get("requested", ()))
    missing_reps = sorted(set(Z11_REPRESENTATIONS).difference(reps))
    if missing_reps:
        raise ProtocolError(f"Z1.1 representation list is missing: {missing_reps}")


def discover_support_audit_artifacts(
    *,
    config: Z11Config,
    repo_root: Path,
    limits: Z11RunLimits = Z11RunLimits(),
) -> tuple[SupportAuditArtifact, ...]:
    support_paths = sorted(Path(path) for path in glob.glob(str(repo_root / config.support_selection_glob)))
    artifacts: list[SupportAuditArtifact] = []
    if support_paths:
        for path in support_paths:
            run_dir = path.parent.parent
            seed = _seed_from_path(run_dir) or _seed_from_path(path)
            if seed is None:
                seed = _seed_from_config_resolved(run_dir / "config_resolved.yaml")
            if seed is None:
                raise ProtocolError(f"Could not infer experiment seed from support artifact: {path}")
            artifacts.append(_support_artifact(seed, run_dir, path))
    else:
        root = repo_root / config.expected_support_run_root
        for seed in config.experiment_seeds:
            run_dir = root / config.expected_support_run_dir_pattern.format(seed=seed)
            artifacts.append(_support_artifact(int(seed), run_dir, run_dir / "reports" / "support_response_sample_selections.csv"))

    if limits.experiment_seeds is not None:
        allowed = {int(seed) for seed in limits.experiment_seeds}
        artifacts = [artifact for artifact in artifacts if int(artifact.experiment_seed) in allowed]
    return tuple(sorted(artifacts, key=lambda item: int(item.experiment_seed)))


def run_z11_ceiling_audit(
    *,
    config: Z11Config,
    repo_root: Path,
    limits: Z11RunLimits = Z11RunLimits(),
) -> Z11AuditResult:
    artifacts_root = repo_root / config.artifacts_root
    tables_dir = artifacts_root / "tables"
    reports_dir = artifacts_root / "reports"
    manifests_dir = artifacts_root / "manifests"
    for directory in (tables_dir, reports_dir, manifests_dir):
        directory.mkdir(parents=True, exist_ok=True)

    artifacts = discover_support_audit_artifacts(config=config, repo_root=repo_root, limits=limits)
    fingerprint_rows = build_feature_tensor_fingerprint_rows(repo_root=repo_root, artifacts=artifacts)
    real_rows = build_real_feature_ceiling_rows(
        config=config,
        repo_root=repo_root,
        artifacts=artifacts,
        limits=limits,
    )
    pca_rows = build_pca_capacity_rows(config=config, real_rows=real_rows)
    center_rows = build_center_summary_rows(config=config, real_rows=real_rows, pca_rows=pca_rows)
    synthetic_rows = build_synthetic_preservation_rows(
        config=config,
        repo_root=repo_root,
        center_summary_rows=center_rows,
    )
    labels = compute_decision_labels(
        config=config,
        fingerprint_rows=fingerprint_rows,
        real_rows=real_rows,
        pca_rows=pca_rows,
        center_summary_rows=center_rows,
        synthetic_rows=synthetic_rows,
    )

    output_paths = {
        "fingerprint": tables_dir / "z11_feature_tensor_fingerprint.csv",
        "real_feature": tables_dir / "z11_real_feature_ceiling_matrix.csv",
        "pca_capacity": tables_dir / "z11_pca_capacity_ceiling_matrix.csv",
        "synthetic_preservation": tables_dir / "z11_synthetic_preservation_gap.csv",
        "center_summary": tables_dir / "z11_center_summary.csv",
        "protocol_manifest": manifests_dir / "z11_protocol_manifest.json",
        "leakage_report": reports_dir / "z11_leakage_report.json",
        "decision_report": reports_dir / "z11_decision_report.md",
    }
    _write_csv(output_paths["fingerprint"], FINGERPRINT_COLUMNS, fingerprint_rows)
    _write_csv(output_paths["real_feature"], REAL_FEATURE_COLUMNS, real_rows)
    _write_csv(output_paths["pca_capacity"], PCA_CAPACITY_COLUMNS, pca_rows)
    _write_csv(output_paths["synthetic_preservation"], SYNTHETIC_PRESERVATION_COLUMNS, synthetic_rows)
    _write_csv(output_paths["center_summary"], CENTER_SUMMARY_COLUMNS, center_rows)
    write_protocol_manifest(output_paths["protocol_manifest"], config=config, artifacts=artifacts, limits=limits)
    write_leakage_report(output_paths["leakage_report"], labels=labels, real_rows=real_rows)
    write_decision_report(
        output_paths["decision_report"],
        labels=labels,
        center_rows=center_rows,
        synthetic_rows=synthetic_rows,
        real_rows=real_rows,
    )
    return Z11AuditResult(decision_labels=labels, output_paths=output_paths)


def build_feature_tensor_fingerprint_rows(
    *,
    repo_root: Path,
    artifacts: Sequence[SupportAuditArtifact],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for artifact in artifacts:
        manifest_counts = _manifest_split_counts(artifact.samples_manifest)
        for split, tensor_path in (
            ("train", artifact.train_cache),
            ("val", artifact.val_cache),
            ("test", artifact.test_cache),
        ):
            rows.append(
                fingerprint_tensor_path(
                    tensor_path,
                    split=split,
                    experiment_seed=artifact.experiment_seed,
                    expected_rows=manifest_counts.get(split),
                    repo_root=repo_root,
                )
            )
    return rows


def fingerprint_tensor_path(
    path: Path,
    *,
    split: str,
    experiment_seed: int,
    expected_rows: int | None,
    repo_root: Path,
) -> dict[str, object]:
    row = {
        "experiment_seed": int(experiment_seed),
        "split": split,
        "tensor_path": str(path),
        "exists": str(path.exists()).lower(),
        "num_rows": "",
        "embedding_dim": "",
        "dtype": "",
        "sha256_or_fast_hash": "",
        "feature_cache_hash": "",
        "matches_manifest": "",
        "fingerprint_status": "missing_not_failed" if not path.exists() else "pending",
        "error_message": "",
    }
    if not path.exists():
        return row
    row["sha256_or_fast_hash"] = _fast_file_hash(path)
    try:
        payload = _safe_torch_load(repo_root)(path, map_location="cpu")
        embeddings = payload.get("embeddings") if isinstance(payload, Mapping) else None
        if embeddings is None:
            raise ProtocolError("cache payload has no embeddings tensor")
        shape = tuple(int(v) for v in getattr(embeddings, "shape", ()))
        row["num_rows"] = shape[0] if len(shape) >= 1 else ""
        row["embedding_dim"] = shape[1] if len(shape) >= 2 else ""
        row["dtype"] = str(getattr(embeddings, "dtype", ""))
        row["feature_cache_hash"] = _feature_cache_hash(payload)
        if expected_rows is None:
            row["matches_manifest"] = "unknown_manifest_missing"
        else:
            row["matches_manifest"] = str(int(row["num_rows"]) == int(expected_rows)).lower()
        row["fingerprint_status"] = "ok"
    except Exception as exc:
        row["fingerprint_status"] = "load_failed"
        row["error_message"] = str(exc)
    return row


def build_real_feature_ceiling_rows(
    *,
    config: Z11Config,
    repo_root: Path,
    artifacts: Sequence[SupportAuditArtifact],
    limits: Z11RunLimits = Z11RunLimits(),
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    centers = tuple(str(value) for value in (limits.heldout_centers or config.candidate_centers))
    representations = tuple(str(value) for value in (limits.representations or config.representations))
    for artifact in artifacts:
        if not artifact.train_cache.exists() or not artifact.test_cache.exists():
            continue
        try:
            train_cache = _load_embedding_cache(artifact.train_cache, repo_root=repo_root)
            test_cache = _load_embedding_cache(artifact.test_cache, repo_root=repo_root)
        except Exception:
            continue
        for heldout in centers:
            for train_scope in ("source_only", "target_train_diagnostic"):
                for representation in representations:
                    rows.append(
                        _real_feature_row(
                            config=config,
                            experiment_seed=int(artifact.experiment_seed),
                            heldout_center=str(heldout),
                            train_scope=train_scope,
                            representation=representation,
                            train_cache=train_cache,
                            test_cache=test_cache,
                        )
                    )
    return rows


def build_pca_capacity_rows(
    *,
    config: Z11Config,
    real_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    keyed = {
        (
            int(row["experiment_seed"]),
            str(row["heldout_center"]),
            str(row["representation"]),
            str(row["train_scope"]),
        ): row
        for row in real_rows
        if str(row.get("status")) == "ok"
    }
    out: list[dict[str, object]] = []
    for seed in sorted({int(row["experiment_seed"]) for row in real_rows}):
        for center in sorted({str(row["heldout_center"]) for row in real_rows}):
            base = keyed.get((seed, center, Z11_PREDECLARED_REPRESENTATION, "source_only"))
            if base is None:
                continue
            for candidate in ("PCA128", "PCA256"):
                cand = keyed.get((seed, center, candidate, "source_only"))
                if cand is None:
                    continue
                out.append(
                    {
                        "experiment_seed": seed,
                        "heldout_center": center,
                        "baseline_representation": Z11_PREDECLARED_REPRESENTATION,
                        "candidate_representation": candidate,
                        "baseline_bacc": _float(base.get("bacc")),
                        "candidate_bacc": _float(cand.get("bacc")),
                        "delta_bacc": _float(cand.get("bacc")) - _float(base.get("bacc")),
                        "mean_gain_threshold": config.pca_mean_gain,
                        "weak_center_gain_threshold": config.pca_weak_center_gain,
                        "eligibility": ELIGIBILITY_AUDIT_ONLY,
                        "pca_dim_warning": cand.get("pca_dim_warning", ""),
                    }
                )
    return out


def build_center_summary_rows(
    *,
    config: Z11Config,
    real_rows: Sequence[Mapping[str, object]],
    pca_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    centers = sorted({str(row["heldout_center"]) for row in real_rows})
    out: list[dict[str, object]] = []
    for center in centers:
        source = [
            row for row in real_rows
            if str(row["heldout_center"]) == center
            and str(row["train_scope"]) == "source_only"
            and str(row.get("status")) == "ok"
        ]
        target = [
            row for row in real_rows
            if str(row["heldout_center"]) == center
            and str(row["train_scope"]) == "target_train_diagnostic"
            and str(row.get("status")) == "ok"
        ]
        if not source:
            continue
        by_rep = {
            rep: _nanmean(_float(row.get("bacc")) for row in source if str(row["representation"]) == rep)
            for rep in sorted({str(row["representation"]) for row in source})
        }
        best_rep, best_bacc = max(by_rep.items(), key=lambda item: (_nan_to_low(item[1]), item[0]))
        predeclared = by_rep.get(Z11_PREDECLARED_REPRESENTATION, math.nan)
        target_bacc = _nanmean(_float(row.get("bacc")) for row in target)
        pca_gain = _nanmax(
            _float(row.get("delta_bacc"))
            for row in pca_rows
            if str(row["heldout_center"]) == center
        )
        source_gap = target_bacc - best_bacc if not math.isnan(target_bacc) else math.nan
        out.append(
            {
                "heldout_center": center,
                "best_posthoc_source_only_representation": best_rep,
                "best_posthoc_source_only_bacc": best_bacc,
                "predeclared_or_sourceval_selected_representation": Z11_PREDECLARED_REPRESENTATION,
                "predeclared_or_sourceval_selected_bacc": predeclared,
                "target_train_diagnostic_bacc": target_bacc,
                "target_minus_source_gap": source_gap,
                "pca_capacity_best_gain": pca_gain,
                "weak_center_bottleneck": str(best_bacc < config.feasible_worst_center_bacc).lower(),
                "source_transfer_bottleneck": str(source_gap >= config.source_transfer_gap).lower()
                if not math.isnan(source_gap)
                else "false",
                "eligibility": ELIGIBILITY_AUDIT_ONLY,
            }
        )
    if out:
        out.append(
            {
                "heldout_center": "__mean__",
                "best_posthoc_source_only_representation": "posthoc_best_by_center",
                "best_posthoc_source_only_bacc": _nanmean(_float(row["best_posthoc_source_only_bacc"]) for row in out),
                "predeclared_or_sourceval_selected_representation": Z11_PREDECLARED_REPRESENTATION,
                "predeclared_or_sourceval_selected_bacc": _nanmean(
                    _float(row["predeclared_or_sourceval_selected_bacc"]) for row in out
                ),
                "target_train_diagnostic_bacc": _nanmean(_float(row["target_train_diagnostic_bacc"]) for row in out),
                "target_minus_source_gap": _nanmean(_float(row["target_minus_source_gap"]) for row in out),
                "pca_capacity_best_gain": _nanmax(_float(row["pca_capacity_best_gain"]) for row in out),
                "weak_center_bottleneck": str(
                    any(str(row["weak_center_bottleneck"]) == "true" for row in out)
                ).lower(),
                "source_transfer_bottleneck": str(
                    any(str(row["source_transfer_bottleneck"]) == "true" for row in out)
                ).lower(),
                "eligibility": ELIGIBILITY_AUDIT_ONLY,
            }
        )
    return out


def build_synthetic_preservation_rows(
    *,
    config: Z11Config,
    repo_root: Path,
    center_summary_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    evidence = load_c63_synthetic_evidence(config=config, repo_root=repo_root)
    center_reference = {
        str(row["heldout_center"]): _float(row["best_posthoc_source_only_bacc"])
        for row in center_summary_rows
        if str(row["heldout_center"]) != "__mean__"
    }
    if not center_reference:
        return [
            _synthetic_preservation_row(
                heldout_center="__mean__",
                real_bacc=math.nan,
                oracle_bacc=math.nan,
                geometric_bacc=math.nan,
                chance_bacc=config.chance_bacc,
                matched=0,
                missing=0,
                status="real_feature_evidence_missing",
            )
        ]
    rows: list[dict[str, object]] = []
    matched = 0
    missing = 0
    for center, real_bacc in sorted(center_reference.items()):
        center_evidence = evidence.get(center, {})
        oracle = center_evidence.get("oracle", math.nan)
        geometric = center_evidence.get("geometric", math.nan)
        status = "matched" if not math.isnan(geometric) or not math.isnan(oracle) else "missing"
        if status == "matched":
            matched += 1
        else:
            missing += 1
        rows.append(
            _synthetic_preservation_row(
                heldout_center=center,
                real_bacc=real_bacc,
                oracle_bacc=oracle,
                geometric_bacc=geometric,
                chance_bacc=config.chance_bacc,
                matched=1 if status == "matched" else 0,
                missing=0 if status == "matched" else 1,
                status=status,
            )
        )
    rows.append(
        _synthetic_preservation_row(
            heldout_center="__mean__",
            real_bacc=_nanmean(center_reference.values()),
            oracle_bacc=_nanmean(row["c63_oracle_bacc"] for row in rows),
            geometric_bacc=_nanmean(row["c63_geometric_bacc"] for row in rows),
            chance_bacc=config.chance_bacc,
            matched=matched,
            missing=missing,
            status="matched" if matched else "missing",
        )
    )
    return rows


def load_c63_synthetic_evidence(
    *,
    config: Z11Config,
    repo_root: Path,
) -> dict[str, dict[str, float]]:
    paths: list[Path] = []
    for pattern in config.synthetic_evidence_globs:
        paths.extend(Path(path) for path in glob.glob(str(repo_root / pattern), recursive=True))
    evidence: dict[str, dict[str, list[float]]] = {}
    for path in sorted(set(paths)):
        if not path.exists() or path.suffix.lower() != ".csv":
            continue
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    kind = _c63_evidence_kind(row)
                    if kind is None:
                        continue
                    center = str(row.get("heldout_center") or row.get("center") or row.get("target_center") or "").strip()
                    if not center:
                        continue
                    bacc = _float(row.get("bacc") or row.get("mean_bacc") or row.get("selected_bacc"))
                    if math.isnan(bacc):
                        continue
                    evidence.setdefault(center, {}).setdefault(kind, []).append(bacc)
        except Exception:
            continue
    return {
        center: {kind: _nanmean(values) for kind, values in values_by_kind.items()}
        for center, values_by_kind in evidence.items()
    }


def compute_decision_labels(
    *,
    config: Z11Config,
    fingerprint_rows: Sequence[Mapping[str, object]],
    real_rows: Sequence[Mapping[str, object]],
    pca_rows: Sequence[Mapping[str, object]],
    center_summary_rows: Sequence[Mapping[str, object]],
    synthetic_rows: Sequence[Mapping[str, object]],
) -> list[str]:
    labels: list[str] = []
    if fingerprint_rows and all(str(row.get("fingerprint_status")) == "ok" for row in fingerprint_rows):
        labels.append(LABEL_IDENTITY_PASS)
    else:
        labels.append(LABEL_IDENTITY_INCOMPLETE)

    center_rows = [row for row in center_summary_rows if str(row["heldout_center"]) != "__mean__"]
    mean_row = next((row for row in center_summary_rows if str(row["heldout_center"]) == "__mean__"), None)
    if center_rows and mean_row is not None:
        mean_bacc = _float(mean_row.get("best_posthoc_source_only_bacc"))
        worst_center = _nanmin(_float(row["best_posthoc_source_only_bacc"]) for row in center_rows)
        if mean_bacc >= config.feasible_mean_bacc and worst_center >= config.feasible_worst_center_bacc:
            labels.append(LABEL_FEASIBLE)
        else:
            labels.append(LABEL_NOT_SUPPORTED)
        if any(str(row["source_transfer_bottleneck"]) == "true" for row in center_rows):
            labels.append(LABEL_SOURCE_TRANSFER)
        if any(str(row["weak_center_bottleneck"]) == "true" for row in center_rows):
            labels.append(LABEL_WEAK_CENTER)

    pca_mean_gain = _pca_mean_gain(pca_rows)
    pca_any_weak_gain = any(_float(row.get("delta_bacc")) >= config.pca_weak_center_gain for row in pca_rows)
    if pca_rows:
        labels.append(LABEL_PCA_BOTTLENECK if pca_mean_gain >= config.pca_mean_gain or pca_any_weak_gain else LABEL_PCA_NO_GAIN)

    synthetic_mean = next((row for row in synthetic_rows if str(row["heldout_center"]) == "__mean__"), None)
    if synthetic_mean is None or str(synthetic_mean.get("evidence_status")) != "matched":
        labels.append(LABEL_SYNTHETIC_MISSING)
    else:
        real_bacc = _float(synthetic_mean.get("real_feature_reference_bacc"))
        ratio = _float(synthetic_mean.get("geometric_preservation_ratio"))
        if real_bacc >= config.feasible_mean_bacc and not math.isnan(ratio) and ratio < config.preservation_ratio_min:
            labels.append(LABEL_CVAE_PRESERVATION)
    return labels


def preservation_ratio(real_feature_bacc: float, synthetic_bacc: float, *, chance_bacc: float = 0.50) -> float:
    real_headroom = float(real_feature_bacc) - float(chance_bacc)
    if real_headroom <= 0.0 or math.isnan(real_feature_bacc) or math.isnan(synthetic_bacc):
        return math.nan
    return (float(synthetic_bacc) - float(chance_bacc)) / real_headroom


def pca_dim_warning(min_class_train_n: int, pca_dim: int | None, *, multiplier: int = 3) -> str:
    if pca_dim is None:
        return ""
    if int(min_class_train_n) < int(multiplier) * int(pca_dim):
        return "low_sample_to_dimension_ratio"
    return ""


def eligibility_for_train_scope(train_scope: str) -> str:
    if str(train_scope) == "source_only":
        return ELIGIBILITY_AUDIT_ONLY
    if str(train_scope) == "target_train_diagnostic":
        return ELIGIBILITY_NON_DEPLOYABLE
    raise ProtocolError(f"Unknown train_scope: {train_scope}")


def write_protocol_manifest(
    path: Path,
    *,
    config: Z11Config,
    artifacts: Sequence[SupportAuditArtifact],
    limits: Z11RunLimits,
) -> None:
    payload = {
        "schema_version": "z11_protocol_manifest_v1",
        "experiment_name": Z11_EXPERIMENT_NAME,
        "dataset_name": Z11_DATASET_NAME,
        "audit_only": True,
        "cvae_retraining": "forbidden",
        "source_only_rows": ELIGIBILITY_AUDIT_ONLY,
        "target_train_rows": ELIGIBILITY_NON_DEPLOYABLE,
        "posthoc_best_over_pca_dims": ELIGIBILITY_AUDIT_ONLY,
        "classifier": _classifier_hparams(),
        "candidate_centers": config.candidate_centers,
        "experiment_seeds": limits.experiment_seeds or config.experiment_seeds,
        "support_sizes": config.support_sizes,
        "support_seeds": config.support_seeds,
        "representations": limits.representations or config.representations,
        "support_artifact_run_dirs": [str(artifact.run_dir) for artifact in artifacts],
        "forbidden": [
            "target_eval_labels_for_training",
            "target_eval_tuned_pca_dim_selection_for_deployment",
            "cvae_expert_retraining",
            "routing_tweak",
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_leakage_report(
    path: Path,
    *,
    labels: Sequence[str],
    real_rows: Sequence[Mapping[str, object]],
) -> None:
    violations = []
    for row in real_rows:
        if str(row.get("uses_target_eval_labels_for_training")) == "true":
            violations.append(f"target eval labels used for training in row {row.get('row_id')}")
        if str(row.get("train_scope")) == "source_only" and str(row.get("eligibility")) != ELIGIBILITY_AUDIT_ONLY:
            violations.append(f"source-only row has wrong eligibility in row {row.get('row_id')}")
        if str(row.get("train_scope")) == "target_train_diagnostic" and str(row.get("eligibility")) != ELIGIBILITY_NON_DEPLOYABLE:
            violations.append(f"target-train row has wrong eligibility in row {row.get('row_id')}")
    payload = {
        "schema_version": "z11_leakage_report_v1",
        "status": "PASS" if not violations else "BLOCKED",
        "violations": violations,
        "decision_labels": list(labels),
        "target_eval_labels_for_training": False,
        "target_eval_labels_for_scoring_only": True,
        "cvae_experts_modified": False,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_decision_report(
    path: Path,
    *,
    labels: Sequence[str],
    center_rows: Sequence[Mapping[str, object]],
    synthetic_rows: Sequence[Mapping[str, object]],
    real_rows: Sequence[Mapping[str, object]],
) -> None:
    mean_row = next((row for row in center_rows if str(row["heldout_center"]) == "__mean__"), None)
    synthetic_mean = next((row for row in synthetic_rows if str(row["heldout_center"]) == "__mean__"), None)
    lines = [
        "# Z1.1 Current-Setup Ceiling Audit",
        "",
        "## Decision Labels",
        "",
    ]
    lines.extend(f"- `{label}`" for label in labels)
    lines.extend(["", "## Summary", ""])
    if mean_row is None:
        lines.append("Real-feature ceiling rows were not available; sync support artifacts and rerun.")
    else:
        lines.append(
            "- Best post-hoc source-only mean BACC: "
            f"{_format_float(mean_row.get('best_posthoc_source_only_bacc'))}"
        )
        lines.append(
            "- Predeclared/source-val selected PCA64 mean BACC: "
            f"{_format_float(mean_row.get('predeclared_or_sourceval_selected_bacc'))}"
        )
        lines.append(
            "- Target-train diagnostic mean BACC: "
            f"{_format_float(mean_row.get('target_train_diagnostic_bacc'))}"
        )
    if synthetic_mean is not None:
        lines.append(f"- Synthetic preservation evidence status: `{synthetic_mean.get('evidence_status')}`")
        lines.append(
            "- C6.3 geometric preservation ratio: "
            f"{_format_float(synthetic_mean.get('geometric_preservation_ratio'))}"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "Target-trained rows and post-hoc best-over-PCA rows are audit evidence only. "
            "They are not deployable method claims.",
            "",
            "C6.3 evidence is used only when a synced, predeclared geometric/oracle artifact is present. "
            "Missing synthetic evidence is reported as missing rather than as a failed preservation result.",
            "",
            "## Artifact Counts",
            "",
            f"- Real-feature rows: {len(real_rows)}",
            f"- Center-summary rows: {len(center_rows)}",
            f"- Synthetic-preservation rows: {len(synthetic_rows)}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _real_feature_row(
    *,
    config: Z11Config,
    experiment_seed: int,
    heldout_center: str,
    train_scope: str,
    representation: str,
    train_cache: Mapping[str, Any],
    test_cache: Mapping[str, Any],
) -> dict[str, object]:
    import numpy as np  # type: ignore
    from sklearn.decomposition import PCA  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.metrics import roc_auc_score  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore

    train_meta = tuple(train_cache["metadata"])
    test_meta = tuple(test_cache["metadata"])
    x_train_all = _to_numpy(train_cache["embeddings"])
    x_test_all = _to_numpy(test_cache["embeddings"])
    source_centers = tuple(center for center in config.candidate_centers if str(center) != str(heldout_center))
    if train_scope == "source_only":
        train_centers = source_centers
        train_indices = [idx for idx, row in enumerate(train_meta) if _domain(row) in set(source_centers)]
    elif train_scope == "target_train_diagnostic":
        train_centers = (str(heldout_center),)
        train_indices = [idx for idx, row in enumerate(train_meta) if _domain(row) == str(heldout_center)]
    else:
        raise ProtocolError(f"Unknown train_scope: {train_scope}")
    target_pool = build_target_eval_pool(
        test_metadata=test_meta,
        heldout_center=str(heldout_center),
        support_sizes=config.support_sizes,
        support_seeds=config.support_seeds,
    )
    eval_indices = list(target_pool.eval_indices)
    y_train = np.asarray([_label(train_meta[idx]) for idx in train_indices], dtype=int)
    y_eval = np.asarray([_label(test_meta[idx]) for idx in eval_indices], dtype=int)
    x_train = x_train_all[train_indices]
    x_eval = x_test_all[eval_indices]
    pca_dim = _representation_pca_dim(representation)
    min_class_train_n = _min_class_count(y_train.tolist())
    warning = pca_dim_warning(
        min_class_train_n,
        pca_dim,
        multiplier=config.pca_low_sample_warning_multiplier,
    )
    base = {
        "row_id": _row_id(experiment_seed, heldout_center, train_scope, representation),
        "experiment_seed": int(experiment_seed),
        "heldout_center": heldout_center,
        "train_centers": "|".join(str(v) for v in train_centers),
        "eval_center": heldout_center,
        "eval_split": "test_excluding_configured_support_union",
        "train_scope": train_scope,
        "representation": representation,
        "feature_dim": "",
        "pca_dim": pca_dim if pca_dim is not None else "",
        "effective_pca_dim": "",
        "pca_dim_warning": warning,
        "projection_fit_scope": "classifier_train_rows_only" if pca_dim is not None else "identity",
        "scaler_fit_scope": "classifier_train_rows_only",
        "classifier": "sklearn_logistic_regression",
        "classifier_hparams": json.dumps(_classifier_hparams(), sort_keys=True),
        "uses_target_train_labels": str(train_scope == "target_train_diagnostic").lower(),
        "uses_target_support_labels": "false",
        "uses_target_eval_labels_for_training": "false",
        "uses_target_eval_labels_for_scoring": "true",
        "eligibility": eligibility_for_train_scope(train_scope),
        "bacc": math.nan,
        "macro_f1": math.nan,
        "auroc_if_valid": math.nan,
        "n_train": len(train_indices),
        "n_eval": len(eval_indices),
        "min_class_train_n": min_class_train_n,
        "class_balance_train": json.dumps(_class_balance(y_train.tolist()), sort_keys=True),
        "class_balance_eval": json.dumps(_class_balance(y_eval.tolist()), sort_keys=True),
        "status": "ok",
        "error_message": "",
    }
    try:
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
            C=1.0,
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
        base.update(
            {
                "feature_dim": int(x_train_rep.shape[1]),
                "effective_pca_dim": effective_dim if effective_dim is not None else "",
                "bacc": balanced_accuracy(y_eval.tolist(), pred.tolist()),
                "macro_f1": macro_f1(y_eval.tolist(), pred.tolist()),
                "auroc_if_valid": auroc,
            }
        )
    except Exception as exc:
        base["status"] = "failed"
        base["error_message"] = str(exc)
    return base


def _project_representation(
    x_train: Any,
    x_eval: Any,
    *,
    representation: str,
    requested_pca_dim: int | None,
    pca_cls: Any,
) -> tuple[Any, Any, int | None]:
    rep = str(representation)
    if rep == "raw":
        return x_train, x_eval, None
    if requested_pca_dim is None:
        raise ProtocolError(f"Representation has no PCA dimension: {representation}")
    max_components = max(1, min(int(x_train.shape[0]), int(x_train.shape[1])))
    effective = min(int(requested_pca_dim), max_components)
    pca = pca_cls(n_components=effective, random_state=0)
    train_pca = pca.fit_transform(x_train)
    eval_pca = pca.transform(x_eval)
    if rep.endswith("_reconstruction"):
        return pca.inverse_transform(train_pca), pca.inverse_transform(eval_pca), effective
    return train_pca, eval_pca, effective


def _support_artifact(seed: int, run_dir: Path, support_selection_path: Path) -> SupportAuditArtifact:
    return SupportAuditArtifact(
        experiment_seed=int(seed),
        run_dir=run_dir,
        train_cache=run_dir / "embeddings" / "train.pt",
        val_cache=run_dir / "embeddings" / "val.pt",
        test_cache=run_dir / "embeddings" / "test.pt",
        samples_manifest=run_dir / "manifests" / "samples.csv",
        config_resolved=run_dir / "config_resolved.yaml",
        support_selection_path=support_selection_path,
    )


def _z11_config_from_mapping(config: Mapping[str, Any]) -> Z11Config:
    defaults = default_z11_config()
    dataset = _mapping(_mapping(config.get("datasets"), "datasets").get("camelyon17"), "datasets.camelyon17")
    reps = _mapping(config.get("representations"), "representations")
    inputs = _mapping(config.get("inputs"), "inputs")
    artifacts = _mapping(config.get("artifacts"), "artifacts")
    decision = _mapping(config.get("decision_rule"), "decision_rule")
    return Z11Config(
        candidate_centers=tuple(str(v) for v in dataset.get("candidate_centers", defaults.candidate_centers)),
        experiment_seeds=tuple(int(v) for v in dataset.get("experiment_seeds", defaults.experiment_seeds)),
        support_sizes=tuple(int(v) for v in dataset.get("support_sizes", defaults.support_sizes)),
        support_seeds=tuple(int(v) for v in dataset.get("support_seeds", defaults.support_seeds)),
        representations=tuple(str(v) for v in reps.get("requested", defaults.representations)),
        support_selection_glob=str(inputs.get("support_selection_glob", defaults.support_selection_glob)),
        expected_support_run_root=str(inputs.get("expected_support_run_root", defaults.expected_support_run_root)),
        expected_support_run_dir_pattern=str(
            inputs.get("expected_support_run_dir_pattern", defaults.expected_support_run_dir_pattern)
        ),
        synthetic_evidence_globs=tuple(str(v) for v in inputs.get("synthetic_evidence_globs", defaults.synthetic_evidence_globs)),
        artifacts_root=str(artifacts.get("root", defaults.artifacts_root)),
        chance_bacc=float(decision.get("chance_bacc", defaults.chance_bacc)),
        feasible_mean_bacc=float(decision.get("feasible_mean_bacc", defaults.feasible_mean_bacc)),
        feasible_worst_center_bacc=float(
            decision.get("feasible_worst_center_bacc", defaults.feasible_worst_center_bacc)
        ),
        source_transfer_gap=float(decision.get("source_transfer_gap", defaults.source_transfer_gap)),
        pca_mean_gain=float(decision.get("pca_mean_gain", defaults.pca_mean_gain)),
        pca_weak_center_gain=float(decision.get("pca_weak_center_gain", defaults.pca_weak_center_gain)),
        preservation_ratio_min=float(decision.get("preservation_ratio_min", defaults.preservation_ratio_min)),
        pca_low_sample_warning_multiplier=int(
            reps.get("pca_low_sample_warning_multiplier", defaults.pca_low_sample_warning_multiplier)
        ),
    )


def _safe_torch_load(repo_root: Path) -> Any:
    cvae_testing_root = repo_root / "cvae_testing"
    if str(cvae_testing_root) not in sys.path:
        sys.path.insert(0, str(cvae_testing_root))
    from src.torch_utils import safe_torch_load  # type: ignore

    return safe_torch_load


def _load_embedding_cache(path: Path, *, repo_root: Path) -> Mapping[str, Any]:
    payload = _safe_torch_load(repo_root)(path, map_location="cpu")
    if not isinstance(payload, Mapping) or "embeddings" not in payload or "metadata" not in payload:
        raise ProtocolError(f"Embedding cache must contain embeddings and metadata: {path}")
    return payload


def _manifest_split_counts(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    counts: dict[str, int] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                split = str(row.get("split", "")).strip().lower()
                if split:
                    counts[split] = counts.get(split, 0) + 1
    except Exception:
        return {}
    return counts


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{name} must be a mapping")
    return value


def _seed_from_path(path: Path) -> int | None:
    match = re.search(r"seed(\d+)", str(path))
    return int(match.group(1)) if match else None


def _seed_from_config_resolved(path: Path) -> int | None:
    if not path.exists():
        return None
    match = re.search(r"seed:\s*(\d+)", path.read_text(encoding="utf-8", errors="ignore"))
    return int(match.group(1)) if match else None


def _fast_file_hash(path: Path) -> str:
    h = hashlib.sha256()
    stat = path.stat()
    h.update(str(stat.st_size).encode("utf-8"))
    h.update(str(int(stat.st_mtime_ns)).encode("utf-8"))
    with path.open("rb") as handle:
        h.update(handle.read(1024 * 1024))
        if stat.st_size > 1024 * 1024:
            handle.seek(max(0, stat.st_size - (1024 * 1024)))
            h.update(handle.read(1024 * 1024))
    return "fast_sha256:" + h.hexdigest()


def _feature_cache_hash(payload: Mapping[str, Any]) -> str:
    feature = payload.get("feature_extractor", {}) if isinstance(payload, Mapping) else {}
    metadata = payload.get("metadata", ()) if isinstance(payload, Mapping) else ()
    digest_payload = {
        "feature_extractor": feature,
        "metadata_count": len(metadata) if hasattr(metadata, "__len__") else None,
    }
    return hashlib.sha256(json.dumps(digest_payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _domain(row: Mapping[str, Any]) -> str:
    for key in ("magnification", "center", "domain", "hospital", "site"):
        if key in row and str(row[key]).strip() != "":
            return str(int(float(row[key])))
    raise ProtocolError(f"Could not resolve domain from metadata row keys={sorted(row.keys())}")


def _label(row: Mapping[str, Any]) -> int:
    for key in ("label", "y", "target"):
        if key in row and str(row[key]).strip() != "":
            return int(float(row[key]))
    raise ProtocolError(f"Could not resolve label from metadata row keys={sorted(row.keys())}")


def _to_numpy(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return value


def _class_balance(values: Sequence[int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(int(value))
        out[key] = out.get(key, 0) + 1
    return out


def _min_class_count(values: Sequence[int]) -> int:
    counts = _class_balance(values)
    return min(counts.values()) if counts else 0


def _classifier_hparams() -> dict[str, object]:
    return {
        "solver": "lbfgs",
        "C": 1.0,
        "max_iter": 2000,
        "class_weight": None,
        "random_state": "experiment_seed",
    }


def _representation_pca_dim(representation: str) -> int | None:
    rep = str(representation)
    if rep == "raw":
        return None
    match = re.search(r"PCA(\d+)", rep)
    if not match:
        raise ProtocolError(f"Unknown Z1.1 representation: {representation}")
    return int(match.group(1))


def _row_id(seed: int, center: str, train_scope: str, representation: str) -> str:
    return f"z11_seed{int(seed)}_center{center}_{train_scope}_{representation}"


def _c63_evidence_kind(row: Mapping[str, str]) -> str | None:
    text = " ".join(str(value).lower() for value in row.values())
    method = str(row.get("method", "")).lower()
    if "oracle" in text and ("c6.3" in text or "c63" in text or "downstream_oracle" in method):
        return "oracle"
    if ("c6.3" in text or "c63" in text) and ("geometric" in text or "log" in text):
        return "geometric"
    return None


def _synthetic_preservation_row(
    *,
    heldout_center: str,
    real_bacc: float,
    oracle_bacc: float,
    geometric_bacc: float,
    chance_bacc: float,
    matched: int,
    missing: int,
    status: str,
) -> dict[str, object]:
    return {
        "heldout_center": heldout_center,
        "real_feature_reference_bacc": real_bacc,
        "c63_oracle_bacc": oracle_bacc,
        "c63_geometric_bacc": geometric_bacc,
        "chance_bacc": chance_bacc,
        "oracle_preservation_ratio": preservation_ratio(real_bacc, oracle_bacc, chance_bacc=chance_bacc),
        "geometric_preservation_ratio": preservation_ratio(real_bacc, geometric_bacc, chance_bacc=chance_bacc),
        "real_minus_c63_gap": real_bacc - geometric_bacc
        if not math.isnan(real_bacc) and not math.isnan(geometric_bacc)
        else math.nan,
        "matched_cell_count": int(matched),
        "missing_cell_count": int(missing),
        "evidence_status": status,
    }


def _pca_mean_gain(pca_rows: Sequence[Mapping[str, object]]) -> float:
    if not pca_rows:
        return math.nan
    by_candidate = sorted({str(row["candidate_representation"]) for row in pca_rows})
    gains = [
        _nanmean(_float(row["delta_bacc"]) for row in pca_rows if str(row["candidate_representation"]) == candidate)
        for candidate in by_candidate
    ]
    return _nanmax(gains)


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _nanmean(values: Iterable[object]) -> float:
    vals = [_float(value) for value in values]
    vals = [value for value in vals if not math.isnan(value)]
    return mean(vals) if vals else math.nan


def _nanmax(values: Iterable[object]) -> float:
    vals = [_float(value) for value in values]
    vals = [value for value in vals if not math.isnan(value)]
    return max(vals) if vals else math.nan


def _nanmin(values: Iterable[object]) -> float:
    vals = [_float(value) for value in values]
    vals = [value for value in vals if not math.isnan(value)]
    return min(vals) if vals else math.nan


def _nan_to_low(value: float) -> float:
    return -1.0 if math.isnan(value) else value


def _format_float(value: object) -> str:
    parsed = _float(value)
    return "nan" if math.isnan(parsed) else f"{parsed:.4f}"
