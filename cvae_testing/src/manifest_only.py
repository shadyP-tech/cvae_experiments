from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import platform
from pathlib import Path
import random
import sys
from typing import Any, Dict


def load_config(path: Path) -> Dict[str, Any]:
    from src.config.load_config import load_config as _load_config

    return _load_config(path)


def prepare_dataset_records(project_root: Path, cfg: Dict[str, Any]) -> tuple[list[Any], Dict[str, Any]]:
    from src.data.registry import prepare_dataset_records as _prepare_dataset_records

    return _prepare_dataset_records(project_root, cfg)


def write_manifest(records: list[Any], out_path: Path) -> None:
    from src.data.datasets.breakhis import write_manifest as _write_manifest

    _write_manifest(records, out_path)


@dataclass(frozen=True)
class ManifestOnlyResult:
    run_root: Path
    samples_manifest: Path
    split_manifest: Path
    leakage_report: Path
    manifest_only_report: Path
    config_resolved: Path
    n_records: int
    split_counts: Dict[str, int]


@dataclass(frozen=True)
class _ManifestRunContext:
    run_root: Path
    reports_dir: Path
    manifests_dir: Path
    latest_file: Path


def resolve_config_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    parts = path.parts
    if len(parts) >= 2 and parts[0] == "codebase" and parts[1] == "cvae_testing":
        return project_root.joinpath(*parts[2:])
    return project_root / path


def _set_lightweight_determinism(seed: int) -> None:
    random.seed(seed)


def _compute_config_hash(cfg: Dict[str, Any]) -> str:
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_environment_snapshot(seed: int) -> Dict[str, Any]:
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "seed": int(seed),
        "manifest_only": True,
    }


def _build_manifest_run_context(project_root: Path, cfg: Dict[str, Any], run_id: str) -> _ManifestRunContext:
    run_root = _expected_run_root(project_root, cfg, run_id)
    reports_dir = run_root / "reports"
    manifests_dir = run_root / "manifests"
    for path in [run_root, reports_dir, manifests_dir]:
        path.mkdir(parents=True, exist_ok=True)

    latest_file = run_root.parent / "latest.txt"
    latest_file.parent.mkdir(parents=True, exist_ok=True)
    latest_file.write_text(run_id, encoding="utf-8")
    return _ManifestRunContext(
        run_root=run_root,
        reports_dir=reports_dir,
        manifests_dir=manifests_dir,
        latest_file=latest_file,
    )


def _write_run_metadata(cfg: Dict[str, Any], run_ctx: _ManifestRunContext) -> None:
    import yaml

    with (run_ctx.run_root / "config_resolved.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with (run_ctx.reports_dir / "config_hash.txt").open("w", encoding="utf-8") as f:
        f.write(_compute_config_hash(cfg) + "\n")
    with (run_ctx.reports_dir / "environment_snapshot.json").open("w", encoding="utf-8") as f:
        json.dump(_build_environment_snapshot(seed=int(cfg["seed"])), f, indent=2)


def _write_split_manifest(records: list[Any], out_path: Path) -> None:
    by_split: Dict[str, int] = {}
    by_domain_split: Dict[str, Dict[str, int]] = {}
    by_label_split: Dict[str, Dict[str, int]] = {}

    for rec in records:
        split = str(getattr(rec, "split", "")) or "unknown"
        domain = str(getattr(rec, "domain_name", "unknown"))
        label_name = str(getattr(rec, "label_name", "unknown"))

        by_split[split] = by_split.get(split, 0) + 1
        domain_map = by_domain_split.setdefault(domain, {})
        domain_map[split] = domain_map.get(split, 0) + 1
        label_map = by_label_split.setdefault(label_name, {})
        label_map[split] = label_map.get(split, 0) + 1

    payload = {
        "n_total": len(records),
        "by_split": by_split,
        "by_domain_split": by_domain_split,
        "by_label_split": by_label_split,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _expected_run_root(project_root: Path, cfg: Dict[str, Any], run_id: str) -> Path:
    exp_cfg = cfg.get("experiment", {})
    dataset_name = str(exp_cfg.get("dataset_name", "breakhis"))
    experiment_name = str(exp_cfg.get("name", "learned_utility_routing_v1"))
    output_cfg = cfg.get("output", {})
    output_root = resolve_config_path(project_root, str(output_cfg.get("root", "outputs")))
    return output_root / dataset_name / experiment_name / run_id


def _guard_existing_manifest_outputs(run_root: Path, *, overwrite: bool) -> None:
    guarded = [
        run_root / "config_resolved.yaml",
        run_root / "manifests" / "samples.csv",
        run_root / "reports" / "split_manifest.json",
        run_root / "reports" / "leakage_report.json",
        run_root / "reports" / "manifest_only_report.json",
    ]
    existing = [path for path in guarded if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing manifest-only outputs. "
            "Pass --overwrite-manifests if this rerun is intentional.\n"
            f"{formatted}"
        )


def _validate_required_domain_split_coverage(records: list[Any], cfg: Dict[str, Any]) -> Dict[str, int]:
    requested_domains = sorted({int(d) for d in cfg.get("data", {}).get("magnifications", [])})
    split_counts_by_domain = {
        "train": {d: 0 for d in requested_domains},
        "val": {d: 0 for d in requested_domains},
        "test": {d: 0 for d in requested_domains},
    }
    split_counts = {"train": 0, "val": 0, "test": 0}
    for rec in records:
        rec_domain = int(getattr(rec, "magnification"))
        rec_split = str(getattr(rec, "split", ""))
        if rec_split in split_counts:
            split_counts[rec_split] += 1
        if rec_domain in split_counts_by_domain.get(rec_split, {}):
            split_counts_by_domain[rec_split][rec_domain] += 1

    missing_train = [d for d in requested_domains if split_counts_by_domain["train"][d] == 0]
    missing_val = [d for d in requested_domains if split_counts_by_domain["val"][d] == 0]
    missing_test = [d for d in requested_domains if split_counts_by_domain["test"][d] == 0]
    if missing_train or missing_val or missing_test:
        raise RuntimeError(
            "Configured domains are not fully represented across required splits. "
            f"Missing train domains: {missing_train}; "
            f"missing val domains: {missing_val}; "
            f"missing test domains: {missing_test}."
        )
    return split_counts


def materialize_manifest_only_run(
    *,
    project_root: Path,
    config_path: Path,
    seed: int,
    run_id: str,
    overwrite: bool = False,
    dry_run: bool = False,
) -> ManifestOnlyResult:
    cfg = load_config(config_path)
    cfg["seed"] = int(seed)

    mode = str(cfg.get("experiment", {}).get("mode", "")).strip()
    if not mode:
        raise ValueError("experiment.mode is required; implicit legacy_routed_cvae defaults are quarantined")

    _set_lightweight_determinism(seed=int(cfg["seed"]))
    records, leakage = prepare_dataset_records(project_root, cfg)
    if not records:
        root = resolve_config_path(project_root, str(cfg["data"]["root"]))
        exts = ", ".join(cfg["data"]["image_extensions"])
        raise RuntimeError(
            "No dataset images were found for processing. "
            f"Checked root: {root}. Expected extensions: {exts}. "
            "Verify dataset files are present under the configured data root."
        )
    split_counts = _validate_required_domain_split_coverage(records, cfg)

    expected_run_root = _expected_run_root(project_root, cfg, run_id)
    if dry_run:
        return ManifestOnlyResult(
            run_root=expected_run_root,
            samples_manifest=expected_run_root / "manifests" / "samples.csv",
            split_manifest=expected_run_root / "reports" / "split_manifest.json",
            leakage_report=expected_run_root / "reports" / "leakage_report.json",
            manifest_only_report=expected_run_root / "reports" / "manifest_only_report.json",
            config_resolved=expected_run_root / "config_resolved.yaml",
            n_records=len(records),
            split_counts=split_counts,
        )

    _guard_existing_manifest_outputs(expected_run_root, overwrite=overwrite)
    run_ctx = _build_manifest_run_context(project_root, cfg, run_id)

    _write_run_metadata(cfg, run_ctx)
    write_manifest(records, run_ctx.manifests_dir / "samples.csv")
    _write_split_manifest(records, run_ctx.reports_dir / "split_manifest.json")
    with (run_ctx.reports_dir / "leakage_report.json").open("w", encoding="utf-8") as f:
        json.dump(leakage, f, indent=2)

    manifest_only_report = {
        "schema_version": "cvae_testing_manifest_only_run_v1",
        "manifest_only": True,
        "seed": int(cfg["seed"]),
        "run_id": str(run_id),
        "config_path": str(config_path),
        "run_root": str(run_ctx.run_root),
        "n_records": int(len(records)),
        "split_counts": split_counts,
        "training_executed": False,
        "embedding_extraction_executed": False,
        "routing_or_selection_executed": False,
        "checkpoints_written_by_manifest_only_builder": False,
        "target_eval_metrics_used": False,
        "target_labels_used_for_selection": False,
        "intended_use": "seeded split/sample manifest materialization for downstream feature-cache building",
        "samples_manifest": str(run_ctx.manifests_dir / "samples.csv"),
        "split_manifest": str(run_ctx.reports_dir / "split_manifest.json"),
        "leakage_report": str(run_ctx.reports_dir / "leakage_report.json"),
    }
    report_path = run_ctx.reports_dir / "manifest_only_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_only_report, f, indent=2)

    return ManifestOnlyResult(
        run_root=run_ctx.run_root,
        samples_manifest=run_ctx.manifests_dir / "samples.csv",
        split_manifest=run_ctx.reports_dir / "split_manifest.json",
        leakage_report=run_ctx.reports_dir / "leakage_report.json",
        manifest_only_report=report_path,
        config_resolved=run_ctx.run_root / "config_resolved.yaml",
        n_records=len(records),
        split_counts=split_counts,
    )
