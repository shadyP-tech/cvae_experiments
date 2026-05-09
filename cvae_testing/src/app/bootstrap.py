from __future__ import annotations

from datetime import datetime
import hashlib
import json
import platform
from pathlib import Path
import random
import sys
from typing import Any, Dict

import numpy as np
import torch
import yaml

from src.engine.contracts import RunContext


def resolve_config_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path

    parts = path.parts
    if len(parts) >= 2 and parts[0] == "codebase" and parts[1] == "cvae_testing":
        return project_root.joinpath(*parts[2:])
    return project_root / path


def build_run_context(project_root: Path, cfg: Dict[str, Any], run_id_override: str | None) -> RunContext:
    exp_cfg = cfg.get("experiment", {})
    dataset_name = str(exp_cfg.get("dataset_name", "breakhis"))
    experiment_name = str(exp_cfg.get("name", "learned_utility_routing_v1"))
    seed = int(cfg["seed"])

    output_cfg = cfg.get("output", {})
    output_root = resolve_config_path(project_root, str(output_cfg.get("root", "outputs")))
    run_id = run_id_override or datetime.now().strftime(f"%Y-%m-%d_%H%M_seed{seed}")

    run_root = output_root / dataset_name / experiment_name / run_id
    reports_dir = run_root / "reports"
    embeddings_dir = run_root / "embeddings"
    checkpoints_dir = run_root / "checkpoints"
    manifests_dir = run_root / "manifests"
    plots_dir = run_root / "plots"

    for p in [run_root, reports_dir, embeddings_dir, checkpoints_dir, manifests_dir, plots_dir]:
        p.mkdir(parents=True, exist_ok=True)

    latest_file = output_root / dataset_name / experiment_name / "latest.txt"
    latest_file.parent.mkdir(parents=True, exist_ok=True)
    latest_file.write_text(run_id, encoding="utf-8")

    return RunContext(
        output_root=output_root,
        run_root=run_root,
        reports_dir=reports_dir,
        embeddings_dir=embeddings_dir,
        checkpoints_dir=checkpoints_dir,
        manifests_dir=manifests_dir,
        plots_dir=plots_dir,
        latest_file=latest_file,
    )


def set_global_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_config_hash(cfg: Dict[str, Any]) -> str:
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_environment_snapshot(seed: int) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "seed": int(seed),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "mps_available": bool(torch.backends.mps.is_available()),
    }
    if torch.cuda.is_available():
        snapshot["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    else:
        snapshot["cuda_devices"] = []
    return snapshot


def write_split_manifest(records: list[Any], out_path: Path) -> None:
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


def write_run_metadata(cfg: Dict[str, Any], run_ctx: RunContext) -> None:
    with (run_ctx.run_root / "config_resolved.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with (run_ctx.reports_dir / "config_hash.txt").open("w", encoding="utf-8") as f:
        f.write(compute_config_hash(cfg) + "\n")
    with (run_ctx.reports_dir / "environment_snapshot.json").open("w", encoding="utf-8") as f:
        json.dump(build_environment_snapshot(seed=int(cfg["seed"])), f, indent=2)
