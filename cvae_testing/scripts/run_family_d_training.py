"""Train Family D discriminative label-conditioned Camelyon17 CVAE experts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import torch
import yaml


CVAE_TESTING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CVAE_TESTING_ROOT))

from src.train.train_experts import train_domain_experts  # noqa: E402


FAMILY_D_TRAINING_NAME = "family_d_discriminative_label_conditioned_cvae_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Family D discriminative label-conditioned CVAE experts.")
    parser.add_argument("--config", required=True, help="Path to Family D training YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and report expected outputs.")
    parser.add_argument(
        "--require-heavy-artifacts",
        action="store_true",
        help="During --dry-run, require train/val embedding caches to exist.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override configured epochs for smoke runs.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path.cwd()
    cfg = load_config(repo_root / args.config if not Path(args.config).is_absolute() else Path(args.config))
    if args.epochs is not None:
        cfg["training"]["epochs"] = int(args.epochs)
    preflight_result = preflight(cfg, repo_root=repo_root, require_heavy_artifacts=bool(args.require_heavy_artifacts))
    if args.dry_run:
        print(json.dumps({"status": "dry_run_passed", **preflight_result}, indent=2, sort_keys=True))
        return

    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    output_root = resolve(repo_root, cfg["outputs"]["run_root"])
    checkpoints_dir = output_root / "checkpoints"
    reports_dir = output_root / "reports"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    experts = train_domain_experts(
        train_cache=resolve(repo_root, cfg["inputs"]["train_cache"]),
        val_cache=resolve(repo_root, cfg["inputs"]["val_cache"]),
        out_dir=checkpoints_dir,
        domains=[int(v) for v in cfg["data"]["centers"]],
        hidden_dim=int(model_cfg["hidden_dim"]),
        latent_dim=int(model_cfg["latent_dim"]),
        lr=float(training_cfg["learning_rate"]),
        epochs=int(training_cfg["epochs"]),
        patience=int(training_cfg["patience"]),
        batch_size=int(training_cfg["batch_size"]),
        conditioning_cfg=model_cfg.get("conditioning", {}),
        configured_domains=cfg["data"]["centers"],
        metadata_constraint_cfg=model_cfg.get("metadata_constraint", {}),
        label_conditioning_cfg=model_cfg.get("label_conditioning", {}),
        label_utility_cfg=model_cfg.get("label_utility", {}),
    )

    summary = {
        "status": "complete",
        "experiment": FAMILY_D_TRAINING_NAME,
        "run_root": str(output_root),
        "checkpoints_dir": str(checkpoints_dir),
        "reports_dir": str(reports_dir),
        "n_experts": len(experts),
        "experts": experts,
        "required_artifacts": [
            "family_d_checkpoint_provenance.csv",
            "family_d_training_history.csv",
            "family_d_training_protocol_audit.csv",
            "family_d_source_val_diagnostics.csv",
        ],
    }
    (reports_dir / "family_d_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    required = (
        f"name: {FAMILY_D_TRAINING_NAME}",
        "expert_family: family_d_discriminative_label_conditioned_v1",
        "discriminative_training_enabled: true",
        "early_stopping_metric: source_val_total_loss",
        "lambda_prior_cls: 0.50",
        "label_values: [0, 1]",
    )
    missing = [value for value in required if value not in text]
    if missing:
        raise SystemExit(f"Family D training config missing required fields: {missing}")
    cfg = yaml.safe_load(text) or {}
    if cfg.get("experiment", {}).get("name") != FAMILY_D_TRAINING_NAME:
        raise SystemExit(f"experiment.name must be {FAMILY_D_TRAINING_NAME}")
    if cfg.get("experiment", {}).get("dataset") != "camelyon17":
        raise SystemExit("Family D v1 is Camelyon17-only.")
    label_cfg = cfg.get("model", {}).get("label_conditioning", {})
    if label_cfg.get("label_values") != [0, 1] or label_cfg.get("enabled") is not True:
        raise SystemExit("Family D requires model.label_conditioning.enabled=true and label_values [0, 1].")
    utility_cfg = cfg.get("model", {}).get("label_utility", {})
    if utility_cfg.get("enabled") is not True:
        raise SystemExit("Family D requires model.label_utility.enabled=true.")
    if utility_cfg.get("early_stopping_metric") != "source_val_total_loss":
        raise SystemExit("Family D early_stopping_metric must be source_val_total_loss.")
    return cfg


def preflight(cfg: dict, *, repo_root: Path, require_heavy_artifacts: bool) -> dict[str, object]:
    output_root = resolve(repo_root, cfg["outputs"]["run_root"])
    heavy = [resolve(repo_root, cfg["inputs"]["train_cache"]), resolve(repo_root, cfg["inputs"]["val_cache"])]
    missing = [str(path) for path in heavy if not path.exists()]
    if require_heavy_artifacts and missing:
        raise SystemExit(f"Missing Family D training heavyweight artifacts: {missing}")
    return {
        "run_root": str(output_root),
        "checkpoints_dir": str(output_root / "checkpoints"),
        "reports_dir": str(output_root / "reports"),
        "n_centers": len(cfg["data"]["centers"]),
        "heavy_artifacts_available": int(not missing),
        "missing_heavy_artifacts": missing,
    }


def resolve(repo_root: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else repo_root / path


if __name__ == "__main__":
    main()
