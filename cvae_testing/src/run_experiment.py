from __future__ import annotations

import json
from pathlib import Path
import time

from src.app.bootstrap import (
    build_run_context,
    resolve_config_path,
    set_global_determinism,
    write_run_metadata,
    write_split_manifest,
)
from src.app.cli import parse_cli_args, resolve_resume_run_id
from src.app.progress import ProgressTracker
from src.app.tracking import create_tracking_client
from src.config.load_config import load_config
from src.data.datasets.breakhis import write_manifest
from src.data.registry import prepare_dataset_records
from src.experiments.registry import create_experiment
from src.features.extract_embeddings import extract_and_cache_embeddings, validate_embedding_cache
from src.train.train_global import train_global_model


def main() -> None:
    args = parse_cli_args()

    project_root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else (project_root / args.config)

    cfg = load_config(config_path)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)

    mode = str(cfg.get("experiment", {}).get("mode", "legacy_routed_cvae"))
    experiment = create_experiment(mode)

    resolved_run_id = resolve_resume_run_id(project_root, cfg, run_id=args.run_id, resume=args.resume)
    set_global_determinism(seed=int(cfg["seed"]))

    run_ctx = build_run_context(project_root, cfg, run_id_override=resolved_run_id)
    resume_checkpoints_dir = run_ctx.checkpoints_dir if args.resume else None
    progress = ProgressTracker(total=experiment.estimate_total_steps(cfg), desc="Experiment")
    tracker = create_tracking_client(cfg, run_ctx, mode=mode, resume=args.resume)
    run_status = "success"
    failure_message: str | None = None
    stage = 0
    run_started = time.perf_counter()
    stage_timings: list[dict[str, object]] = []

    def log_stage(stage_name: str, **extra: object) -> None:
        nonlocal stage
        stage += 1
        metrics = {
            "stage/index": stage,
            "stage/name": stage_name,
        }
        metrics.update(extra)
        tracker.log_metrics(metrics, step=stage)

    def complete_stage(stage_name: str, stage_started: float, **extra: object) -> None:
        duration_sec = time.perf_counter() - stage_started
        payload: dict[str, object] = {"stage/duration_sec": duration_sec}
        payload.update(extra)
        log_stage(stage_name, **payload)
        stage_timings.append(
            {
                "index": stage,
                "name": stage_name,
                "duration_sec": round(duration_sec, 6),
                **{k: str(v) for k, v in extra.items()},
            }
        )

    try:
        stage_started = time.perf_counter()
        write_run_metadata(cfg, run_ctx)
        tracker.log_artifact(run_ctx.run_root / "config_resolved.yaml", artifact_name="config_resolved", artifact_type="config")
        tracker.log_artifact(run_ctx.reports_dir / "config_hash.txt", artifact_name="config_hash", artifact_type="report")
        tracker.log_artifact(
            run_ctx.reports_dir / "environment_snapshot.json",
            artifact_name="environment_snapshot",
            artifact_type="report",
        )
        progress.advance("config resolved")
        complete_stage("config_resolved", stage_started, seed=int(cfg["seed"]))

        stage_started = time.perf_counter()
        records, leakage = prepare_dataset_records(project_root, cfg)
        progress.advance("data prepared")
        complete_stage("data_prepared", stage_started, n_records=len(records))

        if not records:
            root = resolve_config_path(project_root, str(cfg["data"]["root"]))
            exts = ", ".join(cfg["data"]["image_extensions"])
            raise RuntimeError(
                "No dataset images were found for processing. "
                f"Checked root: {root}. Expected extensions: {exts}. "
                "Verify dataset files are present under the configured data root."
            )

        requested_domains = sorted({int(d) for d in cfg.get("data", {}).get("magnifications", [])})
        split_counts = {
            "train": {d: 0 for d in requested_domains},
            "val": {d: 0 for d in requested_domains},
            "test": {d: 0 for d in requested_domains},
        }
        for rec in records:
            rec_domain = int(getattr(rec, "magnification"))
            rec_split = str(getattr(rec, "split", ""))
            if rec_domain in split_counts.get(rec_split, {}):
                split_counts[rec_split][rec_domain] += 1

        missing_train = [d for d in requested_domains if split_counts["train"][d] == 0]
        missing_val = [d for d in requested_domains if split_counts["val"][d] == 0]
        missing_test = [d for d in requested_domains if split_counts["test"][d] == 0]
        if missing_train or missing_val or missing_test:
            raise RuntimeError(
                "Configured domains are not fully represented across required splits. "
                f"Missing train domains: {missing_train}; "
                f"missing val domains: {missing_val}; "
                f"missing test domains: {missing_test}."
            )

        stage_started = time.perf_counter()
        write_manifest(records, run_ctx.manifests_dir / "samples.csv")
        write_split_manifest(records, run_ctx.reports_dir / "split_manifest.json")
        with (run_ctx.reports_dir / "leakage_report.json").open("w", encoding="utf-8") as f:
            json.dump(leakage, f, indent=2)
        tracker.log_artifact(run_ctx.manifests_dir / "samples.csv", artifact_name="samples_manifest", artifact_type="manifest")
        tracker.log_artifact(run_ctx.reports_dir / "split_manifest.json", artifact_name="split_manifest", artifact_type="report")
        tracker.log_artifact(run_ctx.reports_dir / "leakage_report.json", artifact_name="leakage_report", artifact_type="report")
        progress.advance("manifest, split summary, and leakage report written")
        complete_stage("manifests_written", stage_started)

        stage_started = time.perf_counter()
        expected_dim = cfg.get("features", {}).get("embedding_dim")
        cache_paths, extraction_info = extract_and_cache_embeddings(
            records=records,
            cache_dir=run_ctx.embeddings_dir,
            image_size=int(cfg["features"]["image_size"]),
            batch_size=int(cfg.get("features", {}).get("extraction_batch_size", cfg["training"]["batch_size"])),
            backbone_type=str(cfg.get("features", {}).get("backbone_type", "resnet18")),
            expected_dim=int(expected_dim) if expected_dim is not None else None,
        )
        resolved_dim = int(extraction_info["resolved_embedding_dim"])
        cfg.setdefault("features", {})["embedding_dim"] = resolved_dim
        cache_report = validate_embedding_cache(
            cache_paths,
            expected_dim=resolved_dim,
            expected_backbone_type=str(extraction_info["backbone_type"]),
        )
        cache_report["feature_extractor"] = extraction_info
        with (run_ctx.reports_dir / "cache_report.json").open("w", encoding="utf-8") as f:
            json.dump(cache_report, f, indent=2)
        tracker.log_artifact(run_ctx.reports_dir / "cache_report.json", artifact_name="cache_report", artifact_type="report")
        progress.advance("embeddings extracted and validated")
        complete_stage(
            "embeddings_cached",
            stage_started,
            train_cache=str(cache_paths.get("train", "")),
            backbone_type=str(extraction_info["backbone_type"]),
            embedding_dim=resolved_dim,
        )

        stage_started = time.perf_counter()
        global_ckpt = train_global_model(
            train_cache=cache_paths["train"],
            val_cache=cache_paths["val"],
            out_dir=run_ctx.checkpoints_dir,
            hidden_dim=int(cfg["model"]["hidden_dim"]),
            latent_dim=int(cfg["model"]["latent_dim"]),
            lr=float(cfg["training"]["learning_rate"]),
            epochs=int(cfg["training"]["epochs"]),
            patience=int(cfg["training"]["patience"]),
            batch_size=int(cfg["training"]["batch_size"]),
            resume_from=(resume_checkpoints_dir / "global_cvae.pt") if resume_checkpoints_dir is not None else None,
            conditioning_cfg=cfg.get("model", {}).get("conditioning", {}),
            configured_domains=cfg.get("data", {}).get("magnifications", []),
        )
        progress.advance("legacy global baseline trained")
        tracker.log_artifact(Path(global_ckpt), artifact_name="global_cvae", artifact_type="checkpoint")
        complete_stage("global_baseline_trained", stage_started, checkpoint=str(global_ckpt))

        stage_started = time.perf_counter()
        experiment.run(
            cfg=cfg,
            run_ctx=run_ctx,
            cache_paths=cache_paths,
            global_ckpt=Path(global_ckpt),
            progress=progress,
            resume_checkpoints_dir=resume_checkpoints_dir,
        )
        complete_stage("experiment_complete", stage_started, mode=mode)
    except KeyboardInterrupt as exc:
        run_status = "interrupted"
        failure_message = f"{type(exc).__name__}: {exc}"
        raise
    except Exception as exc:
        run_status = "failed"
        failure_message = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        total_duration_sec = time.perf_counter() - run_started
        stage_report = {
            "run_status": run_status,
            "total_duration_sec": round(total_duration_sec, 6),
            "n_stages": len(stage_timings),
            "stages": stage_timings,
        }
        if failure_message is not None:
            stage_report["error"] = failure_message

        stage_timing_path = run_ctx.reports_dir / "stage_timing_report.json"
        with stage_timing_path.open("w", encoding="utf-8") as f:
            json.dump(stage_report, f, indent=2)

        tracker.log_artifact(stage_timing_path, artifact_name="stage_timing_report", artifact_type="report")
        tracker.log_metrics(
            {
                "run/total_duration_sec": total_duration_sec,
                "run/failed": run_status != "success",
                "run/n_stages": len(stage_timings),
            },
            step=stage + 1,
        )
        progress.close()
        tracker.finish(status=run_status)


if __name__ == "__main__":
    main()
