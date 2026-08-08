"""End-to-end fresh Stage-70 runner with a global pre-label prediction seal."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Callable, Mapping

from ...protocol import ProtocolError
from .bundle import write_residual_topup_fresh_bundle, write_validation_report
from .config import EXPERIMENT_NAME, ResidualTopupFreshConfig
from .execution import (
    FrozenPolicySurface,
    PredictionCache,
    PredictionTaskExecutor,
    load_frozen_policy_actions,
    materialize_prediction_cache,
)
from .inference import evaluate_sealed_predictions
from .label_access import open_scoring_labels_after_prediction_seal
from .planning import build_evaluation_plan
from .prediction_seal import seal_predictions
from .source_cache import (
    FreshSourceCache,
    SourceTaskExecutor,
    load_validated_generation_lock,
    materialize_source_cache,
)
from .target_cache import (
    FreshTargetSurface,
    require_fresh_target_artifacts,
    load_fresh_target_surface,
)
from .validation import validate_residual_topup_fresh_bundle
from .workstation import WorkstationProbes, run_workstation_preflight
from .workspace_binding import validate_residual_topup_fresh_workspace_binding


@dataclass(frozen=True)
class FreshRunnerDependencies:
    """Narrow injection seam for CPU-only protocol/order tests."""

    require_inputs: Callable[[ResidualTopupFreshConfig], None] | None = None
    validate_workspace: Callable[[ResidualTopupFreshConfig], object] | None = None
    run_preflight: Callable[..., Mapping[str, object]] | None = None
    load_policy: Callable[[ResidualTopupFreshConfig], FrozenPolicySurface] | None = None
    load_target: Callable[[ResidualTopupFreshConfig], FreshTargetSurface] | None = None
    load_generation: Callable[[ResidualTopupFreshConfig], object] | None = None
    materialize_source: Callable[..., FreshSourceCache] | None = None
    materialize_prediction: Callable[..., PredictionCache] | None = None
    open_labels: Callable[..., Mapping[str, int]] | None = None
    evaluate: Callable[..., object] | None = None
    write_bundle: Callable[..., Mapping[str, object]] | None = None
    validate_bundle: Callable[..., Mapping[str, object]] | None = None


def require_fresh_execution_inputs(config: ResidualTopupFreshConfig) -> None:
    """Fail immediately while the planned fresh artifacts remain absent."""

    require_fresh_target_artifacts(config)
    expected = (
        config.expert_bank_root / "manifests/expert_bank_index.json",
        config.expert_bank_root / "reports/validation_report.json",
        config.generation_lock_root / "config.resolved.yaml",
        config.generation_lock_root / "manifests/generation_lock.json",
        config.generation_lock_root / "reports/validation_report.json",
        config.policy_root / "manifests/action_library.json",
        config.policy_root / "manifests/policy_lock.json",
        config.policy_root / "manifests/content_index.json",
        config.policy_root / "reports/run_state.json",
        config.policy_root / "reports/validation_report.json",
    )
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise ProtocolError(
            "Fresh Stage-70 is blocked before runtime admission: independently "
            f"validated planned inputs are absent ({missing}). No consumed "
            "Stage-70 or Stage-90 artifact may substitute."
        )


def run_residual_topup_fresh(
    config: ResidualTopupFreshConfig,
    *,
    enable_optional_local_scratch: bool = False,
    workstation_probes: WorkstationProbes | None = None,
    source_executor: SourceTaskExecutor | None = None,
    prediction_executor: PredictionTaskExecutor | None = None,
    dependencies: FreshRunnerDependencies | None = None,
) -> Path:
    """Generate, predict, globally seal, then—and only then—open labels."""

    deps = dependencies or FreshRunnerDependencies()
    (deps.require_inputs or require_fresh_execution_inputs)(config)
    (deps.validate_workspace or validate_residual_topup_fresh_workspace_binding)(
        config
    )
    root = config.artifact_root
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "reports/run_state.json"
    if state_path.is_file() and _json(state_path).get("status") == "COMPLETE":
        (deps.validate_bundle or validate_residual_topup_fresh_bundle)(
            root,
            config=config,
        )
        return root

    _write_state(root, "RUNNING")
    try:
        preflight = (deps.run_preflight or run_workstation_preflight)(
            root,
            runtime=config.runtime,
            probes=workstation_probes,
            enable_optional_local_scratch=enable_optional_local_scratch,
        )
        policy = (deps.load_policy or load_frozen_policy_actions)(config)
        target = (deps.load_target or load_fresh_target_surface)(config)
        generation_lock = (deps.load_generation or load_validated_generation_lock)(
            config
        )
        generation_lock_hash = str(generation_lock.generation_lock_hash)
        scratch_root = None
        if enable_optional_local_scratch:
            scratch_root = (
                Path(str(config.runtime["optional_local_scratch_root"]))
                / EXPERIMENT_NAME
                / "checkpoints/source"
            )
        source = (deps.materialize_source or materialize_source_cache)(
            config,
            generation_lock,
            root=root / "checkpoints/source",
            scratch_root=scratch_root,
            executor=source_executor,
        )
        plan = build_evaluation_plan(
            policy.actions_by_target,
            evaluation_row_ids_by_target=target.evaluation_row_ids_by_target,
        )
        prediction = (
            deps.materialize_prediction or materialize_prediction_cache
        )(
            config,
            plan=plan,
            policy=policy,
            source_cache=source,
            target_surface=target,
            generation_lock_hash=generation_lock_hash,
            root=root / "checkpoints/predictions",
            executor=prediction_executor,
        )

        # This is the only capability that can authorize label parsing.  It is
        # issued only after all 1,053 action/target/seed cells are complete.
        prediction_seal = seal_predictions(plan, prediction.predictions)
        labels = (deps.open_labels or open_scoring_labels_after_prediction_seal)(
            target,
            prediction_seal,
        )
        report = (deps.evaluate or evaluate_sealed_predictions)(
            prediction_seal,
            labels,
        )
        (deps.write_bundle or write_residual_topup_fresh_bundle)(
            root,
            config=config,
            policy=policy,
            target_surface=target,
            source_cache=source,
            prediction_cache=prediction,
            plan=plan,
            prediction_seal=prediction_seal,
            report=report,
            workstation_report=preflight,
        )
        checks = (deps.validate_bundle or validate_residual_topup_fresh_bundle)(
            root,
            config=config,
            allow_pending=True,
        )
        write_validation_report(root, checks)
        (deps.validate_bundle or validate_residual_topup_fresh_bundle)(
            root,
            config=config,
        )
    except Exception:
        _write_state(root, "FAILED")
        raise
    return root


def _write_state(root: Path, status: str) -> None:
    path = root / "reports/run_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": "midogpp_residual_topup_fresh_run_state_v1",
                "status": status,
                "claim_scope": "synthetic_downstream_utility",
                "prediction_seal_hash": None,
                "policy_update_emitted": False,
            },
            handle,
            sort_keys=True,
            separators=(",", ":"),
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("Cannot read fresh Stage-70 run state.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("Fresh Stage-70 run state must be a mapping.")
    return payload


__all__ = (
    "FreshRunnerDependencies",
    "require_fresh_execution_inputs",
    "run_residual_topup_fresh",
)
