"""End-to-end utility-aligned Stage-70 runner with strict phase ordering."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Callable, Mapping

from ...protocol import ProtocolError
from .bundle import (
    validate_utility_aligned_residual_fresh_bundle,
    write_utility_aligned_residual_fresh_bundle,
)
from .config import (
    EXPERIMENT_NAME,
    UtilityAlignedResidualFreshConfig,
)
from .inference import evaluate_sealed_predictions
from .label_access import open_scoring_labels_after_prediction_seal
from .planning import build_evaluation_plan
from .policy_loading import (
    FrozenUtilityAlignedPolicySurface,
    load_frozen_utility_aligned_policy,
)
from .prediction_cache import (
    PredictionCache,
    PredictionTaskExecutor,
    materialize_prediction_cache,
)
from .prediction_seal import seal_predictions
from .source_cache import (
    FreshSourceCache,
    SourceTaskExecutor,
    load_validated_generation_lock,
    materialize_source_cache,
)
from .target_surface import (
    FreshTargetSurface,
    load_fresh_target_surface,
    require_active_fresh_target_artifacts,
)
from .workstation import WorkstationProbes, run_workstation_preflight
from .workspace_binding import (
    validate_utility_aligned_residual_fresh_workspace_binding,
)


@dataclass(frozen=True)
class _UtilityAlignedFreshRunnerDependencies:
    """Narrow injection seam for protocol-order and CPU-only tests."""

    validate_workspace: Callable[[UtilityAlignedResidualFreshConfig], object] | None = None
    require_inputs: Callable[[UtilityAlignedResidualFreshConfig], None] | None = None
    load_policy: Callable[[UtilityAlignedResidualFreshConfig], FrozenUtilityAlignedPolicySurface] | None = None
    load_target: Callable[..., FreshTargetSurface] | None = None
    run_preflight: Callable[..., Mapping[str, object]] | None = None
    load_generation: Callable[[UtilityAlignedResidualFreshConfig], object] | None = None
    materialize_source: Callable[..., FreshSourceCache] | None = None
    materialize_prediction: Callable[..., PredictionCache] | None = None
    open_labels: Callable[..., Mapping[str, int]] | None = None
    evaluate: Callable[..., object] | None = None
    write_bundle: Callable[..., Mapping[str, object]] | None = None
    validate_bundle: Callable[..., Mapping[str, object]] | None = None


def require_fresh_execution_inputs(
    config: UtilityAlignedResidualFreshConfig,
) -> None:
    """Fail before preflight/generation if any active fresh input is absent."""

    require_active_fresh_target_artifacts(config)
    expected = (
        config.expert_bank_root / "manifests/expert_bank_index.json",
        config.expert_bank_root / "reports/validation_report.json",
        config.generation_lock_root / "config.resolved.yaml",
        config.generation_lock_root / "manifests/generation_lock.json",
        config.generation_lock_root / "reports/validation_report.json",
        config.policy_root / "manifests/action_library.json",
        config.policy_root / "manifests/target_policy_lock.json",
        config.policy_root / "manifests/policy_lock.json",
        config.policy_root / "reports/run_state.json",
        config.policy_root / "reports/validation_report.json",
    )
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise ProtocolError(
            "Utility-aligned Stage-70 is blocked before runtime admission: "
            f"validated fresh inputs are absent ({missing})."
        )


def run_utility_aligned_residual_fresh(
    config: UtilityAlignedResidualFreshConfig,
    *,
    enable_optional_local_scratch: bool = False,
    workstation_probes: WorkstationProbes | None = None,
    source_executor: SourceTaskExecutor | None = None,
    prediction_executor: PredictionTaskExecutor | None = None,
    dependencies: _UtilityAlignedFreshRunnerDependencies | None = None,
) -> Path:
    """Freeze inputs, generate, predict, seal, then and only then score."""

    deps = dependencies or _UtilityAlignedFreshRunnerDependencies()
    # Both gates precede output creation, hardware probing, cache access and GPU
    # initialization.  A stale workspace entry or inactive reservation is a
    # terminal protocol error, not an invitation to reuse consumed evidence.
    (deps.validate_workspace or validate_utility_aligned_residual_fresh_workspace_binding)(
        config
    )
    (deps.require_inputs or require_fresh_execution_inputs)(config)
    policy = (deps.load_policy or load_frozen_utility_aligned_policy)(config)
    target = (deps.load_target or load_fresh_target_surface)(config, policy)

    root = config.artifact_root
    state_path = root / "reports/run_state.json"
    if state_path.is_file() and _json(state_path).get("status") == "COMPLETE":
        (deps.validate_bundle or validate_utility_aligned_residual_fresh_bundle)(
            root, config=config
        )
        return root
    root.mkdir(parents=True, exist_ok=True)
    _write_state(root, "RUNNING")
    try:
        preflight = (deps.run_preflight or run_workstation_preflight)(
            root,
            runtime=config.runtime,
            probes=workstation_probes,
            enable_optional_local_scratch=enable_optional_local_scratch,
        )
        generation_lock = (deps.load_generation or load_validated_generation_lock)(
            config
        )
        generation_lock_hash = str(generation_lock.generation_lock_hash)
        scratch_base = None
        if enable_optional_local_scratch:
            scratch_base = (
                Path(str(config.runtime["optional_local_scratch_root"]))
                / EXPERIMENT_NAME
                / "checkpoints"
            )
        source = (deps.materialize_source or materialize_source_cache)(
            config,
            generation_lock,
            root=root / "checkpoints/source",
            scratch_root=None if scratch_base is None else scratch_base / "source",
            executor=source_executor,
        )
        plan = build_evaluation_plan(
            policy.actions_by_target,
            evaluation_row_ids_by_target=target.evaluation_row_ids_by_target,
        )
        prediction = (deps.materialize_prediction or materialize_prediction_cache)(
            config,
            plan=plan,
            policy=policy,
            source_cache=source,
            target_surface=target,
            generation_lock_hash=generation_lock_hash,
            root=root / "checkpoints/predictions",
            scratch_root=(
                None if scratch_base is None else scratch_base / "predictions"
            ),
            executor=prediction_executor,
        )

        # This opaque capability can only be issued after all 1,053 logical
        # identities exist.  Composition aliases save fits, never seal cells.
        prediction_seal = seal_predictions(plan, prediction.predictions)
        labels = (deps.open_labels or open_scoring_labels_after_prediction_seal)(
            target, prediction_seal
        )
        report = (deps.evaluate or evaluate_sealed_predictions)(
            prediction_seal, labels
        )
        (deps.write_bundle or write_utility_aligned_residual_fresh_bundle)(
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
        (deps.validate_bundle or validate_utility_aligned_residual_fresh_bundle)(
            root, config=config
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
                "schema_version": "midogpp_utility_aligned_fresh_run_state_v1",
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
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError("Utility-aligned run state must be a mapping.")
    return payload


__all__ = (
    "run_utility_aligned_residual_fresh",
)
