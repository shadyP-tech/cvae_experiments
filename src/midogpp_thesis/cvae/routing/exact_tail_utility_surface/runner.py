"""Capability-ordered orchestration for the exact-tail utility surface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

from ...protocol import ProtocolError
from ..utility_aligned import CandidateFeatureRow
from .bundle import REQUIRED_FILES, ExactTailUtilitySurfaceLock, validate_surface_bundle
from .config import (
    ExactTailUtilitySurfaceConfig,
    FreshInputAttestation,
    validate_fresh_inputs_ready,
)
from .contracts import DevelopmentPartition
from .label_access import open_globally_sealed_development_labels
from .runtime import WorkstationSnapshot, validate_workstation_snapshot
from .scoring import (
    ScoredExactTailUtilityRow,
    SealedPredictionSurface,
    score_exact_tail_utility_surface,
)
from .seals import GlobalPredictionSeal
from .workspace_binding import validate_production_workspace_binding


@dataclass(frozen=True)
class PreparedPredictionCapability:
    """Durable label-free outputs returned by a workstation execution adapter."""

    partitions: Mapping[str, DevelopmentPartition]
    predictions: SealedPredictionSurface
    seal: GlobalPredictionSeal
    seal_path: Path
    prediction_index_path: Path
    prediction_arrays_path: Path
    feature_rows: tuple[CandidateFeatureRow, ...]

    def __post_init__(self) -> None:
        if self.predictions.seal.seal_hash != self.seal.seal_hash:
            raise ProtocolError("Exact-tail adapter mixed prediction capabilities.")


class ExactTailExecutionAdapter(Protocol):
    """Injectable workstation I/O boundary; scientific logic stays in this package."""

    def collect_workstation_snapshot(
        self, config: ExactTailUtilitySurfaceConfig
    ) -> WorkstationSnapshot: ...

    def materialize_label_free_predictions(
        self,
        config: ExactTailUtilitySurfaceConfig,
        attestation: FreshInputAttestation,
    ) -> PreparedPredictionCapability: ...

    def persist_scored_bundle(
        self,
        config: ExactTailUtilitySurfaceConfig,
        capability: PreparedPredictionCapability,
        rows: Sequence[ScoredExactTailUtilityRow],
    ) -> Path: ...


def run_exact_tail_utility_surface(
    config: ExactTailUtilitySurfaceConfig,
    *,
    adapter: ExactTailExecutionAdapter | None = None,
    workspace_validator: Callable[[ExactTailUtilitySurfaceConfig], None] = (
        validate_production_workspace_binding
    ),
) -> ExactTailUtilitySurfaceLock:
    """Run with irreversible label access strictly after durable global sealing."""

    if adapter is None:
        # Lazy import keeps the orchestration module free of workstation-only
        # dependencies and makes dependency injection available to focused tests.
        from .production_adapter import ProductionExactTailAdapter

        adapter = ProductionExactTailAdapter()

    # Registry authorization precedes even a completed-artifact fast path.
    workspace_validator(config)
    if all((config.artifact_root / member).is_file() for member in REQUIRED_FILES):
        return validate_surface_bundle(config.artifact_root, config=config)
    run_state = config.artifact_root / "reports/run_state.json"
    if run_state.is_file():
        import json

        try:
            state = json.loads(run_state.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProtocolError("Exact-tail partial run-state is unreadable.") from exc
        if isinstance(state, Mapping) and state.get("status") == "COMPLETE":
            raise ProtocolError(
                "Exact-tail COMPLETE artifact is incomplete; refusing silent regeneration."
            )
    # Fresh-input authorization precedes workstation probing/GPU initialization.
    attestation = validate_fresh_inputs_ready(config)
    snapshot = adapter.collect_workstation_snapshot(config)
    validate_workstation_snapshot(snapshot)

    capability = adapter.materialize_label_free_predictions(config, attestation)
    capability.seal.verify_complete()
    expected_binding = (
        capability.seal.reservation_index_hash,
        capability.seal.development_cache_binding_hash,
        capability.seal.development_manifest_sha256,
        capability.seal.target_evaluation_binding_hash,
    )
    observed_binding = (
        attestation.reservation_index_hash,
        attestation.development_cache_binding_hash,
        attestation.development_manifest_sha256,
        attestation.target_evaluation_binding_hash,
    )
    if expected_binding != observed_binding:
        raise ProtocolError("Exact-tail prediction capability escaped fresh inputs.")

    labels = open_globally_sealed_development_labels(
        config.development_manifest_path,
        capability.partitions,
        seal=capability.seal,
        seal_path=capability.seal_path,
        prediction_index_path=capability.prediction_index_path,
        prediction_arrays_path=capability.prediction_arrays_path,
    )
    rows = score_exact_tail_utility_surface(
        capability.predictions, labels, capability.partitions
    )
    bundle_root = adapter.persist_scored_bundle(config, capability, rows)
    if Path(bundle_root).resolve() != config.artifact_root.resolve():
        raise ProtocolError("Exact-tail adapter published outside the workspace output.")
    return validate_surface_bundle(bundle_root, config=config)


__all__ = (
    "ExactTailExecutionAdapter",
    "PreparedPredictionCapability",
    "run_exact_tail_utility_surface",
)
