"""Optimized HARP v2 runner over the byte-equivalent shared numerical core."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ..fixed_bank_harp_router_v1.runner import (
    HarpStage90RunnerServices,
    dry_run_harp_stage90,
    inspect_harp_stage90,
    run_harp_stage90,
)
from .authorization import (
    HarpV2Authorization,
    HarpV2AuthorizationLease,
    claim_authorization,
    finalize_authorization,
    load_authorization,
)
from .config import HarpStage90V2Config
from .identity import (
    EXECUTION_REVISION,
    EXPERIMENT_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
)
from .input_surfaces import (
    HarpConsumedCacheIndex,
    load_cache_index,
    load_development_labels,
    load_evaluation_truth,
)


V2_RUNNER_SERVICES = HarpStage90RunnerServices(
    config_type=HarpStage90V2Config,
    authorization_type=HarpV2Authorization,
    lease_type=HarpV2AuthorizationLease,
    experiment_id=EXPERIMENT_ID,
    publication_status=PUBLICATION_STATUS,
    terminal_decision=TERMINAL_DECISION,
    execution_revision=EXECUTION_REVISION,
    phase_prefix="harp-stage90-v2",
    load_authorization=load_authorization,
    claim_authorization=claim_authorization,
    finalize_authorization=finalize_authorization,
    load_cache_index=load_cache_index,
    load_development_labels=load_development_labels,
    load_evaluation_truth=load_evaluation_truth,
)


def inspect_harp_stage90_v2(config: HarpStage90V2Config) -> Mapping[str, object]:
    return inspect_harp_stage90(config, services=V2_RUNNER_SERVICES)


def dry_run_harp_stage90_v2(
    config: HarpStage90V2Config, *, artifact_root: str | Path,
) -> Mapping[str, object]:
    return dry_run_harp_stage90(
        config, artifact_root=artifact_root, services=V2_RUNNER_SERVICES
    )


def run_harp_stage90_v2(
    config: HarpStage90V2Config, *, artifact_root: str | Path,
) -> str:
    return run_harp_stage90(
        config, artifact_root=artifact_root, services=V2_RUNNER_SERVICES
    )


__all__ = (
    "V2_RUNNER_SERVICES", "dry_run_harp_stage90_v2",
    "inspect_harp_stage90_v2", "run_harp_stage90_v2",
)
