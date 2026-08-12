"""Narrow dispatch for registered exact-existing-snapshot recovery.

Recovery is deliberately not a general workspace lifecycle mode. Each
registered strategy recognizes one experiment-specific failed state and lets
the normal registered runner continue from snapshots whose bytes predate the
repair checkout.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


EXACT_EXISTING_SNAPSHOT_DISAGREEMENT_REGRET_PREDICTION_ONLY_V1 = (
    "exact_existing_snapshot_disagreement_regret_prediction_only_v1"
)
EXACT_EXISTING_SNAPSHOT_UTILITY_ALIGNED_CONSUMED_TEST_ENDPOINT_ROUTER_V1 = (
    "exact_existing_snapshot_utility_aligned_consumed_test_endpoint_router_v1"
)

_RUN_STATE_SCHEMA_BY_STRATEGY = {
    EXACT_EXISTING_SNAPSHOT_DISAGREEMENT_REGRET_PREDICTION_ONLY_V1: (
        "midogpp_disagreement_regret_prediction_only_run_state_v1"
    ),
    EXACT_EXISTING_SNAPSHOT_UTILITY_ALIGNED_CONSUMED_TEST_ENDPOINT_ROUTER_V1: (
        "midogpp_consumed_test_endpoint_router_run_state_v1"
    ),
}

_EXPERIMENT_ID = (
    "midogpp.oracle."
    "uniform_b_v2_consumed_test_fixed_bank_disagreement_regret_prediction_only.v1"
)
_CONFIG_PATH = (
    "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_v2_consumed_test_fixed_bank_disagreement_regret_prediction_only_v1.yaml"
)
_OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_test_"
    "fixed_bank_disagreement_regret_prediction_only_v1"
)
_OUTPUT_CANONICAL_PATH = (
    "artifacts/midogpp/90_oracles_and_diagnostics/"
    "uniform_b_v2_consumed_test_fixed_bank_disagreement_regret_prediction_only/v1"
)
_INPUT_ARTIFACT_IDS = (
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1",
    "midogpp_output_uniform_b_v2_generation_lock_v1",
    "midogpp_stage90_fixed_bank_disagreement_regret_prediction_only_train_cache_v1",
    "midogpp_stage90_fixed_bank_disagreement_regret_prediction_only_test_cache_v1",
    "midogpp_uniform_b_test_consumption_ledger_"
    "fixed_bank_disagreement_regret_prediction_only_parent_v1",
    "midogpp_uniform_b_test_consumption_ledger_"
    "fixed_bank_disagreement_regret_prediction_only_amendment_v1",
)
_RUNNER_ARGV = (
    "{python}",
    "-m",
    "midogpp_thesis",
    "cvae-diagnostics",
    "fixed-bank-disagreement-regret-prediction-only",
    "--config",
    "{resolved_config}",
    "--artifact-root",
    "output://midogpp_output_uniform_b_v2_consumed_test_"
    "fixed_bank_disagreement_regret_prediction_only_v1",
)
_RUNNER_ENV = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "CUDA_VISIBLE_DEVICES": "0,1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONUNBUFFERED": "1",
}
_ENDPOINT_EXPERIMENT_ID = (
    "midogpp.oracle."
    "uniform_b_v2_consumed_test_utility_aligned_target_static_endpoint_router.v1"
)
_ENDPOINT_CONFIG_PATH = (
    "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_v2_consumed_test_utility_aligned_target_static_endpoint_router_v1.yaml"
)
_ENDPOINT_OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_test_"
    "utility_aligned_target_static_endpoint_router_v1"
)
_ENDPOINT_OUTPUT_CANONICAL_PATH = (
    "artifacts/midogpp/90_oracles_and_diagnostics/"
    "uniform_b_v2_consumed_test_utility_aligned_target_static_endpoint_router/v1"
)
_ENDPOINT_INPUT_ARTIFACT_IDS = (
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1",
    "midogpp_output_uniform_b_v2_generation_lock_v1",
    "midogpp_stage90_utility_aligned_target_static_endpoint_router_test_cache_v1",
    "midogpp_stage90_utility_aligned_target_static_endpoint_router_test_manifest_v1",
    "midogpp_uniform_b_test_consumption_ledger_"
    "utility_aligned_target_static_endpoint_router_parent_v1",
    "midogpp_uniform_b_test_consumption_ledger_"
    "utility_aligned_target_static_endpoint_router_amendment_v1",
)
_ENDPOINT_RUNNER_ARGV = (
    "{python}",
    "-m",
    "midogpp_thesis",
    "cvae-diagnostics",
    "utility-aligned-consumed-test-endpoint-router",
    "--config",
    "{resolved_config}",
    "--artifact-root",
    "output://midogpp_output_uniform_b_v2_consumed_test_"
    "utility_aligned_target_static_endpoint_router_v1",
)
_ENDPOINT_RUNNER_ENV = dict(_RUNNER_ENV)
_REPOSITORY_STATE_KEYS = frozenset(
    {
        "repository_revision",
        "repository_dirty",
        "repository_status_hash",
    }
)


class RecoveryContractError(ValueError):
    """Raised when registered recovery is not byte- and contract-exact."""


@dataclass(frozen=True)
class SnapshotBytesGuard:
    """Immutable byte snapshots checked before and after runner execution."""

    resolved_config_path: Path
    input_manifest_path: Path
    resolved_config_bytes: bytes
    input_manifest_bytes: bytes
    resolved_config_sha256: str
    input_manifest_sha256: str

    @classmethod
    def capture(
        cls,
        resolved_config_path: Path,
        input_manifest_path: Path,
    ) -> "SnapshotBytesGuard":
        try:
            resolved_config_bytes = resolved_config_path.read_bytes()
            input_manifest_bytes = input_manifest_path.read_bytes()
        except OSError as exc:
            raise RecoveryContractError(
                "Exact recovery requires the existing resolved config and input manifest."
            ) from exc
        return cls(
            resolved_config_path=resolved_config_path,
            input_manifest_path=input_manifest_path,
            resolved_config_bytes=resolved_config_bytes,
            input_manifest_bytes=input_manifest_bytes,
            resolved_config_sha256=hashlib.sha256(resolved_config_bytes).hexdigest(),
            input_manifest_sha256=hashlib.sha256(input_manifest_bytes).hexdigest(),
        )

    def assert_unchanged(self) -> None:
        for path, expected_bytes, expected_hash in (
            (
                self.resolved_config_path,
                self.resolved_config_bytes,
                self.resolved_config_sha256,
            ),
            (
                self.input_manifest_path,
                self.input_manifest_bytes,
                self.input_manifest_sha256,
            ),
        ):
            try:
                observed = path.read_bytes()
            except OSError as exc:
                raise RecoveryContractError(
                    f"Registered recovery snapshot disappeared during execution: {path}"
                ) from exc
            observed_hash = hashlib.sha256(observed).hexdigest()
            if observed != expected_bytes or observed_hash != expected_hash:
                raise RecoveryContractError(
                    "Registered recovery modified an immutable run snapshot: "
                    f"{path} (expected sha256={expected_hash}, got sha256={observed_hash})."
                )


def required_strategy_for_experiment(experiment_id: str) -> str | None:
    """Return the narrow strategy that must remain bound to each exact experiment."""

    if experiment_id == _EXPERIMENT_ID:
        return EXACT_EXISTING_SNAPSHOT_DISAGREEMENT_REGRET_PREDICTION_ONLY_V1
    if experiment_id == _ENDPOINT_EXPERIMENT_ID:
        return EXACT_EXISTING_SNAPSHOT_UTILITY_ALIGNED_CONSUMED_TEST_ENDPOINT_ROUTER_V1
    return None


def registration_errors(
    strategy_id: str | None,
    *,
    experiment_id: str,
    stage: str,
    status: str,
    claim_scope: str,
    config_path: str | None,
    output_artifact_id: str,
    output_canonical_path: str | None,
    input_artifact_ids: Sequence[str],
    runner_argv: Sequence[str],
    runner_env: Mapping[str, str],
) -> tuple[str, ...]:
    """Validate recovery metadata against its single allowlisted registration."""

    required = required_strategy_for_experiment(experiment_id)
    if strategy_id is None:
        if required is None:
            return ()
        return (
            f"{experiment_id}: runner.run_recovery_strategy must remain {required!r}",
        )
    if strategy_id == EXACT_EXISTING_SNAPSHOT_DISAGREEMENT_REGRET_PREDICTION_ONLY_V1:
        wanted = (
            _EXPERIMENT_ID,
            _CONFIG_PATH,
            _OUTPUT_ARTIFACT_ID,
            _OUTPUT_CANONICAL_PATH,
            _INPUT_ARTIFACT_IDS,
            _RUNNER_ARGV,
            _RUNNER_ENV,
        )
    elif strategy_id == EXACT_EXISTING_SNAPSHOT_UTILITY_ALIGNED_CONSUMED_TEST_ENDPOINT_ROUTER_V1:
        wanted = (
            _ENDPOINT_EXPERIMENT_ID,
            _ENDPOINT_CONFIG_PATH,
            _ENDPOINT_OUTPUT_ARTIFACT_ID,
            _ENDPOINT_OUTPUT_CANONICAL_PATH,
            _ENDPOINT_INPUT_ARTIFACT_IDS,
            _ENDPOINT_RUNNER_ARGV,
            _ENDPOINT_RUNNER_ENV,
        )
    else:
        return (
            f"{experiment_id}: unknown runner.run_recovery_strategy {strategy_id!r}",
        )

    (
        wanted_experiment_id,
        wanted_config_path,
        wanted_output_artifact_id,
        wanted_output_canonical_path,
        wanted_input_artifact_ids,
        wanted_runner_argv,
        wanted_runner_env,
    ) = wanted

    expected: tuple[tuple[str, object, object], ...] = (
        ("experiment_id", experiment_id, wanted_experiment_id),
        ("stage", stage, "90_oracles_and_diagnostics"),
        ("status", status, "diagnostic"),
        ("claim_scope", claim_scope, "diagnostic_only"),
        ("config_path", config_path, wanted_config_path),
        ("output_artifact_id", output_artifact_id, wanted_output_artifact_id),
        ("output canonical_path", output_canonical_path, wanted_output_canonical_path),
        ("input_artifact_ids", tuple(input_artifact_ids), wanted_input_artifact_ids),
        ("runner.argv", tuple(runner_argv), wanted_runner_argv),
        ("runner.environment", dict(runner_env), wanted_runner_env),
    )
    return tuple(
        f"{experiment_id}: recovery strategy {strategy_id!r} requires exact {label}"
        for label, actual, wanted in expected
        if actual != wanted
    )


def detect_registered_exact_recovery(strategy_id: str, artifact_root: Path) -> bool:
    """Recognize the registered failure and propagate any boundary drift."""

    # Import lazily so ordinary workspace validation and preparation do not load
    # the CVAE runtime (and its heavy numerical dependencies).
    if strategy_id == EXACT_EXISTING_SNAPSHOT_DISAGREEMENT_REGRET_PREDICTION_ONLY_V1:
        from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only.recovery_contracts import (  # noqa: E501
            detect_post_test_seal_recovery,
        )

        return bool(detect_post_test_seal_recovery(artifact_root))
    if strategy_id == EXACT_EXISTING_SNAPSHOT_UTILITY_ALIGNED_CONSUMED_TEST_ENDPOINT_ROUTER_V1:
        from midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router.recovery import (  # noqa: E501
            detect_registered_endpoint_router_recovery,
        )

        return bool(detect_registered_endpoint_router_recovery(artifact_root))
    raise RecoveryContractError(f"Unknown workspace recovery strategy: {strategy_id!r}")


def registered_recovery_state_status(
    strategy_id: str,
    artifact_root: Path,
) -> str | None:
    """Read a registered run state without following unsafe filesystem aliases."""

    state_path = artifact_root / "reports/run_state.json"
    try:
        state_path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RecoveryContractError(
            "Registered exact-existing-snapshot recovery state is unreadable."
        ) from exc
    if not state_path.is_file() or state_path.is_symlink():
        raise RecoveryContractError(
            "Registered exact-existing-snapshot recovery state is unsafe."
        )
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecoveryContractError(
            "Registered exact-existing-snapshot recovery state is unreadable."
        ) from exc
    if not isinstance(state, Mapping):
        raise RecoveryContractError(
            "Registered exact-existing-snapshot recovery state is malformed."
        )
    expected_schema = _RUN_STATE_SCHEMA_BY_STRATEGY.get(strategy_id)
    if expected_schema is None:
        raise RecoveryContractError(
            f"Unknown workspace recovery strategy: {strategy_id!r}"
        )
    status = state.get("status")
    if (
        state.get("schema_version") != expected_schema
        or not isinstance(status, str)
        or status not in {"FAILED", "RUNNING", "COMPLETE"}
        or (status == "COMPLETE" and state.get("phase") != "COMPLETE")
        or (status == "COMPLETE" and state.get("error") is not None)
    ):
        raise RecoveryContractError(
            "Registered exact-existing-snapshot recovery state is malformed."
        )
    return str(status)


def validate_preserved_snapshots(
    guard: SnapshotBytesGuard,
    *,
    current_resolved_config_bytes: bytes,
    current_input_manifest: Mapping[str, Any],
) -> None:
    """Require current resolution/input bytes to match the preserved run."""

    if current_resolved_config_bytes != guard.resolved_config_bytes:
        raise RecoveryContractError(
            "Current resolved config does not exactly match the preserved recovery snapshot."
        )
    try:
        preserved_manifest = json.loads(guard.input_manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecoveryContractError(
            "Preserved recovery input manifest is not valid UTF-8 JSON."
        ) from exc
    if not isinstance(preserved_manifest, dict):
        raise RecoveryContractError(
            "Preserved recovery input manifest must be a JSON object."
        )
    current = dict(current_input_manifest)
    missing_repository_fields = sorted(
        key
        for key in _REPOSITORY_STATE_KEYS
        if key not in preserved_manifest or key not in current
    )
    if missing_repository_fields:
        raise RecoveryContractError(
            "Recovery input manifests lack required repository-state fields: "
            f"{missing_repository_fields}."
        )
    preserved_without_repository = {
        key: value
        for key, value in preserved_manifest.items()
        if key not in _REPOSITORY_STATE_KEYS
    }
    current_without_repository = {
        key: value for key, value in current.items() if key not in _REPOSITORY_STATE_KEYS
    }
    if preserved_without_repository != current_without_repository:
        raise RecoveryContractError(
            "Current input manifest differs from the preserved recovery snapshot outside "
            "the three allowed top-level repository-state fields."
        )


__all__ = (
    "EXACT_EXISTING_SNAPSHOT_DISAGREEMENT_REGRET_PREDICTION_ONLY_V1",
    "EXACT_EXISTING_SNAPSHOT_UTILITY_ALIGNED_CONSUMED_TEST_ENDPOINT_ROUTER_V1",
    "RecoveryContractError",
    "SnapshotBytesGuard",
    "detect_registered_exact_recovery",
    "registered_recovery_state_status",
    "registration_errors",
    "required_strategy_for_experiment",
    "validate_preserved_snapshots",
)
