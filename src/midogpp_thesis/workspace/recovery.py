"""Narrow dispatch for registered exact-existing-snapshot recovery.

Recovery is deliberately not a general workspace lifecycle mode.  The sole
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
    """Return the sole strategy that must remain bound to its exact experiment."""

    if experiment_id == _EXPERIMENT_ID:
        return EXACT_EXISTING_SNAPSHOT_DISAGREEMENT_REGRET_PREDICTION_ONLY_V1
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
    if strategy_id != EXACT_EXISTING_SNAPSHOT_DISAGREEMENT_REGRET_PREDICTION_ONLY_V1:
        return (
            f"{experiment_id}: unknown runner.run_recovery_strategy {strategy_id!r}",
        )

    expected: tuple[tuple[str, object, object], ...] = (
        ("experiment_id", experiment_id, _EXPERIMENT_ID),
        ("stage", stage, "90_oracles_and_diagnostics"),
        ("status", status, "diagnostic"),
        ("claim_scope", claim_scope, "diagnostic_only"),
        ("config_path", config_path, _CONFIG_PATH),
        ("output_artifact_id", output_artifact_id, _OUTPUT_ARTIFACT_ID),
        ("output canonical_path", output_canonical_path, _OUTPUT_CANONICAL_PATH),
        ("input_artifact_ids", tuple(input_artifact_ids), _INPUT_ARTIFACT_IDS),
        ("runner.argv", tuple(runner_argv), _RUNNER_ARGV),
        ("runner.environment", dict(runner_env), _RUNNER_ENV),
    )
    return tuple(
        f"{experiment_id}: recovery strategy {strategy_id!r} requires exact {label}"
        for label, actual, wanted in expected
        if actual != wanted
    )


def detect_registered_exact_recovery(strategy_id: str, artifact_root: Path) -> bool:
    """Recognize the registered failure and propagate any boundary drift."""

    if strategy_id != EXACT_EXISTING_SNAPSHOT_DISAGREEMENT_REGRET_PREDICTION_ONLY_V1:
        raise RecoveryContractError(f"Unknown workspace recovery strategy: {strategy_id!r}")

    # Import lazily so ordinary workspace validation and preparation do not load
    # the CVAE runtime (and its heavy numerical dependencies).
    from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only.recovery_contracts import (  # noqa: E501
        detect_post_test_seal_recovery,
    )
    return bool(detect_post_test_seal_recovery(artifact_root))


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
    "RecoveryContractError",
    "SnapshotBytesGuard",
    "detect_registered_exact_recovery",
    "registration_errors",
    "required_strategy_for_experiment",
    "validate_preserved_snapshots",
)
