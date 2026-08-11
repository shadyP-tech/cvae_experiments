"""Exact state and typed product contracts for post-test-seal recovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.artifact_io import atomic_json, read_json
from .bundle import REQUIRED_FILES, cleanup_owned_atomic_temps


RECOVERY_TERMINAL_FILES = frozenset(
    {
        "manifests/frozen_test_prediction_seal.json",
        "manifests/content_index.json",
        "tables/test_case_features.csv",
        "tables/test_candidate_contrasts.csv",
        "tables/test_selection_diagnostics.csv",
        "tables/test_prediction_summary.csv",
        "reports/leakage_report.json",
        "reports/publication_decision.json",
        "reports/runtime_summary.json",
        "reports/validation_report.json",
    }
)
POST_TEST_SEAL_RECOVERY_FILES = frozenset(REQUIRED_FILES).difference(
    RECOVERY_TERMINAL_FILES
)
FAILED_INFERENCE_STATE = {
    "schema_version": "midogpp_disagreement_regret_prediction_only_run_state_v1",
    "status": "FAILED",
    "phase": "LABEL_FREE_TEST_ADMISSION_AND_FROZEN_INFERENCE",
    "prediction_only": True,
    "test_labels_opened": False,
    "error": "ProtocolError: Inference feature topology drifted.",
}


@dataclass(frozen=True)
class FrozenModelBankView:
    """The only development state consumed by label-free inference recovery."""

    model_banks: tuple[object, ...]
    model_bank_hash: str


@dataclass(frozen=True)
class PostTestSealRecovery:
    generated_sources: object
    source_predictions: object
    development: FrozenModelBankView
    source_label_capability_report: Mapping[str, object]
    test_predictions: object
    workstation_preflight: Mapping[str, object]
    train_test_disjointness: Mapping[str, object]
    audit: Mapping[str, object]


def detect_post_test_seal_recovery(root: Path) -> bool:
    """Recognize only the deterministic failure immediately after test sealing."""

    model_seal = root / "manifests/model_bank_seal.json"
    test_seal = root / "manifests/test_prediction_seal.json"
    if not test_seal.is_file():
        return False
    if not model_seal.is_file():
        raise ProtocolError(
            "Post-test recovery found a test seal without its source-only model seal."
        )
    observed = _observed_recovery_files(root)
    state = read_json(root / "reports/run_state.json")
    if observed != POST_TEST_SEAL_RECOVERY_FILES or state != FAILED_INFERENCE_STATE:
        missing = sorted(POST_TEST_SEAL_RECOVERY_FILES.difference(observed))
        extras = sorted(observed.difference(POST_TEST_SEAL_RECOVERY_FILES))
        raise ProtocolError(
            "Post-test recovery boundary drifted: "
            f"missing={missing}, extras={extras}, state_matches="
            f"{state == FAILED_INFERENCE_STATE}."
        )
    return True


def rollback_post_test_seal_recovery(root: Path) -> None:
    """Remove only recovery-created terminal files and restore the sealed gate."""

    for relative in RECOVERY_TERMINAL_FILES:
        (root / relative).unlink(missing_ok=True)
    cleanup_owned_atomic_temps(root)
    atomic_json(root / "reports/run_state.json", FAILED_INFERENCE_STATE)
    observed = _observed_recovery_files(root)
    if observed != POST_TEST_SEAL_RECOVERY_FILES:
        missing = sorted(POST_TEST_SEAL_RECOVERY_FILES.difference(observed))
        extras = sorted(observed.difference(POST_TEST_SEAL_RECOVERY_FILES))
        raise ProtocolError(
            "Post-test recovery rollback could not restore the sealed boundary: "
            f"missing={missing}, extras={extras}."
        )


def _observed_recovery_files(root: Path) -> frozenset[str]:
    """Inventory every durable file while excluding only the root lock."""

    return frozenset(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() != ".run.lock"
    )


__all__ = (
    "FAILED_INFERENCE_STATE",
    "FrozenModelBankView",
    "POST_TEST_SEAL_RECOVERY_FILES",
    "RECOVERY_TERMINAL_FILES",
    "PostTestSealRecovery",
    "detect_post_test_seal_recovery",
    "rollback_post_test_seal_recovery",
)
