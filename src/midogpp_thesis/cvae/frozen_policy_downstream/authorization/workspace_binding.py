"""Strict canonical-workspace binding for Stage-70 authorization flows."""

from __future__ import annotations

from pathlib import Path

from ....workspace.runtime import MidogppWorkspace
from ...expert_bank.uniform_b_v2_promotion.contracts import (
    OUTPUT_ARTIFACT_ID as BANK_ARTIFACT_ID,
)
from ...generation.contracts import OUTPUT_ARTIFACT_ID as GENERATION_ARTIFACT_ID
from ...protocol import ProtocolError
from ...routing.contracts import OUTPUT_ARTIFACT_ID as EQUAL_POLICY_ARTIFACT_ID
from ...routing.metadata_tie_union.contracts import (
    OUTPUT_ARTIFACT_ID as METADATA_POLICY_ARTIFACT_ID,
)
from ...routing.utility_regret_policy.contracts import (
    OUTPUT_ARTIFACT_ID as UTILITY_POLICY_ARTIFACT_ID,
)
from .config import (
    CACHE_ARTIFACT_ID,
    FinalAuthorizationConfig,
    ReservationConfig,
)
from .contracts import (
    CLAIM_SCOPE,
    FINAL_AUTHORIZATION_EXPERIMENT_ID,
    FINAL_AUTHORIZATION_OUTPUT_ARTIFACT_ID,
    RESERVATION_EXPERIMENT_ID,
    RESERVATION_OUTPUT_ARTIFACT_ID,
)
from .inputs import CANONICAL_REFERENCE_ARTIFACT_ID


DATASET_CONTRACT_ARTIFACT_ID = "midogpp_dataset_contract_annotation_patch_v1"
TEST_CONSUMPTION_LEDGER_ARTIFACT_ID = (
    "midogpp_uniform_b_test_consumption_ledger_v1"
)
TEST_SCORING_MANIFEST_ARTIFACT_ID = (
    "midogpp_frozen_policy_test_scoring_manifest_v1"
)
STAGE_ID = "70_frozen_policy_downstream"
COMMON_INPUT_IDS = (
    CANONICAL_REFERENCE_ARTIFACT_ID,
    BANK_ARTIFACT_ID,
    GENERATION_ARTIFACT_ID,
    EQUAL_POLICY_ARTIFACT_ID,
    METADATA_POLICY_ARTIFACT_ID,
    UTILITY_POLICY_ARTIFACT_ID,
)
RESERVATION_INPUT_IDS = (
    DATASET_CONTRACT_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    *COMMON_INPUT_IDS,
)
FINAL_INPUT_IDS = (
    TEST_SCORING_MANIFEST_ARTIFACT_ID,
    *COMMON_INPUT_IDS,
    RESERVATION_OUTPUT_ARTIFACT_ID,
    CACHE_ARTIFACT_ID,
)


def validate_reservation_production_workspace_binding(
    config: ReservationConfig,
) -> None:
    workspace = MidogppWorkspace.load()
    workspace.validate()
    experiment = workspace.get_experiment(RESERVATION_EXPERIMENT_ID)
    output = workspace.artifacts[RESERVATION_OUTPUT_ARTIFACT_ID]
    stage = workspace.stages[STAGE_ID]
    if (
        experiment.status != "active"
        or experiment.stage != STAGE_ID
        or experiment.claim_scope != CLAIM_SCOPE
        or experiment.output_artifact_id != RESERVATION_OUTPUT_ARTIFACT_ID
        or experiment.input_artifact_ids != RESERVATION_INPUT_IDS
        or output.claim_scope != CLAIM_SCOPE
        or CLAIM_SCOPE not in stage.get("allowed_claim_scopes", ())
    ):
        raise ProtocolError("Stage-70 reservation workspace binding drifted.")
    expected = _common_paths(workspace)
    expected.update(
        {
            "artifact_root": workspace.resolve_artifact(
                RESERVATION_OUTPUT_ARTIFACT_ID,
                for_output=True,
                require_exists=False,
            ),
            "prospective_cache_root": workspace.resolve_artifact(
                CACHE_ARTIFACT_ID,
                require_exists=False,
            ),
            "scoring_manifest_path": workspace.resolve_artifact(
                DATASET_CONTRACT_ARTIFACT_ID
            )
            / "manifest.csv",
            "test_consumption_ledger_path": workspace.resolve_artifact(
                TEST_CONSUMPTION_LEDGER_ARTIFACT_ID
            )
            / "reports/test_consumption_ledger.json",
        }
    )
    configured = _reservation_paths(config)
    _require_paths(configured, expected, "reservation")


def validate_final_production_workspace_binding(
    config: FinalAuthorizationConfig,
) -> None:
    workspace = MidogppWorkspace.load()
    workspace.validate()
    experiment = workspace.get_experiment(FINAL_AUTHORIZATION_EXPERIMENT_ID)
    output = workspace.artifacts[FINAL_AUTHORIZATION_OUTPUT_ARTIFACT_ID]
    consumer = workspace.get_experiment(config.consumer_experiment_id)
    stage = workspace.stages[STAGE_ID]
    if (
        experiment.status != "active"
        or experiment.stage != STAGE_ID
        or experiment.claim_scope != CLAIM_SCOPE
        or experiment.output_artifact_id != FINAL_AUTHORIZATION_OUTPUT_ARTIFACT_ID
        or experiment.input_artifact_ids != FINAL_INPUT_IDS
        or output.claim_scope != CLAIM_SCOPE
        or consumer.stage != STAGE_ID
        or FINAL_AUTHORIZATION_OUTPUT_ARTIFACT_ID
        not in consumer.input_artifact_ids
        or CACHE_ARTIFACT_ID not in consumer.input_artifact_ids
        or CLAIM_SCOPE not in stage.get("allowed_claim_scopes", ())
    ):
        raise ProtocolError("Stage-70 final authorization workspace binding drifted.")
    expected = _common_paths(workspace)
    expected.update(
        {
            "artifact_root": workspace.resolve_artifact(
                FINAL_AUTHORIZATION_OUTPUT_ARTIFACT_ID,
                for_output=True,
                require_exists=False,
            ),
            "reservation_root": workspace.resolve_artifact(
                RESERVATION_OUTPUT_ARTIFACT_ID
            ),
            "cache_root": workspace.resolve_artifact(CACHE_ARTIFACT_ID),
            "scoring_manifest_path": workspace.resolve_artifact(
                TEST_SCORING_MANIFEST_ARTIFACT_ID
            )
            / "manifest.csv",
        }
    )
    configured = _final_paths(config)
    _require_paths(configured, expected, "final authorization")


def _common_paths(workspace: MidogppWorkspace) -> dict[str, Path]:
    return {
        "canonical_reference_root": workspace.resolve_artifact(
            CANONICAL_REFERENCE_ARTIFACT_ID
        ),
        "bank_root": workspace.resolve_artifact(BANK_ARTIFACT_ID),
        "generation_lock_root": workspace.resolve_artifact(GENERATION_ARTIFACT_ID),
        "equal_union_policy_root": workspace.resolve_artifact(EQUAL_POLICY_ARTIFACT_ID),
        "metadata_policy_root": workspace.resolve_artifact(METADATA_POLICY_ARTIFACT_ID),
        "utility_policy_root": workspace.resolve_artifact(UTILITY_POLICY_ARTIFACT_ID),
    }


def _reservation_paths(config: ReservationConfig) -> dict[str, Path]:
    return {
        "artifact_root": config.artifact_root,
        "canonical_reference_root": config.canonical_reference_root,
        "bank_root": config.bank_root,
        "generation_lock_root": config.generation_lock_root,
        "equal_union_policy_root": config.equal_union_policy_root,
        "metadata_policy_root": config.metadata_policy_root,
        "utility_policy_root": config.utility_policy_root,
        "scoring_manifest_path": config.scoring_manifest_path,
        "test_consumption_ledger_path": config.test_consumption_ledger_path,
        "prospective_cache_root": config.prospective_cache_root,
    }


def _final_paths(config: FinalAuthorizationConfig) -> dict[str, Path]:
    return {
        "artifact_root": config.artifact_root,
        "reservation_root": config.reservation_root,
        "cache_root": config.cache_root,
        "canonical_reference_root": config.canonical_reference_root,
        "bank_root": config.bank_root,
        "generation_lock_root": config.generation_lock_root,
        "equal_union_policy_root": config.equal_union_policy_root,
        "metadata_policy_root": config.metadata_policy_root,
        "utility_policy_root": config.utility_policy_root,
        "scoring_manifest_path": config.scoring_manifest_path,
    }


def _require_paths(
    configured: dict[str, Path],
    expected: dict[str, Path],
    label: str,
) -> None:
    if set(configured) != set(expected):
        raise ProtocolError(f"Stage-70 {label} workspace path coverage drifted.")
    mismatch = [
        key
        for key in configured
        if configured[key].resolve() != Path(expected[key]).resolve()
    ]
    if mismatch:
        raise ProtocolError(f"Stage-70 {label} workspace paths drifted: {mismatch}.")


__all__ = (
    "COMMON_INPUT_IDS",
    "DATASET_CONTRACT_ARTIFACT_ID",
    "FINAL_INPUT_IDS",
    "RESERVATION_INPUT_IDS",
    "TEST_CONSUMPTION_LEDGER_ARTIFACT_ID",
    "TEST_SCORING_MANIFEST_ARTIFACT_ID",
    "validate_final_production_workspace_binding",
    "validate_reservation_production_workspace_binding",
)
