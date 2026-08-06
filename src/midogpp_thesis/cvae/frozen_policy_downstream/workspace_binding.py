"""Canonical workspace binding for the Stage-70 descriptive evaluator."""

from __future__ import annotations

from pathlib import Path

from ...workspace.runtime import MidogppWorkspace
from ..expert_bank.uniform_b_v2_promotion.contracts import (
    OUTPUT_ARTIFACT_ID as BANK_ARTIFACT_ID,
)
from ..generation.contracts import OUTPUT_ARTIFACT_ID as GENERATION_ARTIFACT_ID
from ..protocol import ProtocolError
from ..routing.contracts import OUTPUT_ARTIFACT_ID as EQUAL_POLICY_ARTIFACT_ID
from ..routing.metadata_tie_union.contracts import (
    OUTPUT_ARTIFACT_ID as METADATA_POLICY_ARTIFACT_ID,
)
from ..routing.utility_regret_policy.contracts import (
    OUTPUT_ARTIFACT_ID as UTILITY_POLICY_ARTIFACT_ID,
)
from .authorization.contracts import FINAL_AUTHORIZATION_OUTPUT_ARTIFACT_ID
from .authorization.config import CACHE_ARTIFACT_ID
from .config import FrozenPolicyDownstreamConfig
from .contracts import CLAIM_SCOPE, EXPERIMENT_ID, OUTPUT_ARTIFACT_ID


STAGE_ID = "70_frozen_policy_downstream"
SCORING_MANIFEST_ARTIFACT_ID = "midogpp_frozen_policy_test_scoring_manifest_v1"
EVALUATOR_INPUT_IDS = (
    FINAL_AUTHORIZATION_OUTPUT_ARTIFACT_ID,
    CACHE_ARTIFACT_ID,
    BANK_ARTIFACT_ID,
    GENERATION_ARTIFACT_ID,
    EQUAL_POLICY_ARTIFACT_ID,
    METADATA_POLICY_ARTIFACT_ID,
    UTILITY_POLICY_ARTIFACT_ID,
    SCORING_MANIFEST_ARTIFACT_ID,
)


def validate_frozen_policy_downstream_workspace_binding(
    config: FrozenPolicyDownstreamConfig,
) -> None:
    """Require the exact registered descriptive consumer and canonical paths."""

    workspace = MidogppWorkspace.load()
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]
    stage = workspace.stages[STAGE_ID]
    if (
        experiment.status != "active"
        or experiment.stage != STAGE_ID
        or experiment.claim_scope != CLAIM_SCOPE
        or experiment.output_artifact_id != OUTPUT_ARTIFACT_ID
        or experiment.input_artifact_ids != EVALUATOR_INPUT_IDS
        or config.input_artifact_ids != EVALUATOR_INPUT_IDS
        or output.claim_scope != CLAIM_SCOPE
        or CLAIM_SCOPE not in stage.get("allowed_claim_scopes", ())
    ):
        raise ProtocolError("Stage-70 descriptive evaluator workspace binding drifted.")

    scoring_root = workspace.resolve_artifact(SCORING_MANIFEST_ARTIFACT_ID)
    expected = {
        "artifact_root": workspace.resolve_artifact(
            OUTPUT_ARTIFACT_ID,
            for_output=True,
            require_exists=False,
        ),
        "final_authorization_root": workspace.resolve_artifact(
            FINAL_AUTHORIZATION_OUTPUT_ARTIFACT_ID
        ),
        "bank_root": workspace.resolve_artifact(BANK_ARTIFACT_ID),
        "generation_lock_root": workspace.resolve_artifact(GENERATION_ARTIFACT_ID),
        "equal_union_policy_root": workspace.resolve_artifact(
            EQUAL_POLICY_ARTIFACT_ID
        ),
        "metadata_policy_root": workspace.resolve_artifact(
            METADATA_POLICY_ARTIFACT_ID
        ),
        "utility_policy_root": workspace.resolve_artifact(
            UTILITY_POLICY_ARTIFACT_ID
        ),
        "target_cache_root": workspace.resolve_artifact(CACHE_ARTIFACT_ID),
        "scoring_manifest_path": Path(scoring_root) / "manifest.csv",
    }
    configured = {
        "artifact_root": config.artifact_root,
        "final_authorization_root": config.final_authorization_root,
        "bank_root": config.bank_root,
        "generation_lock_root": config.generation_lock_root,
        "equal_union_policy_root": config.equal_union_policy_root,
        "metadata_policy_root": config.metadata_policy_root,
        "utility_policy_root": config.utility_policy_root,
        "target_cache_root": config.target_cache_root,
        "scoring_manifest_path": config.scoring_manifest_path,
    }
    mismatch = [
        key
        for key in expected
        if Path(configured[key]).resolve() != Path(expected[key]).resolve()
    ]
    if mismatch:
        raise ProtocolError(
            f"Stage-70 descriptive evaluator workspace paths drifted: {mismatch}."
        )
    expected_source = config.artifact_root / "config.resolved.yaml"
    if config.source_path.resolve() != expected_source.resolve():
        raise ProtocolError("Stage-70 evaluator must consume its workspace-resolved config.")


__all__ = (
    "EVALUATOR_INPUT_IDS",
    "SCORING_MANIFEST_ARTIFACT_ID",
    "validate_frozen_policy_downstream_workspace_binding",
)
