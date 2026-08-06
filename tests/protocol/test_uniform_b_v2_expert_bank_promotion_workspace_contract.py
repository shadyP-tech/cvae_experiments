from __future__ import annotations

from copy import deepcopy

from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID,
    SOURCE_ARTIFACT_ID,
)
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.workspace_binding import (
    CACHE_ID,
    INPUT_IDS,
)
from midogpp_thesis.workspace.runtime import MidogppWorkspace


def test_v2_promotion_is_the_only_routing_authorization_edge() -> None:
    workspace = MidogppWorkspace.load()
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    source = workspace.artifacts[SOURCE_ARTIFACT_ID]
    cache = workspace.artifacts[CACHE_ID]
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]

    assert experiment.stage == "30_expert_bank"
    assert experiment.status == "active"
    assert experiment.input_artifact_ids == INPUT_IDS
    assert SOURCE_ARTIFACT_ID in experiment.input_claim_scope_exceptions
    assert "expert_bank_evidence" not in source.forbidden_reuse
    assert "routing_evidence" in source.forbidden_reuse
    assert source.may_feed_deployable_selection is False
    assert "expert_bank_evidence" not in cache.forbidden_reuse
    assert "routing_evidence" in cache.forbidden_reuse
    assert cache.may_feed_deployable_selection is False
    assert output.claim_scope == "expert_bank_construction_only"
    assert output.may_feed_deployable_selection is True
    assert "routing_evidence" not in output.forbidden_reuse
    assert "expert_selection_evidence" not in output.forbidden_reuse


def test_promoted_bank_is_a_valid_future_stage60_input() -> None:
    source = MidogppWorkspace.load()
    registry = deepcopy(source.registry_payload)
    catalog = deepcopy(source.catalog_payload)
    catalog["artifacts"].append(
        {
            "artifact_id": "test_uniform_b_router_output",
            "stage": "60_routing_and_composition",
            "canonical_path": "artifacts/midogpp/60_routing_and_composition/test_uniform_b_router/v1",
            "availability": "generated_on_run",
            "migration": "canonical_output",
            "evidence_label": "TODO_VERIFY_ARTIFACT",
            "claim_scope": "routing_and_composition",
        }
    )
    registry["experiments"].append(
        {
            "experiment_id": "test.uniform_b.router",
            "stage": "60_routing_and_composition",
            "status": "active",
            "claim_scope": "routing_and_composition",
            "output_artifact_id": "test_uniform_b_router_output",
            "input_artifact_ids": [OUTPUT_ARTIFACT_ID],
            "runner": {"argv": ["{python}", "-c", "pass"]},
        }
    )
    candidate = MidogppWorkspace(
        repo_root=source.repo_root,
        registry=registry,
        catalog=catalog,
        workspace=source.workspace_payload,
        protocol_defaults=source.protocol_defaults_payload,
    )

    candidate.validate()
