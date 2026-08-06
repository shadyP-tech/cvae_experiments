from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from midogpp_thesis.cvae.frozen_policy_downstream.authorization.bundle import (
    FINAL_REQUIRED_FILES,
    RESERVATION_REQUIRED_FILES,
)
from midogpp_thesis.cvae.frozen_policy_downstream.authorization.config import (
    load_final_authorization_config,
    load_reservation_config,
)
from midogpp_thesis.cvae.frozen_policy_downstream.authorization.workspace_binding import (
    FINAL_INPUT_IDS,
    RESERVATION_INPUT_IDS,
)
from midogpp_thesis.cvae.frozen_policy_downstream.bundle import (
    REQUIRED_FILES as EVALUATOR_REQUIRED_FILES,
)
from midogpp_thesis.cvae.frozen_policy_downstream.workspace_binding import (
    EVALUATOR_INPUT_IDS,
)
from midogpp_thesis.data.features.stage70_test_cache import (
    CACHE_REQUIRED_FILES,
    load_stage70_test_cache_config,
)
from midogpp_thesis.workspace.runtime import MidogppWorkspace, WorkspaceError


RESERVATION_EXPERIMENT_ID = (
    "midogpp.frozen_policy_downstream."
    "uniform_b_v2_descriptive_test_reservation.v1"
)
CACHE_EXPERIMENT_ID = (
    "midogpp.frozen_policy_downstream.uniform_b_v2_descriptive_test_cache.v1"
)
FINAL_EXPERIMENT_ID = (
    "midogpp.frozen_policy_downstream."
    "uniform_b_v2_descriptive_test_final_authorization.v1"
)
EVALUATOR_EXPERIMENT_ID = (
    "midogpp.frozen_policy_downstream."
    "uniform_b_v2_descriptive_frozen_policy_comparison.v1"
)
RESERVATION_OUTPUT_ID = (
    "midogpp_output_uniform_b_v2_descriptive_test_reservation_v1"
)
CACHE_ARTIFACT_ID = (
    "midogpp_virchow2_uniform_b_v2_descriptive_test_cache_seed42"
)
FINAL_OUTPUT_ID = (
    "midogpp_output_uniform_b_v2_descriptive_test_final_authorization_v1"
)
EVALUATOR_OUTPUT_ID = (
    "midogpp_output_uniform_b_v2_descriptive_frozen_policy_comparison_v1"
)
CANONICAL_REFERENCE_ID = "midogpp_output_uniform_b_canonical_reference_v1"
LEDGER_ID = "midogpp_uniform_b_test_consumption_ledger_v1"
SCORING_MANIFEST_ID = "midogpp_frozen_policy_test_scoring_manifest_v1"
MANIFEST_SHA256 = (
    "db661ac7e3dbafde8e283528de6706ab35f2c26629b389706c4504e458cc5869"
)
RESERVATION_ID = (
    "reservation_b541f28d223a315edbec2b630c915fc2e7cd47cbfcc064dabecda860aa14c9e3"
)
RESERVATION_PROTOCOL_HASH = (
    "30596a19ea6a2e37a5ae5baa51bd1f1eb63fa64bd589b44e6d7a87df4800b15a"
)
EXTRACTOR_PROTOCOL_HASH = (
    "8bae5653e53b087f1662216308dcf2352975f5efa6fdda5fae4e72ff04d3790b"
)
FINAL_AUTHORIZATION_TOKEN_HASH = "a344cd66fc88daae"
CACHE_CONTENT_HASH = (
    "df0bdbf64881ee000fe7c56bc486724313accf373ef8e90896344f8d03d187db"
)
CONFIG_BASE = Path(
    "experiments/midogpp/stages/70_frozen_policy_downstream/configs"
)


def test_stage70_registration_separates_gates_cache_and_activated_evaluator() -> None:
    workspace = MidogppWorkspace.load()
    workspace.validate()

    reservation = workspace.get_experiment(RESERVATION_EXPERIMENT_ID)
    final = workspace.get_experiment(FINAL_EXPERIMENT_ID)
    evaluator = workspace.get_experiment(EVALUATOR_EXPERIMENT_ID)

    assert reservation.status == "active"
    assert reservation.input_artifact_ids == RESERVATION_INPUT_IDS
    assert reservation.input_artifact_ids[1] == LEDGER_ID
    assert final.status == "active"
    assert final.input_artifact_ids == FINAL_INPUT_IDS
    assert final.input_artifact_ids[0] == SCORING_MANIFEST_ID
    assert evaluator.status == "active"
    assert evaluator.runnable is True
    assert evaluator.input_artifact_ids == EVALUATOR_INPUT_IDS
    assert evaluator.input_artifact_ids[-1] == SCORING_MANIFEST_ID
    assert CACHE_EXPERIMENT_ID not in workspace.experiments

    assert set(workspace.artifacts[RESERVATION_OUTPUT_ID].required_files) == set(
        RESERVATION_REQUIRED_FILES
    )
    assert set(workspace.artifacts[CACHE_ARTIFACT_ID].required_files) == set(
        CACHE_REQUIRED_FILES
    )
    assert set(workspace.artifacts[FINAL_OUTPUT_ID].required_files) == set(
        FINAL_REQUIRED_FILES
    )
    assert set(workspace.artifacts[EVALUATOR_OUTPUT_ID].required_files) == set(
        EVALUATOR_REQUIRED_FILES
    )
    assert workspace.artifacts[EVALUATOR_OUTPUT_ID].availability == "generated_on_run"
    assert (
        workspace.artifacts[EVALUATOR_OUTPUT_ID].evidence_label
        == "APPROVED_DESCRIPTIVE_EVALUATOR_PENDING_RUN"
    )
    assert (
        workspace.artifacts[EVALUATOR_OUTPUT_ID].semantic_identities[
            "hardened_evaluator_status"
        ]
        == "approved_for_execution"
    )
    assert (
        workspace.artifacts[EVALUATOR_OUTPUT_ID].semantic_identities["fresh_evidence"]
        == "false"
    )
    assert workspace.artifacts[CACHE_ARTIFACT_ID].stage == "derived_features"
    assert workspace.artifacts[CACHE_ARTIFACT_ID].migration == (
        "canonical_derived_feature_cache"
    )


def test_stage70_known_identities_and_config_boundaries(tmp_path: Path) -> None:
    workspace = MidogppWorkspace.load()
    reservation_raw = _config(workspace, "uniform_b_v2_descriptive_test_reservation_v1.yaml")
    final_raw = _config(
        workspace, "uniform_b_v2_descriptive_test_final_authorization_v1.yaml"
    )
    cache_raw = _config(workspace, "uniform_b_v2_descriptive_test_cache_v1.yaml")
    evaluator_raw = _config(
        workspace, "uniform_b_v2_descriptive_frozen_policy_comparison_v1.yaml"
    )

    used: set[str] = set()
    reservation_resolved = workspace.resolve_value(
        reservation_raw, require_inputs=False, used_inputs=used
    )
    assert used == set(RESERVATION_INPUT_IDS)
    reservation_path = tmp_path / "reservation.yaml"
    reservation_path.write_text(
        yaml.safe_dump(reservation_resolved, sort_keys=False), encoding="utf-8"
    )
    reservation = load_reservation_config(reservation_path)
    assert reservation.expected_scoring_manifest_sha256 == MANIFEST_SHA256
    assert reservation.expected_cache_extractor_protocol_hash == EXTRACTOR_PROTOCOL_HASH

    used.clear()
    final_resolved = workspace.resolve_value(
        final_raw, require_inputs=False, used_inputs=used
    )
    assert used == set(FINAL_INPUT_IDS)
    final_path = tmp_path / "final.yaml"
    final_path.write_text(
        yaml.safe_dump(final_resolved, sort_keys=False), encoding="utf-8"
    )
    final = load_final_authorization_config(final_path)
    assert final.expected_scoring_manifest_sha256 == MANIFEST_SHA256
    assert final.expected_cache_extractor_protocol_hash == EXTRACTOR_PROTOCOL_HASH

    cache = load_stage70_test_cache_config(
        workspace.repo_root / CONFIG_BASE / "uniform_b_v2_descriptive_test_cache_v1.yaml"
    )
    assert cache.expected_manifest_sha256 == MANIFEST_SHA256
    assert cache.expected_reservation_id == RESERVATION_ID
    assert cache.expected_reservation_protocol_hash == RESERVATION_PROTOCOL_HASH
    assert cache.expected_cache_extractor_protocol_hash == EXTRACTOR_PROTOCOL_HASH
    assert cache.fresh_evidence is False

    assert (
        evaluator_raw["protocol"]["final_authorization_hash"]
        == FINAL_AUTHORIZATION_TOKEN_HASH
    )
    assert evaluator_raw["protocol"]["target_cache_content_hash"] == CACHE_CONTENT_HASH
    assert workspace.get_experiment(EVALUATOR_EXPERIMENT_ID).status == "active"
    assert workspace.get_experiment(EVALUATOR_EXPERIMENT_ID).runnable is True
    assert evaluator_raw["claim_boundary"]["fresh_confirmatory_evidence"] is False


def test_canonical_reference_exception_is_narrow_and_only_where_required() -> None:
    workspace = MidogppWorkspace.load()
    consumers = {
        RESERVATION_EXPERIMENT_ID,
        FINAL_EXPERIMENT_ID,
    }
    reference = workspace.artifacts[CANONICAL_REFERENCE_ID]
    assert set(
        reference.semantic_identities["authorized_consumer_experiment_ids"].split("|")
    ) == consumers
    for experiment_id in consumers:
        experiment = workspace.get_experiment(experiment_id)
        rationale = experiment.input_claim_scope_exceptions[CANONICAL_REFERENCE_ID]
        assert "independent" in rationale
        assert "no tuning, selection" in rationale
        assert set(experiment.input_claim_scope_exceptions) == {CANONICAL_REFERENCE_ID}


@pytest.mark.parametrize(
    "restricted_artifact_id",
    (RESERVATION_OUTPUT_ID, CACHE_ARTIFACT_ID, FINAL_OUTPUT_ID, SCORING_MANIFEST_ID),
)
def test_generic_or_fresh_stage70_consumers_reject_fenced_artifacts(
    restricted_artifact_id: str,
) -> None:
    source = MidogppWorkspace.load()
    registry = deepcopy(source.registry_payload)
    catalog = deepcopy(source.catalog_payload)
    output_id = f"test_output_for_{restricted_artifact_id}"
    catalog["artifacts"].append(
        {
            "artifact_id": output_id,
            "stage": "70_frozen_policy_downstream",
            "canonical_path": f"artifacts/midogpp/70_frozen_policy_downstream/{output_id}/v1",
            "availability": "planned",
            "migration": "canonical_output",
            "evidence_label": "PLANNED_NO_FRESH_EVIDENCE",
            "claim_scope": "synthetic_downstream_utility",
        }
    )
    registry["experiments"].append(
        {
            "experiment_id": f"test.fresh_or_generic.{restricted_artifact_id}",
            "stage": "70_frozen_policy_downstream",
            "status": "planned",
            "claim_scope": "synthetic_downstream_utility",
            "output_artifact_id": output_id,
            "input_artifact_ids": [restricted_artifact_id],
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

    with pytest.raises(WorkspaceError, match="fenced to authorized consumers"):
        candidate.validate()


def _config(workspace: MidogppWorkspace, filename: str) -> dict[str, object]:
    payload = yaml.safe_load((workspace.repo_root / CONFIG_BASE / filename).read_text())
    assert isinstance(payload, dict)
    return payload
