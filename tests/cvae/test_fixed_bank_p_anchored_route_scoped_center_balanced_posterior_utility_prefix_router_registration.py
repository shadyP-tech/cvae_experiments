from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.bundle import (
    REQUIRED_FILES,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.config import (
    load_cbpupr_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.constants import (
    EXECUTION_REVISION,
    EXECUTION_SCHEMA_REVISION,
    QUARANTINED_V1_EXPERIMENT_ID,
    QUARANTINED_V1_OUTPUT_ARTIFACT_ID,
    QUARANTINED_V1_SCRATCH_ROOT,
    REPAIR_CODE_IDENTITY,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.experiment_contracts import (
    EXPERIMENT_ID,
    INPUT_ARTIFACT_IDS,
    OUTPUT_ARTIFACT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.inputs import (
    _load_ledger_chain,
    assert_input_fence,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router import (
    inputs as inputs_module,
    workspace_inputs as workspace_inputs_module,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.protocol import (
    FROZEN_PROTOCOL_HASH,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router.source_seal import (
    source_seal_identity,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import sha256_file
from midogpp_thesis.workspace.runtime import MidogppWorkspace


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / (
    "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_route_scoped_center_"
    "balanced_posterior_utility_prefix_router_v2.yaml"
)
V1_CONFIG = ROOT / (
    "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_route_scoped_center_"
    "balanced_posterior_utility_prefix_router_v1.yaml"
)
AMENDMENT = ROOT / (
    "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts/"
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_route_scoped_center_"
    "balanced_posterior_utility_prefix_router_ledger_amendment_v2.json"
)


def test_v2_registration_is_exhausted_failed_and_terminal_only() -> None:
    config = load_cbpupr_config(CONFIG)
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]

    assert V1_CONFIG.is_file()
    assert config.contract_hash == "3d15d57df00263e1"
    assert FROZEN_PROTOCOL_HASH == (
        "173828cebe4c54fd965f0629802bb4412b519b189c3aa7f32c470fa6b1790b9f"
    )
    assert sha256_file(AMENDMENT) == config.expected_ledger_amendment_sha256
    assert experiment.status == "failed"
    assert experiment.runnable is False
    assert experiment.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert len(INPUT_ARTIFACT_IDS) == len(set(INPUT_ARTIFACT_IDS)) == 6
    assert all(value.endswith("_v2") for value in INPUT_ARTIFACT_IDS[2:])
    assert output.required_files == REQUIRED_FILES
    assert output.availability == "workstation_failed_preterminal"
    assert output.evidence_label == "REJECTED"
    assert output.semantic_identities["run_state_status"] == "FAILED"
    assert output.semantic_identities["failure_phase"] == (
        "ROUTE_ENDPOINTS_436_POSTERIORS_AND_CANDIDATE_SEAL"
    )
    assert output.semantic_identities["failure_error"] == (
        "CBPUPR endpoint worker plan lineage drifted."
    )
    assert output.semantic_identities["terminal_access_journal_count"] == "0"
    assert output.semantic_identities["original_execution_authorized"] == "true"
    assert output.semantic_identities["further_execution_authorized"] == "false"
    assert output.semantic_identities["authorization_exhausted"] == "true"
    assert output.semantic_identities["recoverable"] == "false"
    assert output.semantic_identities["rerunnable"] == "false"
    assert output.semantic_identities["execution_revision"] == EXECUTION_REVISION
    assert (
        output.semantic_identities["execution_schema_revision"]
        == EXECUTION_SCHEMA_REVISION
    )
    assert output.semantic_identities["mechanical_repair_only"] == "true"
    assert output.semantic_identities["fresh_evidence"] == "false"
    assert output.semantic_identities["routing_success_claimed"] == "false"
    assert output.semantic_identities["may_feed_another_experiment"] == "false"
    assert output.semantic_identities[
        "preterminal_validation_attested_before_terminal_labels"
    ] == "false"
    assert config.claim_boundary["repair_code_identity"] == REPAIR_CODE_IDENTITY
    source_identity = source_seal_identity()
    assert config.protocol["repair_source_manifest_sha256"] == source_identity[
        "repair_source_manifest_sha256"
    ]
    assert config.protocol["repair_source_tree_sha256"] == source_identity[
        "repair_source_tree_sha256"
    ]
    assert config.protocol["repair_source_manifest_checked_pre_gpu"] is True
    assert (
        config.protocol["repair_source_identity_persisted_in_protocol_manifest"]
        is True
    )
    assert config.claim_boundary["scientific_method_changed_from_v1"] is False
    assert config.claim_boundary["quarantined_v1_output_used"] is False
    assert config.claim_boundary["prior_v1_label_capability_history_used"] is False


def test_v1_stays_failed_quarantined_and_receipt_bound() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    experiment = workspace.get_experiment(QUARANTINED_V1_EXPERIMENT_ID)
    output = workspace.artifacts[QUARANTINED_V1_OUTPUT_ARTIFACT_ID]

    assert experiment.status == "failed"
    assert experiment.runnable is False
    assert output.evidence_label == "REJECTED"
    assert output.semantic_identities["quarantined"] == "true"
    assert output.semantic_identities["recoverable"] == "false"
    assert output.semantic_identities["execution_authorized"] == "false"
    assert output.semantic_identities["quarantine_timestamp_utc"] == (
        "20260822T155847Z"
    )
    assert output.semantic_identities["quarantine_receipt_hash"] == (
        "089c3d2e128d9be9afec43050d3619e1dae14933798cca9e906f9a126b95d757"
    )
    assert output.semantic_identities["quarantine_receipt_file_sha256"] == (
        "bcaae8d961772876fae8c5b6f453b5fba9dc755429ebdf4b0221b73233381397"
    )


def test_v2_amendment_binds_canonical_repair_and_preterminal_gate() -> None:
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))

    assert payload["authorized_consumer_experiment_ids"] == [EXPERIMENT_ID]
    assert payload["repair_code_identity"] == REPAIR_CODE_IDENTITY
    assert payload["repair_source_manifest_required"] is True
    assert payload["repair_source_manifest_checked_pre_gpu"] is True
    assert payload["repair_source_identity_persisted_in_protocol_manifest"] is True
    source_identity = source_seal_identity()
    assert payload["repair_source_manifest_sha256"] == source_identity[
        "repair_source_manifest_sha256"
    ]
    assert payload["repair_source_tree_sha256"] == source_identity[
        "repair_source_tree_sha256"
    ]
    assert payload["repair_source_member_count"] == source_identity[
        "repair_source_member_count"
    ]
    assert payload["mechanical_repair_only"] is True
    assert payload["scientific_protocol_unchanged_from_v1"] is True
    assert payload["scientific_method_changed_from_v1"] is False
    assert payload["canonical_row_order_repair_verified"] is True
    assert payload["v1_target_terminal_capability_had_opened"] is True
    assert payload["v1_terminal_outputs_had_persisted"] is True
    assert payload["v1_final_validation_passed"] is False
    assert payload["quarantined_v1_output_used"] is False
    assert payload["quarantined_v1_scratch_or_checkpoint_used"] is False
    assert payload["quarantined_v1_terminal_outputs_used"] is False
    assert payload["prior_v1_label_capability_history_used"] is False
    assert payload["prior_v1_amendment_used"] is False
    assert payload["preterminal_fresh_process_validation_count"] == 2
    assert payload["preterminal_validation_attested_before_terminal_labels"] is True
    assert payload["final_fresh_process_validation_count"] == 2


def test_amendment_source_hash_mismatch_fails_even_when_file_hash_is_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_cbpupr_config(CONFIG)
    parent = tmp_path / "parent.json"
    amendment = tmp_path / "amendment.json"
    parent.write_text(
        json.dumps(
            {"status": "CONSUMED_FOR_REPRESENTATION_ADOPTION", "split": "test"}
        ),
        encoding="utf-8",
    )
    poisoned = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    poisoned["repair_source_tree_sha256"] = "0" * 64
    amendment.write_text(json.dumps(poisoned), encoding="utf-8")
    amendment_hash = hashlib.sha256(amendment.read_bytes()).hexdigest()
    configured = replace(
        config,
        test_consumption_ledger_path=parent,
        ledger_amendment_path=amendment,
        expected_ledger_amendment_sha256=amendment_hash,
    )
    real_sha256_file = inputs_module.sha256_file

    def fake_sha256_file(path: Path) -> str:
        if Path(path) == parent:
            return config.expected_test_consumption_ledger_sha256
        return real_sha256_file(Path(path))

    monkeypatch.setattr(inputs_module, "sha256_file", fake_sha256_file)
    with pytest.raises(ProtocolError, match="consumption-ledger chain drifted"):
        _load_ledger_chain(configured)


def test_failed_v2_registration_cannot_be_reactivated_by_its_old_config() -> None:
    config = load_cbpupr_config(CONFIG)

    with pytest.raises(ProtocolError, match="workspace catalog drifted"):
        workspace_inputs_module.validate_active_workspace_binding(config)


@pytest.mark.parametrize(
    "role,value",
    (
        (
            "test_cache_root",
            "/tmp/fixed_bank_p_anchored_route_scoped_center_balanced_"
            "posterior_utility_prefix_router_test_cache_v1",
        ),
        (
            "test_manifest_path",
            "/tmp/fixed_bank_p_anchored_route_scoped_center_balanced_"
            "posterior_utility_prefix_router_test_manifest_v1/manifest.csv",
        ),
        (
            "test_consumption_ledger_path",
            "/tmp/fixed_bank_p_anchored_route_scoped_center_balanced_"
            "posterior_utility_prefix_router_parent_v1/reports/"
            "test_consumption_ledger.json",
        ),
        (
            "ledger_amendment_path",
            "/tmp/fixed_bank_p_anchored_route_scoped_center_balanced_"
            "posterior_utility_prefix_router_amendment_v1/ledger_amendment.json",
        ),
        (
            "test_cache_root",
            f"/tmp/{QUARANTINED_V1_OUTPUT_ARTIFACT_ID}/cache",
        ),
        (
            "test_cache_root",
            f"{QUARANTINED_V1_SCRATCH_ROOT}/checkpoints",
        ),
        (
            "test_cache_root",
            "/tmp/cbpupr.quarantine-terminal-lineage-20260822T155847Z/cache",
        ),
    ),
)
def test_v2_input_fence_rejects_v1_and_quarantine_state(
    role: str, value: str
) -> None:
    config = load_cbpupr_config(CONFIG)
    assert_input_fence(config)
    with pytest.raises(ProtocolError, match="predecessor diagnostic input"):
        assert_input_fence(replace(config, **{role: Path(value)}))
