from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from midogpp_thesis.cvae.diagnostics import cli as diagnostics_cli
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v5 import runner
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v5.config import (
    CLASSIFIER,
    experiment_payload,
    input_policy_payload,
    load_config,
    source_provenance_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v5.execution.inputs import (
    canonical_execution_amendment_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v5.execution.persistence import (
    DURABLE_ATTESTATION_MEMBER,
    FINAL_INDEX_MEMBER,
    FINAL_SUMMARY_MEMBER,
    FINAL_VALIDATION_MEMBER,
    PRETERMINAL_BUNDLE_MEMBER,
    PRETERMINAL_INDEX_MEMBER,
    TERMINAL_RESULT_MEMBER,
    VALIDATION_REPORT_MEMBER,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v5.experiment_contracts import (
    EXECUTION_AMENDMENT_ARTIFACT_ID,
    EXECUTION_AMENDMENT_FILENAME,
    INPUT_ARTIFACT_IDS,
    SOURCE_INNER_AMENDMENT_ARTIFACT_ID,
    SOURCE_INNER_AMENDMENT_FILENAME,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v5.identity import (
    AUTHORIZATION_BASIS,
    AUTHORIZATION_SCOPE,
    CLI_SURFACE,
    EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
    file_sha256,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v5.physical.prediction_contracts import (
    CANDIDATE_ARRAY_MEMBER,
    EXACT_B_ARRAY_MEMBER,
    PREDICTION_INDEX_MEMBER,
    PREDICTION_RECEIPT_MEMBER,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v5.protocol import (
    claim_boundary_payload,
    protocol_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v5.execution.workstation import (
    workstation_payload,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.workspace.preparation_authority import (
    SCEPTRE_V4_EXECUTION_AMENDMENT_GATE,
    SCEPTRE_V5_EXECUTION_AMENDMENT_GATE,
    preparation_authority_registration_error,
)
from midogpp_thesis.workspace.runtime import MidogppWorkspace


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs"
    / "uniform_b_v2_consumed_test_fixed_bank_sceptre_router_v5.yaml"
)
CONTRACT_ROOT = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts"
    / "sceptre_router_v5"
)
EXECUTION_AMENDMENT = CONTRACT_ROOT / EXECUTION_AMENDMENT_FILENAME
SOURCE_INNER_AMENDMENT = CONTRACT_ROOT / SOURCE_INNER_AMENDMENT_FILENAME


def _expected_config_payload() -> dict[str, object]:
    config = load_config(CONFIG)
    return {
        "experiment": {
            **experiment_payload(),
            "artifact_root": f"output://{OUTPUT_ARTIFACT_ID}",
        },
        "inputs": {
            **input_policy_payload(
                execution_amendment_sha256=(
                    config.expected_execution_amendment_sha256
                )
            ),
            "expert_bank_root": f"artifact://{INPUT_ARTIFACT_IDS[0]}",
            "generation_lock_root": f"artifact://{INPUT_ARTIFACT_IDS[1]}",
            "source_inner_root": f"artifact://{INPUT_ARTIFACT_IDS[2]}",
            "source_inner_amendment_path": (
                f"artifact://{SOURCE_INNER_AMENDMENT_ARTIFACT_ID}/"
                f"{SOURCE_INNER_AMENDMENT_FILENAME}"
            ),
            "test_cache_root": f"artifact://{INPUT_ARTIFACT_IDS[4]}",
            "test_manifest_path": f"artifact://{INPUT_ARTIFACT_IDS[5]}/manifest.csv",
            "test_consumption_ledger_path": (
                f"artifact://{INPUT_ARTIFACT_IDS[6]}/"
                "reports/test_consumption_ledger.json"
            ),
            "execution_amendment_path": (
                f"artifact://{EXECUTION_AMENDMENT_ARTIFACT_ID}/"
                f"{EXECUTION_AMENDMENT_FILENAME}"
            ),
        },
        "protocol": protocol_payload(),
        "classifier": CLASSIFIER.to_payload(),
        "source_provenance": source_provenance_payload(),
        "runtime": workstation_payload(),
        "claim_boundary": claim_boundary_payload(),
    }


def test_executable_config_is_the_exact_live_contract() -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert raw == _expected_config_payload()
    config = load_config(CONFIG)
    assert config.experiment_id == EXPERIMENT_ID
    assert config.execution_authorized is True
    assert config.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert len(INPUT_ARTIFACT_IDS) == len(set(INPUT_ARTIFACT_IDS)) == 8
    assert config.claim_boundary["may_feed_another_experiment"] is False
    assert config.claim_boundary["fresh_evidence"] is False


def test_both_amendments_are_exact_and_hash_pinned() -> None:
    config = load_config(CONFIG)
    execution_payload = json.loads(EXECUTION_AMENDMENT.read_text(encoding="utf-8"))
    assert execution_payload == canonical_execution_amendment_payload(config)
    assert file_sha256(EXECUTION_AMENDMENT) == (
        config.expected_execution_amendment_sha256
    )
    assert file_sha256(SOURCE_INNER_AMENDMENT) == (
        config.expected_source_inner_amendment_sha256
    )
    assert execution_payload["publication_status"] == PUBLICATION_STATUS
    assert execution_payload["terminal_decision"] == TERMINAL_DECISION
    assert execution_payload["fresh_evidence"] is False


def test_inspection_is_path_free_and_mutation_free(tmp_path: Path) -> None:
    config = load_config(CONFIG)
    before = tuple(tmp_path.rglob("*"))
    receipt = runner.inspect_sceptre_v5(config)
    assert tuple(tmp_path.rglob("*")) == before
    assert receipt["status"] == "EXECUTABLE_AUTHORIZED_UNPROBED"
    assert receipt["execution_amendment_declared"] is True
    assert receipt["paths_resolved"] is False
    assert receipt["hardware_probed"] is False
    assert receipt["filesystem_mutations"] == 0
    assert receipt["target_labels_opened"] is False
    assert receipt["publication_status"] == PUBLICATION_STATUS
    assert receipt["terminal_decision"] == TERMINAL_DECISION


def test_cli_exposes_inspection_dry_run_and_default_production(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    parser = diagnostics_cli.build_parser()
    inspected = parser.parse_args(
        [CLI_SURFACE, "--config", str(CONFIG), "--inspect-plan"]
    )
    dried = parser.parse_args(
        [
            CLI_SURFACE,
            "--config",
            str(CONFIG),
            "--artifact-root",
            str(tmp_path),
            "--dry-run",
        ]
    )
    assert inspected.inspect_plan is True and inspected.dry_run is False
    assert dried.dry_run is True and dried.inspect_plan is False

    assert diagnostics_cli.main(
        [CLI_SURFACE, "--config", str(CONFIG), "--inspect-plan"]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "EXECUTABLE_AUTHORIZED_UNPROBED"

    called: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        runner,
        "run_sceptre_v5",
        lambda config, *, artifact_root: (
            called.append((config.experiment_id, Path(artifact_root))) or "done"
        ),
    )
    assert diagnostics_cli.main(
        [
            CLI_SURFACE,
            "--config",
            str(CONFIG),
            "--artifact-root",
            str(tmp_path),
        ]
    ) == 0
    assert called == [(EXPERIMENT_ID, tmp_path)]
    assert capsys.readouterr().out.strip() == "done"


def test_workspace_registration_is_executable_terminal_only_and_pre_gated() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]
    config = load_config(CONFIG)

    assert experiment.status == "diagnostic"
    assert experiment.claim_scope == "diagnostic_only"
    assert experiment.preparation_authority_gate == (
        SCEPTRE_V5_EXECUTION_AMENDMENT_GATE
    )
    assert experiment.run_recovery_strategy is None
    assert experiment.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert output.availability == "generated_on_run"
    semantic = output.semantic_identities
    assert semantic["config_contract_hash"] == config.config_hash
    assert semantic["protocol_contract_hash"] == config.protocol["protocol_hash"]
    assert semantic["expected_execution_amendment_sha256"] == (
        config.expected_execution_amendment_sha256
    )
    assert semantic["execution_authorization_basis"] == AUTHORIZATION_BASIS
    assert semantic["authorization_scope"] == AUTHORIZATION_SCOPE
    assert semantic["execution_authorized"] == "true"
    assert semantic["consumed_test_reuse_authorized"] == "true"
    assert semantic["single_use_execution_identity"] == "true"
    assert semantic["authorization_exhausted"] == "false"
    assert semantic["fresh_evidence"] == "false"
    assert semantic["routing_success_claimed"] == "false"
    assert semantic["nelbo_compatibility_claimed"] == "false"
    assert semantic["may_feed_another_experiment"] == "false"
    assert output.may_feed_recipe_selection is False
    assert output.may_feed_deployable_selection is False

    receipt = workspace._enforce_preparation_authority(experiment)  # noqa: SLF001
    assert receipt is not None
    assert receipt.authority_sha256 == config.expected_execution_amendment_sha256


def test_v4_and_v5_authority_gates_cannot_cross_bind() -> None:
    v4_experiment = (
        "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_sceptre_router.v4"
    )
    assert preparation_authority_registration_error(
        SCEPTRE_V4_EXECUTION_AMENDMENT_GATE,
        experiment_id=EXPERIMENT_ID,
    ) is not None
    assert preparation_authority_registration_error(
        SCEPTRE_V5_EXECUTION_AMENDMENT_GATE,
        experiment_id=v4_experiment,
    ) is not None


def test_v5_scoped_inputs_are_single_consumer_authorized_and_predecessor_free() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    for artifact_id in INPUT_ARTIFACT_IDS[2:]:
        artifact = workspace.artifacts[artifact_id]
        identities = artifact.semantic_identities
        assert identities["authorized_consumer_experiment_ids"] == EXPERIMENT_ID
        assert "registered_consumer_experiment_ids" not in identities
        assert "consumer_resolution_fence_only" not in identities
        assert identities["fresh_evidence"] == "false"
        assert artifact.may_feed_recipe_selection is False
        assert artifact.may_feed_deployable_selection is False
        assert not any(
            predecessor in artifact_id
            for predecessor in (
                "sceptre_router_v1",
                "sceptre_router_v2",
                "sceptre_router_v3",
                "sceptre_router_v4",
            )
        )

    source_amendment = workspace.artifacts[SOURCE_INNER_AMENDMENT_ARTIFACT_ID]
    assert source_amendment.provenance_files == (SOURCE_INNER_AMENDMENT_FILENAME,)
    assert source_amendment.expected_file_hashes[
        SOURCE_INNER_AMENDMENT_FILENAME
    ].digest == file_sha256(SOURCE_INNER_AMENDMENT)

    execution_amendment = workspace.artifacts[EXECUTION_AMENDMENT_ARTIFACT_ID]
    assert execution_amendment.provenance_files == (EXECUTION_AMENDMENT_FILENAME,)
    assert execution_amendment.expected_file_hashes[
        EXECUTION_AMENDMENT_FILENAME
    ].digest == file_sha256(EXECUTION_AMENDMENT)


def test_output_catalog_matches_the_actual_success_persistence_surface() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]
    expected = {
        "config.resolved.yaml",
        "provenance/input_artifacts.json",
        f"prediction_store/{CANDIDATE_ARRAY_MEMBER}",
        f"prediction_store/{EXACT_B_ARRAY_MEMBER}",
        f"prediction_store/{PREDICTION_INDEX_MEMBER}",
        f"prediction_store/{PREDICTION_RECEIPT_MEMBER}",
        PRETERMINAL_BUNDLE_MEMBER,
        PRETERMINAL_INDEX_MEMBER,
        DURABLE_ATTESTATION_MEMBER,
        TERMINAL_RESULT_MEMBER,
        FINAL_SUMMARY_MEMBER,
        FINAL_INDEX_MEMBER,
        FINAL_VALIDATION_MEMBER,
        VALIDATION_REPORT_MEMBER,
        "reports/run_state.json",
    }
    assert set(output.required_files) == expected


def test_config_mutation_is_rejected_before_execution(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["claim_boundary"]["routing_success_claimed"] = True
    poisoned = tmp_path / CONFIG.name
    poisoned.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="claim boundary drifted"):
        load_config(poisoned)
