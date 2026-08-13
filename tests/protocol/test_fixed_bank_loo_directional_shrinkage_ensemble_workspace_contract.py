from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble.bundle import (
    REQUIRED_FILES,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble.config import (
    load_fixed_bank_loo_directional_shrinkage_ensemble_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble import (
    experiment_contracts as contracts,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble.inputs import (
    assert_input_fence,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble.protocol import (
    LooDirectionalShrinkageEnsembleProtocol,
    assert_terminal_consumed_test_protocol,
    canonical_consumed_test_protocol,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.workspace.runtime import MidogppWorkspace


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs"
    / "uniform_b_v2_consumed_test_fixed_bank_"
    "loo_directional_shrinkage_ensemble_v1.yaml"
)


def _workspace() -> MidogppWorkspace:
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    return workspace


def test_registry_catalog_aliases_output_and_inventory_are_fenced() -> None:
    workspace = _workspace()
    experiment = workspace.get_experiment(contracts.EXPERIMENT_ID)
    output = workspace.artifacts[contracts.OUTPUT_ARTIFACT_ID]

    assert experiment.stage == "90_oracles_and_diagnostics"
    assert experiment.status == "diagnostic"
    assert experiment.claim_scope == "diagnostic_only"
    assert experiment.input_artifact_ids == contracts.INPUT_ARTIFACT_IDS
    assert experiment.run_recovery_strategy is None
    assert experiment.config_path == (
        "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
        "uniform_b_v2_consumed_test_fixed_bank_"
        "loo_directional_shrinkage_ensemble_v1.yaml"
    )
    assert experiment.runner_argv == (
        "{python}",
        "-m",
        "midogpp_thesis",
        "cvae-diagnostics",
        "fixed-bank-loo-directional-shrinkage-ensemble",
        "--config",
        "{resolved_config}",
        "--artifact-root",
        "output://midogpp_output_uniform_b_v2_consumed_test_fixed_bank_"
        "loo_directional_shrinkage_ensemble_v1",
    )
    assert experiment.runner_env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert experiment.runner_env["OMP_NUM_THREADS"] == "1"

    for artifact_id in (
        contracts.TEST_CACHE_ARTIFACT_ID,
        contracts.TEST_MANIFEST_ARTIFACT_ID,
        contracts.TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
        contracts.LEDGER_AMENDMENT_ARTIFACT_ID,
    ):
        artifact = workspace.artifacts[artifact_id]
        assert artifact.semantic_identities[
            "authorized_consumer_experiment_ids"
        ] == contracts.EXPERIMENT_ID
        assert artifact.semantic_identities["fresh_evidence"] == "false"
        assert artifact.may_feed_recipe_selection is False
        assert artifact.may_feed_deployable_selection is False

    assert output.canonical_path == contracts.CANONICAL_OUTPUT_ROOT
    assert output.semantic_identities["config_contract_hash"] == (
        "500dc61f9f8d3bd0"
    )
    assert output.semantic_identities["protocol_contract_hash"] == (
        "d3dfdfb4d612a97b"
    )
    assert output.semantic_identities["evaluation_case_count"] == "218"
    assert output.semantic_identities["K_grid"] == "4|5|6"
    assert output.semantic_identities["w_grid"] == "0.5|0.6|0.7"
    assert output.semantic_identities["arm_count"] == "9"
    assert output.semantic_identities["workstation_scratch_root"] == (
        contracts.SCRATCH_ROOT
    )
    assert output.semantic_identities["scratch_role"] == "throughput_only"
    assert output.semantic_identities["owned_task_checkpoint_replay_allowed"] == (
        "false"
    )
    assert output.semantic_identities[
        "successful_phase_checkpoint_cleanup_after_validated_global_seal"
    ] == "true"
    assert output.semantic_identities["fresh_evidence"] == "false"
    assert output.semantic_identities["downstream_utility_claimed"] == "false"
    assert output.required_files == REQUIRED_FILES
    assert len(REQUIRED_FILES) == 43
    assert "tables/exact_nine_probability_index.csv" in REQUIRED_FILES
    assert "manifests/aggregate_plan_decision_seal.json" in REQUIRED_FILES
    assert not any(member.startswith("checkpoints/") for member in REQUIRED_FILES)
    assert "oracle_and_diagnostic_evidence" in output.forbidden_reuse
    assert output.may_feed_recipe_selection is False
    assert output.may_feed_deployable_selection is False
    assert not any(
        contracts.OUTPUT_ARTIFACT_ID in candidate.input_artifact_ids
        for candidate in workspace.experiments.values()
    )


def test_exact_six_input_fence_rejects_prior_stage90_and_extra_input() -> None:
    config = load_fixed_bank_loo_directional_shrinkage_ensemble_config(CONFIG)
    assert_input_fence(config)

    with pytest.raises(ProtocolError, match="prior diagnostic input"):
        assert_input_fence(
            replace(
                config,
                test_cache_root=Path(
                    "artifacts/midogpp/90_oracles_and_diagnostics/"
                    "uniform_b_v2_consumed_test_fixed_bank_support_static_"
                    "router_s4/v1"
                ),
            )
        )
    with pytest.raises(ProtocolError, match="exactly six fenced inputs"):
        assert_input_fence(
            SimpleNamespace(
                experiment_id=config.experiment_id,
                output_artifact_id=config.output_artifact_id,
                input_artifact_ids=(*contracts.INPUT_ARTIFACT_IDS, "unexpected"),
                expert_bank_root=config.expert_bank_root,
                generation_lock_root=config.generation_lock_root,
                test_cache_root=config.test_cache_root,
                test_manifest_path=config.test_manifest_path,
                test_consumption_ledger_path=config.test_consumption_ledger_path,
                ledger_amendment_path=config.ledger_amendment_path,
            )
        )


def test_protocol_manifest_rejects_any_boundary_widening() -> None:
    protocol = canonical_consumed_test_protocol()
    assert_terminal_consumed_test_protocol(protocol)
    assert protocol.contract_hash == "d3dfdfb4d612a97b"
    assert protocol.to_payload()["may_authorize_downstream_utility"] is False

    widened = dict(protocol.payload)
    widened["fresh_evidence"] = True
    with pytest.raises(ProtocolError, match="protocol contract drifted"):
        LooDirectionalShrinkageEnsembleProtocol(
            widened,
            protocol.contract_hash,
        )
    with pytest.raises(ProtocolError, match="protocol contract drifted"):
        assert_terminal_consumed_test_protocol(
            replace(protocol, contract_hash="0" * 16)
        )
