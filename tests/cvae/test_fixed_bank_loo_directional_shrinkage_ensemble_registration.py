from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from midogpp_thesis.cvae.diagnostics import cli
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble import (
    constants,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble.config import (  # noqa: E501
    load_fixed_bank_loo_directional_shrinkage_ensemble_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble import (
    experiment_contracts as contracts,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.fixed_bank_a1_prediction_contracts import (
    assert_runtime as assert_fixed_bank_a1_runtime,
)
from midogpp_thesis.cvae.runtime.frozen_source_streams import (
    _assert_runtime as assert_frozen_source_runtime,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs"
    / "uniform_b_v2_consumed_test_fixed_bank_"
    "loo_directional_shrinkage_ensemble_v1.yaml"
)
AMENDMENT = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts"
    / "uniform_b_v2_consumed_test_fixed_bank_"
    "loo_directional_shrinkage_ensemble_ledger_amendment_v1.json"
)


def test_config_freezes_exact_six_inputs_case_loo_grid_and_workstation() -> None:
    config = load_fixed_bank_loo_directional_shrinkage_ensemble_config(CONFIG)

    assert config.experiment_id == contracts.EXPERIMENT_ID
    assert config.output_artifact_id == contracts.OUTPUT_ARTIFACT_ID
    assert config.input_artifact_ids == contracts.INPUT_ARTIFACT_IDS
    assert len(config.input_artifact_ids) == 6
    assert config.contract_hash == "500dc61f9f8d3bd0"
    assert config.config_hash == config.contract_hash
    assert config.protocol["held_unit_count"] == 218
    assert config.protocol["arbitrary_folds_used"] is False
    assert config.protocol["support_scope"] == (
        "same_H_all_whole_cases_except_held_c"
    )
    assert config.protocol["donor_prior_excludes_H_and_e"] is True
    assert config.protocol[
        "all_held_case_endpoint_plans_sealed_before_terminal_label_access"
    ] is True
    assert config.protocol[
        "all_aggregate_method_seals_complete_before_terminal_label_access"
    ] is True

    ensemble = config.directional_ensemble
    assert ensemble["direction_ids"] == ["zero_to_one", "one_to_zero"]
    assert ensemble["K_grid"] == [4, 5, 6]
    assert ensemble["w_grid"] == [0.5, 0.6, 0.7]
    assert ensemble["w_rational_grid"] == ["1/2", "3/5", "7/10"]
    assert ensemble["arm_count"] == 9
    assert tuple(row["arm_id"] for row in ensemble["arm_grid"]) == (
        constants.ARM_IDS
    )
    assert ensemble["off_action_id"] == "OFF"
    assert ensemble["final_tie_tolerance"] == 1.0e-12
    assert ensemble["sole_final_threshold"] == 0.5
    assert ensemble["final_threshold_equal_maps_to_positive"] is True
    assert ensemble["matched_G_pipeline"] == (
        "identical_nine_arm_pipeline_with_S_set_equal_to_G"
    )

    assert config.runtime["generation_devices"] == ["cuda:0", "cuda:1"]
    assert config.runtime["persistent_source_workers"] is True
    assert config.runtime["classifier_workers"] == 4
    assert config.runtime["classifier_threads_per_worker"] == 3
    assert config.runtime["launch_blas_threads"] == 1
    assert config.runtime["parent_cuda_context_forbidden"] is True
    assert config.runtime["target_probability_cell_count"] == 810
    assert config.runtime["scratch_preference"] == [
        contracts.SCRATCH_ROOT,
        "artifact_parent",
    ]
    assert config.runtime["owned_task_checkpoint_replay_allowed"] is False
    assert config.runtime["resume_policy"] == (
        "no_cross_run_recovery_intra_launch_atomic_task_checkpoints_only"
    )
    assert config.runtime[
        "successful_phase_checkpoint_cleanup_after_validated_global_seal"
    ] is True
    assert "clean_scratch_only_after_closed_world_validation_pass" not in (
        config.runtime
    )
    assert config.runtime["terminal_recovery_allowed"] is False
    assert config.runtime["cross_run_recovery_allowed"] is False

    for key in (
        "fresh_evidence",
        "routing_success_claimed",
        "routing_quality_claimed",
        "downstream_utility_claimed",
        "action_selection_authorized",
        "policy_update_authorized",
        "promotion_eligible",
        "may_feed_stage50",
        "may_feed_stage60",
        "may_feed_stage70",
        "may_feed_another_stage90",
        "may_feed_another_experiment",
        "may_feed_recipe_selection",
        "may_feed_deployable_selection",
    ):
        assert config.claim_boundary[key] is False


def test_constants_expose_only_the_frozen_executable_grid() -> None:
    assert constants.K_GRID == (4, 5, 6)
    assert constants.W_RATIONAL_GRID == ((1, 2), (3, 5), (7, 10))
    assert constants.W_GRID == (0.5, 0.6, 0.7)
    assert len(constants.ARM_IDS) == 9
    assert constants.arm_id(6, (7, 10)) == "K6::w=7/10"
    with pytest.raises(ProtocolError, match="arm must use"):
        constants.arm_id(6, 0.9)
    with pytest.raises(ProtocolError, match="arm must use"):
        constants.arm_id(7, 0.7)
    for target in constants.CENTERS:
        sources = constants.candidate_sources(target)
        assert len(sources) == 8
        assert target not in sources
        assert len(constants.physical_action_ids(target)) == 10


def test_runtime_payload_satisfies_both_neutral_execution_contracts() -> None:
    runtime = load_fixed_bank_loo_directional_shrinkage_ensemble_config(
        CONFIG
    ).runtime

    assert_frozen_source_runtime(runtime)
    assert_fixed_bank_a1_runtime(runtime)


def test_direct_amendment_is_hash_bound_single_consumer_and_terminal() -> None:
    raw = AMENDMENT.read_bytes()
    payload = json.loads(raw)

    assert hashlib.sha256(raw).hexdigest() == (
        contracts.EXPECTED_LEDGER_AMENDMENT_SHA256
    )
    assert payload["parent_artifact_id"] == (
        "midogpp_uniform_b_test_consumption_ledger_v1"
    )
    assert payload["parent_sha256"] == (
        contracts.EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
    )
    assert payload["authorized_consumer_experiment_ids"] == [
        contracts.EXPERIMENT_ID
    ]
    assert payload["authorization_scope"] == contracts.AUTHORIZATION_SCOPE
    assert payload[
        "all_held_case_endpoint_plans_sealed_before_terminal_label_access"
    ] is True
    assert payload[
        "all_aggregate_method_seals_complete_before_terminal_label_access"
    ] is True
    assert payload["previous_stage90_outputs_used"] is False
    assert payload["previous_stage90_amendments_used"] is False
    assert payload["previous_prediction_surfaces_used"] is False
    assert payload["fresh_evidence"] is False
    assert payload["generic_consumer_authorized"] is False
    assert payload["may_feed_another_experiment"] is False


def test_config_rejects_grid_input_and_claim_widening(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["directional_ensemble"]["w_grid"].append(0.9)
    drifted = tmp_path / "drifted.yaml"
    drifted.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="section drifted: directional_ensemble"):
        load_fixed_bank_loo_directional_shrinkage_ensemble_config(drifted)

    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["inputs"]["unexpected_artifact_id"] = "prior_stage90"
    drifted.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="exact-six input schema drifted"):
        load_fixed_bank_loo_directional_shrinkage_ensemble_config(drifted)

    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["claim_boundary"]["fresh_evidence"] = True
    drifted.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="section drifted: claim_boundary"):
        load_fixed_bank_loo_directional_shrinkage_ensemble_config(drifted)


def test_cli_registers_and_lazily_dispatches_directional_surface(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parsed = cli.build_parser().parse_args(
        (
            "fixed-bank-loo-directional-shrinkage-ensemble",
            "--config",
            str(CONFIG),
            "--artifact-root",
            "/tmp/fixed-bank-loo-directional-shrinkage-ensemble-v1",
        )
    )
    assert parsed.surface == "fixed-bank-loo-directional-shrinkage-ensemble"

    import midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble as surface

    sentinel = object()
    calls: list[tuple[object, Path]] = []
    monkeypatch.setattr(
        surface,
        "load_fixed_bank_loo_directional_shrinkage_ensemble_config",
        lambda _: sentinel,
    )

    def _run(config: object, *, artifact_root: Path) -> Path:
        calls.append((config, artifact_root))
        return Path("/tmp/fixed-bank-loo-directional-shrinkage-result-v1")

    monkeypatch.setattr(
        surface,
        "run_fixed_bank_loo_directional_shrinkage_ensemble",
        _run,
    )
    assert cli.main(
        [
            "fixed-bank-loo-directional-shrinkage-ensemble",
            "--config",
            str(CONFIG),
            "--artifact-root",
            "/tmp/fixed-bank-loo-directional-shrinkage-ensemble-v1",
        ]
    ) == 0
    assert calls == [
        (
            sentinel,
            Path("/tmp/fixed-bank-loo-directional-shrinkage-ensemble-v1"),
        )
    ]
    assert capsys.readouterr().out.strip() == (
        "/tmp/fixed-bank-loo-directional-shrinkage-result-v1"
    )
