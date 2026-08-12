from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from midogpp_thesis.cvae.diagnostics import cli
from midogpp_thesis.cvae.diagnostics.fixed_bank_multi_challenger_hierarchical_flip_router.config import (
    load_fixed_bank_multi_challenger_hierarchical_flip_router_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_multi_challenger_hierarchical_flip_router import (
    constants as runtime_constants,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_multi_challenger_hierarchical_flip_router.experiment_contracts import (
    AUTHORIZATION_SCOPE,
    EXPERIMENT_ID,
    EXPECTED_LEDGER_AMENDMENT_SHA256,
    INPUT_ARTIFACT_IDS,
    LEDGER_AMENDMENT_ARTIFACT_ID,
    METHOD_IDS,
    OOF_PARTITION_NAMESPACE,
    OUTPUT_ARTIFACT_ID,
    TEST_CACHE_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.workspace.recovery import (
    EXACT_EXISTING_SNAPSHOT_FIXED_BANK_MULTI_CHALLENGER_HIERARCHICAL_FLIP_ROUTER_V1,
    detect_registered_exact_recovery,
    required_strategy_for_experiment,
)
from midogpp_thesis.workspace.runtime import MidogppWorkspace


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs"
    / "uniform_b_v2_consumed_test_fixed_bank_multi_challenger_hierarchical_"
    "flip_router_v1.yaml"
)
AMENDMENT = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts"
    / "uniform_b_v2_consumed_test_fixed_bank_multi_challenger_hierarchical_"
    "flip_router_ledger_amendment_v1.json"
)


def _workspace() -> MidogppWorkspace:
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    return workspace


def test_config_freezes_menu_models_margin_and_workstation_topology() -> None:
    config = load_fixed_bank_multi_challenger_hierarchical_flip_router_config(
        CONFIG
    )

    assert config.experiment_id == EXPERIMENT_ID
    assert config.output_artifact_id == OUTPUT_ARTIFACT_ID
    assert config.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert len(config.input_artifact_ids) == 6
    assert config.contract_hash == (
        "c53aa5fa1c6e818e7bc274f3251e283d4402d6a8fb793b539ea87e916abceed6"
    )
    assert config.protocol["partition_namespace"] == OOF_PARTITION_NAMESPACE
    assert config.protocol["selection_calibration_evaluation_case_disjoint"] is True
    assert config.protocol["strict_H_q_e_distinct"] is True
    assert config.routing["candidate_menu_top_k"] == 3
    assert config.routing["support_ranking_reference"] == "B"
    assert config.routing["support_prior_cases"] == 8.0
    assert config.routing["anchor_fallback"] == (
        "S_static_or_B_when_S_static_equals_B"
    )
    assert config.routing["donor_model_families"] == ["G", "R", "P"]
    assert config.routing["feature_alpha"] == 1.0
    assert config.routing["source_alpha"] == 4.0
    assert config.routing["query_alpha"] == 4.0
    assert config.routing["intercept_alpha"] == 0.25
    assert config.routing["calibration_alpha"] == 4.0
    assert config.routing["action_margin_z"] == 1.96
    assert config.routing[
        "residual_outcome_variance_in_action_standard_error"
    ] is False
    assert config.controls["method_ids"] == list(METHOD_IDS)
    assert config.controls["method_ids"] == list(runtime_constants.METHOD_IDS)
    assert config.controls["O_binary_role"].endswith("B_and_S_static")
    assert config.runtime["generation_devices"] == ["cuda:0", "cuda:1"]
    assert config.runtime["persistent_source_workers"] is True
    assert config.runtime["model_workers"] == 4
    assert config.runtime["model_threads_per_worker"] == 3
    assert config.runtime["bootstrap_workers"] == 4
    assert config.runtime["bootstrap_threads_per_worker"] == 3
    assert config.runtime["multiprocessing_start_method"] == "spawn"
    assert config.runtime["scratch_preference"][0] == (
        "/data/local/fixed_bank_multi_challenger_hierarchical_flip_router_v1"
    )
    assert config.runtime["two_fresh_process_validation_required"] is True
    assert config.runtime["generation_workers_per_device"] == 1
    assert config.runtime["source_workers_per_device"] == 1
    assert config.claim_boundary["publication_status"] == (
        "EXPLORATORY_CONSUMED_DATA_ONLY"
    )
    assert config.claim_boundary["routing_status"] == (
        "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"
    )


def test_amendment_is_direct_single_consumer_terminal_hash_chain() -> None:
    raw = AMENDMENT.read_bytes()
    payload = json.loads(raw)

    assert hashlib.sha256(raw).hexdigest() == EXPECTED_LEDGER_AMENDMENT_SHA256
    assert payload["parent_artifact_id"] == (
        "midogpp_uniform_b_test_consumption_ledger_v1"
    )
    assert payload["parent_sha256"] == (
        "8b16eae7bfdb5d20945e8ba3e02447ec74ab857adae441e493b2e37114feab16"
    )
    assert payload["authorized_consumer_experiment_ids"] == [EXPERIMENT_ID]
    assert payload["authorization_scope"] == AUTHORIZATION_SCOPE
    assert payload["partition_namespace"] == OOF_PARTITION_NAMESPACE
    assert payload["candidate_menu_top_k"] == 3
    assert payload["support_ranking_reference"] == "B"
    assert payload["every_donor_row_requires_H_q_e_distinct"] is True
    assert payload["target_support_labels_may_update_shared_model"] is False
    assert payload[
        "held_evaluation_label_mutation_must_leave_menus_models_calibrations_"
        "decisions_and_seals_unchanged"
    ] is True
    assert payload["O_binary_action_set"] == ["B", "S_static"]
    assert payload["terminal_oracles_available_before_evaluation_labels"] is False
    assert payload["fresh_evidence"] is False
    assert payload["generic_consumer_authorized"] is False
    assert payload["may_feed_another_stage90"] is False
    assert payload["may_feed_another_experiment"] is False


def test_registry_catalog_aliases_and_output_are_experiment_fenced() -> None:
    workspace = _workspace()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]

    assert experiment.stage == "90_oracles_and_diagnostics"
    assert experiment.status == "diagnostic"
    assert experiment.claim_scope == "diagnostic_only"
    assert experiment.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert experiment.run_recovery_strategy == (
        EXACT_EXISTING_SNAPSHOT_FIXED_BANK_MULTI_CHALLENGER_HIERARCHICAL_FLIP_ROUTER_V1
    )
    assert required_strategy_for_experiment(EXPERIMENT_ID) == (
        experiment.run_recovery_strategy
    )
    assert experiment.runner_argv[3:5] == (
        "cvae-diagnostics",
        "fixed-bank-multi-challenger-hierarchical-flip-router",
    )
    for artifact_id in (
        TEST_CACHE_ARTIFACT_ID,
        TEST_MANIFEST_ARTIFACT_ID,
        TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
        LEDGER_AMENDMENT_ARTIFACT_ID,
    ):
        artifact = workspace.artifacts[artifact_id]
        assert artifact.semantic_identities[
            "authorized_consumer_experiment_ids"
        ] == EXPERIMENT_ID
        assert artifact.semantic_identities["fresh_evidence"] == "false"
    assert output.canonical_path == (
        "artifacts/midogpp/90_oracles_and_diagnostics/"
        "uniform_b_v2_consumed_test_fixed_bank_multi_challenger_"
        "hierarchical_flip_router/v1"
    )
    assert output.semantic_identities["config_contract_hash"] == (
        "c53aa5fa1c6e818e"
    )
    assert output.semantic_identities["candidate_menu_top_k"] == "3"
    assert output.semantic_identities["primary_router_id"] == "R_multi"
    assert output.semantic_identities[
        "residual_outcome_variance_in_action_standard_error"
    ] == "false"
    assert output.semantic_identities["workstation_model_topology"] == (
        "4_workers_x_3_threads"
    )
    assert "oracle_and_diagnostic_evidence" in output.forbidden_reuse
    assert output.may_feed_recipe_selection is False
    assert output.may_feed_deployable_selection is False
    assert not any(
        OUTPUT_ARTIFACT_ID in candidate.input_artifact_ids
        for candidate in workspace.experiments.values()
    )


def test_workspace_recovery_strategy_lazily_dispatches_package_detector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from midogpp_thesis.cvae.diagnostics.fixed_bank_multi_challenger_hierarchical_flip_router import (
        recovery as package_recovery,
    )

    calls: list[Path] = []

    def detect(root: Path) -> bool:
        calls.append(root)
        return True

    monkeypatch.setattr(
        package_recovery,
        "detect_registered_multi_challenger_recovery",
        detect,
    )
    root = tmp_path / "exact-snapshot"

    assert detect_registered_exact_recovery(
        EXACT_EXISTING_SNAPSHOT_FIXED_BANK_MULTI_CHALLENGER_HIERARCHICAL_FLIP_ROUTER_V1,
        root,
    ) is True
    assert calls == [root]


def test_config_rejects_menu_model_and_claim_drift(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["routing"]["candidate_menu_top_k"] = 1
    drifted = tmp_path / "drifted.yaml"
    drifted.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="config section drifted: routing"):
        load_fixed_bank_multi_challenger_hierarchical_flip_router_config(drifted)

    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["routing"]["residual_outcome_variance_in_action_standard_error"] = True
    drifted.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="config section drifted: routing"):
        load_fixed_bank_multi_challenger_hierarchical_flip_router_config(drifted)

    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["claim_boundary"]["promotion_eligible"] = True
    drifted.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="config section drifted: claim_boundary"):
        load_fixed_bank_multi_challenger_hierarchical_flip_router_config(drifted)


def test_cli_registers_and_lazily_dispatches_multi_challenger_router(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    surface_id = "fixed-bank-multi-challenger-hierarchical-flip-router"
    parsed = cli.build_parser().parse_args(
        (
            surface_id,
            "--config",
            str(CONFIG),
            "--artifact-root",
            "/tmp/fixed-bank-multi-challenger-router-v1",
        )
    )
    assert parsed.surface == surface_id

    import midogpp_thesis.cvae.diagnostics.fixed_bank_multi_challenger_hierarchical_flip_router as surface

    sentinel = object()
    calls: list[tuple[object, Path]] = []
    monkeypatch.setattr(
        surface,
        "load_fixed_bank_multi_challenger_hierarchical_flip_router_config",
        lambda _: sentinel,
    )

    def _run(config: object, *, artifact_root: Path) -> Path:
        calls.append((config, artifact_root))
        return Path("/tmp/fixed-bank-multi-challenger-router-result-v1")

    monkeypatch.setattr(
        surface,
        "run_fixed_bank_multi_challenger_hierarchical_flip_router",
        _run,
    )
    assert cli.main(
        [
            surface_id,
            "--config",
            str(CONFIG),
            "--artifact-root",
            "/tmp/fixed-bank-multi-challenger-router-v1",
        ]
    ) == 0
    assert calls == [
        (sentinel, Path("/tmp/fixed-bank-multi-challenger-router-v1"))
    ]
    assert capsys.readouterr().out.strip() == (
        "/tmp/fixed-bank-multi-challenger-router-result-v1"
    )
