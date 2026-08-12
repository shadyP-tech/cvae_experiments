from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from midogpp_thesis.cvae.diagnostics import cli
from midogpp_thesis.cvae.diagnostics.fixed_bank_labeled_support_case_conditional_flip_router.config import (
    load_fixed_bank_labeled_support_case_conditional_flip_router_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_labeled_support_case_conditional_flip_router.constants import (
    FEATURE_NAMES,
    METHOD_IDS,
    OOF_PARTITION_NAMESPACE,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_labeled_support_case_conditional_flip_router.experiment_contracts import (
    AUTHORIZATION_SCOPE,
    EXPERIMENT_ID,
    EXPECTED_LEDGER_AMENDMENT_SHA256,
    INPUT_ARTIFACT_IDS,
    LEDGER_AMENDMENT_ARTIFACT_ID,
    OUTPUT_ARTIFACT_ID,
    TEST_CACHE_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.workspace.runtime import MidogppWorkspace


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs"
    / "uniform_b_v2_consumed_test_fixed_bank_labeled_support_case_conditional_"
    "flip_router_v1.yaml"
)
AMENDMENT = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts"
    / "uniform_b_v2_consumed_test_fixed_bank_labeled_support_case_conditional_"
    "flip_router_ledger_amendment_v1.json"
)


def _workspace() -> MidogppWorkspace:
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    return workspace


def test_config_freezes_case_roles_science_and_workstation_topology() -> None:
    config = load_fixed_bank_labeled_support_case_conditional_flip_router_config(
        CONFIG
    )

    assert config.experiment_id == EXPERIMENT_ID
    assert config.output_artifact_id == OUTPUT_ARTIFACT_ID
    assert config.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert len(config.input_artifact_ids) == 6
    assert config.contract_hash == (
        "4b8567a6d5e6f13f1d18abf7bd8c8e219593555d411f66f8fb08f6fc79a605dc"
    )
    assert config.protocol["partition_seed"] == 90902026
    assert config.protocol["partition_namespace"] == OOF_PARTITION_NAMESPACE
    assert config.protocol["role_rotation"] == (
        "eval=f_calibration=(f+1)%5_selection=remaining_three"
    )
    assert config.flip_features["feature_names"] == list(FEATURE_NAMES)
    assert config.flip_features["no_B_vs_U_flip_feature"] is True
    assert config.routing["static_challenger_candidates"] == (
        "eight_A1_source_actions_only"
    )
    assert config.routing["static_challenger_fallback"] == (
        "B_when_no_positive_authorized_A1"
    )
    assert config.routing["ridge_alpha"] == 1.0
    assert config.routing["heuristic_score_multiplier"] == 1.96
    assert config.routing["primary_router"] == "F_S"
    assert config.controls["method_ids"] == list(METHOD_IDS)
    assert config.evaluation["primary_contrasts"] == [
        "F_S-B",
        "F_S-U",
        "F_S-F_G",
        "F_S-F_P",
        "F_S-S_static",
    ]
    assert config.evaluation["diagnostic_recoverability_gate"] == {
        "gate_id": "all_primary_contrast_outer_center_lcbs_positive_v1",
        "lcb_field": "one_sided_95_lcb",
        "threshold": 0.0,
        "comparison": "strictly_greater_than",
        "required_contrast_count": 5,
        "pass_status": "PASS",
        "fail_status": "FAIL",
        "diagnostic_only": True,
    }
    assert config.runtime["generation_devices"] == ["cuda:0", "cuda:1"]
    assert config.runtime["classifier_workers"] == 4
    assert config.runtime["classifier_threads_per_worker"] == 3
    assert config.runtime["multiprocessing_start_method"] == "spawn"
    assert config.runtime["target_probability_cell_count"] == 810
    assert config.runtime["scratch_preference"][0] == (
        "/data/local/fixed_bank_labeled_support_case_conditional_flip_router_v1"
    )
    assert config.claim_boundary["fresh_evidence"] is False
    assert config.claim_boundary[
        "may_feed_stage50_stage60_stage70_or_another_experiment"
    ] is False


def test_amendment_is_direct_single_consumer_hash_chain_and_terminal() -> None:
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
    assert payload["static_source_selection_candidate_set"] == (
        "eight_frozen_A1_source_actions_only"
    )
    assert payload["B_and_U_are_fixed_controls_not_static_source_candidates"]
    assert payload["B_vs_U_flip_feature_or_gate_used"] is False
    assert payload["routing_identification_metrics"][-1] == "fold_stability"
    assert payload["diagnostic_recoverability_gate"]["diagnostic_only"] is True
    assert payload["diagnostic_recoverability_gate"]["routing_success_claimed"] is False
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
    assert experiment.runner_argv[3:5] == (
        "cvae-diagnostics",
        "fixed-bank-labeled-support-case-conditional-flip-router",
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
        "uniform_b_v2_consumed_test_fixed_bank_labeled_support_case_"
        "conditional_flip_router/v1"
    )
    assert output.semantic_identities["config_contract_hash"] == (
        "4b8567a6d5e6f13f"
    )
    assert output.semantic_identities["action_count_per_target"] == "10"
    assert output.semantic_identities["target_probability_cell_count"] == "810"
    assert output.semantic_identities["primary_heuristic_router_id"] == "F_S"
    assert output.semantic_identities[
        "heuristic_prediction_bound_descriptive_only"
    ] == "true"
    assert "oracle_and_diagnostic_evidence" in output.forbidden_reuse
    assert output.may_feed_recipe_selection is False
    assert output.may_feed_deployable_selection is False
    assert not any(
        OUTPUT_ARTIFACT_ID in candidate.input_artifact_ids
        for candidate in workspace.experiments.values()
    )


def test_config_rejects_role_feature_and_routing_drift(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["protocol"]["role_rotation"] = "leaky_same_fold"
    drifted = tmp_path / "drifted.yaml"
    drifted.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="config section drifted: protocol"):
        load_fixed_bank_labeled_support_case_conditional_flip_router_config(
            drifted
        )

    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["flip_features"]["feature_names"].append("target_label")
    drifted.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="config section drifted: flip_features"):
        load_fixed_bank_labeled_support_case_conditional_flip_router_config(
            drifted
        )

    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["routing"]["ridge_alpha_selection_used"] = True
    drifted.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="config section drifted: routing"):
        load_fixed_bank_labeled_support_case_conditional_flip_router_config(
            drifted
        )


def test_cli_registers_and_lazily_dispatches_flip_router(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parsed = cli.build_parser().parse_args(
        (
            "fixed-bank-labeled-support-case-conditional-flip-router",
            "--config",
            str(CONFIG),
            "--artifact-root",
            "/tmp/fixed-bank-labeled-support-flip-router-v1",
        )
    )
    assert parsed.surface == (
        "fixed-bank-labeled-support-case-conditional-flip-router"
    )

    import midogpp_thesis.cvae.diagnostics.fixed_bank_labeled_support_case_conditional_flip_router as surface

    sentinel = object()
    calls: list[tuple[object, Path]] = []
    monkeypatch.setattr(
        surface,
        "load_fixed_bank_labeled_support_case_conditional_flip_router_config",
        lambda _: sentinel,
    )

    def _run(config: object, *, artifact_root: Path) -> Path:
        calls.append((config, artifact_root))
        return Path("/tmp/fixed-bank-labeled-support-flip-router-result-v1")

    monkeypatch.setattr(
        surface,
        "run_fixed_bank_labeled_support_case_conditional_flip_router",
        _run,
    )
    assert cli.main(
        [
            "fixed-bank-labeled-support-case-conditional-flip-router",
            "--config",
            str(CONFIG),
            "--artifact-root",
            "/tmp/fixed-bank-labeled-support-flip-router-v1",
        ]
    ) == 0
    assert calls == [
        (sentinel, Path("/tmp/fixed-bank-labeled-support-flip-router-v1"))
    ]
    assert capsys.readouterr().out.strip() == (
        "/tmp/fixed-bank-labeled-support-flip-router-result-v1"
    )
