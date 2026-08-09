from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from midogpp_thesis.cvae.diagnostics import cli
from midogpp_thesis.cvae.diagnostics.fixed_bank_decision_audit.config import (
    load_fixed_bank_decision_audit_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_decision_audit.bundle import (
    REQUIRED_FILES,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_decision_audit.experiment_contracts import (
    EXPERIMENT_ID,
    EXPECTED_LEDGER_AMENDMENT_SHA256,
    EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256,
    INPUT_ARTIFACT_IDS,
    LEDGER_AMENDMENT_ARTIFACT_ID,
    OUTPUT_ARTIFACT_ID,
    TEST_CACHE_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
)
from midogpp_thesis.workspace.runtime import MidogppWorkspace


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs"
    / "uniform_b_v2_consumed_test_fixed_bank_decision_audit_v1.yaml"
)
AMENDMENT = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts"
    / "uniform_b_v2_consumed_test_fixed_bank_decision_audit_"
    "ledger_amendment_v1.json"
)


def _workspace() -> MidogppWorkspace:
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    return workspace


def test_config_freezes_known_bank_exact_only_decision_contract() -> None:
    config = load_fixed_bank_decision_audit_config(CONFIG)

    assert config.experiment_id == EXPERIMENT_ID
    assert config.output_artifact_id == OUTPUT_ARTIFACT_ID
    assert config.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert len(config.input_artifact_ids) == 7
    assert config.protocol["candidate_generalization"] == "known_fixed_bank_reuse"
    assert config.protocol["unseen_expert_transfer_claim"] is False
    assert config.protocol["strict_crossfit_training_row_count"] == 210
    assert config.protocol["one_shared_model_per_heldout_H_q"] is True
    assert config.protocol[
        "heldout_H_q_excluded_from_outer_query_and_candidate_roles"
    ] is True
    assert config.protocol["candidate_pool_excludes_H_and_q"] is True
    assert config.protocol["support_case_count_total"] == 72
    assert config.protocol["evaluation_case_count_total"] == 146
    assert config.protocol["support_labels_used"] is False
    assert config.protocol["prediction_and_feature_seals_before_test_labels"] is True
    assert config.features["metadata_similarity_role"] == (
        "persisted_descriptive_only_not_used_by_any_exact_or_smooth_family"
    )
    assert all(
        "metadata_similarity" not in predictors
        for predictors in config.model["exact_family_predictors"].values()
    )
    assert config.model["primary_r_family_id"] == "case_balanced_rich_exact"
    assert config.model["exact_prediction_row_count"] == 4536
    assert config.model["exact_fold_audit_row_count"] == 648
    assert config.model["smooth_prediction_row_count"] == 1512
    assert config.model["smooth_fold_audit_row_count"] == 216
    assert config.model["smooth_models_are_separate_from_exact_models"] is True
    assert config.model["smooth_may_feed_exact_coefficients_selection_or_gate"] is False
    assert config.evaluation["global_control_family_id"] == (
        "global_source_exact_control"
    )
    assert config.evaluation["method_rows_are_actions_or_policy"] is False
    assert config.evaluation["exact_B_abstention_when_gate_fails"] is True
    assert config.runtime["generation_devices"] == ["cuda:0", "cuda:1"]
    assert config.runtime["generation_workers_per_device"] == 1
    assert config.runtime["classifier_workers"] == 4
    assert config.runtime["classifier_threads_per_worker"] == 3
    assert config.runtime["parent_cuda_context_forbidden"] is True
    assert config.runtime["phase_disjoint_gpu_and_cpu_pools"] is True
    assert config.runtime["scratch_preference"][0] == (
        "/data/local/fixed_bank_decision_audit_v1"
    )
    for key in (
        "fresh_evidence",
        "unseen_expert_transfer_claim",
        "routing_quality_claimed",
        "target_actions_built",
        "action_selection_authorized",
        "policy_update_authorized",
        "promotion_eligible",
        "may_feed_stage60",
        "may_feed_stage70",
        "may_feed_another_stage90",
        "may_feed_recipe_selection",
        "may_feed_deployable_selection",
    ):
        assert config.claim_boundary[key] is False


def test_hash_chained_ledger_amendment_is_single_consumer() -> None:
    raw = AMENDMENT.read_bytes()
    payload = json.loads(raw)
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_LEDGER_AMENDMENT_SHA256
    assert payload["parent_sha256"] == EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
    assert payload["authorized_consumer_experiment_ids"] == [EXPERIMENT_ID]
    assert payload["authorization_scope"] == (
        "one_additional_terminal_posthoc_fixed_bank_diagnostic"
    )
    assert payload["fresh_evidence"] is False
    assert payload["generic_consumer_authorized"] is False
    assert payload["may_feed_another_stage90"] is False


def test_registry_catalog_and_aliases_are_experiment_fenced() -> None:
    workspace = _workspace()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]

    assert experiment.stage == "90_oracles_and_diagnostics"
    assert experiment.status == "diagnostic"
    assert experiment.claim_scope == "diagnostic_only"
    assert experiment.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert experiment.runner_argv[3:5] == (
        "cvae-diagnostics",
        "fixed-bank-decision-audit",
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
        "uniform_b_v2_consumed_test_fixed_bank_decision_audit/v1"
    )
    assert output.semantic_identities["candidate_generalization"] == (
        "known_fixed_bank_reuse"
    )
    assert output.semantic_identities["strict_crossfit_training_row_count"] == (
        "210"
    )
    assert output.semantic_identities[
        "smooth_may_affect_exact_fit_selection_gate_or_decision"
    ] == "false"
    assert output.semantic_identities["metadata_similarity_role"] == (
        "persisted_descriptive_only_not_used_by_any_exact_or_smooth_family"
    )
    assert output.required_files == REQUIRED_FILES
    assert "oracle_and_diagnostic_evidence" in output.forbidden_reuse
    assert output.may_feed_recipe_selection is False
    assert output.may_feed_deployable_selection is False


def test_cli_parser_and_lazy_dispatch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import midogpp_thesis.cvae.diagnostics.fixed_bank_decision_audit as surface

    parsed = cli.build_parser().parse_args(
        (
            "fixed-bank-decision-audit",
            "--config",
            str(CONFIG),
            "--artifact-root",
            "/tmp/fixed-bank-decision-audit",
        )
    )
    assert parsed.surface == "fixed-bank-decision-audit"
    sentinel = object()
    calls: list[tuple[object, Path]] = []
    monkeypatch.setattr(surface, "load_fixed_bank_decision_audit_config", lambda _: sentinel)

    def _run(config: object, *, artifact_root: Path) -> Path:
        calls.append((config, artifact_root))
        return Path("/tmp/fixed-bank-decision-result")

    monkeypatch.setattr(surface, "run_fixed_bank_decision_audit", _run)
    assert cli.main(
        [
            "fixed-bank-decision-audit",
            "--config",
            str(CONFIG),
            "--artifact-root",
            "/tmp/fixed-bank-decision-audit",
        ]
    ) == 0
    assert calls == [(sentinel, Path("/tmp/fixed-bank-decision-audit"))]
    assert capsys.readouterr().out.strip() == "/tmp/fixed-bank-decision-result"
