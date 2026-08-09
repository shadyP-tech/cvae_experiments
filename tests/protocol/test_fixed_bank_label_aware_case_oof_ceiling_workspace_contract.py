from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from midogpp_thesis.cvae.diagnostics import cli
from midogpp_thesis.cvae.diagnostics.fixed_bank_label_aware_case_oof_ceiling.bundle import (
    REQUIRED_FILES,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_label_aware_case_oof_ceiling.config import (
    load_fixed_bank_label_aware_case_oof_ceiling_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_label_aware_case_oof_ceiling.experiment_contracts import (
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
    / "uniform_b_v2_consumed_test_fixed_bank_label_aware_case_oof_ceiling_v1.yaml"
)
AMENDMENT = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts"
    / "uniform_b_v2_consumed_test_fixed_bank_label_aware_case_oof_"
    "ceiling_ledger_amendment_v1.json"
)


def _workspace() -> MidogppWorkspace:
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    return workspace


def test_config_freezes_direct_target_label_aware_oof_ceiling() -> None:
    config = load_fixed_bank_label_aware_case_oof_ceiling_config(CONFIG)

    assert config.experiment_id == EXPERIMENT_ID
    assert config.output_artifact_id == OUTPUT_ARTIFACT_ID
    assert config.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert len(config.input_artifact_ids) == 6
    assert not any("metadata" in value for value in config.input_artifact_ids)
    assert config.protocol["target_geometry"] == (
        "direct_H_with_B_and_eight_Hxe_actions"
    )
    assert config.protocol["candidate_pool_excludes_target_H"] is True
    assert config.protocol["oof_fold_count"] == 5
    assert config.protocol["center_fold_decision_count"] == 45
    assert config.protocol["target_probability_cell_count"] == 729
    assert config.protocol["each_case_evaluated_exactly_once"] is True
    assert config.protocol[
        "heldout_fold_absent_from_its_support_and_decision_fit"
    ] is True
    assert config.protocol["global_target_probability_seal_before_any_label_access"] is True
    assert config.protocol["support_labels_used"] is True
    assert config.protocol["support_labels_may_update_shared_model"] is False
    assert config.protocol[
        "evaluation_role_labels_inaccessible_until_all_decisions_sealed"
    ] is True
    assert config.protocol[
        "permutation_null_actions_sealed_before_evaluation_role_labels"
    ] is True
    assert config.global_prior["family"] == "label_derived_LOCO_global_prior"
    assert config.global_prior["G_H_uses_other_consumed_test_centers"] is True
    assert config.global_prior["H_labels_used_in_G_H"] is False
    assert config.global_prior["G_H_shared_across_H"] is False
    assert config.global_prior["G_H_hyperparameters_fixed_prelabel"] is True
    assert config.global_prior["G_H_sealed_before_H_support_access"] is True
    assert config.global_prior["other_center_contribution_unit"] == (
        "equal_weight_per_target_center"
    )
    assert config.global_prior["prior_strength"] == 8.0
    assert config.posterior["prior_strength"] == 8.0
    assert config.posterior["support_observation"] == (
        "exact_support_bacc_gain_vs_G_H"
    )
    assert config.posterior["variance_floor"] == 1.0e-6
    assert config.posterior["confidence_multiplier"] == 1.96
    assert config.posterior["minimum_gain"] == 0.0
    assert config.posterior["no_shared_target_label_fit"] is True
    assert config.decision["expected_decision_count"] == 45
    assert config.decision["evaluation_labels_may_affect_decision"] is False
    assert config.decision[
        "all_permutation_null_decisions_sealed_before_evaluation_capability"
    ] is True
    assert config.evaluation["permutation_count"] == 10_000
    assert config.evaluation["permutation_unit"] == (
        "candidate_source_label_derangement_within_H_fold_and_support_case"
    )
    assert config.evaluation["permutation_baseline_B_fixed"] is True
    assert config.evaluation["permutation_eight_Hxe_multiset_preserved"] is True
    assert config.evaluation["permutation_evaluation_donors_used"] is False
    assert config.evaluation[
        "permutation_actions_sealed_before_evaluation_labels"
    ] is True
    assert config.evaluation["exact_metric_only_may_enter_gates"] is True
    assert config.evaluation[
        "smooth_metric_may_affect_fit_selection_gate_or_decision"
    ] is False
    assert config.runtime["generation_devices"] == ["cuda:0", "cuda:1"]
    assert config.runtime["classifier_workers"] == 4
    assert config.runtime["classifier_threads_per_worker"] == 3
    assert config.runtime["parent_cuda_context_forbidden"] is True
    assert config.runtime["tf32_enabled"] is False
    assert config.runtime["amp_enabled"] is False
    assert config.runtime["target_unique_classifier_fit_count"] == 729
    assert config.runtime["scratch_preference"][0] == (
        "/data/local/fixed_bank_label_aware_case_oof_ceiling_v1"
    )
    for key in (
        "fresh_evidence",
        "fresh_confirmation",
        "routing_quality_claimed",
        "target_performance_claimed",
        "source_expert_updated",
        "target_expert_used",
        "shared_model_updated_with_target_labels",
        "action_selection_authorized",
        "policy_update_authorized",
        "promotion_eligible",
        "may_feed_stage50",
        "may_feed_stage60",
        "may_feed_stage70",
        "may_feed_another_stage90",
        "may_feed_recipe_selection",
        "may_feed_deployable_selection",
        "previous_stage90_outputs_used",
    ):
        assert config.claim_boundary[key] is False


def test_amendment_is_hash_chained_single_consumer_and_terminal() -> None:
    raw = AMENDMENT.read_bytes()
    payload = json.loads(raw)
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_LEDGER_AMENDMENT_SHA256
    assert payload["parent_sha256"] == EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
    assert payload["authorized_consumer_experiment_ids"] == [EXPERIMENT_ID]
    assert payload["authorization_scope"] == (
        "one_additional_terminal_label_aware_case_oof_ceiling"
    )
    assert payload["authorization_basis"] == (
        "explicit_user_request_to_reuse_consumed_test_set_for_this_diagnostic"
    )
    assert payload["fresh_evidence"] is False
    assert payload["support_labels_used"] is True
    assert payload["H_labels_used_in_G_H"] is False
    assert payload["G_H_shared_across_H"] is False
    assert payload["G_H_sealed_before_H_support_access"] is True
    for key in (
        "posterior_family_fixed_before_support_labels",
        "prior_hyperparameters_fixed_before_support_labels",
        "decision_thresholds_fixed_before_support_labels",
        "tie_policy_fixed_before_support_labels",
        "permutation_plan_fixed_before_support_labels",
    ):
        assert payload[key] is True
    assert payload["permutation_unit"] == (
        "candidate_source_label_derangement_within_H_fold_and_support_case"
    )
    assert payload["permutation_baseline_B_fixed"] is True
    assert payload["permutation_eight_Hxe_multiset_preserved"] is True
    assert payload["permutation_evaluation_donors_used"] is False
    assert payload[
        "all_permutation_null_decisions_sealed_before_evaluation_labels"
    ] is True
    assert payload["evaluation_labels_opened_only_after_all_fold_decisions_sealed"] is True
    assert payload["source_expert_updated"] is False
    assert payload["target_expert_used"] is False
    assert payload["shared_model_updated_with_target_labels"] is False
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
        "fixed-bank-label-aware-case-oof-ceiling",
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
        "uniform_b_v2_consumed_test_fixed_bank_label_aware_case_oof_ceiling/v1"
    )
    assert output.semantic_identities["architecture_family"] == (
        "fixed_bank_label_aware_case_oof_ceiling_v1"
    )
    assert output.semantic_identities["label_derived_LOCO_global_prior"] == "true"
    assert output.semantic_identities["H_labels_used_in_G_H"] == "false"
    assert output.semantic_identities["evaluation_case_count"] == "218"
    assert output.semantic_identities["whole_case_oof_fold_count"] == "5"
    assert output.semantic_identities["all_fold_decision_count"] == "45"
    assert output.semantic_identities["target_probability_cell_count"] == "729"
    assert output.semantic_identities[
        "smooth_metric_may_affect_fit_selection_gate_or_decision"
    ] == "false"
    assert output.semantic_identities["permutation_unit"] == (
        "candidate_source_label_derangement_within_H_fold_and_support_case"
    )
    assert output.semantic_identities["permutation_baseline_B_fixed"] == "true"
    assert output.semantic_identities["permutation_evaluation_donors_used"] == (
        "false"
    )
    assert output.semantic_identities["permutation_decision_tie_break"] == (
        "lexicographic_action_id_no_evaluation_utility_access"
    )
    assert output.semantic_identities[
        "evaluation_utility_used_for_permutation_tie_break"
    ] == "false"
    assert output.semantic_identities["action_selection_metric_row_count"] == "20"
    assert output.semantic_identities[
        "all_permutation_null_decisions_sealed_before_evaluation_labels"
    ] == "true"
    assert output.required_files == REQUIRED_FILES
    assert "oracle_and_diagnostic_evidence" in output.forbidden_reuse
    assert output.may_feed_recipe_selection is False
    assert output.may_feed_deployable_selection is False


def test_cli_parser_and_lazy_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import midogpp_thesis.cvae.diagnostics.fixed_bank_label_aware_case_oof_ceiling as surface

    parsed = cli.build_parser().parse_args(
        (
            "fixed-bank-label-aware-case-oof-ceiling",
            "--config",
            str(CONFIG),
            "--artifact-root",
            "/tmp/fixed-bank-label-aware-case-oof-ceiling",
        )
    )
    assert parsed.surface == "fixed-bank-label-aware-case-oof-ceiling"
    sentinel = object()
    calls: list[tuple[object, Path]] = []
    monkeypatch.setattr(
        surface,
        "load_fixed_bank_label_aware_case_oof_ceiling_config",
        lambda _: sentinel,
    )

    def _run(config: object, *, artifact_root: Path) -> Path:
        calls.append((config, artifact_root))
        return Path("/tmp/fixed-bank-label-aware-case-oof-result")

    monkeypatch.setattr(
        surface,
        "run_fixed_bank_label_aware_case_oof_ceiling",
        _run,
    )
    assert cli.main(
        [
            "fixed-bank-label-aware-case-oof-ceiling",
            "--config",
            str(CONFIG),
            "--artifact-root",
            "/tmp/fixed-bank-label-aware-case-oof-ceiling",
        ]
    ) == 0
    assert calls == [
        (sentinel, Path("/tmp/fixed-bank-label-aware-case-oof-ceiling"))
    ]
    assert capsys.readouterr().out.strip() == (
        "/tmp/fixed-bank-label-aware-case-oof-result"
    )
