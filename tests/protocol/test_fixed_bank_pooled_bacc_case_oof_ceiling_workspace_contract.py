from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from midogpp_thesis.cvae.diagnostics import cli
from midogpp_thesis.cvae.diagnostics.fixed_bank_pooled_bacc_case_oof_ceiling.bundle import (
    REQUIRED_FILES,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_pooled_bacc_case_oof_ceiling.config import (
    load_fixed_bank_pooled_bacc_case_oof_ceiling_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_pooled_bacc_case_oof_ceiling.experiment_contracts import (
    EXPERIMENT_ID,
    EXPECTED_LEDGER_AMENDMENT_SHA256,
    EXPECTED_NULL_ACTION_COUNT,
    EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256,
    INPUT_ARTIFACT_IDS,
    LEDGER_AMENDMENT_ARTIFACT_ID,
    OUTPUT_ARTIFACT_ID,
    QUARANTINED_V1_EXPERIMENT_ID,
    QUARANTINED_V1_OUTPUT_ARTIFACT_ID,
    TEST_CACHE_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_pooled_bacc_case_oof_ceiling.inputs import (
    assert_input_fence,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.workspace.runtime import MidogppWorkspace


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs"
    / "uniform_b_v2_consumed_test_fixed_bank_pooled_bacc_case_oof_"
    "ceiling_v2.yaml"
)
AMENDMENT = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts"
    / "uniform_b_v2_consumed_test_fixed_bank_pooled_bacc_case_oof_"
    "ceiling_ledger_amendment_v2.json"
)


def _workspace() -> MidogppWorkspace:
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    return workspace


def test_config_freezes_pooled_exact_bacc_and_whole_case_cluster_math() -> None:
    config = load_fixed_bank_pooled_bacc_case_oof_ceiling_config(CONFIG)

    assert config.experiment_id == EXPERIMENT_ID
    assert config.output_artifact_id == OUTPUT_ARTIFACT_ID
    assert config.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert len(config.input_artifact_ids) == 6
    assert QUARANTINED_V1_OUTPUT_ARTIFACT_ID not in config.input_artifact_ids
    assert config.protocol["support_utility"] == "pooled_exact_bacc"
    assert config.protocol["uncertainty_unit"] == "paired_whole_case_cluster"
    assert config.protocol["case_sufficient_statistic_fields"] == [
        "n_positive",
        "true_positive",
        "n_negative",
        "true_negative",
    ]
    assert config.protocol["per_case_bacc_stored_or_used"] is False
    assert config.protocol["mixed_class_case_count"] == 213
    assert config.protocol["negative_only_case_count"] == 4
    assert config.protocol["positive_only_case_count"] == 1
    assert config.protocol[
        "stage70_prediction_scoring_or_policy_outputs_used"
    ] is False
    assert config.protocol["label_free_cache_lineage"] == (
        "stage70_derived_feature_cache_alias_only"
    )
    assert "stage70_outputs_used" not in config.protocol
    assert config.global_prior["loco_donor_count_per_candidate"] == 7
    assert config.global_prior["pairwise_alternative_count_when_G_H_is_B"] == 8
    assert config.global_prior["pairwise_alternative_count_when_G_H_is_source"] == 7
    assert config.global_prior["pairwise_donor_count_when_G_H_is_B"] == 7
    assert config.global_prior["pairwise_donor_count_when_G_H_is_source"] == 6
    assert "prior_strength" not in config.global_prior
    assert "prior_strength" not in config.posterior
    assert config.variance_floor == 1.0e-6
    assert config.confidence_multiplier == 1.96
    assert config.minimum_gain == 0.0
    assert config.tie_tolerance == 1.0e-12
    assert config.decision["expected_decision_count"] == 45
    assert config.decision["expected_permutation_null_action_count"] == 450_000
    assert EXPECTED_NULL_ACTION_COUNT == 450_000
    assert config.evaluation[
        "permutation_recomputes_same_pooled_bacc_cluster_posterior"
    ] is True
    assert config.evaluation["zero_headroom_normalized_regret"] == 0.0
    assert config.evaluation["zero_headroom_tolerance"] == 1.0e-12
    assert config.evaluation["zero_headroom_interpretation"] == (
        "no_routing_opportunity"
    )
    assert config.evaluation["permutation_baseline_B_fixed"] is True
    assert config.evaluation["permutation_eight_Hxe_multiset_preserved"] is True
    assert config.evaluation["permutation_primary_statistic"] == (
        "equal_center_R_minus_G_H"
    )
    assert config.evaluation["permutation_upper_tail_output_field"] == (
        "one_sided_p_value"
    )
    assert config.evaluation["permutation_lower_tail_output_field"] == (
        "lower_tail_p_value"
    )
    assert config.evaluation["permutation_two_sided_output_field"] == (
        "two_sided_p_value"
    )
    assert config.evaluation["permutation_upper_tail_p_value_formula"] == (
        "(1+count(null>=observed))/(K+1)"
    )
    assert config.evaluation["permutation_lower_tail_p_value_formula"] == (
        "(1+count(null<=observed))/(K+1)"
    )
    assert config.evaluation["permutation_two_sided_p_value_formula"] == (
        "min(1,2*min(upper,lower))"
    )
    assert config.evaluation["permutation_derangement_family"] == (
        "case_sha256_candidate_order_counter_splitmix64_nonzero_cyclic_"
        "shift_1_to_7_v1"
    )
    assert config.evaluation["permutation_candidate_order"] == (
        "case_specific_sha256_of_seed_fold_id_case_id_action_then_action"
    )
    assert config.evaluation["permutation_shift_generator"] == (
        "independent_counter_splitmix64_per_fold_case_permutation_index"
    )
    assert config.evaluation["permutation_shift_range_inclusive"] == [1, 7]
    assert config.evaluation["permutation_zero_shift_allowed"] is False
    assert config.evaluation["uniform_over_all_derangements"] is False
    assert config.runtime["generation_devices"] == ["cuda:0", "cuda:1"]
    assert config.runtime["classifier_workers"] == 4
    assert config.runtime["classifier_threads_per_worker"] == 3
    assert config.runtime["parent_cuda_context_forbidden"] is True
    assert config.runtime["scratch_preference"][0] == (
        "/data/local/fixed_bank_pooled_bacc_case_oof_ceiling_v2"
    )
    assert config.claim_boundary["terminal_decision"] == "DO_NOT_PROMOTE"
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
        "quarantined_v1_output_used",
        "quarantined_v1_scratch_or_checkpoint_used",
    ):
        assert config.claim_boundary[key] is False


def test_new_amendment_is_exactly_hash_chained_single_consumer_and_terminal() -> None:
    raw = AMENDMENT.read_bytes()
    payload = json.loads(raw)

    assert hashlib.sha256(raw).hexdigest() == EXPECTED_LEDGER_AMENDMENT_SHA256
    assert payload["parent_sha256"] == EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
    assert payload["authorized_consumer_experiment_ids"] == [EXPERIMENT_ID]
    assert payload["authorization_scope"] == (
        "one_additional_terminal_label_aware_pooled_bacc_case_oof_ceiling_v2"
    )
    assert payload["authorization_basis"] == (
        "explicit_user_authorization_for_one_additional_terminal_midogpp_"
        "consumed_test_v2_diagnostic"
    )
    assert payload["support_utility"] == "pooled_exact_bacc"
    assert payload["uncertainty_unit"] == "paired_whole_case_cluster"
    assert payload["zero_headroom_normalized_regret"] == 0.0
    assert payload["zero_headroom_tolerance"] == 1.0e-12
    assert payload["zero_headroom_interpretation"] == "no_routing_opportunity"
    assert payload["permutation_primary_statistic"] == "equal_center_R_minus_G_H"
    assert payload["permutation_upper_tail_p_value_formula"] == (
        "(1+count(null>=observed))/(K+1)"
    )
    assert payload["permutation_lower_tail_p_value_formula"] == (
        "(1+count(null<=observed))/(K+1)"
    )
    assert payload["permutation_two_sided_p_value_formula"] == (
        "min(1,2*min(upper,lower))"
    )
    assert payload["permutation_derangement_family"] == (
        "case_sha256_candidate_order_counter_splitmix64_nonzero_cyclic_"
        "shift_1_to_7_v1"
    )
    assert payload["permutation_shift_range_inclusive"] == [1, 7]
    assert payload["permutation_zero_shift_allowed"] is False
    assert payload["uniform_over_all_derangements"] is False
    assert payload["single_class_cases_retained_as_sufficient_statistics"] is True
    assert payload["v1_output_used"] is False
    assert payload["v1_scratch_or_checkpoint_used"] is False
    assert payload["fresh_evidence"] is False
    assert payload["generic_consumer_authorized"] is False
    assert payload["may_feed_another_stage90"] is False


def test_config_rejects_any_utility_or_uncertainty_drift(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["protocol"]["support_utility"] = "equal_weight_per_case_bacc"
    drifted = tmp_path / "drifted.yaml"
    drifted.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="protocol drifted"):
        load_fixed_bank_pooled_bacc_case_oof_ceiling_config(drifted)

    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["protocol"]["uncertainty_unit"] = "row"
    drifted.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="protocol drifted"):
        load_fixed_bank_pooled_bacc_case_oof_ceiling_config(drifted)


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
        "fixed-bank-pooled-bacc-case-oof-ceiling",
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
        "uniform_b_v2_consumed_test_fixed_bank_pooled_bacc_case_oof_ceiling/v2"
    )
    assert output.semantic_identities["support_utility"] == "pooled_exact_bacc"
    assert output.semantic_identities["uncertainty_unit"] == (
        "paired_whole_case_cluster"
    )
    assert output.semantic_identities["single_class_cases_retained"] == "true"
    assert output.semantic_identities["per_case_bacc_stored_or_used"] == "false"
    assert output.semantic_identities[
        "stage70_prediction_scoring_or_policy_outputs_used"
    ] == "false"
    assert output.semantic_identities["label_free_cache_lineage"] == (
        "stage70_derived_feature_cache_alias_only"
    )
    assert output.semantic_identities[
        "all_observed_and_null_actions_sealed_before_evaluation_labels"
    ] == "true"
    assert output.semantic_identities["permutation_primary_statistic"] == (
        "equal_center_R_minus_G_H"
    )
    assert output.semantic_identities["permutation_derangement_family"] == (
        "case_sha256_candidate_order_counter_splitmix64_nonzero_cyclic_"
        "shift_1_to_7_v1"
    )
    assert output.semantic_identities["permutation_shift_range_inclusive"] == "1|7"
    assert output.semantic_identities["uniform_over_all_derangements"] == "false"
    assert output.semantic_identities["quarantined_v1_output_used"] == "false"
    assert output.semantic_identities[
        "quarantined_v1_scratch_or_checkpoint_used"
    ] == "false"
    assert output.required_files == REQUIRED_FILES
    assert len(REQUIRED_FILES) == 44
    assert not any("per_case_bacc" in member for member in REQUIRED_FILES)
    assert "oracle_and_diagnostic_evidence" in output.forbidden_reuse
    assert output.may_feed_recipe_selection is False
    assert output.may_feed_deployable_selection is False
    assert not any(
        OUTPUT_ARTIFACT_ID in candidate.input_artifact_ids
        for candidate in workspace.experiments.values()
    )


def test_input_fence_rejects_quarantined_v1_and_prior_outputs() -> None:
    config = load_fixed_bank_pooled_bacc_case_oof_ceiling_config(CONFIG)
    assert_input_fence(config)

    with pytest.raises(ProtocolError, match="cannot consume v1"):
        assert_input_fence(
            replace(
                config,
                test_cache_root=Path(
                    "artifacts/midogpp/90_oracles_and_diagnostics/"
                    "uniform_b_v2_consumed_test_fixed_bank_label_aware_"
                    "case_oof_ceiling/v1"
                ),
            )
        )
    with pytest.raises(ProtocolError, match="cannot consume v1"):
        assert_input_fence(
            replace(
                config,
                expert_bank_root=Path(QUARANTINED_V1_EXPERIMENT_ID),
            )
        )


def test_cli_parser_exposes_only_the_new_v2_surface() -> None:
    parsed = cli.build_parser().parse_args(
        (
            "fixed-bank-pooled-bacc-case-oof-ceiling",
            "--config",
            str(CONFIG),
            "--artifact-root",
            "/tmp/fixed-bank-pooled-bacc-case-oof-ceiling-v2",
        )
    )
    assert parsed.surface == "fixed-bank-pooled-bacc-case-oof-ceiling"


def test_cli_lazy_dispatches_to_the_v2_runner(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import midogpp_thesis.cvae.diagnostics.fixed_bank_pooled_bacc_case_oof_ceiling as surface

    sentinel = object()
    calls: list[tuple[object, Path]] = []
    monkeypatch.setattr(
        surface,
        "load_fixed_bank_pooled_bacc_case_oof_ceiling_config",
        lambda _: sentinel,
    )

    def _run(config: object, *, artifact_root: Path) -> Path:
        calls.append((config, artifact_root))
        return Path("/tmp/fixed-bank-pooled-bacc-result-v2")

    monkeypatch.setattr(
        surface,
        "run_fixed_bank_pooled_bacc_case_oof_ceiling",
        _run,
    )
    assert cli.main(
        [
            "fixed-bank-pooled-bacc-case-oof-ceiling",
            "--config",
            str(CONFIG),
            "--artifact-root",
            "/tmp/fixed-bank-pooled-bacc-root-v2",
        ]
    ) == 0
    assert calls == [(sentinel, Path("/tmp/fixed-bank-pooled-bacc-root-v2"))]
    assert capsys.readouterr().out.strip() == (
        "/tmp/fixed-bank-pooled-bacc-result-v2"
    )
