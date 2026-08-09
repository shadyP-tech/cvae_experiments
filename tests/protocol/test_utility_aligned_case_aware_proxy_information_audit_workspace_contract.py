from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from midogpp_thesis.cvae.diagnostics import cli
from midogpp_thesis.cvae.diagnostics.utility_aligned_case_aware_proxy_information_audit.config import (
    load_utility_aligned_case_aware_proxy_information_audit_config,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_case_aware_proxy_information_audit.bundle import (
    REQUIRED_FILES,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_case_aware_proxy_information_audit.experiment_contracts import (
    EXPERIMENT_ID,
    INPUT_ARTIFACT_IDS,
    METADATA_PROFILE_ARTIFACT_ID,
    OUTPUT_ARTIFACT_ID,
    TEST_CACHE_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_case_aware_proxy_information_audit.inputs import (
    assert_input_fence,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.workspace.runtime import MidogppWorkspace


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs"
    / "uniform_b_v2_consumed_test_utility_aligned_case_aware_"
    "proxy_information_audit_v1.yaml"
)
UNDERLYING_TEST_CACHE_ID = (
    "midogpp_virchow2_uniform_b_v2_descriptive_test_cache_seed42"
)
UNDERLYING_MANIFEST_ID = "midogpp_dataset_contract_annotation_patch_v1"
LEGACY_CACHE_ALIAS = (
    "midogpp_stage90_utility_aligned_ensemble_endpoint_proxy_information_"
    "audit_validation_cache_v1"
)
LEGACY_MANIFEST_ALIAS = (
    "midogpp_stage90_utility_aligned_ensemble_endpoint_proxy_information_"
    "audit_validation_manifest_v1"
)
FORBIDDEN_INPUT_STAGES = {
    "60_routing_and_composition",
    "70_frozen_policy_downstream",
    "90_oracles_and_diagnostics",
}


def _workspace() -> MidogppWorkspace:
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    return workspace


def test_config_freezes_consumed_test_case_aware_contract() -> None:
    config = load_utility_aligned_case_aware_proxy_information_audit_config(CONFIG)
    assert_input_fence(config)

    assert config.experiment_id == EXPERIMENT_ID
    assert config.output_artifact_id == OUTPUT_ARTIFACT_ID
    assert config.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert len(config.input_artifact_ids) == 6
    assert config.input_artifact_ids[-1] == TEST_CONSUMPTION_LEDGER_ARTIFACT_ID
    assert config.input_artifact_ids[4] == METADATA_PROFILE_ARTIFACT_ID
    assert LEGACY_CACHE_ALIAS not in config.input_artifact_ids
    assert LEGACY_MANIFEST_ALIAS not in config.input_artifact_ids
    assert config.evaluation_split == "test"
    assert config.expected_manifest_sha256 == (
        "db661ac7e3dbafde8e283528de6706ab35f2c26629b389706c4504e458cc5869"
    )
    assert config.fixed_support_case_count_per_center == 8
    assert config.protocol["eligible_test_case_count"] == 218
    assert config.protocol["support_case_count_total"] == 72
    assert config.protocol["evaluation_case_count_total"] == 146
    assert config.protocol["evaluation_case_counts_by_center"] == {
        "0": 15,
        "1": 12,
        "2": 16,
        "3": 31,
        "5": 15,
        "6": 15,
        "7": 13,
        "8": 14,
        "9": 15,
    }
    assert config.protocol["support_split_seed"] == 20260809
    assert config.protocol["strict_crossfit_training_row_count"] == 120
    assert config.protocol["primary_response_name"] == "exact_bacc_delta"
    assert config.protocol["diagnostic_response_name"] == "smooth_bacc_delta"
    assert config.protocol[
        "diagnostic_response_may_feed_fit_selection_or_gate"
    ] is False
    assert config.protocol["support_labels_used"] is False
    assert config.protocol["development_predictions_sealed_before_test_labels"] is True
    assert config.protocol["test_labels_opened_only_after_global_prediction_seal"] is True
    assert config.protocol["test_labels_construct_postseal_response_rows"] is True
    assert config.protocol[
        "label_derived_responses_feed_strict_crossfit_diagnostic_models"
    ] is True
    assert config.protocol["test_labels_used_for_feature_construction"] is False
    assert config.protocol["test_labels_used_for_policy_or_action_fit"] is False
    for key in (
        "stage60_outputs_used",
        "stage70_prediction_scoring_or_policy_outputs_used",
        "previous_stage90_outputs_used",
    ):
        assert config.protocol[key] is False

    assert config.model["ridge_alpha"] == 1.0
    assert config.model["maximum_predictors_per_family"] == 3
    assert config.model["diagnostic_response_crossfit_role"] == (
        "separately_fit_descriptive_models_only"
    )
    assert config.model["diagnostic_response_may_feed_primary_model_or_gate"] is False
    assert config.model["hyperparameter_selection"] == (
        "none_predeclared_before_labels"
    )
    assert config.model["family_ids"] == [
        "equal_union_null",
        "metadata_only_control",
        "pooled_row_weighted_shift_control",
        "case_balanced_shift_compact",
        "case_balanced_rich_compact",
        "case_aware_hybrid_compact",
        "cyclic_directional_permutation_control",
    ]
    assert config.model["family_predictors"]["case_balanced_shift_compact"] == [
        "equal_case_abs_shift",
        "case_abs_shift_sd",
        "equal_case_signed_margin",
    ]
    assert config.proxy_features["primitive_formulas"]["case_balanced_log_mmd"] == (
        "mean_support_cases(mean_exact9(log1p(linear_kernel_mmd2("
        "case_embedding_mean, generated_stream_mean))))"
    )
    assert config.runtime["generation_devices"] == ["cuda:0", "cuda:1"]
    assert config.runtime["source_workers_per_device"] == 1
    assert config.runtime["classifier_workers"] == 4
    assert config.runtime["classifier_threads_per_worker"] == 3
    assert config.runtime["multiprocessing_start_method"] == "spawn"
    assert config.runtime["tf32_enabled"] is False
    assert config.runtime["amp_enabled"] is False
    assert config.runtime["source_job_count"] == 27
    assert config.runtime["source_stream_count"] == 81
    assert config.runtime["development_coarse_task_count"] == 648
    assert config.runtime["development_classifier_fit_count"] == 5184
    assert config.runtime["scratch_preference"] == [
        "/data/local",
        "artifact_parent",
    ]
    assert config.claim_boundary["publication_status"] == (
        "EXPLORATORY_CONSUMED_DATA_ONLY"
    )
    assert config.claim_boundary["consumed_test_data"] is True
    assert config.claim_boundary["user_authorized_consumed_test_repurposing"] is True
    assert config.claim_boundary["fresh_evidence"] is False
    assert config.claim_boundary["test_labels_construct_postseal_response_rows"] is True
    assert config.claim_boundary[
        "label_derived_responses_feed_strict_crossfit_diagnostic_models"
    ] is True
    assert config.claim_boundary["test_labels_used_for_feature_construction"] is False
    assert config.claim_boundary["test_labels_used_for_policy_or_action_fit"] is False
    for key in (
        "policy_update_authorized",
        "action_selection_authorized",
        "promotion_eligible",
        "may_feed_stage60",
        "may_feed_stage70",
        "may_feed_recipe_selection",
        "may_feed_deployable_selection",
        "may_feed_another_stage90_experiment",
    ):
        assert config.claim_boundary[key] is False


def test_config_facade_reexports_leaf_payload_contracts() -> None:
    from midogpp_thesis.cvae.diagnostics.utility_aligned_case_aware_proxy_information_audit import config as facade
    from midogpp_thesis.cvae.diagnostics.utility_aligned_case_aware_proxy_information_audit import config_payloads as leaf

    config = facade.load_utility_aligned_case_aware_proxy_information_audit_config(
        CONFIG
    )
    assert facade.CLASSIFIER == leaf.CLASSIFIER == config.classifier
    for name, observed in (
        ("canonical_protocol_payload", config.protocol),
        ("canonical_proxy_features_payload", config.proxy_features),
        ("canonical_model_payload", config.model),
        ("canonical_evaluation_payload", config.evaluation),
        ("canonical_runtime_payload", config.runtime),
        ("canonical_claim_boundary_payload", config.claim_boundary),
    ):
        assert getattr(facade, name)() == getattr(leaf, name)()
        assert getattr(facade, name)() == observed


@pytest.mark.parametrize(
    ("field", "legacy_alias"),
    (
        ("test_cache_root", LEGACY_CACHE_ALIAS),
        ("test_manifest_path", LEGACY_MANIFEST_ALIAS),
        ("test_cache_artifact_id", LEGACY_CACHE_ALIAS),
        ("test_manifest_artifact_id", LEGACY_MANIFEST_ALIAS),
    ),
)
def test_config_rejects_consumed_validation_v1_aliases(
    tmp_path: Path, field: str, legacy_alias: str
) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if field.endswith("_root"):
        payload["inputs"][field] = f"artifact://{legacy_alias}"
    elif field.endswith("_path"):
        payload["inputs"][field] = f"artifact://{legacy_alias}/manifest.csv"
    else:
        payload["inputs"][field] = legacy_alias
    mutated = tmp_path / "legacy-alias.yaml"
    mutated.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ProtocolError):
        load_utility_aligned_case_aware_proxy_information_audit_config(mutated)


def test_registry_has_exact_six_input_terminal_fence() -> None:
    workspace = _workspace()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]

    assert experiment.stage == "90_oracles_and_diagnostics"
    assert experiment.status == "diagnostic"
    assert experiment.claim_scope == "diagnostic_only"
    assert experiment.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert len(experiment.input_artifact_ids) == 6
    assert not FORBIDDEN_INPUT_STAGES.intersection(
        workspace.artifacts[artifact_id].stage
        for artifact_id in experiment.input_artifact_ids
    )
    assert experiment.runner_argv == (
        "{python}",
        "-m",
        "midogpp_thesis",
        "cvae-diagnostics",
        "utility-aligned-case-aware-proxy-information-audit",
        "--config",
        "{resolved_config}",
        "--artifact-root",
        f"output://{OUTPUT_ARTIFACT_ID}",
    )
    assert output.stage == "90_oracles_and_diagnostics"
    assert output.claim_scope == "diagnostic_only"
    assert output.canonical_path == (
        "artifacts/midogpp/90_oracles_and_diagnostics/"
        "uniform_b_v2_consumed_test_utility_aligned_case_aware_"
        "proxy_information_audit/v1"
    )
    assert output.semantic_identities["publication_status"] == (
        "EXPLORATORY_CONSUMED_DATA_ONLY"
    )
    assert output.semantic_identities["consumed_test_data"] == "true"
    assert output.semantic_identities["fixed_support_case_count_per_center"] == "8"
    assert output.semantic_identities["evaluation_case_count_total"] == "146"
    assert output.semantic_identities["crossfit_fold_audit_row_count"] == "7056"
    assert output.semantic_identities[
        "stage70_prediction_scoring_or_policy_outputs_used"
    ] == "false"
    assert output.semantic_identities["previous_stage90_outputs_used"] == "false"
    assert output.semantic_identities["may_feed_another_stage90_experiment"] == (
        "false"
    )
    assert output.required_files == REQUIRED_FILES
    assert "manifests/support_partition_lock.json" in output.required_files
    assert "manifests/audit_result.json" in output.required_files
    assert "oracle_and_diagnostic_evidence" in output.forbidden_reuse
    assert output.may_feed_recipe_selection is False
    assert output.may_feed_deployable_selection is False


def test_test_aliases_and_authorizations_are_experiment_fenced() -> None:
    workspace = _workspace()
    cache_alias = workspace.artifacts[TEST_CACHE_ARTIFACT_ID]
    cache = workspace.artifacts[UNDERLYING_TEST_CACHE_ID]
    manifest_alias = workspace.artifacts[TEST_MANIFEST_ARTIFACT_ID]
    manifest = workspace.artifacts[UNDERLYING_MANIFEST_ID]
    ledger = workspace.artifacts[TEST_CONSUMPTION_LEDGER_ARTIFACT_ID]

    assert cache_alias.canonical_path == cache.canonical_path
    assert cache_alias.required_files == cache.required_files
    assert cache_alias.expected_file_hashes == cache.expected_file_hashes
    assert cache_alias.semantic_identities["alias_of_artifact_id"] == (
        UNDERLYING_TEST_CACHE_ID
    )
    assert cache_alias.semantic_identities["split"] == "test"
    assert cache_alias.semantic_identities["fresh_evidence"] == "false"
    assert cache_alias.semantic_identities["labels_absent"] == "true"
    assert cache_alias.semantic_identities["authorized_consumer_experiment_ids"] == (
        EXPERIMENT_ID
    )

    assert manifest_alias.physical_path == manifest.physical_path
    assert manifest_alias.required_files == ("manifest.csv",)
    assert manifest_alias.expected_file_hashes["manifest.csv"] == (
        manifest.expected_file_hashes["manifest.csv"]
    )
    assert manifest_alias.semantic_identities["alias_of_artifact_id"] == (
        UNDERLYING_MANIFEST_ID
    )
    assert manifest_alias.semantic_identities[
        "labels_available_before_global_prediction_seal"
    ] == "false"
    assert manifest_alias.semantic_identities[
        "labels_construct_postseal_response_rows"
    ] == "true"
    assert manifest_alias.semantic_identities[
        "label_derived_responses_feed_strict_crossfit_diagnostic_models"
    ] == "true"
    assert manifest_alias.semantic_identities[
        "labels_used_for_feature_construction"
    ] == "false"
    assert manifest_alias.semantic_identities[
        "labels_used_for_policy_or_action_fit"
    ] == "false"
    assert manifest_alias.semantic_identities[
        "authorized_consumer_experiment_ids"
    ] == EXPERIMENT_ID

    for artifact in (cache, ledger):
        assert EXPERIMENT_ID in artifact.semantic_identities[
            "authorized_consumer_experiment_ids"
        ].split("|")


def test_cli_parser_and_lazy_dispatch_use_case_aware_surface(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import midogpp_thesis.cvae.diagnostics.utility_aligned_case_aware_proxy_information_audit as surface

    parsed = cli.build_parser().parse_args(
        (
            "utility-aligned-case-aware-proxy-information-audit",
            "--config",
            str(CONFIG),
            "--artifact-root",
            "/tmp/case-aware-proxy-information-audit",
        )
    )
    assert parsed.surface == "utility-aligned-case-aware-proxy-information-audit"

    sentinel_config = object()
    calls: list[tuple[object, Path]] = []
    monkeypatch.setattr(
        surface,
        "load_utility_aligned_case_aware_proxy_information_audit_config",
        lambda _path: sentinel_config,
    )

    def _run(config: object, *, artifact_root: Path) -> Path:
        calls.append((config, artifact_root))
        return Path("/tmp/case-aware-proxy-information-result")

    monkeypatch.setattr(
        surface,
        "run_utility_aligned_case_aware_proxy_information_audit",
        _run,
    )
    result = cli.main(
        [
            "utility-aligned-case-aware-proxy-information-audit",
            "--config",
            str(CONFIG),
            "--artifact-root",
            "/tmp/case-aware-proxy-information-audit",
        ]
    )

    assert result == 0
    assert calls == [
        (sentinel_config, Path("/tmp/case-aware-proxy-information-audit"))
    ]
    assert capsys.readouterr().out.strip() == (
        "/tmp/case-aware-proxy-information-result"
    )
