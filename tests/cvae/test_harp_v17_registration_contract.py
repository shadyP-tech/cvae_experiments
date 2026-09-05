from __future__ import annotations

from pathlib import Path

from midogpp_thesis.cvae.diagnostics.cli import build_parser
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v17.config import (
    INPUT_ARTIFACT_IDS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v17.identity import (
    EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID,
)
from midogpp_thesis.workspace.preparation_authority import (
    HARP_V17_EXECUTION_AMENDMENT_GATE,
)
from midogpp_thesis.workspace.runtime import MidogppWorkspace


ROOT = Path(__file__).resolve().parents[2]


def test_v17_registry_and_catalog_are_planned_terminal_only() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]

    assert experiment.status == "planned"
    assert experiment.runnable is False
    assert experiment.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert experiment.output_artifact_id == OUTPUT_ARTIFACT_ID
    assert experiment.preparation_authority_gate == HARP_V17_EXECUTION_AMENDMENT_GATE
    assert output.canonical_path == (
        "artifacts/midogpp/90_oracles_and_diagnostics/"
        "uniform_b_v2_consumed_test_fixed_bank_harp_router/v17"
    )
    assert output.semantic_identities["publication_status"] == (
        "POST_HOC_CONSUMED_TEST_SENSITIVITY"
    )
    assert output.semantic_identities["terminal_decision"] == (
        "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"
    )
    assert output.semantic_identities["u_full_opportunity_head"] == (
        "EXACT_U_POSITIVE_BACC_AND_NONNEGATIVE_GAIN"
    )
    assert output.semantic_identities["selected_action_family_route_score"] == "true"
    assert output.semantic_identities["may_feed_another_experiment"] == "false"


def test_v17_output_catalog_requires_fresh_role_seals_and_pooled_state() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    required = set(workspace.artifacts[OUTPUT_ARTIFACT_ID].required_files)

    expected = {
        "manifests/source_train_menu_seals.json",
        "manifests/target_evaluation_menu_seals.json",
        "manifests/bank_independence_attestations.json",
        "manifests/source_target_role_seals/fixed_bank_independence_attestation.json",
        "manifests/source_train_label_access_begun.json",
        "reports/source_train_label_access.json",
        "stores/source_train_case_surface/manifest.json",
        "stores/pooled_router_policy/manifest.json",
        "manifests/source_policy_admission_seal.json",
    }
    for center in ("0", "1", "2", "3", "5", "6", "7", "8", "9"):
        expected.update(
            {
                f"stores/physical_menu/center_{center}/manifest.json",
                "manifests/source_target_role_seals/"
                f"center_{center}/source_train_menu_seal.json",
                "manifests/source_target_role_seals/"
                f"center_{center}/target_evaluation_menu_seal.json",
            }
        )
    assert expected.issubset(required)
    assert not any("target_local_models" in member for member in required)
    assert not any("target_support_menu" in member for member in required)


def test_v17_cli_exposes_full_lifecycle() -> None:
    parser = build_parser()
    commands = (
        ("fixed-bank-harp-router-v17", "--config", "x", "--inspect-plan"),
        ("fixed-bank-harp-router-v17", "--config", "x", "--dry-run"),
        ("prepare-fixed-bank-harp-router-v17-inputs", "--repository-root", "."),
        (
            "activate-fixed-bank-harp-router-v17",
            "--authorization-basis",
            "basis",
            "--authorization-date",
            "2026-09-04",
            "--repository-root",
            ".",
        ),
        ("supersede-fixed-bank-harp-router-v17-activation", "--repository-root", "."),
        (
            "supersede-rolled-back-fixed-bank-harp-router-v17-activation",
            "--repository-root",
            ".",
        ),
    )
    for argv in commands:
        assert parser.parse_args(argv).surface == argv[0]
