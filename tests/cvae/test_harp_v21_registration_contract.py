from __future__ import annotations

from pathlib import Path
import json

import pytest

from midogpp_thesis.cvae.diagnostics.cli import build_parser
from midogpp_thesis.cvae.diagnostics.cli import main
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v21.config import (
    INPUT_ARTIFACT_IDS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v21.identity import (
    EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID,
)
from midogpp_thesis.workspace.preparation_authority import (
    HARP_V21_EXECUTION_AMENDMENT_GATE,
)
from midogpp_thesis.workspace.runtime import MidogppWorkspace


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/uniform_b_v2_consumed_test_fixed_bank_harp_router_v21.yaml"


def test_v21_cli_dispatches_path_free_planned_dry_run(tmp_path, capsys):
    absent = tmp_path / "never-created"
    assert main(["fixed-bank-harp-router-v21", "--config", str(CONFIG), "--dry-run", "--artifact-root", str(absent)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "NEEDS_SEPARATE_EXECUTION_AMENDMENT"
    assert result["paths_resolved"] is False
    assert result["filesystem_mutations"] == 0
    assert not absent.exists()


def test_v21_cli_binds_run_output_without_launching(monkeypatch, tmp_path):
    from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v21 import runner
    observed = {}
    def fake_run(config, **kwargs):
        observed.update(kwargs)
        return "SYNTHETIC_DISPATCH_ONLY"
    monkeypatch.setattr(runner, "run_harp_stage90_v21", fake_run)
    output = tmp_path / "run"
    assert main(["fixed-bank-harp-router-v21", "--config", str(CONFIG), "--artifact-root", str(output), "--confirm", runner.HARP_V21_RUN_CONFIRMATION_TOKEN]) == 0
    assert observed["artifact_root"] == output
    assert not output.exists()


def test_v21_cli_requires_confirmation_before_opening_config():
    with pytest.raises(ProtocolError, match="exact confirmation token"):
        main(["fixed-bank-harp-router-v21", "--config", "/must-not-be-read"])


def test_v21_registry_and_catalog_are_planned_terminal_only() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]

    assert experiment.status == "planned"
    assert experiment.runnable is False
    assert experiment.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert experiment.output_artifact_id == OUTPUT_ARTIFACT_ID
    assert experiment.preparation_authority_gate == HARP_V21_EXECUTION_AMENDMENT_GATE
    assert output.canonical_path == (
        "artifacts/midogpp/90_oracles_and_diagnostics/"
        "uniform_b_v2_consumed_test_fixed_bank_harp_router/v21"
    )
    assert output.semantic_identities["publication_status"] == (
        "POST_HOC_CONSUMED_TEST_SENSITIVITY"
    )
    assert output.semantic_identities["terminal_decision"] == (
        "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"
    )
    from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v21.identity import EXECUTION_REVISION
    assert output.semantic_identities['execution_revision'] == EXECUTION_REVISION
    assert output.semantic_identities['patch_features_target_fit'] == 'false'
    assert output.semantic_identities['evidence_selection_nested'] == 'true'
    assert output.semantic_identities['predecessor_v1_through_v20_state_reused'] == 'false'
    assert output.semantic_identities["case_local_action_eligibility"] == "true"
    assert output.semantic_identities["signed_action_outcomes"] == "true"
    assert output.semantic_identities["may_feed_another_experiment"] == "false"


def test_v21_output_catalog_requires_fresh_role_seals_and_pooled_state() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    required = set(workspace.artifacts[OUTPUT_ARTIFACT_ID].required_files)

    expected = {
        "reports/candidate_frontier.json",
        "reports/source_headroom_diagnostics.json",
        "reports/source_candidate_winner_joins.json",
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


def test_v21_cli_exposes_full_lifecycle() -> None:
    parser = build_parser()
    commands = (
        ("fixed-bank-harp-router-v21", "--config", "x", "--inspect-plan"),
        ("fixed-bank-harp-router-v21", "--config", "x", "--dry-run"),
        ("prepare-fixed-bank-harp-router-v21-inputs", "--repository-root", "."),
        (
            "activate-fixed-bank-harp-router-v21",
            "--authorization-basis",
            "basis",
            "--authorization-date",
            "2026-09-04",
            "--repository-root",
            ".",
        ),
        ("supersede-fixed-bank-harp-router-v21-activation", "--repository-root", "."),
        (
            "supersede-rolled-back-fixed-bank-harp-router-v21-activation",
            "--repository-root",
            ".",
        ),
    )
    for argv in commands:
        assert parser.parse_args(argv).surface == argv[0]


def test_pure_activation_projection_accepts_v21_science_and_rejects_old_policy():
    """In-memory projection only: no amendment, lease or authorized file is issued."""
    import copy
    import yaml
    from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v21.activation_workspace import (
        _render_registry_mapping, validate_rendered_workspace,
    )
    registry=_render_registry_mapping(yaml.safe_load((ROOT/'experiments/midogpp/registry.yaml').read_text()))
    catalog=yaml.safe_load((ROOT/'experiments/midogpp/artifact_catalog.yaml').read_text())
    selected={*INPUT_ARTIFACT_IDS[2:],OUTPUT_ARTIFACT_ID}
    for row in catalog['artifacts']:
        if row['artifact_id'] in selected:
            row['semantic_identities'].update(execution_authorized='true',consumed_test_reuse_authorized='true')
    validate_rendered_workspace(registry,catalog)
    for field,bad in [('policy_family','case_conditional_composite_signed_utility_with_exact_B_abstention'),
                      ('patch_features_target_fit','true'),('evidence_selection_nested','false'),
                      ('predecessor_v1_through_v20_state_reused','true')]:
        poisoned=copy.deepcopy(catalog)
        row=next(r for r in poisoned['artifacts'] if r['artifact_id']==OUTPUT_ARTIFACT_ID)
        row['semantic_identities'][field]=bad
        with pytest.raises(ProtocolError,match='protocol drifted'):
            validate_rendered_workspace(registry,poisoned)
