from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from midogpp_thesis.cvae.diagnostics import cli
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.config import (
    CONFIG_TOP_LEVEL,
    frozen_config_contract_payload,
    load_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.execution_admission import (
    BLOCKED_MESSAGE,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.identity import (
    CLI_SURFACE,
    EXPERIMENT_ID,
    INPUT_ARTIFACT_IDS,
    OUTPUT_ARTIFACT_ID,
    SOURCE_INNER_ALIAS_ARTIFACT_ID,
    EXPECTED_SOURCE_REUSE_AMENDMENT_SHA256,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.runner import (
    run_planned_sceptre_router,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.workspace.runtime import MidogppWorkspace, WorkspaceError


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs"
    / "uniform_b_v2_consumed_test_fixed_bank_sceptre_router_v1.yaml"
)


def test_scoped_config_is_exact_and_source_sealed() -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config = load_config(CONFIG)

    assert set(raw) == set(CONFIG_TOP_LEVEL)
    assert raw == frozen_config_contract_payload()
    assert config.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert raw["inputs"]["direct_input_count"] == 5
    assert raw["inputs"]["source_inner_reuse_role"] == (
        "DEVELOPMENT_ONLY_ADAPTIVE_DESCRIPTIVE"
    )
    assert raw["inputs"]["test_cache_capability_registered"] is False
    assert raw["inputs"]["source_inner_evidence_members_label_free"] is True
    assert raw["protocol"]["strict_outer_center_exclusion"] == {
        "rule": "DELETE_ALL_ROWS_WITH_QUERY_H_OR_CANDIDATE_H",
        "q_equal_H_removed": True,
        "e_equal_H_removed": True,
        "q_equal_e_forbidden": True,
        "applied_before_feature_transforms": True,
        "applied_before_normalization": True,
        "applied_before_fitting": True,
        "applied_before_hyperparameter_tuning": True,
    }
    assert raw["source_provenance"][
        "label_free_core_may_import_source_inner_utility"
    ] is False
    assert raw["source_provenance"]["scientific_member_count"] == 38
    future = raw["protocol"]["future_consumed_test_execution"]
    assert future[
        "manager_owned_calibration_uncertainty_registration_required"
    ] is True
    assert future["final_route_policy_exact_fold_count"] == 45
    assert future["terminal_capability_binds_final_route_policy_hash"] is True
    assert raw["protocol"]["future_consumed_test_execution"][
        "one_shot_label_reader_implemented"
    ] is False
    assert len(raw["source_provenance"]["scientific_source_tree_sha256"]) == 64


def test_workspace_registration_is_planned_and_consumer_fenced() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    config = load_config(CONFIG)
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    alias = workspace.artifacts[SOURCE_INNER_ALIAS_ARTIFACT_ID]
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]

    assert experiment.status == "planned"
    assert experiment.runnable is False
    assert experiment.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert alias.semantic_identities["consumer_resolution_fence_only"] == "true"
    assert alias.semantic_identities["execution_authorized"] == "false"
    assert len(alias.required_files) == 7
    assert alias.semantic_identities["prediction_packet_contains_labels"] == "false"
    assert output.availability == "planned_execution_not_authorized"
    assert output.semantic_identities["config_contract_hash"] == config.contract_hash
    assert output.semantic_identities["protocol_contract_hash"] == config.protocol[
        "protocol_hash"
    ]
    assert output.semantic_identities["scientific_source_tree_sha256"] == (
        config.source_provenance["scientific_source_tree_sha256"]
    )
    assert output.semantic_identities["source_inner_adaptive_development_reuse_registered"] == "true"
    assert output.semantic_identities["source_inner_reuse_amendment_sha256"] == (
        EXPECTED_SOURCE_REUSE_AMENDMENT_SHA256
    )
    assert output.semantic_identities["test_cache_resolution_present"] == "false"
    assert output.semantic_identities["may_feed_another_experiment"] == "false"


def test_workspace_direct_runner_and_cli_refuse_before_mutation(tmp_path: Path) -> None:
    parsed = cli.build_parser().parse_args(
        [CLI_SURFACE, "--config", str(CONFIG), "--artifact-root", "/tmp/unused"]
    )
    assert parsed.surface == CLI_SURFACE

    workspace = MidogppWorkspace.load(ROOT)
    with pytest.raises(WorkspaceError, match="status='planned'"):
        workspace._render_run(  # noqa: SLF001 - exact pre-run refusal seam
            EXPERIMENT_ID,
            require_inputs=False,
            validate_workspace=True,
            include_all_declared_inputs=True,
        )

    config = load_config(CONFIG)
    output = tmp_path / "direct" / "output"
    with pytest.raises(ProtocolError, match="execution is not authorized"):
        run_planned_sceptre_router(config, artifact_root=output)
    assert not (tmp_path / "direct").exists()

    cli_output = tmp_path / "cli" / "output"
    with pytest.raises(ProtocolError, match="execution is not authorized"):
        cli.main(
            [
                CLI_SURFACE,
                "--config",
                str(CONFIG),
                "--artifact-root",
                str(cli_output),
            ]
        )
    assert not (tmp_path / "cli").exists()
    assert "source-inner reuse is authorized" in BLOCKED_MESSAGE


def test_planned_output_has_no_downstream_consumer() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    consumers = [
        experiment.experiment_id
        for experiment in workspace.experiments.values()
        if OUTPUT_ARTIFACT_ID in experiment.input_artifact_ids
    ]
    assert consumers == []
