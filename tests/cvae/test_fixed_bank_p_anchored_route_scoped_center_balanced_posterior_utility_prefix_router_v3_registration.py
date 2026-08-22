from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from midogpp_thesis.cvae.diagnostics import cli
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_v3.bundle import (
    REQUIRED_FILES,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_v3.config import (
    load_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_v3.experiment_contracts import (
    AUTHORIZATION_BASIS,
    INPUT_ARTIFACT_IDS,
    LEDGER_AMENDMENT_ARTIFACT_ID,
    TEST_CACHE_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
)
from midogpp_thesis.workspace.runtime import MidogppWorkspace


ROOT = Path(__file__).resolve().parents[2]
BASE = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_p_anchored_route_scoped_"
    "center_balanced_posterior_utility_prefix_router"
)
OUTPUT_BASE = (
    "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_p_anchored_route_scoped_"
    "center_balanced_posterior_utility_prefix_router"
)
V2_EXPERIMENT_ID = f"{BASE}.v2"
V3_EXPERIMENT_ID = f"{BASE}.v3"
V2_OUTPUT_ARTIFACT_ID = f"{OUTPUT_BASE}_v2"
V3_OUTPUT_ARTIFACT_ID = f"{OUTPUT_BASE}_v3"
V3_SURFACE = (
    "fixed-bank-p-anchored-route-scoped-center-balanced-"
    "posterior-utility-prefix-router-v3"
)
CONFIG = ROOT / (
    "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_route_scoped_center_"
    "balanced_posterior_utility_prefix_router_v3.yaml"
)
AMENDMENT = ROOT / (
    "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts/"
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_route_scoped_center_"
    "balanced_posterior_utility_prefix_router_ledger_amendment_v3.json"
)
EXPECTED_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "CUDA_VISIBLE_DEVICES": "0,1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONUNBUFFERED": "1",
}
SOURCE_MANIFEST_SHA256 = (
    "7bdd9459a39820ab7b28627b13b35fe6318887d7d331e38366c23bdd03cba401"
)
SOURCE_TREE_SHA256 = (
    "5eadd5a7d031ca959ef73bfb601a7fe6102c071b84184caa22ad1523ae7585b8"
)
AMENDMENT_SHA256 = (
    "46ee5362b0f44f6ec095eb4dfb3fd47fd335363f0f5d6a866995cc69d224ec63"
)


def test_v2_is_failed_preterminally_and_non_runnable() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    experiment = workspace.get_experiment(V2_EXPERIMENT_ID)
    output = workspace.artifacts[V2_OUTPUT_ARTIFACT_ID]

    assert experiment.status == "failed"
    assert experiment.runnable is False
    assert experiment.runner_argv[:2] == ("{python}", "-c")
    assert "cannot be recovered or rerun" in experiment.runner_argv[-1]
    assert output.evidence_label == "REJECTED"
    assert output.availability == "workstation_failed_preterminal"
    assert output.semantic_identities["run_state_status"] == "FAILED"
    assert output.semantic_identities["failure_phase"] == (
        "ROUTE_ENDPOINTS_436_POSTERIORS_AND_CANDIDATE_SEAL"
    )
    assert output.semantic_identities["failure_error"] == (
        "CBPUPR endpoint worker plan lineage drifted."
    )
    assert output.semantic_identities["failure_error_class"] == "ProtocolError"
    assert output.semantic_identities["terminal_access_journal_count"] == "0"
    assert output.semantic_identities["terminal_evaluation_labels_opened"] == "false"
    assert output.semantic_identities["terminal_metrics_computed"] == "false"
    assert output.semantic_identities["diagnostic_result_valid"] == "false"
    assert output.semantic_identities["recoverable"] == "false"
    assert output.semantic_identities["rerunnable"] == "false"
    assert output.semantic_identities["execution_authorized"] == "true"
    assert output.semantic_identities["original_execution_authorized"] == "true"
    assert output.semantic_identities["further_execution_authorized"] == "false"
    assert output.semantic_identities["authorization_exhausted"] == "true"


def test_v3_is_authorized_diagnostic_registered_and_not_yet_created() -> None:
    config = (
        load_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_config(
            CONFIG
        )
    )
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    experiment = workspace.get_experiment(V3_EXPERIMENT_ID)
    output = workspace.artifacts[V3_OUTPUT_ARTIFACT_ID]

    assert config.contract_hash == "c1758de53eabb61a"
    assert config.expected_ledger_amendment_sha256 == AMENDMENT_SHA256
    assert hashlib.sha256(AMENDMENT.read_bytes()).hexdigest() == AMENDMENT_SHA256
    assert experiment.status == "diagnostic"
    assert experiment.runnable is True
    assert experiment.config_path == CONFIG.relative_to(ROOT).as_posix()
    assert experiment.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert len(experiment.input_claim_scope_exceptions) == 5
    assert experiment.runner_env == EXPECTED_ENVIRONMENT
    assert experiment.runner_argv == (
        "{python}",
        "-m",
        "midogpp_thesis",
        "cvae-diagnostics",
        V3_SURFACE,
        "--config",
        "{resolved_config}",
        "--artifact-root",
        f"output://{V3_OUTPUT_ARTIFACT_ID}",
    )

    assert output.availability == "generated_on_run"
    assert output.evidence_label == "POST_HOC_CONSUMED_TEST_SENSITIVITY"
    assert output.required_files == REQUIRED_FILES
    assert output.semantic_identities["execution_authorized"] == "true"
    assert not workspace.resolve_artifact(
        V3_OUTPUT_ARTIFACT_ID,
        for_output=True,
        require_exists=False,
    ).exists()
    assert output.semantic_identities["config_contract_hash"] == "c1758de53eabb61a"
    assert output.semantic_identities["execution_authorization_basis"] == (
        AUTHORIZATION_BASIS
    )
    assert output.semantic_identities["repair_source_manifest_sha256"] == (
        SOURCE_MANIFEST_SHA256
    )
    assert output.semantic_identities["repair_source_tree_sha256"] == (
        SOURCE_TREE_SHA256
    )
    assert output.semantic_identities["repair_source_member_count"] == "93"
    assert output.semantic_identities["expected_ledger_amendment_sha256"] == (
        AMENDMENT_SHA256
    )
    assert output.semantic_identities["fresh_evidence"] == "false"
    assert output.semantic_identities["current_evidence"] == "false"
    assert output.semantic_identities["routing_success_claimed"] == "false"
    assert output.semantic_identities["promotion_eligible"] == "false"
    assert output.semantic_identities["may_feed_another_experiment"] == "false"
    assert output.semantic_identities["publication_status"] == (
        "POST_HOC_CONSUMED_TEST_SENSITIVITY"
    )
    assert output.semantic_identities["terminal_decision"] == (
        "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"
    )


def test_v3_has_exact_six_inputs_and_direct_original_alias_lineage() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    scoped_v3_artifacts = {
        artifact_id
        for artifact_id in workspace.artifacts
        if "center_balanced_posterior_utility_prefix_router" in artifact_id
        and artifact_id.endswith("_v3")
    }

    assert scoped_v3_artifacts == {
        TEST_CACHE_ARTIFACT_ID,
        TEST_MANIFEST_ARTIFACT_ID,
        TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
        LEDGER_AMENDMENT_ARTIFACT_ID,
        V3_OUTPUT_ARTIFACT_ID,
    }
    assert CONFIG.is_file()
    assert AMENDMENT.is_file()
    assert workspace.artifacts[TEST_CACHE_ARTIFACT_ID].semantic_identities[
        "alias_of_artifact_id"
    ] == "midogpp_virchow2_uniform_b_v2_descriptive_test_cache_seed42"
    assert workspace.artifacts[TEST_MANIFEST_ARTIFACT_ID].semantic_identities[
        "alias_of_artifact_id"
    ] == "midogpp_dataset_contract_annotation_patch_v1"
    parent = workspace.artifacts[TEST_CONSUMPTION_LEDGER_ARTIFACT_ID]
    amendment = workspace.artifacts[LEDGER_AMENDMENT_ARTIFACT_ID]
    assert parent.semantic_identities["alias_of_artifact_id"] == (
        "midogpp_uniform_b_test_consumption_ledger_v1"
    )
    assert amendment.semantic_identities["parent_artifact_id"] == (
        "midogpp_uniform_b_test_consumption_ledger_v1"
    )
    for artifact_id in INPUT_ARTIFACT_IDS[2:]:
        identities = workspace.artifacts[artifact_id].semantic_identities
        assert identities["authorized_consumer_experiment_ids"] == V3_EXPERIMENT_ID

    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    assert payload["authorized_consumer_experiment_ids"] == [V3_EXPERIMENT_ID]
    assert payload["authorization_basis"] == AUTHORIZATION_BASIS
    assert payload["failed_v2_output_used"] is False
    assert payload["prior_v2_execution_authorization_reused"] is False
    assert payload["previous_stage90_outputs_used"] is False


def test_v3_cli_surface_is_unique_and_dispatches_only_v3(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parsed = cli.build_parser().parse_args(
        [V3_SURFACE, "--config", str(CONFIG), "--artifact-root", "/tmp/cbpupr-v3"]
    )
    assert parsed.surface == V3_SURFACE

    import midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router as v2_surface
    import midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_v3 as v3_surface

    sentinel = object()
    calls: list[tuple[object, Path]] = []

    def reject_v2(*args: object, **kwargs: object) -> None:
        raise AssertionError("v3 CLI dispatched through the v2 package")

    monkeypatch.setattr(
        v2_surface,
        "load_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_config",
        reject_v2,
    )
    monkeypatch.setattr(
        v2_surface,
        "run_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router",
        reject_v2,
    )
    monkeypatch.setattr(
        v3_surface,
        "load_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_config",
        lambda _: sentinel,
    )
    monkeypatch.setattr(
        v3_surface,
        "run_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router",
        lambda config, *, artifact_root: calls.append((config, artifact_root))
        or Path("/tmp/cbpupr-v3-result"),
    )

    assert cli.main(
        [V3_SURFACE, "--config", str(CONFIG), "--artifact-root", "/tmp/cbpupr-v3"]
    ) == 0
    assert calls == [(sentinel, Path("/tmp/cbpupr-v3"))]
    assert capsys.readouterr().out.strip() == "/tmp/cbpupr-v3-result"
