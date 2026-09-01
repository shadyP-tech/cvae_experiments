from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v6 import authorization
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v6.config import (
    INPUT_ARTIFACT_IDS,
    load_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v6.identity import (
    EXECUTION_REVISION,
    EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v6.source_seal import (
    FORBIDDEN_PREDECESSOR_MODULE_PREFIXES,
    source_members,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.workspace.preparation_authority import (
    HARP_EXECUTION_AMENDMENT_GATES,
    HARP_V6_EXECUTION_AMENDMENT_GATE,
    HARP_V6_EXPERIMENT_ID,
    HARP_V6_RUN_CONFIRMATION_TOKEN,
    KNOWN_PREPARATION_AUTHORITY_GATES,
    PreparationAuthorityError,
    harp_run_confirmation_token,
    preparation_authority_registration_error,
    validate_preparation_authority_extra_args,
)
from midogpp_thesis.workspace.runtime import MidogppWorkspace, WorkspaceError


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / authorization.WORKSPACE_CONFIG_RELATIVE_PATH
PACKAGE = (
    ROOT
    / "src/midogpp_thesis/cvae/diagnostics/fixed_bank_harp_router_v6"
)


def test_v6_is_a_new_planned_exact_seven_input_identity() -> None:
    config = load_config(CONFIG)

    assert config.experiment_id == HARP_V6_EXPERIMENT_ID == EXPERIMENT_ID
    assert config.output_artifact_id == OUTPUT_ARTIFACT_ID
    assert config.execution_revision == EXECUTION_REVISION
    assert config.execution_authorized is False
    assert len(INPUT_ARTIFACT_IDS) == len(set(INPUT_ARTIFACT_IDS)) == 7
    assert all(value.endswith("_v6") for value in INPUT_ARTIFACT_IDS[2:])
    assert config.claim_boundary["publication_status"] == PUBLICATION_STATUS
    assert config.claim_boundary["terminal_decision"] == TERMINAL_DECISION
    assert config.claim_boundary["fresh_evidence"] is False
    assert config.claim_boundary["implementation_authorizes_execution"] is False


def test_v6_scientific_contract_names_the_router_that_is_implemented() -> None:
    config = load_config(CONFIG)
    model = config.model
    protocol = config.protocol

    assert protocol["routing_stage_compatibility_estimated"] is True
    assert protocol["compatibility_is_label_free_proxy_not_nelbo_or_utility"] is True
    assert protocol["target_support_and_evaluation_cases_disjoint"] is True
    assert model["opportunity_model"] == (
        "source_only_candidate_aware_hurdle_then_pairwise_directional_risk"
    )
    assert model["compatibility_proxy_is_exact_nelbo"] is False
    assert model["compatibility_proxy_is_true_utility"] is False
    assert model["candidate_pool_indexed"] is True
    assert model["learnability_admission_required"] is True
    assert model["soft_composition_reference"] == "exact_directional_action_surface"
    assert model["soft_top_k"] == 2
    assert model["exact_b_byte_identical_fallback"] is True


def test_v6_workstation_contract_separates_classifier_and_science_pools() -> None:
    config = load_config(CONFIG)

    assert config.runtime["persistent_gpu_workers"] == 2
    assert config.runtime["global_parent_blas_threads"] == 1
    assert config.runtime["classifier_workers"] == 4
    assert config.runtime["classifier_blas_threads_per_worker"] == 3
    assert config.runtime["science_workers"] == 4
    assert config.runtime["science_blas_threads_per_worker"] == 1
    assert config.runtime["phase_disjoint_cpu_pools"] is True
    assert config.runtime["cuda_hidden_from_cpu_workers"] is True
    assert config.runtime["no_nested_process_pools"] is True
    assert authorization.WORKSPACE_RUNNER_ENV["OMP_NUM_THREADS"] == "1"
    assert authorization.WORKSPACE_RUNNER_ENV["MKL_NUM_THREADS"] == "1"
    assert authorization.WORKSPACE_RUNNER_ENV["OPENBLAS_NUM_THREADS"] == "1"


@pytest.mark.parametrize("version", ("v1", "v2", "v3", "v4", "v5"))
def test_v6_rejects_all_predecessor_paths(version: str) -> None:
    from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v6.activation_paths import (
        reject_predecessor_path,
    )

    with pytest.raises(ProtocolError, match="predecessor path"):
        reject_predecessor_path(
            f"artifacts/midogpp/fixed_bank_harp_router/{version}/member",
            label="input",
        )


def test_v6_config_loader_rejects_exact_v5_output_path_form(tmp_path: Path) -> None:
    configured = CONFIG.read_text(encoding="utf-8").replace(
        "artifact://midogpp_stage90_harp_consumed_test_cache_v6",
        "artifacts/midogpp/90_oracles_and_diagnostics/"
        "uniform_b_v2_consumed_test_fixed_bank_harp_router/v5",
    )
    candidate = tmp_path / "harp_v6_with_v5_cache.yaml"
    candidate.write_text(configured, encoding="utf-8")

    with pytest.raises(ProtocolError, match="predecessor path"):
        load_config(candidate)


def test_v6_package_has_no_predecessor_imports() -> None:
    for path in PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert not any(
            module == prefix or module.startswith(prefix + ".")
            for module in imported
            for prefix in FORBIDDEN_PREDECESSOR_MODULE_PREFIXES
        ), path


def test_v6_transitive_execution_closure_excludes_predecessor_router_code() -> None:
    relative = {
        path.relative_to(ROOT / "src").as_posix() for path in source_members(ROOT)
    }

    assert not any("/fixed_bank_harp_router_v5/" in path for path in relative)
    assert not any("/runtime/harp_v5_execution/" in path for path in relative)
    assert not any("/routing/dense_residual_soft_router/" in path for path in relative)
    assert not any("/routing/harp_v6" in path for path in relative)


def test_v6_workspace_registration_is_closed_and_non_runnable() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    experiment = workspace.experiments[EXPERIMENT_ID]

    assert experiment.status == "planned"
    assert experiment.runnable is False
    assert experiment.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert experiment.output_artifact_id == OUTPUT_ARTIFACT_ID
    assert experiment.preparation_authority_gate == HARP_V6_EXECUTION_AMENDMENT_GATE
    assert tuple(experiment.runner_argv) == authorization.WORKSPACE_RUNNER_ARGV
    assert dict(experiment.runner_env) == dict(authorization.WORKSPACE_RUNNER_ENV)
    assert HARP_V6_EXECUTION_AMENDMENT_GATE in KNOWN_PREPARATION_AUTHORITY_GATES
    assert HARP_V6_EXECUTION_AMENDMENT_GATE in HARP_EXECUTION_AMENDMENT_GATES
    projection = workspace._preparation_authority_registration_projection(
        replace(experiment, status="diagnostic")
    )
    assert authorization.validate_workspace_registration_execution_projection(
        projection
    ) == authorization.workspace_registration_execution_contract()[
        "workspace_registration_execution_contract_hash"
    ]
    assert preparation_authority_registration_error(
        None,
        experiment_id=EXPERIMENT_ID,
    ) == (
        f"{EXPERIMENT_ID}: runner.preparation_authority_gate must remain "
        f"{HARP_V6_EXECUTION_AMENDMENT_GATE!r}"
    )
    assert preparation_authority_registration_error(
        HARP_V6_EXECUTION_AMENDMENT_GATE,
        experiment_id="wrong.consumer",
    ) == (
        "wrong.consumer: runner.preparation_authority_gate "
        f"{HARP_V6_EXECUTION_AMENDMENT_GATE!r} is bound only to {EXPERIMENT_ID}"
    )
    with pytest.raises(WorkspaceError, match="status='planned'"):
        workspace.run(EXPERIMENT_ID)


def test_v6_launch_gate_accepts_only_dry_run_or_exact_confirmation() -> None:
    assert harp_run_confirmation_token(HARP_V6_EXECUTION_AMENDMENT_GATE) == (
        HARP_V6_RUN_CONFIRMATION_TOKEN
    )
    assert validate_preparation_authority_extra_args(
        HARP_V6_EXECUTION_AMENDMENT_GATE, ("--dry-run",)
    ) == ("--dry-run",)
    assert validate_preparation_authority_extra_args(
        HARP_V6_EXECUTION_AMENDMENT_GATE,
        ("--confirm", HARP_V6_RUN_CONFIRMATION_TOKEN),
    ) == ("--confirm", HARP_V6_RUN_CONFIRMATION_TOKEN)
    for arguments in ((), ("--confirm", "wrong"), ("--dry-run", "extra")):
        with pytest.raises(PreparationAuthorityError, match="accepts only"):
            validate_preparation_authority_extra_args(
                HARP_V6_EXECUTION_AMENDMENT_GATE, arguments
            )
