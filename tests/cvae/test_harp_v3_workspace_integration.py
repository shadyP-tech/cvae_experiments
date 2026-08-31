from __future__ import annotations

from pathlib import Path

import pytest

from midogpp_thesis.cvae.diagnostics.cli import build_parser
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3 import authorization
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3.config import (
    INPUT_ARTIFACT_IDS,
    load_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3.identity import (
    EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3.input_surfaces import (
    CACHE_INDEX,
    CONTENT_INDEX,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3.preparation import (
    CASE_PARTITION,
    LABEL_FREE_BARRIER,
    LABEL_FREE_CONTENT_INDEX,
    PREPARATION_RECEIPT,
)
from midogpp_thesis.workspace.preparation_authority import (
    HARP_V3_EXECUTION_AMENDMENT_GATE,
    HARP_V3_EXPERIMENT_ID,
    KNOWN_PREPARATION_AUTHORITY_GATES,
    preparation_authority_registration_error,
    validate_preparation_authority_extra_args,
)
from midogpp_thesis.workspace.runtime import MidogppWorkspace, WorkspaceError


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / authorization.WORKSPACE_CONFIG_RELATIVE_PATH


def test_checked_in_v3_is_planned_fenced_and_workspace_valid() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    experiment = workspace.experiments[EXPERIMENT_ID]
    config = load_config(CONFIG)

    assert HARP_V3_EXPERIMENT_ID == EXPERIMENT_ID
    assert experiment.status == "planned"
    assert config.execution_authorized is False
    assert config.protocol["utility_kind"] == (
        "downstream_classifier_utility_not_NELBO"
    )
    assert config.protocol["routing_stage_compatibility_estimated"] is False
    assert config.protocol["generative_expert_compatibility_claimed"] is False
    assert all(
        config.expected_hashes[role] is None
        for role in (
            "test_cache_content_sha256",
            "development_manifest_sha256",
            "evaluation_manifest_sha256",
            "parent_ledger_sha256",
            "execution_amendment_sha256",
        )
    )
    assert experiment.output_artifact_id == OUTPUT_ARTIFACT_ID
    assert experiment.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert experiment.preparation_authority_gate == (
        HARP_V3_EXECUTION_AMENDMENT_GATE
    )
    assert tuple(experiment.runner_argv) == authorization.WORKSPACE_RUNNER_ARGV
    assert dict(experiment.runner_env) == dict(authorization.WORKSPACE_RUNNER_ENV)

    with pytest.raises(WorkspaceError, match="status='planned'"):
        workspace.run(EXPERIMENT_ID)


def test_v3_gate_is_closed_world_and_bound_only_to_v3() -> None:
    assert HARP_V3_EXECUTION_AMENDMENT_GATE == (
        "harp_v3_consumed_test_execution_amendment_v1"
    )
    assert HARP_V3_EXECUTION_AMENDMENT_GATE in KNOWN_PREPARATION_AUTHORITY_GATES
    assert preparation_authority_registration_error(
        None,
        experiment_id=EXPERIMENT_ID,
    ) == (
        f"{EXPERIMENT_ID}: runner.preparation_authority_gate must remain "
        f"{HARP_V3_EXECUTION_AMENDMENT_GATE!r}"
    )
    assert preparation_authority_registration_error(
        HARP_V3_EXECUTION_AMENDMENT_GATE,
        experiment_id="wrong.consumer",
    ) == (
        "wrong.consumer: runner.preparation_authority_gate "
        f"{HARP_V3_EXECUTION_AMENDMENT_GATE!r} is bound only to {EXPERIMENT_ID}"
    )
    assert validate_preparation_authority_extra_args(
        HARP_V3_EXECUTION_AMENDMENT_GATE,
        (),
    ) == ()
    assert validate_preparation_authority_extra_args(
        HARP_V3_EXECUTION_AMENDMENT_GATE,
        ("--dry-run",),
    ) == ("--dry-run",)


@pytest.mark.parametrize(
    "surface",
    (
        "fixed-bank-harp-router-v3",
        "prepare-fixed-bank-harp-router-v3-inputs",
        "publish-fixed-bank-harp-router-v3-amendment",
    ),
)
def test_v3_cli_surfaces_are_registered(surface: str) -> None:
    parser = build_parser()
    action = next(
        item
        for item in parser._actions
        if getattr(item, "dest", None) == "surface"
    )
    assert surface in action.choices


def test_v3_registration_determinism_and_workstation_controls_are_exact() -> None:
    assert authorization.WORKSPACE_RUNNER_ENV == {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_VISIBLE_DEVICES": "0,1",
        "OMP_NUM_THREADS": "3",
        "MKL_NUM_THREADS": "3",
        "OPENBLAS_NUM_THREADS": "3",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONHASHSEED": "0",
        "OMP_DYNAMIC": "FALSE",
        "MKL_DYNAMIC": "FALSE",
    }
    assert "fixed-bank-harp-router-v3" in authorization.WORKSPACE_RUNNER_ARGV
    assert "fixed-bank-harp-router-v1" not in authorization.WORKSPACE_RUNNER_ARGV
    assert "fixed-bank-harp-router-v2" not in authorization.WORKSPACE_RUNNER_ARGV


def test_v3_catalog_cache_members_match_preparation_contract() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    cache = workspace.artifacts["midogpp_stage90_harp_consumed_test_cache_v3"]
    required = set(cache.required_files)
    assert {
        CACHE_INDEX.as_posix(),
        CONTENT_INDEX.as_posix(),
        CASE_PARTITION.as_posix(),
        LABEL_FREE_BARRIER.as_posix(),
        LABEL_FREE_CONTENT_INDEX.as_posix(),
        PREPARATION_RECEIPT.as_posix(),
    }.issubset(required)
    assert "manifests/case_partition.json" not in required
    assert "manifests/label_free_partition_barrier.json" not in required
    assert "manifests/label_free_content_index.json" not in required
