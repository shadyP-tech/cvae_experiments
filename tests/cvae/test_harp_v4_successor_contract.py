from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

from midogpp_thesis.cvae.diagnostics import cli
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v4 import authorization
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v4 import source_seal
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v4.activation_paths import (
    reject_predecessor_path,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v4.config import (
    INPUT_ARTIFACT_IDS,
    load_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v4.identity import (
    EXECUTION_REVISION,
    EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v4.input_surfaces import (
    V4_CACHE_IDENTITY,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v4.source_seal import (
    FORBIDDEN_PREDECESSOR_MODULE_PREFIXES,
    source_members,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.harp_v4_execution.hash_contracts import (
    runtime_hash_contract_payload,
)
from midogpp_thesis.workspace.preparation_authority import (
    HARP_V4_EXECUTION_AMENDMENT_GATE,
    HARP_V4_EXPERIMENT_ID,
    HARP_V4_RUN_CONFIRMATION_TOKEN,
    KNOWN_PREPARATION_AUTHORITY_GATES,
    PreparationAuthorityError,
    harp_run_confirmation_token,
    preparation_authority_registration_error,
    validate_preparation_authority_extra_args,
)
from midogpp_thesis.workspace.runtime import MidogppWorkspace, WorkspaceError


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / authorization.WORKSPACE_CONFIG_RELATIVE_PATH


def test_v4_is_a_planned_non_authorizing_successor_with_unique_state() -> None:
    config = load_config(CONFIG)

    assert config.experiment_id == HARP_V4_EXPERIMENT_ID == EXPERIMENT_ID
    assert config.output_artifact_id == OUTPUT_ARTIFACT_ID
    assert config.execution_revision == EXECUTION_REVISION
    assert config.execution_authorized is False
    assert config.claim_boundary["implementation_authorizes_execution"] is False
    assert config.claim_boundary["fresh_evidence"] is False
    assert config.claim_boundary["publication_status"] == PUBLICATION_STATUS
    assert config.claim_boundary["terminal_decision"] == TERMINAL_DECISION
    assert V4_CACHE_IDENTITY.artifact_id == INPUT_ARTIFACT_IDS[2]
    assert all(value.endswith("_v4") for value in INPUT_ARTIFACT_IDS[2:])
    assert len(set(INPUT_ARTIFACT_IDS)) == len(INPUT_ARTIFACT_IDS)
    assert "/v4" in authorization.WORKSPACE_OUTPUT_CANONICAL_PATH
    assert "harp_router_v4" in authorization.WORKSPACE_CONFIG_RELATIVE_PATH
    assert "harp_router_v4" in authorization.WORKSPACE_AMENDMENT_RELATIVE_PATH
    assert "harp_router_v3" not in " ".join(config.input_locations.values())


@pytest.mark.parametrize(
    "fragment",
    (
        "fixed_bank_harp_router/v1",
        "fixed_bank_harp_router/v2",
        "fixed_bank_harp_router/v3",
        "harp_router_v1",
        "harp_router_v2",
        "harp_router_v3",
        "harp_consumed_test_cache_v1",
        "harp_consumed_test_cache_v2",
        "harp_consumed_test_cache_v3",
    ),
)
def test_v4_rejects_every_predecessor_path(fragment: str) -> None:
    with pytest.raises(ProtocolError, match="predecessor path"):
        reject_predecessor_path(f"artifacts/midogpp/{fragment}/member", label="input")


def test_v4_config_rejects_predecessor_location_before_resolution(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["inputs"]["test_cache_root"] = (
        "artifact://midogpp_stage90_harp_consumed_test_cache_v3"
    )
    drifted = tmp_path / "v4-with-predecessor.yaml"
    drifted.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ProtocolError, match="predecessor path"):
        load_config(drifted)


def test_v4_config_rejects_sha256_in_a_semantic_lock_field(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["inputs"]["expert_bank_lock_hash"] = "a" * 64
    drifted = tmp_path / "v4-with-mixed-hash-role.yaml"
    drifted.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ProtocolError, match="exact 16-hex semantic hash"):
        load_config(drifted)


def test_v4_source_seal_contains_only_v4_versioned_harp_modules() -> None:
    members = source_members(ROOT)
    relative = tuple(path.relative_to(ROOT / "src").as_posix() for path in members)

    assert any("fixed_bank_harp_router_v4/config.py" in value for value in relative)
    assert any("runtime/harp_v4_execution/hash_contracts.py" in value for value in relative)
    assert any("runtime/harp_v4_execution/frame_binding.py" in value for value in relative)
    assert any("routing/harp_v4/fitting.py" in value for value in relative)
    assert "midogpp_thesis/cvae/diagnostics/cli.py" in relative
    assert "midogpp_thesis/workspace/runtime.py" in relative
    module_names = tuple(value.removesuffix(".py").replace("/", ".") for value in relative)
    assert not any(
        module == prefix or module.startswith(prefix + ".")
        for module in module_names
        for prefix in FORBIDDEN_PREDECESSOR_MODULE_PREFIXES
    )


@pytest.mark.parametrize("prefix", FORBIDDEN_PREDECESSOR_MODULE_PREFIXES)
def test_v4_source_seal_rejects_predecessor_imports(prefix: str) -> None:
    with pytest.raises(ProtocolError, match="exhausted predecessor module"):
        source_seal._reject_predecessor_import(prefix + ".runner")


def test_v4_runtime_has_no_ambiguous_source_cache_hash_field() -> None:
    runtime_root = ROOT / "src/midogpp_thesis/cvae/runtime/harp_v4_execution"
    for path in runtime_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        assert not any(
            isinstance(node, ast.Constant) and node.value == "source_cache_hash"
            for node in ast.walk(tree)
        ), path

    contract = runtime_hash_contract_payload()
    assert contract["semantic_hash_width"] == 16
    assert contract["ambiguous_source_cache_hash_allowed"] is False
    assert contract["mixed_width_field_names_allowed"] is False
    assert "classifier_task.source_stream_lock_hash" in contract[
        "semantic_hash_field_paths"
    ]
    assert "classifier_task.source_stream_lock_sha256" in contract[
        "content_sha256_field_paths"
    ]
    assert "classifier_checkpoint.actions[].scaler_state_hash" in contract[
        "semantic_hash_field_paths"
    ]
    assert "classifier_checkpoint.actions[].probability_sha256" in contract[
        "content_sha256_field_paths"
    ]
    assert len(str(contract["runtime_hash_contract_hash"])) == 64


def test_v4_routing_science_matches_the_declared_v3_method_exactly() -> None:
    routing_root = ROOT / "src/midogpp_thesis/cvae/routing"
    v3_root = routing_root / "harp_v3"
    v4_root = routing_root / "harp_v4"
    assert {path.name for path in v4_root.glob("*.py")} == {
        path.name for path in v3_root.glob("*.py")
    }
    for v4_path in v4_root.glob("*.py"):
        normalized = (
            v4_path.read_text(encoding="utf-8")
            .replace("HarpV4", "HarpV3")
            .replace("HARP v4", "HARP v3")
            .replace("harp_v4", "harp_v3")
            .replace("HARP_V4", "HARP_V3")
            .replace("v4", "v3")
            .replace("V4", "V3")
        )
        assert normalized == (v3_root / v4_path.name).read_text(encoding="utf-8")


def test_v4_workspace_registration_is_closed_and_non_runnable() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    experiment = workspace.experiments[EXPERIMENT_ID]

    assert experiment.status == "planned"
    assert experiment.runnable is False
    assert experiment.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert experiment.output_artifact_id == OUTPUT_ARTIFACT_ID
    assert experiment.preparation_authority_gate == HARP_V4_EXECUTION_AMENDMENT_GATE
    assert tuple(experiment.runner_argv) == authorization.WORKSPACE_RUNNER_ARGV
    assert dict(experiment.runner_env) == dict(authorization.WORKSPACE_RUNNER_ENV)
    assert HARP_V4_EXECUTION_AMENDMENT_GATE in KNOWN_PREPARATION_AUTHORITY_GATES
    assert preparation_authority_registration_error(
        HARP_V4_EXECUTION_AMENDMENT_GATE,
        experiment_id="wrong.consumer",
    ) == (
        "wrong.consumer: runner.preparation_authority_gate "
        f"{HARP_V4_EXECUTION_AMENDMENT_GATE!r} is bound only to {EXPERIMENT_ID}"
    )
    with pytest.raises(WorkspaceError, match="status='planned'"):
        workspace.run(EXPERIMENT_ID)


def test_v4_catalog_declares_physical_rebuild_and_no_predecessor_state() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    cache = workspace.artifacts[INPUT_ARTIFACT_IDS[2]]
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]

    assert cache.canonical_path and cache.canonical_path.endswith(
        "harp_consumed_test_cache_v4"
    )
    assert cache.semantic_identities["predecessor_cache_or_output_used"] == "false"
    assert cache.semantic_identities["preparation_cli"].endswith(
        "prepare-fixed-bank-harp-router-v4-inputs"
    )
    assert output.canonical_path and output.canonical_path.endswith(
        "fixed_bank_harp_router/v4"
    )
    assert output.semantic_identities["predecessor_output_or_state_used"] == "false"
    assert output.semantic_identities["semantic_hash_width"] == "16"
    assert output.semantic_identities["content_hash_width"] == "64"
    assert output.semantic_identities["ambiguous_source_cache_hash_allowed"] == "false"
    assert output.semantic_identities["fresh_evidence"] == "false"
    assert output.semantic_identities["routing_success_claimed"] == "false"


def test_v4_launch_gate_accepts_only_exact_dry_run_or_confirmation() -> None:
    assert harp_run_confirmation_token(HARP_V4_EXECUTION_AMENDMENT_GATE) == (
        HARP_V4_RUN_CONFIRMATION_TOKEN
    )
    assert validate_preparation_authority_extra_args(
        HARP_V4_EXECUTION_AMENDMENT_GATE,
        ("--dry-run",),
    ) == ("--dry-run",)
    assert validate_preparation_authority_extra_args(
        HARP_V4_EXECUTION_AMENDMENT_GATE,
        ("--confirm", HARP_V4_RUN_CONFIRMATION_TOKEN),
    ) == ("--confirm", HARP_V4_RUN_CONFIRMATION_TOKEN)

    for extra_args in (
        (),
        ("--confirm",),
        ("--confirm", "wrong"),
        ("--confirm=" + HARP_V4_RUN_CONFIRMATION_TOKEN,),
        ("--dry-run", "extra"),
    ):
        with pytest.raises(PreparationAuthorityError, match="accepts only"):
            validate_preparation_authority_extra_args(
                HARP_V4_EXECUTION_AMENDMENT_GATE,
                extra_args,
            )
    with pytest.raises(PreparationAuthorityError, match="reject --force"):
        validate_preparation_authority_extra_args(
            HARP_V4_EXECUTION_AMENDMENT_GATE,
            ("--dry-run",),
            force=True,
        )


def test_v4_direct_run_rejects_before_missing_config_access() -> None:
    with pytest.raises(ProtocolError, match="exact confirmation token"):
        cli.main(
            [
                "fixed-bank-harp-router-v4",
                "--config",
                "/definitely/not/a/readable/harp-v4-config.yaml",
            ]
        )
