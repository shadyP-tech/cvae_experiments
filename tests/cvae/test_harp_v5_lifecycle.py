from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path

import pytest

from midogpp_thesis.cvae.diagnostics import cli
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v5 import (
    activation,
    authorization,
    workstation_preparation,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v5.activation_transaction import (
    _resume,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v5.config import load_config
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v5.runner import (
    HARP_V5_RUN_CONFIRMATION_TOKEN,
    dry_run_harp_stage90_v5,
    inspect_harp_stage90_v5,
    run_harp_stage90_v5,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v5.execution.admission import (
    dedicated_scratch,
    validate_pristine_or_label_free_recovery,
)
from midogpp_thesis.cvae.routing.harp_protocol import canonical_hash
from midogpp_thesis.cvae.runtime.artifact_io import atomic_json
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v5.workspace_paths import (
    _EXPECTED_LOCATIONS,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.workspace.runtime import MidogppWorkspace


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / authorization.WORKSPACE_CONFIG_RELATIVE_PATH


def test_planned_inspection_and_dry_run_are_path_free_and_mutation_free() -> None:
    config = load_config(CONFIG)
    before = CONFIG.read_bytes()

    inspection = inspect_harp_stage90_v5(config)
    dry_run = dry_run_harp_stage90_v5(config, artifact_root="ignored")

    assert inspection["status"] == "PLANNED_NEEDS_SEPARATE_EXECUTION_AMENDMENT"
    assert inspection["paths_resolved"] is False
    assert inspection["filesystem_mutations"] == 0
    assert dry_run["status"] == "NEEDS_SEPARATE_EXECUTION_AMENDMENT"
    assert dry_run["authorization_probed"] is False
    assert dry_run["paths_resolved"] is False
    assert dry_run["filesystem_mutations"] == 0
    assert CONFIG.read_bytes() == before


def test_run_rejects_wrong_confirmation_before_typed_config_or_path_access() -> None:
    with pytest.raises(ProtocolError, match="exact confirmation token"):
        run_harp_stage90_v5(
            object(),  # type: ignore[arg-type]
            artifact_root="must-not-be-resolved",
            confirmation_token="wrong",
        )


def test_v5_has_distinct_preparation_activation_and_run_tokens() -> None:
    assert workstation_preparation.PREPARATION_CONFIRMATION == (
        "PREPARE_HARP_V5_CONSUMED_TEST_INPUTS"
    )
    assert activation.ACTIVATION_CONFIRMATION == (
        "ACTIVATE_HARP_V5_TERMINAL_CONSUMED_TEST_DIAGNOSTIC"
    )
    assert HARP_V5_RUN_CONFIRMATION_TOKEN == (
        "RUN_HARP_V5_TERMINAL_CONSUMED_TEST_DIAGNOSTIC"
    )
    assert len(
        {
            workstation_preparation.PREPARATION_CONFIRMATION,
            activation.ACTIVATION_CONFIRMATION,
            HARP_V5_RUN_CONFIRMATION_TOKEN,
        }
    ) == 3


def test_catalog_resolver_projects_only_v5_destinations() -> None:
    # This contract check must remain usable in a clean checkout where the
    # large expert bank and canonical cache have not been materialized yet.
    workspace = MidogppWorkspace.load(ROOT)
    rendered = "\n".join(
        str(getattr(workspace.artifacts[artifact_id], field))
        for artifact_id, (field, _expected) in _EXPECTED_LOCATIONS.items()
        if artifact_id.endswith("_v5")
    )
    assert "v5" in rendered
    assert "harp_router_v4" not in rendered
    assert "harp_consumed_test_cache_v4" not in rendered
    assert "fixed_bank_harp_router/v4" not in rendered


def test_activation_commit_sequence_keeps_registry_last() -> None:
    source = inspect.getsource(_resume)
    assert source.index('"amendment_committed"') < source.index('"config_committed"')
    assert source.index('"config_committed"') < source.index('"catalog_committed"')
    assert source.index('"catalog_committed"') < source.index('"registry_committed"')
    assert "Registry is deliberately the only runnable-gate commit point" in source


def test_scratch_binding_is_process_independent_and_same_lease_stable(
    tmp_path: Path,
) -> None:
    config = load_config(CONFIG)
    runtime = {**config.runtime, "scratch_root": str(tmp_path / "scratch")}
    config = replace(config, runtime=runtime)
    output = tmp_path / "output"
    output.mkdir()

    first = dedicated_scratch(
        config,
        admission_hash="a" * 64,
        authorization_lease_hash="b" * 64,
        root=output,
    )
    second = dedicated_scratch(
        config,
        admission_hash="a" * 64,
        authorization_lease_hash="b" * 64,
        root=output,
    )
    binding = json.loads((first / "scratch_binding.json").read_text())

    assert first == second
    assert binding["process_independent_identity"] is True
    assert "process_id" not in binding


def test_output_recovery_is_only_label_free_and_same_admission(tmp_path: Path) -> None:
    root = tmp_path / "output"
    (root / "manifests").mkdir(parents=True)
    admission_hash = "c" * 64
    atomic_json(root / "manifests/admission.json", {"admission_hash": admission_hash})
    journal_body = {
        "schema_version": "midogpp_harp_v5_label_free_progress_journal_v1",
        "admission_hash": admission_hash,
        "phase": "LABEL_FREE_PHYSICAL_MENU",
        "labels_available": False,
        "entries": [],
    }
    atomic_json(
        root / "manifests/label_free_progress_journal.json",
        {**journal_body, "journal_hash": canonical_hash(journal_body)},
    )
    assert validate_pristine_or_label_free_recovery(
        root, admission_hash=admission_hash
    ) == "LABEL_FREE_RECOVERY"

    (root / "reports").mkdir()
    atomic_json(root / "reports/development_label_access.json", {"opened": True})
    with pytest.raises(ProtocolError, match="closed after a label capability"):
        validate_pristine_or_label_free_recovery(root, admission_hash=admission_hash)


def test_retired_publisher_fails_closed_before_input_access() -> None:
    arguments = [
        "publish-fixed-bank-harp-router-v5-amendment",
        "--config", "absent",
        "--expert-bank-root", "absent",
        "--generation-lock-root", "absent",
        "--prepared-cache-root", "absent",
        "--development-manifest", "absent",
        "--evaluation-manifest", "absent",
        "--parent-ledger", "absent",
        "--amendment-path", "absent",
        "--authorization-basis", "absent",
        "--authorization-date", "2026-09-01",
        "--repository-root", "absent",
    ]
    with pytest.raises(ProtocolError, match="direct amendment publication is disabled"):
        cli.main(arguments)
