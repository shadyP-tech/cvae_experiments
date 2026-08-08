from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.residual_topup_policy.bundle import REQUIRED_FILES
from midogpp_thesis.cvae.routing.residual_topup_policy.config import (
    EXPERIMENT_ID,
    INPUT_ARTIFACT_IDS,
    OUTPUT_ARTIFACT_ID,
    PROXY_SURFACE_ARTIFACT_ID,
    load_residual_topup_policy_lock_config,
)
from midogpp_thesis.cvae.routing.residual_topup_policy.workspace_binding import (
    CONFIG_PATH,
    OUTPUT_CANONICAL_PATH,
    PROXY_CANONICAL_PATH,
    PROXY_REQUIRED_FILES,
    validate_planned_workspace_contract,
    validate_production_workspace_binding,
)
from midogpp_thesis.workspace.runtime import MidogppWorkspace, WorkspaceError


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / CONFIG_PATH


def test_fresh_stage60_workspace_entries_are_exact_and_remain_planned() -> None:
    workspace = MidogppWorkspace.load()
    workspace.validate()
    config = load_residual_topup_policy_lock_config(CONFIG)
    validate_planned_workspace_contract(config, _workspace=workspace)

    experiment = workspace.get_experiment(EXPERIMENT_ID)
    proxy = workspace.artifacts[PROXY_SURFACE_ARTIFACT_ID]
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]
    assert experiment.status == "planned"
    assert experiment.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert experiment.output_artifact_id == OUTPUT_ARTIFACT_ID
    assert experiment.config_path == CONFIG_PATH
    assert proxy.availability == "planned"
    assert proxy.canonical_path == PROXY_CANONICAL_PATH
    assert proxy.required_files == PROXY_REQUIRED_FILES
    assert proxy.semantic_identities["fresh_evidence"] == "true"
    assert proxy.semantic_identities["target_support_evaluation_case_disjoint"] == "true"
    assert output.availability == "planned"
    assert output.canonical_path == OUTPUT_CANONICAL_PATH
    assert output.required_files == REQUIRED_FILES
    assert output.semantic_identities["policy_frozen_before_stage70"] == "true"

    with pytest.raises(WorkspaceError, match="status='planned'"):
        workspace.prepare(EXPERIMENT_ID, require_inputs=False)
    with pytest.raises(ProtocolError, match="status='planned'"):
        validate_production_workspace_binding(config, _workspace=workspace)


def test_stage60_config_resolves_every_declared_input_and_only_that_input_set() -> None:
    workspace = MidogppWorkspace.load()
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    used: set[str] = set()
    resolved = workspace.resolve_value(
        payload,
        require_inputs=False,
        used_inputs=used,
    )
    assert used == set(INPUT_ARTIFACT_IDS)
    assert resolved["experiment"]["artifact_root"] == str(
        (workspace.repo_root / OUTPUT_CANONICAL_PATH).resolve()
    )
    assert resolved["inputs"]["proxy_surface_root"] == str(
        (workspace.repo_root / PROXY_CANONICAL_PATH).resolve()
    )
    assert resolved["inputs"]["proxy_score_table_path"].endswith(
        PROXY_REQUIRED_FILES[0]
    )
    assert resolved["inputs"]["proxy_attestation_path"].endswith(
        PROXY_REQUIRED_FILES[1]
    )


def test_permutation_scheme_and_index_are_exact_and_drift_rejected(
    tmp_path: Path,
) -> None:
    config = load_residual_topup_policy_lock_config(CONFIG)
    assert config.actions["permutation_scheme"] == (
        "canonical_source_order_nonzero_cyclic_rotation"
    )
    assert config.permutation_index == 1

    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["actions"]["permutation_index"] = 2
    candidate = tmp_path / "drifted.yaml"
    candidate.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="actions drifted"):
        load_residual_topup_policy_lock_config(candidate)

