from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from midogpp_thesis.cvae.frozen_policy_downstream import cli as downstream_cli
from midogpp_thesis.cvae.frozen_policy_downstream.utility_aligned_residual_fresh.bundle_contracts import (
    REQUIRED_FILES as STAGE70_REQUIRED_FILES,
)
from midogpp_thesis.cvae.frozen_policy_downstream.utility_aligned_residual_fresh.config import (
    EXPERIMENT_ID as STAGE70_EXPERIMENT_ID,
    INPUT_ARTIFACT_IDS as STAGE70_INPUTS,
    OUTPUT_ARTIFACT_ID as STAGE70_OUTPUT,
    load_utility_aligned_residual_fresh_config,
)
from midogpp_thesis.cvae.frozen_policy_downstream.utility_aligned_residual_fresh.workspace_binding import (
    validate_utility_aligned_residual_fresh_workspace_binding,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing import cli as routing_cli
from midogpp_thesis.cvae.routing.exact_tail_utility_surface.bundle import (
    REQUIRED_FILES as EXACT_REQUIRED_FILES,
)
from midogpp_thesis.cvae.routing.exact_tail_utility_surface.config import (
    INPUT_ARTIFACT_IDS as EXACT_INPUTS,
    load_exact_tail_utility_surface_config,
)
from midogpp_thesis.cvae.routing.exact_tail_utility_surface.contracts import (
    EXPERIMENT_ID as EXACT_EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID as EXACT_OUTPUT,
)
from midogpp_thesis.cvae.routing.exact_tail_utility_surface.workspace_binding import (
    validate_production_workspace_binding as validate_exact_workspace,
)
from midogpp_thesis.cvae.routing.utility_aligned_residual_policy.bundle import (
    REQUIRED_FILES as POLICY_REQUIRED_FILES,
)
from midogpp_thesis.cvae.routing.utility_aligned_residual_policy.config import (
    load_utility_aligned_residual_policy_config,
)
from midogpp_thesis.cvae.routing.utility_aligned_residual_policy.contracts import (
    EXPERIMENT_ID as POLICY_EXPERIMENT_ID,
    INPUT_ARTIFACT_IDS as POLICY_INPUTS,
    OUTPUT_ARTIFACT_ID as POLICY_OUTPUT,
    TARGET_RESERVATION_ARTIFACT_ID,
    TARGET_SUPPORT_PARENT_RESERVATION_ARTIFACT_ID,
)
from midogpp_thesis.cvae.routing.utility_aligned_residual_policy.workspace_binding import (
    validate_production_workspace_binding as validate_policy_workspace,
)
from midogpp_thesis.cvae.routing.utility_aligned_target_support_surface.config import (
    load_utility_aligned_target_support_surface_config,
)
from midogpp_thesis.cvae.routing.utility_aligned_target_support_surface.contracts import (
    EXPERIMENT_ID as SUPPORT_EXPERIMENT_ID,
    INPUT_ARTIFACT_IDS as SUPPORT_INPUTS,
    OUTPUT_ARTIFACT_ID as SUPPORT_OUTPUT,
    REQUIRED_FILES as SUPPORT_REQUIRED_FILES,
)
from midogpp_thesis.cvae.routing.utility_aligned_target_support_surface.workspace_binding import (
    validate_production_workspace_binding as validate_support_workspace,
)
from midogpp_thesis.workspace.runtime import MidogppWorkspace, WorkspaceError


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = {
    EXACT_EXPERIMENT_ID: ROOT
    / "experiments/midogpp/stages/60_routing_and_composition/configs"
    / "uniform_b_v2_exact_tail_utility_surface_v1.yaml",
    SUPPORT_EXPERIMENT_ID: ROOT
    / "experiments/midogpp/stages/60_routing_and_composition/configs"
    / "uniform_b_v2_utility_aligned_target_support_surface_v1.yaml",
    POLICY_EXPERIMENT_ID: ROOT
    / "experiments/midogpp/stages/60_routing_and_composition/configs"
    / "uniform_b_v2_utility_aligned_residual_policy_lock_v1.yaml",
    STAGE70_EXPERIMENT_ID: ROOT
    / "experiments/midogpp/stages/70_frozen_policy_downstream/configs"
    / "uniform_b_v2_utility_aligned_residual_fresh_v1.yaml",
}
EXPERIMENTS = (
    (EXACT_EXPERIMENT_ID, EXACT_INPUTS, EXACT_OUTPUT, EXACT_REQUIRED_FILES),
    (SUPPORT_EXPERIMENT_ID, SUPPORT_INPUTS, SUPPORT_OUTPUT, SUPPORT_REQUIRED_FILES),
    (POLICY_EXPERIMENT_ID, POLICY_INPUTS, POLICY_OUTPUT, POLICY_REQUIRED_FILES),
    (STAGE70_EXPERIMENT_ID, STAGE70_INPUTS, STAGE70_OUTPUT, STAGE70_REQUIRED_FILES),
)


def test_workspace_graph_is_complete_exact_and_still_planned() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()

    for experiment_id, inputs, output_id, required_files in EXPERIMENTS:
        experiment = workspace.get_experiment(experiment_id)
        output = workspace.artifacts[output_id]
        assert experiment.status == "planned"
        assert experiment.input_artifact_ids == inputs
        assert experiment.output_artifact_id == output_id
        assert output.availability == "planned"
        assert output.required_files == required_files
        with pytest.raises(WorkspaceError, match="status='planned'"):
            workspace.prepare(experiment_id, require_inputs=False)


def test_all_configs_resolve_exactly_their_declared_inputs() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    for experiment_id, inputs, output_id, _ in EXPERIMENTS:
        payload = yaml.safe_load(CONFIGS[experiment_id].read_text(encoding="utf-8"))
        used: set[str] = set()
        resolved = workspace.resolve_value(
            payload,
            require_inputs=False,
            used_inputs=used,
        )
        assert used == set(inputs)
        assert Path(resolved["experiment"]["artifact_root"]).resolve() == (
            ROOT / workspace.artifacts[output_id].canonical_path
        ).resolve()


def test_planned_registry_blocks_every_production_entrypoint() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    exact = load_exact_tail_utility_surface_config(CONFIGS[EXACT_EXPERIMENT_ID])
    support = load_utility_aligned_target_support_surface_config(
        CONFIGS[SUPPORT_EXPERIMENT_ID]
    )
    policy = load_utility_aligned_residual_policy_config(CONFIGS[POLICY_EXPERIMENT_ID])
    stage70 = load_utility_aligned_residual_fresh_config(CONFIGS[STAGE70_EXPERIMENT_ID])

    with pytest.raises(ProtocolError, match="planned"):
        validate_exact_workspace(exact, _workspace=workspace)
    with pytest.raises(ProtocolError, match="planned"):
        validate_support_workspace(support, _workspace=workspace)
    with pytest.raises(ProtocolError, match="planned"):
        validate_policy_workspace(policy, _workspace=workspace)
    with pytest.raises(ProtocolError, match="not active"):
        validate_utility_aligned_residual_fresh_workspace_binding(stage70)


def test_support_parent_and_stage70_reservations_are_distinct_and_fenced() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    assert TARGET_SUPPORT_PARENT_RESERVATION_ARTIFACT_ID != (
        TARGET_RESERVATION_ARTIFACT_ID
    )
    assert POLICY_INPUTS[3:5] == (
        TARGET_SUPPORT_PARENT_RESERVATION_ARTIFACT_ID,
        TARGET_RESERVATION_ARTIFACT_ID,
    )
    support_parent = workspace.artifacts[TARGET_SUPPORT_PARENT_RESERVATION_ARTIFACT_ID]
    stage70 = workspace.artifacts[TARGET_RESERVATION_ARTIFACT_ID]
    assert support_parent.semantic_identities[
        "authorized_consumer_experiment_ids"
    ] == f"{SUPPORT_EXPERIMENT_ID}|{POLICY_EXPERIMENT_ID}"
    assert stage70.semantic_identities[
        "authorized_consumer_experiment_ids"
    ] == f"{POLICY_EXPERIMENT_ID}|{STAGE70_EXPERIMENT_ID}"
    assert support_parent.semantic_identities["labels_persisted"] == "false"
    assert support_parent.semantic_identities[
        "target_evaluation_embeddings_available"
    ] == "false"


def test_new_family_has_no_consumed_stage70_or_stage90_input() -> None:
    forbidden_tokens = ("consumed_validation", "stage90", "oracle")
    for _, inputs, _, _ in EXPERIMENTS:
        assert not any(
            token in artifact_id
            for artifact_id in inputs
            for token in forbidden_tokens
        )


def test_cli_surfaces_match_registered_runner_identities() -> None:
    routing = routing_cli.main
    assert callable(routing)
    for surface in (
        "uniform-b-v2-exact-tail-utility-surface",
        "uniform-b-v2-utility-aligned-target-support-surface",
        "uniform-b-v2-utility-aligned-residual-policy-lock",
    ):
        with pytest.raises(SystemExit) as exit_info:
            routing_cli.main([surface, "--help"])
        assert exit_info.value.code == 0
    with pytest.raises(SystemExit) as exit_info:
        downstream_cli.main(["evaluate-utility-aligned-residual-fresh", "--help"])
    assert exit_info.value.code == 0
