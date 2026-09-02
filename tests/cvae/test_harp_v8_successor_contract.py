from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v8 import (
    ACTIVATION_CONFIRMATION,
    PREPARATION_CONFIRMATION,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v8.config import (
    load_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v8.execution.admission import (
    validate_pristine_or_label_free_recovery,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v8.runner import (
    HARP_V8_RUN_CONFIRMATION_TOKEN,
    dry_run_harp_stage90_v8,
    inspect_harp_stage90_v8,
    run_harp_stage90_v8,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v8.source_seal import (
    source_members,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.harp_v8_execution import production
from midogpp_thesis.cvae.runtime.harp_v8_execution.contracts import (
    ActionKind,
    HarpV8Pipeline,
    PrelabelRouteSet,
    RoutedCase,
)
from midogpp_thesis.cvae.runtime.harp_v8_execution.phases import PHASE_ORDER
from midogpp_thesis.cvae.runtime.harp_v8_execution.production import (
    HarpV8ProductionPipeline,
)
from midogpp_thesis.cvae.runtime.harp_v8_execution.production_validation import (
    validate_model_config,
)
from midogpp_thesis.cvae.runtime.harp_v8_execution.stores import (
    read_prelabel_routes,
    write_prelabel_routes,
)
from midogpp_thesis.workspace.runtime import MidogppWorkspace


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_v2_consumed_test_fixed_bank_harp_router_v8.yaml"
)
EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_harp_router.v8"
)
OUTPUT_ID = "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_harp_router_v8"
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def test_planned_successor_is_path_free_and_does_not_authorize_execution() -> None:
    config = load_config(CONFIG)
    inspection = inspect_harp_stage90_v8(config)
    dry_run = dry_run_harp_stage90_v8(config, artifact_root="must-not-resolve")

    assert inspection["status"] == "PLANNED_NEEDS_SEPARATE_EXECUTION_AMENDMENT"
    assert inspection["paths_resolved"] is False
    assert inspection["phase_order"] == list(PHASE_ORDER)
    assert inspection["fresh_evidence"] is False
    assert dry_run["status"] == "NEEDS_SEPARATE_EXECUTION_AMENDMENT"
    assert dry_run["authorization_probed"] is False
    assert dry_run["filesystem_mutations"] == 0
    assert config.execution_authorized is False


def test_canonical_model_contract_passes_production_validation() -> None:
    config = load_config(CONFIG)
    validate_model_config(config)
    assert config.model["safe_action_set"] == (
        "harm_ucb_brier_ucb_log_ucb_and_harm_proper_loss_certified"
    )


def test_v8_lifecycle_tokens_are_distinct_and_run_fails_before_path_access() -> None:
    assert PREPARATION_CONFIRMATION == "PREPARE_HARP_V8_CONSUMED_TEST_INPUTS"
    assert ACTIVATION_CONFIRMATION == (
        "ACTIVATE_HARP_V8_TERMINAL_CONSUMED_TEST_DIAGNOSTIC"
    )
    assert HARP_V8_RUN_CONFIRMATION_TOKEN == (
        "RUN_HARP_V8_TERMINAL_CONSUMED_TEST_DIAGNOSTIC"
    )
    assert len(
        {
            PREPARATION_CONFIRMATION,
            ACTIVATION_CONFIRMATION,
            HARP_V8_RUN_CONFIRMATION_TOKEN,
        }
    ) == 3
    with pytest.raises(ProtocolError, match="exact confirmation token"):
        run_harp_stage90_v8(
            object(),  # type: ignore[arg-type]
            artifact_root="must-not-resolve",
            confirmation_token="wrong",
        )


def test_workspace_contract_registers_independent_v8_replay_artifacts() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    experiment = workspace.experiments[EXPERIMENT_ID]
    output = workspace.artifacts[OUTPUT_ID]

    assert experiment.status == "planned"
    assert experiment.runnable is False
    assert len(experiment.input_artifact_ids) == 7
    assert all("harp_router_v7" not in value for value in experiment.input_artifact_ids)
    assert output.semantic_identities["execution_authorized"] == "false"
    required = set(output.required_files)
    assert "stores/effective_menu/manifest.json" in required
    assert "stores/source_only_policy_oof_replay/arrays.npz" in required
    assert "manifests/source_policy_admission_seal.json" in required
    assert "manifests/learnability_admission_seal.json" not in required


def test_production_boundary_is_modular_and_predecessor_free() -> None:
    methods = (
        "preflight",
        "materialize_label_free_outer_menus",
        "materialize_label_free_support_compatibility",
        "build_development_case_surface",
        "fit_source_only_router",
        "admit_source_only_router",
        "build_complete_target_case_actions",
        "route_case_actions",
        "evaluate_terminal",
    )
    for method in methods:
        expected = inspect.signature(getattr(HarpV8Pipeline, method)).parameters
        actual = inspect.signature(getattr(HarpV8ProductionPipeline, method)).parameters
        assert tuple(actual) == tuple(expected)
        assert tuple(row.kind for row in actual.values()) == tuple(
            row.kind for row in expected.values()
        )
    source = Path(production.__file__).read_text(encoding="utf-8")
    assert "compatibility_conditioned_directional_router" not in source
    assert "soft_top_k" not in source
    assert "compose_directional_soft_probability" not in source
    assert len(source.splitlines()) < 260

    members = source_members(ROOT)
    relative = {path.relative_to(ROOT / "src").as_posix() for path in members}
    assert any("baseline_inclusive_action_safe_router_v8" in path for path in relative)
    for predecessor in range(1, 8):
        assert not any(
            f"fixed_bank_harp_router_v{predecessor}" in path for path in relative
        )
        assert not any(f"harp_v{predecessor}_execution" in path for path in relative)
    assert not any("source_active_selective_router_v7" in path for path in relative)

    science_root = (
        ROOT
        / "src/midogpp_thesis/cvae/routing/"
        "baseline_inclusive_action_safe_router_v8"
    )
    science_source = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(science_root.glob("*.py"))
    )
    assert "source_active_selective_router_v7" not in science_source


def _routed_case(*, shrinkage: float = 1.0) -> RoutedCase:
    baseline = np.asarray((0.2, 0.8), dtype=np.float32)
    uniform = np.asarray((0.6, 0.4), dtype=np.float32)
    selected = np.asarray((0.7, 0.3), dtype=np.float32)
    return RoutedCase(
        outer_target_id="H",
        case_id="case-0",
        sample_ids=("s0", "s1"),
        selected_kind=ActionKind.HXE,
        selected_source_id="C",
        reason="ROUTED_CERTIFIED_EXACT_TOP1",
        baseline_probabilities=baseline,
        uniform_probabilities=uniform,
        selected_probabilities=selected,
        routed_probabilities=selected.copy(),
        direction="D01",
        shrinkage=shrinkage,
        component_action_ids=("HXE:C:D01",),
        component_weights=(1.0,),
        component_probabilities=(selected.copy(),),
        decision_payload={
            "deployed_action": "CERTIFIED_EXACT_TOP1_PHYSICAL_OR_EXACT_B",
            "failed_gates": [],
        },
    )


def test_exact_top1_route_store_roundtrip_and_mixture_rejection(tmp_path: Path) -> None:
    case = _routed_case()
    routes = PrelabelRouteSet(
        cases=(case,),
        policy_hash=SHA_A,
        model_hash=SHA_B,
        target_action_hash=SHA_C,
    )
    write_prelabel_routes(tmp_path / "routes", routes)
    restored = read_prelabel_routes(tmp_path / "routes")

    assert restored.route_hash == routes.route_hash
    assert restored.cases[0].component_weights == (1.0,)
    assert restored.cases[0].shrinkage == 1.0
    assert (
        restored.cases[0].routed_probabilities.tobytes(order="C")
        == case.selected_probabilities.tobytes(order="C")
    )
    with pytest.raises(ProtocolError, match="exact-top-1"):
        _routed_case(shrinkage=0.5)


def test_effective_menu_crash_window_remains_label_free_recoverable(
    tmp_path: Path,
) -> None:
    admission_hash = "d" * 64
    manifests = tmp_path / "manifests"
    effective = tmp_path / "stores/effective_menu"
    manifests.mkdir(parents=True)
    effective.mkdir(parents=True)
    (manifests / "admission.json").write_text(
        '{"admission_hash":"' + admission_hash + '"}\n', encoding="utf-8"
    )
    (manifests / "protocol_manifest.json").write_text("{}\n", encoding="utf-8")
    (manifests / "label_free_progress_journal.json").write_text(
        '{"admission_hash":"'
        + admission_hash
        + '","labels_available":false}\n',
        encoding="utf-8",
    )
    (effective / "manifest.json").write_text("{}\n", encoding="utf-8")
    (effective / "arrays.npz").write_bytes(b"label-free-effective-menu")

    assert (
        validate_pristine_or_label_free_recovery(
            tmp_path,
            admission_hash=admission_hash,
        )
        == "LABEL_FREE_RECOVERY"
    )

    forbidden = tmp_path / "stores/development_case_surface/manifest.json"
    forbidden.parent.mkdir(parents=True)
    forbidden.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="closed after a label capability"):
        validate_pristine_or_label_free_recovery(
            tmp_path,
            admission_hash=admission_hash,
        )
