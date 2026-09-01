from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.compatibility_conditioned_directional_router import (
    Direction,
    LearnabilityAdmission,
    ReplicaEnergyInput,
    RoutingDecision,
    SupportPartitionReceipt,
    TargetAction,
    build_compatibility_receipts,
    build_target_candidate_pool,
)
from midogpp_thesis.cvae.runtime.harp_v6_execution import production
from midogpp_thesis.cvae.runtime.harp_v6_execution.contracts import (
    ActionKind,
    ArtifactValue,
    HarpV6Pipeline,
    LabelFreeActionBlock,
    LabelFreeOuterMenu,
    PrelabelRouteSet,
    RoutedCase,
    compose_directional_soft_probability,
)
from midogpp_thesis.cvae.runtime.harp_v6_execution.directional_surfaces import (
    build_target_directional_actions,
    directional_probability_bytes,
)
from midogpp_thesis.cvae.runtime.harp_v6_execution.model_adapter import (
    RouterAdmissionState,
    RouterFitState,
    TargetEvidenceState,
)
from midogpp_thesis.cvae.runtime.harp_v6_execution.production import (
    HarpV6ProductionPipeline,
)
from midogpp_thesis.cvae.runtime.harp_v6_execution import terminal
from midogpp_thesis.cvae.runtime.harp_v6_execution.stores import (
    write_artifact_value,
    write_label_free_outer_menu,
    write_prelabel_routes,
)
from midogpp_thesis.cvae.runtime.harp_v6_execution.validation import (
    reconstruct_prelabel_routes,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
CENTERS = ("C", "H", "Q")
SAMPLES = ("s0", "s1", "s2", "s3")
CASES = ("case-0", "case-0", "case-1", "case-1")


def _block(
    *,
    role: str,
    query: str,
    kind: ActionKind,
    source: str | None,
    values: tuple[float, ...],
) -> LabelFreeActionBlock:
    return LabelFreeActionBlock(
        surface_role=role,
        outer_target_id="H",
        query_center_id=query,
        action_kind=kind,
        selected_source_id=source,
        sample_ids=SAMPLES,
        case_ids=CASES,
        probabilities=np.asarray(values, dtype=np.float32),
        seed_dispersion=np.asarray((0.01, 0.02, 0.03, 0.04), dtype=np.float32),
    )


def _menu() -> LabelFreeOuterMenu:
    baseline = (0.2, 0.8, 0.2, 0.8)
    uniform = (0.6, 0.4, 0.1, 0.9)
    expert_c = (0.7, 0.3, 0.1, 0.9)
    expert_q = (0.3, 0.7, 0.8, 0.2)
    blocks = (
        _block(
            role="development", query="Q", kind=ActionKind.B, source=None, values=baseline
        ),
        _block(
            role="development", query="Q", kind=ActionKind.U, source=None, values=uniform
        ),
        _block(
            role="development", query="Q", kind=ActionKind.HXE, source="C", values=expert_c
        ),
        _block(role="target", query="H", kind=ActionKind.B, source=None, values=baseline),
        _block(role="target", query="H", kind=ActionKind.U, source=None, values=uniform),
        _block(
            role="target", query="H", kind=ActionKind.HXE, source="C", values=expert_c
        ),
        _block(
            role="target", query="H", kind=ActionKind.HXE, source="Q", values=expert_q
        ),
    )
    return LabelFreeOuterMenu(
        outer_target_id="H",
        blocks=tuple(sorted(blocks, key=lambda row: row.key)),
        lineage={"physical": True},
    )


def _target_pool_and_compatibility():
    pool = build_target_candidate_pool(
        outer_target_id="H", all_center_ids=CENTERS, bank_lock_hash=SHA_A
    )
    partition = SupportPartitionReceipt(
        center_id="H",
        support_case_ids=("H-support",),
        evaluation_case_ids=("H-evaluation",),
        support_manifest_hash=SHA_A,
        evaluation_manifest_hash=SHA_B,
    )
    replicas = tuple(
        ReplicaEnergyInput(
            candidate_source_id=source,
            training_seed=seed,
            query_case_equal_energy=1.0 + 0.1 * source_index + 0.001 * seed_index,
            own_source_location=1.0,
            own_source_scale=0.5,
            checkpoint_hash=SHA_A,
            source_frame_hash=SHA_B,
            sampler_hash=SHA_C,
        )
        for source_index, source in enumerate(pool.candidate_center_ids)
        for seed_index, seed in enumerate((17, 42, 101))
    )
    return pool, build_compatibility_receipts(
        candidate_pool=pool,
        support_partition=partition,
        replica_energies=replicas,
    )


def _target_actions() -> tuple[TargetAction, ...]:
    pool, receipts = _target_pool_and_compatibility()
    return build_target_directional_actions(
        _menu(), candidate_pool=pool, compatibility_receipts=receipts
    )


def _fit_artifact() -> ArtifactValue:
    # Routing only checks the sealed fitted-state type; it cannot read the
    # fitted coefficients once target evidence has been durably materialized.
    return ArtifactValue(
        state=object.__new__(RouterFitState), manifest={"model_hash": SHA_B}
    )


def _passing_admission() -> LearnabilityAdmission:
    return LearnabilityAdmission(
        passed=True,
        center_ids=("A", "B", "C", "D"),
        case_count=12,
        sign_accuracy=0.75,
        top1_accuracy=0.5,
        minimum_delete_center_tau=0.1,
        safe_coverage=0.25,
        selected_count=3,
        harmful_selected_count=0,
        proper_loss_violation_count=0,
        reasons=(),
        source_oof_hash=SHA_A,
    )


def _admission_artifact() -> ArtifactValue:
    state = RouterAdmissionState((('H', _passing_admission()),), True)
    return ArtifactValue(state=state, manifest={"admission_hash": SHA_C})


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        model={
            "policy": {"proper_loss_safe_vs_B": True},
            "soft_top_k": 2,
            "soft_mixture_lambda": 0.5,
            "opportunity_probability_threshold": 0.5,
            "softmax_temperature": 0.25,
        }
    )


def _complete_target_artifact(monkeypatch: pytest.MonkeyPatch) -> ArtifactValue:
    pool, receipts = _target_pool_and_compatibility()

    class CompatibilityState:
        def pool(self, outer: str, query: str):
            assert (outer, query) == ("H", "H")
            return pool

        def receipt(self, outer: str, query: str, source: str):
            assert (outer, query) == ("H", "H")
            return next(row for row in receipts if row.candidate_source_id == source)

    monkeypatch.setattr(
        production, "compatibility_state_from_artifact", lambda value: CompatibilityState()
    )
    monkeypatch.setattr(
        production,
        "predict_target_evidence",
        lambda rows, fitted: TargetEvidenceState(tuple(rows), (), ()),
    )
    return HarpV6ProductionPipeline(
        development_role="development", evaluation_role="evaluation"
    ).build_complete_target_case_actions(
        (_menu(),),
        ArtifactValue(state=None, manifest={"compatibility_hash": SHA_A}),
        _fit_artifact(),
        _admission_artifact(),
        config=SimpleNamespace(),
    )


def _action_values(action: TargetAction) -> np.ndarray:
    return np.frombuffer(b"".join(action.probability_bytes), dtype="<f4").astype(
        np.float32, copy=True
    )


def _single_case_route(
    *,
    target_hash: str,
    action_id: str,
    component: np.ndarray,
) -> PrelabelRouteSet:
    menu = _menu()
    baseline_block = menu.target_block(ActionKind.B)
    uniform_block = menu.target_block(ActionKind.U)
    indices = np.flatnonzero(
        np.asarray(baseline_block.case_ids, dtype=object) == "case-0"
    )
    baseline = np.asarray(baseline_block.probabilities[indices], dtype=np.float32)
    uniform = np.asarray(uniform_block.probabilities[indices], dtype=np.float32)
    selected, routed = compose_directional_soft_probability(
        baseline,
        (component,),
        (1.0,),
        direction="D01",
        shrinkage=0.5,
    )
    case = RoutedCase(
        outer_target_id="H",
        case_id="case-0",
        sample_ids=tuple(baseline_block.sample_ids[int(index)] for index in indices),
        selected_kind=ActionKind.HXE,
        selected_source_id="C",
        reason="FRESH_BINDING_REGRESSION",
        baseline_probabilities=baseline,
        uniform_probabilities=uniform,
        selected_probabilities=selected,
        routed_probabilities=routed,
        direction="D01",
        shrinkage=0.5,
        component_action_ids=(action_id,),
        component_weights=(1.0,),
        component_probabilities=(component,),
        decision_payload={"test_only": True},
    )
    return PrelabelRouteSet(
        cases=(case,),
        policy_hash=SHA_A,
        model_hash=SHA_B,
        target_action_hash=target_hash,
    )


def test_production_implements_protocol_signatures_without_stale_harp_imports() -> None:
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
        expected = inspect.signature(getattr(HarpV6Pipeline, method)).parameters
        actual = inspect.signature(getattr(HarpV6ProductionPipeline, method)).parameters
        assert tuple(actual) == tuple(expected)
        assert tuple(row.kind for row in actual.values()) == tuple(
            row.kind for row in expected.values()
        )

    source = Path(production.__file__).read_text(encoding="utf-8")
    stale_modules = (
        "harp_v1_execution",
        "harp_v2_execution",
        "harp_v3_execution",
        "harp_v4_execution",
        "harp_v5_execution",
        "fixed_bank_harp_router_v1",
        "fixed_bank_harp_router_v2",
        "fixed_bank_harp_router_v3",
        "fixed_bank_harp_router_v4",
        "fixed_bank_harp_router_v5",
        "routing.harp_v6",
    )
    assert not any(module in source for module in stale_modules)


def test_production_remains_a_thin_orchestration_boundary() -> None:
    source = Path(production.__file__).read_text(encoding="utf-8")
    assert len(source.splitlines()) < 260
    assert "numpy" not in source
    assert "build_label_free_opportunity(" not in source
    assert "compose_directional_soft_probability(" not in source
    assert "source_response_hashes" not in source
    assert "decision_payloads" not in source
    for module in (
        "production_validation",
        "source_development",
        "source_model_artifacts",
        "target_action_artifacts",
        "routing_artifacts",
    ):
        assert f"from .{module} import" in source


def test_label_free_target_path_keeps_complete_inventory_after_opportunity_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actions = _target_actions()
    pool, receipts = _target_pool_and_compatibility()

    class CompatibilityState:
        def pool(self, outer: str, query: str):
            assert (outer, query) == ("H", "H")
            return pool

        def receipt(self, outer: str, query: str, source: str):
            assert (outer, query) == ("H", "H")
            return next(row for row in receipts if row.candidate_source_id == source)

    monkeypatch.setattr(
        production, "compatibility_state_from_artifact", lambda value: CompatibilityState()
    )
    monkeypatch.setattr(
        production,
        "predict_target_evidence",
        lambda rows, fitted: TargetEvidenceState(tuple(rows), (), ()),
    )
    pipeline = HarpV6ProductionPipeline(
        development_role="development", evaluation_role="evaluation"
    )
    output = pipeline.build_complete_target_case_actions(
        (_menu(),),
        ArtifactValue(state=None, manifest={"compatibility_hash": SHA_A}),
        _fit_artifact(),
        _admission_artifact(),
        config=SimpleNamespace(),
    )

    assert all(
        "label" not in name
        for name in inspect.signature(
            HarpV6ProductionPipeline.build_complete_target_case_actions
        ).parameters
    )
    assert output.manifest["evaluation_labels_used"] is False
    assert output.manifest["complete_actions_retained_for_audit"] is True
    assert len(output.state.actions) == len(actions) == 18
    assert {row.target_action_hash for row in output.state.actions} == {
        row.target_action_hash for row in actions
    }
    assert 0 < output.manifest["active_label_free_action_count"] < len(actions)
    assert len(output.manifest["rows"]) == len(actions)
    assert output.state.evidence == ()


def test_route_wiring_composes_one_direction_and_preserves_exact_b_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actions = _target_actions()
    selected = next(
        row
        for row in actions
        if row.feature.case_id == "case-0" and row.feature.action_id == "HXE:C:D01"
    )

    def choose(*args, outer_target_id: str, case_id: str, admission, **kwargs):
        del args, kwargs
        if case_id == "case-0":
            return RoutingDecision(
                outer_target_id=outer_target_id,
                case_id=case_id,
                enabled=True,
                selected_direction=Direction.D01,
                selected_action_ids=(selected.feature.action_id,),
                selected_weights=(1.0,),
                mixture_lambda=0.5,
                reason="TEST_DIRECTIONAL_ROUTE",
                admission_hash=admission.admission_hash,
                evidence_hashes=(SHA_D,),
            )
        return RoutingDecision(
            outer_target_id=outer_target_id,
            case_id=case_id,
            enabled=False,
            selected_direction=None,
            selected_action_ids=(),
            selected_weights=(),
            mixture_lambda=0.0,
            reason="TEST_EXACT_B_FALLBACK",
            admission_hash=admission.admission_hash,
            evidence_hashes=(),
        )

    monkeypatch.setattr(production, "select_baseline_anchored_route", choose)
    target = ArtifactValue(
        state=TargetEvidenceState(actions, (), ()),
        manifest={"target_action_hash": SHA_D},
    )
    routes = HarpV6ProductionPipeline(
        development_role="development", evaluation_role="evaluation"
    ).route_case_actions(
        (_menu(),),
        target,
        _fit_artifact(),
        _admission_artifact(),
        config=_config(),
    )

    enabled, fallback = routes.cases
    assert enabled.case_id == "case-0"
    assert enabled.selected_kind is ActionKind.HXE
    assert enabled.selected_source_id == "C"
    assert enabled.direction == "D01"
    assert enabled.routed_probabilities == pytest.approx(
        np.asarray((0.45, 0.8), dtype=np.float32)
    )
    assert enabled.routed_probabilities[1:].tobytes() == enabled.baseline_probabilities[
        1:
    ].tobytes()

    assert fallback.case_id == "case-1"
    assert fallback.selected_kind is ActionKind.B
    assert fallback.routed_probabilities.tobytes() == fallback.baseline_probabilities.tobytes()
    assert fallback.selected_probabilities.tobytes() == fallback.baseline_probabilities.tobytes()


@pytest.mark.parametrize("direction", (Direction.D01, Direction.D10, Direction.ALL))
@pytest.mark.parametrize("case_id", ("case-0", "case-1"))
def test_terminal_oracle_direction_masks_match_sealed_directional_surfaces(
    direction: Direction, case_id: str
) -> None:
    menu = _menu()
    baseline = menu.target_block(ActionKind.B)
    challenger = menu.target_block(ActionKind.HXE, "C")
    sample_ids, expected_cells = directional_probability_bytes(
        baseline, challenger, case_id=case_id, direction=direction
    )
    indices = np.asarray(
        [baseline.sample_ids.index(sample_id) for sample_id in sample_ids], dtype=np.int64
    )
    observed = terminal._directional_surface(
        baseline.probabilities[indices],
        challenger.probabilities[indices],
        direction.value,
    )
    expected = np.frombuffer(b"".join(expected_cells), dtype="<f4")
    assert observed.tobytes(order="C") == expected.tobytes(order="C")


@pytest.mark.parametrize("tamper", ("component_bytes", "action_id"))
def test_fresh_validation_binds_route_components_to_named_persisted_target_actions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    target = _complete_target_artifact(monkeypatch)
    actions = target.state.actions
    named = next(
        row
        for row in actions
        if row.feature.case_id == "case-0" and row.feature.action_id == "HXE:C:D01"
    )
    other_direction = next(
        row
        for row in actions
        if row.feature.case_id == "case-0" and row.feature.action_id == "HXE:C:D10"
    )
    target_hash = str(target.manifest["target_action_hash"])

    menu_root = tmp_path / "menu-H"
    development_root = tmp_path / "development"
    model_root = tmp_path / "model"
    target_root = tmp_path / "target"
    write_label_free_outer_menu(menu_root, _menu())
    write_artifact_value(
        development_root,
        ArtifactValue(state=None, manifest={"surface_hash": SHA_A}),
        role="source_development_case_surface",
    )
    write_artifact_value(
        model_root,
        ArtifactValue(state=None, manifest={"model_hash": SHA_B}),
        role="source_only_router",
    )
    write_artifact_value(
        target_root, target, role="complete_target_case_actions"
    )

    clean = _single_case_route(
        target_hash=target_hash,
        action_id=named.feature.action_id,
        component=_action_values(named),
    )
    clean_root = tmp_path / "clean-routes"
    write_prelabel_routes(clean_root, clean)
    receipt = reconstruct_prelabel_routes(
        clean_root,
        {"H": menu_root},
        development_root,
        model_root,
        target_root,
        validator_id="clean_binding_control",
        expected_center_ids=("H",),
        expected_config_hash=SHA_D,
    )
    assert receipt["soft_formula_reconstructed"] is True

    tampered = _single_case_route(
        target_hash=target_hash,
        action_id=(
            other_direction.feature.action_id
            if tamper == "action_id"
            else named.feature.action_id
        ),
        component=(
            _action_values(other_direction)
            if tamper == "component_bytes"
            else _action_values(named)
        ),
    )
    tampered_root = tmp_path / f"tampered-routes-{tamper}"
    write_prelabel_routes(tampered_root, tampered)
    with pytest.raises(
        ProtocolError, match="not the named persisted target action"
    ):
        reconstruct_prelabel_routes(
            tampered_root,
            {"H": menu_root},
            development_root,
            model_root,
            target_root,
            validator_id=f"tampered_binding_{tamper}",
            expected_center_ids=("H",),
            expected_config_hash=SHA_D,
        )
