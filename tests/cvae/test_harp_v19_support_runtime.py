from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.safe_winner_router_v19 import (
    Direction,
    RouteDecision,
    RouterFitConfig,
    SurfaceRole,
    build_baseline_composite,
    build_exact_u_composite,
    build_soft_topk_composite,
)
from midogpp_thesis.cvae.runtime.harp_v19_execution.contracts import (
    ActionKind,
    LabelFreeActionBlock,
    LabelFreeOuterMenu,
)
from midogpp_thesis.cvae.runtime.harp_v19_execution.prelabel_diagnostics import (
    build_prelabel_diagnostics,
)
from midogpp_thesis.cvae.runtime.harp_v19_execution.journal import (
    LabelFreeProgressJournal,
)
from midogpp_thesis.cvae.runtime.harp_v19_execution.support_model_artifacts import (
    _as_router_config,
    build_support_outcome_artifact,
    build_support_router_artifact,
    build_support_target_routes,
    report_support_router_artifact,
)
from midogpp_thesis.cvae.runtime.harp_v19_execution.support_target_adapter import (
    FULL_U_ARM_ID,
    LABEL_FREE_FEATURE_NAMES,
    attach_support_outcome_inventory,
    compile_support_target_menus,
    route_target_bundle,
)
from midogpp_thesis.cvae.runtime.harp_v19_execution.stores import (
    write_label_free_outer_menu,
)


def _identities(
    role: str, center: str, case_count: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    cases = tuple(f"{role}-{center}-case-{index:02d}" for index in range(case_count))
    return (
        tuple(f"{case}-sample-{index}" for case in cases for index in range(4)),
        tuple(case for case in cases for _index in range(4)),
    )


def _probabilities(
    center: str,
    kind: ActionKind,
    source: str | None,
    case_count: int,
) -> np.ndarray:
    if kind is ActionKind.B:
        pair = (0.20, 0.30, 0.80, 0.70)
    elif kind is ActionKind.U:
        pair = (0.58, 0.57, 0.42, 0.43)
    else:
        candidates = tuple(value for value in CENTERS if value != center)
        position = candidates.index(str(source))
        pair = (
            0.60 + 0.02 * position,
            0.56 + 0.02 * position,
            0.40 - 0.02 * position,
            0.44 - 0.02 * position,
        )
    return np.asarray(pair * case_count, dtype=np.float32)


def _block(
    center: str,
    role: str,
    kind: ActionKind,
    source: str | None,
    *,
    case_count: int,
) -> LabelFreeActionBlock:
    prefix = "source" if role == "source_train" else "target"
    samples, cases = _identities(prefix, center, case_count)
    values = _probabilities(center, kind, source, case_count)
    return LabelFreeActionBlock(
        surface_role=role,
        outer_target_id=center,
        query_center_id=center,
        action_kind=kind,
        selected_source_id=source,
        sample_ids=samples,
        case_ids=cases,
        probabilities=values,
        seed_dispersion=np.full(len(values), 0.01, dtype=np.float32),
    )


def _physical_menu(
    center: str, *, source_cases: int = 8, target_cases: int = 1
) -> LabelFreeOuterMenu:
    candidates = tuple(value for value in CENTERS if value != center)
    blocks: list[LabelFreeActionBlock] = []
    for role, count in (("source_train", source_cases), ("target", target_cases)):
        blocks.extend(
            (
                _block(center, role, ActionKind.B, None, case_count=count),
                _block(center, role, ActionKind.U, None, case_count=count),
                *(
                    _block(center, role, ActionKind.HXE, donor, case_count=count)
                    for donor in candidates
                ),
            )
        )
    return LabelFreeOuterMenu(
        outer_target_id=center,
        blocks=tuple(sorted(blocks, key=lambda row: row.key)),
        lineage={"fixture": "v19-source-q-target-H", "center": center},
    )


@pytest.mark.parametrize("incomplete_role", ("source_train", "target"))
def test_outer_menu_rejects_incomplete_c_minus_context_donor_inventory(
    incomplete_role: str,
) -> None:
    center = CENTERS[0]
    candidates = tuple(value for value in CENTERS if value != center)
    blocks: list[LabelFreeActionBlock] = []
    for role, count in (("source_train", 4), ("target", 1)):
        role_candidates = candidates[:-1] if role == incomplete_role else candidates
        blocks.extend(
            (
                _block(center, role, ActionKind.B, None, case_count=count),
                _block(center, role, ActionKind.U, None, case_count=count),
                *(
                    _block(center, role, ActionKind.HXE, donor, case_count=count)
                    for donor in role_candidates
                ),
            )
        )

    with pytest.raises(ProtocolError, match="exactly the eight legal"):
        LabelFreeOuterMenu(
            outer_target_id=center,
            blocks=tuple(sorted(blocks, key=lambda row: row.key)),
            lineage={"fixture": "incomplete-c-minus-context"},
        )


def test_outer_menu_rejects_noncanonical_donor_at_the_same_cardinality() -> None:
    center = CENTERS[0]
    candidates = tuple(value for value in CENTERS if value != center)
    blocks: list[LabelFreeActionBlock] = []
    for role, count in (("source_train", 4), ("target", 1)):
        role_candidates = (
            (*candidates[:-1], "outside-canonical-C")
            if role == "target"
            else candidates
        )
        role_blocks = [
            _block(center, role, ActionKind.B, None, case_count=count),
            _block(center, role, ActionKind.U, None, case_count=count),
        ]
        for donor in role_candidates:
            if donor != "outside-canonical-C":
                role_blocks.append(
                    _block(center, role, ActionKind.HXE, donor, case_count=count)
                )
                continue
            samples, cases = _identities("target", center, count)
            role_blocks.append(
                LabelFreeActionBlock(
                    surface_role=role,
                    outer_target_id=center,
                    query_center_id=center,
                    action_kind=ActionKind.HXE,
                    selected_source_id=donor,
                    sample_ids=samples,
                    case_ids=cases,
                    probabilities=np.full(len(samples), 0.5, dtype=np.float32),
                    seed_dispersion=np.full(len(samples), 0.01, dtype=np.float32),
                )
            )
        blocks.extend(role_blocks)

    with pytest.raises(ProtocolError, match="exactly the eight legal"):
        LabelFreeOuterMenu(
            outer_target_id=center,
            blocks=tuple(sorted(blocks, key=lambda row: row.key)),
            lineage={"fixture": "noncanonical-c-minus-context"},
        )


def _source_labels(bundle: object) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "center": bundle.center_id,
            "case_id": case_id,
            "sample_id": sample_id,
            "label": label,
        }
        for case_id, samples in bundle.source_case_samples
        for sample_id, label in zip(samples, (1, 1, 0, 0), strict=True)
    )


def _fit_config() -> RouterFitConfig:
    return RouterFitConfig(
        outer_folds=2,
        inner_folds=2,
        stack_folds=2,
        opportunity_ridge_alphas=(1.0,),
        ranker_ridge_alphas=(1.0,),
        k_values=(1, 2),
        lambda_values=(0.5, 1.0),
        route_thresholds=(0.0,),
        required_source_case_count=8 * len(CENTERS),
        required_source_center_count=len(CENTERS),
        minimum_cases_per_center=4,
        minimum_routed_oof_cases=2,
        minimum_routed_oof_centers=1,
        minimum_routed_oof_cases_per_center=1,
        bootstrap_replicates=32,
    )


def test_workstation_model_config_is_explicitly_translated_to_science_config() -> None:
    raw = _fit_config().public_payload()
    translations = {
        "outer_folds": "nested_outer_folds",
        "inner_folds": "nested_inner_folds",
        "minimum_routed_oof_cases_per_center": (
            "minimum_routed_oof_cases_per_counted_center"
        ),
        "bootstrap_replicates": "source_oof_bootstrap_replicates",
        "bootstrap_alpha": "source_oof_bootstrap_alpha",
        "bootstrap_seed": "source_oof_bootstrap_seed",
    }
    for field, model_key in translations.items():
        raw[model_key] = raw.pop(field)
    raw["schema_version"] = (
        "midogpp_harp_stage90_safe_winner_router_v19"
    )

    translated = _as_router_config(SimpleNamespace(model=raw))

    assert translated == _fit_config()
    assert isinstance(translated.k_values, tuple)
    with pytest.raises(ProtocolError, match="unknown settings"):
        _as_router_config({"outer_folds": 2, "unbound_science_knob": 3})


def test_adapter_exposes_exact_u_full_and_only_directional_hxe_components() -> None:
    center = CENTERS[0]
    bundle = compile_support_target_menus(_physical_menu(center))

    assert bundle.candidate_source_ids == tuple(value for value in CENTERS if value != center)
    assert bundle.source_menu_hash != bundle.target_menu_hash
    assert all(
        menu.surface_role is SurfaceRole.SOURCE_TRAIN_DEVELOPMENT
        for menu in bundle.source_menus
    )
    assert all(
        menu.surface_role is SurfaceRole.TARGET_EVALUATION
        for menu in bundle.target_menus
    )
    for menu in (*bundle.source_menus, *bundle.target_menus):
        assert menu.full_action.arm_id == FULL_U_ARM_ID
        assert menu.full_action.direction is Direction.FULL
        assert menu.full_action.donor_id is None
        assert tuple(menu.full_action.feature_names) == LABEL_FREE_FEATURE_NAMES
        assert all(
            action.direction in {Direction.D01, Direction.D10}
            and action.donor_id != center
            for action in menu.actions
            if action.arm_id != FULL_U_ARM_ID
        )


def test_source_truth_is_one_memory_only_capability_per_case() -> None:
    bundle = compile_support_target_menus(_physical_menu(CENTERS[0]))
    attached = attach_support_outcome_inventory(bundle, _source_labels(bundle))

    assert len(attached.truth_capabilities) == len(bundle.source_menus)
    assert all(len(capability.case_keys) == 1 for capability in attached.truth_capabilities)
    assert all(capability.selected_score_count == 0 for capability in attached.truth_capabilities)
    payload = attached.public_payload()
    rendered = json.dumps(payload, sort_keys=True)
    assert '"label":' not in rendered
    assert payload["raw_source_labels_persisted"] is False
    with pytest.raises(ProtocolError, match="cannot be serialized"):
        attached.truth_capabilities[0].__getstate__()


def test_single_pooled_policy_fits_all_source_centers_and_routes_all_targets(tmp_path, monkeypatch) -> None:
    from threadpoolctl import threadpool_info, threadpool_limits
    from midogpp_thesis.cvae.routing import safe_winner_router_v19 as science
    from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v19.execution.source_diagnostics import write_source_diagnostics

    fit = science.fit_source_router
    observed_threads = []
    def checked_fit(*args, **kwargs):
        observed_threads.extend(row["num_threads"] for row in threadpool_info())
        return fit(*args, **kwargs)
    monkeypatch.setattr(science, "fit_source_router", checked_fit)
    bundles = tuple(
        compile_support_target_menus(_physical_menu(center)) for center in CENTERS
    )
    labels = {bundle.center_id: _source_labels(bundle) for bundle in bundles}
    source = build_support_outcome_artifact(bundles, labels)
    with threadpool_limits(limits=2):
        fitted = build_support_router_artifact(source, config=_fit_config())
    assert observed_threads and set(observed_threads) == {1}
    paths = write_source_diagnostics(tmp_path, fitted=fitted, source_surface=source, config_hash="c" * 64)
    frontier, headroom, joins = (json.loads(p.read_text()) for p in paths)
    assert joins["candidate_row_count"] > 0
    assert joins["winner_row_count"] > 0
    assert joins["raw_sample_labels_persisted"] is False
    assert frontier["row_count"] > 0
    assert all("failed_constraints" in row and "utility_risk_moments" in row for row in frontier["rows"])
    assert headroom["primitive_case_count"] == 72
    assert headroom["primitive_proper_loss_safe_positive_case_count"] == 72
    assert headroom["oracle_used_for_selection"] is False

    assert fitted.manifest["pooled_source_center_count"] == 9
    assert fitted.manifest["pooled_source_case_count"] == 72
    assert fitted.manifest["truth_capability_count"] == 72
    assert fitted.manifest["one_pooled_policy_fit"] is True
    assert fitted.manifest["raw_source_labels_persisted"] is False
    assert fitted.manifest["science_pool_topology"] == {
        "schema_version": "midogpp_harp_v19_pooled_fit_execution_v1",
        "execution_mode": "parent_process_nonserializable_truth_capability",
        "worker_count": 0,
        "blas_threads": 1,
        "cuda_used": False,
        "truth_capability_cross_process_transport": False,
        "phase_disjoint_from_gpu_and_classifier_pools": True,
    }
    assert fitted.manifest["configured_science_pool_used_for_truth_bearing_fit"] is False
    assert fitted.state.routers == (fitted.state.policy,)
    report = report_support_router_artifact(fitted)
    assert report["pooled_policy_count"] == 1
    assert report["target_evaluation_labels_consumed"] is False
    json.dumps(report, sort_keys=True)

    routes = build_support_target_routes(bundles, fitted)
    assert len(routes.cases) == len(CENTERS)
    assert routes.policy_hash == fitted.state.policy.policy_hash
    assert all(
        case.decision_payload["router_hash"] == routes.policy_hash
        for case in routes.cases
    )
    diagnostics = build_prelabel_diagnostics(routes)
    assert diagnostics["pooled_policy_count"] == 1
    assert diagnostics["case_count"] == len(CENTERS)
    assert diagnostics["utility_success_count"] is None


def test_soft_k2_route_and_exact_b_fallback_reconstruct_science_bytes() -> None:
    bundle = compile_support_target_menus(_physical_menu(CENTERS[0]))
    menu = bundle.target_menus[0]
    d01 = menu.actions_for(Direction.D01)[:2]
    d10 = menu.actions_for(Direction.D10)[:2]
    composite = build_soft_topk_composite(
        menu,
        d01_ranked_actions=d01,
        d10_ranked_actions=d10,
        k=2,
        mixing_lambda=0.5,
    )
    policy_hash = "a" * 64
    decision = RouteDecision(
        composite=composite,
        requested_arm_id=composite.arm_id,
        route_score=1.0,
        route_threshold=0.0,
        policy_hash=policy_hash,
        admitted=True,
    )

    soft = route_target_bundle(
        bundle,
        SimpleNamespace(policy_hash=policy_hash),
        decisions=(decision,),
    )[0]

    assert soft.selected_kind is ActionKind.SOFT_TOPK_PROBABILITY_BLEND
    assert soft.direction == "MIXED"
    assert len(soft.component_action_ids) == 4
    assert soft.component_weights == (0.5, 0.5, 0.5, 0.5)
    assert soft.routed_probabilities.tobytes() == np.frombuffer(
        b"".join(bytes.fromhex(cell) for cell in composite.probability_hex),
        dtype="<f4",
    ).tobytes()
    assert soft.decision_payload["selection_status"] == "ROUTE_SELECTED"
    assert soft.decision_payload["utility_status"] == "NOT_OPENED"

    baseline_composite = build_baseline_composite(menu)
    fallback = RouteDecision(
        composite=baseline_composite,
        requested_arm_id=composite.arm_id,
        route_score=0.0,
        route_threshold=0.1,
        policy_hash=policy_hash,
        admitted=False,
        fallback_reason="SOURCE_OOF_ADMISSION_FAILED",
    )
    exact_b = route_target_bundle(
        bundle,
        SimpleNamespace(policy_hash=policy_hash),
        decisions=(fallback,),
    )[0]
    assert exact_b.selected_kind is ActionKind.B
    assert exact_b.recipe_kind == "EXACT_B"
    assert exact_b.routed_probabilities.tobytes() == exact_b.baseline_probabilities.tobytes()


def test_exact_u_full_is_copied_without_directional_projection() -> None:
    bundle = compile_support_target_menus(_physical_menu(CENTERS[0]))
    menu = bundle.target_menus[0]
    composite = build_exact_u_composite(menu)
    decision = RouteDecision(
        composite=composite,
        requested_arm_id=FULL_U_ARM_ID,
        route_score=1.0,
        route_threshold=0.0,
        policy_hash="b" * 64,
        admitted=True,
    )
    routed = route_target_bundle(
        bundle,
        SimpleNamespace(policy_hash="b" * 64),
        decisions=(decision,),
    )[0]

    assert routed.selected_kind is ActionKind.U
    assert routed.direction == Direction.FULL.value
    assert routed.recipe_kind == "EXACT_U_FULL"
    assert routed.component_action_ids == (FULL_U_ARM_ID,)
    assert routed.routed_probabilities.tobytes() == routed.uniform_probabilities.tobytes()


def test_recovery_materializes_only_centers_without_complete_checkpoints(tmp_path) -> None:
    from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v19.runner_recovery import (
        recover_or_materialize_menus,
    )

    root = tmp_path / "run"
    physical_root = root / "stores/physical_menu"
    physical_root.mkdir(parents=True)
    first = _physical_menu(CENTERS[0])
    receipt = write_label_free_outer_menu(physical_root / "center_0", first)
    journal = LabelFreeProgressJournal(root / "progress.json", "c" * 64)
    journal.initialize()
    journal.record(
        outer_target_id=CENTERS[0],
        menu_hash=first.menu_hash,
        manifest_path=receipt.manifest_path,
        npz_path=receipt.npz_path,
    )

    class RecordingPipeline:
        requested: tuple[str, ...] | None = None

        def materialize_label_free_outer_menus(
            self,
            _config: object,
            _cache: object,
            *,
            outer_targets: tuple[str, ...],
            scratch_root: object,
        ) -> tuple[LabelFreeOuterMenu, ...]:
            del scratch_root
            self.requested = tuple(outer_targets)
            return tuple(_physical_menu(center) for center in outer_targets)

    pipeline = RecordingPipeline()
    menus, receipts = recover_or_materialize_menus(
        root=root,
        centers=tuple(CENTERS),
        journal=journal,
        pipeline=pipeline,  # type: ignore[arg-type]
        config=SimpleNamespace(),  # type: ignore[arg-type]
        cache=object(),
        scratch=tmp_path / "scratch",
    )

    assert pipeline.requested == tuple(CENTERS[1:])
    assert tuple(menu.outer_target_id for menu in menus) == tuple(CENTERS)
    assert menus[0].menu_hash == first.menu_hash
    assert len(receipts) == len(CENTERS)


def test_recovery_is_forbidden_after_source_train_label_fence(tmp_path) -> None:
    from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v19.runner_recovery import (
        recover_or_materialize_menus,
    )
    from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v19.source_train_label_access_fence import (
        SOURCE_TRAIN_LABEL_ACCESS_FENCE_MEMBER,
    )

    root = tmp_path / "run"
    fence = root / SOURCE_TRAIN_LABEL_ACCESS_FENCE_MEMBER
    fence.parent.mkdir(parents=True)
    fence.write_text("fenced\n", encoding="utf-8")
    journal = LabelFreeProgressJournal(root / "progress.json", "d" * 64)

    with pytest.raises(ProtocolError, match="after source-train label access"):
        recover_or_materialize_menus(
            root=root,
            centers=tuple(CENTERS),
            journal=journal,
            pipeline=SimpleNamespace(),  # type: ignore[arg-type]
            config=SimpleNamespace(),  # type: ignore[arg-type]
            cache=object(),
            scratch=tmp_path / "scratch",
        )
