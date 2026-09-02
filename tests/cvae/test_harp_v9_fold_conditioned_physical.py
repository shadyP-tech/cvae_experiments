from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from midogpp_thesis.cvae.routing.harp_protocol import canonical_hash
from midogpp_thesis.cvae.runtime.harp_v9_execution import crossfit_surface
from midogpp_thesis.cvae.runtime.harp_v9_execution.crossfit_actions import (
    SIX_SOURCE_PURE_TOPUP_EFFECTIVE_SOURCES,
    SIX_SOURCE_PURE_TOPUP_MAX_WEIGHT,
    build_fold_conditioned_action_menu,
    six_source_geometry_audit,
)
from midogpp_thesis.cvae.runtime.harp_v9_execution.crossfit_contracts import (
    FoldConditionedActionBlock,
    FoldConditionedCompatibility,
    FoldConditionedSourceSurface,
)
from midogpp_thesis.cvae.runtime.harp_v9_execution.crossfit_durability import (
    issue_source_crossfit_label_capability,
    persist_source_crossfit_surface,
    reconstruct_source_crossfit_surface,
)
from midogpp_thesis.cvae.runtime.harp_v9_execution.crossfit_effective_menus import (
    build_fold_conditioned_effective_surface,
)
from midogpp_thesis.cvae.runtime.harp_v9_execution.compatibility_contracts import (
    CandidatePoolReceipt,
    CompatibilityReceipt,
    ReplicaEnergyInput,
)
from midogpp_thesis.cvae.runtime.harp_v9_execution.contracts import (
    ActionKind,
    LabelFreeActionBlock,
    LabelFreeTargetMenu,
)
from midogpp_thesis.cvae.runtime.harp_v9_execution.directional_surfaces import (
    build_label_free_directional_actions,
)
from midogpp_thesis.cvae.runtime.harp_v9_execution.geometry_features import (
    GEOMETRY_FEATURE_NAMES,
)
from midogpp_thesis.cvae.runtime.harp_v9_execution.physical_actions import (
    build_target_action_menu,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.harp_v9_execution.execution_profile import (
    DEFAULT_WORKSTATION_PROFILE,
)


def _synthetic_surface() -> FoldConditionedSourceSurface:
    blocks = []
    compatibility = []
    outer = "0"
    for heldout in CENTERS:
        if heldout == outer:
            continue
        for query in CENTERS:
            if query == outer:
                continue
            samples = (f"{query}-sample-a", f"{query}-sample-b")
            cases = (f"{query}-case-a", f"{query}-case-b")
            actions = build_fold_conditioned_action_menu(outer, heldout, query)
            for action in actions:
                values = (
                    np.asarray((0.4, 0.6), dtype=np.float32)
                    if action.action_id == "B"
                    else np.asarray((0.6, 0.4), dtype=np.float32)
                )
                blocks.append(
                    FoldConditionedActionBlock(
                        action=action,
                        sample_ids=samples,
                        case_ids=cases,
                        probabilities=values,
                        seed_dispersion=np.zeros(2, dtype=np.float32),
                    )
                )
            candidates = tuple(
                center for center in CENTERS if center not in {outer, heldout, query}
            )
            for case_index, case_id in enumerate(cases):
                order = candidates if case_index == 0 else tuple(reversed(candidates))
                rank_by_source = {
                    source: rank for rank, source in enumerate(order, 1)
                }
                for source in candidates:
                    rank = rank_by_source[source]
                    score = float(rank)
                    compatibility.append(
                        FoldConditionedCompatibility(
                            outer_target_id=outer,
                            heldout_center_id=heldout,
                            current_query_center_id=query,
                            case_id=case_id,
                            candidate_source_id=source,
                            replica_z_scores=(score, score, score),
                            mean_z=score,
                            std_z=0.0,
                            rank=rank,
                            rank_margin=0.25,
                            source_checkpoint_hashes=tuple(
                                canonical_hash({"source": source, "seed": seed})
                                for seed in (17, 42, 101)
                            ),
                        )
                    )
    return FoldConditionedSourceSurface(
        outer_target_ids=(outer,),
        blocks=tuple(sorted(blocks, key=lambda row: row.key)),
        compatibility=tuple(sorted(compatibility, key=lambda row: row.key)),
        lineage={"fixture": "case_local_roundtrip"},
    )


def test_every_hqr_action_physically_excludes_heldout_q() -> None:
    for outer in CENTERS:
        for heldout in CENTERS:
            if heldout == outer:
                continue
            for query in CENTERS:
                if query == outer:
                    continue
                menu = build_fold_conditioned_action_menu(outer, heldout, query)
                expected_sources = tuple(
                    center
                    for center in CENTERS
                    if center not in {outer, heldout, query}
                )
                assert len(menu) == 2 + len(expected_sources)
                assert all(action.source_order == expected_sources for action in menu)
                assert all(heldout not in action.source_order for action in menu)
                assert all(outer not in action.source_order for action in menu)
                assert all(query not in action.source_order for action in menu)
                assert {
                    action.selected_source_id
                    for action in menu
                    if action.selected_source_id is not None
                } == set(expected_sources)


def test_six_source_geometry_records_relaxed_diagnostic_bounds() -> None:
    menu = build_fold_conditioned_action_menu("0", "1", "2")
    selected = next(action for action in menu if action.selected_source_id == "3")
    assert selected.geometry.base_per_source == 168
    assert selected.geometry.base_total_per_class == 1008
    assert selected.geometry.topup_total_per_class == 126
    assert selected.geometry.final_total_per_class == 1134
    assert selected.residual_action is not None
    assert selected.residual_action.maximum_source_weight == (
        SIX_SOURCE_PURE_TOPUP_MAX_WEIGHT
    )
    assert selected.residual_action.effective_source_count_by_class[0] == pytest.approx(
        SIX_SOURCE_PURE_TOPUP_EFFECTIVE_SOURCES, rel=0.0, abs=1e-14
    )
    audit = six_source_geometry_audit()
    assert audit["generic_max_weight_quarter_claimed"] is False
    assert audit["generic_min_effective_sources_six_claimed"] is False
    assert audit["pure_selected_topup_maximum_source_weight"] == 7.0 / 27.0
    assert audit["pure_selected_topup_effective_source_count"] == 243.0 / 43.0


def test_action_and_compatibility_identities_include_heldout_q() -> None:
    action_q1 = next(
        row
        for row in build_fold_conditioned_action_menu("0", "1", "3")
        if row.selected_source_id == "5"
    )
    action_q2 = next(
        row
        for row in build_fold_conditioned_action_menu("0", "2", "3")
        if row.selected_source_id == "5"
    )
    assert action_q1.action_hash != action_q2.action_hash
    assert action_q1.to_payload()["heldout_center_id"] == "1"
    assert action_q2.to_payload()["heldout_center_id"] == "2"

    common = dict(
        outer_target_id="0",
        current_query_center_id="3",
        case_id="case-a",
        candidate_source_id="5",
        replica_z_scores=(0.1, 0.2, 0.3),
        mean_z=0.2,
        std_z=float(np.std(np.asarray((0.1, 0.2, 0.3), dtype=np.float64))),
        rank=1,
        rank_margin=0.1,
        source_checkpoint_hashes=("a" * 64, "b" * 64, "c" * 64),
    )
    first = FoldConditionedCompatibility(heldout_center_id="1", **common)
    second = FoldConditionedCompatibility(heldout_center_id="2", **common)
    assert first.receipt_hash != second.receipt_hash
    assert first.key[1] == "1"
    assert second.key[1] == "2"


def test_fold_plan_preserves_all_seed_cells_and_hqr_task_grouping() -> None:
    plan = crossfit_surface.fold_conditioned_physical_plan(CENTERS)
    assert plan["prediction_context_count"] == 72
    assert plan["calibration_context_count"] == 504
    assert plan["context_count"] == 576
    assert plan["action_count"] == 4680
    assert plan["classifier_task_count"] == 5184
    assert plan["seed_cell_count"] == 42120
    assert plan["seed_cells_are_technical_replications"] is True
    assert plan["seed_selection_performed"] is False
    assert plan["classifier_task_grouping"] == (
        "one_H_q_r_seed_pair_complete_action_slate"
    )


def test_task_builder_groups_complete_actions_and_binds_q(monkeypatch, tmp_path) -> None:
    records = tuple(
        {
            "block_ordinal": ordinal,
            "source_center": source,
            "training_seed": training_seed,
            "generation_seed": generation_seed,
        }
        for ordinal, (source, training_seed, generation_seed) in enumerate(
            (source, training_seed, generation_seed)
            for source in CENTERS
            for training_seed in (17, 42, 101)
            for generation_seed in (17, 42, 101)
        )
    )
    source_binding = SimpleNamespace(
        array_path=tmp_path / "sources.npy",
        array_sha256="a" * 64,
        index_path=tmp_path / "source_index.json",
        index_sha256="b" * 64,
        index_hash="c" * 64,
        records=records,
        lock_hash="d" * 64,
        lock_sha256="e" * 64,
    )
    monkeypatch.setattr(
        crossfit_surface, "validate_source_task_binding", lambda _cache: source_binding
    )
    role = "harp_source_train_development"
    frame_path = tmp_path / "frames.npy"
    np.save(frame_path, np.arange(len(CENTERS) * 4, dtype=np.float32).reshape(-1, 2))
    frames = SimpleNamespace(
        path=frame_path,
        sha256="f" * 64,
        receipt_hash="1" * 64,
        contexts=MappingProxyType(
            {
                (role, center): (index * 2, index * 2 + 2)
                for index, center in enumerate(CENTERS)
            }
        ),
        sample_ids=MappingProxyType(
            {(role, center): (f"{center}-s0", f"{center}-s1") for center in CENTERS}
        ),
        case_ids=MappingProxyType(
            {(role, center): (f"{center}-c0", f"{center}-c1") for center in CENTERS}
        ),
    )
    classifier = SimpleNamespace(to_payload=lambda: {"family": "fixture"})
    inputs = SimpleNamespace(
        generation_hash="3" * 64,
        bank_hash="4" * 64,
        classifier=classifier,
    )
    tasks = crossfit_surface.build_fold_conditioned_classifier_tasks(
        scratch_root=tmp_path,
        frames=frames,
        source_cache=object(),
        inputs=inputs,
        workstation=DEFAULT_WORKSTATION_PROFILE,
        source_role=role,
        outer_targets=("0",),
    )
    assert len(tasks) == 8 * 8 * 9
    assert len({task["task_hash"] for task in tasks}) == len(tasks)
    for task in tasks:
        heldout = task["heldout_center_id"]
        query = task["current_query_center_id"]
        assert task["query_center_id"] == query
        assert task["heldout_q_physically_excluded"] is True
        assert all(
            heldout not in action["source_order"] for action in task["actions"]
        )
        assert set(task["allowed_source_ids"]) == {
            center for center in CENTERS if center not in {"0", heldout, query}
        }
        assert {
            record["source_center"] for record in task["source_records"]
        } == set(task["allowed_source_ids"])
        assert heldout not in {
            record["source_center"] for record in task["source_records"]
        }
        assert len(task["actions"]) == (9 if heldout == query else 8)


def test_compatibility_is_reranked_inside_each_hqr_pool() -> None:
    replicas = []
    for source_index, source in enumerate(CENTERS):
        for seed_index, seed in enumerate((17, 42, 101)):
            contexts = []
            for query_index, query in enumerate(CENTERS):
                values = (
                    [0.0, 1.0, 2.0]
                    if query == source
                    else [
                        float(source_index),
                        float(len(CENTERS) - source_index),
                        float(query_index + seed_index),
                    ]
                )
                contexts.append(
                    {
                        "role": "harp_source_train_development",
                        "query_center": query,
                        "case_order": ["a", "b", "c"],
                        "per_case_energy_float32": values,
                        "case_equal_mean_float64": float(np.mean(values)),
                    }
                )
            replicas.append(
                {
                    "source_center": source,
                    "training_seed": seed,
                    "checkpoint_sha256": f"{source_index + 1:x}" * 64,
                    "contexts": contexts,
                }
            )
    body = {
        "schema_version": "midogpp_harp_v9_role_qualified_compatibility_surface_v2",
        "support_binding": {
            "source_role": "harp_source_train_development",
            "target_role": "harp_full_test_evaluation",
        },
        "replicas": replicas,
        "labels_consumed": False,
    }
    rows = crossfit_surface.build_fold_conditioned_compatibility(
        {**body, "compatibility_hash": canonical_hash(body)},
        outer_targets=("0",),
    )
    assert len(rows) == (8 * 7 + 8 * 7 * 6) * 3
    for row in rows:
        assert row.heldout_center_id not in {
            candidate.candidate_source_id
            for candidate in rows
            if (
                candidate.outer_target_id,
                candidate.heldout_center_id,
                candidate.current_query_center_id,
            )
            == (
                row.outer_target_id,
                row.heldout_center_id,
                row.current_query_center_id,
            )
        }
        assert row.candidate_source_id not in {
            row.outer_target_id,
            row.heldout_center_id,
            row.current_query_center_id,
        }
    scoped = [
        row
        for row in rows
        if (
            row.outer_target_id,
            row.heldout_center_id,
            row.current_query_center_id,
        )
        == ("0", "1", "2")
    ]
    ranks_a = {row.candidate_source_id: row.rank for row in scoped if row.case_id == "a"}
    ranks_b = {row.candidate_source_id: row.rank for row in scoped if row.case_id == "b"}
    assert ranks_a != ranks_b


def test_effective_menus_use_case_local_hqr_features_and_exact_candidates() -> None:
    surface = _synthetic_surface()
    effective = build_fold_conditioned_effective_surface(surface)
    menus = [
        row
        for row in effective.fitting_menus("0", "1")
        if row.current_query_center_id == "2"
    ]
    assert len(menus) == 2
    expected = tuple(center for center in CENTERS if center not in {"0", "1", "2"})
    assert all(row.candidate_source_ids == expected for row in menus)
    assert all("1" not in row.candidate_source_ids for row in menus)
    assert all("2" not in row.candidate_source_ids for row in menus)
    candidate = expected[0]
    values_by_case = {}
    for row in menus:
        action = next(
            action
            for action in row.menu.actions
            if action.candidate_source_id == candidate
        )
        values_by_case[row.menu.case_id] = action.feature_values[
            action.feature_names.index("compatibility_reciprocal_rank")
        ]
    assert len(set(values_by_case.values())) == 2


def test_durable_surface_closed_world_roundtrip_and_fold_capability(tmp_path) -> None:
    surface = _synthetic_surface()
    root = tmp_path / "surface"
    receipt = persist_source_crossfit_surface(root, surface)
    reconstructed, reconstructed_receipt = reconstruct_source_crossfit_surface(root)
    assert reconstructed.surface_hash == surface.surface_hash
    assert reconstructed_receipt.receipt_hash == receipt.receipt_hash
    label_manifest = tmp_path / "source_labels.csv"
    label_manifest.write_text("center,case_id,sample_id,label\n", encoding="utf-8")
    from midogpp_thesis.cvae.runtime.artifact_io import sha256_file

    capability = issue_source_crossfit_label_capability(
        receipt,
        outer_target_id="0",
        heldout_center_id="1",
        label_manifest_path=label_manifest,
        expected_label_manifest_sha256=sha256_file(label_manifest),
    )
    assert capability.outer_target_id == "0"
    assert capability.heldout_center_id == "1"
    assert capability.authorized_source_center_ids == tuple(
        center for center in CENTERS if center not in {"0", "1"}
    )


def test_durable_surface_rejects_partial_and_tampered_members(tmp_path) -> None:
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "manifest.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="inventory"):
        reconstruct_source_crossfit_surface(partial)

    root = tmp_path / "tampered"
    persist_source_crossfit_surface(root, _synthetic_surface())
    path = root / "probabilities.npy"
    values = np.load(path, allow_pickle=False).copy()
    values[0] = np.float32(1.0 - values[0])
    np.save(path, values)
    with pytest.raises(ProtocolError, match="bytes drifted"):
        reconstruct_source_crossfit_surface(root)


def test_source_and_target_share_geometry_feature_schema_and_target_is_c_minus_h() -> None:
    surface = _synthetic_surface()
    effective = build_fold_conditioned_effective_surface(surface)
    source_action = effective.prediction_menus("0", "1")[0].menu.actions[0]

    target_blocks = []
    sample_ids = ("test-sample-a", "test-sample-b")
    case_ids = ("test-case-a", "test-case-b")
    for action in build_target_action_menu("0"):
        target_blocks.append(
            LabelFreeActionBlock(
                surface_role="target",
                outer_target_id="0",
                query_center_id="0",
                action_kind=(
                    ActionKind.B
                    if action.action_id == "B"
                    else ActionKind.U
                    if action.action_id == "U"
                    else ActionKind.HXE
                ),
                selected_source_id=action.selected_source_id,
                sample_ids=sample_ids,
                case_ids=case_ids,
                probabilities=(
                    np.asarray((0.4, 0.6), dtype=np.float32)
                    if action.action_id == "B"
                    else np.asarray((0.6, 0.4), dtype=np.float32)
                ),
                seed_dispersion=np.zeros(2, dtype=np.float32),
            )
        )
    target_only = LabelFreeTargetMenu(
        outer_target_id="0",
        blocks=tuple(sorted(target_blocks, key=lambda row: row.key)),
        lineage={"fixture": "target_c_minus_h"},
    )
    combined = crossfit_surface.bind_crossfit_prediction_folds_to_target_menus(
        surface, (target_only,)
    )[0]
    candidates = tuple(center for center in CENTERS if center != "0")
    assert {
        block.selected_source_id
        for block in target_only.blocks
        if block.action_kind is ActionKind.HXE
    } == set(candidates)
    pool = CandidatePoolReceipt(
        outer_target_id="0",
        query_center_id="0",
        all_center_ids=CENTERS,
        candidate_center_ids=candidates,
        bank_lock_hash="a" * 64,
    )
    receipts = []
    for rank, candidate in enumerate(candidates, 1):
        replicas = tuple(
            ReplicaEnergyInput(
                candidate_source_id=candidate,
                training_seed=seed,
                query_case_equal_energy=float(rank),
                own_source_location=0.0,
                own_source_scale=1.0,
                checkpoint_hash=canonical_hash((candidate, seed, "checkpoint")),
                source_frame_hash=canonical_hash((candidate, seed, "frame")),
                sampler_hash=canonical_hash((candidate, seed, "sampler")),
            )
            for seed in (17, 42, 101)
        )
        receipts.append(
            CompatibilityReceipt(
                outer_target_id="0",
                query_center_id="0",
                candidate_source_id=candidate,
                candidate_pool_hash=pool.pool_hash,
                support_partition_hash="b" * 64,
                support_hash="c" * 64,
                support_manifest_hash="d" * 64,
                replica_scores=replicas,
                mean_z=float(rank),
                std_z=0.0,
                rank=rank,
                rank_margin=0.25,
            )
        )
    local = {
        (case_id, candidate): (float(rank), 0.0, 1.0 / rank, 0.25, 1.0)
        for case_id in case_ids
        for rank, candidate in enumerate(candidates, 1)
    }
    target_actions = build_label_free_directional_actions(
        combined,
        candidate_pool=pool,
        compatibility_receipts=tuple(receipts),
        expected_role="target",
        case_local_compatibility=local,
    )
    assert target_actions
    assert source_action.feature_names == target_actions[0].feature_names
    assert tuple(
        name for name in target_actions[0].feature_names if name.startswith("geometry_")
    ) == GEOMETRY_FEATURE_NAMES
