from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v9.source_crossfit_orchestration as orchestration
import midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v9.source_crossfit_fold_store as fold_store
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v9.input_surfaces import (
    _authenticate_source_label_scope,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v9.source_crossfit_orchestration import (
    FoldFitTask,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v9.source_label_capability import (
    issue_fold_source_label_capability,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v9.workspace_paths import (
    SOURCE_CROSSFIT_EFFECTIVE_MENU_SEAL_MEMBER,
    SOURCE_CROSSFIT_EFFECTIVE_MENU_STORE,
    SOURCE_CROSSFIT_PHYSICAL_SURFACE_SEAL_MEMBER,
    SOURCE_CROSSFIT_PHYSICAL_SURFACE_STORE,
    SOURCE_FOLD_LABEL_CAPABILITY_SEALS_MEMBER,
    SOURCE_PRELABEL_Q_PREDICTION_SEAL_MEMBER,
    SOURCE_PRELABEL_Q_PREDICTIONS_STORE,
)
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.harp_protocol import HarpSourceLabelRow, canonical_hash
from midogpp_thesis.cvae.routing.policy_calibrated_residual_router_v9 import (
    PairwiseFitConfig,
)
from midogpp_thesis.cvae.runtime.artifact_io import read_json, sha256_file
from midogpp_thesis.cvae.runtime.harp_v9_execution.contracts import (
    ActionKind,
    LabelFreeActionBlock,
)
from midogpp_thesis.cvae.runtime.harp_v9_execution.crossfit_actions import (
    build_fold_conditioned_action_menu,
)
from midogpp_thesis.cvae.runtime.harp_v9_execution.crossfit_contracts import (
    FoldConditionedActionBlock,
    FoldConditionedCompatibility,
    FoldConditionedSourceSurface,
)
from midogpp_thesis.cvae.runtime.harp_v9_execution.crossfit_durability import (
    issue_source_crossfit_label_capability,
    persist_source_crossfit_surface,
)
from midogpp_thesis.cvae.runtime.harp_v9_execution.crossfit_effective_menus import (
    build_fold_conditioned_effective_surface,
)
from midogpp_thesis.cvae.runtime.harp_v9_execution.phases import PHASE_ORDER


class _WorkerScopeObserved(RuntimeError):
    pass


def _one_outer_surface(outer: str = "0") -> FoldConditionedSourceSurface:
    blocks: list[FoldConditionedActionBlock] = []
    compatibility: list[FoldConditionedCompatibility] = []
    sample_ids = ("sample-a0", "sample-a1", "sample-b0", "sample-b1")
    case_ids = ("case-a", "case-a", "case-b", "case-b")
    for heldout in CENTERS:
        if heldout == outer:
            continue
        for query in CENTERS:
            if query == outer:
                continue
            actions = build_fold_conditioned_action_menu(outer, heldout, query)
            for ordinal, action in enumerate(actions):
                offset = np.float32(min(0.35, 0.02 * ordinal))
                probabilities = np.asarray(
                    (0.2 + offset, 0.8 - offset, 0.25 + offset, 0.75 - offset),
                    dtype=np.float32,
                )
                blocks.append(
                    FoldConditionedActionBlock(
                        action=action,
                        sample_ids=sample_ids,
                        case_ids=case_ids,
                        probabilities=probabilities,
                        seed_dispersion=np.zeros(4, dtype=np.float32),
                    )
                )
            candidates = tuple(
                center
                for center in CENTERS
                if center not in {outer, heldout, query}
            )
            for case_index, case_id in enumerate(("case-a", "case-b")):
                order = candidates if case_index == 0 else tuple(reversed(candidates))
                rank_by_source = {
                    source: rank for rank, source in enumerate(order, start=1)
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
                                canonical_hash(
                                    {"source": source, "training_seed": seed}
                                )
                                for seed in (17, 42, 101)
                            ),
                        )
                    )
    return FoldConditionedSourceSurface(
        outer_target_ids=(outer,),
        blocks=tuple(sorted(blocks, key=lambda row: row.key)),
        compatibility=tuple(sorted(compatibility, key=lambda row: row.key)),
        lineage={"fixture": "worker_typed_loader_scope"},
    )


def _fold_task(
    tmp_path: Path,
    *,
    loader: object,
) -> tuple[FoldFitTask, object]:
    outer = "0"
    heldout = "1"
    surface = _one_outer_surface(outer)
    receipt = persist_source_crossfit_surface(tmp_path / "surface", surface)
    effective = build_fold_conditioned_effective_surface(surface)
    label_index = tmp_path / "source-label-index.json"
    label_index.write_text("{}\n", encoding="utf-8")
    capability = issue_fold_source_label_capability(
        surface_receipt=receipt,
        effective_surface=effective,
        outer_target_id=outer,
        heldout_center_id=heldout,
        label_index_path=label_index,
        label_index_sha256=sha256_file(label_index),
    )
    allowed = tuple(
        center for center in CENTERS if center not in {outer, heldout}
    )
    baselines: list[tuple[str, LabelFreeActionBlock]] = []
    for query in allowed:
        raw = next(
            row
            for row in surface.blocks_for(outer, heldout, query)
            if row.action.action_id == "B"
        )
        baselines.append(
            (
                query,
                LabelFreeActionBlock(
                    surface_role="development",
                    outer_target_id=outer,
                    query_center_id=query,
                    action_kind=ActionKind.B,
                    selected_source_id=None,
                    sample_ids=raw.sample_ids,
                    case_ids=raw.case_ids,
                    probabilities=raw.probabilities,
                    seed_dispersion=raw.seed_dispersion,
                ),
            )
        )
    task = FoldFitTask(
        outer_target_id=outer,
        heldout_center_id=heldout,
        config=SimpleNamespace(),
        cache=SimpleNamespace(),
        source_label_loader=loader,  # type: ignore[arg-type]
        label_capability=capability,
        baseline_blocks=tuple(baselines),
        fitting_menus=tuple(
            row.menu for row in effective.fitting_menus(outer, heldout)
        ),
        prediction_menus=tuple(
            row.menu for row in effective.prediction_menus(outer, heldout)
        ),
        fit_config=PairwiseFitConfig(
            pairwise_alpha=0.1,
            residual_alpha=0.1,
            acceptor_alpha=0.1,
        ),
        label_capability_hash=capability.capability_hash,
        source_surface_receipt_hash=receipt.receipt_hash,
        source_surface_hash=surface.surface_hash,
        effective_adapter_hash=effective.adapter_hash,
        prediction_surface_hash=capability.prediction_surface_hash,
        fitting_surface_hash=capability.fitting_surface_hash,
    )
    return task, receipt


def test_own_hq_worker_typed_loader_requests_exact_c_minus_h_minus_q(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}

    def loader(_config, _cache, *, allowed_centers, source_label_capability):
        rows = tuple(
            HarpSourceLabelRow(
                center=center,
                case_id=f"{center}-case",
                sample_id=f"{center}-sample",
                label=0,
            )
            for center in allowed_centers
        )
        observed["allowed"] = tuple(allowed_centers)
        observed["capability"] = source_label_capability
        observed["rows"] = rows
        return rows

    task, _receipt = _fold_task(tmp_path, loader=loader)

    def observe_join(received_task, rows):
        assert received_task is task
        assert {row.center for row in rows} == set(task.label_capability.allowed_center_ids)
        assert task.heldout_center_id not in {row.center for row in rows}
        assert task.outer_target_id not in {row.center for row in rows}
        raise _WorkerScopeObserved

    monkeypatch.setattr(orchestration, "_join_scoped_worker_outcomes", observe_join)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(_WorkerScopeObserved):
        orchestration._fit_fold_worker(task)

    assert observed["allowed"] == tuple(
        center
        for center in CENTERS
        if center not in {task.outer_target_id, task.heldout_center_id}
    )
    assert observed["capability"] is task.label_capability
    assert task.heldout_center_id not in {
        row.center for row in observed["rows"]  # type: ignore[union-attr]
    }


def test_parent_fold_orchestrator_does_not_call_loader_before_executor(
    tmp_path: Path,
) -> None:
    calls: list[object] = []

    def forbidden_parent_loader(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("parent opened a source-label shard")

    task, receipt = _fold_task(tmp_path, loader=forbidden_parent_loader)
    # This check is structural and deliberately stops at the child-dispatch
    # seam.  The runner phase-order assertion below proves aggregate loading is
    # later than the complete fold-set seal.
    source = inspect.getsource(orchestration.fit_and_seal_prelabel_source_folds)
    assert "source_label_loader(" not in source
    assert "run_executor(tuple(tasks), workers)" in source
    assert calls == []
    assert task.source_surface_receipt_hash == receipt.receipt_hash


def test_generic_and_runtime_only_capabilities_cannot_open_diagnostic_labels(
    tmp_path: Path,
) -> None:
    task, receipt = _fold_task(tmp_path, loader=lambda *_args, **_kwargs: ())
    with pytest.raises(ProtocolError, match="typed crossfit capability"):
        _authenticate_source_label_scope(CENTERS, capability={"authorized": True})

    runtime_only = issue_source_crossfit_label_capability(
        receipt,
        outer_target_id=task.outer_target_id,
        heldout_center_id=task.heldout_center_id,
        label_manifest_path=task.label_capability.label_index_path,
        expected_label_manifest_sha256=task.label_capability.label_index_sha256,
    )
    runtime_scope = tuple(
        center
        for center in CENTERS
        if center not in {task.outer_target_id, task.heldout_center_id}
    )
    assert task.heldout_center_id not in runtime_only.authorized_source_center_ids
    with pytest.raises(ProtocolError, match="typed crossfit capability"):
        _authenticate_source_label_scope(runtime_scope, capability=runtime_only)


def test_runner_orders_fold_seals_before_aggregate_labels_and_matches_catalog_paths() -> None:
    from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v9 import runner

    assert PHASE_ORDER.index("SOURCE_FOLD_LABEL_CAPABILITIES_OPENED") < PHASE_ORDER.index(
        "SOURCE_PRELABEL_Q_FOLDS_SEALED"
    ) < PHASE_ORDER.index("FULL_SOURCE_LABELS_OPENED")
    source = inspect.getsource(runner.run_harp_stage90_v9)
    assert "fit_source_lodo(" not in source
    for relative in (
        SOURCE_CROSSFIT_PHYSICAL_SURFACE_STORE,
        SOURCE_CROSSFIT_PHYSICAL_SURFACE_SEAL_MEMBER,
        SOURCE_CROSSFIT_EFFECTIVE_MENU_STORE,
        SOURCE_CROSSFIT_EFFECTIVE_MENU_SEAL_MEMBER,
        SOURCE_FOLD_LABEL_CAPABILITY_SEALS_MEMBER,
        SOURCE_PRELABEL_Q_PREDICTIONS_STORE,
        SOURCE_PRELABEL_Q_PREDICTION_SEAL_MEMBER,
    ):
        assert relative in source


def test_materializer_reconstructs_durable_bytes_before_effective_adapter() -> None:
    source = inspect.getsource(orchestration.materialize_label_free_source_crossfit)
    reconstruct_at = source.index("reconstruct_source_crossfit_surface")
    effective_at = source.index("effective = build_effective(reconstructed)")
    assert reconstruct_at < effective_at
    assert "return LabelFreeSourceCrossfitBundle(reconstructed, receipt, effective)" in source


def test_fold_set_paths_are_relative_to_bundle_not_manifest_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "bundle"
    store = bundle / "stores/source_crossfit_folds"
    manifest = bundle / "manifests/source_prelabel_fold_set_internal.json"
    centers = ("a", "b", "c")
    seals = []
    for outer in centers:
        for heldout in centers:
            if heldout == outer:
                continue
            member = store / f"outer_{outer}" / f"heldout_{heldout}.json"
            member.parent.mkdir(parents=True, exist_ok=True)
            member.write_text("{}\n", encoding="utf-8")
            seals.append(
                SimpleNamespace(
                    outer_target_id=outer,
                    heldout_center_id=heldout,
                    path=member.resolve(),
                    manifest_sha256=sha256_file(member),
                    seal_hash=canonical_hash((outer, heldout, "seal")),
                    nested_fold=SimpleNamespace(
                        fold_hash=canonical_hash((outer, heldout, "fold"))
                    ),
                )
            )
    sentinel = object()
    monkeypatch.setattr(fold_store, "load_source_crossfit_fold_set", lambda _path: sentinel)
    result = fold_store.persist_source_crossfit_fold_set(
        manifest,
        expected_center_ids=centers,
        source_surface_receipt_hash="1" * 64,
        source_surface_hash="2" * 64,
        effective_adapter_hash="3" * 64,
        fold_seals=seals,  # type: ignore[arg-type]
    )
    assert result is sentinel
    payload = read_json(manifest)
    assert all(
        str(row["relative_path"]).startswith("stores/source_crossfit_folds/")
        and ".." not in Path(str(row["relative_path"])).parts
        for row in payload["folds"]
    )
