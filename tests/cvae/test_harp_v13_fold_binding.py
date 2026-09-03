from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v13 import (
    fold_menu_binding,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v13 import (
    source_crossfit_orchestration as orchestration,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v13.execution import (
    admission as recovery_admission,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v13.fold_outcome_universes import (
    build_exact_fold_outcome_universes,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v13.runner import (
    run_harp_stage90_v13,
)
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from midogpp_thesis.cvae.routing.harp_protocol import (
    HarpSourceLabelRow,
    canonical_hash,
)
from midogpp_thesis.cvae.routing.policy_calibrated_residual_router_v13 import (
    Direction,
    LabelFreeAction,
    build_effective_menu,
    float32_probability_hex,
)
from midogpp_thesis.cvae.runtime.artifact_io import sha256_file
from midogpp_thesis.cvae.runtime.harp_v13_execution.crossfit_durability import (
    SourceCrossfitSurfaceReceipt,
)
from midogpp_thesis.cvae.runtime.harp_v13_execution.crossfit_effective_menus import (
    FoldConditionedEffectiveMenu,
    FoldConditionedEffectiveSurface,
)


def _certificate_fixture(tmp_path: Path):
    source_hash = "a" * 64
    wrappers: list[FoldConditionedEffectiveMenu] = []
    center_index = {center: index for index, center in enumerate(CENTERS)}
    baseline = float32_probability_hex((0.2, 0.8))
    for h in CENTERS:
        for q in CENTERS:
            if q == h:
                continue
            for r in CENTERS:
                if r == h:
                    continue
                candidates = tuple(
                    center for center in CENTERS if center not in {h, q, r}
                )
                first = 0.55 + 0.01 * center_index[q] + 0.001 * center_index[r]
                action = LabelFreeAction(
                    outer_target_id=h,
                    query_center_id=r,
                    case_id=f"case-{r}",
                    action_id="U:D01",
                    action_kind="U",
                    direction=Direction.D01,
                    candidate_source_id=None,
                    feature_names=("context_kind_code", "candidate_count"),
                    feature_values=(float(q == r), float(len(candidates))),
                    baseline_probability_hex=baseline,
                    action_probability_hex=float32_probability_hex(
                        (first, 1.0 - first)
                    ),
                )
                menu = build_effective_menu((action,))
                block_hashes = tuple(
                    canonical_hash(
                        {
                            "kind": "block",
                            "h": h,
                            "q": q,
                            "r": r,
                            "ordinal": index,
                        }
                    )
                    for index in range(2 + len(candidates))
                )
                compatibility_hashes = tuple(
                    canonical_hash(
                        {
                            "kind": "compatibility",
                            "h": h,
                            "q": q,
                            "r": r,
                            "candidate": candidate,
                        }
                    )
                    for candidate in candidates
                )
                wrappers.append(
                    FoldConditionedEffectiveMenu(
                        outer_target_id=h,
                        heldout_center_id=q,
                        current_query_center_id=r,
                        menu=menu,
                        candidate_source_ids=candidates,
                        physical_block_hashes=block_hashes,
                        compatibility_receipt_hashes=compatibility_hashes,
                    )
                )
    surface = FoldConditionedEffectiveSurface(source_hash, tuple(wrappers))

    receipt_root = tmp_path / "source-receipt"
    receipt_root.mkdir()
    paths = tuple(receipt_root / name for name in ("manifest", "p", "d", "c"))
    for index, path in enumerate(paths):
        path.write_bytes(f"receipt-{index}".encode("ascii"))
    receipt = SourceCrossfitSurfaceReceipt(
        root=receipt_root,
        manifest_path=paths[0],
        probabilities_path=paths[1],
        dispersion_path=paths[2],
        compatibility_path=paths[3],
        surface_hash=source_hash,
        inventory_hash="b" * 64,
        manifest_hash="c" * 64,
        manifest_sha256=sha256_file(paths[0]),
        probabilities_sha256=sha256_file(paths[1]),
        dispersion_sha256=sha256_file(paths[2]),
        compatibility_sha256=sha256_file(paths[3]),
        outer_target_ids=CENTERS,
        outer_heldout_pairs=tuple(
            (h, q) for h in CENTERS for q in CENTERS if h != q
        ),
        action_block_count=1,
        compatibility_receipt_count=1,
    )
    label_index = tmp_path / "source-label-index.json"
    label_index.write_text('{"label_capability_only":true}\n', encoding="utf-8")
    certificate = fold_menu_binding.build_fold_menu_binding_certificate(
        effective_surface=surface,
        surface_receipt=receipt,
        label_index_path=label_index,
        label_index_sha256=sha256_file(label_index),
        admission_hash="d" * 64,
        authorization_lease_hash="e" * 64,
    )
    durable = fold_menu_binding.persist_fold_menu_binding_certificate(
        tmp_path / fold_menu_binding.CERTIFICATE_RELATIVE_PATH,
        certificate,
    )
    return surface, receipt, label_index, durable


def test_all_72_fold_bindings_are_exact_unique_and_durable(tmp_path: Path) -> None:
    surface, _, _, durable = _certificate_fixture(tmp_path)
    certificate = durable.certificate

    assert len(certificate.folds) == len(CENTERS) * (len(CENTERS) - 1) == 72
    assert sum(len(row.wrappers) for row in certificate.folds) == 576
    assert durable.path.name == "source_fold_menu_binding_certificate.json"
    assert durable.path.read_text(encoding="utf-8")

    for h in CENTERS:
        for q in CENTERS:
            if h == q:
                continue
            binding = certificate.for_fold(h, q)
            expected = tuple(
                sorted(
                    (
                        *surface.fitting_menus(h, q),
                        *surface.prediction_menus(h, q),
                    ),
                    key=lambda row: (
                        row.outer_target_id,
                        row.heldout_center_id,
                        row.current_query_center_id,
                        row.menu.case_id,
                    ),
                )
            )
            assert binding.wrappers == expected
            assert {row.heldout_center_id for row in binding.wrappers} == {q}
            assert len({
                (row.outer_target_id, row.heldout_center_id,
                 row.current_query_center_id, row.menu.case_id)
                for row in binding.wrappers
            }) == len(binding.wrappers)

    recovered = fold_menu_binding.validate_persisted_fold_menu_binding_payload(
        durable.path,
        expected_admission_hash="d" * 64,
        expected_authorization_lease_hash="e" * 64,
    )
    assert recovered["certificate_hash"] == certificate.certificate_hash
    with pytest.raises(Exception, match="certificate drifted"):
        fold_menu_binding.validate_persisted_fold_menu_binding_payload(
            durable.path,
            expected_admission_hash="f" * 64,
            expected_authorization_lease_hash="e" * 64,
        )


def test_orchestrator_persists_only_each_exact_H_q_union(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    surface, receipt, label_index, durable = _certificate_fixture(tmp_path)
    bundle = SimpleNamespace(
        surface_receipt=receipt,
        physical_surface=SimpleNamespace(surface_hash=receipt.surface_hash),
        effective_surface=surface,
    )
    config = SimpleNamespace(
        resolved_path=lambda _role: label_index,
        expected_hashes={"development_manifest_sha256": sha256_file(label_index)},
        model={
            "pairwise_alpha_grid": [1.0],
            "residual_alpha_grid": [1.0],
            "acceptor_alpha_grid": [1.0],
        },
    )
    persisted: dict[tuple[str, str], tuple[object, ...]] = {}
    sentinel = object()

    def issue(**kwargs):
        binding = durable.for_fold(
            kwargs["outer_target_id"], kwargs["heldout_center_id"]
        )
        return SimpleNamespace(
            outer_target_id=binding.outer_target_id,
            heldout_center_id=binding.heldout_center_id,
            prediction_surface_hash=binding.prediction_surface_hash,
            fitting_surface_hash=binding.fitting_surface_hash,
            capability_hash=binding.capability_hash,
        )

    def make_task(_bundle, capability, binding, certificate_hash, *_args, **_kwargs):
        assert binding is durable.for_fold(
            capability.outer_target_id, capability.heldout_center_id
        )
        assert certificate_hash == durable.certificate.certificate_hash
        return SimpleNamespace(
            outer_target_id=capability.outer_target_id,
            heldout_center_id=capability.heldout_center_id,
        )

    def execute(tasks, _workers):
        return tuple(
            SimpleNamespace(
                outer_target_id=task.outer_target_id,
                heldout_center_id=task.heldout_center_id,
                nested_fold=object(),
                isolation_receipt_hash="1" * 64,
            )
            for task in tasks
        )

    def persist(_root, **kwargs):
        key = (kwargs["outer_target_id"], kwargs["heldout_center_id"])
        persisted[key] = tuple(kwargs["effective_menus"])
        binding = durable.for_fold(*key)
        assert kwargs["fold_menu_binding_hash"] == binding.binding_hash
        assert kwargs["fold_menu_binding_certificate_hash"] == (
            durable.certificate.certificate_hash
        )
        assert kwargs["fold_menu_binding_certificate_receipt_hash"] == durable.receipt_hash
        return SimpleNamespace(outer_target_id=key[0], heldout_center_id=key[1])

    def persist_set(_path, **kwargs):
        assert kwargs["fold_menu_binding_certificate_hash"] == (
            durable.certificate.certificate_hash
        )
        assert kwargs["fold_menu_binding_certificate_receipt_hash"] == durable.receipt_hash
        assert len(kwargs["fold_seals"]) == 72
        return sentinel

    monkeypatch.setattr(orchestration, "issue_fold_source_label_capability", issue)
    monkeypatch.setattr(orchestration, "_fold_task", make_task)
    monkeypatch.setattr(orchestration, "persist_source_crossfit_fold", persist)
    monkeypatch.setattr(orchestration, "persist_source_crossfit_fold_set", persist_set)

    result = orchestration.fit_and_seal_prelabel_source_folds(
        bundle=bundle,
        config=config,
        cache=object(),
        source_label_loader=lambda *_args, **_kwargs: (),
        fold_store_root=tmp_path / "stores/source_crossfit_folds",
        fold_set_path=tmp_path / "manifests/source_prelabel_fold_set_internal.json",
        binding_certificate=durable,
        workers=4,
        executor=execute,
    )

    assert result is sentinel
    assert len(persisted) == 72
    for key, menus in persisted.items():
        assert menus == durable.for_fold(*key).effective_menus
        assert tuple(row.menu_hash for row in menus) == tuple(
            row.menu.menu_hash for row in durable.for_fold(*key).wrappers
        )


def test_exact_fold_outcomes_bind_to_certified_H_q_r_menus_only(
    tmp_path: Path,
) -> None:
    surface, receipt, _, durable = _certificate_fixture(tmp_path)

    class Physical:
        surface_hash = receipt.surface_hash

        @staticmethod
        def blocks_for(h: str, q: str, r: str):
            del h, q
            return (
                SimpleNamespace(
                    action=SimpleNamespace(action_id="B"),
                    sample_ids=(f"sample-{r}-0", f"sample-{r}-1"),
                    case_ids=(f"case-{r}", f"case-{r}"),
                    probabilities=np.asarray((0.2, 0.8), dtype=np.float32),
                    seed_dispersion=np.zeros(2, dtype=np.float32),
                ),
            )

    bundle = SimpleNamespace(
        surface_receipt=receipt,
        physical_surface=Physical(),
        effective_surface=surface,
    )
    labels = tuple(
        HarpSourceLabelRow(
            center=center,
            case_id=f"case-{center}",
            sample_id=f"sample-{center}-{index}",
            label=index,
        )
        for center in CENTERS
        for index in (0, 1)
    )
    universes = build_exact_fold_outcome_universes(
        bundle=bundle,
        binding_certificate=durable,
        labels=labels,
    )

    assert len(universes.folds) == 72
    for row in universes.folds:
        binding = durable.for_fold(row.outer_target_id, row.heldout_center_id)
        assert row.fold_menu_binding_hash == binding.binding_hash
        assert row.universe.effective_menus == tuple(
            sorted(binding.effective_menus, key=lambda menu: (
                menu.outer_target_id, menu.query_center_id, menu.case_id
            ))
        )
        assert len(row.universe.outcomes) == len(binding.effective_menus)


def test_runner_certifies_durable_bindings_before_source_label_phase() -> None:
    source = inspect.getsource(run_harp_stage90_v13)
    surface_sealed = source.index('ledger.advance("SOURCE_CROSSFIT_SURFACE_SEALED")')
    build = source.index("build_fold_menu_binding_certificate(")
    persist = source.index("persist_fold_menu_binding_certificate(")
    barrier = source.index("durable_barrier((durable_fold_menu_certificate.path,))")
    certified = source.index('ledger.advance("SOURCE_FOLD_MENU_BINDINGS_CERTIFIED")')
    labels = source.index('ledger.advance("SOURCE_FOLD_LABEL_CAPABILITIES_OPENED")')

    assert surface_sealed < build < persist < barrier < certified < labels
    admission_source = inspect.getsource(
        recovery_admission.validate_pristine_or_label_free_recovery
    )
    assert "CERTIFICATE_RELATIVE_PATH.as_posix()" in admission_source
    assert "validate_persisted_fold_menu_binding_payload" in admission_source
