from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.harp_protocol import canonical_hash
from midogpp_thesis.cvae.runtime.artifact_io import atomic_json
from midogpp_thesis.cvae.runtime.harp_v6_execution.compatibility_adapter import (
    bind_compatibility_artifact_to_outer_menus,
    build_compatibility_artifact,
    compatibility_state_from_artifact,
)
from midogpp_thesis.cvae.runtime.harp_v6_execution.contracts import (
    ActionKind,
    ArtifactValue,
    LabelFreeActionBlock,
    LabelFreeOuterMenu,
)
from midogpp_thesis.cvae.runtime.harp_v6_execution.stores import (
    read_artifact_value,
    write_artifact_value,
)


CENTERS = ("0", "1", "2", "3", "5", "6", "7", "8", "9")
SEEDS = (17, 42, 101)
DEV = "harp_consumed_test_development"
EVAL = "harp_consumed_test_evaluation"


@dataclass(frozen=True)
class _Row:
    center: str
    case_id: str
    split_role: str


def _block(
    *, outer: str, query: str, role: str, kind: ActionKind, source: str | None
) -> LabelFreeActionBlock:
    prefix = f"{role}-{query}"
    return LabelFreeActionBlock(
        surface_role=role,
        outer_target_id=outer,
        query_center_id=query,
        action_kind=kind,
        selected_source_id=source,
        sample_ids=(f"{prefix}-s0", f"{prefix}-s1"),
        case_ids=(f"dev-{query}-a", f"dev-{query}-b"),
        probabilities=np.asarray((0.25, 0.75), dtype=np.float32),
        seed_dispersion=np.asarray((0.01, 0.02), dtype=np.float32),
    )


def _menus() -> tuple[LabelFreeOuterMenu, ...]:
    output = []
    for outer in CENTERS:
        query = next(value for value in CENTERS if value != outer)
        source = next(value for value in CENTERS if value not in {outer, query})
        target_source = next(value for value in CENTERS if value != outer)
        blocks = (
            _block(outer=outer, query=query, role="development", kind=ActionKind.B, source=None),
            _block(outer=outer, query=query, role="development", kind=ActionKind.U, source=None),
            _block(
                outer=outer,
                query=query,
                role="development",
                kind=ActionKind.HXE,
                source=source,
            ),
            _block(outer=outer, query=outer, role="target", kind=ActionKind.B, source=None),
            _block(outer=outer, query=outer, role="target", kind=ActionKind.U, source=None),
            _block(
                outer=outer,
                query=outer,
                role="target",
                kind=ActionKind.HXE,
                source=target_source,
            ),
        )
        output.append(
            LabelFreeOuterMenu(
                outer_target_id=outer,
                blocks=tuple(sorted(blocks, key=lambda row: row.key)),
                lineage={"bank_hash": "9972a41dcd4814cd"},
            )
        )
    return tuple(output)


def _write_resident_surface(root: Path) -> None:
    support_contexts = [
        {
            "center": center,
            "frame_start": 2 * index,
            "frame_stop": 2 * index + 2,
            "case_ids": [f"dev-{center}-a", f"dev-{center}-b"],
            "sample_ids_hash": canonical_hash([f"sample-{center}-a", f"sample-{center}-b"]),
        }
        for index, center in enumerate(CENTERS)
    ]
    support_body = {
        "schema_version": "midogpp_harp_v6_label_free_support_binding_v1",
        "frame_array_path": str(root / "unused.npy"),
        "frame_array_sha256": "1" * 64,
        "frame_receipt_hash": "frame-receipt",
        "cache_hash": "cache-hash",
        "support_manifest_sha256": "2" * 64,
        "support_role": DEV,
        "contexts": support_contexts,
        "support_evaluation_case_disjoint": True,
        "labels_present": False,
        "evaluation_rows_included": False,
    }
    support = {**support_body, "support_binding_hash": canonical_hash(support_body)}
    replicas = []
    for source_index, source in enumerate(CENTERS):
        for seed_index, seed in enumerate(SEEDS):
            contexts = []
            for query_index, query in enumerate(CENTERS):
                first = float(source_index + 0.1 * seed_index + 0.01 * query_index)
                energies = [float(np.float32(first)), float(np.float32(first + 0.2))]
                contexts.append(
                    {
                        "query_center": query,
                        "case_order": [f"dev-{query}-a", f"dev-{query}-b"],
                        "per_case_energy_float32": energies,
                        "case_equal_mean_float64": float(
                            np.mean(np.asarray(energies, dtype=np.float64), dtype=np.float64)
                        ),
                        "row_count": 2,
                        "case_count": 2,
                        "energy_semantics": "variational_proxy",
                        "exact_nelbo": False,
                        "labels_consumed": False,
                    }
                )
            replicas.append(
                {
                    "source_center": source,
                    "training_seed": seed,
                    "expert_lock_hash": f"expert-{source}-{seed}",
                    "checkpoint_sha256": f"checkpoint-{source}-{seed}",
                    "source_frame_hash": f"frame-{source}",
                    "sampler_state_hash": f"sampler-{seed}",
                    "contexts": contexts,
                    "compatibility_checkpoint_hash": f"compat-{source}-{seed}",
                }
            )
    body = {
        "schema_version": "midogpp_harp_v6_support_compatibility_surface_v1",
        "support_binding": support,
        "support_binding_hash": support["support_binding_hash"],
        "training_seeds": list(SEEDS),
        "energy_semantics": "variational_proxy",
        "replicas": replicas,
        "all_replicas_used_without_selection": True,
        "computed_while_expert_resident": True,
        "exact_nelbo": False,
        "labels_consumed": False,
        "evaluation_rows_consumed": False,
    }
    path = root / "source_streams/manifests/support_compatibility.json"
    path.parent.mkdir(parents=True)
    atomic_json(path, {**body, "compatibility_hash": canonical_hash(body)})


def _build_artifact(
    tmp_path: Path,
) -> tuple[tuple[LabelFreeOuterMenu, ...], ArtifactValue]:
    _write_resident_surface(tmp_path)
    rows = tuple(
        _Row(center, case, role)
        for center in CENTERS
        for role, cases in (
            (DEV, (f"dev-{center}-a", f"dev-{center}-b")),
            (EVAL, (f"eval-{center}-a",)),
        )
        for case in cases
    )
    cache = SimpleNamespace(rows=rows)
    config = SimpleNamespace(
        protocol={"centers": list(CENTERS)},
        expected_hashes={
            "development_manifest_sha256": "2" * 64,
            "evaluation_manifest_sha256": "3" * 64,
        },
    )
    menus = _menus()
    return menus, build_compatibility_artifact(
        menus,
        cache,
        config=config,
        scratch_root=tmp_path,
        development_role=DEV,
        evaluation_role=EVAL,
    )


def test_adapter_is_strict_and_recovers_from_compact_store(tmp_path: Path) -> None:
    menus, artifact = _build_artifact(tmp_path)
    state = compatibility_state_from_artifact(artifact)
    assert len(state.candidate_pools) == len(CENTERS) ** 2
    assert len(state.receipts) == len(CENTERS) * (8 + 8 * 7)
    assert state.pool("0", "0").candidate_center_ids == CENTERS[1:]
    assert "0" not in state.pool("0", "1").candidate_center_ids
    assert "1" not in state.pool("0", "1").candidate_center_ids
    receipt = state.receipt("0", "1", "2")
    assert tuple(row.training_seed for row in receipt.replica_scores) == SEEDS
    assert all(len(row.checkpoint_hash) == 64 for row in receipt.replica_scores)
    assert all(row.own_source_scale > 0.0 for row in receipt.replica_scores)

    store = tmp_path / "durable"
    write_artifact_value(store, artifact, role="compatibility-test")
    recovered_value = read_artifact_value(store, role="compatibility-test")
    assert recovered_value.state is None
    with pytest.raises(
        ProtocolError,
        match="recovery requires exact outer-menu binding",
    ):
        compatibility_state_from_artifact(recovered_value)
    hydrated = bind_compatibility_artifact_to_outer_menus(recovered_value, menus)
    recovered = compatibility_state_from_artifact(hydrated)
    assert dict(recovered.outer_menu_hashes) == {
        menu.outer_target_id: menu.menu_hash for menu in menus
    }
    assert recovered.raw_compatibility_hash == state.raw_compatibility_hash
    assert tuple(row.receipt_hash for row in recovered.receipts) == tuple(
        row.receipt_hash for row in state.receipts
    )


def test_recovery_rejects_rehashed_outer_menu_binding_tamper(tmp_path: Path) -> None:
    menus, artifact = _build_artifact(tmp_path)
    manifest = dict(artifact.manifest)
    menu_hashes = dict(manifest["outer_menu_hashes"])
    menu_hashes["0"] = "f" * 64
    tampered_body = {
        **{key: value for key, value in manifest.items() if key != "compatibility_hash"},
        "outer_menu_hashes": menu_hashes,
    }
    tampered = ArtifactValue(
        state=None,
        manifest={
            **tampered_body,
            "compatibility_hash": canonical_hash(tampered_body),
        },
        arrays=artifact.arrays,
    )

    with pytest.raises(
        ProtocolError,
        match="escaped the exact reconstructed outer menus",
    ):
        bind_compatibility_artifact_to_outer_menus(tampered, menus)


def test_recovery_rejects_different_physical_outer_menu(tmp_path: Path) -> None:
    menus, artifact = _build_artifact(tmp_path)
    recovered = ArtifactValue(
        state=None,
        manifest=artifact.manifest,
        arrays=artifact.arrays,
    )
    changed_first = LabelFreeOuterMenu(
        outer_target_id=menus[0].outer_target_id,
        blocks=menus[0].blocks,
        lineage={**dict(menus[0].lineage), "recovery_tamper": True},
    )
    changed_menus = (changed_first, *menus[1:])

    with pytest.raises(
        ProtocolError,
        match="escaped the exact reconstructed outer menus",
    ):
        bind_compatibility_artifact_to_outer_menus(recovered, changed_menus)
