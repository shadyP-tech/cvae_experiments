from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    TRAINING_SEEDS,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.harp_protocol import canonical_hash
from midogpp_thesis.cvae.runtime.artifact_io import atomic_json
from midogpp_thesis.cvae.runtime.harp_v15_execution.contracts import (
    ActionKind,
    ArtifactValue,
    LabelFreeActionBlock,
    LabelFreeOuterMenu,
)
from midogpp_thesis.cvae.runtime.harp_v15_execution.gpu_surface import (
    COMPATIBILITY_MEMBER,
)
from midogpp_thesis.cvae.runtime.harp_v15_execution.menu_root_binding import (
    CenterMenuRootBinding,
)
from midogpp_thesis.cvae.runtime.harp_v15_execution.stores import (
    read_artifact_value,
    write_artifact_value,
    write_label_free_outer_menu,
)
from midogpp_thesis.cvae.runtime.harp_v15_execution.support_compatibility import (
    build_case_local_compatibility_surface,
    validate_case_local_compatibility_artifact,
)
from midogpp_thesis.cvae.runtime.harp_v15_execution.support_validation import (
    _verified_model_manifest,
    _verified_target_manifest,
)


def _block(
    outer: str,
    role: str,
    kind: ActionKind,
    source: str | None,
) -> LabelFreeActionBlock:
    cases = tuple(f"{role}-H{outer}-case-{index}" for index in range(2))
    samples = tuple(f"{case}-sample" for case in cases)
    if kind is ActionKind.B:
        values = (0.25, 0.75)
    elif kind is ActionKind.U:
        values = (0.55, 0.45)
    else:
        ordinal = CENTERS.index(str(source))
        values = (0.56 + ordinal * 0.01, 0.44 - ordinal * 0.01)
    return LabelFreeActionBlock(
        surface_role=role,
        outer_target_id=outer,
        query_center_id=outer,
        action_kind=kind,
        selected_source_id=source,
        sample_ids=samples,
        case_ids=cases,
        probabilities=np.asarray(values, dtype=np.float32),
        seed_dispersion=np.full(2, 0.01, dtype=np.float32),
    )


def _menu(outer: str) -> LabelFreeOuterMenu:
    candidates = tuple(center for center in CENTERS if center != outer)
    blocks = tuple(
        _block(outer, role, kind, source)
        for role in ("support", "target")
        for kind, source in (
            (ActionKind.B, None),
            (ActionKind.U, None),
            *((ActionKind.HXE, candidate) for candidate in candidates),
        )
    )
    return LabelFreeOuterMenu(
        outer_target_id=outer,
        blocks=tuple(sorted(blocks, key=lambda row: row.key)),
        lineage={"fixture": "support-validation"},
    )


def _durable_menus(tmp_path: Path) -> tuple[tuple[LabelFreeOuterMenu, ...], CenterMenuRootBinding]:
    parent = (tmp_path / "physical_menus").resolve()
    menus = tuple(_menu(center) for center in CENTERS)
    roots = {center: parent / f"outer_{center}" for center in CENTERS}
    receipts = tuple(
        write_label_free_outer_menu(roots[center], menu)
        for center, menu in zip(CENTERS, menus, strict=True)
    )
    binding = CenterMenuRootBinding.create(
        common_parent=parent,
        centers=CENTERS,
        menu_roots=roots,
        menus=menus,
        receipts=receipts,
    )
    return menus, binding


def _compatibility_payload(menus: tuple[LabelFreeOuterMenu, ...]) -> dict[str, object]:
    binding_body = {
        "schema_version": "midogpp_harp_v15_role_qualified_label_free_binding_v2",
        "source_role": "target_train_support",
        "target_role": "target_test_evaluation",
        "labels_present": False,
        "evaluation_labels_included": False,
    }
    binding = {
        **binding_body,
        "support_binding_hash": canonical_hash(binding_body),
    }
    menu_by_outer = {menu.outer_target_id: menu for menu in menus}
    replicas = []
    for source in CENTERS:
        for seed in TRAINING_SEEDS:
            contexts = []
            for raw_role, physical_role in (
                ("target_train_support", "support"),
                ("target_test_evaluation", "target"),
            ):
                for query in CENTERS:
                    baseline = next(
                        block
                        for block in menu_by_outer[query].blocks
                        if block.surface_role == physical_role
                        and block.action_kind is ActionKind.B
                    )
                    cases = tuple(dict.fromkeys(baseline.case_ids))
                    source_offset = float(CENTERS.index(source) * 10)
                    query_offset = float(abs(CENTERS.index(query) - CENTERS.index(source)))
                    seed_offset = float(TRAINING_SEEDS.index(seed)) * 0.05
                    role_offset = 0.25 if physical_role == "target" else 0.0
                    energy = [
                        source_offset
                        + query_offset
                        + seed_offset
                        + role_offset
                        + float(index)
                        for index, _case in enumerate(cases)
                    ]
                    contexts.append(
                        {
                            "role": raw_role,
                            "query_center": query,
                            "case_order": list(cases),
                            "per_case_energy_float32": energy,
                            "case_count": len(cases),
                            "exact_nelbo": False,
                            "labels_consumed": False,
                        }
                    )
            replicas.append(
                {
                    "source_center": source,
                    "training_seed": seed,
                    "contexts": contexts,
                }
            )
    body = {
        "schema_version": "midogpp_harp_v15_role_qualified_compatibility_surface_v2",
        "support_binding": binding,
        "support_binding_hash": binding["support_binding_hash"],
        "training_seeds": list(TRAINING_SEEDS),
        "replicas": replicas,
        "all_replicas_used_without_selection": True,
        "computed_while_expert_resident": True,
        "exact_nelbo": False,
        "labels_consumed": False,
        "evaluation_labels_consumed": False,
    }
    return {**body, "compatibility_hash": canonical_hash(body)}


def _support_model_manifest() -> dict[str, object]:
    routers = []
    model_rows = []
    policy_rows = []
    state_rows = []
    for center in CENTERS:
        endpoint_hash = canonical_hash({"endpoint": center})
        admission_hash = canonical_hash({"admission": center})
        router_hash = canonical_hash({"router": center})
        routers.append(
            {
                "outer_target_id": center,
                "endpoint_model": {
                    "outer_target_id": center,
                    "model_hash": endpoint_hash,
                    "evaluation_labels_consumed": False,
                },
                "admission": {
                    "outer_target_id": center,
                    "admission_hash": admission_hash,
                },
                "router_hash": router_hash,
                "evaluation_labels_consumed": False,
            }
        )
        model_rows.append((center, endpoint_hash))
        policy_rows.append((center, router_hash, admission_hash))
        state_rows.append((center, router_hash))
    support_hash = canonical_hash({"support": "surface"})
    model_hash = canonical_hash(
        {
            "schema_version": "midogpp_harp_v15_target_local_model_set_v1",
            "models": tuple(model_rows),
            "evaluation_labels_consumed": False,
        }
    )
    policy_hash = canonical_hash(
        {
            "schema_version": "midogpp_harp_v15_target_local_policy_set_v1",
            "routers": tuple(policy_rows),
            "evaluation_labels_consumed": False,
        }
    )
    state_hash = canonical_hash(
        {
            "schema_version": "midogpp_harp_v15_support_router_fit_state_v1",
            "support_surface_hash": support_hash,
            "router_hashes": tuple(state_rows),
            "support_labels_consumed": True,
            "evaluation_labels_consumed": False,
        }
    )
    body = {
        "schema_version": "midogpp_harp_v15_support_router_fit_state_v1",
        "support_surface_hash": support_hash,
        "routers": routers,
        "state_hash": state_hash,
        "support_labels_consumed": True,
        "evaluation_labels_consumed": False,
        "model_hash": model_hash,
        "policy_hash": policy_hash,
        "config_hash": "a" * 64,
        "expected_center_ids": list(CENTERS),
        "target_train_support_only": True,
        "target_evaluation_features_used_for_fit": False,
        "target_evaluation_labels_used": False,
    }
    return {**body, "artifact_hash": canonical_hash(body)}


def _target_manifest(model: dict[str, object]) -> dict[str, object]:
    physical = {center: canonical_hash({"physical": center}) for center in CENTERS}
    effective = {center: canonical_hash({"effective": center}) for center in CENTERS}
    rows = [
        [center, f"target-H{center}-case", canonical_hash({"case": center})]
        for center in CENTERS
    ]
    body = {
        "schema_version": "midogpp_harp_v15_target_action_set_v1",
        "config_hash": "a" * 64,
        "expected_center_ids": list(CENTERS),
        "model_hash": model["model_hash"],
        "policy_hash": model["policy_hash"],
        "physical_outer_menu_hashes": physical,
        "target_effective_menu_hashes": effective,
        "case_menu_rows": rows,
        "target_case_count": len(rows),
        "exact_top1_physical_action_only": True,
        "evaluation_labels_consumed": False,
    }
    semantic = {**body, "target_action_hash": canonical_hash(body)}
    return {**semantic, "artifact_hash": canonical_hash(semantic)}


def test_support_menu_binding_accepts_only_support_h_and_target_h(tmp_path: Path) -> None:
    menus, binding = _durable_menus(tmp_path)

    restored = binding.validate_durable()
    assert tuple(row.outer_target_id for row in restored) == tuple(CENTERS)
    assert tuple(row.menu_hash for row in restored) == tuple(
        row.menu_hash for row in menus
    )
    assert CenterMenuRootBinding.from_payload(binding.to_payload()).binding_hash == binding.binding_hash


def test_case_local_compatibility_round_trips_semantic_and_artifact_hashes(
    tmp_path: Path,
) -> None:
    menus, _binding = _durable_menus(tmp_path)
    member = tmp_path / "scratch" / "source_streams" / COMPATIBILITY_MEMBER
    atomic_json(member, _compatibility_payload(menus))

    surface = build_case_local_compatibility_surface(
        menus, scratch_root=tmp_path / "scratch"
    )
    artifact = surface.artifact()
    assert validate_case_local_compatibility_artifact(artifact) == surface.surface_hash
    assert all(len(surface.for_outer(center)) == 32 for center in CENTERS)

    store_root = tmp_path / "compatibility_store"
    write_artifact_value(store_root, artifact, role="label_free_support_compatibility")
    restored = read_artifact_value(
        store_root, role="label_free_support_compatibility"
    )
    assert validate_case_local_compatibility_artifact(restored) == surface.surface_hash

    semantic_drift = dict(restored.manifest)
    semantic_drift.pop("artifact_hash")
    semantic_drift["compatibility_feature_hash"] = "0" * 64
    drifted = ArtifactValue(
        state=None,
        manifest={**semantic_drift, "artifact_hash": canonical_hash(semantic_drift)},
        arrays=restored.arrays,
    )
    with pytest.raises(ProtocolError, match="feature identity"):
        validate_case_local_compatibility_artifact(drifted)


def test_model_and_target_semantic_hashes_are_not_artifact_hash_aliases() -> None:
    model = _support_model_manifest()
    model_hash, policy_hash = _verified_model_manifest(model, centers=CENTERS)
    assert model_hash == model["model_hash"]
    assert policy_hash == model["policy_hash"]

    target = _target_manifest(model)
    target_hash, rows = _verified_target_manifest(target, centers=CENTERS)
    assert target_hash == target["target_action_hash"]
    assert len(rows) == len(CENTERS)

    stale_artifact = {**model, "config_hash": "b" * 64}
    with pytest.raises(ProtocolError, match="artifact identity"):
        _verified_model_manifest(stale_artifact, centers=CENTERS)

    stale_semantic_body = dict(model)
    stale_semantic_body.pop("artifact_hash")
    stale_semantic_body["routers"][0]["endpoint_model"]["model_hash"] = "f" * 64
    stale_semantic = {
        **stale_semantic_body,
        "artifact_hash": canonical_hash(stale_semantic_body),
    }
    with pytest.raises(ProtocolError, match="semantic identity"):
        _verified_model_manifest(stale_semantic, centers=CENTERS)
