from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v18.identity import (
    EXPERIMENT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v18.runner_payloads import (
    build_surface_seal_indexes,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v18.source_label_capability import (
    issue_source_train_label_capabilities,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v18.source_train_label_access_fence import (
    begin_source_train_label_access,
)
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.harp_protocol import canonical_hash
from midogpp_thesis.cvae.runtime.artifact_io import atomic_json, sha256_file


def _sealed(body: dict[str, object], field: str) -> dict[str, object]:
    return {**body, field: canonical_hash(body)}


def _role_seal(
    *, center: str, role: str, candidates: tuple[str, ...]
) -> dict[str, object]:
    body = {
        "schema_version": (
            "midogpp_harp_v18_source_train_menu_seal_v1"
            if role == "source_train"
            else "midogpp_harp_v18_target_evaluation_menu_seal_v1"
        ),
        "experiment_id": EXPERIMENT_ID,
        "center_id": center,
        "surface_role": role,
        "candidate_source_ids": list(candidates),
        "action_identity_hash": canonical_hash({"center": center, "kind": "actions"}),
        "menu_hash": canonical_hash({"center": center, "role": role}),
        "store_receipt_hash": canonical_hash({"center": center, "kind": "store"}),
        "labels_consumed": False,
    }
    return _sealed(body, "seal_hash")


def _prelabel_inventory(root: Path) -> tuple[tuple[object, ...], Path, str]:
    seal_root = root / "manifests/source_target_role_seals"
    bank_body = {
        "schema_version": "midogpp_harp_v18_fixed_bank_independence_v1",
        "bank_index_sha256": "1" * 64,
        "generation_lock_sha256": "2" * 64,
        "source_local_lineage_hash": "3" * 64,
        "per_center_hashes": {
            center: canonical_hash({"center": center, "kind": "bank-proof"})
            for center in CENTERS
        },
        "candidate_pool_semantics": "C_MINUS_CONTEXT_CENTER",
        "own_center_expert_unrepresentable": True,
        "source_frames_and_samplers_source_center_local": True,
        "classifier_scaler_fit": "synthetic_train_only",
        "source_train_labels_may_update": "POOLED_ROUTER_ONLY",
        "source_train_labels_may_not_update": [
            "expert_checkpoint",
            "source_frame",
            "aggregate_prior",
            "generation",
            "classifier",
            "menu_geometry",
            "shared_transform",
            "hyperparameter_grid",
        ],
        "labels_consumed": False,
    }
    bank_path = seal_root / "fixed_bank_independence_attestation.json"
    atomic_json(bank_path, _sealed(bank_body, "attestation_hash"))
    rows = []
    for center in CENTERS:
        candidates = tuple(value for value in CENTERS if value != center)
        center_root = seal_root / f"center_{center}"
        source_path = center_root / "source_train_menu_seal.json"
        target_path = center_root / "target_evaluation_menu_seal.json"
        source = _role_seal(center=center, role="source_train", candidates=candidates)
        target = _role_seal(center=center, role="target", candidates=candidates)
        atomic_json(source_path, source)
        atomic_json(target_path, target)
        rows.append(
            SimpleNamespace(
                center_id=center,
                physical_store_receipt_hash=source["store_receipt_hash"],
                source_train_menu_seal_path=source_path.resolve(),
                source_train_menu_seal_sha256=sha256_file(source_path),
                source_train_menu_seal_hash=source["seal_hash"],
                target_evaluation_menu_seal_path=target_path.resolve(),
                target_evaluation_menu_seal_sha256=sha256_file(target_path),
                target_evaluation_menu_seal_hash=target["seal_hash"],
                bank_independence_attestation_path=bank_path.resolve(),
                bank_independence_attestation_sha256=sha256_file(bank_path),
            )
        )
    label_path = root / "source_labels/index.json"
    atomic_json(label_path, {"labels": "sealed-elsewhere"})
    return tuple(rows), label_path, sha256_file(label_path)


def _fence(root: Path, rows: tuple[object, ...], label_sha: str):
    source, target, bank = build_surface_seal_indexes(rows)
    source_path = root / "manifests/source_train_menu_seals.json"
    target_path = root / "manifests/target_evaluation_menu_seals.json"
    bank_path = root / "manifests/bank_independence_attestations.json"
    atomic_json(source_path, source)
    atomic_json(target_path, target)
    atomic_json(bank_path, bank)
    return begin_source_train_label_access(
        root,
        config_hash="4" * 64,
        admission_hash="5" * 64,
        authorization_lease_hash="6" * 64,
        ordered_center_ids=CENTERS,
        source_train_surface_seal_index=source,
        source_train_surface_seal_index_path=source_path,
        target_surface_seal_index=target,
        target_surface_seal_index_path=target_path,
        bank_independence_index=bank,
        bank_independence_index_path=bank_path,
        label_index_sha256=label_sha,
    )


def test_v18_issues_exactly_one_capability_per_q_after_global_reauthentication(
    tmp_path: Path,
) -> None:
    rows, label_path, label_sha = _prelabel_inventory(tmp_path)
    fence = _fence(tmp_path, rows, label_sha)

    capabilities = issue_source_train_label_capabilities(
        seal_sets=rows,
        label_index_path=label_path,
        label_index_sha256=label_sha,
        source_train_label_access_fence=fence,
    )

    assert tuple(row.center_id for row in capabilities.capabilities) == CENTERS
    assert len({row.capability_hash for row in capabilities.capabilities}) == len(CENTERS)
    for row in capabilities.capabilities:
        assert row.candidate_source_ids == tuple(
            center for center in CENTERS if center != row.center_id
        )


def test_v18_refuses_any_drifted_member_before_issuing_first_q_capability(
    tmp_path: Path,
) -> None:
    rows, label_path, label_sha = _prelabel_inventory(tmp_path)
    fence = _fence(tmp_path, rows, label_sha)
    Path(rows[-1].target_evaluation_menu_seal_path).write_text("{}\n", encoding="utf-8")

    with pytest.raises(ProtocolError, match="menu seal is absent or drifted"):
        issue_source_train_label_capabilities(
            seal_sets=rows,
            label_index_path=label_path,
            label_index_sha256=label_sha,
            source_train_label_access_fence=fence,
        )

