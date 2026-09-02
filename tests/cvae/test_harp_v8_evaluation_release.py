from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v8.config import (
    HarpStage90V8Config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v8.input_surfaces import (
    DEVELOPMENT_ROLE,
    EVALUATION_ROLE,
    HarpCacheRow,
    HarpConsumedCacheIndex,
    V8_CACHE_IDENTITY,
    evaluation_row_id,
    load_evaluation_truth,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v8.preparation_contracts import (
    CanonicalFrameRow,
    CanonicalLabelBlindFrame,
    V8_PREPARATION_IDENTITY,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v8.preparation_role_manifests import (
    publish_role_pure_manifests,
)
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.harp_protocol import canonical_hash
from midogpp_thesis.cvae.runtime.artifact_io import atomic_json, sha256_file
from midogpp_thesis.cvae.runtime.harp_v8_execution.contracts import FrozenRouteReceipt


def _prepared_release(tmp_path: Path):
    repository = tmp_path / "repository"
    manifest = (
        repository
        / "datasets/midogpp/contract/annotation_patch_v1/manifest.csv"
    )
    manifest.parent.mkdir(parents=True)
    raw_rows: list[tuple[str, str, str, int]] = []
    for center in CENTERS:
        for index in range(4):
            raw_rows.append((f"case-{center}-{index}", center, "test", index % 2))
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("case_id", "center", "split", "label"))
        writer.writerows(raw_rows)
    manifest_sha = sha256_file(manifest)

    by_center: dict[str, tuple[CanonicalFrameRow, ...]] = {}
    rows_by_role: dict[str, list[HarpCacheRow]] = {
        DEVELOPMENT_ROLE: [],
        EVALUATION_ROLE: [],
    }
    ordinal = 0
    for center in CENTERS:
        source_rows: list[CanonicalFrameRow] = []
        role_offsets = {DEVELOPMENT_ROLE: 0, EVALUATION_ROLE: 0}
        for center_index in range(4):
            role = DEVELOPMENT_ROLE if center_index < 2 else EVALUATION_ROLE
            case_id = f"case-{center}-{center_index}"
            sample_id = evaluation_row_id(manifest_sha, ordinal)
            source_rows.append(
                CanonicalFrameRow(
                    center=center,
                    case_id=case_id,
                    sample_id=sample_id,
                    contract_row_index=ordinal,
                    center_row_index=center_index,
                )
            )
            rows_by_role[role].append(
                HarpCacheRow(
                    center=center,
                    case_id=case_id,
                    sample_id=sample_id,
                    split_role=role,
                    split_row_index=role_offsets[role],
                    embedding_file=f"embeddings/by_center/center_{center}.npy",
                    embedding_row_index=center_index,
                )
            )
            role_offsets[role] += 1
            ordinal += 1
        by_center[center] = tuple(source_rows)
    rows = tuple(rows_by_role[DEVELOPMENT_ROLE] + rows_by_role[EVALUATION_ROLE])
    cache_root = repository / "prepared-cache"
    barrier_path = cache_root / V8_PREPARATION_IDENTITY.label_free_barrier
    barrier_path.parent.mkdir(parents=True)
    partition_hash = "c" * 64
    barrier_base = {
        "schema_version": V8_PREPARATION_IDENTITY.label_free_barrier_schema,
        "partition_hash": partition_hash,
        "canonical_scoring_manifest_opened": False,
    }
    atomic_json(
        barrier_path,
        {**barrier_base, "barrier_hash": canonical_hash(barrier_base)},
    )
    pre_members = {"manifests/cache_index.json": "a" * 64}
    pre_hash = canonical_hash(
        {
            "schema_version": V8_CACHE_IDENTITY.content_schema,
            "members": pre_members,
        }
    )
    cache_hash = "b" * 64
    pre_cache = HarpConsumedCacheIndex(
        root=cache_root,
        rows=rows,
        shards={},
        member_sha256=pre_members,
        content_sha256=pre_hash,
        cache_hash=cache_hash,
    )
    frame = CanonicalLabelBlindFrame(
        rows_by_center=by_center,
        embeddings_by_center={
            center: np.empty((4, 1), dtype=np.float32) for center in CENTERS
        },
        cache_content_hash="d" * 64,
        row_order_hash="e" * 64,
        source_member_sha256={},
    )
    development_path = (
        repository
        / "datasets/midogpp/contract/harp_consumed_test_development_v8/manifest.csv"
    )
    release_path = (
        repository
        / "datasets/midogpp/contract/harp_consumed_test_evaluation_v8/release.json"
    )
    development_sha, release_sha = publish_role_pure_manifests(
        manifest,
        expected_manifest_sha256=manifest_sha,
        cache=pre_cache,
        frame=frame,
        development_path=development_path,
        evaluation_path=release_path,
        identity=V8_PREPARATION_IDENTITY,
    )
    final_members = {
        **pre_members,
        "manifests/harp_v8_consumed_test_preparation_receipt.json": "f" * 64,
    }
    final_cache = HarpConsumedCacheIndex(
        root=cache_root,
        rows=rows,
        shards={},
        member_sha256=final_members,
        content_sha256="1" * 64,
        cache_hash=cache_hash,
    )
    config_hash = "2" * 64
    config = HarpStage90V8Config(
        source_path=repository / "synthetic-v8.yaml",
        artifact_root="synthetic",
        input_locations={"evaluation_manifest_path": release_path.as_posix()},
        expected_hashes={"evaluation_manifest_sha256": release_sha},
        execution_authorized=True,
        protocol={"centers": list(CENTERS)},
        model={},
        runtime={},
        claim_boundary={},
        config_hash=config_hash,
    )
    receipt = FrozenRouteReceipt(
        seal_hash="3" * 64,
        config_hash=config_hash,
        route_hash="4" * 64,
        policy_hash="5" * 64,
        model_hash="6" * 64,
        target_action_hash="7" * 64,
        validation_bundle_hash="8" * 64,
        independent_validation_hashes=("9" * 64, "a" * 64),
        expected_center_ids=tuple(CENTERS),
        case_count=len(
            {
                (row.center, row.case_id)
                for row in rows
                if row.split_role == EVALUATION_ROLE
            }
        ),
    )
    return {
        "manifest": manifest,
        "release": release_path,
        "development_sha": development_sha,
        "cache": final_cache,
        "config": config,
        "receipt": receipt,
        "rows": rows,
    }


def test_preparation_publishes_no_readable_evaluation_truth_and_requires_receipt(
    tmp_path: Path,
) -> None:
    prepared = _prepared_release(tmp_path)
    release_text = prepared["release"].read_text(encoding="utf-8")

    assert '"label"' not in release_text
    assert not (prepared["release"].parent / "manifest.csv").exists()
    assert prepared["development_sha"] == sha256_file(
        prepared["release"].parents[1]
        / "harp_consumed_test_development_v8/manifest.csv"
    )

    canonical = prepared["manifest"]
    original = canonical.read_bytes()
    canonical.unlink()
    with pytest.raises(ProtocolError, match="typed frozen-route receipt"):
        load_evaluation_truth(
            prepared["config"],
            prepared["cache"],
            object(),  # type: ignore[arg-type]
        )
    canonical.write_bytes(original)

    truth = load_evaluation_truth(
        prepared["config"],
        prepared["cache"],
        prepared["receipt"],
    )
    evaluation_rows = tuple(
        row for row in prepared["rows"] if row.split_role == EVALUATION_ROLE
    )
    assert tuple(truth) == tuple(row.key for row in evaluation_rows)
    assert set(truth.values()) == {0, 1}

