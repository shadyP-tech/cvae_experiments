from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.cvae.diagnostics.b_paired_reparameterization_audit import (
    AUDIT_CANDIDATES,
    AUDIT_CENTERS,
    CONTROLLED_CANDIDATES,
    HASH_PROMOTED,
    INITIALIZATION_SEEDS,
    LEGACY_CANDIDATE,
    PENDING_HASH_PROMOTION,
    SNAPSHOT_ARTIFACT_ID,
    ArrayBinding,
    AuditConfig,
    CenterPreparedData,
    ClaimFirewall,
    ContentEntry,
    EpsilonTraceLedger,
    EpsilonTraceSpec,
    FrozenBRecipe,
    PreparedPartition,
    build_key_record,
    build_snapshot,
    comparison_pairs,
    load_epsilon_trace,
    snapshot_manifest_hash,
    snapshot_from_mapping,
    trace_content_hash,
    validate_key_inventory,
    validate_snapshot,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _metric_sha(metric: dict[str, int | float]) -> str:
    encoded = json.dumps(
        metric, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _binding(path: str, shape: tuple[int, ...], role: str) -> ArrayBinding:
    return ArrayBinding(
        relative_path=path,
        sha256=_sha(f"bytes:{path}"),
        content_hash=_sha(f"content:{path}"),
        dtype="float32" if "features" in path else "<U32",
        shape=shape,
        role=role,
    )


def _partition(
    prefix: str, rows: int, role_prefix: str, cases: tuple[str, ...]
) -> PreparedPartition:
    return PreparedPartition(
        features=_binding(
            f"{prefix}_features.npy", (rows, 128), f"{role_prefix}_features"
        ),
        sample_ids=_binding(
            f"{prefix}_sample_ids.npy", (rows,), f"{role_prefix}_sample_ids"
        ),
        case_ids=_binding(
            f"{prefix}_case_ids.npy", (rows,), f"{role_prefix}_case_ids"
        ),
        class_labels=_binding(
            f"{prefix}_class_labels.npy",
            (rows,),
            "source_fit_only"
            if role_prefix == "source_fit"
            else "final_diagnostic_scoring_only",
        ),
        case_id_inventory=cases,
        row_inventory_hash=_sha(f"rows:{prefix}"),
        sample_id_inventory_hash=_sha(f"samples:{prefix}"),
        case_id_inventory_hash=_sha(f"cases:{prefix}"),
        row_count=rows,
        sample_count=rows,
        case_count=len(cases),
    )


def _prepared_centers() -> tuple[CenterPreparedData, ...]:
    return tuple(
        CenterPreparedData(
            center=center,
            prepared_bundle=ContentEntry(
                relative_path=f"prepared/center-{center}.npz",
                sha256=_sha(f"prepared-bytes:{center}"),
                content_hash=_sha(f"prepared-content:{center}"),
                size_bytes=1,
                role="prepared_center_bundle",
            ),
            fit=_partition(
                f"prepared/center-{center}-fit",
                8,
                "source_fit",
                (f"{center}-fit-a", f"{center}-fit-b"),
            ),
            evaluation=_partition(
                f"prepared/center-{center}-eval",
                4,
                "diagnostic_eval",
                (f"{center}-eval-a",),
            ),
        )
        for center in AUDIT_CENTERS
    )


def _records(
    manifest_hash: str, *, published: bool
) -> tuple:
    records = []
    metric = {
        "bacc": 0.75,
        "positive_recall": 0.70,
        "specificity": 0.80,
        "fn": 3,
        "fp": 2,
        "tn": 8,
        "tp": 7,
    }
    expected = _sha("legacy-expected") if published else None
    semantic = stable_hash({"legacy": "semantic"}) if published else None
    for center in AUDIT_CENTERS:
        prepared_path = f"prepared/center-{center}.npz"
        prepared_sha = _sha(f"prepared-bytes:{center}")
        prepared_content = _sha(f"prepared-content:{center}")
        fixed_schedule_path = f"schedules/fixed-center-{center}.npy"
        fixed_schedule_sha = _sha(f"fixed-schedule-bytes:{center}")
        fixed_schedule_content = _sha(f"fixed-schedule-content:{center}")
        fixed_trace_path = f"traces/fixed-center-{center}.npy"
        fixed_trace_sha = _sha(f"fixed-trace-bytes:{center}")
        fixed_trace_content = _sha(f"fixed-trace-content:{center}")
        for seed in INITIALIZATION_SEEDS:
            records.append(
                build_key_record(
                    center=center,
                    initialization_seed=seed,
                    execution_device={17: "cuda:1", 42: "cuda:0", 101: "cuda:1"}[seed],
                    candidate=LEGACY_CANDIDATE,
                    prepared_relpath=prepared_path,
                    prepared_sha256=prepared_sha,
                    prepared_content_hash=prepared_content,
                    schedule_relpath=f"schedules/legacy-center-{center}-seed-{seed}.npy",
                    schedule_sha256=_sha(f"legacy-schedule-bytes:{center}:{seed}"),
                    schedule_content_hash=_sha(
                        f"legacy-schedule-content:{center}:{seed}"
                    ),
                    epsilon_trace_relpath=(
                        f"traces/legacy-center-{center}-seed-{seed}.npy"
                    ),
                    epsilon_trace_sha256=_sha(f"legacy-trace-bytes:{center}:{seed}"),
                    epsilon_trace_content_hash=_sha(
                        f"legacy-trace-content:{center}:{seed}"
                    ),
                    snapshot_manifest_hash=manifest_hash,
                    legacy_expected_checkpoint_hash=expected,
                    legacy_expected_prediction_hash=expected,
                    legacy_expected_metric_hash=(
                        _metric_sha(metric) if published else None
                    ),
                    legacy_expected_initialization_hash=expected,
                    legacy_historical_training_key_hash=semantic,
                    legacy_historical_schedule_hash=semantic,
                    legacy_historical_posterior_stream_hash=semantic,
                    legacy_historical_frame_hash=semantic,
                    legacy_historical_fit_row_hash=semantic,
                    legacy_historical_eval_row_hash=semantic,
                    legacy_expected_decode_metric=metric if published else None,
                )
            )
            for candidate in CONTROLLED_CANDIDATES:
                records.append(
                    build_key_record(
                        center=center,
                        initialization_seed=seed,
                        execution_device={
                            17: "cuda:1",
                            42: "cuda:0",
                            101: "cuda:1",
                        }[seed],
                        candidate=candidate,
                        prepared_relpath=prepared_path,
                        prepared_sha256=prepared_sha,
                        prepared_content_hash=prepared_content,
                        schedule_relpath=fixed_schedule_path,
                        schedule_sha256=fixed_schedule_sha,
                        schedule_content_hash=fixed_schedule_content,
                        epsilon_trace_relpath=fixed_trace_path,
                        epsilon_trace_sha256=fixed_trace_sha,
                        epsilon_trace_content_hash=fixed_trace_content,
                        snapshot_manifest_hash=manifest_hash,
                    )
                )
    return tuple(records)


def _content_index(
    prepared_centers: tuple[CenterPreparedData, ...],
    records: tuple,
) -> tuple[ContentEntry, ...]:
    bindings = [
        binding
        for prepared in prepared_centers
        for partition in (prepared.fit, prepared.evaluation)
        for binding in (
            partition.features,
            partition.sample_ids,
            partition.case_ids,
            partition.class_labels,
        )
    ]
    by_path = {
        binding.relative_path: ContentEntry(
            relative_path=binding.relative_path,
            sha256=binding.sha256,
            content_hash=binding.content_hash,
            size_bytes=1,
            role=binding.role,
        )
        for binding in bindings
    }
    for prepared in prepared_centers:
        by_path[prepared.prepared_bundle.relative_path] = prepared.prepared_bundle
    for record in records:
        for path, byte_hash, content_hash, role in (
            (
                record.prepared_relpath,
                record.prepared_sha256,
                record.prepared_content_hash,
                "prepared_center_bundle",
            ),
            (
                record.schedule_relpath,
                record.schedule_sha256,
                record.schedule_content_hash,
                "immutable_batch_schedule",
            ),
            (
                record.epsilon_trace_relpath,
                record.epsilon_trace_sha256,
                record.epsilon_trace_content_hash,
                "explicit_epsilon_trace",
            ),
        ):
            by_path[path] = ContentEntry(
                relative_path=path,
                sha256=byte_hash,
                content_hash=content_hash,
                size_bytes=1,
                role=role,
            )
    return tuple(by_path.values())


def test_config_freezes_legacy_v2_recipe_and_false_claim_firewall() -> None:
    config = AuditConfig(
        name="paired-audit",
        code_version="test",
        snapshot_artifact_id=SNAPSHOT_ARTIFACT_ID,
        snapshot_root=f"artifact://{SNAPSHOT_ARTIFACT_ID}",
    )
    assert config.centers == ("2", "5", "6", "9")
    assert config.initialization_seeds == (17, 42, 101)
    assert config.candidates == AUDIT_CANDIDATES
    assert (config.recipe.block_global_pca_dim, config.recipe.block_local_pca_dim) == (
        96,
        32,
    )
    assert not any(config.claim_firewall.to_payload().values())
    with pytest.raises(ProtocolError, match="legacy-v2 B PCA96\\+32"):
        FrozenBRecipe(optimizer_steps=999)
    with pytest.raises(ProtocolError, match="claim-firewall"):
        ClaimFirewall(may_feed_expert_bank=True)


def test_inventory_is_exactly_12_replay_keys_and_12_fixed_pairs() -> None:
    records = _records(stable_hash({"manifest": 1}), published=True)
    validated = validate_key_inventory(records, require_publication_hashes=True)
    assert len(validated) == 36
    assert sum(record.is_legacy for record in validated) == 12
    pairs = comparison_pairs(validated)
    assert len(pairs) == 12
    assert all(
        tuple(record.candidate for record in pair) == CONTROLLED_CANDIDATES
        for pair in pairs
    )
    assert all(pair[0].pair_id == pair[1].pair_id for pair in pairs)
    assert len(
        {
            record.epsilon_trace_content_hash
            for record in validated
            if record.center == "2" and not record.is_legacy
        }
    ) == 1


def test_inventory_rejects_legacy_comparison_and_fixed_trace_seed_drift() -> None:
    records = list(_records(stable_hash({"manifest": 2}), published=False))
    controlled_index = next(
        index for index, record in enumerate(records) if not record.is_legacy
    )
    record = records[controlled_index]
    drifted = replace(
        record,
        epsilon_trace_sha256=_sha("drifted-bytes"),
        epsilon_trace_content_hash=_sha("drifted-content"),
    )
    drifted = replace(
        drifted,
        pair_id=stable_hash({"wrong": "pair"}),
    )
    # Even a self-consistent-looking mutation is rejected at recomputation first.
    with pytest.raises(ProtocolError, match="key hash"):
        validate_key_inventory(
            records[:controlled_index] + [drifted] + records[controlled_index + 1 :],
            require_publication_hashes=False,
        )


def test_snapshot_is_hash_bound_and_requires_promotion_for_consumption() -> None:
    prepared_centers = _prepared_centers()
    config_hash = stable_hash({"config": 1})
    protocol_hash = stable_hash({"protocol": 1})
    manifest_hash = snapshot_manifest_hash(
        config_hash=config_hash,
        protocol_hash=protocol_hash,
        dataset_id="MIDOG++",
        feature_frame="virchow2_canonical_b_pca96_32",
        domain_axis="center",
        prepared_centers=prepared_centers,
        historical_paths=("/opaque/legacy/path",),
    )
    pending_records = _records(manifest_hash, published=False)
    pending = build_snapshot(
        publication_state=PENDING_HASH_PROMOTION,
        config_hash=config_hash,
        protocol_hash=protocol_hash,
        dataset_id="MIDOG++",
        feature_frame="virchow2_canonical_b_pca96_32",
        domain_axis="center",
        prepared_centers=prepared_centers,
        keys=pending_records,
        content_index=_content_index(prepared_centers, pending_records),
        historical_paths=("/opaque/legacy/path",),
    )
    validate_snapshot(pending, require_hash_promoted=False)
    with pytest.raises(ProtocolError, match="HASH_PROMOTED"):
        validate_snapshot(pending)

    published_records = _records(manifest_hash, published=True)
    published = build_snapshot(
        publication_state=HASH_PROMOTED,
        config_hash=config_hash,
        protocol_hash=protocol_hash,
        dataset_id="MIDOG++",
        feature_frame="virchow2_canonical_b_pca96_32",
        domain_axis="center",
        prepared_centers=prepared_centers,
        keys=published_records,
        content_index=_content_index(prepared_centers, published_records),
        historical_paths=("/opaque/legacy/path",),
    )
    validate_snapshot(published)
    validate_snapshot(snapshot_from_mapping(published.to_payload()))
    with pytest.raises(ProtocolError, match="overlap"):
        validate_snapshot(
            replace(
                published,
                prepared_centers=(
                    replace(
                        published.prepared_centers[0],
                        evaluation=replace(
                            published.prepared_centers[0].evaluation,
                            case_id_inventory=("2-fit-a",),
                        ),
                    ),
                    *published.prepared_centers[1:],
                ),
            )
        )


def test_execution_device_is_hash_bound_and_shared_by_each_coordinate() -> None:
    records = list(_records(stable_hash({"manifest": 4}), published=False))
    index = next(
        index
        for index, record in enumerate(records)
        if record.center == "2"
        and record.initialization_seed == 17
        and not record.is_legacy
    )
    original = records[index]
    drifted = replace(original, execution_device="cuda:0")
    from midogpp_thesis.cvae.diagnostics.b_paired_reparameterization_audit import (
        compute_key_hash,
        compute_pair_id,
    )

    drifted = replace(
        drifted,
        pair_id=compute_pair_id(
            center=drifted.center,
            initialization_seed=drifted.initialization_seed,
            execution_device=drifted.execution_device,
            snapshot_manifest_hash=drifted.snapshot_manifest_hash,
            prepared_content_hash=drifted.prepared_content_hash,
            schedule_content_hash=drifted.schedule_content_hash,
            epsilon_trace_content_hash=drifted.epsilon_trace_content_hash,
        ),
    )
    drifted = replace(drifted, key_hash=compute_key_hash(drifted))
    records[index] = drifted
    with pytest.raises(ProtocolError, match="execution device"):
        validate_key_inventory(records, require_publication_hashes=False)

    with pytest.raises(ProtocolError, match="Controlled keys cannot carry legacy"):
        build_key_record(
            center="2",
            initialization_seed=17,
            execution_device="cuda:1",
            candidate=CONTROLLED_CANDIDATES[0],
            prepared_relpath="prepared/center-2.npz",
            prepared_sha256=_sha("prepared-bytes"),
            prepared_content_hash=_sha("prepared-content"),
            schedule_relpath="schedules/fixed.npy",
            schedule_sha256=_sha("schedule-bytes"),
            schedule_content_hash=_sha("schedule-content"),
            epsilon_trace_relpath="traces/fixed.npy",
            epsilon_trace_sha256=_sha("trace-bytes"),
            epsilon_trace_content_hash=_sha("trace-content"),
            snapshot_manifest_hash=stable_hash({"manifest": 4}),
            legacy_expected_checkpoint_hash=_sha("forbidden"),
        )


def test_explicit_float32_trace_hashes_bytes_and_is_consumed_once(
    tmp_path: Path,
) -> None:
    values = np.arange(24, dtype="<f4").reshape(2, 3, 4)
    path = tmp_path / "traces" / "fixed.npy"
    path.parent.mkdir()
    np.save(path, values, allow_pickle=False)
    file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    content_sha = trace_content_hash(values)
    spec = EpsilonTraceSpec(
        relative_path="traces/fixed.npy",
        file_sha256=file_sha,
        content_sha256=content_sha,
        steps=2,
        batch_size=3,
        latent_dim=4,
    )
    loaded = load_epsilon_trace(tmp_path, spec)
    record = build_key_record(
        center="2",
        initialization_seed=17,
        execution_device="cuda:1",
        candidate=CONTROLLED_CANDIDATES[0],
        prepared_relpath="prepared/center-2.npz",
        prepared_sha256=_sha("prepared-bytes"),
        prepared_content_hash=_sha("prepared-content"),
        schedule_relpath="schedules/fixed.npy",
        schedule_sha256=_sha("schedule-bytes"),
        schedule_content_hash=_sha("schedule-content"),
        epsilon_trace_relpath=spec.relative_path,
        epsilon_trace_sha256=file_sha,
        epsilon_trace_content_hash=content_sha,
        snapshot_manifest_hash=stable_hash({"manifest": 3}),
    )
    ledger = EpsilonTraceLedger((record,))
    consumed = ledger.consume(record, loaded)
    np.testing.assert_array_equal(consumed, values)
    ledger.assert_complete()
    with pytest.raises(ProtocolError, match="consumed twice"):
        ledger.consume(record, loaded)

    wrong = replace(spec, content_sha256=_sha("wrong-content"))
    with pytest.raises(ProtocolError, match="content hash"):
        load_epsilon_trace(tmp_path, wrong)
