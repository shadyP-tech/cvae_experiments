"""Publish source-train truth capabilities and a sealed all-test descriptor.

Preparation may publish center-sharded source-train truth after the durable
label-free barrier. Evaluation truth is deliberately different: only a
label-free descriptor is written.  The canonical scoring manifest is reopened
by the terminal reader after it receives the typed frozen-route capability.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import io
from pathlib import Path

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from ...runtime.artifact_io import atomic_json, read_json, sha256_file
from .identity import EXPERIMENT_ID, PUBLICATION_STATUS, TERMINAL_DECISION
from .input_surfaces import (
    CANONICAL_SCORING_MANIFEST_RELATIVE_PATH,
    SOURCE_TRAIN_ROLE,
    EVALUATION_RELEASE_SCHEMA,
    TARGET_EVALUATION_ROLE,
    SOURCE_LABEL_INDEX_SCHEMA,
    evaluation_row_id,
)
from .preparation_contracts import (
    CANONICAL_SOURCE_TRAIN_TENSOR_SHA256,
    EXPECTED_SOURCE_TRAIN_CASE_COUNT,
    EXPECTED_SOURCE_TRAIN_ROW_COUNT,
    EXPECTED_TARGET_TEST_CASE_COUNT,
    EXPECTED_TARGET_TEST_ROW_COUNT,
    CanonicalLabelBlindFrame,
    HarpPreparationIdentity,
)
from .preparation_durable_io import atomic_text
from .preparation_source_train_cache import source_train_row_id


def publish_role_pure_manifests(
    canonical_manifest: Path,
    *,
    expected_manifest_sha256: str,
    cache,
    canonical_train_cache_root: Path,
    source_train_frame: CanonicalLabelBlindFrame,
    target_test_frame: CanonicalLabelBlindFrame,
    development_path: Path,
    evaluation_path: Path,
    identity: HarpPreparationIdentity,
) -> tuple[str, str]:
    """Publish source-train truth capabilities plus a label-free evaluation capability."""

    # Do not hash or parse the outcome-bearing test manifest here.  Its
    # predeclared digest is already bound by the authenticated target cache and
    # it is opened only by the terminal reader after a frozen-route receipt.
    if not canonical_manifest.is_file() or canonical_manifest.is_symlink():
        raise ProtocolError("HARP canonical scoring manifest is absent or unsafe.")
    barrier = read_json(cache.root / identity.label_free_barrier)
    if barrier.get("canonical_scoring_manifest_opened") is not False:
        raise ProtocolError("HARP scoring manifest opened before the label-free barrier.")

    development_labels = _open_source_train_labels_after_barrier(
        canonical_train_cache_root,
        source_train_frame=source_train_frame,
    )
    development_cache_rows = tuple(
        row for row in cache.rows if row.split_role == SOURCE_TRAIN_ROLE
    )
    evaluation_cache_rows = tuple(
        row for row in cache.rows if row.split_role == TARGET_EVALUATION_ROLE
    )
    source_frame_keys = tuple(
        (row.center, row.case_id, row.sample_id)
        for center in CENTERS
        for row in source_train_frame.rows_by_center[center]
    )
    target_frame_keys = tuple(
        (row.center, row.case_id, row.sample_id)
        for center in CENTERS
        for row in target_test_frame.rows_by_center[center]
    )
    if (
        set(development_labels)
        != {
            row.sample_id for row in development_cache_rows
        }
        or tuple(row.key for row in development_cache_rows) != source_frame_keys
        or tuple(row.key for row in evaluation_cache_rows) != target_frame_keys
        or len(development_cache_rows) != EXPECTED_SOURCE_TRAIN_ROW_COUNT
        or len(evaluation_cache_rows) != EXPECTED_TARGET_TEST_ROW_COUNT
        or len({(row.center, row.case_id) for row in development_cache_rows})
        != EXPECTED_SOURCE_TRAIN_CASE_COUNT
        or len({(row.center, row.case_id) for row in evaluation_cache_rows})
        != EXPECTED_TARGET_TEST_CASE_COUNT
    ):
        raise ProtocolError("HARP development truth coverage drifted.")
    # Reconstruct the permitted source capability in prepared-cache order only
    # after the post-barrier source-label identity audit has completed.
    development_rows = [
        (
            row.center,
            row.case_id,
            row.sample_id,
            development_labels[row.sample_id],
        )
        for row in development_cache_rows
    ]
    evaluation_keys = [row.key for row in evaluation_cache_rows]
    if not development_rows or not evaluation_keys:
        raise ProtocolError("HARP role publication is empty.")

    shard_rows: list[dict[str, object]] = []
    for center in CENTERS:
        scoped = tuple(row for row in development_rows if row[0] == center)
        if not scoped:
            raise ProtocolError("HARP source-train center capability is empty.")
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(("center", "case_id", "sample_id", "label", "split_role"))
        writer.writerows((*values, SOURCE_TRAIN_ROLE) for values in scoped)
        relative = f"by_center/center_{center}.csv"
        shard_path = development_path.parent / relative
        atomic_text(shard_path, buffer.getvalue())
        ordered_keys = [list(row[:3]) for row in scoped]
        shard_rows.append(
            {
                "center": center,
                "relative_path": relative,
                "sha256": sha256_file(shard_path),
                "row_count": len(scoped),
                "case_count": len({row[1] for row in scoped}),
                "ordered_key_hash": canonical_hash(
                    {"ordered_keys": ordered_keys}
                ),
            }
        )
    source_index_base: dict[str, object] = {
        "schema_version": SOURCE_LABEL_INDEX_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "artifact_role": "center_sharded_source_train_label_capability",
        "split_role": SOURCE_TRAIN_ROLE,
        "cache_index_hash": cache.cache_hash,
        "pre_manifest_cache_content_sha256": cache.content_sha256,
        "source_train_tensor_sha256": CANONICAL_SOURCE_TRAIN_TENSOR_SHA256,
        "shards": shard_rows,
        "row_count": len(development_rows),
        "case_count": len({(row[0], row[1]) for row in development_rows}),
        "labels_stored_in_index": False,
        "capability_state": (
            "SOURCE_TRAIN_CENTER_SCOPED_OPEN_AFTER_ALL_SOURCE_AND_TARGET_MENU_SEALS_AND_BANK_ATTESTATIONS"
        ),
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "fresh_evidence": False,
        "may_feed_stage60_or_stage70": False,
        "may_feed_another_experiment": False,
    }
    atomic_json(
        development_path,
        {**source_index_base, "index_hash": canonical_hash(source_index_base)},
    )

    barrier = read_json(cache.root / identity.label_free_barrier)
    barrier_base = {
        key: value for key, value in barrier.items() if key != "barrier_hash"
    }
    partition_hash = barrier.get("partition_hash")
    if (
        barrier.get("barrier_hash") != canonical_hash(barrier_base)
        or type(partition_hash) is not str
        or len(partition_hash) != 64
    ):
        raise ProtocolError("HARP evaluation release lacks its partition identity.")
    ordered_keys = [list(key) for key in evaluation_keys]
    key_binding = {"ordered_cache_keys": ordered_keys}
    release_base: dict[str, object] = {
        "schema_version": EVALUATION_RELEASE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "artifact_role": "sealed_evaluation_release_descriptor",
        "split_role": TARGET_EVALUATION_ROLE,
        "canonical_scoring_manifest_relative_path": (
            CANONICAL_SCORING_MANIFEST_RELATIVE_PATH.as_posix()
        ),
        "canonical_scoring_manifest_sha256": expected_manifest_sha256,
        "pre_manifest_cache_content_sha256": cache.content_sha256,
        "cache_index_hash": cache.cache_hash,
        "partition_hash": partition_hash,
        "ordered_cache_keys": ordered_keys,
        "ordered_cache_key_hash": canonical_hash(key_binding),
        "row_count": len(ordered_keys),
        "case_count": len({(center, case) for center, case, _sample in evaluation_keys}),
        "evaluation_scope": "all_218_canonical_test_cases",
        "release_state": "SEALED_UNTIL_TYPED_FROZEN_ROUTE_RECEIPT",
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "fresh_evidence": False,
        "may_feed_stage60_or_stage70": False,
        "may_feed_another_experiment": False,
    }
    atomic_json(
        evaluation_path,
        {**release_base, "descriptor_hash": canonical_hash(release_base)},
    )
    return sha256_file(development_path), sha256_file(evaluation_path)


def _open_source_train_labels_after_barrier(
    canonical_train_cache_root: Path,
    *,
    source_train_frame: CanonicalLabelBlindFrame,
) -> dict[str, int]:
    """Open train-only outcomes after the composite label-free cache is durable."""

    tensor_path = canonical_train_cache_root / "embeddings/train.pt"
    if (
        canonical_train_cache_root.is_symlink()
        or not tensor_path.is_file()
        or tensor_path.is_symlink()
        or sha256_file(tensor_path) != CANONICAL_SOURCE_TRAIN_TENSOR_SHA256
    ):
        raise ProtocolError("HARP v17 source-train label capability drifted.")
    try:
        import torch
        payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - old workstation torch
        payload = torch.load(tensor_path, map_location="cpu")
    except Exception as exc:
        raise ProtocolError("HARP v17 source-train labels are unreadable.") from exc
    metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
    if not isinstance(metadata, Sequence) or isinstance(metadata, (str, bytes)):
        raise ProtocolError("HARP v17 source-train label metadata is malformed.")
    expected = {
        row.sample_id: row
        for center in CENTERS
        for row in source_train_frame.rows_by_center[center]
    }
    labels: dict[str, int] = {}
    for raw in metadata:
        if not isinstance(raw, Mapping):
            raise ProtocolError("HARP v17 source-train label metadata is malformed.")
        raw_id = str(raw.get("sample_id", ""))
        contract_index = raw.get("contract_row_index")
        if type(contract_index) is not int:
            raise ProtocolError("HARP v17 source-train label identity drifted.")
        opaque_id = source_train_row_id(raw_id, int(contract_index))
        row = expected.get(opaque_id)
        try:
            label = int(raw["label"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("HARP v17 source-train label is not binary.") from exc
        if (
            row is None
            or label not in (0, 1)
            or str(raw.get("split")) != "train"
            or str(raw.get("center")) != row.center
            or str(raw.get("case_id")) != row.case_id
            or opaque_id in labels
        ):
            raise ProtocolError("HARP v17 source-train label alignment drifted.")
        labels[opaque_id] = label
    if len(labels) != EXPECTED_SOURCE_TRAIN_ROW_COUNT:
        raise ProtocolError("HARP v17 source-train label coverage drifted.")
    return labels
__all__ = ("evaluation_row_id", "publish_role_pure_manifests")
