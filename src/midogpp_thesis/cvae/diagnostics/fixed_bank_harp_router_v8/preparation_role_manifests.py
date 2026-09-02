"""Publish development truth and a sealed evaluation-release descriptor.

The preparation process may publish development truth after the durable
label-free barrier.  Evaluation truth is deliberately different: only a
label-free descriptor is written.  The canonical scoring manifest is reopened
by the terminal reader after it receives the typed frozen-route capability.
"""

from __future__ import annotations

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
    DEVELOPMENT_ROLE,
    EVALUATION_RELEASE_SCHEMA,
    EVALUATION_ROLE,
    evaluation_row_id,
)
from .preparation_contracts import (
    CanonicalLabelBlindFrame,
    HarpPreparationIdentity,
)
from .preparation_durable_io import atomic_text


def publish_role_pure_manifests(
    canonical_manifest: Path,
    *,
    expected_manifest_sha256: str,
    cache,
    frame: CanonicalLabelBlindFrame,
    development_path: Path,
    evaluation_path: Path,
    identity: HarpPreparationIdentity,
) -> tuple[str, str]:
    """Publish development truth plus a label-free evaluation capability."""

    if (
        not canonical_manifest.is_file()
        or canonical_manifest.is_symlink()
        or sha256_file(canonical_manifest) != expected_manifest_sha256
    ):
        raise ProtocolError("HARP canonical scoring manifest is absent or drifted.")
    source_by_sample = {
        row.sample_id: row
        for center in CENTERS
        for row in frame.rows_by_center[center]
    }
    if len(source_by_sample) != len(cache.rows):
        raise ProtocolError("HARP canonical label-blind identity coverage drifted.")
    barrier = read_json(cache.root / identity.label_free_barrier)
    if barrier.get("canonical_scoring_manifest_opened") is not False:
        raise ProtocolError("HARP scoring manifest opened before the label-free barrier.")

    development_rows: list[tuple[str, str, str, int]] = []
    evaluation_keys: list[tuple[str, str, str]] = []
    expected_by_ordinal: dict[int, tuple[object, object]] = {}
    for cache_row in cache.rows:
        source = source_by_sample.get(cache_row.sample_id)
        if (
            source is None
            or source.contract_row_index in expected_by_ordinal
            or source.center != cache_row.center
            or source.case_id != cache_row.case_id
            or source.center_row_index != cache_row.embedding_row_index
            or cache_row.sample_id
            != evaluation_row_id(
                expected_manifest_sha256, source.contract_row_index
            )
        ):
            raise ProtocolError("HARP cache/source identity alignment drifted.")
        expected_by_ordinal[source.contract_row_index] = (cache_row, source)

    used_manifest_rows: set[int] = set()
    required_fields = {"case_id", "center", "split", "label"}
    try:
        with canonical_manifest.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not required_fields.issubset(reader.fieldnames or ()):
                raise ProtocolError("HARP canonical scoring manifest schema drifted.")
            for ordinal, raw in enumerate(reader):
                expected = expected_by_ordinal.get(ordinal)
                if expected is None:
                    continue
                row, _source = expected
                # The CSV parser must materialize one row.  Immediately erase
                # the outcome cell for evaluation-role rows before retaining
                # any identity, so no tuple/dict holding that value survives
                # this iteration or crosses the preparation boundary.
                raw_value = (
                    raw.get("label")
                    if row.split_role == DEVELOPMENT_ROLE
                    else raw.pop("label", None)
                )
                if (
                    raw.get("split") != "test"
                    or str(raw.get("center")) != row.center
                    or str(raw.get("case_id")) != row.case_id
                    or ordinal in used_manifest_rows
                ):
                    raise ProtocolError("HARP cache/manifest identity alignment drifted.")
                used_manifest_rows.add(ordinal)
                if row.split_role == DEVELOPMENT_ROLE:
                    if str(raw_value) not in {"0", "1"}:
                        raise ProtocolError("HARP development truth is malformed.")
                    development_rows.append(
                        (row.center, row.case_id, row.sample_id, int(raw_value))
                    )
                elif row.split_role == EVALUATION_ROLE:
                    # Do not parse, retain, or publish the erased evaluation
                    # value. The row identity alone binds terminal release.
                    del raw_value
                    evaluation_keys.append((row.center, row.case_id, row.sample_id))
                else:  # pragma: no cover - cache reader closes this set
                    raise ProtocolError("HARP preparation split role drifted.")
    except ProtocolError:
        raise
    except (OSError, csv.Error) as exc:
        raise ProtocolError("HARP canonical scoring manifest is unreadable.") from exc
    if len(used_manifest_rows) != len(cache.rows):
        raise ProtocolError("HARP cache/manifest row coverage drifted.")
    if not development_rows or not evaluation_keys:
        raise ProtocolError("HARP role publication is empty.")

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(("center", "case_id", "sample_id", "label", "split_role"))
    writer.writerows((*values, DEVELOPMENT_ROLE) for values in development_rows)
    atomic_text(development_path, buffer.getvalue())

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
        "split_role": EVALUATION_ROLE,
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
__all__ = ("evaluation_row_id", "publish_role_pure_manifests")
