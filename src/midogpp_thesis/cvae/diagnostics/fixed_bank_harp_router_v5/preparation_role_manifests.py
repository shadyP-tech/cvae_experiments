"""Publication of role-pure label capabilities after the label-free barrier."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json, sha256_file
from .input_surfaces import DEVELOPMENT_ROLE, EVALUATION_ROLE
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
    """Publish disjoint label capabilities only after the durable barrier."""

    if (
        not canonical_manifest.is_file()
        or canonical_manifest.is_symlink()
        or sha256_file(canonical_manifest) != expected_manifest_sha256
    ):
        raise ProtocolError("HARP canonical scoring manifest is absent or drifted.")
    try:
        with canonical_manifest.open("r", encoding="utf-8", newline="") as handle:
            raw_rows = tuple(dict(row) for row in csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise ProtocolError("HARP canonical scoring manifest is unreadable.") from exc
    required_fields = {"case_id", "center", "split", "label"}
    if not raw_rows or not required_fields.issubset(raw_rows[0]):
        raise ProtocolError("HARP canonical scoring manifest schema drifted.")
    by_index = {ordinal: row for ordinal, row in enumerate(raw_rows)}
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

    output_by_role: dict[str, list[tuple[str, str, str, int]]] = {
        DEVELOPMENT_ROLE: [],
        EVALUATION_ROLE: [],
    }
    used_manifest_rows: set[int] = set()
    for row in cache.rows:
        source = source_by_sample.get(row.sample_id)
        if source is None:
            raise ProtocolError("HARP cache/source identity alignment drifted.")
        ordinal = source.contract_row_index
        raw = by_index.get(ordinal)
        if (
            raw is None
            or source.center != row.center
            or source.case_id != row.case_id
            or source.center_row_index != row.embedding_row_index
            or raw.get("split") != "test"
            or str(raw.get("center")) != row.center
            or str(raw.get("case_id")) != row.case_id
            or row.sample_id != evaluation_row_id(expected_manifest_sha256, ordinal)
            or ordinal in used_manifest_rows
            or str(raw.get("label")) not in {"0", "1"}
        ):
            raise ProtocolError("HARP cache/manifest identity alignment drifted.")
        used_manifest_rows.add(ordinal)
        output_by_role[row.split_role].append(
            (row.center, row.case_id, row.sample_id, int(raw["label"]))
        )
    if len(used_manifest_rows) != len(cache.rows):
        raise ProtocolError("HARP cache/manifest row coverage drifted.")
    for role, path in (
        (DEVELOPMENT_ROLE, development_path),
        (EVALUATION_ROLE, evaluation_path),
    ):
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(("center", "case_id", "sample_id", "label", "split_role"))
        writer.writerows((*values, role) for values in output_by_role[role])
        atomic_text(path, buffer.getvalue())
    return sha256_file(development_path), sha256_file(evaluation_path)


def evaluation_row_id(manifest_sha256: str, contract_row_index: int) -> str:
    payload = {
        "manifest_sha256": manifest_sha256,
        "contract_row_index": contract_row_index,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return f"eval_{hashlib.sha256(encoded).hexdigest()}"


__all__ = ("evaluation_row_id", "publish_role_pure_manifests")
