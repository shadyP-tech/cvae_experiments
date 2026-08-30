"""File-backed one-shot scoring-label reader behind the prelabel seal."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping

from ...protocol import ProtocolError
from ...routing.harp_protocol.hashing import canonical_hash
from ...runtime.harp_probability_menu.hashing import require_sha256
from .sealing import HarpFreshPrelabelSeal
from .target_loading import HarpFreshLoadedTarget


SCORING_COLUMNS = (
    "schema_version",
    "row_id",
    "center",
    "case_id",
    "label",
    "reservation_hash",
    "target_cache_content_hash",
)
SCORING_ROW_SCHEMA = "midogpp_harp_fresh_scoring_row_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def open_harp_fresh_scoring_labels(
    target: HarpFreshLoadedTarget,
    seal: HarpFreshPrelabelSeal,
) -> tuple[Mapping[tuple[str, str, str], int], str]:
    """Read labels only after the complete menu/routes/vectors are sealed."""

    if not isinstance(target, HarpFreshLoadedTarget) or not isinstance(
        seal, HarpFreshPrelabelSeal
    ):
        raise ProtocolError("Fresh HARP labels require a typed prelabel seal.")
    if (
        seal.reservation_hash != target.cache.reservation.reservation_hash
        or seal.target_cache_hash != target.cache.cache_hash
        or _sha256_file(target.scoring_manifest_path)
        != target.scoring_manifest_sha256
    ):
        raise ProtocolError("Fresh HARP scoring inputs drifted after route sealing.")
    authorization_path = (
        target.scoring_manifest_path.parent
        / "manifests/scoring_authorization.json"
    )
    try:
        authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("Fresh HARP scoring authorization is unreadable.") from exc
    if not isinstance(authorization, dict):
        raise ProtocolError("Fresh HARP scoring authorization must be an object.")
    observed = authorization.get("authorization_hash")
    if (
        authorization.get("schema_version")
        != "midogpp_harp_fresh_scoring_authorization_v1"
        or authorization.get("status") != "ACTIVE_ONE_SHOT_AFTER_PRELABEL_SEAL"
        or authorization.get("reservation_hash") != seal.reservation_hash
        or authorization.get("target_cache_content_hash")
        != target.cache_content_hash
        or authorization.get("labels_available_before_global_route_seal") is not False
        or authorization.get("labels_may_update_policy") is not False
        or observed
        != canonical_hash(
            {key: value for key, value in authorization.items() if key != "authorization_hash"}
        )
    ):
        raise ProtocolError("Fresh HARP scoring authorization drifted.")
    authorization_hash = require_sha256(observed, name="scoring authorization")
    try:
        with target.scoring_manifest_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != SCORING_COLUMNS:
                raise ProtocolError("Fresh HARP scoring columns drifted.")
            rows = tuple(dict(row) for row in reader)
    except OSError as exc:
        raise ProtocolError("Cannot open fresh HARP scoring manifest.") from exc
    labels: dict[tuple[str, str, str], int] = {}
    for row in rows:
        if (
            row.get("schema_version") != SCORING_ROW_SCHEMA
            or row.get("reservation_hash") != seal.reservation_hash
            or row.get("target_cache_content_hash") != target.cache_content_hash
        ):
            raise ProtocolError("Fresh HARP scoring row lineage drifted.")
        key = (
            str(row.get("center", "")),
            str(row.get("case_id", "")),
            str(row.get("row_id", "")),
        )
        try:
            label = int(str(row.get("label", "")))
        except ValueError as exc:
            raise ProtocolError("Fresh HARP scoring labels must be binary.") from exc
        if key in labels or label not in (0, 1) or str(label) != row.get("label"):
            raise ProtocolError("Fresh HARP scoring label identity drifted.")
        labels[key] = label
    if tuple(labels) != seal.row_keys:
        raise ProtocolError("Fresh HARP scoring rows do not exactly cover sealed rows.")
    return labels, authorization_hash


__all__ = (
    "SCORING_COLUMNS",
    "SCORING_ROW_SCHEMA",
    "open_harp_fresh_scoring_labels",
)
