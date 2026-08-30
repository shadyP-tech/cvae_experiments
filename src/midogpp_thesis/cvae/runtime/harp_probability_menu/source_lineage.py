"""Storage-independent identity for frozen HARP source streams."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ...expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from ...protocol import ProtocolError
from .hashing import canonical_sha256, require_digest, require_sha256


def _value(record: object, name: str) -> object:
    if isinstance(record, Mapping):
        return record.get(name)
    return getattr(record, name, None)


def harp_source_stream_content_hash(records: Sequence[object]) -> str:
    """Hash the ordered generated outputs without paths or container metadata.

    Stage-60 and Stage-70 have different artifact/config identities.  Their
    source-stream lock files therefore cannot be equal even when every generated
    float32 block is equal.  This receipt binds the scientific stream content:
    exact source/seed keys, expert identity, capacity, and output bytes.
    """

    expected = tuple(
        (center, training_seed, generation_seed)
        for center in CENTERS
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
    )
    rows: list[dict[str, object]] = []
    observed: list[tuple[str, int, int]] = []
    for record in records:
        try:
            center = str(_value(record, "source_center"))
            training_seed = int(_value(record, "training_seed"))
            generation_seed = int(_value(record, "generation_seed"))
            stream_id = str(_value(record, "stream_id"))
            rows_per_class = int(_value(record, "rows_per_class"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProtocolError("HARP source-stream record is malformed.") from exc
        key = (center, training_seed, generation_seed)
        expert_hash = require_digest(
            _value(record, "expert_lock_hash"), name="source expert lock"
        )
        output_hash = require_sha256(
            _value(record, "output_sha256"), name="source output bytes"
        )
        if not stream_id or rows_per_class <= 0:
            raise ProtocolError("HARP source-stream identity or capacity is invalid.")
        observed.append(key)
        rows.append(
            {
                "source_center": center,
                "training_seed": training_seed,
                "generation_seed": generation_seed,
                "stream_id": stream_id,
                "expert_lock_hash": expert_hash,
                "rows_per_class": rows_per_class,
                "output_sha256": output_hash,
            }
        )
    if tuple(observed) != expected:
        raise ProtocolError("HARP source-stream content coverage/order drifted.")
    return canonical_sha256(
        {
            "schema_version": "midogpp_harp_source_stream_content_v1",
            "records": rows,
        }
    )


__all__ = ("harp_source_stream_content_hash",)
