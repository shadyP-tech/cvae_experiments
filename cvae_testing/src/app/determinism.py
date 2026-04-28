from __future__ import annotations

import hashlib
from typing import Iterable


RESPONSE_SEED_SCHEME_VERSION = "sha256_v1"


def stable_hash_to_int(parts: Iterable[object], modulus: int = 2_147_483_647) -> int:
    if int(modulus) <= 0:
        raise ValueError("modulus must be > 0")

    text = "||".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % int(modulus)


def stable_response_seed(
    *,
    dataset: str,
    seed: int,
    query_id: str | int,
    expert_domain: str | int,
    repeat_id: int,
    stream_name: str,
    modulus: int = 2_147_483_647,
) -> int:
    return stable_hash_to_int(
        [
            "response_seed",
            RESPONSE_SEED_SCHEME_VERSION,
            str(dataset),
            int(seed),
            str(query_id),
            str(expert_domain),
            int(repeat_id),
            str(stream_name),
        ],
        modulus=modulus,
    )
