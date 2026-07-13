"""Domain and target support/evaluation split helpers."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Mapping, Sequence

from .protocol import ProtocolError, assert_disjoint_ids


@dataclass(frozen=True)
class TargetEvalPool:
    eval_indices: tuple[int, ...]
    excluded_support_sample_ids: tuple[str, ...]
    target_eval_pool_id: str


def domain(row: Mapping[str, object]) -> str:
    for key in ("center", "magnification", "domain"):
        value = str(row.get(key, "")).strip()
        if not value:
            continue
        if key == "domain" and value.startswith("center_"):
            return value.split("_", 1)[1]
        return value.replace("x", "")
    raise ProtocolError(f"Metadata row lacks center/domain: {row}")


def label(row: Mapping[str, object]) -> int:
    return int(float(str(row.get("label", 0))))


def sample_id(row: Mapping[str, object]) -> str:
    value = str(row.get("sample_id", "")).strip()
    if not value:
        raise ProtocolError(f"Metadata row lacks sample_id: {row}")
    return value


def build_target_eval_pool(
    *,
    test_metadata: Sequence[Mapping[str, object]],
    heldout_center: str,
    support_sizes: Sequence[int],
    support_seeds: Sequence[int],
) -> TargetEvalPool:
    """Exclude the union of configured target-support samples by sample id.

    The fallback splitter is random and uses only target sample ids/domains,
    never target labels. Labels are consumed later only for final scoring.
    """

    target_indices = tuple(
        idx for idx, row in enumerate(test_metadata) if domain(row) == str(heldout_center)
    )
    support_ids: set[str] = set()
    for support_size in support_sizes:
        for support_seed in support_seeds:
            support_ids.update(
                sample_id(test_metadata[idx])
                for idx in _random_support_indices(
                    target_indices=target_indices,
                    target_domain=str(heldout_center),
                    support_size=int(support_size),
                    support_seed=int(support_seed),
                )
            )
    eval_indices = tuple(
        idx for idx in target_indices if sample_id(test_metadata[idx]) not in support_ids
    )
    assert_disjoint_ids(support_ids, {sample_id(test_metadata[idx]) for idx in eval_indices})
    digest = hashlib.sha256("|".join(sorted(support_ids)).encode("utf-8")).hexdigest()[:12]
    return TargetEvalPool(
        eval_indices=eval_indices,
        excluded_support_sample_ids=tuple(sorted(support_ids)),
        target_eval_pool_id=f"target{heldout_center}_exclude_configured_support_union_{digest}",
    )


def _random_support_indices(
    *,
    target_indices: Sequence[int],
    target_domain: str,
    support_size: int,
    support_seed: int,
) -> tuple[int, ...]:
    if support_size <= 0:
        return ()
    try:
        domain_int = int(str(target_domain).replace("x", ""))
    except ValueError:
        domain_int = sum(ord(ch) for ch in str(target_domain))
    indices = sorted(int(idx) for idx in target_indices)
    rng = random.Random(int(support_seed) + domain_int * 1009)
    rng.shuffle(indices)
    return tuple(indices[: int(support_size)])
