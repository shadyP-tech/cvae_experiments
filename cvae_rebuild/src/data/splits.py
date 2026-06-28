from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Mapping, Sequence

from protocol import ProtocolError, assert_support_eval_disjoint


@dataclass(frozen=True)
class SupportEvalSplit:
    heldout_center: str
    support_indices: tuple[int, ...]
    eval_indices: tuple[int, ...]
    support_sample_ids: tuple[str, ...]
    eval_sample_ids: tuple[str, ...]
    support_size_requested: int
    support_size_actual: int
    support_seed: int
    support_eval_split_id: str
    support_labels_used: bool = False


@dataclass(frozen=True)
class SourceTrainValSplit:
    center: str
    train_indices: tuple[int, ...]
    val_indices: tuple[int, ...]
    train_sample_ids: tuple[str, ...]
    val_sample_ids: tuple[str, ...]
    split_id: str
    val_fraction: float = 0.2


@dataclass(frozen=True)
class TargetEvalPool:
    eval_indices: tuple[int, ...]
    excluded_support_sample_ids: tuple[str, ...]
    target_eval_pool_id: str


def sail_domain(row: Mapping[str, object]) -> str:
    for key in ("center", "magnification", "domain"):
        value = str(row.get(key, "")).strip()
        if not value:
            continue
        if key == "domain" and value.startswith("center_"):
            return value.split("_", 1)[1]
        return value.replace("x", "")
    raise ProtocolError(f"Metadata row lacks center/domain: {row}")


def sail_sample_id(row: Mapping[str, object]) -> str:
    value = str(row.get("sample_id", "")).strip()
    if not value:
        raise ProtocolError(f"Metadata row lacks sample_id: {row}")
    return value


def build_sail_target_eval_pool(
    *,
    test_metadata: Sequence[Mapping[str, object]],
    heldout_center: str,
    support_sizes: Sequence[int],
    support_seeds: Sequence[int],
) -> TargetEvalPool:
    """Mirror the SAIL support-union exclusion split without importing SAIL."""

    target_indices = tuple(
        idx for idx, row in enumerate(test_metadata) if sail_domain(row) == str(heldout_center)
    )
    support_ids: set[str] = set()
    for support_size in support_sizes:
        for support_seed in support_seeds:
            support_ids.update(
                sail_sample_id(test_metadata[idx])
                for idx in _sail_random_support_indices(
                    target_indices=target_indices,
                    target_domain=str(heldout_center),
                    support_size=int(support_size),
                    support_seed=int(support_seed),
                )
            )
    eval_indices = tuple(
        idx for idx in target_indices if sail_sample_id(test_metadata[idx]) not in support_ids
    )
    assert_support_eval_disjoint(support_ids, {sail_sample_id(test_metadata[idx]) for idx in eval_indices})
    digest = hashlib.sha256("|".join(sorted(support_ids)).encode("utf-8")).hexdigest()[:12]
    return TargetEvalPool(
        eval_indices=eval_indices,
        excluded_support_sample_ids=tuple(sorted(support_ids)),
        target_eval_pool_id=f"target{heldout_center}_exclude_configured_support_union_{digest}",
    )


def candidate_experts(centers: Sequence[str], heldout_center: str) -> tuple[str, ...]:
    out = tuple(str(v) for v in sorted({str(c) for c in centers}) if str(v) != str(heldout_center))
    if str(heldout_center) in out:
        raise ProtocolError("Held-out center appeared in source candidate experts.")
    return out


def random_unlabeled_support_eval_split(
    metadata: Sequence[Mapping[str, object]],
    *,
    heldout_center: str,
    support_size: int,
    support_seed: int,
    center_key: str = "center",
    sample_id_key: str = "sample_id",
) -> SupportEvalSplit:
    """Select support/eval IDs without consulting target labels."""

    target_indices = [
        idx
        for idx, row in enumerate(metadata)
        if _row_center(row, center_key=center_key) == str(heldout_center)
    ]
    if len(target_indices) <= int(support_size):
        raise ProtocolError(
            f"Need more than {support_size} target samples for support/eval split; got {len(target_indices)}."
        )
    rng = random.Random(int(support_seed))
    shuffled = list(target_indices)
    rng.shuffle(shuffled)
    support = tuple(sorted(shuffled[: int(support_size)]))
    eval_indices = tuple(sorted(idx for idx in target_indices if idx not in set(support)))
    support_ids = tuple(_sample_id(metadata[idx], idx, sample_id_key=sample_id_key) for idx in support)
    eval_ids = tuple(_sample_id(metadata[idx], idx, sample_id_key=sample_id_key) for idx in eval_indices)
    assert_support_eval_disjoint(support_ids, eval_ids)
    return SupportEvalSplit(
        heldout_center=str(heldout_center),
        support_indices=support,
        eval_indices=eval_indices,
        support_sample_ids=support_ids,
        eval_sample_ids=eval_ids,
        support_size_requested=int(support_size),
        support_size_actual=len(support),
        support_seed=int(support_seed),
        support_eval_split_id=f"target{heldout_center}_seed{support_seed}_random_unlabeled_k{support_size}",
        support_labels_used=False,
    )


def stratified_source_train_val_split(
    metadata: Sequence[Mapping[str, object]],
    *,
    center: str,
    experiment_seed: int,
    val_fraction: float = 0.2,
    center_key: str = "center",
    sample_id_key: str = "sample_id",
    label_key: str = "label",
) -> SourceTrainValSplit:
    """Split one source center's train rows into expert-train/source-val."""

    by_label: dict[int, list[int]] = {}
    for idx, row in enumerate(metadata):
        if _row_center(row, center_key=center_key) != str(center):
            continue
        label = int(float(str(row.get(label_key, 0))))
        by_label.setdefault(label, []).append(idx)
    if set(by_label) != {0, 1}:
        raise ProtocolError(f"Source center {center} must contain labels 0 and 1.")
    train: list[int] = []
    val: list[int] = []
    stable_seed = _stable_seed(str(experiment_seed), str(center), "source_train_val")
    for label in sorted(by_label):
        indices = sorted(by_label[label])
        if len(indices) < 2:
            raise ProtocolError(f"Source center {center} label {label} needs at least two rows.")
        rng = random.Random(stable_seed + int(label) * 1009)
        rng.shuffle(indices)
        val_count = max(1, int(round(len(indices) * float(val_fraction))))
        val_count = min(val_count, len(indices) - 1)
        val.extend(indices[:val_count])
        train.extend(indices[val_count:])
    train_indices = tuple(sorted(train))
    val_indices = tuple(sorted(val))
    if not train_indices or not val_indices:
        raise ProtocolError(f"Empty source train/validation split for center {center}.")
    return SourceTrainValSplit(
        center=str(center),
        train_indices=train_indices,
        val_indices=val_indices,
        train_sample_ids=tuple(_sample_id(metadata[idx], idx, sample_id_key=sample_id_key) for idx in train_indices),
        val_sample_ids=tuple(_sample_id(metadata[idx], idx, sample_id_key=sample_id_key) for idx in val_indices),
        split_id=f"source{center}_seed{experiment_seed}_stratified_train80_val20",
    )


def _row_center(row: Mapping[str, object], *, center_key: str) -> str:
    if center_key in row:
        return str(row[center_key])
    if "magnification" in row:
        return str(row["magnification"])
    raise ProtocolError(f"Metadata row missing center key {center_key!r}.")


def _sample_id(row: Mapping[str, object], idx: int, *, sample_id_key: str) -> str:
    value = row.get(sample_id_key, "")
    return str(value) if str(value) else f"row_{idx}"


def _sail_random_support_indices(
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


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:12], 16)
