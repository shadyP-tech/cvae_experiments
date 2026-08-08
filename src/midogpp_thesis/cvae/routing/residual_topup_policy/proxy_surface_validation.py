"""Closed-grid and embedding validation for fresh proxy surfaces."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from .contracts import (
    FIXED_TRAINING_SEEDS,
    GLOBAL_PSEUDOQUERY_ROLE,
    TARGET_SUPPORT_ROLE,
    FreshProxyScoreRow,
)
from .proxy_surface_contracts import (
    ArrayLoader,
    EXPECTED_QUERY_SHARD_COUNT,
    FreshProxyScoreTask,
    FreshQueryShard,
    embedding_array_sha256,
    shard_sort_key,
    source_is_legal_for_shard,
    validated_embedding_array,
)


def validate_fresh_proxy_score_surface(
    rows: Iterable[FreshProxyScoreRow],
    *,
    shards: Iterable[FreshQueryShard],
) -> tuple[FreshProxyScoreRow, ...]:
    """Validate exact case/source/seed coverage and return canonical row order."""

    shard_tuple = validate_query_shards(tuple(shards))
    score_rows = tuple(rows)
    if not score_rows or any(
        not isinstance(row, FreshProxyScoreRow) for row in score_rows
    ):
        raise ProtocolError("Fresh proxy score surface rows are invalid or empty.")
    expected: set[tuple[str, str, str, str, str, int]] = set()
    for shard in shard_tuple:
        for source in CENTERS:
            if not source_is_legal_for_shard(source, shard):
                continue
            for training_seed in FIXED_TRAINING_SEEDS:
                for case_id in shard.unique_case_ids:
                    expected.add(
                        (
                            shard.outer_target,
                            shard.query_role,
                            shard.query_center,
                            case_id,
                            source,
                            training_seed,
                        )
                    )
    observed = {score_row_key(row) for row in score_rows}
    if len(score_rows) != len(observed) or observed != expected:
        raise ProtocolError("Fresh proxy score surface grid coverage drifted.")
    return tuple(sorted(score_rows, key=score_row_sort_key))


def validate_query_shards(
    shards: Sequence[FreshQueryShard],
) -> tuple[FreshQueryShard, ...]:
    if len(shards) != EXPECTED_QUERY_SHARD_COUNT or any(
        not isinstance(shard, FreshQueryShard) for shard in shards
    ):
        raise ProtocolError(
            "Fresh proxy surface requires eight G shards and one S shard per H."
        )
    by_key = {shard.key: shard for shard in shards}
    if len(by_key) != len(shards):
        raise ProtocolError("Fresh proxy query shards duplicate.")
    expected = {
        (target, GLOBAL_PSEUDOQUERY_ROLE, query)
        for target in CENTERS
        for query in CENTERS
        if query != target
    }.union(
        {
            (target, TARGET_SUPPORT_ROLE, target)
            for target in CENTERS
        }
    )
    if set(by_key) != expected:
        raise ProtocolError("Fresh proxy query shard H/role/q grid is incomplete.")
    validated: list[FreshQueryShard] = []
    for shard in shards:
        validated.append(
            FreshQueryShard(
                outer_target=shard.outer_target,
                query_role=shard.query_role,
                query_center=shard.query_center,
                embedding_path=shard.embedding_path,
                embedding_array_sha256=shard.embedding_array_sha256,
                case_ids=shard.case_ids,
                evaluation_case_ids=shard.evaluation_case_ids,
                shard_hash=shard.shard_hash,
                labels_consumed=shard.labels_consumed,
                evaluation_overlap=shard.evaluation_overlap,
                source_experts_updated=shard.source_experts_updated,
            )
        )

    # A global q is one physical, H-independent pseudoquery.  The H index only
    # controls its legal candidate set; it must never trigger a second score.
    pseudoquery_cases: dict[str, tuple[str, ...]] = {}
    for query in CENTERS:
        aliases = [
            shard
            for shard in validated
            if shard.query_role == GLOBAL_PSEUDOQUERY_ROLE
            and shard.query_center == query
        ]
        identities = {
            (shard.embedding_array_sha256, shard.case_ids) for shard in aliases
        }
        if len(aliases) != len(CENTERS) - 1 or len(identities) != 1:
            raise ProtocolError(
                "Fresh G pseudoquery q must keep the same embedding hash and "
                "row-aligned case IDs across every eligible H."
            )
        pseudoquery_cases[query] = aliases[0].unique_case_ids
    if len({len(cases) for cases in pseudoquery_cases.values()}) != 1:
        raise ProtocolError(
            "Fresh global pseudoquery shards require equal case coverage."
        )

    support_cases = {
        target: by_key[(target, TARGET_SUPPORT_ROLE, target)].unique_case_ids
        for target in CENTERS
    }
    evaluation_cases: dict[str, tuple[str, ...]] = {}
    for target in CENTERS:
        target_shards = [
            shard for shard in validated if shard.outer_target == target
        ]
        sets = {
            frozenset(shard.evaluation_case_ids) for shard in target_shards
        }
        if len(sets) != 1:
            raise ProtocolError(
                "Fresh query shards for each H must attest the same evaluation set."
            )
        evaluation_cases[target] = tuple(sorted(sets.pop()))

    require_globally_disjoint_case_grids(
        pseudoquery_cases=pseudoquery_cases,
        support_cases=support_cases,
        evaluation_cases=evaluation_cases,
    )
    return tuple(sorted(validated, key=shard_sort_key))


def derive_case_grids(
    shards: Sequence[FreshQueryShard],
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
    by_key = {shard.key: shard for shard in shards}
    pseudoquery = {
        query: list(
            by_key[
                (
                    next(target for target in CENTERS if target != query),
                    GLOBAL_PSEUDOQUERY_ROLE,
                    query,
                )
            ].unique_case_ids
        )
        for query in CENTERS
    }
    support = {
        target: list(
            by_key[(target, TARGET_SUPPORT_ROLE, target)].unique_case_ids
        )
        for target in CENTERS
    }
    evaluation = {
        target: sorted(
            set(by_key[(target, TARGET_SUPPORT_ROLE, target)].evaluation_case_ids)
        )
        for target in CENTERS
    }
    return pseudoquery, support, evaluation


def require_globally_disjoint_case_grids(
    *,
    pseudoquery_cases: Mapping[str, Sequence[str]],
    support_cases: Mapping[str, Sequence[str]],
    evaluation_cases: Mapping[str, Sequence[str]],
) -> None:
    flattened: dict[str, set[str]] = {}
    for label, mapping in (
        ("pseudoquery", pseudoquery_cases),
        ("support", support_cases),
        ("evaluation", evaluation_cases),
    ):
        seen: set[str] = set()
        for center in CENTERS:
            cases = set(mapping[center])
            if not cases or seen.intersection(cases):
                raise ProtocolError(
                    f"Fresh {label} case IDs must be globally disjoint by center."
                )
            seen.update(cases)
        flattened[label] = seen
    if (
        flattened["pseudoquery"].intersection(flattened["support"])
        or flattened["pseudoquery"].intersection(flattened["evaluation"])
        or flattened["support"].intersection(flattened["evaluation"])
    ):
        raise ProtocolError(
            "Fresh pseudoquery, support, and evaluation case grids must be "
            "globally disjoint."
        )


def load_shard_embeddings(
    shard: FreshQueryShard,
    *,
    array_loader: ArrayLoader,
) -> np.ndarray:
    values = validated_embedding_array(
        array_loader(shard.embedding_path),
        expected_row_count=len(shard.case_ids),
    )
    if embedding_array_sha256(values) != shard.embedding_array_sha256:
        raise ProtocolError("Fresh proxy embedding shard bytes drifted.")
    return values


def deduplicated_task_scoring_groups(
    task: FreshProxyScoreTask,
) -> tuple[tuple[FreshQueryShard, tuple[FreshQueryShard, ...]], ...]:
    """Return 8 canonical G scores plus 8 target-specific S scores per task."""

    grouped: dict[tuple[str, str], list[FreshQueryShard]] = {}
    for shard in task.shards:
        key = (shard.query_role, shard.query_center)
        grouped.setdefault(key, []).append(shard)
    ordered: list[tuple[FreshQueryShard, tuple[FreshQueryShard, ...]]] = []
    for key in sorted(
        grouped,
        key=lambda value: (
            0 if value[0] == GLOBAL_PSEUDOQUERY_ROLE else 1,
            CENTERS.index(value[1]),
        ),
    ):
        members = tuple(sorted(grouped[key], key=shard_sort_key))
        representative = members[0]
        if representative.query_role == GLOBAL_PSEUDOQUERY_ROLE:
            if any(
                member.embedding_array_sha256
                != representative.embedding_array_sha256
                or member.case_ids != representative.case_ids
                for member in members[1:]
            ):
                raise ProtocolError(
                    "Fresh G pseudoquery aliases drifted within an expert task."
                )
        elif len(members) != 1:
            raise ProtocolError("Fresh S query unexpectedly replicated across H.")
        ordered.append((representative, members))
    expected_group_count = 2 * (len(CENTERS) - 1)
    if len(ordered) != expected_group_count:
        raise ProtocolError("Fresh proxy task unique query coverage drifted.")
    return tuple(ordered)


def expected_task_row_keys(
    task: FreshProxyScoreTask,
) -> set[tuple[str, str, str, str, str, int]]:
    return {
        (
            shard.outer_target,
            shard.query_role,
            shard.query_center,
            case_id,
            task.source_center,
            task.training_seed,
        )
        for shard in task.shards
        for case_id in shard.unique_case_ids
    }


def query_shard_attestation_key(shard: FreshQueryShard) -> str:
    return "::".join(
        (shard.outer_target, shard.query_role, shard.query_center)
    )


def score_row_key(
    row: FreshProxyScoreRow,
) -> tuple[str, str, str, str, str, int]:
    return (
        row.outer_target,
        row.query_role,
        row.query_center,
        row.case_id,
        row.candidate_source,
        row.training_seed,
    )


def score_row_sort_key(
    row: FreshProxyScoreRow,
) -> tuple[int, int, int, str, int, int]:
    role_order = 0 if row.query_role == GLOBAL_PSEUDOQUERY_ROLE else 1
    return (
        CENTERS.index(row.outer_target),
        role_order,
        CENTERS.index(row.query_center),
        row.case_id,
        CENTERS.index(row.candidate_source),
        FIXED_TRAINING_SEEDS.index(row.training_seed),
    )


__all__ = (
    "deduplicated_task_scoring_groups",
    "derive_case_grids",
    "expected_task_row_keys",
    "load_shard_embeddings",
    "query_shard_attestation_key",
    "require_globally_disjoint_case_grids",
    "score_row_key",
    "score_row_sort_key",
    "validate_fresh_proxy_score_surface",
    "validate_query_shards",
)
