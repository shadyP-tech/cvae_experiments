from __future__ import annotations

from dataclasses import replace
import json

import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.residual_topup_policy import (
    FIXED_TRAINING_SEEDS,
    GLOBAL_POLICY_ID,
    GLOBAL_PSEUDOQUERY_ROLE,
    PROXY_ENERGY_SEMANTICS,
    SUPPORT_POLICY_ID,
    TARGET_SUPPORT_ROLE,
    FreshProxyScoreRow,
    average_replica_scores_before_ballot,
    build_target_proxy_policy,
    canonical_source_identity_permutation,
    normalized_midranks,
)


TARGET = "H"
SOURCES = ("a", "b", "c", "d")


def _row(
    *,
    role: str,
    query: str,
    case_id: str,
    source: str,
    seed: int,
    energy: float,
) -> FreshProxyScoreRow:
    return FreshProxyScoreRow(
        outer_target=TARGET,
        query_role=role,
        query_center=query,
        case_id=case_id,
        candidate_source=source,
        training_seed=seed,
        proxy_energy=energy,
        labels_consumed=False,
        evaluation_overlap=False,
        source_expert_updated=False,
    )


def _valid_rows(
    *, global_case_count_by_query: dict[str, int] | None = None
) -> list[FreshProxyScoreRow]:
    case_counts = global_case_count_by_query or {
        query: 2 for query in SOURCES
    }
    rows: list[FreshProxyScoreRow] = []
    for query in SOURCES:
        for case_index in range(case_counts[query]):
            case_id = f"global-{query}-{case_index}"
            for source_index, source in enumerate(SOURCES):
                if source == query:
                    continue
                for seed_index, seed in enumerate(FIXED_TRAINING_SEEDS):
                    rows.append(
                        _row(
                            role=GLOBAL_PSEUDOQUERY_ROLE,
                            query=query,
                            case_id=case_id,
                            source=source,
                            seed=seed,
                            energy=float(source_index * 10 + case_index + seed_index),
                        )
                    )
    for case_index in range(2):
        case_id = f"support-{case_index}"
        for source_index, source in enumerate(SOURCES):
            for seed_index, seed in enumerate(FIXED_TRAINING_SEEDS):
                rows.append(
                    _row(
                        role=TARGET_SUPPORT_ROLE,
                        query=TARGET,
                        case_id=case_id,
                        source=source,
                        seed=seed,
                        energy=float(source_index * 10 + case_index + seed_index),
                    )
                )
    return rows


def test_normalized_true_midranks_preserve_ties() -> None:
    ranks = normalized_midranks({"d": 4.0, "c": 2.0, "b": 2.0, "a": 1.0})
    assert dict(ranks) == {
        "a": 0.0,
        "b": 0.5,
        "c": 0.5,
        "d": 1.0,
    }
    reverse = normalized_midranks(
        [4.0, 2.0, 2.0, 1.0],
        source_ids=["d", "c", "b", "a"],
        lower_is_better=False,
    )
    assert dict(reverse) == {
        "a": 1.0,
        "b": 0.5,
        "c": 0.5,
        "d": 0.0,
    }
    with pytest.raises(TypeError):
        ranks["a"] = 1.0  # type: ignore[index]


def test_three_replicas_are_averaged_before_each_case_ballot() -> None:
    rows: list[FreshProxyScoreRow] = []
    replica_values = {
        "a": (0.0, 0.0, 9.0),
        "b": (2.0, 2.0, 2.0),
        "c": (4.0, 4.0, 4.0),
        "d": (5.0, 5.0, 5.0),
    }
    for source, energies in replica_values.items():
        for seed, energy in zip(FIXED_TRAINING_SEEDS, energies, strict=True):
            rows.append(
                _row(
                    role=TARGET_SUPPORT_ROLE,
                    query=TARGET,
                    case_id="support-one",
                    source=source,
                    seed=seed,
                    energy=energy,
                )
            )
    ballots = average_replica_scores_before_ballot(
        rows,
        outer_target=TARGET,
        candidate_sources=reversed(SOURCES),
        query_role=TARGET_SUPPORT_ROLE,
    )
    assert len(ballots) == 1
    ballot = ballots[0]
    assert ballot.mean_proxy_energy_by_source["a"] == 3.0
    assert ballot.mean_proxy_energy_by_source["b"] == 2.0
    assert ballot.normalized_midrank_by_source["b"] == 0.0
    assert ballot.normalized_midrank_by_source["a"] == pytest.approx(1.0 / 3.0)


def test_policy_builds_leave_h_q_out_g_and_target_support_only_s() -> None:
    policy = build_target_proxy_policy(
        reversed(_valid_rows()),
        outer_target=TARGET,
        candidate_sources=reversed(SOURCES),
    )
    assert policy.candidate_sources == SOURCES
    assert policy.global_summary.policy_id == GLOBAL_POLICY_ID
    assert policy.support_summary.policy_id == SUPPORT_POLICY_ID
    assert policy.global_summary.query_centers == SOURCES
    assert policy.support_summary.query_centers == (TARGET,)
    assert dict(policy.global_summary.case_count_by_query_center) == {
        source: 2 for source in SOURCES
    }
    assert dict(policy.global_summary.ballot_count_by_source) == {
        source: 6 for source in SOURCES
    }
    assert dict(policy.support_summary.ballot_count_by_source) == {
        source: 2 for source in SOURCES
    }
    for ballot in policy.global_summary.ballots:
        assert TARGET not in ballot.candidate_sources
        assert ballot.query_center not in ballot.candidate_sources
    for summary in (policy.global_summary, policy.support_summary):
        assert all(
            summary.priority_by_source[source]
            == pytest.approx(
                1.0 - summary.mean_normalized_midrank_by_source[source]
            )
            for source in SOURCES
        )
    assert policy.actions_constructed is False
    assert PROXY_ENERGY_SEMANTICS.startswith("proxy_only_")
    json.dumps(policy.to_payload(), sort_keys=True)
    with pytest.raises(TypeError):
        policy.global_summary.priority_by_source["a"] = 0.0  # type: ignore[index]


def test_source_identity_permutation_is_canonical_deterministic_derangement() -> None:
    first = canonical_source_identity_permutation(
        ["d", "a", "c", "b"], permutation_index=1
    )
    second = canonical_source_identity_permutation(
        ["b", "c", "a", "d"], permutation_index=1
    )
    assert dict(first) == dict(second) == {
        "a": "b",
        "b": "c",
        "c": "d",
        "d": "a",
    }
    assert set(first) == set(first.values()) == set(SOURCES)
    assert all(source != permuted for source, permuted in first.items())
    with pytest.raises(ProtocolError, match="nonzero canonical rotation"):
        canonical_source_identity_permutation(SOURCES, permutation_index=0)


def test_missing_and_duplicate_proxy_grid_cells_fail_closed() -> None:
    rows = _valid_rows()
    with pytest.raises(ProtocolError, match="missing, extra, or seed-drifted"):
        build_target_proxy_policy(
            rows[:-1], outer_target=TARGET, candidate_sources=SOURCES
        )
    with pytest.raises(ProtocolError, match="Duplicate proxy score grid cell"):
        build_target_proxy_policy(
            [*rows, rows[0]], outer_target=TARGET, candidate_sources=SOURCES
        )


def test_unequal_global_pseudoquery_case_coverage_fails_closed() -> None:
    rows = _valid_rows(
        global_case_count_by_query={"a": 1, "b": 2, "c": 2, "d": 2}
    )
    with pytest.raises(ProtocolError, match="case coverage must be equal"):
        build_target_proxy_policy(
            rows, outer_target=TARGET, candidate_sources=SOURCES
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"proxy_energy": float("nan")}, "finite"),
        ({"training_seed": 999}, "training-seed drift"),
        ({"labels_consumed": True}, "no-label attestation"),
        ({"evaluation_overlap": True}, "no-evaluation-overlap"),
        ({"source_expert_updated": True}, "no-source-expert-update"),
        ({"proxy_energy_semantics": "calibrated_score"}, "semantics drifted"),
    ),
)
def test_score_row_attestations_and_fixed_semantics_fail_closed(
    changes: dict[str, object], message: str
) -> None:
    valid = _row(
        role=TARGET_SUPPORT_ROLE,
        query=TARGET,
        case_id="case",
        source="a",
        seed=17,
        energy=1.0,
    )
    with pytest.raises(ProtocolError, match=message):
        replace(valid, **changes)


def test_h_and_q_candidate_leakage_fail_closed_at_row_boundary() -> None:
    with pytest.raises(ProtocolError, match="Outer target H leaked"):
        _row(
            role=TARGET_SUPPORT_ROLE,
            query=TARGET,
            case_id="case",
            source=TARGET,
            seed=17,
            energy=1.0,
        )
    with pytest.raises(ProtocolError, match="pseudoquery q leaked"):
        _row(
            role=GLOBAL_PSEUDOQUERY_ROLE,
            query="a",
            case_id="case",
            source="a",
            seed=17,
            energy=1.0,
        )


def test_case_ids_must_be_disjoint_across_query_surfaces() -> None:
    rows = _valid_rows()
    first_support = next(
        row for row in rows if row.query_role == TARGET_SUPPORT_ROLE
    )
    first_global = next(
        row for row in rows if row.query_role == GLOBAL_PSEUDOQUERY_ROLE
    )
    rows[rows.index(first_support)] = replace(
        first_support, case_id=first_global.case_id
    )
    with pytest.raises(ProtocolError, match="disjoint across query surfaces"):
        build_target_proxy_policy(
            rows, outer_target=TARGET, candidate_sources=SOURCES
        )
