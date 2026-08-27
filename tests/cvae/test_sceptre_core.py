from __future__ import annotations

from dataclasses import replace
from itertools import product
from pathlib import Path

import pytest

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
    legal_routing_sources,
)
from midogpp_thesis.cvae.generation.contracts import (
    CLAIM_SCOPE,
    EXPECTED_BANK_LOCK_HASH,
    SOURCE_BUDGET_PER_CLASS,
    SOURCE_STREAM_NAMESPACE,
    TOTAL_PER_CLASS,
    GenerationLock,
)
from midogpp_thesis.cvae.generation.generation import source_generation_plan
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.sceptre import (
    ExactBFallback,
    FamilyProxyScore,
    RawRoute,
    aggregate_menu_proxy_scores,
    aggregate_training_replica_scores,
    assert_import_source_fence,
    build_candidate_menu,
    build_candidate_menu_from_keys,
    rank_family_proxy_scores,
    replay_semantic_contract,
    route_raw_proxy_evidence_or_exact_b,
    route_unique_winner_or_exact_b,
    validate_candidate_and_b_control,
)


def _generation_lock_payload() -> dict[str, object]:
    expert_locks = [
        {
            "source_center": center,
            "training_seed": training_seed,
            "expert_lock_hash": stable_hash(
                {"source_center": center, "training_seed": training_seed}
            ),
        }
        for center in CENTERS
        for training_seed in TRAINING_SEEDS
    ]
    payload: dict[str, object] = {
        "schema_version": "midogpp_uniform_b_v2_generation_lock_v1",
        "claim_scope": CLAIM_SCOPE,
        "bank": {
            "bank_lock_hash": EXPECTED_BANK_LOCK_HASH,
            "expert_locks": expert_locks,
            "candidate_sources_by_target": {
                target: list(legal_routing_sources(target)) for target in CENTERS
            },
        },
        "generation": {
            "training_seeds": list(TRAINING_SEEDS),
            "generation_seeds": list(GENERATION_SEEDS),
            "source_stream_namespace": SOURCE_STREAM_NAMESPACE,
            "max_source_block_per_class": TOTAL_PER_CLASS,
            "equal_union_source_budget_per_class": SOURCE_BUDGET_PER_CLASS,
            "total_per_class": TOTAL_PER_CLASS,
        },
    }
    payload["generation_lock_hash"] = stable_hash(payload)
    return payload


def _lock() -> GenerationLock:
    return GenerationLock(_generation_lock_payload())


def _candidate_keys(lock: GenerationLock, target: str):
    sources = set(legal_routing_sources(target))
    return tuple(
        row for row in source_generation_plan(lock) if row.source_center in sources
    )


def _scores(menu, *, tied: bool = False):
    rows = {}
    for index, source in enumerate(menu.candidate_sources):
        base = 0.0 if tied and index < 2 else float(index)
        rows[source] = {17: base + 0.1, 42: base + 0.2, 101: base + 0.3}
    return aggregate_menu_proxy_scores(menu, rows)


def test_all_nine_targets_have_exact_eight_source_family_candidates() -> None:
    lock = _lock()
    for target in CENTERS:
        menu = build_candidate_menu(lock, target)
        assert menu.candidate_sources == legal_routing_sources(target)
        assert len(menu.families) == 8
        assert target not in menu.candidate_sources
        assert len({family.family_hash for family in menu.families}) == 8


def test_each_source_family_binds_exact_three_by_three_replica_grid() -> None:
    menu = build_candidate_menu(_lock(), "0")
    expected = set(product(TRAINING_SEEDS, GENERATION_SEEDS))
    for family in menu.families:
        assert len(family.stream_keys) == 9
        assert {
            (row.training_seed, row.generation_seed) for row in family.stream_keys
        } == expected
        assert len({row.training_seed for row in family.stream_keys}) == 3
        assert len({row.generation_seed for row in family.stream_keys}) == 3


def test_candidate_menu_rejects_target_missing_duplicate_and_seed_drift() -> None:
    lock = _lock()
    rows = _candidate_keys(lock, "0")
    with pytest.raises(ProtocolError, match="target expert"):
        build_candidate_menu_from_keys(
            lock,
            "0",
            (*rows, next(row for row in source_generation_plan(lock) if row.source_center == "0")),
        )
    with pytest.raises(ProtocolError, match="three-by-three"):
        build_candidate_menu_from_keys(lock, "0", rows[:-1])
    with pytest.raises(ProtocolError, match="duplicate stream ids"):
        build_candidate_menu_from_keys(lock, "0", (*rows, rows[0]))

    source = rows[0].source_center
    source_rows = [row for row in rows if row.source_center == source]
    replacement = replace(
        source_rows[-1],
        training_seed=source_rows[0].training_seed,
        generation_seed=source_rows[0].generation_seed,
        stream_id="unique-but-seed-duplicated",
    )
    drifted = tuple(replacement if row is source_rows[-1] else row for row in rows)
    with pytest.raises(ProtocolError, match="three-by-three"):
        build_candidate_menu_from_keys(lock, "0", drifted)

    forged_rows = tuple(
        replace(row, expert_lock_hash="forged-expert-lock")
        if row.source_center == source and row.training_seed == TRAINING_SEEDS[0]
        else row
        for row in rows
    )
    with pytest.raises(ProtocolError, match="differs from GenerationLock"):
        build_candidate_menu_from_keys(lock, "0", forged_rows)


def test_proxy_mean_is_seed_order_invariant_and_exactly_three_replicas() -> None:
    forward = aggregate_training_replica_scores(
        target_center="0",
        source_center="1",
        scores_by_training_seed=((17, 3.0), (42, 6.0), (101, 9.0)),
    )
    reverse = aggregate_training_replica_scores(
        target_center="0",
        source_center="1",
        scores_by_training_seed=((101, 9.0), (42, 6.0), (17, 3.0)),
    )
    assert forward.mean_proxy_energy == 6.0
    assert forward.to_payload() == reverse.to_payload()
    with pytest.raises(ProtocolError, match="exactly three"):
        aggregate_training_replica_scores(
            target_center="0",
            source_center="1",
            scores_by_training_seed={17: 1.0, 42: 2.0},
        )
    with pytest.raises(ProtocolError, match="duplicate seed"):
        aggregate_training_replica_scores(
            target_center="0",
            source_center="1",
            scores_by_training_seed=((17, 1.0), (17, 2.0), (101, 3.0)),
        )


def test_proxy_contract_is_truthful_about_labels_and_nelbo() -> None:
    score = aggregate_training_replica_scores(
        target_center="0",
        source_center="1",
        scores_by_training_seed={17: 1.0, 42: 2.0, 101: 3.0},
    )
    assert score.exact_nelbo is False
    assert score.labels_consumed is False
    assert score.score_semantics == "PROXY_ENERGY_RANK"
    assert score.to_payload()["lower_is_better"] is True
    with pytest.raises(ProtocolError, match="exact NELBO"):
        FamilyProxyScore(
            target_center="0",
            source_center="1",
            training_replica_scores={17: 1.0, 42: 2.0, 101: 3.0},
            exact_nelbo=True,
        )
    with pytest.raises(ProtocolError, match="label-free"):
        FamilyProxyScore(
            target_center="0",
            source_center="1",
            training_replica_scores={17: 1.0, 42: 2.0, 101: 3.0},
            labels_consumed=True,
        )


def test_core_contracts_reject_semantic_hash_tampering() -> None:
    menu = build_candidate_menu(_lock(), "0")
    with pytest.raises(ProtocolError, match="semantic hash drifted"):
        replace(menu, menu_hash="tampered")
    score = aggregate_training_replica_scores(
        target_center="0",
        source_center="1",
        scores_by_training_seed={17: 1.0, 42: 2.0, 101: 3.0},
    )
    with pytest.raises(ProtocolError, match="semantic hash drifted"):
        replace(score, score_hash="tampered")


def test_exact_tie_preserves_true_midranks_and_falls_back_to_b() -> None:
    menu = build_candidate_menu(_lock(), "0")
    ranking = rank_family_proxy_scores(menu, _scores(menu, tied=True))
    assert ranking.winner_sources == menu.candidate_sources[:2]
    expected_tied_midrank = 0.5 / 7.0
    assert ranking.normalized_midrank_by_source[menu.candidate_sources[0]] == pytest.approx(
        expected_tied_midrank
    )
    assert ranking.normalized_midrank_by_source[menu.candidate_sources[1]] == pytest.approx(
        expected_tied_midrank
    )
    decision = route_unique_winner_or_exact_b(menu, ranking)
    assert isinstance(decision, ExactBFallback)
    assert decision.control_id == "B"
    assert decision.winner_sources == menu.candidate_sources[:2]
    assert decision.to_payload()["fake_tie_breaking"] is False


def test_unique_lower_energy_winner_produces_raw_source_family_route() -> None:
    menu = build_candidate_menu(_lock(), "0")
    ranking = rank_family_proxy_scores(menu, _scores(menu))
    decision = route_unique_winner_or_exact_b(menu, ranking)
    assert ranking.winner_sources == (menu.candidate_sources[0],)
    assert isinstance(decision, RawRoute)
    assert decision.selected_source_center == menu.candidate_sources[0]
    assert decision.to_payload()["route_unit"] == "source_center_family"


def test_missing_and_nonfinite_proxy_evidence_fall_back_to_exact_b() -> None:
    menu = build_candidate_menu(_lock(), "0")
    raw = {
        source: {17: float(index), 42: float(index), 101: float(index)}
        for index, source in enumerate(menu.candidate_sources)
    }

    missing = route_raw_proxy_evidence_or_exact_b(
        menu, {source: value for source, value in raw.items() if source != "1"}
    )
    assert isinstance(missing, ExactBFallback)
    assert missing.reason == "UNSUPPORTED_EVIDENCE"
    assert missing.winner_sources == ()

    raw["1"][42] = float("nan")
    nonfinite = route_raw_proxy_evidence_or_exact_b(menu, raw)
    assert isinstance(nonfinite, ExactBFallback)
    assert nonfinite.reason == "NONFINITE_EVIDENCE"
    assert nonfinite.winner_sources == ()


def test_control_validates_candidate_1024_and_exact_b_eight_by_128() -> None:
    lock = _lock()
    menu = build_candidate_menu(lock, "0")
    receipt = validate_candidate_and_b_control(lock, menu)
    assert receipt.candidate_budget_per_class == 1024
    assert receipt.b_source_budget_per_class == 128
    assert receipt.b_source_count == 8
    assert receipt.b_total_per_class == 1024
    assert len(receipt.b_replicate_ids) == 9
    assert receipt.to_payload()["generation_performed"] is False


@pytest.mark.parametrize(
    ("key", "value"),
    (("max_source_block_per_class", 512), ("equal_union_source_budget_per_class", 64)),
)
def test_control_rejects_candidate_or_b_budget_drift(key: str, value: int) -> None:
    payload = _generation_lock_payload()
    payload["generation"][key] = value
    payload["generation_lock_hash"] = stable_hash(
        {name: row for name, row in payload.items() if name != "generation_lock_hash"}
    )
    lock = GenerationLock(payload)
    menu = build_candidate_menu(lock, "0")
    with pytest.raises(ProtocolError, match="budget drifted"):
        validate_candidate_and_b_control(lock, menu)


def test_semantic_replay_reconstructs_menu_ranking_decision_and_control() -> None:
    lock = _lock()
    menu = build_candidate_menu(lock, "0")
    scores = _scores(menu)
    ranking = rank_family_proxy_scores(menu, scores)
    decision = route_unique_winner_or_exact_b(menu, ranking)
    control = validate_candidate_and_b_control(lock, menu)
    replay = replay_semantic_contract(
        generation_lock=lock,
        candidate_menu=menu,
        family_scores=scores,
        ranking=ranking,
        decision=decision,
        control_receipt=control,
    )
    assert replay.to_payload()["status"] == "PASS"
    assert replay.ranking_hash == ranking.ranking_hash


def test_package_source_fence_excludes_diagnostics_source_inner_and_dynamic_imports() -> None:
    package_root = (
        Path(__file__).resolve().parents[2]
        / "src/midogpp_thesis/cvae/routing/sceptre"
    )
    source_by_module = {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(package_root.glob("*.py"))
    }
    assert set(assert_import_source_fence(source_by_module)) == set(source_by_module)
    with pytest.raises(ProtocolError, match="import fence"):
        assert_import_source_fence(
            {"poison.py": "from midogpp_thesis.cvae.diagnostics import cli\n"}
        )
    with pytest.raises(ProtocolError, match="import fence"):
        assert_import_source_fence(
            {
                "poison.py": (
                    "from midogpp_thesis.cvae.routing.source_inner_utility "
                    "import contracts\n"
                )
            }
        )
    with pytest.raises(ProtocolError, match="import fence"):
        assert_import_source_fence(
            {
                "poison.py": (
                    "from midogpp_thesis.cvae.routing import source_inner_utility\n"
                )
            }
        )
    with pytest.raises(ProtocolError, match="dynamically"):
        assert_import_source_fence({"poison.py": "__import__('forbidden')\n"})
