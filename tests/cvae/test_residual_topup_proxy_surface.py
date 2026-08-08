from __future__ import annotations

from collections import Counter
from dataclasses import replace
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.dense_residual_soft_router.compatibility import (
    CLASS_PRIOR as COMPATIBILITY_CLASS_PRIOR,
    ENERGY_SEMANTICS as COMPATIBILITY_ENERGY_SEMANTICS,
)
from midogpp_thesis.cvae.routing.residual_topup_policy import io as io_module
from midogpp_thesis.cvae.routing.residual_topup_policy.config import (
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_EQUAL_UNION_POLICY_LOCK_HASH,
    EXPECTED_GENERATION_LOCK_HASH,
    load_residual_topup_policy_lock_config,
)
from midogpp_thesis.cvae.routing.residual_topup_policy.contracts import (
    FIXED_TRAINING_SEEDS,
    GLOBAL_PSEUDOQUERY_ROLE,
    TARGET_SUPPORT_ROLE,
)
from midogpp_thesis.cvae.routing.residual_topup_policy.proxy_surface import (
    CENTERS,
    COMMON_FEATURE_DIM,
    EXPECTED_EXPERT_TASK_COUNT,
    EXPECTED_QUERY_SHARD_COUNT,
    SCORE_CHUNK_ROWS,
    FreshProxyScoreSurface,
    FreshQueryShard,
    build_fresh_proxy_score_surface,
    build_fresh_proxy_score_tasks,
    execute_fresh_proxy_score_task,
    make_fresh_query_shard,
    materialize_fresh_proxy_inputs,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/60_routing_and_composition/configs"
    / "uniform_b_v2_residual_topup_b_u_g_s_policy_lock_v1.yaml"
)


class _FakeRuntime:
    def __init__(self) -> None:
        self.arrays: dict[Path, np.ndarray] = {}
        self.expert_loads: Counter[tuple[str, int]] = Counter()
        self.score_calls: Counter[tuple[str, int]] = Counter()
        self.executed_devices: list[str] = []

    def array_loader(self, path: Path) -> np.ndarray:
        return self.arrays[path]

    def expert_loader(
        self,
        _root: Path,
        *,
        source_center: str,
        training_seed: int,
        device: str,
    ) -> object:
        self.expert_loads[(source_center, training_seed)] += 1
        return SimpleNamespace(
            source_center=source_center,
            training_seed=training_seed,
            device=device,
            expert_lock_hash=f"expert::{source_center}::{training_seed}",
            checkpoint_hash=f"checkpoint::{source_center}::{training_seed}",
        )

    def scorer(
        self,
        expert: object,
        embeddings: np.ndarray,
        _case_ids: tuple[str, ...],
    ) -> object:
        key = (str(expert.source_center), int(expert.training_seed))
        self.score_calls[key] += 1
        source_offset = float(CENTERS.index(key[0]))
        seed_offset = float(FIXED_TRAINING_SEEDS.index(key[1])) / 10.0
        return SimpleNamespace(
            source_center=key[0],
            training_seed=key[1],
            per_row=np.asarray(embeddings[:, 0], dtype=np.float64)
            + source_offset
            + seed_offset,
            exact_nelbo=False,
            labels_consumed=False,
            energy_semantics=COMPATIBILITY_ENERGY_SEMANTICS,
            class_prior=COMPATIBILITY_CLASS_PRIOR,
        )

    def executor(self, tasks, worker):
        results = []
        for task in tasks:
            self.executed_devices.append(task.device)
            results.append(worker(task))
        return tuple(results)


def _make_shards(
    tmp_path: Path,
    runtime: _FakeRuntime,
    *,
    large_query: str | None = None,
) -> tuple[FreshQueryShard, ...]:
    shards: list[FreshQueryShard] = []
    for target in CENTERS:
        for query in CENTERS:
            if query == target:
                continue
            if query == large_query:
                case_ids = tuple(
                    [f"g::{query}::a"] * 1500
                    + [f"g::{query}::b"] * 549
                )
                values = np.zeros(
                    (len(case_ids), COMMON_FEATURE_DIM), dtype=np.float32
                )
                values[:, 0] = np.arange(len(case_ids), dtype=np.float32)
            else:
                case_ids = (
                    f"g::{query}::a",
                    f"g::{query}::a",
                    f"g::{query}::b",
                    f"g::{query}::b",
                )
                values = np.zeros((4, COMMON_FEATURE_DIM), dtype=np.float32)
                values[:, 0] = np.asarray([0.0, 2.0, 4.0, 6.0])
            path = (tmp_path / f"g_{query}.npy").resolve()
            runtime.arrays[path] = values
            shards.append(
                make_fresh_query_shard(
                    outer_target=target,
                    query_role=GLOBAL_PSEUDOQUERY_ROLE,
                    query_center=query,
                    embedding_path=path,
                    case_ids=case_ids,
                    evaluation_case_ids=(f"eval::{target}::a", f"eval::{target}::b"),
                    array_loader=runtime.array_loader,
                )
            )
        support_path = (tmp_path / f"s_{target}.npy").resolve()
        runtime.arrays[support_path] = np.zeros(
            (4, COMMON_FEATURE_DIM), dtype=np.float32
        )
        runtime.arrays[support_path][:, 0] = np.asarray([1.0, 3.0, 5.0, 7.0])
        shards.append(
            make_fresh_query_shard(
                outer_target=target,
                query_role=TARGET_SUPPORT_ROLE,
                query_center=target,
                embedding_path=support_path,
                case_ids=(
                    f"s::{target}::a",
                    f"s::{target}::a",
                    f"s::{target}::b",
                    f"s::{target}::b",
                ),
                evaluation_case_ids=(f"eval::{target}::a", f"eval::{target}::b"),
                array_loader=runtime.array_loader,
            )
        )
    return tuple(shards)


def _run_surface(
    tmp_path: Path,
    runtime: _FakeRuntime,
    shards: tuple[FreshQueryShard, ...],
) -> FreshProxyScoreSurface:
    return build_fresh_proxy_score_surface(
        shards,
        expert_bank_root=tmp_path / "expert-bank",
        expert_bank_binding_hash=EXPECTED_BANK_LOCK_HASH,
        checkpoint_root=tmp_path / "checkpoints",
        devices=("fake:0", "fake:1"),
        array_loader=runtime.array_loader,
        expert_loader=runtime.expert_loader,
        scorer=runtime.scorer,
        executor=runtime.executor,
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_query_shards_and_tasks_are_exact_hash_bound_and_leave_h_q_out(
    tmp_path: Path,
) -> None:
    runtime = _FakeRuntime()
    shards = _make_shards(tmp_path, runtime)
    assert len(shards) == EXPECTED_QUERY_SHARD_COUNT == 81
    tasks = build_fresh_proxy_score_tasks(
        reversed(shards),
        expert_bank_root=tmp_path / "expert-bank",
        expert_bank_binding_hash=EXPECTED_BANK_LOCK_HASH,
        checkpoint_root=tmp_path / "checkpoints",
        devices=("fake:0", "fake:1"),
    )

    assert len(tasks) == EXPECTED_EXPERT_TASK_COUNT == 27
    assert [task.device for task in tasks] == [
        ("fake:0", "fake:1")[index % 2] for index in range(27)
    ]
    assert all(len(task.shards) == 64 for task in tasks)
    for task in tasks:
        assert all(task.source_center != shard.outer_target for shard in task.shards)
        assert all(
            shard.query_role != GLOBAL_PSEUDOQUERY_ROLE
            or task.source_center != shard.query_center
            for shard in task.shards
        )
        assert task.chunk_rows == SCORE_CHUNK_ROWS

    with pytest.raises(ProtocolError, match="eight G shards"):
        build_fresh_proxy_score_tasks(
            shards[:-1],
            expert_bank_root=tmp_path / "expert-bank",
            expert_bank_binding_hash=EXPECTED_BANK_LOCK_HASH,
            checkpoint_root=tmp_path / "checkpoints",
            devices=("fake:0", "fake:1"),
        )
    with pytest.raises(ProtocolError, match="hash drifted"):
        replace(shards[0], shard_hash="0" * 16)


def test_surface_loads_each_expert_once_scores_all_legal_shards_and_resumes(
    tmp_path: Path,
) -> None:
    runtime = _FakeRuntime()
    shards = _make_shards(tmp_path, runtime)
    surface = _run_surface(tmp_path, runtime, shards)

    assert len(surface) == 3456
    assert surface.executed_task_count == 27
    assert surface.resumed_task_count == 0
    assert runtime.expert_loads == Counter(
        {
            (source, seed): 1
            for source in CENTERS
            for seed in FIXED_TRAINING_SEEDS
        }
    )
    assert set(runtime.score_calls.values()) == {16}
    assert runtime.executed_devices == [
        ("fake:0", "fake:1")[index % 2] for index in range(27)
    ]
    assert all(
        not row.labels_consumed
        and not row.evaluation_overlap
        and not row.source_expert_updated
        for row in surface
    )
    assert all(
        row.candidate_source not in {row.outer_target, row.query_center}
        for row in surface
        if row.query_role == GLOBAL_PSEUDOQUERY_ROLE
    )
    assert all(
        row.candidate_source != row.outer_target
        for row in surface
        if row.query_role == TARGET_SUPPORT_ROLE
    )
    for query in CENTERS:
        source = next(center for center in CENTERS if center != query)
        legal_targets = [
            target for target in CENTERS if target not in {query, source}
        ]
        energies = {
            next(
                row.proxy_energy
                for row in surface
                if row.outer_target == target
                and row.query_role == GLOBAL_PSEUDOQUERY_ROLE
                and row.query_center == query
                and row.case_id == f"g::{query}::a"
                and row.candidate_source == source
                and row.training_seed == FIXED_TRAINING_SEEDS[0]
            )
            for target in legal_targets
        }
        assert len(energies) == 1

    resumed_runtime = _FakeRuntime()
    resumed_runtime.arrays.update(runtime.arrays)

    def must_not_execute(_tasks, _worker):
        raise AssertionError("all hash-valid expert checkpoints should resume")

    resumed = build_fresh_proxy_score_surface(
        tuple(reversed(shards)),
        expert_bank_root=tmp_path / "expert-bank",
        expert_bank_binding_hash=EXPECTED_BANK_LOCK_HASH,
        checkpoint_root=tmp_path / "checkpoints",
        devices=("fake:0", "fake:1"),
        array_loader=resumed_runtime.array_loader,
        expert_loader=resumed_runtime.expert_loader,
        scorer=resumed_runtime.scorer,
        executor=must_not_execute,
    )
    assert resumed.rows == surface.rows
    assert resumed.executed_task_count == 0
    assert resumed.resumed_task_count == 27
    assert resumed_runtime.expert_loads == Counter()
    assert resumed_runtime.score_calls == Counter()

    checkpoint = next((tmp_path / "checkpoints").glob("*.json"))
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["records"][0]["proxy_energy"] += 1.0
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProtocolError, match="checkpoint binding"):
        build_fresh_proxy_score_surface(
            shards,
            expert_bank_root=tmp_path / "expert-bank",
            expert_bank_binding_hash=EXPECTED_BANK_LOCK_HASH,
            checkpoint_root=tmp_path / "checkpoints",
            devices=("fake:0", "fake:1"),
            array_loader=runtime.array_loader,
            expert_loader=runtime.expert_loader,
            scorer=runtime.scorer,
            executor=runtime.executor,
        )


def test_chunking_preserves_exact_case_level_means(tmp_path: Path) -> None:
    runtime = _FakeRuntime()
    target = CENTERS[0]
    query = CENTERS[1]
    shards = _make_shards(
        tmp_path,
        runtime,
        large_query=query,
    )
    source = next(center for center in CENTERS if center not in {target, query})
    tasks = build_fresh_proxy_score_tasks(
        shards,
        expert_bank_root=tmp_path / "expert-bank",
        expert_bank_binding_hash=EXPECTED_BANK_LOCK_HASH,
        checkpoint_root=tmp_path / "chunk-checkpoints",
        devices=("fake:0", "fake:1"),
    )
    task = next(
        value
        for value in tasks
        if value.source_center == source
        and value.training_seed == FIXED_TRAINING_SEEDS[0]
    )
    result = execute_fresh_proxy_score_task(
        task,
        array_loader=runtime.array_loader,
        expert_loader=runtime.expert_loader,
        scorer=runtime.scorer,
    )
    row = next(
        value
        for value in result.rows
        if value.outer_target == target
        and value.query_role == GLOBAL_PSEUDOQUERY_ROLE
        and value.query_center == query
        and value.case_id == f"g::{query}::a"
        and value.candidate_source == source
        and value.training_seed == FIXED_TRAINING_SEEDS[0]
    )
    expected = float(np.mean(np.arange(1500, dtype=np.float64)))
    expected += float(CENTERS.index(source))
    assert row.proxy_energy == pytest.approx(expected)
    # One 2,049-row canonical q requires two chunks; the other 15 unique G/S
    # queries require one call each.  H replication performs no extra scoring.
    assert runtime.score_calls[(source, FIXED_TRAINING_SEEDS[0])] == 17
    assert runtime.expert_loads[(source, FIXED_TRAINING_SEEDS[0])] == 1


def test_shards_reject_evaluation_overlap_and_embedding_drift(
    tmp_path: Path,
) -> None:
    runtime = _FakeRuntime()
    path = (tmp_path / "overlap.npy").resolve()
    runtime.arrays[path] = np.zeros((2, COMMON_FEATURE_DIM), dtype=np.float32)
    with pytest.raises(ProtocolError, match="identity/hash geometry"):
        make_fresh_query_shard(
            outer_target=CENTERS[0],
            query_role=TARGET_SUPPORT_ROLE,
            query_center=CENTERS[0],
            embedding_path=path,
            case_ids=("same", "support"),
            evaluation_case_ids=("same", "evaluation"),
            array_loader=runtime.array_loader,
        )

    shards = _make_shards(tmp_path, runtime)
    runtime.arrays[shards[0].embedding_path] = np.ones(
        (4, COMMON_FEATURE_DIM), dtype=np.float32
    )
    with pytest.raises(ProtocolError, match="bytes drifted"):
        _run_surface(tmp_path, runtime, shards)


def test_shard_grid_requires_canonical_g_and_one_evaluation_set_per_h(
    tmp_path: Path,
) -> None:
    runtime = _FakeRuntime()
    shards = list(_make_shards(tmp_path, runtime))
    original = next(
        shard
        for shard in shards
        if shard.query_role == GLOBAL_PSEUDOQUERY_ROLE
    )
    index = shards.index(original)
    mismatched_cases = (
        f"g::{original.query_center}::different-a",
        f"g::{original.query_center}::different-a",
        f"g::{original.query_center}::different-b",
        f"g::{original.query_center}::different-b",
    )
    shards[index] = make_fresh_query_shard(
        outer_target=original.outer_target,
        query_role=original.query_role,
        query_center=original.query_center,
        embedding_path=original.embedding_path,
        case_ids=mismatched_cases,
        evaluation_case_ids=original.evaluation_case_ids,
        array_loader=runtime.array_loader,
    )
    with pytest.raises(ProtocolError, match="same embedding hash.*case IDs"):
        build_fresh_proxy_score_tasks(
            shards,
            expert_bank_root=tmp_path / "expert-bank",
            expert_bank_binding_hash=EXPECTED_BANK_LOCK_HASH,
            checkpoint_root=tmp_path / "bad-g-checkpoints",
            devices=("fake:0", "fake:1"),
        )

    shards = list(_make_shards(tmp_path / "evaluation", runtime))
    original = next(
        shard
        for shard in shards
        if shard.query_role == GLOBAL_PSEUDOQUERY_ROLE
    )
    index = shards.index(original)
    shards[index] = make_fresh_query_shard(
        outer_target=original.outer_target,
        query_role=original.query_role,
        query_center=original.query_center,
        embedding_path=original.embedding_path,
        case_ids=original.case_ids,
        evaluation_case_ids=("eval::fresh::a", "eval::fresh::b"),
        array_loader=runtime.array_loader,
    )
    with pytest.raises(ProtocolError, match="same evaluation set"):
        build_fresh_proxy_score_tasks(
            shards,
            expert_bank_root=tmp_path / "expert-bank",
            expert_bank_binding_hash=EXPECTED_BANK_LOCK_HASH,
            checkpoint_root=tmp_path / "bad-eval-checkpoints",
            devices=("fake:0", "fake:1"),
        )


def test_query_shard_requires_exact_common_feature_frame(tmp_path: Path) -> None:
    runtime = _FakeRuntime()
    path = (tmp_path / "wrong-frame.npy").resolve()
    runtime.arrays[path] = np.zeros(
        (2, COMMON_FEATURE_DIM - 1), dtype=np.float32
    )
    with pytest.raises(ProtocolError, match="exact 3840-D common frame"):
        make_fresh_query_shard(
            outer_target=CENTERS[0],
            query_role=TARGET_SUPPORT_ROLE,
            query_center=CENTERS[0],
            embedding_path=path,
            case_ids=("support-a", "support-b"),
            evaluation_case_ids=("evaluation-a", "evaluation-b"),
            array_loader=runtime.array_loader,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("exact_nelbo", None, "exact_nelbo=False"),
        ("labels_consumed", None, "labels_consumed=False"),
        ("energy_semantics", "wrong", "semantics drifted"),
        ("class_prior", (0.4, 0.6), "class-prior semantics drifted"),
    ),
)
def test_scorer_requires_explicit_exact_compatibility_attestations(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    runtime = _FakeRuntime()
    shards = _make_shards(tmp_path, runtime)
    task = build_fresh_proxy_score_tasks(
        shards,
        expert_bank_root=tmp_path / "expert-bank",
        expert_bank_binding_hash=EXPECTED_BANK_LOCK_HASH,
        checkpoint_root=tmp_path / "attestation-checkpoints",
        devices=("fake:0", "fake:1"),
    )[0]

    def invalid_scorer(expert, embeddings, _case_ids):
        payload = {
            "source_center": expert.source_center,
            "training_seed": expert.training_seed,
            "per_row": np.asarray(embeddings[:, 0], dtype=np.float64),
            "exact_nelbo": False,
            "labels_consumed": False,
            "energy_semantics": COMPATIBILITY_ENERGY_SEMANTICS,
            "class_prior": COMPATIBILITY_CLASS_PRIOR,
        }
        payload[field] = value
        return SimpleNamespace(**payload)

    with pytest.raises(ProtocolError, match=message):
        execute_fresh_proxy_score_task(
            task,
            array_loader=runtime.array_loader,
            expert_loader=runtime.expert_loader,
            scorer=invalid_scorer,
        )


def test_materialized_surface_round_trips_through_stage60_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeRuntime()
    shards = _make_shards(tmp_path / "queries", runtime)
    surface = _run_surface(tmp_path, runtime, shards)
    bank_root = tmp_path / "bank"
    generation_root = tmp_path / "generation"
    equal_root = tmp_path / "equal"
    proxy_root = tmp_path / "proxy"
    bank_path = bank_root / "manifests/expert_bank_index.json"
    generation_path = generation_root / "manifests/generation_lock.json"
    equal_path = equal_root / "manifests/policy_lock.json"
    _write_json(bank_path, {"bank_lock_hash": EXPECTED_BANK_LOCK_HASH})
    _write_json(
        generation_path,
        {"generation_lock_hash": EXPECTED_GENERATION_LOCK_HASH},
    )
    _write_json(
        equal_path,
        {"policy_lock_hash": EXPECTED_EQUAL_UNION_POLICY_LOCK_HASH},
    )
    base_config = load_residual_topup_policy_lock_config(CONFIG)
    config = replace(
        base_config,
        artifact_root=tmp_path / "output",
        expert_bank_root=bank_root,
        generation_lock_root=generation_root,
        equal_union_policy_root=equal_root,
        proxy_surface_root=proxy_root,
        proxy_score_table_path=proxy_root / "tables/proxy_scores.csv",
        proxy_attestation_path=(
            proxy_root / "manifests/fresh_surface_attestation.json"
        ),
    )
    materialized = materialize_fresh_proxy_inputs(
        surface,
        shards=shards,
        config=config,
        reservation_id="fresh-reservation-proxy-surface-test-v1",
    )
    attestation_payload = json.loads(
        materialized.proxy_attestation_path.read_text(encoding="utf-8")
    )
    assert attestation_payload["proxy_surface_hash"] == surface.surface_hash
    assert attestation_payload["query_shard_hashes"] == {
        "::".join(
            (shard.outer_target, shard.query_role, shard.query_center)
        ): shard.shard_hash
        for shard in sorted(
            shards,
            key=lambda shard: (
                CENTERS.index(shard.outer_target),
                0 if shard.query_role == GLOBAL_PSEUDOQUERY_ROLE else 1,
                CENTERS.index(shard.query_center),
            ),
        )
    }
    assert {
        path.relative_to(proxy_root)
        for path in proxy_root.rglob("*")
        if path.is_file()
    } == {
        Path("tables/proxy_scores.csv"),
        Path("manifests/fresh_surface_attestation.json"),
    }

    monkeypatch.setattr(
        io_module,
        "read_generation_lock",
        lambda _path: SimpleNamespace(
            generation_lock_hash=EXPECTED_GENERATION_LOCK_HASH
        ),
    )
    monkeypatch.setattr(
        io_module,
        "read_equal_union_policy_lock",
        lambda _path: SimpleNamespace(
            policy_lock_hash=EXPECTED_EQUAL_UNION_POLICY_LOCK_HASH
        ),
    )
    loaded = io_module.load_validated_fresh_proxy_inputs(config)
    assert loaded.rows == surface.rows
    assert loaded.proxy_score_table_sha256 == materialized.proxy_score_table_sha256
    assert (
        loaded.attestation_file_sha256
        == materialized.proxy_attestation_sha256
    )
    assert loaded.attestation.attestation_hash == materialized.attestation_hash
    assert loaded.attestation.proxy_surface_hash == surface.surface_hash
    assert dict(loaded.attestation.query_shard_hashes) == attestation_payload[
        "query_shard_hashes"
    ]
    assert loaded.attestation.pseudoquery_case_ids_by_center == {
        center: (f"g::{center}::a", f"g::{center}::b")
        for center in CENTERS
    }


def test_proxy_surface_has_no_stage90_import_or_label_utility_api() -> None:
    import inspect
    import midogpp_thesis.cvae.routing.residual_topup_policy.proxy_surface as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "diagnostics." not in source
    tree = ast.parse(source)
    imported_modules = {
        name.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for name in node.names
    }.union(
        {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
    )
    assert all("stage90" not in name.lower() for name in imported_modules)
    signature = inspect.signature(build_fresh_proxy_score_surface)
    assert "labels" not in signature.parameters
    assert "utility" not in signature.parameters
    assert "nelbo" not in signature.parameters
