from __future__ import annotations

import hashlib
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.exact_tail_utility_surface.contracts import (
    expected_ensemble_endpoint_keys,
    expected_utility_keys,
)
from midogpp_thesis.cvae.routing.exact_tail_utility_surface.ensemble_scoring import (
    ENSEMBLE_ENDPOINT_LOCK_SCHEMA,
    ExactTailEnsembleEndpointLock,
)
from midogpp_thesis.cvae.routing.exact_tail_utility_surface.support_shift_surface import (
    SUPPORT_SHIFT_LOCK_SCHEMA,
    SUPPORT_SHIFT_ROW_ROLE,
    SUPPORT_SHIFT_ROW_SCALAR_SEMANTICS,
    ExactTailSupportActionShiftLock,
    ExactTailSupportActionShiftRow,
)
from midogpp_thesis.cvae.routing.residual_topup.hashing import canonical_sha256
from midogpp_thesis.cvae.routing.utility_aligned import (
    ENSEMBLE_SEED_KEYS,
    SUPPORT_ACTION_PROBABILITY_SHIFT_NAME,
    SUPPORT_ACTION_PROBABILITY_SHIFT_SCHEMA,
    SUPPORT_ACTION_PROBABILITY_SHIFT_SEMANTICS,
    SUPPORT_ACTION_TECHNICAL_SEED_SPREAD_SEMANTICS,
    build_case_bootstrap_plan,
)
from midogpp_thesis.cvae.routing.utility_aligned_identities import CENTERS
from midogpp_thesis.cvae.routing.utility_aligned_residual_policy.ensemble_inputs import (
    _load_support_shifts,
)
from midogpp_thesis.cvae.routing.utility_aligned_residual_policy.ensemble_model_adapter import (
    make_endpoint_worker_payload,
)
from midogpp_thesis.cvae.routing.utility_aligned_residual_policy import exact_inputs
from midogpp_thesis.cvae.routing.utility_aligned_residual_policy.target_inputs import (
    _load_action_shift_lock,
)
from midogpp_thesis.cvae.routing.utility_aligned_target_support_surface.action_probe_contracts import (
    TargetSupportActionShiftRow,
    workstation_action_probe_runtime,
)
from midogpp_thesis.cvae.routing.utility_aligned_target_support_surface.action_probe_surface import (
    build_action_shift_lock,
)
from midogpp_thesis.cvae.routing.utility_aligned_target_support_surface.artifact_writer import (
    write_csv,
    write_json,
)


def _sha(value: object) -> str:
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


def _source_endpoint_lock() -> ExactTailEnsembleEndpointLock:
    values: dict[str, object] = {
        "config_contract_hash": "1" * 16,
        "prediction_seal_hash": "2" * 16,
        "prediction_index_sha256": "3" * 64,
        "prediction_arrays_sha256": "4" * 64,
        "probability_cell_surface_hash": "5" * 16,
        "endpoint_keys_hash": stable_hash(
            [list(key) for key in expected_ensemble_endpoint_keys()]
        ),
        "endpoint_row_hashes_hash": "6" * 16,
        "endpoint_table_sha256": "7" * 64,
        "endpoint_row_count": len(expected_ensemble_endpoint_keys()),
        "endpoint_lock_hash": "",
        "schema_version": ENSEMBLE_ENDPOINT_LOCK_SCHEMA,
    }
    provisional = ExactTailEnsembleEndpointLock.__new__(
        ExactTailEnsembleEndpointLock
    )
    for key, value in values.items():
        object.__setattr__(provisional, key, value)
    values["endpoint_lock_hash"] = stable_hash(provisional._unhashed_payload())
    return ExactTailEnsembleEndpointLock(**values)  # type: ignore[arg-type]


def _seed_vector_hash(
    *, seed_key: tuple[int, int], row_hash: str, row_count: int, probability_sha: str
) -> str:
    return canonical_sha256(
        {
            "schema_version": "midogpp_utility_aligned_seed_probability_vector_v1",
            "training_seed": seed_key[0],
            "generation_seed": seed_key[1],
            "row_identity_hash": row_hash,
            "prediction_provenance_hash": probability_sha,
            "row_count": row_count,
            "probability_sha256": probability_sha,
        }
    )


def _source_shift_group(
    outer: str, query: str, source: str, *, aggregate_override: str | None = None
) -> tuple[ExactTailSupportActionShiftRow, ...]:
    row_identity = stable_hash(["support", query])
    partition = stable_hash(["partition", query])
    per_seed = tuple(0.01 + 0.001 * index for index in range(9))
    base_sha = tuple(_sha((outer, query, source, seed, "base")) for seed in ENSEMBLE_SEED_KEYS)
    tail_sha = tuple(_sha((outer, query, source, seed, "tail")) for seed in ENSEMBLE_SEED_KEYS)
    base_vectors = tuple(
        _seed_vector_hash(
            seed_key=seed,
            row_hash=row_identity,
            row_count=3,
            probability_sha=probability_sha,
        )
        for seed, probability_sha in zip(ENSEMBLE_SEED_KEYS, base_sha, strict=True)
    )
    tail_vectors = tuple(
        _seed_vector_hash(
            seed_key=seed,
            row_hash=row_identity,
            row_count=3,
            probability_sha=probability_sha,
        )
        for seed, probability_sha in zip(ENSEMBLE_SEED_KEYS, tail_sha, strict=True)
    )
    values = np.asarray(per_seed, dtype=np.float64)
    sd = float(np.std(values, ddof=0, dtype=np.float64))
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    ensemble_value = 0.005
    base_ensemble_sha = _sha((outer, query, source, "base-ensemble"))
    tail_ensemble_sha = _sha((outer, query, source, "tail-ensemble"))
    difference_sha = _sha((outer, query, source, "ensemble-absolute-difference"))
    aggregate_hash = canonical_sha256(
        {
            "schema_version": SUPPORT_ACTION_PROBABILITY_SHIFT_SCHEMA,
            "row_identity_hash": row_identity,
            "seed_pair_count": 9,
            "seed_keys": [list(seed) for seed in ENSEMBLE_SEED_KEYS],
            "base_component_vector_hashes": list(base_vectors),
            "tail_component_vector_hashes": list(tail_vectors),
            "per_seed_mean_absolute_shifts": list(per_seed),
            "technical_seed_spread_semantics": (
                SUPPORT_ACTION_TECHNICAL_SEED_SPREAD_SEMANTICS
            ),
            "technical_seed_values_may_feed_model": False,
            "base_ensemble_probability_sha256": base_ensemble_sha,
            "tail_ensemble_probability_sha256": tail_ensemble_sha,
            "ensemble_absolute_difference_sha256": difference_sha,
            "value": ensemble_value,
            "seed_standard_deviation": sd,
            "seed_minimum": minimum,
            "seed_maximum": maximum,
            "seed_range": maximum - minimum,
            "scalar_name": SUPPORT_ACTION_PROBABILITY_SHIFT_NAME,
            "scalar_semantics": SUPPORT_ACTION_PROBABILITY_SHIFT_SEMANTICS,
            "labels_used": False,
        }
    )
    aggregate_hash = aggregate_override or aggregate_hash
    rows = []
    for index, (training_seed, generation_seed) in enumerate(ENSEMBLE_SEED_KEYS):
        raw: dict[str, object] = {
            "outer_target": outer,
            "pseudo_query": query,
            "candidate_source": source,
            "training_seed": training_seed,
            "generation_seed": generation_seed,
            "descriptive_seed_mean_absolute_shift": per_seed[index],
            "candidate_ensemble_mean_absolute_shift": ensemble_value,
            "support_row_count": 3,
            "support_case_count": 2,
            "support_row_hash": row_identity,
            "support_partition_hash": partition,
            "prediction_seal_hash": "2" * 16,
            "base_support_probability_sha256": base_sha[index],
            "tail_support_probability_sha256": tail_sha[index],
            "base_component_vector_hash": base_vectors[index],
            "tail_component_vector_hash": tail_vectors[index],
            "candidate_base_ensemble_probability_sha256": base_ensemble_sha,
            "candidate_tail_ensemble_probability_sha256": tail_ensemble_sha,
            "candidate_ensemble_absolute_difference_sha256": difference_sha,
            "candidate_aggregate_shift_hash": aggregate_hash,
            "shift_row_hash": "",
        }
        provisional = ExactTailSupportActionShiftRow.__new__(
            ExactTailSupportActionShiftRow
        )
        for key, value in raw.items():
            object.__setattr__(provisional, key, value)
        for key, value in {
            "scalar_name": SUPPORT_ACTION_PROBABILITY_SHIFT_NAME,
            "scalar_semantics": SUPPORT_SHIFT_ROW_SCALAR_SEMANTICS,
            "row_role": SUPPORT_SHIFT_ROW_ROLE,
            "labels_used": False,
            "support_labels_available": False,
            "target_labels_used": False,
            "seed_selection_performed": False,
        }.items():
            object.__setattr__(provisional, key, value)
        raw["shift_row_hash"] = stable_hash(provisional._unhashed_payload())
        rows.append(ExactTailSupportActionShiftRow(**raw))  # type: ignore[arg-type]
    return tuple(rows)


def _write_source_shift_surface(
    root: Path,
    *,
    aggregate_override: str | None = None,
) -> tuple[ExactTailSupportActionShiftRow, ...]:
    groups = {
        key: {
            (row.training_seed, row.generation_seed): row
            for row in _source_shift_group(
                *key,
                aggregate_override=(
                    aggregate_override
                    if key == expected_ensemble_endpoint_keys()[0]
                    else None
                ),
            )
        }
        for key in expected_ensemble_endpoint_keys()
    }
    rows = tuple(
        groups[(outer, query, source)][(training_seed, generation_seed)]
        for outer, query, source, training_seed, generation_seed
        in expected_utility_keys()
    )
    table = root / "tables/exact_tail_support_action_shifts.csv"
    write_csv(table, [row.to_payload() for row in rows])
    lock_values: dict[str, object] = {
        "config_contract_hash": "1" * 16,
        "prediction_seal_hash": "2" * 16,
        "prediction_index_sha256": "3" * 64,
        "prediction_arrays_sha256": "4" * 64,
        "support_probability_cell_surface_hash": "8" * 16,
        "shift_keys_hash": stable_hash([list(key) for key in expected_utility_keys()]),
        "shift_row_hashes_hash": stable_hash([row.shift_row_hash for row in rows]),
        "shift_table_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
        "shift_row_count": len(rows),
        "shift_lock_hash": "",
        "schema_version": SUPPORT_SHIFT_LOCK_SCHEMA,
    }
    provisional = ExactTailSupportActionShiftLock.__new__(
        ExactTailSupportActionShiftLock
    )
    for key, value in lock_values.items():
        object.__setattr__(provisional, key, value)
    lock_values["shift_lock_hash"] = stable_hash(provisional._unhashed_payload())
    lock = ExactTailSupportActionShiftLock(**lock_values)  # type: ignore[arg-type]
    write_json(
        root / "manifests/exact_tail_support_action_shifts_lock.json",
        lock.to_payload(),
    )
    return rows


def test_source_inner_production_loader_rebuilds_typed_vector_hashes_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    rows = _write_source_shift_surface(tmp_path)
    shifts, _lock = _load_support_shifts(
        tmp_path, endpoint_lock=_source_endpoint_lock()
    )
    first = rows[0]
    typed = shifts[first.outer_target][
        (first.outer_target, first.pseudo_query, first.candidate_source)
    ]
    assert typed.shift_hash == first.candidate_aggregate_shift_hash
    assert typed.value == first.candidate_ensemble_mean_absolute_shift
    assert typed.value != pytest.approx(
        np.mean(typed.per_seed_mean_absolute_shifts, dtype=np.float64)
    )
    assert typed.base_component_vector_hashes[0] != first.base_support_probability_sha256
    assert typed.base_component_vector_hashes[0] == first.base_component_vector_hash
    worker_payload = make_endpoint_worker_payload(
        SimpleNamespace(
            utility_surface=SimpleNamespace(
                rows=(
                    SimpleNamespace(
                        to_payload=lambda: {
                            "schema_version": "endpoint-boundary-test"
                        }
                    ),
                )
            ),
            support_shifts_by_outer={
                first.outer_target: {
                    (
                        first.outer_target,
                        first.pseudo_query,
                        first.candidate_source,
                    ): typed
                }
            },
        ),
        outer_target_id=first.outer_target,
    )
    restored_payload = pickle.loads(pickle.dumps(worker_payload))
    assert restored_payload == worker_payload
    assert restored_payload.support_shift_rows[0]["shift"][
        "technical_seed_values_may_feed_model"
    ] is False

    _write_source_shift_surface(tmp_path, aggregate_override="f" * 64)
    with pytest.raises(ProtocolError, match="aggregate hash cannot be reconstructed"):
        _load_support_shifts(tmp_path, endpoint_lock=_source_endpoint_lock())


def _target_rows(
    cases: dict[str, tuple[str, ...]], plans: dict[str, object]
) -> tuple[TargetSupportActionShiftRow, ...]:
    rows = []
    for target in CENTERS:
        plan = plans[target]
        for source in CENTERS:
            if source == target:
                continue
            for case_id in cases[target]:
                row_identity = _sha((target, case_id, "rows"))
                seed_keys = tuple(
                    (target, source, training_seed, generation_seed, case_id)
                    for training_seed, generation_seed in ENSEMBLE_SEED_KEYS
                )
                descriptive_values = tuple(
                    0.05 + 0.001 * index for index in range(9)
                )
                base_probability_hashes = tuple(
                    _sha((*key, "base")) for key in seed_keys
                )
                tail_probability_hashes = tuple(
                    _sha((*key, "tail")) for key in seed_keys
                )
                base_component_hashes = tuple(
                    _sha((*key, "base-vector")) for key in seed_keys
                )
                tail_component_hashes = tuple(
                    _sha((*key, "tail-vector")) for key in seed_keys
                )
                base_ensemble_hash = _sha((target, source, case_id, "base-ensemble"))
                tail_ensemble_hash = _sha((target, source, case_id, "tail-ensemble"))
                difference_hash = _sha((target, source, case_id, "difference"))
                ensemble_value = 0.02
                diagnostic = np.asarray(descriptive_values, dtype=np.float64)
                case_shift_hash = canonical_sha256(
                    {
                        "schema_version": SUPPORT_ACTION_PROBABILITY_SHIFT_SCHEMA,
                        "row_identity_hash": row_identity,
                        "seed_pair_count": 9,
                        "seed_keys": [list(seed) for seed in ENSEMBLE_SEED_KEYS],
                        "base_component_vector_hashes": list(base_component_hashes),
                        "tail_component_vector_hashes": list(tail_component_hashes),
                        "per_seed_mean_absolute_shifts": list(descriptive_values),
                        "technical_seed_spread_semantics": (
                            SUPPORT_ACTION_TECHNICAL_SEED_SPREAD_SEMANTICS
                        ),
                        "technical_seed_values_may_feed_model": False,
                        "base_ensemble_probability_sha256": base_ensemble_hash,
                        "tail_ensemble_probability_sha256": tail_ensemble_hash,
                        "ensemble_absolute_difference_sha256": difference_hash,
                        "value": ensemble_value,
                        "seed_standard_deviation": float(
                            np.std(diagnostic, ddof=0, dtype=np.float64)
                        ),
                        "seed_minimum": float(np.min(diagnostic)),
                        "seed_maximum": float(np.max(diagnostic)),
                        "seed_range": float(
                            np.max(diagnostic) - np.min(diagnostic)
                        ),
                        "scalar_name": SUPPORT_ACTION_PROBABILITY_SHIFT_NAME,
                        "scalar_semantics": SUPPORT_ACTION_PROBABILITY_SHIFT_SEMANTICS,
                        "labels_used": False,
                    }
                )
                for index, (training_seed, generation_seed) in enumerate(
                    ENSEMBLE_SEED_KEYS
                ):
                    rows.append(
                        TargetSupportActionShiftRow(
                            outer_target_id=target,
                            query_id=target,
                            candidate_source=source,
                            training_seed=training_seed,
                            generation_seed=generation_seed,
                            case_id=case_id,
                            support_partition_hash=plan.support_partition_hash,
                            case_row_identity_hash=row_identity,
                            support_row_count=3,
                            base_probability_sha256=base_probability_hashes[index],
                            tail_probability_sha256=tail_probability_hashes[index],
                            base_component_vector_hash=base_component_hashes[index],
                            tail_component_vector_hash=tail_component_hashes[index],
                            descriptive_seed_mean_absolute_positive_probability_shift=(
                                descriptive_values[index]
                            ),
                            case_ensemble_mean_absolute_positive_probability_shift=(
                                ensemble_value
                            ),
                            case_base_ensemble_probability_sha256=(
                                base_ensemble_hash
                            ),
                            case_tail_ensemble_probability_sha256=(
                                tail_ensemble_hash
                            ),
                            case_ensemble_absolute_difference_sha256=(
                                difference_hash
                            ),
                            case_ensemble_shift_hash=case_shift_hash,
                        )
                    )
    return tuple(sorted(rows, key=lambda row: row.row_key))


def _write_target_shift_surface(
    root: Path,
    rows: tuple[TargetSupportActionShiftRow, ...],
) -> None:
    table = root / "tables/target_support_action_shifts.csv"
    write_csv(table, [row.to_payload() for row in rows])
    lock = build_action_shift_lock(
        rows=rows,
        table_path=table,
        support_reservation_hash="1" * 64,
        target_support_cache_binding_hash="2" * 64,
        source_generation_lock_hash="3" * 64,
        generated_cache_hash="4" * 64,
        runtime=workstation_action_probe_runtime(),
    )
    write_json(root / "manifests/target_support_action_shifts_lock.json", lock)


def test_target_production_loader_roundtrip_and_complete_case_grid_tamper(
    tmp_path: Path,
) -> None:
    cases = {
        target: tuple(f"{target}-case-{index}" for index in range(8))
        for target in CENTERS
    }
    plans = {
        target: build_case_bootstrap_plan(
            target_id=target,
            support_case_ids=cases[target],
            replicate_count=32,
        )
        for target in CENTERS
    }
    feature_sets = {
        target: SimpleNamespace(plan=plans[target]) for target in CENTERS
    }
    rows = _target_rows(cases, plans)
    _write_target_shift_surface(tmp_path, rows)

    bindings, grouped = _load_action_shift_lock(
        tmp_path,
        support_case_ids_by_target=cases,
        feature_sets=feature_sets,
    )
    assert bindings["target_support_action_shift_row_count"] == 5184
    assert bindings[
        "target_support_action_shift_case_ensemble_group_count"
    ] == 576
    assert bindings[
        "target_support_action_shift_descriptive_seed_values_may_feed_model"
    ] is False
    assert all(len(grouped[target]) == 64 for target in CENTERS)
    first_case = grouped[CENTERS[0]][0]
    assert first_case.ensemble_mean_absolute_shift == 0.02
    assert first_case.ensemble_mean_absolute_shift != pytest.approx(
        np.mean(first_case.per_seed_mean_absolute_shifts, dtype=np.float64)
    )

    removed_group = (
        rows[-1].outer_target_id,
        rows[-1].candidate_source,
        rows[-1].case_id,
    )
    tampered_rows = tuple(
        row
        for row in rows
        if (row.outer_target_id, row.candidate_source, row.case_id)
        != removed_group
    )
    _write_target_shift_surface(tmp_path, tampered_rows)
    with pytest.raises(
        ProtocolError,
        match="(?:target/source/case grid drifted|lacks an exact-nine seed grid)",
    ):
        _load_action_shift_lock(
            tmp_path,
            support_case_ids_by_target=cases,
            feature_sets=feature_sets,
        )


def test_endpoint_only_exact_input_loader_never_opens_descriptive_utility_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    partition_hashes = {
        target: stable_hash(["development-partition", target])
        for target in CENTERS
    }
    feature_rows = tuple(
        SimpleNamespace(
            row_key=key,
            row_hash=stable_hash(["feature-row", *key]),
            outer_target_id=key[0],
            query_id=key[1],
            support_partition_hash=partition_hashes[key[1]],
        )
        for key in expected_utility_keys()
    )
    lock = SimpleNamespace(
        development_manifest_sha256="d" * 64,
        reservation_index_hash="reservation-hash",
        feature_row_hashes_hash=stable_hash(
            [row.row_hash for row in feature_rows]
        ),
    )
    manifest = SimpleNamespace(
        reservation_hash="reservation-hash",
        partition_hashes_by_center=partition_hashes,
    )
    opened: list[str] = []

    monkeypatch.setattr(
        exact_inputs, "_validate_endpoint_only_bundle", lambda _root: lock
    )
    monkeypatch.setattr(
        exact_inputs, "load_development_case_manifest", lambda _root: manifest
    )
    monkeypatch.setattr(
        exact_inputs, "sha256_file", lambda _path: lock.development_manifest_sha256
    )
    monkeypatch.setattr(
        exact_inputs,
        "read_csv",
        lambda path: (opened.append(Path(path).name) or feature_rows),
    )
    monkeypatch.setattr(exact_inputs, "parse_feature_row", lambda row: row)
    monkeypatch.setattr(
        exact_inputs,
        "build_distributional_feature_surface",
        lambda rows: SimpleNamespace(row_count=len(rows)),
    )
    monkeypatch.setattr(
        exact_inputs,
        "_utility",
        lambda _root: (_ for _ in ()).throw(
            AssertionError("legacy per-seed utility parser crossed endpoint boundary")
        ),
    )

    loaded = exact_inputs.load_exact_ensemble_policy_inputs(
        SimpleNamespace(exact_tail_surface_root=tmp_path)
    )
    assert tuple(loaded.inner_feature_surfaces) == CENTERS
    assert opened == ["candidate_features.csv"]

    tampered = list(feature_rows)
    tampered[0] = SimpleNamespace(
        **{
            **vars(tampered[0]),
            "support_partition_hash": "wrong-partition",
        }
    )
    monkeypatch.setattr(exact_inputs, "read_csv", lambda _path: tuple(tampered))
    with pytest.raises(ProtocolError, match="candidate feature grid escaped"):
        exact_inputs.load_exact_ensemble_policy_inputs(
            SimpleNamespace(exact_tail_surface_root=tmp_path)
        )
