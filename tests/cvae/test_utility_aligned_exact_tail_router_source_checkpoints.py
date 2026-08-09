from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router.contracts import (
    CENTERS,
    GENERATION_SEEDS,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router.input_contracts import (
    ValidationRowIdentity,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router.source_cache_planning import (
    write_support_scratch,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router import (
    source_cache_worker as worker,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router.source_cache_store import (
    atomic_write_json,
    read_json,
)
from midogpp_thesis.cvae.generation.contracts import SourceGenerationKey
from midogpp_thesis.cvae.protocol import ProtocolError


def test_source_checkpoint_restores_typed_seed_keys_after_json_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _source_task(tmp_path)
    monkeypatch.setattr(worker, "COMMON_OUTPUT_DIM", 4)
    monkeypatch.setattr(
        worker,
        "load_routing_authorized_expert",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(worker, "generate_source_block", _generated_block)
    monkeypatch.setattr(
        worker,
        "score_variational_compatibility",
        _compatibility_score,
    )

    written = worker.generate_source_task(task)
    serialized = read_json(Path(str(task["checkpoint_path"])))
    mmd = serialized["component_records"][0][  # type: ignore[index]
        "linear_kernel_mmd2_by_generation_seed"
    ]
    assert set(mmd) == {str(seed) for seed in GENERATION_SEEDS}  # type: ignore[arg-type]

    resumed = worker.load_generation_checkpoint(
        Path(str(task["checkpoint_path"])), task=task
    )
    assert resumed["checkpoint_hash"] == written["checkpoint_hash"]

    first_record = serialized["component_records"][0]  # type: ignore[index]
    raw_mmd = first_record["linear_kernel_mmd2_by_generation_seed"]
    raw_mmd["017"] = raw_mmd.pop("17")  # type: ignore[index,union-attr]
    atomic_write_json(Path(str(task["checkpoint_path"])), serialized)
    with pytest.raises(ProtocolError, match="MMD seed is not canonical"):
        worker.load_generation_checkpoint(
            Path(str(task["checkpoint_path"])), task=task
        )


def _source_task(tmp_path: Path) -> dict[str, object]:
    support_array_path = tmp_path / "support.npy"
    support_index_path = tmp_path / "support.json"
    support = write_support_scratch(
        support_array_path,
        support_index_path,
        frame=_SupportFrame(),
        partitions=_support_partitions(),
    )
    generation_keys = tuple(
        SourceGenerationKey(
            source_center="0",
            training_seed=17,
            generation_seed=seed,
            expert_lock_hash="expert-lock",
            stream_id=f"stream::{seed}",
            class_seed_by_label={"0": seed, "1": seed + 1},
        )
        for seed in GENERATION_SEEDS
    )
    return {
        "schema_version": "midogpp_stage90_utility_aligned_source_task_v1",
        "task_ordinal": 0,
        "source_center": "0",
        "training_seed": 17,
        "generation_keys": generation_keys,
        "device": "cpu",
        "expert_bank_root": str(tmp_path / "bank"),
        "support_array_path": str(support_array_path),
        "support_index_path": str(support_index_path),
        "checkpoint_path": str(tmp_path / "source.json"),
        "source_array_path": str(tmp_path / "source.npy"),
        "component_array_path": str(tmp_path / "components.npy"),
        "config_contract_hash": "config-contract",
        "generation_lock_hash": "generation-lock",
        "support_scratch_hash": str(support["support_scratch_hash"]),
        "labels_available": False,
        "amp_enabled": False,
    }


class _SupportFrame:
    cache_binding_hash = "cache-binding"

    def embeddings_for(
        self, rows: tuple[ValidationRowIdentity, ...]
    ) -> np.ndarray:
        return np.asarray(
            [[float(row.row_ordinal + column) for column in range(4)] for row in rows],
            dtype=np.float32,
        )


def _support_partitions() -> SimpleNamespace:
    rows_by_center: dict[str, tuple[ValidationRowIdentity, ...]] = {}
    ordinal = 0
    for center in CENTERS:
        rows = []
        for local in range(2):
            rows.append(
                ValidationRowIdentity(
                    row_ordinal=ordinal,
                    manifest_row_index=ordinal,
                    sample_id=f"sample::{center}::{local}",
                    case_id=f"case::{center}::{local}",
                    center=center,
                    partition_role="support",
                )
            )
            ordinal += 1
        rows_by_center[center] = tuple(rows)
    return SimpleNamespace(
        support_rows_by_center=rows_by_center,
        lock_hash="partition-lock",
    )


def _generated_block(
    _expert: object,
    key: SourceGenerationKey,
    *,
    per_class: int,
    device: str,
) -> SimpleNamespace:
    del device
    value = float(GENERATION_SEEDS.index(key.generation_seed) + 1)
    embeddings = np.full((2 * per_class, 4), value, dtype=np.float32)
    labels = np.concatenate(
        (
            np.zeros(per_class, dtype=np.int64),
            np.ones(per_class, dtype=np.int64),
        )
    )
    return SimpleNamespace(
        embeddings=embeddings,
        labels=labels,
        output_sha256=_array_bundle_sha256(embeddings, labels),
    )


def _compatibility_score(
    _expert: object,
    values: np.ndarray,
    _cases: tuple[str, ...],
) -> SimpleNamespace:
    count = len(values)
    return SimpleNamespace(
        per_class_reconstruction_mse={
            0: np.full(count, 0.1, dtype=np.float32),
            1: np.full(count, 0.2, dtype=np.float32),
        },
        per_class_normalized_ps_kl={
            0: np.full(count, 0.3, dtype=np.float32),
            1: np.full(count, 0.4, dtype=np.float32),
        },
        case_equal_mean=0.25,
    )


def _array_bundle_sha256(embeddings: np.ndarray, labels: np.ndarray) -> str:
    digest = hashlib.sha256()
    for values in (embeddings, labels):
        array = np.ascontiguousarray(values)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()
