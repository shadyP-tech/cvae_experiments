from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.cvae.diagnostics.local_marginal_utility_router import execution
from midogpp_thesis.cvae.diagnostics.local_marginal_utility_router.contracts import (
    CENTERS,
    EXPECTED_DEVELOPMENT_CLASSIFIER_FIT_COUNT,
    ValidationRowIdentity,
)
from midogpp_thesis.cvae.diagnostics.local_marginal_utility_router.execution import (
    LabelFreeValidationFrame,
    PartitionSurface,
)


def _frame_and_partitions() -> tuple[LabelFreeValidationFrame, PartitionSurface]:
    rows: list[ValidationRowIdentity] = []
    rows_by_center: dict[str, tuple[ValidationRowIdentity, ...]] = {}
    ordinal = 0
    for center in CENTERS:
        center_rows = tuple(
            ValidationRowIdentity(
                row_ordinal=ordinal + local,
                manifest_row_index=ordinal + local,
                sample_id=f"sample-{center}-{local}",
                case_id=f"case-{center}-{local}",
                center=center,
                partition_role="evaluation",
            )
            for local in range(2)
        )
        rows.extend(center_rows)
        rows_by_center[center] = center_rows
        ordinal += 2
    frame = LabelFreeValidationFrame(
        embeddings=np.zeros((len(rows), 3840), dtype=np.float32),
        rows=tuple(rows),
        rows_by_center=rows_by_center,
        cache_binding={
            "schema_version": "test",
            "labels_persisted": False,
            "manifest_opened": False,
        },
    )
    partitions = PartitionSurface(
        support_rows_by_center=rows_by_center,
        evaluation_rows_by_center=rows_by_center,
        table_rows=(),
        lock_payload={"support_partition_lock_hash": stable_hash("partition")},
    )
    return frame, partitions


def test_materialization_writes_complete_label_free_perturbation_schema(
    monkeypatch,
) -> None:
    frame, partitions = _frame_and_partitions()
    config = SimpleNamespace(
        generation_device="cpu",
        classifier=SimpleNamespace(config_hash=stable_hash("classifier")),
    )
    generation_lock = SimpleNamespace(generation_lock_hash=stable_hash("generation"))
    counters = {"composition": 0, "fit": 0}

    monkeypatch.setattr(execution, "_generation_key_map", lambda _lock: {})
    monkeypatch.setattr(
        execution,
        "_generate_seed_cell_blocks",
        lambda *_args, **_kwargs: {source: object() for source in CENTERS},
    )

    def fake_compose(
        source_blocks,
        allocation_per_class,
        *,
        shuffle_seed_by_class,
        total_per_class,
    ):
        counters["composition"] += 1
        assert set(source_blocks) == set(allocation_per_class)
        assert sum(allocation_per_class.values()) == total_per_class == 1008
        return SimpleNamespace(
            embeddings=np.zeros((2, 4), dtype=np.float32),
            labels=np.asarray([0, 1], dtype=np.uint8),
            composition_hash=stable_hash(
                {
                    "sources": sorted(source_blocks),
                    "allocation": dict(allocation_per_class),
                    "shuffle": dict(shuffle_seed_by_class),
                }
            ),
        )

    def fake_fit(_config, _train_x, _train_y, eval_x):
        counters["fit"] += 1
        assert len(eval_x) == 2
        return {
            "predictions": np.asarray([0, 1], dtype=np.uint8),
            "probabilities": np.asarray([0.1, 0.9], dtype=np.float32),
            "classes": (0, 1),
            "n_iter": (1,),
            "converged": True,
            "classifier_config_hash": config.classifier.config_hash,
            "scaler_state_hash": stable_hash("scaler"),
        }

    monkeypatch.setattr(execution, "compose_prefix_blocks", fake_compose)
    monkeypatch.setattr(execution, "_fit_classifier", fake_fit)

    surface = execution.materialize_development_predictions(
        config,
        generation_lock,
        frame,
        partitions,
    )

    assert len(surface.store.index_rows) == EXPECTED_DEVELOPMENT_CLASSIFIER_FIT_COUNT
    assert counters == {
        "composition": EXPECTED_DEVELOPMENT_CLASSIFIER_FIT_COUNT,
        "fit": EXPECTED_DEVELOPMENT_CLASSIFIER_FIT_COUNT,
    }
    assert all(
        row["phase"] == "development_utility_surface"
        and row["labels_available_to_fit_or_predict"] is False
        and row["seed_selection_performed"] is False
        for row in surface.store.index_rows
    )
    assert {
        row["arm_role"] for row in surface.store.index_rows
    } == {"control", "source_perturbation"}
