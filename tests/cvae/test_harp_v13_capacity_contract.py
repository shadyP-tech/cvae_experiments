from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.harp_v13_execution.action_capacity import (
    GLOBAL_MAX_REQUIRED_PER_CLASS,
    build_action_capacity_certificate,
    enumerate_complete_action_capacity,
    validate_action_capacity,
    validate_action_capacity_certificate,
)
from midogpp_thesis.cvae.runtime.harp_v13_execution.crossfit_actions import (
    build_fold_conditioned_action_menu,
    compose_fold_conditioned_action,
)
from midogpp_thesis.cvae.runtime.harp_v13_execution.crossfit_surface import (
    fold_conditioned_physical_plan,
)
from midogpp_thesis.cvae.runtime.harp_v13_execution.physical import (
    build_physical_plan,
    build_target_only_physical_plan,
)
from midogpp_thesis.cvae.runtime.harp_v13_execution.resident_stream_contracts import (
    SOURCE_ROWS_PER_CLASS,
)
from midogpp_thesis.cvae.runtime.harp_v13_execution.progress import (
    classifier_progress_due,
)


CENTERS = ("0", "1", "2", "3", "5", "6", "7", "8", "9")


def test_complete_capacity_certificate_covers_every_target_and_hqr_action() -> None:
    rows = enumerate_complete_action_capacity()
    certificate = dict(build_action_capacity_certificate())

    assert SOURCE_ROWS_PER_CLASS == 294
    assert GLOBAL_MAX_REQUIRED_PER_CLASS == SOURCE_ROWS_PER_CLASS
    assert len(rows) == 4770
    assert len({row.action_hash for row in rows}) == len(rows)
    assert certificate["target_action_count"] == 90
    assert certificate["source_prediction_action_count"] == 648
    assert certificate["source_calibration_action_count"] == 4032
    assert certificate["enumerated_action_count"] == 4770
    assert certificate["required_rows_per_class_by_surface"] == {
        "target": 256,
        "source_prediction_seven_source": 270,
        "source_calibration_six_source": 294,
    }
    assert certificate["maximum_requirement_histogram"] == {
        "128": 9,
        "144": 81,
        "162": 72,
        "168": 504,
        "189": 504,
        "256": 72,
        "270": 504,
        "294": 3024,
    }

    target_contexts = {
        (row.outer_target_id, row.current_query_center_id)
        for row in rows
        if row.surface_kind == "target"
    }
    prediction_contexts = {
        (
            row.outer_target_id,
            row.heldout_center_id,
            row.current_query_center_id,
        )
        for row in rows
        if row.surface_kind == "source_crossfit"
        and row.heldout_center_id == row.current_query_center_id
    }
    calibration_contexts = {
        (
            row.outer_target_id,
            row.heldout_center_id,
            row.current_query_center_id,
        )
        for row in rows
        if row.surface_kind == "source_crossfit"
        and row.heldout_center_id != row.current_query_center_id
    }
    assert len(target_contexts) == 9
    assert len(prediction_contexts) == 72
    assert len(calibration_contexts) == 504
    assert Counter(row.maximum_required_per_class for row in rows)[294] == 3024


def test_capacity_certificate_is_pure_reconstructible_and_tamper_evident() -> None:
    first = dict(
        build_action_capacity_certificate(
            centers=CENTERS,
            stream_rows_per_class=294,
        )
    )
    second = dict(
        build_action_capacity_certificate(
            centers=CENTERS,
            stream_rows_per_class=294,
        )
    )
    assert first == second
    assert dict(
        validate_action_capacity_certificate(
            first,
            centers=CENTERS,
            stream_rows_per_class=294,
        )
    ) == first

    tampered = {**first, "stream_rows_per_class": 293}
    with pytest.raises(ProtocolError, match="certificate drifted"):
        validate_action_capacity_certificate(tampered)


def test_all_physical_plans_bind_the_same_capacity_certificate() -> None:
    certificate = dict(build_action_capacity_certificate())
    expected_hash = certificate["capacity_certificate_hash"]

    physical = build_physical_plan()
    target = build_target_only_physical_plan()
    crossfit = dict(fold_conditioned_physical_plan())

    assert physical["action_capacity_certificate_hash"] == expected_hash
    assert target["action_capacity_certificate_hash"] == expected_hash
    assert crossfit["action_capacity_certificate_hash"] == expected_hash
    assert physical["stream_rows_per_class"] == 294
    assert target["target_maximum_required_rows_per_class"] == 256
    assert crossfit["maximum_required_rows_per_class"] == 294


@pytest.mark.parametrize("capacity", [270, 293])
def test_complete_capacity_certificate_rejects_old_or_short_streams(
    capacity: int,
) -> None:
    with pytest.raises(
        ProtocolError,
        match=r"requires 294 rows/class.*only (270|293) are available",
    ):
        build_action_capacity_certificate(stream_rows_per_class=capacity)


def test_maximal_six_source_action_materializes_at_exact_capacity() -> None:
    actions = build_fold_conditioned_action_menu("0", "1", "2")
    selected = next(
        action for action in actions if action.selected_source_id is not None
    )
    validated = validate_action_capacity((selected,))
    assert validated[0].maximum_required_per_class == 294

    source_blocks = {
        source: {
            "embeddings": np.arange(
                2 * SOURCE_ROWS_PER_CLASS, dtype=np.float32
            ).reshape(-1, 1),
            "labels": np.concatenate(
                (
                    np.zeros(SOURCE_ROWS_PER_CLASS, dtype=np.int64),
                    np.ones(SOURCE_ROWS_PER_CLASS, dtype=np.int64),
                )
            ),
        }
        for source in selected.source_order
    }
    composed = compose_fold_conditioned_action(
        source_blocks,
        selected,
        shuffle_seed_by_class={0: 11, 1: 12},
    )
    assert composed.total_per_class == 1134
    assert composed.embeddings.shape == (2268, 1)

    short_blocks = {
        source: {
            "embeddings": block["embeddings"][:-2],
            "labels": np.concatenate(
                (
                    np.zeros(SOURCE_ROWS_PER_CLASS - 1, dtype=np.int64),
                    np.ones(SOURCE_ROWS_PER_CLASS - 1, dtype=np.int64),
                )
            ),
        }
        for source, block in source_blocks.items()
    }
    with pytest.raises(ProtocolError, match="insufficient class capacity"):
        compose_fold_conditioned_action(
            short_blocks,
            selected,
            shuffle_seed_by_class={0: 11, 1: 12},
        )


def test_classifier_progress_is_bounded_without_changing_task_count() -> None:
    reported = [
        completed
        for completed in range(1, 5184 + 1)
        if classifier_progress_due(completed, 5184)
    ]

    assert reported[0] == 1
    assert reported[-1] == 5184
    assert len(reported) <= 34
    with pytest.raises(ProtocolError, match="progress state"):
        classifier_progress_due(0, 5184)
