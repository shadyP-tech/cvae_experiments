"""Source-only development-surface construction for HARP v5.

This module owns the only development-label join in the production pipeline.
It consumes labels only after the runner has sealed the label-free physical and
compatibility artifacts, and it rejects any label outside that sealed universe.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...routing.compatibility_conditioned_directional_router import (
    SourceActionObservation,
)
from ...routing.harp_protocol import canonical_hash
from .compatibility_adapter import (
    CompatibilityAdapterState,
    compatibility_state_from_artifact,
)
from .contracts import ActionKind, ArtifactValue, LabelFreeOuterMenu
from .directional_surfaces import build_source_directional_observations
from .production_validation import receipts_for_pool


CompatibilityLoader = Callable[[ArtifactValue], CompatibilityAdapterState]


def build_source_development_artifact(
    menus: Sequence[LabelFreeOuterMenu],
    compatibility: ArtifactValue,
    development_labels: object,
    *,
    config: object,
    compatibility_loader: CompatibilityLoader = compatibility_state_from_artifact,
) -> ArtifactValue:
    """Join sealed development labels and serialize the source-only surface."""

    menu_rows = tuple(menus)
    state = compatibility_loader(compatibility)
    label_rows = tuple(development_labels)  # type: ignore[arg-type]
    try:
        label_index = {
            (str(row.center), str(row.case_id), str(row.sample_id)): int(row.label)
            for row in label_rows
        }
    except (AttributeError, TypeError, ValueError) as exc:
        raise ProtocolError("HARP v5 source-development labels are malformed.") from exc
    if (
        len(label_index) != len(label_rows)
        or any(value not in (0, 1) for value in label_index.values())
    ):
        raise ProtocolError("HARP v5 source-development label identities drifted.")

    observations: list[SourceActionObservation] = []
    expected_label_keys: set[tuple[str, str, str]] = set()
    for menu in menu_rows:
        queries = tuple(
            sorted(
                {
                    block.query_center_id
                    for block in menu.blocks
                    if block.surface_role == "development"
                }
            )
        )
        for query in queries:
            pool = state.pool(menu.outer_target_id, query)
            baseline = tuple(
                block
                for block in menu.blocks
                if block.surface_role == "development"
                and block.query_center_id == query
                and block.action_kind is ActionKind.B
            )
            if len(baseline) != 1:
                raise ProtocolError("HARP v5 development context lacks exact B.")
            scoped_keys = {
                (case, sample)
                for case, sample in zip(
                    baseline[0].case_ids, baseline[0].sample_ids, strict=True
                )
            }
            expected_label_keys.update(
                (query, case, sample) for case, sample in scoped_keys
            )
            try:
                scoped_labels = {
                    (case, sample): label_index[(query, case, sample)]
                    for case, sample in scoped_keys
                }
            except KeyError as exc:
                raise ProtocolError(
                    "HARP v5 development labels do not cover the sealed source menu."
                ) from exc
            observations.extend(
                build_source_directional_observations(
                    menu,
                    candidate_pool=pool,
                    compatibility_receipts=receipts_for_pool(
                        state, menu.outer_target_id, query
                    ),
                    source_labels=scoped_labels,
                )
            )
    if set(label_index) != expected_label_keys:
        raise ProtocolError(
            "HARP v5 source-development labels exceed or omit the sealed menu universe."
        )
    ordered = tuple(
        sorted(
            observations,
            key=lambda row: (
                row.feature.outer_target_id,
                row.feature.query_center_id,
                row.feature.case_id,
                row.feature.action_id,
            ),
        )
    )
    if not ordered or len({row.source_response_hash for row in ordered}) != len(
        ordered
    ):
        raise ProtocolError("HARP v5 source-development response inventory drifted.")
    names = ordered[0].feature.feature_names
    if any(row.feature.feature_names != names for row in ordered):
        raise ProtocolError("HARP v5 source feature schema differs across actions.")
    body = {
        "schema_version": "midogpp_harp_v5_source_development_case_surface_v1",
        "config_hash": getattr(config, "config_hash"),
        "outer_targets": list(getattr(config, "protocol")["centers"]),
        "observation_count": len(ordered),
        "feature_names": list(names),
        "source_response_hashes": [row.source_response_hash for row in ordered],
        "rows": [
            {
                "outer_target_id": row.feature.outer_target_id,
                "query_center_id": row.feature.query_center_id,
                "case_id": row.feature.case_id,
                "action_id": row.feature.action_id,
                "action_kind": row.feature.action_kind.value,
                "direction": row.feature.direction.value,
                "candidate_source_id": row.feature.candidate_source_id,
                "candidate_pool_hash": row.candidate_pool.pool_hash,
                "feature_hash": row.feature.feature_hash,
                "source_response_hash": row.source_response_hash,
            }
            for row in ordered
        ],
        "strict_outer_H_query_candidate_exclusion": True,
        "response_scope": "SOURCE_DEVELOPMENT_ONLY",
        "evaluation_labels_used": False,
    }
    return ArtifactValue(
        state=ordered,
        manifest={**body, "surface_hash": canonical_hash(body)},
        arrays={
            "feature_values": np.asarray(
                [row.feature.feature_values for row in ordered], dtype=np.float64
            ),
            "endpoint_effects": np.asarray(
                [row.effects.as_tuple() for row in ordered], dtype=np.float64
            ),
        },
    )


__all__ = ("build_source_development_artifact",)
