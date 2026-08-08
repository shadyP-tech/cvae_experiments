"""Validate frozen B/U/G/S/P/Hxe actions and expand their seed grid."""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .contracts import (
    BASE_ACTION_ID,
    BASE_BUDGET_PER_CLASS,
    BASE_PER_SOURCE,
    CENTERS,
    CORE_ACTION_IDS,
    EXPECTED_PLAN_CELL_COUNT,
    EvaluationCell,
    EvaluationPlan,
    FrozenActionPayload,
    GENERATION_SEEDS,
    GLOBAL_ACTION_ID,
    MATCHED_BUDGET_PER_CLASS,
    PERMUTATION_ACTION_ID,
    SUPPORT_ACTION_ID,
    TRAINING_SEEDS,
    UNIFORM_ACTION_ID,
    expected_action_ids,
    legal_sources,
    tail_source,
)


_COUNT_FIELDS = (
    "source_counts_by_class",
    "final_counts_by_class",
    "source_budget_by_class",
)
_SCORE_FIELDS = (
    "mean_normalized_midrank_by_source",
    "support_score_by_source",
    "support_scores_by_source",
    "rank_score_by_source",
)
_HASH_FIELDS = (
    "action_hash",
    "frozen_action_hash",
    "policy_action_hash",
    "policy_lock_hash",
    "lock_hash",
)
_FORBIDDEN_VALUE_FIELDS = frozenset(
    {
        "labels",
        "target_labels",
        "evaluation_labels",
        "y_true",
        "bacc",
        "macro_f1",
        "utility_by_source",
        "oracle_source",
        "oracle_action",
    }
)


def build_evaluation_plan(
    actions_by_target: Mapping[object, object],
    *,
    evaluation_row_ids_by_target: Mapping[object, Sequence[object]] | None = None,
) -> EvaluationPlan:
    """Build the immutable 9 target x 3 x 3 all-action evaluation plan.

    Values may be sequences of :class:`FrozenActionPayload`, mappings keyed by
    action id, or lightweight wrappers exposing the same fields.  The planner
    validates scientific geometry only; it does not import or execute Stage-60
    routing code.
    """

    if not isinstance(actions_by_target, Mapping):
        raise ProtocolError("Fresh Stage-70 frozen actions must be target keyed.")
    normalized_targets = {str(key): value for key, value in actions_by_target.items()}
    if set(normalized_targets) != set(CENTERS):
        raise ProtocolError("Fresh Stage-70 must contain all nine target actions.")

    normalized: dict[str, tuple[FrozenActionPayload, ...]] = {}
    cells: list[EvaluationCell] = []
    for target in CENTERS:
        actions = _normalize_target_actions(target, normalized_targets[target])
        _validate_target_actions(target, actions)
        by_id = {action.action_id: action for action in actions}
        ordered = tuple(by_id[action_id] for action_id in expected_action_ids(target))
        normalized[target] = ordered
        for training_seed in TRAINING_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                for action in ordered:
                    cells.append(
                        EvaluationCell(
                            target_center=target,
                            training_seed=training_seed,
                            generation_seed=generation_seed,
                            action_id=action.action_id,
                            action_hash=action.action_hash,
                        )
                    )
    if len(cells) != EXPECTED_PLAN_CELL_COUNT or len({cell.key for cell in cells}) != len(cells):
        raise ProtocolError("Fresh Stage-70 action/target/seed expansion drifted.")

    row_ids = _normalize_evaluation_rows(evaluation_row_ids_by_target)
    hash_payload = {
        "schema_version": "midogpp_residual_topup_fresh_evaluation_plan_v1",
        "targets": list(CENTERS),
        "training_seeds": list(TRAINING_SEEDS),
        "generation_seeds": list(GENERATION_SEEDS),
        "actions_by_target": {
            target: [action.to_payload() for action in normalized[target]]
            for target in CENTERS
        },
        "evaluation_row_ids_by_target": {
            target: list(row_ids.get(target, ())) for target in CENTERS
        },
        "primary_endpoint": "all_nine_seed_probability_ensemble_bacc",
        "seed_cell_endpoint_role": (
            "paired_seed_cell_mean_bacc_descriptive_only"
        ),
        "all_actions_frozen_before_label_access": True,
        "target_labels_used": False,
    }
    return EvaluationPlan(
        actions_by_target=MappingProxyType(normalized),
        cells=tuple(cells),
        evaluation_row_ids_by_target=row_ids,
        plan_hash=stable_hash(hash_payload),
    )


# Explicit alias for callers that describe this as seed-grid expansion.
expand_frozen_action_plan = build_evaluation_plan


def _normalize_target_actions(
    target: str,
    raw_actions: object,
) -> tuple[FrozenActionPayload, ...]:
    values: list[tuple[str | None, object]] = []
    if isinstance(raw_actions, Mapping):
        values.extend((str(key), value) for key, value in raw_actions.items())
    elif isinstance(raw_actions, Sequence) and not isinstance(
        raw_actions, (str, bytes, bytearray)
    ):
        values.extend((None, value) for value in raw_actions)
    else:
        raise ProtocolError("Fresh Stage-70 target actions must be a mapping or sequence.")
    actions = tuple(
        _normalize_action(raw, target_hint=target, action_id_hint=action_id)
        for action_id, raw in values
    )
    if not actions:
        raise ProtocolError("Fresh Stage-70 target has no frozen actions.")
    return actions


def _normalize_action(
    raw: object,
    *,
    target_hint: str,
    action_id_hint: str | None,
) -> FrozenActionPayload:
    if isinstance(raw, FrozenActionPayload):
        if (
            raw.target_center != target_hint
            or (action_id_hint is not None and raw.action_id != action_id_hint)
        ):
            raise ProtocolError("Fresh Stage-70 action key/payload identity drifted.")
        return raw

    payload = _payload_mapping(raw)
    _reject_label_or_oracle_values(payload)
    nested = payload.get("action_payload", payload.get("action", {}))
    nested_mapping = nested if isinstance(nested, Mapping) else {}
    target = str(payload.get("target_center", payload.get("outer_target", target_hint)))
    action_id = str(payload.get("action_id", payload.get("policy_id", action_id_hint or "")))
    action_hash = _first_present(payload, _HASH_FIELDS)
    if action_hash is None:
        action_hash = _first_present(nested_mapping, _HASH_FIELDS)

    counts = _first_present(payload, _COUNT_FIELDS)
    if counts is None:
        counts = _first_present(nested_mapping, _COUNT_FIELDS)
    if counts is None:
        # A compact wrapper may describe a single class allocation because the
        # fixed residual action is class symmetric.
        compact = payload.get("source_counts_per_class")
        if compact is None:
            compact = nested_mapping.get("source_counts_per_class")
        if isinstance(compact, Mapping) and all(
            not isinstance(value, Mapping) for value in compact.values()
        ):
            counts = {0: compact, 1: compact}
    if counts is None:
        counts = _counts_from_geometry(payload, nested_mapping, action_id)

    scores = _first_present(payload, _SCORE_FIELDS)
    if scores is None:
        scores = _first_present(nested_mapping, _SCORE_FIELDS)
    permutation = payload.get(
        "source_identity_permutation",
        nested_mapping.get("source_identity_permutation", {}),
    )
    frozen = payload.get("frozen_before_label_access", True)
    midrank_semantics = payload.get(
        "normalized_midrank_semantics", "lower_is_better"
    )
    if "target_labels_used" in payload and payload["target_labels_used"] is not False:
        raise ProtocolError("Fresh Stage-70 action payload contains target-label use.")
    if "labels_used" in payload and payload["labels_used"] is not False:
        raise ProtocolError("Fresh Stage-70 action payload contains label use.")
    return FrozenActionPayload(
        target_center=target,
        action_id=action_id,
        source_counts_by_class=counts,  # type: ignore[arg-type]
        action_hash=str(action_hash or ""),
        mean_normalized_midrank_by_source=scores or {},  # type: ignore[arg-type]
        source_identity_permutation=permutation,  # type: ignore[arg-type]
        normalized_midrank_semantics=str(midrank_semantics),
        frozen_before_label_access=frozen is True,
    )


def _payload_mapping(raw: object) -> Mapping[object, object]:
    if isinstance(raw, Mapping):
        return raw
    to_payload = getattr(raw, "to_payload", None)
    if callable(to_payload):
        payload = to_payload()
        if isinstance(payload, Mapping):
            return payload
    fields = (
        "target_center",
        "outer_target",
        "action_id",
        "policy_id",
        *_HASH_FIELDS,
        *_COUNT_FIELDS,
        *_SCORE_FIELDS,
        "source_identity_permutation",
        "normalized_midrank_semantics",
        "frozen_before_label_access",
        "target_labels_used",
        "labels_used",
    )
    payload = {
        field: getattr(raw, field)
        for field in fields
        if hasattr(raw, field)
    }
    if payload:
        return payload
    raise ProtocolError("Fresh Stage-70 action payload is unsupported.")


def _counts_from_geometry(
    payload: Mapping[object, object],
    nested: Mapping[object, object],
    action_id: str,
) -> Mapping[int, Mapping[str, int]] | None:
    geometry = payload.get("geometry", nested.get("geometry"))
    if not isinstance(geometry, Mapping):
        return None
    sources = geometry.get("source_order")
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
        return None
    if action_id == BASE_ACTION_ID:
        base = geometry.get("base_per_source", BASE_PER_SOURCE)
        try:
            counts = {str(source): int(base) for source in sources}
        except (TypeError, ValueError):
            return None
        return {0: counts, 1: counts}
    return None


def _first_present(
    payload: Mapping[object, object],
    fields: Sequence[str],
) -> object | None:
    for field in fields:
        if field in payload:
            return payload[field]
    return None


def _reject_label_or_oracle_values(payload: Mapping[object, object]) -> None:
    for raw_key, value in payload.items():
        key = str(raw_key).lower()
        if key in _FORBIDDEN_VALUE_FIELDS:
            raise ProtocolError(
                "Frozen Stage-70 actions cannot contain labels, utility, or oracle values."
            )
        if key in {"target_labels_used", "labels_used"} and value is not False:
            raise ProtocolError("Frozen Stage-70 actions cannot use target labels.")


def _validate_target_actions(
    target: str,
    actions: Sequence[FrozenActionPayload],
) -> None:
    by_id: dict[str, FrozenActionPayload] = {}
    for action in actions:
        if action.target_center != target or action.action_id in by_id:
            raise ProtocolError("Fresh Stage-70 action identities are not distinct.")
        by_id[action.action_id] = action
    expected_ids = set(expected_action_ids(target))
    if set(by_id) != expected_ids:
        raise ProtocolError(
            "Fresh Stage-70 requires B/U/G/S/P and every H x e tail action."
        )

    sources = legal_sources(target)
    source_set = set(sources)
    for action_id in expected_action_ids(target):
        action = by_id[action_id]
        expected_budget = (
            BASE_BUDGET_PER_CLASS
            if action_id == BASE_ACTION_ID
            else MATCHED_BUDGET_PER_CLASS
        )
        for label in (0, 1):
            counts = action.source_counts_by_class[label]
            if set(counts) != source_set:
                raise ProtocolError(
                    "Fresh Stage-70 action includes H or omits a legal source."
                )
            if sum(counts.values()) != expected_budget:
                raise ProtocolError("Fresh Stage-70 per-class action budget drifted.")
        if action.budget_per_class != expected_budget:
            raise ProtocolError("Fresh Stage-70 action class budgets disagree.")

        if action_id == BASE_ACTION_ID and any(
            count != BASE_PER_SOURCE
            for label in (0, 1)
            for count in action.source_counts_by_class[label].values()
        ):
            raise ProtocolError("B must be the exact 1,024/class equal-union base.")
        if action_id == UNIFORM_ACTION_ID and any(
            count != MATCHED_BUDGET_PER_CLASS // len(sources)
            for label in (0, 1)
            for count in action.source_counts_by_class[label].values()
        ):
            raise ProtocolError("U must be the exact uniform 1,152/class control.")
        if action_id != BASE_ACTION_ID and any(
            count < BASE_PER_SOURCE
            for label in (0, 1)
            for count in action.source_counts_by_class[label].values()
        ):
            raise ProtocolError("Residual top-up action replaced rows from the base.")

        selected_source = tail_source(action_id)
        if selected_source is not None:
            if selected_source == target:
                raise ProtocolError("H x e tail includes the held-out target expert.")
            for label in (0, 1):
                expected = {
                    source: BASE_PER_SOURCE
                    + (128 if source == selected_source else 0)
                    for source in sources
                }
                if dict(action.source_counts_by_class[label]) != expected:
                    raise ProtocolError("H x e single-source tail geometry drifted.")

    for rank_action_id in (
        GLOBAL_ACTION_ID,
        SUPPORT_ACTION_ID,
        PERMUTATION_ACTION_ID,
    ):
        rank_action = by_id[rank_action_id]
        if set(rank_action.mean_normalized_midrank_by_source) != source_set or any(
            score < 0.0 or score > 1.0
            for score in rank_action.mean_normalized_midrank_by_source.values()
        ):
            raise ProtocolError(
                "G/S/P must bind finite lower-is-better normalized midranks for every source."
            )
    support = by_id[SUPPORT_ACTION_ID]

    permutation_action = by_id[PERMUTATION_ACTION_ID]
    permutation = permutation_action.source_identity_permutation
    if (
        set(permutation) != source_set
        or set(permutation.values()) != source_set
        or all(permutation[source] == source for source in sources)
    ):
        raise ProtocolError("P must bind a fixed non-identity source permutation.")
    for label in (0, 1):
        support_counts = support.source_counts_by_class[label]
        permutation_counts = permutation_action.source_counts_by_class[label]
        forward = all(
            permutation_counts[permutation[source]] == support_counts[source]
            for source in sources
        )
        inverse = all(
            permutation_counts[source] == support_counts[permutation[source]]
            for source in sources
        )
        if not (forward or inverse):
            raise ProtocolError("P is not a source-identity permutation of S.")

    # These are the three rank-derived identities; none may be represented by
    # the historical calibrated-energy action id.
    rank_ids = {
        by_id[GLOBAL_ACTION_ID].action_id,
        support.action_id,
        permutation_action.action_id,
    }
    if rank_ids != {
        GLOBAL_ACTION_ID,
        SUPPORT_ACTION_ID,
        PERMUTATION_ACTION_ID,
    }:
        raise ProtocolError("Fresh Stage-70 rank-action identities drifted.")


def _normalize_evaluation_rows(
    values: Mapping[object, Sequence[object]] | None,
) -> Mapping[str, tuple[str, ...]]:
    if values is None:
        return MappingProxyType({})
    if not isinstance(values, Mapping):
        raise ProtocolError("Fresh Stage-70 evaluation row plan must be target keyed.")
    normalized_raw = {str(key): value for key, value in values.items()}
    if set(normalized_raw) != set(CENTERS):
        raise ProtocolError("Fresh Stage-70 row plan must cover all nine targets.")
    normalized: dict[str, tuple[str, ...]] = {}
    seen: set[str] = set()
    for target in CENTERS:
        rows = tuple(str(value) for value in normalized_raw[target])
        if (
            not rows
            or len(rows) != len(set(rows))
            or any(not row or row.strip() != row for row in rows)
            or seen.intersection(rows)
        ):
            raise ProtocolError("Fresh Stage-70 evaluation row coverage is invalid.")
        normalized[target] = rows
        seen.update(rows)
    return MappingProxyType(normalized)


__all__ = (
    "build_evaluation_plan",
    "expand_frozen_action_plan",
)
