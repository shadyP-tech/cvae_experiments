"""Frozen-action validation and composition-deduplicated evaluation planning."""

from __future__ import annotations

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
    EXPECTED_LOGICAL_PREDICTION_COUNT,
    EvaluationCell,
    EvaluationPlan,
    FrozenActionPayload,
    GENERATION_SEEDS,
    GLOBAL_ACTION_ID,
    MATCHED_BUDGET_PER_CLASS,
    PERMUTATION_ACTION_ID,
    ROUTED_ACTION_ID,
    TRAINING_SEEDS,
    TOPUP_TOTAL_PER_CLASS,
    UNIFORM_ACTION_ID,
    CompositionCell,
    expected_action_ids,
    legal_sources,
    tail_source,
)


_ROLE_BY_CORE_ACTION = {
    BASE_ACTION_ID: "base",
    UNIFORM_ACTION_ID: "uniform_control",
    GLOBAL_ACTION_ID: "global_ablation",
    ROUTED_ACTION_ID: "utility_aligned_router",
    PERMUTATION_ACTION_ID: "target_feature_permutation_control",
}
_FORBIDDEN_KEYS = frozenset(
    {
        "labels",
        "target_labels",
        "evaluation_labels",
        "y_true",
        "bacc",
        "macro_f1",
        "oracle_source",
        "oracle_action",
        "utility_by_source",
    }
)


def build_evaluation_plan(
    actions_by_target: Mapping[object, object],
    *,
    evaluation_row_ids_by_target: Mapping[object, Sequence[object]] | None = None,
) -> EvaluationPlan:
    """Expand 9 x 13 x 9 logical cells and deduplicate only compute cells."""

    if not isinstance(actions_by_target, Mapping):
        raise ProtocolError("Utility-aligned frozen actions must be target keyed.")
    raw_targets = {str(key): value for key, value in actions_by_target.items()}
    if set(raw_targets) != set(CENTERS):
        raise ProtocolError("Utility-aligned Stage-70 requires all target centers.")

    normalized: dict[str, tuple[FrozenActionPayload, ...]] = {}
    logical: list[EvaluationCell] = []
    compositions: list[CompositionCell] = []
    for target in CENTERS:
        actions = _normalize_target_actions(target, raw_targets[target])
        _validate_target_actions(target, actions)
        by_id = {action.action_id: action for action in actions}
        ordered = tuple(by_id[action_id] for action_id in expected_action_ids(target))
        normalized[target] = ordered
        for training_seed in TRAINING_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                seen_compositions: set[str] = set()
                for action in ordered:
                    logical.append(
                        EvaluationCell(
                            target_center=target,
                            training_seed=training_seed,
                            generation_seed=generation_seed,
                            action_id=action.action_id,
                            action_hash=action.action_hash,
                            composition_hash=action.composition_hash,
                        )
                    )
                    if action.composition_hash not in seen_compositions:
                        seen_compositions.add(action.composition_hash)
                        compositions.append(
                            CompositionCell(
                                target_center=target,
                                training_seed=training_seed,
                                generation_seed=generation_seed,
                                composition_hash=action.composition_hash,
                                representative_action_id=action.action_id,
                            )
                        )
    if (
        len(logical) != EXPECTED_LOGICAL_PREDICTION_COUNT
        or len({cell.key for cell in logical}) != len(logical)
        or len({cell.key for cell in compositions}) != len(compositions)
    ):
        raise ProtocolError("Utility-aligned evaluation-plan coverage drifted.")

    rows = _normalize_evaluation_rows(evaluation_row_ids_by_target)
    payload = {
        "schema_version": "midogpp_utility_aligned_residual_fresh_plan_v1",
        "targets": list(CENTERS),
        "training_seeds": list(TRAINING_SEEDS),
        "generation_seeds": list(GENERATION_SEEDS),
        "logical_actions_by_target": {
            target: [action.to_payload() for action in normalized[target]]
            for target in CENTERS
        },
        "logical_prediction_count": len(logical),
        "unique_composition_fit_count": len(compositions),
        "evaluation_row_ids_by_target": {
            target: list(rows.get(target, ())) for target in CENTERS
        },
        "deduplication_key": "final_source_counts_by_class_hash",
        "logical_actions_remain_distinct": True,
        "target_labels_used": False,
    }
    return EvaluationPlan(
        actions_by_target=MappingProxyType(normalized),
        logical_cells=tuple(logical),
        composition_cells=tuple(compositions),
        evaluation_row_ids_by_target=rows,
        plan_hash=stable_hash(payload),
    )


expand_frozen_action_plan = build_evaluation_plan


def _normalize_target_actions(
    target: str,
    raw_actions: object,
) -> tuple[FrozenActionPayload, ...]:
    if isinstance(raw_actions, Mapping):
        values = tuple((str(key), value) for key, value in raw_actions.items())
    elif isinstance(raw_actions, Sequence) and not isinstance(raw_actions, (str, bytes)):
        values = tuple((None, value) for value in raw_actions)
    else:
        raise ProtocolError("Utility-aligned target actions must be a mapping or list.")
    return tuple(
        _normalize_action(target, value, action_id_hint=hint)
        for hint, value in values
    )


def _normalize_action(
    target: str,
    raw: object,
    *,
    action_id_hint: str | None,
) -> FrozenActionPayload:
    if isinstance(raw, FrozenActionPayload):
        if raw.target_center != target or (
            action_id_hint is not None and raw.action_id != action_id_hint
        ):
            raise ProtocolError("Utility-aligned action key/payload identity drifted.")
        return raw
    if isinstance(raw, Mapping):
        payload = {str(key): value for key, value in raw.items()}
    else:
        to_payload = getattr(raw, "to_payload", None)
        payload = dict(to_payload()) if callable(to_payload) else {}
    if not payload:
        raise ProtocolError("Utility-aligned action payload is unsupported.")
    for key, value in payload.items():
        if key.lower() in _FORBIDDEN_KEYS:
            raise ProtocolError("Frozen Stage-70 actions contain labels/oracle utility.")
        if key in {"target_labels_used", "support_labels_used"} and value is not False:
            raise ProtocolError("Frozen Stage-70 actions cannot consume target labels.")

    action_id = str(payload.get("action_id", action_id_hint or ""))
    counts = payload.get(
        "source_counts_by_class",
        payload.get("counts_per_class", payload.get("final_counts_by_class")),
    )
    action_hash = str(
        payload.get("action_hash", payload.get("decision_hash", ""))
    )
    selected = payload.get("selected_source")
    selected_source = None if selected is None else str(selected)
    role = str(
        payload.get(
            "action_role",
            _ROLE_BY_CORE_ACTION.get(action_id, "terminal_oracle_diagnostic"),
        )
    )
    return FrozenActionPayload(
        target_center=str(payload.get("target_center", target)),
        action_id=action_id,
        action_role=role,
        source_counts_by_class=counts,  # type: ignore[arg-type]
        action_hash=action_hash,
        selected_source=selected_source,
        abstained_to_base=payload.get("abstained_to_base") is True,
        fallback_reason=(
            None
            if payload.get("fallback_reason") is None
            else str(payload.get("fallback_reason"))
        ),
        target_labels_used=payload.get("target_labels_used", False) is True,
        support_labels_used=payload.get("support_labels_used", False) is True,
    )


def _validate_target_actions(
    target: str,
    actions: Sequence[FrozenActionPayload],
) -> None:
    by_id: dict[str, FrozenActionPayload] = {}
    for action in actions:
        if action.target_center != target or action.action_id in by_id:
            raise ProtocolError("Utility-aligned logical action identities duplicate.")
        by_id[action.action_id] = action
    if set(by_id) != set(expected_action_ids(target)):
        raise ProtocolError(
            "Utility-aligned Stage-70 requires B/U/G_delta/R/P and every H x e."
        )

    sources = legal_sources(target)
    source_set = set(sources)
    base = {source: BASE_PER_SOURCE for source in sources}
    uniform = {
        source: MATCHED_BUDGET_PER_CLASS // len(sources) for source in sources
    }
    for action_id in expected_action_ids(target):
        action = by_id[action_id]
        expected_role = _ROLE_BY_CORE_ACTION.get(
            action_id, "terminal_oracle_diagnostic"
        )
        if action.action_role != expected_role:
            raise ProtocolError(
                "Utility-aligned action role drifted from its frozen identity."
            )
        for label in (0, 1):
            if set(action.source_counts_by_class[label]) != source_set:
                raise ProtocolError("Utility-aligned action includes H or omits a source.")
        class_counts = [dict(action.source_counts_by_class[label]) for label in (0, 1)]
        if class_counts[0] != class_counts[1]:
            raise ProtocolError("Utility-aligned actions must be class symmetric.")

        selected = tail_source(action_id)
        if action_id == BASE_ACTION_ID:
            expected = base
            if (
                action.selected_source is not None
                or action.abstained_to_base
                or action.fallback_reason is not None
            ):
                raise ProtocolError("B must be the exact non-abstention base.")
        elif action_id == UNIFORM_ACTION_ID:
            expected = uniform
            if (
                action.selected_source is not None
                or action.abstained_to_base
                or action.fallback_reason is not None
            ):
                raise ProtocolError("U must be the exact uniform top-up control.")
        elif selected is not None:
            expected = {
                source: BASE_PER_SOURCE + (TOPUP_TOTAL_PER_CLASS if source == selected else 0)
                for source in sources
            }
            if (
                action.selected_source != selected
                or action.abstained_to_base
                or action.fallback_reason is not None
            ):
                raise ProtocolError("H x e selected-source identity drifted.")
        elif action_id in {GLOBAL_ACTION_ID, ROUTED_ACTION_ID, PERMUTATION_ACTION_ID}:
            if action.abstained_to_base:
                expected = base
                if action.selected_source is not None:
                    raise ProtocolError("Abstained utility action selected a source.")
            else:
                if action.selected_source not in source_set:
                    raise ProtocolError("Utility action lacks a legal selected source.")
                if action.fallback_reason is not None:
                    raise ProtocolError("Active utility action retained a fallback reason.")
                expected = {
                    source: BASE_PER_SOURCE
                    + (
                        TOPUP_TOTAL_PER_CLASS
                        if source == action.selected_source
                        else 0
                    )
                    for source in sources
                }
        else:  # pragma: no cover - set equality above makes this defensive.
            raise ProtocolError("Utility-aligned action identity is unknown.")
        if any(counts != expected for counts in class_counts):
            raise ProtocolError("Utility-aligned action composition geometry drifted.")
        if action.budget_per_class not in {
            BASE_BUDGET_PER_CLASS,
            MATCHED_BUDGET_PER_CLASS,
        }:
            raise ProtocolError("Utility-aligned action budget drifted.")

def _normalize_evaluation_rows(
    values: Mapping[object, Sequence[object]] | None,
) -> Mapping[str, tuple[str, ...]]:
    if values is None:
        return MappingProxyType({})
    if not isinstance(values, Mapping):
        raise ProtocolError("Utility-aligned evaluation rows must be target keyed.")
    raw = {str(key): value for key, value in values.items()}
    if set(raw) != set(CENTERS):
        raise ProtocolError("Utility-aligned row plan must cover all targets.")
    output: dict[str, tuple[str, ...]] = {}
    seen: set[str] = set()
    for target in CENTERS:
        rows = tuple(str(value) for value in raw[target])
        if (
            not rows
            or len(rows) != len(set(rows))
            or any(not row or row.strip() != row for row in rows)
            or seen.intersection(rows)
        ):
            raise ProtocolError("Utility-aligned evaluation row plan is invalid.")
        output[target] = rows
        seen.update(rows)
    return MappingProxyType(output)


__all__ = ("build_evaluation_plan", "expand_frozen_action_plan")
