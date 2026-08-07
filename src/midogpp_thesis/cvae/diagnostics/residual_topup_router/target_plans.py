"""Label-free fixed action library and outer-fold/target plan materialization."""

from __future__ import annotations

from dataclasses import dataclass
import json
from types import MappingProxyType
from typing import Mapping

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from ...routing.residual_topup import (
    build_energy_directed_topup_action,
    build_uniform_topup_action,
    inner_topup_geometry,
    target_topup_geometry,
)
from .contracts import (
    BASE_ONLY_ACTION_ID,
    CENTERS,
    DEVELOPMENT_ACTION_IDS,
    ENERGY_TOPUP_ACTION_ID,
    TARGET_ACTION_IDS,
    UNIFORM_TOPUP_ACTION_ID,
    development_queries,
    legal_development_sources,
    target_sources,
)


PLAN_COLUMNS = (
    "schema_version",
    "plan_ordinal",
    "phase",
    "outer_target",
    "query_center",
    "action_id",
    "arm_role",
    "budget_role",
    "candidate_sources_json",
    "calibrated_energy_json",
    "direction_semantics",
    "direction_weights_json",
    "base_per_source",
    "base_total_per_class",
    "topup_total_per_class",
    "final_total_per_class",
    "topup_counts_json",
    "final_counts_by_class_json",
    "final_weights_by_class_json",
    "windows_by_class_json",
    "effective_source_count_by_class_json",
    "maximum_source_weight",
    "allocation_hash",
    "window_hash",
    "action_hash",
    "plan_hash",
    "support_labels_used",
    "target_H_labels_used",
    "target_expert_excluded",
    "outer_and_query_experts_excluded",
    "seed_selection_performed",
    "selection_source",
    "claim_role",
)

ASSIGNMENT_COLUMNS = (
    "schema_version",
    "plan_ordinal",
    "phase",
    "outer_target",
    "query_center",
    "action_id",
    "class_label",
    "source_center",
    "base_start",
    "base_stop",
    "topup_start",
    "topup_stop",
    "base_count",
    "topup_count",
    "final_count",
    "final_weight",
    "target_expert_excluded",
    "outer_and_query_experts_excluded",
)


@dataclass(frozen=True)
class PlanSurface:
    plans_by_key: Mapping[tuple[str, str, str, str], Mapping[str, object]]
    table_rows: tuple[Mapping[str, object], ...]
    assignment_rows: tuple[Mapping[str, object], ...]
    lock_payload: Mapping[str, object]

    @property
    def lock_hash(self) -> str:
        return str(self.lock_payload["router_plan_lock_hash"])

    def plan(
        self, *, phase: str, outer_target: str, query_center: str, action_id: str
    ) -> Mapping[str, object]:
        try:
            return self.plans_by_key[(phase, outer_target, query_center, action_id)]
        except KeyError as exc:
            raise ProtocolError("Residual top-up plan key is absent.") from exc


def build_plan_surface(
    config: object,
    source_cache: object,
    *,
    source_cache_lock_hash: str,
    support_partition_lock_hash: str,
) -> PlanSurface:
    """Build all finite actions without labels or evaluation embeddings."""

    plans: dict[tuple[str, str, str, str], Mapping[str, object]] = {}
    rows: list[dict[str, object]] = []
    assignments: list[dict[str, object]] = []

    def append_plan(
        *,
        phase: str,
        outer: str,
        query: str,
        action_id: str,
        payload: Mapping[str, object],
    ) -> None:
        key = (phase, outer, query, action_id)
        if key in plans:
            raise ProtocolError("Residual top-up plan duplicated.")
        plan_ordinal = len(rows)
        geometry = _mapping(payload, "geometry")
        final_counts = _mapping(payload, "final_counts_by_class")
        final_weights = _mapping(payload, "final_weights_by_class")
        windows = _mapping(payload, "windows_by_class")
        candidates = tuple(str(value) for value in geometry["source_order"])
        unhashed = {
            "schema_version": "midogpp_residual_topup_plan_v1",
            "plan_ordinal": plan_ordinal,
            "phase": phase,
            "outer_target": outer,
            "query_center": query,
            "action_id": action_id,
            "arm_role": "development_action" if phase == "development" else "target_action",
            "budget_role": "base_budget_reference" if action_id == BASE_ONLY_ACTION_ID else "matched_topup_primary",
            "candidate_sources": list(candidates),
            "action_payload": dict(payload),
            "support_labels_used": False,
            "target_H_labels_used": False,
            "target_expert_excluded": True,
            "outer_and_query_experts_excluded": phase == "development",
            "seed_selection_performed": False,
            "selection_source": "fixed_label_free_energy_rank" if action_id == ENERGY_TOPUP_ACTION_ID else "fixed_control",
            "claim_role": "terminal_consumed_validation_diagnostic_only",
        }
        plan = {**unhashed, "plan_hash": stable_hash(unhashed)}
        plans[key] = MappingProxyType(plan)
        rows.append(
            {
                "schema_version": "midogpp_residual_topup_plan_row_v1",
                "plan_ordinal": plan_ordinal,
                "phase": phase,
                "outer_target": outer,
                "query_center": query,
                "action_id": action_id,
                "arm_role": unhashed["arm_role"],
                "budget_role": unhashed["budget_role"],
                "candidate_sources_json": _compact(list(candidates)),
                "calibrated_energy_json": _compact(payload.get("calibrated_energy_by_source", {})),
                "direction_semantics": str(payload["direction_semantics"]),
                "direction_weights_json": _compact(payload.get("direction_weights", {})),
                "base_per_source": int(geometry["base_per_source"]),
                "base_total_per_class": int(geometry["base_total_per_class"]),
                "topup_total_per_class": int(geometry["topup_total_per_class"]),
                "final_total_per_class": int(geometry["final_total_per_class"]),
                "topup_counts_json": _compact(payload["topup_counts"]),
                "final_counts_by_class_json": _compact(final_counts),
                "final_weights_by_class_json": _compact(final_weights),
                "windows_by_class_json": _compact(windows),
                "effective_source_count_by_class_json": _compact(payload["effective_source_count_by_class"]),
                "maximum_source_weight": float(payload["maximum_source_weight"]),
                "allocation_hash": str(payload["allocation_hash"]),
                "window_hash": str(payload["window_hash"]),
                "action_hash": str(payload["action_hash"]),
                "plan_hash": plan["plan_hash"],
                "support_labels_used": False,
                "target_H_labels_used": False,
                "target_expert_excluded": True,
                "outer_and_query_experts_excluded": phase == "development",
                "seed_selection_performed": False,
                "selection_source": unhashed["selection_source"],
                "claim_role": unhashed["claim_role"],
            }
        )
        for class_label in (0, 1):
            label = str(class_label)
            for source in candidates:
                window = _mapping(_mapping(windows, label), source)
                assignments.append(
                    {
                        "schema_version": "midogpp_residual_topup_assignment_v1",
                        "plan_ordinal": plan_ordinal,
                        "phase": phase,
                        "outer_target": outer,
                        "query_center": query,
                        "action_id": action_id,
                        "class_label": class_label,
                        "source_center": source,
                        "base_start": int(window["base"][0]),
                        "base_stop": int(window["base"][1]),
                        "topup_start": int(window["topup"][0]),
                        "topup_stop": int(window["topup"][1]),
                        "base_count": int(window["base_count"]),
                        "topup_count": int(window["topup_count"]),
                        "final_count": int(_mapping(final_counts, label)[source]),
                        "final_weight": float(_mapping(final_weights, label)[source]),
                        "target_expert_excluded": True,
                        "outer_and_query_experts_excluded": phase == "development",
                    }
                )

    for outer in CENTERS:
        for query in development_queries(outer):
            candidates = legal_development_sources(
                outer_target=outer, query_center=query
            )
            calibration = source_cache.calibrated_energy_for(query, candidates)
            energies = dict(calibration.mean_z_by_source)
            geometry = inner_topup_geometry(candidates)
            actions = {
                UNIFORM_TOPUP_ACTION_ID: build_uniform_topup_action(geometry),
                ENERGY_TOPUP_ACTION_ID: build_energy_directed_topup_action(
                    energies, geometry=geometry
                ),
            }
            for action_id in DEVELOPMENT_ACTION_IDS:
                append_plan(
                    phase="development",
                    outer=outer,
                    query=query,
                    action_id=action_id,
                    payload=actions[action_id].to_payload(),
                )
    for target in CENTERS:
        candidates = target_sources(target)
        calibration = source_cache.calibrated_energy_for(target, candidates)
        energies = dict(calibration.mean_z_by_source)
        geometry = target_topup_geometry(candidates)
        uniform = build_uniform_topup_action(geometry)
        energy = build_energy_directed_topup_action(energies, geometry=geometry)
        target_payloads = {
            BASE_ONLY_ACTION_ID: _base_only_payload(geometry, energies),
            UNIFORM_TOPUP_ACTION_ID: uniform.to_payload(),
            ENERGY_TOPUP_ACTION_ID: energy.to_payload(),
        }
        for action_id in TARGET_ACTION_IDS:
            append_plan(
                phase="target",
                outer=target,
                query=target,
                action_id=action_id,
                payload=target_payloads[action_id],
            )
    unhashed = {
        "schema_version": "midogpp_residual_topup_router_plan_lock_v1",
        "status": "LOCKED_LABEL_FREE_FINITE_ACTION_LIBRARY",
        "config_contract_hash": str(getattr(config, "contract_hash")),
        "source_cache_lock_hash": source_cache_lock_hash,
        "support_partition_lock_hash": support_partition_lock_hash,
        "plan_count": len(rows),
        "assignment_count": len(assignments),
        "development_plan_count": sum(row["phase"] == "development" for row in rows),
        "target_plan_count": sum(row["phase"] == "target" for row in rows),
        "plan_rows_hash": stable_hash(rows),
        "assignment_rows_hash": stable_hash(assignments),
        "action_ids": list(TARGET_ACTION_IDS),
        "all_actions_fixed_before_labels": True,
        "support_labels_used": False,
        "evaluation_embeddings_used": False,
        "previous_stage90_outputs_used": False,
        "diagnostic_only": True,
    }
    lock = {**unhashed, "router_plan_lock_hash": stable_hash(unhashed)}
    return PlanSurface(
        plans_by_key=MappingProxyType(plans),
        table_rows=tuple(rows),
        assignment_rows=tuple(assignments),
        lock_payload=MappingProxyType(lock),
    )


def _base_only_payload(geometry: object, energies: Mapping[str, float]) -> dict[str, object]:
    source_order = tuple(getattr(geometry, "source_order"))
    base = int(getattr(geometry, "base_per_source"))
    total = int(getattr(geometry, "base_total_per_class"))
    count_by_class = {str(label): {source: base for source in source_order} for label in (0, 1)}
    weight_by_class = {
        str(label): {source: 1.0 / len(source_order) for source in source_order}
        for label in (0, 1)
    }
    windows = {
        str(label): {
            source: {
                "base": [0, base],
                "topup": [base, base],
                "base_count": base,
                "topup_count": 0,
                "required_capacity": base,
            }
            for source in source_order
        }
        for label in (0, 1)
    }
    geometry_payload = getattr(geometry, "to_payload")()
    geometry_payload = {
        **geometry_payload,
        "topup_total_per_class": 0,
        "final_total_per_class": total,
    }
    unhashed = {
        "schema_version": "midogpp_residual_topup_base_only_action_v1",
        "geometry": geometry_payload,
        "action_kind": "exact_equal_union_base_budget_reference",
        "direction_semantics": "no_topup_base_only",
        "temperature": None,
        "calibrated_energy_by_source": dict(energies),
        "direction_weights": {},
        "topup_counts": {source: 0 for source in source_order},
        "final_counts_by_class": count_by_class,
        "final_weights_by_class": weight_by_class,
        "windows_by_class": windows,
        "effective_source_count_by_class": {"0": float(len(source_order)), "1": float(len(source_order))},
        "maximum_source_weight": 1.0 / len(source_order),
        "allocation_hash": stable_hash({"base_only": True, "counts": count_by_class}),
        "window_hash": stable_hash(windows),
        "density_constraint_semantics": "exact_equal_union_base",
    }
    return {**unhashed, "action_hash": stable_hash(unhashed)}


def action_payload(plan: Mapping[str, object]) -> Mapping[str, object]:
    return _mapping(plan, "action_payload")


def _mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ProtocolError(f"Residual top-up plan lacks mapping {key!r}.")
    return result


def _compact(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


__all__ = (
    "ASSIGNMENT_COLUMNS",
    "PLAN_COLUMNS",
    "PlanSurface",
    "action_payload",
    "build_plan_surface",
)
