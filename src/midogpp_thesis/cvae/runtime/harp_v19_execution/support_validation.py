"""Fresh-process reconstruction for the HARP v19 pooled selected policy.

Validators deserialize no model object and see no labels. They authenticate the
public pooled-policy manifest, reconstruct every selected component from the
physical B/U/Hxe cache, and independently replay the branchwise float64 recipe.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash, require_sha256
from ...routing.safe_winner_router_v19.model_integrity import verify_complete_model_payload
from .contracts import (
    ActionKind,
    LabelFreeOuterMenu,
    reconstruct_selected_probability_blend,
    reconstruct_shrunk_probability_blend,
)
from .branch_recipe import validate_branch_recipe
from .menu_root_binding import CenterMenuRootBinding
from .stores import read_artifact_value, read_prelabel_routes
from .winner_evidence import validate_winner_evidence


FRESH_VALIDATION_TIMEOUT_SECONDS = 300
POOLED_POLICY_ARTIFACT_ROLE = "pooled_source_router_policy"
TARGET_EVALUATION_ACTION_ARTIFACT_ROLE = "target_evaluation_action_menus"


def _verified_artifact_body(
    value: Mapping[str, object], *, schema: str, role: str
) -> dict[str, object]:
    body = dict(value)
    observed = require_sha256(
        body.pop("artifact_hash", None), name=f"HARP v19 {role} artifact hash"
    )
    if value.get("schema_version") != schema or canonical_hash(body) != observed:
        raise ProtocolError(f"HARP v19 {role} artifact identity drifted.")
    return body


def _verified_model_manifest(
    value: Mapping[str, object], *, centers: Sequence[str]
) -> tuple[str, str]:
    """Recompute the sole pooled-policy and runtime-state identities."""

    body = _verified_artifact_body(
        value,
        schema="midogpp_harp_v19_pooled_policy_fit_state_v1",
        role="pooled source model",
    )
    policy = body.get("policy")
    if not isinstance(policy, Mapping):
        raise ProtocolError("HARP v19 pooled policy projection is absent.")
    model = policy.get("model")
    crossfit = policy.get("crossfit")
    admission = policy.get("admission")
    config = policy.get("config")
    if not all(isinstance(row, Mapping) for row in (model, crossfit, admission, config)):
        raise ProtocolError("HARP v19 pooled policy projection is malformed.")
    model_hash = require_sha256(policy.get("model_hash"), name="pooled model hash")
    if verify_complete_model_payload(model) != model_hash:
        raise ProtocolError("HARP v19 complete winner model identity drifted.")
    crossfit_hash = require_sha256(
        crossfit.get("result_hash"), name="pooled crossfit hash"  # type: ignore[union-attr]
    )
    admission_hash = require_sha256(
        admission.get("admission_hash"), name="pooled admission hash"  # type: ignore[union-attr]
    )
    source_menu_hash = require_sha256(
        policy.get("source_menu_hash"), name="pooled source menu hash"
    )
    truth_hash = require_sha256(
        policy.get("truth_capability_hash"), name="pooled truth capability hash"
    )
    expected_policy_hash = canonical_hash(
        {
            "schema_version": "safe_winner_router_v19",
            "model_hash": model_hash,
            "crossfit_hash": crossfit_hash,
            "admission_hash": admission_hash,
            "config": dict(config),  # type: ignore[arg-type]
            "source_menu_hash": source_menu_hash,
            "truth_capability_hash": truth_hash,
            "selected_arm": crossfit.get("final_arm"),  # type: ignore[union-attr]
            "route_threshold": crossfit.get("final_route_threshold"),  # type: ignore[union-attr]
            "pooled_known_center_policy_count": 1,
            "nested_oof_evaluates_selection_algorithm_not_final_refit": True,
            "target_evaluation_labels_consumed": False,
        }
    )
    policy_hash = require_sha256(policy.get("policy_hash"), name="pooled policy hash")
    support_surface_hash = require_sha256(
        body.get("support_surface_hash"), name="source outcome surface hash"
    )
    expected_state_hash = canonical_hash(
        {
            "schema_version": "midogpp_harp_v19_pooled_policy_fit_state_v1",
            "support_surface_hash": support_surface_hash,
            "policy_hash": policy_hash,
            "pooled_policy_count": 1,
            "source_labels_consumed": True,
            "target_evaluation_labels_consumed": False,
        }
    )
    expected_centers = tuple(str(center) for center in centers)
    if (
        policy_hash != expected_policy_hash
        or model.get("model_hash") != model_hash  # type: ignore[union-attr]
        or policy.get("source_center_count") != len(expected_centers)
        or tuple(body.get("expected_center_ids", ())) != expected_centers
        or body.get("model_hash") != model_hash
        or body.get("policy_hash") != policy_hash
        or body.get("state_hash") != expected_state_hash
        or body.get("pooled_policy_count") != 1
        or body.get("one_pooled_policy_fit") is not True
        or body.get("source_train_development_only") is not True
        or body.get("raw_source_labels_persisted") is not False
        or body.get("target_evaluation_features_used_for_fit") is not False
        or body.get("target_evaluation_labels_used") is not False
    ):
        raise ProtocolError("HARP v19 pooled model semantic identity drifted.")
    return model_hash, policy_hash


def _verified_target_manifest(
    value: Mapping[str, object], *, centers: Sequence[str]
) -> tuple[str, dict[tuple[str, str], str]]:
    artifact_body = _verified_artifact_body(
        value,
        schema="midogpp_harp_v19_target_action_set_v1",
        role="target action",
    )
    semantic_body = dict(artifact_body)
    target_hash = require_sha256(
        semantic_body.pop("target_action_hash", None), name="target action hash"
    )
    if canonical_hash(semantic_body) != target_hash:
        raise ProtocolError("HARP v19 target action semantic identity drifted.")
    expected_centers = tuple(str(center) for center in centers)
    physical = semantic_body.get("physical_outer_menu_hashes")
    effective = semantic_body.get("target_effective_menu_hashes")
    raw_rows = semantic_body.get("case_menu_rows")
    if (
        tuple(semantic_body.get("expected_center_ids", ())) != expected_centers
        or not isinstance(physical, Mapping)
        or tuple(physical) != expected_centers
        or not isinstance(effective, Mapping)
        or tuple(effective) != expected_centers
        or not isinstance(raw_rows, list)
        or semantic_body.get("soft_topk_probability_blends_allowed") is not True
        or semantic_body.get("action_families") != ["B", "U_FULL", "D01_ONLY", "D10_ONLY", "BOTH"]
        or semantic_body.get("unselected_branch_preserves_exact_B_bytes") is not True
        or semantic_body.get("case_conditional_action_selection") is not True
        or semantic_body.get("all_k_lambda_probability_matrices_persisted") is not False
        or semantic_body.get("zero_additional_classifier_or_gpu_fits_for_soft_arms")
        is not True
        or semantic_body.get("evaluation_labels_consumed") is not False
    ):
        raise ProtocolError("HARP v19 target action inventory drifted.")
    for inventory in (physical, effective):
        for observed in inventory.values():
            require_sha256(observed, name="target menu hash")
    rows: list[tuple[str, str, str]] = []
    for raw in raw_rows:
        if (
            not isinstance(raw, list)
            or len(raw) != 3
            or type(raw[0]) is not str
            or raw[0] not in expected_centers
            or type(raw[1]) is not str
            or not raw[1]
        ):
            raise ProtocolError("HARP v19 target case-menu row is malformed.")
        rows.append((raw[0], raw[1], require_sha256(raw[2], name="case-menu hash")))
    if (
        tuple(rows) != tuple(sorted(rows, key=lambda row: (row[0], row[1])))
        or len({(center, case) for center, case, _hash in rows}) != len(rows)
        or {center for center, _case, _hash in rows} != set(expected_centers)
        or semantic_body.get("target_case_count") != len(rows)
    ):
        raise ProtocolError("HARP v19 target case-menu coverage drifted.")
    return target_hash, {(center, case): digest for center, case, digest in rows}


def _case_indices(block: object, case_id: str) -> np.ndarray:
    values = np.asarray(getattr(block, "case_ids"), dtype=object)
    indices = np.flatnonzero(values == case_id)
    if not len(indices):
        raise ProtocolError("HARP v19 routed case is absent from its physical menu.")
    return indices


def _directional(
    baseline: np.ndarray, challenger: np.ndarray, direction: str
) -> np.ndarray:
    b = np.ascontiguousarray(baseline, dtype=np.float32)
    a = np.ascontiguousarray(challenger, dtype=np.float32)
    if b.shape != a.shape or b.ndim != 1:
        raise ProtocolError("HARP v19 validator action geometry drifted.")
    b_positive = b >= np.float32(0.5)
    a_positive = a >= np.float32(0.5)
    active = (
        (~b_positive) & a_positive
        if direction == "D01"
        else b_positive & (~a_positive)
        if direction == "D10"
        else None
    )
    if active is None:
        raise ProtocolError("HARP v19 validator direction is unknown.")
    output = b.copy()
    output[active] = a[active]
    return output


def _target_blocks(menu: LabelFreeOuterMenu) -> tuple[object, ...]:
    rows = tuple(block for block in menu.blocks if block.surface_role == "target")
    expected_sources = tuple(center for center in CENTERS if center != menu.outer_target_id)
    observed_sources = tuple(
        block.selected_source_id
        for block in rows
        if block.action_kind is ActionKind.HXE
    )
    if (
        len(rows) != len(expected_sources) + 2
        or sum(block.action_kind is ActionKind.B for block in rows) != 1
        or sum(block.action_kind is ActionKind.U for block in rows) != 1
        or observed_sources != expected_sources
    ):
        raise ProtocolError("HARP v19 validator target action inventory drifted.")
    return rows


def _component_from_physical(
    action_id: str,
    *,
    center_id: str,
    blocks: Sequence[object],
    indices: np.ndarray,
    baseline: np.ndarray,
) -> np.ndarray:
    if action_id == "U:FULL":
        block = next(row for row in blocks if row.action_kind is ActionKind.U)
        return np.ascontiguousarray(block.probabilities[indices], dtype=np.float32)
    parts = action_id.split(":")
    if len(parts) != 3 or parts[0] != "HXE" or parts[2] not in {"D01", "D10"}:
        raise ProtocolError("HARP v19 route component identity is malformed.")
    donor, direction = parts[1], parts[2]
    if donor == center_id or donor not in CENTERS:
        raise ProtocolError("HARP v19 route component crossed C-minus-H.")
    selected = tuple(
        row
        for row in blocks
        if row.action_kind is ActionKind.HXE and row.selected_source_id == donor
    )
    if len(selected) != 1:
        raise ProtocolError("HARP v19 route component is absent from the physical menu.")
    challenger = np.ascontiguousarray(
        selected[0].probabilities[indices], dtype=np.float32
    )
    return _directional(baseline, challenger, direction)


def validate_pooled_policy_bundle(
    *,
    route_root: Path,
    menu_binding: CenterMenuRootBinding,
    model_root: Path,
    target_action_root: Path,
    expected_center_ids: Sequence[str],
    expected_config_hash: str,
) -> Mapping[str, object]:
    """Authenticate and independently replay every frozen target recipe."""

    centers = tuple(str(value) for value in expected_center_ids)
    if centers != tuple(CENTERS):
        raise ProtocolError("HARP v19 validator center universe drifted.")
    menus = menu_binding.validate_durable()
    routes = read_prelabel_routes(Path(route_root))
    model = read_artifact_value(Path(model_root), role=POOLED_POLICY_ARTIFACT_ROLE)
    target = read_artifact_value(
        Path(target_action_root), role=TARGET_EVALUATION_ACTION_ARTIFACT_ROLE
    )
    model_hash, policy_hash = _verified_model_manifest(model.manifest, centers=centers)
    target_hash, target_case_menus = _verified_target_manifest(
        target.manifest, centers=centers
    )
    if (
        model.manifest.get("config_hash") != expected_config_hash
        or target.manifest.get("config_hash") != expected_config_hash
        or target.manifest.get("model_hash") != model_hash
        or target.manifest.get("policy_hash") != policy_hash
        or routes.model_hash != model_hash
        or routes.policy_hash != policy_hash
        or routes.target_action_hash != target_hash
    ):
        raise ProtocolError("HARP v19 validator model/action/route binding drifted.")

    menu_by_center = {menu.outer_target_id: menu for menu in menus}
    expected_cases: set[tuple[str, str]] = set()
    for menu in menus:
        baseline = next(
            row for row in _target_blocks(menu) if row.action_kind is ActionKind.B
        )
        expected_cases.update((menu.outer_target_id, case) for case in set(baseline.case_ids))
    if (
        {(row.outer_target_id, row.case_id) for row in routes.cases} != expected_cases
        or set(target_case_menus) != expected_cases
        or target.manifest.get("physical_outer_menu_hashes")
        != {center: menu_by_center[center].menu_hash for center in centers}
    ):
        raise ProtocolError("HARP v19 validator target coverage drifted.")

    decision_hashes: list[str] = []
    policy_manifest = model.manifest["policy"]
    gate_model = policy_manifest["model"]["winner_gate"]
    threshold = policy_manifest["crossfit"]["final_route_threshold"]
    admitted = bool(policy_manifest["admission"]["admitted"]
                    and policy_manifest["crossfit"]["final_policy_enabled"])
    for case in routes.cases:
        validate_winner_evidence(case.decision_payload, gate_model=gate_model, admitted=admitted,
            threshold=threshold, routed=case.selected_kind is not ActionKind.B)
        menu = menu_by_center[case.outer_target_id]
        blocks = _target_blocks(menu)
        baseline_block = next(row for row in blocks if row.action_kind is ActionKind.B)
        uniform_block = next(row for row in blocks if row.action_kind is ActionKind.U)
        indices = _case_indices(baseline_block, case.case_id)
        sample_ids = tuple(baseline_block.sample_ids[int(index)] for index in indices)
        baseline = np.ascontiguousarray(
            baseline_block.probabilities[indices], dtype=np.float32
        )
        uniform = np.ascontiguousarray(
            uniform_block.probabilities[indices], dtype=np.float32
        )
        if (
            case.sample_ids != sample_ids
            or case.baseline_probabilities.tobytes() != baseline.tobytes()
            or case.uniform_probabilities.tobytes() != uniform.tobytes()
            or case.decision_payload.get("menu_hash")
            != target_case_menus[(case.outer_target_id, case.case_id)]
            or case.decision_payload.get("surface_role") != "TARGET_EVALUATION"
            or case.decision_payload.get("evaluation_labels_used") is not False
        ):
            raise ProtocolError("HARP v19 route escaped its sealed target menu.")
        components = tuple(
            _component_from_physical(
                action_id,
                center_id=case.outer_target_id,
                blocks=blocks,
                indices=indices,
                baseline=baseline,
            )
            for action_id in case.component_action_ids
        )
        if len(components) != len(case.component_probabilities) or any(
            left.tobytes() != right.tobytes()
            for left, right in zip(components, case.component_probabilities, strict=True)
        ):
            raise ProtocolError("HARP v19 persisted route component bytes drifted.")
        if case.selected_kind is ActionKind.SOFT_TOPK_PROBABILITY_BLEND:
            validate_branch_recipe(
                direction=case.direction, component_ids=case.component_action_ids,
                components=components, baseline=baseline, routed=case.routed_probabilities,
                payload=case.decision_payload, require_family=True,
            )
            if case.decision_payload.get("composite_lambda") != case.shrinkage:
                raise ProtocolError("HARP v19 selected lambda disagrees with its sealed recipe.")
        else:
            expected_family = "B" if case.selected_kind is ActionKind.B else "U_FULL"
            if case.decision_payload.get("composite_kind") != expected_family:
                raise ProtocolError("HARP v19 exact action family drifted.")
        if case.selected_kind is ActionKind.B:
            selected = routed = baseline
        else:
            selected = reconstruct_selected_probability_blend(
                components,
                case.component_weights,
                baseline_probabilities=baseline,
                component_action_ids=case.component_action_ids,
            )
            routed = reconstruct_shrunk_probability_blend(
                baseline, selected, case.shrinkage
            )
        if (
            case.selected_probabilities.tobytes() != selected.tobytes()
            or case.routed_probabilities.tobytes() != routed.tobytes()
        ):
            raise ProtocolError("HARP v19 route recipe replay changed bytes.")
        decision_hashes.append(case.decision_hash)

    body = {
        "schema_version": "midogpp_harp_v19_pooled_route_reconstruction_v1",
        "menu_binding_hash": menu_binding.binding_hash,
        "route_hash": routes.route_hash,
        "model_hash": model_hash,
        "policy_hash": policy_hash,
        "target_action_hash": target_hash,
        "expected_config_hash": expected_config_hash,
        "expected_center_ids": list(centers),
        "case_count": len(routes.cases),
        "decision_hashes_hash": canonical_hash(decision_hashes),
        "selected_components_reconstructed_from_physical_cache": True,
        "branchwise_soft_topk_recipe_replayed": True,
        "exact_b_bytes_independently_verified": True,
        "complete_winner_gate_model_identity_verified": True,
        "signed_winner_and_harm_gate_rule_verified": True,
        "winner_gate_prediction_replayed_from_sealed_features": True,
        "model_fitted_or_labels_used_by_validator": False,
        "all_k_lambda_probability_matrices_loaded": False,
        "selection_status": "FROZEN_SOURCE_TRAIN_POLICY",
        "probability_status": "BYTE_RECONSTRUCTED",
        "prediction_status": "SEALED_BEFORE_EVALUATION_LABELS",
        "utility_status": "NOT_OPENED",
        "evaluation_labels_opened": False,
    }
    return {**body, "reconstruction_hash": canonical_hash(body)}


def _child_validate(payload: Mapping[str, object], queue: object) -> None:
    try:
        binding = CenterMenuRootBinding.from_payload(
            payload["menu_binding"], validate_durable=True  # type: ignore[arg-type]
        )
        reconstructed = validate_pooled_policy_bundle(
            route_root=Path(str(payload["route_root"])),
            menu_binding=binding,
            model_root=Path(str(payload["model_root"])),
            target_action_root=Path(str(payload["target_action_root"])),
            expected_center_ids=tuple(payload["expected_center_ids"]),  # type: ignore[arg-type]
            expected_config_hash=str(payload["expected_config_hash"]),
        )
        body = {
            "schema_version": "midogpp_harp_v19_fresh_validation_v1",
            "process_id": os.getpid(),
            **dict(reconstructed),
        }
        queue.put(
            {"ok": True, "value": {**body, "validation_hash": canonical_hash(body)}}
        )
    except BaseException as exc:  # pragma: no cover - exercised through parent
        queue.put(
            {
                "ok": False,
                "error_class": exc.__class__.__name__,
                "error": str(exc)[:2000],
            }
        )


def run_two_fresh_pooled_policy_validations(
    *,
    route_root: Path,
    menu_binding: CenterMenuRootBinding,
    model_root: Path,
    target_action_root: Path,
    expected_center_ids: Sequence[str],
    expected_config_hash: str,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    """Run two independent spawn-process reconstructions before truth opens."""

    menu_binding.validate_durable()
    payload = {
        "route_root": str(Path(route_root).resolve()),
        "menu_binding": menu_binding.to_payload(),
        "model_root": str(Path(model_root).resolve()),
        "target_action_root": str(Path(target_action_root).resolve()),
        "expected_center_ids": list(expected_center_ids),
        "expected_config_hash": expected_config_hash,
    }
    context = mp.get_context("spawn")
    rows: list[Mapping[str, object]] = []
    for _ in range(2):
        queue = context.Queue(maxsize=1)
        process = context.Process(target=_child_validate, args=(payload, queue))
        process.start()
        process.join(FRESH_VALIDATION_TIMEOUT_SECONDS)
        if process.is_alive():
            process.terminate()
            process.join(10)
            raise ProtocolError("HARP v19 fresh validation timed out.")
        try:
            message = queue.get(timeout=5)
        except Empty as exc:
            raise ProtocolError("HARP v19 fresh validation returned no result.") from exc
        finally:
            queue.close()
            queue.join_thread()
        if (
            process.exitcode != 0
            or not isinstance(message, Mapping)
            or message.get("ok") is not True
        ):
            detail = "unknown validation failure"
            if isinstance(message, Mapping):
                detail = f"{message.get('error_class')}: {message.get('error')}"
            raise ProtocolError(f"HARP v19 fresh validation failed: {detail}")
        value = message.get("value")
        if not isinstance(value, Mapping):
            raise ProtocolError("HARP v19 fresh validation payload is malformed.")
        rows.append(dict(value))
    if (
        len({row.get("process_id") for row in rows}) != 2
        or len({row.get("validation_hash") for row in rows}) != 2
        or len({row.get("reconstruction_hash") for row in rows}) != 1
        or any(row.get("evaluation_labels_opened") is not False for row in rows)
    ):
        raise ProtocolError("HARP v19 fresh validation independence drifted.")
    return rows[0], rows[1]


__all__ = (
    "POOLED_POLICY_ARTIFACT_ROLE",
    "TARGET_EVALUATION_ACTION_ARTIFACT_ROLE",
    "run_two_fresh_pooled_policy_validations",
    "validate_pooled_policy_bundle",
)
