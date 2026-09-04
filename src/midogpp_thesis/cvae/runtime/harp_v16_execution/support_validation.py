"""Fresh-process reconstruction for the HARP v16 support-adapted router.

The validators never deserialize Python model objects and never see labels.
They reconstruct the compact physical menus and frozen route bytes, verify the
model/action manifests, and prove that every emitted vector is either the
named exact directional action or byte-identical protected B.
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
from .contracts import ActionKind, LabelFreeOuterMenu, PrelabelRouteSet
from .menu_root_binding import CenterMenuRootBinding
from .stores import read_artifact_value, read_prelabel_routes


FRESH_VALIDATION_TIMEOUT_SECONDS = 300
MODEL_ARTIFACT_ROLE = "target_support_router_models"
TARGET_ACTION_ARTIFACT_ROLE = "target_support_target_actions"


def _verified_artifact_body(
    value: Mapping[str, object], *, schema: str, role: str
) -> dict[str, object]:
    """Verify the full manifest independently of its scientific identity."""

    body = dict(value)
    observed = require_sha256(
        body.pop("artifact_hash", None), name=f"HARP v16 {role} artifact hash"
    )
    if value.get("schema_version") != schema or canonical_hash(body) != observed:
        raise ProtocolError(f"HARP v16 {role} artifact identity drifted.")
    return body


def _verified_model_manifest(
    value: Mapping[str, object], *, centers: Sequence[str]
) -> tuple[str, str]:
    """Reconstruct the target-local model and policy semantic identities."""

    body = _verified_artifact_body(
        value,
        schema="midogpp_harp_v16_support_router_fit_state_v1",
        role="support model",
    )
    routers = body.get("routers")
    expected_centers = tuple(str(center) for center in centers)
    if not isinstance(routers, list):
        raise ProtocolError("HARP v16 support model router inventory is malformed.")
    model_rows: list[tuple[str, str]] = []
    policy_rows: list[tuple[str, str, str]] = []
    state_rows: list[tuple[str, str]] = []
    for raw in routers:
        if not isinstance(raw, Mapping):
            raise ProtocolError("HARP v16 support model router is malformed.")
        outer = str(raw.get("outer_target_id", ""))
        endpoint = raw.get("endpoint_model")
        admission = raw.get("admission")
        if not isinstance(endpoint, Mapping) or not isinstance(admission, Mapping):
            raise ProtocolError("HARP v16 support model projection is malformed.")
        endpoint_hash = require_sha256(
            endpoint.get("model_hash"), name="HARP v16 endpoint model hash"
        )
        router_hash = require_sha256(
            raw.get("router_hash"), name="HARP v16 support router hash"
        )
        admission_hash = require_sha256(
            admission.get("admission_hash"), name="HARP v16 support admission hash"
        )
        if (
            endpoint.get("outer_target_id") != outer
            or admission.get("outer_target_id") != outer
            or raw.get("evaluation_labels_consumed") is not False
            or endpoint.get("evaluation_labels_consumed") is not False
        ):
            raise ProtocolError("HARP v16 support model crossed a target boundary.")
        model_rows.append((outer, endpoint_hash))
        policy_rows.append((outer, router_hash, admission_hash))
        state_rows.append((outer, router_hash))
    if tuple(row[0] for row in model_rows) != expected_centers:
        raise ProtocolError("HARP v16 support model center inventory drifted.")

    model_hash = canonical_hash(
        {
            "schema_version": "midogpp_harp_v16_target_local_model_set_v1",
            "models": tuple(model_rows),
            "evaluation_labels_consumed": False,
        }
    )
    policy_hash = canonical_hash(
        {
            "schema_version": "midogpp_harp_v16_target_local_policy_set_v1",
            "routers": tuple(policy_rows),
            "evaluation_labels_consumed": False,
        }
    )
    support_surface_hash = require_sha256(
        body.get("support_surface_hash"),
        name="HARP v16 support outcome surface hash",
    )
    state_hash = canonical_hash(
        {
            "schema_version": "midogpp_harp_v16_support_router_fit_state_v1",
            "support_surface_hash": support_surface_hash,
            "router_hashes": tuple(state_rows),
            "support_labels_consumed": True,
            "evaluation_labels_consumed": False,
        }
    )
    if (
        body.get("model_hash") != model_hash
        or body.get("policy_hash") != policy_hash
        or body.get("state_hash") != state_hash
        or tuple(body.get("expected_center_ids", ())) != expected_centers
        or body.get("support_labels_consumed") is not True
        or body.get("target_train_support_only") is not True
        or body.get("target_evaluation_features_used_for_fit") is not False
        or body.get("target_evaluation_labels_used") is not False
        or body.get("evaluation_labels_consumed") is not False
    ):
        raise ProtocolError("HARP v16 support model semantic identity drifted.")
    return model_hash, policy_hash


def _verified_target_manifest(
    value: Mapping[str, object], *, centers: Sequence[str]
) -> tuple[str, dict[tuple[str, str], str]]:
    """Verify the complete pre-label target action projection."""

    artifact_body = _verified_artifact_body(
        value,
        schema="midogpp_harp_v16_target_action_set_v1",
        role="target action",
    )
    semantic_body = dict(artifact_body)
    target_hash = require_sha256(
        semantic_body.pop("target_action_hash", None),
        name="HARP v16 target action hash",
    )
    if canonical_hash(semantic_body) != target_hash:
        raise ProtocolError("HARP v16 target action semantic identity drifted.")
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
        or semantic_body.get("exact_top1_physical_action_only") is not True
        or semantic_body.get("evaluation_labels_consumed") is not False
    ):
        raise ProtocolError("HARP v16 target action inventory drifted.")
    for name, inventory in (("physical", physical), ("effective", effective)):
        for observed in inventory.values():
            require_sha256(observed, name=f"HARP v16 target {name} menu hash")
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
            raise ProtocolError("HARP v16 target case-menu row is malformed.")
        rows.append(
            (
                raw[0],
                raw[1],
                require_sha256(raw[2], name="HARP v16 target case-menu hash"),
            )
        )
    if (
        tuple(rows) != tuple(sorted(rows, key=lambda row: (row[0], row[1])))
        or len({(outer, case) for outer, case, _hash in rows}) != len(rows)
        or {outer for outer, _case, _hash in rows} != set(expected_centers)
        or semantic_body.get("target_case_count") != len(rows)
    ):
        raise ProtocolError("HARP v16 target case-menu coverage drifted.")
    return target_hash, {
        (outer, case): menu_hash for outer, case, menu_hash in rows
    }


def _case_indices(block: object, case_id: str) -> np.ndarray:
    values = np.asarray(getattr(block, "case_ids"), dtype=object)
    indices = np.flatnonzero(values == case_id)
    if not len(indices):
        raise ProtocolError("HARP v16 routed case is absent from its physical menu.")
    return indices


def _directional(
    baseline: np.ndarray, challenger: np.ndarray, direction: str
) -> np.ndarray:
    b = np.ascontiguousarray(baseline, dtype=np.float32)
    a = np.ascontiguousarray(challenger, dtype=np.float32)
    if b.shape != a.shape or b.ndim != 1:
        raise ProtocolError("HARP v16 validator action geometry drifted.")
    b_positive = b >= np.float32(0.5)
    a_positive = a >= np.float32(0.5)
    if direction == "D01":
        active = (~b_positive) & a_positive
    elif direction == "D10":
        active = b_positive & (~a_positive)
    else:
        raise ProtocolError("HARP v16 validator direction is unknown.")
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
        len(rows) != 10
        or sum(block.action_kind is ActionKind.B for block in rows) != 1
        or sum(block.action_kind is ActionKind.U for block in rows) != 1
        or observed_sources != expected_sources
    ):
        raise ProtocolError("HARP v16 validator target action inventory drifted.")
    return rows


def validate_support_adapted_bundle(
    *,
    route_root: Path,
    menu_binding: CenterMenuRootBinding,
    model_root: Path,
    target_action_root: Path,
    expected_center_ids: Sequence[str],
    expected_config_hash: str,
) -> Mapping[str, object]:
    """Reconstruct one complete pre-label bundle in the current process."""

    centers = tuple(str(value) for value in expected_center_ids)
    if centers != tuple(CENTERS):
        raise ProtocolError("HARP v16 validator center universe drifted.")
    menus = menu_binding.validate_durable()
    routes = read_prelabel_routes(Path(route_root))
    model = read_artifact_value(Path(model_root), role=MODEL_ARTIFACT_ROLE)
    target = read_artifact_value(
        Path(target_action_root), role=TARGET_ACTION_ARTIFACT_ROLE
    )
    model_hash, policy_hash = _verified_model_manifest(
        model.manifest, centers=centers
    )
    target_hash, target_case_menus = _verified_target_manifest(
        target.manifest, centers=centers
    )
    if (
        model.manifest.get("config_hash") != expected_config_hash
        or tuple(model.manifest.get("expected_center_ids", ())) != centers
        or target.manifest.get("config_hash") != expected_config_hash
        or tuple(target.manifest.get("expected_center_ids", ())) != centers
        or target.manifest.get("model_hash") != model_hash
        or target.manifest.get("policy_hash") != policy_hash
        or routes.model_hash != model_hash
        or routes.policy_hash != policy_hash
        or routes.target_action_hash != target_hash
    ):
        raise ProtocolError("HARP v16 validator model/action/route binding drifted.")

    menu_by_center = {menu.outer_target_id: menu for menu in menus}
    expected_cases: set[tuple[str, str]] = set()
    for menu in menus:
        blocks = _target_blocks(menu)
        baseline = next(row for row in blocks if row.action_kind is ActionKind.B)
        expected_cases.update((menu.outer_target_id, case) for case in set(baseline.case_ids))
    if {(row.outer_target_id, row.case_id) for row in routes.cases} != expected_cases:
        raise ProtocolError("HARP v16 validator routed case coverage drifted.")
    if set(target_case_menus) != expected_cases:
        raise ProtocolError("HARP v16 target manifest case coverage drifted.")
    if target.manifest.get("physical_outer_menu_hashes") != {
        center: menu_by_center[center].menu_hash for center in centers
    }:
        raise ProtocolError("HARP v16 target manifest physical menus drifted.")

    decision_hashes: list[str] = []
    for case in routes.cases:
        menu = menu_by_center.get(case.outer_target_id)
        if menu is None:
            raise ProtocolError("HARP v16 validator routed center is absent.")
        blocks = _target_blocks(menu)
        baseline = next(row for row in blocks if row.action_kind is ActionKind.B)
        uniform = next(row for row in blocks if row.action_kind is ActionKind.U)
        indices = _case_indices(baseline, case.case_id)
        sample_ids = tuple(baseline.sample_ids[int(index)] for index in indices)
        b = np.ascontiguousarray(baseline.probabilities[indices], dtype=np.float32)
        u = np.ascontiguousarray(uniform.probabilities[indices], dtype=np.float32)
        if (
            case.sample_ids != sample_ids
            or case.baseline_probabilities.tobytes(order="C") != b.tobytes(order="C")
            or case.uniform_probabilities.tobytes(order="C") != u.tobytes(order="C")
        ):
            raise ProtocolError("HARP v16 validator case identity/B/U bytes drifted.")
        if (
            case.decision_payload.get("menu_hash")
            != target_case_menus[(case.outer_target_id, case.case_id)]
            or case.decision_payload.get("surface_role") != "TARGET_EVALUATION"
            or case.decision_payload.get("evaluation_labels_used") is not False
        ):
            raise ProtocolError("HARP v16 route escaped its sealed target case menu.")
        if case.selected_kind is ActionKind.B:
            expected = b
        else:
            selected = tuple(
                row
                for row in blocks
                if row.action_kind is case.selected_kind
                and row.selected_source_id == case.selected_source_id
            )
            if len(selected) != 1 or case.direction not in {"D01", "D10"}:
                raise ProtocolError("HARP v16 validator selected action is absent.")
            expected = _directional(
                b,
                np.ascontiguousarray(selected[0].probabilities[indices], dtype=np.float32),
                str(case.direction),
            )
        if (
            case.selected_probabilities.tobytes(order="C") != expected.tobytes(order="C")
            or case.routed_probabilities.tobytes(order="C") != expected.tobytes(order="C")
        ):
            raise ProtocolError("HARP v16 validator route is not an exact menu action.")
        decision_hashes.append(case.decision_hash)

    body = {
        "schema_version": "midogpp_harp_v16_support_route_reconstruction_v1",
        "menu_binding_hash": menu_binding.binding_hash,
        "route_hash": routes.route_hash,
        "model_hash": model_hash,
        "policy_hash": policy_hash,
        "target_action_hash": target_hash,
        "expected_config_hash": expected_config_hash,
        "expected_center_ids": list(centers),
        "case_count": len(routes.cases),
        "decision_hashes_hash": canonical_hash(decision_hashes),
        "exact_physical_action_or_byte_identical_B": True,
        "evaluation_labels_opened": False,
    }
    return {**body, "reconstruction_hash": canonical_hash(body)}


def _child_validate(payload: Mapping[str, object], queue: object) -> None:
    try:
        binding = CenterMenuRootBinding.from_payload(
            payload["menu_binding"], validate_durable=True  # type: ignore[arg-type]
        )
        reconstructed = validate_support_adapted_bundle(
            route_root=Path(str(payload["route_root"])),
            menu_binding=binding,
            model_root=Path(str(payload["model_root"])),
            target_action_root=Path(str(payload["target_action_root"])),
            expected_center_ids=tuple(payload["expected_center_ids"]),  # type: ignore[arg-type]
            expected_config_hash=str(payload["expected_config_hash"]),
        )
        body = {
            "schema_version": "midogpp_harp_v16_fresh_validation_v1",
            "process_id": os.getpid(),
            **dict(reconstructed),
        }
        queue.put({"ok": True, "value": {**body, "validation_hash": canonical_hash(body)}})
    except BaseException as exc:  # pragma: no cover - exercised through parent
        queue.put(
            {
                "ok": False,
                "error_class": exc.__class__.__name__,
                "error": str(exc)[:2000],
            }
        )


def run_two_fresh_support_validations(
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
            raise ProtocolError("HARP v16 fresh validation timed out.")
        try:
            message = queue.get(timeout=5)
        except Empty as exc:
            raise ProtocolError("HARP v16 fresh validation returned no result.") from exc
        finally:
            queue.close()
            queue.join_thread()
        if process.exitcode != 0 or not isinstance(message, Mapping) or message.get("ok") is not True:
            detail = "unknown validation failure"
            if isinstance(message, Mapping):
                detail = f"{message.get('error_class')}: {message.get('error')}"
            raise ProtocolError(f"HARP v16 fresh validation failed: {detail}")
        value = message.get("value")
        if not isinstance(value, Mapping):
            raise ProtocolError("HARP v16 fresh validation payload is malformed.")
        rows.append(dict(value))
    if (
        len({row.get("process_id") for row in rows}) != 2
        or len({row.get("validation_hash") for row in rows}) != 2
        or len({row.get("reconstruction_hash") for row in rows}) != 1
        or any(row.get("evaluation_labels_opened") is not False for row in rows)
    ):
        raise ProtocolError("HARP v16 fresh validation independence drifted.")
    return rows[0], rows[1]


__all__ = (
    "MODEL_ARTIFACT_ROLE",
    "TARGET_ACTION_ARTIFACT_ROLE",
    "run_two_fresh_support_validations",
    "validate_support_adapted_bundle",
)
