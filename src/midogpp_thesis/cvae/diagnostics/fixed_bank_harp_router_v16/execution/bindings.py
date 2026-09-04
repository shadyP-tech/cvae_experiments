"""Cross-artifact identity bindings for the HARP v16 phase machine."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ....protocol import ProtocolError
from ....routing.harp_protocol import canonical_hash
from ....runtime.harp_v16_execution.contracts import (
    ArtifactValue,
    FrozenRouteReceipt,
    PrelabelRouteSet,
)
from ....runtime.harp_v16_execution.stores import read_prelabel_routes


def bind_development_artifact(
    value: ArtifactValue,
    *,
    config_hash: str,
    centers: tuple[str, ...],
) -> ArtifactValue:
    if not isinstance(value, ArtifactValue):
        raise ProtocolError("HARP v16 development surface is not a typed artifact.")
    body = dict(value.manifest)
    body.pop("surface_hash", None)
    if tuple(body.get("outer_targets", ())) != centers:
        raise ProtocolError("HARP v16 development surface target universe drifted.")
    for key, expected in (
        ("config_hash", config_hash),
        ("expected_center_ids", list(centers)),
    ):
        if key in body and body[key] != expected:
            raise ProtocolError("HARP v16 development identity binding conflicted.")
        body[key] = expected
    return ArtifactValue(
        state=value.state,
        manifest={**body, "surface_hash": canonical_hash(body)},
        arrays=value.arrays,
    )


def bind_model_artifact(
    value: ArtifactValue,
    *,
    development_surface_hash: str,
    compatibility_hash: str,
    config_hash: str,
    centers: tuple[str, ...],
) -> ArtifactValue:
    if not isinstance(value, ArtifactValue):
        raise ProtocolError("HARP v16 source-only model is not a typed artifact.")
    body = dict(value.manifest)
    body.pop("model_hash", None)
    if body.get("development_surface_hash") != development_surface_hash:
        raise ProtocolError("HARP v16 fitted model escaped its development surface.")
    for key, expected in (
        ("compatibility_hash", compatibility_hash),
        ("config_hash", config_hash),
        ("expected_center_ids", list(centers)),
    ):
        if key in body and body[key] != expected:
            raise ProtocolError("HARP v16 model identity binding conflicted.")
        body[key] = expected
    return ArtifactValue(
        state=value.state,
        manifest={**body, "model_hash": canonical_hash(body)},
        arrays=value.arrays,
    )


def bind_admission_artifact(
    value: ArtifactValue,
    *,
    model_hash: str,
    development_surface_hash: str,
    config_hash: str,
    centers: tuple[str, ...],
) -> ArtifactValue:
    """Bind the source-only, per-outer policy decision to its frozen inputs."""

    if not isinstance(value, ArtifactValue):
        raise ProtocolError("HARP v16 source-policy admission is not a typed artifact.")
    body = dict(value.manifest)
    body.pop("admission_hash", None)
    bindings: tuple[tuple[str, object], ...] = (
        ("model_hash", model_hash),
        ("development_surface_hash", development_surface_hash),
        ("config_hash", config_hash),
        ("expected_center_ids", list(centers)),
    )
    for key, expected in bindings:
        if key in body and body[key] != expected:
            raise ProtocolError("HARP v16 source-policy admission binding conflicted.")
        body[key] = expected
    outer_policies = body.get("outer_policies")
    admitted_count = body.get("admitted_outer_count")
    if (
        not isinstance(outer_policies, list)
        or len(outer_policies) != len(centers)
        or type(admitted_count) is not int
        or not 0 <= admitted_count <= len(centers)
        or body.get("global_kill_switch_used") is not False
    ):
        raise ProtocolError("HARP v16 per-outer policy inventory is malformed.")
    return ArtifactValue(
        state=value.state,
        manifest={**body, "admission_hash": canonical_hash(body)},
        arrays=value.arrays,
    )


def bind_target_action_artifact(
    value: ArtifactValue,
    *,
    model_hash: str,
    compatibility_hash: str,
    admission_hash: str,
    menu_hashes: Mapping[str, str],
    config_hash: str,
    centers: tuple[str, ...],
) -> ArtifactValue:
    if not isinstance(value, ArtifactValue):
        raise ProtocolError("HARP v16 target action surface is not a typed artifact.")
    body = dict(value.manifest)
    body.pop("target_action_hash", None)
    bindings: tuple[tuple[str, object], ...] = (
        ("config_hash", config_hash),
        ("expected_center_ids", list(centers)),
        ("model_hash", model_hash),
        ("compatibility_hash", compatibility_hash),
        ("admission_hash", admission_hash),
        ("outer_menu_hashes", dict(sorted(menu_hashes.items()))),
    )
    for key, expected in bindings:
        if key in body and body[key] != expected:
            raise ProtocolError("HARP v16 target action identity binding conflicted.")
        body[key] = expected
    return ArtifactValue(
        state=value.state,
        manifest={**body, "target_action_hash": canonical_hash(body)},
        arrays=value.arrays,
    )


def validate_in_memory_route_bindings(
    routes: PrelabelRouteSet,
    *,
    model_hash: str,
    target_action_hash: str,
    centers: tuple[str, ...],
) -> None:
    if not isinstance(routes, PrelabelRouteSet):
        raise ProtocolError("HARP v16 routing returned an untyped route set.")
    if (
        routes.model_hash != model_hash
        or routes.target_action_hash != target_action_hash
        or tuple(sorted({case.outer_target_id for case in routes.cases})) != centers
    ):
        raise ProtocolError("HARP v16 in-memory route cross-binding drifted.")


def reconstruct_frozen_routes_for_evaluation(
    route_root: Path,
    *,
    frozen: Mapping[str, object],
    model_hash: str,
    target_action_hash: str,
    centers: tuple[str, ...],
    config_hash: str,
) -> tuple[PrelabelRouteSet, FrozenRouteReceipt]:
    frozen_body = dict(frozen)
    seal_hash = frozen_body.pop("seal_hash", None)
    if (
        frozen.get("schema_version") != "midogpp_harp_v16_frozen_route_seal_v1"
        or frozen.get("status") != "FROZEN_AFTER_TWO_FRESH_RECONSTRUCTIONS"
        or seal_hash != canonical_hash(frozen_body)
        or frozen.get("config_hash") != config_hash
        or tuple(frozen.get("expected_center_ids", ())) != centers
        or frozen.get("model_hash") != model_hash
        or frozen.get("target_action_hash") != target_action_hash
        or frozen.get("evaluation_labels_opened") is not False
    ):
        raise ProtocolError("HARP v16 frozen route identity binding drifted.")
    routes = read_prelabel_routes(route_root)
    if (
        routes.route_hash != frozen.get("route_hash")
        or routes.policy_hash != frozen.get("policy_hash")
        or routes.model_hash != model_hash
        or routes.target_action_hash != target_action_hash
        or len(routes.cases) != frozen.get("case_count")
        or routes.ordered_case_identity_hash
        != frozen.get("ordered_case_identity_hash")
        or routes.ordered_sample_identity_hash
        != frozen.get("ordered_sample_identity_hash")
        or tuple(sorted({case.outer_target_id for case in routes.cases})) != centers
    ):
        raise ProtocolError("HARP v16 disk-reconstructed frozen routes drifted.")
    receipt = FrozenRouteReceipt(
        seal_hash=str(seal_hash),
        config_hash=config_hash,
        route_hash=routes.route_hash,
        policy_hash=routes.policy_hash,
        model_hash=routes.model_hash,
        target_action_hash=routes.target_action_hash,
        validation_bundle_hash=str(frozen.get("validation_bundle_hash")),
        independent_validation_hashes=tuple(  # type: ignore[arg-type]
            frozen.get("independent_validation_hashes", ())
        ),
        expected_center_ids=centers,
        case_count=len(routes.cases),
        ordered_case_identity_hash=routes.ordered_case_identity_hash,
        ordered_sample_identity_hash=routes.ordered_sample_identity_hash,
    )
    return routes, receipt


__all__ = (
    "bind_admission_artifact",
    "bind_development_artifact",
    "bind_model_artifact",
    "bind_target_action_artifact",
    "reconstruct_frozen_routes_for_evaluation",
    "validate_in_memory_route_bindings",
)
