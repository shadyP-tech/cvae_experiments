"""Cross-artifact identity bindings for the HARP v4 phase machine."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ....protocol import ProtocolError
from ....routing.harp_protocol import canonical_hash
from ....runtime.harp_v4_execution.contracts import ArtifactValue, PrelabelRouteSet
from ....runtime.harp_v4_execution.stores import read_prelabel_routes


def bind_development_artifact(
    value: ArtifactValue,
    *,
    config_hash: str,
    centers: tuple[str, ...],
) -> ArtifactValue:
    if not isinstance(value, ArtifactValue):
        raise ProtocolError("HARP v4 development surface is not a typed artifact.")
    body = dict(value.manifest)
    body.pop("surface_hash", None)
    if tuple(body.get("outer_targets", ())) != centers:
        raise ProtocolError("HARP v4 development surface target universe drifted.")
    for key, expected in (
        ("config_hash", config_hash),
        ("expected_center_ids", list(centers)),
    ):
        if key in body and body[key] != expected:
            raise ProtocolError("HARP v4 development identity binding conflicted.")
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
    config_hash: str,
    centers: tuple[str, ...],
) -> ArtifactValue:
    if not isinstance(value, ArtifactValue):
        raise ProtocolError("HARP v4 source-only model is not a typed artifact.")
    body = dict(value.manifest)
    body.pop("model_hash", None)
    if body.get("development_surface_hash") != development_surface_hash:
        raise ProtocolError("HARP v4 fitted model escaped its development surface.")
    for key, expected in (
        ("config_hash", config_hash),
        ("expected_center_ids", list(centers)),
    ):
        if key in body and body[key] != expected:
            raise ProtocolError("HARP v4 model identity binding conflicted.")
        body[key] = expected
    return ArtifactValue(
        state=value.state,
        manifest={**body, "model_hash": canonical_hash(body)},
        arrays=value.arrays,
    )


def bind_target_action_artifact(
    value: ArtifactValue,
    *,
    model_hash: str,
    menu_hashes: Mapping[str, str],
    config_hash: str,
    centers: tuple[str, ...],
) -> ArtifactValue:
    if not isinstance(value, ArtifactValue):
        raise ProtocolError("HARP v4 target action surface is not a typed artifact.")
    body = dict(value.manifest)
    body.pop("target_action_hash", None)
    bindings: tuple[tuple[str, object], ...] = (
        ("config_hash", config_hash),
        ("expected_center_ids", list(centers)),
        ("model_hash", model_hash),
        ("outer_menu_hashes", dict(sorted(menu_hashes.items()))),
    )
    for key, expected in bindings:
        if key in body and body[key] != expected:
            raise ProtocolError("HARP v4 target action identity binding conflicted.")
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
        raise ProtocolError("HARP v4 routing returned an untyped route set.")
    if (
        routes.model_hash != model_hash
        or routes.target_action_hash != target_action_hash
        or tuple(sorted({case.outer_target_id for case in routes.cases})) != centers
    ):
        raise ProtocolError("HARP v4 in-memory route cross-binding drifted.")


def reconstruct_frozen_routes_for_evaluation(
    route_root: Path,
    *,
    frozen: Mapping[str, object],
    model_hash: str,
    target_action_hash: str,
    centers: tuple[str, ...],
    config_hash: str,
) -> PrelabelRouteSet:
    if (
        frozen.get("status") != "FROZEN_AFTER_TWO_FRESH_RECONSTRUCTIONS"
        or frozen.get("config_hash") != config_hash
        or tuple(frozen.get("expected_center_ids", ())) != centers
        or frozen.get("model_hash") != model_hash
        or frozen.get("target_action_hash") != target_action_hash
    ):
        raise ProtocolError("HARP v4 frozen route identity binding drifted.")
    routes = read_prelabel_routes(route_root)
    if (
        routes.route_hash != frozen.get("route_hash")
        or routes.policy_hash != frozen.get("policy_hash")
        or routes.model_hash != model_hash
        or routes.target_action_hash != target_action_hash
        or len(routes.cases) != frozen.get("case_count")
        or tuple(sorted({case.outer_target_id for case in routes.cases})) != centers
    ):
        raise ProtocolError("HARP v4 disk-reconstructed frozen routes drifted.")
    return routes


__all__ = (
    "bind_development_artifact",
    "bind_model_artifact",
    "bind_target_action_artifact",
    "reconstruct_frozen_routes_for_evaluation",
    "validate_in_memory_route_bindings",
)
