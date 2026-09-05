"""Cross-artifact identity bindings for the HARP v18 phase machine."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ....protocol import ProtocolError
from ....routing.harp_protocol import canonical_hash
from ....runtime.harp_v18_execution.contracts import FrozenRouteReceipt, PrelabelRouteSet
from ....runtime.harp_v18_execution.stores import read_prelabel_routes


def validate_in_memory_route_bindings(
    routes: PrelabelRouteSet,
    *,
    model_hash: str,
    target_action_hash: str,
    centers: tuple[str, ...],
) -> None:
    if not isinstance(routes, PrelabelRouteSet):
        raise ProtocolError("HARP v18 routing returned an untyped route set.")
    if (
        routes.model_hash != model_hash
        or routes.target_action_hash != target_action_hash
        or tuple(sorted({case.outer_target_id for case in routes.cases})) != centers
    ):
        raise ProtocolError("HARP v18 in-memory route cross-binding drifted.")


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
        frozen.get("schema_version") != "midogpp_harp_v18_frozen_route_seal_v1"
        or frozen.get("status") != "FROZEN_AFTER_TWO_FRESH_RECONSTRUCTIONS"
        or seal_hash != canonical_hash(frozen_body)
        or frozen.get("config_hash") != config_hash
        or tuple(frozen.get("expected_center_ids", ())) != centers
        or frozen.get("model_hash") != model_hash
        or frozen.get("target_action_hash") != target_action_hash
        or frozen.get("evaluation_labels_opened") is not False
    ):
        raise ProtocolError("HARP v18 frozen route identity binding drifted.")
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
        raise ProtocolError("HARP v18 disk-reconstructed frozen routes drifted.")
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
    "reconstruct_frozen_routes_for_evaluation",
    "validate_in_memory_route_bindings",
)
