from __future__ import annotations

import hashlib

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.contracts import (
    RouteKey,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.endpoint_runtime import (
    build_case_endpoints,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.label_firewall import (
    LabelFirewall,
    SupportLabelCapability,
    support_scope,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.physical_adapter import (
    CenterPhysicalSurface,
    action_library_by_target,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.route_support import (
    BinaryLabel,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.target_local_runtime import (
    build_fingerprint_surface,
    fit_route_posterior,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _hash(value: object) -> str:
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


def _surface() -> CenterPhysicalSurface:
    sample_ids = ("a", "b", "c", "d")
    case_ids = ("case-1", "case-1", "case-2", "case-2")
    rows = []
    for action_index, action in enumerate(action_library_by_target()["0"]):
        base = np.asarray([0.2, 0.8, 0.3, 0.7], dtype=np.float32)
        values = np.stack(
            [
                np.clip(
                    base + np.float32(0.002 * (action_index + seed)),
                    0.0,
                    1.0,
                )
                for seed in range(9)
            ]
        )
        rows.append((action.action_id, values))
    return CenterPhysicalSurface(
        "0", sample_ids, case_ids, tuple(rows), _hash("store")
    )


def _capabilities() -> tuple[object, object, object]:
    values = {
        ("0", "case-1", "a"): 0,
        ("0", "case-1", "b"): 1,
        ("0", "case-2", "c"): 0,
        ("0", "case-2", "d"): 1,
    }

    def loader(keys: object, scope: str) -> tuple[BinaryLabel, ...]:
        # Deliberately return the loader rows out of order.  The firewall owns
        # the request order and canonicalizes the issued capability.
        requested = tuple(keys)  # type: ignore[arg-type]
        return tuple(
            BinaryLabel(*key, values[key], scope) for key in reversed(requested)
        )

    firewall = LabelFirewall(loader)
    firewall.advance_support()
    support = firewall.open_support(
        center="0",
        held_case_id="case-2",
        keys=(("0", "case-1", "a"), ("0", "case-1", "b")),
    )
    firewall.seal_action_surface(_hash("action-surface"))
    firewall.advance_pseudo_response()
    route = RouteKey(
        "pseudo", "1", "0", "case-2", "1", "0", _hash("pseudo-fit")
    )
    pseudo = firewall.open_pseudo_response(route_key=route, sample_ids=("c", "d"))
    firewall.attest_preterminal(_hash("preterminal"))
    firewall.open_terminal()
    terminal = firewall.open_terminal_labels(
        center="0", keys=(("0", "case-2", "c"), ("0", "case-2", "d"))
    )
    return support, pseudo, terminal


def test_support_fit_entrypoints_accept_only_authenticated_exact_capability() -> None:
    surface = _surface()
    fingerprint = build_fingerprint_surface(
        surface,
        physical_surface_hash=_hash("physical"),
        control_id="IDENTITY",
    )
    support, pseudo, terminal = _capabilities()

    endpoint = build_case_endpoints(
        surface,
        physical_surface_hash=_hash("physical"),
        held_case_id="case-2",
        support_capability=support,  # type: ignore[arg-type]
    )
    model, prediction = fit_route_posterior(
        fingerprint,
        held_case_id="case-2",
        support_capability=support,  # type: ignore[arg-type]
    )
    assert endpoint.support_capability_hash == support.capability_hash  # type: ignore[union-attr]
    assert model.support_capability_hash == support.capability_hash  # type: ignore[union-attr]
    assert prediction.sample_ids == ("c", "d")

    fabricated_rows = (
        BinaryLabel("0", "case-1", "a", 0, support_scope("0", "case-2")),
        BinaryLabel("0", "case-1", "b", 1, support_scope("0", "case-2")),
    )
    with pytest.raises(ProtocolError, match="capability drifted"):
        SupportLabelCapability(
            "0",
            "case-2",
            fabricated_rows,
            support_scope("0", "case-2"),
        )

    for adversary in (fabricated_rows, pseudo, terminal):
        with pytest.raises(ProtocolError, match="requires a SupportLabelCapability"):
            build_case_endpoints(
                surface,
                physical_surface_hash=_hash("physical"),
                held_case_id="case-2",
                support_capability=adversary,  # type: ignore[arg-type]
            )
        with pytest.raises(ProtocolError, match="requires a SupportLabelCapability"):
            fit_route_posterior(
                fingerprint,
                held_case_id="case-2",
                support_capability=adversary,  # type: ignore[arg-type]
            )


def test_support_capability_is_bound_to_held_case_and_canonical_row_order() -> None:
    surface = _surface()
    fingerprint = build_fingerprint_surface(
        surface,
        physical_surface_hash=_hash("physical"),
        control_id="IDENTITY",
    )
    support, _pseudo, _terminal = _capabilities()
    with pytest.raises(ProtocolError, match="fit binding drifted"):
        fit_route_posterior(
            fingerprint,
            held_case_id="case-1",
            support_capability=support,  # type: ignore[arg-type]
        )

    values = {("0", "case-1", "a"): 0, ("0", "case-1", "b"): 1}

    def loader(keys: object, scope: str) -> tuple[BinaryLabel, ...]:
        return tuple(BinaryLabel(*key, values[key], scope) for key in keys)  # type: ignore[arg-type]

    firewall = LabelFirewall(loader)
    firewall.advance_support()
    reversed_support = firewall.open_support(
        center="0",
        held_case_id="case-2",
        keys=(("0", "case-1", "b"), ("0", "case-1", "a")),
    )
    with pytest.raises(ProtocolError, match="fit binding drifted"):
        build_case_endpoints(
            surface,
            physical_surface_hash=_hash("physical"),
            held_case_id="case-2",
            support_capability=reversed_support,
        )

