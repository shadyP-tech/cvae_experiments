from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.action_surface import (
    ActionDraft,
    ActionResponse,
    ResponseDenominators,
    RouteActionDraftSurface,
    build_route_action_draft_surface,
    probability_sha256,
    seal_action_surface,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.composition import (
    compose_center_prediction,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.contracts import (
    BankViability,
    FavorableUtility,
    RouteKey,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.endpoint_runtime import (
    EndpointPrediction,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.engine import (
    fit_outer_action_policy_surface,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.identity import (
    ACTION_FAMILIES,
    ACTION_STRATA,
    METHOD_MENU,
    P_METHOD_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.inventory import (
    ExpectedRouteInventory,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.label_firewall import (
    LabelPhase,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.lifecycle import (
    PDCAPSLabelLifecycle,
    PreterminalOutputHashes,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.legacy_control import (
    LegacyControlSeal,
    seal_legacy_control,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.route_support import (
    BinaryLabel,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.routing import (
    build_admission_from_pseudo_policies,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.target_local_runtime import (
    CasePosteriorPrediction,
)
from midogpp_thesis.cvae.protocol import ProtocolError


CENTERS = ("0", "1", "2", "3", "5", "6", "7", "8", "9")


def _hash(value: object) -> str:
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


def _viability(value: object) -> BankViability:
    return BankViability(
        True,
        True,
        (("0", 30.0), ("1", 30.0)),
        5.0,
        _hash((value, "viability")),
    )


def _route(*, outer: str, center: str, target: bool) -> RouteKey:
    return RouteKey(
        "target" if target else "pseudo",
        outer,
        outer if target else center,
        f"case-{center}",
        outer,
        None if target else center,
        _hash((outer, center, target, "fit")),
    )


def _draft_route(*, outer: str, center: str, target: bool) -> RouteActionDraftSurface:
    route = _route(outer=outer, center=center, target=target)
    baseline = np.asarray([0.2, 0.8], dtype=np.float32)
    drafts = []
    center_index = CENTERS.index(center)
    for stratum_index, (family, direction) in enumerate(ACTION_STRATA):
        action = (
            np.asarray([0.7, 0.8], dtype=np.float32)
            if direction == "zero_to_one"
            else np.asarray([0.2, 0.3], dtype=np.float32)
        )
        level = 0.01 * (center_index + 1) + 0.001 * (stratum_index + 1)
        drafts.append(
            ActionDraft(
                route,
                family,
                direction,
                f"{family}::{direction}",
                action,
                FavorableUtility(level, level / 2.0, level / 3.0),
                0.5,
                _viability((center, family, direction)),
                _hash((center, "endpoint")),
                _hash((center, "posterior")),
            )
        )
    return RouteActionDraftSurface(
        route,
        (f"{center}-a", f"{center}-b"),
        baseline,
        tuple(drafts),
        _hash((center, "endpoint")),
        _hash((center, "posterior")),
        _hash("physical"),
        "IDENTITY",
    )


def test_probability_bridge_keeps_only_nonempty_directional_crossings() -> None:
    route = _route(outer="0", center="0", target=True)
    sample_ids = ("0-a", "0-b")
    endpoint = EndpointPrediction(
        "0",
        "case-0",
        sample_ids,
        (
            ("B", np.asarray([0.7, 0.3], dtype=np.float32)),
            ("I_OPPORTUNITY_GATED", np.asarray([0.2, 0.8], dtype=np.float32)),
            ("R_NINE_ARM_ROBUST", np.asarray([0.7, 0.3], dtype=np.float32)),
            ("P_PROTECTED", np.asarray([0.2, 0.8], dtype=np.float32)),
        ),
        ("support-a", "support-b"),
        _hash("support-capability"),
        (),
        _hash("physical"),
        _hash("center"),
    )
    posterior = CasePosteriorPrediction(
        "0",
        "case-0",
        "IDENTITY",
        sample_ids,
        np.asarray([0.8, 0.2], dtype=np.float64),
        _hash("posterior-model"),
        _hash("fingerprint"),
    )
    surface = build_route_action_draft_surface(
        endpoint,
        posterior,
        route,
        support_n_positive=10,
        support_n_negative=10,
        bank_viability_by_family={family: _viability(family) for family in ACTION_FAMILIES},
    )
    assert tuple((row.family, row.direction) for row in surface.drafts) == (
        ("B", "one_to_zero"),
        ("B", "zero_to_one"),
        ("R_NINE_ARM_ROBUST", "one_to_zero"),
        ("R_NINE_ARM_ROBUST", "zero_to_one"),
    )
    assert surface.posterior_control_id == "IDENTITY"
    sealed = seal_action_surface((surface,), expected_outer_centers=None)
    assert sealed.posterior_control_id == "IDENTITY"
    assert len(sealed.predictions) == 4
    assert all(
        row.key.action_surface_seal_hash == sealed.action_surface_seal_hash
        for row in sealed.predictions
    )


def test_nested_engine_excludes_h_j_k_and_reuses_symmetric_numeric_fits() -> None:
    outer = "0"
    draft_routes = [_draft_route(outer=outer, center=outer, target=True)]
    draft_routes.extend(
        _draft_route(outer=outer, center=center, target=False)
        for center in CENTERS
        if center != outer
    )
    sealed = seal_action_surface(draft_routes, expected_outer_centers=(outer,))
    assert sealed.physical_surface_hash == _hash("physical")
    with pytest.raises(ProtocolError, match="inventory drifted"):
        seal_action_surface(
            (
                *draft_routes[:-1],
                replace(
                    draft_routes[-1], physical_surface_hash=_hash("foreign")
                ),
            ),
            expected_outer_centers=(outer,),
        )
    with pytest.raises(ProtocolError, match="inventory drifted"):
        seal_action_surface(
            (
                *draft_routes[:-1],
                replace(
                    draft_routes[-1],
                    posterior_control_id="WITHIN_CASE_CYCLIC_SHIFT",
                ),
            ),
            expected_outer_centers=(outer,),
        )
    responses = []
    for prediction in sealed.predictions:
        route = prediction.key.route_key
        if route.surface_role != "pseudo":
            continue
        center_index = CENTERS.index(route.route_center)
        stratum_index = ACTION_STRATA.index(prediction.key.stratum)
        level = 0.012 * (center_index + 1) + 0.001 * (stratum_index + 1)
        responses.append(
            ActionResponse(
                prediction.key,
                prediction.prediction_hash,
                FavorableUtility(level, level / 2.0, level / 3.0),
                2,
                10,
                10,
                20,
                _hash("P"),
                _hash((route.route_center, "rows")),
            )
        )
    result = fit_outer_action_policy_surface(
        sealed,
        responses,
        outer_center=outer,
    )
    assert result.calibration_families.numerical_metric_fit_count == 111
    assert result.policy_calibration_families.numerical_metric_fit_count == 111
    assert result.calibration_families.serialized_model_count == 195
    assert result.policy_calibration_families.serialized_model_count == 195
    for context, rows in result.calibration_families.pseudo_reliability_oof_by_context:
        for scored, models in rows:
            assert context not in models[0].training_centers
            assert scored not in models[0].training_centers
            assert outer not in models[0].training_centers
    for context, envelope in result.policy_calibration_families.pseudo_envelopes_by_center:
        assert envelope.excluded_scored_center == context
        assert context not in {center for center, _value in envelope.center_means}
    assert result.target_policy_surface.responses_available is False
    assert all(row.responses_available for row in result.pseudo_policy_response_surfaces)
    assert len(result.result_hash) == 64
    assert result.physical_surface_hash == sealed.physical_surface_hash
    assert result.posterior_control_id == "IDENTITY"

    legacy_control = seal_legacy_control(result)
    assert seal_legacy_control(result) == legacy_control
    assert legacy_control.to_payload()["target_labels_used"] is False
    legacy_target = legacy_control.surface.target_decision
    assert legacy_target.selected_action_hashes == (
        result.target_policy_surface.cells[legacy_target.selected_k]
        .ordered_action_hashes
    )
    assert legacy_target.matched_pseudo_decision_hashes == tuple(
        row.decision_hash for row in legacy_control.surface.decisions
    )
    assert all(
        row.decision.realized_utility
        == result.pseudo_policy_response_surfaces[index].cells[
            row.decision.selected_k
        ].realized_utility
        for index, row in enumerate(legacy_control.references)
    )
    admission = build_admission_from_pseudo_policies(result, legacy_control)
    assert admission.outer_center == outer
    assert admission.target_labels_opened is False
    assert (
        build_admission_from_pseudo_policies(result, legacy_control.references)
        == admission
    )

    for field_name, value in (
        ("physical_surface_hash", _hash("historical-physical")),
        ("action_surface_seal_hash", _hash("historical-action")),
    ):
        drifted = LegacyControlSeal(
            replace(legacy_control.surface, **{field_name: value})
        )
        with pytest.raises(ProtocolError, match="same-run"):
            build_admission_from_pseudo_policies(result, drifted)

    references = legacy_control.references
    with pytest.raises(ProtocolError, match="inventory"):
        build_admission_from_pseudo_policies(result, references[:-1])

    opaque = (
        replace(
            references[0],
            legacy_control_seal_hash=_hash("invented-opaque-seal"),
        ),
        *references[1:],
    )
    with pytest.raises(ProtocolError, match="reference"):
        build_admission_from_pseudo_policies(result, opaque)

    foreign_decision = replace(
        references[0].decision,
        pseudo_response_surface_hash=_hash("historical-response"),
    )
    foreign = (replace(references[0], decision=foreign_decision), *references[1:])
    with pytest.raises(ProtocolError, match="reference|same-run|surface"):
        build_admission_from_pseudo_policies(result, foreign)

    original = references[0].decision
    fabricated_utility = FavorableUtility(
        original.realized_utility.bacc_gain + 0.001,
        original.realized_utility.brier_gain,
        original.realized_utility.log_gain,
    )
    fabricated_decision = replace(
        original,
        realized_utility=fabricated_utility,
        jointly_safe=bool(
            original.selected_k > 0
            and fabricated_utility.bacc_gain > 0.0
            and fabricated_utility.brier_gain >= 0.0
            and fabricated_utility.log_gain >= 0.0
        ),
        absolute_oracle_regret=abs(
            original.endpoint_oracle_bacc_gain - fabricated_utility.bacc_gain
        ),
    )
    fabricated = (
        replace(references[0], decision=fabricated_decision),
        *references[1:],
    )
    with pytest.raises(ProtocolError, match="reference|decision|surface"):
        build_admission_from_pseudo_policies(result, fabricated)

    fabricated_target = replace(
        legacy_control.target_decision,
        reason="FABRICATED_TARGET_CONTROL_LINEAGE",
    )
    drifted_target = LegacyControlSeal(
        replace(legacy_control.surface, target_decision=fabricated_target)
    )
    with pytest.raises(ProtocolError, match="decision"):
        build_admission_from_pseudo_policies(result, drifted_target)


def test_composition_returns_byte_exact_p_when_admission_fails() -> None:
    route = _draft_route(outer="0", center="0", target=True)
    sealed = seal_action_surface((route,), expected_outer_centers=("0",))
    target_route = sealed.routes[0]
    chosen = target_route.cells[0].prediction.key.action_key_hash
    composed = compose_center_prediction(
        (target_route,),
        center_sample_order=target_route.sample_ids,
        selected_action_hashes=(chosen,),
        method_id=P_METHOD_ID,
        selection_enabled=False,
    )
    assert composed.selected_action_hashes == ()
    assert np.array_equal(composed.probabilities, target_route.baseline_probabilities)
    assert probability_sha256(composed.probabilities) == composed.protected_probability_hash


def test_lifecycle_poisoned_target_labels_cannot_change_preterminal_seal() -> None:
    target_0 = _draft_route(outer="0", center="0", target=True)
    pseudo_0_1 = _draft_route(outer="0", center="1", target=False)
    target_1 = _draft_route(outer="1", center="1", target=True)
    pseudo_1_0 = _draft_route(outer="1", center="0", target=False)
    routes = (target_0, pseudo_0_1, target_1, pseudo_1_0)
    inventory = ExpectedRouteInventory.focused_fixture(
        (
            *(('0', 'case-0', sample) for sample in target_0.sample_ids),
            *(('1', 'case-1', sample) for sample in target_1.sample_ids),
        )
    )

    def run_to_attestation(
        poisoned_target: tuple[int, int],
    ) -> tuple[str, PDCAPSLabelLifecycle, list[str]]:
        opened_scopes: list[str] = []

        def loader(keys: object, scope: str) -> tuple[BinaryLabel, ...]:
            opened_scopes.append(scope)
            key_rows = tuple(keys)  # type: ignore[arg-type]
            values = poisoned_target if scope.startswith("TERMINAL::") else (0, 1)
            return tuple(
                BinaryLabel(*key, value, scope)
                for key, value in zip(key_rows, values, strict=True)
            )

        lifecycle = PDCAPSLabelLifecycle(
            loader,
            protocol_hash=_hash("protocol"),
            expected_inventory=inventory,
        )
        lifecycle.begin_support()
        cyclic_routes = tuple(
            replace(
                route,
                drafts=tuple(
                    replace(
                        draft,
                        posterior_prediction_hash=_hash(
                            (
                                route.route_key.outer_center,
                                route.route_key.held_case_id,
                                "cyclic-posterior",
                            )
                        ),
                    )
                    for draft in route.drafts
                ),
                posterior_prediction_hash=_hash(
                    (
                        route.route_key.outer_center,
                        route.route_key.held_case_id,
                        "cyclic-posterior",
                    )
                ),
                posterior_control_id="WITHIN_CASE_CYCLIC_SHIFT",
            )
            for route in routes
        )
        surface = lifecycle.seal_actions(
            routes, cyclic_control_routes=cyclic_routes
        )
        lifecycle.begin_pseudo_responses()

        with pytest.raises(ProtocolError, match="target scope"):
            lifecycle.open_pseudo_action_responses(
                target_0.route_key,
                denominators=ResponseDenominators(1, 1),
            )
        lifecycle.open_pseudo_action_responses(
            pseudo_0_1.route_key,
            denominators=ResponseDenominators(1, 1),
        )
        lifecycle.open_pseudo_action_responses(
            pseudo_1_0.route_key,
            denominators=ResponseDenominators(1, 1),
        )
        with pytest.raises(ProtocolError, match="terminal-only"):
            lifecycle.open_terminal_center_labels("0")

        surface_set = lifecycle.action_surface_set
        centers = inventory.centers
        outputs = PreterminalOutputHashes(
            surface_set.surface_set_seal_hash,
            tuple(
                (row.posterior_control_id, row.action_surface_seal_hash)
                for row in surface_set.surfaces
            ),
            inventory.inventory_hash,
            tuple(
                (control, center, _hash((control, center, "outer-result")))
                for control in surface_set.control_ids
                for center in centers
            ),
            tuple(
                (control, center, _hash((control, center, "legacy-control")))
                for control in surface_set.control_ids
                for center in centers
            ),
            tuple(
                (center, method, _hash((center, method, "decision")))
                for center in centers
                for method in METHOD_MENU
            ),
            tuple(
                (center, method, _hash((center, method, "composition")))
                for center in centers
                for method in METHOD_MENU
            ),
        )
        seal = lifecycle.attest_preterminal(outputs)
        assert lifecycle.phase is LabelPhase.PRETERMINAL_ATTESTED
        assert not any(scope.startswith("TERMINAL::") for scope in opened_scopes)
        return str(seal["seal_hash"]), lifecycle, opened_scopes

    zero_seal, zero_lifecycle, zero_scopes = run_to_attestation((0, 0))
    one_seal, _one_lifecycle, one_scopes = run_to_attestation((1, 1))
    assert zero_seal == one_seal
    assert zero_scopes == one_scopes

    zero_lifecycle.begin_terminal_evaluation()
    terminal = zero_lifecycle.open_terminal_center_labels("0")
    assert terminal.values == (0, 0)
    assert zero_lifecycle.phase is LabelPhase.TERMINAL
    audit = zero_lifecycle.audit_payload()
    assert audit["target_labels_can_change_preterminal_decisions"] is False
    assert audit["publication_status"] == "POST_HOC_CONSUMED_TEST_SENSITIVITY"
    assert audit["terminal_decision"] == "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"
