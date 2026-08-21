from __future__ import annotations

import ast
from collections import OrderedDict
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router.constants import (
    CENTERS,
    EXPECTED_DOUBLE_EXCLUSION_PAIR_COUNT,
    EXPECTED_LEGACY_UTILITY_MODEL_FIT_COUNT,
    EXPECTED_PARC_MODEL_FIT_COUNT_PER_GEOMETRY,
    EXPECTED_POLICY_REPLAY_COUNT,
    EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT,
    EXPECTED_UTILITY_MODEL_FIT_COUNT,
    LEGACY_GEOMETRY_ID,
    PORTFOLIO_METHOD_ID,
    PRIMARY_METHOD_ID,
    PROJECTION_GEOMETRY_ID,
    UNPROJECTED_GEOMETRY_ID,
    UNPROJECTED_PARC_METHOD_ID,
    UTILITY_FEATURE_NAMES,
    UTILITY_RESPONSE_IDS,
    physical_action_ids,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router.contracts import (
    CenterProbabilitySurface,
    EndpointCasePrediction,
    PhysicalProbabilitySurface,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router.engine import (
    assert_transport_contract_executable,
    build_preterminal_result,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router import (
    engine as pcsi_engine,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router.hashing import (
    canonical_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router.policy_regret import (
    WholePolicyReplay,
    authorize_center_policy,
    build_center_candidate_policy,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router.label_capabilities import (
    PCSIPARCLabelFirewall,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router.outer_plans import (
    build_outer_plans,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router.policy_selection import (
    select_and_compose_case_policy,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router.projected_contracts import (
    ProjectedDonorUtilityRow,
    ProjectedUtilityPrediction,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router.projected_features import (
    build_projected_descriptors,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router.projected_model import (
    fit_projected_model_family,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router.projected_uncertainty import (
    predict_projected_surface,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router.projection import (
    build_action_equivalence_classes,
    emit_directional_action,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router.projection_lattice import (
    THRESHOLD,
    THRESHOLD_PREDECESSOR,
    canonical_bytes,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router.sample_influence_contracts import (
    InfluencePrediction,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router.transport import (
    TRANSPORT_PROTOCOL_CONTRACT,
    TransportEndpointLineage,
    TransportRuntimeSeal,
    TransportScreen,
)
from midogpp_thesis.cvae.protocol import ProtocolError


PACKAGE = Path(
    "src/midogpp_thesis/cvae/diagnostics/"
    "fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router"
)


def _endpoint(center: str = "0") -> EndpointCasePrediction:
    return EndpointCasePrediction(
        center,
        f"case-{center}",
        ("s0", "s1", "s2", "s3"),
        MappingProxyType(
            {
                "B": (0.90, 0.10, 0.30, 0.70),
                "I_OPPORTUNITY_GATED": (0.80, 0.20, 0.25, 0.75),
                "R_NINE_ARM_ROBUST": (0.30, 0.70, 0.25, 0.75),
                "P_PROTECTED": (0.40, 0.60, 0.20, 0.80),
            }
        ),
        "a" * 64,
    )


def _three_distinct_crossings_endpoint() -> EndpointCasePrediction:
    return EndpointCasePrediction(
        "0",
        "tie-case",
        ("s0", "s1", "s2", "s3", "s4", "s5"),
        MappingProxyType(
            {
                "B": (0.9, 0.4, 0.4, 0.1, 0.6, 0.6),
                "I_OPPORTUNITY_GATED": (0.4, 0.9, 0.4, 0.6, 0.1, 0.6),
                "R_NINE_ARM_ROBUST": (0.4, 0.4, 0.9, 0.6, 0.6, 0.1),
                "P_PROTECTED": (0.4, 0.4, 0.4, 0.6, 0.6, 0.6),
            }
        ),
        "a" * 64,
    )


def _training_rows(outer: str = "0") -> tuple[ProjectedDonorUtilityRow, ...]:
    rows: list[ProjectedDonorUtilityRow] = []
    for donor_index, donor in enumerate(center for center in CENTERS if center != outer):
        for case_index in range(2):
            for direction_index, direction in enumerate(("zero_to_one", "one_to_zero")):
                for class_index in range(1 + case_index):
                    signal = donor_index / 20.0 + case_index / 10.0
                    features = tuple(
                        signal + direction_index / 50.0 + class_index / 100.0 + index / 200.0
                        for index in range(len(UTILITY_FEATURE_NAMES))
                    )
                    rows.append(
                        ProjectedDonorUtilityRow(
                            outer,
                            donor,
                            f"{donor}-case-{case_index}",
                            PROJECTION_GEOMETRY_ID,
                            direction,
                            "B",
                            features,
                            1,
                            0.01 + signal,
                            -0.02 - signal,
                            -0.03 - signal,
                            f"{len(rows) + 1:064x}",
                        )
                    )
    return tuple(rows)


def _utility_prediction(descriptor_hash: str, *, bacc: float) -> ProjectedUtilityPrediction:
    robust = (
        ("bacc_contribution_delta", bacc),
        ("brier_contribution_delta", -0.20),
        ("log_loss_contribution_delta", -0.20),
    )
    deletion = tuple(
        (response, (("1", value),)) for response, value in robust
    )
    zeros = tuple((response, 0.0) for response in UTILITY_RESPONSE_IDS)
    return ProjectedUtilityPrediction(
        descriptor_hash,
        PROJECTION_GEOMETRY_ID,
        robust,
        deletion,
        zeros,
        zeros,
        robust,
        ("b" * 64,),
    )


def _transport(candidate: str, *, passed: bool) -> TransportScreen:
    references = tuple(center for center in CENTERS if center != candidate)
    distance = 1.0 if passed else 2.0
    threshold = 1.0
    return TransportScreen(
        "0",
        candidate,
        references,
        distance,
        threshold,
        tuple((center, threshold) for center in references),
        passed,
        "c" * 64,
        tuple("d" * 64 for _center in references),
        "e" * 64,
        tuple("f" * 64 for _center in references),
    )


def _small_plan_firewall() -> PCSIPARCLabelFirewall:
    identities = tuple(
        SimpleNamespace(
            center=center,
            case_id=f"{center}-case-{case_index}",
            group_id=f"{center}-case-{case_index}",
            sample_id=f"{center}-sample-{case_index}",
        )
        for center in CENTERS
        for case_index in range(2)
    )
    plans = build_outer_plans(
        identities,
        probability_surface_hash="f" * 64,
        strict_canonical_topology=False,
    )

    def loader(allowed: frozenset[tuple[str, str, str]], _role: str) -> tuple[object, ...]:
        return tuple(
            SimpleNamespace(center=center, case_id=case, sample_id=sample, value=0)
            for center, case, sample in sorted(allowed)
        )

    return PCSIPARCLabelFirewall(plans, loader)


def _small_end_to_end_surface() -> tuple[
    PhysicalProbabilitySurface, dict[tuple[str, str, str], int]
]:
    centers: OrderedDict[str, CenterProbabilitySurface] = OrderedDict()
    truth: dict[tuple[str, str, str], int] = {}
    for center_index, center in enumerate(CENTERS):
        samples: list[str] = []
        cases: list[str] = []
        for case_index in range(3):
            for label in range(2):
                case = f"{center}-c{case_index}"
                sample = f"{case}-y{label}"
                samples.append(sample)
                cases.append(case)
                truth[(center, case, sample)] = label
        arrays: OrderedDict[str, np.ndarray] = OrderedDict()
        for action_index, action in enumerate(physical_action_ids(center)):
            base = np.asarray(
                [0.35 if index % 2 == 0 else 0.65 for index in range(len(samples))],
                dtype=np.float32,
            )
            if action_index >= 2:
                base = np.asarray(
                    [
                        (
                            0.78
                            if (index + action_index + center_index) % 4 == 0
                            else 0.22
                        )
                        if index % 3 != 2
                        else base[index]
                        for index in range(len(samples))
                    ],
                    dtype=np.float32,
                )
            arrays[action] = np.stack(
                [
                    np.clip(base + np.float32((seed - 4) * 0.003), 0.0, 1.0)
                    for seed in range(9)
                ]
            ).astype(np.float32)
        centers[center] = CenterProbabilitySurface(
            center,
            tuple(samples),
            tuple(cases),
            arrays,
            "a" * 64,
        )
    return PhysicalProbabilitySurface(
        centers,
        "a" * 64,
        strict_canonical_topology=False,
    ), truth


def test_workload_arithmetic_is_frozen_for_the_workstation() -> None:
    assert EXPECTED_LEGACY_UTILITY_MODEL_FIT_COUNT == 81
    assert EXPECTED_PARC_MODEL_FIT_COUNT_PER_GEOMETRY == 657
    assert EXPECTED_UTILITY_MODEL_FIT_COUNT == 1_395
    assert EXPECTED_DOUBLE_EXCLUSION_PAIR_COUNT == 72
    assert EXPECTED_POLICY_REPLAY_COUNT == 144
    assert EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT == 436


def test_canonical_runtime_fails_closed_on_transport_label_claim() -> None:
    audited = SimpleNamespace(payload=dict(TRANSPORT_PROTOCOL_CONTRACT))
    with pytest.raises(
        ProtocolError,
        match="identity-level route noninterference is false.*authorization is invalid",
    ):
        assert_transport_contract_executable(
            audited,  # type: ignore[arg-type]
            strict_canonical_topology=True,
        )

    legacy = SimpleNamespace(
        payload={**dict(TRANSPORT_PROTOCOL_CONTRACT), "transport_uses_labels": False}
    )
    with pytest.raises(ProtocolError, match="not the exact audited"):
        assert_transport_contract_executable(
            legacy,  # type: ignore[arg-type]
            strict_canonical_topology=True,
        )


def test_HJ_pseudo_rows_models_and_predictions_exclude_J_source_prior_poison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pcsi_engine,
        "assert_transport_contract_executable",
        lambda _protocol, *, strict_canonical_topology: None,
    )
    surface, truth = _small_end_to_end_surface()

    def run(*, poison_center_one_source_prior: bool):
        def loader(
            allowed: frozenset[tuple[str, str, str]], role: str
        ) -> tuple[object, ...]:
            return tuple(
                SimpleNamespace(
                    center=center,
                    case_id=case,
                    sample_id=sample,
                    value=(
                        1 - truth[(center, case, sample)]
                        if poison_center_one_source_prior
                        and role.startswith("source_prior::")
                        and center == "1"
                        else truth[(center, case, sample)]
                    ),
                )
                for center, case, sample in sorted(allowed)
            )

        return build_preterminal_result(surface, loader, use_processes=False)

    clean = run(poison_center_one_source_prior=False)
    poisoned = run(poison_center_one_source_prior=True)
    outer, pseudo = "0", "1"
    expected_triple_count = len(CENTERS) * (len(CENTERS) - 1) * (
        len(CENTERS) - 2
    )
    assert len(clean.donor_runtime.pseudo_prior_provenance) == expected_triple_count
    assert (
        len(clean.donor_runtime.pseudo_donor_endpoint_products)
        == expected_triple_count
    )

    def donor_row_hash(rows: tuple[ProjectedDonorUtilityRow, ...]) -> str:
        return canonical_hash([row.to_payload() for row in rows])

    # The poison is active: actual H/K rows are allowed to use J as a q center.
    assert donor_row_hash(
        clean.donor_runtime.geometry_results[
            PROJECTION_GEOMETRY_ID
        ].donor_rows_by_outer[outer]
    ) != donor_row_hash(
        poisoned.donor_runtime.geometry_results[
            PROJECTION_GEOMETRY_ID
        ].donor_rows_by_outer[outer]
    )

    training = tuple(
        center for center in CENTERS if center not in {outer, pseudo}
    )
    for donor in training:
        key = outer, pseudo, donor
        clean_prior = clean.donor_runtime.pseudo_prior_provenance[key]
        poisoned_prior = poisoned.donor_runtime.pseudo_prior_provenance[key]
        assert clean_prior.prior_hash == poisoned_prior.prior_hash
        for source, queries in clean_prior.query_centers_by_source:
            assert queries == tuple(
                center
                for center in CENTERS
                if center not in {outer, pseudo, donor, source}
            )
        clean_endpoint = clean.donor_runtime.pseudo_donor_endpoint_products[key]
        poisoned_endpoint = poisoned.donor_runtime.pseudo_donor_endpoint_products[key]
        assert clean_endpoint.endpoint_model_fit_count == 0
        assert poisoned_endpoint.endpoint_model_fit_count == 0
        assert clean_endpoint.state_hashes == poisoned_endpoint.state_hashes
        assert tuple(
            row.prediction_hash for row in clean_endpoint.predictions
        ) == tuple(row.prediction_hash for row in poisoned_endpoint.predictions)

    for geometry_id in (PROJECTION_GEOMETRY_ID, UNPROJECTED_GEOMETRY_ID):
        clean_geometry = clean.donor_runtime.geometry_results[geometry_id]
        poisoned_geometry = poisoned.donor_runtime.geometry_results[geometry_id]
        pair = outer, pseudo
        assert donor_row_hash(
            clean_geometry.pseudo_donor_rows_by_pair[pair]
        ) == donor_row_hash(poisoned_geometry.pseudo_donor_rows_by_pair[pair])
        assert tuple(
            row.descriptor_hash
            for row in clean_geometry.pseudo_descriptors_by_pair[pair]
        ) == tuple(
            row.descriptor_hash
            for row in poisoned_geometry.pseudo_descriptors_by_pair[pair]
        )
        assert clean_geometry.pseudo_full_models[pair].training_centers == training
        assert clean_geometry.pseudo_full_models[pair].model_hash == (
            poisoned_geometry.pseudo_full_models[pair].model_hash
        )
        assert tuple(clean_geometry.pseudo_delete_models[pair]) == training
        assert tuple(
            model.model_hash
            for model in clean_geometry.pseudo_delete_models[pair].values()
        ) == tuple(
            model.model_hash
            for model in poisoned_geometry.pseudo_delete_models[pair].values()
        )
        assert tuple(
            row.prediction_hash
            for row in clean_geometry.pseudo_predictions_by_pair[pair]
        ) == tuple(
            row.prediction_hash
            for row in poisoned_geometry.pseudo_predictions_by_pair[pair]
        )


def test_transport_identity_feedback_poison_is_detected_and_blocks_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Canonical execution is intentionally blocked. This bypass exists only to
    # reconstruct and seal the negative identity-level poison evidence.
    monkeypatch.setattr(
        pcsi_engine,
        "assert_transport_contract_executable",
        lambda _protocol, *, strict_canonical_topology: None,
    )
    surface, truth = _small_end_to_end_surface()

    def run(poison: tuple[str, str] | None):
        def loader(
            allowed: frozenset[tuple[str, str, str]], _role: str
        ) -> tuple[object, ...]:
            return tuple(
                SimpleNamespace(
                    center=center,
                    case_id=case,
                    sample_id=sample,
                    value=(
                        1 - truth[(center, case, sample)]
                        if poison == (center, case)
                        else truth[(center, case, sample)]
                    ),
                )
                for center, case, sample in sorted(allowed)
            )

        return build_preterminal_result(surface, loader, use_processes=False)

    clean = run(None)
    held_case_poison = run(("0", "0-c0"))
    pseudo_case_poison = run(("1", "1-c0"))
    runtime = clean.policy_runtime
    seal = runtime.transport_seal
    assert isinstance(seal, TransportRuntimeSeal)
    assert seal.descriptor_count == len(CENTERS) ** 2 == 81
    assert seal.screen_count == len(CENTERS) ** 2 == 81
    assert runtime.transport_hash == seal.transport_hash
    seal_payload = seal.to_payload()
    assert seal_payload["identity_level_route_noninterference_required"] is True
    assert seal_payload["identity_level_route_noninterference_proven"] is False
    assert seal_payload["authorization_valid"] is False
    assert (
        seal_payload["protocol_status"]
        == "BLOCKED_IDENTITY_LEVEL_ROUTE_FEEDBACK"
    )

    descriptor = runtime.transport_descriptors_by_outer_candidate[("0", "0")]
    assert isinstance(descriptor.lineage, TransportEndpointLineage)
    descriptor_payload = descriptor.to_payload()
    assert "labels_used" not in descriptor_payload
    lineage_payload = descriptor_payload["transport_lineage"]
    assert lineage_payload["endpoint_support_scope"] == (
        "endpoint_target_T_minus_held_case_c"
    )
    assert lineage_payload["identity_level_route_noninterference_proven"] is False
    screen_payload = runtime.transport_screens[("0", None)].to_payload()
    assert "labels_used" not in screen_payload
    assert screen_payload["authorization_valid"] is False

    assert descriptor.descriptor_hash != (
        held_case_poison.policy_runtime.transport_descriptors_by_outer_candidate[
            ("0", "0")
        ].descriptor_hash
    )
    assert runtime.transport_screens[("0", None)].screen_hash != (
        held_case_poison.policy_runtime.transport_screens[("0", None)].screen_hash
    )
    affected_target_authorizations = {
        key
        for key in runtime.authorizations
        if key[1] == "0"
        and runtime.authorizations[key].authorization_hash
        != held_case_poison.policy_runtime.authorizations[key].authorization_hash
    }
    assert affected_target_authorizations == {
        (PRIMARY_METHOD_ID, "0"),
        (UNPROJECTED_PARC_METHOD_ID, "0"),
    }

    def final_hash(preterminal: object, policy: str) -> str:
        rows = preterminal.policy_runtime.final_predictions_by_policy[policy]
        return next(
            row.prediction_hash
            for row in rows
            if row.target_center == "0" and row.case_id == "0-c0"
        )

    assert all(
        final_hash(clean, policy) != final_hash(held_case_poison, policy)
        for policy in (PRIMARY_METHOD_ID, UNPROJECTED_PARC_METHOD_ID)
    )
    assert runtime.transport_descriptors_by_outer_candidate[
        ("0", "1")
    ].descriptor_hash != (
        pseudo_case_poison.policy_runtime.transport_descriptors_by_outer_candidate[
            ("0", "1")
        ].descriptor_hash
    )
    assert runtime.transport_screens[("0", "1")].screen_hash != (
        pseudo_case_poison.policy_runtime.transport_screens[("0", "1")].screen_hash
    )
    assert all(
        runtime.authorizations[(policy, "0")].authorization_hash
        != pseudo_case_poison.policy_runtime.authorizations[
            (policy, "0")
        ].authorization_hash
        for policy in (PRIMARY_METHOD_ID, UNPROJECTED_PARC_METHOD_ID)
    )


def test_science_package_has_no_predecessor_semantic_imports() -> None:
    forbidden = {
        "fixed_bank_p_anchored_crossfit_sample_influence_router",
        "fixed_bank_p_anchored_simultaneous_shift_calibrated_utility_router",
        "fixed_bank_loo_nested_donor_endpoint_regret_router",
        "fixed_bank_loo_directional_shrinkage_ensemble",
    }
    for path in PACKAGE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        assert not any(fragment in module for fragment in forbidden for module in modules)


def test_binary32_projection_is_minimal_and_preserves_off_mask_P_bytes() -> None:
    endpoint = _endpoint()
    projected, mask = emit_directional_action(
        endpoint,
        "B",
        "zero_to_one",
        geometry_id=PROJECTION_GEOMETRY_ID,
    )
    portfolio = np.asarray(endpoint.probabilities["P_PROTECTED"], dtype=np.float32)
    assert THRESHOLD == np.float32(0.5)
    assert THRESHOLD_PREDECESSOR == np.nextafter(
        np.float32(0.5), np.float32(0.0), dtype=np.float32
    )
    assert projected[mask].tolist() == [float(THRESHOLD)]
    assert canonical_bytes(projected[~mask]) == canonical_bytes(portfolio[~mask])

    down, down_mask = emit_directional_action(
        endpoint,
        "B",
        "one_to_zero",
        geometry_id=PROJECTION_GEOMETRY_ID,
    )
    assert down[down_mask].tolist() == [float(THRESHOLD_PREDECESSOR)]
    assert np.all((down >= THRESHOLD)[down_mask] != (portfolio >= THRESHOLD)[down_mask])


def test_projection_collapses_complete_vectors_before_feature_construction() -> None:
    endpoint = _endpoint()
    projected = build_action_equivalence_classes(
        endpoint, geometry_id=PROJECTION_GEOMETRY_ID
    )
    raw = build_action_equivalence_classes(
        endpoint, geometry_id=UNPROJECTED_GEOMETRY_ID
    )
    descriptors = build_projected_descriptors(endpoint, projected)

    for direction in ("zero_to_one", "one_to_zero"):
        rows = [row for row in projected if row.direction == direction]
        assert len(rows) == 2
        assert rows[0].members in (("B", "I_OPPORTUNITY_GATED"), ("R_NINE_ARM_ROBUST",))
        assert any(row.structural_zero for row in rows)
    assert len(raw) == 6
    assert len(descriptors) == len(projected)
    assert {row.action_hash for row in descriptors} == {row.action_hash for row in projected}


def test_raw_and_legacy_actions_are_distinct_from_boundary_projection() -> None:
    endpoint = _endpoint()
    projected, mask = emit_directional_action(
        endpoint, "B", "zero_to_one", geometry_id=PROJECTION_GEOMETRY_ID
    )
    raw, _ = emit_directional_action(
        endpoint, "B", "zero_to_one", geometry_id=UNPROJECTED_GEOMETRY_ID
    )
    legacy, _ = emit_directional_action(
        endpoint, "B", "zero_to_one", geometry_id=LEGACY_GEOMETRY_ID
    )
    assert projected[mask].tolist() == [pytest.approx(0.5)]
    assert raw[mask].tolist() == [pytest.approx(0.9)]
    assert legacy[mask].tolist() == [pytest.approx(0.6)]
    assert len({canonical_bytes(projected), canonical_bytes(raw), canonical_bytes(legacy)}) == 3


def test_only_projected_geometry_collapses_byte_identical_structural_zero_cells() -> None:
    portfolio = (0.20, 0.80, 0.30, 0.70)
    endpoint = EndpointCasePrediction(
        "0",
        "case-0",
        ("s0", "s1", "s2", "s3"),
        MappingProxyType(
            {
                "B": portfolio,
                "I_OPPORTUNITY_GATED": portfolio,
                "R_NINE_ARM_ROBUST": portfolio,
                "P_PROTECTED": portfolio,
            }
        ),
        "a" * 64,
    )

    projected = build_action_equivalence_classes(
        endpoint, geometry_id=PROJECTION_GEOMETRY_ID
    )
    raw = build_action_equivalence_classes(
        endpoint, geometry_id=UNPROJECTED_GEOMETRY_ID
    )

    assert len(projected) == 2
    assert all(
        row.members
        == ("B", "I_OPPORTUNITY_GATED", "R_NINE_ARM_ROBUST")
        for row in projected
    )
    assert len(raw) == 6
    assert len({row.key for row in raw}) == 6
    assert all(len(row.members) == 1 and row.structural_zero for row in raw)


def test_joint_ridge_fits_actual_and_HJ_double_excluded_families() -> None:
    rows = _training_rows()
    actual_centers = tuple(center for center in CENTERS if center != "0")
    full, deleted = fit_projected_model_family(
        rows,
        outer_target_center="0",
        geometry_id=PROJECTION_GEOMETRY_ID,
        training_centers=actual_centers,
    )
    assert len(deleted) == 8
    assert len(full.direction_intercepts) == 3
    assert all(len(values) == 2 for _response, values in full.direction_intercepts)
    assert all(len(values) == 12 for _response, values in full.slope_coefficients)

    pseudo_centers = tuple(center for center in actual_centers if center != "1")
    pseudo_full, pseudo_deleted = fit_projected_model_family(
        rows,
        outer_target_center="0",
        geometry_id=PROJECTION_GEOMETRY_ID,
        training_centers=pseudo_centers,
    )
    assert len(pseudo_deleted) == 7
    assert "0" not in pseudo_full.training_centers
    assert "1" not in pseudo_full.training_centers

    endpoint = _endpoint("1")
    actions = build_action_equivalence_classes(
        endpoint, geometry_id=PROJECTION_GEOMETRY_ID
    )
    descriptors = build_projected_descriptors(endpoint, actions)
    predictions = predict_projected_surface(
        descriptors,
        donor_rows=rows,
        full_model=pseudo_full,
        delete_models=pseudo_deleted,
        candidate_center="1",
    )
    assert len(predictions) == len(descriptors)
    with pytest.raises(ProtocolError, match="actual/pseudo prediction scope"):
        predict_projected_surface(
            descriptors,
            donor_rows=rows,
            full_model=pseudo_full,
            delete_models=pseudo_deleted,
            candidate_center="0",
        )


def test_label_firewall_enforces_Hc_and_HJ_policy_seal_timeline() -> None:
    firewall = _small_plan_firewall()
    with pytest.raises(ProtocolError, match="prior and donor grants"):
        firewall.open_outer_support_labels(
            "0", "0-case-0", plan_hash=firewall._outer[("0", "0-case-0")].plan_hash
        )

    for heldout in CENTERS:
        for source in CENTERS:
            if source != heldout:
                labels = firewall.open_source_prior_labels(heldout, source)
                assert heldout not in {row.center for row in labels}
                assert source not in {row.center for row in labels}
    for outer in CENTERS:
        for donor in CENTERS:
            if donor != outer:
                labels = firewall.open_utility_donor_labels(outer, donor)
                assert {row.center for row in labels} == {donor}

    for center in CENTERS:
        for case_index in range(2):
            case = f"{center}-case-{case_index}"
            plan = firewall._outer[(center, case)]
            support = firewall.open_outer_support_labels(
                center, case, plan_hash=plan.plan_hash
            )
            assert case not in {row.case_id for row in support}
            firewall.record_outer_state_seal(center, case, "1" * 64)

    outer, pseudo = "0", "1"
    geometry = PROJECTION_GEOMETRY_ID
    double = firewall._double[(outer, pseudo)]
    assert outer not in double.model_training_centers
    assert pseudo not in double.model_training_centers
    with pytest.raises(ProtocolError, match="every target and pseudo policy seal"):
        firewall.open_pseudo_evaluation_labels(
            outer, pseudo, geometry, policy_seal_hash="2" * 64
        )

    for policy in (
        "PCSI_PARC_PROJECTED",
        "PCSI_PARC_PROJECTED_NO_POLICY_REGRET",
        "PCSI_PARC_UNPROJECTED",
        "PCSI_PARC_FRESH_LEGACY_DUAL_VETO",
        "PCSI_PARC_BLOCKED_FINGERPRINT",
    ):
        for center in CENTERS:
            for case_index in range(2):
                firewall.record_target_case_policy_seal(
                    center,
                    f"{center}-case-{case_index}",
                    policy,
                    "2" * 64,
                )
            firewall.record_target_center_policy_seal(center, policy, "2" * 64)

    firewall.record_pseudo_case_policy_seal(
        outer, pseudo, geometry, double.pseudo_case_ids[0], "3" * 64
    )
    with pytest.raises(ProtocolError, match="all per-case decisions"):
        firewall.record_pseudo_policy_seal(outer, pseudo, geometry, "4" * 64)
    for case in double.pseudo_case_ids[1:]:
        firewall.record_pseudo_case_policy_seal(
            outer, pseudo, geometry, case, "3" * 64
        )
    firewall.record_pseudo_policy_seal(outer, pseudo, geometry, "4" * 64)

    with pytest.raises(ProtocolError, match="every target and pseudo policy seal"):
        firewall.open_pseudo_evaluation_labels(
            outer, pseudo, geometry, policy_seal_hash="4" * 64
        )

    for candidate_geometry in (PROJECTION_GEOMETRY_ID, UNPROJECTED_GEOMETRY_ID):
        for candidate_outer in CENTERS:
            for candidate_pseudo in CENTERS:
                if candidate_pseudo == candidate_outer or (
                    candidate_geometry,
                    candidate_outer,
                    candidate_pseudo,
                ) == (geometry, outer, pseudo):
                    continue
                candidate_plan = firewall._double[(candidate_outer, candidate_pseudo)]
                for case in candidate_plan.pseudo_case_ids:
                    firewall.record_pseudo_case_policy_seal(
                        candidate_outer,
                        candidate_pseudo,
                        candidate_geometry,
                        case,
                        "3" * 64,
                    )
                firewall.record_pseudo_policy_seal(
                    candidate_outer,
                    candidate_pseudo,
                    candidate_geometry,
                    "4" * 64,
                )

    pseudo_labels = firewall.open_pseudo_evaluation_labels(
        outer, pseudo, geometry, policy_seal_hash="4" * 64
    )
    assert {row.center for row in pseudo_labels} == {pseudo}
    with pytest.raises(ProtocolError, match="aggregate seal"):
        firewall.open_terminal_labels()


def test_projected_selection_uses_influence_and_proper_losses_not_bacc_veto() -> None:
    endpoint = _endpoint()
    actions = build_action_equivalence_classes(
        endpoint, geometry_id=PROJECTION_GEOMETRY_ID
    )
    descriptors = build_projected_descriptors(endpoint, actions)
    influences = tuple(
        InfluencePrediction(
            row.descriptor_hash,
            row.target_center,
            row.case_id,
            row.representative,
            row.direction,
            row.crossing_count,
            0.1 if row.crossing_count else 0.0,
            "e" * 64,
        )
        for row in descriptors
    )
    utilities = tuple(
        _utility_prediction(row.descriptor_hash, bacc=-0.25) for row in descriptors
    )
    selected = select_and_compose_case_policy(
        endpoint,
        actions,
        descriptors,
        influences,
        utilities,
        policy_id=PRIMARY_METHOD_ID,
        require_positive_bacc_prediction=False,
    )
    assert selected.changed is True
    assert any(row.selected_action_hash is not None for row in selected.decisions)
    assert selected.predicted_favorable_endpoint_vector[0] < 0.0

    vetoed = select_and_compose_case_policy(
        endpoint,
        actions,
        descriptors,
        influences,
        utilities,
        policy_id=PRIMARY_METHOD_ID,
        require_positive_bacc_prediction=True,
    )
    assert vetoed.changed is False


@pytest.mark.parametrize(
    ("scores", "expected"),
    (
        ({"B": 1.0e-15}, PORTFOLIO_METHOD_ID),
        ({"B": np.nextafter(1.0e-15, np.inf)}, "B"),
        ({"B": 1.0e-13}, "B"),
        ({"B": 1.0e-13, "I_OPPORTUNITY_GATED": 6.0e-13}, "I_OPPORTUNITY_GATED"),
        ({"B": 1.0e-4, "I_OPPORTUNITY_GATED": 1.0e-4}, "B"),
    ),
)
def test_projected_selection_uses_exact_frozen_threshold_and_ties(
    scores: dict[str, float], expected: str
) -> None:
    endpoint = _three_distinct_crossings_endpoint()
    actions = build_action_equivalence_classes(
        endpoint, geometry_id=PROJECTION_GEOMETRY_ID
    )
    descriptors = build_projected_descriptors(endpoint, actions)
    influences = tuple(
        InfluencePrediction(
            row.descriptor_hash,
            row.target_center,
            row.case_id,
            row.representative,
            row.direction,
            row.crossing_count,
            (
                scores.get(row.representative, 0.0)
                if row.direction == "zero_to_one"
                else 0.0
            ),
            "e" * 64,
        )
        for row in descriptors
    )
    utilities = tuple(
        _utility_prediction(row.descriptor_hash, bacc=0.1) for row in descriptors
    )
    selected = select_and_compose_case_policy(
        endpoint,
        actions,
        descriptors,
        influences,
        utilities,
        policy_id=PRIMARY_METHOD_ID,
        require_positive_bacc_prediction=False,
    )
    decision = next(
        row for row in selected.decisions if row.direction == "zero_to_one"
    )
    assert decision.selected_representative == expected


def test_policy_regret_uses_all_eight_residuals_and_strict_positive_lower_vector() -> None:
    endpoint = _endpoint()
    actions = build_action_equivalence_classes(
        endpoint, geometry_id=PROJECTION_GEOMETRY_ID
    )
    descriptors = build_projected_descriptors(endpoint, actions)
    influences = tuple(
        InfluencePrediction(
            row.descriptor_hash,
            row.target_center,
            row.case_id,
            row.representative,
            row.direction,
            row.crossing_count,
            0.1 if row.crossing_count else 0.0,
            "e" * 64,
        )
        for row in descriptors
    )
    utilities = tuple(
        _utility_prediction(row.descriptor_hash, bacc=0.4) for row in descriptors
    )
    case = select_and_compose_case_policy(
        endpoint,
        actions,
        descriptors,
        influences,
        utilities,
        policy_id=PRIMARY_METHOD_ID,
        require_positive_bacc_prediction=False,
    )
    target = build_center_candidate_policy((case,))
    donors = tuple(center for center in CENTERS if center != "0")
    replays = tuple(
        WholePolicyReplay(
            "0",
            donor,
            PROJECTION_GEOMETRY_ID,
            (0.2, 0.2, 0.2),
            (0.1, 0.1, 0.1),
            (0.1, 0.1, 0.1),
            f"{index + 10:064x}",
            f"{index + 20:064x}",
            f"{index + 30:064x}",
        )
        for index, donor in enumerate(donors)
    )
    pseudo_screens = {donor: _transport(donor, passed=True) for donor in donors}
    authorization = authorize_center_policy(
        target,
        replays,
        target_transport_screen=_transport("0", passed=True),
        pseudo_transport_screens=pseudo_screens,
    )
    assert authorization.effective_donor_count == 8
    assert authorization.regret_radius_vector == pytest.approx((0.1, 0.1, 0.1))
    assert authorization.authorized is True

    failed = authorize_center_policy(
        target,
        replays,
        target_transport_screen=_transport("0", passed=True),
        pseudo_transport_screens={
            **pseudo_screens,
            donors[0]: _transport(donors[0], passed=False),
        },
    )
    assert failed.effective_donor_count == 7
    assert failed.authorized is False
