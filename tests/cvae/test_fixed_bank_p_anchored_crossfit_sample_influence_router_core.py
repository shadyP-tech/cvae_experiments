from __future__ import annotations

import ast
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_crossfit_sample_influence_router.bacc_influence import (
    score_sample_influences,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_crossfit_sample_influence_router.calibration import (
    directional_candidate,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_crossfit_sample_influence_router.composition import (
    compose_case_probabilities,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_crossfit_sample_influence_router.config import (
    load_p_anchored_crossfit_sample_influence_router_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_crossfit_sample_influence_router.constants import (
    BLOCKED_FINGERPRINT_CONTROL_ID,
    CENTERS,
    COMPOSED_POLICY_IDS,
    EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT,
    EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT,
    EXPECTED_UTILITY_MODEL_FIT_COUNT,
    FINGERPRINT_FEATURE_COUNT,
    MODEL_BASED_METHOD_ID,
    PORTFOLIO_METHOD_ID,
    PRIMARY_FINGERPRINT_CONTROL_ID,
    SIGN_PRESERVING_SHRINKAGE,
    UTILITY_FEATURE_NAMES,
    UTILITY_RESPONSE_IDS,
    physical_action_ids,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_crossfit_sample_influence_router.contracts import (
    BinaryLabel,
    CenterProbabilitySurface,
    EndpointCasePrediction,
    PhysicalProbabilitySurface,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_crossfit_sample_influence_router.engine import (
    build_preterminal_result,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_crossfit_sample_influence_router.evaluation import (
    evaluate_terminal,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_crossfit_sample_influence_router.physical_fingerprint import (
    blocked_within_case_fingerprint,
    build_physical_fingerprint_surface,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_crossfit_sample_influence_router.sample_influence_contracts import (
    InfluencePrediction,
    TargetLocalPosteriorModel,
    TargetLocalPosteriorPrediction,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_crossfit_sample_influence_router.target_local_posterior import (
    fit_target_local_posterior,
    predict_held_case_posterior,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_crossfit_sample_influence_router.utility_contracts import (
    DonorUtilityRow,
    UtilityPrediction,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_crossfit_sample_influence_router.utility_features import (
    build_utility_descriptors,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_crossfit_sample_influence_router.utility_model import (
    fit_response_model_family,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_crossfit_sample_influence_router.utility_responses import (
    build_donor_utility_rows,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_crossfit_sample_influence_router.uncertainty import (
    predict_utility_surface,
)
from midogpp_thesis.cvae.protocol import ProtocolError


CONFIG = Path(
    "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_crossfit_"
    "sample_influence_router_v1.yaml"
)
PACKAGE = Path(
    "src/midogpp_thesis/cvae/diagnostics/"
    "fixed_bank_p_anchored_crossfit_sample_influence_router"
)


def _endpoint() -> EndpointCasePrediction:
    return EndpointCasePrediction(
        "0",
        "case-0",
        ("s0", "s1", "s2", "s3"),
        MappingProxyType(
            {
                "B": (0.8, 0.4, 0.45, 0.55),
                "I_OPPORTUNITY_GATED": (0.4, 0.6, 0.55, 0.45),
                "R_NINE_ARM_ROBUST": (0.4, 0.6, 0.45, 0.55),
                "P_PROTECTED": (0.4, 0.6, 0.45, 0.55),
            }
        ),
        "1" * 64,
    )


def _center_surface(center: str = "0") -> CenterProbabilitySurface:
    sample_ids = tuple(f"s{case}-{sample}" for case in range(3) for sample in range(4))
    case_ids = tuple(f"case-{case}" for case in range(3) for _sample in range(4))
    signal = np.asarray((0.15, 0.75, 0.30, 0.85) * 3, dtype=np.float32)[None, :]
    offsets = np.linspace(-0.04, 0.04, 9, dtype=np.float32)[:, None]
    actions = {
        action: np.clip(
            signal if index % 2 == 0 else 1.0 - signal,
            0.01,
            0.99,
        ) + offsets
        for index, action in enumerate(physical_action_ids(center))
    }
    actions = {
        action: np.clip(values, 0.01, 0.99).astype(np.float32)
        for action, values in actions.items()
    }
    return CenterProbabilitySurface(center, sample_ids, case_ids, actions, "3" * 64)


def _donor_prediction(
    descriptor_hash: str, *, bacc: float, proper_safe: bool = True
) -> UtilityPrediction:
    donors = tuple(center for center in CENTERS if center != "0")
    values = (
        ("bacc_contribution_delta", bacc),
        ("brier_contribution_delta", -0.01 if proper_safe else 0.01),
        ("log_loss_contribution_delta", -0.02 if proper_safe else 0.02),
    )
    deletion = tuple(
        (response, tuple((center, value) for center in donors))
        for response, value in values
    )
    return UtilityPrediction(
        descriptor_hash,
        values,
        deletion,
        tuple((response, 0.0) for response in UTILITY_RESPONSE_IDS),
        tuple((response, 0.0) for response in UTILITY_RESPONSE_IDS),
        values,
        tuple((response, 1.0) for response in UTILITY_RESPONSE_IDS),
        tuple(f"{index + 1:064x}" for index in range(9)),
    )


def _influence(descriptor: object, score: float) -> InfluencePrediction:
    return InfluencePrediction(
        descriptor.descriptor_hash,
        descriptor.target_center,
        descriptor.case_id,
        descriptor.alternative,
        descriptor.direction,
        descriptor.crossing_count,
        score if descriptor.crossing_count else 0.0,
        "a" * 64,
    )


def _synthetic_training_rows() -> tuple[DonorUtilityRow, ...]:
    rows = []
    for donor_index, donor in enumerate(center for center in CENTERS if center != "0"):
        for case_index in range(3):
            for alternative_index, alternative in enumerate(
                ("B", "I_OPPORTUNITY_GATED", "R_NINE_ARM_ROBUST")
            ):
                for direction_index, direction in enumerate(("zero_to_one", "one_to_zero")):
                    signal = (donor_index - 3.5) / 20 + (case_index - 1) / 10
                    feature = tuple(
                        signal
                        + (index + 1) / 100
                        + alternative_index / 20
                        + direction_index / 30
                        for index in range(len(UTILITY_FEATURE_NAMES))
                    )
                    bacc = 0.002 * signal
                    rows.append(
                        DonorUtilityRow(
                            "0",
                            donor,
                            f"{donor}-case-{case_index}",
                            alternative,
                            direction,
                            feature,
                            1,
                            bacc,
                            -bacc / 2,
                            -bacc,
                            f"{len(rows) + 1:064x}",
                        )
                    )
    return tuple(rows)


def test_config_and_workstation_budget_are_frozen() -> None:
    config = load_p_anchored_crossfit_sample_influence_router_config(CONFIG)
    assert config.experiment_id.endswith("crossfit_sample_influence_router.v1")
    assert config.runtime["expected_outer_endpoint_model_fit_count"] == 3_488
    assert config.runtime["expected_utility_model_fit_count"] == 81
    assert config.runtime["expected_target_posterior_model_fit_count"] == 436
    assert EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT == 3_488
    assert EXPECTED_UTILITY_MODEL_FIT_COUNT == 81
    assert EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT == 436
    assert config.claim_boundary["consumed_test_data"] is True
    assert config.claim_boundary["fresh_evidence"] is False


def test_science_package_does_not_import_predecessor_diagnostics() -> None:
    forbidden = {
        "fixed_bank_p_anchored_directional_signed_utility_router",
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


def test_fingerprint_is_exact_thirty_features_and_block_is_case_local() -> None:
    fingerprint = build_physical_fingerprint_surface(_center_surface())
    blocked = blocked_within_case_fingerprint(fingerprint)
    assert fingerprint.feature_values.shape == (12, FINGERPRINT_FEATURE_COUNT)
    assert fingerprint.control_id == PRIMARY_FINGERPRINT_CONTROL_ID
    assert blocked.control_id == BLOCKED_FINGERPRINT_CONTROL_ID
    assert not np.array_equal(fingerprint.feature_values, blocked.feature_values)
    for case in fingerprint.cases:
        positions = fingerprint.positions(case)
        original_rows = sorted(map(tuple, fingerprint.feature_values[positions]))
        blocked_rows = sorted(map(tuple, blocked.feature_values[positions]))
        assert original_rows == blocked_rows


def test_target_posterior_uses_exact_H_minus_c_and_corrects_prevalence() -> None:
    fingerprint = build_physical_fingerprint_surface(_center_surface())
    held = "case-0"
    support_values = (0, 0, 1, 0, 1, 0, 1, 0)
    support_positions = np.flatnonzero(np.asarray(fingerprint.case_ids) != held)
    labels = tuple(
        BinaryLabel(
            "0",
            fingerprint.case_ids[index],
            fingerprint.sample_ids[index],
            value,
            "outer_support::H=0::excluded_c=case-0",
        )
        for index, value in zip(support_positions, support_values, strict=True)
    )
    model = fit_target_local_posterior(
        fingerprint, held_case_id=held, support_labels=labels
    )
    prediction = predict_held_case_posterior(model, fingerprint)
    assert model.support_n_positive == 3
    assert model.support_n_negative == 5
    assert model.held_case_id not in model.support_case_ids
    assert prediction.sample_ids == tuple(
        fingerprint.sample_ids[index] for index in fingerprint.positions(held)
    )
    assert prediction.balanced_probabilities != pytest.approx(
        prediction.natural_probabilities
    )

    held_poison = BinaryLabel(
        "0",
        held,
        fingerprint.sample_ids[0],
        1,
        "outer_support::H=0::excluded_c=case-0",
    )
    with pytest.raises(ProtocolError, match="exact H-c capability"):
        fit_target_local_posterior(
            fingerprint,
            held_case_id=held,
            support_labels=(*labels, held_poison),
        )


def test_influence_matches_closed_form_balanced_accuracy_score() -> None:
    endpoint = _endpoint()
    descriptors = build_utility_descriptors(endpoint)
    names = tuple(f"f{index}" for index in range(FINGERPRINT_FEATURE_COUNT))
    model = TargetLocalPosteriorModel(
        "0",
        "case-0",
        ("case-1", "case-2"),
        names,
        (0.0,) * FINGERPRINT_FEATURE_COUNT,
        (1.0,) * FINGERPRINT_FEATURE_COUNT,
        (0.0,) * FINGERPRINT_FEATURE_COUNT,
        0.0,
        10,
        4,
        6,
        "b" * 64,
        "c" * 64,
        1,
        True,
    )
    posterior = TargetLocalPosteriorPrediction(
        "0",
        "case-0",
        endpoint.sample_ids,
        (0.8, 0.2, 0.7, 0.1),
        (0.8, 0.2, 0.7, 0.1),
        model.model_hash,
        model.fingerprint_hash,
    )
    scored = score_sample_influences(descriptors, posterior=posterior, model=model)
    b_up = next(
        row
        for row in scored
        if row.alternative == "B" and row.direction == "zero_to_one"
    )
    assert b_up.crossing_count == 1
    assert b_up.target_score == pytest.approx(0.5 * (0.8 / 4 - 0.2 / 6))
    assert all(row.target_score == 0.0 for row in scored if row.crossing_count == 0)


def test_donor_veto_models_refit_all_eight_donors() -> None:
    rows = _synthetic_training_rows()
    full, deleted = fit_response_model_family(rows, outer_target_center="0")
    predictions = predict_utility_surface(
        build_utility_descriptors(_endpoint()),
        donor_rows=rows,
        full_model=full,
        delete_models=deleted,
    )
    assert len(deleted) == 8
    assert len(predictions) == 6
    assert all(len(row.model_hashes) == 9 for row in predictions)


def test_donor_responses_match_exact_branch_local_composition() -> None:
    endpoint = _endpoint()
    descriptors = build_utility_descriptors(endpoint)
    labels = tuple(
        BinaryLabel(
            "0",
            "case-0",
            sample,
            value,
            "crossing_donor::outer_H=1::donor_J=0",
        )
        for sample, value in zip(endpoint.sample_ids, (1, 0, 1, 0), strict=True)
    )
    rows = build_donor_utility_rows(
        outer_target_center="1",
        prediction=endpoint,
        descriptors=descriptors,
        case_labels=labels,
        center_n_positive=2,
        center_n_negative=2,
    )
    assert len(rows) == 6
    assert all(
        row.bacc_contribution_delta
        == row.brier_contribution_delta
        == row.log_loss_contribution_delta
        == 0.0
        for row in rows
        if row.crossing_count == 0
    )


def test_dual_veto_selects_one_action_per_direction_with_exact_p_fallback() -> None:
    endpoint = _endpoint()
    descriptors = build_utility_descriptors(endpoint)
    target_scores = {
        ("B", "zero_to_one"): 0.02,
        ("I_OPPORTUNITY_GATED", "one_to_zero"): 0.03,
    }
    influences = [
        _influence(row, target_scores.get((row.alternative, row.direction), -0.01))
        for row in descriptors
    ]
    donors = [
        _donor_prediction(
            row.descriptor_hash,
            bacc=(
                0.01
                if (row.alternative, row.direction) in target_scores
                else -0.01
            ),
        )
        for row in descriptors
    ]
    composed = compose_case_probabilities(
        endpoint,
        descriptors,
        influences,
        donors,
        policy_id=MODEL_BASED_METHOD_ID,
    )
    assert [row.selected_alternative for row in composed.decisions] == [
        "B",
        "I_OPPORTUNITY_GATED",
    ]
    assert composed.probabilities == pytest.approx((0.575, 0.6, 0.45, 0.4875))
    assert dict(composed.switched_sample_counts) == {"zero_to_one": 1, "one_to_zero": 1}

    fallback = compose_case_probabilities(
        endpoint,
        descriptors,
        [_influence(row, -0.01) for row in descriptors],
        donors,
        policy_id=MODEL_BASED_METHOD_ID,
    )
    assert fallback.probabilities == endpoint.probabilities[PORTFOLIO_METHOD_ID]
    assert all(row.selected_alternative == PORTFOLIO_METHOD_ID for row in fallback.decisions)


def test_sign_preserving_shrinkage_keeps_selected_hard_class() -> None:
    candidate, mask = directional_candidate(_endpoint(), "B", "zero_to_one")
    assert SIGN_PRESERVING_SHRINKAGE == 0.25
    assert tuple(np.flatnonzero(mask)) == (0,)
    assert candidate[0] == pytest.approx(0.575)
    assert candidate[0] >= 0.5


def test_small_end_to_end_surface_seals_before_terminal_labels() -> None:
    store_hash = "3" * 64
    centers = {}
    labels: dict[tuple[str, str, str], int] = {}
    seed_offsets = np.linspace(-0.02, 0.02, 9, dtype=np.float32)[:, None]
    for center in CENTERS:
        sample_ids = tuple(
            f"{center}-case-{case}-sample-{sample}"
            for case in range(3)
            for sample in range(2)
        )
        case_ids = tuple(
            f"{center}-case-{case}" for case in range(3) for _sample in range(2)
        )
        base = np.asarray((0.30, 0.70) * 3, dtype=np.float32)[None, :]
        actions = {}
        for index, action in enumerate(physical_action_ids(center)):
            mean = base if index < 2 or index % 2 == 0 else 1.0 - base
            actions[action] = np.clip(mean + seed_offsets, 0.01, 0.99).astype(np.float32)
        centers[center] = CenterProbabilitySurface(
            center, sample_ids, case_ids, actions, store_hash
        )
        labels.update(
            {
                (center, case_id, sample_id): sample_index % 2
                for sample_index, (case_id, sample_id) in enumerate(
                    zip(case_ids, sample_ids, strict=True)
                )
            }
        )
    surface = PhysicalProbabilitySurface(
        centers, store_hash, strict_canonical_topology=False
    )

    def load(
        granted: frozenset[tuple[str, str, str]], role: str
    ) -> tuple[SimpleNamespace, ...]:
        return tuple(
            SimpleNamespace(
                center=center,
                case_id=case_id,
                sample_id=sample_id,
                value=labels[(center, case_id, sample_id)],
                role=role,
            )
            for center, case_id, sample_id in sorted(granted)
        )

    preterminal = build_preterminal_result(surface, load, use_processes=False)
    assert preterminal.label_firewall.report_payload()["terminal_opened"] is False
    assert all(
        len(rows) == 27 for rows in preterminal.composed_predictions_by_policy.values()
    )
    assert sum(
        len(rows) for rows in preterminal.target_posterior_models_by_control.values()
    ) == 54
    assert sum(
        1 + len(preterminal.delete_models_by_target[center]) for center in CENTERS
    ) == EXPECTED_UTILITY_MODEL_FIT_COUNT
    terminal = evaluate_terminal(preterminal)
    assert terminal.capability_report["status"] == "PASS"
    assert terminal.capability_report["route_decision_seal_count"] == 27
    assert terminal.diagnostic_summary["promotion_eligible"] is False
    assert all(row["row_type"] == "action" for row in terminal.utility_information_rows)
    assert all("actual_helpful" not in row for row in terminal.utility_information_rows)
    assert set(COMPOSED_POLICY_IDS) <= {
        row["method_id"] for row in terminal.method_metrics
    }
