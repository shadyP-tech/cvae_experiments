from __future__ import annotations

import ast
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_directional_signed_utility_router.calibration import (
    directional_candidate,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_directional_signed_utility_router.composition import (
    compose_case_probabilities,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_directional_signed_utility_router.config import (
    load_p_anchored_directional_signed_utility_router_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_directional_signed_utility_router.constants import (
    CENTERS,
    COMPOSED_POLICY_IDS,
    EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT,
    EXPECTED_UTILITY_MODEL_FIT_COUNT,
    MODEL_BASED_METHOD_ID,
    PORTFOLIO_METHOD_ID,
    SIGN_PRESERVING_SHRINKAGE,
    UTILITY_CELL_IDS,
    UTILITY_FEATURE_NAMES,
    UTILITY_RESPONSE_IDS,
    physical_action_ids,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_directional_signed_utility_router.contracts import (
    BinaryLabel,
    CenterProbabilitySurface,
    EndpointCasePrediction,
    PhysicalProbabilitySurface,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_directional_signed_utility_router.engine import (
    build_preterminal_result,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_directional_signed_utility_router.evaluation import (
    evaluate_terminal,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_directional_signed_utility_router.utility_contracts import (
    DonorUtilityRow,
    UtilityPrediction,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_directional_signed_utility_router.utility_features import (
    build_utility_descriptors,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_directional_signed_utility_router.utility_model import (
    fit_response_model_family,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_directional_signed_utility_router.utility_responses import (
    blocked_feature_permutation,
    build_donor_utility_rows,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_directional_signed_utility_router.uncertainty import (
    predict_utility_surface,
)


CONFIG = Path(
    "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_directional_"
    "signed_utility_router_v1.yaml"
)
PACKAGE = Path(
    "src/midogpp_thesis/cvae/diagnostics/"
    "fixed_bank_p_anchored_directional_signed_utility_router"
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


def _labels(values: tuple[int, ...]) -> tuple[BinaryLabel, ...]:
    return tuple(
        BinaryLabel(
            "0",
            "case-0",
            sample,
            value,
            "crossing_donor::outer_H=1::donor_J=0",
        )
        for sample, value in zip(_endpoint().sample_ids, values, strict=True)
    )


def _prediction(descriptor_hash: str, *, bacc: float, safe: bool = True) -> UtilityPrediction:
    donors = tuple(center for center in CENTERS if center != "0")
    full = (
        ("bacc_contribution_delta", bacc),
        ("brier_contribution_delta", -0.01 if safe else 0.01),
        ("log_loss_contribution_delta", -0.02 if safe else 0.02),
    )
    deletion = tuple(
        (
            response,
            tuple((center, value) for center in donors),
        )
        for response, value in full
    )
    return UtilityPrediction(
        descriptor_hash,
        full,
        deletion,
        tuple((response, 0.0) for response in UTILITY_RESPONSE_IDS),
        tuple((response, 0.0) for response in UTILITY_RESPONSE_IDS),
        full,
        tuple((response, 1.0) for response in UTILITY_RESPONSE_IDS),
        tuple(f"{index + 1:064x}" for index in range(27)),
    )


def _synthetic_training_rows() -> tuple[DonorUtilityRow, ...]:
    rows = []
    for donor_index, donor in enumerate(center for center in CENTERS if center != "0"):
        for case_index in range(3):
            for alternative_index, alternative in enumerate(("B", "I_OPPORTUNITY_GATED", "R_NINE_ARM_ROBUST")):
                for direction_index, direction in enumerate(("zero_to_one", "one_to_zero")):
                    signal = (donor_index - 3.5) / 20 + (case_index - 1) / 10
                    feature = tuple(
                        signal + (index + 1) / 100 + alternative_index / 20 + direction_index / 30
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
    config = load_p_anchored_directional_signed_utility_router_config(CONFIG)
    assert config.experiment_id.endswith("directional_signed_utility_router.v1")
    assert config.runtime["expected_outer_endpoint_model_fit_count"] == 3_488
    assert config.runtime["expected_utility_model_fit_count"] == 162
    assert EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT == 3_488
    assert EXPECTED_UTILITY_MODEL_FIT_COUNT == 162
    assert config.claim_boundary["consumed_test_data"] is True
    assert config.claim_boundary["fresh_evidence"] is False


def test_science_package_does_not_import_predecessor_diagnostics() -> None:
    forbidden = {
        "fixed_bank_p_anchored_directional_crossing_bagging",
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


def test_complete_descriptor_rectangle_retains_structural_zeros() -> None:
    rows = build_utility_descriptors(_endpoint())
    assert len(rows) == 6
    assert len({(row.alternative, row.direction) for row in rows}) == 6
    assert all(row.feature_names == UTILITY_FEATURE_NAMES for row in rows)
    structural = [row for row in rows if row.alternative == "R_NINE_ARM_ROBUST"]
    assert len(structural) == 2
    assert all(row.crossing_count == 0 for row in structural)


def test_direct_responses_match_composition_and_are_label_only_in_response() -> None:
    endpoint = _endpoint()
    descriptors = build_utility_descriptors(endpoint)
    first = build_donor_utility_rows(
        outer_target_center="1",
        prediction=endpoint,
        descriptors=descriptors,
        case_labels=_labels((1, 0, 1, 0)),
        center_n_positive=2,
        center_n_negative=2,
    )
    second = build_donor_utility_rows(
        outer_target_center="1",
        prediction=endpoint,
        descriptors=descriptors,
        case_labels=_labels((0, 1, 0, 1)),
        center_n_positive=2,
        center_n_negative=2,
    )
    assert [row.feature_values for row in first] == [row.feature_values for row in second]
    assert [row.descriptor_hash for row in first] == [row.descriptor_hash for row in second]
    assert any(a.bacc_contribution_delta != b.bacc_contribution_delta for a, b in zip(first, second, strict=True))
    assert all(
        row.bacc_contribution_delta == row.brier_contribution_delta == row.log_loss_contribution_delta == 0.0
        for row in first
        if row.crossing_count == 0
    )


def test_sign_preserving_shrinkage_keeps_hard_class() -> None:
    endpoint = _endpoint()
    candidate, mask = directional_candidate(endpoint, "B", "zero_to_one")
    assert SIGN_PRESERVING_SHRINKAGE == 0.25
    assert tuple(np.flatnonzero(mask)) == (0,)
    assert candidate[0] == pytest.approx(0.575)
    assert candidate[0] >= 0.5


def test_response_models_refit_all_donors_and_calibrate_on_held_donors() -> None:
    rows = _synthetic_training_rows()
    full, deleted = fit_response_model_family(rows, outer_target_center="0")
    assert full.cell_ids == UTILITY_CELL_IDS
    assert len(deleted) == 8
    predictions = predict_utility_surface(
        build_utility_descriptors(_endpoint()),
        donor_rows=rows,
        full_model=full,
        delete_models=deleted,
    )
    assert len(predictions) == 6
    assert all(len(dict(row.deletion_values)[UTILITY_RESPONSE_IDS[0]]) == 8 for row in predictions)
    assert all(row.scale("bacc_contribution_delta") >= 0.0 for row in predictions)


def test_blocked_control_moves_features_but_preserves_signed_responses() -> None:
    rows = _synthetic_training_rows()
    blocked = blocked_feature_permutation(rows)
    assert [row.bacc_contribution_delta for row in rows] == [
        row.bacc_contribution_delta for row in blocked
    ]
    assert [row.brier_contribution_delta for row in rows] == [
        row.brier_contribution_delta for row in blocked
    ]
    assert any(a.feature_values != b.feature_values for a, b in zip(rows, blocked, strict=True))


def test_selection_is_one_action_per_direction_with_exact_p_fallback() -> None:
    endpoint = _endpoint()
    descriptors = build_utility_descriptors(endpoint)
    predictions = []
    for row in descriptors:
        score = -0.01
        if row.alternative == "B" and row.direction == "zero_to_one":
            score = 0.02
        if row.alternative == "I_OPPORTUNITY_GATED" and row.direction == "one_to_zero":
            score = 0.03
        predictions.append(_prediction(row.descriptor_hash, bacc=score))
    composed = compose_case_probabilities(
        endpoint,
        descriptors,
        predictions,
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
        [_prediction(row.descriptor_hash, bacc=-0.01) for row in descriptors],
        policy_id=MODEL_BASED_METHOD_ID,
    )
    assert fallback.probabilities == endpoint.probabilities[PORTFOLIO_METHOD_ID]
    assert all(row.selected_alternative == PORTFOLIO_METHOD_ID for row in fallback.decisions)


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
    assert all(len(rows) == 27 for rows in preterminal.composed_predictions_by_policy.values())
    assert all(
        len(preterminal.utility_descriptors_by_center[center]) == 18
        for center in CENTERS
    )
    assert sum(
        1
        + len(preterminal.delete_models_by_target[center])
        + 1
        + len(preterminal.permutation_delete_models_by_target[center])
        for center in CENTERS
    ) == EXPECTED_UTILITY_MODEL_FIT_COUNT
    assert sum(
        len(composition.decisions)
        for policy in COMPOSED_POLICY_IDS
        for composition in preterminal.composed_predictions_by_policy[policy]
    ) == 27 * len(COMPOSED_POLICY_IDS) * 2
    terminal = evaluate_terminal(preterminal)
    assert terminal.capability_report["status"] == "PASS"
    assert terminal.capability_report["route_decision_seal_count"] == 27
    assert terminal.diagnostic_summary["promotion_eligible"] is False
    assert set(COMPOSED_POLICY_IDS) <= {
        row["method_id"] for row in terminal.method_metrics
    }
