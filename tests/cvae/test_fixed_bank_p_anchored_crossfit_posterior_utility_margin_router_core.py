from __future__ import annotations

import ast
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_crossfit_posterior_utility_margin_router.composition import (
    compose_case_probabilities,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_crossfit_posterior_utility_margin_router.constants import (
    ALTERNATIVE_METHOD_IDS,
    BASELINE_METHOD_ID,
    CENTERS,
    DIRECTION_IDS,
    FINGERPRINT_STATISTIC_IDS,
    IDENTIFICATION_METHOD_ID,
    MARGIN_MIN,
    MODEL_BASED_METHOD_ID,
    PORTFOLIO_METHOD_ID,
    PRIMARY_FINGERPRINT_CONTROL_ID,
    ROBUST_MAD_FLOOR,
    ROBUST_MAD_SCALE,
    ROBUST_METHOD_ID,
    UTILITY_FEATURE_NAMES,
    physical_action_ids,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_crossfit_posterior_utility_margin_router.contracts import (
    BinaryLabel,
    CenterProbabilitySurface,
    EndpointCasePrediction,
    PhysicalProbabilitySurface,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_crossfit_posterior_utility_margin_router.engine import (
    build_preterminal_result,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_crossfit_posterior_utility_margin_router.evaluation import (
    evaluate_terminal,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_crossfit_posterior_utility_margin_router.hashing import (
    canonical_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_crossfit_posterior_utility_margin_router.margin_calibration import (
    calibrate_margin,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_crossfit_posterior_utility_margin_router.posterior_contracts import (
    PhysicalFingerprintSurface,
    RoutePosteriorEnsemble,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_crossfit_posterior_utility_margin_router.posterior_utility import (
    score_posterior_utilities,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_crossfit_posterior_utility_margin_router.support_crossfit import (
    build_support_fold_plans,
    fit_route_posterior_ensemble,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_crossfit_posterior_utility_margin_router.utility_contracts import (
    DonorUtilityRow,
    PosteriorUtilityPrediction,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_crossfit_posterior_utility_margin_router.utility_features import (
    build_utility_descriptors,
)
from midogpp_thesis.cvae.protocol import ProtocolError


PACKAGE = Path(
    "src/midogpp_thesis/cvae/diagnostics/"
    "fixed_bank_p_anchored_crossfit_posterior_utility_margin_router"
)


def _fingerprint() -> tuple[PhysicalFingerprintSurface, tuple[int, ...]]:
    samples: list[str] = []
    cases: list[str] = []
    labels: list[int] = []
    for case_index in range(7):
        for label in (0, 1):
            samples.append(f"s-{case_index}-{label}")
            cases.append(f"case-{case_index}")
            labels.append(label)
    names = tuple(
        f"{action}::{statistic}"
        for action in physical_action_ids("0")
        for statistic in FINGERPRINT_STATISTIC_IDS
    )
    values = np.zeros((len(samples), len(names)), dtype=np.float64)
    for row, label in enumerate(labels):
        for action in range(10):
            values[row, 3 * action] = 0.2 + 0.6 * label
            values[row, 3 * action + 1] = 0.1
            values[row, 3 * action + 2] = label
    return (
        PhysicalFingerprintSurface(
            "0",
            tuple(samples),
            tuple(cases),
            names,
            values,
            canonical_hash({"surface": "fixture"}),
            PRIMARY_FINGERPRINT_CONTROL_ID,
        ),
        tuple(labels),
    )


def _endpoint() -> EndpointCasePrediction:
    return EndpointCasePrediction(
        "0",
        "case-0",
        ("s0", "s1", "s2", "s3"),
        MappingProxyType(
            {
                BASELINE_METHOD_ID: (0.8, 0.4, 0.45, 0.55),
                IDENTIFICATION_METHOD_ID: (0.4, 0.6, 0.55, 0.45),
                ROBUST_METHOD_ID: (0.4, 0.6, 0.45, 0.55),
                PORTFOLIO_METHOD_ID: (0.4, 0.6, 0.45, 0.55),
            }
        ),
        canonical_hash({"state": "fixture"}),
    )


def _ensemble(endpoint: EndpointCasePrediction) -> RoutePosteriorEnsemble:
    eta = (0.9, 0.1, 0.8, 0.2)
    return RoutePosteriorEnsemble(
        "0",
        endpoint.case_id,
        PRIMARY_FINGERPRINT_CONTROL_ID,
        tuple(canonical_hash(["plan", index]) for index in range(5)),
        tuple(canonical_hash(["model", index]) for index in range(5)),
        tuple(canonical_hash(["prediction", index]) for index in range(5)),
        endpoint.sample_ids,
        (eta,) * 5,
        20,
        10,
        10,
        20,
        0.8,
        0.15,
        0.25,
        0.10,
        canonical_hash({"oof": "identity"}),
        canonical_hash({"oof": "prediction"}),
    )


def _posterior_prediction(
    *,
    center: str,
    case: str,
    alternative: str,
    direction: str,
    descriptor_hash: str,
    score: float,
    proper_safe: bool = True,
) -> PosteriorUtilityPrediction:
    proper = -0.01 if proper_safe else 0.01
    return PosteriorUtilityPrediction(
        center,
        case,
        alternative,
        direction,
        PRIMARY_FINGERPRINT_CONTROL_ID,
        1,
        (score,) * 5,
        (proper,) * 5,
        (proper,) * 5,
        score,
        proper,
        proper,
        0.8,
        0.1,
        True,
        descriptor_hash,
        canonical_hash([center, case, "ensemble"]),
    )


def _calibration_rows(
    *, helpful: bool
) -> tuple[tuple[PosteriorUtilityPrediction, ...], tuple[DonorUtilityRow, ...]]:
    predictions: list[PosteriorUtilityPrediction] = []
    responses: list[DonorUtilityRow] = []
    for donor in CENTERS[1:]:
        for alternative in ALTERNATIVE_METHOD_IDS:
            for direction in DIRECTION_IDS:
                descriptor_hash = canonical_hash([donor, alternative, direction])
                candidate = (
                    alternative == BASELINE_METHOD_ID and direction == "zero_to_one"
                )
                predictions.append(
                    _posterior_prediction(
                        center=donor,
                        case=f"{donor}-case",
                        alternative=alternative,
                        direction=direction,
                        descriptor_hash=descriptor_hash,
                        score=0.1 if candidate else -0.1,
                    )
                )
                actual = 0.01 if helpful and candidate else -0.01
                responses.append(
                    DonorUtilityRow(
                        "0",
                        donor,
                        f"{donor}-case",
                        alternative,
                        direction,
                        (0.0,) * len(UTILITY_FEATURE_NAMES),
                        1,
                        actual,
                        -0.001 if helpful and candidate else 0.001,
                        -0.001 if helpful and candidate else 0.001,
                        descriptor_hash,
                    )
                )
    return tuple(predictions), tuple(responses)


def test_science_package_has_no_cross_diagnostic_imports() -> None:
    forbidden = {
        "fixed_bank_p_anchored_crossfit_sample_influence_router",
        "fixed_bank_p_anchored_directional_signed_utility_router",
        "fixed_bank_p_anchored_directional_crossing_bagging",
        "fixed_bank_loo_nested_donor_endpoint_regret_router",
    }
    for path in PACKAGE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules = {
            *(node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)),
            *(
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            ),
        }
        assert not any(fragment in module for fragment in forbidden for module in modules)


def test_five_fold_plan_is_whole_case_and_label_free() -> None:
    fingerprint, _labels = _fingerprint()
    plans = build_support_fold_plans(fingerprint, held_case_id="case-0")
    assert len(plans) == 5
    assert {case for row in plans for case in row.validation_case_ids} == {
        f"case-{index}" for index in range(1, 7)
    }
    assert sum(len(row.validation_case_ids) for row in plans) == 6
    assert all("case-0" not in (*row.training_case_ids, *row.validation_case_ids) for row in plans)
    assert all(not (set(row.training_case_ids) & set(row.validation_case_ids)) for row in plans)


def test_route_ensemble_excludes_held_labels_and_has_exact_oof_coverage() -> None:
    fingerprint, labels = _fingerprint()
    support_positions = np.flatnonzero(
        np.asarray(fingerprint.case_ids, dtype=object) != "case-0"
    )
    scoped = tuple(
        BinaryLabel(
            "0",
            fingerprint.case_ids[index],
            fingerprint.sample_ids[index],
            labels[index],
            "outer_support::H=0::excluded_c=case-0",
        )
        for index in support_positions
    )
    models, predictions, ensemble = fit_route_posterior_ensemble(
        fingerprint, held_case_id="case-0", support_labels=scoped
    )
    assert len(models) == len(predictions) == 5
    assert ensemble.oof_sample_count == len(scoped)
    assert ensemble.oof_auc == pytest.approx(1.0)
    assert ensemble.reliability_pass
    with pytest.raises(ProtocolError, match="exact H-c capability"):
        fit_route_posterior_ensemble(
            fingerprint,
            held_case_id="case-0",
            support_labels=(
                *scoped,
                BinaryLabel(
                    "0",
                    "case-0",
                    fingerprint.sample_ids[0],
                    labels[0],
                    "outer_support::H=0::excluded_c=case-0",
                ),
            ),
        )


def test_posterior_expected_bacc_matches_closed_form_with_mad_floor() -> None:
    endpoint = _endpoint()
    descriptors = build_utility_descriptors(endpoint)
    scored = score_posterior_utilities(endpoint, descriptors, _ensemble(endpoint))
    candidate = next(
        row
        for row in scored
        if row.alternative == BASELINE_METHOD_ID and row.direction == "zero_to_one"
    )
    expected = 0.5 * (0.9 / 12.0 - 0.1 / 12.0)
    assert candidate.fold_bacc_deltas == pytest.approx((expected,) * 5)
    assert candidate.robust_bacc_lower == pytest.approx(
        expected - ROBUST_MAD_SCALE * ROBUST_MAD_FLOOR
    )
    assert candidate.crossing_count == 1


def test_nested_margin_is_strictly_positive_and_falls_back_when_harmful() -> None:
    predictions, responses = _calibration_rows(helpful=True)
    calibration = calibrate_margin(
        outer_target_center="0",
        control_id=PRIMARY_FINGERPRINT_CONTROL_ID,
        predictions=predictions,
        donor_rows=responses,
    )
    assert calibration.authorized
    assert calibration.selected_margin == MARGIN_MIN
    assert calibration.selected_action_count == 8
    assert len(calibration.inner_replays) == 8

    predictions, responses = _calibration_rows(helpful=False)
    fallback = calibrate_margin(
        outer_target_center="0",
        control_id=PRIMARY_FINGERPRINT_CONTROL_ID,
        predictions=predictions,
        donor_rows=responses,
    )
    assert fallback.selected_margin == 1.0
    assert fallback.selected_action_count == 0


def test_composition_uses_margin_and_preserves_exact_p_on_unauthorized_center() -> None:
    endpoint = _endpoint()
    descriptors = build_utility_descriptors(endpoint)
    donor_predictions, donor_responses = _calibration_rows(helpful=True)
    calibration = calibrate_margin(
        outer_target_center="0",
        control_id=PRIMARY_FINGERPRINT_CONTROL_ID,
        predictions=donor_predictions,
        donor_rows=donor_responses,
    )
    utilities = tuple(
        _posterior_prediction(
            center="0",
            case=endpoint.case_id,
            alternative=row.alternative,
            direction=row.direction,
            descriptor_hash=row.descriptor_hash,
            score=(
                0.1
                if row.alternative == BASELINE_METHOD_ID
                and row.direction == "zero_to_one"
                else -0.1
            ),
        )
        for row in descriptors
    )
    composed = compose_case_probabilities(
        endpoint,
        descriptors,
        utilities,
        calibration,
        policy_id=MODEL_BASED_METHOD_ID,
    )
    assert composed.decisions[0].selected_alternative == BASELINE_METHOD_ID
    assert composed.probabilities[0] == pytest.approx(0.575)

    unsafe = type(calibration)(
        calibration.outer_target_center,
        calibration.control_id,
        calibration.selected_margin,
        False,
        calibration.candidate_margins,
        calibration.selected_action_count,
        calibration.donor_bacc_delta,
        calibration.donor_brier_delta,
        calibration.donor_log_loss_delta,
        calibration.inner_replays,
        calibration.source_utility_hash,
        calibration.source_response_hash,
    )
    fallback = compose_case_probabilities(
        endpoint,
        descriptors,
        utilities,
        unsafe,
        policy_id=MODEL_BASED_METHOD_ID,
    )
    assert fallback.probabilities == endpoint.probabilities[PORTFOLIO_METHOD_ID]


def test_small_end_to_end_surface_seals_before_terminal_labels() -> None:
    store_hash = canonical_hash({"store": "fixture"})
    centers = {}
    labels: dict[tuple[str, str, str], int] = {}
    seed_offsets = np.linspace(-0.02, 0.02, 9, dtype=np.float32)[:, None]
    for center in CENTERS:
        sample_ids = tuple(
            f"{center}-case-{case}-sample-{sample}"
            for case in range(7)
            for sample in range(2)
        )
        case_ids = tuple(
            f"{center}-case-{case}" for case in range(7) for _sample in range(2)
        )
        base = np.asarray((0.30, 0.70) * 7, dtype=np.float32)[None, :]
        actions = {}
        for index, action in enumerate(physical_action_ids(center)):
            mean = base if index < 2 or index % 2 == 0 else 1.0 - base
            actions[action] = np.clip(
                mean + seed_offsets, 0.01, 0.99
            ).astype(np.float32)
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
        len(rows) == 63
        for rows in preterminal.composed_predictions_by_policy.values()
    )
    assert sum(
        len(rows) for rows in preterminal.target_posterior_models_by_control.values()
    ) == 630
    assert len(preterminal.margin_calibrations) == 18
    terminal = evaluate_terminal(preterminal)
    assert terminal.capability_report["status"] == "PASS"
    assert terminal.capability_report["route_decision_seal_count"] == 63
    assert terminal.diagnostic_summary["promotion_eligible"] is False
