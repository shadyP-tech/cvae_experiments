from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_simultaneous_shift_calibrated_utility_router.cell_residuals import (
    build_residual_observations,
    posterior_point,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_simultaneous_shift_calibrated_utility_router.constants import (
    ALTERNATIVE_METHOD_IDS,
    BASELINE_METHOD_ID,
    CENTERS,
    DIRECTION_IDS,
    ZERO_SHIFT_CONTROL_METHOD_ID,
    MODEL_BASED_METHOD_ID,
    MINIMAX_CONTROL_METHOD_ID,
    PRIMARY_FINGERPRINT_CONTROL_ID,
    PORTFOLIO_METHOD_ID,
    UTILITY_FEATURE_NAMES,
    physical_action_ids,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_simultaneous_shift_calibrated_utility_router.contracts import (
    CenterProbabilitySurface,
    PhysicalProbabilitySurface,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_simultaneous_shift_calibrated_utility_router.envelope_calibration import (
    calibrate_envelope,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_simultaneous_shift_calibrated_utility_router.engine import (
    build_preterminal_result,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_simultaneous_shift_calibrated_utility_router.evaluation import (
    evaluate_terminal,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_simultaneous_shift_calibrated_utility_router.hashing import (
    canonical_hash,
    json_native,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_simultaneous_shift_calibrated_utility_router.persistence import (
    persist_terminal,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_simultaneous_shift_calibrated_utility_router.selection import (
    select_certificate_for_direction,
    select_directional_actions,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_simultaneous_shift_calibrated_utility_router.shift_certificate import (
    certify_utility,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_simultaneous_shift_calibrated_utility_router.simultaneous_envelope import (
    fit_simultaneous_envelope,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_simultaneous_shift_calibrated_utility_router.tail_risk import (
    lower_tail_mean,
    upper_tail_mean,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_simultaneous_shift_calibrated_utility_router.utility_contracts import (
    DonorUtilityRow,
    PosteriorUtilityPrediction,
    UtilityDescriptor,
)
from midogpp_thesis.cvae.protocol import ProtocolError


PACKAGE = Path(
    "src/midogpp_thesis/cvae/diagnostics/"
    "fixed_bank_p_anchored_simultaneous_shift_calibrated_utility_router"
)


def _features(alternative: str, direction: str) -> tuple[float, ...]:
    offset = 0.01 * ALTERNATIVE_METHOD_IDS.index(alternative)
    offset += 0.005 * DIRECTION_IDS.index(direction)
    return tuple(offset + 0.01 * index for index in range(len(UTILITY_FEATURE_NAMES)))


def _prediction(
    *,
    center: str,
    case: str,
    alternative: str,
    direction: str,
    descriptor_hash: str,
    bacc: float,
    brier: float,
    log_loss: float,
) -> PosteriorUtilityPrediction:
    return PosteriorUtilityPrediction(
        center,
        case,
        alternative,
        direction,
        PRIMARY_FINGERPRINT_CONTROL_ID,
        1,
        (bacc,) * 5,
        (brier,) * 5,
        (log_loss,) * 5,
        bacc,
        brier,
        log_loss,
        0.8,
        0.1,
        True,
        descriptor_hash,
        canonical_hash([center, case, "ensemble"]),
    )


def _donor_fixture(
    *, helpful: bool = True
) -> tuple[tuple[PosteriorUtilityPrediction, ...], tuple[DonorUtilityRow, ...]]:
    predictions: list[PosteriorUtilityPrediction] = []
    outcomes: list[DonorUtilityRow] = []
    for donor in CENTERS[1:]:
        case = f"case-{donor}"
        for alternative in ALTERNATIVE_METHOD_IDS:
            for direction in DIRECTION_IDS:
                descriptor_hash = canonical_hash(
                    [donor, case, alternative, direction]
                )
                candidate = (
                    alternative == BASELINE_METHOD_ID
                    and direction == "zero_to_one"
                )
                if candidate:
                    point = (0.02, -0.01, -0.01)
                    actual = (
                        (0.01, -0.005, -0.005)
                        if helpful
                        else (-0.01, 0.005, 0.005)
                    )
                else:
                    point = (-0.02, 0.01, 0.01)
                    actual = (-0.02, 0.01, 0.01)
                predictions.append(
                    _prediction(
                        center=donor,
                        case=case,
                        alternative=alternative,
                        direction=direction,
                        descriptor_hash=descriptor_hash,
                        bacc=point[0],
                        brier=point[1],
                        log_loss=point[2],
                    )
                )
                outcomes.append(
                    DonorUtilityRow(
                        "0",
                        donor,
                        case,
                        alternative,
                        direction,
                        _features(alternative, direction),
                        1,
                        actual[0],
                        actual[1],
                        actual[2],
                        descriptor_hash,
                    )
                )
    return tuple(predictions), tuple(outcomes)


def _target_rectangle(
    *, far_shift: bool = False
) -> tuple[tuple[UtilityDescriptor, ...], tuple[PosteriorUtilityPrediction, ...]]:
    descriptors: list[UtilityDescriptor] = []
    predictions: list[PosteriorUtilityPrediction] = []
    endpoint_hash = canonical_hash(["target", "endpoint"])
    for alternative in ALTERNATIVE_METHOD_IDS:
        for direction in DIRECTION_IDS:
            values = _features(alternative, direction)
            if far_shift:
                values = tuple(value + 100.0 for value in values)
            descriptor = UtilityDescriptor(
                "0",
                "held-case",
                alternative,
                direction,
                UTILITY_FEATURE_NAMES,
                values,
                (f"sample-{alternative}-{direction}",),
                endpoint_hash,
            )
            candidate = (
                alternative == BASELINE_METHOD_ID
                and direction == "zero_to_one"
            )
            point = (0.02, -0.01, -0.01) if candidate else (-0.02, 0.01, 0.01)
            descriptors.append(descriptor)
            predictions.append(
                _prediction(
                    center="0",
                    case="held-case",
                    alternative=alternative,
                    direction=direction,
                    descriptor_hash=descriptor.descriptor_hash,
                    bacc=point[0],
                    brier=point[1],
                    log_loss=point[2],
                )
            )
    return tuple(descriptors), tuple(predictions)


def test_science_package_has_no_cross_diagnostic_imports() -> None:
    forbidden = {
        "fixed_bank_p_anchored_crossfit_posterior_utility_margin_router",
        "fixed_bank_p_anchored_directional_signed_utility_router",
        "fixed_bank_loo_nested_donor_endpoint_regret_router",
    }
    for path in PACKAGE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules = {
            *(
                node.module or ""
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            ),
            *(
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            ),
        }
        assert not any(
            fragment in module for fragment in forbidden for module in modules
        )


def test_json_boundary_normalizes_numpy_scalars_but_rejects_arrays() -> None:
    wrapped = {
        "passed": np.bool_(True),
        "count": np.int64(7),
        "score": np.float32(0.25),
    }
    assert json_native(wrapped) == {
        "passed": True,
        "count": 7,
        "score": 0.25,
    }
    with pytest.raises(ProtocolError, match="cannot convert ndarray"):
        json_native(np.asarray([True]))


def test_residual_orientation_matches_one_sided_safety_bounds() -> None:
    predictions, outcomes = _donor_fixture()
    observations = build_residual_observations(
        predictions, outcomes, allowed_donors=CENTERS[1:]
    )
    candidate = next(
        row
        for row in observations
        if row.alternative == BASELINE_METHOD_ID
        and row.direction == "zero_to_one"
    )
    assert candidate.bacc_error == pytest.approx(0.01)
    assert candidate.brier_error == pytest.approx(0.005)
    assert candidate.log_loss_error == pytest.approx(0.005)
    prediction = predictions[0]
    assert posterior_point(prediction, "bacc_contribution_delta") == pytest.approx(
        prediction.fold_bacc_deltas[0]
    )


def test_simultaneous_envelope_maximizes_before_center_quantile() -> None:
    predictions, outcomes = _donor_fixture()
    model = fit_simultaneous_envelope(
        predictions, outcomes, allowed_donors=CENTERS[1:]
    )
    assert len(model.residual_scales) == 18
    assert len(model.feature_references) == 6
    for envelope in model.direction_envelopes:
        assert len(envelope.donor_block_scores) == 8
        assert envelope.radius <= envelope.maximum_radius
        assert envelope.radius >= 0.0
        assert not envelope.to_payload()["finite_sample_coverage_claimed"]
    assert model.envelope_for("zero_to_one").radius > 0.0


def test_nested_tail_gate_authorizes_only_nonvacuous_safe_envelope() -> None:
    predictions, outcomes = _donor_fixture(helpful=True)
    calibration = calibrate_envelope(
        outer_target_center="0",
        control_id=PRIMARY_FINGERPRINT_CONTROL_ID,
        predictions=predictions,
        donor_rows=outcomes,
    )
    assert calibration.authorized
    assert calibration.selected_action_count == 8
    assert sum(row.selected_action_count for row in calibration.inner_replays) == 8
    assert calibration.donor_lower_tail_bacc_delta > 0.0

    harmful_predictions, harmful_outcomes = _donor_fixture(helpful=False)
    harmful = calibrate_envelope(
        outer_target_center="0",
        control_id=PRIMARY_FINGERPRINT_CONTROL_ID,
        predictions=harmful_predictions,
        donor_rows=harmful_outcomes,
    )
    assert not harmful.authorized


def test_label_free_shift_widens_bounds_and_controls_are_ordered() -> None:
    donor_predictions, donor_outcomes = _donor_fixture()
    calibration = calibrate_envelope(
        outer_target_center="0",
        control_id=PRIMARY_FINGERPRINT_CONTROL_ID,
        predictions=donor_predictions,
        donor_rows=donor_outcomes,
    )
    descriptors, predictions = _target_rectangle()
    shifted_descriptors, _ = _target_rectangle(far_shift=True)
    descriptor = next(
        row
        for row in descriptors
        if row.alternative == BASELINE_METHOD_ID
        and row.direction == "zero_to_one"
    )
    shifted = next(
        row
        for row in shifted_descriptors
        if row.alternative == BASELINE_METHOD_ID
        and row.direction == "zero_to_one"
    )
    prediction = next(
        row for row in predictions if row.descriptor_hash == descriptor.descriptor_hash
    )
    primary = certify_utility(
        descriptor,
        prediction,
        calibration,
        policy_id=MODEL_BASED_METHOD_ID,
        calibration_hash=calibration.calibration_hash,
    )
    shifted_prediction = replace(
        prediction, descriptor_hash=shifted.descriptor_hash
    )
    shifted_certificate = certify_utility(
        shifted,
        shifted_prediction,
        calibration,
        policy_id=MODEL_BASED_METHOD_ID,
        calibration_hash=calibration.calibration_hash,
    )
    zero_shift = certify_utility(
        descriptor,
        prediction,
        calibration,
        policy_id=ZERO_SHIFT_CONTROL_METHOD_ID,
        calibration_hash=calibration.calibration_hash,
    )
    minimax = certify_utility(
        descriptor,
        prediction,
        calibration,
        policy_id=MINIMAX_CONTROL_METHOD_ID,
        calibration_hash=calibration.calibration_hash,
    )
    assert shifted_certificate.shift_inflation > primary.shift_inflation
    assert shifted_certificate.lower_bacc_delta < primary.lower_bacc_delta
    assert zero_shift.shift_inflation == 1.0
    assert minimax.envelope_radius >= primary.envelope_radius


def test_selection_is_deterministic_and_p_wins_zero_bound() -> None:
    donor_predictions, donor_outcomes = _donor_fixture()
    calibration = calibrate_envelope(
        outer_target_center="0",
        control_id=PRIMARY_FINGERPRINT_CONTROL_ID,
        predictions=donor_predictions,
        donor_rows=donor_outcomes,
    )
    descriptors, predictions = _target_rectangle()
    certificates = tuple(
        certify_utility(
            descriptor,
            next(
                row
                for row in predictions
                if row.descriptor_hash == descriptor.descriptor_hash
            ),
            calibration,
            policy_id=MODEL_BASED_METHOD_ID,
            calibration_hash=calibration.calibration_hash,
        )
        for descriptor in descriptors
        if descriptor.direction == "zero_to_one"
    )
    selected = select_certificate_for_direction(certificates)
    assert selected is not None
    assert selected.alternative == BASELINE_METHOD_ID
    zeroed = tuple(
        replace(row, lower_bacc_delta=0.0) for row in certificates
    )
    assert select_certificate_for_direction(zeroed) is None


def test_primary_falls_back_on_failed_tail_gate_but_controls_remain_diagnostic() -> None:
    donor_predictions, donor_outcomes = _donor_fixture()
    authorized = calibrate_envelope(
        outer_target_center="0",
        control_id=PRIMARY_FINGERPRINT_CONTROL_ID,
        predictions=donor_predictions,
        donor_rows=donor_outcomes,
    )
    calibration = replace(authorized, authorized=False)
    descriptors, predictions = _target_rectangle()
    prediction_by_hash = {row.descriptor_hash: row for row in predictions}

    def certificates(policy_id: str):
        return tuple(
            certify_utility(
                descriptor,
                prediction_by_hash[descriptor.descriptor_hash],
                calibration,
                policy_id=policy_id,
                calibration_hash=calibration.calibration_hash,
            )
            for descriptor in descriptors
        )

    primary = select_directional_actions(
        descriptors,
        certificates(MODEL_BASED_METHOD_ID),
        calibration,
        policy_id=MODEL_BASED_METHOD_ID,
    )
    control = select_directional_actions(
        descriptors,
        certificates(ZERO_SHIFT_CONTROL_METHOD_ID),
        calibration,
        policy_id=ZERO_SHIFT_CONTROL_METHOD_ID,
    )
    assert all(row.selected_alternative == PORTFOLIO_METHOD_ID for row in primary)
    assert any(row.selected_alternative != PORTFOLIO_METHOD_ID for row in control)


def test_center_tail_risk_uses_worst_two_of_eight() -> None:
    values = tuple(float(value) for value in range(8))
    assert lower_tail_mean(values) == pytest.approx(0.5)
    assert upper_tail_mean(values) == pytest.approx(6.5)


def test_small_end_to_end_surface_seals_certificates_and_terminal_tree(
    tmp_path: Path,
) -> None:
    store_hash = canonical_hash({"store": "psscur-fixture"})
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
    assert all(
        len(rows) == 378
        for rows in preterminal.utility_certificates_by_policy.values()
    )
    assert len(preterminal.envelope_calibrations) == 18
    terminal = evaluate_terminal(preterminal)
    assert terminal.capability_report["status"] == "PASS"
    assert terminal.capability_report["route_decision_seal_count"] == 63
    assert terminal.diagnostic_summary["promotion_eligible"] is False
    persist_terminal(
        tmp_path,
        terminal=terminal,
        leakage_report={"status": "PASS", "numpy_flag": np.bool_(True)},
        publication_decision={"status": "DIAGNOSTIC_ONLY"},
        runtime_summary={"posterior_fit_count": np.int64(630)},
    )
    assert (tmp_path / "reports/diagnostic_summary.json").is_file()
    assert (tmp_path / "tables/information_gate.json").is_file()
