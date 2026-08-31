from __future__ import annotations

from dataclasses import fields, replace
from copy import deepcopy
import struct

import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.harp_v3 import fitting as harp_v3_fitting
from midogpp_thesis.cvae.routing.harp_v3 import (
    ActionKind,
    CaseActionSet,
    CaseTargetAction,
    CaseTrainingObservation,
    Comparison,
    EffectVector,
    PolicyConfig,
    decision_from_payload,
    decision_to_payload,
    deserialize_fit,
    fit_harp_v3,
    fit_to_payload,
    route_case,
    score_comparison,
    serialize_fit,
)
from midogpp_thesis.cvae.routing.harp_v3.compatibility import (
    assess_geometry,
    calibrate_geometry,
    upper_empirical_quantile,
)
from midogpp_thesis.cvae.routing.harp_v3.calibration import (
    FINITE_SAMPLE_RULE,
    calibrate_donor_residuals,
)


DONORS = ("0", "1", "2", "3", "4", "5")
FEATURES = ("case_coordinate", "candidate_quality")
SEAL = "a" * 64


def _training_rows() -> tuple[CaseTrainingObservation, ...]:
    rows: list[CaseTrainingObservation] = []
    for q_index, query in enumerate(DONORS):
        for case_index in range(12):
            coordinate = (case_index - 5.5) / 12.0 + 0.01 * q_index
            classes = (2 + case_index % 2, 2 + (case_index + 1) % 2)
            u_features = (coordinate, 0.0)
            rows.append(
                CaseTrainingObservation(
                    outer_target_id="H",
                    pseudo_query_id=query,
                    candidate_source_id=None,
                    case_id=f"{query}-case-{case_index:02d}",
                    comparison=Comparison.U_VS_B,
                    feature_names=FEATURES,
                    feature_values=u_features,
                    effects=EffectVector(
                        0.070 + 0.005 * coordinate,
                        -0.035 + 0.001 * coordinate,
                        -0.050 + 0.001 * coordinate,
                    ),
                    class_counts=classes,
                    pseudo_query_case_count=12,
                    pseudo_query_class_support_case_counts=(12, 12),
                )
            )
            for source_index, source in enumerate(DONORS):
                if source == query:
                    continue
                quality = (source_index - 2.5) / 10.0
                features = (coordinate, quality)
                rows.append(
                    CaseTrainingObservation(
                        outer_target_id="H",
                        pseudo_query_id=query,
                        candidate_source_id=source,
                        case_id=f"{query}-case-{case_index:02d}",
                        comparison=Comparison.HXE_VS_B,
                        feature_names=FEATURES,
                        feature_values=features,
                        effects=EffectVector(
                            0.145 + 0.010 * coordinate + 0.020 * quality,
                            -0.075 + 0.002 * coordinate - 0.005 * quality,
                            -0.100 + 0.002 * coordinate - 0.005 * quality,
                        ),
                        class_counts=classes,
                        pseudo_query_case_count=12,
                        pseudo_query_class_support_case_counts=(12, 12),
                    )
                )
                rows.append(
                    CaseTrainingObservation(
                        outer_target_id="H",
                        pseudo_query_id=query,
                        candidate_source_id=source,
                        case_id=f"{query}-case-{case_index:02d}",
                        comparison=Comparison.HXE_VS_U,
                        feature_names=FEATURES,
                        feature_values=features,
                        effects=EffectVector(
                            0.075 + 0.005 * coordinate + 0.020 * quality,
                            -0.040 + 0.001 * coordinate - 0.005 * quality,
                            -0.050 + 0.001 * coordinate - 0.005 * quality,
                        ),
                        class_counts=classes,
                        pseudo_query_case_count=12,
                        pseudo_query_class_support_case_counts=(12, 12),
                    )
                )
    return tuple(rows)


@pytest.fixture(scope="module")
def fitted():
    return fit_harp_v3(
        _training_rows(),
        outer_target_id="H",
        alpha_grid=(0.001, 0.01),
        residual_quantile=0.8,
        geometry_quantile=0.95,
    )


def _action(
    kind: ActionKind,
    *,
    candidate: str | None = None,
    features: tuple[float, float] = (0.0, -0.05),
    probabilities: tuple[float, float] = (0.4, 0.6),
) -> CaseTargetAction:
    return CaseTargetAction(
        outer_target_id="H",
        target_query_id="H",
        case_id="target-case",
        action_kind=kind,
        candidate_source_id=candidate,
        feature_names=FEATURES,
        feature_values=features,
        sample_ids=("sample-a", "sample-b"),
        probability_bytes=tuple(struct.pack("<f", value) for value in probabilities),
        prediction_seal_hash=SEAL,
        expert_weight=1.0 if kind is ActionKind.HXE else 0.0,
    )


def _policy() -> PolicyConfig:
    return PolicyConfig(
        min_donor_count=4,
        min_paired_case_count=20,
        min_compatibility_shrinkage=0.05,
    )


def test_target_contract_cannot_represent_labels_and_hxe_is_physical() -> None:
    forbidden = {"truth", "truth_class", "label", "outcome", "effects"}
    assert forbidden.isdisjoint({field.name for field in fields(CaseTargetAction)})
    with pytest.raises(ProtocolError, match="lambda=1"):
        replace(_action(ActionKind.HXE, candidate="0"), expert_weight=0.75)
    with pytest.raises(ProtocolError, match="held-out target expert"):
        _action(ActionKind.HXE, candidate="H")


def test_outer_and_nested_delete_donor_exclusion_is_structural(fitted) -> None:
    row = _training_rows()[0]
    with pytest.raises(ProtocolError, match="Outer H"):
        replace(row, pseudo_query_id="H")
    for deleted in fitted.delete_donor_fits:
        assert "H" in deleted.model.excluded_center_ids
        assert deleted.donor_id in deleted.model.excluded_center_ids
        assert deleted.donor_id not in deleted.model.training_query_ids
        assert deleted.donor_id not in deleted.model.training_candidate_ids
        for fold in deleted.inner_selection.fold_scores:
            assert fold.heldout_donor_id not in fold.training_query_ids
            assert fold.heldout_donor_id not in fold.training_candidate_ids
    geometry = fitted.geometry(Comparison.HXE_VS_B)
    assert geometry.source_donor_ids == DONORS
    assert set(geometry.heldout_block_sizes) == {len(DONORS) - 1}
    assert not geometry.formal_conformal_claimed
    assert "d_minus_1" in geometry.calibration_method


def test_per_fit_ridge_memo_reuses_only_the_exact_bound_surface(monkeypatch) -> None:
    rows = _training_rows()
    memo = harp_v3_fitting._RidgeFitMemo(rows)
    deleted = tuple(
        row
        for row in rows
        if row.pseudo_query_id != "0" and row.candidate_source_id != "0"
    )
    real_fit = harp_v3_fitting.fit_shared_design_ridge
    calls = 0

    def counted_fit(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(harp_v3_fitting, "fit_shared_design_ridge", counted_fit)
    first = memo.fit(deleted, alpha=0.01, excluded_center_ids=("H", "0"))
    equivalent = memo.fit(
        deleted,
        alpha=0.01,
        excluded_center_ids=("0", "H", "0"),
    )
    assert equivalent is first
    assert calls == 1

    # Row order, alpha, and normalized exclusions are all scientific cache-key
    # dimensions; changing any one must trigger a distinct deterministic fit.
    assert memo.fit(
        tuple(reversed(deleted)),
        alpha=0.01,
        excluded_center_ids=("H", "0"),
    ) is not first
    assert memo.fit(
        deleted,
        alpha=0.1,
        excluded_center_ids=("H", "0"),
    ) is not first
    assert memo.fit(
        deleted,
        alpha=0.01,
        excluded_center_ids=("H", "0", "unused"),
    ) is not first
    assert calls == 4

    counterfeit = list(deleted)
    counterfeit[0] = replace(
        counterfeit[0],
        feature_values=(
            counterfeit[0].feature_values[0] + 0.01,
            *counterfeit[0].feature_values[1:],
        ),
    )
    with pytest.raises(ProtocolError, match="identity is ambiguous"):
        memo.fit(
            tuple(counterfeit),
            alpha=0.01,
            excluded_center_ids=("H", "0"),
        )
    with pytest.raises(ProtocolError, match="exclusion contract"):
        memo.fit(deleted, alpha=0.01, excluded_center_ids=("0",))
    assert calls == 4


def test_memoized_fit_is_payload_identical_and_reduces_ridge_solves(monkeypatch) -> None:
    rows = _training_rows()
    real_fit = harp_v3_fitting.fit_shared_design_ridge
    memoized_fit = harp_v3_fitting._RidgeFitMemo.fit
    calls = 0

    def counted_fit(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_fit(*args, **kwargs)

    def uncached_fit(_memo, fit_rows, *, alpha, excluded_center_ids):
        return harp_v3_fitting.fit_shared_design_ridge(
            fit_rows,
            alpha=alpha,
            excluded_center_ids=excluded_center_ids,
        )

    monkeypatch.setattr(harp_v3_fitting, "fit_shared_design_ridge", counted_fit)
    monkeypatch.setattr(harp_v3_fitting._RidgeFitMemo, "fit", uncached_fit)
    reference = fit_harp_v3(rows, outer_target_id="H", alpha_grid=(0.001, 0.01))
    uncached_calls = calls

    calls = 0
    monkeypatch.setattr(harp_v3_fitting._RidgeFitMemo, "fit", memoized_fit)
    optimized = fit_harp_v3(rows, outer_target_id="H", alpha_grid=(0.001, 0.01))
    optimized_calls = calls

    assert fit_to_payload(optimized) == fit_to_payload(reference)
    assert uncached_calls == 214
    assert optimized_calls == 83


def test_incomplete_or_incoherent_training_hierarchy_fails_closed() -> None:
    rows = _training_rows()
    missing = next(
        index
        for index, row in enumerate(rows)
        if row.comparison is Comparison.HXE_VS_U
    )
    with pytest.raises(ProtocolError, match="incomplete physical expert triplet"):
        fit_harp_v3(
            (*rows[:missing], *rows[missing + 1 :]),
            outer_target_id="H",
            alpha_grid=(0.01,),
        )
    corrupt = list(rows)
    index = next(
        index
        for index, row in enumerate(corrupt)
        if row.comparison is Comparison.HXE_VS_U
    )
    corrupt[index] = replace(
        corrupt[index],
        effects=replace(
            corrupt[index].effects,
            case_equal_bacc_contribution_gain=(
                corrupt[index].effects.case_equal_bacc_contribution_gain + 0.01
            ),
        ),
    )
    with pytest.raises(ProtocolError, match="algebraically incoherent"):
        fit_harp_v3(corrupt, outer_target_id="H", alpha_grid=(0.01,))


def test_residual_envelope_is_donor_case_balanced_and_finite_sample() -> None:
    rows = (
        ("a", "a-case-0", 0.5),
        ("a", "a-case-1", 1.0),
        ("b", "b-case-0", 4.0),
        ("b", "b-case-1", 8.0),
    )
    predicted = tuple(EffectVector(value, -value, -value) for _, _, value in rows)
    observed = tuple(EffectVector(0.0, 0.0, 0.0) for _ in rows)
    calibration = calibrate_donor_residuals(
        Comparison.HXE_VS_B,
        predicted,
        observed,
        tuple(donor for donor, _, _ in rows),
        tuple(case for _, case, _ in rows),
        quantile_level=0.8,
    )
    assert calibration.donor_ids == ("a", "b")
    assert calibration.donor_case_counts == (2, 2)
    assert calibration.calibration_case_block_count == 4
    assert calibration.joint_harm_quantile == max(
        calibration.donor_joint_harm_quantiles
    )
    assert calibration.finite_sample_rule == FINITE_SAMPLE_RULE

    # Repeating candidate rows inside one donor/case changes only the raw-row
    # audit count; it cannot manufacture extra calibration mass.
    duplicated = calibrate_donor_residuals(
        Comparison.HXE_VS_B,
        (*predicted, *predicted[:2]),
        (*observed, *observed[:2]),
        (*tuple(donor for donor, _, _ in rows), "a", "a"),
        (*tuple(case for _, case, _ in rows), "a-case-0", "a-case-1"),
        quantile_level=0.8,
    )
    assert duplicated.calibration_row_count == calibration.calibration_row_count + 2
    assert duplicated.calibration_case_block_count == calibration.calibration_case_block_count
    assert duplicated.endpoint_scales == calibration.endpoint_scales
    assert duplicated.joint_harm_quantile == calibration.joint_harm_quantile


def test_residual_envelope_does_not_let_large_low_harm_donor_hide_small_donor() -> None:
    rows = [
        ("large", f"large-case-{index:03d}", 0.25) for index in range(100)
    ]
    rows.extend(("small", f"small-case-{index}", value) for index, value in enumerate((4.0, 8.0)))
    predicted = tuple(EffectVector(value, -value, -value) for _, _, value in rows)
    observed = tuple(EffectVector(0.0, 0.0, 0.0) for _ in rows)
    calibration = calibrate_donor_residuals(
        Comparison.HXE_VS_U,
        predicted,
        observed,
        tuple(donor for donor, _, _ in rows),
        tuple(case for _, case, _ in rows),
        quantile_level=0.8,
    )
    assert calibration.donor_case_counts == (100, 2)
    small_index = calibration.donor_ids.index("small")
    assert calibration.joint_harm_quantile == calibration.donor_joint_harm_quantiles[
        small_index
    ]


def test_calibrated_geometry_handles_intercept_scale_and_rejects_extrapolation(fitted) -> None:
    in_support = _action(ActionKind.HXE, candidate="2")
    accepted = score_comparison(
        fitted, in_support, Comparison.HXE_VS_B, config=_policy()
    )
    assert min(accepted.geometry.raw_leverages) > 1.0
    assert accepted.geometry.maximum_ratio <= 1.0
    assert accepted.eligible

    extrapolated = replace(in_support, feature_values=(100.0, 100.0))
    rejected = score_comparison(
        fitted, extrapolated, Comparison.HXE_VS_B, config=_policy()
    )
    assert rejected.geometry.maximum_ratio > 1.0
    assert "calibrated_geometry_extrapolation" in rejected.rejection_reasons
    assert not rejected.eligible
    assert rejected.geometry.empirical_tail_probability == pytest.approx(
        rejected.geometry.finite_sample_tail_floor
    )

    with pytest.raises(ProtocolError, match="predeclared source quantile exactly"):
        score_comparison(
            fitted,
            in_support,
            Comparison.HXE_VS_B,
            config=replace(_policy(), max_calibrated_geometry_ratio=1.01),
        )


def test_geometry_calibrates_ensemble_max_not_individual_leverage_q95() -> None:
    raw: list[float] = []
    donors: list[str] = []
    block_ids: list[str] = []
    maximum = 0
    for donor in DONORS:
        for action_index in range(10):
            maximum += 1
            block = f"{donor}:action:{action_index}"
            # One large member plus four benign members.  The q95 of pooled
            # individual leverages is therefore the wrong (lower) statistic.
            values = (0.25, 0.5, 0.75, 1.0, float(maximum))
            raw.extend(values)
            donors.extend((donor,) * len(values))
            block_ids.extend((block,) * len(values))

    calibration = calibrate_geometry(
        Comparison.HXE_VS_B,
        raw,
        donors,
        block_ids,
        quantile_level=0.95,
    )
    individual_q95 = upper_empirical_quantile(raw, 0.95)
    target_ensemble = (10.0, 20.0, 30.0, 40.0, 50.0, 50.0)
    assessment = assess_geometry(calibration, target_ensemble)

    assert individual_q95 < max(target_ensemble) < calibration.reference_quantile
    assert assessment.maximum_ratio < 1.0
    assert assessment.empirical_tail_probability > assessment.finite_sample_tail_floor
    assert assessment.calibration_block_count == 60

    extrapolated = assess_geometry(calibration, (100.0,) * len(DONORS))
    assert extrapolated.maximum_ratio > 1.0
    assert extrapolated.empirical_tail_probability == pytest.approx(
        extrapolated.finite_sample_tail_floor
    )


def test_hierarchical_policy_prefers_safe_h_then_u_then_exact_b(fitted) -> None:
    baseline = _action(ActionKind.B, probabilities=(0.4123456789, 0.5876543211))
    uniform = _action(ActionKind.U, features=(0.0, 0.0), probabilities=(0.45, 0.55))
    expert = _action(ActionKind.HXE, candidate="2", probabilities=(0.3, 0.7))
    expert_decision = route_case(
        CaseActionSet(baseline=baseline, uniform=uniform, experts=(expert,), expected_candidate_source_ids=("2",)),
        fitted,
        config=_policy(),
    )
    assert expert_decision.selected_kind is ActionKind.HXE
    assert expert_decision.selected_source_id == "2"
    assert expert_decision.output_probability_bytes == expert.probability_bytes

    bad_expert = replace(expert, feature_values=(100.0, 100.0))
    uniform_decision = route_case(
        CaseActionSet(baseline=baseline, uniform=uniform, experts=(bad_expert,), expected_candidate_source_ids=("2",)),
        fitted,
        config=_policy(),
    )
    assert uniform_decision.selected_kind is ActionKind.U
    assert uniform_decision.output_probability_bytes == uniform.probability_bytes

    bad_uniform = replace(uniform, feature_values=(100.0, 100.0))
    fallback = route_case(
        CaseActionSet(baseline=baseline, uniform=bad_uniform, experts=(bad_expert,), expected_candidate_source_ids=("2",)),
        fitted,
        config=_policy(),
    )
    assert fallback.selected_kind is ActionKind.B
    assert fallback.output_probability_bytes == baseline.probability_bytes
    assert b"".join(fallback.output_probability_bytes) == b"".join(
        baseline.probability_bytes
    )
    assert fallback.reason == "EXACT_B_FALLBACK_NO_HIERARCHICALLY_SAFE_ACTION"


def test_decision_is_case_consistent_and_ties_are_deterministic(fitted) -> None:
    baseline = _action(ActionKind.B)
    uniform = _action(ActionKind.U, features=(0.0, 0.0))
    expert_0 = _action(ActionKind.HXE, candidate="2", probabilities=(0.2, 0.8))
    expert_1 = _action(ActionKind.HXE, candidate="3", features=(0.0, 0.05), probabilities=(0.1, 0.9))
    decision = route_case(
        CaseActionSet(
            baseline=baseline,
            uniform=uniform,
            experts=(expert_0, expert_1),
            expected_candidate_source_ids=("2", "3"),
        ),
        fitted,
        config=_policy(),
    )
    assert decision.sample_ids == ("sample-a", "sample-b")
    assert len(decision.output_probability_bytes) == 2
    assert decision.selected_source_id in {"2", "3"}
    # Repeated calls preserve the same expert and the whole-case byte vector.
    repeated = route_case(
        CaseActionSet(
            baseline=baseline,
            uniform=uniform,
            experts=(expert_0, expert_1),
            expected_candidate_source_ids=("2", "3"),
        ),
        fitted,
        config=_policy(),
    )
    assert repeated.selected_source_id == decision.selected_source_id
    assert repeated.output_probability_bytes == decision.output_probability_bytes

    with pytest.raises(ProtocolError, match="complete sealed physical candidate universe"):
        CaseActionSet(
            baseline=baseline,
            uniform=uniform,
            experts=(expert_0,),
            expected_candidate_source_ids=("2", "3"),
        )


def test_fit_and_full_action_audit_round_trip_with_hash_binding(fitted) -> None:
    text = serialize_fit(fitted)
    rebuilt = deserialize_fit(text)
    assert fit_to_payload(rebuilt) == fit_to_payload(fitted)
    assert not rebuilt.full_model.coefficients.flags.writeable

    baseline = _action(ActionKind.B)
    uniform = _action(ActionKind.U, features=(0.0, 0.0))
    expert = _action(ActionKind.HXE, candidate="2")
    decision = route_case(
        CaseActionSet(
            baseline=baseline,
            uniform=uniform,
            experts=(expert,),
            expected_candidate_source_ids=("2",),
        ),
        rebuilt,
        config=_policy(),
    )
    payload = decision_to_payload(decision)
    reconstructed = decision_from_payload(payload)
    assert decision_to_payload(reconstructed) == payload
    expert_audit = reconstructed.action_audits[-1]
    assert expert_audit.comparison_scores[0].geometry.calibrated_ratios
    assert expert_audit.comparison_scores[0].support.donor_count >= 4
    assert (
        expert_audit.comparison_scores[
            0
        ].geometry_adjusted_bounds.case_equal_bacc_contribution_gain_lower
        > 0
    )

    tampered = deepcopy(fit_to_payload(fitted))
    tampered["full_model"]["coefficients"]["data_base64"] = "AAAA"
    with pytest.raises(ProtocolError, match="hash drifted"):
        from midogpp_thesis.cvae.routing.harp_v3 import fit_from_payload

        fit_from_payload(tampered)
