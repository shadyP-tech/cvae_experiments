from __future__ import annotations

from dataclasses import replace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_hierarchical_residual_stacker.contracts import (
    BinaryLabel,
    SampleActionProbability,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_hierarchical_residual_stacker.scientific_constants import (  # noqa: E501
    MIDOGPP_CENTERS,
    candidate_sources,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_signed_error_gate import (
    CorrectionRow,
    SignedErrorLabelCapability,
    SignedFoldProducts,
    Standardization,
    assert_consumed_test_diagnostic_only,
    build_gradient_targets,
    build_signed_features,
    canonical_consumed_test_protocol,
    compose_signed_predictions,
    correction_surface_hash,
    fit_signed_gate,
    fit_signed_gate_decision,
    margin_gate,
    permute_feature_alignment,
    predict_corrections,
    record_durable_fold_seals,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_signed_error_gate.features import (
    feature_context_hash,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.diagnostics.fixed_bank_signed_error_gate.constants import (
    FEATURE_NAMES,
    LAMBDA_GRID,
    METHOD_IDS,
)


def _surface() -> tuple[tuple[SampleActionProbability, ...], tuple[BinaryLabel, ...]]:
    probabilities: list[SampleActionProbability] = []
    labels: list[BinaryLabel] = []
    for center_index, center in enumerate(MIDOGPP_CENTERS):
        for case_index in range(2):
            case = f"case-{center}-{case_index}"
            for sample_index in range(4):
                label = sample_index % 2
                sample = f"{case}-sample-{sample_index}"
                # Deliberately wrong near-threshold baseline. The fixed bank's
                # signed residual points toward the proper-loss correction.
                baseline = (0.58, 0.42)[label]
                probabilities.append(
                    SampleActionProbability(center, case, sample, "B", baseline)
                )
                for source_index, source in enumerate(candidate_sources(center)):
                    signed_shift = (1 if label else -1) * (
                        0.08 + 0.002 * source_index + 0.0001 * center_index
                    )
                    probabilities.append(
                        SampleActionProbability(
                            center,
                            case,
                            sample,
                            source,
                            min(max(baseline + signed_shift, 0.001), 0.999),
                        )
                    )
                labels.append(
                    BinaryLabel(center, case, sample, label, "loco_donor")
                )
    return tuple(probabilities), tuple(labels)


def _fit_inputs(probabilities, labels, target="0"):
    outer = build_signed_features(
        probabilities, excluded_candidate_centers=(target,)
    )
    nested = {
        query: build_signed_features(
            probabilities, excluded_candidate_centers=(target, query)
        )
        for query in MIDOGPP_CENTERS
        if query != target
    }
    donor_labels = tuple(row for row in labels if row.target_center != target)
    gradients = build_gradient_targets(
        probabilities, donor_labels, heldout_target=target
    )
    keys = {row.sample_key for row in gradients}
    return (
        tuple(row for row in outer if row.sample_key in keys),
        {
            query: tuple(row for row in rows if row.sample_key in keys)
            for query, rows in nested.items()
        },
        outer,
        nested,
        gradients,
    )


class _RecordingSealCapability:
    def __init__(self) -> None:
        self.method_seals: list[tuple[str, int, str, str]] = []
        self.preevaluation_seal: tuple[str, str, int] | None = None

    def open_loco_donor_labels(self, heldout_target: str):
        raise AssertionError("not used by durable-fold seal tests")

    def open_fold_support_labels(self, target_center: str, fold_ordinal: int):
        raise AssertionError("not used by durable-fold seal tests")

    def record_loco_model_seals(
        self,
        heldout_target: str,
        global_model_hash: str,
        residual_model_hash: str,
        permuted_model_hash: str,
    ) -> None:
        raise AssertionError("not used by durable-fold seal tests")

    def record_fold_method_decision(
        self,
        target_center: str,
        fold_ordinal: int,
        method_id: str,
        decision_hash: str,
    ) -> None:
        self.method_seals.append(
            (target_center, fold_ordinal, method_id, decision_hash)
        )

    def record_preevaluation_seals(
        self,
        decision_seal_hash: str,
        permutation_provenance_hash: str,
        *,
        decision_count: int,
    ) -> None:
        self.preevaluation_seal = (
            decision_seal_hash,
            permutation_provenance_hash,
            decision_count,
        )


def _fold_products() -> SignedFoldProducts:
    decisions = tuple(
        {
            "target_center": target,
            "fold_ordinal": ordinal,
            "method_decision_hashes": {
                method: "a" * 64 for method in METHOD_IDS
            },
        }
        for target in MIDOGPP_CENTERS
        for ordinal in range(5)
    )
    return SignedFoldProducts(
        decisions,
        {method: () for method in METHOD_IDS},
        "b" * 64,
        "c" * 64,
        "d" * 64,
        "e" * 64,
    )


def test_features_are_sample_level_label_blind_and_have_no_pseudoclass_branch() -> None:
    probabilities, _labels = _surface()
    features = build_signed_features(probabilities)
    assert len(features) == len(MIDOGPP_CENTERS) * 2 * 4
    assert "absolute_baseline_logit_margin" in FEATURE_NAMES
    assert all("class" not in name for name in FEATURE_NAMES)
    assert all(
        row.to_payload()["baseline_predicted_class_branch_used"] is False
        for row in features
    )


def test_permutation_control_is_deterministic_complete_block_derangement() -> None:
    probabilities, _labels = _surface()
    features = build_signed_features(probabilities)
    first = permute_feature_alignment(features)
    second = permute_feature_alignment(tuple(reversed(features)))
    assert first == second
    assert all(row.feature_origin_sample_id != row.sample_id for row in first)
    for center in MIDOGPP_CENTERS:
        original = sorted(row.values for row in features if row.target_center == center)
        permuted = sorted(row.values for row in first if row.target_center == center)
        assert original == permuted


def test_strict_outer_center_fit_is_invariant_to_target_expert_probability_poison() -> None:
    probabilities, labels = _surface()
    legal, nested_legal, _outer, _nested, gradients = _fit_inputs(
        probabilities, labels
    )
    original = fit_signed_gate(
        legal,
        gradients,
        target_center="0",
        family="R",
        nested_training_features=nested_legal,
    )
    poisoned_probabilities = tuple(
        replace(row, probability=0.999 - 0.01 * (index % 3))
        if row.action_id == "0" and row.target_center != "0"
        else row
        for index, row in enumerate(probabilities)
    )
    poisoned_legal, poisoned_nested, *_rest = _fit_inputs(
        poisoned_probabilities, labels
    )
    refit = fit_signed_gate(
        poisoned_legal,
        gradients,
        target_center="0",
        family="R",
        nested_training_features=poisoned_nested,
    )
    assert original.final_model.model_hash == refit.final_model.model_hash
    assert original.final_model.donor_centers == tuple(
        center for center in MIDOGPP_CENTERS if center != "0"
    )


def test_signed_model_learns_one_direct_correction_without_class_branch() -> None:
    probabilities, labels = _surface()
    features, nested_legal, outer, nested, gradients = _fit_inputs(
        probabilities, labels
    )
    fitted = fit_signed_gate(
        features,
        gradients,
        target_center="0",
        family="R",
        nested_training_features=nested_legal,
    )
    corrections = predict_corrections(
        fitted, outer, nested_prediction_features=nested
    )
    label_by_key = {row.sample_key: row.label for row in labels}
    assert corrections
    assert all(
        row.raw_correction > 0.0 if label_by_key[row.sample_key] else row.raw_correction < 0.0
        for row in corrections
    )


def test_prediction_rejects_duplicate_main_and_nested_feature_rows() -> None:
    probabilities, labels = _surface()
    features, nested_legal, outer, nested, gradients = _fit_inputs(
        probabilities, labels
    )
    fitted = fit_signed_gate(
        features,
        gradients,
        target_center="0",
        family="R",
        nested_training_features=nested_legal,
    )
    with pytest.raises(ProtocolError, match="duplicate sample rows"):
        predict_corrections(
            fitted,
            (*outer, outer[0]),
            nested_prediction_features=nested,
        )
    duplicate_nested = dict(nested)
    duplicate_nested["1"] = (*nested["1"], nested["1"][0])
    with pytest.raises(ProtocolError, match="duplicate sample rows"):
        predict_corrections(
            fitted,
            outer,
            nested_prediction_features=duplicate_nested,
        )


def test_prediction_rejects_final_and_nested_context_tampering() -> None:
    probabilities, labels = _surface()
    features, nested_legal, outer, nested, gradients = _fit_inputs(
        probabilities, labels
    )
    fitted = fit_signed_gate(
        features,
        gradients,
        target_center="0",
        family="R",
        nested_training_features=nested_legal,
    )
    target_index = next(
        index for index, row in enumerate(outer) if row.target_center == "0"
    )
    wrong_final_context = list(outer)
    wrong_final_context[target_index] = replace(
        outer[target_index], context_excluded_centers=()
    )
    with pytest.raises(ProtocolError, match="final feature exclusions"):
        predict_corrections(
            fitted,
            wrong_final_context,
            nested_prediction_features=nested,
        )

    wrong_final_candidates = list(outer)
    target_row = outer[target_index]
    wrong_final_candidates[target_index] = replace(
        target_row,
        candidate_source_ids=target_row.candidate_source_ids[:-1],
    )
    with pytest.raises(ProtocolError, match="final candidate sources"):
        predict_corrections(
            fitted,
            wrong_final_candidates,
            nested_prediction_features=nested,
        )

    query = "1"
    nested_rows = list(nested[query])
    nested_target_index = next(
        index for index, row in enumerate(nested_rows) if row.target_center == "0"
    )
    nested_target_row = nested_rows[nested_target_index]
    wrong_nested_exclusions = tuple(sorted(("0", "2")))
    nested_rows[nested_target_index] = replace(
        nested_target_row,
        context_excluded_centers=wrong_nested_exclusions,
        candidate_source_ids=tuple(
            source
            for source in candidate_sources(nested_target_row.target_center)
            if source not in wrong_nested_exclusions
        ),
    )
    wrong_nested_context = {**nested, query: tuple(nested_rows)}
    with pytest.raises(ProtocolError, match="nested.*exclusions"):
        predict_corrections(
            fitted,
            outer,
            nested_prediction_features=wrong_nested_context,
        )

    nested_rows = list(nested[query])
    nested_target_row = nested_rows[nested_target_index]
    nested_rows[nested_target_index] = replace(
        nested_target_row,
        candidate_source_ids=nested_target_row.candidate_source_ids[:-1],
    )
    wrong_nested_candidates = {**nested, query: tuple(nested_rows)}
    with pytest.raises(ProtocolError, match="nested.*candidate sources"):
        predict_corrections(
            fitted,
            outer,
            nested_prediction_features=wrong_nested_candidates,
        )


def test_nested_query_context_excludes_query_expert_and_is_poison_invariant() -> None:
    probabilities, _labels = _surface()
    context = build_signed_features(
        probabilities, excluded_candidate_centers=("0", "1")
    )
    poisoned = tuple(
        replace(row, probability=0.999)
        if row.action_id in {"0", "1"} and row.target_center not in {"0", "1"}
        else row
        for row in probabilities
    )
    after = build_signed_features(
        poisoned, excluded_candidate_centers=("0", "1")
    )
    assert feature_context_hash(context, control="aligned") == feature_context_hash(
        after, control="aligned"
    )
    assert all("0" not in row.candidate_source_ids for row in context)
    assert all("1" not in row.candidate_source_ids for row in context)


@pytest.mark.parametrize("scope", ("target_support", "terminal_evaluation"))
def test_gradient_fit_rejects_support_and_evaluation_label_scopes(scope: str) -> None:
    probabilities, labels = _surface()
    donor = tuple(
        replace(row, label_scope=scope)
        for row in labels
        if row.target_center != "0"
    )
    with pytest.raises(ProtocolError, match="LOCO labels"):
        build_gradient_targets(probabilities, donor, heldout_target="0")


def test_margin_gate_concentrates_correction_near_threshold_and_zero_scale_is_bcal() -> None:
    probabilities, _labels = _surface()
    target_probabilities = tuple(row for row in probabilities if row.target_center == "0")
    keys = sorted({row.sample_key for row in target_probabilities})
    corrections = tuple(
        CorrectionRow(*key, "R", 0.5, 0.01, 0.5, True) for key in keys
    )
    zero = compose_signed_predictions(
        target_probabilities,
        corrections,
        intercept=0.05,
        residual_scale=0.0,
        method_id="R_safe",
        safe=True,
    )
    assert margin_gate(0.5) > margin_gate(0.9) > margin_gate(0.99)
    from midogpp_thesis.cvae.diagnostics.fixed_bank_hierarchical_residual_stacker.composition import (  # noqa: E501
        calibrated_baseline_predictions,
    )

    baseline = calibrated_baseline_predictions(target_probabilities, intercept=0.05)
    assert tuple(row.probability for row in zero) == tuple(
        row.probability for row in baseline
    )


def test_lambda_path_and_fail_closed_reason_are_always_recorded() -> None:
    probabilities, labels = _surface()
    target_probabilities = tuple(row for row in probabilities if row.target_center == "0")
    support_labels = tuple(
        replace(row, label_scope="target_support")
        for row in labels
        if row.target_center == "0"
    )
    keys = sorted({row.sample_key for row in target_probabilities})
    corrections = tuple(
        CorrectionRow(*key, "R", 0.0, 0.0, 0.0, False) for key in keys
    )
    decision = fit_signed_gate_decision(
        target_probabilities, corrections, support_labels
    )
    assert tuple(row.residual_scale for row in decision.lambda_path) == LAMBDA_GRID
    assert all(row.threshold_crossing_count == 0 for row in decision.lambda_path)
    assert decision.selected_scale == 0.0
    assert decision.fallback_reason == "no_uncertainty_admissible_corrections"


def test_consumed_test_protocol_cannot_authorize_or_feed() -> None:
    protocol = canonical_consumed_test_protocol()
    assert_consumed_test_diagnostic_only(protocol)
    payload = protocol.to_payload()
    assert payload["evidence_status"] == "EXPLORATORY_CONSUMED_DATA_ONLY"
    assert payload["fresh_evidence"] is False
    assert payload["may_authorize_routing"] is False
    assert payload["may_authorize_promotion"] is False
    assert payload["may_feed_another_experiment"] is False
    assert payload["exact_bacc_lcb_relaxed"] is False
    assert payload["R_raw_and_R_safe_separately_sealed"] is True
    assert payload["terminal_evaluation_runtime"] == {
        "bootstrap_replicates": 10_000,
        "bootstrap_seed": 90_912_028,
        "multiprocessing_start_method": "spawn",
    }
    with pytest.raises(ProtocolError, match="boundary"):
        assert_consumed_test_diagnostic_only(
            replace(protocol, may_authorize_routing=True)
        )
    with pytest.raises(ProtocolError, match="boundary"):
        assert_consumed_test_diagnostic_only(
            replace(protocol, bootstrap_replicates=9_999)
        )


def test_gradient_and_composition_reject_duplicate_or_wrong_method_surfaces() -> None:
    probabilities, labels = _surface()
    donor = tuple(row for row in labels if row.target_center != "0")
    baseline_duplicate = next(
        row
        for row in probabilities
        if row.target_center != "0" and row.action_id == "B"
    )
    with pytest.raises(ProtocolError, match="aligned baseline"):
        build_gradient_targets(
            (*probabilities, baseline_duplicate), donor, heldout_target="0"
        )

    target_probabilities = tuple(
        row for row in probabilities if row.target_center == "0"
    )
    keys = sorted(
        {
            row.sample_key
            for row in target_probabilities
            if row.action_id == "B"
        }
    )
    corrections = tuple(
        CorrectionRow(*key, "R", 0.1, 0.01, 0.1, True) for key in keys
    )
    with pytest.raises(ProtocolError, match="method/grid"):
        compose_signed_predictions(
            target_probabilities,
            corrections,
            intercept=0.0,
            residual_scale=0.05,
            method_id="G",
            safe=True,
        )
    with pytest.raises(ProtocolError, match="duplicated"):
        compose_signed_predictions(
            (
                *target_probabilities,
                next(row for row in target_probabilities if row.action_id == "B"),
            ),
            corrections,
            intercept=0.0,
            residual_scale=0.05,
            method_id="R_safe",
            safe=True,
        )


def test_model_provenance_G_zero_coordinates_and_separate_surface_hashes() -> None:
    probabilities, labels = _surface()
    features, nested_legal, outer, nested, gradients = _fit_inputs(
        probabilities, labels
    )
    fitted = fit_signed_gate(
        features,
        gradients,
        target_center="0",
        family="G",
        nested_training_features=nested_legal,
    )
    assert fitted.fit_hash
    assert tuple(alpha for alpha, _mse in fitted.validation_mse_by_alpha) == (
        0.1,
        1.0,
        10.0,
    )
    assert all(value == 0.0 for value in fitted.final_model.coefficients[1:])
    corrections = predict_corrections(
        fitted, outer, nested_prediction_features=nested
    )
    assert correction_surface_hash(corrections, surface="raw") != (
        correction_surface_hash(corrections, surface="safe")
    )

    means = [0.0] * 10
    scales = [1.0] * 10
    standardization = Standardization(means, scales)  # type: ignore[arg-type]
    means[0] = 99.0
    scales[0] = 99.0
    assert standardization.means[0] == 0.0
    assert standardization.scales[0] == 1.0


def test_durable_fold_seals_require_exact_center_by_five_topology() -> None:
    products = _fold_products()
    capability = _RecordingSealCapability()
    typed_capability: SignedErrorLabelCapability = capability
    record_durable_fold_seals(typed_capability, products)
    assert len(capability.method_seals) == len(MIDOGPP_CENTERS) * 5 * len(
        METHOD_IDS
    )
    assert capability.preevaluation_seal == (
        "b" * 64,
        "c" * 64,
        len(MIDOGPP_CENTERS) * 5 * len(METHOD_IDS),
    )

    malformed = list(products.decisions)
    malformed[0] = {**malformed[0], "target_center": "4"}
    untouched = _RecordingSealCapability()
    with pytest.raises(ProtocolError, match="exact center-by-five-fold topology"):
        record_durable_fold_seals(
            untouched,
            replace(products, decisions=tuple(malformed)),
        )
    assert untouched.method_seals == []
    assert untouched.preevaluation_seal is None


@pytest.mark.parametrize(
    "bad_hash",
    (
        "A" * 64,
        "a" * 63,
        "g" * 64,
        7,
    ),
)
def test_durable_fold_seals_reject_noncanonical_method_hashes(
    bad_hash: object,
) -> None:
    products = _fold_products()
    malformed = list(products.decisions)
    hashes = dict(malformed[0]["method_decision_hashes"])
    hashes[METHOD_IDS[0]] = bad_hash
    malformed[0] = {**malformed[0], "method_decision_hashes": hashes}
    capability = _RecordingSealCapability()
    with pytest.raises(ProtocolError, match="lowercase SHA-256"):
        record_durable_fold_seals(
            capability,
            replace(products, decisions=tuple(malformed)),
        )
    assert capability.method_seals == []
    assert capability.preevaluation_seal is None
