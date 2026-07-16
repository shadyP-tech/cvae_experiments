from __future__ import annotations

from dataclasses import replace

import pytest

from midogpp_thesis.cvae.preservation.source_inner_studies.contracts import (
    FisherStudyMetricV2,
    PriorStudyMetricV2,
)
from midogpp_thesis.cvae.preservation.source_inner_studies.fisher_decision import (
    INVALID_DECISION as INVALID_FISHER_DECISION,
    NO_STABLE_FISHER,
    PANEL_STABLE_FISHER_ALPHA,
    select_fisher_study_decision,
)
from midogpp_thesis.cvae.preservation.source_inner_studies.prior_decision import (
    E_VS_C_UNAVAILABLE,
    INVALID_DECISION as INVALID_PRIOR_DECISION,
    NO_STABLE_E,
    PANEL_STABLE_E_OVER_C,
    PANEL_STABLE_E_VS_A,
    select_prior_study_decision,
)


OUTER = "0"
INNERS = ("1", "2", "3", "5", "6", "7", "8", "9")
TRAINING_SEEDS = (17, 42, 101)
GENERATION_SEEDS = (17, 42, 101)


def test_prior_decision_averages_generation_seeds_then_emits_separate_consensus() -> None:
    decision = _prior_decision(
        _prior_metrics(e_delta={17: 0.06, 42: 0.06, 101: 0.06}, c_delta=0.04)
    )

    assert decision.status == PANEL_STABLE_E_OVER_C
    assert decision.e_vs_a_consensus_status == PANEL_STABLE_E_VS_A
    assert decision.e_vs_c_consensus_status == PANEL_STABLE_E_OVER_C
    seed = decision.per_training_seed["17"]
    assert seed["e_vs_a"]["mean_preservation_ratio_delta"] == pytest.approx(0.06)
    assert seed["e_vs_a"]["strict_inner_wins"] == 8
    assert seed["e_vs_c_diag"]["mean_preservation_ratio_delta"] == pytest.approx(0.02)
    payload = decision.to_payload()
    assert payload["may_feed_model_recipe"] is False
    assert payload["may_feed_deployable_selection"] is False
    assert "recipe_lock" not in payload
    assert "publication_state" not in payload
    assert "recipe_export_ready" not in payload


def test_prior_e_vs_c_strict_boundary_does_not_erase_e_vs_a_result() -> None:
    decision = _prior_decision(
        _prior_metrics(e_delta={17: 0.06, 42: 0.06, 101: 0.06}, c_delta=0.05)
    )

    assert decision.status == PANEL_STABLE_E_VS_A
    assert decision.e_vs_a_consensus_status == PANEL_STABLE_E_VS_A
    assert decision.e_vs_c_consensus_status == "NO_STABLE_E_OVER_C"


def test_prior_training_seed_disagreement_is_complete_negative() -> None:
    decision = _prior_decision(
        _prior_metrics(e_delta={17: 0.06, 42: 0.04, 101: 0.06}, c_delta=0.02)
    )
    assert decision.status == NO_STABLE_E
    assert decision.reason.startswith("complete_valid_panel")


def test_prior_e_vs_c_comparison_is_recorded_independently_of_e_vs_a_gate() -> None:
    decision = _prior_decision(
        _prior_metrics(e_delta={17: 0.04, 42: 0.04, 101: 0.04}, c_delta=0.02)
    )
    assert decision.status == NO_STABLE_E
    assert decision.e_vs_a_consensus_status == "NO_STABLE_E_VS_A"
    assert decision.e_vs_c_consensus_status == PANEL_STABLE_E_OVER_C


def test_prior_mechanism_ineligibility_is_negative_not_invalid() -> None:
    rows = _prior_metrics(
        e_delta={17: 0.06, 42: 0.06, 101: 0.06},
        c_delta=0.02,
    )
    rows = [
        replace(
            row,
            eligible=False,
            ineligibility_reason="zero_scale_normalized_active_units",
        )
        if row.arm == "E" and row.training_seed == 42
        else row
        for row in rows
    ]
    decision = _prior_decision(rows)

    assert decision.status == NO_STABLE_E
    assert decision.status != INVALID_PRIOR_DECISION
    summary = decision.per_training_seed["42"]["e_vs_a"]
    assert summary["e_mechanism_eligible"] is False
    assert summary["e_ineligibility_reasons"] == [
        "zero_scale_normalized_active_units"
    ]


@pytest.mark.parametrize("corruption", ["missing", "invalid", "nonfinite"])
def test_prior_unavailable_c_diag_disables_only_secondary_comparison(
    corruption: str,
) -> None:
    rows = _prior_metrics(
        e_delta={17: 0.06, 42: 0.06, 101: 0.06},
        c_delta=0.02,
    )
    target_index = next(
        index
        for index, row in enumerate(rows)
        if row.arm == "C-diag" and row.training_seed == 42
    )
    if corruption == "missing":
        rows.pop(target_index)
    elif corruption == "invalid":
        rows[target_index] = replace(rows[target_index], valid=False)
    else:
        rows[target_index] = replace(
            rows[target_index], preservation_ratio=float("nan")
        )

    decision = _prior_decision(rows)

    assert decision.status == PANEL_STABLE_E_VS_A
    assert decision.e_vs_a_consensus_status == PANEL_STABLE_E_VS_A
    assert decision.e_vs_c_consensus_status == E_VS_C_UNAVAILABLE
    assert decision.per_training_seed["17"]["e_vs_c_diag"]["available"] is True
    unavailable = decision.per_training_seed["42"]["e_vs_c_diag"]
    assert unavailable["available"] is False
    assert unavailable["pass"] is False
    assert unavailable["unavailability_reason"]


@pytest.mark.parametrize("arm", ["A", "E"])
@pytest.mark.parametrize("corruption", ["invalid", "nonfinite"])
def test_prior_corrupt_a_or_e_cell_invalidates_decision(
    arm: str,
    corruption: str,
) -> None:
    rows = _prior_metrics(
        e_delta={17: 0.06, 42: 0.06, 101: 0.06},
        c_delta=0.02,
    )
    target_index = next(index for index, row in enumerate(rows) if row.arm == arm)
    rows[target_index] = (
        replace(rows[target_index], valid=False)
        if corruption == "invalid"
        else replace(rows[target_index], preservation_ratio=float("nan"))
    )

    decision = _prior_decision(rows)

    assert decision.status == INVALID_PRIOR_DECISION
    assert decision.e_vs_a_consensus_status == INVALID_PRIOR_DECISION
    assert decision.e_vs_c_consensus_status == INVALID_PRIOR_DECISION


def test_prior_missing_e_cell_invalidates_decision() -> None:
    rows = _prior_metrics(
        e_delta={17: 0.06, 42: 0.06, 101: 0.06},
        c_delta=0.02,
    )
    rows.pop(next(index for index, row in enumerate(rows) if row.arm == "E"))
    assert _prior_decision(rows).status == INVALID_PRIOR_DECISION


def test_fisher_selects_smallest_qualifier_within_inclusive_tie_margin() -> None:
    rows = _fisher_metrics(
        improvements={
            17: {0.05: 0.03, 0.10: 0.05, 0.25: 0.06},
            42: {0.05: 0.03, 0.10: 0.05, 0.25: 0.06},
            101: {0.05: 0.03, 0.10: 0.05, 0.25: 0.06},
        }
    )
    decision = _fisher_decision(rows)

    assert decision.status == PANEL_STABLE_FISHER_ALPHA
    assert decision.selected_alpha == pytest.approx(0.10)
    assert all(
        summary["selected_alpha"] == pytest.approx(0.10)
        for summary in decision.per_training_seed.values()
    )


def test_fisher_requires_strict_gain_and_exact_training_seed_consensus() -> None:
    exact_boundary = _fisher_metrics(
        improvements={
            seed: {0.05: 0.01, 0.10: 0.01, 0.25: 0.01}
            for seed in TRAINING_SEEDS
        }
    )
    assert _fisher_decision(exact_boundary).status == NO_STABLE_FISHER

    disagreement = _fisher_metrics(
        improvements={
            17: {0.05: 0.03, 0.10: 0.05, 0.25: 0.06},
            42: {0.05: 0.03, 0.10: 0.05, 0.25: 0.09},
            101: {0.05: 0.03, 0.10: 0.05, 0.25: 0.06},
        }
    )
    decision = _fisher_decision(disagreement)
    assert decision.status == NO_STABLE_FISHER
    assert decision.selected_alpha is None


def test_fisher_safety_and_integrity_fail_closed() -> None:
    rows = _fisher_metrics(
        improvements={
            seed: {0.05: 0.03, 0.10: 0.05, 0.25: 0.06}
            for seed in TRAINING_SEEDS
        },
        decode_delta_by_alpha={0.05: 0.0, 0.10: -0.02, 0.25: -0.02},
    )
    decision = _fisher_decision(rows)
    assert decision.status == PANEL_STABLE_FISHER_ALPHA
    assert decision.selected_alpha == pytest.approx(0.05)

    invalid = [replace(rows[0], valid=False), *rows[1:]]
    assert _fisher_decision(invalid).status == INVALID_FISHER_DECISION


def _prior_decision(rows: list[PriorStudyMetricV2]):
    return select_prior_study_decision(
        rows,
        outer_target_center=OUTER,
        expected_inner_centers=INNERS,
        protocol_hash="protocol",
        decision_contract_hash="decision-contract",
    )


def _fisher_decision(rows: list[FisherStudyMetricV2]):
    return select_fisher_study_decision(
        rows,
        outer_target_center=OUTER,
        expected_inner_centers=INNERS,
        protocol_hash="protocol",
        decision_contract_hash="decision-contract",
    )


def _prior_metrics(
    *,
    e_delta: dict[int, float],
    c_delta: float,
) -> list[PriorStudyMetricV2]:
    rows: list[PriorStudyMetricV2] = []
    generation_offsets = {17: -0.03, 42: 0.0, 101: 0.03}
    for training_seed in TRAINING_SEEDS:
        for inner in INNERS:
            for generation_seed in GENERATION_SEEDS:
                baseline = 0.50 + generation_offsets[generation_seed]
                for arm, delta in (
                    ("A", 0.0),
                    ("C-diag", c_delta),
                    ("E", e_delta[training_seed]),
                ):
                    rows.append(
                        PriorStudyMetricV2(
                            outer_target_center=OUTER,
                            inner_pseudo_target_center=inner,
                            training_seed=training_seed,
                            generation_seed=generation_seed,
                            arm=arm,
                            preservation_ratio=baseline + delta,
                            decode_bacc=0.70,
                            posterior_bacc=0.69,
                        )
                    )
    return rows


def _fisher_metrics(
    *,
    improvements: dict[int, dict[float, float]],
    decode_delta_by_alpha: dict[float, float] | None = None,
) -> list[FisherStudyMetricV2]:
    rows: list[FisherStudyMetricV2] = []
    generation_offsets = {17: -0.03, 42: 0.0, 101: 0.03}
    decode_deltas = decode_delta_by_alpha or {}
    for training_seed in TRAINING_SEEDS:
        for inner in INNERS:
            for generation_seed in GENERATION_SEEDS:
                baseline = 0.50 + generation_offsets[generation_seed]
                for alpha in (0.0, 0.05, 0.10, 0.25):
                    delta = 0.0 if alpha == 0.0 else improvements[training_seed][alpha]
                    rows.append(
                        FisherStudyMetricV2(
                            outer_target_center=OUTER,
                            inner_pseudo_target_center=inner,
                            training_seed=training_seed,
                            generation_seed=generation_seed,
                            alpha=alpha,
                            preservation_ratio=baseline + delta,
                            decode_bacc=0.70 + decode_deltas.get(alpha, 0.0),
                            posterior_bacc=0.69,
                        )
                    )
    return rows
