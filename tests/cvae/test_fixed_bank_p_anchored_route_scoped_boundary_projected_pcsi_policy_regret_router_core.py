from __future__ import annotations

from collections import OrderedDict
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_boundary_projected_pcsi_policy_regret_router.calibration import (
    build_route_calibration,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_boundary_projected_pcsi_policy_regret_router.case_regret import (
    PseudoCaseReplay,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_boundary_projected_pcsi_policy_regret_router.constants import (
    CENTERS,
    EXPECTED_FINAL_CASE_PREDICTION_COUNT,
    EXPECTED_POLICY_REPLAY_COUNT,
    EXPECTED_UTILITY_MODEL_FIT_COUNT,
    PORTFOLIO_METHOD_ID,
    PRIMARY_METHOD_ID,
    PROJECTION_GEOMETRY_ID,
    PROJECTED_NO_ENVELOPE_METHOD_ID,
    RAW_OBSERVED_MAX_METHOD_ID,
    UNPROJECTED_GEOMETRY_ID,
    physical_action_ids,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_boundary_projected_pcsi_policy_regret_router.contracts import (
    CenterProbabilitySurface,
    PhysicalProbabilitySurface,
    PseudoRouteKey,
    TargetRouteKey,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_boundary_projected_pcsi_policy_regret_router.engine import (
    build_preterminal_result,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_boundary_projected_pcsi_policy_regret_router.decision import (
    ABSTAIN_TO_P,
    REASON_MARGIN,
    make_route_decision,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_boundary_projected_pcsi_policy_regret_router.evaluation import (
    evaluate_terminal,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_boundary_projected_pcsi_policy_regret_router.hashing import (
    canonical_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_boundary_projected_pcsi_policy_regret_router.persistence import (
    persist_preterminal,
    persist_terminal,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_boundary_projected_pcsi_policy_regret_router.reports import (
    leakage_report_payload,
    publication_decision_payload,
)


def _small_surface() -> tuple[
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
                    np.clip(
                        base + np.float32((seed - 4) * 0.003),
                        0.0,
                        1.0,
                    )
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
    return (
        PhysicalProbabilitySurface(
            centers,
            "a" * 64,
            strict_canonical_topology=False,
        ),
        truth,
    )


def _loader(
    truth: dict[tuple[str, str, str], int],
):
    def load(
        allowed: frozenset[tuple[str, str, str]], role: str
    ) -> tuple[object, ...]:
        return tuple(
            type(
                "LabelRow",
                (),
                {
                    "center": center,
                    "case_id": case_id,
                    "sample_id": sample_id,
                    "value": truth[(center, case_id, sample_id)],
                },
            )()
            for center, case_id, sample_id in sorted(allowed)
        )

    return load


def test_approved_workload_counts_are_frozen() -> None:
    assert EXPECTED_UTILITY_MODEL_FIT_COUNT == 1_314
    assert EXPECTED_POLICY_REPLAY_COUNT == 3_488
    assert EXPECTED_FINAL_CASE_PREDICTION_COUNT == 872


def test_observed_envelope_is_complete_componentwise_max() -> None:
    case_ids = {center: (f"{center}-a", f"{center}-b") for center in CENTERS}
    rows = []
    for donor in CENTERS[1:]:
        for index, case_id in enumerate(case_ids[donor]):
            predicted = (0.2 + index, 0.3 + index, 0.4 + index)
            realized = (0.1, 0.1, 0.1)
            residual = tuple(predicted[k] - realized[k] for k in range(3))
            rows.append(
                PseudoCaseReplay(
                    PROJECTION_GEOMETRY_ID,
                    PseudoRouteKey("0", donor, case_id),
                    "a" * 64,
                    "b" * 64,
                    predicted,
                    realized,
                    residual,
                    canonical_hash([donor, case_id]),
                )
            )
    calibration = build_route_calibration(
        rows,
        geometry_id=PROJECTION_GEOMETRY_ID,
        outer_center="0",
        expected_case_ids_by_center=case_ids,
    )
    assert calibration.valid
    assert calibration.margin == pytest.approx((1.1, 1.2, 1.3))
    assert all(
        calibration.margin[k] >= replay.overprediction_residual[k]
        for replay in rows
        for k in range(3)
    )


def test_equality_in_any_corrected_coordinate_abstains_to_exact_p() -> None:
    candidate = SimpleNamespace(
        target_center="0",
        case_id="0-c0",
        geometry_id=PROJECTION_GEOMETRY_ID,
        predicted_favorable_endpoint_vector=(1.0, 2.0, 3.0),
        changed=True,
        policy_hash="a" * 64,
    )
    screen = SimpleNamespace(
        outer_center="0",
        candidate_center="0",
        candidate_case_id="0-c0",
        passed=True,
        screen_hash="b" * 64,
    )
    calibration = SimpleNamespace(
        outer_center="0",
        geometry_id=PROJECTION_GEOMETRY_ID,
        margin=(1.0, 0.5, 0.5),
        valid=True,
        calibration_hash="c" * 64,
    )
    decision = make_route_decision(
        candidate, screen, calibration, policy_id=PRIMARY_METHOD_ID
    )
    assert decision.outcome == ABSTAIN_TO_P
    assert decision.reason_code == REASON_MARGIN
    assert decision.corrected_vector == (0.0, 1.5, 2.5)


def test_small_end_to_end_pipeline_is_route_scoped(tmp_path) -> None:
    surface, truth = _small_surface()
    preterminal = build_preterminal_result(
        surface,
        _loader(truth),
        use_processes=False,
    )
    runtime = preterminal.policy_runtime
    assert preterminal.donor_runtime.model_fit_count == 1_314
    assert len(runtime.transport_descriptors_by_outer_candidate) == 1_971
    assert runtime.transport_seal.numeric_leaf_count == 1_755
    assert len(runtime.transport_reference_blocks) == 576
    assert len(runtime.transport_screens) == 243
    assert len(runtime.replays) == 432
    assert len(runtime.calibrations) == 18
    assert len(runtime.decisions) == 81
    assert all(
        screen.audit_only
        for key, screen in runtime.transport_screens.items()
        if isinstance(key, PseudoRouteKey)
    )
    report = preterminal.label_firewall.report_payload()
    assert report["terminal_opened"] is False
    assert report["own_route_noninterference_required"] is True
    assert preterminal.aggregate_seal["terminal_labels_used"] is False
    terminal = evaluate_terminal(preterminal)
    assert terminal.capability_report["status"] == "PASS"
    assert len(terminal.method_metrics) == 7
    persist_preterminal(tmp_path, preterminal)
    leakage = leakage_report_payload(
        probability_surface_hash=surface.surface_hash,
        preterminal=preterminal,
        capability_report=terminal.capability_report,
    )
    persist_terminal(
        tmp_path,
        terminal=terminal,
        leakage_report=leakage,
        publication_decision=publication_decision_payload(terminal),
        runtime_summary={"status": "SYNTHETIC_TEST"},
    )
    assert (tmp_path / "tables/route_calibrations.json").is_file()
    assert (tmp_path / "tables/transport_reference_blocks.json").is_file()
    assert (tmp_path / "reports/leakage_report.json").is_file()


def test_own_target_and_pseudo_route_preevaluation_hashes_ignore_own_label() -> None:
    surface, truth = _small_surface()
    clean = build_preterminal_result(surface, _loader(truth), use_processes=False)

    target_truth = dict(truth)
    target_label = ("0", "0-c0", "0-c0-y0")
    target_truth[target_label] = 1 - target_truth[target_label]
    target_poison = build_preterminal_result(
        surface, _loader(target_truth), use_processes=False
    )
    route = TargetRouteKey("0", "0-c0")
    assert (
        clean.policy_runtime.transport_descriptors_by_outer_candidate[route].descriptor_hash
        == target_poison.policy_runtime.transport_descriptors_by_outer_candidate[
            route
        ].descriptor_hash
    )
    assert (
        clean.policy_runtime.transport_screens[route].screen_hash
        == target_poison.policy_runtime.transport_screens[route].screen_hash
    )
    for policy in (
        PRIMARY_METHOD_ID,
        RAW_OBSERVED_MAX_METHOD_ID,
        PROJECTED_NO_ENVELOPE_METHOD_ID,
    ):
        key = policy, "0", "0-c0"
        assert (
            clean.policy_runtime.target_candidate_policies[key].policy_hash
            == target_poison.policy_runtime.target_candidate_policies[key].policy_hash
        )
        assert (
            clean.policy_runtime.decisions[key].decision_hash
            == target_poison.policy_runtime.decisions[key].decision_hash
        )

    pseudo_truth = dict(truth)
    pseudo_label = ("1", "1-c0", "1-c0-y0")
    pseudo_truth[pseudo_label] = 1 - pseudo_truth[pseudo_label]
    pseudo_poison = build_preterminal_result(
        surface, _loader(pseudo_truth), use_processes=False
    )
    pseudo_route = PseudoRouteKey("0", "1", "1-c0")
    assert (
        clean.policy_runtime.transport_descriptors_by_outer_candidate[
            pseudo_route
        ].descriptor_hash
        == pseudo_poison.policy_runtime.transport_descriptors_by_outer_candidate[
            pseudo_route
        ].descriptor_hash
    )
    assert (
        clean.policy_runtime.transport_screens[pseudo_route].screen_hash
        == pseudo_poison.policy_runtime.transport_screens[pseudo_route].screen_hash
    )
    for geometry in (PROJECTION_GEOMETRY_ID, UNPROJECTED_GEOMETRY_ID):
        key = geometry, "0", "1", "1-c0"
        assert (
            clean.policy_runtime.pseudo_candidate_policies[key].policy_hash
            == pseudo_poison.policy_runtime.pseudo_candidate_policies[key].policy_hash
        )
        assert clean.policy_runtime.replays[key].replay_hash != (
            pseudo_poison.policy_runtime.replays[key].replay_hash
        )
