"""Persisted donor-case replay and main candidate-bias validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .calibration import UnsupportedCalibration
from .constants import CENTERS
from .donor_replay_runtime import DonorReplayResult
from .posterior_contracts import CONTROL_IDS
from .posterior_expected_utility import FavorableUtility
from .utility_calibration import (
    CenterBalancedUtilityCalibration,
    build_center_balanced_utility_calibration,
)
from .validation_replay_shared import (
    CaseKey,
    RouteKey,
    Row,
    fail,
    list_value,
    require_mapping,
)


def validate_donor_case_replays(
    rows: Sequence[Row],
    pseudos: Mapping[RouteKey, Row],
    sample_counts: Mapping[CaseKey, int],
    selected_candidate_utilities: Mapping[RouteKey, FavorableUtility],
) -> tuple[tuple[DonorReplayResult, ...], dict[RouteKey, DonorReplayResult]]:
    expected_keys = {
        key
        for key, row in pseudos.items()
        if row.get("selected_candidate_hash") is not None
    }
    if set(selected_candidate_utilities) != expected_keys:
        fail("selected pseudo candidate utility rectangle")
    indexed: dict[RouteKey, DonorReplayResult] = {}
    replay_hashes: set[str] = set()
    result_hashes: set[str] = set()
    ordered: list[DonorReplayResult] = []
    for payload in rows:
        require_mapping(payload, "donor replay")
        if payload.get("record_type") != "donor_case_replay":
            fail("donor replay record type")
        result = DonorReplayResult.from_payload(payload)
        replay = result.replay
        key = (
            replay.outer_center,
            replay.donor_center,
            replay.case_id,
            replay.control_id,
        )
        if key not in pseudos:
            fail("donor replay route key")
        pseudo = pseudos[key]
        expected_payload = {"record_type": "donor_case_replay", **result.to_payload()}
        if (
            key in indexed
            or result.replay.replay_hash in replay_hashes
            or result.result_hash in result_hashes
            or dict(payload) != expected_payload
            or tuple(replay.lineage_excluded_centers)
            != tuple(sorted((replay.outer_center, replay.donor_center)))
            or replay.candidate_hash
            != str(pseudo.get("selected_candidate_hash"))
            or replay.predicted_utility != selected_candidate_utilities[key]
            or result.endpoint_lineage_hash
            != str(pseudo.get("endpoint_lineage_hash"))
            or result.label_count
            != sample_counts[(replay.donor_center, replay.case_id)]
        ):
            fail("donor replay/result lineage or hash")
        indexed[key] = result
        replay_hashes.add(replay.replay_hash)
        result_hashes.add(result.result_hash)
        ordered.append(result)
    if set(indexed) != expected_keys:
        fail("selected pseudo candidate/donor replay rectangle")
    return tuple(ordered), indexed


def validate_main_calibrations(
    rows: Sequence[Row], donor_results: Sequence[DonorReplayResult]
) -> tuple[
    tuple[str, ...],
    dict[tuple[str, str], dict[str, object]],
]:
    expected: list[
        tuple[
            tuple[str, str],
            CenterBalancedUtilityCalibration | UnsupportedCalibration,
        ]
    ] = []
    replays = tuple(result.replay for result in donor_results)
    for outer in CENTERS:
        for control in CONTROL_IDS:
            calibration_rows = tuple(
                row
                for row in replays
                if row.outer_center == outer and row.control_id == control
            )
            donors = tuple(
                sorted({row.donor_center for row in calibration_rows})
            )
            if len(donors) >= 6:
                expected.append(
                    (
                        (outer, control),
                        build_center_balanced_utility_calibration(
                            calibration_rows,
                            outer_center=outer,
                            calibration_excluded_centers=(outer,),
                        ),
                    )
                )
            else:
                expected.append(
                    (
                        (outer, control),
                        UnsupportedCalibration(
                            "candidate_utility", outer, (outer,), donors
                        ),
                    )
                )
    if len(rows) != len(expected):
        fail("donor-bias calibration rectangle")

    hashes: list[str] = []
    calibration_map: dict[tuple[str, str], dict[str, object]] = {}
    for payload, (key, expected_row) in zip(rows, expected, strict=True):
        require_mapping(payload, "donor-bias calibration")
        if isinstance(expected_row, CenterBalancedUtilityCalibration):
            reconstructed = CenterBalancedUtilityCalibration.from_payload(payload)
        else:
            reconstructed = unsupported_from_payload(payload)
        if (
            dict(payload) != expected_row.to_payload()
            or reconstructed.calibration_hash != expected_row.calibration_hash
        ):
            fail("donor-bias calibration support, bias, or hash")
        hashes.append(expected_row.calibration_hash)
        if isinstance(expected_row, CenterBalancedUtilityCalibration):
            calibration_map[key] = {
                "supported": True,
                "bias": expected_row.bias.to_payload(),
                "calibration_hash": expected_row.calibration_hash,
                "reason_code": None,
                "forces_exact_P": False,
            }
        else:
            calibration_map[key] = {
                "supported": False,
                "bias": None,
                "calibration_hash": expected_row.calibration_hash,
                "reason_code": expected_row.reason_code,
                "forces_exact_P": True,
            }
    if len(hashes) != len(set(hashes)):
        fail("duplicate donor-bias calibration hashes")
    return tuple(sorted(hashes)), calibration_map


def unsupported_from_payload(payload: Row) -> UnsupportedCalibration:
    row = UnsupportedCalibration(
        str(payload["calibration_kind"]),
        str(payload["outer_center"]),
        tuple(str(value) for value in list_value(payload, "excluded_centers")),
        tuple(
            str(value)
            for value in list_value(payload, "supported_donor_centers")
        ),
        str(payload["reason_code"]),
    )
    if dict(payload) != row.to_payload():
        fail("unsupported calibration hash")
    return row


__all__ = (
    "unsupported_from_payload",
    "validate_donor_case_replays",
    "validate_main_calibrations",
)
