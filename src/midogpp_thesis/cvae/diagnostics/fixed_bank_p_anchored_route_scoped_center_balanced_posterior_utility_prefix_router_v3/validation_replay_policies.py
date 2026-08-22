"""Persisted leave-J policy replay and compact-prefix validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .constants import (
    CENTERS,
    PRIMARY_FINGERPRINT_CONTROL_ID,
    UTILITY_ZERO_TOLERANCE,
)
from .donor_replay_runtime import DonorReplayResult
from .hashing import canonical_hash
from .policy_calibration import PolicyReplay
from .policy_prefixes import PrefixEvaluation
from .posterior_expected_utility import FavorableUtility
from .utility_calibration import (
    CenterBalancedUtilityCalibration,
    build_center_balanced_utility_calibration,
)
from .validation_replay_shared import (
    RouteKey,
    Row,
    fail,
    list_value,
    mapping_value,
    require_mapping,
)


def validate_policy_replays(
    rows: Sequence[Row],
    pseudos: Mapping[RouteKey, Row],
    donor_results: Sequence[DonorReplayResult],
    donor_index: Mapping[RouteKey, DonorReplayResult],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    replay_rows = tuple(result.replay for result in donor_results)
    expected_pairs: list[tuple[str, str]] = []
    leave_calibrations: dict[
        tuple[str, str], CenterBalancedUtilityCalibration
    ] = {}
    for outer in CENTERS:
        for donor in CENTERS:
            if donor == outer:
                continue
            leave_rows = tuple(
                row
                for row in replay_rows
                if row.outer_center == outer
                and row.control_id == PRIMARY_FINGERPRINT_CONTROL_ID
                and row.donor_center != donor
            )
            supported = {row.donor_center for row in leave_rows}
            if len(supported) >= 6:
                expected_pairs.append((outer, donor))
                leave_calibrations[(outer, donor)] = (
                    build_center_balanced_utility_calibration(
                        leave_rows,
                        outer_center=outer,
                        calibration_excluded_centers=(outer, donor),
                    )
                )

    indexed: dict[tuple[str, str], Row] = {}
    observed_order: list[tuple[str, str]] = []
    for payload in rows:
        require_mapping(payload, "policy replay")
        replay = PolicyReplay.from_payload(mapping_value(payload, "replay"))
        key = (replay.outer_center, replay.donor_center)
        if (
            payload.get("record_type") != "policy_replay"
            or key in indexed
        ):
            fail("policy replay record topology")
        indexed[key] = payload
        observed_order.append(key)
    if observed_order != expected_pairs or set(indexed) != set(expected_pairs):
        fail("supported leave-J policy replay rectangle")

    runtime_hashes: list[str] = []
    replay_hashes: list[str] = []
    for key in expected_pairs:
        runtime_hash, replay_hash = _validate_policy_row(
            indexed[key],
            outer=key[0],
            donor=key[1],
            pseudos=pseudos,
            donor_index=donor_index,
            calibration=leave_calibrations[key],
        )
        runtime_hashes.append(runtime_hash)
        replay_hashes.append(replay_hash)
    if (
        len(runtime_hashes) != len(set(runtime_hashes))
        or len(replay_hashes) != len(set(replay_hashes))
    ):
        fail("duplicate policy replay/runtime hashes")
    return tuple(sorted(runtime_hashes)), tuple(sorted(replay_hashes))


def _validate_policy_row(
    payload: Row,
    *,
    outer: str,
    donor: str,
    pseudos: Mapping[RouteKey, Row],
    donor_index: Mapping[RouteKey, DonorReplayResult],
    calibration: CenterBalancedUtilityCalibration,
) -> tuple[str, str]:
    control = PRIMARY_FINGERPRINT_CONTROL_ID
    route_rows = tuple(
        (key, row)
        for key, row in pseudos.items()
        if key[0] == outer and key[1] == donor and key[3] == control
    )
    candidate_runtime_hashes = tuple(
        sorted(str(row.get("runtime_hash")) for _key, row in route_rows)
    )
    ranked = []
    for key, row in route_rows:
        selected = row.get("selected_candidate_hash")
        if selected is None:
            continue
        result = donor_index[key]
        corrected = calibration.correct(result.replay.predicted_utility)
        policy_hash = canonical_hash(
            {
                "schema_version": "cbpupr_prefix_candidate_v1",
                "center": donor,
                "case_id": key[2],
                "action_hash": str(selected),
                "corrected_utility": corrected.to_payload(),
                "calibration_hash": calibration.calibration_hash,
            }
        )
        ranked.append(
            {
                "center": donor,
                "case_id": key[2],
                "control_id": control,
                "action_hash": str(selected),
                "policy_hash": policy_hash,
                "corrected_utility": corrected.to_payload(),
                "calibration_hash": calibration.calibration_hash,
            }
        )
    ranked.sort(
        key=lambda row: (
            -float(mapping_value(row, "corrected_utility")["bacc_gain"]),
            str(row["case_id"]),
            str(row["policy_hash"]),
        )
    )
    selection = _expected_selection_summary(tuple(ranked))
    selected_hashes = tuple(
        str(value) for value in selection["selected_candidate_hashes"]
    )
    selected_cases = tuple(str(value) for value in selection["selected_case_ids"])
    selected_k = int(selection["selected_k"])
    predicted = FavorableUtility.from_payload(
        mapping_value(
            list_value(selection, "evaluations")[selected_k],
            "aggregate_utility",
        )
    )
    realized = FavorableUtility.zeros()
    for case, candidate_hash in zip(selected_cases, selected_hashes, strict=True):
        result = donor_index[(outer, donor, case, control)]
        if result.replay.candidate_hash != candidate_hash:
            fail("policy selected candidate/donor replay lineage")
        realized = realized + result.replay.realized_utility
    replay = PolicyReplay(
        outer,
        donor,
        selected_k,
        selected_hashes,
        predicted,
        realized,
        tuple(sorted((outer, donor))),
        str(selection["selection_hash"]),
    )
    runtime_hash = canonical_hash(
        {
            "schema_version": "cbpupr_policy_replay_runtime_v1",
            "selection_hash": selection["selection_hash"],
            "replay_hash": replay.replay_hash,
            "candidate_calibration_hash": calibration.calibration_hash,
            "candidate_runtime_hashes": list(candidate_runtime_hashes),
        }
    )
    expected_payload = {
        "record_type": "policy_replay",
        "selection": selection,
        "replay": replay.to_payload(),
        "candidate_calibration_hash": calibration.calibration_hash,
        "candidate_runtime_hashes": list(candidate_runtime_hashes),
        "runtime_hash": runtime_hash,
        "dense_probabilities_persisted": False,
    }
    reconstructed_replay = PolicyReplay.from_payload(
        mapping_value(payload, "replay")
    )
    if (
        dict(payload) != expected_payload
        or reconstructed_replay.replay_hash != replay.replay_hash
        or payload.get("runtime_hash") != runtime_hash
    ):
        fail("policy replay selection, one-bias, or runtime hash")
    return runtime_hash, replay.replay_hash


def _expected_selection_summary(
    ranked: tuple[dict[str, object], ...]
) -> dict[str, object]:
    evaluations = [
        PrefixEvaluation(
            0,
            (),
            FavorableUtility.zeros(),
            True,
            ("EXACT_P_BASELINE",),
        )
    ]
    aggregate = FavorableUtility.zeros()
    candidate_hashes: list[str] = []
    for index, row in enumerate(ranked, start=1):
        aggregate = aggregate + FavorableUtility.from_payload(
            mapping_value(row, "corrected_utility")
        )
        candidate_hashes.append(str(row["action_hash"]))
        reasons = []
        if aggregate.bacc_gain <= UTILITY_ZERO_TOLERANCE:
            reasons.append("NONPOSITIVE_AGGREGATE_BACC")
        if aggregate.brier_gain < -UTILITY_ZERO_TOLERANCE:
            reasons.append("NEGATIVE_AGGREGATE_BRIER_GAIN")
        if aggregate.log_gain < -UTILITY_ZERO_TOLERANCE:
            reasons.append("NEGATIVE_AGGREGATE_LOG_GAIN")
        evaluations.append(
            PrefixEvaluation(
                index,
                tuple(candidate_hashes),
                aggregate,
                not reasons,
                ("AGGREGATE_UTILITY_PASS",) if not reasons else tuple(reasons),
            )
        )
    feasible = tuple(row for row in evaluations if row.feasible)
    maximum = max(row.aggregate_utility.bacc_gain for row in feasible)
    tied = tuple(
        row
        for row in feasible
        if abs(row.aggregate_utility.bacc_gain - maximum)
        <= UTILITY_ZERO_TOLERANCE
    )
    selected = min(tied, key=lambda row: (row.k, row.prefix_hash))
    selection_hash = canonical_hash(
        {
            "schema_version": "cbpupr_prefix_selection_v1",
            "ranked_policy_hashes": [row["policy_hash"] for row in ranked],
            "evaluation_hashes": [row.prefix_hash for row in evaluations],
            "selected_k": selected.k,
            "selected_prefix_hash": selected.prefix_hash,
        }
    )
    return {
        "ranked_candidates": list(ranked),
        "evaluations": [row.to_payload() for row in evaluations],
        "selected_k": selected.k,
        "selected_prefix_hash": selected.prefix_hash,
        "selected_case_ids": [
            str(row["case_id"]) for row in ranked[: selected.k]
        ],
        "selected_candidate_hashes": [
            str(row["action_hash"]) for row in ranked[: selected.k]
        ],
        "selection_hash": selection_hash,
        "dense_probabilities_persisted": False,
    }


__all__ = ("validate_policy_replays",)
