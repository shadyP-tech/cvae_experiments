"""Public facade for persisted pseudo replay/calibration validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ...protocol import ProtocolError
from .hashing import canonical_hash
from .validation_replay_donors import (
    validate_donor_case_replays,
    validate_main_calibrations,
)
from .validation_replay_policies import validate_policy_replays
from .validation_replay_shared import (
    CaseKey,
    Row,
    index_pseudo_candidates,
    validate_case_sample_counts,
)
from .posterior_expected_utility import FavorableUtility


def validate_persisted_replays_and_calibrations(
    *,
    pseudo_candidate_rows: Sequence[Row],
    donor_case_replay_rows: Sequence[Row],
    policy_replay_rows: Sequence[Row],
    donor_bias_calibration_rows: Sequence[Row],
    case_sample_count_by_center_case: Mapping[CaseKey, int],
    selected_candidate_utilities: Mapping[RouteKey, FavorableUtility],
) -> dict[str, object]:
    """Validate persisted replay products against the sealed pseudo routes.

    The case-count mapping must be derived from the already validated outer
    plans.  This validator accepts no raw labels and reconstructs none.
    """

    try:
        pseudo_index, cases_by_center = index_pseudo_candidates(
            tuple(pseudo_candidate_rows)
        )
        sample_counts = validate_case_sample_counts(
            case_sample_count_by_center_case, cases_by_center
        )
        donor_results, donor_index = validate_donor_case_replays(
            tuple(donor_case_replay_rows),
            pseudo_index,
            sample_counts,
            selected_candidate_utilities,
        )
        calibration_hashes, calibration_map = validate_main_calibrations(
            tuple(donor_bias_calibration_rows), donor_results
        )
        policy_runtime_hashes, policy_replay_hashes = validate_policy_replays(
            tuple(policy_replay_rows),
            pseudo_index,
            donor_results,
            donor_index,
        )
    except ProtocolError:
        raise
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise ProtocolError(
            "CBPUPR persisted replay/calibration payload is malformed."
        ) from exc

    result_hashes = tuple(sorted(row.result_hash for row in donor_results))
    payload = {
        "schema_version": "fixed_bank_cbpupr_persisted_replay_validation_v1",
        "status": "PASS",
        "pseudo_candidate_route_control_count": len(pseudo_index),
        "selected_pseudo_candidate_count": len(donor_results),
        "donor_case_replay_count": len(donor_results),
        "donor_bias_calibration_count": len(calibration_hashes),
        "policy_replay_count": len(policy_runtime_hashes),
        "donor_result_hash_set_hash": canonical_hash(list(result_hashes)),
        "donor_bias_calibration_hash_set_hash": canonical_hash(
            list(calibration_hashes)
        ),
        "policy_replay_hash_set_hash": canonical_hash(
            list(policy_replay_hashes)
        ),
        "policy_runtime_hash_set_hash": canonical_hash(
            list(policy_runtime_hashes)
        ),
        "exact_selected_H_J_d_control_topology": True,
        "exact_H_J_source_exclusion": True,
        "whole_case_label_count_bound_to_outer_plan": True,
        "candidate_utility_bias_applied_once": True,
        "policy_replay_bias_used": False,
        "duplicate_rows_or_hashes_present": False,
        "raw_labels_reconstructed": False,
        "formal_claim_authorized": False,
    }
    return {
        **payload,
        "validation_hash": canonical_hash(payload),
        "calibrations_by_outer_control": calibration_map,
    }


__all__ = ("validate_persisted_replays_and_calibrations",)
