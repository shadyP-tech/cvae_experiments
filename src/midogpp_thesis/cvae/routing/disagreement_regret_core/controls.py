"""Matched G/R/P label-free feature controls."""

from __future__ import annotations

from collections import defaultdict

from ...protocol import ProtocolError
from .contracts import CaseActionFeatureRow, DisagreementFeatureSurface


def assert_control_surface_matches_parent(
    surface: DisagreementFeatureSurface,
    parent: DisagreementFeatureSurface,
) -> None:
    """Replay G/P feature construction against the exact aligned R parent."""

    if (
        not isinstance(surface, DisagreementFeatureSurface)
        or not isinstance(parent, DisagreementFeatureSurface)
        or parent.family != "R"
        or parent.parent_surface_hash is not None
        or surface.family not in ("G", "P")
        or surface.parent_surface_hash != parent.surface_hash
        or surface.prediction_seal_hash != parent.prediction_seal_hash
        or surface.development_context_hash != parent.development_context_hash
        or surface.sample_keys != parent.sample_keys
        or surface.baseline_action_id != parent.baseline_action_id
        or surface.control_action_id != parent.control_action_id
        or surface.candidate_source_by_action != parent.candidate_source_by_action
        or surface.surface_role != parent.surface_role
    ):
        raise ProtocolError("G/P control lineage drifted from its aligned R parent.")
    parent_by_key = {row.row_key: row for row in parent.rows}
    if {row.row_key for row in surface.rows} != set(parent_by_key):
        raise ProtocolError("G/P control rows drifted from their aligned R parent.")
    for row in surface.rows:
        destination = parent_by_key[row.row_key]
        if (
            row.source_id != destination.source_id
            or row.sample_count != destination.sample_count
        ):
            raise ProtocolError("G/P destination identity drifted from aligned R.")
        if surface.family == "G":
            if (
                row.feature_origin_action_id != row.action_id
                or row.disagreement_count != 0
                or any(value != 0.0 for value in row.values)
            ):
                raise ProtocolError("G control replay failed exact-zero validation.")
            continue
        donor_key = (row.query_id, row.case_id, row.feature_origin_action_id)
        donor = parent_by_key.get(donor_key)
        if donor is None or (
            row.values != donor.values
            or row.disagreement_count != donor.disagreement_count
        ):
            raise ProtocolError("P control replay failed parent-row attestation.")


def feature_surface_for_family(
    surface: DisagreementFeatureSurface,
    *,
    family: str,
) -> DisagreementFeatureSurface:
    """Return global-only G, aligned R, or blocked-refit P features.

    Responses are intentionally absent.  P rotates the complete feature vector
    within each query/case candidate block, while B remains fixed.  The caller
    must refit P; model reuse across R/P is not supported.
    """

    if not isinstance(surface, DisagreementFeatureSurface):
        raise ProtocolError("G/R/P controls require a typed feature surface.")
    if surface.family != "R" or surface.parent_surface_hash is not None:
        raise ProtocolError("G/P controls must derive directly from aligned R features.")
    if family == "R":
        return surface
    if family not in ("G", "P"):
        raise ProtocolError("Feature family must be G, R, or P.")

    if family == "G":
        rows = tuple(
            CaseActionFeatureRow(
                query_id=row.query_id,
                case_id=row.case_id,
                action_id=row.action_id,
                source_id=row.source_id,
                values=(0.0,) * len(row.values),
                sample_count=row.sample_count,
                disagreement_count=0,
                prediction_seal_hash=row.prediction_seal_hash,
                feature_origin_action_id=row.action_id,
            )
            for row in surface.rows
        )
    else:
        grouped: dict[tuple[str, str], list[CaseActionFeatureRow]] = defaultdict(list)
        baseline_rows: list[CaseActionFeatureRow] = []
        for row in surface.rows:
            if row.action_id == surface.baseline_action_id:
                baseline_rows.append(row)
            else:
                grouped[(row.query_id, row.case_id)].append(row)
        output = list(baseline_rows)
        for block_key, block in sorted(grouped.items()):
            ordered = tuple(sorted(block, key=lambda row: row.action_id))
            if len(ordered) < 2:
                raise ProtocolError("Blocked P requires at least two candidates per case.")
            donors = ordered[1:] + ordered[:1]
            for destination, donor in zip(ordered, donors, strict=True):
                if donor.action_id == destination.action_id:
                    raise ProtocolError("Blocked P must be a derangement.")
                output.append(
                    CaseActionFeatureRow(
                        query_id=destination.query_id,
                        case_id=destination.case_id,
                        action_id=destination.action_id,
                        source_id=destination.source_id,
                        values=donor.values,
                        sample_count=destination.sample_count,
                        disagreement_count=donor.disagreement_count,
                        prediction_seal_hash=destination.prediction_seal_hash,
                        feature_origin_action_id=donor.action_id,
                    )
                )
        rows = tuple(sorted(output, key=lambda row: row.row_key))
        for block_key, original in grouped.items():
            original_multiset = sorted(row.values for row in original)
            permuted_multiset = sorted(
                row.values
                for row in rows
                if (row.query_id, row.case_id) == block_key
                and row.action_id != surface.baseline_action_id
            )
            if original_multiset != permuted_multiset:
                raise ProtocolError("Blocked P did not preserve the feature multiset.")

    return DisagreementFeatureSurface(
        rows=tuple(sorted(rows, key=lambda row: row.row_key)),
        disagreements=(),
        baseline_action_id=surface.baseline_action_id,
        control_action_id=surface.control_action_id,
        candidate_source_by_action=surface.candidate_source_by_action,
        prediction_seal_hash=surface.prediction_seal_hash,
        sample_keys=surface.sample_keys,
        development_context_hash=surface.development_context_hash,
        dataset_family=surface.dataset_family,
        outer_target_id=surface.outer_target_id,
        surface_role=surface.surface_role,
        family=family,
        parent_surface_hash=surface.surface_hash,
    )


__all__ = ("assert_control_surface_matches_parent", "feature_surface_for_family")
