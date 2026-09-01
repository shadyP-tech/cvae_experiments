"""Strict outer-H and nested source-center LODO fitting for HARP v4."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
import math
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .calibration import DonorResidualCalibration, calibrate_donor_residuals
from .compatibility import GeometryCalibration, calibrate_geometry
from .contracts import (
    CaseTrainingObservation,
    Comparison,
    EffectVector,
    SupportSummary,
)
from .ridge import SharedDesignRidge, fit_shared_design_ridge


_EFFECT_ATOL = 5e-12
_EFFECT_RTOL = 1e-10


class _RidgeFitMemo:
    """Per-bundle memo for byte-equivalent source-only ridge fits.

    Nested delete-donor and pair-LODO calibration reaches the same training
    surface through several exclusion paths.  The ridge solve is deterministic,
    so repeating a solve whose ordered case identities, penalty, and normalized
    exclusion set are identical only wastes workstation CPU time.  Binding every
    identity to the immutable observation from the admitted outer-H surface
    prevents a matching row key with different contents from aliasing a cached
    model.
    """

    def __init__(self, source_rows: tuple[CaseTrainingObservation, ...]) -> None:
        if not source_rows or any(
            not isinstance(row, CaseTrainingObservation) for row in source_rows
        ):
            raise ProtocolError("HARP v4 ridge memo requires typed source rows.")
        outer_ids = {row.outer_target_id for row in source_rows}
        if len(outer_ids) != 1:
            raise ProtocolError("HARP v4 ridge memo crossed outer-target surfaces.")
        self._outer_target_id = next(iter(outer_ids))
        self._source_rows: dict[
            tuple[str, str, str, str, str], CaseTrainingObservation
        ] = {}
        for row in source_rows:
            if row.row_key in self._source_rows:
                raise ProtocolError("HARP v4 ridge memo row identity is ambiguous.")
            self._source_rows[row.row_key] = row
        self._models: dict[
            tuple[
                tuple[tuple[str, str, str, str, str], ...],
                float,
                tuple[str, ...],
            ],
            SharedDesignRidge,
        ] = {}

    def fit(
        self,
        rows: tuple[CaseTrainingObservation, ...],
        *,
        alpha: float,
        excluded_center_ids: Sequence[str],
    ) -> SharedDesignRidge:
        if not rows:
            raise ProtocolError("HARP v4 ridge memo cannot fit an empty surface.")
        identities: list[tuple[str, str, str, str, str]] = []
        for row in rows:
            if not isinstance(row, CaseTrainingObservation):
                raise ProtocolError("HARP v4 ridge memo received an untyped row.")
            identity = row.row_key
            admitted = self._source_rows.get(identity)
            if admitted is None or admitted != row:
                raise ProtocolError("HARP v4 ridge memo row identity is ambiguous.")
            identities.append(identity)
        ordered_identities = tuple(identities)
        if len(set(ordered_identities)) != len(ordered_identities):
            raise ProtocolError("HARP v4 ridge memo row identity is ambiguous.")
        try:
            penalty = float(alpha)
        except (TypeError, ValueError) as exc:
            raise ProtocolError("HARP v4 ridge memo alpha is invalid.") from exc
        excluded = tuple(sorted({str(value) for value in excluded_center_ids}))
        if (
            not math.isfinite(penalty)
            or penalty <= 0
            or not excluded
            or self._outer_target_id not in excluded
            or any(
                row.pseudo_query_id in excluded
                or row.candidate_source_id in excluded
                for row in rows
            )
        ):
            raise ProtocolError("HARP v4 ridge memo exclusion contract is invalid.")
        key = (ordered_identities, penalty, excluded)
        cached = self._models.get(key)
        if cached is not None:
            return cached
        model = fit_shared_design_ridge(
            rows,
            alpha=penalty,
            excluded_center_ids=excluded,
        )
        if model.alpha != penalty or model.excluded_center_ids != excluded:
            raise ProtocolError("HARP v4 ridge memo fit binding drifted.")
        self._models[key] = model
        return model


@dataclass(frozen=True)
class AlphaFoldScore:
    heldout_donor_id: str
    alpha: float
    standardized_mse: float
    training_query_ids: tuple[str, ...]
    training_candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not self.heldout_donor_id
            or not math.isfinite(float(self.alpha))
            or self.alpha <= 0
            or not math.isfinite(float(self.standardized_mse))
            or self.standardized_mse < 0
            or self.heldout_donor_id in self.training_query_ids
            or self.heldout_donor_id in self.training_candidate_ids
        ):
            raise ProtocolError("HARP v4 alpha-fold audit is malformed.")


@dataclass(frozen=True)
class AlphaSelection:
    selected_alpha: float
    alpha_grid: tuple[float, ...]
    fold_scores: tuple[AlphaFoldScore, ...]

    def __post_init__(self) -> None:
        if (
            self.alpha_grid != tuple(sorted(set(self.alpha_grid)))
            or not self.alpha_grid
            or any(not math.isfinite(value) or value <= 0 for value in self.alpha_grid)
            or self.selected_alpha not in self.alpha_grid
            or not self.fold_scores
        ):
            raise ProtocolError("HARP v4 alpha selection is malformed.")


@dataclass(frozen=True)
class DeleteDonorFit:
    donor_id: str
    model: SharedDesignRidge
    inner_selection: AlphaSelection

    def __post_init__(self) -> None:
        if (
            not self.donor_id
            or not isinstance(self.model, SharedDesignRidge)
            or not isinstance(self.inner_selection, AlphaSelection)
            or self.donor_id not in self.model.excluded_center_ids
            or self.donor_id in self.model.training_query_ids
            or self.donor_id in self.model.training_candidate_ids
        ):
            raise ProtocolError("HARP v4 delete-donor fit leaked its donor.")


@dataclass(frozen=True)
class HarpV4Fit:
    outer_target_id: str
    feature_names: tuple[str, ...]
    alpha_selection: AlphaSelection
    full_model: SharedDesignRidge
    delete_donor_fits: tuple[DeleteDonorFit, ...]
    geometry_calibrations: tuple[GeometryCalibration, ...]
    residual_calibrations: tuple[DonorResidualCalibration, ...]
    support_summaries: tuple[SupportSummary, ...]
    residual_quantile: float
    geometry_quantile: float

    def __post_init__(self) -> None:
        if (
            not self.outer_target_id
            or not self.feature_names
            or not isinstance(self.full_model, SharedDesignRidge)
            or self.outer_target_id not in self.full_model.excluded_center_ids
            or self.full_model.feature_names != self.feature_names
            or not self.delete_donor_fits
            or tuple(fit.donor_id for fit in self.delete_donor_fits)
            != tuple(sorted({fit.donor_id for fit in self.delete_donor_fits}))
            or {value.comparison for value in self.geometry_calibrations} != set(Comparison)
            or {value.comparison for value in self.residual_calibrations} != set(Comparison)
            or not 0.5 <= float(self.residual_quantile) < 1.0
            or not 0.5 <= float(self.geometry_quantile) < 1.0
        ):
            raise ProtocolError("HARP v4 fitted bundle is malformed.")

    @property
    def selected_alpha(self) -> float:
        return self.alpha_selection.selected_alpha

    @property
    def donor_ids(self) -> tuple[str, ...]:
        return tuple(value.donor_id for value in self.delete_donor_fits)

    def geometry(self, comparison: Comparison) -> GeometryCalibration:
        key = Comparison(comparison)
        return next(value for value in self.geometry_calibrations if value.comparison is key)

    def residuals(self, comparison: Comparison) -> DonorResidualCalibration:
        key = Comparison(comparison)
        return next(value for value in self.residual_calibrations if value.comparison is key)

    def support(
        self, comparison: Comparison, candidate_source_id: str | None
    ) -> SupportSummary:
        key = (Comparison(comparison), candidate_source_id)
        try:
            return next(
                value
                for value in self.support_summaries
                if (value.comparison, value.candidate_source_id) == key
            )
        except StopIteration as exc:
            raise ProtocolError("Target action has no matching source-development support.") from exc


def _without_centers(
    rows: tuple[CaseTrainingObservation, ...], centers: set[str]
) -> tuple[CaseTrainingObservation, ...]:
    return tuple(
        row
        for row in rows
        if row.pseudo_query_id not in centers and row.candidate_source_id not in centers
    )


def _validation_for(
    rows: tuple[CaseTrainingObservation, ...], donor_id: str
) -> tuple[CaseTrainingObservation, ...]:
    # Role-complete donor holdout: the calibration surface contains both an
    # unseen pseudo-query view and an unseen candidate view.  This makes the
    # high raw leverage of the delete-candidate stress model part of the
    # source reference distribution instead of comparing it to query-only
    # leverage on an incompatible scale.
    return tuple(
        row
        for row in rows
        if row.pseudo_query_id == donor_id or row.candidate_source_id == donor_id
    )


def _predict_rows(
    model: SharedDesignRidge, rows: tuple[CaseTrainingObservation, ...]
) -> tuple[np.ndarray, np.ndarray]:
    return model.predict(
        [row.feature_values for row in rows],
        [row.pseudo_query_id for row in rows],
        [row.candidate_source_id for row in rows],
        [row.comparison for row in rows],
    )


def _select_alpha(
    rows: tuple[CaseTrainingObservation, ...],
    *,
    outer_target_id: str,
    alpha_grid: tuple[float, ...],
    ridge_memo: _RidgeFitMemo,
    permanently_excluded: tuple[str, ...] = (),
) -> AlphaSelection:
    donors = tuple(sorted(set(row.pseudo_query_id for row in rows)))
    if len(donors) < 3:
        raise ProtocolError("Nested HARP v4 LODO requires at least three source query centers.")
    fold_scores: list[AlphaFoldScore] = []
    aggregate: dict[float, list[float]] = {alpha: [] for alpha in alpha_grid}
    permanent = set(permanently_excluded)
    for donor in donors:
        training = _without_centers(rows, {donor})
        validation = _validation_for(rows, donor)
        if not training or not validation:
            raise ProtocolError("A nested HARP v4 LODO fold has no training or validation rows.")
        train_response = np.asarray([row.effects.as_tuple() for row in training], dtype=np.float64)
        scale = np.std(train_response, axis=0)
        scale[scale <= np.sqrt(np.finfo(np.float64).eps)] = 1.0
        observed = np.asarray([row.effects.as_tuple() for row in validation], dtype=np.float64)
        for alpha in alpha_grid:
            model = ridge_memo.fit(
                training,
                alpha=alpha,
                excluded_center_ids=tuple(sorted({outer_target_id, donor, *permanent})),
            )
            predicted, _ = _predict_rows(model, validation)
            mse = float(np.mean(((predicted - observed) / scale) ** 2))
            aggregate[alpha].append(mse)
            fold_scores.append(
                AlphaFoldScore(
                    heldout_donor_id=donor,
                    alpha=alpha,
                    standardized_mse=mse,
                    training_query_ids=model.training_query_ids,
                    training_candidate_ids=model.training_candidate_ids,
                )
            )
    selected = min(alpha_grid, key=lambda alpha: (float(np.mean(aggregate[alpha])), alpha))
    return AlphaSelection(selected, alpha_grid, tuple(fold_scores))


def _fit_pair_deleted_geometry_models(
    rows: tuple[CaseTrainingObservation, ...],
    *,
    outer_target_id: str,
    donors: tuple[str, ...],
    alpha_grid: tuple[float, ...],
    ridge_memo: _RidgeFitMemo,
) -> dict[tuple[str, str], SharedDesignRidge]:
    """Fit each unordered source exclusion pair once for geometry only."""

    if len(donors) < 5:
        raise ProtocolError(
            "Matched HARP v4 geometry calibration requires at least five source donors."
        )
    result: dict[tuple[str, str], SharedDesignRidge] = {}
    for left, right in combinations(donors, 2):
        pair = (left, right)
        deleted_rows = _without_centers(rows, set(pair))
        selection = _select_alpha(
            deleted_rows,
            outer_target_id=outer_target_id,
            alpha_grid=alpha_grid,
            ridge_memo=ridge_memo,
            permanently_excluded=pair,
        )
        model = ridge_memo.fit(
            deleted_rows,
            alpha=selection.selected_alpha,
            excluded_center_ids=tuple(sorted((outer_target_id, *pair))),
        )
        if (
            not set(pair).issubset(model.excluded_center_ids)
            or set(pair).intersection(model.training_query_ids)
            or set(pair).intersection(model.training_candidate_ids)
        ):
            raise ProtocolError("HARP v4 pair-deleted geometry model leaked h or d.")
        result[pair] = model
    return result


def _matched_geometry_inputs(
    rows: tuple[CaseTrainingObservation, ...],
    *,
    donors: tuple[str, ...],
    pair_models: dict[tuple[str, str], SharedDesignRidge],
) -> tuple[
    dict[Comparison, list[float]],
    dict[Comparison, list[str]],
    dict[Comparison, list[str]],
]:
    """Build one max-over-models block per held-out source query action."""

    raw_by_comparison = {comparison: [] for comparison in Comparison}
    donor_by_comparison = {comparison: [] for comparison in Comparison}
    block_by_comparison = {comparison: [] for comparison in Comparison}
    for heldout in donors:
        validation = tuple(row for row in rows if row.pseudo_query_id == heldout)
        if not validation or any(row.candidate_source_id == heldout for row in validation):
            raise ProtocolError("HARP v4 pseudo-target geometry surface is malformed.")
        models = tuple(
            pair_models[tuple(sorted((heldout, additional)))]
            for additional in donors
            if additional != heldout
        )
        leverage_columns: list[np.ndarray] = []
        for model in models:
            if (
                heldout not in model.excluded_center_ids
                or heldout in model.training_query_ids
                or heldout in model.training_candidate_ids
            ):
                raise ProtocolError("HARP v4 geometry ensemble leaked its pseudo-target.")
            _prediction, leverage = _predict_rows(model, validation)
            leverage_columns.append(leverage)
        matrix = np.column_stack(leverage_columns)
        if matrix.shape != (len(validation), len(donors) - 1):
            raise ProtocolError("HARP v4 geometry ensemble matrix is misaligned.")
        for index, row in enumerate(validation):
            block = json.dumps(row.row_key, ensure_ascii=True, separators=(",", ":"))
            for leverage in matrix[index]:
                raw_by_comparison[row.comparison].append(float(leverage))
                donor_by_comparison[row.comparison].append(heldout)
                block_by_comparison[row.comparison].append(block)
    if any(not raw_by_comparison[comparison] for comparison in Comparison):
        raise ProtocolError("HARP v4 geometry calibration lost a comparison.")
    return raw_by_comparison, donor_by_comparison, block_by_comparison


def _support_summaries(
    rows: tuple[CaseTrainingObservation, ...]
) -> tuple[SupportSummary, ...]:
    keys = sorted(
        {(row.comparison, row.candidate_source_id) for row in rows},
        key=lambda value: (value[0].value, value[1] or ""),
    )
    result: list[SupportSummary] = []
    for comparison, candidate in keys:
        block = tuple(
            row
            for row in rows
            if row.comparison is comparison and row.candidate_source_id == candidate
        )
        result.append(
            SupportSummary(
                comparison=comparison,
                candidate_source_id=candidate,
                donor_ids=tuple(sorted({row.pseudo_query_id for row in block})),
                paired_case_count=len({(row.pseudo_query_id, row.case_id) for row in block}),
                class_counts=(
                    sum(row.class_counts[0] for row in block),
                    sum(row.class_counts[1] for row in block),
                ),
            )
        )
    return tuple(result)


def _validate_paired_hierarchy(
    rows: tuple[CaseTrainingObservation, ...]
) -> None:
    """Require coherent U/B, H/B, and H/U source case triplets.

    The U/B effect is case-local and candidate independent.  Each physical
    candidate must pair H/B and H/U rows on that same source case, feature
    schema, exact feature vector, and class counts.  Their three response
    vectors must close algebraically, preventing independently constructed
    comparisons from teaching contradictory hierarchy decisions.
    """

    uniform: dict[tuple[str, str], CaseTrainingObservation] = {}
    expert: dict[
        tuple[str, str, str], dict[Comparison, CaseTrainingObservation]
    ] = {}
    for row in rows:
        case_key = (row.pseudo_query_id, row.case_id)
        if row.comparison is Comparison.U_VS_B:
            if case_key in uniform:
                raise ProtocolError("HARP v4 hierarchy has duplicate U-vs-B case support.")
            uniform[case_key] = row
            continue
        assert row.candidate_source_id is not None
        key = (*case_key, row.candidate_source_id)
        block = expert.setdefault(key, {})
        if row.comparison in block:
            raise ProtocolError("HARP v4 hierarchy has duplicate expert comparison support.")
        block[row.comparison] = row
    if not uniform or not expert:
        raise ProtocolError("HARP v4 hierarchy requires U and physical expert support.")
    for query in sorted({key[0] for key in uniform}):
        query_rows = tuple(
            row for (row_query, _case), row in uniform.items() if row_query == query
        )
        declared = {
            (
                row.pseudo_query_case_count,
                row.pseudo_query_class_support_case_counts,
            )
            for row in query_rows
        }
        observed_support = tuple(
            sum(row.class_counts[label] > 0 for row in query_rows)
            for label in (0, 1)
        )
        if declared != {(len(query_rows), observed_support)}:
            raise ProtocolError(
                "HARP v4 pseudo-query case-equal BACC normalization drifted."
            )
    for (query, case, _candidate), block in expert.items():
        if set(block) != {Comparison.HXE_VS_B, Comparison.HXE_VS_U}:
            raise ProtocolError("HARP v4 hierarchy has an incomplete physical expert triplet.")
        try:
            u_vs_b = uniform[(query, case)]
        except KeyError as exc:
            raise ProtocolError("HARP v4 expert triplet lacks paired U-vs-B case support.") from exc
        h_vs_b = block[Comparison.HXE_VS_B]
        h_vs_u = block[Comparison.HXE_VS_U]
        if (
            h_vs_b.feature_names != h_vs_u.feature_names
            or h_vs_b.feature_values != h_vs_u.feature_values
            or h_vs_b.class_counts != h_vs_u.class_counts
            or h_vs_b.class_counts != u_vs_b.class_counts
            or h_vs_b.feature_names != u_vs_b.feature_names
            or h_vs_b.pseudo_query_case_count
            != h_vs_u.pseudo_query_case_count
            or h_vs_b.pseudo_query_case_count
            != u_vs_b.pseudo_query_case_count
            or h_vs_b.pseudo_query_class_support_case_counts
            != h_vs_u.pseudo_query_class_support_case_counts
            or h_vs_b.pseudo_query_class_support_case_counts
            != u_vs_b.pseudo_query_class_support_case_counts
        ):
            raise ProtocolError("HARP v4 hierarchy triplet drifted in features or class counts.")
        left = np.asarray(h_vs_b.effects.as_tuple(), dtype=np.float64)
        right = np.asarray(h_vs_u.effects.as_tuple(), dtype=np.float64) + np.asarray(
            u_vs_b.effects.as_tuple(), dtype=np.float64
        )
        if not np.allclose(left, right, rtol=_EFFECT_RTOL, atol=_EFFECT_ATOL):
            raise ProtocolError("HARP v4 hierarchy effects are algebraically incoherent.")


def fit_harp_v4(
    observations: Sequence[CaseTrainingObservation],
    *,
    outer_target_id: str,
    alpha_grid: Sequence[float] = (0.01, 0.1, 1.0),
    residual_quantile: float = 0.9,
    geometry_quantile: float = 0.95,
) -> HarpV4Fit:
    """Fit one outer-target bundle without opening target labels.

    Alpha selection for each delete-donor model is nested inside that donor's
    training surface.  All training, normalization, selection, residual
    calibration, and leverage calibration remove a held-out center from both
    pseudo-query and candidate roles.
    """

    rows = tuple(observations)
    if not rows or any(not isinstance(row, CaseTrainingObservation) for row in rows):
        raise ProtocolError("HARP v4 fitting requires typed source-development rows.")
    target = str(outer_target_id)
    if any(row.outer_target_id != target for row in rows):
        raise ProtocolError("HARP v4 rows escaped their declared outer target.")
    if len({row.row_key for row in rows}) != len(rows):
        raise ProtocolError("HARP v4 source-development case identities are duplicated.")
    names = rows[0].feature_names
    if any(row.feature_names != names for row in rows):
        raise ProtocolError("HARP v4 feature schema drifted across source cases.")
    _validate_paired_hierarchy(rows)
    grid = tuple(sorted({float(value) for value in alpha_grid}))
    if not grid or any(not math.isfinite(value) or value <= 0 for value in grid):
        raise ProtocolError("HARP v4 alpha grid must be finite, positive, and nonempty.")
    if not 0.5 <= float(residual_quantile) < 1.0 or not 0.5 <= float(geometry_quantile) < 1.0:
        raise ProtocolError("HARP v4 calibration quantiles must lie in [0.5, 1).")
    if set(row.comparison for row in rows) != set(Comparison):
        raise ProtocolError("HARP v4 fitting requires all three hierarchical comparisons.")

    ridge_memo = _RidgeFitMemo(rows)
    outer_selection = _select_alpha(
        rows,
        outer_target_id=target,
        alpha_grid=grid,
        ridge_memo=ridge_memo,
    )
    full_model = ridge_memo.fit(
        rows,
        alpha=outer_selection.selected_alpha,
        excluded_center_ids=(target,),
    )
    donors = tuple(sorted(set(row.pseudo_query_id for row in rows)))
    candidate_donors = tuple(
        sorted({row.candidate_source_id for row in rows if row.candidate_source_id is not None})
    )
    if candidate_donors != donors:
        raise ProtocolError(
            "HARP v4 geometry requires the source query and candidate donor universes to match."
        )
    delete_fits: list[DeleteDonorFit] = []
    for donor in donors:
        deleted_rows = _without_centers(rows, {donor})
        inner = _select_alpha(
            deleted_rows,
            outer_target_id=target,
            alpha_grid=grid,
            ridge_memo=ridge_memo,
            permanently_excluded=(donor,),
        )
        model = ridge_memo.fit(
            deleted_rows,
            alpha=inner.selected_alpha,
            excluded_center_ids=tuple(sorted((target, donor))),
        )
        delete_fits.append(DeleteDonorFit(donor, model, inner))

    pair_models = _fit_pair_deleted_geometry_models(
        rows,
        outer_target_id=target,
        donors=donors,
        alpha_grid=grid,
        ridge_memo=ridge_memo,
    )
    geometry_raw, geometry_donors, geometry_blocks = _matched_geometry_inputs(
        rows, donors=donors, pair_models=pair_models
    )

    predicted_by_comparison: dict[Comparison, list[EffectVector]] = {
        value: [] for value in Comparison
    }
    observed_by_comparison: dict[Comparison, list[EffectVector]] = {
        value: [] for value in Comparison
    }
    donor_by_comparison: dict[Comparison, list[str]] = {
        value: [] for value in Comparison
    }
    case_block_by_comparison: dict[Comparison, list[str]] = {
        value: [] for value in Comparison
    }
    for deleted in delete_fits:
        validation = _validation_for(rows, deleted.donor_id)
        predictions, _leverages = _predict_rows(deleted.model, validation)
        for row, prediction in zip(validation, predictions, strict=True):
            predicted_by_comparison[row.comparison].append(EffectVector(*prediction))
            observed_by_comparison[row.comparison].append(row.effects)
            donor_by_comparison[row.comparison].append(deleted.donor_id)
            case_block_by_comparison[row.comparison].append(
                json.dumps(
                    (row.pseudo_query_id, row.case_id),
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
            )

    geometries = tuple(
        calibrate_geometry(
            comparison,
            geometry_raw[comparison],
            geometry_donors[comparison],
            geometry_blocks[comparison],
            quantile_level=float(geometry_quantile),
        )
        for comparison in Comparison
    )
    residuals = tuple(
        calibrate_donor_residuals(
            comparison,
            predicted_by_comparison[comparison],
            observed_by_comparison[comparison],
            donor_by_comparison[comparison],
            case_block_by_comparison[comparison],
            quantile_level=float(residual_quantile),
        )
        for comparison in Comparison
    )
    return HarpV4Fit(
        outer_target_id=target,
        feature_names=names,
        alpha_selection=outer_selection,
        full_model=full_model,
        delete_donor_fits=tuple(delete_fits),
        geometry_calibrations=geometries,
        residual_calibrations=residuals,
        support_summaries=_support_summaries(rows),
        residual_quantile=float(residual_quantile),
        geometry_quantile=float(geometry_quantile),
    )


__all__ = (
    "AlphaFoldScore",
    "AlphaSelection",
    "DeleteDonorFit",
    "HarpV4Fit",
    "fit_harp_v4",
)
