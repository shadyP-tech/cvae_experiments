"""Final/nested recoverability model fitting on the bounded CPU pool."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import math
import multiprocessing
from types import MappingProxyType

import numpy as np

from ...protocol import ProtocolError
from .constants import GEOMETRY_IDS, MIDOGPP_CENTERS, candidate_sources
from .contracts import ActionScoreRow, CaseActionFeatureRow, RidgeActionModel, UtilityTargetRow
from .execution_support import model_payload, score_payload
from .features import matched_blocked_feature_permutation
from .hashing import canonical_hash, finite, require_sha256
from .models import fit_fixed_alpha_ridge_models, predict_action_scores
from .utility_execution import LocoUtilityProduct, PrelabelProducts


FAMILIES = ("G", "R", "P")


@dataclass(frozen=True, order=True)
class NestedPredictionDiagnostic:
    outer_target_center: str
    heldout_query_center: str
    case_id: str
    geometry_id: str
    selected_source: str
    family: str
    observed_gain: float
    predicted_gain: float
    squared_error: float
    model_hash: str

    def __post_init__(self) -> None:
        outer, query = str(self.outer_target_center), str(self.heldout_query_center)
        if (
            outer not in MIDOGPP_CENTERS
            or query not in candidate_sources(outer)
            or self.selected_source not in set(MIDOGPP_CENTERS).difference((outer, query))
            or self.geometry_id not in GEOMETRY_IDS
            or self.family not in FAMILIES
        ):
            raise ProtocolError("Nested diagnostic violates H/q/e exclusion.")
        observed = finite(self.observed_gain, "observed_gain")
        predicted = finite(self.predicted_gain, "predicted_gain")
        squared = finite(self.squared_error, "squared_error")
        if squared < 0.0 or not math.isclose(
            squared, (observed - predicted) ** 2, rel_tol=1.0e-12, abs_tol=1.0e-15
        ):
            raise ProtocolError("Nested squared error is inconsistent.")
        require_sha256(self.model_hash, "model_hash")

    def to_payload(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True, order=True)
class NestedMseSummary:
    outer_target_center: str
    geometry_id: str
    family: str
    row_count: int
    mse: float

    def __post_init__(self) -> None:
        if (
            self.outer_target_center not in MIDOGPP_CENTERS
            or self.geometry_id not in GEOMETRY_IDS
            or self.family not in FAMILIES
            or self.row_count <= 0
            or finite(self.mse, "nested mse") < 0.0
        ):
            raise ProtocolError("Nested MSE summary is invalid.")


@dataclass(frozen=True)
class TargetModelProduct:
    outer_target_center: str
    models: tuple[RidgeActionModel, ...]
    scores: tuple[ActionScoreRow, ...]
    nested_predictions: tuple[NestedPredictionDiagnostic, ...]
    nested_mse: tuple[NestedMseSummary, ...]
    model_seals: Mapping[str, str]
    utility_product_hash: str
    feature_surface_hash: str
    permutation_provenance_hash: str
    probability_surface_hash: str
    target_product_hash: str = field(init=False)

    def __post_init__(self) -> None:
        outer, models = str(self.outer_target_center), tuple(self.models)
        final = tuple(row for row in models if row.heldout_donor_center is None)
        nested = tuple(row for row in models if row.heldout_donor_center is not None)
        if outer not in MIDOGPP_CENTERS or any(row.outer_target_center != outer for row in models):
            raise ProtocolError("Target model product mixes outer targets.")
        if len(final) != 2 * 3 * 8 or len(nested) != 2 * 3 * 8 * 7:
            raise ProtocolError("Target model product has incomplete final/nested topology.")
        expected_seals = {f"{g}:{f}" for g in GEOMETRY_IDS for f in FAMILIES}
        seals = {str(key): str(value) for key, value in self.model_seals.items()}
        if set(seals) != expected_seals:
            raise ProtocolError("Target model family seals are incomplete.")
        for key, value in seals.items():
            require_sha256(value, f"model seal {key}")
        if tuple((row.geometry_id, row.family) for row in self.nested_mse) != tuple(
            (g, f) for g in GEOMETRY_IDS for f in FAMILIES
        ):
            raise ProtocolError("Nested MSE summaries are not canonical.")
        for value, name in (
            (self.utility_product_hash, "utility_product_hash"),
            (self.feature_surface_hash, "feature_surface_hash"),
            (self.permutation_provenance_hash, "permutation_provenance_hash"),
            (self.probability_surface_hash, "probability_surface_hash"),
        ):
            require_sha256(value, name)
        object.__setattr__(self, "outer_target_center", outer)
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "scores", tuple(sorted(self.scores, key=lambda row: row.row_key)))
        object.__setattr__(self, "nested_predictions", tuple(sorted(self.nested_predictions)))
        object.__setattr__(self, "model_seals", MappingProxyType(dict(sorted(seals.items()))))
        object.__setattr__(self, "target_product_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_actionability_target_model_product_v1",
            "outer_target_center": self.outer_target_center,
            "models": [model_payload(row) for row in self.models],
            "scores": [score_payload(row) for row in self.scores],
            "nested_predictions": [row.to_payload() for row in self.nested_predictions],
            "nested_mse": [dict(row.__dict__) for row in self.nested_mse],
            "model_seals": dict(self.model_seals),
            "utility_product_hash": self.utility_product_hash,
            "feature_surface_hash": self.feature_surface_hash,
            "permutation_provenance_hash": self.permutation_provenance_hash,
            "probability_surface_hash": self.probability_surface_hash,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "target_product_hash": self.target_product_hash}


def _predict_nested(
    models: Sequence[RidgeActionModel],
    features: Sequence[CaseActionFeatureRow],
    targets: Sequence[UtilityTargetRow],
) -> tuple[NestedPredictionDiagnostic, ...]:
    fitted = tuple(models)
    contexts = {
        (row.outer_target_center, row.heldout_donor_center, row.geometry_id, row.family)
        for row in fitted
    }
    if len(contexts) != 1:
        raise ProtocolError("Nested prediction models mix contexts.")
    outer, query, geometry, family = next(iter(contexts))
    if query is None:
        raise ProtocolError("Nested prediction requires a heldout query q.")
    feature_rows = (
        matched_blocked_feature_permutation(features, excluded_candidate_centers=(outer, query))
        if family == "P"
        else tuple(features)
    )
    feature_by_key = {row.row_key: row for row in feature_rows}
    target_by_key = {row.row_key: row for row in targets}
    model_by_source = {row.selected_source: row for row in fitted}
    expected_sources = tuple(source for source in candidate_sources(outer) if source != query)
    if tuple(model_by_source) != expected_sources:
        raise ProtocolError("Nested prediction lacks its seven legal e candidates.")
    cases = sorted(
        {
            row.case_id
            for row in targets
            if row.query_center == query and row.geometry_id == geometry
        }
    )
    output: list[NestedPredictionDiagnostic] = []
    for case_id in cases:
        for source in expected_sources:
            key = (query, case_id, geometry, source)
            if key not in feature_by_key or key not in target_by_key:
                raise ProtocolError("Nested q diagnostic surface is incomplete.")
            model, feature = model_by_source[source], feature_by_key[key]
            raw = np.ones(1, dtype=np.float64) if family == "G" else np.asarray(feature.values, dtype=np.float64)
            design = (raw - np.asarray(model.means, dtype=np.float64)) / np.asarray(model.scales, dtype=np.float64)
            design[0] = 1.0
            prediction = float(design @ np.asarray(model.coefficients, dtype=np.float64))
            observed = target_by_key[key].response
            output.append(
                NestedPredictionDiagnostic(
                    outer, query, case_id, geometry, source, family,
                    observed, prediction, (observed - prediction) ** 2, model.model_hash,
                )
            )
    return tuple(output)


def _fit_family_task(
    task: tuple[tuple[CaseActionFeatureRow, ...], tuple[UtilityTargetRow, ...], str, str, str, int]
) -> tuple[tuple[RidgeActionModel, ...], tuple[ActionScoreRow, ...], tuple[NestedPredictionDiagnostic, ...]]:
    features, targets, outer, geometry, family, threads = task
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=threads):
        final = fit_fixed_alpha_ridge_models(
            features, targets, outer_target_center=outer, geometry_id=geometry, family=family
        )
        scores = predict_action_scores(final, features)
        all_models, diagnostics = list(final), []
        for query in candidate_sources(outer):
            nested = fit_fixed_alpha_ridge_models(
                features, targets, outer_target_center=outer,
                heldout_donor_center=query, geometry_id=geometry, family=family,
            )
            all_models.extend(nested)
            diagnostics.extend(_predict_nested(nested, features, targets))
    return tuple(all_models), scores, tuple(diagnostics)


def _validate_runtime(workers: int, threads: int, start_method: str) -> None:
    if (
        isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 4
        or isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 3
        or workers * threads > 12
    ):
        raise ProtocolError("CPU model pool exceeds the frozen 4x3 workstation budget.")
    if start_method != "spawn":
        raise ProtocolError("CPU model workers require deterministic spawn semantics.")


def fit_target_model_product(
    prelabel: PrelabelProducts,
    utility: LocoUtilityProduct,
    *,
    workers: int = 4,
    threads_per_worker: int = 3,
    start_method: str = "spawn",
) -> TargetModelProduct:
    if utility.probability_surface_hash != prelabel.probability_surface_hash:
        raise ProtocolError("Utility and prelabel probability seals differ.")
    _validate_runtime(workers, threads_per_worker, start_method)
    tasks = tuple(
        (prelabel.features, utility.rows, utility.outer_target_center, geometry, family, threads_per_worker)
        for geometry in GEOMETRY_IDS for family in FAMILIES
    )
    if workers == 1:
        values = tuple(_fit_family_task(task) for task in tasks)
    else:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)),
            mp_context=multiprocessing.get_context(start_method),
        ) as pool:
            values = tuple(pool.map(_fit_family_task, tasks, chunksize=1))
    mse_rows, seals = [], {}
    for (geometry, family), value in zip(
        ((g, f) for g in GEOMETRY_IDS for f in FAMILIES), values, strict=True
    ):
        family_models, family_scores, nested = value
        mse = NestedMseSummary(
            utility.outer_target_center, geometry, family, len(nested),
            float(np.asarray([row.squared_error for row in nested], dtype=np.float64).mean()),
        )
        mse_rows.append(mse)
        seals[f"{geometry}:{family}"] = canonical_hash(
            {
                "schema_version": "fixed_bank_actionability_model_family_seal_v1",
                "outer_target_center": utility.outer_target_center,
                "geometry_id": geometry,
                "family": family,
                "models": [model_payload(row) for row in family_models],
                "target_scores": [score_payload(row) for row in family_scores],
                "nested_mse": mse.mse,
                "feature_surface_hash": prelabel.feature_surface_hash,
                "permutation_provenance_hash": prelabel.permutation_provenance_hash,
            }
        )
    return TargetModelProduct(
        utility.outer_target_center,
        tuple(row for value in values for row in value[0]),
        tuple(row for value in values for row in value[1]),
        tuple(row for value in values for row in value[2]),
        tuple(mse_rows), seals, utility.utility_product_hash,
        prelabel.feature_surface_hash, prelabel.permutation_provenance_hash,
        prelabel.probability_surface_hash,
    )


@dataclass(frozen=True)
class ModelProducts:
    models: tuple[RidgeActionModel, ...]
    scores: tuple[ActionScoreRow, ...]
    nested_predictions: tuple[NestedPredictionDiagnostic, ...]
    nested_mse: tuple[NestedMseSummary, ...]
    model_seals_by_target: Mapping[str, Mapping[str, str]]
    all_models_seal_hash: str
    permutation_provenance_hash: str
    protocol_contract_hash: str
    probability_surface_hash: str
    model_products_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if {row.outer_target_center for row in self.models} != set(MIDOGPP_CENTERS):
            raise ProtocolError("Model products must cover all outer targets.")
        seals = {
            str(target): MappingProxyType(dict(sorted(dict(value).items())))
            for target, value in self.model_seals_by_target.items()
        }
        if set(seals) != set(MIDOGPP_CENTERS):
            raise ProtocolError("Model seals must cover every target H.")
        for value, name in (
            (self.all_models_seal_hash, "all_models_seal_hash"),
            (self.permutation_provenance_hash, "permutation_provenance_hash"),
            (self.protocol_contract_hash, "protocol_contract_hash"),
            (self.probability_surface_hash, "probability_surface_hash"),
        ):
            require_sha256(value, name)
        object.__setattr__(self, "model_seals_by_target", MappingProxyType(seals))
        object.__setattr__(self, "model_products_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_actionability_model_products_v1",
            "model_count": len(self.models), "score_count": len(self.scores),
            "nested_prediction_count": len(self.nested_predictions),
            "nested_mse": [dict(row.__dict__) for row in self.nested_mse],
            "model_seals_by_target": {key: dict(value) for key, value in self.model_seals_by_target.items()},
            "all_models_seal_hash": self.all_models_seal_hash,
            "permutation_provenance_hash": self.permutation_provenance_hash,
            "protocol_contract_hash": self.protocol_contract_hash,
            "probability_surface_hash": self.probability_surface_hash,
            "target_labels_used_for_shared_fit": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {
            **self._unhashed(), "models": [model_payload(row) for row in self.models],
            "scores": [score_payload(row) for row in self.scores],
            "nested_predictions": [row.to_payload() for row in self.nested_predictions],
            "model_products_hash": self.model_products_hash,
        }


def combine_model_products(
    prelabel: PrelabelProducts, target_products: Sequence[TargetModelProduct]
) -> ModelProducts:
    products = tuple(target_products)
    if tuple(row.outer_target_center for row in products) != MIDOGPP_CENTERS:
        raise ProtocolError("Target model products must be supplied in canonical H order.")
    if any(
        row.feature_surface_hash != prelabel.feature_surface_hash
        or row.permutation_provenance_hash != prelabel.permutation_provenance_hash
        or row.probability_surface_hash != prelabel.probability_surface_hash
        for row in products
    ):
        raise ProtocolError("Target model product prelabel provenance drifted.")
    seals = {row.outer_target_center: row.model_seals for row in products}
    all_seal = canonical_hash(
        {
            "schema_version": "fixed_bank_actionability_all_models_seal_v1",
            "target_product_hashes": [row.target_product_hash for row in products],
            "model_seals_by_target": {key: dict(value) for key, value in seals.items()},
            "feature_surface_hash": prelabel.feature_surface_hash,
            "permutation_provenance_hash": prelabel.permutation_provenance_hash,
            "protocol_contract_hash": prelabel.protocol_contract_hash,
        }
    )
    return ModelProducts(
        tuple(row for product in products for row in product.models),
        tuple(sorted((row for product in products for row in product.scores), key=lambda row: row.row_key)),
        tuple(sorted(row for product in products for row in product.nested_predictions)),
        tuple(row for product in products for row in product.nested_mse),
        seals, all_seal, prelabel.permutation_provenance_hash, prelabel.protocol_contract_hash,
        prelabel.probability_surface_hash,
    )


__all__ = (
    "ModelProducts", "NestedMseSummary", "NestedPredictionDiagnostic", "TargetModelProduct",
    "combine_model_products", "fit_target_model_product",
)
