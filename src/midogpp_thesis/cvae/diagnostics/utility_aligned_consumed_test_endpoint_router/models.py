"""Strict source-inner M0/M1/P modeling and transfer authorization."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import canonical_sha256
from ...routing.utility_aligned import (
    EnsembleCardinalityTransferResult,
    EnsembleUtilityModel,
    evaluate_ensemble_cardinality_transfer,
    fit_ensemble_utility_model,
)
from .contracts import (
    CENTERS,
    M0_PREDICTOR_NAMES,
    M1_PREDICTOR_NAMES,
    PERMUTATION_SEED,
    candidate_sources,
)
from .endpoint_adapter import DevelopmentEndpointResponseSet
from .features import SourceInnerFeatureSurfaceSet, SourceInnerFeatureSurfaces


@dataclass(frozen=True)
class EndpointRouterModels:
    """One H-excluded fitted model triplet plus its source-transfer gate."""

    outer_target_id: str
    global_model: EnsembleUtilityModel
    routed_model: EnsembleUtilityModel
    permutation_model: EnsembleUtilityModel
    cardinality_transfer: EnsembleCardinalityTransferResult
    source_feature_surface_hash: str
    development_response_set_hash: str
    model_hash: str

    def __post_init__(self) -> None:
        target = str(self.outer_target_id)
        models = (self.global_model, self.routed_model, self.permutation_model)
        if (
            target not in CENTERS
            or any(
                not isinstance(model, EnsembleUtilityModel)
                or model.outer_target_id != target
                for model in models
            )
            or self.global_model.feature_names != M0_PREDICTOR_NAMES
            or self.routed_model.feature_names != M1_PREDICTOR_NAMES
            or self.permutation_model.feature_names != M1_PREDICTOR_NAMES
            or self.global_model.permutation_seed is not None
            or self.routed_model.permutation_seed is not None
            or self.permutation_model.permutation_seed != PERMUTATION_SEED
            or not isinstance(
                self.cardinality_transfer, EnsembleCardinalityTransferResult
            )
            or self.cardinality_transfer.outer_target_id != target
            or not _text(self.source_feature_surface_hash)
            or not _text(self.development_response_set_hash)
        ):
            raise ProtocolError("Endpoint-router model boundary drifted.")
        expected_keys = {
            (target, query, source)
            for query in candidate_sources(target)
            for source in candidate_sources(target)
            if source != query
        }
        for model in models:
            if set(model.crossfit_row_keys) != expected_keys or len(model.fold_audits) != 56:
                raise ProtocolError("Endpoint-router model H/q/e crossfit coverage drifted.")
            for audit in model.fold_audits:
                outer, query, source = audit.predicted_row_key
                if (
                    outer != target
                    or set(audit.excluded_domain_ids) != {target, query, source}
                    or set(audit.training_query_ids) & {target, query, source}
                    or set(audit.training_source_ids) & {target, query, source}
                    or audit.strict_h_q_e_exclusion is not True
                ):
                    raise ProtocolError("Endpoint-router strict H/q/e audit failed.")
        if self.model_hash != canonical_sha256(self._unhashed_payload(target)):
            raise ProtocolError("Endpoint-router model hash drifted.")
        object.__setattr__(self, "outer_target_id", target)

    @property
    def by_role(self) -> Mapping[str, EnsembleUtilityModel]:
        return MappingProxyType(
            {
                "G": self.global_model,
                "R": self.routed_model,
                "P": self.permutation_model,
            }
        )

    def _unhashed_payload(self, target: str | None = None) -> dict[str, object]:
        return {
            "schema_version": "midogpp_consumed_test_endpoint_router_models_v1",
            "outer_target_id": target or self.outer_target_id,
            "model_hashes_by_role": {
                "G": self.global_model.model_hash,
                "R": self.routed_model.model_hash,
                "P": self.permutation_model.model_hash,
            },
            "cardinality_transfer_hash": self.cardinality_transfer.transfer_hash,
            "source_feature_surface_hash": self.source_feature_surface_hash,
            "development_response_set_hash": self.development_response_set_hash,
            "training_response_count": len(self.routed_model.crossfit_row_keys),
            "response_unit": "candidate_H_q_e_exact_nine_probability_ensemble",
            "alpha_tuning_endpoint": self.routed_model.routing_tuning_endpoint,
            "strict_H_q_e_exclusion": True,
            "same_outer_H_evaluation_labels_used_for_fit": False,
            "support_labels_used_for_fit": False,
            "target_features_used_for_fit": False,
            "seed_rows_are_independent_observations": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "model_hash": self.model_hash}


@dataclass(frozen=True)
class EndpointRouterModelSet:
    by_target: Mapping[str, EndpointRouterModels]
    source_feature_surface_set_hash: str
    development_response_set_hash: str
    model_set_hash: str

    def __post_init__(self) -> None:
        values = {str(key): value for key, value in self.by_target.items()}
        if (
            tuple(values) != CENTERS
            or any(value.outer_target_id != target for target, value in values.items())
        ):
            raise ProtocolError("Endpoint-router model set is incomplete.")
        payload = _model_set_payload(
            values,
            source_feature_surface_set_hash=self.source_feature_surface_set_hash,
            development_response_set_hash=self.development_response_set_hash,
        )
        if self.model_set_hash != canonical_sha256(payload):
            raise ProtocolError("Endpoint-router model-set hash drifted.")
        object.__setattr__(self, "by_target", MappingProxyType(values))

    def to_payload(self) -> dict[str, object]:
        return {
            **_model_set_payload(
                self.by_target,
                source_feature_surface_set_hash=self.source_feature_surface_set_hash,
                development_response_set_hash=self.development_response_set_hash,
            ),
            "model_set_hash": self.model_set_hash,
        }


def fit_endpoint_router_models(
    source_features: SourceInnerFeatureSurfaces,
    development_responses: DevelopmentEndpointResponseSet,
    *,
    alphas: Sequence[float] | None = None,
) -> EndpointRouterModels:
    """Fit G/R/P using only q!=H development responses.

    There is no target-label or target-feature argument.  The optional alpha
    override exists for focused tests; production omission delegates exactly
    to the neutral utility-aligned default grid and regret selector.
    """

    if (
        not isinstance(source_features, SourceInnerFeatureSurfaces)
        or not isinstance(development_responses, DevelopmentEndpointResponseSet)
    ):
        raise ProtocolError("Endpoint-router fitting requires typed sealed inputs.")
    target = source_features.outer_target_id
    outer_response_rows = development_responses.rows_for_outer_target(target)
    if len(outer_response_rows) != 56:
        raise ProtocolError("Endpoint-router fitting requires exactly 56 rows for H.")
    kwargs = {} if alphas is None else {"alphas": tuple(alphas)}
    global_model = fit_ensemble_utility_model(
        source_features.m0, outer_response_rows, **kwargs
    )
    routed_model = fit_ensemble_utility_model(
        source_features.m1, outer_response_rows, **kwargs
    )
    permutation_model = fit_ensemble_utility_model(
        source_features.permutation, outer_response_rows, **kwargs
    )
    transfer = evaluate_ensemble_cardinality_transfer(
        global_model,
        routed_model,
        permutation_model,
        outer_response_rows,
    )
    outer_response_binding_hash = (
        development_responses.binding_hash_for_outer_target(target)
    )
    payload = {
        "schema_version": "midogpp_consumed_test_endpoint_router_models_v1",
        "outer_target_id": target,
        "model_hashes_by_role": {
            "G": global_model.model_hash,
            "R": routed_model.model_hash,
            "P": permutation_model.model_hash,
        },
        "cardinality_transfer_hash": transfer.transfer_hash,
        "source_feature_surface_hash": source_features.surface_hash,
        "development_response_set_hash": outer_response_binding_hash,
        "training_response_count": len(routed_model.crossfit_row_keys),
        "response_unit": "candidate_H_q_e_exact_nine_probability_ensemble",
        "alpha_tuning_endpoint": routed_model.routing_tuning_endpoint,
        "strict_H_q_e_exclusion": True,
        "same_outer_H_evaluation_labels_used_for_fit": False,
        "support_labels_used_for_fit": False,
        "target_features_used_for_fit": False,
        "seed_rows_are_independent_observations": False,
    }
    return EndpointRouterModels(
        outer_target_id=target,
        global_model=global_model,
        routed_model=routed_model,
        permutation_model=permutation_model,
        cardinality_transfer=transfer,
        source_feature_surface_hash=source_features.surface_hash,
        development_response_set_hash=outer_response_binding_hash,
        model_hash=canonical_sha256(payload),
    )


def fit_endpoint_router_model_set(
    source_features: SourceInnerFeatureSurfaceSet,
    development_responses: DevelopmentEndpointResponseSet,
    *,
    alphas: Sequence[float] | None = None,
) -> EndpointRouterModelSet:
    if not isinstance(source_features, SourceInnerFeatureSurfaceSet):
        raise ProtocolError("Endpoint-router model set requires typed feature inputs.")
    values = {
        target: fit_endpoint_router_models(
            source_features.by_target[target],
            development_responses,
            alphas=alphas,
        )
        for target in CENTERS
    }
    payload = _model_set_payload(
        values,
        source_feature_surface_set_hash=source_features.surface_set_hash,
        development_response_set_hash=development_responses.response_set_hash,
    )
    return EndpointRouterModelSet(
        by_target=values,
        source_feature_surface_set_hash=source_features.surface_set_hash,
        development_response_set_hash=development_responses.response_set_hash,
        model_set_hash=canonical_sha256(payload),
    )


def _model_set_payload(
    values: Mapping[str, EndpointRouterModels],
    *,
    source_feature_surface_set_hash: str,
    development_response_set_hash: str,
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_consumed_test_endpoint_router_model_set_v1",
        "centers": list(CENTERS),
        "model_hashes_by_target": {
            target: values[target].model_hash for target in CENTERS
        },
        "cardinality_transfer_hashes_by_target": {
            target: values[target].cardinality_transfer.transfer_hash
            for target in CENTERS
        },
        "source_feature_surface_set_hash": source_feature_surface_set_hash,
        "development_response_set_hash": development_response_set_hash,
        "strict_H_q_e_exclusion": True,
        "support_labels_used": False,
        "same_outer_H_evaluation_labels_used_for_model_H": False,
    }


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value) and value.strip() == value


__all__ = (
    "EndpointRouterModelSet",
    "EndpointRouterModels",
    "fit_endpoint_router_model_set",
    "fit_endpoint_router_models",
)
