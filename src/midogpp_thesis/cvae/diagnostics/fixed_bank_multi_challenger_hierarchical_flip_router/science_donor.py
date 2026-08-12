"""Strict-H/q/e pooled directional donor-model construction."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
from types import MappingProxyType
from typing import Mapping

from ...protocol import ProtocolError
from ...routing.hierarchical_multi_challenger import (
    DirectionalDonorRow,
    DirectionalLogitModel,
    fit_directional_logit,
    permute_complete_case_feature_blocks,
)
from ...routing.threshold_flip_case_router import DonorRow, fit_two_head_ridge
from .constants import CENTERS, a1_action_id, candidate_sources
from .hashing import canonical_hash
from .products import DonorPhaseResult
from .science_common import (
    case_contribution,
    cases_for_center,
    core_feature,
    direction_counts,
    feature_index,
    label_index,
    probability_index,
)


def fit_h_specific_donor_phase(
    *,
    probability_surface: object,
    prelabel: object,
    partition: object,
    manager: object,
    config: object,
) -> DonorPhaseResult:
    """Fit G/R/P pooled-binomial models and the frozen prior single control."""

    _assert_model_runtime(config)
    probability = probability_index(probability_surface)
    features = feature_index(prelabel)
    directional_by_target: dict[str, tuple[DirectionalDonorRow, ...]] = {}
    single_by_target: dict[str, tuple[DonorRow, ...]] = {}
    contribution_rows: list[Mapping[str, object]] = []

    for heldout_h in CENTERS:
        labels = tuple(manager.open_loco_donor_labels(heldout_h))
        scoped = label_index(labels)
        directional: list[DirectionalDonorRow] = []
        single: list[DonorRow] = []
        for query in CENTERS:
            if query == heldout_h:
                continue
            for source in candidate_sources(query):
                if source == heldout_h:
                    continue
                action_id = a1_action_id(source)
                for case_id in cases_for_center(partition, query):
                    feature = core_feature(features[(query, case_id, action_id)])
                    contribution = case_contribution(
                        probability,
                        scoped,
                        target_center=query,
                        case_id=case_id,
                        action_id=action_id,
                    )
                    single.append(
                        DonorRow(
                            model_target=heldout_h,
                            query_center=query,
                            candidate_source=source,
                            case_id=case_id,
                            action_id=action_id,
                            feature_case_id=case_id,
                            feature_names=feature.feature_names,
                            values=feature.values,
                            target=contribution,
                        )
                    )
                    counts = direction_counts(
                        probability,
                        scoped,
                        target_center=query,
                        case_id=case_id,
                        action_id=action_id,
                    )
                    if (
                        counts["0to1"][1] != feature.flip_0to1_count
                        or counts["1to0"][1] != feature.flip_1to0_count
                    ):
                        raise ProtocolError(
                            "Directional donor counts differ from the prelabel seal."
                        )
                    for direction in ("0to1", "1to0"):
                        success_count, trial_count = counts[direction]
                        row = DirectionalDonorRow(
                            model_target=heldout_h,
                            query_center=query,
                            candidate_source=source,
                            case_id=case_id,
                            action_id=action_id,
                            feature_case_id=case_id,
                            direction=direction,
                            success_count=success_count,
                            trial_count=trial_count,
                            feature_names=feature.feature_names,
                            values=feature.values,
                        )
                        directional.append(row)
                        payload = row.to_payload()
                        contribution_rows.append(
                            {**payload, "row_hash": canonical_hash(payload)}
                        )
        directional_by_target[heldout_h] = tuple(directional)
        single_by_target[heldout_h] = tuple(single)

    workers = int(getattr(config, "runtime")["model_workers"])
    threads = int(getattr(config, "runtime")["model_threads_per_worker"])
    seed = int(getattr(config, "protocol")["partition_seed"])
    jobs = tuple(
        (
            target,
            directional_by_target[target],
            single_by_target[target],
            seed,
            threads,
        )
        for target in CENTERS
    )
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
        initializer=_worker_initializer,
        initargs=(threads,),
    ) as executor:
        fitted = tuple(executor.map(_fit_target_job, jobs))

    fit_rows: list[Mapping[str, object]] = []
    seals: dict[str, Mapping[str, object]] = {}
    models_by_target: dict[
        str, Mapping[str, Mapping[str, DirectionalLogitModel]]
    ] = {}
    single_models = {}
    for target, families, single_model, permutation_hash in fitted:
        models_by_target[target] = _freeze_received_families(families)
        single_models[target] = single_model
        fit_payload = {
            "heldout_target_H": target,
            "families": {
                family: {
                    direction: model.to_payload()
                    for direction, model in sorted(by_direction.items())
                }
                for family, by_direction in sorted(families.items())
            },
            "single_challenger_model": single_model.to_payload(),
            "permutation_row_surface_hash": permutation_hash,
        }
        fit_rows.append(fit_payload)
        provenance_hash = canonical_hash(
            {
                family: {
                    direction: model.provenance_hash
                    for direction, model in sorted(by_direction.items())
                }
                for family, by_direction in sorted(families.items())
            }
        )
        fit_contracts = {
            family: {
                direction: _directional_model_semantic_contract(model)
                for direction, model in sorted(by_direction.items())
            }
            for family, by_direction in sorted(families.items())
        }
        single_fit_contract = _single_model_semantic_contract(single_model)
        composite_model_hash = canonical_hash(
            {
                "schema_version": "fixed_bank_multi_challenger_H_models_v1",
                "heldout_target_H": target,
                "fit_contracts": fit_contracts,
                "single_challenger_fit_contract": single_fit_contract,
                "strict_H_q_e_exclusion": True,
                "fitted_numeric_validation": (
                    "raw_values_persisted_and_replayed_with_"
                    "isclose_atol_5e-12_rtol_5e-12"
                ),
            }
        )
        seal_unhashed = {
            "schema_version": "fixed_bank_multi_challenger_H_model_seal_v1",
            "heldout_target_H": target,
            "composite_model_hash": composite_model_hash,
            "composite_provenance_hash": provenance_hash,
            "fit_contracts": fit_contracts,
            "single_challenger_fit_contract": single_fit_contract,
            "fitted_numeric_validation": (
                "raw_values_persisted_and_replayed_with_"
                "isclose_atol_5e-12_rtol_5e-12"
            ),
            "permutation_row_surface_hash": permutation_hash,
            "strict_H_q_e_exclusion": True,
            "heldout_H_labels_used": False,
        }
        seal = {
            **seal_unhashed,
            # These reconstruct the raw fitted payload in this bundle but are
            # deliberately outside the process-stable semantic seal.
            "fit_fingerprints": {
                family: {
                    direction: model.fit_fingerprint
                    for direction, model in sorted(by_direction.items())
                }
                for family, by_direction in sorted(families.items())
            },
            "single_challenger_model_hash": single_model.model_hash,
            "seal_hash": canonical_hash(seal_unhashed),
        }
        manager.record_H_specific_donor_model_seal(
            target,
            model_heldout_target=target,
            model_hash=composite_model_hash,
            provenance_hash=provenance_hash,
        )
        seals[target] = MappingProxyType(seal)

    permutation_unhashed = {
        "schema_version": "fixed_bank_multi_challenger_permutation_provenance_v1",
        "seed": seed,
        "scope": "complete_case_action_direction_blocks_within_H_and_query_q",
        "same_capacity_refit": True,
        "label_responses_permuted": False,
        "feature_blocks_deranged": True,
        "surface_hashes_by_target": {
            target: next(
                row[3] for row in fitted if row[0] == target
            )
            for target in CENTERS
        },
    }
    return DonorPhaseResult(
        contribution_rows=tuple(contribution_rows),
        fit_rows=tuple(fit_rows),
        model_seals=MappingProxyType(seals),
        permutation_provenance=MappingProxyType(
            {
                **permutation_unhashed,
                "permutation_provenance_hash": canonical_hash(permutation_unhashed),
            }
        ),
        models_by_target_family=MappingProxyType(models_by_target),
        single_models_by_target=MappingProxyType(single_models),
    )


def _directional_model_semantic_contract(
    model: DirectionalLogitModel,
) -> Mapping[str, object]:
    return {
        "schema_version": "hierarchical_directional_logit_fit_v2",
        "model_target": model.model_target,
        "family": model.family,
        "direction": model.direction,
        "feature_names": list(model.feature_names),
        "candidate_sources": list(model.candidate_sources),
        "query_centers": list(model.query_centers),
        "feature_alpha": model.feature_alpha,
        "source_alpha": model.source_alpha,
        "query_alpha": model.query_alpha,
        "intercept_alpha": model.intercept_alpha,
        "training_row_count": model.training_row_count,
        "training_trial_count": model.training_trial_count,
        "training_case_clusters": list(model.training_case_clusters),
        "provenance_hash": model.provenance_hash,
        "fitted_numeric_fields": [
            "feature_mean",
            "feature_scale",
            "coefficients",
            "covariance",
        ],
        "fitted_numeric_validation": "replay_isclose_atol_5e-12_rtol_5e-12",
    }


def _single_model_semantic_contract(model: object) -> Mapping[str, object]:
    payload = model.to_payload()
    return {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "feature_mean",
            "feature_scale",
            "tp_head",
            "tn_head",
            "model_hash",
        }
    } | {
        "fitted_numeric_fields": [
            "feature_mean",
            "feature_scale",
            "tp_head",
            "tn_head",
        ],
        "fitted_numeric_validation": "replay_isclose_atol_5e-12_rtol_5e-12",
    }


def _fit_target_job(
    job: tuple[
        str,
        tuple[DirectionalDonorRow, ...],
        tuple[DonorRow, ...],
        int,
        int,
    ],
) -> tuple[
    str,
    Mapping[str, Mapping[str, DirectionalLogitModel]],
    object,
    str,
]:
    target, directional, single_rows, seed, threads = job
    try:
        from threadpoolctl import threadpool_limits
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Multi-challenger fitting requires threadpoolctl.") from exc
    with threadpool_limits(limits=threads):
        ordinary = {
            direction: tuple(row for row in directional if row.direction == direction)
            for direction in ("0to1", "1to0")
        }
        permuted_all = permute_complete_case_feature_blocks(
            directional, seed=seed + int(target)
        )
        permuted = {
            direction: tuple(
                row for row in permuted_all if row.direction == direction
            )
            for direction in ("0to1", "1to0")
        }
        families = {
            "G": {
                direction: fit_directional_logit(
                    ordinary[direction], heldout_h=target, family="G"
                )
                for direction in ("0to1", "1to0")
            },
            "R": {
                direction: fit_directional_logit(
                    ordinary[direction], heldout_h=target, family="R"
                )
                for direction in ("0to1", "1to0")
            },
            "P": {
                direction: fit_directional_logit(
                    permuted[direction], heldout_h=target, family="P"
                )
                for direction in ("0to1", "1to0")
            },
        }
        single_model = fit_two_head_ridge(single_rows, heldout_h=target)
    return (
        target,
        _transport_families(families),
        single_model,
        canonical_hash([row.to_payload() for row in permuted_all]),
    )


def _transport_families(
    families: Mapping[str, Mapping[str, DirectionalLogitModel]],
) -> dict[str, dict[str, DirectionalLogitModel]]:
    """Strip read-only wrappers before crossing the spawned worker boundary."""

    return {
        str(family): dict(by_direction)
        for family, by_direction in families.items()
    }


def _freeze_received_families(
    families: Mapping[str, Mapping[str, DirectionalLogitModel]],
) -> Mapping[str, Mapping[str, DirectionalLogitModel]]:
    """Restore the immutable public model tree after worker deserialization."""

    return MappingProxyType(
        {
            str(family): MappingProxyType(dict(by_direction))
            for family, by_direction in families.items()
        }
    )


def _worker_initializer(threads: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(int(threads))


def _assert_model_runtime(config: object) -> None:
    runtime = getattr(config, "runtime")
    routing = getattr(config, "routing")
    if (
        int(runtime.get("model_workers", -1)) != 4
        or int(runtime.get("model_threads_per_worker", -1)) != 3
        or runtime.get("multiprocessing_start_method") != "spawn"
        or float(routing.get("feature_alpha", -1.0)) != 1.0
        or float(routing.get("source_alpha", -1.0)) != 4.0
        or float(routing.get("query_alpha", -1.0)) != 4.0
        or float(routing.get("intercept_alpha", -1.0)) != 0.25
    ):
        raise ProtocolError("Multi-challenger model runtime contract drifted.")


__all__ = ("fit_h_specific_donor_phase",)
