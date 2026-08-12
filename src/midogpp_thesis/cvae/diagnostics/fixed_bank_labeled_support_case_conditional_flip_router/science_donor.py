"""H-specific LOCO donor-model construction for the flip-router diagnostic."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
from types import MappingProxyType
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...routing.threshold_flip_case_router import (
    ContributionTarget,
    DonorRow,
    StaticSelection,
    TwoHeadRidgeModel,
    fit_two_head_ridge,
    refit_blocked_permutation_control,
    select_query_fixed_effect_static_source,
    select_static_source,
)
from .constants import B_ACTION_ID, CENTERS, a1_action_id, candidate_sources
from .hashing import canonical_hash
from .science_common import (
    _assert_science_config,
    _cases_for_center,
    _case_contribution,
    _core_feature,
    _feature_index,
    _label_index,
    _probability_index,
)
from .science_contracts import DonorPhaseResult


def fit_h_specific_donor_phase(
    *,
    probability_surface: object,
    prelabel: object,
    partition: object,
    manager: object,
    config: object,
) -> DonorPhaseResult:
    """Fit exactly one ordinary and one permuted donor model for each H."""

    _assert_science_config(config)
    probability = _probability_index(probability_surface)
    features = _feature_index(prelabel)
    all_target_rows: list[Mapping[str, object]] = []
    model_rows: list[Mapping[str, object]] = []
    seals: dict[str, Mapping[str, object]] = {}
    models: dict[str, TwoHeadRidgeModel] = {}
    permutation_models: dict[str, TwoHeadRidgeModel] = {}
    global_selections: dict[str, StaticSelection] = {}
    global_selection_fits = {}
    donor_batches: dict[str, tuple[DonorRow, ...]] = {}

    for heldout_h in CENTERS:
        labels = tuple(manager.open_loco_donor_labels(heldout_h))
        label_index = _label_index(labels)
        donors: list[DonorRow] = []
        for query in CENTERS:
            if query == heldout_h:
                continue
            for source in candidate_sources(query):
                if source == heldout_h:
                    continue
                action = a1_action_id(source)
                for case_id in _cases_for_center(partition, query):
                    target = _case_contribution(
                        probability,
                        label_index,
                        target_center=query,
                        case_id=case_id,
                        action_id=action,
                    )
                    feature = _core_feature(features[(query, case_id, action)])
                    donor = DonorRow(
                        model_target=heldout_h,
                        query_center=query,
                        candidate_source=source,
                        case_id=case_id,
                        action_id=action,
                        feature_case_id=case_id,
                        feature_names=feature.feature_names,
                        values=feature.values,
                        target=target,
                    )
                    donors.append(donor)
                    payload = {
                        "heldout_target_H": heldout_h,
                        "query_center_q": query,
                        "candidate_source_e": source,
                        "case_id": case_id,
                        "action_id": action,
                        "feature_hash": feature.feature_hash,
                        "delta_tp": target.delta_tp,
                        "delta_tn": target.delta_tn,
                        "n_positive": target.n_positive,
                        "n_negative": target.n_negative,
                    }
                    all_target_rows.append(
                        {**payload, "row_hash": canonical_hash(payload)}
                    )
        donor_batches[heldout_h] = tuple(donors)

    model_workers = int(getattr(config, "runtime").get("model_workers", 4))
    model_threads = int(getattr(config, "runtime").get("model_threads_per_worker", 3))
    if model_workers != 4 or model_threads != 3:
        raise ProtocolError("Flip-router model topology requires four workers and three threads.")
    jobs = tuple(
        (target, donor_batches[target], int(getattr(config, "protocol")["partition_seed"]))
        for target in CENTERS
    )
    with ProcessPoolExecutor(
        max_workers=model_workers,
        mp_context=mp.get_context("spawn"),
        initializer=_science_worker_initializer,
        initargs=(model_threads,),
    ) as executor:
        fitted = tuple(executor.map(_fit_donor_models_job, jobs))

    for heldout_h, model, permutation in fitted:
        if model.model_hash == permutation.model_hash:
            raise ProtocolError("Permutation control reproduced the ordinary model.")
        global_selection_fit = select_query_fixed_effect_static_source(
            donor_batches[heldout_h], heldout_h=heldout_h
        )
        global_selection = global_selection_fit.selection
        global_selections[heldout_h] = global_selection
        global_selection_fits[heldout_h] = global_selection_fit
        seal_payload = {
            "schema_version": "fixed_bank_flip_router_H_model_seal_v1",
            "heldout_target_H": heldout_h,
            "model_hash": model.model_hash,
            "model_provenance_hash": model.provenance_hash,
            "permutation_model_hash": permutation.model_hash,
            "permutation_provenance_hash": permutation.provenance_hash,
            "global_static_selection": global_selection.to_payload(),
            "global_static_query_fixed_effect_fit": global_selection_fit.to_payload(),
            "donor_row_count": len(donor_batches[heldout_h]),
            "donor_query_centers": list(model.donor_query_centers),
            "donor_candidate_sources": list(model.donor_candidate_sources),
            "strict_H_q_e_exclusion": True,
            "heldout_H_labels_used": False,
        }
        seal = {**seal_payload, "seal_hash": canonical_hash(seal_payload)}
        manager.record_H_specific_donor_model_seal(
            heldout_h,
            model_heldout_target=model.model_target,
            model_hash=model.model_hash,
            provenance_hash=model.provenance_hash,
        )
        models[heldout_h] = model
        permutation_models[heldout_h] = permutation
        seals[heldout_h] = seal
        model_rows.append(
            {
                "heldout_target_H": heldout_h,
                "ordinary_model": model.to_payload(),
                "permutation_model": permutation.to_payload(),
                "global_static_selection": global_selection.to_payload(),
                "global_static_query_fixed_effect_fit": global_selection_fit.to_payload(),
                "model_seal_hash": seal["seal_hash"],
            }
        )

    permutation_unhashed = {
        "schema_version": "fixed_bank_flip_router_permutation_provenance_v1",
        "seed": int(getattr(config, "protocol")["partition_seed"]),
        "scope": "whole_case_within_H_and_query_q",
        "same_capacity_refit": True,
        "label_targets_permuted": False,
        "feature_case_blocks_deranged": True,
        "model_hashes_by_target": {
            target: permutation_models[target].model_hash for target in CENTERS
        },
    }
    permutation_payload = {
        **permutation_unhashed,
        "permutation_provenance_hash": canonical_hash(permutation_unhashed),
    }
    return DonorPhaseResult(
        contribution_targets=tuple(all_target_rows),
        models=tuple(model_rows),
        seals=MappingProxyType(seals),
        permutation_payload=MappingProxyType(permutation_payload),
        model_by_target=MappingProxyType(models),
        permutation_model_by_target=MappingProxyType(permutation_models),
        global_selection_by_target=MappingProxyType(global_selections),
        global_selection_fit_by_target=MappingProxyType(global_selection_fits),
    )


def _science_worker_initializer(threads: int) -> None:
    """Bind each spawned model worker to the workstation BLAS budget."""

    value = str(int(threads))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = value


def _fit_donor_models_job(
    job: tuple[str, tuple[DonorRow, ...], int],
) -> tuple[str, TwoHeadRidgeModel, TwoHeadRidgeModel]:
    """Spawn-safe numerical fit for one held-out target H."""

    heldout_h, donors, seed = job
    try:
        from threadpoolctl import threadpool_limits
    except ModuleNotFoundError as exc:  # pragma: no cover - workstation dependency
        raise RuntimeError("Flip-router donor fitting requires threadpoolctl.") from exc
    with threadpool_limits(limits=3):
        model = fit_two_head_ridge(donors, heldout_h=heldout_h)
        permutation = refit_blocked_permutation_control(
            donors,
            heldout_h=heldout_h,
            seed=seed,
        )
    return heldout_h, model, permutation

def _safe_static_selection(
    action_targets: Mapping[str, Sequence[ContributionTarget]],
) -> StaticSelection:
    """Fail to B when the selection pool cannot identify pooled BACC."""

    rows = tuple(row for values in action_targets.values() for row in values)
    if not rows:
        return StaticSelection(B_ACTION_ID, 0.0, 0.0, True)
    # Every action has identical support cases/class totals.  Inspect the first
    # action only so candidates are not implicitly reweighted by menu size.
    first = tuple(next(iter(action_targets.values())))
    if (
        sum(row.n_positive for row in first) <= 0
        or sum(row.n_negative for row in first) <= 0
    ):
        return StaticSelection(B_ACTION_ID, 0.0, 0.0, True)
    return select_static_source(action_targets)
