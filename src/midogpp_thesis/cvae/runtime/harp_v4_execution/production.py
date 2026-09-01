"""Concrete source-only HARP v4 production pipeline."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import math
import multiprocessing as mp
from pathlib import Path
import struct
from typing import Any

import numpy as np

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from ...routing.harp_v4 import (
    ActionKind as CoreActionKind,
    CASE_CONTRIBUTION_METRIC_NAME,
    CaseActionSet,
    CaseTargetAction,
    CaseTrainingObservation,
    Comparison,
    EffectVector,
    OUTCOME_NAMES,
    PRIMARY_ESTIMAND,
    PRIMARY_METRIC_NAME,
    SINGLE_CLASS_CASE_RULE,
    PolicyConfig,
    aggregate_case_equal_metrics,
    case_class_support_counts,
    case_effects,
    case_metrics,
    fit_harp_v4,
    route_case,
)
from ...routing.harp_v4.serialization import (
    decision_to_payload,
    fit_collection_from_payload,
    fit_collection_to_payload,
)
from .contracts import (
    ActionKind,
    ArtifactValue,
    LabelFreeActionBlock,
    LabelFreeOuterMenu,
    PrelabelRouteSet,
    RoutedCase,
    TerminalEvaluation,
    array_bytes_sha256,
)
from .physical import (
    build_physical_plan,
    materialize_physical_outer_menus,
    validate_physical_inputs,
)
from .workstation import inspect_harp_v4_workstation


FEATURE_NAMES = (
    "b_mean",
    "u_mean",
    "action_mean",
    "u_minus_b_mean",
    "action_minus_b_mean",
    "action_minus_u_mean",
    "action_minus_b_abs_mean",
    "action_minus_u_abs_mean",
    "action_minus_b_std",
    "action_minus_u_std",
    "b_abs_margin_mean",
    "u_abs_margin_mean",
    "action_abs_margin_mean",
    "b_u_crossing_fraction",
    "u_action_crossing_fraction",
    "b_action_crossing_fraction",
    "case_size_log1p",
)


def _model_parameters(config: object) -> tuple[tuple[float, ...], float, float, PolicyConfig]:
    model = getattr(config, "model")
    if not isinstance(model, Mapping):
        raise ProtocolError("HARP v4 strict model configuration is absent.")
    try:
        alpha_grid = tuple(float(value) for value in model["alpha_grid"])
        residual_quantile = float(model["residual_quantile"])
        geometry_quantile = float(model["geometry_quantile"])
        policy_raw = model["policy"]
        if not isinstance(policy_raw, Mapping):
            raise TypeError
        policy = PolicyConfig(**dict(policy_raw))
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("HARP v4 fitted-policy parameters are malformed.") from exc
    return alpha_grid, residual_quantile, geometry_quantile, policy


def _case_indices(case_ids: Sequence[str]) -> tuple[tuple[str, np.ndarray], ...]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for ordinal, case in enumerate(case_ids):
        grouped[str(case)].append(ordinal)
    return tuple(
        (case, np.asarray(ordinals, dtype=np.int64))
        for case, ordinals in sorted(grouped.items())
    )


def _features(
    baseline: np.ndarray, uniform: np.ndarray, action: np.ndarray
) -> tuple[float, ...]:
    b = np.asarray(baseline, dtype=np.float64)
    u = np.asarray(uniform, dtype=np.float64)
    a = np.asarray(action, dtype=np.float64)
    if b.shape != u.shape or b.shape != a.shape or b.ndim != 1 or not len(b):
        raise ProtocolError("HARP v4 case feature probabilities are misaligned.")
    ub, ab, au = u - b, a - b, a - u
    crossing = lambda left, right: float(np.mean((left >= 0.5) != (right >= 0.5)))
    values = (
        float(np.mean(b, dtype=np.float64)),
        float(np.mean(u, dtype=np.float64)),
        float(np.mean(a, dtype=np.float64)),
        float(np.mean(ub, dtype=np.float64)),
        float(np.mean(ab, dtype=np.float64)),
        float(np.mean(au, dtype=np.float64)),
        float(np.mean(np.abs(ab), dtype=np.float64)),
        float(np.mean(np.abs(au), dtype=np.float64)),
        float(np.std(ab, dtype=np.float64)),
        float(np.std(au, dtype=np.float64)),
        float(np.mean(np.abs(b - 0.5), dtype=np.float64)),
        float(np.mean(np.abs(u - 0.5), dtype=np.float64)),
        float(np.mean(np.abs(a - 0.5), dtype=np.float64)),
        crossing(b, u),
        crossing(u, a),
        crossing(b, a),
        math.log1p(len(b)),
    )
    if len(values) != len(FEATURE_NAMES) or not all(math.isfinite(value) for value in values):
        raise ProtocolError("HARP v4 case feature construction drifted.")
    return values


def _effects(
    reference: np.ndarray,
    action: np.ndarray,
    truth: np.ndarray,
    *,
    total_case_count: int,
    class_support_case_counts: tuple[int, int],
) -> EffectVector:
    return case_effects(
        reference,
        action,
        truth,
        total_case_count=total_case_count,
        class_support_case_counts=class_support_case_counts,
    )


def _float32_bytes(values: np.ndarray) -> tuple[bytes, ...]:
    raw = np.ascontiguousarray(values, dtype=np.float32)
    return tuple(raw[index:index + 1].tobytes(order="C") for index in range(len(raw)))


def _decode_float32(values: Sequence[bytes]) -> np.ndarray:
    if not values or any(type(value) is not bytes or len(value) != 4 for value in values):
        raise ProtocolError("HARP v4 physical probability bytes are not float32.")
    return np.frombuffer(b"".join(values), dtype=np.dtype("<f4")).copy()


def _fit_worker(
    payload: tuple[
        str,
        tuple[CaseTrainingObservation, ...],
        tuple[float, ...],
        float,
        float,
        int,
    ]
) -> object:
    outer, rows, alphas, residual, geometry, threads = payload
    if type(threads) is not int or threads <= 0:
        raise ProtocolError("HARP v4 fit-worker BLAS limit is malformed.")
    try:
        from threadpoolctl import threadpool_limits
    except ModuleNotFoundError as exc:  # pragma: no cover - workstation dependency
        raise RuntimeError("HARP v4 fit workers require threadpoolctl.") from exc
    # The process topology is part of the sealed workstation contract.  Set
    # the native-library limit around the complete nested LODO fit so four
    # spawned workers cannot silently oversubscribe the twelve assigned CPU
    # threads through OpenBLAS/MKL defaults inherited from the login shell.
    with threadpool_limits(limits=threads):
        return fit_harp_v4(
            rows,
            outer_target_id=outer,
            alpha_grid=alphas,
            residual_quantile=residual,
            geometry_quantile=geometry,
        )


class HarpV4ProductionPipeline:
    """Concrete workstation pipeline over immutable v4 inputs and core."""

    def __init__(self, *, development_role: str, evaluation_role: str) -> None:
        self._development_role = str(development_role)
        self._evaluation_role = str(evaluation_role)
        self._last_menus: tuple[LabelFreeOuterMenu, ...] = ()

    def preflight(self, config: object, cache: object) -> Mapping[str, object]:
        receipt = validate_physical_inputs(config, cache)
        plan = build_physical_plan()
        _model_parameters(config)
        live = dict(inspect_harp_v4_workstation(getattr(config, "runtime")))
        return {
            **live,
            "schema_version": "midogpp_harp_v4_workstation_preflight_v2",
            "physical_input_receipt": receipt.public_payload(),
            "physical_plan": plan,
        }

    def materialize_label_free_outer_menus(
        self,
        config: object,
        cache: object,
        *,
        outer_targets: Sequence[str],
        scratch_root: Path,
    ) -> Sequence[LabelFreeOuterMenu]:
        if tuple(outer_targets) != tuple(getattr(config, "protocol")["centers"]):
            raise ProtocolError("HARP v4 production target universe drifted.")
        menus = materialize_physical_outer_menus(
            config,
            cache,
            scratch_root=Path(scratch_root),
            development_role=self._development_role,
            evaluation_role=self._evaluation_role,
        )
        self._last_menus = tuple(menus)
        return menus

    def build_development_case_surface(
        self,
        menus: Sequence[LabelFreeOuterMenu],
        development_labels: object,
        *,
        config: object,
    ) -> ArtifactValue:
        label_rows = tuple(development_labels)
        label_index = {
            (str(row.center), str(row.case_id), str(row.sample_id)): int(row.label)
            for row in label_rows
        }
        if len(label_index) != len(label_rows):
            raise ProtocolError("HARP v4 development labels duplicate sample identities.")
        observations: list[CaseTrainingObservation] = []
        for menu in menus:
            contexts: dict[str, list[LabelFreeActionBlock]] = defaultdict(list)
            for block in menu.blocks:
                if block.surface_role == "development":
                    contexts[block.query_center_id].append(block)
            for query, blocks in sorted(contexts.items()):
                baseline = next(block for block in blocks if block.action_kind is ActionKind.B)
                uniform = next(block for block in blocks if block.action_kind is ActionKind.U)
                experts = tuple(
                    sorted(
                        (block for block in blocks if block.action_kind is ActionKind.HXE),
                        key=lambda block: block.selected_source_id or "",
                    )
                )
                case_rows = []
                for case_id, indices in _case_indices(baseline.case_ids):
                    sample_ids = tuple(baseline.sample_ids[int(index)] for index in indices)
                    truth = np.asarray(
                        [label_index[(query, case_id, sample)] for sample in sample_ids],
                        dtype=np.int64,
                    )
                    case_rows.append((case_id, indices, sample_ids, truth))
                total_case_count = len(case_rows)
                support_counts = case_class_support_counts(
                    tuple(row[3] for row in case_rows)
                )
                effect_normalization = {
                    "total_case_count": total_case_count,
                    "class_support_case_counts": support_counts,
                }
                for case_id, indices, _sample_ids, truth in case_rows:
                    b = baseline.probabilities[indices]
                    u = uniform.probabilities[indices]
                    class_counts = (int(np.sum(truth == 0)), int(np.sum(truth == 1)))
                    observations.append(
                        CaseTrainingObservation(
                            outer_target_id=menu.outer_target_id,
                            pseudo_query_id=query,
                            candidate_source_id=None,
                            case_id=case_id,
                            comparison=Comparison.U_VS_B,
                            feature_names=FEATURE_NAMES,
                            feature_values=_features(b, u, u),
                            effects=_effects(b, u, truth, **effect_normalization),
                            class_counts=class_counts,
                            pseudo_query_case_count=total_case_count,
                            pseudo_query_class_support_case_counts=support_counts,
                        )
                    )
                    for expert in experts:
                        h = expert.probabilities[indices]
                        common = {
                            "outer_target_id": menu.outer_target_id,
                            "pseudo_query_id": query,
                            "candidate_source_id": expert.selected_source_id,
                            "case_id": case_id,
                            "feature_names": FEATURE_NAMES,
                            "feature_values": _features(b, u, h),
                            "class_counts": class_counts,
                            "pseudo_query_case_count": total_case_count,
                            "pseudo_query_class_support_case_counts": support_counts,
                        }
                        observations.append(
                            CaseTrainingObservation(
                                **common,
                                comparison=Comparison.HXE_VS_B,
                                effects=_effects(b, h, truth, **effect_normalization),
                            )
                        )
                        observations.append(
                            CaseTrainingObservation(
                                **common,
                                comparison=Comparison.HXE_VS_U,
                                effects=_effects(u, h, truth, **effect_normalization),
                            )
                        )
        ordered = tuple(sorted(observations, key=lambda row: row.row_key))
        manifest = {
            "schema_version": "midogpp_harp_v4_development_case_surface_v2",
            "observation_count": len(ordered),
            "outer_targets": sorted({row.outer_target_id for row in ordered}),
            "feature_names": list(FEATURE_NAMES),
            "comparison_counts": dict(sorted(Counter(row.comparison.value for row in ordered).items())),
            "strict_outer_target_exclusion": all(
                row.outer_target_id not in {row.pseudo_query_id, row.candidate_source_id}
                for row in ordered
            ),
            "decision_unit": "case",
            "joint_endpoint_names": list(OUTCOME_NAMES),
            "primary_estimand": PRIMARY_ESTIMAND,
            "single_class_case_rule": SINGLE_CLASS_CASE_RULE,
            "evaluation_labels_used": False,
            "rows": [
                {
                    "outer_target_id": row.outer_target_id,
                    "pseudo_query_id": row.pseudo_query_id,
                    "candidate_source_id": row.candidate_source_id,
                    "case_id": row.case_id,
                    "comparison": row.comparison.value,
                    "pseudo_query_case_count": row.pseudo_query_case_count,
                    "pseudo_query_class_support_case_counts": list(
                        row.pseudo_query_class_support_case_counts
                    ),
                }
                for row in ordered
            ],
        }
        arrays = {
            "feature_values": np.asarray(
                [row.feature_values for row in ordered], dtype=np.float64
            ),
            "effects": np.asarray(
                [row.effects.as_tuple() for row in ordered], dtype=np.float64
            ),
            "class_counts": np.asarray(
                [row.class_counts for row in ordered], dtype=np.int64
            ),
            "pseudo_query_case_count": np.asarray(
                [row.pseudo_query_case_count for row in ordered], dtype=np.int64
            ),
            "pseudo_query_class_support_case_counts": np.asarray(
                [row.pseudo_query_class_support_case_counts for row in ordered],
                dtype=np.int64,
            ),
        }
        return ArtifactValue(
            state=ordered,
            manifest={**manifest, "surface_hash": canonical_hash(manifest)},
            arrays=arrays,
        )

    def fit_source_only_model(
        self, development: ArtifactValue, *, config: object
    ) -> ArtifactValue:
        rows = tuple(development.state)
        alphas, residual, geometry, policy = _model_parameters(config)
        centers = tuple(getattr(config, "protocol")["centers"])
        payloads = tuple(
            (
                outer,
                tuple(row for row in rows if row.outer_target_id == outer),
                alphas,
                residual,
                geometry,
                int(getattr(config, "runtime")["blas_threads_per_worker"]),
            )
            for outer in centers
        )
        workers = int(getattr(config, "runtime")["cpu_fit_workers"])
        if workers == 1:
            fits = tuple(_fit_worker(payload) for payload in payloads)
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
                fits = tuple(pool.map(_fit_worker, payloads))
        payload = fit_collection_to_payload(fits)
        manifest = {
            "schema_version": "midogpp_harp_v4_source_only_model_lock_v2",
            "fit_collection": payload,
            "development_surface_hash": development.manifest["surface_hash"],
            "alpha_grid": list(alphas),
            "residual_quantile": residual,
            "geometry_quantile": geometry,
            "policy": asdict(policy),
            "strict_outer_target_exclusion": True,
            "nested_source_center_lodo": True,
            "delete_donor_ensemble": True,
            "shared_multi_output_rhs": True,
            "joint_endpoint_names": list(OUTCOME_NAMES),
            "primary_estimand": PRIMARY_ESTIMAND,
            "policy_gain_threshold_units": CASE_CONTRIBUTION_METRIC_NAME,
            "residual_calibration_weighting": "equal_donor_then_equal_case",
            "evaluation_labels_used": False,
        }
        model_hash = canonical_hash(manifest)
        return ArtifactValue(
            state=fits,
            manifest={**manifest, "model_hash": model_hash},
        )

    def build_complete_target_case_actions(
        self,
        menus: Sequence[LabelFreeOuterMenu],
        fit: ArtifactValue,
        *,
        config: object,
    ) -> ArtifactValue:
        sets: list[CaseActionSet] = []
        rows = []
        feature_rows: list[tuple[float, ...]] = []
        probability_rows: list[np.ndarray] = []
        probability_offsets = [0]
        for menu in menus:
            baseline = menu.target_block(ActionKind.B)
            uniform = menu.target_block(ActionKind.U)
            experts = tuple(
                menu.target_block(ActionKind.HXE, source)
                for source in sorted(
                    block.selected_source_id
                    for block in menu.blocks
                    if block.surface_role == "target" and block.action_kind is ActionKind.HXE
                )
                if source is not None
            )
            for case_id, indices in _case_indices(baseline.case_ids):
                sample_ids = tuple(baseline.sample_ids[int(index)] for index in indices)
                b = baseline.probabilities[indices]
                u = uniform.probabilities[indices]
                base_action = CaseTargetAction(
                    outer_target_id=menu.outer_target_id,
                    target_query_id=menu.outer_target_id,
                    case_id=case_id,
                    action_kind=CoreActionKind.B,
                    candidate_source_id=None,
                    feature_names=FEATURE_NAMES,
                    feature_values=_features(b, u, b),
                    sample_ids=sample_ids,
                    probability_bytes=_float32_bytes(b),
                    prediction_seal_hash=menu.menu_hash,
                    expert_weight=0.0,
                )
                uniform_action = CaseTargetAction(
                    outer_target_id=menu.outer_target_id,
                    target_query_id=menu.outer_target_id,
                    case_id=case_id,
                    action_kind=CoreActionKind.U,
                    candidate_source_id=None,
                    feature_names=FEATURE_NAMES,
                    feature_values=_features(b, u, u),
                    sample_ids=sample_ids,
                    probability_bytes=_float32_bytes(u),
                    prediction_seal_hash=menu.menu_hash,
                    expert_weight=0.0,
                )
                expert_actions = tuple(
                    CaseTargetAction(
                        outer_target_id=menu.outer_target_id,
                        target_query_id=menu.outer_target_id,
                        case_id=case_id,
                        action_kind=CoreActionKind.HXE,
                        candidate_source_id=expert.selected_source_id,
                        feature_names=FEATURE_NAMES,
                        feature_values=_features(b, u, expert.probabilities[indices]),
                        sample_ids=sample_ids,
                        probability_bytes=_float32_bytes(expert.probabilities[indices]),
                        prediction_seal_hash=menu.menu_hash,
                        expert_weight=1.0,
                    )
                    for expert in experts
                )
                expected_sources = tuple(
                    center
                    for center in tuple(getattr(config, "protocol")["centers"])
                    if center != menu.outer_target_id
                )
                observed_sources = tuple(
                    action.candidate_source_id for action in expert_actions
                )
                if observed_sources != expected_sources:
                    raise ProtocolError(
                        "HARP v4 target case lacks the sealed complete expert universe."
                    )
                action_set = CaseActionSet(
                    baseline=base_action,
                    uniform=uniform_action,
                    experts=expert_actions,
                    expected_candidate_source_ids=expected_sources,
                )
                sets.append(action_set)
                rows.extend(
                    {
                        "outer_target_id": action.outer_target_id,
                        "case_id": action.case_id,
                        "action_id": action.action_id,
                        "action_kind": action.action_kind.value,
                        "candidate_source_id": action.candidate_source_id,
                        "sample_ids": list(action.sample_ids),
                        "prediction_seal_hash": action.prediction_seal_hash,
                        "expert_weight": action.expert_weight,
                        "expected_candidate_source_ids": list(expected_sources),
                        "sample_count": len(action.sample_ids),
                        "probability_bytes_sha256": hashlib_sha256_bytes(action.probability_bytes),
                    }
                    for action in (base_action, uniform_action, *expert_actions)
                )
                for action in (base_action, uniform_action, *expert_actions):
                    feature_rows.append(action.feature_values)
                    probability = _decode_float32(action.probability_bytes)
                    probability_rows.append(probability)
                    probability_offsets.append(
                        probability_offsets[-1] + len(probability)
                    )
        sets.sort(key=lambda value: value.baseline.case_key)
        manifest = {
            "schema_version": "midogpp_harp_v4_complete_target_case_actions_v1",
            "rows": rows,
            "feature_names": list(FEATURE_NAMES),
            "case_count": len(sets),
            "action_count": len(rows),
            "complete_B_U_Hxe": True,
            "physical_expert_weight": 1.0,
            "case_level": True,
            "evaluation_labels_used": False,
        }
        return ArtifactValue(
            state=tuple(sets),
            manifest={**manifest, "target_action_hash": canonical_hash(manifest)},
            arrays={
                "feature_values": np.asarray(feature_rows, dtype=np.float64),
                "probabilities": np.ascontiguousarray(
                    np.concatenate(probability_rows), dtype=np.float32
                ),
                "probability_offsets": np.asarray(
                    probability_offsets, dtype=np.int64
                ),
            },
        )

    def route_case_actions(
        self,
        menus: Sequence[LabelFreeOuterMenu],
        target_actions: ArtifactValue,
        fit: ArtifactValue,
        *,
        config: object,
    ) -> PrelabelRouteSet:
        fits = {value.outer_target_id: value for value in tuple(fit.state)}
        _, _, _, policy = _model_parameters(config)
        routed: list[RoutedCase] = []
        decision_payloads = []
        for actions in tuple(target_actions.state):
            decision = route_case(
                actions, fits[actions.baseline.outer_target_id], config=policy
            )
            payload = decision_to_payload(decision)
            decision_payloads.append(payload)
            selected_action = (
                actions.baseline
                if decision.selected_kind is CoreActionKind.B
                else actions.uniform
                if decision.selected_kind is CoreActionKind.U
                else next(
                    value
                    for value in actions.experts
                    if value.candidate_source_id == decision.selected_source_id
                )
            )
            runtime_kind = (
                ActionKind.HXE
                if decision.selected_kind is CoreActionKind.HXE
                else ActionKind(decision.selected_kind.value)
            )
            baseline = _decode_float32(actions.baseline.probability_bytes)
            uniform = _decode_float32(actions.uniform.probability_bytes)
            selected = _decode_float32(selected_action.probability_bytes)
            output = _decode_float32(decision.output_probability_bytes)
            routed.append(
                RoutedCase(
                    outer_target_id=decision.outer_target_id,
                    case_id=decision.case_id,
                    sample_ids=decision.sample_ids,
                    selected_kind=runtime_kind,
                    selected_source_id=decision.selected_source_id,
                    reason=decision.reason,
                    baseline_probabilities=baseline,
                    uniform_probabilities=uniform,
                    selected_probabilities=selected,
                    routed_probabilities=output,
                    decision_payload=payload,
                )
            )
        model_hash = str(fit.manifest["model_hash"])
        target_hash = str(target_actions.manifest["target_action_hash"])
        policy_hash = canonical_hash(
            {
                "schema_version": "midogpp_harp_v4_frozen_hierarchical_policy_v1",
                "model_hash": model_hash,
                "target_action_hash": target_hash,
                "policy": asdict(policy),
                "decisions": decision_payloads,
                "evaluation_labels_used": False,
            }
        )
        return PrelabelRouteSet(
            cases=tuple(sorted(routed, key=lambda value: (value.outer_target_id, value.case_id))),
            policy_hash=policy_hash,
            model_hash=model_hash,
            target_action_hash=target_hash,
        )

    def evaluate_terminal(
        self,
        routes: PrelabelRouteSet,
        evaluation_truth: object,
        *,
        config: object,
    ) -> TerminalEvaluation:
        if not isinstance(evaluation_truth, Mapping):
            raise ProtocolError("HARP v4 evaluation truth must be a role-scoped mapping.")
        truth = {tuple(key): int(value) for key, value in evaluation_truth.items()}
        by_center: dict[str, list[dict[str, object]]] = defaultdict(list)
        reasons: Counter[str] = Counter()
        exact_fallback = True
        for case in routes.cases:
            labels = np.asarray(
                [truth[(case.outer_target_id, case.case_id, sample)] for sample in case.sample_ids],
                dtype=np.int64,
            )
            baseline = case.baseline_probabilities.astype(np.float64)
            uniform = case.uniform_probabilities.astype(np.float64)
            routed = case.routed_probabilities.astype(np.float64)
            by_center[case.outer_target_id].append(
                {
                    "case_id": case.case_id,
                    "labels": labels,
                    "baseline": baseline,
                    "uniform": uniform,
                    "routed": routed,
                    "selected_kind": case.selected_kind.value,
                }
            )
            reasons[case.reason] += 1
            if case.selected_kind is ActionKind.B:
                exact_fallback &= (
                    case.routed_probabilities.tobytes(order="C")
                    == case.baseline_probabilities.tobytes(order="C")
                )
        if set(truth) != {
            (case.outer_target_id, case.case_id, sample)
            for case in routes.cases
            for sample in case.sample_ids
        }:
            raise ProtocolError("HARP v4 evaluation labels do not exactly cover sealed routes.")
        center_metrics = {
            center: {
                role: _center_metrics(rows, role)
                for role in ("baseline", "uniform", "routed")
            }
            for center, rows in sorted(by_center.items())
        }
        aggregate = {
            role: {
                metric: float(
                    np.mean(
                        [center_metrics[center][role][metric] for center in sorted(center_metrics)],
                        dtype=np.float64,
                    )
                )
                for metric in (PRIMARY_METRIC_NAME, "brier", "log_loss")
            }
            for role in ("baseline", "uniform", "routed")
        }
        route_count = sum(case.selected_kind is not ActionKind.B for case in routes.cases)
        expert_count = sum(case.selected_kind is ActionKind.HXE for case in routes.cases)
        metrics = {
            "schema_version": "midogpp_harp_v4_terminal_result_v2",
            "status": "TERMINAL_POST_HOC_CONSUMED_TEST_SENSITIVITY",
            "case_count": len(routes.cases),
            "row_count": sum(len(case.sample_ids) for case in routes.cases),
            "routed_case_count": route_count,
            "expert_routed_case_count": expert_count,
            "case_route_rate": route_count / len(routes.cases),
            "equal_center_metrics": aggregate,
            "center_metrics": center_metrics,
            "primary_estimand": PRIMARY_ESTIMAND,
            "primary_metric_name": PRIMARY_METRIC_NAME,
            "single_class_case_rule": SINGLE_CLASS_CASE_RULE,
            "fit_policy_terminal_estimand_aligned": True,
            "exact_b_fallback_byte_identity": exact_fallback,
            "utility_kind": "downstream_classifier_utility_not_NELBO",
            "routing_stage_compatibility_estimated": False,
            "generative_expert_compatibility_claimed": False,
            "publication_status": "POST_HOC_CONSUMED_TEST_SENSITIVITY",
            "terminal_decision": "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE",
            "fresh_evidence": False,
        }
        metrics = {**metrics, "result_hash": canonical_hash(metrics)}
        oracle = self._oracle_diagnostic(truth)
        route_reasons = {
            "schema_version": "midogpp_harp_v4_route_reason_summary_v1",
            "reason_counts": dict(sorted(reasons.items())),
            "selected_action_counts": dict(
                sorted(Counter(case.selected_kind.value for case in routes.cases).items())
            ),
            "exact_b_fallback_byte_identity": exact_fallback,
        }
        return TerminalEvaluation(metrics, oracle, route_reasons)

    def _oracle_diagnostic(
        self, truth: Mapping[tuple[str, str, str], int]
    ) -> Mapping[str, object]:
        rows = []
        for menu in self._last_menus:
            baseline = menu.target_block(ActionKind.B)
            candidates = tuple(
                block for block in menu.blocks if block.surface_role == "target"
            )
            case_rows = []
            for case_id, indices in _case_indices(baseline.case_ids):
                sample_ids = tuple(baseline.sample_ids[int(index)] for index in indices)
                labels = np.asarray(
                    [truth[(menu.outer_target_id, case_id, sample)] for sample in sample_ids],
                    dtype=np.int64,
                )
                case_rows.append((case_id, indices, labels))
            support_counts = case_class_support_counts(
                tuple(row[2] for row in case_rows)
            )
            metric_normalization = {
                "total_case_count": len(case_rows),
                "class_support_case_counts": support_counts,
            }
            for case_id, indices, labels in case_rows:
                scored = []
                for block in candidates:
                    probabilities = block.probabilities[indices].astype(np.float64)
                    score = _case_metric(
                        probabilities, labels, **metric_normalization
                    )
                    scored.append((score, block))
                best_score, best = min(
                    scored,
                    key=lambda value: (
                        -value[0][CASE_CONTRIBUTION_METRIC_NAME],
                        value[0]["brier"],
                        value[0]["log_loss"],
                        value[1].action_kind.value,
                        value[1].selected_source_id or "",
                    ),
                )
                base_score = _case_metric(
                    baseline.probabilities[indices], labels, **metric_normalization
                )
                rows.append(
                    {
                        "outer_target_id": menu.outer_target_id,
                        "case_id": case_id,
                        "oracle_action": best.action_kind.value,
                        "oracle_source": best.selected_source_id,
                        "oracle_case_equal_bacc_contribution_gain_vs_B": best_score[
                            CASE_CONTRIBUTION_METRIC_NAME
                        ]
                        - base_score[CASE_CONTRIBUTION_METRIC_NAME],
                    }
                )
        centers = tuple(sorted({str(row["outer_target_id"]) for row in rows}))
        center_gains = {
            center: float(
                np.mean(
                    [
                        row["oracle_case_equal_bacc_contribution_gain_vs_B"]
                        for row in rows
                        if row["outer_target_id"] == center
                    ],
                    dtype=np.float64,
                )
            )
            for center in centers
        }
        body = {
            "schema_version": "midogpp_harp_v4_terminal_action_oracle_diagnostic_v2",
            "rows": rows,
            "case_count": len(rows),
            "center_oracle_case_equal_bacc_gain_vs_B": center_gains,
            "equal_center_mean_oracle_case_equal_bacc_gain_vs_B": float(
                np.mean(tuple(center_gains.values()), dtype=np.float64)
            ),
            "primary_estimand": PRIMARY_ESTIMAND,
            "single_class_case_rule": SINGLE_CLASS_CASE_RULE,
            "opened_after_frozen_route_seal": True,
            "may_feed_policy_or_thresholds": False,
            "diagnostic_only": True,
        }
        return {**body, "diagnostic_hash": canonical_hash(body)}


def hashlib_sha256_bytes(values: Sequence[bytes]) -> str:
    import hashlib

    return hashlib.sha256(b"".join(values)).hexdigest()


def development_observations_from_artifact(
    value: ArtifactValue,
) -> tuple[CaseTrainingObservation, ...]:
    """Reconstruct all source-development rows from the compact disk projection."""

    rows = value.manifest.get("rows")
    names = tuple(str(name) for name in value.manifest.get("feature_names", ()))
    features = np.asarray(value.arrays.get("feature_values"))
    effects = np.asarray(value.arrays.get("effects"))
    counts = np.asarray(value.arrays.get("class_counts"))
    query_case_counts = np.asarray(
        value.arrays.get("pseudo_query_case_count")
    )
    query_class_support = np.asarray(
        value.arrays.get("pseudo_query_class_support_case_counts")
    )
    if (
        not isinstance(rows, list)
        or features.shape != (len(rows), len(names))
        or effects.shape != (len(rows), 3)
        or counts.shape != (len(rows), 2)
        or query_case_counts.shape != (len(rows),)
        or query_class_support.shape != (len(rows), 2)
        or features.dtype != np.float64
        or effects.dtype != np.float64
        or counts.dtype != np.int64
        or query_case_counts.dtype != np.int64
        or query_class_support.dtype != np.int64
    ):
        raise ProtocolError("HARP v4 compact development surface geometry drifted.")
    observations = tuple(
        CaseTrainingObservation(
            outer_target_id=str(raw["outer_target_id"]),
            pseudo_query_id=str(raw["pseudo_query_id"]),
            candidate_source_id=(
                None
                if raw.get("candidate_source_id") is None
                else str(raw["candidate_source_id"])
            ),
            case_id=str(raw["case_id"]),
            comparison=Comparison(str(raw["comparison"])),
            feature_names=names,
            feature_values=tuple(float(item) for item in features[index]),
            effects=EffectVector(*tuple(float(item) for item in effects[index])),
            class_counts=tuple(int(item) for item in counts[index]),
            pseudo_query_case_count=int(query_case_counts[index]),
            pseudo_query_class_support_case_counts=tuple(
                int(item) for item in query_class_support[index]
            ),
        )
        for index, raw in enumerate(rows)
        if isinstance(raw, Mapping)
    )
    if len(observations) != len(rows) or tuple(sorted(observations, key=lambda row: row.row_key)) != observations:
        raise ProtocolError("HARP v4 compact development row order drifted.")
    return observations


def fit_collection_from_artifact(value: ArtifactValue) -> tuple[object, ...]:
    payload = value.manifest.get("fit_collection")
    if not isinstance(payload, Mapping):
        raise ProtocolError("HARP v4 compact model lacks its fit collection.")
    fits = fit_collection_from_payload(payload)
    if fit_collection_to_payload(fits) != dict(payload):
        raise ProtocolError("HARP v4 fit collection failed durable reconstruction.")
    return fits


def target_action_sets_from_artifact(
    value: ArtifactValue,
) -> tuple[CaseActionSet, ...]:
    """Reconstruct complete physical case slates without any in-memory state."""

    rows = value.manifest.get("rows")
    names = tuple(str(name) for name in value.manifest.get("feature_names", ()))
    features = np.asarray(value.arrays.get("feature_values"))
    probabilities = np.asarray(value.arrays.get("probabilities"))
    offsets = np.asarray(value.arrays.get("probability_offsets"))
    if (
        not isinstance(rows, list)
        or features.shape != (len(rows), len(names))
        or features.dtype != np.float64
        or probabilities.dtype != np.float32
        or probabilities.ndim != 1
        or offsets.dtype != np.int64
        or offsets.shape != (len(rows) + 1,)
        or offsets[0] != 0
        or offsets[-1] != len(probabilities)
        or np.any(np.diff(offsets) <= 0)
    ):
        raise ProtocolError("HARP v4 compact target-action geometry drifted.")
    actions: list[CaseTargetAction] = []
    expected_by_case: dict[tuple[str, str], tuple[str, ...]] = {}
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ProtocolError("HARP v4 compact target-action row is malformed.")
        start, stop = int(offsets[index]), int(offsets[index + 1])
        probability_bytes = _float32_bytes(probabilities[start:stop])
        if hashlib_sha256_bytes(probability_bytes) != raw.get("probability_bytes_sha256"):
            raise ProtocolError("HARP v4 compact target probability bytes drifted.")
        action = CaseTargetAction(
            outer_target_id=str(raw["outer_target_id"]),
            target_query_id=str(raw["outer_target_id"]),
            case_id=str(raw["case_id"]),
            action_kind=CoreActionKind(str(raw["action_kind"])),
            candidate_source_id=(
                None
                if raw.get("candidate_source_id") is None
                else str(raw["candidate_source_id"])
            ),
            feature_names=names,
            feature_values=tuple(float(item) for item in features[index]),
            sample_ids=tuple(str(item) for item in raw.get("sample_ids", ())),
            probability_bytes=probability_bytes,
            prediction_seal_hash=str(raw["prediction_seal_hash"]),
            expert_weight=float(raw["expert_weight"]),
        )
        if action.action_id != raw.get("action_id"):
            raise ProtocolError("HARP v4 compact target action identity drifted.")
        actions.append(action)
        expected_by_case[action.case_key] = tuple(
            str(item) for item in raw.get("expected_candidate_source_ids", ())
        )
    by_case: dict[tuple[str, str], list[CaseTargetAction]] = defaultdict(list)
    for action in actions:
        by_case[action.case_key].append(action)
    sets: list[CaseActionSet] = []
    for key, scoped in sorted(by_case.items()):
        baseline = tuple(row for row in scoped if row.action_kind is CoreActionKind.B)
        uniform = tuple(row for row in scoped if row.action_kind is CoreActionKind.U)
        experts = tuple(
            sorted(
                (row for row in scoped if row.action_kind is CoreActionKind.HXE),
                key=lambda row: row.candidate_source_id or "",
            )
        )
        if len(baseline) != 1 or len(uniform) != 1:
            raise ProtocolError("HARP v4 compact case lacks exactly one B/U action.")
        sets.append(
            CaseActionSet(
                baseline=baseline[0],
                uniform=uniform[0],
                experts=experts,
                expected_candidate_source_ids=expected_by_case[key],
            )
        )
    if len(sets) != value.manifest.get("case_count"):
        raise ProtocolError("HARP v4 compact target case count drifted.")
    return tuple(sets)


def _case_metric(
    probability: np.ndarray,
    labels: np.ndarray,
    *,
    total_case_count: int,
    class_support_case_counts: tuple[int, int],
) -> dict[str, float]:
    return case_metrics(
        probability,
        labels,
        total_case_count=total_case_count,
        class_support_case_counts=class_support_case_counts,
    ).as_dict()


def _center_metrics(rows: Sequence[Mapping[str, object]], role: str) -> dict[str, float]:
    labels = tuple(np.asarray(row["labels"], dtype=np.int64) for row in rows)
    support_counts = case_class_support_counts(labels)
    return aggregate_case_equal_metrics(
        tuple(
            case_metrics(
                np.asarray(row[role]),
                truth,
                total_case_count=len(rows),
                class_support_case_counts=support_counts,
            )
            for row, truth in zip(rows, labels, strict=True)
        )
    )


__all__ = (
    "FEATURE_NAMES",
    "HarpV4ProductionPipeline",
    "development_observations_from_artifact",
    "fit_collection_from_artifact",
    "target_action_sets_from_artifact",
)
