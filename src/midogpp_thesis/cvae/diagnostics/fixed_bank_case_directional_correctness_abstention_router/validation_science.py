"""Exact serial reconstruction of CDCA plans, routes, and terminal science."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from .constants import (
    ACTION_COUNT_PER_TARGET,
    B_ACTION_ID,
    CENTERS,
    DESCRIPTIVE_METHOD_IDS,
    DIRECTION_IDS,
    EXPECTED_TEST_ROW_COUNT,
    EXPECTED_TOTAL_CASE_COUNT,
    METHOD_IDS,
    PRE_TERMINAL_METHOD_IDS,
    U_ACTION_ID,
    candidate_sources,
)
from .decisions import (
    fit_route_directional_models,
    select_case_directional_abstention_decision,
)
from .donor_priors import compute_donor_priors
from .ensemble import compose_case_predictions, compose_fixed_action_predictions
from .features import (
    build_label_free_case_candidate_features,
    permute_route_candidate_feature_blocks,
)
from .hashing import canonical_hash
from .held_case_plans import build_held_case_plans, seal_held_case_plans
from .label_capabilities import DirectionalCorrectnessLabelFirewall
from .persistence import object_payload, read_rows
from .reports import (
    leakage_report_payload,
    publication_decision_payload,
    seal_payload,
)
from .scoring import (
    score_directional_correctness_observations,
    score_directional_gains,
    score_permuted_directional_correctness_observations,
    support_class_denominators,
)


LabelLoader = Callable[[frozenset[tuple[str, str, str]]], Sequence[object]]

PLAN_AND_FEATURE_TABLE_MEMBERS = (
    "tables/held_case_plans.csv",
    "tables/held_case_features.csv",
)
ROUTE_TABLE_MEMBERS = {
    "support_responses": "tables/support_response_counts.csv",
    "donor_priors": "tables/donor_priors.csv",
    "model_fits": "tables/route_model_fits.csv",
    "candidate_scores": "tables/route_candidate_scores.csv",
    "decisions": "tables/route_decisions.csv",
    "method_predictions": "tables/method_predictions.csv",
    "descriptive_predictions": "tables/descriptive_method_predictions.csv",
}
TERMINAL_TABLE_MEMBERS = {
    "case_confusions": "tables/terminal_case_confusions.csv",
    "method_metrics": "tables/terminal_method_metrics.csv",
    "center_metrics": "tables/terminal_center_metrics.csv",
    "contrasts": "tables/terminal_contrasts.csv",
    "router_identification": "tables/router_identification_metrics.csv",
    "feature_permutation_summary": "tables/feature_permutation_summary.csv",
}
ALL_RECONSTRUCTED_TABLE_MEMBERS = (
    "tables/action_library.csv",
    "tables/exact_nine_probability_index.csv",
    *PLAN_AND_FEATURE_TABLE_MEMBERS,
    *ROUTE_TABLE_MEMBERS.values(),
    *TERMINAL_TABLE_MEMBERS.values(),
)


def reconstruct_plan_and_feature_products(
    root: Path,
    *,
    frame: object,
    probability_surface: object,
    probability_surface_hash: str,
    physical_prelabel_seal_hash: str,
) -> Mapping[str, object]:
    """Rebuild all 218 plans and the complete label-free feature table."""

    plans = build_held_case_plans(
        getattr(frame, "rows"),
        probability_surface_hash=probability_surface_hash,
    )
    science_plan_seal = seal_held_case_plans(
        plans, probability_surface_hash=probability_surface_hash
    )
    features = build_label_free_case_candidate_features(probability_surface)
    plan_rows = _payloads(plans, "held-case plans")
    feature_rows = _payloads(features, "held-case features")
    expected_feature_count = (
        EXPECTED_TOTAL_CASE_COUNT
        * (ACTION_COUNT_PER_TARGET - 2)
        * len(DIRECTION_IDS)
    )
    if (
        len(plan_rows) != EXPECTED_TOTAL_CASE_COUNT
        or len(feature_rows) != expected_feature_count
        or any(row.get("labels_used") is not False for row in feature_rows)
    ):
        raise ProtocolError("Case-directional plan/feature topology drifted.")
    _compare_table(root, "tables/held_case_plans.csv", plan_rows)
    _compare_table(root, "tables/held_case_features.csv", feature_rows)
    plan_payload = object_payload(science_plan_seal)
    expected_plan_seal = seal_payload(
        "fixed_bank_cdca_held_case_plan_seal_v1",
        bindings={
            "physical_prelabel_seal_hash": physical_prelabel_seal_hash,
            "plans_hash": canonical_hash(plan_rows),
            "science_plan_seal_hash": plan_payload["plan_seal_hash"],
        },
        plan_count=len(plan_rows),
        held_case_excluded=True,
        labels_used=False,
    )
    expected_feature_seal = seal_payload(
        "fixed_bank_cdca_held_case_feature_seal_v1",
        bindings={
            "physical_prelabel_seal_hash": physical_prelabel_seal_hash,
            "held_case_plan_seal_hash": expected_plan_seal["seal_hash"],
            "held_case_features_hash": canonical_hash(feature_rows),
        },
        feature_count=len(feature_rows),
        labels_used=False,
        feature_schema_is_label_free=True,
        signed_delta_is_candidate_probability_minus_B=True,
    )
    _compare_json(
        root, "manifests/held_case_plan_seal.json", expected_plan_seal
    )
    _compare_json(
        root, "manifests/held_case_feature_seal.json", expected_feature_seal
    )
    return {
        "plans": plans,
        "science_plan_seal": science_plan_seal,
        "features": features,
        "persisted_plan_seal": expected_plan_seal,
        "feature_seal": expected_feature_seal,
        "held_case_plan_count": len(plan_rows),
        "held_case_feature_count": len(feature_rows),
    }


def reconstruct_route_products(
    root: Path,
    *,
    probability_surface: object,
    plans: Sequence[object],
    science_plan_seal: object,
    features: Sequence[object],
    persisted_plan_seal_hash: str,
    feature_seal_hash: str,
    label_loader: LabelLoader,
) -> Mapping[str, object]:
    """Serially replay all donor grants and all H-minus-c route fits."""

    plan_rows = tuple(plans)
    feature_rows = tuple(features)
    firewall = DirectionalCorrectnessLabelFirewall(
        science_plan_seal, label_loader
    )

    all_priors: list[object] = []
    priors_by_target: dict[str, tuple[object, ...]] = {}
    for target in CENTERS:
        gains_by_source: dict[str, tuple[object, ...]] = {}
        for source in candidate_sources(target):
            donor_labels = firewall.open_donor_labels(target, source)
            gains_by_source[source] = score_directional_gains(
                probability_surface, donor_labels
            )
        priors = compute_donor_priors(
            gains_by_source, heldout_center=target
        )
        priors_by_target[target] = priors
        all_priors.extend(priors)

    support_responses: list[object] = []
    model_fits: list[object] = []
    candidate_scores: list[dict[str, object]] = []
    decisions: list[object] = []
    routed_predictions: list[object] = []
    for plan in plan_rows:
        support_labels = firewall.open_route_support_labels(
            getattr(plan, "target_center"),
            getattr(plan, "case_id"),
            plan_hash=str(getattr(plan, "plan_hash")),
        )
        route_features = tuple(
            row
            for row in feature_rows
            if getattr(row, "target_center") == getattr(plan, "target_center")
            and getattr(row, "case_id")
            in {*getattr(plan, "support_case_ids"), getattr(plan, "case_id")}
        )
        held_features = tuple(
            row
            for row in route_features
            if getattr(row, "case_id") == getattr(plan, "case_id")
        )
        observations = score_directional_correctness_observations(
            probability_surface,
            support_labels,
            plan,
            features=route_features,
        )
        denominators = support_class_denominators(
            support_labels,
            plan,
            probability_surface_or_rows=probability_surface,
        )
        models = fit_route_directional_models(observations, plan)
        canonical_decisions = tuple(
            select_case_directional_abstention_decision(
                method_id=method_id,
                target_center=getattr(plan, "target_center"),
                case_id=getattr(plan, "case_id"),
                models=models,
                held_features=held_features,
                donor_priors=priors_by_target[getattr(plan, "target_center")],
                denominators=denominators,
            )
            for method_id in (
                "CDCA_LOO",
                "G_directional_matched",
                "CDCA_case_proxy_only",
            )
        )
        permuted_features = permute_route_candidate_feature_blocks(
            route_features, plan
        )
        permuted_observations = (
            score_permuted_directional_correctness_observations(
                probability_surface,
                support_labels,
                plan,
                permuted_features=permuted_features,
            )
        )
        permuted_models = fit_route_directional_models(
            permuted_observations, plan
        )
        permuted_held_features = tuple(
            row
            for row in permuted_features
            if getattr(row, "case_id") == getattr(plan, "case_id")
        )
        permuted_decision = select_case_directional_abstention_decision(
            method_id="CDCA_feature_block_permutation_descriptive",
            target_center=getattr(plan, "target_center"),
            case_id=getattr(plan, "case_id"),
            models=permuted_models,
            held_features=permuted_held_features,
            donor_priors=priors_by_target[getattr(plan, "target_center")],
            denominators=denominators,
        )
        route_decisions = (*canonical_decisions, permuted_decision)
        route_predictions = tuple(
            prediction
            for decision in route_decisions
            for prediction in compose_case_predictions(
                probability_surface, decision
            )
        )
        support_responses.extend(observations)
        model_fits.extend(models)
        candidate_scores.extend(
            {
                "method_id": decision.method_id,
                **score.to_payload(),
            }
            for decision in route_decisions
            for directional in (decision.zero_to_one, decision.one_to_zero)
            for score in directional.candidate_scores
        )
        decisions.extend(route_decisions)
        routed_predictions.extend(route_predictions)

    fixed = {
        B_ACTION_ID: compose_fixed_action_predictions(
            probability_surface, method_id=B_ACTION_ID
        ),
        U_ACTION_ID: compose_fixed_action_predictions(
            probability_surface, method_id=U_ACTION_ID
        ),
    }
    primary_predictions = tuple(
        row
        for method_id in PRE_TERMINAL_METHOD_IDS
        for row in (
            fixed[method_id]
            if method_id in fixed
            else tuple(
                value
                for value in routed_predictions
                if getattr(value, "method_id") == method_id
            )
        )
    )
    descriptive_predictions = tuple(
        row
        for method_id in DESCRIPTIVE_METHOD_IDS
        for row in routed_predictions
        if getattr(row, "method_id") == method_id
    )
    for plan in plan_rows:
        key = (getattr(plan, "target_center"), getattr(plan, "case_id"))
        route_payload = {
            "decisions": [
                row.to_payload()
                for row in decisions
                if (row.target_center, row.case_id) == key
            ],
            "predictions": [
                row.to_payload()
                for row in (*primary_predictions, *descriptive_predictions)
                if (row.target_center, row.case_id) == key
            ],
        }
        firewall.record_route_decision_seal(*key, canonical_hash(route_payload))
    route_barrier = firewall.decision_barrier_payload()

    expected = {
        "support_responses": _payloads(
            support_responses, "support response counts"
        ),
        "donor_priors": _payloads(all_priors, "donor priors"),
        "model_fits": _payloads(model_fits, "route model fits"),
        "candidate_scores": _payloads(
            candidate_scores, "route candidate scores"
        ),
        "decisions": _payloads(decisions, "route decisions"),
        "method_predictions": _payloads(
            primary_predictions, "method predictions"
        ),
        "descriptive_predictions": _payloads(
            descriptive_predictions, "descriptive method predictions"
        ),
    }
    _validate_route_topology(expected, plan_rows)
    for key, member in ROUTE_TABLE_MEMBERS.items():
        _compare_table(root, member, expected[key])

    prior_seal = seal_payload(
        "fixed_bank_cdca_donor_prior_seal_v1",
        bindings={
            "held_case_plan_seal_hash": persisted_plan_seal_hash,
            "donor_priors_hash": canonical_hash(expected["donor_priors"]),
        },
        donor_prior_count=len(expected["donor_priors"]),
        donor_scope="q_not_in_H_or_e",
    )
    model_seal = seal_payload(
        "fixed_bank_cdca_route_model_seal_v1",
        bindings={
            "held_case_feature_seal_hash": feature_seal_hash,
            "donor_prior_seal_hash": prior_seal["seal_hash"],
            "support_responses_hash": canonical_hash(
                expected["support_responses"]
            ),
            "route_model_fits_hash": canonical_hash(expected["model_fits"]),
            "route_candidate_scores_hash": canonical_hash(
                expected["candidate_scores"]
            ),
        },
        model_fit_count=len(expected["model_fits"]),
        every_fit_is_H_minus_c=True,
        route_local_state_not_shared=True,
    )
    decision_seal = seal_payload(
        "fixed_bank_cdca_route_decision_seal_v1",
        bindings={
            "route_model_seal_hash": model_seal["seal_hash"],
            "route_decisions_hash": canonical_hash(expected["decisions"]),
            "method_predictions_hash": canonical_hash(
                expected["method_predictions"]
            ),
            "descriptive_predictions_hash": canonical_hash(
                expected["descriptive_predictions"]
            ),
            "route_decision_barrier_hash": route_barrier[
                "decision_barrier_hash"
            ],
        },
        route_direction_decision_count=len(expected["decisions"]),
        final_predictions_sealed=True,
        terminal_labels_used=False,
    )
    aggregate_seal = seal_payload(
        "fixed_bank_cdca_aggregate_plan_decision_seal_v1",
        bindings={
            "held_case_plan_seal_hash": persisted_plan_seal_hash,
            "held_case_feature_seal_hash": feature_seal_hash,
            "donor_prior_seal_hash": prior_seal["seal_hash"],
            "route_model_seal_hash": model_seal["seal_hash"],
            "route_decision_seal_hash": decision_seal["seal_hash"],
            "route_decision_barrier_hash": route_barrier[
                "decision_barrier_hash"
            ],
        },
        route_count=EXPECTED_TOTAL_CASE_COUNT,
        all_route_probabilities_and_decisions_sealed=True,
        terminal_labels_used=False,
    )
    for member, payload in (
        ("manifests/donor_prior_seal.json", prior_seal),
        ("manifests/route_model_seal.json", model_seal),
        ("manifests/route_decision_seal.json", decision_seal),
        ("manifests/aggregate_plan_decision_seal.json", aggregate_seal),
    ):
        _compare_json(root, member, payload)
    firewall.record_aggregate_plan_decision_seal(
        str(aggregate_seal["seal_hash"]),
        plan_seal_hash=str(getattr(science_plan_seal, "plan_seal_hash")),
        decision_barrier_hash=str(route_barrier["decision_barrier_hash"]),
    )
    return {
        "firewall": firewall,
        "donor_priors": tuple(all_priors),
        "support_responses": tuple(support_responses),
        "model_fits": tuple(model_fits),
        "candidate_scores": tuple(candidate_scores),
        "decisions": tuple(decisions),
        "method_predictions": primary_predictions,
        "descriptive_predictions": descriptive_predictions,
        "route_barrier": route_barrier,
        "prior_seal": prior_seal,
        "model_seal": model_seal,
        "decision_seal": decision_seal,
        "aggregate_seal": aggregate_seal,
        "support_response_count": len(expected["support_responses"]),
        "donor_prior_count": len(expected["donor_priors"]),
        "route_model_fit_count": len(expected["model_fits"]),
        "route_candidate_score_count": len(expected["candidate_scores"]),
        "route_decision_count": len(expected["decisions"]),
        "preterminal_prediction_count": len(expected["method_predictions"]),
        "descriptive_prediction_count": len(
            expected["descriptive_predictions"]
        ),
    }


def reconstruct_terminal_products(
    root: Path,
    *,
    probability_surface: object,
    method_predictions: Sequence[object],
    descriptive_predictions: Sequence[object],
    decisions: Sequence[object],
    aggregate_seal: Mapping[str, object],
    firewall: DirectionalCorrectnessLabelFirewall,
    prediction: object,
    source: object,
    preflight: Mapping[str, object],
    runtime: Mapping[str, object],
    physical_prelabel_seal_hash: str,
    feature_seal_hash: str,
) -> Mapping[str, object]:
    """Open terminal labels only now, then rebuild every terminal product/report."""

    terminal_labels = firewall.open_terminal_labels()
    try:
        from .terminal import evaluate_terminal
    except ImportError as exc:  # pragma: no cover - installation failure
        raise ProtocolError("Case-directional terminal evaluator is unavailable.") from exc
    terminal = evaluate_terminal(
        probability_surface=probability_surface,
        method_predictions=method_predictions,
        descriptive_predictions=descriptive_predictions,
        decisions=decisions,
        aggregate_plan_decision_seal_hash=str(aggregate_seal["seal_hash"]),
        terminal_labels=terminal_labels,
    )
    terminal_rows: dict[str, tuple[dict[str, object], ...]] = {}
    for key, member in TERMINAL_TABLE_MEMBERS.items():
        values = terminal.get(key)
        if not isinstance(values, (tuple, list)):
            raise ProtocolError(
                f"Case-directional reconstructed terminal surface absent: {key}."
            )
        rows = _payloads(values, key)
        terminal_rows[key] = rows
        _compare_table(root, member, rows)
    _validate_terminal_topology(terminal_rows)
    terminal_seal = terminal.get("terminal_seal")
    if not isinstance(terminal_seal, Mapping):
        raise ProtocolError("Case-directional reconstructed terminal seal is absent.")
    _compare_json(
        root, "manifests/terminal_evaluation_seal.json", terminal_seal
    )

    capability = firewall.report_payload()
    expected_leakage = leakage_report_payload(
        prediction_seal_hash=str(getattr(prediction, "seal_hash")),
        physical_prelabel_seal_hash=physical_prelabel_seal_hash,
        held_case_feature_seal_hash=feature_seal_hash,
        aggregate_plan_decision_seal_hash=str(aggregate_seal["seal_hash"]),
        capability_report=capability,
    )
    descriptive_summary = terminal.get("descriptive_summary")
    if not isinstance(descriptive_summary, Mapping):
        raise ProtocolError("Case-directional descriptive summary is absent.")
    expected_publication = publication_decision_payload(
        str(terminal_seal["seal_hash"]),
        descriptive_summary=descriptive_summary,
    )
    expected_runtime = _runtime_summary(
        source=source,
        prediction=prediction,
        preflight=preflight,
        runtime=runtime,
    )
    for member, payload in (
        ("reports/label_capability_report.json", capability),
        ("reports/leakage_report.json", expected_leakage),
        ("reports/publication_decision.json", expected_publication),
        ("reports/runtime_summary.json", expected_runtime),
    ):
        _compare_json(root, member, payload)
    if (
        capability.get("status") != "PASS"
        or capability.get("donor_grant_count") != 72
        or capability.get("route_support_grant_count")
        != EXPECTED_TOTAL_CASE_COUNT
        or capability.get("route_decision_seal_count")
        != EXPECTED_TOTAL_CASE_COUNT
        or capability.get("terminal_opened") is not True
    ):
        raise ProtocolError("Case-directional label capability replay drifted.")
    return {
        "terminal_case_confusion_count": len(terminal_rows["case_confusions"]),
        "terminal_method_metric_count": len(terminal_rows["method_metrics"]),
        "terminal_center_metric_count": len(terminal_rows["center_metrics"]),
        "terminal_contrast_count": len(terminal_rows["contrasts"]),
        "router_identification_count": len(
            terminal_rows["router_identification"]
        ),
        "feature_permutation_summary_count": len(
            terminal_rows["feature_permutation_summary"]
        ),
        "terminal_seal_hash": terminal_seal["seal_hash"],
        "donor_grant_count": capability["donor_grant_count"],
        "route_support_grant_count": capability["route_support_grant_count"],
        "terminal_opened_after_aggregate_barrier": True,
        "terminal_reports_exact": True,
    }


def _validate_route_topology(
    expected: Mapping[str, tuple[dict[str, object], ...]],
    plans: Sequence[object],
) -> None:
    support_count = sum(
        len(getattr(plan, "support_case_ids"))
        * len(candidate_sources(getattr(plan, "target_center")))
        * len(DIRECTION_IDS)
        for plan in plans
    )
    if (
        len(expected["support_responses"]) != support_count
        or len(expected["donor_priors"])
        != len(CENTERS) * (ACTION_COUNT_PER_TARGET - 2) * len(DIRECTION_IDS)
        or len(expected["model_fits"])
        != EXPECTED_TOTAL_CASE_COUNT
        * (ACTION_COUNT_PER_TARGET - 2)
        * len(DIRECTION_IDS)
        or len(expected["candidate_scores"])
        != EXPECTED_TOTAL_CASE_COUNT
        * (3 + len(DESCRIPTIVE_METHOD_IDS))
        * len(DIRECTION_IDS)
        * (ACTION_COUNT_PER_TARGET - 1)
        or len(expected["decisions"])
        != EXPECTED_TOTAL_CASE_COUNT * (3 + len(DESCRIPTIVE_METHOD_IDS))
        or len(expected["method_predictions"])
        != EXPECTED_TEST_ROW_COUNT * len(PRE_TERMINAL_METHOD_IDS)
        or len(expected["descriptive_predictions"])
        != EXPECTED_TEST_ROW_COUNT * len(DESCRIPTIVE_METHOD_IDS)
        or tuple(
            dict.fromkeys(
                str(row["method_id"])
                for row in expected["method_predictions"]
            )
        )
        != PRE_TERMINAL_METHOD_IDS
        or tuple(
            dict.fromkeys(
                str(row["method_id"])
                for row in expected["descriptive_predictions"]
            )
        )
        != DESCRIPTIVE_METHOD_IDS
        or any(
            row["case_id"] in row["training_case_ids"]
            for row in expected["model_fits"]
        )
        or any(
            row["heldout_center"] in row["query_centers"]
            or row["source"] in row["query_centers"]
            for row in expected["donor_priors"]
        )
    ):
        raise ProtocolError("Case-directional reconstructed route topology drifted.")


def _validate_terminal_topology(
    tables: Mapping[str, tuple[dict[str, object], ...]],
) -> None:
    reported_methods = (*METHOD_IDS, *DESCRIPTIVE_METHOD_IDS)
    if (
        len(tables["case_confusions"])
        != EXPECTED_TOTAL_CASE_COUNT * len(reported_methods)
        or len(tables["method_metrics"]) != len(reported_methods)
        or len(tables["center_metrics"]) != len(CENTERS) * len(reported_methods)
        or len(tables["contrasts"]) != 5
        or len(tables["router_identification"]) != len(CENTERS)
        or len(tables["feature_permutation_summary"]) != 1
        or tuple(row["method_id"] for row in tables["method_metrics"])
        != reported_methods
        or Counter(row["method_id"] for row in tables["case_confusions"])
        != Counter({method: EXPECTED_TOTAL_CASE_COUNT for method in reported_methods})
        or Counter(row["method_id"] for row in tables["center_metrics"])
        != Counter({method: len(CENTERS) for method in reported_methods})
        or tuple(row["baseline_id"] for row in tables["contrasts"])
        != (
            "B",
            "U",
            "G_directional_matched",
            "CDCA_case_proxy_only",
            "CDCA_feature_block_permutation_descriptive",
        )
        or tuple(row["target_center"] for row in tables["router_identification"])
        != CENTERS
    ):
        raise ProtocolError("Case-directional terminal table topology drifted.")


def _runtime_summary(
    *,
    source: object,
    prediction: object,
    preflight: Mapping[str, object],
    runtime: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_cdca_runtime_summary_v1",
        "status": "PASS",
        "source_stream_lock_hash": str(getattr(source, "lock_hash")),
        "global_prediction_seal_hash": str(getattr(prediction, "seal_hash")),
        "source_stream_count": len(getattr(source, "records")),
        "classifier_cell_count": len(getattr(prediction, "store").cells),
        "unique_classifier_fit_count": len(getattr(prediction, "store").cells),
        "workstation_preflight": dict(preflight),
        "source_generation_devices": ["cuda:0", "cuda:1"],
        "persistent_generation_worker_count": 2,
        "gpu_generation_completed_before_cpu_phase": True,
        "cuda_visible_devices_during_cpu_phase": "",
        "classifier_workers": int(runtime["classifier_workers"]),
        "classifier_threads_per_worker": int(
            runtime["classifier_threads_per_worker"]
        ),
        "multiprocessing_start_method": runtime["multiprocessing_start_method"],
        "source_storage_dtype": "float32",
        "probability_storage_dtype": "float32",
        "confusion_count_dtype": "int64",
        "scientific_reductions_dtype": "float64",
        "resume_policy": runtime["resume_policy"],
        "task_checkpoints_are_intra_launch_atomicity_only": True,
        "terminal_or_cross_run_recovery_used": False,
        "dedicated_local_scratch_used_for_throughput": True,
        "classifier_source_cache_role": "dedicated_intra_launch_scratch",
        "canonical_source_cache_role": "current_artifact_root",
        "scratch_root_id": (
            "fixed_bank_case_directional_correctness_abstention_router_v1"
        ),
        "local_and_canonical_source_lock_identical": True,
        "prior_run_scratch_used_as_evidence": False,
        "previous_stage90_artifact_checkpoint_or_scratch_reused": False,
        "recomputed_from_original_six_inputs": True,
    }


def _payloads(
    values: Sequence[object], role: str
) -> tuple[dict[str, object], ...]:
    rows = tuple(object_payload(value) for value in values)
    if not rows:
        raise ProtocolError(f"Case-directional reconstructed {role} are empty.")
    return rows


def _compare_table(
    root: Path, member: str, expected: tuple[dict[str, object], ...]
) -> None:
    if read_rows(root / member) != expected:
        raise ProtocolError(
            f"Case-directional persisted table is not reconstructive: {member}."
        )


def _compare_json(
    root: Path, member: str, expected: Mapping[str, object]
) -> None:
    if read_json(root / member) != dict(expected):
        raise ProtocolError(
            f"Case-directional persisted seal/report is not reconstructive: {member}."
        )


__all__ = (
    "ALL_RECONSTRUCTED_TABLE_MEMBERS",
    "PLAN_AND_FEATURE_TABLE_MEMBERS",
    "ROUTE_TABLE_MEMBERS",
    "TERMINAL_TABLE_MEMBERS",
    "reconstruct_plan_and_feature_products",
    "reconstruct_route_products",
    "reconstruct_terminal_products",
)
