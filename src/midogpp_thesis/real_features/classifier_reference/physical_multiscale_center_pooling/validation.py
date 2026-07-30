"""Independent completed-bundle validation for the Stage-10 pilot."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from midogpp_thesis.common.hashing import stable_hash

from ..classifiers import ClassifierSpec
from ..downstream import balanced_accuracy, macro_f1
from ..protocol import ProtocolError
from .artifacts import REQUIRED_STATIC_FILES
from .config import (
    PhysicalMultiscalePilotConfig,
    representation_candidate_grid_hash,
)
from .decision_lock import read_decision_lock, selector_table_hash
from .frames import CenterShardedRepresentationStore
from .input_lineage import compute_input_hashes
from .reporting import decision_summary
from .selection import choose_representation_from_vectors
from .statistics import paired_case_cluster_bootstrap


def validate_physical_multiscale_pilot_bundle(
    root: str | Path,
    *,
    config: PhysicalMultiscalePilotConfig,
    allow_pending: bool = False,
) -> Mapping[str, object]:
    """Reconstruct every claim-bearing decision from immutable inputs and tables."""

    path = Path(root)
    required = set(REQUIRED_STATIC_FILES)
    if allow_pending:
        required.remove("reports/validation_report.json")
    missing = [relative for relative in sorted(required) if not (path / relative).is_file()]
    lock_paths = tuple(
        path / "manifests" / "decision_locks" / f"center_{center}.json"
        for center in config.heldout_centers
    )
    missing.extend(
        str(lock.relative_to(path)) for lock in lock_paths if not lock.is_file()
    )
    if missing:
        raise ProtocolError(f"Physical multiscale bundle is incomplete: {missing}")

    protocol = _json(path / "manifests" / "protocol_manifest.json")
    frozen = _json(path / "manifests" / "frozen_protocol_snapshot.json")
    leakage = _json(path / "reports" / "leakage_provenance_report.json")
    lock_index = _json(path / "manifests" / "decision_lock_index.json")
    expected_status = "PENDING_INDEPENDENT_VALIDATION" if allow_pending else "PASS"
    frozen_without_hashes = {
        key: value
        for key, value in frozen.items()
        if key not in {"config_hash", "protocol_hash"}
    }
    frozen_without_protocol_hash = {
        key: value for key, value in frozen.items() if key != "protocol_hash"
    }
    if (
        protocol.get("status") != expected_status
        or protocol.get("claim_scope") != "real_feature_transfer_only"
        or protocol.get("claim_role")
        != (
            "complete_deterministic_representation_plus_classifier_"
            "pipeline_diagnostic"
        )
        or protocol.get("non_adoptive") is not True
        or protocol.get("may_feed_recipe_selection") is not False
        or protocol.get("may_feed_deployable_selection") is not False
        or protocol.get("inner_delta_role") != "optimistic_selection_statistic"
        or protocol.get("not_performance_estimate") is not True
        or protocol.get("gate_is_statistical_test") is not False
        or protocol.get("probabilities_calibrated") is not False
        or protocol.get("covers_new_center_uncertainty") is not False
        or protocol.get("representation_c_combination")
        != "feature_concatenation_not_mixture"
        or protocol.get("uses_likelihood") is not False
        or protocol.get("uses_latent_prior") is not False
        or protocol.get("uses_posterior") is not False
        or protocol.get("uses_mixture_model") is not False
        or protocol.get("uses_experts") is not False
        or protocol.get("performs_expert_aggregation") is not False
        or protocol.get("uses_generative_sampling") is not False
        or protocol.get("global_representation_adoption_allowed") is not False
        or protocol.get("bootstrap_conditions_on_fixed_fits_and_locked_selection")
        is not True
        or leakage.get("status") != expected_status
        or leakage.get("target_labels_used_for_selection") is not False
        or leakage.get("fit_used_target_center") is not False
        or leakage.get("decision_locks_written_before_outer_evaluation") is not True
        or leakage.get("posthoc_rows_used_for_selection") is not False
        or stable_hash(frozen_without_hashes) != frozen.get("config_hash")
        or stable_hash(frozen_without_protocol_hash) != frozen.get("protocol_hash")
        or protocol.get("protocol_hash") != frozen.get("protocol_hash")
        or frozen.get("profile_id") != config.profile.profile_id
        or frozen.get("representations") != dict(config.representation_dims)
    ):
        raise ProtocolError("Physical multiscale protocol/leakage manifest failed.")
    if not allow_pending and (
        protocol.get("independent_validation_status") != "PASS"
        or leakage.get("independent_validation_status") != "PASS"
    ):
        raise ProtocolError("Physical multiscale independent validation status failed.")

    locks = [read_decision_lock(lock) for lock in lock_paths]
    expected_bundle_hash = stable_hash(
        [
            {
                "outer_target_center": str(lock.payload["outer_target_center"]),
                "path": str(lock.path.relative_to(path)),
                "decision_hash": lock.decision_hash,
            }
            for lock in locks
        ]
    )
    if (
        lock_index.get("status") != "LOCKED_BEFORE_OUTER_EVALUATION"
        or lock_index.get("lock_count") != len(config.heldout_centers)
        or lock_index.get("bundle_lock_hash") != expected_bundle_hash
        or lock_index.get("posthoc_rows_in_hash") is not False
        or protocol.get("bundle_lock_hash") != expected_bundle_hash
    ):
        raise ProtocolError("Decision-lock bundle hash failed validation.")

    cells = _csv(path / "tables" / "source_inner_selector_cells.csv")
    summaries = _csv(path / "tables" / "source_inner_candidate_summary.csv")
    decisions = _csv(path / "tables" / "representation_decisions.csv")
    outer_results = _csv(path / "tables" / "outer_locked_results.csv")
    outer_predictions = _csv(path / "tables" / "outer_locked_predictions.csv")
    fit_audit = _csv(path / "tables" / "outer_fit_audit.csv")
    replay = _csv(path / "tables" / "canonical_a_replay.csv")
    posthoc = _csv(path / "tables" / "posthoc_candidate_isolation.csv")
    try:
        input_hashes = compute_input_hashes(config)
    except ValueError as exc:
        raise ProtocolError(str(exc)) from exc
    if (
        frozen.get("input_hashes") != input_hashes
        or protocol.get("input_hashes") != input_hashes
    ):
        raise ProtocolError("Physical multiscale current input hashes drifted.")

    _validate_cardinality_and_cartesian(config, cells, summaries, decisions)
    _validate_selector_source_rows(config, cells)
    if any(
        row.get("selection_used_target_labels") != "False"
        or row.get("fit_used_target_center") != "False"
        or row.get("inner_delta_role") != "optimistic_selection_statistic"
        or row.get("not_performance_estimate") != "True"
        or row.get("gate_is_statistical_test") != "False"
        for row in cells
    ):
        raise ProtocolError("Selector table violates target or claim firewall.")
    if any(
        row.get("status") != "PASS"
        or row.get("predictions_exact") != "True"
        or row.get("classifier_hash_exact") != "True"
        for row in replay
    ):
        raise ProtocolError("Canonical A replay failed.")
    if any(
        row.get("fresh_outer_fit") != "True"
        or row.get("target_center_absent_from_fit") != "True"
        or row.get("inner_fit_state_reused") != "False"
        for row in fit_audit
    ):
        raise ProtocolError("Fresh outer-fit audit failed.")

    reconstructed = _reconstruct_decisions(config, cells, summaries)
    cells_by_center = {
        center: [row for row in cells if row["outer_target_center"] == center]
        for center in config.heldout_centers
    }
    decision_by_center = _unique_by(decisions, "outer_target_center")
    expected_candidate_hash = representation_candidate_grid_hash(
        config.classifier_specs,
        config.profile,
    )
    for lock in locks:
        center = str(lock.payload["outer_target_center"])
        source_centers = tuple(
            value for value in config.heldout_centers if value != center
        )
        expected = reconstructed[center]
        locked_specs = lock.payload.get("representation_classifier_specs")
        if not isinstance(locked_specs, Mapping):
            raise ProtocolError(f"Decision lock lacks classifier map for center {center}.")
        expected_specs = {
            rep: expected["selected_by_rep"][rep][0].to_payload()
            for rep in config.representation_order
        }
        decision_row = decision_by_center.get(center)
        if (
            lock.payload.get("config_hash") != frozen.get("config_hash")
            or lock.payload.get("candidate_grid_hash") != expected_candidate_hash
            or lock.payload.get("input_hashes") != input_hashes
            or lock.payload.get("selector_table_hash")
            != selector_table_hash(cells_by_center[center])
            or tuple(lock.payload.get("source_centers", ())) != source_centers
            or dict(locked_specs) != expected_specs
            or lock.payload.get("selected_representation")
            != expected["selected_representation"]
            or lock.payload.get("selected_classifier_hash")
            != expected["selected_classifier_hash"]
            or lock.payload.get("canonical_a_classifier_hash")
            != expected["canonical_a_classifier_hash"]
            or not _float_equal(lock.payload.get("mean_delta"), expected["mean_delta"])
            or not _float_equal(lock.payload.get("worst_delta"), expected["worst_delta"])
            or int(lock.payload.get("strict_wins", -1)) != expected["strict_wins"]
            or bool(lock.payload.get("gate_passed")) != expected["gate_passed"]
            or decision_row is None
            or decision_row.get("decision_hash") != lock.decision_hash
            or decision_row.get("selected_representation")
            != expected["selected_representation"]
            or decision_row.get("selected_classifier_hash")
            != expected["selected_classifier_hash"]
            or decision_row.get("canonical_a_classifier_hash")
            != expected["canonical_a_classifier_hash"]
            or decision_row.get("source_centers") != ",".join(source_centers)
            or not _float_equal(decision_row.get("mean_delta"), expected["mean_delta"])
            or not _float_equal(decision_row.get("worst_delta"), expected["worst_delta"])
            or int(decision_row.get("strict_wins", -1)) != expected["strict_wins"]
            or (decision_row.get("gate_passed") == "True") != expected["gate_passed"]
            or decision_row.get("inner_delta_role")
            != "optimistic_selection_statistic"
            or decision_row.get("not_performance_estimate") != "True"
            or decision_row.get("gate_is_statistical_test") != "False"
        ):
            raise ProtocolError(f"Decision-lock reconstruction failed for center {center}.")

    if len(outer_results) != 2 * len(config.heldout_centers) or len(
        fit_audit
    ) != 2 * len(config.heldout_centers) or len(replay) != len(
        config.heldout_centers
    ):
        raise ProtocolError("Physical multiscale outer table cardinalities failed.")
    decision_hashes = {row["decision_hash"] for row in decisions}
    if any(
        row.get("row_role") != "posthoc_target_scored_candidate"
        or row.get("may_feed_selection") != "False"
        or row.get("decision_hash") not in decision_hashes
        for row in posthoc
    ):
        raise ProtocolError("Posthoc candidate isolation failed.")
    _validate_outer_roles(
        config,
        decisions,
        outer_results,
        outer_predictions,
        fit_audit,
        posthoc,
    )
    _validate_outer_metrics_and_canonical_reference(
        config,
        outer_results,
        outer_predictions,
        replay,
    )
    _validate_fallback_zero(decisions, outer_results, outer_predictions)

    bootstrap = _json(path / "reports" / "conditional_bootstrap.json")
    expected_bootstrap = paired_case_cluster_bootstrap(
        outer_predictions, config=config.bootstrap
    )
    if dict(bootstrap) != dict(expected_bootstrap):
        raise ProtocolError("Conditional paired bootstrap failed reconstruction.")
    summary = _json(path / "reports" / "decision_summary.json")
    expected_summary = decision_summary(
        decisions,
        outer_results,
        bootstrap,
        profile_id=config.profile.profile_id,
    )
    if dict(summary) != expected_summary:
        raise ProtocolError("Decision summary failed reconstruction.")

    _validate_content_index(path)
    result = {
        "status": "PASS",
        "selector_cells": len(cells),
        "candidate_summaries": len(summaries),
        "decision_locks": len(locks),
        "outer_results": len(outer_results),
        "posthoc_rows": len(posthoc),
    }
    if not allow_pending:
        report = _json(path / "reports" / "validation_report.json")
        if (
            report.get("status") != "PASS"
            or report.get("validator")
            != "validate_physical_multiscale_pilot_bundle"
            or report.get("authoritative_bundle_verdict") is not True
            or report.get("checks") != result
        ):
            raise ProtocolError("Independent validation report failed verification.")
    return result


def _validate_cardinality_and_cartesian(
    config: PhysicalMultiscalePilotConfig,
    cells: Sequence[Mapping[str, str]],
    summaries: Sequence[Mapping[str, str]],
    decisions: Sequence[Mapping[str, str]],
) -> None:
    if (
        len(cells) != config.expected_selector_cells
        or len(summaries) != config.expected_candidate_summaries
        or len(decisions) != len(config.heldout_centers)
    ):
        raise ProtocolError("Physical multiscale selector cardinalities failed.")
    spec_hashes = tuple(spec.config_hash for spec in config.classifier_specs)
    expected_cells = {
        (heldout, inner, rep, spec_hash)
        for heldout in config.heldout_centers
        for inner in config.heldout_centers
        if inner != heldout
        for rep in config.representation_order
        for spec_hash in spec_hashes
    }
    observed_cells = [
        (
            row.get("outer_target_center", ""),
            row.get("inner_pseudo_target_center", ""),
            row.get("representation_id", ""),
            row.get("classifier_config_hash", ""),
        )
        for row in cells
    ]
    expected_summaries = {
        (heldout, rep, spec_hash)
        for heldout in config.heldout_centers
        for rep in config.representation_order
        for spec_hash in spec_hashes
    }
    observed_summaries = [
        (
            row.get("outer_target_center", ""),
            row.get("representation_id", ""),
            row.get("classifier_config_hash", ""),
        )
        for row in summaries
    ]
    if (
        len(observed_cells) != len(set(observed_cells))
        or set(observed_cells) != expected_cells
        or len(observed_summaries) != len(set(observed_summaries))
        or set(observed_summaries) != expected_summaries
    ):
        raise ProtocolError("Physical multiscale selector Cartesian product is incomplete.")


def _validate_selector_source_rows(
    config: PhysicalMultiscalePilotConfig,
    cells: Sequence[Mapping[str, str]],
) -> None:
    store = CenterShardedRepresentationStore(
        b_cache_root=config.b_cache_root,
        c_cache_root=config.c_cache_root,
        profile=config.profile,
    )
    for heldout in config.heldout_centers:
        source_centers = tuple(
            center for center in config.heldout_centers if center != heldout
        )
        frame = store.selector_frame(
            outer_target_center=heldout,
            eligible_centers=config.heldout_centers,
        )
        expected_by_inner: dict[str, tuple[str, str, str]] = {}
        for inner in source_centers:
            train_centers = tuple(center for center in source_centers if center != inner)
            train_idx = frame.indices_for(train_centers)
            eval_idx = frame.indices_for((inner,))
            expected_by_inner[inner] = (
                ",".join(train_centers),
                stable_hash(tuple(frame.sample_ids[index] for index in train_idx)),
                stable_hash(tuple(frame.sample_ids[index] for index in eval_idx)),
            )
        for row in cells:
            if row["outer_target_center"] != heldout:
                continue
            expected = expected_by_inner[row["inner_pseudo_target_center"]]
            if (
                row.get("train_centers") != expected[0]
                or row.get("fit_sample_id_hash") != expected[1]
                or row.get("eval_sample_id_hash") != expected[2]
            ):
                raise ProtocolError(
                    f"Selector source-row lineage failed for center {heldout}."
                )


def _reconstruct_decisions(
    config: PhysicalMultiscalePilotConfig,
    cells: Sequence[Mapping[str, str]],
    summaries: Sequence[Mapping[str, str]],
) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    specs_by_hash = {spec.config_hash: spec for spec in config.classifier_specs}
    summary_by_key = _unique_by(
        summaries,
        "outer_target_center",
        "representation_id",
        "classifier_config_hash",
    )
    for heldout in config.heldout_centers:
        centers = tuple(center for center in config.heldout_centers if center != heldout)
        selected_by_rep: dict[str, tuple[ClassifierSpec, Mapping[str, float]]] = {}
        for rep in config.representation_order:
            candidates: list[tuple[float, ClassifierSpec, dict[str, float]]] = []
            for spec in config.classifier_specs:
                vector = {
                    row["inner_pseudo_target_center"]: float(row["bacc"])
                    for row in cells
                    if row["outer_target_center"] == heldout
                    and row["representation_id"] == rep
                    and row["classifier_config_hash"] == spec.config_hash
                }
                if set(vector) != set(centers):
                    raise ProtocolError("Selector score vector is incomplete.")
                mean_bacc = sum(vector.values()) / float(len(vector))
                candidates.append((mean_bacc, spec, vector))
                summary_row = summary_by_key.get((heldout, rep, spec.config_hash))
                expected_vector = json.dumps(
                    dict(sorted(vector.items(), key=lambda item: int(item[0]))),
                    sort_keys=True,
                )
                if (
                    summary_row is None
                    or summary_row.get("feature_dim")
                    != str(config.representation_dims[rep])
                    or not _float_equal(
                        summary_row.get("equal_center_mean_bacc"), mean_bacc
                    )
                    or summary_row.get("center_bacc_vector") != expected_vector
                    or summary_row.get("row_role")
                    != "source_inner_candidate_summary"
                ):
                    raise ProtocolError(
                        f"Candidate-summary reconstruction failed for center {heldout}."
                    )
            best_score = max(item[0] for item in candidates)
            tied = [
                item
                for item in candidates
                if math.isclose(item[0], best_score, abs_tol=1.0e-12, rel_tol=0.0)
            ]
            selected = min(tied, key=lambda item: item[1].tie_break_key())
            selected_by_rep[rep] = (selected[1], selected[2])
        representation_id, selected_spec, mean_delta, worst_delta, wins, passed = (
            choose_representation_from_vectors(
                selected_by_rep,
                centers=centers,
                gate=config.gate,
                representation_order=config.representation_order,
                representation_dims=config.representation_dims,
            )
        )
        out[heldout] = {
            "selected_by_rep": selected_by_rep,
            "selected_representation": representation_id,
            "selected_classifier_hash": selected_spec.config_hash,
            "canonical_a_classifier_hash": selected_by_rep["canonical_a"][0].config_hash,
            "mean_delta": mean_delta,
            "worst_delta": worst_delta,
            "strict_wins": wins,
            "gate_passed": passed,
        }
    if set(specs_by_hash) != {spec.config_hash for spec in config.classifier_specs}:
        raise ProtocolError("Classifier specification hash collision detected.")
    return out


def _validate_fallback_zero(
    decisions: list[dict[str, str]],
    results: list[dict[str, str]],
    predictions: list[dict[str, str]],
) -> None:
    selected_by_center = {
        row["outer_target_center"]: row["selected_representation"] for row in decisions
    }
    for center, representation in selected_by_center.items():
        if representation != "canonical_a":
            continue
        by_role = {
            role: next(
                row
                for row in results
                if row["heldout_center"] == center and row["role"] == role
            )
            for role in ("canonical_a", "selected_policy")
        }
        if by_role["canonical_a"]["bacc"] != by_role["selected_policy"]["bacc"]:
            raise ProtocolError("Fallback-A result delta is not exactly zero.")
        keyed = {
            role: {
                row["sample_id"]: row["prediction"]
                for row in predictions
                if row["heldout_center"] == center and row["role"] == role
            }
            for role in ("canonical_a", "selected_policy")
        }
        if keyed["canonical_a"] != keyed["selected_policy"]:
            raise ProtocolError("Fallback-A predictions are not exactly identical.")


def _validate_outer_roles(
    config: PhysicalMultiscalePilotConfig,
    decisions: list[dict[str, str]],
    results: list[dict[str, str]],
    predictions: list[dict[str, str]],
    fit_audit: list[dict[str, str]],
    posthoc: list[dict[str, str]],
) -> None:
    selected = {
        row["outer_target_center"]: row["selected_representation"]
        for row in decisions
    }
    result_keys = [(row.get("heldout_center"), row.get("role")) for row in results]
    audit_keys = [(row.get("heldout_center"), row.get("role")) for row in fit_audit]
    if len(result_keys) != len(set(result_keys)) or len(audit_keys) != len(
        set(audit_keys)
    ):
        raise ProtocolError("Locked outer result/audit rows are duplicated.")
    for center in config.heldout_centers:
        center_results = [row for row in results if row["heldout_center"] == center]
        center_audits = [row for row in fit_audit if row["heldout_center"] == center]
        if (
            {row["role"] for row in center_results}
            != {"canonical_a", "selected_policy"}
            or {row["role"] for row in center_audits}
            != {"canonical_a", "selected_policy"}
        ):
            raise ProtocolError(f"Locked outer roles failed for center {center}.")
        result_by_role = {row["role"]: row for row in center_results}
        if (
            result_by_role["canonical_a"]["representation_id"] != "canonical_a"
            or result_by_role["selected_policy"]["representation_id"] != selected[center]
        ):
            raise ProtocolError(f"Locked outer representation drifted for center {center}.")
        center_predictions = [
            row for row in predictions if row["heldout_center"] == center
        ]
        prediction_roles = {row["role"] for row in center_predictions}
        prediction_keys = [
            (row.get("sample_id"), row.get("role")) for row in center_predictions
        ]
        if (
            prediction_roles != {"canonical_a", "selected_policy"}
            or len(prediction_keys) != len(set(prediction_keys))
        ):
            raise ProtocolError(f"Locked outer predictions are incomplete for center {center}.")
        expected_posthoc = set(config.representation_order).difference(
            {"canonical_a", selected[center]}
        )
        center_posthoc = [row for row in posthoc if row["heldout_center"] == center]
        observed_posthoc = {row["representation_id"] for row in center_posthoc}
        if (
            observed_posthoc != expected_posthoc
            or len(center_posthoc) != len(expected_posthoc)
        ):
            raise ProtocolError(f"Posthoc candidate coverage failed for center {center}.")


def _validate_outer_metrics_and_canonical_reference(
    config: PhysicalMultiscalePilotConfig,
    results: Sequence[Mapping[str, str]],
    predictions: Sequence[Mapping[str, str]],
    replay: Sequence[Mapping[str, str]],
) -> None:
    store = CenterShardedRepresentationStore(
        b_cache_root=config.b_cache_root,
        c_cache_root=config.c_cache_root,
        profile=config.profile,
    )
    reference_results = _unique_by(
        _csv(
            config.canonical_reference_root
            / "tables"
            / "classifier_tuned_source_results.csv"
        ),
        "heldout_center",
    )
    reference_prediction_rows = _csv(
        config.canonical_reference_root
        / "tables"
        / "classifier_tuned_predictions.csv"
    )
    replay_by_center = _unique_by(replay, "heldout_center")
    for center in config.heldout_centers:
        target = store.outer_frame(center)
        expected_identity = {
            sample_id: (case_id, int(target.labels[index]))
            for index, (sample_id, case_id) in enumerate(
                zip(target.sample_ids, target.case_ids, strict=True)
            )
        }
        for role in ("canonical_a", "selected_policy"):
            role_predictions = [
                row
                for row in predictions
                if row["heldout_center"] == center and row["role"] == role
            ]
            keyed = {row["sample_id"]: row for row in role_predictions}
            if set(keyed) != set(expected_identity):
                raise ProtocolError(
                    f"Outer prediction sample coverage failed for center {center}."
                )
            if any(
                row.get("case_id") != expected_identity[sample_id][0]
                or int(row.get("label", -1)) != expected_identity[sample_id][1]
                or row.get("target_labels_used_for_scoring_only") != "True"
                for sample_id, row in keyed.items()
            ):
                raise ProtocolError(
                    f"Outer prediction target identity failed for center {center}."
                )
            result = next(
                row
                for row in results
                if row["heldout_center"] == center and row["role"] == role
            )
            labels = [int(row["label"]) for row in role_predictions]
            predicted = [int(row["prediction"]) for row in role_predictions]
            if (
                not _float_equal(result.get("bacc"), balanced_accuracy(labels, predicted))
                or not _float_equal(result.get("macro_f1"), macro_f1(labels, predicted))
                or int(result.get("n_eval", -1)) != len(target.sample_ids)
            ):
                raise ProtocolError(
                    f"Outer result metrics failed reconstruction for center {center}."
                )
        canonical_rows = [
            row
            for row in predictions
            if row["heldout_center"] == center and row["role"] == "canonical_a"
        ]
        actual_canonical = {
            row["sample_id"]: int(row["prediction"]) for row in canonical_rows
        }
        expected_canonical = {
            row["sample_id"]: int(float(row["y_pred"]))
            for row in reference_prediction_rows
            if row["heldout_center"] == center
        }
        canonical_result = next(
            row
            for row in results
            if row["heldout_center"] == center and row["role"] == "canonical_a"
        )
        reference_result = reference_results.get(center)
        replay_row = replay_by_center.get(center)
        if (
            reference_result is None
            or replay_row is None
            or actual_canonical != expected_canonical
            or canonical_result.get("classifier_config_hash")
            != reference_result.get("selected_classifier_config_hash")
            or not _float_equal(
                canonical_result.get("bacc"), reference_result.get("heldout_bacc")
            )
            or not _float_equal(
                canonical_result.get("macro_f1"),
                reference_result.get("heldout_macro_f1"),
            )
            or replay_row.get("decision_hash")
            != canonical_result.get("decision_hash")
        ):
            raise ProtocolError(
                f"Canonical A reference failed independent replay for center {center}."
            )


def _validate_content_index(root: Path) -> None:
    payload = _json(root / "manifests" / "content_index.json")
    files = payload.get("files")
    if not isinstance(files, list):
        raise ProtocolError("Content index files must be a list.")
    indexed_paths: list[str] = []
    for row in files:
        if not isinstance(row, Mapping):
            raise ProtocolError("Content index row must be a mapping.")
        relative = str(row["path"])
        indexed_paths.append(relative)
        member = root / relative
        if (
            not member.is_file()
            or member.stat().st_size != int(row["size_bytes"])
            or _sha256(member) != str(row["sha256"])
        ):
            raise ProtocolError(f"Content index hash mismatch: {member}")
    actual_paths = {
        str(member.relative_to(root))
        for member in root.rglob("*")
        if member.is_file() and member != root / "manifests" / "content_index.json"
    }
    if (
        len(indexed_paths) != len(set(indexed_paths))
        or set(indexed_paths) != actual_paths
        or stable_hash(files) != payload.get("content_hash")
    ):
        raise ProtocolError("Content-index completeness or semantic hash mismatch.")


def _unique_by(
    rows: Sequence[Mapping[str, str]], *keys: str
) -> dict[object, Mapping[str, str]]:
    out: dict[object, Mapping[str, str]] = {}
    for row in rows:
        key_values = tuple(row.get(key, "") for key in keys)
        key: object = key_values[0] if len(key_values) == 1 else key_values
        if key in out:
            raise ProtocolError(f"Duplicate table key: {key!r}")
        out[key] = row
    return out


def _float_equal(value: object, expected: object) -> bool:
    try:
        return math.isclose(
            float(value), float(expected), rel_tol=0.0, abs_tol=1.0e-12
        )
    except (TypeError, ValueError):
        return False


def _json(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ProtocolError(f"Expected JSON object: {path}")
    return payload


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
