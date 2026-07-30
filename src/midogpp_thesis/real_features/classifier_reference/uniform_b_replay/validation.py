"""Independent validation for completed retrospective uniform-B bundles."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Mapping

from midogpp_thesis.common.hashing import stable_hash

from ..downstream import balanced_accuracy, macro_f1
from ..protocol import ProtocolError
from .artifacts import (
    REQUIRED_FILES,
    input_hashes,
    paired_case_bootstrap,
    read_csv,
    read_json,
    validate_content_index,
    validate_source_bundle,
)
from .config import CANONICAL_A, UNIFORM_B, UniformBReplayConfig


def validate_uniform_b_replay_bundle(
    root: str | Path,
    *,
    config: UniformBReplayConfig,
    allow_pending: bool = False,
) -> dict[str, object]:
    path = Path(root)
    required = set(REQUIRED_FILES)
    if allow_pending:
        required.remove("reports/validation_report.json")
    missing = sorted(relative for relative in required if not (path / relative).is_file())
    if missing:
        raise ProtocolError(f"Uniform-B replay bundle is incomplete: {missing}")
    validate_source_bundle(config)
    current_hashes = input_hashes(config)
    frozen = read_json(path / "manifests/frozen_protocol_snapshot.json")
    protocol = read_json(path / "manifests/protocol_manifest.json")
    uniform_lock = read_json(path / "manifests/uniform_representation_lock.json")
    source_index = read_json(path / "manifests/source_lock_index.json")
    leakage = read_json(path / "reports/leakage_provenance_report.json")
    expected_status = "PENDING_INDEPENDENT_VALIDATION" if allow_pending else "PASS"
    unhashed_frozen = {key: value for key, value in frozen.items() if key != "protocol_hash"}
    unhashed_uniform = {
        key: value for key, value in uniform_lock.items() if key != "uniform_lock_hash"
    }
    unhashed_source = {
        key: value
        for key, value in source_index.items()
        if key != "source_lock_bundle_hash"
    }
    if (
        protocol.get("status") != expected_status
        or protocol.get("claim_scope") != "diagnostic_only"
        or protocol.get("evidence_role") != "retrospective_uniform_representation_replay"
        or protocol.get("study_design_informed_by_prior_target_scores") is not True
        or protocol.get("selection_performed_this_run") is not False
        or protocol.get("non_adoptive") is not True
        or protocol.get("adoption_eligible") is not False
        or protocol.get("may_feed_recipe_selection") is not False
        or protocol.get("may_feed_deployable_selection") is not False
        or protocol.get("covers_new_center_uncertainty") is not False
        or protocol.get("input_hashes") != current_hashes
        or frozen.get("input_hashes") != current_hashes
        or stable_hash(unhashed_frozen) != frozen.get("protocol_hash")
        or protocol.get("protocol_hash") != frozen.get("protocol_hash")
        or uniform_lock.get("representation_id") != UNIFORM_B
        or uniform_lock.get("choice_informed_by_prior_target_scores") is not True
        or uniform_lock.get("choice_pre_original_outer_scoring") is not False
        or uniform_lock.get("adoption_eligible") is not False
        or stable_hash(unhashed_uniform) != uniform_lock.get("uniform_lock_hash")
        or protocol.get("uniform_lock_hash") != uniform_lock.get("uniform_lock_hash")
        or source_index.get("lock_count") != len(config.heldout_centers)
        or stable_hash(unhashed_source) != source_index.get("source_lock_bundle_hash")
        or protocol.get("source_lock_bundle_hash")
        != source_index.get("source_lock_bundle_hash")
        or leakage.get("fit_leakage_status") != "PASS"
        or leakage.get("study_design_status") != "POSTHOC_DISCOVERY"
        or leakage.get("uniform_choice_is_retrospective") is not True
        or leakage.get("independent_confirmation") is not False
    ):
        raise ProtocolError("Uniform-B protocol/claim boundary validation failed.")
    if not allow_pending and (
        protocol.get("independent_validation_status") != "PASS"
        or leakage.get("status") != "PASS_WITH_POSTHOC_DESIGN"
        or leakage.get("independent_validation_status") != "PASS"
    ):
        raise ProtocolError("Uniform-B final validation status failed.")

    lock_rows = read_csv(path / "tables/source_lock_replay.csv")
    alignment = read_csv(path / "tables/cache_alignment_audit.csv")
    results = read_csv(path / "tables/uniform_b_outer_results.csv")
    predictions = read_csv(path / "tables/uniform_b_outer_predictions.csv")
    comparisons = read_csv(path / "tables/paired_center_comparison.csv")
    audits = read_csv(path / "tables/outer_fit_audit.csv")
    canonical = read_csv(path / "tables/canonical_a_replay.csv")
    source_replay = read_csv(path / "tables/v3_result_replay.csv")
    centers = config.heldout_centers
    if (
        len(lock_rows) != len(centers)
        or len(results) != 2 * len(centers)
        or len(comparisons) != len(centers)
        or len(audits) != 2 * len(centers)
        or len(canonical) != len(centers)
        or len(source_replay) != len(centers)
        or len(alignment) != 1
    ):
        raise ProtocolError("Uniform-B artifact cardinality failed.")
    if (
        alignment[0].get("representation_id") != UNIFORM_B
        or alignment[0].get("feature_dim") != "3840"
        or alignment[0].get("status") != "PASS"
        or alignment[0].get("c_cache_accessed") != "False"
    ):
        raise ProtocolError("Uniform-B cache alignment/C-access firewall failed.")
    if any(
        row.get("target_center_absent_from_fit") != "True"
        or row.get("scaler_fit_on_source_only") != "True"
        or row.get("fresh_outer_fit") != "True"
        for row in audits
    ):
        raise ProtocolError("Uniform-B fit audit failed.")
    if any(row.get("status") != "PASS" for row in canonical + source_replay):
        raise ProtocolError("Uniform-B source/canonical replay failed.")
    if any(
        row.get("canonical_a_metric_exact") != "True"
        or row.get("uniform_b_metric_exact") != "True"
        for row in source_replay
    ):
        raise ProtocolError("Uniform-B v3 metric replay failed.")
    allowed_representations = {CANONICAL_A, UNIFORM_B}
    if any(row.get("representation_id") not in allowed_representations for row in results + predictions):
        raise ProtocolError("Uniform-B result contains C or unknown representation rows.")
    _validate_metrics(centers, results, predictions, comparisons)
    expected_bootstrap = paired_case_bootstrap(predictions, config=config.bootstrap)
    bootstrap = read_json(path / "reports/conditional_bootstrap.json")
    if bootstrap != expected_bootstrap:
        raise ProtocolError("Uniform-B conditional bootstrap failed reconstruction.")
    expected_summary = _summary(config, comparisons, expected_bootstrap)
    summary = read_json(path / "reports/diagnostic_summary.json")
    if summary != expected_summary:
        raise ProtocolError("Uniform-B diagnostic summary failed reconstruction.")
    validate_content_index(path)
    checks = {
        "status": "PASS",
        "source_locks": len(lock_rows),
        "outer_results": len(results),
        "outer_predictions": len(predictions),
        "center_comparisons": len(comparisons),
        "source_replays": len(source_replay),
    }
    if not allow_pending:
        report = read_json(path / "reports/validation_report.json")
        if (
            report.get("status") != "PASS"
            or report.get("validator") != "validate_uniform_b_replay_bundle"
            or report.get("authoritative_bundle_verdict") is not True
            or report.get("checks") != checks
        ):
            raise ProtocolError("Uniform-B validation report failed.")
    return checks


def _validate_metrics(
    centers: tuple[str, ...],
    results: list[dict[str, str]],
    predictions: list[dict[str, str]],
    comparisons: list[dict[str, str]],
) -> None:
    result_by = {(row["heldout_center"], row["role"]): row for row in results}
    comparison_by = {row["heldout_center"]: row for row in comparisons}
    if len(result_by) != len(results) or len(comparison_by) != len(comparisons):
        raise ProtocolError("Uniform-B result/comparison keys are duplicated.")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in predictions:
        grouped[(row["heldout_center"], row["role"])].append(row)
        probability = float(row["probability_positive"])
        if not 0.0 <= probability <= 1.0:
            raise ProtocolError("Uniform-B probability is outside [0,1].")
    for center in centers:
        sample_sets = []
        computed = {}
        for role in ("canonical_a", "uniform_b"):
            rows = grouped[(center, role)]
            if not rows or len({row["sample_id"] for row in rows}) != len(rows):
                raise ProtocolError(f"Uniform-B predictions invalid: {center}/{role}.")
            sample_sets.append(
                {(row["sample_id"], row["case_id"], row["label"]) for row in rows}
            )
            labels = [int(row["label"]) for row in rows]
            pred = [int(row["prediction"]) for row in rows]
            if set(labels) != {0, 1}:
                raise ProtocolError("Uniform-B outer center lacks both classes.")
            computed[role] = (
                balanced_accuracy(labels, pred),
                macro_f1(labels, pred),
            )
            result = result_by.get((center, role))
            if (
                result is None
                or float(result["bacc"]) != computed[role][0]
                or float(result["macro_f1"]) != computed[role][1]
                or int(result["n_eval"]) != len(rows)
            ):
                raise ProtocolError(f"Uniform-B result metric drift: {center}/{role}.")
        if sample_sets[0] != sample_sets[1]:
            raise ProtocolError(f"Uniform-B A/B evaluation pools differ: {center}.")
        comparison = comparison_by.get(center)
        if (
            comparison is None
            or float(comparison["canonical_a_bacc"]) != computed["canonical_a"][0]
            or float(comparison["uniform_b_bacc"]) != computed["uniform_b"][0]
            or float(comparison["delta_bacc"])
            != computed["uniform_b"][0] - computed["canonical_a"][0]
            or float(comparison["delta_macro_f1"])
            != computed["uniform_b"][1] - computed["canonical_a"][1]
            or comparison.get("uniform_choice_is_retrospective") != "True"
        ):
            raise ProtocolError(f"Uniform-B paired comparison drift: {center}.")


def _summary(
    config: UniformBReplayConfig,
    comparisons: list[dict[str, str]],
    bootstrap: Mapping[str, object],
) -> dict[str, object]:
    mean_a = sum(float(row["canonical_a_bacc"]) for row in comparisons) / len(
        comparisons
    )
    mean_b = sum(float(row["uniform_b_bacc"]) for row in comparisons) / len(
        comparisons
    )
    deltas = [float(row["delta_bacc"]) for row in comparisons]
    return {
        "schema_version": "midogpp_uniform_b_diagnostic_summary_v1",
        "status": "COMPLETE",
        "experiment_name": config.name,
        "claim_scope": "diagnostic_only",
        "study_design_status": "POSTHOC_DISCOVERY",
        "independent_confirmation": False,
        "adoption_eligible": False,
        "equal_center_mean_canonical_a_bacc": mean_a,
        "equal_center_mean_uniform_b_bacc": mean_b,
        "paired_mean_delta": mean_b - mean_a,
        "strict_wins": sum(delta > 0.0 for delta in deltas),
        "worst_center_delta": min(deltas),
        "conditional_bootstrap": dict(bootstrap),
        "covers_new_center_uncertainty": False,
    }
