"""Independent reconstruction of the local marginal-utility artifact.

The validator reopens labels only through the already-persisted global
prediction capability.  It then rebuilds the paired utility surface, strict
domain-role LOQDO models, uncertainty, and unscored optimizer plans.  Numeric
results are compared with tight tolerances; identities, candidate pools,
allocations, hashes, and claim-boundary flags remain exact.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .bundle import (
    CONTENT_INDEX_MEMBERS,
    REQUIRED_FILES,
    assert_non_adoptive_payload,
    perturbation_library_payload,
)
from .config import (
    LocalMarginalUtilityRouterConfig,
    load_local_marginal_utility_router_config,
)
from .contracts import (
    CENTERS,
    EXPECTED_DEVELOPMENT_CLASSIFIER_FIT_COUNT,
    EXPECTED_MARGINAL_UTILITY_ROW_COUNT,
    EXPERIMENT_ID,
    GENERATION_SEEDS,
    PERTURBATION_LIBRARY_HASH,
    PUBLICATION_STATUS,
    TRAINING_SEEDS,
    ValidationRowIdentity,
    perturbation_library_for,
    row_identity_hash,
)
from .label_access import open_globally_sealed_development_labels
from .modeling import fit_models_and_build_unscored_target_plans
from .prediction_io import (
    DEVELOPMENT_ARRAY_MEMBER,
    read_prediction_store,
    sha256_file,
)
from .seals import (
    GLOBAL_DEVELOPMENT_SEAL_STATUS,
    GlobalDevelopmentPredictionSeal,
    PredictionCellSeal,
    expected_prediction_keys,
)
from .utility_surface import (
    build_paired_marginal_utility_rows,
    score_sealed_development_predictions,
)


def validate_local_marginal_utility_router_bundle(
    artifact_root: str | Path,
    *,
    config: LocalMarginalUtilityRouterConfig | None = None,
    allow_pending: bool = False,
) -> dict[str, object]:
    """Validate bytes, protocol geometry, outcomes, model, and claim firewall."""

    root = Path(artifact_root).resolve()
    if not root.is_dir():
        raise ProtocolError(f"Local-utility artifact root is missing: {root}.")
    resolved_config = config or load_local_marginal_utility_router_config(
        root / "config.resolved.yaml"
    )
    _validate_file_set(root, allow_pending=allow_pending)
    _validate_content_index(root)

    protocol = _read_json(root / "manifests/protocol_manifest.json")
    if (
        protocol.get("experiment_id") != EXPERIMENT_ID
        or protocol.get("config_contract_hash") != resolved_config.contract_hash
        or protocol.get("validation_manifest_sha256")
        != resolved_config.expected_manifest_sha256
        or protocol.get("model") != dict(resolved_config.model)
        or protocol.get("optimizer") != dict(resolved_config.optimizer)
    ):
        raise ProtocolError("Local-utility protocol manifest/config binding drifted.")
    _assert_stable_hash(protocol, "protocol_hash")
    assert_non_adoptive_payload(protocol)
    if _read_json(root / "manifests/perturbation_library.json") != (
        perturbation_library_payload(resolved_config)
    ):
        raise ProtocolError("Local-utility perturbation library drifted.")

    partition_rows = _read_csv(root / "tables/support_partitions.csv")
    rows_by_role, support_lock = _reconstruct_partitions(
        partition_rows,
        root=root,
        config=resolved_config,
    )
    compatibility_scores = _validate_compatibility(root, config=resolved_config)

    index_rows = _read_csv(root / "tables/development_prediction_index.csv")
    if len(index_rows) != EXPECTED_DEVELOPMENT_CLASSIFIER_FIT_COUNT:
        raise ProtocolError("Local-utility prediction-index row count drifted.")
    store = read_prediction_store(root / DEVELOPMENT_ARRAY_MEMBER, index_rows)
    seal = _reconstruct_global_seal(
        root=root,
        config=resolved_config,
        index_rows=index_rows,
        evaluation_rows_by_query={
            center: rows_by_role[center]["evaluation"] for center in CENTERS
        },
    )
    if seal.support_partition_lock_hash != support_lock["support_partition_lock_hash"]:
        raise ProtocolError("Local-utility seal/partition binding drifted.")

    opened = open_globally_sealed_development_labels(
        resolved_config.validation_manifest_path,
        {
            center: rows_by_role[center]["evaluation"] for center in CENTERS
        },
        seal=seal,
        seal_path=root / "manifests/global_development_prediction_seal.json",
        prediction_index_path=root / "tables/development_prediction_index.csv",
        prediction_arrays_path=root / DEVELOPMENT_ARRAY_MEMBER,
        expected_manifest_sha256=resolved_config.expected_manifest_sha256,
    )
    recomputed_metrics: list[Mapping[str, object]] = []
    for outer_target in CENTERS:
        labels = {
            row.sample_id: label
            for query, vector in opened.items()
            if query != outer_target
            for row, label in zip(vector.rows, vector.labels, strict=True)
        }
        recomputed_metrics.extend(
            score_sealed_development_predictions(
                store,
                labels_by_sample_id=labels,
                outer_target=outer_target,
            )
        )
    observed_metrics = _read_csv(root / "tables/development_metrics.csv")
    _assert_rows_semantically_equal(
        observed_metrics,
        recomputed_metrics,
        role="development metrics",
    )
    recomputed_marginals = build_paired_marginal_utility_rows(
        recomputed_metrics,
        epsilon=float(resolved_config.perturbations["epsilon"]),
    )
    if len(recomputed_marginals) != EXPECTED_MARGINAL_UTILITY_ROW_COUNT:
        raise ProtocolError("Local-utility reconstructed marginal coverage drifted.")
    observed_marginals = _read_csv(root / "tables/marginal_utilities.csv")
    _assert_rows_semantically_equal(
        observed_marginals,
        recomputed_marginals,
        role="marginal utilities",
    )

    learned = fit_models_and_build_unscored_target_plans(
        calibrated_energy_by_query=compatibility_scores,
        marginal_utility_rows=recomputed_marginals,
        alpha_grid=tuple(
            float(value) for value in resolved_config.model["ridge_alpha_grid"]
        ),
        kappa=float(resolved_config.optimizer["kappa"]),
        l2_penalty=float(resolved_config.optimizer["l2_penalty"]),
    )
    for member, expected, role in (
        (
            "tables/loqdo_predictions.csv",
            learned.learnability_prediction_rows,
            "LOQDO predictions",
        ),
        (
            "tables/loqdo_summary.csv",
            learned.learnability_summary_rows,
            "LOQDO summaries",
        ),
        ("tables/model_fits.csv", learned.model_fit_rows, "model fits"),
        ("tables/target_plans.csv", learned.target_plan_rows, "target plans"),
    ):
        _assert_rows_semantically_equal(
            _read_csv(root / member),
            expected,
            role=role,
            relative_tolerance=2e-9,
            absolute_tolerance=2e-11,
        )
    _validate_target_plan_constraints(learned.target_plan_rows)
    _validate_reports(root, config=resolved_config)

    if not allow_pending:
        validation_report = _read_json(root / "reports/validation_report.json")
        if (
            validation_report.get("status") != "PASS"
            or validation_report.get("validator")
            != "validate_local_marginal_utility_router_bundle"
        ):
            raise ProtocolError("Local-utility validation report is not PASS.")
    return {
        "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "config_contract_hash": resolved_config.contract_hash,
        "prediction_cell_count": len(index_rows),
        "marginal_utility_row_count": len(recomputed_marginals),
        "loqdo_prediction_row_count": len(learned.learnability_prediction_rows),
        "loqdo_summary_row_count": len(learned.learnability_summary_rows),
        "target_plan_row_count": len(learned.target_plan_rows),
        "global_prediction_seal_hash": seal.seal_hash,
        "labels_reopened_only_through_global_seal": True,
        "model_and_optimizer_independently_reconstructed": True,
        "target_performance_scored": False,
        "publication_status": PUBLICATION_STATUS,
    }


def _reconstruct_partitions(
    rows: Sequence[Mapping[str, str]],
    *,
    root: Path,
    config: LocalMarginalUtilityRouterConfig,
) -> tuple[
    dict[str, dict[str, tuple[ValidationRowIdentity, ...]]],
    dict[str, object],
]:
    by_center: dict[str, dict[str, list[ValidationRowIdentity]]] = {
        center: {"support": [], "evaluation": []} for center in CENTERS
    }
    ordinals: set[int] = set()
    samples: set[str] = set()
    for raw in rows:
        center = str(raw.get("center", ""))
        role = str(raw.get("partition_role", ""))
        if center not in by_center or role not in {"support", "evaluation"}:
            raise ProtocolError("Local-utility partition row role/center drifted.")
        if not _false(raw.get("label_present")):
            raise ProtocolError("Local-utility partition table persisted a label.")
        row = ValidationRowIdentity(
            row_ordinal=_integer(raw.get("row_ordinal"), "partition row ordinal"),
            manifest_row_index=_integer(
                raw.get("manifest_row_index"), "manifest row index"
            ),
            sample_id=str(raw.get("sample_id", "")),
            case_id=str(raw.get("case_id", "")),
            center=center,
            split=str(raw.get("split", "")),
            partition_role=role,
        )
        if row.row_ordinal in ordinals or row.sample_id in samples:
            raise ProtocolError("Local-utility partition identities duplicate.")
        ordinals.add(row.row_ordinal)
        samples.add(row.sample_id)
        by_center[center][role].append(row)
    if ordinals != set(range(len(rows))):
        raise ProtocolError("Local-utility partition ordinals are not contiguous.")
    normalized: dict[str, dict[str, tuple[ValidationRowIdentity, ...]]] = {}
    for center in CENTERS:
        support = tuple(by_center[center]["support"])
        evaluation = tuple(by_center[center]["evaluation"])
        if not support or not evaluation:
            raise ProtocolError("Local-utility partition lacks support/evaluation rows.")
        if {row.case_id for row in support}.intersection(
            row.case_id for row in evaluation
        ):
            raise ProtocolError("Local-utility support/evaluation cases overlap.")
        normalized[center] = {"support": support, "evaluation": evaluation}

    lock = _read_json(root / "manifests/support_partition_lock.json")
    _assert_stable_hash(lock, "support_partition_lock_hash")
    if (
        lock.get("config_contract_hash") != config.contract_hash
        or lock.get("labels_used") is not False
        or lock.get("support_evaluation_case_disjoint") is not True
        or lock.get("support_evaluation_sample_disjoint") is not True
    ):
        raise ProtocolError("Local-utility support partition lock drifted.")
    centers_payload = lock.get("centers")
    if not isinstance(centers_payload, Mapping) or tuple(centers_payload) != CENTERS:
        raise ProtocolError("Local-utility partition lock center coverage drifted.")
    for center in CENTERS:
        payload = centers_payload[center]
        if not isinstance(payload, Mapping) or (
            payload.get("support_row_identity_hash")
            != row_identity_hash(normalized[center]["support"])
            or payload.get("evaluation_row_identity_hash")
            != row_identity_hash(normalized[center]["evaluation"])
        ):
            raise ProtocolError("Local-utility partition row hash drifted.")
    return normalized, lock


def _validate_compatibility(
    root: Path,
    *,
    config: LocalMarginalUtilityRouterConfig,
) -> dict[str, dict[str, float]]:
    case_rows = _read_csv(root / "tables/compatibility_case_energy.csv")
    score_rows = _read_csv(root / "tables/compatibility_scores.csv")
    expected_score_keys = {
        (query, source)
        for query in CENTERS
        for source in CENTERS
        if source != query
    }
    observed: dict[tuple[str, str], float] = {}
    for row in score_rows:
        key = (str(row.get("query_center", "")), str(row.get("source_center", "")))
        if key in observed or key not in expected_score_keys:
            raise ProtocolError("Local-utility compatibility score geometry drifted.")
        for field in (
            "training_seed_17_z",
            "training_seed_42_z",
            "training_seed_101_z",
            "mean_calibrated_energy_z",
        ):
            _finite(row.get(field), f"compatibility {field}")
        if not _false(row.get("query_support_labels_used")) or not _false(
            row.get("exact_nelbo_claimed")
        ):
            raise ProtocolError("Local-utility compatibility claim boundary drifted.")
        observed[key] = float(row["mean_calibrated_energy_z"])
    if set(observed) != expected_score_keys:
        raise ProtocolError("Local-utility compatibility scores are incomplete.")
    for row in case_rows:
        if not _false(row.get("labels_used")) or not _false(
            row.get("exact_nelbo_claimed")
        ):
            raise ProtocolError("Local-utility compatibility case used labels.")
        _finite(row.get("marginal_variational_energy"), "case compatibility")

    index = _read_json(root / "manifests/compatibility_index.json")
    _assert_stable_hash(index, "compatibility_index_hash")
    if (
        index.get("config_contract_hash") != config.contract_hash
        or index.get("case_energy_sha256")
        != sha256_file(root / "tables/compatibility_case_energy.csv")
        or index.get("score_sha256")
        != sha256_file(root / "tables/compatibility_scores.csv")
        or int(index.get("case_energy_row_count", -1)) != len(case_rows)
        or int(index.get("score_row_count", -1)) != len(score_rows)
    ):
        raise ProtocolError("Local-utility compatibility index drifted.")
    return {
        query: {
            source: observed[(query, source)]
            for source in CENTERS
            if source != query
        }
        for query in CENTERS
    }


def _reconstruct_global_seal(
    *,
    root: Path,
    config: LocalMarginalUtilityRouterConfig,
    index_rows: Sequence[Mapping[str, str]],
    evaluation_rows_by_query: Mapping[str, Sequence[ValidationRowIdentity]],
) -> GlobalDevelopmentPredictionSeal:
    payload = _read_json(root / "manifests/global_development_prediction_seal.json")
    if payload.get("status") != GLOBAL_DEVELOPMENT_SEAL_STATUS:
        raise ProtocolError("Local-utility global seal is incomplete.")
    by_key: dict[tuple[str, str, str, int, int], PredictionCellSeal] = {}
    expected_ids_by_query = {
        query: tuple(row.sample_id for row in evaluation_rows_by_query[query])
        for query in CENTERS
    }
    for row in index_rows:
        outer = str(row.get("outer_target", ""))
        query = str(row.get("query_center", ""))
        action_id = str(row.get("action_id", ""))
        specs = {
            spec.action_id: spec
            for spec in perturbation_library_for(
                outer_target=outer,
                query_center=query,
            )
        }
        if action_id not in specs:
            raise ProtocolError("Local-utility index action is not predeclared.")
        spec = specs[action_id]
        candidates = tuple(_json_list(row.get("candidate_sources_json"), str))
        weights = _json_mapping(row.get("weights_json"), float)
        allocations = _json_mapping(row.get("allocations_json"), int)
        if (
            candidates != spec.candidate_sources
            or not _nested_semantic_equal(weights, dict(spec.weights), 0.0, 1e-15)
            or allocations != dict(spec.allocations)
            or float(row.get("epsilon", "nan")) != 0.125
            or not math.isclose(
                float(row.get("effective_source_count", "nan")),
                spec.effective_source_count,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not _false(row.get("labels_available_to_fit_or_predict"))
            or not _false(row.get("seed_selection_performed"))
            or str(row.get("phase")) != "development_utility_surface"
            or str(row.get("classifier_converged", "")).lower() != "true"
            or _json_list(row.get("classifier_classes_json"), int) != [0, 1]
        ):
            raise ProtocolError("Local-utility prediction perturbation drifted.")
        eval_ids = tuple(_json_list(row.get("evaluation_row_ids_json"), str))
        if eval_ids != expected_ids_by_query[query]:
            raise ProtocolError("Local-utility prediction evaluation rows drifted.")
        cell = PredictionCellSeal(
            outer_target=outer,
            query_center=query,
            action_id=action_id,
            arm_role=str(row.get("arm_role", "")),
            boosted_source=(
                str(row.get("boosted_source"))
                if str(row.get("boosted_source", ""))
                else None
            ),
            candidate_sources=candidates,
            training_seed=_integer(row.get("training_seed"), "training seed"),
            generation_seed=_integer(row.get("generation_seed"), "generation seed"),
            evaluation_row_ids=eval_ids,
            evaluation_row_identity_hash=str(row.get("evaluation_row_identity_hash", "")),
            perturbation_hash=stable_hash(spec.to_payload()),
            prediction_sha256=str(row.get("prediction_sha256", "")),
            probability_sha256=str(row.get("probability_sha256", "")),
            composition_hash=str(row.get("composition_hash", "")),
            classifier_config_hash=str(row.get("classifier_config_hash", "")),
        )
        if cell.key in by_key:
            raise ProtocolError("Local-utility prediction index duplicates a cell.")
        by_key[cell.key] = cell
    expected_keys = expected_prediction_keys()
    if set(by_key) != set(expected_keys):
        raise ProtocolError("Local-utility prediction index coverage drifted.")
    seal = GlobalDevelopmentPredictionSeal(
        config_contract_hash=str(payload.get("config_contract_hash", "")),
        perturbation_library_hash=str(payload.get("perturbation_library_hash", "")),
        support_partition_lock_hash=str(payload.get("support_partition_lock_hash", "")),
        compatibility_index_hash=str(payload.get("compatibility_index_hash", "")),
        validation_cache_binding_hash=str(payload.get("validation_cache_binding_hash", "")),
        validation_manifest_sha256=str(payload.get("validation_manifest_sha256", "")),
        prediction_index_sha256=str(payload.get("prediction_index_sha256", "")),
        prediction_arrays_sha256=str(payload.get("prediction_arrays_sha256", "")),
        evaluation_row_ids_by_query={
            query: tuple(_payload_query_values(payload, "evaluation_row_ids_by_query", query))
            for query in CENTERS
        },
        evaluation_row_identity_hash_by_query={
            query: str(
                _payload_query_value(
                    payload, "evaluation_row_identity_hash_by_query", query
                )
            )
            for query in CENTERS
        },
        cells=tuple(by_key[key] for key in expected_keys),
        seal_hash=str(payload.get("seal_hash", "")),
        status=str(payload.get("status", "")),
    )
    if (
        seal.to_payload() != payload
        or seal.config_contract_hash != config.contract_hash
        or seal.perturbation_library_hash != PERTURBATION_LIBRARY_HASH
        or seal.prediction_index_sha256
        != sha256_file(root / "tables/development_prediction_index.csv")
        or seal.prediction_arrays_sha256 != sha256_file(root / DEVELOPMENT_ARRAY_MEMBER)
        or seal.validation_manifest_sha256 != config.expected_manifest_sha256
    ):
        raise ProtocolError("Local-utility global seal byte binding drifted.")
    return seal


def _validate_target_plan_constraints(rows: Sequence[Mapping[str, object]]) -> None:
    if tuple(str(row["target_center"]) for row in rows) != CENTERS:
        raise ProtocolError("Local-utility target plan center coverage drifted.")
    for row in rows:
        weights = _json_mapping(row["weights_json"], float)
        allocations = _json_mapping(row["allocations_per_class_json"], int)
        values = tuple(float(value) for value in weights.values())
        effective = 1.0 / sum(value * value for value in values)
        if (
            not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-8)
            or min(values) < -1e-10
            or max(values) > 0.25 + 1e-8
            or effective < 6.0 - 1e-7
            or sum(int(value) for value in allocations.values()) != 1024
            or row.get("target_labels_used") is not False
            or row.get("target_performance_scored") is not False
            or row.get("oracle_eligible") is not False
            or row.get("may_feed_stage60") is not False
            or row.get("may_feed_stage70") is not False
        ):
            raise ProtocolError("Local-utility target plan violates its constraints.")


def _validate_reports(
    root: Path,
    *,
    config: LocalMarginalUtilityRouterConfig,
) -> None:
    state = _read_json(root / "reports/run_state.json")
    publication = _read_json(root / "reports/publication_decision.json")
    leakage = _read_json(root / "reports/leakage_report.json")
    label_access = _read_json(root / "reports/label_access_report.json")
    if (
        state.get("status") != "COMPLETE"
        or publication.get("decision") != PUBLICATION_STATUS
        or leakage.get("status") != "PASS"
        or leakage.get("target_H_labels_used_for_target_plan") is not False
        or leakage.get("seed_selection_performed") is not False
        or label_access.get("development_labels_opened_after_global_prediction_seal")
        is not True
        or label_access.get("target_labels_opened_for_target_scoring") is not False
    ):
        raise ProtocolError("Local-utility report state or firewall drifted.")
    for member in (
        "manifests/protocol_manifest.json",
        "manifests/perturbation_library.json",
        "reports/phase_01_support_and_compatibility_complete.json",
        "reports/phase_02_global_predictions_sealed.json",
        "reports/phase_03_utility_surface_complete.json",
        "reports/phase_04_model_and_plans_complete.json",
        "reports/label_access_report.json",
        "reports/leakage_report.json",
        "reports/publication_decision.json",
        "reports/run_state.json",
    ):
        assert_non_adoptive_payload(_read_json(root / member))
    phase_04 = _read_json(root / "reports/phase_04_model_and_plans_complete.json")
    expected_hashes = {
        "loqdo_predictions_sha256": "tables/loqdo_predictions.csv",
        "loqdo_summary_sha256": "tables/loqdo_summary.csv",
        "model_fits_sha256": "tables/model_fits.csv",
        "target_plans_sha256": "tables/target_plans.csv",
        "learnability_report_sha256": "reports/learnability_report.json",
        "optimizer_report_sha256": "reports/optimizer_report.json",
    }
    if any(
        phase_04.get(field) != sha256_file(root / member)
        for field, member in expected_hashes.items()
    ):
        raise ProtocolError("Local-utility Phase-04 member hashes drifted.")
    for member in ("reports/learnability_report.json", "reports/optimizer_report.json"):
        report = _read_json(root / member)
        _assert_stable_hash(report, "report_hash")
        if (
            report.get("routing_quality_claimed") is not False
            or report.get("target_performance_scored") is not False
            or report.get("may_feed_stage60") is not False
            or report.get("may_feed_stage70") is not False
        ):
            raise ProtocolError("Local-utility diagnostic report became adoptive.")
    if config.claim_boundary.get("diagnostic_only") is not True:
        raise ProtocolError("Local-utility config is not diagnostic-only.")


def _validate_content_index(root: Path) -> None:
    payload = _read_json(root / "manifests/content_index.json")
    _assert_stable_hash(payload, "content_hash")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != len(CONTENT_INDEX_MEMBERS):
        raise ProtocolError("Local-utility content index coverage drifted.")
    by_path = {
        str(record.get("relative_path", "")): record
        for record in records
        if isinstance(record, Mapping)
    }
    if tuple(by_path) != CONTENT_INDEX_MEMBERS:
        raise ProtocolError("Local-utility content index member order drifted.")
    for relative in CONTENT_INDEX_MEMBERS:
        member = root / relative
        record = by_path[relative]
        if (
            record.get("sha256") != sha256_file(member)
            or int(record.get("size_bytes", -1)) != member.stat().st_size
        ):
            raise ProtocolError(f"Local-utility content member drifted: {relative}.")


def _validate_file_set(root: Path, *, allow_pending: bool) -> None:
    actual = {
        member.relative_to(root).as_posix()
        for member in root.rglob("*")
        if member.is_file()
    }
    required = set(REQUIRED_FILES)
    if allow_pending:
        required.remove("reports/validation_report.json")
    missing = sorted(required.difference(actual))
    unexpected = sorted(actual.difference(REQUIRED_FILES))
    if missing or unexpected:
        raise ProtocolError(
            "Local-utility closed-world file set drifted: "
            f"missing={missing!r}, unexpected={unexpected!r}."
        )


def _assert_rows_semantically_equal(
    observed: Sequence[Mapping[str, str]],
    expected: Sequence[Mapping[str, object]],
    *,
    role: str,
    relative_tolerance: float = 5e-12,
    absolute_tolerance: float = 5e-13,
) -> None:
    if len(observed) != len(expected):
        raise ProtocolError(
            f"Local-utility {role} row count drifted: {len(observed)} != {len(expected)}."
        )
    for index, (raw, rebuilt) in enumerate(zip(observed, expected, strict=True)):
        if set(raw) != set(rebuilt):
            raise ProtocolError(f"Local-utility {role} columns drifted at row {index}.")
        for key, expected_value in rebuilt.items():
            if not _csv_semantic_equal(
                raw[key],
                expected_value,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            ):
                raise ProtocolError(
                    f"Local-utility {role} drifted at row {index}, field {key!r}: "
                    f"observed={raw[key]!r}, expected={expected_value!r}."
                )


def _csv_semantic_equal(
    observed: str,
    expected: object,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> bool:
    if expected is None:
        return observed == ""
    if isinstance(expected, bool):
        return observed.lower() == str(expected).lower()
    if isinstance(expected, int):
        try:
            return int(observed) == expected
        except ValueError:
            return False
    if isinstance(expected, float):
        try:
            value = float(observed)
        except ValueError:
            return False
        return math.isfinite(value) and math.isclose(
            value,
            expected,
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        )
    if isinstance(expected, str) and _looks_like_json(expected):
        try:
            return _nested_semantic_equal(
                json.loads(observed),
                json.loads(expected),
                relative_tolerance,
                absolute_tolerance,
            )
        except json.JSONDecodeError:
            return False
    return observed == str(expected)


def _nested_semantic_equal(
    observed: object,
    expected: object,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> bool:
    if isinstance(expected, Mapping):
        return isinstance(observed, Mapping) and set(observed) == set(expected) and all(
            _nested_semantic_equal(
                observed[key], expected[key], relative_tolerance, absolute_tolerance
            )
            for key in expected
        )
    if isinstance(expected, list):
        return isinstance(observed, list) and len(observed) == len(expected) and all(
            _nested_semantic_equal(a, b, relative_tolerance, absolute_tolerance)
            for a, b in zip(observed, expected, strict=True)
        )
    if isinstance(expected, bool):
        return observed is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            value = float(observed)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False
        return math.isfinite(value) and math.isclose(
            value,
            float(expected),
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        )
    return observed == expected


def _assert_stable_hash(payload: Mapping[str, object], field: str) -> None:
    observed = payload.get(field)
    unhashed = {key: value for key, value in payload.items() if key != field}
    if observed != stable_hash(unhashed):
        raise ProtocolError(f"Local-utility {field} drifted.")


def _payload_query_values(
    payload: Mapping[str, object], field: str, query: str
) -> list[str]:
    value = _payload_query_value(payload, field, query)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ProtocolError(f"Local-utility seal {field} is malformed.")
    return value


def _payload_query_value(
    payload: Mapping[str, object], field: str, query: str
) -> object:
    mapping = payload.get(field)
    if not isinstance(mapping, Mapping) or tuple(mapping) != CENTERS:
        raise ProtocolError(f"Local-utility seal {field} coverage drifted.")
    return mapping[query]


def _json_list(value: object, item_type: type) -> list[object]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ProtocolError("Local-utility CSV JSON list is malformed.") from exc
    if not isinstance(parsed, list) or any(not isinstance(item, item_type) for item in parsed):
        raise ProtocolError("Local-utility CSV JSON list geometry drifted.")
    return parsed


def _json_mapping(value: object, cast: type) -> dict[str, object]:
    if isinstance(value, Mapping):
        parsed = value
    else:
        try:
            parsed = json.loads(str(value))
        except json.JSONDecodeError as exc:
            raise ProtocolError("Local-utility CSV JSON mapping is malformed.") from exc
    if not isinstance(parsed, Mapping):
        raise ProtocolError("Local-utility CSV JSON mapping geometry drifted.")
    try:
        return {str(key): cast(item) for key, item in parsed.items()}
    except (TypeError, ValueError) as exc:
        raise ProtocolError("Local-utility CSV JSON mapping values drifted.") from exc


def _looks_like_json(value: str) -> bool:
    stripped = value.strip()
    return bool(stripped) and stripped[0] in "[{" and stripped[-1] in "]}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        raise ProtocolError(f"Cannot read local-utility CSV: {path}.") from exc


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read local-utility JSON: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"Local-utility JSON must be an object: {path}.")
    return payload


def _integer(value: object, role: str) -> int:
    if isinstance(value, bool):
        raise ProtocolError(f"Local-utility {role} must be an integer.")
    try:
        return int(str(value))
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"Local-utility {role} must be an integer.") from exc


def _finite(value: object, role: str) -> float:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"Local-utility {role} must be numeric.") from exc
    if not math.isfinite(parsed):
        raise ProtocolError(f"Local-utility {role} must be finite.")
    return parsed


def _false(value: object) -> bool:
    return value is False or str(value).lower() == "false"


__all__ = ("validate_local_marginal_utility_router_bundle",)
