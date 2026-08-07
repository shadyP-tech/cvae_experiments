"""Independent reconstruction of the MMD/KMM diagnostic bundle."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .bundle import CONTENT_INDEX_MEMBERS, REQUIRED_FILES
from .config import MMDKMMRouterDiagnosticConfig, load_mmd_kmm_router_config
from .contracts import (
    CENTERS,
    EXPECTED_PREDICTION_CELL_COUNT,
    EXPECTED_SOURCE_BLOCK_COUNT,
    GENERATION_SEEDS,
    MAX_SOURCE_PREFIX_PER_CLASS,
    SUPPORT_CASE_COUNT,
    TRAINING_SEEDS,
    candidate_sources,
)
from .inputs import (
    build_partition_surface,
    load_label_free_validation_frame,
    load_validated_locks,
    validate_workspace_provenance,
)
from .metrics import score_predictions
from .planning import load_router_plans
from .prediction import (
    TARGET_PREDICTION_ARRAY_MEMBER,
    TARGET_PREDICTION_INDEX_MEMBER,
    read_prediction_store,
    validate_prediction_store_binding,
)
from .seals import open_evaluation_labels, validate_global_prediction_seal
from .source_products import load_source_products, validate_source_products_lock


def validate_mmd_kmm_router_bundle(
    root: str | Path,
    *,
    config: MMDKMMRouterDiagnosticConfig,
    allow_pending: bool = False,
) -> dict[str, object]:
    path = Path(root)
    required = set(REQUIRED_FILES)
    if allow_pending:
        required.remove("reports/validation_report.json")
    missing = sorted(relative for relative in required if not (path / relative).is_file())
    if missing:
        raise ProtocolError(f"MMD/KMM artifact is incomplete: {missing}.")
    _validate_closed_world(path, allow_pending=allow_pending)
    resolved = load_mmd_kmm_router_config(path / "config.resolved.yaml")
    if resolved.contract_hash != config.contract_hash:
        raise ProtocolError("MMD/KMM resolved config contract drifted.")
    provenance = validate_workspace_provenance(path, resolved)
    locks = load_validated_locks(resolved)
    frame = load_label_free_validation_frame(resolved)
    partitions = build_partition_surface(frame, config_contract_hash=resolved.contract_hash)
    if _json(path / "manifests/support_partition_lock.json") != dict(partitions.lock_payload):
        raise ProtocolError("MMD/KMM support partition lock is not reconstructible.")

    protocol = _json(path / "manifests/protocol_manifest.json")
    protocol_unhashed = {key: value for key, value in protocol.items() if key != "protocol_hash"}
    if (
        protocol.get("protocol_hash") != stable_hash(protocol_unhashed)
        or protocol.get("config_contract_hash") != resolved.contract_hash
        or protocol.get("validation_cache_binding_hash") != frame.cache_binding_hash
        or protocol.get("support_partition_lock_hash") != partitions.lock_hash
        or protocol.get("input_artifact_hashes")
        != {
            artifact_id: stable_hash(dict(provenance[artifact_id]))
            for artifact_id in resolved.input_artifact_ids
        }
    ):
        raise ProtocolError("MMD/KMM protocol manifest drifted.")

    source_products = load_source_products(path)
    _validate_source_products(path, source_products, partitions=partitions)
    source_lock = validate_source_products_lock(
        path,
        config=resolved,
        generation_lock=locks.generation,
        frame=frame,
        partitions=partitions,
        source_products=source_products,
    )

    plans = load_router_plans(
        path,
        expected_config_contract_hash=resolved.contract_hash,
        expected_support_partition_lock_hash=partitions.lock_hash,
        expected_source_products_hash=source_products.source_products_hash,
        expected_source_products_lock_hash=str(
            source_lock["source_products_lock_hash"]
        ),
    )
    _validate_plans(plans.plans_by_target)
    predictions = read_prediction_store(
        path / TARGET_PREDICTION_ARRAY_MEMBER,
        path / TARGET_PREDICTION_INDEX_MEMBER,
    )
    if len(predictions.index_rows) != EXPECTED_PREDICTION_CELL_COUNT:
        raise ProtocolError("MMD/KMM prediction coverage drifted.")
    validate_prediction_store_binding(
        predictions,
        config=resolved,
        generation_lock_hash=locks.generation.generation_lock_hash,
        source_products_lock_hash=str(source_lock["source_products_lock_hash"]),
        plans=plans,
        partitions=partitions,
    )
    validate_global_prediction_seal(
        resolved,
        partitions,
        plans,
        predictions,
        root=path,
    )
    labels_by_sample, label_report = open_evaluation_labels(
        resolved,
        partitions,
        root=path,
    )
    stored_label_report = _json(path / "reports/label_access_report.json")
    if stored_label_report != label_report:
        raise ProtocolError("MMD/KMM label-access report is not reconstructible.")
    metrics, deltas, scoring = score_predictions(
        predictions,
        labels_by_sample_id=labels_by_sample,
    )
    _compare_rows(_read_csv(path / "tables/target_metrics.csv"), metrics, "target metrics")
    _compare_rows(_read_csv(path / "tables/paired_deltas.csv"), deltas, "paired deltas")
    stored_scoring = _json(path / "reports/phase_04_scoring_complete.json")
    _require_numeric_mapping_equal(stored_scoring, scoring, "scoring report")

    leakage = _json(path / "reports/leakage_report.json")
    required_leakage = {
        "status": "PASS",
        "target_expert_excluded_from_every_pool": True,
        "support_and_evaluation_case_disjoint": True,
        "support_labels_used": False,
        "evaluation_embeddings_used_for_router": False,
        "evaluation_labels_available_before_global_prediction_seal": False,
        "evaluation_labels_used_for_scoring_only": True,
        "individual_expert_or_seed_selection_performed": False,
        "previous_stage90_router_or_utility_inputs_used": False,
        "routing_quality_claimed": False,
        "promotion_eligible": False,
    }
    if any(leakage.get(key) != value for key, value in required_leakage.items()):
        raise ProtocolError("MMD/KMM leakage report claim boundary drifted.")
    publication = _json(path / "reports/publication_decision.json")
    required_publication = {
        "decision": "PUBLISH_AS_EXPLORATORY_CONSUMED_DATA_DIAGNOSTIC_ONLY",
        "routing_quality_claimed": False,
        "fresh_evidence": False,
        "fresh_confirmation": False,
        "promotion_eligible": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
    }
    if any(publication.get(key) != value for key, value in required_publication.items()):
        raise ProtocolError("MMD/KMM publication decision escaped diagnostic scope.")
    _validate_content_index(path)
    if not allow_pending:
        report = _json(path / "reports/validation_report.json")
        state = _json(path / "reports/run_state.json")
        if report.get("status") != "PASS" or state.get("status") != "COMPLETE":
            raise ProtocolError("MMD/KMM final validation/run state is incomplete.")
    return {
        "status": "PASS",
        "config_contract_hash": resolved.contract_hash,
        "generation_lock_hash": locks.generation.generation_lock_hash,
        "equal_union_policy_lock_hash": locks.equal_union.policy_lock_hash,
        "support_partition_lock_hash": partitions.lock_hash,
        "source_block_count": len(source_products.index_rows),
        "target_plan_count": len(plans.plans_by_target),
        "prediction_cell_count": len(predictions.index_rows),
        "unique_classifier_fit_count": predictions.unique_classifier_fit_count,
        "metric_row_count": len(metrics),
        "paired_delta_row_count": len(deltas),
        "global_prediction_seal_verified_before_label_access": True,
        "content_index_verified": True,
        "closed_world_verified": True,
        "routing_quality_claimed": False,
        "promotion_eligible": False,
    }


def _validate_source_products(
    path: Path,
    source_products: object,
    *,
    partitions: object,
) -> None:
    array = np.load(path / "arrays/source_prefix_blocks.npy", mmap_mode="r")
    if array.shape != (EXPECTED_SOURCE_BLOCK_COUNT, 2 * MAX_SOURCE_PREFIX_PER_CLASS, 3840):
        raise ProtocolError("MMD/KMM source array geometry drifted.")
    labels = np.concatenate(
        (
            np.zeros(MAX_SOURCE_PREFIX_PER_CLASS, dtype=np.int64),
            np.ones(MAX_SOURCE_PREFIX_PER_CLASS, dtype=np.int64),
        )
    )
    expected_keys = tuple(
        (source, training_seed, generation_seed)
        for source in CENTERS
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
    )
    observed_keys = []
    for ordinal, row in enumerate(source_products.index_rows):
        observed_keys.append(
            (
                str(row["source_center"]),
                int(row["training_seed"]),
                int(row["generation_seed"]),
            )
        )
        if (
            int(row["block_ordinal"]) != ordinal
            or int(row["samples_per_class"]) != MAX_SOURCE_PREFIX_PER_CLASS
            or str(row["output_sha256"])
            != _generated_bundle_hash(np.asarray(array[ordinal]), labels)
        ):
            raise ProtocolError("MMD/KMM source block hash or ordinal drifted.")
    if tuple(observed_keys) != expected_keys:
        raise ProtocolError("MMD/KMM source block key coverage drifted.")
    expected_score_keys = tuple(
        (query, source)
        for query in CENTERS
        for source in candidate_sources(query)
    )
    observed_score_keys = tuple(
        (str(row["query_center"]), str(row["source_center"]))
        for row in source_products.compatibility_score_rows
    )
    score_columns = (
        "training_seed_17_z",
        "training_seed_42_z",
        "training_seed_101_z",
        "mean_calibrated_energy_z",
    )
    if observed_score_keys != expected_score_keys or any(
        int(row["query_support_case_count"]) != SUPPORT_CASE_COUNT
        or not _truthy(row["legal_target_candidate"])
        or _truthy(row["query_support_labels_used"])
        or _truthy(row["exact_nelbo_claimed"])
        or not all(np.isfinite(float(row[column])) for column in score_columns)
        for row in source_products.compatibility_score_rows
    ):
        raise ProtocolError("MMD/KMM compatibility-score coverage drifted.")
    expected_case_keys = tuple(
        sorted(
            (
                source,
                training_seed,
                query,
                case_id,
            )
            for source in CENTERS
            for training_seed in TRAINING_SEEDS
            for query in CENTERS
            for case_id in sorted(
                {row.case_id for row in partitions.support_rows_by_center[query]}
            )
        )
    )
    observed_case_keys = tuple(
        (
            str(row["source_center"]),
            int(row["training_seed"]),
            str(row["query_center"]),
            str(row["case_id"]),
        )
        for row in source_products.compatibility_case_rows
    )
    energy_columns = (
        "marginal_variational_energy",
        "class_0_energy",
        "class_1_energy",
        "class_0_common_reconstruction_mse",
        "class_1_common_reconstruction_mse",
        "class_0_normalized_ps_kl",
        "class_1_normalized_ps_kl",
    )
    if observed_case_keys != expected_case_keys or any(
        str(row["query_partition_role"]) != "support"
        or int(row["row_count"]) <= 0
        or str(row["class_prior_json"]) != "[0.5,0.5]"
        or _truthy(row["labels_used"])
        or _truthy(row["exact_nelbo_claimed"])
        or not all(np.isfinite(float(row[column])) for column in energy_columns)
        for row in source_products.compatibility_case_rows
    ):
        raise ProtocolError("MMD/KMM compatibility-case coverage drifted.")


def _validate_plans(plans: Mapping[str, Mapping[str, object]]) -> None:
    if tuple(plans) != CENTERS:
        raise ProtocolError("MMD/KMM plan target coverage drifted.")
    for target in CENTERS:
        plan = plans[target]
        sources = candidate_sources(target)
        weights = {str(key): float(value) for key, value in plan["final_weights"].items()}
        control = {str(key): float(value) for key, value in plan["control_weights"].items()}
        allocations = {str(key): int(value) for key, value in plan["mmd_allocations_per_class"].items()}
        control_allocations = {str(key): int(value) for key, value in plan["control_allocations_per_class"].items()}
        vector = np.asarray([weights[source] for source in sources])
        if (
            tuple(weights) != sources
            or tuple(control) != sources
            or tuple(allocations) != sources
            or tuple(control_allocations) != sources
            or not np.isclose(vector.sum(), 1.0, atol=1e-10, rtol=0.0)
            or float(vector.max()) > 0.25 + 1e-8
            or 1.0 / float(np.dot(vector, vector)) < 6.0 - 1e-7
            or sum(allocations.values()) != 1024
            or max(allocations.values()) > MAX_SOURCE_PREFIX_PER_CLASS
            or set(control_allocations.values()) != {128}
            or set(control.values()) != {0.125}
        ):
            raise ProtocolError(f"MMD/KMM target {target} plan constraints drifted.")
        if bool(plan["used_uniform_fallback"]) and (
            weights != control or allocations != control_allocations
        ):
            raise ProtocolError("MMD/KMM uniform fallback is not exact equal union.")
        unhashed = {key: value for key, value in plan.items() if key != "plan_hash"}
        if plan.get("plan_hash") != stable_hash(unhashed):
            raise ProtocolError("MMD/KMM target plan hash drifted.")


def _validate_content_index(path: Path) -> None:
    payload = _json(path / "manifests/content_index.json")
    unhashed = {key: value for key, value in payload.items() if key != "content_hash"}
    records = payload.get("records")
    if payload.get("content_hash") != stable_hash(unhashed) or not isinstance(records, list):
        raise ProtocolError("MMD/KMM content-index hash drifted.")
    if tuple(str(row.get("relative_path")) for row in records if isinstance(row, Mapping)) != CONTENT_INDEX_MEMBERS:
        raise ProtocolError("MMD/KMM content-index member order drifted.")
    for row in records:
        if not isinstance(row, Mapping):
            raise ProtocolError("MMD/KMM content-index row is malformed.")
        member = path / str(row["relative_path"])
        if (
            not member.is_file()
            or row.get("sha256") != _sha256_file(member)
            or int(row.get("size_bytes", -1)) != member.stat().st_size
        ):
            raise ProtocolError(f"MMD/KMM content member drifted: {member}.")


def _validate_closed_world(path: Path, *, allow_pending: bool) -> None:
    allowed = set(REQUIRED_FILES)
    actual = {
        member.relative_to(path).as_posix()
        for member in path.rglob("*")
        if member.is_file() and member.name != ".run.lock"
    }
    unexpected = sorted(actual.difference(allowed))
    if unexpected:
        raise ProtocolError(f"MMD/KMM artifact is not closed-world: {unexpected}.")


def _compare_rows(
    observed: Sequence[Mapping[str, str]],
    expected: Sequence[Mapping[str, object]],
    role: str,
) -> None:
    if len(observed) != len(expected):
        raise ProtocolError(f"MMD/KMM {role} row count drifted.")
    for left, right in zip(observed, expected, strict=True):
        if set(left) != set(right):
            raise ProtocolError(f"MMD/KMM {role} columns drifted.")
        for key, expected_value in right.items():
            raw = left[key]
            if isinstance(expected_value, bool):
                equal = raw.lower() == str(expected_value).lower()
            elif isinstance(expected_value, (int, float)) and not isinstance(expected_value, bool):
                try:
                    equal = np.isclose(float(raw), float(expected_value), atol=1e-12, rtol=1e-12)
                except ValueError:
                    equal = False
            else:
                equal = raw == str(expected_value)
            if not equal:
                raise ProtocolError(f"MMD/KMM {role} value drifted at {key!r}.")


def _require_numeric_mapping_equal(observed: Mapping[str, object], expected: Mapping[str, object], role: str) -> None:
    if set(observed) != set(expected):
        raise ProtocolError(f"MMD/KMM {role} keys drifted.")
    for key, value in expected.items():
        actual = observed[key]
        if isinstance(value, float):
            if not np.isclose(float(actual), value, atol=1e-12, rtol=1e-12):
                raise ProtocolError(f"MMD/KMM {role} numeric value drifted: {key}.")
        elif isinstance(value, list) and value and isinstance(value[0], float):
            if not np.allclose(np.asarray(actual, dtype=float), np.asarray(value), atol=1e-12, rtol=1e-12):
                raise ProtocolError(f"MMD/KMM {role} numeric vector drifted: {key}.")
        elif actual != value:
            raise ProtocolError(f"MMD/KMM {role} value drifted: {key}.")


def _generated_bundle_hash(embeddings: np.ndarray, labels: np.ndarray) -> str:
    digest = hashlib.sha256()
    for values in (embeddings, labels):
        array = np.ascontiguousarray(values)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _truthy(value: object) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except OSError as exc:
        raise ProtocolError(f"Cannot read MMD/KMM CSV: {path}.") from exc


def _json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read MMD/KMM JSON: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("MMD/KMM JSON must be an object.")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ("validate_mmd_kmm_router_bundle",)
