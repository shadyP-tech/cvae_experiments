"""Reconstruction and report checks for the closed-world diagnostic bundle."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .artifact_io import read_json, sha256_file
from .bundle import CONTENT_INDEX_MEMBERS
from .contracts import (
    CENTERS,
    EXPECTED_CROSS_FIT_FOLD_COUNT,
    EXPECTED_SOURCE_BLOCK_COUNT,
    GENERATION_SEEDS,
    MAXIMUM_UNIQUE_CLASSIFIER_FIT_COUNT,
    MAX_SOURCE_PREFIX_PER_CLASS,
    TRAINING_SEEDS,
)


def _validate_source_products(path: Path, source_products: object) -> None:
    array = np.load(path / "arrays/source_prefix_blocks.npy", mmap_mode="r")
    if array.shape != (
        EXPECTED_SOURCE_BLOCK_COUNT,
        2 * MAX_SOURCE_PREFIX_PER_CLASS,
        3840,
    ):
        raise ProtocolError("Antisymmetric source-array geometry drifted.")
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
    observed = []
    for ordinal, row in enumerate(source_products.index_rows):
        observed.append(
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
            raise ProtocolError("Antisymmetric source block hash drifted.")
    if tuple(observed) != expected_keys:
        raise ProtocolError("Antisymmetric source block key grid drifted.")


def _validate_claim_reports(path: Path) -> None:
    leakage = read_json(path / "reports/leakage_report.json")
    required_leakage = {
        "status": "PASS",
        "target_expert_excluded_from_every_pool": True,
        "fixed_support_cases_never_scored": True,
        "heldout_case_excluded_from_own_route": True,
        "heldout_case_embeddings_used_for_own_route": False,
        "cohort_evaluation_embeddings_used_for_other_case_routes": True,
        "support_labels_used": False,
        "evaluation_labels_available_before_global_prediction_seal": False,
        "evaluation_labels_used_for_scoring_only": True,
        "previous_stage90_router_or_utility_inputs_used": False,
        "proxy_is_nelbo_compatibility": False,
        "proxy_is_downstream_utility": False,
        "routing_quality_claimed": False,
        "heldout_target_utility_claimed": False,
        "promotion_eligible": False,
    }
    if any(leakage.get(key) != value for key, value in required_leakage.items()):
        raise ProtocolError("Antisymmetric leakage report claim boundary drifted.")
    publication = read_json(path / "reports/publication_decision.json")
    required_publication = {
        "decision": "PUBLISH_AS_EXPLORATORY_CONSUMED_DATA_DIAGNOSTIC_ONLY",
        "cross_fitted_transductive_diagnostic": True,
        "routing_quality_claimed": False,
        "heldout_target_utility_claimed": False,
        "fresh_evidence": False,
        "fresh_confirmation": False,
        "promotion_eligible": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
    }
    if any(
        publication.get(key) != value
        for key, value in required_publication.items()
    ):
        raise ProtocolError(
            "Antisymmetric publication decision escaped diagnostic scope."
        )
    runtime = read_json(path / "reports/runtime_summary.json")
    preflight = runtime.get("workstation_preflight")
    gpu_rows = preflight.get("gpus") if isinstance(preflight, Mapping) else None
    if (
        runtime.get("target_kernel_workspace_count") != len(CENTERS)
        or runtime.get("crossfit_fold_count") != EXPECTED_CROSS_FIT_FOLD_COUNT
        or runtime.get("prediction_task_count") != 81
        or int(runtime.get("unique_classifier_fit_count", -1))
        > MAXIMUM_UNIQUE_CLASSIFIER_FIT_COUNT
        or runtime.get("maximum_unique_classifier_fit_count")
        != MAXIMUM_UNIQUE_CLASSIFIER_FIT_COUNT
        or not isinstance(preflight, Mapping)
        or preflight.get("status") != "PASS"
        or not isinstance(gpu_rows, list)
        or len(gpu_rows) != 2
        or {
            int(row.get("index", -1))
            for row in gpu_rows
            if isinstance(row, Mapping)
        }
        != {0, 1}
        or preflight.get("parent_cuda_context_initialized") is not False
        or preflight.get("classifier_worker_thread_product") != 12
    ):
        raise ProtocolError("Antisymmetric runtime contract drifted.")


def _validate_phase_reports(path: Path) -> None:
    expected = (
        (
            "phase_01_source_products_complete.json",
            "PHASE_01_SOURCE_PRODUCTS_COMPLETE",
        ),
        (
            "phase_02_router_plans_complete.json",
            "PHASE_02_CASE_CROSSFIT_ROUTER_PLANS_COMPLETE",
        ),
        (
            "phase_03_predictions_sealed.json",
            "PHASE_03_ALL_CASE_CROSSFIT_PREDICTIONS_SEALED",
        ),
    )
    for filename, phase in expected:
        report = read_json(path / f"reports/{filename}")
        unhashed = {key: value for key, value in report.items() if key != "phase_hash"}
        if (
            report.get("phase_hash") != stable_hash(unhashed)
            or report.get("phase") != phase
            or report.get("diagnostic_only") is not True
            or report.get("promotion_eligible") is not False
        ):
            raise ProtocolError(
                f"Antisymmetric phase report drifted: {filename}."
            )


def _validate_content_index(path: Path) -> None:
    payload = read_json(path / "manifests/content_index.json")
    unhashed = {key: value for key, value in payload.items() if key != "content_hash"}
    records = payload.get("records")
    if (
        payload.get("content_hash") != stable_hash(unhashed)
        or not isinstance(records, list)
    ):
        raise ProtocolError("Antisymmetric content-index hash drifted.")
    if (
        tuple(
            str(row.get("relative_path"))
            for row in records
            if isinstance(row, Mapping)
        )
        != CONTENT_INDEX_MEMBERS
    ):
        raise ProtocolError("Antisymmetric content-index member order drifted.")
    for row in records:
        if not isinstance(row, Mapping):
            raise ProtocolError("Antisymmetric content-index row is malformed.")
        member = path / str(row["relative_path"])
        if (
            not member.is_file()
            or row.get("sha256") != sha256_file(member)
            or int(row.get("size_bytes", -1)) != member.stat().st_size
        ):
            raise ProtocolError(f"Antisymmetric content member drifted: {member}.")


def _compare_rows(
    observed: Sequence[Mapping[str, str]],
    expected: Sequence[Mapping[str, object]],
    role: str,
) -> None:
    if len(observed) != len(expected):
        raise ProtocolError(f"Antisymmetric {role} row count drifted.")
    for left, right in zip(observed, expected, strict=True):
        if set(left) != set(right):
            raise ProtocolError(f"Antisymmetric {role} columns drifted.")
        for key, expected_value in right.items():
            raw = left[key]
            if isinstance(expected_value, bool):
                equal = raw.lower() == str(expected_value).lower()
            elif isinstance(expected_value, (int, float)) and not isinstance(
                expected_value, bool
            ):
                try:
                    equal = np.isclose(
                        float(raw),
                        float(expected_value),
                        atol=1e-12,
                        rtol=1e-12,
                    )
                except ValueError:
                    equal = False
            else:
                equal = raw == str(expected_value)
            if not equal:
                raise ProtocolError(
                    f"Antisymmetric {role} value drifted at {key!r}."
                )


def _require_numeric_mapping_equal(
    observed: Mapping[str, object],
    expected: Mapping[str, object],
    role: str,
) -> None:
    if set(observed) != set(expected):
        raise ProtocolError(f"Antisymmetric {role} keys drifted.")
    for key, value in expected.items():
        actual = observed[key]
        if isinstance(value, float):
            if not np.isclose(float(actual), value, atol=1e-12, rtol=1e-12):
                raise ProtocolError(
                    f"Antisymmetric {role} numeric drifted: {key}."
                )
        elif isinstance(value, list) and value and isinstance(value[0], float):
            if not np.allclose(
                np.asarray(actual, dtype=float),
                np.asarray(value),
                atol=1e-12,
                rtol=1e-12,
            ):
                raise ProtocolError(
                    f"Antisymmetric {role} vector drifted: {key}."
                )
        elif actual != value:
            raise ProtocolError(f"Antisymmetric {role} value drifted: {key}.")


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
        raise ProtocolError(f"Cannot read antisymmetric CSV: {path}.") from exc


__all__: tuple[str, ...] = ()
