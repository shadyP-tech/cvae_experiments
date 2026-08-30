"""File-backed Stage-60 adapter for freezing the HARP policy lock.

This adapter has no outcome-loading API.  It binds already completed action and
support products, the immutable exact-B policy, and two unopened reservations
into a CPU-only policy lock.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Mapping, Sequence

from ...protocol import ProtocolError
from ..harp_action_surface.inference_binding import HarpActionInferenceBinding
from ..harp_action_model import (
    HarpTrainingObservation,
    fit_harp_action_model_bank,
    model_bank_collection_from_payload,
    model_bank_collection_payload,
    model_bank_from_payload,
    model_bank_payload,
)
from ..harp_protocol.hashing import canonical_hash, require_sha256
from .support_envelope import (
    HarpSupportEnvelope,
    build_support_envelope,
    load_target_support_feature_surface,
)
from ..harp_stage60.config import HarpInputReadiness, HarpStage60Config
from ..harp_stage60.constants import POLICY_LOCK
from ..harp_stage60.execution_contracts import (
    HarpBuiltProduct,
    HarpDurablePrelabelSeal,
    HarpRunReceipt,
)


ACTION_STATE = "reports/run_state.json"
SUPPORT_STATE = "reports/run_state.json"
EXACT_B_LOCK = "manifests/policy_lock.json"
RESERVATION = "manifests/reservation.json"
MODEL_MEMBER = "manifests/model_lock.json"
DELETE_DONOR_MEMBER = "manifests/delete_donor_lock.json"
ACTION_LIBRARY_MEMBER = "manifests/action_library.json"
TARGET_POLICY_MEMBER = "manifests/target_policy_lock.json"
POLICY_MEMBER = "manifests/policy_lock.json"
CONTENT_INDEX_MEMBER = "manifests/content_index.json"
STATE_MEMBER = "reports/run_state.json"
LEAKAGE_MEMBER = "reports/leakage_report.json"
VALIDATION_MEMBER = "reports/validation_report.json"
SEAL_MEMBER = "manifests/global_prediction_seal.json"
TRAINING_SURFACE = "surfaces/harp_training_observations.json"
INFERENCE_BINDING = "manifests/harp_action_inference_binding.json"
TARGET_SUPPORT_SURFACE = "surfaces/target_support_features.json"


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"HARP policy input is absent or unreadable: {path}.") from exc
    if not isinstance(value, dict):
        raise ProtocolError("HARP policy inputs must be JSON objects.")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ProtocolError(f"Cannot hash HARP policy input: {path}.") from exc
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(_json_bytes(payload))
    temporary.replace(path)


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(dict(payload), sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _payload_sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _completed_state(path: Path, *, surface: str) -> dict[str, object]:
    raw = _read_json(path)
    required = {
        "schema_version", "status", "surface", "experiment_id", "product_hash",
        "validation_hash", "target_support_labels_used", "target_evaluation_labels_used",
    }
    if (
        set(raw) != required
        or raw.get("schema_version") != "midogpp_harp_run_state_v1"
        or raw.get("status") != "COMPLETE"
        or raw.get("surface") != surface
        or raw.get("target_support_labels_used") is not False
        or raw.get("target_evaluation_labels_used") is not False
    ):
        raise ProtocolError("HARP upstream completion receipt failed closed.")
    require_sha256(raw.get("product_hash"), name="HARP upstream product hash")
    require_sha256(raw.get("validation_hash"), name="HARP upstream validation hash")
    return raw


def _load_training_surface(path: Path) -> tuple[HarpTrainingObservation, ...]:
    raw = _read_json(path)
    required = {"schema_version", "feature_surface_hash", "response_surface_hash", "rows", "training_surface_hash"}
    if (
        set(raw) != required
        or raw.get("schema_version") != "midogpp_harp_training_observation_surface_v1"
        or raw.get("training_surface_hash") != canonical_hash({key: value for key, value in raw.items() if key != "training_surface_hash"})
    ):
        raise ProtocolError("HARP training-observation surface schema or hash drifted.")
    require_sha256(raw.get("feature_surface_hash"), name="HARP feature surface hash")
    require_sha256(raw.get("response_surface_hash"), name="HARP response surface hash")
    rows = raw.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ProtocolError("HARP policy requires source-inner training observations.")
    expected = {
        "outer_target_id", "pseudo_query_id", "candidate_source_id", "case_id", "sample_id",
        "lambda_value", "direction", "feature_names", "feature_values",
        "weighted_correctness_surrogate", "brier_delta", "log_loss_delta", "truth_class",
        "ensemble_size", "ensemble_receipt_hash", "case_aggregation_receipt_hash", "prediction_seal_hash", "response_receipt_hash",
    }
    output: list[HarpTrainingObservation] = []
    try:
        for value in rows:
            if not isinstance(value, Mapping) or set(value) != expected:
                raise ProtocolError("HARP training-observation row schema drifted.")
            output.append(
                HarpTrainingObservation(
                    outer_target_id=str(value["outer_target_id"]),
                    pseudo_query_id=str(value["pseudo_query_id"]),
                    candidate_source_id=str(value["candidate_source_id"]),
                    case_id=str(value["case_id"]), sample_id=str(value["sample_id"]),
                    lambda_value=float(value["lambda_value"]), direction=str(value["direction"]),
                    feature_names=tuple(str(item) for item in value["feature_names"]),
                    feature_values=tuple(float(item) for item in value["feature_values"]),
                    weighted_correctness_surrogate=float(value["weighted_correctness_surrogate"]),
                    brier_delta=float(value["brier_delta"]), log_loss_delta=float(value["log_loss_delta"]),
                    truth_class=int(value["truth_class"]), ensemble_size=int(value["ensemble_size"]),
                    ensemble_receipt_hash=str(value["ensemble_receipt_hash"]),
                    case_aggregation_receipt_hash=str(value["case_aggregation_receipt_hash"]),
                    prediction_seal_hash=str(value["prediction_seal_hash"]),
                    response_receipt_hash=str(value["response_receipt_hash"]),
                )
            )
    except (TypeError, ValueError) as exc:
        raise ProtocolError("HARP training-observation values are malformed.") from exc
    typed = tuple(sorted(output, key=lambda row: (row.outer_target_id, row.row_key)))
    if len({(row.outer_target_id, row.row_key) for row in typed}) != len(typed):
        raise ProtocolError("HARP training-observation surface contains duplicate rows.")
    return typed


def _load_inference_binding(path: Path) -> HarpActionInferenceBinding:
    return HarpActionInferenceBinding.from_payload(_read_json(path))


def _observation_payload(row: HarpTrainingObservation) -> dict[str, object]:
    return {
        "outer_target_id": row.outer_target_id, "pseudo_query_id": row.pseudo_query_id,
        "candidate_source_id": row.candidate_source_id, "case_id": row.case_id,
        "sample_id": row.sample_id, "lambda_value": row.lambda_value,
        "direction": row.direction, "feature_names": list(row.feature_names),
        "feature_values": list(row.feature_values),
        "weighted_correctness_surrogate": row.weighted_correctness_surrogate,
        "brier_delta": row.brier_delta, "log_loss_delta": row.log_loss_delta,
        "truth_class": row.truth_class, "ensemble_size": row.ensemble_size,
        "ensemble_receipt_hash": row.ensemble_receipt_hash,
        "case_aggregation_receipt_hash": row.case_aggregation_receipt_hash,
        "prediction_seal_hash": row.prediction_seal_hash,
        "response_receipt_hash": row.response_receipt_hash,
    }


def _observation_from_payload(value: Mapping[str, object]) -> HarpTrainingObservation:
    return HarpTrainingObservation(
        outer_target_id=str(value["outer_target_id"]), pseudo_query_id=str(value["pseudo_query_id"]),
        candidate_source_id=str(value["candidate_source_id"]), case_id=str(value["case_id"]),
        sample_id=str(value["sample_id"]), lambda_value=float(value["lambda_value"]),
        direction=str(value["direction"]), feature_names=tuple(str(item) for item in value["feature_names"]),
        feature_values=tuple(float(item) for item in value["feature_values"]),
        weighted_correctness_surrogate=float(value["weighted_correctness_surrogate"]),
        brier_delta=float(value["brier_delta"]), log_loss_delta=float(value["log_loss_delta"]),
        truth_class=int(value["truth_class"]), ensemble_size=int(value["ensemble_size"]),
        ensemble_receipt_hash=str(value["ensemble_receipt_hash"]),
        case_aggregation_receipt_hash=str(value["case_aggregation_receipt_hash"]),
        prediction_seal_hash=str(value["prediction_seal_hash"]),
        response_receipt_hash=str(value["response_receipt_hash"]),
    )


def _fit_outer_worker(payload: tuple[str, tuple[dict[str, object], ...], tuple[float, ...], int]) -> dict[str, object]:
    outer_target, rows, alphas, blas_threads = payload
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=blas_threads):
        bank = fit_harp_action_model_bank(
            tuple(_observation_from_payload(row) for row in rows),
            outer_target_id=outer_target,
            alphas=alphas,
        )
    return model_bank_payload(bank)


def fit_harp_model_banks(
    rows: Sequence[HarpTrainingObservation],
    *,
    outer_targets: Sequence[str],
    alphas: Sequence[float],
    worker_count: int = 4,
    blas_threads_per_worker: int = 3,
    executor_factory: Callable[..., object] | None = None,
) -> tuple[object, ...]:
    """Fit independent outer-H banks in one non-nested spawn pool.

    ``worker_count=1`` is the deterministic sequential injection path used by
    focused tests.  Workers receive only primitive observation payloads.
    """

    targets = tuple(str(value) for value in outer_targets)
    alpha_values = tuple(float(value) for value in alphas)
    if worker_count < 1 or blas_threads_per_worker < 1:
        raise ProtocolError("HARP model worker topology must be positive.")
    tasks = tuple(
        (
            target,
            tuple(_observation_payload(row) for row in rows if row.outer_target_id == target),
            alpha_values,
            int(blas_threads_per_worker),
        )
        for target in targets
    )
    if any(not task[1] for task in tasks):
        raise ProtocolError("HARP model worker received an empty outer-target surface.")
    if worker_count == 1:
        payloads = tuple(_fit_outer_worker(task) for task in tasks)
    else:
        factory = executor_factory or ProcessPoolExecutor
        kwargs = {"max_workers": min(worker_count, len(tasks)), "mp_context": multiprocessing.get_context("spawn")}
        with factory(**kwargs) as executor:  # type: ignore[attr-defined]
            payloads = tuple(executor.map(_fit_outer_worker, tasks))  # type: ignore[attr-defined]
    banks = tuple(model_bank_from_payload(payload) for payload in payloads)
    if tuple(bank.outer_target_id for bank in banks) != targets:
        raise ProtocolError("HARP parallel model results drifted from target order.")
    return banks


def _model_lock_payload(policy: Mapping[str, object], config: HarpStage60Config) -> dict[str, object]:
    unhashed = {
        "schema_version": "midogpp_harp_model_lock_v1",
        "model_bank_collection": policy["model_bank_collection"],
        "execution_topology": {
            "outer_bank_worker_count": int(config.runtime.get("model_workers", 4)),
            "blas_threads_per_worker": int(config.runtime.get("model_threads_per_worker", 3)),
            "multiprocessing_start_method": "spawn",
            "nested_process_pools": False,
            "scientific_reductions_dtype": "float64",
            "worker_payload": "primitive_observation_records",
            "torch_interop_state_modified": False,
        },
    }
    return {**unhashed, "model_lock_hash": canonical_hash(unhashed)}


def _delete_donor_lock_payload(collection: Mapping[str, object]) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for bank in collection["banks"]:  # type: ignore[index]
        for model in bank["models"]:  # type: ignore[index]
            entries.append(
                {
                    "outer_target_id": bank["outer_target_id"],  # type: ignore[index]
                    "outcome": model["outcome"],
                    "direction": model["direction"],
                    "donors": [
                        {
                            "donor_id": value["donor_id"],
                            "model_hash": canonical_hash(value["model"]),
                            "excluded_donor_ids": value["model"]["excluded_donor_ids"],
                        }
                        for value in model["delete_donor_models"]
                    ],
                }
            )
    unhashed = {
        "schema_version": "midogpp_harp_delete_donor_lock_v1",
        "model_bank_collection_hash": collection["collection_hash"],
        "entries": entries,
    }
    return {**unhashed, "delete_donor_lock_hash": canonical_hash(unhashed)}


def _target_policy_lock_payload(policy: Mapping[str, object]) -> dict[str, object]:
    fields = {key: value for key, value in policy.items() if key not in {"model_bank_collection", "action_library"}}
    unhashed = {
        "schema_version": "midogpp_harp_target_policy_lock_v1",
        "policy_fields": fields,
        "model_bank_collection_hash": policy["model_bank_collection"]["collection_hash"],  # type: ignore[index]
        "action_library_hash": policy["action_library"]["action_library_hash"],  # type: ignore[index]
    }
    return {**unhashed, "target_policy_lock_hash": canonical_hash(unhashed)}


def _unopened_reservation(path: Path, *, artifact_id: str) -> dict[str, object]:
    raw = _read_json(path)
    if (
        raw.get("artifact_id") != artifact_id
        or raw.get("dataset_family") != "MIDOG++"
        or raw.get("status") != "ACTIVE"
        or raw.get("fresh_unconsumed_surface") is not True
        or raw.get("labels_opened") not in (None, False)
        or raw.get("labels_present") not in (None, False)
        or raw.get("target_evaluation_rows_present") not in (None, False)
        or any(raw.get(name) is True for name in ("consumed_test_used", "consumed_validation_used", "consumed_stage70_used", "consumed_stage90_used"))
    ):
        raise ProtocolError("HARP policy reservation is not active, fresh, and unopened.")
    observed = raw.get("reservation_hash")
    if observed != canonical_hash({key: value for key, value in raw.items() if key != "reservation_hash"}):
        raise ProtocolError("HARP policy reservation hash drifted.")
    return raw


def policy_input_binding(config: HarpStage60Config) -> tuple[str, str, dict[str, str]]:
    """Return input and combined-reservation identities without opening data."""

    if config.contract != POLICY_LOCK:
        raise ProtocolError("Policy input binding was requested for another HARP surface.")
    members = {
        "action_state": config.input_paths["action_surface_root"] / ACTION_STATE,
        "training_surface": config.input_paths["action_surface_root"] / TRAINING_SURFACE,
        "inference_binding": config.input_paths["action_surface_root"] / INFERENCE_BINDING,
        "exact_b_lock": config.input_paths["exact_b_policy_root"] / EXACT_B_LOCK,
        "support_state": config.input_paths["target_support_surface_root"] / SUPPORT_STATE,
        "support_feature_surface": (
            config.input_paths["target_support_surface_root"] / TARGET_SUPPORT_SURFACE
        ),
        "support_reservation": config.input_paths["target_support_reservation_root"] / RESERVATION,
        "fresh_reservation": config.input_paths["fresh_target_reservation_root"] / RESERVATION,
    }
    hashes = {name: _file_sha256(path) for name, path in members.items()}
    reservation_hash = canonical_hash({name: hashes[name] for name in ("support_reservation", "fresh_reservation")})
    input_hash = canonical_hash(hashes)
    return input_hash, reservation_hash, hashes


class ProductionPolicyLockAdapter:
    """Freeze a HARP policy without any source or target outcome capability."""

    def __init__(self) -> None:
        self._bindings: dict[str, str] | None = None
        self._states: tuple[dict[str, object], dict[str, object]] | None = None
        self._exact_b_hash: str | None = None
        self._reservation_hashes: tuple[str, str] | None = None
        self._training_rows: tuple[HarpTrainingObservation, ...] | None = None
        self._inference_binding: HarpActionInferenceBinding | None = None
        self._support_surface: dict[str, object] | None = None

    def preflight(self, config: HarpStage60Config, readiness: HarpInputReadiness) -> None:
        if config.contract != POLICY_LOCK or readiness.surface != POLICY_LOCK.surface or readiness.experiment_id != POLICY_LOCK.experiment_id:
            raise ProtocolError("HARP policy adapter received another surface or readiness receipt.")
        action = _completed_state(config.input_paths["action_surface_root"] / ACTION_STATE, surface="uniform-b-v2-harp-action-surface")
        training_rows = _load_training_surface(config.input_paths["action_surface_root"] / TRAINING_SURFACE)
        inference_binding = _load_inference_binding(config.input_paths["action_surface_root"] / INFERENCE_BINDING)
        training_payload = _read_json(config.input_paths["action_surface_root"] / TRAINING_SURFACE)
        if (
            inference_binding.feature_surface_semantic_id
            != training_payload["feature_surface_hash"]
            or inference_binding.response_surface_semantic_id
            != training_payload["response_surface_hash"]
        ):
            raise ProtocolError("HARP inference binding escaped its training surfaces.")
        support = _completed_state(config.input_paths["target_support_surface_root"] / SUPPORT_STATE, surface="uniform-b-v2-harp-target-support-surface")
        support_surface = load_target_support_feature_surface(
            _read_json(
                config.input_paths["target_support_surface_root"]
                / TARGET_SUPPORT_SURFACE
            )
        )
        exact_b = _read_json(config.input_paths["exact_b_policy_root"] / EXACT_B_LOCK)
        if exact_b.get("schema_version") != "midogpp_uniform_b_v2_equal_union_policy_lock_v1":
            raise ProtocolError("HARP fallback is not the immutable Uniform-B v2 exact-B lock.")
        exact_hash = exact_b.get("policy_lock_hash")
        if exact_hash != canonical_hash({key: value for key, value in exact_b.items() if key != "policy_lock_hash"}):
            # Legacy exact-B locks use the repository's stable hash rather than
            # the full canonical digest; validate them through their own type.
            from ..policy import read_policy_lock

            exact_hash = read_policy_lock(config.input_paths["exact_b_policy_root"] / EXACT_B_LOCK).policy_lock_hash
        support_reservation = _unopened_reservation(config.input_paths["target_support_reservation_root"] / RESERVATION, artifact_id="midogpp_harp_target_support_reservation_v1")
        fresh_reservation = _unopened_reservation(config.input_paths["fresh_target_reservation_root"] / RESERVATION, artifact_id="midogpp_harp_fresh_target_reservation_v1")
        input_hash, reservation_hash, bindings = policy_input_binding(config)
        if input_hash != readiness.input_binding_sha256 or reservation_hash != readiness.reservation_sha256:
            raise ProtocolError("HARP readiness attestation is not bound to policy input bytes.")
        self._bindings = bindings
        self._states = (action, support)
        self._exact_b_hash = str(exact_hash)
        self._reservation_hashes = (str(support_reservation["reservation_hash"]), str(fresh_reservation["reservation_hash"]))
        self._training_rows = training_rows
        self._inference_binding = inference_binding
        self._support_surface = support_surface

    def materialize_and_seal_label_free_menu(self, config: HarpStage60Config, readiness: HarpInputReadiness) -> HarpDurablePrelabelSeal:
        if self._bindings is None or self._states is None:
            raise ProtocolError("HARP policy inputs were not preflighted before sealing.")
        action, support = self._states
        probability_menu_hash = canonical_hash({"action_product_hash": action["product_hash"], "support_product_hash": support["product_hash"], "exact_b_policy_lock_hash": self._exact_b_hash})
        row_identity_hash = canonical_hash({"input_file_sha256": self._bindings, "exact_nine_seed_ensemble": True, "seed_cells_may_feed_model": False})
        unhashed = {
            "schema_version": "midogpp_harp_durable_prelabel_seal_v1",
            "status": "SEALED_COMPLETE_LABEL_FREE_MENU",
            "surface": config.contract.surface,
            "probability_menu_hash": probability_menu_hash,
            "row_identity_hash": row_identity_hash,
            "target_support_labels_used": False,
            "target_evaluation_labels_used": False,
            "source_development_labels_opened": False,
        }
        seal_hash = canonical_hash(unhashed)
        path = config.artifact_root / SEAL_MEMBER
        _atomic_json(path, {**unhashed, "seal_hash": seal_hash})
        return HarpDurablePrelabelSeal(config.contract.surface, path, seal_hash, probability_menu_hash, row_identity_hash)

    def open_source_development_labels(self, config: HarpStage60Config, seal: HarpDurablePrelabelSeal) -> object:
        raise ProtocolError("The HARP policy-lock adapter has no outcome-loading capability.")

    def build_product(self, config: HarpStage60Config, seal: HarpDurablePrelabelSeal, source_development_labels: object | None) -> HarpBuiltProduct:
        if source_development_labels is not None or self._states is None or self._bindings is None or self._reservation_hashes is None or self._training_rows is None or self._inference_binding is None or self._support_surface is None:
            raise ProtocolError("HARP policy building cannot receive or load outcomes.")
        seal.verify_durable()
        action, support = self._states
        model = config.model
        outer_targets = tuple(str(value) for value in config.protocol["center_universe"])
        observed_targets = tuple(sorted({row.outer_target_id for row in self._training_rows}))
        if observed_targets != outer_targets:
            raise ProtocolError("HARP training surface lacks exact outer-target coverage.")
        banks = fit_harp_model_banks(
            self._training_rows,
            outer_targets=outer_targets,
            alphas=tuple(float(value) for value in model["ridge_alphas"]),
            worker_count=int(config.runtime.get("model_workers", 4)),
            blas_threads_per_worker=int(config.runtime.get("model_threads_per_worker", 3)),
        )
        model_collection = model_bank_collection_payload(banks)
        feature_names = banks[0].feature_names
        if any(bank.feature_names != feature_names for bank in banks):
            raise ProtocolError("HARP outer-target feature schemas drifted.")
        action_library = {
            "schema_version": "midogpp_harp_action_library_v2",
            "candidate_sources_by_target": {
                bank.outer_target_id: list(bank.model("gain", "ALL_MARGINS").full_model.candidate_levels)
                for bank in banks
            },
            "lambda_grid": list(model["lambda_grid"]),
            "directions": ["D01", "D10", "ALL_MARGINS"],
            "feature_names": list(feature_names),
            "probability_endpoint": "exact_nine_seed_ensemble_float64",
            "predictive_reference_action_id": "U",
            "operational_fallback_action_id": "B",
            "lambda_semantics": "post_classifier_predictive_probability_ensemble_not_generated_distribution",
            "lambda_one_is_physical_hxe_endpoint": True,
            "selection_order": ["gain_lower_desc", "brier_upper_asc", "log_loss_upper_asc", "lambda_asc", "source_id_asc"],
        }
        action_library["action_library_hash"] = canonical_hash(action_library)
        support_envelope = build_support_envelope(
            self._support_surface,
            banks,
            maximum_allowed_leverage=float(model["maximum_leverage"]),
            center_universe=outer_targets,
        )
        payload: dict[str, object] = {
            "schema_version": "midogpp_harp_policy_lock_v2",
            "artifact_id": config.output_artifact_id,
            "experiment_id": config.experiment_id,
            "status": "FROZEN_BEFORE_TARGET_EVALUATION",
            "dataset_family": "MIDOG++",
            "config_contract_hash": config.contract_hash,
            "prelabel_seal_hash": seal.seal_hash,
            "action_surface_product_hash": action["product_hash"],
            "target_support_surface_product_hash": support["product_hash"],
            "exact_b_policy_lock_hash": self._exact_b_hash,
            "support_reservation_hash": self._reservation_hashes[0],
            "fresh_target_reservation_hash": self._reservation_hashes[1],
            "lambda_grid": list(model["lambda_grid"]),
            "ridge_alphas": list(model["ridge_alphas"]),
            "gain_kappa": model["conservative_gain_kappa"],
            "loss_kappa": model["conservative_loss_kappa"],
            "minimum_paired_cases": model["minimum_paired_cases"],
            "minimum_donor_centers": model["minimum_donor_centers"],
            "minimum_truth_classes": model["minimum_truth_classes"],
            "minimum_positive_gain": model["minimum_positive_gain"],
            "maximum_brier_delta": model["maximum_brier_delta"],
            "maximum_log_loss_delta": model["maximum_log_loss_delta"],
            "maximum_leverage": model["maximum_leverage"],
            "minimum_compatibility_shrinkage": model["minimum_compatibility_shrinkage"],
            "probability_endpoint": "exact_nine_seed_ensemble",
            "matched_budget_reference_action": model["matched_budget_reference_action"],
            "utility_deltas_reference_action": model["utility_deltas_reference_action"],
            "lambda_semantics": model["lambda_semantics"],
            "physical_expert_routing_primary_lambda": model[
                "physical_expert_routing_primary_lambda"
            ],
            "operational_fallback_action": "B",
            "case_equal_weighting": True,
            "delete_donor_predictions": True,
            "proper_loss_noninferiority": True,
            "exact_b_byte_identical_fallback": True,
            "policy_accepts_outcomes": False,
            "target_support_outcomes_used": False,
            "target_support_feature_geometry_used_for_shrink_only": True,
            "support_predicted_outcomes_used": False,
            "target_evaluation_outcomes_used": False,
            "stage50_artifacts_used": False,
            "stage90_artifacts_used": False,
            "model_bank_collection": model_collection,
            "action_library": action_library,
            "action_inference_binding": self._inference_binding.to_payload(),
            "action_inference_binding_sha256": (
                self._inference_binding.binding_sha256
            ),
            "support_compatibility_envelope": support_envelope.to_payload(),
            "support_compatibility_envelope_sha256": (
                support_envelope.envelope_sha256
            ),
        }
        return HarpBuiltProduct(config.contract.surface, payload, canonical_hash(payload), False)

    def persist_product(self, config: HarpStage60Config, seal: HarpDurablePrelabelSeal, product: HarpBuiltProduct) -> Path:
        seal.verify_durable()
        if product.surface != POLICY_LOCK.surface or product.source_development_labels_used_for_scoring_only or product.product_hash != canonical_hash(dict(product.payload)):
            raise ProtocolError("HARP policy product crossed its outcome-free boundary.")
        policy = dict(product.payload)
        lock_payload = {**policy, "policy_lock_hash": product.product_hash}
        model_lock = _model_lock_payload(policy, config)
        delete_lock = _delete_donor_lock_payload(policy["model_bank_collection"])  # type: ignore[arg-type]
        action_library = policy["action_library"]
        if not isinstance(action_library, Mapping):
            raise ProtocolError("HARP action library is not a mapping.")
        target_policy = _target_policy_lock_payload(policy)
        reconstructed = model_bank_collection_payload(
            model_bank_collection_from_payload(policy["model_bank_collection"])
        )
        if reconstructed != policy["model_bank_collection"]:
            raise ProtocolError("HARP model banks did not survive independent reconstruction.")
        _atomic_json(config.artifact_root / MODEL_MEMBER, model_lock)
        _atomic_json(config.artifact_root / DELETE_DONOR_MEMBER, delete_lock)
        _atomic_json(config.artifact_root / ACTION_LIBRARY_MEMBER, action_library)
        _atomic_json(config.artifact_root / TARGET_POLICY_MEMBER, target_policy)
        _atomic_json(config.artifact_root / POLICY_MEMBER, lock_payload)
        leakage_unhashed = {
            "schema_version": "midogpp_harp_policy_leakage_report_v1",
            "status": "PASS",
            "source_development_outcomes_used_for_action_model_only": True,
            "target_support_outcomes_used": False,
            "target_support_feature_geometry_used_for_shrink_only": True,
            "support_predicted_outcomes_used": False,
            "target_evaluation_outcomes_used": False,
            "seed_cells_treated_as_independent_observations": False,
            "stage50_artifacts_used": False,
            "stage90_artifacts_used": False,
        }
        leakage = {**leakage_unhashed, "leakage_report_hash": canonical_hash(leakage_unhashed)}
        _atomic_json(config.artifact_root / LEAKAGE_MEMBER, leakage)
        validation_unhashed = {
            "schema_version": "midogpp_harp_policy_validation_report_v1",
            "status": "PASS",
            "policy_lock_hash": product.product_hash,
            "model_lock_hash": model_lock["model_lock_hash"],
            "delete_donor_lock_hash": delete_lock["delete_donor_lock_hash"],
            "action_library_hash": action_library["action_library_hash"],
            "target_policy_lock_hash": target_policy["target_policy_lock_hash"],
            "leakage_report_hash": leakage["leakage_report_hash"],
            "prelabel_seal_hash": seal.seal_hash,
            "outcomes_accessible_to_policy": False,
            "model_banks_independently_reconstructed": True,
            "exact_b_byte_identical_fallback": True,
            "support_compatibility_envelope_sha256": policy[
                "support_compatibility_envelope_sha256"
            ],
        }
        validation_hash = canonical_hash(validation_unhashed)
        _atomic_json(config.artifact_root / VALIDATION_MEMBER, {**validation_unhashed, "validation_hash": validation_hash})
        run_state = {
            "schema_version": "midogpp_harp_run_state_v1", "status": "COMPLETE",
            "surface": config.contract.surface, "experiment_id": config.experiment_id,
            "product_hash": product.product_hash, "validation_hash": validation_hash,
            "target_support_labels_used": False, "target_evaluation_labels_used": False,
        }
        indexed_members = (
            SEAL_MEMBER, MODEL_MEMBER, DELETE_DONOR_MEMBER, ACTION_LIBRARY_MEMBER,
            TARGET_POLICY_MEMBER, POLICY_MEMBER, LEAKAGE_MEMBER, VALIDATION_MEMBER,
            STATE_MEMBER,
        )
        index_unhashed = {
            "schema_version": "midogpp_harp_policy_content_index_v1",
            "members": {
                member: (
                    _payload_sha256(run_state)
                    if member == STATE_MEMBER
                    else _file_sha256(config.artifact_root / member)
                )
                for member in indexed_members
            },
        }
        content_index = {**index_unhashed, "content_index_hash": canonical_hash(index_unhashed)}
        _atomic_json(config.artifact_root / CONTENT_INDEX_MEMBER, content_index)
        # The state is the commit marker and remains last, even though its
        # deterministic bytes were pre-hashed into the content index above.
        _atomic_json(config.artifact_root / STATE_MEMBER, run_state)
        return config.artifact_root

    def validate_completed_bundle(self, config: HarpStage60Config) -> HarpRunReceipt:
        state = _completed_state(config.artifact_root / STATE_MEMBER, surface=POLICY_LOCK.surface)
        lock = _read_json(config.artifact_root / POLICY_MEMBER)
        model_lock = _read_json(config.artifact_root / MODEL_MEMBER)
        delete_lock = _read_json(config.artifact_root / DELETE_DONOR_MEMBER)
        action_library_file = _read_json(config.artifact_root / ACTION_LIBRARY_MEMBER)
        target_policy = _read_json(config.artifact_root / TARGET_POLICY_MEMBER)
        leakage = _read_json(config.artifact_root / LEAKAGE_MEMBER)
        validation = _read_json(config.artifact_root / VALIDATION_MEMBER)
        content_index = _read_json(config.artifact_root / CONTENT_INDEX_MEMBER)
        observed = lock.get("policy_lock_hash")
        if observed != canonical_hash({key: value for key, value in lock.items() if key != "policy_lock_hash"}) or observed != state["product_hash"]:
            raise ProtocolError("Frozen HARP policy-lock hash drifted.")
        banks = model_bank_collection_from_payload(lock.get("model_bank_collection"))
        rebuilt_collection = model_bank_collection_payload(banks)
        if rebuilt_collection != lock.get("model_bank_collection"):
            raise ProtocolError("Frozen HARP model-bank state cannot be independently reconstructed.")
        expected_model_lock = _model_lock_payload({"model_bank_collection": rebuilt_collection}, config)
        if model_lock != expected_model_lock:
            raise ProtocolError("Frozen HARP model lock or execution topology drifted.")
        expected_delete_lock = _delete_donor_lock_payload(rebuilt_collection)
        if delete_lock != expected_delete_lock:
            raise ProtocolError("Frozen HARP delete-donor lock drifted.")
        library = lock.get("action_library")
        if not isinstance(library, Mapping) or library.get("action_library_hash") != canonical_hash({key: value for key, value in library.items() if key != "action_library_hash"}):
            raise ProtocolError("Frozen HARP action library hash drifted.")
        if dict(library) != action_library_file:
            raise ProtocolError("Standalone HARP action library drifted from the policy lock.")
        if target_policy != _target_policy_lock_payload({key: value for key, value in lock.items() if key != "policy_lock_hash"}):
            raise ProtocolError("Frozen HARP target policy lock drifted.")
        inference_binding = HarpActionInferenceBinding.from_payload(
            lock.get("action_inference_binding")
        )
        if lock.get("action_inference_binding_sha256") != (
            inference_binding.binding_sha256
        ):
            raise ProtocolError("Frozen HARP inference-binding reference drifted.")
        support_envelope = HarpSupportEnvelope.from_payload(
            lock.get("support_compatibility_envelope")
        )
        if lock.get("support_compatibility_envelope_sha256") != (
            support_envelope.envelope_sha256
        ):
            raise ProtocolError("Frozen HARP support-envelope reference drifted.")
        validation_hash = validation.get("validation_hash")
        if (
            validation_hash != canonical_hash({key: value for key, value in validation.items() if key != "validation_hash"})
            or validation_hash != state["validation_hash"]
            or validation.get("outcomes_accessible_to_policy") is not False
            or validation.get("model_lock_hash") != model_lock["model_lock_hash"]
            or validation.get("delete_donor_lock_hash") != delete_lock["delete_donor_lock_hash"]
            or validation.get("action_library_hash") != action_library_file["action_library_hash"]
            or validation.get("target_policy_lock_hash") != target_policy["target_policy_lock_hash"]
            or validation.get("leakage_report_hash") != leakage.get("leakage_report_hash")
            or validation.get("support_compatibility_envelope_sha256")
            != support_envelope.envelope_sha256
        ):
            raise ProtocolError("Frozen HARP policy validation drifted.")
        if leakage.get("leakage_report_hash") != canonical_hash({key: value for key, value in leakage.items() if key != "leakage_report_hash"}) or leakage.get("status") != "PASS" or leakage.get("target_support_feature_geometry_used_for_shrink_only") is not True or any(leakage.get(key) is not False for key in ("target_support_outcomes_used", "support_predicted_outcomes_used", "target_evaluation_outcomes_used", "stage50_artifacts_used", "stage90_artifacts_used")):
            raise ProtocolError("Frozen HARP leakage report drifted.")
        expected_members = {
            SEAL_MEMBER, MODEL_MEMBER, DELETE_DONOR_MEMBER, ACTION_LIBRARY_MEMBER,
            TARGET_POLICY_MEMBER, POLICY_MEMBER, LEAKAGE_MEMBER, VALIDATION_MEMBER,
            STATE_MEMBER,
        }
        members = content_index.get("members")
        if (
            not isinstance(members, Mapping)
            or set(members) != expected_members
            or content_index.get("content_index_hash") != canonical_hash({key: value for key, value in content_index.items() if key != "content_index_hash"})
            or any(members[member] != _file_sha256(config.artifact_root / member) for member in expected_members)
        ):
            raise ProtocolError("Frozen HARP content index drifted.")
        return HarpRunReceipt(POLICY_LOCK.surface, config.artifact_root, str(observed), str(validation_hash))


__all__ = ("ProductionPolicyLockAdapter", "fit_harp_model_banks", "policy_input_binding")
