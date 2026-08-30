"""Independent file-backed validation of a completed fresh HARP bundle."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...runtime.harp_probability_menu import (
    EXACT_NINE_SEED_PAIRS,
    TARGET_SURFACE,
    HarpActionSpec,
    HarpPredictionCell,
    HarpPredictionMenuSeal,
    HarpRouteDecision,
    route_harp_probability_vector,
    seal_harp_prediction_menu,
)
from ...runtime.harp_probability_menu.hashing import (
    canonical_sha256,
    raw_array_sha256,
)
from .config import HarpFreshStage70Config, load_harp_fresh_stage70_config
from .oracle_diagnostics import (
    HarpFreshActionMatrixMetric,
    HarpFreshCenterOracleDiagnostic,
    HarpFreshOracleDiagnosticResult,
)
from .sealing import physical_ablation_reference_preserving_vector


REQUIRED_CATALOG_MEMBERS = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "manifests/prediction_menu_seal.json",
    "manifests/route_set_seal.json",
    "manifests/prelabel_content_index.json",
    "arrays/routed_probabilities.npz",
    "tables/case_metrics.csv",
    "tables/center_metrics.csv",
    "tables/action_matrix_metrics.csv",
    "reports/center_inference.json",
    "reports/action_oracle_diagnostics.json",
    "reports/leakage_report.json",
    "manifests/content_index.json",
    "reports/run_state.json",
    "reports/validation_report.json",
)


def _json(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Fresh HARP bundle JSON is unreadable: {path}.") from exc
    if not isinstance(raw, dict):
        raise ProtocolError("Fresh HARP bundle JSON must be an object.")
    return raw


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _require_self_hash(raw: Mapping[str, object], member: str, *, role: str) -> str:
    observed = raw.get(member)
    if observed != canonical_sha256({key: value for key, value in raw.items() if key != member}):
        raise ProtocolError(f"Fresh HARP {role} hash drifted.")
    return str(observed)


def _validate_index(root: Path, payload: Mapping[str, object]) -> None:
    _require_self_hash(payload, "content_hash", role="content index")
    rows = payload.get("files")
    if not isinstance(rows, list) or payload.get("file_count") != len(rows):
        raise ProtocolError("Fresh HARP content-index coverage drifted.")
    seen: set[str] = set()
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ProtocolError("Fresh HARP content-index row is malformed.")
        member = str(raw.get("path", ""))
        path = (root / member).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise ProtocolError("Fresh HARP content-index member escaped root.") from exc
        if (
            not member
            or member in seen
            or not path.is_file()
            or path.is_symlink()
            or raw.get("sha256") != _sha256_file(path)
        ):
            raise ProtocolError("Fresh HARP content-index member drifted.")
        seen.add(member)


def _read_csv(path: Path) -> tuple[dict[str, str], ...]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return tuple(dict(row) for row in csv.DictReader(handle))
    except OSError as exc:
        raise ProtocolError("Fresh HARP metric CSV is unreadable.") from exc


def _safe_member(root: Path, relative: object) -> Path:
    member = str(relative)
    candidate = (root / member).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ProtocolError("Fresh HARP prediction checkpoint escaped its root.") from exc
    if not member or candidate.is_symlink() or not candidate.is_file():
        raise ProtocolError("Fresh HARP prediction checkpoint is absent or unsafe.")
    return candidate


def _validate_prediction_checkpoints(
    root: Path,
    menu: Mapping[str, object],
) -> HarpPredictionMenuSeal:
    raw_actions = menu.get("actions")
    raw_cells = menu.get("prediction_cells")
    if not isinstance(raw_actions, list) or not isinstance(raw_cells, list):
        raise ProtocolError("Fresh HARP prediction-menu inventory is absent.")
    actions: list[HarpActionSpec] = []
    for raw in raw_actions:
        if not isinstance(raw, Mapping):
            raise ProtocolError("Fresh HARP prediction-menu action is malformed.")
        action = HarpActionSpec(
            surface_kind=str(raw.get("surface_kind")),
            outer_target_id=str(raw.get("outer_target_id")),
            query_center_id=str(raw.get("query_center_id")),
            selected_source_id=(
                None
                if raw.get("selected_source_id") is None
                else str(raw.get("selected_source_id"))
            ),
            action_id=str(raw.get("action_id")),
        )
        if action.to_payload() != dict(raw):
            raise ProtocolError("Fresh HARP persisted action semantics drifted.")
        actions.append(action)
    expected = tuple(
        (action, training_seed, generation_seed)
        for action in actions
        for training_seed, generation_seed in EXACT_NINE_SEED_PAIRS
    )
    if len(raw_cells) != len(expected) or len(expected) != 810:
        raise ProtocolError("Fresh HARP persisted prediction-cell coverage drifted.")
    checkpoint_root = root / str(menu.get("prediction_checkpoint_root", ""))
    if checkpoint_root.resolve() != (root / "checkpoints/predictions").resolve():
        raise ProtocolError("Fresh HARP prediction-checkpoint root drifted.")
    cells: list[HarpPredictionCell] = []
    for raw, (action, training_seed, generation_seed) in zip(
        raw_cells, expected, strict=True
    ):
        if not isinstance(raw, Mapping):
            raise ProtocolError("Fresh HARP persisted prediction cell is malformed.")
        source = action.action_id if action.selected_source_id is None else action.selected_source_id
        task_id = (
            f"H_{action.outer_target_id}__e_{source}__"
            f"train_{training_seed}__gen_{generation_seed}"
        )
        receipt = _json(checkpoint_root / f"tasks/{task_id}.json")
        if (
            receipt.get("schema_version")
            != "midogpp_harp_fresh_prediction_task_result_v2"
            or receipt.get("task_id") != task_id
            or receipt.get("action_hash") != action.action_hash
            or receipt.get("training_seed") != training_seed
            or receipt.get("generation_seed") != generation_seed
            or receipt.get("outer_target_id") != action.outer_target_id
            or receipt.get("selected_source_id") != action.selected_source_id
            or receipt.get("action_id") != action.action_id
            or receipt.get("labels_available_to_fit_or_predict") is not False
            or receipt.get("classifier_converged") is not True
            or receipt.get("result_hash")
            != canonical_sha256(
                {key: value for key, value in receipt.items() if key != "result_hash"}
            )
        ):
            raise ProtocolError("Fresh HARP prediction task receipt drifted.")
        task_keys = (
            "task_id",
            "outer_target_id",
            "selected_source_id",
            "action_id",
            "training_seed",
            "generation_seed",
            "action_hash",
            "frame_hash",
            "target_embedding_bytes_sha256",
            "row_count",
            "policy_lock_hash",
            "source_stream_content_hash",
            "target_cache_hash",
            "classifier_config_hash",
        )
        task_payload = {
            "schema_version": "midogpp_harp_fresh_prediction_task_v2",
            **{key: receipt.get(key) for key in task_keys},
        }
        if receipt.get("task_hash") != canonical_sha256(task_payload):
            raise ProtocolError("Fresh HARP prediction task semantic hash drifted.")
        expected_probability_member = f"arrays/{task_id}.probabilities.npy"
        if receipt.get("probability_member") != expected_probability_member:
            raise ProtocolError("Fresh HARP probability member identity drifted.")
        probability_path = _safe_member(checkpoint_root, expected_probability_member)
        if receipt.get("probability_file_sha256") != _sha256_file(probability_path):
            raise ProtocolError("Fresh HARP probability file bytes drifted.")
        values = np.load(probability_path, mmap_mode="r", allow_pickle=False)
        if (
            values.dtype != np.float32
            or values.shape != (int(receipt.get("row_count", -1)),)
            or raw_array_sha256(values) != receipt.get("probability_bytes_sha256")
        ):
            raise ProtocolError("Fresh HARP probability vector bytes drifted.")
        try:
            cell = HarpPredictionCell(
                action=action,
                training_seed=training_seed,
                generation_seed=generation_seed,
                row_ids=tuple(str(value) for value in raw.get("row_ids", ())),
                case_ids=tuple(str(value) for value in raw.get("case_ids", ())),
                probabilities=np.asarray(values),
                bank_hash=str(raw.get("bank_hash")),
                generation_lock_hash=str(raw.get("generation_lock_hash")),
                source_cache_hash=str(raw.get("source_cache_hash")),
                frame_hash=str(raw.get("frame_hash")),
                classifier_hash=str(raw.get("classifier_hash")),
                composition_hash=str(raw.get("composition_hash")),
                scaler_state_hash=str(raw.get("scaler_state_hash")),
            )
        except (TypeError, ValueError) as exc:
            raise ProtocolError("Fresh HARP prediction cell cannot be reconstructed.") from exc
        if (
            cell.to_payload() != dict(raw)
            or cell.probability_bytes_sha256 != receipt.get("probability_bytes_sha256")
            or cell.composition_hash != receipt.get("composition_hash")
            or cell.scaler_state_hash != receipt.get("scaler_state_hash")
            or cell.classifier_hash != receipt.get("classifier_config_hash")
            or cell.frame_hash != receipt.get("frame_hash")
        ):
            raise ProtocolError("Fresh HARP prediction cell/receipt cross-link drifted.")
        cells.append(cell)
    reconstructed = seal_harp_prediction_menu(tuple(actions), tuple(cells))
    if (
        menu.get("seed_pairs") != [list(pair) for pair in EXACT_NINE_SEED_PAIRS]
        or menu.get("workstation") != reconstructed.workstation.to_payload()
        or menu.get("workstation_hash") != reconstructed.workstation.runtime_hash
        or menu.get("action_menu_hash") != reconstructed.action_menu_hash
        or menu.get("prediction_store_hash") != reconstructed.prediction_store_hash
        or menu.get("prediction_menu_seal_hash") != reconstructed.seal_hash
    ):
        raise ProtocolError("Fresh HARP global prediction-menu seal cannot be reconstructed.")
    return reconstructed


def _validate_routed_arrays(
    path: Path,
    route: Mapping[str, object],
    menu: HarpPredictionMenuSeal,
) -> tuple[int, int, int, int]:
    raw_primary = route.get("decisions")
    raw_physical = route.get("physical_ablation_decisions")
    if not isinstance(raw_primary, list) or not isinstance(raw_physical, list):
        raise ProtocolError("Fresh HARP route manifest lacks both decision sets.")

    def reconstruct(raw: object) -> HarpRouteDecision:
        if not isinstance(raw, Mapping) or type(raw.get("eligible")) is not bool:
            raise ProtocolError("Fresh HARP persisted route decision is malformed.")
        try:
            decision = HarpRouteDecision(
                surface_kind=str(raw.get("surface_kind", "")),
                outer_target_id=str(raw.get("outer_target_id", "")),
                query_center_id=str(raw.get("query_center_id", "")),
                row_id=str(raw.get("row_id", "")),
                case_id=str(raw.get("case_id", "")),
                eligible=bool(raw.get("eligible")),
                selected_source_id=(
                    None
                    if raw.get("selected_source_id") is None
                    else str(raw.get("selected_source_id"))
                ),
                lambda_value=float(raw.get("lambda_value", "nan")),
                direction=str(raw.get("direction", "")),
                decision_reason=str(raw.get("decision_reason", "")),
                policy_hash=str(raw.get("policy_hash", "")),
                prediction_menu_seal_hash=str(
                    raw.get("prediction_menu_seal_hash", "")
                ),
                labels_consumed=bool(raw.get("labels_consumed")),
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProtocolError(
                "Fresh HARP persisted route decision cannot be reconstructed."
            ) from exc
        if decision.to_payload() != dict(raw):
            raise ProtocolError("Fresh HARP persisted route decision semantics drifted.")
        return decision

    primary = tuple(reconstruct(raw) for raw in raw_primary)
    physical = tuple(reconstruct(raw) for raw in raw_physical)
    if (
        len(primary) != len(physical)
        or tuple((row.outer_target_id, row.row_id, row.case_id) for row in primary)
        != tuple((row.outer_target_id, row.row_id, row.case_id) for row in physical)
        or any(row.eligible and row.lambda_value != 1.0 for row in physical)
    ):
        raise ProtocolError("Fresh HARP physical-ablation decision coverage drifted.")
    primary_fallback = primary_eligible = 0
    physical_fallback = physical_eligible = 0
    primary_hashes: list[str] = []
    physical_hashes: list[str] = []
    physical_reference_preserving_hashes: list[str] = []
    primary_decision_hashes: list[str] = []
    physical_decision_hashes: list[str] = []
    primary_routed_probability_hashes: list[str] = []
    physical_routed_probability_hashes: list[str] = []
    with np.load(path, allow_pickle=False) as arrays:
        expected_members = {
            member
            for center in CENTERS
            for member in (
                f"center_{center}_baseline",
                f"center_{center}_reference",
                f"center_{center}_selected",
                f"center_{center}_routed",
                f"center_{center}_physical_ablation_baseline",
                f"center_{center}_physical_ablation_reference",
                f"center_{center}_physical_ablation_selected",
                f"center_{center}_physical_ablation_routed",
                f"center_{center}_physical_ablation_reference_preserving",
            )
        }
        if set(arrays.files) != expected_members:
            raise ProtocolError("Fresh HARP routed-vector archive members drifted.")
        cursor = 0
        for center in CENTERS:
            row_count = len(menu.identities_for(menu.action_for(
                surface_kind=TARGET_SURFACE,
                outer_target_id=center,
                query_center_id=center,
                selected_source_id=None,
            ))[0])
            primary_block = primary[cursor : cursor + row_count]
            physical_block = physical[cursor : cursor + row_count]
            if len(primary_block) != row_count or len(physical_block) != row_count:
                raise ProtocolError("Fresh HARP route/vector row coverage drifted.")
            primary_vector = route_harp_probability_vector(menu, primary_block)
            physical_vector = route_harp_probability_vector(menu, physical_block)
            physical_reference_preserving = (
                physical_ablation_reference_preserving_vector(physical_vector)
            )
            for prefix, vector in (
                (f"center_{center}", primary_vector),
                (f"center_{center}_physical_ablation", physical_vector),
            ):
                expected_arrays = {
                    "baseline": vector.baseline_probabilities,
                    "reference": vector.reference_probabilities,
                    "selected": vector.selected_action_probabilities,
                    "routed": vector.routed_probabilities,
                }
                for suffix, expected in expected_arrays.items():
                    observed = np.asarray(arrays[f"{prefix}_{suffix}"])
                    if (
                        observed.dtype != np.float64
                        or observed.shape != expected.shape
                        or raw_array_sha256(observed) != raw_array_sha256(expected)
                    ):
                        raise ProtocolError(
                            "Fresh HARP routed-vector archive bytes drifted."
                        )
            observed_reference_preserving = np.asarray(
                arrays[
                    f"center_{center}_physical_ablation_reference_preserving"
                ]
            )
            if (
                observed_reference_preserving.dtype != np.float64
                or observed_reference_preserving.shape
                != physical_reference_preserving.shape
                or raw_array_sha256(observed_reference_preserving)
                != raw_array_sha256(physical_reference_preserving)
            ):
                raise ProtocolError(
                    "Fresh HARP physical reference-preserving vector bytes drifted."
                )
            primary_hashes.append(primary_vector.routed_vector_seal_hash)
            physical_hashes.append(physical_vector.routed_vector_seal_hash)
            physical_reference_preserving_hashes.append(
                raw_array_sha256(physical_reference_preserving)
            )
            primary_decision_hashes.append(primary_vector.decision_set_hash)
            physical_decision_hashes.append(physical_vector.decision_set_hash)
            primary_routed_probability_hashes.append(
                primary_vector.routed_bytes_sha256
            )
            physical_routed_probability_hashes.append(
                physical_vector.routed_bytes_sha256
            )
            primary_eligible += sum(row.eligible for row in primary_block)
            primary_fallback += sum(not row.eligible for row in primary_block)
            physical_eligible += sum(row.eligible for row in physical_block)
            physical_fallback += sum(not row.eligible for row in physical_block)
            cursor += row_count
    if cursor != len(primary) or cursor != len(physical):
        raise ProtocolError("Fresh HARP route manifest contains surplus decisions.")
    if (
        primary_hashes != route.get("routed_vector_hashes")
        or physical_hashes != route.get("physical_ablation_routed_vector_hashes")
        or physical_reference_preserving_hashes
        != route.get("physical_ablation_reference_preserving_sha256")
    ):
        raise ProtocolError("Fresh HARP routed-vector hash inventory drifted.")
    expected_validation_hashes = [
        canonical_sha256(
            {
                "schema_version": "midogpp_harp_fresh_prelabel_validator_a_v1",
                "prediction_menu_seal_hash": menu.seal_hash,
                "routed_vector_hashes": primary_hashes,
                "physical_ablation_routed_vector_hashes": physical_hashes,
                "physical_ablation_reference_preserving_sha256": (
                    physical_reference_preserving_hashes
                ),
                "physical_ablation_action_universe": "Hxe_lambda_one_only",
                "exact_b_fallback_byte_identity": True,
                "labels_opened": False,
            }
        ),
        canonical_sha256(
            {
                "schema_version": "midogpp_harp_fresh_prelabel_validator_b_v1",
                "prediction_store_hash": menu.prediction_store_hash,
                "decision_set_hashes": primary_decision_hashes,
                "physical_ablation_decision_set_hashes": (
                    physical_decision_hashes
                ),
                "routed_probability_hashes": primary_routed_probability_hashes,
                "physical_ablation_routed_probability_hashes": (
                    physical_routed_probability_hashes
                ),
                "physical_ablation_reference_preserving_sha256": (
                    physical_reference_preserving_hashes
                ),
                "labels_opened": False,
            }
        ),
    ]
    if route.get("independent_validation_hashes") != expected_validation_hashes:
        raise ProtocolError(
            "Fresh HARP independent prelabel validation hashes drifted."
        )
    return (
        primary_fallback,
        primary_eligible,
        physical_fallback,
        physical_eligible,
    )


def _validate_prelabel_lineage_hashes(
    *,
    config: HarpFreshStage70Config,
    provenance: Mapping[str, object],
    route: Mapping[str, object],
    menu: HarpPredictionMenuSeal,
) -> None:
    """Recompute the durable plan, route set, and prelabel seal from semantics."""

    if (
        provenance.get("policy_lock_hash") != route.get("policy_lock_hash")
        or provenance.get("reservation_hash") != route.get("reservation_hash")
        or provenance.get("target_cache_hash") != route.get("target_cache_hash")
    ):
        raise ProtocolError("Fresh HARP route/provenance identity drifted.")
    routed_hashes = route.get("routed_vector_hashes")
    physical_hashes = route.get("physical_ablation_routed_vector_hashes")
    reference_hashes = route.get(
        "physical_ablation_reference_preserving_sha256"
    )
    validation_hashes = route.get("independent_validation_hashes")
    if not all(
        isinstance(values, list) and len(values) == len(CENTERS)
        for values in (routed_hashes, physical_hashes, reference_hashes)
    ) or not isinstance(validation_hashes, list):
        raise ProtocolError("Fresh HARP route hash inventory is malformed.")

    expected_durable = canonical_sha256(
        {
            "schema_version": "midogpp_harp_fresh_durable_prelabel_plan_v2",
            "config_contract_hash": config.contract_hash,
            "policy_lock_hash": route.get("policy_lock_hash"),
            "policy_receipt_hash": provenance.get("policy_receipt_hash"),
            "reservation_hash": route.get("reservation_hash"),
            "target_cache_hash": route.get("target_cache_hash"),
            "target_cache_content_hash": provenance.get(
                "target_cache_content_hash"
            ),
            "prediction_menu_seal_hash": menu.seal_hash,
            "routed_vector_hashes": routed_hashes,
            "physical_ablation_routed_vector_hashes": physical_hashes,
            "physical_ablation_reference_preserving_sha256": reference_hashes,
            "physical_ablation_action_universe": "Hxe_lambda_one_only",
            "physical_ablation_reference_preserving_semantics": (
                "eligible_Hxe_lambda_one_else_exact_U"
            ),
            "physical_ablation_selection_labels_used": False,
            "all_routes_and_vectors_complete": True,
            "labels_opened": False,
        }
    )
    expected_route_set = canonical_sha256(
        {
            "schema_version": "midogpp_harp_fresh_route_set_v2",
            "menu_seal_hash": menu.seal_hash,
            "policy_hash": route.get("policy_lock_hash"),
            "reservation_hash": route.get("reservation_hash"),
            "target_cache_hash": route.get("target_cache_hash"),
            "routed_vectors": routed_hashes,
            "physical_ablation_routed_vectors": physical_hashes,
            "physical_ablation_reference_preserving_sha256": reference_hashes,
            "physical_ablation_action_universe": "Hxe_lambda_one_only",
            "physical_ablation_reference_preserving_semantics": (
                "eligible_Hxe_lambda_one_else_exact_U"
            ),
            "physical_ablation_selection_labels_used": False,
            "all_routes_selected": True,
            "exact_b_fallback_byte_identity": True,
            "labels_opened": False,
        }
    )
    expected_prelabel_seal = canonical_sha256(
        {
            "schema_version": "midogpp_harp_fresh_prelabel_seal_v2",
            "status": "DURABLE_ALL_ROUTES_SEALED_BEFORE_LABELS",
            "route_set_hash": expected_route_set,
            "menu_seal_hash": menu.seal_hash,
            "durable_bundle_hash": expected_durable,
            "independent_validation_hashes": validation_hashes,
            "labels_opened_before_seal": False,
        }
    )
    if (
        route.get("durable_bundle_hash") != expected_durable
        or route.get("route_set_hash") != expected_route_set
        or route.get("prelabel_seal_hash") != expected_prelabel_seal
    ):
        raise ProtocolError("Fresh HARP prelabel lineage hashes drifted.")


def _csv_bool(value: object, *, field: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ProtocolError(f"Fresh HARP oracle CSV boolean {field} drifted.")


def _reconstruct_oracle_result(
    rows: tuple[dict[str, str], ...],
    report: Mapping[str, object],
) -> HarpFreshOracleDiagnosticResult:
    matrix: list[HarpFreshActionMatrixMetric] = []
    for raw in rows:
        source_text = raw.get("selected_source_id", "")
        matrix.append(
            HarpFreshActionMatrixMetric(
                center=str(raw.get("center", "")),
                action_id=str(raw.get("action_id", "")),
                physical_action_id=str(raw.get("physical_action_id", "")),
                selected_source_id=None if source_text == "" else source_text,
                lambda_value=float(raw.get("lambda_value", "nan")),
                physical_generated_action=_csv_bool(
                    raw.get("physical_generated_action"),
                    field="physical_generated_action",
                ),
                matched_budget_action=_csv_bool(
                    raw.get("matched_budget_action"), field="matched_budget_action"
                ),
                row_count=int(raw.get("row_count", "-1")),
                case_count=int(raw.get("case_count", "-1")),
                balanced_accuracy=float(raw.get("balanced_accuracy", "nan")),
                brier=float(raw.get("brier", "nan")),
                log_loss=float(raw.get("log_loss", "nan")),
                balanced_accuracy_delta_vs_b=float(
                    raw.get("balanced_accuracy_delta_vs_b", "nan")
                ),
                balanced_accuracy_delta_vs_u=float(
                    raw.get("balanced_accuracy_delta_vs_u", "nan")
                ),
                brier_delta_vs_b=float(raw.get("brier_delta_vs_b", "nan")),
                brier_delta_vs_u=float(raw.get("brier_delta_vs_u", "nan")),
                log_loss_delta_vs_b=float(
                    raw.get("log_loss_delta_vs_b", "nan")
                ),
                log_loss_delta_vs_u=float(
                    raw.get("log_loss_delta_vs_u", "nan")
                ),
                diagnostic_only=_csv_bool(
                    raw.get("diagnostic_only"), field="diagnostic_only"
                ),
            )
        )
    raw_centers = report.get("center_diagnostics")
    if not isinstance(raw_centers, list):
        raise ProtocolError("Fresh HARP oracle center diagnostics are absent.")
    centers: list[HarpFreshCenterOracleDiagnostic] = []
    for raw in raw_centers:
        if not isinstance(raw, Mapping):
            raise ProtocolError("Fresh HARP oracle center diagnostic is malformed.")
        centers.append(
            HarpFreshCenterOracleDiagnostic(
                center=str(raw.get("center", "")),
                action_count=int(raw.get("action_count", -1)),
                physical_matched_action_count=int(
                    raw.get("physical_matched_action_count", -1)
                ),
                budget_bacc_effect_u_minus_b=float(
                    raw.get("budget_bacc_effect_u_minus_b", "nan")
                ),
                budget_brier_effect_u_minus_b=float(
                    raw.get("budget_brier_effect_u_minus_b", "nan")
                ),
                budget_log_loss_effect_u_minus_b=float(
                    raw.get("budget_log_loss_effect_u_minus_b", "nan")
                ),
                best_fixed_action_ids=tuple(
                    str(value) for value in raw.get("best_fixed_action_ids", ())
                ),
                best_fixed_action_bacc=float(
                    raw.get("best_fixed_action_bacc", "nan")
                ),
                best_physical_action_ids=tuple(
                    str(value) for value in raw.get("best_physical_action_ids", ())
                ),
                best_physical_action_bacc=float(
                    raw.get("best_physical_action_bacc", "nan")
                ),
                best_physical_bacc_effect_vs_u=float(
                    raw.get("best_physical_bacc_effect_vs_u", "nan")
                ),
                frozen_lambda_one_policy_balanced_accuracy_delta_vs_u=float(
                    raw.get(
                        "frozen_lambda_one_policy_balanced_accuracy_delta_vs_u",
                        "nan",
                    )
                ),
                frozen_lambda_one_policy_brier_delta_vs_u=float(
                    raw.get("frozen_lambda_one_policy_brier_delta_vs_u", "nan")
                ),
                frozen_lambda_one_policy_log_loss_delta_vs_u=float(
                    raw.get("frozen_lambda_one_policy_log_loss_delta_vs_u", "nan")
                ),
                frozen_lambda_one_policy_route_rate=float(
                    raw.get("frozen_lambda_one_policy_route_rate", "nan")
                ),
                frozen_lambda_one_policy_reference_preserving=(
                    raw.get("frozen_lambda_one_policy_reference_preserving") is True
                ),
                selected_predictive_bacc_effect_vs_u=float(
                    raw.get("selected_predictive_bacc_effect_vs_u", "nan")
                ),
                final_operational_bacc_effect_vs_b=float(
                    raw.get("final_operational_bacc_effect_vs_b", "nan")
                ),
                selected_top1_oracle_tie_credit=float(
                    raw.get("selected_top1_oracle_tie_credit", "nan")
                ),
                selected_mean_true_probability_rank=float(
                    raw.get("selected_mean_true_probability_rank", "nan")
                ),
                selected_mean_log_loss_regret=float(
                    raw.get("selected_mean_log_loss_regret", "nan")
                ),
                selected_mean_brier_regret=float(
                    raw.get("selected_mean_brier_regret", "nan")
                ),
                best_fixed_action_bacc_minus_policy_bacc=float(
                    raw.get("best_fixed_action_bacc_minus_policy_bacc", "nan")
                ),
                case_equal=raw.get("case_equal") is True,
                labels_used_after_route_seal_only=(
                    raw.get("labels_used_after_route_seal_only") is True
                ),
                diagnostic_may_feed_policy=(
                    raw.get("diagnostic_may_feed_policy") is True
                ),
            )
        )
    return HarpFreshOracleDiagnosticResult(
        prelabel_seal_hash=str(report.get("prelabel_seal_hash", "")),
        action_matrix=tuple(matrix),
        center_diagnostics=tuple(centers),
        diagnostic_only=report.get("diagnostic_only") is True,
        labels_used_after_route_seal_only=(
            report.get("labels_used_after_route_seal_only") is True
        ),
        labels_available_to_policy=report.get("labels_available_to_policy") is True,
        policy_or_threshold_update_emitted=(
            report.get("policy_or_threshold_update_emitted") is True
        ),
    )


def validate_harp_fresh_completed_bundle(
    root: str | Path,
    *,
    config: HarpFreshStage70Config,
    allow_pending_validation_report: bool = False,
) -> dict[str, object]:
    output = Path(root).resolve()
    required = set(REQUIRED_CATALOG_MEMBERS)
    if allow_pending_validation_report:
        required.remove("reports/validation_report.json")
        required.remove("reports/run_state.json")
    missing = sorted(member for member in required if not (output / member).is_file())
    if missing:
        raise ProtocolError(f"Fresh HARP completed bundle is incomplete: {missing}.")
    resolved = load_harp_fresh_stage70_config(output / "config.resolved.yaml")
    if resolved.contract_hash != config.contract_hash:
        raise ProtocolError("Fresh HARP resolved config contract drifted.")
    provenance = _json(output / "provenance/input_artifacts.json")
    menu = _json(output / "manifests/prediction_menu_seal.json")
    route = _json(output / "manifests/route_set_seal.json")
    prelabel = _json(output / "manifests/prelabel_content_index.json")
    inference = _json(output / "reports/center_inference.json")
    oracle_report = _json(output / "reports/action_oracle_diagnostics.json")
    leakage = _json(output / "reports/leakage_report.json")
    state = (
        None
        if allow_pending_validation_report
        else _json(output / "reports/run_state.json")
    )
    content = _json(output / "manifests/content_index.json")
    _require_self_hash(provenance, "provenance_hash", role="provenance")
    _require_self_hash(menu, "manifest_hash", role="prediction-menu manifest")
    _require_self_hash(route, "manifest_hash", role="route-set manifest")
    _require_self_hash(inference, "inference_hash", role="center inference")
    _require_self_hash(
        oracle_report, "oracle_report_hash", role="action-oracle report"
    )
    _require_self_hash(leakage, "leakage_hash", role="leakage report")
    if state is not None:
        _require_self_hash(state, "state_hash", role="run state")
    _validate_index(output, prelabel)
    _validate_index(output, content)
    reconstructed_menu = _validate_prediction_checkpoints(output, menu)
    if (
        menu.get("status") != "SEALED_COMPLETE_LABEL_FREE_HARP_MENU"
        or menu.get("action_count") != 90
        or menu.get("prediction_cell_count") != 810
        or menu.get("labels_consumed") is not False
        or route.get("status") != "DURABLE_ALL_ROUTES_SEALED_BEFORE_LABELS"
        or route.get("schema_version")
        != "midogpp_harp_fresh_route_set_manifest_v2"
        or route.get("physical_ablation_action_universe")
        != "Hxe_lambda_one_only"
        or route.get("physical_ablation_reference_preserving_semantics")
        != "eligible_Hxe_lambda_one_else_exact_U"
        or route.get("physical_ablation_selection_labels_used") is not False
        or route.get("prediction_menu_seal_hash")
        != menu.get("prediction_menu_seal_hash")
        or route.get("labels_opened") is not False
        or prelabel.get("status") != "COMPLETE_BEFORE_LABEL_ACCESS"
        or prelabel.get("prelabel_seal_hash") != route.get("prelabel_seal_hash")
        or inference.get("inference_unit") != "target_center"
        or inference.get("inference_unit_count") != 9
        or inference.get("seed_cells_are_inference_units") is not False
        or leakage.get("status") != "PASS"
        or leakage.get("complete_action_menu_before_routing") is not True
        or leakage.get("complete_routes_and_vectors_before_labels") is not True
        or leakage.get(
            "complete_physical_ablation_routes_and_vectors_before_labels"
        )
        is not True
        or leakage.get("labels_used_for_scoring_only") is not True
        or leakage.get("full_action_matrix_scored_after_route_seal_only") is not True
        or leakage.get("oracle_diagnostics_available_to_policy") is not False
        or leakage.get("physical_ablation_selection_labels_used") is not False
        or leakage.get("oracle_diagnostics_may_update_policy_or_thresholds") is not False
        or oracle_report.get("physical_ablation_scored_after_prelabel_seal_only")
        is not True
        or provenance.get("physical_ablation_selection_used_target_labels")
        is not False
        or any(
            leakage.get(key) is not False
            for key in (
                "consumed_test_used",
                "consumed_test_rows_used",
                "consumed_validation_used",
                "consumed_validation_rows_used",
                "consumed_stage90_used",
                "stage50_or_stage90_artifacts_used",
                "policy_update_emitted",
            )
        )
    ):
        raise ProtocolError("Fresh HARP completed protocol boundary drifted.")
    if state is not None:
        validation = _json(output / "reports/validation_report.json")
        validation_hash = _require_self_hash(
            validation, "validation_hash", role="validation report"
        )
        if (
            state.get("status") != "COMPLETE"
            or state.get("prelabel_seal_hash") != route.get("prelabel_seal_hash")
            or state.get("prediction_menu_seal_hash")
            != menu.get("prediction_menu_seal_hash")
            or state.get("route_set_hash") != route.get("route_set_hash")
            or state.get("result_hash") != inference.get("result_hash")
            or state.get("content_hash") != content.get("content_hash")
            or state.get("validation_hash") != validation_hash
            or state.get("labels_used_for_scoring_only") is not True
            or state.get("policy_update_emitted") is not False
        ):
            raise ProtocolError("Fresh HARP COMPLETE commit marker drifted.")
    case_rows = _read_csv(output / "tables/case_metrics.csv")
    center_rows = _read_csv(output / "tables/center_metrics.csv")
    action_rows = _read_csv(output / "tables/action_matrix_metrics.csv")
    oracle = _reconstruct_oracle_result(action_rows, oracle_report)
    summaries = inference.get("summaries")
    if (
        not case_rows
        or len(center_rows) != len(CENTERS)
        or tuple(row.get("center") for row in center_rows) != CENTERS
        or oracle.prelabel_seal_hash != route.get("prelabel_seal_hash")
        or oracle_report.get("oracle_result_hash") != oracle.result_hash
        or oracle_report.get("action_matrix_row_count") != len(action_rows)
        or len(action_rows) != len(CENTERS) * (2 + (len(CENTERS) - 1) * 4)
        or inference.get("oracle_diagnostics_result_hash") != oracle.result_hash
        or not isinstance(summaries, list)
        or tuple(row.get("endpoint") for row in summaries if isinstance(row, Mapping))
        != (
            "balanced_accuracy_improvement",
            "brier_improvement",
            "log_loss_improvement",
        )
    ):
        raise ProtocolError("Fresh HARP scored metric coverage drifted.")
    (
        fallback_count,
        eligible_count,
        physical_fallback_count,
        physical_eligible_count,
    ) = _validate_routed_arrays(
        output / "arrays/routed_probabilities.npz", route, reconstructed_menu
    )
    _validate_prelabel_lineage_hashes(
        config=config,
        provenance=provenance,
        route=route,
        menu=reconstructed_menu,
    )
    checks = {
        "status": "PASS",
        "config_contract_hash": config.contract_hash,
        "policy_lock_hash": route.get("policy_lock_hash"),
        "prediction_menu_seal_hash": menu.get("prediction_menu_seal_hash"),
        "route_set_hash": route.get("route_set_hash"),
        "prelabel_seal_hash": route.get("prelabel_seal_hash"),
        "prelabel_content_hash": prelabel.get("content_hash"),
        "content_hash": content.get("content_hash"),
        "prediction_cell_count": 810,
        "target_center_count": 9,
        "fallback_count": fallback_count,
        "eligible_count": eligible_count,
        "physical_ablation_fallback_count": physical_fallback_count,
        "physical_ablation_eligible_count": physical_eligible_count,
        "physical_ablation_lambda_one_only": True,
        "physical_ablation_reference_preserving_vectors_byte_validated": True,
        "physical_ablation_reference_preserving_semantics": (
            "eligible_Hxe_lambda_one_else_exact_U"
        ),
        "physical_ablation_labels_used_for_selection": False,
        "all_predictions_routes_vectors_sealed_before_labels": True,
        "exact_b_fallback_byte_identical": True,
        "labels_used_for_scoring_only": True,
        "center_level_inference_only": True,
        "matched_budget_reference_action": "U",
        "operational_fallback_action": "B",
        "action_oracle_result_hash": oracle.result_hash,
        "action_oracle_row_count": len(action_rows),
        "action_oracle_diagnostic_only": True,
        "action_oracle_feedback_to_policy": False,
        "consumed_test_or_validation_or_stage90_used": False,
        "policy_update_emitted": False,
    }
    if not allow_pending_validation_report:
        expected = {
            "schema_version": "midogpp_harp_fresh_validation_v1",
            "status": "PASS",
            "validator": "validate_harp_fresh_completed_bundle",
            "checks": checks,
        }
        expected["validation_hash"] = canonical_sha256(expected)
        if _json(output / "reports/validation_report.json") != expected:
            raise ProtocolError("Fresh HARP validation report drifted.")
    return checks


def validate_and_write_harp_fresh_completed_bundle(
    root: str | Path,
    *,
    config: HarpFreshStage70Config,
) -> dict[str, object]:
    checks = validate_harp_fresh_completed_bundle(
        root, config=config, allow_pending_validation_report=True
    )
    payload = {
        "schema_version": "midogpp_harp_fresh_validation_v1",
        "status": "PASS",
        "validator": "validate_harp_fresh_completed_bundle",
        "checks": checks,
    }
    payload["validation_hash"] = canonical_sha256(payload)
    output = Path(root)
    path = output / "reports/validation_report.json"
    _atomic_json(path, payload)
    content = _json(output / "manifests/content_index.json")
    inference = _json(output / "reports/center_inference.json")
    state = {
        "schema_version": "midogpp_harp_fresh_run_state_v1",
        "status": "COMPLETE",
        "prediction_menu_seal_hash": checks["prediction_menu_seal_hash"],
        "route_set_hash": checks["route_set_hash"],
        "prelabel_seal_hash": checks["prelabel_seal_hash"],
        "result_hash": inference.get("result_hash"),
        "content_hash": content.get("content_hash"),
        "validation_hash": payload["validation_hash"],
        "labels_used_for_scoring_only": True,
        "policy_update_emitted": False,
    }
    state["state_hash"] = canonical_sha256(state)
    # This is the sole success commit marker and is written only after every
    # authoritative member and the validation report are durable.
    _atomic_json(output / "reports/run_state.json", state)
    validate_harp_fresh_completed_bundle(root, config=config)
    return checks


__all__ = (
    "REQUIRED_CATALOG_MEMBERS",
    "validate_and_write_harp_fresh_completed_bundle",
    "validate_harp_fresh_completed_bundle",
)
