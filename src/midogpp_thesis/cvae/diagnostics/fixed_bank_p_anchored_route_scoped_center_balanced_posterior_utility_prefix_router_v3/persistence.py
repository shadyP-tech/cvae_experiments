"""Small-product JSON and deduplicated dense-array persistence."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np

from ...protocol import ProtocolError
from ...runtime.artifact_io import sha256_array, sha256_file
from .artifact_io import persist_json, persist_rows
from .constants import (
    CANONICAL_PHYSICAL_ROW_ORDER,
    ENDPOINT_METHOD_IDS,
    SOURCE_PROBABILITY_INDEX_ROW_ORDER,
)
from .endpoint_surface_lineage import (
    ROUTE_ENDPOINT_STATES_SCHEMA_VERSION,
    endpoint_surface_lineage_payload,
)
from .hashing import canonical_hash
from .reports import run_state_payload
from .protocol import FROZEN_PROTOCOL_HASH


def persist_dense_npz(
    path: Path,
    arrays: Mapping[str, object],
    *,
    role: str,
) -> dict[str, object]:
    """Persist each dense value once; JSON contains hashes and shapes only."""

    if not arrays or any(not str(key) or "/" in str(key) for key in arrays):
        raise ProtocolError("CBPUPR dense-array namespace drifted.")
    ordered: dict[str, np.ndarray] = {}
    for key in sorted(arrays):
        value = np.asarray(arrays[key])
        if value.dtype.kind == "f":
            value = np.ascontiguousarray(value, dtype=np.float32)
        elif value.dtype.kind in "iu?":
            value = np.ascontiguousarray(value)
        else:
            raise ProtocolError("CBPUPR dense arrays must be numeric.")
        if not np.isfinite(value).all():
            raise ProtocolError("CBPUPR dense array contains nonfinite values.")
        ordered[str(key)] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ProtocolError("CBPUPR dense-array path is a symlink.")
    with tempfile.NamedTemporaryFile(
        mode="w+b", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **ordered)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if path.exists():
            if not path.is_file() or sha256_file(path) != sha256_file(temporary):
                raise ProtocolError("CBPUPR refuses repair of a dense array store.")
            temporary.unlink()
        else:
            os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    payload = {
        "schema_version": "fixed_bank_cbpupr_dense_array_manifest_v1",
        "role": str(role),
        "member": path.name,
        "store_sha256": sha256_file(path),
        "arrays": [
            {
                "key": key,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "array_sha256": sha256_array(value),
            }
            for key, value in ordered.items()
        ],
        "raw_labels_persisted": False,
        "sample_or_image_paths_persisted": False,
    }
    return {**payload, "manifest_hash": canonical_hash(payload)}


def persist_phase(
    root: Path,
    phase_id: str,
    *,
    summary: Mapping[str, object],
    rows: object | None = None,
    arrays: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Persist one phase without embedding dense vectors in JSON."""

    safe = str(phase_id).strip().casefold().replace("-", "_")
    if not safe or not safe.replace("_", "").isalnum():
        raise ProtocolError("CBPUPR phase identity drifted.")
    payload = dict(summary)
    if arrays:
        manifest = persist_dense_npz(root / "arrays" / f"{safe}.npz", arrays, role=safe)
        persist_json(root / "manifests" / f"{safe}_arrays.json", manifest)
        payload["dense_array_manifest_hash"] = manifest["manifest_hash"]
    if rows is not None:
        persist_rows(
            root / "tables" / f"{safe}.json",
            rows,
            schema_version=f"fixed_bank_cbpupr_{safe}_rows_v1",
            allow_empty=True,
        )
    payload.update(
        {
            "schema_version": f"fixed_bank_cbpupr_{safe}_phase_v1",
            "phase_id": safe,
            "raw_labels_persisted": False,
        }
    )
    payload["phase_hash"] = canonical_hash(payload)
    persist_json(root / "reports" / f"{safe}.json", payload)
    return payload


def write_run_state(
    root: Path,
    *,
    status: str,
    phase: str,
    error: str | None = None,
    error_class: str | None = None,
) -> None:
    # Run state is the only intentionally replaceable report inside one launch.
    from ...runtime.artifact_io import atomic_json

    atomic_json(
        root / "reports/run_state.json",
        run_state_payload(
            status=status,
            phase=phase,
            error=error,
            error_class=error_class,
        ),
    )


def persist_admission(
    root: Path,
    *,
    config: object,
    provenance: Mapping[str, Mapping[str, object]],
    frame: object,
    pre_gpu_firewall: Mapping[str, object],
) -> None:
    """Persist only hash/count admission products; launch files already exist."""

    from .config_payloads import (
        canonical_action_library_payload,
        canonical_policy_menu_payload,
    )
    from .reports import protocol_manifest_payload

    persist_json(
        root / "manifests/protocol_manifest.json",
        protocol_manifest_payload(
            config,
            protocol_hash=FROZEN_PROTOCOL_HASH,
            provenance=provenance,
            cache_binding_hash=canonical_hash(dict(getattr(frame, "cache_binding"))),
            pre_gpu_firewall=pre_gpu_firewall,
        ),
    )
    persist_json(
        root / "manifests/action_library.json",
        canonical_action_library_payload(),
    )
    persist_json(
        root / "manifests/policy_menu.json", canonical_policy_menu_payload()
    )


def persist_physical_surface(
    root: Path,
    *,
    physical: object,
    surface: object,
    probability_index: object,
) -> None:
    rows = tuple(probability_index)
    row_payloads = [row.to_payload() for row in rows]
    if len(row_payloads) != 90 or any(
        row.get("row_order") != SOURCE_PROBABILITY_INDEX_ROW_ORDER
        for row in row_payloads
    ):
        raise ProtocolError("CBPUPR source probability index order drifted.")
    persist_rows(
        root / "tables/exact_nine_probability_index.json",
        row_payloads,
        schema_version="fixed_bank_cbpupr_exact_nine_probability_index_v1",
    )
    payload = {
        "schema_version": "fixed_bank_cbpupr_physical_surface_seal_v1",
        "surface_hash": str(getattr(surface, "surface_hash")),
        "probability_store_hash": str(getattr(surface, "probability_store_hash")),
        "source_stream_lock_hash": str(
            getattr(getattr(physical, "canonical_source_cache"), "lock_hash")
        ),
        "global_prediction_seal_hash": str(
            getattr(getattr(physical, "prediction"), "seal_hash")
        ),
        "probability_index_hash": canonical_hash(row_payloads),
        "source_probability_index_row_order": SOURCE_PROBABILITY_INDEX_ROW_ORDER,
        "canonical_physical_row_order": CANONICAL_PHYSICAL_ROW_ORDER,
        "target_probability_cell_count": 810,
        "labels_used": False,
    }
    persist_json(
        root / "manifests/physical_surface_seal.json",
        {**payload, "physical_surface_seal_hash": canonical_hash(payload)},
    )


def persist_preterminal(root: Path, result: object) -> None:
    """Persist compact rows plus four hash-indexed float32 NPZ stores."""

    candidates = getattr(result, "candidates")
    decisions = getattr(result, "decisions")
    plans = getattr(candidates, "plan_seal")
    endpoint_products = tuple(getattr(candidates, "endpoint_products"))
    endpoint_surface_lineage = endpoint_surface_lineage_payload(
        endpoint_products
    )
    physical_surface_hash = str(
        endpoint_surface_lineage["physical_surface_hash"]
    )
    center_surface_hashes = dict(
        endpoint_surface_lineage["center_surface_hashes"]
    )
    persist_json(root / "manifests/outer_plan_seal.json", plans.to_payload())
    persist_rows(
        root / "tables/outer_plans.json",
        [row.to_payload() for row in plans.outer_plans],
        schema_version="fixed_bank_cbpupr_outer_plans_v1",
    )
    persist_rows(
        root / "tables/physical_fingerprints.json",
        [
            row.summary_payload()
            for row in (
                *getattr(candidates, "primary_fingerprints"),
                *getattr(candidates, "blocked_fingerprints"),
            )
        ],
        schema_version="fixed_bank_cbpupr_physical_fingerprints_v1",
    )
    audit = candidates.firewall.audit_payload()
    if audit.get("terminal_opened") is not False:
        raise ProtocolError(
            "CBPUPR preterminal products must be durable before terminal labels open."
        )
    support_events = [
        row for row in audit["events"] if str(row["role"]).startswith("outer_support::")
    ]
    persist_rows(
        root / "tables/route_support_capabilities.json",
        support_events,
        schema_version="fixed_bank_cbpupr_route_support_capabilities_v1",
    )

    endpoint_arrays: dict[str, object] = {}
    endpoint_rows = []
    for product in endpoint_products:
        for prediction in product.predictions:
            for method, values in prediction.probabilities.items():
                key = f"{prediction.prediction_hash}__{method}"
                endpoint_arrays[key] = values
                endpoint_rows.append(
                    {
                        "target_center": prediction.center,
                        "case_id": prediction.case_id,
                        "method_id": method,
                        "sample_ids": list(prediction.sample_ids),
                        "array_key": key,
                        "prediction_hash": prediction.prediction_hash,
                        "state_hash": prediction.state_hash,
                        "physical_surface_hash": physical_surface_hash,
                        "center_surface_hash": center_surface_hashes[
                            prediction.center
                        ],
                    }
                )
    endpoint_manifest = persist_dense_npz(
        root / "arrays/route_endpoint_probabilities.npz",
        endpoint_arrays,
        role="route_endpoint_probabilities",
    )
    endpoint_manifest = _augment_dense_manifest(
        endpoint_manifest, "index_rows", endpoint_rows
    )
    endpoint_manifest = _augment_dense_manifest(
        endpoint_manifest,
        "endpoint_surface_lineage",
        endpoint_surface_lineage,
    )
    persist_json(
        root / "manifests/route_endpoint_probability_index.json",
        endpoint_manifest,
    )
    persist_rows(
        root / "tables/route_endpoint_states.json",
        [
            {
                "target_center": product.target_center,
                "held_case_id": case,
                "physical_surface_hash": physical_surface_hash,
                "center_surface_hash": center_surface_hashes[
                    product.target_center
                ],
                "state": state.to_payload(),
            }
            for product in endpoint_products
            for case, state in product.states
        ],
        schema_version=ROUTE_ENDPOINT_STATES_SCHEMA_VERSION,
    )

    persist_rows(
        root / "tables/pseudo_source_priors.json",
        [row.to_payload() for row in candidates.pseudo_source_prior_evidence],
        schema_version="fixed_bank_cbpupr_pseudo_source_priors_v1",
    )
    pseudo_endpoint_arrays: dict[str, object] = {}
    pseudo_endpoint_rows = []
    for evidence in candidates.pseudo_endpoint_evidence:
        prediction = evidence.prediction
        array_keys = {
            method: (
                f"{prediction.prediction_hash}__H_{evidence.outer_center}__{method}"
            )
            for method in ENDPOINT_METHOD_IDS
        }
        for method in ENDPOINT_METHOD_IDS:
            pseudo_endpoint_arrays[array_keys[method]] = prediction.probabilities[
                method
            ]
        pseudo_endpoint_rows.append(
            {
                **evidence.to_payload(),
                "physical_surface_hash": physical_surface_hash,
                "center_surface_hash": center_surface_hashes[
                    prediction.center
                ],
                "array_keys": array_keys,
            }
        )
    pseudo_endpoint_manifest = persist_dense_npz(
        root / "arrays/pseudo_route_endpoint_probabilities.npz",
        pseudo_endpoint_arrays,
        role="pseudo_route_endpoint_probabilities",
    )
    pseudo_endpoint_manifest = _augment_dense_manifest(
        pseudo_endpoint_manifest, "index_rows", pseudo_endpoint_rows
    )
    pseudo_endpoint_manifest = _augment_dense_manifest(
        pseudo_endpoint_manifest,
        "endpoint_surface_lineage",
        endpoint_surface_lineage,
    )
    persist_json(
        root / "manifests/pseudo_route_endpoint_probability_index.json",
        pseudo_endpoint_manifest,
    )

    posterior_arrays: dict[str, object] = {}
    posterior_rows = []
    for row in candidates.posterior_predictions:
        key = row.prediction_hash
        posterior_arrays[key] = row.natural_probabilities
        posterior_rows.append(
            {
                "target_center": row.target_center,
                "held_case_id": row.held_case_id,
                "control_id": row.control_id,
                "sample_ids": list(row.sample_ids),
                "array_key": key,
                "prediction_hash": row.prediction_hash,
                "model_hash": row.model_hash,
                "fingerprint_hash": row.fingerprint_hash,
                "sample_identity_hash": canonical_hash(list(row.sample_ids)),
            }
        )
    posterior_manifest = persist_dense_npz(
        root / "arrays/target_local_posterior_probabilities.npz",
        posterior_arrays,
        role="target_local_posterior_probabilities",
    )
    posterior_manifest = _augment_dense_manifest(
        posterior_manifest, "index_rows", posterior_rows
    )
    persist_json(
        root / "manifests/target_local_posterior_probability_index.json",
        posterior_manifest,
    )
    persist_rows(
        root / "tables/target_local_posterior_predictions.json",
        posterior_rows,
        schema_version="fixed_bank_cbpupr_target_local_posterior_predictions_v1",
    )
    persist_rows(
        root / "tables/target_local_posterior_models.json",
        [_posterior_model_summary(row) for row in candidates.posterior_models],
        schema_version="fixed_bank_cbpupr_target_local_posterior_models_v1",
    )
    persist_rows(
        root / "tables/pseudo_posterior_references.json",
        [row.to_payload() for row in candidates.pseudo_posterior_references],
        schema_version="fixed_bank_cbpupr_pseudo_posterior_references_v1",
    )

    all_runtime = (*candidates.target_candidates, *candidates.pseudo_candidates)
    candidate_arrays: dict[str, object] = {}
    expected_rows = []
    eligibility_rows = []
    runtime_rows = []
    for runtime in all_runtime:
        for candidate, eligibility in zip(
            runtime.candidates, runtime.eligibility, strict=True
        ):
            candidate_arrays.setdefault(
                candidate.action_hash, candidate.probabilities.as_array()
            )
            expected_rows.append(candidate.estimate.to_payload())
            eligibility_rows.append(
                {
                    "outer_center": runtime.outer_center,
                    "center": runtime.center,
                    "case_id": runtime.case_id,
                    "control_id": runtime.control_id,
                    **eligibility.to_payload(),
                }
            )
        runtime_rows.append(_candidate_runtime_summary(runtime))
    candidate_manifest = persist_dense_npz(
        root / "arrays/candidate_probabilities.npz",
        candidate_arrays,
        role="candidate_probabilities",
    )
    candidate_manifest = _augment_dense_manifest(
        candidate_manifest, "runtime_rows", runtime_rows
    )
    persist_json(
        root / "manifests/candidate_probability_index.json", candidate_manifest
    )
    persist_rows(
        root / "tables/expected_utility_predictions.json",
        expected_rows,
        schema_version="fixed_bank_cbpupr_expected_utility_predictions_v1",
        allow_empty=True,
    )
    persist_rows(
        root / "tables/candidate_eligibility.json",
        eligibility_rows,
        schema_version="fixed_bank_cbpupr_candidate_eligibility_v1",
        allow_empty=True,
    )
    persist_rows(
        root / "tables/target_candidate_policies.json",
        runtime_rows[: len(candidates.target_candidates)],
        schema_version="fixed_bank_cbpupr_target_candidate_policies_v1",
    )
    persist_rows(
        root / "tables/pseudo_candidate_policies.json",
        runtime_rows[len(candidates.target_candidates) :],
        schema_version="fixed_bank_cbpupr_pseudo_candidate_policies_v1",
    )
    persist_rows(
        root / "tables/pseudo_policy_replays.json",
        [
            *[
                {"record_type": "donor_case_replay", **row.to_payload()}
                for row in decisions.donor_replays
            ],
            *[
                {"record_type": "policy_replay", **_policy_replay_summary(row)}
                for row in decisions.policy_replays
            ],
            *[
                {"record_type": "policy_replay_diagnostic", **dict(row)}
                for row in decisions.policy_replay_diagnostics
            ],
        ],
        schema_version="fixed_bank_cbpupr_pseudo_policy_replays_v1",
        allow_empty=True,
    )
    persist_rows(
        root / "tables/donor_bias_calibrations.json",
        [row.to_payload() for row in decisions.utility_calibrations],
        schema_version="fixed_bank_cbpupr_donor_bias_calibrations_v1",
    )
    persist_rows(
        root / "tables/prefix_decisions.json",
        [
            {
                "center": row.center,
                "method_id": row.method_id,
                **_prefix_selection_summary(row.prefix_selection),
            }
            for row in decisions.route_decisions
        ],
        schema_version="fixed_bank_cbpupr_prefix_decisions_v1",
    )
    persist_rows(
        root / "tables/transport_diagnostics.json",
        [
            *[row.to_payload() for row in decisions.structural_gates],
            *[row.to_payload() for row in decisions.transport_audits],
        ],
        schema_version="fixed_bank_cbpupr_transport_diagnostics_v1",
    )

    composed_arrays = {
        f"{method}__{center}": np.concatenate(
            [
                np.asarray(decisions.probabilities[method][center][case], dtype=np.float32)
                for case in decisions.sample_ids[center]
            ]
        )
        for method in decisions.probabilities
        for center in decisions.probabilities[method]
    }
    composed_manifest = persist_dense_npz(
        root / "arrays/composed_probabilities.npz",
        composed_arrays,
        role="endpoint_and_composed_probabilities",
    )
    persist_json(
        root / "manifests/composed_probability_index.json", composed_manifest
    )
    persist_rows(
        root / "tables/composed_predictions.json",
        [
            {
                "method_id": method,
                "target_center": center,
                "array_key": f"{method}__{center}",
                "case_count": len(decisions.sample_ids[center]),
                "case_ids": list(decisions.sample_ids[center]),
                "case_identity_hash": canonical_hash(list(decisions.sample_ids[center])),
                "policy_hash": _policy_hash_for_method(
                    decisions=decisions, method=method, center=center
                ),
                "control_policy": _control_policy_for_method(
                    decisions=decisions, method=method, center=center
                ),
                "selected_case_ids": list(
                    _selected_cases_for_method(
                        candidates=candidates,
                        decisions=decisions,
                        method=method,
                        center=center,
                    )
                ),
            }
            for method in decisions.probabilities
            for center in decisions.probabilities[method]
        ],
        schema_version="fixed_bank_cbpupr_composed_predictions_v1",
    )
    persist_rows(
        root / "tables/route_decisions.json",
        [_route_decision_summary(row) for row in decisions.route_decisions],
        schema_version="fixed_bank_cbpupr_route_decisions_v1",
    )
    persist_rows(
        root / "tables/gate_funnel.json",
        [result.gate_funnel.to_payload()],
        schema_version="fixed_bank_cbpupr_gate_funnel_v1",
    )
    persist_rows(
        root / "tables/information_diagnostics.json",
        [_information_summary(result)],
        schema_version="fixed_bank_cbpupr_information_diagnostics_v1",
    )
    barrier = {
        "schema_version": "fixed_bank_cbpupr_decision_barrier_v1",
        "candidate_seal_hash": candidates.target_candidate_seal_hash,
        "pre_evaluation_seal_hash": candidates.pre_evaluation_seal_hash,
        "replay_calibration_seal_hash": decisions.replay_calibration_seal_hash,
        "pseudo_evaluation_opened_after_candidate_seal": True,
        "target_evaluation_opened": False,
    }
    persist_json(
        root / "manifests/decision_barrier.json",
        {**barrier, "decision_barrier_hash": canonical_hash(barrier)},
    )
    aggregate = {
        "schema_version": "fixed_bank_cbpupr_preterminal_aggregate_seal_v1",
        "aggregate_seal_hash": decisions.aggregate_seal_hash,
        "preterminal_hash": result.preterminal_hash,
        "target_evaluation_opened": False,
    }
    persist_json(root / "manifests/preterminal_aggregate_seal.json", aggregate)


def persist_label_capability_report(
    root: Path, capability_report: Mapping[str, object]
) -> None:
    """Persist the final audit only after the terminal capability has opened."""

    if capability_report.get("terminal_opened") is not True:
        raise ProtocolError("CBPUPR final capability report precedes terminal opening.")
    persist_json(root / "reports/label_capability_report.json", capability_report)


def persist_terminal(
    root: Path,
    *,
    terminal: object,
    leakage_report: Mapping[str, object],
    publication_decision: Mapping[str, object],
    runtime_summary: Mapping[str, object],
    aggregate_seal_hash: str,
) -> None:
    persist_rows(
        root / "tables/terminal_method_metrics.json",
        list(terminal.method_rows),
        schema_version="fixed_bank_cbpupr_terminal_method_metrics_v1",
    )
    persist_rows(
        root / "tables/terminal_center_contrasts.json",
        list(terminal.center_rows),
        schema_version="fixed_bank_cbpupr_terminal_center_contrasts_v1",
    )
    persist_rows(
        root / "tables/terminal_case_oracles.json",
        list(terminal.oracle_rows),
        schema_version="fixed_bank_cbpupr_terminal_case_oracles_v1",
    )
    persist_json(
        root / "manifests/terminal_evaluation_seal.json",
        {
            "schema_version": "fixed_bank_cbpupr_terminal_evaluation_seal_v1",
            "aggregate_seal_hash": aggregate_seal_hash,
            "terminal_seal_hash": terminal.terminal_seal_hash,
            "terminal_result_hash": terminal.result_hash,
            "raw_labels_persisted": False,
        },
    )
    persist_json(root / "reports/diagnostic_summary.json", terminal.diagnostic_summary)
    persist_json(root / "reports/leakage_report.json", leakage_report)
    persist_json(root / "reports/publication_decision.json", publication_decision)
    persist_json(root / "reports/runtime_summary.json", runtime_summary)


def persist_validation_report(root: Path, payload: Mapping[str, object]) -> None:
    persist_json(root / "reports/validation_report.json", payload)


def _posterior_model_summary(row: object) -> dict[str, object]:
    # Persist the complete fitted model contract.  Fresh validators can then
    # reconstruct every held-case eta from the independently rebuilt physical
    # fingerprint instead of trusting an opaque model hash.
    return {
        **row.to_payload(),
        "structural_reference_reuse_allowed": True,
    }


def _candidate_runtime_summary(row: object) -> dict[str, object]:
    return {
        "outer_center": row.outer_center,
        "center": row.center,
        "case_id": row.case_id,
        "control_id": row.control_id,
        "descriptor_count": row.descriptor_count,
        "no_crossing_count": row.no_crossing_count,
        "candidate_hashes": [value.action_hash for value in row.candidates],
        "selected_candidate_hash": None if row.selected_candidate is None else row.selected_candidate.action_hash,
        "posterior_model_reference_count": row.posterior_model_reference_count,
        "posterior_fit_increment": 0,
        "posterior_refit": False,
        "posterior_refit_performed_in_candidate_runtime": False,
        "sealed_posterior_reference_reused": row.outer_center != row.center,
        "posterior_model_hash": row.posterior_model_hash,
        "support_capability_hash": row.support_capability_hash,
        "source_excluded_centers": list(row.source_excluded_centers),
        "source_excluded_centers_role": (
            "actionable_endpoint_source_selection_only_not_posterior_"
            "fingerprint_covariates"
        ),
        "endpoint_lineage_hash": row.endpoint_lineage_hash,
        "runtime_hash": row.runtime_hash,
    }


def _information_summary(result: object) -> dict[str, object]:
    selected = sum(
        row.selected_candidate is not None for row in result.candidates.target_candidates
    )
    return {
        "target_route_control_count": len(result.candidates.target_candidates),
        "selected_case_control_count": selected,
        "candidate_selection_rate": selected / len(result.candidates.target_candidates),
        "numeric_transport_is_authorization_gate": False,
        "formal_claim_authorized": False,
    }


def _prefix_selection_summary(selection: object) -> dict[str, object]:
    ranked = tuple(selection.ranked_candidates)
    return {
        "ranked_candidates": [
            {
                "center": row.candidate.center,
                "case_id": row.candidate.case_id,
                "control_id": row.candidate.control_id,
                "action_hash": row.candidate.action_hash,
                "policy_hash": row.policy_hash,
                "corrected_utility": row.corrected_utility.to_payload(),
                "calibration_hash": row.calibration_hash,
            }
            for row in ranked
        ],
        "evaluations": [row.to_payload() for row in selection.evaluations],
        "selected_k": selection.selected_k,
        "selected_prefix_hash": selection.selected_prefix_hash,
        "selected_case_ids": [
            row.candidate.case_id for row in ranked[: selection.selected_k]
        ],
        "selected_candidate_hashes": [
            row.candidate.action_hash for row in ranked[: selection.selected_k]
        ],
        "selection_hash": selection.selection_hash,
        "dense_probabilities_persisted": False,
    }


def _route_decision_summary(row: object) -> dict[str, object]:
    composition = row.composition
    return {
        "center": row.center,
        "method_id": row.method_id,
        "action": row.action,
        "reason_codes": list(row.reason_codes),
        "prefix_selection": _prefix_selection_summary(row.prefix_selection),
        "composition": {
            "selected_case_ids": list(composition.selected_case_ids),
            "selected_candidate_hashes": list(
                composition.selected_candidate_hashes
            ),
            "probability_sha256": composition.probabilities.sha256,
            "changed_probability_count": composition.changed_probability_count,
            "exact_p": composition.exact_p,
            "composition_hash": composition.composition_hash,
            "dense_probabilities_persisted": False,
        },
        "structural_transport": row.structural_transport.to_payload(),
        "utility_calibration_hash": row.utility_calibration_hash,
        "candidate_runtime_hashes": list(row.candidate_runtime_hashes),
        "policy_replay_bias_used": False,
        "decision_hash": row.decision_hash,
    }


def _policy_replay_summary(row: object) -> dict[str, object]:
    return {
        "selection": _prefix_selection_summary(row.selection),
        "replay": row.replay.to_payload(),
        "candidate_calibration_hash": row.candidate_calibration_hash,
        "candidate_runtime_hashes": list(row.candidate_runtime_hashes),
        "runtime_hash": row.runtime_hash,
        "dense_probabilities_persisted": False,
    }


def _augment_dense_manifest(
    manifest: Mapping[str, object], key: str, rows: object
) -> dict[str, object]:
    payload = {
        name: value for name, value in dict(manifest).items() if name != "manifest_hash"
    }
    payload[str(key)] = rows
    return {**payload, "manifest_hash": canonical_hash(payload)}


def _selected_cases_for_method(
    *, candidates: object, decisions: object, method: str, center: str
) -> tuple[str, ...]:
    route = next(
        (
            row
            for row in decisions.route_decisions
            if row.method_id == method and row.center == center
        ),
        None,
    )
    if route is not None:
        return tuple(route.composition.selected_case_ids)
    control = next(
        (
            row
            for observed_method, observed_center, row in decisions.control_policies
            if observed_method == method and observed_center == center
        ),
        None,
    )
    if control is None:
        return ()
    case_by_hash = {
        runtime.selected_candidate.action_hash: runtime.case_id
        for runtime in candidates.target_candidates
        if runtime.center == center
        and runtime.outer_center == center
        and runtime.control_id == "IDENTITY"
        and runtime.selected_candidate is not None
    }
    try:
        return tuple(case_by_hash[value] for value in control.selected_candidate_hashes)
    except KeyError as exc:
        raise ProtocolError("CBPUPR control selection lacks target candidate lineage.") from exc


def _policy_hash_for_method(*, decisions: object, method: str, center: str) -> str | None:
    route = next(
        (
            row
            for row in decisions.route_decisions
            if row.method_id == method and row.center == center
        ),
        None,
    )
    if route is not None:
        return str(route.decision_hash)
    control = next(
        (
            row
            for observed_method, observed_center, row in decisions.control_policies
            if observed_method == method and observed_center == center
        ),
        None,
    )
    return None if control is None else str(control.policy_hash)


def _control_policy_for_method(
    *, decisions: object, method: str, center: str
) -> dict[str, object] | None:
    if any(
        row.method_id == method and row.center == center
        for row in decisions.route_decisions
    ):
        return None
    control = next(
        (
            row
            for observed_method, observed_center, row in decisions.control_policies
            if observed_method == method and observed_center == center
        ),
        None,
    )
    return None if control is None else control.to_payload()


__all__ = (
    "persist_admission",
    "persist_dense_npz",
    "persist_label_capability_report",
    "persist_phase",
    "persist_physical_surface",
    "persist_preterminal",
    "persist_terminal",
    "persist_validation_report",
    "write_run_state",
)
