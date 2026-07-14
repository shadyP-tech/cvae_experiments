"""Outer artifact writer and fail-closed preservation-bundle validation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

from ...real_features.classifier_reference.artifacts import stable_hash
from ...real_features.classifier_reference.protocol import ProtocolError
from ...real_features.classifier_reference.schemas.midogpp import (
    MIDOGPP_EXCLUDED_CENTERS,
)
from ..generation_samplers import STANDARD_SAMPLER
from ..objectives import ISOTROPIC_OBJECTIVE, TASK_FISHER_OBJECTIVE
from ..reporting import prepare_artifact_dirs, write_csv_rows, write_json
from .prior_recovery_artifact_shared import (
    _assert_columns,
    _assert_common_metric_identity,
    _read_csv,
    _read_json,
    _require_files,
    _validate_centers,
    _validate_cross_arm_generation_budgets,
    _validate_identity_rows,
    _validate_metric_provenance,
    _validate_metric_values,
    _validate_sampler_rows,
    _validate_workspace_provenance,
)
from .prior_recovery_common import PRIOR_RECOVERY_METHOD, canonical_rows_hash
from .prior_recovery_config import (
    OuterPriorRecoveryConfig,
    outer_decision_contract_hash,
    recipe_contract_hash,
)
from .prior_recovery_decision import aggregate_outer, outer_coverage
from .prior_recovery_provenance import validate_provenance_indices
from .prior_recovery_runtime_cache import validate_feature_frame_index
from .prior_recovery_timing import validate_runtime_reports, write_run_state
from .prior_recovery_schema import OUTER_METRIC_COLUMNS, OUTER_METRIC_SCHEMA
from .runtime import EvaluationKey, GenerationKey
from .source_inner_selection import recipe_lock_from_payload


def write_outer_bundle(
    root: Path,
    *,
    metric_rows: Sequence[Mapping[str, object]],
    sampler_rows: Sequence[Mapping[str, object]],
    paired_delta_rows: Sequence[Mapping[str, object]],
    aggregation_rows: Sequence[Mapping[str, object]],
    checkpoint_audit_rows: Sequence[Mapping[str, object]],
    identity_audit_rows: Sequence[Mapping[str, object]],
    protocol_manifest: Mapping[str, object],
    coverage_manifest: Mapping[str, object],
    decision_report: Mapping[str, object],
    leakage_report: Mapping[str, object],
) -> Path:
    root = prepare_artifact_dirs(root)
    write_csv_rows(
        root / "tables/preservation_metrics.csv",
        metric_rows,
        OUTER_METRIC_COLUMNS,
    )
    write_csv_rows(root / "tables/sampler_realizations.csv", sampler_rows)
    write_csv_rows(root / "tables/paired_deltas.csv", paired_delta_rows)
    write_csv_rows(root / "tables/aggregation_summary.csv", aggregation_rows)
    write_csv_rows(
        root / "tables/checkpoint_reuse_audit.csv",
        checkpoint_audit_rows,
    )
    write_csv_rows(root / "tables/identity_overlap_audit.csv", identity_audit_rows)
    write_json(root / "manifests/protocol_manifest.json", dict(protocol_manifest))
    write_json(root / "manifests/coverage_manifest.json", dict(coverage_manifest))
    write_json(root / "reports/decision_report.json", dict(decision_report))
    write_json(root / "reports/leakage_report.json", dict(leakage_report))
    write_run_state(
        root,
        protocol_hash=str(protocol_manifest["protocol_hash"]),
        mode="outer",
        status="COMPLETE",
    )
    try:
        validate_outer_bundle(root)
    except Exception:
        write_run_state(
            root,
            protocol_hash=str(protocol_manifest["protocol_hash"]),
            mode="outer",
            status="FAILED",
        )
        raise
    return root


def validate_outer_bundle(
    root: Path,
    *,
    expected_config: OuterPriorRecoveryConfig | None = None,
) -> None:
    root = Path(root)
    required = (
        "tables/preservation_metrics.csv",
        "tables/sampler_realizations.csv",
        "tables/paired_deltas.csv",
        "tables/aggregation_summary.csv",
        "tables/checkpoint_reuse_audit.csv",
        "tables/identity_overlap_audit.csv",
        "manifests/protocol_manifest.json",
        "manifests/coverage_manifest.json",
        "manifests/checkpoint_index.json",
        "manifests/task_fisher_index.json",
        "manifests/feature_frame_index.json",
        "reports/decision_report.json",
        "reports/leakage_report.json",
        "tables/runtime_timings.csv",
        "reports/runtime_summary.json",
        "reports/run_state.json",
    )
    _require_files(root, required)
    protocol = _read_json(root / "manifests/protocol_manifest.json")
    _validate_workspace_provenance(root, protocol=protocol, mode="outer")
    coverage = _read_json(root / "manifests/coverage_manifest.json")
    decision = _read_json(root / "reports/decision_report.json")
    leakage = _read_json(root / "reports/leakage_report.json")
    checkpoint_index, fisher_index = validate_provenance_indices(root)
    rows = _read_csv(root / "tables/preservation_metrics.csv")
    frame_index = validate_feature_frame_index(
        root,
        expected_frame_hashes={str(row.get("frame_hash", "")) for row in rows},
    )
    sampler_rows = _read_csv(root / "tables/sampler_realizations.csv")
    paired_rows = _read_csv(root / "tables/paired_deltas.csv")
    aggregation_rows = _read_csv(root / "tables/aggregation_summary.csv")
    audits = _read_csv(root / "tables/checkpoint_reuse_audit.csv")
    identity_rows = _read_csv(root / "tables/identity_overlap_audit.csv")
    _assert_columns(rows, OUTER_METRIC_COLUMNS, "preservation_metrics.csv")
    _validate_outer_protocol(protocol, expected_config=expected_config)
    validate_runtime_reports(
        root,
        protocol_hash=str(protocol["protocol_hash"]),
        mode="outer",
        checkpoint_index=checkpoint_index,
        frame_index=frame_index,
    )
    _validate_outer_rows(rows, protocol=protocol)
    _validate_sampler_rows(sampler_rows, metric_rows=rows)
    _validate_metric_provenance(
        rows,
        checkpoint_index=checkpoint_index,
        fisher_index=fisher_index,
        protocol=protocol,
    )
    heldouts = tuple(str(value) for value in protocol["heldout_centers"])
    eligible = tuple(str(value) for value in protocol["eligible_centers"])
    _validate_identity_rows(
        identity_rows,
        heldouts=heldouts,
        eligible=eligible,
        source_inner=False,
    )
    _validate_checkpoint_audits(
        audits,
        rows=rows,
        protocol=protocol,
        checkpoint_index=checkpoint_index,
    )
    expected_coverage = _coverage_from_protocol(protocol, rows)
    if coverage != expected_coverage:
        raise ProtocolError("Outer coverage manifest does not match metric rows.")
    lock_payloads = protocol["locked_recipes"]
    assert isinstance(lock_payloads, Mapping)
    locks = {
        str(center): recipe_lock_from_payload(payload)
        for center, payload in lock_payloads.items()
        if isinstance(payload, Mapping)
    }
    decision_contract = protocol["outer_decision_contract"]
    assert isinstance(decision_contract, Mapping)
    config_view = SimpleNamespace(
        heldout_centers=tuple(
            str(value) for value in protocol["heldout_centers"]
        ),
        training_seeds=tuple(int(value) for value in protocol["training_seeds"]),
        generation_seeds=tuple(
            int(value) for value in protocol["generation_seeds"]
        ),
        positive_claim_min_ratio=float(
            decision_contract["positive_claim_min_ratio"]
        ),
        positive_claim_min_center_wins=int(
            decision_contract["positive_claim_min_center_wins"]
        ),
        safety_max_bacc_regression=float(
            decision_contract["safety_max_bacc_regression"]
        ),
    )
    expected_aggregation, expected_paired, expected_decision, recomputed_coverage = (
        aggregate_outer(config_view, rows, locks)
    )
    if recomputed_coverage != coverage:
        raise ProtocolError(
            "Outer aggregation coverage differs from the coverage manifest."
        )
    if canonical_rows_hash(aggregation_rows) != canonical_rows_hash(
        expected_aggregation
    ):
        raise ProtocolError(
            "Outer aggregation table does not recompute from metric rows."
        )
    if canonical_rows_hash(paired_rows) != canonical_rows_hash(expected_paired):
        raise ProtocolError(
            "Outer paired-delta table does not recompute from metric rows."
        )
    if stable_hash(decision) != stable_hash(expected_decision):
        raise ProtocolError(
            "Outer decision report does not recompute from metric rows."
        )
    if coverage.get("status") != "PASS":
        if decision.get("claim_scope") != "diagnostic_only":
            raise ProtocolError(
                "Incomplete outer factorial was not downgraded to diagnostic_only."
            )
    if decision.get("factorial_coverage_pass") is not (
        coverage.get("status") == "PASS"
    ):
        raise ProtocolError("Outer decision coverage flag is inconsistent.")
    expected_leakage = {
        "status": "PASS" if coverage.get("status") == "PASS" else "FAIL",
        "target_eval_labels_used_for_scoring_only": True,
        "target_eval_labels_used_for_selection": False,
        "outer_metrics_may_feed_model_recipe": False,
        "may_feed_deployable_selection": False,
        "center_4_excluded": True,
        "identity_overlap_status": "PASS",
        "routing_performed": False,
        "composition_performed": False,
        "query_object": "none",
        "forbidden_reuse": [
            "expert_bank_evidence",
            "routing_evidence",
            "expert_selection_evidence",
            "nelbo_compatibility_evidence",
        ],
    }
    if leakage != expected_leakage:
        raise ProtocolError(
            "Outer leakage report does not match the fail-closed contract."
        )


def _validate_outer_protocol(
    protocol: Mapping[str, object],
    *,
    expected_config: OuterPriorRecoveryConfig | None,
) -> None:
    if protocol.get("schema_version") != (
        "midogpp_prior_recovery_outer_protocol_v1"
    ):
        raise ProtocolError("Unexpected outer prior-recovery protocol schema.")
    if (
        protocol.get("method") != PRIOR_RECOVERY_METHOD
        or protocol.get("claim_scope") != "cvae_preservation_only"
    ):
        raise ProtocolError("Outer method/claim scope mismatch.")
    semantic_contract = {
        "target_eval_labels_used_for_scoring_only": True,
        "target_eval_labels_used_for_selection": False,
        "support_labels_used": False,
        "oracle_eligible": False,
        "may_feed_model_recipe": False,
        "may_feed_deployable_selection": False,
        "routing_performed": False,
        "composition_performed": False,
    }
    if any(protocol.get(field) != value for field, value in semantic_contract.items()):
        raise ProtocolError("Outer protocol semantic contract mismatch.")
    contract = protocol.get("recipe_contract")
    if (
        not isinstance(contract, Mapping)
        or stable_hash(contract) != protocol.get("recipe_contract_hash")
    ):
        raise ProtocolError("Outer recipe contract hash mismatch.")
    decision_contract = protocol.get("outer_decision_contract")
    if (
        not isinstance(decision_contract, Mapping)
        or stable_hash(decision_contract)
        != protocol.get("outer_decision_contract_hash")
    ):
        raise ProtocolError("Outer decision contract hash mismatch.")
    _, heldouts = _validate_centers(protocol)
    if (
        expected_config is not None
        and recipe_contract_hash(expected_config)
        != protocol.get("recipe_contract_hash")
    ):
        raise ProtocolError("Outer protocol recipe contract differs from its config.")
    if (
        expected_config is not None
        and outer_decision_contract_hash(expected_config)
        != protocol.get("outer_decision_contract_hash")
    ):
        raise ProtocolError("Outer decision contract differs from its config.")
    if tuple(str(value) for value in contract.get("heldout_centers", ())) != heldouts:
        raise ProtocolError(
            "Outer recipe contract heldout centers differ from the protocol."
        )
    expected_runtime_hash = stable_hash(
        {
            "schema_version": "midogpp_prior_recovery_runtime_protocol_v1",
            "name": protocol["experiment_name"],
            "mode": "outer",
            "recipe_contract_hash": protocol["recipe_contract_hash"],
            "manifest_hash": protocol["manifest_hash"],
            "feature_cache_hash": protocol["feature_cache_hash"],
            "reference_protocol_hash": protocol["real_reference_protocol_hash"],
            "outer_decision_contract_hash": protocol[
                "outer_decision_contract_hash"
            ],
            "selection_bundle_hash": protocol["selection_bundle_hash"],
            "source_inner_protocol_hash": protocol[
                "source_inner_protocol_hash"
            ],
            "frozen_reference_identity_hash": protocol[
                "frozen_reference_identity_hash"
            ],
        }
    )
    if protocol.get("protocol_hash") != expected_runtime_hash:
        raise ProtocolError("Outer runtime protocol hash mismatch.")
    locks = protocol.get("locked_recipes")
    lock_hashes = protocol.get("recipe_lock_hashes")
    if not isinstance(locks, Mapping) or not isinstance(lock_hashes, Mapping):
        raise ProtocolError("Outer protocol lacks bound RecipeLock payloads.")
    if set(str(key) for key in locks) != set(heldouts) or set(
        str(key) for key in lock_hashes
    ) != set(heldouts):
        raise ProtocolError("Outer RecipeLock coverage mismatch.")
    for center in heldouts:
        payload = locks.get(center)
        if not isinstance(payload, Mapping):
            raise ProtocolError(
                f"Malformed outer RecipeLock payload for center {center}."
            )
        embedded = payload.get("recipe_lock_hash")
        without_hash = dict(payload)
        without_hash.pop("recipe_lock_hash", None)
        if (
            embedded != stable_hash(without_hash)
            or lock_hashes.get(center) != embedded
        ):
            raise ProtocolError(f"Outer RecipeLock hash mismatch for center {center}.")
        if (
            payload.get("status") != "VALID"
            or payload.get("primary_arm") not in {"C", "D"}
            or payload.get("recipe_contract_hash")
            != protocol.get("recipe_contract_hash")
            or payload.get("selection_bundle_hash")
            != protocol.get("selection_bundle_hash")
            or payload.get("protocol_hash")
            != protocol.get("source_inner_protocol_hash")
        ):
            raise ProtocolError(
                f"Outer RecipeLock is not factorial-eligible for center {center}."
            )
    spec_hashes = protocol.get("classifier_spec_hashes")
    reference_bacc = protocol.get("real_reference_bacc_by_center")
    reference_eval_hashes = protocol.get("real_reference_eval_row_hashes")
    if (
        not isinstance(spec_hashes, Mapping)
        or not isinstance(reference_bacc, Mapping)
        or not isinstance(reference_eval_hashes, Mapping)
        or not protocol.get("real_reference_bundle_hash")
    ):
        raise ProtocolError("Outer protocol lacks frozen-reference identities.")
    if (
        set(str(key) for key in spec_hashes) != set(heldouts)
        or set(str(key) for key in reference_bacc) != set(heldouts)
        or set(str(key) for key in reference_eval_hashes) != set(heldouts)
    ):
        raise ProtocolError("Outer frozen-reference coverage mismatch.")
    frozen_reference_identity = {
        "real_reference_protocol_hash": protocol[
            "real_reference_protocol_hash"
        ],
        "real_reference_bundle_hash": protocol["real_reference_bundle_hash"],
        "classifier_spec_hashes": spec_hashes,
        "real_reference_bacc_by_center": reference_bacc,
        "real_reference_eval_row_hashes": reference_eval_hashes,
    }
    if stable_hash(frozen_reference_identity) != protocol.get(
        "frozen_reference_identity_hash"
    ):
        raise ProtocolError("Outer frozen-reference identity hash mismatch.")


def _validate_outer_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    protocol: Mapping[str, object],
) -> None:
    eligible = tuple(str(value) for value in protocol["eligible_centers"])
    heldouts = tuple(str(value) for value in protocol["heldout_centers"])
    generation_seeds = tuple(int(value) for value in protocol["generation_seeds"])
    training_seeds = tuple(int(value) for value in protocol["training_seeds"])
    unique: set[tuple[str, ...]] = set()
    for row in rows:
        outer = row["outer_target_center"]
        if row["schema_version"] != OUTER_METRIC_SCHEMA:
            raise ProtocolError("Unexpected outer prior-recovery metric schema.")
        role = row["representation_role"]
        if role not in {"prior", "posterior", "decode"}:
            raise ProtocolError(
                "Outer metric has an undeclared representation role."
            )
        if outer not in heldouts:
            raise ProtocolError("Outer metric row uses an undeclared center.")
        expected_fit = tuple(center for center in eligible if center != outer)
        if tuple(json.loads(row["fit_centers"])) != expected_fit:
            raise ProtocolError("Outer metric fit-center set mismatch.")
        if outer in expected_fit or set(expected_fit).intersection(
            MIDOGPP_EXCLUDED_CENTERS
        ):
            raise ProtocolError("Outer metric leaked target/quarantined center.")
        _assert_common_metric_identity(row, protocol=protocol, outer=True)
        _validate_metric_values(row, protocol=protocol)
        spec_hashes = protocol["classifier_spec_hashes"]
        reference_bacc = protocol["real_reference_bacc_by_center"]
        assert isinstance(spec_hashes, Mapping)
        assert isinstance(reference_bacc, Mapping)
        if row["classifier_spec_hash"] != str(spec_hashes[outer]):
            raise ProtocolError(
                "Outer metric classifier spec differs from the frozen reference."
            )
        if row["real_reference_protocol_hash"] != str(
            protocol["real_reference_protocol_hash"]
        ):
            raise ProtocolError("Outer metric real-reference protocol hash mismatch.")
        if not math.isclose(
            float(row["real_reference_bacc"]),
            float(reference_bacc[outer]),
            abs_tol=1e-12,
        ):
            raise ProtocolError("Outer metric real-reference BACC mismatch.")
        reference_eval_hashes = protocol["real_reference_eval_row_hashes"]
        assert isinstance(reference_eval_hashes, Mapping)
        if row["eval_row_hash"] != str(reference_eval_hashes[outer]):
            raise ProtocolError(
                "Outer metric eval-row identity differs from the frozen reference."
            )
        lock = protocol["locked_recipes"][outer]  # type: ignore[index]
        assert isinstance(lock, Mapping)
        expected_status_scope = (
            "cvae_preservation_only"
            if row["status"] == "ok"
            else "diagnostic_only"
        )
        if row["status"] not in {"ok", "classifier_nonconverged"}:
            raise ProtocolError("Outer metric has an unknown execution status.")
        if (
            row["recipe_lock_hash"] != str(lock["recipe_lock_hash"])
            or row["is_prelocked_primary"]
            != str(lock["primary_arm"] == row["arm"]).lower()
            or row["claim_scope"] != expected_status_scope
            or row["selection_source"] != "source_inner_recipe_lock"
        ):
            raise ProtocolError(
                "Outer metric is not bound to its embedded RecipeLock."
            )
        expected_family = (
            STANDARD_SAMPLER
            if row["arm"] in {"A", "B"}
            else str(lock["sampler_family"])
        )
        expected_objective = (
            ISOTROPIC_OBJECTIVE
            if row["arm"] in {"A", "C"}
            else TASK_FISHER_OBJECTIVE
        )
        if (
            row["sampler_family"] != expected_family
            or row["requested_sampler_family"] != expected_family
        ):
            raise ProtocolError(
                "Outer arm sampler family differs from its locked factorial cell."
            )
        if row["objective_id"] != expected_objective:
            raise ProtocolError(
                "Outer arm objective differs from its factorial cell."
            )
        _validate_generation_evaluation_keys(row, protocol=protocol)
        realized = json.loads(row["realized_sampler_by_class"])
        fallbacks = json.loads(row["fallback_reason_by_class"])
        if (
            set(realized) != {"0", "1"}
            or any(value != expected_family for value in realized.values())
            or any(str(value) for value in fallbacks.values())
            or row["sampler_viable"] != "true"
        ):
            raise ProtocolError(
                "Outer sampler realization changed meaning through fallback."
            )
        if int(row["training_seed"]) not in training_seeds:
            raise ProtocolError("Outer metric uses an undeclared training seed.")
        seed = int(row["generation_seed"])
        if role == "decode" and seed != -1:
            raise ProtocolError("Outer decode rows must use generation_seed=-1.")
        if role in {"prior", "posterior"} and seed not in generation_seeds:
            raise ProtocolError(
                "Outer stochastic row uses an undeclared generation seed."
            )
        key = (
            outer,
            row["arm"],
            row["training_seed"],
            row["generation_seed"],
            role,
        )
        if key in unique:
            raise ProtocolError(f"Duplicate outer metric key: {key}")
        unique.add(key)
    _validate_cross_arm_generation_budgets(rows, source_inner=False)


def _validate_generation_evaluation_keys(
    row: Mapping[str, str],
    *,
    protocol: Mapping[str, object],
) -> None:
    try:
        counts = tuple(
            int(value) for value in json.loads(row["generation_class_counts"])
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ProtocolError(
            "Malformed outer generation class-count identity."
        ) from exc
    if len(counts) != 2 or any(value <= 0 for value in counts):
        raise ProtocolError(
            "Outer generation class-count identity must contain both classes."
        )
    role = row["representation_role"]
    source_state_hash = (
        row["sampler_state_hash"]
        if role == "prior"
        else stable_hash(
            {
                "checkpoint_hash": row["checkpoint_hash"],
                "fit_row_hash": row["fit_row_hash"],
                "role": role,
            }
        )
    )
    generation_hash = GenerationKey(
        source_state_hash=source_state_hash,
        generation_seed=int(row["generation_seed"]),
        class_count_vector=(counts[0], counts[1]),
        representation_role=role,
    ).hash
    evaluation_hash = EvaluationKey(
        generated_artifact_hash=generation_hash,
        frozen_classifier_spec_hash=row["classifier_spec_hash"],
        eval_center=row["outer_target_center"],
        eval_row_hash=row["eval_row_hash"],
        metric_schema_version="chance_corrected_bacc_preservation_v1",
        protocol_hash=str(protocol["protocol_hash"]),
    ).hash
    if row["generation_key_hash"] != generation_hash:
        raise ProtocolError(
            "Outer generation key does not recompute from row provenance."
        )
    if row["evaluation_key_hash"] != evaluation_hash:
        raise ProtocolError(
            "Outer evaluation key does not recompute from row provenance."
        )


def _validate_checkpoint_audits(
    audits: Sequence[Mapping[str, str]],
    *,
    rows: Sequence[Mapping[str, str]],
    protocol: Mapping[str, object],
    checkpoint_index: Mapping[str, object],
) -> None:
    expected = {
        (str(outer), str(seed))
        for outer in protocol["heldout_centers"]
        for seed in protocol["training_seeds"]
    }
    observed = {
        (
            row.get("outer_target_center", ""),
            row.get("training_seed", ""),
        )
        for row in audits
    }
    if observed != expected or len(audits) != len(expected):
        raise ProtocolError("Checkpoint reuse audit coverage mismatch.")
    records = checkpoint_index.get("records")
    if not isinstance(records, list):
        raise ProtocolError("Malformed checkpoint index for reuse audit.")
    checkpoints = {
        str(record["checkpoint_hash"]): record
        for record in records
        if isinstance(record, Mapping)
    }
    for audit in audits:
        if audit.get("status") != "PASS":
            raise ProtocolError("Checkpoint reuse audit is not PASS.")
        if audit.get("checkpoint_a_hash") != audit.get("checkpoint_c_hash"):
            raise ProtocolError("A/C checkpoint identity failed.")
        if audit.get("checkpoint_b_hash") != audit.get("checkpoint_d_hash"):
            raise ProtocolError("B/D checkpoint identity failed.")
        if audit.get("a_c_identity") not in {"True", "true"}:
            raise ProtocolError("A/C checkpoint identity flag is false.")
        if audit.get("b_d_identity") not in {"True", "true"}:
            raise ProtocolError("B/D checkpoint identity flag is false.")
        if audit.get("a_b_initialization_paired") not in {"True", "true"}:
            raise ProtocolError("A/B initialization was not paired.")
        if audit.get("a_b_stochastic_stream_paired") not in {"True", "true"}:
            raise ProtocolError("A/B stochastic streams were not paired.")
        outer = audit["outer_target_center"]
        seed = audit["training_seed"]
        fold_rows = [
            row
            for row in rows
            if row["outer_target_center"] == outer
            and row["training_seed"] == seed
        ]
        checkpoint_by_arm = {
            arm: {
                row["checkpoint_hash"]
                for row in fold_rows
                if row["arm"] == arm
            }
            for arm in ("A", "B", "C", "D")
        }
        if any(len(values) != 1 for values in checkpoint_by_arm.values()):
            raise ProtocolError(
                "Outer factorial arm does not have one checkpoint identity."
            )
        a_hash = next(iter(checkpoint_by_arm["A"]))
        b_hash = next(iter(checkpoint_by_arm["B"]))
        if (
            checkpoint_by_arm["A"] != checkpoint_by_arm["C"]
            or checkpoint_by_arm["B"] != checkpoint_by_arm["D"]
            or audit.get("checkpoint_a_hash") != a_hash
            or audit.get("checkpoint_c_hash") != a_hash
            or audit.get("checkpoint_b_hash") != b_hash
            or audit.get("checkpoint_d_hash") != b_hash
        ):
            raise ProtocolError(
                "Checkpoint reuse audit differs from factorial metric rows."
            )
        a_record = checkpoints.get(a_hash)
        b_record = checkpoints.get(b_hash)
        if not isinstance(a_record, Mapping) or not isinstance(b_record, Mapping):
            raise ProtocolError(
                "Checkpoint reuse audit references unpersisted state."
            )
        if (
            a_record.get("initialization_hash")
            != b_record.get("initialization_hash")
            or a_record.get("stochastic_stream_hash")
            != b_record.get("stochastic_stream_hash")
            or a_record.get("stochastic_pairing_hash")
            != b_record.get("stochastic_pairing_hash")
            or audit.get("stochastic_pairing_hash")
            != a_record.get("stochastic_pairing_hash")
        ):
            raise ProtocolError(
                "A/B paired stochastic provenance does not recompute."
            )


def _coverage_from_protocol(
    protocol: Mapping[str, object],
    rows: Sequence[Mapping[str, str]],
) -> dict[str, object]:
    class _CoverageConfig:
        heldout_centers = tuple(
            str(value) for value in protocol["heldout_centers"]
        )
        training_seeds = tuple(
            int(value) for value in protocol["training_seeds"]
        )
        generation_seeds = tuple(
            int(value) for value in protocol["generation_seeds"]
        )

    return outer_coverage(_CoverageConfig(), rows)  # type: ignore[arg-type]
