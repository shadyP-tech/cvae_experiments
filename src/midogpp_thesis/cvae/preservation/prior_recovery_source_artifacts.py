"""Source-inner artifact writer, validator, and RecipeLock reconstruction."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from ...real_features.classifier_reference.artifacts import stable_hash
from ...real_features.classifier_reference.protocol import ProtocolError
from ...real_features.classifier_reference.schemas.midogpp import (
    MIDOGPP_EXCLUDED_CENTERS,
)
from ..generation_samplers import DIAGONAL_SAMPLER, FULL_SAMPLER, STANDARD_SAMPLER
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
from .prior_recovery_common import (
    PRIOR_RECOVERY_METHOD,
    canonical_rows_hash,
    mean,
    selection_evidence_hash,
)
from .prior_recovery_classifier import (
    SOURCE_INNER_CLASSIFIER_GRID_HASH,
    source_inner_classifier_specs,
)
from .prior_recovery_config import PriorRecoveryConfig, recipe_contract_hash
from .prior_recovery_provenance import validate_provenance_indices
from .prior_recovery_runtime_cache import validate_feature_frame_index
from .prior_recovery_timing import validate_runtime_reports, write_run_state
from .prior_recovery_schema import (
    NESTED_REAL_REFERENCE_SCHEMA,
    SOURCE_CHECKPOINT_AUDIT_COLUMNS,
    SOURCE_INNER_METRIC_COLUMNS,
)
from .runtime import EvaluationKey, GenerationKey
from .source_inner_selection import (
    InnerCenterMetric,
    RecipeLock,
    load_recipe_lock,
    select_recipe_lock,
    write_recipe_lock,
)


def write_source_inner_bundle(
    root: Path,
    *,
    metric_rows: Sequence[Mapping[str, object]],
    nested_reference_rows: Sequence[Mapping[str, object]],
    nested_tuning_rows: Sequence[Mapping[str, object]],
    sampler_rows: Sequence[Mapping[str, object]],
    identity_audit_rows: Sequence[Mapping[str, object]],
    locks: Sequence[RecipeLock],
    protocol_manifest: Mapping[str, object],
    selection_bundle_hash: str,
) -> Path:
    root = prepare_artifact_dirs(root)
    checkpoint_index = _read_json(root / "manifests/checkpoint_index.json")
    checkpoint_audit_rows = _derive_source_checkpoint_audits(
        metric_rows,
        checkpoint_index=checkpoint_index,
    )
    write_csv_rows(
        root / "tables/source_inner_metrics.csv",
        metric_rows,
        SOURCE_INNER_METRIC_COLUMNS,
    )
    write_csv_rows(root / "tables/nested_real_references.csv", nested_reference_rows)
    write_csv_rows(root / "tables/nested_classifier_tuning.csv", nested_tuning_rows)
    write_csv_rows(root / "tables/sampler_realizations.csv", sampler_rows)
    write_csv_rows(root / "tables/identity_overlap_audit.csv", identity_audit_rows)
    write_csv_rows(
        root / "tables/checkpoint_reuse_audit.csv",
        checkpoint_audit_rows,
        SOURCE_CHECKPOINT_AUDIT_COLUMNS,
    )
    write_json(root / "manifests/protocol_manifest.json", dict(protocol_manifest))
    write_json(
        root / "manifests/selection_evidence_manifest.json",
        {
            "schema_version": "midogpp_prior_recovery_selection_evidence_v1",
            "selection_bundle_hash": selection_bundle_hash,
        },
    )
    for lock in locks:
        write_recipe_lock(
            root / f"manifests/recipe_locks/{lock.outer_target_center}.json",
            lock,
        )
    valid = [lock for lock in locks if lock.status == "VALID"]
    conditional = [lock for lock in valid if lock.primary_arm in {"C", "D"}]
    factorial_triggered = (
        len(valid) == len(locks)
        and len(conditional) == len(locks)
        and bool(locks)
    )
    gate = {
        "status": (
            "FACTORIAL_TRIGGERED"
            if factorial_triggered
            else (
                "NEGATIVE_GATE_COMPLETE"
                if len(valid) == len(locks)
                else "INVALID_LOCKS_PRESENT"
            )
        ),
        "n_locks": len(locks),
        "n_valid_locks": len(valid),
        "n_conditional_locks": len(conditional),
        "factorial_triggered": factorial_triggered,
        "invalid_centers": [
            lock.outer_target_center for lock in locks if lock.status != "VALID"
        ],
        "outer_scoring_used": False,
        "selection_bundle_hash": selection_bundle_hash,
    }
    identity_pass = all(row.get("status") == "PASS" for row in identity_audit_rows)
    leakage = {
        "status": "PASS" if len(locks) == len(valid) and identity_pass else "FAIL",
        "outer_target_rows_passed_to_training_or_selection": False,
        "outer_target_labels_used_for_selection": False,
        "target_eval_labels_used_for_selection": False,
        "center_4_excluded": True,
        "identity_overlap_status": "PASS" if identity_pass else "FAIL",
        "routing_performed": False,
        "composition_performed": False,
        "selection_bundle_hash": selection_bundle_hash,
    }
    write_json(root / "reports/gate_decision.json", gate)
    write_json(root / "reports/leakage_report.json", leakage)
    write_run_state(
        root,
        protocol_hash=str(protocol_manifest["protocol_hash"]),
        mode="source_inner",
        status="COMPLETE",
    )
    try:
        validate_source_inner_bundle(root)
    except Exception:
        write_run_state(
            root,
            protocol_hash=str(protocol_manifest["protocol_hash"]),
            mode="source_inner",
            status="FAILED",
        )
        raise
    return root


def validate_source_inner_bundle(
    root: Path,
    *,
    expected_config: PriorRecoveryConfig | None = None,
    require_factorial: bool = False,
) -> dict[str, RecipeLock]:
    root = Path(root)
    required = (
        "tables/source_inner_metrics.csv",
        "tables/nested_real_references.csv",
        "tables/nested_classifier_tuning.csv",
        "tables/sampler_realizations.csv",
        "tables/identity_overlap_audit.csv",
        "tables/checkpoint_reuse_audit.csv",
        "manifests/protocol_manifest.json",
        "manifests/selection_evidence_manifest.json",
        "manifests/checkpoint_index.json",
        "manifests/task_fisher_index.json",
        "manifests/feature_frame_index.json",
        "reports/gate_decision.json",
        "reports/leakage_report.json",
        "tables/runtime_timings.csv",
        "reports/runtime_summary.json",
        "reports/run_state.json",
    )
    _require_files(root, required)
    protocol = _read_json(root / "manifests/protocol_manifest.json")
    _validate_workspace_provenance(root, protocol=protocol, mode="source_inner")
    evidence = _read_json(root / "manifests/selection_evidence_manifest.json")
    gate = _read_json(root / "reports/gate_decision.json")
    leakage = _read_json(root / "reports/leakage_report.json")
    checkpoint_index, fisher_index = validate_provenance_indices(root)
    rows = _read_csv(root / "tables/source_inner_metrics.csv")
    nested_rows = _read_csv(root / "tables/nested_real_references.csv")
    nested_tuning_rows = _read_csv(root / "tables/nested_classifier_tuning.csv")
    sampler_rows = _read_csv(root / "tables/sampler_realizations.csv")
    identity_rows = _read_csv(root / "tables/identity_overlap_audit.csv")
    checkpoint_audits = _read_csv(root / "tables/checkpoint_reuse_audit.csv")
    frame_index = validate_feature_frame_index(
        root,
        expected_frame_hashes={str(row.get("frame_hash", "")) for row in rows},
    )
    _assert_columns(rows, SOURCE_INNER_METRIC_COLUMNS, "source_inner_metrics.csv")
    _assert_columns(
        checkpoint_audits,
        SOURCE_CHECKPOINT_AUDIT_COLUMNS,
        "checkpoint_reuse_audit.csv",
    )
    _validate_source_protocol(protocol, expected_config=expected_config)
    validate_runtime_reports(
        root,
        protocol_hash=str(protocol["protocol_hash"]),
        mode="source_inner",
        checkpoint_index=checkpoint_index,
        frame_index=frame_index,
    )
    heldouts = tuple(str(value) for value in protocol["heldout_centers"])
    eligible = tuple(str(value) for value in protocol["eligible_centers"])
    bundle_hash = selection_evidence_hash(
        metric_rows=rows,
        nested_reference_rows=nested_rows,
        nested_tuning_rows=nested_tuning_rows,
        sampler_rows=sampler_rows,
        identity_rows=identity_rows,
        protocol_manifest=protocol,
        checkpoint_index=checkpoint_index,
        task_fisher_index=fisher_index,
        feature_frame_index=frame_index,
    )
    if evidence.get("selection_bundle_hash") != bundle_hash:
        raise ProtocolError("Source-inner selection evidence bundle hash mismatch.")
    if evidence.get("schema_version") != (
        "midogpp_prior_recovery_selection_evidence_v1"
    ):
        raise ProtocolError("Unexpected source-inner selection evidence schema.")
    if any(row.get("selection_bundle_hash") != bundle_hash for row in rows):
        raise ProtocolError(
            "Source-inner metric row is not bound to the selection evidence bundle."
        )
    _validate_identity_rows(
        identity_rows,
        heldouts=heldouts,
        eligible=eligible,
        source_inner=True,
    )
    _validate_nested_reference_rows(
        nested_rows,
        metric_rows=rows,
        heldouts=heldouts,
        eligible=eligible,
        protocol=protocol,
    )
    _validate_nested_tuning_rows(
        nested_tuning_rows,
        nested_reference_rows=nested_rows,
        heldouts=heldouts,
        eligible=eligible,
        protocol=protocol,
    )
    _validate_sampler_rows(sampler_rows, metric_rows=rows)
    _validate_source_rows(rows, protocol=protocol)
    _validate_metric_provenance(
        rows,
        checkpoint_index=checkpoint_index,
        fisher_index=fisher_index,
        protocol=protocol,
    )
    _validate_source_checkpoint_audits(
        checkpoint_audits,
        rows=rows,
        checkpoint_index=checkpoint_index,
    )
    metrics_by_outer = _reconstruct_inner_metrics(
        rows,
        heldouts=heldouts,
        eligible=eligible,
        protocol=protocol,
    )
    locks: dict[str, RecipeLock] = {}
    for outer in heldouts:
        path = root / f"manifests/recipe_locks/{outer}.json"
        if not path.is_file():
            raise ProtocolError(f"Missing RecipeLock for center {outer}.")
        observed = load_recipe_lock(path)
        outer_rows = [
            row for row in rows if row["outer_target_center"] == outer
        ]
        fit_sets = {
            inner: json.loads(
                next(
                    row["fit_centers"]
                    for row in outer_rows
                    if row["inner_pseudo_target_center"] == inner
                )
            )
            for inner in eligible
            if inner != outer
        }
        preliminary = _reselect_lock(
            metrics_by_outer[outer],
            protocol=protocol,
            outer=outer,
            fit_sets=fit_sets,
            metric_hash=canonical_rows_hash(outer_rows),
            bundle_hash=bundle_hash,
            require_task_factorial=False,
        )
        recomputed = _reselect_lock(
            metrics_by_outer[outer],
            protocol=protocol,
            outer=outer,
            fit_sets=fit_sets,
            metric_hash=canonical_rows_hash(outer_rows),
            bundle_hash=bundle_hash,
            require_task_factorial=preliminary.primary_arm == "C",
        )
        if observed.to_payload() != recomputed.to_payload():
            raise ProtocolError(
                f"RecipeLock does not recompute from source metrics for center {outer}."
            )
        locks[outer] = observed
    expected_gate = _expected_gate(locks, bundle_hash=bundle_hash)
    if gate != expected_gate:
        raise ProtocolError("Source-inner gate report does not match recomputed locks.")
    if leakage.get("selection_bundle_hash") != bundle_hash:
        raise ProtocolError("Source-inner leakage report bundle hash mismatch.")
    expected_leakage_status = (
        "PASS" if all(lock.status == "VALID" for lock in locks.values()) else "FAIL"
    )
    if (
        leakage.get("status") != expected_leakage_status
        or leakage.get("identity_overlap_status") != "PASS"
    ):
        raise ProtocolError(
            "Source-inner leakage report status is inconsistent with evidence."
        )
    for field, expected in {
        "outer_target_rows_passed_to_training_or_selection": False,
        "outer_target_labels_used_for_selection": False,
        "target_eval_labels_used_for_selection": False,
        "center_4_excluded": True,
        "routing_performed": False,
        "composition_performed": False,
    }.items():
        if leakage.get(field) is not expected:
            raise ProtocolError(f"Source-inner leakage field {field} mismatch.")
    if require_factorial and (
        gate.get("factorial_triggered") is not True
        or leakage.get("status") != "PASS"
    ):
        raise ProtocolError(
            "Source-inner artifact is not eligible for the outer factorial run."
        )
    return locks


def _validate_source_protocol(
    protocol: Mapping[str, object],
    *,
    expected_config: PriorRecoveryConfig | None,
) -> None:
    if (
        protocol.get("schema_version")
        != "midogpp_prior_recovery_source_inner_protocol_v1"
    ):
        raise ProtocolError("Unexpected source-inner protocol schema.")
    if (
        protocol.get("method") != PRIOR_RECOVERY_METHOD
        or protocol.get("claim_scope") != "cvae_recipe_lock_only"
    ):
        raise ProtocolError("Source-inner method/claim scope mismatch.")
    contract = protocol.get("recipe_contract")
    if (
        not isinstance(contract, Mapping)
        or stable_hash(contract) != protocol.get("recipe_contract_hash")
    ):
        raise ProtocolError("Source-inner recipe contract hash mismatch.")
    expected_specs = source_inner_classifier_specs(classifier_seed=23)
    if (
        contract.get("classifier_grid_hash") != SOURCE_INNER_CLASSIFIER_GRID_HASH
        or protocol.get("classifier_grid_hash") != SOURCE_INNER_CLASSIFIER_GRID_HASH
        or protocol.get("classifier_grid") != [spec.to_payload() for spec in expected_specs]
    ):
        raise ProtocolError("Source-inner protocol and recipe contract do not share the frozen Stage-20 grid.")
    if (
        expected_config is not None
        and recipe_contract_hash(expected_config)
        != protocol.get("recipe_contract_hash")
    ):
        raise ProtocolError(
            "Source-inner recipe contract differs from the requested outer config."
        )
    eligible, heldouts = _validate_centers(protocol)
    if tuple(str(value) for value in contract.get("heldout_centers", ())) != heldouts:
        raise ProtocolError("Recipe contract heldout centers differ from the protocol.")
    expected_runtime_hash = stable_hash(
        {
            "schema_version": "midogpp_prior_recovery_runtime_protocol_v1",
            "name": protocol["experiment_name"],
            "mode": "source_inner",
            "recipe_contract_hash": protocol["recipe_contract_hash"],
            "manifest_hash": protocol["manifest_hash"],
            "feature_cache_hash": protocol["feature_cache_hash"],
            "reference_protocol_hash": "none",
        }
    )
    if protocol.get("protocol_hash") != expected_runtime_hash:
        raise ProtocolError("Source-inner runtime protocol hash mismatch.")
    if set(eligible).intersection(MIDOGPP_EXCLUDED_CENTERS):
        raise ProtocolError("Quarantined center appears in source-inner protocol.")
    semantic_contract = {
        "source_inner_labels_used_for_selection": True,
        "outer_target_rows_passed_to_training_or_selection": False,
        "target_eval_labels_used_for_selection": False,
        "target_eval_labels_used_for_scoring_only": False,
        "support_labels_used": False,
        "oracle_eligible": False,
        "may_feed_model_recipe": True,
        "may_feed_deployable_selection": False,
        "routing_performed": False,
        "composition_performed": False,
        "query_object": "none",
    }
    if any(protocol.get(field) != value for field, value in semantic_contract.items()):
        raise ProtocolError("Source-inner protocol semantic contract mismatch.")


def _validate_source_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    protocol: Mapping[str, object],
) -> None:
    eligible = tuple(str(value) for value in protocol["eligible_centers"])
    heldouts = tuple(str(value) for value in protocol["heldout_centers"])
    generation_seeds = tuple(
        int(value)
        for value in protocol["recipe_contract"]["generation_seeds"]  # type: ignore[index]
    )
    unique: set[tuple[str, ...]] = set()
    for row in rows:
        if row["schema_version"] != (
            "midogpp_prior_recovery_source_inner_metric_v1"
        ):
            raise ProtocolError("Unexpected source-inner metric schema.")
        outer = row["outer_target_center"]
        inner = row["inner_pseudo_target_center"]
        if outer not in heldouts or inner not in eligible or inner == outer:
            raise ProtocolError("Source-inner row has invalid outer/inner center.")
        expected_fit = tuple(
            center for center in eligible if center not in {outer, inner}
        )
        if tuple(json.loads(row["fit_centers"])) != expected_fit:
            raise ProtocolError("Source-inner metric fit-center set mismatch.")
        if set(expected_fit).intersection(
            {outer, inner, *MIDOGPP_EXCLUDED_CENTERS}
        ):
            raise ProtocolError("Source-inner metric leaked heldout/quarantined center.")
        _assert_common_metric_identity(row, protocol=protocol, outer=False)
        _validate_metric_values(row, protocol=protocol)
        if (
            row["selection_source"] != "fully_nested_source_inner"
            or row["source_inner_labels_used_for_selection"] != "true"
        ):
            raise ProtocolError("Source-inner metric selection identity mismatch.")
        if row["arm"] not in {"A", "B", "C", "D"}:
            raise ProtocolError("Source-inner metric has an unknown factorial arm.")
        expected_objective = (
            ISOTROPIC_OBJECTIVE
            if row["arm"] in {"A", "C"}
            else TASK_FISHER_OBJECTIVE
        )
        if row["objective_id"] != expected_objective:
            raise ProtocolError(
                "Source-inner arm objective differs from its factorial cell."
            )
        if row["sampler_family"] != row["requested_sampler_family"]:
            raise ProtocolError("Source-inner requested sampler identity mismatch.")
        if (
            row["arm"] in {"A", "B"}
            and row["sampler_family"] != STANDARD_SAMPLER
        ):
            raise ProtocolError(
                "Source-inner standard-prior arm uses a conditional sampler."
            )
        key = (
            outer,
            inner,
            row["arm"],
            row["sampler_family"],
            row["representation_role"],
            row["training_seed"],
            row["generation_seed"],
        )
        if key in unique:
            raise ProtocolError(f"Duplicate source-inner metric key: {key}")
        unique.add(key)
        role = row["representation_role"]
        if role not in {"prior", "posterior", "decode"}:
            raise ProtocolError(
                "Source-inner metric has an undeclared representation role."
            )
        seed = int(row["generation_seed"])
        if role == "decode" and seed != -1:
            raise ProtocolError(
                "Source-inner decode rows must use generation_seed=-1."
            )
        if role in {"prior", "posterior"} and seed not in generation_seeds:
            raise ProtocolError(
                "Source-inner stochastic row uses an undeclared generation seed."
            )
        _validate_generation_evaluation_keys(row, protocol=protocol)
    _validate_cross_arm_generation_budgets(rows, source_inner=True)


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
            "Malformed source-inner generation class-count identity."
        ) from exc
    if len(counts) != 2 or any(value <= 0 for value in counts):
        raise ProtocolError(
            "Source-inner generation class-count identity must contain both classes."
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
        eval_center=row["inner_pseudo_target_center"],
        eval_row_hash=row["eval_row_hash"],
        metric_schema_version="chance_corrected_bacc_preservation_v1",
        protocol_hash=str(protocol["protocol_hash"]),
    ).hash
    if row["generation_key_hash"] != generation_hash:
        raise ProtocolError(
            "Source-inner generation key does not recompute from row provenance."
        )
    if row["evaluation_key_hash"] != evaluation_hash:
        raise ProtocolError(
            "Source-inner evaluation key does not recompute from row provenance."
        )


def _validate_source_checkpoint_audits(
    audits: Sequence[Mapping[str, str]],
    *,
    rows: Sequence[Mapping[str, str]],
    checkpoint_index: Mapping[str, object],
) -> None:
    expected = _derive_source_checkpoint_audits(
        rows,
        checkpoint_index=checkpoint_index,
    )
    if canonical_rows_hash(audits) != canonical_rows_hash(expected):
        raise ProtocolError(
            "Source-inner checkpoint reuse audit does not recompute from persisted provenance."
        )


def _derive_source_checkpoint_audits(
    rows: Sequence[Mapping[str, object]],
    *,
    checkpoint_index: Mapping[str, object],
) -> list[dict[str, object]]:
    records = checkpoint_index.get("records")
    if not isinstance(records, list) or not all(
        isinstance(record, Mapping) for record in records
    ):
        raise ProtocolError("Malformed checkpoint index for source-inner reuse audit.")
    checkpoints = {
        str(record["checkpoint_hash"]): record
        for record in records
        if isinstance(record, Mapping)
    }
    groups: dict[tuple[str, str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        key = (
            str(row.get("outer_target_center", "")),
            str(row.get("inner_pseudo_target_center", "")),
            str(row.get("training_seed", "")),
        )
        groups.setdefault(key, []).append(row)
    output: list[dict[str, object]] = []
    for (outer, inner, seed), fold_rows in sorted(groups.items()):
        checkpoint_by_arm = {
            arm: {
                str(row.get("checkpoint_hash", ""))
                for row in fold_rows
                if row.get("arm") == arm
            }
            for arm in ("A", "B", "C", "D")
        }
        if (
            len(checkpoint_by_arm["A"]) != 1
            or len(checkpoint_by_arm["C"]) != 1
            or checkpoint_by_arm["A"] != checkpoint_by_arm["C"]
        ):
            raise ProtocolError("Source-inner A/C checkpoint reuse identity failed.")
        task_present = bool(checkpoint_by_arm["B"] or checkpoint_by_arm["D"])
        if task_present and (
            len(checkpoint_by_arm["B"]) != 1
            or len(checkpoint_by_arm["D"]) != 1
            or checkpoint_by_arm["B"] != checkpoint_by_arm["D"]
        ):
            raise ProtocolError("Source-inner B/D checkpoint reuse identity failed.")
        a_hash = next(iter(checkpoint_by_arm["A"]))
        a_record = checkpoints.get(a_hash)
        if not isinstance(a_record, Mapping):
            raise ProtocolError("Source-inner A/C audit references unpersisted state.")
        b_hash = next(iter(checkpoint_by_arm["B"])) if task_present else ""
        b_record = checkpoints.get(b_hash) if task_present else None
        if task_present and not isinstance(b_record, Mapping):
            raise ProtocolError("Source-inner B/D audit references unpersisted state.")
        if task_present:
            assert isinstance(b_record, Mapping)
            if a_record.get("initialization_hash") != b_record.get("initialization_hash"):
                raise ProtocolError("Source-inner A/B initialization pairing failed.")
            if a_record.get("stochastic_stream_hash") != b_record.get("stochastic_stream_hash"):
                raise ProtocolError("Source-inner A/B stochastic-stream pairing failed.")
            if a_record.get("stochastic_pairing_hash") != b_record.get("stochastic_pairing_hash"):
                raise ProtocolError("Source-inner A/B stochastic-pairing identity failed.")
        task_fisher_hash = (
            _single_fold_value(fold_rows, "task_fisher_state_hash", arms={"B", "D"})
            if task_present
            else "none"
        )
        output.append(
            {
                "outer_target_center": outer,
                "inner_pseudo_target_center": inner,
                "training_seed": seed,
                "task_factorial_present": task_present,
                "checkpoint_a_hash": a_hash,
                "checkpoint_c_hash": a_hash,
                "checkpoint_b_hash": b_hash,
                "checkpoint_d_hash": b_hash,
                "a_c_identity": True,
                "b_d_identity": True if task_present else "NOT_APPLICABLE",
                "a_b_initialization_paired": True if task_present else "NOT_APPLICABLE",
                "a_b_stochastic_stream_paired": True if task_present else "NOT_APPLICABLE",
                "stochastic_pairing_hash": str(a_record.get("stochastic_pairing_hash", "")),
                "task_fisher_state_hash": task_fisher_hash,
                "classifier_spec_hash": _single_fold_value(fold_rows, "classifier_spec_hash"),
                "frame_hash": _single_fold_value(fold_rows, "frame_hash"),
                "fit_row_hash": _single_fold_value(fold_rows, "fit_row_hash"),
                "eval_row_hash": _single_fold_value(fold_rows, "eval_row_hash"),
                "status": "PASS",
            }
        )
    return output


def _single_fold_value(
    rows: Sequence[Mapping[str, object]],
    field: str,
    *,
    arms: set[str] | None = None,
) -> str:
    values = {
        str(row.get(field, ""))
        for row in rows
        if arms is None or row.get("arm") in arms
    }
    if len(values) != 1 or not next(iter(values), ""):
        raise ProtocolError(f"Source-inner checkpoint audit has inconsistent {field}.")
    return next(iter(values))


def _validate_nested_tuning_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    nested_reference_rows: Sequence[Mapping[str, str]],
    heldouts: Sequence[str],
    eligible: Sequence[str],
    protocol: Mapping[str, object],
) -> None:
    specs = source_inner_classifier_specs(classifier_seed=23)
    specs_by_hash = {spec.config_hash: spec for spec in specs}
    nested_by_fold = {
        (row["outer_target_center"], row["inner_pseudo_target_center"]): row
        for row in nested_reference_rows
    }
    expected_folds = {
        (outer, inner)
        for outer in heldouts
        for inner in eligible
        if inner != outer
    }
    grouped: dict[tuple[str, str], list[Mapping[str, str]]] = {}
    for row in rows:
        fold = (row.get("outer_target_center", ""), row.get("inner_pseudo_target_center", ""))
        grouped.setdefault(fold, []).append(row)
    if set(grouped) != expected_folds or any(len(group) != len(specs) for group in grouped.values()):
        raise ProtocolError("Nested classifier tuning coverage mismatch.")
    for (outer, inner), group in grouped.items():
        validation_centers = tuple(center for center in eligible if center not in {outer, inner})
        by_hash: dict[str, tuple[Mapping[str, str], float]] = {}
        for row in group:
            try:
                spec_payload = json.loads(row["classifier_spec"])
                vector = json.loads(row["center_bacc_vector"])
                convergence = json.loads(row["convergence_by_center"])
            except (KeyError, json.JSONDecodeError) as exc:
                raise ProtocolError("Malformed nested classifier tuning row.") from exc
            config_hash = row.get("classifier_config_hash", "")
            expected_spec = specs_by_hash.get(config_hash)
            if (
                row.get("schema_version") != "midogpp_eligible_predict_spec_selection_v2"
                or expected_spec is None
                or spec_payload != expected_spec.to_payload()
                or row.get("classifier_grid_hash") != SOURCE_INNER_CLASSIFIER_GRID_HASH
                or tuple(json.loads(row["deeper_validation_centers"])) != validation_centers
                or tuple(json.loads(row["excluded_centers"])) != (outer, inner)
                or set(vector) != set(validation_centers)
                or set(convergence) != set(validation_centers)
                or not all(bool(value) for value in convergence.values())
            ):
                raise ProtocolError("Nested classifier tuning identity or fold coverage mismatch.")
            values = [float(vector[center]) for center in validation_centers]
            aggregate = sum(values) / float(len(values))
            if (
                not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values)
                or not math.isclose(float(row["aggregate_bacc"]), aggregate, abs_tol=1e-12)
                or any(
                    row.get(field) != "false"
                    for field in (
                        "selection_used_target_labels",
                        "fit_used_outer_target_center",
                        "fit_used_inner_pseudo_target_center",
                    )
                )
                or row.get("selection_source") != "nested_source_inner_predict"
            ):
                raise ProtocolError("Nested classifier tuning scores or leakage fields are invalid.")
            by_hash[config_hash] = (row, aggregate)
        if set(by_hash) != set(specs_by_hash):
            raise ProtocolError("Nested classifier tuning grid membership mismatch.")
        best_score = max(aggregate for _, aggregate in by_hash.values())
        winners = [
            specs_by_hash[config_hash]
            for config_hash, (_, aggregate) in by_hash.items()
            if aggregate == best_score
        ]
        selected_spec = min(winners, key=lambda spec: spec.tie_break_key())
        selected_rows = [row for row, _ in by_hash.values() if row.get("selected") == "true"]
        if (
            len(selected_rows) != 1
            or selected_rows[0].get("classifier_config_hash") != selected_spec.config_hash
            or nested_by_fold[(outer, inner)].get("selected_classifier_spec_hash") != selected_spec.config_hash
        ):
            raise ProtocolError("Nested classifier selection does not recompute from its tuning rows.")


def _validate_nested_reference_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    metric_rows: Sequence[Mapping[str, str]],
    heldouts: Sequence[str],
    eligible: Sequence[str],
    protocol: Mapping[str, object],
) -> None:
    expected_specs = source_inner_classifier_specs(classifier_seed=23)
    expected_payloads = {stable_hash(spec.to_payload()) for spec in expected_specs}
    if (
        protocol.get("classifier_grid_hash") != SOURCE_INNER_CLASSIFIER_GRID_HASH
        or protocol.get("classifier_grid") != [spec.to_payload() for spec in expected_specs]
    ):
        raise ProtocolError("Source-inner protocol classifier grid is not the frozen Stage-20 grid.")
    expected = {
        (outer, inner)
        for outer in heldouts
        for inner in eligible
        if inner != outer
    }
    observed = {
        (
            row.get("outer_target_center", ""),
            row.get("inner_pseudo_target_center", ""),
        )
        for row in rows
    }
    if observed != expected or len(rows) != len(expected):
        raise ProtocolError("Nested real-reference coverage mismatch.")
    for row in rows:
        outer = row["outer_target_center"]
        inner = row["inner_pseudo_target_center"]
        fit_centers = tuple(
            center for center in eligible if center not in {outer, inner}
        )
        if row.get("schema_version") != NESTED_REAL_REFERENCE_SCHEMA:
            raise ProtocolError("Unexpected nested real-reference schema.")
        if tuple(json.loads(row["fit_centers"])) != fit_centers:
            raise ProtocolError("Nested real-reference fit-center set mismatch.")
        if set(json.loads(row["deeper_validation_centers"])) != set(fit_centers):
            raise ProtocolError(
                "Nested classifier selection center coverage mismatch."
            )
        if row["classifier_grid_hash"] != protocol["classifier_grid_hash"]:
            raise ProtocolError("Nested classifier grid identity mismatch.")
        try:
            spec = json.loads(row["selected_classifier_spec"])
        except json.JSONDecodeError as exc:
            raise ProtocolError("Malformed nested classifier specification.") from exc
        if (
            not isinstance(spec, Mapping)
            or stable_hash(spec) != row["selected_classifier_spec_hash"]
        ):
            raise ProtocolError("Nested classifier specification hash mismatch.")
        if stable_hash(spec) not in expected_payloads:
            raise ProtocolError("Nested classifier specification is not a member of the frozen Stage-20 grid.")
        expected_reference_hash = stable_hash(
            {
                "outer": outer,
                "inner": inner,
                "fit_row_hash": row["fit_row_hash"],
                "eval_row_hash": row["eval_row_hash"],
                "classifier_spec_hash": row["selected_classifier_spec_hash"],
                "grid_hash": row["classifier_grid_hash"],
            }
        )
        if row["real_reference_protocol_hash"] != expected_reference_hash:
            raise ProtocolError("Nested real-reference protocol hash mismatch.")
        if (
            row.get("status") != "ok"
            or row.get("converged") not in {"True", "true"}
            or row.get("target_eval_labels_used_for_scoring_only")
            not in {"False", "false"}
            or row.get("selection_used_outer_or_inner_labels")
            not in {"False", "false"}
        ):
            raise ProtocolError(
                "Nested real-reference row violates the selection protocol."
            )
        fold_metrics = [
            metric
            for metric in metric_rows
            if metric["outer_target_center"] == outer
            and metric["inner_pseudo_target_center"] == inner
        ]
        if not fold_metrics:
            raise ProtocolError(
                "Nested real-reference row has no preservation metrics."
            )
        for metric in fold_metrics:
            if (
                metric["classifier_spec_hash"]
                != row["selected_classifier_spec_hash"]
                or metric["real_reference_protocol_hash"]
                != expected_reference_hash
                or metric["fit_row_hash"] != row["fit_row_hash"]
                or metric["eval_row_hash"] != row["eval_row_hash"]
                or not math.isclose(
                    float(metric["real_reference_bacc"]),
                    float(row["bacc"]),
                    abs_tol=1e-12,
                )
            ):
                raise ProtocolError(
                    "Nested real-reference and metric lineage differ."
                )


def _reconstruct_inner_metrics(
    rows: Sequence[Mapping[str, str]],
    *,
    heldouts: Sequence[str],
    eligible: Sequence[str],
    protocol: Mapping[str, object],
) -> dict[str, list[InnerCenterMetric]]:
    output: dict[str, list[InnerCenterMetric]] = {outer: [] for outer in heldouts}
    generation_seeds = tuple(
        int(value)
        for value in protocol["recipe_contract"]["generation_seeds"]  # type: ignore[index]
    )
    for outer in heldouts:
        for inner in eligible:
            if inner == outer:
                continue
            fold = [
                row
                for row in rows
                if row["outer_target_center"] == outer
                and row["inner_pseudo_target_center"] == inner
            ]
            a_decode = _single(
                fold,
                arm="A",
                family=STANDARD_SAMPLER,
                role="decode",
                seed=-1,
                required=True,
            )
            a_posterior = _seed_rows(
                fold,
                arm="A",
                family=STANDARD_SAMPLER,
                role="posterior",
                seeds=generation_seeds,
                required=True,
            )
            for arm, family in (
                ("A", STANDARD_SAMPLER),
                ("C", DIAGONAL_SAMPLER),
                ("C", FULL_SAMPLER),
            ):
                prior = _seed_rows(
                    fold,
                    arm=arm,
                    family=family,
                    role="prior",
                    seeds=generation_seeds,
                    required=True,
                )
                output[outer].append(
                    _inner_metric(
                        outer,
                        inner,
                        arm,
                        family,
                        prior,
                        a_decode,
                        a_posterior,
                    )
                )
            task_rows = [row for row in fold if row["arm"] in {"B", "D"}]
            if task_rows:
                b_decode = _single(
                    fold,
                    arm="B",
                    family=STANDARD_SAMPLER,
                    role="decode",
                    seed=-1,
                    required=True,
                )
                b_posterior = _seed_rows(
                    fold,
                    arm="B",
                    family=STANDARD_SAMPLER,
                    role="posterior",
                    seeds=generation_seeds,
                    required=True,
                )
                b_prior = _seed_rows(
                    fold,
                    arm="B",
                    family=STANDARD_SAMPLER,
                    role="prior",
                    seeds=generation_seeds,
                    required=True,
                )
                output[outer].append(
                    _inner_metric(
                        outer,
                        inner,
                        "B",
                        STANDARD_SAMPLER,
                        b_prior,
                        b_decode,
                        b_posterior,
                    )
                )
                d_families = {
                    row["sampler_family"]
                    for row in task_rows
                    if row["arm"] == "D"
                }
                if len(d_families) != 1:
                    raise ProtocolError(
                        "Source-inner D arm must realize one requested sampler family."
                    )
                d_family = d_families.pop()
                d_prior = _seed_rows(
                    fold,
                    arm="D",
                    family=d_family,
                    role="prior",
                    seeds=generation_seeds,
                    required=True,
                )
                output[outer].append(
                    _inner_metric(
                        outer,
                        inner,
                        "D",
                        d_family,
                        d_prior,
                        b_decode,
                        b_posterior,
                    )
                )
    return output


def _inner_metric(
    outer: str,
    inner: str,
    arm: str,
    family: str,
    prior: Sequence[Mapping[str, str]],
    decode: Mapping[str, str],
    posterior: Sequence[Mapping[str, str]],
) -> InnerCenterMetric:
    first = prior[0]
    return InnerCenterMetric(
        outer_target_center=outer,
        inner_pseudo_target_center=inner,
        arm=arm,
        sampler_family=family,
        objective_id=first["objective_id"],
        prior_ratio=mean(float(row["preservation_ratio"]) for row in prior),
        decode_bacc=float(decode["bacc"]),
        posterior_bacc=mean(float(row["bacc"]) for row in posterior),
        real_reference_bacc=float(first["real_reference_bacc"]),
        valid=all(
            row["status"] == "ok" for row in (*prior, decode, *posterior)
        ),
        task_fisher_valid=all(
            row["task_fisher_valid"] == "true" for row in prior
        ),
        sampler_viable=all(row["sampler_viable"] == "true" for row in prior),
        realized_sampler_by_class=json.loads(first["realized_sampler_by_class"]),
        fallback_reason_by_class=json.loads(first["fallback_reason_by_class"]),
    )


def _reselect_lock(
    metrics: Sequence[InnerCenterMetric],
    *,
    protocol: Mapping[str, object],
    outer: str,
    fit_sets: Mapping[str, object],
    metric_hash: str,
    bundle_hash: str,
    require_task_factorial: bool,
) -> RecipeLock:
    contract = protocol["recipe_contract"]
    assert isinstance(contract, Mapping)
    expected_inner = tuple(
        center for center in protocol["eligible_centers"] if center != outer
    )
    return select_recipe_lock(
        metrics,
        outer_target_center=outer,
        expected_inner_centers=expected_inner,
        generation_seeds=tuple(
            int(value) for value in contract["generation_seeds"]
        ),
        beta_final=float(contract["isotropic_variant"]["beta_final"]),  # type: ignore[index]
        classifier_grid_hash=str(contract["classifier_grid_hash"]),
        protocol_hash=str(protocol["protocol_hash"]),
        fit_center_sets_hash=stable_hash(dict(fit_sets)),
        recipe_contract_hash=str(protocol["recipe_contract_hash"]),
        selection_bundle_hash=bundle_hash,
        source_metric_table_hash=metric_hash,
        gate_min_ratio_improvement=float(contract["gate_min_ratio_improvement"]),
        gate_min_inner_wins=min(
            int(contract["gate_min_inner_wins"]),
            len(expected_inner),
        ),
        sampler_tie_margin=float(contract["sampler_tie_margin"]),
        task_increment_min_ratio=float(contract["task_increment_min_ratio"]),
        safety_max_bacc_regression=float(
            contract["safety_max_bacc_regression"]
        ),
        minimum_real_bacc=float(contract["minimum_real_bacc"]),
        require_task_factorial=require_task_factorial,
    )


def _expected_gate(
    locks: Mapping[str, RecipeLock],
    *,
    bundle_hash: str,
) -> dict[str, object]:
    values = list(locks.values())
    valid = [lock for lock in values if lock.status == "VALID"]
    conditional = [lock for lock in valid if lock.primary_arm in {"C", "D"}]
    triggered = (
        len(valid) == len(values)
        and len(conditional) == len(values)
        and bool(values)
    )
    return {
        "status": (
            "FACTORIAL_TRIGGERED"
            if triggered
            else (
                "NEGATIVE_GATE_COMPLETE"
                if len(valid) == len(values)
                else "INVALID_LOCKS_PRESENT"
            )
        ),
        "n_locks": len(values),
        "n_valid_locks": len(valid),
        "n_conditional_locks": len(conditional),
        "factorial_triggered": triggered,
        "invalid_centers": [
            lock.outer_target_center for lock in values if lock.status != "VALID"
        ],
        "outer_scoring_used": False,
        "selection_bundle_hash": bundle_hash,
    }


def _single(
    rows: Sequence[Mapping[str, str]],
    *,
    arm: str,
    family: str,
    role: str,
    seed: int,
    required: bool,
) -> Mapping[str, str]:
    selected = [
        row
        for row in rows
        if row["arm"] == arm
        and row["sampler_family"] == family
        and row["representation_role"] == role
        and int(row["generation_seed"]) == seed
    ]
    if len(selected) != 1:
        if required:
            raise ProtocolError(f"Expected one {arm}/{family}/{role}/{seed} row.")
        return {}
    return selected[0]


def _seed_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    arm: str,
    family: str,
    role: str,
    seeds: Sequence[int],
    required: bool,
) -> list[Mapping[str, str]]:
    selected = [
        _single(
            rows,
            arm=arm,
            family=family,
            role=role,
            seed=seed,
            required=required,
        )
        for seed in seeds
    ]
    return [row for row in selected if row]
