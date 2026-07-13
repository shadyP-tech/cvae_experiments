"""Shared I/O, workspace provenance, and lineage validation helpers."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from ...real_features.classifier_reference.artifacts import stable_hash
from ...real_features.classifier_reference.protocol import ProtocolError
from ...real_features.classifier_reference.schemas.midogpp import (
    MIDOGPP_ELIGIBLE_CENTERS,
    MIDOGPP_EXCLUDED_CENTERS,
)
from ..objectives import ISOTROPIC_OBJECTIVE
from .prior_recovery_common import PRIOR_RECOVERY_METHOD, safe_ratio
from .prior_recovery_config import (
    OuterPriorRecoveryConfig,
    load_prior_recovery_config,
    outer_decision_contract_hash,
    recipe_contract_hash,
)
from .prior_recovery_schema import SAMPLER_REALIZATION_SCHEMA


def _validate_centers(
    protocol: Mapping[str, object],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    eligible = tuple(str(value) for value in protocol.get("eligible_centers", ()))
    heldouts = tuple(str(value) for value in protocol.get("heldout_centers", ()))
    if not eligible or not heldouts or not set(heldouts).issubset(eligible):
        raise ProtocolError("Protocol center coverage is empty or inconsistent.")
    if not set(eligible).issubset(MIDOGPP_ELIGIBLE_CENTERS) or set(
        eligible
    ).intersection(MIDOGPP_EXCLUDED_CENTERS):
        raise ProtocolError("Protocol contains unknown or quarantined centers.")
    coverage_mode = protocol.get("coverage_mode")
    if coverage_mode == "complete":
        if (
            eligible != MIDOGPP_ELIGIBLE_CENTERS
            or heldouts != MIDOGPP_ELIGIBLE_CENTERS
        ):
            raise ProtocolError(
                "Complete prior-recovery protocol requires exact nine-center coverage."
            )
    elif coverage_mode != "partial_test":
        raise ProtocolError("Unknown prior-recovery coverage mode.")
    return eligible, heldouts


def _validate_metric_values(
    row: Mapping[str, str],
    *,
    protocol: Mapping[str, object],
) -> None:
    try:
        bacc = float(row["bacc"])
        macro_f1 = float(row["macro_f1"])
        real_bacc = float(row["real_reference_bacc"])
        observed_ratio = float(row["preservation_ratio"])
        minimum_real_bacc = float(
            protocol["recipe_contract"]["minimum_real_bacc"]  # type: ignore[index]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("Metric row contains malformed numeric values.") from exc
    if not all(
        math.isfinite(value) and 0.0 <= value <= 1.0
        for value in (bacc, macro_f1, real_bacc)
    ):
        raise ProtocolError("Metric row contains an invalid bounded score.")
    expected_ratio = safe_ratio(
        bacc,
        real_bacc,
        minimum_real_bacc=minimum_real_bacc,
    )
    if not math.isfinite(expected_ratio) or not math.isclose(
        observed_ratio,
        expected_ratio,
        abs_tol=1e-12,
    ):
        raise ProtocolError(
            "Metric preservation ratio does not match its BACC/reference denominator."
        )


def _assert_common_metric_identity(
    row: Mapping[str, str],
    *,
    protocol: Mapping[str, object],
    outer: bool,
) -> None:
    expected = {
        "method": PRIOR_RECOVERY_METHOD,
        "protocol_hash": str(protocol["protocol_hash"]),
        "recipe_contract_hash": str(protocol["recipe_contract_hash"]),
        "selection_bundle_hash": str(
            protocol.get(
                "selection_bundle_hash",
                row.get("selection_bundle_hash"),
            )
        ),
        "claim_role": "cvae_preservation" if outer else "cvae_recipe_selection",
        "row_role": row["representation_role"],
        "leakage_status": "PASS",
        "support_labels_used": "false",
        "oracle_eligible": "false",
        "target_eval_labels_used_for_selection": "false",
        "may_feed_model_recipe": "false" if outer else "true",
        "may_feed_deployable_selection": "false",
        "routing_performed": "false",
        "composition_performed": "false",
        "query_object": "none",
    }
    if outer:
        expected["target_eval_labels_used_for_scoring_only"] = "true"
    else:
        expected["target_eval_labels_used_for_scoring_only"] = "false"
    for field, value in expected.items():
        if row.get(field) != value:
            raise ProtocolError(f"Metric identity field {field} mismatch.")
    for field in (
        "classifier_spec_hash",
        "frame_hash",
        "checkpoint_hash",
        "training_key_hash",
        "variant_hash",
        "stochastic_pairing_hash",
        "sampler_state_hash",
        "fit_row_hash",
        "eval_row_hash",
        "generation_key_hash",
        "evaluation_key_hash",
    ):
        if not row.get(field):
            raise ProtocolError(f"Metric row lacks provenance field {field}.")


def _validate_cross_arm_generation_budgets(
    rows: Sequence[Mapping[str, str]],
    *,
    source_inner: bool,
) -> None:
    groups: dict[tuple[str, ...], set[tuple[int, int]]] = {}
    for row in rows:
        try:
            counts = tuple(
                int(value) for value in json.loads(row["generation_class_counts"])
            )
        except (KeyError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ProtocolError("Malformed generation class-count identity.") from exc
        if len(counts) != 2:
            raise ProtocolError("Generation class-count identity must have two classes.")
        key = (
            row["outer_target_center"],
            *(
                (row["inner_pseudo_target_center"],)
                if source_inner
                else ()
            ),
            row["training_seed"],
            row["generation_seed"],
            row["representation_role"],
        )
        groups.setdefault(key, set()).add((counts[0], counts[1]))
    if any(len(values) != 1 for values in groups.values()):
        raise ProtocolError(
            "Factorial arms use unequal generation class-count budgets."
        )


def _validate_metric_provenance(
    rows: Sequence[Mapping[str, str]],
    *,
    checkpoint_index: Mapping[str, object],
    fisher_index: Mapping[str, object],
    protocol: Mapping[str, object],
) -> None:
    checkpoint_records = checkpoint_index.get("records")
    fisher_records = fisher_index.get("records")
    if not isinstance(checkpoint_records, list) or not isinstance(
        fisher_records,
        list,
    ):
        raise ProtocolError("Malformed provenance indices.")
    checkpoints = {
        str(record["checkpoint_hash"]): record
        for record in checkpoint_records
        if isinstance(record, Mapping)
    }
    fishers = {
        str(record["task_fisher_state_hash"]): record
        for record in fisher_records
        if isinstance(record, Mapping)
    }
    referenced_checkpoints: set[str] = set()
    for row in rows:
        checkpoint_id = row["checkpoint_hash"]
        record = checkpoints.get(checkpoint_id)
        if not isinstance(record, Mapping):
            raise ProtocolError("Metric row references an unpersisted checkpoint.")
        referenced_checkpoints.add(checkpoint_id)
        training_key = record.get("training_key")
        if not isinstance(training_key, Mapping):
            raise ProtocolError("Checkpoint index lacks its full training key.")
        expected = {
            "training_key_hash": str(record["training_key_hash"]),
            "variant_hash": str(record["variant_hash"]),
            "objective_id": str(record["objective_id"]),
            "task_fisher_state_hash": str(record["task_fisher_state_hash"]),
            "classifier_spec_hash": str(record["classifier_spec_hash"]),
            "stochastic_pairing_hash": str(record["stochastic_pairing_hash"]),
        }
        for field, value in expected.items():
            if row.get(field) != value:
                raise ProtocolError(
                    f"Metric/checkpoint provenance field {field} mismatch."
                )
        key_expected = {
            "fit_row_hash": row["fit_row_hash"],
            "objective_id": row["objective_id"],
            "training_seed": int(row["training_seed"]),
            "frame_hash": row["frame_hash"],
            "dataset_contract_hash": protocol["manifest_hash"],
            "feature_cache_hash": protocol["feature_cache_hash"],
            "protocol_hash": protocol["protocol_hash"],
            "variant_hash": row["variant_hash"],
            "stochastic_pairing_hash": row["stochastic_pairing_hash"],
            "objective_context_hash": row["task_fisher_state_hash"],
        }
        for field, value in key_expected.items():
            if training_key.get(field) != value:
                raise ProtocolError(f"Metric/training-key field {field} mismatch.")
        if tuple(str(value) for value in training_key.get("fit_centers", ())) != (
            tuple(str(value) for value in json.loads(row["fit_centers"]))
        ):
            raise ProtocolError("Metric/training-key fit-center set mismatch.")
        fisher_id = row["task_fisher_state_hash"]
        if row["objective_id"] == ISOTROPIC_OBJECTIVE:
            if fisher_id != "none":
                raise ProtocolError(
                    "Isotropic checkpoint unexpectedly references Task-Fisher state."
                )
        else:
            fisher_record = fishers.get(fisher_id)
            if not isinstance(fisher_record, Mapping):
                raise ProtocolError(
                    "Task-Fisher metric row references an unpersisted state."
                )
            if (
                fisher_record.get("valid") is not True
                or fisher_record.get("probe_config_hash")
                != row["classifier_spec_hash"]
                or (
                    "task_fisher_valid" in row
                    and row.get("task_fisher_valid") != "true"
                )
            ):
                raise ProtocolError(
                    "Task-Fisher state is invalid or bound to another classifier."
                )
    if referenced_checkpoints != set(checkpoints):
        raise ProtocolError("Checkpoint index contains unreferenced training state.")


def _validate_sampler_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    metric_rows: Sequence[Mapping[str, str]],
) -> None:
    groups: dict[tuple[str, ...], list[Mapping[str, str]]] = {}
    for row in rows:
        if row.get("schema_version") != SAMPLER_REALIZATION_SCHEMA:
            raise ProtocolError("Unexpected sampler realization schema.")
        key = (
            row.get("outer_target_center", ""),
            row.get("inner_pseudo_target_center", ""),
            row.get("arm", ""),
            row.get("checkpoint_hash", ""),
            row.get("sampler_state_hash", ""),
        )
        groups.setdefault(key, []).append(row)
    expected_groups = {
        (
            row["outer_target_center"],
            row.get("inner_pseudo_target_center", ""),
            row["arm"],
            row["checkpoint_hash"],
            row["sampler_state_hash"],
        )
        for row in metric_rows
    }
    if set(groups) != expected_groups:
        raise ProtocolError(
            "Sampler realization coverage differs from metric provenance."
        )
    for key, class_rows in groups.items():
        if len(class_rows) != 2 or {
            int(row["class_label"]) for row in class_rows
        } != {0, 1}:
            raise ProtocolError(
                "Sampler realization must contain exactly both classes."
            )
        first = class_rows[0]
        requested = first["requested_family"]
        if any(row["requested_family"] != requested for row in class_rows):
            raise ProtocolError(
                "Sampler classes disagree on the requested family."
            )
        classes = {
            str(int(row["class_label"])): _sampler_class_payload(row)
            for row in class_rows
        }
        expected_hash = stable_hash(
            {
                "requested_family": requested,
                "latent_dim": int(first["latent_dim"]),
                "source_row_hash": first["source_row_hash"],
                "classes": classes,
            }
        )
        if expected_hash != key[-1]:
            raise ProtocolError(
                "Sampler state hash does not match its persisted class states."
            )
        matching_metrics = [
            row
            for row in metric_rows
            if (
                row["outer_target_center"],
                row.get("inner_pseudo_target_center", ""),
                row["arm"],
                row["checkpoint_hash"],
                row["sampler_state_hash"],
            )
            == key
        ]
        realized = {
            label: str(payload["realized_family"])
            for label, payload in classes.items()
        }
        fallback = {
            label: str(payload["fallback_reason"])
            for label, payload in classes.items()
        }
        viable = all(value == requested for value in realized.values())
        for metric in matching_metrics:
            if (
                metric["sampler_family"] != requested
                or json.loads(metric["realized_sampler_by_class"]) != realized
                or json.loads(metric["fallback_reason_by_class"]) != fallback
                or (metric["sampler_viable"] == "true") is not viable
            ):
                raise ProtocolError(
                    "Sampler realization and metric row semantics differ."
                )


def _sampler_class_payload(row: Mapping[str, str]) -> dict[str, object]:
    return {
        "class_label": int(row["class_label"]),
        "requested_family": row["requested_family"],
        "realized_family": row["realized_family"],
        "mean": json.loads(row["mean"]),
        "covariance": json.loads(row["covariance"]),
        "n_rows": int(row["n_rows"]),
        "raw_between_covariance": json.loads(row["raw_between_covariance"]),
        "within_posterior_diagonal": json.loads(
            row["within_posterior_diagonal"]
        ),
        "shrinkage": (
            None if row["shrinkage"] == "" else float(row["shrinkage"])
        ),
        "shrinkage_target": (
            None
            if row["shrinkage_target"] == ""
            else float(row["shrinkage_target"])
        ),
        "jitter": float(row["jitter"]),
        "condition_number": float(row["condition_number"]),
        "eigenvalues": json.loads(row["eigenvalues"]),
        "fallback_reason": row["fallback_reason"],
    }


def _validate_identity_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    heldouts: Sequence[str],
    eligible: Sequence[str],
    source_inner: bool,
) -> None:
    expected = (
        {
            (outer, inner)
            for outer in heldouts
            for inner in eligible
            if inner != outer
        }
        if source_inner
        else {(outer, "") for outer in heldouts}
    )
    observed = {
        (
            row.get("outer_target_center", ""),
            row.get("inner_pseudo_target_center", ""),
        )
        for row in rows
    }
    if observed != expected or len(rows) != len(expected):
        raise ProtocolError("Identity-overlap audit coverage mismatch.")
    if any(
        row.get("status") != "PASS"
        or int(row.get("sample_overlap_count", -1)) != 0
        or int(row.get("case_overlap_count", -1)) != 0
        or row.get("sample_overlap_hash") != _empty_identity_hash()
        or row.get("case_overlap_hash") != _empty_identity_hash()
        or int(row.get("n_fit_samples", 0)) <= 0
        or int(row.get("n_eval_samples", 0)) <= 0
        or int(row.get("n_fit_cases", 0)) <= 0
        or int(row.get("n_eval_cases", 0)) <= 0
        for row in rows
    ):
        raise ProtocolError("Identity-overlap audit is not PASS.")


def _empty_identity_hash() -> str:
    import hashlib

    return hashlib.sha256(b"").hexdigest()


def _validate_workspace_provenance(
    root: Path,
    *,
    protocol: Mapping[str, object],
    mode: str,
) -> None:
    if protocol.get("coverage_mode") != "complete":
        return
    _require_files(root, ("config.resolved.yaml", "provenance/input_artifacts.json"))
    resolved = load_prior_recovery_config(
        root / "config.resolved.yaml",
        expected_mode=mode,
    )
    if recipe_contract_hash(resolved) != protocol.get("recipe_contract_hash"):
        raise ProtocolError(
            "Resolved config recipe contract differs from the artifact protocol."
        )
    if mode == "outer":
        if not isinstance(resolved, OuterPriorRecoveryConfig):
            raise ProtocolError(
                "Resolved outer config has the wrong mode-specific type."
            )
        if outer_decision_contract_hash(resolved) != protocol.get(
            "outer_decision_contract_hash"
        ):
            raise ProtocolError(
                "Resolved config outer decision contract differs from the artifact."
            )
    manifest = _read_json(root / "provenance/input_artifacts.json")
    expected_experiment = (
        "midogpp.cvae.prior_recovery_source_inner.v1"
        if mode == "source_inner"
        else "midogpp.cvae.prior_recovery_outer.v1"
    )
    expected_scope = (
        "cvae_recipe_lock_only"
        if mode == "source_inner"
        else "cvae_preservation_only"
    )
    if (
        manifest.get("schema_version") != "midogpp_input_artifacts_v2"
        or manifest.get("dataset_id") != "midogpp"
        or manifest.get("experiment_id") != expected_experiment
        or manifest.get("stage") != "20_cvae_preservation"
        or manifest.get("claim_scope") != expected_scope
        or manifest.get("selection_used_target_eval_artifacts") is not False
    ):
        raise ProtocolError(
            "Workspace input-provenance manifest identity mismatch."
        )
    inputs = manifest.get("input_artifacts")
    if not isinstance(inputs, list) or not all(
        isinstance(row, Mapping) for row in inputs
    ):
        raise ProtocolError("Malformed workspace input-artifact records.")
    artifact_ids = [str(row.get("artifact_id", "")) for row in inputs]
    if any(not artifact_id for artifact_id in artifact_ids):
        raise ProtocolError("Workspace input artifact record lacks an artifact ID.")
    by_id = {artifact_id: row for artifact_id, row in zip(artifact_ids, inputs)}
    expected_ids = {
        "midogpp_dataset_contract_annotation_patch_v1",
        "midogpp_virchow2_xyxy_feature_cache_seed42",
    }
    if mode == "outer":
        expected_ids.update(
            {
                "midogpp_output_eligible_tuned_real_reference_v2",
                "midogpp_output_cvae_prior_recovery_source_inner_v1",
            }
        )
    if (
        len(inputs) != len(by_id)
        or len(by_id) != len(expected_ids)
        or set(by_id) != expected_ids
    ):
        raise ProtocolError(
            "Workspace input artifact IDs differ from the registered contract."
        )
    for artifact_id, row in zip(artifact_ids, inputs):
        if (
            row.get("exists") is not True
            or row.get("semantic_identities_are_file_hashes") is not False
        ):
            raise ProtocolError(
                f"Input artifact {artifact_id} is missing or mislabels semantic identities."
            )
        integrity = row.get("file_integrity")
        if not isinstance(integrity, Mapping) or str(
            integrity.get("status", "")
        ).startswith("MISSING"):
            raise ProtocolError(
                f"Input artifact {artifact_id} lacks valid file integrity."
            )
        files = integrity.get("files")
        if not isinstance(files, list) or not files:
            raise ProtocolError(
                f"Input artifact {artifact_id} has no hashed provenance files."
            )
        for file_row in files:
            if (
                not isinstance(file_row, Mapping)
                or file_row.get("exists") is not True
            ):
                raise ProtocolError(
                    f"Input artifact {artifact_id} has a missing provenance file."
                )
            computed = file_row.get("computed")
            if not isinstance(computed, Mapping) or not _is_sha256(
                computed.get("sha256")
            ):
                raise ProtocolError(
                    f"Input artifact {artifact_id} lacks a computed SHA-256."
                )
            expected = file_row.get("expected")
            if isinstance(expected, Mapping):
                algorithm = str(expected.get("algorithm", ""))
                if computed.get(algorithm) != expected.get("digest"):
                    raise ProtocolError(
                        f"Input artifact {artifact_id} failed expected hash verification."
                    )
    dataset = by_id["midogpp_dataset_contract_annotation_patch_v1"]
    cache = by_id["midogpp_virchow2_xyxy_feature_cache_seed42"]
    if _recorded_file_hash(dataset, "manifest.csv") != protocol.get(
        "manifest_hash"
    ):
        raise ProtocolError(
            "Dataset manifest provenance hash differs from the runtime protocol."
        )
    if _recorded_file_hash(cache, "embeddings/train.pt") != protocol.get(
        "feature_cache_hash"
    ):
        raise ProtocolError(
            "Feature-cache provenance hash differs from the runtime protocol."
        )
    if mode == "outer":
        reference = by_id["midogpp_output_eligible_tuned_real_reference_v2"]
        source = by_id["midogpp_output_cvae_prior_recovery_source_inner_v1"]
        if (
            _recorded_file_hash(
                reference,
                "manifests/protocol_manifest.json",
            )
            != protocol.get("real_reference_protocol_file_sha256")
            or _recorded_file_hash(
                source,
                "manifests/protocol_manifest.json",
            )
            != protocol.get("source_inner_protocol_file_sha256")
            or _recorded_file_hash(
                source,
                "manifests/selection_evidence_manifest.json",
            )
            != protocol.get("source_selection_evidence_file_sha256")
        ):
            raise ProtocolError(
                "Upstream source/reference file identity differs from input provenance."
            )


def _recorded_file_hash(
    artifact: Mapping[str, object],
    relative_path: str,
) -> str:
    integrity = artifact.get("file_integrity")
    if not isinstance(integrity, Mapping):
        raise ProtocolError("Malformed input artifact file integrity.")
    files = integrity.get("files")
    if not isinstance(files, list):
        raise ProtocolError("Malformed input artifact file list.")
    matches = [
        row
        for row in files
        if isinstance(row, Mapping) and row.get("path") == relative_path
    ]
    if len(matches) != 1:
        raise ProtocolError(
            f"Input provenance lacks unique file identity for {relative_path}."
        )
    computed = matches[0].get("computed")
    if not isinstance(computed, Mapping) or not _is_sha256(
        computed.get("sha256")
    ):
        raise ProtocolError(
            f"Input provenance lacks SHA-256 for {relative_path}."
        )
    return str(computed["sha256"])


def _is_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _require_files(root: Path, required: Sequence[str]) -> None:
    missing = [relative for relative in required if not (root / relative).exists()]
    if missing:
        raise ProtocolError(f"Prior-recovery bundle missing outputs: {missing}")


def _assert_columns(
    rows: Sequence[Mapping[str, object]],
    required: Sequence[str],
    label: str,
) -> None:
    if not rows:
        raise ProtocolError(f"{label} is empty.")
    missing = set(required).difference(rows[0])
    if missing:
        raise ProtocolError(f"{label} missing columns: {sorted(missing)}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ProtocolError(f"Empty CSV: {path}")
        return [dict(row) for row in reader]


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Malformed JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"Expected JSON object: {path}")
    return payload
