"""Mechanical artifact checks shared by the two Stage-20 v2 studies."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Mapping, Sequence

from ....real_features.classifier_reference.artifacts import stable_hash
from ....real_features.classifier_reference.protocol import ProtocolError
from ....real_features.classifier_reference.schemas.midogpp import (
    MIDOGPP_ELIGIBLE_CENTERS,
)
from ...reporting import prepare_artifact_dirs, write_csv_rows, write_json


PROTOCOL_SCHEMA = "midogpp_source_inner_study_protocol_v2"
COVERAGE_SCHEMA = "midogpp_source_inner_study_coverage_v2"
SELECTION_EVIDENCE_SCHEMA = "midogpp_source_inner_study_selection_evidence_v2"
METRIC_SCHEMA = "midogpp_source_inner_study_metric_v2"
PAIRING_AUDIT_SCHEMA = "midogpp_source_inner_study_rng_pairing_v2"
GENERATION_BUDGET_SCHEMA = "midogpp_source_inner_study_generation_budget_v2"
INITIALIZATION_INDEX_SCHEMA = "midogpp_source_inner_study_initialization_index_v2"
RUN_STATE_SCHEMA = "midogpp_source_inner_study_run_state_v2"
RUNTIME_SUMMARY_SCHEMA = "midogpp_source_inner_study_runtime_summary_v2"
PRIOR_SAMPLER_SCHEMA = "midogpp_learned_prior_study_sampler_realization_v2"
FISHER_SAMPLER_SCHEMA = "midogpp_fisher_shrinkage_sampler_realization_v2"


COMMON_STATIC_FILES = (
    "manifests/protocol_manifest.json",
    "manifests/coverage_manifest.json",
    "manifests/selection_evidence_manifest.json",
    "manifests/embedded_v1_preparation_lineage.json",
    "manifests/checkpoint_index.json",
    "manifests/initialization_index.json",
    "manifests/feature_frame_index.json",
    "manifests/generation_budget_manifest.json",
    "reports/study_decision.json",
    "reports/leakage_report.json",
    "reports/runtime_summary.json",
    "reports/run_state.json",
    "tables/source_inner_metrics.csv",
    "tables/paired_deltas.csv",
    "tables/nested_real_references.csv",
    "tables/nested_classifier_tuning.csv",
    "tables/sampler_realizations.csv",
    "tables/checkpoint_reuse_audit.csv",
    "tables/initialization_pairing_audit.csv",
    "tables/generation_budget_audit.csv",
    "tables/rng_pairing_audit.csv",
    "tables/identity_overlap_audit.csv",
    "tables/runtime_timings.csv",
)


class StudyTimingRecorder:
    """Non-selective resumable timing surface with a v2-owned schema."""

    def __init__(self, root: Path, *, protocol_hash: str, mode: str) -> None:
        self.root = Path(root)
        self.protocol_hash = str(protocol_hash)
        self.mode = str(mode)
        self.rows: dict[str, dict[str, object]] = {}
        table = self.root / "tables/runtime_timings.csv"
        summary = self.root / "reports/runtime_summary.json"
        if table.is_file() and summary.is_file():
            payload = read_json(summary)
            if payload.get("protocol_hash") == self.protocol_hash and payload.get(
                "mode"
            ) == self.mode:
                for row in read_csv(table):
                    self.rows[str(row["record_key"])] = dict(row)
        self._write("RUNNING")

    def record(
        self,
        *,
        phase: str,
        elapsed_seconds: float,
        outer_target_center: str,
        inner_pseudo_target_center: str = "",
        objective_id: str = "",
        training_key_hash: str = "",
        cache_status: str = "not_applicable",
    ) -> None:
        elapsed = float(elapsed_seconds)
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise ProtocolError("Study runtime duration must be finite and nonnegative.")
        identity = {
            "mode": self.mode,
            "outer_target_center": str(outer_target_center),
            "inner_pseudo_target_center": str(inner_pseudo_target_center),
            "phase": str(phase),
            "objective_id": str(objective_id),
            "training_key_hash": str(training_key_hash),
        }
        key = stable_hash(identity)
        self.rows[key] = {
            "record_key": key,
            **identity,
            "cache_status": str(cache_status),
            "elapsed_seconds": elapsed,
            "used_for_selection": "false",
            "claim_scope": "diagnostic_only",
        }
        self._write("RUNNING")

    def finalize(self) -> None:
        self._write("COMPLETE")

    def _write(self, status: str) -> None:
        rows = [self.rows[key] for key in sorted(self.rows)]
        write_csv_rows(self.root / "tables/runtime_timings.csv", rows)
        write_json(
            self.root / "reports/runtime_summary.json",
            {
                "schema_version": RUNTIME_SUMMARY_SCHEMA,
                "protocol_hash": self.protocol_hash,
                "mode": self.mode,
                "status": status,
                "n_records": len(rows),
                "total_seconds": sum(float(row["elapsed_seconds"]) for row in rows),
                "used_for_selection": False,
                "claim_scope": "diagnostic_only",
            },
        )


def write_study_run_state(
    root: Path, *, protocol_hash: str, mode: str, status: str
) -> None:
    if status not in {"RUNNING", "COMPLETE", "FAILED"}:
        raise ValueError(f"Unsupported study run state: {status}")
    write_json(
        Path(root) / "reports/run_state.json",
        {
            "schema_version": RUN_STATE_SCHEMA,
            "protocol_hash": str(protocol_hash),
            "mode": str(mode),
            "status": status,
        },
    )


def canonical_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, str]]:
    # CSV writing rectangularizes heterogeneous rows by taking the union of
    # their columns and serializing missing values as empty strings.  Mirror
    # that surface before hashing so evidence computed in memory is identical
    # to evidence recomputed from the persisted CSV bundle.
    fieldnames = sorted({str(key) for row in rows for key in row})
    normalized = [
        {
            field: "" if row.get(field) is None else str(row.get(field))
            for field in fieldnames
        }
        for row in rows
    ]
    return sorted(normalized, key=stable_hash)


def selection_evidence_hash(
    *,
    metric_rows: Sequence[Mapping[str, object]],
    paired_delta_rows: Sequence[Mapping[str, object]],
    nested_reference_rows: Sequence[Mapping[str, object]],
    nested_tuning_rows: Sequence[Mapping[str, object]],
    sampler_rows: Sequence[Mapping[str, object]],
    identity_rows: Sequence[Mapping[str, object]],
    checkpoint_reuse_rows: Sequence[Mapping[str, object]],
    initialization_pairing_rows: Sequence[Mapping[str, object]],
    generation_budget_rows: Sequence[Mapping[str, object]],
    checkpoint_index: Mapping[str, object],
    initialization_index: Mapping[str, object],
    feature_frame_index: Mapping[str, object],
    generation_budget_manifest: Mapping[str, object],
    rng_rows: Sequence[Mapping[str, object]],
    protocol_manifest: Mapping[str, object],
    study_state_index: Mapping[str, object],
) -> str:
    """Hash everything permitted to influence a v2 StudyDecision."""

    return stable_hash(
        {
            "metrics": canonical_rows(metric_rows),
            "paired_deltas": canonical_rows(paired_delta_rows),
            "nested_references": canonical_rows(nested_reference_rows),
            "nested_tuning": canonical_rows(nested_tuning_rows),
            "sampler_realizations": canonical_rows(sampler_rows),
            "identity_audits": canonical_rows(identity_rows),
            "checkpoint_reuse_audits": canonical_rows(checkpoint_reuse_rows),
            "initialization_pairing_audits": canonical_rows(
                initialization_pairing_rows
            ),
            "generation_budget_audits": canonical_rows(generation_budget_rows),
            "checkpoint_index": dict(checkpoint_index),
            "initialization_index": dict(initialization_index),
            "feature_frame_index": dict(feature_frame_index),
            "generation_budget_manifest": dict(generation_budget_manifest),
            "rng_pairing": canonical_rows(rng_rows),
            "protocol_manifest": dict(protocol_manifest),
            "study_state_index": dict(study_state_index),
        }
    )


def study_implementation_lineage(mode: str) -> dict[str, object]:
    """Hash the v2 execution kernel and numerical runtime into protocol identity."""

    import numpy as np
    import sklearn
    import torch

    current = Path(__file__).resolve()
    studies = current.parent
    preservation = studies.parent
    cvae = preservation.parent
    common = (
        studies / "contracts.py",
        studies / "config.py",
        studies / "preparation.py",
        studies / "training.py",
        studies / "checkpoint_store.py",
        studies / "validation_common.py",
        cvae / "models/cvae.py",
        cvae / "feature_frame.py",
        cvae / "generation_samplers.py",
        preservation / "scoring.py",
        preservation / "representations.py",
        preservation / "splits.py",
        cvae.parent / "real_features/classifier_reference/classifiers.py",
        cvae.parent / "real_features/classifier_reference/real_feature_frame.py",
        cvae.parent
        / "real_features/classifier_reference/midogpp_real_feature_classifier.py",
    )
    if mode == "learned_conditional_prior_source_inner_study":
        specific = (
            studies / "prior_runner.py",
            studies / "prior_validation.py",
            studies / "prior_decision.py",
            studies / "prior_artifacts.py",
            cvae / "latent_priors.py",
            cvae / "models/learned_conditional_prior.py",
        )
    elif mode == "task_fisher_shrinkage_source_inner_study":
        specific = (
            studies / "fisher_runner.py",
            studies / "fisher_validation.py",
            studies / "fisher_decision.py",
            studies / "fisher_artifacts.py",
            cvae / "task_fisher.py",
            cvae / "objectives.py",
        )
    else:
        raise ProtocolError(f"Unsupported source-inner study lineage mode: {mode!r}")
    files = (*common, *specific)
    if any(not path.is_file() for path in files):
        raise ProtocolError("Source-inner v2 implementation lineage file is missing.")
    payload: dict[str, object] = {
        "schema_version": "midogpp_source_inner_study_implementation_lineage_v2",
        "mode": mode,
        "component_sha256": {
            path.relative_to(cvae.parent).as_posix(): file_sha256(path)
            for path in files
        },
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": str(np.__version__),
            "sklearn": str(sklearn.__version__),
            "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "cudnn": str(torch.backends.cudnn.version()),
        },
        "determinism_policy": (
            "torch_deterministic_algorithms_cublas_4096_8_tf32_disabled"
        ),
    }
    payload["lineage_hash"] = stable_hash(payload)
    return payload


def expected_bundle_files(config: object, *, state_index_relative: str) -> tuple[str, ...]:
    files = list(COMMON_STATIC_FILES)
    files.append(state_index_relative)
    for seed in getattr(config, "training_seeds"):
        for outer in getattr(config, "heldout_centers"):
            files.append(f"reports/child_decisions/seed{int(seed)}/{outer}.json")
    for outer in getattr(config, "heldout_centers"):
        files.append(f"reports/consensus_decisions/{outer}.json")
    if _complete_production_coverage(config):
        files.extend(("config.resolved.yaml", "provenance/input_artifacts.json"))
    return tuple(files)


def write_common_artifacts(
    root: Path,
    *,
    metric_rows: Sequence[Mapping[str, object]],
    paired_delta_rows: Sequence[Mapping[str, object]],
    nested_reference_rows: Sequence[Mapping[str, object]],
    nested_tuning_rows: Sequence[Mapping[str, object]],
    sampler_rows: Sequence[Mapping[str, object]],
    checkpoint_reuse_rows: Sequence[Mapping[str, object]],
    initialization_pairing_rows: Sequence[Mapping[str, object]],
    generation_budget_rows: Sequence[Mapping[str, object]],
    rng_rows: Sequence[Mapping[str, object]],
    identity_rows: Sequence[Mapping[str, object]],
    protocol_manifest: Mapping[str, object],
    coverage_manifest: Mapping[str, object],
    selection_evidence_manifest: Mapping[str, object],
    embedded_preparation_lineage: Mapping[str, object],
    generation_budget_manifest: Mapping[str, object],
    child_decisions: Mapping[tuple[int, str], Mapping[str, object]],
    consensus_decisions: Mapping[str, Mapping[str, object]],
    study_decision: Mapping[str, object],
    leakage_report: Mapping[str, object],
) -> Path:
    """Write only the mechanically common v2 bundle surfaces."""

    root = prepare_artifact_dirs(root)
    write_csv_rows(root / "tables/source_inner_metrics.csv", metric_rows)
    write_csv_rows(root / "tables/paired_deltas.csv", paired_delta_rows)
    write_csv_rows(root / "tables/nested_real_references.csv", nested_reference_rows)
    write_csv_rows(root / "tables/nested_classifier_tuning.csv", nested_tuning_rows)
    write_csv_rows(root / "tables/sampler_realizations.csv", sampler_rows)
    write_csv_rows(root / "tables/checkpoint_reuse_audit.csv", checkpoint_reuse_rows)
    write_csv_rows(
        root / "tables/initialization_pairing_audit.csv",
        initialization_pairing_rows,
    )
    write_csv_rows(root / "tables/generation_budget_audit.csv", generation_budget_rows)
    write_csv_rows(root / "tables/rng_pairing_audit.csv", rng_rows)
    write_csv_rows(root / "tables/identity_overlap_audit.csv", identity_rows)
    write_json(root / "manifests/protocol_manifest.json", protocol_manifest)
    write_json(root / "manifests/coverage_manifest.json", coverage_manifest)
    write_json(
        root / "manifests/selection_evidence_manifest.json",
        selection_evidence_manifest,
    )
    write_json(
        root / "manifests/embedded_v1_preparation_lineage.json",
        embedded_preparation_lineage,
    )
    write_json(
        root / "manifests/generation_budget_manifest.json",
        generation_budget_manifest,
    )
    for (seed, outer), decision in child_decisions.items():
        write_json(
            root / f"reports/child_decisions/seed{int(seed)}/{outer}.json",
            decision,
        )
    for outer, decision in consensus_decisions.items():
        write_json(root / f"reports/consensus_decisions/{outer}.json", decision)
    write_json(root / "reports/study_decision.json", study_decision)
    write_json(root / "reports/leakage_report.json", leakage_report)
    return root


def require_files(root: Path, relatives: Sequence[str]) -> None:
    missing = [relative for relative in relatives if not (Path(root) / relative).is_file()]
    if missing:
        raise ProtocolError(f"Source-inner study bundle is missing files: {missing}")


def read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Malformed study JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"Expected study JSON object: {path}")
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except OSError as exc:
        raise ProtocolError(f"Cannot read study table: {path}") from exc


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_common_rows(
    config: object,
    *,
    metric_rows: Sequence[Mapping[str, str]],
    identity_rows: Sequence[Mapping[str, str]],
) -> None:
    heldouts = tuple(str(value) for value in getattr(config, "heldout_centers"))
    seeds = tuple(int(value) for value in getattr(config, "training_seeds"))
    generation_seeds = tuple(int(value) for value in getattr(config, "generation_seeds"))
    if not metric_rows:
        raise ProtocolError("Source-inner study metric table is empty.")
    for row in metric_rows:
        if row.get("schema_version") != METRIC_SCHEMA:
            raise ProtocolError("Unexpected source-inner study metric schema.")
        if row.get("outer_target_center") not in heldouts:
            raise ProtocolError("Metric row references an undeclared outer center.")
        outer = str(row.get("outer_target_center", ""))
        inner = str(row.get("inner_pseudo_target_center", ""))
        if inner not in MIDOGPP_ELIGIBLE_CENTERS or inner == outer:
            raise ProtocolError("Metric row references an invalid inner pseudo-target center.")
        if int(row.get("training_seed", -1)) not in seeds:
            raise ProtocolError("Metric row references an undeclared training seed.")
        role = row.get("representation_role")
        generation_seed = int(row.get("generation_seed", -1))
        if role in {"prior", "posterior"} and generation_seed not in generation_seeds:
            raise ProtocolError("Metric row references an undeclared generation seed.")
        if role == "decode" and generation_seed != -1:
            raise ProtocolError("Decode metric row must use generation_seed=-1.")
        if row.get("may_feed_model_recipe") != "false" or row.get(
            "may_feed_deployable_selection"
        ) != "false":
            raise ProtocolError("Study metric row is incorrectly consumable.")
        if (
            row.get("claim_scope") != "cvae_source_inner_study_only"
            or row.get("selection_source") != "fully_nested_source_inner"
            or row.get("target_eval_labels_used_for_selection") != "false"
            or row.get("routing_performed") != "false"
            or row.get("composition_performed") != "false"
        ):
            raise ProtocolError("Study metric row crossed its Stage-20 claim boundary.")
        valid = row.get("valid") == "true"
        if (row.get("status") == "ok") != valid:
            raise ProtocolError("Study metric status/valid flags disagree.")
        try:
            bacc = float(row.get("bacc", "nan"))
            macro_f1 = float(row.get("macro_f1", "nan"))
            real_bacc = float(row.get("real_reference_bacc", "nan"))
            ratio = float(row.get("preservation_ratio", "nan"))
        except (TypeError, ValueError) as exc:
            raise ProtocolError("Study metric contains malformed numeric values.") from exc
        if row.get("status") == "raw_fisher_invalid":
            if (
                any(math.isfinite(value) for value in (bacc, macro_f1, ratio))
                or not math.isfinite(real_bacc)
                or not 0.0 <= real_bacc <= 1.0
            ):
                raise ProtocolError("Invalid Fisher placeholder metric is malformed.")
        else:
            if not all(
                math.isfinite(value) and 0.0 <= value <= 1.0
                for value in (bacc, macro_f1, real_bacc)
            ):
                raise ProtocolError("Study metric contains an invalid bounded score.")
            minimum_real_bacc = float(getattr(config, "minimum_real_bacc"))
            expected_ratio = (
                math.nan
                if real_bacc < minimum_real_bacc
                else (bacc - 0.5) / (real_bacc - 0.5)
            )
            if (
                math.isfinite(expected_ratio)
                and (
                    not math.isfinite(ratio)
                    or not math.isclose(ratio, expected_ratio, abs_tol=1e-12)
                )
            ) or (
                not math.isfinite(expected_ratio) and math.isfinite(ratio)
            ):
                raise ProtocolError(
                    "Study preservation ratio differs from BACC/reference values."
                )
            if valid and not math.isfinite(expected_ratio):
                raise ProtocolError(
                    "Valid study metric is below the real-reference denominator floor."
                )
        try:
            fit_centers = tuple(str(value) for value in json.loads(str(row["fit_centers"])))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ProtocolError("Study metric fit-centers identity is malformed.") from exc
        expected_fit = tuple(
            center
            for center in MIDOGPP_ELIGIBLE_CENTERS
            if center not in {outer, inner}
        )
        if fit_centers != expected_fit:
            raise ProtocolError("Study metric fit centers are not the exact H/I-excluded set.")
    if not identity_rows or any(row.get("status") != "PASS" for row in identity_rows):
        raise ProtocolError("Study identity-overlap audit is missing or not PASS.")


def validate_embedded_preparation_rows(
    config: object,
    *,
    metric_rows: Sequence[Mapping[str, str]],
    nested_reference_rows: Sequence[Mapping[str, str]],
    nested_tuning_rows: Sequence[Mapping[str, str]],
    identity_rows: Sequence[Mapping[str, str]],
) -> None:
    """Revalidate the exact v1 preparation evidence embedded by v2."""

    from ..prior_recovery_artifact_shared import _validate_identity_rows
    from ..prior_recovery_classifier import (
        SOURCE_INNER_CLASSIFIER_GRID_HASH,
        source_inner_classifier_specs,
    )
    from ..prior_recovery_schema import NESTED_REAL_REFERENCE_SCHEMA
    from ..prior_recovery_source_validation import _validate_nested_tuning_rows

    heldouts = tuple(str(value) for value in getattr(config, "heldout_centers"))
    _validate_identity_rows(
        identity_rows,
        heldouts=heldouts,
        eligible=MIDOGPP_ELIGIBLE_CENTERS,
        source_inner=True,
    )
    _validate_nested_tuning_rows(
        nested_tuning_rows,
        nested_reference_rows=nested_reference_rows,
        heldouts=heldouts,
        eligible=MIDOGPP_ELIGIBLE_CENTERS,
        protocol={},
    )
    specs = source_inner_classifier_specs(classifier_seed=23)
    spec_hashes = {spec.config_hash for spec in specs}
    expected_folds = {
        (outer, inner)
        for outer in heldouts
        for inner in MIDOGPP_ELIGIBLE_CENTERS
        if inner != outer
    }
    by_fold: dict[tuple[str, str], Mapping[str, str]] = {}
    for row in nested_reference_rows:
        fold = (
            str(row.get("outer_target_center", "")),
            str(row.get("inner_pseudo_target_center", "")),
        )
        if fold in by_fold:
            raise ProtocolError("Nested real-reference table contains a duplicate fold.")
        outer, inner = fold
        fit_centers = tuple(
            center
            for center in MIDOGPP_ELIGIBLE_CENTERS
            if center not in {outer, inner}
        )
        try:
            recorded_fit = tuple(json.loads(row["fit_centers"]))
            deeper = tuple(json.loads(row["deeper_validation_centers"]))
            spec = json.loads(row["selected_classifier_spec"])
            bacc = float(row["bacc"])
            macro_f1 = float(row["macro_f1"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ProtocolError("Malformed nested real-reference row.") from exc
        expected_reference_hash = stable_hash(
            {
                "outer": outer,
                "inner": inner,
                "fit_row_hash": row.get("fit_row_hash", ""),
                "eval_row_hash": row.get("eval_row_hash", ""),
                "classifier_spec_hash": row.get(
                    "selected_classifier_spec_hash", ""
                ),
                "grid_hash": row.get("classifier_grid_hash", ""),
            }
        )
        if (
            row.get("schema_version") != NESTED_REAL_REFERENCE_SCHEMA
            or recorded_fit != fit_centers
            or deeper != fit_centers
            or row.get("classifier_grid_hash") != SOURCE_INNER_CLASSIFIER_GRID_HASH
            or row.get("selected_classifier_spec_hash") not in spec_hashes
            or stable_hash(spec) != row.get("selected_classifier_spec_hash")
            or row.get("real_reference_protocol_hash") != expected_reference_hash
            or row.get("status") != "ok"
            or row.get("converged") not in {"True", "true"}
            or row.get("target_eval_labels_used_for_scoring_only")
            not in {"False", "false"}
            or row.get("selection_used_outer_or_inner_labels")
            not in {"False", "false"}
            or not all(
                math.isfinite(value) and 0.0 <= value <= 1.0
                for value in (bacc, macro_f1)
            )
            or int(row.get("n_fit", 0)) <= 0
            or int(row.get("n_eval", 0)) <= 0
        ):
            raise ProtocolError("Nested real-reference identity or values are invalid.")
        by_fold[fold] = row
    if set(by_fold) != expected_folds:
        raise ProtocolError("Nested real-reference coverage mismatch.")
    for metric in metric_rows:
        fold = (
            str(metric.get("outer_target_center", "")),
            str(metric.get("inner_pseudo_target_center", "")),
        )
        reference = by_fold.get(fold)
        if not isinstance(reference, Mapping) or (
            metric.get("classifier_spec_hash")
            != reference.get("selected_classifier_spec_hash")
            or metric.get("fit_row_hash") != reference.get("fit_row_hash")
            or metric.get("eval_row_hash") != reference.get("eval_row_hash")
            or not math.isclose(
                float(metric.get("real_reference_bacc", "nan")),
                float(reference.get("bacc", "nan")),
                abs_tol=1e-12,
            )
        ):
            raise ProtocolError(
                "Metric lineage differs from its nested real-reference row."
            )


def validate_metric_grid(
    config: object,
    *,
    metric_rows: Sequence[Mapping[str, str]],
    axis_field: str,
    axis_values: Sequence[object],
    protocol_hash: str,
) -> None:
    """Require the exact decode/prior/posterior grid with no duplicate cells."""

    def axis_key(value: object) -> str:
        return (
            format(float(value), ".12g")
            if axis_field == "alpha"
            else str(value)
        )

    expected: set[tuple[str, str, int, int, str, str]] = set()
    for outer in getattr(config, "heldout_centers"):
        for inner in MIDOGPP_ELIGIBLE_CENTERS:
            if inner == outer:
                continue
            for training_seed in getattr(config, "training_seeds"):
                for axis in axis_values:
                    expected.add(
                        (str(outer), inner, int(training_seed), -1, axis_key(axis), "decode")
                    )
                    for generation_seed in getattr(config, "generation_seeds"):
                        for role in ("prior", "posterior"):
                            expected.add(
                                (
                                    str(outer),
                                    inner,
                                    int(training_seed),
                                    int(generation_seed),
                                    axis_key(axis),
                                    role,
                                )
                            )
    observed: set[tuple[str, str, int, int, str, str]] = set()
    for row in metric_rows:
        if row.get("protocol_hash") != protocol_hash:
            raise ProtocolError("Metric row protocol hash differs from its bundle.")
        try:
            key = (
                str(row["outer_target_center"]),
                str(row["inner_pseudo_target_center"]),
                int(row["training_seed"]),
                int(row["generation_seed"]),
                axis_key(row[axis_field]),
                str(row["representation_role"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("Study metric grid key is malformed.") from exc
        if key in observed:
            raise ProtocolError("Study metric grid contains a duplicate cell.")
        observed.add(key)
    if observed != expected:
        raise ProtocolError(
            "Study metric grid coverage mismatch: "
            f"missing={len(expected - observed)}, extra={len(observed - expected)}."
        )


def validate_initialization_index(
    checkpoint_index: Mapping[str, object],
    initialization_index: Mapping[str, object],
) -> None:
    records = checkpoint_index.get("records")
    observed = initialization_index.get("records")
    if (
        initialization_index.get("schema_version") != INITIALIZATION_INDEX_SCHEMA
        or not isinstance(records, list)
        or not isinstance(observed, list)
    ):
        raise ProtocolError("Malformed study initialization index.")
    fields = (
        "training_key_hash",
        "model_family",
        "shared_initialization_hash",
        "prior_initialization_hash",
        "full_initialization_hash",
        "training_stream_hash",
    )
    expected = [
        {field: row.get(field) for field in fields}
        for row in records
        if isinstance(row, Mapping)
    ]
    if len(expected) != len(records) or observed != expected:
        raise ProtocolError("Initialization index differs from the checkpoint index.")
    if any(not all(str(row.get(field, "")) for field in fields) for row in observed if isinstance(row, Mapping)):
        raise ProtocolError("Initialization index contains an empty identity.")


def validate_generation_budgets(
    config: object,
    *,
    budget_rows: Sequence[Mapping[str, str]],
    budget_manifest: Mapping[str, object],
    metric_rows: Sequence[Mapping[str, str]],
) -> None:
    if (
        budget_manifest.get("schema_version") != GENERATION_BUDGET_SCHEMA
        or budget_manifest.get("policy")
        != getattr(config, "generation_budget_policy")
        or budget_manifest.get("derived_from_y_fit_only") is not True
        or int(budget_manifest.get("n_records", -1)) != len(budget_rows)
        or budget_manifest.get("records_hash")
        != stable_hash(canonical_rows(budget_rows))
    ):
        raise ProtocolError("Generation-budget manifest mismatch.")
    expected = {
        (str(outer), inner)
        for outer in getattr(config, "heldout_centers")
        for inner in MIDOGPP_ELIGIBLE_CENTERS
        if inner != outer
    }
    observed: dict[tuple[str, str], tuple[int, int]] = {}
    for row in budget_rows:
        key = (
            str(row.get("outer_target_center", "")),
            str(row.get("inner_pseudo_target_center", "")),
        )
        try:
            counts = tuple(int(value) for value in json.loads(str(row["class_counts"])))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ProtocolError("Generation-budget class counts are malformed.") from exc
        if (
            key in observed
            or len(counts) != 2
            or min(counts) <= 0
            or row.get("schema_version") != GENERATION_BUDGET_SCHEMA
            or row.get("budget_policy") != getattr(config, "generation_budget_policy")
            or row.get("derived_from_y_fit_only") != "true"
            or row.get("used_inner_labels") != "false"
        ):
            raise ProtocolError("Generation-budget audit row is invalid.")
        observed[key] = (counts[0], counts[1])
    if set(observed) != expected:
        raise ProtocolError("Generation-budget audit has incomplete H/I coverage.")
    for row in metric_rows:
        key = (
            str(row.get("outer_target_center", "")),
            str(row.get("inner_pseudo_target_center", "")),
        )
        try:
            counts = tuple(
                int(value)
                for value in json.loads(str(row["generation_class_counts"]))
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ProtocolError("Metric generation counts are malformed.") from exc
        if counts != observed.get(key):
            raise ProtocolError("Metric generation counts differ from the source budget.")


def validate_rng_rows(
    *,
    metric_rows: Sequence[Mapping[str, str]],
    rng_rows: Sequence[Mapping[str, str]],
    axis_field: str,
) -> None:
    """Bind every realized prior cell to both training-seed-neutral streams."""

    def axis_key(value: object) -> str:
        return format(float(value), ".12g") if axis_field == "alpha" else str(value)

    expected: set[tuple[str, str, int, int, str, str]] = set()
    for row in metric_rows:
        if row.get("representation_role") != "prior" or row.get(
            "training_key_hash"
        ) == "none":
            continue
        base = (
            str(row["outer_target_center"]),
            str(row["inner_pseudo_target_center"]),
            int(row["training_seed"]),
            int(row["generation_seed"]),
            axis_key(row[axis_field]),
        )
        expected.add(base + ("prior_generation",))
        expected.add(base + ("posterior_evaluation",))
    observed: set[tuple[str, str, int, int, str, str]] = set()
    paired: dict[tuple[str, str, int, str], set[str]] = {}
    for row in rng_rows:
        try:
            key = (
                str(row["outer_target_center"]),
                str(row["inner_pseudo_target_center"]),
                int(row["training_seed"]),
                int(row["generation_seed"]),
                axis_key(row[axis_field]),
                str(row["stream"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("RNG-pairing audit key is malformed.") from exc
        epsilon_hash = str(row.get("epsilon_hash", ""))
        neutral_key = (key[0], key[1], key[3], key[5])
        if (
            key in observed
            or row.get("schema_version") != PAIRING_AUDIT_SCHEMA
            or row.get("status") != "PASS"
            or row.get("epsilon_depends_on_training_seed") != "false"
            or not epsilon_hash
        ):
            raise ProtocolError("RNG-pairing audit row is invalid.")
        observed.add(key)
        paired.setdefault(neutral_key, set()).add(epsilon_hash)
    if observed != expected:
        raise ProtocolError("RNG-pairing audit coverage differs from realized metric cells.")
    if not paired or any(len(values) != 1 for values in paired.values()):
        raise ProtocolError("Evaluation epsilon is not paired across arms and training seeds.")


def validate_workspace_provenance(
    root: Path,
    config: object,
    *,
    experiment_id: str,
    protocol: Mapping[str, object],
) -> None:
    if not _complete_production_coverage(config):
        return
    manifest = read_json(Path(root) / "provenance/input_artifacts.json")
    if (
        manifest.get("schema_version") != "midogpp_input_artifacts_v2"
        or manifest.get("dataset_id") != "midogpp"
        or manifest.get("experiment_id") != experiment_id
        or manifest.get("stage") != "20_cvae_preservation"
        or manifest.get("claim_scope") != "cvae_source_inner_study_only"
        or manifest.get("selection_used_target_eval_artifacts") is not False
    ):
        raise ProtocolError("Study workspace provenance identity mismatch.")
    inputs = manifest.get("input_artifacts")
    if not isinstance(inputs, list) or not all(
        isinstance(row, Mapping) for row in inputs
    ):
        raise ProtocolError("Malformed study workspace input-artifact records.")
    artifact_ids = [str(row.get("artifact_id", "")) for row in inputs]
    by_id = {artifact_id: row for artifact_id, row in zip(artifact_ids, inputs)}
    expected_ids = {
        "midogpp_dataset_contract_annotation_patch_v1",
        "midogpp_virchow2_xyxy_feature_cache_seed42",
    }
    if (
        any(not artifact_id for artifact_id in artifact_ids)
        or len(inputs) != len(by_id)
        or set(by_id) != expected_ids
    ):
        raise ProtocolError("Study workspace inputs differ from the canonical pair.")
    expected_files = {
        "midogpp_dataset_contract_annotation_patch_v1": {
            "dataset_contract.json",
            "manifest.csv",
            "split_manifest.csv",
            "leakage_report.json",
            "path_relocation.json",
        },
        "midogpp_virchow2_xyxy_feature_cache_seed42": {
            "embeddings/train.pt"
        },
    }
    for artifact_id, row in by_id.items():
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
        if not isinstance(files, list) or not all(
            isinstance(file_row, Mapping) for file_row in files
        ):
            raise ProtocolError(
                f"Input artifact {artifact_id} has malformed provenance files."
            )
        paths = [str(file_row.get("path", "")) for file_row in files]
        if len(paths) != len(set(paths)) or set(paths) != expected_files[artifact_id]:
            raise ProtocolError(
                f"Input artifact {artifact_id} provenance-file coverage changed."
            )
        for file_row in files:
            if file_row.get("exists") is not True:
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
                if (
                    not algorithm
                    or computed.get(algorithm) != expected.get("digest")
                    or file_row.get("verification") != "MATCH"
                ):
                    raise ProtocolError(
                        f"Input artifact {artifact_id} failed expected hash verification."
                    )
            elif file_row.get("verification") != "RECORDED_NO_EXPECTATION":
                raise ProtocolError(
                    f"Input artifact {artifact_id} has invalid hash-verification state."
                )
    if _recorded_file_hash(
        by_id["midogpp_dataset_contract_annotation_patch_v1"], "manifest.csv"
    ) != protocol.get("manifest_hash"):
        raise ProtocolError(
            "Dataset manifest provenance hash differs from the study protocol."
        )
    if _recorded_file_hash(
        by_id["midogpp_virchow2_xyxy_feature_cache_seed42"],
        "embeddings/train.pt",
    ) != protocol.get("feature_cache_hash"):
        raise ProtocolError(
            "Feature-cache provenance hash differs from the study protocol."
        )


def _recorded_file_hash(
    artifact: Mapping[str, object], relative_path: str
) -> str:
    integrity = artifact.get("file_integrity")
    if not isinstance(integrity, Mapping):
        raise ProtocolError("Malformed study input-artifact file integrity.")
    files = integrity.get("files")
    if not isinstance(files, list):
        raise ProtocolError("Malformed study input-artifact file list.")
    matches = [
        row
        for row in files
        if isinstance(row, Mapping) and row.get("path") == relative_path
    ]
    if len(matches) != 1:
        raise ProtocolError(
            f"Study input provenance lacks unique identity for {relative_path}."
        )
    computed = matches[0].get("computed")
    if not isinstance(computed, Mapping) or not _is_sha256(
        computed.get("sha256")
    ):
        raise ProtocolError(
            f"Study input provenance lacks SHA-256 for {relative_path}."
        )
    return str(computed["sha256"])


def _is_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _complete_production_coverage(config: object) -> bool:
    return (
        tuple(str(value) for value in getattr(config, "heldout_centers"))
        == MIDOGPP_ELIGIBLE_CENTERS
        and tuple(int(value) for value in getattr(config, "training_seeds"))
        == (17, 42, 101)
        and tuple(int(value) for value in getattr(config, "generation_seeds"))
        == (17, 42, 101)
    )
