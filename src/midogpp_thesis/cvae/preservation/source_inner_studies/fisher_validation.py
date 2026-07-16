"""Fail-closed validator for Task-Fisher shrinkage study bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from ....real_features.classifier_reference.artifacts import stable_hash
from ....real_features.classifier_reference.protocol import ProtocolError
from ....real_features.classifier_reference.schemas.midogpp import MIDOGPP_ELIGIBLE_CENTERS
from ..prior_recovery_runtime_cache import validate_feature_frame_index
from .checkpoint_store import validate_study_checkpoint_index
from .config import TaskFisherShrinkageStudyConfig, decision_contract_hash, load_fisher_study_config, study_contract_hash
from .contracts import FisherStudyMetricV2
from .fisher_artifacts import FISHER_STATE_INDEX_SCHEMA
from .fisher_runner import EXPERIMENT_ID, _decisions, _study_summary
from .preparation import embedded_v1_preparation_lineage
from .validation_common import (
    COVERAGE_SCHEMA,
    FISHER_SAMPLER_SCHEMA,
    PROTOCOL_SCHEMA,
    SELECTION_EVIDENCE_SCHEMA,
    canonical_rows,
    expected_bundle_files,
    read_csv,
    read_json,
    require_files,
    selection_evidence_hash,
    study_implementation_lineage,
    validate_common_rows,
    validate_embedded_preparation_rows,
    validate_generation_budgets,
    validate_initialization_index,
    validate_metric_grid,
    validate_rng_rows,
    validate_workspace_provenance,
)


def validate_fisher_study_bundle(root: Path, *, expected_config: TaskFisherShrinkageStudyConfig | None = None) -> dict[str, Mapping[str, object]]:
    root = Path(root)
    config = expected_config or load_fisher_study_config(root / "config.resolved.yaml")
    require_files(root, expected_bundle_files(config, state_index_relative="manifests/task_fisher_shrinkage_state_index.json"))
    protocol = read_json(root / "manifests/protocol_manifest.json")
    _validate_protocol(protocol, config)
    lineage = embedded_v1_preparation_lineage()
    if read_json(root / "manifests/embedded_v1_preparation_lineage.json") != lineage:
        raise ProtocolError("Embedded v1 preparation lineage changed.")
    if protocol.get("embedded_preparation_lineage_hash") != lineage["lineage_hash"]:
        raise ProtocolError("Protocol is not bound to the embedded preparation lineage.")
    coverage = read_json(root / "manifests/coverage_manifest.json")
    if coverage.get("schema_version") != COVERAGE_SCHEMA or coverage.get("status") != "PASS":
        raise ProtocolError("Fisher study coverage is not PASS.")
    leakage = read_json(root / "reports/leakage_report.json")
    run_state = read_json(root / "reports/run_state.json")
    if (
        leakage.get("status") != "PASS"
        or leakage.get("protocol_hash") != protocol.get("protocol_hash")
        or leakage.get("outer_target_rows_used") is not False
        or leakage.get("inner_rows_used_for_fit") is not False
        or leakage.get("target_eval_labels_used_for_selection") is not False
        or leakage.get("selection_used_target_eval_artifacts") is not False
        or leakage.get("identity_overlap_pass") is not True
        or run_state.get("status") != "COMPLETE"
        or run_state.get("protocol_hash") != protocol.get("protocol_hash")
        or run_state.get("mode") != config.mode
    ):
        raise ProtocolError("Fisher study leakage/run state is incomplete.")
    checkpoint_index = validate_study_checkpoint_index(root)
    initialization_index = read_json(root / "manifests/initialization_index.json")
    validate_initialization_index(checkpoint_index, initialization_index)
    frame_index = validate_feature_frame_index(root)
    budget_manifest = read_json(root / "manifests/generation_budget_manifest.json")
    state_index = read_json(root / "manifests/task_fisher_shrinkage_state_index.json")
    if state_index.get("schema_version") != FISHER_STATE_INDEX_SCHEMA or not isinstance(state_index.get("records"), list):
        raise ProtocolError("Malformed Task-Fisher shrinkage state index.")
    state_by_fold = _validate_fisher_states(state_index, config=config)
    metrics = read_csv(root / "tables/source_inner_metrics.csv")
    deltas = read_csv(root / "tables/paired_deltas.csv")
    nested = read_csv(root / "tables/nested_real_references.csv")
    tuning = read_csv(root / "tables/nested_classifier_tuning.csv")
    samplers = read_csv(root / "tables/sampler_realizations.csv")
    checkpoint_audits = read_csv(root / "tables/checkpoint_reuse_audit.csv")
    initialization_audits = read_csv(root / "tables/initialization_pairing_audit.csv")
    budget_rows = read_csv(root / "tables/generation_budget_audit.csv")
    rng_rows = read_csv(root / "tables/rng_pairing_audit.csv")
    identity = read_csv(root / "tables/identity_overlap_audit.csv")
    validate_common_rows(config, metric_rows=metrics, identity_rows=identity)
    validate_embedded_preparation_rows(
        config,
        metric_rows=metrics,
        nested_reference_rows=nested,
        nested_tuning_rows=tuning,
        identity_rows=identity,
    )
    validate_metric_grid(
        config,
        metric_rows=metrics,
        axis_field="alpha",
        axis_values=config.alphas,
        protocol_hash=str(protocol["protocol_hash"]),
    )
    _validate_fisher_metric_semantics(metrics, state_by_fold)
    _validate_sampler_rows(
        samplers,
        metric_rows=metrics,
        checkpoint_index=checkpoint_index,
        latent_dim=config.latent_dim,
    )
    _validate_fisher_checkpoint_binding(checkpoint_index, state_by_fold)
    _validate_coverage_manifest(
        coverage,
        config=config,
        metric_rows=metrics,
        n_fisher_states=len(state_by_fold),
    )
    validate_generation_budgets(
        config,
        budget_rows=budget_rows,
        budget_manifest=budget_manifest,
        metric_rows=metrics,
    )
    _validate_pairing_audits(
        config,
        checkpoint_rows=checkpoint_audits,
        initialization_rows=initialization_audits,
        checkpoint_index=checkpoint_index,
        states=state_by_fold,
    )
    validate_rng_rows(metric_rows=metrics, rng_rows=rng_rows, axis_field="alpha")
    _validate_metric_checkpoint_references(metrics, checkpoint_index)
    validate_feature_frame_index(root, expected_frame_hashes={str(row["frame_hash"]) for row in metrics})
    observed_hash = selection_evidence_hash(
        metric_rows=metrics,
        paired_delta_rows=deltas,
        nested_reference_rows=nested,
        nested_tuning_rows=tuning,
        sampler_rows=samplers,
        identity_rows=identity,
        checkpoint_reuse_rows=checkpoint_audits,
        initialization_pairing_rows=initialization_audits,
        generation_budget_rows=budget_rows,
        checkpoint_index=checkpoint_index,
        initialization_index=initialization_index,
        feature_frame_index=frame_index,
        generation_budget_manifest=budget_manifest,
        rng_rows=rng_rows,
        protocol_manifest=protocol,
        study_state_index=state_index,
    )
    manifest = read_json(root / "manifests/selection_evidence_manifest.json")
    if (
        manifest.get("schema_version") != SELECTION_EVIDENCE_SCHEMA
        or manifest.get("selection_evidence_hash") != observed_hash
        or manifest.get("runtime_rows_included") is not False
        or manifest.get("decisions_may_feed_model_recipe") is not False
    ):
        raise ProtocolError("Fisher selection-evidence hash mismatch.")
    decision_metrics = _decision_metrics(metrics)
    decisions, children, expected_deltas = _decisions(
        config,
        decision_metrics=decision_metrics,
        protocol_hash=str(protocol["protocol_hash"]),
        evidence_hash=observed_hash,
    )
    if canonical_rows(deltas) != canonical_rows(expected_deltas):
        raise ProtocolError("Fisher paired deltas are not recomputable.")
    for outer, expected in decisions.items():
        if read_json(root / f"reports/consensus_decisions/{outer}.json") != expected:
            raise ProtocolError("Fisher consensus decision is not recomputable.")
    for (seed, outer), expected in children.items():
        if read_json(root / f"reports/child_decisions/seed{seed}/{outer}.json") != expected:
            raise ProtocolError("Fisher child decision is not recomputable.")
    expected_summary = _study_summary(
        decisions,
        protocol_hash=str(protocol["protocol_hash"]),
        evidence_hash=observed_hash,
    )
    if read_json(root / "reports/study_decision.json") != expected_summary:
        raise ProtocolError("Fisher study summary is not recomputable.")
    _validate_nonconsumable(root, config)
    validate_workspace_provenance(
        root,
        config,
        experiment_id=EXPERIMENT_ID,
        protocol=protocol,
    )
    return decisions


def _validate_protocol(protocol: Mapping[str, object], config: TaskFisherShrinkageStudyConfig) -> None:
    payload = dict(protocol)
    recorded = payload.pop("protocol_hash", None)
    if protocol.get("schema_version") != PROTOCOL_SCHEMA or protocol.get("experiment_id") != EXPERIMENT_ID or protocol.get("mode") != config.mode or protocol.get("claim_scope") != "cvae_source_inner_study_only" or protocol.get("study_contract_hash") != study_contract_hash(config) or protocol.get("implementation_lineage") != study_implementation_lineage(config.mode) or protocol.get("decision_contract_hash") != decision_contract_hash(config) or stable_hash(payload) != recorded:
        raise ProtocolError("Fisher protocol manifest mismatch.")


def _validate_fisher_states(
    index: Mapping[str, object], *, config: TaskFisherShrinkageStudyConfig
) -> dict[tuple[str, str], Mapping[str, object]]:
    import numpy as np

    observed: dict[tuple[str, str], Mapping[str, object]] = {}
    for record in index["records"]:
        if not isinstance(record, Mapping):
            raise ProtocolError("Malformed raw Fisher state record.")
        base = {key: value for key, value in record.items() if key not in {"raw_fisher_state_hash", "derived_metrics"}}
        if stable_hash(base) != record.get("raw_fisher_state_hash"):
            raise ProtocolError("Raw Fisher state hash mismatch.")
        key = (
            str(record.get("outer_target_center", "")),
            str(record.get("inner_pseudo_target_center", "")),
        )
        try:
            raw = np.asarray(record["raw_fisher"], dtype=np.float64)
            trace_raw = float(record["trace_raw"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("Raw Fisher numerical state is malformed.") from exc
        if (
            key in observed
            or key[0] not in config.heldout_centers
            or key[1] not in MIDOGPP_ELIGIBLE_CENTERS
            or key[0] == key[1]
            or raw.shape != (config.pca_dim, config.pca_dim)
            or not np.isfinite(raw).all()
            or not np.allclose(raw, raw.T)
            or not np.isclose(float(np.trace(raw)), trace_raw)
            or record.get("fit_scope") != config.raw_fisher_fit_scope
            or record.get("shared_across_training_seeds") is not True
            or not str(record.get("probe_config_hash", ""))
            or tuple(str(value) for value in record.get("fit_centers", ()))
            != tuple(
                center
                for center in MIDOGPP_ELIGIBLE_CENTERS
                if center not in {key[0], key[1]}
            )
            or not str(record.get("source_row_hash", ""))
            or not str(record.get("frame_hash", ""))
        ):
            raise ProtocolError("Raw Fisher state contract mismatch.")
        raw_valid = record.get("valid") is True
        if raw_valid and (
            trace_raw <= 0.0
            or int(record.get("rank", -1)) not in {0, 1}
            or float(np.linalg.eigvalsh(raw).min()) < -1e-8
        ):
            raise ProtocolError("Valid raw Fisher state is not rank-one PSD.")
        derived = record.get("derived_metrics")
        if not isinstance(derived, Mapping) or set(derived) != {"0.00", "0.05", "0.10", "0.25"}:
            raise ProtocolError("Fisher derived metric coverage mismatch.")
        normalized = (
            float(config.pca_dim) * raw / trace_raw if raw_valid else None
        )
        for alpha in config.alphas:
            alpha_key = format(alpha, ".2f")
            state = derived[alpha_key]
            if not isinstance(state, Mapping):
                raise ProtocolError("Malformed derived Fisher metric state.")
            hashed = dict(state)
            state_hash = hashed.pop("metric_state_hash", None)
            if stable_hash(hashed) != state_hash:
                raise ProtocolError("Derived Fisher metric-state hash mismatch.")
            expected_raw_hash = (
                "none" if alpha == 0.0 else record["raw_fisher_state_hash"]
            )
            expected_ratio = 1.0 + float(alpha) * float(config.pca_dim)
            if (
                float(state.get("alpha", float("nan"))) != float(alpha)
                or state.get("raw_fisher_state_hash") != expected_raw_hash
                or not np.isclose(
                    float(state.get("directional_ratio", float("nan"))),
                    expected_ratio,
                )
                or state.get("literal_isotropic_metric_none") is not (alpha == 0.0)
                or state.get("valid") is not (True if alpha == 0.0 else raw_valid)
            ):
                raise ProtocolError("Derived Fisher metric identity mismatch.")
            metric = state.get("metric")
            if alpha == 0.0:
                if (
                    metric is not None
                    or not np.isclose(float(state["fisher_direction_eigenvalue"]), 1.0)
                    or not np.isclose(float(state["orthogonal_eigenvalue"]), 1.0)
                ):
                    raise ProtocolError("alpha=0 is not the literal isotropic path.")
            elif raw_valid:
                expected_metric = (
                    np.eye(config.pca_dim) + float(alpha) * normalized
                ) / (1.0 + float(alpha))
                observed_metric = np.asarray(metric, dtype=np.float64)
                if (
                    observed_metric.shape != expected_metric.shape
                    or not np.allclose(observed_metric, expected_metric)
                    or not np.isclose(np.trace(observed_metric), config.pca_dim)
                    or not np.isclose(
                        float(state["fisher_direction_eigenvalue"]),
                        (1.0 + float(alpha) * config.pca_dim)
                        / (1.0 + float(alpha)),
                    )
                    or not np.isclose(
                        float(state["orthogonal_eigenvalue"]),
                        1.0 / (1.0 + float(alpha)),
                    )
                ):
                    raise ProtocolError("Derived shrunk Fisher metric formula mismatch.")
            elif metric is not None:
                raise ProtocolError("Invalid raw Fisher state emitted a derived metric.")
        observed[key] = record
    expected = {
        (str(outer), inner)
        for outer in config.heldout_centers
        for inner in MIDOGPP_ELIGIBLE_CENTERS
        if inner != outer
    }
    if set(observed) != expected:
        raise ProtocolError("Raw Fisher state index has incomplete H/I coverage.")
    return observed


def _validate_fisher_metric_semantics(
    rows: list[dict[str, str]],
    states: Mapping[tuple[str, str], Mapping[str, object]],
) -> None:
    for row in rows:
        alpha = float(row["alpha"])
        state = states[(row["outer_target_center"], row["inner_pseudo_target_center"])]
        raw_valid = state.get("valid") is True
        derived = state["derived_metrics"]
        if not isinstance(derived, Mapping):
            raise ProtocolError("Malformed derived Fisher metric mapping.")
        metric_state = derived[format(alpha, ".2f")]
        if not isinstance(metric_state, Mapping):
            raise ProtocolError("Malformed alpha-specific Fisher metric state.")
        if (
            row.get("model_family") != "class_conditioned_cvae_v1"
            or row.get("prior_family") != "standard_normal"
            or row.get("objective_id")
            != ("stochastic_isotropic_v1" if alpha == 0.0 else "stochastic_task_fisher_v1")
            or row.get("raw_fisher_state_hash")
            != ("none" if alpha == 0.0 else state["raw_fisher_state_hash"])
            or row.get("objective_context_hash")
            != ("none" if alpha == 0.0 else metric_state["metric_state_hash"])
            or row.get("classifier_spec_hash") != state.get("probe_config_hash")
            or row.get("fit_row_hash") != state.get("source_row_hash")
            or row.get("frame_hash") != state.get("frame_hash")
        ):
            raise ProtocolError("Fisher metric row objective identity mismatch.")
        placeholder_expected = alpha > 0.0 and not raw_valid
        if (
            (row.get("training_key_hash") == "none") is not placeholder_expected
            or (row.get("checkpoint_hash") == "none") is not placeholder_expected
        ):
            raise ProtocolError("Fisher metric checkpoint placeholder semantics mismatch.")
        if placeholder_expected:
            if (
                row.get("training_key_hash") != "none"
                or row.get("checkpoint_hash") != "none"
                or row.get("status") != "raw_fisher_invalid"
                or row.get("valid") != "false"
            ):
                raise ProtocolError("Invalid raw Fisher evidence was not rendered fail-closed.")


def _validate_sampler_rows(
    rows: list[dict[str, str]],
    *,
    metric_rows: list[dict[str, str]],
    checkpoint_index: Mapping[str, object],
    latent_dim: int,
) -> None:
    import numpy as np

    records = checkpoint_index.get("records")
    if not isinstance(records, list):
        raise ProtocolError("Malformed Fisher checkpoint records.")
    by_key = {
        str(record.get("training_key_hash", "")): record
        for record in records
        if isinstance(record, Mapping)
    }
    grouped: dict[tuple[str, str, int, float], list[Mapping[str, str]]] = {}
    for row in rows:
        key = (
            row.get("outer_target_center", ""),
            row.get("inner_pseudo_target_center", ""),
            int(row.get("training_seed", -1)),
            float(row.get("alpha", "nan")),
        )
        grouped.setdefault(key, []).append(row)
    expected = {
        (
            str(record["training_key"]["outer_target_center"]),
            str(record["training_key"]["inner_pseudo_target_center"]),
            int(record["training_key"]["training_seed"]),
            float(record["training_key"]["alpha"]),
        )
        for record in records
        if isinstance(record, Mapping) and isinstance(record.get("training_key"), Mapping)
    }
    if set(grouped) != expected:
        raise ProtocolError("Fisher standard-sampler coverage mismatch.")
    state_by_key: dict[str, str] = {}
    for key, class_rows in grouped.items():
        if (
            len(class_rows) != 2
            or {int(row.get("class_label", -1)) for row in class_rows} != {0, 1}
        ):
            raise ProtocolError("Fisher sampler must record exactly both classes.")
        first = class_rows[0]
        training_key_hash = first.get("training_key_hash", "")
        checkpoint = by_key.get(training_key_hash)
        if not isinstance(checkpoint, Mapping):
            raise ProtocolError("Fisher sampler references an unknown checkpoint.")
        training_key = checkpoint.get("training_key")
        if not isinstance(training_key, Mapping):
            raise ProtocolError("Fisher sampler checkpoint lacks its training key.")
        expected_hash = stable_hash(
            {
                "family": "standard_normal",
                "latent_dim": latent_dim,
                "source_row_hash": training_key["fit_row_hash"],
                "training_key_hash": training_key_hash,
                "checkpoint_hash": checkpoint["checkpoint_hash"],
            }
        )
        for row in class_rows:
            try:
                mean = np.asarray(json.loads(row["mean"]), dtype=np.float64)
                variance = np.asarray(json.loads(row["variance"]), dtype=np.float64)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ProtocolError("Malformed Fisher standard-sampler state.") from exc
            if (
                row.get("schema_version") != FISHER_SAMPLER_SCHEMA
                or row.get("mechanism") != "standard_normal"
                or row.get("requested_family") != "standard_normal"
                or row.get("realized_family") != "standard_normal"
                or int(row.get("latent_dim", -1)) != latent_dim
                or row.get("source_row_hash") != training_key.get("fit_row_hash")
                or (
                    training_key.get("outer_target_center"),
                    training_key.get("inner_pseudo_target_center"),
                    int(training_key.get("training_seed", -1)),
                    float(training_key.get("alpha", float("nan"))),
                )
                != key
                or row.get("training_key_hash") != training_key_hash
                or row.get("checkpoint_hash") != checkpoint.get("checkpoint_hash")
                or row.get("sampler_state_hash") != expected_hash
                or mean.shape != (latent_dim,)
                or variance.shape != (latent_dim,)
                or not np.allclose(mean, 0.0)
                or not np.allclose(variance, 1.0)
                or row.get("fallback_reason", "") != ""
            ):
                raise ProtocolError("Fisher standard-sampler identity mismatch.")
        state_by_key[training_key_hash] = expected_hash
    for row in metric_rows:
        training_key_hash = row.get("training_key_hash", "")
        if training_key_hash == "none":
            if row.get("sampler_state_hash") != "none":
                raise ProtocolError("Invalid Fisher metric references a sampler state.")
        elif row.get("sampler_state_hash") != state_by_key.get(training_key_hash):
            raise ProtocolError("Fisher metric/sampler-state binding mismatch.")


def _validate_fisher_checkpoint_binding(
    checkpoint_index: Mapping[str, object],
    states: Mapping[tuple[str, str], Mapping[str, object]],
) -> None:
    records = checkpoint_index.get("records")
    if not isinstance(records, list):
        raise ProtocolError("Malformed Fisher checkpoint records.")
    for record in records:
        if not isinstance(record, Mapping):
            raise ProtocolError("Malformed Fisher checkpoint record.")
        key = record.get("training_key")
        if not isinstance(key, Mapping):
            raise ProtocolError("Fisher checkpoint lacks its training key.")
        alpha = float(key.get("alpha", float("nan")))
        state = states.get(
            (
                str(key.get("outer_target_center", "")),
                str(key.get("inner_pseudo_target_center", "")),
            )
        )
        if not isinstance(state, Mapping):
            raise ProtocolError("Fisher checkpoint references an unknown H/I state.")
        expected = "none" if alpha == 0.0 else state["raw_fisher_state_hash"]
        derived = state.get("derived_metrics")
        if not isinstance(derived, Mapping):
            raise ProtocolError("Fisher checkpoint references malformed derived state.")
        expected_context = (
            "none"
            if alpha == 0.0
            else derived[format(alpha, ".2f")]["metric_state_hash"]
        )
        if (
            key.get("raw_fisher_state_hash") != expected
            or key.get("objective_context_hash") != expected_context
        ):
            raise ProtocolError("Fisher checkpoint/derived-metric binding mismatch.")


def _validate_coverage_manifest(
    coverage: Mapping[str, object],
    *,
    config: TaskFisherShrinkageStudyConfig,
    metric_rows: list[dict[str, str]],
    n_fisher_states: int,
) -> None:
    expected_cells = (
        len(config.heldout_centers)
        * (len(MIDOGPP_ELIGIBLE_CENTERS) - 1)
        * len(config.training_seeds)
        * len(config.generation_seeds)
        * len(config.alphas)
    )
    observed_cells = sum(
        row.get("representation_role") == "prior" for row in metric_rows
    )
    if (
        coverage.get("status") != "PASS"
        or int(coverage.get("expected_decision_cells", -1)) != expected_cells
        or int(coverage.get("observed_decision_cells", -1)) != observed_cells
        or observed_cells != expected_cells
        or int(coverage.get("metric_rows", -1)) != len(metric_rows)
        or int(coverage.get("raw_fisher_states", -1)) != n_fisher_states
        or coverage.get("complete_training_generation_seed_cross") is not True
    ):
        raise ProtocolError("Fisher coverage manifest is not recomputable.")


def _validate_pairing_audits(
    config: TaskFisherShrinkageStudyConfig,
    *,
    checkpoint_rows: list[dict[str, str]],
    initialization_rows: list[dict[str, str]],
    checkpoint_index: Mapping[str, object],
    states: Mapping[tuple[str, str], Mapping[str, object]],
) -> None:
    records = checkpoint_index.get("records")
    if not isinstance(records, list):
        raise ProtocolError("Malformed Fisher checkpoint records.")
    by_key = {
        str(record.get("training_key_hash", "")): record
        for record in records
        if isinstance(record, Mapping)
    }
    expected_checkpoint = {
        (str(outer), inner, int(seed), float(alpha))
        for outer in config.heldout_centers
        for inner in MIDOGPP_ELIGIBLE_CENTERS
        if inner != outer
        for seed in config.training_seeds
        for alpha in config.alphas
        if alpha == 0.0 or states[(str(outer), inner)].get("valid") is True
    }
    observed_checkpoint: set[tuple[str, str, int, float]] = set()
    for row in checkpoint_rows:
        key = (
            row.get("outer_target_center", ""),
            row.get("inner_pseudo_target_center", ""),
            int(row.get("training_seed", -1)),
            float(row.get("alpha", "nan")),
        )
        record = by_key.get(row.get("training_key_hash", ""))
        record_key = record.get("training_key") if isinstance(record, Mapping) else None
        if (
            key in observed_checkpoint
            or not isinstance(record, Mapping)
            or not isinstance(record_key, Mapping)
            or (
                record_key.get("outer_target_center"),
                record_key.get("inner_pseudo_target_center"),
                int(record_key.get("training_seed", -1)),
                float(record_key.get("alpha", float("nan"))),
            )
            != key
            or record.get("checkpoint_hash") != row.get("checkpoint_hash")
            or row.get("literal_alpha_zero") != str(key[3] == 0.0).lower()
            or row.get("raw_fisher_state_hash")
            != (
                "none"
                if key[3] == 0.0
                else states[(key[0], key[1])]["raw_fisher_state_hash"]
            )
            or row.get("status") != "PASS"
        ):
            raise ProtocolError("Fisher checkpoint audit mismatch.")
        observed_checkpoint.add(key)
    if observed_checkpoint != expected_checkpoint:
        raise ProtocolError("Fisher checkpoint audit coverage mismatch.")
    expected_initialization = {
        (str(outer), inner, int(seed))
        for outer in config.heldout_centers
        for inner in MIDOGPP_ELIGIBLE_CENTERS
        if inner != outer
        for seed in config.training_seeds
    }
    observed_initialization: set[tuple[str, str, int]] = set()
    for row in initialization_rows:
        key = (
            row.get("outer_target_center", ""),
            row.get("inner_pseudo_target_center", ""),
            int(row.get("training_seed", -1)),
        )
        fold_records = [
            record
            for record in records
            if isinstance(record, Mapping)
            and isinstance(record.get("training_key"), Mapping)
            and record["training_key"].get("outer_target_center") == key[0]
            and record["training_key"].get("inner_pseudo_target_center") == key[1]
            and int(record["training_key"].get("training_seed", -1)) == key[2]
        ]
        expected_alphas = [
            alpha
            for alpha in config.alphas
            if alpha == 0.0 or states[(key[0], key[1])].get("valid") is True
        ]
        try:
            recorded_alphas = [float(value) for value in json.loads(row["alphas_present"])]
            shared_hashes = set(json.loads(row["shared_initialization_hashes"]))
            stream_hashes = set(json.loads(row["training_stream_hashes"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ProtocolError("Malformed Fisher initialization audit row.") from exc
        if (
            key in observed_initialization
            or recorded_alphas != expected_alphas
            or len(fold_records) != len(expected_alphas)
            or shared_hashes
            != {str(record.get("shared_initialization_hash", "")) for record in fold_records}
            or stream_hashes
            != {str(record.get("training_stream_hash", "")) for record in fold_records}
            or len(shared_hashes) != 1
            or len(stream_hashes) != 1
            or row.get("raw_fisher_valid")
            != str(states[(key[0], key[1])].get("valid") is True).lower()
            or row.get("status") != "PASS"
        ):
            raise ProtocolError("Fisher initialization/stochastic pairing mismatch.")
        observed_initialization.add(key)
    if observed_initialization != expected_initialization:
        raise ProtocolError("Fisher initialization audit coverage mismatch.")


def _decision_metrics(rows: list[dict[str, str]]) -> list[FisherStudyMetricV2]:
    decode = {(row["outer_target_center"], row["inner_pseudo_target_center"], int(row["training_seed"]), float(row["alpha"])): row for row in rows if row["representation_role"] == "decode"}
    posterior = {(row["outer_target_center"], row["inner_pseudo_target_center"], int(row["training_seed"]), int(row["generation_seed"]), float(row["alpha"])): row for row in rows if row["representation_role"] == "posterior"}
    output: list[FisherStudyMetricV2] = []
    for row in rows:
        if row["representation_role"] != "prior":
            continue
        base = (row["outer_target_center"], row["inner_pseudo_target_center"], int(row["training_seed"]), float(row["alpha"]))
        post = posterior[base[:3] + (int(row["generation_seed"]), base[3])]
        dec = decode[base]
        output.append(FisherStudyMetricV2(outer_target_center=base[0], inner_pseudo_target_center=base[1], training_seed=base[2], generation_seed=int(row["generation_seed"]), alpha=base[3], preservation_ratio=float(row["preservation_ratio"]), decode_bacc=float(dec["bacc"]), posterior_bacc=float(post["bacc"]), valid=(row["valid"] == "true" and post["valid"] == "true" and dec["valid"] == "true")))
    return output


def _validate_rng_pairing(rows: list[dict[str, str]]) -> None:
    groups: dict[tuple[str, str, str, str], set[str]] = {}
    for row in rows:
        key = (row["outer_target_center"], row["inner_pseudo_target_center"], row["generation_seed"], row["stream"])
        groups.setdefault(key, set()).add(row["epsilon_hash"])
        if row.get("epsilon_depends_on_training_seed") != "false":
            raise ProtocolError("Fisher evaluation epsilon depends on training seed.")
    if not groups or any(len(values) != 1 for values in groups.values()):
        raise ProtocolError("Fisher epsilon is not paired across alpha/seeds.")


def _validate_metric_checkpoint_references(rows: list[dict[str, str]], index: Mapping[str, object]) -> None:
    records = index.get("records")
    if not isinstance(records, list):
        raise ProtocolError("Malformed Fisher checkpoint index records.")
    by_key = {str(record["training_key_hash"]): record for record in records if isinstance(record, Mapping)}
    for row in rows:
        key = row.get("training_key_hash", "")
        if key == "none":
            continue
        record = by_key.get(key)
        training_key = record.get("training_key") if isinstance(record, Mapping) else None
        try:
            fit_centers = json.loads(row["fit_centers"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ProtocolError("Fisher metric/checkpoint fit centers are malformed.") from exc
        if (
            not isinstance(record, Mapping)
            or not isinstance(training_key, Mapping)
            or record.get("checkpoint_hash") != row.get("checkpoint_hash")
            or training_key.get("outer_target_center")
            != row.get("outer_target_center")
            or training_key.get("inner_pseudo_target_center")
            != row.get("inner_pseudo_target_center")
            or int(training_key.get("training_seed", -1))
            != int(row.get("training_seed", -2))
            or training_key.get("fit_centers") != fit_centers
            or training_key.get("fit_row_hash") != row.get("fit_row_hash")
            or training_key.get("frame_hash") != row.get("frame_hash")
            or training_key.get("protocol_hash") != row.get("protocol_hash")
            or training_key.get("model_family") != row.get("model_family")
            or training_key.get("prior_family") != row.get("prior_family")
            or training_key.get("objective_id") != row.get("objective_id")
            or float(training_key.get("alpha", float("nan")))
            != float(row.get("alpha", "nan"))
            or training_key.get("raw_fisher_state_hash")
            != row.get("raw_fisher_state_hash")
            or training_key.get("objective_context_hash")
            != row.get("objective_context_hash")
        ):
            raise ProtocolError("Fisher metric references an unpersisted checkpoint.")
    metric_keys = {
        row.get("training_key_hash", "")
        for row in rows
        if row.get("training_key_hash") != "none"
    }
    if metric_keys != set(by_key):
        raise ProtocolError("Fisher checkpoint index contains unused or missing training keys.")


def _validate_nonconsumable(root: Path, config: TaskFisherShrinkageStudyConfig) -> None:
    if (root / "reports/publication_state.json").exists():
        raise ProtocolError("Non-adoptive Fisher study emitted publication state.")
    for outer in config.heldout_centers:
        payload = read_json(root / f"reports/consensus_decisions/{outer}.json")
        if {"recipe_export_ready", "stage30_recipe_ready", "primary_arm"}.intersection(payload) or payload.get("may_feed_model_recipe") is not False or payload.get("may_feed_deployable_selection") is not False:
            raise ProtocolError("Fisher decision exposes recipe semantics.")
