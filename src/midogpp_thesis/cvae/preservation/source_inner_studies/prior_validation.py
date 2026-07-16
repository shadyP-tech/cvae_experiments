"""Fail-closed validator for learned conditional-prior study bundles."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

from ....real_features.classifier_reference.artifacts import stable_hash
from ....real_features.classifier_reference.protocol import ProtocolError
from ....real_features.classifier_reference.schemas.midogpp import MIDOGPP_ELIGIBLE_CENTERS
from ...latent_priors import (
    ACTIVE_UNIT_THRESHOLD,
    CLASS_SEPARATION_THRESHOLD,
    PRIOR_SATURATION_THRESHOLD,
)
from ..prior_recovery_runtime_cache import validate_feature_frame_index
from .checkpoint_store import validate_study_checkpoint_index
from .config import (
    LearnedConditionalPriorStudyConfig,
    decision_contract_hash,
    load_prior_study_config,
    study_contract_hash,
)
from .contracts import LEARNED_PRIOR_MODEL_FAMILY, PriorStudyMetricV2
from .preparation import embedded_v1_preparation_lineage
from .prior_artifacts import PRIOR_STATE_INDEX_SCHEMA
from .prior_runner import EXPERIMENT_ID, _decisions, _study_summary
from .validation_common import (
    COVERAGE_SCHEMA,
    PROTOCOL_SCHEMA,
    PRIOR_SAMPLER_SCHEMA,
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


def validate_prior_study_bundle(
    root: Path,
    *,
    expected_config: LearnedConditionalPriorStudyConfig | None = None,
) -> dict[str, Mapping[str, object]]:
    root = Path(root)
    config = expected_config or load_prior_study_config(root / "config.resolved.yaml")
    require_files(
        root,
        expected_bundle_files(
            config,
            state_index_relative="manifests/learned_prior_state_index.json",
        ),
    )
    protocol = read_json(root / "manifests/protocol_manifest.json")
    _validate_protocol(protocol, config)
    lineage = embedded_v1_preparation_lineage()
    if read_json(root / "manifests/embedded_v1_preparation_lineage.json") != lineage:
        raise ProtocolError("Embedded v1 preparation lineage changed.")
    if protocol.get("embedded_preparation_lineage_hash") != lineage["lineage_hash"]:
        raise ProtocolError("Protocol is not bound to the embedded preparation lineage.")
    coverage = read_json(root / "manifests/coverage_manifest.json")
    if coverage.get("schema_version") != COVERAGE_SCHEMA or coverage.get("status") != "PASS":
        raise ProtocolError("Learned-prior study coverage is not PASS.")
    leakage = read_json(root / "reports/leakage_report.json")
    if (
        leakage.get("status") != "PASS"
        or leakage.get("protocol_hash") != protocol.get("protocol_hash")
        or leakage.get("outer_target_rows_used") is not False
        or leakage.get("inner_rows_used_for_fit") is not False
        or leakage.get("target_eval_labels_used_for_selection") is not False
        or leakage.get("selection_used_target_eval_artifacts") is not False
        or leakage.get("identity_overlap_pass") is not True
    ):
        raise ProtocolError("Learned-prior study leakage report is not PASS.")
    run_state = read_json(root / "reports/run_state.json")
    if (
        run_state.get("status") != "COMPLETE"
        or run_state.get("protocol_hash") != protocol.get("protocol_hash")
        or run_state.get("mode") != config.mode
    ):
        raise ProtocolError("Learned-prior study run state is not COMPLETE.")

    checkpoint_index = validate_study_checkpoint_index(root)
    initialization_index = read_json(root / "manifests/initialization_index.json")
    validate_initialization_index(checkpoint_index, initialization_index)
    frame_index = validate_feature_frame_index(root)
    budget_manifest = read_json(root / "manifests/generation_budget_manifest.json")
    state_index = read_json(root / "manifests/learned_prior_state_index.json")
    if state_index.get("schema_version") != PRIOR_STATE_INDEX_SCHEMA or not isinstance(
        state_index.get("records"), list
    ):
        raise ProtocolError("Malformed learned-prior state index.")
    state_records: dict[tuple[str, str, int], Mapping[str, object]] = {}
    checkpoint_records = {
        str(record["training_key_hash"]): record
        for record in checkpoint_index["records"]
        if isinstance(record, Mapping)
    }
    for record in state_index["records"]:
        if not isinstance(record, Mapping) or stable_hash(
            {key: value for key, value in record.items() if key != "state_hash"}
        ) != record.get("state_hash"):
            raise ProtocolError("Learned-prior state record hash mismatch.")
        key = (
            str(record.get("outer_target_center", "")),
            str(record.get("inner_pseudo_target_center", "")),
            int(record.get("training_seed", -1)),
        )
        checkpoint = checkpoint_records.get(str(record.get("training_key_hash", "")))
        checkpoint_key = checkpoint.get("training_key") if isinstance(checkpoint, Mapping) else None
        if key in state_records or not isinstance(checkpoint, Mapping) or checkpoint.get(
            "checkpoint_hash"
        ) != record.get("checkpoint_hash") or checkpoint.get(
            "model_family"
        ) != LEARNED_PRIOR_MODEL_FAMILY or not isinstance(
            checkpoint_key, Mapping
        ) or (
            checkpoint_key.get("outer_target_center"),
            checkpoint_key.get("inner_pseudo_target_center"),
            int(checkpoint_key.get("training_seed", -1)),
        ) != key or checkpoint_key.get("fit_row_hash") != record.get(
            "source_row_hash"
        ) or checkpoint_key.get("frame_hash") != record.get(
            "frame_hash"
        ) or checkpoint.get("prior_partition_hash") != record.get(
            "final_prior_partition_hash"
        ):
            raise ProtocolError("Learned-prior state/checkpoint identity mismatch.")
        _validate_prior_state_record(
            record,
            latent_dim=config.latent_dim,
            train_epochs=config.train_epochs,
        )
        state_records[key] = record

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
        axis_field="arm",
        axis_values=config.arms,
        protocol_hash=str(protocol["protocol_hash"]),
    )
    _validate_state_coverage(config, set(state_records))
    _validate_coverage_manifest(
        coverage,
        config=config,
        metric_rows=metrics,
        n_prior_states=len(state_records),
    )
    sampler_states = _validate_sampler_rows(
        samplers,
        config=config,
        prior_states=state_records,
        checkpoint_index=checkpoint_index,
    )
    _validate_prior_metric_semantics(
        metrics,
        config,
        prior_states=state_records,
        sampler_states=sampler_states,
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
    )
    validate_rng_rows(metric_rows=metrics, rng_rows=rng_rows, axis_field="arm")
    _validate_metric_checkpoint_references(metrics, checkpoint_index)
    expected_frames = {str(row["frame_hash"]) for row in metrics}
    validate_feature_frame_index(root, expected_frame_hashes=expected_frames)

    observed_evidence_hash = selection_evidence_hash(
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
    selection_manifest = read_json(root / "manifests/selection_evidence_manifest.json")
    if (
        selection_manifest.get("schema_version") != SELECTION_EVIDENCE_SCHEMA
        or selection_manifest.get("selection_evidence_hash") != observed_evidence_hash
        or selection_manifest.get("decisions_may_feed_model_recipe") is not False
        or selection_manifest.get("runtime_rows_included") is not False
    ):
        raise ProtocolError("Learned-prior selection-evidence hash mismatch.")

    decision_metrics = _decision_metrics(metrics)
    decisions, children, expected_deltas = _decisions(
        config,
        decision_metrics=decision_metrics,
        protocol_hash=str(protocol["protocol_hash"]),
        selection_evidence_hash_value=observed_evidence_hash,
    )
    if canonical_rows(deltas) != canonical_rows(expected_deltas):
        raise ProtocolError("Learned-prior paired deltas are not recomputable.")
    for outer, expected in decisions.items():
        if read_json(root / f"reports/consensus_decisions/{outer}.json") != expected:
            raise ProtocolError("Learned-prior consensus decision is not recomputable.")
    for (seed, outer), expected in children.items():
        if read_json(root / f"reports/child_decisions/seed{seed}/{outer}.json") != expected:
            raise ProtocolError("Learned-prior child decision is not recomputable.")
    expected_summary = _study_summary(
        decisions,
        protocol_hash=str(protocol["protocol_hash"]),
        evidence_hash=observed_evidence_hash,
    )
    if read_json(root / "reports/study_decision.json") != expected_summary:
        raise ProtocolError("Learned-prior study summary is not recomputable.")
    _validate_nonconsumable(root, config)
    validate_workspace_provenance(
        root,
        config,
        experiment_id=EXPERIMENT_ID,
        protocol=protocol,
    )
    return decisions


def _validate_protocol(protocol: Mapping[str, object], config: LearnedConditionalPriorStudyConfig) -> None:
    payload = dict(protocol)
    recorded = payload.pop("protocol_hash", None)
    if (
        protocol.get("schema_version") != PROTOCOL_SCHEMA
        or protocol.get("experiment_id") != EXPERIMENT_ID
        or protocol.get("mode") != config.mode
        or protocol.get("claim_scope") != "cvae_source_inner_study_only"
        or protocol.get("study_contract_hash") != study_contract_hash(config)
        or protocol.get("implementation_lineage")
        != study_implementation_lineage(config.mode)
        or protocol.get("decision_contract_hash") != decision_contract_hash(config)
        or stable_hash(payload) != recorded
    ):
        raise ProtocolError("Learned-prior protocol manifest mismatch.")


def _validate_prior_state_record(
    record: Mapping[str, object], *, latent_dim: int, train_epochs: int
) -> None:
    import numpy as np

    state = record.get("state")
    diagnostics = record.get("diagnostics")
    gaps = record.get("ex_post_diagonal_moment_gap_by_class")
    statistics = record.get("posterior_sufficient_statistics_by_class")
    kl_audit = record.get("kl_to_learned_prior_by_class")
    trajectory = record.get("prior_training_trajectory")
    if not isinstance(state, Mapping) or not isinstance(diagnostics, Mapping):
        raise ProtocolError("Malformed learned-prior state payload.")
    try:
        prior_mu = np.asarray(state["prior_mu"], dtype=np.float64)
        prior_rho = np.asarray(state["prior_rho"], dtype=np.float64)
        logvar = np.asarray(state["effective_logvar"], dtype=np.float64)
        scores = np.asarray(
            diagnostics["standardized_active_unit_scores"], dtype=np.float64
        )
        active_mask = np.asarray(diagnostics["active_unit_mask"], dtype=bool)
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("Learned-prior numerical state is malformed.") from exc
    if (
        state.get("schema_version")
        != "midogpp_learned_conditional_diagonal_prior_v1"
        or int(state.get("n_classes", -1)) != 2
        or int(state.get("latent_dim", -1)) != int(latent_dim)
        or state.get("logvar_parameterization") != "6*tanh(rho/6)"
        or float(state.get("logvar_limit", float("nan"))) != 6.0
        or prior_mu.shape != (2, latent_dim)
        or prior_rho.shape != (2, latent_dim)
        or logvar.shape != (2, latent_dim)
        or scores.shape != (latent_dim,)
        or active_mask.shape != (latent_dim,)
        or not all(
            np.isfinite(value).all()
            for value in (prior_mu, prior_rho, logvar, scores)
        )
        or not np.allclose(logvar, 6.0 * np.tanh(prior_rho / 6.0))
        or np.max(np.abs(logvar)) > 6.0 + 1e-12
    ):
        raise ProtocolError("Learned-prior bounded state contract mismatch.")
    if (
        not isinstance(statistics, Mapping)
        or set(statistics) != {"0", "1"}
        or record.get("posterior_sufficient_statistics_hash")
        != stable_hash(statistics)
        or not isinstance(kl_audit, Mapping)
        or set(kl_audit) != {"0", "1"}
    ):
        raise ProtocolError("Learned-prior posterior audit coverage mismatch.")
    posterior_variances: list[object] = []
    for class_label in (0, 1):
        class_stats = statistics[str(class_label)]
        class_kl = kl_audit[str(class_label)]
        if not isinstance(class_stats, Mapping) or not isinstance(class_kl, Mapping):
            raise ProtocolError("Malformed learned-prior per-class audit state.")
        try:
            mu_mean = np.asarray(class_stats["posterior_mu_mean"], dtype=np.float64)
            mu_variance = np.asarray(
                class_stats["posterior_mu_variance"], dtype=np.float64
            )
            logvar_mean = np.asarray(
                class_stats["posterior_logvar_mean"], dtype=np.float64
            )
            posterior_variance_mean = np.asarray(
                class_stats["posterior_variance_mean"], dtype=np.float64
            )
            recorded_kl = np.asarray(
                class_kl["mean_kl_per_dimension"], dtype=np.float64
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("Malformed learned-prior sufficient statistics.") from exc
        if (
            int(class_stats.get("n_rows", 0)) <= 0
            or any(
                value.shape != (latent_dim,)
                or not np.isfinite(value).all()
                for value in (
                    mu_mean,
                    mu_variance,
                    logvar_mean,
                    posterior_variance_mean,
                    recorded_kl,
                )
            )
            or np.any(mu_variance < -1e-12)
            or np.any(posterior_variance_mean <= 0.0)
        ):
            raise ProtocolError("Invalid learned-prior sufficient statistics.")
        expected_kl = 0.5 * (
            logvar[class_label]
            - logvar_mean
            + posterior_variance_mean * np.exp(-logvar[class_label])
            + (
                mu_variance
                + (mu_mean - prior_mu[class_label]) ** 2
            )
            * np.exp(-logvar[class_label])
            - 1.0
        )
        if (
            not np.allclose(recorded_kl, expected_kl)
            or not np.isclose(
                float(class_kl.get("latent_normalized_mean_kl", float("nan"))),
                float(expected_kl.mean()),
            )
            or not np.isclose(
                float(class_kl.get("mean_kl_sum", float("nan"))),
                float(expected_kl.sum()),
            )
        ):
            raise ProtocolError("Learned-prior per-dimension KL audit mismatch.")
        posterior_variances.append(mu_variance)
    expected_scores = 0.5 * (
        np.asarray(posterior_variances[0]) * np.exp(-logvar[0])
        + np.asarray(posterior_variances[1]) * np.exp(-logvar[1])
    )
    expected_active = expected_scores > ACTIVE_UNIT_THRESHOLD
    expected_saturated = bool(
        np.any(np.abs(logvar) >= PRIOR_SATURATION_THRESHOLD)
    )
    forward = 0.5 * np.sum(
        logvar[1]
        - logvar[0]
        + np.exp(logvar[0] - logvar[1])
        + (prior_mu[0] - prior_mu[1]) ** 2 * np.exp(-logvar[1])
        - 1.0
    ) / float(latent_dim)
    reverse = 0.5 * np.sum(
        logvar[0]
        - logvar[1]
        + np.exp(logvar[1] - logvar[0])
        + (prior_mu[1] - prior_mu[0]) ** 2 * np.exp(-logvar[0])
        - 1.0
    ) / float(latent_dim)
    expected_symmetric_kl = 0.5 * (forward + reverse)
    symmetric_kl = float(diagnostics.get("normalized_symmetric_kl", float("nan")))
    if (
        diagnostics.get("schema_version")
        != "midogpp_learned_conditional_prior_diagnostics_v1"
        or diagnostics.get("finite") is not True
        or not np.isclose(symmetric_kl, expected_symmetric_kl)
        or not np.allclose(scores, expected_scores)
        or not np.array_equal(active_mask, expected_active)
        or int(diagnostics.get("active_unit_count", -1))
        != int(expected_active.sum())
        or diagnostics.get("saturated") is not expected_saturated
        or int(diagnostics.get("saturation_count", -1))
        != int(np.sum(np.abs(logvar) >= PRIOR_SATURATION_THRESHOLD))
        or not np.isclose(
            float(diagnostics.get("max_abs_logvar", float("nan"))),
            float(np.max(np.abs(logvar))),
        )
        or diagnostics.get("near_class_independent")
        is not bool(symmetric_kl <= CLASS_SEPARATION_THRESHOLD)
        or float(diagnostics.get("active_unit_threshold", float("nan")))
        != ACTIVE_UNIT_THRESHOLD
        or float(diagnostics.get("class_separation_threshold", float("nan")))
        != CLASS_SEPARATION_THRESHOLD
        or float(diagnostics.get("saturation_threshold", float("nan")))
        != PRIOR_SATURATION_THRESHOLD
    ):
        raise ProtocolError("Learned-prior diagnostic state is inconsistent.")
    expected_eligible = bool(not expected_saturated and expected_active.any())
    expected_separation = (
        "NO_REALIZED_CLASS_SEPARATION"
        if symmetric_kl <= CLASS_SEPARATION_THRESHOLD
        else "REALIZED_CLASS_SEPARATION"
    )
    if (
        record.get("integrity_valid") is not True
        or record.get("primary_preservation_eligible") is not expected_eligible
        or record.get("class_separation_status") != expected_separation
        or not isinstance(gaps, Mapping)
        or set(gaps) != {"0", "1"}
    ):
        raise ProtocolError("Learned-prior mechanism eligibility is inconsistent.")
    expected_partition_hash = stable_hash(
        {
            "prior_mu": state["prior_mu"],
            "prior_rho": state["prior_rho"],
            "effective_logvar": state["effective_logvar"],
        }
    )
    if record.get("final_prior_partition_hash") != expected_partition_hash:
        raise ProtocolError("Final learned-prior partition hash mismatch.")
    if (
        not isinstance(trajectory, list)
        or len(trajectory) != int(train_epochs)
        or record.get("prior_training_trajectory_hash") != stable_hash(trajectory)
    ):
        raise ProtocolError("Learned-prior training trajectory coverage mismatch.")
    transient_saturation = False
    for epoch, row in enumerate(trajectory, start=1):
        if not isinstance(row, Mapping) or int(row.get("epoch", -1)) != epoch:
            raise ProtocolError("Learned-prior trajectory epoch identity mismatch.")
        try:
            logvar_min = float(row["effective_logvar_min"])
            logvar_max = float(row["effective_logvar_max"])
            std_min = float(row["prior_std_min"])
            std_max = float(row["prior_std_max"])
            mu_min = float(row["prior_mu_min"])
            mu_max = float(row["prior_mu_max"])
            rho_min = float(row["prior_rho_min"])
            rho_max = float(row["prior_rho_max"])
            l2 = np.asarray(row["prior_mu_l2_by_class"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("Malformed learned-prior trajectory diagnostics.") from exc
        expected_epoch_saturated = bool(
            max(abs(logvar_min), abs(logvar_max))
            >= PRIOR_SATURATION_THRESHOLD
        )
        if (
            not all(
                np.isfinite(value)
                for value in (
                    logvar_min,
                    logvar_max,
                    std_min,
                    std_max,
                    mu_min,
                    mu_max,
                    rho_min,
                    rho_max,
                )
            )
            or l2.shape != (2,)
            or not np.isfinite(l2).all()
            or not 0.0 < std_min <= std_max
            or not np.isclose(std_min, np.exp(0.5 * logvar_min))
            or not np.isclose(std_max, np.exp(0.5 * logvar_max))
            or row.get("prior_saturated") is not expected_epoch_saturated
            or (int(row.get("prior_saturation_count", -1)) > 0)
            is not expected_epoch_saturated
        ):
            raise ProtocolError("Learned-prior trajectory diagnostics are inconsistent.")
        transient_saturation = transient_saturation or expected_epoch_saturated
    final = trajectory[-1]
    if (
        not np.isclose(float(final["effective_logvar_min"]), float(logvar.min()))
        or not np.isclose(float(final["effective_logvar_max"]), float(logvar.max()))
        or not np.isclose(float(final["prior_mu_min"]), float(prior_mu.min()))
        or not np.isclose(float(final["prior_mu_max"]), float(prior_mu.max()))
        or record.get("transient_saturation_observed") is not transient_saturation
    ):
        raise ProtocolError("Final learned-prior trajectory state mismatch.")
    for value in gaps.values():
        if not isinstance(value, Mapping) or any(
            not np.isfinite(float(value.get(field, float("nan"))))
            or float(value.get(field, -1.0)) < 0.0
            for field in ("mean_l2", "variance_l2")
        ):
            raise ProtocolError("Learned/ex-post prior gap diagnostic is invalid.")


def _validate_state_coverage(
    config: LearnedConditionalPriorStudyConfig,
    observed: set[tuple[str, str, int]],
) -> None:
    expected = {
        (str(outer), inner, int(seed))
        for outer in config.heldout_centers
        for inner in MIDOGPP_ELIGIBLE_CENTERS
        if inner != outer
        for seed in config.training_seeds
    }
    if observed != expected:
        raise ProtocolError("Learned-prior state index has incomplete H/I/seed coverage.")


def _validate_coverage_manifest(
    coverage: Mapping[str, object],
    *,
    config: LearnedConditionalPriorStudyConfig,
    metric_rows: list[dict[str, str]],
    n_prior_states: int,
) -> None:
    expected_cells = (
        len(config.heldout_centers)
        * (len(MIDOGPP_ELIGIBLE_CENTERS) - 1)
        * len(config.training_seeds)
        * len(config.generation_seeds)
        * len(config.arms)
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
        or int(coverage.get("learned_prior_states", -1)) != n_prior_states
        or coverage.get("complete_training_generation_seed_cross") is not True
    ):
        raise ProtocolError("Learned-prior coverage manifest is not recomputable.")


def _validate_sampler_rows(
    rows: list[dict[str, str]],
    *,
    config: LearnedConditionalPriorStudyConfig,
    prior_states: Mapping[tuple[str, str, int], Mapping[str, object]],
    checkpoint_index: Mapping[str, object],
) -> dict[tuple[str, str, int, str], Mapping[str, object]]:
    records = checkpoint_index.get("records")
    if not isinstance(records, list):
        raise ProtocolError("Malformed learned-prior checkpoint records.")
    grouped: dict[tuple[str, str, int, str], list[Mapping[str, str]]] = {}
    for row in rows:
        key = (
            row.get("outer_target_center", ""),
            row.get("inner_pseudo_target_center", ""),
            int(row.get("training_seed", -1)),
            row.get("arm", ""),
        )
        grouped.setdefault(key, []).append(row)
    expected = {
        (*key, arm)
        for key in prior_states
        for arm in ("C-diag", "E")
    }
    if set(grouped) != expected:
        raise ProtocolError("Learned-prior sampler table coverage mismatch.")
    output: dict[tuple[str, str, int, str], Mapping[str, object]] = {}
    for key, class_rows in grouped.items():
        if (
            len(class_rows) != 2
            or {int(row.get("class_label", -1)) for row in class_rows} != {0, 1}
            or any(row.get("schema_version") != PRIOR_SAMPLER_SCHEMA for row in class_rows)
        ):
            raise ProtocolError("Sampler realization must contain exactly both classes.")
        outer, inner, seed, arm = key
        prior_record = prior_states[(outer, inner, seed)]
        by_class = {int(row["class_label"]): row for row in class_rows}
        if arm == "C-diag":
            first = by_class[0]
            if any(
                row.get("mechanism") != "ex_post_aggregate_posterior_diagonal"
                or row.get("requested_family") != config.ex_post_prior_family
                or int(row.get("latent_dim", -1)) != config.latent_dim
                or row.get("source_row_hash") != prior_record.get("source_row_hash")
                for row in class_rows
            ):
                raise ProtocolError("C-diag sampler identity mismatch.")
            classes: dict[str, object] = {}
            for class_label, row in by_class.items():
                try:
                    classes[str(class_label)] = {
                        "class_label": class_label,
                        "requested_family": row["requested_family"],
                        "realized_family": row["realized_family"],
                        "mean": json.loads(row["mean"]),
                        "covariance": json.loads(row["covariance"]),
                        "n_rows": int(row["n_rows"]),
                        "raw_between_covariance": json.loads(
                            row["raw_between_covariance"]
                        ),
                        "within_posterior_diagonal": json.loads(
                            row["within_posterior_diagonal"]
                        ),
                        "shrinkage": (
                            None if row.get("shrinkage", "") == "" else float(row["shrinkage"])
                        ),
                        "shrinkage_target": (
                            None
                            if row.get("shrinkage_target", "") == ""
                            else float(row["shrinkage_target"])
                        ),
                        "jitter": float(row["jitter"]),
                        "condition_number": float(row["condition_number"]),
                        "eigenvalues": json.loads(row["eigenvalues"]),
                        "fallback_reason": row.get("fallback_reason", ""),
                    }
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ProtocolError("Malformed C-diag sampler state.") from exc
            expected_hash = stable_hash(
                {
                    "requested_family": config.ex_post_prior_family,
                    "latent_dim": config.latent_dim,
                    "source_row_hash": first["source_row_hash"],
                    "classes": classes,
                }
            )
            if any(row.get("sampler_state_hash") != expected_hash for row in class_rows):
                raise ProtocolError("C-diag sampler-state hash mismatch.")
            matching = [
                record
                for record in records
                if isinstance(record, Mapping)
                and record.get("model_family") == "class_conditioned_cvae_v1"
                and isinstance(record.get("training_key"), Mapping)
                and record["training_key"].get("outer_target_center") == outer
                and record["training_key"].get("inner_pseudo_target_center") == inner
                and int(record["training_key"].get("training_seed", -1)) == seed
            ]
            if len(matching) != 1 or any(
                row.get("training_key_hash") != matching[0].get("training_key_hash")
                or row.get("checkpoint_hash") != matching[0].get("checkpoint_hash")
                for row in class_rows
            ):
                raise ProtocolError("C-diag sampler/checkpoint binding mismatch.")
            viable = all(
                row.get("realized_family") == config.ex_post_prior_family
                for row in class_rows
            )
            output[key] = {"sampler_state_hash": expected_hash, "eligible": viable}
        else:
            state = prior_record.get("state")
            if not isinstance(state, Mapping):
                raise ProtocolError("Malformed learned-prior state for sampler binding.")
            for class_label, row in by_class.items():
                try:
                    mean = json.loads(row["mean"])
                    logvar = json.loads(row["logvar"])
                    variance = json.loads(row["variance"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ProtocolError("Malformed E sampler state.") from exc
                if (
                    row.get("mechanism")
                    != "jointly_learned_class_conditional_diagonal_prior"
                    or row.get("requested_family") != config.learned_prior_family
                    or row.get("realized_family") != config.learned_prior_family
                    or int(row.get("latent_dim", -1)) != config.latent_dim
                    or row.get("source_row_hash") != prior_record.get("source_row_hash")
                    or row.get("training_key_hash") != prior_record.get("training_key_hash")
                    or row.get("checkpoint_hash") != prior_record.get("checkpoint_hash")
                    or row.get("sampler_state_hash") != prior_record.get("state_hash")
                    or mean != state["prior_mu"][class_label]
                    or logvar != state["effective_logvar"][class_label]
                    or len(variance) != config.latent_dim
                    or any(
                        abs(float(observed) - math.exp(float(value))) > 1e-12
                        for observed, value in zip(variance, logvar)
                    )
                    or row.get("fallback_reason", "") != ""
                ):
                    raise ProtocolError("E sampler/state binding mismatch.")
            output[key] = {
                "sampler_state_hash": prior_record["state_hash"],
                "eligible": prior_record["primary_preservation_eligible"],
            }
    return output


def _validate_prior_metric_semantics(
    rows: list[dict[str, str]],
    config: LearnedConditionalPriorStudyConfig,
    *,
    prior_states: Mapping[tuple[str, str, int], Mapping[str, object]],
    sampler_states: Mapping[tuple[str, str, int, str], Mapping[str, object]],
) -> None:
    expected = {
        "A": ("class_conditioned_cvae_v1", config.standard_prior_family),
        "C-diag": ("class_conditioned_cvae_v1", config.ex_post_prior_family),
        "E": (LEARNED_PRIOR_MODEL_FAMILY, config.learned_prior_family),
    }
    for row in rows:
        arm = row.get("arm", "")
        if arm not in expected or (
            row.get("model_family"), row.get("prior_family")
        ) != expected[arm]:
            raise ProtocolError("Learned-prior metric arm identity mismatch.")
        eligible = row.get("eligible") == "true"
        reason = row.get("ineligibility_reason", "")
        fold = (
            row.get("outer_target_center", ""),
            row.get("inner_pseudo_target_center", ""),
            int(row.get("training_seed", -1)),
        )
        if arm == "A" and (
            not eligible
            or reason
            or row.get("sampler_state_hash")
            != stable_hash(
                {"family": config.standard_prior_family, "latent_dim": config.latent_dim}
            )
        ):
            raise ProtocolError("Matched baseline A must remain mechanism-eligible.")
        if arm == "C-diag" and (
            sampler_states.get((*fold, arm), {}).get("eligible") is not eligible
            or row.get("sampler_state_hash")
            != sampler_states.get((*fold, arm), {}).get("sampler_state_hash")
            or (reason != "" if eligible else reason != "conditional_sampler_not_realized")
        ):
            raise ProtocolError("C-diag eligibility reason is inconsistent.")
        if arm == "E" and (
            fold not in prior_states
            or prior_states[fold].get("primary_preservation_eligible") is not eligible
            or row.get("sampler_state_hash") != prior_states[fold].get("state_hash")
            or (reason != "" if eligible else reason != "learned_prior_mechanism_ineligible")
        ):
            raise ProtocolError("Learned-prior eligibility reason is inconsistent.")


def _validate_pairing_audits(
    config: LearnedConditionalPriorStudyConfig,
    *,
    checkpoint_rows: list[dict[str, str]],
    initialization_rows: list[dict[str, str]],
    checkpoint_index: Mapping[str, object],
) -> None:
    expected = {
        (str(outer), inner, int(seed))
        for outer in config.heldout_centers
        for inner in MIDOGPP_ELIGIBLE_CENTERS
        if inner != outer
        for seed in config.training_seeds
    }
    records = checkpoint_index.get("records")
    if not isinstance(records, list):
        raise ProtocolError("Malformed learned-prior checkpoint records.")
    by_key = {
        str(record.get("training_key_hash", "")): record
        for record in records
        if isinstance(record, Mapping)
    }
    observed_checkpoint: set[tuple[str, str, int]] = set()
    for row in checkpoint_rows:
        key = (
            row.get("outer_target_center", ""),
            row.get("inner_pseudo_target_center", ""),
            int(row.get("training_seed", -1)),
        )
        a_key = row.get("a_training_key_hash", "")
        c_key = row.get("c_training_key_hash", "")
        record = by_key.get(a_key)
        record_key = record.get("training_key") if isinstance(record, Mapping) else None
        if (
            key in observed_checkpoint
            or row.get("arm_pair") != "A/C-diag"
            or a_key != c_key
            or row.get("a_checkpoint_hash") != row.get("c_checkpoint_hash")
            or row.get("single_checkpoint_reused") != "true"
            or row.get("status") != "PASS"
            or not isinstance(record, Mapping)
            or record.get("checkpoint_hash") != row.get("a_checkpoint_hash")
            or record.get("model_family") != "class_conditioned_cvae_v1"
            or not isinstance(record_key, Mapping)
            or (
                record_key.get("outer_target_center"),
                record_key.get("inner_pseudo_target_center"),
                int(record_key.get("training_seed", -1)),
            )
            != key
        ):
            raise ProtocolError("A/C-diag checkpoint-reuse audit mismatch.")
        observed_checkpoint.add(key)
    if observed_checkpoint != expected:
        raise ProtocolError("A/C-diag checkpoint audit coverage mismatch.")
    observed_initialization: set[tuple[str, str, int]] = set()
    for row in initialization_rows:
        key = (
            row.get("outer_target_center", ""),
            row.get("inner_pseudo_target_center", ""),
            int(row.get("training_seed", -1)),
        )
        matching = [
            record
            for record in records
            if isinstance(record, Mapping)
            and isinstance(record.get("training_key"), Mapping)
            and record["training_key"].get("outer_target_center") == key[0]
            and record["training_key"].get("inner_pseudo_target_center") == key[1]
            and int(record["training_key"].get("training_seed", -1)) == key[2]
        ]
        a_records = [
            record
            for record in matching
            if record.get("model_family") == "class_conditioned_cvae_v1"
        ]
        e_records = [
            record
            for record in matching
            if record.get("model_family") == LEARNED_PRIOR_MODEL_FAMILY
        ]
        if (
            key in observed_initialization
            or len(a_records) != 1
            or len(e_records) != 1
            or row.get("arm_pair") != "A/E"
            or row.get("shared_initialization_equal") != "true"
            or row.get("training_stream_equal") != "true"
            or row.get("full_training_identity_distinct") != "true"
            or row.get("shared_initialization_hash_a")
            != row.get("shared_initialization_hash_e")
            or row.get("training_stream_hash_a")
            != row.get("training_stream_hash_e")
            or row.get("full_initialization_hash_a")
            == row.get("full_initialization_hash_e")
            or row.get("shared_initialization_hash_a")
            != a_records[0].get("shared_initialization_hash")
            or row.get("shared_initialization_hash_e")
            != e_records[0].get("shared_initialization_hash")
            or row.get("training_stream_hash_a")
            != a_records[0].get("training_stream_hash")
            or row.get("training_stream_hash_e")
            != e_records[0].get("training_stream_hash")
            or row.get("full_initialization_hash_a")
            != a_records[0].get("full_initialization_hash")
            or row.get("full_initialization_hash_e")
            != e_records[0].get("full_initialization_hash")
            or row.get("status") != "PASS"
        ):
            raise ProtocolError("A/E initialization or stochastic pairing mismatch.")
        observed_initialization.add(key)
    if observed_initialization != expected:
        raise ProtocolError("A/E initialization audit coverage mismatch.")


def _decision_metrics(rows: list[dict[str, str]]) -> list[PriorStudyMetricV2]:
    decode = {
        (row["outer_target_center"], row["inner_pseudo_target_center"], int(row["training_seed"]), row["arm"]): row
        for row in rows
        if row["representation_role"] == "decode"
    }
    posterior = {
        (row["outer_target_center"], row["inner_pseudo_target_center"], int(row["training_seed"]), int(row["generation_seed"]), row["arm"]): row
        for row in rows
        if row["representation_role"] == "posterior"
    }
    output: list[PriorStudyMetricV2] = []
    for row in rows:
        if row["representation_role"] != "prior":
            continue
        base = (row["outer_target_center"], row["inner_pseudo_target_center"], int(row["training_seed"]), row["arm"])
        post = posterior[base[:3] + (int(row["generation_seed"]), base[3])]
        dec = decode[base]
        output.append(
            PriorStudyMetricV2(
                outer_target_center=base[0],
                inner_pseudo_target_center=base[1],
                training_seed=base[2],
                generation_seed=int(row["generation_seed"]),
                arm=base[3],
                preservation_ratio=float(row["preservation_ratio"]),
                decode_bacc=float(dec["bacc"]),
                posterior_bacc=float(post["bacc"]),
                valid=(row["valid"] == "true" and post["valid"] == "true" and dec["valid"] == "true"),
                eligible=row.get("eligible") == "true",
                ineligibility_reason=row.get("ineligibility_reason", ""),
            )
        )
    return output


def _validate_metric_checkpoint_references(rows: list[dict[str, str]], index: Mapping[str, object]) -> None:
    records = index.get("records")
    if not isinstance(records, list):
        raise ProtocolError("Malformed study checkpoint index records.")
    by_key = {str(record["training_key_hash"]): record for record in records if isinstance(record, Mapping)}
    for row in rows:
        key = row.get("training_key_hash", "")
        if key == "none":
            raise ProtocolError("Prior-study metric row lacks its runtime checkpoint.")
        record = by_key.get(key)
        training_key = record.get("training_key") if isinstance(record, Mapping) else None
        try:
            fit_centers = json.loads(row["fit_centers"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ProtocolError("Metric/checkpoint fit-center identity is malformed.") from exc
        expected_checkpoint_prior = (
            "standard_normal"
            if row.get("arm") == "C-diag"
            else row.get("prior_family")
        )
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
            or training_key.get("prior_family") != expected_checkpoint_prior
            or training_key.get("objective_id") != row.get("objective_id")
            or float(training_key.get("alpha", float("nan")))
            != float(row.get("alpha", "nan"))
        ):
            raise ProtocolError("Study metric references an unpersisted checkpoint.")
    metric_keys = {
        row.get("training_key_hash", "")
        for row in rows
        if row.get("training_key_hash") != "none"
    }
    if metric_keys != set(by_key):
        raise ProtocolError("Study checkpoint index contains unused or missing training keys.")


def _validate_nonconsumable(root: Path, config: LearnedConditionalPriorStudyConfig) -> None:
    forbidden_fields = {"recipe_export_ready", "stage30_recipe_ready", "primary_arm"}
    if (root / "reports/publication_state.json").exists():
        raise ProtocolError("Non-adoptive study emitted a publication state.")
    for outer in config.heldout_centers:
        payload = read_json(root / f"reports/consensus_decisions/{outer}.json")
        if forbidden_fields.intersection(payload) or payload.get("may_feed_model_recipe") is not False or payload.get("may_feed_deployable_selection") is not False:
            raise ProtocolError("Study decision exposes recipe-consumption semantics.")
