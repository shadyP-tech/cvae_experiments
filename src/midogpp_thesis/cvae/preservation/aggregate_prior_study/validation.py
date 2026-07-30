"""Independent fail-closed validation for completed v3 study bundles."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from ....real_features.classifier_reference.artifacts import stable_hash
from ....real_features.classifier_reference.protocol import ProtocolError
from ..prior_recovery_runtime_cache import validate_feature_frame_index
from ..splits import frame_arrays, indices_for_centers
from .checkpoint_store import validate_checkpoint_index
from .config import AggregatePriorStudyConfig, load_aggregate_prior_study_config
from .contracts import (
    ARMS,
    CLAIM_SCOPE,
    EXPERIMENT_ID,
    PRIMARY_ARM,
    SourceExpertEvaluationKey,
    SourceExpertTrainingKey,
    objective_family,
    prior_family,
    rate_family,
)
from .preparation import load_frame
from .runner import (
    COVERAGE_SCHEMA,
    METRIC_SCHEMA,
    PROTOCOL_SCHEMA,
    PUBLICATION_SCHEMA,
    SELECTION_SCHEMA,
    _canonical_rows,
    _balanced_source_indices,
    _coverage_manifest,
    _decisions,
    _geco_state_index_payload,
    _protocol_manifest,
    _study_decision,
)
from .training import paired_generation_noise


STATIC_FILES = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "manifests/protocol_manifest.json",
    "manifests/coverage_manifest.json",
    "manifests/selection_evidence_manifest.json",
    "manifests/source_expert_checkpoint_index.json",
    "manifests/feature_frame_index.json",
    "manifests/mixture_prior_state_index.json",
    "manifests/geco_state_index.json",
    "manifests/generation_budget_manifest.json",
    "reports/study_decision.json",
    "reports/expert_isolation_report.json",
    "reports/leakage_report.json",
    "reports/publication_state.json",
    "reports/runtime_summary.json",
    "reports/run_state.json",
    "tables/source_expert_metrics.csv",
    "tables/paired_deltas.csv",
    "tables/source_local_real_references.csv",
    "tables/nested_classifier_tuning.csv",
    "tables/source_expert_training_audit.csv",
    "tables/mixture_prior_diagnostics.csv",
    "tables/geco_trajectory.csv",
    "tables/training_epochs.csv",
    "tables/generation_budget_audit.csv",
    "tables/rng_pairing_audit.csv",
    "tables/identity_overlap_audit.csv",
    "tables/runtime_timings.csv",
)


def validate_aggregate_prior_study_bundle(
    root: Path,
    *,
    expected_config: AggregatePriorStudyConfig | None = None,
) -> Mapping[str, object]:
    root = Path(root)
    _require_files(root, STATIC_FILES)
    config = expected_config or load_aggregate_prior_study_config(
        root / "config.resolved.yaml"
    )
    frame = load_frame(config)
    protocol = _read_json(root / "manifests/protocol_manifest.json")
    expected_protocol = _protocol_manifest(config, frame=frame)
    if (
        protocol != expected_protocol
        or protocol.get("schema_version") != PROTOCOL_SCHEMA
        or protocol.get("experiment_id") != EXPERIMENT_ID
        or protocol.get("claim_scope") != CLAIM_SCOPE
    ):
        raise ProtocolError("Aggregate-prior protocol manifest mismatch.")
    protocol_hash = str(protocol["protocol_hash"])

    run_state = _read_json(root / "reports/run_state.json")
    leakage = _read_json(root / "reports/leakage_report.json")
    publication = _read_json(root / "reports/publication_state.json")
    isolation_report = _read_json(root / "reports/expert_isolation_report.json")
    if (
        run_state.get("status") != "COMPLETE"
        or run_state.get("protocol_hash") != protocol_hash
        or leakage.get("status") != "PASS"
        or leakage.get("protocol_hash") != protocol_hash
        or leakage.get("outer_target_rows_used") is not False
        or leakage.get("outer_target_labels_used") is not False
        or leakage.get("inner_rows_used_for_fit") is not False
        or leakage.get("inner_labels_used_for_scoring_only") is not True
        or leakage.get("source_experts_independently_trained") is not True
        or leakage.get("target_or_inner_data_used_for_geco_target") is not False
        or leakage.get("target_or_inner_data_used_for_mixture_fit") is not False
        or isolation_report.get("status") != "PASS"
    ):
        raise ProtocolError("Aggregate-prior leakage/isolation gate failed.")
    if (
        publication.get("schema_version") != PUBLICATION_SCHEMA
        or publication.get("status") != "NON_CONSUMABLE_STUDY_COMPLETE"
        or publication.get("claim_scope") != CLAIM_SCOPE
        or publication.get("may_feed_model_recipe") is not False
        or publication.get("may_feed_expert_bank") is not False
        or publication.get("stage30_recipe_ready") is not False
        or publication.get("separate_promotion_artifact_required") is not True
    ):
        raise ProtocolError("V3 study became consumable without a promotion artifact.")

    checkpoint_index = validate_checkpoint_index(root, config=config)
    validate_feature_frame_index(root)
    metrics = _read_csv(root / "tables/source_expert_metrics.csv")
    deltas = _read_csv(root / "tables/paired_deltas.csv")
    references = _read_csv(root / "tables/source_local_real_references.csv")
    tuning = _read_csv(root / "tables/nested_classifier_tuning.csv")
    training = _read_csv(root / "tables/source_expert_training_audit.csv")
    isolation = _read_csv(root / "tables/identity_overlap_audit.csv")
    mixture = _read_csv(root / "tables/mixture_prior_diagnostics.csv")
    geco = _read_csv(root / "tables/geco_trajectory.csv")
    epochs = _read_csv(root / "tables/training_epochs.csv")
    budgets = _read_csv(root / "tables/generation_budget_audit.csv")
    rng = _read_csv(root / "tables/rng_pairing_audit.csv")
    source_labels = {
        center: frame_arrays(
            frame,
            indices_for_centers(frame, (center,)),
        )[1]
        for center in config.heldout_centers
    }
    _validate_training_rows(
        config,
        training,
        protocol_hash=protocol_hash,
        checkpoint_index=checkpoint_index,
    )
    _validate_metric_rows(
        config,
        metrics,
        training_rows=training,
        protocol_hash=str(protocol["protocol_hash"]),
        source_labels=source_labels,
    )
    _validate_isolation_rows(config, isolation)
    _validate_mixture_rows(config, mixture)
    _validate_geco_rows(geco)
    _validate_generation_rows(config, budgets, rng)

    observed_coverage = _read_json(root / "manifests/coverage_manifest.json")
    expected_coverage = _coverage_manifest(
        config,
        metric_rows=metrics,
        training_rows=training,
        isolation_rows=isolation,
    )
    if (
        observed_coverage.get("schema_version") != COVERAGE_SCHEMA
        or observed_coverage != expected_coverage
        or observed_coverage.get("status") != "PASS"
    ):
        raise ProtocolError("Aggregate-prior coverage is incomplete.")
    mixture_index = _read_json(
        root / "manifests/mixture_prior_state_index.json"
    )
    if (
        mixture_index.get("schema_version")
        != "midogpp_mixture_prior_state_index_v3"
        or int(mixture_index.get("n_refit_records", -1)) != len(mixture)
        or _canonical_rows(mixture_index.get("records", []))  # type: ignore[arg-type]
        != _canonical_rows(mixture)
    ):
        raise ProtocolError("Mixture-prior state index mismatch.")
    geco_index = _read_json(root / "manifests/geco_state_index.json")
    expected_geco_index = _geco_state_index_payload(
        geco_rows=geco,
        checkpoint_records=checkpoint_index["records"],  # type: ignore[arg-type]
    )
    if geco_index != expected_geco_index:
        raise ProtocolError("GECO controller-state index mismatch.")
    budget_manifest = _read_json(
        root / "manifests/generation_budget_manifest.json"
    )
    if (
        budget_manifest.get("policy") != "fixed_balanced_per_source_per_class"
        or int(budget_manifest.get("per_source_per_class", -1))
        != config.generation_per_class
        or budget_manifest.get("inner_or_outer_prevalence_used") is not False
        or int(budget_manifest.get("n_rows", -1)) != len(budgets)
    ):
        raise ProtocolError("Generation-budget manifest mismatch.")

    child, consensus, expected_deltas = _decisions(
        config,
        metric_rows=metrics,
        protocol_hash=protocol_hash,
    )
    if _canonical_rows(deltas) != _canonical_rows(expected_deltas):
        raise ProtocolError("Paired decision deltas are not recomputable.")
    for (seed, outer), expected in child.items():
        if _read_json(
            root / f"reports/child_decisions/seed{seed}/{outer}.json"
        ) != expected:
            raise ProtocolError("Child decision is not recomputable.")
    for outer, expected in consensus.items():
        if _read_json(
            root / f"reports/consensus_decisions/{outer}.json"
        ) != expected:
            raise ProtocolError("Consensus decision is not recomputable.")

    selection_hash = stable_hash(
        {
            "protocol_hash": protocol_hash,
            "metric_rows": _canonical_rows(metrics),
            "delta_rows": _canonical_rows(deltas),
            "training_rows": _canonical_rows(training),
            "isolation_rows": _canonical_rows(isolation),
            "source_local_real_reference_rows": _canonical_rows(references),
            "nested_classifier_tuning_rows": _canonical_rows(tuning),
            "training_epoch_rows": _canonical_rows(epochs),
            "generation_budget_rows": _canonical_rows(budgets),
            "rng_pairing_rows": _canonical_rows(rng),
            "coverage": observed_coverage,
            "checkpoint_index_path": (
                "manifests/source_expert_checkpoint_index.json"
            ),
            "checkpoint_index_hash": stable_hash(checkpoint_index),
            "mixture_index_hash": stable_hash(mixture_index),
            "geco_index_hash": stable_hash(geco_index),
            "generation_budget_manifest": budget_manifest,
        }
    )
    selection = _read_json(
        root / "manifests/selection_evidence_manifest.json"
    )
    if (
        selection.get("schema_version") != SELECTION_SCHEMA
        or selection.get("selection_evidence_hash") != selection_hash
        or selection.get("decisions_may_feed_model_recipe") is not False
        or selection.get("may_feed_deployable_selection") is not False
        or selection.get("source_local_axis_complete") is not True
    ):
        raise ProtocolError("Selection-evidence manifest mismatch.")
    if publication.get("selection_evidence_hash") != selection_hash:
        raise ProtocolError("Publication state is not evidence-bound.")
    expected_study = _study_decision(
        consensus,
        protocol_hash=protocol_hash,
        selection_evidence_hash=selection_hash,
    )
    if _read_json(root / "reports/study_decision.json") != expected_study:
        raise ProtocolError("Study decision is not recomputable.")
    if run_state.get("selection_evidence_hash") != selection_hash:
        raise ProtocolError("Run state is not evidence-bound.")
    return {
        "status": "PASS",
        "protocol_hash": protocol_hash,
        "selection_evidence_hash": selection_hash,
        "n_checkpoints": checkpoint_index["n_checkpoints"],
        "n_metric_rows": len(metrics),
        "claim_scope": CLAIM_SCOPE,
        "may_feed_model_recipe": False,
    }


def _validate_training_rows(
    config: AggregatePriorStudyConfig,
    rows: Sequence[Mapping[str, str]],
    *,
    protocol_hash: str,
    checkpoint_index: Mapping[str, object],
) -> None:
    expected = {
        (source, str(seed), arm)
        for source in config.heldout_centers
        for seed in config.training_seeds
        for arm in ARMS
    }
    raw_checkpoint_records = checkpoint_index.get("records")
    if not isinstance(raw_checkpoint_records, list):
        raise ProtocolError("Checkpoint index records are unavailable.")
    checkpoint_records = {
        str(record.get("training_key_hash", "")): record
        for record in raw_checkpoint_records
        if isinstance(record, Mapping)
    }
    observed: set[tuple[str, str, str]] = set()
    paired_hashes: dict[tuple[str, str], dict[str, set[str]]] = {}
    for row in rows:
        key = (
            str(row.get("source_center", "")),
            str(row.get("training_seed", "")),
            str(row.get("arm", "")),
        )
        try:
            training_key = SourceExpertTrainingKey(
                source_center=key[0],
                training_seed=int(key[1]),
                arm=key[2],
                source_row_hash=str(row.get("source_row_hash", "")),
                source_case_hash=str(row.get("source_case_hash", "")),
                source_frame_hash=str(row.get("source_frame_hash", "")),
                manifest_hash=str(row.get("manifest_hash", "")),
                feature_cache_hash=str(row.get("feature_cache_hash", "")),
                protocol_hash=str(row.get("protocol_hash", "")),
                config_hash=str(row.get("config_hash", "")),
            )
        except (ProtocolError, TypeError, ValueError) as exc:
            raise ProtocolError(
                "Source-expert training row has an invalid content key."
            ) from exc
        checkpoint_record = checkpoint_records.get(training_key.hash)
        if (
            key in observed
            or json.loads(row.get("fit_centers", "[]")) != [key[0]]
            or row.get("source_only") != "True"
            or row.get("outer_or_inner_identity_in_key") != "False"
            or not row.get("source_row_hash")
            or not row.get("source_case_hash")
            or not row.get("source_frame_hash")
            or row.get("protocol_hash") != protocol_hash
            or row.get("config_hash") != config.contract_hash
            or row.get("training_key_hash") != training_key.hash
            or checkpoint_record is None
            or row.get("checkpoint_hash")
            != str(checkpoint_record.get("checkpoint_hash", ""))
            or row.get("warmup_checkpoint_hash")
            != str(checkpoint_record.get("warmup_checkpoint_hash", ""))
            or row.get("shared_initialization_hash")
            != str(checkpoint_record.get("shared_initialization_hash", ""))
            or row.get("training_stream_hash")
            != str(checkpoint_record.get("training_stream_hash", ""))
        ):
            raise ProtocolError("Source-expert training audit is not isolated.")
        observed.add(key)
        group = paired_hashes.setdefault(
            (key[0], key[1]),
            {
                "warmup_checkpoint_hash": set(),
                "shared_initialization_hash": set(),
                "training_stream_hash": set(),
            },
        )
        for field in group:
            value = str(row.get(field, ""))
            if not value:
                raise ProtocolError(f"Training audit lacks {field}.")
            group[field].add(value)
    if observed != expected or any(
        len(values) != 1
        for group in paired_hashes.values()
        for values in group.values()
    ):
        raise ProtocolError("Source-expert training grid/pairing mismatch.")
    if set(checkpoint_records) != {
        str(row["training_key_hash"]) for row in rows
    }:
        raise ProtocolError(
            "Training audit and checkpoint index cover different experts."
        )


def _validate_metric_rows(
    config: AggregatePriorStudyConfig,
    rows: Sequence[Mapping[str, str]],
    *,
    training_rows: Sequence[Mapping[str, str]],
    protocol_hash: str,
    source_labels: Mapping[str, Sequence[int]],
) -> None:
    training_keys = {
        str(row["training_key_hash"]): row for row in training_rows
    }
    observed: set[tuple[str, ...]] = set()
    generation_labels = tuple(
        [0] * config.generation_per_class
        + [1] * config.generation_per_class
    )
    pairing_cache: dict[tuple[str, str, str, int], tuple[str, str]] = {}
    for row in rows:
        outer = str(row.get("outer_target_center", ""))
        inner = str(row.get("inner_pseudo_target_center", ""))
        source = str(row.get("source_center", ""))
        arm = str(row.get("arm", ""))
        key = (
            outer,
            inner,
            source,
            str(row.get("training_seed", "")),
            str(row.get("generation_seed", "")),
            arm,
            str(row.get("representation_role", "")),
        )
        training = training_keys.get(str(row.get("training_key_hash", "")))
        pairing_key = (
            outer,
            inner,
            source,
            int(row.get("generation_seed", -1)),
        )
        if pairing_key not in pairing_cache:
            neutral_evaluation_hash = stable_hash(
                {
                    "protocol_hash": protocol_hash,
                    "outer_target_center": outer,
                    "inner_pseudo_target_center": inner,
                    "source_center": source,
                    "generation_seed": pairing_key[3],
                    "generation_labels": generation_labels,
                }
            )
            _, _, expected_noise_hash = paired_generation_noise(
                neutral_evaluation_hash=neutral_evaluation_hash,
                labels=generation_labels,
                latent_dim=config.latent_dim,
            )
            _, expected_posterior_hash = _balanced_source_indices(
                source_labels.get(source, ()),
                per_class=config.generation_per_class,
                neutral_evaluation_hash=neutral_evaluation_hash,
            )
            pairing_cache[pairing_key] = (
                expected_noise_hash,
                expected_posterior_hash,
            )
        expected_noise_hash, expected_posterior_hash = pairing_cache[pairing_key]
        try:
            evaluation_key = SourceExpertEvaluationKey(
                outer_target_center=outer,
                inner_pseudo_target_center=inner,
                source_center=source,
                training_seed=int(row.get("training_seed", -1)),
                generation_seed=int(row.get("generation_seed", -1)),
                arm=arm,
                representation_role=str(
                    row.get("representation_role", "")
                ),
                generation_noise_hash=str(row.get("noise_hash", "")),
                posterior_source_index_hash=(
                    str(row.get("posterior_source_index_hash", ""))
                    if row.get("representation_role") == "posterior"
                    else None
                ),
                training_key_hash=str(row.get("training_key_hash", "")),
                inner_eval_row_hash=str(row.get("inner_eval_row_hash", "")),
                classifier_spec_hash=str(
                    row.get("classifier_spec_hash", "")
                ),
                protocol_hash=protocol_hash,
            )
        except (ProtocolError, TypeError, ValueError) as exc:
            raise ProtocolError(
                "Metric row has an invalid evaluation content key."
            ) from exc
        if (
            row.get("schema_version") != METRIC_SCHEMA
            or key in observed
            or len({outer, inner, source}) != 3
            or outer not in config.heldout_centers
            or inner not in config.heldout_centers
            or source not in config.heldout_centers
            or arm not in ARMS
            or row.get("prior_family") != prior_family(arm)
            or row.get("objective_family") != objective_family(arm)
            or row.get("rate_family") != rate_family(arm)
            or row.get("rate_is_exact_nelbo") != "False"
            or row.get("representation_role") not in {"prior", "posterior"}
            or row.get("inverse_transformed_to_common_frame") != "True"
            or int(row.get("generated_output_dim", -1))
            != config.expected_feature_dim
            or row.get("valid") != "true"
            or row.get("eligible") != "true"
            or row.get("claim_scope") != CLAIM_SCOPE
            or row.get("outer_target_rows_used") != "false"
            or row.get("inner_rows_used_for_fit") != "false"
            or row.get("inner_labels_used_for_scoring_only") != "true"
            or row.get("may_feed_model_recipe") != "false"
            or row.get("noise_hash") != expected_noise_hash
            or row.get("posterior_source_index_hash")
            != expected_posterior_hash
            or row.get("protocol_hash") != protocol_hash
            or row.get("evaluation_key_hash") != evaluation_key.hash
            or training is None
            or training.get("source_center") != source
            or training.get("arm") != arm
        ):
            raise ProtocolError("Malformed or leaky source-expert metric row.")
        for metric in ("bacc", "macro_f1"):
            value = float(row[metric])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ProtocolError("Metric row contains an invalid score.")
        observed.add(key)
    expected_count = (
        len(config.heldout_centers)
        * (len(config.heldout_centers) - 1)
        * (len(config.heldout_centers) - 2)
        * len(config.training_seeds)
        * len(config.generation_seeds)
        * len(ARMS)
        * 2
    )
    if len(observed) != expected_count:
        raise ProtocolError("Metric grid is incomplete.")


def _validate_isolation_rows(
    config: AggregatePriorStudyConfig,
    rows: Sequence[Mapping[str, str]],
) -> None:
    expected = {
        (outer, inner, source)
        for outer in config.heldout_centers
        for inner in config.heldout_centers
        if inner != outer
        for source in config.heldout_centers
        if source not in {outer, inner}
    }
    observed = set()
    for row in rows:
        key = (
            str(row.get("outer_target_center", "")),
            str(row.get("inner_pseudo_target_center", "")),
            str(row.get("source_center", "")),
        )
        if (
            key in observed
            or row.get("status") != "PASS"
            or row.get("outer_center_absent_from_fit") != "True"
            or row.get("inner_center_absent_from_fit") != "True"
            or any(
                int(row.get(field, -1)) != 0
                for field in (
                    "sample_overlap_count",
                    "case_overlap_count",
                    "image_overlap_count",
                )
            )
            or json.loads(row.get("fit_centers", "[]")) != [key[2]]
        ):
            raise ProtocolError("Expert-isolation row failed.")
        observed.add(key)
    if observed != expected:
        raise ProtocolError("Expert-isolation grid is incomplete.")


def _validate_mixture_rows(
    config: AggregatePriorStudyConfig,
    rows: Sequence[Mapping[str, str]],
) -> None:
    if not rows:
        raise ProtocolError("Mixture diagnostics are missing.")
    for row in rows:
        if (
            row.get("arm") not in {"KF", "KG"}
            or row.get("coordinate_update") != "True"
            or row.get("optimizer_updates_prior_parameters") != "False"
            or row.get("finite") != "True"
            or row.get("weight_floor_respected") != "True"
            or row.get("covariance_positive_definite") != "True"
            or float(row.get("minimum_weight", "nan")) + 1e-7
            < config.weight_floor
            or float(row.get("maximum_condition_number", "inf"))
            > config.maximum_condition_number
            or float(row.get("minimum_eigenvalue", "nan")) <= 0.0
            or row.get("fit_scope") != "source_center_only_all_rows"
        ):
            raise ProtocolError("Mixture prior failed its health/freeze contract.")
        case_counts = json.loads(row.get("component_case_counts", "[]"))
        row_counts = json.loads(row.get("component_row_counts", "[]"))
        if any(
            int(value) < config.minimum_component_cases
            for values in case_counts
            for value in values
        ) or any(
            int(value) < config.minimum_component_rows
            for values in row_counts
            for value in values
        ):
            raise ProtocolError("Mixture component effective counts are too small.")


def _validate_geco_rows(rows: Sequence[Mapping[str, str]]) -> None:
    if not rows:
        raise ProtocolError("GECO trajectory is missing.")
    for row in rows:
        if (
            row.get("arm") not in {"SG", "KG"}
            or row.get("target_provenance")
            != "source_only_warmup_reconstruction"
            or not math.isfinite(float(row.get("target", "nan")))
            or float(row.get("target", "nan")) <= 0.0
            or int(row.get("update_count", 0)) <= 0
        ):
            raise ProtocolError("GECO trajectory violated source-only calibration.")


def _validate_generation_rows(
    config: AggregatePriorStudyConfig,
    budgets: Sequence[Mapping[str, str]],
    rng: Sequence[Mapping[str, str]],
) -> None:
    if len(budgets) != len(rng):
        raise ProtocolError("Generation budget/RNG audit grids differ.")
    for row in budgets:
        if (
            int(row.get("per_class", -1)) != config.generation_per_class
            or int(row.get("total", -1)) != 2 * config.generation_per_class
            or row.get("inner_prevalence_used") != "False"
            or row.get("source_prevalence_used") != "False"
            or row.get("same_across_arms") != "True"
        ):
            raise ProtocolError("Generation budget is not fixed and balanced.")
    for row in rng:
        if (
            row.get("same_epsilon_across_arms") != "True"
            or row.get("same_component_uniform_across_arms") != "True"
            or not row.get("noise_hash")
            or not row.get("posterior_source_index_hash")
        ):
            raise ProtocolError("Generation RNG is not paired across arms.")


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"JSON artifact is not a mapping: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except OSError as exc:
        raise ProtocolError(f"Cannot read CSV artifact: {path}") from exc


def _require_files(root: Path, relatives: Sequence[str]) -> None:
    missing = [relative for relative in relatives if not (root / relative).is_file()]
    if missing:
        raise ProtocolError(f"Aggregate-prior bundle is incomplete: {missing}")
