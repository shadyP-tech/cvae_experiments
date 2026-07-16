from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

from midogpp_thesis.cvae.objectives import (
    ISOTROPIC_OBJECTIVE,
    TASK_FISHER_OBJECTIVE,
)
from midogpp_thesis.cvae.preservation.source_inner_studies.config import (
    LearnedConditionalPriorStudyConfig,
    TaskFisherShrinkageStudyConfig,
    load_fisher_study_config,
    load_prior_study_config,
)
from midogpp_thesis.cvae.preservation.source_inner_studies.contracts import (
    LEARNED_PRIOR_MODEL_FAMILY,
    STANDARD_MODEL_FAMILY,
)
from midogpp_thesis.cvae.preservation.source_inner_studies.fisher_artifacts import (
    FISHER_STATE_INDEX_SCHEMA,
    write_fisher_study_bundle,
)
from midogpp_thesis.cvae.preservation.source_inner_studies.fisher_runner import (
    _coverage_manifest as fisher_coverage_manifest,
    _leakage_report as fisher_leakage_report,
    _protocol_manifest as fisher_protocol_manifest,
    _study_summary as fisher_study_summary,
)
from midogpp_thesis.cvae.preservation.source_inner_studies.preparation import (
    embedded_v1_preparation_lineage,
)
from midogpp_thesis.cvae.preservation.source_inner_studies.prior_artifacts import (
    PRIOR_STATE_INDEX_SCHEMA,
    write_prior_study_bundle,
)
from midogpp_thesis.cvae.preservation.source_inner_studies.prior_runner import (
    _coverage_manifest as prior_coverage_manifest,
    _leakage_report as prior_leakage_report,
    _protocol_manifest as prior_protocol_manifest,
    _study_summary as prior_study_summary,
)
from midogpp_thesis.cvae.preservation.source_inner_studies.validation_common import (
    FISHER_SAMPLER_SCHEMA,
    GENERATION_BUDGET_SCHEMA,
    INITIALIZATION_INDEX_SCHEMA,
    METRIC_SCHEMA,
    PAIRING_AUDIT_SCHEMA,
    PRIOR_SAMPLER_SCHEMA,
    SELECTION_EVIDENCE_SCHEMA,
    StudyTimingRecorder,
    canonical_rows,
    expected_bundle_files,
    read_json,
    selection_evidence_hash,
    write_study_run_state,
)
from midogpp_thesis.cvae.reporting import write_json
from midogpp_thesis.real_features.classifier_reference.artifacts import stable_hash
from midogpp_thesis.real_features.classifier_reference.schemas.midogpp import (
    MIDOGPP_ELIGIBLE_CENTERS,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OUTER = "0"
TRAINING_SEED = 17
GENERATION_SEED = 17
INNERS = tuple(center for center in MIDOGPP_ELIGIBLE_CENTERS if center != OUTER)
MANIFEST_HASH = "a" * 64
FEATURE_CACHE_HASH = "b" * 64


def prior_fixture_config(root: Path) -> LearnedConditionalPriorStudyConfig:
    production = load_prior_study_config(
        REPOSITORY_ROOT
        / "experiments/midogpp/stages/20_cvae_preservation/configs/learned_conditional_prior_source_inner_v2.yaml"
    )
    return replace(
        production,
        artifact_root=root,
        heldout_centers=(OUTER,),
        training_seeds=(TRAINING_SEED,),
        generation_seeds=(GENERATION_SEED,),
    )


def fisher_fixture_config(root: Path) -> TaskFisherShrinkageStudyConfig:
    production = load_fisher_study_config(
        REPOSITORY_ROOT
        / "experiments/midogpp/stages/20_cvae_preservation/configs/task_fisher_shrinkage_source_inner_v2.yaml"
    )
    return replace(
        production,
        artifact_root=root,
        heldout_centers=(OUTER,),
        training_seeds=(TRAINING_SEED,),
        generation_seeds=(GENERATION_SEED,),
    )


def fixture_prior_decisions(
    config: LearnedConditionalPriorStudyConfig,
    *,
    decision_metrics: object,
    protocol_hash: str,
    selection_evidence_hash_value: str,
):
    del decision_metrics
    deltas = prior_fixture_deltas(config)
    consensus = {
        OUTER: {
            "schema_version": "midogpp_learned_conditional_prior_study_decision_v2",
            "outer_target_center": OUTER,
            "status": "FIXTURE_E_PASS",
            "protocol_hash": protocol_hash,
            "selection_evidence_hash": selection_evidence_hash_value,
            "may_feed_model_recipe": False,
            "may_feed_deployable_selection": False,
        }
    }
    child = {
        "schema_version": "midogpp_learned_prior_child_decision_v2",
        "outer_target_center": OUTER,
        "training_seed": TRAINING_SEED,
        "status": "FIXTURE_E_PASS",
        "summary": {"fixture": True},
        "selection_evidence_hash": selection_evidence_hash_value,
        "may_feed_model_recipe": False,
        "may_feed_deployable_selection": False,
    }
    child["study_decision_hash"] = stable_hash(child)
    return consensus, {(TRAINING_SEED, OUTER): child}, deltas


def fixture_fisher_decisions(
    config: TaskFisherShrinkageStudyConfig,
    *,
    decision_metrics: object,
    protocol_hash: str,
    evidence_hash: str,
):
    del decision_metrics
    deltas = fisher_fixture_deltas(config)
    consensus = {
        OUTER: {
            "schema_version": "midogpp_task_fisher_shrinkage_study_decision_v2",
            "outer_target_center": OUTER,
            "status": "FIXTURE_ALPHA_0_10_PASS",
            "protocol_hash": protocol_hash,
            "selection_evidence_hash": evidence_hash,
            "may_feed_model_recipe": False,
            "may_feed_deployable_selection": False,
        }
    }
    child = {
        "schema_version": "midogpp_fisher_shrinkage_child_decision_v2",
        "outer_target_center": OUTER,
        "training_seed": TRAINING_SEED,
        "status": "FIXTURE_ALPHA_0_10_PASS",
        "summary": {"fixture": True},
        "selection_evidence_hash": evidence_hash,
        "may_feed_model_recipe": False,
        "may_feed_deployable_selection": False,
    }
    child["study_decision_hash"] = stable_hash(child)
    return consensus, {(TRAINING_SEED, OUTER): child}, deltas


def prior_fixture_deltas(
    config: LearnedConditionalPriorStudyConfig,
) -> list[dict[str, object]]:
    return [
        {
            "outer_target_center": OUTER,
            "inner_pseudo_target_center": inner,
            "training_seed": TRAINING_SEED,
            "comparison": "e_vs_a",
            "preservation_ratio_delta": 0.08,
        }
        for inner in INNERS
    ]


def fisher_fixture_deltas(
    config: TaskFisherShrinkageStudyConfig,
) -> list[dict[str, object]]:
    del config
    return [
        {
            "outer_target_center": OUTER,
            "inner_pseudo_target_center": inner,
            "training_seed": TRAINING_SEED,
            "alpha": "0.10",
            "preservation_ratio_delta": 0.04,
        }
        for inner in INNERS
    ]


def patch_common_fixture_validators(monkeypatch, module, root: Path) -> None:
    monkeypatch.setattr(
        module,
        "validate_study_checkpoint_index",
        lambda _root: read_json(root / "manifests/checkpoint_index.json"),
    )

    def validate_frame(_root, *, expected_frame_hashes=None):
        payload = read_json(root / "manifests/feature_frame_index.json")
        observed = {
            str(record["frame_hash"])
            for record in payload["records"]
            if isinstance(record, Mapping)
        }
        if expected_frame_hashes is not None:
            assert observed == set(expected_frame_hashes)
        return payload

    monkeypatch.setattr(module, "validate_feature_frame_index", validate_frame)
    monkeypatch.setattr(
        module,
        "validate_embedded_preparation_rows",
        lambda *args, **kwargs: None,
    )


def write_prior_fixture_bundle(
    root: Path, config: LearnedConditionalPriorStudyConfig
) -> dict[str, object]:
    lineage = embedded_v1_preparation_lineage()
    frame = SimpleNamespace(
        manifest_hash=MANIFEST_HASH,
        feature_cache_hash=FEATURE_CACHE_HASH,
    )
    protocol = prior_protocol_manifest(config, frame=frame, lineage=lineage)
    checkpoint_index, initialization_index, frame_index = _prior_runtime_indices(
        config, protocol_hash=str(protocol["protocol_hash"])
    )
    rows = _prior_rows(config, protocol, checkpoint_index)
    state_index = rows.pop("state_index")
    budget_manifest = _budget_manifest(config, rows["budget_rows"])
    deltas = prior_fixture_deltas(config)
    evidence_hash = _evidence_hash(
        protocol=protocol,
        checkpoint_index=checkpoint_index,
        initialization_index=initialization_index,
        frame_index=frame_index,
        budget_manifest=budget_manifest,
        state_index=state_index,
        deltas=deltas,
        rows=rows,
    )
    decisions, children, _ = fixture_prior_decisions(
        config,
        decision_metrics=None,
        protocol_hash=str(protocol["protocol_hash"]),
        selection_evidence_hash_value=evidence_hash,
    )
    decision_metrics = [object()] * (
        len(INNERS)
        * len(config.training_seeds)
        * len(config.generation_seeds)
        * len(config.arms)
    )
    write_prior_study_bundle(
        root,
        learned_prior_state_index=state_index,
        metric_rows=rows["metric_rows"],
        paired_delta_rows=deltas,
        nested_reference_rows=rows["nested_rows"],
        nested_tuning_rows=rows["tuning_rows"],
        sampler_rows=rows["sampler_rows"],
        checkpoint_reuse_rows=rows["checkpoint_rows"],
        initialization_pairing_rows=rows["initialization_rows"],
        generation_budget_rows=rows["budget_rows"],
        rng_rows=rows["rng_rows"],
        identity_rows=rows["identity_rows"],
        protocol_manifest=protocol,
        coverage_manifest=prior_coverage_manifest(
            config,
            decision_metrics=decision_metrics,
            metric_rows=rows["metric_rows"],
            prior_state_records=state_index["records"],
        ),
        selection_evidence_manifest=_selection_manifest(evidence_hash),
        embedded_preparation_lineage=lineage,
        generation_budget_manifest=budget_manifest,
        child_decisions=children,
        consensus_decisions=decisions,
        study_decision=prior_study_summary(
            decisions,
            protocol_hash=str(protocol["protocol_hash"]),
            evidence_hash=evidence_hash,
        ),
        leakage_report=prior_leakage_report(
            config,
            protocol_hash=str(protocol["protocol_hash"]),
            identity_rows=rows["identity_rows"],
        ),
    )
    _write_runtime_surfaces(
        root,
        config=config,
        protocol_hash=str(protocol["protocol_hash"]),
        checkpoint_index=checkpoint_index,
        initialization_index=initialization_index,
        frame_index=frame_index,
    )
    return {"protocol": protocol, "evidence_hash": evidence_hash}


def write_fisher_fixture_bundle(
    root: Path, config: TaskFisherShrinkageStudyConfig
) -> dict[str, object]:
    lineage = embedded_v1_preparation_lineage()
    frame = SimpleNamespace(
        manifest_hash=MANIFEST_HASH,
        feature_cache_hash=FEATURE_CACHE_HASH,
    )
    protocol = fisher_protocol_manifest(config, frame=frame, lineage=lineage)
    state_index = _fisher_state_index(config)
    checkpoint_index, initialization_index, frame_index = _fisher_runtime_indices(
        config,
        protocol_hash=str(protocol["protocol_hash"]),
        state_index=state_index,
    )
    rows = _fisher_rows(config, protocol, checkpoint_index, state_index)
    budget_manifest = _budget_manifest(config, rows["budget_rows"])
    deltas = fisher_fixture_deltas(config)
    evidence_hash = _evidence_hash(
        protocol=protocol,
        checkpoint_index=checkpoint_index,
        initialization_index=initialization_index,
        frame_index=frame_index,
        budget_manifest=budget_manifest,
        state_index=state_index,
        deltas=deltas,
        rows=rows,
    )
    decisions, children, _ = fixture_fisher_decisions(
        config,
        decision_metrics=None,
        protocol_hash=str(protocol["protocol_hash"]),
        evidence_hash=evidence_hash,
    )
    decision_metrics = [object()] * (
        len(INNERS)
        * len(config.training_seeds)
        * len(config.generation_seeds)
        * len(config.alphas)
    )
    write_fisher_study_bundle(
        root,
        task_fisher_state_index=state_index,
        metric_rows=rows["metric_rows"],
        paired_delta_rows=deltas,
        nested_reference_rows=rows["nested_rows"],
        nested_tuning_rows=rows["tuning_rows"],
        sampler_rows=rows["sampler_rows"],
        checkpoint_reuse_rows=rows["checkpoint_rows"],
        initialization_pairing_rows=rows["initialization_rows"],
        generation_budget_rows=rows["budget_rows"],
        rng_rows=rows["rng_rows"],
        identity_rows=rows["identity_rows"],
        protocol_manifest=protocol,
        coverage_manifest=fisher_coverage_manifest(
            config,
            decision_metrics=decision_metrics,
            metric_rows=rows["metric_rows"],
            fisher_records=state_index["records"],
        ),
        selection_evidence_manifest=_selection_manifest(evidence_hash),
        embedded_preparation_lineage=lineage,
        generation_budget_manifest=budget_manifest,
        child_decisions=children,
        consensus_decisions=decisions,
        study_decision=fisher_study_summary(
            decisions,
            protocol_hash=str(protocol["protocol_hash"]),
            evidence_hash=evidence_hash,
        ),
        leakage_report=fisher_leakage_report(
            protocol_hash=str(protocol["protocol_hash"]),
            identity_rows=rows["identity_rows"],
        ),
    )
    _write_runtime_surfaces(
        root,
        config=config,
        protocol_hash=str(protocol["protocol_hash"]),
        checkpoint_index=checkpoint_index,
        initialization_index=initialization_index,
        frame_index=frame_index,
    )
    return {"protocol": protocol, "evidence_hash": evidence_hash}


def assert_exact_fixture_surface(root: Path, config, state_index_relative: str) -> None:
    expected = set(
        expected_bundle_files(config, state_index_relative=state_index_relative)
    )
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    assert expected == observed


def _prior_runtime_indices(config, *, protocol_hash: str):
    records: list[dict[str, object]] = []
    frame_records: list[dict[str, object]] = []
    for inner in INNERS:
        identity = _fold_identity(inner)
        shared = f"shared-{inner}"
        stream = f"stream-{inner}"
        for arm in ("A", "E"):
            learned = arm == "E"
            key_hash = f"prior-{arm.lower()}-{inner}"
            prior_partition_hash = f"prior-partition-{inner}" if learned else "none"
            records.append(
                _checkpoint_record(
                    key_hash=key_hash,
                    checkpoint_hash=f"checkpoint-{arm.lower()}-{inner}",
                    identity=identity,
                    protocol_hash=protocol_hash,
                    model_family=(
                        LEARNED_PRIOR_MODEL_FAMILY
                        if learned
                        else STANDARD_MODEL_FAMILY
                    ),
                    prior_family=(
                        config.learned_prior_family
                        if learned
                        else config.standard_prior_family
                    ),
                    objective_id=ISOTROPIC_OBJECTIVE,
                    alpha=0.0,
                    raw_fisher_state_hash="none",
                    objective_context_hash="none",
                    shared_hash=shared,
                    stream_hash=stream,
                    full_hash=f"full-{arm.lower()}-{inner}",
                    prior_partition_hash=prior_partition_hash,
                )
            )
        frame_records.append({"frame_hash": identity["frame_hash"]})
    return _indices(records, frame_records)


def _fisher_runtime_indices(config, *, protocol_hash: str, state_index):
    records: list[dict[str, object]] = []
    frame_records: list[dict[str, object]] = []
    states = {
        record["inner_pseudo_target_center"]: record
        for record in state_index["records"]
    }
    for inner in INNERS:
        identity = _fold_identity(inner)
        state = states[inner]
        for alpha in config.alphas:
            alpha_key = format(alpha, ".2f")
            raw_hash = "none" if alpha == 0.0 else state["raw_fisher_state_hash"]
            context_hash = (
                "none"
                if alpha == 0.0
                else state["derived_metrics"][alpha_key]["metric_state_hash"]
            )
            records.append(
                _checkpoint_record(
                    key_hash=f"fisher-{inner}-{alpha_key}",
                    checkpoint_hash=f"checkpoint-fisher-{inner}-{alpha_key}",
                    identity=identity,
                    protocol_hash=protocol_hash,
                    model_family=STANDARD_MODEL_FAMILY,
                    prior_family="standard_normal",
                    objective_id=(
                        ISOTROPIC_OBJECTIVE
                        if alpha == 0.0
                        else TASK_FISHER_OBJECTIVE
                    ),
                    alpha=alpha,
                    raw_fisher_state_hash=raw_hash,
                    objective_context_hash=context_hash,
                    shared_hash=f"shared-{inner}",
                    stream_hash=f"stream-{inner}",
                    full_hash=f"full-fisher-{inner}-{alpha_key}",
                    prior_partition_hash="none",
                )
            )
        frame_records.append({"frame_hash": identity["frame_hash"]})
    return _indices(records, frame_records)


def _indices(records, frame_records):
    checkpoint_index = {
        "schema_version": "midogpp_source_inner_study_checkpoint_index_v2",
        "records": records,
    }
    fields = (
        "training_key_hash",
        "model_family",
        "shared_initialization_hash",
        "prior_initialization_hash",
        "full_initialization_hash",
        "training_stream_hash",
    )
    initialization_index = {
        "schema_version": INITIALIZATION_INDEX_SCHEMA,
        "records": [
            {field: record[field] for field in fields}
            for record in records
        ],
    }
    frame_index = {
        "schema_version": "midogpp_source_inner_fixture_frame_index_v2",
        "records": frame_records,
    }
    return checkpoint_index, initialization_index, frame_index


def _checkpoint_record(
    *,
    key_hash: str,
    checkpoint_hash: str,
    identity: Mapping[str, object],
    protocol_hash: str,
    model_family: str,
    prior_family: str,
    objective_id: str,
    alpha: float,
    raw_fisher_state_hash: str,
    objective_context_hash: str,
    shared_hash: str,
    stream_hash: str,
    full_hash: str,
    prior_partition_hash: str,
) -> dict[str, object]:
    training_key = {
        **identity,
        "feature_cache_hash": FEATURE_CACHE_HASH,
        "manifest_hash": MANIFEST_HASH,
        "protocol_hash": protocol_hash,
        "training_seed": TRAINING_SEED,
        "model_family": model_family,
        "prior_family": prior_family,
        "objective_id": objective_id,
        "alpha": alpha,
        "raw_fisher_state_hash": raw_fisher_state_hash,
        "objective_context_hash": objective_context_hash,
    }
    return {
        "training_key_hash": key_hash,
        "checkpoint_hash": checkpoint_hash,
        "model_family": model_family,
        "prior_partition_hash": prior_partition_hash,
        "shared_initialization_hash": shared_hash,
        "prior_initialization_hash": f"prior-init-{key_hash}",
        "full_initialization_hash": full_hash,
        "training_stream_hash": stream_hash,
        "training_key": training_key,
    }


def _prior_rows(config, protocol, checkpoint_index):
    by_key = {record["training_key_hash"]: record for record in checkpoint_index["records"]}
    metric_rows: list[dict[str, object]] = []
    sampler_rows: list[dict[str, object]] = []
    checkpoint_rows: list[dict[str, object]] = []
    initialization_rows: list[dict[str, object]] = []
    rng_rows: list[dict[str, object]] = []
    state_records: list[dict[str, object]] = []
    for inner in INNERS:
        identity = _fold_identity(inner)
        a = by_key[f"prior-a-{inner}"]
        e = by_key[f"prior-e-{inner}"]
        state = {
            "prior_mu": [[0.0] * config.latent_dim, [0.1] * config.latent_dim],
            "prior_rho": [[0.0] * config.latent_dim, [0.0] * config.latent_dim],
            "effective_logvar": [[0.0] * config.latent_dim, [0.0] * config.latent_dim],
        }
        state_record = {
            "outer_target_center": OUTER,
            "inner_pseudo_target_center": inner,
            "training_seed": TRAINING_SEED,
            "source_row_hash": identity["fit_row_hash"],
            "frame_hash": identity["frame_hash"],
            "training_key_hash": e["training_key_hash"],
            "checkpoint_hash": e["checkpoint_hash"],
            "final_prior_partition_hash": e["prior_partition_hash"],
            "primary_preservation_eligible": True,
            "state": state,
        }
        state_record["state_hash"] = stable_hash(state_record)
        state_records.append(state_record)
        c_rows, c_hash = _c_sampler_rows(config, identity, a)
        sampler_rows.extend(c_rows)
        sampler_rows.extend(_e_sampler_rows(config, identity, e, state_record))
        sampler_hashes = {
            "A": stable_hash(
                {"family": config.standard_prior_family, "latent_dim": config.latent_dim}
            ),
            "C-diag": c_hash,
            "E": state_record["state_hash"],
        }
        for arm, bacc in (("A", 0.70), ("C-diag", 0.71), ("E", 0.72)):
            runtime = e if arm == "E" else a
            for role, generation_seed in (
                ("decode", -1),
                ("prior", GENERATION_SEED),
                ("posterior", GENERATION_SEED),
            ):
                metric_rows.append(
                    _metric_row(
                        identity=identity,
                        protocol_hash=str(protocol["protocol_hash"]),
                        training_key_hash=str(runtime["training_key_hash"]),
                        checkpoint_hash=str(runtime["checkpoint_hash"]),
                        shared_hash=str(runtime["shared_initialization_hash"]),
                        stream_hash=str(runtime["training_stream_hash"]),
                        sampler_hash=str(sampler_hashes[arm]),
                        role=role,
                        generation_seed=generation_seed,
                        bacc=bacc,
                        axis={
                            "arm": arm,
                            "alpha": 0.0,
                            "model_family": (
                                LEARNED_PRIOR_MODEL_FAMILY
                                if arm == "E"
                                else STANDARD_MODEL_FAMILY
                            ),
                            "prior_family": (
                                config.ex_post_prior_family
                                if arm == "C-diag"
                                else (
                                    config.learned_prior_family
                                    if arm == "E"
                                    else config.standard_prior_family
                                )
                            ),
                            "objective_id": ISOTROPIC_OBJECTIVE,
                            "eligible": "true",
                            "ineligibility_reason": "",
                        },
                    )
                )
            rng_rows.extend(_rng_rows(inner, axis_field="arm", axis_value=arm))
        checkpoint_rows.append(
            {
                "outer_target_center": OUTER,
                "inner_pseudo_target_center": inner,
                "training_seed": TRAINING_SEED,
                "arm_pair": "A/C-diag",
                "a_training_key_hash": a["training_key_hash"],
                "c_training_key_hash": a["training_key_hash"],
                "a_checkpoint_hash": a["checkpoint_hash"],
                "c_checkpoint_hash": a["checkpoint_hash"],
                "single_checkpoint_reused": "true",
                "status": "PASS",
            }
        )
        initialization_rows.append(
            {
                "outer_target_center": OUTER,
                "inner_pseudo_target_center": inner,
                "training_seed": TRAINING_SEED,
                "arm_pair": "A/E",
                "shared_initialization_hash_a": a["shared_initialization_hash"],
                "shared_initialization_hash_e": e["shared_initialization_hash"],
                "training_stream_hash_a": a["training_stream_hash"],
                "training_stream_hash_e": e["training_stream_hash"],
                "full_initialization_hash_a": a["full_initialization_hash"],
                "full_initialization_hash_e": e["full_initialization_hash"],
                "shared_initialization_equal": "true",
                "training_stream_equal": "true",
                "full_training_identity_distinct": "true",
                "status": "PASS",
            }
        )
    return {
        "metric_rows": metric_rows,
        "sampler_rows": sampler_rows,
        "checkpoint_rows": checkpoint_rows,
        "initialization_rows": initialization_rows,
        "budget_rows": _budget_rows(config),
        "rng_rows": rng_rows,
        "identity_rows": _identity_rows(),
        "nested_rows": [],
        "tuning_rows": [],
        "state_index": {
            "schema_version": PRIOR_STATE_INDEX_SCHEMA,
            "records": state_records,
        },
    }


def _fisher_state_index(config):
    records: list[dict[str, object]] = []
    for inner in INNERS:
        identity = _fold_identity(inner)
        raw_hash = f"raw-fisher-{inner}"
        derived = {
            format(alpha, ".2f"): {
                "metric_state_hash": f"metric-{inner}-{alpha:.2f}",
            }
            for alpha in config.alphas
        }
        records.append(
            {
                "outer_target_center": OUTER,
                "inner_pseudo_target_center": inner,
                "fit_centers": identity["fit_centers"],
                "source_row_hash": identity["fit_row_hash"],
                "frame_hash": identity["frame_hash"],
                "raw_fisher_state_hash": raw_hash,
                "probe_config_hash": f"probe-{inner}",
                "valid": True,
                "derived_metrics": derived,
            }
        )
    return {"schema_version": FISHER_STATE_INDEX_SCHEMA, "records": records}


def _fisher_rows(config, protocol, checkpoint_index, state_index):
    by_key = {record["training_key_hash"]: record for record in checkpoint_index["records"]}
    states = {record["inner_pseudo_target_center"]: record for record in state_index["records"]}
    metric_rows: list[dict[str, object]] = []
    sampler_rows: list[dict[str, object]] = []
    checkpoint_rows: list[dict[str, object]] = []
    initialization_rows: list[dict[str, object]] = []
    rng_rows: list[dict[str, object]] = []
    for inner in INNERS:
        identity = _fold_identity(inner)
        state = states[inner]
        fold_records = []
        for alpha in config.alphas:
            alpha_key = format(alpha, ".2f")
            runtime = by_key[f"fisher-{inner}-{alpha_key}"]
            fold_records.append(runtime)
            sampler_hash = stable_hash(
                {
                    "family": "standard_normal",
                    "latent_dim": config.latent_dim,
                    "source_row_hash": identity["fit_row_hash"],
                    "training_key_hash": runtime["training_key_hash"],
                    "checkpoint_hash": runtime["checkpoint_hash"],
                }
            )
            sampler_rows.extend(
                {
                    "schema_version": FISHER_SAMPLER_SCHEMA,
                    "mechanism": "standard_normal",
                    "outer_target_center": OUTER,
                    "inner_pseudo_target_center": inner,
                    "training_seed": TRAINING_SEED,
                    "alpha": alpha,
                    "class_label": class_label,
                    "latent_dim": config.latent_dim,
                    "source_row_hash": identity["fit_row_hash"],
                    "requested_family": "standard_normal",
                    "realized_family": "standard_normal",
                    "mean": json.dumps([0.0] * config.latent_dim),
                    "variance": json.dumps([1.0] * config.latent_dim),
                    "training_key_hash": runtime["training_key_hash"],
                    "checkpoint_hash": runtime["checkpoint_hash"],
                    "sampler_state_hash": sampler_hash,
                    "fallback_reason": "",
                }
                for class_label in (0, 1)
            )
            bacc = 0.70 + alpha * 0.10
            raw_hash = "none" if alpha == 0.0 else state["raw_fisher_state_hash"]
            context_hash = (
                "none"
                if alpha == 0.0
                else state["derived_metrics"][alpha_key]["metric_state_hash"]
            )
            for role, generation_seed in (
                ("decode", -1),
                ("prior", GENERATION_SEED),
                ("posterior", GENERATION_SEED),
            ):
                metric_rows.append(
                    _metric_row(
                        identity=identity,
                        protocol_hash=str(protocol["protocol_hash"]),
                        training_key_hash=str(runtime["training_key_hash"]),
                        checkpoint_hash=str(runtime["checkpoint_hash"]),
                        shared_hash=str(runtime["shared_initialization_hash"]),
                        stream_hash=str(runtime["training_stream_hash"]),
                        sampler_hash=sampler_hash,
                        role=role,
                        generation_seed=generation_seed,
                        bacc=bacc,
                        axis={
                            "arm": f"alpha={alpha:.2f}",
                            "alpha": alpha,
                            "model_family": STANDARD_MODEL_FAMILY,
                            "prior_family": "standard_normal",
                            "objective_id": (
                                ISOTROPIC_OBJECTIVE
                                if alpha == 0.0
                                else TASK_FISHER_OBJECTIVE
                            ),
                            "raw_fisher_state_hash": raw_hash,
                            "objective_context_hash": context_hash,
                            "classifier_spec_hash": state["probe_config_hash"],
                            "eligible": "true",
                        },
                    )
                )
            rng_rows.extend(_rng_rows(inner, axis_field="alpha", axis_value=alpha))
            checkpoint_rows.append(
                {
                    "outer_target_center": OUTER,
                    "inner_pseudo_target_center": inner,
                    "training_seed": TRAINING_SEED,
                    "alpha": alpha,
                    "training_key_hash": runtime["training_key_hash"],
                    "checkpoint_hash": runtime["checkpoint_hash"],
                    "literal_alpha_zero": str(alpha == 0.0).lower(),
                    "raw_fisher_state_hash": raw_hash,
                    "status": "PASS",
                }
            )
        initialization_rows.append(
            {
                "outer_target_center": OUTER,
                "inner_pseudo_target_center": inner,
                "training_seed": TRAINING_SEED,
                "alphas_present": json.dumps(list(config.alphas)),
                "shared_initialization_hashes": json.dumps(
                    sorted({record["shared_initialization_hash"] for record in fold_records})
                ),
                "training_stream_hashes": json.dumps(
                    sorted({record["training_stream_hash"] for record in fold_records})
                ),
                "raw_fisher_valid": "true",
                "status": "PASS",
            }
        )
    return {
        "metric_rows": metric_rows,
        "sampler_rows": sampler_rows,
        "checkpoint_rows": checkpoint_rows,
        "initialization_rows": initialization_rows,
        "budget_rows": _budget_rows(config),
        "rng_rows": rng_rows,
        "identity_rows": _identity_rows(),
        "nested_rows": [],
        "tuning_rows": [],
    }


def _metric_row(
    *,
    identity: Mapping[str, object],
    protocol_hash: str,
    training_key_hash: str,
    checkpoint_hash: str,
    shared_hash: str,
    stream_hash: str,
    sampler_hash: str,
    role: str,
    generation_seed: int,
    bacc: float,
    axis: Mapping[str, object],
) -> dict[str, object]:
    ratio = (bacc - 0.5) / (0.75 - 0.5)
    row = {
        "schema_version": METRIC_SCHEMA,
        "method": "fixture_source_inner_study_v2",
        "protocol_hash": protocol_hash,
        **identity,
        "fit_centers": json.dumps(identity["fit_centers"]),
        "training_seed": TRAINING_SEED,
        "generation_seed": generation_seed,
        "representation_role": role,
        "bacc": bacc,
        "macro_f1": bacc - 0.01,
        "real_reference_bacc": 0.75,
        "preservation_ratio": ratio,
        "generation_class_counts": json.dumps([4, 4]),
        "classifier_spec_hash": f"probe-{identity['inner_pseudo_target_center']}",
        "training_key_hash": training_key_hash,
        "checkpoint_hash": checkpoint_hash,
        "shared_initialization_hash": shared_hash,
        "training_stream_hash": stream_hash,
        "sampler_state_hash": sampler_hash,
        "valid": "true",
        "eligible": "true",
        "status": "ok",
        "claim_scope": "cvae_source_inner_study_only",
        "selection_source": "fully_nested_source_inner",
        "target_eval_labels_used_for_selection": "false",
        "may_feed_model_recipe": "false",
        "may_feed_deployable_selection": "false",
        "routing_performed": "false",
        "composition_performed": "false",
    }
    row.update(axis)
    return row


def _fold_identity(inner: str) -> dict[str, object]:
    return {
        "outer_target_center": OUTER,
        "inner_pseudo_target_center": inner,
        "fit_centers": [
            center
            for center in MIDOGPP_ELIGIBLE_CENTERS
            if center not in {OUTER, inner}
        ],
        "fit_row_hash": f"fit-{inner}",
        "eval_row_hash": f"eval-{inner}",
        "frame_hash": f"frame-{inner}",
    }


def _c_sampler_rows(config, identity, checkpoint):
    classes: dict[str, object] = {}
    rows: list[dict[str, object]] = []
    for class_label in (0, 1):
        mean = [0.05 * class_label] * config.latent_dim
        covariance = [1.0] * config.latent_dim
        classes[str(class_label)] = {
            "class_label": class_label,
            "requested_family": config.ex_post_prior_family,
            "realized_family": config.ex_post_prior_family,
            "mean": mean,
            "covariance": covariance,
            "n_rows": 4,
            "raw_between_covariance": [0.0] * config.latent_dim,
            "within_posterior_diagonal": [1.0] * config.latent_dim,
            "shrinkage": None,
            "shrinkage_target": None,
            "jitter": 0.0,
            "condition_number": 1.0,
            "eigenvalues": [1.0] * config.latent_dim,
            "fallback_reason": "",
        }
    state_hash = stable_hash(
        {
            "requested_family": config.ex_post_prior_family,
            "latent_dim": config.latent_dim,
            "source_row_hash": identity["fit_row_hash"],
            "classes": classes,
        }
    )
    for class_label in (0, 1):
        state = classes[str(class_label)]
        rows.append(
            {
                "schema_version": PRIOR_SAMPLER_SCHEMA,
                "mechanism": "ex_post_aggregate_posterior_diagonal",
                "outer_target_center": OUTER,
                "inner_pseudo_target_center": identity["inner_pseudo_target_center"],
                "training_seed": TRAINING_SEED,
                "arm": "C-diag",
                "class_label": class_label,
                "latent_dim": config.latent_dim,
                "source_row_hash": identity["fit_row_hash"],
                "training_key_hash": checkpoint["training_key_hash"],
                "checkpoint_hash": checkpoint["checkpoint_hash"],
                "sampler_state_hash": state_hash,
                **{
                    key: json.dumps(value)
                    if key
                    in {
                        "mean",
                        "covariance",
                        "raw_between_covariance",
                        "within_posterior_diagonal",
                        "eigenvalues",
                    }
                    else (
                        ""
                        if key in {"shrinkage", "shrinkage_target"} and value is None
                        else value
                    )
                    for key, value in state.items()
                    if key != "class_label"
                },
            }
        )
    return rows, state_hash


def _e_sampler_rows(config, identity, checkpoint, state_record):
    state = state_record["state"]
    return [
        {
            "schema_version": PRIOR_SAMPLER_SCHEMA,
            "mechanism": "jointly_learned_class_conditional_diagonal_prior",
            "outer_target_center": OUTER,
            "inner_pseudo_target_center": identity["inner_pseudo_target_center"],
            "training_seed": TRAINING_SEED,
            "arm": "E",
            "class_label": class_label,
            "latent_dim": config.latent_dim,
            "source_row_hash": identity["fit_row_hash"],
            "requested_family": config.learned_prior_family,
            "realized_family": config.learned_prior_family,
            "mean": json.dumps(state["prior_mu"][class_label]),
            "logvar": json.dumps(state["effective_logvar"][class_label]),
            "variance": json.dumps(
                [math.exp(value) for value in state["effective_logvar"][class_label]]
            ),
            "training_key_hash": checkpoint["training_key_hash"],
            "checkpoint_hash": checkpoint["checkpoint_hash"],
            "sampler_state_hash": state_record["state_hash"],
            "fallback_reason": "",
        }
        for class_label in (0, 1)
    ]


def _budget_rows(config):
    return [
        {
            "schema_version": GENERATION_BUDGET_SCHEMA,
            "outer_target_center": OUTER,
            "inner_pseudo_target_center": inner,
            "source_row_hash": f"fit-{inner}",
            "ordered_label_vector_hash": f"labels-{inner}",
            "class_counts": json.dumps([4, 4]),
            "budget_policy": config.generation_budget_policy,
            "derived_from_y_fit_only": "true",
            "used_inner_labels": "false",
        }
        for inner in INNERS
    ]


def _budget_manifest(config, rows):
    return {
        "schema_version": GENERATION_BUDGET_SCHEMA,
        "policy": config.generation_budget_policy,
        "derived_from_y_fit_only": True,
        "n_records": len(rows),
        "records_hash": stable_hash(canonical_rows(rows)),
    }


def _identity_rows():
    return [
        {
            "outer_target_center": OUTER,
            "inner_pseudo_target_center": inner,
            "status": "PASS",
        }
        for inner in INNERS
    ]


def _rng_rows(inner: str, *, axis_field: str, axis_value: object):
    return [
        {
            "schema_version": PAIRING_AUDIT_SCHEMA,
            "outer_target_center": OUTER,
            "inner_pseudo_target_center": inner,
            "training_seed": TRAINING_SEED,
            "generation_seed": GENERATION_SEED,
            axis_field: axis_value,
            "stream": stream,
            "epsilon_hash": f"epsilon-{inner}-{stream}",
            "epsilon_depends_on_training_seed": "false",
            "status": "PASS",
        }
        for stream in ("prior_generation", "posterior_evaluation")
    ]


def _selection_manifest(evidence_hash: str):
    return {
        "schema_version": SELECTION_EVIDENCE_SCHEMA,
        "selection_evidence_hash": evidence_hash,
        "runtime_rows_included": False,
        "decisions_may_feed_model_recipe": False,
    }


def _evidence_hash(
    *,
    protocol,
    checkpoint_index,
    initialization_index,
    frame_index,
    budget_manifest,
    state_index,
    deltas,
    rows,
):
    return selection_evidence_hash(
        metric_rows=rows["metric_rows"],
        paired_delta_rows=deltas,
        nested_reference_rows=rows["nested_rows"],
        nested_tuning_rows=rows["tuning_rows"],
        sampler_rows=rows["sampler_rows"],
        identity_rows=rows["identity_rows"],
        checkpoint_reuse_rows=rows["checkpoint_rows"],
        initialization_pairing_rows=rows["initialization_rows"],
        generation_budget_rows=rows["budget_rows"],
        checkpoint_index=checkpoint_index,
        initialization_index=initialization_index,
        feature_frame_index=frame_index,
        generation_budget_manifest=budget_manifest,
        rng_rows=rows["rng_rows"],
        protocol_manifest=protocol,
        study_state_index=state_index,
    )


def _write_runtime_surfaces(
    root: Path,
    *,
    config,
    protocol_hash: str,
    checkpoint_index,
    initialization_index,
    frame_index,
):
    write_json(root / "manifests/checkpoint_index.json", checkpoint_index)
    write_json(root / "manifests/initialization_index.json", initialization_index)
    write_json(root / "manifests/feature_frame_index.json", frame_index)
    timing = StudyTimingRecorder(root, protocol_hash=protocol_hash, mode=config.mode)
    timing.finalize()
    write_study_run_state(
        root,
        protocol_hash=protocol_hash,
        mode=config.mode,
        status="COMPLETE",
    )
