"""Promote the completed Uniform-B v2 study into a frozen Stage-30 bank."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Mapping, Sequence

import numpy as np
import torch

from ....common.hashing import stable_hash
from ....real_features.classifier_reference.real_feature_frame import (
    load_midogpp_real_feature_frame,
)
from ...generation_samplers import FULL_SAMPLER, fit_aggregate_posterior_sampler
from ...keyed_training import model_state_hash
from ...models import ClassConditionedCVAE
from ...protocol import ProtocolError
from ...reporting import write_csv_rows, write_json
from ...preservation.independent_source import extract_source_data
from .config import UniformBV2PromotionConfig
from .contracts import (
    CENTERS,
    CLAIM_SCOPE,
    CONTROL_SAMPLER_FAMILY,
    EXPERIMENT_ID,
    N_EXPERTS,
    PROMOTION_DECISION,
    PROMOTION_REVIEW_ID,
    PUBLICATION_STATE,
    SOURCE_EXPERIMENT_ID,
    TRAINING_SEEDS,
    legal_routing_sources,
)
from .serialization import source_frame_from_payload


def run_promotion(
    config: UniformBV2PromotionConfig,
    *,
    artifact_root: Path | None = None,
) -> Path:
    """Run the reviewed evidence gates and materialize the authorized bank."""

    root = Path(artifact_root or config.artifact_root)
    for relative in ("experts", "frames", "samplers", "manifests", "reports", "tables", "provenance"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    state_path = root / "reports/run_state.json"
    if state_path.is_file() and _read_json(state_path).get("status") == "COMPLETE":
        from .validation import validate_promoted_bank

        validate_promoted_bank(root, config=config)
        return root
    _write_state(root, "RUNNING")
    try:
        _run(config, root)
        _write_state(root, "COMPLETE")
        from .validation import validate_promoted_bank

        checks = validate_promoted_bank(root, config=config, allow_pending=True)
        write_json(
            root / "reports/validation_report.json",
            {
                "schema_version": "midogpp_uniform_b_v2_expert_bank_validation_v1",
                "status": "PASS",
                "validator": "validate_promoted_bank",
                "checks": checks,
            },
        )
        validate_promoted_bank(root, config=config)
    except Exception:
        _write_state(root, "FAILED")
        raise
    return root


def _run(config: UniformBV2PromotionConfig, root: Path) -> None:
    source = config.source_study_root
    audit, checkpoints, frame_payloads, source_sampler_hashes = audit_source_bundle(
        source,
        config=config,
    )
    frame = load_midogpp_real_feature_frame(
        manifest_path=config.manifest_path,
        feature_cache_path=config.feature_cache_path,
        expected_feature_dim=3840,
        allow_excluded_center_omission=True,
    )
    if (
        frame.eligible_centers != CENTERS
        or frame.manifest_hash != config.expected_manifest_hash
        or frame.feature_cache_hash != config.expected_feature_cache_hash
    ):
        raise ProtocolError("Promotion canonical source-data identity drifted.")
    sources = {center: extract_source_data(frame, center) for center in CENTERS}
    frames = {}
    frame_files = {}
    for center in CENTERS:
        payload = frame_payloads[center]
        observed = source_frame_from_payload(payload)
        if observed.source_center != center or observed.source_row_hash != sources[center].row_hash:
            raise ProtocolError("Promotion source-frame row identity drifted.")
        frame_path = root / "frames" / f"center_{center}.json"
        write_json(frame_path, payload)
        frames[center] = observed
        frame_files[center] = {
            "path": frame_path.relative_to(root).as_posix(),
            "sha256": _sha256_file(frame_path),
        }

    runtime_device = _resolve_device(config.runtime_device)
    records = []
    expert_rows = []
    sampler_rows = []
    known_hashes: dict[str, str] = {
        str(value["path"]): str(value["sha256"])
        for value in frame_files.values()
    }
    for checkpoint in checkpoints:
        center = str(checkpoint["source_center"])
        seed = int(checkpoint["training_seed"])
        source_checkpoint = _safe_member(source, str(checkpoint["relative_path"]))
        destination = root / "experts" / f"center_{center}" / f"seed_{seed}.pt"
        storage_mode = _materialize_checkpoint(
            source_checkpoint,
            destination,
            expected_sha256=str(checkpoint["file_sha256"]),
        )
        model = _load_verified_model(
            destination,
            config=config,
            expected_model_hash=str(checkpoint["checkpoint_hash"]),
            device=runtime_device,
        )
        projected = np.asarray(
            frames[center].frame.transform(sources[center].embeddings),
            dtype=np.float32,
        )
        labels = np.asarray(sources[center].labels, dtype=np.int64)
        sampler = _fit_sampler(
            model,
            projected,
            labels,
            source_row_hash=sources[center].row_hash,
            config=config,
            device=runtime_device,
        )
        expected_sampler_hash = source_sampler_hashes[(center, seed)]
        if sampler.state_hash != expected_sampler_hash:
            raise ProtocolError("Recomputed promoted sampler does not match v2 audit.")
        sampler_payload = {
            "schema_version": "midogpp_uniform_b_v2_promoted_sampler_v1",
            "requested_family": sampler.requested_family,
            "latent_dim": sampler.latent_dim,
            "source_row_hash": sampler.source_row_hash,
            "classes": {
                str(label): state.to_payload()
                for label, state in sorted(sampler.classes.items())
            },
            "sampler_state_hash": sampler.state_hash,
            "source_only_fit": True,
            "outer_or_target_rows_used": False,
        }
        sampler_path = root / "samplers" / f"center_{center}" / f"seed_{seed}.json"
        write_json(sampler_path, sampler_payload)
        sampler_file_hash = _sha256_file(sampler_path)
        checkpoint_relative = destination.relative_to(root).as_posix()
        sampler_relative = sampler_path.relative_to(root).as_posix()
        record = {
            "schema_version": "midogpp_uniform_b_v2_promoted_expert_lock_v1",
            "source_center": center,
            "training_seed": seed,
            "checkpoint_path": checkpoint_relative,
            "checkpoint_file_sha256": str(checkpoint["file_sha256"]),
            "checkpoint_hash": str(checkpoint["checkpoint_hash"]),
            "training_key_hash": str(checkpoint["training_key_hash"]),
            "completed_step": int(checkpoint["completed_step"]),
            "fresh_source_only_training": checkpoint["fresh_source_only_training"],
            "parent_checkpoint_used": checkpoint["parent_checkpoint_used"],
            "frame_path": str(frame_files[center]["path"]),
            "frame_file_sha256": str(frame_files[center]["sha256"]),
            "frame_hash": frames[center].state_hash,
            "sampler_path": sampler_relative,
            "sampler_file_sha256": sampler_file_hash,
            "sampler_state_hash": sampler.state_hash,
            "sampler_family": CONTROL_SAMPLER_FAMILY,
            "checkpoint_storage_mode": storage_mode,
            "individual_expert_or_seed_selected": False,
            "routing_authorized": True,
        }
        record["expert_lock_hash"] = stable_hash(record)
        records.append(record)
        expert_rows.append(
            {
                "source_center": center,
                "training_seed": seed,
                "checkpoint_hash": checkpoint["checkpoint_hash"],
                "checkpoint_file_sha256": checkpoint["file_sha256"],
                "frame_hash": frames[center].state_hash,
                "sampler_state_hash": sampler.state_hash,
                "storage_mode": storage_mode,
                "routing_authorized": True,
            }
        )
        for label, state in sorted(sampler.classes.items()):
            sampler_rows.append(
                {
                    "source_center": center,
                    "training_seed": seed,
                    "class_label": label,
                    "realized_family": state.realized_family,
                    "n_rows": state.n_rows,
                    "condition_number": state.condition_number,
                    "fallback_reason": state.fallback_reason,
                    "sampler_state_hash": sampler.state_hash,
                }
            )
        known_hashes[checkpoint_relative] = str(checkpoint["file_sha256"])
        known_hashes[sampler_relative] = sampler_file_hash
        del model
        if runtime_device.startswith("cuda:"):
            torch.cuda.empty_cache()
    if len(records) != N_EXPERTS:
        raise ProtocolError("Promotion did not retain all 27 expert replicas.")

    bank = {
        "schema_version": "midogpp_uniform_b_v2_routing_expert_bank_index_v1",
        "experiment_id": EXPERIMENT_ID,
        "claim_scope": CLAIM_SCOPE,
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "source_protocol_hash": config.expected_source_protocol_hash,
        "model": {
            "input_dim": config.input_dim,
            "hidden_dim": config.hidden_dim,
            "latent_dim": config.latent_dim,
            "num_hidden_layers": config.num_hidden_layers,
        },
        "centers": list(CENTERS),
        "training_seeds": list(TRAINING_SEEDS),
        "n_experts": len(records),
        "replica_policy": "retain_all_three_no_validation_based_seed_selection",
        "records": records,
        "routing_authorized": True,
        "may_feed_deployable_selection": True,
    }
    bank["bank_lock_hash"] = stable_hash(bank)
    write_json(root / "manifests/expert_bank_index.json", bank)
    control = _control_lock(config, bank_lock_hash=str(bank["bank_lock_hash"]))
    write_json(root / "manifests/equal_union_ps_control_lock.json", control)
    protocol = _promotion_protocol(config, bank, control)
    write_json(root / "manifests/promotion_protocol.json", protocol)
    review = _review_snapshot(config, audit)
    write_json(root / "manifests/promotion_review_snapshot.json", review)
    write_json(
        root / "manifests/source_evidence_lock.json",
        {
            "schema_version": "midogpp_uniform_b_v2_source_evidence_lock_v1",
            **audit,
            "source_evidence_consumed_for_whole_bank_adoption": True,
            "may_be_reused_for_individual_expert_or_seed_selection": False,
        },
    )
    write_csv_rows(root / "tables/expert_inventory.csv", expert_rows)
    write_csv_rows(root / "tables/sampler_inventory.csv", sampler_rows)
    write_csv_rows(root / "tables/source_gate_audit.csv", _gate_rows(audit))
    decision = _promotion_decision(config, audit, bank, control)
    write_json(root / "reports/promotion_decision.json", decision)
    write_json(
        root / "reports/test_consumption_ledger.json",
        {
            "schema_version": "midogpp_uniform_b_v2_bank_test_consumption_v1",
            "status": "CONSUMED_FOR_WHOLE_BANK_ADOPTION",
            "source_inner_evaluation_labels_consumed": True,
            "individual_expert_or_seed_selection_performed": False,
            "all_27_experts_retained": True,
            "may_be_reused_as_fresh_bank_selection_evidence": False,
            "may_be_reused_for_locked_control_scoring": True,
        },
    )
    write_json(
        root / "reports/leakage_report.json",
        {
            "schema_version": "midogpp_uniform_b_v2_bank_leakage_v1",
            "status": "PASS",
            "fresh_source_only_training": True,
            "target_expert_excluded_in_every_routing_fold": True,
            "target_support_labels_used_for_routing_selection": False,
            "source_inner_evaluation_labels_consumed_for_whole_bank_adoption": True,
            "individual_expert_or_seed_selection_performed": False,
            "identity_overlap_failures": 0,
        },
    )
    (root / "reports/promotion_report.md").write_text(
        _promotion_report(audit), encoding="utf-8"
    )
    _write_content_index(root, known_hashes=known_hashes)


def audit_source_bundle(
    root: str | Path,
    *,
    config: UniformBV2PromotionConfig,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    dict[str, dict[str, object]],
    dict[tuple[str, int], str],
]:
    """Independently re-audit every v2 promotion gate without mutating v2."""

    source = Path(root)
    locked_files = {
        "manifests/content_index.json": config.expected_source_content_index_sha256,
        "manifests/checkpoint_index.json": config.expected_source_checkpoint_index_sha256,
        "manifests/frame_index.json": config.expected_source_frame_index_sha256,
        "reports/study_decision.json": config.expected_source_decision_sha256,
    }
    for relative, expected in locked_files.items():
        member = source / relative
        if not member.is_file() or _sha256_file(member) != expected:
            raise ProtocolError(f"Locked v2 promotion evidence drifted: {relative}.")
    _verify_source_content_index(source)
    protocol = _read_json(source / "manifests/protocol_manifest.json")
    coverage = _read_json(source / "manifests/coverage_manifest.json")
    decision = _read_json(source / "reports/study_decision.json")
    publication = _read_json(source / "reports/publication_state.json")
    run_state = _read_json(source / "reports/run_state.json")
    validation = _read_json(source / "reports/validation_report.json")
    leakage = _read_json(source / "reports/leakage_report.json")
    if (
        protocol.get("experiment_id") != SOURCE_EXPERIMENT_ID
        or protocol.get("protocol_hash") != config.expected_source_protocol_hash
        or protocol.get("config_hash") != config.expected_source_config_hash
        or protocol.get("manifest_hash") != config.expected_manifest_hash
        or protocol.get("feature_cache_hash") != config.expected_feature_cache_hash
        or protocol.get("fresh_source_only_training") is not True
        or protocol.get("outer_or_inner_rows_used_for_fit") is not False
        or protocol.get("routing_or_target_conditioned_selection") is not False
        or run_state.get("status") != "COMPLETE"
        or validation.get("status") != "PASS"
        or leakage.get("status") != "PASS"
        or leakage.get("identity_audit_failures") != 0
        or publication.get("publication_state") != "NON_CONSUMABLE_STUDY_COMPLETE"
        or publication.get("separate_promotion_artifact_required") is not True
        or publication.get("may_feed_deployable_selection") is not False
    ):
        raise ProtocolError("Completed v2 source boundary failed promotion review.")
    if (
        coverage.get("centers") != list(CENTERS)
        or coverage.get("training_seeds") != list(config.training_seeds)
        or coverage.get("generation_seeds") != list(config.generation_seeds)
        or coverage.get("checkpoint_records") != config.required_checkpoint_records
        or coverage.get("generation_blocks") != config.required_generation_blocks
        or coverage.get("task_metric_rows") != config.required_task_metric_rows
        or coverage.get("expected_task_metric_rows") != config.required_task_metric_rows
        or coverage.get("legal_sources_per_task") != 7
    ):
        raise ProtocolError("Completed v2 coverage failed promotion review.")
    metric_rows = _read_csv(source / "tables/task_metrics.csv")
    if len(metric_rows) != config.required_task_metric_rows:
        raise ProtocolError("Completed v2 task-metric rows are incomplete.")
    by_arm = {
        arm: np.asarray(
            [float(row["bacc"]) for row in metric_rows if row["arm"] == arm],
            dtype=np.float64,
        )
        for arm in ("P0", "PS", "Q", "QM", "R")
    }
    if any(len(values) != 648 for values in by_arm.values()) or any(
        row.get("classifier_converged") != "True"
        or row.get("inner_labels_used_for_scoring_only") != "True"
        or row.get("outer_or_inner_rows_used_for_fit") != "False"
        or int(row.get("legal_source_count", -1)) != 7
        for row in metric_rows
    ):
        raise ProtocolError("Completed v2 task metrics failed integrity gates.")
    means = {arm: float(values.mean()) for arm, values in by_arm.items()}
    seed_means = {
        str(seed): float(
            np.mean(
                [
                    float(row["bacc"])
                    for row in metric_rows
                    if row["arm"] == "PS" and int(row["training_seed"]) == seed
                ]
            )
        )
        for seed in config.training_seeds
    }
    posterior = max(means["Q"], means["QM"])
    if (
        means["PS"] < config.min_ps_mean_bacc
        or min(seed_means.values()) < config.min_ps_seed_bacc
        or means["PS"] - means["P0"] < config.min_ps_minus_p0
        or posterior - means["PS"] > config.max_posterior_ceiling_gap
        or decision.get("decision") != "TARGET_METRIC_REACHED_REQUIRES_SEPARATE_PROMOTION"
        or decision.get("ps_reaches_target") is not True
        or decision.get("ps_seed_stability_gate") is not True
    ):
        raise ProtocolError("Completed v2 performance failed promotion gates.")
    sampler_audit = _read_csv(source / "tables/sampler_audit.csv")
    sampler_hashes: dict[tuple[str, int], str] = {}
    if len(sampler_audit) != config.required_sampler_records:
        raise ProtocolError("Completed v2 sampler audit is incomplete.")
    for row in sampler_audit:
        key = (str(row["source_center"]), int(row["training_seed"]))
        observed = str(row["sampler_state_hash"])
        previous = sampler_hashes.setdefault(key, observed)
        if (
            previous != observed
            or row.get("requested_family_realized_for_both_classes") != "True"
            or row.get("effective_ps_family") != CONTROL_SAMPLER_FAMILY
            or row.get("partial_class_fallback_allowed") != "False"
            or row.get("source_only_fit") != "True"
            or row.get("outer_or_inner_rows_used") != "False"
        ):
            raise ProtocolError("Completed v2 sampler audit failed promotion gates.")
    if set(sampler_hashes) != {(center, seed) for center in CENTERS for seed in TRAINING_SEEDS}:
        raise ProtocolError("Completed v2 sampler key coverage drifted.")
    identity = _read_csv(source / "tables/identity_overlap_audit.csv")
    if len(identity) != 648 or any(
        row.get("status") != "PASS"
        or any(int(row[field]) != 0 for field in ("sample_overlap_count", "case_overlap_count", "image_overlap_count"))
        for row in identity
    ):
        raise ProtocolError("Completed v2 identity firewall failed promotion review.")
    checkpoint_index = _read_json(source / "manifests/checkpoint_index.json")
    raw_checkpoints = checkpoint_index.get("records")
    if not isinstance(raw_checkpoints, list) or len(raw_checkpoints) != N_EXPERTS:
        raise ProtocolError("Completed v2 checkpoint index is incomplete.")
    checkpoints = [dict(row) for row in raw_checkpoints if isinstance(row, Mapping)]
    expected_keys = {(center, seed) for center in CENTERS for seed in TRAINING_SEEDS}
    if len(checkpoints) != N_EXPERTS or {
        (str(row.get("source_center")), int(row.get("training_seed", -1)))
        for row in checkpoints
    } != expected_keys:
        raise ProtocolError("Completed v2 checkpoint key coverage drifted.")
    for row in checkpoints:
        member = _safe_member(source, str(row.get("relative_path", "")))
        if (
            row.get("fresh_source_only_training") is not True
            or row.get("parent_checkpoint_used") is not False
            or int(row.get("completed_step", -1)) != 4000
            or not member.is_file()
            or _sha256_file(member) != row.get("file_sha256")
        ):
            raise ProtocolError("Completed v2 checkpoint firewall failed promotion review.")
    frame_index = _read_json(source / "manifests/frame_index.json")
    raw_frames = frame_index.get("records")
    if not isinstance(raw_frames, list) or len(raw_frames) != len(CENTERS):
        raise ProtocolError("Completed v2 source-frame coverage drifted.")
    frames: dict[str, dict[str, object]] = {}
    for row in raw_frames:
        if not isinstance(row, Mapping) or not isinstance(row.get("state"), Mapping):
            raise ProtocolError("Completed v2 source-frame record is invalid.")
        center = str(row.get("source_center"))
        state = dict(row["state"])
        restored = source_frame_from_payload(state)
        if restored.state_hash != row.get("state_hash") or center != restored.source_center:
            raise ProtocolError("Completed v2 source-frame hash drifted.")
        frames[center] = state
    if set(frames) != set(CENTERS):
        raise ProtocolError("Completed v2 source-frame keys drifted.")
    audit = {
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "source_protocol_hash": config.expected_source_protocol_hash,
        "source_config_hash": config.expected_source_config_hash,
        "source_content_index_sha256": config.expected_source_content_index_sha256,
        "source_checkpoint_index_sha256": config.expected_source_checkpoint_index_sha256,
        "source_frame_index_sha256": config.expected_source_frame_index_sha256,
        "source_decision_sha256": config.expected_source_decision_sha256,
        "source_validation_status": "PASS",
        "source_run_state": "COMPLETE",
        "checkpoint_records": len(checkpoints),
        "sampler_records": len(sampler_audit),
        "task_metric_rows": len(metric_rows),
        "generation_blocks": int(coverage["generation_blocks"]),
        "identity_overlap_failures": 0,
        "classifier_convergence_failures": 0,
        "ps_mean_bacc": means["PS"],
        "p0_mean_bacc": means["P0"],
        "ps_minus_p0": means["PS"] - means["P0"],
        "posterior_ceiling_bacc": posterior,
        "posterior_ceiling_minus_ps": posterior - means["PS"],
        "ps_training_seed_mean_bacc": seed_means,
        "minimum_ps_training_seed_mean_bacc": min(seed_means.values()),
        "all_sampler_realizations_full_shrinkage": True,
        "promotion_gates_passed": True,
    }
    return audit, checkpoints, frames, sampler_hashes


def _fit_sampler(
    model: ClassConditionedCVAE,
    projected: np.ndarray,
    labels: np.ndarray,
    *,
    source_row_hash: str,
    config: UniformBV2PromotionConfig,
    device: str,
):
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(projected, dtype=torch.float32, device=device)
        y = torch.as_tensor(labels, dtype=torch.long, device=device)
        mu, logvar = model.encode(x, y)
    sampler = fit_aggregate_posterior_sampler(
        mu.detach().cpu().numpy(),
        logvar.detach().cpu().numpy(),
        labels,
        family=FULL_SAMPLER,
        source_row_hash=source_row_hash,
        min_class_count=config.sampler_min_class_count,
        max_condition_number=config.sampler_max_condition_number,
    )
    if not sampler.requested_family_realized_for_both_classes:
        raise ProtocolError("Promoted sampler unexpectedly requires fallback.")
    return sampler


def _load_verified_model(
    path: Path,
    *,
    config: UniformBV2PromotionConfig,
    expected_model_hash: str,
    device: str,
) -> ClassConditionedCVAE:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model"), Mapping):
        raise ProtocolError("Promoted v2 checkpoint payload is invalid.")
    model = ClassConditionedCVAE(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        num_hidden_layers=config.num_hidden_layers,
    ).to(device)
    model.load_state_dict(payload["model"], strict=True)
    if model_state_hash(model) != expected_model_hash:
        raise ProtocolError("Promoted v2 checkpoint model hash drifted.")
    return model


def _materialize_checkpoint(source: Path, destination: Path, *, expected_sha256: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or _sha256_file(destination) != expected_sha256:
            raise ProtocolError(f"Existing promoted checkpoint is inconsistent: {destination}.")
        return "existing_hash_verified"
    try:
        os.link(source, destination)
        mode = "hard_link"
    except OSError:
        shutil.copy2(source, destination)
        mode = "copy_fallback"
        if _sha256_file(destination) != expected_sha256:
            raise ProtocolError("Copied promoted checkpoint failed hash verification.")
    return mode


def _control_lock(config: UniformBV2PromotionConfig, *, bank_lock_hash: str) -> dict[str, object]:
    payload = {
        "schema_version": "midogpp_uniform_b_v2_equal_union_ps_control_v1",
        "control_id": "uniform_b_v2_equal_union_ps",
        "expert_bank_lock_hash": bank_lock_hash,
        "sampler_family": CONTROL_SAMPLER_FAMILY,
        "composition": "all_eligible_source_experts_equal_union_fixed_total",
        "total_per_class": config.control_total_per_class,
        "training_seeds": list(config.training_seeds),
        "generation_seeds": list(config.generation_seeds),
        "replicate_policy": "report_each_replication_and_predeclared_mean_no_seed_selection",
        "candidate_sources_by_target": {
            target: list(legal_routing_sources(target)) for target in CENTERS
        },
        "source_budget_per_class": 128,
        "target_expert_excluded": True,
        "target_conditioned_source_weighting": False,
        "canonical_control_for_future_routing": True,
    }
    payload["control_lock_hash"] = stable_hash(payload)
    return payload


def _promotion_protocol(
    config: UniformBV2PromotionConfig,
    bank: Mapping[str, object],
    control: Mapping[str, object],
) -> dict[str, object]:
    payload = {
        "schema_version": "midogpp_uniform_b_v2_expert_bank_promotion_protocol_v1",
        "experiment_id": EXPERIMENT_ID,
        "claim_scope": CLAIM_SCOPE,
        "promotion_review_id": PROMOTION_REVIEW_ID,
        "promotion_contract_hash": config.contract_hash,
        "source_protocol_hash": config.expected_source_protocol_hash,
        "bank_lock_hash": bank["bank_lock_hash"],
        "control_lock_hash": control["control_lock_hash"],
        "independently_trained_source_experts": True,
        "all_27_experts_retained": True,
        "individual_expert_or_seed_selection": False,
        "routing_policy_selected": False,
        "routing_quality_claimed": False,
        "may_feed_deployable_selection": True,
    }
    payload["protocol_hash"] = stable_hash(payload)
    return payload


def _review_snapshot(
    config: UniformBV2PromotionConfig,
    audit: Mapping[str, object],
) -> dict[str, object]:
    payload = {
        "schema_version": "midogpp_uniform_b_v2_expert_bank_promotion_review_v1",
        **dict(config.promotion_review),
        "observed_ps_mean_bacc": audit["ps_mean_bacc"],
        "observed_minimum_ps_training_seed_mean_bacc": audit["minimum_ps_training_seed_mean_bacc"],
        "observed_ps_minus_p0": audit["ps_minus_p0"],
        "observed_posterior_ceiling_minus_ps": audit["posterior_ceiling_minus_ps"],
        "observed_checkpoint_records": audit["checkpoint_records"],
        "observed_sampler_records": audit["sampler_records"],
    }
    payload["review_hash"] = stable_hash(payload)
    return payload


def _promotion_decision(
    config: UniformBV2PromotionConfig,
    audit: Mapping[str, object],
    bank: Mapping[str, object],
    control: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_uniform_b_v2_expert_bank_promotion_decision_v1",
        "status": "PASS",
        "decision": PROMOTION_DECISION,
        "publication_state": PUBLICATION_STATE,
        "promotion_review_id": PROMOTION_REVIEW_ID,
        "source_protocol_hash": config.expected_source_protocol_hash,
        "bank_lock_hash": bank["bank_lock_hash"],
        "control_lock_hash": control["control_lock_hash"],
        "expert_count": N_EXPERTS,
        "source_centers": len(CENTERS),
        "training_replicates_per_source": len(TRAINING_SEEDS),
        "ps_mean_bacc": audit["ps_mean_bacc"],
        "minimum_ps_training_seed_mean_bacc": audit["minimum_ps_training_seed_mean_bacc"],
        "ps_minus_p0": audit["ps_minus_p0"],
        "posterior_ceiling_minus_ps": audit["posterior_ceiling_minus_ps"],
        "whole_bank_promoted_without_expert_or_seed_selection": True,
        "canonical_control": "uniform_b_v2_equal_union_ps",
        "routing_quality_claimed": False,
        "may_feed_deployable_selection": True,
    }


def _promotion_report(audit: Mapping[str, object]) -> str:
    return "\n".join(
        (
            "# Uniform-B v2 Routing-Authorized Expert Bank",
            "",
            f"Decision: `{PROMOTION_DECISION}`.",
            "",
            f"- Frozen experts: `{N_EXPERTS}` (9 centers x 3 training seeds)",
            f"- PS mean BACC: `{float(audit['ps_mean_bacc']):.6f}`",
            f"- Minimum training-seed PS mean BACC: `{float(audit['minimum_ps_training_seed_mean_bacc']):.6f}`",
            f"- PS minus P0: `{float(audit['ps_minus_p0']):.6f}`",
            "- Individual expert/seed selection: `false`",
            "- Canonical future-routing control: equal-union PS at a fixed total budget",
            "- Routing-quality claim at promotion: `false`",
            "",
            "The source-inner evidence is consumed once for whole-bank adoption. Future",
            "routing experiments must compare against the frozen equal-union control and",
            "must not reuse these results to choose an expert, seed, or routing policy.",
            "",
        )
    )


def _gate_rows(audit: Mapping[str, object]) -> list[dict[str, object]]:
    return [
        {"gate": "ps_mean_bacc", "observed": audit["ps_mean_bacc"], "required": ">=0.70", "status": "PASS"},
        {"gate": "minimum_ps_training_seed_mean_bacc", "observed": audit["minimum_ps_training_seed_mean_bacc"], "required": ">=0.75", "status": "PASS"},
        {"gate": "ps_minus_p0", "observed": audit["ps_minus_p0"], "required": ">=0.005", "status": "PASS"},
        {"gate": "posterior_ceiling_minus_ps", "observed": audit["posterior_ceiling_minus_ps"], "required": "<=0.01", "status": "PASS"},
        {"gate": "all_sampler_realizations_full_shrinkage", "observed": True, "required": True, "status": "PASS"},
        {"gate": "identity_overlap_failures", "observed": 0, "required": 0, "status": "PASS"},
        {"gate": "classifier_convergence_failures", "observed": 0, "required": 0, "status": "PASS"},
    ]


def _write_content_index(root: Path, *, known_hashes: Mapping[str, str]) -> None:
    excluded = {
        "manifests/content_index.json",
        "reports/run_state.json",
        "reports/validation_report.json",
    }
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        records.append(
            {
                "relative_path": relative,
                "sha256": (
                    known_hashes[relative]
                    if relative in known_hashes
                    else _sha256_file(path)
                ),
                "size_bytes": path.stat().st_size,
            }
        )
    payload = {
        "schema_version": "midogpp_uniform_b_v2_expert_bank_content_index_v1",
        "records": records,
    }
    payload["content_hash"] = stable_hash(payload)
    write_json(root / "manifests/content_index.json", payload)


def _verify_source_content_index(root: Path) -> None:
    payload = _read_json(root / "manifests/content_index.json")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ProtocolError("Completed v2 content index is invalid.")
    for row in records:
        if not isinstance(row, Mapping):
            raise ProtocolError("Completed v2 content-index row is invalid.")
        member = _safe_member(root, str(row.get("relative_path", "")))
        if (
            not member.is_file()
            or member.stat().st_size != int(row.get("size_bytes", -1))
            or _sha256_file(member) != row.get("sha256")
        ):
            raise ProtocolError(f"Completed v2 evidence member drifted: {member.name}.")


def _resolve_device(configured: str) -> str:
    value = os.environ.get("MIDOGPP_V2_PROMOTION_DEVICE", configured).strip()
    if value == "cpu":
        return value
    if not value.startswith("cuda:"):
        raise ProtocolError("Promotion device must be cpu or explicit cuda:N.")
    index = int(value.split(":", 1)[1])
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        raise ProtocolError(f"Configured promotion device is unavailable: {value}.")
    return value


def _write_state(root: Path, status: str) -> None:
    write_json(
        root / "reports/run_state.json",
        {
            "schema_version": "midogpp_uniform_b_v2_expert_bank_run_state_v1",
            "status": status,
            "claim_scope": CLAIM_SCOPE,
        },
    )


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError(f"Promotion JSON must be an object: {path}.")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _safe_member(root: Path, relative: str) -> Path:
    resolved_root = root.resolve()
    member = (resolved_root / relative).resolve()
    if member == resolved_root or not member.is_relative_to(resolved_root):
        raise ProtocolError("Promotion input path escapes the source artifact.")
    return member


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ("audit_source_bundle", "run_promotion")
