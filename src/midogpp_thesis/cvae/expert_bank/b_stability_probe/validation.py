"""Independent fail-closed validation for the B-block stability diagnostic."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping

from ....real_features.classifier_reference.artifacts import stable_hash
from ...models import ClassConditionedCVAE
from ..b_adaptation_pilot.step_training import model_state_hash
from .config import (
    CENTERS,
    PREDECESSOR_HASHES,
    PREDECESSOR_PROTOCOL_HASH,
    READOUTS,
    TRAINING_SEEDS,
    load_stability_config,
)


INDEX_EXCLUSIONS = frozenset(
    {
        "manifests/content_index.json",
        "reports/validation_report.json",
        "reports/run_state.json",
        ".run.lock",
    }
)


def validate_stability_bundle(root: str | Path) -> dict[str, object]:
    """Validate lineage, replay, arithmetic, coverage, and claim firewalls."""

    import torch

    path = Path(root).resolve()
    errors: list[str] = []
    required = (
        "config.resolved.yaml",
        "provenance/input_artifacts.json",
        "manifests/frozen_protocol.json",
        "manifests/content_index.json",
        "reports/predecessor_audit.json",
        "reports/stability_decision.json",
        "reports/runtime_summary.json",
        "reports/leakage_provenance_report.json",
        "reports/run_state.json",
        "tables/frozen_comparators.csv",
        "tables/job_inventory.csv",
        "tables/stability_metrics.csv",
        "tables/heldout_predictions.csv",
        "tables/case_class_sampling_audit.csv",
        "tables/gate_audit.csv",
    )
    for relative in required:
        if not (path / relative).is_file():
            errors.append(f"missing required member: {relative}")
    if errors:
        return _report(errors)

    try:
        config = load_stability_config(path / "config.resolved.yaml")
    except Exception as exc:
        return _report([f"resolved config failed exact validation: {exc}"])
    if config.artifact_root.resolve() != path:
        errors.append("resolved artifact root does not match bundle root")

    protocol = _json(path / "manifests/frozen_protocol.json")
    observed_protocol_hash = str(protocol.pop("protocol_hash", ""))
    if observed_protocol_hash != stable_hash(protocol):
        errors.append("frozen protocol self-hash mismatch")
    snapshots = protocol.get("workspace_snapshot_hashes", {})
    if not isinstance(snapshots, Mapping) or (
        snapshots.get("config_resolved_sha256")
        != _file_sha256(path / "config.resolved.yaml")
        or snapshots.get("input_artifacts_sha256")
        != _file_sha256(path / "provenance/input_artifacts.json")
    ):
        errors.append("workspace snapshot hashes do not match")
    if (
        protocol.get("experiment")
        != "uniform_b_block_tail_average_stability_probe_v1"
        or protocol.get("predecessor", {}).get("predecessor_protocol_hash")
        != PREDECESSOR_PROTOCOL_HASH
        or protocol.get("test_or_validation_split_used") is not False
        or protocol.get("claim_scope") != "diagnostic_only"
        or protocol.get("confirmation_eligible") is not False
        or protocol.get("may_export_recipe_lock") is not False
        or protocol.get("may_feed_expert_bank") is not False
        or protocol.get("may_feed_generation") is not False
        or protocol.get("may_feed_routing") is not False
    ):
        errors.append("frozen protocol identity or claim firewall mismatch")
    expected_components = _component_hashes()
    if protocol.get("component_hashes") != expected_components:
        errors.append("frozen protocol component hashes do not match source")

    predecessor_audit = _json(path / "reports/predecessor_audit.json")
    if (
        predecessor_audit.get("status") != "PASS"
        or predecessor_audit.get("predecessor_protocol_hash")
        != PREDECESSOR_PROTOCOL_HASH
        or predecessor_audit.get("predecessor_file_hashes")
        != PREDECESSOR_HASHES
        or int(predecessor_audit.get("bound_job_count", -1)) != 12
        or int(predecessor_audit.get("bound_comparator_count", -1)) != 36
        or predecessor_audit.get("confirmation_eligible") is not False
    ):
        errors.append("predecessor audit is not exact")
    for relative, expected in PREDECESSOR_HASHES.items():
        member = config.predecessor_root / relative
        if not member.is_file() or _file_sha256(member) != expected:
            errors.append(f"predecessor file changed: {relative}")

    run_state = _json(path / "reports/run_state.json")
    if run_state.get("status") not in {"VALIDATING", "COMPLETE"}:
        errors.append(f"invalid success-path run status: {run_state.get('status')!r}")
    leakage = _json(path / "reports/leakage_provenance_report.json")
    if (
        leakage.get("status") != "PASS"
        or leakage.get("canonical_validation_or_test_features_used") is not False
        or leakage.get("heldout_labels_used_for_classifier_fit") is not False
        or leakage.get("heldout_labels_used_for_cvae_fit") is not False
        or leakage.get("prior_or_generation_evaluated") is not False
        or leakage.get("heldout_labels_used_for_confirmation") is not False
    ):
        errors.append("leakage/label-role report mismatch")

    comparators = _csv(path / "tables/frozen_comparators.csv")
    expected_keys = {
        (center, arm, str(seed))
        for center in CENTERS
        for arm in (
            "a_global_pca128",
            "b_joint_pca128",
            "b_block_pca96_32",
        )
        for seed in TRAINING_SEEDS
    }
    comparator_keys = [
        (row["center"], row["arm"], row["training_seed"]) for row in comparators
    ]
    if (
        len(comparators) != 36
        or set(comparator_keys) != expected_keys
        or len(comparator_keys) != len(set(comparator_keys))
        or any(row["representation_role"] != "decode_mu" for row in comparators)
    ):
        errors.append("frozen comparator coverage is not exact")
    predecessor_metrics = _csv(
        config.predecessor_root / "tables/pilot_metrics.csv"
    )
    expected_comparators = [
        row
        for row in predecessor_metrics
        if (
            row["center"],
            row["arm"],
            row["training_seed"],
        )
        in expected_keys
        and row["representation_role"] == "decode_mu"
        and row["generation_seed"] == ""
    ]
    if _canonical_rows(comparators) != _canonical_rows(expected_comparators):
        errors.append("frozen comparators differ from hash-bound v2")

    jobs = _csv(path / "tables/job_inventory.csv")
    expected_job_keys = {
        (center, str(seed)) for center in CENTERS for seed in TRAINING_SEEDS
    }
    job_keys = [(row["center"], row["training_seed"]) for row in jobs]
    if (
        len(jobs) != 12
        or set(job_keys) != expected_job_keys
        or len(job_keys) != len(set(job_keys))
    ):
        errors.append("job coverage is not exact 4x3")
    predecessor_jobs = _csv(
        config.predecessor_root / "tables/job_inventory.csv"
    )
    predecessor_by_key = {
        (row["center"], row["training_seed"]): row
        for row in predecessor_jobs
        if row["arm"] == "b_block_pca96_32"
    }
    for row in jobs:
        try:
            key = (row["center"], row["training_seed"])
            source = predecessor_by_key[key]
            if (
                row["arm"] != "b_block_pca96_32"
                or row["device"] != source["device"]
                or row["predecessor_training_key_hash"]
                != source["training_key_hash"]
                or row["predecessor_checkpoint_hash"] != source["checkpoint_hash"]
                or row["initialization_hash"] != source["initialization_hash"]
                or row["schedule_hash"] != source["schedule_hash"]
                or row["posterior_stream_hash"]
                != source["posterior_stream_hash"]
                or row["endpoint_hash"] != source["checkpoint_hash"]
                or row["endpoint_replay_exact"] != "True"
                or int(row["optimizer_steps"]) != 1000
                or int(row["tail_state_count"]) != 250
                or int(row["tail_start_step"]) != 751
                or int(row["tail_end_step"]) != 1000
            ):
                errors.append(f"job replay contract mismatch for {key}")
            checkpoint = Path(row["checkpoint_path"]).resolve()
            if checkpoint.parent != path / "checkpoints":
                errors.append(f"checkpoint escaped bundle for {key}")
                continue
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if (
                payload["schema_version"]
                != "midogpp_b_tail_average_checkpoint_v1"
                or payload["training_key_hash"] != row["training_key_hash"]
                or payload["endpoint_hash"] != row["endpoint_hash"]
                or payload["averaged_hash"] != row["averaged_hash"]
                or payload["averaging_derivation_hash"]
                != row["averaging_derivation_hash"]
                or tuple(int(value) for value in payload["tail_steps"])
                != tuple(range(751, 1001))
                or int(payload["tail_state_count"]) != 250
            ):
                errors.append(f"checkpoint metadata mismatch for {key}")
                continue
            endpoint = ClassConditionedCVAE(
                input_dim=128, hidden_dim=512, latent_dim=32
            )
            averaged = ClassConditionedCVAE(
                input_dim=128, hidden_dim=512, latent_dim=32
            )
            endpoint.load_state_dict(payload["endpoint_state_dict"], strict=True)
            averaged.load_state_dict(payload["averaged_state_dict"], strict=True)
            if (
                model_state_hash(endpoint) != row["endpoint_hash"]
                or model_state_hash(averaged) != row["averaged_hash"]
            ):
                errors.append(f"checkpoint state hash mismatch for {key}")
            expected_derivation = stable_hash(
                {
                    "schema_version": "midogpp_b_tail_average_derivation_v1",
                    "method": "uniform_fp32_online_parameter_mean_v1",
                    "update_timing": "after_optimizer_step",
                    "tail_steps": list(range(751, 1001)),
                    "tail_state_count": 250,
                    "endpoint_hash": row["endpoint_hash"],
                    "averaged_hash": row["averaged_hash"],
                }
            )
            if expected_derivation != row["averaging_derivation_hash"]:
                errors.append(f"averaging derivation hash mismatch for {key}")
        except Exception as exc:
            errors.append(
                f"invalid job/checkpoint {row.get('center')}/"
                f"{row.get('training_seed')}: {exc}"
            )
    checkpoint_files = list((path / "checkpoints").glob("*.pt"))
    job_files = list((path / "jobs").glob("*.json"))
    if len(checkpoint_files) != 12 or len(job_files) != 12:
        errors.append("unexpected checkpoint/job file count")

    metrics = _csv(path / "tables/stability_metrics.csv")
    expected_metric_keys = {
        (center, str(seed), readout)
        for center in CENTERS
        for seed in TRAINING_SEEDS
        for readout in READOUTS
    }
    metric_keys = [
        (row["center"], row["training_seed"], row["readout"]) for row in metrics
    ]
    if (
        len(metrics) != 24
        or set(metric_keys) != expected_metric_keys
        or len(metric_keys) != len(set(metric_keys))
    ):
        errors.append("metric coverage is not exact 4x3x2")
    for row in metrics:
        try:
            tp, fn, tn, fp = (int(row[key]) for key in ("tp", "fn", "tn", "fp"))
            recall = tp / max(1, tp + fn)
            specificity = tn / max(1, tn + fp)
            bacc = 0.5 * (recall + specificity)
            if (
                max(
                    abs(float(row["positive_recall"]) - recall),
                    abs(float(row["specificity"]) - specificity),
                    abs(float(row["bacc"]) - bacc),
                )
                > 1e-12
                or float(row["real_reference_bacc"]) < 0.60
                or row["heldout_labels_used_for_classifier_fit"] != "False"
                or row["heldout_labels_used_for_cvae_fit"] != "False"
                or row["heldout_labels_used_for_confirmation"] != "False"
                or row["prior_or_generation_metric"] != "False"
            ):
                errors.append(
                    f"metric arithmetic/semantics mismatch for "
                    f"{row['center']}/{row['training_seed']}/{row['readout']}"
                )
        except Exception as exc:
            errors.append(f"invalid metric row: {exc}")

    predictions = _csv(path / "tables/heldout_predictions.csv")
    grouped_predictions: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in predictions:
        grouped_predictions.setdefault(
            (row["center"], row["training_seed"], row["readout"]), []
        ).append(row)
        if (
            row["heldout_labels_used_for_classifier_fit"] != "False"
            or row["heldout_labels_used_for_cvae_fit"] != "False"
            or row["heldout_labels_used_for_confirmation"] != "False"
            or row["oracle_eligible"] != "False"
        ):
            errors.append("prediction label-role firewall mismatch")
            break
    metric_by_key = {
        (row["center"], row["training_seed"], row["readout"]): row
        for row in metrics
    }
    if set(grouped_predictions) != expected_metric_keys:
        errors.append("prediction coverage does not match metric coverage")
    for key, rows in grouped_predictions.items():
        counts = _confusion_from_rows(rows)
        metric = metric_by_key.get(key)
        if metric is None or any(
            int(metric[name]) != counts[name] for name in ("tp", "fn", "tn", "fp")
        ):
            errors.append(f"prediction/metric confusion mismatch for {key}")

    predecessor_predictions = _csv(
        config.predecessor_root / "tables/heldout_predictions.csv"
    )
    expected_endpoint_predictions = [
        {
            "center": row["center"],
            "training_seed": row["training_seed"],
            "sample_id": row["sample_id"],
            "case_id": row["case_id"],
            "y_true": row["y_true"],
            "y_pred": row["y_pred"],
        }
        for row in predecessor_predictions
        if row["arm"] == "b_block_pca96_32"
        and row["representation_role"] == "decode_mu"
        and (row["center"], row["training_seed"]) in expected_job_keys
    ]
    observed_endpoint_predictions = [
        {
            "center": row["center"],
            "training_seed": row["training_seed"],
            "sample_id": row["sample_id"],
            "case_id": row["case_id"],
            "y_true": row["y_true"],
            "y_pred": row["y_pred"],
        }
        for row in predictions
        if row["readout"] == READOUTS[0]
    ]
    if observed_endpoint_predictions != expected_endpoint_predictions:
        errors.append("terminal predictions do not exactly replay v2")

    schedule = _csv(path / "tables/case_class_sampling_audit.csv")
    predecessor_schedule = [
        row
        for row in _csv(
            config.predecessor_root / "tables/case_class_sampling_audit.csv"
        )
        if row["arm"] == "b_block_pca96_32"
        and (row["center"], row["training_seed"]) in expected_job_keys
    ]
    if _canonical_rows(schedule) != _canonical_rows(predecessor_schedule):
        errors.append("sampling audit does not exactly replay v2")

    decision = _json(path / "reports/stability_decision.json")
    try:
        from .runner import stability_decision

        recomputed = stability_decision(
            metrics,
            comparators,
            config.gates,
            endpoint_replay_exact=all(
                row.get("endpoint_replay_exact") == "True" for row in jobs
            ),
        )
        if stable_hash(decision) != stable_hash(recomputed):
            errors.append("decision does not independently recompute")
    except Exception as exc:
        errors.append(f"decision recomputation failed: {exc}")
    gates = _csv(path / "tables/gate_audit.csv")
    if len(gates) != 14 or any(
        row["passed"] not in {"True", "False"} for row in gates
    ):
        errors.append("gate audit coverage is not exact")
    if (
        decision.get("claim_scope") != "diagnostic_only"
        or decision.get("confirmation_eligible") is not False
        or decision.get("may_export_recipe_lock") is not False
        or decision.get("may_feed_expert_bank") is not False
        or decision.get("may_feed_generation") is not False
        or decision.get("may_feed_routing") is not False
        or decision.get("prior_or_generation_evaluated") is not False
        or decision.get("next_step_if_pass")
        != "separately_reviewed_b_block_prior_only_replay"
    ):
        errors.append("decision claim firewall mismatch")

    index = _json(path / "manifests/content_index.json")
    indexed = {
        str(row["path"]): (str(row["sha256"]), int(row["size_bytes"]))
        for row in index.get("files", [])
        if isinstance(row, Mapping)
    }
    actual = {
        str(member.relative_to(path)): (_file_sha256(member), member.stat().st_size)
        for member in path.rglob("*")
        if member.is_file()
        and str(member.relative_to(path)) not in INDEX_EXCLUSIONS
        and ".tmp" not in member.name
    }
    if indexed != actual:
        errors.append("content index differs from bundle contents")

    return _report(errors)


def _component_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {
        name: _file_sha256(directory / name)
        for name in (
            "__init__.py",
            "config.py",
            "runner.py",
            "tail_training.py",
            "validation.py",
        )
    }


def _canonical_rows(rows: list[Mapping[str, object]]) -> list[str]:
    return sorted(json.dumps(dict(row), sort_keys=True) for row in rows)


def _confusion_from_rows(rows: list[Mapping[str, object]]) -> dict[str, int]:
    pairs = [(int(row["y_true"]), int(row["y_pred"])) for row in rows]
    return {
        "tp": sum(t == 1 and p == 1 for t, p in pairs),
        "fn": sum(t == 1 and p == 0 for t, p in pairs),
        "tn": sum(t == 0 and p == 0 for t, p in pairs),
        "fp": sum(t == 0 and p == 1 for t, p in pairs),
    }


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _report(errors: list[str]) -> dict[str, object]:
    return {
        "schema_version": "midogpp_b_tail_average_validation_report_v1",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "checks": {
            "lineage_hashes": "PASS" if not errors else "SEE_ERRORS",
            "terminal_control_replay": "PASS" if not errors else "SEE_ERRORS",
            "tail_average_derivation": "PASS" if not errors else "SEE_ERRORS",
            "metric_prediction_reconciliation": (
                "PASS" if not errors else "SEE_ERRORS"
            ),
            "decision_recomputation": "PASS" if not errors else "SEE_ERRORS",
            "claim_firewall": "PASS" if not errors else "SEE_ERRORS",
        },
    }


__all__ = ("INDEX_EXCLUSIONS", "validate_stability_bundle")
