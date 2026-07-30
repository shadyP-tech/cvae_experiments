"""Independent fail-closed validation for the canonical-B adaptation pilot."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping

from ....real_features.classifier_reference.artifacts import stable_hash
from ....real_features.classifier_reference.protocol import ProtocolError
from ...models import ClassConditionedCVAE
from .config import PILOT_ARMS, PILOT_CENTERS, PILOT_GENERATION_SEEDS, PILOT_TRAINING_SEEDS
from .conservative_prior import CONDITIONAL_PRIOR_FAMILY
from .step_training import model_state_hash


INDEX_EXCLUSIONS = frozenset(
    {
        "manifests/content_index.json",
        "reports/run_state.json",
        "reports/validation_report.json",
    }
)


def validate_final_bundle(root: str | Path) -> dict[str, object]:
    import numpy as np
    import torch

    path = Path(root).resolve()
    errors: list[str] = []
    required = (
        "config.resolved.yaml",
        "provenance/input_artifacts.json",
        "manifests/frozen_protocol.json",
        "manifests/content_index.json",
        "tables/frame_preparation.csv",
        "tables/job_inventory.csv",
        "tables/pilot_metrics.csv",
        "tables/heldout_predictions.csv",
        "tables/case_class_sampling_audit.csv",
        "reports/a_prefix_bridge.json",
        "reports/leakage_provenance_report.json",
        "reports/pilot_decision.json",
        "reports/runtime_summary.json",
        "reports/run_state.json",
    )
    for relative in required:
        if not (path / relative).is_file():
            errors.append(f"missing required member: {relative}")
    if errors:
        return _report(errors)

    protocol = _json(path / "manifests/frozen_protocol.json")
    observed_protocol_hash = str(protocol.pop("protocol_hash", ""))
    if observed_protocol_hash != stable_hash(protocol):
        errors.append("frozen protocol self-hash mismatch")
    if (
        protocol.get("claim_scope") != "diagnostic_only"
        or protocol.get("may_export_recipe_lock") is not False
        or protocol.get("may_feed_expert_bank") is not False
    ):
        errors.append("frozen protocol claim firewall mismatch")
    bridge = _json(path / "reports/a_prefix_bridge.json")
    if (
        bridge.get("status") != "PASS"
        or int(bridge.get("n_rows", -1)) != 9648
        or float(bridge.get("minimum_cosine", 0.0)) < 0.99999
        or float(bridge.get("maximum_relative_l2", 1.0)) > 0.001
    ):
        errors.append("canonical A/B bridge failed")

    frames = _csv(path / "tables/frame_preparation.csv")
    frame_keys = [(row["center"], row["arm"]) for row in frames]
    expected_frames = {(center, arm) for center in PILOT_CENTERS for arm in PILOT_ARMS}
    if len(frames) != 12 or set(frame_keys) != expected_frames or len(frame_keys) != len(set(frame_keys)):
        errors.append("frame coverage is not exact 4x3")
    for row in frames:
        try:
            fit_cases = set(json.loads(row["fit_cases"]))
            eval_cases = set(json.loads(row["eval_cases"]))
            if fit_cases.intersection(eval_cases):
                errors.append(f"case overlap in frame {row['center']}/{row['arm']}")
            arrays = np.load(row["prepared_path"], allow_pickle=False)
            if set(arrays.files) != {
                "x_fit", "y_fit", "case_fit", "sample_fit",
                "x_eval", "y_eval", "case_eval", "sample_eval",
            }:
                errors.append(f"prepared-array schema mismatch for {row['center']}/{row['arm']}")
            if arrays["x_fit"].shape[1] != 128 or arrays["x_eval"].shape[1] != 128:
                errors.append(f"prepared dimension mismatch for {row['center']}/{row['arm']}")
            if set(arrays["case_fit"].tolist()).intersection(arrays["case_eval"].tolist()):
                errors.append(f"prepared case leakage for {row['center']}/{row['arm']}")
            frame_state = _json(
                path / "prepared" / row["center"] / row["arm"] / "frame_state.json"
            )
            if stable_hash(frame_state) != row["frame_hash"]:
                errors.append(f"frame-state hash mismatch for {row['center']}/{row['arm']}")
            if frame_state.get("may_feed_expert_bank") is not False:
                errors.append(f"frame promotion firewall mismatch for {row['center']}/{row['arm']}")
        except Exception as exc:
            errors.append(f"invalid frame evidence {row.get('center')}/{row.get('arm')}: {exc}")

    jobs = _csv(path / "tables/job_inventory.csv")
    expected_jobs = {
        (center, arm, str(seed))
        for center in PILOT_CENTERS
        for arm in PILOT_ARMS
        for seed in PILOT_TRAINING_SEEDS
    }
    job_keys = [(row["center"], row["arm"], row["training_seed"]) for row in jobs]
    if len(jobs) != 36 or set(job_keys) != expected_jobs or len(job_keys) != len(set(job_keys)):
        errors.append("job coverage is not exact 4x3x3")
    for row in jobs:
        try:
            if int(row["optimizer_steps"]) != 1000:
                errors.append(f"wrong step count for {row['training_key_hash']}")
            checkpoint = Path(row["checkpoint_path"])
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if payload["training_key_hash"] != row["training_key_hash"]:
                errors.append(f"checkpoint training-key mismatch for {row['training_key_hash']}")
            if payload["schedule_hash"] != row["schedule_hash"]:
                errors.append(f"checkpoint schedule mismatch for {row['training_key_hash']}")
            model = ClassConditionedCVAE(input_dim=128, hidden_dim=512, latent_dim=32)
            model.load_state_dict(payload["state_dict"], strict=True)
            if model_state_hash(model) != row["checkpoint_hash"]:
                errors.append(f"checkpoint hash mismatch for {row['training_key_hash']}")
            diagnostics = payload["diagnostics"]
            if not diagnostics or int(diagnostics[-1]["step"]) != 1000:
                errors.append(f"checkpoint final-step diagnostic mismatch for {row['training_key_hash']}")
            prior = _json(path / "priors" / f"{row['training_key_hash']}.json")
            prior_payload = {
                key: prior[key]
                for key in (
                    "requested_family", "realized_family", "means", "variances",
                    "n_rows_by_class", "n_cases_by_class", "condition_number_by_class",
                    "fallback_reason", "source_state_hash",
                )
            }
            prior_payload["schema_version"] = "midogpp_b_adaptation_shrunk_prior_v1"
            if stable_hash(prior_payload) != prior["state_hash"]:
                errors.append(f"prior-state hash mismatch for {row['training_key_hash']}")
            if prior.get("may_feed_model_recipe") is not False:
                errors.append(f"prior promotion firewall mismatch for {row['training_key_hash']}")
        except Exception as exc:
            errors.append(f"invalid job/checkpoint/prior {row.get('training_key_hash')}: {exc}")

    for center in PILOT_CENTERS:
        for seed in map(str, PILOT_TRAINING_SEEDS):
            paired = [row for row in jobs if row["center"] == center and row["training_seed"] == seed]
            for field in ("schedule_hash", "initialization_hash", "posterior_stream_hash"):
                if len({row[field] for row in paired}) != 1:
                    errors.append(f"paired {field} mismatch for center={center},seed={seed}")

    metrics = _csv(path / "tables/pilot_metrics.csv")
    expected_metric_keys = set()
    for center, arm, seed in expected_jobs:
        expected_metric_keys.add((center, arm, seed, "real", ""))
        expected_metric_keys.add((center, arm, seed, "decode_mu", ""))
        for generation_seed in map(str, PILOT_GENERATION_SEEDS):
            for role in ("posterior_sample", "prior_standard", "prior_conditional"):
                expected_metric_keys.add((center, arm, seed, role, generation_seed))
    metric_keys = [
        (
            row["center"], row["arm"], row["training_seed"],
            row["representation_role"], row["generation_seed"],
        )
        for row in metrics
    ]
    if len(metrics) != 396 or set(metric_keys) != expected_metric_keys or len(metric_keys) != len(set(metric_keys)):
        errors.append("metric role/seed coverage is not exact")
    for row in metrics:
        tp, fn, tn, fp = (int(row[key]) for key in ("tp", "fn", "tn", "fp"))
        recall = tp / max(1, tp + fn)
        specificity = tn / max(1, tn + fp)
        bacc = 0.5 * (recall + specificity)
        if max(
            abs(float(row["positive_recall"]) - recall),
            abs(float(row["specificity"]) - specificity),
            abs(float(row["bacc"]) - bacc),
        ) > 1e-10:
            errors.append(f"metric/confusion mismatch for {metric_keys[len(errors) % len(metric_keys)]}")
            break
        role = row["representation_role"]
        if role == "real" and row["target_labels_used_for_scoring_only"] != "True":
            errors.append("real-row label semantics mismatch")
        if role in {"decode_mu", "posterior_sample"} and row["heldout_class_used_for_cvae_conditioning"] != "True":
            errors.append("conditioned-row label semantics mismatch")
        if role.startswith("prior_") and row["truth_semantics"] != "requested_synthetic_class":
            errors.append("prior truth semantics mismatch")

    predictions = _csv(path / "tables/heldout_predictions.csv")
    predicted_roles = {"real", "decode_mu", "posterior_sample"}
    if not predictions or {row["representation_role"] for row in predictions} != predicted_roles:
        errors.append("heldout prediction role coverage mismatch")
    by_metric: dict[tuple[str, str, str, str, str], list[Mapping[str, str]]] = {}
    for row in predictions:
        key = (
            row["center"], row["arm"], row["training_seed"],
            row["representation_role"], row["generation_seed"],
        )
        by_metric.setdefault(key, []).append(row)
    for row in metrics:
        key = (
            row["center"], row["arm"], row["training_seed"],
            row["representation_role"], row["generation_seed"],
        )
        if row["representation_role"] not in predicted_roles:
            continue
        evidence = by_metric.get(key, [])
        counts = {
            "tp": sum(r["y_true"] == "1" and r["y_pred"] == "1" for r in evidence),
            "fn": sum(r["y_true"] == "1" and r["y_pred"] == "0" for r in evidence),
            "tn": sum(r["y_true"] == "0" and r["y_pred"] == "0" for r in evidence),
            "fp": sum(r["y_true"] == "0" and r["y_pred"] == "1" for r in evidence),
        }
        if any(int(row[name]) != value for name, value in counts.items()):
            errors.append(f"prediction/confusion mismatch for {key}")

    schedules = _csv(path / "tables/case_class_sampling_audit.csv")
    for row in jobs:
        evidence = [
            item for item in schedules
            if item["center"] == row["center"]
            and item["arm"] == row["arm"]
            and item["training_seed"] == row["training_seed"]
        ]
        for label in ("0", "1"):
            total = sum(
                int(item["exposure"])
                for item in evidence
                if item["group"].startswith(label + ":")
            )
            if total != 64000:
                errors.append(f"schedule exposure mismatch for {row['training_key_hash']}/class{label}")

    try:
        from .runner import _decision

        typed_metrics = [_typed_metric(row) for row in metrics]
        typed_jobs = [_typed_job(row) for row in jobs]
        recomputed = _decision(typed_metrics, typed_jobs, minimum_real_bacc=0.60)
        if recomputed != _json(path / "reports/pilot_decision.json"):
            errors.append("pilot decision does not match independent recomputation")
    except Exception as exc:
        errors.append(f"decision recomputation failed: {exc}")

    index = _json(path / "manifests/content_index.json")
    indexed = {row["path"]: row for row in index.get("files", [])}
    actual = {
        str(member.relative_to(path)): member
        for member in path.rglob("*")
        if member.is_file()
        and str(member.relative_to(path)) not in INDEX_EXCLUSIONS
        and ".tmp" not in member.name
    }
    if set(indexed) != set(actual):
        errors.append("content-index coverage mismatch or stale unreferenced members")
    for relative, member in actual.items():
        record = indexed.get(relative, {})
        if (
            record.get("sha256") != hashlib.sha256(member.read_bytes()).hexdigest()
            or int(record.get("size_bytes", -1)) != member.stat().st_size
        ):
            errors.append(f"content-index hash/size mismatch: {relative}")

    return _report(errors)


def _typed_metric(row: Mapping[str, str]) -> dict[str, object]:
    output: dict[str, object] = dict(row)
    for key in (
        "training_seed", "tp", "fn", "tn", "fp", "n_positive", "n_negative",
    ):
        output[key] = int(row[key])
    for key in (
        "bacc", "positive_recall", "specificity", "preservation_ratio", "real_reference_bacc",
    ):
        output[key] = float(row[key])
    return output


def _typed_job(row: Mapping[str, str]) -> dict[str, object]:
    output: dict[str, object] = dict(row)
    output["training_seed"] = int(row["training_seed"])
    output["optimizer_steps"] = int(row["optimizer_steps"])
    return output


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProtocolError(f"Expected JSON object: {path}")
    return value


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _report(errors: list[str]) -> dict[str, object]:
    return {
        "schema_version": "midogpp_b_adaptation_validation_v2",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "claim_scope": "diagnostic_only",
        "may_feed_expert_bank": False,
    }


__all__ = ("INDEX_EXCLUSIONS", "validate_final_bundle")
