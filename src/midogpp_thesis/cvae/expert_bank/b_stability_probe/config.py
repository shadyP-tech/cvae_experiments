"""Fail-closed configuration for the bounded B-block stability probe."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

from ....real_features.classifier_reference.protocol import ProtocolError


SCHEMA = "midogpp_uniform_b_block_tail_average_stability_probe_v1"
IDENTITY = "uniform_b_block_tail_average_stability_probe_v1"
CENTERS = ("2", "5", "6", "9")
TRAINING_SEEDS = (17, 42, 101)
READOUTS = ("terminal_step_1000", "tail_average_steps_751_1000")
TAIL_STEPS = tuple(range(751, 1001))
PREDECESSOR_PROTOCOL_HASH = "4a65f62dfeae9914"
PREDECESSOR_HASHES = {
    "manifests/frozen_protocol.json": "fb9372a318f5362f56509c40c60401b014e8e937e696c82be7433682e7b8a50c",
    "manifests/content_index.json": "cd971e94f72eb232d5b8759533ab2ce1ac41963b0e8448809826c34128c73ae1",
    "reports/pilot_decision.json": "abb2dcb0ae71cbd035ef1b3efcb9fc2cf4fb52d36c09305fad97259bd0b80344",
    "reports/validation_report.json": "ab624a458f2f7bb14aab256aadf07c7aa7886cbb26523277957a316dd0ec8fb7",
    "reports/run_state.json": "43c9734c4a381fae6266de406c12f15f3b18aa07b685278833cc6a40418d2e70",
    "tables/job_inventory.csv": "cef52ffc64c6d7e3e213d3b9692b757c360e85fb46aa0afad13ec3dcab871ed2",
    "tables/pilot_metrics.csv": "b9fa980d0b30cc4f8096dd369c4aab56ce44a7b50a0c02e75243c2819faf898f",
    "tables/heldout_predictions.csv": "8504512bcec15482a01698310cc8921648d40e3201030924221241588f7a77cd",
    "tables/case_class_sampling_audit.csv": "8c48fdd84a74a792bfe025eed5a2bf7f71cd74666e62ef702001e7976a1b2398",
}


@dataclass(frozen=True)
class StabilityConfig:
    name: str
    artifact_root: Path
    predecessor_root: Path
    centers: tuple[str, ...]
    training_seeds: tuple[int, ...]
    devices: tuple[str, ...]
    cpu_threads_per_worker: int
    arm: str
    optimizer_steps: int
    batch_size: int
    hidden_dim: int
    latent_dim: int
    learning_rate: float
    weight_decay: float
    beta_final: float
    kl_warmup_steps: int
    gradient_clip_norm: float
    tail_steps: tuple[int, ...]
    classifier_c: float
    minimum_real_bacc: float
    gates: Mapping[str, float]
    lineage: Mapping[str, object]
    code_version: str


def load_stability_config(path: str | Path) -> StabilityConfig:
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Stability-probe configuration requires PyYAML.") from exc

    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ProtocolError("Stability-probe config must be a mapping.")
    experiment = _mapping(payload, "experiment")
    inputs = _mapping(payload, "inputs")
    run = _mapping(payload, "run")
    replay = _mapping(payload, "frozen_replay")
    averaging = _mapping(payload, "tail_averaging")
    evaluation = _mapping(payload, "evaluation")
    decision = _mapping(payload, "decision")
    lineage = _mapping(payload, "lineage")
    claim = _mapping(payload, "claim_boundary")

    exact_identity = (
        (experiment.get("schema_version"), SCHEMA, "schema_version"),
        (experiment.get("name"), IDENTITY, "name"),
        (experiment.get("code_version"), IDENTITY, "code_version"),
    )
    for observed, expected, name in exact_identity:
        if observed != expected:
            raise ProtocolError(
                f"Stability-probe {name} must remain frozen at {expected!r}."
            )
    if tuple(str(v) for v in run.get("centers", ())) != CENTERS:
        raise ProtocolError("Stability-probe centers must be 2,5,6,9.")
    if tuple(int(v) for v in run.get("training_seeds", ())) != TRAINING_SEEDS:
        raise ProtocolError("Stability-probe training seeds must be 17,42,101.")
    if tuple(str(v) for v in run.get("devices", ())) != ("cuda:0", "cuda:1"):
        raise ProtocolError("Stability probe requires the two A5000 devices.")
    if int(run.get("cpu_threads_per_worker", -1)) != 1:
        raise ProtocolError("Stability-probe GPU workers require one CPU thread.")

    exact_replay: dict[str, object] = {
        "arm": "b_block_pca96_32",
        "predecessor_protocol_hash": PREDECESSOR_PROTOCOL_HASH,
        "representation_dim": 128,
        "block_pca_dims": [96, 32],
        "pca_solver": "randomized",
        "pca_random_state": 0,
        "pca_n_oversamples": 10,
        "pca_iterated_power": 4,
        "whiten": False,
        "post_fit_reweighting": False,
        "case_split_fraction": 0.20,
        "case_split_seed": 2718,
        "batch_policy": "class_to_case_to_row_balanced_with_replacement_v1",
        "optimizer_steps": 1000,
        "batch_size": 128,
        "hidden_dim": 512,
        "latent_dim": 32,
        "num_hidden_layers": 2,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "beta_final": 0.001,
        "kl_warmup_steps": 250,
        "gradient_clip_norm": 5.0,
    }
    if dict(replay) != exact_replay:
        raise ProtocolError("Stability-probe replay contract is not exact.")

    exact_averaging: dict[str, object] = {
        "method": "uniform_fp32_online_parameter_mean_v1",
        "update_timing": "after_optimizer_step",
        "start_step": 751,
        "end_step": 1000,
        "stride": 1,
        "expected_state_count": 250,
        "accumulator_dtype": "float32",
        "average_optimizer_state": False,
        "heldout_selection": False,
    }
    if dict(averaging) != exact_averaging:
        raise ProtocolError("Stability-probe tail-averaging rule is not exact.")

    exact_evaluation = {
        "classifier_family": "standard_scaler_l2_logistic",
        "classifier_c": 0.01,
        "classifier_class_weight": "balanced",
        "classifier_threshold": 0.5,
        "minimum_real_bacc": 0.60,
        "roles": ["terminal_step_1000", "tail_average_steps_751_1000"],
        "decode_uses_heldout_true_class": True,
        "heldout_labels_used_for_classifier_fit": False,
        "heldout_labels_used_for_cvae_fit": False,
        "heldout_labels_used_for_scoring": True,
        "heldout_labels_used_for_diagnostic_progression_decision": True,
        "heldout_labels_used_for_confirmation": False,
        "prior_or_generation_evaluated": False,
    }
    if dict(evaluation) != exact_evaluation:
        raise ProtocolError("Stability-probe evaluation contract is not exact.")

    gates = {
        "mean_preservation_min": 0.80,
        "mean_minus_a_preservation_min": -0.02,
        "worst_center_minus_a_preservation_min": -0.05,
        "mean_minus_a_bacc_min": -0.01,
        "mean_minus_a_recall_min": -0.05,
        "mean_minus_a_specificity_min": -0.05,
        "minimum_seed_mean_preservation": 0.75,
        "maximum_seed_mean_preservation_range": 0.05,
        "mean_center_minus_joint_preservation_min": 0.01,
        "minimum_strict_center_wins_over_joint": 3.0,
        "mean_bacc_delta_vs_terminal_min": -0.01,
        "maximum_within_center_class_direction_seed_range": 0.15,
    }
    if dict(decision) != gates:
        raise ProtocolError("Stability-probe decision gates are not exact.")

    expected_lineage: dict[str, object] = {
        "predecessor_experiment": "midogpp.oracle.uniform_b_source_expert_adaptation_pilot.v2",
        "predecessor_artifact": "midogpp_output_uniform_b_source_expert_adaptation_pilot_v2",
        "predecessor_status": "COMPLETE",
        "predecessor_validation": "PASS",
        "predecessor_decision": "B_ADAPTATION_NOT_FEASIBLE",
        "predecessor_protocol_hash": PREDECESSOR_PROTOCOL_HASH,
        "predecessor_hashes": PREDECESSOR_HASHES,
        "observed_seed_mean_preservation_range": 0.06949044912501856,
        "observed_macro_recall_range": 0.07884013071895424,
        "observed_macro_specificity_range": 0.08568367346938777,
        "v2_outcomes_inspected_before_intervention": True,
        "confirmation_eligible": False,
        "rehabilitates_v2": False,
        "sole_intervention": "uniform_parameter_average_steps_751_through_1000",
    }
    if dict(lineage) != expected_lineage:
        raise ProtocolError("Stability-probe v2 lineage is not exact.")
    if (
        claim.get("claim_scope") != "diagnostic_only"
        or claim.get("next_step_if_pass")
        != "separately_reviewed_b_block_prior_only_replay"
        or any(
            bool(claim.get(key, True))
            for key in (
                "may_export_recipe_lock",
                "may_feed_expert_bank",
                "may_feed_generation",
                "may_feed_routing",
                "may_use_validation_or_test",
            )
        )
    ):
        raise ProtocolError("Stability-probe claim firewall is not exact.")

    base = config_path.parent
    predecessor_root = _path(
        base, str(inputs.get("predecessor_root", "")), require_exists=True
    )
    for relative, expected_hash in PREDECESSOR_HASHES.items():
        member = predecessor_root / relative
        if not member.is_file() or _file_sha256(member) != expected_hash:
            raise ProtocolError(
                f"Stability-probe predecessor hash mismatch: {relative}"
            )
    run_state = _json(predecessor_root / "reports/run_state.json")
    validation = _json(predecessor_root / "reports/validation_report.json")
    predecessor_decision = _json(predecessor_root / "reports/pilot_decision.json")
    predecessor_protocol = _json(predecessor_root / "manifests/frozen_protocol.json")
    if (
        run_state.get("status") != "COMPLETE"
        or validation.get("status") != "PASS"
        or predecessor_decision.get("decision") != "B_ADAPTATION_NOT_FEASIBLE"
        or predecessor_protocol.get("protocol_hash") != PREDECESSOR_PROTOCOL_HASH
    ):
        raise ProtocolError("Stability-probe predecessor semantics mismatch.")

    return StabilityConfig(
        name=IDENTITY,
        artifact_root=_path(
            base, str(experiment.get("artifact_root", "")), require_exists=False
        ),
        predecessor_root=predecessor_root,
        centers=CENTERS,
        training_seeds=TRAINING_SEEDS,
        devices=("cuda:0", "cuda:1"),
        cpu_threads_per_worker=1,
        arm="b_block_pca96_32",
        optimizer_steps=1000,
        batch_size=128,
        hidden_dim=512,
        latent_dim=32,
        learning_rate=0.001,
        weight_decay=0.0001,
        beta_final=0.001,
        kl_warmup_steps=250,
        gradient_clip_norm=5.0,
        tail_steps=TAIL_STEPS,
        classifier_c=0.01,
        minimum_real_bacc=0.60,
        gates=gates,
        lineage=dict(lineage),
        code_version=IDENTITY,
    )


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Stability-probe config requires mapping {key!r}.")
    return value


def _path(base: Path, value: str, *, require_exists: bool) -> Path:
    if not value:
        raise ProtocolError("Stability-probe paths must be explicit.")
    if value.startswith(("artifact://", "output://")):
        from ....workspace.runtime import MidogppWorkspace

        scheme, remainder = value.split("://", 1)
        artifact_id, separator, member = remainder.partition("/")
        root = MidogppWorkspace.load().resolve_artifact(
            artifact_id,
            for_output=scheme == "output",
            require_exists=require_exists and scheme == "artifact",
        )
        path = root / member if separator else root
    else:
        raw = Path(value)
        path = raw if raw.is_absolute() else base / raw
    resolved = path.resolve()
    if require_exists and not resolved.exists():
        raise ProtocolError(f"Stability-probe input does not exist: {resolved}")
    return resolved


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProtocolError(f"Expected JSON object: {path}")
    return value


__all__ = (
    "CENTERS",
    "IDENTITY",
    "PREDECESSOR_HASHES",
    "PREDECESSOR_PROTOCOL_HASH",
    "READOUTS",
    "SCHEMA",
    "StabilityConfig",
    "TAIL_STEPS",
    "TRAINING_SEEDS",
    "load_stability_config",
)
