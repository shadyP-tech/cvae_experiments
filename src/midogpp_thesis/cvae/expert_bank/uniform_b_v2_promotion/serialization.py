"""Serialization and fail-closed loading for routing-authorized v2 experts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from ....common.hashing import stable_hash
from ...block_frame import PCAState, PilotFeatureFrame
from ...generation_samplers import AggregatePosteriorSampler, ClassSamplerState
from ...keyed_training import model_state_hash
from ...models import ClassConditionedCVAE
from ...protocol import ProtocolError
from ...preservation.uniform_b_optimized_prior.core import OptimizedSourceFrame
from .contracts import (
    CENTERS,
    N_EXPERTS,
    PROMOTION_DECISION,
    PUBLICATION_STATE,
    TRAINING_SEEDS,
)


@dataclass(frozen=True)
class RoutingAuthorizedExpert:
    source_center: str
    training_seed: int
    model: ClassConditionedCVAE
    source_frame: OptimizedSourceFrame
    sampler: AggregatePosteriorSampler
    checkpoint_hash: str
    expert_lock_hash: str


def source_frame_from_payload(payload: Mapping[str, object]) -> OptimizedSourceFrame:
    frame_raw = _mapping(payload, "frame")
    blocks_raw = frame_raw.get("blocks")
    if isinstance(blocks_raw, (str, bytes)) or not isinstance(blocks_raw, list):
        raise ProtocolError("Promoted source frame lacks PCA blocks.")
    blocks = []
    for raw in blocks_raw:
        if not isinstance(raw, Mapping):
            raise ProtocolError("Promoted source frame contains an invalid PCA block.")
        blocks.append(
            PCAState(
                start=int(raw["start"]),
                stop=int(raw["stop"]),
                output_dim=int(raw["output_dim"]),
                scaler_mean=np.asarray(raw["scaler_mean"], dtype=np.float64),
                scaler_scale=np.asarray(raw["scaler_scale"], dtype=np.float64),
                pca_mean=np.asarray(raw["pca_mean"], dtype=np.float64),
                pca_components=np.asarray(raw["pca_components"], dtype=np.float64),
                explained_variance=np.asarray(raw["explained_variance"], dtype=np.float64),
                explained_variance_ratio_sum=float(raw["explained_variance_ratio_sum"]),
            )
        )
    frame = PilotFeatureFrame(
        arm=str(frame_raw["arm"]),
        input_dim=int(frame_raw["input_dim"]),
        output_dim=int(frame_raw["output_dim"]),
        blocks=tuple(blocks),
        fit_sample_hash=str(frame_raw["fit_sample_hash"]),
    )
    observed = OptimizedSourceFrame(
        source_center=str(payload["source_center"]),
        source_row_hash=str(payload["source_row_hash"]),
        frame=frame,
    )
    return observed


def sampler_from_payload(payload: Mapping[str, object]) -> AggregatePosteriorSampler:
    classes_raw = _mapping(payload, "classes")
    classes: dict[int, ClassSamplerState] = {}
    for label in (0, 1):
        raw = classes_raw.get(str(label))
        if not isinstance(raw, Mapping):
            raise ProtocolError(f"Promoted sampler lacks class {label}.")
        covariance = np.asarray(raw["covariance"], dtype=np.float64)
        try:
            cholesky = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ProtocolError("Promoted sampler covariance is not positive definite.") from exc
        classes[label] = ClassSamplerState(
            class_label=int(raw["class_label"]),
            requested_family=str(raw["requested_family"]),
            realized_family=str(raw["realized_family"]),
            mean=np.asarray(raw["mean"], dtype=np.float64),
            covariance=covariance,
            cholesky=cholesky,
            n_rows=int(raw["n_rows"]),
            raw_between_covariance=np.asarray(raw["raw_between_covariance"], dtype=np.float64),
            within_posterior_diagonal=np.asarray(raw["within_posterior_diagonal"], dtype=np.float64),
            shrinkage=None if raw.get("shrinkage") is None else float(raw["shrinkage"]),
            shrinkage_target=(
                None if raw.get("shrinkage_target") is None else float(raw["shrinkage_target"])
            ),
            jitter=float(raw["jitter"]),
            condition_number=float(raw["condition_number"]),
            eigenvalues=tuple(float(value) for value in raw["eigenvalues"]),
            fallback_reason=str(raw.get("fallback_reason", "")),
        )
    sampler = AggregatePosteriorSampler(
        requested_family=str(payload["requested_family"]),
        classes=classes,
        latent_dim=int(payload["latent_dim"]),
        source_row_hash=str(payload["source_row_hash"]),
    )
    if sampler.state_hash != payload.get("sampler_state_hash"):
        raise ProtocolError("Promoted sampler state hash drifted.")
    return sampler


def load_routing_authorized_expert(
    root: str | Path,
    *,
    source_center: str,
    training_seed: int,
    device: str = "cpu",
) -> RoutingAuthorizedExpert:
    """Load one expert only after checking the Stage-30 authorization boundary."""

    path = Path(root)
    _assert_authorized_boundary(path)
    center = str(source_center)
    seed = int(training_seed)
    if center not in CENTERS or seed not in TRAINING_SEEDS:
        raise ProtocolError("Requested expert is outside the frozen v2 bank.")
    bank = _read_json(path / "manifests/expert_bank_index.json")
    records = bank.get("records")
    if not isinstance(records, list) or len(records) != N_EXPERTS:
        raise ProtocolError("Promoted expert-bank index coverage drifted.")
    matches = [
        row for row in records
        if isinstance(row, Mapping)
        and str(row.get("source_center")) == center
        and int(row.get("training_seed", -1)) == seed
    ]
    if len(matches) != 1:
        raise ProtocolError("Promoted expert-bank key is missing or duplicated.")
    record = matches[0]
    checkpoint_path = _safe_member(path, str(record["checkpoint_path"]))
    frame_path = _safe_member(path, str(record["frame_path"]))
    sampler_path = _safe_member(path, str(record["sampler_path"]))
    for member, digest in (
        (checkpoint_path, record["checkpoint_file_sha256"]),
        (frame_path, record["frame_file_sha256"]),
        (sampler_path, record["sampler_file_sha256"]),
    ):
        if not member.is_file() or _sha256_file(member) != digest:
            raise ProtocolError(f"Promoted expert member drifted: {member.name}.")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model"), Mapping):
        raise ProtocolError("Promoted checkpoint payload is invalid.")
    model = ClassConditionedCVAE(
        input_dim=int(bank["model"]["input_dim"]),
        hidden_dim=int(bank["model"]["hidden_dim"]),
        latent_dim=int(bank["model"]["latent_dim"]),
        num_hidden_layers=int(bank["model"]["num_hidden_layers"]),
    ).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    if model_state_hash(model) != record.get("checkpoint_hash"):
        raise ProtocolError("Promoted expert model hash drifted.")
    frame_payload = _read_json(frame_path)
    source_frame = source_frame_from_payload(frame_payload)
    if source_frame.state_hash != record.get("frame_hash"):
        raise ProtocolError("Promoted expert frame hash drifted.")
    sampler = sampler_from_payload(_read_json(sampler_path))
    if sampler.state_hash != record.get("sampler_state_hash"):
        raise ProtocolError("Promoted expert sampler hash drifted.")
    lock_unhashed = {key: value for key, value in record.items() if key != "expert_lock_hash"}
    if stable_hash(lock_unhashed) != record.get("expert_lock_hash"):
        raise ProtocolError("Promoted expert lock hash drifted.")
    return RoutingAuthorizedExpert(
        source_center=center,
        training_seed=seed,
        model=model,
        source_frame=source_frame,
        sampler=sampler,
        checkpoint_hash=str(record["checkpoint_hash"]),
        expert_lock_hash=str(record["expert_lock_hash"]),
    )


def _assert_authorized_boundary(root: Path) -> None:
    validation = _read_json(root / "reports/validation_report.json")
    decision = _read_json(root / "reports/promotion_decision.json")
    state = _read_json(root / "reports/run_state.json")
    if (
        validation.get("status") != "PASS"
        or decision.get("decision") != PROMOTION_DECISION
        or decision.get("publication_state") != PUBLICATION_STATE
        or decision.get("may_feed_deployable_selection") is not True
        or state.get("status") != "COMPLETE"
    ):
        raise ProtocolError("Expert bank is not routing-authorized.")


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Promoted payload lacks mapping {key!r}.")
    return value


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read promoted expert-bank member: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"Promoted JSON must be an object: {path}.")
    return payload


def _safe_member(root: Path, relative: str) -> Path:
    resolved_root = root.resolve()
    member = (resolved_root / relative).resolve()
    if member == resolved_root or not member.is_relative_to(resolved_root):
        raise ProtocolError("Promoted expert-bank path escapes its artifact root.")
    return member


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = (
    "RoutingAuthorizedExpert",
    "load_routing_authorized_expert",
    "sampler_from_payload",
    "source_frame_from_payload",
)
