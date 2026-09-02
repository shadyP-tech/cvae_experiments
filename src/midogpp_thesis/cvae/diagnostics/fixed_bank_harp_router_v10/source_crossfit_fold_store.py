"""Durable, reconstructable pre-label pseudo-target fold predictions.

The source-crossfit runner must be able to prove that every pseudo-target
``q`` prediction was fixed before the aggregate source-label capability was
issued.  This module deliberately stores predictions, not models or outcomes:
the stored payload is label-free and can be reconstructed without reopening a
source label shard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from ...routing.policy_calibrated_residual_router_v10 import (
    ActionScore,
    CasePrediction,
    Direction,
    NestedPolicyFold,
)
from ...runtime.artifact_io import atomic_json, read_json, sha256_file
from ...runtime.harp_v10_execution.durability import durable_barrier
from ...runtime.harp_v10_execution.hash_contracts import require_sha256


_FOLD_SCHEMA = "midogpp_harp_v10_prelabel_pseudo_target_fold_v1"
_SET_SCHEMA = "midogpp_harp_v10_prelabel_pseudo_target_fold_set_v1"


@dataclass(frozen=True, slots=True)
class SourceCrossfitFoldSeal:
    """Freshly reconstructed identity for one ``(H, q)`` fold."""

    path: Path
    outer_target_id: str
    heldout_center_id: str
    source_surface_receipt_hash: str
    source_surface_hash: str
    effective_adapter_hash: str
    prediction_surface_hash: str
    fitting_surface_hash: str
    label_capability_hash: str
    isolation_receipt_hash: str
    nested_fold: NestedPolicyFold
    manifest_hash: str
    manifest_sha256: str
    seal_hash: str = field(init=False)

    def __post_init__(self) -> None:
        path = Path(self.path).resolve()
        h = str(self.outer_target_id)
        q = str(self.heldout_center_id)
        if (
            not path.is_file()
            or path.is_symlink()
            or h == q
            or not isinstance(self.nested_fold, NestedPolicyFold)
            or self.nested_fold.outer_target_id != h
            or self.nested_fold.heldout_center_id != q
            or sha256_file(path)
            != require_sha256(self.manifest_sha256, name="crossfit fold manifest")
        ):
            raise ProtocolError("HARP v10 prelabel fold seal is malformed.")
        hashes = (
            self.source_surface_receipt_hash,
            self.source_surface_hash,
            self.effective_adapter_hash,
            self.prediction_surface_hash,
            self.fitting_surface_hash,
            self.label_capability_hash,
            self.isolation_receipt_hash,
            self.manifest_hash,
        )
        for value in hashes:
            require_sha256(value, name="crossfit fold binding")
        body = {
            "schema_version": "midogpp_harp_v10_prelabel_fold_seal_receipt_v1",
            "outer_target_id": h,
            "heldout_center_id": q,
            "source_surface_receipt_hash": self.source_surface_receipt_hash,
            "source_surface_hash": self.source_surface_hash,
            "effective_adapter_hash": self.effective_adapter_hash,
            "prediction_surface_hash": self.prediction_surface_hash,
            "fitting_surface_hash": self.fitting_surface_hash,
            "label_capability_hash": self.label_capability_hash,
            "isolation_receipt_hash": self.isolation_receipt_hash,
            "nested_fold_hash": self.nested_fold.fold_hash,
            "manifest_hash": self.manifest_hash,
            "manifest_sha256": self.manifest_sha256,
            "q_predictions_sealed_before_aggregate_source_labels": True,
        }
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "outer_target_id", h)
        object.__setattr__(self, "heldout_center_id", q)
        object.__setattr__(self, "seal_hash", canonical_hash(body))


@dataclass(frozen=True, slots=True)
class SourceCrossfitFoldSealSet:
    """Complete all-outer inventory required for aggregate label access."""

    path: Path
    expected_center_ids: tuple[str, ...]
    source_surface_receipt_hash: str
    source_surface_hash: str
    effective_adapter_hash: str
    fold_seals: tuple[SourceCrossfitFoldSeal, ...]
    manifest_hash: str
    manifest_sha256: str
    seal_set_hash: str = field(init=False)

    def __post_init__(self) -> None:
        path = Path(self.path).resolve()
        centers = tuple(str(value) for value in self.expected_center_ids)
        seals = tuple(
            sorted(self.fold_seals, key=lambda row: (row.outer_target_id, row.heldout_center_id))
        )
        expected_pairs = tuple((h, q) for h in centers for q in centers if q != h)
        observed_pairs = tuple((row.outer_target_id, row.heldout_center_id) for row in seals)
        if (
            not path.is_file()
            or path.is_symlink()
            or len(centers) < 3
            or len(set(centers)) != len(centers)
            or observed_pairs != expected_pairs
            or seals != self.fold_seals
            or any(
                row.source_surface_receipt_hash != self.source_surface_receipt_hash
                or row.source_surface_hash != self.source_surface_hash
                or row.effective_adapter_hash != self.effective_adapter_hash
                for row in seals
            )
            or sha256_file(path)
            != require_sha256(self.manifest_sha256, name="crossfit fold-set manifest")
        ):
            raise ProtocolError("HARP v10 prelabel fold-set coverage is incomplete.")
        for value in (
            self.source_surface_receipt_hash,
            self.source_surface_hash,
            self.effective_adapter_hash,
            self.manifest_hash,
        ):
            require_sha256(value, name="crossfit fold-set binding")
        body = {
            "schema_version": "midogpp_harp_v10_prelabel_fold_seal_set_receipt_v1",
            "expected_center_ids": list(centers),
            "source_surface_receipt_hash": self.source_surface_receipt_hash,
            "source_surface_hash": self.source_surface_hash,
            "effective_adapter_hash": self.effective_adapter_hash,
            "fold_seal_hashes": [row.seal_hash for row in seals],
            "manifest_hash": self.manifest_hash,
            "manifest_sha256": self.manifest_sha256,
            "all_pseudo_target_predictions_presealed": True,
            "aggregate_source_labels_opened": False,
        }
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "expected_center_ids", centers)
        object.__setattr__(self, "fold_seals", seals)
        object.__setattr__(self, "seal_set_hash", canonical_hash(body))

    def for_outer(self, outer_target_id: str) -> tuple[SourceCrossfitFoldSeal, ...]:
        rows = tuple(row for row in self.fold_seals if row.outer_target_id == str(outer_target_id))
        if len(rows) != len(self.expected_center_ids) - 1:
            raise ProtocolError("HARP v10 outer fold-seal inventory is incomplete.")
        return rows


def persist_source_crossfit_fold(
    root: Path,
    *,
    nested_fold: NestedPolicyFold,
    outer_target_id: str,
    heldout_center_id: str,
    source_surface_receipt_hash: str,
    source_surface_hash: str,
    effective_adapter_hash: str,
    prediction_surface_hash: str,
    fitting_surface_hash: str,
    label_capability_hash: str,
    isolation_receipt_hash: str,
) -> SourceCrossfitFoldSeal:
    """Write and freshly reconstruct one fold before returning authority."""

    if not isinstance(nested_fold, NestedPolicyFold):
        raise ProtocolError("HARP v10 can persist only a typed nested fold.")
    path = Path(root).resolve() / f"outer_{outer_target_id}" / f"heldout_{heldout_center_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "schema_version": _FOLD_SCHEMA,
        "outer_target_id": str(outer_target_id),
        "heldout_center_id": str(heldout_center_id),
        "source_surface_receipt_hash": source_surface_receipt_hash,
        "source_surface_hash": source_surface_hash,
        "effective_adapter_hash": effective_adapter_hash,
        "prediction_surface_hash": prediction_surface_hash,
        "fitting_surface_hash": fitting_surface_hash,
        "label_capability_hash": label_capability_hash,
        "isolation_receipt_hash": isolation_receipt_hash,
        "nested_fold_hash": nested_fold.fold_hash,
        "training_center_ids": list(nested_fold.training_center_ids),
        "training_predictions": [row.public_payload() for row in nested_fold.predictions],
        "heldout_predictions": [row.public_payload() for row in nested_fold.heldout_predictions],
        "q_outcomes_consumed": False,
        "evaluation_labels_consumed": False,
    }
    payload = {**body, "manifest_hash": canonical_hash(body)}
    if path.exists():
        if read_json(path) != payload:
            raise ProtocolError("Existing HARP v10 prelabel fold differs; refusing repair.")
    else:
        atomic_json(path, payload)
    durable_barrier((path,))
    return load_source_crossfit_fold(path)


def load_source_crossfit_fold(path: Path) -> SourceCrossfitFoldSeal:
    path = Path(path).resolve()
    payload = read_json(path)
    body = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if (
        payload.get("schema_version") != _FOLD_SCHEMA
        or payload.get("manifest_hash") != canonical_hash(body)
        or payload.get("q_outcomes_consumed") is not False
        or payload.get("evaluation_labels_consumed") is not False
    ):
        raise ProtocolError("HARP v10 prelabel fold manifest drifted.")
    training = tuple(_prediction_from_payload(value) for value in _list(payload, "training_predictions"))
    heldout = tuple(_prediction_from_payload(value) for value in _list(payload, "heldout_predictions"))
    fold = NestedPolicyFold(
        heldout_center_id=str(payload["heldout_center_id"]),
        training_center_ids=tuple(str(value) for value in _list(payload, "training_center_ids")),
        predictions=training,
        heldout_predictions=heldout,
    )
    if fold.fold_hash != payload.get("nested_fold_hash"):
        raise ProtocolError("HARP v10 reconstructed prelabel fold changed identity.")
    return SourceCrossfitFoldSeal(
        path=path,
        outer_target_id=str(payload["outer_target_id"]),
        heldout_center_id=str(payload["heldout_center_id"]),
        source_surface_receipt_hash=str(payload["source_surface_receipt_hash"]),
        source_surface_hash=str(payload["source_surface_hash"]),
        effective_adapter_hash=str(payload["effective_adapter_hash"]),
        prediction_surface_hash=str(payload["prediction_surface_hash"]),
        fitting_surface_hash=str(payload["fitting_surface_hash"]),
        label_capability_hash=str(payload["label_capability_hash"]),
        isolation_receipt_hash=str(payload["isolation_receipt_hash"]),
        nested_fold=fold,
        manifest_hash=str(payload["manifest_hash"]),
        manifest_sha256=sha256_file(path),
    )


def persist_source_crossfit_fold_set(
    path: Path,
    *,
    expected_center_ids: Sequence[str],
    source_surface_receipt_hash: str,
    source_surface_hash: str,
    effective_adapter_hash: str,
    fold_seals: Sequence[SourceCrossfitFoldSeal],
) -> SourceCrossfitFoldSealSet:
    """Commit the exact all-``(H,q)`` fold inventory as one barrier."""

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle_root = path.parent.parent.resolve()
    if (
        path.parent.name != "manifests"
        or bundle_root == Path(bundle_root.anchor)
        or bundle_root.is_symlink()
    ):
        raise ProtocolError("HARP v10 fold-set path escaped the output bundle.")
    centers = tuple(str(value) for value in expected_center_ids)
    seals = tuple(sorted(fold_seals, key=lambda row: (row.outer_target_id, row.heldout_center_id)))
    expected_pairs = tuple((h, q) for h in centers for q in centers if h != q)
    if (
        tuple((row.outer_target_id, row.heldout_center_id) for row in seals)
        != expected_pairs
        or any(
            bundle_root not in row.path.parents
            or row.path.is_symlink()
            or row.path.relative_to(bundle_root).parts[:2]
            != ("stores", "source_crossfit_folds")
            for row in seals
        )
    ):
        raise ProtocolError("HARP v10 cannot seal an incomplete pseudo-target fold set.")
    body = {
        "schema_version": _SET_SCHEMA,
        "expected_center_ids": list(centers),
        "source_surface_receipt_hash": source_surface_receipt_hash,
        "source_surface_hash": source_surface_hash,
        "effective_adapter_hash": effective_adapter_hash,
        "folds": [
            {
                "outer_target_id": row.outer_target_id,
                "heldout_center_id": row.heldout_center_id,
                "relative_path": row.path.relative_to(bundle_root).as_posix(),
                "manifest_sha256": row.manifest_sha256,
                "seal_hash": row.seal_hash,
                "nested_fold_hash": row.nested_fold.fold_hash,
            }
            for row in seals
        ],
        "fold_count": len(seals),
        "all_q_predictions_durable_before_aggregate_source_labels": True,
        "aggregate_source_labels_opened": False,
    }
    payload = {**body, "manifest_hash": canonical_hash(body)}
    if path.exists():
        if read_json(path) != payload:
            raise ProtocolError("Existing HARP v10 fold-set seal differs; refusing repair.")
    else:
        atomic_json(path, payload)
    durable_barrier((*tuple(row.path for row in seals), path))
    return load_source_crossfit_fold_set(path)


def load_source_crossfit_fold_set(path: Path) -> SourceCrossfitFoldSealSet:
    path = Path(path).resolve()
    bundle_root = path.parent.parent.resolve()
    if (
        path.parent.name != "manifests"
        or bundle_root == Path(bundle_root.anchor)
        or bundle_root.is_symlink()
    ):
        raise ProtocolError("HARP v10 fold-set path escaped the output bundle.")
    payload = read_json(path)
    body = {key: value for key, value in payload.items() if key != "manifest_hash"}
    raw_folds = _list(payload, "folds")
    if (
        payload.get("schema_version") != _SET_SCHEMA
        or payload.get("manifest_hash") != canonical_hash(body)
        or payload.get("fold_count") != len(raw_folds)
        or payload.get("all_q_predictions_durable_before_aggregate_source_labels") is not True
        or payload.get("aggregate_source_labels_opened") is not False
    ):
        raise ProtocolError("HARP v10 prelabel fold-set manifest drifted.")
    seals: list[SourceCrossfitFoldSeal] = []
    for raw in raw_folds:
        if not isinstance(raw, Mapping):
            raise ProtocolError("HARP v10 prelabel fold-set row is malformed.")
        relative = Path(str(raw.get("relative_path")))
        member = (bundle_root / relative).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.parts[:2] != ("stores", "source_crossfit_folds")
            or bundle_root not in member.parents
            or member.is_symlink()
        ):
            raise ProtocolError("HARP v10 prelabel fold-set member escaped its root.")
        seal = load_source_crossfit_fold(member)
        if (
            sha256_file(member) != raw.get("manifest_sha256")
            or seal.seal_hash != raw.get("seal_hash")
            or seal.nested_fold.fold_hash != raw.get("nested_fold_hash")
            or (seal.outer_target_id, seal.heldout_center_id)
            != (str(raw.get("outer_target_id")), str(raw.get("heldout_center_id")))
        ):
            raise ProtocolError("HARP v10 prelabel fold-set member drifted.")
        seals.append(seal)
    return SourceCrossfitFoldSealSet(
        path=path,
        expected_center_ids=tuple(str(value) for value in _list(payload, "expected_center_ids")),
        source_surface_receipt_hash=str(payload["source_surface_receipt_hash"]),
        source_surface_hash=str(payload["source_surface_hash"]),
        effective_adapter_hash=str(payload["effective_adapter_hash"]),
        fold_seals=tuple(seals),
        manifest_hash=str(payload["manifest_hash"]),
        manifest_sha256=sha256_file(path),
    )


def _prediction_from_payload(raw: object) -> CasePrediction:
    if not isinstance(raw, Mapping):
        raise ProtocolError("HARP v10 stored case prediction is malformed.")
    scores = tuple(_score_from_payload(value) for value in _list(raw, "action_scores"))
    prediction = CasePrediction(
        outer_target_id=str(raw["outer_target_id"]),
        query_center_id=str(raw["query_center_id"]),
        case_id=str(raw["case_id"]),
        action_scores=scores,
        raw_top_action_id=str(raw["raw_top_action_id"]),
        top_action_id=str(raw["top_action_id"]),
        acceptance_probability=float(raw["acceptance_probability"]),
        rank_margin=float(raw["rank_margin"]),
        model_hash=str(raw["model_hash"]),
        ranker_hash=str(raw["ranker_hash"]),
        acceptor_hash=str(raw["acceptor_hash"]),
        training_center_ids=tuple(str(value) for value in _list(raw, "training_center_ids")),
        training_candidate_ids=tuple(str(value) for value in _list(raw, "training_candidate_ids")),
        excluded_center_ids=tuple(str(value) for value in _list(raw, "excluded_center_ids")),
        menu_hash=str(raw["menu_hash"]),
    )
    if prediction.prediction_hash != raw.get("prediction_hash"):
        raise ProtocolError("HARP v10 stored case prediction changed identity.")
    return prediction


def _score_from_payload(raw: object) -> ActionScore:
    if not isinstance(raw, Mapping):
        raise ProtocolError("HARP v10 stored action score is malformed.")
    score = ActionScore(
        action_id=str(raw["action_id"]),
        action_hash=str(raw["action_hash"]),
        action_group=str(raw["action_group"]),
        direction=Direction(str(raw["direction"])),
        pairwise_score=float(raw["pairwise_score"]),
        predicted_budget_gain=float(raw["predicted_budget_gain"]),
        predicted_allocation_gain=float(raw["predicted_allocation_gain"]),
        predicted_total_gain=float(raw["predicted_total_gain"]),
        predicted_harm_probability=float(raw["predicted_harm_probability"]),
        predicted_brier_delta=float(raw["predicted_brier_delta"]),
        predicted_log_delta=float(raw["predicted_log_delta"]),
        acceptance_probability=float(raw["acceptance_probability"]),
        model_available=bool(raw["model_available"]),
    )
    if score.score_hash != raw.get("score_hash"):
        raise ProtocolError("HARP v10 stored action score changed identity.")
    return score


def _list(payload: Mapping[str, object], key: str) -> list[object]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ProtocolError(f"HARP v10 stored {key} is not a list.")
    return value


__all__ = (
    "SourceCrossfitFoldSeal",
    "SourceCrossfitFoldSealSet",
    "load_source_crossfit_fold",
    "load_source_crossfit_fold_set",
    "persist_source_crossfit_fold",
    "persist_source_crossfit_fold_set",
)
