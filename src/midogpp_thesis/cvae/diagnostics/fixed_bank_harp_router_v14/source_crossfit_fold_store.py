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
from ...routing.policy_calibrated_residual_router_v14 import (
    ActionScore,
    CasePrediction,
    Direction,
    EffectiveMenu,
    NestedPolicyFold,
    build_label_free_case_inventory,
)
from ...runtime.artifact_io import atomic_json, read_json, sha256_file
from ...runtime.harp_v14_execution.durability import durable_barrier
from ...runtime.harp_v14_execution.hash_contracts import require_sha256


_FOLD_SCHEMA = "midogpp_harp_v14_prelabel_pseudo_target_fold_v1"
_SET_SCHEMA = "midogpp_harp_v14_prelabel_pseudo_target_fold_set_v1"


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
    fold_menu_binding_hash: str
    fold_menu_binding_certificate_hash: str
    fold_menu_binding_certificate_receipt_hash: str
    label_free_case_inventory_hash: str
    exact_b_control_count: int
    active_menu_count: int
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
            raise ProtocolError("HARP v14 prelabel fold seal is malformed.")
        hashes = (
            self.source_surface_receipt_hash,
            self.source_surface_hash,
            self.effective_adapter_hash,
            self.prediction_surface_hash,
            self.fitting_surface_hash,
            self.label_capability_hash,
            self.isolation_receipt_hash,
            self.fold_menu_binding_hash,
            self.fold_menu_binding_certificate_hash,
            self.fold_menu_binding_certificate_receipt_hash,
            self.label_free_case_inventory_hash,
            self.manifest_hash,
        )
        for value in hashes:
            require_sha256(value, name="crossfit fold binding")
        body = {
            "schema_version": "midogpp_harp_v14_prelabel_fold_seal_receipt_v1",
            "outer_target_id": h,
            "heldout_center_id": q,
            "source_surface_receipt_hash": self.source_surface_receipt_hash,
            "source_surface_hash": self.source_surface_hash,
            "effective_adapter_hash": self.effective_adapter_hash,
            "prediction_surface_hash": self.prediction_surface_hash,
            "fitting_surface_hash": self.fitting_surface_hash,
            "label_capability_hash": self.label_capability_hash,
            "isolation_receipt_hash": self.isolation_receipt_hash,
            "fold_menu_binding_hash": self.fold_menu_binding_hash,
            "fold_menu_binding_certificate_hash": (
                self.fold_menu_binding_certificate_hash
            ),
            "fold_menu_binding_certificate_receipt_hash": (
                self.fold_menu_binding_certificate_receipt_hash
            ),
            "label_free_case_inventory_hash": self.label_free_case_inventory_hash,
            "exact_b_control_count": self.exact_b_control_count,
            "active_menu_count": self.active_menu_count,
            "nested_fold_hash": self.nested_fold.fold_hash,
            "manifest_hash": self.manifest_hash,
            "manifest_sha256": self.manifest_sha256,
            "q_predictions_sealed_before_aggregate_source_labels": True,
            "label_free_case_inventory_sealed_before_q_outcomes": True,
        }
        if (
            type(self.exact_b_control_count) is not int
            or type(self.active_menu_count) is not int
            or self.exact_b_control_count < 0
            or self.active_menu_count < 0
            or self.exact_b_control_count + self.active_menu_count
            != len((*self.nested_fold.predictions, *self.nested_fold.heldout_predictions))
        ):
            raise ProtocolError("HARP v14 prelabel case inventory counts drifted.")
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
    fold_menu_binding_certificate_hash: str
    fold_menu_binding_certificate_receipt_hash: str
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
                or row.fold_menu_binding_certificate_hash
                != self.fold_menu_binding_certificate_hash
                or row.fold_menu_binding_certificate_receipt_hash
                != self.fold_menu_binding_certificate_receipt_hash
                for row in seals
            )
            or sha256_file(path)
            != require_sha256(self.manifest_sha256, name="crossfit fold-set manifest")
        ):
            raise ProtocolError("HARP v14 prelabel fold-set coverage is incomplete.")
        for value in (
            self.source_surface_receipt_hash,
            self.source_surface_hash,
            self.effective_adapter_hash,
            self.fold_menu_binding_certificate_hash,
            self.fold_menu_binding_certificate_receipt_hash,
            self.manifest_hash,
        ):
            require_sha256(value, name="crossfit fold-set binding")
        body = {
            "schema_version": "midogpp_harp_v14_prelabel_fold_seal_set_receipt_v1",
            "expected_center_ids": list(centers),
            "source_surface_receipt_hash": self.source_surface_receipt_hash,
            "source_surface_hash": self.source_surface_hash,
            "effective_adapter_hash": self.effective_adapter_hash,
            "fold_menu_binding_certificate_hash": (
                self.fold_menu_binding_certificate_hash
            ),
            "fold_menu_binding_certificate_receipt_hash": (
                self.fold_menu_binding_certificate_receipt_hash
            ),
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
            raise ProtocolError("HARP v14 outer fold-seal inventory is incomplete.")
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
    fold_menu_binding_hash: str,
    fold_menu_binding_certificate_hash: str,
    fold_menu_binding_certificate_receipt_hash: str,
    effective_menus: Sequence[EffectiveMenu],
) -> SourceCrossfitFoldSeal:
    """Write and freshly reconstruct one fold before returning authority."""

    if not isinstance(nested_fold, NestedPolicyFold):
        raise ProtocolError("HARP v14 can persist only a typed nested fold.")
    path = Path(root).resolve() / f"outer_{outer_target_id}" / f"heldout_{heldout_center_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    label_free_inventory = build_label_free_case_inventory(
        (*nested_fold.predictions, *nested_fold.heldout_predictions),
        effective_menus,
        require_complete=True,
    )
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
        "fold_menu_binding_hash": fold_menu_binding_hash,
        "fold_menu_binding_certificate_hash": fold_menu_binding_certificate_hash,
        "fold_menu_binding_certificate_receipt_hash": (
            fold_menu_binding_certificate_receipt_hash
        ),
        "label_free_case_inventory": label_free_inventory.public_payload(),
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
            raise ProtocolError("Existing HARP v14 prelabel fold differs; refusing repair.")
    else:
        atomic_json(path, payload)
    durable_barrier((path,))
    return load_source_crossfit_fold(path, effective_menus=effective_menus)


def load_source_crossfit_fold(
    path: Path,
    *,
    effective_menus: Sequence[EffectiveMenu] | None = None,
) -> SourceCrossfitFoldSeal:
    path = Path(path).resolve()
    payload = read_json(path)
    body = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if (
        payload.get("schema_version") != _FOLD_SCHEMA
        or payload.get("manifest_hash") != canonical_hash(body)
        or payload.get("q_outcomes_consumed") is not False
        or payload.get("evaluation_labels_consumed") is not False
    ):
        raise ProtocolError("HARP v14 prelabel fold manifest drifted.")
    training = tuple(_prediction_from_payload(value) for value in _list(payload, "training_predictions"))
    heldout = tuple(_prediction_from_payload(value) for value in _list(payload, "heldout_predictions"))
    fold = NestedPolicyFold(
        heldout_center_id=str(payload["heldout_center_id"]),
        training_center_ids=tuple(str(value) for value in _list(payload, "training_center_ids")),
        predictions=training,
        heldout_predictions=heldout,
    )
    if fold.fold_hash != payload.get("nested_fold_hash"):
        raise ProtocolError("HARP v14 reconstructed prelabel fold changed identity.")
    inventory_payload = _validated_label_free_inventory_payload(
        payload.get("label_free_case_inventory"),
        predictions=(*fold.predictions, *fold.heldout_predictions),
    )
    if effective_menus is not None:
        reconstructed_inventory = build_label_free_case_inventory(
            (*fold.predictions, *fold.heldout_predictions),
            effective_menus,
            require_complete=True,
        )
        if reconstructed_inventory.public_payload() != inventory_payload:
            raise ProtocolError(
                "HARP v14 durable prelabel case inventory changed identity."
            )
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
        fold_menu_binding_hash=str(payload["fold_menu_binding_hash"]),
        fold_menu_binding_certificate_hash=str(
            payload["fold_menu_binding_certificate_hash"]
        ),
        fold_menu_binding_certificate_receipt_hash=str(
            payload["fold_menu_binding_certificate_receipt_hash"]
        ),
        label_free_case_inventory_hash=str(inventory_payload["inventory_hash"]),
        exact_b_control_count=int(inventory_payload["exact_b_control_count"]),
        active_menu_count=int(inventory_payload["active_menu_count"]),
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
    fold_menu_binding_certificate_hash: str,
    fold_menu_binding_certificate_receipt_hash: str,
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
        raise ProtocolError("HARP v14 fold-set path escaped the output bundle.")
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
        raise ProtocolError("HARP v14 cannot seal an incomplete pseudo-target fold set.")
    body = {
        "schema_version": _SET_SCHEMA,
        "expected_center_ids": list(centers),
        "source_surface_receipt_hash": source_surface_receipt_hash,
        "source_surface_hash": source_surface_hash,
        "effective_adapter_hash": effective_adapter_hash,
        "fold_menu_binding_certificate_hash": fold_menu_binding_certificate_hash,
        "fold_menu_binding_certificate_receipt_hash": (
            fold_menu_binding_certificate_receipt_hash
        ),
        "folds": [
            {
                "outer_target_id": row.outer_target_id,
                "heldout_center_id": row.heldout_center_id,
                "relative_path": row.path.relative_to(bundle_root).as_posix(),
                "manifest_sha256": row.manifest_sha256,
                "seal_hash": row.seal_hash,
                "nested_fold_hash": row.nested_fold.fold_hash,
                "fold_menu_binding_hash": row.fold_menu_binding_hash,
                "fold_menu_binding_certificate_hash": (
                    row.fold_menu_binding_certificate_hash
                ),
                "fold_menu_binding_certificate_receipt_hash": (
                    row.fold_menu_binding_certificate_receipt_hash
                ),
                "label_free_case_inventory_hash": (
                    row.label_free_case_inventory_hash
                ),
                "exact_b_control_count": row.exact_b_control_count,
                "active_menu_count": row.active_menu_count,
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
            raise ProtocolError("Existing HARP v14 fold-set seal differs; refusing repair.")
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
        raise ProtocolError("HARP v14 fold-set path escaped the output bundle.")
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
        raise ProtocolError("HARP v14 prelabel fold-set manifest drifted.")
    seals: list[SourceCrossfitFoldSeal] = []
    for raw in raw_folds:
        if not isinstance(raw, Mapping):
            raise ProtocolError("HARP v14 prelabel fold-set row is malformed.")
        relative = Path(str(raw.get("relative_path")))
        member = (bundle_root / relative).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.parts[:2] != ("stores", "source_crossfit_folds")
            or bundle_root not in member.parents
            or member.is_symlink()
        ):
            raise ProtocolError("HARP v14 prelabel fold-set member escaped its root.")
        seal = load_source_crossfit_fold(member)
        if (
            sha256_file(member) != raw.get("manifest_sha256")
            or seal.seal_hash != raw.get("seal_hash")
            or seal.nested_fold.fold_hash != raw.get("nested_fold_hash")
            or seal.fold_menu_binding_hash != raw.get("fold_menu_binding_hash")
            or seal.fold_menu_binding_certificate_hash
            != raw.get("fold_menu_binding_certificate_hash")
            or seal.fold_menu_binding_certificate_receipt_hash
            != raw.get("fold_menu_binding_certificate_receipt_hash")
            or seal.label_free_case_inventory_hash
            != raw.get("label_free_case_inventory_hash")
            or seal.exact_b_control_count != raw.get("exact_b_control_count")
            or seal.active_menu_count != raw.get("active_menu_count")
            or (seal.outer_target_id, seal.heldout_center_id)
            != (str(raw.get("outer_target_id")), str(raw.get("heldout_center_id")))
        ):
            raise ProtocolError("HARP v14 prelabel fold-set member drifted.")
        seals.append(seal)
    return SourceCrossfitFoldSealSet(
        path=path,
        expected_center_ids=tuple(str(value) for value in _list(payload, "expected_center_ids")),
        source_surface_receipt_hash=str(payload["source_surface_receipt_hash"]),
        source_surface_hash=str(payload["source_surface_hash"]),
        effective_adapter_hash=str(payload["effective_adapter_hash"]),
        fold_menu_binding_certificate_hash=str(
            payload["fold_menu_binding_certificate_hash"]
        ),
        fold_menu_binding_certificate_receipt_hash=str(
            payload["fold_menu_binding_certificate_receipt_hash"]
        ),
        fold_seals=tuple(seals),
        manifest_hash=str(payload["manifest_hash"]),
        manifest_sha256=sha256_file(path),
    )


def _validated_label_free_inventory_payload(
    raw: object,
    *,
    predictions: Sequence[CasePrediction],
) -> dict[str, object]:
    """Reconstruct the label-free inventory binding from durable predictions."""

    if not isinstance(raw, Mapping):
        raise ProtocolError("HARP v14 prelabel case inventory is absent.")
    required = {
        "schema_version",
        "case_count",
        "exact_b_control_count",
        "active_menu_count",
        "complete_for_menu_universe",
        "contexts",
        "inventory_hash",
        "source_outcomes_consumed",
        "evaluation_labels_consumed",
    }
    contexts = raw.get("contexts")
    if (
        set(raw) != required
        or raw.get("schema_version")
        != "policy_calibrated_label_free_case_inventory_v14"
        or raw.get("complete_for_menu_universe") is not True
        or raw.get("source_outcomes_consumed") is not False
        or raw.get("evaluation_labels_consumed") is not False
        or not isinstance(contexts, list)
    ):
        raise ProtocolError("HARP v14 prelabel case inventory schema drifted.")
    predictions_by_key = {
        (row.outer_target_id, row.query_center_id, row.case_id): row
        for row in predictions
    }
    if len(predictions_by_key) != len(tuple(predictions)):
        raise ProtocolError("HARP v14 prelabel predictions duplicate a case.")
    context_hashes: list[str] = []
    observed_keys: list[tuple[str, str, str]] = []
    control_count = 0
    for value in contexts:
        if not isinstance(value, Mapping) or set(value) != {
            "outer_target_id",
            "query_center_id",
            "case_id",
            "kind",
            "effective_menu_hash",
            "action_ids",
            "action_hashes",
            "prediction_hash",
            "context_hash",
        }:
            raise ProtocolError("HARP v14 prelabel case context schema drifted.")
        key = (
            str(value["outer_target_id"]),
            str(value["query_center_id"]),
            str(value["case_id"]),
        )
        prediction = predictions_by_key.get(key)
        action_ids = value.get("action_ids")
        action_hashes = value.get("action_hashes")
        if (
            prediction is None
            or not isinstance(action_ids, list)
            or not isinstance(action_hashes, list)
            or len(action_ids) != len(action_hashes)
            or len(set(str(item) for item in action_ids)) != len(action_ids)
            or value.get("effective_menu_hash") != prediction.menu_hash
            or value.get("prediction_hash") != prediction.prediction_hash
        ):
            raise ProtocolError("HARP v14 prelabel case context is prediction-unbound.")
        score_actions = {
            row.action_id: row.action_hash for row in prediction.action_scores
        }
        menu_actions = {
            str(action_id): str(action_hash)
            for action_id, action_hash in zip(action_ids, action_hashes, strict=True)
        }
        if score_actions != menu_actions:
            raise ProtocolError("HARP v14 prelabel action identity drifted.")
        kind = str(value.get("kind"))
        expected_kind = "ACTIVE_MENU" if action_ids else "EXACT_B_CONTROL"
        if kind != expected_kind or (
            kind == "EXACT_B_CONTROL"
            and (
                prediction.raw_top_action_id != "B"
                or prediction.top_action_id != "B"
                or prediction.acceptance_probability != 0.0
                or prediction.rank_margin != 0.0
            )
        ):
            raise ProtocolError("HARP v14 prelabel exact-B control drifted.")
        control_count += kind == "EXACT_B_CONTROL"
        context_body = {
            "schema_version": "policy_calibrated_label_free_case_context_v14",
            "case_key": key,
            "kind": kind,
            "effective_menu_hash": prediction.menu_hash,
            "action_ids": tuple(str(item) for item in action_ids),
            "action_hashes": tuple(str(item) for item in action_hashes),
            "prediction_hash": prediction.prediction_hash,
            "source_outcomes_consumed": False,
            "evaluation_labels_consumed": False,
        }
        expected_context_hash = canonical_hash(context_body)
        if value.get("context_hash") != expected_context_hash:
            raise ProtocolError("HARP v14 prelabel case-context hash drifted.")
        observed_keys.append(key)
        context_hashes.append(expected_context_hash)
    expected_keys = sorted(predictions_by_key)
    if observed_keys != expected_keys:
        raise ProtocolError("HARP v14 prelabel case inventory coverage drifted.")
    active_count = len(contexts) - control_count
    inventory_body = {
        "schema_version": "policy_calibrated_label_free_case_inventory_v14",
        "context_hashes": tuple(context_hashes),
        "case_count": len(contexts),
        "exact_b_control_count": control_count,
        "complete_for_menu_universe": True,
        "source_outcomes_consumed": False,
        "evaluation_labels_consumed": False,
    }
    if (
        raw.get("case_count") != len(contexts)
        or raw.get("exact_b_control_count") != control_count
        or raw.get("active_menu_count") != active_count
        or raw.get("inventory_hash") != canonical_hash(inventory_body)
    ):
        raise ProtocolError("HARP v14 prelabel case-inventory hash drifted.")
    return dict(raw)


def _prediction_from_payload(raw: object) -> CasePrediction:
    if not isinstance(raw, Mapping):
        raise ProtocolError("HARP v14 stored case prediction is malformed.")
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
        raise ProtocolError("HARP v14 stored case prediction changed identity.")
    return prediction


def _score_from_payload(raw: object) -> ActionScore:
    if not isinstance(raw, Mapping):
        raise ProtocolError("HARP v14 stored action score is malformed.")
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
        raise ProtocolError("HARP v14 stored action score changed identity.")
    return score


def _list(payload: Mapping[str, object], key: str) -> list[object]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ProtocolError(f"HARP v14 stored {key} is not a list.")
    return value


__all__ = (
    "SourceCrossfitFoldSeal",
    "SourceCrossfitFoldSealSet",
    "load_source_crossfit_fold",
    "load_source_crossfit_fold_set",
    "persist_source_crossfit_fold",
    "persist_source_crossfit_fold_set",
)
