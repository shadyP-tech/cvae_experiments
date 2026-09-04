"""Durable HARP v14 source-crossfit arrays and post-seal label capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from ...runtime.artifact_io import (
    atomic_json,
    atomic_npy,
    read_json,
    sha256_file,
)
from .crossfit_actions import fold_conditioned_action_from_payload, six_source_geometry_audit
from .crossfit_contracts import (
    FoldConditionedActionBlock,
    FoldConditionedCompatibility,
    FoldConditionedSourceSurface,
)
from .durability import durable_barrier
from .geometry_features import geometry_feature_audit
from .hash_contracts import require_sha256


_MANIFEST_SCHEMA = "midogpp_harp_v14_fold_conditioned_surface_store_v1"


@dataclass(frozen=True, slots=True)
class SourceCrossfitSurfaceReceipt:
    root: Path
    manifest_path: Path
    probabilities_path: Path
    dispersion_path: Path
    compatibility_path: Path
    surface_hash: str
    inventory_hash: str
    manifest_hash: str
    manifest_sha256: str
    probabilities_sha256: str
    dispersion_sha256: str
    compatibility_sha256: str
    outer_target_ids: tuple[str, ...]
    outer_heldout_pairs: tuple[tuple[str, str], ...]
    action_block_count: int
    compatibility_receipt_count: int
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        root = Path(self.root).resolve()
        paths = tuple(
            Path(value).resolve()
            for value in (
                self.manifest_path,
                self.probabilities_path,
                self.dispersion_path,
                self.compatibility_path,
            )
        )
        if (
            not root.is_dir()
            or root.is_symlink()
            or any(not path.is_file() or path.is_symlink() or root not in path.parents for path in paths)
        ):
            raise ProtocolError("HARP v14 source-crossfit receipt paths are unsafe.")
        expected = (
            self.manifest_sha256,
            self.probabilities_sha256,
            self.dispersion_sha256,
            self.compatibility_sha256,
        )
        if any(require_sha256(value, name="source-crossfit file hash") != sha256_file(path) for value, path in zip(expected, paths, strict=True)):
            raise ProtocolError("HARP v14 source-crossfit receipt bytes drifted.")
        for value in (self.surface_hash, self.inventory_hash, self.manifest_hash):
            require_sha256(value, name="source-crossfit identity hash")
        outers = tuple(str(value) for value in self.outer_target_ids)
        pairs = tuple((str(h), str(q)) for h, q in self.outer_heldout_pairs)
        if (
            not outers
            or not pairs
            or len(pairs) != len(set(pairs))
            or any(h not in outers or h == q for h, q in pairs)
            or type(self.action_block_count) is not int
            or self.action_block_count < 1
            or type(self.compatibility_receipt_count) is not int
            or self.compatibility_receipt_count < 1
        ):
            raise ProtocolError("HARP v14 source-crossfit receipt inventory drifted.")
        body = {
            "schema_version": "midogpp_harp_v14_source_crossfit_surface_receipt_v1",
            "surface_hash": self.surface_hash,
            "inventory_hash": self.inventory_hash,
            "manifest_hash": self.manifest_hash,
            "manifest_sha256": self.manifest_sha256,
            "probabilities_sha256": self.probabilities_sha256,
            "dispersion_sha256": self.dispersion_sha256,
            "compatibility_sha256": self.compatibility_sha256,
            "outer_target_ids": list(outers),
            "outer_heldout_pairs": [list(value) for value in pairs],
            "action_block_count": self.action_block_count,
            "compatibility_receipt_count": self.compatibility_receipt_count,
            "durable_before_source_labels": True,
            "labels_consumed": False,
        }
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "manifest_path", paths[0])
        object.__setattr__(self, "probabilities_path", paths[1])
        object.__setattr__(self, "dispersion_path", paths[2])
        object.__setattr__(self, "compatibility_path", paths[3])
        object.__setattr__(self, "outer_target_ids", outers)
        object.__setattr__(self, "outer_heldout_pairs", pairs)
        object.__setattr__(self, "receipt_hash", canonical_hash(body))

    def public_payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "schema_version": "midogpp_harp_v14_source_crossfit_surface_receipt_v1",
                "surface_hash": self.surface_hash,
                "inventory_hash": self.inventory_hash,
                "manifest_path": str(self.manifest_path),
                "manifest_hash": self.manifest_hash,
                "manifest_sha256": self.manifest_sha256,
                "probabilities_path": str(self.probabilities_path),
                "probabilities_sha256": self.probabilities_sha256,
                "dispersion_path": str(self.dispersion_path),
                "dispersion_sha256": self.dispersion_sha256,
                "compatibility_path": str(self.compatibility_path),
                "compatibility_sha256": self.compatibility_sha256,
                "outer_target_ids": list(self.outer_target_ids),
                "outer_heldout_pairs": [list(value) for value in self.outer_heldout_pairs],
                "action_block_count": self.action_block_count,
                "compatibility_receipt_count": self.compatibility_receipt_count,
                "receipt_hash": self.receipt_hash,
                "labels_consumed": False,
            }
        )


@dataclass(frozen=True, slots=True)
class SourceCrossfitLabelCapability:
    """Fold-local authority for exactly ``C - {H, q}`` fitting labels."""

    surface_receipt: SourceCrossfitSurfaceReceipt
    outer_target_id: str
    heldout_center_id: str
    authorized_source_center_ids: tuple[str, ...]
    fold_inventory_hash: str
    label_manifest_path: Path
    label_manifest_sha256: str
    capability_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.surface_receipt, SourceCrossfitSurfaceReceipt):
            raise ProtocolError("HARP v14 source labels require a typed surface receipt.")
        h = str(self.outer_target_id)
        q = str(self.heldout_center_id)
        authorized = tuple(str(value) for value in self.authorized_source_center_ids)
        expected_authorized = tuple(
            center for center in CENTERS if center not in {h, q}
        )
        if (
            (h, q) not in self.surface_receipt.outer_heldout_pairs
            or h == q
            or authorized != expected_authorized
        ):
            raise ProtocolError("HARP v14 source-label capability fold drifted.")
        fold_hash = require_sha256(
            self.fold_inventory_hash, name="source-label fold inventory hash"
        )
        path = Path(self.label_manifest_path).resolve()
        digest = require_sha256(
            self.label_manifest_sha256, name="source-label manifest hash"
        )
        if (
            not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != digest
        ):
            raise ProtocolError("HARP v14 source-label manifest identity drifted.")
        body = {
            "schema_version": "midogpp_harp_v14_source_crossfit_label_capability_v1",
            "surface_receipt_hash": self.surface_receipt.receipt_hash,
            "surface_hash": self.surface_receipt.surface_hash,
            "surface_inventory_hash": self.surface_receipt.inventory_hash,
            "surface_manifest_path": str(self.surface_receipt.manifest_path),
            "surface_manifest_sha256": self.surface_receipt.manifest_sha256,
            "outer_target_id": h,
            "heldout_center_id": q,
            "authorized_source_center_ids": list(authorized),
            "fitting_label_scope": "C_MINUS_H_MINUS_Q",
            "prediction_label_scope": "NOT_AUTHORIZED_BY_THIS_CAPABILITY",
            "fold_inventory_hash": fold_hash,
            "label_manifest_path": str(path),
            "label_manifest_sha256": digest,
            "label_access_phase": "AFTER_SOURCE_CROSSFIT_SURFACE_SEALED",
            "capability_scope": "SOURCE_TRAIN_LABELS_ONLY",
            "evaluation_labels_authorized": False,
        }
        object.__setattr__(self, "label_manifest_path", path)
        object.__setattr__(self, "label_manifest_sha256", digest)
        object.__setattr__(self, "outer_target_id", h)
        object.__setattr__(self, "heldout_center_id", q)
        object.__setattr__(self, "authorized_source_center_ids", authorized)
        object.__setattr__(self, "fold_inventory_hash", fold_hash)
        object.__setattr__(self, "capability_hash", canonical_hash(body))

    def public_payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "schema_version": (
                    "midogpp_harp_v14_source_crossfit_label_capability_v1"
                ),
                "surface_receipt_hash": self.surface_receipt.receipt_hash,
                "outer_target_id": self.outer_target_id,
                "heldout_center_id": self.heldout_center_id,
                "authorized_source_center_ids": list(
                    self.authorized_source_center_ids
                ),
                "fold_inventory_hash": self.fold_inventory_hash,
                "label_manifest_path": str(self.label_manifest_path),
                "label_manifest_sha256": self.label_manifest_sha256,
                "capability_hash": self.capability_hash,
                "evaluation_labels_authorized": False,
            }
        )


def persist_source_crossfit_surface(
    root: Path, surface: FoldConditionedSourceSurface
) -> SourceCrossfitSurfaceReceipt:
    """Persist all probability, dispersion, and compatibility numeric bytes."""

    if not isinstance(surface, FoldConditionedSourceSurface):
        raise ProtocolError("HARP v14 durable source store requires a typed surface.")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    probability_path = root / "probabilities.npy"
    dispersion_path = root / "seed_dispersion.npy"
    compatibility_path = root / "compatibility.npy"
    final_paths = (probability_path, dispersion_path, compatibility_path, manifest_path)
    present = tuple(path.exists() for path in final_paths)
    if any(path.is_symlink() for path in final_paths):
        raise ProtocolError("HARP v14 source-crossfit store contains a symlink.")
    if present not in {
        (False, False, False, False),
        (True, False, False, False),
        (True, True, False, False),
        (True, True, True, False),
        (True, True, True, True),
    }:
        raise ProtocolError("HARP v14 source-crossfit store is an unsafe partial state.")

    offsets = [0]
    probability_rows: list[np.ndarray] = []
    dispersion_rows: list[np.ndarray] = []
    blocks = []
    context_rows: dict[tuple[str, str, str], dict[str, object]] = {}
    for ordinal, block in enumerate(surface.blocks):
        probability_rows.append(block.probabilities)
        dispersion_rows.append(block.seed_dispersion)
        offsets.append(offsets[-1] + len(block.probabilities))
        action = block.action
        context = (
            action.outer_target_id,
            action.heldout_center_id,
            action.current_query_center_id,
        )
        if context not in context_rows:
            context_rows[context] = {
                "outer_target_id": context[0],
                "heldout_center_id": context[1],
                "current_query_center_id": context[2],
                "sample_ids": list(block.sample_ids),
                "case_ids": list(block.case_ids),
                "action_ordinals": [],
            }
        elif (
            context_rows[context]["sample_ids"] != list(block.sample_ids)
            or context_rows[context]["case_ids"] != list(block.case_ids)
        ):
            raise ProtocolError("HARP v14 durable crossfit context rows drifted.")
        context_rows[context]["action_ordinals"].append(ordinal)  # type: ignore[union-attr]
        blocks.append(
            {
                "ordinal": ordinal,
                "action": action.to_payload(),
                "offset_start": offsets[-2],
                "offset_stop": offsets[-1],
                "block_hash": block.block_hash,
            }
        )
    probabilities = np.ascontiguousarray(
        np.concatenate(probability_rows), dtype=np.float32
    )
    dispersion = np.ascontiguousarray(
        np.concatenate(dispersion_rows), dtype=np.float32
    )
    compatibility = np.asarray(
        [
            (
                *row.replica_z_scores,
                row.mean_z,
                row.std_z,
                float(row.rank),
                row.rank_margin,
            )
            for row in surface.compatibility
        ],
        dtype=np.float64,
    )
    expected_arrays = (
        (probability_path, probabilities),
        (dispersion_path, dispersion),
        (compatibility_path, compatibility),
    )
    for path, values in expected_arrays:
        if path.exists():
            observed = np.load(path, mmap_mode="r", allow_pickle=False)
            if (
                observed.dtype != values.dtype
                or observed.shape != values.shape
                or not np.array_equal(observed, values)
            ):
                raise ProtocolError(
                    "Existing HARP v14 source-crossfit array differs; refusing repair."
                )
        else:
            atomic_npy(path, values)
    array_hashes = {
        "probabilities_sha256": sha256_file(probability_path),
        "dispersion_sha256": sha256_file(dispersion_path),
        "compatibility_sha256": sha256_file(compatibility_path),
    }
    compatibility_rows = [
        {
            "ordinal": ordinal,
            "outer_target_id": row.outer_target_id,
            "heldout_center_id": row.heldout_center_id,
            "current_query_center_id": row.current_query_center_id,
            "case_id": row.case_id,
            "candidate_source_id": row.candidate_source_id,
            "source_checkpoint_hashes": list(row.source_checkpoint_hashes),
            "receipt_hash": row.receipt_hash,
        }
        for ordinal, row in enumerate(surface.compatibility)
    ]
    inventory_body = {
        "surface_hash": surface.surface_hash,
        "block_hashes": [row.block_hash for row in surface.blocks],
        "compatibility_receipt_hashes": [
            row.receipt_hash for row in surface.compatibility
        ],
        "outer_heldout_pairs": sorted(
            {
                (row.action.outer_target_id, row.action.heldout_center_id)
                for row in surface.blocks
            }
        ),
    }
    inventory_hash = canonical_hash(inventory_body)
    body = {
        "schema_version": _MANIFEST_SCHEMA,
        "status": "COMPLETE_LABEL_FREE_SOURCE_CROSSFIT",
        "surface_hash": surface.surface_hash,
        "inventory_hash": inventory_hash,
        "outer_target_ids": list(surface.outer_target_ids),
        "outer_heldout_pairs": [list(value) for value in inventory_body["outer_heldout_pairs"]],
        "contexts": [context_rows[key] for key in sorted(context_rows)],
        "blocks": blocks,
        "compatibility": compatibility_rows,
        "probability_offsets": offsets,
        "probabilities_member": probability_path.name,
        "dispersion_member": dispersion_path.name,
        "compatibility_member": compatibility_path.name,
        **array_hashes,
        "probability_dtype": "float32",
        "dispersion_dtype": "float32",
        "compatibility_dtype": "float64",
        "six_source_geometry_audit": dict(six_source_geometry_audit()),
        "shared_geometry_feature_audit": dict(geometry_feature_audit()),
        "lineage": dict(surface.lineage),
        "labels_consumed": False,
    }
    manifest = {**body, "manifest_hash": canonical_hash(body)}
    if manifest_path.exists():
        if read_json(manifest_path) != manifest:
            raise ProtocolError(
                "Existing HARP v14 source-crossfit manifest differs; refusing repair."
            )
    else:
        atomic_json(manifest_path, manifest)
    durable_barrier(final_paths)
    return SourceCrossfitSurfaceReceipt(
        root=root,
        manifest_path=manifest_path,
        probabilities_path=probability_path,
        dispersion_path=dispersion_path,
        compatibility_path=compatibility_path,
        surface_hash=surface.surface_hash,
        inventory_hash=inventory_hash,
        manifest_hash=str(manifest["manifest_hash"]),
        manifest_sha256=sha256_file(manifest_path),
        probabilities_sha256=array_hashes["probabilities_sha256"],
        dispersion_sha256=array_hashes["dispersion_sha256"],
        compatibility_sha256=array_hashes["compatibility_sha256"],
        outer_target_ids=surface.outer_target_ids,
        outer_heldout_pairs=tuple(inventory_body["outer_heldout_pairs"]),
        action_block_count=len(surface.blocks),
        compatibility_receipt_count=len(surface.compatibility),
    )


def reconstruct_source_crossfit_surface(
    root: Path, *, expected_surface_hash: str | None = None
) -> tuple[FoldConditionedSourceSurface, SourceCrossfitSurfaceReceipt]:
    """Rebuild every typed block from the closed-world durable store.

    This is intentionally stronger than rehashing a receipt constructor: the
    action inventory, H/q/r coverage, row identities, offsets, case-local
    compatibility inventory, numeric arrays, typed hashes, and global surface
    hash are all reconstructed from independent durable members.
    """

    root = Path(root).resolve()
    if not root.is_dir() or root.is_symlink():
        raise ProtocolError("HARP v14 source-crossfit store root is unsafe.")
    expected_members = {
        "manifest.json",
        "probabilities.npy",
        "seed_dispersion.npy",
        "compatibility.npy",
    }
    observed_members = {path.name for path in root.iterdir()}
    if observed_members != expected_members or any(
        path.is_symlink() or not path.is_file() for path in root.iterdir()
    ):
        raise ProtocolError("HARP v14 source-crossfit store inventory is not closed.")
    manifest_path = root / "manifest.json"
    probability_path = root / "probabilities.npy"
    dispersion_path = root / "seed_dispersion.npy"
    compatibility_path = root / "compatibility.npy"
    manifest = read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise ProtocolError("HARP v14 source-crossfit manifest is malformed.")
    body = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if (
        manifest.get("schema_version") != _MANIFEST_SCHEMA
        or manifest.get("status") != "COMPLETE_LABEL_FREE_SOURCE_CROSSFIT"
        or manifest.get("manifest_hash") != canonical_hash(body)
        or manifest.get("labels_consumed") is not False
        or manifest.get("probabilities_member") != probability_path.name
        or manifest.get("dispersion_member") != dispersion_path.name
        or manifest.get("compatibility_member") != compatibility_path.name
        or manifest.get("probability_dtype") != "float32"
        or manifest.get("dispersion_dtype") != "float32"
        or manifest.get("compatibility_dtype") != "float64"
        or manifest.get("six_source_geometry_audit")
        != dict(six_source_geometry_audit())
        or manifest.get("shared_geometry_feature_audit")
        != dict(geometry_feature_audit())
    ):
        raise ProtocolError("HARP v14 source-crossfit manifest identity drifted.")
    array_bindings = (
        (probability_path, "probabilities_sha256", np.dtype("float32")),
        (dispersion_path, "dispersion_sha256", np.dtype("float32")),
        (compatibility_path, "compatibility_sha256", np.dtype("float64")),
    )
    arrays: list[np.ndarray] = []
    for path, hash_key, dtype in array_bindings:
        expected_hash = require_sha256(
            manifest.get(hash_key), name="source-crossfit member hash"
        )
        if sha256_file(path) != expected_hash:
            raise ProtocolError("HARP v14 source-crossfit member bytes drifted.")
        try:
            values = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ProtocolError("HARP v14 source-crossfit member is unreadable.") from exc
        if values.dtype != dtype or not np.isfinite(values).all():
            raise ProtocolError("HARP v14 source-crossfit numeric member drifted.")
        arrays.append(values)
    probabilities, dispersion, compatibility_values = arrays
    if (
        probabilities.ndim != 1
        or dispersion.shape != probabilities.shape
        or compatibility_values.ndim != 2
        or compatibility_values.shape[1:] != (7,)
    ):
        raise ProtocolError("HARP v14 source-crossfit array geometry drifted.")

    raw_contexts = manifest.get("contexts")
    raw_blocks = manifest.get("blocks")
    raw_compatibility = manifest.get("compatibility")
    raw_offsets = manifest.get("probability_offsets")
    raw_outers = manifest.get("outer_target_ids")
    raw_pairs = manifest.get("outer_heldout_pairs")
    lineage = manifest.get("lineage")
    if (
        not isinstance(raw_contexts, list)
        or not isinstance(raw_blocks, list)
        or not isinstance(raw_compatibility, list)
        or not isinstance(raw_offsets, list)
        or not isinstance(raw_outers, list)
        or not isinstance(raw_pairs, list)
        or not isinstance(lineage, Mapping)
        or len(raw_offsets) != len(raw_blocks) + 1
        or raw_offsets[:1] != [0]
        or raw_offsets[-1:] != [len(probabilities)]
        or len(raw_compatibility) != len(compatibility_values)
    ):
        raise ProtocolError("HARP v14 source-crossfit manifest inventory drifted.")
    context_by_key: dict[tuple[str, str, str], Mapping[str, object]] = {}
    for raw in raw_contexts:
        if not isinstance(raw, Mapping):
            raise ProtocolError("HARP v14 source-crossfit context is malformed.")
        key = (
            str(raw.get("outer_target_id")),
            str(raw.get("heldout_center_id")),
            str(raw.get("current_query_center_id")),
        )
        if key in context_by_key:
            raise ProtocolError("HARP v14 source-crossfit context is duplicated.")
        context_by_key[key] = raw

    blocks: list[FoldConditionedActionBlock] = []
    ordinals_by_context: dict[tuple[str, str, str], list[int]] = {}
    for ordinal, raw in enumerate(raw_blocks):
        if not isinstance(raw, Mapping) or set(raw) != {
            "ordinal",
            "action",
            "offset_start",
            "offset_stop",
            "block_hash",
        }:
            raise ProtocolError("HARP v14 source-crossfit block schema drifted.")
        start = raw.get("offset_start")
        stop = raw.get("offset_stop")
        if (
            raw.get("ordinal") != ordinal
            or type(start) is not int
            or type(stop) is not int
            or start != raw_offsets[ordinal]
            or stop != raw_offsets[ordinal + 1]
            or start < 0
            or stop <= start
            or stop > len(probabilities)
            or not isinstance(raw.get("action"), Mapping)
        ):
            raise ProtocolError("HARP v14 source-crossfit block offsets drifted.")
        action = fold_conditioned_action_from_payload(raw["action"])
        key = (
            action.outer_target_id,
            action.heldout_center_id,
            action.current_query_center_id,
        )
        context = context_by_key.get(key)
        if context is None:
            raise ProtocolError("HARP v14 source-crossfit block context is absent.")
        samples = context.get("sample_ids")
        cases = context.get("case_ids")
        if (
            not isinstance(samples, list)
            or not isinstance(cases, list)
            or len(samples) != stop - start
            or len(cases) != stop - start
        ):
            raise ProtocolError("HARP v14 source-crossfit context rows drifted.")
        block = FoldConditionedActionBlock(
            action=action,
            sample_ids=tuple(str(value) for value in samples),
            case_ids=tuple(str(value) for value in cases),
            probabilities=np.ascontiguousarray(probabilities[start:stop]),
            seed_dispersion=np.ascontiguousarray(dispersion[start:stop]),
        )
        if block.block_hash != raw.get("block_hash"):
            raise ProtocolError("HARP v14 source-crossfit block hash drifted.")
        blocks.append(block)
        ordinals_by_context.setdefault(key, []).append(ordinal)
    for key, context in context_by_key.items():
        if context.get("action_ordinals") != ordinals_by_context.get(key):
            raise ProtocolError("HARP v14 source-crossfit context ordinal drifted.")

    compatibility: list[FoldConditionedCompatibility] = []
    for ordinal, (raw, values) in enumerate(
        zip(raw_compatibility, compatibility_values, strict=True)
    ):
        if not isinstance(raw, Mapping) or set(raw) != {
            "ordinal",
            "outer_target_id",
            "heldout_center_id",
            "current_query_center_id",
            "case_id",
            "candidate_source_id",
            "source_checkpoint_hashes",
            "receipt_hash",
        }:
            raise ProtocolError("HARP v14 compatibility row schema drifted.")
        rank_value = float(values[5])
        if raw.get("ordinal") != ordinal or not rank_value.is_integer():
            raise ProtocolError("HARP v14 compatibility ordinal/rank drifted.")
        row = FoldConditionedCompatibility(
            outer_target_id=str(raw.get("outer_target_id")),
            heldout_center_id=str(raw.get("heldout_center_id")),
            current_query_center_id=str(raw.get("current_query_center_id")),
            case_id=str(raw.get("case_id")),
            candidate_source_id=str(raw.get("candidate_source_id")),
            replica_z_scores=(float(values[0]), float(values[1]), float(values[2])),
            mean_z=float(values[3]),
            std_z=float(values[4]),
            rank=int(rank_value),
            rank_margin=float(values[6]),
            source_checkpoint_hashes=tuple(
                str(value) for value in raw.get("source_checkpoint_hashes", ())
            ),
        )
        if row.receipt_hash != raw.get("receipt_hash"):
            raise ProtocolError("HARP v14 compatibility receipt hash drifted.")
        compatibility.append(row)

    surface = FoldConditionedSourceSurface(
        outer_target_ids=tuple(str(value) for value in raw_outers),
        blocks=tuple(blocks),
        compatibility=tuple(compatibility),
        lineage=lineage,
    )
    if surface.surface_hash != manifest.get("surface_hash") or (
        expected_surface_hash is not None
        and surface.surface_hash
        != require_sha256(expected_surface_hash, name="expected source surface hash")
    ):
        raise ProtocolError("HARP v14 reconstructed source surface hash drifted.")
    pairs = tuple((str(row[0]), str(row[1])) for row in raw_pairs)
    inventory_body = {
        "surface_hash": surface.surface_hash,
        "block_hashes": [row.block_hash for row in surface.blocks],
        "compatibility_receipt_hashes": [
            row.receipt_hash for row in surface.compatibility
        ],
        "outer_heldout_pairs": sorted(
            {
                (row.action.outer_target_id, row.action.heldout_center_id)
                for row in surface.blocks
            }
        ),
    }
    inventory_hash = canonical_hash(inventory_body)
    if (
        inventory_hash != manifest.get("inventory_hash")
        or tuple(inventory_body["outer_heldout_pairs"]) != pairs
    ):
        raise ProtocolError("HARP v14 source-crossfit closed inventory drifted.")
    receipt = SourceCrossfitSurfaceReceipt(
        root=root,
        manifest_path=manifest_path,
        probabilities_path=probability_path,
        dispersion_path=dispersion_path,
        compatibility_path=compatibility_path,
        surface_hash=surface.surface_hash,
        inventory_hash=inventory_hash,
        manifest_hash=str(manifest["manifest_hash"]),
        manifest_sha256=sha256_file(manifest_path),
        probabilities_sha256=sha256_file(probability_path),
        dispersion_sha256=sha256_file(dispersion_path),
        compatibility_sha256=sha256_file(compatibility_path),
        outer_target_ids=surface.outer_target_ids,
        outer_heldout_pairs=pairs,
        action_block_count=len(surface.blocks),
        compatibility_receipt_count=len(surface.compatibility),
    )
    return surface, receipt


def load_source_crossfit_surface_receipt(
    root: Path, *, expected_surface_hash: str | None = None
) -> SourceCrossfitSurfaceReceipt:
    """Return a receipt only after full typed closed-world reconstruction."""

    _, receipt = reconstruct_source_crossfit_surface(
        root, expected_surface_hash=expected_surface_hash
    )
    return receipt


def _fold_inventory_hash(
    surface: FoldConditionedSourceSurface, *, outer_target_id: str, heldout_center_id: str
) -> str:
    h = str(outer_target_id)
    q = str(heldout_center_id)
    scoped_blocks = tuple(
        row
        for row in surface.blocks
        if row.action.outer_target_id == h and row.action.heldout_center_id == q
    )
    scoped_compatibility = tuple(
        row
        for row in surface.compatibility
        if row.outer_target_id == h and row.heldout_center_id == q
    )
    if not scoped_blocks or not scoped_compatibility:
        raise ProtocolError("HARP v14 source-label fold inventory is absent.")
    return canonical_hash(
        {
            "schema_version": "midogpp_harp_v14_source_label_fold_inventory_v1",
            "surface_hash": surface.surface_hash,
            "outer_target_id": h,
            "heldout_center_id": q,
            "block_hashes": [row.block_hash for row in scoped_blocks],
            "compatibility_receipt_hashes": [
                row.receipt_hash for row in scoped_compatibility
            ],
            "prediction_context_sealed": any(
                row.action.current_query_center_id == q for row in scoped_blocks
            ),
            "fitting_contexts_sealed": sorted(
                {
                    row.action.current_query_center_id
                    for row in scoped_blocks
                    if row.action.current_query_center_id != q
                }
            ),
            "labels_consumed": False,
        }
    )


def issue_source_crossfit_label_capability(
    surface_receipt: SourceCrossfitSurfaceReceipt,
    *,
    outer_target_id: str,
    heldout_center_id: str,
    label_manifest_path: Path,
    expected_label_manifest_sha256: str,
) -> SourceCrossfitLabelCapability:
    """Issue label access only from a fully reconstructed durable receipt."""

    if not isinstance(surface_receipt, SourceCrossfitSurfaceReceipt):
        raise ProtocolError("HARP v14 rejects untyped source-crossfit receipts.")
    surface, reconstructed = reconstruct_source_crossfit_surface(
        surface_receipt.root, expected_surface_hash=surface_receipt.surface_hash
    )
    if reconstructed.receipt_hash != surface_receipt.receipt_hash:
        raise ProtocolError("HARP v14 source-crossfit receipt failed reconstruction.")
    h = str(outer_target_id)
    q = str(heldout_center_id)
    return SourceCrossfitLabelCapability(
        surface_receipt=reconstructed,
        outer_target_id=h,
        heldout_center_id=q,
        authorized_source_center_ids=tuple(
            center for center in CENTERS if center not in {h, q}
        ),
        fold_inventory_hash=_fold_inventory_hash(
            surface, outer_target_id=h, heldout_center_id=q
        ),
        label_manifest_path=Path(label_manifest_path),
        label_manifest_sha256=expected_label_manifest_sha256,
    )


def require_source_crossfit_label_capability(
    capability: object,
    *,
    surface_receipt: SourceCrossfitSurfaceReceipt,
    outer_target_id: str,
    heldout_center_id: str,
) -> Path:
    if (
        not isinstance(capability, SourceCrossfitLabelCapability)
        or not isinstance(surface_receipt, SourceCrossfitSurfaceReceipt)
        or capability.surface_receipt.receipt_hash != surface_receipt.receipt_hash
        or capability.outer_target_id != str(outer_target_id)
        or capability.heldout_center_id != str(heldout_center_id)
    ):
        raise ProtocolError("HARP v14 source-label capability is absent or cross-bound.")
    return capability.label_manifest_path


__all__ = (
    "SourceCrossfitLabelCapability",
    "SourceCrossfitSurfaceReceipt",
    "issue_source_crossfit_label_capability",
    "load_source_crossfit_surface_receipt",
    "persist_source_crossfit_surface",
    "reconstruct_source_crossfit_surface",
    "require_source_crossfit_label_capability",
)
