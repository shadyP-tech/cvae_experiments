"""File-backed Stage-60 adapters for HARP action and target-support surfaces.

Both adapters consume only a complete neutral ``HarpPredictionMenuSeal``.  The
action adapter durably publishes every pre-label probability receipt before it
constructs the one-shot source-label capability.  The target-support adapter
has no outcome-loading path at all.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
from typing import Any

import numpy as np
import yaml

from ...protocol import ProtocolError
from ...runtime.artifact_io import (
    atomic_json,
    atomic_npy,
    read_json,
    sha256_file,
)
from ...runtime.harp_probability_menu import (
    EXACT_NINE_SEED_PAIRS,
    HarpPredictionMenuSeal,
    build_all_development_actions,
    build_all_target_actions,
)
from ..harp_protocol.hashing import canonical_hash, require_sha256
from ..harp_protocol.label_access import (
    HarpDurablePredictionSeal,
    HarpSourceLabelCapability,
    HarpSourceLabelRow,
    OpenedHarpSourceLabels,
    build_durable_prediction_seal,
)
from ..harp_stage60.config import HarpInputReadiness, HarpStage60Config
from ..harp_stage60.constants import ACTION_SURFACE, CENTERS, TARGET_SUPPORT_SURFACE
from ..harp_stage60.execution_contracts import (
    HarpBuiltProduct,
    HarpDurablePrelabelSeal,
    HarpRunReceipt,
)
from .build import (
    build_action_feature_surface,
    build_directional_response_surface,
    build_probability_ensemble_surface,
    build_probability_surface,
)
from .contracts import (
    ACTION_FEATURE_NAMES,
    ACTION_LAMBDAS,
    HarpActionFeatureSurface,
    HarpDirectionalResponseSurface,
    HarpProbabilityRow,
)
from .artifact_contract import *  # noqa: F403 - closed-world member vocabulary
from .catalog_persistence import (
    persist_prelabel_catalog_members as _persist_prelabel_catalog_members,
    write_directional_response_table as _write_directional_response_table,
)
from .bundle_validation import validate_completed_bundle as _validate_completed_bundle
from .lineage import (
    HarpAuthoritativeLineage,
    load_authoritative_lineage,
    menu_semantic_lineage,
)
from .inference_binding import HarpActionInferenceBinding
from .payloads import (
    action_feature_payload as _action_feature_payload,
    development_seed_surface as _development_seed_surface,
    read_fresh_source_labels as _read_fresh_source_labels,
    response_payload as _response_payload,
    target_support_payload as _target_support_payload,
)
from .workstation_runtime import LINEAGE_RECEIPT_MEMBER
from .transport import (
    EXPECTED_SEED_IDS,
    load_probability_menu_transport as _load_probability_menu_transport,
    seed_id as _seed_id,
    write_probability_menu_transport,
)


MenuLoader = Callable[[HarpStage60Config, HarpInputReadiness], HarpPredictionMenuSeal]


class _ProductionSurfaceBase:
    contract: object

    def __init__(self, *, menu_loader: MenuLoader | None = None) -> None:
        # ``menu_loader`` is a focused-test seam.  Production always enters the
        # checked-in workstation producer and never treats a precomputed
        # probability transport as the physical execution path.
        self._menu_loader = menu_loader or _materialize_workstation_menu
        self._menu: HarpPredictionMenuSeal | None = None
        self._readiness: HarpInputReadiness | None = None
        self._feature_payload: dict[str, object] | None = None
        self._lineage: HarpAuthoritativeLineage | None = None

    def preflight(self, config: HarpStage60Config, readiness: HarpInputReadiness) -> None:
        if (
            config.contract != self.contract
            or readiness.surface != config.contract.surface
            or readiness.experiment_id != config.experiment_id
        ):
            raise ProtocolError("HARP production adapter received another surface.")
        _validate_fresh_input_members(config, readiness)
        self._readiness = readiness

    def validate_completed_bundle(self, config: HarpStage60Config) -> HarpRunReceipt:
        return _validate_completed_bundle(config, expected_contract=self.contract)

    def _load_lineage(self, config: HarpStage60Config) -> HarpAuthoritativeLineage:
        if self._lineage is None:
            if self._menu is None:
                raise ProtocolError("HARP physical menu is absent from lineage binding.")
            self._lineage = load_authoritative_lineage(
                artifact_root=config.artifact_root,
                expert_bank_root=config.input_paths["expert_bank_root"],
                generation_lock_root=config.input_paths["generation_lock_root"],
                menu=self._menu,
            )
        return self._lineage


class ProductionActionSurfaceAdapter(_ProductionSurfaceBase):
    """Source-development adapter whose sole label edge is post-seal."""

    contract = ACTION_SURFACE

    def __init__(self, *, menu_loader: MenuLoader | None = None) -> None:
        super().__init__(menu_loader=menu_loader)
        self._seed_surface = None
        self._ensemble_surface = None
        self._features: HarpActionFeatureSurface | None = None
        self._source_capability_seal: HarpDurablePredictionSeal | None = None
        self._transport_arrays_path: Path | None = None
        self._responses: HarpDirectionalResponseSurface | None = None
        self._training_observation_payload: dict[str, object] | None = None
        self._inference_binding: HarpActionInferenceBinding | None = None

    def materialize_and_seal_label_free_menu(
        self, config: HarpStage60Config, readiness: HarpInputReadiness
    ) -> HarpDurablePrelabelSeal:
        menu = _require_preflight(self, config, readiness)
        seed_surface = _development_seed_surface(menu)
        ensembles = build_probability_ensemble_surface(
            seed_surface, expected_seed_ids=EXPECTED_SEED_IDS
        )
        features = build_action_feature_surface(ensembles)
        feature_payload = _action_feature_payload(features)
        atomic_json(config.artifact_root / ACTION_FEATURE_MEMBER, feature_payload)
        arrays_path = _persist_prelabel_catalog_members(
            config,
            readiness,
            menu,
            feature_payload=feature_payload,
        )
        capability_seal = build_durable_prediction_seal(
            probability_surface_hash=seed_surface.surface_hash,
            upstream_prediction_seal_hash=menu.seal_hash,
            prediction_artifact_sha256=sha256_file(arrays_path),
            prediction_row_count=len(seed_surface.rows),
        )
        atomic_json(
            config.artifact_root / SOURCE_CAPABILITY_SEAL_MEMBER,
            capability_seal.to_payload(),
        )
        seal = _persist_global_seal(
            config,
            probability_menu_hash=menu.seal_hash,
            row_identity_hash=canonical_hash(
                {
                    "seed_surface_hash": seed_surface.surface_hash,
                    "ensemble_surface_hash": ensembles.surface_hash,
                    "feature_surface_hash": features.surface_hash,
                    "exact_nine_before_model": True,
                }
            ),
            prelabel_members={
                PROBABILITY_ARRAY_MEMBER: sha256_file(arrays_path),
                PROBABILITY_INDEX_MEMBER: sha256_file(
                    config.artifact_root / PROBABILITY_INDEX_MEMBER
                ),
                DIRECTIONAL_FEATURES_MEMBER: sha256_file(
                    config.artifact_root / DIRECTIONAL_FEATURES_MEMBER
                ),
            },
        )
        self._seed_surface = seed_surface
        self._ensemble_surface = ensembles
        self._features = features
        self._feature_payload = feature_payload
        self._source_capability_seal = capability_seal
        self._transport_arrays_path = arrays_path
        return seal

    def open_source_development_labels(
        self, config: HarpStage60Config, seal: HarpDurablePrelabelSeal
    ) -> OpenedHarpSourceLabels:
        seal.verify_durable()
        if (
            self._source_capability_seal is None
            or self._transport_arrays_path is None
            or seal.surface != ACTION_SURFACE.surface
        ):
            raise ProtocolError("HARP source labels were requested before complete sealing.")
        capability = HarpSourceLabelCapability(
            centers=tuple(config.protocol["center_universe"]),
            seal=self._source_capability_seal,
            seal_path=config.artifact_root / SOURCE_CAPABILITY_SEAL_MEMBER,
            prediction_artifact_path=self._transport_arrays_path,
            label_loader=lambda: _read_fresh_source_labels(
                config.input_paths["development_manifest_path"]
            ),
        )
        return capability.open()

    def build_product(
        self,
        config: HarpStage60Config,
        seal: HarpDurablePrelabelSeal,
        source_development_labels: object | None,
    ) -> HarpBuiltProduct:
        seal.verify_durable()
        if not isinstance(source_development_labels, OpenedHarpSourceLabels) or self._features is None:
            raise ProtocolError("HARP action product requires post-seal source labels.")
        responses = build_directional_response_surface(
            self._features, source_development_labels
        )
        self._responses = responses
        from ..harp_action_model import (
            training_observation_surface_payload,
            training_observations_from_surfaces,
        )

        training_rows = training_observations_from_surfaces(self._features, responses)
        training_payload = training_observation_surface_payload(
            training_rows,
            feature_surface_hash=self._features.surface_hash,
            response_surface_hash=responses.surface_hash,
        )
        menu = self._menu
        if menu is None:
            raise ProtocolError("HARP action menu disappeared before product binding.")
        lineage = self._load_lineage(config)
        semantic_lineage = lineage.semantic_payload()
        authoritative_lineage = lineage.authoritative_receipt_payload()
        inference_binding = HarpActionInferenceBinding.from_stage60_lineage(
            lineage,
            global_prediction_seal_semantic_id=seal.seal_hash,
            feature_surface_semantic_id=self._features.surface_hash,
            response_surface_semantic_id=responses.surface_hash,
        )
        self._training_observation_payload = training_payload
        self._inference_binding = inference_binding
        payload: dict[str, object] = {
            "schema_version": "midogpp_harp_action_surface_product_v1",
            "status": "COMPLETE_SOURCE_INNER_ACTION_SURFACE",
            "surface": config.contract.surface,
            "experiment_id": config.experiment_id,
            "artifact_id": config.output_artifact_id,
            "config_contract_hash": config.contract_hash,
            "prelabel_seal_hash": seal.seal_hash,
            "probability_menu_hash": seal.probability_menu_hash,
            "feature_surface_hash": self._features.surface_hash,
            "feature_artifact_hash": canonical_hash(self._feature_payload),
            "response_surface_hash": responses.surface_hash,
            "training_surface_hash": training_payload["training_surface_hash"],
            "action_inference_binding_sha256": inference_binding.binding_sha256,
            "feature_row_count": len(self._features.rows),
            "response_row_count": len(responses.rows),
            "probability_endpoint": "exact_nine_seed_ensemble_per_sample",
            "seed_cells_may_feed_model": False,
            "case_equal_weighting_required": True,
            "source_development_labels_used_for_scoring_only": True,
            "target_support_labels_used": False,
            "target_evaluation_labels_used": False,
            **semantic_lineage,
            **authoritative_lineage,
            "menu_seal_hash": menu.seal_hash,
        }
        return HarpBuiltProduct(
            config.contract.surface,
            payload,
            canonical_hash(payload),
            True,
        )

    def persist_product(
        self,
        config: HarpStage60Config,
        seal: HarpDurablePrelabelSeal,
        product: HarpBuiltProduct,
    ) -> Path:
        if (
            self._responses is None
            or self._feature_payload is None
            or self._training_observation_payload is None
            or self._inference_binding is None
        ):
            raise ProtocolError("HARP action responses were not built before persistence.")
        seal.verify_durable()
        response_payload = _response_payload(self._responses)
        if product.payload.get("response_surface_hash") != self._responses.surface_hash:
            raise ProtocolError("HARP action product escaped its response surface.")
        atomic_json(config.artifact_root / ACTION_RESPONSE_MEMBER, response_payload)
        atomic_json(
            config.artifact_root / TRAINING_OBSERVATION_MEMBER,
            self._training_observation_payload,
        )
        atomic_json(
            config.artifact_root / ACTION_INFERENCE_BINDING_MEMBER,
            self._inference_binding.to_payload(),
        )
        _write_directional_response_table(
            config.artifact_root / DIRECTIONAL_RESPONSES_MEMBER,
            self._responses,
        )
        surface_lock = _action_surface_lock_payload(
            product,
            seal=seal,
            feature_payload=self._feature_payload,
            response_payload=response_payload,
            training_payload=self._training_observation_payload,
            inference_binding=self._inference_binding,
        )
        atomic_json(config.artifact_root / ACTION_LOCK_MEMBER, surface_lock)
        _persist_product_and_receipts(
            config,
            seal,
            product,
            surface_lock_member=ACTION_LOCK_MEMBER,
            authoritative_extra=(
                ACTION_FEATURE_MEMBER,
                ACTION_RESPONSE_MEMBER,
                TRAINING_OBSERVATION_MEMBER,
                ACTION_INFERENCE_BINDING_MEMBER,
                SOURCE_CAPABILITY_SEAL_MEMBER,
                DIRECTIONAL_RESPONSES_MEMBER,
                LINEAGE_RECEIPT_MEMBER,
            ),
            validation_extra={
                "feature_artifact_hash": canonical_hash(self._feature_payload),
                "response_artifact_hash": canonical_hash(response_payload),
                "training_surface_hash": self._training_observation_payload[
                    "training_surface_hash"
                ],
                "action_inference_binding_sha256": (
                    self._inference_binding.binding_sha256
                ),
                "action_surface_lock_hash": surface_lock["action_surface_lock_hash"],
                "source_development_labels_used_for_scoring_only": True,
                "seed_cells_may_feed_model": False,
            },
        )
        return config.artifact_root


class ProductionTargetSupportAdapter(_ProductionSurfaceBase):
    """Permanently label-free target-support adapter."""

    contract = TARGET_SUPPORT_SURFACE

    def materialize_and_seal_label_free_menu(
        self, config: HarpStage60Config, readiness: HarpInputReadiness
    ) -> HarpDurablePrelabelSeal:
        menu = _require_preflight(self, config, readiness)
        payload = _target_support_payload(menu)
        atomic_json(config.artifact_root / TARGET_SUPPORT_MEMBER, payload)
        arrays_path = _persist_prelabel_catalog_members(
            config,
            readiness,
            menu,
            feature_payload=payload,
        )
        self._feature_payload = payload
        return _persist_global_seal(
            config,
            probability_menu_hash=menu.seal_hash,
            row_identity_hash=str(payload["surface_hash"]),
            prelabel_members={
                PROBABILITY_ARRAY_MEMBER: sha256_file(arrays_path),
                PROBABILITY_INDEX_MEMBER: sha256_file(
                    config.artifact_root / PROBABILITY_INDEX_MEMBER
                ),
                DIRECTIONAL_FEATURES_MEMBER: sha256_file(
                    config.artifact_root / DIRECTIONAL_FEATURES_MEMBER
                ),
            },
        )

    def open_source_development_labels(
        self, config: HarpStage60Config, seal: HarpDurablePrelabelSeal
    ) -> object:
        del config, seal
        raise ProtocolError("HARP target-support adapter has no label capability.")

    def build_product(
        self,
        config: HarpStage60Config,
        seal: HarpDurablePrelabelSeal,
        source_development_labels: object | None,
    ) -> HarpBuiltProduct:
        seal.verify_durable()
        if source_development_labels is not None or self._feature_payload is None:
            raise ProtocolError("HARP target-support product cannot receive outcomes.")
        lineage = self._load_lineage(config)
        payload: dict[str, object] = {
            "schema_version": "midogpp_harp_target_support_surface_product_v1",
            "status": "COMPLETE_LABEL_FREE_TARGET_SUPPORT_SURFACE",
            "surface": config.contract.surface,
            "experiment_id": config.experiment_id,
            "artifact_id": config.output_artifact_id,
            "config_contract_hash": config.contract_hash,
            "prelabel_seal_hash": seal.seal_hash,
            "probability_menu_hash": seal.probability_menu_hash,
            "target_support_surface_hash": self._feature_payload["surface_hash"],
            "target_support_artifact_hash": canonical_hash(self._feature_payload),
            "feature_row_count": self._feature_payload["row_count"],
            "probability_endpoint": "exact_nine_seed_ensemble_per_sample",
            "seed_cells_may_feed_model": False,
            "case_equal_weighting_required": True,
            "source_development_labels_used_for_scoring_only": False,
            "target_support_labels_used": False,
            "target_evaluation_labels_used": False,
            **lineage.semantic_payload(),
            **lineage.authoritative_receipt_payload(),
            "menu_seal_hash": self._menu.seal_hash,
        }
        return HarpBuiltProduct(
            config.contract.surface,
            payload,
            canonical_hash(payload),
            False,
        )

    def persist_product(
        self,
        config: HarpStage60Config,
        seal: HarpDurablePrelabelSeal,
        product: HarpBuiltProduct,
    ) -> Path:
        if self._feature_payload is None:
            raise ProtocolError("HARP target-support surface was not built.")
        seal.verify_durable()
        surface_lock = _target_support_lock_payload(
            product,
            seal=seal,
            feature_payload=self._feature_payload,
        )
        atomic_json(config.artifact_root / TARGET_SUPPORT_LOCK_MEMBER, surface_lock)
        _persist_product_and_receipts(
            config,
            seal,
            product,
            surface_lock_member=TARGET_SUPPORT_LOCK_MEMBER,
            authoritative_extra=(TARGET_SUPPORT_MEMBER, LINEAGE_RECEIPT_MEMBER),
            validation_extra={
                "target_support_artifact_hash": canonical_hash(self._feature_payload),
                "target_support_surface_lock_hash": surface_lock[
                    "target_support_surface_lock_hash"
                ],
                "outcomes_accessible": False,
                "seed_cells_may_feed_model": False,
            },
        )
        return config.artifact_root


def _cache_root(config: HarpStage60Config) -> Path:
    return config.input_paths[
        "development_cache_root"
        if config.contract == ACTION_SURFACE
        else "target_support_cache_root"
    ]


def _reservation_root(config: HarpStage60Config) -> Path:
    return config.input_paths[
        "development_reservation_root"
        if config.contract == ACTION_SURFACE
        else "target_support_reservation_root"
    ]


def _validate_fresh_input_members(
    config: HarpStage60Config, readiness: HarpInputReadiness
) -> None:
    for key in ("expert_bank_root", "generation_lock_root"):
        if not config.input_paths[key].is_dir():
            raise ProtocolError(f"HARP fresh input root is absent: {key}.")
    cache = _cache_root(config)
    reservation = _reservation_root(config)
    if not cache.is_dir() or not reservation.is_dir():
        raise ProtocolError("HARP fresh cache or reservation root is absent.")
    reservation_member = reservation / RESERVATION_MEMBER
    if not reservation_member.is_file() or sha256_file(reservation_member) != readiness.reservation_sha256:
        raise ProtocolError("HARP reservation bytes differ from readiness.")
    attestation = config.input_paths["readiness_attestation_path"]
    if (
        not attestation.is_file()
        or sha256_file(attestation) != readiness.attestation_sha256
    ):
        raise ProtocolError("HARP readiness-attestation bytes drifted.")
    cache_members = (
        "manifests/cache_index.json",
        "manifests/content_index.json",
        "tables/row_index.csv",
    )
    missing_cache = tuple(member for member in cache_members if not (cache / member).is_file())
    if missing_cache:
        raise ProtocolError(f"HARP fresh cache is incomplete: {missing_cache}.")
    content = read_json(cache / "manifests/content_index.json")
    indexed = content.get("members", content.get("member_sha256"))
    if not isinstance(indexed, Mapping) and isinstance(content.get("files"), list):
        indexed = {
            str(item.get("path", "")): item.get("sha256")
            for item in content["files"]
            if isinstance(item, Mapping)
        }
    if not isinstance(indexed, Mapping) or not indexed:
        raise ProtocolError("HARP fresh cache content index has no member inventory.")
    for relative, digest in indexed.items():
        if (
            type(relative) is not str
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ProtocolError("HARP fresh cache content path is unsafe.")
        expected = require_sha256(digest, name="HARP cache member hash")
        member = cache / relative
        if not member.is_file() or sha256_file(member) != expected:
            raise ProtocolError("HARP fresh cache content bytes drifted.")
    if config.contract == ACTION_SURFACE:
        manifest = config.input_paths["development_manifest_path"]
        if not manifest.is_file() or sha256_file(manifest) != readiness.manifest_sha256:
            raise ProtocolError("HARP fresh source manifest differs from readiness.")


def _require_preflight(
    adapter: _ProductionSurfaceBase,
    config: HarpStage60Config,
    readiness: HarpInputReadiness,
) -> HarpPredictionMenuSeal:
    if adapter._readiness != readiness or config.contract != adapter.contract:
        raise ProtocolError("HARP adapter was not preflighted before materialization.")
    if adapter._menu is None:
        menu = adapter._menu_loader(config, readiness)
        if not isinstance(menu, HarpPredictionMenuSeal):
            raise ProtocolError("HARP menu materializer returned an untyped product.")
        menu.assert_valid()
        expected_actions = (
            build_all_development_actions()
            if config.contract == ACTION_SURFACE
            else build_all_target_actions()
        )
        if tuple(action.action_hash for action in menu.actions) != tuple(
            action.action_hash for action in expected_actions
        ):
            raise ProtocolError("HARP fresh menu lacks global center/action coverage.")
        adapter._menu = menu
    adapter._menu.assert_valid()
    return adapter._menu


def _materialize_workstation_menu(
    config: HarpStage60Config, readiness: HarpInputReadiness
) -> HarpPredictionMenuSeal:
    # Lazy import preserves the CUDA-free/readiness-before-runtime boundary.
    from .workstation_producer import materialize_harp_probability_menu

    return materialize_harp_probability_menu(config, readiness)


LINEAGE_PRODUCT_FIELDS = (
    "bank_semantic_lock_hash",
    "generation_semantic_lock_hash",
    "source_stream_lock_hash",
    "source_stream_index_hash",
    "source_stream_content_hash",
    "classifier_config_hash",
    "expert_bank_index_sha256",
    "generation_lock_file_sha256",
    "source_cache_lock_sha256",
    "source_cache_index_sha256",
    "source_stream_artifact_binding_hash",
    "classifier_contract_sha256",
)


def _product_lineage(product: HarpBuiltProduct) -> dict[str, object]:
    try:
        values = {field: product.payload[field] for field in LINEAGE_PRODUCT_FIELDS}
    except KeyError as exc:
        raise ProtocolError("HARP product lacks authoritative lineage.") from exc
    return values


def _action_surface_lock_payload(
    product: HarpBuiltProduct,
    *,
    seal: HarpDurablePrelabelSeal,
    feature_payload: Mapping[str, object],
    response_payload: Mapping[str, object],
    training_payload: Mapping[str, object],
    inference_binding: HarpActionInferenceBinding,
) -> dict[str, object]:
    unhashed: dict[str, object] = {
        "schema_version": "midogpp_harp_action_surface_lock_v1",
        "status": "COMPLETE_SOURCE_INNER_ACTION_SURFACE",
        "product_hash": product.product_hash,
        "global_prediction_seal_hash": seal.seal_hash,
        "feature_surface_hash": feature_payload["surface_hash"],
        "response_surface_hash": response_payload["surface_hash"],
        "training_surface_hash": training_payload["training_surface_hash"],
        "action_inference_binding_sha256": inference_binding.binding_sha256,
        **_product_lineage(product),
        "menu_seal_hash": product.payload["menu_seal_hash"],
        "seed_cells_may_feed_model": False,
        "target_support_labels_used": False,
        "target_evaluation_labels_used": False,
    }
    return {**unhashed, "action_surface_lock_hash": canonical_hash(unhashed)}


def _target_support_lock_payload(
    product: HarpBuiltProduct,
    *,
    seal: HarpDurablePrelabelSeal,
    feature_payload: Mapping[str, object],
) -> dict[str, object]:
    unhashed: dict[str, object] = {
        "schema_version": "midogpp_harp_target_support_surface_lock_v1",
        "status": "COMPLETE_LABEL_FREE_TARGET_SUPPORT_SURFACE",
        "product_hash": product.product_hash,
        "global_prediction_seal_hash": seal.seal_hash,
        "target_support_surface_hash": feature_payload["surface_hash"],
        **_product_lineage(product),
        "menu_seal_hash": product.payload["menu_seal_hash"],
        "outcomes_accessible": False,
        "target_support_labels_used": False,
        "target_evaluation_labels_used": False,
    }
    return {
        **unhashed,
        "target_support_surface_lock_hash": canonical_hash(unhashed),
    }


def _persist_global_seal(
    config: HarpStage60Config,
    *,
    probability_menu_hash: str,
    row_identity_hash: str,
    prelabel_members: Mapping[str, str],
) -> HarpDurablePrelabelSeal:
    expected_prelabel = {
        PROBABILITY_ARRAY_MEMBER,
        PROBABILITY_INDEX_MEMBER,
        DIRECTIONAL_FEATURES_MEMBER,
    }
    if set(prelabel_members) != expected_prelabel:
        raise ProtocolError("HARP prelabel member inventory drifted.")
    member_hashes = {
        name: require_sha256(value, name=f"HARP prelabel {name}")
        for name, value in sorted(prelabel_members.items())
    }
    unhashed = {
        "schema_version": "midogpp_harp_durable_prelabel_seal_v1",
        "status": "SEALED_COMPLETE_LABEL_FREE_MENU",
        "surface": config.contract.surface,
        "probability_menu_hash": require_sha256(
            probability_menu_hash, name="HARP probability menu hash"
        ),
        "row_identity_hash": require_sha256(row_identity_hash, name="HARP row identity hash"),
        "prelabel_member_sha256": member_hashes,
        "all_probability_and_feature_members_durable": True,
        "target_support_labels_used": False,
        "target_evaluation_labels_used": False,
        "source_development_labels_opened": False,
    }
    seal_hash = canonical_hash(unhashed)
    path = config.artifact_root / GLOBAL_SEAL_MEMBER
    atomic_json(path, {**unhashed, "seal_hash": seal_hash})
    return HarpDurablePrelabelSeal(
        config.contract.surface,
        path,
        seal_hash,
        probability_menu_hash,
        row_identity_hash,
    )


def _persist_product_and_receipts(
    config: HarpStage60Config,
    seal: HarpDurablePrelabelSeal,
    product: HarpBuiltProduct,
    *,
    surface_lock_member: str,
    authoritative_extra: Sequence[str],
    validation_extra: Mapping[str, object],
) -> None:
    if product.product_hash != canonical_hash(dict(product.payload)):
        raise ProtocolError("HARP product hash drifted before persistence.")
    atomic_json(
        config.artifact_root / PRODUCT_MEMBER,
        {**dict(product.payload), "product_hash": product.product_hash},
    )
    leakage_unhashed: dict[str, object] = {
        "schema_version": "midogpp_harp_surface_leakage_report_v1",
        "status": "PASS",
        "surface": config.contract.surface,
        "strict_outer_target_query_candidate_exclusion": True,
        "outer_target_excluded_before_transform": True,
        "seed_cells_treated_as_model_observations": False,
        "model_observation_unit": "sample_with_equal_case_total_mass",
        "source_development_labels_used_for_scoring_only": (
            config.contract == ACTION_SURFACE
        ),
        "source_labels_opened_after_complete_global_prediction_seal": (
            config.contract == ACTION_SURFACE
        ),
        "target_support_labels_used": False,
        "target_evaluation_labels_used": False,
        "stage50_artifacts_used": False,
        "stage90_artifacts_used": False,
        "consumed_test_rows_used": False,
    }
    leakage = {
        **leakage_unhashed,
        "leakage_report_hash": canonical_hash(leakage_unhashed),
    }
    atomic_json(config.artifact_root / LEAKAGE_MEMBER, leakage)
    validation_unhashed = {
        "schema_version": "midogpp_harp_surface_validation_v1",
        "status": "PASS",
        "surface": config.contract.surface,
        "product_hash": product.product_hash,
        "prelabel_seal_hash": seal.seal_hash,
        "surface_lock_member": surface_lock_member,
        "surface_lock_sha256": sha256_file(config.artifact_root / surface_lock_member),
        "leakage_report_hash": leakage["leakage_report_hash"],
        "target_support_labels_used": False,
        "target_evaluation_labels_used": False,
        **dict(validation_extra),
    }
    validation_hash = canonical_hash(validation_unhashed)
    atomic_json(
        config.artifact_root / VALIDATION_MEMBER,
        {**validation_unhashed, "validation_hash": validation_hash},
    )
    required = (
        ACTION_REQUIRED_MEMBERS
        if config.contract == ACTION_SURFACE
        else TARGET_REQUIRED_MEMBERS
    )
    indexed = tuple(
        sorted(
            (required - {CONTENT_INDEX_MEMBER, STATE_MEMBER})
            | {PRODUCT_MEMBER, *authoritative_extra}
        )
    )
    missing = tuple(member for member in indexed if not (config.artifact_root / member).is_file())
    if missing:
        raise ProtocolError(f"HARP authoritative product members are incomplete: {missing}.")
    index_unhashed: dict[str, object] = {
        "schema_version": "midogpp_harp_surface_content_index_v1",
        "surface": config.contract.surface,
        "members": {
            member: sha256_file(config.artifact_root / member) for member in indexed
        },
    }
    atomic_json(
        config.artifact_root / CONTENT_INDEX_MEMBER,
        {**index_unhashed, "content_index_hash": canonical_hash(index_unhashed)},
    )
    # This is the commit marker and is intentionally published last.
    atomic_json(
        config.artifact_root / STATE_MEMBER,
        {
            "schema_version": "midogpp_harp_run_state_v1",
            "status": "COMPLETE",
            "surface": config.contract.surface,
            "experiment_id": config.experiment_id,
            "product_hash": product.product_hash,
            "validation_hash": validation_hash,
            "target_support_labels_used": False,
            "target_evaluation_labels_used": False,
        },
    )


__all__ = (
    "ProductionActionSurfaceAdapter",
    "ProductionTargetSupportAdapter",
    "write_probability_menu_transport",
)
