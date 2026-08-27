"""Frozen adaptive-development model and higher-is-better route replay.

This module is deliberately diagnostic-owned.  It freezes the pairwise model
learned from the explicitly authorized historical source-inner surface and
binds it to one outer target, one exact GenerationLock, one exact candidate
menu, and the canonical exact-B control.  It does not change the reusable
label-free core's lower-is-better proxy-energy semantics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from typing import TypeAlias

import numpy as np

from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    legal_routing_sources,
)
from midogpp_thesis.cvae.generation.contracts import GenerationLock
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.sceptre.candidate_menu import build_candidate_menu
from midogpp_thesis.cvae.routing.sceptre.contracts import CandidateMenu
from midogpp_thesis.cvae.routing.sceptre.control import (
    ControlValidationReceipt,
    validate_candidate_and_b_control,
)

from .development_model import (
    EvidenceFeatureRow,
    NestedLodoFit,
    PairwiseUtilityModel,
)
from .hashing import canonical_bytes, canonical_hash, require_sha256


FROZEN_MODEL_SCHEMA = "sceptre_adaptive_utility_model_freeze_v1"
FROZEN_MODEL_ROLE = "HISTORICAL_SOURCE_INNER_ADAPTIVE_DEVELOPMENT_ONLY"
FROZEN_MODEL_PUBLICATION_STATUS = "POST_HOC_CONSUMED_TEST_SENSITIVITY"
PREDICTED_UTILITY_SEMANTICS = "ADAPTIVE_PREDICTED_UTILITY_HIGHER_IS_BETTER"
PREDICTED_UTILITY_POLICY_ID = "G_HISTORICAL_ADAPTIVE_PREDICTED_UTILITY"
EXACT_B_CONTROL_ID = "B"
EXACT_UTILITY_TIE_REASON = "EXACT_PREDICTED_UTILITY_TIE_FALLBACK_TO_B"
MISSING_UTILITY_EVIDENCE_REASON = "MISSING_PREDICTED_UTILITY_EVIDENCE_FALLBACK_TO_B"
INVALID_UTILITY_EVIDENCE_REASON = "INVALID_PREDICTED_UTILITY_EVIDENCE_FALLBACK_TO_B"
_FALLBACK_REASONS = frozenset(
    {
        EXACT_UTILITY_TIE_REASON,
        MISSING_UTILITY_EVIDENCE_REASON,
        INVALID_UTILITY_EVIDENCE_REASON,
    }
)


def _finite(value: object, role: str) -> float:
    if isinstance(value, bool):
        raise ProtocolError(f"SCEPTRE frozen {role} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError(f"SCEPTRE frozen {role} must be finite.") from exc
    if not math.isfinite(parsed):
        raise ProtocolError(f"SCEPTRE frozen {role} must be finite.")
    return parsed


def _identifier(value: object, role: str) -> str:
    text = str(value)
    if not text or text.strip() != text:
        raise ProtocolError(f"SCEPTRE frozen {role} is invalid.")
    return text


def _tuple_field(value: object, role: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ProtocolError(f"SCEPTRE frozen {role} is not a sequence.")
    return tuple(value)


def _pairwise_model_receipt_hash(
    *,
    outer_target: str,
    candidate_centers: tuple[str, ...],
    feature_names: tuple[str, ...],
    feature_means: tuple[float, ...],
    feature_scales: tuple[float, ...],
    coefficients: tuple[float, ...],
    alpha: float,
    training_query_centers: tuple[str, ...],
    parent_exclusion_receipt_hash: str,
    evidence_receipt_hash: str,
) -> str:
    training_keys = tuple(
        sorted(
            (query, candidate)
            for query in training_query_centers
            for candidate in candidate_centers
            if query != candidate
        )
    )
    return canonical_hash(
        {
            "schema_version": "sceptre_pairwise_utility_model_v1",
            "outer_target": outer_target,
            "candidate_centers": list(candidate_centers),
            "feature_names": list(feature_names),
            "feature_means": list(feature_means),
            "feature_scales": list(feature_scales),
            "coefficients": list(coefficients),
            "alpha": alpha,
            "training_query_centers": list(training_query_centers),
            "training_keys": [list(key) for key in training_keys],
            "parent_exclusion_receipt_hash": parent_exclusion_receipt_hash,
            "evidence_transform_receipt_hash": evidence_receipt_hash,
        }
    )


@dataclass(frozen=True, slots=True)
class FrozenAdaptiveUtilityModel:
    """Canonical, target-specific freeze of one historical development model."""

    outer_target: str
    candidate_sources: tuple[str, ...]
    generation_lock_hash: str
    generation_lock_payload_sha256: str
    bank_lock_hash: str
    candidate_menu_hash: str
    candidate_menu_payload_sha256: str
    candidate_family_hashes: tuple[str, ...]
    exact_b_control_receipt_hash: str
    exact_b_control_payload_sha256: str
    feature_names: tuple[str, ...]
    selected_alpha: float
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    training_query_centers: tuple[str, ...]
    parent_exclusion_receipt_sha256: str
    evidence_transform_receipt_sha256: str
    outer_evidence_receipt_sha256: str
    training_receipt_sha256: str
    nested_lodo_receipt_sha256: str
    model_sha256: str = ""

    def __post_init__(self) -> None:
        target = _identifier(self.outer_target, "outer target")
        if target not in CENTERS:
            raise ProtocolError("SCEPTRE frozen model outer target is unknown.")
        sources = tuple(
            _identifier(value, "candidate source") for value in self.candidate_sources
        )
        expected_sources = legal_routing_sources(target)
        if sources != expected_sources:
            raise ProtocolError("SCEPTRE frozen model is not bound to exact C minus H.")
        training_queries = tuple(
            _identifier(value, "training query center")
            for value in self.training_query_centers
        )
        if training_queries != expected_sources:
            raise ProtocolError(
                "SCEPTRE frozen model training-query inventory drifted."
            )

        feature_names = tuple(
            _identifier(value, "feature name") for value in self.feature_names
        )
        from .evidence_builder import FEATURE_NAMES

        if feature_names != FEATURE_NAMES:
            raise ProtocolError("SCEPTRE frozen feature schema drifted.")
        if len(set(feature_names)) != len(feature_names):
            raise ProtocolError("SCEPTRE frozen feature schema is invalid.")
        means = tuple(_finite(value, "feature mean") for value in self.feature_means)
        scales = tuple(_finite(value, "feature scale") for value in self.feature_scales)
        coefficients = tuple(
            _finite(value, "model coefficient") for value in self.coefficients
        )
        if len(means) != len(feature_names) or len(scales) != len(feature_names):
            raise ProtocolError("SCEPTRE frozen normalization geometry drifted.")
        if any(value <= 0.0 for value in scales):
            raise ProtocolError("SCEPTRE frozen feature scales must be positive.")
        expected_coefficients = (
            len(sources) + len(feature_names) + len(sources) * len(feature_names)
        )
        if len(coefficients) != expected_coefficients:
            raise ProtocolError("SCEPTRE frozen coefficient geometry drifted.")
        alpha = _finite(self.selected_alpha, "selected alpha")
        if alpha <= 0.0:
            raise ProtocolError("SCEPTRE frozen selected alpha must be positive.")

        family_hashes = tuple(
            _identifier(value, "candidate-family hash")
            for value in self.candidate_family_hashes
        )
        if len(family_hashes) != len(sources) or len(set(family_hashes)) != len(
            family_hashes
        ):
            raise ProtocolError("SCEPTRE frozen candidate-family hashes drifted.")

        object.__setattr__(self, "outer_target", target)
        object.__setattr__(self, "candidate_sources", sources)
        object.__setattr__(self, "generation_lock_hash", _identifier(
            self.generation_lock_hash, "GenerationLock hash"
        ))
        object.__setattr__(self, "generation_lock_payload_sha256", require_sha256(
            self.generation_lock_payload_sha256, "GenerationLock payload"
        ))
        object.__setattr__(self, "bank_lock_hash", _identifier(
            self.bank_lock_hash, "bank-lock hash"
        ))
        object.__setattr__(self, "candidate_menu_hash", _identifier(
            self.candidate_menu_hash, "candidate-menu hash"
        ))
        object.__setattr__(self, "candidate_menu_payload_sha256", require_sha256(
            self.candidate_menu_payload_sha256, "candidate-menu payload"
        ))
        object.__setattr__(self, "candidate_family_hashes", family_hashes)
        object.__setattr__(self, "exact_b_control_receipt_hash", _identifier(
            self.exact_b_control_receipt_hash, "exact-B control receipt hash"
        ))
        object.__setattr__(self, "exact_b_control_payload_sha256", require_sha256(
            self.exact_b_control_payload_sha256, "exact-B control payload"
        ))
        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "selected_alpha", alpha)
        object.__setattr__(self, "feature_means", means)
        object.__setattr__(self, "feature_scales", scales)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "training_query_centers", training_queries)
        object.__setattr__(self, "parent_exclusion_receipt_sha256", require_sha256(
            self.parent_exclusion_receipt_sha256, "parent exclusion receipt"
        ))
        object.__setattr__(self, "evidence_transform_receipt_sha256", require_sha256(
            self.evidence_transform_receipt_sha256, "evidence-transform receipt"
        ))
        object.__setattr__(self, "outer_evidence_receipt_sha256", require_sha256(
            self.outer_evidence_receipt_sha256, "outer-evidence receipt"
        ))
        if (
            self.evidence_transform_receipt_sha256
            != self.outer_evidence_receipt_sha256
        ):
            raise ProtocolError(
                "SCEPTRE frozen model and outer evidence receipts differ."
            )
        object.__setattr__(self, "training_receipt_sha256", require_sha256(
            self.training_receipt_sha256, "model training receipt"
        ))
        object.__setattr__(self, "nested_lodo_receipt_sha256", require_sha256(
            self.nested_lodo_receipt_sha256, "nested-LODO receipt"
        ))
        expected_training_receipt = _pairwise_model_receipt_hash(
            outer_target=target,
            candidate_centers=sources,
            feature_names=feature_names,
            feature_means=means,
            feature_scales=scales,
            coefficients=coefficients,
            alpha=alpha,
            training_query_centers=training_queries,
            parent_exclusion_receipt_hash=(
                self.parent_exclusion_receipt_sha256
            ),
            evidence_receipt_hash=self.evidence_transform_receipt_sha256,
        )
        if self.training_receipt_sha256 != expected_training_receipt:
            raise ProtocolError(
                "SCEPTRE frozen pairwise-model receipt does not replay."
            )

        expected_hash = canonical_hash(self._payload_without_hash())
        if self.model_sha256 and require_sha256(
            self.model_sha256, "frozen model"
        ) != expected_hash:
            raise ProtocolError("SCEPTRE frozen model SHA-256 drifted.")
        object.__setattr__(self, "model_sha256", expected_hash)

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": FROZEN_MODEL_SCHEMA,
            "artifact_role": FROZEN_MODEL_ROLE,
            "publication_status": FROZEN_MODEL_PUBLICATION_STATUS,
            "score_semantics": PREDICTED_UTILITY_SEMANTICS,
            "higher_is_better": True,
            "route_time_labels_consumed": False,
            "outer_target": self.outer_target,
            "identities": {
                "generation_lock_hash": self.generation_lock_hash,
                "generation_lock_payload_sha256": self.generation_lock_payload_sha256,
                "bank_lock_hash": self.bank_lock_hash,
                "candidate_menu_hash": self.candidate_menu_hash,
                "candidate_menu_payload_sha256": self.candidate_menu_payload_sha256,
                "candidate_sources": list(self.candidate_sources),
                "candidate_family_hashes": list(self.candidate_family_hashes),
                "exact_b_control_receipt_hash": self.exact_b_control_receipt_hash,
                "exact_b_control_payload_sha256": (
                    self.exact_b_control_payload_sha256
                ),
            },
            "model": {
                "feature_names": list(self.feature_names),
                "selected_alpha": self.selected_alpha,
                "feature_means": list(self.feature_means),
                "feature_scales": list(self.feature_scales),
                "coefficients": list(self.coefficients),
                "training_query_centers": list(self.training_query_centers),
                "parent_exclusion_receipt_sha256": (
                    self.parent_exclusion_receipt_sha256
                ),
                "evidence_transform_receipt_sha256": (
                    self.evidence_transform_receipt_sha256
                ),
                "outer_evidence_receipt_sha256": (
                    self.outer_evidence_receipt_sha256
                ),
                "training_receipt_sha256": self.training_receipt_sha256,
                "nested_lodo_receipt_sha256": self.nested_lodo_receipt_sha256,
            },
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload_without_hash(), "model_sha256": self.model_sha256}

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_payload())

    def reconstruct_model(self) -> PairwiseUtilityModel:
        return PairwiseUtilityModel(
            outer_target=self.outer_target,
            candidate_centers=self.candidate_sources,
            feature_names=self.feature_names,
            feature_means=self.feature_means,
            feature_scales=self.feature_scales,
            coefficients=self.coefficients,
            alpha=self.selected_alpha,
            training_query_centers=self.training_query_centers,
            parent_exclusion_receipt_hash=(
                self.parent_exclusion_receipt_sha256
            ),
            evidence_receipt_hash=self.evidence_transform_receipt_sha256,
            training_receipt_hash=self.training_receipt_sha256,
        )

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
    ) -> "FrozenAdaptiveUtilityModel":
        top_keys = {
            "schema_version",
            "artifact_role",
            "publication_status",
            "score_semantics",
            "higher_is_better",
            "route_time_labels_consumed",
            "outer_target",
            "identities",
            "model",
            "model_sha256",
        }
        if set(payload) != top_keys:
            raise ProtocolError("SCEPTRE frozen model top-level schema drifted.")
        if (
            payload.get("schema_version") != FROZEN_MODEL_SCHEMA
            or payload.get("artifact_role") != FROZEN_MODEL_ROLE
            or payload.get("publication_status") != FROZEN_MODEL_PUBLICATION_STATUS
            or payload.get("score_semantics") != PREDICTED_UTILITY_SEMANTICS
            or payload.get("higher_is_better") is not True
            or payload.get("route_time_labels_consumed") is not False
        ):
            raise ProtocolError("SCEPTRE frozen model semantics drifted.")
        identities = payload.get("identities")
        model = payload.get("model")
        if not isinstance(identities, Mapping) or not isinstance(model, Mapping):
            raise ProtocolError("SCEPTRE frozen model sections are invalid.")
        identity_keys = {
            "generation_lock_hash",
            "generation_lock_payload_sha256",
            "bank_lock_hash",
            "candidate_menu_hash",
            "candidate_menu_payload_sha256",
            "candidate_sources",
            "candidate_family_hashes",
            "exact_b_control_receipt_hash",
            "exact_b_control_payload_sha256",
        }
        model_keys = {
            "feature_names",
            "selected_alpha",
            "feature_means",
            "feature_scales",
            "coefficients",
            "training_query_centers",
            "parent_exclusion_receipt_sha256",
            "evidence_transform_receipt_sha256",
            "outer_evidence_receipt_sha256",
            "training_receipt_sha256",
            "nested_lodo_receipt_sha256",
        }
        if set(identities) != identity_keys or set(model) != model_keys:
            raise ProtocolError("SCEPTRE frozen model nested schema drifted.")
        try:
            return cls(
                outer_target=str(payload["outer_target"]),
                candidate_sources=tuple(
                    str(value)
                    for value in _tuple_field(
                        identities["candidate_sources"], "candidate sources"
                    )
                ),
                generation_lock_hash=str(identities["generation_lock_hash"]),
                generation_lock_payload_sha256=str(
                    identities["generation_lock_payload_sha256"]
                ),
                bank_lock_hash=str(identities["bank_lock_hash"]),
                candidate_menu_hash=str(identities["candidate_menu_hash"]),
                candidate_menu_payload_sha256=str(
                    identities["candidate_menu_payload_sha256"]
                ),
                candidate_family_hashes=tuple(
                    str(value)
                    for value in _tuple_field(
                        identities["candidate_family_hashes"],
                        "candidate-family hashes",
                    )
                ),
                exact_b_control_receipt_hash=str(
                    identities["exact_b_control_receipt_hash"]
                ),
                exact_b_control_payload_sha256=str(
                    identities["exact_b_control_payload_sha256"]
                ),
                feature_names=tuple(
                    str(value)
                    for value in _tuple_field(model["feature_names"], "feature names")
                ),
                selected_alpha=_finite(model["selected_alpha"], "selected alpha"),
                feature_means=tuple(
                    _finite(value, "feature mean")
                    for value in _tuple_field(model["feature_means"], "feature means")
                ),
                feature_scales=tuple(
                    _finite(value, "feature scale")
                    for value in _tuple_field(model["feature_scales"], "feature scales")
                ),
                coefficients=tuple(
                    _finite(value, "model coefficient")
                    for value in _tuple_field(model["coefficients"], "coefficients")
                ),
                training_query_centers=tuple(
                    str(value)
                    for value in _tuple_field(
                        model["training_query_centers"], "training query centers"
                    )
                ),
                parent_exclusion_receipt_sha256=str(
                    model["parent_exclusion_receipt_sha256"]
                ),
                evidence_transform_receipt_sha256=str(
                    model["evidence_transform_receipt_sha256"]
                ),
                outer_evidence_receipt_sha256=str(
                    model["outer_evidence_receipt_sha256"]
                ),
                training_receipt_sha256=str(model["training_receipt_sha256"]),
                nested_lodo_receipt_sha256=str(
                    model["nested_lodo_receipt_sha256"]
                ),
                model_sha256=str(payload["model_sha256"]),
            )
        except KeyError as exc:
            raise ProtocolError("SCEPTRE frozen model field is absent.") from exc

    @classmethod
    def from_canonical_bytes(cls, payload: bytes) -> "FrozenAdaptiveUtilityModel":
        if not isinstance(payload, bytes):
            raise ProtocolError("SCEPTRE frozen model serialization must be bytes.")
        try:
            raw = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolError("Cannot parse SCEPTRE frozen model bytes.") from exc
        if not isinstance(raw, Mapping):
            raise ProtocolError("SCEPTRE frozen model payload is not a mapping.")
        if payload != canonical_bytes(raw):
            raise ProtocolError("SCEPTRE frozen model bytes are not canonical.")
        return cls.from_payload(raw)


@dataclass(frozen=True, slots=True)
class FrozenModelReplayReceipt:
    model_sha256: str
    outer_target: str
    generation_lock_payload_sha256: str
    candidate_menu_payload_sha256: str
    exact_b_control_payload_sha256: str
    parent_exclusion_receipt_sha256: str
    evidence_transform_receipt_sha256: str
    outer_evidence_receipt_sha256: str
    nested_lodo_receipt_sha256: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        body = {
            "schema_version": "sceptre_frozen_model_replay_receipt_v1",
            "status": "PASS",
            "model_sha256": require_sha256(self.model_sha256, "frozen model"),
            "outer_target": _identifier(self.outer_target, "outer target"),
            "generation_lock_payload_sha256": require_sha256(
                self.generation_lock_payload_sha256, "GenerationLock payload"
            ),
            "candidate_menu_payload_sha256": require_sha256(
                self.candidate_menu_payload_sha256, "candidate-menu payload"
            ),
            "exact_b_control_payload_sha256": require_sha256(
                self.exact_b_control_payload_sha256, "exact-B control payload"
            ),
            "parent_exclusion_receipt_sha256": require_sha256(
                self.parent_exclusion_receipt_sha256,
                "parent exclusion receipt",
            ),
            "evidence_transform_receipt_sha256": require_sha256(
                self.evidence_transform_receipt_sha256,
                "evidence-transform receipt",
            ),
            "outer_evidence_receipt_sha256": require_sha256(
                self.outer_evidence_receipt_sha256,
                "outer-evidence receipt",
            ),
            "nested_lodo_receipt_sha256": require_sha256(
                self.nested_lodo_receipt_sha256, "nested-LODO receipt"
            ),
        }
        expected = canonical_hash(body)
        if self.receipt_sha256 and require_sha256(
            self.receipt_sha256, "model replay receipt"
        ) != expected:
            raise ProtocolError("SCEPTRE model replay receipt SHA-256 drifted.")
        object.__setattr__(self, "receipt_sha256", expected)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "sceptre_frozen_model_replay_receipt_v1",
            "status": "PASS",
            "model_sha256": self.model_sha256,
            "outer_target": self.outer_target,
            "generation_lock_payload_sha256": self.generation_lock_payload_sha256,
            "candidate_menu_payload_sha256": self.candidate_menu_payload_sha256,
            "exact_b_control_payload_sha256": self.exact_b_control_payload_sha256,
            "parent_exclusion_receipt_sha256": (
                self.parent_exclusion_receipt_sha256
            ),
            "evidence_transform_receipt_sha256": (
                self.evidence_transform_receipt_sha256
            ),
            "outer_evidence_receipt_sha256": self.outer_evidence_receipt_sha256,
            "nested_lodo_receipt_sha256": self.nested_lodo_receipt_sha256,
            "receipt_sha256": self.receipt_sha256,
        }

def freeze_adaptive_utility_model(
    fit: NestedLodoFit,
    *,
    generation_lock: GenerationLock,
    candidate_menu: CandidateMenu,
) -> FrozenAdaptiveUtilityModel:
    """Freeze one validated historical nested-LODO model without test access."""

    _validate_nested_lodo_fit(fit)
    control = _validate_live_bindings(
        outer_target=fit.outer_target,
        generation_lock=generation_lock,
        candidate_menu=candidate_menu,
    )
    model = fit.final_model
    return FrozenAdaptiveUtilityModel(
        outer_target=fit.outer_target,
        candidate_sources=candidate_menu.candidate_sources,
        generation_lock_hash=generation_lock.generation_lock_hash,
        generation_lock_payload_sha256=canonical_hash(generation_lock.to_payload()),
        bank_lock_hash=generation_lock.bank_lock_hash,
        candidate_menu_hash=candidate_menu.menu_hash,
        candidate_menu_payload_sha256=canonical_hash(candidate_menu.to_payload()),
        candidate_family_hashes=tuple(
            family.family_hash for family in candidate_menu.families
        ),
        exact_b_control_receipt_hash=control.receipt_hash,
        exact_b_control_payload_sha256=canonical_hash(control.to_payload()),
        feature_names=model.feature_names,
        selected_alpha=fit.selected_alpha,
        feature_means=model.feature_means,
        feature_scales=model.feature_scales,
        coefficients=model.coefficients,
        training_query_centers=model.training_query_centers,
        parent_exclusion_receipt_sha256=model.parent_exclusion_receipt_hash,
        evidence_transform_receipt_sha256=model.evidence_receipt_hash,
        outer_evidence_receipt_sha256=fit.outer_evidence_receipt_hash,
        training_receipt_sha256=model.training_receipt_hash,
        nested_lodo_receipt_sha256=fit.receipt_hash,
    )


def replay_frozen_adaptive_utility_model(
    frozen: FrozenAdaptiveUtilityModel,
    fit: NestedLodoFit,
    *,
    generation_lock: GenerationLock,
    candidate_menu: CandidateMenu,
) -> FrozenModelReplayReceipt:
    """Re-freeze from development inputs and require byte-identical semantics."""

    if not isinstance(frozen, FrozenAdaptiveUtilityModel):
        raise ProtocolError("SCEPTRE model replay requires a frozen model.")
    replayed = freeze_adaptive_utility_model(
        fit,
        generation_lock=generation_lock,
        candidate_menu=candidate_menu,
    )
    if replayed.to_canonical_bytes() != frozen.to_canonical_bytes():
        raise ProtocolError("SCEPTRE frozen adaptive model replay differs.")
    round_trip = FrozenAdaptiveUtilityModel.from_canonical_bytes(
        frozen.to_canonical_bytes()
    )
    if round_trip != frozen:
        raise ProtocolError("SCEPTRE frozen adaptive model round-trip differs.")
    return FrozenModelReplayReceipt(
        model_sha256=frozen.model_sha256,
        outer_target=frozen.outer_target,
        generation_lock_payload_sha256=frozen.generation_lock_payload_sha256,
        candidate_menu_payload_sha256=frozen.candidate_menu_payload_sha256,
        exact_b_control_payload_sha256=frozen.exact_b_control_payload_sha256,
        parent_exclusion_receipt_sha256=(
            frozen.parent_exclusion_receipt_sha256
        ),
        evidence_transform_receipt_sha256=(
            frozen.evidence_transform_receipt_sha256
        ),
        outer_evidence_receipt_sha256=frozen.outer_evidence_receipt_sha256,
        nested_lodo_receipt_sha256=frozen.nested_lodo_receipt_sha256,
    )


def _validate_nested_lodo_fit(fit: NestedLodoFit) -> None:
    if not isinstance(fit, NestedLodoFit):
        raise ProtocolError("SCEPTRE freeze requires a nested-LODO fit.")
    if fit.outer_target not in CENTERS:
        raise ProtocolError("SCEPTRE nested-LODO outer target is unknown.")
    if fit.descriptive_only is not True or fit.adaptive_surface is not True:
        raise ProtocolError("SCEPTRE freeze requires adaptive descriptive development.")
    if not isinstance(fit.final_model, PairwiseUtilityModel):
        raise ProtocolError("SCEPTRE nested-LODO final model type drifted.")
    model = fit.final_model
    expected_sources = legal_routing_sources(fit.outer_target)
    if (
        model.outer_target != fit.outer_target
        or model.candidate_centers != expected_sources
        or model.training_query_centers != expected_sources
        or model.alpha != fit.selected_alpha
    ):
        raise ProtocolError("SCEPTRE nested-LODO final-model identity drifted.")
    parent_receipt = require_sha256(
        model.parent_exclusion_receipt_hash,
        "parent exclusion receipt",
    )
    evidence_receipt = require_sha256(
        model.evidence_receipt_hash,
        "evidence-transform receipt",
    )
    outer_evidence_receipt = require_sha256(
        fit.outer_evidence_receipt_hash,
        "outer-evidence receipt",
    )
    if evidence_receipt != outer_evidence_receipt:
        raise ProtocolError(
            "SCEPTRE final model and outer evidence receipts differ."
        )
    expected_model_receipt = _pairwise_model_receipt_hash(
        outer_target=model.outer_target,
        candidate_centers=model.candidate_centers,
        feature_names=model.feature_names,
        feature_means=model.feature_means,
        feature_scales=model.feature_scales,
        coefficients=model.coefficients,
        alpha=model.alpha,
        training_query_centers=model.training_query_centers,
        parent_exclusion_receipt_hash=parent_receipt,
        evidence_receipt_hash=evidence_receipt,
    )
    if expected_model_receipt != require_sha256(
        model.training_receipt_hash,
        "model training receipt",
    ):
        raise ProtocolError("SCEPTRE pairwise model receipt does not replay.")
    alpha_grid = tuple(assessment.alpha for assessment in fit.assessments)
    if (
        not alpha_grid
        or alpha_grid != tuple(sorted(set(alpha_grid)))
        or any(not math.isfinite(alpha) or alpha <= 0.0 for alpha in alpha_grid)
    ):
        raise ProtocolError("SCEPTRE nested-LODO alpha grid drifted.")
    selected_rows = tuple(
        assessment
        for assessment in fit.assessments
        if assessment.alpha == fit.selected_alpha
    )
    if len(selected_rows) != 1:
        raise ProtocolError(
            "SCEPTRE nested-LODO selected alpha is absent or duplicated."
        )
    selected = selected_rows[0]
    expected_selected = min(
        fit.assessments,
        key=lambda item: (
            item.mean_center_regret,
            item.worst_center_regret,
            -item.alpha,
        ),
    )
    if selected != expected_selected:
        raise ProtocolError("SCEPTRE nested-LODO selected alpha does not replay.")
    expected_held = expected_sources
    for assessment in fit.assessments:
        if tuple(fold.held_center for fold in assessment.folds) != expected_held:
            raise ProtocolError("SCEPTRE nested-LODO held-center rotation drifted.")
        regrets: list[float] = []
        for fold in assessment.folds:
            candidate_universe = tuple(
                center
                for center in CENTERS
                if center not in {fit.outer_target, fold.held_center}
            )
            if (
                fold.alpha != assessment.alpha
                or fold.training_query_centers != candidate_universe
                or fold.training_candidate_centers != candidate_universe
                or fold.validation_candidate_count != len(candidate_universe)
                or fold.strict_query_deletion is not True
                or fold.strict_candidate_deletion is not True
                or not fold.selected_candidate_set
                or any(
                    candidate not in candidate_universe
                    for candidate in fold.selected_candidate_set
                )
            ):
                raise ProtocolError("SCEPTRE nested-LODO fold contract drifted.")
            if tuple(
                candidate
                for candidate in candidate_universe
                if candidate in set(fold.selected_candidate_set)
            ) != fold.selected_candidate_set:
                raise ProtocolError("SCEPTRE nested-LODO winner-set order drifted.")
            require_sha256(
                fold.training_transform_receipt_hash,
                "nested training-transform receipt",
            )
            require_sha256(
                fold.validation_transform_receipt_hash,
                "nested validation-transform receipt",
            )
            regret = _finite(fold.bacc_regret, "nested-LODO regret")
            if regret < 0.0:
                raise ProtocolError("SCEPTRE nested-LODO regret is negative.")
            regrets.append(regret)
        expected_mean = float(np.mean(tuple(regrets), dtype=np.float64))
        if (
            _finite(assessment.mean_center_regret, "nested-LODO mean regret")
            != expected_mean
            or _finite(assessment.worst_center_regret, "nested-LODO worst regret")
            != max(regrets)
        ):
            raise ProtocolError("SCEPTRE nested-LODO assessment drifted.")
    receipt_body = {
        "schema_version": "sceptre_nested_lodo_fit_v1",
        "outer_target": fit.outer_target,
        "outer_exclusion_receipt_hash": parent_receipt,
        "outer_evidence_transform_receipt_hash": outer_evidence_receipt,
        "alpha_grid": list(alpha_grid),
        "selected_alpha": fit.selected_alpha,
        "folds": [
            {
                "held_center": fold.held_center,
                "bacc_regret": fold.bacc_regret,
                "selected_candidate_set": list(fold.selected_candidate_set),
                "q_and_e_deleted_before_fit": True,
                "training_transform_receipt_hash": (
                    fold.training_transform_receipt_hash
                ),
                "validation_transform_receipt_hash": (
                    fold.validation_transform_receipt_hash
                ),
            }
            for fold in selected.folds
        ],
        "seed_cells_are_nuisance_replications": True,
        "adaptive_surface": True,
        "descriptive_only": True,
        "model_receipt_hash": model.training_receipt_hash,
    }
    if canonical_hash(receipt_body) != require_sha256(
        fit.receipt_hash,
        "nested-LODO receipt",
    ):
        raise ProtocolError("SCEPTRE nested-LODO fit receipt does not replay.")


def _validate_live_bindings(
    *,
    outer_target: str,
    generation_lock: GenerationLock,
    candidate_menu: CandidateMenu,
) -> ControlValidationReceipt:
    if not isinstance(generation_lock, GenerationLock):
        raise ProtocolError("SCEPTRE freeze requires a GenerationLock.")
    if not isinstance(candidate_menu, CandidateMenu):
        raise ProtocolError("SCEPTRE freeze requires a candidate menu.")
    if candidate_menu.target_center != outer_target:
        raise ProtocolError("SCEPTRE frozen model and candidate target differ.")
    canonical_menu = build_candidate_menu(generation_lock, outer_target)
    if canonical_menu.to_payload() != candidate_menu.to_payload():
        raise ProtocolError(
            "SCEPTRE frozen candidate menu differs from GenerationLock."
        )
    return validate_candidate_and_b_control(generation_lock, candidate_menu)


def _validate_frozen_bindings(
    frozen: FrozenAdaptiveUtilityModel,
    *,
    generation_lock: GenerationLock,
    candidate_menu: CandidateMenu,
) -> ControlValidationReceipt:
    control = _validate_live_bindings(
        outer_target=frozen.outer_target,
        generation_lock=generation_lock,
        candidate_menu=candidate_menu,
    )
    observed = {
        "generation_lock_hash": generation_lock.generation_lock_hash,
        "generation_lock_payload_sha256": canonical_hash(generation_lock.to_payload()),
        "bank_lock_hash": generation_lock.bank_lock_hash,
        "candidate_menu_hash": candidate_menu.menu_hash,
        "candidate_menu_payload_sha256": canonical_hash(candidate_menu.to_payload()),
        "candidate_sources": candidate_menu.candidate_sources,
        "candidate_family_hashes": tuple(
            family.family_hash for family in candidate_menu.families
        ),
        "exact_b_control_receipt_hash": control.receipt_hash,
        "exact_b_control_payload_sha256": canonical_hash(control.to_payload()),
    }
    expected = {
        "generation_lock_hash": frozen.generation_lock_hash,
        "generation_lock_payload_sha256": frozen.generation_lock_payload_sha256,
        "bank_lock_hash": frozen.bank_lock_hash,
        "candidate_menu_hash": frozen.candidate_menu_hash,
        "candidate_menu_payload_sha256": frozen.candidate_menu_payload_sha256,
        "candidate_sources": frozen.candidate_sources,
        "candidate_family_hashes": frozen.candidate_family_hashes,
        "exact_b_control_receipt_hash": frozen.exact_b_control_receipt_hash,
        "exact_b_control_payload_sha256": frozen.exact_b_control_payload_sha256,
    }
    if observed != expected:
        raise ProtocolError("SCEPTRE frozen model identity binding drifted.")
    return control


@dataclass(frozen=True, slots=True)
class AdaptiveUtilityRanking:
    outer_target: str
    candidate_sources: tuple[str, ...]
    predicted_utility_by_source: tuple[tuple[str, float], ...]
    winner_sources: tuple[str, ...]
    frozen_model_sha256: str
    evidence_sha256: str
    ranking_sha256: str = ""

    def __post_init__(self) -> None:
        target = _identifier(self.outer_target, "utility-ranking target")
        sources = tuple(self.candidate_sources)
        if sources != legal_routing_sources(target):
            raise ProtocolError("SCEPTRE utility-ranking candidate inventory drifted.")
        scores = tuple(
            (
                _identifier(source, "utility-ranking source"),
                _finite(value, "predicted utility"),
            )
            for source, value in self.predicted_utility_by_source
        )
        if tuple(source for source, _ in scores) != sources:
            raise ProtocolError("SCEPTRE utility-ranking score order drifted.")
        maximum = max(value for _, value in scores)
        expected_winners = tuple(
            source for source, value in scores if value == maximum
        )
        winners = tuple(self.winner_sources)
        if winners != expected_winners:
            raise ProtocolError("SCEPTRE utility-ranking full tie set drifted.")
        object.__setattr__(self, "outer_target", target)
        object.__setattr__(self, "candidate_sources", sources)
        object.__setattr__(self, "predicted_utility_by_source", scores)
        object.__setattr__(self, "winner_sources", winners)
        object.__setattr__(self, "frozen_model_sha256", require_sha256(
            self.frozen_model_sha256, "frozen model"
        ))
        object.__setattr__(self, "evidence_sha256", require_sha256(
            self.evidence_sha256, "route evidence"
        ))
        expected_hash = canonical_hash(self._payload_without_hash())
        if self.ranking_sha256 and require_sha256(
            self.ranking_sha256, "utility ranking"
        ) != expected_hash:
            raise ProtocolError("SCEPTRE utility-ranking SHA-256 drifted.")
        object.__setattr__(self, "ranking_sha256", expected_hash)

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": "sceptre_adaptive_utility_ranking_v1",
            "score_semantics": PREDICTED_UTILITY_SEMANTICS,
            "higher_is_better": True,
            "outer_target": self.outer_target,
            "candidate_sources": list(self.candidate_sources),
            "predicted_utility_by_source": {
                source: value for source, value in self.predicted_utility_by_source
            },
            "winner_sources": list(self.winner_sources),
            "tie_semantics": "complete_exact_winner_set",
            "frozen_model_sha256": self.frozen_model_sha256,
            "evidence_sha256": self.evidence_sha256,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload_without_hash(), "ranking_sha256": self.ranking_sha256}


@dataclass(frozen=True, slots=True)
class AdaptiveUtilityRoute:
    outer_target: str
    candidate_sources: tuple[str, ...]
    selected_source_center: str
    frozen_model_sha256: str
    candidate_menu_payload_sha256: str
    exact_b_control_payload_sha256: str
    ranking_sha256: str
    evidence_sha256: str
    decision_sha256: str = ""

    def __post_init__(self) -> None:
        target = _identifier(self.outer_target, "adaptive-route target")
        sources = tuple(self.candidate_sources)
        selected = _identifier(self.selected_source_center, "adaptive-route source")
        if sources != legal_routing_sources(target) or selected not in sources:
            raise ProtocolError("SCEPTRE adaptive route identity drifted.")
        object.__setattr__(self, "outer_target", target)
        object.__setattr__(self, "candidate_sources", sources)
        object.__setattr__(self, "selected_source_center", selected)
        for field_name, role in (
            ("frozen_model_sha256", "frozen model"),
            ("candidate_menu_payload_sha256", "candidate-menu payload"),
            ("exact_b_control_payload_sha256", "exact-B control payload"),
            ("ranking_sha256", "utility ranking"),
            ("evidence_sha256", "route evidence"),
        ):
            object.__setattr__(
                self, field_name, require_sha256(getattr(self, field_name), role)
            )
        expected = canonical_hash(self._payload_without_hash())
        if self.decision_sha256 and require_sha256(
            self.decision_sha256, "adaptive route"
        ) != expected:
            raise ProtocolError("SCEPTRE adaptive-route SHA-256 drifted.")
        object.__setattr__(self, "decision_sha256", expected)

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": "sceptre_adaptive_utility_route_v1",
            "policy_id": PREDICTED_UTILITY_POLICY_ID,
            "score_semantics": PREDICTED_UTILITY_SEMANTICS,
            "higher_is_better": True,
            "outer_target": self.outer_target,
            "candidate_sources": list(self.candidate_sources),
            "selected_source_center": self.selected_source_center,
            "frozen_model_sha256": self.frozen_model_sha256,
            "candidate_menu_payload_sha256": self.candidate_menu_payload_sha256,
            "exact_b_control_payload_sha256": self.exact_b_control_payload_sha256,
            "ranking_sha256": self.ranking_sha256,
            "evidence_sha256": self.evidence_sha256,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload_without_hash(), "decision_sha256": self.decision_sha256}


@dataclass(frozen=True, slots=True)
class AdaptiveUtilityExactBFallback:
    outer_target: str
    candidate_sources: tuple[str, ...]
    winner_sources: tuple[str, ...]
    reason: str
    frozen_model_sha256: str
    candidate_menu_payload_sha256: str
    exact_b_control_payload_sha256: str
    evidence_sha256: str
    ranking_sha256: str | None = None
    decision_sha256: str = ""

    def __post_init__(self) -> None:
        target = _identifier(self.outer_target, "adaptive-fallback target")
        sources = tuple(self.candidate_sources)
        winners = tuple(self.winner_sources)
        reason = _identifier(self.reason, "adaptive-fallback reason")
        if sources != legal_routing_sources(target) or reason not in _FALLBACK_REASONS:
            raise ProtocolError("SCEPTRE adaptive exact-B fallback identity drifted.")
        if reason == EXACT_UTILITY_TIE_REASON:
            if len(winners) < 2 or any(source not in sources for source in winners):
                raise ProtocolError("SCEPTRE adaptive fallback lost its full tie set.")
            if tuple(source for source in sources if source in set(winners)) != winners:
                raise ProtocolError("SCEPTRE adaptive fallback tie order drifted.")
            if self.ranking_sha256 is None:
                raise ProtocolError("SCEPTRE adaptive tie fallback lacks its ranking.")
        elif winners or self.ranking_sha256 is not None:
            raise ProtocolError("SCEPTRE invalid-evidence fallback invented a ranking.")
        object.__setattr__(self, "outer_target", target)
        object.__setattr__(self, "candidate_sources", sources)
        object.__setattr__(self, "winner_sources", winners)
        object.__setattr__(self, "reason", reason)
        for field_name, role in (
            ("frozen_model_sha256", "frozen model"),
            ("candidate_menu_payload_sha256", "candidate-menu payload"),
            ("exact_b_control_payload_sha256", "exact-B control payload"),
            ("evidence_sha256", "route evidence"),
        ):
            object.__setattr__(
                self, field_name, require_sha256(getattr(self, field_name), role)
            )
        if self.ranking_sha256 is not None:
            object.__setattr__(self, "ranking_sha256", require_sha256(
                self.ranking_sha256, "utility ranking"
            ))
        expected = canonical_hash(self._payload_without_hash())
        if self.decision_sha256 and require_sha256(
            self.decision_sha256, "adaptive exact-B fallback"
        ) != expected:
            raise ProtocolError("SCEPTRE adaptive exact-B fallback SHA-256 drifted.")
        object.__setattr__(self, "decision_sha256", expected)

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": "sceptre_adaptive_utility_exact_b_fallback_v1",
            "policy_id": PREDICTED_UTILITY_POLICY_ID,
            "score_semantics": PREDICTED_UTILITY_SEMANTICS,
            "higher_is_better": True,
            "control_id": EXACT_B_CONTROL_ID,
            "outer_target": self.outer_target,
            "candidate_sources": list(self.candidate_sources),
            "winner_sources": list(self.winner_sources),
            "reason": self.reason,
            "frozen_model_sha256": self.frozen_model_sha256,
            "candidate_menu_payload_sha256": self.candidate_menu_payload_sha256,
            "exact_b_control_payload_sha256": self.exact_b_control_payload_sha256,
            "ranking_sha256": self.ranking_sha256,
            "evidence_sha256": self.evidence_sha256,
            "fake_tie_breaking": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload_without_hash(), "decision_sha256": self.decision_sha256}


AdaptiveUtilityDecision: TypeAlias = (
    AdaptiveUtilityRoute | AdaptiveUtilityExactBFallback
)


def route_frozen_predicted_utility_or_exact_b(
    frozen: FrozenAdaptiveUtilityModel,
    evidence_bundle: object,
    *,
    generation_lock: GenerationLock,
    candidate_menu: CandidateMenu,
) -> AdaptiveUtilityDecision:
    """Route from receipt-bound target evidence or deterministically use exact B."""

    if not isinstance(frozen, FrozenAdaptiveUtilityModel):
        raise ProtocolError("SCEPTRE adaptive router requires a frozen model.")
    _validate_frozen_bindings(
        frozen,
        generation_lock=generation_lock,
        candidate_menu=candidate_menu,
    )
    if evidence_bundle is None:
        return _evidence_fallback(frozen, MISSING_UTILITY_EVIDENCE_REASON)
    try:
        ordered, transform_receipt_sha256 = _validated_target_evidence(
            frozen,
            evidence_bundle,
        )
    except (ProtocolError, TypeError, ValueError, RuntimeError):
        return _evidence_fallback(frozen, INVALID_UTILITY_EVIDENCE_REASON)
    evidence_sha256 = canonical_hash(
        {
            "schema_version": "sceptre_adaptive_route_evidence_v1",
            "frozen_model_sha256": frozen.model_sha256,
            "outer_target": frozen.outer_target,
            "feature_names": list(frozen.feature_names),
            "target_transform_receipt_sha256": transform_receipt_sha256,
            "rows": [
                {
                    "candidate_center": row.candidate_center,
                    "values": list(row.values),
                    "labels_consumed": False,
                    "feature_scope": row.feature_scope,
                }
                for row in ordered
            ],
        }
    )
    model = frozen.reconstruct_model()
    try:
        scores = tuple(
            (row.candidate_center, model.predict(row)) for row in ordered
        )
    except (ProtocolError, TypeError, ValueError, OverflowError, FloatingPointError):
        return _evidence_fallback(frozen, INVALID_UTILITY_EVIDENCE_REASON)
    if any(not math.isfinite(value) for _, value in scores):
        return _evidence_fallback(frozen, INVALID_UTILITY_EVIDENCE_REASON)
    maximum = max(value for _, value in scores)
    winners = tuple(source for source, value in scores if value == maximum)
    ranking = AdaptiveUtilityRanking(
        outer_target=frozen.outer_target,
        candidate_sources=frozen.candidate_sources,
        predicted_utility_by_source=scores,
        winner_sources=winners,
        frozen_model_sha256=frozen.model_sha256,
        evidence_sha256=evidence_sha256,
    )
    if len(winners) != 1:
        return AdaptiveUtilityExactBFallback(
            outer_target=frozen.outer_target,
            candidate_sources=frozen.candidate_sources,
            winner_sources=winners,
            reason=EXACT_UTILITY_TIE_REASON,
            frozen_model_sha256=frozen.model_sha256,
            candidate_menu_payload_sha256=frozen.candidate_menu_payload_sha256,
            exact_b_control_payload_sha256=frozen.exact_b_control_payload_sha256,
            ranking_sha256=ranking.ranking_sha256,
            evidence_sha256=evidence_sha256,
        )
    return AdaptiveUtilityRoute(
        outer_target=frozen.outer_target,
        candidate_sources=frozen.candidate_sources,
        selected_source_center=winners[0],
        frozen_model_sha256=frozen.model_sha256,
        candidate_menu_payload_sha256=frozen.candidate_menu_payload_sha256,
        exact_b_control_payload_sha256=frozen.exact_b_control_payload_sha256,
        ranking_sha256=ranking.ranking_sha256,
        evidence_sha256=evidence_sha256,
    )


def _validated_target_evidence(
    frozen: FrozenAdaptiveUtilityModel,
    evidence_bundle: object,
) -> tuple[tuple[EvidenceFeatureRow, ...], str]:
    from .evidence_builder import (
        EvidenceFeatureBundle,
        build_target_prediction_evidence,
    )

    if not isinstance(evidence_bundle, EvidenceFeatureBundle):
        raise ProtocolError(
            "SCEPTRE adaptive routing requires a receipt-bound evidence bundle."
        )
    receipt = evidence_bundle.receipt
    expected_keys = tuple(
        (frozen.outer_target, candidate)
        for candidate in frozen.candidate_sources
    )
    if (
        receipt.role != "TARGET_PREDICTION"
        or receipt.target_center != frozen.outer_target
        or receipt.retained_row_count != len(frozen.candidate_sources)
        or receipt.feature_names != frozen.feature_names
        or receipt.retained_keys != expected_keys
        or receipt.labels_consumed is not False
        or receipt.exact_nelbo is not False
        or receipt.strict_filter
        != "q==H_and_e!=H_before_mean_variance_and_rank"
    ):
        raise ProtocolError("SCEPTRE target-evidence receipt identity drifted.")
    raw_keys = tuple(
        (row.query_center, row.candidate_center)
        for row in evidence_bundle.raw_rows
    )
    rows = tuple(evidence_bundle.rows)
    row_keys = tuple(row.key for row in rows)
    if raw_keys != expected_keys or row_keys != expected_keys:
        raise ProtocolError("SCEPTRE target evidence is not exact H by C minus H.")
    if any(
        row.feature_names != frozen.feature_names
        or row.labels_consumed is not False
        or row.feature_scope != "LABEL_FREE_PREDECISION_EVIDENCE"
        for row in rows
    ):
        raise ProtocolError("SCEPTRE target-evidence feature contract drifted.")
    raw_source_receipt_hash = require_sha256(
        receipt.raw_source_receipt_hash,
        "target raw-source receipt",
    )
    retained_raw_hash = require_sha256(
        receipt.retained_raw_hash,
        "target retained-raw evidence",
    )
    transformed_feature_hash = require_sha256(
        receipt.transformed_feature_hash,
        "target transformed evidence",
    )
    replayed_transform = build_target_prediction_evidence(
        evidence_bundle.raw_rows,
        target_center=frozen.outer_target,
        raw_source_receipt_hash=raw_source_receipt_hash,
    )
    if (
        replayed_transform.rows != rows
        or replayed_transform.receipt.retained_raw_hash != retained_raw_hash
        or replayed_transform.receipt.transformed_feature_hash
        != transformed_feature_hash
    ):
        raise ProtocolError("SCEPTRE target evidence transformation does not replay.")
    receipt_body = {
        "schema_version": "sceptre_evidence_transform_receipt_v1",
        "role": "TARGET_PREDICTION",
        "target_center": frozen.outer_target,
        "input_row_count": receipt.input_row_count,
        "retained_row_count": len(rows),
        "strict_filter": receipt.strict_filter,
        "retained_keys": [list(key) for key in expected_keys],
        "feature_names": list(frozen.feature_names),
        "raw_source_receipt_hash": raw_source_receipt_hash,
        "retained_raw_hash": retained_raw_hash,
        "transformed_feature_hash": transformed_feature_hash,
        "labels_consumed": False,
        "exact_nelbo": False,
    }
    expected_receipt = canonical_hash(receipt_body)
    if expected_receipt != require_sha256(
        receipt.receipt_hash,
        "target evidence-transform receipt",
    ):
        raise ProtocolError(
            "SCEPTRE target evidence-transform receipt does not replay."
        )
    return rows, expected_receipt


def _evidence_fallback(
    frozen: FrozenAdaptiveUtilityModel,
    reason: str,
) -> AdaptiveUtilityExactBFallback:
    evidence_sha256 = canonical_hash(
        {
            "schema_version": "sceptre_adaptive_route_evidence_rejection_v1",
            "frozen_model_sha256": frozen.model_sha256,
            "outer_target": frozen.outer_target,
            "reason": reason,
        }
    )
    return AdaptiveUtilityExactBFallback(
        outer_target=frozen.outer_target,
        candidate_sources=frozen.candidate_sources,
        winner_sources=(),
        reason=reason,
        frozen_model_sha256=frozen.model_sha256,
        candidate_menu_payload_sha256=frozen.candidate_menu_payload_sha256,
        exact_b_control_payload_sha256=frozen.exact_b_control_payload_sha256,
        evidence_sha256=evidence_sha256,
    )



__all__ = (
    "AdaptiveUtilityDecision",
    "AdaptiveUtilityExactBFallback",
    "AdaptiveUtilityRanking",
    "AdaptiveUtilityRoute",
    "EXACT_UTILITY_TIE_REASON",
    "FROZEN_MODEL_PUBLICATION_STATUS",
    "FROZEN_MODEL_ROLE",
    "FROZEN_MODEL_SCHEMA",
    "FrozenAdaptiveUtilityModel",
    "FrozenModelReplayReceipt",
    "INVALID_UTILITY_EVIDENCE_REASON",
    "MISSING_UTILITY_EVIDENCE_REASON",
    "PREDICTED_UTILITY_POLICY_ID",
    "PREDICTED_UTILITY_SEMANTICS",
    "freeze_adaptive_utility_model",
    "replay_frozen_adaptive_utility_model",
    "route_frozen_predicted_utility_or_exact_b",
)

