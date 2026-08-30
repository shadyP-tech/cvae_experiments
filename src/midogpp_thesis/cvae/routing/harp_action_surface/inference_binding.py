"""Typed cross-stage binding for executable HARP action inference.

The binding deliberately separates scientific/semantic identities from hashes
of authoritative files.  A semantic identity may itself be a digest, but it is
never accepted as a substitute for a ``*_sha256`` file receipt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import ClassVar, Mapping, TYPE_CHECKING

from ...protocol import ProtocolError
from ..harp_protocol.hashing import canonical_hash, require_sha256

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for type checkers
    from .lineage import HarpAuthoritativeLineage


_SEMANTIC_ID = re.compile(r"(?:[0-9a-f]{16}|[0-9a-f]{64})")


def _require_semantic_id(value: object, *, name: str) -> str:
    if type(value) is not str or _SEMANTIC_ID.fullmatch(value) is None:
        raise ProtocolError(
            f"HARP inference semantic identifier is malformed: {name}."
        )
    return value


@dataclass(frozen=True, kw_only=True)
class HarpActionInferenceBinding:
    """One versioned Stage-60 -> policy -> Stage-70 inference contract."""

    SCHEMA_VERSION: ClassVar[str] = "midogpp_harp_action_inference_binding_v2"

    expert_bank_semantic_id: str
    generation_semantic_id: str
    source_stream_lock_semantic_id: str
    source_stream_index_semantic_id: str
    source_stream_content_semantic_id: str
    classifier_config_semantic_id: str
    source_stream_artifact_binding_semantic_id: str
    classifier_contract_semantic_id: str
    global_prediction_seal_semantic_id: str
    feature_surface_semantic_id: str
    response_surface_semantic_id: str

    expert_bank_index_file_sha256: str
    generation_lock_file_sha256: str
    source_cache_lock_file_sha256: str
    source_cache_index_file_sha256: str

    binding_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        semantic_names = (
            "expert_bank_semantic_id",
            "generation_semantic_id",
            "source_stream_lock_semantic_id",
            "source_stream_index_semantic_id",
            "source_stream_content_semantic_id",
            "classifier_config_semantic_id",
            "source_stream_artifact_binding_semantic_id",
            "classifier_contract_semantic_id",
            "global_prediction_seal_semantic_id",
            "feature_surface_semantic_id",
            "response_surface_semantic_id",
        )
        file_names = (
            "expert_bank_index_file_sha256",
            "generation_lock_file_sha256",
            "source_cache_lock_file_sha256",
            "source_cache_index_file_sha256",
        )
        for name in semantic_names:
            object.__setattr__(
                self,
                name,
                _require_semantic_id(getattr(self, name), name=name),
            )
        for name in file_names:
            object.__setattr__(
                self,
                name,
                require_sha256(getattr(self, name), name=f"HARP inference {name}"),
            )
        expected_source_binding = canonical_hash(
            {
                "schema_version": "midogpp_harp_source_stream_artifact_binding_v1",
                "source_cache_lock_sha256": self.source_cache_lock_file_sha256,
                "source_cache_index_sha256": self.source_cache_index_file_sha256,
                "source_stream_content_hash": (
                    self.source_stream_content_semantic_id
                ),
            }
        )
        if self.source_stream_artifact_binding_semantic_id != expected_source_binding:
            raise ProtocolError(
                "HARP inference source-stream artifact binding drifted."
            )
        object.__setattr__(self, "binding_sha256", canonical_hash(self._unhashed_payload()))

    @classmethod
    def from_stage60_lineage(
        cls,
        lineage: HarpAuthoritativeLineage,
        *,
        global_prediction_seal_semantic_id: str,
        feature_surface_semantic_id: str,
        response_surface_semantic_id: str,
    ) -> HarpActionInferenceBinding:
        """Construct the shared binding from the authoritative Stage-60 receipt."""

        return cls(
            expert_bank_semantic_id=lineage.bank_semantic_lock_hash,
            generation_semantic_id=lineage.generation_semantic_lock_hash,
            source_stream_lock_semantic_id=lineage.source_stream_lock_hash,
            source_stream_index_semantic_id=lineage.source_stream_index_hash,
            source_stream_content_semantic_id=lineage.source_stream_content_hash,
            classifier_config_semantic_id=lineage.classifier_config_hash,
            source_stream_artifact_binding_semantic_id=(
                lineage.source_stream_artifact_binding_hash
            ),
            classifier_contract_semantic_id=lineage.classifier_contract_sha256,
            global_prediction_seal_semantic_id=(
                global_prediction_seal_semantic_id
            ),
            feature_surface_semantic_id=feature_surface_semantic_id,
            response_surface_semantic_id=response_surface_semantic_id,
            expert_bank_index_file_sha256=lineage.expert_bank_index_sha256,
            generation_lock_file_sha256=lineage.generation_lock_file_sha256,
            source_cache_lock_file_sha256=lineage.source_cache_lock_sha256,
            source_cache_index_file_sha256=lineage.source_cache_index_sha256,
        )

    @classmethod
    def from_payload(cls, value: object) -> HarpActionInferenceBinding:
        """Validate exact keys and reconstruct one binding without aliases."""

        if not isinstance(value, Mapping) or set(value) != {
            "schema_version",
            "semantic_identifiers",
            "file_sha256",
            "binding_sha256",
        }:
            raise ProtocolError("HARP action inference-binding schema drifted.")
        if value.get("schema_version") != cls.SCHEMA_VERSION:
            raise ProtocolError("HARP action inference-binding version drifted.")
        semantics = value.get("semantic_identifiers")
        file_hashes = value.get("file_sha256")
        semantic_keys = {
            "expert_bank",
            "generation",
            "source_stream_lock",
            "source_stream_index",
            "source_stream_content",
            "classifier_config",
            "source_stream_artifact_binding",
            "classifier_contract",
            "global_prediction_seal",
            "feature_surface",
            "response_surface",
        }
        file_keys = {
            "expert_bank_index",
            "generation_lock",
            "source_cache_lock",
            "source_cache_index",
        }
        if (
            not isinstance(semantics, Mapping)
            or set(semantics) != semantic_keys
            or not isinstance(file_hashes, Mapping)
            or set(file_hashes) != file_keys
        ):
            raise ProtocolError("HARP action inference-binding members drifted.")
        binding = cls(
            expert_bank_semantic_id=str(semantics["expert_bank"]),
            generation_semantic_id=str(semantics["generation"]),
            source_stream_lock_semantic_id=str(semantics["source_stream_lock"]),
            source_stream_index_semantic_id=str(semantics["source_stream_index"]),
            source_stream_content_semantic_id=str(
                semantics["source_stream_content"]
            ),
            classifier_config_semantic_id=str(semantics["classifier_config"]),
            source_stream_artifact_binding_semantic_id=str(
                semantics["source_stream_artifact_binding"]
            ),
            classifier_contract_semantic_id=str(
                semantics["classifier_contract"]
            ),
            global_prediction_seal_semantic_id=str(
                semantics["global_prediction_seal"]
            ),
            feature_surface_semantic_id=str(semantics["feature_surface"]),
            response_surface_semantic_id=str(semantics["response_surface"]),
            expert_bank_index_file_sha256=str(file_hashes["expert_bank_index"]),
            generation_lock_file_sha256=str(file_hashes["generation_lock"]),
            source_cache_lock_file_sha256=str(file_hashes["source_cache_lock"]),
            source_cache_index_file_sha256=str(file_hashes["source_cache_index"]),
        )
        observed = require_sha256(
            value.get("binding_sha256"), name="HARP inference binding_sha256"
        )
        if observed != binding.binding_sha256:
            raise ProtocolError("HARP action inference-binding hash drifted.")
        return binding

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "semantic_identifiers": {
                "expert_bank": self.expert_bank_semantic_id,
                "generation": self.generation_semantic_id,
                "source_stream_lock": self.source_stream_lock_semantic_id,
                "source_stream_index": self.source_stream_index_semantic_id,
                "source_stream_content": self.source_stream_content_semantic_id,
                "classifier_config": self.classifier_config_semantic_id,
                "source_stream_artifact_binding": (
                    self.source_stream_artifact_binding_semantic_id
                ),
                "classifier_contract": self.classifier_contract_semantic_id,
                "global_prediction_seal": (
                    self.global_prediction_seal_semantic_id
                ),
                "feature_surface": self.feature_surface_semantic_id,
                "response_surface": self.response_surface_semantic_id,
            },
            "file_sha256": {
                "expert_bank_index": self.expert_bank_index_file_sha256,
                "generation_lock": self.generation_lock_file_sha256,
                "source_cache_lock": self.source_cache_lock_file_sha256,
                "source_cache_index": self.source_cache_index_file_sha256,
            },
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "binding_sha256": self.binding_sha256}


__all__ = ("HarpActionInferenceBinding",)
