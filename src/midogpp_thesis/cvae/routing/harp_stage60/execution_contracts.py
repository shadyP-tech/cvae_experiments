"""Neutral execution DTOs shared by HARP Stage-60 orchestration and adapters."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Protocol

from ...protocol import ProtocolError
from ..harp_protocol.hashing import canonical_hash, require_sha256
from .config import HarpInputReadiness, HarpStage60Config


@dataclass(frozen=True)
class HarpDurablePrelabelSeal:
    """Adapter-neutral proof that every label-free action is durable."""

    surface: str
    seal_path: Path
    seal_hash: str
    probability_menu_hash: str
    row_identity_hash: str
    target_support_labels_used: bool = False
    target_evaluation_labels_used: bool = False
    source_development_labels_opened: bool = False

    def verify_durable(self) -> None:
        if (
            bool(self.target_support_labels_used)
            or bool(self.target_evaluation_labels_used)
            or bool(self.source_development_labels_opened)
        ):
            raise ProtocolError("HARP prelabel seal crossed a label boundary.")
        require_sha256(self.seal_hash, name="HARP prelabel seal hash")
        require_sha256(self.probability_menu_hash, name="HARP probability menu hash")
        require_sha256(self.row_identity_hash, name="HARP row identity hash")
        try:
            payload = json.loads(Path(self.seal_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProtocolError("HARP prelabel seal is not durably readable.") from exc
        if not isinstance(payload, Mapping):
            raise ProtocolError("HARP prelabel seal payload is not a mapping.")
        unhashed = {key: value for key, value in payload.items() if key != "seal_hash"}
        if (
            payload.get("schema_version") != "midogpp_harp_durable_prelabel_seal_v1"
            or payload.get("status") != "SEALED_COMPLETE_LABEL_FREE_MENU"
            or payload.get("surface") != self.surface
            or payload.get("seal_hash") != self.seal_hash
            or canonical_hash(unhashed) != self.seal_hash
            or payload.get("probability_menu_hash") != self.probability_menu_hash
            or payload.get("row_identity_hash") != self.row_identity_hash
            or payload.get("target_support_labels_used") is not False
            or payload.get("target_evaluation_labels_used") is not False
            or payload.get("source_development_labels_opened") is not False
        ):
            raise ProtocolError("HARP durable prelabel seal drifted.")


@dataclass(frozen=True)
class HarpBuiltProduct:
    """Closed label-use receipt passed to persistence, never raw labels."""

    surface: str
    payload: Mapping[str, object]
    product_hash: str
    source_development_labels_used_for_scoring_only: bool
    target_support_labels_used: bool = False
    target_evaluation_labels_used: bool = False

    def __post_init__(self) -> None:
        normalized = dict(self.payload)
        if (
            self.product_hash != canonical_hash(normalized)
            or bool(self.target_support_labels_used)
            or bool(self.target_evaluation_labels_used)
        ):
            raise ProtocolError("HARP built product violates its label firewall.")
        object.__setattr__(self, "payload", MappingProxyType(normalized))


@dataclass(frozen=True)
class HarpRunReceipt:
    surface: str
    artifact_root: Path
    product_hash: str
    validation_hash: str
    status: str = "COMPLETE"

    def __post_init__(self) -> None:
        if self.status != "COMPLETE":
            raise ProtocolError("HARP run receipt is not complete.")
        require_sha256(self.product_hash, name="HARP product hash")
        require_sha256(self.validation_hash, name="HARP validation hash")


class HarpStage60ExecutionAdapter(Protocol):
    """Workstation boundary; orchestration owns all irreversible ordering."""

    def validate_completed_bundle(
        self, config: HarpStage60Config
    ) -> HarpRunReceipt: ...

    def preflight(
        self, config: HarpStage60Config, readiness: HarpInputReadiness
    ) -> None: ...

    def materialize_and_seal_label_free_menu(
        self, config: HarpStage60Config, readiness: HarpInputReadiness
    ) -> HarpDurablePrelabelSeal: ...

    def open_source_development_labels(
        self, config: HarpStage60Config, seal: HarpDurablePrelabelSeal
    ) -> object: ...

    def build_product(
        self,
        config: HarpStage60Config,
        seal: HarpDurablePrelabelSeal,
        source_development_labels: object | None,
    ) -> HarpBuiltProduct: ...

    def persist_product(
        self,
        config: HarpStage60Config,
        seal: HarpDurablePrelabelSeal,
        product: HarpBuiltProduct,
    ) -> Path: ...


__all__ = (
    "HarpBuiltProduct",
    "HarpDurablePrelabelSeal",
    "HarpRunReceipt",
    "HarpStage60ExecutionAdapter",
)
