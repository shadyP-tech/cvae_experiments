"""Typed, path-free upstream lineage for OE-PPUR preterminal sealing.

Artifact paths are used only by the explicit validation factory and are never
stored in a receipt.  The registered v1 runner does not call that factory and
cannot resolve these inputs; it remains planning-only and non-authorized.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
import json
import os
from pathlib import Path

from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.config import (
    UniformBV2PromotionConfig,
)
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS as BANK_CENTERS,
    N_EXPERTS,
    OUTPUT_ARTIFACT_ID as BANK_ARTIFACT_ID,
    TRAINING_SEEDS as BANK_TRAINING_SEEDS,
)
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.validation import (
    validate_promoted_bank,
)
from midogpp_thesis.cvae.generation.contracts import (
    EXPECTED_BANK_INDEX_SHA256,
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_CONTENT_HASH,
    EXPECTED_CONTENT_INDEX_SHA256,
    EXPECTED_CONTROL_LOCK_HASH,
    EXPECTED_CONTROL_LOCK_SHA256,
    EXPECTED_GENERATION_LOCK_HASH,
    GenerationLock,
)

from ..hashing import canonical_hash, require_sha256
from ..identity import (
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
)
from ..manifest_contract import CanonicalTerminalManifestReceipt
from ..protocol import ProtocolError
from ..source_fence import SourceFenceReceipt, validate_source_fence_receipt
from .file_evidence import hash_read_only_regular_file
from .memmap import (
    CanonicalRowAlignmentReceipt,
    validate_canonical_row_alignment_receipt,
)
from .surfaces import (
    CandidateProbabilitySurfaceReceipt,
    validate_candidate_probability_surface_receipt,
)


_CONFIG_PROTOCOL_TOKEN = object()
_BANK_TOKEN = object()
_GENERATION_TOKEN = object()
_LINEAGE_TOKEN = object()


@dataclass(frozen=True, slots=True)
class PlannedConfigProtocolReceipt:
    """Validated path-free v1 config, protocol, and source identities."""

    config_contract_hash: str
    protocol_contract_hash: str
    source_fence: SourceFenceReceipt
    runner_blueprint_hash: str
    _factory_token: InitVar[object] = None
    receipt_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _CONFIG_PROTOCOL_TOKEN:
            raise ProtocolError(
                "OE-PPUR config/protocol receipt bypassed its guarded factory."
            )
        config_hash = require_sha256(
            self.config_contract_hash,
            "config contract hash",
        )
        protocol_hash = require_sha256(
            self.protocol_contract_hash,
            "protocol contract hash",
        )
        source = validate_source_fence_receipt(self.source_fence)
        blueprint_hash = require_sha256(
            self.runner_blueprint_hash,
            "runner blueprint hash",
        )
        body = {
            "schema_version": "oe_ppur_v1_planned_config_protocol_receipt_v1",
            "config_contract_hash": config_hash,
            "protocol_contract_hash": protocol_hash,
            "source_fence_receipt_hash": source.receipt_hash,
            "combined_source_seal_hash": source.combined_source_seal_hash,
            "runner_blueprint_hash": blueprint_hash,
            "execution_authorized": False,
        }
        object.__setattr__(self, "config_contract_hash", config_hash)
        object.__setattr__(self, "protocol_contract_hash", protocol_hash)
        object.__setattr__(self, "source_fence", source)
        object.__setattr__(self, "runner_blueprint_hash", blueprint_hash)
        object.__setattr__(self, "receipt_hash", canonical_hash(body))


@dataclass(frozen=True, slots=True)
class PromotedBankValidationReceipt:
    """Path-free result of validating the exact promoted 27-expert bank."""

    artifact_id: str
    bank_lock_hash: str
    control_lock_hash: str
    bank_index_sha256: str
    control_lock_sha256: str
    content_index_sha256: str
    content_hash: str
    validation_checks_sha256: str
    centers: tuple[str, ...]
    training_seeds: tuple[int, ...]
    expert_count: int
    artifact_bytes_validated: bool
    _factory_token: InitVar[object] = None
    receipt_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _BANK_TOKEN:
            raise ProtocolError(
                "OE-PPUR promoted-bank receipt bypassed its guarded factory."
            )
        bank_lock = _require_short_hash(self.bank_lock_hash, "bank lock")
        control_lock = _require_short_hash(
            self.control_lock_hash,
            "control lock",
        )
        content_hash = _require_short_hash(self.content_hash, "bank content")
        hashes = {
            "bank_index_sha256": require_sha256(
                self.bank_index_sha256,
                "bank index",
            ),
            "control_lock_sha256": require_sha256(
                self.control_lock_sha256,
                "control lock file",
            ),
            "content_index_sha256": require_sha256(
                self.content_index_sha256,
                "content index",
            ),
            "validation_checks_sha256": require_sha256(
                self.validation_checks_sha256,
                "bank validation checks",
            ),
        }
        centers = tuple(str(value) for value in self.centers)
        seeds = tuple(int(value) for value in self.training_seeds)
        if (
            self.artifact_id != BANK_ARTIFACT_ID
            or self.artifact_id != EXPERT_BANK_ARTIFACT_ID
            or bank_lock != EXPECTED_BANK_LOCK_HASH
            or control_lock != EXPECTED_CONTROL_LOCK_HASH
            or hashes["bank_index_sha256"] != EXPECTED_BANK_INDEX_SHA256
            or hashes["control_lock_sha256"] != EXPECTED_CONTROL_LOCK_SHA256
            or hashes["content_index_sha256"]
            != EXPECTED_CONTENT_INDEX_SHA256
            or content_hash != EXPECTED_CONTENT_HASH
            or centers != BANK_CENTERS
            or seeds != BANK_TRAINING_SEEDS
            or int(self.expert_count) != N_EXPERTS
            or type(self.artifact_bytes_validated) is not bool
        ):
            raise ProtocolError("OE-PPUR promoted-bank lineage drifted.")
        body = {
            "schema_version": "oe_ppur_v1_promoted_bank_validation_receipt_v1",
            "artifact_id": self.artifact_id,
            "bank_lock_hash": bank_lock,
            "control_lock_hash": control_lock,
            **hashes,
            "content_hash": content_hash,
            "centers": centers,
            "training_seeds": seeds,
            "expert_count": self.expert_count,
            "artifact_bytes_validated": self.artifact_bytes_validated,
            "routing_authorized": True,
        }
        object.__setattr__(self, "bank_lock_hash", bank_lock)
        object.__setattr__(self, "control_lock_hash", control_lock)
        object.__setattr__(self, "content_hash", content_hash)
        for name, value in hashes.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "training_seeds", seeds)
        object.__setattr__(self, "receipt_hash", canonical_hash(body))


@dataclass(frozen=True, slots=True)
class FrozenGenerationLockReceipt:
    """Exact semantic and full-content identities of one GenerationLock."""

    artifact_id: str
    generation_lock_hash: str
    generation_lock_payload_sha256: str
    bank_lock_hash: str
    bank_index_sha256: str
    control_lock_sha256: str
    content_index_sha256: str
    content_hash: str
    expert_count: int
    payload_validated: bool
    _factory_token: InitVar[object] = None
    receipt_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _GENERATION_TOKEN:
            raise ProtocolError(
                "OE-PPUR generation-lock receipt bypassed its guarded factory."
            )
        generation = _require_short_hash(
            self.generation_lock_hash,
            "generation lock",
        )
        bank = _require_short_hash(self.bank_lock_hash, "generation bank lock")
        content = _require_short_hash(
            self.content_hash,
            "generation content",
        )
        hashes = {
            "generation_lock_payload_sha256": require_sha256(
                self.generation_lock_payload_sha256,
                "generation-lock payload",
            ),
            "bank_index_sha256": require_sha256(
                self.bank_index_sha256,
                "generation bank index",
            ),
            "control_lock_sha256": require_sha256(
                self.control_lock_sha256,
                "generation control lock",
            ),
            "content_index_sha256": require_sha256(
                self.content_index_sha256,
                "generation content index",
            ),
        }
        if (
            self.artifact_id != GENERATION_LOCK_ARTIFACT_ID
            or generation != EXPECTED_GENERATION_LOCK_HASH
            or bank != EXPECTED_BANK_LOCK_HASH
            or hashes["bank_index_sha256"] != EXPECTED_BANK_INDEX_SHA256
            or hashes["control_lock_sha256"] != EXPECTED_CONTROL_LOCK_SHA256
            or hashes["content_index_sha256"]
            != EXPECTED_CONTENT_INDEX_SHA256
            or content != EXPECTED_CONTENT_HASH
            or int(self.expert_count) != N_EXPERTS
            or type(self.payload_validated) is not bool
        ):
            raise ProtocolError("OE-PPUR frozen GenerationLock lineage drifted.")
        body = {
            "schema_version": "oe_ppur_v1_frozen_generation_lock_receipt_v1",
            "artifact_id": self.artifact_id,
            "generation_lock_hash": generation,
            "bank_lock_hash": bank,
            **hashes,
            "content_hash": content,
            "expert_count": self.expert_count,
            "payload_validated": self.payload_validated,
        }
        object.__setattr__(self, "generation_lock_hash", generation)
        object.__setattr__(self, "bank_lock_hash", bank)
        object.__setattr__(self, "content_hash", content)
        for name, value in hashes.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "receipt_hash", canonical_hash(body))


@dataclass(frozen=True, slots=True)
class PreterminalInputLineage:
    """All typed, label-free parents of a future preterminal phase."""

    config_protocol: PlannedConfigProtocolReceipt
    expert_bank: PromotedBankValidationReceipt
    generation_lock: FrozenGenerationLockReceipt
    candidate_surface: CandidateProbabilitySurfaceReceipt
    manifest: CanonicalTerminalManifestReceipt
    rows: CanonicalRowAlignmentReceipt
    test_fixture_only: bool
    execution_authorized: bool = False
    _factory_token: InitVar[object] = None
    lineage_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _LINEAGE_TOKEN:
            raise ProtocolError(
                "OE-PPUR preterminal input lineage bypassed its guarded factory."
            )
        if not isinstance(self.config_protocol, PlannedConfigProtocolReceipt):
            raise ProtocolError("OE-PPUR config/protocol lineage is untyped.")
        if not isinstance(self.expert_bank, PromotedBankValidationReceipt):
            raise ProtocolError("OE-PPUR expert-bank lineage is untyped.")
        if not isinstance(self.generation_lock, FrozenGenerationLockReceipt):
            raise ProtocolError("OE-PPUR generation lineage is untyped.")
        rows = validate_canonical_row_alignment_receipt(self.rows)
        candidate = validate_candidate_probability_surface_receipt(
            self.candidate_surface,
            row_alignment_receipt=rows,
        )
        source = validate_source_fence_receipt(
            self.config_protocol.source_fence
        )
        expected_fixture = not (
            self.expert_bank.artifact_bytes_validated
            and self.generation_lock.payload_validated
        )
        if (
            not isinstance(self.manifest, CanonicalTerminalManifestReceipt)
            or rows.manifest_receipt != self.manifest
            or candidate.row_index_sha256 != rows.row_index_sha256
            or self.config_protocol.source_fence.receipt_hash
            != source.receipt_hash
            or self.generation_lock.bank_lock_hash
            != self.expert_bank.bank_lock_hash
            or self.generation_lock.bank_index_sha256
            != self.expert_bank.bank_index_sha256
            or self.generation_lock.control_lock_sha256
            != self.expert_bank.control_lock_sha256
            or self.generation_lock.content_index_sha256
            != self.expert_bank.content_index_sha256
            or self.generation_lock.content_hash != self.expert_bank.content_hash
            or bool(self.test_fixture_only) != expected_fixture
            or (expected_fixture and "PYTEST_CURRENT_TEST" not in os.environ)
            or bool(self.execution_authorized)
        ):
            raise ProtocolError("OE-PPUR typed preterminal input lineage drifted.")
        body = {
            "schema_version": "oe_ppur_v1_preterminal_input_lineage_v1",
            "config_protocol_receipt_hash": self.config_protocol.receipt_hash,
            "source_fence_receipt_hash": source.receipt_hash,
            "expert_bank_receipt_hash": self.expert_bank.receipt_hash,
            "generation_lock_receipt_hash": self.generation_lock.receipt_hash,
            "candidate_surface_receipt_hash": candidate.receipt_hash,
            "manifest_receipt_hash": self.manifest.receipt_hash,
            "row_alignment_receipt_hash": rows.receipt_hash,
            "test_fixture_only": self.test_fixture_only,
            "execution_authorized": False,
            "labels_present": False,
        }
        object.__setattr__(self, "candidate_surface", candidate)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "lineage_hash", canonical_hash(body))


def build_planned_config_protocol_receipt(
    config: object,
) -> PlannedConfigProtocolReceipt:
    """Validate the current path-free planned contract and derive a receipt."""

    from ..execution_admission import validate_planned_execution_contract
    from ..runner_blueprint import build_runner_blueprint

    source = validate_planned_execution_contract(config)
    blueprint = build_runner_blueprint(config, source)
    protocol = getattr(config, "protocol", None)
    if not isinstance(protocol, Mapping):
        raise ProtocolError("OE-PPUR planned protocol is absent.")
    return PlannedConfigProtocolReceipt(
        config_contract_hash=str(getattr(config, "contract_hash", "")),
        protocol_contract_hash=str(protocol.get("protocol_hash", "")),
        source_fence=source,
        runner_blueprint_hash=blueprint.blueprint_hash,
        _factory_token=_CONFIG_PROTOCOL_TOKEN,
    )


def validate_promoted_bank_input(
    root: str | Path,
    *,
    config: UniformBV2PromotionConfig,
) -> PromotedBankValidationReceipt:
    """Validate the exact Stage-30 artifact and discard its filesystem path."""

    input_root = Path(root)
    try:
        resolved = input_root.resolve(strict=True)
    except OSError as exc:
        raise ProtocolError("OE-PPUR promoted-bank root is absent.") from exc
    if input_root.is_symlink() or not resolved.is_dir():
        raise ProtocolError("OE-PPUR promoted-bank root is unsafe.")
    checks = validate_promoted_bank(resolved, config=config, allow_pending=False)
    bank_path = resolved / "manifests/expert_bank_index.json"
    control_path = resolved / "manifests/equal_union_ps_control_lock.json"
    content_path = resolved / "manifests/content_index.json"
    bank = _read_json_object(bank_path)
    control = _read_json_object(control_path)
    content = _read_json_object(content_path)
    return PromotedBankValidationReceipt(
        artifact_id=BANK_ARTIFACT_ID,
        bank_lock_hash=str(bank.get("bank_lock_hash", "")),
        control_lock_hash=str(control.get("control_lock_hash", "")),
        bank_index_sha256=hash_read_only_regular_file(
            bank_path.as_posix(),
            role="promoted bank index",
        ),
        control_lock_sha256=hash_read_only_regular_file(
            control_path.as_posix(),
            role="promoted control lock",
        ),
        content_index_sha256=hash_read_only_regular_file(
            content_path.as_posix(),
            role="promoted content index",
        ),
        content_hash=str(content.get("content_hash", "")),
        validation_checks_sha256=canonical_hash(checks),
        centers=BANK_CENTERS,
        training_seeds=BANK_TRAINING_SEEDS,
        expert_count=N_EXPERTS,
        artifact_bytes_validated=True,
        _factory_token=_BANK_TOKEN,
    )


def build_frozen_generation_lock_receipt(
    generation_lock: GenerationLock,
) -> FrozenGenerationLockReceipt:
    """Validate one typed GenerationLock and preserve short/full hash roles."""

    if not isinstance(generation_lock, GenerationLock):
        raise ProtocolError("OE-PPUR generation input is not a GenerationLock.")
    payload = generation_lock.to_payload()
    bank = payload.get("bank")
    if not isinstance(bank, Mapping):
        raise ProtocolError("OE-PPUR GenerationLock bank payload is absent.")
    experts = bank.get("expert_locks")
    if (
        payload.get("schema_version")
        != "midogpp_uniform_b_v2_generation_lock_v1"
        or bank.get("artifact_id") != BANK_ARTIFACT_ID
        or tuple(str(value) for value in bank.get("centers", ()))
        != BANK_CENTERS
        or not isinstance(experts, list)
        or len(experts) != N_EXPERTS
    ):
        raise ProtocolError("OE-PPUR GenerationLock inventory drifted.")
    return FrozenGenerationLockReceipt(
        artifact_id=GENERATION_LOCK_ARTIFACT_ID,
        generation_lock_hash=generation_lock.generation_lock_hash,
        generation_lock_payload_sha256=canonical_hash(payload),
        bank_lock_hash=generation_lock.bank_lock_hash,
        bank_index_sha256=str(bank.get("bank_index_sha256", "")),
        control_lock_sha256=str(bank.get("control_lock_sha256", "")),
        content_index_sha256=str(bank.get("content_index_sha256", "")),
        content_hash=str(bank.get("content_hash", "")),
        expert_count=len(experts),
        payload_validated=True,
        _factory_token=_GENERATION_TOKEN,
    )


def build_preterminal_input_lineage(
    *,
    config_protocol: PlannedConfigProtocolReceipt,
    expert_bank: PromotedBankValidationReceipt,
    generation_lock: FrozenGenerationLockReceipt,
    candidate_surface: CandidateProbabilitySurfaceReceipt,
    manifest: CanonicalTerminalManifestReceipt,
    rows: CanonicalRowAlignmentReceipt,
) -> PreterminalInputLineage:
    """Cross-check every typed parent and derive one path-free lineage."""

    test_fixture = not (
        expert_bank.artifact_bytes_validated
        and generation_lock.payload_validated
    )
    return PreterminalInputLineage(
        config_protocol=config_protocol,
        expert_bank=expert_bank,
        generation_lock=generation_lock,
        candidate_surface=candidate_surface,
        manifest=manifest,
        rows=rows,
        test_fixture_only=test_fixture,
        _factory_token=_LINEAGE_TOKEN,
    )


def _build_strict_test_upstream_receipts(
) -> tuple[PromotedBankValidationReceipt, FrozenGenerationLockReceipt]:
    """Canonical semantic fixtures with no claim of artifact-byte validation."""

    if "PYTEST_CURRENT_TEST" not in os.environ:
        raise ProtocolError("OE-PPUR synthetic upstream lineage is pytest-only.")
    bank = PromotedBankValidationReceipt(
        artifact_id=BANK_ARTIFACT_ID,
        bank_lock_hash=EXPECTED_BANK_LOCK_HASH,
        control_lock_hash=EXPECTED_CONTROL_LOCK_HASH,
        bank_index_sha256=EXPECTED_BANK_INDEX_SHA256,
        control_lock_sha256=EXPECTED_CONTROL_LOCK_SHA256,
        content_index_sha256=EXPECTED_CONTENT_INDEX_SHA256,
        content_hash=EXPECTED_CONTENT_HASH,
        validation_checks_sha256=canonical_hash(
            {"schema_version": "oe_ppur_v1_strict_test_bank_checks_v1"}
        ),
        centers=BANK_CENTERS,
        training_seeds=BANK_TRAINING_SEEDS,
        expert_count=N_EXPERTS,
        artifact_bytes_validated=False,
        _factory_token=_BANK_TOKEN,
    )
    generation = FrozenGenerationLockReceipt(
        artifact_id=GENERATION_LOCK_ARTIFACT_ID,
        generation_lock_hash=EXPECTED_GENERATION_LOCK_HASH,
        generation_lock_payload_sha256=canonical_hash(
            {"schema_version": "oe_ppur_v1_strict_test_generation_lock_v1"}
        ),
        bank_lock_hash=EXPECTED_BANK_LOCK_HASH,
        bank_index_sha256=EXPECTED_BANK_INDEX_SHA256,
        control_lock_sha256=EXPECTED_CONTROL_LOCK_SHA256,
        content_index_sha256=EXPECTED_CONTENT_INDEX_SHA256,
        content_hash=EXPECTED_CONTENT_HASH,
        expert_count=N_EXPERTS,
        payload_validated=False,
        _factory_token=_GENERATION_TOKEN,
    )
    return bank, generation


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("OE-PPUR upstream JSON could not be read.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("OE-PPUR upstream JSON is not an object.")
    return payload


def _require_short_hash(value: object, role: str) -> str:
    text = str(value)
    if len(text) != 16 or any(char not in "0123456789abcdef" for char in text):
        raise ProtocolError(f"OE-PPUR {role} is not a short semantic hash.")
    return text


__all__ = (
    "FrozenGenerationLockReceipt",
    "PlannedConfigProtocolReceipt",
    "PreterminalInputLineage",
    "PromotedBankValidationReceipt",
    "build_frozen_generation_lock_receipt",
    "build_planned_config_protocol_receipt",
    "build_preterminal_input_lineage",
    "validate_promoted_bank_input",
)
