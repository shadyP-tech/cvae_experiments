"""Production validation traversal for Stage-70 authorization inputs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping

from ...expert_bank.uniform_b_v2_promotion import (
    load_promotion_config,
    validate_promoted_bank,
)
from ...expert_bank.uniform_b_v2_promotion.contracts import (
    OUTPUT_ARTIFACT_ID as BANK_ARTIFACT_ID,
)
from ...generation import (
    load_generation_lock_config,
    read_generation_lock,
    validate_generation_bundle,
)
from ...generation.contracts import OUTPUT_ARTIFACT_ID as GENERATION_ARTIFACT_ID
from ...protocol import ProtocolError
from ...routing import (
    load_equal_union_policy_config,
    read_policy_lock as read_equal_union_policy_lock,
    validate_equal_union_policy_bundle,
)
from ...routing.contracts import OUTPUT_ARTIFACT_ID as EQUAL_POLICY_ARTIFACT_ID
from ...routing.metadata_tie_union import (
    load_metadata_tie_union_policy_config,
    read_policy_lock as read_metadata_policy_lock,
    validate_metadata_tie_union_policy_bundle,
)
from ...routing.metadata_tie_union.contracts import (
    OUTPUT_ARTIFACT_ID as METADATA_POLICY_ARTIFACT_ID,
)
from ...routing.utility_regret_policy import (
    load_utility_regret_policy_config,
    read_policy_lock as read_utility_policy_lock,
    validate_utility_regret_policy_bundle,
)
from ...routing.utility_regret_policy.contracts import (
    OUTPUT_ARTIFACT_ID as UTILITY_POLICY_ARTIFACT_ID,
)
from ....real_features.classifier_reference.uniform_b_reference.config import (
    load_uniform_b_canonical_reference_config,
)
from ....real_features.classifier_reference.uniform_b_reference.validation import (
    validate_uniform_b_canonical_reference_bundle,
)
from ..contracts import CONTROL_ARM, METADATA_ARM, UTILITY_ARM
from ..policy_adapters import load_frozen_policy_replicates
from .config import FinalAuthorizationConfig, ReservationConfig
from .contracts import (
    ArtifactBinding,
    AuthorizationValidationInputs,
    CacheBinding,
    PolicyBinding,
)


CANONICAL_REFERENCE_ARTIFACT_ID = "midogpp_output_uniform_b_canonical_reference_v1"


def load_validated_authorization_inputs(
    config: ReservationConfig | FinalAuthorizationConfig,
) -> AuthorizationValidationInputs:
    """Rerun every public upstream validator and cross-bind exact locks."""

    reference_config = load_uniform_b_canonical_reference_config(
        config.canonical_reference_root / "config.resolved.yaml"
    )
    validate_uniform_b_canonical_reference_bundle(
        config.canonical_reference_root,
        config=reference_config,
    )
    if reference_config.artifact_root.resolve() != config.canonical_reference_root.resolve():
        raise ProtocolError("Stage-70 canonical-reference root binding drifted.")
    ledger_path = (
        config.test_consumption_ledger_path
        if isinstance(config, ReservationConfig)
        else config.canonical_reference_root / "reports/test_consumption_ledger.json"
    )
    ledger = _json(ledger_path)
    reference_lock = _json(
        config.canonical_reference_root
        / "manifests/uniform_b_canonical_representation_lock.json"
    )
    reference_content = _json(
        config.canonical_reference_root / "manifests/content_index.json"
    )

    bank_config = load_promotion_config(config.bank_root / "config.resolved.yaml")
    validate_promoted_bank(config.bank_root, config=bank_config)
    if bank_config.artifact_root.resolve() != config.bank_root.resolve():
        raise ProtocolError("Stage-70 expert-bank root binding drifted.")
    bank_index = _json(config.bank_root / "manifests/expert_bank_index.json")
    bank_control = _json(config.bank_root / "manifests/equal_union_ps_control_lock.json")
    bank_content = _json(config.bank_root / "manifests/content_index.json")

    generation_config = load_generation_lock_config(
        config.generation_lock_root / "config.resolved.yaml"
    )
    validate_generation_bundle(
        config.generation_lock_root,
        config=generation_config,
    )
    generation_lock = read_generation_lock(
        config.generation_lock_root / "manifests/generation_lock.json"
    )
    generation_content = _json(
        config.generation_lock_root / "manifests/content_index.json"
    )
    if (
        generation_config.artifact_root.resolve()
        != config.generation_lock_root.resolve()
        or generation_config.bank_root.resolve() != config.bank_root.resolve()
        or generation_lock.bank_lock_hash != str(bank_index.get("bank_lock_hash", ""))
    ):
        raise ProtocolError("Stage-70 GenerationLock/bank binding drifted.")

    equal_config = load_equal_union_policy_config(
        config.equal_union_policy_root / "config.resolved.yaml"
    )
    validate_equal_union_policy_bundle(
        config.equal_union_policy_root,
        config=equal_config,
    )
    equal_lock = read_equal_union_policy_lock(
        config.equal_union_policy_root / "manifests/policy_lock.json"
    )
    equal_payload = equal_lock.to_payload()
    _require_policy_roots(
        equal_config,
        artifact_root=config.equal_union_policy_root,
        generation_root=config.generation_lock_root,
        bank_root=config.bank_root,
        label="equal-union",
    )

    metadata_config = load_metadata_tie_union_policy_config(
        config.metadata_policy_root / "config.resolved.yaml"
    )
    validate_metadata_tie_union_policy_bundle(
        config.metadata_policy_root,
        config=metadata_config,
    )
    metadata_lock = read_metadata_policy_lock(
        config.metadata_policy_root / "manifests/policy_lock.json"
    )
    metadata_payload = metadata_lock.to_payload()
    _require_policy_roots(
        metadata_config,
        artifact_root=config.metadata_policy_root,
        generation_root=config.generation_lock_root,
        bank_root=config.bank_root,
        label="metadata",
    )

    utility_config = load_utility_regret_policy_config(
        config.utility_policy_root / "config.resolved.yaml"
    )
    validate_utility_regret_policy_bundle(
        config.utility_policy_root,
        config=utility_config,
    )
    utility_lock = read_utility_policy_lock(
        config.utility_policy_root / "manifests/policy_lock.json"
    )
    utility_payload = utility_lock.to_payload()
    _require_policy_roots(
        utility_config,
        artifact_root=config.utility_policy_root,
        generation_root=config.generation_lock_root,
        bank_root=config.bank_root,
        label="utility/regret",
    )

    replicates = load_frozen_policy_replicates(
        generation_lock=generation_lock,
        equal_union_root=config.equal_union_policy_root,
        metadata_tie_union_root=config.metadata_policy_root,
        utility_regret_root=config.utility_policy_root,
    )
    policies = (
        _policy_binding(
            policy_id=CONTROL_ARM,
            artifact_id=EQUAL_POLICY_ARTIFACT_ID,
            root=config.equal_union_policy_root,
            payload=equal_payload,
        ),
        _policy_binding(
            policy_id=METADATA_ARM,
            artifact_id=METADATA_POLICY_ARTIFACT_ID,
            root=config.metadata_policy_root,
            payload=metadata_payload,
        ),
        _policy_binding(
            policy_id=UTILITY_ARM,
            artifact_id=UTILITY_POLICY_ARTIFACT_ID,
            root=config.utility_policy_root,
            payload=utility_payload,
        ),
    )
    generation_payload = generation_lock.to_payload()
    classifier = generation_payload.get("classifier")
    if not isinstance(classifier, Mapping):
        raise ProtocolError("Stage-70 GenerationLock lacks its classifier spec.")
    return AuthorizationValidationInputs(
        consumption_ledger=ledger,
        canonical_reference=ArtifactBinding(
            artifact_id=CANONICAL_REFERENCE_ARTIFACT_ID,
            content_index_sha256=_sha256_file(
                config.canonical_reference_root / "manifests/content_index.json"
            ),
            semantic_hashes={
                "representation_lock_hash": str(
                    reference_lock.get("representation_lock_hash", "")
                ),
                "content_hash": str(reference_content.get("content_hash", "")),
            },
            validator="validate_uniform_b_canonical_reference_bundle",
        ),
        bank=ArtifactBinding(
            artifact_id=BANK_ARTIFACT_ID,
            content_index_sha256=_sha256_file(
                config.bank_root / "manifests/content_index.json"
            ),
            semantic_hashes={
                "bank_lock_hash": str(bank_index.get("bank_lock_hash", "")),
                "control_lock_hash": str(bank_control.get("control_lock_hash", "")),
                "content_hash": str(bank_content.get("content_hash", "")),
            },
            validator="validate_promoted_bank",
        ),
        generation=ArtifactBinding(
            artifact_id=GENERATION_ARTIFACT_ID,
            content_index_sha256=_sha256_file(
                config.generation_lock_root / "manifests/content_index.json"
            ),
            semantic_hashes={
                "generation_lock_hash": generation_lock.generation_lock_hash,
                "content_hash": str(generation_content.get("content_hash", "")),
            },
            validator="validate_generation_bundle",
        ),
        policies=policies,
        generation_lock=generation_lock,
        policy_replicates=replicates,
        classifier_spec=dict(classifier),
    )


def load_validated_cache_binding(
    config: FinalAuthorizationConfig,
) -> CacheBinding:
    """Validate the label-blind Stage-70 cache through its public validator."""

    try:
        from ....data.features.stage70_test_cache import (
            validate_stage70_test_cache,
        )
    except ImportError as exc:  # pragma: no cover - only during partial installs.
        raise ProtocolError("Stage-70 test-cache validator is unavailable.") from exc
    checks = validate_stage70_test_cache(config.cache_root)
    if not isinstance(checks, Mapping):
        raise ProtocolError("Stage-70 test-cache validator returned invalid evidence.")
    return cache_binding_from_summary(config.cache_artifact_id, checks)


def cache_binding_from_summary(
    artifact_id: str,
    summary: Mapping[str, object],
) -> CacheBinding:
    rows = summary.get("rows_by_center")
    shards = summary.get("shard_sha256_by_center")
    if not isinstance(rows, Mapping) or not isinstance(shards, Mapping):
        raise ProtocolError("Stage-70 cache summary lacks center bindings.")
    if summary.get("status") != "PASS":
        raise ProtocolError("Stage-70 target cache did not validate PASS.")
    return CacheBinding(
        artifact_id=artifact_id,
        manifest_sha256=str(summary.get("manifest_sha256", "")),
        target_evaluation_reservation_id=str(
            summary.get("target_evaluation_reservation_id", "")
        ),
        target_evaluation_reservation_protocol_hash=str(
            summary.get("target_evaluation_reservation_protocol_hash", "")
        ),
        cache_extractor_protocol_hash=str(
            summary.get("cache_extractor_protocol_hash", "")
        ),
        row_count=int(summary.get("row_count", -1)),
        rows_by_center={str(key): int(value) for key, value in rows.items()},
        row_order_hash=str(summary.get("row_order_hash", "")),
        shard_sha256_by_center={str(key): str(value) for key, value in shards.items()},
        content_hash=str(summary.get("content_hash", "")),
        purpose=str(summary.get("purpose", "")),
        fresh_evidence=bool(summary.get("fresh_evidence", True)),
    )


def _policy_binding(
    *,
    policy_id: str,
    artifact_id: str,
    root: Path,
    payload: Mapping[str, object],
) -> PolicyBinding:
    if policy_id == UTILITY_ARM:
        outputs = payload.get("outputs")
        if not isinstance(outputs, Mapping):
            raise ProtocolError("Utility/regret policy lock lacks output identities.")
        plan_hash = str(outputs.get("policy_plan_hash", ""))
        assignment_hash = str(outputs.get("assignment_table_hash", ""))
    else:
        plan_hash = str(payload.get("policy_plan_hash", ""))
        assignment_hash = str(payload.get("assignment_table_hash", ""))
    assignment_path = root / "tables/policy_assignments.csv"
    return PolicyBinding(
        policy_id=policy_id,
        policy_artifact_id=artifact_id,
        policy_lock_hash=str(payload.get("policy_lock_hash", "")),
        policy_plan_hash=plan_hash,
        assignment_table_hash=assignment_hash,
        assignment_table_sha256=_sha256_file(assignment_path),
        assignment_count=_csv_row_count(assignment_path),
    )


def _require_policy_roots(
    policy_config: object,
    *,
    artifact_root: Path,
    generation_root: Path,
    bank_root: Path,
    label: str,
) -> None:
    if (
        Path(getattr(policy_config, "artifact_root")).resolve()
        != artifact_root.resolve()
        or Path(getattr(policy_config, "generation_lock_root")).resolve()
        != generation_root.resolve()
        or Path(getattr(policy_config, "bank_root")).resolve() != bank_root.resolve()
    ):
        raise ProtocolError(f"Stage-70 {label} policy root binding drifted.")


def _csv_row_count(path: Path) -> int:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return sum(1 for _row in csv.DictReader(handle))
    except OSError as exc:
        raise ProtocolError(f"Cannot read Stage-70 policy assignments: {path}.") from exc


def _json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read validated Stage-70 upstream JSON: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"Validated Stage-70 upstream JSON must be an object: {path}.")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ProtocolError(f"Cannot hash Stage-70 input file: {path}.") from exc
    return digest.hexdigest()


__all__ = (
    "CANONICAL_REFERENCE_ARTIFACT_ID",
    "cache_binding_from_summary",
    "load_validated_authorization_inputs",
    "load_validated_cache_binding",
)
