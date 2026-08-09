"""Exact hash-chain validation for the one-off consumed-test amendment."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Protocol

from ...protocol import ProtocolError
from ...runtime.artifact_io import sha256_file
from .experiment_contracts import (
    EXPERIMENT_ID,
    EXPECTED_LEDGER_AMENDMENT_SHA256,
    EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256,
    LEDGER_AMENDMENT_ARTIFACT_ID,
)


class LedgerInputConfig(Protocol):
    experiment_id: str
    test_consumption_ledger_path: Path
    ledger_amendment_path: Path


@dataclass(frozen=True)
class ValidatedLedgerChain:
    parent: Mapping[str, object]
    amendment: Mapping[str, object]


def load_validated_ledger_chain(config: LedgerInputConfig) -> ValidatedLedgerChain:
    parent = _json(config.test_consumption_ledger_path)
    parent_sha = sha256_file(config.test_consumption_ledger_path)
    descriptive_reuse = _descriptive_reuse_permission(parent)
    if (
        config.experiment_id != EXPERIMENT_ID
        or parent_sha != EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
        or parent.get("schema_version") != "midogpp_uniform_b_test_consumption_ledger_v1"
        or parent.get("status") != "CONSUMED_FOR_REPRESENTATION_ADOPTION"
        or parent.get("split") != "test"
        or parent.get("may_be_reused_as_fresh_representation_selection_evidence") is not False
        or descriptive_reuse is not True
    ):
        raise ProtocolError("Label-aware parent test-consumption ledger drifted.")
    amendment = _json(config.ledger_amendment_path)
    if (
        sha256_file(config.ledger_amendment_path) != EXPECTED_LEDGER_AMENDMENT_SHA256
        or amendment.get("schema_version") != "midogpp_test_consumption_ledger_amendment_v1"
        or amendment.get("amendment_id") != LEDGER_AMENDMENT_ARTIFACT_ID
        or amendment.get("parent_sha256") != parent_sha
        or amendment.get("authorized_consumer_experiment_ids") != [config.experiment_id]
        or amendment.get("authorization_scope") != "one_additional_terminal_label_aware_case_oof_ceiling"
        or amendment.get("all_target_probabilities_globally_sealed_before_any_label_access") is not True
        or amendment.get("support_labels_used") is not True
        or amendment.get("evaluation_labels_opened_only_after_all_fold_decisions_sealed") is not True
        or amendment.get("target_expert_used") is not False
        or amendment.get("shared_model_updated_with_target_labels") is not False
        or amendment.get("may_feed_another_stage90") is not False
        or amendment.get("generic_consumer_authorized") is not False
    ):
        raise ProtocolError("Label-aware ledger amendment chain or whitelist drifted.")
    return ValidatedLedgerChain(
        parent=MappingProxyType(dict(parent)),
        amendment=MappingProxyType(dict(amendment)),
    )


def _descriptive_reuse_permission(parent: Mapping[str, object]) -> object:
    """Read the published ledger key while rejecting contradictory aliases."""

    canonical_key = "may_be_reused_for_descriptive_locked_model_scoring"
    published_key = "may_be_reused_for_descriptive_locked-model_scoring"
    canonical_present = canonical_key in parent
    published_present = published_key in parent
    if (
        canonical_present
        and published_present
        and parent[canonical_key] != parent[published_key]
    ):
        raise ProtocolError(
            "Label-aware parent ledger has conflicting descriptive-use aliases."
        )
    if published_present:
        return parent[published_key]
    if canonical_present:
        return parent[canonical_key]
    return None


def _json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read label-aware ledger JSON: {path}.") from exc
    if not isinstance(value, dict):
        raise ProtocolError("Label-aware ledger must be a JSON object.")
    return value


__all__ = ("ValidatedLedgerChain", "load_validated_ledger_chain")
