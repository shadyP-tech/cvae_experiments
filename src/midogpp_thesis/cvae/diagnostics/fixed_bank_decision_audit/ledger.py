"""Immutable consumed-test ledger and hash-chained amendment validation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Protocol

from ...protocol import ProtocolError
from .experiment_contracts import (
    EXPERIMENT_ID,
    EXPECTED_EVALUATION_CASE_COUNT,
    EXPECTED_LEDGER_AMENDMENT_SHA256,
    EXPECTED_SUPPORT_CASE_COUNT,
    EXPECTED_TEST_ROW_COUNT,
    EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256,
    FIXED_SUPPORT_CASE_COUNT_PER_CENTER,
    LEDGER_AMENDMENT_ARTIFACT_ID,
)


_ORIGINAL_LEDGER_ARTIFACT_ID = "midogpp_uniform_b_test_consumption_ledger_v1"


class LedgerInputConfig(Protocol):
    experiment_id: str
    test_consumption_ledger_path: Path
    ledger_amendment_path: Path


@dataclass(frozen=True)
class ValidatedLedgerChain:
    parent: Mapping[str, object]
    amendment: Mapping[str, object]


def load_validated_ledger_chain(config: LedgerInputConfig) -> ValidatedLedgerChain:
    ledger = _json(config.test_consumption_ledger_path)
    ledger_sha = sha256_file(config.test_consumption_ledger_path)
    if (
        config.experiment_id != EXPERIMENT_ID
        or ledger_sha != EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
        or ledger.get("schema_version")
        != "midogpp_uniform_b_test_consumption_ledger_v1"
        or ledger.get("status") != "CONSUMED_FOR_REPRESENTATION_ADOPTION"
        or ledger.get("split") != "test"
        or int(ledger.get("row_count", -1)) != EXPECTED_TEST_ROW_COUNT
        or int(ledger.get("observed_centers", -1)) != 9
        or ledger.get("may_be_reused_as_fresh_representation_selection_evidence")
        is not False
        or ledger.get("may_be_reused_for_descriptive_locked-model_scoring") is not True
    ):
        raise ProtocolError("Fixed-bank original test-consumption ledger drifted.")
    amendment = _json(config.ledger_amendment_path)
    if (
        sha256_file(config.ledger_amendment_path)
        != EXPECTED_LEDGER_AMENDMENT_SHA256
        or amendment != expected_amendment_payload()
        or amendment.get("parent_sha256") != ledger_sha
        or amendment.get("authorized_consumer_experiment_ids") != [config.experiment_id]
        or amendment.get("generic_consumer_authorized") is not False
    ):
        raise ProtocolError("Fixed-bank ledger amendment chain or whitelist drifted.")
    return ValidatedLedgerChain(
        parent=MappingProxyType(dict(ledger)),
        amendment=MappingProxyType(dict(amendment)),
    )


def expected_amendment_payload() -> dict[str, object]:
    return {
        "schema_version": "midogpp_test_consumption_ledger_amendment_v1",
        "amendment_id": LEDGER_AMENDMENT_ARTIFACT_ID,
        "parent_artifact_id": _ORIGINAL_LEDGER_ARTIFACT_ID,
        "parent_member": "reports/test_consumption_ledger.json",
        "parent_sha256": EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256,
        "authorized_consumer_experiment_ids": [EXPERIMENT_ID],
        "authorization_scope": "one_additional_terminal_posthoc_fixed_bank_diagnostic",
        "dataset_family": "MIDOG++",
        "split": "test",
        "split_previously_consumed": True,
        "fresh_evidence": False,
        "claim_scope": "diagnostic_only",
        "candidate_generalization": "known_fixed_bank_reuse",
        "unseen_expert_transfer_claim": False,
        "support_case_count_per_center": FIXED_SUPPORT_CASE_COUNT_PER_CENTER,
        "support_case_count_total": EXPECTED_SUPPORT_CASE_COUNT,
        "evaluation_case_count_total": EXPECTED_EVALUATION_CASE_COUNT,
        "whole_case_support_evaluation_disjoint": True,
        "support_labels_used": False,
        "labels_opened_only_after_prediction_and_feature_seals": True,
        "exact_bacc_is_primary_terminal_response": True,
        "smooth_bacc_is_isolated_descriptive_response": True,
        "target_actions_built": False,
        "action_selection_authorized": False,
        "policy_update_authorized": False,
        "routing_quality_claimed": False,
        "promotion_eligible": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_another_stage90": False,
        "generic_consumer_authorized": False,
        "prior_stage90_outputs_used": False,
        "authorization_recorded_at_utc": "2026-08-09T00:00:00Z",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ProtocolError(f"Cannot hash fixed-bank ledger input: {path}.") from exc
    return digest.hexdigest()


def _json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read fixed-bank ledger JSON: {path}.") from exc
    if not isinstance(value, dict):
        raise ProtocolError("Fixed-bank ledger input must be a JSON object.")
    return value


__all__ = (
    "LedgerInputConfig",
    "ValidatedLedgerChain",
    "expected_amendment_payload",
    "load_validated_ledger_chain",
    "sha256_file",
)
