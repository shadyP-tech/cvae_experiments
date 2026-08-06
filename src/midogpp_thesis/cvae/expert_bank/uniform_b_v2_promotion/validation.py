"""Independent validation of the routing-authorized Uniform-B v2 bank."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .config import UniformBV2PromotionConfig
from .contracts import (
    CENTERS,
    N_EXPERTS,
    PROMOTION_DECISION,
    PROMOTION_REVIEW_ID,
    PUBLICATION_STATE,
    TRAINING_SEEDS,
    legal_routing_sources,
)


REQUIRED_FILES = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "manifests/promotion_protocol.json",
    "manifests/promotion_review_snapshot.json",
    "manifests/source_evidence_lock.json",
    "manifests/expert_bank_index.json",
    "manifests/equal_union_ps_control_lock.json",
    "manifests/content_index.json",
    "reports/promotion_decision.json",
    "reports/test_consumption_ledger.json",
    "reports/leakage_report.json",
    "reports/promotion_report.md",
    "reports/run_state.json",
    "tables/expert_inventory.csv",
    "tables/sampler_inventory.csv",
    "tables/source_gate_audit.csv",
)


def validate_promoted_bank(
    root: str | Path,
    *,
    config: UniformBV2PromotionConfig,
    allow_pending: bool = False,
) -> dict[str, object]:
    path = Path(root)
    required = set(REQUIRED_FILES)
    if not allow_pending:
        required.add("reports/validation_report.json")
    missing = sorted(relative for relative in required if not (path / relative).is_file())
    if missing:
        raise ProtocolError(f"Promoted Uniform-B v2 bank is incomplete: {missing}.")
    bank = _read_json(path / "manifests/expert_bank_index.json")
    control = _read_json(path / "manifests/equal_union_ps_control_lock.json")
    protocol = _read_json(path / "manifests/promotion_protocol.json")
    review = _read_json(path / "manifests/promotion_review_snapshot.json")
    source = _read_json(path / "manifests/source_evidence_lock.json")
    decision = _read_json(path / "reports/promotion_decision.json")
    ledger = _read_json(path / "reports/test_consumption_ledger.json")
    leakage = _read_json(path / "reports/leakage_report.json")
    state = _read_json(path / "reports/run_state.json")
    _assert_stable_hash(bank, "bank_lock_hash")
    _assert_stable_hash(control, "control_lock_hash")
    _assert_stable_hash(protocol, "protocol_hash")
    _assert_stable_hash(review, "review_hash")
    if (
        bank.get("n_experts") != N_EXPERTS
        or bank.get("centers") != list(CENTERS)
        or bank.get("training_seeds") != list(TRAINING_SEEDS)
        or bank.get("routing_authorized") is not True
        or bank.get("may_feed_deployable_selection") is not True
        or bank.get("replica_policy") != "retain_all_three_no_validation_based_seed_selection"
        or protocol.get("promotion_contract_hash") != config.contract_hash
        or protocol.get("promotion_review_id") != PROMOTION_REVIEW_ID
        or protocol.get("all_27_experts_retained") is not True
        or protocol.get("individual_expert_or_seed_selection") is not False
        or protocol.get("routing_quality_claimed") is not False
        or protocol.get("may_feed_deployable_selection") is not True
        or review.get("review_id") != PROMOTION_REVIEW_ID
        or review.get("status") != "approved"
        or review.get("review_effect") != "authorizes_new_stage30_expert_bank_only"
        or source.get("promotion_gates_passed") is not True
        or source.get("source_evidence_consumed_for_whole_bank_adoption") is not True
        or source.get("may_be_reused_for_individual_expert_or_seed_selection") is not False
        or decision.get("decision") != PROMOTION_DECISION
        or decision.get("publication_state") != PUBLICATION_STATE
        or decision.get("whole_bank_promoted_without_expert_or_seed_selection") is not True
        or decision.get("routing_quality_claimed") is not False
        or decision.get("may_feed_deployable_selection") is not True
        or ledger.get("status") != "CONSUMED_FOR_WHOLE_BANK_ADOPTION"
        or ledger.get("individual_expert_or_seed_selection_performed") is not False
        or ledger.get("may_be_reused_as_fresh_bank_selection_evidence") is not False
        or leakage.get("status") != "PASS"
        or leakage.get("target_expert_excluded_in_every_routing_fold") is not True
        or leakage.get("target_support_labels_used_for_routing_selection") is not False
        or leakage.get("individual_expert_or_seed_selection_performed") is not False
        or state.get("status") != "COMPLETE"
    ):
        raise ProtocolError("Promoted Uniform-B v2 authorization boundary failed.")
    records = bank.get("records")
    if not isinstance(records, list) or len(records) != N_EXPERTS:
        raise ProtocolError("Promoted Uniform-B v2 expert coverage drifted.")
    expected_keys = {(center, seed) for center in CENTERS for seed in TRAINING_SEEDS}
    observed_keys = set()
    for raw in records:
        if not isinstance(raw, Mapping):
            raise ProtocolError("Promoted expert-bank record is invalid.")
        record = dict(raw)
        _assert_stable_hash(record, "expert_lock_hash")
        key = (str(record.get("source_center")), int(record.get("training_seed", -1)))
        observed_keys.add(key)
        if (
            record.get("fresh_source_only_training") is not True
            or record.get("parent_checkpoint_used") is not False
            or record.get("individual_expert_or_seed_selected") is not False
            or record.get("routing_authorized") is not True
        ):
            raise ProtocolError("Promoted expert record violates its firewall.")
        for path_key, hash_key in (
            ("checkpoint_path", "checkpoint_file_sha256"),
            ("frame_path", "frame_file_sha256"),
            ("sampler_path", "sampler_file_sha256"),
        ):
            member = _safe_member(path, str(record.get(path_key, "")))
            if not member.is_file() or _sha256_file(member) != record.get(hash_key):
                raise ProtocolError(f"Promoted expert member drifted: {path_key}.")
    if observed_keys != expected_keys:
        raise ProtocolError("Promoted expert-bank keys drifted.")
    if (
        control.get("candidate_sources_by_target")
        != {target: list(legal_routing_sources(target)) for target in CENTERS}
        or control.get("source_budget_per_class") != 128
        or control.get("total_per_class") != 1024
        or control.get("target_expert_excluded") is not True
        or control.get("target_conditioned_source_weighting") is not False
        or control.get("canonical_control_for_future_routing") is not True
    ):
        raise ProtocolError("Promoted equal-union control lock drifted.")
    expert_rows = _read_csv(path / "tables/expert_inventory.csv")
    sampler_rows = _read_csv(path / "tables/sampler_inventory.csv")
    gate_rows = _read_csv(path / "tables/source_gate_audit.csv")
    if (
        len(expert_rows) != N_EXPERTS
        or len(sampler_rows) != 2 * N_EXPERTS
        or len(gate_rows) != 7
        or any(row.get("routing_authorized") != "True" for row in expert_rows)
        or any(
            row.get("realized_family")
            != "class_conditional_shrinkage_full_total_moment"
            or row.get("fallback_reason") != ""
            for row in sampler_rows
        )
        or any(row.get("status") != "PASS" for row in gate_rows)
    ):
        raise ProtocolError("Promoted expert-bank inventory failed validation.")
    _validate_content_index(path)
    checks = {
        "status": "PASS",
        "promotion_decision": PROMOTION_DECISION,
        "publication_state": PUBLICATION_STATE,
        "expert_count": N_EXPERTS,
        "source_center_count": len(CENTERS),
        "training_replicates_per_source": len(TRAINING_SEEDS),
        "sampler_class_records": 2 * N_EXPERTS,
        "all_experts_source_only": True,
        "all_sampler_realizations_full_shrinkage": True,
        "individual_expert_or_seed_selection": False,
        "target_expert_excluded": True,
        "canonical_equal_union_control_locked": True,
        "may_feed_deployable_selection": True,
    }
    if not allow_pending:
        report = _read_json(path / "reports/validation_report.json")
        if report.get("status") != "PASS" or report.get("checks") != checks:
            raise ProtocolError("Promoted expert-bank validation report drifted.")
    return checks


def _validate_content_index(root: Path) -> None:
    payload = _read_json(root / "manifests/content_index.json")
    _assert_stable_hash(payload, "content_hash")
    rows = payload.get("records")
    if not isinstance(rows, list):
        raise ProtocolError("Promoted expert-bank content index is invalid.")
    excluded = {
        "manifests/content_index.json",
        "reports/run_state.json",
        "reports/validation_report.json",
    }
    expected = {
        member.relative_to(root).as_posix()
        for member in root.rglob("*")
        if member.is_file() and member.relative_to(root).as_posix() not in excluded
    }
    observed = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ProtocolError("Promoted content-index row is invalid.")
        relative = str(row.get("relative_path", ""))
        member = _safe_member(root, relative)
        if (
            not member.is_file()
            or member.stat().st_size != int(row.get("size_bytes", -1))
            or _sha256_file(member) != row.get("sha256")
        ):
            raise ProtocolError(f"Promoted content member drifted: {relative}.")
        observed.add(relative)
    if observed != expected:
        raise ProtocolError("Promoted content-index coverage drifted.")


def _assert_stable_hash(payload: Mapping[str, object], field: str) -> None:
    unhashed = {key: value for key, value in payload.items() if key != field}
    if stable_hash(unhashed) != payload.get(field):
        raise ProtocolError(f"Promoted lock hash drifted: {field}.")


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError(f"Promoted JSON must be an object: {path}.")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _safe_member(root: Path, relative: str) -> Path:
    resolved_root = root.resolve()
    member = (resolved_root / relative).resolve()
    if member == resolved_root or not member.is_relative_to(resolved_root):
        raise ProtocolError("Promoted path escapes its artifact root.")
    return member


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ("REQUIRED_FILES", "validate_promoted_bank")
