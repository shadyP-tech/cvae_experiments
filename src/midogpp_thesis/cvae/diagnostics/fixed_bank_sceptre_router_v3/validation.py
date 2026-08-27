"""Independent semantic reconstruction of persisted SCEPTRE v3 bundles."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import read_json, sha256_file
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)

from ..fixed_bank_sceptre_router.calibration_gate import CalibrationGateDecision
from ..fixed_bank_sceptre_router.frozen_router_bundle import FrozenPrelabelRouter
from ..fixed_bank_sceptre_router.g_proposal_persistence import FrozenGProposal
from ..fixed_bank_sceptre_router.hashing import canonical_hash, require_sha256
from ..fixed_bank_sceptre_router.partitions import (
    CaseIdentity,
    build_three_role_partition,
)
from ..fixed_bank_sceptre_router.route_policy import FrozenRoutePolicy
from ..fixed_bank_sceptre_router.seals import build_global_decision_seal
from ..fixed_bank_sceptre_router.seals import (
    DurablePreterminalAttestation,
    FreshProcessValidation,
)
from ..fixed_bank_sceptre_router.support_tournament import SupportTournamentDecision
from ..fixed_bank_sceptre_router.uncertainty import (
    ActionUncertaintySummary,
    UncertaintyRouteDecision,
)
from .persistence import (
    DURABLE_ATTESTATION_MEMBER,
    FINAL_FRESH_ATTESTATION_MEMBER,
    FINAL_INDEX_MEMBER,
    PRETERMINAL_INDEX_MEMBER,
    PRETERMINAL_JSON_MEMBERS,
    PRETERMINAL_LAUNCH_MEMBERS,
    TERMINAL_RESULT_MEMBER,
    VALIDATION_INDEX_MEMBER,
    VALIDATION_REPORT_MEMBER,
    partition_payload,
)
from .authorization_lease import validate_authorization_lease_payload
from .config import load_config
from .experiment_contracts import (
    EXPECTED_EXECUTION_AMENDMENT_SHA256,
    EXPECTED_GENERATION_LOCK_HASH,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_SOURCE_INNER_AMENDMENT_SHA256,
    EXPECTED_TEST_ROWS,
    EXPECTED_TEST_ROWS_BY_CENTER,
    INPUT_ARTIFACT_IDS,
)
from .identity import EXPERIMENT_ID, PUBLICATION_STATUS, TERMINAL_DECISION
from .prediction_surface import (
    CANDIDATE_ARRAY_MEMBER,
    CANDIDATE_EXCLUSION_SENTINEL,
    EXACT_B_ARRAY_MEMBER,
    PREDICTION_INDEX_MEMBER,
    PREDICTION_RECEIPT_MEMBER,
    LOCKED_CLASSIFIER_SPEC,
)
from .reports import FINAL_REPORT_MEMBERS, build_validation_report
from .source_seal import build_source_snapshot_payload
from .workspace_manifest import validate_workspace_manifest_header
from .terminal_evaluation import (
    ActionAggregate,
    TerminalEvaluationResult,
    TerminalFoldMetric,
)
from .worker_runtime import validate_worker_runtime_smoke
from ..fixed_bank_sceptre_router.outcome_surface import ConfusionCounts


def validate_preterminal_bundle(root: str | Path) -> dict[str, object]:
    destination = Path(root).resolve()
    index = read_json(destination / PRETERMINAL_INDEX_MEMBER)
    if index.get("schema_version") != "sceptre_v3_preterminal_content_index_v1":
        raise ProtocolError("SCEPTRE v3 preterminal index schema drifted.")
    unhashed = {key: value for key, value in index.items() if key != "content_index_hash"}
    if index.get("content_index_hash") != canonical_hash(unhashed):
        raise ProtocolError("SCEPTRE v3 preterminal index hash drifted.")
    members = index.get("members")
    required_base_members = set(PRETERMINAL_JSON_MEMBERS) | set(
        PRETERMINAL_LAUNCH_MEMBERS
    )
    if not isinstance(members, Mapping) or not required_base_members <= set(members):
        raise ProtocolError("SCEPTRE v3 preterminal member inventory drifted.")
    for member, digest in members.items():
        path = _safe_member(destination, str(member))
        if sha256_file(path) != require_sha256(digest, "preterminal member"):
            raise ProtocolError("SCEPTRE v3 preterminal member hash drifted.")
    if (
        index.get("target_labels_opened_for_evaluation") is not False
        or index.get("raw_labels_persisted") is not False
        or index.get("closed_world") is not True
    ):
        raise ProtocolError("SCEPTRE v3 preterminal firewall drifted.")

    config = load_config(destination / "config.resolved.yaml")
    if index.get("config_hash") != config.config_hash:
        raise ProtocolError("SCEPTRE v3 preterminal config binding drifted.")
    input_binding = _validate_input_binding(destination, config_hash=config.config_hash)
    source = _validate_source_snapshot(
        destination,
        config_source_provenance=config.source_provenance,
    )
    lease = _validate_persisted_lease(destination, input_binding=input_binding)
    prediction, store_hash = _validate_prediction_store(
        destination,
        input_binding=input_binding,
        indexed_members=members,
    )
    expected_members = required_base_members | set(
        prediction["member_sha256"]
    )
    if set(members) != expected_members:
        raise ProtocolError("SCEPTRE v3 preterminal closed-world inventory drifted.")

    router = FrozenPrelabelRouter.from_payload(
        read_json(destination / "models/frozen_prelabel_router.json")
    )
    partition_raw = read_json(destination / "manifests/partition.json")
    identities_raw = partition_raw.get("identities")
    if not isinstance(identities_raw, list):
        raise ProtocolError("SCEPTRE v3 partition identities are absent.")
    partition = build_three_role_partition(
        tuple(CaseIdentity(**row) for row in identities_raw if isinstance(row, Mapping))
    )
    if _json_value(partition_payload(partition)) != partition_raw:
        raise ProtocolError("SCEPTRE v3 partition semantic replay drifted.")
    if partition.partition_hash != router.partition_hash:
        raise ProtocolError("SCEPTRE v3 router/partition binding drifted.")

    proposals_raw = read_json(destination / "models/g_proposals.json")
    proposal_rows = proposals_raw.get("proposals")
    if not isinstance(proposal_rows, list):
        raise ProtocolError("SCEPTRE v3 G proposals are absent.")
    proposals = tuple(_g_proposal(row) for row in proposal_rows)
    if tuple(row.target_center for row in proposals) != tuple(
        model.outer_target for model in router.models
    ):
        raise ProtocolError("SCEPTRE v3 G proposal inventory drifted.")

    supports = tuple(
        SupportTournamentDecision(**row)
        for row in _decision_rows(
            destination / "tables/support_decisions.json", "SUPPORT"
        )
    )
    calibration = tuple(
        CalibrationGateDecision(**row)
        for row in _decision_rows(
            destination / "tables/calibration_decisions.json", "CALIBRATION"
        )
    )
    uncertainty = tuple(
        _uncertainty_decision(row)
        for row in _decision_rows(
            destination / "tables/uncertainty_decisions.json", "UNCERTAINTY"
        )
    )
    if any(
        row.support_decision_hash
        not in {support.decision_hash for support in supports}
        for row in uncertainty
    ):
        raise ProtocolError("SCEPTRE v3 uncertainty/support lineage drifted.")

    g_decisions = {
        (proposal.target_center, fold): proposal.to_fold_receipt(fold).receipt_hash
        for proposal in proposals
        for fold in range(5)
    }
    support_decisions = {
        (row.target_center, row.fold_ordinal): row.decision_hash for row in supports
    }
    calibration_decisions = {
        (row.target_center, row.fold_ordinal): row.decision_hash
        for row in calibration
    }
    g_seal = build_global_decision_seal("G_LABEL_FREE", g_decisions)
    selection_seal = build_global_decision_seal(
        "S_Y_SELECTION", support_decisions, predecessor_seal_hash=g_seal.seal_hash
    )
    policy_seal = build_global_decision_seal(
        "A_CALIBRATED_ROUTE_OR_EXACT_B",
        calibration_decisions,
        predecessor_seal_hash=selection_seal.seal_hash,
    )
    for member, observed in (
        ("manifests/g_seal.json", g_seal),
        ("manifests/selection_seal.json", selection_seal),
        ("manifests/policy_seal.json", policy_seal),
    ):
        if read_json(destination / member) != _json_value(asdict(observed)):
            raise ProtocolError("SCEPTRE v3 global decision seal replay drifted.")
    policy = FrozenRoutePolicy.from_payload(
        read_json(destination / "manifests/frozen_route_policy.json")
    )
    if (
        policy.router_bundle_hash != router.full_router_sha256
        or policy.partition_hash != partition.partition_hash
        or policy.g_seal_hash != g_seal.seal_hash
        or policy.selection_seal_hash != selection_seal.seal_hash
        or policy.policy_seal_hash != policy_seal.seal_hash
    ):
        raise ProtocolError("SCEPTRE v3 frozen route policy replay drifted.")

    journal = read_json(destination / "reports/label_capability_journal.json")
    _validate_label_journal(
        journal,
        partition_hash=partition.partition_hash,
        prediction_store_hash=store_hash,
        authorization_lease_hash=str(lease["lease_hash"]),
    )
    if store_hash != index.get("prediction_store_hash"):
        raise ProtocolError("SCEPTRE v3 prediction store binding drifted.")
    development = read_json(destination / "tables/development_replay.json")
    if (
        development.get("replay_hash") != index.get("development_replay_hash")
        or development.get("full_router_sha256") != router.full_router_sha256
    ):
        raise ProtocolError("SCEPTRE v3 development replay binding drifted.")
    source_tree = _source_tree_hash(source)
    reconstruction = canonical_hash(
        {
            "schema_version": "sceptre_v3_preterminal_reconstruction_v1",
            "content_index_hash": index["content_index_hash"],
            "router_hash": router.full_router_sha256,
            "partition_hash": partition.partition_hash,
            "prediction_store_hash": store_hash,
            "g_proposal_hashes": [row.proposal_sha256 for row in proposals],
            "support_decision_hashes": [row.decision_hash for row in supports],
            "uncertainty_decision_hashes": [row.decision_hash for row in uncertainty],
            "calibration_decision_hashes": [row.decision_hash for row in calibration],
            "policy_seal_hash": policy_seal.seal_hash,
            "route_policy_hash": policy.policy_artifact_hash,
            "label_journal_hash": journal["journal_hash"],
            "source_tree_sha256": source_tree,
            "semantic_reconstruction_without_refit": True,
            "raw_labels_read": False,
        }
    )
    return {
        "status": "PASS",
        "content_index_hash": index["content_index_hash"],
        "policy_seal_hash": policy_seal.seal_hash,
        "route_policy_hash": policy.policy_artifact_hash,
        "prediction_store_hash": store_hash,
        "source_tree_sha256": source_tree,
        "reconstruction_hash": reconstruction,
        "semantic_reconstruction_without_refit": True,
        "raw_labels_read": False,
    }


def validate_final_bundle(root: str | Path) -> dict[str, object]:
    destination = Path(root).resolve()
    preterminal = validate_preterminal_bundle(destination)
    attestation = read_json(destination / DURABLE_ATTESTATION_MEMBER)
    _validate_durable_attestation(
        attestation,
        preterminal_content_index_hash=str(preterminal["content_index_hash"]),
        policy_seal_hash=str(preterminal["policy_seal_hash"]),
        source_tree_sha256=str(preterminal["source_tree_sha256"]),
        reconstruction_hash=str(preterminal["reconstruction_hash"]),
    )
    final_index = read_json(destination / FINAL_INDEX_MEMBER)
    unhashed = {
        key: value for key, value in final_index.items() if key != "content_index_hash"
    }
    if (
        final_index.get("schema_version") != "sceptre_v3_final_content_index_v1"
        or final_index.get("content_index_hash") != canonical_hash(unhashed)
        or final_index.get("preterminal_content_index_hash")
        != preterminal["content_index_hash"]
        or final_index.get("durable_attestation_hash")
        != attestation.get("attestation_hash")
        or final_index.get("publication_status") != PUBLICATION_STATUS
        or final_index.get("terminal_decision") != TERMINAL_DECISION
        or final_index.get("fresh_evidence") is not False
        or final_index.get("raw_labels_persisted") is not False
        or final_index.get("closed_world") is not True
    ):
        raise ProtocolError("SCEPTRE v3 final content index drifted.")
    result_raw = read_json(destination / TERMINAL_RESULT_MEMBER)
    result = _terminal_result(result_raw)
    if result.to_payload() != result_raw:
        raise ProtocolError("SCEPTRE v3 terminal result semantic replay drifted.")
    if (
        result.result_hash != final_index.get("terminal_result_hash")
        or result.route_policy_hash != preterminal.get("route_policy_hash")
        or result.prediction_store_hash != preterminal.get("prediction_store_hash")
        or sha256_file(destination / TERMINAL_RESULT_MEMBER)
        != final_index.get("terminal_result_file_sha256")
    ):
        raise ProtocolError("SCEPTRE v3 terminal result bytes drifted.")
    reports = final_index.get("report_member_hashes")
    if not isinstance(reports, Mapping) or set(reports) != set(FINAL_REPORT_MEMBERS):
        raise ProtocolError("SCEPTRE v3 final reports are absent.")
    for member, digest in reports.items():
        if sha256_file(_safe_member(destination, str(member))) != require_sha256(
            digest, "final report"
        ):
            raise ProtocolError("SCEPTRE v3 final report bytes drifted.")
    _validate_final_reports(destination, terminal_result_hash=result.result_hash)
    return {
        **preterminal,
        "final_content_index_hash": final_index["content_index_hash"],
        "terminal_result_hash": result.result_hash,
        "durable_attestation_hash": attestation["attestation_hash"],
        "final_reconstruction_hash": canonical_hash(
            {
                "preterminal_reconstruction_hash": preterminal[
                    "reconstruction_hash"
                ],
                "final_content_index_hash": final_index["content_index_hash"],
                "terminal_result_hash": result.result_hash,
                "durable_attestation_hash": attestation["attestation_hash"],
            }
        ),
    }


def validate_publication_bundle(root: str | Path) -> dict[str, object]:
    """Authenticate the final validators, report, and post-validation index."""

    destination = Path(root).resolve()
    final = validate_final_bundle(destination)
    attestation = read_json(destination / FINAL_FRESH_ATTESTATION_MEMBER)
    attestation_body = {
        key: value
        for key, value in attestation.items()
        if key != "attestation_hash"
    }
    validator_process_ids = attestation.get("validator_process_ids")
    validator_result_hashes = attestation.get("validator_result_hashes")
    if (
        attestation.get("schema_version")
        != "sceptre_v3_final_fresh_process_attestation_v1"
        or attestation.get("status") != "PASS"
        or attestation.get("attestation_hash") != canonical_hash(attestation_body)
        or attestation.get("checks") != final
        or attestation.get("checks_hash") != canonical_hash(final)
        or not isinstance(validator_process_ids, list)
        or len(validator_process_ids) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in validator_process_ids
        )
        or len(set(validator_process_ids)) != 2
        or not isinstance(validator_result_hashes, list)
        or len(validator_result_hashes) != 2
        or any(
            require_sha256(value, "final validator result") != value
            for value in validator_result_hashes
        )
        or attestation.get("fresh_process_count") != 2
        or attestation.get("process_launches_sequential") is not True
        or attestation.get("cuda_hidden") is not True
        or attestation.get("thread_count") != 1
        or attestation.get("semantic_reconstruction_without_refit") is not True
        or attestation.get("raw_labels_read") is not False
    ):
        raise ProtocolError("SCEPTRE v3 final validator attestation drifted.")

    report = read_json(destination / VALIDATION_REPORT_MEMBER)
    if report != build_validation_report(attestation):
        raise ProtocolError("SCEPTRE v3 final validation report drifted.")

    index = read_json(destination / VALIDATION_INDEX_MEMBER)
    index_body = {
        key: value for key, value in index.items() if key != "validation_index_hash"
    }
    members = index.get("members")
    expected_members = {
        FINAL_FRESH_ATTESTATION_MEMBER,
        VALIDATION_REPORT_MEMBER,
    }
    if (
        index.get("schema_version") != "sceptre_v3_postvalidation_index_v1"
        or index.get("status") != "PASS"
        or index.get("validation_index_hash") != canonical_hash(index_body)
        or index.get("final_content_index_hash")
        != final["final_content_index_hash"]
        or index.get("final_content_index_file_sha256")
        != sha256_file(destination / FINAL_INDEX_MEMBER)
        or index.get("final_fresh_process_attestation_hash")
        != attestation["attestation_hash"]
        or index.get("validation_report_hash") != report["report_hash"]
        or not isinstance(members, Mapping)
        or set(members) != expected_members
        or any(
            sha256_file(_safe_member(destination, member))
            != require_sha256(members[member], "post-validation member")
            for member in expected_members
        )
        or index.get("fresh_process_count") != 2
        or index.get("self_validation_claimed") is not False
        or index.get("publication_status") != PUBLICATION_STATUS
        or index.get("terminal_decision") != TERMINAL_DECISION
        or index.get("fresh_evidence") is not False
        or index.get("raw_labels_persisted") is not False
    ):
        raise ProtocolError("SCEPTRE v3 post-validation index drifted.")
    reconstruction = canonical_hash(
        {
            "final_reconstruction_hash": final["final_reconstruction_hash"],
            "final_fresh_process_attestation_hash": attestation[
                "attestation_hash"
            ],
            "validation_report_hash": report["report_hash"],
            "validation_index_hash": index["validation_index_hash"],
        }
    )
    return {
        **final,
        "final_fresh_process_attestation_hash": attestation["attestation_hash"],
        "validation_report_hash": report["report_hash"],
        "validation_index_hash": index["validation_index_hash"],
        "publication_reconstruction_hash": reconstruction,
        "postvalidation_index_authenticated": True,
    }


def _validate_input_binding(
    destination: Path, *, config_hash: str
) -> dict[str, object]:
    payload = read_json(destination / "provenance/input_binding.json")
    unhashed = {key: value for key, value in payload.items() if key != "binding_hash"}
    provenance = payload.get("workspace_provenance")
    if (
        payload.get("schema_version")
        != "sceptre_v3_exact_eight_input_binding_v1"
        or payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("config_hash") != require_sha256(config_hash, "config")
        or tuple(payload.get("direct_input_artifact_ids", ()))
        != INPUT_ARTIFACT_IDS
        or payload.get("direct_input_count") != 8
        or not isinstance(provenance, Mapping)
        or set(provenance) != set(INPUT_ARTIFACT_IDS)
        or payload.get("binding_hash") != canonical_hash(unhashed)
        or require_sha256(payload.get("admission_hash"), "admission")
        != payload.get("admission_hash")
        or require_sha256(
            payload.get("read_only_admission_hash"), "read-only admission"
        )
        != payload.get("read_only_admission_hash")
        or require_sha256(
            payload.get("worker_runtime_smoke_hash"), "worker runtime smoke"
        )
        != payload.get("worker_runtime_smoke_hash")
        or require_sha256(payload.get("cache_binding_hash"), "cache binding")
        != payload.get("cache_binding_hash")
        or payload.get("generation_lock_hash") != EXPECTED_GENERATION_LOCK_HASH
        or payload.get("source_inner_amendment_sha256")
        != EXPECTED_SOURCE_INNER_AMENDMENT_SHA256
        or payload.get("execution_amendment_sha256")
        != EXPECTED_EXECUTION_AMENDMENT_SHA256
        or payload.get("all_inputs_validated_before_authorization_claim") is not True
        or payload.get("target_labels_opened") is not False
        or payload.get("previous_stage90_output_used") is not False
        or payload.get("v2_output_used", False) is not False
        or payload.get("v2_run_state_used", False) is not False
        or payload.get("v2_scratch_or_checkpoint_used", False) is not False
        or payload.get("v2_execution_amendment_used", False) is not False
        or payload.get("prior_v2_execution_authorization_reused", False) is not False
    ):
        raise ProtocolError("SCEPTRE v3 persisted input binding drifted.")

    workspace = read_json(destination / "provenance/input_artifacts.json")
    validate_workspace_manifest_header(workspace)
    rows = workspace.get("input_artifacts")
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise ProtocolError("SCEPTRE v3 persisted workspace rows are malformed.")
    by_id = {str(row.get("artifact_id")): dict(row) for row in rows}
    if set(by_id) != set(INPUT_ARTIFACT_IDS) or any(
        dict(provenance[artifact_id]) != by_id[artifact_id]
        for artifact_id in INPUT_ARTIFACT_IDS
    ):
        raise ProtocolError("SCEPTRE v3 input/workspace provenance lineage drifted.")
    return payload


def _validate_source_snapshot(
    destination: Path,
    *,
    config_source_provenance: Mapping[str, object],
) -> dict[str, object]:
    payload = read_json(destination / "provenance/source_snapshot.json")
    current = build_source_snapshot_payload()
    if payload != current:
        raise ProtocolError("SCEPTRE v3 persisted/current source snapshot drifted.")
    expected = {
        "source_snapshot_schema": payload.get("schema_version"),
        "source_snapshot_manifest_sha256": payload.get("manifest_sha256"),
        "source_snapshot_tree_sha256": payload.get("tree_sha256"),
        "source_snapshot_member_count": payload.get("member_count"),
        "source_snapshot_member_pattern": payload.get("member_pattern"),
        "source_snapshot_excludes_bytecode_and_cache": True,
    }
    if any(config_source_provenance.get(key) != value for key, value in expected.items()):
        raise ProtocolError("SCEPTRE v3 config/source provenance drifted.")
    return payload


def _validate_persisted_lease(
    destination: Path, *, input_binding: Mapping[str, object]
) -> Mapping[str, object]:
    payload = read_json(
        destination / "provenance/authorization_consumption_lease.json"
    )
    authenticated = validate_authorization_lease_payload(
        payload, expected_status="CLAIMED_IN_PROGRESS"
    )
    if (
        authenticated.get("experiment_id") != EXPERIMENT_ID
        or authenticated.get("admission_hash") != input_binding.get("admission_hash")
        or authenticated.get("authorization_exhausted") is not True
        or authenticated.get("recovery_allowed") is not False
        or authenticated.get("output_deletion_restores_authority") is not False
        or authenticated.get("predecessor_lease_hash") is not None
    ):
        raise ProtocolError("SCEPTRE v3 persisted authorization lease drifted.")
    return authenticated


def _validate_prediction_store(
    destination: Path,
    *,
    input_binding: Mapping[str, object],
    indexed_members: Mapping[str, object],
) -> tuple[dict[str, object], str]:
    payload = read_json(destination / "manifests/prediction_store.json")
    expected_members = {
        f"prediction_store/{CANDIDATE_ARRAY_MEMBER}",
        f"prediction_store/{EXACT_B_ARRAY_MEMBER}",
        f"prediction_store/{PREDICTION_INDEX_MEMBER}",
        f"prediction_store/{PREDICTION_RECEIPT_MEMBER}",
    }
    member_hashes = payload.get("member_sha256")
    receipt = payload.get("physical_receipt")
    if (
        payload.get("schema_version") != "sceptre_v3_durable_prediction_store_v1"
        or not isinstance(member_hashes, Mapping)
        or set(member_hashes) != expected_members
        or not isinstance(receipt, Mapping)
        or payload.get("store_hash") != receipt.get("receipt_sha256")
        or tuple(payload.get("candidate_source_order", ())) != tuple(CENTERS)
        or payload.get("manifest_opened") is not False
        or payload.get("outcomes_available") is not False
        or payload.get("raw_sample_paths_available") is not False
    ):
        raise ProtocolError("SCEPTRE v3 durable prediction-store receipt drifted.")
    for member, digest in member_hashes.items():
        authenticated = require_sha256(digest, "prediction member")
        if indexed_members.get(member) != authenticated:
            raise ProtocolError("SCEPTRE v3 prediction/index member lineage drifted.")

    prediction_root = destination / "prediction_store"
    index = read_json(prediction_root / PREDICTION_INDEX_MEMBER)
    disk_receipt = read_json(prediction_root / PREDICTION_RECEIPT_MEMBER)
    index_body = {key: value for key, value in index.items() if key != "index_sha256"}
    receipt_body = {
        key: value for key, value in disk_receipt.items() if key != "receipt_sha256"
    }
    geometry = disk_receipt.get("geometry")
    row_ids = index.get("row_ids")
    row_centers = index.get("row_centers")
    fit_rows = index.get("fit_rows")
    if (
        dict(receipt) != disk_receipt
        or disk_receipt.get("schema_version")
        != "midogpp_sceptre_v3_prediction_receipt_v1"
        or disk_receipt.get("status")
        != "SEALED_ALL_LABEL_FREE_CANDIDATE_AND_EXACT_B_PREDICTIONS"
        or disk_receipt.get("receipt_sha256") != canonical_hash(receipt_body)
        or index.get("schema_version") != "midogpp_sceptre_v3_prediction_index_v1"
        or index.get("index_sha256") != canonical_hash(index_body)
        or disk_receipt.get("prediction_index_sha256") != index.get("index_sha256")
        or disk_receipt.get("cache_binding_hash")
        != input_binding.get("cache_binding_hash")
        or index.get("cache_binding_hash") != input_binding.get("cache_binding_hash")
        or not isinstance(geometry, Mapping)
        or geometry != index.get("geometry")
        or tuple(geometry.get("centers", ())) != tuple(CENTERS)
        or tuple(geometry.get("training_seeds", ())) != tuple(TRAINING_SEEDS)
        or tuple(geometry.get("generation_seeds", ())) != tuple(GENERATION_SEEDS)
        or geometry.get("evaluation_rows") != EXPECTED_TEST_ROWS
        or geometry.get("rows_by_center") != EXPECTED_TEST_ROWS_BY_CENTER
        or not isinstance(row_ids, list)
        or not isinstance(row_centers, list)
        or len(row_ids) != EXPECTED_TEST_ROWS
        or len(row_centers) != EXPECTED_TEST_ROWS
        or len(set(str(value) for value in row_ids)) != EXPECTED_TEST_ROWS
        or {center: row_centers.count(center) for center in CENTERS}
        != EXPECTED_TEST_ROWS_BY_CENTER
        or not isinstance(fit_rows, list)
        or index.get("fit_index_sha256") != canonical_hash(fit_rows)
        or disk_receipt.get("fit_index_sha256") != index.get("fit_index_sha256")
        or disk_receipt.get("target_expert_excluded_from_every_exact_b_fit")
        is not True
        or disk_receipt.get("target_expert_excluded_from_every_candidate_score")
        is not True
        or disk_receipt.get("candidate_exclusion_sentinel")
        != float(CANDIDATE_EXCLUSION_SENTINEL)
        or index.get("candidate_target_exclusion_mode")
        != "MASKED_BEFORE_SCORING"
        or index.get("candidate_exclusion_sentinel")
        != float(CANDIDATE_EXCLUSION_SENTINEL)
        or disk_receipt.get("manifest_opened") is not False
        or disk_receipt.get("outcomes_available") is not False
        or disk_receipt.get("raw_sample_paths_available") is not False
        or disk_receipt.get("classifier_refit_after_seal") is not False
        or disk_receipt.get("seed_selection_performed") is not False
        or disk_receipt.get("synthetic_test_mode") is not False
        or index.get("candidate_source_order") != list(CENTERS)
    ):
        raise ProtocolError("SCEPTRE v3 physical prediction receipt drifted.")

    candidate_path = prediction_root / CANDIDATE_ARRAY_MEMBER
    exact_b_path = prediction_root / EXACT_B_ARRAY_MEMBER
    candidate = np.load(candidate_path, mmap_mode="r", allow_pickle=False)
    exact_b = np.load(exact_b_path, mmap_mode="r", allow_pickle=False)
    candidate_shape = (len(TRAINING_SEEDS) * len(GENERATION_SEEDS), len(CENTERS), EXPECTED_TEST_ROWS)
    exact_b_shape = (len(TRAINING_SEEDS) * len(GENERATION_SEEDS), EXPECTED_TEST_ROWS)
    if (
        candidate.shape != candidate_shape
        or exact_b.shape != exact_b_shape
        or candidate.dtype != np.float32
        or exact_b.dtype != np.float32
        or candidate.flags.writeable
        or exact_b.flags.writeable
        or disk_receipt.get("candidate_shape") != list(candidate_shape)
        or disk_receipt.get("exact_b_shape") != list(exact_b_shape)
        or disk_receipt.get("candidate_array_file_sha256") != sha256_file(candidate_path)
        or disk_receipt.get("exact_b_array_file_sha256") != sha256_file(exact_b_path)
        or not np.isfinite(candidate).all()
        or not np.isfinite(exact_b).all()
        or np.any((exact_b < 0.0) | (exact_b > 1.0))
    ):
        raise ProtocolError("SCEPTRE v3 physical prediction arrays drifted.")
    centers_array = np.asarray(row_centers, dtype=object)
    for source_ordinal, source in enumerate(CENTERS):
        forbidden = centers_array == source
        legal = ~forbidden
        if (
            not np.all(
                candidate[:, source_ordinal, forbidden]
                == CANDIDATE_EXCLUSION_SENTINEL
            )
            or np.any(
                (candidate[:, source_ordinal, legal] < 0.0)
                | (candidate[:, source_ordinal, legal] > 1.0)
            )
        ):
            raise ProtocolError("SCEPTRE v3 candidate target mask drifted.")
    _validate_fit_rows(fit_rows, row_centers=row_centers)
    store_hash = require_sha256(payload.get("store_hash"), "prediction store")
    return payload, store_hash


def _validate_fit_rows(
    rows: list[object], *, row_centers: list[object]
) -> None:
    expected_fit_count = len(TRAINING_SEEDS) * len(GENERATION_SEEDS) * 2 * len(CENTERS)
    counts = {center: row_centers.count(center) for center in CENTERS}
    if len(rows) != expected_fit_count:
        raise ProtocolError("SCEPTRE v3 prediction fit inventory drifted.")
    for global_ordinal, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ProtocolError("SCEPTRE v3 prediction fit row is malformed.")
        wrapper_fields = {
            "global_fit_ordinal",
            "seed_cell_ordinal",
            "within_cell_fit_ordinal",
        }
        body = {
            key: value
            for key, value in raw.items()
            if key != "fit_sha256" and key not in wrapper_fields
        }
        family = raw.get("family")
        if (
            raw.get("fit_sha256") != canonical_hash(body)
            or raw.get("global_fit_ordinal") != global_ordinal
            or raw.get("seed_cell_ordinal") != global_ordinal // (2 * len(CENTERS))
            or raw.get("within_cell_fit_ordinal")
            != global_ordinal % (2 * len(CENTERS))
            or raw.get("converged") is not True
            or raw.get("classifier_config_hash") != LOCKED_CLASSIFIER_SPEC.config_hash
        ):
            raise ProtocolError("SCEPTRE v3 prediction fit receipt drifted.")
        if family == "single_source":
            source = str(raw.get("source_center"))
            if (
                source not in CENTERS
                or raw.get("excluded_evaluation_center") != source
                or raw.get("masked_row_count") != counts[source]
                or raw.get("evaluated_row_count") != EXPECTED_TEST_ROWS - counts[source]
            ):
                raise ProtocolError("SCEPTRE v3 single-source exclusion drifted.")
        elif family == "exact_B":
            target = str(raw.get("target_center"))
            sources = tuple(raw.get("source_centers", ()))
            if (
                target not in CENTERS
                or target in sources
                or set(sources) != set(CENTERS) - {target}
                or raw.get("excluded_evaluation_center") is not None
                or raw.get("masked_row_count") != 0
                or raw.get("evaluated_row_count") != counts[target]
            ):
                raise ProtocolError("SCEPTRE v3 exact-B exclusion drifted.")
        else:
            raise ProtocolError("SCEPTRE v3 prediction fit family drifted.")


def _validate_durable_attestation(
    payload: Mapping[str, object],
    *,
    preterminal_content_index_hash: str,
    policy_seal_hash: str,
    source_tree_sha256: str,
    reconstruction_hash: str,
) -> None:
    process_ids = payload.get("validator_process_ids")
    receipts = payload.get("validator_receipt_hashes")
    if (
        payload.get("schema_version")
        != "sceptre_v3_durable_preterminal_attestation_v1"
        or payload.get("preterminal_content_index_hash")
        != preterminal_content_index_hash
        or payload.get("policy_seal_hash") != policy_seal_hash
        or payload.get("source_tree_sha256") != source_tree_sha256
        or payload.get("reconstruction_hash") != reconstruction_hash
        or not isinstance(process_ids, list)
        or len(process_ids) != 2
        or len(set(process_ids)) != 2
        or not isinstance(receipts, list)
        or len(receipts) != 2
        or payload.get("fresh_process_count") != 2
        or payload.get("byte_identical_reconstruction") is not True
        or payload.get("target_labels_opened_for_evaluation") is not False
    ):
        raise ProtocolError("SCEPTRE v3 durable attestation drifted.")
    validations = tuple(
        FreshProcessValidation(
            process_id=int(process_id),
            policy_seal_hash=policy_seal_hash,
            source_tree_sha256=source_tree_sha256,
            reconstruction_hash=reconstruction_hash,
            receipt_hash=str(receipt),
        )
        for process_id, receipt in zip(process_ids, receipts, strict=True)
    )
    durable = DurablePreterminalAttestation(
        policy_seal_hash=policy_seal_hash,
        validations=validations,  # type: ignore[arg-type]
        attestation_hash=str(payload.get("attestation_hash", "")),
    )
    if durable.attestation_hash != payload.get("attestation_hash"):
        raise ProtocolError("SCEPTRE v3 durable attestation hash drifted.")


def _validate_final_reports(
    destination: Path, *, terminal_result_hash: str
) -> None:
    reports = {
        member: read_json(destination / member) for member in FINAL_REPORT_MEMBERS
    }
    for payload in reports.values():
        body = {key: value for key, value in payload.items() if key != "report_hash"}
        if payload.get("report_hash") != canonical_hash(body):
            raise ProtocolError("SCEPTRE v3 final report hash drifted.")
    summary = reports["reports/diagnostic_summary.json"]
    leakage = reports["reports/leakage_report.json"]
    publication = reports["reports/publication_decision.json"]
    runtime_report = reports["reports/runtime_summary.json"]
    boundary = reports["reports/claim_boundary.json"]
    runtime = runtime_report.get("runtime")
    smoke = runtime.get("worker_runtime_smoke") if isinstance(runtime, Mapping) else None
    if not isinstance(smoke, Mapping):
        raise ProtocolError("SCEPTRE v3 runtime smoke report is absent.")
    authenticated_smoke = validate_worker_runtime_smoke(smoke)
    if (
        summary.get("terminal_result_hash") != terminal_result_hash
        or summary.get("publication_status") != PUBLICATION_STATUS
        or summary.get("terminal_decision") != TERMINAL_DECISION
        or summary.get("metrics_are_nelbo_compatibility") is not False
        or leakage.get("status") != "PASS"
        or leakage.get("strict_outer_query_and_candidate_center_exclusion")
        is not True
        or leakage.get("selection_calibration_evaluation_whole_case_disjoint")
        is not True
        or leakage.get("target_expert_rows_masked_before_candidate_scoring")
        is not True
        or leakage.get("route_policy_durable_before_evaluation_labels") is not True
        or leakage.get("raw_labels_persisted") is not False
        or leakage.get("fresh_evidence") is not False
        or publication.get("decision") != TERMINAL_DECISION
        or publication.get("publication_status") != PUBLICATION_STATUS
        or publication.get("routing_success_claim_allowed") is not False
        or publication.get("nelbo_compatibility_claim_allowed") is not False
        or publication.get("promotion_or_deployment_allowed") is not False
        or publication.get("may_feed_another_experiment") is not False
        or runtime.get("worker_runtime_smoke_hash")
        != authenticated_smoke["worker_runtime_smoke_hash"]
        or runtime.get("runtime_launch_admission_hash") is None
        or boundary.get("decision") != TERMINAL_DECISION
        or boundary.get("downstream_classifier_utility_is_not_CVAE_NELBO_routing_evidence")
        is not True
    ):
        raise ProtocolError("SCEPTRE v3 final claim firewall drifted.")


def _g_proposal(raw: object) -> FrozenGProposal:
    if not isinstance(raw, Mapping):
        raise ProtocolError("SCEPTRE v3 G proposal row is malformed.")
    excluded = {
        "schema_version",
        "phase",
        "score_semantics",
        "higher_is_better",
        "labels_consumed",
        "publication_status",
        "terminal_decision",
    }
    kwargs = {key: value for key, value in raw.items() if key not in excluded}
    if "winner_sources" in kwargs:
        kwargs["winner_sources"] = tuple(kwargs["winner_sources"])
    return FrozenGProposal(**kwargs)


def _decision_rows(path: Path, role: str) -> tuple[Mapping[str, object], ...]:
    payload = read_json(path)
    rows = payload.get("rows")
    if (
        payload.get("schema_version") != "sceptre_v3_decision_inventory_v1"
        or payload.get("role") != role
        or not isinstance(rows, list)
        or payload.get("row_count") != len(rows)
        or payload.get("inventory_hash") != canonical_hash(rows)
        or any(not isinstance(row, Mapping) for row in rows)
    ):
        raise ProtocolError("SCEPTRE v3 decision inventory drifted.")
    return tuple(rows)  # type: ignore[return-value]


def _uncertainty_decision(raw: Mapping[str, object]) -> UncertaintyRouteDecision:
    kwargs = dict(raw)
    summaries = kwargs.get("action_summaries")
    if not isinstance(summaries, list):
        raise ProtocolError("SCEPTRE v3 uncertainty summaries are absent.")
    kwargs["action_summaries"] = tuple(
        ActionUncertaintySummary(**row)
        for row in summaries
        if isinstance(row, Mapping)
    )
    if len(kwargs["action_summaries"]) != len(summaries):
        raise ProtocolError("SCEPTRE v3 uncertainty summary row drifted.")
    return UncertaintyRouteDecision(**kwargs)


def _validate_label_journal(
    payload: Mapping[str, object],
    *,
    partition_hash: str,
    prediction_store_hash: str,
    authorization_lease_hash: str,
) -> None:
    events = payload.get("events")
    expected_keys = tuple(
        (center, fold) for center in CENTERS for fold in range(5)
    )
    if (
        not isinstance(events, list)
        or len(events) != 2 * len(expected_keys)
        or payload.get("schema_version")
        != "sceptre_v3_label_capability_journal_v1"
        or payload.get("partition_hash") != partition_hash
        or payload.get("prediction_store_hash") != prediction_store_hash
        or payload.get("authorization_lease_hash") != authorization_lease_hash
        or payload.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256
        or payload.get("raw_labels_persisted") is not False
        or payload.get("sample_paths_persisted") is not False
    ):
        raise ProtocolError("SCEPTRE v3 label journal drifted.")
    predecessor = None
    hashes = []
    for ordinal, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise ProtocolError("SCEPTRE v3 label event is malformed.")
        forbidden = {"labels", "label", "image_path", "sample_path"} & set(event)
        unhashed = {key: value for key, value in event.items() if key != "event_hash"}
        if (
            forbidden
            or event.get("event_ordinal") != ordinal
            or event.get("predecessor_event_hash") != predecessor
            or event.get("event_hash") != canonical_hash(unhashed)
            or event.get("prediction_store_hash") != prediction_store_hash
            or event.get("authorization_lease_hash") != authorization_lease_hash
            or event.get("raw_labels_persisted") is not False
        ):
            raise ProtocolError("SCEPTRE v3 label event chain drifted.")
        predecessor = event["event_hash"]
        hashes.append(predecessor)
    selection = events[: len(expected_keys)]
    calibration = events[len(expected_keys) :]
    for event, (center, fold) in zip(selection, expected_keys, strict=True):
        if (
            event.get("event") != "SELECTION_LABELS_DECODED"
            or event.get("target_center") != center
            or event.get("fold_ordinal") != fold
            or not isinstance(event.get("row_count"), int)
            or int(event["row_count"]) <= 0
            or event.get("manifest_rows_decoded") != event.get("row_count")
        ):
            raise ProtocolError("SCEPTRE v3 selection label journal drifted.")
    for event, (center, fold) in zip(calibration, expected_keys, strict=True):
        event_type = event.get("event")
        decoded = event_type == "CALIBRATION_LABELS_DECODED"
        skipped = event_type == "CALIBRATION_SKIPPED_SUPPORT_FALLBACK"
        if (
            (not decoded and not skipped)
            or event.get("target_center") != center
            or event.get("fold_ordinal") != fold
            or (
                decoded
                and (
                    not isinstance(event.get("row_count"), int)
                    or int(event["row_count"]) <= 0
                    or event.get("manifest_rows_decoded") != event.get("row_count")
                )
            )
            or (
                skipped
                and (
                    event.get("row_count") != 0
                    or event.get("manifest_rows_decoded") != 0
                )
            )
        ):
            raise ProtocolError("SCEPTRE v3 calibration label journal drifted.")
    expected = canonical_hash(
        {
            "schema_version": "sceptre_v3_label_journal_chain_v1",
            "partition_hash": payload.get("partition_hash"),
            "prediction_store_hash": payload.get("prediction_store_hash"),
            "authorization_lease_hash": payload.get("authorization_lease_hash"),
            "manifest_sha256": payload.get("manifest_sha256"),
            "event_hashes": hashes,
            "raw_labels_persisted": False,
        }
    )
    if payload.get("journal_hash") != expected:
        raise ProtocolError("SCEPTRE v3 label journal hash drifted.")


def _terminal_result(raw: Mapping[str, object]) -> TerminalEvaluationResult:
    folds_raw = raw.get("folds")
    if not isinstance(folds_raw, list):
        raise ProtocolError("SCEPTRE v3 terminal folds are absent.")
    folds = tuple(_terminal_fold(row) for row in folds_raw if isinstance(row, Mapping))
    if len(folds) != len(folds_raw):
        raise ProtocolError("SCEPTRE v3 terminal fold row drifted.")
    return TerminalEvaluationResult(
        route_policy_hash=str(raw.get("route_policy_hash", "")),
        prediction_store_hash=str(raw.get("prediction_store_hash", "")),
        terminal_capability_hash=str(raw.get("terminal_capability_hash", "")),
        folds=folds,
        result_hash=str(raw.get("result_hash", "")),
    )


def _terminal_fold(raw: Mapping[str, object]) -> TerminalFoldMetric:
    return TerminalFoldMetric(
        target_center=str(raw["target_center"]),
        fold_ordinal=int(raw["fold_ordinal"]),
        fold_hash=str(raw["fold_hash"]),
        evaluation_case_set_hash=str(raw["evaluation_case_set_hash"]),
        route=str(raw["route"]),
        route_aggregate=_action_aggregate(raw["route_metrics"]),
        exact_b_aggregate=_action_aggregate(raw["exact_b_metrics"]),
        oracle_action=str(raw["oracle_action_descriptive_only"]),
        oracle_bacc=float(raw["oracle_bacc_descriptive_only"]),
        case_count=int(raw["case_count"]),
        fold_metric_hash=str(raw["fold_metric_hash"]),
    )


def _action_aggregate(raw: object) -> ActionAggregate:
    if not isinstance(raw, Mapping) or not isinstance(raw.get("confusion"), Mapping):
        raise ProtocolError("SCEPTRE v3 terminal aggregate is malformed.")
    confusion = raw["confusion"]
    return ActionAggregate(
        confusion=ConfusionCounts(
            int(confusion["tn"]),
            int(confusion["fp"]),
            int(confusion["fn"]),
            int(confusion["tp"]),
        ),
        brier_sum=float(raw["brier_sum"]),
        log_loss_sum=float(raw["log_loss_sum"]),
        observation_count=int(raw["observation_count"]),
    )


def _source_tree_hash(payload: Mapping[str, object]) -> str:
    for key in (
        "source_tree_sha256",
        "scientific_source_tree_sha256",
        "tree_sha256",
    ):
        value = payload.get(key)
        if value is not None:
            return require_sha256(value, "source tree")
    raise ProtocolError("SCEPTRE v3 source snapshot lacks a tree hash.")


def _safe_member(root: Path, relative: str) -> Path:
    candidate = root / relative
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ProtocolError("SCEPTRE v3 persisted member is absent.") from exc
    if candidate.is_symlink() or root not in resolved.parents or not resolved.is_file():
        raise ProtocolError("SCEPTRE v3 persisted member is unsafe.")
    return resolved


def _json_value(value: object) -> object:
    return json.loads(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )


__all__ = (
    "validate_final_bundle",
    "validate_preterminal_bundle",
    "validate_publication_bundle",
)
