"""Execute the label-sealed Stage-70 descriptive policy comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from ...common.hashing import stable_hash
from ...common.staged_directory import staged_existing_directory
from ...data.features.stage70_test_cache import (
    REPRESENTATION_ID,
    load_validated_stage70_test_cache,
)
from ..generation import read_generation_lock
from ..protocol import ProtocolError
from ..reporting import write_json
from .authorization import (
    load_final_authorization_config,
    read_final_authorization_token,
    validate_final_prediction_authorization,
)
from .bootstrap import paired_descriptive_bootstrap
from .bundle import (
    seal_prediction_pass,
    write_authorization_phase,
    write_content_index,
    write_scored_bundle,
)
from .config import FrozenPolicyDownstreamConfig
from .contracts import (
    CLAIM_SCOPE,
    EXPERIMENT_ID,
    POLICY_ARMS,
)
from .contrasts import build_descriptive_contrasts
from .policy_adapters import load_frozen_policy_replicates
from .prediction import FrozenPolicyPredictionPass, run_label_free_prediction_pass
from .prediction_seal import load_canonical_prediction_seal_binding
from .scoring import score_persisted_predictions
from .source_blocks import materialize_source_blocks
from .target_loader import (
    load_label_sealed_target_frames,
    open_scoring_labels_after_prediction_seal,
)
from .validation import (
    validate_frozen_policy_downstream_bundle,
    write_validation_report,
)
from .workspace_binding import validate_frozen_policy_downstream_workspace_binding


_PREPARED_FILES = {
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
}


def run_frozen_policy_downstream(
    config: FrozenPolicyDownstreamConfig,
) -> Path:
    """Run prediction first and permit scoring labels only after durable sealing."""

    validate_frozen_policy_downstream_workspace_binding(config)
    final_root = config.artifact_root
    if _is_complete(final_root):
        validate_frozen_policy_downstream_bundle(final_root)
        return final_root
    _require_workspace_prepared_root(final_root)

    with staged_existing_directory(final_root) as root:
        write_json(
            root / "reports/run_state.json",
            {
                "schema_version": "midogpp_stage70_run_state_v1",
                "status": "RUNNING",
                "phase": "AUTHORIZATION_VALIDATION",
            },
        )
        try:
            _execute(config, root)
        except Exception:
            write_json(
                root / "reports/run_state.json",
                {
                    "schema_version": "midogpp_stage70_run_state_v1",
                    "status": "FAILED",
                    "phase": "FAILED_BEFORE_PUBLICATION",
                },
            )
            raise
    return final_root


def _execute(config: FrozenPolicyDownstreamConfig, root: Path) -> None:
    final_config = load_final_authorization_config(
        config.final_authorization_root / "config.resolved.yaml"
    )
    authorization_checks = validate_final_prediction_authorization(
        config.final_authorization_root,
        config=final_config,
    )
    token = read_final_authorization_token(config.final_authorization_root)
    token_payload = token.to_payload()
    if (
        token_payload.get("authorized_consumer_experiment_id") != EXPERIMENT_ID
        or token_payload.get("claim_scope") != "target_evaluation_authorization"
        or token_payload.get("prediction_allowed") is not True
        or token_payload.get("label_access_allowed") is not False
        or token_payload.get("metric_scoring_allowed") is not False
        or authorization_checks.get("status") != "PASS"
    ):
        raise ProtocolError("Stage-70 final authorization does not permit this consumer.")

    cache = load_validated_stage70_test_cache(config.target_cache_root)
    cache_summary = dict(cache.summary)
    cache_report = _json(
        config.target_cache_root / "reports/cache_builder_report.json"
    )
    model_identity = _mapping(cache_report, "model_identity")
    backbone_identity_hash = stable_hash(dict(model_identity))
    _validate_runtime_bindings(
        config,
        token_payload=token_payload,
        cache_summary=cache_summary,
        cache_report=cache_report,
        backbone_identity_hash=backbone_identity_hash,
    )

    generation_lock = read_generation_lock(
        config.generation_lock_root / "manifests/generation_lock.json"
    )
    if token_payload.get("generation_lock_hash") != generation_lock.generation_lock_hash:
        raise ProtocolError("Stage-70 authorization/GenerationLock identity drifted.")
    replicates = load_frozen_policy_replicates(
        generation_lock=generation_lock,
        equal_union_root=config.equal_union_policy_root,
        metadata_tie_union_root=config.metadata_policy_root,
        utility_regret_root=config.utility_policy_root,
    )
    if config.classifier.config_hash != token_payload.get("classifier_config_hash"):
        raise ProtocolError("Stage-70 classifier differs from final authorization.")

    target_frames = load_label_sealed_target_frames(cache)
    seal_binding = load_canonical_prediction_seal_binding(
        final_authorization_root=config.final_authorization_root,
        target_cache_root=config.target_cache_root,
        scoring_manifest_path=config.scoring_manifest_path,
        expected_manifest_sha256=config.scoring_manifest_sha256,
    )
    write_authorization_phase(
        root,
        binding=seal_binding,
    )
    write_json(
        root / "manifests/protocol_manifest.json",
        _protocol_manifest(
            config,
            token_payload=token_payload,
            cache_summary=cache_summary,
            backbone_identity_hash=backbone_identity_hash,
        ),
    )
    write_json(
        root / "manifests/evaluation_plan.json",
        _evaluation_plan(config, token_payload=token_payload, replicates=replicates),
    )

    source_cache_root = (
        config.artifact_root.parent / ".uniform_b_v2_stage70_source_blocks_v1"
    )
    source_publication_root = root / "arrays/source_blocks"
    source_blocks, source_records = materialize_source_blocks(
        generation_lock=generation_lock,
        bank_root=config.bank_root,
        cache_root=source_cache_root,
        dataset_contract_hash=config.dataset_contract_hash,
        representation_id=config.representation_id,
        backbone_identity_hash=config.backbone_identity_hash,
        device=config.device,
        publication_root=source_publication_root,
    )
    normalized_source_records = [
        {
            **record,
            "artifact_member": str(record["member_path"]),
        }
        for record in source_records
    ]
    write_json(
        root / "manifests/source_block_index.json",
        {
            "schema_version": "midogpp_stage70_source_block_index_v1",
            "source_block_count": len(normalized_source_records),
            "records": normalized_source_records,
            "target_labels_opened": False,
        },
    )

    prediction_pass = run_label_free_prediction_pass(
        replicates=replicates,
        source_blocks=source_blocks,
        target_frames=target_frames,
        classifier_spec=config.classifier,
    )
    _write_composition_index(root, prediction_pass)
    sealed = seal_prediction_pass(
        root,
        prediction_pass,
        expected_binding=seal_binding,
    )

    # This accessor is intentionally unreachable until every prediction cell and
    # its two SHA-256 seals have been durably written above.
    labels = open_scoring_labels_after_prediction_seal(
        sealed,
        manifest_path=config.scoring_manifest_path,
        expected_manifest_sha256=config.scoring_manifest_sha256,
        final_authorization_root=config.final_authorization_root,
        target_cache_root=config.target_cache_root,
    )
    scored = score_persisted_predictions(sealed, labels)
    summaries, deltas = build_descriptive_contrasts(scored.metrics)
    bootstrap = paired_descriptive_bootstrap(
        scored.case_confusions,
        seed=config.bootstrap_seed,
        valid_replicates=config.bootstrap_valid_replicates,
        max_attempts=config.bootstrap_max_attempts,
    )
    write_scored_bundle(
        root,
        scored=scored,
        summaries=summaries,
        deltas=deltas,
        bootstrap=bootstrap,
        final_authorization_hash=token.authorization_token_hash,
    )
    write_content_index(root)
    checks = validate_frozen_policy_downstream_bundle(root, allow_pending=True)
    write_validation_report(root, checks)
    validate_frozen_policy_downstream_bundle(root)


def _validate_runtime_bindings(
    config: FrozenPolicyDownstreamConfig,
    *,
    token_payload: Mapping[str, object],
    cache_summary: Mapping[str, object],
    cache_report: Mapping[str, object],
    backbone_identity_hash: str,
) -> None:
    manifest_sha256 = _sha256_file(config.scoring_manifest_path)
    expected = {
        "final_authorization_hash": (
            config.final_authorization_hash,
            token_payload.get("authorization_token_hash"),
        ),
        "target_cache_content_hash": (
            config.target_cache_content_hash,
            cache_summary.get("content_hash"),
        ),
        "target_cache_token_hash": (
            config.target_cache_content_hash,
            token_payload.get("target_cache_content_hash"),
        ),
        "target_row_order_hash": (
            config.target_row_order_hash,
            cache_summary.get("row_order_hash"),
        ),
        "target_row_order_token_hash": (
            config.target_row_order_hash,
            token_payload.get("target_cache_row_order_hash"),
        ),
        "scoring_manifest_sha256": (
            config.scoring_manifest_sha256,
            manifest_sha256,
        ),
        "scoring_manifest_cache_sha256": (
            config.scoring_manifest_sha256,
            cache_summary.get("manifest_sha256"),
        ),
        "scoring_manifest_token_sha256": (
            config.scoring_manifest_sha256,
            token_payload.get("scoring_manifest_sha256"),
        ),
        "dataset_contract_hash": (
            config.dataset_contract_hash,
            config.scoring_manifest_sha256,
        ),
        "representation_id": (
            config.representation_id,
            cache_report.get("representation_id"),
        ),
        "canonical_representation_id": (
            config.representation_id,
            REPRESENTATION_ID,
        ),
        "backbone_identity_hash": (
            config.backbone_identity_hash,
            backbone_identity_hash,
        ),
    }
    mismatch = [key for key, values in expected.items() if values[0] != values[1]]
    if mismatch:
        raise ProtocolError(f"Stage-70 runtime identity binding drifted: {mismatch}.")


def _protocol_manifest(
    config: FrozenPolicyDownstreamConfig,
    *,
    token_payload: Mapping[str, object],
    cache_summary: Mapping[str, object],
    backbone_identity_hash: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "midogpp_stage70_descriptive_protocol_v1",
        "experiment_id": EXPERIMENT_ID,
        "claim_scope": CLAIM_SCOPE,
        "config_contract_hash": config.contract_hash,
        "final_authorization_hash": token_payload["authorization_token_hash"],
        "authorization_protocol_hash": token_payload["authorization_protocol_hash"],
        "target_cache_content_hash": cache_summary["content_hash"],
        "target_cache_row_order_hash": cache_summary["row_order_hash"],
        "dataset_contract_hash": config.dataset_contract_hash,
        "scoring_manifest_sha256": config.scoring_manifest_sha256,
        "representation_id": config.representation_id,
        "backbone_identity_hash": backbone_identity_hash,
        "policy_arms": list(POLICY_ARMS),
        "evaluation_split": "test_previously_consumed_for_representation_adoption",
        "fresh_confirmatory_evidence": False,
        "fresh_confirmatory_status": "BLOCKED_NO_UNCONSUMED_ELIGIBLE_SPLIT",
        "routing_policy_promotion_allowed": False,
        "deployment_claim_allowed": False,
        "target_support_used": False,
        "policy_or_seed_selection_performed": False,
        "predictions_persisted_before_labels_opened": True,
        "labels_used_for_scoring_only": True,
    }
    payload["protocol_hash"] = stable_hash(payload)
    return payload


def _evaluation_plan(
    config: FrozenPolicyDownstreamConfig,
    *,
    token_payload: Mapping[str, object],
    replicates: Sequence[object],
) -> dict[str, object]:
    records = [
        {
            "policy_id": str(getattr(row, "policy_id")),
            "target_center": str(getattr(row, "target_center")),
            "training_seed": int(getattr(row, "training_seed")),
            "generation_seed": int(getattr(row, "generation_seed")),
            "replicate_id": str(getattr(row, "replicate_id")),
            "policy_lock_hash": str(getattr(row, "policy_lock_hash")),
            "policy_plan_hash": str(getattr(row, "policy_plan_hash")),
            "assignment_table_hash": str(getattr(row, "assignment_table_hash")),
            "assignment_count": len(tuple(getattr(row, "assignments"))),
            "synthetic_rows_per_class": 1024,
            "target_expert_excluded": True,
        }
        for row in replicates
    ]
    if len(records) != 243:
        raise ProtocolError("Stage-70 evaluator plan must contain 243 cells.")
    payload: dict[str, object] = {
        "schema_version": "midogpp_stage70_descriptive_evaluation_plan_v1",
        "final_authorization_hash": token_payload["authorization_token_hash"],
        "classifier_config_hash": config.classifier.config_hash,
        "policy_arms": list(POLICY_ARMS),
        "records": records,
        "prediction_cells": 243,
        "training_seeds_retained": [17, 42, 101],
        "generation_seeds_retained": [17, 42, 101],
        "seed_selection": False,
        "target_labels_opened": False,
    }
    payload["evaluation_plan_hash"] = stable_hash(payload)
    return payload


def _write_composition_index(
    root: Path,
    prediction_pass: FrozenPolicyPredictionPass,
) -> None:
    records = [
        {
            "policy_id": row.policy_id,
            "target_center": row.target_center,
            "training_seed": row.training_seed,
            "generation_seed": row.generation_seed,
            "replicate_id": row.replicate_id,
            "policy_lock_hash": row.policy_lock_hash,
            "assignment_table_hash": row.assignment_table_hash,
            "composition_manifest_hash": row.composition_manifest_hash,
            "train_content_sha256": row.train_content_sha256,
            "pre_shuffle_sha256_by_label": dict(row.pre_shuffle_sha256_by_label),
            "post_shuffle_sha256_by_label": dict(row.post_shuffle_sha256_by_label),
        }
        for row in prediction_pass.compositions
    ]
    write_json(
        root / "manifests/composition_index.json",
        {
            "schema_version": "midogpp_stage70_composition_index_v1",
            "composition_count": len(records),
            "records": records,
            "target_labels_opened": False,
        },
    )


def _require_workspace_prepared_root(root: Path) -> None:
    if not root.is_dir() or root.is_symlink():
        raise ProtocolError(
            "Stage-70 evaluator requires a workspace-prepared output directory."
        )
    symlinks = [member for member in root.rglob("*") if member.is_symlink()]
    if symlinks:
        raise ProtocolError("Stage-70 prepared evaluator root contains a symlink.")
    actual = {
        member.relative_to(root).as_posix()
        for member in root.rglob("*")
        if member.is_file()
    }
    if actual != _PREPARED_FILES:
        raise ProtocolError(
            "Stage-70 prepared evaluator root is not closed-world: "
            f"missing={sorted(_PREPARED_FILES - actual)}, "
            f"unexpected={sorted(actual - _PREPARED_FILES)}."
        )


def _is_complete(root: Path) -> bool:
    state = root / "reports/run_state.json"
    if not state.is_file():
        return False
    try:
        return _json(state).get("status") == "COMPLETE"
    except ProtocolError:
        return False


def _json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read Stage-70 evaluator JSON: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"Stage-70 evaluator JSON must be an object: {path}.")
    return payload


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Stage-70 evaluator payload lacks mapping {key!r}.")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ProtocolError(f"Cannot hash Stage-70 evaluator input: {path}.") from exc
    return digest.hexdigest()


__all__ = ("run_frozen_policy_downstream",)
