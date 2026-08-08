"""Byte-reconstructive validation of a completed Stage-70 bundle."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Mapping

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .bundle_contracts import REQUIRED_FILES
from .bundle_io import (
    primary_result_payload,
    read_json,
    require_prediction_index,
    require_table,
    sha256_file,
)
from .config import (
    OUTPUT_ARTIFACT_ID,
    UtilityAlignedResidualFreshConfig,
    load_utility_aligned_residual_fresh_config,
)
from .contracts import CENTERS, EXPECTED_LOGICAL_PREDICTION_COUNT
from .inference import evaluate_sealed_predictions
from .label_access import open_scoring_labels_after_prediction_seal
from .planning import build_evaluation_plan
from .policy_loading import load_frozen_utility_aligned_policy
from .prediction_cache import load_prediction_cache
from .prediction_seal import seal_predictions, validate_prediction_seal
from .source_cache import load_source_cache
from .target_surface import load_fresh_target_surface


def validate_utility_aligned_residual_fresh_bundle(
    root: str | Path,
    *,
    config: UtilityAlignedResidualFreshConfig | None = None,
    allow_pending: bool = False,
) -> Mapping[str, object]:
    output = Path(root)
    missing = [member for member in REQUIRED_FILES if not (output / member).is_file()]
    if missing:
        raise ProtocolError(f"Utility-aligned fresh bundle is incomplete: {missing}.")
    if any((output / member).is_symlink() for member in REQUIRED_FILES):
        raise ProtocolError("Utility-aligned fresh bundle cannot contain symlink members.")
    state = read_json(output / "reports/run_state.json")
    validation = read_json(output / "reports/validation_report.json")
    seal = read_json(output / "manifests/prediction_seal.json")
    protocol = read_json(output / "manifests/protocol_manifest.json")
    result = read_json(output / "reports/primary_result.json")
    checks = validation.get("checks")
    status_ok = (
        validation.get("status") == "PENDING_RECONSTRUCTION"
        if allow_pending
        else validation.get("status") == "PASS"
        and isinstance(checks, Mapping)
        and checks.get("reconstructive_validation_passed") is True
    )
    if (
        state.get("status") != "COMPLETE"
        or state.get("policy_update_emitted") is not False
        or not status_ok
        or seal.get("logical_prediction_count") != EXPECTED_LOGICAL_PREDICTION_COUNT
        or seal.get("logical_action_coverage_complete") is not True
        or protocol.get("minimum_independent_support_cases_per_target") != 8
        or protocol.get("typed_case_bootstrap_plan_validated") is not True
        or protocol.get("target_feature_geometry_validated") is not True
        or protocol.get("bootstrap_surfaces_validated") is not True
        or protocol.get("inference_center_count") != len(CENTERS)
        or protocol.get("prior_cardinality_transfer_role") != "eligibility_only"
        or protocol.get("prior_expected_improvement_claimed") is not False
        or protocol.get("fresh_stage70_router_contrasts")
        != ["R-G_delta", "R-U", "R-B", "R-P"]
        or result.get("policy_update_emitted") is not False
    ):
        raise ProtocolError("Utility-aligned fresh bundle protocol validation failed.")
    active_config = config or load_utility_aligned_residual_fresh_config(
        output / "config.resolved.yaml"
    )
    if protocol.get("config_contract_hash") != active_config.contract_hash:
        raise ProtocolError("Utility-aligned fresh bundle/config binding drifted.")
    _validate_content_index(output, state=state, checks=checks, allow_pending=allow_pending)

    policy = load_frozen_utility_aligned_policy(active_config)
    target = load_fresh_target_surface(active_config, policy)
    plan = build_evaluation_plan(
        policy.actions_by_target,
        evaluation_row_ids_by_target=target.evaluation_row_ids_by_target,
    )
    source = load_source_cache(output / "checkpoints/source")
    prediction = load_prediction_cache(
        active_config,
        plan=plan,
        policy=policy,
        source_cache=source,
        target_surface=target,
        generation_lock_hash=source.generation_lock_hash,
        root=output / "checkpoints/predictions",
    )
    capability = seal_predictions(plan, prediction.predictions)
    rebuilt_seal = validate_prediction_seal(capability, expected_plan=plan)
    if rebuilt_seal.seal_hash != state.get("prediction_seal_hash"):
        raise ProtocolError("Utility-aligned persisted prediction seal drifted.")
    labels = open_scoring_labels_after_prediction_seal(target, capability)
    rebuilt = evaluate_sealed_predictions(capability, labels)
    for name, rows in (
        ("seed_cell_metrics", rebuilt.scored.seed_cell_metrics),
        ("ensemble_metrics", rebuilt.scored.ensemble_metrics),
        ("center_contrasts", rebuilt.center_contrasts),
        ("contrast_inference", rebuilt.contrast_inference),
        ("oracle_diagnostics", rebuilt.oracle_diagnostics),
    ):
        require_table(output / f"tables/{name}.csv", [asdict(row) for row in rows])
    if result != primary_result_payload(rebuilt):
        raise ProtocolError("Utility-aligned primary result drifted from reconstruction.")
    require_prediction_index(output / "tables/prediction_index.csv", prediction)
    label_access = read_json(output / "reports/label_access.json")
    if (
        label_access.get("prediction_seal_hash") != rebuilt_seal.seal_hash
        or label_access.get("scoring_manifest_sha256") != target.scoring_manifest_sha256
        or label_access.get("evaluation_row_count") != len(labels)
        or label_access.get("labels_opened_after_complete_global_prediction_seal") is not True
    ):
        raise ProtocolError("Utility-aligned label-access report drifted.")
    return {
        "status": "PASS",
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "prediction_seal_hash": state["prediction_seal_hash"],
        "logical_prediction_count": seal["logical_prediction_count"],
        "unique_composition_fit_count": protocol["unique_composition_fit_count"],
        "policy_update_emitted": False,
    }


def _validate_content_index(
    output: Path,
    *,
    state: Mapping[str, object],
    checks: object,
    allow_pending: bool,
) -> None:
    index = read_json(output / "manifests/content_index.json")
    unhashed = {key: value for key, value in index.items() if key != "content_hash"}
    if index.get("content_hash") != stable_hash(unhashed):
        raise ProtocolError("Utility-aligned content index hash drifted.")
    if not allow_pending and (
        not isinstance(checks, Mapping)
        or checks.get("content_index_hash") != index.get("content_hash")
        or checks.get("prediction_seal_hash") != state.get("prediction_seal_hash")
    ):
        raise ProtocolError("Utility-aligned persisted reconstructive attestation drifted.")
    records = index.get("records")
    if not isinstance(records, list):
        raise ProtocolError("Utility-aligned content index is malformed.")
    indexed: set[str] = set()
    for raw in records:
        if not isinstance(raw, Mapping):
            raise ProtocolError("Utility-aligned content record is malformed.")
        member = str(raw.get("relative_path", ""))
        path = (output / member).resolve()
        if (
            member in indexed
            or not path.is_relative_to(output.resolve())
            or not path.is_file()
            or raw.get("sha256") != sha256_file(path)
            or raw.get("size_bytes") != path.stat().st_size
        ):
            raise ProtocolError("Utility-aligned content member drifted.")
        indexed.add(member)
    excluded = {
        "manifests/content_index.json",
        "reports/run_state.json",
        "reports/validation_report.json",
    }
    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.relative_to(output).as_posix() not in excluded
    }
    if indexed != actual:
        raise ProtocolError("Utility-aligned content index coverage drifted.")


__all__ = ("validate_utility_aligned_residual_fresh_bundle",)
