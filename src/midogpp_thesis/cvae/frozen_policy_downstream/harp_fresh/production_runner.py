"""End-to-end file-backed fresh HARP Stage-70 production runner."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...runtime.harp_probability_menu import route_harp_probability_vector
from ...runtime.harp_probability_menu.hashing import (
    canonical_sha256,
    raw_array_sha256,
)
from .workstation import (
    WorkstationProbes,
    run_workstation_preflight,
)
from .bundle import (
    harp_prelabel_durable_hash,
    write_harp_fresh_content_index,
    write_harp_fresh_prelabel_bundle,
    write_harp_fresh_scored_bundle,
)
from .config import HarpFreshStage70Config
from .label_access import issue_harp_fresh_evaluation_capability
from .policy_loading import load_frozen_harp_policy
from .production_prediction import (
    materialize_harp_production_probability_menu,
    prepare_harp_production_prediction,
)
from .scoring import score_harp_fresh_routes
from .scoring_labels import open_harp_fresh_scoring_labels
from .sealing import (
    HarpFreshPrelabelSeal,
    physical_ablation_reference_preserving_vector,
)
from .target_loading import (
    CACHE_PROTOCOL_MEMBER,
    CONTENT_INDEX_MEMBER,
    RESERVATION_MEMBER,
    ROW_INDEX_MEMBER,
    load_harp_fresh_target,
)
from .validation import (
    validate_and_write_harp_fresh_completed_bundle,
    validate_harp_fresh_completed_bundle,
)
from .workspace_binding import validate_harp_fresh_workspace_binding


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _state(root: Path, status: str, *, error: str | None = None) -> None:
    payload = {
        "schema_version": "midogpp_harp_fresh_run_state_v1",
        "status": status,
        "prediction_menu_seal_hash": None,
        "route_set_hash": None,
        "prelabel_seal_hash": None,
        "result_hash": None,
        "labels_used_for_scoring_only": False,
        "policy_update_emitted": False,
        "error": error,
    }
    payload["state_hash"] = canonical_sha256(payload)
    _atomic_json(root / "reports/run_state.json", payload)


def require_harp_fresh_stage70_inputs(binding: object) -> None:
    from .workspace_binding import HarpFreshWorkspaceBinding

    if not isinstance(binding, HarpFreshWorkspaceBinding):
        raise ProtocolError("Fresh HARP input admission requires workspace binding.")
    required = (
        binding.policy_root / "manifests/policy_lock.json",
        binding.expert_bank_root / "manifests/expert_bank_index.json",
        binding.generation_lock_root / "config.resolved.yaml",
        binding.generation_lock_root / "manifests/generation_lock.json",
        binding.reservation_root / RESERVATION_MEMBER,
        binding.target_cache_root / CACHE_PROTOCOL_MEMBER,
        binding.target_cache_root / CONTENT_INDEX_MEMBER,
        binding.target_cache_root / ROW_INDEX_MEMBER,
        binding.scoring_manifest_path,
        binding.scoring_manifest_path.parent / "manifests/scoring_authorization.json",
        *(
            binding.target_cache_root
            / f"embeddings/by_center/center_{center}.npy"
            for center in CENTERS
        ),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ProtocolError(
            "Fresh HARP Stage-70 is blocked before mutation; planned fresh inputs "
            f"are absent: {missing}."
        )


def _select_prelabel_seal(
    *,
    config: HarpFreshStage70Config,
    policy: object,
    target: object,
    menu: object,
) -> HarpFreshPrelabelSeal:
    from .policy import FrozenHarpPolicy
    from .target_loading import HarpFreshLoadedTarget
    from ...runtime.harp_probability_menu import HarpPredictionMenuSeal

    if (
        not isinstance(policy, FrozenHarpPolicy)
        or not isinstance(target, HarpFreshLoadedTarget)
        or not isinstance(menu, HarpPredictionMenuSeal)
    ):
        raise ProtocolError("Fresh HARP prelabel selection received untyped inputs.")
    decisions = policy.select_all_routes(menu, target.cache)
    physical_decisions = policy.select_all_physical_routes(menu, target.cache)
    vectors = []
    physical_vectors = []
    cursor = 0
    for center in CENTERS:
        row_count = len(target.cache.frames_by_center[center].row_ids)
        vector = route_harp_probability_vector(
            menu, decisions[cursor : cursor + row_count]
        )
        vector.assert_valid()
        vectors.append(vector)
        physical_vector = route_harp_probability_vector(
            menu, physical_decisions[cursor : cursor + row_count]
        )
        physical_vector.assert_valid()
        if any(
            row.eligible and row.lambda_value != 1.0
            for row in physical_vector.decisions
        ):
            raise ProtocolError(
                "Fresh HARP physical ablation escaped lambda=1 before sealing."
            )
        physical_vectors.append(physical_vector)
        cursor += row_count
    if cursor != len(decisions):
        raise ProtocolError("Fresh HARP policy emitted surplus route decisions.")
    if cursor != len(physical_decisions):
        raise ProtocolError(
            "Fresh HARP policy emitted surplus physical-ablation decisions."
        )
    durable_hash = harp_prelabel_durable_hash(
        config,
        policy,
        target,
        menu.seal_hash,
        [vector.routed_vector_seal_hash for vector in vectors],
        [vector.routed_vector_seal_hash for vector in physical_vectors],
        [
            raw_array_sha256(
                physical_ablation_reference_preserving_vector(vector)
            )
            for vector in physical_vectors
        ],
    )
    validation_one = canonical_sha256(
        {
            "schema_version": "midogpp_harp_fresh_prelabel_validator_a_v1",
            "prediction_menu_seal_hash": menu.seal_hash,
            "routed_vector_hashes": [
                vector.routed_vector_seal_hash for vector in vectors
            ],
            "physical_ablation_routed_vector_hashes": [
                vector.routed_vector_seal_hash for vector in physical_vectors
            ],
            "physical_ablation_reference_preserving_sha256": [
                raw_array_sha256(
                    physical_ablation_reference_preserving_vector(vector)
                )
                for vector in physical_vectors
            ],
            "physical_ablation_action_universe": "Hxe_lambda_one_only",
            "exact_b_fallback_byte_identity": True,
            "labels_opened": False,
        }
    )
    rebuilt = tuple(
        route_harp_probability_vector(menu, vector.decisions) for vector in vectors
    )
    rebuilt_physical = tuple(
        route_harp_probability_vector(menu, vector.decisions)
        for vector in physical_vectors
    )
    if tuple(vector.routed_vector_seal_hash for vector in rebuilt) != tuple(
        vector.routed_vector_seal_hash for vector in vectors
    ):
        raise ProtocolError("Fresh HARP independent route reconstruction drifted.")
    if tuple(vector.routed_vector_seal_hash for vector in rebuilt_physical) != tuple(
        vector.routed_vector_seal_hash for vector in physical_vectors
    ):
        raise ProtocolError(
            "Fresh HARP physical-ablation route reconstruction drifted."
        )
    validation_two = canonical_sha256(
        {
            "schema_version": "midogpp_harp_fresh_prelabel_validator_b_v1",
            "prediction_store_hash": menu.prediction_store_hash,
            "decision_set_hashes": [vector.decision_set_hash for vector in rebuilt],
            "physical_ablation_decision_set_hashes": [
                vector.decision_set_hash for vector in rebuilt_physical
            ],
            "routed_probability_hashes": [
                vector.routed_bytes_sha256 for vector in rebuilt
            ],
            "physical_ablation_routed_probability_hashes": [
                vector.routed_bytes_sha256 for vector in rebuilt_physical
            ],
            "physical_ablation_reference_preserving_sha256": [
                raw_array_sha256(
                    physical_ablation_reference_preserving_vector(vector)
                )
                for vector in rebuilt_physical
            ],
            "labels_opened": False,
        }
    )
    return HarpFreshPrelabelSeal(
        menu=menu,
        routed_vectors=tuple(vectors),
        physical_ablation_vectors=tuple(physical_vectors),
        policy_hash=policy.metadata.policy_lock_hash,
        reservation_hash=target.cache.reservation.reservation_hash,
        target_cache_hash=target.cache.cache_hash,
        durable_bundle_hash=durable_hash,
        independent_validation_hashes=(validation_one, validation_two),
    )


def run_harp_fresh_stage70(
    config: HarpFreshStage70Config,
    *,
    workstation_probes: WorkstationProbes | None = None,
    enable_optional_local_scratch: bool = False,
) -> Path:
    """Generate, batch-predict, route, fsync, then open labels exactly once."""

    if not isinstance(config, HarpFreshStage70Config):
        raise ProtocolError("Fresh HARP production runner requires typed config.")
    binding = validate_harp_fresh_workspace_binding(config)
    require_harp_fresh_stage70_inputs(binding)
    root = binding.artifact_root
    completed_state = root / "reports/run_state.json"
    if completed_state.is_file():
        try:
            current = json.loads(completed_state.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = {}
        if isinstance(current, Mapping) and current.get("status") == "COMPLETE":
            validate_harp_fresh_completed_bundle(root, config=config)
            return root
    root.mkdir(parents=True, exist_ok=True)
    _state(root, "RUNNING")
    try:
        preflight = run_workstation_preflight(
            root,
            runtime=config.runtime,
            probes=workstation_probes,
            enable_optional_local_scratch=enable_optional_local_scratch,
        )
        _atomic_json(root / "reports/workstation_preflight.json", preflight)
        target = load_harp_fresh_target(binding)
        policy = load_frozen_harp_policy(
            binding.policy_root,
            expected_fresh_reservation_hash=target.cache.reservation.reservation_hash,
        )
        if target.policy_lock_hash != policy.metadata.policy_lock_hash:
            raise ProtocolError("Fresh HARP target cache is bound to another policy lock.")

        # GPU workers finish and join inside this call.  Only after that phase
        # completes is the disjoint four-process CPU classifier pool created.
        prediction_state = prepare_harp_production_prediction(
            config,
            binding,
            policy,
            source_cache_root=root / "checkpoints/source",
        )
        menu = materialize_harp_production_probability_menu(
            config,
            binding,
            policy,
            prediction_state,
            target.cache,
            root=root / "checkpoints/predictions",
        )
        prelabel = _select_prelabel_seal(
            config=config,
            policy=policy,
            target=target,
            menu=menu,
        )
        prelabel_content_hash = write_harp_fresh_prelabel_bundle(
            root,
            config=config,
            binding=binding,
            policy=policy,
            target=target,
            seal=prelabel,
        )

        # Sole label-opening edge: every menu cell, route, routed vector, two
        # validations, and the durable prelabel content index already exist.
        labels, authorization_hash = open_harp_fresh_scoring_labels(target, prelabel)
        capability = issue_harp_fresh_evaluation_capability(
            prelabel,
            labels_by_row_key=labels,
            reservation_hash=target.cache.reservation.reservation_hash,
            target_cache_hash=target.cache.cache_hash,
            authorization_hash=authorization_hash,
        )
        result = score_harp_fresh_routes(prelabel, capability)
        write_harp_fresh_scored_bundle(
            root,
            seal=prelabel,
            result=result,
            prelabel_content_hash=prelabel_content_hash,
        )
        write_harp_fresh_content_index(root)
        validate_and_write_harp_fresh_completed_bundle(root, config=config)
    except Exception as exc:
        _state(root, "FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    return root


__all__ = ("require_harp_fresh_stage70_inputs", "run_harp_fresh_stage70")
