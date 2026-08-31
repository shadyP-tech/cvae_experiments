"""Two independent spawned reconstructions of durable prelabel routes."""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
import time
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from ...routing.harp_v3 import ActionKind as CoreActionKind
from ...routing.harp_v3 import PolicyConfig, route_case
from ...routing.harp_v3.serialization import decision_to_payload
from .contracts import ActionKind
from .production import (
    FEATURE_NAMES,
    _features,
    development_observations_from_artifact,
    fit_collection_from_artifact,
    target_action_sets_from_artifact,
)
from .stores import (
    read_artifact_value,
    read_label_free_outer_menu,
    read_prelabel_routes,
)


FRESH_VALIDATION_TIMEOUT_SECONDS = 300


def _expected_centers(values: Sequence[str]) -> tuple[str, ...]:
    centers = tuple(values)
    if (
        not centers
        or any(type(value) is not str or not value or value.strip() != value for value in centers)
        or len(set(centers)) != len(centers)
        or centers != tuple(sorted(centers))
    ):
        raise ProtocolError("HARP v3 validator expected-center identity is malformed.")
    return centers


def _expected_sha256(value: object, *, role: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProtocolError(f"HARP v3 validator {role} is not SHA-256.")
    return value


def _scientific_hash(manifest: Mapping[str, object], *, key: str) -> str:
    stored = _expected_sha256(manifest.get(key), role=key)
    body = {name: value for name, value in manifest.items() if name != key}
    if canonical_hash(body) != stored:
        raise ProtocolError(f"HARP v3 validator {key} self-binding drifted.")
    return stored


def _validate_identity_binding(
    manifest: Mapping[str, object],
    *,
    expected_center_ids: tuple[str, ...],
    expected_config_hash: str,
    role: str,
) -> None:
    if (
        manifest.get("config_hash") != expected_config_hash
        or tuple(manifest.get("expected_center_ids", ())) != expected_center_ids
    ):
        raise ProtocolError(f"HARP v3 {role} external identity binding drifted.")


def _validate_menu_universe(
    menus: Mapping[str, object], *, expected_center_ids: tuple[str, ...]
) -> None:
    if tuple(menus) != expected_center_ids:
        raise ProtocolError("HARP v3 validator menu target universe drifted.")
    for outer, menu in menus.items():
        if getattr(menu, "outer_target_id") != outer:
            raise ProtocolError("HARP v3 validator menu/root identity drifted.")
        by_context: dict[tuple[str, str], list[object]] = {}
        for block in getattr(menu, "blocks"):
            by_context.setdefault(
                (getattr(block, "surface_role"), getattr(block, "query_center_id")),
                [],
            ).append(block)
        expected_contexts = {
            *(("development", query) for query in expected_center_ids if query != outer),
            ("target", outer),
        }
        if set(by_context) != expected_contexts:
            raise ProtocolError("HARP v3 validator physical context universe drifted.")
        for (role, query), blocks in by_context.items():
            baseline = tuple(
                row for row in blocks if getattr(row, "action_kind") is ActionKind.B
            )
            uniform = tuple(
                row for row in blocks if getattr(row, "action_kind") is ActionKind.U
            )
            experts = tuple(
                sorted(
                    (
                        row
                        for row in blocks
                        if getattr(row, "action_kind") is ActionKind.HXE
                    ),
                    key=lambda row: getattr(row, "selected_source_id") or "",
                )
            )
            expected_sources = tuple(
                center
                for center in expected_center_ids
                if center != outer and (role == "target" or center != query)
            )
            if (
                len(baseline) != 1
                or len(uniform) != 1
                or tuple(getattr(row, "selected_source_id") for row in experts)
                != expected_sources
                or any(
                    getattr(row, "sample_ids") != getattr(baseline[0], "sample_ids")
                    or getattr(row, "case_ids") != getattr(baseline[0], "case_ids")
                    for row in (*uniform, *experts)
                )
            ):
                raise ProtocolError(
                    "HARP v3 validator requires exact B/U/all legal Hxe coverage."
                )


def _validate_development_binding(
    development_artifact: object,
    observations: Sequence[object],
    *,
    expected_center_ids: tuple[str, ...],
    expected_config_hash: str,
) -> str:
    manifest = getattr(development_artifact, "manifest")
    _validate_identity_binding(
        manifest,
        expected_center_ids=expected_center_ids,
        expected_config_hash=expected_config_hash,
        role="development surface",
    )
    surface_hash = _scientific_hash(manifest, key="surface_hash")
    if tuple(manifest.get("outer_targets", ())) != expected_center_ids:
        raise ProtocolError("HARP v3 development outer-target universe drifted.")
    outer_rows: dict[str, list[object]] = {center: [] for center in expected_center_ids}
    for row in observations:
        outer = getattr(row, "outer_target_id")
        if outer not in outer_rows:
            raise ProtocolError("HARP v3 development row escaped expected centers.")
        if outer in {
            getattr(row, "pseudo_query_id"),
            getattr(row, "candidate_source_id"),
        }:
            raise ProtocolError("HARP v3 development outer exclusion drifted.")
        outer_rows[outer].append(row)
    for outer, rows in outer_rows.items():
        expected_sources = set(expected_center_ids) - {outer}
        if (
            {getattr(row, "pseudo_query_id") for row in rows} != expected_sources
            or {
                getattr(row, "candidate_source_id")
                for row in rows
                if getattr(row, "candidate_source_id") is not None
            }
            != expected_sources
        ):
            raise ProtocolError(
                "HARP v3 development query/candidate universe is incomplete."
            )
    return surface_hash


def _validate_fit_universe(
    fits: Sequence[object], *, expected_center_ids: tuple[str, ...]
) -> None:
    if tuple(getattr(fit, "outer_target_id") for fit in fits) != expected_center_ids:
        raise ProtocolError("HARP v3 model fit-center universe drifted.")
    for fit in fits:
        outer = getattr(fit, "outer_target_id")
        expected_donors = tuple(center for center in expected_center_ids if center != outer)
        full_model = getattr(fit, "full_model")
        if (
            tuple(getattr(fit, "donor_ids")) != expected_donors
            or tuple(getattr(full_model, "training_query_ids")) != expected_donors
            or tuple(getattr(full_model, "training_candidate_ids")) != expected_donors
            or tuple(getattr(full_model, "excluded_center_ids")) != (outer,)
        ):
            raise ProtocolError("HARP v3 model outer exclusion or donor universe drifted.")
        for deleted in getattr(fit, "delete_donor_fits"):
            donor = getattr(deleted, "donor_id")
            model = getattr(deleted, "model")
            expected_training = tuple(
                center for center in expected_center_ids if center not in {outer, donor}
            )
            if (
                tuple(getattr(model, "training_query_ids")) != expected_training
                or tuple(getattr(model, "training_candidate_ids")) != expected_training
                or tuple(getattr(model, "excluded_center_ids"))
                != tuple(sorted((outer, donor)))
            ):
                raise ProtocolError("HARP v3 delete-donor fit exclusion drifted.")


def _expected_target_row_order(
    menus: Mapping[str, object], *, expected_center_ids: tuple[str, ...]
) -> tuple[tuple[str, str, str], ...]:
    rows: list[tuple[str, str, str]] = []
    for outer in expected_center_ids:
        menu = menus[outer]
        baseline = getattr(menu, "target_block")(ActionKind.B)
        case_ids = tuple(sorted(set(getattr(baseline, "case_ids"))))
        sources = tuple(center for center in expected_center_ids if center != outer)
        for case_id in case_ids:
            rows.extend(((outer, case_id, "B"), (outer, case_id, "U")))
            rows.extend((outer, case_id, f"HXE:{source}") for source in sources)
    return tuple(rows)


def _validate_all_target_actions(
    action_sets: Sequence[object],
    target_artifact: object,
    menus: Mapping[str, object],
    *,
    expected_center_ids: tuple[str, ...],
) -> None:
    manifest = getattr(target_artifact, "manifest")
    raw_rows = manifest.get("rows")
    if not isinstance(raw_rows, list):
        raise ProtocolError("HARP v3 target action row manifest is absent.")
    observed_order = tuple(
        (
            str(row.get("outer_target_id")),
            str(row.get("case_id")),
            str(row.get("action_id")),
        )
        for row in raw_rows
        if isinstance(row, Mapping)
    )
    if observed_order != _expected_target_row_order(
        menus, expected_center_ids=expected_center_ids
    ):
        raise ProtocolError("HARP v3 target action row order or coverage drifted.")
    expected_cases = {
        (outer, case_id)
        for outer in expected_center_ids
        for case_id in sorted(
            set(getattr(menus[outer].target_block(ActionKind.B), "case_ids"))
        )
    }
    observed_cases = {getattr(actions, "baseline").case_key for actions in action_sets}
    if observed_cases != expected_cases or len(action_sets) != len(expected_cases):
        raise ProtocolError("HARP v3 target case universe drifted.")
    for actions in action_sets:
        baseline_action = getattr(actions, "baseline")
        outer, case_id = baseline_action.case_key
        menu = menus[outer]
        physical_baseline = _select_rows(
            menu.target_block(ActionKind.B), baseline_action.sample_ids, case_id
        )
        physical_uniform = _select_rows(
            menu.target_block(ActionKind.U), baseline_action.sample_ids, case_id
        )
        expected_sources = tuple(
            center for center in expected_center_ids if center != outer
        )
        expert_actions = tuple(getattr(actions, "experts"))
        if (
            tuple(getattr(actions, "expected_candidate_source_ids")) != expected_sources
            or tuple(action.candidate_source_id for action in expert_actions)
            != expected_sources
        ):
            raise ProtocolError("HARP v3 target candidate universe drifted.")
        all_actions = (baseline_action, getattr(actions, "uniform"), *expert_actions)
        for action in all_actions:
            if action.action_kind is CoreActionKind.B:
                block = menu.target_block(ActionKind.B)
            elif action.action_kind is CoreActionKind.U:
                block = menu.target_block(ActionKind.U)
            else:
                block = menu.target_block(ActionKind.HXE, action.candidate_source_id)
            physical = _select_rows(block, action.sample_ids, case_id)
            action_bytes = b"".join(action.probability_bytes)
            expected_features = _features(
                physical_baseline, physical_uniform, physical
            )
            if (
                action.prediction_seal_hash != menu.menu_hash
                or action.target_query_id != outer
                or action.outer_target_id != outer
                or physical.tobytes(order="C") != action_bytes
                or action.feature_names != FEATURE_NAMES
                or action.feature_values != expected_features
            ):
                raise ProtocolError(
                    "HARP v3 target candidate vector/physical menu binding drifted."
                )


def _select_rows(block: object, sample_ids: Sequence[str], case_id: str) -> np.ndarray:
    block_samples = getattr(block, "sample_ids")
    block_cases = getattr(block, "case_ids")
    block_values = getattr(block, "probabilities")
    index = {sample: ordinal for ordinal, sample in enumerate(block_samples)}
    if len(index) != len(block_samples) or any(sample not in index for sample in sample_ids):
        raise ProtocolError("HARP v3 validator cannot align target samples.")
    ordinals = [index[sample] for sample in sample_ids]
    if any(block_cases[ordinal] != case_id for ordinal in ordinals):
        raise ProtocolError("HARP v3 validator observed cross-case routing.")
    return np.ascontiguousarray(block_values[ordinals], dtype=np.float32)


def reconstruct_prelabel_routes(
    route_root: str | Path,
    menu_roots: Mapping[str, str | Path],
    development_root: str | Path,
    model_root: str | Path,
    target_action_root: str | Path,
    *,
    validator_id: str,
    expected_center_ids: Sequence[str],
    expected_config_hash: str,
) -> dict[str, object]:
    centers = _expected_centers(expected_center_ids)
    config_hash = _expected_sha256(expected_config_hash, role="config hash")
    routes = read_prelabel_routes(Path(route_root))
    development_artifact = read_artifact_value(
        Path(development_root), role="source_development_case_surface"
    )
    model_artifact = read_artifact_value(
        Path(model_root), role="source_only_model"
    )
    target_artifact = read_artifact_value(
        Path(target_action_root), role="complete_target_case_actions"
    )
    stored_fits = fit_collection_from_artifact(model_artifact)
    observations = development_observations_from_artifact(development_artifact)
    development_hash = _validate_development_binding(
        development_artifact,
        observations,
        expected_center_ids=centers,
        expected_config_hash=config_hash,
    )
    _validate_identity_binding(
        model_artifact.manifest,
        expected_center_ids=centers,
        expected_config_hash=config_hash,
        role="model",
    )
    model_hash = _scientific_hash(model_artifact.manifest, key="model_hash")
    if model_artifact.manifest.get("development_surface_hash") != development_hash:
        raise ProtocolError("HARP v3 model/development hash binding drifted.")
    _validate_fit_universe(stored_fits, expected_center_ids=centers)
    fits = {fit.outer_target_id: fit for fit in stored_fits}
    action_sets = target_action_sets_from_artifact(target_artifact)
    _validate_identity_binding(
        target_artifact.manifest,
        expected_center_ids=centers,
        expected_config_hash=config_hash,
        role="target action",
    )
    target_action_hash = _scientific_hash(
        target_artifact.manifest, key="target_action_hash"
    )
    if target_artifact.manifest.get("model_hash") != model_hash:
        raise ProtocolError("HARP v3 target action/model hash binding drifted.")
    raw_policy = model_artifact.manifest.get("policy")
    if not isinstance(raw_policy, Mapping):
        raise ProtocolError("HARP v3 validator model lacks frozen policy thresholds.")
    policy = PolicyConfig(**dict(raw_policy))
    menus = {
        outer: read_label_free_outer_menu(Path(root))
        for outer, root in sorted(menu_roots.items())
    }
    _validate_menu_universe(menus, expected_center_ids=centers)
    expected_menu_hashes = {
        outer: menu.menu_hash for outer, menu in menus.items()
    }
    if target_artifact.manifest.get("outer_menu_hashes") != expected_menu_hashes:
        raise ProtocolError("HARP v3 target action/menu hash binding drifted.")
    _validate_all_target_actions(
        action_sets,
        target_artifact,
        menus,
        expected_center_ids=centers,
    )
    if routes.model_hash != model_hash or routes.target_action_hash != target_action_hash:
        raise ProtocolError("HARP v3 route/model/target hash binding drifted.")
    durable_by_case = {
        (case.outer_target_id, case.case_id): case for case in routes.cases
    }
    if {
        actions.baseline.case_key for actions in action_sets
    } != set(durable_by_case):
        raise ProtocolError("HARP v3 validator target actions/routes differ by case.")
    decision_payloads: list[dict[str, object]] = []
    for actions in action_sets:
        decision = route_case(
            actions,
            fits[actions.baseline.outer_target_id],
            config=policy,
        )
        decision_payloads.append(decision_to_payload(decision))
        durable = durable_by_case[actions.baseline.case_key]
        if decision_to_payload(decision) != dict(durable.decision_payload):
            raise ProtocolError("HARP v3 fresh process reconstructed another decision.")
        output = np.frombuffer(
            b"".join(decision.output_probability_bytes), dtype=np.dtype("<f4")
        )
        if output.tobytes(order="C") != durable.routed_probabilities.tobytes(order="C"):
            raise ProtocolError("HARP v3 fresh policy reconstruction changed route bytes.")
    expected_policy_hash = canonical_hash(
        {
            "schema_version": "midogpp_harp_v3_frozen_hierarchical_policy_v1",
            "model_hash": model_hash,
            "target_action_hash": target_action_hash,
            "policy": dict(raw_policy),
            "decisions": decision_payloads,
            "evaluation_labels_used": False,
        }
    )
    if routes.policy_hash != expected_policy_hash:
        raise ProtocolError("HARP v3 route policy hash binding drifted.")
    if set(menus) != {case.outer_target_id for case in routes.cases}:
        raise ProtocolError("HARP v3 validator menu/route target coverage drifted.")
    fallback_count = 0
    for case in routes.cases:
        menu = menus[case.outer_target_id]
        baseline = _select_rows(
            menu.target_block(ActionKind.B), case.sample_ids, case.case_id
        )
        uniform = _select_rows(
            menu.target_block(ActionKind.U), case.sample_ids, case.case_id
        )
        if case.selected_kind is ActionKind.B:
            selected = baseline
            fallback_count += 1
        elif case.selected_kind is ActionKind.U:
            selected = uniform
        else:
            selected = _select_rows(
                menu.target_block(ActionKind.HXE, case.selected_source_id),
                case.sample_ids,
                case.case_id,
            )
        expected = baseline if case.selected_kind is ActionKind.B else selected
        if (
            baseline.tobytes(order="C") != case.baseline_probabilities.tobytes(order="C")
            or uniform.tobytes(order="C") != case.uniform_probabilities.tobytes(order="C")
            or selected.tobytes(order="C") != case.selected_probabilities.tobytes(order="C")
            or expected.tobytes(order="C") != case.routed_probabilities.tobytes(order="C")
        ):
            raise ProtocolError("HARP v3 fresh route reconstruction changed physical bytes.")
    body = {
        "schema_version": "midogpp_harp_v3_fresh_route_reconstruction_v1",
        "validator_id": validator_id,
        "process_id": os.getpid(),
        "route_hash": routes.route_hash,
        "config_hash": config_hash,
        "expected_center_ids": list(centers),
        "menu_hashes": {outer: menu.menu_hash for outer, menu in sorted(menus.items())},
        "case_count": len(routes.cases),
        "policy_hash": routes.policy_hash,
        "model_hash": routes.model_hash,
        "target_action_hash": routes.target_action_hash,
        "development_surface_hash": development_hash,
        "model_deserialized_and_bound_to_durable_development_rows": True,
        "policy_recomputed_from_disk": True,
        "fallback_count": fallback_count,
        "case_consistent": True,
        "strict_outer_exclusion": True,
        "physical_expert_weight": 1.0,
        "exact_b_fallback_byte_identity": True,
        "evaluation_labels_opened": False,
    }
    return {**body, "validation_hash": canonical_hash(body)}


def _worker(
    route_root: str,
    menu_roots: dict[str, str],
    development_root: str,
    model_root: str,
    target_action_root: str,
    validator_id: str,
    expected_center_ids: tuple[str, ...],
    expected_config_hash: str,
    queue: object,
) -> None:
    try:
        value = reconstruct_prelabel_routes(
            route_root,
            menu_roots,
            development_root,
            model_root,
            target_action_root,
            validator_id=validator_id,
            expected_center_ids=expected_center_ids,
            expected_config_hash=expected_config_hash,
        )
        queue.put((validator_id, True, value))
    except BaseException as exc:
        queue.put((validator_id, False, (exc.__class__.__name__, str(exc))))


def run_two_fresh_validations(
    route_root: Path,
    menu_roots: Mapping[str, Path],
    development_root: Path,
    model_root: Path,
    target_action_root: Path,
    *,
    expected_center_ids: Sequence[str],
    expected_config_hash: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Spawn two parallel validators and require distinct fresh processes."""

    centers = _expected_centers(expected_center_ids)
    config_hash = _expected_sha256(expected_config_hash, role="config hash")
    context = mp.get_context("spawn")
    queue = context.Queue()
    roots = {outer: str(path.resolve()) for outer, path in sorted(menu_roots.items())}
    processes = [
        context.Process(
            target=_worker,
            args=(
                str(route_root.resolve()),
                roots,
                str(development_root.resolve()),
                str(model_root.resolve()),
                str(target_action_root.resolve()),
                validator_id,
                centers,
                config_hash,
                queue,
            ),
            name=f"harp-v3-{validator_id}",
        )
        for validator_id in ("fresh_reconstruction_A", "fresh_reconstruction_B")
    ]
    for process in processes:
        process.start()
    messages: dict[str, tuple[bool, object]] = {}
    deadline = time.monotonic() + FRESH_VALIDATION_TIMEOUT_SECONDS
    try:
        for _ in processes:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ProtocolError("HARP v3 fresh validation timed out.")
                try:
                    validator_id, success, value = queue.get(
                        timeout=min(1.0, remaining)
                    )
                    break
                except Empty:
                    failed = {
                        process.name: process.exitcode
                        for process in processes
                        if process.exitcode not in (None, 0)
                    }
                    if failed:
                        raise ProtocolError(
                            f"HARP v3 fresh validator exited before receipt: {failed}."
                        )
            messages[str(validator_id)] = (bool(success), value)
    finally:
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    if len(messages) != 2 or any(not success for success, _ in messages.values()):
        detail = {key: value for key, (success, value) in messages.items() if not success}
        raise ProtocolError(f"HARP v3 fresh validation failed: {detail}.")
    first = messages["fresh_reconstruction_A"][1]
    second = messages["fresh_reconstruction_B"][1]
    if not isinstance(first, dict) or not isinstance(second, dict):
        raise ProtocolError("HARP v3 fresh validator returned an invalid result.")
    pids = {first.get("process_id"), second.get("process_id")}
    if len(pids) != 2 or os.getpid() in pids:
        raise ProtocolError("HARP v3 validations were not independent fresh processes.")
    if (
        first.get("route_hash") != second.get("route_hash")
        or first.get("menu_hashes") != second.get("menu_hashes")
        or first.get("development_surface_hash")
        != second.get("development_surface_hash")
        or first.get("model_hash") != second.get("model_hash")
        or first.get("target_action_hash") != second.get("target_action_hash")
        or first.get("policy_hash") != second.get("policy_hash")
        or first.get("config_hash") != config_hash
        or second.get("config_hash") != config_hash
        or tuple(first.get("expected_center_ids", ())) != centers
        or tuple(second.get("expected_center_ids", ())) != centers
        or first.get("validation_hash") == second.get("validation_hash")
    ):
        raise ProtocolError("HARP v3 fresh reconstruction identities drifted.")
    return first, second


__all__ = ("reconstruct_prelabel_routes", "run_two_fresh_validations")
