"""Thin phase orchestrator for the terminal label-aware case-OOF ceiling."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable, Mapping

from ...protocol import ProtocolError
from .artifact_io import read_json
from .bundle import assert_closed_world, write_content_index
from .config import FixedBankLabelAwareCaseOofCeilingConfig
from .execution_adapter import (
    build_case_partition,
    materialize_probabilities,
    materialize_sources,
    run_label_free_workstation_preflight,
    runtime_summary_payload,
    seed_probability_rows,
    stage_sources_for_cpu,
)
from .inputs import (
    assert_input_fence,
    load_label_free_test_frame,
    load_validated_locks,
    validate_active_diagnostic_workspace_binding,
    validate_pre_gpu_firewall,
    validate_workspace_provenance,
)
from .label_capabilities import LabelCapabilityManager
from .persistence import (
    persist_and_validate_loco_prior_seals,
    persist_and_validate_preevaluation_seals,
    persist_initial_surfaces,
    persist_postseal_results,
    persist_probability_surface,
    persist_validation_report,
    write_run_state,
)
from .reports import leakage_report_payload


@dataclass(frozen=True)
class ScientificRunProducts:
    priors: tuple[object, ...]
    posteriors: tuple[object, ...]
    decisions: tuple[object, ...]
    decision_seal: object
    permutation_seal: object
    evaluation: object
    capability_report: Mapping[str, object]


@dataclass(frozen=True)
class FixedBankLabelAwareRunnerDependencies:
    validate_inputs: Callable[..., object] | None = None
    validate_workspace: Callable[..., object] | None = None
    validate_provenance: Callable[..., object] | None = None
    load_locks: Callable[..., object] | None = None
    load_frame: Callable[..., object] | None = None
    validate_firewall: Callable[..., object] | None = None
    build_partition: Callable[..., object] | None = None
    persist_initial: Callable[..., None] | None = None
    preflight: Callable[..., object] | None = None
    materialize_source: Callable[..., object] | None = None
    stage_source: Callable[..., object] | None = None
    materialize_predictions: Callable[..., object] | None = None
    build_seed_rows: Callable[..., object] | None = None
    aggregate_probabilities: Callable[..., object] | None = None
    persist_probabilities: Callable[..., None] | None = None
    build_label_manager: Callable[..., object] | None = None
    fit_loco_priors: Callable[..., object] | None = None
    fit_fold_decisions: Callable[..., object] | None = None
    evaluate_sealed_decisions: Callable[..., object] | None = None
    persist_loco_priors: Callable[..., object] | None = None
    persist_preevaluation: Callable[..., object] | None = None
    persist_postseal: Callable[..., None] | None = None
    write_index: Callable[..., object] | None = None
    validate_bundle: Callable[..., object] | None = None
    persist_validation: Callable[..., None] | None = None
    write_state: Callable[..., None] | None = None
    phase_observer: Callable[[str], None] | None = None


def run_fixed_bank_label_aware_case_oof_ceiling(
    config: FixedBankLabelAwareCaseOofCeilingConfig,
    *,
    artifact_root: str | Path | None = None,
    dependencies: FixedBankLabelAwareRunnerDependencies | None = None,
) -> Path:
    root = Path(artifact_root or config.artifact_root)
    deps = dependencies or FixedBankLabelAwareRunnerDependencies()
    _assert_workspace_resolved_paths(config, root=root)
    for relative in ("arrays", "manifests", "provenance", "reports", "tables"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    _assert_launch_files(root)
    assert_closed_world(root, allow_incomplete=True)
    with _exclusive_run_lock(root):
        state_path = root / "reports/run_state.json"
        if state_path.is_file() and read_json(state_path).get("status") == "COMPLETE":
            assert_closed_world(root, allow_incomplete=False)
            (deps.validate_bundle or _validate_bundle)(root, config=config)
            return root

        phase = "INITIALIZING"
        _write_state(deps, root, status="RUNNING", phase=phase)
        try:
            _observe(deps, "input_fence")
            (deps.validate_inputs or assert_input_fence)(config)
            workspace = (deps.validate_workspace or validate_active_diagnostic_workspace_binding)(config)
            provenance = (deps.validate_provenance or validate_workspace_provenance)(root, config)
            locks = (deps.load_locks or load_validated_locks)(config)
            frame = (deps.load_frame or load_label_free_test_frame)(config)
            firewall = dict((deps.validate_firewall or validate_pre_gpu_firewall)(config, frame, locks))
            firewall["workspace_binding"] = workspace
            partition = (deps.build_partition or build_case_partition)(frame, config=config)
            (deps.persist_initial or persist_initial_surfaces)(
                root,
                config=config,
                provenance=provenance,
                frame=frame,
                firewall=firewall,
                partition=partition,
            )

            phase = "WORKSTATION_PREFLIGHT"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "preflight")
            preflight = (deps.preflight or run_label_free_workstation_preflight)(root, runtime=config.runtime)

            phase = "FROZEN_SOURCE_STREAMS_GPU"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "source_streams")
            canonical_source = (deps.materialize_source or materialize_sources)(
                config, locks.generation, root=root
            )
            source_for_cpu = canonical_source
            staging: dict[str, object] = {"attempted": True, "used": False, "status": "CANONICAL_FALLBACK"}
            try:
                source_for_cpu = (deps.stage_source or stage_sources_for_cpu)(canonical_source, config=config, root=root)
            except (OSError, ProtocolError) as exc:
                staging["failure"] = f"{type(exc).__name__}: {exc}"
            else:
                staging.update(
                    {
                        "used": source_for_cpu is not canonical_source,
                        "status": "STAGED_LOCAL_CPU_CACHE" if source_for_cpu is not canonical_source else "CANONICAL_ALREADY_LOCAL",
                    }
                )

            phase = "GLOBAL_LABEL_FREE_ACTION_PROBABILITY_SEAL"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "global_predictions")
            prediction_capability = (deps.materialize_predictions or materialize_probabilities)(
                config, source_for_cpu, frame, partition, root=root
            )
            seed_rows = (deps.build_seed_rows or seed_probability_rows)(prediction_capability)
            probabilities = (deps.aggregate_probabilities or _aggregate_probabilities)(seed_rows)
            (deps.persist_probabilities or persist_probability_surface)(
                root,
                prediction_capability=prediction_capability,
                seed_rows=seed_rows,
                probabilities=probabilities,
            )
            persisted_surface = read_json(root / "manifests/sealed_probability_surface.json")
            if persisted_surface.get("surface_hash") != probabilities.surface_hash:
                raise ProtocolError("Persisted probability surface differs before label access.")

            phase = "CAPABILITY_SEALED_LOCO_SUPPORT_AND_OOF_AUDIT"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "label_capabilities")
            manager = (deps.build_label_manager or LabelCapabilityManager)(
                config.test_manifest_path,
                frame,
                partition,
                global_prediction_seal_hash=prediction_capability.seal_hash,
            )
            phase = "LABEL_DERIVED_LOCO_PRIORS"
            _write_state(deps, root, status="RUNNING", phase=phase)
            priors = (deps.fit_loco_priors or _fit_loco_priors)(
                probabilities=probabilities,
                label_manager=manager,
                config=config,
            )
            (deps.persist_loco_priors or persist_and_validate_loco_prior_seals)(root, priors)
            for prior in priors:
                manager.record_loco_prior_seal(prior.target_center, prior.prior_hash)
            _observe(deps, "loco_prior_seals_durable_before_support")

            phase = "FOLD_LOCAL_SUPPORT_POSTERIORS_AND_DECISIONS"
            _write_state(deps, root, status="RUNNING", phase=phase)
            decision_products = (deps.fit_fold_decisions or _fit_fold_decisions)(
                probabilities=probabilities,
                partition=partition,
                label_manager=manager,
                priors=priors,
                config=config,
            )
            permutation_payload = (
                deps.persist_preevaluation or persist_and_validate_preevaluation_seals
            )(
                root,
                posteriors=decision_products.posteriors,
                decisions=decision_products.decisions,
                decision_seal=decision_products.decision_seal,
                permutation_seal=decision_products.permutation_seal,
                config_contract_hash=config.contract_hash,
            )
            permutation_seal_hash = str(
                permutation_payload.get(
                    "permutation_decision_seal_hash",
                    permutation_payload.get("permutation_seal_hash", ""),
                )
            )
            if not permutation_seal_hash:
                permutation_seal_hash = str(permutation_payload.get("plan_hash", ""))
            manager.record_preevaluation_seals(
                decision_products.decision_seal.decision_seal_hash,
                permutation_seal_hash,
                decision_count=len(decision_products.decision_seal.decisions),
            )
            _observe(deps, "all_decision_and_permutation_seals_durable_before_evaluation")

            phase = "TERMINAL_OOF_EVALUATION"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "evaluation_labels_after_durable_seals")
            evaluation_labels = manager.open_oof_evaluation_labels()
            evaluation = (deps.evaluate_sealed_decisions or _evaluate_sealed_decisions)(
                probabilities=probabilities,
                partition=partition,
                labels=evaluation_labels,
                decision_seal=decision_products.decision_seal,
                permutation_seal=decision_products.permutation_seal,
                config=config,
            )
            products = ScientificRunProducts(
                priors=tuple(priors),
                posteriors=decision_products.posteriors,
                decisions=decision_products.decisions,
                decision_seal=decision_products.decision_seal,
                permutation_seal=decision_products.permutation_seal,
                evaluation=evaluation,
                capability_report=manager.access_report(),
            )
            leakage = leakage_report_payload(
                prediction_seal_hash=prediction_capability.seal_hash,
                prior_count=len(products.priors),
                decision_count=len(products.decisions),
                capability_report=products.capability_report,
            )
            runtime_summary = runtime_summary_payload(
                source_cache=canonical_source,
                prediction_capability=prediction_capability,
                local_staging={**staging, "workstation_preflight": dict(preflight)},
            )
            (deps.persist_postseal or persist_postseal_results)(
                root,
                evaluation=products.evaluation,
                capability_report=products.capability_report,
                leakage_report=leakage,
                runtime_summary=runtime_summary,
            )

            phase = "CLOSED_WORLD_CONTENT_FIRST_VALIDATION"
            _write_state(deps, root, status="RUNNING", phase=phase)
            (deps.write_index or write_content_index)(root, config_contract_hash=config.contract_hash)
            checks = (deps.validate_bundle or _validate_bundle)(root, config=config)
            (deps.persist_validation or persist_validation_report)(root, checks)
            _write_state(deps, root, status="COMPLETE", phase="COMPLETE")
            (deps.validate_bundle or _validate_bundle)(root, config=config)
            return root
        except BaseException as exc:
            _write_state(deps, root, status="FAILED", phase=phase, error=f"{type(exc).__name__}: {exc}")
            raise


def _aggregate_probabilities(seed_rows: object) -> object:
    from .scientific_core import aggregate_exact_nine_probabilities

    return aggregate_exact_nine_probabilities(seed_rows)


def _fit_loco_priors(
    *,
    probabilities: object,
    label_manager: LabelCapabilityManager,
    config: FixedBankLabelAwareCaseOofCeilingConfig,
) -> tuple[object, ...]:
    from .scientific_core import (
        PriorConfig,
        fit_label_derived_loco_global_prior,
        score_loco_prior_utilities,
    )

    prior_config = PriorConfig(
        prior_strength=float(config.global_prior["prior_strength"]),
        variance_floor=float(config.global_prior["variance_floor"]),
        confidence_multiplier=float(config.global_prior["confidence_multiplier"]),
        minimum_gain=float(config.global_prior["minimum_gain"]),
    )
    priors: list[object] = []
    for target in tuple(config.protocol["centers"]):
        labels = label_manager.open_loco_prior_labels(str(target))
        utilities = score_loco_prior_utilities(probabilities, labels, target_center=str(target))
        priors.append(fit_label_derived_loco_global_prior(str(target), utilities, config=prior_config))
    return tuple(priors)


@dataclass(frozen=True)
class DecisionPhaseProducts:
    posteriors: tuple[object, ...]
    decisions: tuple[object, ...]
    decision_seal: object
    permutation_seal: object


def _fit_fold_decisions(
    *,
    probabilities: object,
    partition: object,
    label_manager: LabelCapabilityManager,
    priors: tuple[object, ...],
    config: FixedBankLabelAwareCaseOofCeilingConfig,
) -> DecisionPhaseProducts:
    from .scientific_core import (
        DecisionConfig,
        PosteriorConfig,
        build_permutation_decision_plan,
        fit_fold_local_posterior,
        make_fold_decision,
        score_fold_support_utilities,
        seal_fold_decisions,
    )

    posterior_config = PosteriorConfig(
        prior_strength=float(config.posterior["prior_strength"]),
        variance_floor=float(config.posterior["variance_floor"]),
        confidence_multiplier=float(config.posterior["confidence_multiplier"]),
        minimum_gain=float(config.posterior["minimum_gain"]),
    )
    decision_config = DecisionConfig(
        minimum_gain=float(config.decision["minimum_gain"]),
        tie_tolerance=float(config.decision["tie_tolerance"]),
    )
    prior_by_target = {prior.target_center: prior for prior in priors}

    posteriors: list[object] = []
    decisions: list[object] = []
    support_surfaces: dict[tuple[str, int], object] = {}
    for fold in partition.folds:
        labels = label_manager.open_fold_support_labels(fold.target_center, fold.fold_ordinal)
        prior = prior_by_target[fold.target_center]
        utilities = score_fold_support_utilities(probabilities, labels, fold=fold, global_prior=prior)
        posterior = fit_fold_local_posterior(fold, utilities, prior, config=posterior_config)
        decision = make_fold_decision(fold, posterior, prior, config=decision_config)
        label_manager.record_fold_decision(fold.target_center, fold.fold_ordinal, decision.decision_hash)
        posteriors.append(posterior)
        decisions.append(decision)
        support_surfaces[(fold.target_center, fold.fold_ordinal)] = utilities
    decision_seal = seal_fold_decisions(decisions, partition, probabilities)
    permutation_seal = build_permutation_decision_plan(
        partition,
        probabilities,
        tuple(priors),
        support_surfaces,
        posterior_config=posterior_config,
        decision_config=decision_config,
        permutation_seed=int(config.evaluation["permutation_seed"]),
        permutation_count=int(config.evaluation["permutation_count"]),
    )
    return DecisionPhaseProducts(
        posteriors=tuple(posteriors),
        decisions=tuple(decisions),
        decision_seal=decision_seal,
        permutation_seal=permutation_seal,
    )


def _evaluate_sealed_decisions(
    *,
    probabilities: object,
    partition: object,
    labels: object,
    decision_seal: object,
    permutation_seal: object,
    config: FixedBankLabelAwareCaseOofCeilingConfig,
) -> object:
    from .scientific_core import evaluate_decision_seal

    return evaluate_decision_seal(
        decision_seal,
        partition,
        probabilities,
        labels,
        permutation_plan=permutation_seal,
        tie_tolerance=float(config.decision["tie_tolerance"]),
        confidence_level=float(config.evaluation["confidence_level"]),
    )


def _validate_bundle(root: Path, **kwargs: object) -> Mapping[str, object]:
    from .validation import validate_fixed_bank_label_aware_case_oof_ceiling_bundle

    return validate_fixed_bank_label_aware_case_oof_ceiling_bundle(root, **kwargs)


def _observe(deps: FixedBankLabelAwareRunnerDependencies, phase: str) -> None:
    if deps.phase_observer is not None:
        deps.phase_observer(phase)


def _write_state(
    deps: FixedBankLabelAwareRunnerDependencies,
    root: Path,
    *,
    status: str,
    phase: str,
    error: str | None = None,
) -> None:
    (deps.write_state or write_run_state)(root, status=status, phase=phase, error=error)


@contextmanager
def _exclusive_run_lock(root: Path):
    path = root / ".run.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ProtocolError("Label-aware case-OOF ceiling is already running.") from exc
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.close(descriptor)
        descriptor = -1
        yield
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        path.unlink(missing_ok=True)


def _assert_workspace_resolved_paths(
    config: FixedBankLabelAwareCaseOofCeilingConfig, *, root: Path
) -> None:
    paths = {
        "artifact root": root,
        "configured artifact root": config.artifact_root,
        "expert-bank root": config.expert_bank_root,
        "generation-lock root": config.generation_lock_root,
        "test-cache root": config.test_cache_root,
        "test manifest": config.test_manifest_path,
        "test-consumption ledger": config.test_consumption_ledger_path,
        "ledger amendment": config.ledger_amendment_path,
    }
    unresolved = [role for role, path in paths.items() if not Path(path).is_absolute()]
    if unresolved or root.resolve() != config.artifact_root.resolve():
        raise ProtocolError(
            "Label-aware ceiling requires workspace-resolved paths; "
            f"unresolved={unresolved}."
        )


def _assert_launch_files(root: Path) -> None:
    missing = [
        member
        for member in ("config.resolved.yaml", "provenance/input_artifacts.json")
        if not (root / member).is_file()
    ]
    if missing:
        raise ProtocolError(f"Label-aware launch files are absent: {missing}.")


__all__ = (
    "FixedBankLabelAwareRunnerDependencies",
    "DecisionPhaseProducts",
    "ScientificRunProducts",
    "run_fixed_bank_label_aware_case_oof_ceiling",
)
