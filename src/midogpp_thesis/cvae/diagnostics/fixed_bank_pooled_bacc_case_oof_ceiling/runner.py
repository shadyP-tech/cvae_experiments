"""Thin phase orchestrator for the terminal pooled-BACC v2 diagnostic."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable, Mapping

from ...protocol import ProtocolError
from .artifact_io import read_json
from .bundle import assert_closed_world, write_content_index
from .config import FixedBankPooledBaccCaseOofCeilingConfig
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
class PriorPhaseProducts:
    statistic_surfaces: tuple[object, ...]
    priors: tuple[object, ...]


@dataclass(frozen=True)
class DecisionPhaseProducts:
    support_surfaces: tuple[object, ...]
    posteriors: tuple[object, ...]
    decisions: tuple[object, ...]
    decision_seal: object
    permutation_seal: object


@dataclass(frozen=True)
class EvaluationPhaseProducts:
    statistics: object
    evaluation: object


@dataclass(frozen=True)
class FixedBankPooledBaccRunnerDependencies:
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
    persist_loco_priors: Callable[..., object] | None = None
    fit_fold_decisions: Callable[..., object] | None = None
    persist_preevaluation: Callable[..., object] | None = None
    evaluate_sealed_decisions: Callable[..., object] | None = None
    persist_postseal: Callable[..., None] | None = None
    write_index: Callable[..., object] | None = None
    validate_bundle: Callable[..., object] | None = None
    persist_validation: Callable[..., None] | None = None
    write_state: Callable[..., None] | None = None
    phase_observer: Callable[[str], None] | None = None


def run_fixed_bank_pooled_bacc_case_oof_ceiling(
    config: FixedBankPooledBaccCaseOofCeilingConfig,
    *,
    artifact_root: str | Path | None = None,
    dependencies: FixedBankPooledBaccRunnerDependencies | None = None,
) -> Path:
    root = Path(artifact_root or config.artifact_root)
    deps = dependencies or FixedBankPooledBaccRunnerDependencies()
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
            workspace = (
                deps.validate_workspace or validate_active_diagnostic_workspace_binding
            )(config)
            provenance = (deps.validate_provenance or validate_workspace_provenance)(
                root, config
            )
            locks = (deps.load_locks or load_validated_locks)(config)
            frame = (deps.load_frame or load_label_free_test_frame)(config)
            firewall = dict(
                (deps.validate_firewall or validate_pre_gpu_firewall)(
                    config, frame, locks
                )
            )
            firewall["workspace_binding"] = workspace
            partition = (deps.build_partition or build_case_partition)(
                frame, config=config
            )
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
            preflight = (deps.preflight or run_label_free_workstation_preflight)(
                root, runtime=config.runtime
            )

            phase = "FROZEN_SOURCE_STREAMS_GPU"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "source_streams")
            canonical_source = (deps.materialize_source or materialize_sources)(
                config, locks.generation, root=root
            )
            source_for_cpu = canonical_source
            staging: dict[str, object] = {
                "attempted": True,
                "used": False,
                "status": "CANONICAL_FALLBACK",
            }
            try:
                source_for_cpu = (deps.stage_source or stage_sources_for_cpu)(
                    canonical_source, config=config, root=root
                )
            except (OSError, ProtocolError) as exc:
                staging["failure"] = f"{type(exc).__name__}: {exc}"
            else:
                staging.update(
                    {
                        "used": source_for_cpu is not canonical_source,
                        "status": (
                            "STAGED_LOCAL_CPU_CACHE"
                            if source_for_cpu is not canonical_source
                            else "CANONICAL_ALREADY_LOCAL"
                        ),
                    }
                )

            phase = "GLOBAL_LABEL_FREE_ACTION_PROBABILITY_SEAL"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "global_predictions")
            prediction_capability = (
                deps.materialize_predictions or materialize_probabilities
            )(config, source_for_cpu, frame, partition, root=root)
            seed_rows = (deps.build_seed_rows or seed_probability_rows)(
                prediction_capability
            )
            probabilities = (
                deps.aggregate_probabilities or _aggregate_probabilities
            )(seed_rows)
            (deps.persist_probabilities or persist_probability_surface)(
                root,
                prediction_capability=prediction_capability,
                seed_rows=seed_rows,
                probabilities=probabilities,
            )
            if (
                read_json(root / "manifests/sealed_probability_surface.json").get(
                    "surface_hash"
                )
                != probabilities.surface_hash
            ):
                raise ProtocolError("Persisted pooled probability surface drifted prelabel.")

            manager = (deps.build_label_manager or LabelCapabilityManager)(
                config.test_manifest_path,
                frame,
                partition,
                global_prediction_seal_hash=prediction_capability.seal_hash,
            )
            phase = "NINE_LABEL_DERIVED_LOCO_GLOBAL_AND_PAIRWISE_PRIORS"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "loco_prior_capabilities")
            prior_products = (deps.fit_loco_priors or _fit_loco_priors)(
                probabilities=probabilities,
                label_manager=manager,
                config=config,
            )
            (deps.persist_loco_priors or persist_and_validate_loco_prior_seals)(
                root,
                statistic_surfaces=prior_products.statistic_surfaces,
                priors=prior_products.priors,
            )
            for prior in prior_products.priors:
                manager.record_loco_prior_seal(
                    prior.target_center, prior.prior_hash
                )
            _observe(deps, "all_global_and_pairwise_priors_durable_before_support")

            phase = "FORTY_FIVE_SUPPORT_POSTERIORS_AND_NULL_ACTIONS"
            _write_state(deps, root, status="RUNNING", phase=phase)
            decision_products = (deps.fit_fold_decisions or _fit_fold_decisions)(
                probabilities=probabilities,
                partition=partition,
                label_manager=manager,
                priors=prior_products.priors,
                config=config,
            )
            null_payload = (
                deps.persist_preevaluation
                or persist_and_validate_preevaluation_seals
            )(
                root,
                support_surfaces=decision_products.support_surfaces,
                posteriors=decision_products.posteriors,
                decisions=decision_products.decisions,
                decision_seal=decision_products.decision_seal,
                permutation_seal=decision_products.permutation_seal,
                config_contract_hash=config.contract_hash,
            )
            null_hash = str(
                null_payload.get(
                    "permutation_decision_seal_hash",
                    null_payload.get("plan_hash", ""),
                )
            )
            manager.record_preevaluation_seals(
                decision_products.decision_seal.decision_seal_hash,
                null_hash,
                decision_count=len(decision_products.decisions),
                permutation_count=int(config.evaluation["permutation_count"]),
                null_action_count=int(null_payload.get("null_action_count", 0)),
            )
            _observe(
                deps,
                "all_45_observed_and_450000_null_actions_durable_before_evaluation",
            )

            phase = "TERMINAL_POOLED_BACC_OOF_EVALUATION"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "evaluation_labels_after_all_observed_and_null_seals")
            evaluation_labels = manager.open_oof_evaluation_labels()
            evaluation_products = (
                deps.evaluate_sealed_decisions or _evaluate_sealed_decisions
            )(
                probabilities=probabilities,
                partition=partition,
                labels=evaluation_labels,
                decision_seal=decision_products.decision_seal,
                permutation_seal=decision_products.permutation_seal,
                config=config,
            )
            capability_report = manager.access_report()
            null_action_count = int(
                getattr(decision_products.permutation_seal, "action_codes").size
            )
            leakage = leakage_report_payload(
                prediction_seal_hash=prediction_capability.seal_hash,
                prior_count=len(prior_products.priors),
                decision_count=len(decision_products.decisions),
                null_action_count=null_action_count,
                capability_report=capability_report,
            )
            runtime_summary = runtime_summary_payload(
                source_cache=canonical_source,
                prediction_capability=prediction_capability,
                local_staging={
                    **staging,
                    "workstation_preflight": dict(preflight),
                },
            )
            (deps.persist_postseal or persist_postseal_results)(
                root,
                evaluation_statistics=evaluation_products.statistics,
                evaluation=evaluation_products.evaluation,
                capability_report=capability_report,
                leakage_report=leakage,
                runtime_summary=runtime_summary,
            )

            phase = "CLOSED_WORLD_CONTENT_FIRST_VALIDATION"
            _write_state(deps, root, status="RUNNING", phase=phase)
            (deps.write_index or write_content_index)(
                root, config_contract_hash=config.contract_hash
            )
            checks = (deps.validate_bundle or _validate_bundle)(root, config=config)
            (deps.persist_validation or persist_validation_report)(root, checks)
            _write_state(deps, root, status="COMPLETE", phase="COMPLETE")
            (deps.validate_bundle or _validate_bundle)(root, config=config)
            return root
        except BaseException as exc:
            _write_state(
                deps,
                root,
                status="FAILED",
                phase=phase,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise


def _aggregate_probabilities(seed_rows: object) -> object:
    from .probability_surface import aggregate_exact_nine_probabilities

    return aggregate_exact_nine_probabilities(seed_rows)


def _fit_loco_priors(
    *, probabilities: object, label_manager: LabelCapabilityManager, config: object
) -> PriorPhaseProducts:
    from .pooled_metrics import score_loco_prior_statistics
    from .pooled_prior import PriorConfig, fit_pooled_loco_prior

    prior_config = PriorConfig(
        variance_floor=float(config.global_prior["variance_floor"]),
        confidence_multiplier=float(config.global_prior["confidence_multiplier"]),
        minimum_gain=float(config.global_prior["minimum_gain"]),
        tie_tolerance=float(config.global_prior["tie_tolerance"]),
    )
    surfaces: list[object] = []
    priors: list[object] = []
    for target in tuple(config.protocol["centers"]):
        labels = label_manager.open_loco_prior_labels(str(target))
        surface = score_loco_prior_statistics(
            probabilities, labels, target_center=str(target)
        )
        prior = fit_pooled_loco_prior(
            str(target), surface, config=prior_config
        )
        surfaces.append(surface)
        priors.append(prior)
    return PriorPhaseProducts(tuple(surfaces), tuple(priors))


def _fit_fold_decisions(
    *,
    probabilities: object,
    partition: object,
    label_manager: LabelCapabilityManager,
    priors: tuple[object, ...],
    config: object,
) -> DecisionPhaseProducts:
    from .decisions import DecisionConfig, make_fold_decision, seal_fold_decisions
    from .permutation_plan import build_permutation_decision_plan
    from .pooled_metrics import score_fold_support_statistics
    from .pooled_posterior import PosteriorConfig, fit_pooled_fold_posterior

    posterior_config = PosteriorConfig(
        variance_floor=float(config.posterior["variance_floor"]),
        confidence_multiplier=float(config.posterior["confidence_multiplier"]),
        minimum_gain=float(config.posterior["minimum_gain"]),
    )
    decision_config = DecisionConfig(
        minimum_gain=float(config.decision["minimum_gain"]),
        tie_tolerance=float(config.decision["tie_tolerance"]),
    )
    prior_by_target = {prior.target_center: prior for prior in priors}
    surfaces: list[object] = []
    support_by_fold: dict[tuple[str, int], object] = {}
    posteriors: list[object] = []
    decisions: list[object] = []
    for fold in partition.folds:
        labels = label_manager.open_fold_support_labels(
            fold.target_center, fold.fold_ordinal
        )
        prior = prior_by_target[fold.target_center]
        surface = score_fold_support_statistics(
            probabilities, labels, fold=fold, global_prior=prior
        )
        posterior = fit_pooled_fold_posterior(
            fold, surface, prior, config=posterior_config
        )
        decision = make_fold_decision(
            fold, posterior, prior, config=decision_config
        )
        label_manager.record_fold_decision(
            fold.target_center, fold.fold_ordinal, decision.decision_hash
        )
        key = (fold.target_center, fold.fold_ordinal)
        surfaces.append(surface)
        support_by_fold[key] = surface
        posteriors.append(posterior)
        decisions.append(decision)
    decision_seal = seal_fold_decisions(decisions, partition, probabilities)
    permutation = build_permutation_decision_plan(
        partition,
        probabilities,
        priors,
        support_by_fold,
        posterior_config=posterior_config,
        decision_config=decision_config,
        permutation_seed=int(config.evaluation["permutation_seed"]),
        permutation_count=int(config.evaluation["permutation_count"]),
    )
    return DecisionPhaseProducts(
        support_surfaces=tuple(surfaces),
        posteriors=tuple(posteriors),
        decisions=tuple(decisions),
        decision_seal=decision_seal,
        permutation_seal=permutation,
    )


def _evaluate_sealed_decisions(
    *,
    probabilities: object,
    partition: object,
    labels: object,
    decision_seal: object,
    permutation_seal: object,
    config: object,
) -> EvaluationPhaseProducts:
    from .pooled_evaluation import evaluate_statistics_seal
    from .pooled_metrics import score_evaluation_statistics_after_preevaluation_seals

    statistics = score_evaluation_statistics_after_preevaluation_seals(
        probabilities,
        labels,
        decision_seal=decision_seal,
        permutation_plan=permutation_seal,
    )
    evaluation = evaluate_statistics_seal(
        decision_seal,
        partition,
        statistics,
        permutation_plan=permutation_seal,
        tie_tolerance=float(config.decision["tie_tolerance"]),
        confidence_level=float(config.evaluation["confidence_level"]),
    )
    return EvaluationPhaseProducts(statistics=statistics, evaluation=evaluation)


def _validate_bundle(root: Path, **kwargs: object) -> Mapping[str, object]:
    from .validation import validate_fixed_bank_pooled_bacc_case_oof_ceiling_bundle

    return validate_fixed_bank_pooled_bacc_case_oof_ceiling_bundle(root, **kwargs)


def _observe(deps: FixedBankPooledBaccRunnerDependencies, phase: str) -> None:
    if deps.phase_observer is not None:
        deps.phase_observer(phase)


def _write_state(
    deps: FixedBankPooledBaccRunnerDependencies,
    root: Path,
    *,
    status: str,
    phase: str,
    error: str | None = None,
) -> None:
    (deps.write_state or write_run_state)(
        root, status=status, phase=phase, error=error
    )


@contextmanager
def _exclusive_run_lock(root: Path):
    path = root / ".run.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ProtocolError("Pooled-BACC v2 is already running.") from exc
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.close(descriptor)
        descriptor = -1
        yield
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        path.unlink(missing_ok=True)


def _assert_workspace_resolved_paths(config: object, *, root: Path) -> None:
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
    if unresolved or root.resolve() != Path(config.artifact_root).resolve():
        raise ProtocolError(
            f"Pooled-BACC v2 requires workspace-resolved paths; unresolved={unresolved}."
        )


def _assert_launch_files(root: Path) -> None:
    missing = [
        member
        for member in ("config.resolved.yaml", "provenance/input_artifacts.json")
        if not (root / member).is_file()
    ]
    if missing:
        raise ProtocolError(f"Pooled-BACC v2 launch files are absent: {missing}.")


__all__ = (
    "DecisionPhaseProducts",
    "EvaluationPhaseProducts",
    "FixedBankPooledBaccRunnerDependencies",
    "PriorPhaseProducts",
    "run_fixed_bank_pooled_bacc_case_oof_ceiling",
)
