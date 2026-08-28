"""Typed production-phase adapters used by the thin SCEPTRE v4 runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from midogpp_thesis.cvae.protocol import ProtocolError

from ...fixed_bank_sceptre_router.partitions import CaseIdentity, FOLD_COUNT
from ..config import SceptreV4Config
from ..label_broker import RoleLabelBroker
from ..outcome_builder import build_role_evidence
from ..phase_manager import CandidateSetPhaseManager
from .persistence import prediction_store_payload, source_store_payload
from .scratch import SOURCE_DIRECTORY, ScratchLease
from .services import ProductionServices


@dataclass(frozen=True, slots=True)
class PhysicalSurfaces:
    source: object
    source_binding: Mapping[str, object]
    prediction: object
    prediction_binding: Mapping[str, object]
    prediction_member_hashes: Mapping[str, str]
    prediction_store_hash: str


def freeze_development(
    config: SceptreV4Config,
    validated: object,
    *,
    services: ProductionServices,
) -> tuple[object, object]:
    frame = getattr(validated, "frame")
    partition = services.build_partition(
        tuple(
            CaseIdentity(row.center, row.case_id, row.sample_id)
            for row in frame.rows
        ),
        expected_total_case_count=frame.case_count,
    )
    development_surface, source_inner_predictions = services.load_development(
        config.source_inner_root,
        receipt=getattr(validated, "source_inner"),
    )
    development = services.fit_development(
        development_surface,
        source_inner_predictions,
        generation_lock=getattr(validated, "generation_lock"),
        partition=partition,
    )
    return partition, development


def materialize_physical_surfaces(
    config: SceptreV4Config,
    validated: object,
    *,
    root: Path,
    scratch: ScratchLease,
    attempt_id: str,
    services: ProductionServices,
    source_observer: Callable[[str], None],
) -> PhysicalSurfaces:
    source_root = scratch.root / SOURCE_DIRECTORY
    source_root.mkdir(parents=True, exist_ok=False)
    source = services.materialize_sources(
        config,
        getattr(validated, "generation_lock"),
        root=source_root,
        expert_bank_root=config.expert_bank_root,
        attempt_id=attempt_id,
    )
    source_binding = source_store_payload(source)
    source_observer(str(source_binding["source_store_hash"]))

    prediction_root = root / "prediction_store"
    if prediction_root.is_symlink() or (
        prediction_root.exists() and not prediction_root.is_dir()
    ):
        raise ProtocolError("SCEPTRE v4 prepared prediction root is unsafe.")
    prediction_root.mkdir(parents=True, exist_ok=True)
    prediction = services.materialize_predictions(
        config,
        source,
        getattr(validated, "frame"),
        root=prediction_root,
        attempt_id=attempt_id,
    )
    prediction_binding, prediction_hashes = prediction_store_payload(root, prediction)
    return PhysicalSurfaces(
        source=source,
        source_binding=source_binding,
        prediction=prediction,
        prediction_binding=prediction_binding,
        prediction_member_hashes=prediction_hashes,
        prediction_store_hash=str(getattr(prediction, "receipt_hash")),
    )


def form_route_policy(
    config: SceptreV4Config,
    validated: object,
    partition: object,
    development: object,
    physical: PhysicalSurfaces,
    *,
    authorization_lease_hash: str,
    services: ProductionServices,
    phase_observer: Callable[[str, str], None],
) -> tuple[CandidateSetPhaseManager, RoleLabelBroker, object]:
    manager = CandidateSetPhaseManager(partition, development.context)
    broker = RoleLabelBroker(
        manager=manager,
        partition=partition,
        frame=getattr(validated, "frame"),
        manifest_path=config.test_manifest_path,
        expected_manifest_sha256=config.expected_manifest_sha256,
        prediction_store_hash=physical.prediction_store_hash,
        authorization_lease_hash=authorization_lease_hash,
    )
    prediction = physical.prediction
    phases = services.route(
        development,
        partition=partition,
        manager=manager,
        broker=broker,
        candidate_probabilities=prediction.candidate_probabilities,
        exact_b_probabilities=prediction.exact_b_probabilities,
        candidate_source_order=prediction.geometry.centers,
        prediction_store_hash=physical.prediction_store_hash,
        phase_observer=phase_observer,
    )
    return manager, broker, phases


def evaluate_terminal_policy(
    manager: CandidateSetPhaseManager,
    broker: RoleLabelBroker,
    partition: object,
    development: object,
    phases: object,
    physical: PhysicalSurfaces,
    durable_attestation: object,
    *,
    services: ProductionServices,
) -> object:
    terminal = manager.begin_terminal_evaluation(durable_attestation)
    broker.activate_terminal(terminal)
    prediction = physical.prediction
    surfaces = []
    for target in CENTERS:
        for fold_ordinal in range(FOLD_COUNT):
            fold = partition.fold(target, fold_ordinal)
            scoped = broker.open_evaluation(target, fold_ordinal, terminal)
            model = development.context.model_for_target(target)
            evidence = build_role_evidence(
                scoped,
                fold=fold,
                partition_hash=partition.partition_hash,
                candidate_probabilities=prediction.candidate_probabilities,
                exact_b_probabilities=prediction.exact_b_probabilities,
                candidate_source_order=prediction.geometry.centers,
                prediction_store_hash=physical.prediction_store_hash,
                candidate_menu_hash=model.candidate_menu_hash,
                exact_b_control_receipt_hash=model.exact_b_control_receipt_hash,
                phase_capability=terminal,
            )
            surfaces.append(evidence.surface)
            del evidence, scoped
    return services.evaluate_terminal(
        phases.route_policy,
        surfaces,
        routing_context=development.context,
        prediction_store_hash=physical.prediction_store_hash,
        terminal_capability_hash=terminal.capability_hash,
    )


__all__ = (
    "PhysicalSurfaces",
    "evaluate_terminal_policy",
    "form_route_policy",
    "freeze_development",
    "materialize_physical_surfaces",
)
