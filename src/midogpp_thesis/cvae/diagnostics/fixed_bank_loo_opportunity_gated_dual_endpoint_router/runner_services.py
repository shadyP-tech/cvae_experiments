"""Four cohesive service groups used by the thin experiment runner."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

from ....data.contract.stage70_target_evaluation.contracts import evaluation_row_id
from ...protocol import ProtocolError


class AdmissionServices:
    def admit(self, root: Path, config: object, protocol: object) -> Mapping[str, object]:
        from .actions import build_action_library
        from .inputs import (
            assert_input_fence,
            load_label_free_test_frame,
            load_validated_locks,
            validate_active_diagnostic_workspace_binding,
            validate_pre_gpu_firewall,
            validate_workspace_provenance,
        )
        from .persistence import persist_initial_surfaces

        assert_input_fence(config)
        workspace = validate_active_diagnostic_workspace_binding(config)
        provenance = validate_workspace_provenance(root, config)
        locks = load_validated_locks(config)
        frame = load_label_free_test_frame(config)
        firewall = dict(validate_pre_gpu_firewall(config, frame, locks))
        firewall["workspace_binding"] = workspace
        actions = tuple(build_action_library())
        action_seal = persist_initial_surfaces(
            root,
            config=config,
            protocol=protocol,
            provenance=provenance,
            frame=frame,
            firewall=firewall,
            actions=actions,
        )
        return {
            "workspace": workspace,
            "provenance": provenance,
            "locks": locks,
            "frame": frame,
            "firewall": firewall,
            "actions": actions,
            "action_seal": action_seal,
        }


class PhysicalRuntimeServices:
    def preflight(self, root: Path, runtime: Mapping[str, object]) -> Mapping[str, object]:
        from .runtime_adapter import run_label_free_workstation_preflight

        return run_label_free_workstation_preflight(root, runtime=runtime)

    def source_streams(self, config: object, generation: object, root: Path) -> object:
        from .runtime_adapter import materialize_sources

        return materialize_sources(config, generation, root=root)

    def probabilities(
        self, config: object, source: object, frame: object, root: Path
    ) -> Mapping[str, object]:
        from .actions import action_library_by_target
        from .persistence import persist_physical_prelabel
        from .runtime_adapter import (
            build_exact_nine_surface,
            materialize_probabilities,
            physical_partition_hash,
            probability_index_rows,
        )

        prediction = materialize_probabilities(
            config,
            source,
            frame,
            partition_hash=physical_partition_hash(frame),
            action_library=action_library_by_target(),
            root=root,
        )
        surface = build_exact_nine_surface(prediction)
        index = probability_index_rows(prediction)
        seal = persist_physical_prelabel(
            root,
            prediction=prediction,
            probability_index=index,
            probability_surface_hash=str(getattr(surface, "surface_hash")),
        )
        return {"prediction": prediction, "surface": surface, "seal": seal}


class ScienceServices:
    def label_free(
        self, root: Path, frame: object, surface: object, physical_seal_hash: str
    ) -> Mapping[str, object]:
        from .correctness_proxy import build_label_free_features
        from .persistence import persist_label_free_products
        from .split_plans import build_whole_case_loo_plans, seal_whole_case_loo_plans

        plans = build_whole_case_loo_plans(
            getattr(frame, "rows"),
            probability_surface_hash=str(getattr(surface, "surface_hash")),
        )
        plan_seal = seal_whole_case_loo_plans(
            plans, probability_surface_hash=str(getattr(surface, "surface_hash"))
        )
        features = build_label_free_features(surface)
        persisted_plan, feature_seal = persist_label_free_products(
            root,
            plans=plans,
            plan_seal=plan_seal,
            features=features,
            physical_prelabel_seal_hash=physical_seal_hash,
        )
        return {
            "plans": plans,
            "plan_seal": plan_seal,
            "features": features,
            "persisted_plan_seal": persisted_plan,
            "feature_seal": feature_seal,
        }

    def route(self, **kwargs: object) -> Mapping[str, object]:
        from .runner_science import execute_and_compose_route_products

        return execute_and_compose_route_products(**kwargs)


class FinalizationServices:
    def evaluate(self, **kwargs: object) -> Mapping[str, object]:
        from .terminal import evaluate_terminal

        return evaluate_terminal(**kwargs)

    def validate(self, root: Path, *, config: object, pending: bool) -> Mapping[str, object]:
        from .validation import validate_fixed_bank_loo_opportunity_gated_dual_endpoint_router_bundle

        return validate_fixed_bank_loo_opportunity_gated_dual_endpoint_router_bundle(
            root, config=config, allow_pending_validation=pending
        )


@dataclass(frozen=True)
class RunnerServices:
    admission: AdmissionServices = field(default_factory=AdmissionServices)
    physical: PhysicalRuntimeServices = field(default_factory=PhysicalRuntimeServices)
    science: ScienceServices = field(default_factory=ScienceServices)
    finalization: FinalizationServices = field(default_factory=FinalizationServices)
    phase_observer: Callable[[str], None] | None = None
    state_writer: Callable[..., object] | None = None


DualEndpointRunnerServices = RunnerServices


def read_scoped_manifest_labels(
    config: object,
    frame: object,
    *,
    allowed_keys: frozenset[tuple[str, str, str]],
) -> Sequence[object]:
    from .response_products import BinaryLabel

    frame_rows = tuple(frame.rows)
    universe = {(row.center, row.case_id, row.sample_id): row for row in frame_rows}
    frame_by_ordinal = {
        row.manifest_row_index: (row.center, row.case_id, row.sample_id)
        for row in frame_rows
    }
    if len(universe) != len(frame_rows) or len(frame_by_ordinal) != len(frame_rows):
        raise ProtocolError("Dual-endpoint sealed manifest identities duplicate.")
    if not allowed_keys or not set(allowed_keys) <= set(universe):
        raise ProtocolError("Dual-endpoint label grant escapes sealed rows.")
    ordered = tuple(
        key
        for row in frame_rows
        if (key := (row.center, row.case_id, row.sample_id)) in allowed_keys
    )
    requested = {key: universe[key] for key in ordered}
    requested_by_ordinal = {
        row.manifest_row_index: key for key, row in requested.items()
    }
    if len(requested_by_ordinal) != len(requested):
        raise ProtocolError("Dual-endpoint granted manifest ordinals duplicate.")
    found: dict[tuple[str, str, str], object] = {}
    seen_frame_ordinals: set[int] = set()
    manifest_path = Path(getattr(config, "test_manifest_path"))
    manifest_hash = str(getattr(config, "expected_manifest_sha256"))
    try:
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            header_line = handle.readline()
            header = tuple(next(csv.reader((header_line,))))
            required_columns = ("center", "case_id", "label")
            if any(column not in header for column in required_columns):
                raise ProtocolError("Dual-endpoint manifest header drifted.")
            positions = {column: header.index(column) for column in required_columns}
            for ordinal, raw_line in enumerate(handle):
                if ordinal in frame_by_ordinal:
                    seen_frame_ordinals.add(ordinal)
                expected_key = requested_by_ordinal.get(ordinal)
                if expected_key is None:
                    # Capability firewall: excluded rows are never CSV-decoded,
                    # so their label field is never materialized.
                    continue
                values = tuple(next(csv.reader((raw_line,))))
                if len(values) != len(header):
                    raise ProtocolError("Dual-endpoint granted manifest row drifted.")
                key = (
                    values[positions["center"]],
                    values[positions["case_id"]],
                    evaluation_row_id(manifest_hash, ordinal),
                )
                if key != expected_key or key in found:
                    raise ProtocolError("Dual-endpoint manifest order drifted.")
                found[key] = BinaryLabel(
                    *key, int(values[positions["label"]]), "scoped_loader"
                )
        # The sealed frame is a 9,928-row test projection of the full canonical
        # manifest, whose original ordinals span a larger row surface.  Require
        # the source manifest to cover every sealed frame ordinal without
        # conflating its cardinality with the projected frame cardinality.
        if seen_frame_ordinals != set(frame_by_ordinal):
            raise ProtocolError(
                "Dual-endpoint manifest does not cover the sealed frame ordinals."
            )
    except ProtocolError:
        raise
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("Cannot load dual-endpoint scoped labels.") from exc
    if set(found) != set(requested):
        raise ProtocolError("Dual-endpoint label coverage drifted.")
    return tuple(found[key] for key in ordered)


__all__ = (
    "AdmissionServices",
    "DualEndpointRunnerServices",
    "FinalizationServices",
    "PhysicalRuntimeServices",
    "RunnerServices",
    "ScienceServices",
    "read_scoped_manifest_labels",
)
