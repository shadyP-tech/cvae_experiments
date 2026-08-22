"""Replay target and H/J-recomposed endpoints from persisted state DTOs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from .constants import CENTERS, ENDPOINT_METHOD_IDS, candidate_sources
from .endpoint_fitting import EndpointState, rebind_endpoint_state_priors
from .endpoint_preparation import compute_donor_priors, prepare_center
from .endpoint_reconstruction import reconstruct_case_endpoints
from .endpoint_surface_lineage import (
    ROUTE_ENDPOINT_STATES_SCHEMA_VERSION,
    expected_endpoint_surface_lineage,
)
from .hashing import canonical_hash
from .manifest_labels import read_scoped_manifest_labels
from .pseudo_endpoint_evidence import (
    PseudoEndpointEvidence,
    PseudoSourcePriorEvidence,
)
from .validation_origin import PhysicalOriginTopology
from .validation_plans import PlanPosteriorTopology
from .validation_shared import Row, fail, index_rows, mapping_field, table_rows


@dataclass(frozen=True)
class EndpointEvidenceTopology:
    target_probabilities: Mapping[tuple[str, str, str], np.ndarray]
    pseudo_probabilities: Mapping[tuple[str, str, str, str], np.ndarray]
    pseudo_evidence_hashes: Mapping[tuple[str, str, str], str]
    pseudo_prediction_hashes: Mapping[tuple[str, str, str], str]


def validate_endpoint_evidence(
    root: Path,
    *,
    config: object,
    origin: PhysicalOriginTopology,
    topology: PlanPosteriorTopology,
    capability: Row,
) -> EndpointEvidenceTopology:
    """Reconstruct all 218 target and 1,744 pseudo endpoint predictions."""

    _validate_persisted_endpoint_surface_lineage(root, origin=origin)
    states = _load_base_states(root, origin=origin, topology=topology)
    target = _validate_target_endpoints(
        root, origin=origin, topology=topology, states=states
    )
    priors = _load_pseudo_priors(
        root,
        config=config,
        origin=origin,
        capability=capability,
    )
    pseudo, evidence_hashes, prediction_hashes = _validate_pseudo_endpoints(
        root,
        origin=origin,
        topology=topology,
        states=states,
        priors=priors,
    )
    return EndpointEvidenceTopology(
        MappingProxyType(target),
        MappingProxyType(pseudo),
        MappingProxyType(evidence_hashes),
        MappingProxyType(prediction_hashes),
    )


def _load_base_states(
    root: Path,
    *,
    origin: PhysicalOriginTopology,
    topology: PlanPosteriorTopology,
) -> dict[tuple[str, str], EndpointState]:
    rows = table_rows(root, "route_endpoint_states")
    indexed = index_rows(rows, ("target_center", "held_case_id"), "endpoint states")
    if set(indexed) != set(topology.plans):
        fail("endpoint-state rectangle")
    output: dict[tuple[str, str], EndpointState] = {}
    for key, row in indexed.items():
        center, case = key
        state = EndpointState.from_payload(mapping_field(row, "state"))
        if (
            row
            != {
                "target_center": center,
                "held_case_id": case,
                "physical_surface_hash": origin.surface.surface_hash,
                "center_surface_hash": origin.surface.centers[
                    center
                ].surface_hash,
                "state": state.to_payload(),
            }
            or state.target_center != center
            or state.support_case_ids
            != tuple(topology.plans[(center, case)]["support_case_ids"])  # type: ignore[arg-type]
            or state.allowed_sources != candidate_sources(center)
            or state.model_fit_count != 2 * len(candidate_sources(center))
        ):
            fail("endpoint-state scope")
        output[key] = state
    return output


def _validate_target_endpoints(
    root: Path,
    *,
    origin: PhysicalOriginTopology,
    topology: PlanPosteriorTopology,
    states: Mapping[tuple[str, str], EndpointState],
) -> dict[tuple[str, str, str], np.ndarray]:
    manifest = read_json(root / "manifests/route_endpoint_probability_index.json")
    raw_rows = manifest.get("index_rows")
    if not isinstance(raw_rows, list) or any(
        not isinstance(row, Mapping) for row in raw_rows
    ):
        fail("target endpoint index")
    rows = tuple(raw_rows)
    indexed = index_rows(
        rows,
        ("target_center", "case_id", "method_id"),
        "target endpoint index",
    )
    expected = {
        (center, case, method)
        for center, case in topology.plans
        for method in ENDPOINT_METHOD_IDS
    }
    if set(indexed) != expected:
        fail("target endpoint rectangle")
    output: dict[tuple[str, str, str], np.ndarray] = {}
    prepared = {
        center: prepare_center(origin.surface.centers[center]) for center in CENTERS
    }
    with np.load(root / "arrays/route_endpoint_probabilities.npz", allow_pickle=False) as store:
        for center, case in topology.plans:
            prediction = reconstruct_case_endpoints(
                prepared[center], states[(center, case)], evaluation_case_id=case
            )
            expected_samples = tuple(
                topology.plans[(center, case)]["evaluation_sample_ids"]  # type: ignore[arg-type]
            )
            if prediction.sample_ids != expected_samples:
                fail("target endpoint sample lineage")
            for method in ENDPOINT_METHOD_IDS:
                row = indexed[(center, case, method)]
                key = f"{prediction.prediction_hash}__{method}"
                observed = np.ascontiguousarray(
                    np.asarray(store[key], dtype=np.float32)
                )
                expected_values = np.ascontiguousarray(
                    prediction.probabilities[method], dtype=np.float32
                )
                if (
                    row
                    != {
                        "target_center": center,
                        "case_id": case,
                        "method_id": method,
                        "sample_ids": list(prediction.sample_ids),
                        "array_key": key,
                        "prediction_hash": prediction.prediction_hash,
                        "state_hash": prediction.state_hash,
                        "physical_surface_hash": origin.surface.surface_hash,
                        "center_surface_hash": origin.surface.centers[
                            center
                        ].surface_hash,
                    }
                    or observed.tobytes(order="C")
                    != expected_values.tobytes(order="C")
                ):
                    fail("target endpoint scientific reconstruction")
                output[(center, case, method)] = observed
    return output


def _load_pseudo_priors(
    root: Path,
    *,
    config: object,
    origin: PhysicalOriginTopology,
    capability: Row,
) -> dict[tuple[str, str], PseudoSourcePriorEvidence]:
    rows = table_rows(root, "pseudo_source_priors")
    indexed = index_rows(rows, ("outer_center", "target_center"), "pseudo priors")
    expected = {
        (outer, target)
        for outer in CENTERS
        for target in CENTERS
        if target != outer
    }
    events = capability.get("events")
    if set(indexed) != expected or not isinstance(events, list):
        fail("pseudo-prior rectangle/capabilities")
    output: dict[tuple[str, str], PseudoSourcePriorEvidence] = {}
    prepared = {
        center: prepare_center(origin.surface.centers[center]) for center in CENTERS
    }
    frame_rows = tuple(getattr(origin.frame, "rows"))
    for (outer, target), row in indexed.items():
        raw_priors = row.get("priors")
        raw_capabilities = row.get("capability_hashes")
        if not isinstance(raw_priors, list) or not isinstance(raw_capabilities, list):
            fail("pseudo-prior payload")
        try:
            priors = {
                (str(source), str(direction)): float(value)
                for source, direction, value in raw_priors
            }
            capability_hashes = {
                str(source): str(value) for source, value in raw_capabilities
            }
        except (TypeError, ValueError) as exc:
            raise ProtocolError("CBPUPR pseudo-prior payload is malformed.") from exc
        evidence = PseudoSourcePriorEvidence(
            outer, target, priors, capability_hashes
        )
        if dict(row) != evidence.to_payload():
            fail("pseudo-prior hash")
        for source, digest in capability_hashes.items():
            role = f"source_prior::outer_H={outer}::J={target}::source={source}"
            matches = [
                event
                for event in events
                if isinstance(event, Mapping) and event.get("role") == role
            ]
            if len(matches) != 1 or canonical_hash(dict(matches[0])) != digest:
                fail("pseudo-prior capability lineage")
        labels_by_source = {}
        for source in candidate_sources(target):
            if source == outer:
                continue
            role = f"source_prior::outer_H={outer}::J={target}::source={source}"
            allowed = frozenset(
                (identity.center, identity.case_id, identity.sample_id)
                for identity in frame_rows
                if identity.center not in {outer, target, source}
            )
            labels = tuple(
                read_scoped_manifest_labels(
                    config,
                    origin.frame,
                    allowed_keys=allowed,
                    role=role,
                )
            )
            labels_by_source[source] = {
                center: tuple(value for value in labels if value.center == center)
                for center in CENTERS
                if center not in {outer, target, source}
            }
        recomputed = compute_donor_priors(
            prepared,
            labels_by_source,
            heldout_center=target,
            excluded_query_centers=(outer,),
            excluded_source_centers=(outer,),
        )
        if dict(recomputed) != dict(evidence.priors):
            fail("pseudo-prior scientific reconstruction")
        output[(outer, target)] = evidence
    return output


def _validate_pseudo_endpoints(
    root: Path,
    *,
    origin: PhysicalOriginTopology,
    topology: PlanPosteriorTopology,
    states: Mapping[tuple[str, str], EndpointState],
    priors: Mapping[tuple[str, str], PseudoSourcePriorEvidence],
) -> tuple[
    dict[tuple[str, str, str, str], np.ndarray],
    dict[tuple[str, str, str], str],
    dict[tuple[str, str, str], str],
]:
    manifest = read_json(
        root / "manifests/pseudo_route_endpoint_probability_index.json"
    )
    raw_rows = manifest.get("index_rows")
    if not isinstance(raw_rows, list) or any(
        not isinstance(row, Mapping) for row in raw_rows
    ):
        fail("pseudo endpoint index")
    indexed = index_rows(
        raw_rows,
        ("outer_center", "target_center", "case_id"),
        "pseudo endpoint index",
    )
    expected = {
        (outer, target, case)
        for outer in CENTERS
        for target in CENTERS
        if target != outer
        for case in topology.cases_by_center[target]
    }
    if set(indexed) != expected:
        fail("pseudo endpoint rectangle")
    prepared = {
        center: prepare_center(origin.surface.centers[center]) for center in CENTERS
    }
    probabilities: dict[tuple[str, str, str, str], np.ndarray] = {}
    evidence_hashes: dict[tuple[str, str, str], str] = {}
    prediction_hashes: dict[tuple[str, str, str], str] = {}
    with np.load(
        root / "arrays/pseudo_route_endpoint_probabilities.npz", allow_pickle=False
    ) as store:
        for outer, target, case in expected:
            prior = priors[(outer, target)]
            rebound = rebind_endpoint_state_priors(
                states[(target, case)],
                prior.priors,
                excluded_source_centers=(outer,),
            )
            prediction = reconstruct_case_endpoints(
                prepared[target], rebound, evaluation_case_id=case
            )
            evidence = PseudoEndpointEvidence(
                outer, prediction, prior.source_prior_hash
            )
            keys = {
                method: f"{prediction.prediction_hash}__H_{outer}__{method}"
                for method in ENDPOINT_METHOD_IDS
            }
            expected_row = {
                **evidence.to_payload(),
                "physical_surface_hash": origin.surface.surface_hash,
                "center_surface_hash": origin.surface.centers[
                    target
                ].surface_hash,
                "array_keys": keys,
            }
            if dict(indexed[(outer, target, case)]) != expected_row:
                fail("pseudo endpoint evidence reconstruction")
            for method in ENDPOINT_METHOD_IDS:
                observed = np.ascontiguousarray(
                    np.asarray(store[keys[method]], dtype=np.float32)
                )
                expected_values = np.ascontiguousarray(
                    prediction.probabilities[method], dtype=np.float32
                )
                if observed.tobytes(order="C") != expected_values.tobytes(order="C"):
                    fail("pseudo endpoint probability reconstruction")
                probabilities[(outer, target, case, method)] = observed
            evidence_hashes[(outer, target, case)] = evidence.evidence_hash
            prediction_hashes[(outer, target, case)] = prediction.prediction_hash
    return probabilities, evidence_hashes, prediction_hashes


def _validate_persisted_endpoint_surface_lineage(
    root: Path,
    *,
    origin: PhysicalOriginTopology,
) -> None:
    """Require unambiguous surface roles in every durable endpoint artifact."""

    try:
        expected_lineage = expected_endpoint_surface_lineage(origin.surface)
    except ProtocolError:
        fail("endpoint origin surface roles")
    physical_surface_hash = str(expected_lineage["physical_surface_hash"])
    center_surface_hashes = dict(expected_lineage["center_surface_hashes"])

    state_table = read_json(root / "tables/route_endpoint_states.json")
    state_rows = state_table.get("rows")
    if (
        state_table.get("schema_version") != ROUTE_ENDPOINT_STATES_SCHEMA_VERSION
        or not isinstance(state_rows, list)
        or state_table.get("row_count") != len(state_rows)
        or any(not isinstance(row, Mapping) for row in state_rows)
    ):
        fail("endpoint-state surface-lineage table")

    indexed_rows: list[Mapping[str, object]] = []
    for name in (
        "route_endpoint_probability_index",
        "pseudo_route_endpoint_probability_index",
    ):
        manifest = read_json(root / "manifests" / f"{name}.json")
        rows = manifest.get("index_rows")
        if (
            manifest.get("endpoint_surface_lineage") != expected_lineage
            or not isinstance(rows, list)
            or any(not isinstance(row, Mapping) for row in rows)
        ):
            fail("endpoint index surface-lineage envelope")
        indexed_rows.extend(rows)

    for row in (*state_rows, *indexed_rows):
        target = str(row.get("target_center", ""))
        if (
            target not in center_surface_hashes
            or row.get("physical_surface_hash") != physical_surface_hash
            or row.get("center_surface_hash") != center_surface_hashes[target]
        ):
            fail("endpoint persisted physical/center surface lineage")


__all__ = ("EndpointEvidenceTopology", "validate_endpoint_evidence")
