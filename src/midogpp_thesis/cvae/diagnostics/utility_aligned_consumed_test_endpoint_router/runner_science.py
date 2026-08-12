"""Runtime-to-science adapters and deterministic persistence projections."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import canonical_sha256
from .actions import build_frozen_target_action_library
from .contracts import (
    BASE_ACTION_ID, CENTERS, GLOBAL_ACTION_ID, PERMUTATION_ACTION_ID,
    ROUTED_ACTION_ID, UNIFORM_ACTION_ID, candidate_sources,
    expected_target_action_ids, h_x_e_action_id,
)
from .endpoint_adapter import (
    build_development_endpoint_response, validate_development_endpoint_responses,
)
from .features import (
    build_source_inner_feature_surface_set,
    build_source_inner_feature_surfaces, build_target_case_bootstrap_plan,
    build_target_feature_production,
)
from .inference import (
    build_terminal_inference_products, score_terminal_target_action,
    validate_terminal_endpoint_scores,
)
from .input_contracts import row_identity_hash
from .label_capabilities import EndpointRouterLabelCapabilityManager
from .models import fit_endpoint_router_model_set
from .policy import build_target_policy_set
from .products import (
    build_core_scientific_products, build_prelabel_scientific_products,
)
from .reports import (
    development_label_access_report_payload, leakage_report_payload,
    publication_decision_payload, runtime_summary_payload,
)
from .seals import seal_target_predictions


@dataclass(frozen=True)
class PrelabelRuntimeProducts:
    manager: EndpointRouterLabelCapabilityManager
    responses: object
    source_features: object
    models: object
    target_features_by_target: Mapping[str, object]
    policies: object
    actions: object
    global_prelabel_seal_hash: str
    development_persistence: Mapping[str, object]

    def seal_target(self, store: object, *, root: Path) -> object:
        for target in CENTERS:
            self.manager.record_target_policy_plan(
                target, self.policies.by_target[target].policy_hash
            )
        capability = seal_target_predictions(
            store,
            root=root,
            target_policy_plan_hashes_by_center={
                target: self.policies.by_target[target].policy_hash
                for target in CENTERS
            },
            target_policy_plan_set_hash=self.policies.policy_set_hash,
            frozen_action_set_hash=self.actions.action_library_hash,
            global_prelabel_seal_hash=self.global_prelabel_seal_hash,
        )
        self.manager.record_global_target_seal(
            capability,
            global_prelabel_seal_hash=self.global_prelabel_seal_hash,
        )
        return capability

    def model_plan_persistence(self, target_capability: object) -> Mapping[str, object]:
        model_rows = [_flat(self.models.by_target[target].to_payload()) for target in CENTERS]
        target_rows = [
            _flat({**row.to_payload(), "row_hash": row.row_hash})
            for target in CENTERS
            for row in self.target_features_by_target[target].point_surface.rows
        ]
        plan_rows = [
            _flat({
                **self.policies.by_target[target].to_payload(),
                "selected_action_role": self.policies.by_target[target].selected_action_id,
            })
            for target in CENTERS
        ]
        action_rows = [
            _flat(action.to_payload())
            for target in CENTERS
            for action in self.actions.by_target[target].actions
        ]
        transfers = {
            target: self.models.by_target[target].cardinality_transfer.to_payload()
            for target in CENTERS
        }
        transfer_unhashed = {
            "schema_version": "midogpp_consumed_test_cardinality_transfer_seal_v1",
            "transfer_by_target": transfers,
            "transfer_hashes_by_target": {
                target: self.models.by_target[target].cardinality_transfer.transfer_hash
                for target in CENTERS
            },
            "target_count": len(CENTERS),
            "same_outer_H_evaluation_labels_used_for_model_H": False,
        }
        global_payload = {
            "schema_version": "midogpp_consumed_test_global_prelabel_seal_v1",
            "status": "SEALED_ALL_TARGET_PLANS_ACTIONS_AND_PROBABILITIES",
            "global_prelabel_seal_hash": self.global_prelabel_seal_hash,
            "target_plan_count": len(CENTERS),
            "target_policy_plan_set_hash": self.policies.policy_set_hash,
            "frozen_action_library_hash": self.actions.action_library_hash,
            "global_target_prediction_seal_hash": target_capability.seal_hash,
            "target_prediction_store_hash": target_capability.store.store_hash,
            "support_labels_used": False,
            "same_outer_H_evaluation_labels_used_for_plan_H": False,
            "all_target_actions_static": True,
        }
        return MappingProxyType({
            "model_index": self.models.to_payload(),
            "model_rows": model_rows,
            "cardinality_transfer_seal": {
                **transfer_unhashed,
                "cardinality_transfer_seal_hash": canonical_sha256(transfer_unhashed),
            },
            "target_feature_rows": target_rows,
            "target_policy_plans": {
                **self.policies.to_payload(),
                "target_plan_count": len(CENTERS),
                "target_features_by_target": {
                    target: self.target_features_by_target[target].to_payload()
                    for target in CENTERS
                },
                "target_feature_productions_by_target": {
                    target: self.target_features_by_target[
                        target
                    ].production.to_payload()
                    for target in CENTERS
                },
                "target_point_row_hashes_by_target": {
                    target: [
                        row.row_hash
                        for row in self.target_features_by_target[
                            target
                        ].point_surface.rows
                    ]
                    for target in CENTERS
                },
                "core_policies_by_target": {
                    target: self.policies.by_target[target].core_policy.to_payload()
                    for target in CENTERS
                },
            },
            "target_policy_plan_rows": plan_rows,
            "frozen_actions": self.actions.to_payload(),
            "frozen_action_rows": action_rows,
            "global_prelabel_seal": global_payload,
        })


@dataclass(frozen=True)
class TerminalRuntimeProducts:
    core: object
    persistence: Mapping[str, object]


def run_prelabel_science(
    *, config: object, root: Path, partitions: object, development: object,
    seed_features: object, shifts: object, target_store: object, frame: object,
) -> PrelabelRuntimeProducts:
    """Open only q!=H labels, then fit and freeze all nine target plans."""

    manager = EndpointRouterLabelCapabilityManager(
        config.test_manifest_path,
        frame,
        partitions,
        expected_manifest_sha256=config.expected_manifest_sha256,
        development_capability=development,
    )
    responses = []
    for outer in CENTERS:
        scoped = manager.open_development_labels(outer)
        for query in candidate_sources(outer):
            for source in candidate_sources(outer):
                if source == query:
                    continue
                responses.append(build_development_endpoint_response(
                    outer_target_id=outer,
                    query_id=query,
                    candidate_source=source,
                    base_vectors=development.store.vectors(
                        outer_target=outer, query_center=query,
                        action_id=BASE_ACTION_ID, role="evaluation",
                    ),
                    tail_vectors=development.store.vectors(
                        outer_target=outer, query_center=query,
                        action_id=h_x_e_action_id(source), role="evaluation",
                    ),
                    development_query_evaluation_labels=scoped.labels_for(query),
                    support_partition_hash=row_identity_hash(
                        partitions.support_rows_by_center[query]
                    ),
                    evaluation_partition_hash=row_identity_hash(
                        partitions.evaluation_rows_by_center[query]
                    ),
                    development_prediction_seal_hash=development.seal_hash,
                ))
    response_set = validate_development_endpoint_responses(
        responses, development_prediction_seal_hash=development.seal_hash
    )
    feature_input_hash = canonical_sha256({
        "seed_feature_production_hash": seed_features.production_hash,
        "support_shift_production_hash": shifts.production_hash,
        "development_prediction_seal_hash": development.seal_hash,
    })
    source_features = build_source_inner_feature_surface_set({
        outer: build_source_inner_feature_surfaces(
            tuple(row for row in seed_features.inner_rows if row.outer_target_id == outer),
            support_action_shift_by_candidate={
                key: value for key, value in shifts.source_inner_by_candidate.items()
                if key[0] == outer
            },
            outer_target_id=outer,
            feature_input_seal_hash=feature_input_hash,
        )
        for outer in CENTERS
    })
    models = fit_endpoint_router_model_set(source_features, response_set)
    target_features = {}
    for target in CENTERS:
        bootstrap = build_target_case_bootstrap_plan(
            target_id=target,
            support_case_ids=partitions.by_center[target].support_case_ids,
        )
        target_features[target] = build_target_feature_production(
            tuple(row for row in seed_features.target_rows if row.outer_target_id == target),
            tuple(row for row in shifts.target_case_rows if row.target_id == target),
            source_features=source_features.by_target[target],
            case_bootstrap_plan=bootstrap,
            support_partition_lock_hash=partitions.lock_hash,
            target_feature_seal_hash=canonical_sha256({
                "target": target, "bootstrap_plan_hash": bootstrap.plan_hash,
                "runtime_feature_input_hash": feature_input_hash,
            }),
        )
    policy_input_hash = canonical_sha256({
        "model_set_hash": models.model_set_hash,
        "target_feature_hashes": {
            target: target_features[target].feature_hash for target in CENTERS
        },
        "target_prediction_store_hash": target_store.store_hash,
    })
    policies = build_target_policy_set(
        models, target_features, target_policy_seal_hash=policy_input_hash
    )
    actions = build_frozen_target_action_library(policies)
    global_hash = canonical_sha256({
        "schema_version": "midogpp_consumed_test_global_prelabel_basis_v1",
        "target_prediction_store_hash": target_store.store_hash,
        "policy_set_hash": policies.policy_set_hash,
        "action_library_hash": actions.action_library_hash,
        "target_plan_count": len(CENTERS),
        "support_labels_used": False,
        "same_outer_H_evaluation_labels_used_for_plan_H": False,
    })
    build_prelabel_scientific_products(
        partitions=partitions, development_responses=response_set,
        source_features=source_features, models=models,
        target_features_by_target=target_features, policies=policies,
        actions=actions, global_prelabel_seal_hash=global_hash,
    )
    development_persistence = MappingProxyType({
        "response_rows": [
            _flat({**row.to_payload(), "row_hash": row.row_hash})
            for row in response_set.rows
        ],
        "source_inner_feature_rows": [
            _flat({**row.to_payload(), "row_hash": row.row_hash})
            for target in CENTERS for row in source_features.by_target[target].m1.rows
        ],
        "response_seal": response_set.to_payload(),
        "feature_surface_set": {
            **source_features.to_payload(),
            "source_surfaces_by_target": {
                target: source_features.by_target[target].to_payload()
                for target in CENTERS
            },
            "global_source_controls_by_target": {
                target: source_features.by_target[
                    target
                ].global_source_control.to_payload()
                for target in CENTERS
            },
            "m1_row_hashes_by_target": {
                target: [
                    row.row_hash
                    for row in source_features.by_target[target].m1.rows
                ]
                for target in CENTERS
            },
        },
        "development_label_access_report": development_label_access_report_payload(
            development_prediction_seal_hash=development.seal_hash,
            outer_target_ids=CENTERS,
        ),
    })
    return PrelabelRuntimeProducts(
        manager=manager, responses=response_set, source_features=source_features,
        models=models, target_features_by_target=MappingProxyType(target_features),
        policies=policies, actions=actions,
        global_prelabel_seal_hash=global_hash,
        development_persistence=development_persistence,
    )


def run_terminal_science(
    *, config: object, root: Path, partitions: object, development: object,
    target_capability: object, prelabel: PrelabelRuntimeProducts,
    preflight: Mapping[str, object], source_cache_staging: Mapping[str, object],
) -> TerminalRuntimeProducts:
    labels = prelabel.manager.open_terminal_evaluation_labels()
    scores = []
    for target in CENTERS:
        for action_id in expected_target_action_ids(target):
            action = prelabel.actions.by_target[target].by_action_id[action_id]
            physical = _physical_action_id(action)
            scores.append(score_terminal_target_action(
                action=action,
                vectors=target_capability.store.vectors(
                    outer_target=target, query_center=target,
                    action_id=physical, role="evaluation",
                ),
                terminal_evaluation_labels=labels.labels_for(target),
                support_partition_lock_hash=partitions.lock_hash,
                evaluation_partition_hash=row_identity_hash(
                    partitions.evaluation_rows_by_center[target]
                ),
                global_target_prediction_seal_hash=target_capability.seal_hash,
                global_prelabel_seal_hash=prelabel.global_prelabel_seal_hash,
                evaluation_case_count=len(partitions.by_center[target].evaluation_case_ids),
                global_target_prediction_seal_verified=True,
            ))
    score_set = validate_terminal_endpoint_scores(scores, prelabel.actions)
    inference = build_terminal_inference_products(score_set, prelabel.policies)
    prelabel_product = build_prelabel_scientific_products(
        partitions=partitions, development_responses=prelabel.responses,
        source_features=prelabel.source_features, models=prelabel.models,
        target_features_by_target=prelabel.target_features_by_target,
        policies=prelabel.policies, actions=prelabel.actions,
        global_prelabel_seal_hash=prelabel.global_prelabel_seal_hash,
    )
    core = build_core_scientific_products(
        prelabel=prelabel_product, terminal_scores=score_set,
        terminal_inference=inference,
    )
    sealed = {
        "schema_version": "midogpp_consumed_test_sealed_terminal_evaluation_v1",
        "status": "COMPLETE_TERMINAL_SCORING",
        "target_prediction_seal_hash": target_capability.seal_hash,
        "global_prelabel_seal_hash": prelabel.global_prelabel_seal_hash,
        "terminal_score_set_hash": score_set.score_set_hash,
        "terminal_inference_hash": inference.inference_hash,
        "evaluation_center_count": len(CENTERS),
        "same_outer_H_evaluation_labels_used_for_plan_H": False,
        "support_labels_used": False,
        "terminal_only_no_plan_or_policy_update": True,
    }
    counts = {
        "source_stream_count": 81,
        "development_prediction_cell_count": len(development.store.cells),
        "development_response_count": len(prelabel.responses.rows),
        "target_prediction_cell_count": len(target_capability.store.cells),
        "target_reported_action_count": sum(
            len(prelabel.actions.by_target[target].actions) for target in CENTERS
        ),
        "terminal_score_count": len(score_set.rows),
    }
    runtime = runtime_summary_payload(
        preflight, counts=counts, source_cache_staging=source_cache_staging
    )
    persistence = MappingProxyType({
        "endpoint_rows": [
            _flat({**row.to_payload(), "score_hash": row.score_hash})
            for row in score_set.rows
        ],
        "contrast_rows": [_flat(row.to_payload()) for row in inference.center_contrasts],
        "aggregate_contrast_rows": [
            _flat(row.to_payload()) for row in inference.aggregate_contrasts
        ],
        "oracle_rows": [
            _flat(row.to_payload()) for row in inference.oracle_rank_diagnostics
        ],
        "sealed_terminal_evaluation": sealed,
        "label_capability_report": dict(prelabel.manager.access_report()),
        "leakage_report": leakage_report_payload(
            support_partition_lock_hash=partitions.lock_hash,
            development_prediction_seal_hash=development.seal_hash,
            target_prediction_seal_hash=target_capability.seal_hash,
            global_prelabel_seal_hash=prelabel.global_prelabel_seal_hash,
            model_set_hash=prelabel.models.model_set_hash,
            policy_set_hash=prelabel.policies.policy_set_hash,
            action_library_hash=prelabel.actions.action_library_hash,
        ),
        "runtime_summary": runtime,
        "publication_decision": publication_decision_payload(inference.to_payload()),
    })
    return TerminalRuntimeProducts(core=core, persistence=persistence)


def _physical_action_id(action: object) -> str:
    if action.action_id in {BASE_ACTION_ID, UNIFORM_ACTION_ID}:
        return action.action_id
    if action.action_id.startswith("Hxe::"):
        return action.action_id
    if action.action_id not in {
        GLOBAL_ACTION_ID, ROUTED_ACTION_ID, PERMUTATION_ACTION_ID
    }:
        raise ProtocolError("Target logical action cannot resolve to a physical cell.")
    return (
        BASE_ACTION_ID if action.core_action is None
        else h_x_e_action_id(action.selected_source)
    )


def _flat(payload: Mapping[str, object]) -> dict[str, object]:
    """Render one typed payload as scalar-only, stable CSV fields."""

    result: dict[str, object] = {}
    for key, value in payload.items():
        if isinstance(value, (Mapping, list, tuple)):
            result[str(key)] = json.dumps(value, sort_keys=True, separators=(",", ":"))
        else:
            result[str(key)] = value
    return result


__all__ = (
    "PrelabelRuntimeProducts", "TerminalRuntimeProducts", "run_prelabel_science",
    "run_terminal_science",
)
