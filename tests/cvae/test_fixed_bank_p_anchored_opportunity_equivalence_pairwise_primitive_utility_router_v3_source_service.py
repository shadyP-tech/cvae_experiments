from __future__ import annotations

import inspect
import os
from pathlib import Path
from types import MappingProxyType

import numpy as np

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.config import (
    build_planned_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.execution.services import (
    CanonicalScientificRouterService,
    ServicePreflightRequest,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.identity import (
    CENTERS,
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_PROBABILITY_MATRIX_SHAPE,
    EXPECTED_TEST_ROWS_BY_CENTER,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.physical import (
    PHASE_ORDER,
    PhysicalRuntimeConfig,
    action_library_by_target,
    assemble_compiled_probability_matrix,
    physical_runtime_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.action_compiler import (
    BasePredictionSurface,
    canonical_compiler_receipt,
    compile_action_surface,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.candidate_pools import (
    build_final_outer_candidate_pool,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.physical.cpu_pool import (
    _initialize_one_thread_cpu_worker,
    _validated_plain_task,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.physical import prediction_runtime as prediction_runtime_module
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.physical.cache_loader import (
    _safe_root as safe_cache_root,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.physical.upstream import (
    _safe_root as safe_upstream_root,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.service_factory import (
    prepare_canonical_scientific_service_factory,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.source_seal import (
    SourceSealReceipt,
    _reject_unsealed_project_imports,
    build_source_seal,
    validate_live_producer_seal_binding,
    validate_source_seal,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def test_source_seal_covers_v3_and_neutral_core_without_predecessor_imports() -> None:
    receipt = build_source_seal()
    payload = receipt.to_payload()

    assert validate_source_seal(receipt) == receipt
    assert receipt.adapter_member_count > 0
    assert receipt.neutral_member_count > 0
    assert receipt.production_member_count > 0
    assert payload["predecessor_imports_present"] is False
    assert payload["unsealed_project_imports_present"] is False


def test_source_seal_receipt_cannot_be_fabricated() -> None:
    receipt = build_source_seal()
    with pytest.raises(ProtocolError, match="bypassed source admission"):
        SourceSealReceipt(
            repository_root=receipt.repository_root,
            adapter_member_count=receipt.adapter_member_count,
            adapter_tree_sha256=receipt.adapter_tree_sha256,
            neutral_member_count=receipt.neutral_member_count,
            neutral_tree_sha256=receipt.neutral_tree_sha256,
            production_member_count=receipt.production_member_count,
            production_tree_sha256=receipt.production_tree_sha256,
            shared_protocol_sha256=receipt.shared_protocol_sha256,
            combined_source_sha256=receipt.combined_source_sha256,
        )


def test_authorized_source_bundle_must_bind_the_live_producer_seal() -> None:
    seal = build_source_seal()
    assert (
        validate_live_producer_seal_binding(
            configured_sha256=seal.combined_source_sha256,
            parsed_sha256=seal.combined_source_sha256,
            source_seal=seal,
        )
        == seal.combined_source_sha256
    )
    with pytest.raises(ProtocolError, match="producer seal is not live"):
        validate_live_producer_seal_binding(
            configured_sha256="a" * 64,
            parsed_sha256="a" * 64,
            source_seal=seal,
        )


def test_source_seal_symbol_exception_does_not_allow_module_escape(
    tmp_path: Path,
) -> None:
    module = (
        tmp_path
        / "src/midogpp_thesis/real_features/classifier_reference/schemas/midogpp.py"
    )
    module.parent.mkdir(parents=True)
    module.write_text(
        "from . import DIAGNOSTIC_ONLY, SELECTION_ELIGIBLE\n",
        encoding="utf-8",
    )
    _reject_unsealed_project_imports(module, repository_root=tmp_path)

    module.write_text("from . import unsealed_module\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="unsealed project module"):
        _reject_unsealed_project_imports(module, repository_root=tmp_path)


def test_nominal_service_is_concrete_but_factory_requires_typed_source_surface() -> None:
    assert inspect.isabstract(CanonicalScientificRouterService) is False
    with pytest.raises(ProtocolError, match="bypassed its factory"):
        CanonicalScientificRouterService(
            source_surface=object(),
            source_seal_hash="a" * 64,
            seven_input_contract_hash="b" * 64,
            factory_identity_hash="c" * 64,
        )


def test_service_preflight_rejects_protocol_hash_drift() -> None:
    service = object.__new__(CanonicalScientificRouterService)
    object.__setattr__(service, "_seven_input_contract_hash", "1" * 64)
    object.__setattr__(service, "_source_seal_hash", "2" * 64)
    with pytest.raises(ProtocolError, match="lineage drifted"):
        service.preflight(
            ServicePreflightRequest(
                seven_input_contract_hash="1" * 64,
                protocol_hash="3" * 64,
                source_seal_hash="2" * 64,
                workstation_receipt_hash="4" * 64,
            )
        )


def test_v3_owned_physical_action_library_excludes_target_expert() -> None:
    library = action_library_by_target()
    assert tuple(library) == CENTERS
    for target, actions in library.items():
        assert len(actions) == 10
        assert tuple(action.action_id for action in actions[:2]) == ("B", "U")
        assert len({action.action_hash for action in actions}) == 10
        assert all(target not in action.counts_by_class["0"] for action in actions)
        assert all(action.to_payload()["target_expert_excluded"] for action in actions)
        sources = tuple(center for center in CENTERS if center != target)
        assert set(actions[0].counts_by_class["0"].values()) == {128}
        assert set(actions[1].counts_by_class["0"].values()) == {144}
        selected = actions[2].selected_source
        assert selected in sources
        assert actions[2].counts_by_class["0"][selected] == 256
        assert {
            actions[2].counts_by_class["0"][source]
            for source in sources
            if source != selected
        } == {128}

    with pytest.raises(ProtocolError, match="source-training surface identity drifted"):
        prepare_canonical_scientific_service_factory(
            build_planned_config(),
            source_seal=build_source_seal(),
            source_surface=object(),
        )


def test_v3_physical_runtime_is_four_spawn_workers_with_one_thread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = PhysicalRuntimeConfig(tmp_path / "bank")
    assert config.runtime == physical_runtime_payload()
    assert config.runtime["classifier_workers"] == 4
    assert config.runtime["classifier_threads_per_worker"] == 1
    assert _validated_plain_task({"task_id": "t", "threads_per_fit": 1}) == {
        "task_id": "t",
        "threads_per_fit": 1,
    }
    with pytest.raises(ProtocolError, match="spawn-safe"):
        _validated_plain_task(
            {"task_id": "t", "threads_per_fit": 1, "unsafe": MappingProxyType({})}
        )

    for name in (
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.delenv(name, raising=False)
    _initialize_one_thread_cpu_worker()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert PHASE_ORDER == (
        "two_persistent_gpu_source_materialization",
        "four_spawn_cpu_prediction",
    )


def test_v3_source_phase_delegates_exact_two_device_schedule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = PhysicalRuntimeConfig(tmp_path / "bank")
    observed: dict[str, object] = {}
    sentinel = object()

    def fake_materialize(config_value, lock_value, *, root):
        observed["devices"] = tuple(config_value.runtime["generation_devices"])
        observed["source_workers"] = config_value.runtime["source_workers_per_device"]
        observed["generation_workers"] = config_value.runtime[
            "generation_workers_per_device"
        ]
        observed["lock"] = lock_value
        observed["root"] = root
        return sentinel

    monkeypatch.setattr(
        prediction_runtime_module,
        "materialize_frozen_source_streams",
        fake_materialize,
    )
    lock = object()
    result = prediction_runtime_module._materialize_source_phase(
        config,
        lock,
        root=tmp_path / "source",
    )
    assert result is sentinel
    assert observed == {
        "devices": ("cuda:0", "cuda:1"),
        "source_workers": 1,
        "generation_workers": 1,
        "lock": lock,
        "root": tmp_path / "source",
    }


def test_v3_physical_input_roots_reject_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)

    with pytest.raises(ProtocolError, match="symlink"):
        safe_cache_root(alias)
    with pytest.raises(ProtocolError, match="symlink"):
        safe_upstream_root(alias, role="expert bank")

    with pytest.raises(ProtocolError, match="symlink"):
        prediction_runtime_module._existing_plain_directory(alias, role="artifact")

    parent_target = tmp_path / "parent-target"
    parent_target.mkdir()
    parent_alias = tmp_path / "parent-alias"
    parent_alias.symlink_to(parent_target, target_is_directory=True)
    nested = parent_alias / "nested"
    nested.mkdir()
    with pytest.raises(ProtocolError, match="symlink"):
        prediction_runtime_module._existing_plain_directory(
            nested, role="scratch"
        )


def test_v3_compiles_canonical_read_only_full_probability_matrix() -> None:
    compiler = canonical_compiler_receipt()
    inventory = tuple((f"expert-{center}", center) for center in CENTERS)
    surfaces = []
    cursor = 0
    for target, count in EXPECTED_TEST_ROWS_BY_CENTER:
        pool = build_final_outer_candidate_pool(
            outer_target_center=target,
            all_center_ids=CENTERS,
            expert_inventory=inventory,
            bank_lock_hash=EXPECTED_BANK_LOCK_HASH,
            source_supervision_contract_hash="a" * 64,
            compiler=compiler,
        )
        row_ids = tuple(f"row-{cursor + index:05d}" for index in range(count))
        cursor += count
        base = BasePredictionSurface(
            outer_target_center=target,
            evaluated_center=target,
            row_ids=row_ids,
            equal_union_probabilities=(0.2,) * count,
            union_probabilities=(0.8,) * count,
            expert_probabilities=tuple(
                (source, (0.3 + ordinal / 100.0,) * count)
                for ordinal, source in enumerate(pool.candidate_center_ids)
            ),
            candidate_pool_receipt_hash=pool.receipt_hash,
        )
        surfaces.append(
            compile_action_surface(base, candidate_pool=pool, compiler=compiler)
        )
    matrix = assemble_compiled_probability_matrix(surfaces)
    assert matrix.values.shape == EXPECTED_PROBABILITY_MATRIX_SHAPE
    assert matrix.values.dtype == np.dtype("<f4")
    assert matrix.values.flags.c_contiguous
    assert matrix.values.flags.writeable is False
    assert tuple(matrix.center_offsets) == CENTERS
