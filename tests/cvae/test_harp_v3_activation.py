from __future__ import annotations

import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import yaml

from midogpp_thesis.cvae.diagnostics import cli
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3 import activation
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3 import (
    activation_transaction,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3 import (
    amendment_publisher,
    authorization,
    preparation as preparation_module,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3.config import (
    load_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3.identity import (
    EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3.input_surfaces import (
    V3_CACHE_IDENTITY,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3.preparation import (
    PREPARATION_RECEIPT,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3.workspace_paths import (
    resolve_harp_v3_workspace_paths,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.harp_protocol import canonical_hash
from midogpp_thesis.cvae.runtime.artifact_io import atomic_json, sha256_file


ROOT = Path(__file__).resolve().parents[2]


_BANK = Path(
    "artifacts/midogpp/30_expert_bank/"
    "uniform_b_v2_routing_authorized_expert_bank_v1"
)
_GENERATION = Path(
    "artifacts/midogpp/40_prior_and_generation/"
    "uniform_b_v2_generation_lock/v1"
)
_CANONICAL_CACHE = Path(
    "datasets/midogpp/derived/features/virchow2/"
    "uniform_b_v2_descriptive_test_cache_v1/seed42"
)
_CANONICAL_MANIFEST = Path(
    "datasets/midogpp/contract/annotation_patch_v1/manifest.csv"
)
_PREPARED_CACHE = Path(
    "datasets/midogpp/derived/features/virchow2/harp_consumed_test_cache_v3"
)
_DEVELOPMENT = Path(
    "datasets/midogpp/contract/harp_consumed_test_development_v3/manifest.csv"
)
_EVALUATION = Path(
    "datasets/midogpp/contract/harp_consumed_test_evaluation_v3/manifest.csv"
)
_PARENT = Path(
    "artifacts/midogpp/10_real_feature_reference/"
    "uniform_b_canonical_real_feature_reference_v1/seed42/"
    "reports/test_consumption_ledger.json"
)


def _synthetic_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[activation.HarpV3ActivationPlan, Path]:
    repository = tmp_path / "repository"
    shutil.copytree(
        ROOT / "experiments/midogpp", repository / "experiments/midogpp"
    )
    amendment = repository / authorization.WORKSPACE_AMENDMENT_RELATIVE_PATH
    amendment.parent.mkdir(parents=True, exist_ok=True)
    (repository / "artifacts/midogpp/90_oracles_and_diagnostics").mkdir(
        parents=True, exist_ok=True
    )
    config = load_config(repository / authorization.WORKSPACE_CONFIG_RELATIVE_PATH)

    bank = repository / _BANK
    generation = repository / _GENERATION
    cache = repository / _PREPARED_CACHE
    canonical_cache = repository / _CANONICAL_CACHE
    canonical_manifest = repository / _CANONICAL_MANIFEST
    for path in (bank, generation, canonical_cache, cache / "manifests"):
        path.mkdir(parents=True, exist_ok=True)
    canonical_manifest.parent.mkdir(parents=True, exist_ok=True)
    canonical_manifest.write_text("sample_id,label\nsource,0\n", encoding="utf-8")
    development = repository / _DEVELOPMENT
    evaluation = repository / _EVALUATION
    parent = repository / _PARENT
    for path in (development, evaluation, parent):
        path.parent.mkdir(parents=True, exist_ok=True)
    development.write_text("sample_id,label\ndev,0\n", encoding="utf-8")
    evaluation.write_text("sample_id,label\neval,1\n", encoding="utf-8")
    atomic_json(parent, {"schema_version": "synthetic_harp_v3_parent"})
    content_base = {
        "schema_version": V3_CACHE_IDENTITY.content_schema,
        "members": {},
    }
    content_hash = canonical_hash(content_base)
    atomic_json(
        cache / "manifests/content_index.json",
        {**content_base, "content_index_hash": content_hash},
    )
    atomic_json(cache / "manifests/cache_index.json", {"synthetic": True})
    atomic_json(
        cache / PREPARATION_RECEIPT,
        {"partition_hash": "4" * 64, "receipt_hash": "5" * 64},
    )
    fake_cache = SimpleNamespace(
        root=cache,
        content_sha256=content_hash,
        cache_hash="3" * 64,
        member_sha256={},
    )
    fake_source = {
        "source_snapshot_schema": "midogpp_harp_stage90_source_snapshot_v3",
        "source_snapshot_manifest_sha256": "1" * 64,
        "source_snapshot_tree_sha256": "2" * 64,
        "source_snapshot_member_count": 1,
        "source_snapshot_member_pattern": "synthetic_closed_world_source",
        "source_snapshot_excludes_bytecode_and_cache": True,
    }
    monkeypatch.setattr(
        authorization, "source_snapshot_identity", lambda _root=None: fake_source
    )
    monkeypatch.setattr(
        activation, "source_snapshot_identity", lambda _root=None: fake_source
    )
    monkeypatch.setattr(
        amendment_publisher, "load_cache_index", lambda _config: fake_cache
    )
    monkeypatch.setattr(
        amendment_publisher,
        "_validate_preparation_receipt",
        lambda observed, _computed: "5" * 64
        if observed is fake_cache
        else (_ for _ in ()).throw(AssertionError("wrong cache")),
    )
    monkeypatch.setattr(
        amendment_publisher,
        "validate_physical_inputs",
        lambda _config, observed: SimpleNamespace(receipt_hash="6" * 64)
        if observed is fake_cache
        else (_ for _ in ()).throw(AssertionError("wrong cache")),
    )
    plan = activation.plan_harp_v3_activation(
        config,
        expert_bank_root=bank,
        generation_lock_root=generation,
        prepared_cache_root=cache,
        development_manifest_path=development,
        evaluation_manifest_path=evaluation,
        parent_ledger_path=parent,
        repository_root=repository,
        authorization_basis=authorization.AUTHORIZATION_BASIS,
        authorization_date="2026-09-01",
    )
    return plan, repository


def test_activation_plan_is_deterministic_and_mutation_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, repository = _synthetic_plan(tmp_path, monkeypatch)
    before = {
        path: path.read_bytes()
        for path in (plan.config_path, plan.registry_path, plan.catalog_path)
    }
    second = activation.plan_harp_v3_activation(
        load_config(plan.config_path),
        **resolve_harp_v3_workspace_paths(
            repository, require_prepared=True
        ).activation_kwargs(),
        repository_root=repository,
        authorization_basis=authorization.AUTHORIZATION_BASIS,
        authorization_date="2026-09-01",
    )
    assert plan.activation_plan_hash == second.activation_plan_hash
    assert plan.final_config_bytes == second.final_config_bytes
    assert plan.final_registry_bytes == second.final_registry_bytes
    assert plan.final_catalog_bytes == second.final_catalog_bytes
    assert plan.to_payload()["filesystem_mutations"] == 0
    assert plan.to_payload()["checked_in_config_remains_planned"] is True
    assert plan.authorized_config.execution_authorized is True
    assert plan.authorized_config.expected_execution_amendment_sha256 == (
        plan.amendment_draft.amendment_sha256
    )
    assert not plan.amendment_draft.amendment_path.exists()
    assert not (
        repository / activation_transaction.TRANSACTION_RELATIVE_PATH
    ).exists()
    assert not (
        repository / activation_transaction.LOCK_RELATIVE_PATH
    ).exists()
    assert all(path.read_bytes() == raw for path, raw in before.items())

    rendered_config = yaml.safe_load(plan.final_config_bytes)
    assert rendered_config["experiment"]["status"] == "diagnostic"
    assert rendered_config["experiment"]["execution_authorized"] is True
    assert rendered_config["claim_boundary"]["fresh_evidence"] is False
    assert rendered_config["claim_boundary"]["routing_success_claimed"] is False
    rendered_registry = yaml.safe_load(plan.final_registry_bytes)
    experiment = next(
        row
        for row in rendered_registry["experiments"]
        if row["experiment_id"] == EXPERIMENT_ID
    )
    assert experiment["status"] == "diagnostic"
    rendered_catalog = yaml.safe_load(plan.final_catalog_bytes)
    output = next(
        row
        for row in rendered_catalog["artifacts"]
        if row["artifact_id"] == OUTPUT_ARTIFACT_ID
    )
    assert output["semantic_identities"]["execution_authorized"] == "true"
    assert output["semantic_identities"]["fresh_evidence"] == "false"
    assert output["semantic_identities"]["routing_success_claimed"] == "false"


def test_confirmation_mismatch_leaves_everything_planned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _ = _synthetic_plan(tmp_path, monkeypatch)
    before = (
        plan.config_path.read_bytes(),
        plan.registry_path.read_bytes(),
        plan.catalog_path.read_bytes(),
    )
    with pytest.raises(ProtocolError, match="confirmation"):
        activation.activate_harp_v3(plan, confirmation="implementation-request")
    assert not plan.amendment_draft.amendment_path.exists()
    assert before == (
        plan.config_path.read_bytes(),
        plan.registry_path.read_bytes(),
        plan.catalog_path.read_bytes(),
    )


def test_retired_arbitrary_path_preparation_rejects_without_path_access(
    tmp_path: Path,
) -> None:
    class PoisonPath:
        def __fspath__(self) -> str:
            raise AssertionError("retired preparation touched a path")

    poison = PoisonPath()
    with pytest.raises(ProtocolError, match="arbitrary-path preparation is disabled"):
        preparation_module.prepare_harp_consumed_test_inputs_v3(
            canonical_cache_root=poison,  # type: ignore[arg-type]
            canonical_manifest_path=poison,  # type: ignore[arg-type]
            parent_ledger_path=poison,  # type: ignore[arg-type]
            cache_root=tmp_path / "cache-must-remain-absent",
            development_manifest_path=tmp_path / "dev-must-remain-absent.csv",
            evaluation_manifest_path=tmp_path / "eval-must-remain-absent.csv",
        )
    assert tuple(tmp_path.iterdir()) == ()


def test_retired_direct_publisher_rejects_before_draft_or_input_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        amendment_publisher,
        "build_harp_v3_execution_amendment_draft",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("retired publisher constructed a draft")
        ),
    )
    with pytest.raises(ProtocolError, match="direct amendment publication is disabled"):
        amendment_publisher.publish_harp_v3_execution_amendment(
            object(),  # type: ignore[arg-type]
            expert_bank_root="unread-bank",
            generation_lock_root="unread-generation",
            prepared_cache_root="unread-cache",
            development_manifest_path="unread-development",
            evaluation_manifest_path="unread-evaluation",
            parent_ledger_path="unread-parent",
            amendment_path="unwritten-amendment",
            authorization_basis="unread-basis",
            authorization_date="unread-date",
            repository_root="unread-repository",
        )


def test_explicit_transaction_commits_registry_last_and_remains_terminal_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, repository = _synthetic_plan(tmp_path, monkeypatch)
    receipt = activation.activate_harp_v3(
        plan, confirmation=activation.ACTIVATION_CONFIRMATION
    )
    assert receipt.amendment_sha256 == sha256_file(
        plan.amendment_draft.amendment_path
    )
    assert load_config(plan.config_path).execution_authorized is True
    registry = yaml.safe_load(plan.registry_path.read_text(encoding="utf-8"))
    experiment = next(
        row
        for row in registry["experiments"]
        if row["experiment_id"] == EXPERIMENT_ID
    )
    assert experiment["status"] == "diagnostic"
    assert not authorization.lease_path(repository).exists()
    assert not (repository / authorization.WORKSPACE_OUTPUT_CANONICAL_PATH).exists()
    amendment = json.loads(
        plan.amendment_draft.amendment_path.read_text(encoding="utf-8")
    )
    assert amendment["predecessor_authority_reused"] is False
    assert amendment["predecessor_output_or_policy_used"] is False
    assert amendment["fresh_evidence"] is False
    with pytest.raises(ProtocolError, match="direct amendment publication is disabled"):
        amendment_publisher.publish_harp_v3_amendment_draft_exclusive(
            plan.amendment_draft
        )


def test_failed_commit_restores_planned_metadata_and_exact_amendment_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, repository = _synthetic_plan(tmp_path, monkeypatch)

    def fail_after_config(point: str) -> None:
        if point == "config_committed":
            raise RuntimeError("synthetic post-publication failure")

    with pytest.raises(ProtocolError, match="failed closed"):
        activation.activate_harp_v3(
            plan,
            confirmation=activation.ACTIVATION_CONFIRMATION,
            _fault_injector=fail_after_config,
        )
    assert plan.config_path.read_bytes() == plan.original_config_bytes
    assert plan.registry_path.read_bytes() == plan.original_registry_bytes
    assert plan.catalog_path.read_bytes() == plan.original_catalog_bytes
    assert plan.amendment_draft.amendment_path.read_bytes() == (
        plan.amendment_draft.amendment_raw
    )

    resumed = activation.plan_harp_v3_activation(
        load_config(plan.config_path),
        **resolve_harp_v3_workspace_paths(
            repository, require_prepared=True
        ).activation_kwargs(),
        repository_root=repository,
        authorization_basis=authorization.AUTHORIZATION_BASIS,
        authorization_date="2026-09-01",
    )
    assert resumed.amendment_already_issued is True
    assert resumed.activation_plan_hash == plan.activation_plan_hash
    activation.activate_harp_v3(
        resumed, confirmation=activation.ACTIVATION_CONFIRMATION
    )
    assert load_config(resumed.config_path).execution_authorized is True


class _SyntheticSigkill(BaseException):
    pass


@pytest.mark.parametrize(
    "commit_point",
    [
        "journal_durable",
        "amendment_committed",
        "config_committed",
        "catalog_committed",
        "registry_committed",
    ],
)
def test_sigkill_at_each_commit_point_resumes_from_exact_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    commit_point: str,
) -> None:
    plan, repository = _synthetic_plan(tmp_path, monkeypatch)

    def kill(point: str) -> None:
        if point == commit_point:
            raise _SyntheticSigkill(point)

    with pytest.raises(_SyntheticSigkill):
        activation.activate_harp_v3(
            plan,
            confirmation=activation.ACTIVATION_CONFIRMATION,
            _fault_injector=kill,
        )
    inspection = activation.inspect_harp_v3_activation_recovery(repository)
    assert inspection is not None
    assert inspection["activation_plan_hash"] == plan.activation_plan_hash
    receipt = activation.recover_harp_v3_activation(
        repository,
        confirmation=activation.ACTIVATION_CONFIRMATION,
    )
    assert receipt.recovered_from_journal is True
    assert load_config(plan.config_path).execution_authorized is True
    registry = yaml.safe_load(plan.registry_path.read_text(encoding="utf-8"))
    experiment = next(
        row
        for row in registry["experiments"]
        if row["experiment_id"] == EXPERIMENT_ID
    )
    assert experiment["status"] == "diagnostic"


def test_rollback_failure_retains_recoverable_exact_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, repository = _synthetic_plan(tmp_path, monkeypatch)

    def fail_forward_and_rollback(point: str) -> None:
        if point == "catalog_committed":
            raise RuntimeError("synthetic forward failure")
        if point == "rollback_config_before_commit":
            raise OSError("synthetic rollback failure")

    with pytest.raises(ProtocolError, match="rollback was incomplete"):
        activation.activate_harp_v3(
            plan,
            confirmation=activation.ACTIVATION_CONFIRMATION,
            _fault_injector=fail_forward_and_rollback,
        )
    inspection = activation.inspect_harp_v3_activation_recovery(repository)
    assert inspection is not None
    assert inspection["status"] == "RECOVERY_REQUIRED_EXACT_CONFIRMATION"
    assert inspection["observed_states"]["registry"] == "original"
    activation.recover_harp_v3_activation(
        repository,
        confirmation=activation.ACTIVATION_CONFIRMATION,
    )
    assert load_config(plan.config_path).execution_authorized is True


def test_cli_recovers_before_loading_partially_committed_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan, repository = _synthetic_plan(tmp_path, monkeypatch)

    def kill_after_config(point: str) -> None:
        if point == "config_committed":
            raise _SyntheticSigkill(point)

    with pytest.raises(_SyntheticSigkill):
        activation.activate_harp_v3(
            plan,
            confirmation=activation.ACTIVATION_CONFIRMATION,
            _fault_injector=kill_after_config,
        )
    assert load_config(plan.config_path).execution_authorized is True
    code = cli.main(
        [
            "activate-fixed-bank-harp-router-v3",
            "--authorization-basis",
            authorization.AUTHORIZATION_BASIS,
            "--authorization-date",
            "2026-09-01",
            "--repository-root",
            str(repository),
            "--confirm",
            activation.ACTIVATION_CONFIRMATION,
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ACTIVATED_AUTHORIZED_SINGLE_USE_NOT_CONSUMED"
    assert payload["recovered_from_journal"] is True


def test_lexical_symlink_and_predecessor_input_paths_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, repository = _synthetic_plan(tmp_path, monkeypatch)
    cache_link = repository / "synthetic/cache_link"
    cache_link.parent.mkdir(parents=True, exist_ok=True)
    cache_link.symlink_to(repository / _PREPARED_CACHE, target_is_directory=True)
    common = {
        "config": load_config(plan.config_path),
        "expert_bank_root": repository / _BANK,
        "generation_lock_root": repository / _GENERATION,
        "development_manifest_path": repository / _DEVELOPMENT,
        "evaluation_manifest_path": repository / _EVALUATION,
        "parent_ledger_path": repository / _PARENT,
        "repository_root": repository,
        "authorization_basis": authorization.AUTHORIZATION_BASIS,
        "authorization_date": "2026-09-01",
    }
    with pytest.raises(ProtocolError, match="lexical symlink"):
        activation.plan_harp_v3_activation(
            **common,
            prepared_cache_root=cache_link,
        )

    predecessor = repository / "synthetic/harp_router_v2"
    predecessor.mkdir()
    with pytest.raises(ProtocolError, match="predecessor path"):
        activation.plan_harp_v3_activation(
            **{
                **common,
                "expert_bank_root": predecessor,
            },
            prepared_cache_root=repository / _PREPARED_CACHE,
        )


def test_byte_identical_alternate_input_location_is_not_activation_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, repository = _synthetic_plan(tmp_path, monkeypatch)
    alternate = repository / "alternate/v3_cache_clone"
    alternate.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(repository / _PREPARED_CACHE, alternate)
    exact = resolve_harp_v3_workspace_paths(
        repository, require_prepared=True
    ).activation_kwargs()

    with pytest.raises(ProtocolError, match="catalog identities"):
        activation.plan_harp_v3_activation(
            load_config(plan.config_path),
            **{**exact, "prepared_cache_root": alternate},
            repository_root=repository,
            authorization_basis=authorization.AUTHORIZATION_BASIS,
            authorization_date="2026-09-01",
        )


def test_existing_nonmatching_amendment_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, repository = _synthetic_plan(tmp_path, monkeypatch)
    plan.amendment_draft.amendment_path.write_bytes(b"{}\n")
    with pytest.raises(ProtocolError, match="does not match"):
        activation.plan_harp_v3_activation(
            load_config(plan.config_path),
            **resolve_harp_v3_workspace_paths(
                repository, require_prepared=True
            ).activation_kwargs(),
            repository_root=repository,
            authorization_basis=authorization.AUTHORIZATION_BASIS,
            authorization_date="2026-09-01",
        )


def test_activation_cli_is_plan_by_default_and_requires_exact_confirm() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "activate-fixed-bank-harp-router-v3",
            "--authorization-basis", authorization.AUTHORIZATION_BASIS,
            "--authorization-date", "2026-09-01",
            "--repository-root", ".",
        ]
    )
    assert args.confirm is None
    assert activation.ACTIVATION_CONFIRMATION not in vars(args).values()
