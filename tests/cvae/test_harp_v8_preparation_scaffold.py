from __future__ import annotations

from pathlib import Path
import shutil

from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v8 import authorization
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v8.workstation_preparation import (
    plan_harp_v8_workstation_preparation,
)
from midogpp_thesis.cvae.runtime.artifact_io import atomic_json


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_RELATIVE = Path(
    "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts/harp_router_v8"
)


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    shutil.copytree(ROOT / "experiments/midogpp", repository / "experiments/midogpp")
    contract_root = repository / CONTRACT_RELATIVE
    required_directories = {
        "artifacts/midogpp/30_expert_bank/"
        "uniform_b_v2_routing_authorized_expert_bank_v1",
        "artifacts/midogpp/40_prior_and_generation/"
        "uniform_b_v2_generation_lock/v1",
        "datasets/midogpp/derived/features/virchow2/"
        "uniform_b_v2_descriptive_test_cache_v1/seed42",
        "datasets/midogpp/contract/annotation_patch_v1",
        "artifacts/midogpp/10_real_feature_reference/"
        "uniform_b_canonical_real_feature_reference_v1/seed42/reports",
        "artifacts/midogpp/90_oracles_and_diagnostics",
    }
    assert contract_root.is_dir()
    for relative in required_directories:
        (repository / relative).mkdir(parents=True, exist_ok=True)
    (
        repository / "datasets/midogpp/contract/annotation_patch_v1/manifest.csv"
    ).write_text("case_id,center,split,label\ncase,0,test,0\n", encoding="utf-8")
    atomic_json(
        repository
        / "artifacts/midogpp/10_real_feature_reference/"
        "uniform_b_canonical_real_feature_reference_v1/seed42/"
        "reports/test_consumption_ledger.json",
        {"schema_version": "synthetic_parent"},
    )
    return repository


def _inventory(root: Path) -> tuple[str, ...]:
    return tuple(sorted(path.relative_to(root).as_posix() for path in root.rglob("*")))


def test_v8_contract_scaffold_supports_mutation_free_preparation_planning(
    tmp_path: Path,
) -> None:
    packaged_root = ROOT / CONTRACT_RELATIVE
    scaffold = packaged_root / ".gitkeep"
    assert packaged_root.is_dir() and not packaged_root.is_symlink()
    assert scaffold.is_file() and not scaffold.is_symlink()

    repository = _repository(tmp_path)
    amendment = repository / authorization.WORKSPACE_AMENDMENT_RELATIVE_PATH
    lease = authorization.lease_path(repository)
    output = repository / authorization.WORKSPACE_OUTPUT_CANONICAL_PATH
    before = _inventory(repository)

    plan = plan_harp_v8_workstation_preparation(repository)
    payload = plan.to_payload()

    assert payload["status"] == "READY_FOR_EXPLICIT_PREPARATION_CONFIRMATION"
    assert payload["filesystem_mutations"] == 0
    assert payload["execution_amendment_created"] is False
    assert payload["execution_authorized"] is False
    assert payload["canonical_scoring_manifest_opened"] is False
    assert payload["canonical_scoring_manifest_hashed"] is False
    assert plan.paths.amendment_path == amendment.resolve()
    assert not amendment.exists()
    assert not lease.exists()
    assert not output.exists()
    assert _inventory(repository) == before
