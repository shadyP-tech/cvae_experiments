"""Persist immutable input, protocol, host, and worker launch receipts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ..artifacts.hashing import canonical_hash, sha256_file
from ..artifacts.io import atomic_json
from ..config import ScaleBPV2Config
from ..identity import DIRECT_INPUT_ARTIFACT_IDS
from ..input_contracts import LabelFreeTestFrame
from ..inputs import ValidatedInputs
from ..reports import protocol_manifest_payload


def persist_launch_receipts(
    root: Path,
    *,
    config: ScaleBPV2Config,
    inputs: ValidatedInputs,
    frame: LabelFreeTestFrame,
    firewall: Mapping[str, object],
    workstation: Mapping[str, object],
    admission_hash: str,
    source_fence_hash: str,
    run_identity_hash: str,
    worker_contract: Mapping[str, object],
    authorization_lease_claim_hash: str,
) -> None:
    """Bind every pre-execution admission receipt into the run artifact."""

    input_hashes = {
        DIRECT_INPUT_ARTIFACT_IDS[0]: canonical_hash(dict(inputs.expert_bank_receipt)),
        DIRECT_INPUT_ARTIFACT_IDS[1]: canonical_hash(
            inputs.generation_lock.to_payload()
        ),
        DIRECT_INPUT_ARTIFACT_IDS[2]: frame.cache_binding_hash,
        DIRECT_INPUT_ARTIFACT_IDS[3]: sha256_file(config.test_manifest_path),
        DIRECT_INPUT_ARTIFACT_IDS[4]: sha256_file(config.test_consumption_ledger_path),
        DIRECT_INPUT_ARTIFACT_IDS[5]: sha256_file(config.ledger_amendment_path),
    }
    manifest = protocol_manifest_payload(
        config_hash=config.contract_hash,
        protocol_hash=str(config.protocol["protocol_hash"]),
        run_identity_hash=run_identity_hash,
        admission_receipt_hash=admission_hash,
        input_artifact_hashes=input_hashes,
        source_fence_hash=source_fence_hash,
        workstation_plan_hash=str(workstation["plan_hash"]),
        authorization_lease_claim_hash=authorization_lease_claim_hash,
    )
    atomic_json(root / "manifests/protocol_manifest.json", manifest)
    atomic_json(root / "reports/pre_gpu_firewall.json", dict(firewall))
    atomic_json(root / "reports/workstation_preflight.json", dict(workstation))
    atomic_json(root / "reports/worker_contract_preflight.json", dict(worker_contract))


__all__ = ("persist_launch_receipts",)
