"""Two independent fresh-process validation replays for the S4 bundle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping

from ...protocol import ProtocolError
from .hashing import canonical_hash


def run_two_fresh_process_replays(
    root: Path, *, config_path: Path
) -> Mapping[str, object]:
    results = tuple(
        _one_replay(root, config_path=config_path, replay_ordinal=ordinal)
        for ordinal in (0, 1)
    )
    if results[0] != results[1]:
        raise ProtocolError("S4 fresh-process validation replays disagree.")
    digest = canonical_hash(results[0])
    return {
        "schema_version": "fixed_bank_support_static_router_fresh_process_validation_v1",
        "status": "PASS",
        "replay_count": 2,
        "replay_hashes": [digest, digest],
        "byte_identical_replay_results": True,
        "independent_processes": True,
        "cuda_visible_devices": "",
        "blas_threads_per_fresh_process": 1,
        "python_hash_seed": 0,
        "validation_result": results[0],
        "fresh_evidence": False,
        "promotion_eligible": False,
    }


def _one_replay(
    root: Path, *, config_path: Path, replay_ordinal: int
) -> Mapping[str, object]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = "1"
    command = (
        sys.executable,
        "-m",
        "midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router.fresh_process_validation",
        "--child",
        "--root",
        str(root.resolve()),
        "--config",
        str(config_path.resolve()),
    )
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=600,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip()[-1000:]
        raise ProtocolError(
            f"S4 fresh-process replay {replay_ordinal} failed: {detail}."
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ProtocolError("S4 fresh-process replay returned malformed JSON.") from exc
    if not isinstance(payload, Mapping) or payload.get("status") != "PASS":
        raise ProtocolError("S4 fresh-process replay did not pass.")
    return dict(payload)


def _child(root: Path, config_path: Path) -> Mapping[str, object]:
    from .config import load_fixed_bank_support_static_router_config
    from .validation import validate_fixed_bank_support_static_router_bundle

    config = load_fixed_bank_support_static_router_config(config_path)
    if Path(config.artifact_root).resolve() != root.resolve():
        raise ProtocolError("S4 fresh-process config root differs from requested bundle.")
    return validate_fixed_bank_support_static_router_bundle(
        root,
        config=config,
        allow_pending_validation=True,
        skip_fresh_process_report=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.child:
        parser.error("This module's command-line surface is child-only.")
    payload = _child(args.root, args.config)
    sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a fresh process
    raise SystemExit(main())


__all__ = ("run_two_fresh_process_replays",)
