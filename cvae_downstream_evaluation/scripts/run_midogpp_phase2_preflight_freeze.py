"""Materialize MIDOG++ phase-2 routing-freeze artifacts only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvae_downstream_evaluation.artifacts.midogpp_phase2 import materialize_phase2_preflight_freeze  # noqa: E402
from cvae_downstream_evaluation.protocol import ProtocolError  # noqa: E402
from cvae_downstream_evaluation.schemas.midogpp_phase2 import assert_phase2_preflight_config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write and validate MIDOG++ phase-2 routing-freeze artifacts before downstream utility."
    )
    parser.add_argument("--config", required=True, help="JSON preflight config.")
    parser.add_argument("--out-dir", default=None, help="Override artifact root from config.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config shape without writing artifacts.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config)
    config = _read_json(config_path)
    assert_phase2_preflight_config(config)
    root = Path(args.out_dir or str(config.get("out_dir", "")))
    if not str(root):
        raise ProtocolError("Phase-2 preflight requires out_dir in config or --out-dir.")
    if args.dry_run:
        print(json.dumps({"status": "dry_run_passed", "out_dir": str(root)}, indent=2, sort_keys=True))
        return
    report = materialize_phase2_preflight_freeze(
        root=root,
        source_rows=_sequence(config, "source_rows"),
        target_rows=_sequence(config, "target_rows"),
        support_score_inputs=_sequence(config, "support_scores"),
        heldout_center=str(config["heldout_center"]),
        support_size=int(config["support_size"]),
        support_seed=int(config["support_seed"]),
        replicate=str(config.get("replicate", "0")),
        freeze_run_id=str(config["freeze_run_id"]),
        freeze_timestamp=str(config["freeze_timestamp"]),
        snapshot_fields=dict(config.get("snapshot_fields", {})),
        center_column=str(config.get("center_column", "center")),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"Malformed phase-2 preflight config: {path}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("Phase-2 preflight config must be a JSON object.")
    return payload


def _sequence(config: dict[str, object], key: str) -> list[dict[str, object]]:
    value = config.get(key)
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ProtocolError(f"Phase-2 preflight config field {key!r} must be a list of objects.")
    return [dict(row) for row in value]


if __name__ == "__main__":
    try:
        main()
    except ProtocolError as exc:
        raise SystemExit(str(exc)) from exc
