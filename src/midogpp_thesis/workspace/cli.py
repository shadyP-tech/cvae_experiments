from __future__ import annotations

import argparse
import json
import shlex

from .runtime import MidogppWorkspace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Canonical MIDOG++ experiment workspace.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("validate", help="Validate stage, catalog, and protocol ownership metadata.")
    sub.add_parser("list", help="List registered MIDOG++ experiments.")

    show = sub.add_parser("show", help="Show one registered experiment.")
    show.add_argument("experiment_id")

    resolve = sub.add_parser("resolve", help="Resolve one logical artifact ID.")
    resolve.add_argument("artifact_id")
    resolve.add_argument("--output", action="store_true")
    resolve.add_argument("--allow-missing", action="store_true")

    command = sub.add_parser("command", help="Print the canonical launcher command.")
    command.add_argument("experiment_id")

    prepare = sub.add_parser("prepare", help="Resolve inputs and write frozen run snapshots.")
    prepare.add_argument("experiment_id")
    prepare.add_argument("--allow-missing-inputs", action="store_true")
    prepare.add_argument("--force", action="store_true")

    run = sub.add_parser("run", help="Prepare and execute a registered experiment.")
    run.add_argument("experiment_id")
    run.add_argument("--force", action="store_true")
    run.add_argument("extra_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = MidogppWorkspace.load()
    if args.command == "validate":
        workspace.validate()
        print("PASS: MIDOG++ workspace registry, catalog, and protocol boundaries are valid.")
        return 0
    if args.command == "list":
        for experiment in workspace.experiments.values():
            print(
                "\t".join(
                    (
                        experiment.experiment_id,
                        experiment.stage,
                        experiment.status,
                        experiment.claim_scope,
                    )
                )
            )
        return 0
    if args.command == "show":
        experiment = workspace.get_experiment(args.experiment_id)
        print(json.dumps(experiment.__dict__, indent=2, sort_keys=True))
        return 0
    if args.command == "resolve":
        path = workspace.resolve_artifact(
            args.artifact_id,
            for_output=bool(args.output),
            require_exists=not bool(args.allow_missing),
        )
        print(path)
        return 0
    if args.command == "command":
        print(workspace.central_command(args.experiment_id))
        return 0
    if args.command == "prepare":
        prepared = workspace.prepare(
            args.experiment_id,
            require_inputs=not bool(args.allow_missing_inputs),
            force=bool(args.force),
        )
        print(f"artifact_root={prepared.artifact_root}")
        print(f"resolved_config={prepared.resolved_config_path}")
        print(f"input_manifest={prepared.input_manifest_path}")
        print(f"command={shlex.join(prepared.argv)}")
        return 0
    if args.command == "run":
        extra = tuple(args.extra_args)
        if extra and extra[0] == "--":
            extra = extra[1:]
        return workspace.run(args.experiment_id, force=bool(args.force), extra_args=extra)
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
