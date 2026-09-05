"""HARP v18 command registration and lifecycle dispatch."""

from __future__ import annotations

from pathlib import Path


def register_commands(sub) -> None:
    harp_stage90_v18 = sub.add_parser(
        "fixed-bank-harp-router-v18",
        help=(
            "Inspect, dry-run, or execute the separately authorized HARP v18 "
            "pooled source-train selected-policy router."
        ),
    )
    harp_stage90_v18.add_argument("--config", required=True)
    harp_stage90_v18.add_argument(
        "--artifact-root",
        default=".",
        help="Prepared v18 output root; ignored by path-free planned inspection.",
    )
    harp_stage90_v18_mode = harp_stage90_v18.add_mutually_exclusive_group()
    harp_stage90_v18_mode.add_argument(
        "--inspect-plan", action="store_true",
        help="Inspect v18 identities and architecture without resolving inputs.",
    )
    harp_stage90_v18_mode.add_argument(
        "--dry-run", action="store_true",
        help="Validate authorized v18 inputs without claiming the single-use lease.",
    )
    harp_stage90_v18_mode.add_argument(
        "--confirm", help="Exact single-use v18 launch confirmation token."
    )
    harp_prepare_v18 = sub.add_parser(
        "prepare-fixed-bank-harp-router-v18-inputs",
        help=(
            "Plan or materialize the catalog-bound v18-only label-blind cache, "
            "source-train label capability, and sealed full-test release; this "
            "issues no execution authority."
        ),
    )
    harp_prepare_v18.add_argument("--repository-root", required=True)
    harp_prepare_v18.add_argument(
        "--confirm", help="Exact v18 preparation token; omit for a mutation-free plan."
    )
    harp_activate_v18 = sub.add_parser(
        "activate-fixed-bank-harp-router-v18",
        help=(
            "Render a mutation-free v18 activation plan from exact prepared "
            "inputs, or commit it only with the exact confirmation token."
        ),
    )
    harp_activate_v18.add_argument("--authorization-basis", required=True)
    harp_activate_v18.add_argument("--authorization-date", required=True)
    harp_activate_v18.add_argument("--repository-root", required=True)
    harp_activate_v18.add_argument(
        "--confirm", help="Exact v18 activation token; omit for a mutation-free plan."
    )
    harp_supersede_v18 = sub.add_parser(
        "supersede-fixed-bank-harp-router-v18-activation",
        help=(
            "Archive and retire an authenticated active, unconsumed, pre-lease "
            "v18 activation; this never creates run authority."
        ),
    )
    harp_supersede_v18.add_argument("--repository-root", required=True)
    harp_supersede_v18.add_argument(
        "--confirm", help="Exact active v18 supersession token; omit for a plan."
    )
    harp_supersede_rollback_v18 = sub.add_parser(
        "supersede-rolled-back-fixed-bank-harp-router-v18-activation",
        help=(
            "Rollback exact partial activation bytes when necessary, then archive "
            "an authenticated source-drifted v18 activation attempt."
        ),
    )
    harp_supersede_rollback_v18.add_argument("--repository-root", required=True)
    harp_supersede_rollback_v18.add_argument(
        "--confirm", help="Exact rolled-back v18 supersession token; omit for a plan."
    )


def dispatch(args) -> int | None:
    if args.surface == "prepare-fixed-bank-harp-router-v18-inputs":
        import json

        from .fixed_bank_harp_router_v18.workstation_preparation import (
            plan_harp_v18_workstation_preparation,
            prepare_harp_v18_workstation_inputs,
        )

        plan = plan_harp_v18_workstation_preparation(args.repository_root)
        result = (
            plan.to_payload()
            if args.confirm is None
            else prepare_harp_v18_workstation_inputs(
                plan, confirmation=args.confirm
            ).to_payload()
        )
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0

    if args.surface == "activate-fixed-bank-harp-router-v18":
        import json

        from .fixed_bank_harp_router_v18.activation import (
            activate_harp_v18,
            inspect_harp_v18_activation_recovery,
            plan_harp_v18_activation,
            recover_harp_v18_activation,
        )
        from .fixed_bank_harp_router_v18.config import load_config
        from .fixed_bank_harp_router_v18.workspace_paths import (
            resolve_harp_v18_workspace_paths,
        )

        recovery = inspect_harp_v18_activation_recovery(args.repository_root)
        if recovery is not None:
            result = (
                recovery
                if args.confirm is None
                else recover_harp_v18_activation(
                    args.repository_root, confirmation=args.confirm
                ).to_payload()
            )
        else:
            paths = resolve_harp_v18_workspace_paths(
                args.repository_root, require_prepared=True
            )
            plan = plan_harp_v18_activation(
                load_config(paths.config_path),
                **paths.activation_kwargs(),
                repository_root=args.repository_root,
                authorization_basis=args.authorization_basis,
                authorization_date=args.authorization_date,
            )
            result = (
                plan.to_payload()
                if args.confirm is None
                else activate_harp_v18(plan, confirmation=args.confirm).to_payload()
            )
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0

    if args.surface == "supersede-fixed-bank-harp-router-v18-activation":
        import json

        from .fixed_bank_harp_router_v18.activation_supersession import (
            plan_harp_v18_active_activation_supersession,
            supersede_harp_v18_active_activation,
        )

        plan = plan_harp_v18_active_activation_supersession(args.repository_root)
        result = (
            plan.to_payload()
            if args.confirm is None
            else supersede_harp_v18_active_activation(
                plan, confirmation=args.confirm
            ).to_payload()
        )
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0

    if args.surface == "supersede-rolled-back-fixed-bank-harp-router-v18-activation":
        import json

        from .fixed_bank_harp_router_v18.activation_supersession import (
            plan_harp_v18_activation_supersession,
            supersede_harp_v18_activation,
        )

        plan = plan_harp_v18_activation_supersession(args.repository_root)
        result = (
            plan.to_payload()
            if args.confirm is None
            else supersede_harp_v18_activation(
                plan, confirmation=args.confirm
            ).to_payload()
        )
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0

    if args.surface == "fixed-bank-harp-router-v18":
        import json

        from ..protocol import ProtocolError
        from .fixed_bank_harp_router_v18.config import load_config
        from .fixed_bank_harp_router_v18.runner import (
            HARP_V18_RUN_CONFIRMATION_TOKEN,
            dry_run_harp_stage90_v18,
            inspect_harp_stage90_v18,
            run_harp_stage90_v18,
        )

        if (
            not args.inspect_plan
            and not args.dry_run
            and args.confirm != HARP_V18_RUN_CONFIRMATION_TOKEN
        ):
            raise ProtocolError(
                "HARP v18 execution requires the exact confirmation token "
                f"{HARP_V18_RUN_CONFIRMATION_TOKEN}."
            )
        config = load_config(args.config)
        artifact_root = Path(args.artifact_root)
        if args.inspect_plan:
            print(json.dumps(inspect_harp_stage90_v18(config), sort_keys=True, separators=(",", ":")))
        elif args.dry_run:
            print(json.dumps(dry_run_harp_stage90_v18(config, artifact_root=artifact_root), sort_keys=True, separators=(",", ":")))
        else:
            print(run_harp_stage90_v18(
                config,
                artifact_root=artifact_root,
                confirmation_token=args.confirm,
            ))
        return 0

    return None
