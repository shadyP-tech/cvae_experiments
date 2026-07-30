"""CLI for MIDOG++ CVAE preservation-only surfaces."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable


Loader = Callable[[str | Path], object]
Runner = Callable[..., Path]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="surface", required=True)
    for name, help_text in (
        ("sanity", "Run the preservation mechanics sanity surface."),
        ("gate", "Run the PCA128 preservation gate."),
        ("condition-audit", "Run the condition-capacity audit."),
        ("tuned-classifier", "Run tuned-classifier preservation."),
        ("source-inner-prior-recovery", "Fit source-inner sampler and objective RecipeLocks."),
        (
            "source-inner-prior-recovery-training-seed-stability",
            "Run the bounded source-inner training-seed stability panel.",
        ),
        (
            "source-inner-learned-conditional-prior-study",
            "Run the non-adoptive learned conditional-prior source-inner study.",
        ),
        (
            "source-inner-task-fisher-shrinkage-study",
            "Run the non-adoptive Task-Fisher shrinkage source-inner study.",
        ),
        (
            "source-inner-aggregate-posterior-mixture-geco",
            "Run the independent-source K=2 aggregate-prior plus GECO study.",
        ),
        (
            "source-inner-uniform-b-geco-task-geometry",
            "Run the non-consumable Uniform-B GECO/task-geometry study.",
        ),
        ("prior-recovery-outer", "Run the locked outer A/B/C/D preservation matrix."),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--config", required=True)
        command.add_argument("--artifact-root", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    loader, runner = _surface_handler(args.surface)
    config = loader(args.config)
    artifact_root = Path(args.artifact_root) if args.artifact_root else None
    output = runner(config, artifact_root=artifact_root)
    print(output)
    return 0


def _surface_handler(surface: str) -> tuple[Loader, Runner]:
    if surface == "sanity":
        from .sanity import (
            load_midogpp_preservation_sanity_config,
            run_midogpp_preservation_sanity,
        )

        return load_midogpp_preservation_sanity_config, run_midogpp_preservation_sanity
    if surface == "gate":
        from .gate import (
            load_midogpp_preservation_gate_config,
            run_midogpp_preservation_gate,
        )

        return load_midogpp_preservation_gate_config, run_midogpp_preservation_gate
    if surface == "condition-audit":
        from .condition_audit import (
            load_midogpp_condition_audit_config,
            run_midogpp_condition_audit,
        )

        return load_midogpp_condition_audit_config, run_midogpp_condition_audit
    if surface == "tuned-classifier":
        from .tuned_classifier import (
            load_midogpp_tuned_classifier_preservation_config,
            run_midogpp_tuned_classifier_preservation,
        )

        return (
            load_midogpp_tuned_classifier_preservation_config,
            run_midogpp_tuned_classifier_preservation,
        )
    if surface == "source-inner-prior-recovery":
        from .prior_recovery import run_source_inner_prior_recovery
        from .prior_recovery_config import (
            load_prior_recovery_config as load_source_inner_config,
        )

        return (
            lambda path: load_source_inner_config(path, expected_mode="source_inner"),
            run_source_inner_prior_recovery,
        )
    if surface == "source-inner-prior-recovery-training-seed-stability":
        from .prior_recovery import run_source_inner_training_seed_stability
        from .prior_recovery_config import (
            load_prior_recovery_config as load_stability_config,
        )

        return (
            lambda path: load_stability_config(
                path,
                expected_mode="source_inner_training_seed_stability",
            ),
            run_source_inner_training_seed_stability,
        )
    if surface == "source-inner-learned-conditional-prior-study":
        from .source_inner_studies.config import (
            LEARNED_PRIOR_MODE,
            load_source_inner_study_config,
        )
        from .source_inner_studies.prior_runner import (
            run_learned_conditional_prior_source_inner_study,
        )

        return (
            lambda path: load_source_inner_study_config(
                path,
                expected_mode=LEARNED_PRIOR_MODE,
            ),
            run_learned_conditional_prior_source_inner_study,
        )
    if surface == "source-inner-task-fisher-shrinkage-study":
        from .source_inner_studies.config import (
            FISHER_SHRINKAGE_MODE,
            load_source_inner_study_config,
        )
        from .source_inner_studies.fisher_runner import (
            run_task_fisher_shrinkage_source_inner_study,
        )

        return (
            lambda path: load_source_inner_study_config(
                path,
                expected_mode=FISHER_SHRINKAGE_MODE,
            ),
            run_task_fisher_shrinkage_source_inner_study,
        )
    if surface == "source-inner-aggregate-posterior-mixture-geco":
        from .aggregate_prior_study.config import (
            load_aggregate_prior_study_config,
        )
        from .aggregate_prior_study.runner import (
            run_aggregate_prior_source_inner_study,
        )

        return (
            load_aggregate_prior_study_config,
            run_aggregate_prior_source_inner_study,
        )
    if surface == "source-inner-uniform-b-geco-task-geometry":
        from .uniform_b_task_geometry.config import (
            load_uniform_b_task_geometry_config,
        )
        from .uniform_b_task_geometry.runner import (
            run_uniform_b_task_geometry_source_inner_study,
        )

        return (
            load_uniform_b_task_geometry_config,
            run_uniform_b_task_geometry_source_inner_study,
        )
    if surface == "prior-recovery-outer":
        from .prior_recovery import run_outer_prior_recovery
        from .prior_recovery_config import (
            load_prior_recovery_config as load_outer_config,
        )

        return (
            lambda path: load_outer_config(path, expected_mode="outer"),
            run_outer_prior_recovery,
        )
    raise AssertionError(f"Unknown preservation surface: {surface}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
