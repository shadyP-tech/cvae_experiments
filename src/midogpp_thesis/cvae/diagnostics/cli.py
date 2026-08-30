"""CLI for non-deployable CVAE diagnostic snapshots and audits."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build the import-light diagnostic command parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="surface", required=True)

    snapshot = sub.add_parser(
        "build-b-paired-reparameterization-snapshot",
        help="Build the portable canonical-B paired-replay snapshot.",
    )
    snapshot.add_argument("--config", required=True)
    snapshot.add_argument("--artifact-root", required=True)

    audit = sub.add_parser(
        "b-paired-reparameterization-audit",
        help="Run the bounded canonical-B paired reparameterization audit.",
    )
    audit.add_argument("--config", required=True)
    audit.add_argument("--artifact-root", required=True)

    residual = sub.add_parser(
        "dense-residual-router-diagnostic",
        help=(
            "Run the consumed-validation Stage-90 dense residual router "
            "diagnostic."
        ),
    )
    residual.add_argument("--config", required=True)
    residual.add_argument("--artifact-root", required=True)

    marginal_utility = sub.add_parser(
        "local-marginal-utility-router-diagnostic",
        help=(
            "Run the consumed-validation Stage-90 local marginal-utility "
            "router diagnostic."
        ),
    )
    marginal_utility.add_argument("--config", required=True)
    marginal_utility.add_argument("--artifact-root", required=True)

    mmd_kmm = sub.add_parser(
        "mmd-kmm-router-diagnostic",
        help=(
            "Run the consumed-validation Stage-90 label-free MMD/KMM "
            "mixture-router diagnostic."
        ),
    )
    mmd_kmm.add_argument("--config", required=True)
    mmd_kmm.add_argument("--artifact-root", required=True)

    conditional = sub.add_parser(
        "conditional-contrast-mmd-router-diagnostic",
        help=(
            "Run the consumed-validation Stage-90 uncertainty-gated "
            "class-conditional contrast-MMD router diagnostic."
        ),
    )
    conditional.add_argument("--config", required=True)
    conditional.add_argument("--artifact-root", required=True)

    antisymmetric = sub.add_parser(
        "antisymmetric-residual-mmd-router-diagnostic",
        help=(
            "Run the consumed-validation Stage-90 case-cross-fitted "
            "antisymmetric residual-MMD router diagnostic."
        ),
    )
    antisymmetric.add_argument("--config", required=True)
    antisymmetric.add_argument("--artifact-root", required=True)

    residual_topup = sub.add_parser(
        "residual-topup-router-diagnostic",
        help=(
            "Run the consumed-validation Stage-90 immutable equal-union "
            "backbone plus residual top-up diagnostic."
        ),
    )
    residual_topup.add_argument("--config", required=True)
    residual_topup.add_argument("--artifact-root", required=True)

    residual_topup_case_oof = sub.add_parser(
        "residual-topup-case-oof-diagnostic",
        help=(
            "Run the consumed-validation Stage-90 case-OOF B/U/G/S "
            "residual top-up decomposition diagnostic."
        ),
    )
    residual_topup_case_oof.add_argument("--config", required=True)
    residual_topup_case_oof.add_argument("--artifact-root", required=True)

    utility_aligned_exact_tail = sub.add_parser(
        "utility-aligned-exact-tail-router-diagnostic",
        help=(
            "Run the consumed-validation Stage-90 utility-aligned exact-tail "
            "router diagnostic."
        ),
    )
    utility_aligned_exact_tail.add_argument("--config", required=True)
    utility_aligned_exact_tail.add_argument("--artifact-root", required=True)

    utility_aligned_ensemble_endpoint = sub.add_parser(
        "utility-aligned-ensemble-endpoint-router-diagnostic",
        help=(
            "Run the consumed-validation Stage-90 utility-aligned exact-nine "
            "ensemble-endpoint router diagnostic."
        ),
    )
    utility_aligned_ensemble_endpoint.add_argument("--config", required=True)
    utility_aligned_ensemble_endpoint.add_argument(
        "--artifact-root", required=True
    )

    proxy_information_audit = sub.add_parser(
        "utility-aligned-ensemble-endpoint-proxy-information-audit",
        help=(
            "Run the independent consumed-validation Stage-90 exact-nine "
            "ensemble-endpoint proxy-information audit."
        ),
    )
    proxy_information_audit.add_argument("--config", required=True)
    proxy_information_audit.add_argument("--artifact-root", required=True)

    case_aware_proxy_information_audit = sub.add_parser(
        "utility-aligned-case-aware-proxy-information-audit",
        help=(
            "Run the user-authorized consumed-test Stage-90 case-aware "
            "proxy-information audit."
        ),
    )
    case_aware_proxy_information_audit.add_argument("--config", required=True)
    case_aware_proxy_information_audit.add_argument(
        "--artifact-root", required=True
    )

    fixed_bank_decision_audit = sub.add_parser(
        "fixed-bank-decision-audit",
        help=(
            "Run the consumed-test terminal fixed-bank decision diagnostic."
        ),
    )
    fixed_bank_decision_audit.add_argument("--config", required=True)
    fixed_bank_decision_audit.add_argument("--artifact-root", required=True)

    label_aware_case_oof_ceiling = sub.add_parser(
        "fixed-bank-label-aware-case-oof-ceiling",
        help=(
            "Run the consumed-test terminal fixed-bank label-aware "
            "case-OOF routing ceiling."
        ),
    )
    label_aware_case_oof_ceiling.add_argument("--config", required=True)
    label_aware_case_oof_ceiling.add_argument("--artifact-root", required=True)

    pooled_bacc_case_oof_ceiling = sub.add_parser(
        "fixed-bank-pooled-bacc-case-oof-ceiling",
        help=(
            "Run the additionally authorized consumed-test terminal fixed-bank "
            "pooled-BACC case-OOF routing ceiling."
        ),
    )
    pooled_bacc_case_oof_ceiling.add_argument("--config", required=True)
    pooled_bacc_case_oof_ceiling.add_argument("--artifact-root", required=True)

    hierarchical_residual_stacker = sub.add_parser(
        "fixed-bank-hierarchical-residual-stacker",
        help=(
            "Run the separately authorized consumed-test terminal fixed-bank "
            "hierarchical residual-stacker diagnostic."
        ),
    )
    hierarchical_residual_stacker.add_argument("--config", required=True)
    hierarchical_residual_stacker.add_argument("--artifact-root", required=True)
    hierarchical_residual_stacker.add_argument(
        "--recover-validation-only",
        action="store_true",
        help=(
            "Recover only the excluded validation controls of the exact known "
            "closed-world validator failure."
        ),
    )

    signed_error_gate = sub.add_parser(
        "fixed-bank-signed-error-gate",
        help=(
            "Run the independently authorized consumed-test terminal fixed-bank "
            "signed-error mechanism diagnostic."
        ),
    )
    signed_error_gate.add_argument("--config", required=True)
    signed_error_gate.add_argument("--artifact-root", required=True)

    actionability_recoverability = sub.add_parser(
        "fixed-bank-actionability-recoverability",
        help=(
            "Run the explicitly authorized consumed-test terminal fixed-bank "
            "actionability/recoverability mechanism diagnostic."
        ),
    )
    actionability_recoverability.add_argument("--config", required=True)
    actionability_recoverability.add_argument("--artifact-root", required=True)

    disagreement_regret_prediction_only = sub.add_parser(
        "fixed-bank-disagreement-regret-prediction-only",
        help=(
            "Run the source-OOF-trained, whole consumed-test label-free "
            "disagreement-regret prediction diagnostic."
        ),
    )
    disagreement_regret_prediction_only.add_argument("--config", required=True)
    disagreement_regret_prediction_only.add_argument(
        "--artifact-root", required=True
    )

    consumed_test_endpoint_router = sub.add_parser(
        "utility-aligned-consumed-test-endpoint-router",
        help=(
            "Run the explicitly authorized consumed-test target-static "
            "utility-aligned endpoint-router diagnostic."
        ),
    )
    consumed_test_endpoint_router.add_argument("--config", required=True)
    consumed_test_endpoint_router.add_argument("--artifact-root", required=True)

    labeled_support_case_conditional_flip_router = sub.add_parser(
        "fixed-bank-labeled-support-case-conditional-flip-router",
        help=(
            "Run the explicitly authorized consumed-test labeled-support "
            "case-conditional flip-router diagnostic."
        ),
    )
    labeled_support_case_conditional_flip_router.add_argument(
        "--config", required=True
    )
    labeled_support_case_conditional_flip_router.add_argument(
        "--artifact-root", required=True
    )

    multi_challenger_hierarchical_flip_router = sub.add_parser(
        "fixed-bank-multi-challenger-hierarchical-flip-router",
        help=(
            "Run the explicitly authorized terminal consumed-test "
            "multi-challenger hierarchical flip-router diagnostic."
        ),
    )
    multi_challenger_hierarchical_flip_router.add_argument(
        "--config", required=True
    )
    multi_challenger_hierarchical_flip_router.add_argument(
        "--artifact-root", required=True
    )

    support_static_router = sub.add_parser(
        "fixed-bank-support-static-router",
        help=(
            "Run the terminal consumed-test fixed-bank support-static S4 "
            "sensitivity diagnostic."
        ),
    )
    support_static_router.add_argument("--config", required=True)
    support_static_router.add_argument("--artifact-root", required=True)

    loo_directional_shrinkage_ensemble = sub.add_parser(
        "fixed-bank-loo-directional-shrinkage-ensemble",
        help=(
            "Run the terminal consumed-test fixed-bank whole-case LOO "
            "directional-shrinkage ensemble diagnostic."
        ),
    )
    loo_directional_shrinkage_ensemble.add_argument("--config", required=True)
    loo_directional_shrinkage_ensemble.add_argument(
        "--artifact-root", required=True
    )

    case_directional_correctness_abstention_router = sub.add_parser(
        "fixed-bank-case-directional-correctness-abstention-router",
        help=(
            "Run the terminal consumed-test held-case directional-correctness "
            "and abstention diagnostic."
        ),
    )
    case_directional_correctness_abstention_router.add_argument(
        "--config", required=True
    )
    case_directional_correctness_abstention_router.add_argument(
        "--artifact-root", required=True
    )

    opportunity_gated_dual_endpoint_router = sub.add_parser(
        "fixed-bank-loo-opportunity-gated-dual-endpoint-router",
        help=(
            "Run the terminal consumed-test fixed-bank opportunity-gated "
            "dual-endpoint probability-router diagnostic."
        ),
    )
    opportunity_gated_dual_endpoint_router.add_argument(
        "--config", required=True
    )
    opportunity_gated_dual_endpoint_router.add_argument(
        "--artifact-root", required=True
    )

    nested_donor_endpoint_regret_router = sub.add_parser(
        "fixed-bank-loo-nested-donor-endpoint-regret-router",
        help=(
            "Run the terminal consumed-test fixed-bank nested donor "
            "endpoint-regret routing diagnostic."
        ),
    )
    nested_donor_endpoint_regret_router.add_argument("--config", required=True)
    nested_donor_endpoint_regret_router.add_argument(
        "--artifact-root", required=True
    )

    pdcb = sub.add_parser(
        "fixed-bank-p-anchored-directional-crossing-bagging",
        help=(
            "Run the terminal consumed-test fixed-bank P-anchored "
            "directional crossing-bagging diagnostic."
        ),
    )
    pdcb.add_argument("--config", required=True)
    pdcb.add_argument("--artifact-root", required=True)

    pdsur = sub.add_parser(
        "fixed-bank-p-anchored-directional-signed-utility-router",
        help=(
            "Run the terminal consumed-test fixed-bank P-anchored "
            "directional signed-utility routing diagnostic."
        ),
    )
    pdsur.add_argument("--config", required=True)
    pdsur.add_argument("--artifact-root", required=True)

    pcsi = sub.add_parser(
        "fixed-bank-p-anchored-crossfit-sample-influence-router",
        help=(
            "Run the terminal consumed-test fixed-bank P-anchored "
            "cross-fit sample-influence routing diagnostic."
        ),
    )
    pcsi.add_argument("--config", required=True)
    pcsi.add_argument("--artifact-root", required=True)

    pumr = sub.add_parser(
        "fixed-bank-p-anchored-crossfit-posterior-utility-margin-router",
        help=(
            "Run the terminal consumed-test fixed-bank P-anchored five-fold "
            "posterior-utility margin routing diagnostic."
        ),
    )
    pumr.add_argument("--config", required=True)
    pumr.add_argument("--artifact-root", required=True)

    psscur = sub.add_parser(
        "fixed-bank-p-anchored-simultaneous-shift-calibrated-utility-router",
        help=(
            "Run the terminal consumed-test fixed-bank P-anchored "
            "simultaneous shift-calibrated utility routing diagnostic."
        ),
    )
    psscur.add_argument("--config", required=True)
    psscur.add_argument("--artifact-root", required=True)

    cbpupr = sub.add_parser(
        "fixed-bank-p-anchored-route-scoped-center-balanced-posterior-utility-prefix-router",
        help=(
            "Run the terminal consumed-test fixed-bank P-anchored "
            "route-scoped center-balanced posterior-utility prefix diagnostic."
        ),
    )
    cbpupr.add_argument("--config", required=True)
    cbpupr.add_argument("--artifact-root", required=True)

    cbpupr_v3 = sub.add_parser(
        "fixed-bank-p-anchored-route-scoped-center-balanced-"
        "posterior-utility-prefix-router-v3",
        help=(
            "Run the separately authorized terminal consumed-test CBPUPR v3 "
            "global-and-center surface-lineage mechanical repair."
        ),
    )
    cbpupr_v3.add_argument("--config", required=True)
    cbpupr_v3.add_argument("--artifact-root", required=True)

    pdcaps = sub.add_parser(
        "fixed-bank-p-anchored-route-scoped-donor-crossfit-"
        "action-policy-surface-router",
        help=(
            "Run the planned terminal consumed-test P-DCAPS action- and "
            "policy-surface diagnostic (execution authorization required)."
        ),
    )
    pdcaps.add_argument("--config", required=True)
    pdcaps.add_argument("--artifact-root", required=True)

    pdcaps_v2 = sub.add_parser(
        "fixed-bank-p-anchored-route-scoped-donor-crossfit-"
        "action-policy-surface-router-v2",
        help=(
            "Reject P-DCAPS v2: the terminal consumed-test run failed "
            "preterminally and its one-shot authorization is exhausted."
        ),
    )
    pdcaps_v2.add_argument("--config", required=True)
    pdcaps_v2.add_argument("--artifact-root", required=True)

    pdcaps_v3 = sub.add_parser(
        "fixed-bank-p-anchored-route-scoped-donor-crossfit-"
        "action-policy-surface-router-v3",
        help=(
            "Inspect the planned, non-authorized P-DCAPS v3 nullable-"
            "admission repair; direct execution is refused before mutation."
        ),
    )
    pdcaps_v3.add_argument("--config", required=True)
    pdcaps_v3.add_argument("--artifact-root", required=True)

    pdcaps_v4 = sub.add_parser(
        "fixed-bank-p-anchored-route-scoped-donor-crossfit-"
        "action-policy-surface-router-v4",
        help=(
            "Run the authorized one-shot P-DCAPS v4 executable nullable-"
            "admission repair on the consumed MIDOG++ test split."
        ),
    )
    pdcaps_v4.add_argument("--config", required=True)
    pdcaps_v4.add_argument("--artifact-root", required=True)

    scale_bp_v2 = sub.add_parser(
        "fixed-bank-p-anchored-support-calibrated-local-action-empirical-bayes-"
        "boundary-projected-router-v2",
        help=(
            "Run the explicitly authorized single-use SCALE-BP v2 terminal "
            "consumed-test diagnostic, or its mutation-free no-label dry run."
        ),
    )
    scale_bp_v2.add_argument("--config", required=True)
    scale_bp_v2.add_argument("--artifact-root", required=True)
    scale_bp_v2.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate source and direct-original inputs without consuming authorization.",
    )

    scale_bp = sub.add_parser(
        "fixed-bank-p-anchored-support-calibrated-local-action-empirical-bayes-"
        "boundary-projected-router",
        help=(
            "Inspect the planned, non-authorized SCALE-BP v1 implementation; "
            "direct execution is refused before mutation."
        ),
    )
    scale_bp.add_argument("--config", required=True)
    scale_bp.add_argument("--artifact-root", required=True)

    opportunity_pairwise = sub.add_parser(
        "fixed-bank-p-anchored-opportunity-equivalence-pairwise-primitive-"
        "utility-router-v1",
        help=(
            "Inspect the planned, non-authorized opportunity-equivalence "
            "pairwise primitive-utility router; direct execution is refused "
            "before mutation."
        ),
    )
    opportunity_pairwise.add_argument("--config", required=True)
    opportunity_pairwise.add_argument("--artifact-root", required=True)

    opportunity_pairwise_v2 = sub.add_parser(
        "fixed-bank-p-anchored-opportunity-equivalence-pairwise-primitive-"
        "utility-router-v2",
        help=(
            "Inspect the separately identified OE-PPUR v2 executable "
            "successor. Its checked-in config remains non-authorized and "
            "cannot claim the single-use lease or open terminal labels."
        ),
    )
    opportunity_pairwise_v2.add_argument("--config", required=True)
    opportunity_pairwise_v2.add_argument("--artifact-root", required=True)
    opportunity_pairwise_v2.add_argument("--scratch-root")
    opportunity_pairwise_v2.add_argument(
        "--inspect-plan",
        action="store_true",
        help="Emit the mutation-free six-input/matrix implementation receipt.",
    )

    opportunity_pairwise_v3 = sub.add_parser(
        "fixed-bank-p-anchored-opportunity-equivalence-pairwise-primitive-"
        "utility-router-v3",
        help=(
            "Inspect the planned OE-PPUR v3 seven-input source-supervised "
            "successor. Its source bundle is not materialized, its amendment "
            "is absent, and direct execution is refused before mutation."
        ),
    )
    opportunity_pairwise_v3.add_argument("--config", required=True)
    opportunity_pairwise_v3.add_argument("--artifact-root", required=True)
    opportunity_pairwise_v3.add_argument("--scratch-root")
    opportunity_pairwise_v3.add_argument(
        "--inspect-plan",
        action="store_true",
        help="Emit the mutation-free seven-input scientific implementation receipt.",
    )

    sceptre = sub.add_parser(
        "fixed-bank-sceptre-router-v1",
        help=(
            "Inspect the scoped SCEPTRE post-hoc development architecture; "
            "consumed-test execution is refused before mutation."
        ),
    )
    sceptre.add_argument("--config", required=True)
    sceptre.add_argument("--artifact-root", required=True)

    sceptre_v2 = sub.add_parser(
        "fixed-bank-sceptre-router-v2",
        help=(
            "Run the explicitly authorized single-use SCEPTRE v2 downstream-"
            "utility consumed-test sensitivity diagnostic, or its mutation-"
            "free dry run; this is not NELBO routing evidence."
        ),
    )
    sceptre_v2.add_argument("--config", required=True)
    sceptre_v2.add_argument("--artifact-root", required=True)
    sceptre_v2.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate source, exact-eight inputs, workspace binding, and host "
            "admission without claiming authorization or opening labels."
        ),
    )

    sceptre_v3 = sub.add_parser(
        "fixed-bank-sceptre-router-v3",
        help=(
            "Run the separately authorized SCEPTRE v3 persistent-worker "
            "repair diagnostic, or its pre-lease GPU-worker dry run. The "
            "consumed-test result remains terminal and descriptive only."
        ),
    )
    sceptre_v3.add_argument("--config", required=True)
    sceptre_v3.add_argument("--artifact-root", required=True)
    sceptre_v3.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate exact-eight inputs, source seal, workstation, and two "
            "repeated tasks per exact GPU initializer without claiming the "
            "v3 authorization lease or opening labels."
        ),
    )

    sceptre_v4 = sub.add_parser(
        "fixed-bank-sceptre-router-v4",
        help=(
            "Run the separately authorized, single-use SCEPTRE v4 candidate-"
            "set consumed-test diagnostic, its read-only dry run, or inspect "
            "its executable identities."
        ),
    )
    sceptre_v4.add_argument("--config", required=True)
    sceptre_v4.add_argument(
        "--artifact-root",
        default=".",
        help="Prepared output root; ignored by the path-free inspection surface.",
    )
    sceptre_v4_mode = sceptre_v4.add_mutually_exclusive_group()
    sceptre_v4_mode.add_argument(
        "--inspect-plan",
        action="store_true",
        help=(
            "Emit the mutation-free executable/source receipt without "
            "resolving inputs, probing hardware, or opening target data."
        ),
    )
    sceptre_v4_mode.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate exact-eight inputs, source seal, workspace binding, "
            "workstation capacity, and the exact GPU worker initializer "
            "without claiming the single-use lease or opening labels."
        ),
    )

    sceptre_v5 = sub.add_parser(
        "fixed-bank-sceptre-router-v5",
        help=(
            "Run the separately authorized, single-use SCEPTRE v5 fit-"
            "semantics repair diagnostic, its read-only dry run, or inspect "
            "its executable identities."
        ),
    )
    sceptre_v5.add_argument("--config", required=True)
    sceptre_v5.add_argument(
        "--artifact-root",
        default=".",
        help="Prepared output root; ignored by the path-free inspection surface.",
    )
    sceptre_v5_mode = sceptre_v5.add_mutually_exclusive_group()
    sceptre_v5_mode.add_argument(
        "--inspect-plan",
        action="store_true",
        help=(
            "Emit the mutation-free executable/source receipt without "
            "resolving inputs, probing hardware, or opening target data."
        ),
    )

    harp_stage90 = sub.add_parser(
        "fixed-bank-harp-router-v1",
        help=(
            "Inspect, dry-run, or execute the separately authorized terminal "
            "HARP consumed-test sensitivity. The physical probability menu is "
            "rebuilt from the frozen bank, GenerationLock, and label-blind cache."
        ),
    )
    harp_stage90.add_argument("--config", required=True)
    harp_stage90.add_argument(
        "--artifact-root",
        default=".",
        help="Prepared output root; ignored by path-free planned inspection.",
    )
    harp_stage90_mode = harp_stage90.add_mutually_exclusive_group()
    harp_stage90_mode.add_argument(
        "--inspect-plan",
        action="store_true",
        help="Inspect identities and complete physical topology without resolving inputs.",
    )
    harp_stage90_mode.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate authorized inputs and the physical plan without claiming "
            "the HARP lease, generating sources, or opening labels."
        ),
    )
    harp_prepare = sub.add_parser(
        "prepare-fixed-bank-harp-router-v1-inputs",
        help=(
            "Materialize the HARP-only consumed-test label-blind cache and "
            "deterministic whole-case role manifests. This does not create "
            "or activate execution authority."
        ),
    )
    harp_prepare.add_argument("--canonical-cache-root", required=True)
    harp_prepare.add_argument("--canonical-manifest", required=True)
    harp_prepare.add_argument("--parent-ledger", required=True)
    harp_prepare.add_argument("--cache-root", required=True)
    harp_prepare.add_argument("--development-manifest", required=True)
    harp_prepare.add_argument("--evaluation-manifest", required=True)
    harp_publish = sub.add_parser(
        "publish-fixed-bank-harp-router-v1-amendment",
        help=(
            "Independently validate the exact prepared HARP inputs and issue "
            "the HARP-only terminal amendment once. This does not activate "
            "registration, claim the lease, create output, or launch HARP."
        ),
    )
    harp_publish.add_argument("--config", required=True)
    harp_publish.add_argument("--expert-bank-root", required=True)
    harp_publish.add_argument("--generation-lock-root", required=True)
    harp_publish.add_argument("--prepared-cache-root", required=True)
    harp_publish.add_argument("--development-manifest", required=True)
    harp_publish.add_argument("--evaluation-manifest", required=True)
    harp_publish.add_argument("--parent-ledger", required=True)
    harp_publish.add_argument("--amendment-path", required=True)
    harp_publish.add_argument("--authorization-basis", required=True)
    harp_publish.add_argument("--authorization-date", required=True)
    harp_publish.add_argument(
        "--repository-root",
        required=True,
        help=(
            "Exact checkout root that owns the registered amendment member and "
            "the transitive source snapshot."
        ),
    )
    sceptre_v5_mode.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate exact-eight inputs, source seal, workspace binding, "
            "workstation capacity, and the exact GPU worker initializer "
            "without claiming the single-use lease or opening labels."
        ),
    )

    pcsi_parc = sub.add_parser(
        "fixed-bank-p-anchored-boundary-projected-pcsi-policy-regret-router",
        help=(
            "Run the terminal consumed-test fixed-bank P-anchored "
            "boundary-projected PCSI whole-policy-regret diagnostic."
        ),
    )
    pcsi_parc.add_argument("--config", required=True)
    pcsi_parc.add_argument("--artifact-root", required=True)

    pcsi_racr = sub.add_parser(
        "fixed-bank-p-anchored-route-scoped-boundary-projected-pcsi-policy-regret-router",
        help=(
            "Run the terminal consumed-test route-scoped P-anchored "
            "boundary-projected PCSI case-regret diagnostic."
        ),
    )
    pcsi_racr.add_argument("--config", required=True)
    pcsi_racr.add_argument("--artifact-root", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.surface == "prepare-fixed-bank-harp-router-v1-inputs":
        import json

        from .fixed_bank_harp_router_v1.preparation import (
            prepare_harp_consumed_test_inputs,
        )

        prepared = prepare_harp_consumed_test_inputs(
            canonical_cache_root=args.canonical_cache_root,
            canonical_manifest_path=args.canonical_manifest,
            parent_ledger_path=args.parent_ledger,
            cache_root=args.cache_root,
            development_manifest_path=args.development_manifest,
            evaluation_manifest_path=args.evaluation_manifest,
        )
        print(json.dumps(prepared.to_payload(), sort_keys=True, separators=(",", ":")))
        return 0
    if args.surface == "publish-fixed-bank-harp-router-v1-amendment":
        import json

        from .fixed_bank_harp_router_v1.amendment_publisher import (
            publish_harp_execution_amendment,
        )
        from .fixed_bank_harp_router_v1.config import load_config

        receipt = publish_harp_execution_amendment(
            load_config(args.config),
            expert_bank_root=args.expert_bank_root,
            generation_lock_root=args.generation_lock_root,
            prepared_cache_root=args.prepared_cache_root,
            development_manifest_path=args.development_manifest,
            evaluation_manifest_path=args.evaluation_manifest,
            parent_ledger_path=args.parent_ledger,
            amendment_path=args.amendment_path,
            authorization_basis=args.authorization_basis,
            authorization_date=args.authorization_date,
            repository_root=args.repository_root,
        )
        print(json.dumps(receipt.to_payload(), sort_keys=True, separators=(",", ":")))
        return 0
    artifact_root = Path(args.artifact_root)

    if args.surface == (
        "fixed-bank-p-anchored-support-calibrated-local-action-empirical-bayes-"
        "boundary-projected-router-v2"
    ):
        import json

        from .fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.config import (
            load_config,
        )
        from .fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2.runner import (
            dry_run_scale_bp_v2,
            run_scale_bp_v2,
        )

        config = load_config(args.config)
        if args.dry_run:
            print(
                json.dumps(
                    dry_run_scale_bp_v2(config, artifact_root=artifact_root),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        else:
            print(run_scale_bp_v2(config, artifact_root=artifact_root))
        return 0

    if args.surface == (
        "fixed-bank-p-anchored-support-calibrated-local-action-empirical-bayes-"
        "boundary-projected-router"
    ):
        from .fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.config import (
            load_support_calibrated_local_action_empirical_bayes_boundary_projected_router_config,
        )
        from .fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.runner import (
            run_support_calibrated_local_action_empirical_bayes_boundary_projected_router,
        )

        config = (
            load_support_calibrated_local_action_empirical_bayes_boundary_projected_router_config(
                args.config
            )
        )
        output = (
            run_support_calibrated_local_action_empirical_bayes_boundary_projected_router(
                config,
                artifact_root=artifact_root,
            )
        )
        print(output)
        return 0

    if args.surface == (
        "fixed-bank-p-anchored-opportunity-equivalence-pairwise-primitive-"
        "utility-router-v1"
    ):
        from .fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.config import (
            load_config,
        )
        from .fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v1.runner import (
            run_planned_router,
        )

        config = load_config(args.config)
        output = run_planned_router(config, artifact_root=artifact_root)
        print(output)
        return 0

    if args.surface == (
        "fixed-bank-p-anchored-opportunity-equivalence-pairwise-primitive-"
        "utility-router-v2"
    ):
        import json

        from ..protocol import ProtocolError

        from .fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.config import (
            load_config,
            load_resolved_config,
        )
        from .fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.runner import (
            inspect_planned_router,
            run_oe_ppur_v2,
        )

        if args.inspect_plan:
            config = load_config(args.config)
            print(
                json.dumps(
                    inspect_planned_router(config),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0
        config_path = Path(args.config)
        if config_path.name == "config.resolved.yaml":
            resolved = load_resolved_config(config_path)
            if Path(artifact_root) != resolved.artifact_root:
                raise ProtocolError(
                    "OE-PPUR v2 CLI artifact root differs from config.resolved.yaml."
                )
            run_input = resolved
        else:
            # The checked-in path-free config reaches the public runner only to
            # reject before source, service, input, output, or scratch access.
            run_input = load_config(config_path)
        output = run_oe_ppur_v2(
            run_input,
            scratch_root=(
                args.scratch_root
                if args.scratch_root is not None
                else (
                    "/data/local/fixed_bank_p_anchored_opportunity_"
                    "equivalence_pairwise_primitive_utility_router_v2"
                )
            ),
        )
        print(output)
        return 0

    if args.surface == (
        "fixed-bank-p-anchored-opportunity-equivalence-pairwise-primitive-"
        "utility-router-v3"
    ):
        import json

        from ..protocol import ProtocolError

        from .fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.config import (
            load_config,
            load_resolved_config,
        )
        from .fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.runner import (
            inspect_planned_router,
            run_oe_ppur_v3,
        )

        config_path = Path(args.config)
        if config_path.name == "config.resolved.yaml":
            if args.inspect_plan:
                raise ProtocolError(
                    "OE-PPUR v3 inspection requires the path-free planned config."
                )
            resolved = load_resolved_config(config_path)
            if Path(artifact_root) != resolved.artifact_root:
                raise ProtocolError(
                    "OE-PPUR v3 CLI artifact root differs from config.resolved.yaml."
                )
            run_input = resolved
        else:
            run_input = load_config(config_path)
        if args.inspect_plan:
            print(
                json.dumps(
                    inspect_planned_router(run_input),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0
        output = run_oe_ppur_v3(
            run_input,
            artifact_root=artifact_root,
            scratch_root=(
                args.scratch_root
                if args.scratch_root is not None
                else (
                    "/data/local/fixed_bank_p_anchored_opportunity_"
                    "equivalence_pairwise_primitive_utility_router_v3"
                )
            ),
        )
        print(output)
        return 0

    if args.surface == "fixed-bank-sceptre-router-v1":
        from .fixed_bank_sceptre_router.config import load_config
        from .fixed_bank_sceptre_router.runner import run_planned_sceptre_router

        config = load_config(args.config)
        output = run_planned_sceptre_router(config, artifact_root=artifact_root)
        print(output)
        return 0

    if args.surface == "fixed-bank-sceptre-router-v2":
        import json

        from .fixed_bank_sceptre_router_v2.config import load_config
        from .fixed_bank_sceptre_router_v2.runner import (
            dry_run_sceptre_v2,
            run_sceptre_v2,
        )

        config = load_config(args.config)
        if args.dry_run:
            print(
                json.dumps(
                    dry_run_sceptre_v2(config, artifact_root=artifact_root),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        else:
            print(run_sceptre_v2(config, artifact_root=artifact_root))
        return 0

    if args.surface == "fixed-bank-sceptre-router-v3":
        import json

        from .fixed_bank_sceptre_router_v3.config import load_config
        from .fixed_bank_sceptre_router_v3.runner import (
            dry_run_sceptre_v3,
            run_sceptre_v3,
        )

        config = load_config(args.config)
        if args.dry_run:
            print(
                json.dumps(
                    dry_run_sceptre_v3(config, artifact_root=artifact_root),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        else:
            print(run_sceptre_v3(config, artifact_root=artifact_root))
        return 0

    if args.surface == "fixed-bank-sceptre-router-v4":
        import json

        from .fixed_bank_sceptre_router_v4.config import load_config
        from .fixed_bank_sceptre_router_v4.runner import (
            dry_run_sceptre_v4,
            inspect_planned_sceptre_v4,
            run_sceptre_v4,
        )

        config = load_config(args.config)
        if args.inspect_plan:
            print(
                json.dumps(
                    inspect_planned_sceptre_v4(config),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        elif args.dry_run:
            print(
                json.dumps(
                    dry_run_sceptre_v4(config, artifact_root=artifact_root),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        else:
            print(run_sceptre_v4(config, artifact_root=artifact_root))
        return 0

    if args.surface == "fixed-bank-sceptre-router-v5":
        import json

        from .fixed_bank_sceptre_router_v5.config import load_config
        from .fixed_bank_sceptre_router_v5.runner import (
            dry_run_sceptre_v5,
            inspect_planned_sceptre_v5,
            run_sceptre_v5,
        )

        config = load_config(args.config)
        if args.inspect_plan:
            print(
                json.dumps(
                    inspect_planned_sceptre_v5(config),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        elif args.dry_run:
            print(
                json.dumps(
                    dry_run_sceptre_v5(config, artifact_root=artifact_root),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        else:
            print(run_sceptre_v5(config, artifact_root=artifact_root))
        return 0

    if args.surface == "fixed-bank-harp-router-v1":
        import json

        from .fixed_bank_harp_router_v1.config import load_config
        from .fixed_bank_harp_router_v1.runner import (
            dry_run_harp_stage90,
            inspect_harp_stage90,
            run_harp_stage90,
        )

        config = load_config(args.config)
        if args.inspect_plan:
            print(
                json.dumps(
                    inspect_harp_stage90(config),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        elif args.dry_run:
            print(
                json.dumps(
                    dry_run_harp_stage90(config, artifact_root=artifact_root),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        else:
            print(run_harp_stage90(config, artifact_root=artifact_root))
        return 0

    if args.surface == (
        "fixed-bank-p-anchored-route-scoped-donor-crossfit-"
        "action-policy-surface-router-v4"
    ):
        from .fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v4.config import (
            load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v4_config,
        )
        from .fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v4.runner import (
            run_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v4,
        )

        config = (
            load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v4_config(
                args.config
            )
        )
        output = (
            run_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v4(
                config,
                artifact_root=artifact_root,
            )
        )
        print(output)
        return 0

    if args.surface == (
        "fixed-bank-p-anchored-route-scoped-donor-crossfit-"
        "action-policy-surface-router-v3"
    ):
        from .fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3 import (
            load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3_config,
            run_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3,
        )

        config = (
            load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3_config(
                args.config
            )
        )
        output = (
            run_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3(
                config,
                artifact_root=artifact_root,
            )
        )
        print(output)
        return 0

    if args.surface == (
        "fixed-bank-p-anchored-route-scoped-donor-crossfit-"
        "action-policy-surface-router-v2"
    ):
        from ..protocol import ProtocolError

        raise ProtocolError(
            "P-DCAPS v2 is terminally failed and its one-shot authorization "
            "is exhausted; recovery and rerun are forbidden."
        )

    if args.surface == (
        "fixed-bank-p-anchored-route-scoped-donor-crossfit-"
        "action-policy-surface-router"
    ):
        from .fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router import (
            load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_config,
            run_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router,
        )

        config = (
            load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_config(
                args.config
            )
        )
        output = run_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router(
            config,
            artifact_root=artifact_root,
        )
        print(output)
        return 0

    if args.surface == (
        "fixed-bank-p-anchored-route-scoped-center-balanced-"
        "posterior-utility-prefix-router-v3"
    ):
        from .fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_v3 import (
            load_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_config,
            run_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router,
        )

        config = (
            load_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_config(
                args.config
            )
        )
        output = (
            run_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router(
                config,
                artifact_root=artifact_root,
            )
        )
        print(output)
        return 0

    if args.surface == (
        "fixed-bank-p-anchored-route-scoped-center-balanced-"
        "posterior-utility-prefix-router"
    ):
        from .fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router import (
            load_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_config,
            run_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router,
        )

        config = (
            load_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_config(
                args.config
            )
        )
        output = (
            run_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router(
                config,
                artifact_root=artifact_root,
            )
        )
        print(output)
        return 0

    if args.surface == (
        "fixed-bank-p-anchored-route-scoped-boundary-projected-"
        "pcsi-policy-regret-router"
    ):
        from .fixed_bank_p_anchored_route_scoped_boundary_projected_pcsi_policy_regret_router import (
            load_p_anchored_route_scoped_boundary_projected_pcsi_policy_regret_router_config,
            run_p_anchored_route_scoped_boundary_projected_pcsi_policy_regret_router,
        )

        config = (
            load_p_anchored_route_scoped_boundary_projected_pcsi_policy_regret_router_config(
                args.config
            )
        )
        output = (
            run_p_anchored_route_scoped_boundary_projected_pcsi_policy_regret_router(
                config,
                artifact_root=artifact_root,
            )
        )
        print(output)
        return 0

    if (
        args.surface
        == "fixed-bank-p-anchored-boundary-projected-pcsi-policy-regret-router"
    ):
        from .fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router import (
            load_p_anchored_boundary_projected_pcsi_policy_regret_router_config,
            run_p_anchored_boundary_projected_pcsi_policy_regret_router,
        )

        config = (
            load_p_anchored_boundary_projected_pcsi_policy_regret_router_config(
                args.config
            )
        )
        output = run_p_anchored_boundary_projected_pcsi_policy_regret_router(
            config,
            artifact_root=artifact_root,
        )
        print(output)
        return 0

    if args.surface == "build-b-paired-reparameterization-snapshot":
        from .b_paired_reparameterization_audit import build_snapshot_from_config

        output = build_snapshot_from_config(
            args.config,
            artifact_root=artifact_root,
        )
        print(output)
        return 0

    if (
        args.surface
        == "fixed-bank-p-anchored-simultaneous-shift-calibrated-utility-router"
    ):
        from .fixed_bank_p_anchored_simultaneous_shift_calibrated_utility_router import (
            load_p_anchored_simultaneous_shift_calibrated_utility_router_config,
            run_p_anchored_simultaneous_shift_calibrated_utility_router,
        )

        config = load_p_anchored_simultaneous_shift_calibrated_utility_router_config(
            args.config
        )
        output = run_p_anchored_simultaneous_shift_calibrated_utility_router(
            config,
            artifact_root=artifact_root,
        )
        print(output)
        return 0

    if args.surface == "b-paired-reparameterization-audit":
        from .b_paired_reparameterization_audit import (
            load_audit_config,
            run_b_paired_reparameterization_audit,
        )

        config = load_audit_config(args.config)
        output = run_b_paired_reparameterization_audit(
            config,
            artifact_root=artifact_root,
            resolved_config_path=args.config,
        )
        print(output)
        return 0

    if args.surface == "dense-residual-router-diagnostic":
        from .dense_residual_router.config import (
            load_dense_residual_diagnostic_config,
        )
        from .dense_residual_router.runner import (
            run_dense_residual_router_diagnostic,
        )

        config = load_dense_residual_diagnostic_config(args.config)
        output = run_dense_residual_router_diagnostic(
            config,
            artifact_root=artifact_root,
        )
        print(output)
        return 0

    if args.surface == "local-marginal-utility-router-diagnostic":
        from .local_marginal_utility_router.config import (
            load_local_marginal_utility_router_config,
        )
        from .local_marginal_utility_router.runner import (
            run_local_marginal_utility_router_diagnostic,
        )

        config = load_local_marginal_utility_router_config(args.config)
        output = run_local_marginal_utility_router_diagnostic(
            config,
            artifact_root=artifact_root,
        )
        print(output)
        return 0

    if args.surface == "mmd-kmm-router-diagnostic":
        from .mmd_kmm_router.config import load_mmd_kmm_router_config
        from .mmd_kmm_router.runner import run_mmd_kmm_router_diagnostic

        config = load_mmd_kmm_router_config(args.config)
        output = run_mmd_kmm_router_diagnostic(
            config,
            artifact_root=artifact_root,
        )
        print(output)
        return 0

    if args.surface == "conditional-contrast-mmd-router-diagnostic":
        from .conditional_contrast_mmd_router.config import (
            load_conditional_contrast_mmd_router_config,
        )
        from .conditional_contrast_mmd_router.runner import (
            run_conditional_contrast_mmd_router_diagnostic,
        )

        config = load_conditional_contrast_mmd_router_config(args.config)
        output = run_conditional_contrast_mmd_router_diagnostic(
            config,
            artifact_root=artifact_root,
        )
        print(output)
        return 0

    if args.surface == "antisymmetric-residual-mmd-router-diagnostic":
        from .antisymmetric_residual_mmd_router.config import (
            load_antisymmetric_residual_mmd_config,
        )
        from .antisymmetric_residual_mmd_router.runner import (
            run_antisymmetric_residual_mmd_router_diagnostic,
        )

        config = load_antisymmetric_residual_mmd_config(args.config)
        output = run_antisymmetric_residual_mmd_router_diagnostic(
            config,
            artifact_root=artifact_root,
        )
        print(output)
        return 0

    if args.surface == "residual-topup-router-diagnostic":
        from .residual_topup_router.config import load_residual_topup_config
        from .residual_topup_router.runner import (
            run_residual_topup_router_diagnostic,
        )

        config = load_residual_topup_config(args.config)
        output = run_residual_topup_router_diagnostic(
            config,
            artifact_root=artifact_root,
        )
        print(output)
        return 0

    if args.surface == "residual-topup-case-oof-diagnostic":
        from .residual_topup_case_oof import (
            load_residual_topup_case_oof_config,
            run_residual_topup_case_oof_diagnostic,
        )

        config = load_residual_topup_case_oof_config(args.config)
        output = run_residual_topup_case_oof_diagnostic(
            config,
            artifact_root=artifact_root,
        )
        print(output)
        return 0

    if args.surface == "utility-aligned-exact-tail-router-diagnostic":
        from .utility_aligned_exact_tail_router import (
            load_utility_aligned_exact_tail_router_config,
            run_utility_aligned_exact_tail_router_diagnostic,
        )

        config = load_utility_aligned_exact_tail_router_config(args.config)
        output = run_utility_aligned_exact_tail_router_diagnostic(
            config,
            artifact_root=artifact_root,
        )
        print(output)
        return 0

    if args.surface == "utility-aligned-ensemble-endpoint-router-diagnostic":
        from .utility_aligned_ensemble_endpoint_router import (
            load_utility_aligned_ensemble_endpoint_router_config,
            run_utility_aligned_ensemble_endpoint_router_diagnostic,
        )

        config = load_utility_aligned_ensemble_endpoint_router_config(args.config)
        output = run_utility_aligned_ensemble_endpoint_router_diagnostic(
            config,
            artifact_root=artifact_root,
        )
        print(output)
        return 0

    if (
        args.surface
        == "utility-aligned-ensemble-endpoint-proxy-information-audit"
    ):
        from .utility_aligned_ensemble_endpoint_proxy_information_audit import (
            load_utility_aligned_ensemble_endpoint_proxy_information_audit_config,
            run_utility_aligned_ensemble_endpoint_proxy_information_audit,
        )

        config = (
            load_utility_aligned_ensemble_endpoint_proxy_information_audit_config(
                args.config
            )
        )
        output = run_utility_aligned_ensemble_endpoint_proxy_information_audit(
            config,
            artifact_root=artifact_root,
        )
        print(output)
        return 0

    if args.surface == "utility-aligned-case-aware-proxy-information-audit":
        from .utility_aligned_case_aware_proxy_information_audit import (
            load_utility_aligned_case_aware_proxy_information_audit_config,
            run_utility_aligned_case_aware_proxy_information_audit,
        )

        config = load_utility_aligned_case_aware_proxy_information_audit_config(
            args.config
        )
        output = run_utility_aligned_case_aware_proxy_information_audit(
            config,
            artifact_root=artifact_root,
        )
        print(output)
        return 0

    if args.surface == "fixed-bank-decision-audit":
        from .fixed_bank_decision_audit import (
            load_fixed_bank_decision_audit_config,
            run_fixed_bank_decision_audit,
        )

        config = load_fixed_bank_decision_audit_config(args.config)
        output = run_fixed_bank_decision_audit(
            config,
            artifact_root=artifact_root,
        )
        print(output)
        return 0

    if args.surface == "fixed-bank-label-aware-case-oof-ceiling":
        from .fixed_bank_label_aware_case_oof_ceiling import (
            load_fixed_bank_label_aware_case_oof_ceiling_config,
            run_fixed_bank_label_aware_case_oof_ceiling,
        )

        config = load_fixed_bank_label_aware_case_oof_ceiling_config(args.config)
        output = run_fixed_bank_label_aware_case_oof_ceiling(
            config,
            artifact_root=artifact_root,
        )
        print(output)
        return 0

    if args.surface == "fixed-bank-pooled-bacc-case-oof-ceiling":
        from .fixed_bank_pooled_bacc_case_oof_ceiling import (
            load_fixed_bank_pooled_bacc_case_oof_ceiling_config,
            run_fixed_bank_pooled_bacc_case_oof_ceiling,
        )

        config = load_fixed_bank_pooled_bacc_case_oof_ceiling_config(args.config)
        output = run_fixed_bank_pooled_bacc_case_oof_ceiling(
            config,
            artifact_root=artifact_root,
        )
        print(output)
        return 0

    if args.surface == "fixed-bank-hierarchical-residual-stacker":
        from .fixed_bank_hierarchical_residual_stacker import (
            load_fixed_bank_hierarchical_residual_stacker_config,
        )

        config = load_fixed_bank_hierarchical_residual_stacker_config(args.config)
        if args.recover_validation_only:
            from .fixed_bank_hierarchical_residual_stacker import (
                recover_fixed_bank_hierarchical_residual_stacker_validation,
            )

            output = recover_fixed_bank_hierarchical_residual_stacker_validation(
                config,
                artifact_root=artifact_root,
            )
        else:
            from .fixed_bank_hierarchical_residual_stacker import (
                run_fixed_bank_hierarchical_residual_stacker,
            )

            output = run_fixed_bank_hierarchical_residual_stacker(
                config,
                artifact_root=artifact_root,
            )
        print(output)
        return 0

    if args.surface == "fixed-bank-signed-error-gate":
        from .fixed_bank_signed_error_gate import (
            load_fixed_bank_signed_error_gate_config,
            run_fixed_bank_signed_error_gate,
        )

        config = load_fixed_bank_signed_error_gate_config(args.config)
        output = run_fixed_bank_signed_error_gate(
            config,
            artifact_root=artifact_root,
        )
        print(output)
        return 0

    if args.surface == "fixed-bank-actionability-recoverability":
        from .fixed_bank_actionability_recoverability import (
            load_fixed_bank_actionability_recoverability_config,
            run_fixed_bank_actionability_recoverability,
        )

        config = load_fixed_bank_actionability_recoverability_config(args.config)
        output = run_fixed_bank_actionability_recoverability(
            config,
            artifact_root=artifact_root,
        )
        print(output)
        return 0

    if args.surface == "fixed-bank-disagreement-regret-prediction-only":
        from .fixed_bank_disagreement_regret_prediction_only import (
            load_fixed_bank_disagreement_regret_prediction_only_config,
            run_fixed_bank_disagreement_regret_prediction_only,
        )

        config = load_fixed_bank_disagreement_regret_prediction_only_config(
            args.config
        )
        output = run_fixed_bank_disagreement_regret_prediction_only(
            config,
            artifact_root=artifact_root,
        )
        print(output)
        return 0

    if args.surface == "utility-aligned-consumed-test-endpoint-router":
        from .utility_aligned_consumed_test_endpoint_router import (
            load_utility_aligned_consumed_test_endpoint_router_config,
            run_utility_aligned_consumed_test_endpoint_router,
        )

        config = load_utility_aligned_consumed_test_endpoint_router_config(
            args.config
        )
        output = run_utility_aligned_consumed_test_endpoint_router(
            config,
            artifact_root=artifact_root,
        )
        print(output)
        return 0

    if args.surface == "fixed-bank-labeled-support-case-conditional-flip-router":
        from .fixed_bank_labeled_support_case_conditional_flip_router import (
            load_fixed_bank_labeled_support_case_conditional_flip_router_config,
            run_fixed_bank_labeled_support_case_conditional_flip_router,
        )

        config = load_fixed_bank_labeled_support_case_conditional_flip_router_config(
            args.config
        )
        output = run_fixed_bank_labeled_support_case_conditional_flip_router(
            config,
            artifact_root=artifact_root,
        )
        print(output)
        return 0

    if args.surface == "fixed-bank-multi-challenger-hierarchical-flip-router":
        from .fixed_bank_multi_challenger_hierarchical_flip_router import (
            load_fixed_bank_multi_challenger_hierarchical_flip_router_config,
            run_fixed_bank_multi_challenger_hierarchical_flip_router,
        )

        config = load_fixed_bank_multi_challenger_hierarchical_flip_router_config(
            args.config
        )
        output = run_fixed_bank_multi_challenger_hierarchical_flip_router(
            config,
            artifact_root=artifact_root,
        )
        print(output)
        return 0

    if args.surface == "fixed-bank-support-static-router":
        from .fixed_bank_support_static_router import (
            load_fixed_bank_support_static_router_config,
            run_fixed_bank_support_static_router,
        )

        config = load_fixed_bank_support_static_router_config(args.config)
        output = run_fixed_bank_support_static_router(
            config,
            artifact_root=artifact_root,
        )
        print(output)
        return 0

    if args.surface == "fixed-bank-loo-directional-shrinkage-ensemble":
        from .fixed_bank_loo_directional_shrinkage_ensemble import (
            load_fixed_bank_loo_directional_shrinkage_ensemble_config,
            run_fixed_bank_loo_directional_shrinkage_ensemble,
        )

        config = load_fixed_bank_loo_directional_shrinkage_ensemble_config(
            args.config
        )
        output = run_fixed_bank_loo_directional_shrinkage_ensemble(
            config,
            artifact_root=artifact_root,
        )
        print(output)
        return 0

    if args.surface == "fixed-bank-case-directional-correctness-abstention-router":
        from .fixed_bank_case_directional_correctness_abstention_router import (
            load_fixed_bank_case_directional_correctness_abstention_router_config,
            run_fixed_bank_case_directional_correctness_abstention_router,
        )

        config = (
            load_fixed_bank_case_directional_correctness_abstention_router_config(
                args.config
            )
        )
        output = run_fixed_bank_case_directional_correctness_abstention_router(
            config,
            artifact_root=artifact_root,
        )
        print(output)
        return 0

    if args.surface == "fixed-bank-loo-opportunity-gated-dual-endpoint-router":
        from .fixed_bank_loo_opportunity_gated_dual_endpoint_router import (
            load_fixed_bank_loo_opportunity_gated_dual_endpoint_router_config,
            run_fixed_bank_loo_opportunity_gated_dual_endpoint_router,
        )

        config = load_fixed_bank_loo_opportunity_gated_dual_endpoint_router_config(
            args.config
        )
        output = run_fixed_bank_loo_opportunity_gated_dual_endpoint_router(
            config,
            artifact_root=artifact_root,
        )
        print(output)
        return 0

    if args.surface == "fixed-bank-loo-nested-donor-endpoint-regret-router":
        from .fixed_bank_loo_nested_donor_endpoint_regret_router import (
            load_nested_donor_endpoint_regret_config,
            run_nested_donor_endpoint_regret_router,
        )

        config = load_nested_donor_endpoint_regret_config(args.config)
        output = run_nested_donor_endpoint_regret_router(
            config,
            artifact_root=artifact_root,
        )
        print(output)
        return 0

    if args.surface == "fixed-bank-p-anchored-directional-crossing-bagging":
        from .fixed_bank_p_anchored_directional_crossing_bagging import (
            load_p_anchored_directional_crossing_bagging_config,
            run_p_anchored_directional_crossing_bagging,
        )

        config = load_p_anchored_directional_crossing_bagging_config(args.config)
        output = run_p_anchored_directional_crossing_bagging(
            config,
            artifact_root=artifact_root,
        )
        print(output)
        return 0

    if args.surface == "fixed-bank-p-anchored-directional-signed-utility-router":
        from .fixed_bank_p_anchored_directional_signed_utility_router import (
            load_p_anchored_directional_signed_utility_router_config,
            run_p_anchored_directional_signed_utility_router,
        )

        config = load_p_anchored_directional_signed_utility_router_config(
            args.config
        )
        output = run_p_anchored_directional_signed_utility_router(
            config,
            artifact_root=artifact_root,
        )
        print(output)
        return 0

    if args.surface == "fixed-bank-p-anchored-crossfit-sample-influence-router":
        from .fixed_bank_p_anchored_crossfit_sample_influence_router import (
            load_p_anchored_crossfit_sample_influence_router_config,
            run_p_anchored_crossfit_sample_influence_router,
        )

        config = load_p_anchored_crossfit_sample_influence_router_config(
            args.config
        )
        output = run_p_anchored_crossfit_sample_influence_router(
            config,
            artifact_root=artifact_root,
        )
        print(output)
        return 0

    if (
        args.surface
        == "fixed-bank-p-anchored-crossfit-posterior-utility-margin-router"
    ):
        from .fixed_bank_p_anchored_crossfit_posterior_utility_margin_router import (
            load_p_anchored_crossfit_posterior_utility_margin_router_config,
            run_p_anchored_crossfit_posterior_utility_margin_router,
        )

        config = load_p_anchored_crossfit_posterior_utility_margin_router_config(
            args.config
        )
        output = run_p_anchored_crossfit_posterior_utility_margin_router(
            config,
            artifact_root=artifact_root,
        )
        print(output)
        return 0

    raise AssertionError(f"Unknown CVAE diagnostic surface: {args.surface}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
