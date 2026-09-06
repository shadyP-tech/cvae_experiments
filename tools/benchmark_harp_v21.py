"""Synthetic HARP v21 CPU construction benchmark; never reads dataset labels.

Run from the repository with PYTHONPATH=src. This is not an experiment launch
and cannot establish real routing utility. It fabricates embeddings, labels,
and physical probability vectors in memory without an authorization lease.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

from midogpp_thesis.cvae.routing.correction_mass_router_v21 import (
    Direction, LabelFreeAction, LabelFreeCaseMenu, RouterFitConfig,
    SurfaceRole, SupportTruthCapability, fit_source_router,
    float32_probability_hex,
)

CENTERS = ('0', '1', '2', '3', '5', '6', '7', '8', '9')


def synthetic_surface(cases_per_center: int, patches: int, seed: int):
    rng = np.random.default_rng(seed)
    menus, truth = [], {}
    for center in CENTERS:
        donors = tuple(c for c in CENTERS if c != center)
        for ordinal in range(cases_per_center):
            case = f'synthetic_{ordinal:03d}'
            labels = np.tile((0, 1), patches//2)
            rng.shuffle(labels)
            hard = labels.copy()
            for k in (0, 1):
                chosen = rng.choice(np.flatnonzero(labels == k), max(1, patches//12), replace=False)
                hard[chosen] = 1-hard[chosen]
            base = np.where(hard, .65, .35)
            good = np.where(labels, .8, .2)
            samples = tuple(f'{center}/{case}/{j}' for j in range(patches))
            base_hex = float32_probability_hex(base.tolist())
            actions = []
            for direction in (Direction.D01, Direction.D10):
                for index, donor in enumerate(donors):
                    quality = 1. if index < 4 else -1.
                    probability = good if quality > 0 else 1-good
                    active = hard == (0 if direction is Direction.D01 else 1)
                    probability = np.where(active, probability, base)
                    actions.append(LabelFreeAction(
                        SurfaceRole.SOURCE_TRAIN_DEVELOPMENT, center, case,
                        f'{direction.value}_{donor}', direction, donor,
                        ('compatibility_fixed_fixture',), (quality,), samples,
                        base_hex, float32_probability_hex(probability.tolist())))
            actions.append(LabelFreeAction(
                SurfaceRole.SOURCE_TRAIN_DEVELOPMENT, center, case, 'U:FULL', Direction.FULL, None,
                ('compatibility_fixed_fixture',), (1.,), samples,
                base_hex, float32_probability_hex(good.tolist())))
            features = rng.normal(0., .1, (patches, 3840)).astype(np.float32)
            features[:, 3179] = 2*labels-1+rng.normal(0., .05, patches)
            menus.append(LabelFreeCaseMenu(SurfaceRole.SOURCE_TRAIN_DEVELOPMENT,
                center, case, samples, base_hex, tuple(actions), features))
            truth[(center, case)] = tuple(zip(samples, map(int, labels), strict=True))
    return tuple(menus), SupportTruthCapability(truth)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases-per-center', type=int, default=24)
    parser.add_argument('--patches', type=int, default=48)
    parser.add_argument('--outer-folds', type=int, default=5)
    parser.add_argument('--inner-folds', type=int, default=4)
    parser.add_argument('--seed', type=int, default=21021)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.cases_per_center < 12 or args.patches < 12 or args.patches % 2:
        parser.error('Use at least 12 cases per center and an even patch count >=12.')
    started = time.perf_counter()
    menus, capability = synthetic_surface(args.cases_per_center, args.patches, args.seed)
    config = replace(RouterFitConfig(), required_source_case_count=len(menus),
        required_source_center_count=len(CENTERS), outer_folds=args.outer_folds,
        inner_folds=args.inner_folds)
    preparation = time.perf_counter()-started
    print(f'[harp-v21-benchmark] fabricated {len(menus)} cases, {args.patches} patches, 3840 features', flush=True)
    with threadpool_limits(limits=1):
        policy = fit_source_router(menus, capability, config=config)
    elapsed = time.perf_counter()-started
    crossfit = policy.crossfit.public_payload()
    admission = policy.admission.public_payload()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = dict(schema='harp_v21_synthetic_cpu_benchmark_v1', scientific_data_used=False,
        raw_dataset_labels_opened=False, experiment_launched=False, execution_authorized=False,
        claim='Synthetic construction and CPU geometry only; no real MIDOG++ utility evidence.',
        cases=len(menus), patches_per_case=args.patches, feature_dimension=3840,
        seed=args.seed, outer_folds=config.outer_folds, inner_folds=config.inner_folds,
        evidence_variants=config.evidence_variants, source_oof_cases=len(crossfit['records']),
        final_evidence_variant=crossfit['final_evidence_variant'],
        admission=admission, model_hash=policy.model_hash, policy_hash=policy.policy_hash,
        preparation_seconds=preparation, fit_seconds=elapsed-preparation, total_seconds=elapsed,
        peak_rss_bytes=int(rss if sys.platform == 'darwin' else rss*1024),
        python=sys.version.split()[0], platform=sys.platform, blas_threads=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2)+'\n')
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
