# Rebuild Gates

## Purpose

Define the gate from SAIL to a vanilla Virchow2 CVAE preservation test.

## Key Claims

SAIL justifies R1.3a Vanilla Virchow2 CVAE Rebuild only if Virchow2 dense primary rows satisfy:

```text
mean BACC >= 0.92
worst center >= 0.85
seed mean-BACC std <= 0.03
no seed has worst-center BACC < 0.75
```

Optional condition:

```text
clear positive delta vs R1.2b Virchow2 top-1
or clear weak-center/stability improvement
```

## Evidence / Source Artifacts

- `../../context/current_experimental_state.md`
- `../../../sail/configs/sail_virchow2.yaml`
- `../../../sail/src/sail/`

## Interpretation

The gate protects the thesis from rebuilding CVAEs in a feature space whose real-feature transfer remains unstable.

## Implication For Thesis

Passing SAIL justifies a preservation test. It does not prove CVAE generation will work.

## Limitations

Current gate status is unknown because SAIL output artifacts are absent locally. TODO: verify against artifact.

## Next Checks

- Populate mean BACC, worst center, seed std, and seed worst-center floor from SAIL artifacts.
- Record pass/fail in `../../context/current_experimental_state.md`.
