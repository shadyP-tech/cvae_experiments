# Hybrid Ablation Multi-Seed Summary (Seeds 42/43/44)

## Scope

- Camelyon17 exact metadata routing (`categorical_exact`)
- BreakHis exact metadata routing (`categorical_exact`)
- BreakHis ordinal diagnostic routing (`ordinal_magnification`)

## Routing-Compatibility Winners (budget 1.0x)

### Camelyon17 exact
- Best oracle gap: variant B (0.0028 +/- 0.0001)
- Best Spearman: variant B (0.448 +/- 0.147)
- Best top-1 agreement: variant B (0.667 +/- 0.231)

### BreakHis exact
- Best oracle gap: variant B (0.0440 +/- 0.0030)
- Best Spearman: variant B (0.473 +/- 0.149)
- Best top-1 agreement: variant B (0.667 +/- 0.144)

### BreakHis ordinal
- Best oracle gap: variant B (0.0440 +/- 0.0030)
- Best Spearman: variant B (0.433 +/- 0.153)
- Best top-1 agreement: variant B (0.667 +/- 0.144)

## Aggregated Table (mean +/- std over seeds)

| dataset | policy | variant | budget | gap | spearman | top1 | rank | AUROC delta routed-real | AUROC delta routed-random | AUROC delta routed-pooled |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| breakhis | categorical_exact | A | budget_0.5x | 0.0799 +/- 0.0057 | 0.129 +/- 0.000 | 0.417 +/- 0.144 | 2.25 +/- 0.00 | 0.0158 +/- 0.0092 | -0.0013 +/- 0.0169 | -0.0033 +/- 0.0150 |
| breakhis | categorical_exact | A | budget_1.0x | 0.0799 +/- 0.0057 | 0.129 +/- 0.000 | 0.417 +/- 0.144 | 2.25 +/- 0.00 | 0.0218 +/- 0.0200 | 0.0107 +/- 0.0190 | 0.0110 +/- 0.0286 |
| breakhis | categorical_exact | B | budget_0.5x | 0.0440 +/- 0.0030 | 0.473 +/- 0.149 | 0.667 +/- 0.144 | 1.58 +/- 0.29 | -0.0096 +/- 0.0291 | -0.0083 +/- 0.0243 | 0.0061 +/- 0.0118 |
| breakhis | categorical_exact | B | budget_1.0x | 0.0440 +/- 0.0030 | 0.473 +/- 0.149 | 0.667 +/- 0.144 | 1.58 +/- 0.29 | -0.0022 +/- 0.0214 | 0.0145 +/- 0.0058 | 0.0042 +/- 0.0179 |
| breakhis | categorical_exact | C | budget_0.5x | 0.0812 +/- 0.0075 | 0.086 +/- 0.197 | 0.250 +/- 0.250 | 2.33 +/- 0.38 | -0.0199 +/- 0.0062 | -0.0085 +/- 0.0376 | -0.0198 +/- 0.0066 |
| breakhis | categorical_exact | C | budget_1.0x | 0.0812 +/- 0.0075 | 0.086 +/- 0.197 | 0.250 +/- 0.250 | 2.33 +/- 0.38 | -0.0215 +/- 0.0173 | -0.0342 +/- 0.0305 | -0.0080 +/- 0.0237 |
| breakhis | ordinal_magnification | A | budget_0.5x | 0.0799 +/- 0.0057 | 0.217 +/- 0.252 | 0.417 +/- 0.144 | 2.25 +/- 0.00 | 0.0158 +/- 0.0092 | -0.0013 +/- 0.0169 | 0.0245 +/- 0.0131 |
| breakhis | ordinal_magnification | A | budget_1.0x | 0.0799 +/- 0.0057 | 0.217 +/- 0.252 | 0.417 +/- 0.144 | 2.25 +/- 0.00 | 0.0218 +/- 0.0200 | 0.0107 +/- 0.0190 | 0.0124 +/- 0.0109 |
| breakhis | ordinal_magnification | B | budget_0.5x | 0.0440 +/- 0.0030 | 0.433 +/- 0.153 | 0.667 +/- 0.144 | 1.58 +/- 0.29 | -0.0096 +/- 0.0291 | -0.0083 +/- 0.0243 | 0.0045 +/- 0.0296 |
| breakhis | ordinal_magnification | B | budget_1.0x | 0.0440 +/- 0.0030 | 0.433 +/- 0.153 | 0.667 +/- 0.144 | 1.58 +/- 0.29 | -0.0022 +/- 0.0214 | 0.0145 +/- 0.0058 | 0.0097 +/- 0.0053 |
| breakhis | ordinal_magnification | C | budget_0.5x | 0.0812 +/- 0.0075 | 0.050 +/- 0.150 | 0.250 +/- 0.250 | 2.33 +/- 0.38 | -0.0199 +/- 0.0062 | -0.0085 +/- 0.0376 | -0.0203 +/- 0.0151 |
| breakhis | ordinal_magnification | C | budget_1.0x | 0.0812 +/- 0.0075 | 0.050 +/- 0.150 | 0.250 +/- 0.250 | 2.33 +/- 0.38 | -0.0215 +/- 0.0173 | -0.0342 +/- 0.0305 | -0.0347 +/- 0.0284 |
| camelyon17 | categorical_exact | A | budget_0.5x | 0.0069 +/- 0.0002 | 0.047 +/- 0.041 | 0.200 +/- 0.000 | 2.87 +/- 0.12 | 0.0072 +/- 0.0123 | 0.0024 +/- 0.0071 | 0.0144 +/- 0.0073 |
| camelyon17 | categorical_exact | A | budget_1.0x | 0.0069 +/- 0.0002 | 0.047 +/- 0.041 | 0.200 +/- 0.000 | 2.87 +/- 0.12 | -0.0086 +/- 0.0095 | -0.0081 +/- 0.0098 | -0.0008 +/- 0.0005 |
| camelyon17 | categorical_exact | B | budget_0.5x | 0.0028 +/- 0.0001 | 0.448 +/- 0.147 | 0.667 +/- 0.231 | 1.73 +/- 0.42 | 0.0007 +/- 0.0098 | 0.0158 +/- 0.0252 | 0.0053 +/- 0.0052 |
| camelyon17 | categorical_exact | B | budget_1.0x | 0.0028 +/- 0.0001 | 0.448 +/- 0.147 | 0.667 +/- 0.231 | 1.73 +/- 0.42 | -0.0165 +/- 0.0355 | 0.0024 +/- 0.0058 | -0.0062 +/- 0.0035 |
| camelyon17 | categorical_exact | C | budget_0.5x | 0.0070 +/- 0.0003 | 0.377 +/- 0.041 | 0.533 +/- 0.115 | 1.93 +/- 0.12 | -0.0165 +/- 0.0267 | -0.0060 +/- 0.0094 | -0.0077 +/- 0.0013 |
| camelyon17 | categorical_exact | C | budget_1.0x | 0.0070 +/- 0.0003 | 0.377 +/- 0.041 | 0.533 +/- 0.115 | 1.93 +/- 0.12 | -0.0089 +/- 0.0298 | -0.0046 +/- 0.0295 | -0.0050 +/- 0.0235 |

## Notes

- Balanced-accuracy deltas remained 0.0 in these runs and are therefore less informative than AUROC deltas for this analysis.
- Use this summary as the seed-sweep view; keep single-seed files for run-level traceability.
