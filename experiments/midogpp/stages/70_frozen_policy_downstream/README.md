# Stage 70: Frozen-Policy Downstream Evaluation

This is the thesis-facing downstream utility stage. It evaluates a routing or
composition policy only after expert selection, generation settings,
classifier settings, thresholds, budgets, and seeds have been frozen.

Held-out target labels are scoring-only. Reports must include matched
baselines, candidate eligibility, budget matching, leakage and identity audits,
BACC, macro-F1, and stability where available.

Status: `PLANNED`. This stage cannot run until a stage-60 policy and all
generation/classifier settings have been frozen under the new protocol-safe
implementation.
