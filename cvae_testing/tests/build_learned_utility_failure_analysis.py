#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


DEFAULT_INPUTS = {
    "breakhis_learned_utility": Path(
        "results/comparison_tables/learned_utility_breakhis_v2_decision_strict.csv"
    ),
    "breakhis_response_learned_utility": Path(
        "results/comparison_tables/learned_utility_response_breakhis_v2_decision_strict.csv"
    ),
    "camelyon17_response_learned_utility": Path(
        "results/comparison_tables/learned_utility_response_camelyon17_v2_decision_strict.csv"
    ),
}

SCENARIO_LABELS = {
    "breakhis_learned_utility": "BreakHis learned utility",
    "breakhis_response_learned_utility": "BreakHis response learned utility",
    "camelyon17_response_learned_utility": "Camelyon17 response learned utility",
}

METHOD_DISPLAY = {
    "metadata_routing": "Metadata routing",
    "linear_regressor": "Linear regressor",
    "mlp_regressor": "MLP regressor",
    "pairwise_ranker_combined": "Pairwise ranker combined",
    "pairwise_ranker_latent_only": "Pairwise ranker latent only",
    "pairwise_ranker_metadata_only": "Pairwise ranker metadata only",
    "candidate_oracle_routing": "Candidate oracle routing",
    "latent_wasserstein_routing": "Latent Wasserstein routing",
    "random_rank_floor": "Random rank floor",
    "random_score_floor": "Random score floor",
}

METHOD_ORDER = [
    "metadata_routing",
    "linear_regressor",
    "mlp_regressor",
    "pairwise_ranker_combined",
    "pairwise_ranker_latent_only",
    "pairwise_ranker_metadata_only",
    "candidate_oracle_routing",
    "latent_wasserstein_routing",
    "random_rank_floor",
    "random_score_floor",
]

PLOT_METHODS = [
    "metadata_routing",
    "linear_regressor",
    "mlp_regressor",
    "pairwise_ranker_combined",
    "pairwise_ranker_latent_only",
    "pairwise_ranker_metadata_only",
]

TABLE_FIELDS = [
    "scenario",
    "scenario_label",
    "method",
    "method_label",
    "method_role",
    "adoption_eligible",
    "diagnostic_only",
    "tier",
    "decision",
    "top1_mean",
    "top1_std",
    "spearman_mean",
    "spearman_std",
    "gap_pct_mean",
    "gap_pct_std",
    "top1_uplift_mean",
    "top1_uplift_std",
    "spearman_uplift_mean",
    "spearman_uplift_std",
    "gap_pct_reduction_mean",
    "gap_pct_reduction_std",
    "improving_seed_count",
    "instability_breach",
    "fail_reason",
]


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(result):
        return float(default)
    return result


def _to_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _fmt(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def _method_sort_key(row: Dict[str, Any]) -> tuple[int, str]:
    method = str(row.get("method", ""))
    try:
        idx = METHOD_ORDER.index(method)
    except ValueError:
        idx = len(METHOD_ORDER)
    return idx, method


def _failure_reason(row: Dict[str, Any]) -> str:
    role = str(row.get("method_role", ""))
    method = str(row.get("method", ""))
    if method == "metadata_routing":
        return "baseline reference"
    if role == "diagnostic":
        return "reference only; diagnostic or uses unavailable utility/statistics"
    if role == "control":
        return "reference only; control floor"
    if _to_float(row.get("top1_uplift_mean")) <= 0:
        return "no mean top-1 improvement over metadata"
    if _to_float(row.get("spearman_uplift_mean")) <= 0:
        return "no mean Spearman improvement over metadata"
    if _to_float(row.get("gap_pct_reduction_mean")) <= 0:
        return "no mean oracle-gap reduction over metadata"
    if _to_int(row.get("improving_seed_count")) < 3:
        return "metric gains are inconsistent across seeds"
    if _to_int(row.get("instability_breach")):
        return "metric gains fail stability/sign-CI gate"
    return "failed adoption threshold"


def _read_scenario_rows(base_dir: Path, scenario: str, path: Path) -> List[Dict[str, Any]]:
    resolved = path if path.is_absolute() else base_dir / path
    if not resolved.exists():
        raise FileNotFoundError(f"Missing learned-utility strict table: {resolved}")

    rows: List[Dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            method = str(raw.get("method", "")).strip()
            if method not in METHOD_DISPLAY:
                continue
            row = {
                "scenario": scenario,
                "scenario_label": SCENARIO_LABELS.get(scenario, scenario),
                "method": method,
                "method_label": METHOD_DISPLAY.get(method, method),
                "method_role": str(raw.get("method_role", "")),
                "adoption_eligible": _to_int(raw.get("adoption_eligible")),
                "diagnostic_only": _to_int(raw.get("diagnostic_only")),
                "tier": str(raw.get("tier", "")),
                "decision": str(raw.get("decision", "")),
                "top1_mean": _to_float(raw.get("top1_oracle_hit_mean")),
                "top1_std": _to_float(raw.get("top1_oracle_hit_std")),
                "spearman_mean": _to_float(raw.get("spearman_mean")),
                "spearman_std": _to_float(raw.get("spearman_std")),
                "gap_pct_mean": _to_float(raw.get("mean_oracle_gap_pct_mean")),
                "gap_pct_std": _to_float(raw.get("mean_oracle_gap_pct_std")),
                "top1_uplift_mean": _to_float(raw.get("top1_uplift_vs_metadata_mean")),
                "top1_uplift_std": _to_float(raw.get("top1_uplift_vs_metadata_std")),
                "spearman_uplift_mean": _to_float(raw.get("spearman_uplift_vs_metadata_mean")),
                "spearman_uplift_std": _to_float(raw.get("spearman_uplift_vs_metadata_std")),
                "gap_pct_reduction_mean": _to_float(
                    raw.get("oracle_gap_pct_reduction_vs_metadata_mean")
                ),
                "gap_pct_reduction_std": _to_float(
                    raw.get("oracle_gap_pct_reduction_vs_metadata_std")
                ),
                "improving_seed_count": _to_int(raw.get("improving_seed_count")),
                "instability_breach": _to_int(raw.get("instability_breach")),
            }
            row["fail_reason"] = _failure_reason(row)
            rows.append(row)
    return sorted(rows, key=_method_sort_key)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, rows: Sequence[Dict[str, Any]], plots: Sequence[str]) -> None:
    learned = [
        r
        for r in rows
        if _to_int(r.get("adoption_eligible")) == 1
        and str(r.get("method_role")) == "learned"
    ]
    summary = {
        "classification": "FAIL",
        "reason": (
            "All adoption-eligible learned-utility rows are rejected by the strict "
            "decision gate; metadata routing remains the selected baseline."
        ),
        "n_rows": len(rows),
        "n_learned_rows": len(learned),
        "n_learned_failures": sum(1 for r in learned if str(r.get("tier")) == "fail"),
        "scenarios": sorted({str(r["scenario"]) for r in rows}),
        "plots": list(plots),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _markdown_table(rows: Sequence[Dict[str, Any]]) -> str:
    header = [
        "Scenario",
        "Method",
        "Role",
        "Top1",
        "Spearman",
        "Gap %",
        "Top1 uplift",
        "Gap reduction",
        "Improve seeds",
        "Gate",
        "Reason",
    ]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["scenario_label"]),
                    str(r["method_label"]),
                    str(r["method_role"]),
                    f"{_fmt(r['top1_mean'])} +/- {_fmt(r['top1_std'])}",
                    f"{_fmt(r['spearman_mean'])} +/- {_fmt(r['spearman_std'])}",
                    f"{_fmt(r['gap_pct_mean'])} +/- {_fmt(r['gap_pct_std'])}",
                    f"{_fmt(r['top1_uplift_mean'])} +/- {_fmt(r['top1_uplift_std'])}",
                    f"{_fmt(r['gap_pct_reduction_mean'])} +/- {_fmt(r['gap_pct_reduction_std'])}",
                    str(r["improving_seed_count"]),
                    "breach" if _to_int(r["instability_breach"]) else "ok/reference",
                    str(r["fail_reason"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _write_md(path: Path, rows: Sequence[Dict[str, Any]], plots: Sequence[str]) -> None:
    learned_rows = [
        r
        for r in rows
        if str(r["method_role"]) in {"baseline", "learned"}
    ]
    best_by_scenario = []
    for scenario in sorted({str(r["scenario"]) for r in rows}):
        candidates = [
            r
            for r in rows
            if str(r["scenario"]) == scenario and str(r["method_role"]) == "learned"
        ]
        if candidates:
            best_by_scenario.append(
                max(candidates, key=lambda r: _to_float(r["gap_pct_reduction_mean"]))
            )

    text = [
        "# Learned Utility Adoption-Readiness Analysis",
        "",
        "Classification: `FAIL`.",
        "",
        "This table focuses on thesis-facing adoption candidates. Diagnostic/oracle rows are retained in the CSV for context, but they are not adoption-eligible.",
        "",
        "## Baseline And Learned Methods",
        "",
        _markdown_table(learned_rows),
        "",
        "## Best Learned Method Per Scenario",
        "",
        _markdown_table(best_by_scenario),
        "",
        "## Plots",
        "",
    ]
    for plot in plots:
        text.append(f"- `{plot}`")
    text.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Learned methods sometimes improve top-1, Spearman, and oracle gap versus metadata.",
            "- The strict gate still rejects them because adoption requires stable, non-regressive improvement, not isolated metric gains.",
            "- Metadata routing remains the selected baseline in all three strict learned-utility summaries.",
            "- Candidate-oracle and latent/utility diagnostic rows should be discussed only as reference bounds or diagnostics.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text), encoding="utf-8")


def _setup_matplotlib(plot_dir: Path):
    mpl_config = Path("/private/tmp/cvae_metadata_routing_mplconfig")
    font_cache = Path("/private/tmp/cvae_metadata_routing_fontconfig")
    mpl_config.mkdir(parents=True, exist_ok=True)
    font_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    os.environ.setdefault("XDG_CACHE_HOME", str(font_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _plot_metric_comparison(plot_dir: Path, rows: Sequence[Dict[str, Any]]) -> str:
    plt = _setup_matplotlib(plot_dir)
    path = plot_dir / "learned_utility_metric_comparison.png"
    plot_rows = [r for r in rows if str(r["method"]) in PLOT_METHODS]
    scenarios = sorted({str(r["scenario"]) for r in plot_rows})
    methods = PLOT_METHODS
    metrics = [
        ("top1_mean", "Top-1 oracle hit", "higher is better"),
        ("spearman_mean", "Spearman", "higher is better"),
        ("gap_pct_mean", "Oracle gap %", "lower is better"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=False)
    colors = {
        "metadata_routing": "#4C78A8",
        "linear_regressor": "#F58518",
        "mlp_regressor": "#54A24B",
        "pairwise_ranker_combined": "#B279A2",
        "pairwise_ranker_latent_only": "#E45756",
        "pairwise_ranker_metadata_only": "#72B7B2",
    }
    for ax, (metric, title, subtitle) in zip(axes, metrics):
        width = 0.12
        centers = list(range(len(scenarios)))
        for i, method in enumerate(methods):
            values = []
            errors = []
            for scenario in scenarios:
                row = next(
                    r
                    for r in plot_rows
                    if str(r["scenario"]) == scenario and str(r["method"]) == method
                )
                values.append(_to_float(row[metric]))
                errors.append(_to_float(row[metric.replace("_mean", "_std")]))
            offsets = [c + (i - (len(methods) - 1) / 2) * width for c in centers]
            ax.bar(
                offsets,
                values,
                width=width,
                yerr=errors,
                capsize=2,
                label=METHOD_DISPLAY[method],
                color=colors[method],
                alpha=0.88,
            )
        ax.set_title(f"{title}\n{subtitle}")
        ax.set_xticks(centers)
        ax.set_xticklabels([SCENARIO_LABELS[s].replace(" ", "\n") for s in scenarios], fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.32), ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _plot_uplifts(plot_dir: Path, rows: Sequence[Dict[str, Any]]) -> str:
    plt = _setup_matplotlib(plot_dir)
    path = plot_dir / "learned_utility_uplifts_vs_metadata.png"
    learned = [
        r
        for r in rows
        if str(r["method_role"]) == "learned" and _to_int(r["adoption_eligible"]) == 1
    ]
    scenarios = sorted({str(r["scenario"]) for r in learned})
    metrics = [
        ("top1_uplift_mean", "Top-1 uplift"),
        ("spearman_uplift_mean", "Spearman uplift"),
        ("gap_pct_reduction_mean", "Gap % reduction"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (metric, title) in zip(axes, metrics):
        labels: List[str] = []
        values: List[float] = []
        colors: List[str] = []
        for scenario in scenarios:
            for method in PLOT_METHODS[1:]:
                row = next(
                    r
                    for r in learned
                    if str(r["scenario"]) == scenario and str(r["method"]) == method
                )
                method_label = METHOD_DISPLAY[method].replace(" ", "\n")
                labels.append(f"{SCENARIO_LABELS[scenario].split()[0]}\n{method_label}")
                value = _to_float(row[metric])
                values.append(value)
                colors.append("#D62728" if _to_int(row["instability_breach"]) else "#2CA02C")
        ax.bar(range(len(values)), values, color=colors, alpha=0.82)
        ax.axhline(0.0, color="#333333", linewidth=1.0)
        ax.set_title(title)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=6, rotation=0)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Learned methods versus metadata routing; red = strict gate breach", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _plot_gate_failures(plot_dir: Path, rows: Sequence[Dict[str, Any]]) -> str:
    plt = _setup_matplotlib(plot_dir)
    path = plot_dir / "learned_utility_gate_failures.png"
    learned = [
        r
        for r in rows
        if str(r["method_role"]) == "learned" and _to_int(r["adoption_eligible"]) == 1
    ]
    scenarios = sorted({str(r["scenario"]) for r in learned})
    methods = PLOT_METHODS[1:]
    matrix: List[List[float]] = []
    labels: List[str] = []
    gate_rows: List[Dict[str, Any]] = []
    for scenario in scenarios:
        for method in methods:
            row = next(
                r
                for r in learned
                if str(r["scenario"]) == scenario and str(r["method"]) == method
            )
            # 1.0 means fully passed, 0.5 means metric-improving but unstable, 0.0 means not improving.
            improving = _to_int(row["improving_seed_count"])
            breach = _to_int(row["instability_breach"])
            score = 1.0 if improving >= 3 and not breach else 0.5 if improving >= 2 else 0.0
            matrix.append([score])
            labels.append(f"{SCENARIO_LABELS[scenario]} - {METHOD_DISPLAY[method]}")
            gate_rows.append(row)

    fig, ax = plt.subplots(figsize=(7.5, max(4.5, len(labels) * 0.35)))
    ax.imshow(matrix, cmap=plt.get_cmap("RdYlGn"), vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xticks([0])
    ax.set_xticklabels(["Strict adoption gate"])
    for i, _label in enumerate(labels):
        row = gate_rows[i]
        text = f"improving seeds={row['improving_seed_count']}/3, instability={row['instability_breach']}"
        ax.text(0, i, text, ha="center", va="center", fontsize=8, color="black")
    ax.set_title("Why learned utility is not adoption-ready")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _plot_tradeoff(plot_dir: Path, rows: Sequence[Dict[str, Any]]) -> str:
    plt = _setup_matplotlib(plot_dir)
    path = plot_dir / "learned_utility_top1_gap_tradeoff.png"
    learned = [
        r
        for r in rows
        if str(r["method_role"]) == "learned" and _to_int(r["adoption_eligible"]) == 1
    ]
    fig, ax = plt.subplots(figsize=(8, 6))
    scenario_markers = {
        "breakhis_learned_utility": "o",
        "breakhis_response_learned_utility": "s",
        "camelyon17_response_learned_utility": "^",
    }
    for row in learned:
        color = "#D62728" if _to_int(row["instability_breach"]) else "#2CA02C"
        ax.scatter(
            _to_float(row["top1_uplift_mean"]),
            _to_float(row["gap_pct_reduction_mean"]),
            s=90 + 160 * max(0.0, _to_float(row["spearman_uplift_mean"])),
            marker=scenario_markers.get(str(row["scenario"]), "o"),
            color=color,
            alpha=0.78,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.text(
            _to_float(row["top1_uplift_mean"]) + 0.006,
            _to_float(row["gap_pct_reduction_mean"]),
            METHOD_DISPLAY[str(row["method"])].replace("Pairwise ranker ", "PR "),
            fontsize=7,
        )
    ax.axvline(0.0, color="#333333", linewidth=1.0)
    ax.axhline(0.0, color="#333333", linewidth=1.0)
    ax.set_xlabel("Top-1 uplift vs metadata")
    ax.set_ylabel("Oracle-gap reduction vs metadata")
    ax.set_title("Metric gains exist, but strict adoption rejects unstable learned methods")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _write_plots(plot_dir: Path, rows: Sequence[Dict[str, Any]]) -> List[str]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    return [
        _plot_metric_comparison(plot_dir, rows),
        _plot_uplifts(plot_dir, rows),
        _plot_gate_failures(plot_dir, rows),
        _plot_tradeoff(plot_dir, rows),
    ]


def _collect_rows(base_dir: Path, inputs: Dict[str, Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scenario, path in inputs.items():
        rows.extend(_read_scenario_rows(base_dir, scenario, path))
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build learned-utility failure comparison table and plots from strict decision CSVs."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="cvae_testing directory",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/comparison_tables/learned_utility_failure_comparison_table.csv"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/summaries/learned_utility_failure_comparison_table.md"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/comparison_tables/learned_utility_failure_comparison_summary.json"),
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("results/plots/learned_utility_failure"),
    )
    args = parser.parse_args(argv)

    base_dir = args.base_dir.resolve()
    rows = _collect_rows(base_dir, DEFAULT_INPUTS)
    if not rows:
        raise RuntimeError("No learned-utility comparison rows were collected.")

    output_csv = args.output_csv if args.output_csv.is_absolute() else base_dir / args.output_csv
    output_md = args.output_md if args.output_md.is_absolute() else base_dir / args.output_md
    output_json = args.output_json if args.output_json.is_absolute() else base_dir / args.output_json
    plot_dir = args.plot_dir if args.plot_dir.is_absolute() else base_dir / args.plot_dir

    plots = _write_plots(plot_dir, rows)
    _write_csv(output_csv, rows)
    _write_md(output_md, rows, plots)
    _write_json(output_json, rows, plots)

    print(f"Wrote {output_csv}")
    print(f"Wrote {output_md}")
    print(f"Wrote {output_json}")
    for plot in plots:
        print(f"Wrote {plot}")


if __name__ == "__main__":
    main()
