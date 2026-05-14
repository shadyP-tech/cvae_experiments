import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
AGG_CSV = ROOT / "cvae_testing/results/comparison_tables/ae_first_routing_decision_table.csv"

METHODS = [
    ("metadata_routing", "Metadata"),
    ("ae_argmin_zscore", "AE argmin"),
    ("ae_first_margin_gated_v1", "AE margin-gated"),
]

METHOD_COLORS = {
    "metadata_routing": "#6B7280",
    "ae_argmin_zscore": "#0072B2",
    "ae_first_margin_gated_v1": "#D55E00",
}

DATASET_LABELS = {
    "breakhis": "BreakHis",
    "camelyon17": "Camelyon17",
}


def fnum(value):
    if value is None or value == "" or value == "nan":
        return float("nan")
    return float(value)


def load_aggregate():
    rows = []
    with AGG_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["method"] not in {m[0] for m in METHODS}:
                continue
            row = dict(row)
            for key in [
                "top1_oracle_hit",
                "raw_predicted_delta_spearman",
                "mean_oracle_gap_pct",
                "mean_oracle_gap",
                "metadata_relative_gain",
                "source_prior_relative_gain",
                "harmful_vs_metadata_rate",
                "improving_vs_metadata_rate",
                "ae_coverage_rate",
                "fallback_rate",
            ]:
                row[key] = fnum(row.get(key))
            rows.append(row)
    order = {(dataset, method): i for i, (dataset, method) in enumerate(
        (d, m[0]) for d in ["breakhis", "camelyon17"] for m in METHODS
    )}
    rows.sort(key=lambda r: order[(r["dataset"], r["method"])])
    return rows


def load_seed_rows():
    rows = []
    base_patterns = [
        ("breakhis", ROOT / "cvae_testing/outputs/breakhis/learned_utility_ae_first_routing_v1"),
        ("camelyon17", ROOT / "cvae_testing/outputs/camelyon17/learned_utility_ae_first_routing_v1"),
    ]
    wanted = {m[0] for m in METHODS}
    for dataset, base in base_patterns:
        for path in sorted(base.glob("*/reports/learned_utility_method_summary.csv")):
            seed = path.parent.parent.name.split("seed")[-1]
            with path.open(newline="") as f:
                for row in csv.DictReader(f):
                    if row["method"] not in wanted:
                        continue
                    rows.append({
                        "dataset": dataset,
                        "seed": seed,
                        "method": row["method"],
                        "top1_oracle_hit": fnum(row["top1_oracle_hit"]),
                        "mean_oracle_gap_pct": fnum(row["mean_oracle_gap_pct"]),
                        "raw_predicted_delta_spearman": fnum(row["raw_predicted_delta_spearman"]),
                    })
    return rows


def label_method(method):
    return dict(METHODS)[method]


def fmt_pct(value, digits=1):
    if math.isnan(value):
        return ""
    return f"{100 * value:.{digits}f}%"


def fmt_num(value, digits=2):
    if math.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def write_tables(rows):
    output_rows = []
    baseline_by_dataset = {
        r["dataset"]: r for r in rows if r["method"] == "metadata_routing"
    }
    for r in rows:
        baseline = baseline_by_dataset[r["dataset"]]
        top1_delta = r["top1_oracle_hit"] - baseline["top1_oracle_hit"]
        gap_delta = baseline["mean_oracle_gap_pct"] - r["mean_oracle_gap_pct"]
        output_rows.append({
            "Dataset": DATASET_LABELS[r["dataset"]],
            "Routing method": label_method(r["method"]),
            "Top-1 oracle hit": fmt_pct(r["top1_oracle_hit"]),
            "Delta vs metadata": fmt_pct(top1_delta),
            "Spearman": fmt_num(r["raw_predicted_delta_spearman"], 2),
            "Oracle gap pct": fmt_num(r["mean_oracle_gap_pct"], 1),
            "Gap reduction vs metadata": fmt_num(gap_delta, 1),
            "AE coverage": fmt_pct(r["ae_coverage_rate"]),
            "Harmful vs metadata": fmt_pct(r["harmful_vs_metadata_rate"]),
            "Verdict": r["verdict"] or "",
        })

    csv_path = OUT / "source_only_ae_comparison_table.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    md_path = OUT / "source_only_ae_comparison_table.md"
    headers = list(output_rows[0].keys())
    with md_path.open("w") as f:
        f.write("# Source-Only AE Routing: Slide Comparison Table\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in output_rows:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)


def grouped_bar(rows, metric, ylabel, title, filename, ylim=None, invert=False):
    datasets = ["breakhis", "camelyon17"]
    x = np.arange(len(datasets))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for idx, (method, label) in enumerate(METHODS):
        values = []
        for dataset in datasets:
            row = next(r for r in rows if r["dataset"] == dataset and r["method"] == method)
            values.append(row[metric])
        offset = (idx - 1) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=label,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.8,
        )
        for bar, value in zip(bars, values):
            text = f"{value * 100:.0f}%" if metric == "top1_oracle_hit" else f"{value:.1f}"
            ax.annotate(
                text,
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in datasets])
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=14, weight="bold")
    if ylim:
        ax.set_ylim(*ylim)
    if invert:
        ax.text(0.01, 0.94, "lower is better", transform=ax.transAxes, fontsize=9, color="#4B5563")
    else:
        ax.text(0.01, 0.94, "higher is better", transform=ax.transAxes, fontsize=9, color="#4B5563")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.09))
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def combined_top1_gap(rows):
    datasets = ["breakhis", "camelyon17"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharex=False)
    specs = [
        ("top1_oracle_hit", "Top-1 oracle hit", "higher is better", (0, 0.82)),
        ("mean_oracle_gap_pct", "Normalized oracle gap", "lower is better", (0, 40)),
    ]
    for ax, (metric, ylabel, note, ylim) in zip(axes, specs):
        x = np.arange(len(datasets))
        width = 0.24
        for idx, (method, label) in enumerate(METHODS):
            values = [next(r for r in rows if r["dataset"] == d and r["method"] == method)[metric] for d in datasets]
            offset = (idx - 1) * width
            bars = ax.bar(x + offset, values, width, label=label, color=METHOD_COLORS[method])
            for bar, value in zip(bars, values):
                text = f"{value * 100:.0f}%" if metric == "top1_oracle_hit" else f"{value:.1f}"
                ax.annotate(text, (bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABELS[d] for d in datasets])
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.text(0.02, 0.93, note, transform=ax.transAxes, fontsize=9, color="#4B5563")
        style_axes(ax)
    axes[0].set_title("Source-only AE has signal, but remains diagnostic", loc="left", fontsize=14, weight="bold")
    axes[1].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(-0.18, -0.12))
    fig.tight_layout()
    fig.savefig(OUT / "source_only_ae_top1_gap_panel.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def risk_plot(rows):
    gated = [r for r in rows if r["method"] == "ae_first_margin_gated_v1"]
    datasets = ["breakhis", "camelyon17"]
    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    y = np.arange(len(datasets))
    improving = []
    harmful = []
    neutral = []
    coverage = []
    fallback = []
    for dataset in datasets:
        row = next(r for r in gated if r["dataset"] == dataset)
        imp = row["improving_vs_metadata_rate"]
        harm = row["harmful_vs_metadata_rate"]
        improving.append(imp)
        harmful.append(harm)
        neutral.append(max(0.0, 1.0 - imp - harm))
        coverage.append(row["ae_coverage_rate"])
        fallback.append(row["fallback_rate"])
    ax.barh(y, improving, color="#009E73", label="Improves vs metadata")
    ax.barh(y, neutral, left=improving, color="#D1D5DB", label="Neutral / mixed")
    ax.barh(y, harmful, left=np.array(improving) + np.array(neutral), color="#D55E00", label="Harmful vs metadata")
    for i, dataset in enumerate(datasets):
        ax.text(1.02, i, f"AE coverage {coverage[i] * 100:.0f}% / fallback {fallback[i] * 100:.0f}%", va="center", fontsize=9, color="#374151")
        ax.text(improving[i] / 2, i, f"{improving[i] * 100:.0f}%", va="center", ha="center", fontsize=9, color="white")
        ax.text(1 - harmful[i] / 2, i, f"{harmful[i] * 100:.0f}%", va="center", ha="center", fontsize=9, color="white")
    ax.set_yticks(y)
    ax.set_yticklabels([DATASET_LABELS[d] for d in datasets])
    ax.set_xlim(0, 1.33)
    ax.set_xlabel("Share of routing decisions")
    ax.set_title("Why the AE gate is diagnostic: gains coexist with harmful overrides", loc="left", fontsize=14, weight="bold")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / "source_only_ae_gated_risk_breakdown.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def stability_plot(seed_rows):
    datasets = ["breakhis", "camelyon17"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=False)
    for ax, dataset in zip(axes, datasets):
        rows = [r for r in seed_rows if r["dataset"] == dataset]
        for idx, (method, label) in enumerate(METHODS):
            vals = [r["top1_oracle_hit"] for r in rows if r["method"] == method]
            if not vals:
                continue
            xs = np.full(len(vals), idx, dtype=float) + np.linspace(-0.055, 0.055, len(vals))
            ax.scatter(xs, vals, color=METHOD_COLORS[method], s=42, zorder=3)
            ax.hlines(np.mean(vals), idx - 0.22, idx + 0.22, colors=METHOD_COLORS[method], linewidth=3)
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels([label for _, label in METHODS], rotation=16, ha="right")
        ax.set_ylim(0, 0.85)
        ax.set_ylabel("Top-1 oracle hit")
        ax.set_title(DATASET_LABELS[dataset], loc="left", fontsize=12, weight="bold")
        style_axes(ax)
    fig.suptitle("Seed stability: AE improvement is clearer on BreakHis than Camelyon17", x=0.02, ha="left", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "source_only_ae_seed_stability_top1.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def scatter_tradeoff(rows):
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for r in rows:
        method = r["method"]
        marker = "o" if r["dataset"] == "breakhis" else "s"
        ax.scatter(
            r["mean_oracle_gap_pct"],
            r["top1_oracle_hit"],
            s=130,
            color=METHOD_COLORS[method],
            marker=marker,
            edgecolor="white",
            linewidth=1.0,
            label=f"{DATASET_LABELS[r['dataset']]} / {label_method(method)}",
        )
        ax.annotate(
            f"{DATASET_LABELS[r['dataset']]}\n{label_method(method)}",
            (r["mean_oracle_gap_pct"], r["top1_oracle_hit"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Normalized oracle gap pct (lower is better)")
    ax.set_ylabel("Top-1 oracle hit (higher is better)")
    ax.set_xlim(-1, 39)
    ax.set_ylim(0.18, 0.77)
    ax.set_title("AE routing moves toward the oracle corner, but not enough for adoption", loc="left", fontsize=14, weight="bold")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / "source_only_ae_top1_gap_tradeoff.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_interpretation(rows):
    md = OUT / "source_only_ae_result_interpretation.md"
    lines = [
        "RESULT INTERPRETATION:",
        "",
        "### Evidence Source",
        "",
        "- Aggregate decision table: `cvae_testing/results/comparison_tables/ae_first_routing_decision_table.csv`.",
        "- Cross-seed summary: `cvae_testing/results/summaries/ae_first_routing_decision_table.md`.",
        "- Per-seed method summaries: `cvae_testing/outputs/{breakhis,camelyon17}/learned_utility_ae_first_routing_v1/*/reports/learned_utility_method_summary.csv`.",
        "",
        "### Thesis Question",
        "",
        "Can source-trained Autoencoder confidence route target/query samples without target support, and does that proxy recover held-out NELBO utility better than metadata routing?",
        "",
        "### Primary Metrics",
        "",
    ]
    for dataset in ["breakhis", "camelyon17"]:
        ds_rows = {r["method"]: r for r in rows if r["dataset"] == dataset}
        metadata = ds_rows["metadata_routing"]
        argmin = ds_rows["ae_argmin_zscore"]
        gated = ds_rows["ae_first_margin_gated_v1"]
        lines.append(
            f"- {DATASET_LABELS[dataset]}: metadata top-1 {metadata['top1_oracle_hit']:.3f}, "
            f"gap {metadata['mean_oracle_gap_pct']:.1f}; AE argmin top-1 {argmin['top1_oracle_hit']:.3f}, "
            f"gap {argmin['mean_oracle_gap_pct']:.1f}; margin-gated AE top-1 {gated['top1_oracle_hit']:.3f}, "
            f"gap {gated['mean_oracle_gap_pct']:.1f}, coverage {gated['ae_coverage_rate']:.3f}, harmful-vs-metadata {gated['harmful_vs_metadata_rate']:.3f}."
        )
    lines.extend([
        "",
        "### Baseline Comparison",
        "",
        "The source-only AE proxy beats metadata routing on top-1 oracle hit and normalized oracle gap in both datasets, with a stronger effect on BreakHis. The gated policy reduces risk relative to unconstrained AE argmin, but it still has non-trivial harmful routing decisions.",
        "",
        "### Claim Classification",
        "",
        "`DIAGNOSTIC ONLY`: the AE proxy contains useful distributional signal, but the margin-gated policy fails the domain-level non-degradation requirement and therefore should not be presented as an adoption-ready router.",
        "",
        "### Thesis Text",
        "",
        "Source-only AE confidence can partially recover expert compatibility without target support, improving over metadata routing on oracle-hit and oracle-gap metrics. However, because the same proxy still produces harmful overrides and fails the domain-level safety criterion, it supports the thesis pivot from source-only proxy routing toward target-local NELBO utility estimation.",
        "",
        "### Caveats",
        "",
        "- The method estimates reconstruction fit in embedding space, not CVAE utility directly.",
        "- The cross-dataset verdict remains `DIAGNOSTIC ONLY`, so do not claim this as the final routing method.",
        "- The effect is dataset-dependent: BreakHis shows a clearer gain than Camelyon17.",
        "",
        "### Next Evidence Needed",
        "",
        "For the main thesis claim, place this table next to direct support-NELBO routing results to show that target-local utility estimation is more reliable than source-only proxy confidence.",
        "",
    ])
    md.write_text("\n".join(lines))


def main():
    rows = load_aggregate()
    seed_rows = load_seed_rows()
    write_tables(rows)
    combined_top1_gap(rows)
    grouped_bar(rows, "top1_oracle_hit", "Top-1 oracle hit", "Source-only AE improves oracle-hit over metadata", "source_only_ae_top1_oracle_hit.png", ylim=(0, 0.82))
    grouped_bar(rows, "mean_oracle_gap_pct", "Normalized oracle gap pct", "AE routing reduces selected-vs-oracle gap", "source_only_ae_oracle_gap_pct.png", ylim=(0, 40), invert=True)
    risk_plot(rows)
    stability_plot(seed_rows)
    scatter_tradeoff(rows)
    write_interpretation(rows)
    print(f"Wrote source-only AE presentation assets to {OUT}")


if __name__ == "__main__":
    main()
