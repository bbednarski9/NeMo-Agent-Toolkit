#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Cross-tier TTFT/ITL/TPS comparison across multi-domain router benchmark tiers.

Reads per-LLM-call and per-request CSVs from each tier's comparison/ directory
and produces:
  - Aggregate stats table (median, p95, mean TTFT/ITL/TPS per tier)
  - Per-domain breakdown
  - Bar charts comparing tiers
  - TTFT distribution histograms revealing cold/warm bimodality
  - First-call penalty analysis (llm_call_idx=0 vs subsequent)
  - Summary CSV

Usage:
    python cross_tier_ttft_comparison.py \\
        --tiers round_robin=./dynamo_evals/multi_domain_round_robin/comparison \\
                kv_aware=./dynamo_evals/multi_domain_kv_aware/comparison \\
                thompson_no_kv=./dynamo_evals/multi_domain_thompson_no_kv_events/comparison \\
                thompson_kv_on=./dynamo_evals/multi_domain_thompson_kv_on/comparison \\
        --output ./dynamo_evals/cross_tier_comparison
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TIER_DISPLAY_NAMES = {
    "round_robin": "Round-Robin",
    "kv_aware": "KV-Aware",
    "thompson_no_kv": "Thompson (KV OFF)",
    "thompson_kv_on": "Thompson (KV ON)",
    "backpressure_kv": "Backpressure+KV",
}

TIER_COLORS = {
    "round_robin": "#e74c3c",
    "kv_aware": "#2ecc71",
    "thompson_no_kv": "#3498db",
    "thompson_kv_on": "#9b59b6",
    "backpressure_kv": "#f39c12",
}

FALLBACK_COLORS = ["#1abc9c", "#e67e22", "#34495e", "#7f8c8d", "#2c3e50"]

CANONICAL_TIER_ORDER = [
    "round_robin", "kv_aware", "thompson_no_kv", "thompson_kv_on", "backpressure_kv",
]

DOMAIN_ORDER = ["banking", "healthcare", "insurance", "investment", "telecom"]

COLD_THRESHOLD_MS = 300


def get_tier_order(available_tiers):
    """Return canonical tier order filtered to available tiers, with extras appended."""
    ordered = [t for t in CANONICAL_TIER_ORDER if t in available_tiers]
    extras = sorted(set(available_tiers) - set(CANONICAL_TIER_ORDER))
    return ordered + extras


def get_tier_color(tier: str, idx: int = 0) -> str:
    if tier in TIER_COLORS:
        return TIER_COLORS[tier]
    return FALLBACK_COLORS[idx % len(FALLBACK_COLORS)]


def load_tier_data(tiers: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load per-LLM-call and per-request CSVs from each tier."""
    llm_call_frames = []
    request_frames = []

    for tier_name, comparison_dir in tiers.items():
        llm_csv = comparison_dir / "throughput_vs_tsq_per_llm_call_data.csv"
        req_csv = comparison_dir / "throughput_vs_tsq_per_request_data.csv"

        if llm_csv.exists():
            df = pd.read_csv(llm_csv)
            df["tier"] = tier_name
            llm_call_frames.append(df)
            print(f"  {tier_name}: {len(df)} LLM calls from {llm_csv}")
        else:
            print(f"  WARNING: {llm_csv} not found")

        if req_csv.exists():
            df = pd.read_csv(req_csv)
            df["tier"] = tier_name
            request_frames.append(df)
            print(f"  {tier_name}: {len(df)} requests from {req_csv}")
        else:
            print(f"  WARNING: {req_csv} not found")

    llm_df = pd.concat(llm_call_frames, ignore_index=True) if llm_call_frames else pd.DataFrame()
    req_df = pd.concat(request_frames, ignore_index=True) if request_frames else pd.DataFrame()
    return llm_df, req_df


def compute_tier_stats(llm_df: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate TTFT/ITL/TPS stats per tier."""
    rows = []
    for tier in llm_df["tier"].unique():
        td = llm_df[llm_df["tier"] == tier]
        rows.append({
            "tier": tier,
            "display_name": TIER_DISPLAY_NAMES.get(tier, tier),
            "n_calls": len(td),
            "ttft_median_ms": td["ttft_ms"].median(),
            "ttft_p95_ms": td["ttft_ms"].quantile(0.95),
            "ttft_mean_ms": td["ttft_ms"].mean(),
            "ttft_std_ms": td["ttft_ms"].std(),
            "itl_median_ms": td["itl_ms"].median(),
            "itl_p95_ms": td["itl_ms"].quantile(0.95),
            "tps_median": td["tps"].median(),
            "tps_mean": td["tps"].mean(),
            "cold_pct": (td["ttft_ms"] >= COLD_THRESHOLD_MS).mean() * 100,
            "warm_pct": (td["ttft_ms"] < COLD_THRESHOLD_MS).mean() * 100,
        })
    return pd.DataFrame(rows)


def compute_domain_stats(llm_df: pd.DataFrame) -> pd.DataFrame:
    """Compute TTFT stats per tier per domain."""
    rows = []
    for tier in llm_df["tier"].unique():
        for domain in DOMAIN_ORDER:
            td = llm_df[(llm_df["tier"] == tier) & (llm_df["experiment"] == domain)]
            if len(td) == 0:
                continue
            rows.append({
                "tier": tier,
                "display_name": TIER_DISPLAY_NAMES.get(tier, tier),
                "domain": domain,
                "n_calls": len(td),
                "ttft_median_ms": td["ttft_ms"].median(),
                "ttft_p95_ms": td["ttft_ms"].quantile(0.95),
                "ttft_mean_ms": td["ttft_ms"].mean(),
                "itl_median_ms": td["itl_ms"].median(),
                "tps_median": td["tps"].median(),
                "cold_pct": (td["ttft_ms"] >= COLD_THRESHOLD_MS).mean() * 100,
            })
    return pd.DataFrame(rows)


def compute_first_call_stats(llm_df: pd.DataFrame) -> pd.DataFrame:
    """Compare TTFT for first LLM call (idx=0) vs subsequent calls per tier."""
    rows = []
    for tier in llm_df["tier"].unique():
        td = llm_df[llm_df["tier"] == tier]
        first = td[td["llm_call_idx"] == 0]
        subsequent = td[td["llm_call_idx"] > 0]
        rows.append({
            "tier": tier,
            "display_name": TIER_DISPLAY_NAMES.get(tier, tier),
            "first_call_ttft_median_ms": first["ttft_ms"].median(),
            "first_call_ttft_p95_ms": first["ttft_ms"].quantile(0.95),
            "first_call_cold_pct": (first["ttft_ms"] >= COLD_THRESHOLD_MS).mean() * 100,
            "subsequent_ttft_median_ms": subsequent["ttft_ms"].median(),
            "subsequent_ttft_p95_ms": subsequent["ttft_ms"].quantile(0.95),
            "subsequent_cold_pct": (subsequent["ttft_ms"] >= COLD_THRESHOLD_MS).mean() * 100,
            "n_first": len(first),
            "n_subsequent": len(subsequent),
        })
    return pd.DataFrame(rows)


def plot_tier_bar_chart(tier_stats: pd.DataFrame, output_dir: Path):
    """Bar chart comparing median and p95 TTFT across tiers."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    tier_order = get_tier_order(tier_stats["tier"].values)
    stats = tier_stats.set_index("tier").loc[tier_order]
    labels = [TIER_DISPLAY_NAMES.get(t, t) for t in tier_order]
    colors = [get_tier_color(t, i) for i, t in enumerate(tier_order)]
    x = np.arange(len(tier_order))

    # TTFT
    ax = axes[0]
    bars_med = ax.bar(x - 0.2, stats["ttft_median_ms"], 0.35, label="Median", color=colors, alpha=0.8)
    bars_p95 = ax.bar(x + 0.2, stats["ttft_p95_ms"], 0.35, label="p95", color=colors, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Time To First Token")
    ax.legend()
    for bar in bars_med:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8)

    # ITL
    ax = axes[1]
    ax.bar(x, stats["itl_median_ms"], 0.5, color=colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("ITL (ms)")
    ax.set_title("Median Inter-Token Latency")
    for i, v in enumerate(stats["itl_median_ms"]):
        ax.text(i, v + 0.2, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    # TPS
    ax = axes[2]
    ax.bar(x, stats["tps_median"], 0.5, color=colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Tokens/sec")
    ax.set_title("Median Throughput (TPS)")
    for i, v in enumerate(stats["tps_median"]):
        ax.text(i, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Cross-Tier Performance Comparison — Multi-Domain Benchmark", fontsize=13, y=1.02)
    fig.tight_layout()
    path = output_dir / "cross_tier_ttft_itl_tps.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_domain_heatmap(domain_stats: pd.DataFrame, output_dir: Path):
    """Grouped bar chart of median TTFT per domain per tier."""
    tier_order = get_tier_order(domain_stats["tier"].values)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Median TTFT grouped bar chart
    ax = axes[0]
    n_tiers = len(tier_order)
    bar_width = 0.8 / n_tiers
    x = np.arange(len(DOMAIN_ORDER))

    for i, tier in enumerate(tier_order):
        td = domain_stats[domain_stats["tier"] == tier].set_index("domain")
        vals = [td.loc[d, "ttft_median_ms"] if d in td.index else 0 for d in DOMAIN_ORDER]
        offset = (i - n_tiers / 2 + 0.5) * bar_width
        ax.bar(x + offset, vals, bar_width, label=TIER_DISPLAY_NAMES.get(tier, tier),
               color=get_tier_color(tier, i), alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in DOMAIN_ORDER], rotation=15, ha="right")
    ax.set_ylabel("Median TTFT (ms)")
    ax.set_title("Median TTFT by Domain")
    ax.legend(fontsize=8)

    # Cold prefill percentage grouped bar chart
    ax = axes[1]
    for i, tier in enumerate(tier_order):
        td = domain_stats[domain_stats["tier"] == tier].set_index("domain")
        vals = [td.loc[d, "cold_pct"] if d in td.index else 0 for d in DOMAIN_ORDER]
        offset = (i - n_tiers / 2 + 0.5) * bar_width
        ax.bar(x + offset, vals, bar_width, label=TIER_DISPLAY_NAMES.get(tier, tier),
               color=get_tier_color(tier, i), alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in DOMAIN_ORDER], rotation=15, ha="right")
    ax.set_ylabel(f"Cold Prefills (TTFT >= {COLD_THRESHOLD_MS}ms) %")
    ax.set_title("Cold Prefill Rate by Domain")
    ax.legend(fontsize=8)

    fig.suptitle("Per-Domain Performance Comparison", fontsize=13, y=1.02)
    fig.tight_layout()
    path = output_dir / "cross_tier_domain_breakdown.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_ttft_distributions(llm_df: pd.DataFrame, output_dir: Path):
    """Overlaid TTFT histograms per tier showing cold/warm bimodality."""
    tier_order = get_tier_order(llm_df["tier"].unique())

    n_tiers = len(tier_order)
    ncols = min(n_tiers, 3)
    nrows = (n_tiers + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
    if n_tiers == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, tier in enumerate(tier_order):
        ax = axes[i]
        td = llm_df[llm_df["tier"] == tier]["ttft_ms"]
        clipped = td.clip(upper=td.quantile(0.99))

        ax.hist(clipped, bins=80, color=get_tier_color(tier, i),
                alpha=0.7, edgecolor="white", linewidth=0.3)
        ax.axvline(COLD_THRESHOLD_MS, color="red", linestyle="--", alpha=0.7, label=f"Cold threshold ({COLD_THRESHOLD_MS}ms)")
        ax.axvline(td.median(), color="black", linestyle="-", alpha=0.8, label=f"Median ({td.median():.0f}ms)")

        warm_pct = (td < COLD_THRESHOLD_MS).mean() * 100
        cold_pct = (td >= COLD_THRESHOLD_MS).mean() * 100
        ax.set_title(f"{TIER_DISPLAY_NAMES.get(tier, tier)}\n"
                     f"Warm: {warm_pct:.1f}% | Cold: {cold_pct:.1f}% | n={len(td)}")
        ax.set_xlabel("TTFT (ms)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    for j in range(len(tier_order), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("TTFT Distribution — Cold/Warm Bimodality Analysis", fontsize=13, y=1.01)
    fig.tight_layout()
    path = output_dir / "ttft_distribution_bimodality.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_first_call_penalty(first_call_stats: pd.DataFrame, output_dir: Path):
    """Bar chart comparing first-call vs subsequent-call TTFT per tier."""
    tier_order = get_tier_order(first_call_stats["tier"].values)
    stats = first_call_stats.set_index("tier").loc[tier_order]
    labels = [TIER_DISPLAY_NAMES.get(t, t) for t in tier_order]
    colors = [get_tier_color(t, i) for i, t in enumerate(tier_order)]
    x = np.arange(len(tier_order))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Median TTFT: first vs subsequent
    ax = axes[0]
    ax.bar(x - 0.2, stats["first_call_ttft_median_ms"], 0.35,
           label="First call (idx=0)", color=colors, alpha=0.5, edgecolor="black", linewidth=0.8)
    ax.bar(x + 0.2, stats["subsequent_ttft_median_ms"], 0.35,
           label="Subsequent calls", color=colors, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Median TTFT (ms)")
    ax.set_title("First-Call Penalty: Median TTFT")
    ax.legend()
    for i, (f, s) in enumerate(zip(stats["first_call_ttft_median_ms"], stats["subsequent_ttft_median_ms"])):
        ax.text(i - 0.2, f + 20, f"{f:.0f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + 0.2, s + 20, f"{s:.0f}", ha="center", va="bottom", fontsize=7)

    # Cold prefill %: first vs subsequent
    ax = axes[1]
    ax.bar(x - 0.2, stats["first_call_cold_pct"], 0.35,
           label="First call (idx=0)", color=colors, alpha=0.5, edgecolor="black", linewidth=0.8)
    ax.bar(x + 0.2, stats["subsequent_cold_pct"], 0.35,
           label="Subsequent calls", color=colors, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel(f"Cold Prefills (>= {COLD_THRESHOLD_MS}ms) %")
    ax.set_title("First-Call Penalty: Cold Prefill Rate")
    ax.legend()
    for i, (f, s) in enumerate(zip(stats["first_call_cold_pct"], stats["subsequent_cold_pct"])):
        ax.text(i - 0.2, f + 0.5, f"{f:.1f}%", ha="center", va="bottom", fontsize=7)
        ax.text(i + 0.2, s + 0.5, f"{s:.1f}%", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Session Stickiness Analysis — First Call vs Subsequent", fontsize=13, y=1.02)
    fig.tight_layout()
    path = output_dir / "first_call_penalty.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def print_summary_table(tier_stats: pd.DataFrame, domain_stats: pd.DataFrame,
                        first_call_stats: pd.DataFrame):
    """Print markdown-formatted summary tables to stdout."""
    tier_order = get_tier_order(tier_stats["tier"].values)

    print("\n" + "=" * 80)
    print("CROSS-TIER PERFORMANCE SUMMARY")
    print("=" * 80)

    print("\n### Overall TTFT/ITL/TPS by Router Tier\n")
    print("| Router | N calls | TTFT med (ms) | TTFT p95 (ms) | ITL med (ms) | TPS med | Cold % |")
    print("|--------|---------|---------------|---------------|-------------|---------|--------|")
    for tier in tier_order:
        row = tier_stats[tier_stats["tier"] == tier]
        if row.empty:
            continue
        r = row.iloc[0]
        print(f"| {r['display_name']} | {r['n_calls']} | {r['ttft_median_ms']:.1f} | "
              f"{r['ttft_p95_ms']:.1f} | {r['itl_median_ms']:.1f} | {r['tps_median']:.1f} | "
              f"{r['cold_pct']:.1f}% |")

    print("\n### Per-Domain Median TTFT (ms)\n")
    header = "| Router |"
    sep = "|--------|"
    for d in DOMAIN_ORDER:
        header += f" {d.capitalize()} |"
        sep += "---------|"
    print(header)
    print(sep)
    for tier in tier_order:
        td = domain_stats[domain_stats["tier"] == tier].set_index("domain")
        if td.empty:
            continue
        row = f"| {TIER_DISPLAY_NAMES.get(tier, tier)} |"
        for d in DOMAIN_ORDER:
            val = td.loc[d, "ttft_median_ms"] if d in td.index else float("nan")
            row += f" {val:.1f} |"
        print(row)

    print("\n### First-Call Penalty Analysis\n")
    print("| Router | First-call TTFT med | Subsequent TTFT med | First cold % | Subsequent cold % |")
    print("|--------|---------------------|---------------------|--------------|--------------------|")
    for tier in tier_order:
        row = first_call_stats[first_call_stats["tier"] == tier]
        if row.empty:
            continue
        r = row.iloc[0]
        print(f"| {r['display_name']} | {r['first_call_ttft_median_ms']:.1f} | "
              f"{r['subsequent_ttft_median_ms']:.1f} | {r['first_call_cold_pct']:.1f}% | "
              f"{r['subsequent_cold_pct']:.1f}% |")


def main():
    parser = argparse.ArgumentParser(description="Cross-tier TTFT comparison for multi-domain benchmark")
    parser.add_argument("--tiers", nargs="+", required=True,
                        help="Tier definitions as name=path pairs (e.g. round_robin=./path/to/comparison)")
    parser.add_argument("--output", "-o", required=True, help="Output directory for plots and CSVs")
    args = parser.parse_args()

    tiers = {}
    for spec in args.tiers:
        if "=" not in spec:
            print(f"ERROR: tier spec must be name=path, got: {spec}", file=sys.stderr)
            sys.exit(1)
        name, path = spec.split("=", 1)
        tiers[name] = Path(path)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tier data...")
    llm_df, req_df = load_tier_data(tiers)

    if llm_df.empty:
        print("ERROR: No per-LLM-call data loaded", file=sys.stderr)
        sys.exit(1)

    print(f"\nTotal LLM calls: {len(llm_df)}")
    print(f"Total requests: {len(req_df)}")

    print("\nComputing statistics...")
    tier_stats = compute_tier_stats(llm_df)
    domain_stats = compute_domain_stats(llm_df)
    first_call_stats = compute_first_call_stats(llm_df)

    print_summary_table(tier_stats, domain_stats, first_call_stats)

    print("\nGenerating plots...")
    plot_tier_bar_chart(tier_stats, output_dir)
    plot_domain_heatmap(domain_stats, output_dir)
    plot_ttft_distributions(llm_df, output_dir)
    plot_first_call_penalty(first_call_stats, output_dir)

    # Save CSVs
    tier_stats.to_csv(output_dir / "tier_aggregate_stats.csv", index=False)
    domain_stats.to_csv(output_dir / "domain_breakdown_stats.csv", index=False)
    first_call_stats.to_csv(output_dir / "first_call_penalty_stats.csv", index=False)
    print(f"\n  Saved: {output_dir / 'tier_aggregate_stats.csv'}")
    print(f"  Saved: {output_dir / 'domain_breakdown_stats.csv'}")
    print(f"  Saved: {output_dir / 'first_call_penalty_stats.csv'}")

    print(f"\nDone! All outputs in: {output_dir}")


if __name__ == "__main__":
    main()
