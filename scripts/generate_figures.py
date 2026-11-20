#!/usr/bin/env python3
"""Export high-DPI KPI charts used by the README and Gradio explorer."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics import (  # noqa: E402
    airport_dropoff_breakdown,
    hourly_demand,
    load_taxi_frame,
    trip_speeds_mph,
)
from src.paths import ASSETS_DIR  # noqa: E402

# Calm teal/slate palette — avoid the usual AI purple/cream defaults.
COLORS = {
    "bar": "#0F766E",
    "bar_alt": "#334155",
    "hist": "#0EA5E9",
    "scatter": "#115E59",
    "edge": "#0F172A",
    "grid": "#CBD5E1",
}


def _style(ax) -> None:
    ax.set_facecolor("#F8FAFC")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94A3B8")
    ax.spines["bottom"].set_color("#94A3B8")


def export_hourly_demand(df, out: Path) -> None:
    from src.analytics import TIME_BAND_LABELS

    counts = hourly_demand(df)
    labels = [TIME_BAND_LABELS[i] for i in counts.index]
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=160)
    ax.bar(labels, counts.values, color=COLORS["bar"], edgecolor=COLORS["edge"], linewidth=0.4)
    ax.set_title("Pickup demand by time of day (NYC taxis)")
    ax.set_xlabel("Pickup time bucket")
    ax.set_ylabel("Rides")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def export_airport_dropoffs(df, out: Path) -> None:
    breakdown = airport_dropoff_breakdown(df)
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=160)
    ax.bar(
        breakdown.index,
        breakdown.values,
        color=[COLORS["bar"], COLORS["hist"], COLORS["bar_alt"]],
        edgecolor=COLORS["edge"],
        linewidth=0.4,
    )
    ax.set_title("Airport drop-offs (JFK / LGA / EWR)")
    ax.set_ylabel("Rides")
    for i, v in enumerate(breakdown.values):
        ax.text(i, v + max(breakdown.values) * 0.02, f"{v:,}", ha="center", fontsize=10)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def export_speed_vs_distance(df, out: Path) -> None:
    speeds = trip_speeds_mph(df)
    sample = df.assign(speed_mph=speeds)
    sample = sample[(sample["speed_mph"] > 0) & (sample["speed_mph"] < 80)]
    sample = sample[(sample["trip_distance"] > 0) & (sample["trip_distance"] < 40)]
    sample = sample.sample(n=min(8_000, len(sample)), random_state=42)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.scatter(
        sample["trip_distance"],
        sample["speed_mph"],
        s=8,
        alpha=0.25,
        c=COLORS["scatter"],
        linewidths=0,
    )
    ax.set_title("Trip speed vs distance")
    ax.set_xlabel("Trip distance (miles)")
    ax.set_ylabel("Speed (mph)")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def export_fare_breakdown(df, out: Path) -> None:
    components = {
        "Fare": float(df["fare_amount"].mean()),
        "Fees": float(df["fees_amount"].mean()),
        "Tolls": float(df["tolls_amount"].mean()),
        "Tip": float(df["tip_amount"].mean()),
    }
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=160)
    labels = list(components.keys())
    values = list(components.values())
    ax.bar(labels, values, color=COLORS["bar"], edgecolor=COLORS["edge"], linewidth=0.4)
    ax.set_title("Average fare components per trip")
    ax.set_ylabel("USD")
    for i, v in enumerate(values):
        ax.text(i, v + 0.4, f"${v:.2f}", ha="center", fontsize=10)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_taxi_frame()
    export_hourly_demand(df, ASSETS_DIR / "hourly_demand.png")
    export_airport_dropoffs(df, ASSETS_DIR / "airport_dropoffs.png")
    export_speed_vs_distance(df, ASSETS_DIR / "speed_vs_distance.png")
    export_fare_breakdown(df, ASSETS_DIR / "fare_breakdown.png")
    print(f"Wrote charts to {ASSETS_DIR}")


if __name__ == "__main__":
    main()
