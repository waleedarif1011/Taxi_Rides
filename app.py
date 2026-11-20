"""Gradio NYC Taxi Explorer & Fare Calculator (Hugging Face Space entrypoint)."""

from __future__ import annotations

import json

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd

from src.analytics import (
    TIME_BAND_LABELS,
    compute_benchmark_kpis,
    filter_trips,
    filtered_summary,
    hourly_demand,
    load_taxi_frame,
)
from src.fare_model import METRICS_PATH, MODEL_PATH, load_model, predict_fare
from src.paths import ASSETS_DIR, ROOT

ZONE_CHOICES = [
    ("Manhattan (4)", 4),
    ("JFK Airport (2)", 2),
    ("LaGuardia / LGA (3)", 3),
    ("EWR / Newark (5)", 5),
    ("Queens / Brooklyn (6)", 6),
    ("Other / Unknown (1)", 1),
]
ZONE_LABEL_TO_CODE = {label: code for label, code in ZONE_CHOICES}
ZONE_LABELS = [label for label, _ in ZONE_CHOICES]

DAY_CHOICES = [
    ("Monday", 1),
    ("Tuesday", 2),
    ("Wednesday", 3),
    ("Thursday", 4),
    ("Friday", 5),
    ("Saturday", 6),
    ("Sunday", 7),
]
DAY_LABEL_TO_CODE = {label: code for label, code in DAY_CHOICES}

TIME_BAND_CHOICES = [f"{code}: {label}" for code, label in TIME_BAND_LABELS.items()]
TIME_BAND_TO_CODE = {f"{code}: {label}": code for code, label in TIME_BAND_LABELS.items()}

_DF: pd.DataFrame | None = None
_ARTIFACT = None


def _get_df() -> pd.DataFrame:
    global _DF
    if _DF is None:
        _DF = load_taxi_frame()
    return _DF


def _get_artifact():
    global _ARTIFACT
    if _ARTIFACT is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                "artifacts/fare_model.joblib missing — run `python -m src.fare_model` first."
            )
        _ARTIFACT = load_model(MODEL_PATH)
    return _ARTIFACT


def _metrics_blurb() -> str:
    if not METRICS_PATH.is_file():
        return "_Train with `python -m src.fare_model` to populate metrics._"
    m = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    best = m.get("best", {})
    rows = "\n".join(
        f"| `{r['name']}` | ${r['mae_total']:.2f} | ${r['mae_tip']:.2f} | {r['r2_total']:.3f} |"
        for r in m.get("models", [])
    )
    return (
        f"**Best estimator:** `{best.get('name', '—')}` "
        f"(total MAE ${best.get('mae_total', float('nan')):.2f}, "
        f"tip MAE ${best.get('mae_tip', float('nan')):.2f})\n\n"
        "| Model | MAE total | MAE tip | R² total |\n"
        "|-------|-----------|---------|----------|\n"
        f"{rows}\n\n"
        "_Honest positioning: this is an **exploratory demo** on ~89k historical trips, "
        "not a production dispatch or pricing system._"
    )


def _time_band_selection(label: str) -> list[int] | None:
    if label == "All day":
        return None
    mapping = {
        "Morning (04–11)": [1, 2],
        "Afternoon (12–15)": [3],
        "Evening (16–19)": [4],
        "Night (20–03)": [5, 0],
    }
    return mapping[label]


def explore(months: list[str], hour_band: str, airport: str):
    df = _get_df()
    month_ints = [int(m) for m in months] if months else list(range(1, 7))
    bands = _time_band_selection(hour_band)
    airport_key = None if airport == "All airports" else airport
    filtered = filter_trips(df, months=month_ints, time_bands=bands, airport=airport_key)
    summary = filtered_summary(filtered)

    md = (
        f"### Filtered slice\n\n"
        f"- **Rides:** {summary['rides']:,}\n"
        f"- **Avg speed:** {summary['avg_speed_mph']:.2f} mph\n"
        f"- **Avg tip % of fare:** {summary['avg_tip_pct']:.1f}%\n"
        f"- **Avg fare:** ${summary['avg_fare']:.2f}\n"
        f"- **Avg distance:** {summary['avg_distance']:.1f} mi\n"
        f"- **Avg duration:** {summary['avg_duration_min']:.1f} min\n"
    )

    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=120)
    if len(filtered):
        counts = hourly_demand(filtered)
        labels = [TIME_BAND_LABELS[i] for i in counts.index]
        ax.bar(labels, counts.values, color="#0F766E", edgecolor="#0F172A", linewidth=0.3)
    ax.set_title("Demand by time of day (filtered)")
    ax.set_xlabel("Time bucket")
    ax.set_ylabel("Rides")
    ax.set_facecolor("#F8FAFC")
    ax.grid(axis="y", color="#CBD5E1", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    chart = None
    for name in ("hourly_demand.png", "airport_dropoffs.png", "fare_breakdown.png"):
        path = ASSETS_DIR / name
        if path.is_file():
            chart = str(path)
            break
    return md, fig, chart


def calculate_fare(
    distance: float,
    time_band_label: str,
    day_label: str,
    month: int,
    pickup_zone: str,
    dropoff_zone: str,
    duration_min: float,
):
    duration = float(duration_min) if duration_min and duration_min > 0 else None
    pred = predict_fare(
        trip_distance=float(distance),
        pickup_time=TIME_BAND_TO_CODE[time_band_label],
        pickup_dayofweek=DAY_LABEL_TO_CODE[day_label],
        pickup_month=int(month),
        pickup_location_code=ZONE_LABEL_TO_CODE[pickup_zone],
        dropoff_location_code=ZONE_LABEL_TO_CODE[dropoff_zone],
        estimated_duration_min=duration,
        artifact=_get_artifact(),
    )
    tip_pct = (pred.tip / pred.total_fare * 100.0) if pred.total_fare else 0.0
    summary = (
        f"### Estimate\n\n"
        f"- **Predicted total:** **${pred.total_fare:.2f}**\n"
        f"- **Expected tip:** **${pred.tip:.2f}** ({tip_pct:.0f}% of total)\n"
        f"- **Typical duration:** {pred.estimated_duration_min:.1f} min\n\n"
        f"{distance:.1f} mi · {time_band_label} · {day_label} · month {month}\n"
        f"{pickup_zone} → {dropoff_zone}"
    )
    return (
        f"${pred.total_fare:.2f}",
        f"${pred.tip:.2f}",
        f"{pred.estimated_duration_min:.1f} min",
        summary,
    )


def apply_preset(name: str):
    presets = {
        "JFK → Manhattan evening": (
            17.5,
            "4: 16–19",
            "Friday",
            3,
            "JFK Airport (2)",
            "Manhattan (4)",
            45.0,
        ),
        "LGA short hop": (
            8.0,
            "2: 08–11",
            "Tuesday",
            2,
            "LaGuardia / LGA (3)",
            "Manhattan (4)",
            28.0,
        ),
        "Late-night across town": (
            4.2,
            "5: 20–23",
            "Saturday",
            5,
            "Manhattan (4)",
            "Queens / Brooklyn (6)",
            18.0,
        ),
    }
    return presets.get(name, presets["JFK → Manhattan evening"])


def build_ui() -> gr.Blocks:
    kpis = compute_benchmark_kpis(_get_df())
    with gr.Blocks(title="NYC Taxi Explorer & Fare Calculator") as demo:
        gr.Markdown(
            "# NYC Taxi Explorer & Fare Calculator\n\n"
            "Interactive EDA + trip fare/tip estimator on **~89k** NYC taxi trips "
            "(Jan–Jun 2016 sample).\n\n"
            f"**Benchmarks:** mean speed **{kpis.mean_speed_mph:.2f} mph** · "
            f"February rides **{kpis.february_rides:,}** · "
            f"tips > $50 **{kpis.tips_over_50}** · "
            f"airport drop-offs **{kpis.airport_dropoffs_total:,}** "
            f"(JFK {kpis.jfk_dropoffs:,} / LGA {kpis.lga_dropoffs:,} / EWR {kpis.ewr_dropoffs:,})\n\n"
            + _metrics_blurb()
        )
        with gr.Tabs():
            with gr.Tab("KPI Explorer"):
                with gr.Row():
                    with gr.Column(scale=2):
                        months = gr.CheckboxGroup(
                            choices=[str(m) for m in range(1, 7)],
                            value=["1", "2", "3", "4", "5", "6"],
                            label="Pickup months (2016)",
                        )
                        hour_band = gr.Radio(
                            choices=[
                                "All day",
                                "Morning (04–11)",
                                "Afternoon (12–15)",
                                "Evening (16–19)",
                                "Night (20–03)",
                            ],
                            value="All day",
                            label="Time of day",
                        )
                        airport = gr.Radio(
                            choices=["All airports", "JFK", "LGA", "EWR"],
                            value="All airports",
                            label="Airport drop-off filter",
                        )
                        explore_btn = gr.Button("Update explorer", variant="primary")
                    with gr.Column(scale=3):
                        explore_md = gr.Markdown()
                        explore_plot = gr.Plot(label="Demand by time of day")
                        explore_chart = gr.Image(
                            label="Saved KPI chart",
                            type="filepath",
                            value=str(ASSETS_DIR / "airport_dropoffs.png")
                            if (ASSETS_DIR / "airport_dropoffs.png").is_file()
                            else None,
                        )
                explore_btn.click(
                    fn=explore,
                    inputs=[months, hour_band, airport],
                    outputs=[explore_md, explore_plot, explore_chart],
                )
                demo.load(
                    fn=explore,
                    inputs=[months, hour_band, airport],
                    outputs=[explore_md, explore_plot, explore_chart],
                )

            with gr.Tab("Fare & Tip Calculator"):
                with gr.Row():
                    with gr.Column():
                        preset = gr.Dropdown(
                            choices=[
                                "JFK → Manhattan evening",
                                "LGA short hop",
                                "Late-night across town",
                            ],
                            value="JFK → Manhattan evening",
                            label="Trip presets",
                        )
                        distance = gr.Slider(0.5, 40.0, value=17.5, step=0.1, label="Trip distance (miles)")
                        time_band = gr.Dropdown(
                            choices=TIME_BAND_CHOICES,
                            value="4: 16–19",
                            label="Pickup time of day",
                        )
                        day_label = gr.Dropdown(
                            choices=[d for d, _ in DAY_CHOICES],
                            value="Friday",
                            label="Day of week",
                        )
                        month = gr.Slider(1, 6, value=3, step=1, label="Month")
                        pickup_zone = gr.Dropdown(choices=ZONE_LABELS, value="JFK Airport (2)", label="Pickup zone")
                        dropoff_zone = gr.Dropdown(choices=ZONE_LABELS, value="Manhattan (4)", label="Dropoff zone")
                        duration = gr.Number(
                            value=45.0,
                            label="Estimated duration (min, optional)",
                            info="Leave ≤ 0 to use a distance-based heuristic.",
                        )
                        calc_btn = gr.Button("Estimate fare & tip", variant="primary")
                    with gr.Column():
                        total_out = gr.Textbox(label="Predicted total ($)", interactive=False)
                        tip_out = gr.Textbox(label="Expected tip ($)", interactive=False)
                        dur_out = gr.Textbox(label="Typical duration", interactive=False)
                        detail = gr.Markdown()

                preset.change(
                    fn=apply_preset,
                    inputs=[preset],
                    outputs=[distance, time_band, day_label, month, pickup_zone, dropoff_zone, duration],
                )
                calc_inputs = [
                    distance,
                    time_band,
                    day_label,
                    month,
                    pickup_zone,
                    dropoff_zone,
                    duration,
                ]
                calc_btn.click(fn=calculate_fare, inputs=calc_inputs, outputs=[total_out, tip_out, dur_out, detail])
                demo.load(fn=calculate_fare, inputs=calc_inputs, outputs=[total_out, tip_out, dur_out, detail])

        gr.Markdown(
            "---\n"
            f"Source: [`{ROOT.name}`](https://github.com/waleedarif1011/taxi_rides) · "
            "MIT License · Exploratory analytics demo"
        )
    return demo


demo = build_ui()

if __name__ == "__main__":
    demo.launch()
