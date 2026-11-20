"""Reproducible NYC taxi KPIs and filtered summaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.paths import resolve_data_path

# Location codes used by this Dataquest-style nyc_taxis sample.
# Airport drop-offs: 2=JFK, 3=LGA, 5=EWR (locked by portfolio KPI tests).
LOCATION_LABELS = {
    0: "Unknown",
    1: "Other borough",
    2: "JFK",
    3: "LaGuardia (LGA)",
    4: "Manhattan",
    5: "EWR / Newark",
    6: "Queens / Brooklyn",
    7: "Bronx / Other",
}

AIRPORT_CODES = {
    "JFK": 2,
    "LGA": 3,
    "EWR": 5,
}

# ``pickup_time`` in this dataset is a 0–5 time-of-day bucket (not clock hour).
TIME_BAND_LABELS = {
    0: "00–03",
    1: "04–07",
    2: "08–11",
    3: "12–15",
    4: "16–19",
    5: "20–23",
}

# TLC-style payment codes in this Dataquest sample.
PAYMENT_LABELS = {
    1: "Credit card",
    2: "Cash",
    3: "No charge",
    4: "Dispute",
}

COLUMN_NAMES = [
    "pickup_year",
    "pickup_month",
    "pickup_day",
    "pickup_dayofweek",
    "pickup_time",
    "pickup_location_code",
    "dropoff_location_code",
    "trip_distance",
    "trip_length",
    "fare_amount",
    "fees_amount",
    "tolls_amount",
    "tip_amount",
    "total_amount",
    "payment_type",
]


@dataclass(frozen=True)
class BenchmarkKPIs:
    """Canonical numbers verified by the portfolio test suite."""

    n_rows: int
    mean_speed_mph: float
    february_rides: int
    tips_over_50: int
    jfk_dropoffs: int
    lga_dropoffs: int
    ewr_dropoffs: int
    airport_dropoffs_total: int


# Locked against the shipped ``data/nyc_taxis.csv``.
BENCHMARK = BenchmarkKPIs(
    n_rows=89_560,
    mean_speed_mph=32.24258580925573,
    february_rides=13_333,
    tips_over_50=16,
    jfk_dropoffs=11_832,
    lga_dropoffs=16_602,
    ewr_dropoffs=63,
    airport_dropoffs_total=28_497,
)


def load_taxi_frame(path: str | Path | None = None) -> pd.DataFrame:
    """Load the NYC taxi CSV as a typed DataFrame."""
    csv_path = resolve_data_path(path)
    df = pd.read_csv(csv_path)
    expected = set(COLUMN_NAMES)
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")
    return df


def load_taxi_array(path: str | Path | None = None) -> np.ndarray:
    """Load the dataset as a NumPy array (notebook-compatible)."""
    csv_path = resolve_data_path(path)
    return np.genfromtxt(csv_path, delimiter=",", skip_header=1)


def trip_speeds_mph(df: pd.DataFrame) -> pd.Series:
    """Speed (mph) for trips with positive duration."""
    length = df["trip_length"].astype(float)
    distance = df["trip_distance"].astype(float)
    valid = length > 0
    speeds = pd.Series(np.nan, index=df.index, dtype=float)
    speeds.loc[valid] = distance.loc[valid] / (length.loc[valid] / 3600.0)
    return speeds


def compute_benchmark_kpis(df: pd.DataFrame | None = None) -> BenchmarkKPIs:
    """Compute the locked portfolio KPIs from the dataset."""
    frame = df if df is not None else load_taxi_frame()
    speeds = trip_speeds_mph(frame)
    mean_speed = float(speeds.dropna().mean())
    drop = frame["dropoff_location_code"].astype(int)
    jfk = int((drop == AIRPORT_CODES["JFK"]).sum())
    lga = int((drop == AIRPORT_CODES["LGA"]).sum())
    ewr = int((drop == AIRPORT_CODES["EWR"]).sum())
    return BenchmarkKPIs(
        n_rows=int(len(frame)),
        mean_speed_mph=mean_speed,
        february_rides=int((frame["pickup_month"] == 2).sum()),
        tips_over_50=int((frame["tip_amount"] > 50).sum()),
        jfk_dropoffs=jfk,
        lga_dropoffs=lga,
        ewr_dropoffs=ewr,
        airport_dropoffs_total=jfk + lga + ewr,
    )


def hourly_demand(df: pd.DataFrame) -> pd.Series:
    """Ride counts by pickup time-of-day bucket (0–5)."""
    bands = df["pickup_time"].astype(int)
    return bands.value_counts().reindex(range(6), fill_value=0).sort_index()


def monthly_demand(df: pd.DataFrame) -> pd.Series:
    """Ride counts by pickup month (1–12)."""
    months = df["pickup_month"].astype(int)
    return months.value_counts().reindex(range(1, 13), fill_value=0).sort_index()


def airport_dropoff_breakdown(df: pd.DataFrame) -> pd.Series:
    """Airport drop-off counts labeled JFK / LGA / EWR."""
    drop = df["dropoff_location_code"].astype(int)
    return pd.Series(
        {
            "JFK": int((drop == AIRPORT_CODES["JFK"]).sum()),
            "LGA": int((drop == AIRPORT_CODES["LGA"]).sum()),
            "EWR": int((drop == AIRPORT_CODES["EWR"]).sum()),
        }
    )


def tip_pct_of_fare(df: pd.DataFrame) -> pd.Series:
    """Tip as a percent of fare (fare clipped away from zero)."""
    fare = df["fare_amount"].astype(float).clip(lower=0.01)
    return (df["tip_amount"].astype(float) / fare) * 100.0


def tip_rate_by_payment(df: pd.DataFrame) -> pd.DataFrame:
    """Median tip % of fare, ride count, and share with a recorded tip by payment type.

    Cash / no-charge / dispute trips almost never carry an electronic tip field —
    a classic NYC TLC EDA caveat, not evidence that riders tip $0.
    """
    frame = df.copy()
    frame["tip_pct"] = tip_pct_of_fare(frame)
    frame["payment_type"] = frame["payment_type"].astype(int)
    rows: list[dict[str, Any]] = []
    for code in sorted(frame["payment_type"].unique()):
        subset = frame[frame["payment_type"] == code]
        label = PAYMENT_LABELS.get(code, f"Type {code}")
        rows.append(
            {
                "payment_type": code,
                "label": label,
                "rides": int(len(subset)),
                "median_tip_pct": float(subset["tip_pct"].median()),
                "mean_tip_pct": float(subset["tip_pct"].mean()),
                "pct_with_tip": float((subset["tip_amount"].astype(float) > 0).mean() * 100.0),
            }
        )
    return pd.DataFrame(rows).sort_values("payment_type").reset_index(drop=True)


def tip_rate_by_hour(
    df: pd.DataFrame,
    *,
    payment_type: int | None = 1,
) -> pd.Series:
    """Median tip % of fare by pickup time bucket.

    Defaults to credit-card trips (``payment_type=1``) so the series is not
    flattened by cash trips that record tip_amount = 0.
    """
    frame = df
    if payment_type is not None:
        frame = frame[frame["payment_type"].astype(int) == int(payment_type)]
    tip_pct = tip_pct_of_fare(frame)
    bands = frame["pickup_time"].astype(int)
    return (
        tip_pct.groupby(bands)
        .median()
        .reindex(range(6), fill_value=float("nan"))
        .sort_index()
    )


def filter_trips(
    df: pd.DataFrame,
    *,
    months: list[int] | None = None,
    hours: list[int] | None = None,
    time_bands: list[int] | None = None,
    airport: str | None = None,
) -> pd.DataFrame:
    """Filter trips by month, time-of-day bucket, and/or airport drop-off code."""
    out = df
    if months:
        out = out[out["pickup_month"].isin(months)]
    bands = time_bands if time_bands is not None else hours
    if bands:
        out = out[out["pickup_time"].isin(bands)]
    if airport and airport.upper() != "ALL":
        code = AIRPORT_CODES.get(airport.upper())
        if code is None:
            raise ValueError(f"Unknown airport '{airport}'. Use JFK, LGA, EWR, or ALL.")
        out = out[out["dropoff_location_code"] == code]
    return out.copy()


def filtered_summary(df: pd.DataFrame) -> dict[str, Any]:
    """KPI summary for a filtered slice (used by the Gradio explorer)."""
    if len(df) == 0:
        return {
            "rides": 0,
            "avg_speed_mph": float("nan"),
            "avg_tip_pct": float("nan"),
            "avg_fare": float("nan"),
            "avg_distance": float("nan"),
            "avg_duration_min": float("nan"),
        }
    speeds = trip_speeds_mph(df)
    fare = df["fare_amount"].astype(float).clip(lower=0.01)
    tip_pct = (df["tip_amount"].astype(float) / fare) * 100.0
    duration_min = df["trip_length"].astype(float) / 60.0
    return {
        "rides": int(len(df)),
        "avg_speed_mph": float(speeds.dropna().mean()) if speeds.notna().any() else float("nan"),
        "avg_tip_pct": float(tip_pct.mean()),
        "avg_fare": float(df["fare_amount"].mean()),
        "avg_distance": float(df["trip_distance"].mean()),
        "avg_duration_min": float(duration_min.mean()),
    }


def kpis_as_dict(kpis: BenchmarkKPIs | None = None) -> dict[str, Any]:
    return asdict(kpis or compute_benchmark_kpis())
