"""KPI benchmarks + fare model sanity tests for the NYC taxi portfolio upgrade."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.analytics import (
    BENCHMARK,
    airport_dropoff_breakdown,
    compute_benchmark_kpis,
    filter_trips,
    filtered_summary,
    hourly_demand,
    load_taxi_array,
    load_taxi_frame,
)
from src.fare_model import METRICS_PATH, MODEL_PATH, load_model, predict_fare, train
from src.paths import ASSETS_DIR, ROOT, resolve_data_path

DATA = ROOT / "data" / "nyc_taxis.csv"


@pytest.fixture(scope="module")
def df():
    assert DATA.is_file(), "data/nyc_taxis.csv must ship with the repo"
    return load_taxi_frame(DATA)


@pytest.fixture(scope="module")
def trained_metrics(df):
    metrics = train(DATA)
    assert MODEL_PATH.is_file()
    assert METRICS_PATH.is_file()
    assert metrics["best"]["mae_total"] < 15.0
    return metrics


@pytest.fixture(scope="module")
def artifact(trained_metrics):
    return load_model(MODEL_PATH)


def test_resolve_data_path_prefers_data_dir():
    path = resolve_data_path()
    assert path == DATA
    assert path.is_file()


def test_dataset_shape(df):
    assert len(df) == BENCHMARK.n_rows
    assert "trip_distance" in df.columns
    assert "tip_amount" in df.columns


def test_mean_speed_benchmark(df):
    kpis = compute_benchmark_kpis(df)
    assert kpis.mean_speed_mph == pytest.approx(BENCHMARK.mean_speed_mph, rel=1e-6)
    assert round(kpis.mean_speed_mph, 2) == 32.24


def test_february_rides_benchmark(df):
    kpis = compute_benchmark_kpis(df)
    assert kpis.february_rides == BENCHMARK.february_rides == 13_333


def test_tips_over_50_benchmark(df):
    kpis = compute_benchmark_kpis(df)
    assert kpis.tips_over_50 == BENCHMARK.tips_over_50 == 16


def test_airport_dropoff_benchmarks(df):
    kpis = compute_benchmark_kpis(df)
    assert kpis.jfk_dropoffs == BENCHMARK.jfk_dropoffs == 11_832
    assert kpis.lga_dropoffs == BENCHMARK.lga_dropoffs == 16_602
    assert kpis.ewr_dropoffs == BENCHMARK.ewr_dropoffs == 63
    assert kpis.airport_dropoffs_total == BENCHMARK.airport_dropoffs_total == 28_497
    breakdown = airport_dropoff_breakdown(df)
    assert int(breakdown["JFK"]) == 11_832
    assert int(breakdown["LGA"]) == 16_602
    assert int(breakdown["EWR"]) == 63


def test_numpy_loader_matches_frame(df):
    arr = load_taxi_array(DATA)
    assert arr.shape[0] == len(df)
    assert arr.shape[1] == 15


def test_hourly_demand_covers_day(df):
    counts = hourly_demand(df)
    assert len(counts) == 6
    assert int(counts.sum()) == len(df)


def test_filter_and_summary(df):
    filtered = filter_trips(df, months=[2], time_bands=[2, 3], airport="JFK")
    summary = filtered_summary(filtered)
    assert summary["rides"] == len(filtered)
    assert summary["rides"] > 0
    assert summary["avg_speed_mph"] > 0


def test_fare_model_metrics(trained_metrics):
    best = trained_metrics["best"]
    assert best["name"] in {"ridge", "hist_gradient_boosting"}
    assert best["r2_total"] > 0.70
    assert best["mae_total"] < 12.0


def test_fare_prediction_sanity(artifact):
    pred = predict_fare(
        trip_distance=17.5,
        pickup_time=4,
        pickup_dayofweek=5,
        pickup_month=3,
        pickup_location_code=2,
        dropoff_location_code=4,
        estimated_duration_min=45.0,
        artifact=artifact,
    )
    assert pred.total_fare > 10
    assert pred.total_fare < 150
    assert pred.tip >= 0
    assert pred.tip < pred.total_fare
    assert pred.estimated_duration_min == pytest.approx(45.0)


def test_short_trip_cheaper_than_long(artifact):
    short = predict_fare(
        trip_distance=2.0,
        pickup_time=3,
        pickup_location_code=4,
        dropoff_location_code=4,
        estimated_duration_min=12.0,
        artifact=artifact,
    )
    long = predict_fare(
        trip_distance=20.0,
        pickup_time=3,
        pickup_location_code=2,
        dropoff_location_code=4,
        estimated_duration_min=50.0,
        artifact=artifact,
    )
    assert long.total_fare > short.total_fare


def test_assets_exist_or_skippable():
    # Charts are generated in CI/local via scripts/generate_figures.py; require when present.
    expected = [
        "hourly_demand.png",
        "airport_dropoffs.png",
        "speed_vs_distance.png",
        "fare_breakdown.png",
    ]
    present = [ASSETS_DIR / name for name in expected if (ASSETS_DIR / name).is_file()]
    # After generate_figures in the upgrade flow, all four should exist.
    if present:
        assert len(present) == 4
