"""Train and serve a lightweight NYC taxi fare + tip estimator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.analytics import load_taxi_frame
from src.paths import ARTIFACTS_DIR, resolve_data_path

FEATURE_COLUMNS = [
    "pickup_time",  # dataset time-of-day bucket 0–5
    "pickup_dayofweek",
    "pickup_month",
    "pickup_location_code",
    "dropoff_location_code",
    "trip_distance",
    "estimated_duration_min",
]

TARGET_COLUMNS = ["total_amount", "tip_amount"]

MODEL_PATH = ARTIFACTS_DIR / "fare_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"


@dataclass
class FarePrediction:
    total_fare: float
    tip: float
    estimated_duration_min: float


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["pickup_time"] = frame["pickup_time"].astype(int)
    frame["pickup_dayofweek"] = frame["pickup_dayofweek"].astype(int)
    frame["pickup_month"] = frame["pickup_month"].astype(int)
    frame["pickup_location_code"] = frame["pickup_location_code"].astype(int)
    frame["dropoff_location_code"] = frame["dropoff_location_code"].astype(int)
    frame["trip_distance"] = frame["trip_distance"].astype(float)
    frame["estimated_duration_min"] = frame["trip_length"].astype(float) / 60.0
    frame["total_amount"] = frame["total_amount"].astype(float)
    frame["tip_amount"] = frame["tip_amount"].astype(float)

    # Drop extreme / broken rows that hurt a demo model.
    mask = (
        (frame["trip_distance"] > 0)
        & (frame["estimated_duration_min"] > 0)
        & (frame["estimated_duration_min"] < 180)
        & (frame["total_amount"] > 0)
        & (frame["total_amount"] < 250)
        & (frame["tip_amount"] >= 0)
        & (frame["tip_amount"] < 80)
    )
    return frame.loc[mask, FEATURE_COLUMNS + TARGET_COLUMNS].reset_index(drop=True)


def build_feature_matrix(rows: pd.DataFrame) -> pd.DataFrame:
    """Ensure feature columns exist and are ordered for inference."""
    out = rows.copy()
    if "estimated_duration_min" not in out.columns:
        if "trip_length" in out.columns:
            out["estimated_duration_min"] = out["trip_length"].astype(float) / 60.0
        else:
            # Fallback heuristic: ~2.5 min per mile + base time.
            out["estimated_duration_min"] = 3.0 + out["trip_distance"].astype(float) * 2.4
    if "pickup_time" not in out.columns and "pickup_hour" in out.columns:
        # Accept legacy hour (0–23) by mapping into the dataset's 0–5 buckets.
        out["pickup_time"] = (out["pickup_hour"].astype(int) // 4).clip(0, 5)
    return out[FEATURE_COLUMNS].astype(float)


def _candidate_models(random_state: int = 42) -> dict[str, Any]:
    ridge = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", MultiOutputRegressor(Ridge(alpha=1.0))),
        ]
    )
    hgb = MultiOutputRegressor(
        HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.08,
            max_iter=120,
            random_state=random_state,
        )
    )
    return {"ridge": ridge, "hist_gradient_boosting": hgb}


def train(
    data_path: str | Path | None = None,
    *,
    artifacts_dir: Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train fare/tip regressors, pick the best, and persist artifacts."""
    csv_path = resolve_data_path(data_path)
    raw = load_taxi_frame(csv_path)
    prepared = _prepare_frame(raw)
    x = prepared[FEATURE_COLUMNS]
    y = prepared[TARGET_COLUMNS]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    metrics: dict[str, Any] = {
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "features": FEATURE_COLUMNS,
        "targets": TARGET_COLUMNS,
        "models": [],
    }

    best_name = None
    best_mae = float("inf")
    best_model = None

    for name, model in _candidate_models(random_state).items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        mae_total = float(mean_absolute_error(y_test["total_amount"], pred[:, 0]))
        mae_tip = float(mean_absolute_error(y_test["tip_amount"], pred[:, 1]))
        r2_total = float(r2_score(y_test["total_amount"], pred[:, 0]))
        r2_tip = float(r2_score(y_test["tip_amount"], pred[:, 1]))
        entry = {
            "name": name,
            "mae_total": mae_total,
            "mae_tip": mae_tip,
            "r2_total": r2_total,
            "r2_tip": r2_tip,
            "mae_combined": mae_total + mae_tip,
        }
        metrics["models"].append(entry)
        if entry["mae_combined"] < best_mae:
            best_mae = entry["mae_combined"]
            best_name = name
            best_model = model

    assert best_model is not None and best_name is not None
    # Refit on full cleaned data for the shipped artifact.
    best_model.fit(x, y)
    metrics["best"] = {
        "name": best_name,
        **next(m for m in metrics["models"] if m["name"] == best_name),
    }

    out_dir = artifacts_dir or ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "fare_model.joblib"
    metrics_path = out_dir / "metrics.json"
    payload = {
        "model": best_model,
        "features": FEATURE_COLUMNS,
        "targets": TARGET_COLUMNS,
        "best_name": best_name,
    }
    joblib.dump(payload, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metrics["model_path"] = str(model_path)
    metrics["metrics_path"] = str(metrics_path)
    return metrics


def load_model(path: Path | None = None) -> dict[str, Any]:
    model_path = path or MODEL_PATH
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Missing {model_path}. Run `python -m src.fare_model` first."
        )
    return joblib.load(model_path)


def predict_fare(
    *,
    trip_distance: float,
    pickup_time: int | None = None,
    pickup_hour: int | None = None,
    pickup_dayofweek: int = 3,
    pickup_month: int = 3,
    pickup_location_code: int = 4,
    dropoff_location_code: int = 2,
    estimated_duration_min: float | None = None,
    artifact: dict[str, Any] | None = None,
) -> FarePrediction:
    """Predict total fare and tip for a single trip.

    Prefer ``pickup_time`` (dataset bucket 0–5). ``pickup_hour`` (0–23) is
    accepted and mapped into those buckets for convenience.
    """
    duration = (
        float(estimated_duration_min)
        if estimated_duration_min is not None
        else float(3.0 + max(trip_distance, 0.1) * 2.4)
    )
    if pickup_time is None:
        hour = 12 if pickup_hour is None else int(pickup_hour)
        pickup_time = max(0, min(5, hour // 4))
    row = pd.DataFrame(
        [
            {
                "pickup_time": int(pickup_time),
                "pickup_dayofweek": int(pickup_dayofweek),
                "pickup_month": int(pickup_month),
                "pickup_location_code": int(pickup_location_code),
                "dropoff_location_code": int(dropoff_location_code),
                "trip_distance": float(trip_distance),
                "estimated_duration_min": duration,
            }
        ]
    )
    bundle = artifact or load_model()
    features = build_feature_matrix(row)
    pred = np.asarray(bundle["model"].predict(features))[0]
    total = max(float(pred[0]), 0.0)
    tip = max(float(pred[1]), 0.0)
    return FarePrediction(
        total_fare=round(total, 2),
        tip=round(tip, 2),
        estimated_duration_min=round(duration, 1),
    )


def main() -> None:
    metrics = train()
    best = metrics["best"]
    print(
        f"Best model: {best['name']} | "
        f"MAE total=${best['mae_total']:.2f} tip=${best['mae_tip']:.2f} | "
        f"R² total={best['r2_total']:.3f}"
    )
    print(f"Wrote {MODEL_PATH}")
    print(f"Wrote {METRICS_PATH}")


if __name__ == "__main__":
    main()
