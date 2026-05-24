"""
Train per-station LightGBM models and blend with odd-month analog climatology.

Usage (from project root, after placing Zindi CSVs in data/):
  python src/train_predict.py
  python src/train_predict.py --data-dir data --blend 0.55
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from features import (
    ID_COL,
    STATION_COL,
    TARGET_COL,
    TIMESTAMP_COL,
    add_analog_features_test,
    add_analog_features_train,
    add_time_features,
    build_analog_lookup,
    get_feature_columns,
)

RADIATION_MAX = 1400.0


def mbe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred - y_true))


def abs_mbe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return abs(mbe(y_true, y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def postprocess(preds: np.ndarray, solar_elevation: np.ndarray) -> np.ndarray:
    preds = np.clip(preds, 0, RADIATION_MAX)
    preds = np.where(solar_elevation <= 0, 0.0, preds)
    return preds


def make_lgbm() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def cv_odd_months(train_fe: pd.DataFrame, feature_cols: list[str]) -> None:
    """Leave-one-odd-month-out CV to mimic the even-month test setup."""
    months = sorted(train_fe["month"].unique())
    print("\n--- Leave-one-odd-month-out CV ---")
    for holdout in months:
        tr = train_fe[train_fe["month"] != holdout]
        va = train_fe[train_fe["month"] == holdout]
        if len(va) == 0:
            continue

        lookup = build_analog_lookup(tr)
        va_base = va.drop(columns=["radiation_analog"], errors="ignore")
        tr_base = tr.drop(columns=["radiation_analog"], errors="ignore")
        va_a = add_analog_features_train(va_base, lookup, exclude_month=holdout)
        tr_a = add_analog_features_train(tr_base, lookup, exclude_month=holdout)

        global_m = make_lgbm()
        global_m.fit(tr_a[feature_cols], tr_a[TARGET_COL])

        preds = global_m.predict(va_a[feature_cols])
        if "radiation_analog" in va_a.columns:
            analog = va_a["radiation_analog"].fillna(0).to_numpy()
            preds = 0.55 * preds + 0.45 * analog

        preds = postprocess(preds, va_a["solar_elevation"].to_numpy())
        yt = va[TARGET_COL].to_numpy()
        print(
            f"  holdout month {holdout:02d}: "
            f"|MBE|={abs_mbe(yt, preds):.2f}  RMSE={rmse(yt, preds):.2f}  "
            f"MAE={mean_absolute_error(yt, preds):.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("output/submission.csv"))
    parser.add_argument("--blend", type=float, default=0.55, help="Weight on ML vs analog (ML weight)")
    parser.add_argument("--cv", action="store_true", help="Run leave-one-odd-month-out CV only")
    args = parser.parse_args()

    data_dir = args.data_dir
    train_path = data_dir / "Train.csv"
    test_path = data_dir / "Test.csv"
    sample_path = data_dir / "SampleSubmission.csv"

    for p in (train_path, test_path, sample_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Download competition files from Zindi into {data_dir.resolve()}"
            )

    print("Loading data...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_sub = pd.read_csv(sample_path)
    print(f"  train {train.shape}, test {test.shape}")

    train_fe = add_time_features(train, TIMESTAMP_COL)
    test_fe = add_time_features(test, TIMESTAMP_COL)

    lookup = build_analog_lookup(train_fe)
    train_fe = add_analog_features_train(train_fe, lookup)
    test_fe = add_analog_features_test(test_fe, lookup)

    feature_cols = get_feature_columns(train_fe)
    print(f"  features ({len(feature_cols)}): {feature_cols}")

    if args.cv:
        cv_odd_months(train_fe, feature_cols)
        return

    # Global fallback
    global_model = make_lgbm()
    global_model.fit(train_fe[feature_cols], train_fe[TARGET_COL])

    predictions: list[pd.DataFrame] = []
    stations = test_fe[STATION_COL].dropna().unique()
    print(f"Training per-station models for {len(stations)} stations...")

    for station in stations:
        tr_s = train_fe[train_fe[STATION_COL] == station]
        te_s = test_fe[test_fe[STATION_COL] == station]
        if len(te_s) == 0:
            continue

        model = global_model if len(tr_s) < 100 else make_lgbm()
        if len(tr_s) >= 100:
            model.fit(tr_s[feature_cols], tr_s[TARGET_COL])

        preds = model.predict(te_s[feature_cols])
        analog = te_s["radiation_analog"].fillna(0).to_numpy()
        preds = args.blend * preds + (1 - args.blend) * analog
        preds = postprocess(preds, te_s["solar_elevation"].to_numpy())

        predictions.append(pd.DataFrame({ID_COL: te_s[ID_COL].values, "prediction": preds}))

    pred_df = pd.concat(predictions, ignore_index=True)
    print(f"  predictions: {len(pred_df)}")

    submission = sample_sub[[ID_COL]].merge(pred_df, on=ID_COL, how="left")
    missing = submission["prediction"].isna().sum()
    if missing:
        print(f"  warning: {missing} IDs without predictions — filling with global model")
        test_missing = test_fe[~test_fe[ID_COL].isin(pred_df[ID_COL])]
        if len(test_missing):
            fill = global_model.predict(test_missing[feature_cols])
            fill = postprocess(fill, test_missing["solar_elevation"].to_numpy())
            fill_df = pd.DataFrame({ID_COL: test_missing[ID_COL].values, "prediction": fill})
            pred_df = pd.concat([pred_df, fill_df], ignore_index=True)
            submission = sample_sub[[ID_COL]].merge(pred_df, on=ID_COL, how="left")

    submission["TargetMBE"] = submission["prediction"]
    submission["TargetRMSE"] = submission["prediction"]
    submission = submission[[ID_COL, "TargetMBE", "TargetRMSE"]]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(submission)} rows)")
    print(submission.head())


if __name__ == "__main__":
    main()
