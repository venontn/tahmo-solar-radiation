"""Feature engineering for TAHMO solar radiation prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd

STATION_COL = "station"
TIMESTAMP_COL = "timestamp"
TARGET_COL = "radiation (W/m2)"
ID_COL = "ID"

WEATHER_COLS = [
    "precipitation (mm)",
    "relativehumidity (-)",
    "temperature (degrees Celsius)",
]


def solar_elevation_deg(lat: np.ndarray, lon: np.ndarray, ts: pd.Series) -> np.ndarray:
    """Approximate solar elevation angle (degrees) for each row."""
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    doy = ts.dt.dayofyear.to_numpy(dtype=float)
    hour = ts.dt.hour.to_numpy(dtype=float) + ts.dt.minute.to_numpy(dtype=float) / 60.0

    # Solar declination (Cooper, 1969)
    decl = np.radians(23.45 * np.sin(np.radians(360 / 365 * (doy + 284))))

    # Hour angle: solar noon at 12:00 local (approximation; no timezone correction in data)
    ha = np.radians(15 * (hour - 12))

    sin_elev = np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.cos(ha)
    sin_elev = np.clip(sin_elev, -1, 1)
    return np.degrees(np.arcsin(sin_elev))


def add_time_features(df: pd.DataFrame, timestamp_column: str = TIMESTAMP_COL) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df[timestamp_column], errors="coerce")

    df["year"] = ts.dt.year
    df["month"] = ts.dt.month
    df["day"] = ts.dt.day
    df["hour"] = ts.dt.hour
    df["minute"] = ts.dt.minute
    df["day_of_week"] = ts.dt.dayofweek
    df["day_of_year"] = ts.dt.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(np.int8)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)

    df["solar_elevation"] = solar_elevation_deg(
        df["latitude"].to_numpy(dtype=float),
        df["longitude"].to_numpy(dtype=float),
        ts,
    )
    df["solar_elevation_pos"] = np.maximum(df["solar_elevation"], 0)
    df["is_daylight"] = (df["solar_elevation"] > 0).astype(np.int8)

    # Simple clear-sky proxy (scales with sun height)
    df["clearsky_proxy"] = df["solar_elevation_pos"] ** 1.2

    return df


def _time_slot(df: pd.DataFrame) -> pd.Series:
    return (
        df["day"].astype(str)
        + "_"
        + df["hour"].astype(str)
        + "_"
        + df["minute"].astype(str)
    )


def _adjacent_odd_months(month: int) -> tuple[int, int]:
    """For an even test month, return neighboring odd months in the same year."""
    if month == 2:
        return 1, 3
    if month == 4:
        return 3, 5
    if month == 6:
        return 5, 7
    if month == 8:
        return 7, 9
    if month == 10:
        return 9, 11
    if month == 12:
        return 11, 1
    return month - 1, month + 1


def build_analog_lookup(train_fe: pd.DataFrame) -> pd.DataFrame:
    """
    Per station and calendar (day, hour, minute), store mean radiation
    for each odd month present in training.
    """
    slot = _time_slot(train_fe)
    lookup = (
        train_fe.assign(_slot=slot)
        .groupby([STATION_COL, "month", "_slot"], as_index=False)[TARGET_COL]
        .median()
        .rename(columns={TARGET_COL: "radiation_analog"})
    )
    return lookup


def add_analog_features_test(df: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    """For even-month test rows, average radiation from adjacent odd months at same clock slot."""
    df = df.copy()
    df["_slot"] = _time_slot(df)

    parts = []
    for month in sorted(df["month"].unique()):
        block = df[df["month"] == month].copy()
        m_lo, m_hi = _adjacent_odd_months(int(month))

        lo = lookup[lookup["month"] == m_lo][[STATION_COL, "_slot", "radiation_analog"]].rename(
            columns={"radiation_analog": "rad_lo"}
        )
        hi = lookup[lookup["month"] == m_hi][[STATION_COL, "_slot", "radiation_analog"]].rename(
            columns={"radiation_analog": "rad_hi"}
        )

        block = block.merge(lo, on=[STATION_COL, "_slot"], how="left")
        block = block.merge(hi, on=[STATION_COL, "_slot"], how="left")
        block["radiation_analog"] = block[["rad_lo", "rad_hi"]].mean(axis=1, skipna=True)
        parts.append(block.drop(columns=["rad_lo", "rad_hi"], errors="ignore"))

    out = pd.concat(parts, axis=0, ignore_index=True)
    return out.drop(columns=["_slot"], errors="ignore")


def add_analog_features_train(
    df: pd.DataFrame,
    lookup: pd.DataFrame,
    exclude_month: int | None = None,
) -> pd.DataFrame:
    """
    For odd-month training rows, average lookup across other odd months
    (optionally excluding one month for CV).
    """
    df = df.copy()
    df["_slot"] = _time_slot(df)
    lk = lookup.copy()
    if exclude_month is not None:
        lk = lk[lk["month"] != exclude_month]

    agg = (
        lk.groupby([STATION_COL, "_slot"], as_index=False)["radiation_analog"]
        .mean()
        .rename(columns={"radiation_analog": "radiation_analog"})
    )
    out = df.merge(agg, on=[STATION_COL, "_slot"], how="left")
    return out.drop(columns=["_slot"], errors="ignore")


def get_feature_columns(train_fe: pd.DataFrame) -> list[str]:
    exclude = {
        ID_COL,
        STATION_COL,
        "station_name",
        "country",
        TIMESTAMP_COL,
        TARGET_COL,
        "radiation_analog",  # used for blend only, not as ML input
    }
    return [
        c
        for c in train_fe.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]
