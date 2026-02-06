"""
Data Loader - загрузка и предобработка минутных данных NQ
"""
import os
import pandas as pd
import numpy as np
from config import DATA_DIR, CSV_FILES, TRAIN_END_DATE, RESAMPLE_MINUTES


def load_single_csv(filepath: str) -> pd.DataFrame:
    """Load a single CSV file and parse dates."""
    df = pd.read_csv(filepath)

    # Parse the date column
    # Format: "31.12.2012 00:00:00.000 GMT-0600"
    df["datetime"] = pd.to_datetime(
        df["Local time"].str.replace(r" GMT[+-]\d{4}", "", regex=True),
        format="%d.%m.%Y %H:%M:%S.%f",
    )

    df = df[["datetime", "Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["datetime", "open", "high", "low", "close", "volume"]

    # Remove rows where price is 0 or NaN
    df = df[df["close"] > 0].copy()

    return df


def load_all_data() -> pd.DataFrame:
    """Load all CSV files and concatenate into a single DataFrame."""
    frames = []
    for filename in CSV_FILES:
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            print(f"Loading {filename}...")
            df = load_single_csv(filepath)
            frames.append(df)
            print(f"  -> {len(df)} rows, {df['datetime'].min()} to {df['datetime'].max()}")
        else:
            print(f"WARNING: {filename} not found, skipping.")

    if not frames:
        raise FileNotFoundError("No data files found!")

    # Concatenate and sort by datetime
    data = pd.concat(frames, ignore_index=True)
    data.sort_values("datetime", inplace=True)
    data.drop_duplicates(subset="datetime", keep="first", inplace=True)
    data.reset_index(drop=True, inplace=True)

    print(f"\nTotal: {len(data)} rows")
    print(f"Period: {data['datetime'].min()} to {data['datetime'].max()}")

    return data


def resample_data(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Resample to N-minute bars for faster training."""
    if minutes is None or minutes <= 1:
        return df

    df = df.set_index("datetime")
    resampled = df.resample(f"{minutes}min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    resampled.dropna(inplace=True)
    resampled.reset_index(inplace=True)

    print(f"Resampled to {minutes}-min bars: {len(resampled)} rows")
    return resampled


def filter_trading_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to keep only main trading hours (optional).
    NQ futures trade nearly 24h but main volume is 9:30-16:00 ET.
    """
    # Keep all hours for now - futures trade 23h/day
    # Uncomment below to filter:
    # df = df[(df['datetime'].dt.hour >= 9) & (df['datetime'].dt.hour < 16)]
    return df


def split_train_test(df: pd.DataFrame, split_date: str = None):
    """Split data into train and test sets."""
    if split_date is None:
        split_date = TRAIN_END_DATE

    split_dt = pd.to_datetime(split_date)
    train = df[df["datetime"] < split_dt].copy().reset_index(drop=True)
    test = df[df["datetime"] >= split_dt].copy().reset_index(drop=True)

    print(f"\nTrain: {len(train)} rows ({train['datetime'].min()} to {train['datetime'].max()})")
    print(f"Test:  {len(test)} rows ({test['datetime'].min()} to {test['datetime'].max()})")

    return train, test


def prepare_data():
    """Full data preparation pipeline."""
    # Load
    data = load_all_data()

    # Filter trading hours (optional)
    data = filter_trading_hours(data)

    # Resample if needed
    data = resample_data(data, RESAMPLE_MINUTES)

    # Split
    train_df, test_df = split_train_test(data)

    return train_df, test_df


if __name__ == "__main__":
    train, test = prepare_data()
    print(f"\nTrain shape: {train.shape}")
    print(f"Test shape:  {test.shape}")
    print(f"\nTrain sample:\n{train.head()}")
    print(f"\nTest sample:\n{test.head()}")
