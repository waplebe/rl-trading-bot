"""
Technical Indicators - вычисление признаков для агента
"""
import numpy as np
import pandas as pd


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the dataframe.
    These become the features that the RL agent "sees".
    """
    df = df.copy()

    # ==========================================
    # MOVING AVERAGES
    # ==========================================
    df["sma_10"] = df["close"].rolling(window=10).mean()
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["sma_50"] = df["close"].rolling(window=50).mean()
    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()

    # Price relative to MAs (normalized)
    df["price_sma10_ratio"] = df["close"] / df["sma_10"] - 1
    df["price_sma20_ratio"] = df["close"] / df["sma_20"] - 1
    df["price_sma50_ratio"] = df["close"] / df["sma_50"] - 1
    df["sma10_sma20_ratio"] = df["sma_10"] / df["sma_20"] - 1

    # ==========================================
    # RSI (Relative Strength Index)
    # ==========================================
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi_normalized"] = df["rsi"] / 100.0  # Normalize to 0-1

    # ==========================================
    # MACD
    # ==========================================
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Normalize MACD by price
    df["macd_normalized"] = df["macd"] / df["close"]
    df["macd_hist_normalized"] = df["macd_hist"] / df["close"]

    # ==========================================
    # BOLLINGER BANDS
    # ==========================================
    bb_period = 20
    bb_std = df["close"].rolling(window=bb_period).std()
    bb_mid = df["close"].rolling(window=bb_period).mean()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_mid
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (
        df["bb_upper"] - df["bb_lower"] + 1e-10
    )

    # ==========================================
    # ATR (Average True Range) - Volatility
    # ==========================================
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = true_range.rolling(window=14).mean()
    df["atr_normalized"] = df["atr"] / df["close"]

    # ==========================================
    # RETURNS & MOMENTUM
    # ==========================================
    df["return_1"] = df["close"].pct_change(1)
    df["return_5"] = df["close"].pct_change(5)
    df["return_10"] = df["close"].pct_change(10)
    df["return_20"] = df["close"].pct_change(20)

    # Volatility of returns
    df["volatility_10"] = df["return_1"].rolling(window=10).std()
    df["volatility_20"] = df["return_1"].rolling(window=20).std()

    # ==========================================
    # VOLUME FEATURES
    # ==========================================
    df["volume_sma"] = df["volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_sma"] + 1e-10)

    # ==========================================
    # CANDLE PATTERNS (simple)
    # ==========================================
    df["body"] = (df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-10)
    df["upper_shadow"] = (df["high"] - df[["close", "open"]].max(axis=1)) / (
        df["high"] - df["low"] + 1e-10
    )
    df["lower_shadow"] = (df[["close", "open"]].min(axis=1) - df["low"]) / (
        df["high"] - df["low"] + 1e-10
    )

    # ==========================================
    # STOCHASTIC
    # ==========================================
    stoch_period = 14
    low_min = df["low"].rolling(window=stoch_period).min()
    high_max = df["high"].rolling(window=stoch_period).max()
    df["stoch_k"] = (df["close"] - low_min) / (high_max - low_min + 1e-10)
    df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()

    # ==========================================
    # CLEAN UP
    # ==========================================
    # Drop rows with NaN (from rolling calculations)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def get_feature_columns() -> list:
    """Return list of feature columns that the agent will use as observation."""
    return [
        # Price relative to MAs
        "price_sma10_ratio",
        "price_sma20_ratio",
        "price_sma50_ratio",
        "sma10_sma20_ratio",
        # RSI
        "rsi_normalized",
        # MACD
        "macd_normalized",
        "macd_hist_normalized",
        # Bollinger Bands
        "bb_width",
        "bb_position",
        # Volatility
        "atr_normalized",
        "volatility_10",
        "volatility_20",
        # Returns / Momentum
        "return_1",
        "return_5",
        "return_10",
        "return_20",
        # Volume
        "volume_ratio",
        # Candle patterns
        "body",
        "upper_shadow",
        "lower_shadow",
        # Stochastic
        "stoch_k",
        "stoch_d",
    ]
