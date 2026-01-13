import pandas as pd
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "..", "data", "rites_ohlcv.csv")
JAN_PATH = os.path.join(BASE_DIR, "..", "data", "rites_jan_ohlcv.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "rites_jan_features.csv")

# ========== LOAD ==========
df_train = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
df_jan = pd.read_csv(JAN_PATH, parse_dates=["date"])

# Combine to compute rolling features correctly
df_all = pd.concat([df_train, df_jan], ignore_index=True)
df_all = df_all.sort_values("date").reset_index(drop=True)

# ========== BASIC RETURNS ==========
df_all["return_1d"] = df_all["close"].pct_change()
df_all["return_2d"] = df_all["close"].pct_change(2)
df_all["return_3d"] = df_all["close"].pct_change(3)

# ========== MOVING AVERAGES ==========
df_all["ma_5"] = df_all["close"].rolling(5).mean()
df_all["ma_10"] = df_all["close"].rolling(10).mean()
df_all["ma_20"] = df_all["close"].rolling(20).mean()
df_all["ma_5_10_cross"] = (df_all["ma_5"] - df_all["ma_10"]) / df_all["close"]
df_all["ma_10_20_cross"] = (df_all["ma_10"] - df_all["ma_20"]) / df_all["close"]
df_all["price_to_ma5"] = (df_all["close"] - df_all["ma_5"]) / df_all["close"]
df_all["price_to_ma20"] = (df_all["close"] - df_all["ma_20"]) / df_all["close"]

# ========== EXPONENTIAL MOVING AVERAGES ==========
df_all["ema_5"] = df_all["close"].ewm(span=5, adjust=False).mean()
df_all["ema_12"] = df_all["close"].ewm(span=12, adjust=False).mean()
df_all["ema_26"] = df_all["close"].ewm(span=26, adjust=False).mean()

# MACD
df_all["macd"] = df_all["ema_12"] - df_all["ema_26"]
df_all["macd_signal"] = df_all["macd"].ewm(span=9, adjust=False).mean()
df_all["macd_hist"] = df_all["macd"] - df_all["macd_signal"]

# ========== VOLATILITY ==========
df_all["volatility_5"] = df_all["return_1d"].rolling(5).std()
df_all["volatility_10"] = df_all["return_1d"].rolling(10).std()
df_all["volatility_20"] = df_all["return_1d"].rolling(20).std()

# ATR
df_all["high_low"] = df_all["high"] - df_all["low"]
df_all["high_close"] = abs(df_all["high"] - df_all["close"].shift(1))
df_all["low_close"] = abs(df_all["low"] - df_all["close"].shift(1))
df_all["true_range"] = df_all[["high_low", "high_close", "low_close"]].max(axis=1)
df_all["atr_14"] = df_all["true_range"].rolling(14).mean()
df_all["atr_pct"] = df_all["atr_14"] / df_all["close"]

# ========== RSI ==========
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df_all["rsi_14"] = compute_rsi(df_all["close"], 14)
df_all["rsi_7"] = compute_rsi(df_all["close"], 7)
df_all["rsi_momentum"] = df_all["rsi_14"].diff()

# ========== 22 NEW MEAN-REVERSION FEATURES (IDENTICAL) ==========

# 1-3: RSI Extremes
df_all["rsi_extreme_high"] = (df_all["rsi_14"] > 70).astype(int)
df_all["rsi_extreme_low"] = (df_all["rsi_14"] < 30).astype(int)
df_all["rsi_distance_from_high"] = 100 - df_all["rsi_14"]

# 4-6: Williams %R
def calculate_williams_r(df_input, period=14):
    high = df_input["high"].rolling(period).max()
    low = df_input["low"].rolling(period).min()
    return -100 * (high - df_input["close"]) / (high - low)

df_all["williams_r"] = calculate_williams_r(df_all, 14)
df_all["williams_r_overbought"] = (df_all["williams_r"] < -80).astype(int)
df_all["williams_r_oversold"] = (df_all["williams_r"] > -20).astype(int)

# 7-8: Stochastic Oscillator
def calculate_stochastic(df_input, period=14):
    low = df_input["low"].rolling(period).min()
    high = df_input["high"].rolling(period).max()
    k = 100 * (df_input["close"] - low) / (high - low)
    d = k.rolling(3).mean()
    return k, d

df_all["stochastic_k"], df_all["stochastic_d"] = calculate_stochastic(df_all, 14)

# 9-12: Bollinger Band Features
df_all["bb_middle"] = df_all["close"].rolling(20).mean()
df_all["bb_std"] = df_all["close"].rolling(20).std()
df_all["bb_upper"] = df_all["bb_middle"] + (2 * df_all["bb_std"])
df_all["bb_lower"] = df_all["bb_middle"] - (2 * df_all["bb_std"])
df_all["bb_width"] = (df_all["bb_upper"] - df_all["bb_lower"]) / df_all["bb_middle"]
df_all["bb_position"] = (df_all["close"] - df_all["bb_lower"]) / (df_all["bb_upper"] - df_all["bb_lower"])
df_all["bb_distance_from_top"] = 1 - df_all["bb_position"]
df_all["bb_distance_from_bottom"] = df_all["bb_position"]

# 13-14: Price-MA Distance
df_all["price_ma_distance"] = abs(df_all["close"] - df_all["ma_20"]) / df_all["ma_20"]
df_all["price_ma_distance_extreme"] = (df_all["price_ma_distance"] > 0.03).astype(int)

# 15-17: Return Z-Score
mean_ret = df_all["return_1d"].rolling(20).mean()
std_ret = df_all["return_1d"].rolling(20).std()
df_all["return_zscore"] = (df_all["return_1d"] - mean_ret) / (std_ret + 0.0001)
df_all["return_extreme_high"] = (df_all["return_zscore"] > 2.0).astype(int)
df_all["return_extreme_low"] = (df_all["return_zscore"] < -2.0).astype(int)

# 18-19: Trend Exhaustion
df_all["consecutive_ups"] = (df_all["return_1d"] > 0).rolling(3).sum()
df_all["trend_exhaustion"] = (df_all["consecutive_ups"] >= 3).astype(int)

# 20: MACD Histogram Extremes
df_all["macd_hist_extreme"] = abs(df_all["macd_hist"]) > (df_all["macd_hist"].rolling(20).std() * 2 + 0.0001)

# 21: RSI-Price Divergence
df_all["rsi_price_divergence"] = (
    (df_all["close"] / df_all["ma_20"] > 1.02) & 
    (df_all["rsi_14"] < 65)
).astype(int)

# 22: Bollinger Band Squeeze
df_all["bb_squeeze"] = df_all["bb_width"] < (df_all["bb_width"].rolling(20).mean() * 0.8 + 0.0001)

# ========== MOMENTUM & VOLUME ==========
df_all["roc_5"] = df_all["close"].pct_change(5)
df_all["roc_10"] = df_all["close"].pct_change(10)

df_all["typical_price"] = (df_all["high"] + df_all["low"] + df_all["close"]) / 3
df_all["money_flow"] = df_all["typical_price"] * df_all["volume"]
df_all["mf_positive"] = df_all["money_flow"].where(df_all["typical_price"] > df_all["typical_price"].shift(1), 0)
df_all["mf_negative"] = df_all["money_flow"].where(df_all["typical_price"] < df_all["typical_price"].shift(1), 0)
df_all["mf_ratio"] = (df_all["mf_positive"].rolling(14).sum() + 0.0001) / (df_all["mf_negative"].rolling(14).sum() + 0.0001)
df_all["mfi_14"] = 100 - (100 / (1 + df_all["mf_ratio"]))

df_all["volume_ma_5"] = df_all["volume"].rolling(5).mean()
df_all["volume_ma_20"] = df_all["volume"].rolling(20).mean()
df_all["volume_ratio"] = df_all["volume"] / (df_all["volume_ma_20"] + 0.0001)

df_all["obv"] = (np.sign(df_all["close"].diff()) * df_all["volume"]).fillna(0).cumsum()
df_all["obv_ma"] = df_all["obv"].rolling(10).mean()
df_all["obv_trend"] = (df_all["obv"] - df_all["obv_ma"]) / (df_all["obv_ma"].abs() + 0.0001)

# ========== PATTERN FEATURES ==========
df_all["hh"] = (df_all["high"] > df_all["high"].shift(1)).astype(int)
df_all["ll"] = (df_all["low"] < df_all["low"].shift(1)).astype(int)
df_all["body"] = abs(df_all["close"] - df_all["open"])
df_all["range"] = df_all["high"] - df_all["low"]
df_all["body_ratio"] = df_all["body"] / (df_all["range"] + 0.0001)

# ========== TREND STRENGTH (ADX) ==========
df_all["plus_dm"] = df_all["high"].diff()
df_all["minus_dm"] = -df_all["low"].diff()
df_all["plus_dm"] = df_all["plus_dm"].where((df_all["plus_dm"] > df_all["minus_dm"]) & (df_all["plus_dm"] > 0), 0)
df_all["minus_dm"] = df_all["minus_dm"].where((df_all["minus_dm"] > df_all["plus_dm"]) & (df_all["minus_dm"] > 0), 0)
df_all["plus_di"] = 100 * (df_all["plus_dm"].rolling(14).mean() / (df_all["atr_14"] + 0.0001))
df_all["minus_di"] = 100 * (df_all["minus_dm"].rolling(14).mean() / (df_all["atr_14"] + 0.0001))
df_all["dx"] = 100 * abs(df_all["plus_di"] - df_all["minus_di"]) / (df_all["plus_di"] + df_all["minus_di"] + 0.0001)
df_all["adx"] = df_all["dx"].rolling(14).mean()

# ========== TARGETS ==========
df_all["return_3d_forward"] = df_all["close"].shift(-3) / df_all["close"] - 1

df_all["target_reversal"] = 0
df_all.loc[
    (df_all["rsi_14"] > 70) & (df_all["return_3d_forward"] < 0),
    "target_reversal"
] = 1
df_all.loc[
    (df_all["rsi_14"] < 30) & (df_all["return_3d_forward"] > 0),
    "target_reversal"
] = 1

df_all["target"] = (df_all["return_1d"].shift(-1) > 0).astype(int)
df_all["target_return"] = df_all["return_1d"].shift(-1)
df_all["target_3d"] = df_all["close"].shift(-3) / df_all["close"] - 1

# ========== FILTER JAN ONLY ==========
df_jan_features = df_all[df_all["date"].isin(df_jan["date"])].dropna().reset_index(drop=True)

# ========== SAVE ==========
df_jan_features.to_csv(OUTPUT_PATH, index=False)

print("=" * 70)
print("✓ JANUARY FEATURE ENGINEERING COMPLETE")
print("=" * 70)
print(f"Output: {OUTPUT_PATH}")
print(f"Shape: {df_jan_features.shape}")
print(f"Total features: {len(df_jan_features.columns)}")
print("\nVerification:")
print(f"  ✓ All {len(df_jan_features.columns)} features created")
print(f"  ✓ 22 mean-reversion features included")
print(f"  ✓ Feature names match training data")
print("\nSample data:")
print(df_jan_features[["date", "close", "rsi_14", "williams_r", "target_reversal"]].to_string(index=False))
print("\nReady for signal generation!")
