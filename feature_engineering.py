import pandas as pd
import numpy as np
import os

# ========== PATH HANDLING ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "rites_ohlcv.csv")

# ========== LOAD DATA ==========
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# ========== BASIC RETURNS ==========
df["return_1d"] = df["close"].pct_change()
df["return_2d"] = df["close"].pct_change(2)
df["return_3d"] = df["close"].pct_change(3)

# ========== MOVING AVERAGES ==========
df["ma_5"] = df["close"].rolling(window=5).mean()
df["ma_10"] = df["close"].rolling(window=10).mean()
df["ma_20"] = df["close"].rolling(window=20).mean()
df["ma_5_10_cross"] = (df["ma_5"] - df["ma_10"]) / df["close"]
df["ma_10_20_cross"] = (df["ma_10"] - df["ma_20"]) / df["close"]
df["price_to_ma5"] = (df["close"] - df["ma_5"]) / df["close"]
df["price_to_ma20"] = (df["close"] - df["ma_20"]) / df["close"]

# ========== EXPONENTIAL MOVING AVERAGES ==========
df["ema_5"] = df["close"].ewm(span=5, adjust=False).mean()
df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()

# MACD
df["macd"] = df["ema_12"] - df["ema_26"]
df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
df["macd_hist"] = df["macd"] - df["macd_signal"]

# ========== VOLATILITY ==========
df["volatility_5"] = df["return_1d"].rolling(window=5).std()
df["volatility_10"] = df["return_1d"].rolling(window=10).std()
df["volatility_20"] = df["return_1d"].rolling(window=20).std()

# ATR
df["high_low"] = df["high"] - df["low"]
df["high_close"] = abs(df["high"] - df["close"].shift(1))
df["low_close"] = abs(df["low"] - df["close"].shift(1))
df["true_range"] = df[["high_low", "high_close", "low_close"]].max(axis=1)
df["atr_14"] = df["true_range"].rolling(window=14).mean()
df["atr_pct"] = df["atr_14"] / df["close"]

# ========== RSI ==========
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df["rsi_14"] = compute_rsi(df["close"], 14)
df["rsi_7"] = compute_rsi(df["close"], 7)
df["rsi_momentum"] = df["rsi_14"].diff()

# ========== 22 NEW MEAN-REVERSION FEATURES ==========

# 1-3: RSI Extremes
df["rsi_extreme_high"] = (df["rsi_14"] > 70).astype(int)
df["rsi_extreme_low"] = (df["rsi_14"] < 30).astype(int)
df["rsi_distance_from_high"] = 100 - df["rsi_14"]

# 4-6: Williams %R
def calculate_williams_r(df_input, period=14):
    high = df_input["high"].rolling(period).max()
    low = df_input["low"].rolling(period).min()
    return -100 * (high - df_input["close"]) / (high - low)

df["williams_r"] = calculate_williams_r(df, 14)
df["williams_r_overbought"] = (df["williams_r"] < -80).astype(int)
df["williams_r_oversold"] = (df["williams_r"] > -20).astype(int)

# 7-8: Stochastic Oscillator
def calculate_stochastic(df_input, period=14):
    low = df_input["low"].rolling(period).min()
    high = df_input["high"].rolling(period).max()
    k = 100 * (df_input["close"] - low) / (high - low)
    d = k.rolling(3).mean()
    return k, d

df["stochastic_k"], df["stochastic_d"] = calculate_stochastic(df, 14)

# 9-12: Bollinger Band Features (updated positioning)
df["bb_middle"] = df["close"].rolling(window=20).mean()
df["bb_std"] = df["close"].rolling(window=20).std()
df["bb_upper"] = df["bb_middle"] + (2 * df["bb_std"])
df["bb_lower"] = df["bb_middle"] - (2 * df["bb_std"])
df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
df["bb_distance_from_top"] = 1 - df["bb_position"]
df["bb_distance_from_bottom"] = df["bb_position"]

# 13-14: Price-MA Distance
df["price_ma_distance"] = abs(df["close"] - df["ma_20"]) / df["ma_20"]
df["price_ma_distance_extreme"] = (df["price_ma_distance"] > 0.03).astype(int)

# 15-17: Return Z-Score
mean_ret = df["return_1d"].rolling(20).mean()
std_ret = df["return_1d"].rolling(20).std()
df["return_zscore"] = (df["return_1d"] - mean_ret) / (std_ret + 0.0001)
df["return_extreme_high"] = (df["return_zscore"] > 2.0).astype(int)
df["return_extreme_low"] = (df["return_zscore"] < -2.0).astype(int)

# 18-19: Trend Exhaustion
df["consecutive_ups"] = (df["return_1d"] > 0).rolling(3).sum()
df["trend_exhaustion"] = (df["consecutive_ups"] >= 3).astype(int)

# 20: MACD Histogram Extremes
df["macd_hist_extreme"] = abs(df["macd_hist"]) > (df["macd_hist"].rolling(20).std() * 2 + 0.0001)

# 21: RSI-Price Divergence
df["rsi_price_divergence"] = (
    (df["close"] / df["ma_20"] > 1.02) & 
    (df["rsi_14"] < 65)
).astype(int)

# 22: Bollinger Band Squeeze
df["bb_squeeze"] = df["bb_width"] < (df["bb_width"].rolling(20).mean() * 0.8 + 0.0001)

# ========== MOMENTUM & VOLUME ==========
df["roc_5"] = df["close"].pct_change(5)
df["roc_10"] = df["close"].pct_change(10)

df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
df["money_flow"] = df["typical_price"] * df["volume"]
df["mf_positive"] = df["money_flow"].where(df["typical_price"] > df["typical_price"].shift(1), 0)
df["mf_negative"] = df["money_flow"].where(df["typical_price"] < df["typical_price"].shift(1), 0)
df["mf_ratio"] = (df["mf_positive"].rolling(14).sum() + 0.0001) / (df["mf_negative"].rolling(14).sum() + 0.0001)
df["mfi_14"] = 100 - (100 / (1 + df["mf_ratio"]))

df["volume_ma_5"] = df["volume"].rolling(window=5).mean()
df["volume_ma_20"] = df["volume"].rolling(window=20).mean()
df["volume_ratio"] = df["volume"] / (df["volume_ma_20"] + 0.0001)

df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
df["obv_ma"] = df["obv"].rolling(window=10).mean()
df["obv_trend"] = (df["obv"] - df["obv_ma"]) / (df["obv_ma"].abs() + 0.0001)

# ========== PATTERN FEATURES ==========
df["hh"] = (df["high"] > df["high"].shift(1)).astype(int)
df["ll"] = (df["low"] < df["low"].shift(1)).astype(int)
df["body"] = abs(df["close"] - df["open"])
df["range"] = df["high"] - df["low"]
df["body_ratio"] = df["body"] / (df["range"] + 0.0001)
df["upper_wick"] = df["high"] - df[["close", "open"]].max(axis=1)
df["lower_wick"] = df[["close", "open"]].min(axis=1) - df["low"]

# ========== TREND STRENGTH (ADX) ==========
df["plus_dm"] = df["high"].diff()
df["minus_dm"] = -df["low"].diff()
df["plus_dm"] = df["plus_dm"].where((df["plus_dm"] > df["minus_dm"]) & (df["plus_dm"] > 0), 0)
df["minus_dm"] = df["minus_dm"].where((df["minus_dm"] > df["plus_dm"]) & (df["minus_dm"] > 0), 0)
df["plus_di"] = 100 * (df["plus_dm"].rolling(14).mean() / (df["atr_14"] + 0.0001))
df["minus_di"] = 100 * (df["minus_dm"].rolling(14).mean() / (df["atr_14"] + 0.0001))
df["dx"] = 100 * abs(df["plus_di"] - df["minus_di"]) / (df["plus_di"] + df["minus_di"] + 0.0001)
df["adx"] = df["dx"].rolling(14).mean()

# ========== TARGETS ==========
# Forward-looking return (3 days)
df["return_3d_forward"] = df["close"].shift(-3) / df["close"] - 1

# Target: 1 if reversal happens from extreme
df["target_reversal"] = 0
df.loc[
    (df["rsi_14"] > 70) & (df["return_3d_forward"] < 0),
    "target_reversal"
] = 1
df.loc[
    (df["rsi_14"] < 30) & (df["return_3d_forward"] > 0),
    "target_reversal"
] = 1

# Original target (for trend model)
df["target"] = (df["return_1d"].shift(-1) > 0).astype(int)
df["target_return"] = df["return_1d"].shift(-1)
df["target_3d"] = df["close"].shift(-3) / df["close"] - 1

# ========== CLEAN & SAVE ==========
df_features = df.dropna().reset_index(drop=True)

OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "rites_features.csv")
df_features.to_csv(OUTPUT_PATH, index=False)

print("=" * 70)
print("✓ FEATURE ENGINEERING COMPLETE (Nov-Dec Training Data)")
print("=" * 70)
print(f"Output: {OUTPUT_PATH}")
print(f"Shape: {df_features.shape}")
print(f"Total features: {len(df_features.columns)}")
print("\nFeature breakdown:")
print(f"  • OHLCV: 5")
print(f"  • Basic: 3 (returns)")
print(f"  • Moving Averages: 10")
print(f"  • MACD: 3")
print(f"  • Volatility & ATR: 6")
print(f"  • RSI: 3")
print(f"  • Mean-Reversion (NEW): 22")
print(f"  • Momentum & Volume: 15")
print(f"  • Pattern: 7")
print(f"  • ADX: 5")
print(f"  • Targets: 4")
print(f"  ────────────────")
print(f"  TOTAL: {len(df_features.columns)} features")
print("\nNew mean-reversion features:")
print("  ✓ RSI extremes, Williams %R, Stochastic")
print("  ✓ BB distance, Price-MA distance")
print("  ✓ Return Z-score, Trend exhaustion")
print("  ✓ MACD extremes, RSI divergence, BB squeeze")
print("\nReady for model training!")
