import pandas as pd
import numpy as np
import os
import joblib

print("=" * 70)
print("GENERATING INTELLIGENT SIGNALS FOR JANUARY")
print("=" * 70)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "rites_jan_features.csv")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "rites_jan_signals_improved.csv")

# ========== LOAD DATA ==========
df = pd.read_csv(DATA_PATH, parse_dates=["date"])

# ========== LOAD MODELS ==========
print("\nLoading trained models...")

# Original trend ensemble (for comparison)
try:
    gb_trend = joblib.load(os.path.join(MODEL_DIR, "gb_model.pkl"))
    rf_trend = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
    lr_trend = joblib.load(os.path.join(MODEL_DIR, "lr_model.pkl"))
    scaler_trend = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    config_trend = joblib.load(os.path.join(MODEL_DIR, "ensemble_config.pkl"))
    print("✓ Trend models loaded")
except:
    print("⚠ Trend models not found - using MR models for both")
    gb_trend = gb_mr = joblib.load(os.path.join(MODEL_DIR, "gb_mr_model.pkl"))
    rf_trend = rf_mr = joblib.load(os.path.join(MODEL_DIR, "rf_mr_model.pkl"))
    lr_trend = lr_mr = joblib.load(os.path.join(MODEL_DIR, "lr_mr_model.pkl"))
    scaler_trend = scaler_mr = joblib.load(os.path.join(MODEL_DIR, "scaler_mr.pkl"))

# Mean-reversion ensemble
gb_mr = joblib.load(os.path.join(MODEL_DIR, "gb_mr_model.pkl"))
rf_mr = joblib.load(os.path.join(MODEL_DIR, "rf_mr_model.pkl"))
lr_mr = joblib.load(os.path.join(MODEL_DIR, "lr_mr_model.pkl"))
scaler_mr = joblib.load(os.path.join(MODEL_DIR, "scaler_mr.pkl"))
config_mr = joblib.load(os.path.join(MODEL_DIR, "ensemble_config_mr.pkl"))
print("✓ Mean-reversion models loaded")

# ========== MARKET REGIME DETECTION ==========
def detect_market_regime(df_input, idx):
    """Detect current market regime based on technical indicators"""
    
    if idx < 5:
        return 'neutral'
    
    recent = df_input.iloc[max(0, idx-5):idx+1]
    
    rsi_mean = recent['rsi_14'].mean()
    rsi_current = df_input['rsi_14'].iloc[idx]
    williams_r = df_input['williams_r'].iloc[idx] if 'williams_r' in df_input.columns else -50
    stochastic_k = df_input['stochastic_k'].iloc[idx] if 'stochastic_k' in df_input.columns else 50
    
    trend = (df_input['close'].iloc[idx] / df_input['close'].iloc[max(0, idx-5)]) - 1
    volatility = df_input['volatility_10'].iloc[idx]
    
    # Classification
    if rsi_mean > 75 or williams_r < -90 or stochastic_k > 95:
        return 'overbought_extreme'
    elif rsi_mean < 25 or williams_r > -10 or stochastic_k < 5:
        return 'oversold_extreme'
    elif rsi_mean > 65 and trend > 0:
        return 'overbought_uptrend'
    elif rsi_mean < 35 and trend < 0:
        return 'oversold_downtrend'
    elif trend > 0.01:
        return 'trending_up'
    elif trend < -0.01:
        return 'trending_down'
    else:
        return 'neutral'

# ========== GET TREND PREDICTIONS ==========
feature_cols_trend = config_trend['features']
X_trend = df[feature_cols_trend].fillna(0)
X_trend_scaled = scaler_trend.transform(X_trend)

gb_prob_trend = gb_trend.predict_proba(X_trend_scaled)[:, 1]
rf_prob_trend = rf_trend.predict_proba(X_trend_scaled)[:, 1]
lr_prob_trend = lr_trend.predict_proba(X_trend_scaled)[:, 1]

w_gb = config_trend['weights']['gb']
w_rf = config_trend['weights']['rf']
w_lr = config_trend['weights']['lr']

df['prob_up'] = w_gb * gb_prob_trend + w_rf * rf_prob_trend + w_lr * lr_prob_trend
df['agreement_trend'] = ((gb_prob_trend > 0.5).astype(int) + 
                          (rf_prob_trend > 0.5).astype(int) + 
                          (lr_prob_trend > 0.5).astype(int))

# ========== GET MEAN-REVERSION PREDICTIONS ==========
feature_cols_mr = config_mr['features']
X_mr = df[feature_cols_mr].fillna(0)
X_mr_scaled = scaler_mr.transform(X_mr)

gb_prob_mr = gb_mr.predict_proba(X_mr_scaled)[:, 1]
rf_prob_mr = rf_mr.predict_proba(X_mr_scaled)[:, 1]
lr_prob_mr = lr_mr.predict_proba(X_mr_scaled)[:, 1]

w_gb_mr = config_mr['weights']['gb']
w_rf_mr = config_mr['weights']['rf']
w_lr_mr = config_mr['weights']['lr']

df['prob_reversal'] = w_gb_mr * gb_prob_mr + w_rf_mr * rf_prob_mr + w_lr_mr * lr_prob_mr
df['agreement_mr'] = ((gb_prob_mr > 0.5).astype(int) + 
                       (rf_prob_mr > 0.5).astype(int) + 
                       (lr_prob_mr > 0.5).astype(int))

# ========== SIGNAL GENERATION ==========
def generate_intelligent_signal(row, market_regime):
    """Generate regime-aware signals"""
    
    prob_up = row['prob_up']
    prob_reversal = row['prob_reversal']
    agreement_trend = row['agreement_trend']
    agreement_mr = row['agreement_mr']
    rsi = row['rsi_14']
    bb_pos = row['bb_position']
    
    # OVERBOUGHT EXTREME
    if market_regime == 'overbought_extreme':
        if prob_reversal > 0.65 and agreement_mr >= 2 and rsi > 70:
            return "STRONG_SELL"
        elif prob_reversal > 0.55 and agreement_mr >= 1 and rsi > 65:
            return "SELL"
        else:
            return "SELL"
    
    # OVERSOLD EXTREME
    elif market_regime == 'oversold_extreme':
        if prob_reversal > 0.65 and agreement_mr >= 2 and rsi < 30:
            return "STRONG_BUY"
        elif prob_reversal > 0.55 and agreement_mr >= 1:
            return "BUY"
        else:
            return "BUY"
    
    # OVERBOUGHT UPTREND
    elif market_regime == 'overbought_uptrend':
        if prob_up > 0.75 and agreement_trend == 3 and rsi < 60 and prob_reversal < 0.4:
            return "BUY"
        elif prob_reversal > 0.60 and agreement_mr >= 2:
            return "SELL"
        else:
            return "HOLD"
    
    # OVERSOLD DOWNTREND
    elif market_regime == 'oversold_downtrend':
        if prob_reversal > 0.55:
            return "BUY"
        else:
            return "SELL"
    
    # TRENDING UP
    elif market_regime == 'trending_up':
        if prob_up > 0.68 and agreement_trend >= 2:
            return "STRONG_BUY"
        elif prob_up > 0.58 and agreement_trend >= 1:
            return "BUY"
        else:
            return "HOLD"
    
    # TRENDING DOWN
    elif market_regime == 'trending_down':
        if prob_reversal > 0.60:
            return "BUY"
        else:
            return "HOLD"
    
    # NEUTRAL
    else:
        if prob_up > 0.65 and agreement_trend >= 2 and rsi < 70 and bb_pos < 0.9:
            return "STRONG_BUY"
        elif prob_up > 0.58 and agreement_trend >= 2:
            return "BUY"
        elif prob_reversal > 0.60 and rsi > 65:
            return "SELL"
        else:
            return "HOLD"

# Apply signals
print("\nGenerating signals...")
df['market_regime'] = [detect_market_regime(df, i) for i in range(len(df))]
df['signal'] = [generate_intelligent_signal(df.iloc[i], df['market_regime'].iloc[i]) for i in range(len(df))]

# ========== ADAPTIVE POSITION SIZING ==========
def calculate_position_size(row, market_regime):
    """Adaptive position sizing based on regime"""
    
    prob_up = row['prob_up']
    volatility = row['volatility_10']
    signal = row['signal']
    
    # Base Kelly
    edge = prob_up - 0.5
    kelly = edge / (volatility + 0.01)
    kelly = min(kelly * 0.5, 0.25)
    kelly = max(kelly, 0.01)
    
    # Regime adjustment
    if market_regime == 'overbought_extreme':
        regime_mult = 0.0
    elif market_regime == 'overbought_uptrend':
        regime_mult = 0.5
    elif market_regime == 'oversold_downtrend':
        regime_mult = 0.3 if signal == "BUY" else 0.0
    elif market_regime == 'trending_down':
        regime_mult = 0.2 if signal == "BUY" else 0.0
    elif market_regime == 'oversold_extreme':
        regime_mult = 1.3
    else:
        regime_mult = 1.0
    
    sized = kelly * regime_mult
    
    # Signal strength
    if signal == "STRONG_BUY":
        sized *= 1.2
    elif signal == "BUY":
        sized *= 1.0
    elif signal == "HOLD":
        sized = 0.0
    
    return min(max(sized, 0.0), 0.30)

df['position_size'] = [calculate_position_size(df.iloc[i], df['market_regime'].iloc[i]) for i in range(len(df))]

# ========== ADAPTIVE RISK MANAGEMENT ==========
def calculate_stops(row, market_regime):
    """Adaptive stops based on regime"""
    
    atr_pct = row['atr_14'] / row['close']
    signal = row['signal']
    
    if market_regime == 'overbought_extreme':
        stop_loss = atr_pct * 0.8
        take_profit = atr_pct * 1.5
    elif market_regime == 'overbought_uptrend':
        stop_loss = atr_pct * 1.0
        take_profit = atr_pct * 2.5
    elif market_regime == 'oversold_extreme':
        stop_loss = atr_pct * 1.5
        take_profit = atr_pct * 4.0
    elif market_regime == 'oversold_downtrend':
        stop_loss = atr_pct * 1.0
        take_profit = atr_pct * 1.5
    elif market_regime == 'trending_down':
        stop_loss = atr_pct * 0.8
        take_profit = atr_pct * 1.2
    else:
        stop_loss = atr_pct * 2.0
        take_profit = atr_pct * 3.0
    
    return stop_loss, take_profit

stops = [calculate_stops(df.iloc[i], df['market_regime'].iloc[i]) for i in range(len(df))]
df['stop_loss_pct'] = [x for x in stops]
df['take_profit_pct'] = [x for x in stops]

# Max hold days
df['max_hold_days'] = df['market_regime'].apply(lambda x: {
    'overbought_extreme': 2,
    'oversold_extreme': 3,
    'overbought_uptrend': 3,
    'oversold_downtrend': 2,
    'trending_down': 2,
    'trending_up': 5,
    'neutral': 5
}.get(x, 5))

# ========== SAVE SIGNALS ==========
output_cols = [
    "date", "open", "high", "low", "close", "volume",
    "prob_up", "prob_reversal", "agreement_trend", "agreement_mr",
    "market_regime", "signal", "position_size",
    "stop_loss_pct", "take_profit_pct", "max_hold_days",
    "rsi_14", "macd", "bb_position", "adx", "volatility_10"
]

df[output_cols].to_csv(OUTPUT_PATH, index=False)

print(f"\n✓ Signals saved to: {OUTPUT_PATH}")
print("\nSignal distribution:")
print(df['signal'].value_counts())
print("\nMarket regime distribution:")
print(df['market_regime'].value_counts())

print("\n" + "=" * 70)
print("Signal Generation Complete!")
print("=" * 70)
print("\nKey metrics:")
for idx, row in df.iterrows():
    print(f"{row['date']}: Regime={row['market_regime']:20s} | Signal={row['signal']:10s} | "
          f"RSI={row['rsi_14']:5.1f} | Prob_reversal={row['prob_reversal']:.3f}")
