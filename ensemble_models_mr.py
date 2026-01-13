import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("TRAINING MEAN-REVERSION ENSEMBLE")
print("=" * 70)

# ========== LOAD DATA ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "rites_features.csv")

df = pd.read_csv(DATA_PATH)

# ========== MEAN-REVERSION FEATURES ==========
mr_feature_cols = [
    # Extremes detection
    "rsi_14", "rsi_7", "williams_r", "stochastic_k", "stochastic_d",
    "rsi_distance_from_high", "rsi_extreme_high", "rsi_extreme_low",
    "williams_r_overbought", "williams_r_oversold",
    
    # Bollinger Bands
    "bb_position", "bb_width", "bb_distance_from_top", "bb_distance_from_bottom",
    
    # Price deviation
    "price_ma_distance", "price_ma_distance_extreme",
    "price_to_ma5", "price_to_ma20",
    
    # Return extremes
    "return_zscore", "return_extreme_high", "return_extreme_low",
    "return_1d", "return_2d", "return_3d",
    
    # Momentum extremes
    "consecutive_ups", "trend_exhaustion",
    "macd_hist", "macd_hist_extreme", "rsi_momentum",
    
    # Divergence
    "rsi_price_divergence", "bb_squeeze",
    
    # Volatility
    "volatility_5", "volatility_10", "volatility_20", "atr_pct",
    
    # Volume confirmation
    "volume_ratio", "obv_trend", "mfi_14",
    
    # Rate of change
    "roc_5", "roc_10",
]

X_mr = df[mr_feature_cols].copy()
y_mr = df["target_reversal"].copy()

# ========== REMOVE NaN ==========
valid_idx = ~(X_mr.isna().any(axis=1) | y_mr.isna())
X_mr = X_mr[valid_idx].reset_index(drop=True)
y_mr = y_mr[valid_idx].reset_index(drop=True)

print(f"\nTraining data shape: {X_mr.shape}")
print(f"Target distribution: {y_mr.value_counts().to_dict()}")

# ========== TRAIN-TEST SPLIT ==========
split_idx = int(len(X_mr) * 0.75)
X_train = X_mr.iloc[:split_idx]
X_test = X_mr.iloc[split_idx:]
y_train = y_mr.iloc[:split_idx]
y_test = y_mr.iloc[split_idx:]

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# ========== FEATURE SCALING ==========
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========== GRADIENT BOOSTING ==========
print("\n1. Training Gradient Boosting (Mean-Reversion)...")
gb_mr = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=3,
    min_samples_split=15,
    min_samples_leaf=8,
    subsample=0.8,
    random_state=42
)
gb_mr.fit(X_train_scaled, y_train)
gb_mr_prob = gb_mr.predict_proba(X_test_scaled)[:, 1]
gb_mr_preds = (gb_mr_prob > 0.5).astype(int)

gb_mr_acc = accuracy_score(y_test, gb_mr_preds)
gb_mr_auc = roc_auc_score(y_test, gb_mr_prob)
gb_mr_prec = precision_score(y_test, gb_mr_preds, zero_division=0)
gb_mr_rec = recall_score(y_test, gb_mr_preds, zero_division=0)

print(f"   Accuracy: {gb_mr_acc:.4f}")
print(f"   AUC: {gb_mr_auc:.4f}")
print(f"   Precision: {gb_mr_prec:.4f}, Recall: {gb_mr_rec:.4f}")

# ========== RANDOM FOREST ==========
print("2. Training Random Forest (Mean-Reversion)...")
rf_mr = RandomForestClassifier(
    n_estimators=400,
    max_depth=7,
    min_samples_split=12,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf_mr.fit(X_train_scaled, y_train)
rf_mr_prob = rf_mr.predict_proba(X_test_scaled)[:, 1]
rf_mr_preds = (rf_mr_prob > 0.5).astype(int)

rf_mr_acc = accuracy_score(y_test, rf_mr_preds)
rf_mr_auc = roc_auc_score(y_test, rf_mr_prob)
rf_mr_prec = precision_score(y_test, rf_mr_preds, zero_division=0)
rf_mr_rec = recall_score(y_test, rf_mr_preds, zero_division=0)

print(f"   Accuracy: {rf_mr_acc:.4f}")
print(f"   AUC: {rf_mr_auc:.4f}")
print(f"   Precision: {rf_mr_prec:.4f}, Recall: {rf_mr_rec:.4f}")

# ========== LOGISTIC REGRESSION ==========
print("3. Training Logistic Regression (Mean-Reversion)...")
lr_mr = LogisticRegression(max_iter=2000, C=0.01, random_state=42)
lr_mr.fit(X_train_scaled, y_train)
lr_mr_prob = lr_mr.predict_proba(X_test_scaled)[:, 1]
lr_mr_preds = (lr_mr_prob > 0.5).astype(int)

lr_mr_acc = accuracy_score(y_test, lr_mr_preds)
lr_mr_auc = roc_auc_score(y_test, lr_mr_prob)
lr_mr_prec = precision_score(y_test, lr_mr_preds, zero_division=0)
lr_mr_rec = recall_score(y_test, lr_mr_preds, zero_division=0)

print(f"   Accuracy: {lr_mr_acc:.4f}")
print(f"   AUC: {lr_mr_auc:.4f}")
print(f"   Precision: {lr_mr_prec:.4f}, Recall: {lr_mr_rec:.4f}")

# ========== ENSEMBLE COMBINATION ==========
print("\n4. Creating Ensemble Combination...")
total_auc = gb_mr_auc + rf_mr_auc + lr_mr_auc
w_gb_mr = gb_mr_auc / total_auc
w_rf_mr = rf_mr_auc / total_auc
w_lr_mr = lr_mr_auc / total_auc

ensemble_mr_prob = w_gb_mr * gb_mr_prob + w_rf_mr * rf_mr_prob + w_lr_mr * lr_mr_prob
ensemble_mr_preds = (ensemble_mr_prob > 0.5).astype(int)
ensemble_mr_acc = accuracy_score(y_test, ensemble_mr_preds)
ensemble_mr_auc = roc_auc_score(y_test, ensemble_mr_prob)

print(f"\nEnsemble weights:")
print(f"   GB: {w_gb_mr:.3f}, RF: {w_rf_mr:.3f}, LR: {w_lr_mr:.3f}")
print(f"   Accuracy: {ensemble_mr_acc:.4f}")
print(f"   AUC: {ensemble_mr_auc:.4f}")

# ========== SAVE MODELS ==========
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(gb_mr, os.path.join(MODEL_DIR, "gb_mr_model.pkl"))
joblib.dump(rf_mr, os.path.join(MODEL_DIR, "rf_mr_model.pkl"))
joblib.dump(lr_mr, os.path.join(MODEL_DIR, "lr_mr_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_mr.pkl"))

ensemble_config_mr = {
    'weights': {'gb': w_gb_mr, 'rf': w_rf_mr, 'lr': w_lr_mr},
    'features': mr_feature_cols,
    'performance': {'accuracy': ensemble_mr_acc, 'auc': ensemble_mr_auc}
}
joblib.dump(ensemble_config_mr, os.path.join(MODEL_DIR, "ensemble_config_mr.pkl"))

print(f"\n✓ Models saved to {MODEL_DIR}")
print("=" * 70)
print("Mean-Reversion Ensemble Training Complete!")
print("=" * 70)
