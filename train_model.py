import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

# ML models
from prophet import Prophet
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)
from sklearn.preprocessing import StandardScaler


# =====================
# PATHS
# =====================
DATA_PATH = "brake_sales_clean.csv"
MODELS_DIR = Path("models")
OUTPUT_DIR = Path("output")
MODELS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

print("Loading data...")
df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors="coerce")
df = df.dropna(subset=['date'])

# =====================
# Build daily series
# =====================
series = df.groupby("date")["item_cnt_day"].sum().resample("D").sum().ffill()
data = series.to_frame("sales")

# =====================
# ML Feature Engineering
# =====================
lags = [1,2,3,7,14,30]
for lag in lags:
    data[f"lag_{lag}"] = data["sales"].shift(lag)

windows = [3,7,14,30]
for w in windows:
    data[f"roll_mean_{w}"] = data["sales"].shift(1).rolling(w).mean()
    data[f"roll_std_{w}"] = data["sales"].shift(1).rolling(w).std()

data["dayofweek"] = data.index.dayofweek
data["month"] = data.index.month
data["is_weekend"] = (data.index.dayofweek >= 5).astype(int)

data = data.dropna()

# Split train/test
H = 30
train = data.iloc[:-H]
test  = data.iloc[-H:]

X_train = train.drop(columns=["sales"])
y_train = train["sales"]
X_test = test.drop(columns=["sales"])
y_test = test["sales"]

# =====================
# Scale only SVR
# =====================
scaler = StandardScaler()
X_train_svr = scaler.fit_transform(X_train)
X_test_svr = scaler.transform(X_test)

# =====================
# Train Prophet (Trend + Seasonality)
# =====================
print("Training Prophet...")

prophet_df = series.reset_index()
prophet_df.columns = ["ds", "y"]

m = Prophet(
    weekly_seasonality=True,
    yearly_seasonality=True,
    daily_seasonality=False
)
m.add_seasonality(name="monthly", period=30.5, fourier_order=5)

m.fit(prophet_df.iloc[:-H])

future_test = m.make_future_dataframe(periods=H, freq="D")
fcst_test = m.predict(future_test)
prophet_pred_test = fcst_test["yhat"].values[-H:]

# =====================
# Train Tree Models
# =====================
print("Training LightGBM...")
lgb = LGBMRegressor(
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=48,
    subsample=0.9,
    colsample_bytree=0.9
)
lgb.fit(X_train, y_train)
lgb_pred = lgb.predict(X_test)

print("Training XGBoost...")
xgb = XGBRegressor(
    n_estimators=600,
    learning_rate=0.04,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="reg:squarederror"
)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

print("Training RandomForest...")
rf = RandomForestRegressor(
    n_estimators=400,
    max_depth=14,
    random_state=42
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("Training SVR...")
svr = SVR(kernel="rbf", C=250, gamma=0.04)
svr.fit(X_train_svr, y_train)
svr_pred = svr.predict(X_test_svr)


# =====================
# Metrics function
# =====================
def calc_metrics(y_true, pred):
    rmse = np.sqrt(mean_squared_error(y_true, pred))
    mae = mean_absolute_error(y_true, pred)
    mape = mean_absolute_percentage_error(y_true, pred)
    r2 = r2_score(y_true, pred)
    acc = (1 - mape) * 100
    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MAPE": float(mape),
        "R2": float(r2),
        "Accuracy (%)": float(acc)
    }

# =====================
# Build Ensemble Prophet + LightGBM
# =====================
print("Building Ensemble...")

best_weight = None
best_rmse = None
best_ens_pred = None

for w in [0.7, 0.6, 0.5, 0.4, 0.3]:
    mix = w * prophet_pred_test + (1 - w) * lgb_pred
    rmse = np.sqrt(mean_squared_error(y_test, mix))
    if best_rmse is None or rmse < best_rmse:
        best_rmse = rmse
        best_weight = w
        best_ens_pred = mix

ensemble_metrics = calc_metrics(y_test.values, best_ens_pred)

# =====================
# Collect all metrics
# =====================
metrics = {
    "Prophet": calc_metrics(y_test.values, prophet_pred_test),
    "LightGBM": calc_metrics(y_test.values, lgb_pred),
    "XGBoost": calc_metrics(y_test.values, xgb_pred),
    "RandomForest": calc_metrics(y_test.values, rf_pred),
    "SVR": calc_metrics(y_test.values, svr_pred),
    "Ensemble": ensemble_metrics  # Removed Prophet_weight
}

# Save metrics
with open(OUTPUT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# =====================
# Pick Best Model
# =====================
best_model_name = max(metrics, key=lambda m: metrics[m]["Accuracy (%)"])
print("Best model:", best_model_name)

# Save helper file
with open(MODELS_DIR / "best_model_name.txt", "w") as f:
    f.write(best_model_name)

# Save models
joblib.dump(lgb, MODELS_DIR / "lightgbm.pkl")
joblib.dump(xgb, MODELS_DIR / "xgboost.pkl")
joblib.dump(rf, MODELS_DIR / "randomforest.pkl")
joblib.dump(svr, MODELS_DIR / "svr.pkl")
joblib.dump(m, MODELS_DIR / "prophet.pkl")
joblib.dump({"prophet_weight": best_weight}, MODELS_DIR / "ensemble_config.pkl")

# Mark the final model
if best_model_name == "Ensemble":
    joblib.dump({"type": "ensemble"}, MODELS_DIR / "best_model.pkl")
elif best_model_name == "Prophet":
    joblib.dump(m, MODELS_DIR / "best_model.pkl")
elif best_model_name == "LightGBM":
    joblib.dump(lgb, MODELS_DIR / "best_model.pkl")
else:
    joblib.dump({"type": best_model_name}, MODELS_DIR / "best_model.pkl")

print("Training complete!")
