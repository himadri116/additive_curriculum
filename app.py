import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import subprocess

st.set_page_config(page_title="📈 Sales Forecast Dashboard", layout="wide")
st.title("📈 Sales Forecast Dashboard (Prophet + LightGBM Ensemble)")

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "brake_sales_clean.csv"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

# ---------------------------------------
# Load dataset
# ---------------------------------------
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["date"])

series = df.groupby("date")["item_cnt_day"].sum().asfreq("D").ffill()

st.subheader("📊 Sales Trend (Last 200 Days)")
st.line_chart(series.tail(200))

# ---------------------------------------
# Training Button
# ---------------------------------------
st.subheader("⚙️ Train Models")
if st.button("Train Now"):
    with st.spinner("Training Prophet + LightGBM + Ensemble..."):
        subprocess.run(["python", str(BASE_DIR / "train_model.py")])
    st.success("🎉 Training completed!")

# ---------------------------------------
# Show Metrics
# ---------------------------------------
st.subheader("📈 Model Performance Metrics")

metrics_path = OUTPUT_DIR / "metrics.json"
best_name_path = MODELS_DIR / "best_model_name.txt"

if metrics_path.exists():
    metrics = pd.read_json(metrics_path).T
    st.dataframe(metrics)

    if best_name_path.exists():
        best_name = best_name_path.read_text().strip()
        st.success(f"🏆 Best Model: **{best_name}**")

else:
    st.warning("⚠ Train the models first.")
    st.stop()

# ---------------------------------------
# Forecast Section
# ---------------------------------------
st.subheader("🔮 Forecast Future Sales")

best_marker_path = MODELS_DIR / "best_model.pkl"
if not best_marker_path.exists():
    st.error("❌ Model not found. Please train first.")
    st.stop()

best_marker = joblib.load(best_marker_path)

horizon = st.slider("Forecast Horizon (Days)", 7, 180, 30)

# Load all models
prophet_model = joblib.load(MODELS_DIR / "prophet.pkl")
lgb_model = joblib.load(MODELS_DIR / "lightgbm.pkl")
xgb_model = joblib.load(MODELS_DIR / "xgboost.pkl")
rf_model = joblib.load(MODELS_DIR / "randomforest.pkl")
svr_model = joblib.load(MODELS_DIR / "svr.pkl")

ensemble_config = joblib.load(MODELS_DIR / "ensemble_config.pkl")
prophet_weight = ensemble_config["prophet_weight"]

# ---------------------------------------
# LightGBM Recursive Forecast Helper
# ---------------------------------------
def forecast_lgb(h):
    last = series.values[-30:].tolist()
    preds = []

    for i in range(h):
        lag_1 = last[-1]
        lag_2 = last[-2]
        lag_3 = last[-3]
        lag_7 = last[-7]
        lag_14 = last[-14]
        lag_30 = last[-30]

        roll_mean_3  = np.mean(last[-3:])
        roll_mean_7  = np.mean(last[-7:])
        roll_mean_14 = np.mean(last[-14:])
        roll_mean_30 = np.mean(last[-30:])

        roll_std_3  = np.std(last[-3:])
        roll_std_7  = np.std(last[-7:])
        roll_std_14 = np.std(last[-14:])
        roll_std_30 = np.std(last[-30:])

        today = pd.to_datetime(series.index[-1]) + pd.Timedelta(days=i+1)
        dayofweek = today.dayofweek
        month = today.month
        is_weekend = 1 if dayofweek >= 5 else 0

        X = pd.DataFrame([[
            lag_1, lag_2, lag_3, lag_7, lag_14, lag_30,
            roll_mean_3, roll_mean_7, roll_mean_14, roll_mean_30,
            roll_std_3, roll_std_7, roll_std_14, roll_std_30,
            dayofweek, month, is_weekend
        ]], columns=[
            "lag_1","lag_2","lag_3","lag_7","lag_14","lag_30",
            "roll_mean_3","roll_mean_7","roll_mean_14","roll_mean_30",
            "roll_std_3","roll_std_7","roll_std_14","roll_std_30",
            "dayofweek","month","is_weekend"
        ])

        p = lgb_model.predict(X)[0]
        preds.append(p)
        last.append(p)

    return np.array(preds)


# ---------------------------------------
# Forecast Logic
# ---------------------------------------
if isinstance(best_marker, dict) and best_marker.get("type") == "ensemble":
    st.success("Using **Ensemble (Prophet + LightGBM)**")

    # Prophet forecast
    future = prophet_model.make_future_dataframe(periods=horizon)
    fcst = prophet_model.predict(future)
    prophet_fc = fcst["yhat"].values[-horizon:]

    # LightGBM forecast
    lgb_fc = forecast_lgb(horizon)

    # Blend
    final_fc = prophet_weight * prophet_fc + (1 - prophet_weight) * lgb_fc

    st.write(f"### Ensemble Forecast (w={prophet_weight:.2f})")
    st.line_chart(final_fc)
    st.dataframe(pd.DataFrame({"forecast": final_fc}))

else:
    # In case Prophet or LGBM alone is selected
    st.success(f"Using best model: **{best_name}**")

    if best_name == "Prophet":
        future = prophet_model.make_future_dataframe(periods=horizon)
        fcst = prophet_model.predict(future)
        vals = fcst["yhat"].values[-horizon:]
        st.line_chart(vals)
        st.dataframe(pd.DataFrame({"forecast": vals}))

    else:
        # LightGBM / RF / XGB / SVR fallback (recursive)
        preds = forecast_lgb(horizon)
        st.line_chart(preds)
        st.dataframe(pd.DataFrame({"forecast": preds}))
