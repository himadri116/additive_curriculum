from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np

app = FastAPI(title="Sales Forecasting API")

class PredictRequest(BaseModel):
    horizon: int = 30

@app.post("/predict")
def predict(req: PredictRequest):
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/best_model.scaler")

    df = pd.read_csv("brake_sales_clean.csv")
    df['date'] = pd.to_datetime(df['date'])
    series = df.groupby('date')['item_cnt_day'].sum().asfreq('D').fillna(method='ffill')

    last = series.values[-30:]
    preds = []

    for _ in range(req.horizon):
        lag1 = last[-1]
        lag7 = last[-7]
        lag30 = last[-30]
        roll7 = last[-7:].mean()
        roll30 = last[-30:].mean()

        X = pd.DataFrame([[lag1, lag7, lag30, roll7, roll30]], 
                         columns=['lag_1','lag_7','lag_30','roll_7','roll_30'])

        Xs = scaler.transform(X)
        p = model.predict(Xs)[0]
        preds.append(float(p))
        last = np.append(last, p)

    return {"forecast": preds}
