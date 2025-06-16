from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import logging

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reduced feature set
feature_cols = ['Open', 'High', 'Low', 'Volume', 'lag_1_close']

class PredictionInput(BaseModel):
    ticker: str
    data: dict  # Expected to contain feature_cols keys

@app.get("/tickers")
async def get_tickers():
    try:
        df = pd.read_csv("World-Stock-Prices-Dataset.csv")
        return {"tickers": df['Ticker'].unique().tolist()}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found")

@app.post("/predict")
async def predict(input: PredictionInput):
    try:
        model_path = f"models/best_model_{input.ticker}.pkl"
        scaler_path = f"models/scaler_{input.ticker}.pkl"
        features_path = f"models/features_{input.ticker}.txt"
        
        logger.info(f"Checking files for ticker {input.ticker}: {model_path}, {scaler_path}, {features_path}")
        if not all(os.path.exists(p) for p in [model_path, scaler_path, features_path]):
            raise HTTPException(status_code=404, detail=f"Model or scaler for {input.ticker} not found")
        
        logger.info("Loading model and scaler")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        with open(features_path, "r") as f:
            saved_features = f.read().splitlines()
        
        logger.info(f"Saved features: {saved_features}")
        input_data = pd.DataFrame([input.data])
        if not all(f in input_data.columns for f in saved_features):
            raise HTTPException(status_code=400, detail=f"Invalid input features: expected {saved_features}, got {list(input_data.columns)}")
        
        logger.info(f"Input data: {input_data.to_dict()}")
        input_scaled = scaler.transform(input_data[saved_features])
        prediction = model.predict(input_scaled)[0]
        logger.info(f"Prediction for {input.ticker}: {prediction}")
        return {"ticker": input.ticker, "prediction": float(prediction)}
    except Exception as e:
        logger.error(f"Prediction error for {input.ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")