import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import mlflow
import mlflow.sklearn
import joblib
import os

# Set MLflow tracking URI (local for now)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Stock_Price_Prediction")

def load_and_preprocess_data(file_path):
    """Load and preprocess the stock data"""
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        df = df.sort_values(['Ticker', 'Date'])
        if 'Capital Gains' in df.columns:
            df = df.drop(columns=['Capital Gains'])
        df = df.fillna(method='ffill').dropna()
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None

def create_features(df, ticker, lag_periods=1):  # Reduced to 1 lag
    """Create simplified features for a specific ticker"""
    df_ticker = df[df['Ticker'] == ticker].copy()
    if df_ticker.empty or len(df_ticker) < lag_periods + 2:  # Adjusted minimum rows
        print(f"Skipping {ticker}: insufficient data")
        return None
    df_ticker = df_ticker.sort_values('Date')
    
    # Create lag feature (only lag_1_close)
    df_ticker['lag_1_close'] = df_ticker['Close'].shift(1)
    
    # Drop rows with NaN values
    df_ticker = df_ticker.dropna()
    
    return df_ticker

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and return metrics"""
    preds = model.predict(X_test)
    try:
        rmse = mean_squared_error(y_test, preds, squared=False)
    except TypeError:
        rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': preds
    }

def train_and_evaluate_company(df, company, feature_cols, models):
    """Train and evaluate models for a single company"""
    print(f"\nTraining models for: {company}")
    
    df_company = create_features(df, company)
    if df_company is None:
        return None, None
    
    print(f"Dataset shape after feature engineering: {df_company.shape}")
    
    X = df_company[feature_cols]
    y = df_company['Close']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
    
    results = {}
    best_model = None
    best_r2 = -float('inf')
    
    for name, model in models.items():
        print(f"  Training {name}...")
        with mlflow.start_run(run_name=f"{name}_{company}"):
            if hasattr(model, 'get_params'):
                mlflow.log_params(model.get_params())
            mlflow.log_param("company", company)
            mlflow.log_param("features", len(feature_cols))
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            
            model.fit(X_train_scaled, y_train)
            metrics = evaluate_model(model, X_test_scaled, y_test, name)
            results[name] = metrics
            
            mlflow.log_metric("rmse", metrics['rmse'])
            mlflow.log_metric("mae", metrics['mae'])
            mlflow.log_metric("r2", metrics['r2'])
            mlflow.sklearn.log_model(model, f"model_{name.lower().replace(' ', '_')}")
            
            print(f"    RMSE = {metrics['rmse']:.4f}")
            print(f"    MAE = {metrics['mae']:.4f}")
            print(f"    R² = {metrics['r2']:.4f}")
            
            if metrics['r2'] > best_r2:
                best_r2 = metrics['r2']
                best_model = (name, model, scaler)
    
    if best_model:
        model_name, model_obj, scaler_obj = best_model
        print(f"  Best model: {model_name} (R² = {best_r2:.4f})")
        
        os.makedirs("models", exist_ok=True)
        joblib.dump(model_obj, f"models/best_model_{company}.pkl")
        joblib.dump(scaler_obj, f"models/scaler_{company}.pkl")
        with open(f"models/features_{company}.txt", "w") as f:
            f.write("\n".join(feature_cols))
        print(f"  Model saved as: models/best_model_{company}.pkl")
    
    return results, best_model

def main():
    print("Loading dataset...")
    df = load_and_preprocess_data("World-Stock-Prices-Dataset.csv")
    if df is None:
        return None, None
    
    tickers = df['Ticker'].unique()
    print(f"Available tickers: {tickers[:10]}...")
    
    companies = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    companies = [c for c in companies if c in tickers]
    if not companies:
        print("No valid companies found in dataset. Using first 3 tickers.")
        companies = tickers[:3].tolist()
    
    # Reduced feature set
    feature_cols = ['Open', 'High', 'Low', 'Volume', 'lag_1_close']
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(
            n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
        ),
        'XGBoost': XGBRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42
        ),
        'SVR': SVR(kernel='rbf', C=100, gamma='scale')
    }
    
    all_results = {}
    all_best_models = {}
    
    for company in companies:
        results, best_model = train_and_evaluate_company(df, company, feature_cols, models)
        if results is not None:
            all_results[company] = results
            all_best_models[company] = best_model
    
    if all_results:
        print("\nSummary of Best Models:")
        summary_data = []
        for company in all_results:
            best_name = max(all_results[company], key=lambda k: all_results[company][k]['r2'])
            summary_data.append({
                'Company': company,
                'Best Model': best_name,
                'R²': all_results[company][best_name]['r2'],
                'RMSE': all_results[company][best_name]['rmse'],
                'MAE': all_results[company][best_name]['mae']
            })
        summary_df = pd.DataFrame(summary_data)
        print(summary_df)
    
    return all_results, all_best_models

if __name__ == "__main__":
    all_results, all_best_models = main()
    print("\nTraining completed! Check MLflow UI with: mlflow ui")