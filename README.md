# Stock Price Prediction

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi) ![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker) ![XGBoost](https://img.shields.io/badge/XGBoost-1?style=flat) ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=flat&logo=github-actions&logoColor=white) ![AWS EC2](https://img.shields.io/badge/AWS_EC2-FF9900?style=flat&logo=amazon-aws&logoColor=white)

A containerized machine learning REST API providing robust stock price predictions using XGBoost.

## Overview

The Stock Price Prediction API is a production-ready, containerized application built with FastAPI and Streamlit. It leverages an XGBoost regression model trained on 5 years of OHLCV (Open, High, Low, Close, Volume) data to predict stock closing prices. The project features a robust RESTful backend API for fetching supported tickers and generating predictions, paired with an interactive frontend UI. It is fully containerized with Docker and features a zero-downtime CI/CD pipeline integrated via GitHub Actions, designed for seamless deployment on AWS EC2.

## Architecture / Data Flow

1. **Data Ingestion & Preprocessing:** The global stock prices dataset (`World-Stock-Prices-Dataset.csv`) is ingested, cleaned, and enriched with lag features (e.g., `lag_1_close`).
2. **Model Training & Inference:** Multiple regression models are evaluated. The best-performing model per ticker (e.g., XGBoost) is serialized along with its associated `RobustScaler` and feature list into the `models/` directory.
3. **API Routing:** The FastAPI application (`backend/main.py`) exposes endpoints to retrieve available tickers and generate real-time predictions by loading the saved models, scaling the incoming features, and applying inference.
4. **Frontend Integration:** A Streamlit application (`frontend/app.py`) provides an interactive web interface for users to select tickers, input features, and visualize the predicted price via HTTP requests to the backend API.
5. **Deployment:** Both services are orchestrated using `docker-compose.yml`, while a GitHub Actions pipeline (`.github/workflows/docker-ci.yml`) automates the Docker image builds for CI/CD.

## Tech Stack

| Category         | Technologies |
| ---------------- | ------------ |
| **Languages**    | Python 3.13 |
| **Frameworks**   | FastAPI, Streamlit, Uvicorn |
| **ML Libraries** | scikit-learn, XGBoost, pandas, joblib, MLflow |
| **DevOps**       | Docker, Docker Compose, GitHub Actions, AWS EC2 |

## Project Structure

```text
.
├── backend/
│   ├── main.py            # FastAPI application exposing the REST endpoints
│   ├── Dockerfile         # Docker configuration for the backend API service
│   └── requirements.txt   # Backend-specific dependencies
├── frontend/
│   ├── app.py             # Streamlit interactive web application UI
│   ├── Dockerfile         # Docker configuration for the frontend UI service
│   └── requirements.txt   # Frontend-specific dependencies
├── models/                # Serialized trained models, scalers, and feature definitions
├── .github/workflows/
│   └── docker-ci.yml      # GitHub Actions CI workflow for building Docker images
├── World-Stock-Prices-Dataset.csv # Global OHLCV historical dataset used for training
├── train1.py              # ML pipeline script for data processing, training, and evaluation
└── docker-compose.yml     # Orchestration file to run both frontend and backend containers
```

## Setup & Installation

Follow these step-by-step instructions to set up the project locally.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AbdullahRizwan-ML/End-to-End-Stock-Price-Prediction.git
   cd End-to-End-Stock-Price-Prediction
   ```

2. **Set up virtual environments and install dependencies (Optional, for local execution):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r backend/requirements.txt
   pip install -r frontend/requirements.txt
   ```

3. **Run the application using Docker Compose (Recommended):**
   ```bash
   docker-compose up --build
   ```
   - The **Backend API** will be accessible at `http://localhost:8000`
   - The **Frontend UI** will be accessible at `http://localhost:8501`

## Usage / API Endpoints

The FastAPI backend exposes the following REST endpoints:

### 1. Get Available Tickers
Retrieve a list of all supported stock tickers.
- **Endpoint:** `GET /tickers`
- **Example cURL:**
  ```bash
  curl http://localhost:8000/tickers
  ```
- **Example JSON Response:**
  ```json
  {
    "tickers": [
      "AAPL",
      "MSFT",
      "GOOGL"
    ]
  }
  ```

### 2. Predict Stock Price
Generate a closing price prediction for a specified stock ticker based on OHLCV features.
- **Endpoint:** `POST /predict`
- **Example cURL:**
  ```bash
  curl -X POST "http://localhost:8000/predict" \
       -H "Content-Type: application/json" \
       -d '{
             "ticker": "AAPL",
             "data": {
               "Open": 180.50,
               "High": 182.75,
               "Low": 179.25,
               "Volume": 40000000,
               "lag_1_close": 180.00
             }
           }'
  ```
- **Example JSON Response:**
  ```json
  {
    "ticker": "AAPL",
    "prediction": 181.45
  }
  ```
*(You can also use the `requests` library in Python to interact with these endpoints).*

## Key Results / Performance

The machine learning pipeline rigorously evaluated various algorithms (including Linear Regression, Random Forest, Gradient Boosting, SVR, and XGBoost) utilizing Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared (R²) metrics on an 80/20 train-test split. 

The best-performing XGBoost model demonstrated exceptional predictive accuracy, achieving an **R² score of 0.92 on the AAPL** stock dataset. This high R² score highlights the model's strong capability in capturing the underlying stock price variance based on historical prices and generated lag features.

## Author Note

**Abdullah Rizwan - Data Scientist & ML Engineer**
- [GitHub](https://github.com/AbdullahRizwan-ML)
- [LinkedIn](https://www.linkedin.com/in/abdullah-rizwan)
