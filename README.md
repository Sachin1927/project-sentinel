# 🛡️ Real-Time Customer Retention & Intervention Engine
![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://project-sentinel-bwkl4skheyvmajespdxizo.streamlit.app/)

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=for-the-badge&logo=fastapi)
![Feast](https://img.shields.io/badge/Feature_Store-Feast-orange?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)
![Docker](https://img.shields.io/badge/Deployment-Docker-2496ED?style=for-the-badge&logo=docker)

**# Real-Time Customer Retention & Intervention Engine (Project Sentinel)** is an end-to-end MLOps system designed to predict customer churn in real-time. Unlike traditional batch models, Sentinel exposes a low-latency API that allows CRM systems to query a customer's risk score instantly and trigger interventions (e.g., dynamic discounts).

This project demonstrates a complete production lifecycle: form **Feature Engineering** (Feast) to **Model Training** (XGBoost) to **Deployment** (FastAPI & Docker) and **Monitoring** (Streamlit Dashboard).

---

## 🏗️ System Architecture

The system follows a modern Microservices architecture:

1.  **Data Ingestion:** Synthetic telecom data is processed into Parquet format.
2.  **Feature Store (Feast):** Manages offline data for training and online (SQLite/Redis) data for low-latency inference, ensuring point-in-time correctness.
3.  **Modeling Engine:** An **XGBoost Classifier** trained on behavioral signals (usage drops, payment failures, sentiment).
4.  **Serving Layer:** A **FastAPI** backend that serves predictions via REST endpoints.
5.  **Frontend:** A **Streamlit** dashboard for stakeholders to test customer profiles interactively.

---

## 📂 Project Structure

```text
project-sentinel/
├── .github/workflows/    # CI/CD: Automated testing with GitHub Actions
├── config/               # Configuration files (YAML)
├── data/
│   ├── feature_store/    # Feast registry and definitions
│   ├── raw/              # Immutable source data (Parquet)
│   └── processed/        # Batch prediction results
├── docs/                 # Documentation and model performance plots
├── models/               # Serialized model artifacts (.joblib)
├── notebooks/            # Jupyter notebooks for EDA and Prototyping
├── src/
│   ├── api/              # FastAPI application (Backend)
│   ├── dashboard/        # Streamlit application (Frontend)
│   ├── models/           # Training and Batch Prediction scripts
│   └── visualization/    # Plotting utilities
├── tests/                # Unit and Integration tests (Pytest)
├── Dockerfile            # Containerization instructions
├── Makefile              # Shortcut commands
├── requirements.txt      # Python dependencies
└── start_app.py          # Orchestration script to launch the full system

🚀 Quick Start Guide
1. Prerequisites
Python 3.9+

Git

2. Installation
Clone the repository and install dependencies:

git clone [https://github.com/yourusername/project-sentinel.git](https://github.com/yourusername/project-sentinel.git)
cd project-sentinel

# Create virtual environment (Optional but recommended)
python -m venv venv
# Windows
.\venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

Initialize Feature Store
Set up the offline and online stores using Feast:

cd data/feature_store
feast apply
cd ../..

Train the Model
Fetch historical features from Feast and train the XGBoost model:

python src/models/train_model.py

Output: Saves the trained model to models/production/xgb_churn_v1.joblib

🖥️ Usage
Run the Full System (Backend + Frontend)
We have included an orchestration script to start both the FastAPI server and Streamlit dashboard with one command:

python start_app.py

Backend API (Swagger UI): http://127.0.0.1:8000/docs

Frontend Dashboard: http://localhost:8501

Batch Predictions
To score a large dataset of customers at once (e.g., nightly batch job):

python src/models/predict_model.py production
Results saved to: data/processed/batch_predictions.csv

## 📊 Model Performance

The current production model (XGBoost) achieves strong separation between loyal and risky customers:

* **AUC-ROC:** 0.89 (Excellent predictive power)
* **Precision:** 0.63 (Minimizes false alarms while capturing high-value churners)

Run the visualization script to generate current plots:

python src/visualization/visualize.py

Confusion Matrix,ROC Curve
,
### 💰 Business Impact
With a precision of **63%**, this model allows the marketing team to target interventions effectively. For every 100 "High Risk" emails sent, ~63 reach customers who were actually about to leave, significantly improving ROI compared to random targeting (which has a baseline success rate of only ~26%).

🧪 Testing & Quality Assurance
The project enforces code quality using Pytest.

Unit Tests: Validate data generation and feature logic.

Integration Tests: Verify API endpoints return correct risk scores.

Run the test suite:

python -m pytest

🔮 Roadmap (Future Improvements)
While this project implements a complete local MLOps loop, a production-scale deployment would include:

Kubernetes (K8s): To orchestrate Docker containers and enable auto-scaling.

Airflow: To schedule train_model.py pipelines weekly as new data arrives.

Prometheus & Grafana: To monitor API latency and concept drift.

Cloud Deployment: Hosting the Feature Store on AWS/GCP (using Redis for the Online Store).
