This project simulates a small machine learning service for a virtual diabetes clinic.
The goal is to predict the short-term progression risk of diabetes, based on basic clinical data, so that nurses can prioritize patients who might need a follow-up.

The service is built around a regression model (using scikit-learn), exposed through a FastAPI endpoint, and fully automated via GitHub Actions and Docker.

Context

Each week, a virtual clinic receives hundreds of patient check-ins: vitals, lab results, lifestyle notes, etc.
Manual review is slow and inconsistent.
This project automates the process by assigning each patient a continuous risk score representing the likelihood of short-term deterioration.

The model uses the open load_diabetes dataset from scikit-learn as a stand-in for anonymized electronic health records (EHR).

Model & Iterations
v0.1 — Baseline

Preprocessing: StandardScaler

Model: LinearRegression

RMSE (test): 53.85

v0.2 — Improvement

Model: Ridge(alpha=1.0)

RMSE (test): 53.78 → –0.14 % improvement

The L2 regularization slightly improves generalization and stabilizes coefficients.

All experiments are deterministic (random_state=42), and model artifacts are stored in /artifacts.

 API Endpoints
GET /health

Quick health check to verify that the service is running.

{"status": "ok", "model_version": "v0.2"}

POST /predict

Send a patient feature vector and receive a continuous risk score.

Example request:

curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "age": 0.02,
    "sex": -0.044,
    "bmi": 0.06,
    "bp": -0.03,
    "s1": -0.02,
    "s2": 0.03,
    "s3": -0.02,
    "s4": 0.02,
    "s5": 0.02,
    "s6": -0.001
  }'


Response:

{"prediction": 235.94, "model_version": "v0.2"}

 Run the project
 Locally
git clone https://github.com/victordeni/virtual-diabetes-clinic.git
cd virtual-diabetes-clinic

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

PYTHONPATH=src python -m src.cli.train_and_eval
uvicorn app.main:app --port 8000


Then test it:

curl -s http://localhost:8000/health

 With Docker
# Baseline version (LinearRegression)
docker pull ghcr.io/victordeni/virtual-diabetes-clinic:v0.1
docker run -p 8000:8000 ghcr.io/victordeni/virtual-diabetes-clinic:v0.1

# Improved version (Ridge)
docker pull ghcr.io/victordeni/virtual-diabetes-clinic:v0.2
docker run -p 8000:8000 ghcr.io/victordeni/virtual-diabetes-clinic:v0.2

 CI/CD Pipeline (GitHub Actions)

Two GitHub Actions workflows handle automation:

Workflow	Trigger	Main steps
CI	on push/PR	Lint, run unit tests, quick training smoke test
Release	on tag v*	Full training, build Docker image, smoke-test container, push to GHCR, publish GitHub Release

Everything is fully reproducible — same code, same metrics, same Docker image.

🧾 Repository structure
Path	Description
src/	ML logic: training, model I/O, config, evaluation
app/	FastAPI application
artifacts/	Generated model + metrics
.github/workflows/	CI/CD pipelines
Dockerfile	Multi-stage build (train + runtime)
CHANGELOG.md	Model versions & RMSE evolution
tests/	Unit tests
