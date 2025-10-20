# Virtual Diabetes Clinic – ML Service (v0.1)
Triage par score continu (dataset Diabetes scikit-learn).
## Quickstart
make setup && make train && make api
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d @assets/sample_payload.json
