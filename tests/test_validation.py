from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)

def test_bad_input_returns_json_error():
    r = client.post("/predict", json={"age": "not-a-number"})
    assert r.status_code in (400, 422)
