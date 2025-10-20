from __future__ import annotations
from typing import Any
import json
from joblib import dump, load

def save_model(model: Any, meta: dict, path: str) -> None:
    payload = {"model": model, "meta": meta}
    dump(payload, path)

def load_model(path: str) -> tuple[Any, dict]:
    payload = load(path)
    return payload["model"], payload["meta"]

def save_metrics(metrics: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

def save_schema(schema: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
