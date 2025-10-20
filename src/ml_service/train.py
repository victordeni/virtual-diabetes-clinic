from __future__ import annotations
import os
import numpy as np
from typing import Literal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from .config import TrainConfig, Paths, MODEL_VERSION
from .data import load_dataset
from .eval import rmse
from .model_io import save_model, save_metrics, save_schema

def build_pipeline(model_type: Literal["linear","ridge","rf"], cfg: TrainConfig) -> Pipeline:
    if model_type == "linear":
        model = LinearRegression()
        steps = [("scaler", StandardScaler()), ("model", model)]
    elif model_type == "ridge":
        model = Ridge(alpha=cfg.ridge_alpha, random_state=cfg.random_state)
        steps = [("scaler", StandardScaler()), ("model", model)]
    elif model_type == "rf":
        model = RandomForestRegressor(
            n_estimators=cfg.rf_n_estimators,
            max_depth=cfg.rf_max_depth,
            random_state=cfg.random_state,
            n_jobs=-1,
        )
        steps = [("model", model)]
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return Pipeline(steps)

def train_and_eval(paths: Paths, cfg: TrainConfig):
    np.random.seed(cfg.random_state)
    X, y = load_dataset()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=cfg.test_size, random_state=cfg.random_state)
    pipe = build_pipeline(cfg.model_type, cfg)
    pipe.fit(X_tr, y_tr)
    pred_tr = pipe.predict(X_tr)
    pred_te = pipe.predict(X_te)
    metrics = {
        "rmse_train": rmse(y_tr, pred_tr),
        "rmse_test": rmse(y_te, pred_te),
        "model_type": cfg.model_type,
        "random_state": cfg.random_state,
        "test_size": cfg.test_size,
        "model_version": MODEL_VERSION,
    }
    os.makedirs(paths.artifacts_dir, exist_ok=True)
    meta = {"model_version": MODEL_VERSION, "features": X.columns.tolist(), "model_type": cfg.model_type}
    save_model(pipe, meta, paths.model_path)
    save_metrics(metrics, paths.metrics_path)
    schema = {"type": "object", "required": X.columns.tolist(), "properties": {k: {"type": "number"} for k in X.columns}}
    save_schema(schema, paths.schema_path)
    return metrics

if __name__ == "__main__":
    cfg = TrainConfig()
    paths = Paths()
    print(train_and_eval(paths, cfg))
