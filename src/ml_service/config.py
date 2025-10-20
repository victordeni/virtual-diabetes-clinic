from dataclasses import dataclass

@dataclass(frozen=True)
class TrainConfig:
    random_state: int = 42
    test_size: float = 0.2
    model_type: str = "linear"  # v0.1 = linear ; v0.2 = ridge/rf
    ridge_alpha: float = 1.0
    rf_n_estimators: int = 300
    rf_max_depth: int | None = None

@dataclass(frozen=True)
class Paths:
    artifacts_dir: str = "artifacts"
    model_path: str = "artifacts/model.joblib"
    metrics_path: str = "artifacts/metrics.json"
    schema_path: str = "artifacts/input_schema.json"

MODEL_VERSION = "v0.1"
