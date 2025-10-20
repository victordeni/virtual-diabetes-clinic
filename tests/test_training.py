from ml_service.train import train_and_eval
from ml_service.config import TrainConfig, Paths
import os

def test_training_produces_artifacts(tmp_path):
    paths = Paths(
        artifacts_dir=str(tmp_path),
        model_path=str(tmp_path / "model.joblib"),
        metrics_path=str(tmp_path / "metrics.json"),
        schema_path=str(tmp_path / "schema.json"),
    )
    cfg = TrainConfig(model_type="linear")
    m = train_and_eval(paths, cfg)
    assert os.path.exists(paths.model_path)
    assert os.path.exists(paths.metrics_path)
    assert m["rmse_test"] > 0
