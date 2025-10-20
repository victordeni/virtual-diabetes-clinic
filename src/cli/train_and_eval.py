from ml_service.train import train_and_eval
from ml_service.config import TrainConfig, Paths

if __name__ == "__main__":
    cfg = TrainConfig()
    paths = Paths()
    m = train_and_eval(paths, cfg)
    print(m)
