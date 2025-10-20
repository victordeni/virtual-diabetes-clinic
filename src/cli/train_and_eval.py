from ml_service.config import Paths, TrainConfig
from ml_service.train import train_and_eval

if __name__ == "__main__":
    cfg = TrainConfig()
    paths = Paths()
    m = train_and_eval(paths, cfg)
    print(m)
