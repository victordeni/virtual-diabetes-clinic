from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_diabetes

FEATURE_ORDER = ["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"]

def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    Xy = load_diabetes(as_frame=True)
    X = Xy.frame.drop(columns=["target"])
    y = Xy.frame["target"]
    X = X[FEATURE_ORDER]
    return X, y
