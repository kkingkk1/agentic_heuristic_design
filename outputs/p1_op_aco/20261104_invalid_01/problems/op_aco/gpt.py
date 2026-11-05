import numpy as np


def heuristics_v2(prize: np.ndarray, distance: np.ndarray, maxlen: float) -> np.ndarray:
    return np.full(distance.shape, np.nan)
