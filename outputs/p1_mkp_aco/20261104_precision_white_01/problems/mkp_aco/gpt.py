import numpy as np


def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return prize / np.sum(weight, axis=1)
