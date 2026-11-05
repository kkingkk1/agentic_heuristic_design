import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    alpha = 0.7
    return np.ones_like(prize) * alpha
