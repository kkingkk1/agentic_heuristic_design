import numpy as np


def heuristics_v2(distance_matrix: np.ndarray, coordinates: np.ndarray, demands: np.ndarray, capacity: int) -> np.ndarray:
    return np.full(distance_matrix.shape, np.nan)
