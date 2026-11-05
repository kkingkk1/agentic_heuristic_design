import numpy as np


def heuristics_v2(demand: np.ndarray, capacity: int) -> np.ndarray:
    return np.full((len(demand), len(demand)), np.nan)
