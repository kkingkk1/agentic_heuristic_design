import numpy as np


def heuristics_v2(demand: np.ndarray, capacity: int) -> np.ndarray:
    return np.tile(demand / demand.max(), (len(demand), 1))
