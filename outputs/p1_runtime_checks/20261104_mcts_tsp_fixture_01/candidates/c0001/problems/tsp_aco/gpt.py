import numpy as np
def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    return 1.0 / (distance_matrix + 1e-8)
