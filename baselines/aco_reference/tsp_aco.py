"""ReEvo 固定提交的 TSP 白盒参考启发式。"""

import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    return 1 / distance_matrix
