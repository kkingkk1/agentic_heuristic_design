"""ReEvo 固定提交的 MKP 白盒参考启发式。"""

import numpy as np


def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return prize / np.sum(weight, axis=1)
