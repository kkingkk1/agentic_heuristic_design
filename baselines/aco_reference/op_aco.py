"""ReEvo 固定提交的 OP 白盒参考启发式。"""

import numpy as np


def heuristics_v2(prize: np.ndarray, distance: np.ndarray, maxlen: float) -> np.ndarray:
    return prize[np.newaxis, :] / distance
