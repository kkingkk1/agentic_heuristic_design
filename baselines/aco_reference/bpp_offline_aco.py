"""ReEvo 固定提交的离线 BPP 白盒参考启发式。"""

import numpy as np


def heuristics_v2(demand: np.ndarray, capacity: int) -> np.ndarray:
    return np.tile(demand / demand.max(), (demand.shape[0], 1))
