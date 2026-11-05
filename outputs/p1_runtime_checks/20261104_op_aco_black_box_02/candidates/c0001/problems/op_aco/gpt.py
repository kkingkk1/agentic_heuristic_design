import numpy as np


def heuristics_v2(node_attr: np.ndarray, edge_attr: np.ndarray, node_constraint: float) -> np.ndarray:
    return np.ones_like(edge_attr)

