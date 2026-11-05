import numpy as np


def heuristics_v2(node_attr: np.ndarray, node_constraint: int) -> np.ndarray:
    return np.ones((len(node_attr), len(node_attr)))
