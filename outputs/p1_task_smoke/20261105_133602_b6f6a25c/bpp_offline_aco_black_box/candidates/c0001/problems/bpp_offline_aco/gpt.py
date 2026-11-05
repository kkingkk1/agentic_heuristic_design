import numpy as np


def heuristics_v2(node_attr: np.ndarray, node_constraint: int) -> np.ndarray:
    """Estimate how promising it is to group each pair of items."""
    values = np.asarray(node_attr, dtype=float).reshape(-1)
    n = values.size

    if n == 0:
        return np.empty((0, 0), dtype=float)

    capacity = float(node_constraint)
    if not np.isfinite(capacity) or capacity <= 0:
        return np.zeros((n, n), dtype=float)

    values = np.nan_to_num(values, nan=capacity, posinf=capacity, neginf=0.0)
    values = np.maximum(values, 0.0)

    pair_sum = values[:, None] + values[None, :]
    feasible = pair_sum <= capacity

    utilization = np.clip(pair_sum / capacity, 0.0, 1.0)
    tightness = utilization ** 2

    scale = max(capacity, float(np.max(values)), 1.0)
    size_difference = np.abs(values[:, None] - values[None, :]) / scale
    balance = 1.0 - np.clip(size_difference, 0.0, 1.0)

    scores = tightness * (0.65 + 0.35 * balance)
    scores *= feasible.astype(float)

    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
