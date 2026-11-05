import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-badness indicators for a TSP distance matrix."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty_like(d, dtype=float)
    if n == 1:
        return np.zeros_like(d, dtype=float)

    finite = np.isfinite(d)
    positive = finite & (d > 0)
    work = np.where(positive, d, np.nan)

    local = np.full(n, 1.0, dtype=float)
    ranks = np.zeros_like(d, dtype=float)

    for i in range(n):
        values = work[i]
        valid = np.isfinite(values)
        if not np.any(valid):
            continue

        sorted_values = np.sort(values[valid])
        k = max(1, min(3, sorted_values.size))
        local[i] = max(float(np.mean(sorted_values[:k])), np.finfo(float).eps)

        count = np.sum(valid)
        order = np.argsort(np.argsort(np.where(valid, values, np.inf)))
        ranks[i, valid] = (order[valid] + 1.0) / max(count, 1)

    scale = np.sqrt(np.outer(local, local))
    scale = np.maximum(scale, np.finfo(float).eps)

    relative_cost = np.divide(
        np.where(positive, d, 0.0),
        scale,
        out=np.zeros_like(d, dtype=float),
        where=scale > 0,
    )

    pressure = 0.5 * (relative_cost + ranks + relative_cost.T + ranks.T)

    valid_edges = positive | positive.T
    pressure = np.where(valid_edges, pressure, 0.0)
    pressure = np.maximum(pressure, 0.0)
    pressure = 0.5 * (pressure + pressure.T)
    np.fill_diagonal(pressure, 0.0)

    max_value = np.max(pressure) if pressure.size else 0.0
    if np.isfinite(max_value) and max_value > 0:
        pressure /= max_value

    return pressure
