import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return directed edge desirability indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid = np.isfinite(distances) & (distances > 0)
    valid &= ~np.eye(n, dtype=bool)

    positive_values = distances[valid]
    if positive_values.size == 0:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    # Row-local scales preserve directed information and remain robust to
    # heterogeneous distance magnitudes across source nodes.
    row_min = np.full(n, np.inf, dtype=float)
    for i in range(n):
        row_values = distances[i, valid[i]]
        if row_values.size:
            row_min[i] = np.min(row_values)

    global_scale = float(np.median(positive_values))
    row_min = np.where(np.isfinite(row_min), row_min, global_scale)
    row_min = np.maximum(row_min, eps)

    relative_length = distances / row_min[:, None]
    relative_length = np.where(valid, relative_length, np.inf)

    # Main directional affinity: inverse relative length with a smooth tail.
    inverse_relative = np.where(
        valid,
        1.0 / np.maximum(relative_length, eps),
        0.0,
    )
    proximity = np.where(
        valid,
        1.0 / (1.0 + np.maximum(relative_length, 0.0)),
        0.0,
    )

    # Endpoint regret: emphasize rows where the best edge is substantially
    # better than the second-best edge, using a bounded multiplicative bonus.
    best = np.full(n, np.inf, dtype=float)
    second = np.full(n, np.inf, dtype=float)

    for i in range(n):
        row_values = distances[i, valid[i]]
        if row_values.size == 0:
            best[i] = global_scale
            second[i] = global_scale
        elif row_values.size == 1:
            best[i] = row_values[0]
            second[i] = row_values[0]
        else:
            ordered = np.partition(row_values, 1)
            best[i] = ordered[0]
            second[i] = ordered[1]

    regret = np.maximum(second - best, 0.0) / np.maximum(second, eps)
    regret_bonus = 1.0 + 0.45 * regret[:, None]

    score = inverse_relative * proximity * regret_bonus

    # Weak destination-side consistency reward. It is only a tie-breaker and
    # cannot create a score for an invalid directed edge.
    reverse_valid = valid.T
    reverse_score = np.where(
        reverse_valid,
        1.0 / np.maximum(distances.T / row_min[:, None], eps),
        0.0,
    )
    consistency = np.sqrt(
        np.maximum(score, 0.0) * np.maximum(reverse_score, 0.0)
    )
    score += 0.12 * consistency

    score = np.where(valid, score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0:
        score = score / maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
