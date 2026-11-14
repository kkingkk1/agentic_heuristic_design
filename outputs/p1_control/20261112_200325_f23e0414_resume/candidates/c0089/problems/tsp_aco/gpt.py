import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return triangle-shortcut indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_positive = np.isfinite(distances) & (distances > 0.0)
    positive_values = distances[finite_positive]
    if positive_values.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(positive_values))
    d = np.where(np.isfinite(distances), distances, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)

    positive = d[(d > 0.0) & np.isfinite(d)]
    floor = max(
        float(np.median(positive)) * np.finfo(float).eps
        if positive.size
        else np.finfo(float).eps,
        np.finfo(float).tiny,
    )

    np.fill_diagonal(d, np.inf)

    if n < 3:
        score = 1.0 / np.maximum(d, floor)
        score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(score, 0.0)
        maximum = float(np.max(score))
        return score / maximum if maximum > 0.0 else score

    best_detour = np.full((n, n), np.inf, dtype=float)
    best_midpoint = np.full((n, n), -1, dtype=np.intp)

    for middle in range(n):
        candidate = d[:, middle, None] + d[None, middle, :]
        candidate[middle, :] = np.inf
        candidate[:, middle] = np.inf

        improved = candidate < best_detour
        best_detour[improved] = candidate[improved]
        best_midpoint[improved] = middle

    np.fill_diagonal(best_detour, np.inf)

    safe_detour = np.maximum(best_detour, floor)
    direct = np.maximum(d, 0.0)

    shortcut = (safe_detour - direct) / (safe_detour + direct + floor)
    shortcut = np.clip(shortcut, 0.0, 1.0)

    midpoint_valid = best_midpoint >= 0
    rows, cols = np.indices((n, n))
    midpoint_safe = np.maximum(best_midpoint, 0)

    left_leg = d[rows, midpoint_safe]
    right_leg = d[midpoint_safe, cols]
    balance = 1.0 - (
        np.abs(left_leg - right_leg)
        / (left_leg + right_leg + floor)
    )
    balance = np.clip(balance, 0.0, 1.0)

    direct_advantage = 0.5 + 0.5 * shortcut
    score = shortcut * direct_advantage * (0.7 + 0.3 * balance)
    score[~midpoint_valid] = 0.0

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
