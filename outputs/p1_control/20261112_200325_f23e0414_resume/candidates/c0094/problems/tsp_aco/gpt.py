import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally scaled, two-hop-indispensability edge priors for TSP."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(distances)
    positive = distances[finite & (distances > 0.0)]

    if positive.size:
        fallback = float(np.max(positive))
        global_scale = float(np.median(positive))
    else:
        fallback = 1.0
        global_scale = 1.0

    d = np.nan_to_num(
        distances,
        nan=fallback,
        posinf=fallback,
        neginf=0.0,
    )
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    off_diagonal = ~np.eye(n, dtype=bool)
    safe_d = d.copy()
    np.fill_diagonal(safe_d, np.inf)

    if n > 2:
        two_hop = np.full((n, n), np.inf, dtype=float)
        for pivot in range(n):
            candidate = safe_d[:, pivot, None] + safe_d[None, pivot, :]
            two_hop = np.minimum(two_hop, candidate)
    else:
        two_hop = np.full((n, n), np.inf, dtype=float)

    two_hop = np.where(np.isfinite(two_hop), two_hop, d + global_scale)

    indispensability = (
        (two_hop - d)
        / np.maximum(two_hop + d, eps)
    )
    indispensability = np.clip(indispensability, 0.0, 1.0)

    reciprocal_distance = 1.0 / (
        1.0 + d / global_scale
    )

    score = reciprocal_distance * (
        0.55 + 0.45 * indispensability
    )
    score[~off_diagonal] = 0.0
    score = 0.5 * (score + score.T)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
