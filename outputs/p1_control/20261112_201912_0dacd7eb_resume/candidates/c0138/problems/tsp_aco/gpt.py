import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = np.nan_to_num(
        distances,
        nan=np.inf,
        posinf=np.inf,
        neginf=0.0,
    )
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    finite = np.isfinite(d)
    if not np.any(finite):
        return np.zeros((n, n), dtype=float)

    finite_values = d[finite]
    fallback = float(np.max(finite_values))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0
    d = np.where(finite, d, fallback)

    scale_floor = max(fallback * 1e-12, np.finfo(float).eps)

    k = min(n - 1, max(1, int(np.sqrt(n))))
    nearest_values = np.partition(d, k - 1, axis=1)[:, :k]
    local_scale = np.maximum(np.mean(nearest_values, axis=1), scale_floor)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative_distance = d / pair_scale
    score = 1.0 / (1.0 + relative_distance)

    nearest = np.argmin(d, axis=1)
    mutual_nearest = (
        nearest[nearest] == np.arange(n)
    )
    mutual_edges = mutual_nearest[:, None] & mutual_nearest[None, :]
    score *= np.where(mutual_edges, 1.08, 1.0)

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
