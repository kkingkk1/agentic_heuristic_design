import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-priority indicators for stochastic TSP solution sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = np.nan_to_num(
        distances,
        nan=np.inf,
        posinf=np.inf,
        neginf=0.0,
    )
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    finite_values = d[np.isfinite(d)]
    if finite_values.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(finite_values))
    if not np.isfinite(fallback):
        fallback = 1.0
    d = np.where(np.isfinite(d), d, fallback)

    k = max(1, min(n - 1, int(np.sqrt(n)) + 1))
    nearest = np.partition(d, kth=k - 1, axis=1)[:, :k]
    local_scale = np.mean(nearest, axis=1)

    positive = d[d > 0.0]
    scale_floor = (
        float(np.median(positive))
        if positive.size
        else max(fallback, 1.0)
    )
    eps = np.finfo(float).eps
    local_scale = np.maximum(local_scale, max(scale_floor * eps, eps))

    neighbor_indices = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
    candidate = np.zeros((n, n), dtype=bool)
    rows = np.arange(n)[:, None]
    candidate[rows, neighbor_indices] = True
    candidate = candidate | candidate.T

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    pair_scale = np.maximum(pair_scale, eps)

    normalized_distance = d / pair_scale
    distance_term = 1.0 / (1.0 + normalized_distance)

    nearest_baseline = nearest[:, 0]
    detour = d - 0.5 * (
        nearest_baseline[:, None] + nearest_baseline[None, :]
    )
    detour = np.maximum(detour, 0.0)
    detour_term = 1.0 / (1.0 + detour / pair_scale)

    candidate_term = np.where(candidate, 1.0, 0.25)

    score = (
        distance_term
        * (0.65 + 0.35 * detour_term)
        * candidate_term
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
