import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return locally calibrated edge-priority indicators for stochastic TSP sampling."""
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

    finite = d[np.isfinite(d)]
    if finite.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(finite))
    d = np.where(np.isfinite(d), d, fallback)

    eps = np.finfo(float).eps
    k = max(1, min(n - 1, int(np.sqrt(n)) + 1))

    if n == 2:
        nearest = d[:, 1].copy()
        second = nearest.copy()
        local_scale = np.maximum(nearest, eps)
        kth_scale = local_scale.copy()
    else:
        nearest_two = np.partition(d, 1, axis=1)[:, :2]
        nearest_two.sort(axis=1)
        nearest = nearest_two[:, 0]
        second = nearest_two[:, 1]

        local_neighbors = np.partition(d, k - 1, axis=1)[:, :k]
        local_scale = np.mean(local_neighbors, axis=1)
        kth_scale = np.max(local_neighbors, axis=1)

    local_scale = np.maximum(local_scale, eps)
    kth_scale = np.maximum(kth_scale, eps)
    second = np.maximum(second, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    candidate_scale = np.sqrt(kth_scale[:, None] * kth_scale[None, :])

    distance_affinity = np.exp(-np.clip(d / pair_scale, 0.0, 50.0))

    sparse_candidate_affinity = np.exp(
        -0.5 * np.clip(d / candidate_scale, 0.0, 50.0)
    )

    margin_i = np.clip(
        (second[:, None] - d) / second[:, None],
        0.0,
        1.0,
    )
    margin_j = np.clip(
        (second[None, :] - d) / second[None, :],
        0.0,
        1.0,
    )
    local_2opt_bonus = 1.0 + 0.25 * np.sqrt(margin_i * margin_j)

    score = (
        distance_affinity
        * sparse_candidate_affinity
        * local_2opt_bonus
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
