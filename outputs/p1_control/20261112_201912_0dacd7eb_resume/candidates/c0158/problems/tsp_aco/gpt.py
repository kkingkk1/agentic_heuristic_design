import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric endpoint-aware edge priorities for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    off_diagonal = ~np.eye(n, dtype=bool)

    finite_positive = np.isfinite(distances) & (distances > 0.0)
    finite_positive &= off_diagonal

    if not np.any(finite_positive):
        return np.zeros((n, n), dtype=float)

    positive_values = distances[finite_positive]
    fallback = float(np.max(positive_values))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    valid = np.isfinite(distances) & (distances >= 0.0)
    d = np.where(valid, distances, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    finite_offdiag = d[np.isfinite(d) & off_diagonal]
    if finite_offdiag.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(finite_offdiag))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(np.isfinite(d), d, fallback)
    np.fill_diagonal(d, np.inf)

    positive = d[(d > 0.0) & off_diagonal]
    global_scale = (
        float(np.median(positive))
        if positive.size
        else fallback
    )
    global_scale = max(global_scale, eps)

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))

    ordered = np.sort(d, axis=1)
    local_neighbors = ordered[:, :k]
    local_scale = np.mean(local_neighbors, axis=1)
    local_scale = np.maximum(local_scale, eps)

    if n > 2:
        substitute = ordered[:, 1]
    else:
        substitute = ordered[:, 0]
    substitute = np.maximum(substitute, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_length = d / np.maximum(pair_scale, eps)

    pair_affinity = np.exp(
        -np.clip(relative_length ** 0.9, 0.0, 50.0)
    )

    endpoint_i = np.exp(
        -np.clip(d / local_scale[:, None], 0.0, 50.0)
    )
    endpoint_j = np.exp(
        -np.clip(d / local_scale[None, :], 0.0, 50.0)
    )
    mutual_endpoint_affinity = np.sqrt(
        endpoint_i * endpoint_j
    )

    order = np.argsort(d, axis=1, kind="mergesort")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]

    rank_scale = max(float(n - 1), 1.0)
    normalized_rank = ranks / rank_scale
    mutual_rank = np.exp(
        -np.clip(
            0.8 * (normalized_rank + normalized_rank.T),
            0.0,
            50.0,
        )
    )

    competitiveness = 1.0 / (1.0 + relative_length)
    mutual_competitiveness = np.sqrt(
        competitiveness * competitiveness.T
    )

    substitute_scale = np.sqrt(
        substitute[:, None] * substitute[None, :]
    )
    margin = substitute_scale / (
        d + substitute_scale + eps
    )
    margin = np.clip(margin, 0.0, 1.0)
    margin_bonus = 1.0 + 0.14 * margin

    weak_global_term = 0.9 + 0.1 * np.exp(
        -np.clip((d / global_scale) ** 0.85, 0.0, 50.0)
    )

    score = (
        pair_affinity
        * mutual_endpoint_affinity
        * mutual_rank
        * (0.86 + 0.14 * mutual_competitiveness)
        * margin_bonus
        * weak_global_term
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
