import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric endpoint-aware edge priorities for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(distances)
    positive = finite & (distances > 0.0)
    np.fill_diagonal(positive, False)

    positive_values = distances[positive]
    if positive_values.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.percentile(positive_values, 90.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = float(np.max(positive_values))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(finite, distances, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    d = np.where(np.isfinite(d), d, fallback)
    np.fill_diagonal(d, 0.0)

    work = d.copy()
    np.fill_diagonal(work, np.inf)

    eps = np.finfo(float).eps
    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))

    local_neighbors = np.partition(work, k - 1, axis=1)[:, :k]
    local_neighbors.sort(axis=1)

    weights = np.linspace(1.0, 0.65, k, dtype=float)
    weights /= np.sum(weights)

    local_mean = np.sum(local_neighbors * weights[None, :], axis=1)
    local_mean = np.maximum(local_mean, eps)

    local_median = np.maximum(
        np.median(local_neighbors, axis=1),
        eps,
    )

    if n > 2:
        alternative = local_neighbors[:, min(1, k - 1)]
    else:
        alternative = local_neighbors[:, 0]
    alternative = np.maximum(alternative, eps)

    pair_mean = np.sqrt(
        local_mean[:, None] * local_mean[None, :]
    )
    pair_median = np.sqrt(
        local_median[:, None] * local_median[None, :]
    )

    relative_mean = d / (pair_mean + eps)
    relative_median = d / (pair_median + eps)

    mean_affinity = np.exp(
        -np.clip(relative_mean ** 0.90, 0.0, 50.0)
    )
    median_affinity = np.exp(
        -np.clip(relative_median ** 0.80, 0.0, 50.0)
    )

    endpoint_i = d / (local_mean[:, None] + eps)
    endpoint_j = d / (local_mean[None, :] + eps)

    endpoint_affinity = np.exp(
        -np.clip(
            0.55 * (endpoint_i ** 0.82 + endpoint_j ** 0.82),
            0.0,
            50.0,
        )
    )

    order = np.argsort(work, axis=1, kind="mergesort")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]

    normalized_rank = ranks / max(float(n - 1), 1.0)
    rank_affinity = np.exp(
        -1.20 * np.sqrt(
            np.clip(
                normalized_rank * normalized_rank.T,
                0.0,
                None,
            )
        )
    )

    candidate_width = max(float(k), 1.0)
    candidate_i = np.exp(
        -np.clip(ranks / candidate_width, 0.0, 50.0)
    )
    candidate_j = np.exp(
        -np.clip(ranks.T / candidate_width, 0.0, 50.0)
    )
    mutual_candidate = np.sqrt(candidate_i * candidate_j)
    candidate_gate = 0.60 + 0.40 * mutual_candidate

    substitute_scale = np.sqrt(
        alternative[:, None] * alternative[None, :]
    )
    margin = substitute_scale / (
        d + substitute_scale + eps
    )
    margin = np.clip(margin, 0.0, 1.0)
    margin_bonus = 1.0 + 0.09 * np.sqrt(margin)

    positive_values = d[d > 0.0]
    global_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else fallback
    )
    global_scale = max(global_scale, eps)

    global_affinity = np.exp(
        -np.clip((d / global_scale) ** 0.85, 0.0, 50.0)
    )
    global_gate = 0.40 + 0.60 * global_affinity

    score = (
        mean_affinity ** 1.10
        * median_affinity ** 0.35
        * endpoint_affinity
        * rank_affinity ** 1.15
        * candidate_gate
        * margin_bonus
        * global_gate
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
