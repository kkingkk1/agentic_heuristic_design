import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return local endpoint-aware edge-priority indicators for TSP sampling."""
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
    if not np.isfinite(fallback):
        fallback = 1.0
    d = np.where(np.isfinite(d), d, fallback)

    eps = np.finfo(float).eps
    positive = finite[finite > 0.0]
    global_scale = (
        float(np.median(positive))
        if positive.size
        else max(fallback, 1.0)
    )
    global_scale = max(global_scale, eps)

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n))) + 1))
    ordered = np.sort(d, axis=1)

    neighbor_distances = ordered[:, :k]
    weights = np.linspace(1.0, 0.6, k, dtype=float)
    local_scale = (
        neighbor_distances @ weights / np.sum(weights)
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_distance = d / (pair_scale + eps)

    distance_term = np.exp(
        -np.clip(relative_distance ** 0.9, 0.0, 50.0)
    )

    work = d.copy()
    np.fill_diagonal(work, np.inf)

    order = np.argsort(work, axis=1, kind="mergesort")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_scale = max(float(n - 1), 1.0)
    normalized_rank = ranks / rank_scale
    mutual_rank = np.exp(
        -0.9 * (normalized_rank + normalized_rank.T)
    )

    endpoint_ratio_i = d / (local_scale[:, None] + eps)
    endpoint_ratio_j = d / (local_scale[None, :] + eps)

    endpoint_compatibility = np.exp(
        -0.5 * (
            np.log1p(endpoint_ratio_i)
            + np.log1p(endpoint_ratio_j)
        )
    )

    alternative = np.partition(work, k - 1, axis=1)[:, k - 1]
    alternative = np.maximum(alternative, eps)

    substitute_scale = np.sqrt(
        alternative[:, None] * alternative[None, :]
    )
    margin = substitute_scale / (d + substitute_scale + eps)
    margin_bonus = 0.92 + 0.08 * np.clip(margin, 0.0, 1.0)

    global_adjustment = 0.8 + 0.2 * np.exp(
        -np.clip((d / global_scale) ** 0.9, 0.0, 50.0)
    )

    score = (
        distance_term
        * mutual_rank
        * endpoint_compatibility
        * margin_bonus
        * global_adjustment
    )

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
