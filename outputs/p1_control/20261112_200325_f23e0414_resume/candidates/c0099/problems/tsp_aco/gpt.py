import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric endpoint-aware edge-priority indicators for TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_positive = distances[
        np.isfinite(distances) & (distances > 0.0)
    ]
    fallback = (
        float(np.max(finite_positive))
        if finite_positive.size
        else 1.0
    )
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    clean = np.nan_to_num(
        distances,
        nan=fallback,
        posinf=fallback,
        neginf=fallback,
    )
    clean = np.maximum(clean, 0.0)
    clean = 0.5 * (clean + clean.T)
    np.fill_diagonal(clean, np.inf)

    clean = np.where(np.isfinite(clean), clean, fallback)

    eps = np.finfo(float).eps
    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))

    ordered = np.sort(clean, axis=1)
    nearest = ordered[:, :k]

    local_scale = np.maximum(np.mean(nearest, axis=1), eps)
    alternative = np.maximum(ordered[:, k - 1], eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_distance = clean / (pair_scale + eps)

    endpoint_affinity_i = np.exp(
        -np.clip(clean / (local_scale[:, None] + eps), 0.0, 50.0)
    )
    endpoint_affinity_j = np.exp(
        -np.clip(clean / (local_scale[None, :] + eps), 0.0, 50.0)
    )
    symmetric_affinity = np.sqrt(
        endpoint_affinity_i * endpoint_affinity_j
    )
    worst_endpoint_affinity = np.minimum(
        endpoint_affinity_i,
        endpoint_affinity_j,
    )

    order = np.argsort(clean, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    row_index = np.arange(n)[:, None]
    ranks[row_index, order] = np.arange(n, dtype=float)[None, :]

    rank_score = 1.0 / (1.0 + ranks)
    mutual_rank = np.sqrt(rank_score * rank_score.T)

    competitiveness = 1.0 / (1.0 + relative_distance)
    mutual_competitiveness = np.sqrt(
        competitiveness * competitiveness.T
    )

    substitute_scale = np.sqrt(
        alternative[:, None] * alternative[None, :]
    )
    margin = substitute_scale / (
        clean + substitute_scale + eps
    )
    margin_bonus = 1.0 + 0.10 * margin

    score = (
        symmetric_affinity
        * worst_endpoint_affinity
        * (0.75 + 0.25 * mutual_rank)
        * (0.85 + 0.15 * mutual_competitiveness)
        * margin_bonus
    )

    np.fill_diagonal(score, 0.0)
    score = 0.5 * (score + score.T)

    off_diagonal = ~np.eye(n, dtype=bool)
    values = score[off_diagonal]

    if values.size:
        cutoff = float(np.percentile(values, 80.0))
        score *= np.where(score >= cutoff, 1.05, 1.0)

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

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
