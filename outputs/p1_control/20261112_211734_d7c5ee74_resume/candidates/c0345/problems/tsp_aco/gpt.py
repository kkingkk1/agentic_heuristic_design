import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_positive = np.isfinite(d) & (d > 0.0)
    positive = d[finite_positive]
    fallback = float(np.median(positive)) if positive.size else 1.0

    d = np.where(np.isfinite(d), d, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    eps = np.finfo(float).eps
    work = d.copy()
    np.fill_diagonal(work, np.inf)

    m = n - 1
    k = max(1, min(m, int(np.ceil(np.sqrt(n)))))
    candidate_width = max(k, min(m, 2 * k))

    order = np.argsort(work, axis=1, kind="mergesort")
    nearest_order = order[:, :candidate_width]
    nearest_distances = np.take_along_axis(work, nearest_order, axis=1)

    local_scale = np.maximum(
        np.mean(nearest_distances[:, :k], axis=1),
        eps,
    )
    kth_distance = np.maximum(nearest_distances[:, k - 1], eps)

    endpoint_scale = np.sqrt(
        np.maximum(local_scale[:, None] * local_scale[None, :], eps)
    )
    normalized_distance = d / endpoint_scale

    attraction = 1.0 / (1.0 + normalized_distance)
    attraction *= np.exp(-0.35 * np.minimum(normalized_distance, 12.0))

    rank = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    rank[rows, order] = np.arange(m + 1, dtype=float)

    rank_score = 1.0 / np.sqrt(
        (1.0 + rank) * (1.0 + rank.T)
    )
    rank_score = np.clip(rank_score * (k + 1.0), 0.0, 1.0)

    mutual_knn = (
        (rank < float(k)) &
        (rank.T < float(k))
    ).astype(float)

    candidate_score = 0.88 + 0.12 * rank_score
    mutual_score = 1.0 + 0.22 * mutual_knn

    relative_i = d / np.maximum(local_scale[:, None], eps)
    relative_j = d / np.maximum(local_scale[None, :], eps)

    competition = 1.0 / np.sqrt(
        (1.0 + relative_i) * (1.0 + relative_j)
    )
    competition_score = 0.90 + 0.10 * np.clip(competition, 0.0, 1.0)

    regret_i = np.clip(
        (kth_distance[:, None] - d)
        / (kth_distance[:, None] + d + eps),
        0.0,
        1.0,
    )
    regret_j = np.clip(
        (kth_distance[None, :] - d)
        / (kth_distance[None, :] + d + eps),
        0.0,
        1.0,
    )
    regret_score = 0.95 + 0.05 * (0.5 * (regret_i + regret_j))

    neighborhood = np.zeros((n, n), dtype=float)
    neighborhood[rows, nearest_order] = np.exp(
        -nearest_distances
        / np.maximum(local_scale[:, None], eps)
    )
    np.fill_diagonal(neighborhood, 0.0)

    row_norm = np.sqrt(
        np.sum(neighborhood * neighborhood, axis=1)
    )
    compatibility = neighborhood @ neighborhood.T
    compatibility /= np.maximum(
        row_norm[:, None] * row_norm[None, :],
        eps,
    )
    compatibility = np.clip(compatibility, 0.0, 1.0)
    compatibility_score = 0.97 + 0.03 * compatibility

    score = (
        attraction
        * candidate_score
        * mutual_score
        * competition_score
        * regret_score
        * compatibility_score
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
