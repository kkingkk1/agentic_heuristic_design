import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric local-neighborhood edge priorities for TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(distances)
    positive = distances[finite & (distances > 0.0)]

    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.percentile(positive, 90.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = float(np.max(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(finite, distances, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    work = d.copy()
    np.fill_diagonal(work, np.inf)

    eps = np.finfo(float).eps
    k = max(1, min(n - 1, int(np.rint(np.sqrt(n)))))

    neighborhood = np.partition(work, k - 1, axis=1)[:, :k]
    local_scale = np.maximum(neighborhood.mean(axis=1), eps)

    if n > 2:
        alternative = np.partition(work, 1, axis=1)[:, 1]
    else:
        alternative = neighborhood[:, 0]
    alternative = np.maximum(alternative, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_length = d / (pair_scale + eps)

    distance_affinity = np.exp(
        -np.clip(relative_length ** 0.95, 0.0, 50.0)
    )

    endpoint_i = np.exp(
        -np.clip(d / (local_scale[:, None] + eps), 0.0, 50.0)
    )
    endpoint_j = np.exp(
        -np.clip(d / (local_scale[None, :] + eps), 0.0, 50.0)
    )
    endpoint_compatibility = np.sqrt(endpoint_i * endpoint_j)

    order = np.argsort(work, axis=1, kind="mergesort")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]

    rank_score = 1.0 / np.sqrt(1.0 + ranks)
    mutual_rank = np.sqrt(rank_score * rank_score.T)

    competitiveness = 1.0 / (1.0 + relative_length)
    mutual_competitiveness = np.sqrt(
        competitiveness * competitiveness.T
    )

    substitute_scale = np.sqrt(
        alternative[:, None] * alternative[None, :]
    )
    substitute_margin = substitute_scale / (
        d + substitute_scale + eps
    )
    substitute_bonus = 1.0 + 0.08 * np.clip(
        substitute_margin,
        0.0,
        1.0,
    )

    score = (
        distance_affinity
        * endpoint_compatibility
        * mutual_rank
        * (0.82 + 0.18 * mutual_competitiveness)
        * substitute_bonus
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    off_diagonal = ~np.eye(n, dtype=bool)
    values = score[off_diagonal]

    if values.size:
        cutoff = float(np.percentile(values, 82.0))
        score *= np.where(
            (score >= cutoff) & off_diagonal,
            1.04,
            1.0,
        )

    score = 0.5 * (score + score.T)
    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
