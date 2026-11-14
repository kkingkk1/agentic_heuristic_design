import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
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

    finite = d[np.isfinite(d)]
    if finite.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(finite))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(np.isfinite(d), d, fallback)
    np.fill_diagonal(d, np.inf)

    positive = finite[finite > 0.0]
    scale = float(np.median(positive)) if positive.size else fallback
    scale = max(scale, np.finfo(float).eps)

    k = max(2, min(n - 1, int(np.ceil(np.sqrt(n))) + 1))
    order = np.argsort(d, axis=1, kind="stable")
    nearest = order[:, :k]

    local_scale = np.mean(
        np.take_along_axis(d, nearest, axis=1),
        axis=1,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    row_rank = np.empty((n, n), dtype=float)
    row_rank[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_score = np.exp(
        -0.9
        * (
            row_rank / max(float(n - 1), 1.0)
            + row_rank.T / max(float(n - 1), 1.0)
        )
    )

    local_ratio = d / np.sqrt(local_scale[:, None] * local_scale[None, :])
    distance_score = 1.0 / (1.0 + np.power(np.maximum(local_ratio, 0.0), 1.15))

    global_score = 1.0 / (1.0 + np.power(d / scale, 0.8))

    candidate = np.zeros((n, n), dtype=float)
    candidate[np.arange(n)[:, None], nearest] = 1.0
    mutual = candidate * candidate.T
    candidate_bonus = 1.0 + 0.30 * candidate + 0.35 * mutual

    neighbor_distances = np.take_along_axis(d, nearest, axis=1)
    triangle_sum = np.zeros((n, n), dtype=float)

    for column in range(k):
        via = nearest[:, column]
        via_distances = d[:, via]
        detour = via_distances + via_distances.T - d
        triangle_sum += np.maximum(detour, 0.0)

    triangle_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    detour_score = 1.0 / (
        1.0 + triangle_sum / np.maximum(float(k) * triangle_scale, 1e-12)
    )

    score = (
        distance_score
        * rank_score
        * detour_score
        * (0.65 + 0.35 * global_score)
        * candidate_bonus
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return score
