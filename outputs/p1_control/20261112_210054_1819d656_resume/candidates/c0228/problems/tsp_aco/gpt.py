import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return endpoint-aware edge priorities for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_positive = distances[
        np.isfinite(distances) & (distances > 0.0)
    ]

    if finite_positive.size:
        fallback = float(np.percentile(finite_positive, 90.0))
        global_scale = float(np.median(finite_positive))
        global_upper = float(np.percentile(finite_positive, 75.0))
    else:
        fallback = 1.0
        global_scale = 1.0
        global_upper = 1.0

    eps = np.finfo(float).eps
    fallback = max(fallback, eps)
    global_scale = max(global_scale, eps)
    global_upper = max(global_upper, global_scale)

    d = np.nan_to_num(
        distances,
        nan=fallback,
        posinf=fallback,
        neginf=0.0,
    )
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    work = d.copy()
    np.fill_diagonal(work, np.inf)

    k = max(1, min(n - 1, int(np.ceil(1.5 * np.sqrt(n)))))

    nearest = np.partition(work, k - 1, axis=1)[:, :k]
    nearest.sort(axis=1)

    rank_weights = np.linspace(1.0, 0.45, k, dtype=float)
    rank_weights /= rank_weights.sum()

    weighted_local = np.sum(nearest * rank_weights[None, :], axis=1)
    median_local = np.median(nearest, axis=1)

    local_scale = np.sqrt(
        np.maximum(weighted_local, eps)
        * np.maximum(median_local, eps)
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_distance = d / (pair_scale + eps)
    local_score = np.exp(
        -np.clip(relative_distance ** 0.78, 0.0, 50.0)
    )

    order = np.argsort(work, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]

    normalized_rank = ranks / max(float(n - 1), 1.0)
    rank_score = 1.0 / (1.0 + 1.8 * normalized_rank)

    mutual_rank = np.sqrt(rank_score * rank_score.T)

    mutual_candidate = (
        (normalized_rank <= 0.35)
        & (normalized_rank.T <= 0.35)
    )
    candidate_bonus = np.where(mutual_candidate, 1.16, 1.0)

    endpoint_score_i = np.exp(
        -np.clip(
            (d / (local_scale[:, None] + eps)) ** 0.72,
            0.0,
            50.0,
        )
    )
    endpoint_score_j = np.exp(
        -np.clip(
            (d / (local_scale[None, :] + eps)) ** 0.72,
            0.0,
            50.0,
        )
    )
    endpoint_score = np.sqrt(endpoint_score_i * endpoint_score_j)

    if k > 1:
        alternative = nearest[:, 1]
    else:
        alternative = nearest[:, 0]

    substitute_scale = np.sqrt(
        np.maximum(alternative[:, None], eps)
        * np.maximum(alternative[None, :], eps)
    )

    regret = substitute_scale / (
        d + substitute_scale + eps
    )
    regret = np.clip(regret, 0.0, 1.0)

    global_score = np.exp(
        -np.clip((d / global_scale) ** 0.68, 0.0, 50.0)
    )
    global_score = 0.30 + 0.70 * global_score

    upper_tail_penalty = 1.0 / (
        1.0
        + 0.16
        * np.maximum(d / global_upper - 1.0, 0.0) ** 1.35
    )

    score = (
        local_score ** 1.30
        * mutual_rank ** 1.15
        * candidate_bonus
        * endpoint_score ** 1.10
        * global_score
        * upper_tail_penalty
        * (0.90 + 0.18 * regret)
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
