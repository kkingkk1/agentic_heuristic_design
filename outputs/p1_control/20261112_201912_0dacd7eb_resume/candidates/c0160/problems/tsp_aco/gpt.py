import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return endpoint-aware multi-scale edge priorities for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

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
    global_scale = (
        float(np.median(positive)) if positive.size else fallback
    )
    global_scale = max(global_scale, eps)

    work = d.copy()

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n))) + 1))

    # One partial partition supplies both local neighbors and the
    # first outside-neighborhood alternative.
    partitioned = np.partition(work, min(k, n - 1), axis=1)
    nearest = partitioned[:, :k]
    alternative = partitioned[:, min(k, n - 1)]

    local_scale = np.maximum(np.mean(nearest, axis=1), eps)
    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_length = d / (pair_scale + eps)
    geometric_affinity = np.exp(
        -np.clip(relative_length ** 0.88, 0.0, 50.0)
    )

    endpoint_ratio_i = d / (local_scale[:, None] + eps)
    endpoint_ratio_j = d / (local_scale[None, :] + eps)

    endpoint_affinity = np.exp(
        -np.clip(
            0.5 * (endpoint_ratio_i ** 0.82 + endpoint_ratio_j ** 0.82),
            0.0,
            50.0,
        )
    )

    worst_endpoint_affinity = np.exp(
        -np.clip(
            np.minimum(endpoint_ratio_i, endpoint_ratio_j) * 0.35
            + np.maximum(endpoint_ratio_i, endpoint_ratio_j) * 0.65,
            0.0,
            50.0,
        )
    )

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_i = ranks / max(float(n - 1), 1.0)
    rank_j = ranks.T / max(float(n - 1), 1.0)

    mutual_rank = np.exp(
        -1.35 * np.sqrt(np.clip(rank_i * rank_j, 0.0, None))
    )

    candidate_width = float(max(k, 1))
    candidate_i = np.exp(
        -np.clip(ranks / candidate_width, 0.0, 50.0)
    )
    candidate_j = np.exp(
        -np.clip(ranks.T / candidate_width, 0.0, 50.0)
    )
    soft_candidate_gate = 0.55 + 0.45 * np.sqrt(
        candidate_i * candidate_j
    )

    alternative_scale = np.sqrt(
        np.maximum(alternative[:, None], eps)
        * np.maximum(alternative[None, :], eps)
    )
    margin = alternative_scale / (
        d + alternative_scale + eps
    )
    margin_bonus = 1.0 + 0.10 * np.clip(margin, 0.0, 1.0)

    mutual_local = (
        (ranks < float(k))
        & (ranks.T < float(k))
    )
    mutual_bonus = np.ones((n, n), dtype=float)
    mutual_bonus += np.where(
        mutual_local,
        0.12 * np.exp(-1.8 * (rank_i + rank_j)),
        0.0,
    )

    global_affinity = np.exp(
        -np.clip((d / global_scale) ** 0.9, 0.0, 50.0)
    )
    global_affinity = 0.35 + 0.65 * global_affinity

    score = (
        geometric_affinity
        * endpoint_affinity
        * worst_endpoint_affinity
        * mutual_rank
        * soft_candidate_gate
        * margin_bonus
        * mutual_bonus
        * global_affinity
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
