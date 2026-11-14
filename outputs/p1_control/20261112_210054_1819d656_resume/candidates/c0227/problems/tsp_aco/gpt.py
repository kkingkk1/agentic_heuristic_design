import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return endpoint-aware candidate priors for stochastic TSP sampling."""
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

    positive = d[np.isfinite(d) & (d > 0.0)]
    global_scale = float(np.median(positive)) if positive.size else fallback
    global_scale = max(global_scale, eps)

    usable = n - 1
    k = max(1, min(usable, int(np.ceil(np.sqrt(n))) + 1))

    partitioned = np.partition(d, min(k, usable), axis=1)
    nearest = partitioned[:, :k]
    alternative = partitioned[:, min(k, usable)]

    local_scale = np.maximum(np.mean(nearest, axis=1), eps)
    local_q = np.maximum(np.median(nearest, axis=1), eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    pair_q = np.sqrt(local_q[:, None] * local_q[None, :])

    relative = d / (pair_scale + eps)
    robust_relative = d / (pair_q + 0.5 * pair_scale + eps)

    short_edge = np.exp(-np.clip(relative ** 0.78, 0.0, 45.0))
    robust_short_edge = np.exp(
        -np.clip(0.55 * robust_relative ** 0.70, 0.0, 35.0)
    )

    ratio_i = d / (local_scale[:, None] + eps)
    ratio_j = d / (local_scale[None, :] + eps)

    endpoint_fit = np.exp(
        -np.clip(
            0.5 * ratio_i ** 0.80 + 0.5 * ratio_j ** 0.80,
            0.0,
            45.0,
        )
    )

    scarcity = np.exp(
        -np.clip(
            0.35 * np.minimum(ratio_i, ratio_j) ** 0.75
            + 0.65 * np.maximum(ratio_i, ratio_j) ** 0.92,
            0.0,
            45.0,
        )
    )

    order = np.argsort(d, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)

    denominator = float(max(usable, 1))
    rank_i = ranks / denominator
    rank_j = ranks.T / denominator

    rank_agreement = np.exp(
        -1.55 * np.sqrt(np.clip(rank_i * rank_j, 0.0, None))
    )

    candidate_i = np.exp(-np.clip(ranks / float(k), 0.0, 40.0))
    candidate_j = np.exp(-np.clip(ranks.T / float(k), 0.0, 40.0))

    soft_candidate = 0.45 + 0.55 * np.sqrt(candidate_i * candidate_j)

    mutual_candidate = (
        (ranks < float(k)) & (ranks.T < float(k))
    ).astype(float)

    mutual_bonus = 1.0 + 0.16 * mutual_candidate * np.exp(
        -1.2 * (rank_i + rank_j)
    )

    alternative_scale = np.sqrt(
        np.maximum(alternative[:, None], eps)
        * np.maximum(alternative[None, :], eps)
    )

    margin = alternative_scale / (
        d + alternative_scale + eps
    )
    margin_bonus = 1.0 + 0.16 * np.clip(margin, 0.0, 1.0)

    global_fit = np.exp(
        -np.clip((d / global_scale) ** 0.82, 0.0, 40.0)
    )
    global_fit = 0.40 + 0.60 * global_fit

    score = (
        short_edge
        * robust_short_edge
        * endpoint_fit
        * scarcity
        * rank_agreement
        * soft_candidate
        * mutual_bonus
        * margin_bonus
        * global_fit
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
