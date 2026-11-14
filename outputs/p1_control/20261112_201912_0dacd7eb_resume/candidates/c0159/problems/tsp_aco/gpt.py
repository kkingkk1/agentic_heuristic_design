import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return endpoint-aware edge-priority indicators for stochastic TSP sampling."""
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

    diagonal = np.eye(n, dtype=bool)
    finite_offdiag = d[~diagonal & np.isfinite(d)]

    if finite_offdiag.size == 0:
        return np.zeros((n, n), dtype=float)

    positive = finite_offdiag[finite_offdiag > 0.0]
    fallback = (
        float(np.max(positive))
        if positive.size
        else max(float(np.max(finite_offdiag)), 1.0)
    )
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(np.isfinite(d), d, fallback)
    np.fill_diagonal(d, np.inf)

    eps = np.finfo(float).eps
    positive = finite_offdiag[finite_offdiag > 0.0]
    global_scale = (
        float(np.median(positive))
        if positive.size
        else fallback
    )
    global_scale = max(global_scale, eps)

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n))) + 1))
    ordered = np.sort(d, axis=1)
    neighbors = ordered[:, :k]

    local_scale = np.mean(neighbors, axis=1)
    local_scale = np.maximum(local_scale, eps)

    alternative = ordered[:, k - 1]
    alternative = np.maximum(alternative, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_distance = d / (pair_scale + eps)

    relative_affinity = np.exp(
        -np.clip(relative_distance ** 0.85, 0.0, 50.0)
    )

    endpoint_ratio_i = d / local_scale[:, None]
    endpoint_ratio_j = d / local_scale[None, :]

    endpoint_affinity_i = np.exp(
        -np.clip(endpoint_ratio_i ** 0.8, 0.0, 50.0)
    )
    endpoint_affinity_j = np.exp(
        -np.clip(endpoint_ratio_j ** 0.8, 0.0, 50.0)
    )

    worst_endpoint_affinity = np.minimum(
        endpoint_affinity_i,
        endpoint_affinity_j,
    )
    worst_endpoint_affinity = (
        0.35 + 0.65 * worst_endpoint_affinity
    ) ** 1.15

    local_mismatch = np.abs(
        np.log1p(endpoint_ratio_i) - np.log1p(endpoint_ratio_j)
    )
    compatibility = np.exp(
        -np.clip(
            0.7 * (
                np.log1p(endpoint_ratio_i)
                + np.log1p(endpoint_ratio_j)
            )
            + 0.4 * local_mismatch,
            0.0,
            50.0,
        )
    )

    row_order = np.argsort(d, axis=1, kind="stable")
    ranks = np.empty_like(row_order, dtype=float)
    ranks[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)

    normalized_ranks = ranks / max(float(n - 1), 1.0)
    rank_sum = normalized_ranks + normalized_ranks.T

    rank_term = 1.0 / (1.0 + 0.75 * rank_sum)

    mutual_limit = min(n - 1, 2 * k)
    mutual_top = (
        (ranks < float(mutual_limit))
        & (ranks.T < float(mutual_limit))
    )

    mutual_bonus = np.ones((n, n), dtype=float)
    mutual_bonus[mutual_top] += 0.15 * np.exp(
        -2.2 * (
            normalized_ranks[mutual_top]
            + normalized_ranks.T[mutual_top]
        )
    )

    candidate_support = np.sqrt(
        np.exp(-np.clip(ranks / max(float(k), 1.0), 0.0, 50.0))
        * np.exp(-np.clip(ranks.T / max(float(k), 1.0), 0.0, 50.0))
    )
    soft_candidate_gate = 0.55 + 0.45 * candidate_support

    substitute_scale = np.sqrt(
        alternative[:, None] * alternative[None, :]
    )
    margin = substitute_scale / (d + substitute_scale + eps)
    margin_bonus = 0.9 + 0.1 * margin

    global_affinity = np.exp(
        -np.clip((d / global_scale) ** 0.9, 0.0, 50.0)
    )
    global_term = 0.35 + 0.65 * global_affinity

    score = (
        relative_affinity
        * worst_endpoint_affinity
        * rank_term
        * compatibility
        * soft_candidate_gate
        * mutual_bonus
        * margin_bonus
        * global_term
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
