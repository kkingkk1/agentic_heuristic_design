import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, locally normalized TSP edge priors."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)
    if n < 2:
        return result

    finite = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(finite, False)

    positive = d[finite & (d > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    reverse_finite = finite.T
    both = finite & reverse_finite
    only_forward = finite & ~reverse_finite
    only_backward = reverse_finite & ~finite

    dist = np.full((n, n), np.inf, dtype=float)
    dist[both] = 0.5 * (d[both] + d.T[both])
    dist[only_forward] = d[only_forward]
    dist[only_backward] = d.T[only_backward]
    np.fill_diagonal(dist, np.inf)

    valid = np.isfinite(dist)
    if not np.any(valid):
        return result

    local = np.full(n, scale, dtype=float)
    nearest = np.full(n, scale, dtype=float)
    second = np.full(n, scale, dtype=float)
    rank = np.full((n, n), float(n), dtype=float)

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size == 0:
            continue

        values = dist[i, indices]
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]

        nearest[i] = sorted_values[0]
        second[i] = sorted_values[1] if sorted_values.size > 1 else sorted_values[0]
        local[i] = float(np.mean(sorted_values[: min(4, sorted_values.size)]))
        rank[i, indices[order]] = np.arange(sorted_values.size, dtype=float)

    positive_local = local[np.isfinite(local) & (local > 0.0)]
    fallback = float(np.median(positive_local)) if positive_local.size else scale
    local = np.where(
        np.isfinite(local) & (local > 0.0),
        local,
        max(fallback, np.finfo(float).eps),
    )

    eps = np.finfo(float).eps
    pair_scale = np.sqrt(np.maximum(np.outer(local, local), eps))
    normalized = dist / pair_scale

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        smooth_short = np.exp(-np.clip(normalized, 0.0, 60.0))
        inverse_short = 1.0 / (1.0 + np.maximum(dist, 0.0) / scale)

    rank_score = 1.0 / (1.0 + rank)
    rank_consensus = np.sqrt(rank_score * rank_score.T)

    nearest_index = np.full(n, -1, dtype=int)
    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size:
            nearest_index[i] = indices[np.argmin(dist[i, indices])]

    node_ids = np.arange(n)
    reciprocal = (
        (nearest_index[:, None] == node_ids[None, :])
        & (nearest_index[None, :] == node_ids[:, None])
    ).astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        regret = np.clip((second - dist) / np.maximum(second, scale), -1.0, 1.0)
    regret = np.maximum(regret, 0.0)
    regret_bonus = np.sqrt((1.0 + regret[:, None]) * (1.0 + regret.T))

    with np.errstate(divide="ignore", invalid="ignore"):
        competition = np.minimum(
            1.0,
            np.maximum(dist, 0.0) / np.maximum(second[:, None], scale),
        )
    diversity = 0.85 + 0.15 * (1.0 - competition)

    scores = (
        (0.65 * smooth_short + 0.35 * inverse_short)
        * (0.55 + 0.45 * rank_consensus)
        * (1.0 + 0.30 * reciprocal)
        * regret_bonus
        * diversity
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    maximum = float(np.max(scores)) if scores.size else 0.0
    if maximum > 0.0:
        scores /= maximum

    return scores
