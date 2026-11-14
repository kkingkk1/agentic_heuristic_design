import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, scale-invariant edge-priority indicators for TSP."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    finite = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(finite, False)

    valid = finite & finite.T
    if not np.any(valid):
        return scores

    symmetric_distance = 0.5 * (distances + distances.T)
    finite_values = symmetric_distance[valid]

    positive = finite_values[finite_values > 0.0]
    scale = float(np.median(positive)) if positive.size else float(
        np.max(finite_values)
    )
    eps = np.finfo(float).eps
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    scale = max(scale, eps)

    maximum = float(np.max(finite_values))
    sentinel = max(maximum, scale) * 8.0 + scale

    work = np.where(valid, symmetric_distance, sentinel)
    np.fill_diagonal(work, sentinel)

    order = np.argsort(work, axis=1, kind="stable")
    sorted_work = np.take_along_axis(work, order, axis=1)

    k1 = sorted_work[:, 0]
    k2 = sorted_work[:, 1] if n > 2 else k1
    k3 = sorted_work[:, 2] if n > 3 else k2

    local_scale = 0.50 * k1 + 0.30 * k2 + 0.20 * k3
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        scale,
    )
    local_scale = np.maximum(local_scale, eps)

    normalized = np.divide(
        work,
        np.sqrt(local_scale[:, None] * local_scale[None, :]),
        out=np.full((n, n), np.inf, dtype=float),
        where=local_scale[:, None] > 0.0,
    )

    distance_kernel = np.exp(-np.clip(normalized, 0.0, 60.0))

    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    temperature = max(1.0, np.sqrt(float(n)))
    rank_affinity = np.exp(-ranks / temperature)
    rank_affinity = np.sqrt(rank_affinity * rank_affinity.T)

    candidate_width = min(max(3, int(np.ceil(np.sqrt(float(n))))), n - 1)
    candidate_mask = ranks < float(candidate_width)

    candidate_support = candidate_mask.astype(float)
    candidate_support = 0.5 * (
        candidate_support + candidate_support.T
    )

    nearest = order[:, 0]
    node_ids = np.arange(n)
    mutual_nearest = (
        (nearest[:, None] == node_ids[None, :])
        & (nearest[None, :] == node_ids[:, None])
    )

    regret = np.divide(
        np.maximum(k2 - k1, 0.0),
        np.maximum(k2, eps),
    )
    regret = np.clip(regret, 0.0, 1.0)
    regret_bonus = 1.0 + 0.20 * (
        regret[:, None] + regret[None, :]
    )

    local_penalty = np.exp(
        -0.45 * np.maximum(normalized - 1.0, 0.0)
    )

    scores = (
        distance_kernel
        * (0.45 + 0.35 * rank_affinity + 0.20 * candidate_support)
        * (0.85 + 0.15 * local_penalty)
        * regret_bonus
        * np.where(mutual_nearest, 1.25, 1.0)
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum_score = float(np.max(scores))
    if np.isfinite(maximum_score) and maximum_score > 0.0:
        scores /= maximum_score

    return np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
