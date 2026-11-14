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

    d = 0.5 * (distances + distances.T)
    d[~valid] = np.inf
    np.fill_diagonal(d, np.inf)

    eps = np.finfo(float).eps
    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.max(finite_values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    finite_max = float(np.max(finite_values))
    penalty = max(finite_max, global_scale) * 4.0 + global_scale
    work = np.where(valid, d, penalty)
    np.fill_diagonal(work, penalty)

    k = min(4, n - 1)
    nearest = np.partition(work, kth=k - 1, axis=1)[:, :k]

    finite_nearest = np.isfinite(nearest)
    safe_nearest = np.where(finite_nearest, nearest, 0.0)
    counts = finite_nearest.sum(axis=1)

    local_scale = np.divide(
        safe_nearest.sum(axis=1),
        np.maximum(counts, 1),
        out=np.full(n, global_scale, dtype=float),
        where=counts > 0,
    )
    local_scale = np.maximum(local_scale, eps)

    ordered = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], ordered] = np.arange(n, dtype=float)

    best = work[np.arange(n), ordered[:, 0]]
    second = work[
        np.arange(n),
        ordered[:, 1] if n > 2 else ordered[:, 0],
    ]

    safe_best = np.where(np.isfinite(best), best, global_scale)
    safe_second = np.where(np.isfinite(second), second, 2.0 * safe_best)

    regret = np.clip(
        (safe_second - safe_best)
        / np.maximum(safe_second, eps),
        0.0,
        1.0,
    )
    regret_bonus = 1.0 + 0.18 * (
        regret[:, None] + regret[None, :]
    )

    geometric_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_distance = np.divide(
        work,
        geometric_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=geometric_scale > 0.0,
    )

    distance_quality = np.exp(
        -np.clip(1.02 * relative_distance, 0.0, 60.0)
    )

    row_ratio = np.divide(
        work,
        local_scale[:, None],
        out=np.full((n, n), np.inf, dtype=float),
        where=local_scale[:, None] > 0.0,
    )
    col_ratio = np.divide(
        work,
        local_scale[None, :],
        out=np.full((n, n), np.inf, dtype=float),
        where=local_scale[None, :] > 0.0,
    )

    endpoint_quality = np.exp(
        -np.clip(0.5 * (row_ratio + col_ratio), 0.0, 60.0)
    )

    rank_quality = np.exp(
        -0.5
        * (
            ranks / max(1.0, 1.05 * np.sqrt(float(n)))
            + ranks.T / max(1.0, 1.05 * np.sqrt(float(n)))
        )
    )

    nearest = ordered[:, 0]
    indices = np.arange(n)
    mutual_nearest = (
        (nearest[:, None] == indices[None, :])
        & (nearest[None, :] == indices[:, None])
    )
    mutual_bonus = np.where(mutual_nearest, 1.16, 1.0)

    long_edge_penalty = np.exp(
        -0.42 * np.maximum(relative_distance - 1.0, 0.0)
    )

    base_scores = (
        distance_quality
        * (
            0.54
            + 0.26 * endpoint_quality
            + 0.20 * rank_quality
        )
        * regret_bonus
        * mutual_bonus
        * long_edge_penalty
    )

    proximity = np.exp(
        -np.clip(relative_distance, 0.0, 40.0)
    )
    proximity[~valid] = 0.0
    np.fill_diagonal(proximity, 0.0)

    support = proximity.sum(axis=1)
    positive_support = support[support > 0.0]
    support_scale = (
        float(np.median(positive_support))
        if positive_support.size
        else 1.0
    )
    support_scale = max(support_scale, eps)

    rarity = np.clip(
        np.sqrt(support_scale / np.maximum(support, eps)),
        0.80,
        1.60,
    )
    rarity_pair = 0.5 * (
        rarity[:, None] + rarity[None, :]
    )

    profile_norm = np.sqrt(
        np.sum(proximity * proximity, axis=1)
    )
    profile_denominator = (
        profile_norm[:, None] * profile_norm[None, :]
    )
    overlap = np.divide(
        proximity @ proximity.T,
        profile_denominator,
        out=np.zeros((n, n), dtype=float),
        where=profile_denominator > 0.0,
    )
    overlap = np.clip(overlap, 0.0, 1.0)

    bridge_signal = np.clip(
        (1.0 - overlap)
        * (1.0 - np.exp(-relative_distance)),
        0.0,
        1.0,
    )

    diversity_factor = np.clip(
        1.0
        + 0.045 * (rarity_pair - 1.0)
        + 0.040 * bridge_signal
        - 0.025 * overlap,
        0.92,
        1.08,
    )

    scores = base_scores * diversity_factor
    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(scores))
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
