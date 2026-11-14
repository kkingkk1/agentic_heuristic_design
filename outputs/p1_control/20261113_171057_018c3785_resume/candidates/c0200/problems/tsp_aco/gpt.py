import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros_like(d, dtype=float)

    finite = np.isfinite(d) & (d >= 0)
    np.fill_diagonal(finite, False)

    valid = finite & finite.T
    pair_d = np.where(valid, 0.5 * (d + d.T), np.inf)

    if not np.any(valid):
        return np.zeros_like(d, dtype=float)

    order = np.argsort(pair_d, axis=1, kind="stable")
    ranks = np.empty_like(order)
    row_ids = np.arange(n)[:, None]
    ranks[row_ids, order] = np.arange(n)[None, :]
    ranks = np.where(valid, ranks, n)

    nearest = np.min(pair_d, axis=1)
    finite_nearest = np.isfinite(nearest) & (nearest > 0)

    positive = pair_d[valid & (pair_d > 0)]
    if np.any(finite_nearest):
        global_scale = float(np.median(nearest[finite_nearest]))
    elif positive.size:
        global_scale = float(np.median(positive))
    else:
        global_scale = 1.0

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)
    local_scale = np.where(finite_nearest, nearest, global_scale)

    safe_d = np.where(valid, pair_d, np.inf)

    global_ratio = safe_d / global_scale
    local_ratio = safe_d / np.sqrt(
        np.maximum(local_scale[:, None] * local_scale[None, :], eps)
    )

    # Sharper Gibbs affinity, clipped to avoid numerical underflow.
    global_gibbs = np.exp(-1.35 * np.minimum(global_ratio, 40.0))
    local_gibbs = np.exp(-1.05 * np.minimum(local_ratio, 40.0))

    rank_affinity = np.exp(-0.78 * np.minimum(ranks, n))
    rank_affinity = np.where(valid, rank_affinity, 0.0)

    nearest_edge = valid & (ranks == 0)
    mutual_nearest = nearest_edge & nearest_edge.T

    second = np.full(n, np.nan, dtype=float)
    if n >= 3:
        second = safe_d[np.arange(n), order[:, 1]]

    has_two = (
        np.isfinite(nearest)
        & np.isfinite(second)
        & (nearest > 0)
        & (second > 0)
    )

    regret = np.zeros(n, dtype=float)
    regret[has_two] = np.clip(
        (second[has_two] - nearest[has_two])
        / np.maximum(second[has_two], eps),
        0.0,
        1.0,
    )

    regret_bonus = nearest_edge * 0.5 * (
        regret[:, None] + regret[None, :]
    )

    # Favor edges whose endpoints have compatible local neighborhood scales.
    scale_balance = np.minimum(
        local_scale[:, None] / np.maximum(local_scale[None, :], eps),
        local_scale[None, :] / np.maximum(local_scale[:, None], eps),
    )
    scale_balance = np.where(valid, scale_balance, 0.0)

    # Reward edges that are competitive from both endpoints.
    rank_sum = ranks + ranks.T
    endpoint_competition = np.exp(-0.42 * np.minimum(rank_sum, 2 * n))
    endpoint_competition = np.where(valid, endpoint_competition, 0.0)

    candidate_k = min(5, max(1, n - 1))
    candidate = valid & (ranks <= candidate_k)
    candidate_both = candidate & candidate.T

    raw = (
        0.34 * global_gibbs
        + 0.36 * local_gibbs
        + 0.15 * rank_affinity
        + 0.10 * regret_bonus
        + 0.13 * mutual_nearest.astype(float)
        + 0.08 * scale_balance
        + 0.08 * endpoint_competition
        + 0.06 * candidate_both.astype(float)
    )

    raw = np.where(valid, raw, 0.0)
    raw = 0.5 * (raw + raw.T)

    maximum = float(np.max(raw))
    if maximum <= 0.0:
        return np.zeros_like(d, dtype=float)

    normalized = raw / maximum

    # Adaptive diversity floor based on row concentration.
    row_mass = np.sum(normalized, axis=1)
    row_peak = np.max(normalized, axis=1)
    concentration = np.divide(
        row_peak,
        np.maximum(row_mass, eps),
        out=np.zeros_like(row_peak),
        where=row_mass > 0,
    )
    floor_strength = float(np.clip(0.06 + 0.22 * np.mean(concentration), 0.06, 0.24))

    diversity_support = 0.35 * rank_affinity + 0.65 * local_gibbs
    diversity_support = np.where(valid, diversity_support, 0.0)

    score = (1.0 - floor_strength) * normalized
    score += floor_strength * diversity_support

    # Keep implausible edges available but mildly suppressed.
    score = np.where(candidate, score, 0.86 * score)
    score = np.where(valid, score, 0.0)
    score = 0.5 * (score + score.T)

    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
