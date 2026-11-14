import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return robust edge-selection priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    offdiag = ~np.eye(n, dtype=bool)
    finite = np.isfinite(raw)
    valid = finite & offdiag & finite.T

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    distances = np.maximum(0.5 * (raw + raw.T), 0.0)
    observed = distances[valid]

    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    upper_quantile = float(np.quantile(observed, 0.90))
    lower_quantile = float(np.quantile(observed, 0.10))
    robust_span = max(upper_quantile - lower_quantile, 0.25 * scale)
    cap = upper_quantile + 2.0 * robust_span
    penalty = cap + 4.0 * scale

    cleaned = np.where(valid, distances, penalty)
    np.fill_diagonal(cleaned, penalty)
    cleaned = np.minimum(cleaned, cap)

    valid_count = valid.sum(axis=1)
    rank = np.full((n, n), n, dtype=float)

    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size:
            order = candidates[
                np.argsort(cleaned[i, candidates], kind="stable")
            ]
            rank[i, order] = np.arange(order.size, dtype=float)

    denominator = max(n - 1, 1)
    local_rank = np.clip(1.0 - rank / denominator, 0.0, 1.0)
    bidirectional_rank = np.sqrt(local_rank * local_rank.T)
    bidirectional_rank[~valid] = 0.0

    reciprocal = 1.0 / (1.0 + cleaned / scale)
    reciprocal[~valid] = 0.0
    reciprocal_max = float(np.max(reciprocal))
    if reciprocal_max > 0.0:
        reciprocal /= reciprocal_max

    competition = np.zeros((n, n), dtype=float)
    gap_values = []

    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size >= 2:
            ordered = candidates[
                np.argsort(cleaned[i, candidates], kind="stable")
            ]
            gap_values.append(
                cleaned[i, ordered[1]] - cleaned[i, ordered[0]]
            )

    gap_scale = float(np.median(np.asarray(gap_values)))
    if not np.isfinite(gap_scale) or gap_scale <= 0.0:
        gap_scale = scale

    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if not candidates.size:
            continue

        ordered = candidates[
            np.argsort(cleaned[i, candidates], kind="stable")
        ]
        best = cleaned[i, ordered[0]]
        second = cleaned[i, ordered[1]] if ordered.size > 1 else best

        for j in candidates:
            if j == ordered[0]:
                gap = max(second - cleaned[i, j], 0.0)
                competition[i, j] = 1.0 - np.exp(-gap / gap_scale)

    competition = np.sqrt(competition * competition.T)
    competition[~valid] = 0.0

    best_bypass = np.full((n, n), penalty, dtype=float)

    for k in range(n):
        left_valid = valid[:, k]
        right_valid = valid[k, :]
        candidate = cleaned[:, k, None] + cleaned[None, k, :]
        candidate_valid = left_valid[:, None] & right_valid[None, :]
        candidate = np.where(candidate_valid, candidate, penalty)
        best_bypass = np.minimum(best_bypass, candidate)

    np.fill_diagonal(best_bypass, penalty)

    bypass_gain = np.maximum(best_bypass - cleaned, 0.0)
    detour_scale = max(scale, np.finfo(float).eps)
    detour = 1.0 - np.exp(-bypass_gain / detour_scale)
    detour[~valid] = 0.0

    score = (
        0.46 * reciprocal
        + 0.34 * bidirectional_rank
        + 0.10 * competition
        + 0.10 * detour
    )
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)

    maximum = float(np.max(score))
    if not np.isfinite(maximum) or maximum <= 0.0:
        result = valid.astype(float)
    else:
        score /= maximum
        exploration_floor = 0.03
        result = np.where(
            valid,
            exploration_floor + (1.0 - exploration_floor) * score,
            0.0,
        )

    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    return np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)
