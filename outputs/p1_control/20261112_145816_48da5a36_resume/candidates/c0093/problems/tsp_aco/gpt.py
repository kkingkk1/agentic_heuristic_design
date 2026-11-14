import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric TSP edge-priority matrix."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    finite = np.isfinite(d)
    nonnegative = d[finite & (d >= 0)]

    if nonnegative.size == 0:
        return result

    fallback = float(np.median(nonnegative))
    fallback = max(fallback, np.finfo(float).eps)

    valid = finite & (d > 0)
    d_clean = np.where(valid, d, fallback)
    d_clean = 0.5 * (d_clean + d_clean.T)

    valid = valid & valid.T
    np.fill_diagonal(valid, False)
    np.fill_diagonal(d_clean, fallback)

    positive = d_clean[valid]
    if positive.size == 0:
        return result

    scale = max(float(np.median(positive)), np.finfo(float).eps)
    eps = np.finfo(float).eps

    row_sorted = np.sort(
        np.where(valid, d_clean, np.inf),
        axis=1,
    )

    first = np.where(np.isfinite(row_sorted[:, 0]), row_sorted[:, 0], scale)
    second = np.where(
        np.isfinite(row_sorted[:, 1]),
        row_sorted[:, 1],
        first,
    )

    endpoint_baseline = np.sqrt(
        np.maximum(first[:, None], eps)
        * np.maximum(first[None, :], eps)
    )

    direct = np.maximum(d_clean, eps)

    if n > 2:
        two_hop = d_clean[:, :, None] + d_clean[None, :, :]
        allowed = np.ones((n, n, n), dtype=bool)
        indices = np.arange(n)
        allowed[indices, :, indices] = False
        allowed[:, indices, indices] = False
        two_hop[~allowed] = np.inf
        best_detour = np.min(two_hop, axis=2)
        best_detour[~np.isfinite(best_detour)] = 2.0 * scale
    else:
        best_detour = np.full((n, n), 2.0 * scale, dtype=float)

    shortcut_ratio = best_detour / np.maximum(direct, eps)
    shortcut_gain = np.tanh(
        np.maximum(shortcut_ratio - 1.0, 0.0)
    )

    row_scale = np.maximum(second, eps)
    profiles = d_clean / row_scale[:, None]
    profiles[~np.isfinite(profiles)] = 10.0

    profile_center = np.mean(profiles, axis=1, keepdims=True)
    centered = profiles - profile_center
    profile_norm = np.linalg.norm(centered, axis=1)

    profile_similarity = centered @ centered.T
    profile_similarity /= (
        profile_norm[:, None] * profile_norm[None, :] + eps
    )
    profile_similarity = np.clip(
        0.5 * (profile_similarity + 1.0),
        0.0,
        1.0,
    )

    relative_cost = direct / np.maximum(endpoint_baseline, eps)
    direct_preference = np.exp(-np.minimum(relative_cost, 40.0))

    rank_order = np.argsort(
        np.where(valid, d_clean, np.inf),
        axis=1,
        kind="stable",
    )
    ranks = np.empty_like(rank_order, dtype=float)
    ranks[np.arange(n)[:, None], rank_order] = np.arange(n, dtype=float)
    ranks = ranks / max(n - 1, 1)

    mutual_preference = np.exp(
        -2.0 * (ranks + ranks.T)
    )

    endpoint_load = np.sum(
        np.exp(-np.minimum(d_clean / row_scale[:, None], 40.0)) * valid,
        axis=1,
    )
    competition_penalty = 1.0 / np.sqrt(
        (1.0 + endpoint_load[:, None])
        * (1.0 + endpoint_load[None, :])
    )

    score = (
        (0.30 + 0.70 * shortcut_gain)
        * (0.35 + 0.65 * profile_similarity)
        * direct_preference
        * (0.30 + 0.70 * mutual_preference)
        * competition_penalty
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
