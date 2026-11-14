import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return bounded symmetric edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    directed_valid = (
        np.isfinite(raw)
        & (raw >= 0.0)
        & ~diagonal
    )
    valid = directed_valid & directed_valid.T

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    distances = np.full((n, n), np.inf, dtype=float)
    distances[valid] = 0.5 * (
        raw[valid] + raw.T[valid]
    )
    np.fill_diagonal(distances, np.inf)

    values = distances[valid]
    positive = values[values > 0.0]

    if positive.size:
        q25, q50, q75 = np.quantile(
            positive,
            [0.25, 0.50, 0.75],
        )
        scale = float(q75 - q25)
        if not np.isfinite(scale) or scale <= 0.0:
            scale = float(q50)
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps
    scale = max(scale, eps)

    finite_max = float(np.max(values))
    replacement = finite_max + 8.0 * scale
    filled = np.where(valid, distances, replacement)
    np.fill_diagonal(filled, replacement)

    positive_distance = np.maximum(filled, 0.0)

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (
        positive_distance[valid] + 1e-9 * scale
    )
    inverse_max = float(np.max(inverse))
    if inverse_max > 0.0 and np.isfinite(inverse_max):
        inverse /= inverse_max

    row_order = np.argsort(
        filled,
        axis=1,
        kind="stable",
    )
    row_rank = np.empty((n, n), dtype=float)
    row_rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)

    col_order = np.argsort(
        filled,
        axis=0,
        kind="stable",
    )
    col_rank = np.empty((n, n), dtype=float)
    col_rank[
        col_order,
        np.arange(n)[None, :],
    ] = np.arange(n, dtype=float)

    rank_scale = max(1.0, 0.75 * np.sqrt(float(n)))
    row_rank_signal = np.exp(-row_rank / rank_scale)
    col_rank_signal = np.exp(-col_rank / rank_scale)

    rank_signal = np.sqrt(
        row_rank_signal * col_rank_signal
    )
    rank_signal[~valid] = 0.0

    row_mass = np.zeros(n, dtype=float)
    col_mass = np.zeros(n, dtype=float)
    row_mass[:] = np.sum(
        np.where(
            valid,
            1.0 / (1.0 + filled / scale),
            0.0,
        ),
        axis=1,
    )
    col_mass[:] = np.sum(
        np.where(
            valid,
            1.0 / (1.0 + filled / scale),
            0.0,
        ),
        axis=0,
    )

    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = 1.0 / (
        1.0 + filled[valid] / scale
    )

    row_preference = affinity / np.maximum(
        row_mass[:, None],
        eps,
    )
    col_preference = affinity / np.maximum(
        col_mass[None, :],
        eps,
    )

    preference = np.sqrt(
        np.maximum(
            row_preference * col_preference,
            0.0,
        )
    )
    preference[~valid] = 0.0

    row_sorted = np.sort(
        np.where(valid, filled, np.inf),
        axis=1,
    )
    col_sorted = np.sort(
        np.where(valid, filled, np.inf),
        axis=0,
    )

    row_best = row_sorted[:, 0]
    col_best = col_sorted[0, :]
    row_second = row_sorted[:, min(1, n - 1)]
    col_second = col_sorted[min(1, n - 1), :]

    row_gap = np.maximum(
        row_second - row_best,
        0.10 * scale,
    )
    col_gap = np.maximum(
        col_second - col_best,
        0.10 * scale,
    )

    row_competition = 1.0 - np.exp(
        -np.maximum(
            row_second[:, None] - filled,
            0.0,
        ) / row_gap[:, None]
    )
    col_competition = 1.0 - np.exp(
        -np.maximum(
            col_second[None, :] - filled,
            0.0,
        ) / col_gap[None, :]
    )

    competition = np.sqrt(
        np.maximum(
            row_competition * col_competition,
            0.0,
        )
    )
    competition[~valid] = 0.0

    k = min(3, max(1, n - 1))
    row_top = row_rank < float(k)
    col_top = col_rank < float(k)

    endpoint_scarcity = (
        0.5 * row_top.astype(float)
        + 0.5 * col_top.astype(float)
    )
    endpoint_scarcity[~valid] = 0.0

    neighbor_overlap = np.zeros((n, n), dtype=float)
    for i in range(n):
        neighbors_i = row_top[i] & valid[i]
        for j in np.flatnonzero(valid[i]):
            neighbors_j = row_top[j] & valid[j]
            union_size = np.count_nonzero(
                neighbors_i | neighbors_j
            )
            if union_size:
                neighbor_overlap[i, j] = (
                    np.count_nonzero(
                        neighbors_i & neighbors_j
                    )
                    / float(union_size)
                )

    diversity = 1.0 - neighbor_overlap
    diversity[~valid] = 0.0

    base = (
        0.58 * inverse
        + 0.42 * rank_signal
    )
    base *= 0.84 + 0.16 * preference

    bounded_bonus = (
        0.10 * competition
        + 0.07 * endpoint_scarcity
        + 0.05 * diversity
    )
    bounded_bonus = np.clip(
        bounded_bonus,
        0.0,
        0.20,
    )

    score = base * (1.0 + bounded_bonus)
    score = np.maximum(score, 0.035 * valid.astype(float))
    score[~valid] = 0.0

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if not np.isfinite(maximum) or maximum <= 0.0:
        return valid.astype(float)

    result = score / maximum
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
