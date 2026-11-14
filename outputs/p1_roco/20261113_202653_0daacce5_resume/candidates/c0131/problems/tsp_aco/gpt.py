import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally diversified edge priors for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    # Negative, NaN, and infinite distances are not valid TSP edges.
    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    clean = np.where(valid, distances, np.inf)
    positive = clean[valid & (clean > 0.0)]

    if positive.size:
        center = float(np.median(positive))
        mad = float(np.median(np.abs(positive - center)))
        scale = max(center, 1.4826 * mad, np.finfo(float).eps)
    else:
        scale = 1.0

    eps = np.sqrt(np.finfo(float).eps) * max(scale, 1.0)
    temperature = max(scale, eps)

    # Robust short-edge desirability with a mild inverse-distance component.
    scaled = np.zeros((n, n), dtype=float)
    scaled[valid] = clean[valid] / temperature

    distance_score = np.zeros((n, n), dtype=float)
    distance_score[valid] = np.exp(-np.minimum(scaled[valid], 50.0))
    inverse_score = np.zeros((n, n), dtype=float)
    inverse_score[valid] = 1.0 / (1.0 + scaled[valid])

    # Soft rank preserves alternatives that are not the absolute nearest edge.
    row_rank = np.zeros((n, n), dtype=float)
    col_rank = np.zeros((n, n), dtype=float)

    for i in range(n):
        order = np.argsort(clean[i])
        rank = 0
        for j in order:
            if valid[i, j]:
                row_rank[i, j] = rank
                rank += 1

    for j in range(n):
        order = np.argsort(clean[:, j])
        rank = 0
        for i in order:
            if valid[i, j]:
                col_rank[i, j] = rank
                rank += 1

    rank_temperature = max(1.0, 0.35 * np.sqrt(max(n - 1, 1)))
    rank_score = np.zeros((n, n), dtype=float)
    rank_score[valid] = np.exp(
        -(row_rank[valid] + col_rank[valid]) / rank_temperature
    )

    # Row and column regret: protect edges whose removal leaves a difficult endpoint.
    row_sorted = np.sort(np.where(valid, clean, np.inf), axis=1)
    col_sorted = np.sort(np.where(valid, clean, np.inf), axis=0)

    row_best = row_sorted[:, 0]
    row_second = row_sorted[:, 1] if n > 1 else row_best
    col_best = col_sorted[0, :]
    col_second = col_sorted[1, :] if n > 1 else col_best

    row_regret = np.maximum(row_second - row_best, 0.0)
    col_regret = np.maximum(col_second - col_best, 0.0)

    regret = np.zeros((n, n), dtype=float)
    regret[valid] = (
        1.0
        + 0.22 * np.tanh(row_regret[:, None][valid] / (temperature + eps))
        + 0.22 * np.tanh(col_regret[None, :][valid] / (temperature + eps))
    )

    # Two-hop bypass savings identify edges that preserve useful connectivity.
    two_hop = clean[:, :, None] + clean[None, :, :]
    two_hop[~np.isfinite(two_hop)] = np.inf

    for i in range(n):
        two_hop[i, i, :] = np.inf
        two_hop[i, :, i] = np.inf

    bypass = np.min(two_hop, axis=1)
    savings = np.maximum(bypass - clean, 0.0)

    bypass_bonus = np.ones((n, n), dtype=float)
    bypass_bonus[valid] += 0.28 * np.tanh(
        savings[valid] / (temperature + eps)
    )

    # Long-term continuation potential: after taking i -> j, favor j's
    # cheap continuation and cheap re-entry into i from a future node.
    outgoing = np.where(valid, clean, np.inf)
    incoming = np.where(valid, clean, np.inf)

    continuation_out = np.min(outgoing, axis=1)
    continuation_in = np.min(incoming, axis=0)

    future_cost = (
        clean
        + 0.45 * continuation_out[None, :]
        + 0.45 * continuation_in[:, None]
    )

    future_score = np.zeros((n, n), dtype=float)
    finite_future = valid & np.isfinite(future_cost)
    future_score[finite_future] = np.exp(
        -np.minimum(future_cost[finite_future] / (2.0 * temperature), 50.0)
    )

    # Combine complementary evidence using geometric mixing.
    raw = (
        np.power(distance_score, 0.34)
        * np.power(inverse_score, 0.18)
        * np.power(rank_score, 0.12)
        * np.power(regret, 0.12)
        * np.power(bypass_bonus, 0.10)
        * np.power(np.maximum(future_score, eps), 0.14)
    )
    raw[~valid] = 0.0

    # A few damped balancing iterations reduce endpoint monopolies while
    # retaining the original distance signal.
    balanced = raw.copy()
    positive_rows = np.sum(balanced, axis=1)
    positive_cols = np.sum(balanced, axis=0)

    row_target = (
        float(np.mean(positive_rows[positive_rows > 0.0]))
        if np.any(positive_rows > 0.0)
        else 1.0
    )
    col_target = (
        float(np.mean(positive_cols[positive_cols > 0.0]))
        if np.any(positive_cols > 0.0)
        else 1.0
    )

    for _ in range(3):
        row_mass = np.sum(balanced, axis=1)
        row_factor = np.sqrt(
            np.clip(row_target / (row_mass + eps), 0.45, 2.2)
        )
        balanced *= row_factor[:, None]

        col_mass = np.sum(balanced, axis=0)
        col_factor = np.sqrt(
            np.clip(col_target / (col_mass + eps), 0.45, 2.2)
        )
        balanced *= col_factor[None, :]
        balanced[~valid] = 0.0

    # Concave compression preserves globally useful alternatives.
    priors = np.power(np.maximum(balanced, 0.0), 0.72)
    priors[~valid] = 0.0

    maximum = float(np.max(priors))
    if np.isfinite(maximum) and maximum > 0.0:
        priors /= maximum

    priors[~np.isfinite(priors)] = 0.0
    np.fill_diagonal(priors, 0.0)
    return np.clip(priors, 0.0, 1.0)
