import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally diverse, long-horizon edge priors for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    valid = np.isfinite(distances)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    values = distances[valid]
    finite_min = float(np.min(values))
    shifted = np.maximum(values - finite_min, 0.0)

    median = float(np.median(shifted))
    q25, q75 = np.percentile(shifted, [25.0, 75.0])
    iqr = max(float(q75 - q25), np.finfo(float).eps)
    robust_scale = max(median, iqr, np.finfo(float).eps)

    # Bounded replacement is used only for statistics; invalid edges remain zero.
    upper_bound = float(q75 + 4.0 * robust_scale)
    upper_bound = max(upper_bound, float(np.max(shifted)))
    costs = np.full((n, n), upper_bound, dtype=float)
    costs[valid] = shifted

    eps = np.finfo(float).eps * max(1.0, robust_scale)
    normalized = costs / (robust_scale + eps)

    attraction = 1.0 / (1.0 + normalized)
    attraction[~valid] = 0.0

    global_scale = max(float(np.percentile(shifted, 75.0)), robust_scale)
    global_potential = np.exp(-costs / (global_scale + eps))
    global_potential[~valid] = 0.0

    # Rank-based signal is less sensitive to extreme distance magnitudes.
    rank_signal = np.zeros((n, n), dtype=float)
    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size == 0:
            continue

        order = indices[np.argsort(costs[i, indices], kind="mergesort")]
        count = order.size

        if count == 1:
            rank_signal[i, order[0]] = 1.0
        else:
            rank_signal[i, order] = 1.0 - (
                np.arange(count, dtype=float) / float(count - 1)
            )

    # Endpoint continuation quality estimates long-term feasibility after using an edge.
    continuation = np.zeros(n, dtype=float)
    for i in range(n):
        row_values = attraction[i, valid[i]]
        if row_values.size:
            take = min(4, row_values.size)
            continuation[i] = float(
                np.mean(np.sort(row_values)[-take:])
            )

    continuation_scale = max(float(np.max(continuation)), eps)
    endpoint_potential = np.sqrt(
        np.outer(continuation, continuation)
    ) / continuation_scale
    endpoint_potential[~valid] = 0.0

    base = (
        0.35 * attraction
        + 0.25 * rank_signal
        + 0.20 * global_potential
        + 0.20 * endpoint_potential
    )
    base[~valid] = 0.0

    # Bidirectional competition discourages concentration on a few destination hubs.
    row_mass = np.sum(base, axis=1, keepdims=True)
    col_mass = np.sum(base, axis=0, keepdims=True)
    competition = base / np.sqrt((row_mass + eps) * (col_mass + eps))

    score = 0.72 * base + 0.28 * competition
    score[~valid] = 0.0

    # Keep a scale-adaptive candidate set while retaining graded stochastic priors.
    candidate_count = min(n - 1, max(3, int(np.ceil(np.sqrt(n)))))
    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size == 0:
            continue

        if candidates.size > candidate_count:
            order = candidates[
                np.argsort(-score[i, candidates], kind="mergesort")
            ]
            keep = order[:candidate_count]
            mask = np.zeros(n, dtype=bool)
            mask[keep] = True
            score[i, ~mask] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        result = score / maximum

    np.fill_diagonal(result, 0.0)
    result[~valid] = 0.0
    return result
