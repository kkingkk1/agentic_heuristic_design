import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return directed row-normalized stochastic edge priorities for TSP."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    valid = np.isfinite(distances) & (distances >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    positive = distances[valid & (distances > 0.0)]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    row_scale = np.empty(n, dtype=float)
    row_low = np.empty(n, dtype=float)
    row_high = np.empty(n, dtype=float)

    for i in range(n):
        values = distances[i, valid[i]]
        if values.size == 0:
            row_scale[i] = global_scale
            row_low[i] = global_scale
            row_high[i] = global_scale
            continue

        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        robust_scale = 1.4826 * mad

        if not np.isfinite(robust_scale) or robust_scale <= eps:
            robust_scale = max(abs(median), global_scale)

        row_scale[i] = max(robust_scale, eps)
        row_low[i] = float(np.quantile(values, 0.20))
        row_high[i] = float(np.quantile(values, 0.80))

    # Estimate how cheaply each destination can continue to an unvisited node.
    continuation = np.full((n, n), global_scale, dtype=float)

    for i in range(n):
        for j in range(n):
            if not valid[i, j]:
                continue

            candidates = valid[j].copy()
            candidates[i] = False
            candidates[j] = False

            if np.any(candidates):
                next_values = distances[j, candidates]
                continuation[i, j] = float(np.min(next_values))
            else:
                continuation[i, j] = row_scale[j]

    scores = np.zeros((n, n), dtype=float)

    for i in range(n):
        usable = valid[i]
        if not np.any(usable):
            continue

        values = distances[i, usable]
        low = row_low[i]
        high = row_high[i]
        width = max(high - low, row_scale[i], eps)

        for j in np.flatnonzero(usable):
            distance = distances[i, j]

            # Smooth preference for edges below the source node's local scale.
            local_deviation = (distance - row_scale[i]) / row_scale[i]
            local_term = 1.0 / (1.0 + np.exp(np.clip(local_deviation, -40.0, 40.0)))

            # Quantile position rewards unusually short departures while
            # retaining a nonzero exploration floor for longer edges.
            quantile_position = (distance - low) / width
            quantile_term = 1.0 / (
                1.0 + np.exp(np.clip(4.0 * (quantile_position - 0.5), -40.0, 40.0))
            )

            # Prefer destinations that still have an inexpensive continuation.
            continuation_ratio = continuation[i, j] / max(row_scale[j], eps)
            continuation_term = np.exp(
                -np.clip(0.65 * continuation_ratio, 0.0, 40.0)
            )

            # A mild scale correction prevents large-coordinate instances
            # from dominating the score numerically.
            absolute_term = np.exp(
                -np.clip(0.35 * distance / global_scale, 0.0, 40.0)
            )

            scores[i, j] = (
                0.18
                + 0.34 * local_term
                + 0.24 * quantile_term
                + 0.16 * continuation_term
                + 0.08 * absolute_term
            )

    scores[~valid] = 0.0
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    totals = scores.sum(axis=1, keepdims=True)
    return np.divide(
        scores,
        totals,
        out=np.zeros_like(scores),
        where=totals > 0.0,
    )
