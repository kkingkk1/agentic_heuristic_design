import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return row-normalized directional edge priorities for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    scores = np.zeros_like(d, dtype=float)

    if n < 2:
        return scores

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return scores

    eps = np.finfo(float).eps
    work = np.where(valid, d, np.inf)

    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    elif finite_values.size:
        global_scale = float(np.median(finite_values))
    else:
        global_scale = 1.0

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    k = min(3, n - 1)

    def endpoint_statistics(matrix: np.ndarray):
        nearest_values = np.partition(matrix, kth=k - 1, axis=1)[:, :k]
        finite_neighbors = np.isfinite(nearest_values)

        neighbor_sum = np.sum(
            np.where(finite_neighbors, nearest_values, 0.0),
            axis=1,
        )
        neighbor_count = np.sum(finite_neighbors, axis=1)

        local_scale = np.divide(
            neighbor_sum,
            neighbor_count,
            out=np.full(n, global_scale, dtype=float),
            where=neighbor_count > 0,
        )
        local_scale = np.where(
            np.isfinite(local_scale) & (local_scale > 0.0),
            local_scale,
            global_scale,
        )
        local_scale = np.maximum(local_scale, eps)

        best = nearest_values[:, 0]
        second = (
            np.partition(matrix, kth=1, axis=1)[:, 1]
            if n > 2
            else best.copy()
        )

        safe_best = np.where(np.isfinite(best), best, local_scale)
        safe_second = np.where(
            np.isfinite(second),
            second,
            2.0 * safe_best,
        )

        regret = np.divide(
            safe_second - safe_best,
            np.maximum(safe_second, eps),
        )
        regret = np.clip(regret, 0.0, 1.0)

        return local_scale, regret

    outgoing_scale, outgoing_regret = endpoint_statistics(work)
    incoming_scale, incoming_regret = endpoint_statistics(work.T)

    pair_scale = np.sqrt(
        outgoing_scale[:, None] * incoming_scale[None, :]
    )

    relative_distance = np.divide(
        d,
        pair_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid & (pair_scale > 0.0),
    )

    distance_quality = np.exp(
        -np.clip(relative_distance, 0.0, 60.0)
    )

    outgoing_ratio = np.divide(
        d,
        outgoing_scale[:, None],
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )
    incoming_ratio = np.divide(
        d,
        incoming_scale[None, :],
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )

    outgoing_quality = np.exp(
        -np.clip(outgoing_ratio, 0.0, 60.0)
    )
    incoming_quality = np.exp(
        -np.clip(incoming_ratio, 0.0, 60.0)
    )
    mutual_quality = np.sqrt(
        outgoing_quality * incoming_quality
    )

    outgoing_nearest = np.min(work, axis=1)
    incoming_nearest = np.min(work, axis=0)

    nearest_bonus = 1.0 + 0.08 * (
        (valid & (d <= outgoing_nearest[:, None])).astype(float)
        + (valid & (d <= incoming_nearest[None, :])).astype(float)
    )

    regret_bonus = 1.0 + 0.16 * (
        outgoing_regret[:, None]
        + incoming_regret[None, :]
    )

    long_edge_penalty = np.exp(
        -0.20 * np.maximum(relative_distance - 1.0, 0.0)
    )

    scores = (
        distance_quality
        * (0.65 + 0.35 * mutual_quality)
        * nearest_bonus
        * regret_bonus
        * long_edge_penalty
    )

    scores[~valid] = 0.0
    np.fill_diagonal(scores, 0.0)

    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    row_totals = scores.sum(axis=1, keepdims=True)

    return np.divide(
        scores,
        row_totals,
        out=np.zeros_like(scores),
        where=row_totals > 0.0,
    )
