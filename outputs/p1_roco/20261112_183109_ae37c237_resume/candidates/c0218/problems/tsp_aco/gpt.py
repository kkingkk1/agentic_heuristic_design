import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return row-normalized directional edge-priority indicators for TSP."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return scores

    finite_values = distances[valid]
    positive_values = finite_values[finite_values > 0.0]
    eps = np.finfo(float).eps

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    elif finite_values.size:
        global_scale = float(np.max(finite_values))
    else:
        global_scale = 1.0

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    work = np.where(valid, distances, np.inf)

    def endpoint_statistics(matrix: np.ndarray):
        """Estimate local scale and regret for each row of a directed matrix."""
        nearest = np.min(matrix, axis=1)

        if n > 1:
            two_nearest = np.partition(matrix, kth=1, axis=1)[:, :2]
            best = two_nearest[:, 0]
            second = two_nearest[:, 1]
        else:
            best = nearest
            second = nearest

        safe_best = np.where(
            np.isfinite(best),
            best,
            global_scale,
        )
        safe_best = np.maximum(safe_best, eps)

        safe_second = np.where(
            np.isfinite(second),
            second,
            2.0 * safe_best,
        )
        safe_second = np.maximum(safe_second, safe_best)

        local_scale = 0.5 * (safe_best + safe_second)
        local_scale = np.maximum(local_scale, eps)

        regret = np.divide(
            safe_second - safe_best,
            np.maximum(safe_second, eps),
        )
        regret = np.clip(regret, 0.0, 1.0)

        return local_scale, regret, np.argmin(matrix, axis=1)

    outgoing_scale, outgoing_regret, outgoing_nearest = endpoint_statistics(work)
    incoming_scale, incoming_regret, incoming_nearest = endpoint_statistics(work.T)

    pair_scale = np.sqrt(
        outgoing_scale[:, None] * incoming_scale[None, :]
    )

    relative_distance = np.divide(
        distances,
        pair_scale,
        out=np.full_like(distances, np.inf, dtype=float),
        where=valid & (pair_scale > 0.0),
    )

    distance_quality = np.exp(
        -np.clip(relative_distance, 0.0, 60.0)
    )

    outgoing_ratio = np.divide(
        distances,
        outgoing_scale[:, None],
        out=np.full_like(distances, np.inf, dtype=float),
        where=valid,
    )
    incoming_ratio = np.divide(
        distances,
        incoming_scale[None, :],
        out=np.full_like(distances, np.inf, dtype=float),
        where=valid,
    )

    outgoing_quality = np.exp(
        -np.clip(outgoing_ratio, 0.0, 60.0)
    )
    incoming_quality = np.exp(
        -np.clip(incoming_ratio, 0.0, 60.0)
    )
    mutual_quality = np.sqrt(outgoing_quality * incoming_quality)

    continuous_neighbor_quality = np.exp(
        -0.5 * np.clip(outgoing_ratio + incoming_ratio, 0.0, 120.0)
    )

    regret_bonus = 1.0 + 0.22 * (
        outgoing_regret[:, None] + incoming_regret[None, :]
    )

    indices = np.arange(n)
    mutual_nearest = (
        (outgoing_nearest[:, None] == indices[None, :])
        & (incoming_nearest[None, :] == indices[:, None])
        & valid
    )
    nearest_bonus = np.where(mutual_nearest, 1.12, 1.0)

    long_edge_penalty = np.exp(
        -0.28 * np.maximum(relative_distance - 1.0, 0.0)
    )

    scores = (
        distance_quality
        * (0.55 + 0.25 * mutual_quality + 0.20 * continuous_neighbor_quality)
        * regret_bonus
        * nearest_bonus
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
    ).astype(float, copy=False)
