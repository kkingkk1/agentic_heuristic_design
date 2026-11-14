import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return directed, row-normalized edge-priority indicators for TSP."""
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

    eps = np.finfo(float).eps
    work = np.where(valid, distances, np.inf)

    finite_values = distances[valid]
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

    def endpoint_statistics(matrix: np.ndarray):
        ordered = np.sort(matrix, axis=1)

        best = ordered[:, 0]
        second = ordered[:, 1] if n > 2 else best.copy()
        third = ordered[:, 2] if n > 3 else second.copy()

        safe_best = np.where(np.isfinite(best), best, global_scale)
        safe_second = np.where(
            np.isfinite(second),
            second,
            2.0 * safe_best,
        )
        safe_third = np.where(
            np.isfinite(third),
            third,
            2.0 * safe_second,
        )

        local_scale = np.cbrt(
            np.maximum(safe_best, eps)
            * np.maximum(safe_second, eps)
            * np.maximum(safe_third, eps)
        )
        local_scale = np.where(
            np.isfinite(local_scale) & (local_scale > 0.0),
            local_scale,
            global_scale,
        )
        local_scale = np.maximum(local_scale, eps)

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
        distances,
        pair_scale,
        out=np.full_like(distances, np.inf, dtype=float),
        where=valid,
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
    mutual_quality = np.sqrt(
        outgoing_quality * incoming_quality
    )

    row_order = np.argsort(work, axis=1, kind="stable")
    row_ranks = np.full((n, n), np.inf, dtype=float)
    row_ranks[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)[None, :]

    column_order = np.argsort(work, axis=0, kind="stable")
    column_ranks = np.full((n, n), np.inf, dtype=float)
    column_ranks[
        column_order,
        np.arange(n)[None, :],
    ] = np.arange(n, dtype=float)[:, None]

    rank_scale = max(1.0, 0.9 * np.sqrt(float(n)))
    rank_quality = np.exp(
        -0.5 * (row_ranks + column_ranks) / rank_scale
    )
    rank_quality = np.nan_to_num(
        rank_quality,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    regret_bonus = 1.0 + 0.20 * (
        outgoing_regret[:, None] + incoming_regret[None, :]
    )

    long_edge_penalty = np.exp(
        -0.40 * np.maximum(relative_distance - 1.0, 0.0)
    )

    preference = (
        0.50
        + 0.32 * mutual_quality
        + 0.18 * rank_quality
    )

    scores = (
        distance_quality
        * preference
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
    ).astype(float, copy=False)
