import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return row-normalized directional edge-priority indicators for TSP."""
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

    def local_statistics(matrix: np.ndarray):
        ordered = np.sort(matrix, axis=1)

        nearest = ordered[:, 0]
        second = ordered[:, 1] if n > 2 else nearest.copy()

        nearest_safe = np.where(
            np.isfinite(nearest),
            nearest,
            global_scale,
        )
        second_safe = np.where(
            np.isfinite(second),
            second,
            2.0 * nearest_safe,
        )

        local_scale = 0.5 * (nearest_safe + second_safe)
        local_scale = np.where(
            np.isfinite(local_scale) & (local_scale > eps),
            local_scale,
            global_scale,
        )
        local_scale = np.maximum(local_scale, eps)

        gap = np.maximum(second_safe - nearest_safe, 0.0)
        scarcity = gap / np.maximum(second_safe, global_scale)
        scarcity = np.clip(scarcity, 0.0, 1.0)

        return local_scale, scarcity

    outgoing_scale, outgoing_scarcity = local_statistics(work)
    incoming_scale, incoming_scarcity = local_statistics(work.T)

    pair_scale = np.sqrt(
        outgoing_scale[:, None] * incoming_scale[None, :]
    )
    pair_scale = np.maximum(pair_scale, eps)

    relative_distance = np.divide(
        distances,
        pair_scale,
        out=np.full_like(distances, np.inf, dtype=float),
        where=valid,
    )

    distance_term = np.power(
        1.0 + np.clip(relative_distance, 0.0, 1e12),
        -2.0,
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

    endpoint_quality = np.sqrt(
        np.power(1.0 + np.clip(outgoing_ratio, 0.0, 1e12), -1.0)
        * np.power(1.0 + np.clip(incoming_ratio, 0.0, 1e12), -1.0)
    )

    row_order = np.argsort(work, axis=1, kind="stable")
    row_rank = np.full((n, n), np.inf, dtype=float)
    row_rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)[None, :]

    column_order = np.argsort(work, axis=0, kind="stable")
    column_rank = np.full((n, n), np.inf, dtype=float)
    column_rank[
        column_order,
        np.arange(n)[None, :],
    ] = np.arange(n, dtype=float)[:, None]

    rank_scale = max(1.0, np.sqrt(float(n)))
    rank_term = np.exp(
        -0.35 * (row_rank + column_rank) / rank_scale
    )
    rank_term = np.nan_to_num(
        rank_term,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    scarcity_term = 1.0 + 0.30 * (
        outgoing_scarcity[:, None]
        + incoming_scarcity[None, :]
    )

    scores = (
        distance_term
        * (0.60 + 0.40 * endpoint_quality)
        * (0.85 + 0.15 * rank_term)
        * scarcity_term
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
