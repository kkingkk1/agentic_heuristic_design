import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return robust soft edge priors for stochastic TSP solution sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    positive = distances[valid & (distances > 0.0)]

    if positive.size:
        q05, q50, q95 = np.percentile(positive, [5.0, 50.0, 95.0])
        lower = max(float(q05), float(q50) * 1e-6, eps)
        upper = max(float(q95), lower)
        scale = max(float(q50), lower)
    else:
        lower = 1.0
        upper = 1.0
        scale = 1.0

    clipped = np.full((n, n), upper, dtype=float)
    positive_mask = valid & (distances > 0.0)
    clipped[positive_mask] = np.clip(
        distances[positive_mask],
        lower,
        upper,
    )
    clipped[valid & (distances == 0.0)] = lower

    normalized = clipped / max(scale, eps)

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (1.0 + normalized[valid])

    row_rank = np.zeros((n, n), dtype=float)
    col_rank = np.zeros((n, n), dtype=float)

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size == 0:
            continue

        order = indices[
            np.argsort(clipped[i, indices], kind="mergesort")
        ]

        if order.size == 1:
            row_rank[i, order[0]] = 1.0
        else:
            positions = np.arange(order.size, dtype=float)
            row_rank[i, order] = 1.0 - positions / (order.size - 1.0)

    for j in range(n):
        indices = np.flatnonzero(valid[:, j])
        if indices.size == 0:
            continue

        order = indices[
            np.argsort(clipped[indices, j], kind="mergesort")
        ]

        if order.size == 1:
            col_rank[order[0], j] = 1.0
        else:
            positions = np.arange(order.size, dtype=float)
            col_rank[order, j] = 1.0 - positions / (order.size - 1.0)

    row_best = np.max(np.where(valid, inverse, 0.0), axis=1, keepdims=True)
    col_best = np.max(np.where(valid, inverse, 0.0), axis=0, keepdims=True)

    row_competitiveness = np.divide(
        inverse,
        row_best,
        out=np.zeros_like(inverse),
        where=row_best > 0.0,
    )
    col_competitiveness = np.divide(
        inverse,
        col_best,
        out=np.zeros_like(inverse),
        where=col_best > 0.0,
    )

    reciprocal = valid & (row_rank >= 0.999999) & (col_rank >= 0.999999)

    row_degree = np.sum(valid, axis=1).astype(float)
    col_degree = np.sum(valid, axis=0).astype(float)

    row_top = np.zeros(n, dtype=float)
    col_top = np.zeros(n, dtype=float)

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size:
            take = min(5, indices.size)
            values = inverse[i, indices]
            row_top[i] = float(np.mean(np.sort(values)[-take:]))

        indices = np.flatnonzero(valid[:, i])
        if indices.size:
            take = min(5, indices.size)
            values = inverse[indices, i]
            col_top[i] = float(np.mean(np.sort(values)[-take:]))

    row_mass = np.sum(inverse, axis=1)
    col_mass = np.sum(inverse, axis=0)

    row_option = row_top * np.sqrt(np.maximum(row_degree, 1.0))
    col_option = col_top * np.sqrt(np.maximum(col_degree, 1.0))

    row_option = np.divide(
        row_option,
        np.maximum(row_mass, eps),
    )
    col_option = np.divide(
        col_option,
        np.maximum(col_mass, eps),
    )

    future_bridge = np.sqrt(
        col_option[:, None] * row_option[None, :]
    )
    future_bridge[~valid] = 0.0

    global_reference = float(np.percentile(inverse[valid], 90.0))
    global_reference = max(global_reference, eps)
    global_quality = np.clip(inverse / global_reference, 0.0, 1.0)

    priors = (
        0.53 * inverse
        + 0.14 * row_rank
        + 0.12 * col_rank
        + 0.08 * row_competitiveness
        + 0.07 * col_competitiveness
        + 0.03 * global_quality
        + 0.03 * future_bridge
    )

    priors[reciprocal] *= 1.16

    candidate_mask = np.zeros((n, n), dtype=bool)

    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        degree = candidates.size

        if degree == 0:
            continue

        candidate_count = min(
            degree,
            max(3, int(np.ceil(np.sqrt(max(n - 1, 1))))),
        )

        order = candidates[
            np.argsort(-priors[i, candidates], kind="mergesort")
        ]

        candidate_mask[i, order[:candidate_count]] = True

    priors *= np.where(candidate_mask, 1.0, 0.12)
    priors[~valid] = 0.0
    np.fill_diagonal(priors, 0.0)

    maximum = float(np.max(priors))
    if not np.isfinite(maximum) or maximum <= 0.0:
        return result

    result = priors / maximum
    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
