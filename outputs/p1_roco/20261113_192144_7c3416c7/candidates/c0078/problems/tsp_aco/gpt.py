import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return robust, balanced, softly sparse priors for stochastic TSP sampling."""
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

    positive = distances[valid & (distances > 0.0)]
    eps = np.finfo(float).eps

    if positive.size:
        scale = max(float(np.median(positive)), eps)
        lower = max(float(np.percentile(positive, 5.0)), scale * 1e-6, eps)
        upper = max(float(np.percentile(positive, 95.0)), lower)
    else:
        scale = 1.0
        lower = eps
        upper = 1.0

    safe = np.full((n, n), upper, dtype=float)
    safe[valid] = np.where(distances[valid] > 0.0, distances[valid], lower)
    safe = np.clip(safe, lower, upper)

    normalized = safe / scale
    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (normalized[valid] + 1e-12)

    global_best = float(np.max(inverse))
    global_quality = inverse / global_best if global_best > 0.0 else inverse

    row_best = np.max(np.where(valid, inverse, 0.0), axis=1, keepdims=True)
    col_best = np.max(np.where(valid, inverse, 0.0), axis=0, keepdims=True)

    row_competition = np.divide(
        inverse,
        row_best,
        out=np.zeros_like(inverse),
        where=row_best > 0.0,
    )
    col_competition = np.divide(
        inverse,
        col_best,
        out=np.zeros_like(inverse),
        where=col_best > 0.0,
    )

    row_rank = np.zeros((n, n), dtype=float)
    col_rank = np.zeros((n, n), dtype=float)

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size == 0:
            continue
        order = indices[np.argsort(safe[i, indices], kind="mergesort")]
        denominator = max(order.size - 1, 1)
        row_rank[i, order] = 1.0 - np.arange(order.size) / denominator

    for j in range(n):
        indices = np.flatnonzero(valid[:, j])
        if indices.size == 0:
            continue
        order = indices[np.argsort(safe[indices, j], kind="mergesort")]
        denominator = max(order.size - 1, 1)
        col_rank[order, j] = 1.0 - np.arange(order.size) / denominator

    reciprocal = (row_rank == 1.0) & (col_rank == 1.0) & valid
    mutual_rank = np.sqrt(row_rank * col_rank)

    attraction = inverse / max(float(np.max(inverse)), eps)
    potential_scale = max(float(np.percentile(safe[valid], 75.0)), scale, eps)
    potential = np.exp(-safe / potential_scale)
    potential[~valid] = 0.0

    outgoing_reserve = np.zeros(n, dtype=float)
    incoming_reserve = np.zeros(n, dtype=float)

    for i in range(n):
        outgoing = attraction[i, valid[i]]
        incoming = attraction[valid[:, i], i]

        if outgoing.size:
            take = min(5, outgoing.size)
            outgoing_reserve[i] = np.mean(
                np.partition(outgoing, -take)[-take:]
            )

        if incoming.size:
            take = min(5, incoming.size)
            incoming_reserve[i] = np.mean(
                np.partition(incoming, -take)[-take:]
            )

    reserve_scale = max(
        float(np.max(outgoing_reserve)),
        float(np.max(incoming_reserve)),
        eps,
    )
    future_bridge = np.sqrt(
        incoming_reserve[:, None] * outgoing_reserve[None, :]
    ) / reserve_scale
    future_bridge[~valid] = 0.0

    priors = (
        0.44 * attraction
        + 0.13 * row_rank
        + 0.12 * col_rank
        + 0.09 * row_competition
        + 0.08 * col_competition
        + 0.05 * global_quality
        + 0.05 * future_bridge
        + 0.04 * mutual_rank
    )
    priors *= 1.0 + 0.15 * reciprocal

    row_mass = np.sum(priors, axis=1, keepdims=True)
    col_mass = np.sum(priors, axis=0, keepdims=True)
    balanced = priors / np.sqrt((row_mass + eps) * (col_mass + eps))

    opportunity = 1.0 / np.sqrt((row_mass + eps) * (col_mass + eps))
    if np.max(opportunity) > 0.0:
        opportunity /= np.max(opportunity)

    priors = 0.82 * priors + 0.12 * balanced + 0.06 * opportunity
    priors[~valid] = 0.0

    candidate_count = min(n - 1, max(3, int(np.ceil(np.sqrt(n)))))
    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size <= candidate_count:
            continue

        order = candidates[
            np.argsort(-priors[i, candidates], kind="mergesort")
        ]
        gate = np.full(n, 0.10, dtype=float)
        gate[order[:candidate_count]] = 1.0

        rare = opportunity[0, :] > 0.5
        gate[rare & valid[i]] = np.maximum(gate[rare & valid[i]], 0.24)
        priors[i] *= gate

    priors[~valid] = 0.0
    maximum = float(np.max(priors))

    if not np.isfinite(maximum) or maximum <= 0.0:
        return result

    result = np.nan_to_num(priors / maximum, nan=0.0, posinf=0.0, neginf=0.0)
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)
    return result
