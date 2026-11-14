import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return robust, balanced, softly sparse TSP edge priors."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    off_diagonal = ~np.eye(n, dtype=bool)
    valid = np.isfinite(distances) & off_diagonal

    if not np.any(valid):
        return result

    values = distances[valid]
    positive_values = values[values > 0.0]

    eps = np.finfo(float).eps

    if positive_values.size:
        scale = max(float(np.median(positive_values)), eps)
        lower = max(
            float(np.percentile(positive_values, 5.0)),
            scale * 1e-6,
            eps,
        )
        upper = max(
            float(np.percentile(positive_values, 95.0)),
            lower,
        )
    else:
        scale = max(float(np.median(np.abs(values))), eps)
        lower = eps
        upper = max(scale, lower)

    safe = np.full((n, n), upper, dtype=float)
    safe[valid] = np.maximum(distances[valid], lower)
    safe = np.clip(safe, lower, upper)
    safe[~off_diagonal] = np.inf

    normalized = safe / max(scale, eps)

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (normalized[valid] + 1e-12)

    row_rank = np.zeros((n, n), dtype=float)
    col_rank = np.zeros((n, n), dtype=float)

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size == 0:
            continue

        order = indices[
            np.argsort(safe[i, indices], kind="mergesort")
        ]

        if order.size == 1:
            row_rank[i, order[0]] = 1.0
        else:
            row_rank[i, order] = 1.0 - (
                np.arange(order.size, dtype=float)
                / float(order.size - 1)
            )

    for j in range(n):
        indices = np.flatnonzero(valid[:, j])
        if indices.size == 0:
            continue

        order = indices[
            np.argsort(safe[indices, j], kind="mergesort")
        ]

        if order.size == 1:
            col_rank[order[0], j] = 1.0
        else:
            col_rank[order, j] = 1.0 - (
                np.arange(order.size, dtype=float)
                / float(order.size - 1)
            )

    row_best = np.max(
        np.where(valid, inverse, 0.0),
        axis=1,
        keepdims=True,
    )
    col_best = np.max(
        np.where(valid, inverse, 0.0),
        axis=0,
        keepdims=True,
    )

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

    mutual_quality = np.sqrt(row_rank * col_rank)
    reciprocal = (
        (row_rank >= 1.0 - 1e-12)
        & (col_rank >= 1.0 - 1e-12)
        & valid
    ).astype(float)

    outgoing_reserve = np.zeros(n, dtype=float)
    incoming_reserve = np.zeros(n, dtype=float)

    for i in range(n):
        outgoing = inverse[i, valid[i]]
        incoming = inverse[valid[:, i], i]

        if outgoing.size:
            take = min(4, outgoing.size)
            outgoing_reserve[i] = float(
                np.mean(np.sort(outgoing)[-take:])
            )

        if incoming.size:
            take = min(4, incoming.size)
            incoming_reserve[i] = float(
                np.mean(np.sort(incoming)[-take:])
            )

    reserve_scale = max(
        float(np.max(outgoing_reserve)),
        float(np.max(incoming_reserve)),
        eps,
    )

    long_horizon = (
        np.outer(incoming_reserve, outgoing_reserve)
        / reserve_scale
    )
    long_horizon[~valid] = 0.0

    global_scale = max(
        float(np.percentile(safe[valid], 75.0)),
        scale,
        eps,
    )
    global_potential = np.zeros((n, n), dtype=float)
    global_potential[valid] = np.exp(
        -safe[valid] / global_scale
    )

    priors = (
        0.44 * inverse
        + 0.14 * row_rank
        + 0.14 * col_rank
        + 0.10 * row_competitiveness
        + 0.08 * col_competitiveness
        + 0.06 * mutual_quality
        + 0.04 * long_horizon
    )

    priors *= 1.0 + 0.16 * reciprocal
    priors[~valid] = 0.0

    row_mass = np.sum(priors, axis=1, keepdims=True)
    col_mass = np.sum(priors, axis=0, keepdims=True)

    balanced = priors / np.sqrt(
        (row_mass + eps) * (col_mass + eps)
    )
    balanced[~valid] = 0.0

    balanced_max = float(np.max(balanced))
    if balanced_max > 0.0:
        balanced /= balanced_max

    priors = 0.78 * priors + 0.22 * balanced
    priors *= 0.96 + 0.04 * global_potential
    priors[~valid] = 0.0

    candidate_count = min(
        n - 1,
        max(3, int(np.ceil(np.sqrt(n)))),
    )

    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size <= candidate_count:
            continue

        order = candidates[
            np.argsort(
                -priors[i, candidates],
                kind="mergesort",
            )
        ]

        gate = np.full(n, 0.10, dtype=float)
        gate[order[:candidate_count]] = 1.0
        priors[i] *= gate

    priors[~valid] = 0.0
    priors = np.nan_to_num(
        priors,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(priors))
    if maximum > 0.0:
        result = priors / maximum

    np.fill_diagonal(result, 0.0)
    result[~valid] = 0.0
    return result
