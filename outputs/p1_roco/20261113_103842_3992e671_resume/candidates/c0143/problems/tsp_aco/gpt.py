import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    finite = np.isfinite(distance_matrix)
    valid = finite & (distance_matrix >= 0.0)
    np.fill_diagonal(valid, False)

    scores = np.zeros((n, n), dtype=float)
    if not np.any(valid):
        return scores

    distances = np.where(valid, distance_matrix, np.inf)

    finite_values = distances[valid]
    positive_values = finite_values[finite_values > 0.0]
    if positive_values.size:
        scale = np.percentile(positive_values, 55.0)
    else:
        scale = np.percentile(finite_values, 55.0)
    scale = max(float(scale), np.finfo(float).eps)

    nearest = np.min(distances, axis=1)
    usable_nearest = nearest[np.isfinite(nearest)]
    fallback = (
        float(np.median(usable_nearest))
        if usable_nearest.size
        else scale
    )
    nearest = np.where(np.isfinite(nearest), nearest, fallback)

    safe_distances = np.where(valid, distances, scale * 8.0)

    global_desirability = 1.0 / (
        1.0 + (safe_distances / scale) ** 2.2
    )

    row_rank = np.full((n, n), n, dtype=float)
    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size:
            order = indices[np.argsort(distances[i, indices], kind="mergesort")]
            row_rank[i, order] = np.arange(1, order.size + 1)

    rank_signal = np.where(
        valid,
        1.0 / np.sqrt(np.maximum(row_rank, 1.0)),
        0.0,
    )

    local_scale = np.sqrt(np.outer(nearest, nearest))
    locality = local_scale / (safe_distances + 0.75 * scale)
    locality = np.clip(locality, 0.0, 1.0)

    nearest_index = np.argmin(distances, axis=1)
    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest_index[i]
        if valid[i, j] and valid[j, i] and nearest_index[j] == i:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    two_step = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            if not (valid[i, j] or valid[j, i]):
                continue

            candidates = np.flatnonzero(
                np.isfinite(distances[i]) & np.isfinite(distances[j])
            )
            candidates = candidates[(candidates != i) & (candidates != j)]

            if candidates.size:
                completion = np.minimum(
                    distances[i, candidates] + distances[candidates, j],
                    distances[j, candidates] + distances[candidates, i],
                )
                best_completion = np.min(completion)
                signal = 1.0 / (
                    1.0 + (best_completion / (2.0 * scale)) ** 1.5
                )
                two_step[i, j] = signal
                two_step[j, i] = signal

    base = (
        0.62 * global_desirability
        + 0.18 * rank_signal
        + 0.12 * locality
        + 0.08 * two_step
    )

    base *= 1.0 + 0.22 * mutual
    base *= np.where(valid, 1.0, 0.0)

    base[~np.isfinite(base)] = 0.0
    np.fill_diagonal(base, 0.0)

    maximum = np.max(base)
    if maximum > 0.0:
        base /= maximum

    return base
