import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, scale-invariant edge-priority indicators for TSP."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    finite = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(finite, False)

    valid = finite & finite.T
    if not np.any(valid):
        return scores

    symmetric_distances = 0.5 * (distances + distances.T)
    symmetric_distances[~valid] = np.inf
    np.fill_diagonal(symmetric_distances, np.inf)

    eps = np.finfo(float).eps
    finite_values = symmetric_distances[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.max(finite_values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    finite_max = float(np.max(finite_values))
    invalid_value = max(finite_max, global_scale) * 4.0 + global_scale

    work = np.where(valid, symmetric_distances, invalid_value)
    np.fill_diagonal(work, invalid_value)

    order = np.argsort(work, axis=1, kind="stable")
    sorted_distances = np.take_along_axis(work, order, axis=1)

    nearest = sorted_distances[:, 0]
    second_nearest = (
        sorted_distances[:, 1] if n > 2 else sorted_distances[:, 0]
    )

    local_scale = 0.5 * (nearest + second_nearest)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    relative_distance = np.divide(
        work,
        np.sqrt(local_scale[:, None] * local_scale[None, :]),
        out=np.full((n, n), np.inf, dtype=float),
        where=valid,
    )

    distance_affinity = np.exp(
        -np.clip(1.15 * relative_distance, 0.0, 60.0)
    )

    row_affinity = np.divide(
        local_scale[:, None],
        local_scale[:, None] + work,
        out=np.zeros((n, n), dtype=float),
        where=valid,
    )
    column_affinity = row_affinity.T
    endpoint_affinity = np.sqrt(row_affinity * column_affinity)

    regret = np.divide(
        second_nearest - nearest,
        np.maximum(second_nearest, eps),
    )
    regret = np.clip(regret, 0.0, 1.0)

    regret_bonus = 1.0 + 0.24 * (
        regret[:, None] + regret[None, :]
    )

    nearest_indices = order[:, 0]
    indices = np.arange(n)
    mutual_nearest = (
        (nearest_indices[:, None] == indices[None, :])
        & (nearest_indices[None, :] == indices[:, None])
    )
    mutual_bonus = np.where(mutual_nearest, 1.18, 1.0)

    scores = (
        distance_affinity
        * (0.72 + 0.28 * endpoint_affinity)
        * regret_bonus
        * mutual_bonus
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
