import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric, distance-dominant edge-priority matrix."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    diagonal = np.eye(n, dtype=bool)
    finite_valid = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    forward = np.where(finite_valid, distances, np.inf)
    backward = forward.T

    forward_ok = np.isfinite(forward)
    backward_ok = np.isfinite(backward)
    both_ok = forward_ok & backward_ok

    distance = np.full((n, n), np.inf, dtype=float)
    distance[both_ok] = 0.5 * (
        forward[both_ok] + backward[both_ok]
    )

    only_forward = forward_ok & ~backward_ok
    only_backward = backward_ok & ~forward_ok
    distance[only_forward] = forward[only_forward]
    distance[only_backward] = backward[only_backward]

    np.fill_diagonal(distance, np.inf)
    valid = np.isfinite(distance)

    if not np.any(valid):
        return scores

    eps = np.finfo(float).eps
    finite_values = distance[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    order = np.argsort(distance, axis=1, kind="stable")
    row_rank = np.zeros((n, n), dtype=float)
    nearest = np.zeros((n, n), dtype=bool)
    local_scale = np.full(n, global_scale, dtype=float)
    second_distance = np.full(n, global_scale, dtype=float)

    neighborhood_size = min(
        n - 1,
        max(3, int(np.ceil(1.25 * np.sqrt(n)))),
    )
    local_top = np.zeros((n, n), dtype=bool)

    for i in range(n):
        candidates = order[i][valid[i, order[i]]]
        if candidates.size == 0:
            continue

        values = distance[i, candidates]
        count = candidates.size

        row_rank[i, candidates] = 1.0 - (
            np.arange(count, dtype=float) / max(count - 1, 1)
        )
        nearest[i, candidates[0]] = True
        local_top[i, candidates[:neighborhood_size]] = True

        if count > 1:
            second_distance[i] = values[1]
            scale_values = values[: min(6, count)]
        else:
            second_distance[i] = values[0]
            scale_values = values

        positive_local = scale_values[scale_values > 0.0]
        if positive_local.size:
            local_scale[i] = float(np.median(positive_local))

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, 0.05 * global_scale)
    local_scale = np.maximum(local_scale, eps)

    global_rank = np.sqrt(row_rank * row_rank.T)

    local_opportunity = np.divide(
        np.sqrt(second_distance[:, None] * second_distance[None, :]),
        distance + 0.35 * global_scale,
        out=np.zeros((n, n), dtype=float),
        where=valid,
    )
    local_opportunity = np.clip(
        (local_opportunity - 0.55) / 1.45,
        0.0,
        1.0,
    )

    bounded_opportunity = (
        0.60 * global_rank
        + 0.40 * local_opportunity
    )

    mutual_nearest = (
        nearest
        & nearest.T
        & valid
    ).astype(float)

    mutual_top = (
        local_top
        & local_top.T
        & valid
    ).astype(float)

    normalized_distance = np.divide(
        distance,
        global_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=valid,
    )
    normalized_distance = np.clip(normalized_distance, 0.0, 40.0)

    log_scores = (
        -1.55 * normalized_distance
        + 0.24 * bounded_opportunity
        + 0.12 * mutual_top
        + 0.10 * mutual_nearest
    )

    scores[valid] = np.exp(
        np.clip(log_scores[valid], -60.0, 20.0)
    )
    scores[~valid] = 0.0

    scores = 0.5 * (scores + scores.T)
    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
