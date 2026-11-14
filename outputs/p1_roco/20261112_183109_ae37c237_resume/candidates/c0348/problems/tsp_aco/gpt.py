import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return scale-invariant symmetric edge-priority indicators for TSP."""
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

    symmetric_distance = 0.5 * (distances + distances.T)
    symmetric_distance[~valid] = np.inf
    np.fill_diagonal(symmetric_distance, np.inf)

    valid_values = symmetric_distance[valid]
    positive_values = valid_values[valid_values > 0.0]
    eps = np.finfo(float).eps

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    elif valid_values.size:
        global_scale = float(np.max(valid_values))
    else:
        global_scale = 1.0

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    maximum_distance = float(np.max(valid_values))
    if not np.isfinite(maximum_distance):
        maximum_distance = global_scale

    sentinel = maximum_distance + 16.0 * global_scale + eps
    work = np.where(valid, symmetric_distance, sentinel)
    np.fill_diagonal(work, sentinel)

    k = min(4, n - 1)
    nearest = np.partition(work, kth=k - 1, axis=1)[:, :k]
    nearest_valid = nearest < sentinel

    counts = np.sum(nearest_valid, axis=1)
    sums = np.sum(np.where(nearest_valid, nearest, 0.0), axis=1)

    local_scale = np.divide(
        sums,
        np.maximum(counts, 1),
        out=np.full(n, global_scale, dtype=float),
        where=counts > 0,
    )

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    best = nearest[:, 0]
    second = nearest[:, 1] if k >= 2 else best.copy()

    best_safe = np.where(
        np.isfinite(best) & (best < sentinel),
        best,
        local_scale,
    )
    second_safe = np.where(
        np.isfinite(second) & (second < sentinel),
        second,
        best_safe,
    )

    regret = np.divide(
        second_safe - best_safe,
        np.maximum(second_safe, eps),
        out=np.zeros(n, dtype=float),
        where=np.isfinite(second_safe),
    )
    regret = np.clip(regret, 0.0, 1.0)

    nearest_neighbor = np.argmin(work, axis=1)
    indices = np.arange(n)
    mutual_nearest = (
        (nearest_neighbor[:, None] == indices[None, :])
        & (nearest_neighbor[None, :] == indices[:, None])
        & valid
    )

    row_scale = local_scale[:, None]
    col_scale = local_scale[None, :]
    geometric_scale = np.sqrt(row_scale * col_scale)

    normalized_distance = np.divide(
        symmetric_distance,
        geometric_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=valid & (geometric_scale > 0.0),
    )

    row_ratio = np.divide(
        symmetric_distance,
        row_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=valid & (row_scale > 0.0),
    )
    col_ratio = np.divide(
        symmetric_distance,
        col_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=valid & (col_scale > 0.0),
    )

    geometric_ratio = np.sqrt(
        np.maximum(row_ratio, 0.0) * np.maximum(col_ratio, 0.0)
    )

    competition = np.sum(
        valid & (symmetric_distance <= second_safe[:, None]),
        axis=1,
        dtype=float,
    )

    scarcity = 1.0 / np.sqrt(
        np.maximum(competition[:, None], 1.0)
        * np.maximum(competition[None, :], 1.0)
    )

    log_score = np.zeros((n, n), dtype=float)

    log_score += -1.25 * np.minimum(normalized_distance, 60.0)
    log_score += -0.30 * np.minimum(geometric_ratio, 60.0)
    log_score += 0.16 * np.log1p(4.0 * scarcity)
    log_score += 0.10 * (regret[:, None] + regret[None, :])
    log_score += np.where(mutual_nearest, np.log(1.22), 0.0)

    log_score[~valid] = -np.inf
    log_score = 0.5 * (log_score + log_score.T)

    finite_log = np.isfinite(log_score)
    if np.any(finite_log):
        maximum_log = float(np.max(log_score[finite_log]))
        scores = np.zeros((n, n), dtype=float)
        scores[finite_log] = np.exp(
            np.clip(log_score[finite_log] - maximum_log, -60.0, 0.0)
        )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(scores))
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
