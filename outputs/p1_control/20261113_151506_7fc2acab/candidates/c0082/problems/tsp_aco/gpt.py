import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_matrix must contain only finite values")

    eps = np.finfo(float).eps
    d = np.maximum(distances, 0.0).copy()
    np.fill_diagonal(d, np.inf)

    positive = d[np.isfinite(d) & (d > 0.0)]
    base_scale = float(np.median(positive)) if positive.size else 1.0
    base_scale = max(base_scale, eps)

    nearest_index = np.argmin(d, axis=1)
    nearest_distance = d[np.arange(n), nearest_index]
    nearest_distance = np.where(
        np.isfinite(nearest_distance) & (nearest_distance > 0.0),
        nearest_distance,
        base_scale,
    )
    nearest_distance = np.maximum(nearest_distance, eps)

    kth = min(2, n - 2)
    second_local_scale = np.partition(d, kth, axis=1)[:, kth]
    second_local_scale = np.where(
        np.isfinite(second_local_scale) & (second_local_scale > 0.0),
        second_local_scale,
        base_scale,
    )
    second_local_scale = np.maximum(second_local_scale, eps)

    nearest_pair_scale = np.sqrt(
        nearest_distance[:, None] * nearest_distance[None, :]
    )
    second_pair_scale = np.sqrt(
        second_local_scale[:, None] * second_local_scale[None, :]
    )

    nearest_closeness = np.exp(-d / nearest_pair_scale)
    second_closeness = np.exp(-d / second_pair_scale)

    nearest_closeness[~np.isfinite(nearest_closeness)] = 0.0
    second_closeness[~np.isfinite(second_closeness)] = 0.0

    mutual_nearest_closeness = np.sqrt(
        nearest_closeness * nearest_closeness.T
    )
    mutual_second_closeness = np.sqrt(
        second_closeness * second_closeness.T
    )

    nearest_choice = np.zeros((n, n), dtype=float)
    nearest_choice[np.arange(n), nearest_index] = 1.0
    mutual_nearest = nearest_choice * nearest_choice.T

    quality_score = (
        0.48 * mutual_nearest
        + 0.32 * mutual_nearest_closeness
        + 0.20 * mutual_second_closeness
    )

    speed_score = (
        0.65 * mutual_second_closeness
        + 0.35 * second_closeness
    )

    score = 0.60 * quality_score + 0.40 * speed_score
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    row_max = np.max(score, axis=1)
    pair_max = np.sqrt(row_max[:, None] * row_max[None, :])

    score = np.divide(
        score,
        pair_max,
        out=np.zeros_like(score),
        where=pair_max > 0.0,
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    global_max = float(np.max(score))
    if global_max > 0.0 and np.isfinite(global_max):
        score /= global_max
    else:
        score.fill(0.0)

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
