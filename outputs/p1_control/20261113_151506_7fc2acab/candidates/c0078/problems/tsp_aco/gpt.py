import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-promise matrix for a TSP distance matrix."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)
    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_matrix must contain only finite values")

    eps = np.finfo(float).eps

    d = np.maximum(distances, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    positive = d[np.isfinite(d) & (d > 0.0)]
    base_scale = float(np.median(positive)) if positive.size else 1.0
    base_scale = max(base_scale, eps)

    kth = min(2, n - 2)
    local_scale = np.partition(d, kth, axis=1)[:, kth]
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        base_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    pair_scale = np.maximum(pair_scale, eps)

    closeness = np.exp(-np.minimum(d / pair_scale, 30.0))
    closeness[~np.isfinite(closeness)] = 0.0

    mutual_closeness = np.sqrt(closeness * closeness.T)
    directional_average = 0.5 * (closeness + closeness.T)

    top_k = min(2, n - 1)
    nearest_idx = np.argpartition(d, top_k - 1, axis=1)[:, :top_k]
    nearest_choice = np.zeros((n, n), dtype=float)
    nearest_choice[
        np.arange(n)[:, None],
        nearest_idx,
    ] = 1.0

    reciprocal_near = nearest_choice * nearest_choice.T
    reciprocal_near = np.minimum(reciprocal_near, 1.0)

    finite_rows = np.where(np.isfinite(d), d, np.nan)
    row_median = np.nanmedian(finite_rows, axis=1)
    row_median = np.where(
        np.isfinite(row_median) & (row_median > 0.0),
        row_median,
        base_scale,
    )
    row_pair_scale = np.sqrt(row_median[:, None] * row_median[None, :])
    row_pair_scale = np.maximum(row_pair_scale, eps)

    relative_distance = d / row_pair_scale
    outlier_penalty = 1.0 / (
        1.0 + np.maximum(relative_distance - 1.0, 0.0)
    )
    outlier_penalty[~np.isfinite(outlier_penalty)] = 0.0

    score = (
        0.62 * mutual_closeness
        + 0.20 * directional_average
        + 0.13 * reciprocal_near
        + 0.05 * outlier_penalty
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum
    else:
        score.fill(0.0)

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
