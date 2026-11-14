import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    directed_valid = np.isfinite(distances) & (distances > 0)
    np.fill_diagonal(directed_valid, False)

    valid = directed_valid & directed_valid.T
    symmetric_distances = 0.5 * (distances + distances.T)
    safe_distances = np.where(valid, symmetric_distances, np.inf)

    inverse = np.divide(
        1.0,
        safe_distances,
        out=np.zeros_like(symmetric_distances, dtype=float),
        where=valid,
    )

    row_max = np.max(inverse, axis=1, keepdims=True)
    nearest_score = np.divide(
        inverse,
        np.where(row_max > 0, row_max, 1.0),
        out=np.zeros_like(inverse),
    )

    nearest_indices = np.argmin(safe_distances, axis=1)
    rows = np.arange(n)
    mutual = np.zeros_like(inverse)
    mutual_edges = (
        valid[rows, nearest_indices]
        & (nearest_indices[nearest_indices] == rows)
    )
    mutual[rows[mutual_edges], nearest_indices[mutual_edges]] = 1.0

    local_counts = valid.sum(axis=1)
    local_means = np.divide(
        np.where(valid, symmetric_distances, 0.0).sum(axis=1),
        np.maximum(local_counts, 1),
    )

    mean_sum = local_means[:, None] + local_means[None, :]
    accessibility = np.divide(
        2.0 * local_means[:, None] * local_means[None, :],
        mean_sum * safe_distances,
        out=np.zeros_like(inverse),
        where=valid & (mean_sum > 0),
    )

    accessibility_max = np.max(accessibility, axis=1, keepdims=True)
    accessibility = np.divide(
        accessibility,
        np.where(accessibility_max > 0, accessibility_max, 1.0),
        out=np.zeros_like(accessibility),
    )

    score = (
        0.60 * nearest_score
        + 0.25 * mutual
        + 0.15 * accessibility
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    return score
