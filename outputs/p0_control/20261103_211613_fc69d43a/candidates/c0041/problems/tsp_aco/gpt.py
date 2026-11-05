import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a normalized edge-score matrix for a fixed TSP solver."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros_like(distances, dtype=float)

    if n <= 1:
        return scores

    if np.any(np.isnan(distances)):
        raise ValueError("distance_matrix must not contain NaN values")

    work = distances.copy()
    work[work < 0.0] = 0.0
    np.fill_diagonal(work, np.inf)

    finite = np.isfinite(work)
    if not np.any(finite):
        return scores

    finite_values = work[finite]
    fallback = float(np.max(finite_values))
    if not np.isfinite(fallback):
        return scores

    work = np.where(finite, work, fallback)
    eps = np.finfo(float).eps * max(fallback, 1.0)

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))

    nearest = np.partition(work, kth=k - 1, axis=1)[:, :k]
    row_scale = np.mean(nearest, axis=1)
    row_scale = np.maximum(row_scale, eps)

    col_scale = np.mean(
        np.partition(work, kth=k - 1, axis=0)[:k, :],
        axis=0,
    )
    col_scale = np.maximum(col_scale, eps)

    pair_scale = np.sqrt(row_scale[:, None] * col_scale[None, :])
    relative_length = work / np.maximum(pair_scale, eps)
    closeness = np.exp(-relative_length)

    row_reference = np.mean(nearest, axis=1, keepdims=True)
    col_reference = np.mean(
        np.partition(work, kth=k - 1, axis=0)[:k, :],
        axis=0,
        keepdims=True,
    )

    row_advantage = row_reference / np.maximum(work, eps)
    col_advantage = col_reference / np.maximum(work, eps)

    competition = (
        2.0 * row_advantage * col_advantage
        / np.maximum(row_advantage + col_advantage, eps)
    )
    competition /= np.maximum(
        np.max(competition, axis=1, keepdims=True),
        eps,
    )

    neighborhood = np.exp(-work / row_scale[:, None])
    np.fill_diagonal(neighborhood, 0.0)

    overlap = neighborhood @ neighborhood.T
    row_norm = np.sqrt(np.sum(neighborhood * neighborhood, axis=1))
    overlap = overlap / np.maximum(
        row_norm[:, None] * row_norm[None, :],
        eps,
    )
    overlap = np.clip(overlap, 0.0, 1.0)

    scores = (
        closeness
        * (0.5 + 0.5 * competition)
        * (0.5 + 0.5 * overlap)
    )

    np.fill_diagonal(scores, 0.0)
    scores[~np.isfinite(scores)] = 0.0

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores
