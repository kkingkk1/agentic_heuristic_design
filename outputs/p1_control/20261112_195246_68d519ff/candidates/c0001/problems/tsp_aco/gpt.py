import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-prior indicators for stochastic TSP solution sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    work = np.where(valid, distances, np.nan)
    row_median = np.nanmedian(work, axis=1)
    row_median = np.where(
        np.isfinite(row_median) & (row_median > 0.0),
        row_median,
        1.0,
    )

    centered = distances / row_median[:, None]
    centered = np.where(valid, centered, np.inf)

    row_valid_count = np.maximum(valid.sum(axis=1), 1)
    row_mean = np.where(
        valid,
        distances,
        0.0,
    ).sum(axis=1) / row_valid_count

    row_sq = np.where(valid, (distances - row_mean[:, None]) ** 2, 0.0)
    row_std = np.sqrt(row_sq.sum(axis=1) / row_valid_count)
    row_scale = np.maximum(row_std, 0.05 * row_median)

    margin = (row_median[:, None] - distances) / (row_scale[:, None] + 1e-12)
    local_attractiveness = 1.0 / (1.0 + np.exp(-np.clip(margin, -30.0, 30.0)))
    local_attractiveness = np.where(valid, local_attractiveness, 0.0)

    relative_distance = np.exp(-np.clip(centered, 0.0, 30.0))
    relative_distance = np.where(valid, relative_distance, 0.0)

    row_score = local_attractiveness * relative_distance
    col_score = row_score.T

    score = np.sqrt(np.maximum(row_score * col_score, 0.0))

    finite_scores = score[valid]
    if finite_scores.size:
        low = np.percentile(finite_scores, 5.0)
        high = np.percentile(finite_scores, 95.0)
        if high > low:
            score = np.clip((score - low) / (high - low), 0.0, 1.0)
        else:
            maximum = finite_scores.max()
            if maximum > 0.0:
                score = score / maximum

    score = 0.5 * (score + score.T)
    score = np.where(valid | valid.T, score, 0.0)
    np.fill_diagonal(score, 0.0)

    return score.astype(float, copy=False)
