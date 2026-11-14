import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return compatibility- and scarcity-aware priors for directed TSP edges."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    safe = np.where(valid, distances, np.inf)

    valid_values = distances[valid]
    positive_values = valid_values[valid_values > 0.0]
    scale = (
        float(np.median(positive_values))
        if positive_values.size
        else max(float(np.median(valid_values)), 1.0)
    )
    scale = max(scale, eps)

    row_count = valid.sum(axis=1).astype(float)
    col_count = valid.sum(axis=0).astype(float)

    row_best = np.min(safe, axis=1)
    col_best = np.min(safe, axis=0)

    row_mean = np.divide(
        np.where(np.isfinite(safe), safe, 0.0).sum(axis=1),
        row_count,
        out=np.full(n, scale, dtype=float),
        where=row_count > 0.0,
    )
    col_mean = np.divide(
        np.where(np.isfinite(safe), safe, 0.0).sum(axis=0),
        col_count,
        out=np.full(n, scale, dtype=float),
        where=col_count > 0.0,
    )

    row_spread = np.maximum(row_mean - row_best, scale * 1e-6)
    col_spread = np.maximum(col_mean - col_best, scale * 1e-6)

    row_difficulty = np.clip(row_best / np.maximum(row_mean, eps), 0.0, 1.0)
    col_difficulty = np.clip(col_best / np.maximum(col_mean, eps), 0.0, 1.0)

    source_scarcity = 1.0 + 0.85 * row_difficulty
    target_scarcity = 1.0 + 0.85 * col_difficulty

    row_relative = np.divide(
        distances - row_best[:, None],
        row_spread[:, None],
        out=np.zeros_like(distances),
        where=valid,
    )
    col_relative = np.divide(
        distances - col_best[None, :],
        col_spread[None, :],
        out=np.zeros_like(distances),
        where=valid,
    )

    local_quality = 1.0 / (
        1.0 + 0.55 * np.maximum(row_relative, 0.0)
        + 0.45 * np.maximum(col_relative, 0.0)
    )

    reverse_distance = distances.T
    reverse_valid = valid.T
    reverse_scale = np.where(reverse_valid, reverse_distance, np.inf)
    reverse_best = np.min(reverse_scale, axis=1)
    reverse_gap = np.divide(
        reverse_distance - reverse_best[:, None],
        np.maximum(
            np.mean(
                np.where(reverse_valid, reverse_distance, 0.0),
                axis=1,
            ),
            scale * 1e-6,
        )[:, None],
        out=np.zeros_like(distances),
        where=valid,
    )

    reciprocity = 1.0 / (1.0 + 0.35 * np.maximum(reverse_gap, 0.0))

    demand = np.divide(
        local_quality.sum(axis=0),
        np.maximum(col_count, 1.0),
    )
    congestion = 1.0 / (1.0 + 0.75 * demand[None, :])

    score = (
        local_quality
        * source_scarcity[:, None]
        * target_scarcity[None, :]
        * reciprocity
        * congestion
    )

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    positive = score[valid & (score > 0.0)]
    if positive.size:
        floor = max(float(np.percentile(positive, 20)) * 0.025, eps)
        score[valid] = np.maximum(score[valid], floor)

    mean_score = float(np.mean(score[valid]))
    if mean_score > 0.0 and np.isfinite(mean_score):
        score /= mean_score

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
