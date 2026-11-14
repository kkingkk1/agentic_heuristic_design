import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge desirability priors for stochastic TSP solution sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(distances)
    if not np.all(finite):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(distances < 0):
        raise ValueError("distance_matrix must contain non-negative distances")

    mask = ~np.eye(n, dtype=bool)
    positive = distances[mask]
    positive = positive[positive > 0]

    if positive.size == 0:
        result = np.ones((n, n), dtype=float)
        np.fill_diagonal(result, 0.0)
        return result

    scale = np.median(positive)
    if not np.isfinite(scale) or scale <= 0:
        scale = np.mean(positive)
    scale = max(float(scale), np.finfo(float).eps)

    safe_distances = np.maximum(distances, np.finfo(float).eps)
    np.fill_diagonal(safe_distances, np.inf)

    # Robust global distance desirability.
    distance_score = np.exp(-safe_distances / scale)
    distance_score[~mask] = 0.0

    # Row-wise and column-wise competitive ranks.
    row_order = np.argsort(safe_distances, axis=1, kind="stable")
    row_rank = np.empty_like(row_order, dtype=float)
    row_rank[np.arange(n)[:, None], row_order] = np.arange(n)[None, :]
    row_rank = np.maximum(row_rank, 1.0)
    row_rank[~mask] = np.inf

    col_order = np.argsort(safe_distances, axis=0, kind="stable")
    col_rank = np.empty_like(col_order, dtype=float)
    col_rank[col_order, np.arange(n)[None, :]] = np.arange(n)[:, None]
    col_rank = np.maximum(col_rank, 1.0)
    col_rank[~mask] = np.inf

    rank_score = 1.0 / np.sqrt(row_rank * col_rank)
    rank_score[~mask] = 0.0

    # Mutual preference: an edge is favored when both endpoints rank it highly.
    row_best = np.min(safe_distances, axis=1)
    col_best = np.min(safe_distances, axis=0)
    relative_score = np.sqrt(
        (row_best[:, None] / safe_distances)
        * (col_best[None, :] / safe_distances)
    )
    relative_score = np.clip(relative_score, 0.0, 1.0)
    relative_score[~mask] = 0.0

    # Difficult-to-connect nodes receive a mild bonus for their best edges.
    row_mean = np.mean(np.where(mask, safe_distances, np.nan), axis=1)
    col_mean = np.mean(np.where(mask, safe_distances, np.nan), axis=0)
    row_mean = np.nan_to_num(row_mean, nan=scale, posinf=scale)
    col_mean = np.nan_to_num(col_mean, nan=scale, posinf=scale)
    endpoint_bonus = np.sqrt(
        (row_mean[:, None] / safe_distances)
        * (col_mean[None, :] / safe_distances)
    )
    endpoint_bonus = np.clip(endpoint_bonus, 0.0, 2.0)
    endpoint_bonus[~mask] = 0.0

    score = (
        0.40 * distance_score
        + 0.30 * rank_score
        + 0.20 * relative_score
        + 0.10 * endpoint_bonus
    )

    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
