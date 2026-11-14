import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a 2D array")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be square")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if not np.all(np.isfinite(distance_matrix)):
        raise ValueError("distance_matrix must contain only finite values")

    d = np.maximum(distance_matrix, 0.0).copy()
    np.fill_diagonal(d, np.inf)

    if n == 1:
        return np.zeros((1, 1), dtype=float)

    scale_index = min(1, n - 2)

    row_sorted = np.sort(d, axis=1)
    row_scale = row_sorted[:, scale_index]
    col_sorted = np.sort(d, axis=0)
    col_scale = col_sorted[scale_index, :]

    eps = np.finfo(float).eps
    row_scale = np.maximum(row_scale, eps)
    col_scale = np.maximum(col_scale, eps)

    row_affinity = np.exp(-d / row_scale[:, None])
    col_affinity = np.exp(-d / col_scale[None, :])

    row_rank = np.argsort(np.argsort(d, axis=1), axis=1)
    col_rank = np.argsort(np.argsort(d, axis=0), axis=0)

    row_rank_score = 1.0 / (1.0 + row_rank.astype(float))
    col_rank_score = 1.0 / (1.0 + col_rank.astype(float))

    mutual_bonus = np.sqrt(row_affinity * col_affinity)
    scarcity_bonus = np.sqrt(row_rank_score * col_rank_score)

    global_scale = np.mean(
        np.minimum(row_scale[:, None], col_scale[None, :])
    )
    global_scale = max(float(global_scale), eps)

    score = (
        0.40 * mutual_bonus
        + 0.32 * scarcity_bonus
        + 0.18 * np.sqrt(row_affinity * col_affinity)
        + 0.10 * np.exp(-d / global_scale)
    )

    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0:
        score /= maximum

    return score
