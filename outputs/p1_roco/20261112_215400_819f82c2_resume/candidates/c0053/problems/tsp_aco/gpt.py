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

    eps = np.finfo(float).eps

    row_sorted = np.sort(d, axis=1)
    col_sorted = np.sort(d, axis=0)

    row_best = np.maximum(row_sorted[:, 0], eps)
    col_best = np.maximum(col_sorted[0, :], eps)

    row_second = np.maximum(
        row_sorted[:, min(1, n - 1)],
        row_best + eps,
    )
    col_second = np.maximum(
        col_sorted[min(1, n - 1), :],
        col_best + eps,
    )

    row_scale = np.maximum(row_sorted[:, min(2, n - 1)], eps)
    col_scale = np.maximum(col_sorted[min(2, n - 1), :], eps)

    row_quality = np.exp(-d / row_scale[:, None])
    col_quality = np.exp(-d / col_scale[None, :])
    local_quality = np.sqrt(row_quality * col_quality)

    row_rank = np.argsort(np.argsort(d, axis=1), axis=1).astype(float)
    col_rank = np.argsort(np.argsort(d, axis=0), axis=0).astype(float)

    rank_quality = 1.0 / np.sqrt((1.0 + row_rank) * (1.0 + col_rank))

    row_regret = np.clip(
        (row_second[:, None] - d) / (row_second[:, None] + eps),
        0.0,
        1.0,
    )
    col_regret = np.clip(
        (col_second[None, :] - d) / (col_second[None, :] + eps),
        0.0,
        1.0,
    )
    regret = np.sqrt(row_regret * col_regret)

    row_choice = np.exp(-d / row_best[:, None])
    col_choice = np.exp(-d / col_best[None, :])
    mutual_choice = np.sqrt(row_choice * col_choice)

    global_scale = np.median(
        np.concatenate((row_sorted[:, : min(3, n)], col_sorted[: min(3, n), :].ravel()))
    )
    global_scale = max(float(global_scale), eps)
    global_quality = np.exp(-d / global_scale)

    score = (
        0.42 * local_quality
        + 0.23 * regret
        + 0.20 * mutual_choice
        + 0.10 * rank_quality
        + 0.05 * global_quality
    )

    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return np.clip(score, 0.0, 1.0)
