import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty_like(distance_matrix, dtype=float)
    if n == 1:
        return np.zeros_like(distance_matrix, dtype=float)

    d = distance_matrix.copy()
    finite = np.isfinite(d)

    if not np.any(finite):
        return np.zeros_like(d, dtype=float)

    positive = d[finite & (d > 0)]
    fallback = float(np.median(positive)) if positive.size else 1.0
    d[~finite] = fallback
    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, np.inf)

    row_scale = np.median(
        np.where(np.isfinite(d), d, np.nan),
        axis=1,
    )
    row_scale = np.where(
        np.isfinite(row_scale) & (row_scale > 0),
        row_scale,
        fallback,
    )

    col_scale = np.median(
        np.where(np.isfinite(d), d, np.nan),
        axis=0,
    )
    col_scale = np.where(
        np.isfinite(col_scale) & (col_scale > 0),
        col_scale,
        fallback,
    )

    rank = np.argsort(np.argsort(d, axis=1), axis=1).astype(float)
    rank_score = 1.0 - rank / max(n - 1, 1)

    reciprocal = 1.0 / np.maximum(d, np.finfo(float).eps)
    reciprocal /= np.nanmax(reciprocal, axis=1, keepdims=True)

    local_score = np.exp(-d / np.maximum(row_scale[:, None], np.finfo(float).eps))
    reverse_local_score = np.exp(-d / np.maximum(col_scale[None, :], np.finfo(float).eps))

    mutual_rank = np.sqrt(np.maximum(rank_score, 0.0) * np.maximum(rank_score.T, 0.0))
    mutual_local = np.sqrt(local_score * reverse_local_score)

    heuristic = (
        0.35 * reciprocal
        + 0.25 * mutual_rank
        + 0.25 * mutual_local
        + 0.15 * rank_score
    )

    heuristic[~finite] = 0.0
    np.fill_diagonal(heuristic, 0.0)

    row_max = np.max(heuristic, axis=1, keepdims=True)
    heuristic = np.divide(
        heuristic,
        row_max,
        out=np.zeros_like(heuristic),
        where=row_max > 0,
    )

    return heuristic.astype(float)
