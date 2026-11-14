import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-priority indicators for stochastic TSP solution sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros_like(distances, dtype=float)

    if n <= 1:
        return scores

    finite = np.isfinite(distances)
    positive = np.where(finite & (distances > 0), distances, np.nan)
    scale = np.nanmedian(positive)

    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    d = np.where(finite, np.maximum(distances, 0.0), np.inf)
    np.fill_diagonal(d, np.inf)

    row_sorted = np.sort(d, axis=1)
    col_sorted = np.sort(d, axis=0)

    row_best = row_sorted[:, 0][:, None]
    col_best = col_sorted[0, :][None, :]

    row_second = row_sorted[:, 1][:, None] if n > 2 else row_best + scale
    col_second = col_sorted[1, :][None, :] if n > 2 else col_best + scale

    row_regret = np.maximum(row_second - d, 0.0)
    col_regret = np.maximum(col_second - d, 0.0)

    row_appeal = np.exp(-np.maximum(d - row_best, 0.0) / scale)
    col_appeal = np.exp(-np.maximum(d - col_best, 0.0) / scale)

    regret = 1.0 + 0.5 * (
        np.tanh(row_regret / scale) + np.tanh(col_regret / scale)
    )

    scores = np.sqrt(row_appeal * col_appeal) * regret
    scores[~finite] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if np.isfinite(maximum) and maximum > 0:
        scores /= maximum

    return scores.astype(float, copy=False)
