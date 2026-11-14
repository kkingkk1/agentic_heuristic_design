import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-selection priors for stochastic TSP solution sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.zeros_like(distances, dtype=float)

    valid = np.isfinite(distances) & (distances >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    safe_distances = np.where(valid, distances, np.nan)

    positive = safe_distances[(safe_distances > 0.0) & np.isfinite(safe_distances)]
    if positive.size:
        scale = float(np.median(positive))
    else:
        scale = 1.0

    eps = np.finfo(float).eps * max(scale, 1.0)
    clipped = np.maximum(np.nan_to_num(safe_distances, nan=np.inf), eps)

    inverse = np.where(valid, 1.0 / clipped, 0.0)

    row_best = np.max(inverse, axis=1, keepdims=True)
    col_best = np.max(inverse, axis=0, keepdims=True)

    row_relative = np.divide(
        inverse,
        row_best,
        out=np.zeros_like(inverse),
        where=row_best > 0.0,
    )
    col_relative = np.divide(
        inverse,
        col_best,
        out=np.zeros_like(inverse),
        where=col_best > 0.0,
    )

    mutual_preference = np.sqrt(row_relative * col_relative)

    row_scale = np.nanmedian(
        np.where(valid & (clipped > 0.0), clipped, np.nan),
        axis=1,
    )
    row_scale = np.nan_to_num(row_scale, nan=scale, posinf=scale, neginf=scale)

    endpoint_balance = np.minimum.outer(row_scale, row_scale) / np.maximum.outer(
        row_scale, row_scale
    )

    score = (
        0.55 * np.sqrt(row_relative * col_relative)
        + 0.30 * mutual_preference
        + 0.15 * endpoint_balance
    )

    score = np.where(valid, score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0:
        score /= maximum

    return score
