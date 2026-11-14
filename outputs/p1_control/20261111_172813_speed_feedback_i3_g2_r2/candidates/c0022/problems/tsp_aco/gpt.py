import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Return edge-priority indicators for stochastic TSP solution sampling.
    Larger values indicate more promising directed edges.
    """
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros_like(d, dtype=float)

    finite = np.isfinite(d)
    positive = d[finite & (d > 0)]

    if positive.size == 0:
        return np.zeros_like(d, dtype=float)

    scale = float(np.median(positive))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    safe_d = np.maximum(np.where(finite, d, np.inf), 0.0)
    valid = finite & ~np.eye(n, dtype=bool)
    safe_d[~valid] = np.inf

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        inverse = np.divide(
            1.0,
            safe_d + 1e-12,
            out=np.zeros_like(safe_d, dtype=float),
            where=valid,
        )

    inverse_max = np.max(inverse)
    if np.isfinite(inverse_max) and inverse_max > 0:
        inverse /= inverse_max

    # Smooth distance preference reduces the dominance of extremely short edges.
    with np.errstate(over="ignore", invalid="ignore"):
        smooth = np.exp(-np.minimum(safe_d / (scale + 1e-12), 50.0))
    smooth[~valid] = 0.0

    nearest = np.argmin(safe_d, axis=1)
    nearest_valid = np.isfinite(safe_d[np.arange(n), nearest])

    nearest_edges = np.zeros((n, n), dtype=float)
    rows = np.arange(n)[nearest_valid]
    cols = nearest[nearest_valid]
    nearest_edges[rows, cols] = 1.0

    mutual = nearest_edges * nearest_edges.T

    # The second-smallest available edge provides a local alternative cost.
    second = np.partition(safe_d, kth=1, axis=1)[:, 1]
    alternative = second[:, None] + second[None, :]

    with np.errstate(invalid="ignore"):
        savings = np.maximum(alternative - safe_d, 0.0)
    savings[~valid] = 0.0

    savings_max = np.max(savings)
    if np.isfinite(savings_max) and savings_max > 0:
        savings /= savings_max

    reciprocal = 0.5 * (inverse + inverse.T)
    smooth_reciprocal = 0.5 * (smooth + smooth.T)

    score = (
        0.34 * inverse
        + 0.22 * smooth_reciprocal
        + 0.20 * reciprocal
        + 0.14 * savings
        + 0.07 * mutual
        + 0.03 * valid.astype(float)
    )
    score[~valid] = 0.0

    # Normalize within each source node to preserve local exploration.
    row_max = np.max(score, axis=1, keepdims=True)
    score = np.divide(
        score,
        row_max,
        out=np.zeros_like(score),
        where=row_max > 0,
    )

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
