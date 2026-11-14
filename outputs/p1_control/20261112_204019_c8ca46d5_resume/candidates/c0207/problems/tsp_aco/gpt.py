import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally balanced edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)

    finite_values = distances[np.isfinite(distances)]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.percentile(positive_values, 90.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = float(np.max(positive_values))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.nan_to_num(
        distances,
        nan=fallback,
        posinf=fallback,
        neginf=0.0,
    )
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    d = np.where(np.isfinite(d), d, fallback)
    np.fill_diagonal(d, 0.0)

    off_diagonal = d[~diagonal]
    positive_off_diagonal = off_diagonal[off_diagonal > 0.0]

    if positive_off_diagonal.size == 0:
        return np.zeros((n, n), dtype=float)

    center = float(np.median(positive_off_diagonal))
    deviations = np.abs(positive_off_diagonal - center)
    mad = float(np.median(deviations))

    eps = np.finfo(float).eps
    robust_scale = 1.4826 * mad
    if not np.isfinite(robust_scale) or robust_scale <= eps:
        robust_scale = center
    if not np.isfinite(robust_scale) or robust_scale <= eps:
        robust_scale = fallback
    robust_scale = max(robust_scale, eps)

    normalized = d / robust_scale
    normalized = np.clip(normalized, 0.0, 60.0)

    kernel = np.exp(-0.5 * normalized * normalized)
    kernel[diagonal] = 0.0

    if not np.any(kernel > 0.0):
        return np.zeros((n, n), dtype=float)

    balanced = kernel.copy()
    iterations = max(8, min(40, int(np.ceil(4.0 * np.log2(n + 1.0)))))

    for _ in range(iterations):
        row_sums = balanced.sum(axis=1)
        row_sums = np.maximum(row_sums, eps)
        balanced /= row_sums[:, None]

        column_sums = balanced.sum(axis=0)
        column_sums = np.maximum(column_sums, eps)
        balanced /= column_sums[None, :]

        balanced[diagonal] = 0.0

    balanced = 0.5 * (balanced + balanced.T)

    kernel_max = float(np.max(kernel))
    balanced_max = float(np.max(balanced))

    if not np.isfinite(kernel_max) or kernel_max <= 0.0:
        return np.zeros((n, n), dtype=float)
    if not np.isfinite(balanced_max) or balanced_max <= 0.0:
        return np.zeros((n, n), dtype=float)

    kernel_unit = kernel / kernel_max
    balanced_unit = balanced / balanced_max

    score = np.sqrt(
        np.maximum(kernel_unit, 0.0)
        * np.maximum(balanced_unit, 0.0)
    )

    score = 0.5 * (score + score.T)
    score[diagonal] = 0.0
    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    score[diagonal] = 0.0
    return score.astype(float, copy=False)
