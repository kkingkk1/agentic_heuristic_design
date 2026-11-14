import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally calibrated, diversity-aware edge priorities for TSP."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    valid = np.isfinite(distances) & (distances >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return scores

    values = distances[valid]
    positive = values[values > 0.0]
    eps = np.finfo(float).eps

    if positive.size:
        scale = float(np.median(positive))
    else:
        scale = float(np.median(values)) if values.size else 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    scale = max(scale, eps)

    log_values = np.log1p(values / scale)
    center = float(np.median(log_values))
    mad = float(np.median(np.abs(log_values - center)))
    spread = max(1.4826 * mad, 0.25)

    q_low, q_high = np.quantile(log_values, [0.20, 0.80])
    width = max(float(q_high - q_low), spread, eps)

    transformed = np.zeros_like(distances, dtype=float)
    transformed[valid] = np.log1p(distances[valid] / scale)

    robust_z = (transformed - center) / spread
    robust_z = np.clip(robust_z, -8.0, 8.0)

    quantile_position = (transformed - q_low) / width
    quantile_position = np.clip(quantile_position, -2.0, 3.0)

    base = np.exp(
        -0.72 * robust_z
        -0.38 * np.maximum(quantile_position, 0.0)
    )
    base = np.clip(base, 0.0, 1.0e6)
    base[~valid] = 0.0

    row_mass = base.sum(axis=1, keepdims=True)
    column_mass = base.sum(axis=0, keepdims=True)

    row_mass = np.maximum(row_mass, eps)
    column_mass = np.maximum(column_mass, eps)

    column_accessibility = np.sqrt(column_mass / max(float(n), 1.0))
    destination_correction = 1.0 / (
        0.70 + np.minimum(column_accessibility, 6.0)
    )

    scores = base * destination_correction
    scores[~valid] = 0.0

    for _ in range(4):
        row_sum = scores.sum(axis=1, keepdims=True)
        row_sum = np.maximum(row_sum, eps)
        scores *= np.sqrt(1.0 / row_sum)

        col_sum = scores.sum(axis=0, keepdims=True)
        col_sum = np.maximum(col_sum, eps)
        scores *= np.sqrt(1.0 / col_sum)

        scores[~valid] = 0.0

    scores *= np.sqrt(np.maximum(base, 0.0))
    scores[~valid] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
